# banach-engine

Lossless expert streaming on llama.cpp (Phase 2). The model stays memory-mapped and
byte-identical; the OS page cache is the RAM tier, and banach-engine decides which routed
experts live in it.

- **Budget**: a byte budget for routed experts. Dense tensors and the KV cache are left to
  the page cache.
- **Demand fetch**: when a layer's router has chosen its experts (the `ffn_moe_topk` eval
  callback, before the expert mat-muls run), every missing expert is read whole and in
  parallel with `readahead(2)`, then mapped with `MADV_POPULATE_READ`. Without this the
  mat-mul page-faults the expert in small pieces, because llama.cpp advises random access.
- **Residency policy**: LRU eviction with frequency-gated admission (TinyLFU style). A
  missed expert is used for its layer but only stays if its decayed per-token usage beats
  the expert it would evict. Plain LRU is available as `--policy lru`.
- **Eviction** runs on a background thread with `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`.
  Synchronous `MADV_PAGEOUT` cost 27% of decode time in testing.
- **Warm start**: per-expert heat is saved with `--heat-out` and seeds both the preload and
  the admission frequencies with `--heat-in`.
- **Ground truth**: `mincore()` checks real page-cache residency at every access, so hit
  rates are the kernel's, not just the engine's bookkeeping.
- **Byte-exact oracle**: `--exact-check` also runs stock llama.cpp and compares tokens and
  logits bit for bit. `exact_check.sh` sweeps modes, policies and budgets.

`banach run` (the Rust CLI) plans the budget and launches this binary.

## Build

```bash
cmake -S engine -B build-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/llama.cpp
ninja -C build-engine banach-engine
```

Tested against llama.cpp 9309bf1. Linux only for now (`readahead`, `MADV_POPULATE_READ`,
`mincore`); macOS needs a different fetch path.

## Measured (Granite 3.1 1B-A400M Q8_0, 4-thread Xeon, cloud disk at about 1 GB/s)

Expert pool 1.28 GB, budget 520 MB (about 40%), cold start, 1,740-token prompt, 48 greedy
tokens:

| configuration | decode tok/s | hit | disk MB/token |
|---|---|---|---|
| naive (budget kept, llama.cpp page-faults experts in) | 2.58 | 54.7% | 150 |
| parallel fetch, LRU, synchronous PAGEOUT eviction | 3.16 | 54.7% | 144 |
| + frequency-gated admission | 4.08 | 67.4% | 104 |
| + background DONTNEED eviction (default) | 5.54 | 67.4% | 104 |
| stock llama.cpp, everything resident | 18.58 | 100% | 0 |

Every configuration passed `--exact-check` (byte-identical tokens and logits).

The disk in this environment tops out near 1 GB/s, so parallel fetch has little device
bandwidth to unlock here; the gap between naive faulting and whole-expert parallel reads is
expected to be much larger on a Gen4/Gen5 NVMe, which needs deep queues. That still has to
be measured on such a machine (K2).

## Precision ladder (Phase 3)

`patches/0001-banach-precision-ladder.patch` (179 lines, applies to llama.cpp 9309bf1) adds
`llama_set_ladder(ctx, low_model, eps, tau)` and `llama_set_ladder_tau(ctx, tau)`. Per
routed slot the graph computes `gate * eps[layer][expert] * norm(x)` and serves the slot from
the exact weights when that exceeds tau, otherwise from `low_model`. Every expert mat-mul
runs twice over disjoint slot sets (an expert id of -1 skips a slot and yields zeros), so
each slot reads one copy only. tau sits in a tensor, so the controller can move it without a
graph rebuild. With no ladder installed the graph is the stock graph.

```bash
cd llama.cpp && git checkout 9309bf1 && git apply ../engine/patches/0001-banach-precision-ladder.patch
```

Engine options:

- `--ladder LOW.gguf --eps FILE --tau T`: serve with the ladder
- `--fidelity T1,T2,...`: teacher-forced sweep over the prompt; per tau, raw and confident
  top-1 flips against exact, clipped top-20 KL and the share of slots served low
- `--flips-out PREFIX`: per-token flip and exact margin, for `banach certify`
- `--min-margin M`: margin that makes a token "confident" (default 0.1 nats)

Results and the caveats on the metric are in `docs/VALIDATION.md` (Phase 3).

## Not yet

- O_DIRECT/io_uring into an engine-owned buffer instead of the page cache.
- Streaming and the ladder together: the low copy should be the resident tier and exact
  experts should stream only on escalation. Today the ladder runs with both copies mapped.
- Lookahead prefetch for layer l+1 during layer l (opt-in experiment per SPEC-MATH section 3).
- llama.cpp maps the whole file with a full-file WILLNEED prefetch at load; for models
  larger than RAM that load-time read should be disabled (part of the Phase 3 patch).
