# Phase 0 — feasibility tooling

Implements the hardware-independent parts of the Phase 0 protocol in
[`docs/spec/SPEC-DERISK.md` §6](../docs/spec/SPEC-DERISK.md). Nothing here says anything
about DeepSeek V4 Flash until it is fed real routing traces and real K1/K2 measurements;
the synthetic trace generator exists only to exercise the pipeline.

| Step | What | Where |
|---|---|---|
| P0.0 | environment record for every run | `scripts/record_env.sh` |
| K1 | all-resident 2/3-bit decode kill check (<11 tok/s ⇒ 32 GB CPU-only headline dead) | `scripts/k1_resident_bench.sh` |
| K2 | fio grid (4 block sizes × 5 queue depths, io_uring, O_DIRECT) + 10-min sustained run with NVMe temps | `scripts/k2_io_bench.sh` |
| 0.1 | trace format (`.npz`: top-k ids, gates, ‖x‖, ‖f_i(x)‖, session, workload) | `banach_phase0/trace.py` |
| 0.2 | cache simulator: LRU, decayed LFU, static frequency, GDSF + frequency admission, Belady; per-layer / global; warm 60 / score 40 by session; cross-workload transfer | `banach_phase0/cachesim.py` |
| 0.3 | conditional routing entropy, list-decoding Fano floor | `banach_phase0/entropy.py` |
| 0.4 | ladder sweep over τ (`g·ε̂·‖x‖ ≤ τ`), tail-skip rule, Spearman(bound, flip) | `banach_phase0/ladder.py` |
| 0.5 | betting certificate for top-1 disagreement < p0, anytime upper confidence sequence, clipped top-k KL | `banach_phase0/fidelity.py` |
| 0.6 | decode model `max(T_dram, T_cpu) + max(0, T_io − overlap·T_compute) + T_sync`, speculation break-even, fit to measured runs | `banach_phase0/perfmodel.py` |

Not here yet: the llama.cpp `cb_eval` tracer that writes real traces (0.1), and the patched
`build_moe_ffn` precision mask that produces real flips (0.4). Both need the model and a
≥128 GB machine.

## Usage

```bash
pip install -e '.[test]'
pytest -q
```

```bash
python -m banach_phase0 synth --out /tmp/t.npz --tokens 2000 --layers 43 --experts 256 --k 6
python -m banach_phase0 simulate /tmp/t.npz --capacity-gb 20 --bpw 2.06,4.25
python -m banach_phase0 entropy /tmp/t.npz --cache-per-layer 72
python -m banach_phase0 predict --bpw 2.06 --hit 0.9,1.0 --alpha 0.9 --gamma 1 --target 10
python -m banach_phase0 certify flips.txt --p0 0.01 --alpha 0.05
```

`simulate` scales capacity to the trace's pool when the trace is smaller than the model, so
the fraction of experts cached is what the GB × bpw cell implies.

## Model sanity checks (encoded in the tests)

- V4 Flash expert: 25.17M params; 13.37 MB at 4.25 bpw; 147.2 GB pool; 3.45 GB routed/token.
- CPU-only, 70 GB/s DDR5, 100% hit, no speculation: ~7.2 tok/s (spec: ~7.5–8). 10 tok/s is
  unreachable at any hit rate without MTP; with MTP (α = 0.9, γ = 1) it needs ≥ ~72% hit.
- Expert union for 4 tokens of top-6/256 is 23.17 experts (3.86× one token); SPEC-MATH
  quotes 22.9.
- Betting certificate at a true flip rate of 0.4–0.5%: ~2.5k tokens to certify p < 1% at
  α = 0.05, in line with the 2–4k the protocol expects. Type-I error at p = p0 stays ≤ α.

## Notes on the estimators

- The entropy report sums per-expert inclusion entropies conditioned on the previous
  layer's top-1 expert. That is an upper estimate of H(S | side info), so the Fano number
  is an estimate, not a proven floor. Hash layers (0–2 on V4 Flash) are reported with a floor
  of 0 because routing there is a function of the token id.
- The simulator is pure Python (~1 M accesses/s). A full 200k-token V4 Flash trace is ~52 M
  accesses per cell; run cells in parallel or port the inner loop to Rust before the full grid.
