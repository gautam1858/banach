# Phase 0 - feasibility tooling

Implements the Phase 0 protocol in
[docs/spec/SPEC-DERISK.md, section 6](../docs/spec/SPEC-DERISK.md). Nothing here says anything
about DeepSeek V4 Flash until it is fed real V4 Flash traces and real K1/K2 measurements.
The pipeline has been run end to end on a small real MoE (Granite 3.1 1B-A400M, 32 experts,
top-8); see "Pilot run" below.

| Step | What | Where |
|---|---|---|
| P0.0 | environment record for every run | `scripts/record_env.sh` |
| K1 | all-resident 2/3-bit decode kill check (below 11 tok/s means the 32 GB CPU-only headline is dead) | `scripts/k1_resident_bench.sh` |
| K2 | fio grid (4 block sizes x 5 queue depths, io_uring, O_DIRECT) plus a 10 min sustained run with NVMe temps | `scripts/k2_io_bench.sh` |
| 0.1 | routing tracer on llama.cpp (no patch, eval callback only) and importer | `../tracer/`, `banach_phase0/trace.py` |
| 0.2 | cache simulator: LRU, decayed LFU, static frequency, GDSF with frequency admission, Belady; per-layer or global; warm 60 / score 40 by session; cross-workload transfer | `banach_phase0/cachesim.py` |
| 0.3 | conditional routing entropy, list-decoding Fano floor | `banach_phase0/entropy.py` |
| 0.4 | ladder sweep over tau (escalate when g * eps * norm(x) > tau), tail-skip rule, Spearman(bound, flip) | `banach_phase0/ladder.py` |
| 0.5 | betting certificate for top-1 disagreement below p0, anytime upper confidence sequence, clipped top-k KL, trace-vs-trace flips | `banach_phase0/fidelity.py` |
| 0.6 | decode model max(T_dram, T_cpu) + max(0, T_io - overlap * T_compute) + T_sync, speculation break-even, fit to measured runs | `banach_phase0/perfmodel.py` |

Still missing: real per-expert mixed precision for 0.4 (needs the engine hook from Phase 2,
so that a single run can serve each expert from a different rung), and every number that
needs V4 Flash on a machine with 128 GB or more.

## Usage

```bash
pip install -e '.[test]'
pytest -q
```

Trace a model (build the tracer first, see `../tracer/README.md`):

```bash
banach-trace -m model.gguf -i corpus.jsonl -o run1 -c 6144 -ub 512
python -m banach_phase0 import-trace run1.json --out run1.npz --hash-layers 0,1,2
```

Analyses:

```bash
python -m banach_phase0 simulate run1.npz --capacity-gb 20 --bpw 2.06,4.25
python -m banach_phase0 entropy run1.npz --cache-per-layer 72
python -m banach_phase0 flips exact.npz lowbit.npz --by-workload
python -m banach_phase0 predict --bpw 2.06 --hit 0.9,1.0 --alpha 0.9 --gamma 1 --target 10
python -m banach_phase0 certify flips.txt --p0 0.01 --alpha 0.05
```

`simulate` scales capacity to the trace's pool when the trace comes from a smaller model
than `--model`, so the fraction of experts cached matches the GB x bpw cell.
`synth` writes a synthetic trace for testing the pipeline without a model.

## Pilot run (Granite 3.1 1B-A400M, pipeline validation only)

8,215 teacher-forced tokens over 5 sessions (2 coding, 2 prose, 1 chat), context 2048,
4 cores, about 70 tok/s of tracing. These numbers validate the tools; they are not V4 Flash
predictions.

Hit rate, per-layer caches, 60/40 session split:

| cache (share of pool) | LRU | LFU (h=256) | static | GDSF | Belady |
|---|---|---|---|---|---|
| 25% | 0.355 | 0.501 | 0.522 | 0.534 | 0.653 |
| 38% | 0.551 | 0.681 | 0.683 | 0.698 | 0.804 |
| 50% | 0.749 | 0.813 | 0.806 | 0.817 | 0.895 |

Q2_K against Q8_0 as the reference: 61% top-1 disagreement, mean clipped top-20 KL 1.03.
The collapse is real, not a tracer bug: next-token accuracy drops from 54.6% (Q8_0) to 30.6%
(Q2_K). Uniform 2-bit is far outside a 1% budget on this model, in line with the specs.

## Model sanity checks (encoded in the tests)

- V4 Flash expert: 25.17M params; 13.37 MB at 4.25 bpw; 147.2 GB pool; 3.45 GB routed per token.
- CPU-only, 70 GB/s DDR5, 100% hit, no speculation: about 7.2 tok/s (spec: about 7.5 to 8).
  10 tok/s is unreachable at any hit rate without MTP; with MTP (alpha 0.9, gamma 1) it needs
  a hit rate of about 72% or more.
- Expert union for 4 tokens of top-6 of 256 is 23.17 experts (3.86 times one token);
  SPEC-MATH quotes 22.9.
- Betting certificate at a true flip rate of 0.4 to 0.5%: about 2.5k tokens to certify
  p < 1% at alpha 0.05, in line with the 2k to 4k the protocol expects. Type-I error at
  p = p0 stays at or below alpha.

## Notes on the estimators

- The entropy report sums per-expert inclusion entropies conditioned on the previous
  layer's top-1 expert. That is an upper estimate of the conditional entropy, so the Fano
  number is an estimate, not a proven floor. Hash layers (0 to 2 on V4 Flash) are reported
  with a floor of 0 because routing there is a function of the token id.
- The certificate tests "the conditional flip rate is at least p0 at every step". When the
  rate changes between sessions an early rejection stays rejected, so the upper bound can sit
  below the overall mean. Certify per workload (`flips --by-workload`).
- The simulator is pure Python (about 1M accesses per second). A full 200k-token V4 Flash
  trace is about 52M accesses per cell; run cells in parallel or use the Rust planner's
  simulator for the full grid.
