# Validation log

Every number here states the machine, model, quant, commit and command, per
BANACH-SPEC section 7 (non-negotiable 4).

## Phase 1: planner predicted vs measured

Machine: cloud VM, Intel Xeon @ 2.80 GHz, 4 threads, AVX-512, 16.9 GB RAM.
`banach probe`: 23.4 GB/s streaming read, 574 GFLOP/s f32 FMA peak, 0.93 GB/s disk
(O_DIRECT, random 8 MiB).
Measured: `llama-bench -t 4 -p 0 -n 64 -r 3`, llama.cpp 9309bf1.
Model: Granite 3.1 1B-A400M instruct (QuantFactory GGUF), fully resident.

| quant | measured tok/s | predicted tok/s | error | role |
|---|---|---|---|---|
| Q8_0 | 30.72 +- 1.21 | 30.72 | 0.0% | fit (cpu_eff = 0.18) |
| Q2_K | 43.03 +- 0.87 | 53.49 | +24.3% | held out |

The held-out point is inside the Phase 1 bar of +-30%. Two findings went into the model:

- Compute and memory time add rather than overlap on this machine:
  1/30.7 s is 23 ms of memory traffic plus 11 ms of compute. `compute_overlap` defaults to 0
  (additive) and is a calibration parameter; K1 on the target machine settles it there.
- The FLOP probe has to use the CPU's widest FMA (runtime-detected AVX-512 / AVX2), or it
  under-reports peak by about 9x and the CPU term swamps everything else.

Remaining bias: 2-bit K-quants cost more compute per byte than Q8_0 (+24% on Q2_K). A
per-type efficiency table is the next refinement.

## Phase 1: weights-free preflight on DeepSeek V4 Flash

`banach inspect hf:unsloth/DeepSeek-V4-Flash-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf`

5 split files, 155.1 GB, read with 14 HTTP range requests (6.31 MB of headers, 0.96 MB of
read-ahead past them) in 12 s. Reproduces SPEC-DERISK section 1 from the file itself:
284.3B params, 13.8B active, 43 MoE layers x 256 experts, top-6, 1 shared, 13.37 MB MXFP4
experts, about 7.4 GB dense and 3.4 GB routed per token.

Planner on reference machines (`phase1/machines/`, assumed bandwidths, uncalibrated):

| machine | Exact | Balanced (ladder, projected) |
|---|---|---|
| 32 GB DDR5 70 GB/s, Gen4 6 GB/s, no GPU | 4.28 tok/s (3.59 to 4.61), hit 72% | 4.60 (4.24 to 4.94) |
| 64 GB DDR5 + RTX 5080 16 GB, 2x Gen5 | 12.90 (11.81 to 13.94), hit 89% | 16.83 (15.45 to 18.15) |

Both agree with the SPEC-DERISK verdict: 10 tok/s is out of reach on 32 GB CPU-only at any
RAM size (bandwidth-bound), and the 64 GB + 16 GB GPU class is where a certified 10+ tok/s
headline is plausible. These are predictions until K1/K2 run on such machines.

## Hit-rate model

Without a trace the planner uses h(f) = f^a with a = 0.15 + 0.8 k/N (band a x [0.75, 1.35]),
fitted to colibri issue 905 (V4 Flash, 12% cached, 65.3% hit), the SPEC-DERISK estimate
(35% cached, 80 to 88%) and the Phase 0 Granite pilot. On Granite with 41.5% of the pool
cached the curve gives 0.7353; the in-sample static hot set from the 8.2k-token trace
(`python -m banach_phase0 usage`, then `banach plan --usage`) gives 0.7346.

## Phase 0: pilot on Granite

See `phase0/README.md`: tracer validated end to end, cache-policy tables, and Q2_K against
Q8_0 at 61% top-1 disagreement (next-token accuracy 54.6% to 30.6%).
