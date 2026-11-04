# Validation log

Every number here states the machine, model, quant, commit and command, per
BANACH-SPEC section 7 (non-negotiable 4).

## Bigger than RAM: Qwen3.6-35B-A3B (22.1 GB) on a 16 GB machine

Machine: 4-thread Xeon @ 2.1 GHz VM, 16.9 GB RAM, no swap, 19.7 GB/s memory, disk 1.27 GB/s
(random 8 MiB O_DIRECT reads, `banach probe`). Model: unsloth Qwen3.6-35B-A3B-UD-Q4_K_M
(22.1 GB, 40 MoE layers x 256 experts, top-8). Commit c3daa34, llama.cpp 9309bf1 + ladder
patch. Every run starts cold (the whole file dropped from the page cache), greedy decode,
`-c 2048 -t 4`, CPU weight repacking off in all modes.

```
banach-engine -m qwen36-35b-a3b-Q4_K_M.gguf -f PROMPT -n N -c 2048 -t 4 --mode MODE [--budget-mb 10000]
```

| Run | Mode | Prefill | Decode | Expert hit | Disk per token |
|---|---|---|---|---|---|
| 13-token prompt, 32 tokens | stock llama.cpp (mmap, no budget) | 9.32 s | 3.54 tok/s | - | - |
| | naive, 10 GB expert budget | 6.41 s | 2.44 tok/s | 83.2% | 129.1 MB |
| | banach, 10 GB expert budget | 3.49 s | 3.98 tok/s | 83.2% | 90.3 MB |
| 18-token prompt, 256 tokens | stock llama.cpp (mmap, no budget) | 7.64 s | 4.89 tok/s | - | - |
| | naive, 10 GB expert budget | 5.27 s | 4.87 tok/s | 95.7% | 29.1 MB |
| | banach, 10 GB expert budget | 3.44 s | 5.40 tok/s | 95.7% | 23.7 MB |

- Byte-exact: `--mode banach --budget-mb 10000 --exact-check -n 64` PASS, 64 tokens with
  logits bit-equal to stock llama.cpp at every step.
- Banach is 10 to 12% faster than stock llama.cpp at decode and 2.2 to 2.7x faster at
  prefill, while holding routed experts to a 10 GB budget; stock llama.cpp is free to use the
  whole page cache.
- Planner (uncalibrated, generic hit curve): Exact 2.11 tok/s (1.87 to 2.35). Measured
  steady-state decode is 5.40 tok/s, outside the range: the generic hit curve predicted 90%
  hits at this budget, the run reached 95.7%, and the run is short enough that disk traffic
  falls off as the hot set fills. Fitting this run with `banach calibrate` is the next step.
- Found on the way: with llama.cpp's default CPU weight repacking (Q4_K to q4_K_8x8 on
  AVX2), the routed experts are copied out of the mapping into anonymous memory; on this
  machine the process was killed for running out of memory at 13.9 GB resident. The engine
  now turns repacking off (`--no-repack` in llama.cpp terms).

## Where decode time goes on the 22 GB model (is prefetch worth building?)

Same run as "Bigger than RAM" above: `--mode banach --budget-mb 10000`, 256 tokens, cold start,
commit c3daa34 (engine JSON `long_q36_banach.json`).

| Part of decode | Seconds | Share of 47.42 s |
|---|---:|---:|
| Waiting on expert fetches (critical path) | 6.68 | 14.1% |
| Fetch plus admission and eviction bookkeeping (callback) | 7.47 | 15.7% |
| Everything else (compute) | 39.95 | 84.3% |

Expert hit rate 95.7%; 23.7 MB read from disk per token. The most that perfect next-layer
prefetch could save is the fetch wait, about 14 to 16% of decode, and published fine-grained
MoE results report recall of 65 to 80% and gains of 0 to 5% once reads are asynchronous. Below
the 20% bar chosen for building a predictor, so no predictor is built; compute, not disk, is
the bottleneck at this size. On a machine where the model is a larger multiple of RAM the
split will shift: re-measure there before revisiting.

## Planner table in the README ("Which model on my machine?")

`banach plan MODEL --hw PROFILE`, commit after c3daa34, Exact plan (lossless weights), default
target 5 tok/s and ctx 8192. Profiles: the 16 GB VM measured by `banach probe` (above),
phase1/machines/ref-32gb-ddr5-gen4.json and phase1/machines/ref-64gb-rtx5080.json. Models are
the first shard of each unsloth GGUF on Hugging Face; only headers are read.

| Model | 16 GB VM | ref 32 GB | ref 64 GB + RTX 5080 |
|---|---|---|---|
| Kimi-K2-Instruct UD-Q2_K_XL (381.9 GB) | 0.20 (0.16 to 0.25) | 1.04 (0.84 to 1.26) | 3.56 (3.00 to 4.12) |
| DeepSeek-V4-Flash UD-Q4_K_XL (155.1 GB) | 0.44 (0.35 to 0.53) | 2.20 (1.82 to 2.60) | 10.02 (8.64 to 11.35) |
| Qwen3-235B-A22B-Instruct-2507 UD-Q2_K_XL (88.8 GB) | 0.33 (0.26 to 0.39) | 1.73 (1.44 to 2.03) | 7.20 (6.57 to 7.78) |
| gpt-oss-120b F16 (65.4 GB) | 0.98 (0.80 to 1.17) | 4.93 (4.18 to 5.66) | 24.15 (22.12 to 26.10) |
| GLM-4.5-Air UD-Q2_K_XL (47.4 GB) | 0.55 (0.46 to 0.64) | 2.77 (2.42 to 3.10) | 14.89 (13.73 to 15.98) |
| Qwen3.6-35B-A3B UD-Q4_K_M (22.1 GB) | 2.11 (1.87 to 2.35); measured 5.40 | 8.76 (7.98 to 9.52) | 80.60 (73.62 to 87.38) |

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

## Phase 4: serving, APIs, downloads

Same machine. `banach serve` on Granite 3.1 1B-A400M.

- API compatibility: the official OpenAI Python SDK (chat completions, streaming and not)
  and the official Anthropic Python SDK (messages, streaming and not) ran against the server
  without changes beyond `base_url`. Stop reasons, usage counts and SSE event order matched
  what the SDKs expect. Requests with tools get a 400 with a clear message.
- Prefix cache: a second request sharing 15 of its 16 prompt tokens prefilled in 0.04 s
  against 2.39 s cold.
- Stop strings are cut before they are emitted and reported (`stop_sequence` / finish
  reason); text is only emitted on complete UTF-8 boundaries.
- Live fidelity loop with the Q4_K_M ladder at tau 7 (gate only): 5 responses, all shadowed
  on the exact model, 368 confident tokens, 2 flips (0.54%), lifetime upper bound 2.3%: far
  too few tokens to certify, as expected. With a deliberately short 200-token control window
  the controller lowered tau from 7 to 5.63; the default window is 5,000 tokens.
- Lazy start, `banach serve hf:...Q3_K_S.gguf --lazy` (619 MB) over this environment's
  link (about 1.5 MB/s): serving after 41 s with 106 MB on disk (header and dense tensors);
  the first answer (correct) completed at 409 s; the full download finished and passed the
  SHA-256 check against the LFS object id at 415 s. The first prompt needed almost the whole
  expert pool: with 8 of 32 experts routed per layer, a 20-token prefill touches nearly every
  expert. Lazy start pays off in proportion to routing sparsity (V4 Flash routes 6 of 256;
  a 20-token prefill touches about 38% of each layer).
- `bench/run_bench.sh` on Granite Q8_0, default prompt, 5 runs each: resident decode 20.34
  tok/s median, 520 MB expert budget 10.41 tok/s median, byte-exact check passed.

## Phase 3: precision ladder and fidelity

Same machine. Granite 3.1 1B-A400M, exact = Q8_0, low copy = Q4_K_M / Q3_K_M / Q2_K of
the same weights (QuantFactory GGUFs), dense path always Q8_0. `banach-engine --fidelity`:
teacher-forced over a 4,096-token evaluation text (three source files and SPEC-MATH), top-1
compared with the exact model at every position. eps profiled on real MoE inputs from a
separate 3,072-token calibration corpus (`banach-trace --dump-x`, `banach_phase0 eps --real`).

Correctness of the patch:

- tau below every bound (all slots exact): 0 flips, KL exactly 0 over 4,096 tokens.
- exact weights used as the low copy: 0 flips and KL 0 at 0%, 33% and 100% of slots on the
  second path. The two-path graph is numerically exact; every flip below is a precision effect.

Confident flips count only tokens where the exact model's top-1 beats its top-2 by at least
0.1 nats (3,899 of 4,096 tokens):

| low copy | rule | tau | slots served low | raw flips | confident flips | KL |
|---|---|---|---|---|---|---|
| Q4_K_M | gate only | 1.4 | 0.1% | 2.17% | 0.72% | 0.0021 |
| Q4_K_M | gate only | 7 | 21.6% | 2.51% | 0.82% | 0.0030 |
| Q4_K_M | gate only | 14 | 65.7% | 5.05% | 2.95% | 0.0098 |
| Q4_K_M | gate x eps | 0.05 | 10.7% | 2.93% | 1.13% | 0.0026 |
| Q4_K_M | gate x eps | 0.1 | 32.8% | 3.54% | 1.64% | 0.0045 |
| Q4_K_M | all low | - | 100% | 7.10% | 4.90% | 0.0217 |
| Q3_K_M | gate x eps | 0.05 | 1.8% | 2.47% | 0.80% | 0.0023 |
| Q3_K_M | gate x eps | 0.1 | 11.3% | 2.86% | 1.10% | 0.0033 |
| Q3_K_M | all low | - | 100% | 13.09% | 11.03% | 0.0760 |
| Q2_K | gate x eps (Gaussian eps) | 0.5 | 18.0% | 4.29% | - | 0.0084 |
| Q2_K | all low | - | 100% | 22.8% | - | 0.285 |

Findings:

1. Raw top-1 agreement cannot be held within 1% of exact on this model by any lossy change.
   About 3% of the exact model's predictions are near-ties (top-1/top-2 margin below 0.05
   nats); 0.03% of slots at 4 bits already flips 2.3% of tokens, because the perturbation
   reaches every later position through the KV cache. Confident flips have a floor of about
   0.7% for the same reason. The fidelity contract therefore reports raw flips, confident
   flips and KL, and certifies confident flips.
2. The gate weight carries most of the signal. With Q4_K_M the gate-only rule beat
   gate x eps: 21.6% of slots low at 0.82% confident flips, against 10.7% at 1.13%. eps from
   Gaussian inputs did no better than gate-only on Q2_K either; eps from real inputs helped
   only at small low shares on Q2_K (about 18% fewer flips at 33% low). The g x eps x norm(x)
   rule of SPEC-MATH section 6 is not supported by this model; gate-only is the default
   candidate until a larger model says otherwise.
3. Uniform 2-bit experts are far outside budget (22.8% raw flips with a Q8_0 dense path,
   61% when the dense path is also 2-bit: keeping the dense path at native precision matters
   more than the experts, as non-negotiable 7 says).
4. Nothing is certified yet. Best setting (Q4_K_M, gate only, tau 7): 0.82% confident flips
   over 3,899 tokens, betting upper bound 1.11% at alpha 0.05. At a true rate near 0.8% the
   protocol expects 14k to 25k tokens to certify below 1%; the certificate needs a longer
   evaluation run, not a looser threshold.

Where the flips happen. Flip rate by the exact model's top-1/top-2 margin (nats), 4,096
tokens (197 / 651 / 624 / 765 / 1,859 tokens per band):

| configuration | margin < 0.1 | 0.1 to 0.5 | 0.5 to 1 | 1 to 2 | > 2 | all |
|---|---|---|---|---|---|---|
| Q4_K_M, gate only, tau 7 (21.6% low) | 36.0% | 4.9% | 0.0% | 0.0% | 0.0% | 2.51% |
| Q4_K_M, gate only, tau 14 (65.7% low) | 46.7% | 14.9% | 2.6% | 0.3% | 0.0% | 5.05% |
| Q4_K_M, every slot low | 50.8% | 22.6% | 6.6% | 0.4% | 0.0% | 7.10% |
| Q3_K_M, every slot low | 53.8% | 38.9% | 18.6% | 6.5% | 0.6% | 13.09% |

At the best setting every flip is on a token where the exact model's top two choices were
within 0.5 nats (a probability ratio below 1.65); no token with a larger margin changed. As
more slots go low precision, flips reach more confident tokens. Near-ties can still be
meaningful branch points in code or reasoning, so task-level evaluations remain to be done.

The Rust meter (`banach certify`) reproduces the Python reference certificate exactly on a
fixed sequence. The PI controller on log tau converges below budget in simulation, but only
with control windows of thousands of tokens: with 500-token windows the bound can never fall
under 1% and tau collapses to its floor.

## Phase 2: streaming engine, measured vs planner

Same machine. `banach-engine` on Granite 3.1 1B-A400M Q8_0 (expert pool 1.28 GB), cold
start, 1,740-token prompt, 48 greedy tokens, policy admit, 16 fetch threads.
`banach plan --ctx 1790` with RAM set to dense + KV + the expert budget, uncalibrated.

| setup | measured tok/s | measured hit | predicted tok/s (range) | predicted hit | error |
|---|---|---|---|---|---|
| fully resident | 16.45 | 100% | 18.21 (16.68 to 19.68) | 100% | +10.6% |
| 800 MB budget | 7.65 | 85.1% | 9.13 (7.17 to 11.26) | 84.7% | +19.5% |
| 520 MB budget | 5.49 | 67.4% | 6.59 (5.06 to 8.38) | 72.9% | +20.2% |
| 300 MB budget | 3.76 | 46.2% | 5.07 (3.89 to 6.50) | 60.1% | +34.9% |

Three of four are inside +-30% uncalibrated; the 300 MB miss comes from the hit curve
(60% predicted, 46% measured), and the measurement sits at the bottom of the predicted range.
Calibrating on the resident run alone (time scale 0.903) puts all four within +-22%.
End to end, `banach run` planned a 600 MB budget and predicted 7.07 tok/s (5.46 to 8.90) on
a second prompt; the engine measured 6.47 (+9%).

Changes made to the model from these runs: NVMe reads no longer overlap compute
(`overlap` 0: demand fetch blocks the layer), and attention cost is the KV read at 0.2 of
memory bandwidth (stock llama.cpp went from 30.7 to 18.6 tok/s between a short context and
1,740 tokens on this machine).

Byte-exact oracle: `engine/exact_check.sh` passed all 12 configurations (naive and banach
modes, LRU and admit policies, budgets of 20%, 50% and 100%), tokens and logits bit-equal to
stock llama.cpp at every step.

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
