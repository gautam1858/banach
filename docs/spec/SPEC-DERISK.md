# SPEC-DERISK — Verified numbers, evidence, Phase 0 protocol, kill checks
(Condensed from the Phase 0 de-risking report, 2 Nov 2026. Full report: SPEC-DERISK-FULL.md if present.)

## 0. Verdict
Original headline — DeepSeek V4 Flash at 10+ tok/s, ≤1% top-1 disagreement, 32 GB DDR5, one Gen4
NVMe, no GPU — is estimated ~10% likely. Three independent walls:
1. Capacity: always-active FP8/BF16 path ≈ 6.27 GiB dense + 1.06 GiB head (colibri docs). Free for
   experts on 32 GB: ~18–21 GB, not 25 GB.
2. DRAM bandwidth: that path streams ~7 GB per token. Dual-channel DDR5 (65–75 GB/s achieved) caps
   decode at ~7.5–8 tok/s at 100% hit without speculation; ~12–13 with MTP at ~1.8 accepted/forward.
3. Fidelity: published uniform 2–3-bit expert quants of this family: 77–86% top-1 agreement
   (AtomicChat, V4 Flash 0731) to 90.5–95% (Unsloth, V4.1 Flash). ≤1% is 5–20× stricter.
Fallback headlines, in order: (a) certified ≤1% top-1 at ≥10 tok/s on 64 GB + 16 GB GPU
(40–50% likely); (b) same on M5 Max 64 GB; (c) CPU-only 32 GB: "2–4× colibri at certified ≤1%, and
an honest planner that told you so before download."

## 1. DeepSeek V4 Flash 0731 — verified architecture
- 284B total / 13B active; 43 layers, all MoE; first 3 layers hash-routed (token-id → expert, no
  learned router, perfectly predictable); 256 routed experts, top-6, 1 shared per layer.
- hidden 4096, expert intermediate 2048 → 3×4096×2048 = 25.17M params per routed expert;
  11,008 routed experts (≈277B, ~96% of weights). 258 routed + 43 shared touched per token.
- Routing: sqrtsoftplus, noaux_tc, route_scale 1.5, SwiGLU clamp 10.
- Native: experts MXFP4 (E2M1, block-32 UE8M0 scales, 4.25 bpw); shared experts + dense FP8 E4M3
  block-128; some tensors BF16. QAT: Unsloth found Q8_0 on some non-expert tensors is NOT lossless.
  → never requantize the dense path.
- Attention: hybrid SWA(128) + compressed sparse attention; 1 KV head, head_dim 512; KV ≈ 3–4 KB/token
  (estimate) — negligible.
- MTP: one next-token predictor layer. License MIT.
- Nested precision: dropping the E2M1 mantissa bit gives a free, bit-nested ~3.25-bpw slice with the
  same scales. A clean nested 2-bit slice does NOT exist; 2-bit packs must be separately quantized.

Per-expert sizes and pool (derived):
| bpw | bytes/expert | all experts | per token (258) | fits in 12/20/25/40 GB |
| 4.25 MXFP4 | 13.37 MB | 147.2 GB | 3.45 GB | 898/1496/1870/2992 (8–27%) |
| 3.06 IQ3_XXS | 9.63 MB | 106 GB | 2.48 GB | 1246/2077/2596/4154 (11–38%) |
| 2.5 | 7.86 MB | 86.6 GB | 2.03 GB | 1527/2545/3181/5089 (14–46%) |
| 2.06 IQ2_XXS | 6.49 MB | 71.4 GB | 1.67 GB | 1849/3082/3852/6163 (17–56%) |
Cross-checks: colibri ~12.6 MB/expert, ~137 GiB routed; Strata "147 GB experts".

Other models: V4.1 Flash (10 Sep 2026) is a different architecture — 552B backbone + 196B Engram
tables, 384 experts top-6, 40 routed layers, ~2× larger; the DeepSeek API name "deepseek-v4-flash"
now serves V4.1. GLM-5.3-Flash (MIT): 320B/18B, 42 MoE layers × 288 experts top-8, same 25.17M
expert size, native FP8 (likely more low-bit headroom). Kimi K3: 2.78T/104B, 896 experts top-16,
1.56 TB, custom license; ~27.8 GB of expert reads per token — planner showcase only.

## 2. Hit-rate evidence on large MoEs
- Si et al. 2026 (arXiv 2608.12103), trillion-scale: W64 reuse 80–85%; LRU ≈ 75% of Belady;
  kernel page-cache recency 75.3% vs same-domain oracle frequency 74.6%; off-domain oracle collapses
  to 21–34% while LRU holds ~70%; 64.7%-recall router lookahead bought 0.3%. → recency eviction +
  model-informed ADMISSION; don't oversell prefetch.
- colibri issue #905 (V4 Flash, Ryzen 7 6800H, 32 GB): 16.06 GiB cache (~12% of pool) → 65.3% hit,
  0.86 tok/s; ~1.1 GB NVMe reads/token.
- GLM-5.3-Flash patch: 32 GiB cache → 78% hit, ~1.3 GB misses/token.
- SeqMoE (Sep 2026): 45% residency → 96.97% hit with seq2seq prediction (smaller models).
- Liang et al. ICLR 2026: shared experts LOWER routed locality — don't assume Qwen-like numbers.
Expectation (estimate): 35% of pool → 80–88% hit single-domain coding, 75–85% mixed, Belady ~90–93%.
At the realistic 18–21 GB on 32 GB: subtract 3–5 points. Domain switches hurt (top-set Jaccard 0.03–0.09).

## 3. Low-bit fidelity evidence
| source | model | quant (expert bpw) | KLD | top-1 agree |
| AtomicChat, ctx 5632 | V4 Flash 0731 | IQ3_XS 3.20 | 0.207 | 85.4% |
| | | IQ2_S 2.48 | 0.334 | 81.0% |
| | | IQ2_XXS 2.06 | 0.454 | 77.5% |
| | | MXFP4 experts, non-experts quantized | 0.156 | 87.4% |
| Unsloth | V4.1 Flash | UD-IQ2_XXS | 0.20/0.14 | 90.5/91.2% |
| | | UD-IQ3_XXS | 0.063/0.043 | 94.6/95.1% |
| Unsloth | V4 Flash | UD-Q8_K_XL | ~0 | 100% (bit-identical reference) |
Treat AtomicChat as pessimistic, Unsloth V4.1 as optimistic. Task accuracy degrades far less than
top-1 (V4.1 IQ2_XXS keeps 98.8% of benchmark average at ~90.5% top-1). Mixed precision beats uniform
at scale; no per-token hot-high/cold-low test exists on V4 — Phase 0 produces it. Measure at ≥5,632
context (indexer/compressors inert at 512).

## 4. CPU reality
- Anchor: 8-core Xeon slice, V4 Flash IQ3_XXS fully resident: tg 5.61 tok/s, pp 12.3 (floor, not ceiling).
- No measured consumer 16-core dual-channel DDR5 all-resident datapoint exists. K1 produces it.
- Compute: 26 GFLOP/token; 10 tok/s = 260 GFLOP/s ≈ 5% of Zen 5 AVX-512 peak — raw FLOPs are not the
  wall; dequant (no native FP8 on consumer x86, MXFP4 unpack, IQ codebooks) is.
- I/O budget at target with perfect overlap (~75 ms compute/token with MTP, Gen4 ~6 GB/s on 6–13 MB
  reads): ~450 MB/token → ~70 two-bit misses or ~34 MXFP4 misses of 258. Without overlap: ~150 MB →
  needs ≥91% hit.
- AVX2-only CPUs bind on kernels before disk (colibri). Reference box: Zen 4/5 AVX-512.

## 5. Competitive/landscape changes Sep–Nov 2026
colibri v2.0/2.1: Vulkan on every engine, `coli plan`, runs low-bit V4/V4.1 GGUFs; expert-major
prefill. Strata 0.2.x: shipped hot-expert profiles, `--experts N`. llama.cpp: #29887 llama-moe-cache
merged; #25294, #31469 (--moe-direct-io, GPU-centric, byte-exact) open. kTransformers 0.6.x: native
MXFP4 kernels, AVX2/VNNI paths. Ladder prior art: SliceMoE, DyMoE, HOBBIT, PagedWeight — position
g·ε̂·‖x‖ as a refinement. NO system found with an anytime-valid fidelity meter → Banach's clearest moat.
Unsloth vs ISTA-DASLab dispute (30 Oct) over fidelity metrics → certified meter is timely.

## 6. Phase 0 protocol (falsifiable)
P0.0 Record for every run: CPU model/microcode/cores/flags, DIMMs (dmidecode -t memory), kernel +
MGLRU state, NVMe model/firmware, filesystem, governor, ambient temp, llama.cpp commit, model SHA-256s,
script git hash.
K1 (kill check): same-CPU-family ≥128 GB machine, official UD-IQ2_XXS and UD-IQ3_XXS fully resident,
`llama-bench -t <cores> -n 128`. 2-bit tg < 11 tok/s ⇒ 32 GB CPU-only headline dead.
K2: fio randread, io_uring, --direct=1, real 100+ GB file, bs ∈ {6656k,7864k,9856k,13696k} ×
iodepth ∈ {1,4,8,16,32}, 60 s + 5 s ramp, JSON; 10-min sustained with `nvme smart-log` temps every 5 s;
STREAM triad + 7 GB FP8 GEMV. colibri `iobench O_DIRECT=1` as cross-check.
0.1 Traces: mainline llama.cpp cb_eval, teacher-forced prefill with -ub 2048; log ffn_moe_topk-<il>,
ffn_moe_weights-<il> (verify names), ‖x‖ post hyper-connection, per-expert ‖f_i(x)‖ (add one named
tensor), final-logit top-20. ~1.5 KB/token. Workloads (≥50k each, 200k total, split by session 60/40):
V4-generated coding sessions, HumanEval+/LCB-style, chat, zh + European, ≥32k long-context. Discard
synthetic step-0 pass; MTP off for baseline, separate MTP-on trace; tag hash layers.
0.2 Simulator: LRU, decayed-LFU (64–4096), static frequency, GreedyDual-Size + next-layer-gate
admission, global LRU, Belady; 12/20/25/40 GB × 2.06/2.5/3.06/4.25 bpw × per-layer vs global; warm
60%/score 40%; outputs hit, miss-bytes/token, transfer matrix.
0.3 Conditional entropy per layer; Fano/list-decoding floor.
0.4 ε̂_{i,b} median & p95 of ‖f_i(x) − f̂_{i,b}(x)‖/‖x‖; ladder sweep over 12 τ; actual flips via patched
build_moe_ffn with precision mask; validate rank-correlation(bound, flip) ≥ 0.5.
0.5 Betting CS: D_t = 1[top-1 ≠ exact], teacher-forced; reference UD-Q8_K_XL; measure numerical floor
between two exact runs first; H0: p ≥ 0.01; K_t = Π(1 + λ_i(0.01 − D_i)), λ ∈ [0, 1/0.99), certify at
K_t ≥ 1/α, α = 0.05. Expect 2–4k tokens at ≤0.5% true rate, 14–25k at 0.8%.
0.6 Model: T_token = max(T_dram, T_cpu) + max(0, T_io − overlap·T_compute) + T_sync; fit on K1 and a
colibri run on the same box; ≤15% error on colibri required.
Go/no-go: GO original headline only if K1 ≥ 11 tok/s AND simulated ≥10 tok/s with 15% margin on
MEASURED free RAM AND certifying τ's escalation misses fit the I/O budget on coding and chat AND
cross-workload transfer ≥ 80% of in-domain hit. PIVOT if K1 passes but fidelity/capacity fails.
STOP CPU-only if K1 fails.
Pitfalls: page-cache contamination (O_DIRECT + /proc/diskstats deltas; drop caches between cells);
cgroup v2 MemoryMax walls not mlock balloons (balloons overstate device traffic ~2×); thermals and
sustained numbers; cold/first-warm/steady reported separately; no calibration/eval session overlap;
≥5,632 context; corrected chat template; API name serves V4.1.

## 7. Excellence checklist
Benchmark JSON per run: full P0.0 environment; model + SHA-256s; Banach commit; planner prediction vs
measured; prefill/decode tok/s median+IQR over ≥5 runs; cold/warm state; hit rate; NVMe bytes/token;
escalation rate; mean bits; top-1 disagreement with CS upper bound and α; KLD mean/p99; workload id;
token count; NVMe temp; peak RSS.
Leaderboard columns: model × hardware class × engine/config; decode tok/s; NVMe GB/token; certified
top-1 disagreement (α=0.05, tokens); KLD p50/p99; workload suite version; numerical floor;
reproducible artifact hash. Accept colibri/llama.cpp/Strata/kTransformers on the same harness.
Security: SHA-256 vs HF LFS object id; Rust parsers with checked arithmetic + cargo-fuzz; vendor
llama.cpp newer than CVE-2026-33298 (fixed b7824) and CVE-2026-27940; O_NOFOLLOW; signed,
content-addressed pack manifests; daemon unprivileged, network only to HF host.
Credit: colibri (JustVugg, Apache-2.0, keep NOTICE); llama.cpp/ggml (MIT), PRs #25294 (freedomljc),
#31469 (zyxciss), #29887; ik_llama.cpp trellis quants; Unsloth and AtomicChat tables; Si et al.,
Liang et al.; SliceMoE/DyMoE/HOBBIT; Waudby-Smith & Ramdas.
Licensing: V4 Flash and GLM-5.3-Flash MIT (packs redistributable with notice); Kimi K3 custom;
Mistral Large 4 excludes MaaS; GLM-5.3 754B gated.

## 8. What would most impress experts
1. Public V4 Flash routing-trace dataset + simulator with Belady bounds (none exists).
2. Anytime-valid fidelity certificate on user traffic with a measured numerical floor.
3. Weights-free planner validated on ≥10 community machines with ±15% residuals published,
   contributed back as a llama.cpp trace tool + fio harness.
