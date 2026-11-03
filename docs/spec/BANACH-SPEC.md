# BANACH — Master Specification

> Banach–Tarski for language models: more model than your machine should hold.

This is the master spec. Four companion documents go deeper: `SPEC-DERISK.md` (verified
model numbers, hit-rate and fidelity evidence, Phase 0 protocol, kill checks — READ SECOND),
`SPEC-MATH.md` (bit allocation, caching, speculation, fidelity meter), `SPEC-LADDER.md`
(precision ladder, physics), `SPEC-SYSTEM.md` (architecture, planner, UX). Precedence:
this file > SPEC-DERISK > SPEC-MATH > SPEC-LADDER > SPEC-SYSTEM.

## 0. Status after de-risking (2 Nov 2026)
The original headline — DeepSeek V4 Flash at 10+ tok/s, ≤1% top-1 disagreement, 32 GB DDR5,
one Gen4 NVMe, no GPU — is estimated ~10% likely (SPEC-DERISK section 0). Three walls: the always-
active FP8/BF16 path (~6.27 GiB dense + 1.06 GiB head) streams ~7 GB per token and caps
dual-channel DDR5 near 8–10 tok/s even at 100% hit; only ~18–21 GB is free for experts on a
32 GB box; published 2–3-bit quants of this family show 5–19% top-1 disagreement.
The product is still worth building. The headline target is chosen AFTER Phase 0, from:
(a) certified ≤1% top-1 at ≥10 tok/s on 64 GB + 16 GB GPU (~40–50% likely); (b) same on M5 Max
64 GB; (c) 32 GB CPU-only: "2–4× colibri at certified ≤1%, and an honest planner that told you
so before download." The uncontested differentiators are the certified fidelity meter (no
system has one) and the validated weights-free planner.

---

## 1. One-sentence definition

Banach runs very large Mixture-of-Experts models (284B–2.8T parameters) on ordinary machines
by keeping the exact model on NVMe as a lossless floor, serving most tokens from a learned,
reduced-precision hot set in RAM/VRAM, escalating precision only where an error bound says it
matters, and continuously proving fidelity against the exact model on the user's own traffic.

## 2. What is new (and what is not)

Not new, reuse it: lossless expert streaming from NVMe (colibri, llama.cpp PRs #25294 and
#31469), learned hot sets, router-lookahead prefetch, MTP speculation, OpenAI/Anthropic APIs.

New, build it:
1. A **precision ladder with a lossless floor**: exact weights never leave disk; 2/2.5/3-bit
   copies live in a sidecar store; per-expert escalation follows the bound
   `g_i · ε̂_i(b) · ‖x‖ > τ` (gate weight × quantization error × activation norm).
2. **Closed-form bit allocation**: `b*_e,t = b̄ + ½·log2(p_e · F̄_e,t · σ²_e,t) − mean(...)`.
   Doubling an expert's usage earns +½ bit. Pruning is the b = 0 corner of the same formula.
3. A **fidelity contract**: predicted tok/s and top-1 agreement before download; measured
   top-1 agreement and clipped top-k KL during use, with anytime-valid (betting) confidence
   sequences; a PI controller on τ keeps the upper confidence bound under the user's budget.
4. **Weights-free preflight + lazy expert download**: three HTTP range requests answer
   "will it run, how fast, how good"; dense part + hot set download first (~25–40 GB), cold
   experts fetched on demand in the background.
5. **Honest impossibility reporting**: when physics says no, say so with the exact shortfall
   and offer a sibling model, LAN pooling, or slow mode.

## 3. Physics the whole design rests on

Decode time per token ≈ max over tiers of (bytes moved from that tier ÷ its bandwidth).

| Tier | Bandwidth (typical, late 2026) |
|---|---|
| Gen4 NVMe | ~5.5 GB/s usable (7 rated) |
| Gen5 NVMe | ~11 GB/s usable (14.5 rated) |
| DDR5 dual-channel | ~60–65 GB/s usable |
| Apple M5 Max | 460–614 GB/s |
| Consumer VRAM | 400–1,800 GB/s |

Routed-expert bytes per token, cold, at 4-bit: DeepSeek V4 Flash (284B/13B, top-6 of 256)
≈ 3.4 GB; GLM-5.2 (744B) ≈ 11 GB; Kimi K3 (2.8T) ≈ 26 GB.

Levers that multiply: cache bits (hit rate), bits fetched on miss, effective top-k,
speculation tokens per expert fetch, SSD count. Stacked, ~5–6× fewer NVMe bytes per token.

Walls that no cache fixes: V4 Flash's ~7 GB/token always-active path caps CPU-only DDR5 at
~7.5–8 tok/s without speculation (~12–13 with MTP at 100% hit); GLM-5.2's 9.9 GB dense part caps
it at ~6; Kimi K3 is ~0.05–1 tok/s at best on consumer hardware; dense 405B is unusable below
256 GB+. MTP self-speculation is the only CPU-only lever that amortizes the dense read.

## 4. Targets (projections, to be measured in Phase 0)

| Machine | DeepSeek V4 Flash | GLM-5.2 | Kimi K3 |
|---|---|---|---|
| 32 GB DDR5 + Gen4 NVMe, no GPU | 3–8 tok/s realistic; 10 needs MTP + ≥91% hit (SPEC-DERISK section 4) | 1–2 | slow mode |
| 32 GB + RTX 5080 16 GB, 2× Gen5 | 12–18 | 3–6 | slow mode |
| M5 Max 64 GB | 15–22 | ~4 | slow mode |
| 128 GB unified (M5 Max / Strix Halo) | 25+ | 6–10 | ~1 |

Quality target: ≤1% top-1 disagreement vs exact at the default ("Balanced") setting.
Average bits per weight for that: ~3.0–3.5, with 2-bit only on the cold tail.

Candidate headline (decided after Phase 0): **DeepSeek V4 Flash with a certified ≤1% top-1
fidelity meter at ≥10 tok/s** on the cheapest hardware class where Phase 0 proves it. colibri
measures 0.86–1.6 tok/s on 32 GB CPU-only; kTransformers reports ~28 tok/s with a 5090 + DDR4.

## 5. Architecture

```
banach (Rust daemon, single binary)
├── cli / tui / web dashboard         OpenAI + Anthropic APIs (tools, SSE, prefix cache)
├── probe        hardware microbench (60 s): RAM/NVMe/VRAM bandwidth, measured
├── preflight    GGUF/safetensors header via HTTP range reads → model profile
├── planner      roofline cost model → water-filling bit allocation → greedy MCKP per tier
│                → predicted (tok/s ± CI, fidelity) → plan
├── fetcher      lazy per-expert range downloads (HF Xet), hash-verified, sandboxed parser
├── memory       tier manager (VRAM/RAM/NVMe×n), ladder sidecar store (2/2.5/3-bit + exact),
│                robust predictive GreedyDual cache with LRU ghost fallback,
│                next-layer-gate prefetch, io_uring/O_DIRECT reads ≥1 MB, two I/O priorities
├── fidelity     shadow-exact replay (stratified by gate band), betting confidence sequences
│                on top-1 agreement + clipped top-k KL, PI loop on τ
├── speculation  MTP / prompt-lookup with per-step break-even test on expert-union bytes
└── engine hook  "expert provider" callback into vendored libllama/ggml (+ ik_llama.cpp
                 trellis quant types); returns (ptr, precision) per (layer, expert)
```

Kernels are a commodity: reuse ggml. The planner, memory system, ladder store and fidelity
system are the product. Write a custom kernel only where it is the measured bottleneck.

Code strength matches tier bandwidth: trellis codes (QTIP-style / IQ2_KT, IQ3_KT) for
NVMe-streamed bytes, where there is spare compute per byte; scalar/K-quants for RAM-resident
hot experts, where kernels must keep up with DRAM.

## 6. Governing rules (the math, compressed)

- **Allocation**: minimise Σ ½·p_e·F̄_e,t·‖ΔW‖² subject to RAM budget → reverse water-filling
  (closed form above) → project onto palette {2, 2.25, 2.5, 3, 3.5, 4, exact} with the
  multiple-choice knapsack greedy on convex hulls; bisect λ (RAM) and κ (quality).
- **Residency**: default to recency eviction (LRU/GreedyDual-Size) with model-informed ADMISSION
  and precision; trillion-scale traces show recency ≈ same-domain oracle frequency and far better
  off-domain, and router lookahead bought 0.3% (SPEC-DERISK section 2). Keep the predicted-GreedyDual +
  LRU-ghost-cache scheme (SPEC-MATH section 3) as an opt-in experiment, not the first build.
- **Ladder / skip**: escalate expert i iff `g_i · ε̂_i(b) · ‖x‖ > τ`. Skip a tail set S only
  when `2 · g_S · M < τ` (M = profiled high-percentile expert output norm) and S is cold.
  Never skip top-1. Do NOT hard-code "top-1 always exact": it costs ~0.57 GB/token of forced
  NVMe reads on V4 Flash; let the bound decide.
- **I/O**: bytes in flight = bandwidth × latency (Little). Reads ≥1 MB, registered buffers,
  polling; prefetch utilisation ρ ≲ 0.7; demand misses pre-empt prefetch.
- **Speculation**: speculate iff (1 − α^(γ+1))/(1 − α) > [T_draft + T_fixed +
  Σ missing_bytes(union(γ+1))/BW] / T_1. Expert union for 4 tokens of top-6/256 ≈ 3.8× one
  token. Biasing the drafter toward resident experts is lossless under speculative sampling.
- **Fidelity**: primary metric top-1 agreement (flip rate); secondary clipped top-k KL.
  Betting confidence sequences → ~2–3k shadow tokens for ±1% at 95% when p ≈ 3%.
  Control on the upper confidence bound, never the point estimate.

## 7. Non-negotiables

1. 100% local; the only network use is model download on request. CI test fails on any
   other network call.
2. Never silently change quality. Every lossy decision is bound-governed and metered.
   `--exact` is byte-identical to the reference engine.
3. Exact weights on disk are never rewritten; reduced copies live in the sidecar.
4. Reporting: predictions carry confidence intervals; measurements state hardware, model,
   quant, commit and exact command.
5. Model files are untrusted input: Rust parsing with checked arithmetic, hash-verified
   downloads, no symlink following, no pickle.
6. Licenses: generate ladder/pruned packs locally by default; redistribute only for
   MIT-licensed models (DeepSeek V4 Flash, GLM-5.3-Flash) with the copyright notice. Kimi K3
   and Mistral Large 4 have custom licenses: never redistribute derived packs.
7. The dense (non-expert) path stays at native FP8/BF16; DeepSeek's QAT makes even Q8 on some
   non-expert tensors lossy (Unsloth). Nested 3.25-bit expert slices come free from MXFP4;
   2-bit packs must be separately quantized.

## 8. Phases

**Phase 0 — Feasibility (Python, 2–3 weekends). Hard stop at the end. Full protocol: SPEC-DERISK section 6.**
K1  Kill check: all-resident 2-bit V4 Flash decode on a same-CPU-family ≥128 GB machine; <11 tok/s
    kills the 32 GB CPU-only headline. K2 fio/STREAM constants on the target box.
0.1 Routing tracer on DeepSeek V4 Flash (expert ids, gate weights, output norms per token
    per layer); ≥50k tokens of real coding traffic + a generic corpus.
0.2 Cache simulator: LRU / decayed-frequency / predicted GreedyDual / Belady at 12/20/25/40 GB
    with experts at 2/2.5/3/4 bits → hit rate, miss-bytes/token, bandwidth-model tok/s.
0.3 Conditional routing entropy per layer; Fano/list-decoding miss floor.
0.4 Quality probe: quantize sampled experts, compute ε̂, simulate the ladder at several τ,
    estimate top-1 disagreement with confidence sequences on ≥3k held-out tokens; measure the
    cost of "top-1 always exact".
0.5 `phase0/REPORT.md` with a plain verdict: projected tok/s at ≤1% disagreement on 32 GB,
    and the single biggest risk.

Go/no-go (SPEC-DERISK section 6): GO original headline only if K1 ≥ 11 tok/s AND simulated ≥10 tok/s
with 15% margin on MEASURED free RAM AND the certifying τ's escalation misses fit the I/O budget
on coding and chat AND cross-workload hot-set transfer ≥ 80% of in-domain. PIVOT to a fallback
headline if K1 passes but fidelity/capacity fails. STOP CPU-only if K1 fails. Publish the
negative result either way.

**Phase 1 — Core**: probe, preflight, planner v1, preflight card. Predicted vs measured
within ±30%.
**Phase 2 — Memory + lossless streaming**: engine hook, tier manager, io_uring, prefetch,
predictive cache, byte-exact CI oracle.
**Phase 3 — Ladder + fidelity meter**: sidecar store, bound-driven escalation, shadow replay,
confidence sequences, PI loop, speculation toggle, `--exact`. The headline number, reproducible.
**Phase 4 — Launch**: APIs, lazy download, README, benchmark script, comparison table,
credits to colibri and the llama.cpp PR authors.

MVP platforms: Linux x86-64 + CUDA, macOS Apple Silicon. MVP models: Qwen3.6-35B-A3B (fits
everywhere, the "it just works" model), DeepSeek V4 Flash (the hero), Qwen3.8-27B dense.

## 9. Publishable research if validated

1. A Fano/list-decoding lower bound on expert-cache miss rate from measured conditional
   routing entropy, compared against Belady and real predictors across DeepSeek, GLM, Kimi.
2. Successively refinable expert codes: 2-bit trellis base + residual reaching 4-bit
   rate–distortion quality, refuting "nested is lossy" (SliceMoE's 7.0 → 10.2 PPL).
3. Lossless cache-aware drafting for speculative decoding under expert offload.

## 10. Launch

Killer demo (60 s): stock 32 GB desktop. `ollama run deepseek-v4-flash` fails with "requires
more system memory". `banach run deepseek-v4-flash` shows the preflight card, starts chatting
after ~25% of the weights, holds ~10 tok/s with a live "fidelity 99.x% vs exact" meter.
A coding agent connects to the local API and fixes a real bug. End on `banach plan kimi-k3`, which answers
honestly: "0.4 tok/s; try deepseek-v4-flash or pool your Mac (est. 2.1 tok/s)."

Publish the reproducible table (≥10 machines × 3 models: tok/s, top-1 agreement, KL) with
the repo. Show HN + r/LocalLLaMA titled with the contract ("predicted vs measured").
Credit colibri openly; offer to upstream the engine hook.

## 11. Known risks

Physics (NVMe vs DRAM is a 10–100× gap); quality cliffs at 2-bit on outlier-heavy experts;
hit rates measured so far come from ≤80B models; the engine-hook patch may break on upstream
refactors; monthly architecture churn (inherit llama.cpp's lag); Hugging Face rate limits on
range requests; colibri (~46k stars) and Strata (~38k) can copy features fast. The durable
moat is the published, reproducible fidelity dataset.
