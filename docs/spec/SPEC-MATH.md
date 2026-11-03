# SPEC-MATH — Mathematical design (condensed; full: SPEC-MATH-FULL.md if present)

Governing idea: one rate–distortion–latency problem. Every expert tensor gets a precision and a tier
from closed-form water-filling weighted by usage and Fisher information; a knapsack turns that into a
placement; every residency, precision, skip and speculation decision is checked against one bound:
gate weight × expert error × activation norm. Bytes per miss matter more than eviction cleverness
(Belady beats LRU by only 1.07–1.9× on real traces; good bit allocation saves ≥0.25 bits/param free).

## 1. Bit allocation (rate–distortion)
Distortion: D_KL ≈ ½ Δθᵀ F Δθ. Per (expert e, tensor t) with usage frequency p_e, mean Fisher F̄_e,t
(gathered on tokens routed to e; already folds in gate² and ‖x‖²), weight variance σ², n params:
  D = Σ ½ · p_e · F̄_e,t · n_e,t · σ²_e,t · c_Q · 2^(−2 b_e,t)   s.t.  Σ n_e,t · b_e,t ≤ R_RAM
Reverse water-filling (KKT):
  b*_e,t = b̄ + ½·log2(p_e · F̄_e,t · σ²_e,t) − ⟨½·log2(p·F̄·σ²)⟩_param-weighted, clipped at b_min
Consequences: doubling usage or Fisher ⇒ +½ bit; doubling RMS ⇒ +1 bit; pruning (REAP saliency) is
the b=0 corner of the same objective; fractional palettes {2, 2.25, 2.5, 3, 3.5, 4, exact} close most
of the allocation gap. Evidence: Orr/Ribar/Luschi ICLR 2026 (≥0.25 bits/param saved); MxMoE ICML 2025;
Q-Strata EMNLP 2026; AlphaQ 2026 (3.5-bit budget ≈ BF16 on Qwen1.5-MoE; 2.5-bit costs 3–7%);
Colla-Q warns frequency-driven allocation overfits calibration domain ⇒ re-estimate p_e on user traffic.
Working target: ~3.0–3.5 average bits/weight for ≤1% top-1 disagreement, 2-bit only on the cold tail.
Algorithm: offline Fisher diagonals + σ² per (expert, tensor); online decayed p_e; closed-form b*;
project onto palette with MCKP greedy (section 4); re-solve when ‖Δp‖ exceeds threshold.

## 2. Weight formats for 2–3-bit experts
Gaussian bound D(R)=2^(−2R); Lloyd–Max scalar is ~1.9× (2.7 dB) above it at 2 bits. Hadamard/rotation
incoherence (QuIP#, QuaRot, SpinQuant, HARP) makes weights ~i.i.d. Gaussian so the bound applies.
Llama-2-7B 2-bit Wikitext PPL: QuIP# 6.19, AQLM 6.64, PV-Tuning 5.84, QTIP 5.86 (QTIP closes >3× of
QuIP#'s gap to optimal). Decode cost: large-codebook lookup (AQLM) is 5× slower at batch 1; computed
trellis codes (QTIP 1MAD/3INST) cost a few integer ops/weight; ik_llama.cpp ships IQ1–IQ4_KT trellis
types with CPU kernels (quantizing 5× slower; CPU decode "room to optimize").
Rule: match code strength to tier bandwidth. NVMe-streamed bytes (3–14 GB/s) have ~10× spare compute
per byte ⇒ strongest code (trellis 2.25–3 bpw) there, cutting V4 Flash cold bytes 3.4 GB → ~1.9–2.5 GB.
RAM-resident hot experts ⇒ Hadamard-rotated IQ-K/K-quants where kernels keep up with DRAM. Measure
both per machine. Open: no published CPU trellis-vs-scalar benchmark at MoE expert shapes — produce one.

## 3. Hot-set caching
Objective: min Σ miss · size/BW_tier (weighted, variable-size paging). Measured ceilings: Qwen3-30B
LRU 66% / Belady 79% at 13.4% budget; FlashMoE Belady 86% vs LRU 73%; trillion-scale Belady 62% vs
LRU 44%; OLMoE Belady only 1.068× ("misses are compulsory"). Learning-augmented guarantees: follow
predictions + LRU fallback is 2·min{1+2η/OPT, …, k}-competitive (Wei 2020).
Predictors: next-layer gate on current input ≈ 90–96% top-1 next layer (HOBBIT), ProMoE 84.7%,
SeqMoE 84.6/91.7/95.1% hit at 15/25/35% capacity. BUT trillion-scale: 64.7%-recall lookahead as kernel
readahead gained 0.3%; static hot sets don't transfer across domains; recency does.
Information bound (derived): list-decoding Fano, P_miss ≥ (H(X|Y) − 1 − log2 L)/log2(N/L); V4 Flash
needs log2 C(256,6) ≈ 38.4 bits/layer removed. Log H(S_{l+1} | S_l, hidden bucket) from traces.
Algorithm "robust predictive GreedyDual": score_i = P̂(used within horizon) × cost_i/size_i; evict
lowest; P̂ = next-layer-gate replay blended with decayed frequency; run an LRU ghost cache and switch
whenever it has fewer recent miss-bytes. Per SPEC-DERISK: default to recency eviction with
model-informed admission; keep the predictor as an opt-in experiment.

## 4. Placement solver (milliseconds)
Per expert choose option j ∈ {NVMe floor, RAM@b, VRAM@b}: bytes s_{e,j}; time t_{e,j} =
p_e(1−h_{e,j}) s_miss/BW_NVMe + p_e s_{e,j}/BW_tier; distortion D_{e,j}. Minimize Σ t + κ Σ D s.t. per-tier
capacity = multiple-choice knapsack. Solver: lower convex hull of each expert's options in (bytes, cost);
sort hull increments by Δcost/Δbytes; fill greedily (LP optimum, ≤1 fractional item); bisect λ (RAM) and
κ (quality). ~46k items for V4 Flash → sub-10 ms in Rust. Online: λ_{t+1} = [λ_t + η(RAM_used −
RAM_budget)]₊, same for κ vs fidelity set-point. Static-hot-set hits are modular ⇒ greedy exact; union
objectives (speculation, batching) are submodular ⇒ 1−1/e.
I/O (Little + M/G/1): bytes in flight = BW × latency (7 GB/s × ~100 µs ≈ 0.7 MB ⇒ QD ≥ 6 at 128 KB, 1–2 at
4 MB, × SSDs); io_uring saturates only at ≥128 KB blocks; reads ≥1 MB, registered buffers, polling;
prefetch utilization ρ ≲ 0.7; two priority classes, demand pre-empts prefetch.
Autotune ≤6 knobs (λ split, τ, prefetch depth, draft cap, threads, read size) with GP-EI, a few dozen
trials, seeded by the roofline.
Note the cost of "top-1 always exact": ~1/6 of 3.4 GB ≈ 0.57 GB/token forced NVMe reads on V4 Flash
(~80 ms at 7 GB/s). Replace with the section 6 bound.

## 5. Speculation under expert streaming
Expected tokens/pass = (1−α^(γ+1))/(1−α). Expert union for n tokens of top-k/N (independent routing):
U(n) = N[1 − (1−k/N)^n]; top-6/256, n=4 ⇒ 22.9 experts ≈ 3.8× one token. Speculate iff
  (1−α^(γ+1))/(1−α) > [T_draft(γ) + T_fixed + Σ_tiers missing_bytes(U(γ+1))/BW_tier] / T_1.
α=0.7, γ=3 ⇒ 2.53 tokens: loses if the union is cold, wins if ≥90% resident. Best draft length is a
monotone staircase in α (Lan 2026, ±1 at 22 acceptance values); precompute per machine. SpecFit-style
subtree selection by draft-probability/expert-union-cost ratio (Dinkelbach). Biasing the drafter toward
resident experts is LOSSLESS under speculative sampling (unlike cache-aware routing, which costs 1–10%
PPL) — no prior paper found; publishable. V4 Flash has one MTP layer: kTransformers 90% acceptance at
depth 1; colibri saw expert inflation +39% on GLM; measure on V4.

## 6. Skip / ladder error bound (derived)
y = Σ_{i∈K} g_i E_i(x), Σ g_i = 1. Dropping set S and renormalizing: ‖y − y'‖ ≤ 2·g_S·M, M = profiled
high-percentile expert output norm. Serving expert i from a b-bit copy: error ≤ g_i·‖ΔW_{i,b}‖_op·‖x‖.
Ladder rule: escalate i to the next rung iff g_i · ε̂_i(b) · ‖x‖ > τ (ε̂ measured per expert at quantization
time; τ is the single controller knob). Skip only cold tail experts with small 2·g_S·M; never top-1.
Evidence: HOBBIT 10% skip ⇒ >1% PPL; ACE keeps top-1 always; MoBE shared bases cut params 24–30% for
1–2% loss (collapses at 50%); TEAL 40–50% activation sparsity gives 1.5–1.8× on the dense part (GPU).

## 7. Fidelity measurement
Metric: top-1 agreement (flip rate) primary (KL–flip Spearman 0.98); clipped top-k KL secondary.
Hoeffding needs 18,445 tokens for ±1% at 95%; empirical-Bernstein/betting bounds need ~2–3k when
p ≈ 3%. Betting confidence sequences (Waudby-Smith & Ramdas 2024) are valid at every sample size ⇒
continuous monitoring and anytime stopping. Stratify shadow-exact replays by gate band and rung;
importance-weight. Control loop: PI on log τ (or κ) against the UPPER confidence bound, update period
longer than the CS shrink time, anti-windup at palette limits.

## 8. Leapfrog ideas
1. Successive refinement (Equitz–Cover): Gaussian/MSE sources are successively refinable, so 2-bit base +
   2-bit residual can reach 4-bit RD performance; SliceMoE's nested failure (PPL 7.0→10.2) reflects naive
   uniform slicing. Store base + separately coded residual; stream only the residual on escalation.
   Note V4's MXFP4 gives a free nested 3.25-bit slice (SPEC-DERISK section 1).
2. Lossless entropy coding on the NVMe tier: sources conflict (ZipNN useless on 4-bit vs ISCA 2026
   2–10× below stored bitwidth) — measure on target models.
3. KV: TurboQuant 3.5 bits quality-neutral; use the outlier-preserving reference scheme. V4 KV negligible.
4. Disk layout: experts read whole (6–13 MB) ⇒ already sequential-speed; contiguous gate/up/down per
   expert plus co-activation clustering; Hilbert curves add little.

## 9. Priorities (payoff per engineering hour) and publishable research
1. Water-filling + MCKP planner (1–2 weeks). 2. Replace top-1-exact with the section 6 rule (days).
3. Trellis codes for NVMe-streamed copies (reuse ik_llama.cpp). 4. Robust GreedyDual + io_uring tuning
(1–2 weeks; expect 1.1–1.4× I/O over LRU). 5. Anytime-valid meter + PI loop + speculation toggle (1 week).
Publishable: Fano/list-decoding bound on expert-cache misses from measured conditional entropy;
successively refinable expert codes; lossless cache-aware drafting.
