# SPEC-SYSTEM — Architecture, planner, UX (condensed; full: SPEC-SYSTEM-FULL.md if present)

## 1. Positioning
"Any model, any machine" is physically false. The honest, defensible claim: every model gets an honest
answer on every machine — a predicted speed + measured fidelity contract, a lossless floor, lossy rungs
you opt into, and continuous verification against the exact model. Incumbents (colibri `coli plan`,
llama.cpp `--fit`, LM Studio `--estimate-only`, Ollama scheduler, Strata, kTransformers) automate
placement for capacity; none predicts quality and then measures it; their failures are silent (wrong
memory estimates, silent CPU fallback, knob soup). The durable moat is the published, reproducible
fidelity dataset (model × machine × plan → tok/s, top-1, KL).

## 2. Planning problem
Inputs: hardware profile H measured by a 60 s microbench (capacity, sustained bandwidth per tier,
latency, compute, links); model profile M from GGUF/safetensors headers via HTTP range reads (per-tensor
size/dtype, dense vs routed sets, E, top-k, shared, L, attention type, KV bytes/token); user constraints
U (min tok/s, max fidelity loss, context, concurrency).
Outputs: placement x(t,i), precision b(t), prefetch policy, speculation config, KV dtype/window, prefill
chunk, predicted (tok/s, TTFT, fidelity) with confidence intervals.
Decode roofline: Bytes_i = Σ dense tensors in tier i + Σ_layers Σ_e p_l(e)[1 − h_{<i}(e)] size(e,b_e)
+ KV; T ≈ max_i Bytes_i/(η_i B_i) with overlap, Σ without; η ≈ 0.6–0.85 for DRAM; calibrate after a
20-token run; sanity: dense 70B Q4 on Strix Halo 215/40 ≈ 5.4 ceiling vs ~5 measured.
Placement: multiple-choice knapsack; greedy on access density (seconds saved per byte of fast memory);
ILP only offline for multi-GPU validation; Bayesian-tune ~4–6 residual knobs; re-optimize hot set
from live routing stats.
Graceful degradation when physics says no: (1) sibling/distilled model meeting the speed floor; (2) LAN
pooling if peers found (activations ~14 KB/hop; capacity not speed except over RDMA); (3) slow mode
with honest ETA; (4) refuse with the exact shortfall. Never fail like "requires 446 GiB" with no option.
Impossible/unusable pairs: dense 405B on 16 GB (~32 s/token); Kimi K3 on 64 GB (3–5 s/token);
GLM-5.2 usable only ≥128 GB unified or multi-GPU.

## 3. Architecture
Rust daemon (single binary): CLI/TUI/web dashboard; OpenAI + Anthropic APIs (tools, SSE, prefix cache);
preflight (HF header probe → model profile); hardware prober + microbench; PLANNER (roofline → water-
filling → greedy MCKP → plan + predicted tok/s, fidelity); fetcher (lazy per-expert Xet range downloads,
hash-verified, sandboxed parser); MEMORY SYSTEM (tier manager VRAM/RAM/NVMe×n/peer; hot set; prefetch;
io_uring/O_DIRECT; ladder sidecar store with exact + q4 + q2 per expert); FIDELITY METER (shadow-exact
replay, KL/top-1 confidence sequences, eval packs, auto-rollback); autotuner; local-only telemetry.
Engine hook: "expert provider" callback (returns pointer + precision per (layer, expert)) into vendored
libllama/ggml at the seam used by llama.cpp #25294/#31469/#29887 — ~1–2k lines of patch, upstream it.
Why: model churn is the killer cost (colibri writes one C engine per family; llama.cpp gets new
architectures in days); ggml covers CPU SIMD/CUDA/HIP/Vulkan/Metal/SYCL; Rust parsing with checked
arithmetic for the 2026 GGUF CVEs. Own the brain and the memory; borrow the muscles. Custom kernels only
where measured bottleneck (e.g. fused dequant-on-arrival for ladder rungs).
Formats: GGUF canonical for execution; safetensors for planning/lazy fetch; ladder sidecar = per-expert
contiguous q2/q4 blobs + index next to the original file (exact floor never rewritten).

## 4. Decision policy (evaluate in order, stop when the speed target is met)
0 fit resident (best quant that fits) → 1 lossless streaming + learned hot set + lookahead (10–13× over
naive mmap) → 2 warm-start expert history (~1.8× first run) → 3 speculation (MTP/prompt lookup; model the
expert union, not a flat 2×) → 4 precision ladder (low-gate misses from q2/q4; bound-governed) → 5 KV
quant/window → 6 activation sparsity on dense part (TEAL) → 7 opt-in workload pruning pack (REAP, stated
eval delta, never default) → 8 LAN pooling → 9 degrade (sibling / slow mode / refuse).
Quality guard: shadow-exact replay on 1–2% of tokens in idle time; rolling top-1 agreement and clipped
KL shown live; auto-escalate rungs on breach; eval packs (50–200 items per domain) at install;
determinism tests (token-exact oracles) in CI.

## 5. UX
`banach run <model>` → probe (12 s) → header read (3 requests, 0 weight bytes) → preflight card:
Exact (lossless) tok/s ± CI, fidelity 100%, disk; Balanced (ladder) tok/s ± CI, certified fidelity;
Code pack; Faster sibling; "Download now: N GB (dense + hot experts); rest on demand" → proceed?
First-run autotune ≤2 min; nightly hot-set rebuild; pack suggestions from traffic mix. Dashboard:
tier occupancy, expert heat map, bytes/token by tier, fidelity meter, predicted vs actual.
Why incumbents won: `ollama run x` + curated library; LM Studio GUI + estimator; llama.cpp universal
hardware + fast architecture support; colibri jaw-dropping true claim + honest numbers + start-here script.

## 6. MVP and roadmap (adjusted by SPEC-DERISK: Phase 0 first; target chosen from its verdict)
Platforms: Linux x86-64 + CUDA, macOS Apple Silicon. Models: Qwen3.6-35B-A3B (fits everywhere),
DeepSeek V4 Flash (hero, MIT), Qwen3.8-27B dense. Months 0–1 validate physics; 1–2 daemon/probe/
planner/preflight (±30%); 2–3 hook/tier manager/io_uring/warm-start/byte-exact CI; 3–4 ladder + meter +
APIs + launch; 4–6 Windows via Vulkan, lazy Xet download, autotune, code pack; 6–9 LAN pooling, GLM/Kimi
slow mode, Strix Halo; 9–12 continuous self-optimization, community leaderboard.
Killer demo (60 s): on the chosen target, `ollama run deepseek-v4-flash` fails; `banach run` shows the
card, starts chatting after ~25% of weights, holds the measured tok/s with a live certified fidelity
meter; a coding agent connects to the local API and fixes a real bug; `banach plan kimi-k3` answers honestly ("~0.05–0.4
tok/s; try deepseek-v4-flash or pool your Mac").
Risks: NVMe vs DRAM 10–100× gap; hook patch breaking on upstream refactors; monthly architecture churn;
quality cliffs at 2-bit; HF rate limits; derived-weight licensing; GGUF CVEs; support burden; solo
capacity (~10–15 h/week; recruit 2–3 co-maintainers by month 4).
