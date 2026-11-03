# SPEC-LADDER — Precision ladder and physics (condensed; full: SPEC-LADDER-FULL.md if present)
NOTE: numbers here predate SPEC-DERISK; where they conflict, SPEC-DERISK wins (in particular the
always-active ~7 GB/token dense path and the 18–21 GB realistic hot set on 32 GB).

## 1. Physics
Decode time ≈ max over tiers of (bytes from that tier ÷ its bandwidth); prefill on streamed MoE
touches essentially every expert per chunk (P(unused) = (1−6/256)^512 ≈ 5e-6) ⇒ one pass of the
expert pool per chunk; use the largest chunk KV allows and cache prefixes aggressively.
Bandwidth tiers (late 2026): Gen4 NVMe ~5.5 GB/s usable (7 rated); Gen5 ~11 (14.5 rated); DDR5
dual-channel ~60–75 GB/s achieved; Apple M5 Max 460–614 GB/s; consumer VRAM 400–1800 GB/s.
Routed bytes per token, cold, 4-bit: DeepSeek V4 Flash ≈ 3.4 GB; GLM-5.2 ≈ 11 GB; Kimi K3 ≈ 26 GB.
Master equation per decode step: T_ssd = U·(1−h)·k_eff·L·S_e·φ/B_ssd; T_mem = (D + U·h·k_eff·L·S_e·β)/B_mem;
tok/s ≈ τ/max(T_ssd, T_mem, T_compute). h hit rate; k_eff effective experts; S_e bytes/expert at 4-bit;
φ fraction fetched on miss; β cache precision vs 4-bit; D dense bytes; τ tokens accepted per verify step;
U union growth. Multiplicative levers: cache bits (h), bits per miss (φ), k_eff, speculation (τ/U), SSD
count. Stacked: ~5–6× fewer NVMe bytes per token.
Walls: GLM-5.2's 9.9 GB dense part caps CPU-only DDR5 at ~6 tok/s; Kimi K3 ~1 tok/s at best on consumer
hardware; dense 405B unusable below 256 GB+.

## 2. Rungs
R0 skip (low gate mass, never top-1) | R1 2-bit copy in RAM (hot set, ~2× more experts fit) | R2 native
4-bit in RAM/VRAM (top-ranked + hottest) | R3 native 4-bit streamed from NVMe (lossless floor) |
R4 network (not yet downloaded, lazy fetch; never on the critical path).
Policy: escalate per the SPEC-MATH section 6 bound; a fidelity budget (e.g. ≥99% of gate mass served at
≥4-bit, or a certified top-1 bound) adjusts τ automatically. Learning: EMA hotness per workload profile;
overnight shadow-exact replay reports KL and top-1 agreement on the user's own prompts.
Format: MXFP4 experts (V4 Flash, Kimi K3) give a free nested ~3.25-bit slice; 2-bit copies must be
separately quantized (+~50% disk, e.g. ~145 GB + ~72 GB for V4 Flash); GLM-5.2 (FP8 source) can build
AMAT-style nested slices in a later phase.

## 3. Technique verdicts
Lossless tiered streaming (colibri; llama.cpp #25294, #31469 — 16–22 tok/s byte-exact on 68 GB model
with 12 GB VRAM + 16 GB RAM): foundation, reuse. HOBBIT mixed precision on miss (<1% loss, 2.3–3.9×):
core. DynaExq hot-high/cold-low (+4 pts on Qwen3-80B): core. SliceMoE bit-sliced (simulated; 2-bit
nested too lossy): phase 2, 2-bit only for low-gate slots. Cache-Prior/PSR rerouting (2–2.3× but lost on
HumanEval): opt-in, low-ranked slots only. Router fine-tuning for locality: defer (failed <1% PPL gates).
Dynamic skipping (ACE etc.): simple rule only, static-k often as good. REAP pruning: packs and priors
only, never the only copy (generic-calibration collapse). MoBE low-rank: possible future rung 0.
Lossless entropy coding: skip for 4-bit experts; optional for BF16/FP8 downloads. MTP/speculation: core,
with expert-union accounting; int4 MTP heads gave 0% acceptance in colibri — keep heads int8+.

## 4. Projections (ESTIMATES, superseded by SPEC-DERISK where they conflict)
32 GB DDR5 + Gen4, no GPU: V4 Flash 7–10 tok/s claimed earlier; DERISK caps this at ~7.5–8 without MTP,
~12–13 with MTP at 100% hit, and finds the fidelity target the harder wall. 32 GB + RTX 5080 16 GB:
12–18. M5 Max 64 GB: 15–22. 128 GB unified: V4 Flash 25+; GLM-5.2 6–10. Kimi K3 ~1.
Quality: expect ≤1–2 points loss on coding/tool-use at a 99% gate-mass budget (from HOBBIT, DynaExq,
MoBE — all on ≤80B models; untested at 284B). 2-bit applied broadly is too lossy.

## 5. UX that matters
Lazy download: HF Xet supports range requests; publish an expert-major repack (≤50 GB files); fetch
dense + hot set first (~25–45 GB for V4 Flash, estimate), backfill in background; network misses
never block decode. Dashboard: ladder view, day-1 vs day-7 tok/s, fidelity meter. APIs: OpenAI +
Anthropic base URLs (table stakes).
