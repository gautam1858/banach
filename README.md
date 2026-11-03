# Banach

> Banach-Tarski for language models: more model than your machine should hold.

Banach runs very large Mixture-of-Experts models on ordinary machines by keeping the exact
model on NVMe as a lossless floor, serving most tokens from a reduced-precision hot set in
RAM/VRAM, escalating precision only where an error bound says it matters, and continuously
measuring fidelity against the exact model on your own traffic.

## Status

- Phase 0 (feasibility): tooling complete and validated on a small real MoE. The V4 Flash
  verdict needs K1/K2 and traces on a machine with 128 GB or more.
- Phase 1 (probe, preflight, planner, preflight card): working; predicted vs measured
  within the +-30% bar on held-out data. See [docs/VALIDATION.md](docs/VALIDATION.md).
- Phase 2 (lossless streaming): `banach-engine` keeps a RAM budget for routed experts with
  parallel whole-expert fetch, frequency-gated admission and background eviction;
  byte-identical to stock llama.cpp in every tested configuration; 2.1x over naive page
  faulting at the same budget. `banach run` plans the budget and launches it.
- Phase 3 (precision ladder, fidelity meter): llama.cpp patch serving each routed slot from
  the exact or a low-bit copy, teacher-forced fidelity sweeps, betting confidence sequences
  and a tau controller. Numerically exact plumbing; measured trade-offs and the open issues
  (near-tie floor, gate-only beating the eps bound, certification needs longer runs) are in
  docs/VALIDATION.md.

The headline target is chosen from the Phase 0 verdict; see
[docs/spec/BANACH-SPEC.md, section 0](docs/spec/BANACH-SPEC.md).

## Quick start

```bash
cargo build --release
./target/release/banach probe --disk /path/to/model.gguf
./target/release/banach plan hf:unsloth/DeepSeek-V4-Flash-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf --min-tok-s 10
```

`plan` reads only the GGUF headers (a few MB over HTTP range requests), so it answers
"will it run, how fast, how good" before any weights are downloaded.

## Layout

| Path | Contents |
|---|---|
| `docs/spec/` | Specifications. Precedence: BANACH-SPEC, then SPEC-DERISK, SPEC-MATH, SPEC-LADDER, SPEC-SYSTEM |
| `src/` | `banach` (Rust): hardware probe, GGUF header parser (local or HTTP range), model profile, planner, preflight card |
| `phase1/machines/` | reference hardware profiles for planning other machines (`banach plan --hw`) |
| `engine/` | `banach-engine`, lossless expert streaming on llama.cpp, plus the byte-exact oracle script |
| `tracer/` | `banach-trace`, the llama.cpp routing tracer (Phase 0.1) |
| `phase0/` | Phase 0 analysis: trace import, cache simulator, entropy floor, ladder sweep, fidelity certificate, decode model, K1/K2 benchmark scripts |

## License

Apache-2.0, see [LICENSE](LICENSE).
