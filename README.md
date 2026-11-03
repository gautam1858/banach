# Banach

> Banach-Tarski for language models: more model than your machine should hold.

Banach runs very large Mixture-of-Experts models on ordinary machines by keeping the exact
model on NVMe as a lossless floor, serving most tokens from a reduced-precision hot set in
RAM/VRAM, escalating precision only where an error bound says it matters, and continuously
measuring fidelity against the exact model on your own traffic.

## Status

Phase 0 (feasibility). The headline target is chosen from the Phase 0 verdict; see
[docs/spec/BANACH-SPEC.md, section 0](docs/spec/BANACH-SPEC.md).

## Layout

| Path | Contents |
|---|---|
| `docs/spec/` | Specifications. Precedence: BANACH-SPEC, then SPEC-DERISK, SPEC-MATH, SPEC-LADDER, SPEC-SYSTEM |
| `tracer/` | `banach-trace`, the llama.cpp routing tracer (Phase 0.1) |
| `phase0/` | Phase 0 analysis: trace import, cache simulator, entropy floor, ladder sweep, fidelity certificate, decode model, K1/K2 benchmark scripts |

## License

Apache-2.0, see [LICENSE](LICENSE).
