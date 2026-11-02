"""Model constants used throughout Phase 0 (SPEC-DERISK section 1)."""

from __future__ import annotations

from dataclasses import dataclass, field

GIB = 1024**3
GB = 1e9


@dataclass(frozen=True)
class MoEModel:
    name: str
    n_layers: int
    n_experts: int  # routed experts per layer
    top_k: int
    params_per_expert: int
    # Always-active bytes read per token (dense path + head), at native precision.
    dense_bytes_per_token: float
    hash_layers: tuple[int, ...] = field(default_factory=tuple)

    def expert_bytes(self, bpw: float) -> float:
        return self.params_per_expert * bpw / 8.0

    def pool_bytes(self, bpw: float) -> float:
        return self.expert_bytes(bpw) * self.n_layers * self.n_experts

    def routed_bytes_per_token(self, bpw: float) -> float:
        return self.expert_bytes(bpw) * self.n_layers * self.top_k

    def experts_that_fit(self, capacity_bytes: float, bpw: float) -> int:
        return int(capacity_bytes // self.expert_bytes(bpw))


# 43 MoE layers, 256 routed experts, top-6, hidden 4096, expert intermediate 2048.
# Always-active path ~6.27 GiB dense + 1.06 GiB head (colibri docs).
DEEPSEEK_V4_FLASH = MoEModel(
    name="deepseek-v4-flash-0731",
    n_layers=43,
    n_experts=256,
    top_k=6,
    params_per_expert=3 * 4096 * 2048,
    dense_bytes_per_token=(6.27 + 1.06) * GIB,
    hash_layers=(0, 1, 2),
)

# GLM-5.3-Flash: 42 MoE layers x 288 experts, top-8, same expert size.
GLM_53_FLASH = MoEModel(
    name="glm-5.3-flash",
    n_layers=42,
    n_experts=288,
    top_k=8,
    params_per_expert=3 * 4096 * 2048,
    dense_bytes_per_token=0.0,  # unknown until measured; fill from header probe
)

MODELS = {m.name: m for m in (DEEPSEEK_V4_FLASH, GLM_53_FLASH)}

# Bit widths swept in the protocol (IQ2_XXS, 2.5, IQ3_XXS, MXFP4).
BPW_GRID = (2.06, 2.5, 3.06, 4.25)
CAPACITY_GB_GRID = (12, 20, 25, 40)
