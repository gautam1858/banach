"""Bandwidth/compute model of decode speed (SPEC-DERISK P0.6, SPEC-MATH section 5).

    T_compute = max(T_dram, T_cpu)
    T_token   = T_compute + max(0, T_io - overlap * T_compute) + T_sync

T_dram: always-active bytes + resident routed bytes over achieved DRAM bandwidth.
T_io:   miss bytes over sustained NVMe bandwidth at the read size used.
With MTP/speculation one verify step reads the dense path once for gamma+1 tokens and
the union of their routed experts; throughput is accepted tokens per step time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model import GB, MoEModel


@dataclass
class Machine:
    name: str
    dram_bw: float  # achieved bytes/s (STREAM triad)
    nvme_bw: float  # sustained bytes/s at the expert read size (fio, O_DIRECT)
    # Achieved FLOP/s on decode kernels incl. dequant. 26 GFLOP/token at 10 tok/s is ~5% of
    # Zen 5 AVX-512 peak (SPEC-DERISK section 4); 1 TFLOP/s is a placeholder until K1 measures it.
    cpu_flops: float = 1.0e12
    ram_bytes: float = 32 * GB
    overlap: float = 0.8  # fraction of compute time I/O can hide under
    t_sync: float = 0.002  # per-step fixed overhead, seconds


# Reference box assumptions from SPEC-DERISK section 4 (to be replaced by K2 measurements).
REFERENCE_32GB = Machine("32GB-DDR5-Gen4", dram_bw=70 * GB, nvme_bw=6 * GB)


def expert_union(n_tokens: int, k: int, n_experts: int) -> float:
    """Expected distinct experts touched by n tokens with independent top-k of N routing."""
    return n_experts * (1.0 - (1.0 - k / n_experts) ** n_tokens)


def expected_accepted(alpha: float, gamma: int) -> float:
    """Expected tokens produced per verify step with draft length gamma, acceptance alpha."""
    if alpha >= 1.0:
        return float(gamma + 1)
    return (1.0 - alpha ** (gamma + 1)) / (1.0 - alpha)


@dataclass
class Prediction:
    tok_s: float
    t_step: float
    t_dram: float
    t_cpu: float
    t_io: float
    tokens_per_step: float
    miss_bytes_per_step: float

    def as_dict(self) -> dict:
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def predict(
    model: MoEModel,
    machine: Machine,
    bpw: float,
    hit_rate: float,
    flops_per_token: float = 26e9,
    alpha: float = 0.0,
    gamma: int = 0,
    miss_bpw: float | None = None,
) -> Prediction:
    """Predict decode tok/s.

    hit_rate applies to routed experts. Resident routed bytes are read from DRAM at ``bpw``;
    misses are read from NVMe at ``miss_bpw`` (defaults to ``bpw``). With gamma > 0 the step
    covers gamma+1 positions and the routed set is the expected union; hit rate is assumed
    to apply to the union (optimistic if the drafted tokens route colder).
    """
    miss_bpw = bpw if miss_bpw is None else miss_bpw
    n_pos = gamma + 1
    experts_per_layer = expert_union(n_pos, model.top_k, model.n_experts) if gamma else model.top_k
    routed = experts_per_layer * model.n_layers
    resident_bytes = routed * hit_rate * model.expert_bytes(bpw)
    miss_bytes = routed * (1.0 - hit_rate) * model.expert_bytes(miss_bpw)

    # Missed experts land in a RAM buffer and are read from there by the kernels, so every
    # routed byte crosses DRAM once; hit rate only changes what NVMe must supply.
    t_dram = (model.dense_bytes_per_token + resident_bytes + miss_bytes) / machine.dram_bw
    t_cpu = flops_per_token * n_pos / machine.cpu_flops
    t_io = miss_bytes / machine.nvme_bw
    t_compute = max(t_dram, t_cpu)
    t_step = t_compute + max(0.0, t_io - machine.overlap * t_compute) + machine.t_sync
    tps = expected_accepted(alpha, gamma) if gamma else 1.0
    return Prediction(
        tok_s=tps / t_step,
        t_step=t_step,
        t_dram=t_dram,
        t_cpu=t_cpu,
        t_io=t_io,
        tokens_per_step=tps,
        miss_bytes_per_step=miss_bytes,
    )


def should_speculate(t1: float, alpha: float, gamma: int, t_draft: float, t_fixed: float,
                     missing_bytes: float, bw: float) -> bool:
    """SPEC-MATH section 5 break-even test for one step."""
    lhs = expected_accepted(alpha, gamma)
    rhs = (t_draft + t_fixed + missing_bytes / bw) / t1
    return lhs > rhs


def required_hit_rate(model: MoEModel, machine: Machine, bpw: float, target_tok_s: float,
                      **kw) -> float | None:
    """Smallest hit rate reaching ``target_tok_s`` (bisection), or None if unreachable."""
    if predict(model, machine, bpw, 1.0, **kw).tok_s < target_tok_s:
        return None
    lo, hi = 0.0, 1.0
    if predict(model, machine, bpw, 0.0, **kw).tok_s >= target_tok_s:
        return 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if predict(model, machine, bpw, mid, **kw).tok_s >= target_tok_s:
            hi = mid
        else:
            lo = mid
    return hi


def fit_overlap_sync(model: MoEModel, machine: Machine, runs: list[dict]) -> tuple[float, float, float]:
    """Fit (overlap, t_sync) by grid search to measured runs.

    Each run: {"bpw", "hit_rate", "tok_s", optional "alpha", "gamma"}.
    Returns (overlap, t_sync, max relative error). The protocol requires <= 15% on colibri.
    """
    best = (machine.overlap, machine.t_sync, float("inf"))
    for ov in np.linspace(0.0, 1.0, 21):
        for ts in np.linspace(0.0, 0.05, 51):
            m = Machine(**{**asdict(machine), "overlap": float(ov), "t_sync": float(ts)})
            errs = []
            for r in runs:
                p = predict(model, m, r["bpw"], r["hit_rate"],
                            alpha=r.get("alpha", 0.0), gamma=r.get("gamma", 0))
                errs.append(abs(p.tok_s - r["tok_s"]) / r["tok_s"])
            e = max(errs) if errs else 0.0
            if e < best[2]:
                best = (float(ov), float(ts), e)
    return best
