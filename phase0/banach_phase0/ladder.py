"""Precision-ladder sweep (SPEC-DERISK P0.4, SPEC-MATH section 6).

For every routed slot the bound on output error from serving expert i at b bits is

    g_i * eps_i(b) * ||x||

with eps_i(b) the profiled relative error ||f_i(x) - f_i,b(x)|| / ||x||. Rule: use the
cheapest rung whose bound is <= tau; if none qualifies, serve exact (an NVMe read unless
the exact copy is resident). Skipping a tail set S is allowed only when 2 * g_S * M < tau.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .trace import Trace


@dataclass
class LadderResult:
    tau: float
    rung_fraction: np.ndarray  # fraction of slots served at each rung, last entry = exact
    escalations_per_token: float
    escalation_bytes_per_token: float
    mean_bits: float
    skip_fraction: float

    def as_dict(self, rung_bits: tuple[float, ...]) -> dict:
        names = [f"{b}b" for b in rung_bits] + ["exact"]
        return {
            "tau": self.tau,
            "rungs": {n: round(float(f), 5) for n, f in zip(names, self.rung_fraction)},
            "escalations_per_token": round(self.escalations_per_token, 3),
            "escalation_bytes_per_token": self.escalation_bytes_per_token,
            "mean_bits": round(self.mean_bits, 3),
            "skip_fraction": round(self.skip_fraction, 5),
        }


def assign_rungs(gates: np.ndarray, xnorm: np.ndarray, eps: np.ndarray, tau: float) -> np.ndarray:
    """Rung index per slot.

    gates [T, L, k]; xnorm [T, L]; eps [T, L, k, R] relative error per slot per rung, rungs
    ordered cheapest first. Returns int [T, L, k] in 0..R, where R means exact.
    """
    bound = gates[..., None] * eps * xnorm[..., None, None]
    ok = bound <= tau
    R = eps.shape[-1]
    first_ok = np.where(ok.any(axis=-1), np.argmax(ok, axis=-1), R)
    return first_ok


def skip_mask(gates: np.ndarray, m_norm: float, tau: float, max_skip: int | None = None) -> np.ndarray:
    """Slots that may be dropped: the lowest-gate tail S with 2 * g_S * M < tau, never top-1.

    gates are sorted descending along the last axis (top-1 first).
    """
    k = gates.shape[-1]
    tail = np.flip(np.cumsum(np.flip(gates, -1), -1), -1)  # tail[j] = sum of gates j..k-1
    allowed = 2.0 * tail * m_norm < tau
    allowed[..., 0] = False
    if max_skip is not None:
        allowed[..., : max(1, k - max_skip)] = False
    return allowed


def sweep(
    trace: Trace,
    eps_table: np.ndarray,
    rung_bits: tuple[float, ...],
    taus,
    exact_bpw: float,
    params_per_expert: int,
    m_norm: float | None = None,
) -> list[LadderResult]:
    """Run the ladder rule at each tau.

    eps_table: [L, N, R] per-expert relative error at each rung (median or p95 from
    quantisation-time profiling). Exact-rung slots are counted as escalation reads.
    """
    L, N, R = eps_table.shape
    if R != len(rung_bits):
        raise ValueError("eps_table last axis must match rung_bits")
    eps = eps_table[np.arange(L)[None, :, None], trace.topk]  # [T, L, k, R]
    out = []
    T = trace.n_tokens
    for tau in taus:
        r = assign_rungs(trace.gates, trace.xnorm, eps, float(tau))
        skipped = skip_mask(trace.gates, m_norm, float(tau)) if m_norm else np.zeros_like(r, bool)
        served = ~skipped
        frac = np.array([((r == j) & served).sum() for j in range(R + 1)], dtype=float)
        n_served = served.sum()
        frac = frac / n_served if n_served else frac
        bits = np.array(list(rung_bits) + [exact_bpw])
        esc = ((r == R) & served).sum() / T
        out.append(
            LadderResult(
                tau=float(tau),
                rung_fraction=frac,
                escalations_per_token=float(esc),
                escalation_bytes_per_token=float(esc * params_per_expert * exact_bpw / 8),
                mean_bits=float((frac * bits).sum()),
                skip_fraction=float(skipped.sum() / skipped.size),
            )
        )
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties)."""
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def _rank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks
