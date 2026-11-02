"""Routing entropy and the list-decoding Fano floor on expert-cache misses (P0.3).

The quantity of interest is H(S_{l+1} | side information), where S_{l+1} is the set of
experts routed at layer l+1. Exact plug-in estimates over C(256, 6) sets are hopeless, so
we estimate it per expert-inclusion indicator and sum (subadditivity):

    H(S | Y) <= sum_e H(1[e in S] | Y)

with Y = the top-1 expert of the previous layer (a coarse stand-in for the hidden-state
bucket in SPEC-MATH section 3). The sum is an upper estimate of the conditional entropy,
so the floor computed from it is an estimate, not a proven lower bound: report it as such.
"""

from __future__ import annotations

import math

import numpy as np

from .trace import Trace


def _h2(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def inclusion_matrix(trace: Trace, layer: int) -> np.ndarray:
    """Boolean [T, N]: expert e routed at ``layer`` for token t."""
    T = trace.n_tokens
    m = np.zeros((T, trace.n_experts), dtype=bool)
    m[np.arange(T)[:, None], trace.topk[:, layer, :]] = True
    return m


def marginal_set_entropy(trace: Trace, layer: int) -> float:
    """Sum over experts of the binary entropy of inclusion (bits). Upper-bounds H(S_l)."""
    p = inclusion_matrix(trace, layer).mean(axis=0)
    return float(_h2(p).sum())


def conditional_set_entropy(trace: Trace, layer: int, min_count: int = 1) -> float:
    """sum_e H(1[e in S_layer] | top-1 expert at layer-1), plug-in estimate (bits)."""
    if layer == 0:
        return marginal_set_entropy(trace, 0)
    inc = inclusion_matrix(trace, layer).astype(np.float64)
    y = trace.topk[:, layer - 1, 0]
    N = trace.n_experts
    counts = np.bincount(y, minlength=N).astype(np.float64)
    # sum of indicators per (y, e)
    sums = np.zeros((N, N))
    np.add.at(sums, y, inc)
    keep = counts >= min_count
    total = counts[keep].sum()
    if total == 0:
        return float("nan")
    p_e_given_y = sums[keep] / counts[keep][:, None]
    h = _h2(p_e_given_y).sum(axis=1)  # per y value
    return float((counts[keep] / total * h).sum())


def log2_binom(n: int, k: int) -> float:
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / math.log(2)


def fano_list_floor(h_cond_bits: float, n_outcomes_log2: float, list_size_log2: float) -> float:
    """List-decoding Fano: P_miss >= (H(X|Y) - 1 - log2 L) / log2(N/L), clipped to [0, 1].

    ``n_outcomes_log2`` is log2 of the number of possible outcomes, ``list_size_log2`` log2 of
    the number of outcomes a predictor may list (cache-sized guess).
    """
    denom = n_outcomes_log2 - list_size_log2
    if denom <= 0:
        return 0.0
    return float(min(1.0, max(0.0, (h_cond_bits - 1.0 - list_size_log2) / denom)))


def layer_report(trace: Trace, cache_experts_per_layer: int) -> list[dict]:
    """Per layer: marginal and conditional entropy estimates and the implied miss floor.

    The predictor is allowed to list every set fully contained in its cached experts,
    i.e. L = C(c, k) sets for c cached experts per layer. Entropy estimates are clipped to
    log2 C(N, k), which the per-indicator sum can exceed.
    """
    N, k = trace.n_experts, trace.top_k
    n_log2 = log2_binom(N, k)
    c = max(min(cache_experts_per_layer, N), k)
    list_log2 = log2_binom(c, k)
    rows = []
    for l in range(trace.n_layers):
        hm = min(marginal_set_entropy(trace, l), n_log2)
        hc = min(conditional_set_entropy(trace, l), n_log2)
        # Hash layers route by token id, which a predictor knows exactly: floor is 0.
        is_hash = l in trace.hash_layers
        floor = 0.0 if is_hash else fano_list_floor(hc, n_log2, list_log2)
        rows.append(
            {
                "layer": l,
                "hash_layer": is_hash,
                "H_marginal_bits": round(hm, 3),
                "H_cond_prev_top1_bits": round(hc, 3),
                "set_space_bits": round(n_log2, 3),
                "fano_floor_estimate": round(floor, 4),
            }
        )
    return rows
