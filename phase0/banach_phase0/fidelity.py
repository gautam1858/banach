"""Anytime-valid certification of top-1 disagreement (SPEC-DERISK P0.5, SPEC-MATH section 7).

D_t = 1[top-1 under the plan != top-1 under the exact model]. To certify p < p0 we test
H0: p >= p0 with the betting wealth process

    K_t = prod_{i<=t} (1 + lambda_i * (p0 - D_i)),    lambda_i in [0, 1/(1 - p0))

where lambda_i is chosen from D_1..D_{i-1} only (predictable). Under H0, K_t is a
nonnegative supermartingale, so by Ville's inequality P(sup_t K_t >= 1/alpha) <= alpha:
certification is valid at every sample size and the meter may be watched continuously.

The upper confidence sequence is the smallest p0 on a grid not yet rejected; once a p0 is
rejected it stays rejected (running intersection), which keeps the sequence nested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _lambdas(d: np.ndarray, p0: np.ndarray, c: float = 0.5, prior_n: float = 1.0) -> np.ndarray:
    """Predictable bets, aGRAPA-style: lambda = clip(mean/var of (p0 - D), 0, c/(1-p0)).

    d: [T] in {0,1}. p0: [G]. Returns [T, G]. A pseudo-observation at p0/2 regularises
    the first steps.
    """
    T = len(d)
    cum = np.concatenate([[0.0], np.cumsum(d)])[:T, None]  # sum of D strictly before step t
    n = np.arange(T, dtype=float)[:, None]
    p_hat = (cum + prior_n * 0.5 * p0[None, :]) / (n + prior_n)
    mu = p0[None, :] - p_hat  # estimated mean of (p0 - D)
    var = p_hat * (1.0 - p_hat) + 1e-6  # variance of (p0 - D) equals var(D)
    lam = mu / (var + mu**2)
    cap = c / (1.0 - p0)
    return np.clip(lam, 0.0, cap[None, :])


@dataclass
class Certificate:
    p0: float
    alpha: float
    tokens: int
    flips: int
    certified: bool
    certified_at: int | None  # first token count with K_t >= 1/alpha
    log_wealth: float
    ucb: float  # anytime-valid upper confidence bound at the end

    def as_dict(self) -> dict:
        return {
            "p0": self.p0,
            "alpha": self.alpha,
            "tokens": self.tokens,
            "flips": self.flips,
            "point_estimate": self.flips / self.tokens if self.tokens else None,
            "certified": self.certified,
            "certified_at_tokens": self.certified_at,
            "log_wealth": round(self.log_wealth, 4),
            "ucb": self.ucb,
        }


def wealth_paths(d: np.ndarray, p0s: np.ndarray, c: float = 0.5) -> np.ndarray:
    """log K_t for each grid value. Returns [T, G]."""
    d = np.asarray(d, dtype=float)
    p0s = np.atleast_1d(np.asarray(p0s, dtype=float))
    lam = _lambdas(d, p0s, c)
    inc = np.log1p(lam * (p0s[None, :] - d[:, None]))
    return np.cumsum(inc, axis=0)


def certify(d, p0: float = 0.01, alpha: float = 0.05, c: float = 0.5,
            with_ucb: bool = True) -> Certificate:
    """Test H0: p >= p0 on a disagreement sequence and (optionally) compute the anytime UCB."""
    d = np.asarray(d, dtype=float)
    if d.ndim != 1 or (d.size and not np.all((d == 0) | (d == 1))):
        raise ValueError("d must be a 1-D 0/1 sequence")
    thr = np.log(1.0 / alpha)
    if d.size == 0:
        return Certificate(p0, alpha, 0, 0, False, None, 0.0, 1.0)
    lw = wealth_paths(d, np.array([p0]), c)[:, 0]
    hit = np.nonzero(lw >= thr)[0]
    at = int(hit[0]) + 1 if hit.size else None
    return Certificate(
        p0=p0,
        alpha=alpha,
        tokens=int(d.size),
        flips=int(d.sum()),
        certified=at is not None,
        certified_at=at,
        log_wealth=float(lw[-1]),
        ucb=float(upper_confidence_sequence(d, alpha, c=c)[-1]) if with_ucb else float("nan"),
    )


def upper_confidence_sequence(
    d, alpha: float = 0.05, grid: np.ndarray | None = None, c: float = 0.5
) -> np.ndarray:
    """Anytime-valid upper bound on p after each token. Returns [T]."""
    d = np.asarray(d, dtype=float)
    if grid is None:
        grid = np.unique(np.concatenate([np.geomspace(1e-4, 0.2, 240), np.linspace(0.2, 1.0, 81)]))
    grid = np.sort(np.asarray(grid, dtype=float))
    grid = grid[grid < 1.0]
    lw = wealth_paths(d, grid, c)
    rejected = np.maximum.accumulate(lw >= np.log(1.0 / alpha), axis=0)  # [T, G]
    # p0 rejected means "p < p0" established; the UCB is the smallest rejected grid value.
    T = len(d)
    ucb = np.ones(T)
    any_rej = rejected.any(axis=1)
    first = np.argmax(rejected, axis=1)
    ucb[any_rej] = grid[first[any_rej]]
    return ucb


def tokens_to_certify(p_true: float, p0: float = 0.01, alpha: float = 0.05, n_max: int = 100_000,
                      trials: int = 20, seed: int = 0) -> list[int | None]:
    """Monte Carlo: tokens needed to certify p < p0 when the true flip rate is p_true."""
    rng = np.random.default_rng(seed)
    out: list[int | None] = []
    for _ in range(trials):
        d = (rng.random(n_max) < p_true).astype(float)
        out.append(certify(d, p0, alpha, with_ucb=False).certified_at)
    return out


def clipped_topk_kl(logp_ref: np.ndarray, logp_q: np.ndarray, k: int = 20, clip: float = 10.0) -> np.ndarray:
    """Per-token KL(ref || q) restricted to the reference top-k, renormalised, clipped.

    Inputs are log-probabilities [T, V] (or already top-k sliced with the same index order).
    """
    idx = np.argsort(-logp_ref, axis=1)[:, :k]
    r = np.take_along_axis(logp_ref, idx, axis=1)
    q = np.take_along_axis(logp_q, idx, axis=1)
    r = r - np.logaddexp.reduce(r, axis=1, keepdims=True)
    q = q - np.logaddexp.reduce(q, axis=1, keepdims=True)
    kl = (np.exp(r) * (r - q)).sum(axis=1)
    return np.clip(kl, 0.0, clip)


def compare_top_n(ref_ids: np.ndarray, ref_lp: np.ndarray, cand_ids: np.ndarray, cand_lp: np.ndarray,
                  clip: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Top-1 disagreement and clipped top-n KL from two runs' stored top-n distributions.

    Inputs are [T, n] ids and log-probs from the reference (exact) and candidate runs over
    the same tokens. KL(ref || cand) is taken over the reference's top-n ids; an id missing
    from the candidate's top-n gets the candidate's smallest stored log-prob, an upper bound
    on its true log-prob, so the KL estimate errs low for those ids. Both distributions are
    renormalised over the reference ids before the KL.

    Returns (flips [T] in {0,1}, kl [T]).
    """
    if ref_ids.shape != cand_ids.shape:
        raise ValueError("reference and candidate top-n shapes differ")
    flips = (ref_ids[:, 0] != cand_ids[:, 0]).astype(float)
    T, n = ref_ids.shape
    floor = cand_lp.min(axis=1)
    q = np.empty_like(ref_lp)
    for t in range(T):
        lookup = dict(zip(cand_ids[t].tolist(), cand_lp[t].tolist()))
        q[t] = [lookup.get(i, floor[t]) for i in ref_ids[t].tolist()]
    r = ref_lp - np.logaddexp.reduce(ref_lp, axis=1, keepdims=True)
    q = q - np.logaddexp.reduce(q, axis=1, keepdims=True)
    kl = np.clip((np.exp(r) * (r - q)).sum(axis=1), 0.0, clip)
    return flips, kl
