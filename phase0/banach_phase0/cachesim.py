"""Expert-cache simulator (SPEC-DERISK P0.2).

Replays the warm sessions, then the scored sessions, through a cache of ``capacity`` experts
and counts hits only on the scored part. All experts in a run have the same size (one bit
width per run), so capacity is expressed in experts and Belady with bypass is optimal.

Policies
    lru          least recently used
    lfu:<h>      decayed frequency with half-life h tokens (e.g. lfu:256)
    static       top-capacity experts by frequency on the warm part, never changes
    gdsf         GreedyDual-Size-Frequency with frequency-based admission (TinyLFU style):
                 a missed expert enters only if its decayed frequency beats the victim's
    belady       offline optimum (furthest next use, bypass allowed)

Scopes
    per_layer    capacity split evenly over layers, independent cache per layer
    global       one cache shared by all layers
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from .trace import Trace


@dataclass
class SimResult:
    policy: str
    scope: str
    capacity: int
    accesses: int
    hits: int
    tokens: int
    layer_hits: np.ndarray
    layer_accesses: np.ndarray

    @property
    def hit_rate(self) -> float:
        return self.hits / self.accesses if self.accesses else 0.0

    @property
    def misses_per_token(self) -> float:
        return (self.accesses - self.hits) / self.tokens if self.tokens else 0.0

    def miss_bytes_per_token(self, expert_bytes: float) -> float:
        return self.misses_per_token * expert_bytes

    def as_dict(self, expert_bytes: float | None = None) -> dict:
        d = {
            "policy": self.policy,
            "scope": self.scope,
            "capacity_experts": self.capacity,
            "tokens": self.tokens,
            "accesses": self.accesses,
            "hits": self.hits,
            "hit_rate": round(self.hit_rate, 6),
            "misses_per_token": round(self.misses_per_token, 4),
        }
        if expert_bytes is not None:
            d["miss_bytes_per_token"] = self.miss_bytes_per_token(expert_bytes)
        return d


# --------------------------------------------------------------------------- policies


class _Policy:
    """A cache over integer keys. ``access`` returns True on hit and handles admission."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity

    def access(self, key: int, t: int, i: int) -> bool:  # t: token index, i: access index
        raise NotImplementedError


class LRU(_Policy):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.d: dict[int, None] = {}

    def access(self, key: int, t: int, i: int) -> bool:
        d = self.d
        if key in d:
            del d[key]
            d[key] = None
            return True
        if self.capacity <= 0:
            return False
        if len(d) >= self.capacity:
            del d[next(iter(d))]
        d[key] = None
        return False


class _Decay:
    """Exponentially decayed counts using growing increments 2^(t/h), renormalised lazily."""

    def __init__(self, half_life: float) -> None:
        self.h = float(half_life)
        self.offset = 0.0  # tokens folded into the stored values
        self.c: dict[int, float] = {}

    def inc(self, key: int, t: int) -> float:
        e = (t - self.offset) / self.h
        if e > 600:
            self._rescale(t)
            e = 0.0
        v = self.c.get(key, 0.0) + 2.0**e
        self.c[key] = v
        return v

    def get(self, key: int) -> float:
        return self.c.get(key, 0.0)

    def _rescale(self, t: int) -> None:
        f = 2.0 ** (-(t - self.offset) / self.h)
        self.c = {k: v * f for k, v in self.c.items() if v * f > 1e-300}
        self.offset = float(t)

    @property
    def generation(self) -> float:
        return self.offset


class _HeapCache(_Policy):
    """Cache that evicts the minimum priority; priorities live in a lazy heap."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.prio: dict[int, float] = {}
        self.heap: list[tuple[float, int]] = []

    def _set(self, key: int, p: float) -> None:
        self.prio[key] = p
        heapq.heappush(self.heap, (p, key))
        if len(self.heap) > 4 * max(self.capacity, 16):
            self.heap = [(v, k) for k, v in self.prio.items()]
            heapq.heapify(self.heap)

    def _peek_min(self) -> tuple[float, int]:
        h = self.heap
        while True:
            p, k = h[0]
            if self.prio.get(k) == p:
                return p, k
            heapq.heappop(h)

    def _evict(self, key: int) -> None:
        del self.prio[key]


class DecayedLFU(_HeapCache):
    def __init__(self, capacity: int, half_life: float) -> None:
        super().__init__(capacity)
        self.freq = _Decay(half_life)
        self.gen = 0.0

    def access(self, key: int, t: int, i: int) -> bool:
        f = self.freq.inc(key, t)
        if self.freq.generation != self.gen:  # counts were rescaled: rebuild priorities
            self.gen = self.freq.generation
            self.prio = {k: self.freq.get(k) for k in self.prio}
            self.heap = [(v, k) for k, v in self.prio.items()]
            heapq.heapify(self.heap)
            f = self.freq.get(key)
        if key in self.prio:
            self._set(key, f)
            return True
        if self.capacity <= 0:
            return False
        if len(self.prio) >= self.capacity:
            _, victim = self._peek_min()
            self._evict(victim)
        self._set(key, f)
        return False


class GDSF(_HeapCache):
    """GreedyDual-Size-Frequency with uniform size/cost and frequency-gated admission."""

    def __init__(self, capacity: int, half_life: float = 1024.0) -> None:
        super().__init__(capacity)
        self.freq = _Decay(half_life)
        self.clock = 0.0
        self.gen = 0.0

    def _norm_freq(self, key: int, t: int) -> float:
        # frequency in units of "recent accesses", comparable across rescales
        return self.freq.get(key) * 2.0 ** (-(t - self.freq.offset) / self.freq.h)

    def access(self, key: int, t: int, i: int) -> bool:
        self.freq.inc(key, t)
        f = self._norm_freq(key, t)
        if key in self.prio:
            self._set(key, self.clock + f)
            return True
        if self.capacity <= 0:
            return False
        if len(self.prio) >= self.capacity:
            p, victim = self._peek_min()
            if f <= self._norm_freq(victim, t):
                return False  # admission denied: stream it, keep the cache
            self.clock = p
            self._evict(victim)
        self._set(key, self.clock + f)
        return False


class Belady(_HeapCache):
    """Furthest-next-use eviction with bypass. Needs next-use indices for the stream."""

    def __init__(self, capacity: int, next_use: np.ndarray) -> None:
        super().__init__(capacity)
        self.next_use = next_use

    def access(self, key: int, t: int, i: int) -> bool:
        nu = float(self.next_use[i])
        if key in self.prio:
            self._set(key, -nu)
            return True
        if self.capacity <= 0:
            return False
        if len(self.prio) >= self.capacity:
            p, victim = self._peek_min()  # most negative = furthest next use
            if nu >= -p:
                return False  # incoming is needed later than everything cached: bypass
            self._evict(victim)
        self._set(key, -nu)
        return False


class Static(_Policy):
    def __init__(self, capacity: int, hot: set[int]) -> None:
        super().__init__(capacity)
        self.hot = hot

    def access(self, key: int, t: int, i: int) -> bool:
        return key in self.hot


# --------------------------------------------------------------------------- driver


def _stream(trace: Trace, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flattened (key, layer, token_index) for the masked tokens, in token/layer/slot order."""
    topk = trace.topk[mask]
    T, L, k = topk.shape
    layers = np.broadcast_to(np.arange(L)[None, :, None], (T, L, k))
    keys = layers.astype(np.int64) * trace.n_experts + topk.astype(np.int64)
    toks = np.broadcast_to(np.arange(T)[:, None, None], (T, L, k))
    return keys.reshape(-1), layers.reshape(-1).astype(np.int64), toks.reshape(-1).astype(np.int64)


def next_use_indices(keys: np.ndarray) -> np.ndarray:
    """For each position, the index of the next access to the same key (inf if none)."""
    n = len(keys)
    out = np.full(n, np.inf)
    order = np.lexsort((np.arange(n), keys))
    sk = keys[order]
    same = sk[1:] == sk[:-1]
    out[order[:-1][same]] = order[1:][same]
    return out


def _split_capacity(capacity: int, n: int) -> list[int]:
    base, extra = divmod(capacity, n)
    return [base + (1 if i < extra else 0) for i in range(n)]


def _make(policy: str, capacity: int, **kw) -> _Policy:
    if policy == "lru":
        return LRU(capacity)
    if policy.startswith("lfu"):
        h = float(policy.split(":")[1]) if ":" in policy else 256.0
        return DecayedLFU(capacity, h)
    if policy.startswith("gdsf"):
        h = float(policy.split(":")[1]) if ":" in policy else 1024.0
        return GDSF(capacity, h)
    if policy == "belady":
        return Belady(capacity, kw["next_use"])
    if policy == "static":
        return Static(capacity, kw["hot"])
    raise ValueError(f"unknown policy {policy!r}")


def simulate(
    trace: Trace,
    policy: str,
    capacity: int,
    scope: str = "per_layer",
    warm_mask: np.ndarray | None = None,
    score_mask: np.ndarray | None = None,
) -> SimResult:
    """Run one policy. Default split is 60/40 by session within each workload."""
    if scope not in ("per_layer", "global"):
        raise ValueError("scope must be per_layer or global")
    if warm_mask is None or score_mask is None:
        warm_mask, score_mask = trace.split_by_session(0.6)
    L = trace.n_layers

    wk, wl, wt = _stream(trace, warm_mask)
    sk, sl, st = _stream(trace, score_mask)
    n_warm_tok = int(warm_mask.sum())
    keys = np.concatenate([wk, sk])
    layers = np.concatenate([wl, sl])
    toks = np.concatenate([wt, st + n_warm_tok])
    n_warm = len(wk)

    caps = _split_capacity(capacity, L) if scope == "per_layer" else [capacity]

    kw: dict = {}
    if policy == "static":
        counts = np.bincount(wk, minlength=L * trace.n_experts)
        hot: set[int] = set()
        if scope == "global":
            hot = set(np.argsort(-counts, kind="stable")[:capacity].tolist())
        else:
            for l in range(L):
                c = counts[l * trace.n_experts : (l + 1) * trace.n_experts]
                top = np.argsort(-c, kind="stable")[: caps[l]]
                hot.update((l * trace.n_experts + top).tolist())
        kw["hot"] = hot

    if policy == "belady":
        # Keys already encode the layer, so one next-use table serves both scopes:
        # within a layer's cache only the relative order of next uses matters.
        kw["next_use"] = next_use_indices(keys)

    caches = [_make(policy, c, **kw) for c in caps]

    layer_hits = np.zeros(L, dtype=np.int64)
    layer_acc = np.zeros(L, dtype=np.int64)
    keys_l = keys.tolist()
    layers_l = layers.tolist()
    toks_l = toks.tolist()
    per_layer = scope == "per_layer"
    for i in range(len(keys_l)):
        l = layers_l[i]
        c = caches[l] if per_layer else caches[0]
        hit = c.access(keys_l[i], toks_l[i], i)
        if i >= n_warm:
            layer_acc[l] += 1
            if hit:
                layer_hits[l] += 1

    return SimResult(
        policy=policy,
        scope=scope,
        capacity=capacity,
        accesses=int(layer_acc.sum()),
        hits=int(layer_hits.sum()),
        tokens=int(score_mask.sum()),
        layer_hits=layer_hits,
        layer_accesses=layer_acc,
    )


def transfer(
    trace: Trace, policy: str, capacity: int, train_workload: int, test_workload: int,
    scope: str = "per_layer",
) -> tuple[SimResult, SimResult, float]:
    """Cross-workload transfer: warm on one workload, score on another.

    Returns (off-domain result, in-domain result, ratio off/in). The go/no-go check wants the
    ratio >= 0.8 (SPEC-DERISK section 6).
    """
    tr_warm, _ = trace.split_by_session(0.6)
    _, te_score = trace.split_by_session(0.6)
    warm_off = tr_warm & (trace.workload == train_workload)
    warm_in = tr_warm & (trace.workload == test_workload)
    score = te_score & (trace.workload == test_workload)
    off = simulate(trace, policy, capacity, scope, warm_off, score)
    inn = simulate(trace, policy, capacity, scope, warm_in, score)
    ratio = off.hit_rate / inn.hit_rate if inn.hit_rate > 0 else math.nan
    return off, inn, ratio
