"""Routing-trace format and a synthetic generator.

A trace is a single ``.npz`` file:

    topk      int16   [T, L, k]   routed expert ids, per token, per layer
    gates     float32 [T, L, k]   gate weights after normalisation (route_scale applied)
    xnorm     float32 [T, L]      ||x|| of the MoE input (post hyper-connection)
    onorm     float32 [T, L, k]   ||f_i(x)|| per routed expert (optional, zeros if absent)
    session   int32   [T]         session id; sessions are the unit of warm/score splits
    workload  int16   [T]         index into ``workload_names``
    meta      json string         model name, n_experts, top_k, hash_layers, workload_names

The real tracer (llama.cpp cb_eval, SPEC-DERISK P0.1) writes this format. The synthetic
generator exists so the simulator and analyses can be exercised before real traces exist;
its numbers say nothing about V4 Flash.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trace:
    topk: np.ndarray
    gates: np.ndarray
    xnorm: np.ndarray
    session: np.ndarray
    workload: np.ndarray
    n_experts: int
    model: str = "unknown"
    hash_layers: tuple[int, ...] = ()
    workload_names: list[str] = field(default_factory=list)
    onorm: np.ndarray | None = None
    tokens: np.ndarray | None = None        # [T] input token ids (real traces)
    top_ids: np.ndarray | None = None       # [T, n] next-token top-n ids
    top_logprobs: np.ndarray | None = None  # [T, n] their log-probabilities

    def __post_init__(self) -> None:
        T, L, k = self.topk.shape
        if self.gates.shape != (T, L, k):
            raise ValueError(f"gates shape {self.gates.shape} != topk shape {self.topk.shape}")
        if self.xnorm.shape != (T, L):
            raise ValueError(f"xnorm shape {self.xnorm.shape} != {(T, L)}")
        if self.session.shape != (T,) or self.workload.shape != (T,):
            raise ValueError("session and workload must have shape [T]")
        if self.topk.size and (self.topk.min() < 0 or self.topk.max() >= self.n_experts):
            raise ValueError("expert id out of range")
        if self.onorm is not None and self.onorm.shape != (T, L, k):
            raise ValueError("onorm shape mismatch")
        if self.tokens is not None and self.tokens.shape != (T,):
            raise ValueError("tokens shape mismatch")
        if (self.top_ids is None) != (self.top_logprobs is None):
            raise ValueError("top_ids and top_logprobs go together")
        if self.top_ids is not None and (self.top_ids.shape[0] != T or self.top_ids.shape != self.top_logprobs.shape):
            raise ValueError("top_ids/top_logprobs shape mismatch")

    @property
    def n_tokens(self) -> int:
        return self.topk.shape[0]

    @property
    def n_layers(self) -> int:
        return self.topk.shape[1]

    @property
    def top_k(self) -> int:
        return self.topk.shape[2]

    def subset(self, mask: np.ndarray) -> "Trace":
        return Trace(
            topk=self.topk[mask],
            gates=self.gates[mask],
            xnorm=self.xnorm[mask],
            session=self.session[mask],
            workload=self.workload[mask],
            n_experts=self.n_experts,
            model=self.model,
            hash_layers=self.hash_layers,
            workload_names=list(self.workload_names),
            onorm=None if self.onorm is None else self.onorm[mask],
            tokens=None if self.tokens is None else self.tokens[mask],
            top_ids=None if self.top_ids is None else self.top_ids[mask],
            top_logprobs=None if self.top_logprobs is None else self.top_logprobs[mask],
        )

    def split_by_session(self, warm_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks (warm, score), splitting whole sessions in order of first appearance.

        Splits within each workload so every workload contributes to both halves.
        """
        warm = np.zeros(self.n_tokens, dtype=bool)
        for w in np.unique(self.workload):
            in_w = self.workload == w
            _, first = np.unique(self.session[in_w], return_index=True)
            sessions = self.session[in_w][np.sort(first)]
            n_warm = int(round(len(sessions) * warm_frac))
            if len(sessions) > 1:
                n_warm = min(max(n_warm, 1), len(sessions) - 1)
            warm |= in_w & np.isin(self.session, sessions[:n_warm])
        return warm, ~warm

    def save(self, path: str | Path) -> None:
        meta = {
            "model": self.model,
            "n_experts": self.n_experts,
            "hash_layers": list(self.hash_layers),
            "workload_names": self.workload_names,
        }
        arrays = dict(
            topk=self.topk.astype(np.int16),
            gates=self.gates.astype(np.float32),
            xnorm=self.xnorm.astype(np.float32),
            session=self.session.astype(np.int32),
            workload=self.workload.astype(np.int16),
            meta=np.array(json.dumps(meta)),
        )
        if self.onorm is not None:
            arrays["onorm"] = self.onorm.astype(np.float32)
        if self.tokens is not None:
            arrays["tokens"] = self.tokens.astype(np.int32)
        if self.top_ids is not None:
            arrays["top_ids"] = self.top_ids.astype(np.int32)
            arrays["top_logprobs"] = self.top_logprobs.astype(np.float32)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "Trace":
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            return cls(
                topk=z["topk"].astype(np.int64),
                gates=z["gates"],
                xnorm=z["xnorm"],
                session=z["session"],
                workload=z["workload"],
                n_experts=int(meta["n_experts"]),
                model=meta.get("model", "unknown"),
                hash_layers=tuple(meta.get("hash_layers", ())),
                workload_names=list(meta.get("workload_names", [])),
                onorm=z["onorm"] if "onorm" in z.files else None,
                tokens=z["tokens"] if "tokens" in z.files else None,
                top_ids=z["top_ids"] if "top_ids" in z.files else None,
                top_logprobs=z["top_logprobs"] if "top_logprobs" in z.files else None,
            )

    @classmethod
    def from_banach_trace(cls, json_path: str | Path, hash_layers: tuple[int, ...] = ()) -> "Trace":
        """Load the .json/.bin pair written by tracer/banach-trace.

        ``hash_layers`` are indices into the MoE layer list (not llama layer ids).
        """
        json_path = Path(json_path)
        meta = json.loads(json_path.read_text())
        if meta.get("format") != "banach-trace-v1":
            raise ValueError(f"unsupported trace format {meta.get('format')!r}")
        r = meta["record"]
        L, k, n = len(meta["moe_layers"]), int(meta["top_k"]), int(r["ntop"])
        n_experts = int(meta.get("n_experts", -1))
        dt = np.dtype({
            "names": ["token", "pos", "session", "workload", "topk", "gates", "xnorm", "onorm",
                      "top_ids", "top_logprobs"],
            "formats": ["<i4", "<i4", "<i4", "<i4", ("<i2", (L, k)), ("<f4", (L, k)), ("<f4", (L,)),
                        ("<f4", (L, k)), ("<i4", (n,)), ("<f4", (n,))],
            "offsets": [r["token"], r["pos"], r["session"], r["workload"], r["topk"], r["gates"],
                        r["xnorm"], r["onorm"], r["top_ids"], r["top_logprobs"]],
            "itemsize": r["size"],
        })
        bin_path = json_path.with_suffix(".bin")
        size = bin_path.stat().st_size
        if size % dt.itemsize:
            raise ValueError(f"{bin_path} size {size} is not a multiple of the record size {dt.itemsize}")
        rec = np.fromfile(bin_path, dtype=dt)
        if len(rec) != int(meta["n_tokens"]):
            raise ValueError(f"{bin_path}: {len(rec)} records, header says {meta['n_tokens']}")
        topk = rec["topk"].astype(np.int64)
        if n_experts <= 0:
            n_experts = int(topk.max()) + 1
        return cls(
            topk=topk,
            gates=rec["gates"].astype(np.float32),
            xnorm=rec["xnorm"].astype(np.float32),
            session=rec["session"].astype(np.int32),
            workload=rec["workload"].astype(np.int16),
            n_experts=n_experts,
            model=meta.get("model_desc", "unknown"),
            hash_layers=tuple(hash_layers),
            workload_names=list(meta.get("workload_names", [])),
            onorm=rec["onorm"].astype(np.float32),
            tokens=rec["token"].astype(np.int32),
            top_ids=rec["top_ids"].astype(np.int32) if n else None,
            top_logprobs=rec["top_logprobs"].astype(np.float32) if n else None,
        )


def synthetic_trace(
    n_tokens: int = 4000,
    n_layers: int = 8,
    n_experts: int = 64,
    top_k: int = 4,
    n_workloads: int = 2,
    session_len: int = 500,
    zipf_s: float = 1.1,
    reuse_p: float = 0.5,
    hash_layers: tuple[int, ...] = (0,),
    vocab: int = 5000,
    seed: int = 0,
) -> Trace:
    """Generate a trace with Zipf expert popularity, per-workload preferences and recency.

    Each workload has its own popularity permutation per layer (so off-domain static hot sets
    transfer poorly), and each slot reuses an expert from the previous token's set at that layer
    with probability ``reuse_p`` (temporal locality). Hash layers route by token id.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_experts + 1, dtype=float)
    base = ranks**-zipf_s
    base /= base.sum()
    # popularity[w, l, e]
    pop = np.empty((n_workloads, n_layers, n_experts))
    for w in range(n_workloads):
        for l in range(n_layers):
            pop[w, l] = base[rng.permutation(n_experts)]
    tok_p = np.arange(1, vocab + 1, dtype=float) ** -1.0
    tok_p /= tok_p.sum()
    hash_table = rng.integers(0, n_experts, size=(n_layers, vocab, top_k))

    topk = np.empty((n_tokens, n_layers, top_k), dtype=np.int64)
    gates = np.empty((n_tokens, n_layers, top_k), dtype=np.float32)
    session = np.empty(n_tokens, dtype=np.int32)
    workload = np.empty(n_tokens, dtype=np.int16)
    n_sessions = max(1, -(-n_tokens // session_len))
    sess_workload = rng.integers(0, n_workloads, size=n_sessions)
    prev = None
    for t in range(n_tokens):
        s = t // session_len
        w = int(sess_workload[s])
        session[t] = s
        workload[t] = w
        if t % session_len == 0:
            prev = None
        tok = rng.choice(vocab, p=tok_p)
        for l in range(n_layers):
            if l in hash_layers:
                chosen = _unique_fill(list(hash_table[l, tok]), pop[w, l], top_k, rng)
            else:
                chosen: list[int] = []
                if prev is not None:
                    for e in prev[l]:
                        if rng.random() < reuse_p and e not in chosen:
                            chosen.append(int(e))
                chosen = _unique_fill(chosen, pop[w, l], top_k, rng)
            topk[t, l] = chosen
            g = rng.dirichlet(np.full(top_k, 0.7))
            gates[t, l] = np.sort(g)[::-1]
        prev = topk[t]
    xnorm = rng.lognormal(mean=3.0, sigma=0.3, size=(n_tokens, n_layers)).astype(np.float32)
    onorm = rng.lognormal(mean=1.0, sigma=0.5, size=(n_tokens, n_layers, top_k)).astype(np.float32)
    return Trace(
        topk=topk,
        gates=gates,
        xnorm=xnorm,
        session=session,
        workload=workload,
        n_experts=n_experts,
        model="synthetic",
        hash_layers=tuple(hash_layers),
        workload_names=[f"w{i}" for i in range(n_workloads)],
        onorm=onorm,
    )


def _unique_fill(chosen: list[int], p: np.ndarray, k: int, rng: np.random.Generator) -> list[int]:
    out: list[int] = []
    for e in chosen:
        if int(e) not in out:
            out.append(int(e))
        if len(out) == k:
            return out
    while len(out) < k:
        e = int(rng.choice(len(p), p=p))
        if e not in out:
            out.append(e)
    return out
