"""Per-expert ladder error profile (SPEC-MATH section 6, SPEC-DERISK P0.4).

eps_i(b) = median over inputs x of ||f_i(x) - f_i,b(x)|| / ||x||, where f_i is expert i at
the exact precision and f_i,b the same expert from the low-bit copy. Both GGUF files are
dequantized with gguf-py; inputs are Gaussian, scaled to the typical MoE input norm from a
routing trace. Gaussian inputs are a stand-in for real hidden states, so the profile must be
checked against measured flips (rank correlation of bound and flip, protocol target >= 0.5).

Output: one float per line, layer-major (index il * n_expert + e), for every model layer;
layers without routed experts get 1.0. This is the --eps file of banach-engine.
"""

from __future__ import annotations

import numpy as np


def _silu(x: np.ndarray) -> np.ndarray:
    # x * sigmoid(x) via tanh: no overflow for large |x|
    return x * 0.5 * (1.0 + np.tanh(0.5 * x))


def _gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x**3)))


ACTIVATIONS = {"silu": _silu, "gelu": _gelu}


def expert_tensors(path: str):
    """{layer: {"gate": arr, "up": arr, "down": arr}} dequantized, arrays [n_expert, out, in]."""
    import gguf
    from gguf.quants import dequantize

    r = gguf.GGUFReader(path)
    out: dict[int, dict[str, np.ndarray]] = {}
    n_layer = None
    for f in r.fields.values():
        if f.name.endswith(".block_count"):
            n_layer = int(f.parts[f.data[0]][0])
    for t in r.tensors:
        if not t.name.startswith("blk.") or "_exps." not in t.name or "_shexp" in t.name:
            continue
        il = int(t.name.split(".")[1])
        kind = t.name.split(".")[2]  # ffn_gate_exps / ffn_up_exps / ffn_down_exps / ffn_gate_up_exps
        key = {"ffn_gate_exps": "gate", "ffn_up_exps": "up", "ffn_down_exps": "down",
               "ffn_gate_up_exps": "gate_up"}.get(kind)
        if key is None:
            continue
        out.setdefault(il, {})[key] = dequantize(t.data, t.tensor_type).astype(np.float32)
    return out, n_layer


def expert_forward(w: dict[str, np.ndarray], e: int, x: np.ndarray, act) -> np.ndarray:
    """x [n, d_model] -> [n, d_model] for expert e."""
    if "gate_up" in w:
        gu = x @ w["gate_up"][e].T
        g, u = np.split(gu, 2, axis=-1)
    else:
        g = x @ w["gate"][e].T
        u = x @ w["up"][e].T
    return (act(g) * u) @ w["down"][e].T


def profile(exact_path: str, low_path: str, xnorm: float, n_samples: int = 32, activation: str = "silu",
            seed: int = 0, quantile: float = 0.5) -> tuple[np.ndarray, int, int]:
    """Returns (eps [n_layer, n_expert], n_layer, n_expert)."""
    act = ACTIVATIONS[activation]
    hi, n_layer = expert_tensors(exact_path)
    lo, n_layer_lo = expert_tensors(low_path)
    if set(hi) != set(lo):
        raise ValueError("exact and low files have different MoE layers")
    any_l = next(iter(hi.values()))
    some = any_l.get("up", any_l.get("gate_up"))
    n_expert, d_model = some.shape[0], some.shape[-1]
    n_layer = n_layer or (max(hi) + 1)
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n_samples, d_model)).astype(np.float32)
    x *= xnorm / np.linalg.norm(x, axis=1, keepdims=True)
    eps = np.ones((n_layer, n_expert), dtype=np.float64)
    for il in sorted(hi):
        for k in hi[il]:
            if hi[il][k].shape != lo[il][k].shape:
                raise ValueError(f"layer {il} {k}: shapes differ")
        for e in range(n_expert):
            d = expert_forward(hi[il], e, x, act) - expert_forward(lo[il], e, x, act)
            rel = np.linalg.norm(d, axis=1) / xnorm
            eps[il, e] = float(np.quantile(rel, quantile))
    return eps, n_layer, n_expert


def save(eps: np.ndarray, path: str) -> None:
    np.savetxt(path, eps.reshape(-1), fmt="%.8g")


def profile_real(exact_path: str, low_path: str, trace_json: str, activation: str = "silu",
                 quantile: float = 0.5, min_tokens: int = 4) -> tuple[np.ndarray, int, int, float]:
    """eps from real MoE inputs: a banach-trace run with --dump-x.

    For expert e of layer l the inputs are the calibration tokens routed to e at layer l
    (falling back to all tokens of the layer when fewer than `min_tokens` were). Returns
    (eps [n_layer, n_expert], n_layer, n_expert, share of experts that needed the fallback).
    """
    import json
    from pathlib import Path

    from .trace import Trace

    meta = json.loads(Path(trace_json).read_text())
    n_x, d = int(meta["x_tokens"]), int(meta["n_embd"])
    moe_layers = list(meta["moe_layers"])
    xs = np.fromfile(Path(trace_json).with_suffix(".x.bin"), dtype="<f4")
    if xs.size != n_x * len(moe_layers) * d:
        raise ValueError("x dump size does not match the trace header")
    xs = xs.reshape(n_x, len(moe_layers), d)
    tr = Trace.from_banach_trace(trace_json)
    act = ACTIVATIONS[activation]
    hi, n_layer = expert_tensors(exact_path)
    lo, _ = expert_tensors(low_path)
    any_l = next(iter(hi.values()))
    n_expert = any_l.get("up", any_l.get("gate_up")).shape[0]
    n_layer = n_layer or (max(hi) + 1)
    eps = np.ones((n_layer, n_expert), dtype=np.float64)
    fallback = 0
    for li, il in enumerate(moe_layers):
        x_l = xs[:, li, :]
        routed = tr.topk[:n_x, li, :]
        for e in range(n_expert):
            rows = np.nonzero((routed == e).any(axis=1))[0]
            if len(rows) < min_tokens:
                rows = np.arange(n_x)
                fallback += 1
            x = x_l[rows[:256]]
            nx = np.linalg.norm(x, axis=1)
            dlt = expert_forward(hi[il], e, x, act) - expert_forward(lo[il], e, x, act)
            eps[il, e] = float(np.quantile(np.linalg.norm(dlt, axis=1) / np.maximum(nx, 1e-6), quantile))
    return eps, n_layer, n_expert, fallback / (len(moe_layers) * n_expert)
