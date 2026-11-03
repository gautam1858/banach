"""Command line entry: python -m banach_phase0 <command> ..."""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from . import cachesim, entropy, fidelity, perfmodel
from .model import BPW_GRID, CAPACITY_GB_GRID, DEEPSEEK_V4_FLASH, GB, MODELS
from .trace import Trace, synthetic_trace


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x]


def cmd_synth(a: argparse.Namespace) -> None:
    t = synthetic_trace(
        n_tokens=a.tokens, n_layers=a.layers, n_experts=a.experts, top_k=a.k,
        n_workloads=a.workloads, session_len=a.session_len, reuse_p=a.reuse, seed=a.seed,
    )
    t.save(a.out)
    print(f"wrote {a.out}: {t.n_tokens} tokens x {t.n_layers} layers x top-{t.top_k} of {t.n_experts}")


def cmd_simulate(a: argparse.Namespace) -> None:
    trace = Trace.load(a.trace)
    model = MODELS.get(a.model, DEEPSEEK_V4_FLASH)
    machine = perfmodel.Machine("cli", dram_bw=a.dram_gbs * GB, nvme_bw=a.nvme_gbs * GB)
    rows = []
    for gb in _floats(a.capacity_gb):
        for bpw in _floats(a.bpw):
            ebytes = model.expert_bytes(bpw)
            cap = model.experts_that_fit(gb * GB, bpw)
            if trace.n_experts != model.n_experts or trace.n_layers != model.n_layers:
                # scale capacity to the trace's pool so the fraction cached is preserved
                frac = cap / (model.n_experts * model.n_layers)
                cap = int(frac * trace.n_experts * trace.n_layers)
            for scope in a.scope.split(","):
                for pol in a.policy.split(","):
                    r = cachesim.simulate(trace, pol, cap, scope)
                    d = r.as_dict(ebytes)
                    pred = perfmodel.predict(model, machine, bpw, r.hit_rate)
                    d.update(capacity_gb=gb, bpw=bpw, predicted_tok_s=round(pred.tok_s, 3))
                    rows.append(d)
                    if not a.json:
                        print(f"{gb:>5.0f}GB {bpw:>5.2f}b {scope:<9} {pol:<10} "
                              f"cap={cap:<6} hit={r.hit_rate:6.3f} "
                              f"miss/tok={d['miss_bytes_per_token'] / 1e9:6.3f}GB "
                              f"pred={pred.tok_s:6.2f} tok/s")
    if a.json:
        json.dump(rows, sys.stdout, indent=1)
        print()


def cmd_entropy(a: argparse.Namespace) -> None:
    trace = Trace.load(a.trace)
    rows = entropy.layer_report(trace, a.cache_per_layer)
    if a.json:
        json.dump(rows, sys.stdout, indent=1)
        print()
        return
    for r in rows:
        print(" ".join(f"{k}={v}" for k, v in r.items()))


def cmd_certify(a: argparse.Namespace) -> None:
    d = np.loadtxt(a.flips, dtype=float).ravel()
    c = fidelity.certify(d, a.p0, a.alpha)
    json.dump(c.as_dict(), sys.stdout, indent=1)
    print()


def cmd_import(a: argparse.Namespace) -> None:
    hl = tuple(int(x) for x in a.hash_layers.split(",") if x)
    t = Trace.from_banach_trace(a.trace_json, hl)
    t.save(a.out)
    print(f"wrote {a.out}: {t.n_tokens} tokens, {t.n_layers} MoE layers, top-{t.top_k} of {t.n_experts}, "
          f"workloads {t.workload_names}")


def cmd_flips(a: argparse.Namespace) -> None:
    ref, cand = Trace.load(a.reference), Trace.load(a.candidate)
    for name, t in (("reference", ref), ("candidate", cand)):
        if t.top_ids is None:
            raise SystemExit(f"{name} trace has no stored next-token distribution")
    if ref.tokens is None or cand.tokens is None or not np.array_equal(ref.tokens, cand.tokens):
        raise SystemExit("traces are not over the same token sequence")
    flips, kl = fidelity.compare_top_n(ref.top_ids, ref.top_logprobs, cand.top_ids, cand.top_logprobs)
    groups = [("all", np.ones(len(flips), bool))]
    if a.by_workload:
        groups += [(n, ref.workload == i) for i, n in enumerate(ref.workload_names)]
    rows = []
    for name, m in groups:
        c = fidelity.certify(flips[m], a.p0, a.alpha)
        rows.append({"group": name, **c.as_dict(),
                     "kl_mean": round(float(kl[m].mean()), 5),
                     "kl_p99": round(float(np.quantile(kl[m], 0.99)), 5)})
    json.dump(rows, sys.stdout, indent=1)
    print()
    if a.save_flips:
        np.savetxt(a.save_flips, flips, fmt="%d")


def cmd_usage(a: argparse.Namespace) -> None:
    """Per-layer expert usage (mean selections per token) for `banach plan --usage`."""
    t = Trace.load(a.trace)
    if a.workload:
        if a.workload not in t.workload_names:
            raise SystemExit(f"unknown workload {a.workload!r}; have {t.workload_names}")
        t = t.subset(t.workload == t.workload_names.index(a.workload))
    counts = np.zeros((t.n_layers, t.n_experts))
    for l in range(t.n_layers):
        counts[l] = np.bincount(t.topk[:, l, :].ravel(), minlength=t.n_experts)
    usage = counts / max(t.n_tokens, 1)
    out = {"model": t.model, "tokens": int(t.n_tokens), "workloads": t.workload_names if not a.workload else [a.workload],
           "usage": [[round(float(x), 6) for x in row] for row in usage]}
    with open(a.out, "w") as f:
        json.dump(out, f)
    print(f"wrote {a.out}: {t.n_layers} layers x {t.n_experts} experts from {t.n_tokens} tokens")


def cmd_predict(a: argparse.Namespace) -> None:
    model = MODELS[a.model]
    machine = perfmodel.Machine("cli", dram_bw=a.dram_gbs * GB, nvme_bw=a.nvme_gbs * GB,
                                overlap=a.overlap)
    for h in _floats(a.hit):
        p = perfmodel.predict(model, machine, a.bpw, h, alpha=a.alpha, gamma=a.gamma)
        print(json.dumps({"hit_rate": h, **p.as_dict()}))
    need = perfmodel.required_hit_rate(model, machine, a.bpw, a.target, alpha=a.alpha, gamma=a.gamma)
    print(json.dumps({"target_tok_s": a.target, "required_hit_rate": need}))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="banach_phase0")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synth", help="write a synthetic trace (pipeline testing only)")
    s.add_argument("--out", required=True)
    s.add_argument("--tokens", type=int, default=4000)
    s.add_argument("--layers", type=int, default=8)
    s.add_argument("--experts", type=int, default=64)
    s.add_argument("--k", type=int, default=4)
    s.add_argument("--workloads", type=int, default=2)
    s.add_argument("--session-len", type=int, default=500)
    s.add_argument("--reuse", type=float, default=0.5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_synth)

    s = sub.add_parser("simulate", help="cache simulation grid (P0.2)")
    s.add_argument("trace")
    s.add_argument("--model", default=DEEPSEEK_V4_FLASH.name)
    s.add_argument("--capacity-gb", default=",".join(map(str, CAPACITY_GB_GRID)))
    s.add_argument("--bpw", default=",".join(map(str, BPW_GRID)))
    s.add_argument("--policy", default="lru,lfu:256,static,gdsf,belady")
    s.add_argument("--scope", default="per_layer,global")
    s.add_argument("--dram-gbs", type=float, default=70.0)
    s.add_argument("--nvme-gbs", type=float, default=6.0)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("entropy", help="conditional routing entropy and Fano floor (P0.3)")
    s.add_argument("trace")
    s.add_argument("--cache-per-layer", type=int, default=64)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_entropy)

    s = sub.add_parser("certify", help="betting certificate on a 0/1 flip file (P0.5)")
    s.add_argument("flips", help="text file of 0/1 top-1 disagreements, one per token")
    s.add_argument("--p0", type=float, default=0.01)
    s.add_argument("--alpha", type=float, default=0.05)
    s.set_defaults(fn=cmd_certify)

    s = sub.add_parser("import-trace", help="convert a banach-trace .json/.bin pair to .npz (P0.1)")
    s.add_argument("trace_json")
    s.add_argument("--out", required=True)
    s.add_argument("--hash-layers", default="", help="comma-separated MoE layer indices routed by hash")
    s.set_defaults(fn=cmd_import)

    s = sub.add_parser("flips", help="top-1 disagreement + KL between two traces, certified (P0.5)")
    s.add_argument("reference")
    s.add_argument("candidate")
    s.add_argument("--p0", type=float, default=0.01)
    s.add_argument("--alpha", type=float, default=0.05)
    s.add_argument("--by-workload", action="store_true")
    s.add_argument("--save-flips", help="write the 0/1 flip sequence to this file")
    s.set_defaults(fn=cmd_flips)

    s = sub.add_parser("usage", help="export per-layer expert usage for `banach plan --usage`")
    s.add_argument("trace")
    s.add_argument("--out", required=True)
    s.add_argument("--workload", help="only this workload")
    s.set_defaults(fn=cmd_usage)

    s = sub.add_parser("predict", help="bandwidth model tok/s (P0.6)")
    s.add_argument("--model", default=DEEPSEEK_V4_FLASH.name, choices=sorted(MODELS))
    s.add_argument("--bpw", type=float, default=2.06)
    s.add_argument("--hit", default="0.8,0.9,0.95,1.0")
    s.add_argument("--dram-gbs", type=float, default=70.0)
    s.add_argument("--nvme-gbs", type=float, default=6.0)
    s.add_argument("--overlap", type=float, default=0.8)
    s.add_argument("--alpha", type=float, default=0.0)
    s.add_argument("--gamma", type=int, default=0)
    s.add_argument("--target", type=float, default=10.0)
    s.set_defaults(fn=cmd_predict)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
