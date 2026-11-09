#!/usr/bin/env python3
"""Stock llama.cpp vs banach, alternating runs, with the page cache primed the same way each time.

    python3 bench/interleaved.py --engine build-engine/banach-engine --model model.gguf \
        --prompt prompt.txt --budget-mb 10000 --reps 5

Why this exists: one stock run and one banach run say little on a noisy machine (stock llama.cpp
alone varied from 3.4 to 4.9 tok/s between runs on the 22 GB test). This script runs
stock/banach pairs back to back, swaps which goes first on every repetition, reads the model
file sequentially before each run so both start from the same page-cache state, and reports
medians and ranges. Run it on a machine whose disk is healthy and quiet: it prints the disk
read speed it saw while priming so a throttled disk is visible in the results.

Output: a table on stdout and the raw runs as JSON (--json-out).
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import time

STOCK = re.compile(r"reference \(stock llama\.cpp\): prefill (\d+) tok ([\d.]+) s, decode (\d+) tok ([\d.]+) tok/s")
BANACH = re.compile(r"(banach|naive): prefill (\d+) tok ([\d.]+) s; decode (\d+) tok ([\d.]+) tok/s, hit ([\d.]+)%")


def prime(path):
    """Read the whole file sequentially; returns MB/s so a slow disk shows up in the output."""
    t = time.time()
    n = 0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(8 << 20)
            if not b:
                break
            n += len(b)
    return n / 1e6 / max(time.time() - t, 1e-9)


def run(engine, model, mode, args):
    cmd = [engine, "-m", model, "-f", args.prompt, "-n", str(args.tokens), "-c", str(args.ctx)]
    if args.threads:
        cmd += ["-t", str(args.threads)]
    if mode == "stock":
        cmd += ["--mode", "off"]
    else:
        cmd += ["--mode", mode, "--budget-mb", str(args.budget_mb), "--warm"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = STOCK.search(p.stderr)
    if m:
        return {"mode": "stock", "prefill_s": float(m[2]), "tok_s": float(m[4])}
    m = BANACH.search(p.stderr)
    if m:
        return {"mode": m[1], "prefill_s": float(m[3]), "tok_s": float(m[5]), "hit": float(m[6])}
    return {"mode": mode, "error": p.stderr[-300:]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--budget-mb", type=int, default=0, help="expert RAM budget for banach (0: unlimited)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--modes", default="stock,banach", help="comma list from stock, naive, banach")
    ap.add_argument("--json-out", default="interleaved.json")
    args = ap.parse_args()

    modes = args.modes.split(",")
    runs = []
    for rep in range(args.reps):
        order = modes if rep % 2 == 0 else modes[::-1]
        for mode in order:
            speed = prime(args.model)
            r = run(args.engine, args.model, mode, args)
            r.update(rep=rep, prime_mb_s=round(speed))
            runs.append(r)
            print(r, file=sys.stderr, flush=True)
    json.dump(runs, open(args.json_out, "w"), indent=1)

    print(f"\n{args.reps} repetitions, alternating order, page cache primed by a sequential read each time")
    print(f"{'mode':8} {'prefill s (median, min-max)':32} {'decode tok/s (median, min-max)':32} disk MB/s while priming")
    for mode in modes:
        rs = [r for r in runs if r["mode"] == mode and "error" not in r]
        if not rs:
            print(f"{mode:8} no successful runs")
            continue
        pf = [r["prefill_s"] for r in rs]
        dc = [r["tok_s"] for r in rs]
        pm = [r["prime_mb_s"] for r in rs]
        print(f"{mode:8} {statistics.median(pf):6.2f}  ({min(pf):.2f} - {max(pf):.2f}){'':10} "
              f"{statistics.median(dc):6.2f}  ({min(dc):.2f} - {max(dc):.2f}){'':10} {statistics.median(pm)}")


if __name__ == "__main__":
    main()
