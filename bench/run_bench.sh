#!/usr/bin/env bash
# Reproducible benchmark (SPEC-DERISK section 7, excellence checklist): environment, model
# hash, planner prediction vs measured decode, streaming at a RAM budget, byte-exact check.
# Writes one JSON file per run.
#
# Usage: run_bench.sh MODEL.gguf OUT.json [BUDGET_MB] [PROMPT_FILE]
# Needs: banach (cargo build --release), banach-engine (engine/), python3.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 MODEL.gguf OUT.json [BUDGET_MB] [PROMPT_FILE]" >&2
  exit 2
fi
model=$1; out=$2; budget=${3:-0}; prompt=${4:-}
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
banach=${BANACH:-$root/target/release/banach}
engine=${BANACH_ENGINE:-banach-engine}
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

if [ -z "$prompt" ]; then
  prompt=$tmp/prompt.txt
  head -c 6000 "$root/src/planner.rs" > "$prompt"
fi

"$root/phase0/scripts/record_env.sh" --model "$model" > "$tmp/env.json"
"$banach" probe --disk "$model" > "$tmp/probe.txt" 2>&1 || true
ctx=2048
"$banach" plan "$model" --ctx "$ctx" --json > "$tmp/plan.json"

runs=5
for i in $(seq 1 $runs); do
  "$engine" -m "$model" -f "$prompt" -n 48 -c 4096 --mode off --json "$tmp/ref_$i.json" 2>/dev/null
done
if [ "$budget" -gt 0 ]; then
  for i in $(seq 1 $runs); do
    "$engine" -m "$model" -f "$prompt" -n 48 -c 4096 --budget-mb "$budget" --no-residency \
      --json "$tmp/stream_$i.json" 2>/dev/null
  done
  "$engine" -m "$model" -f "$prompt" -n 24 -c 4096 --budget-mb "$budget" --exact-check --no-residency \
    --json "$tmp/exact.json" 2>/dev/null || true
fi

python3 - "$tmp" "$out" "$model" "$budget" "$runs" "$(git -C "$root" rev-parse HEAD)" <<'EOF'
import json, statistics, sys, pathlib
tmp, out, model, budget, runs, commit = sys.argv[1:7]
t = pathlib.Path(tmp)
def stats(xs):
    xs = sorted(xs)
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]
    return {"median": statistics.median(xs), "iqr": [q(0.25), q(0.75)], "n": len(xs)}
ref = [json.loads((t / f"ref_{i}.json").read_text())["reference_decode"]["tok_s"] for i in range(1, int(runs) + 1)]
res = {
    "environment": json.loads((t / "env.json").read_text()),
    "banach_commit": commit,
    "model": model,
    "planner": json.loads((t / "plan.json").read_text()),
    "probe": (t / "probe.txt").read_text(),
    "decode_resident_tok_s": stats(ref),
}
if int(budget) > 0:
    runs_s = [json.loads((t / f"stream_{i}.json").read_text()) for i in range(1, int(runs) + 1)]
    res["budget_mb"] = int(budget)
    res["decode_streaming_tok_s"] = stats([r["decode"]["tok_s"] for r in runs_s])
    res["hit_rate"] = stats([r["decode"]["hit_rate"] for r in runs_s])
    res["disk_bytes_per_token"] = stats([r["decode"]["disk_bytes_per_token"] for r in runs_s])
    ex = t / "exact.json"
    res["exact_check"] = json.loads(ex.read_text()).get("exact_check") if ex.exists() else None
pathlib.Path(out).write_text(json.dumps(res, indent=1))
print(f"wrote {out}: resident {res['decode_resident_tok_s']['median']:.2f} tok/s"
      + (f", streaming {res['decode_streaming_tok_s']['median']:.2f} tok/s at {budget} MB, exact {res['exact_check']}"
         if int(budget) > 0 else ""))
EOF
