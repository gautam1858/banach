#!/usr/bin/env bash
# K1 kill check (SPEC-DERISK section 6): all-resident decode of the official 2-bit and 3-bit
# DeepSeek V4 Flash GGUFs on a same-CPU-family machine with >= 128 GB RAM.
#
#   2-bit tg < 11 tok/s  =>  the 32 GB CPU-only headline is dead.
#
# Usage: k1_resident_bench.sh LLAMA_DIR OUT_DIR MODEL_IQ2_XXS.gguf MODEL_IQ3_XXS.gguf
set -euo pipefail

if [ $# -ne 4 ]; then
  echo "usage: $0 LLAMA_DIR OUT_DIR MODEL_IQ2_XXS.gguf MODEL_IQ3_XXS.gguf" >&2
  exit 2
fi
llama_dir=$1; out=$2; m2=$3; m3=$4
bench="$llama_dir/build/bin/llama-bench"
[ -x "$bench" ] || { echo "llama-bench not found at $bench" >&2; exit 1; }
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)

cores=$(lscpu -p=CORE | grep -v '^#' | sort -u | wc -l)  # physical cores
"$here/record_env.sh" --llama-dir "$llama_dir" --model "$m2" --model "$m3" > "$out/env.json"

for m in "$m2" "$m3"; do
  name=$(basename "$m" .gguf)
  # Warm the page cache so the run is truly resident, then 5 repetitions.
  cat "$m" > /dev/null
  "$bench" -m "$m" -t "$cores" -n 128 -p 0 -r 5 --mmap 1 -o json > "$out/$name.tg.json"
  "$bench" -m "$m" -t "$cores" -n 0 -p 512 -r 3 --mmap 1 -o json > "$out/$name.pp.json"
done

python3 - "$out" "$(basename "$m2" .gguf)" <<'EOF'
import json, sys, pathlib
out, name2 = pathlib.Path(sys.argv[1]), sys.argv[2]
rows = json.loads((out / f"{name2}.tg.json").read_text())
tg = max(r.get("avg_ts", 0.0) for r in rows if r.get("n_gen", 0) > 0)
verdict = "PASS" if tg >= 11.0 else "FAIL (32 GB CPU-only headline dead)"
print(f"K1: 2-bit all-resident decode {tg:.2f} tok/s -> {verdict}")
(out / "k1_verdict.json").write_text(json.dumps({"tg_2bit_tok_s": tg, "threshold": 11.0,
                                                  "pass": tg >= 11.0}, indent=1))
EOF
