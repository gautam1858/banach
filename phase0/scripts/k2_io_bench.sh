#!/usr/bin/env bash
# K2 (SPEC-DERISK section 6): NVMe constants at expert-sized reads, plus a sustained run with
# temperature logging. Run as root on the target box; drop caches between cells.
#
# Usage: k2_io_bench.sh TEST_FILE OUT_DIR [NVME_DEV]
#   TEST_FILE  a real file of 100+ GB on the NVMe under test (e.g. the model GGUF)
#   NVME_DEV   e.g. /dev/nvme0 for smart-log temperature sampling
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 TEST_FILE OUT_DIR [NVME_DEV]" >&2
  exit 2
fi
file=$1; out=$2; dev=${3:-}
command -v fio >/dev/null || { echo "fio is required" >&2; exit 1; }
size=$(stat -c %s "$file")
[ "$size" -ge $((100 * 1000 * 1000 * 1000)) ] || echo "warning: test file < 100 GB" >&2
mkdir -p "$out"
here=$(cd "$(dirname "$0")" && pwd)
"$here/record_env.sh" > "$out/env.json"

drop_caches() { sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true; }

# Block sizes from the protocol (one per expert bit width 2.06 / 2.5 / 3.06 / 4.25).
for bs in 6656k 7864k 9856k 13696k; do
  for qd in 1 4 8 16 32; do
    drop_caches
    fio --name="bs${bs}_qd${qd}" --filename="$file" --readonly --rw=randread \
        --ioengine=io_uring --direct=1 --bs="$bs" --iodepth="$qd" \
        --time_based --runtime=60 --ramp_time=5 --norandommap --randrepeat=0 \
        --output-format=json --output="$out/fio_bs${bs}_qd${qd}.json"
  done
done

# 10-minute sustained run at the 2-bit expert size, sampling temperature every 5 s.
temp_log="$out/nvme_temp.log"
if [ -n "$dev" ] && command -v nvme >/dev/null; then
  ( while true; do
      printf '%s ' "$(date +%s)"; nvme smart-log "$dev" | grep -m1 -i '^temperature'
      sleep 5
    done ) > "$temp_log" &
  temp_pid=$!
  trap 'kill $temp_pid 2>/dev/null || true' EXIT
fi
drop_caches
fio --name=sustained --filename="$file" --readonly --rw=randread --ioengine=io_uring \
    --direct=1 --bs=6656k --iodepth=8 --time_based --runtime=600 --norandommap \
    --write_bw_log="$out/sustained" --log_avg_msec=1000 \
    --output-format=json --output="$out/fio_sustained.json"

python3 - "$out" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows = []
for p in sorted(out.glob("fio_bs*_qd*.json")):
    job = json.loads(p.read_text())["jobs"][0]["read"]
    rows.append({"cell": p.stem[4:], "GBps": job["bw_bytes"] / 1e9,
                 "lat_p99_ms": job["clat_ns"]["percentile"].get("99.000000", 0) / 1e6})
for r in rows:
    print(f"{r['cell']:<16} {r['GBps']:6.2f} GB/s  p99 {r['lat_p99_ms']:7.2f} ms")
(out / "k2_summary.json").write_text(json.dumps(rows, indent=1))
EOF
echo "Next: STREAM triad (dram_bw) and a 7 GB FP8 GEMV; record both in $out."
