#!/usr/bin/env bash
# P0.0: record the environment for a Phase 0 run as JSON on stdout.
# Usage: record_env.sh [--model FILE ...] [--llama-dir DIR] > env.json
# Some fields need root (dmidecode, nvme). Missing tools produce null, never a failure.
set -u

models=()
llama_dir=""
while [ $# -gt 0 ]; do
  case "$1" in
    --model) models+=("$2"); shift 2 ;;
    --llama-dir) llama_dir="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

j() {  # JSON-escape stdin
  python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))'
}
run() {  # run a command, emit JSON string or null
  if out=$("$@" 2>/dev/null) && [ -n "$out" ]; then printf '%s' "$out" | j; else echo null; fi
}

cpu_model=$(run sh -c "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2-")
microcode=$(run sh -c "grep -m1 microcode /proc/cpuinfo | cut -d: -f2-")
flags=$(run sh -c "grep -m1 '^flags' /proc/cpuinfo | cut -d: -f2- | tr ' ' '\n' | grep -E '^(avx2|avx512f|avx512_bf16|avx512_vnni|avx_vnni|amx_tile|fma|f16c)$' | sort | tr '\n' ' '")
cores=$(run nproc)
dimms=$(run dmidecode -t memory)
kernel=$(run uname -a)
mglru=$(run cat /sys/kernel/mm/lru_gen/enabled)
thp=$(run cat /sys/kernel/mm/transparent_hugepage/enabled)
governor=$(run sh -c "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor | sort | uniq -c")
nvme=$(run nvme list)
fs=$(run df -T .)
temps=$(run sensors)
meminfo=$(run sh -c "grep -E 'MemTotal|MemAvailable|SwapTotal' /proc/meminfo")
script_rev=$(run git -C "$(dirname "$0")" rev-parse HEAD)
llama_rev=null
if [ -n "$llama_dir" ]; then llama_rev=$(run git -C "$llama_dir" rev-parse HEAD); fi

model_json="[]"
if [ ${#models[@]} -gt 0 ]; then
  model_json=$(for m in "${models[@]}"; do
    printf '%s\t%s\n' "$m" "$(sha256sum "$m" | cut -d' ' -f1)"
  done | python3 -c 'import json,sys; print(json.dumps([{"path":p,"sha256":h} for p,h in (l.rstrip("\n").split("\t") for l in sys.stdin)]))')
fi

cat <<EOF
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "cpu_model": $cpu_model,
  "microcode": $microcode,
  "cpu_flags": $flags,
  "cores": $cores,
  "dimms": $dimms,
  "meminfo": $meminfo,
  "kernel": $kernel,
  "mglru": $mglru,
  "thp": $thp,
  "governor": $governor,
  "nvme": $nvme,
  "filesystem": $fs,
  "sensors": $temps,
  "llama_cpp_commit": $llama_rev,
  "script_commit": $script_rev,
  "models": $model_json
}
EOF
