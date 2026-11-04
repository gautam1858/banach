#!/usr/bin/env bash
# Byte-exact oracle (non-negotiable 2): every streaming configuration must reproduce stock
# llama.cpp token for token and logit bit for bit.
#
# Usage: exact_check.sh ENGINE MODEL PROMPT_FILE [N_TOKENS]
# Exits non-zero on the first mismatch.
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 ENGINE MODEL PROMPT_FILE [N_TOKENS]" >&2
  exit 2
fi
engine=$1; model=$2; prompt=$3; n=${4:-24}

# model size in MB, to pick budgets of roughly 20%, 50% and 100% of the routed pool
size_mb=$(( $(wc -c < "$model") / 1000000 ))
fail=0
for budget in $((size_mb / 5)) $((size_mb / 2)) 0; do
  for policy in lru admit; do
    for mode in naive banach; do
      args=(-m "$model" -f "$prompt" -n "$n" --mode "$mode" --policy "$policy" --exact-check --no-residency)
      [ "$budget" -gt 0 ] && args+=(--budget-mb "$budget")
      if out=$("$engine" "${args[@]}" 2>&1); then
        echo "PASS mode=$mode policy=$policy budget=${budget}MB"
      else
        echo "FAIL mode=$mode policy=$policy budget=${budget}MB"
        echo "$out" | tail -5
        fail=1
      fi
    done
  done
done
exit $fail
