#!/usr/bin/env bash
# End-to-end checks of a built banach + banach-engine pair, used by CI and by the release gate.
#   scripts/smoke_test.sh BANACH ENGINE MODEL.gguf PROMPT_FILE
# The model should be small (stories15M_MOE works). Exits non-zero on the first failure.
set -euo pipefail
banach=$1; engine=$2; model=$3; prompt=$4
export BANACH_ENGINE=$engine BANACH_HOME=$(mktemp -d)
port=${SMOKE_PORT:-18099}

echo "== version"
"$banach" --version

echo "== compare a model with itself: identical"
"$banach" compare "$model" "$model" -f "$prompt" --json | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert d["top1_agreement"] == 1.0 and d["kl_mean"] == 0, d
print("top-1 agreement", d["top1_agreement"], "KL", d["kl_mean"])'

echo "== pick ranks a local model"
"$banach" pick "$model" | tee /dev/stderr | grep -q 'tok/s'

echo "== server hardening and a completion"
"$banach" serve "$model" --port "$port" --api-key testkey --ctx 512 >"$BANACH_HOME/serve.log" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null || true' EXIT
base=http://127.0.0.1:$port
for _ in $(seq 1 90); do curl -fsS "$base/health" >/dev/null 2>&1 && break; sleep 1; done
curl -fsS "$base/health" | grep -q ok
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }
test "$(code "$base/v1/models")" = 401
test "$(code -H 'Authorization: Bearer wrong' "$base/v1/models")" = 401
test "$(code -H 'Authorization: Bearer testkey' "$base/v1/models")" = 200
test "$(code -H 'x-api-key: testkey' "$base/v1/models")" = 200
test "$(code -H 'Host: evil.example' -H 'Authorization: Bearer testkey' "$base/v1/models")" = 403
test "$(code -X OPTIONS -H 'Authorization: Bearer testkey' "$base/v1/chat/completions")" = 405
test "$(code -X POST -H 'Authorization: Bearer testkey' -H 'Content-Type: text/plain' -d '{}' "$base/v1/chat/completions")" = 415
curl -fsS -H 'Authorization: Bearer testkey' -H 'Content-Type: application/json' \
  -d '{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"Once upon a time"}]}' \
  "$base/v1/chat/completions" | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert d["choices"][0]["message"]["content"] is not None, d
print("completion ok:", repr(d["choices"][0]["message"]["content"])[:60])'
curl -fsS -H 'Authorization: Bearer testkey' "$base/banach/status" | python3 -c '
import json, sys
def strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from strings(x)
leaks = [t for t in strings(json.load(sys.stdin)) if t.startswith(("/", "\\\\")) or t[1:3] == ":\\"]
assert not leaks, "status reveals paths: %s" % leaks'
echo "smoke test passed"
