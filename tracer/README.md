# banach-trace

Teacher-forced routing tracer for MoE models on llama.cpp (Phase 0.1). It uses only the
scheduler eval callback, so it works against an unpatched llama.cpp checkout.

Per token and per MoE layer it records:

- routed expert ids (`ffn_moe_topk`)
- final gate weights (the last of `ffn_moe_weights`, `_softmax`, `_norm`, `_scaled`)
- the norm of the MoE input (src1 of the expert mat-mul)
- the norm of each routed expert's output before gate weighting (the last of
  `ffn_moe_down`, `_scaled`, `_biased`, `ffn_moe_post_exp_norm`)
- the top-N next-token log-probabilities (default 20)

Output is `<prefix>.bin` (fixed-size records) and `<prefix>.json` (layout and metadata).
Convert with `python -m banach_phase0 import-trace <prefix>.json --out trace.npz`.

## Build

```bash
git clone https://github.com/ggml-org/llama.cpp
cmake -S tracer -B build-trace -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/llama.cpp
ninja -C build-trace banach-trace
```

Tested against llama.cpp commit 9309bf1 (3 Nov 2026). Use a llama.cpp newer than the fixes
for CVE-2026-33298 and CVE-2026-27940 (SPEC-DERISK section 7).

## Corpus

JSONL, one session per line. Each session starts from an empty KV cache and is cut at `-c`
tokens.

```json
{"session": "s1", "workload": "coding", "text": "..."}
{"session": "s2", "workload": "chat", "messages": [{"role": "user", "content": "..."}]}
```

`messages` go through the model's built-in chat template (`llama_chat_apply_template`). That
function does not run Jinja; if the model's template is not one it knows, pre-render the
text and pass it as `text`, or pass `--raw`.

## Protocol notes

- Use `-c 6144` or more for V4 Flash: its indexer and compressors are inert at 512 (protocol
  asks for at least 5,632).
- `--warmup 1` (default) discards the first pass of a fresh context.
- Trace MTP off for the baseline; take an MTP-on trace separately.
- Each ask for a tensor forces a scheduler sync, so tracing is slower than plain prefill.
- Run the same corpus through an exact build (for V4 Flash, UD-Q8_K_XL) and a candidate
  build, then compare with `python -m banach_phase0 flips exact.npz candidate.npz`.
