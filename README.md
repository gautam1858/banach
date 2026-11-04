# Banach

> Banach-Tarski for language models: more model than your machine should hold.

Banach runs very large Mixture-of-Experts models on ordinary machines by keeping the exact
model on NVMe as a lossless floor, serving most tokens from a hot set in RAM, escalating
precision only where it matters, and measuring fidelity against the exact model on your own
traffic. Every prediction comes with a range, and every lossy setting comes with a meter.

## What works today

| Piece | Status |
|---|---|
| Weights-free preflight | `banach plan hf:...` reads only GGUF headers (a few MB over HTTP range requests) and predicts tok/s with a range, per plan, before any download |
| Planner | roofline cost model over VRAM/RAM/NVMe, hot-set placement by multiple-choice knapsack, validated within +-30% on held-out runs |
| Lossless streaming | `banach-engine` keeps a RAM budget for routed experts: parallel whole-expert fetch, frequency-gated admission, background eviction; byte-identical to stock llama.cpp |
| Precision ladder | llama.cpp patch: each routed slot reads either the exact or a low-bit copy of its expert; numerically exact plumbing |
| Fidelity meter | shadow-exact replay, betting confidence sequences on top-1 flips, PI controller on the ladder threshold |
| APIs | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, streaming and not, prefix cache; tested with both official SDKs |
| Downloads | `banach pull`: resumable, dense tensors and hot experts first, SHA-256 against the Hugging Face LFS id; `serve --lazy` starts before the download ends |

Measured numbers, with machine, model, quant and command for each, are in
[docs/VALIDATION.md](docs/VALIDATION.md). The DeepSeek V4 Flash headline still needs the
Phase 0 kill checks on a machine with 128 GB or more (SPEC-DERISK section 6).

## Quick start

```bash
cargo build --release
git clone https://github.com/ggml-org/llama.cpp && (cd llama.cpp && git checkout 9309bf1 && git apply ../engine/patches/0001-banach-precision-ladder.patch)
cmake -S engine -B build-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/llama.cpp && ninja -C build-engine banach-engine
export BANACH_ENGINE=$PWD/build-engine/banach-engine
```

Will it run, and how fast, before downloading anything:

```bash
./target/release/banach probe
./target/release/banach plan hf:unsloth/DeepSeek-V4-Flash-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf --min-tok-s 10
```

Serve a model (downloaded first, or while downloading with `--lazy`):

```bash
./target/release/banach serve hf:QuantFactory/granite-3.1-1b-a400m-instruct-GGUF/granite-3.1-1b-a400m-instruct.Q8_0.gguf --lazy
```

Then point any OpenAI or Anthropic client at it:

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

- OpenAI SDKs: `base_url="http://127.0.0.1:8080/v1"`, any API key
- Anthropic SDKs: `base_url="http://127.0.0.1:8080"`, any API key
- `GET /banach/status`: engine, ladder threshold, fidelity meter, download progress

With a precision ladder and a live fidelity meter:

```bash
./target/release/banach serve model-Q8_0.gguf --ladder model-Q4_K_M.gguf --tau 7 --target 0.01
```

## Commands

| Command | What it does |
|---|---|
| `banach probe [--disk FILE]` | measures memory bandwidth, peak FMA, disk reads; saves the hardware profile |
| `banach inspect MODEL` | model profile from headers: experts, bytes per token, KV per token |
| `banach plan MODEL` | preflight card; `--hw FILE` plans for another machine, `--usage FILE` uses a routing trace |
| `banach pull hf:...` | resumable, prioritised, verified download |
| `banach run MODEL -f PROMPT` | plans the expert budget and runs one generation |
| `banach serve MODEL` | local OpenAI/Anthropic compatible server |
| `banach certify FLIPS` | anytime-valid certificate on a recorded flip sequence |
| `banach calibrate MODEL --measured TOK_S` | fits the cost model to one measured run |

## Limits, stated plainly

- Tool use (function calling) is not supported yet; requests with tools get a 400.
- One request at a time; concurrent requests queue.
- Linux only for streaming and lazy mode (`readahead`, `MADV_POPULATE_READ`); macOS needs
  another fetch path.
- The ladder and expert streaming do not yet run together (the low copy should be the
  resident tier with exact experts streamed on escalation).
- Lazy start only helps as much as the first prompt is sparse: a prefill touches a large
  share of the experts of every layer, and those have to be fetched before it can finish.
- Raw top-1 agreement within 1% of the exact model is out of reach for any lossy change on
  the models tested so far; the meter certifies flips on confident tokens and reports raw
  flips and KL alongside. See docs/VALIDATION.md, Phase 3.

## Layout

| Path | Contents |
|---|---|
| `docs/spec/` | specifications; precedence BANACH-SPEC, then SPEC-DERISK, SPEC-MATH, SPEC-LADDER, SPEC-SYSTEM |
| `docs/VALIDATION.md` | every measured number, with machine, model, commit and command |
| `src/` | `banach` (Rust): probe, GGUF parser, planner, fidelity meter, downloader, HTTP server |
| `engine/` | `banach-engine` (C++ on llama.cpp), the llama.cpp ladder patch, the byte-exact oracle |
| `tracer/` | `banach-trace`, the routing tracer |
| `phase0/` | Phase 0 analysis in Python: cache simulator, entropy floor, eps profiler, certificates |
| `bench/` | reproducible benchmark script (one JSON per run) |
| `phase1/machines/` | reference hardware profiles for `banach plan --hw` |

## Credits

Banach builds on other people's work and says so:

- llama.cpp and ggml (MIT): every kernel Banach runs. Expert-streaming work in llama.cpp
  pull requests 25294 and 31469, and the merged MoE cache (29887), shaped this design.
- colibri (Apache-2.0): lossless expert streaming from NVMe and honest published numbers;
  its measurements (issue 905) calibrate the planner's hit-rate curve.
- ik_llama.cpp: trellis quant types.
- Unsloth and AtomicChat: published fidelity tables for DeepSeek V4 quants.
- Research: Si et al. and Liang et al. on expert locality; SliceMoE, DyMoE and HOBBIT on
  mixed-precision experts; Waudby-Smith and Ramdas on betting confidence sequences.

## License

Apache-2.0, see [LICENSE](LICENSE).
