<p align="center">
  <img src="brand/video/hero.gif" alt="One sphere is cut into pieces and reassembled into two, forming the banach logo" width="800">
</p>

<p align="center">
  <b>Run Mixture-of-Experts models bigger than your RAM, with an honest plan before you download and a fidelity meter while you run.</b>
</p>

<p align="center">
  <a href="brand/video/banach-launch-16x9.mp4">Watch the 44 s film</a> &nbsp;|&nbsp;
  <a href="#install">Install</a> &nbsp;|&nbsp;
  <a href="#60-second-start">60-second start</a> &nbsp;|&nbsp;
  <a href="docs/VALIDATION.md">Every measured number</a>
</p>

# Banach

**Run Mixture-of-Experts models bigger than your RAM, and know what it will cost before you
download a byte.**

When a model does not fit, plain `llama.cpp` falls back on the operating system: the file is
memory-mapped and the kernel pages experts in on demand, 4 KB at a time, with no idea which
experts the router is about to use. Banach replaces that with a scheduler that knows the
model:

1. **Predict first.** `banach plan hf:...` reads a few MB of GGUF headers and predicts
   tokens per second for your machine, with a range, plus how the model splits across VRAM,
   RAM and NVMe. Predictions landed within +-30% of measurements on held-out runs.
2. **Stream experts on purpose.** The exact weights stay on disk, untouched. Banach keeps a
   RAM budget for routed experts, fetches whole experts in parallel the moment the router
   picks them, and admits only experts that are used often enough to deserve the space.
   Output stays byte-identical to stock llama.cpp.
3. **Measure every shortcut.** Optional lower-precision copies of cold experts speed things
   up; when they are on, Banach replays responses against the exact model and keeps an
   anytime-valid confidence bound on how often the top token changes.

The name is the Banach-Tarski paradox: one ball, cut into pieces and reassembled, becomes
two. More model than your machine should hold.

### Measured so far, and what is still a projection

| Claim | Status | Evidence |
|---|---|---|
| Byte-identical to stock llama.cpp while streaming | measured, every configuration tested | [VALIDATION, Phase 2](docs/VALIDATION.md) |
| 2.1x faster than llama.cpp page-faulting experts in, at the same RAM budget | measured on a 1B MoE, 1 GB/s disk | [VALIDATION, Phase 2](docs/VALIDATION.md) |
| Planner within +-30% of measured speed | measured on held-out runs, small models | [VALIDATION, Phase 1](docs/VALIDATION.md) |
| Low-precision experts flip only near-tied tokens at a 21.6% low share | measured on a 1B MoE; not yet certified below 1% | [VALIDATION, Phase 3](docs/VALIDATION.md) |
| DeepSeek V4 Flash (284B) speeds on 32 GB and 64 GB + GPU machines | **projection** from the planner, not measured | [VALIDATION, Phase 1](docs/VALIDATION.md) |

### Why not mmap, `-ot` overrides, KTransformers or ik_llama.cpp?

They are good tools and Banach builds on llama.cpp's kernels. The difference is what happens
when the model is larger than memory:

- **llama.cpp with mmap** (and `-ot` tensor overrides) leaves the spill to the kernel's page
  cache, which knows nothing about experts, routing or which bytes matter next.
- **KTransformers** and **ik_llama.cpp** make CPU/GPU hybrid inference fast when the weights
  fit in RAM plus VRAM; past that point they also rely on the operating system.
- **Banach** is built for the past-that-point case: an expert-aware RAM budget over an exact
  copy on NVMe, a prediction before you download, and a meter for every lossy shortcut.

## Install

Pick one. Each gives you the `banach` command with its engine.

```bash
pip install banach
```

```bash
npm install -g banach
```

```bash
curl -fsSL https://raw.githubusercontent.com/gautam1858/banach/main/install.sh | sh
```

```bash
docker run --rm -p 8080:8080 -v banach:/data ghcr.io/gautam1858/banach --help
```

```bash
cargo install banach
```

Prebuilt binaries: Linux x86_64 (AVX2 or newer), Linux arm64, macOS on Apple Silicon.
`cargo install` builds the planner, downloader and server; the inference engine comes with
the other channels or [from source](#build-from-source).

### Hardware

Every row below is built by CI on each commit; "byte-exact" means the streaming engine is
checked token for token and logit for logit against stock llama.cpp on that platform.

| Platform | Engine | Expert streaming | CI |
|---|---|---|---|
| Linux x86_64 | CPU (AVX2, AVX-512) | full: `readahead`, `MADV_POPULATE_READ`, hard RAM budget, `mincore` checks | built, byte-exact |
| Linux arm64 | CPU (NEON) | full | built, byte-exact |
| macOS, Apple Silicon | CPU, Metal | prefetch and touch; the budget is soft (the kernel may keep evicted pages) | built, byte-exact |
| Windows x86_64 | CPU | `PrefetchVirtualMemory`; soft budget by working-set trim; no residency checks | built, byte-exact |
| NVIDIA GPUs | CUDA | dense layers and KV on the GPU (`--gpu-layers`), routed experts streamed from RAM | built (no GPU in CI) |
| AMD, Intel, NVIDIA GPUs | Vulkan | as CUDA | built (no GPU in CI) |

Measured speed numbers so far are from Linux x86_64; numbers for the other rows will be
added to docs/VALIDATION.md as they are run. GPU builds come from source: add
`-DGGML_CUDA=ON`, `-DGGML_VULKAN=ON` or (on macOS, on by default) Metal to the cmake line.

## 60-second start

1. Ask before you download. This reads a few MB of headers, not 155 GB of weights:

   ```bash
   banach plan hf:unsloth/DeepSeek-V4-Flash-GGUF/UD-Q4_K_XL/DeepSeek-V4-Flash-UD-Q4_K_XL-00001-of-00005.gguf
   ```

2. Serve a model with an OpenAI and Anthropic compatible API. `--lazy` starts answering
   while the rest of the file is still downloading:

   ```bash
   banach serve hf:QuantFactory/granite-3.1-1b-a400m-instruct-GGUF/granite-3.1-1b-a400m-instruct.Q8_0.gguf --lazy
   ```

3. Talk to it from anything that speaks OpenAI or Anthropic:

   ```python
   from openai import OpenAI
   client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
   print(client.chat.completions.create(
       model="local", messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
   ```

   ```python
   import anthropic
   client = anthropic.Anthropic(base_url="http://127.0.0.1:8080", api_key="local")
   print(client.messages.create(
       model="local", max_tokens=200, messages=[{"role": "user", "content": "Hello!"}]).content[0].text)
   ```

   ```bash
   curl http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
     -d '{"model":"local","messages":[{"role":"user","content":"Hello!"}],"stream":true}'
   ```

4. Watch it: `http://127.0.0.1:8080/banach/status` shows the engine, the fidelity meter and
   download progress.

## What works today

| Piece | Status |
|---|---|
| Weights-free preflight | `banach plan hf:...` reads only GGUF headers (a few MB over HTTP range requests) and predicts tok/s with a range, per plan, before any download |
| Planner | roofline cost model over VRAM/RAM/NVMe, hot-set placement by multiple-choice knapsack, validated within +-30% on held-out runs |
| Lossless streaming | `banach-engine` keeps a RAM budget for routed experts: parallel whole-expert fetch, frequency-gated admission, background eviction; byte-identical to stock llama.cpp |
| Precision ladder | llama.cpp patch: each routed slot reads either the exact or a low-bit copy of its expert; numerically exact plumbing |
| Fidelity meter | shadow-exact replay, betting confidence sequences on top-1 flips, PI controller on the ladder threshold |
| APIs | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, streaming and not, tool use, prefix cache; tested with both official SDKs |
| Downloads | `banach pull`: resumable, dense tensors and hot experts first, SHA-256 against the Hugging Face LFS id; `serve --lazy` starts before the download ends |

Measured numbers, with machine, model, quant and command for each, are in
[docs/VALIDATION.md](docs/VALIDATION.md). The DeepSeek V4 Flash headline still needs the
Phase 0 kill checks on a machine with 128 GB or more (SPEC-DERISK section 6).

## Build from source

```bash
git clone https://github.com/gautam1858/banach && cd banach
cargo build --release
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
(cd ../llama.cpp && git checkout 9309bf1 && git apply ../banach/engine/patches/0001-banach-precision-ladder.patch)
cmake -S engine -B build-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/../llama.cpp
ninja -C build-engine banach-engine
export BANACH_ENGINE=$PWD/build-engine/banach-engine
./target/release/banach --help
```

A source build tunes the engine for your CPU (`-march=native`), which is faster than the
portable prebuilt binaries. For a GPU, add `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` to the
cmake line and run with `--gpu-layers 99`: dense layers go to the GPU, routed experts stay in
RAM under the streamer. On Windows, run the same steps from a Visual Studio developer
prompt; the binary is `build-engine\banach-engine.exe`.

## Commands

| Command | What it does |
|---|---|
| `banach probe [--disk FILE]` | measures memory bandwidth, peak FMA, disk reads; saves the hardware profile |
| `banach inspect MODEL` | model profile from headers: experts, bytes per token, KV per token |
| `banach plan MODEL` | preflight card; `--hw FILE` plans for another machine, `--usage FILE` uses a routing trace |
| `banach pull hf:...` | resumable, prioritised, verified download |
| `banach run MODEL -f PROMPT` | plans the expert budget and runs one generation; `--gpu-layers N` offloads dense layers |
| `banach serve MODEL` | local OpenAI/Anthropic compatible server |
| `banach certify FLIPS` | anytime-valid certificate on a recorded flip sequence |
| `banach calibrate MODEL --measured TOK_S` | fits the cost model to one measured run |

## Limits, stated plainly

- Tool use works on both APIs (OpenAI `tools` / `tool_calls`, Anthropic `tool_use` /
  `tool_result`), streaming included. Templates llama.cpp knows get grammar-constrained,
  natively parsed calls; others fall back to extracting JSON calls from the output. How well
  a model uses tools is up to the model: small models (1 to 3B) call simple tools reliably
  but are weak agents.
- One request at a time; concurrent requests queue.
- The hard RAM budget and lazy mode are fastest on Linux; macOS and Windows keep the same
  output but their budget is soft (see Hardware).
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
| `brand/` | logo, launch film and its source (render it yourself: `brand/README.md`) |
| `python/`, `npm/`, `install.sh`, `Dockerfile` | packaging; releases are built by `.github/workflows/release.yaml` |
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
