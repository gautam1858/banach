<p align="center">
  <img src="brand/video/hero.gif" alt="One sphere is cut into pieces and reassembled into two, forming the banach logo" width="800">
</p>

<h1 align="center">Banach</h1>

<p align="center">
  <b>Run AI models bigger than your RAM. Faster than llama.cpp. Byte-for-byte the same answers.</b>
</p>

<p align="center">
  <a href="https://pypi.org/project/banach/"><img src="https://img.shields.io/pypi/v/banach?style=flat-square&color=3775A9&label=pip%20install%20banach" alt="PyPI"></a>
  <a href="https://github.com/gautam1858/banach/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/gautam1858/banach/ci.yml?branch=main&style=flat-square&label=byte-exact%20CI" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square" alt="License"></a>
  <a href="https://buymeacoffee.com/gautam1858"><img src="https://img.shields.io/badge/Buy%20me%20a%20coffee-FFDD00?style=flat-square&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"></a>
</p>

<p align="center">
  <a href="brand/video/banach-launch-16x9.mp4">Watch the film</a> &nbsp;|&nbsp;
  <a href="#install">Install</a> &nbsp;|&nbsp;
  <a href="#60-second-start">60-second start</a> &nbsp;|&nbsp;
  <a href="docs/VALIDATION.md">Benchmarks</a>
</p>

---

## A 22 GB model on a 16 GB machine

Qwen3.6-35B-A3B (Q4_K_M, 22.1 GB) on a 4-core cloud VM with 16 GB of RAM, no swap and a
1.27 GB/s disk. Cold start, same prompt, same greedy decode.

| | stock llama.cpp | **Banach** | |
|---|---:|---:|---|
| Prompt processing | 7.64 s | **3.44 s** | **2.2x faster** |
| Generation | 4.89 tok/s | **5.40 tok/s** | **+10%** |
| RAM for experts | all it can take | **10 GB, capped** | the rest of your machine stays usable |
| Output | reference | **identical** | every token, every logit, bit for bit |

Squeeze the RAM budget further and race llama.cpp paging experts in at that same budget: **2.1x faster generation** ([details](docs/VALIDATION.md)).

No approximations, no re-quantizing, no quality loss. The exact weights stay on disk; Banach
just moves them better.

## How

A Mixture-of-Experts model only uses a few of its experts per token. When the file does not
fit in memory, plain llama.cpp leaves it to the operating system, which pages bytes in 4 KB at
a time with no idea which experts come next. Banach knows the model:

- **It plans before you download.** `banach plan hf:...` reads a few MB of headers and tells
  you the speed to expect on your machine and how the model splits across GPU, RAM and disk.
- **It streams experts on purpose.** Whole experts, fetched in parallel the moment the
  router picks them, held in a RAM budget you choose, with the most-used experts kept hot.
- **It proves it.** Every configuration is checked against stock llama.cpp token for token.
  Optional faster low-precision experts come with a live fidelity meter, so a shortcut is
  never silent.

The name comes from the Banach-Tarski paradox: one ball, cut into pieces and reassembled,
becomes two. More model than your machine should hold.

## Install

```bash
pip install banach
```

That is the whole install: the `banach` command and its engine, prebuilt for Linux x86_64,
Linux arm64 and macOS on Apple Silicon. Windows and GPU builds: [from source](#build-from-source).

## 60-second start

1. **See the speed before you download.** This reads a few MB of headers, not 22 GB:

   ```bash
   banach plan hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
   ```

2. **Serve it** with an OpenAI and Anthropic compatible API. `--lazy` starts answering while
   the file is still downloading:

   ```bash
   banach serve hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --lazy
   ```

3. **Use it** from anything that speaks OpenAI or Anthropic, coding agents included (tool use
   works on both APIs):

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

4. **Watch it** at `http://127.0.0.1:8080/banach/status`: engine, fidelity meter, download.

## Runs everywhere

| Platform | Backends | Expert streaming |
|---|---|---|
| Linux x86_64 and arm64 | CPU (AVX2, AVX-512, NEON), CUDA, Vulkan | full: hard RAM budget, parallel prefetch, residency checks |
| Windows x86_64 | CPU (CUDA and Vulkan from source) | prefetch, working-set budget |
| macOS, Apple Silicon | CPU, Metal | prefetch, soft budget |
| Any GPU | `--gpu-layers N` | dense layers on the GPU, experts streamed from RAM |

Byte-exact against stock llama.cpp in CI on every commit, on Linux x64, Linux arm64 and
Windows; CUDA and Vulkan builds are compiled in CI too.

## Features

| | |
|---|---|
| Preflight | speed estimate and memory split from headers alone, before any download |
| Lossless streaming | RAM budget for experts, parallel whole-expert fetch, frequency-gated admission, background eviction |
| GPU offload | dense layers and KV cache on CUDA, Metal or Vulkan; experts stay in RAM |
| APIs | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, streaming, tool use, prefix cache |
| Downloads | resumable, hot experts first, SHA-256 verified; serve before the download ends |
| Fidelity meter | shadow replay against the exact model with anytime-valid confidence bounds |

## Commands

| Command | What it does |
|---|---|
| `banach plan MODEL` | speed estimate and memory split for your machine, from headers |
| `banach serve MODEL` | local OpenAI and Anthropic compatible server (`--lazy`, `--gpu-layers N`) |
| `banach run MODEL -p TEXT` | one generation with the planned expert budget |
| `banach pull hf:...` | resumable, prioritised, verified download |
| `banach probe` | measures your memory bandwidth, compute and disk |
| `banach inspect MODEL` | experts, bytes per token, KV per token |
| `banach calibrate MODEL --measured TOK_S` | tunes the planner to your machine from one run |
| `banach certify FLIPS` | confidence certificate on a recorded fidelity run |

## Build from source

```bash
git clone https://github.com/gautam1858/banach && cd banach
cargo build --release
git clone https://github.com/ggml-org/llama.cpp ../llama.cpp
(cd ../llama.cpp && git checkout 9309bf1 && git apply ../banach/engine/patches/0001-banach-precision-ladder.patch)
cmake -S engine -B build-engine -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=$PWD/../llama.cpp
ninja -C build-engine banach-engine
export BANACH_ENGINE=$PWD/build-engine/banach-engine
```

- **GPU:** add `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` to the cmake line (Metal is on by
  default on macOS) and serve with `--gpu-layers 99`.
- **Windows:** the same steps from a Visual Studio developer prompt; the engine is
  `build-engine\banach-engine.exe`.
- A source build is tuned for your exact CPU and is faster still than the prebuilt binaries.

## Good to know

- One request at a time; concurrent requests queue.
- The hard RAM budget is Linux; macOS and Windows give identical output with a softer budget.
- Small models (1 to 3B) call tools fine but are weak coding agents; use a bigger model for
  agent work.

Every number in this README, with machine, model, commit and command:
[docs/VALIDATION.md](docs/VALIDATION.md).

## Support

Banach is free and open source. If it saves you a hardware upgrade, help fund test machines
and GPUs for the next round of benchmarks:

<a href="https://buymeacoffee.com/gautam1858"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="48"></a>

## Credits

Built on llama.cpp and ggml (MIT), which run every kernel. Shaped by the llama.cpp
expert-streaming pull requests (25294, 31469) and MoE cache (29887), by colibri's lossless
NVMe streaming and published measurements, and by research on expert locality (Si et al.,
Liang et al.), mixed-precision experts (SliceMoE, DyMoE, HOBBIT) and betting confidence
sequences (Waudby-Smith and Ramdas).

## License

Apache-2.0, see [LICENSE](LICENSE).
