<p align="center">
  <img src="brand/video/hero.gif" alt="One sphere is cut into pieces and reassembled into two, forming the banach logo" width="800">
</p>

<h1 align="center">Banach</h1>

<h3 align="center">A trillion parameters on the computer you already own.</h3>

<p align="center">
  Run Mixture-of-Experts models <b>bigger than your RAM</b>, with <b>byte-for-byte the same answers</b><br>
  as llama.cpp. Know the speed <b>before</b> you download a single weight.
</p>

<p align="center">
  <a href="https://pypi.org/project/banach/"><img src="https://img.shields.io/pypi/v/banach?style=for-the-badge&color=3775A9&label=pip%20install%20banach" alt="PyPI"></a>
  <a href="https://github.com/gautam1858/banach/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/gautam1858/banach/ci.yml?branch=main&style=for-the-badge&label=byte-exact%20CI" alt="CI"></a>
  <a href="https://github.com/gautam1858/banach/stargazers"><img src="https://img.shields.io/github/stars/gautam1858/banach?style=for-the-badge&color=f5c518" alt="Stars"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f?style=for-the-badge" alt="License"></a>
</p>

<p align="center">
  <a href="brand/video/banach-launch-16x9.mp4"><b>Watch the film</b></a> &nbsp;&middot;&nbsp;
  <a href="#get-started-in-one-line"><b>Get started</b></a> &nbsp;&middot;&nbsp;
  <a href="#the-giants-on-your-machine"><b>The giants on your machine</b></a> &nbsp;&middot;&nbsp;
  <a href="docs/VALIDATION.md"><b>Every number</b></a>
</p>

---

<p align="center">
  <img src="brand/hero-card.svg" alt="Kimi K2, one trillion parameters, on a desktop with 64 GB of RAM and one RTX 5080: about 3.3 tokens per second with lossless expert streaming, a banach plan estimate" width="900">
</p>

## The giants on your machine

What `banach plan` says, from model headers alone, for the biggest open models on three
machines. **These are estimates, not measurements**, for the engine as it is today: with
`--gpu-layers` the dense layers and KV cache go to the GPU and the routed experts stay in RAM
and on NVMe. On the one big model measured end to end (Qwen3.6-35B on the 16 GB machine) the
planner said 2.1 tok/s and the run gave 5.4.

| Model | Parameters | On disk | 16 GB, 4 cores<br><sub>measured disk 1.3 GB/s</sub> | 32 GB desktop<br><sub>DDR5, Gen4 NVMe, no GPU</sub> | 64 GB + RTX 5080<br><sub>2x Gen5 NVMe</sub> |
|---|---:|---:|---:|---:|---:|
| **Kimi K2** (UD-Q2_K_XL) | 1 trillion | 382 GB | 0.2 tok/s (*) | 1.0 tok/s (*) | **3.3 tok/s (*)** |
| **DeepSeek V4 Flash** (UD-Q4_K_XL) | 284 B | 155 GB | 0.4 tok/s (*) | 2.2 tok/s (*) | **9.2 tok/s** |
| **Qwen3 235B-A22B** (UD-Q2_K_XL) | 235 B | 89 GB | 0.3 tok/s (*) | 1.7 tok/s | **6.2 tok/s** |
| **gpt-oss-120b** (F16) | 117 B | 65 GB | 1.0 tok/s (*) | 4.9 tok/s | **20.7 tok/s** |
| **GLM-4.5-Air** (UD-Q2_K_XL) | 106 B | 47 GB | 0.6 tok/s (*) | 2.8 tok/s | **12.4 tok/s** |
| **Qwen3.6-35B-A3B** (UD-Q4_K_M) | 35 B | 22 GB | **5.4 tok/s measured** | 8.8 tok/s | **48.5 tok/s** |

(*) Fewer than 23% of the experts fit in memory. The planner was compared with measurements
down to that point (Granite, 300 MB budget) and not below, so these cells extrapolate. The 16 GB
column is the VM we measured on; the 32 GB and 64 GB columns are reference hardware profiles
(`phase1/machines/`), not machines we ran. Details: [docs/VALIDATION.md](docs/VALIDATION.md).

A GPU expert cache (not built yet) would lift the 64 GB column, for example Qwen3.6-35B from
48.5 to 80.6 tok/s and Kimi K2 from 3.3 to 3.6; `banach plan --vram-experts` shows that
projection. Ask about your own machine in ten seconds:

```bash
banach probe && banach plan hf:unsloth/Kimi-K2-Instruct-GGUF/UD-Q2_K_XL/Kimi-K2-Instruct-UD-Q2_K_XL-00001-of-00008.gguf
```

**Ran one of these?** [Post your numbers](https://github.com/gautam1858/banach/issues/new?title=Benchmark:%20MODEL%20on%20MACHINE)
with the output of `banach probe` and `banach run`. Measured results replace the planner's
numbers in this table, with your name on them.

## See it

```console
$ pip install banach

# one trillion parameters, a 64 GB desktop with one gaming GPU: ask first, it reads 16 MB
$ banach plan hf:unsloth/Kimi-K2-Instruct-GGUF/UD-Q2_K_XL/Kimi-K2-Instruct-UD-Q2_K_XL-00001-of-00008.gguf --hw phase1/machines/ref-64gb-rtx5080.json
| Kimi-K2-Instruct  (deepseek2)                                          |
| 381.9 GB on disk, 384 experts x 60 layers, top-8                       |
| Exact                3.32 tok/s  (range 2.81 to 3.83)                  |
|   - only 17% of the experts fit in memory; the planner was checked     |
|   against measurements down to 23%, so this estimate is not validated |

# a 22 GB model on a 16 GB machine
$ banach-engine -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f story.txt -n 256 --budget-mb 10000
banach: prefill 18 tok 3.44 s; decode 256 tok 5.40 tok/s, hit 95.7% (page cache 95.7%)

# and the answers are the exact ones (checked for 64 tokens)
$ banach-engine -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf -f story.txt -n 64 --budget-mb 10000 --exact-check
exact-check: PASS byte-identical (64 tokens, logits bit-equal at every step)
```

## How a trillion parameters fit

A Mixture-of-Experts model is huge on disk but small per token: Kimi K2 has a trillion
parameters and uses about 32 billion of them for each token. The trick is to have the right
3% in memory at the right moment.

When the file does not fit, plain llama.cpp leaves it to the operating system, which pages
bytes in 4 KB at a time with no idea which experts come next. Banach knows the model:

<p align="center">
  <img src="brand/how-it-works.svg" alt="One token through Banach: the router picks 8 of 384 experts; 96% are already in the RAM budget and go straight to compute; the rest are fetched whole from NVMe in parallel; often-used experts are kept and cold ones evicted in the background" width="900">
</p>

- **Plan first.** `banach plan` reads a few MB of headers and works out the speed, the hot
  set and the split across GPU, RAM and disk before you download anything.
- **Stream on purpose.** Whole experts, fetched in parallel the moment the router picks them,
  inside a RAM budget you choose. Experts earn their place by how often they are used.
- **Check it.** Streaming is compared with stock llama.cpp, token for token and logit for
  logit: 12 configurations on a small MoE in CI on every commit (Linux, macOS, Windows), and
  64 tokens by hand on the 22 GB model below.
- **Measure shortcuts.** A separate low-precision mode (it does not combine with streaming
  yet) comes with a live fidelity meter, and `banach compare` scores any quantization against
  a reference, so a shortcut is never silent.

## Proof: a model bigger than RAM, measured end to end

Qwen3.6-35B-A3B (Q4_K_M, 22.1 GB) on a 4-core cloud VM (a container with a memory limit) with
16 GB of RAM, no swap and a 1.27 GB/s disk when measured. Greedy decode, 256 tokens, experts
capped at 10 GB. Three runs, each stock vs Banach back to back:

| Run | Prompt processing, stock | Prompt processing, Banach | Generation, stock | Generation, Banach |
|---|---:|---:|---:|---:|
| 1 (32 tokens) | 9.32 s | 3.49 s | 3.54 tok/s | 3.98 tok/s |
| 2 (256 tokens) | 7.64 s | 3.44 s | 4.89 tok/s | 5.40 tok/s |
| 3 (64 tokens, with the exactness check) | 9.30 s | 2.68 s | 3.36 tok/s | 5.44 tok/s |

- **Prompt processing: Banach was 2.2x to 3.5x faster in all three runs.**
- **Generation: Banach was faster in all three runs, by 10% to 62%.** Stock llama.cpp itself
  varied from 3.4 to 4.9 tok/s between runs, so one percentage would claim more than the data
  supports.
- **llama.cpp with its default settings was killed for running out of memory**: its weight
  repacking copies the experts into ordinary RAM. The rows above use it with repacking off,
  the only way it runs on this machine.
- **Output identical** to llama.cpp for the 64 tokens checked, every logit bit for bit.
- Against naive streaming (same 10 GB budget, the kernel paging experts in), Banach decoded
  1.6x faster on the 32-token run and 1.1x on the 256-token run: the gap narrows as the hot
  set fills RAM.

Limits of this evidence: three runs on one machine, the first stock run started with part of
the file in the page cache, and the VM's disk has since been throttled to about 15 MB/s, so
the runs cannot be repeated on it today. To measure on your machine, run
`python3 bench/interleaved.py` (alternating stock and Banach runs, medians reported).
Every number with its machine, commit and command: [docs/VALIDATION.md](docs/VALIDATION.md).

## Get started in one line

```bash
pip install banach
```

The `banach` command and its engine, prebuilt for Linux x86_64, Linux arm64 and macOS on
Apple Silicon. Windows and GPU builds: [from source](#build-from-source).

1. **See the speed before you download:**

   ```bash
   banach plan hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
   ```

2. **Serve it.** `--lazy` starts answering while the file is still downloading:

   ```bash
   banach serve hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --lazy
   ```

3. **Use it** from anything that speaks OpenAI or Anthropic, coding agents included:

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

4. **Watch it live** at `http://127.0.0.1:8080/banach/status`: engine, fidelity meter, download.

## Runs everywhere

| Platform | Backends | Status | Expert streaming |
|---|---|---|---|
| Linux x86_64, arm64 | CPU (AVX2, AVX-512, NEON) | tested, byte-exact in CI | full: hard RAM budget, parallel prefetch, residency checks |
| macOS, Apple Silicon | CPU | tested, byte-exact in CI | prefetch, soft budget |
| Windows x86_64 | CPU | tested, byte-exact in CI; build from source | prefetch, working-set budget |
| NVIDIA, AMD, Intel, Apple GPUs | CUDA, Vulkan, Metal | compiled in CI, **not yet run on a GPU** | dense layers on the GPU (`--gpu-layers N`), experts in RAM |

The CI exactness check is 12 configurations on a small MoE (15M parameters, 16 tokens each).
The GPU rows are build checks only; no benchmark has been run on a GPU yet.

## Everything in the box

| | |
|---|---|
| **Preflight** | speed, hot set and memory split from 16 MB of headers, before any download |
| **Lossless streaming** | expert RAM budget, parallel whole-expert fetch, frequency-gated admission, background eviction |
| **GPU offload** | dense layers and KV cache on CUDA, Metal or Vulkan; experts stay in RAM |
| **Drop-in APIs** | OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`, streaming, tool use, prefix cache |
| **Smart downloads** | resumable, hot experts first, SHA-256 verified; serve before the download ends |
| **Fidelity meter** | shadow replay against the exact model with anytime-valid confidence bounds |

| Command | What it does |
|---|---|
| `banach plan MODEL` | speed and memory split for your machine, from headers |
| `banach compare REF CAND -f TEXT` | how far a quantization or pruned pack drifts from a reference: top-1 agreement, KL, and a 95% bound on flips |
| `banach pick` | ranks the big open MoE models by predicted speed on your machine, biggest fast one first |
| `banach serve MODEL` | local OpenAI and Anthropic server (`--lazy`, `--gpu-layers N`) |
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

- **GPU:** add `-DGGML_CUDA=ON` or `-DGGML_VULKAN=ON` (Metal is on by default on macOS) and
  serve with `--gpu-layers 99`.
- **Windows:** the same steps from a Visual Studio developer prompt; the engine is
  `build-engine\banach-engine.exe`.
- A source build is tuned for your exact CPU and is faster still.

Good to know:
- One request at a time; others queue.
- The hard RAM budget is Linux. macOS and Windows give identical output with a softer budget.
- The low-precision ladder and expert streaming do not combine yet: the ladder needs the model
  in RAM, streaming is lossless only.
- `banach serve` listens on localhost only. For other machines use `--host 0.0.0.0 --api-key KEY`
  (or `BANACH_API_KEY`); clients then send `Authorization: Bearer KEY` or `x-api-key: KEY`.
- Small models (1 to 3B) call tools fine but are weak coding agents.

## Join in

- **Star the repo** so more people find it.
- **Post your numbers.** Every machine and model measured makes the planner sharper for
  everyone. [Open a benchmark issue](https://github.com/gautam1858/banach/issues/new?title=Benchmark:%20MODEL%20on%20MACHINE).
- **Fund the hardware.** The trillion-parameter runs need big NVMe arrays and GPUs:

<p align="center">
  <a href="https://buymeacoffee.com/gautam1858"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy me a coffee" height="52"></a>
</p>

## Why "Banach"?

The Banach-Tarski paradox: cut one ball into a handful of pieces, reassemble them, and you
have two balls the same size as the first. More model than your machine should hold.

## Credits

Built on llama.cpp and ggml (MIT), which run every kernel. Shaped by the llama.cpp
expert-streaming pull requests (25294, 31469) and MoE cache (29887), by colibri's lossless
NVMe streaming and published measurements, and by research on expert locality (Si et al.,
Liang et al.), mixed-precision experts (SliceMoE, DyMoE, HOBBIT) and betting confidence
sequences (Waudby-Smith and Ramdas).

## License

Apache-2.0, see [LICENSE](LICENSE).
