# Qwen3.8-27B at 64K on one RTX 3060 12 GB

This repository publishes the experimental llama.cpp patch and measurements
behind a **29.65-29.69 generated tokens/s fixed C++ workload** and a
**33.21 tokens/s high-acceptance peak** for Qwen3.8-27B while allocating a
65,536-token context on one RTX 3060 12 GB.

The key optimization is **Deferred-Commit Factor Replay (D-CFR)**. It removes
hundreds of MiB of speculative recurrent-state copies, then spends the saved
VRAM on model weights used during target verification.

This is a narrow, disclosed record-candidate result—not a claim that every
prompt runs above 30 tokens/s. Comparisons across quantizations, allocated
contexts, prompts, sampling, and runtimes are not apples-to-apples.

An additional prompt-ingestion sweep improved an 8,595-token RVN IQ3 prompt
from **82.13 to 320.24 prompt tokens/s** on the same RTX 3060 by increasing the
batch and microbatch from 16/16 to a VRAM-safe 128/128. This is a separate
prefill result; it does not change the original generation-throughput claim.

## Results

| Workload | Output | Accepted / drafted | Generated tokens/s |
| --- | ---: | ---: | ---: |
| Fixed palindrome C++ function | 164 tokens | 139 / 168 | 29.65, 29.69 |
| Integers 1 through 100 | 256 tokens | 216 / 233 | 32.78, **33.21** |

The MTP-depth-1 control and MTP-depth-6 Turbo run returned byte-identical text
for the fixed coding workload. Separate fixed-placement tests matched target
output with and without D-CFR for three 128-token responses and one 512-token
response. Internal draft counts can differ because replay changes draft-side
floating-point rounding; the correctness claim is exact target output in the
tested deterministic suite, not bit-identical hidden state or a proof over all
possible executions.

The final graph left about **105 MiB** of VRAM free. This is an experimental
throughput profile with little safety margin, not a general desktop default.

## What changed, in plain English

Qwen3.8 has recurrent Gated Delta Net state. Speculative decoding asks the
model's built-in MTP predictor to guess several future tokens and then makes
the full model verify them. Rejected guesses must not alter the real
conversation state.

The ordinary safe implementation keeps large state backups. At deeper MTP
depths, those backups consume the VRAM needed to keep more target-model weights
on the GPU.

D-CFR works more like a bank ledger:

1. Keep one authoritative committed state.
2. While testing guesses, record the small update ingredients instead of
   copying the complete state after every token.
3. Once verification says which guesses were accepted, replay only those
   accepted updates into the authoritative state.

At MTP depth four, measured process VRAM fell from 10,080 MiB to 9,706 MiB—a
**374 MiB saving**. The fixed 512-token A/B also improved from 13.798 to 14.962
tokens/s, an **8.44% gain**, with identical target output.

The saved memory enabled the larger throughput configuration: MTP depth six,
a 16-token generation workspace, 50 normal GPU layers, and explicit CUDA
residency for target blocks 10 through 16. In short, the runtime stopped
spending scarce VRAM on speculative-state photocopies and spent it on weights
used by the expensive verifier.

## Measured configuration

- GPU: NVIDIA GeForce RTX 3060 12 GB, compute capability 8.6
- CPU: Intel Core i5-12400, 6 cores / 12 threads
- System RAM: 15 GiB
- Model: `Qwen3.8-27B-UD-IQ3_XXS.gguf`
- Model SHA-256: `c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee`
- Allocated context: 65,536 tokens
- KV cache: Q4_0 K and V
- Flash Attention: enabled
- MTP depth: 6
- Batch / microbatch: 16 / 16
- CPU threads: 6
- Normal GPU layers: 50
- Explicit CUDA blocks: 10 through 16
- llama.cpp base: `c060ca974c773c7c3d17fd1b66dc9d312bc292c0`

## Repository contents

- `patches/llama-cpp-dcfr-research.patch`: exact measured research-tree delta
  against the pinned llama.cpp base.
- `scripts/build.sh`: fetches the pinned source, applies the patch, and builds
  the CUDA runtime.
- `scripts/serve.sh`: launches the disclosed 64K Turbo configuration.
- `scripts/serve-rvn-q3.sh`: launches the RVN depth-4 or depth-8 profile with
  a fast-prefill workspace by default.
- `benchmarks/*.json`: machine-readable generation, memory, and prompt-prefill
  measurements, including rejected variants.
- `benchmarks/raw/`: two server logs for the fixed MTP4 A/B.

The patch changes 24 upstream files with 2,105 insertions and 120 deletions.
It intentionally publishes the measured research snapshot rather than
pretending D-CFR existed in isolation. Besides D-CFR/TFR, it contains supporting
recurrent-state and CUDA operations, speculative-server plumbing, phase
profiling, branch verification, and the optional packed-tree experiment. The
packed tree produced exact output in the tested suite but was slower on the
high-acceptance workload, so the final profile leaves it disabled.

## Build

Prerequisites are CMake, a C++ compiler, CUDA with `nvcc`, `curl`, `tar`, and
`patch`. The measured binary used CUDA 13.3, GCC 15, compute capability 8.6,
and Release mode. Override the tool locations as needed:

```bash
CUDA_TOOLKIT_ROOT=/usr/local/cuda \
CC=gcc-15 CXX=g++-15 \
./scripts/build.sh
```

The script refuses to overwrite a non-empty partial source directory. It
applies the patch only while creating a fresh pinned source tree.

## Run

Obtain the GGUF through an authorized source, verify its hash, and pass its
path explicitly:

```bash
sha256sum /path/to/Qwen3.8-27B-UD-IQ3_XXS.gguf

MODEL_PATH=/path/to/Qwen3.8-27B-UD-IQ3_XXS.gguf \
./scripts/serve.sh
```

The server listens only on `127.0.0.1` by default. Set `PORT` to choose another
local port. Other 12 GB cards may require reducing placement because the
measured graph had only about 105 MiB of post-capture headroom.

## Fast prompt ingestion

The original record profile deliberately used a 16-token batch and microbatch
to maximize VRAM available for target-model placement and speculative depth.
That is useful for generation experiments but unnecessarily slow when a chat
client must ingest thousands of tokens before producing its first token.

On the RVN IQ3 64K depth-4 profile, increasing both values produced this
single-run long-prompt comparison:

| Batch / microbatch | Prompt tokens | Prompt speed | Prompt time | Generation speed | Physical VRAM free |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 / 16 | 8,595 | 82.13 tok/s | 104.65 s | 16.12 tok/s | 549 MiB |
| **128 / 128** | 8,595 | **320.24 tok/s** | **26.84 s** | 14.40 tok/s | 449 MiB |
| 256 / 256 | 8,595 | 377.09 tok/s | 22.79 s | 13.57 tok/s | 385 MiB |

The selected 128/128 setting is **3.90x faster at prefill** than 16/16 and
saves about 78 seconds on this prompt while retaining 100 MiB more physical
headroom than 256/256. A separate production-router verification processed a
2,067-token prompt at 311.72 prompt tok/s and generated at 16.52 tok/s.

A 512/512 short-prompt test reached 392.88 prompt tok/s but left only 235 MiB
free, so it was rejected as a display-attached RTX 3060 default. Larger is not
automatically better: stop increasing the microbatch when the speed gain gets
small or physical VRAM headroom becomes unsafe.

The long-prompt rows are one run each. Prompt content, generated content, and
draft acceptance can change slightly with batching because the runtime's
batched floating-point evaluation is not bit-identical. The defensible result
is the measured prefill improvement; the generation figures are included for
visibility, not presented as a controlled decode-speed A/B.

For the tested RVN placement, use the new default:

```bash
MODEL_PATH=/path/to/RVN-IQ3_XXS-multilingual-mtp.gguf \
MTP_DEPTH=4 \
BATCH_SIZE=128 UBATCH_SIZE=128 \
./scripts/serve-rvn-q3.sh
```

To reproduce the earlier 35-run generation suite exactly, restore its smaller
workspace:

```bash
BATCH_SIZE=16 UBATCH_SIZE=16 \
MODEL_PATH=/path/to/RVN-IQ3_XXS-multilingual-mtp.gguf \
MTP_DEPTH=4 ./scripts/serve-rvn-q3.sh
```

The full sweep is recorded in `benchmarks/fast-prefill-20260828.json`.

## Validated RVN IQ3 Zterm presets

Two additional 64K profiles were validated for the RVN uncensored multilingual
MTP checkpoint. These do **not** replace or extend the original record result:
the RVN file is a separate checkpoint, pinned here as
`RVN-IQ3_XXS-multilingual-mtp.gguf` with SHA-256
`338955bd8d67908afb4093b8a7b386dcfcd6eb6184a2171e5d6731c1d296abf6`.

Both generation-validation profiles used D-CFR, Q4_0 K/V,
batch/microbatch 16/16, 47 normal GPU layers, and explicit CUDA placement for
target blocks 10 through 16. The launcher now defaults to the separately
measured 128/128 fast-prefill workspace described above.

| Profile | MTP depth | 35-run mean | Non-easy mean | Full range | Acceptance | Peak GPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default | 4 | 17.27 tok/s | 15.98 tok/s | 11.49-25.07 tok/s | 56.57% | 11,330 MiB |
| Deep-draft experimental | 8 | 19.02 tok/s | 16.04 tok/s | 10.04-36.95 tok/s | 35.05% | 11,342 MiB |

Each profile was run five times on seven 256-token prompts: one easy counting
anchor plus unfamiliar code, multi-step reasoning, factual recall, and prose
continuation. The depth-8 headline mean is lifted by the easy anchor. Across
the 30 non-easy runs it was only 0.4% faster than depth 4, had a lower minimum,
and performed substantially more rejected draft work. Depth 4 is therefore the
recommended everyday default; depth 8 is a workload-dependent experiment.

Run either portable preset with the same patched build:

```bash
# Everyday default
MODEL_PATH=/path/to/RVN-IQ3_XXS-multilingual-mtp.gguf \
MTP_DEPTH=4 ./scripts/serve-rvn-q3.sh

# Optional deeper-draft profile
MODEL_PATH=/path/to/RVN-IQ3_XXS-multilingual-mtp.gguf \
MTP_DEPTH=8 ./scripts/serve-rvn-q3.sh
```

The aggregate measurement and exact configuration are in
`benchmarks/rvn-q3-zterm-presets-20260827.json`.

## Recommended community hardware tests

The published `scripts/serve.sh` is the exact measured RTX 3060 profile, not a
universal auto-tuner. For another GPU, begin with the conservative row below,
confirm that the server is stable, and change only one variable at a time.

| GPU VRAM | Starting quant | Starting context | KV cache | Initial MTP depth | Suggested next test | Status |
| ---: | --- | ---: | --- | ---: | --- | --- |
| 12 GB | `UD-IQ3_XXS` (10.9 GB) | 65,536 | Q4_0 K/V | 4 | Depth 6 if at least 300 MiB remains free | Tested on RTX 3060 at depth 6; other cards untested |
| 16 GB | `UD-IQ4_XS` (14.3 GB) | 32,768 | Q4_0 K/V | 4 | 64K context, then depths 6 and 8 | Recommended candidate; untested here |
| 20 GB | `UD-Q4_K_M` (16.5 GB) | 65,536 | Q4_0 K/V | 4 | Depths 6 and 8 | Recommended candidate; untested here |
| 24 GB | `UD-Q5_K_M` (19.8 GB) | 65,536 | Q4_0 K/V | 4 | Depths 6 and 8, then longer context | Recommended candidate; untested here |
| 32 GB | `UD-Q6_K_XL` (25.3 GB) | 65,536 | Q4_0 K/V | 4 | Depths 6, 8, and 10 | Recommended candidate; untested here |
| 48 GB+ | `Q8_0` (29.0 GB) | 65,536 | Q4_0 K/V | 4 | Depth sweep before increasing context | Recommended candidate; untested here |

Quant sizes refer to the
[Unsloth Qwen3.8-27B GGUF repository](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF).
Use the exact same checkpoint and hash when comparing configurations. Quants
smaller than `UD-Q2_K_XL` may require the separate MTP file; the table avoids
that complication.

For every tier:

1. Start with batch and microbatch 128 on a 12 GB Q3 profile, Flash Attention
   enabled, one parallel sequence, and CPU threads equal to physical CPU
   cores. Try 256 on a 16 GB card and 512 on a 24 GB or larger card, but treat
   those larger-card values as untested starting points rather than results.
2. Keep all weights on the GPU when possible. If they do not fit, use the
   smallest CPU/RAM spill that starts reliably; RAM offload normally trades
   speed for capacity.
3. Leave 300-500 MiB of VRAM free, and more on a display-attached GPU. Reduce
   context or MTP depth first if graph capture fails or the process OOMs.
4. Sweep MTP depth on a representative workload. A deeper draft is not
   automatically faster when acceptance is low.
5. Report GPU, driver/runtime, quant filename and SHA-256, allocated context,
   sampling parameters, prompt and generated token counts, generated tokens/s,
   accepted/drafted tokens, peak VRAM, and any RAM offload.

For a publishable comparison, use the same nontrivial prompts and seeds for
each depth, run each cell five times, and report the mean and full range rather
than only the fastest sample. These rows are starting hypotheses, not measured
performance claims.

## Benchmark evidence and limitations

The JSON files preserve exact speeds, memory measurements, draft acceptance,
correctness scope, and rejected configurations. The raw server logs support
the ordinary-versus-D-CFR MTP4 512-token A/B.

The final 29.7/33.2 tokens/s runs were retained as structured measurement
records rather than separate raw server logs. This repository reproduces the
runtime configuration and optimization; it is not a bit-for-bit benchmark
harness.

## Public comparison

The closest public single-RTX-3060 result found during the August 27, 2026
review reported about 9.7 tokens/s at a 96K allocation using Q4_K_S, with a
10.6 tokens/s peak. A public dual-RTX-3060 test reported about 26.6-27.8
tokens/s. Both use different configurations and workloads:

- [Single RTX 3060 12 GB, Q4_K_S at 96K](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/61)
- [Dual RTX 3060 comparison](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/discussions/77)

The defensible claim is narrow: this is the strongest public Qwen3.8-27B
64K-class single-RTX-3060 12 GB result found in that review, on the disclosed
workloads and settings. Independent reproduction is welcome.

## Safety

This is experimental inference-engine code. The Turbo placement operates near
the VRAM limit and may OOM on a display-attached GPU or a card with less usable
memory. Review the patch before running it, keep the server bound to localhost,
and do not use the experimental state paths where numerical or operational
failure would be safety-critical.
