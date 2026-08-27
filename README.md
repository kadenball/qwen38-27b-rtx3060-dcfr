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
- `benchmarks/*.json`: machine-readable measurements and rejected variants.
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
