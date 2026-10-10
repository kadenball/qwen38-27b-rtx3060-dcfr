![RouteWeaver](docs/assets/routeweaver-banner.png)

# RouteWeaver

**Local inference, further.**

Experimental inference-engine optimizations for running large language models
on memory-constrained hardware. Built on llama.cpp: cache-safe speculative
decoding, queued CPU-to-GPU weight transfers, selective weight placement, and
focused CPU/CUDA kernels.

[Run Q3](docs/q3-quickstart.md) · [Q3 Fast experiment](docs/q3-fast.md) ·
[Hardware](docs/hardware.md) · [How it works](docs/architecture.md) ·
[Contribute](CONTRIBUTING.md)

## Latest: opt-in Q3 Fast, one RTX 3060 12 GB

The October 8 experimental runtime combines probability-based MTP verification,
grow-on-demand KV memory, evictable GPU weight copies and tuned SM86 kernels.
Three-seed short-fixture means are **48.61 tok/s Rust, 41.31 C++ and 34.64
reasoning**. A broader seven-task screen averages **37.03 tok/s** by timed
generation tokens/time, excluding its easy counting anchor; that screen has only one seed.

These are throughput screens, not coding-quality scores. The window is **128K
capacity**, not 128K occupied: the new 30,425-input-token check reaches **35.57
tok/s**, while a full 128K occupied run remains unvalidated. Speeds fall as
context grows and weight copies give VRAM back to KV storage.

[Build/run Fast, exact settings, ranges, acceptance and limitations](docs/q3-fast.md)
· [Measured runs](benchmarks/q3-fast-20261008.json)

Fast is a separate `Q3_VARIANT=fast` build. The default launcher and automatic
tuner below remain on the October 5 candidate; existing profiles are unchanged.

## Default release: Qwen 27B Q3, 128K context, one RTX 3060 12 GB

The October 5 configuration improved generation throughput by **14.5–15.4%**
over the previous same-GGUF setup across three short fixtures, five seeds each.
No change to the weights, allocated context, Q4_0 KV, draft depth or sampling
between the paired arms.

Mean generated tok/s; parentheses show mean accepted/drafted ratio:

| Fixture | Previous setup | Cache-safe setup | Change |
|---|---:|---:|---:|
| Rust coding | 22.94 (77.1%) | **26.36 (77.1%)** | +14.9% |
| C++ coding | 20.97 (68.3%) | **24.01 (68.2%)** | +14.5% |
| Reasoning | 15.90 (44.8%) | **18.34 (45.1%)** | +15.4% |

All 15 paired output messages matched exactly. These are throughput fixtures,
not coding-quality scores or a promise that every conversation reaches these
speeds. The measured improvement combines D-CFR, placement and buffer changes;
it is not an isolated D-CFR-only speedup.

A separate synthetic stress test processed **121,821 input tokens** in a
131,072-token window and then generated at **19.14 tok/s**. It demonstrates
memory fit and operation near a full context, not long-context reasoning
accuracy. The lowest sampled free VRAM was **366 MiB**; other GPU applications
can exhaust that margin.

[Full ranges, acceptance, occupied-context tests and limitations](docs/q3-cache-safe.md)
· [Machine-readable measurements](benchmarks/q3-cache-safe-20261005.json)

## Start here

The current Q3 release targets the hash-pinned
**RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf**, not every Qwen quant.

### Guided setup and automatic tuning (experimental)

For Linux x86-64 with AVX2/FMA and one NVIDIA GPU with at least 12 GB VRAM.
Install the [build prerequisites](docs/q3-quickstart.md#requirements) first;
the setup command detects a compatible installed CUDA/compiler pair.

```bash
git clone https://github.com/kadenball/qwen38-27b-rtx3060-dcfr.git
cd qwen38-27b-rtx3060-dcfr
./routeweaver setup --context 98304
./routeweaver start --context 98304
```

Use `--context 131072` on both commands for 128K. Setup verifies/downloads the
model, builds locally, tests a bounded set of settings, and saves a validated
profile. It keeps your context, quant and Q4_0 KV fixed. Existing GPU compute
processes must be stopped by their owner; setup never closes them automatically.

The default tuning budget is 15 minutes, **excluding download/build and
model-free checks**. A failed or inconclusive tune does not replace an existing
profile. This is the best passing configuration sampled, not a guaranteed
global optimum. A real 96K tuning session passed on the RTX 3060 development
machine; other hardware remains experimental. See
[validation scope and full instructions](docs/automatic-setup.md).

### Manual installation

Prefer explicit settings? This remains a fully independent installation path:

```bash
git clone https://github.com/kadenball/qwen38-27b-rtx3060-dcfr.git
cd qwen38-27b-rtx3060-dcfr
./scripts/download-q3.sh

# Point these at a CUDA installation and supported host compilers.
CUDA_TOOLKIT_ROOT=/usr/local/cuda CC=gcc-15 CXX=g++-15 \
  ./scripts/build-q3.sh
./scripts/check-q3.sh

# 96K default; use CONTEXT=131072 for the 128K preset.
./scripts/serve-q3.sh
```

Tested hardware: RTX 3060 12 GB, Core i5-12400, 16 GB RAM, Fedora Linux.
Stop other GPU model servers first. Read the
[complete setup, vision, client connection and rollback guide](docs/q3-quickstart.md).
The server binds to localhost and exposes an OpenAI-compatible API.

## Where the default release's gain comes from

- **Cache-safe D-CFR:** preserve the recurrent base, pending updates and rollback
  metadata together, making compact speculative state compatible with prompt reuse.
- **More GPU-resident weights:** spend the recovered VRAM on eight additional
  blocks' gate/up weights at the same context allocation.
- **Queued transfers:** three 48 MiB staging regions overlap eligible host-weight
  transfers with execution and avoid a redundant device copy.
- **Specialized kernels:** reuse IQ3 decoding across small CPU query batches and
  reduce Q4_0 attention workspace on the measured CUDA path.

The fresh release builds these components together from pinned source.
Historical measurements used separately built runtime libraries; toolchain and
packaging details are disclosed in the [results](docs/q3-cache-safe.md).
This repository is an experimental fork/patch set, not an upstream llama.cpp release.

## Hardware support is evidence-based

This table describes the October 5 default. The newer Fast snapshot has its own
[narrower validation scope](docs/q3-fast.md#limits-and-correctness-scope).

| Status | Scope |
|---|---|
| **Locally tested** | RTX 3060 12 GB + i5-12400, this Q3 model, 96K/128K |
| **Community reports** | Older configurations; separate from validation of this release |
| **Experimental** | Other NVIDIA GPUs, CPUs, VRAM sizes and operating systems |
| **Not validated** | AMD/Intel/Apple GPU backends, multi-GPU, other model architectures |

Expect to rebuild and retune for another machine. D-CFR is architecture-specific;
prefetch gains depend on how much work crosses the CPU/GPU boundary.
[Compatibility, starting tests and report requirements](docs/hardware.md)

## Project map

- [Q3 quickstart](docs/q3-quickstart.md): download, build, serve, vision and regression tests.
- [Automatic setup](docs/automatic-setup.md): hardware detection, bounded tuning, profiles and rollback.
- [Current results](docs/q3-cache-safe.md): repeat measurements and what they do not establish.
- [Q3 Fast experiment](docs/q3-fast.md): optional new runtime, source export and October 8 results.
- [Architecture](docs/architecture.md): checkpoints, memory placement and transfer scheduling.
- [Q4 interactive guide](docs/q4-quickstart.md): the separate historical 32K IQ4_XS release.
- [Earlier 96K/128K experiment](docs/iq3-large-context.md): different checkpoint and date.
- [Historical landing page](HISTORY.md): original 64K D-CFR work and older benchmarks.
- [Changelog](CHANGELOG.md) and [benchmark index](benchmarks/README.md).

The repository URL stays unchanged so existing links, issues and history keep working.
RouteWeaver is the project brand; D-CFR names the recurrent-state technique.

## Contribute

Independent reproduction matters. Share matched prompts and seeds, model hashes,
draft acceptance, occupied context, memory and timings. Report slower results too.
See [CONTRIBUTING](CONTRIBUTING.md) or open a
[hardware report](https://github.com/kadenball/qwen38-27b-rtx3060-dcfr/issues/new?template=hardware-report.yml).

Code is [MIT licensed](LICENSE). Upstream and model credits are in [NOTICE](NOTICE.md).
