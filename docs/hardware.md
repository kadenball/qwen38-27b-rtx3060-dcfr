# Hardware support and community testing

Support labels refer to a particular runtime, model and configuration, not a
GPU's ability to run upstream llama.cpp in general.

| Label | Hardware/configuration | Evidence |
|---|---|---|
| Locally tested | RTX 3060 12 GB, i5-12400, 16 GB RAM, Linux; RCO Q3 at 96K/128K | [October 5 measurements](q3-cache-safe.md) |
| Community report, historical | Similar setup with Ryzen 5 5600G; older release | [Issue #1](https://github.com/kadenball/qwen38-27b-rtx3060-dcfr/issues/1) reports 14.82 tok/s on its coding task; not a reproduction of the new configuration |
| Experimental | Other RTX 30/40/50-series cards and different CPUs | Rebuild and retune; no new-release local validation |
| Experimental | Windows/WSL, multi-GPU | Current scripts target Linux and one CUDA device |
| Not validated | AMD, Intel or Apple GPU backends | Custom CUDA-specific paths need separate compatibility work/testing |

The historical community report is useful precisely because its speed differs.
Its workload, checkpoint/configuration and occupied context are not a matched
pair with this release. Earlier user-supplied RTX 5070 reports are not treated
as verification of the October 5 patch.

## Why another GPU may benefit differently

D-CFR saves recurrent rollback storage for the supported architecture. The
use of that saving depends on the machine: more GPU weights, more workspace,
or potentially more context. This release spends it on placement and transfers
while keeping context fixed. It does not validate context beyond 131,072.

Queued transfers matter when weights cross PCIe. If a larger GPU already holds
all weights, that component may offer little benefit. CPU instruction support,
RAM bandwidth, PCIe link width, display load and draft acceptance also matter.
An expensive GPU is not a guarantee of the same percentage improvement.

## Starting experiments, not promised presets

| Hardware tier | Starting action |
|---|---|
| Less than 12 GB | Do not copy the published 96K/128K placement. Establish a smaller-context upstream baseline first; fitting this release needs separate tuning. |
| 12 GB NVIDIA | Start with the 96K Q3 preset, one model process and no competing heavy GPU app. Increase host blocks if headroom is insufficient. |
| 16 GB+ NVIDIA | Rebuild natively, begin with the same Q3 fixture/configuration, then reduce host blocks gradually while watching memory. |
| CPU without AVX2/FMA | Do not expect the custom x86 multi-query speedup. Buildability and fallback behavior need validation. |
| Other GPU vendors | Treat as backend-porting work, not a flag-only installation. |

These are test hypotheses. They are not purchase recommendations, confirmed
context limits or predicted throughput. Higher-bit model presets are separate
experiments; the current launcher verifies the exact RCO IQ3 model hash.

For another NVIDIA architecture, use `CUDA_ARCHITECTURES=native` on the target
machine or set the appropriate compute capability. The project does not ship
a universal 3060 binary. See [llama.cpp's CUDA build instructions](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#cuda)
and [NVIDIA's architecture table](https://developer.nvidia.com/cuda/gpus).

## A useful report

Record the GPU/VRAM, CPU, RAM, OS, driver/CUDA/compiler versions, git revision,
GGUF and template hashes, all launch overrides, allocated **and occupied**
context, input/output lengths, sampler and thinking mode. For each fixture,
report five-run mean/range, accepted/drafted counts, prefill and decode speed,
time to first token, and sampled peak VRAM. Include cold and cached follow-ups.

Use the same prompts/seeds and hashes for each arm. Change one variable at a
time when isolating a mechanism. Test tool calls, cancellation and cache reuse
before using an experimental profile for a long work session. Report OOMs and
regressions, including the highest depth that actually completed; do not infer
the maximum depth from a single success.

[Open a hardware report](https://github.com/kadenball/qwen38-27b-rtx3060-dcfr/issues/new?template=hardware-report.yml).
