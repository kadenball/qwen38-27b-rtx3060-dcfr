# Q3 Fast: experimental 128K runtime

October 8 measurements; source export published October 10. This is an opt-in
successor to the [October 5 cache-safe runtime](q3-cache-safe.md). The existing
manual launcher and automatic tuner still select the earlier runtime by default.

## What changed

- Probability-based MTP verification, ported from llama.cpp
  [PR 27694](https://github.com/ggml-org/llama.cpp/pull/27694), plus request-boundary
  draft-state and replay-RNG fixes for this older engine.
- Grow-on-demand physical CUDA KV storage beneath a fixed 131,072-token logical
  capacity. This is allocation management, not attention pruning or compression.
- An evictable cache of immutable host weights in spare VRAM. Context growth
  evicts copies and falls back to the existing queued-transfer path.
- Shape-specific SM86 matrix-vector kernels that reuse quant decoding, and
  fewer redundant synchronization points in the weight-transfer scheduler.
- An initial graph-workspace reservation of 4,096 tokens, growing when needed.
  This does **not** limit the actual context to 4K.

The largest isolated earlier step was sparse KV plus resident weights:
about 34–35% more short-fixture generation throughput than the same
probabilistic engine without that cache. Later kernel and workspace changes
add to that. Neither the overall gain nor the final 5% increment below is a
D-CFR-only result. The probabilistic algorithm is upstream work, not a new
sampling algorithm invented by RouteWeaver.

## Exact preset

RTX 3060 12 GB (SM86), i5-12400, 16 GB system RAM, Fedora Linux; six threads,
one slot, batch/ubatch 128, draft depth four, Q4_0 target and draft K/V, 46 blocks'
gate/up weights assigned to CPU, remaining eligible weights on GPU.

The GGUF and fixed v22.4 chat template are unchanged from the October 5 release:

```text
RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf
475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039

qwen-fixed-v22.4.jinja
c47c82b0544752d454f4e427228d9d9d8c3df64c9e446cbd0229362f67948009
```

The launcher enables complete D-CFR checkpoints, sparse KV, a 2,560 MiB
resident-weight cap, row-tile mode six, defer-sync mode two and a 4,096-token
initial scratch reservation. CPU IQ3 multi-query width is six; operation
offload minimum batch remains two. All inherited engine experiment flags are
cleared first. [The exact environment](../scripts/q3-env.sh) is source-pinned.

## Build and run

Follow the [same prerequisites and download instructions](q3-quickstart.md).
Fast uses its own source and build directories; it does not overwrite the
candidate, baseline or Q4 binaries. A full source build replaces the layered
experimental libraries used for the measurements.

```bash
./scripts/download-q3.sh
Q3_VARIANT=fast CUDA_TOOLKIT_ROOT=/usr/local/cuda CC=gcc-15 CXX=g++-15 \
  ./scripts/build-q3.sh
Q3_VARIANT=fast ./scripts/check-q3.sh

# Inspect the command without loading a model.
Q3_VARIANT=fast ./scripts/serve-q3.sh --print-config

# Stop any other server using the GPU before starting this one.
Q3_VARIANT=fast ./scripts/serve-q3.sh
```

Use the CUDA-supported compiler pair installed on your system; the compiler
names above are examples, not an instruction to replace your system compiler.
The API base is `http://127.0.0.1:8080/v1`, model alias
`routeweaver-q3-fast-131072`. Configure your harness to use that endpoint and
alias. Serve defaults are temperature 1, top-p 0.95, top-k 20, min-p 0 and at
most 65,536 output tokens; requests may override sampling and output length.
Input and output together must fit the context. This preset is text/tools only;
use the older candidate for the validated vision path.

To roll back, stop the Fast server and run `CONTEXT=131072 ./scripts/serve-q3.sh`.
Do not run both on the same 12 GB GPU. The automatic tuner has **not** been
extended to search the Fast memory/cache settings.

## Measurements

These are historical measurements of the selected experimental runtime,
not a fresh performance rerun of the public source export. Context **capacity**
was 128K; most inputs were short. Reported tok/s is the engine's generation
throughput, not end-to-end throughput or prompt ingestion speed.

Three seeds (101, 202, 303), 128 output tokens for coding and 256 for reasoning:

| Fixture | Fast mean tok/s | Min–max | Sample SD | Accepted/drafted |
|---|---:|---:|---:|---:|
| Rust, 943 input tokens | 48.61 | 47.16–50.30 | 1.58 | 80.1% |
| C++, 86 input tokens | 41.31 | 38.42–42.87 | 2.51 | 62.9% |
| Reasoning, 72 input tokens | 34.64 | 33.61–36.11 | 1.30 | 48.4% |

Coding uses temperature 0.7/top-k 40/min-p 0.05; reasoning uses temperature
1/top-k 20/min-p 0. Both use top-p 0.95. These are the existing
[short fixtures](../benchmarks/q3-fixtures.json), not completed coding tasks.

A broader eight-prompt screen used one seed, 512 output tokens each,
temperature 0.7/top-k 40/top-p 0.95/min-p 0.05 throughout:

| Fixture | Fast tok/s | Accepted/drafted |
|---|---:|---:|
| C incremental varint decoder | 40.40 | 59.7% |
| Python topological sort | 38.61 | 56.5% |
| Rust double-ended slice iterator | 39.98 | 60.6% |
| C++ integer geometry | 41.03 | 62.1% |
| Crash-recovery reasoning | 36.37 | 52.9% |
| Factual comparison | 37.86 | 55.5% |
| Prose continuation | 28.51 | 35.6% |
| Easy counting sanity anchor | 58.10 | 99.3% |

The seven non-anchor prompts aggregate to **37.03 tok/s**, computed as timed
generated tokens divided by total generation time (the engine excludes the
first output token of each request: 511 timed tokens per 512-token output).
One seed gives no estimate of run-to-run spread. The easy anchor is excluded
from that aggregate. This is
not evidence of 40 tok/s across general workloads or of improved answer quality.

The final cache/scratch adjustment improves the three short means by
4.7–5.3% over the immediate previous experimental configuration (already using
the new verifier, cache and kernels). All nine paired output hashes and draft
counters match. The broader screen rises from 35.28 to 37.03 aggregate tok/s
excluding the anchor, with all eight paired output hashes/counters matching.
Those matches do not imply identical fixed-seed text between probabilistic and
ordinary sample-and-match verification.

At **30,425 actual input tokens**, one matched Rust run reaches **35.57 tok/s**
with 82.9% draft acceptance, versus 34.55 before the final adjustment. Prefill
is 384.36 versus 391.05 tok/s; whole-request time is 82.75 versus 81.50 seconds.
That is a decode improvement, **not** an end-to-end improvement.

Sampled peak whole-device VRAM is 11,292 MiB for the short suite, 11,284 MiB
for the broader suite, and 11,394 MiB for the 30K test. Minimum sampled free
VRAM is 598, 606 and 496 MiB respectively. One-second sampling can miss peaks;
these numbers include other GPU applications, not just engine allocations.

[Machine-readable runs, comparisons, hashes and acceptance](../benchmarks/q3-fast-20261008.json)
and [broader fixture definitions](../benchmarks/q3-fast-heldout-fixtures.json)
are included. To repeat against a dedicated idle server:

```bash
python3 scripts/bench-q3.py --url http://127.0.0.1:8080 \
  --seeds 101,202,303 --output build/fast-short
python3 scripts/bench-q3.py --url http://127.0.0.1:8080 \
  --fixture benchmarks/q3-fast-heldout-fixtures.json \
  --seeds 101 --output build/fast-broader
python3 scripts/bench-q3.py --url http://127.0.0.1:8080 \
  --occupied-records 1000 --seeds 101 --output build/fast-occupied
```

Output directories must not already exist. Benchmarks disable prompt reuse and
force the stated output lengths. This is intentionally different from normal
interactive use, which should permit stopping and prompt-cache reuse.

## Limits and correctness scope

- **Full 128K occupied context is not validated for this new allocator.** The
  earlier candidate's 121,821-token test cannot be transferred to this build.
  The new structured occupied-context result covers about 30K input tokens.
- Committed KV pages and grown scratch allocations do not shrink on chat reset.
  A new short chat after a long one can remain slower until the model reloads.
  Longer attention and fewer resident weights naturally reduce generation speed.
- Host weights are retained; the GPU cache does not reduce system-RAM needs.
  A nominal 512 MiB admission reserve is not a guarantee against OOM or other
  applications consuming VRAM. Do not infer support for GPUs below 12 GB.
- Only one CUDA GPU and one slot are validated. SM86 shape-specific kernels
  fall back for other dispatches, but other GPUs/OSes, multi-GPU and vision
  remain unvalidated for Fast.
- These Fast screens do not establish its speedup over plain decode. A matched
  no-speculation/checkpoint/D-CFR comparison on the broader prompts remains
  separate work; the published final-step control already uses the new engine.
- Grammar/schema requests, greedy targets and unsupported sampler modes retain
  ordinary verification. The target-distribution formula assumes correct p/q
  distributions and state; finite checks are not a proof of every combination.
- The selected experimental binaries passed seeded cached/fresh repeats,
  schema fallback, prefix edits, chat switching and tool/cancellation checks.
  Inherited cancellation tests use greedy sampling; stochastic cancellation
  and long stochastic checkpoint replay remain outside those checks.
- The model-free sampler test exercises the real verifier with synthetic
  logits: 1.2 million marginal and 1.2 million two-token trials, plus reset,
  clone and copy. It is included in `Q3_VARIANT=fast ./scripts/check-q3.sh`.

Source is pinned in [Q3_FAST_SOURCE_SHA256SUMS](../patches/Q3_FAST_SOURCE_SHA256SUMS).
The older stable-by-comparison installation path remains available while this
snapshot receives independent reproduction and broader validation.

## Export verification, October 10

The source overlay applies with zero fuzz to the pinned archive and matches all
36 modified-file hashes. A fresh CPU-only server build and the CPU, cleanup,
checkpoint and verifier checks pass, along with 12 packaging/result tests and
26 installer tests. [Verification record](../benchmarks/q3-fast-export-verification-20261010.json).
A fresh full CUDA build and model-loaded GPU rerun were **not** performed for
this export; the CPU checks do not validate CUDA allocation or kernel behavior.
