# Q3 cache-safe D-CFR: October 5 results

The primary comparison is the **same RCO GGUF**, same Q4_0 KV and 128K
allocation, same depth-4 MTP and requests. It compares the previous complete
serving configuration with the new one. The model SHA is
`475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039`.
Do not combine this table with the earlier multilingual RVN or Unsloth results.

## Five-seed comparison

Five runs per fixture, seeds 101, 202, 303, 404, 505. Rust/C++ produce 128
tokens; reasoning produces 256. Outputs are fixed-length throughput samples,
not necessarily finished solutions. EOS is ignored for the timing experiment.

| Fixture | Baseline mean tok/s | Candidate mean tok/s | Baseline range | Candidate range | Acceptance, baseline / candidate |
|---|---:|---:|---:|---:|---:|
| Rust | 22.939 | 26.358 | 21.446–24.166 | 24.438–27.649 | 77.063% / 77.063% |
| C++ | 20.966 | 24.013 | 19.731–22.439 | 22.618–26.008 | 68.342% / 68.156% |
| Reasoning | 15.900 | 18.343 | 13.492–16.863 | 15.703–19.519 | 44.790% / 45.132% |

Sample standard deviations, baseline/candidate: Rust 1.004/1.212,
C++ 1.043/1.295, reasoning 1.421/1.600 tok/s. Acceptance is the arithmetic
mean of per-run accepted/drafted ratios, not accepted tokens divided by output
tokens. Every speed has its corresponding raw draft counts in the JSON.
All 15 paired output messages matched exactly; draft counts were not identical
in every pair. Output equality is evidence for these fixtures, not a universal
equivalence proof or a model-quality evaluation.

Coding sampler: temperature 0.7, top-p 0.95, top-k 40, min-p 0.05,
thinking off. Reasoning: temperature 1.0, top-p 0.95, top-k 20, min-p 0,
repeat penalty 1, presence penalty 0, thinking on at medium. Both preserve
thinking history and bypass input caching for the timed fixtures. These are
benchmark settings, distinct from the interactive launcher's defaults.

## What changed between arms

| Setting | Baseline | Candidate |
|---|---|---|
| GGUF, template, context, KV | Same | Same |
| MTP depth, CPU threads, batch/ubatch | 4, 6, 128/128 | 4, 6, 128/128 |
| Rollback at depth 4 | Ordinary state backups | Complete-checkpoint D-CFR |
| Host gate/up weight blocks | 54 | 46 |
| Queued transfer buffers | 3 x 33 MiB | 3 x 48 MiB |
| Minimum sampled free VRAM | 356 MiB | 366 MiB |

The old daily depth-4 route did **not** activate compact replay. The older
compact path at deeper depths cleared prompt caches because checkpoints omitted
pending factors. The new format fixes that omission and explicitly enables
shallow compact replay. The freed memory pays for the changed placement and
larger buffers. The headline gain is therefore a **combined configuration gain**.

This is not the full three-arm, matched-placement depth sweep needed to isolate
D-CFR from ordinary checkpointing and an optimized no-speculation baseline.
One duplicate-flag "plain" probe actually left speculation enabled and was
discarded. A corrected single-seed no-speculation check was not optimized for
its spare VRAM and is also excluded from headline comparisons.

## Occupied context

Same synthetic record generator and Rust prompt; one run per cell:

| Allocated / input tokens | Arm | Prefill tok/s | Decode tok/s | Accepted / drafted | Wall seconds |
|---|---|---:|---:|---:|---:|
| 131,072 / 31,133 | Baseline | 353.23 | 20.68 | 95 / 126 | 94.30 |
| 131,072 / 31,133 | Candidate | 381.99 | 24.04 | 95 / 126 | 86.81 |
| 131,072 / 121,821 | Candidate | 272.29 | 19.14 | 98 / 115 | 454.09 |

The 31K pair was run candidate-first, reversing the primary configuration order.
The near-full test has no same-prompt baseline here. A brief CPU-side router
build overlapped part of that stress test, so treat it as an operational fit
screen rather than a pristine comparative benchmark. All emitted 128 tokens,
completed without truncation/OOM, and passed a subsequent prompt-cache check.
These tests do not establish retrieval or reasoning quality at 121K input.

The 96K single-seed screen measured Rust 26.09 -> 29.64, C++ 25.26 -> 28.83,
and reasoning 15.22 -> 17.35 tok/s. It is **not** a five-run average. Raw
acceptance counts and timings are retained in the [JSON](../benchmarks/q3-cache-safe-20261005.json).

## Correctness and interactive checks

- Exact recurrent-state, convolution-snapshot, factor and metadata round trips.
- Shared-sequence metadata consistency; rejection of malformed/truncated state
  and rollback requests beyond retained factors.
- 20 actual-model CUDA checkpoint/continuation comparisons at depths 4, 5 and 8:
  maximum observed restored-logit difference **0**.
- Cross-request cache reuse, identical requests, fresh-reference requests,
  edited prefixes, tool calls/results, independent chats and streamed cancellation.
- Local shared-router 96K/128K checks: configured runtime/placement, preserved
  output allowances, reasoning/final answer, structured tools, OCR and image input.

The real-model regression allows a maximum logit difference of 1e-5; all
observed differences were exactly zero. Its comparison is restored versus
uninterrupted **D-CFR** continuation, not hidden-state equivalence to ordinary
rollback. Tests and synthetic fixtures are included in this repository.

## Reproduce and interpret

Use [the Q3 build and launcher](q3-quickstart.md). Build candidate and baseline
separately, run only one at a time, and keep the GGUF/template hashes fixed.
Run `scripts/bench-q3.py` against each dedicated server. For the occupied tests:

```bash
python3 scripts/bench-q3.py --seeds 101 --occupied-records 1024 --output build/occupied31k
python3 scripts/bench-q3.py --seeds 101 --occupied-records 4100 --output build/occupied121k
```

Historical runs used separately built libraries from the same pinned source
family: GCC 15/CUDA 13.3 for the server/CUDA runtime and GCC 16 for the CPU
backend. The release assembles the exact changed source files into one build,
using the selected compiler. Source parity is not a claim of binary identity
or guaranteed identical timing under another toolchain. Consult the
[release verification record](../benchmarks/q3-release-verification-20261005.json)
for the standalone package's actual validation scope.

Memory figures are minimum whole-device free VRAM sampled once per second,
not an exact allocator peak. New benchmark runs also record sampled peak used
VRAM. Other desktop activity, PCIe bandwidth, CPU/RAM, compiler, thermals,
prompt acceptance and occupied context all affect results. No maximum-feasible
depth/OOM sweep was completed for this release. No world-record claim is made.
