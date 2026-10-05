# How RouteWeaver works

The current release targets Qwen's hybrid recurrent architecture and its MTP
draft path. It is not a new model or quantization algorithm. Model weights stay
unchanged; the runtime changes how scarce memory and computation are scheduled.

## 1. Compact rollback without losing the prompt cache

Speculative decoding predicts a short chain and verifies it in a target pass.
Rejected tokens must not remain in recurrent state. Ordinary rollback stores
large state copies. Deferred-Commit Factor Replay (D-CFR) instead keeps a
committed state plus compact update ingredients, committing only accepted work.

Across a request boundary, saving just the committed state is insufficient:
there can still be pending updates and a rollback position. The complete
checkpoint serializes the recurrent base, convolution snapshots, factor
tensors, rollback index and pending count as one versioned snapshot. It checks
layout/depth compatibility and rejects incomplete or inconsistent snapshots.

The server retains prompt reuse only on the explicitly enabled complete linear
replay path. Packed speculative trees and inverse-compact rollback are excluded
from these presets. An out-of-range rollback is rejected rather than attempting
to replay factors that were not retained. Old snapshot formats are not portable
into the new D-CFR format.

Sources: `src/llama-memory-recurrent.cpp` and `tools/server/server-context.cpp`
in `patches/llama-cpp-q3-cache-safe.patch`.

## 2. Spend the memory saving on useful weights

At fixed context, the candidate moves eight additional blocks' gate/up weights
from host placement to GPU placement. Attention, recurrent work and FFN down
projections were already GPU-resident. This reduces how much data must cross
the CPU/GPU boundary during verification.

The placement presets are measured hardware choices, not architecture-wide
constants. More VRAM can support fewer host blocks; a different desktop load
can require more. Quantized KV reduces attention-cache size but does not make
allocated or occupied context free.

## 3. Queue the remaining transfers

The scheduler keeps three staging regions on the GPU and queues up to two
future eligible weight transfers. Completion events coordinate readers and
buffer reuse. For safe contiguous matrix-multiply inputs, execution reads the
staged weight directly, avoiding a second device copy.

Alias eligibility checks allocation bounds, views, operation type, later
readers and storage overlap. Unsupported cases take the ordinary scheduler
path. The new 48 MiB regions admit the RCO tensors that exceeded the previous
33 MiB limit. Partial initialization and cleanup have a focused regression.

This implementation explicitly targets a single CUDA backend plus CPU. Its
CUDA-host buffer assumptions are why another vendor's backend is not marked
supported merely because upstream llama.cpp supports that vendor.

Sources: `patches/llama-cpp-q3-prefetch.patch` and the buffer-size portion of
`patches/llama-cpp-q3-cache-safe.patch`.

## 4. Reuse quant decoding and attention workspace

The IQ3_XXS AVX2 multi-query kernel decodes a quantized weight once for a small
group of queries. Integer sums and floating-point accumulation order are kept
aligned with the single-query reference on the tested path. It is guarded by
shape/type checks; other shapes use the ordinary CPU path.

The inherited CUDA head-split attention path reduces Q4_0 workspace. These
pieces already contributed to earlier configurations; the latest 15% table
does not attribute the entire gain to either kernel alone.

## What this does not establish

The release does not prove better model intelligence, identical floating-point
state for all executions, speedups on every GPU, or D-CFR gains on unrelated
architectures. It also does not make deeper drafting automatically better.
Acceptance, transfer cost and rollback cost determine the useful depth.
See the [benchmark scope](q3-cache-safe.md) and [hardware matrix](hardware.md).
