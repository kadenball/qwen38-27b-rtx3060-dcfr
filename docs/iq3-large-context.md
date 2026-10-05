# Q3 at 96K and 128K on RTX 3060 12 GB

**Historical September 20 experiment.** The scheduler and later cache-safe work
are now packaged in the [October 5 Q3 release](q3-quickstart.md), which uses a
different, hash-pinned RCO checkpoint. The availability statement below records
the status of this older experiment, not the current repository.

September 20, 2026 experimental configuration screen. Hardware: RTX 3060
12 GB, Core i5-12400, 16 GB system RAM. Model: RVN multilingual MTP
IQ3_XXS Qwen3.8-27B, not the Unsloth checkpoint in the historical benchmark.

## Results

| Allocated context | Configuration | Rust tok/s | C++ tok/s | Reasoning tok/s |
|---|---|---:|---:|---:|
|98,304|Previous hybrid|13.55|13.41|9.22|
|98,304|Queued selective offload|22.00|21.29|16.53|
|131,072|Previous hybrid|9.18|8.34|7.50|
|131,072|Queued selective offload|20.89|19.70|15.23|

One run per task/configuration, seed 101; coding outputs 128 tokens,
reasoning 256. These are not repeated averages. Coding temperature 0.7;
reasoning temperature 1.0. Draft acceptance and outputs can differ between
configurations, so this is not an isolated kernel speedup or quality comparison.

| Allocated context | Input tokens | Prompt tok/s | Decode tok/s | Accepted/drafted | Total seconds |
|---|---:|---:|---:|---:|---:|
|98,304|89,392|317.88|17.57|96/119|288.49|
|131,072|118,868|285.44|15.50|96/121|424.69|

Filled-context tests used synthetic record prefixes and 128 output tokens.
Both completed and passed a subsequent prompt-cache reuse check. They test
memory fit and speed, not long-context retrieval or reasoning accuracy.
Minimum sampled whole-device free VRAM: 406 MiB at 96K and 366 MiB at 128K.
One-second samples can miss transient peaks; other GPU apps may cause OOM.

## What changed

Attention, recurrent processing and feed-forward down projections remain on
the GPU. Gate/up weights for the first 54 blocks at 96K, or all 64 at 128K,
reside in host memory. A private scheduler queues upcoming weight transfers
in three 33 MiB regions and lets eligible operations read those buffers directly.
This overlaps transfers with computation and avoids a redundant device copy.

Both use Q4_0 target/draft KV, MTP depth 4, batch/microbatch 128, six CPU
threads, Flash Attention, fixed chat template and transactional recurrent replay.
Previous configurations offloaded all three FFN projections for 36/44 blocks
and used minimum GPU-offload batch 32; candidates use gate/up 54/64 and batch 2.
The combined changes, not prefetch alone, account for this comparison.

## Availability and reproduction status

These configurations are deployed and live-verified in the local shared router used by
Zterm2-local and DeepSeek Harness. They are **not yet a standalone reproducible
release of this public repository**. The historical D-CFR and Q4 build scripts
do not include the queued selective-offload scheduler; ordinary llama.cpp does
not acquire it by setting environment variables. A portable source-pinned
prefetch patch/build and exact model hash are still needed for a public
reproduction package. Do not use the historical Unsloth model hash for this run.

The tested large-context launchers were text-only. Image support and quality
were not validated in this experiment. Larger output allowances remain
separate from the small benchmark output lengths.
