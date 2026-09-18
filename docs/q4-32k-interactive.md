# Q4 at 32K on an RTX 3060: prompt reuse matters

Measured September 18, 2026. This is a separate interactive-serving experiment,
not a revision of the original IQ3 throughput result or a new speed record.

## Configuration

- RTX 3060 12 GB; Intel i5-12400; 16 GB system RAM, normal desktop running.
- `Qwen3.8-27B-IQ4_XS.gguf`, 15,475,951,328 bytes.
- Allocated context: 32,768 tokens, up from 20,480 (60% more capacity).
- MTP depth 3, ordinary recurrent-state backups; D-CFR inactive.
- Q4_0 target and draft KV; Flash Attention enabled.
- Batch and microbatch 128; 6 CPU threads; CPU multi-query width 6.
- Attention on GPU; FFN gate/up/down weights for blocks 0–42 on CPU.
- Four context checkpoints; mmap loading; automatic fit disabled; no vision.

These are measured settings, not a universal hardware recommendation. The
runtime includes additional local head-split CUDA and IQ4_XS multi-query CPU
optimizations. Its upstream base is
`8242074aa015e87f9e2e10993f164e0d1ac535b1`, with local modifications.
Those additional runtime changes are **not bundled in this release**;
`scripts/build.sh` still builds the historical IQ3/D-CFR experiment. The base
commit and settings alone do not reproduce the Q4 numbers below.

## Short decode screening

Two synthetic coding prompts, two seeds (101 and 102) per prompt, 128 generated
tokens per run, thinking disabled. Temperature 0.7, top-p 0.95, top-k 40,
min-p 0.05. Rust input: 943 tokens; C++ input: 86 tokens. Separate warmup;
measured runs had no prompt-cache hits.

| Profile | Rust mean (range), tok/s | Accepted/drafted | C++ mean (range), tok/s | Accepted/drafted | Free VRAM after runs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20,480; MTP-5; CPU FFN 0–34 | 18.32 (18.12–18.52) | 202/257 | 14.12 (13.99–14.25) | 185/333 | 265 MiB |
| 32,768; MTP-3; CPU FFN 0–42 | 16.06 (16.02–16.09) | 184/208 | 14.30 (14.11–14.50) | 175/237 | 698 MiB |

The final profile trades about 12.3% of short Rust throughput for capacity and
interactive caching. The small C++ difference is not evidence of a reliable
speedup. Acceptance is pooled accepted/drafted tokens, not the proportion of
all output supplied by drafts. Memory figures are whole-device snapshots,
not peak allocation measurements or guarantees against OOM.

## Long-chat verification

An isolated synthetic conversation through the actual harness and router:

| Request | Cached input | Newly processed input | Output | Wall time |
| --- | ---: | ---: | ---: | ---: |
| MTP-5 follow-up | 0 | 25,692 | 118 | 159.1 s |
| MTP-3 cold request | 0 | 25,843 | 106 | 169.5 s |
| MTP-3 cached follow-up | 25,950 | 25 | 102 | 12.1 s |

The cached follow-up spent **2.02 seconds processing the prompt** and decoded
at **11.09 tok/s**, at roughly 26K occupied context. Replies completed normally;
a synthetic marker and an assertion of more than 20K cached tokens passed.
Medium reasoning remained enabled for this harness test.

The requests had different histories and output lengths. This is not a matched
13× speedup benchmark. It directly demonstrates avoiding repeated full-prefix
prefill. New chats and substantially changed prefixes still require prefill.

### Why shallower speculation helped

In this tested runtime, transactional GDN replay at draft depths above four
deliberately clears prompt state between requests: compact rollback factors
are not captured in saved prompt checkpoints. Removing that safeguard would
risk recurrent-state correctness. MTP-3 uses ordinary backups and permits
cross-request reuse; extra CPU FFN placement makes room for those backups.

This does not invalidate the original D-CFR memory measurements. It identifies
a different optimization target: responsive repeated turns rather than maximum
throughput on independent short requests. Cache-safe deep replay remains open.

## Harness reliability fixes

Two separate adapter issues were also addressed locally:

- A fixed 4,096-token safety reserve could leave only 19 output tokens in a
  20,480-token window with 16,365 estimated input tokens. The local adapter now
  uses a 5% reserve bounded to 512–4,096 tokens, honors explicit output limits,
  and rejects exhausted requests clearly. This is an estimate, not an overflow
  guarantee; nonlocal providers are unchanged.
- Compaction skips prefixes below 512 estimated tokens, avoiding attempts to
  replace tiny fragments with larger structured summaries. Existing tool-pair
  and summary validation protections remain.

Thirty local regression tests passed. These adapter changes are described here,
not shipped as a general-purpose harness installer. Tools were not removed and
reasoning was not curtailed to obtain the improvement.

## Limits and replication priorities

Only two seeds per short task were measured. Placement and speculative depth
both change, and generated outputs can differ. There is no full-32K quality
evaluation, repeated long-chat latency distribution, isolated kernel speedup,
or claim that Q4 achieves the historical IQ3 30–40 tok/s figures.

Useful next tests are matched prompts/outputs, more repeats, cache acceptance
and correctness across tool turns, full-window memory stress, and a complete
release of the additional runtime optimizations. Report cold prefill, cached
prefill, decode, acceptance, and occupied context separately.
