# Run the Q4 32K interactive profile

This guide builds the Q4 runtime, downloads the exact model, and checks prompt
reuse. It does not require Routeweaver, Zterm, or DeepSeek Harness. Any compatible
OpenAI-style chat client can connect to the standalone server.

## Requirements and scope

- **Measured hardware:** NVIDIA RTX 3060 **12 GB**, Intel i5-12400 (6 cores),
  16 GB system RAM, Linux x86-64. This is CPU/GPU hybrid inference, not all-GPU.
- An AVX2-capable x86 CPU. The new CPU path is AVX2-specific. Native compilation
  means binaries should not be copied to a different CPU without rebuilding.
- Working NVIDIA driver and CUDA toolkit. The measured toolchain is **CUDA 13.3
  and GCC/G++ 15**. Use a CUDA-supported host compiler; other versions have not
  been validated here. `nvidia-smi` working does not mean `nvcc` is installed.
- Git, curl, tar, patch, SHA-256 utilities, CMake, Ninja, C/C++ compiler, Python 3.
  Optional DSH patching/tests need Node.js and the exact compatible packages.
- The model download is **15.48 GB**. Allow **30 GB free disk** for the model,
  sources, build and temporary archive; additional toolchain installation needs
  more. A fast local SSD matters for loading/page faults. More system RAM helps
  avoid pressure, but speed on other RAM/CPU configurations is not established.
- Do not load another large model on the same GPU. The measured desktop profile
  retained about 698 MiB device headroom after short tests; other applications
  can consume it. Windows/WSL and other GPUs are not validated by this release.

The head-split CUDA path is explicitly guarded for **SM 8.6** and the tested
Qwen tensor shapes. This build targets architecture 86. Do not assume another
12 GB GPU has the same capacity or speed; fallback kernels may use more memory.

## 1. Clone and build

```sh
git clone https://github.com/kadenball/qwen38-27b-rtx3060-dcfr.git
cd qwen38-27b-rtx3060-dcfr
export CUDA_TOOLKIT_ROOT=/usr/local/cuda
export CC=gcc
export CXX=g++
BUILD_JOBS=2 bash scripts/build-q4.sh
```

Point those variables at the installed, compatible toolkit/compiler if different.
The script fetches upstream `c060ca974c773c7c3d17fd1b66dc9d312bc292c0`, checks
the archive hash, applies the historical patch and Q4 overlay with no fuzz,
checks modified-source hashes, and builds into `build/q4`. It does not modify
an existing server installation. Two build jobs limit RAM pressure; compilation
can take a while. Failed downloads are retained, not silently treated as complete.

The Q4 overlay contains the CPU multi-query/dot2 kernels, head-split attention
workspace reuse, and the shallow-depth cache policy used for the measurements.
Optional research paths in the historical patch remain opt-in. This is an
experimental source release, not an upstream-supported llama.cpp distribution.

## 2. Download and verify the model

```sh
bash scripts/download-q4.sh
```

Pinned publisher: [bartowski/Qwen3.8-27B-GGUF](https://huggingface.co/bartowski/Qwen3.8-27B-GGUF/tree/125a02af4987b57c7deb88d7f2ec58a5725c07c0).
File: `Qwen3.8-27B-IQ4_XS.gguf`, **15,475,951,328 bytes**.
SHA-256: `4b927360fa7c4aa41a734302f0795ca45df0e8b8a258bc76cc5fbf9d98a484db`.
Publisher revision: `125a02af4987b57c7deb88d7f2ec58a5725c07c0`.

This file includes the MTP head: **no separate draft model is needed**. It is
not Q4_K_M, the older IQ4_XS layout, or an uncensored derivative. Model licensing
is separate from this repository's code; consult the publisher's card.

Downloads resume into `.partial`; only a hash-verified file receives the final
name. Existing final files are verified, never overwritten. For an existing
copy, verify the same hash and set `MODEL_PATH` when launching. `MODEL_DIR`
can change the download directory. Do not bypass a checksum mismatch.

## 3. Launch

```sh
bash scripts/serve-q4.sh
```

For an existing model or a different port:

```sh
MODEL_PATH=/path/to/Qwen3.8-27B-IQ4_XS.gguf PORT=8081 bash scripts/serve-q4.sh
```

The server binds to **127.0.0.1**, with one slot, 32,768 context, MTP-3,
Q4_0 target/draft KV, 128/128 batching, 6 threads, four checkpoints and CPU
FFN blocks 0–42. It uses the model's embedded chat template. Vision is disabled.
The default is non-thinking for the short-test profile; reasoning is discussed
below. `--cache-ram 0` disables the separate RAM cache, not live-slot prefix reuse.

Do not expose this unauthenticated listener directly to the internet. Remote
access belongs behind the chosen harness's authenticated remote-access mechanism.

## 4. Verify before connecting a harness

In a second terminal, from the repository:

```sh
curl --fail http://127.0.0.1:8080/health
bash scripts/check-q4-cpu.sh
bash scripts/check-q4-attention.sh
python3 scripts/check-q4-cache.py
```

The CPU check compares initialized reference and optimized calculations at
widths 2–6. The attention check verifies 80 workspace-allocation contracts on
SM 8.6; it is not a numerical-accuracy test. The cache check makes two short synthetic requests in slot zero;
the second must show at least 1,000 cached tokens, and both replies must recall
the synthetic marker. Use a dedicated idle server:
the probe replaces its warm slot prefix, though it never edits a client chat.
Use `--url http://127.0.0.1:8081` for a different port.

To repeat the published short-task protocol after stopping other workloads:

```sh
python3 scripts/bench-q4-short.py
```

This warms the server, then runs the published Rust/C++ fixtures with seeds
101/102, 128 output tokens, thinking off and no cache reuse. It prints JSON
timings and acceptance counts. Four runs are a screening test, not a quality
evaluation or confidence interval. Keep cache checks separate from timing runs.

Example chat request:

```sh
curl --fail http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen38-q4-32k","messages":[{"role":"user","content":"Explain a ring buffer briefly."}],"max_tokens":256,"stream":true}'
```

## 5. Connect DeepSeek Harness or Zterm

Create/edit a local OpenAI-compatible provider with these values; preserve other
providers and models:

| Field | Value |
| --- | --- |
| Provider ID (for the optional DSH patch) | `routeweaver` |
| Base URL | `http://127.0.0.1:8080/v1` |
| Model ID | `qwen38-q4-32k` |
| Display name | `Qwen 27B IQ4_XS · 32K · MTP-3 cached` |
| Context window | `32768` |
| Maximum output allowance | `16384`, further limited by available context |
| API key if the client requires a nonempty value | `local` (placeholder; no server authentication is enabled) |

The client must send stable, complete conversation prefixes to reuse the slot.
Compaction, switching chats, changing tool catalogs or rewriting the system
prompt can require fresh prefill. One slot serializes requests; this is not a
multi-user high-throughput deployment. Registering this alias beside the original
model IDs avoids breaking saved conversations.

For thinking, pass chat-template kwargs `enable_thinking: true`,
`preserve_thinking: true`, `reasoning_effort: "medium"`, and configure the client's
reasoning support rather than merely relabeling a model. A server default can be
set by appending these arguments (the runtime accepts the final option value):

```sh
bash scripts/serve-q4.sh --reasoning on \
  --chat-template-kwargs '{"enable_thinking":true,"preserve_thinking":true,"reasoning_effort":"medium"}'
```

The 128-token short benchmark used thinking off; the long harness check used
medium. Output allowances are not targets to generate that many tokens.

For DSH's local output-budget and compaction corrections, see
[the optional version-pinned harness patch](../harness/dsh/README.md). Zterm has
its own compaction implementation: do **not** apply DSH package patches to it.
Both clients benefit from the same backend kernels and cached server profile.

## Expectations and troubleshooting

- The observed short-task means were **16.06 Rust / 14.30 C++ tok/s**, not 40.
  A roughly 26K occupied chat decoded at **11.09 tok/s**. Cold prefill still costs
  time; the 12.1-second follow-up depended on a 25,950-token cache hit.
- No cache hits: verify MTP depth 3, stable prompt prefix, same slot, and this
  build's libraries. Do not remove the deeper replay state-clear safeguard.
- OOM: close other GPU workloads first. Lowering context or placing more FFN
  blocks on CPU is an experiment, not the measured preset. Do not silently
  turn on auto-fit and describe the result as this exact configuration.
- Slow disk reads: examine RAM pressure/page faults and other running models.
  mmap does not mean CPU weight reads are free. Avoid compiling during timing.
- Truncated answers: check actual output allowance, context metadata and client
  tool overhead. The optional harness patch cannot make oversized requests fit.
- Restore: stop this standalone server with Ctrl-C and launch the previous
  installation. The build/launch scripts do not edit system services or harness
  profiles. Package patch rollback is documented separately.

See [the measurement report](q4-32k-interactive.md) for sample sizes and caveats.

## Release verification

The public build was compiled from a fresh upstream archive with GNU 15.3.1
and CUDA 13.3 on the RTX 3060 (driver 595.71.05). All seven loaded ggml/llama
libraries came from the new build, not the existing private installation.
Checks passed: 7,680 bit-exact CPU cases, 680 CPU graph checks, 80 attention
allocation contracts, and 30 DSH regressions. The DSH installer was also tested
against stock npm package files and reapplied successfully without changing them.

The repeated short fixtures measured **15.83 Rust / 14.36 C++ tok/s** means,
consistent with the earlier small sample, with identical draft counts. The
model's full checksum was rechecked. The chat smoke test recalled the marker in
both replies and reused 2,514 tokens, processing only 17 new tokens on the
follow-up (2.60 seconds total). This is a small functional check, not a repeat
of the earlier 26K occupied-context experiment. See the [release verification data](../benchmarks/q4-release-verification-20260918.json).
