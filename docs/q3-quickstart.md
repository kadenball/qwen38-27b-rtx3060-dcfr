# Run RouteWeaver Q3

The October 5 release packages the measured cache-safe D-CFR, queued transfers,
IQ3_XXS CPU kernels and Q4_0 attention path. These instructions build a separate
runtime; they do not replace another llama.cpp installation or change a harness.

For the opt-in October 8 successor, use the separate [Q3 Fast guide](q3-fast.md).
The commands below intentionally retain the October 5 default.

## Requirements

- Tested: Linux, RTX 3060 12 GB, Core i5-12400, 16 GB installed RAM.
- CUDA toolkit with `nvcc`, a supported C/C++ host compiler, CMake, Ninja,
  Python 3, Git, curl, tar, patch, sha256sum and flock (util-linux).
- The measured CUDA toolchain was CUDA 13.3 with GCC 15; the native CPU backend
  was built with GCC 16. The standalone build uses the selected compiler for both.
- The CPU optimization targets x86 AVX2/FMA. Build on the destination machine;
  do not copy a native binary between different CPUs.
- The model is 10,569,270,720 bytes. Allow extra disk space for source, build
  artifacts and the CUDA toolkit. Vision adds a 927,607,488-byte projector.

16 GB RAM worked on the test machine but leaves limited room for browsers,
multiple harnesses and compilation. More RAM gives operating headroom, not a
promised tokens/s increase. Stop other model servers before starting this one.

## Guided setup and automatic tuning

After installing the prerequisites, the experimental guided route is:

```bash
git clone https://github.com/kadenball/qwen38-27b-rtx3060-dcfr.git
cd qwen38-27b-rtx3060-dcfr
./routeweaver setup --context 98304
./routeweaver start --context 98304
```

Choose `--context 131072` for both commands to tune and launch the 128K profile.
Setup selects from installed CUDA/host compilers using a compile-only probe,
builds natively, and measures configurations rather than assuming six threads
is optimal for every CPU. It never changes the selected quant/context to win a
benchmark, never installs drivers or edits a harness, and refuses existing GPU
compute processes. The initial automatic scope is Linux x86-64 AVX2/FMA,
one 12 GB+ NVIDIA GPU, and the pinned text-only Q3 model.

[Read-only planning, time budgets, reuse an existing model, safety limits,
saved profiles and rollback](automatic-setup.md). This new tuner has offline
and HTTP/process integration coverage plus a real 96K tuning session on the
RTX 3060. Other hardware remains experimental. Use the manual path below for
explicit control or unsupported setups.

## Manual installation: download and build

```bash
git clone https://github.com/kadenball/qwen38-27b-rtx3060-dcfr.git
cd qwen38-27b-rtx3060-dcfr
./scripts/download-q3.sh

# Set CC/CXX to compilers supported by the installed CUDA toolkit.
CUDA_TOOLKIT_ROOT=/usr/local/cuda \
CC=gcc-15 CXX=g++-15 BUILD_JOBS=1 \
./scripts/build-q3.sh

./scripts/check-q3.sh
```

Downloads are revision-pinned, resumable and SHA-256 checked. The default model
is `RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf` from
[0bserverx](https://huggingface.co/0bserverx/Qwen3.8-27B-Heretic-GSQ-RCO-GGUF/tree/1b65f1eb296b08113e89b43d8ab634c6b555bf98).
This is **not** the historical Unsloth checkpoint or the earlier RVN multilingual
GGUF. The MTP head is included; there is no separate draft-model download.

Model SHA-256:

```text
475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039
```

The fixed community chat template is pinned to
[`froggeric/Qwen-Fixed-Chat-Templates` revision 756cfb6](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/tree/756cfb69d742355fd310b4ba9d50815a27d9d241),
v22.4. Model and template license terms remain those of their publishers.

The build applies the historical research and Q4 overlays, then the Q3 CPU,
queued-prefetch and cache-safe overlays. Every changed source file has a checksum.
Existing mismatched source directories are rejected, not overwritten.
`CUDA_ARCHITECTURES` defaults to `native`; use `86` to reproduce the RTX 3060
target. Other GPU builds are [experimental](hardware.md).

## Serve at 96K or 128K

```bash
# 96K: the default profile
./scripts/serve-q3.sh

# Or, after stopping the first server, 128K
CONTEXT=131072 ./scripts/serve-q3.sh
```

The API is `http://127.0.0.1:8080/v1`. Use model ID
`routeweaver-q3-98304` or `routeweaver-q3-131072`. Configure an OpenAI-compatible
client with that base URL and model ID; if the client requires a key, a local
placeholder is sufficient. The server itself has no authentication and binds
only to loopback. Remote use requires an authenticated proxy or tunnel.

Set the client's context window to the same integer as the server. The launcher's
output allowances are 32,768 at 96K and 65,536 at 128K; these are ceilings, not
reserved KV space or guaranteed free context. Input plus output must fit the
window. Reasoning is enabled at medium by default; clients can set template
kwargs per request. The benchmark uses its own pinned sampling and short outputs.

| Setting | 96K | 128K |
|---|---:|---:|
| Context tokens | 98,304 | 131,072 |
| Gate/up blocks placed in host memory | 36 | 46 |
| MTP depth | 4 | 4 |
| K/V cache | Q4_0 / Q4_0 | Q4_0 / Q4_0 |
| Batch / microbatch | 128 / 128 | 128 / 128 |
| Threads / parallel sequences | 6 / 1 | 6 / 1 |
| Queued transfer storage | 3 x 48 MiB | 3 x 48 MiB |

Gate/up **weight placement** is not the same as forcing all those operations to
run on the CPU. Eligible operations transfer those weights and execute on CUDA.
Attention, recurrent state and down projections remain GPU-resident.

Inspect settings without loading a model:

```bash
CONTEXT=131072 ./scripts/serve-q3.sh --print-config
```

Supported overrides: `MODEL_PATH`, `CHAT_TEMPLATE_PATH`, `PORT`, `THREADS`,
`HOST_FFN_BLOCKS`, `MTP_DEPTH`, `MAX_OUTPUT_TOKENS`, `MMPROJ_PATH`,
`SPECULATION=mtp|none`, `BATCH_SIZE`, `UBATCH_SIZE`.
Batch overrides must satisfy `32 <= UBATCH_SIZE <= BATCH_SIZE <= 512`.
`SPECULATION=none` selects real plain decode; it does not merely lower MTP depth.
The model/template hashes still have to match. More host blocks can reduce
weight residency; fewer can improve speed if they fit. Nondefault placement,
threads and depth are experiments, not validated hardware presets.

## Optional vision

```bash
./scripts/download-q3.sh --vision-only
MMPROJ_PATH="$PWD/models/mmproj-F16.gguf" ./scripts/serve-q3.sh
```

The pinned F16 projector runs on the CPU to preserve GPU headroom. The local
96K and 128K deployment passed OCR and image transmission/recognition checks.
A synthetic 1536x2048 image took about 88-89 seconds end to end on the test CPU;
vision is functional, not a low-latency claim. Images must be sent through an
image-capable chat client. Text-only benchmark figures do not include vision.

## Validate and benchmark

`check-q3.sh` runs model-free CPU, scheduler-cleanup and checkpoint tests.
For the actual-model rollback test, ensure no other GPU model is loaded, then:

```bash
./scripts/check-q3.sh --model
```

Start a dedicated server, then use a second terminal:

```bash
# Terminal 1; stop any other loaded model first
PORT=8094 CONTEXT=131072 ./scripts/serve-q3.sh

# Terminal 2
python3 scripts/check-q3-chat.py --url http://127.0.0.1:8094
python3 scripts/bench-q3.py --context 131072 --output build/bench-candidate
```

For the old same-GGUF runtime arm, build with `Q3_VARIANT=baseline`, stop the
candidate, then launch with `Q3_VARIANT=baseline CONTEXT=131072 PORT=8094`.
Its defaults restore ordinary depth-4 rollback, 54 host gate/up blocks and three
33 MiB buffers. Use a different output directory for each run. Never run the
two model processes together on the 12 GB test card.

The benchmark runner checks context, rejects an already-busy server, warms up,
runs five seeds and records timing, output hashes, acceptance and sampled GPU
memory. It sends synthetic requests; don't point it at a shared work session.
See [methodology and results](q3-cache-safe.md).

## Troubleshooting and rollback

- **OOM or CUDA graph allocation failure:** stop other GPU models/apps. The
  measured free margin is only hundreds of MiB. Choose 96K or increase host
  gate/up blocks, and retest; display load can consume the margin.
- **High disk reads / paging:** reduce competing RAM use. Don't use swap as a
  substitute for sufficient working memory. Cold loading and first prefill are
  separate from generation speed.
- **Slow follow-ups:** check `timings.cache_n`. Editing earlier messages,
  changing templates or switching conversations can legitimately invalidate reuse.
- **Old GPU / unsupported compiler:** use the architecture and compiler supported
  by that CUDA toolkit. Our support table does not certify every CUDA device.
- **Revert:** stop this server and launch the separately built baseline or an
  existing historical installation. No shared installation is overwritten.

Checkpoints are runtime/configuration-specific. Do not load old partial D-CFR
checkpoints into the new format or move snapshots across MTP depths.
