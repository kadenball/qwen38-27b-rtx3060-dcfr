# Guided installation and bounded hardware tuning

RouteWeaver can build the pinned Q3 runtime, compare a small set of settings on
the destination machine, and save a profile for later launches. The manual
[installation and launcher](q3-quickstart.md#manual-installation-download-and-build)
remain independent: no profile is needed to use them.

This is an **experimental tuner**, not a universal hardware optimizer. Its
process/safety logic is covered by offline and real HTTP/subprocess tests, and
read-only detection plus busy-GPU refusal were exercised on the development
machine. A real 96K session on the RTX 3060 completed eleven trials, including
repeat confirmation and chat/cache/tool/cancellation validation. It selected
six threads, depth-four MTP, 128/128 batch and 41 host gate/up blocks; final
minimum sampled free VRAM was 884 MiB. That conservative placement preserves
more headroom than the manually tuned 36-block preset. It is not a claim of
additional speed over that manual preset. Other hardware remains experimental.
The existing headline GPU benchmarks predate this tuner and do not measure
automatic setup or certify its profile-selection policy.
The [verification record](../benchmarks/autotune-verification-20261006.json)
separates software tests, real-model checks and remaining validation limits.

## Scope and prerequisites

- Linux x86-64, AVX2 and FMA, one selected NVIDIA GPU with 12 GB or more VRAM.
- The current hash-pinned RCO IQ3_XXS MTP model, text-only, at 98,304 or 131,072
  tokens. Other quants, smaller cards, ARM, multi-GPU and automatic vision tuning
  are not supported in this first version.
- Install the [build tools](q3-quickstart.md#requirements), CUDA toolkit and a
  compatible host compiler. Setup discovers installed toolchains and verifies
  compatibility by compilation; it does not install system packages or drivers.
- Stop other GPU compute applications before setup/tuning/launch. Even an idle
  model process is refused. Some GPU-enabled browsers appear in NVIDIA's compute
  process list and must also be stopped by their owner. Desktop graphics memory
  still counts toward the available-memory measurement.
- At least 4 GiB of available RAM for setup, and a larger placement-dependent
  reserve for GPU trials. Free disk must cover the missing model plus 4 GiB of
  build headroom. Existing model files are reused rather than copied.

## Automatic path

From the repository root:

```bash
# Read-only; does not download, compile, stop processes, or load a model.
./routeweaver doctor
./routeweaver setup --context 98304 --dry-run

# Prompts once before downloading/building/tuning; --yes makes it unattended.
./routeweaver setup --context 98304
./routeweaver start --context 98304
```

The download is about 10.57 GB. `setup` checks model/template hashes, builds with
native CPU instructions and the selected GPU architecture, runs model-free
regressions, then starts tuning. `start` uses the saved profile without retuning.
When the build's recorded CPU/toolchain/GPU architecture differs or is unknown,
setup moves it to a `build/q3-candidate.saved-*` directory and compiles a fresh
one. A failed build restores the previous directory; no old build is deleted.
Successful hardware/runtime changes still require a newly validated profile.
It serves the same localhost OpenAI-compatible endpoint/model alias as the
manual launcher. No service is installed, no harness configuration is changed,
and no remote listener or credentials are created.

For 128K, pass `--context 131072` consistently. To reuse an already-downloaded
copy of the exact pinned model:

```bash
./routeweaver setup --context 98304 --model /path/to/RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf
```

The saved profile remembers that local path. The file must already exist and
match the pinned SHA; setup will not overwrite an unrelated file. If toolchain
detection needs help, set `CUDA_TOOLKIT_ROOT`, `CC`, and `CXX` as in the manual
instructions. Automatic build parallelism is deliberately limited to one job.

After changing hardware or updating the runtime, retune when the GPU is idle:

```bash
./routeweaver tune --context 98304 --model /path/to/RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf
./routeweaver status --context 98304
./routeweaver rollback --context 98304
```

`--gpu 1` selects another NVIDIA device; one device is used, never combined
with the others. Use the same selection on subsequent commands. Rebuild the
runtime on the destination CPU/GPU before tuning; a profile does not make a
copied native binary portable. `rollback` restores the previous matching JSON
profile; it does not restart any running server. Incompatible profiles are
rejected, including after driver/runtime/context changes.
`status` works while serving; stop a server launched by `routeweaver start`
before tuning or rolling back, because it holds the GPU operation lock.

## What is measured

The initial host placement is only an estimate. Tuning first looks for a safe
reference configuration, then screens one-variable alternatives for:

- Physical/logical CPU thread counts.
- CPU-resident gate/up block counts.
- Batch/microbatch sizes.
- MTP depths 2, 4 and 6, plus **real no-speculation** decode.

This is not an exhaustive or combinatorial search. The budget may be exhausted
before every alternative runs. A successful plain-decode arm is required before
any profile is promoted. Failed configurations are recorded, not silently
substituted with a smaller context or another quant.

Each screen uses synthetic Rust, C++, reasoning and longer-input coding
requests: 48 generated tokens for coding and 64 for reasoning. EOS is ignored
for fixed-length timing. Screening uses seed 101; confirmation uses seeds 202
and 303. The JSON report retains per-request prefill/decode speed, timings,
drafted/accepted counts, output hashes and sampled memory. Prompts/settings are
derived from the existing benchmark fixtures; outputs are not quality scores.

Ranking uses the geometric mean of four per-fixture decode medians and the
longer-input prefill median. A candidate is ineligible if any of those metrics
falls more than 10% below the reference. A changed configuration must reproduce
at least a 3% aggregate gain in confirmation; otherwise the reference is
validated as the fallback. Before promotion, the selected profile must pass
cache reuse, edited-prefix, tools, independent-chat and cancellation checks.
Output hashes are recorded, but tuning does not claim exact output equivalence
between different speculation settings or evaluate task correctness.

Allocated context is held fixed. The longer-input fixture is only several
thousand tokens: this is **not** a near-full-context stress certification.
Before important long sessions, run the separate occupied-context and
actual-model checkpoint tests in the manual guide.

## Limits and failure behavior

Defaults: 900 seconds of tuning and at most 12 model-load trials, including
fit attempts and confirmation. Download, build and model-free checks are
additional. Cleanup can take a few seconds beyond the tuning deadline.

```bash
./routeweaver tune --context 98304 --budget-seconds 1800 --max-trials 20 --margin-mib 1024
```

- Default minimum sampled free VRAM: 768 MiB; never configurable below 512 MiB.
- Trials monitor free VRAM, available RAM, system swap-out and competing GPU
  compute processes. Low RAM (below 1.5 GiB), over 64 MiB of new swap-out,
  missing monitoring data, insufficient GPU margin or a competing process
  rejects the trial. Monitoring is sampled, so transient allocation spikes and
  unrelated system activity remain limitations, not guarantees against OOM.
- Only the tuner's own subprocess group is terminated. Existing applications
  are never killed. Ctrl-C and SIGTERM clean up owned trial processes.
- Local endpoint identity is checked using a unique model alias before requests;
  a port collision cannot redirect benchmark prompts into an unrelated session.
- Failed/interrupted tuning does not replace a previously saved profile. If no
  profile passes, use the manual launcher or rerun with more time/headroom.
  An old profile cannot be used with an incompatible newly built runtime; its
  fingerprint check is deliberately not bypassed by rollback.
- A launch checks current RAM/VRAM against the selected profile's requirements.
  Lower available memory causes refusal, not silent offload/context changes.

Profiles and logs live under ignored `build/routeweaver/`. Profiles are JSON,
written atomically with a previous-version backup; they are not executable shell
files. They fingerprint CPU features/core availability, GPU identity/driver,
runtime files, model/template hashes and context. Logs/profiles may contain
local paths and hardware identifiers; inspect them before sharing. Nothing is
uploaded automatically. Repository-default environment experiments are reset
for trials, and the manual launcher remains available independently.

## Manual path

Prefer to control every setting, or outside the automatic scope? Follow
[manual download/build](q3-quickstart.md#manual-installation-download-and-build)
and [manual serving](q3-quickstart.md#serve-at-96k-or-128k).
`THREADS`, `HOST_FFN_BLOCKS`, `BATCH_SIZE`, `UBATCH_SIZE`, `MTP_DEPTH`, and
`SPECULATION=none|mtp` remain explicit overrides. Automatic profiles never
overwrite those scripts or the published historical benchmarks.
