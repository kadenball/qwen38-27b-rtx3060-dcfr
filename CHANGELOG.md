# Changelog

## 2026-10-05 - RouteWeaver Q3 experimental release

- Adopted the RouteWeaver project name and banner; repository URL/history unchanged.
- Packaged the IQ3_XXS CPU multi-query kernel, queued-transfer scheduler,
  complete D-CFR checkpoints and wider 48 MiB staging buffers.
- Added source-pinned candidate/baseline builds and separate 96K/128K launchers.
- Pinned the RCO GGUF, fixed v22.4 template and optional CPU vision projector.
- Published five-seed 128K comparisons, acceptance/ranges, occupied-context
  screens and explicit exclusions/limitations.
- Added model-free checkpoint, CPU and cleanup tests, actual-model continuation
  tests, chat/cache/tool/cancellation checks, and a standalone benchmark runner.
- Added hardware support labels, reporting guidance and release checks.
- Preserved the previous landing page in [HISTORY.md](HISTORY.md).

## Earlier work

- September 20: experimental queued/selective-offload Q3 at 96K/128K, using
  the earlier multilingual RVN model. [Details](docs/iq3-large-context.md).
- September 18: Q4 32K interactive runtime and reproducible build.
  [Guide](docs/q4-quickstart.md).
- August: original D-CFR/TFR work, 64K generation tests, depth sweeps and
  prompt-ingestion tuning. [Historical page](HISTORY.md).
