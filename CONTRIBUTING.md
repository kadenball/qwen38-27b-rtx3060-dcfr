# Contributing to RouteWeaver

The highest-value contributions are reproducible hardware reports, focused
correctness tests, and small changes that improve a measured bottleneck.
Negative results and failed configurations belong in the record too.

## Reproduce first

Start with the [Q3 quickstart](docs/q3-quickstart.md), keep the model/template
hashes fixed, and run the model-free and interactive checks. Use a dedicated
benchmark server rather than a working conversation. The benchmark runner
records five seeds by default; do not publish only the fastest run.

For hardware reports, use the issue form and include launch settings,
acceptance, context occupancy and memory measurements. Label changes to model,
quant, sampler, output length or template so readers can identify confounders.

## Changes

- Discuss a broad design or backend port before preparing a large patch.
- Preserve a failing/reproduction case and test the actual call path.
- Keep source patches against the pinned base, with separate candidate and
  baseline trees. Do not edit a live serving build in place.
- Check prompt caching, tool calls, cancellation and checkpoint behavior in
  addition to throughput. A faster corrupt state path is not an improvement.
- Explain which measured variable changed. Separate an isolated mechanism
  result from a combined tuned-configuration result.

Run these before proposing a change:

```bash
python3 scripts/update-checksums.py
python3 tests/test-release.py
python3 tests/test-autotune.py
./scripts/check-q3.sh
```

`SHA256SUMS` covers shipped files. Source manifests additionally pin all
modified upstream files. When updating a patch, prepare a fresh source tree,
verify its final source hashes, and retain the earlier benchmark artifacts.
CI checks packaging and model-free CPU/state behavior; GPU performance and
actual-model correctness still require real hardware reports.

RouteWeaver is independent of upstream llama.cpp. Changes proposed upstream
must follow that project's contribution rules. Credit upstream work and
disclose meaningful AI assistance; human maintainers remain responsible for
reviewing and understanding contributions.
