# Benchmark records

The structured JSON files are the authoritative result summaries:

- `transactional-factor-replay-20260827.json` records the two-plane TFR A/B
  and depth sweep.
- `deferred-commit-factor-replay-20260827.json` records the one-plane D-CFR
  memory result, exact-output checks, phase profiling, and rejected variants.
- `turbo-30t-and-packed-tree-20260827.json` records the final MTP6 Turbo
  profile, packed-tree comparison, and 29.7/33.2 tokens/s measurements.
- `rvn-q3-zterm-presets-20260827.json` records the five-run, seven-prompt RVN
  depth-4 and depth-8 generation validation.
- `fast-prefill-20260828.json` records the RVN batch/microbatch sweep that
  selected 128/128 as the balanced RTX 3060 prompt-ingestion profile.

`raw/ordinary-mtp4-fixed46-long-20260827.log` and
`raw/dcfr-mtp4-fixed46-profile-v2-20260827.log` record the 512-token
ordinary/D-CFR A/B. Timing and runtime output are retained verbatim, with the
repository prefix normalized to `<repository>`.
