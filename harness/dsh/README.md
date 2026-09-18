# Optional DeepSeek Harness corrections

These patches are independent of the Q4 backend. They target the measured
installation: `@deepseek-ai/dsh` and `@deepseek-ai/dsh-compaction-basic`
**0.1.1-rc.2**, and `@earendil-works/pi-ai` **0.82.1**. Exact file hashes,
not version strings alone, decide compatibility. Locally modified or newer
packages may be rejected; this is intentional. No universal DSH installer or
Zterm compaction patch is implied.

The stock npm compaction package and pi-ai file were checked against their
published tarballs. `stock-compaction.patch` supports that clean baseline;
`checkpoint-safety.patch` also supports the earlier locally patched baseline.

Changes:

- Local provider `routeweaver` at `http://127.0.0.1:8080` or
  `http://localhost:8080` receives a 5% output safety reserve bounded to
  512–4,096 tokens. Other providers/endpoints retain upstream behavior.
  Explicit output caps remain honored; exhausted requests fail clearly.
- Compaction uses a separate tool-free summarizer, validates structured output,
  retains bounded original task excerpts, rejects non-shrinking/invalid output,
  avoids checkpoint-only/tiny-prefix compaction, and backs off failed automatic
  retries. Summary structure checks cannot guarantee semantic accuracy.

## Apply and test

Stop DSH when all sessions are idle. Back up its configuration first. Find the
installed package directory (for an npm global installation, inspect
`npm root -g`, then append `/@deepseek-ai/dsh`). Do not pass a profile directory.

```sh
export DSH_ROOT=/path/to/node_modules/@deepseek-ai/dsh
node harness/dsh/apply.mjs
node --test harness/dsh/compaction.test.mjs harness/dsh/small-context.test.mjs
```

The installer verifies source hashes and preserves original files beside their
targets. It does not restart DSH, modify conversations, install dependencies,
or modify client settings. Reapplication verifies already-patched files.
An unknown-source error means stop and port/retest the patch; do not remove the
hash guard or overwrite a different package version. Patching is not a database
transaction: if a filesystem error interrupts it, retain and inspect backups
before retrying. Existing backup files are never overwritten.

Thirty regression tests cover the actual package implementation. Restart DSH
after they pass. Package upgrades can replace these patches; rerun compatibility
checks after upgrading, not blindly reapply old changes.

## Compaction configuration

In the existing `@deepseek-ai/dsh-compaction-basic` plugin configuration for each
agent preset/profile in use, merge these settings rather than replacing unrelated
configuration:

```yaml
auto: true
thresholdRatio: 0.8
retainRatio: 0.16
maxTokens: 4096
compactionRetries: 0
maxOverflowRetries: 1
```

Remove conflicting fixed `maxThresholdTokens`, `retainTokens`, and per-model
overrides if the intent is proportional behavior across contexts. Set the model
metadata to its actual `32768` context. Standard, code, and Cordis presets can
have independent plugin scopes; changing only the top-level web configuration
may not change those scopes. These settings are not automatically installed.

The `maxTokens` above is the **summary** allowance, not the model's normal answer
limit. Normal output can be configured to 16,384 while the adapter clamps it to
estimated available room. Large tools/system prompts still consume real context.

## Restore

Stop DSH first. In the selected package directory:

- Restore `node_modules/@earendil-works/pi-ai/dist/api/simple-options.js` from
  its `.before-small-context` sibling.
- Restore `node_modules/@deepseek-ai/dsh-compaction-basic/lib/index.js` from
  `.before-checkpoint-safety` to remove both compaction changes. If the broader
  safety patch was already present before this install, only a
  `.before-small-context` backup may exist; that restores that intermediate state.
- Restore edited configuration from the separate configuration backup.

Inspect exact paths before copying. Preserve backups and restart only after
checking the restored package. Original chat event logs are not rewritten by
this installer.
