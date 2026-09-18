import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

import { resolve } from 'node:path';
if (!process.env.DSH_ROOT) throw new Error('Set DSH_ROOT to the installed @deepseek-ai/dsh package directory.');
const live = resolve(process.env.DSH_ROOT, 'node_modules/@deepseek-ai/dsh-compaction-basic/lib/index.js');
const installed = process.env.DSH_COMPACTION_FILE ?? live;
const require = createRequire(live);
const source = (await readFile(installed, 'utf8')).replace(/from "(@[^"\n]+)"/g,
  (_, name) => `from ${JSON.stringify(pathToFileURL(require.resolve(name)).href)}`);
// Exercise installed code rather than a duplicate, exposing private test seams.
const optional = source.includes('function validateCheckpoint(') ? ', validateCheckpoint' : '';
export const mod = await import('data:text/javascript;base64,' + Buffer.from(source + '\nexport { summarizeCompaction, compactSurfaceRegion, selectCompactableRange, buildSummarizationInput, summarizeWithLlm, COMPACTION_INSTRUCTION, resolveConfig, resolveTargetPolicy, resolveCompactSpec '+optional+' };\n//# sourceURL=installed-dsh-compaction-test.js').toString('base64'));
