// Optional, exact-source-pinned package patch. Run only with DSH stopped.
import {readFile,copyFile,constants} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
if (!process.env.DSH_ROOT) throw new Error('Set DSH_ROOT to the installed @deepseek-ai/dsh package directory.');
const root=resolve(process.env.DSH_ROOT,'node_modules/@deepseek-ai/dsh-compaction-basic');
const target=resolve(root,'lib/index.js');
const pi=resolve(process.env.DSH_ROOT,'node_modules/@earendil-works/pi-ai/dist/api/simple-options.js');
const digest=async file=>createHash('sha256').update(await readFile(file)).digest('hex');
const original='cfe41fdfe4f3aaec2cd5aa1a0187cf4877eac0a6318817cca76c3014dcc1595b';
const stock='144202a0f150b9b7984842d6316808aefcbdf14e7a890805cb0819f4cc69740f';
const intermediate='8607af2177941b6f4230e1697c695edd44d2c0060cd87fc666839657e4a7ac07';
const final='f4c4a5f0587b8c5f4aa8433d3fe90d47d56cc6a43ddd4782dbf44bc3ba3736bf';
const current=await digest(target);
if (![stock,original,intermediate,final].includes(current)) throw new Error('Unknown compaction source; no files changed. Port and retest before applying.');
if (!['74dfde37adbd00a6af1fd707c1c5c876577793b078da9fbbd6d40bb75bfb4749','6931bb979e9bee29fd0cf429221a3ff36894147ae11d3cb0b87ec18caaa86bcd'].includes(await digest(pi))) throw new Error('Unknown pi-ai source; no files changed.');
if (current===original || current===stock) {
    const patch=current===stock ? './stock-compaction.patch' : './checkpoint-safety.patch';
    const expected=current===stock ? final : intermediate;
    const args=['--batch','--fuzz=0','-p1','-i',fileURLToPath(new URL(patch,import.meta.url))];
    if (spawnSync('patch',['--dry-run',...args],{cwd:root,stdio:'inherit'}).status!==0) throw new Error('Patch preflight failed.');
    await copyFile(target,target+'.before-checkpoint-safety',constants.COPYFILE_EXCL);
    if (spawnSync('patch',args,{cwd:root,stdio:'inherit'}).status!==0 || await digest(target)!==expected) throw new Error('Patch failed; original backup retained.');
}
await import('./small-context-apply.mjs');
console.log('Verified patches. Run tests before restarting idle DSH. Configuration is not modified.');
