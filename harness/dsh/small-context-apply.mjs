// Version-pinned local runtime patch. Refuse unknown packages; never restart services.
import {readFile,writeFile,copyFile,constants} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import { resolve } from 'node:path';
if (!process.env.DSH_ROOT) throw new Error('Set DSH_ROOT to the installed @deepseek-ai/dsh package directory.');
const base=resolve(process.env.DSH_ROOT, 'node_modules')+'/';
const hash=s=>createHash('sha256').update(s).digest('hex');
const patches=[{
 file:base+'@earendil-works/pi-ai/dist/api/simple-options.js',
 before:'74dfde37adbd00a6af1fd707c1c5c876577793b078da9fbbd6d40bb75bfb4749',
 after:'6931bb979e9bee29fd0cf429221a3ff36894147ae11d3cb0b87ec18caaa86bcd',
 from:'    const available = model.contextWindow - estimateContextTokens(context).tokens - CONTEXT_SAFETY_TOKENS;',
 to: String.raw`    // Local small windows must retain useful answer room, not a fixed 4K reserve.
    const local = model.provider === "routeweaver" && /^http:\/\/(127\.0\.0\.1|localhost):8080(?:\/|$)/.test(model.baseUrl ?? "");
    const safety = local ? Math.min(CONTEXT_SAFETY_TOKENS, Math.max(512, Math.floor(model.contextWindow * 0.05))) : CONTEXT_SAFETY_TOKENS;
    const available = model.contextWindow - estimateContextTokens(context).tokens - safety;
    if (local && maxTokens >= 128 && available < 128)
        throw new Error("Local model context is full: compact the conversation, reduce tool/context overhead, or select a larger context before continuing.");`
},{
 file:base+'@deepseek-ai/dsh-compaction-basic/lib/index.js',
 before:'8607af2177941b6f4230e1697c695edd44d2c0060cd87fc666839657e4a7ac07',
 after:'f4c4a5f0587b8c5f4aa8433d3fe90d47d56cc6a43ddd4782dbf44bc3ba3736bf',
 from:'\tconst candidates = surfaceNodes.slice(0, keepFromIdx);',
 to:'\tconst candidates = surfaceNodes.slice(0, keepFromIdx);\n\t// A structured checkpoint plus task anchors cannot usefully replace a tiny prefix.\n\t// Leave history intact; fixed tool/system overhead is not compactable history.\n\tif (pricedNodes.slice(0, keepFromIdx).reduce((sum, node) => sum + node.tokens, 0) < 512) return null;'
}];
// Preflight every target and replacement before writing any package.
const checked=await Promise.all(patches.map(async p=>{
 const source=await readFile(p.file,'utf8');
 const current=hash(source);
 if(current===p.after) return {...p,done:true};
 if(current!==p.before) throw new Error('Unknown runtime version; preserve and port tests first: '+p.file);
 const replacement=source.replace(p.from,()=>p.to).replace(/\n?$/, '\n');
 if(hash(replacement)!==p.after) throw new Error('Patch output mismatch: '+p.file);
 return {...p,replacement};
}));
for(const p of checked){
 if(!p.done){await copyFile(p.file,p.file+'.before-small-context',constants.COPYFILE_EXCL);await writeFile(p.file,p.replacement);}
 console.log((p.done?'Verified: ':'Patched: ')+p.file);
}
export {patches};
