import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
const { clampMaxTokensToContext } = await import(pathToFileURL(resolve(process.env.DSH_ROOT, 'node_modules/@earendil-works/pi-ai/dist/api/simple-options.js')));
import { mod } from './installed-module.mjs';

const model = {provider:'routeweaver',baseUrl:'http://127.0.0.1:8080/v1',contextWindow:20480};
const context = {messages:[{role:'user',content:'x'.repeat(16365*4),timestamp:1}]};
test('20K local context must not turn ~80% usage into a 19-token reply allowance', () => {
  assert.ok(clampMaxTokensToContext(model,context,16384)>=2048);
});
test('explicit small caller output limit is still honored', () => {
  assert.equal(clampMaxTokensToContext(model,context,16),16);
});
test('nonlocal providers retain upstream safety margin', () => {
  assert.equal(clampMaxTokensToContext({...model,provider:'other'},context,16384),19);
});
test('a remote server named routeweaver retains upstream safety margin', () => {
  assert.equal(clampMaxTokensToContext({...model,baseUrl:'https://example.invalid/v1'},context,16384),19);
});
for (const capacity of [20480,24576,32768,65536,131072,262144]) {
  test(`useful answer allowance at 81% of local ${capacity} context`, () => {
    const used=Math.floor(capacity*.81);
    const allowance=clampMaxTokensToContext({...model,contextWindow:capacity},{messages:[{role:'user',content:'x'.repeat(used*4),timestamp:1}]},16384);
    assert.ok(allowance>=2048 && allowance<=16384);
    assert.ok(used+allowance<capacity);
  });
}
test('exhausted local context fails clearly instead of silently allowing one token', () => {
  assert.throws(()=>clampMaxTokensToContext(model,{messages:[{role:'user',content:'x'.repeat(20480*4),timestamp:1}]},16384),/context|compact/i);
});
test('skip an 11-token compaction prefix crowded by fixed tool overhead', () => {
  const session={events:[0,1].map(seq=>({seq,data:{role:'user',source:{kind:'user'},content:[{type:'text',text:'fixture'}]}})),surface:{nodes:[0,1]},deriveEventMessage:e=>e.data};
  assert.equal(mod.selectCompactableRange(session,{nodes:[{seq:0,tokens:11},{seq:1,tokens:3370}]},3276),null);
});
