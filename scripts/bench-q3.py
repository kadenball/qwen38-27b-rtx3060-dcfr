#!/usr/bin/env python3
"""Run the published synthetic fixtures against an already-started, idle Q3 server."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time
import urllib.request

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--fixture', type=Path, default=root/'benchmarks/q3-fixtures.json')
parser.add_argument('--url', default='http://127.0.0.1:8094')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--context', type=int, choices=[98304, 131072], default=131072)
parser.add_argument('--seeds', default='101,202,303,404,505')
parser.add_argument('--occupied-records', type=int, default=0)
parser.add_argument('--gpu-index', type=int, default=0)
parser.add_argument('--label', default='candidate')
args = parser.parse_args()
if not 0 <= args.occupied_records <= 4100 or args.gpu_index < 0:
    parser.error('Invalid record count or GPU index')
seeds = [int(s) for s in args.seeds.split(',')]
url = args.url.rstrip('/')
fixture = json.loads(args.fixture.read_text())
def get(path):
    with urllib.request.urlopen(url + path, timeout=10) as r:
        return json.load(r)
def post(payload):
    request = urllib.request.Request(url + '/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)
props = get('/props')
if props['default_generation_settings']['n_ctx'] != args.context:
    raise SystemExit('Server context does not match --context')
if any(slot.get('is_processing') for slot in get('/slots')):
    raise SystemExit('Server is busy; use a dedicated idle benchmark server')
args.output.mkdir(parents=True, exist_ok=False)
prompts = fixture['prompts']
if args.occupied_records:
    prefix = '\n'.join(f'Record {i:05d}: module {i%37} has count {(i*17)%997}, status available, checksum {(i*79)%8191}.' for i in range(args.occupied_records))
    prompts = [dict(fixture['prompts'][0], name='occupied-rust', prompt=prefix+'\nUse the above as sample data. '+fixture['prompts'][0]['prompt'])]
samples, errors = [], []
stop = threading.Event()
def monitor():
    while not stop.is_set():
        try:
            result = subprocess.check_output(['nvidia-smi', f'--id={args.gpu_index}', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=5)
            samples.append(list(map(int, result.strip().split(','))))
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            errors.append(type(e).__name__)
        stop.wait(1)
thread = threading.Thread(target=monitor, daemon=True)
thread.start()
report = dict(fixtureSha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(), label=args.label, context=args.context, seeds=seeds, occupiedRecords=args.occupied_records, runs=[], status='running')
try:
    post({'messages':[{'role':'user','content':'Say ready.'}],'max_tokens':16,'chat_template_kwargs':{'enable_thinking':False}})
    for seed in seeds:
        for prompt in prompts:
            payload = dict(fixture['request_defaults'], seed=seed, max_tokens=prompt['outputTokens'],
                messages=[{'role':'user','content':prompt['prompt']}],
                chat_template_kwargs={'enable_thinking':prompt['thinking'],'preserve_thinking':True,'reasoning_effort':'medium'})
            if prompt['thinking']:
                payload.update(fixture['reasoning_defaults'])
            start = time.monotonic()
            result = post(payload)
            timing = result['timings']
            if result.get('truncated') or timing['cache_n'] != 0 or timing['predicted_n'] != prompt['outputTokens']:
                raise RuntimeError('Truncation, cached input or unexpected output length invalidates the comparison')
            message = result['choices'][0]['message']
            digest = hashlib.sha256(json.dumps(message, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            row = dict(prompt=prompt['name'], seed=seed, thinking=prompt['thinking'], wall_seconds=time.monotonic()-start,
                       timings=timing, outputMessageSha256=digest)
            report['runs'].append(row)
            (args.output/f"{prompt['name']}-{seed}.json").write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(row), flush=True)
    report['status'] = 'passed'
except BaseException as error:
    report['status'] = 'cancelled' if isinstance(error, KeyboardInterrupt) else 'failed'
    report['errorType'] = type(error).__name__
    raise
finally:
    stop.set()
    thread.join(timeout=6)
    report['memory'] = dict(sampleIntervalSeconds=1, scope='whole device, including other applications',
        minimumFreeMiB=min((s[2] for s in samples), default=None), peakUsedMiB=max((s[1] for s in samples), default=None),
        samples=len(samples), monitorErrors=len(errors), note='One-second sampling can miss transient peaks')
    report['aggregates'] = []
    for name in sorted({r['prompt'] for r in report['runs']}):
        rows = [r for r in report['runs'] if r['prompt'] == name]
        speeds = [r['timings']['predicted_per_second'] for r in rows]
        accept = [r['timings']['draft_n_accepted']/r['timings']['draft_n'] for r in rows if r['timings'].get('draft_n')]
        report['aggregates'].append(dict(prompt=name, runs=len(rows), mean=statistics.mean(speeds), minimum=min(speeds), maximum=max(speeds),
            standardDeviation=statistics.stdev(speeds) if len(speeds)>1 else None,
            meanDraftAcceptance=statistics.mean(accept) if accept else None))
    (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report['aggregates'], indent=2))
