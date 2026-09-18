#!/usr/bin/env python3
"""Run the published two-prompt/two-seed screening protocol on an idle server."""
import argparse
import json
from pathlib import Path
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8080')
args = parser.parse_args()
fixture = json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/q4-short-fixtures.json').read_text())


def ask(payload):
    request = urllib.request.Request(args.url.rstrip('/') + '/v1/chat/completions',
        json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


ask({'messages': [{'role': 'user', 'content': 'Say ready.'}], 'max_tokens': 16,
     'chat_template_kwargs': {'enable_thinking': False}})
for seed in fixture['seeds']:
    for prompt in fixture['prompts']:
        started = time.monotonic()
        data = ask(dict(fixture['request_defaults'], seed=seed,
            messages=[{'role': 'user', 'content': prompt['prompt']}]))
        timings = data['timings']
        assert timings['predicted_n'] == 128, timings
        assert timings['cache_n'] == 0, timings
        print(json.dumps({'prompt': prompt['name'], 'seed': seed,
            'wall_seconds': time.monotonic()-started, 'timings': timings}), flush=True)
