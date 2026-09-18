#!/usr/bin/env python3
"""Bounded synthetic two-request cache smoke test; no harness/session access."""
import argparse
import json
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--timeout', type=int, default=300)
    args = parser.parse_args()

    def post(path, data):
        request = urllib.request.Request(args.url.rstrip('/') + path,
            json.dumps(data).encode(), {'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            return json.load(response)

    # A long stable prefix makes loss of cache reuse unambiguous.
    prefix = '\n'.join(f'Record {i:04d} has color blue and quantity {i % 19 + 1}.' for i in range(160))
    messages = [{'role': 'user', 'content': prefix + '\nVerification marker cedar-482. Reply with the marker only.'}]
    results = []
    for _ in range(2):
        started = time.monotonic()
        response = post('/v1/chat/completions', {'messages': messages, 'max_tokens': 32,
            'temperature': 0, 'seed': 101, 'cache_prompt': True, 'id_slot': 0,
            'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': True}})
        message = response['choices'][0]['message']
        timing = response.get('timings', {})
        results.append({'wall_seconds': round(time.monotonic()-started, 3),
            'timings': timing, 'marker_present': 'cedar-482' in (message.get('content') or ''),
            'truncated': response.get('truncated', False)})
        messages += [message, {'role': 'user', 'content': 'Repeat the marker only.'}]
    print(json.dumps({'requests': results}, indent=2))
    if results[1]['timings'].get('cache_n', 0) < 1000:
        raise SystemExit('FAIL: less than 1000 cached tokens; check depth, flags, and slot reuse.')
    if any(r['truncated'] for r in results):
        raise SystemExit('FAIL: prompt truncated.')
    if not all(r['marker_present'] for r in results):
        raise SystemExit('FAIL: marker recall failed; inspect output separately before trusting reuse.')
    print('PASS: prompt reuse observed. This is not a long-context quality benchmark.')


if __name__ == '__main__':
    main()
