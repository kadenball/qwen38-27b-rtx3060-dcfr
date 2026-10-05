"""Synthetic cross-request, tool, edited-prefix, and cancellation checks."""
import argparse
import json
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--url', default='http://127.0.0.1:8080')
URL = parser.parse_args().url.rstrip('/')
rows = []
def post(messages, **kw):
    data = dict(messages=messages, max_tokens=48, temperature=0, seed=101,
                cache_prompt=True, id_slot=0,
                chat_template_kwargs=dict(enable_thinking=False, preserve_thinking=True))
    data.update(kw)
    req = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(data).encode(), {'Content-Type':'application/json'})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.load(response)

def check(name, messages, marker, reuse=False, **kw):
    before = time.monotonic()
    r = post(messages, **kw)
    m = r['choices'][0]['message']
    assert marker in (m.get('content') or ''), (name, m)
    assert not r.get('truncated'), name
    if reuse:
        assert r['timings']['cache_n'] >= 1000, (name, r['timings'])
    rows.append(dict(name=name, seconds=time.monotonic()-before, timings=r['timings']))
    print(json.dumps(rows[-1]), flush=True)
    return m

prefix = '\n'.join(f'Record {i:04d} has color blue and quantity {i % 19 + 1}.' for i in range(160))
base = [{'role':'user', 'content':prefix + '\nVerification marker cedar-482. Reply with the marker only.'}]
first = check('initial', base, 'cedar-482')
follow = base + [first, {'role':'user','content':'Repeat the marker only.'}]
check('cached-followup', follow, 'cedar-482', True)
check('identical-request', follow, 'cedar-482', True)
check('fresh-reference', follow, 'cedar-482', cache_prompt=False)

tools = [{'type':'function','function':dict(name='report_status', description='Report readiness.',
    parameters=dict(type='object', properties=dict(status=dict(type='string', enum=['ready'])),
    required=['status'], additionalProperties=False))}]
tool_messages = follow + [{'role':'assistant','content':'cedar-482'},
    {'role':'user','content':'Call report_status to report status ready.'}]
tool_reply = post(tool_messages, tools=tools, tool_choice='auto', max_tokens=128)
message = tool_reply['choices'][0]['message']
call = message['tool_calls'][0]
assert call['function']['name'] == 'report_status'
assert json.loads(call['function']['arguments']) == {'status':'ready'}
result_messages = tool_messages + [message, {'role':'tool','tool_call_id':call['id'],
    'content':'Status saved. The verification marker is cedar-482. Reply with that marker only.'}]
check('tool-result', result_messages, 'cedar-482', True, tools=tools)

edited = [{'role':'user','content':prefix.replace('Record 0120', 'Edited record 0120') +
    '\nVerification marker maple-901. Reply with the marker only.'}]
check('edited-prefix', edited, 'maple-901')
check('unrelated-request', [{'role':'user','content':'Say only: birch-773'}], 'birch-773')
check('return-to-old-conversation', base, 'cedar-482')

stream_messages = base + [first, {'role':'user','content':'Write a long tutorial on building a crash-consistent event log.'}]
payload = dict(messages=stream_messages, max_tokens=512, stream=True, temperature=0, seed=101,
    cache_prompt=True, id_slot=0, chat_template_kwargs=dict(enable_thinking=False, preserve_thinking=True))
req = urllib.request.Request(URL + '/v1/chat/completions', json.dumps(payload).encode(), {'Content-Type':'application/json'})
received = 0
with urllib.request.urlopen(req, timeout=120) as response:
    for line in response:
        if not line.startswith(b'data: ') or line.strip() == b'data: [DONE]': continue
        event = json.loads(line[6:])
        if event.get('choices', [{}])[0].get('delta', {}).get('content'):
            received += 1
        if received >= 8: break
assert received >= 8, 'No streaming content before cancellation'
for _ in range(100):
    with urllib.request.urlopen(URL + '/slots', timeout=3) as response: slots = json.load(response)
    if not any(s.get('is_processing') for s in slots): break
    time.sleep(.05)
else: raise AssertionError('Cancelled request did not release its slot')
check('after-cancellation', follow, 'cedar-482', True)
print('PASS: cache reuse, exact repeat, tools, edited prefixes, independent chats, cancellation', flush=True)
