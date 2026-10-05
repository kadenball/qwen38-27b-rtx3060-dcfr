#!/usr/bin/env python3
"""Offline packaging, launcher and published-result consistency checks."""
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import tempfile
import threading
import unittest
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]

class ReleaseTests(unittest.TestCase):
    def test_manifest(self):
        lines = (ROOT/'SHA256SUMS').read_text().splitlines()
        self.assertGreater(len(lines), 65)
        for line in lines:
            digest, name = line.split('  ', 1)
            with self.subTest(file=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), digest)

    def test_shell_syntax(self):
        for script in (ROOT/'scripts').glob('*.sh'):
            subprocess.run(['bash', '-n', str(script)], check=True)

    def launch(self, **settings):
        return subprocess.run(['bash', str(ROOT/'scripts/serve-q3.sh'), '--print-config'],
            env=dict(os.environ, **settings), capture_output=True, text=True)

    def test_profiles(self):
        for variant, context, blocks, output in [('candidate','98304',36,32768),('candidate','131072',46,65536),
                                                 ('baseline','98304',44,32768),('baseline','131072',54,65536)]:
            with self.subTest(variant=variant, context=context):
                result = self.launch(Q3_VARIANT=variant, CONTEXT=context, HOST_FFN_BLOCKS=str(blocks), MAX_OUTPUT_TOKENS=str(output))
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                value = lambda flag: args[args.index(flag)+1]
                self.assertIn('/build/q3-'+variant+'/bin/llama-server', args[0])
                self.assertEqual(value('--ctx-size'), context)
                self.assertEqual(value('--n-predict'), str(output))
                self.assertEqual(value('--host'), '127.0.0.1')
                self.assertEqual(value('--parallel'), '1')
                self.assertEqual(args.count('--spec-type'), 1)
                self.assertEqual(value('--override-tensor'), r'blk\.('+'|'.join(map(str, range(blocks)))+r')\.ffn_(gate|up)\.weight=CPU')
                self.assertIn('--no-mmproj', args)

    def test_vision_and_all_gpu(self):
        result = self.launch(HOST_FFN_BLOCKS='0', MMPROJ_PATH='/example/projector.gguf')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = shlex.split(result.stdout)
        self.assertNotIn('--override-tensor', args)
        self.assertNotIn('--no-mmproj', args)
        self.assertIn('--no-mmproj-offload', args)

    def test_invalid_profiles(self):
        for settings in [{'CONTEXT':'32768'}, {'CONTEXT':'131073'}, {'HOST_FFN_BLOCKS':'65'},
                         {'THREADS':'0'}, {'PORT':'65536'}, {'MTP_DEPTH':'11'}, {'MAX_OUTPUT_TOKENS':'131072'},
                         {'HOST_FFN_BLOCKS':'08'}, {'Q3_VARIANT':'unknown'}, {'THREADS':'1; false'}]:
            with self.subTest(settings=settings):
                self.assertNotEqual(self.launch(**settings).returncode, 0)

    def test_results_recompute(self):
        report = json.loads((ROOT/'benchmarks/q3-cache-safe-20261005.json').read_text())
        arms = report['runs'][:2]
        for comparison in report['comparison']:
            hashes = []
            for arm, speed_key, acceptance_key in zip(arms, ['baselineTokensPerSecond','candidateTokensPerSecond'], ['baselineAcceptance','candidateAcceptance']):
                rows = [r for r in arm['runs'] if r['prompt'] == comparison['prompt']]
                self.assertEqual(len(rows), 5)
                values = [r['timings']['predicted_per_second'] for r in rows]
                self.assertAlmostEqual(statistics.mean(values), comparison[speed_key]['mean'])
                self.assertAlmostEqual(min(values), comparison[speed_key]['minimum'])
                self.assertAlmostEqual(max(values), comparison[speed_key]['maximum'])
                self.assertAlmostEqual(statistics.stdev(values), comparison[speed_key]['standardDeviation'])
                accept = [r['timings']['draft_n_accepted']/r['timings']['draft_n'] for r in rows]
                self.assertAlmostEqual(statistics.mean(accept), comparison[acceptance_key]['mean'])
                hashes.append({r['seed']:r['outputMessageSha256'] for r in rows})
            self.assertEqual(hashes[0], hashes[1])
            a, b = comparison['baselineTokensPerSecond']['mean'], comparison['candidateTokensPerSecond']['mean']
            self.assertAlmostEqual(100*(b/a-1), comparison['percentImprovement'])
        for arm in report['runs']:
            self.assertEqual(arm['status'], 'passed')
            for row in arm['runs']:
                self.assertEqual(row['timings']['cache_n'], 0)
                self.assertTrue(math.isfinite(row['timings']['predicted_per_second']))
        self.assertEqual(report['occupiedNearFull']['run']['timings']['prompt_n'], 121821)

    def test_release_verification_complete(self):
        report = json.loads((ROOT/'benchmarks/q3-release-verification-20261005.json').read_text())
        self.assertEqual(report['status'], 'passed')
        for name in ['cleanCudaBuild','candidateSourcePins','baselineSourcePins','cpuKernels','recurrentCheckpointCodec','partialPrefetchCleanup']:
            self.assertEqual(report['checks'][name], 'passed')
        self.assertIsInstance(report['standaloneLargeModelGpuRerun'], bool)

    def test_local_markdown_links(self):
        documents = list(ROOT.glob('*.md')) + list((ROOT/'docs').rglob('*.md')) + list((ROOT/'benchmarks').glob('*.md'))
        for doc in documents:
            text = re.sub(r'```.*?```', '', doc.read_text(), flags=re.S)
            for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)', text):
                parsed = urlsplit(target)
                if parsed.scheme or not parsed.path:
                    continue
                with self.subTest(document=doc.relative_to(ROOT), target=target):
                    self.assertTrue((doc.parent/unquote(parsed.path)).exists())

    def test_benchmark_client(self):
        # A fake API tests request shaping and reporting, not model behavior.
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def send_json(self, body):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())
            def do_GET(self):
                self.send_json({'default_generation_settings':{'n_ctx':131072}} if self.path == '/props' else [{'is_processing':False}])
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(request)
                self.send_json({'choices':[{'message':{'role':'assistant','content':'synthetic test reply'}}],
                    'timings':{'cache_n':0,'prompt_n':42,'predicted_n':request['max_tokens'],
                               'predicted_per_second':10,'draft_n':10,'draft_n_accepted':8}})
        with HTTPServer(('127.0.0.1', 0), Handler) as server, tempfile.TemporaryDirectory(prefix='routeweaver-test-') as temp:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = subprocess.run(['python3', str(ROOT/'scripts/bench-q3.py'), '--url', f'http://127.0.0.1:{server.server_port}',
                    '--seeds','101','--output',str(Path(temp)/'results')], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads((Path(temp)/'results/summary.json').read_text())
                self.assertEqual(report['status'], 'passed')
                self.assertEqual(len(report['runs']), 3)
                self.assertEqual([r['max_tokens'] for r in requests], [16,128,128,256])
                self.assertEqual([r['temperature'] for r in requests[1:]], [.7,.7,1.0])
                self.assertEqual([r['min_p'] for r in requests[1:]], [.05,.05,0])
                self.assertTrue(all(not r['cache_prompt'] and r['ignore_eos'] for r in requests[1:]))
                self.assertTrue(requests[-1]['chat_template_kwargs']['enable_thinking'])
            finally:
                server.shutdown()
                thread.join(timeout=5)

    def test_export_boundary(self):
        for line in (ROOT/'SHA256SUMS').read_text().splitlines():
            _, name = line.split('  ', 1)
            path = ROOT/name
            self.assertNotIn(path.suffix, ['.gguf','.so','.key','.pem','.safetensors'])
            if path.suffix == '.png':
                continue
            text = path.read_text()
            forbidden = [r'/(?:home|Users)/[^/\s]+/', r'/tmp/codex', r'trycloudflare\.com', r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
                         r'xox[baprs]-[A-Za-z0-9-]{15,}', r'gh[pousr]_[A-Za-z0-9]{20,}', r'sk-[A-Za-z0-9]{24,}']
            if name == 'tests/test-release.py':
                continue
            for pattern in forbidden:
                self.assertIsNone(re.search(pattern, text), f'Unexpected private/runtime material in {name}')

if __name__ == '__main__':
    unittest.main(verbosity=2)
