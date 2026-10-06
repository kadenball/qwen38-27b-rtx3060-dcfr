"""CPU-only HTTP fixture for tuner lifecycle tests; not an LLM/performance test."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--alias', required=True)
parser.add_argument('--context', type=int, required=True)
parser.add_argument('--draft', type=int, default=10)
args = parser.parse_args()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, value):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(value).encode())

    def do_GET(self):
        if self.path == '/v1/models':
            self.send_json({'data': [{'id': args.alias}]})
        elif self.path == '/props':
            self.send_json({'default_generation_settings': {'n_ctx': args.context}})
        else:
            self.send_json([{'is_processing': False}])

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert data.get('model', args.alias) == args.alias
        if data.get('stream'):
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.end_headers()
            for _ in range(8):
                self.wfile.write(b'data: {"choices":[{"delta":{"content":"test "}}]}\n\n')
            self.wfile.flush()
            return
        last = data['messages'][-1]['content']
        content = 'birch-773' if 'birch-773' in last else ('maple-901' if 'maple-901' in last else 'cedar-482')
        message = {'role': 'assistant', 'content': content}
        if 'Call report_status' in last:
            message['tool_calls'] = [{'id': 'call-test', 'type': 'function', 'function': {
                'name': 'report_status', 'arguments': '{"status":"ready"}'}}]
        self.send_json({'choices': [{'message': message}], 'timings': {
            'cache_n': 1500 if data.get('cache_prompt') else 0, 'prompt_n': 4000,
            'predicted_n': data['max_tokens'], 'predicted_per_second': 20.0,
            'prompt_per_second': 200.0, 'draft_n': args.draft,
            'draft_n_accepted': min(8, args.draft)}})


ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
