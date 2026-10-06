"""Conservative, project-local Q3 setup and empirical tuning; standard library only.

Detection is read-only. Trials own one subprocess group and an identity-checked
loopback endpoint. Profiles are JSON, never sourced as shell instructions.
Hardware probing, trial policy, and profile persistence have independent seams.
"""
import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass, replace
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = 'RVN-Qwen3.8-27B-Heretic-GSQ-RCO-IQ3_XXS-mtp.gguf'
MODEL_HASH = '475be499f4bc4f729a811e24ad11419bd187d345af18eff494b89c8ba55a0039'
TEMPLATE_HASH = 'c47c82b0544752d454f4e427228d9d9d8c3df64c9e446cbd0229362f67948009'
MODEL_BYTES = 10569270720
SCHEMA = 1
MIB = 1024 * 1024
STATE = ROOT / 'build/routeweaver'
MANAGED_ENV = {'CONTEXT', 'HOST_FFN_BLOCKS', 'THREADS', 'MTP_DEPTH', 'SPECULATION',
               'BATCH_SIZE', 'UBATCH_SIZE', 'MAX_OUTPUT_TOKENS', 'PORT', 'Q3_VARIANT',
               'MODEL_PATH', 'MODEL_DIR', 'CHAT_TEMPLATE_PATH', 'MMPROJ_PATH',
               'LD_PRELOAD', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES'}


class Refused(RuntimeError):
    """An unsafe/unsupported operation or incomplete measurement, not a fallback."""


def output(argv, timeout=15):
    return subprocess.check_output(argv, text=True, stderr=subprocess.PIPE, timeout=timeout).strip()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * MIB), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def exclusive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise Refused('Another RouteWeaver operation holds the lock.') from error
        yield


def memory():
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    return {'available': int(info['MemAvailable'].split()[0]) // 1024,
            'total': int(info['MemTotal'].split()[0]) // 1024,
            'swap_out': int(vm['pswpout']) * os.sysconf('SC_PAGE_SIZE') // MIB}


def cpu_info():
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise Refused('Automatic setup currently supports Linux x86-64 only; see the manual guide.')
    allowed = os.sched_getaffinity(0)
    records = []
    for paragraph in Path('/proc/cpuinfo').read_text().split('\n\n'):
        row = dict(line.split(':', 1) for line in paragraph.splitlines() if ':' in line)
        row = {k.strip(): v.strip() for k, v in row.items()}
        if row.get('processor', '').isdigit() and int(row['processor']) in allowed:
            records.append(row)
    if not records:
        raise Refused('Cannot identify allowed CPU cores.')
    flags = set.intersection(*(set(r.get('flags', '').split()) for r in records))
    if not {'avx2', 'fma'} <= flags:
        raise Refused('The automatic Q3 path requires AVX2 and FMA; use manual compatibility testing.')
    cores = {(r.get('physical id', '0'), r.get('core id', r['processor'])) for r in records}
    return {'name': records[0].get('model name', 'unknown'), 'physical_cores': len(cores),
            'logical_cpus': len(records), 'flags': sorted(flags)}


def gpu_info(selector):
    text = output(['nvidia-smi', '--id=' + str(selector),
                   '--query-gpu=index,name,uuid,memory.total,memory.free,driver_version,compute_cap',
                   '--format=csv,noheader,nounits'])
    rows = list(csv.reader(text.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 7:
        raise Refused('Select exactly one NVIDIA GPU with --gpu INDEX.')
    index, name, identifier, total, free, driver, capability = (x.strip() for x in rows[0])
    return {'index': int(index), 'name': name, 'uuid': identifier, 'total': int(total),
            'free': int(free), 'driver': driver, 'capability': capability}


def compute_processes(gpu):
    text = output(['nvidia-smi', '--id=' + gpu['uuid'],
                   '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
    rows = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 2 or not row[0].strip().isdigit():
            raise Refused('Cannot reliably inspect GPU compute processes.')
        rows.append({'pid': int(row[0]), 'name': Path(row[1].split(' ')[0]).name})
    return rows


def require_idle(gpu):
    processes = compute_processes(gpu)
    if processes:
        names = ', '.join(f"{p['name']} (PID {p['pid']})" for p in processes)
        raise Refused('GPU compute processes are already running: ' + names +
                      '. Stop them yourself before tuning/starting; nothing was terminated.')


def detect(selector):
    cpu, gpu, ram = cpu_info(), gpu_info(selector), memory()
    if gpu['total'] < 12000:
        raise Refused('The first automatic release requires a 12 GB+ NVIDIA GPU. Manual tuning remains available.')
    return {'cpu': cpu, 'gpu': gpu, 'ram_total_mib': ram['total']}


def clean_env(hardware, config, context, port, model, template):
    env = {k: v for k, v in os.environ.items()
           if k not in MANAGED_ENV and not k.startswith(('GGML_', 'LLAMA_'))}
    env.update(CUDA_VISIBLE_DEVICES=hardware['gpu']['uuid'], Q3_VARIANT='candidate',
               CONTEXT=str(context), PORT=str(port), MODEL_PATH=str(model), CHAT_TEMPLATE_PATH=str(template),
               HOST_FFN_BLOCKS=str(config.host_blocks), THREADS=str(config.threads),
               MTP_DEPTH=str(config.depth), SPECULATION=config.speculation,
               BATCH_SIZE=str(config.batch), UBATCH_SIZE=str(config.ubatch))
    return env


@dataclass(frozen=True)
class Config:
    host_blocks: int
    threads: int
    depth: int = 4
    speculation: str = 'mtp'
    batch: int = 128
    ubatch: int = 128

    def validate(self):
        if any(type(x) is not int for x in (self.host_blocks, self.threads, self.depth, self.batch, self.ubatch)):
            raise Refused('Profile settings must be integers.')
        if not (0 <= self.host_blocks <= 64 and 1 <= self.threads <= 256 and 1 <= self.depth <= 10
                and 32 <= self.ubatch <= self.batch <= 512 and self.speculation in ('mtp', 'none')):
            raise Refused('Profile settings are outside supported bounds.')
        return self


def initial_config(hardware, context, margin):
    base = 36 if context == 98304 else 46
    # Conservative starting estimate, not a model-independent memory predictor.
    blocks = base + math.ceil((12288 - hardware['gpu']['total'] + margin - 384) / 88)
    return Config(max(0, min(64, blocks)), min(16, hardware['cpu']['physical_cores']))


def candidates(base, hardware):
    # One knob at a time. No-speculation is real plain decode, not merely depth 1.
    yield replace(base, speculation='none', depth=1)
    for threads in sorted({max(1, base.threads // 2), min(32, hardware['cpu']['logical_cpus']),
                           min(256, hardware['cpu']['physical_cores'])} - {base.threads}):
        yield replace(base, threads=threads)
    if base.host_blocks >= 4:
        yield replace(base, host_blocks=base.host_blocks - 4)
    yield replace(base, batch=256)
    yield replace(base, depth=2)
    yield replace(base, depth=6)
    yield replace(base, batch=64, ubatch=64)
    if base.host_blocks >= 8:
        yield replace(base, host_blocks=base.host_blocks - 8)


def verify_assets(model, template):
    for name, path, expected in [('model', model, MODEL_HASH), ('template', template, TEMPLATE_HASH)]:
        if not path.is_file() or digest(path) != expected:
            raise Refused(f'The {name} is missing or its SHA-256 differs from the pinned release.')


def identity(hardware, context):
    directory = ROOT / 'build/q3-candidate/bin'
    server = directory / 'llama-server'
    if not server.is_file():
        raise Refused('Build the candidate runtime first with routeweaver setup or the manual guide.')
    files = [server] + sorted(p for p in directory.glob('*.so*') if not p.is_symlink())
    runtime = {p.name: digest(p) for p in files}
    # Include policy/launch code, so profiles cannot silently outlive behavioral changes.
    for name in ['scripts/routeweaver_cli.py', 'scripts/q3-env.sh', 'scripts/serve-q3.sh']:
        runtime[name] = digest(ROOT / name)
    gpu = {k: hardware['gpu'][k] for k in ('uuid', 'name', 'total', 'driver', 'capability')}
    data = {'schema': SCHEMA, 'cpu': hardware['cpu'], 'gpu': gpu,
            'ram_total_mib': hardware['ram_total_mib'], 'context': context,
            'model_sha256': MODEL_HASH, 'template_sha256': TEMPLATE_HASH, 'runtime': runtime,
            'kv': 'q4_0/q4_0', 'vision': False}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def save_profile(path, profile):
    previous = path.with_suffix('.previous.json')
    if path.exists():
        old = json.loads(path.read_text())
        atomic_json(previous, old)
    atomic_json(path, profile)


def read_profile(path, fingerprint, context):
    if not path.is_file():
        raise Refused('No saved profile for this selection. Run setup/tune first, or use the manual launcher.')
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA or data.get('fingerprint') != fingerprint or data.get('context') != context:
        raise Refused('Saved profile does not match this hardware/runtime/context. Retune or use the manual launcher.')
    if data.get('status') != 'validated' or data.get('model_sha256') != MODEL_HASH:
        raise Refused('Profile has not passed validation for the pinned model.')
    Config(**data['config']).validate()
    for key, low, high in [('margin_mib', 512, 8192), ('required_free_vram_mib', 512, 1048576),
                           ('required_available_ram_mib', 4096, 1048576)]:
        if type(data.get(key)) is not int or not low <= data[key] <= high:
            raise Refused('Invalid saved memory limit.')
    if data['required_free_vram_mib'] < data['margin_mib']:
        raise Refused('Invalid saved memory margin.')
    return data


class Budget:
    def __init__(self, seconds):
        self.end = time.monotonic() + seconds

    def left(self):
        return max(0.0, self.end - time.monotonic())

    def timeout(self, maximum):
        if self.left() < 1:
            raise Refused('Tuning time budget exhausted; existing profile is unchanged.')
        return min(maximum, self.left())


class Guard:
    """Observe headroom/paging and stop only the owned trial if safety changes."""
    def __init__(self, hardware, margin, process, budget):
        self.hardware, self.margin, self.process, self.budget = hardware, margin, process, budget
        self.done = threading.Event()
        self.failure = None
        self.minimum_free = None
        self.minimum_ram = None
        self.start_swap = memory()['swap_out']
        self.thread = threading.Thread(target=self.watch, daemon=True)

    def watch(self):
        try:
            while not self.done.is_set():
                ram, gpu = memory(), gpu_info(self.hardware['gpu']['uuid'])
                self.minimum_free = min(self.minimum_free or gpu['free'], gpu['free'])
                self.minimum_ram = min(self.minimum_ram or ram['available'], ram['available'])
                foreign = [p for p in compute_processes(gpu) if p['pid'] != self.process.pid]
                if foreign:
                    raise Refused('Another GPU compute process appeared during tuning.')
                if ram['available'] < 1536 or ram['swap_out'] - self.start_swap > 64:
                    raise Refused('RAM headroom/paging safety limit reached.')
                if gpu['free'] < self.margin:
                    raise Refused(f"GPU headroom below the requested margin ({gpu['free']} < {self.margin} MiB).")
                if self.budget.left() <= 0:
                    raise Refused('Tuning time budget exhausted.')
                self.done.wait(.25)
        except Exception as error:
            self.failure = str(error)
            terminate_owned(self.process)

    def check(self):
        if self.failure:
            raise Refused(self.failure)
        if self.process.poll() is not None:
            raise Refused('Trial server exited; inspect its local log.')


def terminate_owned(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass


def run_owned(command, *, env, timeout=None, stdout=None):
    process = subprocess.Popen(command, env=env, stdout=stdout,
                               stderr=subprocess.STDOUT if stdout is not None else None,
                               start_new_session=True)
    try:
        result = process.wait(timeout=timeout)
        if result:
            raise subprocess.CalledProcessError(result, command)
    finally:
        terminate_owned(process)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


class Trial:
    def __init__(self, hardware, context, config, model, template, margin, budget, directory):
        self.hardware, self.context, self.config = hardware, context, config
        self.model, self.template, self.margin, self.budget = model, template, margin, budget
        self.directory, self.process, self.guard, self.log = directory, None, None, None
        self.starting_free = None
        self.port = free_port()
        self.alias = 'routeweaver-tuning-' + uuid.uuid4().hex
        self.url = f'http://127.0.0.1:{self.port}'
        # Never send localhost requests through an inherited HTTP proxy.
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, path, body=None, maximum=120):
        if self.guard:
            self.guard.check()
        request = urllib.request.Request(self.url + path,
            None if body is None else json.dumps(body).encode(), {'Content-Type': 'application/json'})
        try:
            with self.http.open(request, timeout=self.budget.timeout(maximum)) as response:
                result = json.load(response)
        except Exception:
            # A safety stop closes in-flight HTTP requests. Preserve the actual
            # monitor reason instead of reporting only a connection reset.
            if self.guard:
                self.guard.check()
            raise
        if self.guard:
            self.guard.check()
        return result

    def __enter__(self):
        try:
            require_idle(self.hardware['gpu'])
            # Estimate CPU-resident gate/up storage plus operating headroom before load.
            if memory()['available'] < max(4096, self.config.host_blocks * 96 + 2048):
                raise Refused('Not enough available RAM for this placement plus safety headroom.')
            self.config.validate()
            self.starting_free = gpu_info(self.hardware['gpu']['uuid'])['free']
            self.directory.mkdir(parents=True, mode=0o700)
            env = clean_env(self.hardware, self.config, self.context, self.port, self.model, self.template)
            command = shlex.split(subprocess.check_output(
                ['bash', str(ROOT / 'scripts/serve-q3.sh'), '--print-config'], env=env, text=True))
            command[command.index('--alias') + 1] = self.alias
            self.log = (self.directory / 'server.log').open('w')
            # Hashes were checked once by the parent. No shared server or shell profile is used.
            self.process = subprocess.Popen(['bash', '-c', 'source "$1/scripts/q3-env.sh"; shift; exec "$@"',
                'routeweaver', str(ROOT), *command], env=env, stdout=self.log, stderr=subprocess.STDOUT,
                start_new_session=True)
            self.guard = Guard(self.hardware, self.margin, self.process, self.budget)
            self.guard.thread.start()
            until = time.monotonic() + self.budget.timeout(180)
            while time.monotonic() < until:
                self.guard.check()
                try:
                    models = self.request('/v1/models', maximum=2)
                    if self.alias not in [item['id'] for item in models.get('data', [])]:
                        raise Refused('Port identity mismatch; refusing to contact another server.')
                    props = self.request('/props', maximum=2)
                    if props['default_generation_settings']['n_ctx'] != self.context:
                        raise Refused('The server changed the requested context.')
                    return self
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(.2)
            raise Refused('Trial server did not become ready within the load timeout.')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.guard:
            self.guard.done.set()
            self.guard.thread.join(timeout=35)
        if self.process:
            terminate_owned(self.process)
        if self.log:
            self.log.close()

    def __exit__(self, *_):
        self.close()

    def measure(self, seeds):
        fixture = json.loads((ROOT / 'benchmarks/q3-fixtures.json').read_text())
        prompts = [dict(p, outputTokens=64 if p['thinking'] else 48) for p in fixture['prompts']]
        prefix = '\n'.join(f'Record {i:04d}: component {i % 37} count {(i * 17) % 997} status ready.' for i in range(160))
        prompts.append(dict(prompts[0], name='prefill-coding', prompt=prefix + '\n' + prompts[0]['prompt']))
        self.request('/v1/chat/completions', {'model': self.alias,
            'messages': [{'role': 'user', 'content': 'Say ready.'}], 'max_tokens': 8,
            'chat_template_kwargs': {'enable_thinking': False}})
        rows = []
        for seed in seeds:
            for prompt in prompts:
                body = dict(fixture['request_defaults'], model=self.alias, seed=seed,
                    max_tokens=prompt['outputTokens'], messages=[{'role': 'user', 'content': prompt['prompt']}],
                    chat_template_kwargs={'enable_thinking': prompt['thinking'], 'preserve_thinking': True,
                                          'reasoning_effort': 'medium'})
                if prompt['thinking']:
                    body.update(fixture['reasoning_defaults'])
                started = time.monotonic()
                response = self.request('/v1/chat/completions', body)
                timing = response['timings']
                if response.get('truncated') or timing['cache_n'] or timing['predicted_n'] != prompt['outputTokens']:
                    raise Refused('Truncated/cached input or incorrect output length invalidates the trial.')
                if self.config.speculation == 'none' and timing.get('draft_n', 0):
                    raise Refused('The no-speculation reference unexpectedly drafted tokens.')
                for key in ('predicted_per_second', 'prompt_per_second'):
                    if not math.isfinite(timing[key]) or timing[key] <= 0:
                        raise Refused('Invalid throughput measurement.')
                message = response['choices'][0]['message']
                rows.append({'fixture': prompt['name'], 'seed': seed, 'wall_seconds': time.monotonic() - started,
                             'timings': timing, 'output_sha256': hashlib.sha256(
                                 json.dumps(message, sort_keys=True).encode()).hexdigest()})
        return {'config': asdict(self.config), 'rows': rows,
                'starting_free_vram_mib': self.starting_free,
                'minimum_free_vram_mib': self.guard.minimum_free,
                'minimum_available_ram_mib': self.guard.minimum_ram}

    def memory_result(self):
        self.guard.check()
        return {'starting_free_vram_mib': self.starting_free,
                'minimum_free_vram_mib': self.guard.minimum_free,
                'minimum_available_ram_mib': self.guard.minimum_ram}

    def validate_chat(self):
        env = dict(os.environ, NO_PROXY='127.0.0.1,localhost', no_proxy='127.0.0.1,localhost')
        with (self.directory / 'chat-check.log').open('w') as log:
            run_owned(['python3', str(ROOT / 'scripts/check-q3-chat.py'), '--url', self.url],
                      env=env, stdout=log, timeout=self.budget.timeout(180))
        self.guard.check()


def metrics(report):
    groups = {}
    for row in report['rows']:
        groups.setdefault(row['fixture'], []).append(row)
    if len(groups) != 4:
        raise Refused('Incomplete trial: all four fixtures are required.')
    result = {name: statistics.median(r['timings']['predicted_per_second'] for r in rows)
              for name, rows in groups.items()}
    result['prefill'] = statistics.median(r['timings']['prompt_per_second'] for r in groups['prefill-coding'])
    return result


def score(report):
    return statistics.geometric_mean(metrics(report).values())


def acceptable(candidate, reference):
    a, b = metrics(candidate), metrics(reference)
    return all(a[k] >= .9 * b[k] for k in b)


def model_free_checks(hardware, context, model, template):
    env = clean_env(hardware, initial_config(hardware, context, 768), context, 8080, model, template)
    env['CUDA_VISIBLE_DEVICES'] = '-1'
    print('Running model-free kernel/checkpoint/cleanup checks before GPU trials.', flush=True)
    run_owned(['bash', str(ROOT / 'scripts/check-q3.sh')], env=env, timeout=120)


def tune(hardware, args, model, template, profile_path, trial_factory=Trial):
    require_idle(hardware['gpu'])
    verify_assets(model, template)
    fingerprint = identity(hardware, args.context)
    model_free_checks(hardware, args.context, model, template)
    budget = Budget(args.budget_seconds)
    run_dir = Path(tempfile.mkdtemp(prefix='tune-', dir=STATE))
    report = {'schema': SCHEMA, 'status': 'running', 'context': args.context,
              'model_sha256': MODEL_HASH, 'margin_mib': args.margin_mib, 'trials': [],
              'scope': 'Bounded short synthetic screen, not a quality or near-full-context certification'}
    attempts = 0

    def run(config, seeds=(101,), validate=False):
        nonlocal attempts
        attempts += 1
        print(f'Trial {attempts}: {asdict(config)}', flush=True)
        item = {'config': asdict(config), 'status': 'running'}
        report['trials'].append(item)
        try:
            with trial_factory(hardware, args.context, config, model, template, args.margin_mib,
                               budget, run_dir / f'trial-{attempts:02d}') as trial:
                result = trial.measure(seeds)
                if validate:
                    trial.validate_chat()
                result.update(trial.memory_result())
            item.update(result, status='passed', score=score(result), chat_checks=validate)
            print(f"  score {item['score']:.2f}; free VRAM >= {item['minimum_free_vram_mib']} MiB", flush=True)
            return item
        except (Refused, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            item.update(status='failed', error=str(error))
            print('  rejected: ' + str(error), flush=True)
            return None
        finally:
            atomic_json(run_dir / 'report.json', report)

    try:
        base = initial_config(hardware, args.context, args.margin_mib)
        reference = None
        while attempts < args.max_trials - 4 and budget.left() > 240:
            reference = run(base)
            if reference:
                break
            if base.host_blocks == 64:
                break
            base = replace(base, host_blocks=min(64, base.host_blocks + 4))
        if reference is None:
            raise Refused('No safe reference fit within the trial/time budget. No profile installed.')
        ranked = [reference]
        for config in candidates(base, hardware):
            if attempts >= args.max_trials - 3 or budget.left() <= 240:
                break
            result = run(config)
            if result and acceptable(result, reference):
                ranked.append(result)
        # Require a measured plain-decode arm before selecting any profile.
        if not any(t.get('status') == 'passed' and t['config']['speculation'] == 'none' for t in report['trials']):
            raise Refused('No successful no-speculation reference; leaving the existing profile unchanged.')
        chosen = max(ranked, key=score)
        if score(chosen) < 1.03 * score(reference):
            chosen = reference
        confirmed_reference = run(base, (202, 303))
        if not confirmed_reference:
            raise Refused('Reference confirmation failed; no profile installed.')
        chosen_config = Config(**chosen['config'])
        confirmation = run(chosen_config, (202, 303), validate=True)
        if not confirmation or not acceptable(confirmation, confirmed_reference):
            raise Refused('Winner confirmation/correctness failed; no profile installed.')
        if chosen_config != base and score(confirmation) < 1.03 * score(confirmed_reference):
            print('Improvement did not repeat by 3%; validating the safe reference instead.', flush=True)
            chosen_config = base
            confirmation = run(base, (202, 303), validate=True)
            if not confirmation or not acceptable(confirmation, confirmed_reference):
                raise Refused('Fallback validation failed; no profile installed.')
        if identity(hardware, args.context) != fingerprint:
            raise Refused('Runtime changed during tuning; no profile installed.')
        profile = {'schema': SCHEMA, 'status': 'validated', 'fingerprint': fingerprint,
                   'context': args.context, 'config': asdict(chosen_config),
                   'model_sha256': MODEL_HASH, 'model_path': str(model), 'template_path': str(template),
                   'required_free_vram_mib': max(0, confirmation['starting_free_vram_mib'] -
                       confirmation['minimum_free_vram_mib']) + args.margin_mib,
                   'required_available_ram_mib': max(4096, chosen_config.host_blocks * 96 + 2048),
                   'margin_mib': args.margin_mib, 'report_path': str(run_dir / 'report.json'),
                   'validation': 'Two confirmation seeds, four fixtures, cache/tool/cancellation checks',
                   'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        save_profile(profile_path, profile)
        report.update(status='passed', selected=asdict(chosen_config))
        print(f'Saved validated profile: {profile_path}\nLaunch with: ./routeweaver start --context {args.context}')
    except BaseException as error:
        report.update(status='cancelled' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        atomic_json(run_dir / 'report.json', report)
        print(f'Local tuning report: {run_dir / "report.json"}')


def toolchain(hardware):
    nvcc = shutil.which('nvcc')
    cuda = Path(os.environ.get('CUDA_TOOLKIT_ROOT') or (str(Path(nvcc).resolve().parent.parent) if nvcc else '/usr/local/cuda'))
    if not (cuda / 'bin/nvcc').is_file():
        raise Refused('CUDA toolkit not found. Install it, or set CUDA_TOOLKIT_ROOT; the NVIDIA driver alone is insufficient.')
    explicit = bool(os.environ.get('CC') or os.environ.get('CXX'))
    pairs = [(os.environ.get('CC', 'cc'), os.environ.get('CXX', 'c++'))] if explicit else [
        ('gcc', 'g++'), ('gcc-15', 'g++-15'), ('gcc-14', 'g++-14'), ('gcc-13', 'g++-13'), ('gcc-12', 'g++-12')]
    arch = hardware['gpu']['capability'].replace('.', '')
    if not arch.isdigit():
        raise Refused('Cannot determine the CUDA architecture.')
    with tempfile.TemporaryDirectory(prefix='routeweaver-compiler-') as directory:
        for cc, cxx in pairs:
            if not shutil.which(cc) or not shutil.which(cxx):
                continue
            probe = subprocess.run([str(cuda / 'bin/nvcc'), '-ccbin', cxx, '-arch=sm_' + arch,
                '-c', str(ROOT / 'scripts/cuda-probe.cu'), '-o', str(Path(directory) / 'probe.o')],
                capture_output=True, text=True, timeout=60)
            if probe.returncode == 0:
                return {'CUDA_TOOLKIT_ROOT': str(cuda), 'CC': cc, 'CXX': cxx, 'CUDA_ARCHITECTURES': arch,
                        'BUILD_JOBS': '1', 'Q3_VARIANT': 'candidate'}
    raise Refused('No compatible CUDA/host-compiler pair passed the compile probe. Set CC/CXX explicitly; see the manual guide.')


def build_runtime(hardware, env):
    directory = ROOT / 'build/q3-candidate'
    stamp = directory / 'routeweaver-build-host.json'
    target = {'cpu': hardware['cpu'], 'cuda_architecture': env['CUDA_ARCHITECTURES'],
              'cc': shutil.which(env['CC']), 'cxx': shutil.which(env['CXX']),
              'compiler_version': output([env['CXX'], '--version']),
              'cuda_version': output([str(Path(env['CUDA_TOOLKIT_ROOT']) / 'bin/nvcc'), '--version'])}
    matching = False
    if stamp.is_file():
        try:
            matching = json.loads(stamp.read_text()) == target
        except json.JSONDecodeError:
            pass  # Unknown build identity is preserved, then rebuilt, never trusted.
    previous = None
    if directory.exists() and not matching:
        previous = directory.with_name('q3-candidate.saved-' + uuid.uuid4().hex[:10])
        directory.rename(previous)
        print(f'Preserved the previous native build at {previous}; rebuilding for detected hardware.', flush=True)
    try:
        run_owned(['bash', str(ROOT / 'scripts/build-q3.sh')], env=env)
        atomic_json(stamp, target)
    except BaseException:
        if previous is not None:
            if directory.exists():
                directory.rename(directory.with_name('q3-candidate.incomplete-' + uuid.uuid4().hex[:10]))
            previous.rename(directory)
            print('Build did not complete; restored the previous build directory.', flush=True)
        raise


def setup(hardware, args, model, template, profile_path):
    require_idle(hardware['gpu'])
    missing = [name for name in ('bash', 'curl', 'tar', 'patch', 'cmake', 'ninja', 'git', 'sha256sum', 'flock')
               if not shutil.which(name)]
    if missing:
        raise Refused('Install these prerequisites first: ' + ', '.join(missing) + '. No system packages were changed.')
    if memory()['available'] < 4096:
        raise Refused('Free at least 4 GiB of available system RAM before setup.')
    tool_env = toolchain(hardware)
    # External model paths are reused, never copied or replaced by a download.
    if args.model and not model.is_file():
        raise Refused('--model must refer to an existing pinned GGUF.')
    need_bytes = (0 if model.exists() else MODEL_BYTES) + 4 * 1024**3
    if shutil.disk_usage(ROOT).free < need_bytes:
        raise Refused('Insufficient disk headroom for the model and build; nothing was deleted.')
    print('Setup will verify/download the pinned 10.57 GB Q3 model, build locally, then run bounded tuning.\n'
          f'Context stays {args.context}; quant and Q4_0 KV stay fixed. No services or drivers will be changed.\n'
          'Build/download time is additional to the tuning budget. Use --yes for an unattended run.')
    if not args.yes and input('Continue? [y/N] ').strip().lower() not in ('y', 'yes'):
        raise Refused('Setup cancelled without changing the installation.')
    env = {k: v for k, v in os.environ.items() if k not in MANAGED_ENV and not k.startswith(('LLAMA_', 'GGML_'))}
    env.update(tool_env, CUDA_VISIBLE_DEVICES=hardware['gpu']['uuid'])
    command = ['bash', str(ROOT / 'scripts/download-q3.sh')]
    if model.exists():
        command.append('--template-only')
    run_owned(command, env=env)
    verify_assets(model, template)
    build_runtime(hardware, env)
    tune(hardware, args, model, template, profile_path)


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('command', choices=['doctor', 'setup', 'tune', 'start', 'status', 'rollback'])
    parser.add_argument('--context', type=int, choices=[98304, 131072], default=98304)
    parser.add_argument('--gpu', type=int, default=0, help='One NVIDIA GPU index (default: 0)')
    parser.add_argument('--model', type=Path, help='Reuse an existing hash-pinned GGUF')
    parser.add_argument('--budget-seconds', type=int, default=900, help='Wall-time budget for tuning only')
    parser.add_argument('--max-trials', type=int, default=12, help='Includes fit and confirmation trials')
    parser.add_argument('--margin-mib', type=int, default=768, help='Minimum sampled free VRAM; never below 512')
    parser.add_argument('--port', type=int, default=8080, help='Port for start only')
    parser.add_argument('--yes', action='store_true', help='Accept setup download/build/tuning without a prompt')
    parser.add_argument('--dry-run', action='store_true', help='Read-only plan; never downloads, builds or loads a model')
    args = parser.parse_args(argv)
    if not (0 <= args.gpu <= 63 and 300 <= args.budget_seconds <= 7200 and 5 <= args.max_trials <= 32
            and 512 <= args.margin_mib <= 8192 and 1 <= args.port <= 65535):
        parser.error('Invalid GPU, budget (300..7200), trials (5..32), margin (512..8192) or port')
    return args


def dispatch(argv=None):
    args = parse(argv)
    try:
        hardware = detect(args.gpu)
        profile_path = STATE / f'q3-{args.context}.json'
        model = (args.model or ROOT / 'models' / MODEL_NAME).resolve()
        template = ROOT / 'models/qwen-fixed-v22.4.jinja'
        if args.command == 'doctor' or args.dry_run:
            base = initial_config(hardware, args.context, args.margin_mib)
            print(json.dumps({'cpu': {k: v for k, v in hardware['cpu'].items() if k != 'flags'},
                'gpu': {k: v for k, v in hardware['gpu'].items() if k != 'uuid'},
                'memory': memory(), 'compute_processes': compute_processes(hardware['gpu']),
                'context': args.context, 'initial_estimate_not_validated': asdict(base),
                'candidate_screen': [asdict(c) for c in candidates(base, hardware)],
                'tuning_budget_seconds': args.budget_seconds, 'max_trials': args.max_trials,
                'minimum_free_vram_mib': args.margin_mib, 'profile_exists': profile_path.exists(),
                'note': 'Read-only. Tuning refuses existing GPU compute processes; no universal optimum promised.'}, indent=2))
            return 0
        STATE.mkdir(parents=True, exist_ok=True, mode=0o700)
        if args.command == 'status':
            data = read_profile(profile_path, identity(hardware, args.context), args.context)
            print(json.dumps(data, indent=2))
            return 0
        # One operation per physical GPU, even across separate repository clones.
        lock_root = Path(os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir()))
        lock_name = hashlib.sha256(hardware['gpu']['uuid'].encode()).hexdigest()[:16]
        with exclusive(lock_root / f'routeweaver-{os.getuid()}-{lock_name}.lock'):
            if args.command in ('setup', 'tune'):
                action = setup if args.command == 'setup' else tune
                action(hardware, args, model, template, profile_path)
                return 0
            fingerprint = identity(hardware, args.context)
            if args.command == 'rollback':
                previous = profile_path.with_suffix('.previous.json')
                profile = read_profile(previous, fingerprint, args.context)
                save_profile(profile_path, profile)
                print('Restored the previous matching profile. No running server was restarted.')
                return 0
            profile = read_profile(profile_path, fingerprint, args.context)
            model = Path(profile['model_path']) if args.model is None else model
            template = Path(profile['template_path'])
            require_idle(hardware['gpu'])
            config = Config(**profile['config']).validate()
            if hardware['gpu']['free'] < profile['required_free_vram_mib']:
                raise Refused('Less free VRAM than this profile needs. Free competing GPU memory or retune; no settings changed.')
            if memory()['available'] < profile['required_available_ram_mib']:
                raise Refused('Less available RAM than this placement needs. Free memory before starting.')
            env = clean_env(hardware, config, args.context, args.port, model, template)
            process = subprocess.Popen(['bash', str(ROOT / 'scripts/serve-q3.sh')], env=env, start_new_session=True)
            try:
                return process.wait()
            finally:
                terminate_owned(process)
        return 0
    except (Refused, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print('RouteWeaver: ' + str(error), file=__import__('sys').stderr)
        return 1
    except (KeyboardInterrupt, EOFError):
        print('Cancelled. Existing profile and other processes were left alone.', file=__import__('sys').stderr)
        return 130


def main(argv=None):
    def stop(_signum, _frame):
        raise KeyboardInterrupt()
    original = signal.signal(signal.SIGTERM, stop)
    try:
        return dispatch(argv)
    finally:
        signal.signal(signal.SIGTERM, original)
