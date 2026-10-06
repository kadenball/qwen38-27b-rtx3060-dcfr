#!/usr/bin/env python3
"""Policy, safety, profile and real-subprocess HTTP integration tests; no GPU needed."""
from contextlib import ExitStack, redirect_stdout
import io
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import routeweaver_cli as rw

HARDWARE = {'cpu': {'name': 'Test AVX2 CPU', 'physical_cores': 6, 'logical_cpus': 12,
                    'flags': ['avx2', 'fma']},
            'gpu': {'index': 0, 'name': 'Test NVIDIA GPU', 'uuid': 'GPU-test', 'total': 12288,
                    'free': 11800, 'driver': 'test', 'capability': '8.6'}, 'ram_total_mib': 32768}
RAM = {'available': 20000, 'total': 32768, 'swap_out': 0}


def result(config, speed=100, seeds=(101,)):
    return {'config': rw.asdict(config), 'rows': [{'fixture': fixture, 'seed': seed,
             'timings': {'predicted_per_second': speed, 'prompt_per_second': speed * 10}}
            for seed in seeds for fixture in ('rust', 'cpp', 'reasoning', 'prefill-coding')],
            'starting_free_vram_mib': 11800, 'minimum_free_vram_mib': 1800,
            'minimum_available_ram_mib': 4000}


def profile(config=rw.Config(41, 6)):
    return {'schema': rw.SCHEMA, 'status': 'validated', 'fingerprint': 'match', 'context': 98304,
            'model_sha256': rw.MODEL_HASH, 'config': rw.asdict(config), 'margin_mib': 768,
            'required_free_vram_mib': 10768, 'required_available_ram_mib': 6000,
            'model_path': '/example/model.gguf', 'template_path': '/example/template.jinja'}


class PolicyTests(unittest.TestCase):
    def test_cpu_topology_respects_affinity_and_shared_instruction_support(self):
        text = '\n\n'.join('processor : %s\nphysical id : 0\ncore id : %s\nmodel name : Test CPU\nflags : avx2 fma sse2' % (i, i % 2) for i in range(4))
        with patch.object(rw.platform, 'system', return_value='Linux'), patch.object(rw.platform, 'machine', return_value='x86_64'), \
             patch.object(rw.os, 'sched_getaffinity', return_value={0, 1}), patch.object(Path, 'read_text', return_value=text):
            cpu = rw.cpu_info()
            self.assertEqual(cpu['physical_cores'], 2)
            self.assertEqual(cpu['logical_cpus'], 2)
        with patch.object(rw.platform, 'system', return_value='Linux'), patch.object(rw.platform, 'machine', return_value='aarch64'):
            with self.assertRaisesRegex(rw.Refused, 'x86-64'):
                rw.cpu_info()

    def test_cpu_without_fma_is_not_given_avx2_kernel(self):
        with patch.object(rw.platform, 'system', return_value='Linux'), patch.object(rw.platform, 'machine', return_value='x86_64'), \
             patch.object(rw.os, 'sched_getaffinity', return_value={0}), \
             patch.object(Path, 'read_text', return_value='processor : 0\nflags : avx2'):
            with self.assertRaisesRegex(rw.Refused, 'AVX2 and FMA'):
                rw.cpu_info()

    def test_build_restores_old_directory_after_failure(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(rw, 'ROOT', Path(temp)), \
             patch.object(rw, 'output', return_value='test-version'), patch.object(rw.shutil, 'which', side_effect=lambda s:s), \
             redirect_stdout(io.StringIO()):
            directory = Path(temp)/'build/q3-candidate'
            directory.mkdir(parents=True)
            (directory/'old-build').write_text('retain')
            def fail(*_args, **_kwargs):
                directory.mkdir()
                (directory/'partial-build').write_text('incomplete')
                raise rw.Refused('compiler failure')
            with patch.object(rw, 'run_owned', side_effect=fail):
                with self.assertRaisesRegex(rw.Refused, 'compiler failure'):
                    rw.build_runtime(HARDWARE, dict(CC='cc', CXX='c++', CUDA_ARCHITECTURES='86', CUDA_TOOLKIT_ROOT='/toolkit'))
            self.assertEqual((directory/'old-build').read_text(), 'retain')
            self.assertEqual(len(list(directory.parent.glob('q3-candidate.incomplete-*'))), 1)

    def test_new_hardware_build_is_fresh_but_matching_build_is_reused(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(rw, 'ROOT', Path(temp)), \
             patch.object(rw, 'output', return_value='test-version'), patch.object(rw.shutil, 'which', side_effect=lambda s:s), \
             redirect_stdout(io.StringIO()):
            directory = Path(temp)/'build/q3-candidate'
            directory.mkdir(parents=True)
            (directory/'old-build').write_text('retain')
            def build(*_args, **_kwargs): directory.mkdir(exist_ok=True)
            env = dict(CC='cc', CXX='c++', CUDA_ARCHITECTURES='86', CUDA_TOOLKIT_ROOT='/toolkit')
            with patch.object(rw, 'run_owned', side_effect=build):
                rw.build_runtime(HARDWARE, env)
                self.assertFalse((directory/'old-build').exists())
                (directory/'new-object').write_text('reusable')
                rw.build_runtime(HARDWARE, env)
                self.assertTrue((directory/'new-object').exists())
                self.assertEqual(len(list(directory.parent.glob('q3-candidate.saved-*'))), 1)

    def test_candidates_preserve_model_context_and_change_one_knob(self):
        base = rw.initial_config(HARDWARE, 98304, 768)
        self.assertEqual(base, rw.Config(41, 6))
        choices = list(rw.candidates(base, HARDWARE))
        self.assertEqual(choices[0].speculation, 'none')
        self.assertTrue(any(c.depth == 2 for c in choices))
        self.assertTrue(any(c.batch == 256 for c in choices))
        self.assertTrue(any(c.threads == 12 for c in choices))
        for candidate in choices:
            candidate.validate()
        larger = dict(HARDWARE, gpu=dict(HARDWARE['gpu'], total=24576))
        self.assertEqual(rw.initial_config(larger, 131072, 768).host_blocks, 0)

    def test_profile_bounds_reject_boolean_or_injected_settings(self):
        for config in [rw.Config(True, 6), rw.Config(65, 6), rw.Config(40, 0),
                       rw.Config(40, 6, speculation='none; false'), rw.Config(40, 6, batch=64, ubatch=128)]:
            with self.assertRaises(rw.Refused):
                config.validate()

    def test_env_drops_inherited_experiments_and_vision(self):
        with patch.dict(os.environ, {'GGML_CPU_IQ3XXS_MULTI': '2', 'LLAMA_ARG_CTX_SIZE': '1',
                                     'CONTEXT': '1', 'MMPROJ_PATH': 'wrong', 'LD_PRELOAD': 'wrong',
                                     'MAX_OUTPUT_TOKENS': '1'}):
            env = rw.clean_env(HARDWARE, rw.Config(41, 6), 98304, 8094, Path('/model'), Path('/template'))
        self.assertEqual(env['CONTEXT'], '98304')
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-test')
        for key in ('GGML_CPU_IQ3XXS_MULTI', 'LLAMA_ARG_CTX_SIZE', 'MMPROJ_PATH', 'LD_PRELOAD', 'MAX_OUTPUT_TOKENS'):
            self.assertNotIn(key, env)

    def test_no_speculation_launcher_really_disables_drafting(self):
        env = rw.clean_env(HARDWARE, rw.Config(41, 6, speculation='none'), 98304, 8094, Path('/model'), Path('/template'))
        command = rw.shlex.split(subprocess.check_output(['bash', str(ROOT/'scripts/serve-q3.sh'), '--print-config'], env=env, text=True))
        self.assertEqual(command.count('--spec-type'), 1)
        self.assertEqual(command[command.index('--spec-type')+1], 'none')
        self.assertNotIn('--spec-draft-n-max', command)
        self.assertEqual(command[command.index('--ctx-size')+1], '98304')

    def test_fast_easy_prompt_cannot_hide_hard_prompt_regression(self):
        base = result(rw.Config(41, 6))
        candidate = result(rw.Config(41, 3), 150)
        candidate['rows'][2]['timings']['predicted_per_second'] = 80
        self.assertGreater(rw.score(candidate), rw.score(base))
        self.assertFalse(rw.acceptable(candidate, base))
        candidate['rows'][2]['timings']['predicted_per_second'] = 95
        self.assertTrue(rw.acceptable(candidate, base))

    def test_budget_expiration_is_failure(self):
        budget = rw.Budget(10)
        budget.end = 0
        with self.assertRaises(rw.Refused):
            budget.timeout(30)

    def test_atomic_profile_previous_and_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'profile.json'
            first = profile()
            second = profile(rw.Config(37, 6))
            rw.save_profile(path, first)
            rw.save_profile(path, second)
            self.assertEqual(rw.read_profile(path, 'match', 98304), second)
            self.assertEqual(json.loads(path.with_suffix('.previous.json').read_text()), first)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            for fingerprint, context in [('changed', 98304), ('match', 131072)]:
                with self.assertRaises(rw.Refused):
                    rw.read_profile(path, fingerprint, context)
            corrupted = dict(second, required_free_vram_mib=-1)
            rw.atomic_json(path, corrupted)
            with self.assertRaises(rw.Refused):
                rw.read_profile(path, 'match', 98304)

    def test_exclusive_lock_prevents_second_operation(self):
        with tempfile.TemporaryDirectory() as temp:
            with rw.exclusive(Path(temp)/'lock'):
                with self.assertRaises(rw.Refused):
                    with rw.exclusive(Path(temp)/'lock'):
                        self.fail('Second lock admitted')

    def test_busy_gpu_never_killed(self):
        with patch.object(rw, 'compute_processes', return_value=[{'pid': 42, 'name': 'llama-server'}]), \
             patch.object(rw, 'terminate_owned') as terminate:
            with self.assertRaisesRegex(rw.Refused, 'already running'):
                rw.require_idle(HARDWARE['gpu'])
            terminate.assert_not_called()

    def test_guard_rejects_low_vram_low_ram_paging_and_foreign_processes(self):
        for kind in ('vram', 'ram', 'paging', 'foreign', 'budget'):
            with self.subTest(kind=kind), patch.object(rw, 'memory', return_value=RAM) as mem, \
                 patch.object(rw, 'gpu_info', return_value=dict(HARDWARE['gpu'], free=400 if kind=='vram' else 1800)), \
                 patch.object(rw, 'compute_processes', return_value=[{'pid': 101, 'name': 'other'}] if kind=='foreign' else []), \
                 patch.object(rw, 'terminate_owned') as terminate:
                process = Mock(pid=100)
                budget = rw.Budget(10)
                if kind == 'budget': budget.end = 0
                guard = rw.Guard(HARDWARE, 768, process, budget)
                if kind == 'ram': mem.return_value = dict(RAM, available=1000)
                if kind == 'paging': mem.return_value = dict(RAM, swap_out=100)
                guard.watch()
                self.assertIsNotNone(guard.failure)
                terminate.assert_called_once_with(process)

    def test_dry_run_has_no_download_build_or_write(self):
        with patch.object(rw, 'detect', return_value=HARDWARE), patch.object(rw, 'memory', return_value=RAM), \
             patch.object(rw, 'compute_processes', return_value=[]), patch.object(rw, 'setup') as setup, \
             patch.object(rw, 'atomic_json') as write, redirect_stdout(io.StringIO()):
            self.assertEqual(rw.main(['setup', '--dry-run']), 0)
            setup.assert_not_called()
            write.assert_not_called()


class FakeTrial:
    noise = False
    fail_chat = False
    fail_plain = False
    interrupt = False

    def __init__(self, hardware, context, config, model, template, margin, budget, directory):
        self.config = config

    def __enter__(self):
        if self.interrupt: raise KeyboardInterrupt()
        if self.fail_plain and self.config.speculation == 'none': raise rw.Refused('No plain result')
        return self

    def __exit__(self, *_): pass

    def measure(self, seeds):
        faster = self.config.threads == 3 and not (self.noise and seeds != (101,))
        return result(self.config, 120 if faster else 100, seeds)

    def memory_result(self):
        return {k:v for k,v in result(self.config).items() if k.endswith('_mib')}

    def validate_chat(self):
        if self.fail_chat: raise rw.Refused('Synthetic chat check failure')


class TuningTests(unittest.TestCase):
    def exercise(self, trial_class=FakeTrial, existing=None):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        for name, value in [('STATE', directory), ('require_idle', None), ('verify_assets', None),
                             ('identity', 'match'), ('model_free_checks', None)]:
            stack.enter_context(patch.object(rw, name, value) if name == 'STATE' else patch.object(rw, name, return_value=value))
        stack.enter_context(redirect_stdout(io.StringIO()))
        path = directory/'profile.json'
        if existing: rw.atomic_json(path, existing)
        args = rw.parse(['tune'])
        return lambda: rw.tune(HARDWARE, args, Path('/model'), Path('/template'), path, trial_class), path

    def test_selects_repeated_gain_and_saves_fallback(self):
        action, path = self.exercise(existing=profile())
        action()
        data = rw.read_profile(path, 'match', 98304)
        self.assertEqual(data['config']['threads'], 3)
        self.assertEqual(data['context'], 98304)
        self.assertEqual(data['required_free_vram_mib'], 10768)
        self.assertEqual(json.loads(path.with_suffix('.previous.json').read_text()), profile())

    def test_noise_uses_validated_reference(self):
        class Noisy(FakeTrial): noise = True
        action, path = self.exercise(Noisy)
        action()
        self.assertEqual(json.loads(path.read_text())['config']['threads'], 6)

    def test_chat_failure_retains_previous_profile(self):
        class Failure(FakeTrial): fail_chat = True
        action, path = self.exercise(Failure, profile())
        with self.assertRaises(rw.Refused): action()
        self.assertEqual(json.loads(path.read_text()), profile())

    def test_plain_decode_reference_is_required(self):
        class Failure(FakeTrial): fail_plain = True
        action, path = self.exercise(Failure, profile())
        with self.assertRaisesRegex(rw.Refused, 'no-speculation'): action()
        self.assertEqual(json.loads(path.read_text()), profile())

    def test_interruption_retains_profile(self):
        class Interrupted(FakeTrial): interrupt = True
        action, path = self.exercise(Interrupted, profile())
        with self.assertRaises(KeyboardInterrupt): action()
        self.assertEqual(json.loads(path.read_text()), profile())
        reports = list(path.parent.glob('tune-*/report.json'))
        self.assertEqual(json.loads(reports[0].read_text())['status'], 'cancelled')


class ProcessIntegrationTests(unittest.TestCase):
    def test_safety_stop_reason_survives_closed_http_connection(self):
        with tempfile.TemporaryDirectory() as temp:
            trial = rw.Trial(HARDWARE, 98304, rw.Config(41, 6), Path('/model'), Path('/template'), 768,
                             rw.Budget(10), Path(temp)/'trial')
            trial.guard = Mock()
            trial.guard.check.side_effect = [None, rw.Refused('GPU headroom below the requested margin.')]
            trial.http = Mock()
            trial.http.open.side_effect = http.client.RemoteDisconnected('Remote end closed connection without response')
            with self.assertRaisesRegex(rw.Refused, 'GPU headroom'):
                trial.request('/v1/chat/completions', {})

    def environment(self, wrong_alias=False, wrong_context=False, leaked_draft=False):
        stack = ExitStack()
        self.addCleanup(stack.close)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        stack.enter_context(patch.object(rw, 'memory', return_value=RAM))
        stack.enter_context(patch.object(rw, 'gpu_info', return_value=HARDWARE['gpu']))
        stack.enter_context(patch.object(rw, 'compute_processes', return_value=[]))
        real_popen = subprocess.Popen

        def popen(command, **kwargs):
            if command[:2] == ['bash', '-c']:
                get = lambda flag: command[command.index(flag) + 1]
                command = [sys.executable, str(ROOT/'tests/fake-q3-server.py'), '--port', get('--port'),
                    '--alias', 'wrong-server' if wrong_alias else get('--alias'),
                    '--context', '4096' if wrong_context else get('--ctx-size'),
                    '--draft', '10' if leaked_draft or get('--spec-type') != 'none' else '0']
            return real_popen(command, **kwargs)

        stack.enter_context(patch.object(rw.subprocess, 'Popen', side_effect=popen))
        return directory

    def test_actual_http_lifecycle_chat_suite_and_owned_cleanup(self):
        directory = self.environment()
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        try:
            trial = rw.Trial(HARDWARE, 98304, rw.Config(41, 6), Path('/model'), Path('/template'), 768,
                             rw.Budget(30), directory/'trial')
            with trial:
                report = trial.measure((101,))
                self.assertEqual(len(report['rows']), 4)
                self.assertTrue(all(r['timings']['cache_n'] == 0 for r in report['rows']))
                trial.validate_chat()
                self.assertIsNone(trial.process.poll())
            self.assertIsNotNone(trial.process.poll())
            self.assertTrue(trial.log.closed)
            self.assertIsNone(unrelated.poll())
        finally:
            rw.terminate_owned(unrelated)

    def test_wrong_server_is_never_benchmarked(self):
        directory = self.environment(wrong_alias=True)
        trial = rw.Trial(HARDWARE, 98304, rw.Config(41, 6), Path('/model'), Path('/template'), 768,
                         rw.Budget(10), directory/'trial')
        with self.assertRaisesRegex(rw.Refused, 'identity mismatch'):
            with trial: self.fail('Wrong server was admitted')
        self.assertIsNotNone(trial.process.poll())

    def test_context_cannot_shrink(self):
        directory = self.environment(wrong_context=True)
        trial = rw.Trial(HARDWARE, 98304, rw.Config(41, 6), Path('/model'), Path('/template'), 768,
                         rw.Budget(10), directory/'trial')
        with self.assertRaisesRegex(rw.Refused, 'changed the requested context'):
            with trial: self.fail('Context reduction was admitted')

    def test_plain_decode_cannot_hide_drafting(self):
        directory = self.environment(leaked_draft=True)
        trial = rw.Trial(HARDWARE, 98304, rw.Config(41, 6, speculation='none'), Path('/model'), Path('/template'), 768,
                         rw.Budget(10), directory/'trial')
        with trial, self.assertRaisesRegex(rw.Refused, 'unexpectedly drafted'):
            trial.measure((101,))

    def test_timeout_cleans_up_owned_command(self):
        real_popen = subprocess.Popen
        processes = []
        def popen(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            processes.append(process)
            return process
        with patch.object(rw.subprocess, 'Popen', side_effect=popen):
            with self.assertRaises(subprocess.TimeoutExpired):
                rw.run_owned([sys.executable, '-c', 'import time; time.sleep(30)'], env=os.environ.copy(), timeout=.1)
        self.assertIsNotNone(processes[0].poll())


if __name__ == '__main__':
    unittest.main(verbosity=2)
