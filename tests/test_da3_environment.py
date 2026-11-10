"""CUDA failure diagnostics and actual shell stop-on-failure behavior."""
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from scripts.check_da3_environment import require_cuda
from scripts.run_da3_occlusion import build_plan

ROOT = Path(__file__).resolve().parents[1]


class DA3EnvironmentTests(unittest.TestCase):
    def test_cuda_error_reports_installed_runtime_and_repair(self):
        torch = SimpleNamespace(__version__='new-version', version=SimpleNamespace(cuda='13.0'),
                                cuda=SimpleNamespace(is_available=lambda: False))
        with self.assertRaisesRegex(RuntimeError, 'wheel CUDA=13.0.*requirements-da3.txt'):
            require_cuda(torch, '3')

    def test_missing_cache_points_to_preprocessing_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(baseline=None, keyframes=None, depth_dir=Path(tmp))
            with self.assertRaisesRegex(RuntimeError, 'DA3 preprocessing has not completed.*Fix the preprocessing error'):
                build_plan(args)

    def run_launcher(self, fail_at='', extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'conda.sh').write_text('conda() { export STUB_ENV="$2"; }\n')
            binary = root/'python'
            binary.write_text('''#!/usr/bin/env bash
printf '%s|%s\\n' "$STUB_ENV" "$*" >> "$TRACE_PATH"
if [[ -n "${FAIL_AT:-}" && "$*" == *"$FAIL_AT"* ]]; then exit 23; fi
''')
            binary.chmod(0o755)
            environment = dict(os.environ, PATH=str(root)+os.pathsep+os.environ['PATH'],
                TRACE_PATH=str(root/'trace'), FAIL_AT=fail_at, DA3_CONDA_SH=str(root/'conda.sh'),
                DA3_BASELINE=str(root/'baseline'), DA3_OUTPUT_ROOT=str(root/'scene outputs'))
            result = subprocess.run(['bash', str(ROOT/'scripts/run_da3_ablation.sh'), '3', *extra],
                                    env=environment, capture_output=True, text=True)
            trace = (root/'trace').read_text()
        return result, trace

    def test_failed_preparation_never_starts_training(self):
        result, trace = self.run_launcher('prepare_da3_occlusion.py')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertIn('da3-depth|-u scripts/prepare_da3_occlusion.py', trace)
        self.assertNotIn('run_da3_occlusion.py', trace)

    def test_failed_cuda_check_never_starts_preparation(self):
        result, trace = self.run_launcher('check_da3_environment.py')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertNotIn('prepare_da3_occlusion.py', trace)

    def test_failed_install_never_checks_cuda(self):
        result, trace = self.run_launcher('-m pip install')
        self.assertEqual(result.returncode, 23, result.stderr)
        self.assertNotIn('check_da3_environment.py', trace)

    def test_success_switches_environments_and_forwards_options(self):
        result, trace = self.run_launcher(extra=('--evaluate-only',))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('da3-depth|-m pip install -r requirements-da3.txt', trace)
        self.assertIn('da3-depth|scripts/check_da3_environment.py --gpu 3', trace)
        self.assertIn('3dgs|-u scripts/run_da3_occlusion.py', trace)
        self.assertTrue(trace.rstrip().endswith('--gpu 3 --evaluate-only'))


if __name__ == '__main__':
    unittest.main()
