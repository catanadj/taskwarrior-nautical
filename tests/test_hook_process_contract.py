from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import coverage

from tests.support.hook_process import ROOT, HookSubprocessFixture


class HookProcessContractTests(HookSubprocessFixture):
    def test_script_runner_forwards_cli_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "argv.py"
            script.write_text(
                "import json, sys\nprint(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            process = self.run_python_command(
                [sys.executable, str(script), "--json", "value"],
                capture_output=True,
                text=True,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), ["--json", "value"])

    def test_python_command_runner_forwards_working_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "cwd.py"
            script.write_text(
                "import os\nprint(os.getcwd())\n",
                encoding="utf-8",
            )
            process = self.run_python_command(
                [sys.executable, str(script)],
                cwd=temporary,
            )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(Path(process.stdout.strip()), Path(temporary))

    def test_python_code_runner_forwards_arguments(self) -> None:
        process = self.run_python_code(
            "import sys\nprint(sys.argv[1:])\n",
            arguments=("contract-value",),
            instrument_coverage=False,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout.strip(), "['contract-value']")

    def test_python_command_runner_supports_module_invocation(self) -> None:
        with tempfile.TemporaryDirectory() as coverage_dir:
            with patch.dict(
                os.environ,
                {
                    "NAUTICAL_SUBPROCESS_COVERAGE_DIR": coverage_dir,
                    "COVERAGE_FILE": str(Path(coverage_dir) / ".coverage"),
                },
            ):
                process = self.run_python_command(
                    [sys.executable, "-m", "json.tool"],
                    input="{}",
                )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), {})
        self.assertEqual(process.stderr, "")

    def test_plain_fast_paths_avoid_loading_core_and_preserve_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hooks = root / "hooks"
            core = root / "nautical_core"
            hooks.mkdir()
            core.mkdir()
            for name in ("on-add.nautical", "on-modify.nautical", "on-exit.nautical"):
                shutil.copy2(ROOT / name, hooks / name)
            for name in (
                "hook_bootstrap.py", "hook_protocol.py", "task_codec.py",
                "task_models.py", "exit_probe.py", "config_support.py",
            ):
                shutil.copy2(ROOT / "nautical_core" / name, core / name)
            (core / "__init__.py").write_text(
                "raise RuntimeError('core must not load on plain fast path')\n",
                encoding="utf-8",
            )

            environment = os.environ.copy()
            environment["TASKDATA"] = str(root)
            for name in ("NAUTICAL_CORE_PATH", "NAUTICAL_TRUST_CORE_PATH", "NAUTICAL_PROFILE", "NAUTICAL_BENCH_FORCE_FULL"):
                environment.pop(name, None)
            plain = {
                "uuid": "00000000-0000-4000-8000-000000000706",
                "status": "pending",
                "description": "Cafe ăîșț ✅",
            }

            cases = (
                ("on-add.nautical", json.dumps(plain, ensure_ascii=False), plain),
                (
                    "on-modify.nautical",
                    json.dumps(plain, ensure_ascii=False) + "\n"
                    + json.dumps(dict(plain, description="Modified ăîșț ✅"), ensure_ascii=False),
                    dict(plain, description="Modified ăîșț ✅"),
                ),
            )
            for hook, payload, expected in cases:
                with self.subTest(hook=hook):
                    process = subprocess.run(
                        [sys.executable, str(hooks / hook)], input=payload, text=True,
                        capture_output=True, env=environment, timeout=10,
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)
                    self.assertEqual(json.loads(process.stdout), expected)
                    self.assertIn("ăîșț ✅", process.stdout)
                    self.assertNotIn("\\u", process.stdout)

            nautical_old = dict(
                plain, cp="P1D", chain="on", chainID="abcd1234", link=3,
                due="20270101T090000Z",
            )
            nautical_new = dict(nautical_old, description="Modified nautical ăîșț ✅")
            ordinary_modify = subprocess.run(
                [sys.executable, str(hooks / "on-modify.nautical")],
                input=json.dumps(nautical_old, ensure_ascii=False) + "\n"
                + json.dumps(nautical_new, ensure_ascii=False),
                text=True, capture_output=True, env=environment, timeout=10,
            )
            self.assertEqual(ordinary_modify.returncode, 0, ordinary_modify.stderr)
            self.assertEqual(json.loads(ordinary_modify.stdout), nautical_new)

            empty_exit = subprocess.run(
                [sys.executable, str(hooks / "on-exit.nautical")],
                text=True, capture_output=True, env=environment, timeout=10,
            )
            self.assertEqual(empty_exit.returncode, 0, empty_exit.stderr)
            self.assertEqual(empty_exit.stdout, "")
            self.assertFalse((root / ".nautical-state").exists())

            forced_environment = dict(environment, NAUTICAL_BENCH_FORCE_FULL="1")
            forced_cases = (
                (hooks / "on-add.nautical", json.dumps(plain, ensure_ascii=False)),
                (hooks / "on-modify.nautical", json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(plain)),
                (hooks / "on-modify.nautical", json.dumps(nautical_old) + "\n" + json.dumps(nautical_new)),
                (hooks / "on-exit.nautical", ""),
            )
            for hook, payload in forced_cases:
                with self.subTest(forced_full=hook.name):
                    process = subprocess.run(
                        [sys.executable, str(hook)], input=payload, text=True,
                        capture_output=True, env=forced_environment, timeout=10,
                    )
                    self.assertNotEqual(process.returncode, 0)

            implementations = core / "hooks"
            implementations.mkdir()
            (implementations / "add_impl.py").write_text(
                "HOOK_IMPL_API = 999\ndef run_hook(**_kwargs):\n"
                "    raise AssertionError('mismatched implementation must not run')\n",
                encoding="utf-8",
            )
            nautical_task = dict(plain, cp="P1D")
            mismatch = subprocess.run(
                [sys.executable, str(hooks / "on-add.nautical")],
                input=json.dumps(nautical_task, ensure_ascii=False), text=True,
                capture_output=True, env=environment, timeout=10,
            )
            self.assertNotEqual(mismatch.returncode, 0)
            self.assertEqual(json.loads(mismatch.stdout), nautical_task)
            self.assertEqual(mismatch.stderr, "")
            diagnostic_environment = dict(environment, NAUTICAL_DIAG="1")
            mismatch_diagnostic = subprocess.run(
                [sys.executable, str(hooks / "on-add.nautical")],
                input=json.dumps(nautical_task, ensure_ascii=False), text=True,
                capture_output=True, env=diagnostic_environment, timeout=10,
            )
            self.assertIn("API mismatch", mismatch_diagnostic.stderr)

            (implementations / "modify_impl.py").write_text(
                "HOOK_IMPL_API = 999\ndef run_hook(**_kwargs):\n"
                "    raise AssertionError('mismatched implementation must not run')\n",
                encoding="utf-8",
            )
            modify_input = json.dumps(nautical_task, ensure_ascii=False) + "\n" + json.dumps(
                dict(nautical_task, cp="P2D"), ensure_ascii=False,
            )
            modify_mismatch = subprocess.run(
                [sys.executable, str(hooks / "on-modify.nautical")],
                input=modify_input, text=True, capture_output=True,
                env=environment, timeout=10,
            )
            self.assertNotEqual(modify_mismatch.returncode, 0)
            self.assertEqual(json.loads(modify_mismatch.stdout)["cp"], "P2D")
            self.assertEqual(modify_mismatch.stderr, "")
            modify_diagnostic = subprocess.run(
                [sys.executable, str(hooks / "on-modify.nautical")],
                input=modify_input, text=True, capture_output=True,
                env=diagnostic_environment, timeout=10,
            )
            self.assertIn("API mismatch", modify_diagnostic.stderr)

            (implementations / "exit_impl.py").write_text(
                "HOOK_IMPL_API = 999\ndef run_hook(**_kwargs):\n"
                "    raise AssertionError('mismatched implementation must not run')\n",
                encoding="utf-8",
            )
            state = root / ".nautical-state"
            state.mkdir(exist_ok=True)
            with sqlite3.connect(str(state / ".nautical_lifecycle_outbox.db")) as connection:
                connection.execute(
                    "CREATE TABLE lifecycle_outbox (intent_id TEXT PRIMARY KEY, processing_state TEXT NOT NULL)"
                )
                connection.execute("INSERT INTO lifecycle_outbox VALUES ('intent-1', 'ready')")
            exit_mismatch = subprocess.run(
                [sys.executable, str(hooks / "on-exit.nautical")],
                text=True, capture_output=True, env=environment, timeout=10,
            )
            self.assertNotEqual(exit_mismatch.returncode, 0)
            self.assertEqual(exit_mismatch.stdout, "")
            self.assertEqual(exit_mismatch.stderr, "")
            exit_diagnostic = subprocess.run(
                [sys.executable, str(hooks / "on-exit.nautical")],
                text=True, capture_output=True, env=diagnostic_environment, timeout=10,
            )
            self.assertIn("API mismatch", exit_diagnostic.stderr)

    def test_full_hook_run_reuses_wrapper_protocol_probe(self) -> None:
        source = textwrap.dedent(f"""
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch
            from nautical_core.hooks import add_impl, modify_impl

            root = Path({str(ROOT)!r})
            taskdata = Path({self.taskdata!r})
            for module, probe_name in ((add_impl, 'probe_on_add'), (modify_impl, 'probe_on_modify')):
                calls = {{'main': 0, 'probe': 0}}
                probe = object()
                def unexpected_probe(*_args, **_kwargs):
                    calls['probe'] += 1
                    raise AssertionError('full implementation reparsed wrapper input')
                protocol = SimpleNamespace(**{{probe_name: unexpected_probe}})
                early = module._EARLY_PROTOCOL_RESULT
                previous_protocol = module._PROTOCOL
                with patch.object(module, 'main', side_effect=lambda: calls.__setitem__('main', calls['main'] + 1)), \\
                     patch.object(module, '_initialize_integration_context'):
                    result = module.run_hook(
                        raw_input=b'{{"uuid":"probe-reuse"}}', argv=(),
                        hook_dir=str(taskdata / 'hooks'), core_base=str(root / 'nautical_core'),
                        protocol=protocol, probe=probe, protocol_error=None,
                    )
                assert result == 0
                assert calls == {{'main': 1, 'probe': 0}}, calls
                assert module._PROTOCOL is protocol
                assert module._EARLY_PROTOCOL_RESULT is probe
                module._EARLY_PROTOCOL_RESULT = early
                module._PROTOCOL = previous_protocol
            print('ok')
        """)
        process = subprocess.run(
            [sys.executable, "-c", source], cwd=ROOT, text=True,
            capture_output=True, timeout=15,
        )

        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(process.stdout.strip(), "ok")

    def test_hook_bootstrap_uses_symlink_path_and_core_path_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            hooks = root / "hooks"
            taskdata = root / "taskdata"
            staging.mkdir()
            hooks.mkdir()
            taskdata.mkdir()
            for name in ("on-add.nautical", "on-modify.nautical", "on-exit.nautical"):
                staged = staging / name
                shutil.copy2(ROOT / name, staged)
                (hooks / name).symlink_to(staged)

            environment = os.environ.copy()
            environment.update({
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TASKDATA": str(taskdata),
                "TZ": "UTC",
            })
            environment.pop("NAUTICAL_DIAG", None)
            cases = (
                ("on-add.nautical", json.dumps({"uuid": "u", "status": "pending"}), True),
                (
                    "on-modify.nautical",
                    json.dumps({"uuid": "u", "status": "pending"}) + "\n"
                    + json.dumps({"uuid": "u", "status": "pending", "description": "changed"}),
                    True,
                ),
                ("on-exit.nautical", "", False),
            )
            for name, payload, expect_json in cases:
                with self.subTest(hook=name):
                    process = subprocess.run(
                        [sys.executable, str(hooks / name)], input=payload, text=True,
                        capture_output=True, env=environment, timeout=10,
                    )
                    self.assertEqual(process.returncode, 0, process.stderr)
                    if expect_json:
                        json.loads(process.stdout)
                    else:
                        self.assertEqual(process.stdout, "")

    def test_hooks_survive_malformed_numeric_environment(self) -> None:
        malformed_names = (
            "NAUTICAL_PROFILE", "NAUTICAL_OUTBOX_DRAIN_MAX_ITEMS", "NAUTICAL_OUTBOX_DIAG_MAX_ITEMS",
            "NAUTICAL_OUTBOX_RETRY_MAX", "NAUTICAL_TASK_TIMEOUT_EXPORT", "NAUTICAL_TASK_TIMEOUT_IMPORT",
            "NAUTICAL_TASK_TIMEOUT_MODIFY", "NAUTICAL_TASK_RETRIES_EXPORT", "NAUTICAL_TASK_RETRIES_MODIFY",
            "NAUTICAL_TASK_RETRY_DELAY", "NAUTICAL_PARENT_LOCK_RETRIES", "NAUTICAL_PARENT_LOCK_SLEEP_BASE",
            "NAUTICAL_PARENT_LOCK_STALE_AFTER", "NAUTICAL_LOCK_STORM_THRESHOLD", "NAUTICAL_LOCK_BACKOFF_BASE",
            "NAUTICAL_LOCK_BACKOFF_MAX", "NAUTICAL_OUTBOX_LEASE_SECONDS", "NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE",
            "NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100", "NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX",
            "NAUTICAL_DIAG_LOG_MAX_BYTES",
        )
        environment = {name: "not-a-number" for name in malformed_names}
        environment["NAUTICAL_BENCH_FORCE_FULL"] = "1"
        task = {"uuid": "00000000-0000-4000-8000-000000000706", "status": "pending", "description": "Malformed env ăîșț"}
        for hook, payload, expected in (
            ("on-add.nautical", json.dumps(task, ensure_ascii=False), task),
            (
                "on-modify.nautical",
                json.dumps(task, ensure_ascii=False) + "\n"
                + json.dumps(dict(task, description="Modified malformed env ăîșț"), ensure_ascii=False),
                dict(task, description="Modified malformed env ăîșț"),
            ),
            ("on-exit.nautical", "", None),
        ):
            with self.subTest(hook=hook):
                process = self.run_hook(hook, payload, extra_environment=environment)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(process.stderr, "")
                if expected is None:
                    self.assertEqual(process.stdout, "")
                else:
                    self.assertEqual(json.loads(process.stdout), expected)

        import nautical_core as nautical

        with patch.dict(
            os.environ,
            {"NAUTICAL_DIAG_LOG": "1", "NAUTICAL_DIAG_LOG_MAX_BYTES": "not-a-number"},
        ):
            nautical.diag_log("malformed env diagnostic", "golden", self.taskdata)
        record = json.loads(
            (Path(self.taskdata) / ".nautical_diag.jsonl").read_text(encoding="utf-8").splitlines()[-1]
        )
        self.assertEqual(record["msg"], "malformed env diagnostic")

    def test_full_hook_modules_defer_core_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "nautical_core"
            hooks = package / "hooks"
            hooks.mkdir(parents=True)
            for name in ("hook_bootstrap.py", "config_support.py"):
                shutil.copy2(ROOT / "nautical_core" / name, package / name)
            (package / "__init__.py").write_text(
                "raise AssertionError('full core must not load while importing hook implementations')\n",
                encoding="utf-8",
            )
            for name in ("add_impl.py", "modify_impl.py", "exit_impl.py"):
                shutil.copy2(ROOT / "nautical_core" / "hooks" / name, hooks / name)
            probe = root / "probe.py"
            probe.write_text(textwrap.dedent("""
                import importlib.util
                import sys
                from pathlib import Path
                root = Path(__file__).parent
                for index, name in enumerate(('add_impl.py', 'modify_impl.py', 'exit_impl.py')):
                    spec = importlib.util.spec_from_file_location(f'probe_hook_{index}', root / 'nautical_core' / 'hooks' / name)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    assert module.core is None, f'{name} imported core eagerly'
                    if name == 'modify_impl.py':
                        for attr in ('_MODIFY_ORDINARY', '_MODIFY_EXPIRATION', '_MODIFY_GENERATION_COMPAT', '_CHAIN_GENERATION', '_QUEUE_STORE'):
                            assert getattr(module, attr, None) is None, f'{attr} loaded during modify import'
                assert 'nautical_core' not in sys.modules
                print('ok')
            """), encoding="utf-8")
            process = subprocess.run(
                [sys.executable, str(probe)], text=True, capture_output=True, timeout=10,
            )

        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertEqual(process.stdout.strip(), "ok")

    def test_on_modify_manual_delete_persists_chain_off(self) -> None:
        old = {
            "uuid": "00000000-0000-4000-8000-000000000422",
            "status": "pending",
            "description": "Manual delete protocol",
            "cp": "7d",
            "chain": "on",
            "chainID": "delete22",
            "link": 1,
            "due": "20260720T090000Z",
            "until": "20260726T235900Z",
        }
        new = dict(old, status="deleted", end="20260725T000000Z")

        process = self.run_hook(
            "on-modify.nautical",
            json.dumps(old) + "\n" + json.dumps(new),
            extra_environment={"NO_COLOR": "1"},
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertEqual(output["status"], "deleted")
        self.assertEqual(output["chain"], "off")

    def test_on_modify_expiration_wrapper_preserves_json_stdout(self) -> None:
        old = {
            "uuid": "00000000-0000-4000-8000-000000000421",
            "status": "pending",
            "description": "Expiration protocol",
            "cp": "7d",
            "chain": "on",
            "chainID": "expire21",
            "link": 1,
            "due": "20260720T090000Z",
            "until": "20260726T235900Z",
        }
        new = dict(old, status="deleted", end="20260727T000000Z")

        process = self.run_hook(
            "on-modify.nautical",
            json.dumps(old) + "\n" + json.dumps(new),
            extra_environment={"NO_COLOR": "1"},
        )

        self.assertEqual(process.returncode, 0, process.stderr)
        output = json.loads(process.stdout)
        self.assertEqual(output["status"], "deleted")
        self.assertEqual(output["chain"], "on")
        self.assertNotIn("Nautical occurrence expired", process.stderr)

    def test_shared_coverage_opt_in_records_hook_execution(self) -> None:
        coverage_directory = Path(
            os.environ.get("NAUTICAL_SUBPROCESS_COVERAGE_DIR")
            or Path(self.taskdata) / "shared-coverage"
        )
        payload = json.dumps(
            {
                "uuid": "00000000-0000-4000-8000-000000000802",
                "description": "shared coverage opt in",
                "status": "pending",
            }
        )
        with patch.dict(
            os.environ,
            {"NAUTICAL_SUBPROCESS_COVERAGE_DIR": str(coverage_directory)},
        ):
            coverage_name = Path(os.environ.get("COVERAGE_FILE", ".coverage")).name
            process = self.run_hook("on-add.nautical", payload)

        self.assertEqual(process.returncode, 0, process.stderr)
        shared_data = list(coverage_directory.glob(f"{coverage_name}.*"))
        self.assertGreaterEqual(len(shared_data), 1, shared_data)
        measured_files: set[str] = set()
        for data_file in shared_data:
            measured_data = coverage.Coverage(data_file=str(data_file))
            measured_data.load()
            measured_files.update(
                Path(path).name for path in measured_data.get_data().measured_files()
            )
        self.assertIn("hook_protocol.py", measured_files)

    def test_coverage_opt_in_records_lines_executed_by_on_add(self) -> None:
        coverage_file = Path(self.taskdata) / "coverage" / ".coverage"
        payload = json.dumps(
            {
                "uuid": "00000000-0000-4000-8000-000000000801",
                "description": "covered café",
                "status": "pending",
            },
            ensure_ascii=False,
        )

        process = self.run_hook("on-add.nautical", payload, coverage_file=coverage_file)

        self.assertEqual(process.returncode, 0, process.stderr)
        self.combine_coverage(coverage_file)
        measured_data = coverage.Coverage(data_file=str(coverage_file))
        measured_data.load()
        data = measured_data.get_data()
        measured = {Path(path).name for path in data.measured_files()}
        self.assertIn("hook_protocol.py", measured)

    def test_child_failure_is_returned_when_coverage_is_enabled(self) -> None:
        script = self._script("raise SystemExit(7)")
        coverage_file = Path(self.taskdata) / "coverage" / ".coverage"

        process = self.run_script(script, coverage_file=coverage_file)

        self.assertEqual(process.returncode, 7)

    def test_timeout_is_not_hidden_by_coverage(self) -> None:
        script = self._script("import time; time.sleep(1)")
        coverage_file = Path(self.taskdata) / "coverage" / ".coverage"

        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_script(script, timeout=0.05, coverage_file=coverage_file)

    def test_malformed_child_output_remains_visible(self) -> None:
        script = self._script("print('{not-json')")
        coverage_file = Path(self.taskdata) / "coverage" / ".coverage"

        process = self.run_script(script, coverage_file=coverage_file)

        self.assertEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "{not-json\n")
        with self.assertRaises(json.JSONDecodeError):
            json.loads(process.stdout)

    def _script(self, body: str) -> Path:
        path = Path(self.taskdata) / "child.py"
        path.write_text(textwrap.dedent(f"""
            {body}
        """), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
