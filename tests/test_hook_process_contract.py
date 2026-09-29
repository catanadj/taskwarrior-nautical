from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

import coverage

from tests.support.hook_process import ROOT, HookSubprocessFixture


class HookProcessContractTests(HookSubprocessFixture):
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
