from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import shutil

from nautical_core.integration_models import CommandFailureKind
from nautical_core.doctor_report import DoctorReport
from nautical_core.installation_report import InstallationVerificationReport
from nautical_core.operator_models import OperatorV2Result
from nautical_core.query_models import QueryCapabilities
from nautical_core.reconcile_report import ReconcileReport
from nautical_core.taskwarrior_client import TaskwarriorClient


ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "nautical_core" / "tools" / "nautical_query.py"
DOCTOR = ROOT / "nautical_core" / "tools" / "nautical_doctor.py"
QUEUE_STATUS = ROOT / "nautical_core" / "tools" / "nautical_queue_status.py"
RECONCILE = ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"
NAVIGATOR = ROOT / "nautical_navigator.py"


class OperatorProcessContractTests(unittest.TestCase):
    def _run(self, path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        merged = os.environ.copy()
        merged.update(env or {})
        return subprocess.run(
            [sys.executable, str(path), *args],
            text=True,
            capture_output=True,
            env=merged,
            timeout=15,
        )

    def _json(self, process: subprocess.CompletedProcess[str]) -> dict[str, object]:
        self.assertEqual(process.stderr, "", process.stderr)
        payload = json.loads(process.stdout)
        self.assertIsInstance(payload, dict)
        return payload

    def test_capabilities_is_strict_json_stdout(self) -> None:
        process = self._run(QUERY, "capabilities")
        self.assertEqual(process.returncode, 0)
        payload = self._json(process)
        self.assertEqual(payload.get("schema"), "nautical.query.capabilities")
        decoded = QueryCapabilities.from_mapping(payload)
        self.assertEqual(decoded.to_dict(), payload)

    def test_valid_operator_matrix_emits_one_json_document(self) -> None:
        """Operational subprocesses keep stdout machine-readable and diagnostics separate."""
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            cases = (
                (QUERY, ("capabilities",), {}),
                (QUEUE_STATUS, ("--taskdata", str(taskdata), "--json"), {}),
                (DOCTOR, ("--taskdata", str(taskdata), "--task-bin", "/bin/false", "--json", "--installation-only"), {}),
                (RECONCILE, ("--json", "--task-bin", str(taskdata / "missing-task")), {"TASKDATA": str(taskdata)}),
            )
            for path, args, environment in cases:
                process = self._run(path, *args, env=environment)
                self.assertNotIn("Traceback", process.stdout)
                self.assertNotIn("Traceback", process.stderr)
                self.assertEqual(process.stderr, "", (path.name, process.stderr))
                self.assertTrue(process.stdout.strip().startswith("{"), (path.name, process.stdout))
                payload = json.loads(process.stdout)
                self.assertIsInstance(payload, dict)
                self.assertTrue(str(payload.get("schema", "")).startswith("nautical."), path.name)

    def test_managed_runtime_operator_matrix_runs_outside_checkout(self) -> None:
        """Installed operator clients resolve the staged package without source imports."""
        with tempfile.TemporaryDirectory(prefix="nautical-managed-matrix-") as directory:
            runtime = Path(directory)
            shutil.copy2(ROOT / "nautical", runtime / "nautical")
            shutil.copy2(NAVIGATOR, runtime / NAVIGATOR.name)
            shutil.copytree(ROOT / "nautical_core", runtime / "nautical_core")
            (runtime / "nautical").chmod(0o755)
            environment = {
                "TASKDATA": str(runtime / "taskdata"),
                "TASKRC": str(runtime / "taskrc"),
                "PYTHONPATH": "",
                "PATH": os.environ.get("PATH", ""),
            }
            (runtime / "taskdata").mkdir()
            commands = (
                ("query", "capabilities"),
                ("query", "integrity", "--all"),
                ("queue-status", "--json"),
                ("doctor", "--json"),
                ("reconcile", "--json", "--task-bin", "/missing/task"),
                ("navigator", "--help"),
            )
            for args in commands:
                process = subprocess.run(
                    [sys.executable, str(runtime / "nautical"), *args],
                    cwd="/tmp",
                    text=True,
                    capture_output=True,
                    env=environment,
                    timeout=20,
                )
                self.assertNotIn("Traceback", process.stderr, args)
                self.assertNotIn(str(ROOT), process.stderr + process.stdout, args)
                if args[-1] == "--help":
                    self.assertEqual(process.returncode, 0, process.stderr)
                else:
                    self.assertTrue(process.stdout.strip(), args)
                    payload = json.loads(process.stdout)
                    self.assertIsInstance(payload, dict)
                    self.assertTrue(str(payload.get("schema", "")).startswith("nautical."), args)

    def test_malformed_request_fails_with_json_and_exit_code(self) -> None:
        process = self._run(QUERY, "occurrences", "--request", "{not-json")
        self.assertEqual(process.returncode, 2)
        payload = self._json(process)
        self.assertEqual(payload.get("status"), "invalid")
        failure = payload.get("failure")
        self.assertIsInstance(failure, dict)
        self.assertEqual(failure.get("code"), "invalid_request")

    def test_malformed_unicode_escape_keeps_json_boundary(self) -> None:
        """A lone surrogate in request JSON must not produce a traceback."""
        request = (
            '{"selector":{"all_tasks":true},"from":"2026-08-24",'
            '"count":1,"label":"\\ud800"}'
        )
        process = self._run(QUERY, "occurrences", "--request", request)
        self.assertIn(process.returncode, {0, 1, 2, 3})
        payload = self._json(process)
        self.assertTrue(str(payload.get("schema", "")).startswith("nautical."))

    def test_empty_taskdata_integrity_is_unavailable_not_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {"TASKDATA": directory, "TASKRC": str(Path(directory) / "taskrc")}
            process = self._run(QUERY, "integrity", "--all", env=environment)
            self.assertEqual(process.returncode, 3)
            payload = self._json(process)
            self.assertEqual(payload.get("status"), "unavailable")

    def test_malformed_taskwarrior_export_is_unavailable(self) -> None:
        """Invalid export JSON must fail closed at the operator process boundary."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = root / "task"
            task.write_text("#!/bin/sh\nprintf '{not-json\\n'\n", encoding="utf-8")
            task.chmod(0o755)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            environment = {
                "TASKDATA": str(taskdata),
                "TASKRC": str(root / "taskrc"),
                "PATH": f"{root}:{os.environ.get('PATH', '')}",
            }
            process = self._run(QUERY, "integrity", "--all", env=environment)
            self.assertEqual(process.returncode, 3)
            payload = self._json(process)
            self.assertEqual(payload.get("status"), "unavailable")

    def test_missing_taskwarrior_doctor_reports_json_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            process = self._run(
                DOCTOR,
                "--taskdata",
                directory,
                "--task-bin",
                str(Path(directory) / "missing-task"),
                "--json",
                "--installation-only",
            )
            self.assertNotEqual(process.returncode, 0)
            payload = self._json(process)
            self.assertEqual(payload.get("schema"), "nautical.doctor")
            self.assertIn(payload.get("status"), {"error", "warn"})

    def test_invalid_configuration_does_not_emit_plaintext_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config-nautical.toml"
            config.write_text("timezone = [broken\n", encoding="utf-8")
            environment = {"NAUTICAL_CONFIG": str(config), "TASKDATA": directory}
            process = self._run(
                DOCTOR,
                "--taskdata",
                directory,
                "--task-bin",
                "/bin/false",
                "--json",
                "--installation-only",
                env=environment,
            )
            self.assertNotEqual(process.returncode, 0)
            payload = self._json(process)
            self.assertEqual(payload.get("schema"), "nautical.doctor")

    def test_missing_binary_is_typed_and_non_retryable(self) -> None:
        result = TaskwarriorClient(("/missing/task",)).execute([], purpose="probe", timeout=1.0)
        self.assertEqual(result.kind, CommandFailureKind.MISSING_BINARY)
        self.assertNotIn(result.kind, {CommandFailureKind.TIMEOUT, CommandFailureKind.BUSY})

    def test_timeout_is_typed_and_retryable(self) -> None:
        result = TaskwarriorClient((sys.executable,)).execute(
            ("-c", "import time; time.sleep(1)"), purpose="timeout", timeout=0.01,
        )
        self.assertEqual(result.kind, CommandFailureKind.TIMEOUT)
        self.assertIn(result.kind, {CommandFailureKind.TIMEOUT, CommandFailureKind.BUSY})

    def test_lock_output_is_retryable_and_bounded(self) -> None:
        result = TaskwarriorClient((sys.executable,)).execute(
            ("-c", "import sys; sys.stderr.write('database is locked') ; sys.exit(1)"),
            purpose="lock", timeout=1.0, attempts=2,
        )
        self.assertEqual(result.kind, CommandFailureKind.BUSY)
        self.assertEqual(result.attempt, 2)

    def test_noisy_stderr_does_not_change_success_classification(self) -> None:
        result = TaskwarriorClient((sys.executable,)).execute(
            ("-c", "import sys; sys.stderr.write('informational noise')"), purpose="noise", timeout=1.0,
        )
        self.assertEqual(result.kind, CommandFailureKind.SUCCESS)
        self.assertEqual(result.stderr, "informational noise")

    def test_operator_json_entry_points_keep_stdout_machine_readable(self) -> None:
        """Queue and reconcile startup failures use the same JSON-only boundary."""
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            queue = self._run(QUEUE_STATUS, "--taskdata", str(taskdata), "--json")
            self.assertIn(queue.returncode, {0, 1, 2, 3})
            queue_payload = self._json(queue)
            self.assertTrue(str(queue_payload.get("schema", "")).startswith("nautical."))

            reconcile = self._run(
                RECONCILE, "--json", "--task-bin", str(taskdata / "missing-task"),
                env={"TASKDATA": str(taskdata)},
            )
            self.assertNotEqual(reconcile.returncode, 0)
            reconcile_payload = self._json(reconcile)
            self.assertEqual(reconcile_payload.get("schema"), "nautical.reconcile")

    def test_operator_documents_round_trip_through_public_decoders(self) -> None:
        """Doctor and queue use v2; reconcile remains JSON-native and stable."""
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            queue = self._run(QUEUE_STATUS, "--taskdata", str(taskdata), "--json")
            queue_payload = self._json(queue)
            queue_decoded = OperatorV2Result.from_mapping(queue_payload)
            self.assertEqual(queue_decoded.to_dict(), queue_payload)

            reconcile = self._run(
                RECONCILE, "--json", "--task-bin", str(taskdata / "missing-task"),
                env={"TASKDATA": str(taskdata)},
            )
            reconcile_payload = self._json(reconcile)
            reconcile_decoded = ReconcileReport.from_mapping(reconcile_payload)
            self.assertEqual(reconcile_decoded.to_dict(), reconcile_payload)

            doctor = self._run(
                DOCTOR, "--taskdata", str(taskdata), "--task-bin",
                str(taskdata / "missing-task"), "--json", "--installation-only",
            )
            doctor_payload = self._json(doctor)
            doctor_decoded = DoctorReport.from_mapping(doctor_payload)
            self.assertEqual(doctor_decoded.to_dict(), doctor_payload)

    def test_installation_report_round_trips_through_public_decoder(self) -> None:
        report = {
            "schema": "nautical.install.verification",
            "version": 1,
            "status": "attention",
            "checks": [{"name": "Runtime", "status": "passed", "detail": "active"}],
            "manual_actions": [],
            "optional_actions": [{"id": "launcher.path", "message": "path", "action": "add it"}],
            "future": {"revision": 2},
        }
        decoded = InstallationVerificationReport.from_mapping(report)
        self.assertEqual(decoded.to_dict(), report)

    def test_navigator_validation_keeps_diagnostics_off_stdout_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            process = self._run(
                NAVIGATOR, "--validate", "w:mon",
                env={"TASKDATA": str(taskdata), "TASKRC": str(taskdata / "taskrc")},
            )
        self.assertEqual(process.returncode, 0, process.stderr or process.stdout)
        self.assertEqual(process.stderr, "", process.stderr)
        self.assertTrue(process.stdout.strip())

    def test_installed_layout_query_runs_outside_source_checkout(self) -> None:
        """A managed release must resolve its package from its own directory."""
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            shutil.copytree(ROOT / "nautical_core", release / "nautical_core")
            query = release / "nautical_core" / "tools" / "nautical_query.py"
            environment = {"TASKDATA": str(Path(directory) / "taskdata")}
            environment["PYTHONPATH"] = ""
            process = self._run(query, "capabilities", env=environment)
            self.assertEqual(process.returncode, 0, process.stderr)
            payload = self._json(process)
            self.assertEqual(payload.get("schema"), "nautical.query.capabilities")

    def test_installed_layout_operator_roots_keep_json_contracts(self) -> None:
        """All core operator roots resolve from an isolated managed release."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            shutil.copytree(ROOT / "nautical_core", release / "nautical_core")
            taskdata = root / "taskdata"
            taskdata.mkdir()
            environment = {"PYTHONPATH": "", "TASKDATA": str(taskdata)}

            doctor = self._run(
                release / "nautical_core" / "tools" / "nautical_doctor.py",
                "--taskdata", str(taskdata), "--task-bin", str(root / "missing-task"),
                "--json", "--installation-only", env=environment,
            )
            self.assertNotEqual(doctor.returncode, 0)
            self.assertEqual(self._json(doctor).get("schema"), "nautical.doctor")

            queue = self._run(
                release / "nautical_core" / "tools" / "nautical_queue_status.py",
                "--taskdata", str(taskdata), "--json", env=environment,
            )
            self.assertIn(queue.returncode, {0, 1, 2, 3})
            self.assertTrue(str(self._json(queue).get("schema", "")).startswith("nautical."))

            reconcile = self._run(
                release / "nautical_core" / "tools" / "nautical_reconcile.py",
                "--json", "--task-bin", str(root / "missing-task"), "--no-housekeeping",
                env=environment,
            )
            self.assertNotEqual(reconcile.returncode, 0)
            self.assertEqual(self._json(reconcile).get("schema"), "nautical.reconcile")


if __name__ == "__main__":
    unittest.main()
