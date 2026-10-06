from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import shutil

from nautical_core.integration_models import CommandFailureKind, FailureEvidence
from nautical_core.doctor_report import DoctorReport
from nautical_core.installation_report import InstallationVerificationReport
from nautical_core.operator_models import OperatorV2Result
from nautical_core.query_models import QueryCapabilities
from nautical_core.taskwarrior_client import TaskwarriorClient
from tests.support.hook_process import HookSubprocessFixture


ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "nautical_core" / "tools" / "nautical_query.py"
DOCTOR = ROOT / "nautical_core" / "tools" / "nautical_doctor.py"
QUEUE_STATUS = ROOT / "nautical_core" / "tools" / "nautical_queue_status.py"
RECONCILE = ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"
NAVIGATOR = ROOT / "nautical_navigator.py"
DEV_QUEUE_STATUS = ROOT / "dev_tools" / "nautical_queue_status.py"
HEALTH_CHECK = ROOT / "dev_tools" / "nautical_health_check.py"


class OperatorProcessContractTests(HookSubprocessFixture):
    def _run(self, path: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        if path.resolve().is_relative_to((ROOT / "nautical_core").resolve()):
            return self.run_python_command(
                [sys.executable, str(path), *args],
                env=env,
                timeout=15,
            )
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

    def test_health_check_json_ok_empty_taskdata(self) -> None:
        with tempfile.TemporaryDirectory() as taskdata:
            process = self._run(HEALTH_CHECK, "--taskdata", taskdata, "--json")

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(self._json(process).get("status"), "ok")

    def test_health_check_critical_outbox_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as taskdata:
            outbox = Path(taskdata) / ".nautical-state" / ".nautical_lifecycle_outbox.db"
            outbox.parent.mkdir()
            outbox.write_text("x" * 64, encoding="utf-8")
            process = self._run(
                HEALTH_CHECK,
                "--taskdata",
                taskdata,
                "--outbox-warn-bytes",
                "32",
                "--outbox-crit-bytes",
                "48",
                "--json",
            )

        self.assertEqual(process.returncode, 2, process.stderr)
        self.assertEqual(self._json(process).get("status"), "crit")

    def test_health_check_critical_outbox_rows(self) -> None:
        import sqlite3

        from nautical_core.lifecycle.outbox import _LifecycleOutboxRepository

        with tempfile.TemporaryDirectory() as taskdata:
            repository = _LifecycleOutboxRepository(Path(taskdata))
            self.assertTrue(repository.open().ok)
            with sqlite3.connect(repository.path) as connection:
                connection.execute(
                    "INSERT INTO lifecycle_outbox "
                    "(intent_id, plan_json, plan_fingerprint, parent_guard_json, "
                    "configuration_fingerprint, schedule_fingerprint, "
                    "lifecycle_stage, processing_state, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        "health-row",
                        "{}",
                        "test",
                        "{}",
                        "test",
                        "test",
                        "planned",
                        "ready",
                        1.0,
                        1.0,
                    ),
                )
            process = self._run(
                HEALTH_CHECK,
                "--taskdata",
                taskdata,
                "--outbox-warn-bytes",
                "1048576",
                "--outbox-crit-bytes",
                "10485760",
                "--outbox-warn-rows",
                "1",
                "--outbox-crit-rows",
                "1",
                "--json",
            )

        self.assertEqual(process.returncode, 2, process.stderr)
        payload = self._json(process)
        self.assertEqual(payload.get("status"), "crit")
        self.assertEqual((payload.get("outbox") or {}).get("rows"), 1)

    def test_queue_status_json_ok_empty_taskdata(self) -> None:
        with tempfile.TemporaryDirectory() as taskdata:
            process = self._run(DEV_QUEUE_STATUS, "--taskdata", taskdata, "--json")

        self.assertEqual(process.returncode, 0, process.stderr)
        payload = self._json(process)
        self.assertEqual(payload.get("status"), "ok")
        outbox = payload.get("outbox") or {}
        self.assertEqual(outbox.get("states"), {})
        self.assertEqual((outbox.get("schema") or {}).get("status"), "absent")

    def test_core_queue_status_does_not_create_missing_outbox(self) -> None:
        from nautical_core.lifecycle.outbox import lifecycle_outbox_path

        with tempfile.TemporaryDirectory() as taskdata:
            state_dir = Path(taskdata) / ".nautical-state"
            process = self._run(QUEUE_STATUS, "--taskdata", taskdata, "--json")
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(process.stderr, "")
            self.assertFalse(state_dir.exists())
            self.assertFalse(lifecycle_outbox_path(Path(taskdata)).exists())

    def test_queue_status_explicit_prune_reports_maintenance_result(self) -> None:
        with tempfile.TemporaryDirectory() as taskdata:
            process = self._run(
                DEV_QUEUE_STATUS,
                "--taskdata",
                taskdata,
                "--prune-acknowledged",
                "--json",
            )

        self.assertEqual(process.returncode, 0, process.stderr)
        maintenance = self._json(process).get("maintenance") or {}
        self.assertIs(maintenance.get("ok"), True)
        self.assertEqual(maintenance.get("removed"), 0)

    def test_operator_queue_status_json_ok_empty_taskdata(self) -> None:
        with tempfile.TemporaryDirectory() as taskdata:
            process = self._run(QUEUE_STATUS, "--taskdata", taskdata, "--json")

        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(self._json(process).get("status"), "ok")

    def test_capabilities_is_strict_json_stdout(self) -> None:
        process = self._run(QUERY, "capabilities")
        self.assertEqual(process.returncode, 0)
        payload = self._json(process)
        self.assertEqual(payload.get("schema"), "nautical.query.capabilities")
        decoded = QueryCapabilities.from_mapping(payload)
        self.assertEqual(decoded.to_dict(), payload)

    def test_query_launcher_keeps_one_document_stdout_and_separates_diagnostics(self) -> None:
        """The managed query CLI keeps invalid-request forms equivalent and diagnostics on stderr."""
        launcher = ROOT / "nautical"
        environment = {**os.environ, "PYTHONPATH": str(ROOT)}

        capabilities = subprocess.run(
            [sys.executable, str(launcher), "query", "capabilities"],
            text=True,
            capture_output=True,
            env=environment,
            timeout=15,
        )
        self.assertEqual(capabilities.returncode, 0, capabilities.stderr)
        self.assertEqual(capabilities.stderr, "")
        self.assertEqual(len(capabilities.stdout.splitlines()), 1)
        self.assertEqual(json.loads(capabilities.stdout).get("schema"), "nautical.query.capabilities")

        inline = subprocess.run(
            [sys.executable, str(launcher), "query", "occurrences", "--request", "{}"],
            text=True,
            capture_output=True,
            env=environment,
            timeout=15,
        )
        stdin = subprocess.run(
            [sys.executable, str(launcher), "query", "occurrences", "--request", "-"],
            input="{}",
            text=True,
            capture_output=True,
            env=environment,
            timeout=15,
        )
        self.assertEqual(inline.returncode, 2)
        self.assertEqual(stdin.returncode, 2)
        self.assertEqual(inline.stderr, "")
        self.assertEqual(stdin.stderr, "")
        self.assertEqual(inline.stdout, stdin.stdout)
        self.assertEqual(len(inline.stdout.splitlines()), 1)
        self.assertEqual(json.loads(inline.stdout).get("schema"), "nautical.query.occurrences")

        diagnostic = subprocess.run(
            [sys.executable, str(launcher), "query", "occurrences", "--request", "{}"],
            text=True,
            capture_output=True,
            env={**environment, "NAUTICAL_DIAG": "1"},
            timeout=15,
        )
        self.assertEqual(diagnostic.returncode, 2)
        self.assertEqual(len(diagnostic.stdout.splitlines()), 1)
        self.assertEqual(json.loads(diagnostic.stdout).get("schema"), "nautical.query.occurrences")
        self.assertTrue(diagnostic.stderr.startswith("[nautical] query:"), diagnostic.stderr)

    def test_process_interruption_is_typed_and_retryable(self) -> None:
        client = TaskwarriorClient((sys.executable, "-c", "import signal; signal.pause()"))
        result = client.execute((), purpose="interruption-test", timeout=0.1, attempts=1)
        self.assertEqual(result.kind, CommandFailureKind.TIMEOUT)
        self.assertEqual(result.returncode, 124)
        self.assertGreaterEqual(result.duration, 0.0)
        evidence = FailureEvidence(
            result.command, result.kind, result.returncode, result.attempt,
            result.duration, retryable=True, detail="process timeout",
        )
        self.assertTrue(evidence.retryable)

    def test_taskwarrior_client_retries_only_transient_failures(self) -> None:
        sleeps = []
        busy = TaskwarriorClient((sys.executable,), sleeper=sleeps.append).execute(
            ("-c", "import sys; print('database is locked', file=sys.stderr); sys.exit(1)"),
            purpose="busy export", timeout=2.0, attempts=3, retry_delay=0.01,
        )
        self.assertEqual(busy.kind, CommandFailureKind.BUSY)
        self.assertEqual(busy.attempt, 3)
        self.assertEqual(sleeps, [0.01, 0.02])

        sleeps.clear()
        rejected = TaskwarriorClient((sys.executable,), sleeper=sleeps.append).execute(
            ("-c", "import sys; print('invalid filter', file=sys.stderr); sys.exit(2)"),
            purpose="rejected export", timeout=2.0, attempts=3, retry_delay=0.01,
        )
        self.assertEqual(rejected.kind, CommandFailureKind.REJECTED)
        self.assertEqual(rejected.attempt, 1)
        self.assertEqual(sleeps, [])

        timed_out = TaskwarriorClient((sys.executable,)).execute(
            ("-c", "import time; time.sleep(2)"),
            purpose="bounded export", timeout=0.02, attempts=1,
        )
        self.assertEqual(timed_out.kind, CommandFailureKind.TIMEOUT)
        self.assertEqual(timed_out.returncode, 124)
        self.assertLess(timed_out.duration, 1.0)

    def test_timeout_terminates_descendant_process_group_within_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ready = Path(directory) / "ready"
            child_pid = Path(directory) / "child.pid"
            code = (
                "import os,signal,subprocess,sys,time; "
                f"child=subprocess.Popen([sys.executable, '-c', "
                "'import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); signal.pause()']); "
                f"open({str(child_pid)!r}, 'w').write(str(child.pid)); "
                f"open({str(ready)!r}, 'w').close(); signal.pause()"
            )
            result_box: list[object] = []

            def run() -> None:
                result_box.append(TaskwarriorClient((sys.executable, "-c", code)).execute(
                    (), purpose="descendant-timeout", timeout=0.5, attempts=1,
                ))

            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and not ready.exists():
                time.sleep(0.01)
            self.assertTrue(ready.exists(), "child did not signal readiness")
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive(), "timeout cleanup did not finish")
            result = result_box[0]
            self.assertEqual(result.kind, CommandFailureKind.TIMEOUT)
            pid = int(child_pid.read_text())
            # The child may briefly remain as a zombie while it is reaped by
            # init; either absence or a zombie state proves it is no longer
            # executing.  Do not mistake ``kill(pid, 0)`` for liveness.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                try:
                    state = Path(f"/proc/{pid}/stat").read_text().split()[2]
                except (FileNotFoundError, ProcessLookupError):
                    state = "Z"
                if state == "Z":
                    break
                time.sleep(0.01)
            self.assertEqual(state, "Z", f"descendant still running (state={state})")

    def test_timeout_with_tempfile_outputs_remains_bounded(self) -> None:
        code = "import signal,sys; print('partial output', flush=True); signal.pause()"
        client = TaskwarriorClient((sys.executable, "-c", code))
        result = client.execute(
            (), purpose="tempfile-timeout", timeout=0.5, attempts=1, use_tempfiles=True,
        )
        self.assertEqual(result.kind, CommandFailureKind.TIMEOUT)
        self.assertIn("partial output", result.stdout)

    def test_integrity_unavailable_result_keeps_failure_evidence(self) -> None:
        from nautical_core.query_report import to_operator_result

        result = to_operator_result({
            "schema": "nautical.query.integrity",
            "version": 1,
            "operation": "integrity",
            "status": "unavailable",
            "findings": [],
            "plans": [],
            "reason": "chain snapshot unavailable",
        })
        self.assertIsInstance(result, OperatorV2Result)
        self.assertEqual(result.status.value, "unavailable")
        self.assertIsNotNone(result.failure)
        assert result.failure is not None
        self.assertEqual(result.failure.code, "query_unavailable")
        self.assertEqual(result.failure.message, "chain snapshot unavailable")

    def test_report_converters_share_one_typed_result_contract(self) -> None:
        from nautical_core.doctor_report import to_operator_result as doctor_result
        from nautical_core.query_report import to_operator_result as query_result
        from nautical_core.reconcile_report import to_operator_result as reconcile_result

        results = (
            query_result({
                "schema": "nautical.query.occurrences",
                "version": 1,
                "operation": "occurrences",
                "status": "found",
                "results": [{"description": "café"}],
            }),
            doctor_result({
                "schema": "nautical.doctor",
                "schema_version": 1,
                "status": "ok",
                "operator_findings": [{"message": "café"}],
            }),
            reconcile_result({
                "schema": "nautical.reconcile",
                "schema_version": 1,
                "status": "ok",
                "mode": "dry-run",
                "message": "café",
            }),
        )
        self.assertTrue(all(isinstance(result, OperatorV2Result) for result in results))
        self.assertEqual(results[0].payload["results"][0]["description"], "café")
        self.assertEqual(results[2].payload["message"], "café")

    def test_report_converters_keep_typed_failure_evidence(self) -> None:
        from nautical_core.doctor_report import to_operator_result as doctor_result
        from nautical_core.query_report import to_operator_result as query_result
        from nautical_core.reconcile_report import to_operator_result as reconcile_result

        results = (
            query_result({
                "schema": "nautical.query.occurrences",
                "version": 1,
                "operation": "occurrences",
                "status": "invalid",
                "failure": {"code": "bad_query", "message": "invalid café"},
            }),
            doctor_result({
                "schema": "nautical.doctor",
                "schema_version": 1,
                "status": "error",
                "operator_findings": [{"code": "broken", "message": "invalid café", "evidence": {"x": 1}}],
            }),
            reconcile_result({
                "schema": "nautical.reconcile",
                "schema_version": 1,
                "status": "error",
                "mode": "apply",
                "errors": ["invalid café"],
            }),
        )
        for result in results:
            self.assertIsInstance(result, OperatorV2Result)
            self.assertIsNotNone(result.failure)
            assert result.failure is not None
            self.assertTrue(result.failure.message)

    def test_reconcile_degraded_status_maps_to_findings_contract(self) -> None:
        from nautical_core.reconcile_report import to_operator_result

        result = to_operator_result({
            "schema": "nautical.reconcile",
            "status": "degraded",
            "mode": "apply",
            "manual_review": 1,
        })
        self.assertEqual(result.status.value, "attention")

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
                    if args == ("query", "capabilities"):
                        self.assertEqual(process.stderr, "")
                        self.assertEqual(len(process.stdout.splitlines()), 1)
                        self.assertEqual(payload.get("schema"), "nautical.query.capabilities")

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
            ("-c", "import signal; signal.pause()"), purpose="timeout", timeout=0.1,
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
        """Doctor, queue, and reconcile expose stable JSON operator documents."""
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
            reconcile_decoded = OperatorV2Result.from_mapping(reconcile_payload)
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
