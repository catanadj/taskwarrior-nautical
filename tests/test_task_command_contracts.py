"""Direct contracts for Taskwarrior command classification and retry policy."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
from nautical_core import hook_support, modify_command_effects, runtime_command
from nautical_core.integration_models import (
    CommandFailureKind,
    TaskCommand,
    TaskCommandResult,
)
from nautical_core.task_command import failure_message, run_task_command


class TaskCommandContractTests(unittest.TestCase):
    def test_client_observation_preserves_evidence_without_command_contents(self) -> None:
        observations = []

        class Observer:
            def observe(self, observation):
                observations.append(observation)

        from nautical_core.taskwarrior_client import TaskwarriorClient

        result = TaskwarriorClient((sys.executable,), observer=Observer()).execute(
            ("-c", "import sys; print('Répéter 🌊'); print('warning', file=sys.stderr)"),
            purpose="test export",
            timeout=2.0,
            use_tempfiles=True,
        )

        self.assertTrue(result.ok, result)
        self.assertEqual(result.stdout.strip(), "Répéter 🌊")
        self.assertEqual(result.stderr.strip(), "warning")
        self.assertEqual(result.command.timeout, 2.0)
        self.assertEqual(result.attempt, 1)
        self.assertGreaterEqual(result.duration, 0.0)
        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation.purpose, "test export")
        self.assertIs(observation.kind, CommandFailureKind.SUCCESS)
        self.assertFalse(hasattr(observation, "argv"))
        self.assertFalse(hasattr(observation, "stdout"))

    def test_boundary_failures_preserve_classification_and_actionable_evidence(self) -> None:
        missing = run_task_command("/missing/nautical-task", ["export"], timeout=1.0)
        self.assertIs(missing.kind, CommandFailureKind.MISSING_BINARY)
        self.assertEqual(missing.returncode, 127)
        self.assertIn(
            "/missing/nautical-task", failure_message(missing, "task export")
        )

        timed_out = run_task_command(
            sys.executable,
            ["-c", "import sys,time; print('partial Ω', flush=True); time.sleep(2)"],
            timeout=0.05,
        )
        self.assertIs(timed_out.kind, CommandFailureKind.TIMEOUT)
        self.assertIn("Ω", timed_out.stdout)
        self.assertIn("0.05s", failure_message(timed_out, "task export"))

        rejected = run_task_command(
            sys.executable,
            ["-c", "import sys; print('bad command', file=sys.stderr); sys.exit(3)"],
        )
        self.assertIs(rejected.kind, CommandFailureKind.REJECTED)
        self.assertEqual(failure_message(rejected, "task export"), "bad command")

    def test_lock_retries_are_opt_in(self) -> None:
        args = [
            "-c",
            "import sys; print('database is locked', file=sys.stderr); sys.exit(1)",
        ]
        retried = run_task_command(
            sys.executable, args, retry_locks=True, retry_delay=0.0
        )
        self.assertIs(retried.kind, CommandFailureKind.BUSY)
        self.assertEqual(retried.attempt, 2)

        single_attempt = run_task_command(sys.executable, args)
        self.assertIs(single_attempt.kind, CommandFailureKind.BUSY)
        self.assertEqual(single_attempt.attempt, 1)


class RuntimeFacadeCommandTests(unittest.TestCase):
    def test_run_task_result_preserves_text_input_with_temporary_output(self) -> None:
        result = core.run_task_result(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
            input_text="hello\n",
            timeout=2.0,
            retries=1,
            use_tempfiles=True,
        )

        self.assertTrue(result.ok, result)
        self.assertEqual(result.stdout, "hello\n")

    def test_run_task_result_classifies_temporary_output_timeout(self) -> None:
        result = core.run_task_result(
            [sys.executable, "-c", "import time; time.sleep(0.25); print('late')"],
            timeout=0.05,
            retries=1,
            use_tempfiles=True,
        )

        self.assertFalse(result.ok)
        self.assertIs(result.kind, CommandFailureKind.TIMEOUT)

    def test_run_task_result_preserves_metadata_and_retry_policy(self) -> None:
        success = core.run_task_result(
            [sys.executable, "-c", "print('typed')"], timeout=2.0, retries=1
        )
        self.assertTrue(success.ok, success)
        self.assertEqual(success.stdout.strip(), "typed")
        self.assertEqual(success.attempt, 1)
        self.assertEqual(success.command.timeout, 2.0)

        busy = core.run_task_result(
            [sys.executable, "-c", "import sys; print('database is locked', file=sys.stderr); sys.exit(3)"],
            timeout=1.0,
            retries=3,
            retry_delay=0.0,
        )
        self.assertFalse(busy.ok)
        self.assertIs(busy.kind, CommandFailureKind.BUSY)
        self.assertEqual(busy.attempt, 3)

        rejected = core.run_task_result(
            [sys.executable, "-c", "import sys; print('invalid task', file=sys.stderr); sys.exit(3)"],
            timeout=1.0,
            retries=3,
            retry_delay=0.0,
        )
        self.assertFalse(rejected.ok)
        self.assertIn("invalid task", rejected.stderr)
        self.assertEqual(rejected.attempt, 1)

    def test_temporary_output_falls_back_to_pipes_when_tempfiles_are_unavailable(self) -> None:
        with patch(
            "nautical_core.taskwarrior_client.tempfile.TemporaryFile",
            side_effect=OSError("tempfile unavailable"),
        ):
            result = core.run_task_result(
                [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
                input_text="abc ✓\n",
                timeout=1.0,
                retries=1,
                use_tempfiles=True,
            )

        self.assertTrue(result.ok, result)
        self.assertEqual(result.stdout, "abc ✓\n")


class HookCommandBoundaryTests(unittest.TestCase):
    def test_hook_runner_preserves_typed_result_identity(self) -> None:
        expected = TaskCommandResult(
            TaskCommand(("task", "export"), "test export", 3.0),
            0,
            "[]",
            "",
            CommandFailureKind.SUCCESS,
            2,
            0.1,
        )
        received = {}

        def runner(command, **kwargs):
            received["command"] = command
            received["kwargs"] = kwargs
            return expected

        result = hook_support.run_task_result(
            run_task=runner,
            cmd=["task", "export"],
            timeout=3.0,
            retries=2,
            use_tempfiles=True,
        )

        self.assertIs(result, expected)
        self.assertEqual(received["command"], ["task", "export"])
        self.assertEqual(received["kwargs"]["timeout"], 3.0)
        self.assertEqual(received["kwargs"]["retries"], 2)
        self.assertTrue(received["kwargs"]["use_tempfiles"])

    def test_shared_runtime_runner_preserves_utf8_output_and_failure_status(self) -> None:
        success = runtime_command.run_task_result(
            [sys.executable, "-c", "print('shared ✓')"], timeout=2.0, retries=1
        )
        self.assertTrue(success.ok, success)
        self.assertEqual(success.stdout.strip(), "shared ✓")

        failed = runtime_command.run_task_result(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            timeout=2.0,
            retries=1,
        )
        self.assertFalse(failed.ok)
        self.assertEqual(failed.returncode, 3)

    def test_modify_command_effects_records_success_and_nonzero_failure(self) -> None:
        counters = []
        records = []
        host = SimpleNamespace(
            _run_task_diag_bucket=lambda _cmd: "contract",
            _diag_count=lambda *args: counters.append(args),
            _diag_record_run_task=lambda *args, **kwargs: records.append((args, kwargs)),
            _diag=lambda _message: None,
            _task_cmd_prefix=lambda: ["task"],
        )
        ports = modify_command_effects.command_ports_for(host)

        succeeded = modify_command_effects.run_task_result(
            ports, [sys.executable, "-c", "print('ok')"], timeout=2.0, retries=1
        )
        failed = modify_command_effects.run_task_result(
            ports,
            [sys.executable, "-c", "import sys; sys.exit(2)"],
            timeout=2.0,
            retries=1,
        )

        self.assertTrue(succeeded.ok)
        self.assertFalse(failed.ok)
        self.assertEqual(len(records), 2)
        self.assertIn(("run_task_failures",), counters)


if __name__ == "__main__":
    unittest.main()
