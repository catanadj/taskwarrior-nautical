from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HookSubprocessFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._taskdata_context = tempfile.TemporaryDirectory()
        self.taskdata = self._taskdata_context.name

    def tearDown(self) -> None:
        self._taskdata_context.cleanup()

    def run_hook(
        self,
        hook: str,
        payload: str,
        *,
        diagnostics: bool = False,
        extra_environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "TASKDATA": self.taskdata,
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TZ": "UTC",
            }
        )
        if extra_environment:
            environment.update(extra_environment)
        if diagnostics:
            environment["NAUTICAL_DIAG"] = "1"
        else:
            environment.pop("NAUTICAL_DIAG", None)
        return subprocess.run(
            [sys.executable, str(ROOT / hook)],
            input=payload,
            text=True,
            capture_output=True,
            env=environment,
            timeout=15,
        )


class HookInputContractTests(HookSubprocessFixture):
    def _run(self, hook: str, payload: str, *, diagnostics: bool = False) -> subprocess.CompletedProcess[str]:
        return self.run_hook(hook, payload, diagnostics=diagnostics)

    def test_invalid_inputs_never_traceback_or_emit_stdout(self) -> None:
        invalid = ("", "{\"uuid\":", "{not-json", "[]", "x" * (10 * 1024 * 1024 + 1))
        for hook in ("on-add.nautical", "on-modify.nautical"):
            for payload in invalid:
                with self.subTest(hook=hook, payload_size=len(payload)):
                    process = self._run(hook, payload)
                    self.assertNotEqual(process.returncode, 0)
                    self.assertEqual(process.stdout, "")
                    self.assertNotIn("Traceback", process.stderr)
                    self.assertNotIn("[nautical]", process.stderr)

    def test_invalid_input_diagnostics_are_opt_in(self) -> None:
        for hook in ("on-add.nautical", "on-modify.nautical"):
            with self.subTest(hook=hook):
                quiet = self._run(hook, "{not-json")
                diagnostic = self._run(hook, "{not-json", diagnostics=True)
                self.assertNotIn("[nautical]", quiet.stderr)
                self.assertIn("[nautical]", diagnostic.stderr)

    def test_on_exit_ignores_malformed_and_oversized_input_silently(self) -> None:
        for payload in ("", "{not-json", "[]", "x" * (10 * 1024 * 1024 + 1)):
            with self.subTest(payload_size=len(payload)):
                process = self._run("on-exit.nautical", payload)
                self.assertEqual(process.returncode, 0)
                self.assertEqual(process.stdout, "")
                self.assertEqual(process.stderr, "")

    def test_unicode_task_payload_keeps_strict_json_stdout(self) -> None:
        task = {
            "uuid": "11111111-1111-1111-1111-111111111111",
            "description": "café ăîșț",
            "status": "pending",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        added = self._run("on-add.nautical", json.dumps(task, ensure_ascii=False))
        self.assertEqual(added.returncode, 0)
        self.assertEqual(json.loads(added.stdout), task)
        self.assertEqual(added.stderr, "")

        modified_task = dict(task, description="updated café ăîșț", modified="20260101T000001Z")
        modified = self._run(
            "on-modify.nautical",
            json.dumps(task, ensure_ascii=False) + "\n" + json.dumps(modified_task, ensure_ascii=False),
        )
        self.assertEqual(modified.returncode, 0)
        self.assertEqual(json.loads(modified.stdout), modified_task)
        self.assertEqual(modified.stderr, "")

    def test_diagnostics_never_contaminate_task_json_stdout(self) -> None:
        task = {
            "uuid": "22222222-2222-4222-8222-222222222222",
            "description": "plain task",
            "status": "pending",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        process = self._run(
            "on-add.nautical",
            json.dumps(task, ensure_ascii=False),
            diagnostics=True,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(json.loads(process.stdout), task)
        self.assertNotIn("[nautical]", process.stdout)
        self.assertTrue(process.stderr == "" or "[nautical]" in process.stderr)

    def test_on_add_anchor_routes_keep_json_stdout_and_panel_stderr(self) -> None:
        task = {
            "uuid": "44444444-4444-4444-8444-444444444444",
            "description": "weekly review",
            "status": "pending",
            "anchor": "w:mon",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        process = self._run("on-add.nautical", json.dumps(task, ensure_ascii=False))
        self.assertEqual(process.returncode, 0, process.stderr)
        result = json.loads(process.stdout)
        self.assertEqual(result["chain"], "on")
        self.assertEqual(result["anchor"], "w:mon")
        self.assertNotIn("Anchor Preview", process.stdout)

    def test_on_add_invalid_anchor_is_a_clean_failure(self) -> None:
        task = {
            "uuid": "55555555-5555-4555-8555-555555555555",
            "description": "invalid recurrence",
            "status": "pending",
            "anchor": "not-a-valid-expression",
            "entry": "20260101T000000Z",
            "modified": "20260101T000000Z",
        }
        process = self._run("on-add.nautical", json.dumps(task, ensure_ascii=False))
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stdout, "")
        self.assertNotIn("Traceback", process.stderr)


if __name__ == "__main__":
    unittest.main()
