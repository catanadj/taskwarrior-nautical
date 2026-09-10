from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "nautical_core" / "tools" / "nautical_query.py"
DOCTOR = ROOT / "nautical_core" / "tools" / "nautical_doctor.py"
RECONCILE = ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"


class OperatorCommandContractTests(unittest.TestCase):
    def _run(
        self,
        path: Path,
        *args: str,
        env: dict[str, str] | None = None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        merged = os.environ.copy()
        merged.update(env or {})
        return subprocess.run(
            [sys.executable, str(path), *args],
            input=input_text,
            text=True,
            capture_output=True,
            env=merged,
            timeout=15,
        )

    def _json(self, process: subprocess.CompletedProcess[str]) -> dict[str, object]:
        payload = json.loads(process.stdout)
        self.assertIsInstance(payload, dict)
        return payload

    def test_query_empty_stdin_is_structured_invalid_request(self) -> None:
        process = self._run(QUERY, "occurrences", input_text="\n")
        self.assertEqual(process.returncode, 2)
        self.assertEqual(process.stderr, "")
        payload = self._json(process)
        self.assertEqual(payload.get("schema"), "nautical.query.occurrences")
        self.assertEqual(payload.get("status"), "invalid")
        self.assertEqual(payload["failure"]["code"], "invalid_request")  # type: ignore[index]

    def test_query_diagnostics_are_stderr_only_when_enabled(self) -> None:
        process = self._run(
            QUERY,
            "occurrences",
            "--request",
            "{not-json",
            env={"NAUTICAL_DIAG": "1"},
        )
        self.assertEqual(process.returncode, 2)
        self.assertTrue(process.stderr.startswith("[nautical] query:"))
        payload = self._json(process)
        self.assertEqual(payload.get("status"), "invalid")
        self.assertNotIn("[nautical]", process.stdout)

    def test_doctor_installation_report_is_json_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            taskdata = Path(directory)
            process = self._run(
                DOCTOR,
                "--taskdata",
                str(taskdata),
                "--task-bin",
                str(taskdata / "missing-task"),
                "--json",
                "--installation-only",
            )
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(process.stderr, "")
        payload = self._json(process)
        self.assertEqual(payload.get("schema"), "nautical.doctor")
        self.assertIn(payload.get("status"), {"error", "warn"})
        self.assertIsInstance(payload.get("findings"), list)

    def test_reconcile_dry_run_reports_mode_without_apply_effects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taskdata = root / "taskdata"
            taskdata.mkdir()
            log = root / "task.log"
            task = root / "task"
            task.write_text(
                "#!/usr/bin/env python3\n"
                "from pathlib import Path\n"
                f"log = Path({str(log)!r})\n"
                "with log.open('a', encoding='utf-8') as stream: stream.write(' '.join(__import__('sys').argv[1:]) + '\\n')\n"
                "print('[]')\n",
                encoding="utf-8",
            )
            task.chmod(0o755)
            process = self._run(
                RECONCILE,
                "--dry-run",
                "--json",
                "--no-housekeeping",
                "--task-bin",
                str(task),
                env={"TASKDATA": str(taskdata)},
            )
            self.assertEqual(process.stderr, "")
            payload = self._json(process)
            self.assertEqual(payload.get("schema"), "nautical.reconcile")
            self.assertEqual(payload.get("mode"), "dry-run")
            self.assertEqual(payload.get("applied"), [])
            self.assertNotIn("modify", log.read_text(encoding="utf-8"))

    def test_reconcile_apply_lock_contention_is_structured(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            taskdata = root / "taskdata"
            lock_dir = taskdata / ".nautical-locks"
            lock_dir.mkdir(parents=True)
            lock_path = lock_dir / ".nautical_reconcile.lock"
            task = root / "task"
            task.write_text("#!/bin/sh\nprintf '%s\\n' '[]'\n", encoding="utf-8")
            task.chmod(0o755)
            ready = root / "ready"
            release = root / "release"
            holder = root / "holder.py"
            holder.write_text(
                "import fcntl, pathlib, sys, time\n"
                "path = pathlib.Path(sys.argv[1])\n"
                "ready = pathlib.Path(sys.argv[2])\n"
                "release = pathlib.Path(sys.argv[3])\n"
                "with path.open('a+') as stream:\n"
                "    fcntl.flock(stream.fileno(), fcntl.LOCK_EX)\n"
                "    ready.touch()\n"
                "    while not release.exists(): time.sleep(0.01)\n",
                encoding="utf-8",
            )
            process = subprocess.Popen(
                [sys.executable, str(holder), str(lock_path), str(ready), str(release)],
                text=True,
            )
            try:
                deadline = time.monotonic() + 3.0
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), "lock holder did not start")
                result = self._run(
                    RECONCILE,
                    "--apply",
                    "--json",
                    "--task-bin",
                    str(task),
                    env={"TASKDATA": str(taskdata)},
                )
                self.assertEqual(result.stderr, "")
                self.assertNotEqual(result.returncode, 0)
                payload = self._json(result)
                self.assertEqual(payload.get("schema"), "nautical.reconcile")
                self.assertEqual(payload.get("stage"), "apply_lock")
                self.assertEqual(payload.get("mode"), "apply")
            finally:
                release.touch()
                process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
