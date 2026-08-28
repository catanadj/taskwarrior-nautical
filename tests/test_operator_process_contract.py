from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "nautical_core" / "tools" / "nautical_query.py"
DOCTOR = ROOT / "nautical_core" / "tools" / "nautical_doctor.py"


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

    def test_malformed_request_fails_with_json_and_exit_code(self) -> None:
        process = self._run(QUERY, "occurrences", "--request", "{not-json")
        self.assertEqual(process.returncode, 2)
        payload = self._json(process)
        self.assertEqual(payload.get("status"), "invalid")
        failure = payload.get("failure")
        self.assertIsInstance(failure, dict)
        self.assertEqual(failure.get("code"), "invalid_request")

    def test_empty_taskdata_integrity_is_unavailable_not_healthy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            environment = {"TASKDATA": directory, "TASKRC": str(Path(directory) / "taskrc")}
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


if __name__ == "__main__":
    unittest.main()
