from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap
import unittest

import coverage

from tests.support.hook_process import HookSubprocessFixture


class HookProcessContractTests(HookSubprocessFixture):
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
