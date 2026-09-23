"""Contract tests for extended performance telemetry comparisons."""

import unittest
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from dev_tools.nautical_perf_compare import _compare_workload_contract, _metric_value


class PerformanceCompareTests(unittest.TestCase):
    def test_workload_contract_comparison_detects_coverage_and_decision_changes(self) -> None:
        base = {"results": {"stable": {"pass": True}, "removed": {"pass": False}}}
        head = {"results": {"stable": {"pass": False}, "added": {"pass": True}}}

        contract = _compare_workload_contract(base, head)

        self.assertEqual(contract["missing"], ["removed"])
        self.assertEqual(contract["added"], ["added"])
        self.assertEqual(contract["decision_changes"], ["stable"])
        self.assertFalse(contract["ok"])

    def test_workload_contract_rejects_malformed_results(self) -> None:
        report = {"results": {"broken": {"median_s": 0.1}}}

        contract = _compare_workload_contract(report, report)

        self.assertFalse(contract["ok"])
        self.assertEqual(contract["invalid_base"], ["broken"])
        self.assertEqual(contract["invalid_head"], ["broken"])

    def test_help_is_a_valid_argparse_contract(self) -> None:
        compare = Path(__file__).parents[1] / "dev_tools" / "nautical_perf_compare.py"
        proc = subprocess.run(
            [sys.executable, str(compare), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("usage:", proc.stdout)
        self.assertIn("15%", proc.stdout)
        self.assertIn("--contract-only", proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_extended_metrics_extract_from_stage_and_workflow_reports(self) -> None:
        result = {
            "cpu_median_s": 0.2,
            "peak_memory_median_bytes": 512,
            "timing_breakdown": [{
                "taskwarrior_seconds": 0.1,
                "startup_seconds": 0.02,
                "drain_seconds": 0.03,
                "presentation_seconds": 0.04,
            }],
        }
        self.assertEqual(_metric_value(result, "cpu_time"), 0.2)
        self.assertEqual(_metric_value(result, "peak_memory"), 512.0)
        self.assertEqual(_metric_value(result, "taskwarrior_time"), 0.1)
        self.assertEqual(_metric_value(result, "presentation_time"), 0.04)

    def test_compare_enforces_extended_operator_safety_ceiling(self) -> None:
        base = {"results": {"stage_operator_failure_matrix": {
            "median_s": 0.01, "cpu_median_s": 0.01,
            "peak_memory_median_bytes": 100,
        }}}
        head = {"results": {"stage_operator_failure_matrix": {
            "median_s": 0.011, "cpu_median_s": 0.011,
            "peak_memory_median_bytes": 200,
        }}}
        compare = Path(__file__).parents[1] / "dev_tools" / "nautical_perf_compare.py"
        with TemporaryDirectory(prefix="nautical-compare-") as td:
            base_path = Path(td) / "base.json"
            head_path = Path(td) / "head.json"
            base_path.write_text(json.dumps(base), encoding="utf-8")
            head_path.write_text(json.dumps(head), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(compare), "--base", str(base_path), "--head", str(head_path), "--enforce", "--json",
                ], capture_output=True, text=True, check=False,
            )
        self.assertEqual(proc.returncode, 1)
        payload = json.loads(proc.stdout)
        self.assertIn("stage_operator_failure_matrix:peak_memory", payload["metric_regressions"])


if __name__ == "__main__":
    unittest.main()
