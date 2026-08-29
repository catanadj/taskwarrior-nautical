"""Contract tests for extended performance telemetry comparisons."""

import unittest

from dev_tools.nautical_perf_compare import _metric_value


class PerformanceCompareTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
