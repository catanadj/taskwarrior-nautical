"""Contract tests for independent performance-budget dimensions."""

from __future__ import annotations

import unittest

from dev_tools import nautical_perf_budget as budget


class PerformanceBudgetContractTests(unittest.TestCase):
    def test_measure_records_cpu_and_wall_attribution(self) -> None:
        result = budget._measure("contract", lambda: 0.001, 2)
        self.assertEqual(result["name"], "contract")
        self.assertEqual(len(result["samples_s"]), 2)
        self.assertEqual(len(result["cpu_samples_s"]), 2)
        self.assertGreaterEqual(result["cpu_median_s"], 0.0)
        self.assertGreaterEqual(result["measured_wall_median_s"], 0.0)

    def test_capabilities_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_capabilities_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_queue_status_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_queue_status_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_navigator_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_navigator_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_task_call_budget_is_independent_from_wall_time(self) -> None:
        result = {"pass": True}
        budget._apply_task_call_budgets(
            result,
            [{"run_task_calls": 12, "run_task_calls_purpose_read": 4}],
            {"run_task_calls": 10, "run_task_calls_purpose_read": 4},
        )
        self.assertFalse(result["pass"])
        self.assertFalse(result["task_call_budget"]["run_task_calls"]["pass"])
        self.assertTrue(result["task_call_budget"]["run_task_calls_purpose_read"]["pass"])

    def test_component_and_sqlite_budgets_are_independent(self) -> None:
        component = {"pass": True}
        budget._apply_component_budgets(
            component,
            [{"startup_seconds": 2.0, "drain_seconds": 0.5}],
            {"startup_seconds": 1.0, "drain_seconds": 1.0},
        )
        self.assertFalse(component["pass"])
        self.assertFalse(component["component_budget"]["startup_seconds"]["pass"])
        self.assertTrue(component["component_budget"]["drain_seconds"]["pass"])

        sqlite = {"pass": True}
        budget._apply_outbox_budgets(
            sqlite,
            [{"outbox_lock_failures": 1.0}],
            {"outbox_lock_failures": 0.0},
        )
        self.assertFalse(sqlite["pass"])
        self.assertFalse(sqlite["sqlite_budget"]["outbox_lock_failures"]["pass"])


if __name__ == "__main__":
    unittest.main()
