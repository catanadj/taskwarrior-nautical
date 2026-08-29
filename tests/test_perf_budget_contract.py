"""Contract tests for independent performance-budget dimensions."""

from __future__ import annotations

import unittest
import ast
from pathlib import Path

from dev_tools import nautical_perf_budget as budget


class PerformanceBudgetContractTests(unittest.TestCase):
    def test_thin_hook_wrappers_do_not_import_heavy_stacks(self) -> None:
        forbidden = {"astral", "rich", "nautical_core.scheduler_service", "nautical_core.recurrence_evaluator"}
        root = Path(__file__).parents[1]
        for name in ("on-add.nautical", "on-modify.nautical", "on-exit.nautical"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            self.assertTrue(
                forbidden.isdisjoint(imported),
                f"{name} eagerly imports heavy modules: {sorted(forbidden & imported)}",
            )

    def test_budget_profiles_are_explicit_and_distinct(self) -> None:
        self.assertEqual(budget._budget_profile_name(slow_device=False), "desktop")
        self.assertEqual(budget._budget_profile_name(slow_device=True), "termux-slow-device")

    def test_measure_records_cpu_and_wall_attribution(self) -> None:
        result = budget._measure("contract", lambda: 0.001, 2)
        self.assertEqual(result["name"], "contract")
        self.assertEqual(len(result["samples_s"]), 2)
        self.assertEqual(len(result["cpu_samples_s"]), 2)
        self.assertGreaterEqual(result["cpu_median_s"], 0.0)
        self.assertGreaterEqual(result["measured_wall_median_s"], 0.0)
        self.assertGreaterEqual(result["peak_memory_median_bytes"], 0)

    def test_measure_does_not_hide_correctness_failures(self) -> None:
        def broken_check() -> float:
            raise RuntimeError("synthetic correctness failure")

        with self.assertRaisesRegex(RuntimeError, "synthetic correctness failure"):
            budget._measure("broken", broken_check, 1)

    def test_workflow_timing_breakdown_keeps_component_attribution(self) -> None:
        result = {}
        budget._attach_timing_breakdown(
            result,
            [1.0],
            [{"run_task_seconds": 0.4, "startup_total_ms": 100.0, "drain_ms": 200.0, "presentation_ms": 50.0}],
        )
        breakdown = result["timing_breakdown"][0]
        self.assertEqual(breakdown["taskwarrior_seconds"], 0.4)
        self.assertEqual(breakdown["startup_seconds"], 0.1)
        self.assertEqual(breakdown["drain_seconds"], 0.2)
        self.assertEqual(breakdown["presentation_seconds"], 0.05)
        self.assertEqual(breakdown["non_taskwarrior_seconds"], 0.6)

    def test_capabilities_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_capabilities_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_queue_status_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_queue_status_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_navigator_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_navigator_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_query_pagination_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_query_pagination_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_query_unavailable_stage_fails_closed(self) -> None:
        elapsed = budget._bench_query_unavailable_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_doctor_installation_stage_has_a_json_guard(self) -> None:
        if budget.shutil.which("task") is None:
            self.skipTest("Taskwarrior is not installed")
        elapsed = budget._bench_doctor_installation_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_housekeeping_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_housekeeping_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_repair_planner_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_repair_planner_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_repair_application_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_repair_application_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_queue_stale_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_queue_stale_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_operator_failure_matrix_is_fail_closed(self) -> None:
        elapsed = budget._bench_operator_failure_matrix_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_operator_interrupted_stage_reclaims_work(self) -> None:
        elapsed = budget._bench_operator_interrupted_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_exit_probe_fast_paths_are_empty(self) -> None:
        elapsed = budget._bench_exit_probe_fast_paths_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_operator_scope_matrix_has_explicit_boundaries(self) -> None:
        elapsed = budget._bench_operator_scope_matrix_stage()
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

    def test_reconcile_budgets_are_independent_from_wall_time(self) -> None:
        result = {"pass": True, "reconcile_reports": [
            {"export_calls": 3, "export_rows": 12, "task_command_calls": 7, "task_command_attempts": 8}
        ]}
        budget._apply_reconcile_budgets(result, {"export_calls": 2, "task_command_calls": 8})
        self.assertFalse(result["pass"])
        self.assertFalse(result["reconcile_budget"]["export_calls"]["pass"])
        self.assertTrue(result["reconcile_budget"]["task_command_calls"]["pass"])


if __name__ == "__main__":
    unittest.main()
