"""Contract tests for independent performance-budget dimensions."""

from __future__ import annotations

import unittest
import ast
import json
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from dev_tools import nautical_perf_budget as budget
from dev_tools.perf import reporting
from nautical_core.exit_probe import probe_exit_work
import nautical_core


class PerformanceBudgetContractTests(unittest.TestCase):
    def test_extracted_reporting_helpers_preserve_schema(self) -> None:
        result: dict = {}
        reporting.attach_timing_breakdown(
            result,
            [1.0],
            [{"run_task_seconds": 0.4, "startup_total_ms": 100.0}],
        )
        self.assertEqual(result["timing_breakdown"][0]["taskwarrior_seconds"], 0.4)
        self.assertEqual(reporting.merge_task_timing_stats({"run_task_seconds": 1.0}, {"run_task_seconds": 2.0}), {"run_task_seconds": 3.0})
        self.assertEqual(
            reporting.compact_reconcile_report({"status": "ok", "ignored": True}),
            {"status": "ok"},
        )

    def test_workflow_workload_module_is_import_safe(self) -> None:
        from dev_tools.perf import workflow_workloads

        self.assertTrue(callable(workflow_workloads.expensive_workflows))
        self.assertTrue(callable(workflow_workloads.integrity_scale))
        self.assertTrue(callable(workflow_workloads.ordinary_modify))
        self.assertTrue(callable(workflow_workloads.expiration_recovery))
        self.assertTrue(callable(workflow_workloads.completion_workflows))
        self.assertTrue(callable(workflow_workloads.queue_preflight))
        self.assertTrue(callable(workflow_workloads.queue_healthy_replay))
        self.assertTrue(callable(workflow_workloads.queue_replay_verify))
        self.assertTrue(callable(workflow_workloads.queue_partial_recovery))
        self.assertTrue(callable(workflow_workloads.queue_shape))
        self.assertTrue(callable(workflow_workloads.reconcile_history_fixture))
        self.assertTrue(callable(workflow_workloads.reconcile_healthy))
        self.assertTrue(callable(workflow_workloads.reconcile_empty))
        self.assertTrue(callable(workflow_workloads.reconcile_candidates))
        self.assertTrue(callable(workflow_workloads.reconcile_candidates_apply))
        self.assertTrue(callable(workflow_workloads.reconcile_candidates_apply_scale))
        self.assertTrue(callable(workflow_workloads.reconcile_report_loop))
        self.assertTrue(callable(workflow_workloads.run_scenarios))
        self.assertTrue(callable(workflow_workloads.workflow_fixture))
        self.assertTrue(hasattr(workflow_workloads, "WorkflowContext"))

    def test_budget_cli_help_is_a_stable_subprocess_contract(self) -> None:
        budget_script = Path(__file__).parents[1] / "dev_tools" / "nautical_perf_budget.py"
        proc = subprocess.run(
            [sys.executable, str(budget_script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0)
        self.assertIn("usage:", proc.stdout)
        for option in ("--budget-file", "--json", "--enforce", "--extended", "--workflows-only"):
            self.assertIn(option, proc.stdout)
        self.assertNotIn("Traceback", proc.stderr)

    def test_budget_manifest_covers_cache_seasonal_hook_and_workflow_paths(self) -> None:
        manifest = json.loads((budget.ROOT / "dev_tools" / "perf_budget.json").read_text(encoding="utf-8"))
        budgets = manifest["budgets_seconds"]
        workload = manifest["workload"]
        self.assertTrue({
            "cache_save", "cache_load_hot", "build_hints_cold", "build_hints_warm",
            "outbox_schema_hot", "outbox_schema_cold", "seasonal_parse_validate",
            "seasonal_next_after", "seasonal_build_hints_cold", "seasonal_build_hints_warm",
        } <= set(budgets))
        self.assertTrue({
            "cache_save_rounds", "cache_load_rounds", "outbox_schema_hot_rounds",
            "outbox_schema_cold_rounds",
        } <= set(workload))
        self.assertTrue({"y:d60,d-1", "y:w20 + w:mon"} <= set(workload["expressions"]))
        self.assertTrue({
            "(w:mon)@in-spring=first,last@t=09:00,17:00",
            "(y:02-29)@in-winter=first",
        } <= set(manifest["seasonal_workload"]["expressions"]))

        slow = manifest["slow_device_budgets_seconds"]
        self.assertTrue({
            "build_hints_cold", "build_hints_warm", "seasonal_build_hints_cold",
            "seasonal_build_hints_warm",
        } <= set(slow))
        hook = manifest["hook_fast_path"]
        self.assertGreaterEqual(hook["repeats"], 3)
        self.assertGreaterEqual(hook["managed_layout_max_ratio"], 1.0)
        self.assertGreaterEqual(hook["staged_layout_max_ratio"], 1.0)
        self.assertTrue({
            "hook_plain_add", "hook_plain_modify", "hook_nautical_ordinary_modify", "hook_empty_exit",
        } <= set(hook["max_ratio"]))
        workflow = manifest["workflow_perf"]
        self.assertTrue({
            "workflow_cp_completion", "workflow_cp_completion_nonfinal",
            "workflow_cp_completion_nonfinal_idempotent", "workflow_anchor_completion",
            "workflow_anchor_completion_nonfinal", "workflow_anchor_completion_nonfinal_idempotent",
            "workflow_queue_drain", "workflow_reconcile",
        } <= set(workflow["budgets_seconds"]))
        self.assertTrue({
            "workflow_queue_drain", "workflow_queue_drain_partial_recovery",
        } <= set(workflow["slow_device_budgets_seconds"]))
        extended = manifest["extended_workload"]
        self.assertTrue({
            "anchor_file_large_cold", "anchor_file_large_hot", "anchor_file_nonmonotonic",
            "anchor_file_business_day_omissions", "business_calendar_large_omissions",
            "native_until_reconcile_dry_run", "native_until_reconcile_apply",
        } <= set(extended["budgets_seconds"]))
        self.assertIsInstance(extended["slow_device_budgets_seconds"], dict)

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

    def test_python_subprocess_env_exposes_checkout_package(self) -> None:
        env = budget._python_subprocess_env({"PYTHONPATH": "/tmp/existing"})
        self.assertEqual(env["PYTHONPATH"].split(budget.os.pathsep)[0], str(budget.ROOT))
        self.assertIn("/tmp/existing", env["PYTHONPATH"])

    def test_reconcile_candidates_match_production_chain_identity(self) -> None:
        tasks = budget._reconcile_candidate_tasks("contract", 3)
        for task in tasks:
            self.assertEqual(task["chainID"], task["uuid"][:8])
        legacy = budget._reconcile_candidate_tasks("contract", 1, legacy_chain_ids=True)[0]
        self.assertNotEqual(legacy["chainID"], legacy["uuid"][:8])

    def test_measure_records_cpu_and_wall_attribution(self) -> None:
        result = budget._measure("contract", lambda: 0.001, 2)
        self.assertEqual(result["name"], "contract")
        self.assertEqual(len(result["samples_s"]), 2)
        self.assertEqual(len(result["cpu_samples_s"]), 2)
        self.assertGreaterEqual(result["cpu_median_s"], 0.0)
        self.assertGreaterEqual(result["measured_wall_median_s"], 0.0)
        self.assertGreaterEqual(result["peak_memory_median_bytes"], 0)
        self.assertFalse(result["memory_tracing"])
        traced = budget._measure("contract-traced", lambda: 0.001, 1, trace_memory=True)
        self.assertTrue(traced["memory_tracing"])

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

    def test_performance_result_exposes_all_cost_attribution_dimensions(self) -> None:
        result = budget._measure("operator", lambda: 0.001, 1)
        for key in ("measured_wall_median_s", "cpu_median_s", "peak_memory_median_bytes"):
            self.assertIn(key, result)
        breakdown = {}
        budget._attach_timing_breakdown(
            breakdown,
            [0.01],
            [{"run_task_seconds": 0.002, "startup_total_ms": 1.0, "drain_ms": 3.0, "presentation_ms": 1.0}],
        )
        dimensions = breakdown["timing_breakdown"][0]
        for key in ("taskwarrior_seconds", "startup_seconds", "drain_seconds", "presentation_seconds", "non_taskwarrior_seconds"):
            self.assertIn(key, dimensions)

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

    def test_lifecycle_staging_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_lifecycle_staging_stage()
        self.assertGreaterEqual(elapsed, 0.0)

    def test_reconcile_snapshot_stage_has_a_correctness_guard(self) -> None:
        elapsed = budget._bench_reconcile_snapshot_stage()
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

    def test_exit_probe_fails_closed_for_unavailable_taskdata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            missing = Path(td) / "missing"
            result = probe_exit_work(missing)
            self.assertFalse(result.definitely_empty)
            self.assertIn("unavailable", result.reason)

    def test_exit_probe_bypasses_only_known_empty_outboxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(probe_exit_work(root).definitely_empty)

            state_dir = root / ".nautical-state"
            state_dir.mkdir()
            database = state_dir / ".nautical_lifecycle_outbox.db"
            with sqlite3.connect(database) as connection:
                connection.execute(
                    "CREATE TABLE lifecycle_outbox (intent_id TEXT PRIMARY KEY, processing_state TEXT NOT NULL)"
                )
                connection.execute("PRAGMA user_version = 1")
            self.assertTrue(probe_exit_work(root).definitely_empty)

            with sqlite3.connect(database) as connection:
                connection.execute("PRAGMA user_version = 3")
            self.assertTrue(probe_exit_work(root).may_have_work)

            with sqlite3.connect(database) as connection:
                connection.execute("PRAGMA user_version = 2")
                connection.execute("INSERT INTO lifecycle_outbox VALUES ('intent-1', 'ready')")
            self.assertTrue(probe_exit_work(root).may_have_work)

            with sqlite3.connect(database) as connection:
                connection.execute("UPDATE lifecycle_outbox SET processing_state='claimed'")
            self.assertTrue(probe_exit_work(root).may_have_work)

            database.write_bytes(b"not-a-sqlite-database")
            self.assertTrue(probe_exit_work(root).may_have_work)

    def test_lazy_facade_defaults_do_not_share_configuration_tables(self) -> None:
        self.assertIsNot(nautical_core.ANCHOR_PRESETS, nautical_core.OMIT_PRESETS)
        self.assertIsNot(nautical_core.BUSINESS_CALENDAR_CONFIG, nautical_core.ASTRONOMY_CONFIG)

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
