"""Taskwarrior workflow workload boundary.

The workflow harness is intentionally exposed through this import-safe boundary
so the CLI can depend on a stable workload-owned entry point while the legacy
fixture-heavy implementation is migrated in smaller slices.
"""

from __future__ import annotations

import time
import uuid
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


class WorkflowDependencies:
    """Explicit runtime services used by the fixture-heavy workflow runner.

    The benchmark scenarios remain pure workload code; the CLI supplies this
    small dependency bundle so importing and testing the workload module does
    not execute CLI setup or rely on hidden module globals.
    """

    def __init__(self, **services: Any) -> None:
        self.__dict__.update(services)


def expensive_workflows(
    runner: Callable[..., dict[str, dict]],
    cfg: dict,
    *,
    slow_device: bool = False,
    panel_mode: str = "minimal",
) -> dict[str, dict]:
    """Run the workflow group through its injected implementation boundary."""
    return runner(cfg, slow_device=slow_device, panel_mode=panel_mode)


def run_scenarios(
    deps: WorkflowDependencies,
    cfg: dict,
    *,
    slow_device: bool = False,
    panel_mode: str = "minimal",
) -> dict[str, dict]:
    """Exercise completion, queue-drain, and reconcile paths in isolation."""
    workflow_cfg = cfg.get("workflow_perf")
    if not isinstance(workflow_cfg, dict) or not workflow_cfg.get("enabled", True):
        return {}
    workflow_cfg = dict(workflow_cfg)
    budgets_override = workflow_cfg.get("slow_device_budgets_seconds") if slow_device else None
    if isinstance(budgets_override, dict):
        budgets = dict(workflow_cfg.get("budgets_seconds") or {})
        budgets.update(budgets_override)
        workflow_cfg["budgets_seconds"] = budgets
    repeats = max(1, int(workflow_cfg.get("repeats", 3)))
    budgets = workflow_cfg.get("budgets_seconds") if isinstance(workflow_cfg.get("budgets_seconds"), dict) else {}
    reconcile_call_purposes: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="nautical-workflow-perf-") as td:
        root = Path(td)
        real_task = shutil.which("task")
        if not real_task:
            raise RuntimeError("Taskwarrior executable was not found for workflow benchmark")
        task_wrapper = root / "task-benchmark-wrapper.py"
        task_wrapper.write_text(
            "#!/usr/bin/env python3\n"
            "import os, subprocess, sys\n"
            f"real = {real_task!r}\n"
            "args = sys.argv[1:]\n"
            "mode = os.environ.get('NAUTICAL_BENCH_FAIL_MODE', '')\n"
            "if mode == 'fail-export' and 'export' in args:\n"
            "    sys.stderr.write('database is locked\\n')\n"
            "    raise SystemExit(1)\n"
            "if mode == 'partial-import' and 'import' in args:\n"
            "    lines = sys.stdin.read().splitlines()\n"
            "    if lines:\n"
            "        done = subprocess.run([real, *args], input=lines[0] + '\\n', text=True, capture_output=True)\n"
            "        sys.stdout.write(done.stdout or '')\n"
            "        sys.stderr.write(done.stderr or '')\n"
            "        if done.returncode:\n"
            "            raise SystemExit(done.returncode)\n"
            "    sys.stderr.write('database is locked after partial import\\n')\n"
            "    raise SystemExit(1)\n"
            "raise SystemExit(subprocess.run([real, *args], text=True).returncode)\n",
            encoding="utf-8",
        )
        task_wrapper.chmod(0o700)
        config_path = root / "config-nautical.toml"
        config_path.write_text(
            f'tz = "UTC"\npanel_mode = "{deps._panel_mode_config(panel_mode)}"\n',
            encoding="utf-8",
        )
        taskrc_path = root / "taskrc"
        taskrc_path.write_text(
            "uda.chainID.type=string\n"
            "uda.chain.type=string\n"
            "uda.link.type=numeric\n"
            "uda.prevLink.type=string\n"
            "uda.nextLink.type=string\n"
            "uda.cp.type=string\n"
            "uda.anchor.type=string\n"
            "uda.anchor_mode.type=string\n",
            encoding="utf-8",
        )
        base_env = deps._python_subprocess_env()
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(deps.ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "NAUTICAL_TRUST_TASKDATA_PATH": "1",
                "TASKRC": str(taskrc_path),
                "TZ": "UTC",
            }
        )
        fingerprint_taskdata = root / "fingerprint-probe"
        fingerprint_taskdata.mkdir()
        fingerprint_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os, nautical_core as core; "
                    "core.reload_taskdata_config(os.environ['TASKDATA']); "
                    "print(json.dumps({'configuration': core.effective_config_fingerprint(), "
                    "'schedule': core.scheduler_config_fingerprint()}))"
                ),
            ],
            text=True,
            capture_output=True,
            env=dict(base_env, TASKDATA=str(fingerprint_taskdata)),
            timeout=30.0,
        )
        if fingerprint_probe.returncode != 0:
            raise RuntimeError(
                "workflow fingerprint probe failed: "
                f"{(fingerprint_probe.stderr or fingerprint_probe.stdout or '').strip()}"
            )
        try:
            workflow_fingerprints = json.loads(fingerprint_probe.stdout or "{}")
            config_fingerprint = str(workflow_fingerprints["configuration"])
            schedule_fingerprint = str(workflow_fingerprints["schedule"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("workflow fingerprint probe returned invalid JSON") from exc
        for key in ("NAUTICAL_DIAG", "NAUTICAL_DIAG_LOG", "NAUTICAL_PROFILE"):
            base_env.pop(key, None)

        completion_cases = (
            ("workflow_cp_completion", "cp", False),
            ("workflow_cp_completion_nonfinal", "cp", True),
            ("workflow_anchor_completion", "anchor", False),
            ("workflow_anchor_completion_nonfinal", "anchor", True),
        )
        results: dict[str, dict] = {}

        scale_counts = tuple(int(item) for item in workflow_cfg.get("integrity_scale_counts", (100, 1000, 10000)))
        results.update(deps.workloads.integrity_scale(scale_counts, budgets, deps._measure_workflow))

        results["workflow_ordinary_modify"] = deps.workloads.ordinary_modify(
            root=deps.ROOT,
            workflow_root=root,
            base_env=base_env,
            repeats=repeats,
            completion_fixture=deps._completion_fixture,
            import_workflow_rows=deps._import_workflow_rows,
            run_workflow_hook_result=deps._run_workflow_hook_result,
            workflow_outbox_pending=deps._workflow_outbox_pending,
            measure_workflow=deps._measure_workflow,
            budgets=budgets,
        )

        results["workflow_expiration_recovery"] = deps.workloads.expiration_recovery(
            root=deps.ROOT,
            workflow_root=root,
            base_env=base_env,
            repeats=repeats,
            import_workflow_rows=deps._import_workflow_rows,
            run_workflow_hook_result=deps._run_workflow_hook_result,
            workflow_outbox_pending=deps._workflow_outbox_pending,
            measure_workflow=deps._measure_workflow,
            budgets=budgets,
        )

        results.update(deps.workloads.completion_workflows(
            completion_cases=completion_cases,
            root=deps.ROOT,
            workflow_root=root,
            base_env=base_env,
            repeats=repeats,
            budgets=budgets,
            workflow_cfg=workflow_cfg,
            completion_fixture=deps._completion_fixture,
            import_workflow_rows=deps._import_workflow_rows,
            import_existing_completion_child=deps._import_existing_completion_child,
            run_workflow_hook_result=deps._run_workflow_hook_result,
            workflow_outbox_pending=deps._workflow_outbox_pending,
            read_exit_task_call_stats=deps._read_exit_task_call_stats,
            measure_workflow=deps._measure_workflow,
            apply_task_call_budgets=deps._apply_task_call_budgets,
        ))

        queue_samples = []
        queue_call_stats: list[dict[str, int]] = []
        queue_idempotent_samples = []
        queue_idempotent_call_stats: list[dict[str, int]] = []
        queue_partial_samples = []
        queue_partial_first_samples = []
        queue_partial_recovery_samples = []
        queue_partial_call_stats: list[dict[str, int]] = []
        queue_timing_stats: list[dict[str, float]] = []
        queue_idempotent_timing_stats: list[dict[str, float]] = []
        queue_partial_timing_stats: list[dict[str, float]] = []
        queue_outbox_stats: list[dict[str, float]] = []
        queue_idempotent_outbox_stats: list[dict[str, float]] = []
        queue_partial_outbox_stats: list[dict[str, float]] = []
        for sample_index in range(repeats):
            queue_data = root / f"populated-queue-{sample_index}"
            deps._init_empty_outbox(queue_data)
            stats_path = queue_data / "on-exit-task-stats.json"
            queue_env = dict(base_env, TASKDATA=str(queue_data), NAUTICAL_BENCH_STATS_FILE=str(stats_path))
            # Healthy drains must measure the real Taskwarrior process.  The
            # wrapper is reserved for the partial-import failure injection
            # below; otherwise every command would include an avoidable Python
            # process and obscure the queue's actual cost.
            queue_env["NAUTICAL_BENCH_TASK_BIN"] = str(real_task)
            if os.environ.get("NAUTICAL_DIAG") == "1":
                queue_env["NAUTICAL_DIAG"] = "1"
            parents, queue_plans, parent_uuids, child_uuids = deps.workloads.queue_preflight(
                task_bin="task", queue_data=queue_data, queue_env=queue_env,
                sample_index=sample_index,
                outbox_lifecycle_fixture=deps._outbox_lifecycle_fixture,
                bind_workflow_plans=deps._bind_workflow_plans_to_parents,
                stage_workflow_plans=deps._stage_workflow_plans,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
            )

            queue_elapsed, _queue_result, _queue_stderr = deps._run_workflow_hook_result(
                deps.ROOT / "on-exit.nautical",
                input_text="",
                env=queue_env,
                expect_output=False,
            )
            queue_samples.append(queue_elapsed)
            queue_call_stats.append(deps._read_exit_task_call_stats(stats_path))
            child_slot_reads = queue_call_stats[-1].get("run_task_calls_purpose_task_read_child_slot", 0)
            if child_slot_reads > 1:
                raise RuntimeError(
                    "queue drain preflight regressed to per-candidate child-slot reads: "
                    f"{child_slot_reads} subprocesses for {len(queue_plans)} candidates"
                )
            queue_timing_stats.append(deps._read_exit_task_timing_stats(stats_path))
            queue_outbox_stats.append(deps._read_exit_outbox_stats(stats_path))

            if deps._workflow_outbox_pending(queue_data):
                raise RuntimeError(
                    "outbox drain benchmark left active intents after successful processing: "
                    f"{deps._workflow_outbox_pending(queue_data)!r}; stderr={_queue_stderr.strip()!r}"
                )

            # Acknowledged outbox records are terminal and must not trigger
            # another Taskwarrior read or mutation on a replay drain.
            idem_elapsed, idem_calls, idem_timing, idem_outbox = deps.workloads.queue_replay_verify(
                root=deps.ROOT,
                queue_data=queue_data,
                queue_env=queue_env,
                stats_path=stats_path,
                parent_uuids=parent_uuids,
                child_uuids=child_uuids,
                run_workflow_hook_result=deps._run_workflow_hook_result,
                read_exit_task_call_stats=deps._read_exit_task_call_stats,
                read_exit_task_timing_stats=deps._read_exit_task_timing_stats,
                read_exit_outbox_stats=deps._read_exit_outbox_stats,
                workflow_outbox_pending=deps._workflow_outbox_pending,
                lifecycle_outbox=deps.lifecycle_outbox,
            )
            queue_idempotent_samples.append(idem_elapsed)
            queue_idempotent_call_stats.append(idem_calls)
            queue_idempotent_timing_stats.append(idem_timing)
            queue_idempotent_outbox_stats.append(idem_outbox)
            export_proc = subprocess.run(
                [
                    "task",
                    "rc.hooks=off",
                    "rc.json.array=1",
                    "rc.verbose=nothing",
                    "rc.color=off",
                    "chain:on",
                    "export",
                ],
                text=True,
                capture_output=True,
                env=queue_env,
                timeout=30.0,
            )
            if export_proc.returncode != 0:
                raise RuntimeError(
                    "queue drain benchmark export failed: "
                    f"{(export_proc.stderr or export_proc.stdout or '').strip()}"
                )
            try:
                exported = json.loads(export_proc.stdout or "[]")
            except json.JSONDecodeError as exc:
                raise RuntimeError("queue drain benchmark export was not valid JSON") from exc
            if not isinstance(exported, list) or len(exported) != 16:
                raise RuntimeError(
                    "queue drain benchmark did not retain 8 parents and import 8 children: "
                    f"{len(exported) if isinstance(exported, list) else type(exported).__name__} tasks; "
                    f"outbox={deps.lifecycle_outbox._LifecycleOutboxRepository(queue_data).status(limit=20)[1]!r}"
                )
            children = [
                row
                for row in exported
                if isinstance(row, dict) and str(row.get("uuid") or "") in child_uuids
            ]
            if len(children) != 8 or any(not str(row.get("prevLink") or "").strip() for row in children):
                raise RuntimeError("queue drain benchmark did not import/link all child tasks")
            parents_after = [
                row
                for row in exported
                if isinstance(row, dict) and str(row.get("uuid") or "") in parent_uuids
            ]
            if len(parents_after) != 8 or any(not str(row.get("nextLink") or "").strip() for row in parents_after):
                raise RuntimeError("queue drain benchmark did not update all parent nextLink values")

            first_partial_elapsed, recovery_elapsed, merged_calls, merged_timing, merged_outbox, first_calls, first_timing, first_outbox = deps.workloads.queue_partial_recovery(
                root=deps.ROOT,
                partial_data=root / f"partial-queue-{sample_index}",
                partial_env=dict(base_env, TASKDATA=str(root / f"partial-queue-{sample_index}"), NAUTICAL_BENCH_STATS_FILE=str(root / f"partial-queue-{sample_index}" / "on-exit-task-stats.json"), NAUTICAL_BENCH_TASK_BIN=str(task_wrapper)),
                partial_stats_path=root / f"partial-queue-{sample_index}" / "on-exit-task-stats.json",
                task_wrapper=task_wrapper,
                sample_index=sample_index,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
                outbox_lifecycle_fixture=deps._outbox_lifecycle_fixture,
                init_empty_outbox=deps._init_empty_outbox,
                bind_workflow_plans=deps._bind_workflow_plans_to_parents,
                stage_workflow_plans=deps._stage_workflow_plans,
                run_workflow_hook_result=deps._run_workflow_hook_result,
                read_exit_task_call_stats=deps._read_exit_task_call_stats,
                read_exit_task_timing_stats=deps._read_exit_task_timing_stats,
                read_exit_outbox_stats=deps._read_exit_outbox_stats,
                workflow_outbox_pending=deps._workflow_outbox_pending,
                merge_task_call_stats=deps._merge_task_call_stats,
                merge_task_timing_stats=deps._merge_task_timing_stats,
            )
            queue_partial_first_samples.append(first_partial_elapsed)
            queue_partial_recovery_samples.append(recovery_elapsed)
            queue_partial_samples.append(first_partial_elapsed + recovery_elapsed)
            queue_partial_call_stats.append(merged_calls)
            queue_partial_timing_stats.append(merged_timing)
            queue_partial_outbox_stats.append(merged_outbox)
            continue

            continue

        queue_result = deps._measure_workflow(
            "workflow_queue_drain", queue_samples,
            float(budgets.get("workflow_queue_drain", 3.0)),
        )
        queue_result["task_call_stats"] = queue_call_stats
        queue_result["outbox_stats"] = queue_outbox_stats
        deps._attach_timing_breakdown(queue_result, queue_samples, queue_timing_stats)
        sqlite_budgets = workflow_cfg.get("sqlite_budgets")
        if isinstance(sqlite_budgets, dict):
            deps._apply_outbox_budgets(queue_result, queue_outbox_stats, sqlite_budgets.get("workflow_queue_drain", {}))
        component_budgets = workflow_cfg.get("component_budgets_seconds")
        if isinstance(component_budgets, dict):
            deps._apply_component_budgets(
                queue_result, queue_result["timing_breakdown"],
                component_budgets.get("workflow_queue_drain", {}),
            )
        call_budgets = workflow_cfg.get("task_call_budgets")
        if isinstance(call_budgets, dict):
            deps._apply_task_call_budgets(
                queue_result,
                queue_call_stats,
                call_budgets.get("workflow_queue_drain", {}),
            )
        results["workflow_queue_drain"] = queue_result
        queue_idempotent_result = deps._measure_workflow(
            "workflow_queue_drain_idempotent",
            queue_idempotent_samples,
            float(budgets.get("workflow_queue_drain_idempotent", 3.0)),
        )
        queue_idempotent_result["task_call_stats"] = queue_idempotent_call_stats
        queue_idempotent_result["outbox_stats"] = queue_idempotent_outbox_stats
        deps._attach_timing_breakdown(queue_idempotent_result, queue_idempotent_samples, queue_idempotent_timing_stats)
        if isinstance(sqlite_budgets, dict):
            deps._apply_outbox_budgets(queue_idempotent_result, queue_idempotent_outbox_stats, sqlite_budgets.get("workflow_queue_drain_idempotent", {}))
        if isinstance(component_budgets, dict):
            deps._apply_component_budgets(
                queue_idempotent_result, queue_idempotent_result["timing_breakdown"],
                component_budgets.get("workflow_queue_drain_idempotent", {}),
            )
        if isinstance(call_budgets, dict):
            deps._apply_task_call_budgets(
                queue_idempotent_result,
                queue_idempotent_call_stats,
                call_budgets.get("workflow_queue_drain_idempotent", {}),
            )
        results["workflow_queue_drain_idempotent"] = queue_idempotent_result
        queue_partial_result = deps._measure_workflow(
            "workflow_queue_drain_partial_recovery",
            queue_partial_samples,
            float(budgets.get("workflow_queue_drain_partial_recovery", 6.0)),
        )
        queue_partial_result["first_attempt_samples_s"] = sorted(queue_partial_first_samples)
        queue_partial_result["first_attempt_median_s"] = float(statistics.median(queue_partial_first_samples))
        queue_partial_result["recovery_samples_s"] = sorted(queue_partial_recovery_samples)
        queue_partial_result["recovery_median_s"] = float(statistics.median(queue_partial_recovery_samples))
        queue_partial_result["task_call_stats"] = queue_partial_call_stats
        queue_partial_result["outbox_stats"] = queue_partial_outbox_stats
        deps._attach_timing_breakdown(queue_partial_result, queue_partial_samples, queue_partial_timing_stats)
        if isinstance(sqlite_budgets, dict):
            deps._apply_outbox_budgets(queue_partial_result, queue_partial_outbox_stats, sqlite_budgets.get("workflow_queue_drain_partial_recovery", {}))
        if isinstance(component_budgets, dict):
            deps._apply_component_budgets(
                queue_partial_result, queue_partial_result["timing_breakdown"],
                component_budgets.get("workflow_queue_drain_partial_recovery", {}),
            )
        if isinstance(call_budgets, dict):
            deps._apply_task_call_budgets(
                queue_partial_result,
                queue_partial_call_stats,
                call_budgets.get("workflow_queue_drain_partial_recovery", {}),
            )
        results["workflow_queue_drain_partial_recovery"] = queue_partial_result

        results["workflow_queue_drain_one_intent"] = deps.workloads.queue_shape(
            name="queue-one-intent", background_rows=0, root=deps.ROOT, base_env=base_env,
            real_task=real_task, slow_device=slow_device,
            config_fingerprint=config_fingerprint, schedule_fingerprint=schedule_fingerprint,
            budgets=budgets, init_empty_outbox=deps._init_empty_outbox,
            outbox_lifecycle_fixture=deps._outbox_lifecycle_fixture,
            bind_workflow_plans=deps._bind_workflow_plans_to_parents,
            stage_workflow_plans=deps._stage_workflow_plans,
            run_workflow_hook_result=deps._run_workflow_hook_result,
            workflow_outbox_pending=deps._workflow_outbox_pending,
            read_exit_task_timing_stats=deps._read_exit_task_timing_stats,
            read_exit_task_call_stats=deps._read_exit_task_call_stats,
            read_exit_outbox_stats=deps._read_exit_outbox_stats,
            measure_workflow=deps._measure_workflow,
            attach_timing_breakdown=deps._attach_timing_breakdown,
        )
        history_key = "slow_device_queue_background_history_rows" if slow_device else "queue_background_history_rows"
        default_history_rows = 1000 if slow_device else 5000
        results["workflow_queue_drain_large_history"] = deps.workloads.queue_shape(
            name="queue-large-history",
            background_rows=max(default_history_rows, int(workflow_cfg.get(history_key, default_history_rows))),
            root=deps.ROOT, base_env=base_env, real_task=real_task, slow_device=slow_device,
            config_fingerprint=config_fingerprint, schedule_fingerprint=schedule_fingerprint,
            budgets=budgets, init_empty_outbox=deps._init_empty_outbox,
            outbox_lifecycle_fixture=deps._outbox_lifecycle_fixture,
            bind_workflow_plans=deps._bind_workflow_plans_to_parents,
            stage_workflow_plans=deps._stage_workflow_plans,
            run_workflow_hook_result=deps._run_workflow_hook_result,
            workflow_outbox_pending=deps._workflow_outbox_pending,
            read_exit_task_timing_stats=deps._read_exit_task_timing_stats,
            read_exit_task_call_stats=deps._read_exit_task_call_stats,
            read_exit_outbox_stats=deps._read_exit_outbox_stats,
            measure_workflow=deps._measure_workflow,
            attach_timing_breakdown=deps._attach_timing_breakdown,
        )

        history_rows = max(1, int(workflow_cfg.get("reconcile_history_rows", 256)))
        reconcile_data, reconcile_env, reconcile_tasks, _reconcile_fixture_details = deps.workloads.reconcile_history_fixture(
            root=root, base_env=base_env, history_rows=history_rows,
        )
        reconcile_cmd = [sys.executable, str(deps.ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"), "--json"]
        healthy_result = deps.workloads.reconcile_healthy(
            root=deps.ROOT,
            reconcile_env=reconcile_env,
            history_rows=history_rows,
            repeats=repeats,
            budgets=budgets,
            reconcile_call_purposes=reconcile_call_purposes,
            compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow,
            attach_reports=deps._attach_reconcile_reports,
        )
        reconcile_samples = []
        reconcile_reports: list[dict] = []
        results["workflow_reconcile"] = healthy_result

        # Keep an empty audit as a first-class workload.  This catches startup,
        # snapshot, and report overhead without accidentally measuring stale
        # queue cleanup or a failed Taskwarrior export.
        empty_data = root / "reconcile-empty"
        empty_env = dict(base_env, TASKDATA=str(empty_data))
        empty_result = deps.workloads.reconcile_empty(
            root=deps.ROOT,
            empty_data=empty_data,
            base_env=base_env,
            repeats=repeats,
            budgets=budgets,
            run_reconcile=lambda env: subprocess.run(reconcile_cmd, text=True, capture_output=True, env=env, timeout=30.0),
            compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow,
            attach_reports=deps._attach_reconcile_reports,
        )
        empty_samples = []
        empty_reports: list[dict] = []
        results["workflow_reconcile_empty"] = empty_result

        # Candidate-heavy audits must prove that the benchmark contains
        # actionable integrity evidence.  A zero-candidate run would only
        # measure the healthy path again.
        candidate_count = max(1, int(workflow_cfg.get("reconcile_candidate_chains", 32)))
        candidate_tasks = deps._reconcile_candidate_tasks("healthy", candidate_count)
        candidate_result = deps.workloads.reconcile_candidates(
            root=root,
            base_env=base_env,
            candidate_count=candidate_count,
            repeats=repeats,
            budgets=budgets,
            candidate_tasks=candidate_tasks,
            reconcile_command=reconcile_cmd,
            compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow,
            attach_reports=deps._attach_reconcile_reports,
        )
        candidate_data = root / "reconcile-candidates"
        candidate_env = dict(base_env, TASKDATA=str(candidate_data))
        candidate_samples = []
        candidate_reports: list[dict] = []
        results["workflow_reconcile_candidates"] = candidate_result

        results["workflow_reconcile_candidates_apply"] = deps.workloads.reconcile_candidates_apply(
            root=root,
            base_env=base_env,
            candidate_tasks=candidate_tasks,
            reconcile_command=reconcile_cmd,
            budgets=budgets,
            compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow,
            attach_reports=deps._attach_reconcile_reports,
        )

        apply_scale_counts = tuple(
            int(value)
            for value in workflow_cfg.get("reconcile_candidate_apply_counts", (1, 8, 32, 200))
            if int(value) > 0
        )
        results["workflow_reconcile_candidates_apply_scale"] = deps.workloads.reconcile_candidates_apply_scale(
            root=root,
            base_env=base_env,
            scale_counts=apply_scale_counts,
            candidate_tasks_factory=deps._reconcile_candidate_tasks,
            reconcile_command=reconcile_cmd,
            compact_report=deps._compact_reconcile_report,
        )
        apply_scale_samples: list[float] = []
        apply_scale_reports: list[dict] = []
        apply_scale_rows: list[dict] = []
        for scale_count in ():
            scale_data = root / f"reconcile-candidates-apply-{scale_count}"
            scale_data.mkdir()
            scale_env = dict(base_env, TASKDATA=str(scale_data))
            scale_tasks = deps._reconcile_candidate_tasks(f"apply-{scale_count}", scale_count)
            scale_import = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in scale_tasks),
                text=True,
                capture_output=True,
                env=scale_env,
                timeout=120.0,
            )
            if scale_import.returncode != 0:
                raise RuntimeError(
                    f"candidate apply scale fixture import failed ({scale_count}): "
                    f"{(scale_import.stderr or scale_import.stdout or '').strip()}"
                )
            scale_started = time.perf_counter()
            scale_proc = subprocess.run(
                [*reconcile_cmd, "--apply"],
                text=True,
                capture_output=True,
                env=scale_env,
                timeout=240.0,
            )
            elapsed = time.perf_counter() - scale_started
            if scale_proc.returncode != 0:
                raise RuntimeError(
                    f"candidate apply scale failed ({scale_count}): "
                    f"{(scale_proc.stderr or scale_proc.stdout or '').strip()}"
                )
            try:
                scale_report = json.loads(scale_proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"candidate apply scale returned invalid JSON ({scale_count})") from exc
            if int(scale_report.get("spawn", 0)) < scale_count or not scale_report.get("applied"):
                raise RuntimeError(f"candidate apply scale did not converge ({scale_count}): {scale_report!r}")
            apply_scale_samples.append(elapsed)
            compact = deps._compact_reconcile_report(scale_report)
            apply_scale_reports.append(compact)
            apply_scale_rows.append({"candidate_count": scale_count, "elapsed_s": round(elapsed, 6), **compact})

        long_data = root / "reconcile-long-history"
        long_data.mkdir()
        long_env = dict(base_env, TASKDATA=str(long_data))
        long_count = max(history_rows, int(workflow_cfg.get("reconcile_long_history_rows", 2048)))
        long_tasks = []
        for link in range(1, long_count + 1):
            task = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-long/{link}")),
                "status": "completed",
                "description": f"Long reconcile benchmark {link}",
                "cp": "P1D",
                "chain": "on",
                "chainID": "reconcile-long-chain",
                "link": link,
                "due": f"202601{min(link, 28):02d}T090000Z",
            }
            if link > 1:
                task["prevLink"] = long_tasks[-1]["uuid"][:8]
            if link < long_count:
                task["nextLink"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-long/{link + 1}"))[:8]
            long_tasks.append(task)
        long_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in long_tasks),
            text=True,
            capture_output=True,
            env=long_env,
            timeout=300.0 if slow_device else 60.0,
        )
        if long_import.returncode != 0:
            raise RuntimeError(
                "long reconcile fixture import failed: "
                f"{(long_import.stderr or long_import.stdout or '').strip()}"
            )
        def _validate_long(report: dict) -> None:
            if not isinstance(report, dict) or report.get("schema") != "nautical.reconcile" or not 1 <= int(report.get("export_calls", 0)) <= 2 or not 1 <= int(report.get("export_rows", 0)) <= long_count or float(report.get("integrity_seconds", -1.0)) < 0.0:
                raise RuntimeError("long reconcile workflow exceeded its single-snapshot row budget")
        long_result = deps.workloads.reconcile_report_loop(
            name="workflow_reconcile_long_history", command=reconcile_cmd, env=long_env,
            repeats=repeats, timeout=300.0 if slow_device else 60.0,
            budget=float(budgets.get("workflow_reconcile_long_history", budgets.get("workflow_reconcile", 3.0))),
            validate=_validate_long, compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow, attach_reports=deps._attach_reconcile_reports,
        )
        long_samples = []
        long_reports: list[dict] = []
        results["workflow_reconcile_long_history"] = long_result

        corrupt_data = root / "reconcile-corrupted"
        corrupt_data.mkdir()
        corrupt_env = dict(base_env, TASKDATA=str(corrupt_data))
        corrupt_tasks = []
        corrupt_count = max(1, int(workflow_cfg.get("reconcile_corrupted_chains", 16)))
        for index in range(corrupt_count):
            chain_id = f"reconcile-corrupt-{index}"
            first_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-corrupt/{index}/first"))
            second_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-corrupt/{index}/second"))
            common = {
                "status": "pending",
                "description": f"Corrupted reconcile benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": chain_id,
                "link": 1,
                "due": "20260101T090000Z",
            }
            corrupt_tasks.extend([
                dict(common, uuid=first_uuid, nextLink=second_uuid[:8]),
                dict(common, uuid=second_uuid, prevLink=first_uuid[:8]),
            ])
        corrupt_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in corrupt_tasks),
            text=True,
            capture_output=True,
            env=corrupt_env,
            timeout=30.0,
        )
        if corrupt_import.returncode != 0:
            raise RuntimeError(
                "corrupted reconcile fixture import failed: "
                f"{(corrupt_import.stderr or corrupt_import.stdout or '').strip()}"
            )
        def _validate_corrupt(report: dict) -> None:
            audit = report.get("integrity_audit") if isinstance(report, dict) else None
            if not isinstance(audit, dict) or not audit.get("findings"):
                raise RuntimeError("corrupted reconcile workflow hid its integrity findings")
        corrupt_result = deps.workloads.reconcile_report_loop(
            name="workflow_reconcile_corrupted", command=reconcile_cmd, env=corrupt_env,
            repeats=repeats, timeout=30.0,
            budget=float(budgets.get("workflow_reconcile_corrupted", budgets.get("workflow_reconcile", 3.0))),
            validate=_validate_corrupt, compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow, attach_reports=deps._attach_reconcile_reports,
        )
        corrupt_samples = []
        corrupt_reports: list[dict] = []
        results["workflow_reconcile_corrupted"] = corrupt_result

        mixed_data = root / "reconcile-mixed"
        mixed_data.mkdir()
        mixed_env = dict(base_env, TASKDATA=str(mixed_data))
        mixed_tasks = []
        mixed_healthy_count = max(2, int(workflow_cfg.get("reconcile_mixed_healthy_rows", 8)))
        healthy_rows = []
        for link in range(1, mixed_healthy_count + 1):
            row = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/healthy/{link}")),
                "status": "completed",
                "description": f"Mixed healthy benchmark {link}",
                "cp": "P1D",
                "chain": "on",
                "chainID": "reconcile-mixed-healthy",
                "link": link,
                "due": "20260101T090000Z",
            }
            if healthy_rows:
                row["prevLink"] = healthy_rows[-1]["uuid"][:8]
            healthy_rows.append(row)
        for index, row in enumerate(healthy_rows):
            if index + 1 < len(healthy_rows):
                row["nextLink"] = healthy_rows[index + 1]["uuid"][:8]
        mixed_tasks.extend(healthy_rows)
        for index in range(max(1, int(workflow_cfg.get("reconcile_mixed_candidates", 8)))):
            mixed_tasks.append({
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/candidate/{index}")),
                "status": "completed",
                "description": f"Mixed candidate benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": f"reconcile-mixed-candidate-{index}",
                "link": 1,
                "due": "20260101T090000Z",
            })
        for index in range(max(1, int(workflow_cfg.get("reconcile_mixed_corrupted", 4)))):
            chain_id = f"reconcile-mixed-corrupt-{index}"
            first_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/corrupt/{index}/first"))
            second_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/corrupt/{index}/second"))
            common = {
                "status": "pending",
                "description": f"Mixed corrupted benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": chain_id,
                "link": 1,
                "due": "20260101T090000Z",
            }
            mixed_tasks.extend([
                dict(common, uuid=first_uuid, nextLink=second_uuid[:8]),
                dict(common, uuid=second_uuid, prevLink=first_uuid[:8]),
            ])
        mixed_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in mixed_tasks),
            text=True,
            capture_output=True,
            env=mixed_env,
            timeout=30.0,
        )
        if mixed_import.returncode != 0:
            raise RuntimeError(
                "mixed reconcile fixture import failed: "
                f"{(mixed_import.stderr or mixed_import.stdout or '').strip()}"
            )
        def _validate_mixed(report: dict) -> None:
            audit = report.get("integrity_audit") if isinstance(report, dict) else None
            if not isinstance(audit, dict) or not audit.get("findings") or int(report.get("candidates", 0)) <= 0:
                raise RuntimeError("mixed reconcile workflow did not preserve both candidate and integrity evidence")
        mixed_result = deps.workloads.reconcile_report_loop(
            name="workflow_reconcile_mixed", command=reconcile_cmd, env=mixed_env,
            repeats=repeats, timeout=30.0,
            budget=float(budgets.get("workflow_reconcile_mixed", budgets.get("workflow_reconcile", 3.0))),
            validate=_validate_mixed, compact_report=deps._compact_reconcile_report,
            measure_workflow=deps._measure_workflow, attach_reports=deps._attach_reconcile_reports,
        )
        mixed_samples = []
        mixed_reports: list[dict] = []
        results["workflow_reconcile_mixed"] = mixed_result
        reconcile_budgets = workflow_cfg.get("reconcile_budgets", {})
        if isinstance(reconcile_budgets, dict):
            for name, result in results.items():
                if name.startswith("workflow_reconcile"):
                    deps._apply_reconcile_budgets(result, reconcile_budgets.get(name, reconcile_budgets.get("default", {})))
        deps.RESOURCE_DETAILS["reconcile_task_call_purposes"] = reconcile_call_purposes
    return results



def integrity_scale(
    scale_counts: tuple[int, ...],
    budgets: dict,
    measure_workflow: Callable[..., dict],
) -> dict[str, dict]:
    """Measure graph/invariant scaling without Taskwarrior subprocesses."""
    from nautical_core.chain_graph import ChainGraph
    from nautical_core.chain_integrity_models import ChainNode, ChainSnapshot, SnapshotCoverage
    from nautical_core.chain_invariants import evaluate_invariants
    from nautical_core.task_models import TaskObservation

    results: dict[str, dict] = {}
    for scale_count in scale_counts:
        if scale_count <= 0:
            continue
        started = time.perf_counter()
        rows = tuple(
            ChainNode.from_observation(TaskObservation.from_mapping({
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/integrity-scale/{scale_count}/{index}")),
                "status": "pending", "description": f"Integrity scale {index}",
                "chain": "on", "chainID": f"integrity-scale-{index}", "link": 1,
                "cp": "P1D", "due": "20260101T090000Z",
            }, source_query="perf:integrity-scale"))
            for index in range(scale_count)
        )
        snapshot = ChainSnapshot(
            f"integrity-scale-{scale_count}", SnapshotCoverage.COMPLETE, "perf.synthetic",
            rows, complete_chain_history=True,
        )
        findings = evaluate_invariants(ChainGraph.from_snapshot(snapshot))
        if findings:
            raise RuntimeError(f"integrity scale fixture produced findings at {scale_count} chains")
        name = f"integrity_scale_{scale_count}"
        results[name] = measure_workflow(name, [time.perf_counter() - started], float(budgets.get(name, 0.0)))
    return results


def ordinary_modify(
    *,
    root: Path,
    workflow_root: Path,
    base_env: dict[str, str],
    repeats: int,
    completion_fixture: Callable[..., dict],
    import_workflow_rows: Callable[..., None],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    workflow_outbox_pending: Callable[[Path], list],
    measure_workflow: Callable[..., dict],
    budgets: dict,
) -> dict:
    """Measure ordinary Nautical edits without staging lifecycle work."""
    samples = []
    for sample_index in range(repeats):
        old = completion_fixture("cp", sample_index, nonfinal=True, mode="ordinary")
        new = dict(old, description=f"Ordinary Nautical edit {sample_index}")
        taskdata = workflow_root / f"ordinary-modify-{sample_index}"
        taskdata.mkdir()
        env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1")
        import_workflow_rows((old,), env=env)
        elapsed, result, _stderr = run_workflow_hook_result(
            root / "on-modify.nautical",
            input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
            env=env,
            expect_output=True,
        )
        if result != new or workflow_outbox_pending(taskdata):
            raise RuntimeError("workflow_ordinary_modify changed the task or staged work")
        samples.append(elapsed)
    return measure_workflow("workflow_ordinary_modify", samples, float(budgets.get("workflow_ordinary_modify", 2.0)))


def expiration_recovery(
    *,
    root: Path,
    workflow_root: Path,
    base_env: dict[str, str],
    repeats: int,
    import_workflow_rows: Callable[..., None],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    workflow_outbox_pending: Callable[[Path], list],
    measure_workflow: Callable[..., dict],
    budgets: dict,
) -> dict:
    """Measure expiration recovery and its replay idempotency guard."""
    samples = []
    expiration_date = (datetime.now(timezone.utc).date() - timedelta(days=1)).strftime("%Y%m%d")
    for sample_index in range(repeats):
        key = f"nautical-perf/expiration/{sample_index}"
        parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, key + "/parent"))
        old = {
            "uuid": parent_uuid, "status": "pending",
            "description": f"Expiration recovery benchmark {sample_index}",
            "cp": "P1D", "chain": "on", "chainID": f"expiration-perf-{sample_index:04d}",
            "link": 1, "due": f"{expiration_date}T090000Z", "until": f"{expiration_date}T200000Z",
            "modified": f"{expiration_date}T090000Z",
        }
        new = dict(old, status="deleted", end=f"{expiration_date}T200000Z")
        taskdata = workflow_root / f"expiration-recovery-{sample_index}"
        taskdata.mkdir()
        env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1")
        import_workflow_rows((old,), env=env)
        elapsed, result, stderr = run_workflow_hook_result(
            root / "on-modify.nautical",
            input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
            env=env,
            expect_output=True,
        )
        staged = workflow_outbox_pending(taskdata)
        if not isinstance(result, dict) or result.get("chain") != "on" or len(staged) != 1:
            raise RuntimeError(f"workflow_expiration_recovery did not stage exactly one successor: result={result!r}; staged={staged!r}; stderr={stderr.strip()!r}")
        replay_result = run_workflow_hook_result(
            root / "on-modify.nautical",
            input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
            env=env,
            expect_output=True,
        )
        replay_staged = workflow_outbox_pending(taskdata)
        if not isinstance(replay_result[1], dict) or len(replay_staged) != 1:
            raise RuntimeError(f"workflow_expiration_recovery replay was not idempotent: result={replay_result[1]!r}; staged={replay_staged!r}; stderr={replay_result[2].strip()!r}")
        samples.append(elapsed)
    return measure_workflow("workflow_expiration_recovery", samples, float(budgets.get("workflow_expiration_recovery", 2.0)))


def completion_workflows(
    *,
    completion_cases: tuple[tuple[str, str, bool], ...],
    root: Path,
    workflow_root: Path,
    base_env: dict[str, str],
    repeats: int,
    budgets: dict,
    workflow_cfg: dict,
    completion_fixture: Callable[..., dict],
    import_workflow_rows: Callable[..., None],
    import_existing_completion_child: Callable[..., None],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    workflow_outbox_pending: Callable[[Path], list],
    read_exit_task_call_stats: Callable[[Path], dict[str, int]],
    measure_workflow: Callable[..., dict],
    apply_task_call_budgets: Callable[..., None],
) -> dict[str, dict]:
    """Measure fresh and idempotent completion paths for each chain kind."""
    results: dict[str, dict] = {}
    call_budgets = workflow_cfg.get("task_call_budgets")
    for name, kind, nonfinal in completion_cases:
        fresh_samples = []
        fresh_call_stats: list[dict[str, int]] = []
        for sample_index in range(repeats):
            old = completion_fixture(kind, sample_index, nonfinal=nonfinal, mode="fresh")
            new = dict(old, status="completed", end="20260101T100000Z" if kind == "cp" else "20260105T100000Z")
            taskdata = workflow_root / f"{name}-fresh-{sample_index}"; taskdata.mkdir()
            stats_path = taskdata / "on-modify-task-stats.json"
            env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1", NAUTICAL_BENCH_STATS_FILE=str(stats_path))
            import_workflow_rows((old,), env=env)
            elapsed, result, stderr = run_workflow_hook_result(
                root / "on-modify.nautical",
                input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                env=env, expect_output=True,
            )
            if not isinstance(result, dict):
                raise RuntimeError(f"{name} fresh sample returned no task object")
            fresh_call_stats.append(read_exit_task_call_stats(stats_path))
            queued = workflow_outbox_pending(taskdata)
            if nonfinal:
                if result.get("chain") != "on" or len(queued) != 1:
                    raise RuntimeError(f"{name} fresh sample did not queue exactly one child: result={result!r}; queued={queued!r}; stderr={stderr.strip()!r}")
                if queued[0].get("stage") != "planned":
                    raise RuntimeError(f"{name} fresh sample staged an invalid lifecycle record")
            elif result.get("chain") != "off" or queued:
                raise RuntimeError(f"{name} final sample did not complete without a successor")
            fresh_samples.append(elapsed)
        results[name] = measure_workflow(name, fresh_samples, float(budgets.get(name, 2.0)))
        results[name]["task_call_stats"] = fresh_call_stats
        if isinstance(call_budgets, dict):
            apply_task_call_budgets(results[name], fresh_call_stats, call_budgets.get(name, {}))

        if nonfinal:
            idem_name = f"{name}_idempotent"
            idem_samples = []
            idem_call_stats: list[dict[str, int]] = []
            for sample_index in range(repeats):
                old = completion_fixture(kind, sample_index, nonfinal=True, mode="idempotent")
                new = dict(old, status="completed", end="20260101T100000Z" if kind == "cp" else "20260105T100000Z")
                taskdata = workflow_root / f"{idem_name}-{sample_index}"; taskdata.mkdir()
                stats_path = taskdata / "on-modify-task-stats.json"
                env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_STATS_FILE=str(stats_path))
                import_workflow_rows((old,), env=env)
                import_existing_completion_child(old, env=env)
                elapsed, result, stderr = run_workflow_hook_result(
                    root / "on-modify.nautical",
                    input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                    env=env, expect_output=True,
                )
                if not isinstance(result, dict) or result.get("chain") != "on":
                    raise RuntimeError(f"{idem_name} sample changed the completed parent unexpectedly")
                if workflow_outbox_pending(taskdata):
                    raise RuntimeError(f"{idem_name} sample staged a duplicate child")
                if "Spawn skipped" not in stderr:
                    raise RuntimeError(f"{idem_name} sample did not report the existing next link")
                idem_call_stats.append(read_exit_task_call_stats(stats_path)); idem_samples.append(elapsed)
            results[idem_name] = measure_workflow(idem_name, idem_samples, float(budgets.get(idem_name, 2.0)))
            results[idem_name]["task_call_stats"] = idem_call_stats
            if isinstance(call_budgets, dict):
                apply_task_call_budgets(results[idem_name], idem_call_stats, call_budgets.get(idem_name, {}))
    return results


def queue_preflight(
    *,
    task_bin: str,
    queue_data: Path,
    queue_env: dict[str, str],
    sample_index: int,
    outbox_lifecycle_fixture: Callable[..., tuple[list[dict], list]],
    bind_workflow_plans: Callable[..., list],
    stage_workflow_plans: Callable[..., None],
    configuration_fingerprint: str,
    schedule_fingerprint: str,
) -> tuple[list[dict], list, set[str], set[str]]:
    """Import, verify, bind, and stage one queue-drain fixture."""
    parents, queue_plans = outbox_lifecycle_fixture("queue", sample_index)
    parent_uuids = {str(parent["uuid"]) for parent in parents}
    child_uuids = {str(plan.child_dict()["uuid"]) for plan in queue_plans}
    imported = subprocess.run(
        [task_bin, "rc.hooks=off", "rc.verbose=nothing", "import"],
        input="".join(json.dumps(parent, ensure_ascii=False) + "\n" for parent in parents),
        text=True, capture_output=True, env=queue_env, timeout=30.0,
    )
    if imported.returncode != 0:
        raise RuntimeError(f"queue drain parent fixture import failed: {(imported.stderr or imported.stdout or '').strip()}")
    probe = subprocess.run(
        [task_bin, f"rc.data.location={queue_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on", "(", "status:completed", "or", "status:deleted", "or", "status:pending", "or", "status:waiting", ")", "export"],
        text=True, capture_output=True, env=queue_env, timeout=30.0,
    )
    rows = json.loads(probe.stdout or "[]")
    if probe.returncode != 0 or not isinstance(rows, list) or {str(row.get("uuid") or "") for row in rows if isinstance(row, dict)} != parent_uuids:
        raise RuntimeError(f"queue drain benchmark parent preflight was not visible to Taskwarrior: {(probe.stderr or probe.stdout or '').strip()}")
    queue_plans = bind_workflow_plans(queue_plans, rows)
    stage_workflow_plans(queue_data, queue_plans, configuration_fingerprint=configuration_fingerprint, schedule_fingerprint=schedule_fingerprint)
    return parents, queue_plans, parent_uuids, child_uuids


def queue_healthy_replay(
    *,
    root: Path,
    queue_data: Path,
    queue_env: dict[str, str],
    stats_path: Path,
    queue_plans: list,
    parent_uuids: set[str],
    child_uuids: set[str],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    read_exit_task_call_stats: Callable[[Path], dict[str, int]],
    read_exit_task_timing_stats: Callable[[Path], dict[str, float]],
    read_exit_outbox_stats: Callable[[Path], dict[str, float]],
    workflow_outbox_pending: Callable[[Path], list],
    lifecycle_outbox: Any,
) -> tuple[float, dict[str, int], dict[str, float], dict[str, float], float, dict[str, int], dict[str, float], dict[str, float]]:
    """Run one healthy queue drain, replay it, and verify convergence."""
    queue_elapsed, _queue_result, queue_stderr = run_workflow_hook_result(
        root / "on-exit.nautical", input_text="", env=queue_env, expect_output=False,
    )
    queue_call_stats = read_exit_task_call_stats(stats_path)
    child_slot_reads = queue_call_stats.get("run_task_calls_purpose_task_read_child_slot", 0)
    if child_slot_reads > 1:
        raise RuntimeError(f"queue drain preflight regressed to per-candidate child-slot reads: {child_slot_reads} subprocesses for {len(queue_plans)} candidates")
    queue_timing_stats = read_exit_task_timing_stats(stats_path)
    queue_outbox_stats = read_exit_outbox_stats(stats_path)
    if workflow_outbox_pending(queue_data):
        raise RuntimeError(f"outbox drain benchmark left active intents after successful processing: {workflow_outbox_pending(queue_data)!r}; stderr={queue_stderr.strip()!r}")
    try:
        stats_path.unlink()
    except FileNotFoundError:
        pass
    idem_elapsed, _idem_result, idem_stderr = run_workflow_hook_result(
        root / "on-exit.nautical", input_text="", env=dict(queue_env, NAUTICAL_BENCH_FORCE_FULL="1"), expect_output=False,
    )
    idem_calls = read_exit_task_call_stats(stats_path)
    idem_timing = read_exit_task_timing_stats(stats_path)
    idem_outbox = read_exit_outbox_stats(stats_path)
    replay_task_calls = sum(value for key, value in idem_calls.items() if key.startswith("run_task_calls"))
    if replay_task_calls or idem_calls.get("task_read_rows", 0):
        raise RuntimeError(f"acknowledged exit replay performed Taskwarrior I/O: calls={replay_task_calls}, rows={idem_calls.get('task_read_rows', 0)}")
    if workflow_outbox_pending(queue_data):
        raise RuntimeError(f"idempotent outbox drain left active intents: {workflow_outbox_pending(queue_data)!r}; stderr={idem_stderr.strip()!r}")
    export_proc = subprocess.run(["task", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", "chain:on", "export"], text=True, capture_output=True, env=queue_env, timeout=30.0)
    if export_proc.returncode != 0:
        raise RuntimeError(f"queue drain benchmark export failed: {(export_proc.stderr or export_proc.stdout or '').strip()}")
    try:
        exported = json.loads(export_proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("queue drain benchmark export was not valid JSON") from exc
    if not isinstance(exported, list) or len(exported) != 16:
        raise RuntimeError(f"queue drain benchmark did not retain 8 parents and import 8 children: {len(exported) if isinstance(exported, list) else type(exported).__name__} tasks; outbox={lifecycle_outbox._LifecycleOutboxRepository(queue_data).status(limit=20)[1]!r}")
    children = [row for row in exported if isinstance(row, dict) and str(row.get("uuid") or "") in child_uuids]
    if len(children) != 8 or any(not str(row.get("prevLink") or "").strip() for row in children):
        raise RuntimeError("queue drain benchmark did not import/link all child tasks")
    parents_after = [row for row in exported if isinstance(row, dict) and str(row.get("uuid") or "") in parent_uuids]
    if len(parents_after) != 8 or any(not str(row.get("nextLink") or "").strip() for row in parents_after):
        raise RuntimeError("queue drain benchmark did not update all parent nextLink values")
    return queue_elapsed, queue_call_stats, queue_timing_stats, queue_outbox_stats, idem_elapsed, idem_calls, idem_timing, idem_outbox


def queue_replay_verify(
    *,
    root: Path,
    queue_data: Path,
    queue_env: dict[str, str],
    stats_path: Path,
    parent_uuids: set[str],
    child_uuids: set[str],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    read_exit_task_call_stats: Callable[[Path], dict[str, int]],
    read_exit_task_timing_stats: Callable[[Path], dict[str, float]],
    read_exit_outbox_stats: Callable[[Path], dict[str, float]],
    workflow_outbox_pending: Callable[[Path], list],
    lifecycle_outbox: Any,
) -> tuple[float, dict[str, int], dict[str, float], dict[str, float]]:
    """Replay an acknowledged drain and verify exported chain convergence."""
    try:
        stats_path.unlink()
    except FileNotFoundError:
        pass
    elapsed, _result, stderr = run_workflow_hook_result(
        root / "on-exit.nautical", input_text="", env=dict(queue_env, NAUTICAL_BENCH_FORCE_FULL="1"), expect_output=False,
    )
    calls = read_exit_task_call_stats(stats_path)
    timing = read_exit_task_timing_stats(stats_path)
    outbox = read_exit_outbox_stats(stats_path)
    replay_task_calls = sum(value for key, value in calls.items() if key.startswith("run_task_calls"))
    if replay_task_calls or calls.get("task_read_rows", 0):
        raise RuntimeError(f"acknowledged exit replay performed Taskwarrior I/O: calls={replay_task_calls}, rows={calls.get('task_read_rows', 0)}")
    if workflow_outbox_pending(queue_data):
        raise RuntimeError(f"idempotent outbox drain left active intents: {workflow_outbox_pending(queue_data)!r}; stderr={stderr.strip()!r}")
    export_proc = subprocess.run(["task", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", "chain:on", "export"], text=True, capture_output=True, env=queue_env, timeout=30.0)
    if export_proc.returncode != 0:
        raise RuntimeError(f"queue drain benchmark export failed: {(export_proc.stderr or export_proc.stdout or '').strip()}")
    try:
        exported = json.loads(export_proc.stdout or "[]")
    except json.JSONDecodeError as exc:
        raise RuntimeError("queue drain benchmark export was not valid JSON") from exc
    if not isinstance(exported, list) or len(exported) != 16:
        raise RuntimeError(f"queue drain benchmark did not retain 8 parents and import 8 children: {len(exported) if isinstance(exported, list) else type(exported).__name__} tasks; outbox={lifecycle_outbox._LifecycleOutboxRepository(queue_data).status(limit=20)[1]!r}")
    children = [row for row in exported if isinstance(row, dict) and str(row.get("uuid") or "") in child_uuids]
    parents = [row for row in exported if isinstance(row, dict) and str(row.get("uuid") or "") in parent_uuids]
    if len(children) != 8 or any(not str(row.get("prevLink") or "").strip() for row in children):
        raise RuntimeError("queue drain benchmark did not import/link all child tasks")
    if len(parents) != 8 or any(not str(row.get("nextLink") or "").strip() for row in parents):
        raise RuntimeError("queue drain benchmark did not update all parent nextLink values")
    return elapsed, calls, timing, outbox


def queue_partial_recovery(
    *,
    root: Path,
    partial_data: Path,
    partial_env: dict[str, str],
    partial_stats_path: Path,
    task_wrapper: Path,
    sample_index: int,
    configuration_fingerprint: str,
    schedule_fingerprint: str,
    outbox_lifecycle_fixture: Callable[..., tuple[list[dict], list]],
    init_empty_outbox: Callable[[Path], None],
    bind_workflow_plans: Callable[..., list],
    stage_workflow_plans: Callable[..., None],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    read_exit_task_call_stats: Callable[[Path], dict[str, int]],
    read_exit_task_timing_stats: Callable[[Path], dict[str, float]],
    read_exit_outbox_stats: Callable[[Path], dict[str, float]],
    workflow_outbox_pending: Callable[[Path], list],
    merge_task_call_stats: Callable[..., dict[str, int]],
    merge_task_timing_stats: Callable[..., dict[str, float]],
) -> tuple[float, float, dict[str, int], dict[str, float], dict[str, float], dict[str, int], dict[str, float], dict[str, float]]:
    """Exercise partial import failure, requeue, and recovery convergence."""
    init_empty_outbox(partial_data)
    partial_stats_path = Path(partial_stats_path)
    partial_parents, partial_plans = outbox_lifecycle_fixture("partial", sample_index)
    imported = subprocess.run(["task", "rc.hooks=off", "rc.verbose=nothing", "import"], input="".join(json.dumps(parent, ensure_ascii=False) + "\n" for parent in partial_parents), text=True, capture_output=True, env=partial_env, timeout=30.0)
    if imported.returncode != 0: raise RuntimeError("partial queue parent fixture import failed")
    probe = subprocess.run(["task", f"rc.data.location={partial_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on", "export"], text=True, capture_output=True, env=partial_env, timeout=30.0)
    if probe.returncode != 0: raise RuntimeError("partial queue parent verification export failed")
    partial_plans = bind_workflow_plans(partial_plans, json.loads(probe.stdout or "[]"))
    stage_workflow_plans(partial_data, partial_plans, configuration_fingerprint=configuration_fingerprint, schedule_fingerprint=schedule_fingerprint)
    partial_env["NAUTICAL_BENCH_FAIL_MODE"] = "partial-import"
    try: partial_stats_path.unlink()
    except FileNotFoundError: pass
    first_t0 = time.perf_counter()
    _first, _result, first_stderr = run_workflow_hook_result(root / "on-exit.nautical", input_text="", env=partial_env, expect_output=False)
    first_elapsed = time.perf_counter() - first_t0
    first_calls = read_exit_task_call_stats(partial_stats_path); first_timing = read_exit_task_timing_stats(partial_stats_path); first_outbox = read_exit_outbox_stats(partial_stats_path)
    pending = workflow_outbox_pending(partial_data)
    if not 1 <= len(pending) <= 8: raise RuntimeError(f"partial import did not requeue all lifecycle intents: {pending!r}; stderr={first_stderr.strip()!r}")
    partial_env.pop("NAUTICAL_BENCH_FAIL_MODE", None)
    try: partial_stats_path.unlink()
    except FileNotFoundError: pass
    recovery_t0 = time.perf_counter()
    _second, _result, second_stderr = run_workflow_hook_result(root / "on-exit.nautical", input_text="", env=partial_env, expect_output=False)
    recovery_elapsed = time.perf_counter() - recovery_t0
    second_calls = read_exit_task_call_stats(partial_stats_path); second_timing = read_exit_task_timing_stats(partial_stats_path); second_outbox = read_exit_outbox_stats(partial_stats_path)
    if workflow_outbox_pending(partial_data): raise RuntimeError(f"partial import recovery left active lifecycle intents: {workflow_outbox_pending(partial_data)!r}; stderr={second_stderr.strip()!r}")
    return first_elapsed, recovery_elapsed, merge_task_call_stats(first_calls, second_calls), merge_task_timing_stats(first_timing, second_timing), merge_task_timing_stats(first_outbox, second_outbox), first_calls, first_timing, first_outbox


def queue_shape(
    *,
    name: str,
    background_rows: int,
    root: Path,
    base_env: dict[str, str],
    real_task: str,
    slow_device: bool,
    config_fingerprint: str,
    schedule_fingerprint: str,
    budgets: dict,
    init_empty_outbox: Callable[[Path], None],
    outbox_lifecycle_fixture: Callable[..., tuple[list[dict], list]],
    bind_workflow_plans: Callable[..., list],
    stage_workflow_plans: Callable[..., None],
    run_workflow_hook_result: Callable[..., tuple[float, dict | None, str]],
    workflow_outbox_pending: Callable[[Path], list],
    read_exit_task_timing_stats: Callable[[Path], dict[str, float]],
    read_exit_task_call_stats: Callable[[Path], dict[str, int]],
    read_exit_outbox_stats: Callable[[Path], dict[str, float]],
    measure_workflow: Callable[..., dict],
    attach_timing_breakdown: Callable[..., None],
) -> dict:
    """Measure one healthy intent with optional unrelated history."""
    shape_data = root / name; init_empty_outbox(shape_data)
    stats_path = shape_data / "on-exit-task-stats.json"
    env = dict(base_env, TASKDATA=str(shape_data), NAUTICAL_BENCH_STATS_FILE=str(stats_path), NAUTICAL_BENCH_TASK_BIN=real_task)
    parents, plans = outbox_lifecycle_fixture(name, 0, count=1)
    background = [{"uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{name}/background/{index}")), "status": "completed", "description": f"Unrelated background history {index}", "due": "20250101T090000Z"} for index in range(background_rows)]
    imported = subprocess.run(["task", "rc.hooks=off", "rc.verbose=nothing", "import"], input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*background, *parents]), text=True, capture_output=True, env=env, timeout=300.0 if slow_device else 120.0)
    if imported.returncode != 0: raise RuntimeError(f"{name} fixture import failed: {(imported.stderr or imported.stdout or '').strip()}")
    probe = subprocess.run(["task", f"rc.data.location={shape_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on", "export"], text=True, capture_output=True, env=env, timeout=120.0)
    if probe.returncode != 0: raise RuntimeError(f"{name} fixture probe failed: {(probe.stderr or probe.stdout or '').strip()}")
    plans = bind_workflow_plans(plans, json.loads(probe.stdout or "[]"))
    stage_workflow_plans(shape_data, plans, configuration_fingerprint=config_fingerprint, schedule_fingerprint=schedule_fingerprint)
    started = time.perf_counter()
    elapsed, _result, stderr = run_workflow_hook_result(root / "on-exit.nautical", input_text="", env=env, expect_output=False)
    if workflow_outbox_pending(shape_data): raise RuntimeError(f"{name} left active outbox work: {workflow_outbox_pending(shape_data)!r}")
    timing = read_exit_task_timing_stats(stats_path); calls = read_exit_task_call_stats(stats_path); outbox = read_exit_outbox_stats(stats_path)
    if not calls.get("run_task_calls"): raise RuntimeError(f"{name} did not execute Taskwarrior commands: {stderr.strip()!r}")
    result = measure_workflow(name, [max(elapsed, time.perf_counter() - started)], float(budgets.get(name, 3.5)))
    result["background_rows"] = background_rows; result["task_call_stats"] = [calls]; result["outbox_stats"] = [outbox]
    attach_timing_breakdown(result, [elapsed], [timing])
    return result


def reconcile_history_fixture(
    *,
    root: Path,
    base_env: dict[str, str],
    history_rows: int,
) -> tuple[Path, dict[str, str], list[dict], list[str]]:
    """Build and import a deterministic completed-chain reconcile fixture."""
    reconcile_data = root / "reconcile"; reconcile_data.mkdir()
    reconcile_env = dict(base_env, TASKDATA=str(reconcile_data))
    tasks = []
    for link in range(1, history_rows + 1):
        task = {"uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile/{link}")), "status": "completed", "description": f"Reconcile performance benchmark {link}", "cp": "P1D", "chain": "on", "chainID": "reconcile-perf-chain", "link": link, "due": f"202601{min(link, 28):02d}T090000Z"}
        if link > 1: task["prevLink"] = tasks[-1]["uuid"][:8]
        if link < history_rows: task["nextLink"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile/{link + 1}"))[:8]
        tasks.append(task)
    imported = subprocess.run(["task", "rc.hooks=off", "rc.verbose=nothing", "import"], input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), text=True, capture_output=True, env=reconcile_env, timeout=30.0)
    if imported.returncode != 0: raise RuntimeError(f"reconcile fixture import failed: {(imported.stderr or imported.stdout or '').strip()}")
    return reconcile_data, reconcile_env, tasks, []


def reconcile_healthy(
    *,
    root: Path,
    reconcile_env: dict[str, str],
    history_rows: int,
    repeats: int,
    budgets: dict,
    reconcile_call_purposes: dict[str, int],
    compact_report: Callable[[dict], dict],
    measure_workflow: Callable[..., dict],
    attach_reports: Callable[..., None],
) -> dict:
    """Measure bounded healthy reconcile snapshots."""
    command = [__import__("sys").executable, str(root / "nautical_core" / "tools" / "nautical_reconcile.py"), "--json"]
    samples = []; reports = []
    for _ in range(repeats):
        started = time.perf_counter()
        proc = subprocess.run(command, text=True, capture_output=True, env=reconcile_env, timeout=30.0)
        if proc.returncode != 0: raise RuntimeError(f"reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
        report = json.loads(proc.stdout or "{}")
        if not isinstance(report, dict) or not 1 <= int(report.get("export_calls", 0)) <= 2 or not 1 <= int(report.get("export_rows", 0)) <= history_rows or float(report.get("integrity_seconds", -1.0)) < 0.0 or float(report.get("integrity_application_seconds", -1.0)) < 0.0 or int(report.get("task_command_calls", -1)) < 1 or int(report.get("task_command_attempts", -1)) < int(report.get("task_command_calls", 0)) or bool(report.get("task_command_budget_exceeded", True)):
            raise RuntimeError(f"healthy reconcile workflow bounded snapshot budget failed: {report!r}")
        for purpose, count in (report.get("task_command_by_purpose") or {}).items(): reconcile_call_purposes[str(purpose)] = max(reconcile_call_purposes.get(str(purpose), 0), int(count))
        reports.append(compact_report(report)); samples.append(time.perf_counter() - started)
    result = measure_workflow("workflow_reconcile", samples, float(budgets.get("workflow_reconcile", 3.0)))
    attach_reports(result, reports)
    return result


def reconcile_empty(
    *,
    root: Path,
    empty_data: Path,
    base_env: dict[str, str],
    repeats: int,
    budgets: dict,
    run_reconcile: Callable[..., Any],
    compact_report: Callable[[dict], dict],
    measure_workflow: Callable[..., dict],
    attach_reports: Callable[..., None],
) -> dict:
    """Measure empty reconcile startup and report overhead."""
    empty_data.mkdir(); env = dict(base_env, TASKDATA=str(empty_data)); samples = []; reports = []
    for _ in range(repeats):
        started = time.perf_counter(); proc = run_reconcile(env)
        if proc.returncode != 0: raise RuntimeError(f"empty reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
        try: report = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc: raise RuntimeError("empty reconcile workflow returned invalid JSON") from exc
        if not isinstance(report, dict) or report.get("schema") != "nautical.reconcile": raise RuntimeError("empty reconcile workflow returned an invalid report")
        if int(report.get("export_calls", 0)) > 2: raise RuntimeError("empty reconcile workflow exceeded its bounded snapshot budget")
        if float(report.get("integrity_seconds", -1.0)) < 0.0: raise RuntimeError("empty reconcile workflow omitted integrity timing")
        reports.append(compact_report(report)); samples.append(time.perf_counter() - started)
    result = measure_workflow("workflow_reconcile_empty", samples, float(budgets.get("workflow_reconcile_empty", budgets.get("workflow_reconcile", 3.0))))
    attach_reports(result, reports)
    return result


def reconcile_candidates(
    *,
    root: Path,
    base_env: dict[str, str],
    candidate_count: int,
    repeats: int,
    budgets: dict,
    candidate_tasks: list[dict],
    reconcile_command: list[str],
    compact_report: Callable[[dict], dict],
    measure_workflow: Callable[..., dict],
    attach_reports: Callable[..., None],
) -> dict:
    """Measure candidate-heavy reconcile audits and require evidence."""
    candidate_data = root / "reconcile-candidates"; candidate_data.mkdir()
    env = dict(base_env, TASKDATA=str(candidate_data))
    imported = subprocess.run(["task", "rc.hooks=off", "rc.verbose=nothing", "import"], input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in candidate_tasks), text=True, capture_output=True, env=env, timeout=30.0)
    if imported.returncode != 0: raise RuntimeError(f"reconcile candidate fixture import failed: {(imported.stderr or imported.stdout or '').strip()}")
    samples = []; reports = []
    for _ in range(repeats):
        started = time.perf_counter(); proc = subprocess.run(reconcile_command, text=True, capture_output=True, env=env, timeout=30.0)
        if proc.returncode not in (0, 1, 2): raise RuntimeError(f"candidate reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
        try: report = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc: raise RuntimeError("candidate reconcile workflow returned invalid JSON") from exc
        summary = report if isinstance(report, dict) else {}
        if int(summary.get("candidates", 0)) <= 0 and not summary.get("plans"): raise RuntimeError("candidate reconcile workflow produced no integrity candidates or plans")
        reports.append(compact_report(summary)); samples.append(time.perf_counter() - started)
    result = measure_workflow("workflow_reconcile_candidates", samples, float(budgets.get("workflow_reconcile_candidates", budgets.get("workflow_reconcile", 3.0))))
    attach_reports(result, reports)
    result["candidate_count"] = candidate_count
    return result


def reconcile_candidates_apply(
    *,
    root: Path,
    base_env: dict[str, str],
    candidate_tasks: list[dict],
    reconcile_command: list[str],
    budgets: dict,
    compact_report: Callable[[dict], dict],
    measure_workflow: Callable[..., dict],
    attach_reports: Callable[..., None],
) -> dict:
    """Measure guarded candidate reconcile application."""
    data = root / "reconcile-candidates-apply"; data.mkdir()
    env = dict(base_env, TASKDATA=str(data))
    imported = subprocess.run(["task", "rc.hooks=off", "rc.verbose=nothing", "import"], input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in candidate_tasks), text=True, capture_output=True, env=env, timeout=30.0)
    if imported.returncode != 0: raise RuntimeError(f"candidate apply fixture import failed: {(imported.stderr or imported.stdout or '').strip()}")
    started = time.perf_counter(); proc = subprocess.run([*reconcile_command, "--apply"], text=True, capture_output=True, env=env, timeout=60.0)
    if proc.returncode != 0: raise RuntimeError(f"candidate reconcile apply failed: {(proc.stderr or proc.stdout or '').strip()}")
    try: report = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc: raise RuntimeError("candidate reconcile apply returned invalid JSON") from exc
    if int(report.get("spawn", 0)) <= 0 or not report.get("applied"): raise RuntimeError("candidate reconcile apply did not create guarded successors")
    result = measure_workflow("workflow_reconcile_candidates_apply", [time.perf_counter() - started], float(budgets.get("workflow_reconcile_candidates_apply", 12.0)))
    attach_reports(result, [compact_report(report)])
    return result


def reconcile_candidates_apply_scale(
    *,
    root: Path,
    base_env: dict[str, str],
    scale_counts: tuple[int, ...],
    candidate_tasks_factory: Callable[[str, int], list[dict]],
    reconcile_command: list[str],
    compact_report: Callable[[dict], dict],
) -> dict:
    """Measure guarded apply convergence across candidate counts."""
    samples: list[float] = []; reports: list[dict] = []; rows: list[dict] = []
    for scale_count in scale_counts:
        data = root / f"reconcile-candidates-apply-{scale_count}"; data.mkdir()
        env = dict(base_env, TASKDATA=str(data)); tasks = candidate_tasks_factory(f"apply-{scale_count}", scale_count)
        imported = subprocess.run(["task", "rc.hooks=off", "rc.verbose=nothing", "import"], input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in tasks), text=True, capture_output=True, env=env, timeout=120.0)
        if imported.returncode != 0: raise RuntimeError(f"candidate apply scale fixture import failed ({scale_count}): {(imported.stderr or imported.stdout or '').strip()}")
        started = time.perf_counter(); proc = subprocess.run([*reconcile_command, "--apply"], text=True, capture_output=True, env=env, timeout=240.0); elapsed = time.perf_counter() - started
        if proc.returncode != 0: raise RuntimeError(f"candidate apply scale failed ({scale_count}): {(proc.stderr or proc.stdout or '').strip()}")
        try: report = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc: raise RuntimeError(f"candidate apply scale returned invalid JSON ({scale_count})") from exc
        if int(report.get("spawn", 0)) < scale_count or not report.get("applied"): raise RuntimeError(f"candidate apply scale did not converge ({scale_count}): {report!r}")
        samples.append(elapsed); compact = compact_report(report); reports.append(compact); rows.append({"candidate_count": scale_count, "elapsed_s": round(elapsed, 6), **compact})
    return {"name": "workflow_reconcile_candidates_apply_scale", "candidate_counts": list(scale_counts), "samples_s": samples, "rows": rows, "reconcile_reports": reports, "pass": bool(scale_counts) and all(value >= 0.0 for value in samples)}


def reconcile_report_loop(
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    repeats: int,
    timeout: float,
    budget: float,
    validate: Callable[[dict], None],
    compact_report: Callable[[dict], dict],
    measure_workflow: Callable[..., dict],
    attach_reports: Callable[..., None],
) -> dict:
    """Run a repeated reconcile report workload with scenario validation."""
    samples: list[float] = []; reports: list[dict] = []
    for _ in range(repeats):
        started = time.perf_counter(); proc = subprocess.run(command, text=True, capture_output=True, env=env, timeout=timeout)
        if proc.returncode not in (0, 1, 2): raise RuntimeError(f"{name} workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
        try: report = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError as exc: raise RuntimeError(f"{name} workflow returned invalid JSON") from exc
        summary = report if isinstance(report, dict) else {}; validate(summary)
        reports.append(compact_report(summary)); samples.append(time.perf_counter() - started)
    result = measure_workflow(name, samples, budget); attach_reports(result, reports); return result
