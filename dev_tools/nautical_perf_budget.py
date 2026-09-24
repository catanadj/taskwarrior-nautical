#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic performance budget checks for core anchor paths.

Usage:
  python3 dev_tools/nautical_perf_budget.py
  python3 dev_tools/nautical_perf_budget.py --enforce
  python3 dev_tools/nautical_perf_budget.py --json --enforce
  python3 dev_tools/nautical_perf_budget.py --extended --json
  python3 dev_tools/nautical_perf_budget.py --extended --slow-device --workflows-only --json
  python3 dev_tools/nautical_perf_budget.py --budget-file dev_tools/perf_budget.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import date, timedelta, timezone
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dev_tools.perf.telemetry import measure as _measure
from dev_tools.perf.telemetry import measure_workflow as _measure_workflow
from dev_tools.perf.reporting import attach_reconcile_reports as _attach_reconcile_reports
from dev_tools.perf.reporting import attach_timing_breakdown as _attach_timing_breakdown
from dev_tools.perf.reporting import compact_reconcile_report as _compact_reconcile_report
from dev_tools.perf.reporting import merge_task_timing_stats as _merge_task_timing_stats
from dev_tools.perf import cache_workloads as _cache_workloads
from dev_tools.perf import outbox_workloads as _outbox_workloads
from dev_tools.perf import anchor_file_workloads as _anchor_file_workloads
from dev_tools.perf import calendar_workloads as _calendar_workloads
from dev_tools.perf import resource_workloads as _resource_workloads
from dev_tools.perf import hook_workloads as _hook_workloads
from dev_tools.perf import operator_workloads as _operator_workloads
from dev_tools.perf import scheduler_workloads as _scheduler_workloads
from dev_tools.perf import workflow_workloads as _workflow_workloads

core = importlib.import_module("nautical_core")
install_runtime = importlib.import_module("nautical_core.install_runtime")
lifecycle_outbox = importlib.import_module("nautical_core.lifecycle_outbox")
task_codec = importlib.import_module("nautical_core.task_codec")
IMPORT_PROFILES: dict[str, int] = {}
RESOURCE_DETAILS: dict[str, object] = {}

_BENCH_PANEL_MODES = {
    "disabled": "quiet",
    "static": "fast",
    "live": "live",
    "minimal": "minimal",
}


def _python_subprocess_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess environment that can import the checkout package."""
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    return env


def _budget_profile_name(*, slow_device: bool) -> str:
    """Return the stable profile label recorded in benchmark reports."""
    return "termux-slow-device" if slow_device else "desktop"


def _panel_mode_config(mode: str) -> str:
    try:
        return _BENCH_PANEL_MODES[str(mode).strip().lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported benchmark panel mode: {mode!r}") from exc


def _load_budget_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read budget file '{path}': {e}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Budget file '{path}' must contain a JSON object.")
    workload = data.get("workload")
    budgets = data.get("budgets_seconds")
    if not isinstance(workload, dict) or not isinstance(budgets, dict):
        raise RuntimeError("Budget file requires 'workload' and 'budgets_seconds' objects.")
    return data


def _clear_caches() -> None:
    try:
        core._clear_all_caches()
    except Exception:
        pass


def _bench_parse_validate(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.validate_anchor_expr_strict(expr)
    return time.perf_counter() - t0


def _bench_capabilities_stage() -> float:
    """Measure the content-free capabilities composition root."""
    from nautical_core.tools.nautical_query import capabilities_payload

    started = time.perf_counter()
    payload = capabilities_payload()
    if not isinstance(payload, dict) or payload.get("status") != "ok" or not payload.get("operations"):
        raise RuntimeError("capabilities stage returned an invalid payload")
    return time.perf_counter() - started


def _bench_queue_status_stage() -> float:
    """Measure queue-status composition against an isolated empty outbox."""
    from nautical_core.tools.nautical_queue_status import status_payload

    with tempfile.TemporaryDirectory(prefix="nautical-perf-queue-status-") as td:
        taskdata = Path(td)
        started = time.perf_counter()
        payload, _budget = status_payload(taskdata, stale_after=300.0, limit=5)
        if not isinstance(payload, dict) or payload.get("taskdata") != str(taskdata):
            raise RuntimeError("queue-status stage returned an invalid payload")
        return time.perf_counter() - started


def _bench_navigator_stage() -> float:
    """Measure one bounded Navigator anchor presentation."""
    import nautical_navigator

    started = time.perf_counter()
    result = nautical_navigator._anchor_presentation_result("w:mon", count=1)
    if not result.next_dates and not result.terminal_note:
        raise RuntimeError("Navigator stage returned no preview or terminal evidence")
    return time.perf_counter() - started


def _bench_query_pagination_stage() -> float:
    """Measure scoped and whole-system query pagination without Taskwarrior I/O."""
    from types import SimpleNamespace

    from nautical_core.query_models import OccurrenceQueryRequest, QueryContractError
    from nautical_core.query_service import OccurrenceQueryService, QueryServiceError

    service = object.__new__(OccurrenceQueryService)
    service._timezone = timezone.utc
    service._scheduler_cache = {}
    service._uow = SimpleNamespace(
        mutation_epoch=0,
        context=SimpleNamespace(configuration=SimpleNamespace(fingerprint="perf-config")),
    )
    rows = tuple(SimpleNamespace(uuid=f"perf-task-{index:04d}") for index in range(128))
    scoped_request = OccurrenceQueryRequest.from_mapping(
        {"selector": {"uuids": [rows[0].uuid]}, "from": "2026-08-24", "count": 1, "max_tasks": 1}
    )
    started = time.perf_counter()
    scoped, scoped_cursor, scoped_complete = service._page_rows(rows[:1], scoped_request)
    if len(scoped) != 1 or scoped_cursor is not None or not scoped_complete:
        raise RuntimeError("scoped query pagination returned an invalid complete page")

    request = OccurrenceQueryRequest.from_mapping(
        {"selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1, "max_tasks": 16}
    )
    cursor = None
    seen: list[str] = []
    while True:
        page_request = request if cursor is None else OccurrenceQueryRequest.from_mapping(
            {"selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1, "max_tasks": 16,
             "cursor": cursor.to_dict()}
        )
        page, cursor, complete = service._page_rows(rows, page_request)
        seen.extend(row.uuid for row in page)
        if complete:
            break
    if seen != [row.uuid for row in rows]:
        raise RuntimeError("whole-system query pagination lost or reordered rows")
    empty_page, empty_cursor, empty_complete = service._page_rows((), request)
    if empty_page or empty_cursor is not None or not empty_complete:
        raise RuntimeError("empty query pagination did not complete cleanly")
    exact_rows = rows[:16]
    exact_page, exact_cursor, exact_complete = service._page_rows(exact_rows, request)
    if len(exact_page) != 16 or exact_cursor is not None or not exact_complete:
        raise RuntimeError("exact query page produced an unexpected continuation")
    plus_one_rows = rows[:17]
    first_page, continuation, first_complete = service._page_rows(plus_one_rows, request)
    if len(first_page) != 16 or continuation is None or first_complete:
        raise RuntimeError("page-size-plus-one query did not produce a continuation")
    incompatible = continuation.to_dict()
    incompatible["snapshot_id"] = "query-snapshot-invalid"
    try:
        service._page_rows(
            plus_one_rows,
            OccurrenceQueryRequest.from_mapping(
                {"selector": {"all_tasks": True}, "from": "2026-08-24",
                 "count": 1, "max_tasks": 16, "cursor": incompatible}
            ),
        )
    except QueryServiceError:
        pass
    else:
        raise RuntimeError("incompatible query cursor was accepted")
    try:
        OccurrenceQueryRequest.from_mapping(
            {"selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1, "max_tasks": 0}
        )
    except QueryContractError:
        pass
    else:
        raise RuntimeError("malformed query page limit was accepted")
    return time.perf_counter() - started


def _bench_query_unavailable_stage() -> float:
    """Measure fail-closed query handling for an unavailable snapshot."""
    from types import SimpleNamespace

    from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable
    from nautical_core.query_models import OccurrenceQueryRequest
    from nautical_core.query_service import OccurrenceQueryService

    evidence = FailureEvidence(
        TaskCommand(("task", "export"), "perf unavailable", 1.0),
        CommandFailureKind.EXECUTION_FAILURE, 1, 1, 0.001, True, "synthetic failure",
    )
    service = object.__new__(OccurrenceQueryService)
    service._uow = SimpleNamespace(repository=SimpleNamespace(
        broad_snapshot=lambda **_kwargs: Unavailable("perf unavailable", evidence),
    ))
    request = OccurrenceQueryRequest.from_mapping(
        {"selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1}
    )
    started = time.perf_counter()
    failure = service._rows_for(request)
    if not getattr(failure, "code", "") == "task_read_unavailable":
        raise RuntimeError("unavailable query snapshot did not fail closed")
    return time.perf_counter() - started


def _bench_doctor_installation_stage() -> float:
    """Measure the read-only Doctor installation composition root in isolation."""
    task_bin = shutil.which("task")
    if not task_bin:
        return 0.0
    from nautical_core.tools.nautical_doctor import JSON_SCHEMA

    with tempfile.TemporaryDirectory(prefix="nautical-perf-doctor-") as td:
        taskdata = Path(td)
        taskrc = taskdata / "taskrc"
        taskrc.write_text(
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
        config = taskdata / "config-nautical.toml"
        config.write_text('tz = "UTC"\npanel_mode = "quiet"\n', encoding="utf-8")
        env = dict(os.environ, TASKRC=str(taskrc), TASKDATA=str(taskdata),
                   NAUTICAL_CONFIG=str(config), NAUTICAL_CORE_PATH=str(ROOT),
                   NAUTICAL_TRUST_CONFIG_PATH="1", NAUTICAL_TRUST_CORE_PATH="1",
                   NAUTICAL_TRUST_TASKDATA_PATH="1", TZ="UTC")
        fixture = {"uuid": "11111111-1111-4111-8111-111111111111", "description": "doctor benchmark", "status": "pending"}
        imported = subprocess.run(
            [task_bin, f"rc.data.location={taskdata}", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input=json.dumps(fixture, ensure_ascii=False) + "\n",
            text=True, capture_output=True, env=env, timeout=30.0,
        )
        if imported.returncode != 0:
            raise RuntimeError(f"Doctor fixture import failed: {(imported.stderr or imported.stdout).strip()}")
        started = time.perf_counter()
        for mode in (("--installation-only",), ()):
            proc = subprocess.run(
                [sys.executable, str(ROOT / "nautical_core/tools/nautical_doctor.py"),
                 *mode, "--json", "--task-bin", task_bin, "--taskdata", str(taskdata)],
                text=True, capture_output=True, env=env, timeout=30.0,
            )
            try:
                payload = json.loads(proc.stdout or "")
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Doctor stage returned invalid JSON ({mode or ('full',)}): "
                    f"{(proc.stderr or proc.stdout).strip()}"
                ) from exc
            if not isinstance(payload, dict) or payload.get("schema") != JSON_SCHEMA:
                raise RuntimeError(f"Doctor stage returned an invalid envelope ({mode or ('full',)})")
        return time.perf_counter() - started


def _bench_housekeeping_stage() -> float:
    """Measure bounded housekeeping against an isolated outbox."""
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    with tempfile.TemporaryDirectory(prefix="nautical-perf-housekeeping-") as td:
        repository = LifecycleOutboxRepository(Path(td))
        opened = repository.open()
        if not opened.ok:
            raise RuntimeError(f"housekeeping outbox setup failed: {opened.reason or opened.kind.value}")
        now = time.time()
        with sqlite3.connect(str(repository.path)) as connection:
            for index in range(2):
                connection.execute(
                    "INSERT INTO lifecycle_outbox "
                    "(intent_id, work_kind, plan_json, plan_fingerprint, parent_guard_json, "
                    "configuration_fingerprint, schedule_fingerprint, lifecycle_stage, processing_state, "
                    "lease_owner, lease_expires_at, attempts, failure_json, created_at, updated_at, acknowledged_at) "
                    "VALUES (?, 'lifecycle', '{}', 'perf', '{}', 'perf', 'perf', 'finalized', ?, '', 0, 0, '', ?, ?, ?)",
                    (f"perf-housekeeping-{index}", "acknowledged", now - 100, now - 100, now - 100),
                )
        started = time.perf_counter()
        result = repository.opportunistic_housekeeping(retention_seconds=0, interval_seconds=0, size_threshold_bytes=0)
        if result.kind.value != "applied" or result.skipped or result.removed != 2:
            raise RuntimeError(f"housekeeping stage removed an unexpected number of rows: {result}")
        return time.perf_counter() - started


def _bench_repair_planner_stage() -> float:
    """Measure deterministic repair planning without applying mutations."""
    from nautical_core.chain_graph import ChainGraph
    from nautical_core.chain_integrity_context import IntegrityContext, OutboxSnapshot
    from nautical_core.chain_integrity_models import (
        ChainSnapshot, FindingSeverity, FindingStatus, IntegrityFinding, SnapshotCoverage,
    )
    from nautical_core.chain_repair_planner import IntegrityRepairPlanner

    snapshot = ChainSnapshot("perf-repair", SnapshotCoverage.COMPLETE, "perf", (), "perf-config", True)
    context = IntegrityContext(ChainGraph.from_snapshot(snapshot), OutboxSnapshot.from_records(()), "perf-config")
    finding = IntegrityFinding(
        "continuity.child_temporal_order", FindingStatus.MANUAL_REVIEW, FindingSeverity.ERROR,
        snapshot.snapshot_id, "perf-chain", ("11111111-1111-4111-8111-111111111111",),
        "child_not_after_parent", "repair benchmark finding",
    )
    started = time.perf_counter()
    result = IntegrityRepairPlanner().plan(context, (finding,))
    if result.plans or len(result.refusals) != 1:
        raise RuntimeError("repair planner did not preserve an unsafe finding as a refusal")
    return time.perf_counter() - started


def _bench_repair_application_stage() -> float:
    """Measure one guarded repair application with a typed fake executor."""
    from nautical_core.chain_integrity_application import IntegrityApplicationService
    from nautical_core.chain_integrity_models import (
        IntegrityOperation, IntegrityRepairPlan, RepairOperationKind, RepairSafety,
    )
    from nautical_core.integration_models import (
        GuardTimestamp, GuardTimestampField, MutationGuard, MutationOperation,
        MutationOutcome, MutationOutcomeKind, MutationPostcondition, MutationRequest,
    )
    task_uuid = "11111111-1111-4111-8111-111111111111"
    operation = IntegrityOperation(
        "perf-repair-operation", RepairOperationKind.METADATA_REPAIR, "perf-chain", task_uuid,
        (("chainID", "perf-chain"), ("link", 2)), ("target remains present",),
        ("metadata repaired",), (("anchor_mode", "all"),),
    )
    plan = IntegrityRepairPlan(
        "perf-repair-plan", "perf-repair-snapshot", "perf-chain", RepairSafety.SAFE,
        "missing_link", "repair one link", (operation,), "perf-config",
    )
    guard = MutationGuard(
        task_uuid, "pending", "perf-chain", 1, "w:mon",
        (GuardTimestamp(GuardTimestampField.MODIFIED, "20260829T000000Z"),), 0,
    )

    class Executor:
        def repair_metadata(self, request):
            return MutationOutcome(
                MutationOperation.METADATA_REPAIR, MutationOutcomeKind.APPLIED, request.guard,
                (MutationPostcondition.METADATA_REPAIRED,),
            )

    def request_factory(item):
        return MutationRequest.metadata_repair(
            guard, item.task_patch(), expected={"anchor_mode": "skip"},
        )

    started = time.perf_counter()
    result = IntegrityApplicationService().apply(plan, Executor(), request_factory)
    if len(result) != 1 or result[0].kind is not MutationOutcomeKind.APPLIED:
        raise RuntimeError(f"repair application stage did not apply its guarded operation: {result!r}")
    return time.perf_counter() - started


def _bench_lifecycle_staging_stage() -> float:
    return _outbox_workloads.lifecycle_staging(
        lifecycle_outbox,
        _init_empty_outbox,
        _outbox_lifecycle_fixture,
        _workflow_outbox_pending,
    )


def _bench_reconcile_snapshot_stage() -> float:
    return _operator_workloads.reconcile_snapshot()


def _bench_queue_stale_stage() -> float:
    return _operator_workloads.queue_stale(
        lifecycle_outbox, _init_empty_outbox, _outbox_lifecycle_fixture, _stage_workflow_plans,
    )


def _bench_operator_interrupted_stage() -> float:
    return _operator_workloads.operator_interrupted(
        lifecycle_outbox, _init_empty_outbox, _outbox_lifecycle_fixture, _stage_workflow_plans,
    )


def _bench_exit_probe_fast_paths_stage() -> float:
    return _operator_workloads.exit_probe()


def _bench_operator_scope_matrix_stage() -> float:
    return _operator_workloads.operator_scope()


def _bench_operator_failure_matrix_stage() -> float:
    from nautical_core.tools.nautical_reconcile import configuration_verification
    return _operator_workloads.operator_failure_matrix(
        _bench_query_pagination_stage,
        _bench_query_unavailable_stage,
        _bench_repair_planner_stage,
        _bench_queue_stale_stage,
        configuration_verification,
    )


def _bench_describe_expr(exprs: list[str], rounds: int) -> float:
    return _scheduler_workloads.describe(core, exprs, rounds, _clear_caches)


def _bench_next_after(exprs: list[str], rounds: int) -> float:
    return _scheduler_workloads.next_after(core, exprs, rounds, _clear_caches)


def _bench_scheduler_decisions(exprs: list[str]) -> float:
    return _scheduler_workloads.decisions(core, exprs, task_codec, RESOURCE_DETAILS)


def _bench_task_codec(rounds: int) -> float:
    return _resource_workloads.task_codec(task_codec, rounds)


def _bench_task_immutability(rounds: int) -> float:
    return _resource_workloads.task_immutability(task_codec, rounds)


def _bench_task_snapshot_reuse(rounds: int, row_count: int = 1000) -> float:
    return _resource_workloads.task_snapshot_reuse(task_codec, rounds, row_count)


def _bench_task_resource_limits(rounds: int) -> float:
    return _resource_workloads.task_resource_limits(task_codec, rounds)


def _bench_task_snapshot_memory(counts: Sequence[int]) -> float:
    return _resource_workloads.task_snapshot_memory(task_codec, counts)


def _bench_build_hints(
    exprs: list[str],
    rounds: int,
    *,
    mode: str = "warm",
    include_per_year: bool = True,
) -> float:
    """Measure hint construction with an explicit persistent-cache state."""
    with _perf_cache_context():
        saved_load = core.cache_load
        saved_save = core.cache_save
        counts = {"hits": 0, "misses": 0, "saves": 0}

        def counted_load(*args, **kwargs):
            value = saved_load(*args, **kwargs)
            counts["hits" if value is not None else "misses"] += 1
            return value

        def counted_save(*args, **kwargs):
            result = saved_save(*args, **kwargs)
            if result:
                counts["saves"] += 1
            return result

        core.cache_load = counted_load
        core.cache_save = counted_save
        try:
            expression_elapsed = {f"{index}:{expr}": 0.0 for index, expr in enumerate(exprs)}
            if mode == "cold":
                root = Path(core.ANCHOR_CACHE_DIR_OVERRIDE)
                setup_started = time.perf_counter()
                sample_dirs = [root / f"cold-{sample_index}" for sample_index in range(max(1, rounds))]
                for sample_dir in sample_dirs:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                setup_elapsed = time.perf_counter() - setup_started
                measured_total = 0.0
                for sample_dir in sample_dirs:
                    core.ANCHOR_CACHE_DIR_OVERRIDE = str(sample_dir)
                    core._CACHE_DIR = None
                    _clear_caches()
                    operation_started = time.perf_counter()
                    for index, expr in enumerate(exprs):
                        expression_started = time.perf_counter()
                        core.build_and_cache_hints(expr, "skip", include_per_year=include_per_year)
                        expression_elapsed[f"{index}:{expr}"] += time.perf_counter() - expression_started
                    measured_total += time.perf_counter() - operation_started
                metric_name = f"build_hints_{'next_only_' if not include_per_year else ''}cold"
                RESOURCE_DETAILS[metric_name] = {
                    "per_expression_seconds": expression_elapsed,
                    "setup_seconds": setup_elapsed,
                }
                if counts["hits"] or not counts["misses"]:
                    raise RuntimeError(f"cold hint benchmark observed unexpected cache state: {counts}")
                return measured_total
            if mode != "warm":
                raise ValueError(f"unknown hint benchmark mode: {mode}")
            core.cache_load = saved_load
            core.cache_save = saved_save
            _clear_caches()
            for expr in exprs:
                core.build_and_cache_hints(expr, "skip", include_per_year=include_per_year)
            core.cache_load = counted_load
            core.cache_save = counted_save
            counts = {"hits": 0, "misses": 0, "saves": 0}
            _clear_caches()
            t0 = time.perf_counter()
            for _ in range(max(1, rounds)):
                for index, expr in enumerate(exprs):
                    expression_started = time.perf_counter()
                    core.build_and_cache_hints(expr, "skip", include_per_year=include_per_year)
                    expression_elapsed[f"{index}:{expr}"] += time.perf_counter() - expression_started
            metric_name = f"build_hints_{'next_only_' if not include_per_year else ''}warm"
            RESOURCE_DETAILS[metric_name] = {"per_expression_seconds": expression_elapsed}
            if counts["misses"] or not counts["hits"] or counts["saves"]:
                raise RuntimeError(f"warm hint benchmark observed unexpected cache state: {counts}")
            return time.perf_counter() - t0
        finally:
            core.cache_load = saved_load
            core.cache_save = saved_save


def _bench_cache_key_hot(exprs: list[str], rounds: int) -> float:
    return _cache_workloads.cache_key_hot(core, _clear_caches, exprs, rounds)


def _perf_cache_context():
    return _cache_workloads.cache_context(core, _clear_caches)


def _cache_payload(expr: str, idx: int) -> dict:
    return _cache_workloads.cache_payload(expr, idx)


def _bench_cache_save(exprs: list[str], rounds: int) -> float:
    return _cache_workloads.cache_save(core, _clear_caches, exprs, rounds)


def _bench_cache_load_hot(exprs: list[str], rounds: int) -> float:
    return _cache_workloads.cache_load_hot(core, _clear_caches, exprs, rounds)


def _bench_outbox_schema_hot(rounds: int) -> float:
    return _outbox_workloads.schema_hot(lifecycle_outbox, rounds)


def _bench_outbox_schema_cold(rounds: int) -> float:
    return _outbox_workloads.schema_cold(ROOT, rounds)


def _bench_cold_import(kind: str, rounds: int) -> float:
    """Measure a fresh-process import without reusing this benchmark process."""
    if kind == "core":
        script = "import nautical_core; print('__NAUTICAL_IMPORT_COUNT__=' + str(len(__import__('sys').modules)))"
    elif kind == "modify_impl":
        script = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('perf_modify_impl', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "print('__NAUTICAL_IMPORT_COUNT__=' + str(len(sys.modules)))"
        )
    else:
        raise ValueError(f"unknown cold import benchmark kind: {kind}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    command = [sys.executable, "-c", script]
    if kind == "modify_impl":
        command.append(str(ROOT / "nautical_core" / "hooks" / "modify_impl.py"))
    started = time.perf_counter()
    for _ in range(max(1, rounds)):
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=30.0,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"cold {kind} import failed: {proc.stderr.strip()}")
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith("__NAUTICAL_IMPORT_COUNT__="):
                try:
                    IMPORT_PROFILES[kind] = int(line.split("=", 1)[1])
                except ValueError:
                    pass
                break
    return time.perf_counter() - started


def _bench_anchor_file_provider(rounds: int) -> float:
    return _anchor_file_workloads.provider(rounds)


def _bench_anchor_file_batch_provider(rounds: int) -> float:
    return _anchor_file_workloads.batch_provider(rounds)


def _bench_large_anchor_file_provider(
    rounds: int,
    *,
    row_count: int = 5000,
    mode: str = "hot",
    business_day_only: bool = False,
) -> float:
    return _anchor_file_workloads.large_provider(
        rounds,
        row_count=row_count,
        mode=mode,
        business_day_only=business_day_only,
    )


def _bench_business_calendar_omissions(rounds: int) -> float:
    return _calendar_workloads.large_omissions(core, rounds)


def _bench_astronomy_provider(rounds: int, *, event: str) -> float | None:
    """Measure deterministic astronomy resolution when Astral is installed."""
    if importlib.util.find_spec("astral") is None:
        return None
    astronomy = importlib.import_module("nautical_core.astronomy")
    config = {
        "default_location": "perf",
        "locations": {
            "perf": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "elevation": 10,
                "timezone": "UTC",
            }
        },
    }
    started = time.perf_counter()
    for index in range(max(1, int(rounds))):
        reference = date(2026, 1, 1) + timedelta(days=index % 60)
        phase_day = astronomy.resolve_phase_date(
            "full",
            reference,
            config=config,
            horizon_days=60,
        )
        resolved = astronomy.resolve_event(event, phase_day, config=config)
        if resolved.tzinfo is None:
            raise RuntimeError("astronomy benchmark returned a naive datetime")
    return time.perf_counter() - started


def _bench_native_until_reconcile(rounds: int, *, apply: bool) -> float:
    """Measure native-until audit/repair on independent valid fixtures."""
    with tempfile.TemporaryDirectory(prefix="nautical-perf-native-until-") as td:
        root = Path(td)
        config_path = root / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\npanel_mode = "minimal"\n', encoding="utf-8")
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
        base_env = dict(os.environ)
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                # The benchmark deliberately supplies isolated temporary
                # Taskdata directories; allow the launcher to use them.
                "NAUTICAL_TRUST_TASKDATA_PATH": "1",
                "TASKRC": str(taskrc_path),
                "TZ": "UTC",
            }
        )
        started = time.perf_counter()
        for sample_index in range(max(1, int(rounds))):
            taskdata = root / f"native-until-{sample_index}"
            taskdata.mkdir()
            parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"native-until/{sample_index}/parent"))
            child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"native-until/{sample_index}/child"))
            rows = [
                {
                    "uuid": parent_uuid,
                    # Native-until predecessor reads intentionally inspect
                    # completed/deleted history, so model the prior link as
                    # completed rather than leaving it active.
                    "status": "completed",
                    "description": "Native-until benchmark predecessor",
                    "cp": "P1D",
                    "chain": "on",
                    "chainID": f"native-until-{sample_index}",
                    "link": 1,
                    "nextLink": child_uuid[:8],
                    "due": "20270101T090000Z",
                    "until": "20270101T200000Z",
                },
                {
                    "uuid": child_uuid,
                    "status": "pending",
                    "description": "Native-until benchmark invalid child",
                    "cp": "P1D",
                    "chain": "on",
                    "chainID": f"native-until-{sample_index}",
                    "link": 2,
                    "prevLink": parent_uuid[:8],
                    "due": "20270102T090000Z",
                    "until": "20270101T200000Z",
                },
            ]
            env = dict(base_env, TASKDATA=str(taskdata))
            imported = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                text=True,
                capture_output=True,
                env=env,
                timeout=30.0,
            )
            if imported.returncode != 0:
                raise RuntimeError(f"native-until fixture import failed: {(imported.stderr or imported.stdout).strip()}")
            command = [sys.executable, str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"), "--json"]
            if apply:
                command.append("--apply")
            repaired = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=env,
                timeout=30.0,
            )
            if repaired.returncode != 0:
                raise RuntimeError(f"native-until reconcile failed: {(repaired.stderr or repaired.stdout).strip()}")
            try:
                summary = json.loads(repaired.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("native-until reconcile returned invalid JSON") from exc
            repairs = summary.get("native_until_repairs") if isinstance(summary, dict) else None
            if not isinstance(repairs, list) or len(repairs) != 1:
                raise RuntimeError("native-until benchmark did not inspect exactly one invalid child")
            item = repairs[0]
            if item.get("action") != "repair_until" or bool(item.get("applied")) != apply:
                raise RuntimeError(f"native-until benchmark did not take the expected path: {item}")
            if apply:
                exported = subprocess.run(
                    [
                        "task",
                        "rc.hooks=off",
                        "rc.json.array=1",
                        "rc.verbose=nothing",
                        "rc.color=off",
                        f"uuid:{child_uuid}",
                        "export",
                    ],
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=30.0,
                )
                if exported.returncode != 0:
                    raise RuntimeError("native-until verification export failed")
                rows_after = json.loads(exported.stdout or "[]")
                if not isinstance(rows_after, list) or len(rows_after) != 1:
                    raise RuntimeError("native-until verification export returned an unexpected task count")
                if str(rows_after[0].get("until") or "") != "20270102T200000Z":
                    raise RuntimeError("native-until apply benchmark did not persist the repaired endpoint")
        return time.perf_counter() - started


def _strict_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(text)
    except Exception as exc:
        raise RuntimeError(f"hook stdout is not valid JSON: {exc}") from exc
    if text[end:].strip() or not isinstance(obj, dict):
        raise RuntimeError("hook stdout must contain exactly one JSON object")
    return obj


def _run_hook_timed(
    hook_path: Path,
    *,
    input_text: str,
    env: dict[str, str],
    expected_task: dict | None,
) -> float:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(hook_path)],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=15.0,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{hook_path.name} failed with exit {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    if expected_task is None:
        if (proc.stdout or "").strip():
            raise RuntimeError(f"{hook_path.name} wrote unexpected stdout")
    else:
        actual = _strict_json_object(proc.stdout or "")
        if actual != expected_task:
            raise RuntimeError(f"{hook_path.name} changed the plain passthrough task")
    return elapsed


def _init_empty_outbox(taskdata: Path) -> None:
    result = lifecycle_outbox.LifecycleOutboxRepository(taskdata).open()
    if not result.ok:
        raise RuntimeError(f"outbox benchmark setup failed: {result.reason or result.kind.value}")


def _measure_hook_fast_path(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    max_ratio: float,
) -> dict:
    fast_env = dict(base_env)
    fast_env.pop("NAUTICAL_BENCH_FORCE_FULL", None)
    full_env = dict(base_env)
    full_env["NAUTICAL_BENCH_FORCE_FULL"] = "1"
    # Full on-exit execution can create durable outbox state.  Keep the two
    # measurements isolated so a forced-full sample cannot invalidate later
    # fast-path probes in the same case.
    taskdata = str(base_env.get("TASKDATA") or "").strip()
    if taskdata:
        root = Path(taskdata).parent
        fast_env["TASKDATA"] = str(root / f"{Path(taskdata).name}-{name}-fast")
        full_env["TASKDATA"] = str(root / f"{Path(taskdata).name}-{name}-full")
        Path(fast_env["TASKDATA"]).mkdir(parents=True, exist_ok=True)
        Path(full_env["TASKDATA"]).mkdir(parents=True, exist_ok=True)

    _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
    _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)

    fast_samples: list[float] = []
    full_samples: list[float] = []
    for index in range(max(1, int(repeats))):
        if index % 2 == 0:
            fast_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
            )
            full_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)
            )
        else:
            full_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)
            )
            fast_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
            )

    fast_samples.sort()
    full_samples.sort()
    fast_median = float(statistics.median(fast_samples))
    full_median = float(statistics.median(full_samples))
    ratio = fast_median / full_median if full_median > 0.0 else 1.0
    return {
        "name": name,
        "samples_s": fast_samples,
        "min_s": fast_samples[0],
        "median_s": fast_median,
        "max_s": fast_samples[-1],
        "full_samples_s": full_samples,
        "full_median_s": full_median,
        "fast_to_full_ratio": ratio,
        "max_ratio": float(max_ratio),
        "budget_s": 0.0,
        "pass": ratio <= float(max_ratio),
    }


def _measure_managed_hook_latency(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    baseline_median_s: float,
    max_ratio: float,
) -> dict:
    env = dict(base_env)
    env.pop("NAUTICAL_CORE_PATH", None)
    env.pop("NAUTICAL_TRUST_CORE_PATH", None)
    env.pop("NAUTICAL_BENCH_FORCE_FULL", None)

    _run_hook_timed(hook_path, input_text=input_text, env=env, expected_task=expected_task)
    samples = sorted(
        _run_hook_timed(hook_path, input_text=input_text, env=env, expected_task=expected_task)
        for _ in range(max(1, int(repeats)))
    )
    median_s = float(statistics.median(samples))
    ratio = median_s / baseline_median_s if baseline_median_s > 0.0 else 1.0
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": median_s,
        "max_s": samples[-1],
        "baseline_median_s": float(baseline_median_s),
        "managed_to_source_ratio": ratio,
        "max_ratio": float(max_ratio),
        "budget_s": 0.0,
        "pass": ratio <= float(max_ratio),
    }


def _measure_staged_hook_latency(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    baseline_median_s: float,
    max_ratio: float,
) -> dict:
    """Measure a staged source tree against the source-layout median."""
    result = _measure_hook_fast_path(
        name,
        hook_path,
        input_text=input_text,
        expected_task=expected_task,
        base_env=base_env,
        repeats=repeats,
        max_ratio=max_ratio,
    )
    staged_median_s = float(result["median_s"])
    ratio = staged_median_s / baseline_median_s if baseline_median_s > 0.0 else 1.0
    result.update(
        {
            "baseline_median_s": float(baseline_median_s),
            "staged_to_source_ratio": ratio,
            "max_ratio": float(max_ratio),
            "pass": bool(result.get("pass", True)) and ratio <= float(max_ratio),
        }
    )
    return result


def _bench_hook_fast_paths(cfg: dict, *, panel_mode: str = "minimal") -> dict[str, dict]:
    return _hook_workloads.fast_paths(
        cfg,
        panel_mode=panel_mode,
        root=ROOT,
        install_runtime=install_runtime,
        python_subprocess_env=_python_subprocess_env,
        panel_mode_config=_panel_mode_config,
        measure_hook_fast_path=_measure_hook_fast_path,
        measure_managed_hook_latency=_measure_managed_hook_latency,
        measure_staged_hook_latency=_measure_staged_hook_latency,
    )


def _run_workflow_hook(hook_path: Path, *, input_text: str, env: dict[str, str], expect_output: bool) -> float:
    elapsed, _result, _stderr = _run_workflow_hook_result(
        hook_path,
        input_text=input_text,
        env=env,
        expect_output=expect_output,
    )
    return elapsed


def _run_workflow_hook_result(
    hook_path: Path,
    *,
    input_text: str,
    env: dict[str, str],
    expect_output: bool,
) -> tuple[float, dict | None, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(hook_path)],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{hook_path.name} workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
    result = None
    if expect_output:
        result = _strict_json_object(proc.stdout or "")
    elif (proc.stdout or "").strip():
        raise RuntimeError(f"{hook_path.name} workflow wrote unexpected stdout")
    return elapsed, result, proc.stderr or ""


def _read_exit_task_call_stats(path: Path) -> dict[str, int]:
    """Read benchmark-only command counters without enabling hook diagnostics."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"queue drain benchmark stats were unavailable: {exc}") from exc
    stats = payload.get("task_stats") if isinstance(payload, dict) else None
    if not isinstance(stats, dict):
        raise RuntimeError("queue drain benchmark stats did not contain task_stats")
    return {
        str(key): int(value)
        for key, value in stats.items()
        if str(key).startswith("run_task_calls") or str(key) == "task_read_rows"
    }


def _read_exit_task_timing_stats(path: Path) -> dict[str, float]:
    """Read benchmark-only timing breakdown emitted by on-exit."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"queue drain benchmark stats were unavailable: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("queue drain benchmark stats were not an object")
    task_stats = payload.get("task_stats")
    startup_stats = payload.get("startup_stats")
    drain_stats = payload.get("drain_stats")
    if not all(isinstance(value, dict) for value in (task_stats, startup_stats, drain_stats)):
        raise RuntimeError("queue drain benchmark stats omitted timing sections")

    def number(section: dict, key: str) -> float:
        try:
            return max(0.0, float(section.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    return {
        "run_task_seconds": number(task_stats, "run_task_seconds"),
        "startup_import_ms": number(startup_stats, "startup_import_ms"),
        "startup_module_ms": number(startup_stats, "startup_module_ms"),
        "startup_request_ms": number(startup_stats, "startup_request_ms"),
        "startup_total_ms": number(startup_stats, "startup_total_ms"),
        "drain_ms": number(drain_stats, "drain_ms"),
        "presentation_ms": number(payload, "presentation_ms"),
    }


def _read_exit_outbox_stats(path: Path) -> dict[str, float]:
    """Read benchmark-only outbox counters and timing."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"queue drain benchmark stats were unavailable: {exc}") from exc
    metrics = payload.get("outbox_metrics") if isinstance(payload, dict) else None
    if not isinstance(metrics, dict):
        raise RuntimeError("queue drain benchmark stats omitted outbox metrics")
    result: dict[str, float] = {}
    for key, value in metrics.items():
        if not str(key).startswith("outbox_"):
            continue
        try:
            result[str(key)] = max(0.0, float(value or 0.0))
        except (TypeError, ValueError):
            result[str(key)] = 0.0
    return result


def _apply_reconcile_budgets(result: dict, budget: dict) -> None:
    """Enforce explicit reconcile export/call/row ceilings independently of time."""
    reports = result.get("reconcile_reports")
    if not isinstance(budget, dict) or not isinstance(reports, list) or not reports:
        return
    checks: dict[str, dict[str, object]] = {}
    for key, raw_limit in budget.items():
        if key not in {"export_calls", "export_rows", "task_command_calls", "task_command_attempts"}:
            continue
        try:
            limit = int(raw_limit)
            observed = max(int(report.get(key, 0) or 0) for report in reports if isinstance(report, dict))
        except (TypeError, ValueError):
            continue
        checks[key] = {"max_observed": observed, "budget": limit, "pass": observed <= limit}
    if checks:
        result["reconcile_budget"] = checks
        result["pass"] = bool(result.get("pass", True)) and all(
            bool(item["pass"]) for item in checks.values()
        )


def _apply_task_call_budgets(result: dict, samples: list[dict[str, int]], budget: dict) -> None:
    """Attach and enforce per-workflow Taskwarrior call-count budgets."""
    if not isinstance(budget, dict) or not samples:
        return
    maxima = {
        key: max(int(sample.get(key, 0)) for sample in samples)
        for key in budget
        if all(isinstance(sample, dict) for sample in samples)
    }
    checks = {
        key: {
            "max_observed": value,
            "budget": int(budget[key]),
            "pass": value <= int(budget[key]),
        }
        for key, value in maxima.items()
    }
    result["task_call_budget"] = checks
    result["pass"] = bool(result.get("pass", True)) and all(item["pass"] for item in checks.values())


def _apply_component_budgets(result: dict, timing_samples: list[dict[str, float]], budget: dict) -> None:
    """Enforce independent timing budgets for expensive workflow components."""
    if not isinstance(budget, dict) or not timing_samples:
        return
    checks: dict[str, dict[str, object]] = {}
    for key, raw_limit in budget.items():
        try:
            limit = float(raw_limit)
            observed = max(float(sample.get(key, 0.0) or 0.0) for sample in timing_samples)
        except (TypeError, ValueError):
            continue
        checks[str(key)] = {
            "max_observed_s": observed,
            "budget_s": limit,
            "pass": limit <= 0.0 or observed <= limit,
        }
    if checks:
        result["component_budget"] = checks
        result["pass"] = bool(result.get("pass", True)) and all(
            bool(item["pass"]) for item in checks.values()
        )


def _apply_resource_budgets(result: dict, name: str, budgets: dict) -> None:
    """Enforce bounded memory/import resources when the benchmark exposes them."""
    if not isinstance(budgets, dict):
        return
    checks: dict[str, dict[str, object]] = {}
    if name == "task_snapshot_memory":
        measurements = RESOURCE_DETAILS.get(name, {})
        for row_count, values in measurements.items():
            if not isinstance(values, dict):
                continue
            key = f"peak_bytes_{row_count}"
            if key not in budgets:
                continue
            observed = int(values.get("peak_bytes", 0) or 0)
            limit = int(budgets[key])
            checks[key] = {"max_observed": observed, "budget": limit, "pass": observed <= limit}
    elif name in {"cold_core_import", "cold_modify_impl_import"} and "module_count" in budgets:
        observed = int(result.get("module_count", 0) or 0)
        limit = int(budgets["module_count"])
        checks["module_count"] = {"max_observed": observed, "budget": limit, "pass": observed <= limit}
    elif name == "scheduler_decisions" and "decision_count" in budgets:
        values = RESOURCE_DETAILS.get(name, {})
        if isinstance(values, dict):
            observed = max((int(value) for value in values.values()), default=0)
            limit = int(budgets["decision_count"])
            checks["decision_count"] = {"max_observed": observed, "budget": limit, "pass": observed <= limit}
    if checks:
        result["resource_budget"] = checks
        result["pass"] = bool(result.get("pass", True)) and all(
            bool(item["pass"]) for item in checks.values()
        )


def _apply_outbox_budgets(result: dict, samples: list[dict[str, float]], budget: dict) -> None:
    """Enforce SQLite/outbox health budgets from exit diagnostics."""
    if not isinstance(budget, dict) or not samples:
        return
    checks: dict[str, dict[str, object]] = {}
    for key, raw_limit in budget.items():
        try:
            limit = float(raw_limit)
            observed = max(float(sample.get(key, 0.0) or 0.0) for sample in samples)
        except (TypeError, ValueError):
            continue
        checks[str(key)] = {"max_observed": observed, "budget": limit, "pass": observed <= limit}
    if checks:
        result["sqlite_budget"] = checks
        result["pass"] = bool(result.get("pass", True)) and all(
            bool(item["pass"]) for item in checks.values()
        )


def _workflow_outbox_pending(taskdata: Path) -> list[dict]:
    """Read active lifecycle outbox records for benchmark mutation assertions."""
    result, status = lifecycle_outbox.LifecycleOutboxRepository(taskdata).status(limit=100)
    if not result.ok:
        raise RuntimeError(f"workflow outbox status failed: {result.reason or result.kind.value}")
    return [
        record
        for record in status.get("records", [])
        if isinstance(record, dict) and record.get("state") in {"ready", "claimed", "retry"}
    ]


def _stage_workflow_plans(
    taskdata: Path,
    plans: list,
    *,
    configuration_fingerprint: str,
    schedule_fingerprint: str,
) -> None:
    repository = lifecycle_outbox.LifecycleOutboxRepository(taskdata)
    for plan in plans:
        result = repository.enqueue(
            plan,
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )
        if not result.ok:
            raise RuntimeError(f"workflow outbox enqueue failed: {result.reason or result.kind.value}")


def _bind_workflow_plans_to_parents(plans: list, rows: list[dict]) -> list:
    """Bind benchmark plans to the guards Taskwarrior assigned on import."""
    from nautical_core.lifecycle_models import LifecyclePlan, ParentGuard, recurrence_fingerprint
    from nautical_core.task_models import NauticalTask, TaskDraft
    from nautical_core.task_codec import DEFAULT_TASK_CODEC

    def draft_for(row: dict) -> TaskDraft:
        task = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(row, source_query="perf:workflow-plan")
        )
        return TaskDraft.from_task(task)

    by_uuid = {str(row.get("uuid") or "").strip(): row for row in rows if isinstance(row, dict)}
    bound = []
    for plan in plans:
        parent = by_uuid.get(plan.identity.parent_uuid)
        if parent is None:
            raise RuntimeError(f"outbox fixture parent is missing: {plan.identity.parent_uuid}")
        guard = ParentGuard(
            status=str(parent.get("status") or ""),
            chain=str(parent.get("chain") or ""),
            chain_id=str(parent.get("chainID") or ""),
            link=int(parent.get("link") or 0),
            recurrence_fingerprint=recurrence_fingerprint(parent),
            modified=str(parent.get("modified") or ""),
        )
        bound.append(
            LifecyclePlan.from_draft(
                identity=plan.identity,
                action=plan.action,
                parent_guard=guard,
                draft=draft_for(plan.child_dict()),
                parent_patch=plan.parent_patch_dict(),
                expected_postconditions=plan.expected_postconditions,
                max_attempts=plan.max_attempts,
                stage=plan.stage,
            )
        )
    return bound


def _outbox_lifecycle_fixture(prefix: str, sample_index: int, count: int = 8) -> tuple[list[dict], list]:
    """Create independent typed lifecycle plans for durable outbox recovery tests."""
    from nautical_core.lifecycle_models import (
        ExecutionStage,
        LifecycleAction,
        LifecycleEvent,
        LifecycleIdentity,
        LifecyclePlan,
        ParentGuard,
    )
    from nautical_core.task_models import NauticalTask, TaskDraft
    from nautical_core.task_codec import DEFAULT_TASK_CODEC

    parents: list[dict] = []
    plans: list = []
    for index in range(count):
        parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{prefix}/{sample_index}/parent/{index}"))
        child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{prefix}/{sample_index}/child/{index}"))
        parent_link = index + 1
        child_link = parent_link + 1
        chain_id = f"queue-{prefix}-chain-{index}"
        parent = {
            "uuid": parent_uuid,
            "status": "completed",
            "description": f"Queue {prefix} benchmark parent",
            "chain": "on",
            "chainID": chain_id,
            "link": str(parent_link),
            "cp": "P1D",
            "due": "20260101T090000Z",
        }
        child = {
            "uuid": child_uuid,
            "status": "pending",
            "description": f"Queue {prefix} benchmark child",
            "chain": "on",
            "chainID": chain_id,
            "link": child_link,
            "prevLink": parent_uuid[:8],
            "cp": "P1D",
            "due": "20260102T090000Z",
        }
        guard = {"status": "completed", "chain": "on", "chainID": chain_id, "link": str(parent_link)}
        child_task = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(child, source_query="perf:workflow-plan")
        )
        plan = LifecyclePlan.from_draft(
            identity=LifecycleIdentity(
                chain_id, parent_uuid, parent_link, child_link, LifecycleEvent.COMPLETE
            ),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", chain_id, parent_link),
            draft=TaskDraft.from_task(child_task),
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
            stage=ExecutionStage.PERSISTED,
        )
        parents.append(parent)
        plans.append(plan)
    return parents, plans


def _reconcile_candidate_tasks(prefix: str, count: int, *, legacy_chain_ids: bool = False) -> list[dict]:
    """Create independent completed roots for reconcile candidate scaling."""
    tasks = []
    for index in range(count):
        root_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{prefix}/{index}"))
        tasks.append(
            {
            "uuid": root_uuid,
            "status": "completed",
            "description": f"Reconcile candidate benchmark {index}",
            "cp": "P1D",
            "chain": "on",
            "chainID": (
                f"reconcile-candidate-{prefix}-{index}"
                if legacy_chain_ids
                else root_uuid[:8]
            ),
            "link": 1,
            "due": "20260101T090000Z",
            }
        )
    return tasks


def _merge_task_call_stats(*stats: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in stats:
        for key, value in item.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _completion_fixture(kind: str, sample_index: int, *, nonfinal: bool, mode: str) -> dict:
    """Build independent, deterministic identities for one completion sample."""
    key = f"nautical-perf/{kind}/{mode}/{sample_index}"
    parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, key + "/parent"))
    chain_id = f"{kind}-perf-{mode}-{sample_index:04d}"
    limit = 2 if nonfinal else 1
    if kind == "cp":
        return {
            "uuid": parent_uuid,
            "status": "pending",
            "description": f"CP completion benchmark {mode} {sample_index}",
            "cp": "P1D",
            "chain": "on",
            "chainID": chain_id,
            "link": 1,
            "chainMax": limit,
            "due": "20260101T090000Z",
            "modified": "20260101T090000Z",
        }
    return {
        "uuid": parent_uuid,
        "status": "pending",
        "description": f"Anchor completion benchmark {mode} {sample_index}",
        "anchor": "w:mon@t=09:00",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": chain_id,
        "link": 1,
        "chainMax": limit,
        "due": "20260105T090000Z",
        "modified": "20260105T090000Z",
    }


def _import_existing_completion_child(parent: dict, *, env: dict[str, str]) -> None:
    """Seed an existing next link for the idempotent completion benchmark."""
    child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(parent["uuid"]) + "/child"))
    child = {
        "uuid": child_uuid,
        "status": "pending",
        "description": parent["description"],
        "chain": "on",
        "chainID": parent["chainID"],
        # Keep link textual so Taskwarrior's JSON export compares exactly with
        # the completion lookup's requested link number.
        "link": "2",
        "prevLink": str(parent["uuid"])[:8],
        "due": "20270102T090000Z" if parent.get("cp") else "20270112T090000Z",
    }
    if parent.get("cp"):
        child["cp"] = parent["cp"]
    else:
        child["anchor"] = parent["anchor"]
        child["anchor_mode"] = parent["anchor_mode"]
    proc = subprocess.run(
        ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
        input=json.dumps(child, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"idempotent completion fixture import failed: {(proc.stderr or proc.stdout or '').strip()}")


def _import_workflow_rows(rows: Sequence[dict], *, env: dict[str, str]) -> None:
    """Seed synthetic lifecycle parents before exercising full hooks.

    Full completion paths intentionally perform authoritative Taskwarrior
    reads.  Keeping the fixture import explicit makes the benchmark measure
    lifecycle work rather than the fail-closed response to an empty data dir.
    """
    proc = subprocess.run(
        ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
        input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"workflow parent fixture import failed: {(proc.stderr or proc.stdout or '').strip()}")


def _run_workflow_workloads(
    cfg: dict,
    *,
    slow_device: bool = False,
    panel_mode: str = "minimal",
) -> dict[str, dict]:
    """Compose CLI services and dispatch the isolated workflow scenarios."""
    deps = _workflow_workloads.WorkflowDependencies(
        RESOURCE_DETAILS=RESOURCE_DETAILS,
        ROOT=ROOT,
        _apply_component_budgets=_apply_component_budgets,
        _apply_outbox_budgets=_apply_outbox_budgets,
        _apply_reconcile_budgets=_apply_reconcile_budgets,
        _apply_task_call_budgets=_apply_task_call_budgets,
        _attach_reconcile_reports=_attach_reconcile_reports,
        _attach_timing_breakdown=_attach_timing_breakdown,
        _bind_workflow_plans_to_parents=_bind_workflow_plans_to_parents,
        _compact_reconcile_report=_compact_reconcile_report,
        _completion_fixture=_completion_fixture,
        _import_existing_completion_child=_import_existing_completion_child,
        _import_workflow_rows=_import_workflow_rows,
        _init_empty_outbox=_init_empty_outbox,
        _measure_workflow=_measure_workflow,
        _merge_task_call_stats=_merge_task_call_stats,
        _merge_task_timing_stats=_merge_task_timing_stats,
        _outbox_lifecycle_fixture=_outbox_lifecycle_fixture,
        _panel_mode_config=_panel_mode_config,
        _python_subprocess_env=_python_subprocess_env,
        _read_exit_outbox_stats=_read_exit_outbox_stats,
        _read_exit_task_call_stats=_read_exit_task_call_stats,
        _read_exit_task_timing_stats=_read_exit_task_timing_stats,
        _reconcile_candidate_tasks=_reconcile_candidate_tasks,
        _run_workflow_hook_result=_run_workflow_hook_result,
        _stage_workflow_plans=_stage_workflow_plans,
        _workflow_outbox_pending=_workflow_outbox_pending,
        lifecycle_outbox=lifecycle_outbox,
        workloads=_workflow_workloads,
    )
    return _workflow_workloads.run_scenarios(
        deps, cfg, slow_device=slow_device, panel_mode=panel_mode
    )


def _bench_expensive_workflows(
    cfg: dict,
    *,
    slow_device: bool = False,
    panel_mode: str = "minimal",
) -> dict[str, dict]:
    return _workflow_workloads.expensive_workflows(
        _run_workflow_workloads,
        cfg,
        slow_device=slow_device,
        panel_mode=panel_mode,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-file", default=str(HERE / "perf_budget.json"))
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--enforce", action="store_true", help="fail non-zero if any budget is exceeded")
    ap.add_argument(
        "--trace-memory",
        action="store_true",
        help="enable tracemalloc while measuring generic benchmark operations",
    )
    ap.add_argument(
        "--extended",
        action="store_true",
        help="run large-file, astronomy, omission, and native-until benchmarks",
    )
    ap.add_argument(
        "--slow-device",
        action="store_true",
        help="use the slower-device budgets for hint and extended benchmarks",
    )
    ap.add_argument(
        "--workflows-only",
        action="store_true",
        help="run only expensive completion, queue, and reconcile workflows",
    )
    ap.add_argument(
        "--panel-mode",
        choices=tuple(_BENCH_PANEL_MODES),
        default="minimal",
        help="panel configuration for hook/workflow baselines (disabled, static, live, or minimal)",
    )
    args = ap.parse_args()

    cfg = _load_budget_config(Path(args.budget_file))
    workload = cfg["workload"]
    budgets = cfg["budgets_seconds"]
    if args.slow_device and isinstance(cfg.get("slow_device_budgets_seconds"), dict):
        budgets = {**budgets, **cfg["slow_device_budgets_seconds"]}
    exprs = [str(x) for x in workload.get("expressions", []) if str(x).strip()]
    if not exprs:
        raise RuntimeError("No expressions defined in workload.expressions")

    repeats = int(workload.get("repeats", 5))
    parse_rounds = int(workload.get("parse_validate_rounds", 220))
    describe_rounds = int(workload.get("describe_expr_rounds", 220))
    next_after_rounds = int(workload.get("next_after_rounds", 220))
    codec_rounds = int(workload.get("codec_rounds", 120))
    snapshot_reuse_rounds = int(workload.get("snapshot_reuse_rounds", 3))
    snapshot_reuse_rows = int(workload.get("snapshot_reuse_rows", 1000))
    immutability_rounds = int(workload.get("immutability_rounds", 20))
    resource_limit_rounds = int(workload.get("resource_limit_rounds", 100))
    snapshot_memory_counts = tuple(
        int(value) for value in workload.get("snapshot_memory_counts", [100, 1000, 10000])
    )
    hints_rounds = int(workload.get("build_hints_rounds", 180))
    hints_cold_rounds = max(1, int(workload.get("build_hints_cold_rounds", 1)))
    hints_warm_rounds = max(1, int(workload.get("build_hints_warm_rounds", hints_rounds)))
    cache_key_rounds = int(workload.get("cache_key_rounds", 2500))
    cache_save_rounds = int(workload.get("cache_save_rounds", 120))
    cache_load_rounds = int(workload.get("cache_load_rounds", 300))
    outbox_schema_hot_rounds = int(workload.get("outbox_schema_hot_rounds", 1000))
    outbox_schema_cold_rounds = int(workload.get("outbox_schema_cold_rounds", 3))
    anchor_file_rounds = int(workload.get("anchor_file_rounds", 300))
    cold_import_rounds = int(workload.get("cold_import_rounds", 3))

    checks = [
        ("stage_capabilities", _bench_capabilities_stage, repeats),
        ("stage_queue_status", _bench_queue_status_stage, repeats),
        ("stage_navigator", _bench_navigator_stage, repeats),
        ("stage_query_pagination", _bench_query_pagination_stage, repeats),
        ("stage_query_unavailable", _bench_query_unavailable_stage, repeats),
        ("cold_core_import", lambda: _bench_cold_import("core", cold_import_rounds), repeats),
        (
            "cold_modify_impl_import",
            lambda: _bench_cold_import("modify_impl", cold_import_rounds),
            repeats,
        ),
        ("parse_validate", lambda: _bench_parse_validate(exprs, parse_rounds), repeats),
        ("describe_expr", lambda: _bench_describe_expr(exprs, describe_rounds), repeats),
        ("next_after", lambda: _bench_next_after(exprs, next_after_rounds), repeats),
        ("scheduler_decisions", lambda: _bench_scheduler_decisions(exprs), 1),
        ("task_codec_decode", lambda: _bench_task_codec(codec_rounds), repeats),
        (
            "task_snapshot_reuse",
            lambda: _bench_task_snapshot_reuse(snapshot_reuse_rounds, snapshot_reuse_rows),
            repeats,
        ),
        ("task_immutability", lambda: _bench_task_immutability(immutability_rounds), repeats),
        (
            "task_resource_limits",
            lambda: _bench_task_resource_limits(resource_limit_rounds),
            repeats,
        ),
        (
            "task_snapshot_memory",
            lambda: _bench_task_snapshot_memory(snapshot_memory_counts),
            1,
        ),
        (
            "build_hints_cold",
            lambda: _bench_build_hints(exprs, hints_cold_rounds, mode="cold"),
            repeats,
        ),
        (
            "build_hints_warm",
            lambda: _bench_build_hints(exprs, hints_warm_rounds, mode="warm"),
            repeats,
        ),
        (
            "build_hints_next_only_cold",
            lambda: _bench_build_hints(
                exprs,
                hints_cold_rounds,
                mode="cold",
                include_per_year=False,
            ),
            repeats,
        ),
        (
            "build_hints_next_only_warm",
            lambda: _bench_build_hints(
                exprs,
                hints_warm_rounds,
                mode="warm",
                include_per_year=False,
            ),
            repeats,
        ),
        ("cache_key_hot", lambda: _bench_cache_key_hot(exprs, cache_key_rounds), repeats),
        ("cache_save", lambda: _bench_cache_save(exprs, cache_save_rounds), repeats),
        ("cache_load_hot", lambda: _bench_cache_load_hot(exprs, cache_load_rounds), repeats),
        ("outbox_schema_hot", lambda: _bench_outbox_schema_hot(outbox_schema_hot_rounds), repeats),
        ("outbox_schema_cold", lambda: _bench_outbox_schema_cold(outbox_schema_cold_rounds), repeats),
        ("anchor_file_provider", lambda: _bench_anchor_file_provider(anchor_file_rounds), repeats),
        ("anchor_file_batch_provider", lambda: _bench_anchor_file_batch_provider(anchor_file_rounds), repeats),
    ]
    if shutil.which("task"):
        checks.append(("stage_doctor_installation", _bench_doctor_installation_stage, repeats))
    checks.append(("stage_housekeeping", _bench_housekeeping_stage, repeats))
    checks.append(("stage_repair_planner", _bench_repair_planner_stage, repeats))
    checks.append(("stage_repair_application", _bench_repair_application_stage, repeats))
    checks.append(("stage_lifecycle_staging", _bench_lifecycle_staging_stage, repeats))
    checks.append(("stage_reconcile_snapshot", _bench_reconcile_snapshot_stage, repeats))
    checks.append(("stage_queue_stale", _bench_queue_stale_stage, repeats))
    checks.append(("stage_operator_failure_matrix", _bench_operator_failure_matrix_stage, repeats))
    checks.append(("stage_operator_interrupted", _bench_operator_interrupted_stage, repeats))
    checks.append(("stage_exit_probe_fast_paths", _bench_exit_probe_fast_paths_stage, repeats))
    checks.append(("stage_operator_scope_matrix", _bench_operator_scope_matrix_stage, repeats))
    if args.workflows_only:
        checks = []

    extended = cfg.get("extended_workload")
    if args.extended and isinstance(extended, dict) and extended.get("enabled", True):
        extended_repeats = max(1, int(extended.get("repeats", 2)))
        extended_rows = max(1000, int(extended.get("anchor_file_rows", 5000)))
        extended_rounds = max(1, int(extended.get("rounds", 8)))
        extended_budgets_key = "slow_device_budgets_seconds" if args.slow_device else "budgets_seconds"
        extended_budgets = extended.get(extended_budgets_key)
        if not isinstance(extended_budgets, dict):
            extended_budgets = extended.get("budgets_seconds")
        if not isinstance(extended_budgets, dict):
            extended_budgets = {}

        checks.extend(
            [
                (
                    "anchor_file_large_cold",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="cold",
                    ),
                    extended_repeats,
                ),
                (
                    "anchor_file_large_hot",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="hot",
                    ),
                    extended_repeats,
                ),
                (
                    "anchor_file_nonmonotonic",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="nonmonotonic",
                    ),
                    extended_repeats,
                ),
                (
                    "anchor_file_business_day_omissions",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="hot",
                        business_day_only=True,
                    ),
                    extended_repeats,
                ),
                (
                    "business_calendar_large_omissions",
                    lambda: _bench_business_calendar_omissions(extended_rounds * 40),
                    extended_repeats,
                ),
                (
                    "native_until_reconcile_dry_run",
                    lambda: _bench_native_until_reconcile(extended_repeats, apply=False),
                    1,
                ),
                (
                    "native_until_reconcile_apply",
                    lambda: _bench_native_until_reconcile(extended_repeats, apply=True),
                    1,
                ),
            ]
        )
        if importlib.util.find_spec("astral") is not None:
            checks.extend(
                [
                    (
                        "astronomy_anchor_add",
                        lambda: _bench_astronomy_provider(extended_rounds, event="sunrise") or 0.0,
                        extended_repeats,
                    ),
                    (
                        "astronomy_anchor_completion",
                        lambda: _bench_astronomy_provider(extended_rounds, event="moonrise") or 0.0,
                        extended_repeats,
                    ),
                ]
            )

    seasonal = cfg.get("seasonal_workload")
    if not args.workflows_only and isinstance(seasonal, dict):
        seasonal_exprs = [
            str(value)
            for value in seasonal.get("expressions", [])
            if str(value).strip()
        ]
        if seasonal_exprs:
            seasonal_repeats = max(1, int(seasonal.get("repeats", 3)))
            seasonal_parse_rounds = max(1, int(seasonal.get("parse_validate_rounds", 100)))
            seasonal_next_rounds = max(1, int(seasonal.get("next_after_rounds", 100)))
            seasonal_hint_rounds = max(1, int(seasonal.get("build_hints_rounds", 1)))
            seasonal_cold_rounds = max(1, int(seasonal.get("build_hints_cold_rounds", 1)))
            seasonal_warm_rounds = max(1, int(seasonal.get("build_hints_warm_rounds", seasonal_hint_rounds)))
            checks.extend(
                [
                    (
                        "seasonal_parse_validate",
                        lambda: _bench_parse_validate(
                            seasonal_exprs,
                            seasonal_parse_rounds,
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_next_after",
                        lambda: _bench_next_after(
                            seasonal_exprs,
                            seasonal_next_rounds,
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_build_hints_cold",
                        lambda: _bench_build_hints(
                            seasonal_exprs,
                            seasonal_cold_rounds,
                            mode="cold",
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_build_hints_warm",
                        lambda: _bench_build_hints(
                            seasonal_exprs,
                            seasonal_warm_rounds,
                            mode="warm",
                        ),
                        seasonal_repeats,
                    ),
                ]
            )

    results = {}
    failures = []
    for name, fn, check_repeats in checks:
        r = _measure(name, fn, check_repeats, trace_memory=args.trace_memory)
        extended_budgets = {}
        if args.extended and isinstance(cfg.get("extended_workload"), dict):
            profile_key = "slow_device_budgets_seconds" if args.slow_device else "budgets_seconds"
            candidate = cfg["extended_workload"].get(profile_key)
            if isinstance(candidate, dict):
                extended_budgets = candidate
        budget = float(extended_budgets.get(name, budgets.get(name, 0.0)))
        r["budget_s"] = budget
        if name == "cold_core_import":
            r["module_count"] = IMPORT_PROFILES.get("core", 0)
            r["cache_semantics"] = "fresh-process; filesystem bytecode cache may be reused"
        elif name == "cold_modify_impl_import":
            r["module_count"] = IMPORT_PROFILES.get("modify_impl", 0)
            r["cache_semantics"] = "fresh-process; filesystem bytecode cache may be reused"
        r["pass"] = (budget <= 0.0) or (r["median_s"] <= budget)
        resource_budgets = cfg.get("resource_budgets")
        if isinstance(resource_budgets, dict):
            _apply_resource_budgets(r, name, resource_budgets.get(name, {}))
        if name in RESOURCE_DETAILS:
            r["details"] = RESOURCE_DETAILS[name]
        results[name] = r
        if args.enforce and not r["pass"]:
            failures.append(name)

    hook_results = {} if args.workflows_only else _bench_hook_fast_paths(cfg, panel_mode=args.panel_mode)
    for name, result in hook_results.items():
        results[name] = result
        if args.enforce and not result["pass"]:
            failures.append(name)

    workflow_results = _bench_expensive_workflows(
        cfg,
        slow_device=args.slow_device,
        panel_mode=args.panel_mode,
    )
    for name, result in workflow_results.items():
        results[name] = result
        if args.enforce and not result["pass"]:
            failures.append(name)

    summary = {
        "budget_file": str(Path(args.budget_file).resolve()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "profile": _budget_profile_name(slow_device=args.slow_device),
        "cwd": os.getcwd(),
        "results": results,
        "enforced": bool(args.enforce),
        "panel_mode": args.panel_mode,
        "ok": len(failures) == 0,
        "failed_checks": failures,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), indent=2))
    else:
        print("Nautical Performance Budget")
        print(f"Budget file: {summary['budget_file']}")
        for name, r in results.items():
            status = "OK" if r["pass"] else "FAIL"
            if "fast_to_full_ratio" in r:
                print(
                    f"- {name}: fast={r['median_s']:.4f}s full={r['full_median_s']:.4f}s "
                    f"ratio={r['fast_to_full_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            elif "managed_to_source_ratio" in r:
                print(
                    f"- {name}: managed={r['median_s']:.4f}s source={r['baseline_median_s']:.4f}s "
                    f"ratio={r['managed_to_source_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            elif "staged_to_source_ratio" in r:
                print(
                    f"- {name}: staged={r['median_s']:.4f}s source={r['baseline_median_s']:.4f}s "
                    f"ratio={r['staged_to_source_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            else:
                print(
                    f"- {name}: median={r['median_s']:.4f}s "
                    f"(min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) "
                    f"budget={r['budget_s']:.4f}s => {status}"
                )
        if args.enforce:
            print("Enforced:", "PASS" if summary["ok"] else f"FAIL ({', '.join(failures)})")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
