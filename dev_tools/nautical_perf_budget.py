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
import tracemalloc
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    from nautical_core.tools.nautical_query import _capabilities_payload

    started = time.perf_counter()
    payload = _capabilities_payload()
    if not isinstance(payload, dict) or payload.get("status") != "ok" or not payload.get("operations"):
        raise RuntimeError("capabilities stage returned an invalid payload")
    return time.perf_counter() - started


def _bench_queue_status_stage() -> float:
    """Measure queue-status composition against an isolated empty outbox."""
    from nautical_core.tools.nautical_queue_status import _status_payload

    with tempfile.TemporaryDirectory(prefix="nautical-perf-queue-status-") as td:
        taskdata = Path(td)
        started = time.perf_counter()
        payload, _budget = _status_payload(taskdata, stale_after=300.0, limit=5)
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
    from nautical_core.tools.nautical_doctor import _JSON_SCHEMA

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
            if not isinstance(payload, dict) or payload.get("schema") != _JSON_SCHEMA:
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
    """Measure one guarded lifecycle plan crossing the durable staging boundary."""
    from nautical_core.lifecycle_application import LifecycleApplicationService
    from dataclasses import replace

    with tempfile.TemporaryDirectory(prefix="nautical-perf-lifecycle-stage-") as td:
        taskdata = Path(td)
        _init_empty_outbox(taskdata)
        _parents, plans = _outbox_lifecycle_fixture("stage", 0, count=1)
        plans[0] = replace(plans[0], parent_guard=replace(plans[0].parent_guard, modified="20260829T000000Z"))
        repository = lifecycle_outbox.LifecycleOutboxRepository(taskdata)
        service = LifecycleApplicationService(outbox=repository, owner="perf-stage")
        started = time.perf_counter()
        outcome = service.stage(
            plans[0], configuration_fingerprint="perf-config", schedule_fingerprint="perf-schedule",
        )
        if outcome.kind.value not in {"applied", "staged", "already_applied"}:
            raise RuntimeError(f"lifecycle staging stage returned an unexpected outcome: {outcome!r}")
        pending = _workflow_outbox_pending(taskdata)
        if len(pending) != 1 or pending[0].get("state") not in {"ready", "claimed", "retry"}:
            raise RuntimeError(f"lifecycle staging stage did not leave one durable intent: {pending!r}")
        return time.perf_counter() - started


def _bench_reconcile_snapshot_stage() -> float:
    """Measure bounded reconcile projection from one authoritative read."""
    from nautical_core.reconcile_snapshot_service import ReconcileSnapshotService
    from nautical_core.task_models import TaskObservation

    class Repository:
        reads = 0

        def lifecycle_candidates(self, **_kwargs):
            self.reads += 1
            return object()

    rows = tuple(
        TaskObservation.from_mapping({
            "uuid": f"00000000-0000-4000-8000-{index:012d}",
            "status": "completed" if index % 2 == 0 else "pending",
            "chain": "on", "chainID": f"reconcile-stage-{index // 2:04d}",
            "link": index + 1,
        }, source_query="perf:reconcile-stage")
        for index in range(128)
    )
    repository = Repository()
    service = ReconcileSnapshotService(repository, read_value=lambda _value, _label: rows)
    started = time.perf_counter()
    candidates = service.candidate_rows()
    active = service.active_rows()
    if repository.reads != 1 or len(candidates) != 64 or len(active) != 64:
        raise RuntimeError(
            f"reconcile snapshot stage lost projection authority: reads={repository.reads} "
            f"candidates={len(candidates)} active={len(active)}"
        )
    return time.perf_counter() - started


def _bench_queue_stale_stage() -> float:
    """Measure queue-status detection of a stale claim with a valid plan."""
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository
    from nautical_core.queue_status_service import QueueStatusService

    with tempfile.TemporaryDirectory(prefix="nautical-perf-queue-stale-") as td:
        taskdata = Path(td)
        _init_empty_outbox(taskdata)
        _parents, plans = _outbox_lifecycle_fixture("stale", 0, count=1)
        _stage_workflow_plans(taskdata, plans, configuration_fingerprint="perf", schedule_fingerprint="perf")
        repository = LifecycleOutboxRepository(taskdata)
        claimed, records = repository.claim_batch(owner="perf-owner", lease_seconds=30.0, limit=1)
        if not claimed.ok or len(records) != 1:
            raise RuntimeError("stale queue fixture could not claim a valid lifecycle plan")
        with sqlite3.connect(str(repository.path)) as connection:
            connection.execute(
                "UPDATE lifecycle_outbox SET lease_expires_at=? WHERE intent_id=?",
                (time.time() - 10.0, records[0].intent_id),
            )
        started = time.perf_counter()
        payload = QueueStatusService().status_payload(taskdata, stale_after=5.0, limit=5)
        if int(payload.get("outbox", {}).get("stale_claims", 0) or 0) != 1:
            raise RuntimeError(f"queue status did not report the valid stale claim: {payload!r}")
        return time.perf_counter() - started


def _bench_operator_failure_matrix_stage() -> float:
    """Exercise fail-closed operator boundaries in one content-free matrix."""
    from types import SimpleNamespace

    from nautical_core.operator_health_service import OperatorHealthService
    from nautical_core.tools.nautical_reconcile import _configuration_verification

    started = time.perf_counter()
    # Query covers malformed pagination and unavailable authoritative reads;
    # repair covers an unsafe finding; queue covers a stale claim.  Each
    # component raises on a fabricated success, so this matrix is a compact
    # composition-root guard rather than a timing-only smoke test.
    _bench_query_pagination_stage()
    _bench_query_unavailable_stage()
    _bench_repair_planner_stage()
    _bench_queue_stale_stage()
    doctor_findings = OperatorHealthService.configuration_schema_findings({"panel_mode": 17})
    if not doctor_findings or not any(item.code == "config.schema.type" for item in doctor_findings):
        raise RuntimeError("Doctor failure matrix accepted malformed configuration")

    class BrokenCore:
        @staticmethod
        def configuration_drift():
            raise RuntimeError("synthetic configuration read failure")

    reconcile_state = _configuration_verification(SimpleNamespace(core=BrokenCore()))
    if reconcile_state.status != "unavailable" or "configuration verification unavailable" not in reconcile_state.reason:
        raise RuntimeError(f"reconcile failure matrix did not fail closed: {reconcile_state!r}")
    return time.perf_counter() - started


def _bench_operator_interrupted_stage() -> float:
    """Verify an interrupted operator claim remains reclaimable."""
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nautical-perf-operator-interrupted-") as td:
        taskdata = Path(td)
        _init_empty_outbox(taskdata)
        _parents, plans = _outbox_lifecycle_fixture("operator-interrupted", 0, count=1)
        _stage_workflow_plans(taskdata, plans, configuration_fingerprint="perf", schedule_fingerprint="perf")
        repository = LifecycleOutboxRepository(taskdata)
        first, records = repository.claim_batch(owner="interrupted-a", lease_seconds=0.05, limit=1)
        if not first.ok or len(records) != 1:
            raise RuntimeError("interrupted operator fixture could not claim its intent")
        time.sleep(0.08)
        second, reclaimed = repository.claim_batch(owner="interrupted-b", lease_seconds=1.0, limit=1)
        if not second.ok or len(reclaimed) != 1 or reclaimed[0].intent_id != records[0].intent_id:
            raise RuntimeError("expired interrupted operator intent was not reclaimed")
        return time.perf_counter() - started


def _bench_exit_probe_fast_paths_stage() -> float:
    """Verify empty and terminal outboxes are classified without Taskwarrior."""
    from nautical_core.exit_probe import probe_exit_work
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nautical-perf-exit-probe-") as td:
        taskdata = Path(td)
        empty = probe_exit_work(taskdata)
        if not empty.definitely_empty:
            raise RuntimeError(f"empty exit probe reported possible work: {empty.reason}")
        opened = LifecycleOutboxRepository(taskdata).open()
        if not opened.ok:
            raise RuntimeError("exit probe fixture outbox could not be initialized")
        terminal = probe_exit_work(taskdata)
        if not terminal.definitely_empty:
            raise RuntimeError(f"terminal exit probe reported possible work: {terminal.reason}")
    return time.perf_counter() - started


def _bench_operator_scope_matrix_stage() -> float:
    """Exercise empty, single-item, and page-boundary operator scopes."""
    from types import SimpleNamespace

    from nautical_core.query_models import OccurrenceQueryRequest
    from nautical_core.query_service import OccurrenceQueryService

    started = time.perf_counter()
    service = object.__new__(OccurrenceQueryService)
    service._timezone = timezone.utc
    service._scheduler_cache = {}
    service._uow = SimpleNamespace(
        mutation_epoch=0,
        context=SimpleNamespace(configuration=SimpleNamespace(fingerprint="scope-matrix")),
    )
    request = OccurrenceQueryRequest.from_mapping(
        {"selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1, "max_tasks": 2}
    )
    empty, empty_cursor, empty_complete = service._page_rows((), request)
    if empty or empty_cursor is not None or not empty_complete:
        raise RuntimeError("empty operator scope was not a complete page")
    one = (SimpleNamespace(uuid="scope-one"),)
    one_page, one_cursor, one_complete = service._page_rows(one, request)
    if tuple(row.uuid for row in one_page) != ("scope-one",) or one_cursor is not None or not one_complete:
        raise RuntimeError("single-item operator scope was not complete")
    boundary = tuple(SimpleNamespace(uuid=f"scope-{index}") for index in range(2))
    page, cursor, complete = service._page_rows(boundary, request)
    if len(page) != 2 or cursor is not None or not complete:
        raise RuntimeError("boundary-sized operator scope produced a continuation")
    plus_one = boundary + (SimpleNamespace(uuid="scope-2"),)
    page, cursor, complete = service._page_rows(plus_one, request)
    if len(page) != 2 or cursor is None or complete:
        raise RuntimeError("boundary-plus-one operator scope omitted its continuation")
    return time.perf_counter() - started


def _bench_describe_expr(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.describe_anchor_expr(expr)
    return time.perf_counter() - t0


def _bench_next_after(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    dnfs = [core.validate_anchor_expr_strict(expr) for expr in exprs]
    ref = date(2026, 1, 1)
    t0 = time.perf_counter()
    for _ in range(rounds):
        for dnf in dnfs:
            core.next_after_expr(dnf, ref)
    return time.perf_counter() - t0


def _bench_scheduler_decisions(exprs: list[str]) -> float:
    """Measure scheduler-service decisions without affecting wall-time samples."""
    from nautical_core.scheduler_cursor import OccurrenceCursor
    from nautical_core.scheduler_service import SchedulerService
    from nautical_core.scheduler_trace import SchedulerTrace

    row = {
        "uuid": "00000000-0000-4000-8000-000000000099",
        "description": "scheduler decision benchmark",
        "status": "pending", "chain": "on", "chainID": "scheduler-perf",
        "link": 1, "due": "20260824T090000Z",
    }
    started = time.perf_counter()
    decisions: dict[str, int] = {}
    for expr in exprs:
        observation = task_codec.DEFAULT_TASK_CODEC.decode_row(
            {**row, "anchor": expr}, source_query="perf:scheduler-decisions",
        )
        trace = SchedulerTrace(enabled=True, max_events=1)
        service = SchedulerService.from_observation(observation, trace=trace)
        cursor = OccurrenceCursor.strict_after(datetime(2026, 1, 1, tzinfo=timezone.utc), timezone=timezone.utc)
        service.next(cursor)
        decisions[expr] = trace.last_decision_count
    RESOURCE_DETAILS["scheduler_decisions"] = decisions
    return time.perf_counter() - started


def _bench_task_codec(rounds: int) -> float:
    """Measure typed task decoding across representative payload sizes.

    The malformed cases are part of the workload so a benchmark cannot pass by
    silently accepting invalid JSON or turning it into an empty snapshot.
    """
    codec = task_codec.DEFAULT_TASK_CODEC
    base = {
        "uuid": "00000000-0000-4000-8000-000000000001",
        "description": "codec benchmark",
        "status": "pending",
        "chain": "on",
        "chainID": "codec-perf",
        "link": 1,
        "anchor": "w:mon",
        "due": "20260824T090000Z",
        "entry": "20260820T090000Z",
    }
    large = {
        **base,
        "uuid": "00000000-0000-4000-8000-000000000002",
        "annotations": [
            {"entry": "20260820T090000Z", "description": "x" * 256}
            for _ in range(64)
        ],
        "tags": [f"tag-{index}" for index in range(64)],
        "depends": [f"00000000-0000-4000-8000-{index:012d}" for index in range(32)],
        "custom": {
            "nested": [{"index": index, "value": "v" * 128} for index in range(32)]
        },
    }
    valid_rows = (base, large)
    malformed_exports = ("{not-json", "[] trailing", '{"uuid":"missing-array"}')
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        for row in valid_rows:
            observation = codec.decode_row(row, source_query="perf:codec")
            if observation.field("uuid").presence.value != "value":
                raise RuntimeError("codec benchmark lost task identity")
        for text in malformed_exports:
            try:
                codec.decode_export(text, source_query="perf:codec")
            except task_codec.TaskCodecError:
                continue
            raise RuntimeError("codec benchmark accepted malformed export")
    return time.perf_counter() - started


def _bench_task_snapshot_reuse(rounds: int, row_count: int = 1000) -> float:
    """Measure indexed and graph reuse after one decode of a broad snapshot."""
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult
    from nautical_core.task_read_repository import AuthoritativeTaskSnapshot, TaskQueryKind, TaskSnapshotScope
    from nautical_core.chain_graph import ChainGraph
    from nautical_core.chain_integrity_models import ChainNode, ChainSnapshot, SnapshotCoverage
    from nautical_core.chain_invariants import evaluate_invariants

    class CountingCodec:
        def __init__(self):
            self.decode_count = 0

        def decode_row(self, row, *, source_query, **kwargs):
            self.decode_count += 1
            return task_codec.DEFAULT_TASK_CODEC.decode_row(
                row, source_query=source_query, **kwargs
            )

    rows = [
        {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/snapshot/{index}")),
            "description": f"snapshot row {index}",
            "status": "pending",
            "chain": "on",
            "chainID": f"snapshot-chain-{index // 10}",
            "link": (index % 10) + 1,
            "anchor": "w:mon",
            "due": "20260824T090000Z",
        }
        for index in range(max(1, int(row_count)))
    ]
    codec = CountingCodec()
    observations = tuple(codec.decode_row(row, source_query="perf:snapshot") for row in rows)
    expected_decode_count = len(rows)
    command = TaskCommand(("task", "export"), "perf snapshot", 1.0)
    result = TaskCommandResult(command, 0, "[]", "", CommandFailureKind.SUCCESS, 1, 0.0)
    scope = TaskSnapshotScope(TaskQueryKind.BROAD, "perf-snapshot", ("pending",))
    snapshot = AuthoritativeTaskSnapshot(scope, observations, result)
    graph_snapshot = ChainSnapshot(
        "perf-snapshot-graph",
        SnapshotCoverage.COMPLETE,
        "perf.snapshot",
        tuple(
            ChainNode.from_observation(row)
            for row in observations
        ),
        complete_chain_history=True,
    )
    graph = ChainGraph.from_snapshot(graph_snapshot)
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        for index in range(0, len(observations), max(1, len(observations) // 20)):
            row = observations[index]
            uuid_value = row.field("uuid").value
            chain_value = row.field("chainID").value
            link_value = row.field("link").value
            if not snapshot.uuid_matches(str(getattr(uuid_value, "value", uuid_value))):
                raise RuntimeError("snapshot UUID index lost a decoded row")
            if not snapshot.chain_rows(str(getattr(chain_value, "value", chain_value))):
                raise RuntimeError("snapshot chain index lost a decoded row")
            if not snapshot.slot_rows(
                str(getattr(chain_value, "value", chain_value)),
                int(getattr(link_value, "value", link_value)),
            ):
                raise RuntimeError("snapshot slot index lost a decoded row")
        findings = evaluate_invariants(graph)
        if findings:
            raise RuntimeError(f"snapshot graph reuse produced findings: {findings[0].reason_code}")
        observation_by_uuid = {
            str(getattr(row.field("uuid").value, "value", row.field("uuid").value)): row
            for row in observations
        }
        if any(node.observation is not observation_by_uuid.get(node.task_uuid) for node in graph.nodes):
            raise RuntimeError("chain graph did not retain decoded observations")
        if codec.decode_count != expected_decode_count:
            raise RuntimeError(
                "downstream snapshot consumers decoded a row more than once"
            )
    if any(snapshot.uuid_matches(str(getattr(row.field("uuid").value, "value", "")))[0] is not row for row in observations):
        raise RuntimeError("snapshot indexes did not reuse immutable observations")
    return time.perf_counter() - started


def _bench_task_immutability(rounds: int) -> float:
    """Verify frozen task fields survive source mutation without copies."""
    source = {
        "uuid": "00000000-0000-4000-8000-000000000003",
        "description": "immutable benchmark",
        "status": "pending",
        "chainID": "immutable-perf",
        "link": 1,
        "custom": {"nested": [{"value": "original"}, {"value": "stable"}]},
    }
    observation = task_codec.DEFAULT_TASK_CODEC.decode_row(source, source_query="perf:immutability")
    frozen = observation.arbitrary["custom"]
    source["custom"]["nested"][0]["value"] = "mutated"
    if observation.arbitrary["custom"] != frozen or "mutated" in repr(observation.arbitrary["custom"]):
        raise RuntimeError("immutable observation changed after source mutation")
    if observation.arbitrary["custom"] is not frozen:
        raise RuntimeError("immutable arbitrary field was rebuilt during access")
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        for _ in range(1000):
            if observation.arbitrary["custom"] is not frozen:
                raise RuntimeError("immutable field access returned a new value")
            if observation.field("description").value != "immutable benchmark":
                raise RuntimeError("immutable scalar field changed")
    return time.perf_counter() - started


def _bench_task_resource_limits(rounds: int) -> float:
    """Measure bounded nested freezing and the existing protocol size guard."""
    from nautical_core.hook_protocol import MAX_JSON_BYTES, probe_on_add

    nested = {"level": [{"value": "x" * 64, "items": [index, index + 1]} for index in range(32)]}
    row = {
        "uuid": "00000000-0000-4000-8000-000000000004",
        "description": "resource limit benchmark",
        "status": "pending",
        "chainID": "resource-perf",
        "link": 1,
        "nested": nested,
    }
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) >= MAX_JSON_BYTES:
        raise RuntimeError("resource benchmark fixture unexpectedly exceeds the protocol limit")
    observation = task_codec.DEFAULT_TASK_CODEC.decode_row(row, source_query="perf:resource-limits")
    oversize = b"{" + b'"description":"' + b"x" * MAX_JSON_BYTES + b'"}'
    if probe_on_add(oversize, max_bytes=MAX_JSON_BYTES).failure is None:
        raise RuntimeError("protocol accepted an oversized hook payload")
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        if not observation.arbitrary.get("nested"):
            raise RuntimeError("bounded nested arbitrary field was lost")
        if len(encoded.encode("utf-8")) >= MAX_JSON_BYTES:
            raise RuntimeError("resource fixture crossed the configured input limit")
    return time.perf_counter() - started


def _bench_task_snapshot_memory(counts: Sequence[int]) -> float:
    """Measure peak memory while decoding and indexing bounded snapshots."""
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult
    from nautical_core.task_read_repository import AuthoritativeTaskSnapshot, TaskQueryKind, TaskSnapshotScope

    command = TaskCommand(("task", "export"), "perf snapshot memory", 1.0)
    result = TaskCommandResult(command, 0, "[]", "", CommandFailureKind.SUCCESS, 1, 0.0)
    measurements: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for requested in counts:
        row_count = max(1, int(requested))
        rows = [
            {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/memory/{index}")),
                "description": f"memory row {index}",
                "status": "pending",
                "chain": "on",
                "chainID": f"memory-chain-{index // 10}",
                "link": (index % 10) + 1,
                "anchor": "w:mon",
                "due": "20260824T090000Z",
            }
            for index in range(row_count)
        ]
        tracemalloc.start()
        observations = tuple(
            task_codec.DEFAULT_TASK_CODEC.decode_row(row, source_query="perf:memory")
            for row in rows
        )
        scope = TaskSnapshotScope(TaskQueryKind.BROAD, f"perf-memory-{row_count}", ("pending",))
        snapshot = AuthoritativeTaskSnapshot(scope, observations, result)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if len(snapshot.rows) != row_count or len(snapshot.by_uuid) != row_count:
            raise RuntimeError("memory benchmark snapshot was truncated")
        measurements[str(row_count)] = {"current_bytes": int(current), "peak_bytes": int(peak)}
    RESOURCE_DETAILS["task_snapshot_memory"] = measurements
    return time.perf_counter() - started


def _bench_build_hints(exprs: list[str], rounds: int, *, mode: str = "warm") -> float:
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
            if mode == "cold":
                root = Path(core.ANCHOR_CACHE_DIR_OVERRIDE)
                started = time.perf_counter()
                for sample_index in range(max(1, rounds)):
                    core.ANCHOR_CACHE_DIR_OVERRIDE = str(root / f"cold-{sample_index}")
                    core._CACHE_DIR = None
                    _clear_caches()
                    for expr in exprs:
                        core.build_and_cache_hints(expr, "skip")
                if counts["hits"] or not counts["misses"]:
                    raise RuntimeError(f"cold hint benchmark observed unexpected cache state: {counts}")
                return time.perf_counter() - started
            if mode != "warm":
                raise ValueError(f"unknown hint benchmark mode: {mode}")
            core.cache_load = saved_load
            core.cache_save = saved_save
            _clear_caches()
            for expr in exprs:
                core.build_and_cache_hints(expr, "skip")
            core.cache_load = counted_load
            core.cache_save = counted_save
            counts = {"hits": 0, "misses": 0, "saves": 0}
            _clear_caches()
            t0 = time.perf_counter()
            for _ in range(max(1, rounds)):
                for expr in exprs:
                    core.build_and_cache_hints(expr, "skip")
            if counts["misses"] or not counts["hits"] or counts["saves"]:
                raise RuntimeError(f"warm hint benchmark observed unexpected cache state: {counts}")
            return time.perf_counter() - t0
        finally:
            core.cache_load = saved_load
            core.cache_save = saved_save


def _bench_cache_key_hot(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.cache_key_for_task(expr, "skip")
    return time.perf_counter() - t0


@contextmanager
def _perf_cache_context():
    """Isolate cache I/O benchmarks from user cache directories."""
    saved_enable = bool(getattr(core, "ENABLE_ANCHOR_CACHE", False))
    saved_override = str(getattr(core, "ANCHOR_CACHE_DIR_OVERRIDE", "") or "")
    saved_cache_dir = getattr(core, "_CACHE_DIR", None)
    saved_ttl = int(getattr(core, "ANCHOR_CACHE_TTL", 0) or 0)
    with tempfile.TemporaryDirectory(prefix="nautical-perf-cache-") as td:
        try:
            core.ENABLE_ANCHOR_CACHE = True
            core.ANCHOR_CACHE_DIR_OVERRIDE = td
            core.ANCHOR_CACHE_TTL = 0
            core._CACHE_DIR = None
            _clear_caches()
            yield td
        finally:
            core.ENABLE_ANCHOR_CACHE = saved_enable
            core.ANCHOR_CACHE_DIR_OVERRIDE = saved_override
            core.ANCHOR_CACHE_TTL = saved_ttl
            core._CACHE_DIR = saved_cache_dir
            _clear_caches()


def _cache_payload(expr: str, idx: int) -> dict:
    return {
        "natural": expr,
        "next_dates": ["2026-01-01", "2026-01-08", "2026-01-15"],
        "meta": {"i": idx},
        # Keep payload shape aligned with cache schema checks.
        "dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]],
    }


def _bench_cache_save(exprs: list[str], rounds: int) -> float:
    with _perf_cache_context():
        keys = [f"perf-save-{i}" for i in range(max(1, len(exprs)))]
        t0 = time.perf_counter()
        idx = 0
        for _ in range(rounds):
            for i, expr in enumerate(exprs):
                payload = _cache_payload(expr, idx)
                if not core.cache_save(keys[i], payload):
                    raise RuntimeError("cache_save benchmark write failed")
                idx += 1
        return time.perf_counter() - t0


def _bench_cache_load_hot(exprs: list[str], rounds: int) -> float:
    with _perf_cache_context():
        keys = [f"perf-load-{i}" for i in range(max(1, len(exprs)))]
        for i, expr in enumerate(exprs):
            if not core.cache_save(keys[i], _cache_payload(expr, i)):
                raise RuntimeError("cache_load benchmark setup write failed")
        _clear_caches()
        t0 = time.perf_counter()
        for _ in range(rounds):
            for key in keys:
                obj = core.cache_load(key)
                if not isinstance(obj, dict):
                    raise RuntimeError("cache_load benchmark read failed")
        return time.perf_counter() - t0


def _bench_outbox_schema_hot(rounds: int) -> float:
    with tempfile.TemporaryDirectory(prefix="nautical-perf-outbox-") as td:
        repository = lifecycle_outbox.LifecycleOutboxRepository(Path(td))
        if not repository.open().ok:
            raise RuntimeError("outbox schema benchmark setup failed")
        t0 = time.perf_counter()
        for _ in range(rounds):
            if not repository.open().ok:
                raise RuntimeError("outbox schema hot open failed")
        return time.perf_counter() - t0


def _bench_outbox_schema_cold(rounds: int) -> float:
    """Measure lifecycle outbox initialization across fresh Python processes."""
    with tempfile.TemporaryDirectory(prefix="nautical-perf-outbox-cold-") as td:
        script = (
            "from pathlib import Path; import sys; "
            "from nautical_core.lifecycle_outbox import LifecycleOutboxRepository; "
            "result = LifecycleOutboxRepository(Path(sys.argv[1])).open(); "
            "raise SystemExit(0 if result.ok else result.reason)"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
        )
        started = time.perf_counter()
        for _ in range(max(1, rounds)):
            proc = subprocess.run(
                [sys.executable, "-c", script, td],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=30.0,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"cold outbox initialization failed: {proc.stderr.strip()}")
        return time.perf_counter() - started


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
    """Exercise cached anchor-file expansion and successor lookup."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-file-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        for index in range(365):
            item_date = date(2026, 1, 1) + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        provider = anchor_files.AnchorFileOccurrenceProvider(
            "calendar.csv@t=09:00",
            td,
            (9, 0),
        )
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        started = time.perf_counter()
        for index in range(max(1, rounds)):
            after = datetime(2026, 1, 1, 8, 0) + timedelta(days=index % 364)
            if provider.next_after(after, build_local_datetime=build, to_local=identity) is None:
                raise RuntimeError("anchor-file provider benchmark unexpectedly exhausted")
        return time.perf_counter() - started


def _bench_anchor_file_batch_provider(rounds: int) -> float:
    """Measure provider-owned finite batch generation on a warm file cache."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    occurrence_provider = importlib.import_module("nautical_core.occurrence_provider")
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-batch-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        for index in range(365):
            item_date = date(2026, 1, 1) + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        provider = anchor_files.AnchorFileOccurrenceProvider("calendar.csv@t=09:00", td, (9, 0))
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        started = time.perf_counter()
        for index in range(max(1, rounds)):
            after = datetime(2026, 1, 1, 8, 0) + timedelta(days=index % 350)
            batch = occurrence_provider.collect_after(
                provider,
                after,
                limit=5,
                build_local_datetime=build,
                to_local=identity,
                require_contract=True,
            )
            if not batch:
                raise RuntimeError("anchor-file batch benchmark unexpectedly returned no occurrences")
        return time.perf_counter() - started


def _bench_large_anchor_file_provider(
    rounds: int,
    *,
    row_count: int = 5000,
    mode: str = "hot",
    business_day_only: bool = False,
) -> float:
    """Exercise large file-backed calendars and their cached lookup cursors."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    business_calendar = importlib.import_module("nautical_core.business_calendar")
    row_count = max(1000, int(row_count))
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-large-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        first_day = date(2020, 1, 1)
        for index in range(row_count):
            item_date = first_day + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},event-{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        suffix = "@bd" if business_day_only else ""
        provider_name = f"calendar.csv{suffix}"
        calendar = business_calendar.DEFAULT_BUSINESS_CALENDAR if business_day_only else None
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        query_indexes = list(range(0, row_count - 2, max(1, row_count // 37)))
        if mode == "nonmonotonic":
            query_indexes = query_indexes[::2] + list(reversed(query_indexes[1::2]))
        started = time.perf_counter()
        provider = None
        if mode != "cold":
            provider = anchor_files.AnchorFileOccurrenceProvider(
                provider_name,
                td,
                (9, 0),
                business_calendar=calendar,
            )
            # Exclude setup from the hot lookup measurement.
            if provider.next_after(
                datetime.combine(first_day - timedelta(days=1), datetime.min.time()),
                build_local_datetime=build,
                to_local=identity,
            ) is None:
                raise RuntimeError("large anchor-file provider failed to load its first occurrence")
        for index in range(max(1, int(rounds))):
            for query_index in query_indexes:
                current = provider
                if mode == "cold":
                    current = anchor_files.AnchorFileOccurrenceProvider(
                        provider_name,
                        td,
                        (9, 0),
                        business_calendar=calendar,
                    )
                after = datetime.combine(
                    first_day + timedelta(days=query_index),
                    datetime.min.time(),
                )
                occurrence = current.next_after(
                    after,
                    build_local_datetime=build,
                    to_local=identity,
                )
                if occurrence is None:
                    raise RuntimeError(
                        f"large anchor-file provider exhausted unexpectedly ({mode}, query={query_index})"
                    )
                if business_day_only and occurrence.day.weekday() >= 5:
                    raise RuntimeError("business-day anchor-file benchmark returned a weekend")
        return time.perf_counter() - started


def _bench_business_calendar_omissions(rounds: int) -> float:
    """Measure recurrence selection with a large explicit omission set."""
    business_calendar = importlib.import_module("nautical_core.business_calendar")
    dnf = core.validate_anchor_expr_strict("m:1..31@bd")
    first_day = date(2026, 1, 1)
    calendar_days = frozenset(first_day + timedelta(days=index) for index in range(366))
    omitted = frozenset(
        item
        for item in calendar_days
        if item.day not in {1, 15} or item.weekday() >= 5
    )
    calendar = business_calendar.ConfiguredBusinessCalendar(
        name="perf-omissions",
        fingerprint="perf-omissions-v1",
        anchor_dates=calendar_days,
        omit_dates=omitted,
        _anchor_matches=lambda _value: False,
        _omit_matches=lambda _value: False,
    )
    started = time.perf_counter()
    for index in range(max(1, int(rounds))):
        result, _meta = core.next_after_expr(
            dnf,
            first_day + timedelta(days=index % 300),
            seed_base="perf-omissions",
            business_calendar=calendar,
        )
        if result is None or result.day not in {1, 15} or result.weekday() >= 5:
            raise RuntimeError("large omission-set benchmark selected an omitted date")
    return time.perf_counter() - started


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


def _measure(name: str, fn, repeats: int) -> dict:
    samples = []
    cpu_samples = []
    wall_samples = []
    peak_memory_samples = []
    # Warmup once for interpreter/cache stabilization.
    _ = fn()
    for _ in range(max(1, repeats)):
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        started_tracing = tracemalloc.is_tracing()
        if not started_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        reported = float(fn())
        elapsed_wall = time.perf_counter() - started_wall
        elapsed_cpu = time.process_time() - started_cpu
        _current, peak_memory = tracemalloc.get_traced_memory()
        if not started_tracing:
            tracemalloc.stop()
        # Existing checks return their own wall duration. Preserve that value
        # while recording measured CPU/wall attribution alongside it.
        samples.append(reported)
        cpu_samples.append(max(0.0, elapsed_cpu))
        wall_samples.append(max(0.0, elapsed_wall))
        peak_memory_samples.append(max(0, int(peak_memory)))
    samples = sorted(samples)
    cpu_samples = sorted(cpu_samples)
    wall_samples = sorted(wall_samples)
    peak_memory_samples = sorted(peak_memory_samples)
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": statistics.median(samples),
        "max_s": samples[-1],
        "cpu_samples_s": cpu_samples,
        "cpu_median_s": statistics.median(cpu_samples),
        "measured_wall_median_s": statistics.median(wall_samples),
        "peak_memory_samples_bytes": peak_memory_samples,
        "peak_memory_median_bytes": statistics.median(peak_memory_samples),
    }


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


def _bench_hook_fast_paths(cfg: dict, *, panel_mode: str = "minimal") -> dict[str, dict]:
    hook_cfg = cfg.get("hook_fast_path")
    if not isinstance(hook_cfg, dict) or not hook_cfg.get("enabled", True):
        return {}
    repeats = max(1, int(hook_cfg.get("repeats", 7)))
    max_ratios = hook_cfg.get("max_ratio") if isinstance(hook_cfg.get("max_ratio"), dict) else {}
    managed_max_ratio = float(hook_cfg.get("managed_layout_max_ratio", 1.5))

    plain = {
        "uuid": "11111111-1111-1111-1111-111111111111",
        "description": "plain hook latency",
        "status": "pending",
        "entry": "20260101T000000Z",
        "modified": "20260101T000000Z",
    }
    modified = dict(plain, modified="20260101T000001Z")

    with tempfile.TemporaryDirectory(prefix="nautical-hook-perf-") as td:
        temp_root = Path(td)
        config_path = temp_root / "config-nautical.toml"
        config_path.write_text(
            f'tz = "UTC"\npanel_mode = "{_panel_mode_config(panel_mode)}"\n',
            encoding="utf-8",
        )
        base_env = os.environ.copy()
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TZ": "UTC",
            }
        )
        for key in ("NAUTICAL_DIAG", "NAUTICAL_DIAG_LOG", "NAUTICAL_PROFILE"):
            base_env.pop(key, None)

        cases = []
        add_data = temp_root / "add-data"
        add_data.mkdir()
        cases.append(
            (
                "hook_plain_add",
                ROOT / "on-add.nautical",
                json.dumps(plain, ensure_ascii=False),
                plain,
                add_data,
            )
        )
        modify_data = temp_root / "modify-data"
        modify_data.mkdir()
        cases.append(
            (
                "hook_plain_modify",
                ROOT / "on-modify.nautical",
                json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False),
                modified,
                modify_data,
            )
        )
        nautical_old = dict(
            plain,
            cp="P1D",
            chain="on",
            chainID="abcd1234",
            link=3,
            due="20270101T090000Z",
        )
        nautical_modified = dict(nautical_old, description="ordinary Nautical edit", modified="20260101T000001Z")
        cases.append(
            (
                "hook_nautical_ordinary_modify",
                ROOT / "on-modify.nautical",
                json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(nautical_modified, ensure_ascii=False),
                nautical_modified,
                modify_data,
            )
        )
        exit_data = temp_root / "exit-data"
        exit_data.mkdir()
        cases.append(("hook_empty_exit", ROOT / "on-exit.nautical", "", None, exit_data))

        results = {}
        for name, hook_path, input_text, expected_task, taskdata in cases:
            env = dict(base_env)
            env["TASKDATA"] = str(taskdata)
            ratio_budget = float(max_ratios.get(name, 0.8))
            results[name] = _measure_hook_fast_path(
                name,
                hook_path,
                input_text=input_text,
                expected_task=expected_task,
                base_env=env,
                repeats=repeats,
                max_ratio=ratio_budget,
            )

        managed_data = temp_root / "managed-data"
        install_runtime.install_release(
            source=ROOT,
            taskdata=managed_data,
            release_id="perf-managed",
            smoke=False,
        )
        managed_env = dict(base_env)
        managed_env["TASKDATA"] = str(managed_data)
        for name, source_hook, input_text, expected_task, _taskdata in cases:
            managed_name = f"managed_{name}"
            results[managed_name] = _measure_managed_hook_latency(
                managed_name,
                managed_data / "hooks" / source_hook.name,
                input_text=input_text,
                expected_task=expected_task,
                base_env=managed_env,
                repeats=repeats,
                baseline_median_s=float(results[name]["median_s"]),
                max_ratio=managed_max_ratio,
            )
        return results


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


def _measure_workflow(name: str, samples: list[float], budget: float) -> dict:
    ordered = sorted(float(value) for value in samples)
    median = float(statistics.median(ordered))
    return {
        "name": name,
        "samples_s": ordered,
        "min_s": ordered[0],
        "median_s": median,
        "max_s": ordered[-1],
        "budget_s": float(budget),
        "pass": float(budget) <= 0.0 or median <= float(budget),
    }


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


def _attach_timing_breakdown(
    result: dict,
    wall_samples: list[float],
    timing_samples: list[dict[str, float]],
) -> None:
    """Attach command, startup, presentation, and derived Python timing."""
    result["timing_stats"] = timing_samples
    result["timing_breakdown"] = [
        {
            "wall_seconds": round(float(wall), 6),
            "taskwarrior_seconds": round(float(timing.get("run_task_seconds", 0.0)), 6),
            "startup_seconds": round(float(timing.get("startup_total_ms", 0.0)) / 1000.0, 6),
            "drain_seconds": round(float(timing.get("drain_ms", 0.0)) / 1000.0, 6),
            "presentation_seconds": round(float(timing.get("presentation_ms", 0.0)) / 1000.0, 6),
            "non_taskwarrior_seconds": round(
                max(0.0, float(wall) - float(timing.get("run_task_seconds", 0.0))), 6
            ),
        }
        for wall, timing in zip(wall_samples, timing_samples)
    ]


def _merge_task_timing_stats(*stats: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for item in stats:
        for key, value in item.items():
            merged[key] = merged.get(key, 0.0) + float(value)
    return merged


def _compact_reconcile_report(report: dict) -> dict:
    """Keep the performance-relevant fields from one reconcile report."""
    return {
        key: report.get(key)
        for key in (
            "status",
            "mode",
            "export_calls",
            "export_rows",
            "task_command_calls",
            "task_command_attempts",
            "task_command_duration",
            "task_command_by_purpose",
            "task_command_budget_exceeded",
            "integrity_seconds",
            "integrity_application_seconds",
        )
        if key in report
    }


def _attach_reconcile_reports(result: dict, reports: list[dict]) -> None:
    """Attach reconcile command, export, and phase metrics to a result."""
    result["reconcile_reports"] = reports


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


def _reconcile_candidate_tasks(prefix: str, count: int) -> list[dict]:
    """Create independent completed roots for reconcile candidate scaling."""
    return [
        {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{prefix}/{index}")),
            "status": "completed",
            "description": f"Reconcile candidate benchmark {index}",
            "cp": "P1D",
            "chain": "on",
            "chainID": f"reconcile-candidate-{prefix}-{index}",
            "link": 1,
            "due": "20260101T090000Z",
        }
        for index in range(count)
    ]


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


def _bench_expensive_workflows(
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
            f'tz = "UTC"\npanel_mode = "{_panel_mode_config(panel_mode)}"\n',
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
        base_env = os.environ.copy()
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
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

        # Scale the pure graph/invariant stage independently from Taskwarrior.
        # This keeps the fixture useful on slow devices without turning a
        # subprocess benchmark into an unbounded database import.
        from nautical_core.chain_graph import ChainGraph
        from nautical_core.chain_integrity_models import ChainNode, ChainSnapshot, SnapshotCoverage
        from nautical_core.chain_invariants import evaluate_invariants
        from nautical_core.task_models import TaskObservation

        scale_counts = tuple(int(item) for item in workflow_cfg.get("integrity_scale_counts", (100, 1000, 10000)))
        for scale_count in scale_counts:
            if scale_count <= 0:
                continue
            started = time.perf_counter()
            rows = tuple(
                ChainNode.from_observation(TaskObservation.from_mapping({
                    "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/integrity-scale/{scale_count}/{index}")),
                    "status": "pending",
                    "description": f"Integrity scale {index}",
                    "chain": "on",
                    "chainID": f"integrity-scale-{index}",
                    "link": 1,
                    "cp": "P1D",
                    "due": "20260101T090000Z",
                }, source_query="perf:integrity-scale"))
                for index in range(scale_count)
            )
            snapshot = ChainSnapshot(
                f"integrity-scale-{scale_count}",
                SnapshotCoverage.COMPLETE,
                "perf.synthetic",
                rows,
                complete_chain_history=True,
            )
            findings = evaluate_invariants(ChainGraph.from_snapshot(snapshot))
            if findings:
                raise RuntimeError(f"integrity scale fixture produced findings at {scale_count} chains")
            results[f"integrity_scale_{scale_count}"] = _measure_workflow(
                f"integrity_scale_{scale_count}",
                [time.perf_counter() - started],
                float(budgets.get(f"integrity_scale_{scale_count}", 0.0)),
            )

        ordinary_samples = []
        for sample_index in range(repeats):
            old = _completion_fixture("cp", sample_index, nonfinal=True, mode="ordinary")
            new = dict(old, description=f"Ordinary Nautical edit {sample_index}")
            taskdata = root / f"ordinary-modify-{sample_index}"
            taskdata.mkdir()
            env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1")
            _import_workflow_rows((old,), env=env)
            elapsed, result, _stderr = _run_workflow_hook_result(
                ROOT / "on-modify.nautical",
                input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                env=env,
                expect_output=True,
            )
            if result != new or _workflow_outbox_pending(taskdata):
                raise RuntimeError("workflow_ordinary_modify changed the task or staged work")
            ordinary_samples.append(elapsed)
        results["workflow_ordinary_modify"] = _measure_workflow(
            "workflow_ordinary_modify",
            ordinary_samples,
            float(budgets.get("workflow_ordinary_modify", 2.0)),
        )

        expiration_samples = []
        expiration_day = datetime.now(timezone.utc).date() - timedelta(days=1)
        expiration_date = expiration_day.strftime("%Y%m%d")
        for sample_index in range(repeats):
            key = f"nautical-perf/expiration/{sample_index}"
            parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, key + "/parent"))
            old = {
                "uuid": parent_uuid,
                "status": "pending",
                "description": f"Expiration recovery benchmark {sample_index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": f"expiration-perf-{sample_index:04d}",
                "link": 1,
                "due": f"{expiration_date}T090000Z",
                "until": f"{expiration_date}T200000Z",
                # Lifecycle plans require the Taskwarrior modified guard;
                # preserve it in the synthetic hook snapshot just as an
                # exported task would.
                "modified": f"{expiration_date}T090000Z",
            }
            # An expiration deletion ends at the native until boundary.  A
            # later end timestamp is a manual/ambiguous deletion and should
            # be deferred by the hook rather than staged for recovery.
            new = dict(old, status="deleted", end=f"{expiration_date}T200000Z")
            taskdata = root / f"expiration-recovery-{sample_index}"
            taskdata.mkdir()
            env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1")
            _import_workflow_rows((old,), env=env)
            elapsed, result, _stderr = _run_workflow_hook_result(
                ROOT / "on-modify.nautical",
                input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                env=env,
                expect_output=True,
            )
            staged = _workflow_outbox_pending(taskdata)
            if not isinstance(result, dict) or result.get("chain") != "on" or len(staged) != 1:
                raise RuntimeError(
                    "workflow_expiration_recovery did not stage exactly one successor: "
                    f"result={result!r}; staged={staged!r}; stderr={_stderr.strip()!r}"
                )
            # Replaying the same deletion must be idempotent: a crash/retry
            # after staging may not create a second successor intent.
            replay_result = _run_workflow_hook_result(
                ROOT / "on-modify.nautical",
                input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                env=env,
                expect_output=True,
            )
            replay_staged = _workflow_outbox_pending(taskdata)
            if not isinstance(replay_result[1], dict) or len(replay_staged) != 1:
                raise RuntimeError(
                    "workflow_expiration_recovery replay was not idempotent: "
                    f"result={replay_result[1]!r}; staged={replay_staged!r}; "
                    f"stderr={replay_result[2].strip()!r}"
                )
            expiration_samples.append(elapsed)
        results["workflow_expiration_recovery"] = _measure_workflow(
            "workflow_expiration_recovery",
            expiration_samples,
            float(budgets.get("workflow_expiration_recovery", 2.0)),
        )

        for name, kind, nonfinal in completion_cases:
            fresh_samples = []
            fresh_call_stats: list[dict[str, int]] = []
            for sample_index in range(repeats):
                old = _completion_fixture(kind, sample_index, nonfinal=nonfinal, mode="fresh")
                new = dict(old, status="completed", end="20260101T100000Z" if kind == "cp" else "20260105T100000Z")
                taskdata = root / f"{name}-fresh-{sample_index}"
                taskdata.mkdir()
                stats_path = taskdata / "on-modify-task-stats.json"
                env = dict(
                    base_env,
                    TASKDATA=str(taskdata),
                    NAUTICAL_BENCH_FORCE_FULL="1",
                    NAUTICAL_BENCH_STATS_FILE=str(stats_path),
                )
                _import_workflow_rows((old,), env=env)
                elapsed, result, _stderr = _run_workflow_hook_result(
                    ROOT / "on-modify.nautical",
                    input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                    env=env,
                    expect_output=True,
                )
                if not isinstance(result, dict):
                    raise RuntimeError(f"{name} fresh sample returned no task object")
                fresh_call_stats.append(_read_exit_task_call_stats(stats_path))
                queued = _workflow_outbox_pending(taskdata)
                if nonfinal:
                    if result.get("chain") != "on" or len(queued) != 1:
                        raise RuntimeError(
                            f"{name} fresh sample did not queue exactly one child: "
                            f"result={result!r}; queued={queued!r}; stderr={_stderr.strip()!r}"
                        )
                    if queued[0].get("stage") != "planned":
                        raise RuntimeError(f"{name} fresh sample staged an invalid lifecycle record")
                elif result.get("chain") != "off" or queued:
                    raise RuntimeError(f"{name} final sample did not complete without a successor")
                fresh_samples.append(elapsed)
            results[name] = _measure_workflow(name, fresh_samples, float(budgets.get(name, 2.0)))
            results[name]["task_call_stats"] = fresh_call_stats
            call_budgets = workflow_cfg.get("task_call_budgets")
            if isinstance(call_budgets, dict):
                _apply_task_call_budgets(results[name], fresh_call_stats, call_budgets.get(name, {}))

            if nonfinal:
                idem_name = f"{name}_idempotent"
                idem_samples = []
                idem_call_stats: list[dict[str, int]] = []
                for sample_index in range(repeats):
                    old = _completion_fixture(kind, sample_index, nonfinal=True, mode="idempotent")
                    new = dict(old, status="completed", end="20260101T100000Z" if kind == "cp" else "20260105T100000Z")
                    taskdata = root / f"{idem_name}-{sample_index}"
                    taskdata.mkdir()
                    stats_path = taskdata / "on-modify-task-stats.json"
                    env = dict(
                        base_env,
                        TASKDATA=str(taskdata),
                        NAUTICAL_BENCH_STATS_FILE=str(stats_path),
                    )
                    _import_workflow_rows((old,), env=env)
                    _import_existing_completion_child(old, env=env)
                    elapsed, result, stderr = _run_workflow_hook_result(
                        ROOT / "on-modify.nautical",
                        input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                        env=env,
                        expect_output=True,
                    )
                    if not isinstance(result, dict) or result.get("chain") != "on":
                        raise RuntimeError(f"{idem_name} sample changed the completed parent unexpectedly")
                    if _workflow_outbox_pending(taskdata):
                        raise RuntimeError(f"{idem_name} sample staged a duplicate child")
                    if "Spawn skipped" not in stderr:
                        raise RuntimeError(f"{idem_name} sample did not report the existing next link")
                    idem_call_stats.append(_read_exit_task_call_stats(stats_path))
                    idem_samples.append(elapsed)
                results[idem_name] = _measure_workflow(
                    idem_name,
                    idem_samples,
                    float(budgets.get(idem_name, 2.0)),
                )
                results[idem_name]["task_call_stats"] = idem_call_stats
                if isinstance(call_budgets, dict):
                    _apply_task_call_budgets(
                        results[idem_name], idem_call_stats, call_budgets.get(idem_name, {})
                    )

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
            _init_empty_outbox(queue_data)
            stats_path = queue_data / "on-exit-task-stats.json"
            queue_env = dict(base_env, TASKDATA=str(queue_data), NAUTICAL_BENCH_STATS_FILE=str(stats_path))
            # Healthy drains must measure the real Taskwarrior process.  The
            # wrapper is reserved for the partial-import failure injection
            # below; otherwise every command would include an avoidable Python
            # process and obscure the queue's actual cost.
            queue_env["NAUTICAL_BENCH_TASK_BIN"] = str(real_task)
            if os.environ.get("NAUTICAL_DIAG") == "1":
                queue_env["NAUTICAL_DIAG"] = "1"
            parents, queue_plans = _outbox_lifecycle_fixture("queue", sample_index)
            parent_uuids = {str(parent["uuid"]) for parent in parents}
            child_uuids = {str(plan.child_dict()["uuid"]) for plan in queue_plans}

            import_proc = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(parent, ensure_ascii=False) + "\n" for parent in parents),
                text=True,
                capture_output=True,
                env=queue_env,
                timeout=30.0,
            )
            if import_proc.returncode != 0:
                raise RuntimeError(
                    "queue drain parent fixture import failed: "
                    f"{(import_proc.stderr or import_proc.stdout or '').strip()}"
                )
            preload_probe = subprocess.run(
                [
                    "task", f"rc.data.location={queue_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on",
                    "(", "status:completed", "or", "status:deleted", "or", "status:pending", "or",
                    "status:waiting", ")", "export",
                ],
                text=True,
                capture_output=True,
                env=queue_env,
                timeout=30.0,
            )
            preload_rows = json.loads(preload_probe.stdout or "[]")
            if (
                preload_probe.returncode != 0
                or not isinstance(preload_rows, list)
                or {str(row.get("uuid") or "") for row in preload_rows if isinstance(row, dict)} != parent_uuids
            ):
                raise RuntimeError(
                    "queue drain benchmark parent preflight was not visible to Taskwarrior: "
                    f"{(preload_probe.stderr or preload_probe.stdout or '').strip()}"
                )
            queue_plans = _bind_workflow_plans_to_parents(queue_plans, preload_rows)
            _stage_workflow_plans(
                queue_data,
                queue_plans,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
            )

            queue_elapsed, _queue_result, _queue_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical",
                input_text="",
                env=queue_env,
                expect_output=False,
            )
            queue_samples.append(queue_elapsed)
            queue_call_stats.append(_read_exit_task_call_stats(stats_path))
            child_slot_reads = queue_call_stats[-1].get("run_task_calls_purpose_task_read_child_slot", 0)
            if child_slot_reads > 1:
                raise RuntimeError(
                    "queue drain preflight regressed to per-candidate child-slot reads: "
                    f"{child_slot_reads} subprocesses for {len(queue_plans)} candidates"
                )
            queue_timing_stats.append(_read_exit_task_timing_stats(stats_path))
            queue_outbox_stats.append(_read_exit_outbox_stats(stats_path))

            if _workflow_outbox_pending(queue_data):
                raise RuntimeError(
                    "outbox drain benchmark left active intents after successful processing: "
                    f"{_workflow_outbox_pending(queue_data)!r}; stderr={_queue_stderr.strip()!r}"
                )

            # Acknowledged outbox records are terminal and must not trigger
            # another Taskwarrior read or mutation on a replay drain.
            try:
                stats_path.unlink()
            except FileNotFoundError:
                pass
            idem_elapsed, _idem_result, _idem_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical",
                input_text="",
                env=dict(queue_env, NAUTICAL_BENCH_FORCE_FULL="1"),
                expect_output=False,
            )
            queue_idempotent_samples.append(idem_elapsed)
            queue_idempotent_call_stats.append(_read_exit_task_call_stats(stats_path))
            queue_idempotent_timing_stats.append(_read_exit_task_timing_stats(stats_path))
            queue_idempotent_outbox_stats.append(_read_exit_outbox_stats(stats_path))
            replay_calls = queue_idempotent_call_stats[-1]
            replay_task_calls = sum(
                value for key, value in replay_calls.items() if key.startswith("run_task_calls")
            )
            replay_rows = replay_calls.get("task_read_rows", 0)
            if replay_task_calls or replay_rows:
                raise RuntimeError(
                    "acknowledged exit replay performed Taskwarrior I/O: "
                    f"calls={replay_task_calls}, rows={replay_rows}"
                )
            if _workflow_outbox_pending(queue_data):
                raise RuntimeError(
                    "idempotent outbox drain left active intents: "
                    f"{_workflow_outbox_pending(queue_data)!r}; stderr={_idem_stderr.strip()!r}"
                )
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
                    f"outbox={lifecycle_outbox.LifecycleOutboxRepository(queue_data).status(limit=20)[1]!r}"
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

            # Exercise a partial batch import: the runner imports only the
            # first child and then fails. The next invocation must requeue,
            # discover that child, import the remaining seven, and converge.
            partial_data = root / f"partial-queue-{sample_index}"
            _init_empty_outbox(partial_data)
            partial_stats_path = partial_data / "on-exit-task-stats.json"
            partial_env = dict(base_env, TASKDATA=str(partial_data), NAUTICAL_BENCH_STATS_FILE=str(partial_stats_path))
            partial_env["NAUTICAL_BENCH_TASK_BIN"] = str(task_wrapper)
            partial_parents, partial_plans = _outbox_lifecycle_fixture("partial", sample_index)
            partial_import_proc = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(parent, ensure_ascii=False) + "\n" for parent in partial_parents),
                text=True,
                capture_output=True,
                env=partial_env,
                timeout=30.0,
            )
            if partial_import_proc.returncode != 0:
                raise RuntimeError("partial queue parent fixture import failed")
            partial_parent_probe = subprocess.run(
                ["task", f"rc.data.location={partial_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on", "export"],
                text=True,
                capture_output=True,
                env=partial_env,
                timeout=30.0,
            )
            if partial_parent_probe.returncode != 0:
                raise RuntimeError("partial queue parent verification export failed")
            partial_plans = _bind_workflow_plans_to_parents(
                partial_plans,
                json.loads(partial_parent_probe.stdout or "[]"),
            )
            _stage_workflow_plans(
                partial_data,
                partial_plans,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
            )
            partial_env["NAUTICAL_BENCH_FAIL_MODE"] = "partial-import"
            try:
                partial_stats_path.unlink()
            except FileNotFoundError:
                pass
            partial_t0 = time.perf_counter()
            _first_partial_elapsed, _first_partial_result, first_partial_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical", input_text="", env=partial_env, expect_output=False
            )
            first_partial_elapsed = time.perf_counter() - partial_t0
            first_partial_stats = _read_exit_task_call_stats(partial_stats_path)
            first_partial_timing = _read_exit_task_timing_stats(partial_stats_path)
            first_partial_outbox = _read_exit_outbox_stats(partial_stats_path)
            pending_after_partial = _workflow_outbox_pending(partial_data)
            if not 1 <= len(pending_after_partial) <= 8:
                raise RuntimeError(
                    "partial import did not requeue all lifecycle intents: "
                    f"{pending_after_partial!r}; stderr={first_partial_stderr.strip()!r}"
                )
            partial_env.pop("NAUTICAL_BENCH_FAIL_MODE", None)
            try:
                partial_stats_path.unlink()
            except FileNotFoundError:
                pass
            recovery_t0 = time.perf_counter()
            _second_partial_elapsed, _second_partial_result, second_partial_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical", input_text="", env=partial_env, expect_output=False
            )
            recovery_elapsed = time.perf_counter() - recovery_t0
            second_partial_stats = _read_exit_task_call_stats(partial_stats_path)
            second_partial_timing = _read_exit_task_timing_stats(partial_stats_path)
            second_partial_outbox = _read_exit_outbox_stats(partial_stats_path)
            if _workflow_outbox_pending(partial_data):
                raise RuntimeError(
                    "partial import recovery left active lifecycle intents: "
                    f"{_workflow_outbox_pending(partial_data)!r}; stderr={second_partial_stderr.strip()!r}"
                )
            queue_partial_first_samples.append(first_partial_elapsed)
            queue_partial_recovery_samples.append(recovery_elapsed)
            queue_partial_samples.append(first_partial_elapsed + recovery_elapsed)
            queue_partial_call_stats.append(_merge_task_call_stats(first_partial_stats, second_partial_stats))
            queue_partial_timing_stats.append(_merge_task_timing_stats(first_partial_timing, second_partial_timing))
            queue_partial_outbox_stats.append(_merge_task_timing_stats(first_partial_outbox, second_partial_outbox))
        queue_result = _measure_workflow(
            "workflow_queue_drain", queue_samples,
            float(budgets.get("workflow_queue_drain", 3.0)),
        )
        queue_result["task_call_stats"] = queue_call_stats
        queue_result["outbox_stats"] = queue_outbox_stats
        _attach_timing_breakdown(queue_result, queue_samples, queue_timing_stats)
        sqlite_budgets = workflow_cfg.get("sqlite_budgets")
        if isinstance(sqlite_budgets, dict):
            _apply_outbox_budgets(queue_result, queue_outbox_stats, sqlite_budgets.get("workflow_queue_drain", {}))
        component_budgets = workflow_cfg.get("component_budgets_seconds")
        if isinstance(component_budgets, dict):
            _apply_component_budgets(
                queue_result, queue_result["timing_breakdown"],
                component_budgets.get("workflow_queue_drain", {}),
            )
        call_budgets = workflow_cfg.get("task_call_budgets")
        if isinstance(call_budgets, dict):
            _apply_task_call_budgets(
                queue_result,
                queue_call_stats,
                call_budgets.get("workflow_queue_drain", {}),
            )
        results["workflow_queue_drain"] = queue_result
        queue_idempotent_result = _measure_workflow(
            "workflow_queue_drain_idempotent",
            queue_idempotent_samples,
            float(budgets.get("workflow_queue_drain_idempotent", 3.0)),
        )
        queue_idempotent_result["task_call_stats"] = queue_idempotent_call_stats
        queue_idempotent_result["outbox_stats"] = queue_idempotent_outbox_stats
        _attach_timing_breakdown(queue_idempotent_result, queue_idempotent_samples, queue_idempotent_timing_stats)
        if isinstance(sqlite_budgets, dict):
            _apply_outbox_budgets(queue_idempotent_result, queue_idempotent_outbox_stats, sqlite_budgets.get("workflow_queue_drain_idempotent", {}))
        if isinstance(component_budgets, dict):
            _apply_component_budgets(
                queue_idempotent_result, queue_idempotent_result["timing_breakdown"],
                component_budgets.get("workflow_queue_drain_idempotent", {}),
            )
        if isinstance(call_budgets, dict):
            _apply_task_call_budgets(
                queue_idempotent_result,
                queue_idempotent_call_stats,
                call_budgets.get("workflow_queue_drain_idempotent", {}),
            )
        results["workflow_queue_drain_idempotent"] = queue_idempotent_result
        queue_partial_result = _measure_workflow(
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
        _attach_timing_breakdown(queue_partial_result, queue_partial_samples, queue_partial_timing_stats)
        if isinstance(sqlite_budgets, dict):
            _apply_outbox_budgets(queue_partial_result, queue_partial_outbox_stats, sqlite_budgets.get("workflow_queue_drain_partial_recovery", {}))
        if isinstance(component_budgets, dict):
            _apply_component_budgets(
                queue_partial_result, queue_partial_result["timing_breakdown"],
                component_budgets.get("workflow_queue_drain_partial_recovery", {}),
            )
        if isinstance(call_budgets, dict):
            _apply_task_call_budgets(
                queue_partial_result,
                queue_partial_call_stats,
                call_budgets.get("workflow_queue_drain_partial_recovery", {}),
            )
        results["workflow_queue_drain_partial_recovery"] = queue_partial_result

        def run_queue_shape(name: str, background_rows: int) -> dict:
            """Measure one healthy intent with an optional unrelated history set."""
            shape_data = root / name
            _init_empty_outbox(shape_data)
            shape_stats_path = shape_data / "on-exit-task-stats.json"
            shape_env = dict(
                base_env,
                TASKDATA=str(shape_data),
                NAUTICAL_BENCH_STATS_FILE=str(shape_stats_path),
                NAUTICAL_BENCH_TASK_BIN=str(real_task),
            )
            parents, plans = _outbox_lifecycle_fixture(name, 0, count=1)
            background = [
                {
                    "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{name}/background/{index}")),
                    "status": "completed",
                    "description": f"Unrelated background history {index}",
                    "due": "20250101T090000Z",
                }
                for index in range(background_rows)
            ]
            imported = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in [*background, *parents]),
                text=True,
                capture_output=True,
                env=shape_env,
                timeout=120.0,
            )
            if imported.returncode != 0:
                raise RuntimeError(f"{name} fixture import failed: {(imported.stderr or imported.stdout or '').strip()}")
            probe = subprocess.run(
                ["task", f"rc.data.location={shape_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on", "export"],
                text=True,
                capture_output=True,
                env=shape_env,
                timeout=120.0,
            )
            if probe.returncode != 0:
                raise RuntimeError(f"{name} fixture probe failed: {(probe.stderr or probe.stdout or '').strip()}")
            plans = _bind_workflow_plans_to_parents(plans, json.loads(probe.stdout or "[]"))
            _stage_workflow_plans(
                shape_data,
                plans,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
            )
            started = time.perf_counter()
            elapsed, _result, stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical", input_text="", env=shape_env, expect_output=False
            )
            if _workflow_outbox_pending(shape_data):
                raise RuntimeError(f"{name} left active outbox work: {_workflow_outbox_pending(shape_data)!r}")
            timing = _read_exit_task_timing_stats(shape_stats_path)
            calls = _read_exit_task_call_stats(shape_stats_path)
            outbox_stats = _read_exit_outbox_stats(shape_stats_path)
            if not calls.get("run_task_calls"):
                raise RuntimeError(f"{name} did not execute Taskwarrior commands: {stderr.strip()!r}")
            result = _measure_workflow(name, [max(elapsed, time.perf_counter() - started)], float(budgets.get(name, 3.5)))
            result["background_rows"] = background_rows
            result["task_call_stats"] = [calls]
            result["outbox_stats"] = [outbox_stats]
            _attach_timing_breakdown(result, [elapsed], [timing])
            return result

        results["workflow_queue_drain_one_intent"] = run_queue_shape("queue-one-intent", 0)
        results["workflow_queue_drain_large_history"] = run_queue_shape(
            "queue-large-history", max(5000, int(workflow_cfg.get("queue_background_history_rows", 5000)))
        )

        reconcile_data = root / "reconcile"
        reconcile_data.mkdir()
        reconcile_env = dict(base_env, TASKDATA=str(reconcile_data))
        history_rows = max(1, int(workflow_cfg.get("reconcile_history_rows", 256)))
        reconcile_tasks = []
        for link in range(1, history_rows + 1):
            task = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile/{link}")),
                "status": "completed",
                "description": f"Reconcile performance benchmark {link}",
                "cp": "P1D",
                "chain": "on",
                "chainID": "reconcile-perf-chain",
                "link": link,
                "due": f"202601{min(link, 28):02d}T090000Z",
            }
            if link > 1:
                task["prevLink"] = reconcile_tasks[-1]["uuid"][:8]
            if link < history_rows:
                task["nextLink"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile/{link + 1}"))[:8]
            reconcile_tasks.append(task)
        import_proc = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in reconcile_tasks),
            text=True,
            capture_output=True,
            env=reconcile_env,
            timeout=30.0,
        )
        if import_proc.returncode != 0:
            raise RuntimeError(f"reconcile fixture import failed: {(import_proc.stderr or import_proc.stdout or '').strip()}")
        reconcile_cmd = [sys.executable, str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"), "--json"]
        reconcile_samples = []
        reconcile_reports: list[dict] = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=reconcile_env, timeout=30.0)
            if proc.returncode != 0:
                raise RuntimeError(f"reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
            report = json.loads(proc.stdout or "{}")
            if (
                not isinstance(report, dict)
                or not 1 <= int(report.get("export_calls", 0)) <= 2
                or not 1 <= int(report.get("export_rows", 0)) <= history_rows
                or float(report.get("integrity_seconds", -1.0)) < 0.0
                or float(report.get("integrity_application_seconds", -1.0)) < 0.0
                or int(report.get("task_command_calls", -1)) < 1
                or int(report.get("task_command_attempts", -1)) < int(report.get("task_command_calls", 0))
                or bool(report.get("task_command_budget_exceeded", True))
            ):
                raise RuntimeError(f"healthy reconcile workflow bounded snapshot budget failed: {report!r}")
            for purpose, count in (report.get("task_command_by_purpose") or {}).items():
                    reconcile_call_purposes[str(purpose)] = max(
                    reconcile_call_purposes.get(str(purpose), 0), int(count)
                )
            reconcile_reports.append(_compact_reconcile_report(report))
            reconcile_samples.append(time.perf_counter() - started)
        results["workflow_reconcile"] = _measure_workflow(
            "workflow_reconcile", reconcile_samples, float(budgets.get("workflow_reconcile", 3.0))
        )
        _attach_reconcile_reports(results["workflow_reconcile"], reconcile_reports)

        # Keep an empty audit as a first-class workload.  This catches startup,
        # snapshot, and report overhead without accidentally measuring stale
        # queue cleanup or a failed Taskwarrior export.
        empty_data = root / "reconcile-empty"
        empty_data.mkdir()
        empty_env = dict(base_env, TASKDATA=str(empty_data))
        empty_samples = []
        empty_reports: list[dict] = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=empty_env, timeout=30.0)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"empty reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("empty reconcile workflow returned invalid JSON") from exc
            if not isinstance(report, dict) or report.get("schema") != "nautical.reconcile":
                raise RuntimeError("empty reconcile workflow returned an invalid report")
            if int(report.get("export_calls", 0)) > 2:
                raise RuntimeError("empty reconcile workflow exceeded its bounded snapshot budget")
            if float(report.get("integrity_seconds", -1.0)) < 0.0:
                raise RuntimeError("empty reconcile workflow omitted integrity timing")
            empty_reports.append(_compact_reconcile_report(report))
            empty_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_empty"] = _measure_workflow(
            "workflow_reconcile_empty",
            empty_samples,
            float(budgets.get("workflow_reconcile_empty", budgets.get("workflow_reconcile", 3.0))),
        )
        _attach_reconcile_reports(results["workflow_reconcile_empty"], empty_reports)

        # Candidate-heavy audits must prove that the benchmark contains
        # actionable integrity evidence.  A zero-candidate run would only
        # measure the healthy path again.
        candidate_data = root / "reconcile-candidates"
        candidate_data.mkdir()
        candidate_env = dict(base_env, TASKDATA=str(candidate_data))
        candidate_count = max(1, int(workflow_cfg.get("reconcile_candidate_chains", 32)))
        candidate_tasks = _reconcile_candidate_tasks("healthy", candidate_count)
        candidate_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in candidate_tasks),
            text=True,
            capture_output=True,
            env=candidate_env,
            timeout=30.0,
        )
        if candidate_import.returncode != 0:
            raise RuntimeError(
                "reconcile candidate fixture import failed: "
                f"{(candidate_import.stderr or candidate_import.stdout or '').strip()}"
            )
        candidate_samples = []
        candidate_reports: list[dict] = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=candidate_env, timeout=30.0)
            if proc.returncode not in (0, 1, 2):
                raise RuntimeError(
                    f"candidate reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("candidate reconcile workflow returned invalid JSON") from exc
            summary = report if isinstance(report, dict) else {}
            if int(summary.get("candidates", 0)) <= 0 and not summary.get("plans"):
                raise RuntimeError("candidate reconcile workflow produced no integrity candidates or plans")
            candidate_reports.append(_compact_reconcile_report(summary))
            candidate_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_candidates"] = _measure_workflow(
            "workflow_reconcile_candidates",
            candidate_samples,
            float(budgets.get("workflow_reconcile_candidates", budgets.get("workflow_reconcile", 3.0))),
        )
        _attach_reconcile_reports(results["workflow_reconcile_candidates"], candidate_reports)

        candidate_apply_data = root / "reconcile-candidates-apply"
        candidate_apply_data.mkdir()
        candidate_apply_env = dict(base_env, TASKDATA=str(candidate_apply_data))
        candidate_apply_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in candidate_tasks),
            text=True,
            capture_output=True,
            env=candidate_apply_env,
            timeout=30.0,
        )
        if candidate_apply_import.returncode != 0:
            raise RuntimeError(
                "candidate apply fixture import failed: "
                f"{(candidate_apply_import.stderr or candidate_apply_import.stdout or '').strip()}"
            )
        apply_started = time.perf_counter()
        apply_proc = subprocess.run(
            [*reconcile_cmd, "--apply"],
            text=True,
            capture_output=True,
            env=candidate_apply_env,
            timeout=60.0,
        )
        if apply_proc.returncode != 0:
            raise RuntimeError(
                "candidate reconcile apply failed: "
                f"{(apply_proc.stderr or apply_proc.stdout or '').strip()}"
            )
        try:
            apply_report = json.loads(apply_proc.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("candidate reconcile apply returned invalid JSON") from exc
        if int(apply_report.get("spawn", 0)) <= 0 or not apply_report.get("applied"):
            raise RuntimeError("candidate reconcile apply did not create guarded successors")
        results["workflow_reconcile_candidates_apply"] = _measure_workflow(
            "workflow_reconcile_candidates_apply",
            [time.perf_counter() - apply_started],
            float(budgets.get("workflow_reconcile_candidates_apply", 12.0)),
        )
        _attach_reconcile_reports(
            results["workflow_reconcile_candidates_apply"],
            [_compact_reconcile_report(apply_report)],
        )

        apply_scale_counts = tuple(
            int(value)
            for value in workflow_cfg.get("reconcile_candidate_apply_counts", (1, 8, 32, 200))
            if int(value) > 0
        )
        apply_scale_samples: list[float] = []
        apply_scale_reports: list[dict] = []
        apply_scale_rows: list[dict] = []
        for scale_count in apply_scale_counts:
            scale_data = root / f"reconcile-candidates-apply-{scale_count}"
            scale_data.mkdir()
            scale_env = dict(base_env, TASKDATA=str(scale_data))
            scale_tasks = _reconcile_candidate_tasks(f"apply-{scale_count}", scale_count)
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
            compact = _compact_reconcile_report(scale_report)
            apply_scale_reports.append(compact)
            apply_scale_rows.append({"candidate_count": scale_count, "elapsed_s": round(elapsed, 6), **compact})
        results["workflow_reconcile_candidates_apply_scale"] = {
            "name": "workflow_reconcile_candidates_apply_scale",
            "candidate_counts": list(apply_scale_counts),
            "samples_s": apply_scale_samples,
            "rows": apply_scale_rows,
            "reconcile_reports": apply_scale_reports,
            "pass": bool(apply_scale_counts) and all(value >= 0.0 for value in apply_scale_samples),
        }

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
            timeout=60.0,
        )
        if long_import.returncode != 0:
            raise RuntimeError(
                "long reconcile fixture import failed: "
                f"{(long_import.stderr or long_import.stdout or '').strip()}"
            )
        long_samples = []
        long_reports: list[dict] = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=long_env, timeout=60.0)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"long reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("long reconcile workflow returned invalid JSON") from exc
            if not isinstance(report, dict) or report.get("schema") != "nautical.reconcile":
                raise RuntimeError("long reconcile workflow returned an invalid report")
            if (
                not 1 <= int(report.get("export_calls", 0)) <= 2
                or not 1 <= int(report.get("export_rows", 0)) <= long_count
                or float(report.get("integrity_seconds", -1.0)) < 0.0
            ):
                raise RuntimeError("long reconcile workflow exceeded its single-snapshot row budget")
            long_reports.append(_compact_reconcile_report(report))
            long_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_long_history"] = _measure_workflow(
            "workflow_reconcile_long_history",
            long_samples,
            float(budgets.get("workflow_reconcile_long_history", budgets.get("workflow_reconcile", 3.0))),
        )
        _attach_reconcile_reports(results["workflow_reconcile_long_history"], long_reports)

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
        corrupt_samples = []
        corrupt_reports: list[dict] = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=corrupt_env, timeout=30.0)
            if proc.returncode not in (0, 1, 2):
                raise RuntimeError(
                    f"corrupted reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("corrupted reconcile workflow returned invalid JSON") from exc
            audit = report.get("integrity_audit") if isinstance(report, dict) else None
            if not isinstance(audit, dict) or not audit.get("findings"):
                raise RuntimeError("corrupted reconcile workflow hid its integrity findings")
            corrupt_reports.append(_compact_reconcile_report(report))
            corrupt_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_corrupted"] = _measure_workflow(
            "workflow_reconcile_corrupted",
            corrupt_samples,
            float(budgets.get("workflow_reconcile_corrupted", budgets.get("workflow_reconcile", 3.0))),
        )
        _attach_reconcile_reports(results["workflow_reconcile_corrupted"], corrupt_reports)

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
        mixed_samples = []
        mixed_reports: list[dict] = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=mixed_env, timeout=30.0)
            if proc.returncode not in (0, 1, 2):
                raise RuntimeError(f"mixed reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("mixed reconcile workflow returned invalid JSON") from exc
            audit = report.get("integrity_audit") if isinstance(report, dict) else None
            if not isinstance(audit, dict) or not audit.get("findings") or int(report.get("candidates", 0)) <= 0:
                raise RuntimeError("mixed reconcile workflow did not preserve both candidate and integrity evidence")
            mixed_reports.append(_compact_reconcile_report(report))
            mixed_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_mixed"] = _measure_workflow(
            "workflow_reconcile_mixed",
            mixed_samples,
            float(budgets.get("workflow_reconcile_mixed", budgets.get("workflow_reconcile", 3.0))),
        )
        _attach_reconcile_reports(results["workflow_reconcile_mixed"], mixed_reports)
        reconcile_budgets = workflow_cfg.get("reconcile_budgets", {})
        if isinstance(reconcile_budgets, dict):
            for name, result in results.items():
                if name.startswith("workflow_reconcile"):
                    _apply_reconcile_budgets(result, reconcile_budgets.get(name, reconcile_budgets.get("default", {})))
        RESOURCE_DETAILS["reconcile_task_call_purposes"] = reconcile_call_purposes
        return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-file", default=str(HERE / "perf_budget.json"))
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--enforce", action="store_true", help="fail non-zero if any budget is exceeded")
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
        r = _measure(name, fn, check_repeats)
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
        elif name == "cold_modify_impl_import":
            r["module_count"] = IMPORT_PROFILES.get("modify_impl", 0)
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
