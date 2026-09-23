"""Operator, queue, and reconcile micro-workloads."""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any, Callable
from types import SimpleNamespace


def reconcile_snapshot() -> float:
    from nautical_core.reconcile_snapshot_service import ReconcileSnapshotService
    from nautical_core.task_models import TaskObservation

    class Repository:
        reads = 0
        def lifecycle_candidates(self, **_kwargs):
            self.reads += 1
            return object()

    rows = tuple(TaskObservation.from_mapping({
        "uuid": f"00000000-0000-4000-8000-{index:012d}",
        "status": "completed" if index % 2 == 0 else "pending",
        "chain": "on", "chainID": f"reconcile-stage-{index // 2:04d}", "link": index + 1,
    }, source_query="perf:reconcile-stage") for index in range(128))
    repository = Repository()
    service = ReconcileSnapshotService(repository, read_value=lambda _value, _label: rows)
    started = time.perf_counter()
    candidates = service.candidate_rows(); active = service.active_rows()
    if repository.reads != 1 or len(candidates) != 64 or len(active) != 64:
        raise RuntimeError(f"reconcile snapshot stage lost projection authority: reads={repository.reads} candidates={len(candidates)} active={len(active)}")
    return time.perf_counter() - started


def queue_stale(
    lifecycle_outbox: Any,
    init_empty_outbox: Callable[[Path], None],
    outbox_lifecycle_fixture: Callable[..., Any],
    stage_workflow_plans: Callable[..., None],
) -> float:
    from nautical_core.queue_status_service import QueueStatusService

    with tempfile.TemporaryDirectory(prefix="nautical-perf-queue-stale-") as td:
        taskdata = Path(td); init_empty_outbox(taskdata)
        _parents, plans = outbox_lifecycle_fixture("stale", 0, count=1)
        stage_workflow_plans(taskdata, plans, configuration_fingerprint="perf", schedule_fingerprint="perf")
        repository = lifecycle_outbox._LifecycleOutboxRepository(taskdata)
        claimed, records = repository.claim_batch(owner="perf-owner", lease_seconds=30.0, limit=1)
        if not claimed.ok or len(records) != 1:
            raise RuntimeError("stale queue fixture could not claim a valid lifecycle plan")
        with sqlite3.connect(str(repository.path)) as connection:
            connection.execute("UPDATE lifecycle_outbox SET lease_expires_at=? WHERE intent_id=?", (time.time() - 10.0, records[0].intent_id))
        started = time.perf_counter()
        payload = QueueStatusService().status_payload(taskdata, stale_after=5.0, limit=5)
        if int(payload.get("outbox", {}).get("stale_claims", 0) or 0) != 1:
            raise RuntimeError(f"queue status did not report the valid stale claim: {payload!r}")
        return time.perf_counter() - started


def operator_interrupted(lifecycle_outbox: Any, init_empty_outbox: Callable[[Path], None], outbox_lifecycle_fixture: Callable[..., Any], stage_workflow_plans: Callable[..., None]) -> float:
    with tempfile.TemporaryDirectory(prefix="nautical-perf-operator-interrupted-") as td:
        taskdata = Path(td); init_empty_outbox(taskdata)
        _parents, plans = outbox_lifecycle_fixture("operator-interrupted", 0, count=1)
        stage_workflow_plans(taskdata, plans, configuration_fingerprint="perf", schedule_fingerprint="perf")
        repository = lifecycle_outbox._LifecycleOutboxRepository(taskdata)
        started = time.perf_counter()
        first, records = repository.claim_batch(owner="interrupted-a", lease_seconds=0.05, limit=1)
        if not first.ok or len(records) != 1:
            raise RuntimeError("interrupted operator fixture could not claim its intent")
        time.sleep(0.08)
        second, reclaimed = repository.claim_batch(owner="interrupted-b", lease_seconds=1.0, limit=1)
        if not second.ok or len(reclaimed) != 1 or reclaimed[0].intent_id != records[0].intent_id:
            raise RuntimeError("expired interrupted operator intent was not reclaimed")
        return time.perf_counter() - started


def exit_probe() -> float:
    from nautical_core.exit_probe import probe_exit_work
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nautical-perf-exit-probe-") as td:
        taskdata = Path(td)
        empty = probe_exit_work(taskdata)
        if not empty.definitely_empty: raise RuntimeError(f"empty exit probe reported possible work: {empty.reason}")
        if not _LifecycleOutboxRepository(taskdata).open().ok: raise RuntimeError("exit probe fixture outbox could not be initialized")
        terminal = probe_exit_work(taskdata)
        if not terminal.definitely_empty: raise RuntimeError(f"terminal exit probe reported possible work: {terminal.reason}")
    return time.perf_counter() - started


def operator_scope() -> float:
    from nautical_core.query_models import OccurrenceQueryRequest
    from nautical_core.query_service import OccurrenceQueryService
    started = time.perf_counter()
    service = object.__new__(OccurrenceQueryService)
    service._timezone = __import__("datetime").timezone.utc
    service._scheduler_cache = {}
    service._uow = SimpleNamespace(mutation_epoch=0, context=SimpleNamespace(configuration=SimpleNamespace(fingerprint="scope-matrix")))
    request = OccurrenceQueryRequest.from_mapping({"selector": {"all_tasks": True}, "from": "2026-08-24", "count": 1, "max_tasks": 2})
    empty, empty_cursor, empty_complete = service._page_rows((), request)
    if empty or empty_cursor is not None or not empty_complete: raise RuntimeError("empty operator scope was not a complete page")
    one_page, one_cursor, one_complete = service._page_rows((SimpleNamespace(uuid="scope-one"),), request)
    if tuple(row.uuid for row in one_page) != ("scope-one",) or one_cursor is not None or not one_complete: raise RuntimeError("single-item operator scope was not complete")
    boundary = tuple(SimpleNamespace(uuid=f"scope-{index}") for index in range(2))
    page, cursor, complete = service._page_rows(boundary, request)
    if len(page) != 2 or cursor is not None or not complete: raise RuntimeError("boundary-sized operator scope produced a continuation")
    page, cursor, complete = service._page_rows(boundary + (SimpleNamespace(uuid="scope-2"),), request)
    if len(page) != 2 or cursor is None or complete: raise RuntimeError("boundary-plus-one operator scope omitted its continuation")
    return time.perf_counter() - started


def operator_failure_matrix(
    query_pagination: Callable[[], float],
    query_unavailable: Callable[[], float],
    repair_planner: Callable[[], float],
    queue_stale: Callable[[], float],
    configuration_verification: Callable[[Any], Any],
) -> float:
    """Exercise fail-closed operator boundaries in one content-free matrix."""
    from nautical_core.operator_health_service import OperatorHealthService

    started = time.perf_counter()
    query_pagination(); query_unavailable(); repair_planner(); queue_stale()
    doctor_findings = OperatorHealthService.configuration_schema_findings({"panel_mode": 17})
    if not doctor_findings or not any(item.code == "config.schema.type" for item in doctor_findings):
        raise RuntimeError("Doctor failure matrix accepted malformed configuration")

    class BrokenCore:
        @staticmethod
        def configuration_drift():
            raise RuntimeError("synthetic configuration read failure")

    reconcile_state = configuration_verification(SimpleNamespace(core=BrokenCore()))
    if reconcile_state.status != "unavailable" or "configuration verification unavailable" not in reconcile_state.reason:
        raise RuntimeError(f"reconcile failure matrix did not fail closed: {reconcile_state!r}")
    return time.perf_counter() - started
