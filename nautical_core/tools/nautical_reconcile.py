#!/usr/bin/env python3
"""Repair Nautical chains missing a successor after completion or expiration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("NAUTICAL_CORE_PATH", str(BASE_DIR))

import nautical_core as nautical_core_package  # noqa: E402
from nautical_core import chain_integrity_lifecycle as lifecycle, safe_lock  # noqa: E402
from nautical_core.lifecycle_state import parent_nextlink_lock_path, reconcile_lock_path  # noqa: E402
from nautical_core import modify_spawn_prep  # noqa: E402
from nautical_core.chain_generation import ChainGenerationService  # noqa: E402
from nautical_core.integration_context import IntegrationAccess  # noqa: E402
from nautical_core.lifecycle_models import (  # noqa: E402
    DeletionDisposition,
    LifecycleAction,
    LifecycleEvent,
    LifecyclePlan,
    TaskSnapshot,
    recurrence_fingerprint,
)
from nautical_core.lifecycle_planner import terminal_plan_for_snapshot  # noqa: E402
from nautical_core.integration_models import (  # noqa: E402
    Absent,
    Found,
    GuardTimestamp,
    GuardTimestampField,
    MutationGuard,
    MutationOperation,
    MutationOutcomeKind,
    MutationRequest,
    NativeUntilRepairPayload,
    Unavailable,
)
from nautical_core.task_read_repository import ALL_TASK_STATUSES, TaskReadRepository  # noqa: E402
from nautical_core.timeutil import compare_datetimes  # noqa: E402
from nautical_core.taskwarrior_uow import (  # noqa: E402
    TaskwarriorUnitOfWork,
    build_operator_uow,
)
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService  # noqa: E402
from nautical_core.reconcile_cli import build_parser  # noqa: E402
from nautical_core.reconcile_report import exit_code, render_human, render_json  # noqa: E402
from nautical_core.integrity_report import components as integrity_components  # noqa: E402
from nautical_core.lifecycle_reconciliation import (  # noqa: E402
    CallbackLifecycleApplyOperations,
    CallbackLifecycleRecoveryOperations,
    LifecycleReconciliationService,
)


_PARENT_LOCK_RETRIES = 600
_PARENT_LOCK_SLEEP_SECONDS = 0.1
_PARENT_LOCK_STALE_SECONDS = 300.0
_RECONCILE_LOCK_STALE_SECONDS = 300.0
_DEFAULT_EXPIRATION_HOPS = 32
_MAX_EXPIRATION_HOPS = 1000
_STABLE_CHILD_UUID_NAMESPACE = uuid.UUID("1f4b2396-df58-5a32-a879-33f0d3fe711f")
_JSON_SCHEMA = "nautical.reconcile"
_JSON_SCHEMA_VERSION = 1
_EXPORT_STATS = {"calls": 0, "rows": 0, "seconds": 0.0, "slowest_seconds": 0.0, "snapshot_hits": 0}
_LOCK_STATS = {"reconcile_busy": 0, "parent_busy": 0}
_UNIT_OF_WORK: TaskwarriorUnitOfWork | None = None


def _opportunistic_housekeeping(taskdata: Path) -> dict[str, Any]:
    """Run bounded outbox maintenance without involving Taskwarrior."""
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    result = LifecycleOutboxRepository(taskdata).opportunistic_housekeeping()
    return {
        "status": "skipped" if result.skipped else ("ok" if result.ok else "deferred"),
        "kind": result.kind.value,
        "removed": result.removed,
        "retention_seconds": result.retention_seconds,
        "checkpoint": result.checkpoint,
        "reason": result.reason,
    }


class _ConfigurationDrift(RuntimeError):
    """Signal that this run must stop before applying under a new configuration."""


class _LifecycleRetryable(TimeoutError):
    """Signal that a lifecycle read/write should remain retryable."""


class _LifecycleManualReview(RuntimeError):
    """Signal that a lifecycle transition needs operator review."""


class _RecoveryLookupUnavailable(TimeoutError):
    """Signal that a narrow recovery-child read must be retried."""


class _PlanReadUnavailable(TimeoutError):
    """Signal that planning reads are unavailable and must not be treated as empty."""


class _ConfigurationVerification:
    """Validated configuration state used to gate reconcile mutations."""

    __slots__ = ("status", "reason")

    def __init__(self, status: str, reason: str = "") -> None:
        self.status = status
        self.reason = reason


class _ConfigurationReason(str):
    """String-compatible reason retaining the tri-state verification result."""

    status: str

    def __new__(cls, reason: str, status: str):
        value = str.__new__(cls, reason)
        value.status = status
        return value


from nautical_core.native_until_integrity import NativeUntilAudit, audit_result
from nautical_core.chain_integrity_recovery import IntegrityRecoveryService


def _native_until_audit_result(
    repairs: list[dict[str, Any]],
    errors: list[str],
) -> NativeUntilAudit:
    return audit_result(repairs, errors)

_ANSI = {
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "reset": "\033[0m",
}


def _style(text: str, color: str, *, stream: Any = None) -> str:
    """Apply terminal color only for interactive, color-capable output."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or not getattr(stream, "isatty", lambda: False)():
        return text
    return f"{_ANSI.get(color, '')}{text}{_ANSI['reset']}"


def _action_style(action: str) -> str:
    return {
        "spawn": "green",
        "backfill_nextlink": "cyan",
        "repair_until": "green",
        "legitimate_final": "yellow",
        "manual_stop": "yellow",
        "stale": "dim",
        "partial": "yellow",
        "error": "red",
        "repair_error": "red",
        "manual_review": "yellow",
    }.get(action, "cyan")


def _runtime_core(runtime: Any) -> Any:
    return getattr(runtime, "core", runtime)


def _format_local_until(hook: Any, value: Any) -> str:
    """Render a repaired native-until target in configured local time when possible."""
    raw = str(value or "").strip()
    if not raw:
        return raw
    parser = getattr(hook, "safe_parse_datetime", None)
    if not callable(parser):
        parser = getattr(hook, "_safe_parse_datetime", None)
    formatter = getattr(_runtime_core(hook), "fmt_dt_local", None)
    if not callable(formatter):
        return raw
    try:
        parsed, error = _safe_parse_datetime(hook, raw)
        if parsed is not None and not error:
            return str(formatter(parsed))
    except Exception:
        pass
    return raw


def _safe_parse_datetime(hook: Any, value: Any):
    # Heavy hook implementations keep this boundary private because they are
    # loaded as executable modules.  Prefer the public adapter when present,
    # then use the hook's typed private parser; never silently treat a missing
    # parser as a valid/absent timestamp.
    parser = getattr(hook, "safe_parse_datetime", None)
    if not callable(parser):
        parser = getattr(hook, "_safe_parse_datetime", None)
    if callable(parser):
        return parser(value)
    parser = getattr(_runtime_core(hook), "parse_dt_any", None)
    if not callable(parser):
        return None, "datetime parser unavailable"
    try:
        parsed = parser(value)
    except Exception as exc:
        return None, str(exc).strip() or type(exc).__name__
    return parsed, None if parsed is not None else f"unrecognized datetime: {value}"


def _stable_child_uuid(hook: Any, parent: dict[str, Any], child: dict[str, Any]) -> str:
    resolver = getattr(hook, "stable_child_uuid", None)
    if callable(resolver):
        return str(resolver(parent, child) or "")
    core = _runtime_core(hook)
    coerce_int = getattr(core, "coerce_int", None)
    if not callable(coerce_int):
        return ""
    return modify_spawn_prep.stable_child_uuid(
        parent,
        child,
        task_uuid_or_empty=lambda task: str(task.get("uuid") or "").strip(),
        coerce_int=coerce_int,
        stable_child_uuid_namespace=_STABLE_CHILD_UUID_NAMESPACE,
    )


def _repository() -> TaskReadRepository:
    state = _reconcile_runtime_state()
    if state is None:
        raise RuntimeError("reconcile task read repository is unavailable")
    return state.repository


def _read_value(read: Any, subject: str) -> Any | None:
    if isinstance(read, Found):
        return read.value
    if isinstance(read, Absent):
        return None
    if isinstance(read, Unavailable):
        raise _PlanReadUnavailable(f"{subject} unavailable: {read.evidence.detail}")
    raise _PlanReadUnavailable(f"{subject} returned an invalid typed result")


def _configuration_drift_reason(hook: Any) -> str:
    """Compatibility string for callers; failures are never treated as valid."""
    check = _configuration_verification(hook)
    if check.status == "valid":
        return ""
    return _ConfigurationReason(check.reason, check.status)


def _configuration_verification(hook: Any) -> _ConfigurationVerification:
    """Return valid, drifted, or unavailable configuration state."""
    core = _runtime_core(hook)
    checker = getattr(core, "configuration_drift", None)
    if not callable(checker):
        # Lightweight operator-test doubles predate the facade verifier. A
        # real imported Nautical module must never silently bypass this gate.
        if not isinstance(core, ModuleType):
            return _ConfigurationVerification("valid")
        return _ConfigurationVerification(
            "unavailable",
            "configuration verifier is unavailable; restart and rerun",
        )
    try:
        drift = checker()
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        return _ConfigurationVerification(
            "unavailable",
            f"configuration verification unavailable: {reason}; restart and rerun",
        )
    if not isinstance(drift, dict):
        return _ConfigurationVerification(
            "unavailable",
            "configuration verification returned an invalid result; restart and rerun",
        )
    if not drift.get("changed"):
        return _ConfigurationVerification("valid")
    source = str(drift.get("source") or "unknown")
    return _ConfigurationVerification(
        "drifted",
        f"configuration changed during reconcile (source: {source}); restart and rerun",
    )


def _configuration_state(hook: Any) -> tuple[str, str]:
    """Resolve the compatibility reason while retaining its tri-state status."""
    reason = _configuration_drift_reason(hook)
    return str(getattr(reason, "status", "drifted" if reason else "valid")), str(reason)


class _ReconcileSnapshot:
    """Immutable read-phase views for active links and recovery candidates.

    By default completed/deleted history is limited to rows without a successor;
    ``full_audit`` explicitly restores complete history for deep validation.
    Native-until repair reads a predecessor by chain/link only when an active
    row actually needs repair.
    """

    def __init__(
        self,
        repository: TaskReadRepository,
        *,
        scope_filter: str | None = None,
        full_audit: bool = False,
    ):
        self.repository = repository
        self.scope_filter = str(scope_filter or "").strip() or None
        self.full_audit = bool(full_audit)
        self._rows: tuple[dict[str, Any], ...] | None = None

    def _all_rows(self) -> tuple[dict[str, Any], ...]:
        if self._rows is None:
            value = _read_value(
                self.repository.lifecycle_candidates(
                    statuses=ALL_TASK_STATUSES,
                    scope_filter=self.scope_filter,
                    bounded=not self.full_audit,
                ),
                "reconcile lifecycle snapshot",
            )
            self._rows = tuple(dict(row) for row in (value or ()))
        else:
            _EXPORT_STATS["snapshot_hits"] += 1
        return self._rows

    def active_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in self._all_rows()
            if str(row.get("chainID") or "").strip()
            and str(row.get("status") or "").strip().lower() not in {"completed", "deleted"}
        ]

    def candidate_rows(self) -> list[dict[str, Any]]:
        return [
            row for row in self._all_rows()
            if str(row.get("chainID") or "").strip()
            and not str(row.get("nextLink") or "").strip()
            and str(row.get("status") or "").strip().lower() in {"completed", "deleted"}
        ]


class _ReconcileRuntimeState:
    """Invocation-scoped read/service state; never shared between runs."""

    __slots__ = ("repository", "snapshot", "lifecycle_service")

    def __init__(
        self,
        repository: TaskReadRepository,
        snapshot: _ReconcileSnapshot,
        lifecycle_service: LifecycleReconciliationService,
    ) -> None:
        self.repository = repository
        self.snapshot = snapshot
        self.lifecycle_service = lifecycle_service


_RECONCILE_RUNTIME: ContextVar[_ReconcileRuntimeState | None] = ContextVar(
    "nautical_reconcile_runtime", default=None,
)


def _reconcile_runtime_state() -> _ReconcileRuntimeState | None:
    return _RECONCILE_RUNTIME.get()


def _lifecycle_reconciliation_service() -> LifecycleReconciliationService:
    state = _reconcile_runtime_state()
    if state is not None:
        return state.lifecycle_service
    raise RuntimeError("lifecycle reconciliation service requires an invocation snapshot")


def _active_chain_rows(
    task_bin: str,
    *,
    include_inactive: bool = False,
    snapshot: _ReconcileSnapshot | None = None,
) -> list[dict[str, Any]]:
    """Export live Nautical links for integrity checks, independently of recovery candidates."""
    if snapshot is None:
        raise RuntimeError("active-chain reads require an authoritative snapshot")
    rows = snapshot.active_rows()
    return sorted(
        (
            row
            for row in rows
            if str(row.get("status") or "").strip().lower() not in {"completed", "deleted"}
        ),
        key=IntegrityRecoveryService.candidate_sort_key,
    )


def _native_until_guard_error(expected: dict[str, Any], fresh: dict[str, Any]) -> str | None:
    """Detect target or recurrence changes made after the audit export."""
    fields = (
        "uuid", "status", "chain", "chainID", "link", "due", "scheduled", "until",
        "anchor", "anchor_file", "cp", "chainMax", "chainUntil",
    )
    for field in fields:
        left = expected.get(field)
        right = fresh.get(field)
        if field == "link":
            left = lifecycle.int_or_default(left, 0)
            right = lifecycle.int_or_default(right, 0)
        else:
            left = str(left or "").strip()
            right = str(right or "").strip()
        if left != right:
            return f"native-until target changed ({field}: {left or '<empty>'} -> {right or '<empty>'})"
    return None


def _fresh_native_until_previous(row: dict[str, Any]) -> dict[str, Any] | None:
    chain_id = str(row.get("chainID") or "").strip()
    link = lifecycle.int_or_default(row.get("link"), 0)
    if not chain_id or link <= 1:
        return None
    value = _read_value(
        _repository().predecessor_slot(chain_id, link - 1, refresh=True),
        f"predecessor {chain_id}:{link - 1}",
    )
    return dict(value) if value is not None else None


def _native_until_repairs(
    task_bin: str,
    hook: Any,
    *,
    apply: bool,
    taskdata: Path | None = None,
    snapshot: _ReconcileSnapshot | None = None,
    lease_held: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find invalid native windows and repair only those with a reliable predecessor."""
    active_rows = _active_chain_rows(
        task_bin,
        include_inactive=False,
        snapshot=snapshot or (_reconcile_runtime_state().snapshot if _reconcile_runtime_state() is not None else None),
    )
    rows = active_rows
    by_chain_link = {
        (
            str(row.get("chainID") or "").strip(),
            lifecycle.int_or_default(row.get("link"), 0),
        ): row
        for row in active_rows
    }
    recovery_audit = IntegrityRecoveryService().audit_native_until(
        active_rows,
        predecessor=_fresh_native_until_previous,
        safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
        fmt_isoz=_runtime_core(hook).fmt_isoz,
        utc_to_local_naive=_runtime_core(hook).utc_to_local_naive,
        local_naive_to_utc=_runtime_core(hook).local_naive_to_utc,
    )
    if not apply:
        return list(recovery_audit.native_until.repairs), list(recovery_audit.native_until.errors)
    repairs = [dict(item) for item in recovery_audit.native_until.repairs]
    errors = list(recovery_audit.native_until.errors)
    repair_items = {
        (str(item.get("chainID") or "").strip(), lifecycle.int_or_default(item.get("link"), 0)): item
        for item in repairs
        if item.get("action") == "repair_until"
    }
    for candidate in recovery_audit.candidates:
        row = candidate.row
        previous = candidate.previous
        chain_id = str(row.get("chainID") or "").strip()
        link = lifecycle.int_or_default(row.get("link"), 0)
        item = repair_items.get((chain_id, link))
        if item is None:
            continue
        repaired = str(item.get("new_until") or "").strip()
        if apply:
            if taskdata is None:
                item["action"] = "repair_error"
                item["repair_error"] = "Taskwarrior data location is unavailable for native-until locking"
                errors.append(f"{item['task']} chain {chain_id} link {link}: {item['repair_error']}")
                continue
            with _reconcile_mutation_lock(taskdata, lease_held=lease_held) as reconcile_acquired:
                if not reconcile_acquired:
                    _LOCK_STATS["reconcile_busy"] += 1
                    item["action"] = "repair_error"
                    item["repair_error"] = "another reconcile apply is already running"
                else:
                    parent_lock = _parent_apply_lock(taskdata, str(row.get("uuid") or ""))
                    with parent_lock as acquired:
                        if not acquired:
                            _LOCK_STATS["parent_busy"] += 1
                            item["action"] = "repair_error"
                            item["repair_error"] = "native-until repair lock busy"
                        else:
                            fresh = _fresh_parent(row)
                            guard_error = _native_until_guard_error(row, fresh) if fresh else "native-until target disappeared"
                            fresh_previous = _fresh_native_until_previous(fresh or row)
                            if not guard_error:
                                if (previous is None) != (fresh_previous is None):
                                    guard_error = "native-until predecessor changed during repair"
                                elif previous is not None and fresh_previous is not None:
                                    guard_error = _native_until_guard_error(previous, fresh_previous)
                            if guard_error:
                                item["action"] = "repair_error"
                                item["repair_error"] = guard_error
                            else:
                                configuration_status, drift_reason = _configuration_state(hook)
                                if configuration_status != "valid":
                                    item["action"] = "manual_review"
                                    item["repair_error"] = drift_reason
                                    item["configuration_drift"] = True
                                    item["configuration_status"] = configuration_status
                                    return repairs, errors
                                try:
                                    _modify_native_until(task_bin, fresh, repaired)
                                    verified = _fresh_parent(fresh)
                                    if verified is None or not _native_until_matches(verified, repaired, hook):
                                        actual = str((verified or {}).get("until") or "<missing>")
                                        item["action"] = "repair_error"
                                        item["repair_error"] = (
                                            f"native until repair verification failed (expected {repaired}; found {actual})"
                                        )
                                    else:
                                        item["applied"] = True
                                        by_chain_link[(chain_id, link)] = dict(verified)
                                except Exception as exc:
                                    item["action"] = "repair_error"
                                    item["repair_error"] = str(exc).strip() or type(exc).__name__
                if item.get("action") == "repair_error":
                    errors.append(f"{item['task']} chain {chain_id} link {link}: {item['repair_error']}")
    return repairs, errors


def _modify_native_until(task_bin: str, row: dict[str, Any], new_until: str) -> None:
    del task_bin
    if _UNIT_OF_WORK is None:
        raise RuntimeError("native until repair requires an integration unit of work")
    uuid = str(row.get("uuid") or "").strip()
    chain_id = str(row.get("chainID") or "").strip()
    link = lifecycle.int_or_default(row.get("link"), 0)
    modified = str(row.get("modified") or "").strip()
    expected_until = str(row.get("until") or "").strip()
    if not uuid or not chain_id or link <= 0 or not modified or not expected_until:
        raise RuntimeError("native until repair lacks task identity")
    from nautical_core.lifecycle_models import recurrence_fingerprint

    guard = MutationGuard(
        task_uuid=uuid,
        status=str(row.get("status") or ""),
        chain_id=chain_id,
        link=link,
        recurrence_identity=recurrence_fingerprint(row),
        timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=_UNIT_OF_WORK.mutation_epoch,
        chain=str(row.get("chain") or "on"),
    )
    request = MutationRequest(
        MutationOperation.NATIVE_UNTIL_REPAIR,
        guard,
        NativeUntilRepairPayload(uuid, expected_until, str(new_until)),
    )
    outcome = TaskwarriorMutationService(_UNIT_OF_WORK).apply(request)
    if outcome.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}:
        raise RuntimeError(outcome.reason or outcome.kind.value)


def _native_until_matches(fresh: dict[str, Any], expected: str, hook: Any) -> bool:
    """Compare native-until timestamps by instant, tolerating Taskwarrior formatting."""
    actual = str(fresh.get("until") or "").strip()
    if actual == str(expected or "").strip():
        return True
    try:
        actual_dt, actual_err = _safe_parse_datetime(hook, actual)
        expected_dt, expected_err = _safe_parse_datetime(hook, expected)
        return not actual_err and not expected_err and actual_dt is not None and actual_dt == expected_dt
    except Exception:
        return False


def _recovery_existing_children(parent: dict[str, Any]) -> list[dict[str, Any]]:
    """Adapt the authoritative repository read to the recovery service."""
    return IntegrityRecoveryService(
        child_lookup=lambda chain_id, link: _read_value(
            _repository().exact_child_slot(chain_id, link, refresh=True),
            f"child slot {chain_id}:{link}",
        ),
    ).existing_children(parent)


def _existing_children_for_plan(task_bin: str, parent: dict[str, Any], hook: Any) -> list[dict[str, Any]]:
    if str(parent.get("status") or "").strip() == "deleted":
        evidence = lifecycle.deleted_chain_disposition(
            parent,
            safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
        )
        if evidence.disposition is not DeletionDisposition.EXPIRATION:
            return []
    return _recovery_existing_children(parent)


def _expiration_hop_limit(value: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError("expiration hop limit must be an integer") from exc
    if parsed < 1 or parsed > _MAX_EXPIRATION_HOPS:
        raise argparse.ArgumentTypeError(
            f"expiration hop limit must be between 1 and {_MAX_EXPIRATION_HOPS}"
        )
    return parsed


@contextmanager
def _parent_apply_lock(taskdata: Path, parent_uuid: str):
    lock_path = parent_nextlink_lock_path(taskdata, parent_uuid)
    with safe_lock(
        lock_path,
        retries=_PARENT_LOCK_RETRIES,
        sleep_base=_PARENT_LOCK_SLEEP_SECONDS,
        stale_after=_PARENT_LOCK_STALE_SECONDS,
    ) as acquired:
        yield acquired


@contextmanager
def _reconcile_apply_lock(taskdata: Path):
    """Serialize reconciler mutations without blocking a second invocation."""
    lock_path = reconcile_lock_path(taskdata)
    with safe_lock(
        lock_path,
        retries=1,
        sleep_base=0.0,
        stale_after=_RECONCILE_LOCK_STALE_SECONDS,
    ) as acquired:
        yield acquired


@contextmanager
def _reconcile_mutation_lock(taskdata: Path, *, lease_held: bool):
    """Reuse the run lease when present, otherwise protect a direct mutation call."""
    if lease_held:
        yield True
        return
    with _reconcile_apply_lock(taskdata) as acquired:
        yield acquired


def _fresh_parent(parent: dict[str, Any]) -> dict[str, Any] | None:
    parent_uuid = str(parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    value = _read_value(
        _repository().verification(parent_uuid),
        f"parent {parent_uuid}",
    )
    return dict(value) if value is not None else None


def _parent_identity_error(parent: dict[str, Any]) -> str:
    """Explain why a parent cannot be used as an atomic reconcile target."""
    chain_id = str(parent.get("chainID") or "").strip()
    if not chain_id:
        return "parent chainID is missing"

    raw_link = parent.get("link")
    if raw_link is None or not str(raw_link).strip():
        return "parent link is missing; post-v2 reconcile requires a stamped link; run chain-repair --apply if deterministic"
    if isinstance(raw_link, bool):
        return f"parent link is invalid: {raw_link!r}"
    try:
        parsed_link = int(raw_link)
    except (TypeError, ValueError, OverflowError):
        return f"parent link is invalid: {raw_link!r}"
    if parsed_link <= 0:
        return f"parent link must be positive; got {parsed_link}"
    return ""


def _parent_guard_filters(parent: dict[str, Any]) -> list[str]:
    parent_uuid = str(parent.get("uuid") or "").strip()
    status = str(parent.get("status") or "").strip().lower()
    chain_id = str(parent.get("chainID") or "").strip()
    link = lifecycle.int_or_default(parent.get("link"), 0)
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    if status not in {"completed", "deleted"}:
        raise RuntimeError("parent status is no longer reconcilable")
    if str(parent.get("chain") or "").strip().lower() != "on":
        raise RuntimeError("parent chain is no longer active")
    identity_error = _parent_identity_error(parent)
    if identity_error:
        raise RuntimeError(identity_error)
    if str(parent.get("nextLink") or "").strip():
        raise RuntimeError("parent nextLink is already set")
    return [
        f"uuid:{parent_uuid}",
        f"status:{status}",
        "chain:on",
        f"chainID:{chain_id}",
        f"link:{link}",
        "nextLink:",
    ]


def _verify_disabled_parent(task_bin: str, parent: dict[str, Any]) -> None:
    """Re-export a terminal parent before reporting chain disablement as applied."""
    fresh_parent = _fresh_parent(parent)
    if fresh_parent is None:
        raise RuntimeError("post-apply verification could not re-export the disabled parent")
    if str(fresh_parent.get("chain") or "").strip().lower() != "off":
        shown = str(fresh_parent.get("chain") or "<empty>").strip() or "<empty>"
        raise RuntimeError(f"post-apply verification found parent chain {shown}; expected off")
    successor = str(fresh_parent.get("nextLink") or "").strip()
    if successor:
        raise RuntimeError(
            f"post-apply verification found successor {successor}; "
            "terminal chain must not remain spawnable"
        )


def _verify_applied_child(
    task_bin: str,
    parent: dict[str, Any],
    child_short: str,
    *,
    hook: Any = None,
    strict_uuid: bool = False,
) -> dict[str, Any]:
    """Re-export both sides of an apply before declaring the repair successful."""
    expected_child = str(child_short or "").strip().lower()
    if not expected_child:
        raise RuntimeError("post-apply verification has no child identity")
    fresh_parent = _fresh_parent(parent)
    if fresh_parent is None:
        raise RuntimeError("post-apply verification could not re-export the parent")
    if str(fresh_parent.get("chainID") or "").strip() != str(parent.get("chainID") or "").strip():
        raise RuntimeError("post-apply verification found a changed parent chainID")
    linked_child = str(fresh_parent.get("nextLink") or "").strip().lower()
    if linked_child != expected_child and not linked_child.startswith(expected_child):
        shown = linked_child or "<empty>"
        raise RuntimeError(
            f"post-apply verification found parent nextLink {shown}; expected {child_short}"
        )
    rows = _recovery_existing_children(fresh_parent)
    resolved, child_error = lifecycle.resolve_existing_child(
        fresh_parent,
        rows,
        include_deleted=True,
    )
    if child_error:
        raise RuntimeError(f"post-apply child verification failed: {child_error}")
    if resolved.lower() != expected_child:
        shown = resolved or "<missing>"
        raise RuntimeError(
            f"post-apply child verification found {shown}; expected {child_short}"
        )
    matched = next(
        (
            row
            for row in rows
            if str(row.get("uuid") or "").strip().lower().startswith(expected_child)
        ),
        None,
    )
    if matched is None:
        raise RuntimeError("post-apply child verification could not identify the resolved child")
    if callable(getattr(hook, "stable_child_uuid", None)):
        expected_uuid = _stable_child_uuid(hook, fresh_parent, matched).strip().lower()
        actual_uuid = str(matched.get("uuid") or "").strip().lower()
        if strict_uuid and expected_uuid and actual_uuid != expected_uuid:
            raise RuntimeError(
                f"post-apply child UUID {actual_uuid[:8] or '<empty>'} "
                f"does not match deterministic slot identity {expected_uuid[:8]}"
            )
    return matched


def _stale_plan(parent: dict[str, Any], reason: str) -> lifecycle.LifecycleRecoveryDecision:
    return lifecycle.LifecycleRecoveryDecision(
        "stale",
        parent,
        lifecycle.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _chain_generation_for_hook(hook: Any) -> ChainGenerationService:
    """Build the shared generator from configured core state only."""
    provided = getattr(hook, "chain_generation_service", None)
    if isinstance(provided, ChainGenerationService):
        return provided
    core = _runtime_core(hook)
    if core is None:
        raise RuntimeError("configured Nautical core is unavailable")
    if not callable(getattr(core, "parse_cp_sequence_tokens", None)):
        return ChainGenerationService.from_hook(hook)
    return ChainGenerationService.from_core(
        core,
        recurrence_update_udas=tuple(getattr(core, "RECURRENCE_UPDATE_UDAS", ()) or ()),
        debug_wait_sched=bool(getattr(core, "DEBUG_WAIT_SCHED", False)),
    )


def _refresh_plan(
    task_bin: str,
    hook: Any,
    original_parent: dict[str, Any],
    *,
    generation: ChainGenerationService | None = None,
) -> lifecycle.LifecycleRecoveryDecision:
    parent = _fresh_parent(original_parent)
    if parent is None:
        return _stale_plan(original_parent, "parent no longer exists")
    status = str(parent.get("status") or "").strip().lower()
    if status == "completed":
        candidate = lifecycle.is_orphan_completion_candidate(parent)
    elif status == "deleted":
        candidate = lifecycle.is_orphan_deleted_chain_candidate(parent)
    else:
        candidate = False
    if not candidate:
        reason = (
            "parent nextLink already set"
            if str(parent.get("nextLink") or "").strip()
            else "parent no longer needs reconciliation"
        )
        return _stale_plan(parent, reason)
    return _plan_for_parent(
        task_bin,
        hook,
        parent,
        generation=generation or _chain_generation_for_hook(hook),
    )


def _plan_for_parent(
    task_bin: str,
    hook: Any,
    parent: dict[str, Any],
    *,
    generation: ChainGenerationService | None = None,
) -> lifecycle.LifecycleRecoveryDecision:
    """Build the one reconcile plan used by both preview and apply paths."""
    configuration_status, configuration_reason = _configuration_state(hook)
    if configuration_status != "valid":
        raise _ConfigurationDrift(configuration_reason)
    try:
        return _lifecycle_reconciliation_service().plan(
            parent,
            hook=hook,
            generation=generation or _chain_generation_for_hook(hook),
            safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
        )
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        raise _PlanReadUnavailable(f"reconcile child read unavailable: {reason}") from exc


def _integrity_request_factory(operation: Any) -> Any:
    """Build a fresh guarded metadata request for one integrity operation."""
    if _UNIT_OF_WORK is None:
        raise RuntimeError("integrity repair requires an integration unit of work")
    from nautical_core.integration_models import (
        GuardTimestamp,
        GuardTimestampField,
        MetadataRepairPayload,
        MutationGuard,
        MutationOperation,
        MutationRequest,
    )

    read = _repository().by_uuid(operation.target_uuid, refresh=True)
    row = _read_value(read, f"integrity target {operation.target_uuid}")
    if row is None:
        raise RuntimeError(f"integrity target {operation.target_uuid} is unavailable")
    row = dict(row)
    modified = str(row.get("modified") or "").strip()
    if not modified:
        raise RuntimeError("integrity target has no modified timestamp")
    link = lifecycle.int_or_default(row.get("link"), 0)
    if link < 0:
        raise RuntimeError("integrity target has an invalid link")
    updates = dict(operation.payload)
    expected = {key: row.get(key) for key in updates}
    guard = MutationGuard(
        task_uuid=str(row.get("uuid") or operation.target_uuid),
        status=str(row.get("status") or "pending"),
        chain_id=str(row.get("chainID") or operation.chain_id),
        link=link,
        recurrence_identity=recurrence_fingerprint(row),
        timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=_UNIT_OF_WORK.mutation_epoch,
        chain=str(row.get("chain") or "on"),
    )
    return MutationRequest(
        MutationOperation.METADATA_REPAIR,
        guard,
        MetadataRepairPayload.from_mapping(guard.task_uuid, updates, expected=expected),
    )


def _drain_integrity_work() -> tuple[Any, ...]:
    """Drain only integrity work through the chain engine's typed boundary."""
    if _UNIT_OF_WORK is None:
        raise RuntimeError("integrity drain requires an integration unit of work")
    from nautical_core.chain_integrity_engine import ChainIntegrityEngine
    from nautical_core.chain_snapshot import ChainSnapshotService
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    configuration = _UNIT_OF_WORK.context.configuration
    engine = ChainIntegrityEngine(
        ChainSnapshotService(_UNIT_OF_WORK, configuration_fingerprint=configuration.fingerprint),
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    return engine.drain(
        LifecycleOutboxRepository(_UNIT_OF_WORK.outbox.taskdata),
        owner=f"reconcile-integrity-{os.getpid()}",
        executor=TaskwarriorMutationService(_UNIT_OF_WORK),
        request_factory=_integrity_request_factory,
    )


def _audit_reconcile_integrity(rows: tuple[dict[str, Any], ...]) -> Any:
    """Audit the authoritative lifecycle export without issuing another export."""
    if _UNIT_OF_WORK is None:
        raise RuntimeError("integrity audit requires an integration unit of work")
    from nautical_core.chain_integrity_engine import ChainIntegrityEngine
    from nautical_core.chain_integrity_models import ChainSnapshot, SnapshotCoverage
    from nautical_core.chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    configuration = _UNIT_OF_WORK.context.configuration
    snapshot_result = ChainSnapshotService(
        _UNIT_OF_WORK,
        configuration_fingerprint=configuration.fingerprint,
    ).from_rows(
        IntegritySnapshotRequest.candidates(complete_chain_history=True),
        rows,
        source="lifecycle.lifecycle_candidates",
        coverage=SnapshotCoverage.CHAIN,
    )
    if not isinstance(snapshot_result, ChainSnapshot):
        raise RuntimeError(f"reconcile lifecycle snapshot rejected: {snapshot_result.reason}")
    snapshot = snapshot_result

    class _NoopProvider:
        def collect(self, _request: Any) -> Any:
            raise RuntimeError("reconcile integrity audit uses its supplied snapshot")

    engine = ChainIntegrityEngine(
        _NoopProvider(),
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    return engine, engine.audit_snapshot(
        snapshot,
        outbox_repository=LifecycleOutboxRepository(_UNIT_OF_WORK.outbox.taskdata),
        mutation_epoch=_UNIT_OF_WORK.mutation_epoch,
    )


def _find_positional_child(lifecycle_plan: LifecyclePlan) -> dict[str, Any] | None:
    """Find a task already occupying this exact chain position, by uuid or
    by (chainID, link, prevLink) match, regardless of whether it carries the
    deterministic stable UUID. Preserves duplicate-avoidance for chains that
    were manually or partially repaired before this run computed one.
    """
    child = lifecycle_plan.child_dict()
    child_uuid = str(child.get("uuid") or "").strip().lower()
    rows = _recovery_existing_children({"chainID": lifecycle_plan.identity.chain_id, "link": lifecycle_plan.identity.source_link})
    parent_short = str(lifecycle_plan.identity.parent_uuid or "").strip().lower()[:8]
    for row in rows:
        row_uuid = str(row.get("uuid") or "").strip().lower()
        if child_uuid and row_uuid == child_uuid:
            return row
        if (
            str(row.get("chainID") or "").strip() == lifecycle_plan.identity.chain_id
            and lifecycle.int_or_default(row.get("link"), None) == lifecycle_plan.identity.target_link
            and str(row.get("prevLink") or "").strip().lower() == parent_short
        ):
            return row
    return None


def _lifecycle_plan_with_resolved_child_uuid(
    recon_plan: "lifecycle.LifecycleRecoveryDecision", hook: Any
) -> "lifecycle.LifecycleRecoveryDecision":
    """Ensure a SPAWN_CHILD plan's child payload targets a real, reproducible UUID.

    Reconcile may re-run against the same broken chain more than once, and
    may run against a chain a human already partially repaired by hand.
    Prefer a task already occupying this exact chain position (found by uuid
    or by chainID+link+prevLink), so a partially-repaired chain doesn't get
    a duplicate child. Otherwise fall back to the deterministic stable UUID,
    so repeated runs against a still-broken chain converge on the same child
    rather than reserving a fresh random one every time.
    """
    lifecycle_plan = recon_plan.lifecycle_plan
    if not isinstance(lifecycle_plan, LifecyclePlan) or lifecycle_plan.action is not LifecycleAction.SPAWN_CHILD:
        return recon_plan
    child = lifecycle_plan.child_dict()
    existing = _find_positional_child(lifecycle_plan)
    resolved_uuid = str(existing.get("uuid") or "").strip() if existing is not None else ""
    if not resolved_uuid:
        resolved_uuid = _stable_child_uuid(hook, recon_plan.parent, child)
    if not resolved_uuid or resolved_uuid == str(child.get("uuid") or "").strip():
        return recon_plan
    child["uuid"] = resolved_uuid
    patch = dict(lifecycle_plan.parent_patch_dict())
    patch["nextLink"] = resolved_uuid[:8]
    resolved_plan = LifecyclePlan.from_mappings(
        identity=lifecycle_plan.identity,
        action=lifecycle_plan.action,
        parent_guard=lifecycle_plan.parent_guard,
        child_payload=child,
        parent_patch=patch,
        expected_postconditions=lifecycle_plan.expected_postconditions,
        max_attempts=lifecycle_plan.max_attempts,
    )
    return lifecycle.LifecycleRecoveryDecision(
        recon_plan.action,
        recon_plan.parent,
        recon_plan.next_link,
        recon_plan.reason,
        child=resolved_plan.child_dict(),
        child_short=resolved_uuid[:8],
        child_due=recon_plan.child_due,
        terminal_kind=recon_plan.terminal_kind,
        lifecycle_plan=resolved_plan,
    )


def _raise_for_lifecycle_outcome(outcome: Any, *, label: str) -> None:
    """Preserve the retryable/manual-review exception contract callers depend on."""
    from nautical_core.lifecycle_application import LifecycleApplicationOutcomeKind

    if outcome.ok:
        return
    if outcome.kind is LifecycleApplicationOutcomeKind.RETRYABLE:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] reconcile {label} retryable: {outcome.reason}", file=sys.stderr)
        raise _LifecycleRetryable(f"lifecycle {label} retryable: {outcome.reason}")
    if os.environ.get("NAUTICAL_DIAG") == "1":
        print(f"[nautical] reconcile {label} review: {outcome.reason}", file=sys.stderr)
    raise _LifecycleManualReview(f"lifecycle {label} requires review: {outcome.reason}")


def _execute_reconcile_lifecycle_plan(
    task_bin: str,
    hook: Any,
    plan: lifecycle.LifecycleRecoveryDecision,
    *,
    verified_children: dict[str, dict[str, Any]] | None,
    label: str,
    strict_uuid: bool,
) -> str:
    """Stage and execute one reconcile spawn/backfill through the shared
    lifecycle application service -- the same staging and execution path the
    live hooks use, so a chain reconcile repairs converges identically to
    whatever the on-exit hook would have produced from the same state.
    """
    lifecycle_plan = getattr(plan, "lifecycle_plan", None)
    if not isinstance(lifecycle_plan, LifecyclePlan):
        raise RuntimeError(f"reconcile {label} plan has no typed lifecycle plan: {plan!r}")
    configuration = _UNIT_OF_WORK.context.configuration
    staged, outcome, child_short, _verified = _lifecycle_reconciliation_service().execute_lifecycle_plan(
        plan,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
        resolve_plan=lambda candidate: _lifecycle_plan_with_resolved_child_uuid(candidate, hook),
        verify_child=lambda parent, short, strict_uuid: _verify_applied_child(
            task_bin, parent, short, hook=hook, strict_uuid=strict_uuid,
        ),
        verified_children=verified_children,
        strict_uuid=strict_uuid,
        label=label,
    )
    if staged is not None and not staged.ok:
        _raise_for_lifecycle_outcome(staged, label=f"{label} staging")
    if outcome is not None:
        _raise_for_lifecycle_outcome(outcome, label=label)
    if not child_short:
        raise RuntimeError(f"lifecycle {label} produced no child identity")
    return child_short


def _terminal_lifecycle_plan(plan: lifecycle.LifecycleRecoveryDecision) -> LifecyclePlan:
    """Create the typed terminal plan for a reconcile final/manual decision."""
    parent = plan.parent
    if plan.action == "manual_stop":
        event = LifecycleEvent.MANUAL_DELETE
    elif str(parent.get("status") or "").strip().lower() == "deleted":
        event = LifecycleEvent.EXPIRE
    elif plan.terminal_kind == "chain_max":
        event = LifecycleEvent.CHAIN_MAX
    elif plan.terminal_kind == "chain_until":
        event = LifecycleEvent.CHAIN_UNTIL
    else:
        event = LifecycleEvent.COMPLETE
    try:
        return terminal_plan_for_snapshot(
            TaskSnapshot.from_mapping(parent),
            event,
            terminal_kind=plan.terminal_kind,
        )
    except Exception as exc:
        raise _LifecycleManualReview(
            f"terminal policy refused mutation: {str(exc).strip() or type(exc).__name__}"
        ) from exc


def _execute_reconcile_terminal_plan(
    task_bin: str,
    hook: Any,
    plan: lifecycle.LifecycleRecoveryDecision,
) -> str:
    """Apply a guarded terminal plan through the shared lifecycle application service."""
    lifecycle_plan = _terminal_lifecycle_plan(plan)
    outcome = _lifecycle_reconciliation_service().apply_terminal_plan(
        plan, terminal_plan_factory=lambda _plan: lifecycle_plan,
    )
    _raise_for_lifecycle_outcome(outcome, label="terminal transition")
    return "off"



def _apply_parent_atomic(
    task_bin: str,
    hook: Any,
    original_parent: dict[str, Any],
    *,
    taskdata: Path,
    lease_held: bool = False,
    verified_children: dict[str, dict[str, Any]] | None = None,
    generation: ChainGenerationService | None = None,
) -> tuple[lifecycle.LifecycleRecoveryDecision, str]:
    def lock_busy(kind: str) -> None:
        _LOCK_STATS[f"{kind}_busy"] += 1

    def validated_configuration(current_hook: Any) -> tuple[str, str]:
        status, reason = _configuration_state(current_hook)
        if status != "valid":
            raise _ConfigurationDrift(reason)
        return status, reason

    operations = CallbackLifecycleApplyOperations(
        configuration_callback=validated_configuration,
        refresh_callback=lambda parent, *, generation: _refresh_plan(
            task_bin, hook, parent, generation=generation,
        ),
        execute_callback=lambda plan, *, verified_children, label, strict_uuid: _execute_reconcile_lifecycle_plan(
            task_bin, hook, plan, verified_children=verified_children,
            label=label, strict_uuid=strict_uuid,
        ),
        terminal_callback=lambda plan: _execute_reconcile_terminal_plan(task_bin, hook, plan),
        lock_callback=lock_busy,
    )
    return _lifecycle_reconciliation_service().apply_parent(
        original_parent,
        operations=operations,
        taskdata=taskdata,
        lease_held=lease_held,
        verified_children=verified_children,
        generation=generation,
        hook=hook,
    )


def _recovery_error(parent: dict[str, Any], reason: str) -> lifecycle.LifecycleRecoveryDecision:
    return lifecycle.LifecycleRecoveryDecision(
        "error",
        parent,
        lifecycle.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _recovery_terminal(parent: dict[str, Any], reason: str) -> lifecycle.LifecycleRecoveryDecision:
    """Classify an expired-but-still-pending child as resumable, not corrupt."""
    if reason.endswith("native until has already elapsed"):
        return _recovery_partial(
            parent,
            f"{reason}; wait for Taskwarrior to mark the child deleted, then rerun reconcile",
        )
    return _recovery_error(parent, reason)


def _recovery_partial(parent: dict[str, Any], reason: str) -> lifecycle.LifecycleRecoveryDecision:
    return lifecycle.LifecycleRecoveryDecision(
        "partial",
        parent,
        lifecycle.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _recovery_manual_review(parent: dict[str, Any], reason: str) -> lifecycle.LifecycleRecoveryDecision:
    return lifecycle.LifecycleRecoveryDecision(
        "manual_review",
        parent,
        lifecycle.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _validate_recovery_child(parent: dict[str, Any], child: dict[str, Any]) -> str:
    _child_short, child_error = lifecycle.resolve_existing_child(
        parent,
        [child],
        include_deleted=True,
    )
    return child_error


def _terminal_recovery_error(child: dict[str, Any], hook: Any, recovery_at: Any) -> str:
    if str(child.get("status") or "").strip().lower() != "pending":
        return ""
    until_raw = child.get("until")
    try:
        until_dt, until_err = _safe_parse_datetime(hook, until_raw)
    except Exception:
        return "live recovery child native until could not be parsed"
    if until_err or until_dt is None:
        return f"live recovery child has no reliable native until: {until_err or 'missing until'}"

    target_field = "due" if child.get("due") else "scheduled"
    target_raw = child.get(target_field)
    try:
        target_dt, target_err = _safe_parse_datetime(hook, target_raw)
    except Exception:
        return f"live recovery child {target_field} could not be parsed"
    if target_err or target_dt is None:
        return f"live recovery child has no reliable {target_field}: {target_err or f'missing {target_field}'}"
    try:
        if compare_datetimes(until_dt, target_dt) <= 0:
            return f"live recovery child native until is not later than its {target_field}"
        if compare_datetimes(until_dt, recovery_at) <= 0:
            return "live recovery child native until has already elapsed"
    except Exception:
        return "live recovery child timing could not be compared"
    return ""


def _next_recovery_child(
    parent: dict[str, Any],
    child_short: str,
) -> dict[str, Any]:
    wanted = str(child_short or "").strip().lower()
    if not wanted:
        raise RuntimeError("recovery action did not identify its child")
    try:
        value = _read_value(
            _repository().by_uuid(wanted, refresh=True),
            f"recovery child {wanted}",
        )
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        raise _RecoveryLookupUnavailable(
            f"recovery child {wanted} lookup unavailable: {reason}"
        ) from exc
    matches = [value] if value is not None else []
    if len(matches) != 1:
        raise RuntimeError(
            f"recovery child {wanted} lookup returned {len(matches)} exact match(es)"
        )
    child = matches[0]
    validation_error = _validate_recovery_child(parent, child)
    if validation_error:
        raise RuntimeError(validation_error)
    return child


def _virtual_expired_child(
    plan: lifecycle.LifecycleRecoveryDecision,
    *,
    hook: Any,
    recovery_at: Any,
) -> tuple[dict[str, Any] | None, str]:
    child = dict(plan.child or {})
    until_raw = child.get("until")
    try:
        until_dt, until_err = _safe_parse_datetime(hook, until_raw)
    except Exception:
        return None, "planned child expiration could not be parsed"
    if until_err or until_dt is None:
        return None, f"planned child has no reliable native until: {until_err or 'missing until'}"
    try:
        if compare_datetimes(until_dt, recovery_at) > 0:
            return None, ""
    except Exception:
        return None, "planned child expiration could not be compared with recovery time"

    child["status"] = "deleted"
    child["end"] = until_raw
    child["uuid"] = (
        f"dryrun-{str(child.get('chainID') or 'chain')}-"
        f"{lifecycle.int_or_default(child.get('link'), plan.next_link)}"
    )
    child.pop("nextLink", None)
    validation_error = _validate_recovery_child(plan.parent, child)
    if validation_error:
        return None, validation_error
    return child, ""


def _reconcile_candidate(
    task_bin: str,
    hook: Any,
    parent: dict[str, Any],
    *,
    taskdata: Path | None,
    apply: bool,
    max_expiration_hops: int,
    recovery_at: Any,
    lease_held: bool = False,
    generation: ChainGenerationService | None = None,
) -> list[tuple[lifecycle.LifecycleRecoveryDecision, str]]:
    def recovery_from_exception(candidate: dict[str, Any], exc: Exception) -> Any:
        reason = str(exc).strip() or type(exc).__name__
        if isinstance(exc, (_ConfigurationDrift, _LifecycleRetryable, _PlanReadUnavailable)):
            return _recovery_partial(candidate, reason)
        if isinstance(exc, _LifecycleManualReview):
            return _recovery_manual_review(candidate, reason)
        return _recovery_error(candidate, reason)

    return _lifecycle_reconciliation_service().recover_candidate(
        parent,
        operations=CallbackLifecycleRecoveryOperations(
            apply_parent_callback=lambda candidate, **kwargs: _apply_parent_atomic(
                task_bin, hook, candidate, **kwargs,
            ),
            plan_parent_callback=lambda candidate, **kwargs: _plan_for_parent(
                task_bin, hook, candidate, **kwargs,
            ),
            next_child_callback=_next_recovery_child,
            virtual_child_callback=lambda candidate, **kwargs: _virtual_expired_child(
                candidate, hook=hook, **kwargs,
            ),
            terminal_error_callback=lambda child, recovery_at: _terminal_recovery_error(
                child, hook, recovery_at,
            ),
            is_orphan_deleted_callback=lifecycle.is_orphan_deleted_chain_candidate,
            recovery_error_callback=_recovery_error,
            recovery_partial_callback=_recovery_partial,
            recovery_manual_review_callback=_recovery_manual_review,
            recovery_terminal_callback=_recovery_terminal,
            recovery_exception_callback=recovery_from_exception,
        ),
        taskdata=taskdata,
        apply=apply,
        max_expiration_hops=max_expiration_hops,
        recovery_at=recovery_at,
        lease_held=lease_held,
        generation=generation,
    )

def _fmt_parent(parent: dict[str, Any]) -> str:
    uuid = lifecycle.short_uuid(parent.get("uuid")) or "????????"
    chain_id = str(parent.get("chainID") or "?")
    link = lifecycle.int_or_default(parent.get("link"), 0)
    desc = str(parent.get("description") or "").strip()
    return f"{uuid} chain {chain_id} link {link}" + (f" · {desc}" if desc else "")


def _print_evidence(evidence: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = evidence.get(key)
        if value in (None, ""):
            continue
        print(f"  {key.replace('_', ' ')}: {value}")


def _describe_plan(plan: lifecycle.LifecycleRecoveryDecision, *, hook: Any, fmt_dt_local=None) -> dict[str, Any]:
    evidence = lifecycle.describe_plan(plan, fmt_dt_local=fmt_dt_local)
    child = plan.child if isinstance(plan.child, dict) else {}
    child_until = child.get("until")
    if not child_until:
        return evidence
    try:
        until_dt, until_err = _safe_parse_datetime(hook, child_until)
    except Exception:
        return evidence
    if until_err or until_dt is None:
        return evidence

    if callable(fmt_dt_local):
        try:
            evidence["child_expires"] = str(fmt_dt_local(until_dt))
        except Exception:
            evidence["child_expires"] = str(child_until)
    else:
        evidence["child_expires"] = str(child_until)

    if plan.child_due is None:
        return evidence
    try:
        core = _runtime_core(hook)
        add_validation = core._import_sibling("add_validation")
        carry = add_validation.describe_native_until_carry(
            until_dt,
            plan.child_due,
            to_local=core.to_local,
        )
    except Exception:
        carry = None
    if carry:
        evidence["expiration"] = carry
    return evidence


def _print_plan(
    plan: lifecycle.LifecycleRecoveryDecision,
    evidence: dict[str, Any] | None = None,
    *,
    applied_short: str = "",
) -> None:
    parent = _fmt_parent(plan.parent)
    if evidence is None:
        evidence = lifecycle.describe_plan(plan)
    if plan.action == "spawn":
        suffix = f" -> created {applied_short}" if applied_short else ""
        print(_style(f"spawn: {parent}{suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("reason", "kind", "next_link", "child_field", "child_target", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "backfill_nextlink":
        suffix = " (applied)" if applied_short else ""
        print(_style(f"backfill nextLink: {parent}{suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("reason", "next_link", "existing_child"))
    elif plan.action == "legitimate_final":
        suffix = " -> set chain:off" if applied_short else ""
        label = "terminal" if lifecycle.is_terminal_plan(plan) else "final"
        print(_style(f"{label}: {parent} ({plan.reason}){suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "manual_stop":
        suffix = " -> set chain:off" if applied_short else ""
        print(_style(f"manual stop: {parent} ({plan.reason}){suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("kind", "next_link"))
    elif plan.action == "stale":
        print(_style(f"skip: {parent} ({plan.reason})", _action_style(plan.action)))
    elif plan.action == "partial":
        print(_style(f"partial: {parent} ({plan.reason})", _action_style(plan.action)))
    else:
        print(_style(f"error: {parent} ({plan.reason})", _action_style("error")))
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))


def _print_recovery_group(
    items: list[tuple[lifecycle.LifecycleRecoveryDecision, dict[str, Any], str]],
) -> None:
    first = items[0][0]
    last, evidence, applied_short = items[-1]
    hops = sum(1 for plan, _evidence, _applied in items if plan.action in {"spawn", "backfill_nextlink"})
    noun = "occurrence" if hops == 1 else "occurrences"
    print(_style(f"recover: {_fmt_parent(first.parent)} -> advanced {hops} {noun}", "cyan"))
    if last.action in {"error", "partial", "legitimate_final", "manual_stop", "stale"}:
        result = "terminal" if lifecycle.is_terminal_plan(last) else last.action.replace("_", " ")
        print(_style(f"  result: {result} ({last.reason})", _action_style(last.action)))
        return
    if applied_short:
        print(f"  child: {applied_short}")
    _print_evidence(evidence, ("next_link", "child_local", "child_due", "child_expires"))


def _startup_failure(args: Any, stage: str, exc: Exception) -> int:
    reason = str(exc).strip() or type(exc).__name__
    if args.json:
        configuration_status = "unavailable" if stage == "taskdata_config" else "valid"
        payload: dict[str, Any] = {
            "schema": _JSON_SCHEMA,
            "schema_version": _JSON_SCHEMA_VERSION,
            "mode": "apply" if args.apply else "dry-run",
            "status": "error",
            "stage": stage,
            "error": reason,
            "configuration_status": configuration_status,
            "configuration_drifted": 0,
            "configuration_drift": reason if configuration_status == "unavailable" else "",
            "candidates": 0,
            "expiration_hops": 0,
            "recovered_chains": 0,
            "spawn": 0,
            "backfill_nextlink": 0,
            "legitimate_final": 0,
            "terminal": 0,
            "manual_stop": 0,
            "stale": 0,
            "partial": 0,
            "native_until_manual_review": 0,
            "native_until_audit_skipped": 0,
            "errors": 1,
            "startup_errors": 1,
            "plan_errors": 0,
            "native_until_error_count": 0,
            "plans": [],
            "applied": [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_style(f"error: {stage.replace('_', ' ')}: {reason}", "red", stream=sys.stderr), file=sys.stderr)
    return 1


def main(
    argv: list[str] | None = None,
    *,
    _apply_lease_held: bool = False,
    _locked_taskdata: Path | None = None,
    _unit_of_work: TaskwarriorUnitOfWork | None = None,
) -> int:
    parser = build_parser(
        expiration_hop_limit=_expiration_hop_limit,
        default_expiration_hops=_DEFAULT_EXPIRATION_HOPS,
        max_expiration_hops=_MAX_EXPIRATION_HOPS,
    )
    args = parser.parse_args(argv)
    _EXPORT_STATS.update(calls=0, rows=0, seconds=0.0, slowest_seconds=0.0, snapshot_hits=0)
    _LOCK_STATS.update(reconcile_busy=0, parent_busy=0)
    if _unit_of_work is None:
        try:
            _unit_of_work = build_operator_uow(
                core=nautical_core_package,
                task_binary=args.task_bin,
                env=os.environ,
                access=IntegrationAccess.MUTATION if args.apply else IntegrationAccess.READ_ONLY,
            )
        except Exception as exc:
            return _startup_failure(args, "integration_context", exc)
    args.task_bin = _unit_of_work.context.command_prefix[0]
    resolved_taskdata = _unit_of_work.context.taskdata
    if args.apply and not _apply_lease_held:
        with _reconcile_apply_lock(resolved_taskdata) as acquired:
            if not acquired:
                _LOCK_STATS["reconcile_busy"] += 1
                return _startup_failure(args, "apply_lock", RuntimeError("another reconcile apply is already running"))
            return main(
                argv,
                _apply_lease_held=True,
                _locked_taskdata=resolved_taskdata,
                _unit_of_work=_unit_of_work,
            )

    # Reconcile is an operator front end over the public core package.  It
    # must not dynamically load or expose a hook-runtime seam.
    hook = nautical_core_package
    try:
        core = _runtime_core(hook)
        fmt_dt_local = getattr(core, "fmt_dt_local", None)
        now_utc = getattr(core, "now_utc", None)
        recovery_at = now_utc() if callable(now_utc) else datetime.now(timezone.utc)
    except Exception as exc:
        return _startup_failure(args, "runtime", exc)
    global _UNIT_OF_WORK
    _UNIT_OF_WORK = _unit_of_work
    repository = _unit_of_work.repository
    repository.configure_commands(timeout=120.0, attempts=2, retry_delay=0.05)
    scope_filter = f"chainID:{args.chain_id}" if args.chain_id else f"uuid:{args.uuid}" if args.uuid else None
    snapshot = _ReconcileSnapshot(
        repository,
        scope_filter=scope_filter,
        full_audit=bool(args.full_audit),
    )
    configuration = _UNIT_OF_WORK.context.configuration
    lifecycle_service = LifecycleReconciliationService(
        snapshot,
        repository,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
        unit_of_work=_UNIT_OF_WORK,
    )
    runtime_state = _ReconcileRuntimeState(repository, snapshot, lifecycle_service)
    _RECONCILE_RUNTIME.set(runtime_state)
    try:
        candidates = lifecycle_service.candidates()
    except Exception as exc:
        return _startup_failure(args, "candidate_export", exc)
    integrity_audit_result: Any = None
    integrity_application_results: tuple[Any, ...] = ()
    integrity_seconds = 0.0
    integrity_application_seconds = 0.0
    try:
        from nautical_core.lifecycle_outbox import LifecycleOutboxRepository
        if snapshot._rows is not None:
            integrity_started = time.perf_counter()
            integrity_engine, integrity_audit_result = _audit_reconcile_integrity(
                tuple(snapshot._rows)
            )
            integrity_seconds = time.perf_counter() - integrity_started
        if integrity_audit_result is not None and integrity_audit_result.status.value == "unavailable":
            configuration_status = "unavailable"
            configuration_drift_reason = integrity_audit_result.reason or "integrity audit unavailable"
        elif integrity_audit_result is not None and args.apply and integrity_audit_result.plans:
            application_started = time.perf_counter()
            integrity_application = integrity_engine.apply(
                integrity_audit_result,
                executor=TaskwarriorMutationService(_UNIT_OF_WORK),
                request_factory=_integrity_request_factory,
                outbox_repository=LifecycleOutboxRepository(_UNIT_OF_WORK.outbox.taskdata),
                owner=f"reconcile-integrity-{os.getpid()}",
            )
            integrity_application_results = integrity_application.applications
            integrity_application_seconds = time.perf_counter() - application_started
            if integrity_application_results:
                snapshot._rows = None
                candidates = lifecycle_service.candidates()
    except Exception as exc:
        integrity_audit_result = None
        if args.apply:
            configuration_status = "unavailable"
            configuration_drift_reason = f"integrity audit unavailable: {type(exc).__name__}: {exc}"
        elif os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] integrity audit skipped: {type(exc).__name__}: {exc}", file=sys.stderr)
    taskdata = _locked_taskdata if args.apply else None
    runtime_taskdata = resolved_taskdata
    try:
        generation = _chain_generation_for_hook(hook)
    except Exception as exc:
        return _startup_failure(args, "chain_generation", exc)
    configuration_status, configuration_drift_reason = _configuration_state(hook)
    integrity_drain_results: tuple[Any, ...] = ()
    if args.apply and configuration_status == "valid":
        try:
            integrity_drain_results = _drain_integrity_work()
        except Exception as exc:
            configuration_status = "unavailable"
            configuration_drift_reason = f"integrity drain unavailable: {type(exc).__name__}: {exc}"
            if not args.json:
                print(_style(f"warning: integrity drain deferred: {configuration_drift_reason}", "yellow"))
        for item in integrity_drain_results:
            if not args.json and item.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}:
                print(_style(f"error: integrity: {item.reason or item.kind.value}", "red", stream=sys.stderr), file=sys.stderr)
    native_until_audit_warning = ""
    native_until_audit_status = "unavailable" if configuration_status != "valid" else "valid"
    if configuration_status != "valid":
        native_until_repairs: list[dict[str, Any]] = []
        native_until_errors: list[str] = []
    else:
        try:
            native_until_repairs, native_until_errors = _native_until_repairs(
                args.task_bin,
                hook,
                apply=args.apply,
                taskdata=taskdata,
                lease_held=_apply_lease_held,
            )
            native_until_audit_status = _native_until_audit_result(
                native_until_repairs, native_until_errors
            ).status
        except Exception as exc:
            # Dry-run can still report planned work, but apply must not mutate
            # after the authoritative integrity read became unavailable.
            native_until_repairs, native_until_errors = [], []
            native_until_audit_warning = str(exc).strip() or type(exc).__name__
            native_until_audit_status = "unavailable"
            if args.apply:
                configuration_status = "unavailable"
                configuration_drift_reason = (
                    f"native-until audit unavailable: {native_until_audit_warning}; restart and rerun"
                )
            if not args.json:
                print(
                    _style(
                        f"warning: native-until audit skipped: {native_until_audit_warning}",
                        "yellow",
                        stream=sys.stderr,
                    ),
                    file=sys.stderr,
                )
    if configuration_status == "valid":
        blocked_item = next(
            (item for item in native_until_repairs if item.get("configuration_drift")),
            None,
        )
        if blocked_item is not None:
            configuration_drift_reason = str(blocked_item.get("repair_error") or "")
            configuration_status = str(blocked_item.get("configuration_status") or "drifted")
    if not args.json:
        for item in native_until_repairs:
            action = item.get("action") or "native_until"
            suffix = f" -> {_format_local_until(hook, item['new_until'])}" if item.get("new_until") else ""
            outcome = " (no change applied)" if action == "manual_review" else ""
            line = (
                f"native-until: {action:<13} {item.get('task') or '?'} "
                f"chain={item.get('chainID') or '?'} link={item.get('link') or '?'}"
                f"  {item.get('reason') or 'invalid native until'}{suffix}{outcome}"
            )
            print(_style(line, _action_style(action)))
        for error in native_until_errors:
            print(_style(f"error: native-until: {error}", "red", stream=sys.stderr), file=sys.stderr)
    plans: list[lifecycle.LifecycleRecoveryDecision] = []
    plan_evidence: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    outcome_groups: list[list[tuple[lifecycle.LifecycleRecoveryDecision, str]]] = []
    processed_slots: set[tuple[str, int]] = set()
    ambiguous_slots = IntegrityRecoveryService.ambiguous_candidate_slots(candidates)

    for parent in candidates:
        if configuration_status == "valid":
            configuration_status, configuration_drift_reason = _configuration_state(hook)
        if configuration_status != "valid":
            break
        parent_slot = (
            str(parent.get("chainID") or "").strip(),
            lifecycle.int_or_default(parent.get("link"), 0),
        )
        if parent_slot in processed_slots:
            continue
        if parent_slot in ambiguous_slots:
            outcomes = [(_recovery_error(parent, ambiguous_slots[parent_slot]), "")]
        else:
            try:
                outcomes = _reconcile_candidate(
                    args.task_bin,
                    hook,
                    parent,
                    taskdata=taskdata,
                    apply=args.apply,
                    max_expiration_hops=args.max_expiration_hops,
                    recovery_at=recovery_at,
                    lease_held=_apply_lease_held,
                    generation=generation,
                )
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                outcomes = [(_recovery_error(parent, reason), "")]
        outcome_groups.append(outcomes)
        if configuration_status == "valid":
            configuration_drift_reason = next(
                (
                    plan.reason
                    for plan, _applied in outcomes
                    if plan.action == "partial" and str(plan.reason).startswith("configuration ")
                ),
                "",
            )
            if configuration_drift_reason:
                configuration_status = (
                    "unavailable"
                    if (
                        "configuration verification unavailable" in configuration_drift_reason
                        or "configuration verifier is unavailable" in configuration_drift_reason
                    )
                    else "drifted"
                )
        rendered: list[tuple[lifecycle.LifecycleRecoveryDecision, dict[str, Any], str]] = []
        for plan, applied_short in outcomes:
            processed_slots.add(
                (
                    str(plan.parent.get("chainID") or "").strip(),
                    lifecycle.int_or_default(plan.parent.get("link"), 0),
                )
            )
            plans.append(plan)
            evidence = _describe_plan(plan, hook=hook, fmt_dt_local=fmt_dt_local)
            plan_evidence.append(evidence)
            rendered.append((plan, evidence, applied_short))
            if args.apply and applied_short:
                disabling = plan.action in {"legitimate_final", "manual_stop"}
                action = "disable_chain" if disabling else plan.action
                record = {
                    "action": action,
                    "parent": lifecycle.short_uuid(plan.parent.get("uuid")),
                }
                if not disabling:
                    record["child"] = applied_short
                applied.append(record)
        if not args.json:
            if args.verbose or len(rendered) <= 1:
                for plan, evidence, applied_short in rendered:
                    _print_plan(plan, evidence, applied_short=applied_short)
            else:
                _print_recovery_group(rendered)

    expiration_hops = sum(
        1
        for plan in plans
        if str(plan.parent.get("status") or "").strip() == "deleted"
        and plan.action in {"spawn", "backfill_nextlink"}
    )
    recovered_chains = sum(
        1
        for outcomes in outcome_groups
        if sum(
            1
            for plan, _applied in outcomes
            if str(plan.parent.get("status") or "").strip() == "deleted"
            and plan.action in {"spawn", "backfill_nextlink"}
        )
        > 1
        and all(plan.action not in {"error", "partial"} for plan, _applied in outcomes)
    )
    native_until_manual_review = sum(
        1 for item in native_until_repairs if item.get("action") == "manual_review"
    )
    native_until_audit_skipped = int(bool(native_until_audit_warning))
    degraded = (
        any(plan.action == "partial" for plan in plans)
        or any(plan.action == "manual_review" for plan in plans)
        or native_until_manual_review > 0
        or native_until_audit_skipped > 0
        or bool(configuration_drift_reason)
        or any(item.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED} for item in integrity_drain_results)
    )
    plan_errors = sum(1 for plan in plans if plan.action == "error")
    native_until_error_count = len(native_until_errors)
    total_errors = plan_errors + native_until_error_count
    has_errors = total_errors > 0

    if not args.apply:
        housekeeping = {"status": "skipped", "reason": "dry_run"}
    elif args.no_housekeeping:
        housekeeping = {"status": "skipped", "reason": "disabled"}
    elif has_errors:
        housekeeping = {"status": "skipped", "reason": "reconcile_errors"}
    else:
        try:
            housekeeping = _opportunistic_housekeeping(runtime_taskdata)
        except Exception as exc:
            housekeeping = {
                "status": "deferred",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if housekeeping.get("status") == "deferred":
            degraded = True
        if not args.json and housekeeping.get("status") == "deferred":
            print(_style(f"warning: lifecycle housekeeping deferred: {housekeeping.get('reason')}", "yellow"))

    read_metrics = _repository().metrics()
    _EXPORT_STATS.update(
        calls=int(read_metrics["calls"]),
        rows=int(read_metrics["rows"]),
        seconds=float(read_metrics["seconds"]),
        slowest_seconds=float(read_metrics["slowest_seconds"]),
    )
    summary = {
        "schema": _JSON_SCHEMA,
        "schema_version": _JSON_SCHEMA_VERSION,
        "status": "error" if has_errors else "degraded" if degraded else "ok",
        "configuration_status": configuration_status,
        "configuration_drifted": int(configuration_status == "drifted"),
        "configuration_drift": configuration_drift_reason,
        "mode": "apply" if args.apply else "dry-run",
        "audit_mode": "full" if args.full_audit else "bounded",
        "scope": {"chainID": args.chain_id, "uuid": args.uuid},
        "candidates": len(candidates),
        "expiration_hops": expiration_hops,
        "recovered_chains": recovered_chains,
        "spawn": sum(1 for p in plans if p.action == "spawn"),
        "backfill_nextlink": sum(1 for p in plans if p.action == "backfill_nextlink"),
        "legitimate_final": sum(1 for p in plans if p.action == "legitimate_final"),
        "terminal": sum(1 for p in plans if lifecycle.is_terminal_plan(p)),
        "manual_stop": sum(1 for p in plans if p.action == "manual_stop"),
        "stale": sum(1 for p in plans if p.action == "stale"),
        "partial": sum(1 for p in plans if p.action == "partial"),
        "manual_review": sum(1 for p in plans if p.action == "manual_review"),
        "errors": total_errors,
        "startup_errors": 0,
        "plan_errors": plan_errors,
        "native_until_error_count": native_until_error_count,
        "native_until_manual_review": native_until_manual_review,
        "native_until_audit_skipped": native_until_audit_skipped,
        "native_until_audit_status": native_until_audit_status,
        "integrity_drain": [
            {
                "plan_id": item.plan_id,
                "operation_id": item.operation_id,
                "status": item.kind.value,
                "reason": item.reason,
            }
            for item in integrity_drain_results
        ],
        "integrity_audit": None if integrity_audit_result is None else integrity_components(integrity_audit_result),
        "integrity_seconds": round(integrity_seconds, 6),
        "integrity_application_seconds": round(integrity_application_seconds, 6),
        "export_calls": _EXPORT_STATS["calls"],
        "export_rows": _EXPORT_STATS["rows"],
        "export_seconds": round(_EXPORT_STATS["seconds"], 4),
        "slowest_export_seconds": round(_EXPORT_STATS["slowest_seconds"], 4),
        "snapshot_hits": _EXPORT_STATS["snapshot_hits"],
        "lock_contention": dict(_LOCK_STATS),
        "plans": [
            {
                "action": plan.action,
                **evidence,
            }
            for plan, evidence in zip(plans, plan_evidence)
        ],
        "applied": applied,
        "native_until_repairs": native_until_repairs,
        "native_until_errors": native_until_errors,
        "native_until_audit_warning": native_until_audit_warning,
        "housekeeping": housekeeping,
    }
    if args.json:
        print(render_json(summary))
    else:
        summary_line, diagnostics_line = render_human(summary, _style)
        print(summary_line)
        print(diagnostics_line)
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
