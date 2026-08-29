#!/usr/bin/env python3
"""Repair Nautical chains missing a successor after completion or expiration."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager, nullcontext
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
import sys
import time
import uuid
from pathlib import Path
from types import ModuleType
from typing import Any, cast


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
from nautical_core.chain_integrity_recovery import IntegrityRecoveryService  # noqa: E402
from nautical_core.integration_context import IntegrationAccess  # noqa: E402
from nautical_core.operator_control_plane import OperatorControlPlane  # noqa: E402
from nautical_core.operator_application import DomainApplicationRegistry  # noqa: E402
from nautical_core.operator_context import OperatorInvocationBudget  # noqa: E402
from nautical_core.operator_models import OperatorLimits  # noqa: E402
from nautical_core.lifecycle_models import (  # noqa: E402
    DeletionDisposition,
    LifecycleAction,
    LifecyclePlan,
    VirtualExpiredChild,
    recurrence_fingerprint,
)
from nautical_core.lifecycle_recovery_models import RecoveryPlanResult, RecoveryRefusal, RecoveryResult, RecoveryStatus  # noqa: E402
from nautical_core.integration_models import (  # noqa: E402
    Absent,
    Found,
    MutationOutcomeKind,
    Unavailable,
)
from nautical_core.task_read_repository import TaskReadRepository  # noqa: E402
from nautical_core.task_models import FieldPresence, NauticalTask, TaskDraft, TaskObservation, TaskPayload  # noqa: E402
from nautical_core.task_codec import DEFAULT_TASK_CODEC  # noqa: E402
from nautical_core.timeutil import compare_datetimes  # noqa: E402
from nautical_core.taskwarrior_uow import (  # noqa: E402
    TaskwarriorUnitOfWork,
    build_operator_uow,
)
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService  # noqa: E402
from nautical_core.reconcile_cli import ReconcileRequest, build_parser  # noqa: E402
from nautical_core.reconcile_report import exit_code, render_human, to_operator_result  # noqa: E402
from nautical_core.operator_presentation import key_value_lines, render_json_document, render_result  # noqa: E402
from nautical_core.integrity_report import components as integrity_components  # noqa: E402
from nautical_core.lifecycle_reconciliation import (  # noqa: E402
    CallbackLifecycleApplyOperations,
    CallbackLifecycleRecoveryOperations,
    LifecycleReconciliationService,
)
from nautical_core.reconcile_snapshot_service import ReconcileSnapshotService  # noqa: E402


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


from nautical_core.native_until_integrity import NativeUntilAudit, audit_result


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


def _stable_child_uuid(hook: Any, parent: TaskPayload, child: TaskPayload) -> str:
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


def _read_value(
    read: Any,
    subject: str,
) -> TaskObservation | tuple[TaskObservation, ...] | None:
    if isinstance(read, Found):
        value = read.value
        if isinstance(value, TaskObservation):
            return value
        if isinstance(value, tuple) and all(isinstance(row, TaskObservation) for row in value):
            return value
        raise _PlanReadUnavailable(f"{subject} returned an untyped task result")
    if isinstance(read, Absent):
        return None
    if isinstance(read, Unavailable):
        raise _PlanReadUnavailable(f"{subject} unavailable: {read.evidence.detail}")
    raise _PlanReadUnavailable(f"{subject} returned an invalid typed result")


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
    """Return the validated configuration state and actionable reason."""
    check = _configuration_verification(hook)
    return check.status, check.reason


def _observation_text(observation: TaskObservation, field: str) -> str:
    state = observation.field(field)
    if state.presence is FieldPresence.ABSENT:
        return ""
    value = state.raw_value()
    return str(value or "").strip()


class _ReconcileRuntimeState:
    """Invocation-scoped read/service state; never shared between runs."""

    __slots__ = ("repository", "snapshot", "lifecycle_service")

    def __init__(
        self,
        repository: TaskReadRepository,
        snapshot: ReconcileSnapshotService,
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


def _active_chain_rows(
    task_bin: str,
    *,
    include_inactive: bool = False,
    snapshot: ReconcileSnapshotService | None = None,
) -> list[TaskObservation]:
    """Export live Nautical links for integrity checks, independently of recovery candidates."""
    if snapshot is None:
        raise RuntimeError("active-chain reads require an authoritative snapshot")
    rows = snapshot.active_rows()
    return sorted(
        (
            row
            for row in rows
            if _observation_text(row, "status").lower() not in {"completed", "deleted"}
        ),
        key=IntegrityRecoveryService.candidate_sort_key,
    )


def _native_until_guard_error(expected: TaskObservation, fresh: TaskObservation) -> str | None:
    """Detect target or recurrence changes made after the audit export."""
    fields = (
        "uuid", "status", "chain", "chainID", "link", "due", "scheduled", "until",
        "anchor", "anchor_file", "cp", "chainMax", "chainUntil",
    )
    for field in fields:
        left = _observation_text(expected, field)
        right = _observation_text(fresh, field)
        if field == "link":
            left_link = lifecycle.int_or_default(left, 0)
            right_link = lifecycle.int_or_default(right, 0)
            if left_link != right_link:
                return f"native-until target changed ({field}: {left_link} -> {right_link})"
            continue
        else:
            left = str(left or "").strip()
            right = str(right or "").strip()
        if left != right:
            return f"native-until target changed ({field}: {left or '<empty>'} -> {right or '<empty>'})"
    return None


def _fresh_native_until_previous(row: TaskObservation) -> TaskObservation | None:
    chain_id = _observation_text(row, "chainID")
    link = lifecycle.int_or_default(getattr(row.field("link").value, "value", row.field("link").value), 0)
    if not chain_id or link <= 1:
        return None
    value = _read_value(
        _repository().predecessor_slot(chain_id, link - 1, refresh=True),
        f"predecessor {chain_id}:{link - 1}",
    )
    return value if isinstance(value, TaskObservation) else None


def _fresh_native_until_parent(row: TaskObservation) -> TaskObservation | None:
    uuid_value = _observation_text(row, "uuid")
    if not uuid_value:
        raise RuntimeError("native-until target has no UUID")
    return cast(TaskObservation | None, _read_value(
        _repository().verification(uuid_value),
        f"native-until parent {uuid_value}",
    ))


def _native_until_repairs(
    task_bin: str,
    hook: Any,
    *,
    apply: bool,
    taskdata: Path | None = None,
    snapshot: ReconcileSnapshotService | None = None,
    lease_held: bool = False,
    control_plane: OperatorControlPlane,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find invalid native windows and repair only those with a reliable predecessor."""
    runtime_state = _reconcile_runtime_state()
    runtime_snapshot = runtime_state.snapshot if runtime_state is not None else None
    active_rows = _active_chain_rows(
        task_bin,
        include_inactive=False,
        snapshot=snapshot or runtime_snapshot,
    )
    rows = active_rows
    recovery_audit = control_plane.audit_native_until(
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
        chain_id = _observation_text(row, "chainID")
        link = lifecycle.int_or_default(
            getattr(row.field("link").value, "value", row.field("link").value),
            0,
        )
        item = repair_items.get((chain_id, link))
        if item is None:
            continue
        repaired = str(item.get("new_until") or "").strip()
        if apply:
            if taskdata is None:
                raise RuntimeError("native-until repair requires Taskwarrior data location")
            error = control_plane.apply_native_until(
                row,
                previous,
                item,
                repaired=repaired,
                taskdata=taskdata,
                lease_held=lease_held,
                mutation_lock=lambda data, held: _reconcile_mutation_lock(data, lease_held=held),
                # The recovery service supplies the parent UUID to this
                # callback; bind the invocation's Taskdata once here.
                parent_lock=lambda parent_uuid: _parent_apply_lock(taskdata, parent_uuid),
                refresh_parent=_fresh_native_until_parent,
                refresh_previous=_fresh_native_until_previous,
                guard_error=lambda expected, fresh, fresh_previous: (
                    _native_until_guard_error(expected, fresh)
                    if fresh is not None
                    else "native-until target disappeared"
                ) or (
                    "native-until predecessor changed during repair"
                    if (previous is None) != (fresh_previous is None)
                    else _native_until_guard_error(previous, fresh_previous)
                    if previous is not None and fresh_previous is not None
                    else None
                ),
                configuration=lambda: _configuration_state(hook),
                mutate=lambda fresh, target: _modify_native_until(task_bin, fresh, target),
                verify=lambda verified, target: verified is not None and _native_until_matches(verified, target, hook),
                on_lock_busy=lambda kind: _LOCK_STATS.__setitem__(f"{kind}_busy", _LOCK_STATS[f"{kind}_busy"] + 1),
            )
            if error:
                errors.append(f"{item['task']} chain {chain_id} link {link}: {error}")
    return repairs, errors


def _modify_native_until(task_bin: str, row: TaskObservation, new_until: str) -> None:
    del task_bin
    if _UNIT_OF_WORK is None:
        raise RuntimeError("native until repair requires an integration unit of work")
    request = IntegrityRecoveryService.native_until_request(
        row,
        new_until,
        mutation_epoch=_UNIT_OF_WORK.mutation_epoch,
    )
    outcome = TaskwarriorMutationService(_UNIT_OF_WORK).apply(request)
    if outcome.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}:
        raise RuntimeError(outcome.reason or outcome.kind.value)


def _native_until_matches(fresh: TaskObservation, expected: str, hook: Any) -> bool:
    """Compare native-until timestamps by instant, tolerating Taskwarrior formatting."""
    actual = _observation_text(fresh, "until")
    if actual == str(expected or "").strip():
        return True
    try:
        actual_dt, actual_err = _safe_parse_datetime(hook, actual)
        expected_dt, expected_err = _safe_parse_datetime(hook, expected)
        return not actual_err and not expected_err and actual_dt is not None and actual_dt == expected_dt
    except Exception:
        return False


def _recovery_existing_children(parent: TaskPayload) -> tuple[TaskObservation, ...]:
    """Read the successor slot as immutable observations."""
    parent_observation = (
        parent
        if isinstance(parent, TaskObservation)
        else DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile child lookup")
    )

    def child_observation(chain_id: str, link: int) -> TaskObservation | None:
        read = _repository().exact_child_slot(chain_id, link, refresh=True)
        if isinstance(read, Found):
            if not isinstance(read.value, TaskObservation):
                raise _PlanReadUnavailable(f"child slot {chain_id}:{link} returned an invalid observation")
            return read.value
        if isinstance(read, Absent):
            return None
        if isinstance(read, Unavailable):
            raise _PlanReadUnavailable(f"child slot {chain_id}:{link} unavailable: {read.evidence.detail}")
        raise _PlanReadUnavailable(f"child slot {chain_id}:{link} returned an invalid typed result")

    return IntegrityRecoveryService(
        child_lookup=child_observation,
    ).existing_children(parent_observation)


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


def _fresh_parent(parent: TaskPayload) -> TaskPayload | None:
    parent_uuid = str(parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    value = _read_value(
        _repository().verification(parent_uuid),
        f"parent {parent_uuid}",
    )
    return value.to_mapping() if isinstance(value, TaskObservation) else None


def _parent_identity_error(parent: TaskPayload) -> str:
    """Explain why a parent cannot be used as an atomic reconcile target."""
    chain_id = str(parent.get("chainID") or "").strip()
    if not chain_id:
        return "parent chainID is missing"

    raw_link = parent.get("link")
    if raw_link is None or not str(raw_link).strip():
        return "parent link is missing; post-v2 reconcile requires a stamped link; inspect with nautical query integrity --all"
    if isinstance(raw_link, bool):
        return f"parent link is invalid: {raw_link!r}"
    try:
        parsed_link = int(raw_link)
    except (TypeError, ValueError, OverflowError):
        return f"parent link is invalid: {raw_link!r}"
    if parsed_link <= 0:
        return f"parent link must be positive; got {parsed_link}"
    return ""


def _parent_guard_filters(parent: TaskPayload) -> list[str]:
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


def _verify_disabled_parent(task_bin: str, parent: TaskPayload) -> None:
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


def _stale_plan(parent: TaskPayload, reason: str) -> RecoveryRefusal:
    return RecoveryRefusal(
        DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile stale plan"),
        RecoveryStatus.STALE,
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
    hook: Any,
    original_parent: TaskPayload,
    *,
    generation: ChainGenerationService | None = None,
    reconciliation_service: LifecycleReconciliationService,
) -> RecoveryResult:
    parent = _fresh_parent(original_parent)
    if parent is None:
        return _stale_plan(original_parent, "parent no longer exists")
    status = str(parent.get("status") or "").strip().lower()
    if status == "completed":
        candidate = lifecycle.is_orphan_completion_candidate(
            DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile refresh")
        )
    elif status == "deleted":
        candidate = lifecycle.is_orphan_deleted_chain_candidate(
            DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile refresh")
        )
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
        hook,
        parent,
        generation=generation or _chain_generation_for_hook(hook),
        reconciliation_service=reconciliation_service,
    )


def _plan_for_parent(
    hook: Any,
    parent: TaskPayload,
    *,
    generation: ChainGenerationService | None = None,
    reconciliation_service: LifecycleReconciliationService,
) -> RecoveryResult:
    """Build the one reconcile plan used by both preview and apply paths."""
    configuration_status, configuration_reason = _configuration_state(hook)
    if configuration_status != "valid":
        raise _ConfigurationDrift(configuration_reason)
    try:
        from nautical_core.task_codec import DEFAULT_TASK_CODEC
        observation = DEFAULT_TASK_CODEC.decode_row(
            parent,
            source_query="reconcile lifecycle planning",
        )
        return reconciliation_service.plan(
            observation,
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
        MutationGuard,
        MutationRequest,
    )
    from nautical_core.task_changes import TaskPatch
    from nautical_core.task_models import FieldPresence, TaskObservation, TaskUUID

    read = _repository().by_uuid(operation.target_uuid, refresh=True)
    row = _read_value(read, f"integrity target {operation.target_uuid}")
    if not isinstance(row, TaskObservation):
        raise RuntimeError(f"integrity target {operation.target_uuid} is unavailable")
    def field_value(name: str) -> object:
        state = row.field(name)
        return None if state.presence is FieldPresence.ABSENT else state.raw_value()

    modified = str(field_value("modified") or "").strip()
    if not modified:
        raise RuntimeError("integrity target has no modified timestamp")
    link = lifecycle.int_or_default(field_value("link"), 0)
    if link < 0:
        raise RuntimeError("integrity target has an invalid link")
    updates = dict(operation.payload)
    expected = {key: field_value(key) for key in updates}
    guard = MutationGuard(
        task_uuid=str(field_value("uuid") or operation.target_uuid),
        status=str(field_value("status") or "pending"),
        chain_id=str(field_value("chainID") or operation.chain_id),
        link=link,
        recurrence_identity=recurrence_fingerprint(row.to_mapping()),
        timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=_UNIT_OF_WORK.mutation_epoch,
        chain=str(field_value("chain") or "on"),
    )
    patch = TaskPatch.metadata_repair(TaskUUID(guard.task_uuid), **updates)
    return MutationRequest.metadata_repair(guard, patch, expected=expected)


def _audit_reconcile_integrity(rows: tuple[dict[str, Any], ...], *, outbox_repository: Any = None) -> Any:
    """Audit the authoritative lifecycle export without issuing another export."""
    if _UNIT_OF_WORK is None:
        raise RuntimeError("integrity audit requires an integration unit of work")
    from nautical_core.chain_integrity_models import SnapshotCoverage
    from nautical_core.integrity_audit_service import audit_authoritative_rows_with_engine

    unit_of_work = _UNIT_OF_WORK
    if unit_of_work is None:
        raise RuntimeError("integrity audit requires an integration unit of work")
    decoded = tuple(
        DEFAULT_TASK_CODEC.decode_row(row, source_query="reconcile integrity snapshot")
        for row in rows
    )
    bundle = audit_authoritative_rows_with_engine(
        unit_of_work,
        decoded,
        source="lifecycle.lifecycle_candidates",
        coverage=SnapshotCoverage.CHAIN,
        outbox_repository=outbox_repository,
    )
    if bundle is None:
        raise RuntimeError("reconcile lifecycle snapshot rejected: configuration unavailable")
    return bundle.engine, bundle.result


def _find_positional_child(lifecycle_plan: LifecyclePlan) -> Any | None:
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
        def value(name: str) -> object:
            state = row.field(name)
            return state.raw_value()
        row_uuid = str(value("uuid") or "").strip().lower()
        if child_uuid and row_uuid == child_uuid:
            return row
        if (
            str(value("chainID") or "").strip() == lifecycle_plan.identity.chain_id
            and lifecycle.int_or_default(value("link"), 0) == lifecycle_plan.identity.target_link
            and str(value("prevLink") or "").strip().lower() == parent_short
        ):
            return row
    return None


def _resolve_lifecycle_plan_child_uuid(
    lifecycle_plan: LifecyclePlan,
    parent: Any,
    hook: Any,
    *,
    child_observation: Any | None = None,
) -> LifecyclePlan:
    """Resolve the child identity on a typed spawn plan.

    Reconcile may re-run against the same broken chain more than once, and
    may run against a chain a human already partially repaired by hand.
    Prefer a task already occupying this exact chain position (found by uuid
    or by chainID+link+prevLink), so a partially-repaired chain doesn't get
    a duplicate child. Otherwise fall back to the deterministic stable UUID,
    so repeated runs against a still-broken chain converge on the same child
    rather than reserving a fresh random one every time.
    """
    if lifecycle_plan.action is not LifecycleAction.SPAWN_CHILD:
        return lifecycle_plan
    child = lifecycle_plan.child_dict()
    existing = child_observation or _find_positional_child(lifecycle_plan)
    resolved_uuid = (
        str(getattr(existing.field("uuid").value, "value", existing.field("uuid").value) or "").strip()
        if existing is not None else ""
    )
    if not resolved_uuid:
        resolved_uuid = _stable_child_uuid(hook, parent.to_mapping(), child)
    if not resolved_uuid or resolved_uuid == str(child.get("uuid") or "").strip():
        return lifecycle_plan
    child["uuid"] = resolved_uuid
    patch = dict(lifecycle_plan.parent_patch_dict())
    patch["nextLink"] = resolved_uuid[:8]
    resolved_task = NauticalTask.from_observation(
        DEFAULT_TASK_CODEC.decode_row(child, source_query="reconcile resolved child")
    )
    resolved_plan = LifecyclePlan.from_draft(
        identity=lifecycle_plan.identity,
        action=lifecycle_plan.action,
        parent_guard=lifecycle_plan.parent_guard,
        draft=TaskDraft.from_task(resolved_task),
        parent_patch=patch,
        expected_postconditions=lifecycle_plan.expected_postconditions,
        max_attempts=lifecycle_plan.max_attempts,
    )
    return resolved_plan

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
    hook: Any,
    plan: LifecyclePlan,
    *,
    parent: TaskObservation,
    child_observation: TaskObservation | None = None,
    verified_children: dict[str, dict[str, Any]] | None,
    label: str,
    strict_uuid: bool,
    reconciliation_service: LifecycleReconciliationService,
) -> str:
    """Stage and execute one reconcile spawn/backfill through the shared
    lifecycle application service -- the same staging and execution path the
    live hooks use, so a chain reconcile repairs converges identically to
    whatever the on-exit hook would have produced from the same state.
    """
    unit_of_work = _UNIT_OF_WORK
    if unit_of_work is None:
        raise RuntimeError("reconcile lifecycle execution requires an integration unit of work")
    configuration = unit_of_work.context.configuration
    staged, outcome, child_short, _verified = reconciliation_service.execute_lifecycle_plan(
        plan,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
        resolve_plan=lambda candidate: _resolve_lifecycle_plan_child_uuid(
            candidate,
            parent,
            hook,
            child_observation=child_observation,
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


def _execute_reconcile_lifecycle_wave(
    hook: Any,
    lifecycle_service: LifecycleReconciliationService,
    application: Any,
    taskdata: Path,
    planned: dict[str, tuple[RecoveryPlanResult, str]],
    *,
    configuration_fingerprint: str,
    schedule_fingerprint: str,
) -> dict[str, tuple[RecoveryResult, str]] | None:
    """Apply independent one-hop completion plans in one guarded wave.

    The wave owns only completed parents whose first plan is a spawn. Deleted
    parents and multi-hop expiration recovery remain on the sequential path:
    their next decision depends on the child created by the preceding step.
    Returning ``None`` means the lock or wave infrastructure was unavailable;
    callers must then use the normal per-parent path.
    """
    if not planned:
        return {}
    from nautical_core.lifecycle_application import LifecycleApplicationOutcomeKind

    ordered = tuple(sorted(planned.values(), key=lambda item: str(item[0].parent.field("uuid").raw_value())))
    with ExitStack() as locks:
        for result, _ in ordered:
            parent_uuid = str(result.parent.field("uuid").raw_value() or "").strip()
            if not parent_uuid:
                return None
            acquired = locks.enter_context(lifecycle_service.parent_lock(taskdata, parent_uuid))
            if not acquired:
                return None
        plans: list[LifecyclePlan] = []
        by_parent: dict[str, tuple[RecoveryPlanResult, LifecyclePlan]] = {}
        for result, _ in ordered:
            if not isinstance(result, RecoveryPlanResult):
                return None
            lifecycle_plan = result.plan
            child_uuid = str(lifecycle_plan.child_dict().get("uuid") or "").strip()
            resolved_plan = (
                lifecycle_plan
                if child_uuid
                else _resolve_lifecycle_plan_child_uuid(
                    lifecycle_plan,
                    result.parent,
                    hook,
                    child_observation=result.child_observation,
                )
            )
            parent_uuid = str(resolved_plan.identity.parent_uuid).strip().lower()
            plans.append(resolved_plan)
            by_parent[parent_uuid] = (result, resolved_plan)
        result = application.execute_wave(
            tuple(plans),
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )
    outcomes: dict[str, tuple[RecoveryResult, str]] = {}
    result_by_parent = {
        str(outcome.identity.parent_uuid).strip().lower(): outcome
        for outcome in result.outcomes
    }
    for parent_uuid, (result, plan) in by_parent.items():
        outcome = result_by_parent.get(parent_uuid)
        if outcome is None:
            return None
        if outcome.kind in {
            LifecycleApplicationOutcomeKind.APPLIED,
            LifecycleApplicationOutcomeKind.ALREADY_APPLIED,
        }:
            child_short = str(plan.parent_patch_dict().get("nextLink") or result.child_short or "").strip()
            if not child_short:
                return None
            outcomes[parent_uuid] = (
                RecoveryPlanResult(
                    result.parent,
                    plan,
                    reason=result.reason,
                    child_short=child_short,
                    child_due=result.child_due,
                    child_observation=result.child_observation,
                    terminal_kind=result.terminal_kind,
                    applied=True,
                ),
                child_short,
            )
            continue
        parent = result.parent.to_mapping()
        reason = f"wave lifecycle apply {outcome.kind.value}: {outcome.reason or 'mutation was not applied'}"
        if outcome.kind is LifecycleApplicationOutcomeKind.RETRYABLE:
            replacement = _recovery_partial(parent, reason)
        else:
            replacement = _recovery_manual_review(parent, reason)
        outcomes[parent_uuid] = (replacement, "")
    return outcomes


def _execute_reconcile_terminal_plan(
    hook: Any,
    plan: LifecyclePlan,
    *,
    reconciliation_service: LifecycleReconciliationService,
) -> str:
    """Apply a guarded terminal plan through the shared lifecycle application service."""
    outcome = reconciliation_service.apply_terminal_plan(
        plan,
    )
    _raise_for_lifecycle_outcome(outcome, label="terminal transition")
    return "off"



def _apply_parent_atomic(
    hook: Any,
    original_parent: TaskPayload,
    *,
    taskdata: Path,
    lease_held: bool = False,
    verified_children: dict[str, dict[str, Any]] | None = None,
    generation: ChainGenerationService | None = None,
    reconciliation_service: LifecycleReconciliationService,
) -> tuple[RecoveryResult, str]:
    def lock_busy(kind: str) -> None:
        _LOCK_STATS[f"{kind}_busy"] += 1

    def validated_configuration(current_hook: Any) -> tuple[str, str]:
        status, reason = _configuration_state(current_hook)
        if status != "valid":
            raise _ConfigurationDrift(reason)
        return status, reason

    def execute_plan(
        plan: LifecyclePlan,
        *,
        parent: TaskObservation,
        child_observation: TaskObservation | None,
        verified_children: dict[str, dict[str, Any]] | None,
        label: str,
        strict_uuid: bool,
    ) -> str:
        return _execute_reconcile_lifecycle_plan(
            hook,
            plan,
            parent=parent,
            child_observation=child_observation,
            verified_children=verified_children,
            label=label,
            strict_uuid=strict_uuid,
            reconciliation_service=reconciliation_service,
        )

    operations = CallbackLifecycleApplyOperations(
        configuration_callback=validated_configuration,
        refresh_callback=lambda parent, *, generation: _refresh_plan(
            hook, parent, generation=generation,
            reconciliation_service=reconciliation_service,
        ),
        execute_callback=execute_plan,
        terminal_callback=lambda plan: _execute_reconcile_terminal_plan(
            hook, plan, reconciliation_service=reconciliation_service,
        ),
        lock_callback=lock_busy,
    )
    return reconciliation_service.apply_parent(
        original_parent,
        operations=operations,
        taskdata=taskdata,
        lease_held=lease_held,
        verified_children=verified_children,
        generation=generation,
        hook=hook,
    )


def _recovery_error(parent: TaskPayload, reason: str) -> RecoveryRefusal:
    return RecoveryRefusal(
        DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile recovery error"),
        RecoveryStatus.ERROR,
        reason,
    )


def _recovery_terminal(parent: TaskPayload, reason: str) -> RecoveryRefusal:
    """Classify an expired-but-still-pending child as resumable, not corrupt."""
    if reason.endswith("native until has already elapsed"):
        return _recovery_partial(
            parent,
            f"{reason}; wait for Taskwarrior to mark the child deleted, then rerun reconcile",
        )
    return _recovery_error(parent, reason)


def _recovery_partial(parent: TaskPayload, reason: str) -> RecoveryRefusal:
    return RecoveryRefusal(
        DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile recovery partial"),
        RecoveryStatus.PARTIAL,
        reason,
    )


def _recovery_manual_review(parent: TaskPayload, reason: str) -> RecoveryRefusal:
    return RecoveryRefusal(
        DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile recovery review"),
        RecoveryStatus.MANUAL_REVIEW,
        reason,
    )


def _validate_recovery_child(parent: TaskPayload, child: TaskPayload) -> str:
    parent_observation = DEFAULT_TASK_CODEC.decode_row(
        parent,
        source_query="reconcile recovery parent verification",
    )
    child_observation = DEFAULT_TASK_CODEC.decode_row(
        child,
        source_query="reconcile recovery child verification",
    )
    _child_short, child_error = lifecycle.resolve_existing_child(
        parent_observation,
        [child_observation],
        include_deleted=True,
    )
    return child_error


def _terminal_recovery_error(child: TaskObservation, hook: Any, recovery_at: Any) -> str:
    if not isinstance(child, TaskObservation):
        raise TypeError("terminal recovery validation requires a TaskObservation")

    def value(field: str) -> Any:
        state = child.field(field)
        return state.raw_value() if state.presence is FieldPresence.VALUE else None

    if str(value("status") or "").strip().lower() != "pending":
        return ""
    until_raw = value("until")
    try:
        until_dt, until_err = _safe_parse_datetime(hook, until_raw)
    except Exception:
        return "live recovery child native until could not be parsed"
    if until_err or until_dt is None:
        return f"live recovery child has no reliable native until: {until_err or 'missing until'}"

    target_field = "due" if value("due") else "scheduled"
    target_raw = value(target_field)
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
    parent: TaskObservation,
    child_short: str,
) -> TaskObservation:
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
    child_observation = matches[0]
    if not isinstance(child_observation, TaskObservation):
        raise RuntimeError(f"recovery child {wanted} lookup returned an untyped task")
    child = child_observation.to_mapping()
    validation_error = _validate_recovery_child(parent.to_mapping(), child)
    if validation_error:
        raise RuntimeError(validation_error)
    return child_observation


def _virtual_expired_child(
    plan: LifecyclePlan,
    *,
    parent: TaskObservation,
    hook: Any,
    recovery_at: Any,
) -> tuple[VirtualExpiredChild | None, str]:
    if plan.action is not LifecycleAction.SPAWN_CHILD:
        return None, "planned child draft is unavailable"
    child = plan.child_dict()
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
        f"{lifecycle.int_or_default(child.get('link'), plan.identity.target_link or 0)}"
    )
    child.pop("nextLink", None)
    validation_error = _validate_recovery_child(parent.to_mapping(), child)
    if validation_error:
        return None, validation_error
    return VirtualExpiredChild(
        TaskObservation.from_mapping(child, source_query="reconcile virtual expiration")
    ), ""


def _reconcile_candidate(
    task_bin: str,
    hook: Any,
    parent: TaskPayload,
    *,
    taskdata: Path | None,
    apply: bool,
    max_expiration_hops: int,
    recovery_at: Any,
    lease_held: bool = False,
    generation: ChainGenerationService | None = None,
    reconciliation_service: LifecycleReconciliationService,
) -> list[tuple[RecoveryResult, str]]:
    def recovery_from_exception(candidate: dict[str, Any], exc: Exception) -> Any:
        reason = str(exc).strip() or type(exc).__name__
        if isinstance(exc, (_ConfigurationDrift, _LifecycleRetryable, _PlanReadUnavailable)):
            return _recovery_partial(candidate, reason)
        if isinstance(exc, _LifecycleManualReview):
            return _recovery_manual_review(candidate, reason)
        return _recovery_error(candidate, reason)

    return reconciliation_service.recover_candidate(
        parent,
        operations=CallbackLifecycleRecoveryOperations(
            apply_parent_callback=lambda candidate, **kwargs: _apply_parent_atomic(
                hook, candidate, reconciliation_service=reconciliation_service, **kwargs,
            ),
            plan_parent_callback=lambda candidate, **kwargs: _plan_for_parent(
                hook, candidate, reconciliation_service=reconciliation_service, **kwargs,
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

def _fmt_parent(parent: TaskPayload) -> str:
    uuid = lifecycle.short_uuid(parent.get("uuid")) or "????????"
    chain_id = str(parent.get("chainID") or "?")
    link = lifecycle.int_or_default(parent.get("link"), 0)
    desc = str(parent.get("description") or "").strip()
    return f"{uuid} chain {chain_id} link {link}" + (f" · {desc}" if desc else "")


def _print_evidence(evidence: dict[str, Any], keys: tuple[str, ...]) -> None:
    values = {
        key.replace("_", " "): evidence[key]
        for key in keys
        if evidence.get(key) not in (None, "")
    }
    for line in key_value_lines(values):
        print(f"  {line}")


def _describe_plan(plan: RecoveryResult, *, hook: Any, fmt_dt_local=None) -> dict[str, Any]:
    if isinstance(plan, RecoveryRefusal):
        return lifecycle.describe_recovery_result(plan, fmt_dt_local=fmt_dt_local)
    evidence = lifecycle.describe_recovery_result(plan, fmt_dt_local=fmt_dt_local)
    child_until = plan.plan.child_dict().get("until")
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


def _recovery_action(result: RecoveryResult) -> str:
    """Project a typed recovery result to the reconcile report action."""
    if isinstance(result, RecoveryRefusal):
        return result.status.value
    return {
        LifecycleAction.SPAWN_CHILD: "spawn",
        LifecycleAction.UPDATE_PARENT: "backfill_nextlink",
        LifecycleAction.FINALIZE_CHAIN: "legitimate_final",
        LifecycleAction.DISABLE_CHAIN: "manual_stop",
    }.get(result.plan.action, result.plan.action.value)


def _print_plan(
    plan: RecoveryResult,
    evidence: dict[str, Any] | None = None,
    *,
    applied_short: str = "",
) -> None:
    parent = _fmt_parent(plan.parent.to_mapping())
    if evidence is None:
        evidence = lifecycle.describe_recovery_result(plan)
    if isinstance(plan, RecoveryPlanResult) and plan.plan.action is LifecycleAction.SPAWN_CHILD:
        suffix = f" -> created {applied_short}" if applied_short else ""
        print(_style(f"spawn: {parent}{suffix}", _action_style("spawn")))
        _print_evidence(evidence, ("reason", "kind", "next_link", "child_field", "child_target", "child_due", "child_local", "child_expires", "expiration"))
    elif isinstance(plan, RecoveryPlanResult) and plan.plan.action is LifecycleAction.UPDATE_PARENT:
        suffix = " (applied)" if applied_short else ""
        print(_style(f"backfill nextLink: {parent}{suffix}", _action_style("backfill_nextlink")))
        _print_evidence(evidence, ("reason", "next_link", "existing_child"))
    elif isinstance(plan, RecoveryPlanResult) and plan.plan.action in {LifecycleAction.FINALIZE_CHAIN, LifecycleAction.DISABLE_CHAIN}:
        suffix = " -> set chain:off" if applied_short else ""
        print(_style(f"terminal: {parent} ({plan.reason}){suffix}", _action_style("legitimate_final")))
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))
    else:
        status = plan.status.value if isinstance(plan, RecoveryRefusal) else "error"
        print(_style(f"{status}: {parent} ({plan.reason})", _action_style(status)))
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))


def _print_recovery_group(
    items: list[tuple[RecoveryResult, dict[str, Any], str]],
) -> None:
    first = items[0][0]
    last, evidence, applied_short = items[-1]
    hops = sum(
        1 for plan, _evidence, _applied in items
        if isinstance(plan, RecoveryPlanResult)
        and plan.plan.action in {LifecycleAction.SPAWN_CHILD, LifecycleAction.UPDATE_PARENT}
    )
    noun = "occurrence" if hops == 1 else "occurrences"
    print(_style(f"recover: {_fmt_parent(first.parent.to_mapping())} -> advanced {hops} {noun}", "cyan"))
    if isinstance(last, RecoveryRefusal) or (
        isinstance(last, RecoveryPlanResult)
        and last.plan.action in {LifecycleAction.FINALIZE_CHAIN, LifecycleAction.DISABLE_CHAIN}
    ):
        result = last.status.value if isinstance(last, RecoveryRefusal) else "terminal"
        print(_style(f"  result: {result} ({last.reason})", _action_style(result)))
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
        print(render_json_document(payload, indent=2))
    else:
        print(_style(f"error: {stage.replace('_', ' ')}: {reason}", "red", stream=sys.stderr), file=sys.stderr)
    return 1


class _ReconcileSession:
    """Validated, task-scoped services shared by one reconcile invocation."""

    __slots__ = ("unit_of_work", "repository", "snapshot", "control_plane", "mutation_gateway", "integrity_outbox", "lifecycle_service", "lifecycle_application", "runtime_state")

    def __init__(self, unit_of_work, repository, snapshot, control_plane, mutation_gateway,
                 integrity_outbox, lifecycle_service, lifecycle_application, runtime_state):
        self.unit_of_work = unit_of_work
        self.repository = repository
        self.snapshot = snapshot
        self.control_plane = control_plane
        self.mutation_gateway = mutation_gateway
        self.integrity_outbox = integrity_outbox
        self.lifecycle_service = lifecycle_service
        self.lifecycle_application = lifecycle_application
        self.runtime_state = runtime_state

    def collect_candidates(self) -> tuple[TaskObservation, ...]:
        """Load the bounded candidate set for this invocation."""
        return tuple(self.lifecycle_service.candidates())

    def audit_integrity(self, *, hook: Any, apply: bool) -> tuple[Any, Any, float, tuple[Any, ...], float]:
        """Audit and, when authorized, apply integrity plans for this snapshot."""
        if self.snapshot._rows is None:
            return None, None, 0.0, (), 0.0
        started = time.perf_counter()
        engine, audit = _audit_reconcile_integrity(
            tuple(row.to_mapping() for row in self.snapshot._rows),
            outbox_repository=self.integrity_outbox,
        )
        audit_seconds = time.perf_counter() - started
        if not (apply and audit is not None and audit.plans):
            return engine, audit, audit_seconds, (), 0.0
        started = time.perf_counter()
        application = engine.apply(
            audit,
            executor=self.mutation_gateway,
            request_factory=_integrity_request_factory,
            outbox_repository=self.integrity_outbox,
            owner=f"reconcile-integrity-{os.getpid()}",
        )
        applications = application.applications
        application_seconds = time.perf_counter() - started
        if applications:
            self.snapshot.invalidate()
        return engine, audit, audit_seconds, applications, application_seconds

    def audit_native_until(self, request: ReconcileRequest, *, hook: Any, taskdata: Path | None, lease_held: bool) -> tuple[list[dict[str, Any]], list[str], str]:
        """Prepare native-until repairs through the shared control plane."""
        repairs, errors = _native_until_repairs(
            request.task_bin,
            hook,
            apply=request.apply,
            taskdata=taskdata,
            lease_held=lease_held,
            control_plane=self.control_plane,
        )
        return repairs, errors, _native_until_audit_result(repairs, errors).status

    def preflight_wave(self, candidates: tuple[TaskObservation, ...]) -> None:
        """Hydrate child-slot evidence once for the candidate wave."""
        self.lifecycle_service.preflight_wave(candidates)

    def plan_candidate(self, task_bin: str, hook: Any, parent: TaskObservation, *, taskdata: Path | None,
                       apply: bool, max_expiration_hops: int, recovery_at: Any,
                       lease_held: bool, generation: Any) -> list[tuple[RecoveryResult, str]]:
        """Plan one candidate through the session's shared lifecycle context."""
        if not isinstance(parent, TaskObservation):
            raise TypeError("reconcile planning requires a typed task observation")
        return _reconcile_candidate(
            task_bin, hook, parent.to_mapping(), taskdata=taskdata, apply=apply,
            max_expiration_hops=max_expiration_hops, recovery_at=recovery_at,
            lease_held=lease_held, generation=generation,
            reconciliation_service=self.lifecycle_service,
        )

    def execute_wave(self, *, hook: Any, taskdata: Path, wave_plans: dict[str, tuple[RecoveryPlanResult, str]],
                     configuration_fingerprint: str, schedule_fingerprint: str) -> Any:
        """Apply one guarded lifecycle wave through the session application service."""
        return _execute_reconcile_lifecycle_wave(
            hook,
            self.lifecycle_service,
            self.lifecycle_application,
            taskdata,
            wave_plans,
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )


def _build_reconcile_session(
    request: ReconcileRequest,
    unit_of_work: TaskwarriorUnitOfWork,
    *,
    budget: OperatorInvocationBudget | None = None,
) -> _ReconcileSession:
    repository = unit_of_work.repository
    repository.configure_commands(timeout=120.0, attempts=2, retry_delay=0.05)
    scope_filter = f"chainID:{request.chain_id}" if request.chain_id else f"uuid:{request.uuid}" if request.uuid else None
    snapshot = ReconcileSnapshotService(
        repository,
        scope_filter=scope_filter,
        full_audit=bool(request.full_audit),
        read_value=_read_value,
        stats=_EXPORT_STATS,
        budget=budget,
    )
    configuration = unit_of_work.context.configuration
    control_plane = OperatorControlPlane.from_configuration(configuration, DomainApplicationRegistry())
    from nautical_core.lifecycle_application import LifecycleApplicationService
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository
    mutation_gateway = TaskwarriorMutationService(unit_of_work)
    integrity_outbox = LifecycleOutboxRepository(unit_of_work.outbox.taskdata)
    lifecycle_application = LifecycleApplicationService(
        unit_of_work=unit_of_work,
        mutations=mutation_gateway,
        outbox=integrity_outbox,
        owner=f"reconcile-{os.getpid()}",
        lease_seconds=120.0,
    )
    lifecycle_service = LifecycleReconciliationService(
        snapshot,
        repository,
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
        unit_of_work=unit_of_work,
        application=lifecycle_application,
    )
    return _ReconcileSession(
        unit_of_work, repository, snapshot, control_plane, mutation_gateway,
        integrity_outbox, lifecycle_service, lifecycle_application,
        _ReconcileRuntimeState(repository, snapshot, lifecycle_service),
    )


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
    args = ReconcileRequest.from_namespace(parser.parse_args(argv))
    # Reconcile is intentionally broader than a scoped query, but remains
    # bounded so a corrupt/unbounded export cannot consume the process.
    budget = OperatorInvocationBudget(OperatorLimits(tasks=10_000, chains=1_000))
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
    session = _build_reconcile_session(args, _UNIT_OF_WORK, budget=budget)
    repository = session.repository
    snapshot = session.snapshot
    operator_control_plane = session.control_plane
    mutation_gateway = session.mutation_gateway
    integrity_outbox = session.integrity_outbox
    lifecycle_service = session.lifecycle_service
    lifecycle_application = session.lifecycle_application
    configuration = _UNIT_OF_WORK.context.configuration
    runtime_state = session.runtime_state
    _RECONCILE_RUNTIME.set(runtime_state)
    try:
        candidates = session.collect_candidates()
    except Exception as exc:
        return _startup_failure(args, "candidate_export", exc)
    integrity_audit_result: Any = None
    integrity_application_results: tuple[Any, ...] = ()
    integrity_seconds = 0.0
    integrity_application_seconds = 0.0
    stage_seconds: dict[str, float] = {
        "export": 0.0,
        "hydration": 0.0,
        "planning": 0.0,
        "mutation": 0.0,
        "verification": 0.0,
        "presentation": 0.0,
    }
    try:
        if snapshot._rows is not None:
            integrity_engine, integrity_audit_result, integrity_seconds, integrity_application_results, integrity_application_seconds = session.audit_integrity(
                hook=hook, apply=args.apply,
            )
        if integrity_audit_result is not None and integrity_audit_result.status.value == "unavailable":
            configuration_status = "unavailable"
            configuration_drift_reason = integrity_audit_result.reason or "integrity audit unavailable"
        elif integrity_audit_result is not None and args.apply and integrity_audit_result.plans:
            if integrity_application_results:
                candidates = session.collect_candidates()
    except Exception as exc:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] integrity audit unavailable: {type(exc).__name__}: {exc}", file=sys.stderr)
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
            integrity_drain_results = operator_control_plane.drain_integrity(
                integrity_outbox,
                unit_of_work=_UNIT_OF_WORK,
                executor=mutation_gateway,
                request_factory=_integrity_request_factory,
                owner=f"reconcile-integrity-{os.getpid()}",
            )
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
            native_until_repairs, native_until_errors, native_until_audit_status = session.audit_native_until(
                args, hook=hook, taskdata=taskdata, lease_held=_apply_lease_held,
            )
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
    plans: list[RecoveryResult] = []
    plan_evidence: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    outcome_groups: list[list[tuple[RecoveryResult, str]]] = []
    processed_slots: set[tuple[str, int]] = set()
    ambiguous_slots = IntegrityRecoveryService.ambiguous_candidate_slots(candidates)
    if configuration_status == "valid":
        hydration_started = time.perf_counter()
        try:
            session.preflight_wave(candidates)
        except Exception as exc:
            configuration_status = "unavailable"
            configuration_drift_reason = f"wave child-slot evidence unavailable: {type(exc).__name__}: {exc}"
        stage_seconds["hydration"] += time.perf_counter() - hydration_started
    wave_results: dict[str, tuple[RecoveryResult, str]] = {}
    # A wave is useful for batching multiple independent candidates.  For a
    # single candidate it only adds a dry planning pass before the same guarded
    # application path, so keep the one-candidate case on the direct route.
    if args.apply and len(candidates) > 1 and configuration_status == "valid" and taskdata is not None:
        wave_plans: dict[str, tuple[RecoveryPlanResult, str]] = {}
        planning_started = time.perf_counter()
        for parent_observation in candidates:
            parent = parent_observation.to_mapping()
            parent_uuid = str(parent.get("uuid") or "").strip().lower()
            if not parent_uuid or parent_uuid in ambiguous_slots:
                continue
            if str(parent.get("status") or "").strip().lower() != "completed":
                continue
            try:
                planned_outcomes = session.plan_candidate(
                    args.task_bin,
                    hook,
                    parent_observation,
                    taskdata=taskdata,
                    apply=False,
                    max_expiration_hops=args.max_expiration_hops,
                    recovery_at=recovery_at,
                    lease_held=_apply_lease_held,
                    generation=generation,
                )
            except Exception:
                continue
            if len(planned_outcomes) == 1:
                planned_outcome = planned_outcomes[0]
                if (
                    isinstance(planned_outcome[0], RecoveryPlanResult)
                    and planned_outcome[0].plan.action is LifecycleAction.SPAWN_CHILD
                ):
                    wave_plans[parent_uuid] = (planned_outcome[0], planned_outcome[1])
        stage_seconds["planning"] += time.perf_counter() - planning_started
        if wave_plans:
            mutation_started = time.perf_counter()
            try:
                wave_result = session.execute_wave(
                    hook=hook,
                    taskdata=taskdata,
                    wave_plans=wave_plans,
                    configuration_fingerprint=configuration.fingerprint,
                    schedule_fingerprint=configuration.scheduler_fingerprint,
                )
                if wave_result is not None:
                    wave_results = wave_result
            except Exception as exc:
                if os.environ.get("NAUTICAL_DIAG") == "1":
                    print(f"[nautical] reconcile lifecycle wave deferred: {type(exc).__name__}: {exc}", file=sys.stderr)
            stage_seconds["mutation"] += time.perf_counter() - mutation_started
    for parent_observation in candidates:
        planning_started = time.perf_counter()
        parent = parent_observation.to_mapping()
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
        parent_uuid = str(parent.get("uuid") or "").strip().lower()
        if parent_uuid in wave_results:
            outcomes = [wave_results[parent_uuid]]
        elif parent_slot in ambiguous_slots:
            outcomes = [(_recovery_error(parent, ambiguous_slots[parent_slot]), "")]
        else:
            mutation_started = time.perf_counter() if args.apply else 0.0
            try:
                outcomes = session.plan_candidate(
                    args.task_bin,
                    hook,
                    parent_observation,
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
            if args.apply:
                stage_seconds["mutation"] += time.perf_counter() - mutation_started
        stage_seconds["planning"] += time.perf_counter() - planning_started
        outcome_groups.append(outcomes)
        if configuration_status == "valid":
            configuration_drift_reason = next(
                (
                    plan.reason
                    for plan, _applied in outcomes
                    if isinstance(plan, RecoveryRefusal)
                    and plan.status is RecoveryStatus.PARTIAL
                    and str(plan.reason).startswith("configuration ")
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
        presentation_started = time.perf_counter()
        rendered: list[tuple[RecoveryResult, dict[str, Any], str]] = []
        for plan, applied_short in outcomes:
            plan_parent = plan.parent.to_mapping()
            processed_slots.add(
                (
                    str(plan_parent.get("chainID") or "").strip(),
                    lifecycle.int_or_default(plan_parent.get("link"), 0),
                )
            )
            plans.append(plan)
            evidence = _describe_plan(plan, hook=hook, fmt_dt_local=fmt_dt_local)
            plan_evidence.append(evidence)
            rendered.append((plan, evidence, applied_short))
            if args.apply and applied_short:
                action = _recovery_action(plan)
                disabling = action in {"legitimate_final", "manual_stop"}
                record = {
                    "action": action,
                    "parent": lifecycle.short_uuid(plan_parent.get("uuid")),
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
        stage_seconds["presentation"] += time.perf_counter() - presentation_started

    expiration_hops = sum(
        1
        for plan in plans
        if str(plan.parent.to_mapping().get("status") or "").strip() == "deleted"
        and _recovery_action(plan) in {"spawn", "backfill_nextlink"}
    )
    recovered_chains = sum(
        1
        for outcomes in outcome_groups
        if sum(
            1
            for plan, _applied in outcomes
        if str(plan.parent.to_mapping().get("status") or "").strip() == "deleted"
            and _recovery_action(plan) in {"spawn", "backfill_nextlink"}
        )
        > 1
        and all(_recovery_action(plan) not in {"error", "partial"} for plan, _applied in outcomes)
    )
    native_until_manual_review = sum(
        1 for item in native_until_repairs if item.get("action") == "manual_review"
    )
    native_until_audit_skipped = int(bool(native_until_audit_warning))
    degraded = (
        any(_recovery_action(plan) == "partial" for plan in plans)
        or any(_recovery_action(plan) == "manual_review" for plan in plans)
        or native_until_manual_review > 0
        or native_until_audit_skipped > 0
        or bool(configuration_drift_reason)
        or any(item.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED} for item in integrity_drain_results)
    )
    plan_errors = sum(1 for plan in plans if _recovery_action(plan) == "error")
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
    # Export timing is owned by the repository metrics; integrity audit timing
    # is the authoritative verification segment for this invocation.
    stage_seconds["export"] = float(_EXPORT_STATS["seconds"])
    stage_seconds["verification"] += integrity_seconds
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
        "spawn": sum(1 for p in plans if _recovery_action(p) == "spawn"),
        "backfill_nextlink": sum(1 for p in plans if _recovery_action(p) == "backfill_nextlink"),
        "legitimate_final": sum(1 for p in plans if _recovery_action(p) == "legitimate_final"),
        "terminal": sum(
            1 for p in plans
            if isinstance(p, RecoveryPlanResult)
            and p.plan.action in {LifecycleAction.FINALIZE_CHAIN, LifecycleAction.DISABLE_CHAIN}
        ),
        "manual_stop": sum(1 for p in plans if _recovery_action(p) == "manual_stop"),
        "stale": sum(1 for p in plans if _recovery_action(p) == "stale"),
        "partial": sum(1 for p in plans if _recovery_action(p) == "partial"),
        "manual_review": sum(1 for p in plans if _recovery_action(p) == "manual_review"),
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
        "stage_seconds": {key: round(value, 6) for key, value in stage_seconds.items()},
        "export_calls": _EXPORT_STATS["calls"],
        "export_rows": _EXPORT_STATS["rows"],
        "export_seconds": round(_EXPORT_STATS["seconds"], 4),
        "slowest_export_seconds": round(_EXPORT_STATS["slowest_seconds"], 4),
        "snapshot_hits": _EXPORT_STATS["snapshot_hits"],
        "task_command_calls": int(_UNIT_OF_WORK.commands.calls),
        "task_command_attempts": int(_UNIT_OF_WORK.commands.attempts),
        "task_command_duration": round(_UNIT_OF_WORK.commands.duration, 6),
        "task_command_failures": int(_UNIT_OF_WORK.commands.failures),
        "task_command_by_purpose": dict(_UNIT_OF_WORK.commands.by_purpose),
        "task_command_budget": int(_UNIT_OF_WORK.commands.context.command_budget),
        "task_command_budget_exceeded": bool(_UNIT_OF_WORK.commands.budget_exceeded),
        "lock_contention": dict(_LOCK_STATS),
        "plans": [
            {
                "action": _recovery_action(plan),
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
        print(render_result(to_operator_result(summary), "json", budget=budget))
    else:
        summary_line, diagnostics_line = render_human(summary, _style)
        print(summary_line)
        print(diagnostics_line)
    return exit_code(summary)


if __name__ == "__main__":
    raise SystemExit(main())
