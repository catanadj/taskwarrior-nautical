from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
import os
import sqlite3
from typing import Any, Protocol

from .chain_generation import AnchorChildDueResult, CpChildDueResult
from .modify_models import BuildChildDraftCallback, DiagnosticCallback, PanelCallback, ShortUuidCallback
from .modify_workflow import TerminalRouteDecision
from .task_changes import TaskTransition
from .task_models import TaskObservation, TaskPayload

from nautical_core.lifecycle.models import DeletionEvidence, LifecycleAction, LifecyclePlan
from nautical_core.lifecycle.outbox import LifecycleOutboxError
from nautical_core.lifecycle.recovery_models import RecoveryPlanResult, RecoveryRefusal, RecoveryResult
from nautical_core.task_codec import DEFAULT_TASK_CODEC, TaskCodecError


class _ExpirationCore(Protocol):
    def coerce_int(self, value: object, default: int) -> int: ...

    def fmt_dt_local(self, value: datetime) -> str: ...

    def _import_sibling(self, name: str) -> Any: ...

    def to_local(self, value: datetime) -> datetime: ...


class _ExpirationReconcile(Protocol):
    def deleted_chain_disposition(
        self,
        task: TaskObservation,
        *,
        safe_parse_datetime: Callable[[object], tuple[datetime | None, str | None]],
    ) -> DeletionEvidence: ...

    def is_orphan_expiration_candidate(
        self,
        task: TaskObservation,
        *,
        safe_parse_datetime: Callable[[object], tuple[datetime | None, str | None]],
    ) -> bool: ...

    def plan_recovery_decision(
        self,
        parent: TaskObservation,
        *,
        existing_children: Sequence[TaskObservation],
        hook: _ExpirationPlannerHost,
    ) -> RecoveryResult: ...


class _ExpirationPlannerHost(Protocol):
    @property
    def core(self) -> _ExpirationCore: ...


@dataclass(frozen=True, slots=True)
class _ExpirationPlannerHostValue:
    core: _ExpirationCore


class _EndChainSummary(Protocol):
    def __call__(
        self,
        current: TaskPayload,
        reason: str,
        now_utc: datetime,
        *,
        current_task: TaskPayload | None = None,
    ) -> None: ...


@dataclass(slots=True)
class ExpirationServices:
    core: _ExpirationCore
    reconcile: _ExpirationReconcile
    safe_parse_datetime: Callable[[object], tuple[datetime | None, str | None]]
    compute_anchor_child_due: Callable[[TaskPayload], AnchorChildDueResult]
    compute_cp_child_due: Callable[[TaskPayload], CpChildDueResult]
    build_child_draft: BuildChildDraftCallback
    stage_recovery_plan: Callable[[LifecyclePlan], tuple[bool, str]]
    panel: PanelCallback
    short: ShortUuidCallback
    diag: DiagnosticCallback


@dataclass(slots=True)
class DeletedModifyServices:
    expiration: ExpirationServices
    terminal_chain_off: Callable[[TaskPayload, str | None], bool]
    now_utc: Callable[[], datetime]
    end_chain_summary: _EndChainSummary
    format_root_and_age: Callable[[TaskPayload, datetime], str]
    short: ShortUuidCallback
    panel: PanelCallback
    diag: DiagnosticCallback
    recovery_warning: Callable[[TaskPayload, str], None]


def classify_deleted_task(
    task: TaskPayload,
    *,
    services: ExpirationServices,
    observation: TaskObservation | None = None,
) -> DeletionEvidence:
    """Return the deletion disposition without turning unavailable evidence into manual stop."""
    if observation is None:
        observation = DEFAULT_TASK_CODEC.decode_row(task, source_query="on-modify deletion classification")
    return services.reconcile.deleted_chain_disposition(
        observation,
        safe_parse_datetime=services.safe_parse_datetime,
    )


def render_recovery_warning(task: TaskPayload, reason: str, *, services: ExpirationServices) -> None:
    services.panel(
        "⚠ Nautical expiration recovery deferred",
        [
            ("Task", services.short(task.get("uuid")) or "–"),
            ("Reason", reason or "The next occurrence could not be prepared."),
            (
                "Action",
                "Lifecycle recovery will retry on-exit; run `nautical reconcile --apply` "
                "if it remains pending.",
            ),
        ],
        kind="warning",
    )


def _render_recovery_panel(
    task: TaskPayload,
    plan: RecoveryPlanResult,
    *,
    services: ExpirationServices,
    result: str = "",
    child_short: str = "",
) -> None:
    current_link = services.core.coerce_int(task.get("link"), 1)
    description = str(task.get("description") or "").strip()
    task_label = f"#{current_link}" + (f" · {description}" if description else "")
    rows = [("Expired", task_label)]
    if result:
        rows.append(("Result", result))
    if plan.child_due is not None:
        next_label = "Blocked next" if plan.plan.action is LifecycleAction.FINALIZE_CHAIN else "Next"
        rows.append((next_label, services.core.fmt_dt_local(plan.child_due)))
    child_until = plan.plan.child_dict().get("until")
    child_until_dt, child_until_err = services.safe_parse_datetime(child_until)
    if child_until_dt is not None and not child_until_err:
        if plan.child_due is not None:
            try:
                add_validation = services.core._import_sibling("add_validation")
                carry = add_validation.describe_native_until_carry(
                    child_until_dt,
                    plan.child_due,
                    to_local=services.core.to_local,
                )
            except ImportError as exc:
                services.diag(f"optional expiration carry description unavailable: {exc}")
                carry = None
            if carry:
                rows.append(("Expiration", carry))
        rows.append(("Next expires", services.core.fmt_dt_local(child_until_dt)))
    rows.append(("Link", f"#{plan.plan.identity.target_link}"))
    if child_short:
        rows.append(("Child", child_short))
    if plan.plan.action is LifecycleAction.FINALIZE_CHAIN:
        rows.append(("Boundary", plan.reason))
    panel_kind = "summary" if plan.plan.action is LifecycleAction.FINALIZE_CHAIN else "note"
    services.panel("⌛ Nautical occurrence expired", rows, kind=panel_kind)


def handle_expired_deleted_modify(task: TaskPayload, *, services: ExpirationServices) -> bool:
    reconcile = services.reconcile
    try:
        observation = DEFAULT_TASK_CODEC.decode_row(
            task,
            source_query="on-modify expiration recovery",
        )
    except TaskCodecError as exc:
        services.diag(f"expiration recovery task decode failed: {exc}")
        render_recovery_warning(task, "The expired task could not be validated for recovery.", services=services)
        return True
    if not reconcile.is_orphan_expiration_candidate(
        observation,
        safe_parse_datetime=services.safe_parse_datetime,
    ):
        return False

    plan_hook = _ExpirationPlannerHostValue(services.core)
    plan = reconcile.plan_recovery_decision(
        observation,
        existing_children=[],
        hook=plan_hook,
    )

    if isinstance(plan, RecoveryRefusal):
        render_recovery_warning(task, plan.reason, services=services)
        return True
    if not isinstance(plan, RecoveryPlanResult):
        render_recovery_warning(task, "Expiration recovery returned an invalid result.", services=services)
        return True

    if plan.plan.action is LifecycleAction.FINALIZE_CHAIN:
        render_recovery_warning(
            task,
            "The expired chain reached its terminal bound; lifecycle drain will finalize it.",
            services=services,
        )
        return True
    if plan.plan.action is not LifecycleAction.SPAWN_CHILD or not plan.plan.child_dict():
        render_recovery_warning(task, plan.reason, services=services)
        return True

    try:
        staged, reason = services.stage_recovery_plan(plan.plan)
    except (LifecycleOutboxError, OSError, sqlite3.Error) as exc:
        services.diag(f"expiration lifecycle staging failed: {exc}")
        reason = "The expired successor could not be staged for lifecycle drain."
        if os.environ.get("NAUTICAL_DIAG") == "1":
            detail = str(exc).strip() or type(exc).__name__
            reason = f"{reason} [{type(exc).__name__}: {detail}]"
        render_recovery_warning(task, reason, services=services)
        return True
    if staged:
        _render_recovery_panel(
            task,
            plan,
            services=services,
            result="[yellow]Next occurrence queued for lifecycle drain[/]",
            child_short=plan.child_short,
        )
    else:
        render_recovery_warning(
            task,
            reason or "The next occurrence could not be staged for lifecycle drain.",
            services=services,
        )
    return True


def handle_deleted_modify(
    old: TaskPayload,
    new: TaskPayload,
    *,
    services: DeletedModifyServices,
    transition: TaskTransition | None = None,
    terminal_decision: TerminalRouteDecision | None = None,
) -> None:
    """Classify one deleted pending task and converge its chain state."""
    old_status = (
        transition.old.field("status").raw_value()
        if transition is not None
        else old.get("status")
    )
    if str(old_status or "").strip().lower() != "pending":
        return
    old_chain_id = (
        transition.old.field("chainID").raw_value()
        if transition is not None
        else old.get("chainID")
    )
    new_chain_id = (
        transition.new.field("chainID").raw_value()
        if transition is not None
        else new.get("chainID")
    )
    if not ((old_chain_id or new_chain_id or "").strip()):
        return
    expiration = services.expiration
    try:
        evidence = classify_deleted_task(
            new,
            services=expiration,
            observation=(transition.new if transition is not None else None),
        )
        disposition = evidence.disposition.value
        disposition_reason = evidence.reason
    except TaskCodecError as exc:
        services.diag(f"deleted-task disposition failed: {exc}")
        services.recovery_warning(new, "Deletion evidence could not be classified safely.")
        return
    if disposition == "ambiguous":
        services.recovery_warning(
            new,
            disposition_reason or "Deletion evidence is unavailable or malformed.",
        )
        return
    if disposition == "expiration":
        if handle_expired_deleted_modify(new, services=expiration):
            return
        services.recovery_warning(
            new,
            "Expiration recovery could not be initialized; the chain remains active.",
        )
        return
    if disposition == "manual":
        services.diag("deleted Nautical task classified as manual stop")

    event = str(getattr(terminal_decision, "event", "manual_delete") or "manual_delete")
    services.terminal_chain_off(new, event)
    now_utc = services.now_utc()
    try:
        services.end_chain_summary(new, "Pending task deleted.", now_utc, current_task=old)
    except Exception as exc:
        services.diag(f"delete chain summary failed: {exc}")
        services.panel(
            "⛔ Nautical chain stopped",
            [
                ("Reason", "Pending Nautical task was deleted."),
                ("Root", services.format_root_and_age(old, now_utc)),
                ("Task", services.short(old.get("uuid")) or "–"),
            ],
            kind="summary",
        )


__all__ = (
    "ExpirationServices",
    "DeletedModifyServices",
    "classify_deleted_task",
    "handle_deleted_modify",
    "handle_expired_deleted_modify",
    "render_recovery_warning",
)
