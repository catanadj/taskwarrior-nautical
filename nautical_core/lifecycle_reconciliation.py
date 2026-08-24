"""Lifecycle recovery ownership for the reconcile operator."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import fcntl
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from . import chain_integrity_lifecycle as lifecycle
from .chain_generation import ChainGenerationService
from .chain_integrity_engine import ChainIntegrityEngine
from .integration_models import Absent, Found, Unavailable
from .lifecycle_models import DeletionDisposition, VirtualExpiredChild
from .lifecycle_state import parent_nextlink_lock_path, reconcile_lock_path
from .cache_locking import safe_lock
from .task_models import FieldPresence, TaskObservation, TaskPayload


_PARENT_LOCK_RETRIES = 600
_PARENT_LOCK_SLEEP_SECONDS = 0.1
_PARENT_LOCK_STALE_SECONDS = 300.0
_RECONCILE_LOCK_STALE_SECONDS = 300.0


class LifecycleSnapshot(Protocol):
    def candidate_rows(self) -> list[TaskObservation]: ...


class LifecycleChildRepository(Protocol):
    def exact_child_slot(self, chain_id: str, link: int, *, refresh: bool = False) -> object: ...


class LifecycleRecoveryOperations(Protocol):
    """Policy operations used by the service-owned recovery loop."""

    def apply_parent(self, parent: TaskPayload, *, taskdata: Path, lease_held: bool,
                     verified_children: dict[str, dict[str, Any]], generation: ChainGenerationService | None) -> tuple[Any, str]: ...
    def plan_parent(self, parent: TaskPayload, *, generation: ChainGenerationService | None) -> Any: ...
    def next_child(self, parent: TaskObservation, child_short: str) -> TaskObservation: ...
    def virtual_child(self, plan: Any, *, recovery_at: Any) -> tuple[VirtualExpiredChild | None, str]: ...
    def terminal_error(self, child: TaskObservation, recovery_at: Any) -> str: ...
    def is_orphan_deleted(self, child: TaskObservation) -> bool: ...
    def recovery_error(self, parent: TaskPayload, reason: str) -> Any: ...
    def recovery_partial(self, parent: TaskPayload, reason: str) -> Any: ...
    def recovery_manual_review(self, parent: TaskPayload, reason: str) -> Any: ...
    def recovery_terminal(self, parent: TaskPayload, reason: str) -> Any: ...
    def recovery_from_exception(self, parent: TaskPayload, exc: Exception) -> Any: ...


class LifecycleApplyOperations(Protocol):
    def configuration_state(self, hook: Any) -> tuple[str, str]: ...
    def refresh_plan(self, parent: TaskPayload, *, generation: ChainGenerationService | None) -> Any: ...
    def execute_plan(self, plan: Any, *, verified_children: dict[str, dict[str, Any]] | None,
                     label: str, strict_uuid: bool) -> str: ...
    def terminal_plan(self, plan: Any) -> str: ...
    def lock_busy(self, kind: str) -> None: ...


@dataclass(frozen=True, slots=True)
class CallbackLifecycleApplyOperations:
    """Taskwarrior-specific callbacks used by the service-owned dispatcher."""

    configuration_callback: Callable[..., tuple[str, str]]
    refresh_callback: Callable[..., Any]
    execute_callback: Callable[..., str]
    terminal_callback: Callable[..., str]
    lock_callback: Callable[..., None]

    def configuration_state(self, hook):
        return self.configuration_callback(hook)

    def refresh_plan(self, parent, *, generation):
        return self.refresh_callback(parent, generation=generation)

    def execute_plan(self, plan, *, verified_children, label, strict_uuid):
        return self.execute_callback(
            plan, verified_children=verified_children, label=label, strict_uuid=strict_uuid,
        )

    def terminal_plan(self, plan):
        return self.terminal_callback(plan)

    def lock_busy(self, kind):
        self.lock_callback(kind)


@dataclass(frozen=True, slots=True)
class CallbackLifecycleRecoveryOperations:
    """Typed callback port for the reconcile-specific Taskwarrior adapter.

    The service owns the recovery loop; this small port keeps Taskwarrior
    mechanics outside it while avoiding another operator-side orchestration
    class.
    """

    apply_parent_callback: Callable[..., tuple[Any, str]]
    plan_parent_callback: Callable[..., Any]
    next_child_callback: Callable[..., TaskObservation]
    virtual_child_callback: Callable[..., tuple[VirtualExpiredChild | None, str]]
    terminal_error_callback: Callable[..., str]
    is_orphan_deleted_callback: Callable[..., bool]
    recovery_error_callback: Callable[..., Any]
    recovery_partial_callback: Callable[..., Any]
    recovery_manual_review_callback: Callable[..., Any]
    recovery_terminal_callback: Callable[..., Any]
    recovery_exception_callback: Callable[..., Any]

    def apply_parent(self, parent, **kwargs):
        return self.apply_parent_callback(parent, **kwargs)

    def plan_parent(self, parent, **kwargs):
        return self.plan_parent_callback(parent, **kwargs)

    def next_child(self, parent, child_short):
        return self.next_child_callback(parent, child_short)

    def virtual_child(self, plan, **kwargs):
        return self.virtual_child_callback(plan, **kwargs)

    def terminal_error(self, child, recovery_at):
        return self.terminal_error_callback(child, recovery_at)

    def is_orphan_deleted(self, child):
        return self.is_orphan_deleted_callback(child)

    def recovery_error(self, parent, reason):
        return self.recovery_error_callback(parent, reason)

    def recovery_partial(self, parent, reason):
        return self.recovery_partial_callback(parent, reason)

    def recovery_manual_review(self, parent, reason):
        return self.recovery_manual_review_callback(parent, reason)

    def recovery_terminal(self, parent, reason):
        return self.recovery_terminal_callback(parent, reason)

    def recovery_from_exception(self, parent, exc):
        return self.recovery_exception_callback(parent, exc)



def _sort_key(row: TaskObservation) -> tuple[str, int, str, str]:
    def value(name: str) -> object:
        state = row.field(name)
        if state.presence is FieldPresence.ABSENT:
            return None
        return getattr(state.value, "value", state.value)

    return (
        str(value("chainID") or "").strip().casefold(),
        lifecycle.int_or_default(value("link"), 0),
        str(value("status") or "").strip().casefold(),
        str(value("uuid") or "").strip().casefold(),
    )


@dataclass(frozen=True, slots=True)
class LifecycleReconciliationService:
    """Own candidate selection and lifecycle schedule projection."""

    snapshot: LifecycleSnapshot
    repository: LifecycleChildRepository
    configuration_fingerprint: str
    schedule_fingerprint: str
    unit_of_work: Any = None

    @contextmanager
    def reconcile_lock(self, taskdata: Path):
        with safe_lock(
            reconcile_lock_path(taskdata), retries=1, sleep_base=0.0,
            stale_after=_RECONCILE_LOCK_STALE_SECONDS,
            fcntl_mod=fcntl, os_mod=os, time_mod=time, random_mod=random,
        ) as acquired:
            yield acquired

    @contextmanager
    def parent_lock(self, taskdata: Path, parent_uuid: str):
        with safe_lock(
            parent_nextlink_lock_path(taskdata, parent_uuid),
            retries=_PARENT_LOCK_RETRIES,
            sleep_base=_PARENT_LOCK_SLEEP_SECONDS,
            stale_after=_PARENT_LOCK_STALE_SECONDS,
            fcntl_mod=fcntl, os_mod=os, time_mod=time, random_mod=random,
        ) as acquired:
            yield acquired

    def application_service(self) -> Any:
        """Build the sole lifecycle mutation service for this invocation."""
        if self.unit_of_work is None:
            raise RuntimeError("lifecycle application requires a mutation-capable unit of work")
        from .lifecycle_application import LifecycleApplicationService
        from .lifecycle_outbox import LifecycleOutboxRepository
        from .taskwarrior_mutations import TaskwarriorMutationService
        import os

        return LifecycleApplicationService(
            unit_of_work=self.unit_of_work,
            mutations=TaskwarriorMutationService(self.unit_of_work),
            outbox=LifecycleOutboxRepository(self.unit_of_work.outbox.taskdata),
            owner=f"reconcile-{os.getpid()}",
            lease_seconds=120.0,
        )

    def execute_lifecycle_plan(
        self,
        plan: Any,
        *,
        configuration_fingerprint: str,
        schedule_fingerprint: str,
        resolve_plan: Any,
        verify_child: Any,
        verified_children: dict[str, dict[str, Any]] | None,
        strict_uuid: bool,
        label: str,
    ) -> tuple[Any, Any, str, dict[str, Any] | None]:
        """Stage, execute, and verify one successor lifecycle plan."""
        service = self.application_service()
        resolved = resolve_plan(plan)
        lifecycle_plan = getattr(resolved, "lifecycle_plan", None)
        if lifecycle_plan is None:
            raise RuntimeError(f"reconcile {label} plan has no typed lifecycle plan: {resolved!r}")
        staged = service.stage(
            lifecycle_plan,
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )
        if not staged.ok:
            return staged, None, "", None
        outcome = service.execute_staged(
            lifecycle_plan,
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )
        if not outcome.ok:
            return staged, outcome, "", None
        child_short = lifecycle_plan.parent_patch_dict().get("nextLink") or resolved.child_short
        if not child_short:
            return staged, outcome, "", None
        verified = verify_child(resolved.parent, child_short, strict_uuid=strict_uuid)
        if verified_children is not None:
            verified_children[str(child_short).strip().lower()] = verified
        return staged, outcome, str(child_short), verified

    def apply_terminal_plan(self, plan: Any, *, terminal_plan_factory: Any) -> Any:
        """Apply a typed terminal lifecycle plan through the shared service."""
        outcome = self.application_service().apply_immediate(terminal_plan_factory(plan))
        return outcome

    def apply_parent(
        self,
        parent: TaskPayload,
        *,
        operations: LifecycleApplyOperations,
        taskdata: Path,
        lease_held: bool = False,
        verified_children: dict[str, dict[str, Any]] | None = None,
        generation: ChainGenerationService | None = None,
        hook: Any = None,
    ) -> tuple[Any, str]:
        """Own locks and action dispatch for one candidate mutation."""
        parent_uuid = str(parent.get("uuid") or "").strip()
        if not parent_uuid:
            raise RuntimeError("parent task has no UUID")
        lock = self.reconcile_lock(taskdata) if not lease_held else nullcontext(True)
        with lock as reconcile_acquired:
            if not reconcile_acquired:
                operations.lock_busy("reconcile")
                raise RuntimeError("another reconcile apply is already running")
            with self.parent_lock(taskdata, parent_uuid) as acquired:
                if not acquired:
                    operations.lock_busy("parent")
                    raise RuntimeError(f"parent reconcile lock busy: {lifecycle.short_uuid(parent_uuid)}")
                status, reason = operations.configuration_state(hook)
                if status != "valid":
                    raise RuntimeError(reason)
                plan = operations.refresh_plan(parent, generation=generation)
                if plan.action == "spawn":
                    return plan, operations.execute_plan(
                        plan, verified_children=verified_children,
                        label="transition", strict_uuid=True,
                    )
                if plan.action == "backfill_nextlink":
                    return plan, operations.execute_plan(
                        plan, verified_children=verified_children,
                        label="backfill", strict_uuid=False,
                    )
                if plan.action in {"legitimate_final", "manual_stop"}:
                    return plan, operations.terminal_plan(plan)
                return plan, ""


    def candidates(self) -> list[TaskObservation]:
        rows = self.snapshot.candidate_rows()
        candidates = [
            row for row in rows
            if str(getattr(row.field("status").value, "value", row.field("status").value) or "").strip().lower() == "completed"
            and lifecycle.is_orphan_completion_candidate(row)
        ]
        candidates.extend(
            row for row in rows
            if str(getattr(row.field("status").value, "value", row.field("status").value) or "").strip().lower() == "deleted"
            and lifecycle.is_orphan_deleted_chain_candidate(row)
        )
        return sorted(candidates, key=_sort_key)

    def plan(
        self,
        parent: TaskObservation,
        *,
        hook: Any,
        generation: ChainGenerationService,
        safe_parse_datetime: Any,
    ) -> lifecycle.LifecycleRecoveryDecision:
        existing_children = self.existing_children(
            parent,
            safe_parse_datetime=safe_parse_datetime,
        )
        engine = ChainIntegrityEngine.lifecycle_only(
            configuration_fingerprint=self.configuration_fingerprint,
            schedule_fingerprint=self.schedule_fingerprint,
        )
        return engine.plan_recovery(
            parent,
            existing_children=existing_children,
            hook=hook,
            generation=generation,
        )

    def existing_children(self, parent: TaskObservation, *, safe_parse_datetime: Any) -> tuple[TaskObservation, ...]:
        if not isinstance(parent, TaskObservation):
            raise TypeError("lifecycle child lookup requires a TaskObservation parent")
        if str(getattr(parent.field("status").value, "value", parent.field("status").value) or "").strip().lower() == "deleted":
            evidence = lifecycle.deleted_chain_disposition(
                parent,
                safe_parse_datetime=safe_parse_datetime,
            )
            if evidence.disposition is not DeletionDisposition.EXPIRATION:
                return []
        chain_id = str(getattr(parent.field("chainID").value, "value", parent.field("chainID").value) or "").strip()
        next_link = lifecycle.int_or_default(getattr(parent.field("link").value, "value", parent.field("link").value), 1) + 1
        if not chain_id:
            return ()
        result = self.repository.exact_child_slot(chain_id, next_link, refresh=True)
        if isinstance(result, Unavailable):
            raise RuntimeError(result.evidence.detail or f"child slot {chain_id}:{next_link} unavailable")
        if isinstance(result, Absent):
            return ()
        if isinstance(result, Found):
            if not isinstance(result.value, TaskObservation):
                raise RuntimeError(f"child slot {chain_id}:{next_link} returned an untyped observation")
            return (result.value,)
        raise RuntimeError(f"child slot {chain_id}:{next_link} returned an invalid read result")

    def recover_candidate(
        self,
        parent: TaskPayload,
        *,
        operations: LifecycleRecoveryOperations,
        taskdata: Path | None,
        apply: bool,
        max_expiration_hops: int,
        recovery_at: Any,
        lease_held: bool = False,
        generation: ChainGenerationService | None = None,
    ) -> list[tuple[Any, str]]:
        """Run bounded successor recovery; policy and mutation stay typed ports."""
        outcomes: list[tuple[Any, str]] = []
        current = parent
        visited: set[tuple[str, int]] = set()
        expiration_hops = 0
        verified_children: dict[str, dict[str, Any]] = {}
        while True:
            slot = (str(current.get("chainID") or "").strip(), lifecycle.int_or_default(current.get("link"), 0))
            if slot in visited:
                outcomes.append((operations.recovery_error(current, "expiration recovery made no progress"), ""))
                break
            visited.add(slot)
            is_deleted = str(current.get("status") or "").strip().lower() == "deleted"
            if is_deleted and expiration_hops >= max_expiration_hops:
                outcomes.append((operations.recovery_partial(current, f"expiration recovery hop limit reached at {max_expiration_hops}; rerun to continue or increase --max-expiration-hops"), ""))
                break
            if apply:
                if taskdata is None:
                    raise RuntimeError("Taskwarrior data location is unavailable")
                try:
                    plan, applied_short = operations.apply_parent(
                        current, taskdata=taskdata, lease_held=lease_held,
                        verified_children=verified_children, generation=generation,
                    )
                except Exception as exc:
                    outcomes.append((operations.recovery_from_exception(current, exc), ""))
                    break
            else:
                try:
                    plan = operations.plan_parent(current, generation=generation)
                except Exception as exc:
                    outcomes.append((operations.recovery_from_exception(current, exc), ""))
                    break
                applied_short = ""
            outcomes.append((plan, applied_short))
            if not is_deleted or plan.action not in {"spawn", "backfill_nextlink"}:
                break
            expiration_hops += 1
            child_short = applied_short or plan.child_short
            plan_parent = plan.parent.to_mapping()
            try:
                child = operations.next_child(plan.parent, child_short) if (apply or plan.action == "backfill_nextlink") else None
                if child is None:
                    virtual_child, child_error = operations.virtual_child(plan, recovery_at=recovery_at)
                    if child_error:
                        outcomes.append((operations.recovery_error(plan_parent, child_error), ""))
                        break
                    if virtual_child is None:
                        if plan.child:
                            child = TaskObservation.from_mapping(
                                plan.child,
                                source_query="reconcile planned child verification",
                            )
                            terminal_error = operations.terminal_error(child, recovery_at)
                            if terminal_error:
                                outcomes.append((operations.recovery_terminal(plan_parent, terminal_error), ""))
                        break
                    child = virtual_child.observation
            except Exception as exc:
                outcomes.append((operations.recovery_from_exception(plan_parent, exc), ""))
                break
            terminal_error = operations.terminal_error(child, recovery_at)
            if terminal_error:
                outcomes.append((operations.recovery_terminal(plan_parent, terminal_error), ""))
                break
            if not operations.is_orphan_deleted(child):
                break
            current = child.to_mapping()
        return outcomes


__all__ = [
    "CallbackLifecycleApplyOperations",
    "CallbackLifecycleRecoveryOperations",
    "LifecycleReconciliationService",
]
