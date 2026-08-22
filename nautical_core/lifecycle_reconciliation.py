"""Lifecycle recovery ownership for the reconcile operator."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from . import chain_integrity_lifecycle as lifecycle
from .chain_generation import ChainGenerationService
from .chain_integrity_engine import ChainIntegrityEngine
from .integration_models import Absent, Found, Unavailable
from .lifecycle_models import DeletionDisposition
from .lifecycle_state import parent_nextlink_lock_path, reconcile_lock_path
from .cache_locking import safe_lock


_PARENT_LOCK_RETRIES = 600
_PARENT_LOCK_SLEEP_SECONDS = 0.1
_PARENT_LOCK_STALE_SECONDS = 300.0
_RECONCILE_LOCK_STALE_SECONDS = 300.0


class LifecycleSnapshot(Protocol):
    def candidate_rows(self) -> list[dict[str, Any]]: ...


class LifecycleChildRepository(Protocol):
    def exact_child_slot(self, chain_id: str, link: int, *, refresh: bool = False) -> object: ...


class LifecycleRecoveryOperations(Protocol):
    """Policy operations used by the service-owned recovery loop."""

    def apply_parent(self, parent: dict[str, Any], *, taskdata: Path, lease_held: bool,
                     verified_children: dict[str, dict[str, Any]], generation: ChainGenerationService | None) -> tuple[Any, str]: ...
    def plan_parent(self, parent: dict[str, Any], *, generation: ChainGenerationService | None) -> Any: ...
    def next_child(self, parent: dict[str, Any], child_short: str) -> dict[str, Any]: ...
    def virtual_child(self, plan: Any, *, recovery_at: Any) -> tuple[dict[str, Any] | None, str]: ...
    def terminal_error(self, child: dict[str, Any], recovery_at: Any) -> str: ...
    def is_orphan_deleted(self, child: dict[str, Any]) -> bool: ...
    def recovery_error(self, parent: dict[str, Any], reason: str) -> Any: ...
    def recovery_partial(self, parent: dict[str, Any], reason: str) -> Any: ...
    def recovery_manual_review(self, parent: dict[str, Any], reason: str) -> Any: ...
    def recovery_terminal(self, parent: dict[str, Any], reason: str) -> Any: ...


def _sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("chainID") or "").strip().casefold(),
        lifecycle.int_or_default(row.get("link"), 0),
        str(row.get("status") or "").strip().casefold(),
        str(row.get("uuid") or "").strip().casefold(),
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
        ) as acquired:
            yield acquired

    @contextmanager
    def parent_lock(self, taskdata: Path, parent_uuid: str):
        with safe_lock(
            parent_nextlink_lock_path(taskdata, parent_uuid),
            retries=_PARENT_LOCK_RETRIES,
            sleep_base=_PARENT_LOCK_SLEEP_SECONDS,
            stale_after=_PARENT_LOCK_STALE_SECONDS,
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

    def candidates(self) -> list[dict[str, Any]]:
        rows = self.snapshot.candidate_rows()
        candidates = [
            row for row in rows
            if str(row.get("status") or "").strip().lower() == "completed"
            and lifecycle.is_orphan_completion_candidate(row)
        ]
        candidates.extend(
            row for row in rows
            if str(row.get("status") or "").strip().lower() == "deleted"
            and lifecycle.is_orphan_deleted_chain_candidate(row)
        )
        return sorted(candidates, key=_sort_key)

    def plan(
        self,
        parent: dict[str, Any],
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

    def existing_children(self, parent: dict[str, Any], *, safe_parse_datetime: Any) -> list[dict[str, Any]]:
        if str(parent.get("status") or "").strip().lower() == "deleted":
            evidence = lifecycle.deleted_chain_disposition(
                parent,
                safe_parse_datetime=safe_parse_datetime,
            )
            if evidence.disposition is not DeletionDisposition.EXPIRATION:
                return []
        chain_id = str(parent.get("chainID") or "").strip()
        next_link = lifecycle.int_or_default(parent.get("link"), 1) + 1
        if not chain_id:
            return []
        result = self.repository.exact_child_slot(chain_id, next_link, refresh=True)
        if isinstance(result, Unavailable):
            raise RuntimeError(result.evidence.detail or f"child slot {chain_id}:{next_link} unavailable")
        if isinstance(result, Absent):
            return []
        if isinstance(result, Found):
            return [dict(result.value)]
        raise RuntimeError(f"child slot {chain_id}:{next_link} returned an invalid read result")

    def recover_candidate(
        self,
        parent: dict[str, Any],
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
                    outcomes.append((operations.recovery_error(current, str(exc).strip() or type(exc).__name__), ""))
                    break
            else:
                try:
                    plan = operations.plan_parent(current, generation=generation)
                except Exception as exc:
                    outcomes.append((operations.recovery_error(current, str(exc).strip() or type(exc).__name__), ""))
                    break
                applied_short = ""
            outcomes.append((plan, applied_short))
            if not is_deleted or plan.action not in {"spawn", "backfill_nextlink"}:
                break
            expiration_hops += 1
            child_short = applied_short or plan.child_short
            try:
                child = operations.next_child(plan.parent, child_short) if (apply or plan.action == "backfill_nextlink") else None
                if child is None:
                    child, child_error = operations.virtual_child(plan, recovery_at=recovery_at)
                    if child_error:
                        outcomes.append((operations.recovery_error(plan.parent, child_error), ""))
                        break
                    if child is None:
                        terminal_error = operations.terminal_error(dict(plan.child or {}), recovery_at)
                        if terminal_error:
                            outcomes.append((operations.recovery_terminal(plan.parent, terminal_error), ""))
                        break
            except Exception as exc:
                outcomes.append((operations.recovery_error(plan.parent, str(exc).strip() or type(exc).__name__), ""))
                break
            terminal_error = operations.terminal_error(child, recovery_at)
            if terminal_error:
                outcomes.append((operations.recovery_terminal(plan.parent, terminal_error), ""))
                break
            if not operations.is_orphan_deleted(child):
                break
            current = child
        return outcomes


__all__ = ["LifecycleReconciliationService"]
