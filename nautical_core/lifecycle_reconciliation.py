"""Lifecycle recovery ownership for the reconcile operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from . import chain_integrity_lifecycle as lifecycle
from .chain_generation import ChainGenerationService
from .chain_integrity_engine import ChainIntegrityEngine
from .integration_models import Absent, Found, Unavailable
from .lifecycle_models import DeletionDisposition


class LifecycleSnapshot(Protocol):
    def candidate_rows(self) -> list[dict[str, Any]]: ...


class LifecycleChildRepository(Protocol):
    def exact_child_slot(self, chain_id: str, link: int, *, refresh: bool = False) -> object: ...


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


__all__ = ["LifecycleReconciliationService"]
