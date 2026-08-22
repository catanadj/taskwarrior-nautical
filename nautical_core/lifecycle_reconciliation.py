"""Lifecycle recovery ownership for the reconcile operator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from . import chain_integrity_lifecycle as lifecycle
from .chain_generation import ChainGenerationService
from .chain_integrity_engine import ChainIntegrityEngine


class LifecycleSnapshot(Protocol):
    def candidate_rows(self) -> list[dict[str, Any]]: ...


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
    configuration_fingerprint: str
    schedule_fingerprint: str

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
        existing_children: list[dict[str, Any]],
        hook: Any,
        generation: ChainGenerationService,
    ) -> lifecycle.LifecycleRecoveryDecision:
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


__all__ = ["LifecycleReconciliationService"]
