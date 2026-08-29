"""Invocation-scoped authoritative snapshot projections for reconcile."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .task_models import TaskObservation
from .task_read_repository import ALL_TASK_STATUSES, TaskReadRepository
from .operator_context import OperatorBudgetLedger
from .reconcile_cli import ReconcileRequest


class ReconcileSnapshotService:
    """Own one bounded export and its active/candidate projections."""

    def __init__(
        self,
        repository: TaskReadRepository,
        *,
        scope_filter: str | None = None,
        full_audit: bool = False,
        read_value: Callable[[object, str], object],
        stats: dict[str, Any] | None = None,
        budget: OperatorBudgetLedger | None = None,
    ) -> None:
        self.repository = repository
        self.scope_filter = str(scope_filter or "").strip() or None
        self.full_audit = bool(full_audit)
        self._read_value = read_value
        self._stats = stats
        self._budget = budget
        self._rows: tuple[TaskObservation, ...] | None = None
        self._active: tuple[TaskObservation, ...] | None = None
        self._candidates: tuple[TaskObservation, ...] | None = None

    @staticmethod
    def scope_filter_for(request: ReconcileRequest) -> str | None:
        """Compile reconcile's explicit scope into the repository selector."""
        if request.chain_id:
            return f"chainID:{request.chain_id}"
        if request.uuid:
            return f"uuid:{request.uuid}"
        return None

    @staticmethod
    def _text(row: TaskObservation, field: str) -> str:
        state = row.field(field)
        value = state.raw_value()
        return str(value or "").strip()

    def _all_rows(self) -> tuple[TaskObservation, ...]:
        if self._rows is None:
            if self._budget is not None and not self._budget.consume("taskwarrior_calls"):
                raise RuntimeError("operator Taskwarrior call budget exhausted before reconcile snapshot")
            value = self._read_value(
                self.repository.lifecycle_candidates(
                    statuses=ALL_TASK_STATUSES,
                    scope_filter=self.scope_filter,
                    bounded=not self.full_audit,
                ),
                "reconcile lifecycle snapshot",
            )
            self._rows = (
                value if isinstance(value, tuple) else (value,) if isinstance(value, TaskObservation) else ()
            )
            if self._budget is not None:
                row_count = len(self._rows)
                if row_count and (
                    not self._budget.consume("exported_rows", row_count)
                    or not self._budget.consume("decoded_rows", row_count)
                ):
                    raise RuntimeError("operator reconcile snapshot row budget exhausted")
                chain_count = len({
                    self._text(row, "chainID")
                    for row in self._rows
                    if self._text(row, "chainID")
                })
                if (
                    not self._budget.consume("tasks", row_count)
                    or (chain_count and not self._budget.consume("chains", chain_count))
                ):
                    raise RuntimeError("operator reconcile snapshot task or chain budget exhausted")
        elif self._stats is not None:
            self._stats["snapshot_hits"] = int(self._stats.get("snapshot_hits", 0)) + 1
        return self._rows

    def active_rows(self) -> list[TaskObservation]:
        if self._active is None:
            self._active = tuple(
                row for row in self._all_rows()
                if self._text(row, "chainID")
                and self._text(row, "status").lower() not in {"completed", "deleted"}
            )
        return list(self._active)

    def candidate_rows(self) -> list[TaskObservation]:
        if self._candidates is None:
            self._candidates = tuple(
                row for row in self._all_rows()
                if self._text(row, "chainID")
                and not self._text(row, "nextLink")
                and self._text(row, "status").lower() in {"completed", "deleted"}
            )
        return list(self._candidates)

    def invalidate(self) -> None:
        self._rows = None
        self._active = None
        self._candidates = None


__all__ = ["ReconcileSnapshotService"]
