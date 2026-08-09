"""Compatibility bindings for the former on-modify generation helpers.

Production lifecycle code binds directly to :class:`ChainGenerationService`.
These delegates keep the historical hook-module names available while callers
and tests migrate to the service boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class GenerationCompatibilityBindings:
    """Legacy-shaped delegates backed by a task-scoped service getter."""

    service_getter: Callable[[], Any]

    def compute_cp_child_due(self, parent: dict[str, Any]):
        return self.service_getter().compute_cp_child_due(parent)

    def compute_anchor_child_due(self, parent: dict[str, Any]):
        try:
            return self.service_getter().compute_anchor_child_due(parent)
        except ValueError as exc:
            if "omission scan exceeded" in str(exc):
                raise ValueError("No valid anchor occurrences found after applying omit rules.") from exc
            raise

    def carry_relative_datetime(
        self,
        parent: dict,
        child: dict,
        child_due_utc: datetime,
        field: str,
        *,
        parent_anchor_field: str = "due",
        child_anchor_field: str = "due",
    ) -> None:
        if not isinstance(parent, dict) or not isinstance(child, dict):
            return
        self.service_getter().carry_relative_datetime(
            parent,
            child,
            child_due_utc,
            field,
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_anchor_field,
        )

    def carry_native_until(
        self,
        parent: dict,
        child: dict,
        child_due_utc: datetime,
        kind: str,
        *,
        parent_anchor_field: str = "due",
        child_anchor_field: str = "due",
    ) -> None:
        if not isinstance(parent, dict) or not isinstance(child, dict):
            return
        self.service_getter().carry_native_until(
            parent,
            child,
            child_due_utc,
            kind,
            parent_anchor_field=parent_anchor_field,
            child_anchor_field=child_anchor_field,
        )

    def build_child_from_parent(
        self,
        parent: dict,
        child_due_utc: datetime,
        child_field: str,
        next_link_no: int,
        parent_short: str,
        kind: str,
        cpmax: int,
        until_dt: Any,
    ) -> dict:
        return self.service_getter().build_child_from_parent(
            parent,
            child_due_utc,
            child_field,
            next_link_no,
            parent_short,
            kind,
            cpmax,
            until_dt,
        )


def bind_generation_compat(service_getter: Callable[[], Any]) -> GenerationCompatibilityBindings:
    """Bind legacy helper names to a dynamic task-scoped service getter."""
    return GenerationCompatibilityBindings(service_getter)


__all__ = ("GenerationCompatibilityBindings", "bind_generation_compat")
