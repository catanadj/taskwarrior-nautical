"""Typed outcomes shared by recurrence scheduling paths."""

from __future__ import annotations

from typing import Any


class OccurrenceSearchExhausted(RuntimeError):
    """Raised when a bounded scheduler search cannot find a valid occurrence."""

    def __init__(
        self,
        scope: str,
        *,
        reference: Any = None,
        limit: int | None = None,
    ) -> None:
        self.scope = str(scope)
        self.reference = reference
        self.limit = limit
        detail = f" for {reference!s}" if reference is not None else ""
        bound = f" after {limit} iterations" if limit is not None else ""
        super().__init__(
            f"No matching occurrence found in {self.scope}{detail}{bound}; "
            "refusing to invent a date."
        )


__all__ = ("OccurrenceSearchExhausted",)
