"""Typed outcomes shared by recurrence scheduling paths."""

from __future__ import annotations

from typing import Any


class OccurrenceSearchExhausted(RuntimeError):
    """Raised when a bounded scheduler search cannot find a valid occurrence."""

    DATE_LIMIT = "date_limit"
    SEARCH_LIMIT = "search_limit"

    def __init__(
        self,
        scope: str,
        *,
        reference: Any = None,
        limit: int | None = None,
        kind: str | None = None,
    ) -> None:
        self.scope = str(scope)
        self.reference = reference
        self.limit = limit
        # Keep legacy constructors source-compatible while making the terminal
        # reason machine-readable for preview, navigator, and reconcile.
        reference_year = getattr(reference, "year", 0)
        requested_kind = kind or self.SEARCH_LIMIT
        # A bounded search can still be the final search when its reference is
        # already at the supported upper date boundary.
        self.kind = self.DATE_LIMIT if reference_year >= 9999 else requested_kind
        if requested_kind not in {self.DATE_LIMIT, self.SEARCH_LIMIT}:
            raise ValueError(f"unsupported occurrence exhaustion kind: {requested_kind}")
        detail = f" for {reference!s}" if reference is not None else ""
        bound = f" after {limit} iterations" if limit is not None else ""
        super().__init__(
            f"No matching occurrence found in {self.scope}{detail}{bound}; "
            "refusing to invent a date."
        )

    @property
    def is_date_limit(self) -> bool:
        return self.kind == self.DATE_LIMIT


def occurrence_exhaustion_message(exc: OccurrenceSearchExhausted) -> str:
    """Return stable, user-facing wording for a terminal scheduler outcome."""
    if exc.is_date_limit:
        return "No further matching occurrences are representable through 9999-12-31."
    return str(exc).strip() or type(exc).__name__


__all__ = ("OccurrenceSearchExhausted", "occurrence_exhaustion_message")
