"""Canonical moon-phase names used by the recurrence grammar."""

from __future__ import annotations

PHASES = ("new", "first-quarter", "full", "last-quarter")

_ALIASES = {
    "new": "new",
    "new-moon": "new",
    "first": "first-quarter",
    "first-quarter": "first-quarter",
    "first_quarter": "first-quarter",
    "firstquarter": "first-quarter",
    "full": "full",
    "full-moon": "full",
    "last": "last-quarter",
    "last-quarter": "last-quarter",
    "last_quarter": "last-quarter",
    "lastquarter": "last-quarter",
    "third-quarter": "last-quarter",
}


def canonical_phase(value: str) -> str | None:
    return _ALIASES.get(str(value or "").strip().lower())


__all__ = ("PHASES", "canonical_phase")
