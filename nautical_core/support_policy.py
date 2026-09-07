"""Authoritative runtime support policy shared by diagnostics and CI."""

from __future__ import annotations

MIN_PYTHON: tuple[int, int] = (3, 11)
MIN_TASKWARRIOR: tuple[int, int, int] = (3, 4, 2)
TESTED_TASKWARRIOR_VERSIONS: tuple[str, ...] = ("3.4.2", "3.5.0")


def policy_document() -> dict[str, object]:
    """Return JSON-safe policy data for operator diagnostics and probes."""
    return {
        "minimum_python": ".".join(str(part) for part in MIN_PYTHON),
        "minimum_taskwarrior": ".".join(str(part) for part in MIN_TASKWARRIOR),
        "tested_taskwarrior": list(TESTED_TASKWARRIOR_VERSIONS),
    }


__all__ = (
    "MIN_PYTHON",
    "MIN_TASKWARRIOR",
    "TESTED_TASKWARRIOR_VERSIONS",
    "policy_document",
)
