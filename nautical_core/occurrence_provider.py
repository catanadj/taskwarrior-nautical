"""Typed occurrence values and provider contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Occurrence:
    """A local calendar occurrence independent of Taskwarrior task shape."""

    day: date
    hour: int
    minute: int
    source: str = "anchor"
    description: str = ""

    @property
    def hhmm(self) -> tuple[int, int]:
        return self.hour, self.minute


class OccurrenceProvider(Protocol):
    def occurrences(self) -> list[Occurrence]:
        """Return sorted, deduplicated local occurrences."""


__all__ = ("Occurrence", "OccurrenceProvider")
