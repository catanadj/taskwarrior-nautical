"""Bounded, diagnostic-only scheduler tracing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping, cast


_ACTIVE_TRACE: ContextVar["SchedulerTrace | None"] = ContextVar("nautical_scheduler_trace", default=None)


def _env_true(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_text(value: object, *, limit: int = 160) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _safe_value(value: object) -> object:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _safe_text(value) if isinstance(value, str) else value
    return _safe_text(value)


def _safe_term(value: object) -> str:
    """Keep recurrence terms useful while hiding file and URL-like values."""
    text = _safe_text(value, limit=96)
    if "/" in text or "\\" in text or "://" in text:
        return "<redacted>"
    return text


@dataclass(frozen=True, slots=True)
class SchedulerTraceEvent:
    """One redacted scheduling decision or terminal explanation."""

    phase: str
    provider: str = ""
    status: str = ""
    cursor: object = None
    candidate: object = None
    term: str = ""
    reason: str = ""
    terminal: Mapping[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {"phase": _safe_text(self.phase, limit=48)}
        for key, value in (
            ("provider", self.provider),
            ("status", self.status),
            ("cursor", self.cursor),
            ("candidate", self.candidate),
            ("term", _safe_term(self.term)),
            ("reason", self.reason),
        ):
            if value not in (None, ""):
                data[key] = _safe_value(value)
        if self.terminal:
            data["terminal"] = {
                _safe_text(key, limit=32): _safe_value(value)
                for key, value in self.terminal.items()
            }
        return data


@dataclass(slots=True)
class SchedulerTrace:
    """A bounded in-memory trace which emits only through diagnostics."""

    enabled: bool = False
    max_events: int = 128
    _events: list[SchedulerTraceEvent] = field(default_factory=list, repr=False)
    _dropped: int = field(default=0, repr=False)
    _decision_count: int = field(default=0, repr=False)
    _last_decision_count: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.max_events, bool) or not isinstance(self.max_events, int) or self.max_events <= 0:
            raise ValueError("Scheduler trace max_events must be a positive integer.")

    @classmethod
    def from_env(cls, env: Mapping[str, object] | None = None) -> "SchedulerTrace":
        values = env if env is not None else os.environ
        enabled = _env_true(values.get("NAUTICAL_SCHEDULER_TRACE")) and (
            _env_true(values.get("NAUTICAL_DIAG")) or _env_true(values.get("NAUTICAL_DIAG_LOG"))
        )
        try:
            max_events = int(str(values.get("NAUTICAL_SCHEDULER_TRACE_MAX", "128")))
        except (TypeError, ValueError):
            max_events = 128
        return cls(enabled=enabled, max_events=max(1, min(max_events, 4096)))

    def record(self, phase: str, **kwargs: object) -> None:
        if not self.enabled:
            return
        self._decision_count += 1
        if len(self._events) >= self.max_events:
            self._dropped += 1
            return
        self._events.append(SchedulerTraceEvent(phase=str(phase), **cast(Any, kwargs)))

    @property
    def events(self) -> tuple[SchedulerTraceEvent, ...]:
        return tuple(self._events)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def decision_count(self) -> int:
        """Number of scheduler decisions, including events beyond the cap."""
        return self._decision_count

    @property
    def last_decision_count(self) -> int:
        """Number of decisions in the most recently cleared trace window."""
        return self._last_decision_count

    def summary(self) -> dict[str, object]:
        return {
            "events": [event.to_dict() for event in self._events],
            "count": len(self._events),
            "decision_count": self._decision_count,
            "dropped": self._dropped,
        }

    def emit(self, *, hook_name: str = "scheduler", data_dir: str | None = None) -> None:
        if not self.enabled or not self._events and not self._dropped:
            return
        from .diagnostic_models import DiagnosticEvent
        from .runtime import diag

        for event in self._events:
            payload = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"))
            diag(DiagnosticEvent("scheduler.trace", payload, hook=hook_name), hook_name, data_dir)
        if self._dropped:
            diag(
                DiagnosticEvent(
                    "scheduler.trace.truncated",
                    f"scheduler trace truncated after {self.max_events} events",
                    hook=hook_name,
                    level="warning",
                    context={"dropped": self._dropped, "max_events": self.max_events},
                ),
                hook_name,
                data_dir,
            )

    def clear(self) -> None:
        """Discard emitted events so one service call cannot replay history."""
        self._events.clear()
        self._dropped = 0
        self._last_decision_count = self._decision_count
        self._decision_count = 0


@contextmanager
def activate(trace: SchedulerTrace | None) -> Iterator[None]:
    """Make one trace visible to callback-based scheduler internals."""
    token = _ACTIVE_TRACE.set(trace if trace is not None and trace.enabled else None)
    try:
        yield
    finally:
        _ACTIVE_TRACE.reset(token)


def active_trace() -> SchedulerTrace | None:
    return _ACTIVE_TRACE.get()


__all__ = ("SchedulerTrace", "SchedulerTraceEvent", "activate", "active_trace")
