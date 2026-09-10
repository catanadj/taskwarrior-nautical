"""The single datetime parsing port used by task-facing workflows.

Taskwarrior fields are deliberately untyped at the wire boundary.  This
module turns that boundary into one small, defensive contract so callers do
not each need to know how the configured date formats are supplied or how a
broken parser should be reported.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Protocol


class TaskDatetimeParser(Protocol):
    """Parse one Taskwarrior value without raising into workflow code."""

    def parse(self, value: object) -> tuple[datetime | None, str | None]:
        ...


DiagnosticSink = Callable[[str], None]
DatetimeParseFn = Callable[[str], datetime | None]


class ConfiguredTaskDatetimeParser:
    """Adapter around the configured ``parse_dt_any`` implementation.

    The adapter owns the error taxonomy.  A parser implementation may return
    ``None`` or raise for malformed data; both are converted to the same
    caller-facing result, while unexpected failures are sent to diagnostics.
    """

    def __init__(
        self,
        parse_dt_any: DatetimeParseFn,
        *,
        diagnostic: DiagnosticSink | None = None,
    ) -> None:
        if not callable(parse_dt_any):
            raise TypeError("datetime parser requires a callable parse_dt_any")
        self._parse_dt_any = parse_dt_any
        self._diagnostic = diagnostic or (lambda _message: None)

    def parse(self, value: object) -> tuple[datetime | None, str | None]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None, None
        if not isinstance(value, str):
            self._diagnose(f"datetime parser received non-text value: {type(value).__name__}")
            return None, "Datetime value must be text"
        try:
            parsed = self._parse_dt_any(value)
        except (ValueError, TypeError) as exc:
            self._diagnose(f"datetime parser rejected value: {exc}")
            return None, "Invalid datetime value"
        except Exception as exc:  # pragma: no cover - defensive integration boundary
            self._diagnose(f"datetime parser failed: {exc}")
            return None, "Datetime parsing failed"
        if parsed is None:
            return None, f"Unrecognized datetime format '{value}'"
        if not isinstance(parsed, datetime):
            self._diagnose(
                f"datetime parser returned {type(parsed).__name__}, expected datetime"
            )
            return None, "Datetime parser returned an invalid value"
        return parsed, None

    def _diagnose(self, message: str) -> None:
        try:
            self._diagnostic(message)
        except Exception:
            # Diagnostics must never make malformed hook input fatal.
            pass


def parser_for_core(core: object, *, diagnostic: DiagnosticSink | None = None) -> TaskDatetimeParser:
    """Build the port from a configured core/facade object."""

    parse_dt_any = getattr(core, "parse_dt_any", None)
    if not callable(parse_dt_any):
        raise TypeError("configured core has no callable parse_dt_any")
    return ConfiguredTaskDatetimeParser(parse_dt_any, diagnostic=diagnostic)


def datetime_value(parser: TaskDatetimeParser, value: object) -> datetime | None:
    """Explicitly adapt the tuple port for callers needing only a value."""
    parsed, _error = parser.parse(value)
    return parsed


def parser_for_host(host: object, *, diagnostic: DiagnosticSink | None = None) -> TaskDatetimeParser:
    """Return the composition-root parser carried by a hook host."""
    parser = getattr(host, "_TASK_DATETIME_PARSER", None)
    if parser is not None and callable(getattr(parser, "parse", None)):
        return parser
    core = getattr(host, "core", None)
    return parser_for_core(core, diagnostic=diagnostic or getattr(host, "_diag", None))


__all__ = (
    "ConfiguredTaskDatetimeParser",
    "DatetimeParseFn",
    "DiagnosticSink",
    "TaskDatetimeParser",
    "datetime_value",
    "parser_for_core",
    "parser_for_host",
)
