"""Presentation helpers for immutable operator-control-plane results."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date, datetime, timezone
from dataclasses import dataclass
from typing import Any

from .operator_models import OperatorResult, OperatorV2Result
from .lifecycle_models import LifecycleDrainProgress


def bounded_text(value: object, *, width: int = 120) -> str:
    """Bound presentation labels without altering the underlying result data."""
    limit = max(1, int(width))
    text = str(value or "").replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    if limit == 1:
        return "…"
    return text[: limit - 1].rstrip() + "…"


def key_value_lines(values: Mapping[str, object]) -> tuple[str, ...]:
    """Format stable key/value summaries without embedding operational logic."""
    return tuple(f"{bounded_text(key, width=32)}: {bounded_text(value)}" for key, value in values.items())


def render_result(result: OperatorResult | OperatorV2Result, mode: str = "json", *, rich_renderer: Any = None) -> str:
    """Route one immutable result to a presentation mode without changing it."""
    if not isinstance(result, (OperatorResult, OperatorV2Result)):
        raise TypeError("operator presentation requires an operator result")
    normalized = str(mode or "json").strip().lower()
    if normalized == "json":
        return render_json(result)
    if normalized == "text":
        return render_text(result)
    if normalized in {"disabled", "none"}:
        return ""
    if normalized == "rich":
        if not callable(rich_renderer):
            raise ValueError("Rich presentation requires a renderer callback")
        try:
            rendered = rich_renderer(result)
        except Exception:
            return f"{render_text(result)} [presentation unavailable]"
        return "" if rendered is None else str(rendered)
    raise ValueError(f"unsupported presentation mode: {mode!r}")


@dataclass(frozen=True, slots=True)
class ProgressView:
    """Immutable, renderer-neutral view of one lifecycle progress event."""

    stage: str
    completed: int
    total: int
    label: str = ""

    @classmethod
    def from_event(cls, event: LifecycleDrainProgress) -> "ProgressView":
        if not isinstance(event, LifecycleDrainProgress):
            raise TypeError("progress presentation requires LifecycleDrainProgress")
        label = str(event.detail or event.outcome or "").replace("_", " ").strip()
        return cls(event.stage.value, event.completed, event.total, label)

    @property
    def fraction(self) -> float | None:
        return None if self.total == 0 else self.completed / self.total


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def render_json(result: OperatorResult | OperatorV2Result) -> str:
    """Serialize one validated result without changing the result or its data."""
    if not isinstance(result, (OperatorResult, OperatorV2Result)):
        raise TypeError("operator presentation requires an operator result")
    return json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=_json_default)


def render_contract_json(value: object) -> str:
    """Serialize any validated contract exposing ``to_dict`` consistently."""
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("presentation value must expose to_dict()")
    return json.dumps(to_dict(), ensure_ascii=False, separators=(",", ":"), default=_json_default)


def render_json_document(value: object, *, indent: int | None = None) -> str:
    """Serialize a contract or mapping with the shared Unicode-safe policy."""
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        to_dict = getattr(value, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("presentation value must be a mapping or contract")
        payload = to_dict()
    separators = None if indent is not None else (",", ":")
    return json.dumps(payload, ensure_ascii=False, separators=separators, indent=indent, default=_json_default)


def render_text(result: OperatorResult | OperatorV2Result) -> str:
    """Render a concise deterministic summary from the same result envelope."""
    if not isinstance(result, (OperatorResult, OperatorV2Result)):
        raise TypeError("operator presentation requires an operator result")
    operation = result.operation.value if hasattr(result.operation, "value") else result.operation
    status = result.status.value if hasattr(result.status, "value") else result.status
    line = f"{operation}: {status}"
    if result.failure is not None:
        line += f" — {bounded_text(result.failure.message)}"
    if result.page is not None:
        position = result.page.cursor.position if result.page.cursor is not None else 0
        suffix = " complete" if result.page.complete else " more available"
        line += f" (items {position + 1}-{position + len(result.page.items)};{suffix})"
    return line


def ordered_records(
    records: object,
    *,
    keys: tuple[str, ...] = (),
) -> tuple[Mapping[str, Any], ...]:
    """Return mapping records in a deterministic, presentation-only order."""
    if not isinstance(records, (list, tuple)):
        return ()
    values = [item for item in records if isinstance(item, Mapping)]
    if not keys:
        keys = ("domain", "severity", "status", "chain_id", "link", "task_uuid", "code")
    return tuple(sorted(values, key=lambda item: tuple(str(item.get(key) or "") for key in keys)))


def ordered_findings(findings: object) -> tuple[Mapping[str, Any], ...]:
    """Apply one stable ordering for finding-like mappings without mutating input."""
    if not isinstance(findings, (list, tuple)):
        return ()
    severity = {"error": 0, "warning": 1, "warn": 1, "info": 2, "ok": 3}
    actionability = {"repairable": 0, "manual_review": 1, "historical": 2}
    values = [item for item in findings if isinstance(item, Mapping)]
    return tuple(sorted(
        values,
        key=lambda item: (
            severity.get(str(item.get("severity") or "").lower(), 9),
            actionability.get(str(item.get("status") or "").lower(), 9),
            str(item.get("chain_id") or ""),
            int(item.get("link") or 0),
            str(item.get("task_uuid") or item.get("id") or ""),
            str(item.get("invariant_id") or item.get("code") or ""),
        ),
    ))


__all__ = ("ProgressView", "bounded_text", "key_value_lines", "ordered_findings", "ordered_records", "render_contract_json", "render_json", "render_json_document", "render_result", "render_text")
