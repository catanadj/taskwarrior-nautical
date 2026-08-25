"""Presentation-neutral rendering adapter for immutable workflow facts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .hook_workflow_models import FeedbackFacts


@dataclass(frozen=True, slots=True)
class PanelView:
    """One deterministic view consumed by Rich, static, or JSON renderers."""

    title: str
    kind: str
    rows: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        title = str(self.title or "").strip()
        kind = str(self.kind or "note").strip() or "note"
        if not title:
            raise ValueError("panel view requires a title")
        rows = tuple((str(label).strip(), str(value)) for label, value in self.rows)
        if any(not label for label, _value in rows):
            raise ValueError("panel view row labels cannot be empty")
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rows", rows)

    def to_diagnostic(self) -> dict[str, Any]:
        """Return the same facts in a machine-readable, non-hook envelope."""
        return {
            "title": self.title,
            "kind": self.kind,
            "rows": [{"label": label, "value": value} for label, value in self.rows],
        }


def panel_view_from_facts(
    facts: FeedbackFacts,
    *,
    title: str = "Nautical workflow",
    kind: str = "note",
) -> PanelView:
    """Build one stable view without importing UI, hooks, or mutation code."""
    if not isinstance(facts, FeedbackFacts):
        raise TypeError("feedback rendering requires FeedbackFacts")
    rows: list[tuple[str, str]] = []
    if facts.task_uuid:
        rows.append(("Task", facts.task_uuid))
    if facts.chain_id:
        rows.append(("Chain", facts.chain_id))
    if facts.natural_explanation:
        rows.append(("Natural", facts.natural_explanation))
    if facts.first_occurrence is not None:
        rows.append(("First", facts.first_occurrence.value.isoformat()))
    if facts.next_occurrence is not None:
        rows.append(("Next", facts.next_occurrence.value.isoformat()))
    if facts.carry_changes:
        rows.append(("Carry", "; ".join(f"{key}: {value}" for key, value in facts.carry_changes)))
    if facts.limits:
        rows.append(("Limits", "; ".join(f"{key}: {value}" for key, value in facts.limits)))
    if facts.changed_fields:
        rows.append(("Changed", ", ".join(facts.changed_fields)))
    rows.extend(("Warning", value) for value in facts.warnings)
    rows.extend(("Recovery", value) for value in facts.recovery_guidance)
    if facts.next_action:
        rows.append(("Next action", facts.next_action))
    return PanelView(title=title, kind=kind, rows=tuple(rows))


def render_panel_view(view: PanelView, panel: Callable[..., Any]) -> bool:
    """Render a view defensively; presentation failure cannot alter workflow state."""
    if not isinstance(view, PanelView):
        raise TypeError("rendering requires a PanelView")
    try:
        panel(view.title, list(view.rows), kind=view.kind)
    except Exception:
        return False
    return True


__all__ = ("PanelView", "panel_view_from_facts", "render_panel_view")
