"""Port-driven presentation effects used by the typed on-modify routes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .task_models import TaskPayload


@dataclass(frozen=True, slots=True)
class LifecycleResultPort:
    panel: Any


@dataclass(frozen=True, slots=True)
class ChainStylePorts:
    root_uuid: Any
    chain_colour_root: Any
    per_chain: bool


def chain_style_ports_for(host: Any) -> ChainStylePorts:
    return ChainStylePorts(
        root_uuid=host._module("modify_task_fields").root_uuid,
        chain_colour_root=host.core.chain_colour_root,
        per_chain=host._CHAIN_COLOR_PER_CHAIN,
    )


def lifecycle_result_port_for(host: Any) -> LifecycleResultPort:
    ui = host._module("modify_ui_effects")
    return LifecycleResultPort(
        panel=lambda title, rows, **kwargs: ui.panel(ui.ui_ports_for(host), title, rows, **kwargs)
    )


def chain_colour_for_task(ports: ChainStylePorts, task: TaskPayload, kind: str) -> str:
    """Resolve the configured presentation colour for a chain root."""
    return ports.chain_colour_root(kind, ports.root_uuid(task))


def future_style_for_chain(ports: ChainStylePorts, task: TaskPayload, kind: str) -> str:
    """Return static or per-chain future styling for timeline presentation."""
    if not ports.per_chain:
        return "dark_orange" if kind == "cp" else "cyan"
    return chain_colour_for_task(ports, task, kind)


def render_lifecycle_result(ports: LifecycleResultPort, result: Any, task: Any) -> None:
    """Render one finalized non-success outcome without deciding its state."""
    state = str(getattr(result, "state", "retryable") or "retryable").strip().lower()
    title = "⛓ Chain warning" if state == "manual_review" else "⛓ Chain error"
    rows = [("Result", state.replace("_", " ").title())]
    reason = str(getattr(result, "reason", "") or "").strip()
    if reason:
        rows.append(("Reason", reason))
    child_short = str(getattr(result, "child_short", "") or "").strip()
    if child_short:
        rows.append(("Child", child_short))
    intent_id = str(getattr(result, "spawn_intent_id", "") or "").strip()
    if intent_id:
        rows.append(("Intent", intent_id))
    ports.panel(title, rows, kind="warning" if state == "manual_review" else "error")


__all__ = (
    "LifecycleResultPort", "ChainStylePorts", "chain_style_ports_for",
    "lifecycle_result_port_for", "chain_colour_for_task", "future_style_for_chain",
    "render_lifecycle_result",
)
