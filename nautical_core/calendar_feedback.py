from __future__ import annotations

from datetime import date, datetime
from typing import Any, Callable


def render_business_calendar_displacement(
    task: dict[str, Any],
    occurrence: date | datetime,
    *,
    core: Any,
    panel: Callable[..., None],
) -> bool:
    calendar_name = str(task.get("bc") or "").strip().lower()
    if not calendar_name:
        return False
    try:
        adjusted = core.to_local(occurrence).date() if isinstance(occurrence, datetime) else occurrence
        displacement = core.business_calendar_displacement_for_date(
            adjusted,
            calendar_name=calendar_name,
        )
    except Exception:
        return False
    if displacement is None:
        return False

    delta_days = (displacement.adjusted - displacement.original).days
    panel(
        "⚓ Business calendar adjusted",
        [
            ("Calendar", displacement.calendar_name),
            ("Original", displacement.original.strftime("%a %Y-%m-%d")),
            (
                "Adjusted",
                f"{displacement.adjusted.strftime('%a %Y-%m-%d')} ({delta_days:+d}d)",
            ),
        ],
        kind="note",
    )
    return True


__all__ = ("render_business_calendar_displacement",)
