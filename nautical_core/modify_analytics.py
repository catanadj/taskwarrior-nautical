"""Completion analytics and chain-integrity diagnostics.

This module owns read-only chain analysis.  It deliberately receives the
small core callbacks it needs so hook orchestration does not own another
copy of the recurrence or formatting policy.
"""
from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .task_models import TaskObservation, TaskPayload


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _compact_delta(delta: timedelta, format_delta: Callable[[timedelta], str]) -> str:
    text = format_delta(delta)
    return text[1:] if text and text[0] in "+-" else text


def sort_chain_for_analytics(
    chain: list[TaskObservation],
    *,
    coerce_int: Callable[[Any, Any], int | None],
    parse_datetime: Callable[[Any], datetime | None],
) -> list[TaskObservation]:
    def link_sort_key(obj: TaskObservation) -> tuple[int, Any]:
        link = coerce_int(obj.get("link"), None)
        if link is not None:
            return (0, link)
        due = parse_datetime(obj.get("due")) or datetime.max.replace(tzinfo=timezone.utc)
        return (1, due)

    try:
        return sorted(chain, key=link_sort_key)
    except Exception:
        return chain[:]


def lateness_stats(
    chain: list[TaskObservation],
    *,
    parse_datetime: Callable[[Any], datetime | None],
    tol_secs: int = 60,
) -> dict[str, Any]:
    """Summarize completed-link timing without hook orchestration state."""
    early = on = late = 0
    deltas: list[float] = []
    best = None
    worst = None
    for obj in chain:
        due = parse_datetime(obj.get("due"))
        end = parse_datetime(obj.get("end"))
        if not (due and end):
            continue
        diff = (end - due).total_seconds()
        deltas.append(diff)
        if diff > tol_secs:
            late += 1
            worst = diff if worst is None or diff > worst else worst
        elif diff < -tol_secs:
            early += 1
            best = diff if best is None or diff < best else best
        else:
            on += 1
    avg = (sum(deltas) / len(deltas)) if deltas else None
    med = _median(deltas) if deltas else None
    return {
        "early": early,
        "on_time": on,
        "late": late,
        "avg": avg,
        "median": med,
        "best_early": best,
        "worst_late": worst,
        "count": len(deltas),
    }


def chain_health_advice(
    chain: list[TaskObservation],
    kind: str,
    task: TaskPayload,
    *,
    core: Any,
    parse_datetime: Callable[[Any], datetime | None],
    format_delta: Callable[[timedelta], str],
    coerce_int: Callable[[Any, Any], int | None],
    tol_secs: int,
    style: str,
) -> str | None:
    if not chain:
        return None

    ordered = sort_chain_for_analytics(
        chain,
        coerce_int=coerce_int,
        parse_datetime=parse_datetime,
    )
    completed = [
        item for item in ordered
        if str(item.get("status") or "").strip().lower() == "completed"
    ]
    completed_with_dates: list[TaskObservation] = []
    deltas: list[float] = []
    for item in completed:
        due = parse_datetime(item.get("due"))
        end = parse_datetime(item.get("end"))
        if due and end:
            completed_with_dates.append(item)
            deltas.append((end - due).total_seconds())

    on_time_rate = None
    streak = 0
    volatility = None
    if deltas:
        on_time = sum(1 for delta in deltas if abs(delta) <= tol_secs)
        on_time_rate = on_time / max(1, len(deltas))
        if len(deltas) >= 2:
            volatility = statistics.pstdev(deltas)
        for item in reversed(completed_with_dates):
            due = parse_datetime(item.get("due"))
            end = parse_datetime(item.get("end"))
            if due and end and abs((end - due).total_seconds()) <= tol_secs:
                streak += 1
            else:
                break

    due_list = [
        due for item in ordered
        if (due := parse_datetime(item.get("due"))) is not None
    ]
    drift_secs = None
    median_gap = None
    if len(due_list) >= 2:
        gaps = [
            (due_list[index] - due_list[index - 1]).total_seconds()
            for index in range(1, len(due_list))
        ]
        gaps = [gap for gap in gaps if gap > 0]
        if gaps:
            median = _median(gaps)
            if median is None:
                median_gap = None
            else:
                median_gap = median
            if median is not None and kind == "cp":
                interval = core.cp_sequence_interval_for_link(
                    task.get("cp") or "",
                    coerce_int(task.get("link"), 1),
                    str(task.get("chainID") or "").strip(),
                )
                if interval:
                    drift_secs = median_gap - interval.total_seconds()
            elif median is not None and len(gaps) >= 2:
                drift_secs = gaps[-1] - median

    style = (style or "coach").strip().lower()
    if style == "clinical":
        parts: list[str] = []
        if on_time_rate is not None:
            parts.append(f"OT {int(round(100.0 * on_time_rate))}%")
        if drift_secs is not None:
            parts.append(f"Drift {format_delta(timedelta(seconds=drift_secs))}")
        if streak:
            parts.append(f"Streak {streak}")
        if isinstance(volatility, (int, float)):
            parts.append(f"Vol {_compact_delta(timedelta(seconds=abs(volatility)), format_delta)}")
        return " | ".join(parts) if parts else None

    issues: list[str] = []
    tips: list[str] = []
    positives: list[str] = []
    if on_time_rate is not None:
        if on_time_rate < 0.6:
            issues.append("on-time rate is low")
            tips.append("try smaller scopes or later due times")
        elif on_time_rate < 0.8:
            issues.append("on-time is inconsistent")
            tips.append("adding a small buffer could help")
        else:
            positives.append("on-time is steady")
    if drift_secs is not None:
        base = median_gap
        if kind == "cp":
            interval = core.cp_sequence_interval_for_link(
                task.get("cp") or "",
                coerce_int(task.get("link"), 1),
                str(task.get("chainID") or "").strip(),
            )
            base = interval.total_seconds() if interval else None
        if base:
            if abs(drift_secs) > max(0.35 * base, 6 * 60 * 60):
                issues.append("cadence is drifting")
                tips.append("review cp/anchors for a better fit")
            else:
                positives.append("cadence is stable")
    if isinstance(volatility, (int, float)):
        if volatility > 24 * 60 * 60:
            issues.append("timing is noisy")
            tips.append("add buffer or split tasks")
        elif volatility < 6 * 60 * 60:
            positives.append("timing is consistent")

    if not issues:
        if streak >= 3:
            return f"Chain looks healthy with a {streak}-link on-time streak; keep the current cadence."
        if positives:
            return "Chain looks healthy; keep the current cadence."
        return None
    issue_text = ", ".join(issues)
    tip_text = "; ".join(tips[:2]) if tips else "keep an eye on due time fit"
    if streak >= 3:
        return f"Chain needs attention ({issue_text}); {tip_text}, and keep the {streak}-link on-time streak going."
    return f"Chain needs attention ({issue_text}); {tip_text}."


def chain_integrity_warnings(
    chain: list[TaskObservation],
    *,
    expected_chain_id: str | None = None,
    coerce_int: Callable[[Any, Any], int | None],
    short: Callable[[Any], str],
) -> list[str]:
    if not isinstance(chain, list) or not chain:
        return []
    warnings: list[str] = []
    short_map: dict[str, TaskObservation] = {}
    link_map: dict[int, TaskObservation] = {}
    missing_link: list[str] = []
    for item in chain:
        uid = item.get("uuid")
        if uid:
            short_map[short(uid)] = item
        link = coerce_int(item.get("link"), None)
        if link:
            if link in link_map:
                warnings.append(f"duplicate link #{link} ({short(link_map[link].get('uuid'))} vs {short(uid)})")
            else:
                link_map[link] = item
        elif uid:
            missing_link.append(short(uid))
        if expected_chain_id is not None:
            chain_id = str(item.get("chainID") or "").strip()
            if not chain_id:
                warnings.append(f"missing chainID on {short(uid)}")
            elif chain_id != expected_chain_id:
                warnings.append(f"chainID mismatch on {short(uid)}")

    if missing_link:
        sample = ", ".join(missing_link[:3])
        warnings.append(f"missing link number on {sample}{'…' if len(missing_link) > 3 else ''}")
    if link_map:
        links = sorted(link_map)
        if links[0] != 1:
            warnings.append(f"chain starts at link #{links[0]} (expected #1)")
        gaps = sorted(set(range(links[0], links[-1] + 1)) - set(links))
        if gaps:
            warnings.append(f"missing link(s): {', '.join(str(gap) for gap in gaps[:5])}{'…' if len(gaps) > 5 else ''}")
    for item in chain:
        if not isinstance(item, dict):
            continue
        current = short(item.get("uuid"))
        previous = str(item.get("prevLink") or "").strip()
        if previous:
            previous_item = short_map.get(previous)
            if not previous_item:
                warnings.append(f"{current} prevLink {previous} not found")
            elif str(previous_item.get("nextLink") or "").strip() != current:
                warnings.append(f"{current} prevLink {previous} not reciprocal")
        following = str(item.get("nextLink") or "").strip()
        if following:
            following_item = short_map.get(following)
            if not following_item:
                warnings.append(f"{current} nextLink {following} not found")
            elif str(following_item.get("prevLink") or "").strip() != current:
                warnings.append(f"{current} nextLink {following} not reciprocal")
    return list(dict.fromkeys(warnings))


__all__ = (
    "chain_health_advice",
    "chain_integrity_warnings",
    "lateness_stats",
    "sort_chain_for_analytics",
)
