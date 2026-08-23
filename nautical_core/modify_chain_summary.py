"""Pure chain-summary row assembly for the on-modify presentation path."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from .task_models import TaskPayload


def summary_current(current: TaskPayload, current_task: TaskPayload | None) -> TaskPayload:
    return current_task if current_task else current


def summary_chain_id(current: TaskPayload) -> str:
    return (current.get("chainID") or "").strip()


def span_fields(
    chain_id: str,
    chain: list[TaskPayload],
    *,
    stop_at: datetime | None = None,
    stopped_by_delete: bool = False,
    export_endpoint: Callable[[str, str], dict | None],
    parse_datetime: Callable[[Any], datetime | None],
    human_delta: Callable[..., str],
) -> tuple[datetime | None, datetime | None, str]:
    first_task = chain[0] if chain else None
    last_task = chain[-1] if chain else None
    if not first_task and chain_id:
        first_task = export_endpoint(chain_id, "first")
    if not last_task and chain_id:
        last_task = export_endpoint(chain_id, "last")
    first = parse_datetime((first_task or {}).get("due")) if first_task else None
    last = parse_datetime((last_task or {}).get("end")) if last_task else None
    span = "–"
    if first and last:
        span = human_delta(first, last, prefer_months=True).replace("in ", "").replace("overdue by ", "")
    elif first and stop_at and stopped_by_delete:
        active = human_delta(first, stop_at, prefer_months=True).replace("in ", "").replace("overdue by ", "")
        span = f"Active for {active} before deletion"
    return first, last, span


def kind_rows(
    rows: list[tuple[str, str]],
    kind: str,
    current: dict,
    *,
    anchor_preset_display: Callable[[str], tuple[str, str] | None],
    validate_anchor: Callable[[str], Any],
    describe_anchor: Callable[[Any, dict], str],
) -> None:
    mode = (current.get("anchor_mode") or "skip").lower()
    tag = {"skip": "[cyan]SKIP[/]", "all": "[yellow]ALL[/]", "flex": "[magenta]FLEX[/]"}.get(
        mode,
        "[cyan]SKIP[/]",
    )
    if kind == "anchor":
        expr = (current.get("anchor") or "").strip()
        try:
            preset_display = anchor_preset_display(expr)
        except Exception:
            preset_display = None
        if preset_display:
            label, text = preset_display
            rows.append((label, f"{text}  {tag}"))
        else:
            rows.append(("Pattern", f"{expr}  {tag}"))
        try:
            rows.append(("Natural", describe_anchor(validate_anchor(expr), current)))
        except Exception:
            pass
        return
    if kind == "anchor_file":
        expr = (current.get("anchor_file") or "").strip()
        rows.append(("Anchor file", f"{expr}  {tag}"))
        rows.append(("Natural", f"Dates from {expr.split('@', 1)[0]}"))
        return
    rows.append(("Period", current.get("cp") or "–"))


def stats_rows(
    rows: list[tuple[str, str]],
    chain: list[dict],
    now_utc: Any,
    *,
    lateness_stats: Callable[..., dict[str, Any]],
    format_seconds_delta: Callable[[Any, float | None], str],
) -> None:
    stats = lateness_stats(chain)
    rows.append(("Performance", f"early {stats['early']}, on-time {stats['on_time']}, late {stats['late']}"))
    rows.append(("Avg lateness", format_seconds_delta(now_utc, stats["avg"])))
    rows.append(("Median lateness", format_seconds_delta(now_utc, stats["median"])))
    rows.append(("Best early", format_seconds_delta(now_utc, stats["best_early"])))
    rows.append(("Worst late", format_seconds_delta(now_utc, stats["worst_late"])))


def limits_row(
    rows: list[tuple[str, str]],
    current: dict,
    *,
    coerce_int: Callable[[Any, Any], int | None],
    parse_datetime: Callable[[Any], datetime | None],
    format_local: Callable[[datetime], str],
) -> None:
    cpmax = coerce_int(current.get("chainMax"), 0)
    until = parse_datetime(current.get("chainUntil"))
    if cpmax:
        rows.append(("Chain cap", f"#{cpmax}"))
    if until:
        rows.append(("Chain end point", format_local(until)))
    if not cpmax and not until:
        rows.append(("Chain limits", "None"))


def last_n_timeline(
    chain: list[dict[str, Any]],
    n: int = 6,
    *,
    coerce_int: Callable[[Any, Any], int | None],
    parse_datetime: Callable[[Any], datetime | None],
    format_local: Callable[[Any], str],
    format_on_time_delta: Callable[[Any, Any], str],
    short_uuid: Callable[[Any], str],
) -> list[str]:
    """Render the compact recent-history rows used by chain summaries."""
    if not chain:
        return []

    def get_link(task: TaskPayload) -> int:
        link = task.get("link")
        if link is None or link == "":
            return -1
        return coerce_int(link, 999999) or 999999

    chain_sorted = sorted(chain, key=get_link, reverse=True)
    chain_with_links = [task for task in chain_sorted if get_link(task) > 0]
    if chain_with_links:
        max_link = max(get_link(task) for task in chain_with_links)
        label_width = len(str(max_link)) + 1
    else:
        label_width = 4

    def history_line(task: TaskPayload, link_no: int) -> str:
        end = parse_datetime(task.get("end"))
        due = parse_datetime(task.get("due"))
        is_deleted = str(task.get("status") or "").strip().lower() == "deleted"
        if is_deleted and not end:
            end_s = "deleted"
            delta = ""
            marker = "×"
        else:
            end_s = format_local(end) if end else "(no end)"
            delta = format_on_time_delta(due, end)
            marker = "✓"
        label = f"[bold]#{link_no:<{label_width}}[/]"
        return f"{label} {marker:<2} {end_s} {delta} [dim]{short_uuid(task.get('uuid'))}[/]"

    if len(chain_with_links) > 10:
        top_tasks = chain_with_links[:3]
        bottom_tasks = chain_with_links[-3:]
        top_lines = []
        for task in top_tasks:
            link_no = get_link(task)
            line = history_line(task, link_no)
            if link_no == get_link(chain_with_links[0]):
                line = f"[green]{line}[/]"
            top_lines.append(line)
        ellipsis = f"[dim]{' ' * (label_width + 4)}... ({len(chain_with_links) - 6} more tasks) ...[/dim]"
        bottom_lines = [history_line(task, get_link(task)) for task in bottom_tasks]
        return top_lines + [ellipsis] + bottom_lines

    lines = []
    for task in chain_with_links[:n]:
        link_no = get_link(task)
        line = history_line(task, link_no)
        if link_no == get_link(chain_with_links[0]):
            line = f"[green]{line}[/]"
        lines.append(line)
    return lines


def render_chain_summary(
    current: dict[str, Any],
    reason: str,
    now_utc: Any,
    current_task: TaskPayload | None = None,
    *,
    export_sorted_chain: Callable[[str, dict[str, Any]], list[dict[str, Any]]],
    root_uuid_from: Callable[[dict[str, Any]], Any],
    short_uuid: Callable[[Any], str],
    format_root_and_age: Callable[[dict[str, Any], Any], str],
    kind_rows: Callable[[list[tuple[str, str]], str, dict[str, Any]], None],
    span_fields: Callable[..., tuple[datetime | None, datetime | None, str]],
    stats_rows: Callable[[list[tuple[str, str]], list[dict[str, Any]], Any], None],
    limits_row: Callable[[list[tuple[str, str]], dict[str, Any]], None],
    last_n_timeline_rows: Callable[[list[dict[str, Any]], int], list[str]],
    format_rows: Callable[[list[tuple[str, str]]], list[tuple[str | None, str]]],
    coerce_int: Callable[[Any, Any], int | None],
    format_local: Callable[[Any], str],
    max_chain_walk: int,
    panel: Callable[..., Any],
    diagnostic: Callable[[str], None],
) -> None:
    """Render the finished/stopped chain summary from focused collaborators."""
    actual_current = current_task if current_task else current
    kind_anchor = bool((actual_current.get("anchor") or "").strip())
    kind_anchor_file = bool((actual_current.get("anchor_file") or "").strip())
    kind = "anchor" if kind_anchor else ("anchor_file" if kind_anchor_file else "cp")
    chain_id = (actual_current.get("chainID") or "").strip()
    if not chain_id:
        panel(
            "⚠ Chain summary skipped",
            [
                ("Reason", "ChainID is required in v3+ and legacy link-walk is removed."),
                ("Fix", "Run dev_tools/nautical_backfill_chainid.py."),
            ],
            kind="warning",
        )
        return

    chain_read_error = ""
    try:
        chain = export_sorted_chain(chain_id, actual_current)
    except Exception as exc:
        chain = []
        chain_read_error = str(exc) or "chain export unavailable"
        diagnostic(f"chain summary export unavailable (chainID={chain_id}): {chain_read_error}")

    link_no = coerce_int(current.get("link"), len(chain))
    root = short_uuid(root_uuid_from(current))
    current_short = short_uuid(current.get("uuid"))
    stopped_by_delete = str(reason or "").strip().lower().startswith("pending task deleted")
    first, last, span = span_fields(
        chain_id,
        chain,
        stop_at=now_utc,
        stopped_by_delete=stopped_by_delete,
    )

    rows: list[tuple[str, str]] = [("Reason", reason), ("Root", format_root_and_age(current, now_utc))]
    chain_display = f"{root} … {current_short}  [dim](#{link_no}, {len(chain)} tasks"
    if len(chain) >= max_chain_walk:
        chain_display += f", truncated at {max_chain_walk})"
    else:
        chain_display += ")"
    rows.append(("Chain", chain_display))
    if chain_read_error:
        rows.append(("Chain read", f"Unavailable: {chain_read_error}"))

    kind_rows(rows, kind, current)
    if first:
        rows.append(("First due", format_local(first)))
    if last:
        rows.append(("Last end", format_local(last)))
    if stopped_by_delete:
        rows.append(("Stopped at", format_local(now_utc)))
    rows.append(("Span", span))
    stats_rows(rows, chain, now_utc)
    limits_row(rows, current)
    tail = last_n_timeline_rows(chain, 6)
    if tail:
        rows.append(("History", "\n".join(tail)))

    title = "⛔ Chain stopped – summary" if stopped_by_delete else "⛔ Chain finished – summary"
    panel(title, format_rows(rows), kind="summary")


__all__ = (
    "kind_rows",
    "last_n_timeline",
    "limits_row",
    "render_chain_summary",
    "span_fields",
    "stats_rows",
    "summary_chain_id",
    "summary_current",
)
