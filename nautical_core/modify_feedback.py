from __future__ import annotations


from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from typing import Any


def append_next_wait_sched_rows(
    rows: list[tuple[str, str]],
    next_task: dict[str, Any],
    next_due_utc: datetime,
    *,
    anchor_field: str = "due",
    format_local: Callable[[Any], str],
    compare_datetimes: Callable[[datetime, datetime], int],
    format_delta: Callable[[timedelta], str],
) -> None:
    """Append next-link timing rows and explain invalid wait/scheduled order."""
    if not (isinstance(next_due_utc, datetime) and next_due_utc):
        return

    scheduled_value = next_task.timestamp("scheduled")
    wait_value = next_task.timestamp("wait")
    scheduled = scheduled_value.value if scheduled_value is not None else None
    wait = wait_value.value if wait_value is not None else None
    anchor_label = "scheduled" if anchor_field == "scheduled" else "due"
    for field, label, value in (
        ("scheduled", "Scheduled", scheduled),
        ("wait", "Wait", wait),
    ):
        if field == anchor_field or not isinstance(value, datetime):
            continue
        rows.append((label, f"{format_local(value)}  (Δ {format_delta(value - next_due_utc)})"))

    issues: list[str] = []
    if anchor_field != "scheduled" and isinstance(scheduled, datetime) and compare_datetimes(scheduled, next_due_utc) > 0:
        issues.append(f"scheduled is after {anchor_label} by {format_delta(scheduled - next_due_utc)}")
    if isinstance(wait, datetime) and compare_datetimes(wait, next_due_utc) > 0:
        issues.append(f"wait is after {anchor_label} by {format_delta(wait - next_due_utc)}")
    if anchor_field != "scheduled" and isinstance(scheduled, datetime) and isinstance(wait, datetime) and compare_datetimes(wait, scheduled) > 0:
        issues.append(f"wait is after scheduled by {format_delta(wait - scheduled)}")
    if not issues:
        return

    expected = "scheduled > wait" if anchor_field == "scheduled" else "due > scheduled > wait"
    rows.append(("⚠ Wait/Sched", f"Expected order: {expected}. " + "; ".join(issues)))
    rows.append(("⚠ Wait/Sched", "This can happen when due is auto-assigned; adjust scheduled/wait if undesired."))


def _format_td_short(td: timedelta) -> str:
    secs = int(td.total_seconds())
    if secs < 0:
        return "-" + _format_td_short(timedelta(seconds=-secs))
    if secs % 86400 == 0:
        return f"{secs // 86400}d"
    units = (("w", 604800), ("d", 86400), ("h", 3600), ("m", 60), ("s", 1))
    parts: list[str] = []
    rem = secs
    for label, unit_secs in units:
        if rem >= unit_secs:
            n, rem = divmod(rem, unit_secs)
            parts.append(f"{n}{label}")
    return "".join(parts) if parts else "0s"


def render_cp_schedule_adjusted_panel(
    adjustment: tuple[Any, Any, list[tuple[str, Any, Any, timedelta]]],
    *,
    format_local: Callable[[Any], str],
    semantic_diff_value: Callable[[str, str], str],
    format_offset: Callable[[timedelta], str],
    panel: Callable[..., Any],
) -> None:
    """Render the relative schedule changes applied after a CP due edit."""
    old_due, new_due, field_adjustments = adjustment
    rows = [("Due", semantic_diff_value(format_local(old_due), format_local(new_due)))]
    rows.extend(
        (field.capitalize(), semantic_diff_value(format_local(old_value), format_local(new_value)))
        for field, old_value, new_value, _offset in field_adjustments
    )
    offset_text = "; ".join(
        f"{field.capitalize()} {format_offset(offset)}"
        for field, _old_value, _new_value, offset in field_adjustments
    )
    rows.append(("Offset" if len(field_adjustments) == 1 else "Offsets", offset_text))
    panel("⚓ Nautical schedule adjusted", rows, kind="note")


def render_explicit_timing_order_warning(
    new: Mapping[str, Any],
    changed_fields: tuple[str, ...],
    *,
    format_offset: Callable[[timedelta], str],
    panel: Callable[..., Any],
) -> None:
    """Warn when an explicit timing edit leaves an invalid field ordering."""
    if not changed_fields:
        return

    def parsed(field: str) -> Any:
        value = new.timestamp(field)
        return value.value if value is not None else None

    due = parsed("due")
    scheduled = parsed("scheduled")
    wait = parsed("wait")
    issues: list[str] = []
    if due and scheduled and scheduled > due:
        issues.append(f"Scheduled is after Due by {format_offset(scheduled - due)}")
    if due and wait and wait > due:
        issues.append(f"Wait is after Due by {format_offset(wait - due)}")
    if scheduled and wait and wait > scheduled:
        issues.append(f"Wait is after Scheduled by {format_offset(wait - scheduled)}")
    if not issues:
        return

    if due:
        expected = "Due >= Scheduled >= Wait"
        action = "Keep Scheduled at/before Due and Wait at/before Scheduled."
    else:
        expected = "Scheduled >= Wait"
        action = "Keep Wait at or before Scheduled."
    rows: list[tuple[str, str]] = [
        ("Changed", ", ".join(field.capitalize() for field in changed_fields)),
        ("Expected", expected),
    ]
    rows.extend(("Problem", issue) for issue in issues)
    rows.append(("Action", action))
    panel("⚠ Nautical timing order", rows, kind="warning")


def _recurrence_update_label(field: str) -> str:
    return {
        "anchor": "Anchor",
        "anchor_file": "Anchor file",
        "omit": "Omit",
        "omit_file": "Omit file",
        "anchor_mode": "Mode",
        "bc": "Business calendar",
        "cp": "Period",
        "until": "Expiration",
        "chainMax": "Max links",
        "chainUntil": "Chain end point",
    }.get(field, field)


def _recurrence_display_value(
    field: str,
    value: str,
    *,
    parse_datetime: Callable[[Any], Any],
    format_local: Callable[[Any], str],
) -> str:
    if not value:
        return "-"
    if field in {"until", "chainUntil"}:
        parsed = parse_datetime(value)
        if parsed:
            return format_local(parsed)
    return value


def _recurrence_change_row(
    field: str,
    old_value: str,
    new_value: str,
    *,
    parse_datetime: Callable[[Any], Any],
    format_local: Callable[[Any], str],
) -> tuple[str, str]:
    label = _recurrence_update_label(field)
    old_text = _recurrence_display_value(field, old_value, parse_datetime=parse_datetime, format_local=format_local)
    new_text = _recurrence_display_value(field, new_value, parse_datetime=parse_datetime, format_local=format_local)
    if old_value and new_value:
        return "Changed", f"{label}: [dim]{old_text}[/] [cyan]→[/] [bold]{new_text}[/]"
    if new_value:
        return "Added", f"{label}: [bold]{new_text}[/]"
    return "Removed", f"{label}: [dim]{old_text}[/]"


def _recurrence_update_panel_rows(
    changes: list[tuple[str, str, str]],
    rows: list[tuple[str | None, str]],
    *,
    panel_mode: str,
    strip_markup: Callable[[str], str],
) -> list[tuple[str | None, str]]:
    if len(changes) > 1:
        recurrence_fields = {"anchor", "anchor_file", "cp", "anchor_mode", "omit", "omit_file", "bc"}
        limit_fields = {"chainMax", "chainUntil"}
        first_limit = next((idx for idx, (field, _old, _new) in enumerate(changes) if field in limit_fields), None)
        if first_limit is not None and any(field in recurrence_fields for field, _old, _new in changes):
            rows = list(rows)
            rows.insert(first_limit, (None, ""))

    mode = str(panel_mode or "rich").strip().lower()
    if mode == "quiet":
        mode = "text"
    if mode == "minimal":
        mode = "line"
    if mode in {"line", "text"}:
        change_rows = [(label, value) for label, value in rows if label in {"Added", "Changed", "Removed"}]
        if len(change_rows) > 1:
            summary = " · ".join(f"{label}: {strip_markup(str(value))}" for label, value in change_rows)
            rows = [("Changes", summary)] + [
                (label, value) for label, value in rows if label not in {"Added", "Changed", "Removed"}
            ]
    return rows


def render_recurrence_updated_panel(
    changes: list[tuple[str, str, str]],
    new: Mapping[str, Any],
    *,
    parse_datetime: Callable[[Any], Any],
    format_local: Callable[[Any], str],
    describe_native_until_carry: Callable[..., Any],
    to_local: Callable[[Any], Any],
    coerce_int: Callable[[Any, Any], int | None],
    describe_anchor: Callable[[str], str],
    resolve_omit_presets: Callable[[str], str],
    first_recurrence_target: Callable[[dict[str, Any], str], Any],
    panel_mode: str,
    strip_markup: Callable[[str], str],
    panel: Callable[..., Any],
) -> None:
    if not changes:
        return
    rows: list[tuple[str | None, str]] = [
        _recurrence_change_row(
            field,
            old_value,
            new_value,
            parse_datetime=parse_datetime,
            format_local=format_local,
        )
        for field, old_value, new_value in changes
    ]

    if any(field == "until" for field, _old, _new in changes):
        try:
            target_field = "due" if new.timestamp("due") else "scheduled" if new.timestamp("scheduled") else ""
            until_value = new.timestamp("until")
            target_value = new.timestamp(target_field) if target_field else None
            until_dt = until_value.value if until_value else None
            target_dt = target_value.value if target_value else None
            carry = describe_native_until_carry(until_dt, target_dt, to_local=to_local)
            if carry:
                rows.append(("Carry", carry))
        except Exception:
            pass

    if any(field in {"chainMax", "chainUntil"} for field, _old, _new in changes):
        max_link = coerce_int(new.get("chainMax"), 0)
        deadline_value = new.timestamp("chainUntil")
        deadline = deadline_value.value if deadline_value else None
        if max_link:
            rows.append(("Final link", f"#{max_link}"))
        if deadline and not any(field == "chainUntil" for field, _old, _new in changes):
            rows.append(("Chain end point", format_local(deadline)))
        if max_link and deadline:
            rows.append(("Effective", "Whichever boundary is reached first"))
        elif not max_link and not deadline:
            rows.append(("Chain limits", "None"))

    anchor_expr = str(new.get("anchor") or "").strip()
    if anchor_expr and any(field == "anchor" for field, _old, _new in changes):
        try:
            rows.append(("Natural", describe_anchor(anchor_expr)))
        except Exception:
            pass

    omit_expr = str(new.get("omit") or "").strip()
    if omit_expr and any(field == "omit" for field, _old, _new in changes):
        try:
            rows.append(("Except", describe_anchor(resolve_omit_presets(omit_expr))))
        except Exception:
            pass

    recurrence_fields = {"anchor", "anchor_file", "cp", "anchor_mode", "omit", "omit_file", "bc"}
    if any(field in recurrence_fields for field, _old, _new in changes):
        source = "anchor" if anchor_expr else "anchor_file" if str(new.get("anchor_file") or "").strip() else "cp"
        first = first_recurrence_target(new, source)
        if first:
            rows.append(("First next", format_local(first)))

    rows = _recurrence_update_panel_rows(
        changes,
        rows,
        panel_mode=panel_mode,
        strip_markup=strip_markup,
    )
    panel("⚓ Nautical recurrence updated", rows, kind="note")


def recurrence_enabled_rows(
    task: Mapping[str, Any],
    source: str,
    *,
    describe_anchor: Callable[[str], str],
    parse_cp_sequence_tokens: Callable[[str], list[dict[str, Any]] | None],
    first_recurrence_target: Callable[[dict[str, Any], str], Any],
    format_local: Callable[[Any], str],
) -> list[tuple[str, str]]:
    """Describe the recurrence added while promoting a plain task."""
    if source == "anchor":
        value = str(task.get("anchor") or "").strip()
        rows = [("Anchor", value)]
        try:
            natural = describe_anchor(value)
        except Exception:
            natural = None
        if natural:
            rows.append(("Natural", natural))
        mode = (task.get("anchor_mode") or "skip").strip().lower()
        mode_explanations = {
            "skip": "Skip missed anchors; use the next anchor after completion",
            "all": "Backfill every missed anchor in order",
            "flex": "Skip missed anchors and continue from the next available anchor",
        }
        rows.append(("Mode", f"{mode.upper()} — {mode_explanations.get(mode, mode)}"))
        first = first_recurrence_target(task, source)
        if first:
            rows.append(("First next", format_local(first)))
        return rows

    if source == "anchor_file":
        value = str(task.get("anchor_file") or "").strip()
        rows = [("Anchor file", value), ("Natural", f"Dates from {value.split('@', 1)[0]}")]
        mode = (task.get("anchor_mode") or "skip").strip().lower()
        rows.append(("Mode", f"{mode.upper()}"))
        first = first_recurrence_target(task, source)
        if first:
            rows.append(("First next", format_local(first)))
        return rows

    value = str(task.get("cp") or "").strip()
    rows = [("Period", value)]
    natural = None
    try:
        def duration_label(duration: Any) -> str:
            seconds = int(duration.total_seconds())
            if seconds % 86400 == 0:
                return f"{seconds // 86400}d"
            if seconds % 3600 == 0:
                return f"{seconds // 3600}h"
            if seconds % 60 == 0:
                return f"{seconds // 60}m"
            return f"{seconds}s"

        tokens = parse_cp_sequence_tokens(value) or []
        descriptions = []
        for token in tokens:
            if token.get("kind") == "rand":
                descriptions.append(f"random interval {token.get('raw') or value}")
            else:
                duration = token.get("duration")
                descriptions.append(duration_label(duration) if duration else str(token.get("raw") or value))
        if len(descriptions) == 1:
            natural = f"Every {descriptions[0]}"
        elif descriptions:
            natural = "Cycle through " + ", then ".join(descriptions)
    except Exception:
        natural = None
    if natural:
        rows.append(("Natural", natural))
    first = first_recurrence_target(task, source)
    if first:
        rows.append(("First next", format_local(first)))
    return rows


def format_chain_summary_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Arrange chain-finished rows into compact presentation sections."""
    groups = (
        {"Reason", "Chain", "Pattern", "Natural", "Period"},
        {"First due", "Last end", "Span"},
        {"Performance", "Avg lateness", "Median lateness", "Best early", "Worst late"},
        {"Chain cap", "Chain end point", "Chain limits"},
        {"History"},
    )
    grouped: list[list[tuple[str, str]]] = [[] for _ in groups]
    other: list[tuple[str, str]] = []
    for key, value in rows:
        for index, names in enumerate(groups):
            if key in names:
                grouped[index].append((key, value))
                break
        else:
            other.append((key, value))
    grouped[0].extend(other)
    out: list[tuple[str | None, str]] = []
    for group in grouped:
        if not group:
            continue
        if out:
            out.append((None, ""))
        out.extend(group)
    return out or rows


def format_next_anchor_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Arrange anchor next-link rows into compact presentation sections."""
    groups = (
        {"Pattern", "Natural", "Basis", "Sanitised"},
        {
            "Next", "Next Due", "Expiration", "Next expires", "Scheduled", "Wait",
            "Link status", "Links left", "Chain cap", "Chain end point",
        },
        {"Last occurrence"},
        {"Timeline"},
        {"Rand"},
    )
    grouped: list[list[tuple[str, str]]] = [[] for _ in groups]
    other: list[tuple[str, str]] = []
    for key, value in rows:
        for index, names in enumerate(groups):
            if key in names:
                grouped[index].append((key, value))
                break
        else:
            other.append((key, value))
    grouped[0].extend(other)
    out: list[tuple[str | None, str]] = []
    for group in grouped:
        if not group:
            continue
        if out:
            out.append((None, ""))
        out.extend(group)
    return out or rows


def format_next_cp_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Arrange CP next-link rows into compact presentation sections."""
    groups = (
        {"Period", "Basis"},
        {
            "Next", "Next Due", "Expiration", "Next expires", "Scheduled", "Wait",
            "Link status", "Links left", "Chain cap", "Chain end point",
        },
        {"Last occurrence"},
        {"Timeline"},
    )
    grouped: list[list[tuple[str, str]]] = [[] for _ in groups]
    other: list[tuple[str, str]] = []
    for key, value in rows:
        for index, names in enumerate(groups):
            if key in names:
                grouped[index].append((key, value))
                break
        else:
            other.append((key, value))
    grouped[0].extend(other)
    out: list[tuple[str | None, str]] = []
    for group in grouped:
        if not group:
            continue
        if out:
            out.append((None, ""))
        out.extend(group)
    return out or rows


def format_line_preview(
    link_no: int,
    task: dict,
    child_due_utc: Any,
    child_short: str,
    now_utc: Any,
    *,
    child_field: str = "due",
    cap_no: int | None = None,
    until_dt: Any = None,
    until_no: int | None = None,
    child_until_dt: Any = None,
    kind: str = "cp",
    minimal: bool = False,
    core: Any,
    format_local,
    on_time_delta,
    human_delta,
) -> str:
    """Render one compact completion preview line."""
    due_local = format_local(child_due_utc) if child_due_utc else "—"
    next_glyph = "⚓" if str(kind or "").lower() == "anchor" else "⛓"
    lead = f"#{link_no} ✓"
    if minimal:
        return " ".join((lead, f"next {next_glyph}", due_local)).strip()
    due_value = task.timestamp("due")
    end_value = task.timestamp("end")
    cur_due = due_value.value if due_value is not None else None
    cur_end = end_value.value if end_value is not None else None
    delta_text = core.strip_rich_markup(on_time_delta(cur_due, cur_end) or "").strip()
    if delta_text.startswith("(") and delta_text.endswith(")"):
        delta_text = delta_text[1:-1].strip()
    due_delta = human_delta(now_utc, child_due_utc, False)
    due_label = "scheduled" if child_field == "scheduled" else "due"
    if due_delta.startswith("in "):
        due_delta = due_label + " " + due_delta
    elif not due_delta.startswith("overdue by "):
        due_delta = due_label + " " + due_delta
    segments = [lead]
    if delta_text:
        segments.append(f"[dim]{delta_text}[/]")
    segments.extend((f"next {next_glyph}", due_local))
    if due_delta:
        segments.append(f"[dim]({due_delta})[/]")
    line = " · ".join(seg for seg in segments if seg).replace("✓ · ", "✓ ", 1)
    if child_until_dt:
        line += f" [magenta]· expires {format_local(child_until_dt)}[/]"
    cap_parts: list[str] = []
    if cap_no:
        cap_parts.extend((f"last link #{cap_no}", f"{max(0, cap_no - link_no)} left"))
    if until_dt:
        cap_parts.append(f"end point {format_local(until_dt)}")
    if cap_parts:
        line += f"[dim] · {' · '.join(cap_parts)}[/]"
    return line.strip()


def _pretty_basis_cp(task: dict, meta: dict, *, parse_cp_duration, parse_cp_sequence=None, cp_sequence_interval_for_link=None) -> str:
    if callable(cp_sequence_interval_for_link):
        td = cp_sequence_interval_for_link(
            task.get("cp") or "",
            int(task.get("link") or 1),
            str(task.get("chainID") or "").strip(),
        )
    elif callable(parse_cp_sequence):
        seq = parse_cp_sequence(task.get("cp") or "")
        step = int(meta.get("cp_sequence_step") or 1)
        if seq:
            td = seq[(max(1, step) - 1) % len(seq)]
        else:
            td = None
    else:
        td = parse_cp_duration(task.get("cp") or "")
    if not td:
        return "end + cp"
    secs = int(td.total_seconds())
    rem = secs % 86400
    if rem != 0:
        hrs, rems = divmod(rem, 3600)
        mins, _ = divmod(rems, 60)
        hint = []
        if hrs:
            hint.append(f"{hrs}h")
        if mins:
            hint.append(f"{mins}m")
        rem_s = " ".join(hint) if hint else f"{rem}s"
        return f"Exact end + cp (remainder {rem_s} vs 24h)"
    return "Preserve wall clock (period is multiple of 24h)"


def _pretty_basis_anchor(meta: Mapping[str, Any], task: Mapping[str, Any], *, fmt_dt_local) -> str:
    mode = (meta.get("mode") or "skip").lower()
    basis = meta.get("basis")
    missed = int(meta.get("missed_count") or 0)
    target_field = "scheduled" if meta.get("target_field") == "scheduled" else "due"
    typed_due = task.timestamp("due") or task.timestamp("scheduled")
    due0 = typed_due.value if typed_due is not None else None
    due_s = fmt_dt_local(due0) if due0 else f"(no {target_field})"
    if mode == "skip":
        return "SKIP — Next anchor after completion (multi-time: between slots counts as previous slot)"
    if mode == "flex":
        return f"FLEX — Skip missed up to now; next after completion ({missed} missed since {due_s})"
    if basis == "missed":
        return f"ALL — Backfilling first of {missed} missed anchor(s) since {due_s}"
    if basis == "after_due":
        return f"ALL (no missed) — Next anchor after original {target_field}"
    return "ALL — Next anchor after completion"


def _anchor_summary(task: dict) -> tuple[str, str]:
    anchor_expr = str(task.get("anchor") or "").strip()
    anchor_file = str(task.get("anchor_file") or "").strip()
    if anchor_expr and anchor_file:
        return "Sources", "anchor + anchor_file"
    if anchor_file:
        return "Anchor file", anchor_file
    return "Pattern", anchor_expr


def _anchor_pattern_row(core, expr: str) -> tuple[str, str]:
    try:
        preset_display = core.anchor_preset_display(expr)
    except Exception:
        preset_display = None
    if preset_display:
        return preset_display
    return "Pattern", expr


def _omit_pattern_row(core, expr: str) -> tuple[str, str]:
    try:
        preset_display = core.omit_preset_display(expr)
    except Exception:
        preset_display = None
    if preset_display:
        return preset_display
    return "Omit", expr


def _anchor_mode_tag(new: dict) -> str:
    return {
        "skip": "[cyan]SKIP[/]",
        "all": "[yellow]ALL[/]",
        "flex": "[magenta]FLEX[/]",
    }.get((new.get("anchor_mode") or "skip").lower(), "[cyan]SKIP[/]")


def _anchor_feedback_natural(core, task: dict, dnf) -> str:
    natural = core.describe_anchor_dnf(dnf, task) if dnf else ''
    omit_raw, omit_natural, _omit_warns, omit_file = _anchor_omit_summary(core, task)
    omit_parts = []
    if omit_raw:
        omit_parts.append(omit_natural or omit_raw)
    if omit_file:
        omit_parts.append(f"Dates from {omit_file.split('@', 1)[0]}")
    if omit_parts and (task.get('anchor_mode') or 'skip').lower() == 'skip':
        tail = '; skip missed anchors'
        if natural.endswith(tail):
            natural = natural[:-len(tail)]
        natural = natural.rstrip()
        return f"{natural}; skip {' and '.join(omit_parts)}" if natural else f"skip {' and '.join(omit_parts)}"
    return natural


def _anchor_omit_summary(core, task: dict) -> tuple[str | None, str | None, list[str], str | None]:
    omit_raw = str(task.get("omit") or "").strip()
    omit_file = str(task.get("omit_file") or "").strip() or None
    if not omit_raw:
        return None, None, [], omit_file
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        omit_expr = core.resolve_omit_presets(omit_raw)
        omit_norm = anchor_omit.normalize_omit_expr(omit_expr)
    except Exception:
        omit_norm = omit_raw
    try:
        natural = core.describe_anchor_expr(omit_norm)
    except Exception:
        natural = None
    try:
        _fatal, warns = core.lint_anchor_expr(omit_norm)
    except Exception:
        warns = []
    return omit_raw, natural, list(warns or []), omit_file


def _append_wait_sched_feedback_rows(fb: list[tuple[str, object]], *, debug_wait_sched: bool, last_wait_sched_debug) -> None:
    if not (debug_wait_sched and last_wait_sched_debug):
        return
    for field in ("scheduled", "wait"):
        data = last_wait_sched_debug.get(field)
        if not data:
            continue
        if data.get("ok"):
            fb.append(
                (
                    f"{field} carry",
                    f"Δ {data.get('delta')}  parent {data.get('parent_val')} vs {data.get('parent_anchor')}  →  child {data.get('child_val')}",
                )
            )
        else:
            fb.append(
                (
                    f"{field} carry",
                    f"[yellow]skip[/] ({data.get('reason')})  parent {data.get('parent_val')} vs {data.get('parent_anchor')}",
                )
            )


def _append_sanitised_fields_row(fb: list[tuple[str, object]], stripped_attrs: list[str]) -> None:
    if stripped_attrs:
        fb.append(("Sanitised", f"Removed unknown fields: {', '.join(sorted(stripped_attrs))}"))


def _append_integrity_warnings_row(fb: list[tuple[str, object]], integrity_warnings: list[str] | None) -> None:
    if not integrity_warnings:
        return
    warn_list = integrity_warnings[:4]
    if len(integrity_warnings) > 4:
        warn_list.append(f"...and {len(integrity_warnings) - 4} more")
    fb.append(("Integrity", "\n".join(warn_list)))


def _append_link_status_rows(
    fb: list[tuple[str, object]],
    cap_no: int | None,
    base_no: int,
    *,
    second_to_last_text: str,
) -> None:
    if not cap_no:
        return
    if base_no >= cap_no:
        fb.append(("Link status", "[bold red]This was the last link[/]"))
    elif base_no == cap_no - 1:
        fb.append(("Link status", second_to_last_text))
    fb.append(("Links left", str(max(0, cap_no - base_no))))


def _effective_last_occurrence(finals: list[tuple[str, Any]]):
    candidates = [when for _label, when in finals if when is not None]
    return min(candidates) if candidates else None


def _append_final_rows(
    fb: list[tuple[str, object]],
    finals: list[tuple[str, object]],
    now_utc,
    *,
    fmt_dt_local,
    human_delta,
) -> None:
    last = _effective_last_occurrence(finals)
    if last is None:
        return
    fb.append(("Last occurrence", f"{fmt_dt_local(last)}  ({human_delta(now_utc, last, True)})"))


def _append_chain_boundary_rows(fb: list[tuple[str, object]], task: dict, until_dt, *, core) -> None:
    chain_max = core.coerce_int(task.get("chainMax"), 0)
    if chain_max:
        fb.append(("Chain cap", f"#{chain_max}"))
    if until_dt:
        fb.append(("Chain end point", core.fmt_dt_local(until_dt)))


def _append_lifecycle_result_row(fb: list[tuple[str, object]], lifecycle_result) -> None:
    """Expose the mutation outcome without making panels part of orchestration."""
    state = str(getattr(lifecycle_result, "state", "") or "").strip().lower()
    if not state:
        return
    labels = {
        "applied": "[green]Applied now[/]",
        "queued": "[yellow]Queued for on-exit[/]",
        "terminal": "[cyan]Chain complete[/]",
        "retryable": "[red]Retryable; no successor finalized[/]",
        "manual_review": "[yellow]Manual review required[/]",
    }
    value = labels.get(state, f"[yellow]{state}[/]")
    child_short = str(getattr(lifecycle_result, "child_short", "") or "").strip()
    if child_short and state in {"applied", "queued"}:
        value += f" · child {child_short}"
    intent_id = str(getattr(lifecycle_result, "spawn_intent_id", "") or "").strip()
    if intent_id and state == "queued":
        value += f" · intent {intent_id}"
    reason = str(getattr(lifecycle_result, "reason", "") or "").strip()
    if reason and state in {"terminal", "retryable", "manual_review"}:
        value += f" · {reason}"
    fb.append(("Result", value))


def _lifecycle_result_label(lifecycle_result) -> str:
    state = str(getattr(lifecycle_result, "state", "") or "").strip().lower()
    return {
        "applied": "Applied now",
        "queued": "Queued for on-exit",
        "terminal": "Chain complete",
        "retryable": "Retryable",
        "manual_review": "Manual review required",
    }.get(state, state.replace("_", " ").title() if state else "")


def _child_expiration(child):
    typed_until = child.timestamp("until")
    return typed_until.value if typed_until is not None else None


def _append_next_expiration_row(
    fb: list[tuple[str, object]],
    child: Mapping[str, Any],
    child_due,
    *,
    core,
    target_field: str = "due",
) -> None:
    expires = _child_expiration(child)
    if expires is None:
        return
    try:
        add_validation = core._import_sibling("add_validation")
        carry = add_validation.describe_native_until_carry(
            expires,
            child_due,
            to_local=core.to_local,
        )
    except Exception:
        carry = None
    if carry:
        fb.append(("Expiration", carry))
    delta = core.humanize_delta(child_due, expires, use_months_days=False)
    if delta.startswith("in "):
        delta = delta[3:]
    basis = "scheduled" if target_field == "scheduled" else "due"
    fb.append(("Next expires", f"{core.fmt_dt_local(expires)}  ({delta} after {basis})"))


def _display_mode_name(core) -> str:
    mode = str(getattr(core, "PANEL_MODE", "rich") or "rich").strip().lower()
    if mode == "quiet":
        return "text"
    return mode


def _rows_are_notable(rows: list[tuple[str, object]]) -> bool:
    notable_labels = {"integrity", "warning", "error", "link status", "links left", "sanitised", "intent"}
    for k, v in rows:
        if k is None:
            continue
        lk = str(k).strip().lower()
        if lk in notable_labels or lk == "last occurrence":
            return True
        if lk == "basis":
            return True
        if lk == "analytics" and str(v or "").strip():
            return True
    return False


def _build_text_feedback(
    core,
    *,
    kind: str,
    parent_short: str,
    next_no: int,
    child_short: str,
    summary: str | None,
    preview_line: str,
    cap_no: int | None,
    base_no: int,
    until_dt,
    child_due=None,
    child_expires=None,
    expiration_basis: str = "due",
    last_occurrence=None,
    lifecycle_result=None,
    extra_line: str | None = None,
) -> str:
    text = core.strip_rich_markup(preview_line or "")
    parts = [part.strip() for part in text.split("·") if part and part.strip()]
    lead = parts[0] if parts else ""
    due_part = parts[2] if len(parts) >= 3 else ""
    status_tail = lead.split(" ", 1)[1].strip() if " " in lead else ""
    status_tail = status_tail.replace(" next ⚓︎", "").replace(" next ⚓", "").replace(" next ⛓", "").strip()

    line1 = f"[bold white]{parent_short}[/]"
    if status_tail:
        status_tokens = status_tail.split(" ", 1)
        first = status_tokens[0]
        rest = status_tokens[1].strip() if len(status_tokens) > 1 else ""
        if first:
            line1 += f" [green]{first}[/]"
        if rest:
            line1 += f" [dim]{rest}[/]"

    due_part = due_part.replace("(due in ", "in ").replace("(due overdue by ", "overdue by ").replace("(", "").replace(")", "").strip()
    accent = "cyan" if str(kind or "").lower() == "anchor" else "yellow"
    due_style = "red" if due_part.startswith("overdue by ") else "bright_white"
    line2 = f"[bold {accent}]Next[/] [{accent}]" + ("⚓︎" if str(kind or "").lower() == "anchor" else "⛓") + f"[/] [bold]{'#' + str(next_no)}[/] [bold white]{child_short}[/]"
    if due_part:
        line2 += f" [dim]→[/] [{due_style}]{due_part}[/]"

    lines = [line1, line2]
    if child_expires:
        try:
            add_validation = core._import_sibling("add_validation")
            carry = add_validation.describe_native_until_carry(
                child_expires,
                child_due,
                to_local=core.to_local,
            )
        except Exception:
            carry = None
        if carry:
            lines.append(f"[bold magenta]Expiration:[/] [white]{carry}[/]")
        expires_delta = core.humanize_delta(child_due, child_expires, use_months_days=False)
        if expires_delta.startswith("in "):
            expires_delta = expires_delta[3:]
        lines.append(
            f"[bold magenta]Next expires:[/] [white]{core.fmt_dt_local(child_expires)}[/]"
            f" [dim]({expires_delta} after {expiration_basis})[/]"
        )
    if summary and str(summary).strip():
        if str(kind or "").lower() == "anchor":
            label = "Sources" if str(summary).strip() == "anchor + anchor_file" else "Pattern"
            summary_text = f"[bold cyan]{label}:[/] [white]{summary.split(':', 1)[1].strip() if ':' in summary else summary}[/]"
        else:
            summary_text = f"[bold yellow]Period:[/] [white]{summary.split(':', 1)[1].strip() if ':' in summary else summary}[/]"
        lines.append(summary_text)
    if extra_line and str(extra_line).strip():
        lines.append(extra_line)
    result_label = _lifecycle_result_label(lifecycle_result)
    if result_label:
        result_reason = str(getattr(lifecycle_result, "reason", "") or "").strip()
        suffix = f": {result_reason}" if result_reason and result_label in {"Retryable", "Manual review required"} else ""
        lines.append(f"[bold cyan]Result:[/] [white]{result_label}{suffix}[/]")

    limit_parts = []
    if cap_no:
        limit_parts.append(f"[yellow]#{cap_no}[/]")
        limit_parts.append(f"[dim]{max(0, cap_no - base_no)} left[/]")
    if limit_parts:
        lines.append("[bold yellow]Last link:[/] " + " [dim]·[/] ".join(limit_parts))
    if until_dt:
        lines.append(f"[bold yellow]Chain end point:[/] [white]{core.fmt_dt_local(until_dt)}[/]")
    if last_occurrence:
        lines.append(f"[bold magenta]Last occurrence:[/] [white]{core.fmt_dt_local(last_occurrence)}[/]")
    return "\n".join(line for line in lines if line)


def _compact_feedback_rows(rows: list[tuple[str, object]], *, include_timeline: bool = True) -> list[tuple[str, object]]:
    keep_labels = {
        "pattern",
        "period",
        "next",
        "natural",
        "basis",
        "root",
        "link status",
        "links left",
        "chain cap",
        "chain end point",
        "last occurrence",
        "expiration",
        "next expires",
        "integrity",
        "timeline",
        "sanitised",
        "warning",
        "error",
        "intent",
        "result",
    }
    out: list[tuple[str, object]] = []
    for k, v in rows:
        if k is None:
            continue
        lk = str(k).strip().lower()
        if lk == "timeline" and not include_timeline:
            continue
        if lk in keep_labels:
            out.append((k, v))
    return out


def render_anchor_completion_feedback(
    *,
    feedback,
    services,
) -> None:
    core = services.core
    debug_wait_sched = services.debug_wait_sched
    last_wait_sched_debug = services.last_wait_sched_debug
    diag_enabled = services.diag_enabled
    format_root_and_age = services.format_root_and_age
    append_next_wait_sched_rows = services.append_next_wait_sched_rows
    timeline_lines = services.timeline_lines
    show_timeline_gaps = services.show_timeline_gaps
    root_uuid_from = services.root_uuid_from
    short = services.short
    format_next_anchor_rows = services.format_next_anchor_rows
    format_line_preview = services.format_line_preview
    panel_line = services.panel_line
    text_line = services.text_line
    panel = services.panel
    chain_color_per_chain = services.chain_color_per_chain
    chain_colour_for_task = services.chain_colour_for_task
    strip_quotes = services.strip_quotes
    human_delta = services.human_delta
    anchor_label, anchor_value = _anchor_summary(feedback.new)
    pattern_label, pattern_value = _anchor_pattern_row(core, str(feedback.new.get("anchor") or "").strip())
    if anchor_label == "Pattern":
        anchor_label, anchor_value = pattern_label, pattern_value
    expr_str = strip_quotes(anchor_value)
    omit_raw, omit_natural, omit_warns, omit_file = _anchor_omit_summary(core, feedback.new)
    mode_tag = _anchor_mode_tag(feedback.new)
    title = f"⚓︎ Next anchor  #{feedback.next_no}  {feedback.parent_short} → {feedback.child_short}"
    mode = _display_mode_name(core)
    if mode in {"line", "minimal"}:
        line = format_line_preview(
            feedback.base_no,
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.now_utc,
            child_field=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            child_until_dt=_child_expiration(feedback.child),
            kind="anchor",
            minimal=(mode == "minimal"),
        )
        result_label = _lifecycle_result_label(feedback.lifecycle_result)
        if result_label:
            line = f"{line} · {result_label}"
        title_style = chain_colour_for_task(feedback.new, "anchor") if chain_color_per_chain else None
        panel_line(title, line, kind="preview_anchor", border_style=title_style, title_style=title_style, markup_body=True)
        return
    if mode == "text":
        line = format_line_preview(
            feedback.base_no,
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.now_utc,
            child_field=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            child_until_dt=_child_expiration(feedback.child),
            kind="anchor",
            minimal=False,
        )
        text_line(
            _build_text_feedback(
                core,
                kind="anchor",
                parent_short=feedback.parent_short,
                next_no=feedback.next_no,
                child_short=feedback.child_short,
                summary=f"{anchor_label}: {expr_str}  {mode_tag}",
                preview_line=line,
                cap_no=feedback.cap_no,
                base_no=feedback.base_no,
                until_dt=feedback.until_dt,
                child_due=feedback.child_due,
                child_expires=_child_expiration(feedback.child),
                expiration_basis=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
                last_occurrence=_effective_last_occurrence(feedback.finals),
                lifecycle_result=feedback.lifecycle_result,
                extra_line=(f"[bold cyan]Except:[/] [white]{omit_natural or omit_raw}[/]" if omit_raw else (f"[bold cyan]Omit file:[/] [white]{omit_file}[/]" if omit_file else None)),
            ),
            kind="preview_anchor",
            markup_body=True,
        )
        return

    fb: list[tuple[str, Any]] = []
    _append_lifecycle_result_row(fb, feedback.lifecycle_result)
    fb.append((anchor_label, f"{expr_str}  {mode_tag}"))
    if omit_raw:
        fb.append(_omit_pattern_row(core, omit_raw))
        if omit_natural:
            fb.append(("Except", omit_natural))
        for warn in omit_warns:
            fb.append(("Warning", warn))
    if omit_file:
        fb.append(("Omit file", omit_file))
    delta = core.humanize_delta(feedback.now_utc, feedback.child_due, use_months_days=core.expr_has_m_or_y(feedback.dnf))
    fb.append(("Next", f"#{feedback.next_no} → {core.fmt_dt_local(feedback.child_due)}  ({delta})"))
    _append_next_expiration_row(
        fb,
        feedback.child,
        feedback.child_due,
        core=core,
        target_field=feedback.meta.get("target_field") or "due",
    )
    if anchor_label == "Sources":
        file_expr = str(feedback.new.get("anchor_file") or "").strip()
        natural_expr = _anchor_feedback_natural(core, feedback.new, feedback.dnf)
        fb.append((pattern_label, pattern_value))
        fb.append(("Anchor file", file_expr))
        if natural_expr:
            fb.append(("Natural", natural_expr))
        else:
            fb.append(("Natural", f"Dates from {file_expr.split('@', 1)[0]}"))
    elif feedback.dnf:
        fb.append(("Natural", _anchor_feedback_natural(core, feedback.new, feedback.dnf)))
    elif anchor_label == "Anchor file":
        fb.append(("Natural", f"Dates from {expr_str.split('@', 1)[0]}"))
    basis_text = _pretty_basis_anchor(feedback.meta, feedback.new, fmt_dt_local=core.fmt_dt_local)
    if basis_text != "SKIP — Next anchor after completion (multi-time: between slots counts as previous slot)":
        fb.append(("Basis", basis_text))
    fb.append(("Root", format_root_and_age(feedback.new, feedback.now_utc)))

    _append_wait_sched_feedback_rows(fb, debug_wait_sched=debug_wait_sched, last_wait_sched_debug=last_wait_sched_debug)
    _append_sanitised_fields_row(fb, feedback.stripped_attrs)
    if core.SHOW_ANALYTICS and feedback.analytics_advice:
        fb.append(("Analytics", feedback.analytics_advice))
    _append_integrity_warnings_row(fb, feedback.integrity_warnings)
    append_next_wait_sched_rows(
        fb,
        feedback.child,
        feedback.child_due,
        anchor_field=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
    )

    _append_chain_boundary_rows(fb, feedback.new, feedback.until_dt, core=core)
    _append_link_status_rows(
        fb,
        feedback.cap_no,
        feedback.base_no,
        second_to_last_text="[yellow]This was the second-to-last link. Next is last.[/]",
    )
    _append_final_rows(fb, feedback.finals, feedback.now_utc, fmt_dt_local=core.fmt_dt_local, human_delta=human_delta)
    if feedback.deferred_spawn and diag_enabled and feedback.spawn_intent_id:
        fb.append(("Intent", feedback.spawn_intent_id))

    if mode not in {"line", "minimal", "text"}:
        tl = timeline_lines(
            "anchor",
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.dnf,
            next_count=3,
            cap_no=feedback.cap_no,
            cur_no=feedback.base_no,
            show_gaps=show_timeline_gaps,
        )
        if tl:
            fb.append(("Timeline", "\n".join(tl)))
    if feedback.dnf and "rand" in expr_str.lower():
        fb.append(("Rand", f"[dim]Deterministic picks seeded by root {short(root_uuid_from(feedback.new))}[/]"))

    fb = format_next_anchor_rows(fb)
    if mode == "compact":
        fb = _compact_feedback_rows(fb, include_timeline=True)
    if chain_color_per_chain:
        chain_colour = chain_colour_for_task(feedback.new, "anchor")
        panel(
            title,
            fb,
            kind="preview_anchor",
            border_style=chain_colour,
            title_style=chain_colour,
        )
        return
    panel(title, fb, kind="preview_anchor")


def render_cp_completion_feedback(
    *,
    feedback,
    services,
) -> None:
    core = services.core
    diag_enabled = services.diag_enabled
    format_root_and_age = services.format_root_and_age
    append_next_wait_sched_rows = services.append_next_wait_sched_rows
    timeline_lines = services.timeline_lines
    show_timeline_gaps = services.show_timeline_gaps
    format_next_cp_rows = services.format_next_cp_rows
    format_line_preview = services.format_line_preview
    panel_line = services.panel_line
    text_line = services.text_line
    panel = services.panel
    chain_color_per_chain = services.chain_color_per_chain
    chain_colour_for_task = services.chain_colour_for_task
    human_delta = services.human_delta
    title = f"⛓ Next link  #{feedback.next_no}  {feedback.parent_short} → {feedback.child_short}"
    mode = _display_mode_name(core)
    if mode in {"line", "minimal"}:
        line = format_line_preview(
            feedback.base_no,
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.now_utc,
            child_field=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            child_until_dt=_child_expiration(feedback.child),
            kind="cp",
            minimal=(mode == "minimal"),
        )
        result_label = _lifecycle_result_label(feedback.lifecycle_result)
        if result_label:
            line = f"{line} · {result_label}"
        title_style = chain_colour_for_task(feedback.new, "cp") if chain_color_per_chain else None
        panel_line(title, line, kind="preview_cp", border_style=title_style, title_style=title_style, markup_body=True)
        return
    if mode == "text":
        line = format_line_preview(
            feedback.base_no,
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.now_utc,
            child_field=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            child_until_dt=_child_expiration(feedback.child),
            kind="cp",
            minimal=False,
        )
        text_line(
            _build_text_feedback(
                core,
                kind="cp",
                parent_short=feedback.parent_short,
                next_no=feedback.next_no,
                child_short=feedback.child_short,
                summary=f"Period: {feedback.new.get('cp')}",
                preview_line=line,
                cap_no=feedback.cap_no,
                base_no=feedback.base_no,
                until_dt=feedback.until_dt,
                child_due=feedback.child_due,
                child_expires=_child_expiration(feedback.child),
                expiration_basis=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
                last_occurrence=_effective_last_occurrence(feedback.finals),
                lifecycle_result=feedback.lifecycle_result,
            ),
            kind="preview_cp",
            markup_body=True,
        )
        return

    fb: list[tuple[str, Any]] = []
    _append_lifecycle_result_row(fb, feedback.lifecycle_result)
    delta = core.humanize_delta(feedback.now_utc, feedback.child_due, use_months_days=False)
    fb.append(("Period", feedback.new.get("cp")))
    if feedback.meta.get("cp_sequence_len"):
        step = int(feedback.meta.get("cp_sequence_step") or 1)
        cp_tokens = [p.strip() for p in str(feedback.new.get("cp") or "").split(",")]
        step_token = cp_tokens[step - 1] if 0 <= step - 1 < len(cp_tokens) else ""
        token_index = max(0, step - 1)
        try:
            tokens = core.parse_cp_sequence_tokens(feedback.new.get("cp") or "")
            if tokens and 0 <= token_index < len(tokens) and tokens[token_index].get("kind") == "rand":
                td = core.cp_sequence_interval_for_token(
                    tokens[token_index],
                    cp=feedback.new.get("cp") or "",
                    link_no=int(feedback.new.get("link") or 1),
                    token_index=token_index,
                    chain_id=str(feedback.new.get("chainID") or "").strip(),
                )
                if td:
                    step_token = _format_td_short(td)
        except Exception:
            pass
        suffix = f" ({step_token})" if step_token else ""
        fb.append(("Step", f"{step}/{feedback.meta.get('cp_sequence_len')}{suffix}"))
    fb.append(("Next", f"#{feedback.next_no} → {core.fmt_dt_local(feedback.child_due)}  ({delta})"))
    _append_next_expiration_row(
        fb,
        feedback.child,
        feedback.child_due,
        core=core,
        target_field=feedback.meta.get("target_field") or "due",
    )
    basis_text = _pretty_basis_cp(
        feedback.new,
        feedback.meta,
        parse_cp_duration=core.parse_cp_duration,
        parse_cp_sequence=getattr(core, "parse_cp_sequence", None),
        cp_sequence_interval_for_link=getattr(core, "cp_sequence_interval_for_link", None),
    )
    if basis_text != "Preserve wall clock (period is multiple of 24h)":
        fb.append(("Basis", basis_text))
    fb.append(("Root", format_root_and_age(feedback.new, feedback.now_utc)))
    if core.SHOW_ANALYTICS and feedback.analytics_advice:
        fb.append(("Analytics", feedback.analytics_advice))
    _append_integrity_warnings_row(fb, feedback.integrity_warnings)
    append_next_wait_sched_rows(
        fb,
        feedback.child,
        feedback.child_due,
        anchor_field=("scheduled" if feedback.meta.get("target_field") == "scheduled" else "due"),
    )

    _append_chain_boundary_rows(fb, feedback.new, feedback.until_dt, core=core)
    if feedback.cap_no:
        _append_link_status_rows(
            fb,
            feedback.cap_no,
            feedback.base_no,
            second_to_last_text="[yellow]Next link is the last in the chain.[/]",
        )

    _append_final_rows(fb, feedback.finals, feedback.now_utc, fmt_dt_local=core.fmt_dt_local, human_delta=human_delta)

    if feedback.deferred_spawn and diag_enabled and feedback.spawn_intent_id:
        fb.append(("Intent", feedback.spawn_intent_id))

    if mode not in {"line", "minimal", "text"}:
        tl = timeline_lines(
            "cp",
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            None,
            next_count=3,
            cap_no=feedback.cap_no,
            cur_no=feedback.base_no,
            show_gaps=show_timeline_gaps,
        )
        if tl:
            fb.append(("Timeline", "\n".join(tl)))

    fb = format_next_cp_rows(fb)
    if mode == "compact":
        fb = _compact_feedback_rows(fb, include_timeline=True)
    if chain_color_per_chain:
        chain_colour = chain_colour_for_task(feedback.new, "cp")
        panel(
            title,
            fb,
            kind="preview_cp",
            border_style=chain_colour,
            title_style=chain_colour,
        )
    else:
        panel(title, fb, kind="preview_cp")


def orchestrate_anchor_completion_feedback(
    *,
    new: Mapping[str, Any],
    child: Mapping[str, Any],
    child_due,
    child_short: str,
    next_no: int,
    parent_short: str,
    cap_no: int | None,
    finals: list[tuple[str, object]],
    now_utc,
    until_dt,
    until_cap_no: int | None,
    dnf,
    meta: dict,
    stripped_attrs: list[str],
    deferred_spawn: bool,
    spawn_intent_id: str | None,
    lifecycle_result=None,
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
    core: Any,
    panel,
    calendar_feedback,
    panel_diagnostics,
    modify_models,
    modify_runtime,
    build_runtime_services,
) -> None:
    """Assemble anchor feedback state and hand it to the feedback renderer."""
    if lifecycle_result is None:
        lifecycle_result = modify_models.CompletionLifecycleResult(
            state="queued" if deferred_spawn else "applied",
            child_short=child_short,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
        )
    calendar_feedback.render_business_calendar_displacement(
        new,
        child_due,
        core=core,
        panel=panel,
    )
    panel_warnings = panel_diagnostics.panel_warnings(core, modify_models.TaskView.from_mapping(new))
    if panel_warnings:
        integrity_warnings = list(integrity_warnings or [])
        integrity_warnings.extend(panel_warnings)
    feedback = modify_models.AnchorCompletionFeedbackModel(
        new=new,
        child=child,
        child_due=child_due,
        child_short=child_short,
        next_no=next_no,
        parent_short=parent_short,
        cap_no=cap_no,
        finals=finals,
        now_utc=now_utc,
        until_dt=until_dt,
        until_cap_no=until_cap_no,
        dnf=dnf,
        meta=meta,
        stripped_attrs=stripped_attrs,
        deferred_spawn=deferred_spawn,
        spawn_intent_id=spawn_intent_id,
        lifecycle_result=lifecycle_result,
        chain_by_short=chain_by_short,
        analytics_advice=analytics_advice,
        integrity_warnings=integrity_warnings,
        base_no=base_no,
    )
    services = modify_runtime.build_anchor_feedback_services(build_runtime_services())
    render_anchor_completion_feedback(feedback=feedback, services=services)


def orchestrate_cp_completion_feedback(
    *,
    new: Mapping[str, Any],
    child: Mapping[str, Any],
    child_due,
    child_short: str,
    next_no: int,
    parent_short: str,
    cap_no: int | None,
    finals: list[tuple[str, object]],
    now_utc,
    until_dt,
    until_cap_no: int | None,
    meta: dict,
    deferred_spawn: bool,
    spawn_intent_id: str | None,
    lifecycle_result=None,
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
    core: Any,
    panel_diagnostics,
    modify_models,
    modify_runtime,
    build_runtime_services,
) -> None:
    """Assemble CP feedback state and hand it to the feedback renderer."""
    if lifecycle_result is None:
        lifecycle_result = modify_models.CompletionLifecycleResult(
            state="queued" if deferred_spawn else "applied",
            child_short=child_short,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
        )
    panel_warnings = panel_diagnostics.panel_warnings(
        core,
        modify_models.TaskView.from_mapping(new),
        include_files=False,
    )
    if panel_warnings:
        integrity_warnings = list(integrity_warnings or [])
        integrity_warnings.extend(panel_warnings)
    feedback = modify_models.CpCompletionFeedbackModel(
        new=new,
        child=child,
        child_due=child_due,
        child_short=child_short,
        next_no=next_no,
        parent_short=parent_short,
        cap_no=cap_no,
        finals=finals,
        now_utc=now_utc,
        until_dt=until_dt,
        until_cap_no=until_cap_no,
        meta=meta,
        deferred_spawn=deferred_spawn,
        spawn_intent_id=spawn_intent_id,
        lifecycle_result=lifecycle_result,
        chain_by_short=chain_by_short,
        analytics_advice=analytics_advice,
        integrity_warnings=integrity_warnings,
        base_no=base_no,
    )
    services = modify_runtime.build_cp_feedback_services(build_runtime_services())
    render_cp_completion_feedback(feedback=feedback, services=services)
