from __future__ import annotations


def _pretty_basis_cp(task: dict, meta: dict, *, parse_cp_duration) -> str:
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


def _pretty_basis_anchor(meta: dict, task: dict, *, parse_dt_any, fmt_dt_local) -> str:
    mode = (meta.get("mode") or "skip").lower()
    basis = meta.get("basis")
    missed = int(meta.get("missed_count") or 0)
    due0 = parse_dt_any(task.get("due"))
    due_s = fmt_dt_local(due0) if due0 else "(no due)"
    if mode == "skip":
        return "SKIP — Next anchor after completion (multi-time: between slots counts as previous slot)"
    if mode == "flex":
        return f"FLEX — Skip missed up to now; next after completion ({missed} missed since {due_s})"
    if basis == "missed":
        return f"ALL — Backfilling first of {missed} missed anchor(s) since {due_s}"
    if basis == "after_due":
        return "ALL (no missed) — Next anchor after original due"
    return "ALL — Next anchor after completion"


def _resolve_child_id(child_short: str, deferred_spawn: bool, chain_by_short: dict | None, *, export_uuid_short_cached) -> str:
    child_id = ""
    if not deferred_spawn and chain_by_short:
        child_id = str(chain_by_short.get(child_short, {}).get("id", "") or "")
    if not deferred_spawn and not child_id:
        child_obj = export_uuid_short_cached(child_short)
        child_id = child_obj.get("id", "") if child_obj else ""
    return child_id


def _anchor_mode_tag(new: dict) -> str:
    return {
        "skip": "[cyan]SKIP[/]",
        "all": "[yellow]ALL[/]",
        "flex": "[magenta]FLEX[/]",
    }.get((new.get("anchor_mode") or "skip").lower(), "[cyan]SKIP[/]")


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
                    f"Δ {data.get('delta')}  parent {data.get('parent_val')} vs {data.get('parent_due')}  →  child {data.get('child_val')}",
                )
            )
        else:
            fb.append(
                (
                    f"{field} carry",
                    f"[yellow]skip[/] ({data.get('reason')})  parent {data.get('parent_val')} vs {data.get('parent_due')}",
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
    fb.append(("Links left", f"{max(0, cap_no - base_no)} left (cap #{cap_no})"))


def _append_final_rows(
    fb: list[tuple[str, object]],
    finals: list[tuple[str, object]],
    now_utc,
    *,
    fmt_dt_local,
    human_delta,
) -> None:
    for label, when in finals:
        fb.append((f"Final ({label})", f"{fmt_dt_local(when)}  ({human_delta(now_utc, when, True)})"))


def render_anchor_completion_feedback(
    *,
    new: dict,
    child: dict,
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
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
    core,
    debug_wait_sched: bool,
    last_wait_sched_debug,
    diag_enabled: bool,
    format_root_and_age,
    append_next_wait_sched_rows,
    timeline_lines,
    show_timeline_gaps: bool,
    root_uuid_from,
    short,
    format_next_anchor_rows,
    format_line_preview,
    panel_line,
    panel,
    chain_color_per_chain: bool,
    chain_colour_for_task,
    strip_quotes,
    human_delta,
) -> None:
    fb = []
    anchor_raw = (new.get("anchor") or "").strip()
    expr_str = strip_quotes(anchor_raw)
    mode_tag = _anchor_mode_tag(new)
    fb.append(("Pattern", f"{expr_str}  {mode_tag}"))
    fb.append(("Natural", core.describe_anchor_dnf(dnf, new)))
    fb.append(("Basis", _pretty_basis_anchor(meta, new, parse_dt_any=core.parse_dt_any, fmt_dt_local=core.fmt_dt_local)))
    fb.append(("Root", format_root_and_age(new, now_utc)))

    _append_wait_sched_feedback_rows(fb, debug_wait_sched=debug_wait_sched, last_wait_sched_debug=last_wait_sched_debug)
    _append_sanitised_fields_row(fb, stripped_attrs)

    delta = core.humanize_delta(now_utc, child_due, use_months_days=core.expr_has_m_or_y(dnf))
    fb.append(("Next Due", f"{core.fmt_dt_local(child_due)}  ({delta})"))
    if analytics_advice:
        fb.append(("Analytics", analytics_advice))
    _append_integrity_warnings_row(fb, integrity_warnings)
    append_next_wait_sched_rows(fb, child, child_due)

    _append_link_status_rows(
        fb,
        cap_no,
        base_no,
        second_to_last_text="[yellow]This was the second-to-last link[/]",
    )
    _append_final_rows(fb, finals, now_utc, fmt_dt_local=core.fmt_dt_local, human_delta=human_delta)
    if deferred_spawn and diag_enabled and spawn_intent_id:
        fb.append(("Intent", spawn_intent_id))

    title = f"⚓︎ Next anchor  #{next_no}  {parent_short} → {child_short}"
    tl = timeline_lines(
        "anchor",
        new,
        child_due,
        child_short,
        dnf,
        next_count=3,
        cap_no=cap_no,
        cur_no=base_no,
        show_gaps=show_timeline_gaps,
    )
    if tl:
        fb.append(("Timeline", "\n".join(tl)))
    if "rand" in expr_str.lower():
        fb.append(("Rand", f"[dim]Deterministic picks seeded by root {short(root_uuid_from(new))}[/]"))

    fb = format_next_anchor_rows(fb)

    if (core.PANEL_MODE or "").strip().lower() == "line":
        line = format_line_preview(
            base_no,
            new,
            child_due,
            child_short,
            now_utc,
            cap_no=cap_no,
            until_dt=until_dt,
            until_no=until_cap_no,
            kind="anchor",
        )
        title_style = chain_colour_for_task(new, "anchor") if chain_color_per_chain else None
        panel_line(title, line, kind="preview_anchor", border_style=title_style, title_style=title_style, markup_body=True)
        return
    if chain_color_per_chain:
        chain_colour = chain_colour_for_task(new, "anchor")
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
    new: dict,
    child: dict,
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
    chain_by_short: dict | None,
    analytics_advice: str | None,
    integrity_warnings: list[str] | None,
    base_no: int,
    core,
    diag_enabled: bool,
    format_root_and_age,
    append_next_wait_sched_rows,
    timeline_lines,
    show_timeline_gaps: bool,
    format_next_cp_rows,
    format_line_preview,
    panel_line,
    panel,
    chain_color_per_chain: bool,
    chain_colour_for_task,
    human_delta,
    export_uuid_short_cached,
) -> None:
    fb = []
    delta = core.humanize_delta(now_utc, child_due, use_months_days=False)
    fb.append(("Period", new.get("cp")))
    fb.append(("Basis", _pretty_basis_cp(new, meta, parse_cp_duration=core.parse_cp_duration)))
    fb.append(("Root", format_root_and_age(new, now_utc)))
    fb.append(("Next Due", f"{core.fmt_dt_local(child_due)}  ({delta})"))
    if analytics_advice:
        fb.append(("Analytics", analytics_advice))
    _append_integrity_warnings_row(fb, integrity_warnings)
    append_next_wait_sched_rows(fb, child, child_due)

    if cap_no:
        _append_link_status_rows(
            fb,
            cap_no,
            base_no,
            second_to_last_text="[yellow]Next link is the last in the chain.[/]",
        )
    else:
        fb.append(("Limits", "—"))

    _append_final_rows(fb, finals, now_utc, fmt_dt_local=core.fmt_dt_local, human_delta=human_delta)

    child_id = _resolve_child_id(
        child_short,
        deferred_spawn,
        chain_by_short,
        export_uuid_short_cached=export_uuid_short_cached,
    )
    if deferred_spawn and diag_enabled and spawn_intent_id:
        fb.append(("Intent", spawn_intent_id))

    title = f"⛓ Next link  #{next_no}  {parent_short} → {child_short}"
    tl = timeline_lines(
        "cp",
        new,
        child_due,
        child_short,
        None,
        next_count=3,
        cap_no=cap_no,
        cur_no=base_no,
        show_gaps=show_timeline_gaps,
    )
    if tl:
        fb.append(("Timeline", "\n".join(tl)))

    fb = format_next_cp_rows(fb)

    if (core.PANEL_MODE or "").strip().lower() == "line":
        line = format_line_preview(
            base_no,
            new,
            child_due,
            child_short,
            now_utc,
            cap_no=cap_no,
            until_dt=until_dt,
            until_no=until_cap_no,
            kind="cp",
        )
        title_style = chain_colour_for_task(new, "cp") if chain_color_per_chain else None
        panel_line(title, line, kind="preview_cp", border_style=title_style, title_style=title_style, markup_body=True)
    elif chain_color_per_chain:
        chain_colour = chain_colour_for_task(new, "cp")
        panel(
            title,
            fb,
            kind="preview_cp",
            border_style=chain_colour,
            title_style=chain_colour,
        )
    else:
        panel(title, fb, kind="preview_cp")
