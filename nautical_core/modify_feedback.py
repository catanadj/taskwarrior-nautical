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
        if lk in notable_labels or lk.startswith("final"):
            return True
        if lk == "basis":
            return True
        if lk == "analytics" and str(v or "").strip():
            return True
    return False


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
        "integrity",
        "timeline",
        "sanitised",
        "warning",
        "error",
        "intent",
    }
    out: list[tuple[str, object]] = []
    for k, v in rows:
        if k is None:
            continue
        lk = str(k).strip().lower()
        if lk == "timeline" and not include_timeline:
            continue
        if lk in keep_labels or lk.startswith("final"):
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
    fb = []
    anchor_raw = (feedback.new.get("anchor") or "").strip()
    expr_str = strip_quotes(anchor_raw)
    mode_tag = _anchor_mode_tag(feedback.new)
    fb.append(("Pattern", f"{expr_str}  {mode_tag}"))
    delta = core.humanize_delta(feedback.now_utc, feedback.child_due, use_months_days=core.expr_has_m_or_y(feedback.dnf))
    fb.append(("Next", f"#{feedback.next_no} → {core.fmt_dt_local(feedback.child_due)}  ({delta})"))
    fb.append(("Natural", core.describe_anchor_dnf(feedback.dnf, feedback.new)))
    basis_text = _pretty_basis_anchor(feedback.meta, feedback.new, parse_dt_any=core.parse_dt_any, fmt_dt_local=core.fmt_dt_local)
    if basis_text != "SKIP — Next anchor after completion (multi-time: between slots counts as previous slot)":
        fb.append(("Basis", basis_text))
    fb.append(("Root", format_root_and_age(feedback.new, feedback.now_utc)))

    _append_wait_sched_feedback_rows(fb, debug_wait_sched=debug_wait_sched, last_wait_sched_debug=last_wait_sched_debug)
    _append_sanitised_fields_row(fb, feedback.stripped_attrs)
    if feedback.analytics_advice:
        fb.append(("Analytics", feedback.analytics_advice))
    _append_integrity_warnings_row(fb, feedback.integrity_warnings)
    append_next_wait_sched_rows(fb, feedback.child, feedback.child_due)

    _append_link_status_rows(
        fb,
        feedback.cap_no,
        feedback.base_no,
        second_to_last_text="[yellow]This was the second-to-last link[/]",
    )
    _append_final_rows(fb, feedback.finals, feedback.now_utc, fmt_dt_local=core.fmt_dt_local, human_delta=human_delta)
    if feedback.deferred_spawn and diag_enabled and feedback.spawn_intent_id:
        fb.append(("Intent", feedback.spawn_intent_id))

    title = f"⚓︎ Next anchor  #{feedback.next_no}  {feedback.parent_short} → {feedback.child_short}"
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
    if "rand" in expr_str.lower():
        fb.append(("Rand", f"[dim]Deterministic picks seeded by root {short(root_uuid_from(feedback.new))}[/]"))

    fb = format_next_anchor_rows(fb)

    mode = _display_mode_name(core)
    if mode in {"line", "minimal"}:
        line = format_line_preview(
            feedback.base_no,
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.now_utc,
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            kind="anchor",
            minimal=(mode == "minimal"),
        )
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
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            kind="anchor",
            minimal=False,
        )
        text_line(line, kind="preview_anchor", markup_body=True)
        return
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
    export_uuid_short_cached = services.export_uuid_short_cached
    fb = []
    delta = core.humanize_delta(feedback.now_utc, feedback.child_due, use_months_days=False)
    fb.append(("Period", feedback.new.get("cp")))
    fb.append(("Next", f"#{feedback.next_no} → {core.fmt_dt_local(feedback.child_due)}  ({delta})"))
    basis_text = _pretty_basis_cp(feedback.new, feedback.meta, parse_cp_duration=core.parse_cp_duration)
    if basis_text != "Preserve wall clock (period is multiple of 24h)":
        fb.append(("Basis", basis_text))
    fb.append(("Root", format_root_and_age(feedback.new, feedback.now_utc)))
    if feedback.analytics_advice:
        fb.append(("Analytics", feedback.analytics_advice))
    _append_integrity_warnings_row(fb, feedback.integrity_warnings)
    append_next_wait_sched_rows(fb, feedback.child, feedback.child_due)

    if feedback.cap_no:
        _append_link_status_rows(
            fb,
            feedback.cap_no,
            feedback.base_no,
            second_to_last_text="[yellow]Next link is the last in the chain.[/]",
        )

    _append_final_rows(fb, feedback.finals, feedback.now_utc, fmt_dt_local=core.fmt_dt_local, human_delta=human_delta)

    child_id = _resolve_child_id(
        feedback.child_short,
        feedback.deferred_spawn,
        feedback.chain_by_short,
        export_uuid_short_cached=export_uuid_short_cached,
    )
    if feedback.deferred_spawn and diag_enabled and feedback.spawn_intent_id:
        fb.append(("Intent", feedback.spawn_intent_id))

    title = f"⛓ Next link  #{feedback.next_no}  {feedback.parent_short} → {feedback.child_short}"
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

    mode = _display_mode_name(core)
    if mode in {"line", "minimal"}:
        line = format_line_preview(
            feedback.base_no,
            feedback.new,
            feedback.child_due,
            feedback.child_short,
            feedback.now_utc,
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            kind="cp",
            minimal=(mode == "minimal"),
        )
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
            cap_no=feedback.cap_no,
            until_dt=feedback.until_dt,
            until_no=feedback.until_cap_no,
            kind="cp",
            minimal=False,
        )
        text_line(line, kind="preview_cp", markup_body=True)
        return
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
