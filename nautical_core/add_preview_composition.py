"""Presentation orchestration for the typed on-add workflow."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _compare_datetimes(host: Any, left: datetime, right: datetime) -> int:
    return host.core._import_sibling("timeutil").compare_datetimes(left, right)


def cp_add_period_builder(host: Any, td: timedelta):
    secs = int(td.total_seconds())
    preserve = secs % 86400 == 0

    def add_period(dt):
        if preserve:
            dl = host.core.to_local(dt)
            base = host.core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
            )
            return base.astimezone(timezone.utc)
        return (dt + td).replace(microsecond=0)

    return add_period


def cp_add_td(host: Any, dt: datetime, td: timedelta) -> datetime:
    return cp_add_period_builder(host, td)(dt)


def cp_sequence_period_for_link(
    host: Any,
    tokens: list[dict],
    cp_str: str,
    link_no: int,
    chain_id: str | None = None,
) -> timedelta:
    idx = (max(1, int(link_no)) - 1) % len(tokens)
    td = host.core.cp_sequence_interval_for_token(
        tokens[idx], cp=cp_str, link_no=link_no, token_index=idx, chain_id=chain_id
    )
    return td or timedelta()


def cp_until_summary(host: Any, due_dt: datetime, until_dt: datetime | None, add_period):
    if not until_dt:
        return None, None
    count = 0
    probe = due_dt
    last = None
    for _ in range(host._MAX_PREVIEW_ITERATIONS):
        if count >= host._MAX_ITERATIONS or _compare_datetimes(host, probe, until_dt) > 0:
            break
        last = probe
        count += 1
        probe = add_period(probe)
    if count <= 0:
        return None, None
    return max(0, count - 1), last


def cp_preview_lines(host: Any, due_dt: datetime, until_dt: datetime | None, limit: int, add_period) -> list[str]:
    preview = []
    nxt = due_dt
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    for i in range(limit):
        nxt = add_period(nxt)
        if until_dt and _compare_datetimes(host, nxt, until_dt) > 0:
            break
        color = colors[min(i, len(colors) - 1)]
        preview.append(f"[{color}]{host.core.fmt_dt_local(nxt)}[/{color}]")
    return preview


def cp_sequence_until_summary(
    host: Any,
    due_dt: datetime,
    until_dt: datetime | None,
    tokens: list[dict],
    cp_str: str,
    *,
    start_link_no: int,
    chain_id: str | None = None,
):
    if not until_dt:
        return None, None
    count = 0
    probe = due_dt
    last = None
    link_no = start_link_no
    for _ in range(host._MAX_PREVIEW_ITERATIONS):
        if _compare_datetimes(host, probe, until_dt) > 0:
            break
        last = probe
        count += 1
        td = cp_sequence_period_for_link(host, tokens, cp_str, link_no, chain_id)
        probe = cp_add_td(host, probe, td)
        link_no += 1
    if count <= 0:
        return None, None
    return max(0, count - 1), last


def cp_sequence_preview_lines(
    host: Any,
    due_dt: datetime,
    until_dt: datetime | None,
    limit: int,
    tokens: list[dict],
    cp_str: str,
    *,
    cp_tokens: list[str],
    start_link_no: int,
    chain_id: str | None = None,
) -> list[str]:
    preview = []
    nxt = due_dt
    link_no = start_link_no
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    for i in range(limit):
        step_idx = (max(1, link_no) - 1) % len(tokens)
        td = cp_sequence_period_for_link(host, tokens, cp_str, link_no, chain_id)
        nxt = cp_add_td(host, nxt, td)
        link_no += 1
        if until_dt and _compare_datetimes(host, nxt, until_dt) > 0:
            break
        color = colors[min(i, len(colors) - 1)]
        suffix = ""
        if 0 <= step_idx < len(cp_tokens):
            token = tokens[step_idx]
            label = host._fmt_cp_interval_token(td) if token.get("kind") == "rand" else cp_tokens[step_idx]
            suffix = f" [dim]({label})[/]"
        preview.append(f"[{color}]{host.core.fmt_dt_local(nxt)}[/{color}]{suffix}")
    return preview


def cp_limit_rows(
    host: Any,
    rows: list[tuple[str, str]],
    *,
    cpmax: int,
    due_dt: datetime,
    until_dt: datetime | None,
    exact_until_count: int | None,
    final_until_dt: datetime | None,
    add_period,
    now_utc: datetime,
    tokens: list[dict] | None = None,
    cp_str: str = "",
    start_link_no: int = 1,
    chain_id: str | None = None,
) -> None:
    final_candidates = []
    future_counts = []
    if cpmax and cpmax > 0:
        rows.append(("Chain cap", f"[bold yellow]#{cpmax}[/]"))
        fmax = due_dt
        steps = max(0, cpmax - 1)
        link_no = start_link_no
        for _ in range(steps):
            if tokens:
                fmax = cp_add_td(host, fmax, cp_sequence_period_for_link(host, tokens, cp_str, link_no, chain_id))
                link_no += 1
            else:
                fmax = add_period(fmax)
        final_candidates.append(fmax)
        future_counts.append(steps)
    if until_dt:
        rows.append(("Chain end point", f"[bold yellow]{host.core.fmt_dt_local(until_dt)}[/]"))
        if exact_until_count is not None:
            future_counts.append(exact_until_count)
        if final_until_dt:
            final_candidates.append(final_until_dt)
    if final_candidates:
        last = min(final_candidates)
        rows.append(("Last occurrence", f"[bright_magenta]{host.core.fmt_dt_local(last)}[/]  [dim]({host._human_delta(now_utc, last, True)})[/]"))
    if future_counts:
        rows.append(("Future links", f"[white]{min(future_counts)}[/]"))


def render_cp(host: Any, task, cp_str: str, ch: str, now_utc, user_provided_due: bool,
              recurrence_field: str, due_dt, until_dt, *, prof=None) -> None:
    core = host.core
    rows: list[tuple[str, str]] = []
    diagnostics = host._module("panel_diagnostics")
    for warning in diagnostics.panel_warnings(
        core, host._module("modify_models").TaskView.from_mapping(task), include_files=False
    ):
        rows.append(("Warning", f"[yellow]{warning}[/]"))

    def fmt(value):
        return core.fmt_dt_local(value)

    def parse(value):
        return core.parse_dt_any(value)

    tokens = core.parse_cp_sequence_tokens(cp_str)
    if not tokens:
        reason = core.cp_sequence_parse_error(cp_str) or f"couldn't parse duration from '{cp_str}'"
        host._error_and_exit([("Invalid cp", reason)])
    chain_id = str(task.get("chainID") or "").strip()
    td = cp_sequence_period_for_link(host, tokens, cp_str, 1, chain_id)
    has_random = any(token.get("kind") == "rand" for token in tokens)
    is_dynamic = len(tokens) > 1 or has_random

    if until_dt:
        reasonable, warning = host._validate_chain_duration_reasonable(
            until_dt, now_utc, now_utc + td if not user_provided_due else due_dt, "cp"
        )
        if not reasonable and warning:
            rows.append(("Warning", f"[yellow]{warning}[/]"))

    add_period = cp_add_period_builder(host, td)
    entry_dt = parse(task.get("entry")) if task.get("entry") else now_utc
    if not user_provided_due:
        due_dt = add_period(entry_dt)
        task["due"] = host._fmt_local_for_task(due_dt)
        rows.append(("[auto-due]", "Due was not set explicitly; assigned to entry+cp."))
        first_label = "First due"
    elif recurrence_field == "scheduled":
        due_dt = parse(task.get("scheduled"))
        first_label = "First scheduled"
    else:
        due_dt = parse(task.get("due"))
        first_label = "First due"

    host._validate_native_until_after_target_or_fail(task, due_dt, recurrence_field)
    rows.append(("Period", f"[bold white]{cp_str}[/]"))
    if is_dynamic:
        step_token = cp_str.split(",")[0].strip()
        if tokens[0].get("kind") == "rand":
            step_token = host._fmt_cp_interval_token(td)
        rows.append(("Step", f"[bold white]1/{len(tokens)}[/] [dim]({step_token})[/]"))
    rows.append((first_label, f"[bold bright_green]{fmt(due_dt)}[/]"))
    host._append_first_expiration_row(rows, task, due_dt, recurrence_field)
    host._append_wait_sched_rows(rows, task, due_dt, auto_due=not user_provided_due, anchor_field=recurrence_field)

    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if is_dynamic:
        exact_count, final_dt = cp_sequence_until_summary(
            host,
            due_dt, until_dt, tokens, cp_str, start_link_no=1, chain_id=chain_id
        )
    else:
        exact_count, final_dt = cp_until_summary(host, due_dt, until_dt, add_period)
    allow_max = (cpmax - 1) if cpmax and cpmax > 0 else 10**9
    allow_until = exact_count if exact_count is not None else 10**9
    limit = max(0, min(host.UPCOMING_PREVIEW, allow_max, allow_until, host._PREVIEW_HARD_CAP))
    if is_dynamic:
        preview = cp_sequence_preview_lines(
            host,
            due_dt, until_dt, limit, tokens, cp_str,
            cp_tokens=[part.strip() for part in cp_str.split(",")],
            start_link_no=1, chain_id=chain_id,
        )
    else:
        preview = cp_preview_lines(host, due_dt, until_dt, limit, add_period)
    rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
    rows.append(("Delta", f"[bright_yellow]{host._human_delta(now_utc, due_dt, False)}[/]"))
    cp_limit_rows(
        host,
        rows, cpmax=cpmax, due_dt=due_dt, until_dt=until_dt,
        exact_until_count=exact_count, final_until_dt=final_dt,
        add_period=add_period, now_utc=now_utc,
        tokens=tokens if is_dynamic else None, cp_str=cp_str,
        start_link_no=1, chain_id=chain_id,
    )
    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    host._panel("⛓ Recurring Chain Preview", host._format_cp_rows(rows), kind="preview_cp", task=task)


def render_anchor(host: Any, *, task, anchor_str, anchor_file_str, ch, now_utc, now_local,
                  user_provided_due, recurrence_field, due_dt, due_day, due_hhmm,
                  until_dt, past_due_warning, prof) -> None:
    core = host.core
    preview = host._module("add_anchor_preview")
    try:
        preview.handle_anchor_preview_on_add(
            task=task, anchor_str=anchor_str, anchor_file_str=anchor_file_str, ch=ch,
            now_utc=now_utc, now_local=now_local, user_provided_due=user_provided_due,
            recurrence_field=recurrence_field, due_dt=due_dt, due_day=due_day,
            due_hhmm=due_hhmm, until_dt=until_dt, past_due_warning=past_due_warning,
            prof=prof, anchor_warn=host.ANCHOR_WARN, upcoming_preview=host.UPCOMING_PREVIEW,
            preview_hard_cap=host._PREVIEW_HARD_CAP, max_summary_links=host._MAX_PREVIEW_ITERATIONS,
            core=core, root_uuid_from=host._root_uuid_from, short=host._short,
            validate_anchor_syntax_strict=host._validate_anchor_syntax_strict,
            validate_omit_syntax_strict=host._validate_omit_syntax_strict,
            validate_anchor_mode=host._validate_anchor_mode,
            validate_chain_duration_reasonable=host._validate_chain_duration_reasonable,
            append_wait_sched_rows=host._append_wait_sched_rows,
            anchor_until_summary=host._anchor_until_summary,
            anchor_build_preview=host._anchor_build_preview,
            to_local_cached=host._to_local_cached,
            fmt_local_for_task=host._fmt_local_for_task,
            format_anchor_rows=host._format_anchor_rows,
            panel=host._panel, human_delta=host._human_delta,
            error_and_exit=host._error_and_exit,
            validate_native_until_after_target=host._validate_native_until_after_target_or_fail,
            validate_native_until_anchor_slots=host._validate_native_until_anchor_slots_or_fail,
            append_first_expiration_row=host._append_first_expiration_row,
        )
    except Exception as exc:
        exhausted = getattr(core, "OccurrenceSearchExhausted", None)
        if exhausted is not None and isinstance(exc, exhausted):
            message = (core._import_sibling("scheduler_models").occurrence_exhaustion_message(exc)
                       if exc.is_date_limit else str(exc))
            host._error_and_exit([("Scheduler", message), ("Fix", "Use a less sparse rule, relax a constraint, or set a due before the date limit.")])
        astronomy = core._import_sibling("astronomy")
        if astronomy.is_astronomy_error(exc):
            host._error_and_exit([("Astronomy", astronomy.scheduling_error_message(exc))])
        raise


__all__ = (
    "cp_add_period_builder",
    "cp_add_td",
    "cp_sequence_period_for_link",
    "cp_until_summary",
    "cp_preview_lines",
    "cp_sequence_until_summary",
    "cp_sequence_preview_lines",
    "cp_limit_rows",
    "render_anchor",
    "render_cp",
)
