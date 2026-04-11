from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable


def anchor_preview_prepare_dnf(
    task: dict[str, Any],
    anchor_str: str,
    due_dt: datetime,
    rows: list[tuple[str, str]],
    prof: Any,
    *,
    core: Any,
    validate_anchor_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    validate_anchor_mode: Callable[[Any], tuple[str, str | None]],
    error_and_exit: Callable[[list[tuple[str, str]]], None],
) -> tuple[list[list[dict[str, Any]]], str]:
    _ = due_dt
    t0 = time.perf_counter()
    dnf, err = validate_anchor_syntax_strict(anchor_str)
    if dnf is None:
        error_and_exit([("Invalid anchor", err or "anchor syntax error")])

    mode, warn_msg = validate_anchor_mode(task.get("anchor_mode"))
    task["anchor_mode"] = mode
    if warn_msg:
        rows.append(("Warning", f"[yellow]{warn_msg}[/]"))
    prof.add_ms("anchor:dnf", (time.perf_counter() - t0) * 1000.0)

    tag = {
        "skip": "[bold bright_cyan]SKIP[/]",
        "all": "[bold yellow]ALL[/]",
        "flex": "[bold magenta]FLEX[/]",
    }.get(mode, "[bold bright_cyan]SKIP[/]")
    rows.append(("Pattern", f"[white]{anchor_str}[/]  {tag}"))
    try:
        rows.append(("Natural", f"[white]{core.describe_anchor_dnf(dnf, task)}[/]"))
    except Exception:
        pass
    return dnf, mode


def anchor_preview_prepare_omit_dnf(
    task: dict[str, Any],
    rows: list[tuple[str, str]],
    *,
    core: Any,
    validate_omit_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    error_and_exit: Callable[[list[tuple[str, str]]], None],
) -> list[list[dict[str, Any]]] | None:
    omit_str = str(task.get("omit") or "").strip()
    if not omit_str:
        return None
    dnf, err = validate_omit_syntax_strict(omit_str)
    if dnf is None:
        error_and_exit([("Invalid omit", err or "omit syntax error")])
    rows.append(("Omit", f"[white]{omit_str}[/]"))
    try:
        rows.append(("Except", f"[white]{core.describe_anchor_expr(omit_str)}[/]"))
    except Exception:
        pass
    try:
        _fatal, warns = core.lint_anchor_expr(omit_str)
        for w in warns or []:
            rows.append(("Warning", f"[yellow]{w}[/]"))
    except Exception:
        pass
    return dnf


def anchor_preview_seed_context(
    task: dict[str, Any],
    due_day: Any,
    now_local: datetime,
    user_provided_due: bool,
    *,
    root_uuid_from: Callable[[dict[str, Any]], str | None],
) -> tuple[Any, Any, str]:
    base_local_date = due_day if user_provided_due else now_local.date()
    seed_base = (task.get("chainID") or "").strip() or root_uuid_from(task) or "preview"
    interval_seed = base_local_date
    return base_local_date, interval_seed, seed_base


def anchor_preview_first_due(
    task: dict[str, Any],
    dnf,
    omit_dnf,
    *,
    now_local: datetime,
    due_dt: datetime,
    user_provided_due: bool,
    recurrence_field: str,
    due_hhmm: tuple[int, int],
    interval_seed: Any,
    seed_base: str,
    rows: list[tuple[str, str]],
    prof: Any,
    core: Any,
    to_local_cached: Callable[[datetime], datetime],
    anchor_pick_occurrence_local: Callable[..., Any],
    error_and_exit: Callable[[list[tuple[str, str]]], None],
    fmt_local_for_task: Callable[[datetime], str],
) -> tuple[Any, datetime, datetime, Any, tuple[int, int]]:
    def _fmt(dt):
        return core.fmt_dt_local(dt)

    fallback_hhmm = due_hhmm if user_provided_due else (9, 0)
    t_first = time.perf_counter()
    if user_provided_due:
        due_local_dt = to_local_cached(due_dt)
        first_due_local_dt = anchor_pick_occurrence_local(
            dnf,
            due_local_dt,
            inclusive=False,
            fallback_hhmm=fallback_hhmm,
            interval_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
        )
        if not first_due_local_dt:
            error_and_exit([("anchor pattern", "No matching anchor occurrences found after the provided due.")])
    else:
        first_due_local_dt = anchor_pick_occurrence_local(
            dnf,
            now_local,
            inclusive=True,
            fallback_hhmm=fallback_hhmm,
            interval_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
        )
        if not first_due_local_dt:
            error_and_exit([("anchor pattern", "No matching anchor occurrences found.")])
    prof.add_ms("anchor:first_occurrence", (time.perf_counter() - t_first) * 1000.0)

    first_hhmm = (first_due_local_dt.hour, first_due_local_dt.minute)
    first_date_local = first_due_local_dt.date()
    first_due_utc = first_due_local_dt.astimezone(timezone.utc)
    if user_provided_due:
        display_first_due_utc = due_dt
        first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
        rows.append((first_label, f"[bold bright_green]{_fmt(display_first_due_utc)}[/]"))
        rows.append(("Next anchor", f"[white]{_fmt(first_due_utc)}[/]"))
    else:
        display_first_due_utc = first_due_utc
        rows.append(("First due", f"[bold bright_green]{_fmt(display_first_due_utc)}[/]"))
        task["due"] = fmt_local_for_task(first_due_utc)
        rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))
    return first_due_local_dt, first_due_utc, display_first_due_utc, first_date_local, first_hhmm


def anchor_preview_misaligned_due_warning(
    rows: list[tuple[str, str]],
    *,
    dnf,
    due_dt: datetime,
    recurrence_field: str,
    interval_seed: Any,
    seed_base: str,
    omit_dnf,
    to_local_cached: Callable[[datetime], datetime],
    anchor_step_once: Callable[..., Any],
) -> None:
    due_local_date = to_local_cached(due_dt).date()
    first_after_due_date = anchor_step_once(
        dnf,
        due_local_date - timedelta(days=1),
        interval_seed,
        seed_base,
        omit_dnf=omit_dnf,
    )
    if first_after_due_date != due_local_date:
        anchor_name = "scheduled" if recurrence_field == "scheduled" else "due"
        rows.append(
            (
                "Note",
                f"[italic yellow]Your {anchor_name} is not an anchor day; chain follows anchors."
                f" To align, set {anchor_name} to a matching anchor day or omit {anchor_name} to auto-assign.[/]",
            )
        )


def anchor_preview_lint_and_validate(
    anchor_str: str,
    prof: Any,
    *,
    core: Any,
    panel: Callable[..., None],
) -> None:
    t_lint = time.perf_counter()
    _, warns = core.lint_anchor_expr(anchor_str)
    prof.add_ms("anchor:lint", (time.perf_counter() - t_lint) * 1000.0)
    if warns:
        panel("ℹ️  Lint", [("Hint", w) for w in warns], kind="note")

    t_val = time.perf_counter()
    core.validate_anchor_expr_strict(anchor_str)
    prof.add_ms("anchor:validate_strict", (time.perf_counter() - t_val) * 1000.0)


def anchor_preview_limit_rows(
    rows: list[tuple[str, str]],
    *,
    cpmax: int,
    until_dt: datetime | None,
    exact_until_count: int | None,
    final_until_dt: datetime | None,
    now_utc: datetime,
    core: Any,
    human_delta: Callable[[Any, Any, bool], str],
) -> None:
    def _fmt(dt):
        return core.fmt_dt_local(dt)

    lim_parts = []
    if cpmax and cpmax > 0:
        lim_parts.append(f"max [bold yellow]{cpmax}[/]")
    if until_dt:
        lim_parts.append(f"until [bold yellow]{_fmt(until_dt)}[/]")
        if exact_until_count is not None:
            lim_parts.append(f"[white]{exact_until_count} more[/]")
    if lim_parts:
        rows.append(("Limits", " [dim]|[/] ".join(lim_parts)))
    if final_until_dt:
        rows.append(
            (
                "Final (until)",
                f"[bright_magenta]{_fmt(final_until_dt)}[/]  [dim]({human_delta(now_utc, final_until_dt, True)})[/]",
            )
        )


def handle_anchor_preview_on_add(
    *,
    task: dict[str, Any],
    anchor_str: str,
    ch: str,
    now_utc: datetime,
    now_local: datetime,
    user_provided_due: bool,
    recurrence_field: str,
    due_dt: datetime,
    due_day: Any,
    due_hhmm: tuple[int, int],
    until_dt: datetime | None,
    past_due_warning: str | None,
    prof: Any,
    anchor_warn: bool,
    upcoming_preview: int,
    preview_hard_cap: int,
    core: Any,
    root_uuid_from: Callable[[dict[str, Any]], str | None],
    short: Callable[[Any], str],
    validate_anchor_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    validate_omit_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    validate_anchor_mode: Callable[[Any], tuple[str, str | None]],
    validate_chain_duration_reasonable: Callable[[Any, datetime, Any, str], tuple[bool, str | None]],
    append_wait_sched_rows: Callable[..., None],
    anchor_step_once: Callable[..., Any],
    anchor_pick_occurrence_local: Callable[..., Any],
    anchor_until_summary: Callable[..., tuple[int | None, datetime | None]],
    anchor_build_preview: Callable[..., list[str]],
    to_local_cached: Callable[[datetime], datetime],
    fmt_local_for_task: Callable[[datetime], str],
    format_anchor_rows: Callable[[list[tuple[str, str]]], list[tuple[str | None, str]]],
    panel: Callable[..., None],
    emit_task_json: Callable[..., None],
    human_delta: Callable[[Any, Any, bool], str],
    error_and_exit: Callable[[list[tuple[str, str]]], None],
) -> None:
    rows: list[tuple[str, str]] = []

    dnf, _ = anchor_preview_prepare_dnf(
        task,
        anchor_str,
        due_dt,
        rows,
        prof,
        core=core,
        validate_anchor_syntax_strict=validate_anchor_syntax_strict,
        validate_anchor_mode=validate_anchor_mode,
        error_and_exit=error_and_exit,
    )
    omit_dnf = anchor_preview_prepare_omit_dnf(
        task,
        rows,
        core=core,
        validate_omit_syntax_strict=validate_omit_syntax_strict,
        error_and_exit=error_and_exit,
    )
    base_local_date, interval_seed, seed_base = anchor_preview_seed_context(
        task,
        due_day,
        now_local,
        user_provided_due,
        root_uuid_from=root_uuid_from,
    )

    first_date_local = anchor_step_once(dnf, base_local_date - timedelta(days=1), interval_seed, seed_base, omit_dnf=omit_dnf)
    if not first_date_local:
        error_and_exit(
            [
                (
                    "anchor pattern",
                    "No matching anchor dates found. Pattern may be invalid, non-advancing, or too restrictive.",
                )
            ]
        )

    (
        first_due_local_dt,
        first_due_utc,
        display_first_due_utc,
        first_date_local,
        first_hhmm,
    ) = anchor_preview_first_due(
        task,
        dnf,
        omit_dnf,
        now_local=now_local,
        due_dt=due_dt,
        user_provided_due=user_provided_due,
        recurrence_field=recurrence_field,
        due_hhmm=due_hhmm,
        interval_seed=interval_seed,
        seed_base=seed_base,
        rows=rows,
        prof=prof,
        core=core,
        to_local_cached=to_local_cached,
        anchor_pick_occurrence_local=anchor_pick_occurrence_local,
        error_and_exit=error_and_exit,
        fmt_local_for_task=fmt_local_for_task,
    )

    append_wait_sched_rows(
        rows,
        task,
        display_first_due_utc,
        auto_due=(not user_provided_due),
        anchor_field=recurrence_field,
    )
    if past_due_warning:
        rows.append(("Warning", f"[yellow]{past_due_warning}[/]"))
    if user_provided_due and anchor_warn:
        anchor_preview_misaligned_due_warning(
            rows,
            dnf=dnf,
            due_dt=due_dt,
            recurrence_field=recurrence_field,
            interval_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            to_local_cached=to_local_cached,
            anchor_step_once=anchor_step_once,
        )

    if until_dt:
        is_reasonable, warn_msg = validate_chain_duration_reasonable(
            until_dt,
            now_utc,
            first_due_utc,
            "anchor",
        )
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    cpmax = core.coerce_int(task.get("chainMax"), 0)
    exact_until_count, final_until_dt = anchor_until_summary(
        dnf,
        until_dt,
        first_date_local,
        first_hhmm,
        interval_seed,
        seed_base,
        omit_dnf=omit_dnf,
    )

    allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
    allow_by_until = exact_until_count if exact_until_count is not None else 10**9

    anchor_preview_lint_and_validate(anchor_str, prof, core=core, panel=panel)

    preview_limit = max(0, min(upcoming_preview, allow_by_max, allow_by_until, preview_hard_cap))
    fallback_hhmm = first_hhmm
    _t_prev = time.perf_counter()
    preview = anchor_build_preview(
        dnf,
        first_due_local_dt,
        preview_limit,
        until_dt,
        fallback_hhmm,
        interval_seed,
        seed_base,
        omit_dnf=omit_dnf,
    )
    prof.add_ms("anchor:preview_occurrences", (time.perf_counter() - _t_prev) * 1000.0)
    rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
    rows.append(
        (
            "Delta",
            f"[bright_yellow]{human_delta(now_utc, display_first_due_utc, core.expr_has_m_or_y(dnf))}[/]",
        )
    )

    anchor_preview_limit_rows(
        rows,
        cpmax=cpmax,
        until_dt=until_dt,
        exact_until_count=exact_until_count,
        final_until_dt=final_until_dt,
        now_utc=now_utc,
        core=core,
        human_delta=human_delta,
    )

    if "rand" in anchor_str.lower():
        base = short(root_uuid_from(task))
        rows.append(("Rand", f"[dim italic]Preview uses provisional seed; final picks are chain-bound to {base}[/]"))

    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    formatted_rows = format_anchor_rows(rows)
    _t_panel = time.perf_counter()
    panel("⚓︎ Anchor Preview", formatted_rows, kind="preview_anchor")
    prof.add_ms("render:anchor_panel", (time.perf_counter() - _t_panel) * 1000.0)

    emit_task_json(task, sanitize=True, prof=prof)
