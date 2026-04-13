from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .add_anchor_compute import anchor_next_occurrence_after_local_dt
from .anchor_inclusion import collect_included_occurrences_local


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
):
    omit_str = str(task.get("omit") or "").strip()
    omit_file = str(task.get("omit_file") or "").strip()
    omit_dnf = None
    omit_dates: frozenset[Any] = frozenset()
    if omit_str:
        dnf, err = validate_omit_syntax_strict(omit_str)
        if dnf is None:
            error_and_exit([("Invalid omit", err or "omit syntax error")])
        omit_dnf = dnf
        rows.append(("Omit", f"[white]{omit_str}[/]"))
        try:
            anchor_omit = core._import_sibling("anchor_omit")
            omit_norm = anchor_omit.normalize_omit_expr(omit_str)
        except Exception:
            omit_norm = omit_str
        try:
            rows.append(("Except", f"[white]{core.describe_anchor_expr(omit_norm)}[/]"))
        except Exception:
            pass
        try:
            _fatal, warns = core.lint_anchor_expr(omit_norm)
            for w in warns or []:
                rows.append(("Warning", f"[yellow]{w}[/]"))
        except Exception:
            pass
    if omit_file:
        try:
            omit_files = core._import_sibling("omit_files")
            omit_dates = omit_files.load_omit_file_dates(omit_file, getattr(core, "OMIT_FILE_DIR", ""))
        except Exception as e:
            error_and_exit([("Invalid omit_file", str(e))])
        rows.append(("Omit file", f"[white]{omit_file}[/]"))
    if not omit_dnf and not omit_dates:
        return None
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        return anchor_omit.combine_omit_state(omit_dnf=omit_dnf, omit_dates=omit_dates)
    except Exception:
        if omit_dates:
            return {"dnf": omit_dnf, "dates": frozenset(omit_dates)}
        return omit_dnf


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


def _anchor_file_occurrences_local(
    anchor_file_str: str,
    *,
    core: Any,
    fallback_hhmm: tuple[int, int],
) -> list[datetime]:
    anchor_files = core._import_sibling("anchor_files")
    dates = sorted(anchor_files.load_anchor_file_dates(anchor_file_str, getattr(core, "ANCHOR_FILE_DIR", "")))
    _file_name, mods = anchor_files.parse_anchor_file_spec(anchor_file_str)
    tval = mods.get("t")
    if isinstance(tval, tuple):
        times = [tval]
    elif isinstance(tval, list):
        times = [item for item in tval if isinstance(item, tuple)]
    else:
        times = []
    if not times:
        times = [fallback_hhmm]
    out: list[datetime] = []
    for d0 in dates:
        for hhmm in times:
            out.append(core.to_local(core.build_local_datetime(d0, hhmm)))
    out.sort()
    return out


def _anchor_file_is_omitted(omit_dnf, item_local: datetime, *, core: Any, seed_base: str) -> bool:
    if not omit_dnf:
        return False
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        return bool(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf,
                item_local.date(),
                item_local.date(),
                seed_base,
                core=core,
            )
        )
    except Exception:
        return False


def _anchor_file_preview_occurrences(
    anchor_file_str: str,
    *,
    core: Any,
    fallback_hhmm: tuple[int, int],
    omit_dnf,
    seed_base: str,
) -> list[datetime]:
    out: list[datetime] = []
    for item_local in _anchor_file_occurrences_local(anchor_file_str, core=core, fallback_hhmm=fallback_hhmm):
        if _anchor_file_is_omitted(omit_dnf, item_local, core=core, seed_base=seed_base):
            continue
        out.append(item_local)
    return out


def _anchor_file_natural_text(anchor_file_str: str) -> str:
    file_name = str(anchor_file_str or "").strip().split("@", 1)[0]
    return f"Dates from {file_name}"


def handle_anchor_file_preview_on_add(
    *,
    task: dict[str, Any],
    anchor_file_str: str,
    ch: str,
    now_utc: datetime,
    now_local: datetime,
    user_provided_due: bool,
    recurrence_field: str,
    due_dt: datetime,
    due_hhmm: tuple[int, int],
    until_dt: datetime | None,
    past_due_warning: str | None,
    prof: Any,
    anchor_warn: bool,
    upcoming_preview: int,
    preview_hard_cap: int,
    core: Any,
    append_wait_sched_rows: Callable[..., None],
    validate_chain_duration_reasonable: Callable[[Any, datetime, Any, str], tuple[bool, str | None]],
    validate_omit_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    format_anchor_rows: Callable[[list[tuple[str, str]]], list[tuple[str | None, str]]],
    panel: Callable[..., None],
    emit_task_json: Callable[..., None],
    fmt_local_for_task: Callable[[datetime], str],
    human_delta: Callable[[Any, Any, bool], str],
    error_and_exit: Callable[[list[tuple[str, str]]], None],
) -> None:
    rows: list[tuple[str, str]] = []
    rows.append(("Anchor file", f"[white]{anchor_file_str}[/]  [bold bright_cyan]SKIP[/]"))
    rows.append(("Natural", f"[white]{_anchor_file_natural_text(anchor_file_str)}[/]"))
    omit_dnf = anchor_preview_prepare_omit_dnf(
        task,
        rows,
        core=core,
        validate_omit_syntax_strict=validate_omit_syntax_strict,
        error_and_exit=error_and_exit,
    )

    t_occ = time.perf_counter()
    seed_base = (task.get("chainID") or "").strip() or "preview"
    all_occurrences = _anchor_file_preview_occurrences(
        anchor_file_str,
        core=core,
        fallback_hhmm=(due_hhmm if user_provided_due else (9, 0)),
        omit_dnf=omit_dnf,
        seed_base=seed_base,
    )
    prof.add_ms("anchor_file:occurrences", (time.perf_counter() - t_occ) * 1000.0)
    if not all_occurrences:
        error_and_exit([("anchor_file", "No matching anchor_file occurrences found.")])

    due_local_dt = core.to_local(due_dt)
    if user_provided_due:
        first_due_local_dt = next((dt for dt in all_occurrences if dt > due_local_dt), None)
        if not first_due_local_dt:
            error_and_exit([("anchor_file", "No matching anchor_file occurrences found after the provided due.")])
    else:
        first_due_local_dt = next((dt for dt in all_occurrences if dt >= now_local), None)
        if not first_due_local_dt:
            error_and_exit([("anchor_file", "No matching anchor_file occurrences found.")])

    first_due_utc = first_due_local_dt.astimezone(timezone.utc)
    display_first_due_utc = due_dt if user_provided_due else first_due_utc
    if user_provided_due:
        first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
        rows.append((first_label, f"[bold bright_green]{core.fmt_dt_local(display_first_due_utc)}[/]"))
        rows.append(("Next anchor", f"[white]{core.fmt_dt_local(first_due_utc)}[/]"))
    else:
        rows.append(("First due", f"[bold bright_green]{core.fmt_dt_local(first_due_utc)}[/]"))
        task["due"] = fmt_local_for_task(first_due_utc)
        rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))

    append_wait_sched_rows(
        rows,
        task,
        display_first_due_utc,
        auto_due=(not user_provided_due),
        anchor_field=recurrence_field,
    )
    if past_due_warning:
        rows.append(("Warning", f"[yellow]{past_due_warning}[/]"))
    if user_provided_due and anchor_warn and due_local_dt.date() != first_due_local_dt.date():
        anchor_name = "scheduled" if recurrence_field == "scheduled" else "due"
        rows.append(
            (
                "Note",
                f"[italic yellow]Your {anchor_name} is not an anchor day; chain follows anchors."
                f" To align, set {anchor_name} to a matching anchor day or omit {anchor_name} to auto-assign.[/]",
            )
        )

    if until_dt:
        is_reasonable, warn_msg = validate_chain_duration_reasonable(until_dt, now_utc, first_due_utc, "anchor")
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    cpmax = core.coerce_int(task.get("chainMax"), 0)
    upcoming = [dt for dt in all_occurrences if dt > first_due_local_dt]
    if until_dt:
        until_local = core.to_local(until_dt)
        upcoming = [dt for dt in upcoming if dt <= until_local]
    exact_until_count = None
    final_until_dt = None
    if until_dt:
        limited = [dt for dt in all_occurrences if dt <= core.to_local(until_dt)]
        exact_until_count = max(0, len(limited) - 1)
        if limited:
            final_until_dt = limited[-1].astimezone(timezone.utc)
    allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
    allow_by_until = exact_until_count if exact_until_count is not None else 10**9
    preview_limit = max(0, min(upcoming_preview, allow_by_max, allow_by_until, preview_hard_cap))
    preview_rows: list[str] = []
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    for idx, item_local in enumerate(upcoming[:preview_limit], start=2):
        color = colors[min(idx - 2, len(colors) - 1)]
        preview_rows.append(f"[{color}]{idx} ▸ {core.fmt_dt_local(item_local.astimezone(timezone.utc))}[/]")
    rows.append(("Upcoming", "\n".join(preview_rows) if preview_rows else "[dim]–[/]"))
    rows.append(("Delta", f"[bright_yellow]{human_delta(now_utc, display_first_due_utc, False)}[/]"))
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
    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    panel("⚓︎ Anchor Preview", format_anchor_rows(rows), kind="preview_anchor")
    emit_task_json(task, sanitize=True, prof=prof)


def handle_anchor_preview_on_add(
    *,
    task: dict[str, Any],
    anchor_str: str,
    anchor_file_str: str = "",
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
    dnf = None
    if anchor_str:
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
    else:
        mode, warn_msg = validate_anchor_mode(task.get("anchor_mode"))
        task["anchor_mode"] = mode
        if warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))
    if anchor_file_str:
        rows.append(("Anchor file", f"[white]{anchor_file_str}[/]"))
        rows.append(("Natural", f"[white]{_anchor_file_natural_text(anchor_file_str)}[/]"))

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

    merged = bool(dnf and anchor_file_str)
    fallback_hhmm = due_hhmm if user_provided_due else (9, 0)

    if not dnf:
        occurrences = collect_included_occurrences_local(
            dnf=None,
            anchor_file_str=anchor_file_str,
            after_local_dt=(to_local_cached(due_dt) if user_provided_due else now_local),
            inclusive=not user_provided_due,
            limit=preview_hard_cap + 16,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=anchor_next_occurrence_after_local_dt,
            pick_occurrence_local=anchor_pick_occurrence_local,
            anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
        )
        if not occurrences:
            error_and_exit([("anchor_file", "No matching anchor_file occurrences found.")])
        first_due_local_dt = occurrences[0]
        first_due_utc = first_due_local_dt.astimezone(timezone.utc)
        display_first_due_utc = due_dt if user_provided_due else first_due_utc
        first_date_local = first_due_local_dt.date()
        first_hhmm = (first_due_local_dt.hour, first_due_local_dt.minute)
        if user_provided_due:
            first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
            rows.append((first_label, f"[bold bright_green]{core.fmt_dt_local(display_first_due_utc)}[/]"))
            rows.append(("Next anchor", f"[white]{core.fmt_dt_local(first_due_utc)}[/]"))
        else:
            rows.append(("First due", f"[bold bright_green]{core.fmt_dt_local(first_due_utc)}[/]"))
            task["due"] = fmt_local_for_task(first_due_utc)
            rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))
    elif not merged:
        first_date_local = anchor_step_once(dnf, base_local_date - timedelta(days=1), interval_seed, seed_base, omit_dnf=omit_dnf)
        if not first_date_local:
            error_and_exit([("anchor pattern", "No matching anchor dates found. Pattern may be invalid, non-advancing, or too restrictive.")])
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
    else:
        occurrences = collect_included_occurrences_local(
            dnf=dnf,
            anchor_file_str=anchor_file_str,
            after_local_dt=(to_local_cached(due_dt) if user_provided_due else now_local),
            inclusive=not user_provided_due,
            limit=preview_hard_cap + 16,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=anchor_next_occurrence_after_local_dt,
            pick_occurrence_local=anchor_pick_occurrence_local,
            anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
        )
        if not occurrences:
            error_and_exit([("anchor pattern", "No matching anchor occurrences found.")])
        first_due_local_dt = occurrences[0]
        first_due_utc = first_due_local_dt.astimezone(timezone.utc)
        display_first_due_utc = due_dt if user_provided_due else first_due_utc
        first_date_local = first_due_local_dt.date()
        first_hhmm = (first_due_local_dt.hour, first_due_local_dt.minute)
        if user_provided_due:
            first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
            rows.append((first_label, f"[bold bright_green]{core.fmt_dt_local(display_first_due_utc)}[/]"))
            rows.append(("Next anchor", f"[white]{core.fmt_dt_local(first_due_utc)}[/]"))
        else:
            rows.append(("First due", f"[bold bright_green]{core.fmt_dt_local(first_due_utc)}[/]"))
            task["due"] = fmt_local_for_task(first_due_utc)
            rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))

    append_wait_sched_rows(
        rows,
        task,
        display_first_due_utc,
        auto_due=(not user_provided_due),
        anchor_field=recurrence_field,
    )
    if past_due_warning:
        rows.append(("Warning", f"[yellow]{past_due_warning}[/]"))
    if user_provided_due and anchor_warn and dnf and not merged:
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
    preview: list[str]
    exact_until_count = None
    final_until_dt = None
    if dnf and not merged:
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
        _t_prev = time.perf_counter()
        preview = anchor_build_preview(
            dnf,
            first_due_local_dt,
            preview_limit,
            until_dt,
            first_hhmm,
            interval_seed,
            seed_base,
            omit_dnf=omit_dnf,
        )
        prof.add_ms("anchor:preview_occurrences", (time.perf_counter() - _t_prev) * 1000.0)
    else:
        all_occurrences = collect_included_occurrences_local(
            dnf=dnf,
            anchor_file_str=anchor_file_str,
            after_local_dt=first_due_local_dt,
            inclusive=True,
            limit=preview_hard_cap + 24,
            fallback_hhmm=first_hhmm,
            default_seed_date=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=anchor_next_occurrence_after_local_dt,
            pick_occurrence_local=anchor_pick_occurrence_local,
            anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
        )
        if until_dt:
            until_local = core.to_local(until_dt)
            limited = [dt for dt in all_occurrences if dt <= until_local]
            exact_until_count = max(0, len(limited) - 1)
            if limited:
                final_until_dt = limited[-1].astimezone(timezone.utc)
        allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
        allow_by_until = exact_until_count if exact_until_count is not None else 10**9
        preview_limit = max(0, min(upcoming_preview, allow_by_max, allow_by_until, preview_hard_cap))
        colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
        preview = []
        upcoming = [dt for dt in all_occurrences if dt > first_due_local_dt]
        if until_dt:
            until_local = core.to_local(until_dt)
            upcoming = [dt for dt in upcoming if dt <= until_local]
        for i, dt in enumerate(upcoming[:preview_limit]):
            color = colors[min(i, len(colors) - 1)]
            preview.append(f"[{color}]{core.fmt_dt_local(dt.astimezone(timezone.utc))}[/{color}]")
        if anchor_str:
            anchor_preview_lint_and_validate(anchor_str, prof, core=core, panel=panel)

    rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
    rows.append(("Delta", f"[bright_yellow]{human_delta(now_utc, display_first_due_utc, bool(dnf and core.expr_has_m_or_y(dnf)))}[/]"))

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

    if anchor_str and "rand" in anchor_str.lower():
        base = short(root_uuid_from(task))
        rows.append(("Rand", f"[dim italic]Preview uses provisional seed; final picks are chain-bound to {base}[/]"))

    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    formatted_rows = format_anchor_rows(rows)
    _t_panel = time.perf_counter()
    panel("⚓︎ Anchor Preview", formatted_rows, kind="preview_anchor")
    prof.add_ms("render:anchor_panel", (time.perf_counter() - _t_panel) * 1000.0)

    emit_task_json(task, sanitize=True, prof=prof)

