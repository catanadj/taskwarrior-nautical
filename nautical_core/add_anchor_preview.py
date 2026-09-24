from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, NoReturn

from . import panel_diagnostics
from .occurrence_provider import Occurrence, OccurrenceBatch
from .scheduler_models import occurrence_exhaustion_message
from .timeutil import compare_datetimes
from .task_models import TaskPayload


@dataclass(frozen=True, slots=True)
class AnchorExpressionPreviewServices:
    """Composition-root dependencies for the anchor preview renderer."""

    panel_mode: str
    panel_warnings: Callable[[TaskPayload], list[str]]
    prepare_anchor_dnf: Callable[..., tuple[Any, str]]
    describe_anchor_natural: Callable[[TaskPayload, Any, str], str]
    prepare_omit_dnf: Callable[[TaskPayload, list[tuple[str, str]]], Any]
    scheduler_service_for_task: Callable[[TaskPayload], Any]
    to_local: Callable[[datetime], datetime]
    fmt_dt_local: Callable[[Any], str]
    coerce_int: Callable[..., Any]
    expr_has_m_or_y: Callable[[Any], bool]
    append_dst_adjustment: Callable[[list[tuple[str, str]], Any, datetime], None]
    render_business_calendar_displacement: Callable[..., Any]
    lint_and_validate: Callable[..., None]
    omit_description_for_task_date: Callable[[TaskPayload, Any], str | None]
    root_uuid_from: Callable[[TaskPayload], str | None]
    short: Callable[[Any], str]
    validate_anchor_mode: Callable[..., Any]
    validate_chain_duration_reasonable: Callable[..., Any]
    append_wait_sched_rows: Callable[..., Any]
    anchor_until_summary: Callable[..., Any]
    to_local_cached: Callable[..., Any]
    fmt_local_for_task: Callable[..., Any]
    format_anchor_rows: Callable[..., Any]
    panel: Callable[..., Any]
    human_delta: Callable[..., Any]
    error_and_exit: Callable[..., Any]
    validate_native_until_after_target: Callable[..., Any]
    validate_native_until_anchor_slots: Callable[..., Any]
    append_first_expiration_row: Callable[..., Any]


@dataclass(frozen=True, slots=True)
class AnchorFilePreviewServices:
    panel_mode: str
    timezone_fallback_warning: Callable[[str], bool]
    prepare_omit_dnf: Callable[[TaskPayload, list[tuple[str, str]]], Any]
    scheduler_service_for_task: Callable[[TaskPayload], Any]
    to_local: Callable[[datetime], datetime]
    fmt_dt_local: Callable[[Any], str]
    coerce_int: Callable[..., Any]
    render_business_calendar_displacement: Callable[..., Any]
    omit_description_for_task_date: Callable[[TaskPayload, Any], str | None]
    append_wait_sched_rows: Callable[..., Any]
    validate_chain_duration_reasonable: Callable[..., Any]
    format_anchor_rows: Callable[..., Any]
    panel: Callable[..., Any]
    fmt_local_for_task: Callable[..., Any]
    human_delta: Callable[..., Any]
    error_and_exit: Callable[..., Any]


def _event_datetime(event: Any) -> datetime | None:
    value = event.local_datetime if isinstance(event, Occurrence) else (event[0] if isinstance(event, tuple) and event else event)
    return value if isinstance(value, datetime) else None


def _event_on_or_before(event: Any, boundary: datetime) -> bool:
    value = _event_datetime(event)
    return value is not None and compare_datetimes(value, boundary) <= 0

def _preview_seed_base(task: TaskPayload, fallback_chain_id: str) -> str:
    """Resolve the stable preview identity at the raw-input boundary."""
    return str(task.get("chainID") or fallback_chain_id).strip()


def _anchor_file_natural_text(expr: str) -> str:
    file_name = str(expr or '').split('@', 1)[0].strip()
    return f"Dates from {file_name}" if file_name else ""


def _anchor_omit_natural_text(task: TaskPayload, *, core: Any) -> str:
    omit_raw = str(task.get('omit') or '').strip()
    omit_file = str(task.get('omit_file') or '').strip()
    parts: list[str] = []
    if omit_raw:
        try:
            anchor_omit = core._import_sibling('anchor_omit')
            omit_expr = core._parser_api.resolve_omit_presets(omit_raw)
            omit_norm = anchor_omit.normalize_omit_expr(omit_expr)
        except (KeyError, TypeError, ValueError):
            omit_norm = omit_raw
        try:
            natural = core.describe_anchor_expr(omit_norm)
        except (KeyError, TypeError, ValueError):
            natural = ''
        parts.append(natural or omit_raw)
    if omit_file:
        file_text = _anchor_file_natural_text(omit_file)
        if file_text:
            parts.append(file_text)
    return ' and '.join(part for part in parts if part)


def _anchor_preview_natural_text(task: TaskPayload, dnf: Any, anchor_file_str: str, *, core: Any) -> str:
    natural = core.describe_anchor_dnf(dnf, task) if dnf else ''
    omit_text = _anchor_omit_natural_text(task, core=core)
    if omit_text and (task.get('anchor_mode') or 'skip').lower() == 'skip':
        tail = '; skip missed anchors'
        if natural.endswith(tail):
            natural = natural[:-len(tail)]
        natural = natural.rstrip()
        return f"{natural}; omit {omit_text}" if natural else f"omit {omit_text}"
    file_text = _anchor_file_natural_text(anchor_file_str) if anchor_file_str else ''
    if natural and file_text:
        return f"{natural} and {file_text}"
    return natural or file_text


def anchor_preview_prepare_dnf(
    task: TaskPayload,
    anchor_str: str,
    due_dt: datetime,
    rows: list[tuple[str, str]],
    prof: Any,
    *,
    core: Any,
    validate_anchor_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    validate_anchor_mode: Callable[[Any], tuple[str, str | None]],
    error_and_exit: Callable[[list[tuple[str, str]]], NoReturn],
) -> tuple[list[list[dict[str, Any]]], str]:
    _ = due_dt
    t0 = time.perf_counter()
    dnf, err = validate_anchor_syntax_strict(anchor_str)
    if dnf is None:
        error_and_exit([("Invalid anchor", err or "anchor syntax error")])
    assert dnf is not None

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
    preset_display = getattr(core, "anchor_preset_display", lambda _expr: None)(anchor_str)
    if preset_display:
        label, text = preset_display
        rows.append((label, f"[white]{text}[/]  {tag}"))
    else:
        rows.append(("Pattern", f"[white]{anchor_str}[/]  {tag}"))
    try:
        rows.append(("Natural", f"[white]{core.describe_anchor_dnf(dnf, task)}[/]"))
    except Exception:
        pass
    try:
        selection = core._import_sibling("position_selection")
        advice = selection.selection_advice_for_dnf(dnf)
        if advice:
            rows.append(("Advice", f"[yellow]{' '.join(advice)}[/]"))
    except Exception:
        pass
    return dnf, mode


def anchor_preview_prepare_omit_dnf(
    task: TaskPayload,
    rows: list[tuple[str, str]],
    *,
    core: Any,
    validate_omit_syntax_strict: Callable[[str | list[list[dict[str, Any]]]], tuple[list[list[dict[str, Any]]] | None, str | None]],
    error_and_exit: Callable[[list[tuple[str, str]]], NoReturn],
) -> Any:
    omit_str = str(task.get("omit") or "").strip()
    omit_file = str(task.get("omit_file") or "").strip()
    omit_dnf = None
    omit_dates: frozenset[Any] = frozenset()
    if omit_str:
        dnf, err = validate_omit_syntax_strict(omit_str)
        if dnf is None:
            error_and_exit([("Invalid omit", err or "omit syntax error")])
        omit_dnf = dnf
        preset_display = getattr(core, "omit_preset_display", lambda _expr: None)(omit_str)
        if preset_display:
            label, text = preset_display
            rows.append((label, f"[white]{text}[/]"))
        else:
            rows.append(("Omit", f"[white]{omit_str}[/]"))
        try:
            anchor_omit = core._import_sibling("anchor_omit")
            omit_expr = core._parser_api.resolve_omit_presets(omit_str)
            omit_norm = anchor_omit.normalize_omit_expr(omit_expr)
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
            omit_dates, _omit_descriptions = omit_files.load_omit_file_data(
                omit_file,
                getattr(core, "OMIT_FILE_DIR", ""),
            )
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
    task: TaskPayload,
    due_day: Any,
    now_local: datetime,
    user_provided_due: bool,
    *,
    root_uuid_from: Callable[[TaskPayload], str | None],
) -> tuple[Any, Any, str]:
    entry_day = None
    if not user_provided_due:
        entry = task.get("entry")
        if isinstance(entry, str) and entry.strip():
            raw_entry = entry.strip()
            try:
                entry_dt = datetime.strptime(raw_entry, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
            except ValueError:
                try:
                    entry_dt = datetime.fromisoformat(raw_entry.replace("Z", "+00:00"))
                except ValueError:
                    entry_dt = None
            if entry_dt is not None:
                if now_local.tzinfo is not None:
                    entry_dt = entry_dt.astimezone(now_local.tzinfo)
                entry_day = entry_dt.date()
    base_local_date = due_day if user_provided_due else (entry_day or now_local.date())
    seed_base = _preview_seed_base(task, root_uuid_from(task) or "preview")
    interval_seed = base_local_date
    return base_local_date, interval_seed, seed_base


def _dnf_uses_astronomy(dnf: Any) -> bool:
    """Return whether a parsed anchor needs an astronomical provider."""
    events = {"sunrise", "sunset", "dawn", "dusk", "moonrise", "moonset"}
    for term in dnf or ():
        for atom in term or ():
            if str(atom.get("typ") or atom.get("type") or "").lower() == "moon":
                return True
            value = (atom.get("mods") or {}).get("t")
            if isinstance(value, str) and value.strip().lower().split("@", 1)[0] in events:
                return True
    return False


def anchor_preview_first_due(
    task: TaskPayload,
    dnf: Any,
    omit_dnf: Any,
    *,
    now_local: datetime,
    due_dt: datetime,
    user_provided_due: bool,
    recurrence_field: str,
    due_hhmm: tuple[int, int],
    interval_seed: Any,
    rows: list[tuple[str, str]],
    prof: Any,
    fmt_dt_local: Callable[[Any], str],
    to_local_cached: Callable[[datetime], datetime],
    scheduler_service: Any,
    error_and_exit: Callable[[list[tuple[str, str]]], NoReturn],
    fmt_local_for_task: Callable[[datetime], str],
) -> tuple[Any, datetime, datetime, Any, tuple[int, int]]:
    def _fmt(dt: Any) -> str:
        return fmt_dt_local(dt)

    fallback_hhmm = due_hhmm if user_provided_due else (9, 0)
    t_first = time.perf_counter()
    inclusive = not user_provided_due
    reference_local = to_local_cached(due_dt) if user_provided_due else now_local
    from .occurrence_outcomes import FoundOccurrence
    from .scheduler_cursor import OccurrenceCursor

    outcome = scheduler_service.next(
        OccurrenceCursor(
            reference_local,
            inclusive=inclusive,
            timezone=getattr(scheduler_service.session.evaluator.context, "timezone", None),
        ),
        fallback_hhmm=fallback_hhmm,
        default_seed_date=interval_seed,
    )
    if isinstance(outcome, FoundOccurrence):
        first_due_local_dt = outcome.local_datetime
    else:
        first_due_local_dt = None
    if first_due_local_dt is None:
        if _dnf_uses_astronomy(dnf):
            message = (
                "No astronomical occurrence could be resolved. Install the optional "
                "'astral' package and configure the astronomy profile with a default_location "
                "and location profile, then run nautical doctor."
            )
            error_and_exit([("Astronomy", message)])
            raise RuntimeError("astronomy preview terminated")
        elif omit_dnf:
            message = "No matching anchor dates found. Omit rules removed every future occurrence."
        else:
            message = (
                "No matching anchor occurrences found after the provided due."
                if user_provided_due
                else "No matching anchor occurrences found."
            )
        error_and_exit([("anchor pattern", message)])
        raise RuntimeError("anchor preview terminated")

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
    dnf: Any,
    due_dt: datetime,
    recurrence_field: str,
    interval_seed: Any,
    seed_base: str,
    omit_dnf: Any,
    to_local_cached: Callable[[datetime], datetime],
    evaluator: Any,
) -> None:
    due_local_date = to_local_cached(due_dt).date()
    first_event = evaluator.next_event_after(
        evaluator.build_local_datetime(due_local_date, (0, 0)),
        fallback_hhmm=(0, 0),
        default_seed_date=interval_seed,
        inclusive=True,
    )
    first_after_due_date = first_event.local_datetime.date() if first_event and first_event.local_datetime else None
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
    core._parser_api.validate_anchor_expr_strict(anchor_str)
    prof.add_ms("anchor:validate_strict", (time.perf_counter() - t_val) * 1000.0)


def anchor_preview_limit_rows(
    rows: list[tuple[str, str]],
    *,
    cpmax: int,
    until_dt: datetime | None,
    exact_until_count: int | None,
    final_until_dt: datetime | None,
    now_utc: datetime,
    fmt_dt_local: Callable[[Any], str],
    human_delta: Callable[[Any, Any, bool], str],
    final_max_dt: datetime | None = None,
) -> None:
    def _fmt(dt: Any) -> str:
        return fmt_dt_local(dt)

    future_counts = []
    if cpmax and cpmax > 0:
        rows.append(("Chain cap", f"[bold yellow]#{cpmax}[/]"))
        future_counts.append(max(0, cpmax - 1))
    if until_dt:
        rows.append(("Chain end point", f"[bold yellow]{_fmt(until_dt)}[/]"))
        if exact_until_count is not None:
            future_counts.append(exact_until_count)
    final_candidates = [dt for dt in (final_max_dt, final_until_dt) if dt is not None]
    if final_candidates:
        last = min(final_candidates)
        rows.append(
            (
                "Last occurrence",
                f"[bright_magenta]{_fmt(last)}[/]  [dim]({human_delta(now_utc, last, True)})[/]",
            )
        )
    if future_counts:
        rows.append(("Future links", f"[white]{min(future_counts)}[/]"))


def _preview_omit_label(
    task: TaskPayload,
    item_local: datetime,
    *,
    omit_description_for_task_date: Callable[[TaskPayload, Any], str | None],
) -> str:
    omit_file = str(task.get("omit_file") or "").strip()
    if not omit_file:
        return "omitted"
    try:
        text = str(omit_description_for_task_date(task, item_local.date()) or "").strip()
    except Exception:
        text = ""
    if not text:
        return "omitted"
    return (text[:14] + "...") if len(text) > 14 else text


def _preview_occurrence_lines(
    events: list[Any],
    *,
    first_due_local_dt: datetime,
    preview_limit: int,
    fmt_dt_local: Callable[[Any], str],
    omit_description_for_task_date: Callable[[TaskPayload, Any], str | None],
    task: TaskPayload,
) -> list[str]:
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    out: list[str] = []
    included_idx = 1
    for event in events:
        if isinstance(event, Occurrence):
            item_local = event.local_datetime
            is_omitted = event.omitted
        else:
            item_local, is_omitted = event
        if item_local is None:
            continue
        if compare_datetimes(item_local, first_due_local_dt) <= 0:
            if not is_omitted and compare_datetimes(item_local, first_due_local_dt) == 0:
                included_idx = 1
            continue
        if is_omitted:
            label = _preview_omit_label(
                task, item_local, omit_description_for_task_date=omit_description_for_task_date
            ).replace('[', '(').replace(']', ')')
            out.append(f"[dim red]·· ×  {fmt_dt_local(item_local.astimezone(timezone.utc))} [italic]({label})[/][/]")
            continue
        included_idx += 1
        if included_idx > preview_limit + 1:
            break
        color = colors[min(included_idx - 2, len(colors) - 1)]
        out.append(f"[{color}]{included_idx} ▸ {fmt_dt_local(item_local.astimezone(timezone.utc))}[/]")
    return out


def _collect_included_with_provider(
    *,
    after_local_dt: datetime,
    inclusive: bool,
    limit: int,
    fallback_hhmm: tuple[int, int],
    default_seed_date: Any,
    max_iterations: int = 512,
    return_occurrences: bool = False,
    scheduler_service: Any,
) -> list[datetime] | list[Occurrence]:
    """Collect included occurrences through the typed provider boundary."""
    stream = _collect_events_with_provider(
        after_local_dt=after_local_dt,
        inclusive=inclusive,
        limit_included=limit,
        fallback_hhmm=fallback_hhmm,
        default_seed_date=default_seed_date,
        max_iterations=max_iterations,
        return_occurrences=True,
        scheduler_service=scheduler_service,
    )
    included = [item for item in stream if not getattr(item, "omitted", False)]
    selected = included[:limit]
    if return_occurrences:
        return OccurrenceBatch(selected, terminal=getattr(stream, "terminal", None))
    return OccurrenceBatch(
        [item.local_datetime for item in selected if item.local_datetime is not None],
        terminal=getattr(stream, "terminal", None),
    )


def _collect_events_with_provider(
    *,
    after_local_dt: datetime,
    inclusive: bool,
    limit_included: int,
    fallback_hhmm: tuple[int, int],
    default_seed_date: Any,
    max_iterations: int = 512,
    return_occurrences: bool = False,
    scheduler_service: Any,
    ) -> list[Any]:
    from .scheduler_cursor import OccurrenceCursor

    result = scheduler_service.collect(
        OccurrenceCursor(
            after_local_dt,
            inclusive=inclusive,
            timezone=scheduler_service.session.evaluator.context.timezone,
        ),
        limit=limit_included,
        count_omitted=False,
        fallback_hhmm=fallback_hhmm,
        default_seed_date=default_seed_date,
        max_iterations=max_iterations,
        max_file_skips=max_iterations,
    )
    collected = list(result.occurrences)
    if return_occurrences:
        return collected
    return OccurrenceBatch(
        [(occurrence.local_datetime, occurrence.omitted) for occurrence in collected if occurrence.local_datetime is not None],
        terminal=result.terminal,
    )


def handle_anchor_file_preview_on_add(
    *,
    task: TaskPayload,
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
    services: AnchorFilePreviewServices,
) -> None:
    append_wait_sched_rows = services.append_wait_sched_rows
    validate_chain_duration_reasonable = services.validate_chain_duration_reasonable
    format_anchor_rows = services.format_anchor_rows
    panel = services.panel
    fmt_local_for_task = services.fmt_local_for_task
    human_delta = services.human_delta
    error_and_exit = services.error_and_exit
    rows: list[tuple[str, str]] = []
    panel_mode = services.panel_mode.strip().lower()
    compact_presentation = panel_mode in {"quiet", "minimal", "line", "text"}
    if services.timezone_fallback_warning(anchor_file_str):
        rows.append(("Warning", "[yellow]Timezone data unavailable; using UTC fallback. Run nautical doctor.[/]"))
    rows.append(("Anchor file", f"[white]{anchor_file_str}[/]  [bold bright_cyan]SKIP[/]"))
    if not compact_presentation:
        rows.append(("Natural", f"[white]{_anchor_file_natural_text(anchor_file_str)}[/]"))
    omit_dnf = services.prepare_omit_dnf(task, rows)
    t_occ = time.perf_counter()
    scheduler_service = services.scheduler_service_for_task(task)
    all_occurrences = _collect_included_with_provider(
        after_local_dt=(services.to_local(due_dt) if user_provided_due else now_local),
        inclusive=False if user_provided_due else True,
        limit=_initial_occurrence_limit(preview_hard_cap, compact_presentation),
        fallback_hhmm=(due_hhmm if user_provided_due else (9, 0)),
        default_seed_date=(services.to_local(due_dt).date() if user_provided_due else now_local.date()),
        scheduler_service=scheduler_service,
    )
    prof.add_ms("anchor_file:occurrences", (time.perf_counter() - t_occ) * 1000.0)
    if not all_occurrences:
        error_and_exit([("anchor_file", "No matching anchor_file occurrences found.")])
        raise RuntimeError("anchor-file preview terminated")

    occurrence_datetimes: list[datetime] = []
    for item in all_occurrences:
        candidate = item.local_datetime if isinstance(item, Occurrence) else item
        if isinstance(candidate, datetime):
            occurrence_datetimes.append(candidate)
    if not occurrence_datetimes:
        error_and_exit([("anchor_file", "Anchor-file occurrences did not contain valid local timestamps.")])
        raise RuntimeError("anchor-file preview terminated")

    due_local_dt = services.to_local(due_dt)
    first_due_local_dt: datetime | None
    if compact_presentation:
        first_due_local_dt = occurrence_datetimes[0]
    elif user_provided_due:
        first_due_local_dt = next((dt for dt in occurrence_datetimes if compare_datetimes(dt, due_local_dt) > 0), None)
        if first_due_local_dt is None:
            error_and_exit([("anchor_file", "No matching anchor_file occurrences found after the provided due.")])
            raise RuntimeError("anchor-file preview terminated")
    else:
        first_due_local_dt = next((dt for dt in occurrence_datetimes if compare_datetimes(dt, now_local) >= 0), None)
        if first_due_local_dt is None:
            error_and_exit([("anchor_file", "No matching anchor_file occurrences found.")])
            raise RuntimeError("anchor-file preview terminated")

    first_due_utc = first_due_local_dt.astimezone(timezone.utc)
    display_first_due_utc = due_dt if user_provided_due else first_due_utc
    if user_provided_due:
        first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
        rows.append((first_label, f"[bold bright_green]{services.fmt_dt_local(display_first_due_utc)}[/]"))
        rows.append(("Next anchor", f"[white]{services.fmt_dt_local(first_due_utc)}[/]"))
    else:
        rows.append(("First due", f"[bold bright_green]{services.fmt_dt_local(first_due_utc)}[/]"))
        task["due"] = fmt_local_for_task(first_due_utc)
        rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))
    services.render_business_calendar_displacement(task, first_due_local_dt, panel=panel)
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
        if compare_datetimes(until_dt, first_due_utc) < 0:
            error_and_exit(
                [
                    ("Invalid chainUntil", "Chain end point is earlier than the first matching anchor occurrence."),
                    ("First due", services.fmt_dt_local(first_due_utc)),
                    ("Chain end point", services.fmt_dt_local(until_dt)),
                    ("Required", "Set chainUntil on or after the first anchor occurrence, or adjust the anchor."),
                ]
            )
        is_reasonable, warn_msg = validate_chain_duration_reasonable(until_dt, now_utc, first_due_utc, "anchor")
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    cpmax = services.coerce_int(task.get("chainMax"), 0)
    exact_until_count = None
    final_until_dt = None
    if not compact_presentation and until_dt:
        until_local = services.to_local(until_dt)
        limited = [dt for dt in occurrence_datetimes if compare_datetimes(dt, until_local) <= 0]
        exact_until_count = max(0, len(limited) - 1)
        if limited:
            final_until_dt = limited[-1].astimezone(timezone.utc)
    if compact_presentation:
        preview_rows = []
    else:
        allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
        allow_by_until = exact_until_count if exact_until_count is not None else 10**9
        preview_limit = max(0, min(upcoming_preview, allow_by_max, allow_by_until, preview_hard_cap))
        event_limit = max(1, preview_limit + 1)
        events = _collect_events_with_provider(
        after_local_dt=first_due_local_dt,
        inclusive=True,
        limit_included=event_limit,
        fallback_hhmm=(due_hhmm if user_provided_due else (9, 0)),
            default_seed_date=first_due_local_dt.date(),
            return_occurrences=True,
        scheduler_service=scheduler_service,
        )
        if until_dt:
            until_local = services.to_local(until_dt)
            events = [
                event
                for event in events
                if _event_on_or_before(event, until_local)
            ]
        preview_rows = _preview_occurrence_lines(
            events,
            first_due_local_dt=first_due_local_dt,
            preview_limit=preview_limit,
            fmt_dt_local=services.fmt_dt_local,
            omit_description_for_task_date=services.omit_description_for_task_date,
            task=task,
        )
        rows.append(("Upcoming", "\n".join(preview_rows) if preview_rows else "[dim]–[/]"))
        rows.append(("Delta", f"[bright_yellow]{human_delta(now_utc, display_first_due_utc, False)}[/]"))
        anchor_preview_limit_rows(
            rows,
            cpmax=cpmax,
            until_dt=until_dt,
            exact_until_count=exact_until_count,
            final_until_dt=final_until_dt,
            now_utc=now_utc,
            fmt_dt_local=services.fmt_dt_local,
            human_delta=human_delta,
        )
    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    panel("⚓︎ Anchor Preview", format_anchor_rows(rows), kind="preview_anchor", task=task)


def _timezone_fallback_warning_needed(core: Any, anchor_str: str, anchor_file_str: str) -> bool:
    from .modify_models import TaskView

    task = TaskView.from_mapping({"anchor": anchor_str, "anchor_file": anchor_file_str})
    return bool(panel_diagnostics.recurrence_timezone_warning(core, task))


def _append_dst_adjustment_row(
    rows: list[tuple[str, str]],
    dnf: Any,
    occurrence_local: datetime,
    *,
    core: Any,
) -> None:
    """Describe a fixed wall-clock slot shifted by a timezone transition."""
    if not dnf or not isinstance(occurrence_local, datetime):
        return
    for term in dnf:
        for atom in term:
            mods = atom.get("mods") or {}
            values = mods.get("time_window_offsets") or mods.get("t")
            if isinstance(values, tuple) and len(values) == 2:
                values = [(0, values[0], values[1])]
            elif isinstance(values, list):
                values = [
                    ((0, value[0], value[1]) if len(value) == 2 else value)
                    for value in values
                    if isinstance(value, tuple) and len(value) in (2, 3)
                ]
            else:
                continue
            for day_offset, hour, minute in values:
                requested_day = occurrence_local.date() - timedelta(days=int(day_offset))
                try:
                    resolved_local = core.to_local(
                        core.build_local_datetime(requested_day, (int(hour), int(minute)))
                    )
                except Exception:
                    continue
                if compare_datetimes(resolved_local, occurrence_local) != 0:
                    continue
                requested = (requested_day, int(hour), int(minute))
                actual = (resolved_local.date(), resolved_local.hour, resolved_local.minute)
                if requested == actual:
                    return
                requested_label = f"{int(hour):02d}:{int(minute):02d}"
                actual_label = f"{resolved_local.hour:02d}:{resolved_local.minute:02d}"
                rows.append(("DST adjusted", f"[yellow]{requested_label} -> {actual_label}[/]"))
                return


def _initial_occurrence_limit(preview_hard_cap: int, compact_presentation: bool) -> int:
    """Only the first match is needed to assign a compact preview's initial due."""
    return 1 if compact_presentation else preview_hard_cap + 16


def handle_anchor_preview_on_add(
    *,
    task: TaskPayload,
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
    max_summary_links: int,
    services: AnchorExpressionPreviewServices,
) -> None:
    root_uuid_from = services.root_uuid_from
    short = services.short
    validate_anchor_mode = services.validate_anchor_mode
    validate_chain_duration_reasonable = services.validate_chain_duration_reasonable
    append_wait_sched_rows = services.append_wait_sched_rows
    anchor_until_summary = services.anchor_until_summary
    to_local_cached = services.to_local_cached
    fmt_local_for_task = services.fmt_local_for_task
    format_anchor_rows = services.format_anchor_rows
    panel = services.panel
    human_delta = services.human_delta
    error_and_exit = services.error_and_exit
    validate_native_until_after_target = services.validate_native_until_after_target
    validate_native_until_anchor_slots = services.validate_native_until_anchor_slots
    append_first_expiration_row = services.append_first_expiration_row
    rows: list[tuple[str, str]] = []
    panel_mode = services.panel_mode.strip().lower()
    compact_presentation = panel_mode in {"quiet", "minimal", "line", "text"}
    for warning in services.panel_warnings(task):
        rows.append(("Warning", f"[yellow]{warning}[/]"))
    dnf = None
    if anchor_str:
        dnf, _ = services.prepare_anchor_dnf(task, anchor_str, due_dt, rows, prof)
    else:
        mode, warn_msg = validate_anchor_mode(task.get("anchor_mode"))
        task["anchor_mode"] = mode
        if warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))
    if anchor_file_str:
        if anchor_str:
            rows.append(("Sources", "[white]anchor + anchor_file[/]"))
            rows.append(("Anchor file", f"[white]{anchor_file_str}[/]"))
            for idx, (label, value) in enumerate(rows):
                if label == "Natural":
                    rows[idx] = ("Natural", f"[white]{services.describe_anchor_natural(task, dnf, anchor_file_str)}[/]")
                    break
        else:
            rows.append(("Anchor file", f"[white]{anchor_file_str}[/]"))
            rows.append(("Natural", f"[white]{_anchor_file_natural_text(anchor_file_str)}[/]"))

    omit_dnf = services.prepare_omit_dnf(task, rows)
    _base_local_date, interval_seed, seed_base = anchor_preview_seed_context(
        task,
        due_day,
        now_local,
        user_provided_due,
        root_uuid_from=root_uuid_from,
    )

    merged = bool(dnf and anchor_file_str)
    fallback_hhmm = due_hhmm if user_provided_due else (9, 0)
    try:
        scheduler_service = services.scheduler_service_for_task(task)
        recurrence_evaluator = scheduler_service.session.evaluator
    except Exception as exc:
        error_and_exit(
            [
                (
                    "Recurrence evaluator",
                    "Could not initialize the shared recurrence evaluator: "
                    f"{type(exc).__name__}: {exc}",
                ),
                (
                    "Fix",
                    "Check the anchor, timezone, astronomy, and business-calendar configuration.",
                ),
            ]
        )
        raise RuntimeError("recurrence evaluator initialization failed") from exc

    if not dnf:
        occurrences = _collect_included_with_provider(
            after_local_dt=(to_local_cached(due_dt) if user_provided_due else now_local),
            inclusive=not user_provided_due,
            limit=_initial_occurrence_limit(preview_hard_cap, compact_presentation),
            fallback_hhmm=fallback_hhmm,
            default_seed_date=interval_seed,
            scheduler_service=scheduler_service,
        )
        if not occurrences:
            error_and_exit([("anchor_file", "No matching anchor_file occurrences found.")])
            raise RuntimeError("anchor-file preview terminated")
        first_due_local_dt = occurrences[0]
        if not isinstance(first_due_local_dt, datetime):
            error_and_exit([("anchor_file", "Anchor-file occurrence has no valid local timestamp.")])
            raise RuntimeError("anchor-file preview terminated")
        first_due_utc = first_due_local_dt.astimezone(timezone.utc)
        display_first_due_utc = due_dt if user_provided_due else first_due_utc
        first_date_local = first_due_local_dt.date()
        first_hhmm = (first_due_local_dt.hour, first_due_local_dt.minute)
        if user_provided_due:
            first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
            rows.append((first_label, f"[bold bright_green]{services.fmt_dt_local(display_first_due_utc)}[/]"))
            rows.append(("Next anchor", f"[white]{services.fmt_dt_local(first_due_utc)}[/]"))
        else:
            rows.append(("First due", f"[bold bright_green]{services.fmt_dt_local(first_due_utc)}[/]"))
            task["due"] = fmt_local_for_task(first_due_utc)
            rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))
    elif not merged:
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
            rows=rows,
            prof=prof,
            fmt_dt_local=services.fmt_dt_local,
            to_local_cached=to_local_cached,
            scheduler_service=scheduler_service,
            error_and_exit=error_and_exit,
            fmt_local_for_task=fmt_local_for_task,
        )
    else:
        occurrences = _collect_included_with_provider(
            after_local_dt=(to_local_cached(due_dt) if user_provided_due else now_local),
            inclusive=not user_provided_due,
            limit=_initial_occurrence_limit(preview_hard_cap, compact_presentation),
            fallback_hhmm=fallback_hhmm,
            default_seed_date=interval_seed,
            scheduler_service=scheduler_service,
        )
        if not occurrences:
            error_and_exit([("anchor pattern", "No matching anchor occurrences found.")])
            raise RuntimeError("anchor preview terminated")
        first_due_local_dt = occurrences[0]
        if not isinstance(first_due_local_dt, datetime):
            error_and_exit([("anchor pattern", "Anchor occurrence has no valid local timestamp.")])
            raise RuntimeError("anchor preview terminated")
        first_due_utc = first_due_local_dt.astimezone(timezone.utc)
        display_first_due_utc = due_dt if user_provided_due else first_due_utc
        first_date_local = first_due_local_dt.date()
        first_hhmm = (first_due_local_dt.hour, first_due_local_dt.minute)
        if user_provided_due:
            first_label = "First scheduled" if recurrence_field == "scheduled" else "First due"
            rows.append((first_label, f"[bold bright_green]{services.fmt_dt_local(display_first_due_utc)}[/]"))
            rows.append(("Next anchor", f"[white]{services.fmt_dt_local(first_due_utc)}[/]"))
        else:
            rows.append(("First due", f"[bold bright_green]{services.fmt_dt_local(first_due_utc)}[/]"))
            task["due"] = fmt_local_for_task(first_due_utc)
            rows.append(("[auto-due]", "Due date was not explicitly set; assigned to first anchor match."))

    services.append_dst_adjustment(rows, dnf, first_due_local_dt)

    validate_native_until_after_target(task, display_first_due_utc, recurrence_field)
    validate_native_until_anchor_slots(
        task,
        display_first_due_utc,
        dnf,
        anchor_file_str,
        fallback_hhmm,
    )
    append_first_expiration_row(rows, task, display_first_due_utc, recurrence_field)
    services.render_business_calendar_displacement(task, first_due_local_dt, panel=panel)
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
            evaluator=recurrence_evaluator,
        )

    if until_dt:
        if compare_datetimes(until_dt, first_due_utc) < 0:
            error_and_exit(
                [
                    ("Invalid chainUntil", "Chain end point is earlier than the first matching anchor occurrence."),
                    ("First due", services.fmt_dt_local(first_due_utc)),
                    ("Chain end point", services.fmt_dt_local(until_dt)),
                    ("Required", "Set chainUntil on or after the first anchor occurrence, or adjust the anchor."),
                ]
            )
        is_reasonable, warn_msg = validate_chain_duration_reasonable(
            until_dt,
            now_utc,
            first_due_utc,
            "anchor",
        )
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    cpmax = services.coerce_int(task.get("chainMax"), 0)
    preview: list[str]
    presentation_terminal = None
    exact_until_count = None
    final_until_dt = None
    if compact_presentation:
        exact_until_count = None
        final_until_dt = None
        preview = []
    elif dnf and not merged:
        exact_until_count, final_until_dt = anchor_until_summary(
            dnf,
            until_dt,
            first_date_local,
            first_hhmm,
            interval_seed,
            seed_base,
            omit_dnf=omit_dnf,
            evaluator=recurrence_evaluator,
        )
        allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
        allow_by_until = exact_until_count if exact_until_count is not None else 10**9
        services.lint_and_validate(anchor_str, prof, panel=panel)
        preview_limit = max(0, min(upcoming_preview, allow_by_max, allow_by_until, preview_hard_cap))
        _t_prev = time.perf_counter()
        events = _collect_events_with_provider(
            after_local_dt=first_due_local_dt,
            inclusive=True,
            limit_included=max(1, preview_limit + 1),
            fallback_hhmm=first_hhmm,
            default_seed_date=interval_seed,
            return_occurrences=True,
            scheduler_service=scheduler_service,
        )
        preview_terminal = getattr(events, "terminal", None)
        if preview_terminal is not None and preview_terminal.is_date_limit:
            rows.append(
                (
                    "Note",
                    f"[yellow]{occurrence_exhaustion_message(preview_terminal)}; "
                    "the occurrences shown above are the final representable matches.[/]",
                )
            )
        preview = _preview_occurrence_lines(
            events,
            first_due_local_dt=first_due_local_dt,
            preview_limit=preview_limit,
            fmt_dt_local=services.fmt_dt_local,
            omit_description_for_task_date=services.omit_description_for_task_date,
            task=task,
        )
        prof.add_ms("anchor:preview_occurrences", (time.perf_counter() - _t_prev) * 1000.0)
    elif not compact_presentation:
        all_occurrences = _collect_included_with_provider(
            after_local_dt=first_due_local_dt,
            inclusive=True,
            limit=preview_hard_cap + 24,
            fallback_hhmm=first_hhmm,
            default_seed_date=interval_seed,
            scheduler_service=scheduler_service,
        )
        presentation_terminal = getattr(all_occurrences, "terminal", None)
        if until_dt:
            until_local = services.to_local(until_dt)
            limited: list[datetime] = []
            for event in all_occurrences:
                event_dt = _event_datetime(event)
                if event_dt is not None and _event_on_or_before(event_dt, until_local):
                    limited.append(event_dt)
            exact_until_count = max(0, len(limited) - 1)
            if limited:
                final_until_dt = limited[-1].astimezone(timezone.utc)
        allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
        allow_by_until = exact_until_count if exact_until_count is not None else 10**9
        preview_limit = max(0, min(upcoming_preview, allow_by_max, allow_by_until, preview_hard_cap))
        event_limit = max(1, preview_limit + 1)
        events = _collect_events_with_provider(
            after_local_dt=first_due_local_dt,
            inclusive=True,
            limit_included=event_limit,
            fallback_hhmm=first_hhmm,
            default_seed_date=interval_seed,
            return_occurrences=True,
            scheduler_service=scheduler_service,
        )
        event_terminal = getattr(events, "terminal", None)
        if event_terminal is not None:
            presentation_terminal = event_terminal
        if until_dt:
            until_local = services.to_local(until_dt)
            events = [
                event
                for event in events
                if _event_on_or_before(event, until_local)
            ]
        preview = _preview_occurrence_lines(
            events,
            first_due_local_dt=first_due_local_dt,
            preview_limit=preview_limit,
            fmt_dt_local=services.fmt_dt_local,
            omit_description_for_task_date=services.omit_description_for_task_date,
            task=task,
        )
        if presentation_terminal is not None and presentation_terminal.is_date_limit:
            rows.append(
                (
                    "Note",
                    f"[yellow]{occurrence_exhaustion_message(presentation_terminal)}; "
                    "the preview shows the final representable matches.[/]",
                )
            )
        if anchor_str:
            services.lint_and_validate(anchor_str, prof, panel=panel)

    if not compact_presentation:
        rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
        rows.append(("Delta", f"[bright_yellow]{human_delta(now_utc, display_first_due_utc, bool(dnf and services.expr_has_m_or_y(dnf)))}[/]"))

    final_max_dt = None
    future_needed = max(0, cpmax - 1)
    if not compact_presentation and cpmax == 1:
        final_max_dt = display_first_due_utc
    elif not compact_presentation and future_needed and future_needed <= max_summary_links:
        future_for_max = _collect_included_with_provider(
            after_local_dt=first_due_local_dt,
            inclusive=user_provided_due,
            limit=future_needed,
            fallback_hhmm=first_hhmm,
            default_seed_date=interval_seed,
            scheduler_service=scheduler_service,
        )
        if len(future_for_max) == future_needed:
            final_max_local = _event_datetime(future_for_max[-1])
            if final_max_local is not None:
                final_max_dt = final_max_local.astimezone(timezone.utc)

    if not compact_presentation:
        anchor_preview_limit_rows(
            rows,
            cpmax=cpmax,
            until_dt=until_dt,
            exact_until_count=exact_until_count,
            final_until_dt=final_until_dt,
            now_utc=now_utc,
            fmt_dt_local=services.fmt_dt_local,
            human_delta=human_delta,
            final_max_dt=final_max_dt,
        )

    if anchor_str and "rand" in anchor_str.lower():
        base = short(root_uuid_from(task))
        rows.append(("Rand", f"[dim italic]Preview uses provisional seed; final picks are chain-bound to {base}[/]"))

    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    formatted_rows = format_anchor_rows(rows)
    _t_panel = time.perf_counter()
    panel("⚓︎ Anchor Preview", formatted_rows, kind="preview_anchor", task=task)
    prof.add_ms("render:anchor_panel", (time.perf_counter() - _t_panel) * 1000.0)
