from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .callback_ports import CallbackPort
from .scheduler_models import OccurrenceSearchExhausted, occurrence_exhaustion_message
from .timeutil import compare_datetimes
from .task_models import TaskObservation, TaskPayload


@dataclass(frozen=True, slots=True)
class TimelineProjectionServices:
    max_iterations: int
    collect_prev_two: CallbackPort
    dtparse: CallbackPort
    to_local_cached: CallbackPort
    safe_parse_datetime: CallbackPort
    omit_dnf_from_parent: CallbackPort
    omit_description_for_date: Callable[[Any, Any], str | None] | None
    recurrence_evaluator_for_task: CallbackPort
    scheduler_service_for_task: CallbackPort


@dataclass(frozen=True, slots=True)
class TimelineFormattingServices:
    future_style_for_chain: CallbackPort
    coerce_int: CallbackPort
    fmt_on_time_delta: CallbackPort
    fmtlocal: CallbackPort
    fmt_dt_local: CallbackPort
    short: CallbackPort
    format_gap: CallbackPort


TimelineItem = tuple[object, Any, TaskPayload, str]


def _build_slot_datetime(day: Any, hhmm: Any) -> datetime:
    return datetime.combine(day, datetime.min.time().replace(hour=int(hhmm[0]), minute=int(hhmm[1])))


def _timeline_seed_base(task: TaskPayload) -> str:
    """Return the stable recurrence identity used by timeline projections."""
    return str(task.get("chainID") or task.get("uuid") or "preview").strip()


def _timeline_omit_label(
    omit_dnf: Any,
    omit_date: Any,
    *,
    omit_description_for_date: Callable[[Any, Any], str | None] | None,
) -> str | None:
    if omit_description_for_date is None:
        return None
    try:
        text = str(omit_description_for_date(omit_dnf, omit_date) or "").strip()
    except Exception:
        return None
    if not text:
        return None
    if len(text) <= 14:
        return text
    return text[:14] + "..."


def _timeline_warning(message: str) -> tuple[object, None, dict[str, Any], str]:
    return ("!", None, {"message": message}, "warning")


def _timeline_styles(
    task: TaskPayload,
    kind: str,
    *,
    future_style_for_chain: Callable[[TaskPayload, str], str],
) -> tuple[str, str, str, str]:
    if kind == "cp":
        prev_style = "dim green"
        cur_style = "spring_green1"
        next_style = "bold yellow"
    else:
        prev_style = "sky_blue3"
        cur_style = "spring_green1"
        next_style = "bold yellow"
    future_style = future_style_for_chain(task, kind)
    return prev_style, cur_style, next_style, future_style


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


def format_gap(prev_dt: Any, next_dt: Any, kind: str = "cp", round_hours: bool = True) -> str:
    """Format the time gap between two timeline items as a compact annotation."""
    if not (prev_dt and next_dt):
        return ""
    gap_seconds = (next_dt - prev_dt).total_seconds()
    if abs(gap_seconds) < 60:
        return ""
    if kind == "cp":
        days = gap_seconds / 86400
        if abs(days) >= 1:
            gap_text = f"{int(days)}d" if days.is_integer() else f"{days:.1f}d"
        else:
            hours = gap_seconds / 3600
            gap_text = f"{hours:.1f}h" if abs(hours) >= 1 else f"{int(gap_seconds / 60)}m"
    else:
        days = gap_seconds / 86400
        if round_hours and abs(days) >= 0.5:
            gap_text = f"{round(days)}d"
        elif abs(days) >= 1:
            gap_text = f"{days:.1f}d"
        else:
            gap_text = f"{gap_seconds / 3600:.0f}h"
    return f" ➔ {gap_text}"


def _timeline_initial_items(
    task: TaskPayload,
    cur_no: int,
    nxt_no: int,
    child_due_utc: Any,
    child_short: str,
    *,
    coerce_int: CallbackPort,
    collect_prev_two: Callable[[TaskPayload], list[TaskObservation]],
    dtparse: Callable[[Any], Any],
) -> list[TimelineItem]:
    items: list[TimelineItem] = []
    prevs = collect_prev_two(task)
    prev_count = len(prevs)
    for idx, observation in enumerate(prevs):
        obj = observation.to_mapping()
        no = coerce_int(obj.get("link"), None) or (cur_no - (prev_count - idx))
        end_dt = dtparse(obj.get("end"))
        items.append((no, end_dt, obj, "prev"))
    cur_end = dtparse(task.get("end"))
    items.append((cur_no, cur_end, task, "current"))
    items.append((nxt_no, child_due_utc, {"uuid": child_short}, "next"))
    return items


def _timeline_future_cp_items(
    task: TaskPayload,
    child_due_utc: datetime,
    *,
    start_no: int,
    allowed_future: int,
    cap_no: int | None,
    max_iterations: int,
    evaluator: Any,
) -> list[tuple[int, datetime, dict[str, Any], str]]:
    cp_str = str(task.get("cp") or "")
    tokens = evaluator.cp_tokens
    if not tokens:
        return []
    cp_tokens = [p.strip() for p in cp_str.split(",")]
    show_interval = len(tokens) > 1 or any(t.get("kind") == "rand" for t in tokens)
    items: list[tuple[int, datetime, dict[str, Any], str]] = []
    fut_dt = child_due_utc
    fut_no = start_no
    iterations = 0
    for _ in range(allowed_future):
        if iterations >= max_iterations:
            break
        iterations += 1
        token_idx = (max(1, fut_no) - 1) % len(tokens)
        td = evaluator.cp_interval_for_link(fut_no)
        if td is None:
            break
        fut_no += 1
        fut_dt = evaluator.project_cp(fut_dt, fut_no - 1)
        if cap_no is not None and fut_no > cap_no:
            break
        meta: dict[str, Any] = {"is_future": True}
        if show_interval:
            step_idx = (max(1, fut_no - 1) - 1) % len(tokens)
            if 0 <= step_idx < len(cp_tokens):
                if tokens[step_idx].get("kind") == "rand":
                    meta["cp_interval"] = _format_td_short(td)
                else:
                    meta["cp_interval"] = cp_tokens[step_idx]
        items.append((fut_no, fut_dt, meta, "future"))
    return items


def _timeline_future_anchor_items(
    task: TaskPayload,
    dnf: Any,
    child_due_utc: datetime,
    *,
    start_no: int,
    allowed_future: int,
    cap_no: int | None,
    to_local_cached: Callable[[datetime], datetime],
    safe_parse_datetime: Callable[[Any], tuple[Any, Any]],
    scheduler_service: Any,
    omit_dnf: Any,
    omit_description_for_date: Callable[[Any, Any], str | None] | None,
    max_iterations: int,
) -> list[tuple[object, Any, dict[str, Any], str]]:
    items: list[tuple[object, Any, dict[str, Any], str]] = []
    fut_no = start_no
    seed_base = _timeline_seed_base(task)
    nxt_local = to_local_cached(child_due_utc)
    fallback_hhmm = (nxt_local.hour, nxt_local.minute)
    due0, _ = safe_parse_datetime(task.get("due"))
    sched0, _ = safe_parse_datetime(task.get("scheduled"))
    default_seed = to_local_cached(due0 or sched0 or child_due_utc).date()
    after_local = nxt_local
    iterations = 0
    actual_future = 0
    iteration_limit_reached = False
    while actual_future < allowed_future:
        if iterations >= max_iterations:
            iteration_limit_reached = True
            break
        iterations += 1
        try:
            from .scheduler_cursor import OccurrenceCursor

            outcome = scheduler_service.collect(
                OccurrenceCursor.strict_after(
                    after_local,
                    timezone=scheduler_service.session.evaluator.context.timezone,
                ),
                limit=1,
                count_omitted=True,
                fallback_hhmm=fallback_hhmm,
                default_seed_date=default_seed,
                max_iterations=max_iterations,
                max_file_skips=max_iterations,
            )
            if outcome.failure is not None:
                raise RuntimeError(outcome.failure.reason or "scheduler lookup failed")
            if outcome.occurrences:
                occurrence = outcome.occurrences[0]
                next_local = occurrence.local_datetime
            elif outcome.terminal is not None:
                raise outcome.terminal
            else:
                next_local = None
        except OccurrenceSearchExhausted as exc:
            if exc.is_date_limit:
                items.append(
                    _timeline_warning(
                        f"Projection ended: {occurrence_exhaustion_message(exc)}"
                    )
                )
            else:
                items.append(
                    _timeline_warning(
                        f"Projection unavailable: {occurrence_exhaustion_message(exc)}"
                    )
                )
            break
        except Exception as exc:
            items.append(_timeline_warning(f"Projection unavailable: {type(exc).__name__}: {exc}"))
            break
        if not next_local:
            break
        fut_dt = next_local.astimezone(timezone.utc)
        after_local = next_local
        if occurrence.omitted:
            items.append(
                (
                    "··",
                    fut_dt,
                    {
                        "is_omit": True,
                        "omit_label": _timeline_omit_label(
                            omit_dnf,
                            next_local.date(),
                            omit_description_for_date=omit_description_for_date,
                        ),
                    },
                    "omitted",
                )
            )
            continue
        fut_no += 1
        if cap_no is not None and fut_no > cap_no:
            break
        items.append((fut_no, fut_dt, {"is_future": True}, "future"))
        actual_future += 1
    if iteration_limit_reached:
        items.append(_timeline_warning("Projection incomplete: iteration limit reached."))
    return items


def _timeline_omitted_before_next_anchor_items(
    task: TaskPayload,
    dnf: Any,
    child_due_utc: datetime,
    *,
    dtparse: Callable[[Any], Any],
    to_local_cached: Callable[[datetime], datetime],
    safe_parse_datetime: Callable[[Any], tuple[Any, Any]],
    scheduler_service: Any,
    omit_dnf: Any,
    omit_description_for_date: Callable[[Any, Any], str | None] | None,
    max_iterations: int,
) -> list[tuple[object, Any, dict[str, Any], str]]:
    if not omit_dnf:
        return []
    cur_end = dtparse(task.get("end"))
    if not cur_end:
        return []

    items: list[tuple[object, Any, dict[str, Any], str]] = []
    seed_base = _timeline_seed_base(task)
    child_local = to_local_cached(child_due_utc)
    after_local = to_local_cached(cur_end)
    fallback_hhmm = (child_local.hour, child_local.minute)
    due0, _ = safe_parse_datetime(task.get("due"))
    sched0, _ = safe_parse_datetime(task.get("scheduled"))
    default_seed = to_local_cached(due0 or sched0 or child_due_utc).date()
    try:
        from .scheduler_cursor import OccurrenceCursor

        result = scheduler_service.collect(
            OccurrenceCursor.strict_after(
                after_local,
                timezone=scheduler_service.session.evaluator.context.timezone,
            ),
            limit=max_iterations,
            count_omitted=True,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed,
            max_iterations=max_iterations,
            max_file_skips=max_iterations,
        )
        for occurrence in result.occurrences:
            next_local = occurrence.local_datetime
            if next_local is None or compare_datetimes(next_local, child_local) >= 0:
                break
            if occurrence.omitted:
                items.append(
                    (
                        "··",
                        next_local.astimezone(timezone.utc),
                        {
                            "is_omit": True,
                            "omit_label": _timeline_omit_label(
                                omit_dnf,
                                next_local.date(),
                                omit_description_for_date=omit_description_for_date,
                            ),
                        },
                        "omitted",
                    )
                )
        if result.terminal is not None and not result.occurrences:
            items.append(_timeline_warning(f"Projection ended: {occurrence_exhaustion_message(result.terminal)}"))
        return items
    except Exception as exc:
        return [_timeline_warning(f"Projection unavailable: {type(exc).__name__}: {exc}")]


def _timeline_no_text(no: object) -> str:
    return f"{str(no):>2}"


def _timeline_base_line(
    no: object,
    dt: Any,
    obj: TaskPayload,
    item_type: str,
    *,
    task: TaskPayload,
    cap_no: int | None,
    prev_style: str,
    cur_style: str,
    next_style: str,
    future_style: str,
    fmt_dt_local: Callable[[Any], str],
    dtparse: Callable[[Any], Any],
    fmt_on_time_delta: Callable[[Any, Any], str],
    fmtlocal: Callable[[Any], str],
    short: Callable[[Any], str],
) -> str:
    no_text = _timeline_no_text(no)
    if item_type == "prev":
        end_dt = dtparse(obj.get("end"))
        due_dt = dtparse(obj.get("due"))
        delta = fmt_on_time_delta(due_dt, end_dt)
        end_s = fmtlocal(end_dt) if end_dt else "(no end)"
        short_id = short(obj.get("uuid"))
        return f"[{prev_style}]{no_text} {'✓':<2}{end_s} {short_id} {delta}[/]"

    if item_type == "current":
        cur_end = dtparse(task.get("end"))
        cur_due = dtparse(task.get("due"))
        cur_delta = fmt_on_time_delta(cur_due, cur_end)
        cur_end_s = fmtlocal(cur_end) if cur_end else "(no end)"
        return f"[{cur_style}]{no_text} {'✓':<2}{cur_end_s} {short(task.get('uuid'))} {cur_delta}[/]"

    if item_type == "next":
        is_last = cap_no is not None and no == cap_no
        next_text = f"{no_text} {'►':<2}{fmt_dt_local(dt)} {short(obj.get('uuid'))}"
        if is_last:
            return f"[{next_style}]{next_text} [bold red](last link)[/][/]"
        return f"[{next_style}]{next_text}[/]"

    if item_type == "omitted":
        omit_label = str(obj.get("omit_label") or "").strip()
        if omit_label:
            omit_label = omit_label.replace("[", "(").replace("]", ")")
        else:
            omit_label = "omitted"
        return f"[dim red]{no_text} {'×':<2}{fmt_dt_local(dt)} [italic]({omit_label})[/][/]"

    if item_type == "warning":
        message = str(obj.get("message") or "Timeline projection unavailable")
        message = message.replace("[", "(").replace("]", ")")
        return f"[bright_yellow]{no_text} {'⚠':<2}{message}[/]"

    is_last = cap_no is not None and no == cap_no
    future_text = f"{no_text} {'»':<2}{fmt_dt_local(dt)}"
    cp_interval = str(obj.get("cp_interval") or "").strip()
    if cp_interval:
        future_text = f"{future_text} [dim]({cp_interval})[/]"
    if is_last:
        return f"[{future_style}]{future_text} [bold red](last link)[/][/]"
    return f"[{future_style}]{future_text}[/]"


def _timeline_with_gap(
    base_line: str,
    *,
    idx: int,
    items: list[TimelineItem],
    show_gaps: bool,
    kind: str,
    round_anchor_gaps: bool,
    format_gap: Callable[[Any, Any, str, bool], str],
) -> str:
    if not show_gaps or idx >= len(items) - 1:
        return base_line
    dt = items[idx][1]
    next_dt = items[idx + 1][1]
    if not (dt and next_dt):
        return base_line
    gap_text = format_gap(dt, next_dt, kind, round_anchor_gaps)
    if not gap_text:
        return base_line
    return f"{base_line}{gap_text}"


def anchor_file_timeline_lines(
    task: TaskPayload,
    child_due_utc: datetime,
    child_short: str,
    *,
    next_count: int,
    cap_no: int | None,
    cur_no: int | None,
    show_gaps: bool,
    round_anchor_gaps: bool,
    coerce_int: CallbackPort,
    fmt_dt_local: Callable[[Any], str],
    max_iterations: int,
    future_style_for_chain: Callable[[TaskPayload, str], str],
    collect_prev_two: Callable[[TaskPayload], list[TaskObservation]],
    dtparse: Callable[[Any], Any],
    fmt_on_time_delta: Callable[[Any, Any], str],
    fmtlocal: Callable[[Any], str],
    short: Callable[[Any], str],
    to_local_cached: Callable[[datetime], datetime],
    scheduler_service: Any,
    evaluator: Any,
    omit_dnf: Any,
    omit_description_for_date: Callable[[Any, Any], str | None] | None,
    format_gap: Callable[[Any, Any, str, bool], str],
) -> list[str]:
    """Project anchor-file events and render their timeline rows."""
    child_local = to_local_cached(child_due_utc)
    fallback_hhmm = (child_local.hour, child_local.minute)
    default_seed = child_local.date()
    projection_warning = None
    try:
        from nautical_core.scheduler_cursor import OccurrenceCursor

        result = scheduler_service.collect(
            OccurrenceCursor(
                child_local,
                inclusive=True,
                timezone=evaluator.context.timezone,
            ),
            limit=max(8, next_count + 6),
            count_omitted=True,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed,
            max_iterations=max_iterations,
            max_file_skips=max_iterations,
        )
        events = [occurrence for occurrence in result.occurrences if occurrence.local_datetime is not None]
    except Exception as exc:
        events = []
        projection_warning = _timeline_warning(
            f"Projection unavailable: {type(exc).__name__}: {exc}"
        )

    cur_no = coerce_int(task.get("link") if cur_no is None else cur_no, 1)
    nxt_no = cur_no + 1
    allowed_future = next_count if cap_no is None else max(0, min(next_count, cap_no - nxt_no))
    prev_style, cur_style, next_style, future_style = _timeline_styles(
        task,
        "anchor",
        future_style_for_chain=future_style_for_chain,
    )
    items: list[TimelineItem] = _timeline_initial_items(
        task,
        cur_no,
        nxt_no,
        child_due_utc,
        child_short,
        coerce_int=coerce_int,
        collect_prev_two=collect_prev_two,
        dtparse=dtparse,
    )
    if projection_warning is not None:
        items.append(projection_warning)
    fut_no = nxt_no
    actual_future = 0
    for occurrence in events:
        item_local = occurrence.local_datetime
        if item_local is None:
            continue
        item_utc = item_local.astimezone(timezone.utc)
        if compare_datetimes(item_utc, child_due_utc) <= 0:
            continue
        if occurrence.omitted:
            items.append(
                (
                    "··",
                    item_utc,
                    {
                        "is_omit": True,
                        "omit_label": (
                            _timeline_omit_label(
                                omit_dnf,
                                item_local.date(),
                                omit_description_for_date=omit_description_for_date,
                            )
                            if omit_dnf
                            else None
                        ),
                    },
                    "omitted",
                )
            )
            continue
        fut_no += 1
        if cap_no is not None and fut_no > cap_no:
            break
        items.append((fut_no, item_utc, {"is_future": True}, "future"))
        actual_future += 1
        if actual_future >= allowed_future:
            break

    lines: list[str] = []
    for index, (number, dt, obj, item_type) in enumerate(items):
        base_line = _timeline_base_line(
            number,
            dt,
            obj,
            item_type,
            task=task,
            cap_no=cap_no,
            prev_style=prev_style,
            cur_style=cur_style,
            next_style=next_style,
            future_style=future_style,
            fmt_dt_local=fmt_dt_local,
            dtparse=dtparse,
            fmt_on_time_delta=fmt_on_time_delta,
            fmtlocal=fmtlocal,
            short=short,
        )
        lines.append(
            _timeline_with_gap(
                base_line,
                idx=index,
                items=items,
                show_gaps=show_gaps,
                kind="anchor",
                round_anchor_gaps=round_anchor_gaps,
                format_gap=format_gap,
            )
        )
    return lines


def timeline_lines(
    kind: str,
    task: TaskPayload,
    child_due_utc: datetime,
    child_short: str,
    dnf: Any,
    *,
    next_count: int = 3,
    cap_no: int | None = None,
    cur_no: int | None = None,
    show_gaps: bool = True,
    round_anchor_gaps: bool = True,
    projection: TimelineProjectionServices,
    formatting: TimelineFormattingServices,
    scheduler_service: Any | None,
    omit_dnf: Any,
    evaluator: Any | None,
) -> list[str]:
    cur_no = formatting.coerce_int(task.get("link") if cur_no is None else cur_no, 1)
    nxt_no = cur_no + 1
    allowed_future = next_count if cap_no is None else max(0, min(next_count, cap_no - nxt_no))
    prev_style, cur_style, next_style, future_style = _timeline_styles(
        task,
        kind,
        future_style_for_chain=formatting.future_style_for_chain,
    )
    items: list[TimelineItem] = _timeline_initial_items(
        task,
        cur_no,
        nxt_no,
        child_due_utc,
        child_short,
        coerce_int=formatting.coerce_int,
        collect_prev_two=projection.collect_prev_two,
        dtparse=projection.dtparse,
    )
    if kind == "anchor":
        omitted_before_next = _timeline_omitted_before_next_anchor_items(
            task,
            dnf,
            child_due_utc,
            dtparse=projection.dtparse,
            to_local_cached=projection.to_local_cached,
            safe_parse_datetime=projection.safe_parse_datetime,
            scheduler_service=scheduler_service,
            omit_dnf=omit_dnf,
            omit_description_for_date=projection.omit_description_for_date,
            max_iterations=projection.max_iterations,
        )
        if omitted_before_next:
            items = items[:-1] + omitted_before_next + items[-1:]
    if allowed_future > 0:
        if kind == "cp":
            items.extend(
                _timeline_future_cp_items(
                    task,
                    child_due_utc,
                    start_no=nxt_no,
                    allowed_future=allowed_future,
                    cap_no=cap_no,
                    max_iterations=projection.max_iterations,
                    evaluator=evaluator,
                )
            )
        else:
            items.extend(
                _timeline_future_anchor_items(
                    task,
                    dnf,
                    child_due_utc,
                    start_no=nxt_no,
                    allowed_future=allowed_future,
                    cap_no=cap_no,
                    to_local_cached=projection.to_local_cached,
                    safe_parse_datetime=projection.safe_parse_datetime,
                    scheduler_service=scheduler_service,
                    omit_dnf=omit_dnf,
                    omit_description_for_date=projection.omit_description_for_date,
                    max_iterations=projection.max_iterations,
                )
            )

    lines: list[str] = []
    for i, (no, dt, obj, item_type) in enumerate(items):
        base_line = _timeline_base_line(
            no,
            dt,
            obj,
            item_type,
            task=task,
            cap_no=cap_no,
            prev_style=prev_style,
            cur_style=cur_style,
            next_style=next_style,
            future_style=future_style,
            fmt_dt_local=formatting.fmt_dt_local,
            dtparse=projection.dtparse,
            fmt_on_time_delta=formatting.fmt_on_time_delta,
            fmtlocal=formatting.fmtlocal,
            short=formatting.short,
        )
        lines.append(
            _timeline_with_gap(
                base_line,
                idx=i,
                items=items,
                show_gaps=show_gaps,
                kind=kind,
                round_anchor_gaps=round_anchor_gaps,
                format_gap=formatting.format_gap,
            )
        )
    return lines


def timeline_lines_for_task(
    kind: str,
    task: TaskPayload,
    child_due_utc: datetime,
    child_short: str,
    dnf: Any,
    *,
    next_count: int = 3,
    cap_no: int | None = None,
    cur_no: int | None = None,
    show_gaps: bool = True,
    round_anchor_gaps: bool = True,
    projection: TimelineProjectionServices,
    formatting: TimelineFormattingServices,
) -> list[str]:
    """Resolve recurrence projection independently from rendering services."""
    if kind == "anchor_file" or (kind == "anchor" and (task.get("anchor_file") or "").strip()):
        _omit_expr, omit_dnf = projection.omit_dnf_from_parent(task)
        scheduler_service = projection.scheduler_service_for_task(task)
        evaluator = scheduler_service.session.evaluator
        return anchor_file_timeline_lines(
            task,
            child_due_utc,
            child_short,
            next_count=next_count,
            cap_no=cap_no,
            cur_no=cur_no,
            show_gaps=show_gaps,
            round_anchor_gaps=round_anchor_gaps,
            coerce_int=formatting.coerce_int,
            fmt_dt_local=formatting.fmt_dt_local,
            max_iterations=projection.max_iterations,
            future_style_for_chain=formatting.future_style_for_chain,
            collect_prev_two=projection.collect_prev_two,
            dtparse=projection.dtparse,
            fmt_on_time_delta=formatting.fmt_on_time_delta,
            fmtlocal=formatting.fmtlocal,
            short=formatting.short,
            to_local_cached=projection.to_local_cached,
            scheduler_service=scheduler_service,
            evaluator=evaluator,
            omit_dnf=omit_dnf,
            omit_description_for_date=projection.omit_description_for_date if omit_dnf else None,
            format_gap=formatting.format_gap,
        )

    _omit_expr, omit_dnf = projection.omit_dnf_from_parent(task) if kind == "anchor" else ("", None)
    scheduler_service = projection.scheduler_service_for_task(task) if kind == "anchor" else None
    evaluator = (
        scheduler_service.session.evaluator
        if scheduler_service is not None
        else (projection.recurrence_evaluator_for_task(task) if kind == "cp" else None)
    )
    return timeline_lines(
        kind,
        task,
        child_due_utc,
        child_short,
        dnf,
        next_count=next_count,
        cap_no=cap_no,
        cur_no=cur_no,
        show_gaps=show_gaps,
        round_anchor_gaps=round_anchor_gaps,
        projection=projection,
        formatting=formatting,
        scheduler_service=scheduler_service,
        omit_dnf=omit_dnf,
        evaluator=evaluator,
    )
