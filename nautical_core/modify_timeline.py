from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable

from .scheduler_models import OccurrenceSearchExhausted, occurrence_exhaustion_message
from .timeutil import compare_datetimes


def _timeline_seed_base(task: dict[str, Any]) -> str:
    """Return the stable recurrence identity used by timeline projections."""
    from .recurrence_context import RecurrenceContext

    return RecurrenceContext.from_task(task, fallback_chain_id="preview").seed_base


def _timeline_omit_label(
    omit_dnf,
    omit_date,
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
    task: dict[str, Any],
    kind: str,
    *,
    future_style_for_chain: Callable[[dict[str, Any], str], str],
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
    task: dict[str, Any],
    cur_no: int,
    nxt_no: int,
    child_due_utc: Any,
    child_short: str,
    *,
    core: Any,
    collect_prev_two: Callable[[dict[str, Any]], list[dict[str, Any]]],
    dtparse: Callable[[Any], Any],
) -> list[tuple[int, Any, dict[str, Any], str]]:
    items: list[tuple[int, Any, dict[str, Any], str]] = []
    prevs = collect_prev_two(task)
    prev_count = len(prevs)
    for idx, obj in enumerate(prevs):
        no = core.coerce_int(obj.get("link"), None) or (cur_no - (prev_count - idx))
        end_dt = dtparse(obj.get("end"))
        items.append((no, end_dt, obj, "prev"))
    cur_end = dtparse(task.get("end"))
    items.append((cur_no, cur_end, task, "current"))
    items.append((nxt_no, child_due_utc, {"uuid": child_short}, "next"))
    return items


def _timeline_future_cp_items(
    task: dict[str, Any],
    child_due_utc: datetime,
    *,
    start_no: int,
    allowed_future: int,
    cap_no: int | None,
    core: Any,
    tolocal: Callable[[datetime], datetime],
    max_iterations: int,
    evaluator: Any | None = None,
) -> list[tuple[int, datetime, dict[str, Any], str]]:
    cp_str = str(task.get("cp") or "")
    if evaluator is None:
        from .scheduler_service import SchedulerService
        from .recurrence_context import RecurrenceContext

        evaluator = SchedulerService.from_task(
            task,
            context=RecurrenceContext.from_task(
                task,
                fallback_chain_id=task.get("uuid") or "preview",
                timezone=getattr(core, "_LOCAL_TZ", None),
                astronomy_config=getattr(core, "ASTRONOMY_CONFIG", None),
                anchor_file_dir=getattr(core, "ANCHOR_FILE_DIR", ""),
            ),
        ).session.evaluator
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
    task: dict[str, Any],
    dnf: Any,
    child_due_utc: datetime,
    *,
    start_no: int,
    allowed_future: int,
    cap_no: int | None,
    to_local_cached: Callable[[datetime], datetime],
    safe_parse_datetime: Callable[[Any], tuple[Any, Any]],
    next_occurrence_after_local_dt: Callable[..., Any],
    scheduler_service: Any | None = None,
    omit_dnf,
    omit_expr_fires_on_date: Callable[..., bool] | None,
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
    provider = None
    if scheduler_service is None:
        from .occurrence_provider import AnchorOccurrenceProvider

        provider = AnchorOccurrenceProvider(
            lambda value: next_occurrence_after_local_dt(
                dnf,
                value,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=None,
                fallback_hhmm=fallback_hhmm,
            ),
        )
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
            if scheduler_service is not None:
                from .occurrence_outcomes import ExhaustedOccurrence, FoundOccurrence
                from .scheduler_cursor import OccurrenceCursor

                outcome = scheduler_service.next(
                    OccurrenceCursor.strict_after(
                        after_local,
                        timezone=scheduler_service.session.evaluator.context.timezone,
                    ),
                    fallback_hhmm=fallback_hhmm,
                    default_seed_date=default_seed,
                )
                if isinstance(outcome, FoundOccurrence):
                    next_local = outcome.local_datetime
                elif isinstance(outcome, ExhaustedOccurrence):
                    raise outcome.error
                elif getattr(outcome, "status", "") in {"unavailable", "invalid"}:
                    raise RuntimeError(getattr(outcome, "reason", "scheduler lookup failed"))
                else:
                    next_local = None
            else:
                occurrence = provider.next_after(
                    after_local,
                    build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                    to_local=lambda value: value,
                )
                next_local = occurrence.local_datetime if occurrence is not None else None
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
        if omit_dnf and omit_expr_fires_on_date is not None:
            try:
                if omit_expr_fires_on_date(
                    omit_dnf,
                    next_local.date(),
                    default_seed,
                    seed_base,
                ):
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
            except Exception as exc:
                items.append(_timeline_warning(f"Omit evaluation unavailable: {type(exc).__name__}: {exc}"))
                break
        fut_no += 1
        if cap_no is not None and fut_no > cap_no:
            break
        items.append((fut_no, fut_dt, {"is_future": True}, "future"))
        actual_future += 1
    if iteration_limit_reached:
        items.append(_timeline_warning("Projection incomplete: iteration limit reached."))
    return items


def _timeline_omitted_before_next_anchor_items(
    task: dict[str, Any],
    dnf: Any,
    child_due_utc: datetime,
    *,
    dtparse: Callable[[Any], Any],
    to_local_cached: Callable[[datetime], datetime],
    safe_parse_datetime: Callable[[Any], tuple[Any, Any]],
    next_occurrence_after_local_dt: Callable[..., Any],
    scheduler_service: Any | None = None,
    omit_dnf,
    omit_expr_fires_on_date: Callable[..., bool] | None,
    omit_description_for_date: Callable[[Any, Any], str | None] | None,
    max_iterations: int,
) -> list[tuple[object, Any, dict[str, Any], str]]:
    if not omit_dnf or omit_expr_fires_on_date is None:
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
    if scheduler_service is not None:
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
    from .occurrence_provider import AnchorOccurrenceProvider

    provider = AnchorOccurrenceProvider(
        lambda value: next_occurrence_after_local_dt(
            dnf,
            value,
            default_seed_date=default_seed,
            seed_base=seed_base,
            omit_dnf=None,
            fallback_hhmm=fallback_hhmm,
        ),
    )
    iterations = 0
    iteration_limit_reached = False
    while True:
        if iterations >= max_iterations:
            iteration_limit_reached = True
            break
        iterations += 1
        try:
            occurrence = provider.next_after(
                after_local,
                build_local_datetime=lambda day, hhmm: datetime.combine(day, hhmm),
                to_local=lambda value: value,
            )
            next_local = occurrence.local_datetime if occurrence is not None else None
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
        if not next_local or compare_datetimes(next_local, child_local) >= 0:
            break
        after_local = next_local
        try:
            if omit_expr_fires_on_date(
                omit_dnf,
                next_local.date(),
                default_seed,
                seed_base,
            ):
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
        except Exception as exc:
            items.append(_timeline_warning(f"Omit evaluation unavailable: {type(exc).__name__}: {exc}"))
            break
    if iteration_limit_reached and (not items or items[-1][3] != "warning"):
        items.append(_timeline_warning("Projection incomplete: iteration limit reached."))
    return items


def _timeline_no_text(no: object) -> str:
    return f"{str(no):>2}"


def _timeline_base_line(
    no: object,
    dt: Any,
    obj: dict[str, Any],
    item_type: str,
    *,
    task: dict[str, Any],
    cap_no: int | None,
    prev_style: str,
    cur_style: str,
    next_style: str,
    future_style: str,
    core: Any,
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
        next_text = f"{no_text} {'►':<2}{core.fmt_dt_local(dt)} {short(obj.get('uuid'))}"
        if is_last:
            return f"[{next_style}]{next_text} [bold red](last link)[/][/]"
        return f"[{next_style}]{next_text}[/]"

    if item_type == "omitted":
        omit_label = str(obj.get("omit_label") or "").strip()
        if omit_label:
            omit_label = omit_label.replace("[", "(").replace("]", ")")
        else:
            omit_label = "omitted"
        return f"[dim red]{no_text} {'×':<2}{core.fmt_dt_local(dt)} [italic]({omit_label})[/][/]"

    if item_type == "warning":
        message = str(obj.get("message") or "Timeline projection unavailable")
        message = message.replace("[", "(").replace("]", ")")
        return f"[bright_yellow]{no_text} {'⚠':<2}{message}[/]"

    is_last = cap_no is not None and no == cap_no
    future_text = f"{no_text} {'»':<2}{core.fmt_dt_local(dt)}"
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
    items: list[tuple[int, Any, dict[str, Any], str]],
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


def timeline_lines(
    kind: str,
    task: dict[str, Any],
    child_due_utc: datetime,
    child_short: str,
    dnf: Any,
    *,
    next_count: int = 3,
    cap_no: int | None = None,
    cur_no: int | None = None,
    show_gaps: bool = True,
    round_anchor_gaps: bool = True,
    core: Any,
    max_iterations: int,
    future_style_for_chain: Callable[[dict[str, Any], str], str],
    collect_prev_two: Callable[[dict[str, Any]], list[dict[str, Any]]],
    dtparse: Callable[[Any], Any],
    fmt_on_time_delta: Callable[[Any, Any], str],
    fmtlocal: Callable[[Any], str],
    short: Callable[[Any], str],
    tolocal: Callable[[datetime], datetime],
    next_occurrence_after_local_dt: Callable[..., Any],
    scheduler_service: Any | None = None,
    to_local_cached: Callable[[datetime], datetime],
    safe_parse_datetime: Callable[[Any], tuple[Any, Any]],
    format_gap: Callable[[Any, Any, str, bool], str],
    omit_dnf=None,
    omit_expr_fires_on_date: Callable[..., bool] | None = None,
    omit_description_for_date: Callable[[Any, Any], str | None] | None = None,
    evaluator: Any | None = None,
) -> list[str]:
    cur_no = core.coerce_int(task.get("link") if cur_no is None else cur_no, 1)
    nxt_no = cur_no + 1
    allowed_future = next_count if cap_no is None else max(0, min(next_count, cap_no - nxt_no))
    prev_style, cur_style, next_style, future_style = _timeline_styles(
        task,
        kind,
        future_style_for_chain=future_style_for_chain,
    )
    items = _timeline_initial_items(
        task,
        cur_no,
        nxt_no,
        child_due_utc,
        child_short,
        core=core,
        collect_prev_two=collect_prev_two,
        dtparse=dtparse,
    )
    if kind == "anchor":
        omitted_before_next = _timeline_omitted_before_next_anchor_items(
            task,
            dnf,
            child_due_utc,
            dtparse=dtparse,
            to_local_cached=to_local_cached,
            safe_parse_datetime=safe_parse_datetime,
            next_occurrence_after_local_dt=next_occurrence_after_local_dt,
            scheduler_service=scheduler_service,
            omit_dnf=omit_dnf,
            omit_expr_fires_on_date=omit_expr_fires_on_date,
            omit_description_for_date=omit_description_for_date,
            max_iterations=max_iterations,
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
                    core=core,
                    tolocal=tolocal,
                    max_iterations=max_iterations,
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
                    to_local_cached=to_local_cached,
                    safe_parse_datetime=safe_parse_datetime,
                    next_occurrence_after_local_dt=next_occurrence_after_local_dt,
                    scheduler_service=scheduler_service,
                    omit_dnf=omit_dnf,
                    omit_expr_fires_on_date=omit_expr_fires_on_date,
                    omit_description_for_date=omit_description_for_date,
                    max_iterations=max_iterations,
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
            core=core,
            dtparse=dtparse,
            fmt_on_time_delta=fmt_on_time_delta,
            fmtlocal=fmtlocal,
            short=short,
        )
        lines.append(
            _timeline_with_gap(
                base_line,
                idx=i,
                items=items,
                show_gaps=show_gaps,
                kind=kind,
                round_anchor_gaps=round_anchor_gaps,
                format_gap=format_gap,
            )
        )
    return lines
