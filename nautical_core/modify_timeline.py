from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Callable


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
) -> list[tuple[int, datetime, dict[str, Any], str]]:
    td = core.parse_cp_duration(task.get("cp") or "")
    if not td:
        return []
    items: list[tuple[int, datetime, dict[str, Any], str]] = []
    fut_dt = child_due_utc
    fut_no = start_no
    secs = int(td.total_seconds())
    iterations = 0
    for _ in range(allowed_future):
        if iterations >= max_iterations:
            break
        iterations += 1
        fut_no += 1
        if secs % 86400 == 0:
            dl = tolocal(fut_dt)
            fut_dt = core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(),
                (dl.hour, dl.minute),
            ).astimezone(timezone.utc)
        else:
            fut_dt = fut_dt + td
        if cap_no is not None and fut_no > cap_no:
            break
        items.append((fut_no, fut_dt, {"is_future": True}, "future"))
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
    omit_dnf,
    omit_expr_fires_on_date: Callable[..., bool] | None,
    max_iterations: int,
) -> list[tuple[object, datetime, dict[str, Any], str]]:
    items: list[tuple[object, datetime, dict[str, Any], str]] = []
    fut_no = start_no
    seed_base = (task.get("chainID") or "").strip() or "preview"
    nxt_local = to_local_cached(child_due_utc)
    fallback_hhmm = (nxt_local.hour, nxt_local.minute)
    due0, _ = safe_parse_datetime(task.get("due"))
    sched0, _ = safe_parse_datetime(task.get("scheduled"))
    default_seed = to_local_cached(due0 or sched0 or child_due_utc).date()
    after_local = nxt_local
    iterations = 0
    actual_future = 0
    while actual_future < allowed_future:
        if iterations >= max_iterations:
            break
        iterations += 1
        try:
            next_local = next_occurrence_after_local_dt(
                dnf,
                after_local,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=None,
                fallback_hhmm=fallback_hhmm,
            )
        except Exception:
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
                    items.append(("··", fut_dt, {"is_omit": True}, "omitted"))
                    continue
            except Exception:
                pass
        fut_no += 1
        if cap_no is not None and fut_no > cap_no:
            break
        items.append((fut_no, fut_dt, {"is_future": True}, "future"))
        actual_future += 1
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
    omit_dnf,
    omit_expr_fires_on_date: Callable[..., bool] | None,
    max_iterations: int,
) -> list[tuple[object, datetime, dict[str, Any], str]]:
    if not omit_dnf or omit_expr_fires_on_date is None:
        return []
    cur_end = dtparse(task.get("end"))
    if not cur_end:
        return []

    items: list[tuple[object, datetime, dict[str, Any], str]] = []
    seed_base = (task.get("chainID") or "").strip() or "preview"
    child_local = to_local_cached(child_due_utc)
    after_local = to_local_cached(cur_end)
    fallback_hhmm = (child_local.hour, child_local.minute)
    due0, _ = safe_parse_datetime(task.get("due"))
    sched0, _ = safe_parse_datetime(task.get("scheduled"))
    default_seed = to_local_cached(due0 or sched0 or child_due_utc).date()
    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        try:
            next_local = next_occurrence_after_local_dt(
                dnf,
                after_local,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=None,
                fallback_hhmm=fallback_hhmm,
            )
        except Exception:
            break
        if not next_local or next_local >= child_local:
            break
        after_local = next_local
        try:
            if omit_expr_fires_on_date(
                omit_dnf,
                next_local.date(),
                default_seed,
                seed_base,
            ):
                items.append(("··", next_local.astimezone(timezone.utc), {"is_omit": True}, "omitted"))
        except Exception:
            continue
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
        return f"[{prev_style}]{no_text} {'✓':<2}{end_s} {delta} {short_id}[/]"

    if item_type == "current":
        cur_end = dtparse(task.get("end"))
        cur_due = dtparse(task.get("due"))
        cur_delta = fmt_on_time_delta(cur_due, cur_end)
        cur_end_s = fmtlocal(cur_end) if cur_end else "(no end)"
        return f"[{cur_style}]{no_text} {'✓':<2}{cur_end_s} {cur_delta} {short(task.get('uuid'))}[/]"

    if item_type == "next":
        is_last = cap_no is not None and no == cap_no
        next_text = f"{no_text} {'►':<2}{core.fmt_dt_local(dt)} {short(obj.get('uuid'))}"
        if is_last:
            return f"[{next_style}]{next_text} [bold red](last link)[/][/]"
        return f"[{next_style}]{next_text}[/]"

    if item_type == "omitted":
        return f"[dim red]{no_text} {'×':<2}{core.fmt_dt_local(dt)} [italic](omitted)[/][/]"

    is_last = cap_no is not None and no == cap_no
    future_text = f"{no_text} {'»':<2}{core.fmt_dt_local(dt)}"
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
    to_local_cached: Callable[[datetime], datetime],
    safe_parse_datetime: Callable[[Any], tuple[Any, Any]],
    format_gap: Callable[[Any, Any, str, bool], str],
    omit_dnf=None,
    omit_expr_fires_on_date: Callable[..., bool] | None = None,
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
            omit_dnf=omit_dnf,
            omit_expr_fires_on_date=omit_expr_fires_on_date,
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
                    omit_dnf=omit_dnf,
                    omit_expr_fires_on_date=omit_expr_fires_on_date,
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
