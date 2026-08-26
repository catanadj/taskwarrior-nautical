from __future__ import annotations

from datetime import date, timedelta, timezone
from typing import Any, Callable

from .timeutil import compare_datetimes
from .occurrence_provider import OccurrenceBatch
from .scheduler_models import OccurrenceSearchExhausted


def _scheduler_engine(core: Any) -> Any:
    engine = getattr(core, "_scheduler_api", None)
    if engine is None:
        raise RuntimeError("Anchor scheduler engine is unavailable")
    return engine


def _scheduler_business_calendar(core: Any) -> Any:
    configured = getattr(core, "business_calendar", None)
    if configured is not None:
        return configured
    try:
        return core._import_sibling("business_calendar").active_business_calendar()
    except Exception:
        return None


def anchor_step_once(dnf, prev_local_date, interval_seed, seed_base, *, core: Any):
    return anchor_step_once_with_omit(
        dnf,
        prev_local_date,
        interval_seed,
        seed_base,
        omit_dnf=None,
        core=core,
    )


def anchor_step_once_with_omit(dnf, prev_local_date, interval_seed, seed_base, *, omit_dnf, core: Any):
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        nxt_date, _ = anchor_omit.next_after_expr_with_omit(
            dnf,
            prev_local_date,
            default_seed=interval_seed,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            max_skip_iterations=max(getattr(core, "MAX_ANCHOR_ITER", 128), 128),
            business_calendar=_scheduler_business_calendar(core),
        )
        if nxt_date is None or nxt_date <= prev_local_date:
            return None
        return nxt_date
    except OccurrenceSearchExhausted:
        raise
    except Exception:
        return None


def anchor_term_fires_on_date(term, d, interval_seed, seed_base, *, core: Any):
    try:
        engine = _scheduler_engine(core)
        return all(engine.factor_matches_on(atom, d, interval_seed, seed_base=seed_base) for atom in term)
    except OccurrenceSearchExhausted:
        raise
    except Exception:
        return False


def anchor_expr_fires_on_date(dnf, d, interval_seed, seed_base, *, core: Any):
    return anchor_expr_fires_on_date_with_omit(
        dnf,
        d,
        interval_seed,
        seed_base,
        omit_dnf=None,
        core=core,
    )


def anchor_expr_fires_on_date_with_omit(dnf, d, interval_seed, seed_base, *, omit_dnf, core: Any):
    try:
        engine = _scheduler_engine(core)
        anchor_omit = core._import_sibling("anchor_omit")
        if anchor_omit.omit_expr_fires_on_date(
            omit_dnf,
            d,
            interval_seed,
            seed_base,
            core=core,
        ):
            return False
        if engine.dnf_has_counted_random(dnf):
            date_is_excluded = None
            if omit_dnf:
                date_is_excluded = lambda candidate: anchor_omit.omit_expr_fires_on_date(
                    omit_dnf,
                    candidate,
                    interval_seed,
                    seed_base,
                    core=core,
                )
            return engine.next_after_expr(
                dnf,
                d - timedelta(days=1),
                default_seed=interval_seed,
                seed_base=seed_base,
                date_is_excluded=date_is_excluded,
            )[0] == d
        return any(anchor_term_fires_on_date(term, d, interval_seed, seed_base, core=core) for term in dnf)
    except OccurrenceSearchExhausted:
        raise
    except Exception:
        return False


def anchor_times_for_date(
    dnf,
    d,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
    resolve_time_slots: Callable[[Any, date], list[tuple[int, int]]] | None = None,
    project_time: Callable[[Any, date], Any] | None = None,
):
    times = set()
    for term in dnf:
        term_matches = anchor_term_fires_on_date(term, d, interval_seed, seed_base, core=core)
        if not term_matches and _scheduler_engine(core).dnf_has_counted_random([term]):
            try:
                engine = _scheduler_engine(core)
                anchor_omit = core._import_sibling("anchor_omit")
                date_is_excluded = None
                if omit_dnf:
                    date_is_excluded = lambda candidate: anchor_omit.omit_expr_fires_on_date(
                        omit_dnf,
                        candidate,
                        interval_seed,
                        seed_base,
                        core=core,
                    )
                term_matches = engine.next_after_expr(
                    [term],
                    d - timedelta(days=1),
                    default_seed=interval_seed,
                    seed_base=seed_base,
                    date_is_excluded=date_is_excluded,
                )[0] == d
            except OccurrenceSearchExhausted:
                raise
            except Exception:
                term_matches = False
        if term_matches:
            for atom in term:
                mods = atom.get("mods") or {}
                slots: list[Any]
                if project_time is not None:
                    from .time_projection import (
                        ProjectedTime,
                        ProjectionInvalid,
                        ProjectionTerminal,
                        ProjectionUnavailable,
                    )

                    projection = project_time(mods, d)
                    if isinstance(projection, ProjectedTime):
                        slots = list(projection.slots)
                    elif isinstance(projection, ProjectionTerminal):
                        raise projection.error
                    elif isinstance(projection, ProjectionUnavailable):
                        raise LookupError(projection.reason)
                    elif isinstance(projection, ProjectionInvalid):
                        raise ValueError(projection.reason)
                    else:
                        raise TypeError("Unknown time projection result.")
                elif mods.get("time_random"):
                    recurrence_context = core._import_sibling("recurrence_context").RecurrenceContext(
                        chain_id=seed_base,
                    )
                    slots = core._import_sibling("time_slots").resolve_time_slots_with_offsets(
                        mods,
                        d,
                        config=getattr(core, "ASTRONOMY_CONFIG", {}),
                        to_local=core.to_local,
                        context=recurrence_context,
                    )
                elif mods.get("time_window") and getattr(core._import_sibling("time_windows").parse_time_window_spec(str(mods["time_window"])), "crosses_midnight", False):
                    slots = core._import_sibling("time_slots").resolve_time_slots_with_offsets(
                        mods, d, config=getattr(core, "ASTRONOMY_CONFIG", {}), to_local=core.to_local
                    )
                else:
                    slots = resolve_time_slots(mods, d) if resolve_time_slots else norm_t_mod(mods.get("t"))
                for slot in slots:
                    times.add(slot)
    return sorted(times)


def _unique_local_candidates(d: date, slots, *, core: Any):
    """Build local candidates once so DST gap normalization cannot duplicate an instant."""
    seen = set()
    for slot in slots:
        if isinstance(slot, tuple) and len(slot) == 3:
            day_offset, hour, minute = slot
            candidate_date = d + timedelta(days=int(day_offset))
            hhmm = (int(hour), int(minute))
        else:
            candidate_date = d
            hhmm = slot
        cand_utc = core.build_local_datetime(candidate_date, hhmm)
        cand_local = core.to_local(cand_utc)
        key = cand_utc
        if key in seen:
            continue
        seen.add(key)
        yield cand_local


def _build_slot_datetime(d: date, slot, *, core: Any):
    if isinstance(slot, tuple) and len(slot) == 3:
        day_offset, hour, minute = slot
        d = d + timedelta(days=int(day_offset))
        slot = (int(hour), int(minute))
    return core.build_local_datetime(d, slot)


def _available_time_after_date(
    dnf,
    start_date,
    interval_seed,
    seed_base,
    fallback_hhmm,
    omit_dnf,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
    resolve_time_slots: Callable[[Any, date], list[tuple[int, int]]] | None,
    project_time: Callable[[Any, date], Any] | None = None,
    max_days: int = 32,
):
    """Search the current phase/month window after an unavailable event date."""
    for offset in range(1, max_days + 1):
        candidate = start_date + timedelta(days=offset)
        if not anchor_expr_fires_on_date_with_omit(
            dnf, candidate, interval_seed, seed_base, omit_dnf=omit_dnf, core=core
        ):
            continue
        try:
            tlist = anchor_times_for_date(
                dnf, candidate, interval_seed, seed_base, omit_dnf=omit_dnf,
                core=core, norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                project_time=project_time,
            ) or [fallback_hhmm]
        except LookupError:
            continue
        return candidate, tlist
    return None


def anchor_pick_occurrence_local(
    dnf,
    ref_dt_local,
    inclusive: bool,
    fallback_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
    resolve_time_slots: Callable[[Any, date], list[tuple[int, int]]] | None = None,
    project_time: Callable[[Any, date], Any] | None = None,
):
    d0 = ref_dt_local.date()
    unavailable = None
    if anchor_expr_fires_on_date_with_omit(dnf, d0, interval_seed, seed_base, omit_dnf=omit_dnf, core=core):
        try:
            tlist = anchor_times_for_date(
                dnf, d0, interval_seed, seed_base, omit_dnf=omit_dnf, core=core,
                norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                project_time=project_time,
            ) or [fallback_hhmm]
            for cand_local in _unique_local_candidates(d0, tlist, core=core):
                comparison = compare_datetimes(cand_local, ref_dt_local)
                matches = comparison >= 0 if inclusive else comparison > 0
                if matches:
                    return cand_local
        except LookupError as exc:
            unavailable = exc
            same_window = _available_time_after_date(
                dnf, d0, interval_seed, seed_base, fallback_hhmm, omit_dnf,
                core=core, norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                project_time=project_time,
            )
            if same_window:
                candidate, tlist = same_window
                return core.to_local(_build_slot_datetime(candidate, tlist[0], core=core))

    try:
        anchor_omit = core._import_sibling("anchor_omit")
        candidate = d0
        for _ in range(max(getattr(core, "MAX_ANCHOR_ITER", 128), 128)):
            nxt_d, _ = anchor_omit.next_after_expr_with_omit(
                dnf, candidate, default_seed=interval_seed, seed_base=seed_base,
                omit_dnf=omit_dnf, core=core,
                max_skip_iterations=max(getattr(core, "MAX_ANCHOR_ITER", 128), 128),
                business_calendar=_scheduler_business_calendar(core),
            )
            if not nxt_d:
                break
            if nxt_d <= candidate:
                candidate = candidate + timedelta(days=1)
                continue
            candidate = nxt_d
            try:
                tlist = anchor_times_for_date(
                    dnf, candidate, interval_seed, seed_base, omit_dnf=omit_dnf, core=core,
                    norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                ) or [fallback_hhmm]
            except LookupError as exc:
                unavailable = exc
                same_window = _available_time_after_date(
                    dnf, candidate, interval_seed, seed_base, fallback_hhmm, omit_dnf,
                    core=core, norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                )
                if same_window:
                    candidate, tlist = same_window
                    return core.to_local(_build_slot_datetime(candidate, tlist[0], core=core))
                continue
            return core.to_local(_build_slot_datetime(candidate, tlist[0], core=core))
    except OccurrenceSearchExhausted:
        raise
    except Exception:
        if unavailable is not None:
            raise unavailable
        return None
    if unavailable is not None:
        raise unavailable
    return None


def anchor_next_occurrence_after_local_dt(
    dnf,
    after_dt_local,
    fallback_hhmm=(9, 0),
    interval_seed=None,
    seed_base="",
    omit_dnf=None,
    *,
    default_seed_date=None,
    core: Any | None = None,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]] | None = None,
    resolve_time_slots: Callable[[Any, date], list[tuple[int, int]]] | None = None,
    project_time: Callable[[Any, date], Any] | None = None,
):
    if default_seed_date is not None:
        interval_seed = default_seed_date
    if core is None:
        from . import _PKG_PROXY
        import sys
        package = sys.modules.get(__package__ or "nautical_core")
        if package is not None:
            _PKG_PROXY.__dict__.update(vars(package))
        core = _PKG_PROXY
    if norm_t_mod is None:
        from .anchor_inclusion import _norm_t_mod
        norm_t_mod = _norm_t_mod
    to_local = getattr(core, "to_local", None)
    if not callable(to_local):
        raise RuntimeError("Anchor scheduler core is missing the timezone conversion callback")
    d0 = after_dt_local.date()
    unavailable = None
    if anchor_expr_fires_on_date_with_omit(
        dnf,
        d0,
        interval_seed,
        seed_base,
        omit_dnf=omit_dnf,
        core=core,
    ):
        try:
            tlist = anchor_times_for_date(
                dnf, d0, interval_seed, seed_base, omit_dnf=omit_dnf, core=core,
                norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                project_time=project_time,
            ) or [fallback_hhmm]
            for cand_local in _unique_local_candidates(d0, tlist, core=core):
                if compare_datetimes(cand_local, after_dt_local) > 0:
                    return cand_local
        except LookupError as exc:
            unavailable = exc
            same_window = _available_time_after_date(
                dnf, d0, interval_seed, seed_base, fallback_hhmm, omit_dnf,
                core=core, norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                project_time=project_time,
            )
            if same_window:
                candidate, tlist = same_window
                return to_local(_build_slot_datetime(candidate, tlist[0], core=core))

    # An overnight window belongs to the previous anchor date. Continue any
    # after-midnight slots before advancing to the next anchor date.
    previous_date = d0 - timedelta(days=1)
    if anchor_expr_fires_on_date_with_omit(
        dnf, previous_date, interval_seed, seed_base, omit_dnf=omit_dnf, core=core
    ):
        previous_slots = anchor_times_for_date(
            dnf, previous_date, interval_seed, seed_base, omit_dnf=omit_dnf,
            core=core, norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
        )
        for cand_local in _unique_local_candidates(previous_date, previous_slots, core=core):
            if compare_datetimes(cand_local, after_dt_local) > 0:
                return cand_local

    candidate = d0
    for _ in range(max(getattr(core, "MAX_ANCHOR_ITER", 128), 128)):
        nxt_d = anchor_step_once_with_omit(dnf, candidate, interval_seed, seed_base, omit_dnf=omit_dnf, core=core)
        if not nxt_d:
            break
        if nxt_d <= candidate:
            candidate = candidate + timedelta(days=1)
            continue
        candidate = nxt_d
        try:
            tlist = anchor_times_for_date(
                dnf, candidate, interval_seed, seed_base, omit_dnf=omit_dnf, core=core,
                norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
            ) or [fallback_hhmm]
        except LookupError as exc:
            unavailable = exc
            same_window = _available_time_after_date(
                dnf, candidate, interval_seed, seed_base, fallback_hhmm, omit_dnf,
                core=core, norm_t_mod=norm_t_mod, resolve_time_slots=resolve_time_slots,
                project_time=project_time,
            )
            if same_window:
                candidate, tlist = same_window
                return to_local(_build_slot_datetime(candidate, tlist[0], core=core))
            continue
        return to_local(_build_slot_datetime(candidate, tlist[0], core=core))
    if unavailable is not None:
        raise unavailable
    return None


def anchor_until_summary(
    dnf,
    until_dt,
    first_date_local,
    first_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    to_local_cached: Callable[[Any], Any],
    max_preview_iterations: int,
    max_iterations: int,
    resolve_time_slots: Callable[[Any, date], list[tuple[int, int]]] | None = None,
    evaluator: Any | None = None,
):
    if not until_dt:
        return None, None
    if evaluator is not None:
        start_local = core.build_local_datetime(first_date_local, first_hhmm)
        end_local = to_local_cached(until_dt)
        events = evaluator.events_between(
            start_local,
            end_local,
            limit=max_iterations,
            fallback_hhmm=first_hhmm,
            default_seed_date=interval_seed,
            inclusive=True,
            max_iterations=max_iterations,
            include_omitted=False,
        )
        count = len(events)
        if count <= 0:
            return 0, None
        last = events[-1].local_datetime
        if last is None:
            return max(0, count - 1), None
        return max(0, count - 1), last.astimezone(timezone.utc)

    end_day = to_local_cached(until_dt).date()
    count = 0
    prev = first_date_local - timedelta(days=1)
    last = None
    iterations = 0
    for _ in range(max_preview_iterations):
        if iterations >= max_iterations:
            break
        iterations += 1
        nxt = anchor_step_once_with_omit(dnf, prev, interval_seed, seed_base, omit_dnf=omit_dnf, core=core)
        if not nxt or nxt > end_day:
            break
        count += 1
        last = nxt
        prev = nxt
    exact_until_count = max(0, count - 1)
    if not last:
        return exact_until_count, None
    final_hhmm = None
    if resolve_time_slots:
        for term in dnf:
            engine = _scheduler_engine(core)
            if all(engine.factor_matches_on(atom, last, first_date_local, seed_base=seed_base) for atom in term):
                for atom in term:
                    mods = atom.get("mods") or {}
                    slots = resolve_time_slots(mods, last)
                    if slots:
                        final_hhmm = slots[0]
                        break
                if final_hhmm:
                    break
    if final_hhmm is None:
        final_hhmm = core.pick_hhmm_from_dnf_for_date(
            dnf,
            last,
            first_date_local,
            seed_base=seed_base,
        ) or first_hhmm
    final_until_dt = core.build_local_datetime(last, final_hhmm).astimezone(timezone.utc)
    return exact_until_count, final_until_dt


def anchor_build_preview(
    dnf,
    first_due_local_dt,
    preview_limit: int,
    until_dt,
    fallback_hhmm,
    interval_seed,
    seed_base,
    omit_dnf=None,
    *,
    core: Any,
    norm_t_mod: Callable[[Any], list[tuple[int, int]]],
    resolve_time_slots: Callable[[Any, date], list[tuple[int, int]]] | None = None,
    evaluator: Any | None = None,
):
    if evaluator is not None:
        events = evaluator.collect_after(
            first_due_local_dt,
            limit=preview_limit,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=interval_seed,
            inclusive=False,
        )
        preview = []
        colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
        for i, event in enumerate(events):
            if event.local_datetime is None:
                continue
            dt_utc = event.local_datetime.astimezone(timezone.utc)
            if until_dt and compare_datetimes(dt_utc, until_dt) > 0:
                break
            color = colors[min(i, len(colors) - 1)]
            preview.append(f"[{color}]{core.fmt_dt_local(dt_utc)}[/{color}]")
        return OccurrenceBatch(preview, terminal=getattr(events, "terminal", None))

    preview = []
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    cur_dt = first_due_local_dt
    for i in range(preview_limit):
        nxt_dt = anchor_next_occurrence_after_local_dt(
            dnf,
            cur_dt,
            fallback_hhmm,
            interval_seed,
            seed_base,
            omit_dnf=omit_dnf,
            core=core,
            norm_t_mod=norm_t_mod,
            resolve_time_slots=resolve_time_slots,
        )
        if not nxt_dt:
            break
        dt_utc = nxt_dt.astimezone(timezone.utc)
        if until_dt and compare_datetimes(dt_utc, until_dt) > 0:
            break
        color = colors[min(i, len(colors) - 1)]
        preview.append(f"[{color}]{core.fmt_dt_local(dt_utc)}[/{color}]")
        cur_dt = nxt_dt
    return OccurrenceBatch(preview)
