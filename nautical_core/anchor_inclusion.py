from __future__ import annotations

from datetime import datetime, timedelta
import inspect
from typing import Any, Callable


def _norm_t_mod(v):
    if v is None:
        return []
    if isinstance(v, tuple) and len(v) == 2:
        return [v]
    if isinstance(v, list):
        out = []
        for it in v:
            if isinstance(it, tuple) and len(it) == 2:
                out.append(it)
            elif isinstance(it, list) and len(it) == 2:
                out.append((int(it[0]), int(it[1])))
        return out
    if isinstance(v, str):
        out = []
        for part in [p.strip() for p in v.split(",") if p.strip()]:
            if len(part) == 5 and part[2] == ":" and part[:2].isdigit() and part[3:].isdigit():
                out.append((int(part[:2]), int(part[3:])))
        return out
    return []


def _call_next_occurrence(
    callback: Callable[..., Any],
    dnf,
    after_local_dt: datetime,
    *,
    default_seed_date,
    seed_base: str,
    omit_dnf,
    fallback_hhmm: tuple[int, int],
    core: Any,
) -> datetime | None:
    """Invoke modern or legacy schedulers without masking callback failures."""
    try:
        parameters = inspect.signature(callback).parameters.values()
    except (TypeError, ValueError):
        parameters = None

    if parameters is None or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return callback(
            dnf,
            after_local_dt,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            fallback_hhmm=fallback_hhmm,
        )

    names = {parameter.name for parameter in parameters}
    if "default_seed_date" in names:
        return callback(
            dnf,
            after_local_dt,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            fallback_hhmm=fallback_hhmm,
        )

    return callback(
        dnf,
        after_local_dt,
        fallback_hhmm,
        default_seed_date,
        seed_base,
        omit_dnf=omit_dnf,
        core=core,
        norm_t_mod=_norm_t_mod,
    )


def _next_anchor_file_occurrence_local(
    anchor_file_str: str,
    *,
    anchor_file_dir: str,
    after_local_dt: datetime,
    inclusive: bool,
    fallback_hhmm: tuple[int, int],
    core: Any,
    anchor_file_provider: Any | None = None,
    recurrence_context: Any | None = None,
    business_calendar: Any | None = None,
) -> datetime | None:
    if not str(anchor_file_str or "").strip():
        return None
    if inclusive:
        after_local_dt = after_local_dt - timedelta(microseconds=1)
    provider = anchor_file_provider
    if provider is None:
        anchor_files = core._import_sibling("anchor_files")
        provider = anchor_files.AnchorFileOccurrenceProvider(
            anchor_file_str,
            anchor_file_dir,
            fallback_hhmm,
            business_calendar=business_calendar,
            context=recurrence_context,
        )
    occurrence = provider.next_after(
        after_local_dt,
        build_local_datetime=core.build_local_datetime,
        to_local=core.to_local,
    )
    return occurrence.local_datetime if occurrence is not None else None


def _build_anchor_file_provider(
    anchor_file_str: str,
    *,
    anchor_file_dir: str,
    fallback_hhmm: tuple[int, int],
    seed_base: str,
    core: Any,
    recurrence_context: Any | None = None,
    business_calendar: Any | None = None,
) -> Any:
    """Build one context-bound file provider for a merged occurrence stream."""
    anchor_files = core._import_sibling("anchor_files")
    if business_calendar is None:
        business_calendar = core._import_sibling("business_calendar").active_business_calendar()
    if recurrence_context is None:
        recurrence_context = core._import_sibling("recurrence_context").RecurrenceContext(
            chain_id=seed_base or "provider",
            business_calendar=business_calendar,
            anchor_file_dir=anchor_file_dir,
        )
    return anchor_files.AnchorFileOccurrenceProvider(
        anchor_file_str,
        anchor_file_dir,
        fallback_hhmm,
        business_calendar=business_calendar,
        context=recurrence_context,
    )


def _anchor_file_occurrence_is_omitted(
    item_local: datetime | None,
    *,
    omit_dnf,
    default_seed_date,
    seed_base: str,
    core: Any,
) -> bool:
    if not item_local or not omit_dnf:
        return False
    try:
        anchor_omit = core._import_sibling("anchor_omit")
        return bool(
            anchor_omit.omit_expr_fires_on_date(
                omit_dnf,
                item_local.date(),
                default_seed_date,
                seed_base,
                core=core,
            )
        )
    except Exception as exc:
        raise ValueError(
            f"Unable to evaluate omit rule for {item_local.date().isoformat()}: {exc}"
        ) from exc


def next_included_occurrence_local(
    *,
    dnf,
    anchor_file_str: str,
    after_local_dt: datetime,
    inclusive: bool,
    fallback_hhmm: tuple[int, int],
    default_seed_date,
    seed_base: str,
    omit_dnf,
    core: Any,
    next_occurrence_after_local_dt: Callable[..., Any],
    pick_occurrence_local: Callable[..., Any] | None = None,
    anchor_file_dir: str = "",
    anchor_file_provider: Any | None = None,
    recurrence_context: Any | None = None,
    business_calendar: Any | None = None,
) -> datetime | None:
    expr_local = None
    if dnf:
        if inclusive and pick_occurrence_local is not None:
            expr_local = pick_occurrence_local(
                dnf,
                after_local_dt,
                inclusive=True,
                fallback_hhmm=fallback_hhmm,
                interval_seed=default_seed_date,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
            )
        else:
            expr_after = after_local_dt - timedelta(microseconds=1) if inclusive else after_local_dt
            expr_local = _call_next_occurrence(
                next_occurrence_after_local_dt,
                dnf,
                expr_after,
                default_seed_date=default_seed_date,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                fallback_hhmm=fallback_hhmm,
                core=core,
            )
    file_local = _next_anchor_file_occurrence_local(
        anchor_file_str,
        anchor_file_dir=anchor_file_dir,
        after_local_dt=after_local_dt,
        inclusive=inclusive,
        fallback_hhmm=fallback_hhmm,
        core=core,
        anchor_file_provider=anchor_file_provider,
        recurrence_context=recurrence_context,
        business_calendar=business_calendar,
    )
    file_cursor = after_local_dt
    file_inclusive = inclusive
    while _anchor_file_occurrence_is_omitted(
        file_local,
        omit_dnf=omit_dnf,
        default_seed_date=default_seed_date,
        seed_base=seed_base,
        core=core,
    ):
        file_cursor = file_local
        file_inclusive = False
        file_local = _next_anchor_file_occurrence_local(
            anchor_file_str,
            anchor_file_dir=anchor_file_dir,
            after_local_dt=file_cursor,
            inclusive=file_inclusive,
            fallback_hhmm=fallback_hhmm,
            core=core,
            anchor_file_provider=anchor_file_provider,
            recurrence_context=recurrence_context,
            business_calendar=business_calendar,
        )
    if expr_local and file_local:
        return expr_local if expr_local <= file_local else file_local
    return expr_local or file_local


def next_occurrence_event_local(
    *,
    dnf,
    anchor_file_str: str,
    after_local_dt: datetime,
    inclusive: bool,
    fallback_hhmm: tuple[int, int],
    default_seed_date,
    seed_base: str,
    omit_dnf,
    core: Any,
    next_occurrence_after_local_dt: Callable[..., Any],
    pick_occurrence_local: Callable[..., Any] | None = None,
    anchor_file_dir: str = "",
    anchor_file_provider: Any | None = None,
    recurrence_context: Any | None = None,
    business_calendar: Any | None = None,
) -> tuple[datetime, bool] | None:
    expr_local = None
    expr_omit_dnf = omit_dnf if dnf and core.dnf_has_counted_random(dnf) else None
    if dnf:
        if inclusive and pick_occurrence_local is not None:
            expr_local = pick_occurrence_local(
                dnf,
                after_local_dt,
                inclusive=True,
                fallback_hhmm=fallback_hhmm,
                interval_seed=default_seed_date,
                seed_base=seed_base,
                omit_dnf=expr_omit_dnf,
            )
        else:
            expr_after = after_local_dt - timedelta(microseconds=1) if inclusive else after_local_dt
            expr_local = _call_next_occurrence(
                next_occurrence_after_local_dt,
                dnf,
                expr_after,
                default_seed_date=default_seed_date,
                seed_base=seed_base,
                omit_dnf=expr_omit_dnf,
                fallback_hhmm=fallback_hhmm,
                core=core,
            )
    file_local = _next_anchor_file_occurrence_local(
        anchor_file_str,
        anchor_file_dir=anchor_file_dir,
        after_local_dt=after_local_dt,
        inclusive=inclusive,
        fallback_hhmm=fallback_hhmm,
        core=core,
        anchor_file_provider=anchor_file_provider,
        recurrence_context=recurrence_context,
        business_calendar=business_calendar,
    )
    nxt = None
    if expr_local and file_local:
        nxt = expr_local if expr_local <= file_local else file_local
    else:
        nxt = expr_local or file_local
    if not nxt:
        return None
    return (
        nxt,
        _anchor_file_occurrence_is_omitted(
            nxt,
            omit_dnf=omit_dnf,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            core=core,
        ),
    )


def collect_included_occurrences_local(
    *,
    dnf,
    anchor_file_str: str,
    after_local_dt: datetime,
    inclusive: bool,
    limit: int,
    fallback_hhmm: tuple[int, int],
    default_seed_date,
    seed_base: str,
    omit_dnf,
    core: Any,
    next_occurrence_after_local_dt: Callable[..., Any],
    pick_occurrence_local: Callable[..., Any] | None = None,
    anchor_file_dir: str = "",
    anchor_file_provider: Any | None = None,
    recurrence_context: Any | None = None,
    business_calendar: Any | None = None,
) -> list[datetime]:
    out: list[datetime] = []
    if anchor_file_str and anchor_file_provider is None:
        anchor_file_provider = _build_anchor_file_provider(
            anchor_file_str,
            anchor_file_dir=anchor_file_dir,
            fallback_hhmm=fallback_hhmm,
            seed_base=seed_base,
            core=core,
            recurrence_context=recurrence_context,
            business_calendar=business_calendar,
        )
    cursor = after_local_dt
    want_inclusive = inclusive
    while len(out) < limit:
        nxt = next_included_occurrence_local(
            dnf=dnf,
            anchor_file_str=anchor_file_str,
            after_local_dt=cursor,
            inclusive=want_inclusive,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=next_occurrence_after_local_dt,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_dir=anchor_file_dir,
            anchor_file_provider=anchor_file_provider,
            recurrence_context=recurrence_context,
            business_calendar=business_calendar,
        )
        if not nxt:
            break
        if out and nxt <= out[-1]:
            break
        out.append(nxt)
        cursor = nxt
        want_inclusive = False
    return out


def collect_occurrence_events_local(
    *,
    dnf,
    anchor_file_str: str,
    after_local_dt: datetime,
    inclusive: bool,
    limit_included: int,
    fallback_hhmm: tuple[int, int],
    default_seed_date,
    seed_base: str,
    omit_dnf,
    core: Any,
    next_occurrence_after_local_dt: Callable[..., Any],
    pick_occurrence_local: Callable[..., Any] | None = None,
    anchor_file_dir: str = "",
    max_iterations: int = 512,
    anchor_file_provider: Any | None = None,
    recurrence_context: Any | None = None,
    business_calendar: Any | None = None,
) -> list[tuple[datetime, bool]]:
    out: list[tuple[datetime, bool]] = []
    if anchor_file_str and anchor_file_provider is None:
        anchor_file_provider = _build_anchor_file_provider(
            anchor_file_str,
            anchor_file_dir=anchor_file_dir,
            fallback_hhmm=fallback_hhmm,
            seed_base=seed_base,
            core=core,
            recurrence_context=recurrence_context,
            business_calendar=business_calendar,
        )
    cursor = after_local_dt
    want_inclusive = inclusive
    included_count = 0
    iterations = 0
    while included_count < limit_included and iterations < max_iterations:
        iterations += 1
        nxt = next_occurrence_event_local(
            dnf=dnf,
            anchor_file_str=anchor_file_str,
            after_local_dt=cursor,
            inclusive=want_inclusive,
            fallback_hhmm=fallback_hhmm,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            omit_dnf=omit_dnf,
            core=core,
            next_occurrence_after_local_dt=next_occurrence_after_local_dt,
            pick_occurrence_local=pick_occurrence_local,
            anchor_file_dir=anchor_file_dir,
            anchor_file_provider=anchor_file_provider,
            recurrence_context=recurrence_context,
            business_calendar=business_calendar,
        )
        if not nxt:
            break
        event_local, is_omitted = nxt
        if out and event_local <= out[-1][0]:
            break
        out.append((event_local, is_omitted))
        if not is_omitted:
            included_count += 1
        cursor = event_local
        want_inclusive = False
    return out
