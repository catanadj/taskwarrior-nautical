from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta
from typing import Any, Callable

from .occurrence_provider import Occurrence, _cursor_before
from .timeutil import compare_datetimes
from .scheduler_models import OccurrenceSearchExhausted


def _scheduler_engine(core: Any) -> Any:
    engine = getattr(core, "_scheduler_api", None)
    if engine is None:
        raise RuntimeError("Anchor inclusion scheduler engine is unavailable")
    return engine


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


def _next_anchor_file_occurrence(
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
) -> Occurrence | None:
    if not str(anchor_file_str or "").strip():
        return None
    if inclusive:
        after_local_dt = _cursor_before(after_local_dt)
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
    return occurrence


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
    occurrence = _next_anchor_file_occurrence(
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
    except OccurrenceSearchExhausted:
        raise
    except Exception as exc:
        raise ValueError(
            f"Unable to evaluate omit rule for {item_local.date().isoformat()}: {exc}"
        ) from exc


def next_included_occurrence(
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
    max_file_skips: int = 512,
) -> Occurrence | None:
    if isinstance(max_file_skips, bool) or not isinstance(max_file_skips, int) or max_file_skips <= 0:
        raise ValueError("Anchor-file omission scan limit must be a positive integer.")
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
            expr_after = _cursor_before(after_local_dt) if inclusive else after_local_dt
            expr_local = next_occurrence_after_local_dt(
                dnf,
                expr_after,
                default_seed_date=default_seed_date,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                fallback_hhmm=fallback_hhmm,
            )
    file_occurrence = _next_anchor_file_occurrence(
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
    skipped_file_occurrences = 0
    while file_occurrence is not None and _anchor_file_occurrence_is_omitted(
        file_occurrence.local_datetime,
        omit_dnf=omit_dnf,
        default_seed_date=default_seed_date,
        seed_base=seed_base,
        core=core,
    ):
        skipped_file_occurrences += 1
        if skipped_file_occurrences > max_file_skips:
            raise ValueError(
                f"Anchor-file omission scan exceeded {max_file_skips} occurrences; "
                "narrow the anchor_file or omit rule."
            )
        file_datetime = file_occurrence.local_datetime
        if file_datetime is None:
            raise ValueError("anchor-file occurrence has no local datetime")
        file_cursor = file_datetime
        file_inclusive = False
        file_occurrence = _next_anchor_file_occurrence(
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
    expr_occurrence = None
    if expr_local is not None:
        expr_occurrence = Occurrence(
            day=expr_local.date(),
            hour=expr_local.hour,
            minute=expr_local.minute,
            source="anchor",
            local_datetime=expr_local,
        )
    selected = expr_occurrence
    if selected is None:
        selected = file_occurrence
    elif expr_local is not None and file_occurrence is not None and file_occurrence.local_datetime is not None:
        comparison = compare_datetimes(expr_local, file_occurrence.local_datetime)
        if comparison > 0 or (comparison == 0 and file_occurrence.description):
            selected = file_occurrence
    return selected


def next_included_occurrence_local(**kwargs) -> datetime | None:
    """Compatibility wrapper returning only the selected local datetime."""
    occurrence = next_included_occurrence(**kwargs)
    return occurrence.local_datetime if occurrence is not None else None


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
    scheduler_omit_dnf=...,
) -> Occurrence | None:
    scheduler_omit = omit_dnf if scheduler_omit_dnf is ... else scheduler_omit_dnf
    expr_local = None
    expr_omit_dnf = omit_dnf if dnf and _scheduler_engine(core).dnf_has_counted_random(dnf) else None
    if dnf:
        if inclusive and pick_occurrence_local is not None:
            expr_local = pick_occurrence_local(
                dnf,
                after_local_dt,
                inclusive=True,
                fallback_hhmm=fallback_hhmm,
                interval_seed=default_seed_date,
                seed_base=seed_base,
                omit_dnf=scheduler_omit,
            )
        else:
            expr_after = _cursor_before(after_local_dt) if inclusive else after_local_dt
            expr_local = next_occurrence_after_local_dt(
                dnf,
                expr_after,
                default_seed_date=default_seed_date,
                seed_base=seed_base,
                omit_dnf=scheduler_omit,
                fallback_hhmm=fallback_hhmm,
            )
    file_occurrence = _next_anchor_file_occurrence(
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
    expr_occurrence = None
    if expr_local is not None:
        expr_occurrence = Occurrence(
            day=expr_local.date(),
            hour=expr_local.hour,
            minute=expr_local.minute,
            source="anchor",
            local_datetime=expr_local,
        )
    selected = expr_occurrence
    if selected is None:
        selected = file_occurrence
    elif expr_local is not None and file_occurrence is not None and file_occurrence.local_datetime is not None:
        comparison = compare_datetimes(expr_local, file_occurrence.local_datetime)
        if comparison > 0 or (comparison == 0 and file_occurrence.description):
            selected = file_occurrence
    if selected is None or selected.local_datetime is None:
        return None
    return replace(
        selected,
        omitted=_anchor_file_occurrence_is_omitted(
            selected.local_datetime,
            omit_dnf=omit_dnf,
            default_seed_date=default_seed_date,
            seed_base=seed_base,
            core=core,
        ),
    )
