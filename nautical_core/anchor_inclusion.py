from __future__ import annotations

from datetime import datetime, timedelta
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


def _next_anchor_file_occurrence_local(
    anchor_file_str: str,
    *,
    anchor_file_dir: str,
    after_local_dt: datetime,
    inclusive: bool,
    fallback_hhmm: tuple[int, int],
    core: Any,
) -> datetime | None:
    if not str(anchor_file_str or "").strip():
        return None
    anchor_files = core._import_sibling("anchor_files")
    if inclusive:
        after_local_dt = after_local_dt - timedelta(microseconds=1)
    return anchor_files.next_anchor_file_occurrence_after(
        anchor_file_str,
        anchor_file_dir,
        after_local_dt,
        fallback_hhmm,
        build_local_datetime=core.build_local_datetime,
        to_local=core.to_local,
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
    except Exception:
        return False


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
            try:
                expr_local = next_occurrence_after_local_dt(
                    dnf,
                    expr_after,
                    default_seed_date=default_seed_date,
                    seed_base=seed_base,
                    omit_dnf=omit_dnf,
                    fallback_hhmm=fallback_hhmm,
                )
            except TypeError:
                expr_local = next_occurrence_after_local_dt(
                    dnf,
                    expr_after,
                    fallback_hhmm,
                    default_seed_date,
                    seed_base,
                    omit_dnf=omit_dnf,
                    core=core,
                    norm_t_mod=_norm_t_mod,
                )
    file_local = _next_anchor_file_occurrence_local(
        anchor_file_str,
        anchor_file_dir=anchor_file_dir,
        after_local_dt=after_local_dt,
        inclusive=inclusive,
        fallback_hhmm=fallback_hhmm,
        core=core,
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
        )
    if expr_local and file_local:
        return expr_local if expr_local <= file_local else file_local
    return expr_local or file_local


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
) -> list[datetime]:
    out: list[datetime] = []
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
        )
        if not nxt:
            break
        if out and nxt <= out[-1]:
            break
        out.append(nxt)
        cursor = nxt
        want_inclusive = False
    return out
