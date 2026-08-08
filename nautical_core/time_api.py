"""Core-bound timezone, date arithmetic, and time-slot adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)
    timeutil = core["_timeutil"]
    dates = core["_dates"]

    def now_utc():
        return timeutil.now_utc()

    def to_local(dt_utc):
        return timeutil.to_local(dt_utc, core["_LOCAL_TZ"])

    def utc_to_local_naive(dt_utc):
        return timeutil.utc_to_local_naive(dt_utc, core["_LOCAL_TZ"])

    def local_naive_to_utc(dt_local_naive):
        return timeutil.local_naive_to_utc(dt_local_naive, core["_LOCAL_TZ"])

    def fmt_dt_local(dt_utc):
        return timeutil.fmt_dt_local(dt_utc, core["_LOCAL_TZ"])

    def fmt_isoz(dt_utc):
        return timeutil.fmt_isoz(dt_utc)

    def ensure_utc(dt_utc):
        return timeutil.ensure_utc(dt_utc)

    def coerce_int(v, default=None):
        return core["_common"].coerce_int(v, default=default)

    def parse_dt_any(s: str):
        return timeutil.parse_dt_any(s, core["DATE_FORMATS"])

    def month_len(year, month):
        return dates.month_len(year, month)

    def add_months(day, months: int):
        return dates.add_months(day, months)

    def months_days_between(first, second):
        return dates.months_days_between(first, second)

    def humanize_delta(from_dt, to_dt, use_months_days: bool):
        return dates.humanize_delta(from_dt, to_dt, use_months_days)

    def expr_has_m_or_y(dnf) -> bool:
        return core["_schedule_utils"].expr_has_m_or_y(dnf)

    def pick_hhmm_from_dnf_for_date(
        dnf,
        target,
        default_seed,
        seed_base=None,
        business_calendar=None,
    ):
        return core["_schedule_utils"].pick_hhmm_from_dnf_for_date(
            dnf,
            target,
            default_seed,
            seed_base=seed_base,
            atom_matches_on=core["_with_business_calendar"](
                core["factor_matches_on"], business_calendar
            ),
        )

    def build_local_datetime(d, hhmm=(core["DEFAULT_DUE_HOUR"], 0)):
        return timeutil.build_local_datetime(d, hhmm, core["_LOCAL_TZ"])

    return SimpleNamespace(
        now_utc=now_utc,
        to_local=to_local,
        utc_to_local_naive=utc_to_local_naive,
        local_naive_to_utc=local_naive_to_utc,
        fmt_dt_local=fmt_dt_local,
        fmt_isoz=fmt_isoz,
        _ensure_utc=ensure_utc,
        coerce_int=coerce_int,
        parse_dt_any=parse_dt_any,
        month_len=month_len,
        add_months=add_months,
        months_days_between=months_days_between,
        humanize_delta=humanize_delta,
        expr_has_m_or_y=expr_has_m_or_y,
        pick_hhmm_from_dnf_for_date=pick_hhmm_from_dnf_for_date,
        build_local_datetime=build_local_datetime,
    )


__all__ = ("for_core",)
