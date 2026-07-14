from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Callable

from .business_calendar import (
    DEFAULT_BUSINESS_CALENDAR,
    BusinessCalendar,
    effective_business_calendar,
)
from .business_calendar_config import validate_calendar_rule_modifiers
from .file_backed_dates import load_file_date_data
from .file_source_expr import (
    FileSourceResolution,
    ResolvedFileSource,
    parse_file_source_expression,
    resolve_file_source_expression,
    resolve_file_sources,
)
from .schedule_utils import apply_day_offset, roll_apply


_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_NEXT_PREV_WD_RE = re.compile(r"^(next|prev)-(mon|tue|wed|thu|fri|sat|sun)$")
_DAY_OFFSET_RE = re.compile(r"^([+-]\d+)d$")
_HHMM_RE = re.compile(r"^(\d{2}):(\d{2})$")
_HOUR_PAD_RE = re.compile(r"^(\d):(\d{2})(?::\d{2})?$")


def _default_mods() -> dict:
    return {
        "t": None,
        "roll": None,
        "wd": None,
        "bd": False,
        "day_offset": 0,
        "business_day_offset": 0,
    }


def validate_anchor_file_name(value: str | None) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if name in {".", ".."} or "/" in name or "\\" in name or os.path.isabs(name):
        raise ValueError("anchor_file must be a file name, not a path.")
    return name


def _parse_hhmm(text: str) -> tuple[int, int] | None:
    raw = str(text or "").strip()
    match = _HHMM_RE.match(raw)
    if not match:
        pad = _HOUR_PAD_RE.match(raw)
        if pad:
            raise ValueError(f"Time '{raw}' needs a leading zero. Use '0{pad.group(1)}:{pad.group(2)}'.")
        return None
    hh = int(match.group(1))
    mm = int(match.group(2))
    if hh > 23 or mm > 59:
        return None
    return (hh, mm)


def parse_anchor_file_spec(value: str | None) -> tuple[str, dict]:
    raw = str(value or "").strip()
    if not raw:
        return "", _default_mods()
    name, mods_str = (raw.split("@", 1) + [""])[:2]
    file_name = validate_anchor_file_name(name.strip())
    mods = _default_mods()
    if not mods_str:
        return file_name, mods

    for raw_tok in mods_str.split("@"):
        tok = raw_tok.strip().lower()
        if not tok:
            continue
        if tok.startswith("t="):
            if mods["t"] is not None:
                raise ValueError("Duplicate '@t=' modifier. Use a single '@t=HH:MM,HH:MM,...' list.")
            values = [part.strip() for part in tok.split("=", 1)[1].split(",") if part.strip()]
            times: list[tuple[int, int]] = []
            seen: set[tuple[int, int]] = set()
            for item in values:
                hhmm = _parse_hhmm(item)
                if not hhmm:
                    raise ValueError(f"Invalid time in anchor_file @t=HH:MM[,HH:MM...]: '{item}'")
                if hhmm not in seen:
                    times.append(hhmm)
                    seen.add(hhmm)
            if not times:
                raise ValueError(f"Invalid time in anchor_file @t=HH:MM[,HH:MM...]: '{tok}'")
            mods["t"] = times[0] if len(times) == 1 else times
            continue
        if tok in ("nw", "pbd", "nbd"):
            mods["roll"] = tok
            continue
        if tok == "bd":
            mods["bd"] = True
            continue
        match = _NEXT_PREV_WD_RE.match(tok)
        if match:
            mods["roll"] = f"{match.group(1)}-wd"
            mods["wd"] = _WEEKDAYS[match.group(2)]
            continue
        match = _DAY_OFFSET_RE.match(tok)
        if match:
            mods["day_offset"] += int(match.group(1))
            continue
        match = re.fullmatch(r"([+-]\d+)bd", tok)
        if match:
            mods["business_day_offset"] += int(match.group(1))
            continue
        raise ValueError(f"Unknown anchor_file modifier '@{tok}'")
    return file_name, mods


def resolve_anchor_file_path(name: str | None, anchor_file_dir: str | None) -> str:
    file_name, _mods = parse_anchor_file_spec(name)
    if not file_name:
        return ""
    resolution = resolve_file_source_expression(file_name, anchor_file_dir, label="anchor_file")
    if len(resolution.sources) != 1:
        raise ValueError("resolve_anchor_file_path requires exactly one matching anchor_file.")
    return resolution.sources[0].path


def _resolved_anchor_sources(name: str | None, anchor_file_dir: str | None) -> FileSourceResolution:
    parsed = parse_file_source_expression(name, label="anchor_file")
    for source in parsed:
        _parse_source_mod_layers(source.pattern, source.modifier_layers)
    return resolve_file_sources(parsed, anchor_file_dir, label="anchor_file")


def unmatched_anchor_file_patterns(name: str | None, anchor_file_dir: str | None) -> tuple[str, ...]:
    return _resolved_anchor_sources(name, anchor_file_dir).unmatched_patterns


def validate_business_calendar_anchor_file(value: str) -> None:
    parsed = parse_file_source_expression(value, label="anchor_file")
    for source in parsed:
        layers, _source_time = _parse_source_mod_layers(
            source.pattern,
            source.modifier_layers,
        )
        for mods in layers:
            validate_calendar_rule_modifiers(mods, label="business calendar anchor_file")


def _parse_source_mod_layers(
    display_name: str,
    modifier_layers: tuple[str, ...],
) -> tuple[list[dict], tuple[int, int] | list[tuple[int, int]] | None]:
    layers: list[dict] = []
    source_time: tuple[int, int] | list[tuple[int, int]] | None = None
    for modifier_text in modifier_layers:
        _file_name, mods = parse_anchor_file_spec(f"source{modifier_text}")
        tval = mods.get("t")
        if tval is not None:
            if source_time is not None:
                raise ValueError(
                    f"anchor_file '{display_name}' has more than one @t modifier across its expression groups."
                )
            source_time = tval
        layers.append(mods)
    if not layers:
        layers.append(_default_mods())
    return layers, source_time


def _load_anchor_source_data(
    source: ResolvedFileSource,
    business_calendar: BusinessCalendar,
) -> tuple[frozenset[date], dict[date, str], tuple[int, int] | list[tuple[int, int]] | None]:
    dates, descriptions = load_file_date_data(
        source.path,
        label=f"anchor_file '{source.display_name}'",
    )
    layers, source_time = _parse_source_mod_layers(source.display_name, source.modifier_layers)
    for mods in layers:
        dates, descriptions = _apply_anchor_file_mods(
            dates,
            descriptions,
            mods,
            business_calendar=business_calendar,
        )
    return dates, descriptions, source_time


def _load_anchor_file_data(
    name: str | None,
    anchor_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> tuple[frozenset[date], dict[date, str]]:
    business_calendar = effective_business_calendar(business_calendar)
    resolution = _resolved_anchor_sources(name, anchor_file_dir)
    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for source in resolution.sources:
        dates, descriptions, _source_time = _load_anchor_source_data(source, business_calendar)
        out_dates.update(dates)
        for item_date, text in descriptions.items():
            if text:
                out_descriptions.setdefault(item_date, text)
    return frozenset(out_dates), out_descriptions


def _apply_anchor_file_mods(
    dates: frozenset[date],
    descriptions: dict[date, str],
    mods: dict,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> tuple[frozenset[date], dict[date, str]]:
    if not dates:
        return frozenset(), {}
    if not any(
        (
            mods.get("bd"),
            mods.get("roll"),
            int(mods.get("day_offset", 0) or 0),
            int(mods.get("business_day_offset", 0) or 0),
        )
    ):
        return dates, dict(descriptions)
    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for item_date in sorted(dates):
        transformed = _transform_anchor_file_date(
            item_date,
            mods,
            business_calendar=business_calendar,
        )
        if transformed is None:
            continue
        out_dates.add(transformed)
        text = str(descriptions.get(item_date) or "").strip()
        if text:
            out_descriptions.setdefault(transformed, text)
    return frozenset(out_dates), out_descriptions


def _transform_anchor_file_date(
    d: date,
    mods: dict,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> date | None:
    rolled = roll_apply(
        d,
        mods,
        parse_error_cls=ValueError,
        business_calendar=business_calendar,
    )
    if mods.get("bd") and not business_calendar.is_business_day(rolled):
        return None
    return apply_day_offset(rolled, mods, business_calendar=business_calendar)


def _norm_t_list(tval) -> list[tuple[int, int]]:
    if not tval:
        return []
    if isinstance(tval, tuple):
        return [tval]
    if isinstance(tval, list):
        return [item for item in tval if isinstance(item, tuple)]
    return []


def load_anchor_file_dates(
    name: str | None,
    anchor_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> frozenset[date]:
    dates, _descriptions = _load_anchor_file_data(
        name,
        anchor_file_dir,
        business_calendar=business_calendar,
    )
    return dates


def load_anchor_file_descriptions(
    name: str | None,
    anchor_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> dict[date, str]:
    _dates, descriptions = _load_anchor_file_data(
        name,
        anchor_file_dir,
        business_calendar=business_calendar,
    )
    return descriptions


def anchor_file_description_for_date(
    name: str | None,
    anchor_file_dir: str | None,
    target: date,
    *,
    business_calendar: BusinessCalendar | None = None,
) -> str | None:
    text = str(
        load_anchor_file_descriptions(
            name,
            anchor_file_dir,
            business_calendar=business_calendar,
        ).get(target)
        or ""
    ).strip()
    return text or None


def load_anchor_file_occurrence_specs(
    name: str | None,
    anchor_file_dir: str | None,
    fallback_hhmm: tuple[int, int],
    *,
    business_calendar: BusinessCalendar | None = None,
) -> list[tuple[date, tuple[int, int]]]:
    business_calendar = effective_business_calendar(business_calendar)
    resolution = _resolved_anchor_sources(name, anchor_file_dir)
    out: list[tuple[date, tuple[int, int]]] = []
    seen: set[tuple[date, tuple[int, int]]] = set()
    for source in resolution.sources:
        dates, _descriptions, source_time = _load_anchor_source_data(source, business_calendar)
        times = _norm_t_list(source_time) or [fallback_hhmm]
        for item_date in sorted(dates):
            for hhmm in times:
                occurrence = (item_date, hhmm)
                if occurrence in seen:
                    continue
                seen.add(occurrence)
                out.append(occurrence)
    out.sort()
    return out


def next_anchor_file_occurrence_after(
    name: str | None,
    anchor_file_dir: str | None,
    after_dt_local: datetime,
    fallback_hhmm: tuple[int, int],
    *,
    build_local_datetime: Callable[[date, tuple[int, int]], datetime],
    to_local: Callable[[datetime], datetime],
    business_calendar: BusinessCalendar | None = None,
) -> datetime | None:
    for d0, hhmm in load_anchor_file_occurrence_specs(
        name,
        anchor_file_dir,
        fallback_hhmm,
        business_calendar=business_calendar,
    ):
        cand_local = to_local(build_local_datetime(d0, hhmm))
        if cand_local > after_dt_local:
            return cand_local
    return None
