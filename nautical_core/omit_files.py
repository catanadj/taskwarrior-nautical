from __future__ import annotations

import os
import re
from datetime import date

from .business_calendar import DEFAULT_BUSINESS_CALENDAR, BusinessCalendar
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


def validate_omit_file_name(value: str | None) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if name in {".", ".."} or "/" in name or "\\" in name or os.path.isabs(name):
        raise ValueError("omit_file must be a file name, not a path.")
    return name


def parse_omit_file_spec(value: str | None) -> tuple[str, dict]:
    raw = str(value or "").strip()
    if not raw:
        return "", {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0, "business_day_offset": 0}
    name, mods_str = (raw.split("@", 1) + [""])[:2]
    file_name = validate_omit_file_name(name.strip())
    mods = {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0, "business_day_offset": 0}
    if not mods_str:
        return file_name, mods
    for raw_tok in mods_str.split("@"):
        tok = raw_tok.strip().lower()
        if not tok:
            continue
        if tok.startswith("t="):
            raise ValueError("omit_file does not support time modifiers (@t). Omit rules are date-based only.")
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
        raise ValueError(f"Unknown omit_file modifier '@{tok}'")
    return file_name, mods


def resolve_omit_file_path(name: str | None, omit_file_dir: str | None) -> str:
    file_name, _mods = parse_omit_file_spec(name)
    if not file_name:
        return ""
    resolution = resolve_file_source_expression(file_name, omit_file_dir, label="omit_file")
    if len(resolution.sources) != 1:
        raise ValueError("resolve_omit_file_path requires exactly one matching omit_file.")
    return resolution.sources[0].path


def _resolved_omit_sources(name: str | None, omit_file_dir: str | None) -> FileSourceResolution:
    parsed = parse_file_source_expression(name, label="omit_file")
    for source in parsed:
        _parse_source_mod_layers(source.modifier_layers)
    return resolve_file_sources(parsed, omit_file_dir, label="omit_file")


def unmatched_omit_file_patterns(name: str | None, omit_file_dir: str | None) -> tuple[str, ...]:
    return _resolved_omit_sources(name, omit_file_dir).unmatched_patterns


def _parse_source_mod_layers(modifier_layers: tuple[str, ...]) -> list[dict]:
    layers: list[dict] = []
    for modifier_text in modifier_layers:
        _file_name, mods = parse_omit_file_spec(f"source{modifier_text}")
        layers.append(mods)
    if not layers:
        _file_name, mods = parse_omit_file_spec("source")
        layers.append(mods)
    return layers


def _load_omit_source_data(
    source: ResolvedFileSource,
    business_calendar: BusinessCalendar,
) -> tuple[frozenset[date], dict[date, str]]:
    dates, descriptions = load_file_date_data(
        source.path,
        label=f"omit_file '{source.display_name}'",
    )
    for mods in _parse_source_mod_layers(source.modifier_layers):
        dates, descriptions = _apply_omit_file_mods(
            dates,
            descriptions,
            mods,
            business_calendar=business_calendar,
        )
    return dates, descriptions


def _load_omit_file_data(
    name: str | None,
    omit_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> tuple[frozenset[date], dict[date, str]]:
    resolution = _resolved_omit_sources(name, omit_file_dir)
    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for source in resolution.sources:
        dates, descriptions = _load_omit_source_data(source, business_calendar)
        out_dates.update(dates)
        for item_date, text in descriptions.items():
            if text:
                out_descriptions.setdefault(item_date, text)
    return frozenset(out_dates), out_descriptions


def _apply_omit_file_mods(
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
        transformed = _transform_omit_file_date(
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


def _transform_omit_file_date(
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


def load_omit_file_dates(
    name: str | None,
    omit_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> frozenset[date]:
    dates, _descriptions = _load_omit_file_data(
        name,
        omit_file_dir,
        business_calendar=business_calendar,
    )
    return dates


def load_omit_file_descriptions(
    name: str | None,
    omit_file_dir: str | None,
    *,
    business_calendar: BusinessCalendar = DEFAULT_BUSINESS_CALENDAR,
) -> dict[date, str]:
    _dates, descriptions = _load_omit_file_data(
        name,
        omit_file_dir,
        business_calendar=business_calendar,
    )
    return descriptions
