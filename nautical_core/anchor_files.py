from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Callable

from .file_backed_dates import load_file_date_data
from .schedule_utils import apply_day_offset, roll_apply


_WEEKDAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}
_NEXT_PREV_WD_RE = re.compile(r"^(next|prev)-(mon|tue|wed|thu|fri|sat|sun)$")
_DAY_OFFSET_RE = re.compile(r"^([+-]\d+)d$")
_HHMM_RE = re.compile(r"^(\d{2}):(\d{2})$")
_HOUR_PAD_RE = re.compile(r"^(\d):(\d{2})(?::\d{2})?$")


def _default_mods() -> dict:
    return {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0}


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
        raise ValueError(f"Unknown anchor_file modifier '@{tok}'")
    return file_name, mods


def resolve_anchor_file_path(name: str | None, anchor_file_dir: str | None) -> str:
    file_name, _mods = parse_anchor_file_spec(name)
    if not file_name:
        return ""
    base_dir = str(anchor_file_dir or "").strip()
    if not base_dir:
        raise ValueError("anchor_file_dir is not configured.")
    root = os.path.abspath(os.path.expanduser(base_dir))
    if not os.path.isdir(root):
        raise ValueError("anchor_file_dir does not exist or is not a directory.")
    path = os.path.abspath(os.path.join(root, file_name))
    if os.path.dirname(path) != root:
        raise ValueError("anchor_file must be a file name, not a path.")
    if not os.path.isfile(path):
        raise ValueError(f"anchor_file '{file_name}' was not found in anchor_file_dir.")
    return path


def _load_anchor_file_data(name: str | None, anchor_file_dir: str | None) -> tuple[frozenset[date], dict[date, str]]:
    file_name, mods = parse_anchor_file_spec(name)
    path = resolve_anchor_file_path(file_name, anchor_file_dir)
    if not path:
        return frozenset(), {}
    dates, descriptions = load_file_date_data(path, label=f"anchor_file '{os.path.basename(path)}'")
    return _apply_anchor_file_mods(dates, descriptions, mods)


def _apply_anchor_file_mods(dates: frozenset[date], descriptions: dict[date, str], mods: dict) -> tuple[frozenset[date], dict[date, str]]:
    if not dates:
        return frozenset(), {}
    if not any((mods.get("bd"), mods.get("roll"), int(mods.get("day_offset", 0) or 0))):
        return dates, dict(descriptions)
    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for item_date in sorted(dates):
        transformed = _transform_anchor_file_date(item_date, mods)
        if transformed is None:
            continue
        out_dates.add(transformed)
        text = str(descriptions.get(item_date) or "").strip()
        if text:
            out_descriptions.setdefault(transformed, text)
    return frozenset(out_dates), out_descriptions


def _transform_anchor_file_date(d: date, mods: dict) -> date | None:
    rolled = roll_apply(d, mods, parse_error_cls=ValueError)
    if mods.get("bd") and rolled.weekday() > 4:
        return None
    return apply_day_offset(rolled, mods)


def _norm_t_list(tval) -> list[tuple[int, int]]:
    if not tval:
        return []
    if isinstance(tval, tuple):
        return [tval]
    if isinstance(tval, list):
        return [item for item in tval if isinstance(item, tuple)]
    return []


def load_anchor_file_dates(name: str | None, anchor_file_dir: str | None) -> frozenset[date]:
    dates, _descriptions = _load_anchor_file_data(name, anchor_file_dir)
    return dates


def load_anchor_file_descriptions(name: str | None, anchor_file_dir: str | None) -> dict[date, str]:
    _dates, descriptions = _load_anchor_file_data(name, anchor_file_dir)
    return descriptions


def anchor_file_description_for_date(name: str | None, anchor_file_dir: str | None, target: date) -> str | None:
    text = str(load_anchor_file_descriptions(name, anchor_file_dir).get(target) or "").strip()
    return text or None


def next_anchor_file_occurrence_after(
    name: str | None,
    anchor_file_dir: str | None,
    after_dt_local: datetime,
    fallback_hhmm: tuple[int, int],
    *,
    build_local_datetime: Callable[[date, tuple[int, int]], datetime],
    to_local: Callable[[datetime], datetime],
) -> datetime | None:
    dates = sorted(load_anchor_file_dates(name, anchor_file_dir))
    if not dates:
        return None
    _file_name, mods = parse_anchor_file_spec(name)
    tlist = _norm_t_list(mods.get("t")) or [fallback_hhmm]
    for d0 in dates:
        for hhmm in tlist:
            cand_local = to_local(build_local_datetime(d0, hhmm))
            if cand_local > after_dt_local:
                return cand_local
    return None
