from __future__ import annotations

import os
import re
from datetime import date

from .file_backed_dates import load_file_date_data
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
        return "", {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0}
    name, mods_str = (raw.split("@", 1) + [""])[:2]
    file_name = validate_omit_file_name(name.strip())
    mods = {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0}
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
        raise ValueError(f"Unknown omit_file modifier '@{tok}'")
    return file_name, mods


def resolve_omit_file_path(name: str | None, omit_file_dir: str | None) -> str:
    file_name, _mods = parse_omit_file_spec(name)
    if not file_name:
        return ""
    base_dir = str(omit_file_dir or "").strip()
    if not base_dir:
        raise ValueError("omit_file_dir is not configured.")
    root = os.path.abspath(os.path.expanduser(base_dir))
    if not os.path.isdir(root):
        raise ValueError("omit_file_dir does not exist or is not a directory.")
    path = os.path.abspath(os.path.join(root, file_name))
    if os.path.dirname(path) != root:
        raise ValueError("omit_file must be a file name, not a path.")
    if not os.path.isfile(path):
        raise ValueError(f"omit_file '{file_name}' was not found in omit_file_dir.")
    return path


def _load_omit_file_data(name: str | None, omit_file_dir: str | None) -> tuple[frozenset[date], dict[date, str]]:
    file_name, mods = parse_omit_file_spec(name)
    path = resolve_omit_file_path(file_name, omit_file_dir)
    if not path:
        return frozenset(), {}
    dates, descriptions = load_file_date_data(path, label=f"omit_file '{os.path.basename(path)}'")
    return _apply_omit_file_mods(dates, descriptions, mods)


def _apply_omit_file_mods(dates: frozenset[date], descriptions: dict[date, str], mods: dict) -> tuple[frozenset[date], dict[date, str]]:
    if not dates:
        return frozenset(), {}
    if not any(
        (
            mods.get("bd"),
            mods.get("roll"),
            int(mods.get("day_offset", 0) or 0),
        )
    ):
        return dates, dict(descriptions)

    out_dates: set[date] = set()
    out_descriptions: dict[date, str] = {}
    for item_date in sorted(dates):
        transformed = _transform_omit_file_date(item_date, mods)
        if transformed is None:
            continue
        out_dates.add(transformed)
        text = str(descriptions.get(item_date) or "").strip()
        if text:
            out_descriptions.setdefault(transformed, text)
    return frozenset(out_dates), out_descriptions


def _transform_omit_file_date(d: date, mods: dict) -> date | None:
    rolled = roll_apply(d, mods, parse_error_cls=ValueError)
    if mods.get("bd") and rolled.weekday() > 4:
        return None
    return apply_day_offset(rolled, mods)


def load_omit_file_dates(name: str | None, omit_file_dir: str | None) -> frozenset[date]:
    dates, _descriptions = _load_omit_file_data(name, omit_file_dir)
    return dates


def load_omit_file_descriptions(name: str | None, omit_file_dir: str | None) -> dict[date, str]:
    _dates, descriptions = _load_omit_file_data(name, omit_file_dir)
    return descriptions
