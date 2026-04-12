from __future__ import annotations

import csv
import io
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from .schedule_utils import apply_day_offset, roll_apply


_CACHE_BY_PATH: dict[str, tuple[int, int, frozenset[date], dict[date, str]]] = {}
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


def _expand_date_spec(spec: str, *, label: str) -> set[date]:
    text = str(spec or "").strip()
    if not text:
        return set()
    try:
        if ".." not in text:
            return {date.fromisoformat(text)}
        left, right = text.split("..", 1)
        start = date.fromisoformat(left.strip())
        end = date.fromisoformat(right.strip())
    except Exception:
        raise ValueError(f"{label} contains an invalid date or range.")
    if end < start:
        raise ValueError(f"{label} contains a backward date range.")
    out: set[date] = set()
    cur = start
    while cur <= end:
        out.add(cur)
        cur += timedelta(days=1)
    return out


def _iter_content_lines(text: str) -> Iterable[tuple[int, str]]:
    for line_no, raw in enumerate(text.splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        yield line_no, raw


def _looks_like_csv(non_comment_lines: list[tuple[int, str]]) -> bool:
    if not non_comment_lines:
        return False
    _line_no, first = non_comment_lines[0]
    try:
        header = next(csv.reader([first]))
    except Exception:
        return False
    norm = {str(col or "").strip().strip('"').lower() for col in header}
    return "date" in norm


def _parse_csv_dates_and_descriptions(text: str, *, label: str) -> tuple[frozenset[date], dict[date, str]]:
    non_comment = [line for _no, line in _iter_content_lines(text)]
    reader = csv.DictReader(io.StringIO("\n".join(non_comment)))
    fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"{label} is empty.")
    date_key = None
    description_key = None
    for field in fieldnames:
        field_name = str(field or "").strip().strip('"').lower()
        if field_name == "date":
            date_key = field
        elif field_name == "description":
            description_key = field
    if date_key is None:
        raise ValueError(f"{label} CSV must contain a 'date' column.")
    out: set[date] = set()
    descriptions: dict[date, str] = {}
    for row_no, row in enumerate(reader, 2):
        if not isinstance(row, dict):
            continue
        value = str(row.get(date_key) or "").strip()
        if not value:
            continue
        row_dates = _expand_date_spec(value, label=f"{label} line {row_no}")
        out.update(row_dates)
        description = str(row.get(description_key) or "").strip() if description_key is not None else ""
        if description:
            for item_date in row_dates:
                descriptions.setdefault(item_date, description)
    return frozenset(out), descriptions


def _parse_text_dates(text: str, *, label: str) -> frozenset[date]:
    out: set[date] = set()
    for line_no, raw in _iter_content_lines(text):
        out.update(_expand_date_spec(raw.strip(), label=f"{label} line {line_no}"))
    return frozenset(out)


def _load_omit_file_data(name: str | None, omit_file_dir: str | None) -> tuple[frozenset[date], dict[date, str]]:
    file_name, mods = parse_omit_file_spec(name)
    path = resolve_omit_file_path(file_name, omit_file_dir)
    if not path:
        return frozenset(), {}
    st = os.stat(path)
    cached = _CACHE_BY_PATH.get(path)
    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2], dict(cached[3])
    text = Path(path).read_text(encoding="utf-8-sig")
    non_comment = list(_iter_content_lines(text))
    if not non_comment:
        dates: frozenset[date] = frozenset()
        descriptions: dict[date, str] = {}
    elif _looks_like_csv(non_comment):
        dates, descriptions = _parse_csv_dates_and_descriptions(text, label=f"omit_file '{os.path.basename(path)}'")
    else:
        dates = _parse_text_dates(text, label=f"omit_file '{os.path.basename(path)}'")
        descriptions = {}
    _CACHE_BY_PATH[path] = (st.st_mtime_ns, st.st_size, dates, dict(descriptions))
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
