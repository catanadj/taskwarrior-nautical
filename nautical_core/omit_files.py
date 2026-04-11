from __future__ import annotations

import csv
import io
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable


_CACHE_BY_PATH: dict[str, tuple[int, int, frozenset[date]]] = {}


def validate_omit_file_name(value: str | None) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if name in {".", ".."} or "/" in name or "\\" in name or os.path.isabs(name):
        raise ValueError("omit_file must be a file name, not a path.")
    return name


def resolve_omit_file_path(name: str | None, omit_file_dir: str | None) -> str:
    file_name = validate_omit_file_name(name)
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


def _parse_csv_dates(text: str, *, label: str) -> frozenset[date]:
    non_comment = [line for _no, line in _iter_content_lines(text)]
    reader = csv.DictReader(io.StringIO("\n".join(non_comment)))
    fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError(f"{label} is empty.")
    date_key = None
    for field in fieldnames:
        if str(field or "").strip().strip('"').lower() == "date":
            date_key = field
            break
    if date_key is None:
        raise ValueError(f"{label} CSV must contain a 'date' column.")
    out: set[date] = set()
    for row_no, row in enumerate(reader, 2):
        if not isinstance(row, dict):
            continue
        value = str(row.get(date_key) or "").strip()
        if not value:
            continue
        out.update(_expand_date_spec(value, label=f"{label} line {row_no}"))
    return frozenset(out)


def _parse_text_dates(text: str, *, label: str) -> frozenset[date]:
    out: set[date] = set()
    for line_no, raw in _iter_content_lines(text):
        out.update(_expand_date_spec(raw.strip(), label=f"{label} line {line_no}"))
    return frozenset(out)


def load_omit_file_dates(name: str | None, omit_file_dir: str | None) -> frozenset[date]:
    path = resolve_omit_file_path(name, omit_file_dir)
    if not path:
        return frozenset()
    st = os.stat(path)
    cached = _CACHE_BY_PATH.get(path)
    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size:
        return cached[2]
    text = Path(path).read_text(encoding="utf-8-sig")
    non_comment = list(_iter_content_lines(text))
    if not non_comment:
        dates: frozenset[date] = frozenset()
    elif _looks_like_csv(non_comment):
        dates = _parse_csv_dates(text, label=f"omit_file '{os.path.basename(path)}'")
    else:
        dates = _parse_text_dates(text, label=f"omit_file '{os.path.basename(path)}'")
    _CACHE_BY_PATH[path] = (st.st_mtime_ns, st.st_size, dates)
    return dates
