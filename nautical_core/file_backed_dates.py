from __future__ import annotations

import csv
import hashlib
import io
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable


_CACHE_BY_PATH: dict[str, tuple[int, int, str, frozenset[date], dict[date, str]]] = {}


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
    return "date" in norm or len(header) > 1


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
        cols = ", ".join(str(field or "").strip() for field in fieldnames if str(field or "").strip())
        suffix = f" Found columns: {cols}." if cols else ""
        raise ValueError(f"{label} CSV must contain a 'date' column.{suffix}")
    out: set[date] = set()
    descriptions: dict[date, str] = {}
    total_rows = 0
    nonempty_date_rows = 0
    for row_no, row in enumerate(reader, 2):
        if not isinstance(row, dict):
            continue
        total_rows += 1
        value = str(row.get(date_key) or "").strip()
        if not value:
            continue
        nonempty_date_rows += 1
        row_dates = _expand_date_spec(value, label=f"{label} line {row_no}")
        out.update(row_dates)
        description = str(row.get(description_key) or "").strip() if description_key is not None else ""
        if description:
            for item_date in row_dates:
                descriptions.setdefault(item_date, description)
    if not out:
        raise ValueError(
            f"{label} CSV did not contain any usable dates in the 'date' column "
            f"({total_rows} data row(s), {nonempty_date_rows} non-empty date value(s))."
        )
    return frozenset(out), descriptions


def _parse_text_dates(text: str, *, label: str) -> frozenset[date]:
    out: set[date] = set()
    for line_no, raw in _iter_content_lines(text):
        out.update(_expand_date_spec(raw.strip(), label=f"{label} line {line_no}"))
    return frozenset(out)


def load_file_date_data(path: str, *, label: str) -> tuple[frozenset[date], dict[date, str]]:
    if not path:
        return frozenset(), {}
    st = os.stat(path)
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    cached = _CACHE_BY_PATH.get(path)
    if cached and cached[0] == st.st_mtime_ns and cached[1] == st.st_size and cached[2] == digest:
        return cached[3], dict(cached[4])

    text = raw.decode("utf-8-sig")
    non_comment = list(_iter_content_lines(text))
    if not non_comment:
        raise ValueError(f"{label} is empty or has no date rows.")
    if _looks_like_csv(non_comment):
        dates, descriptions = _parse_csv_dates_and_descriptions(text, label=label)
    else:
        dates = _parse_text_dates(text, label=label)
        descriptions = {}
        if not dates:
            raise ValueError(f"{label} did not contain any usable dates.")
    _CACHE_BY_PATH[path] = (st.st_mtime_ns, st.st_size, digest, dates, dict(descriptions))
    return dates, descriptions
