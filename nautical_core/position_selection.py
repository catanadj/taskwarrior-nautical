from __future__ import annotations

import re
from datetime import date, timedelta


POSITION_LIMITS = {
    "week": 7,
    "month": 31,
    "quarter": 92,
    "year": 366,
}

_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)?$")
_REVERSE_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)-last$")
_MAX_POSITION_TEXT_LENGTH = 4096


def _ordinal_suffix(value: int) -> str:
    if 10 <= value % 100 <= 20:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


def _normalize_scope(scope: str) -> str:
    normalized = str(scope or "").strip().lower()
    if normalized not in POSITION_LIMITS:
        expected = ", ".join(POSITION_LIMITS)
        raise ValueError(f"Unknown selection scope '{scope}'. Expected one of: {expected}.")
    return normalized


def _parse_ordinal(token: str) -> int:
    normalized = token.strip().lower()
    if normalized == "first":
        return 1
    if normalized == "last":
        return -1

    reverse = _REVERSE_ORDINAL_RE.fullmatch(normalized)
    match = reverse or _ORDINAL_RE.fullmatch(normalized)
    if match is None:
        raise ValueError(
            f"Invalid position '{token}'. Use forms such as first, 3rd, last, or 2nd-last."
        )

    value = int(match.group(1))
    if value == 0:
        raise ValueError("Position zero is invalid; positions start at first or 1st.")

    suffix = match.group(2)
    expected_suffix = _ordinal_suffix(value)
    if suffix is not None and suffix != expected_suffix:
        replacement = f"{value}{expected_suffix}"
        if reverse is not None:
            replacement += "-last"
        raise ValueError(
            f"Invalid ordinal '{token}'. Use '{replacement}' instead."
        )
    return -value if reverse is not None else value


def parse_positions(value: str, scope: str) -> tuple[int, ...]:
    """Parse and validate a comma-separated positional selection."""
    normalized_scope = _normalize_scope(scope)
    text = str(value or "").strip()
    if not text:
        raise ValueError("Position selection cannot be empty.")
    if len(text) > _MAX_POSITION_TEXT_LENGTH:
        raise ValueError(
            f"Position selection is too long (maximum {_MAX_POSITION_TEXT_LENGTH} characters)."
        )

    positions: list[int] = []
    seen: set[int] = set()
    for raw_token in text.split(","):
        if not raw_token.strip():
            raise ValueError("Position selection contains an empty item.")
        position = _parse_ordinal(raw_token)
        limit = POSITION_LIMITS[normalized_scope]
        if abs(position) > limit:
            raise ValueError(
                f"Position '{raw_token.strip()}' exceeds the {normalized_scope} limit of {limit}."
            )
        if position not in seen:
            positions.append(position)
            seen.add(position)
    return tuple(positions)


def period_bounds(scope: str, value: date) -> tuple[date, date]:
    """Return inclusive period boundaries containing ``value``."""
    normalized_scope = _normalize_scope(scope)
    if not isinstance(value, date):
        raise TypeError("Period boundary value must be a date.")
    day = date(value.year, value.month, value.day)

    if normalized_scope == "week":
        start = day - timedelta(days=day.weekday())
        return start, start + timedelta(days=6)

    if normalized_scope == "month":
        start = date(day.year, day.month, 1)
        if day.month == 12:
            next_start = date(day.year + 1, 1, 1)
        else:
            next_start = date(day.year, day.month + 1, 1)
        return start, next_start - timedelta(days=1)

    if normalized_scope == "quarter":
        start_month = ((day.month - 1) // 3) * 3 + 1
        start = date(day.year, start_month, 1)
        if start_month == 10:
            next_start = date(day.year + 1, 1, 1)
        else:
            next_start = date(day.year, start_month + 3, 1)
        return start, next_start - timedelta(days=1)

    return date(day.year, 1, 1), date(day.year, 12, 31)


__all__ = ("POSITION_LIMITS", "parse_positions", "period_bounds")
