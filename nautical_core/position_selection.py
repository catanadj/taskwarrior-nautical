from __future__ import annotations

import json
import re
from datetime import date, timedelta
from functools import lru_cache
from typing import Any, Callable, TypedDict


POSITION_LIMITS = {
    "week": 7,
    "month": 31,
    "quarter": 92,
    "year": 366,
}

_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)?$")
_REVERSE_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)-last$")
_GROUP_SELECTION_RE = re.compile(r"^in-(week|month|quarter|year)=(.*)$")
_MAX_POSITION_TEXT_LENGTH = 4096
_DEFAULT_PERIOD_SCAN_LIMIT = 128


class SelectionNode(TypedDict):
    kind: str
    scope: str
    positions: tuple[int, ...]
    expr: list[list[dict[str, Any]]]
    mods: dict[str, Any]


def is_selection_node(value: object) -> bool:
    return isinstance(value, dict) and value.get("kind") == "select"


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


def parse_group_selection_modifier(
    value: str,
) -> tuple[str, tuple[int, ...], str] | None:
    """Parse the monthly selector and return its remaining modifier suffix."""
    tokens = [token.strip() for token in str(value or "").split("@") if token.strip()]
    selection_indexes = [
        index for index, token in enumerate(tokens) if token.lower().startswith("in-")
    ]
    if not selection_indexes:
        return None
    if len(selection_indexes) != 1:
        raise ValueError("A group can contain only one positional selector.")
    if selection_indexes[0] != 0:
        raise ValueError("The positional selector must appear before its post-selection modifiers.")

    match = _GROUP_SELECTION_RE.fullmatch(tokens[0].lower())
    if match is None:
        raise ValueError(
            "Invalid positional selector. Use forms such as "
            "'(expression)@in-month=first' or '@in-year=last'."
        )
    scope, raw_positions = match.groups()
    remaining = "".join(f"@{token}" for token in tokens[1:])
    return scope, parse_positions(raw_positions, scope), remaining


def format_position(position: int) -> str:
    if position == 1:
        return "first"
    if position == -1:
        return "last"
    if position > 0:
        return f"{position}{_ordinal_suffix(position)}"
    value = abs(position)
    return f"{value}{_ordinal_suffix(value)}-last"


def format_positions(positions: list[int] | tuple[int, ...]) -> str:
    return ",".join(format_position(position) for position in positions)


def describe_selection(node: object, inner_description: str) -> str:
    normalized = normalize_selection_node(node)
    labels = [format_position(position) for position in normalized["positions"]]
    if len(labels) == 1:
        position_text = labels[0]
    elif len(labels) == 2:
        position_text = f"{labels[0]} and {labels[1]}"
    else:
        position_text = ", ".join(labels[:-1]) + f", and {labels[-1]}"
    noun = "date" if len(labels) == 1 else "dates"
    inner = inner_description.strip() or "the candidate expression"
    return f"the {position_text} matching {noun} from {inner} in each {normalized['scope']}"


def normalize_selection_node(node: object) -> SelectionNode:
    """Validate and normalize an internal positional-selection factor."""
    if not isinstance(node, dict) or node.get("kind") != "select":
        raise ValueError("Selection node must be a mapping with kind='select'.")

    scope = _normalize_scope(node.get("scope"))
    raw_positions = node.get("positions")
    if not isinstance(raw_positions, (list, tuple)) or not raw_positions:
        raise ValueError("Selection node positions must be a non-empty list or tuple.")

    positions: list[int] = []
    seen: set[int] = set()
    for raw_position in raw_positions:
        if isinstance(raw_position, bool) or not isinstance(raw_position, int):
            raise ValueError("Selection node positions must contain integers only.")
        if raw_position == 0:
            raise ValueError("Selection node position zero is invalid.")
        if abs(raw_position) > POSITION_LIMITS[scope]:
            raise ValueError(
                f"Selection node position {raw_position} exceeds the {scope} limit "
                f"of {POSITION_LIMITS[scope]}."
            )
        if raw_position not in seen:
            positions.append(raw_position)
            seen.add(raw_position)

    expr = node.get("expr")
    if not isinstance(expr, list) or not expr:
        raise ValueError("Selection node expression must contain at least one OR term.")
    for term in expr:
        if not isinstance(term, list) or not term:
            raise ValueError("Selection node expression terms must be non-empty lists.")
        for factor in term:
            if not isinstance(factor, dict):
                raise ValueError("Selection node expression factors must be mappings.")
            if factor.get("kind") == "select":
                raise ValueError("Nested positional selections are not supported.")

    mods = node.get("mods") or {}
    if not isinstance(mods, dict):
        raise ValueError("Selection node modifiers must be a mapping.")
    return {
        "kind": "select",
        "scope": scope,
        "positions": tuple(positions),
        "expr": expr,
        "mods": dict(mods),
    }


def validate_public_selection_node(node: object) -> SelectionNode:
    """Apply deterministic public positional-selection limits."""
    normalized = normalize_selection_node(node)
    selector = f"@in-{normalized['scope']}"
    mods = normalized["mods"]
    active = {
        key
        for key, value in mods.items()
        if value not in (None, False, 0, 0.0, "", [])
    }
    unsupported = active - {
        "t",
        "roll",
        "wd",
        "bd",
        "day_offset",
        "business_day_offset",
    }
    if unsupported:
        label = ", ".join(sorted(unsupported))
        raise ValueError(f"{selector} does not support post-selection modifier(s): {label}.")
    if mods.get("bd"):
        raise ValueError(
            f"@bd is a candidate filter and is not supported after {selector}."
        )
    roll = mods.get("roll")
    if roll not in (None, "pbd", "nbd", "nw", "next-wd", "prev-wd"):
        raise ValueError(f"Unsupported {selector} roll modifier: {roll}.")
    if roll in ("next-wd", "prev-wd"):
        weekday = mods.get("wd")
        if isinstance(weekday, bool) or not isinstance(weekday, int) or not 0 <= weekday < 7:
            raise ValueError(f"Weekday roll modifiers after {selector} require a valid weekday.")
    elif mods.get("wd") is not None:
        raise ValueError(f"{selector} weekday values require @next-<weekday> or @prev-<weekday>.")
    for key in ("day_offset", "business_day_offset"):
        value = mods.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{selector} {key} must be an integer.")

    for term in normalized["expr"]:
        for factor in term:
            spec = str(factor.get("spec") or factor.get("value") or "").lower()
            if "rand" in spec:
                raise ValueError(f"{selector} candidate expressions cannot contain random selectors.")
            mods = factor.get("mods") or {}
            if any(value not in (None, False, 0, 0.0, "", []) for value in mods.values()):
                raise ValueError(
                    f"{selector} candidate expressions cannot contain modifiers yet."
                )
    return normalized


def _has_date_modifiers(mods: dict[str, Any]) -> bool:
    return bool(
        mods.get("roll")
        or int(mods.get("day_offset", 0) or 0)
        or int(mods.get("business_day_offset", 0) or 0)
    )


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


def next_period_start(scope: str, value: date) -> date:
    """Return the first day of the period following the one containing ``value``."""
    _, end = period_bounds(scope, value)
    return end + timedelta(days=1)


def select_positions(candidates: list[date] | tuple[date, ...], positions: tuple[int, ...]) -> tuple[date, ...]:
    """Select signed positions from sorted unique candidate dates."""
    ordered = sorted(set(candidates))
    selected: set[date] = set()
    for position in positions:
        if isinstance(position, bool) or not isinstance(position, int) or position == 0:
            raise ValueError("Selected positions must be non-zero integers.")
        index = position - 1 if position > 0 else position
        if -len(ordered) <= index < len(ordered):
            selected.add(ordered[index])
    return tuple(sorted(selected))


def _canonical_expr(expr: list[list[dict[str, Any]]]) -> str:
    try:
        canonical_terms = [
            sorted(
                term,
                key=lambda factor: json.dumps(
                    factor,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
            for term in expr
        ]
        canonical_terms.sort(
            key=lambda term: json.dumps(
                term,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return json.dumps(
            canonical_terms,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Selection node expression is not JSON serializable: {exc}") from exc


@lru_cache(maxsize=2048)
def _matching_candidates_cached(
    expr_key: str,
    _scope: str,
    start: date,
    end: date,
    default_seed: date,
    seed_base: str | None,
    _calendar_fingerprint: str,
    matches_on: Callable[..., bool],
) -> tuple[date, ...]:
    expr = json.loads(expr_key)
    matches: list[date] = []
    current = start
    while current <= end:
        if any(
            all(
                matches_on(factor, current, default_seed, seed_base=seed_base)
                for factor in term
            )
            for term in expr
        ):
            matches.append(current)
        current += timedelta(days=1)
    return tuple(matches)


def _selected_candidates_in_period(
    node: SelectionNode,
    value: date,
    *,
    matches_on: Callable[..., bool],
    default_seed: date,
    seed_base: object = None,
    calendar_fingerprint: str = "",
) -> tuple[date, ...]:
    start, end = period_bounds(node["scope"], value)
    candidates = _matching_candidates_cached(
        _canonical_expr(node["expr"]),
        node["scope"],
        start,
        end,
        default_seed,
        None if seed_base is None else str(seed_base),
        str(calendar_fingerprint or ""),
        matches_on,
    )
    return select_positions(candidates, node["positions"])


def selected_candidates_in_period(
    node: object,
    value: date,
    *,
    matches_on: Callable[..., bool],
    default_seed: date,
    seed_base: object = None,
    calendar_fingerprint: str = "",
) -> tuple[date, ...]:
    """Return selected dates from the scope period containing ``value``."""
    normalized = normalize_selection_node(node)
    if not callable(matches_on):
        raise TypeError("Selection matcher must be callable.")
    if not isinstance(default_seed, date):
        raise TypeError("Selection default seed must be a date.")
    return _selected_candidates_in_period(
        normalized,
        value,
        matches_on=matches_on,
        default_seed=date(default_seed.year, default_seed.month, default_seed.day),
        seed_base=seed_base,
        calendar_fingerprint=calendar_fingerprint,
    )


def next_selected_date(
    node: object,
    after_date: date,
    *,
    matches_on: Callable[..., bool],
    default_seed: date,
    seed_base: object = None,
    calendar_fingerprint: str = "",
    max_periods: int = _DEFAULT_PERIOD_SCAN_LIMIT,
) -> date | None:
    """Return the next selected date strictly after ``after_date``."""
    normalized = normalize_selection_node(node)
    if not isinstance(after_date, date):
        raise TypeError("Selection reference must be a date.")
    if not callable(matches_on):
        raise TypeError("Selection matcher must be callable.")
    if not isinstance(default_seed, date):
        raise TypeError("Selection default seed must be a date.")
    if isinstance(max_periods, bool) or not isinstance(max_periods, int) or max_periods < 1:
        raise ValueError("Selection period scan limit must be a positive integer.")

    after_day = date(after_date.year, after_date.month, after_date.day)
    seed_day = date(default_seed.year, default_seed.month, default_seed.day)
    probe = after_day
    for _ in range(max_periods):
        selected = _selected_candidates_in_period(
            normalized,
            probe,
            matches_on=matches_on,
            default_seed=seed_day,
            seed_base=seed_base,
            calendar_fingerprint=calendar_fingerprint,
        )
        for candidate in selected:
            if candidate > after_day:
                return candidate
        try:
            probe = next_period_start(normalized["scope"], probe)
        except (OverflowError, ValueError):
            return None
    return None


def next_selected_date_with_modifiers(
    node: object,
    after_date: date,
    *,
    matches_on: Callable[..., bool],
    apply_modifiers: Callable[[date, dict[str, Any]], date],
    default_seed: date,
    seed_base: object = None,
    calendar_fingerprint: str = "",
    max_periods: int = _DEFAULT_PERIOD_SCAN_LIMIT,
) -> date | None:
    """Select a base occurrence, then apply its date modifiers."""
    normalized = normalize_selection_node(node)
    if not isinstance(after_date, date):
        raise TypeError("Selection reference must be a date.")
    if not isinstance(default_seed, date):
        raise TypeError("Selection default seed must be a date.")
    if isinstance(max_periods, bool) or not isinstance(max_periods, int) or max_periods < 1:
        raise ValueError("Selection period scan limit must be a positive integer.")
    if not _has_date_modifiers(normalized["mods"]):
        return next_selected_date(
            normalized,
            after_date,
            matches_on=matches_on,
            default_seed=default_seed,
            seed_base=seed_base,
            calendar_fingerprint=calendar_fingerprint,
            max_periods=max_periods,
        )
    if not callable(apply_modifiers):
        raise TypeError("Selection modifier application must be callable.")

    after_day = date(after_date.year, after_date.month, after_date.day)
    seed_day = date(default_seed.year, default_seed.month, default_seed.day)

    # Supported rolls and fixed offsets are monotonic. Find the latest selected
    # base whose transformed date is not after the reference, then advance once.
    boundary = None
    period_probe = after_day
    for _ in range(max_periods):
        selected = _selected_candidates_in_period(
            normalized,
            period_probe,
            matches_on=matches_on,
            default_seed=seed_day,
            seed_base=seed_base,
            calendar_fingerprint=calendar_fingerprint,
        )
        for base in reversed(selected):
            if apply_modifiers(base, normalized["mods"]) <= after_day:
                boundary = base
                break
        if boundary is not None:
            break
        period_start, _ = period_bounds(normalized["scope"], period_probe)
        try:
            period_probe = period_start - timedelta(days=1)
        except (OverflowError, ValueError):
            return None

    if boundary is None:
        return None

    probe = boundary
    max_occurrences = max_periods * max(1, len(normalized["positions"]))
    for _ in range(max_occurrences):
        base = next_selected_date(
            normalized,
            probe,
            matches_on=matches_on,
            default_seed=seed_day,
            seed_base=seed_base,
            calendar_fingerprint=calendar_fingerprint,
            max_periods=max_periods,
        )
        if base is None:
            return None
        transformed = apply_modifiers(base, normalized["mods"])
        if transformed > after_day:
            return transformed
        probe = base
    return None


def clear_candidate_cache() -> None:
    _matching_candidates_cached.cache_clear()


def candidate_cache_info():
    return _matching_candidates_cached.cache_info()


__all__ = (
    "POSITION_LIMITS",
    "SelectionNode",
    "candidate_cache_info",
    "clear_candidate_cache",
    "describe_selection",
    "format_position",
    "format_positions",
    "is_selection_node",
    "next_period_start",
    "next_selected_date",
    "next_selected_date_with_modifiers",
    "normalize_selection_node",
    "parse_positions",
    "parse_group_selection_modifier",
    "period_bounds",
    "select_positions",
    "selected_candidates_in_period",
    "validate_public_selection_node",
)
