"""Completion-period parsing and deterministic interval selection."""

from __future__ import annotations

import hashlib
import re
from datetime import timedelta
from typing import Any


_CP_RE = re.compile(
    r"^P(?:(?P<w>\d+)W)?(?:(?P<d>\d+)D)?(?:T(?:(?P<h>\d+)H)?(?:(?P<m>\d+)M)?(?:(?P<s>\d+)S)?)?$",
    re.I,
)
_CP_TOKEN_RE = re.compile(r"(?P<sign>[+-]?)(?P<n>\d+)(?P<u>w|d|h|m|s)", re.I)
_CP_RAND_RE = re.compile(r"^rand\((?P<lo>.+)\.\.(?P<hi>.+)\)$", re.I)
_CP_JITTER_RE = re.compile(r"^(?P<base>.+)~(?P<spread>.+)$", re.I)
_CP_REPEAT_RE = re.compile(r"^(?P<base>.+)\*(?P<count>\d+)$", re.I)
_MAX_CP_SEQUENCE_ITEMS = 1000


def _expanded_cp_parts(cp: str) -> tuple[list[str], str | None]:
    """Expand bounded ``duration*count`` groups before normal CP parsing."""
    raw_parts = [part.strip() for part in str(cp or "").strip().split(",")]
    expanded: list[str] = []
    for position, part in enumerate(raw_parts, start=1):
        repeat = _CP_REPEAT_RE.fullmatch(part)
        if repeat is None:
            if "*" in part:
                return [], (
                    f"invalid repeat '{part}' at position {position}: "
                    "expected <duration>*<count>"
                )
            expanded.append(part)
            continue
        base = repeat.group("base").strip()
        count = int(repeat.group("count"))
        if not base or count < 1:
            return [], (
                f"invalid repeat '{part}' at position {position}: "
                "the duration and a positive repeat count are required"
            )
        if len(expanded) + count > _MAX_CP_SEQUENCE_ITEMS:
            return [], (
                f"cp sequence expands beyond {_MAX_CP_SEQUENCE_ITEMS} items at position {position}; "
                "use a shorter repeat count"
            )
        expanded.extend([base] * count)
    return expanded, None


def _canonical_cp_sequence(cp: str) -> str:
    parts, error = _expanded_cp_parts(cp)
    return str(cp).strip() if error is not None else ",".join(parts)


def parse_cp_duration(dur: str):
    """Parse one ISO-8601 or Nautical completion-period duration."""
    if not dur:
        return None
    s = str(dur).strip()
    if not s or "," in s:
        return None
    match = _CP_RE.match(s)
    if match:
        return timedelta(
            weeks=int(match.group("w") or 0),
            days=int(match.group("d") or 0),
            hours=int(match.group("h") or 0),
            minutes=int(match.group("m") or 0),
            seconds=int(match.group("s") or 0),
        )

    total = timedelta()
    pos = 0
    for token in _CP_TOKEN_RE.finditer(s):
        if token.start() != pos:
            return None
        sign = -1 if token.group("sign") == "-" else 1
        number = sign * int(token.group("n"))
        unit = token.group("u").lower()
        if unit == "w":
            total += timedelta(weeks=number)
        elif unit == "d":
            total += timedelta(days=number)
        elif unit == "h":
            total += timedelta(hours=number)
        elif unit == "m":
            total += timedelta(minutes=number)
        elif unit == "s":
            total += timedelta(seconds=number)
        pos = token.end()
    if pos != len(s) or pos == 0:
        return None
    return total


def _cp_rand_granularity_seconds(lo: timedelta, hi: timedelta) -> int:
    lo_s = int(lo.total_seconds())
    hi_s = int(hi.total_seconds())
    for granularity in (86400, 3600, 60):
        if lo_s % granularity == 0 and hi_s % granularity == 0:
            return granularity
    return 1


def _parse_cp_token(part: str) -> dict[str, Any] | None:
    raw = str(part or "").strip()
    random_match = _CP_RAND_RE.match(raw)
    if not random_match:
        jitter_match = _CP_JITTER_RE.match(raw)
        if jitter_match:
            base_raw = jitter_match.group("base").strip()
            spread_raw = jitter_match.group("spread").strip()
            base = parse_cp_duration(base_raw)
            spread = parse_cp_duration(spread_raw)
            if base is None or spread is None or spread < timedelta() or base - spread < timedelta():
                return None
            lo = base - spread
            hi = base + spread
            return {
                "kind": "rand",
                "raw": raw,
                "base_raw": base_raw,
                "spread_raw": spread_raw,
                "lo_raw": str(lo),
                "hi_raw": str(hi),
                "lo": lo,
                "hi": hi,
                "granularity_seconds": _cp_rand_granularity_seconds(lo, hi),
            }
        duration = parse_cp_duration(raw)
        if duration is None:
            return None
        return {"kind": "fixed", "raw": raw, "duration": duration}

    lo_raw = random_match.group("lo").strip()
    hi_raw = random_match.group("hi").strip()
    lo = parse_cp_duration(lo_raw)
    hi = parse_cp_duration(hi_raw)
    if lo is None or hi is None or lo > hi:
        return None
    return {
        "kind": "rand",
        "raw": raw,
        "lo_raw": lo_raw,
        "hi_raw": hi_raw,
        "lo": lo,
        "hi": hi,
        "granularity_seconds": _cp_rand_granularity_seconds(lo, hi),
    }


def cp_sequence_parse_error(cp: str) -> str | None:
    if not cp:
        return "cp is empty"
    raw = str(cp).strip()
    if not raw:
        return "cp is empty"
    parts, expansion_error = _expanded_cp_parts(raw)
    if expansion_error is not None:
        return expansion_error
    for idx, part in enumerate(parts, start=1):
        if not part:
            return f"empty duration at position {idx}"
        random_match = _CP_RAND_RE.match(part)
        if random_match:
            lo_raw = random_match.group("lo").strip()
            hi_raw = random_match.group("hi").strip()
            if not lo_raw or not hi_raw:
                return f"invalid random cp range '{part}' at position {idx}: both bounds are required"
            lo = parse_cp_duration(lo_raw)
            hi = parse_cp_duration(hi_raw)
            if lo is None or hi is None:
                return f"invalid random cp range '{part}' at position {idx}: invalid duration bound"
            if lo > hi:
                return f"invalid random cp range '{part}' at position {idx}: lower bound must be <= upper bound"
            continue
        if part.lower().startswith("rand("):
            return f"invalid random cp range '{part}' at position {idx}: expected rand(<duration>..<duration>)"
        jitter_match = _CP_JITTER_RE.match(part)
        if jitter_match:
            base_raw = jitter_match.group("base").strip()
            spread_raw = jitter_match.group("spread").strip()
            if not base_raw or not spread_raw:
                return f"invalid jitter cp range '{part}' at position {idx}: expected <duration>~<duration>"
            base = parse_cp_duration(base_raw)
            spread = parse_cp_duration(spread_raw)
            if base is None or spread is None:
                return f"invalid jitter cp range '{part}' at position {idx}: invalid duration bound"
            if spread < timedelta():
                return f"invalid jitter cp range '{part}' at position {idx}: spread must be >= 0"
            if base - spread < timedelta():
                return f"invalid jitter cp range '{part}' at position {idx}: lower bound must be >= 0"
            continue
        if "~" in part:
            return f"invalid jitter cp range '{part}' at position {idx}: expected <duration>~<duration>"
        if parse_cp_duration(part) is None:
            return f"invalid duration '{part}' at position {idx}"
    return None


def parse_cp_sequence_tokens(cp: str):
    """Parse CP into fixed/random period tokens without resolving randomness."""
    if cp_sequence_parse_error(cp):
        return None
    tokens = []
    parts, expansion_error = _expanded_cp_parts(cp)
    if expansion_error is not None:
        return None
    for part in parts:
        token = _parse_cp_token(part)
        if token is None:
            return None
        tokens.append(token)
    return tokens


def _cp_rand_duration_for_token(
    token: dict[str, Any],
    *,
    cp: str,
    link_no: int,
    token_index: int,
    chain_id: str | None = None,
) -> timedelta:
    lo = token.get("lo")
    hi = token.get("hi")
    granularity = int(token.get("granularity_seconds") or 1)
    if not isinstance(lo, timedelta) or not isinstance(hi, timedelta):
        return timedelta()
    lo_units = int(lo.total_seconds()) // granularity
    hi_units = int(hi.total_seconds()) // granularity
    if hi_units <= lo_units:
        return lo
    seed = (
        f"cp-rand-v2|{str(chain_id or '').strip().lower()}|{cp}|{int(link_no)}|"
        f"{int(token_index)}|{token.get('raw')}"
    )
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    span = hi_units - lo_units + 1
    pick = lo_units + (int.from_bytes(digest[:8], "big") % span)
    return timedelta(seconds=pick * granularity)


def cp_sequence_interval_for_token(
    token: dict[str, Any],
    *,
    cp: str,
    link_no: int,
    token_index: int,
    chain_id: str | None = None,
) -> timedelta | None:
    canonical_cp = _canonical_cp_sequence(cp)
    if token.get("kind") == "fixed":
        duration = token.get("duration")
        return duration if isinstance(duration, timedelta) else None
    if token.get("kind") == "rand":
        return _cp_rand_duration_for_token(
            token,
            cp=canonical_cp,
            link_no=link_no,
            token_index=token_index,
            chain_id=chain_id,
        )
    return None


def parse_cp_sequence(cp: str):
    tokens = parse_cp_sequence_tokens(cp)
    if not tokens:
        return None
    durations = []
    for idx, token in enumerate(tokens):
        duration = cp_sequence_interval_for_token(
            token,
            cp=str(cp),
            link_no=idx + 1,
            token_index=idx,
        )
        if duration is None:
            return None
        durations.append(duration)
    return durations


def cp_sequence_interval_for_link(cp: str, link_no: int, chain_id: str | None = None):
    """Return the interval used to spawn ``link_no + 1`` from ``link_no``."""
    tokens = parse_cp_sequence_tokens(cp)
    if not tokens:
        return None
    try:
        idx = max(0, int(link_no) - 1) % len(tokens)
    except Exception:
        idx = 0
    return cp_sequence_interval_for_token(
        tokens[idx],
        cp=str(cp),
        link_no=int(link_no or 1),
        token_index=idx,
        chain_id=chain_id,
    )


__all__ = (
    "cp_sequence_interval_for_link",
    "cp_sequence_interval_for_token",
    "cp_sequence_parse_error",
    "parse_cp_duration",
    "parse_cp_sequence",
    "parse_cp_sequence_tokens",
)
