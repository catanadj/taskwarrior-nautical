from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


def compare_datetimes(left: datetime, right: datetime) -> int:
    """Compare datetimes by instant when aware, otherwise by wall time."""
    if not isinstance(left, datetime) or not isinstance(right, datetime):
        raise TypeError("Datetime comparison requires datetime values.")
    left_aware = left.tzinfo is not None and left.utcoffset() is not None
    right_aware = right.tzinfo is not None and right.utcoffset() is not None
    if left_aware != right_aware:
        raise ValueError("Cannot compare naive and aware datetime values.")
    if left_aware:
        left = left.astimezone(timezone.utc)
        right = right.astimezone(timezone.utc)
    return (left > right) - (left < right)


def ensure_utc(dt_utc: datetime) -> datetime:
    """Return a timezone-aware UTC datetime."""
    if dt_utc.tzinfo is None:
        return dt_utc.replace(tzinfo=timezone.utc)
    return dt_utc.astimezone(timezone.utc)


def now_utc() -> datetime:
    """Get current UTC time without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def to_local(dt_utc: datetime, local_tz) -> datetime:
    """Convert UTC datetime to local timezone."""
    dt_utc = ensure_utc(dt_utc)
    return dt_utc.astimezone(local_tz) if local_tz else dt_utc


def utc_to_local_naive(dt_utc: datetime, local_tz) -> datetime:
    """Convert a UTC timestamp to a naive local wall-clock datetime."""
    if not isinstance(dt_utc, datetime):
        raise TypeError("UTC datetime must be a datetime.")
    return to_local(dt_utc, local_tz).replace(tzinfo=None)


def fmt_dt_local(dt_utc: datetime, local_tz) -> str:
    """Format UTC datetime as local time string."""
    d = to_local(dt_utc, local_tz)
    return d.strftime("%a %Y-%m-%d %H:%M %Z")


def fmt_isoz(dt_utc: datetime) -> str:
    """Format UTC datetime as ISO 8601 with Zulu time."""
    return ensure_utc(dt_utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def local_naive_to_utc(naive: datetime, local_tz) -> datetime:
    """Resolve a local wall time to UTC across arbitrary DST transitions."""
    if not isinstance(naive, datetime):
        raise TypeError("Local datetime must be a datetime.")
    if naive.tzinfo is not None:
        raise ValueError("Local datetime must be naive.")
    if local_tz is None:
        return naive.replace(tzinfo=timezone.utc)

    candidates = []
    for fold in (0, 1):
        aware = naive.replace(tzinfo=local_tz, fold=fold)
        back = aware.astimezone(timezone.utc).astimezone(local_tz)
        wall = back.replace(tzinfo=None)
        if wall == naive:
            candidates.append(aware)
    if candidates:
        return min(candidates, key=lambda value: value.astimezone(timezone.utc)).astimezone(timezone.utc)

    # For a nonexistent wall time, choose the first round-tripped wall time after it.
    after = []
    for fold in (0, 1):
        aware = naive.replace(tzinfo=local_tz, fold=fold)
        back = aware.astimezone(timezone.utc).astimezone(local_tz)
        wall = back.replace(tzinfo=None)
        if wall > naive:
            after.append((wall, aware))
    if after:
        return min(after, key=lambda item: item[0])[1].astimezone(timezone.utc)

    # Defensive fallback for unusual timezone implementations.
    probe = naive + timedelta(minutes=1)
    for _ in range(2880):
        for fold in (0, 1):
            aware = probe.replace(tzinfo=local_tz, fold=fold)
            back = aware.astimezone(timezone.utc).astimezone(local_tz)
            if back.replace(tzinfo=None) == probe:
                return aware.astimezone(timezone.utc)
        probe += timedelta(minutes=1)
    raise ValueError("Local datetime could not be resolved in the configured timezone.")


def parse_dt_any(s: str, date_formats) -> datetime | None:
    """Parse datetime from string using multiple formats."""
    if not s:
        return None
    s = str(s)
    for fmt in date_formats:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            pass
    try:
        d = datetime.strptime(s[:10], "%Y-%m-%d")
        return d.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def build_local_datetime(d: date, hhmm, local_tz) -> datetime:
    """Build a UTC datetime from local wall-clock date+time with DST handling."""
    hh, mm = hhmm
    naive = datetime(d.year, d.month, d.day, hh, mm, 0)
    return local_naive_to_utc(naive, local_tz)
