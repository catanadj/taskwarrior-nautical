from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


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


def fmt_dt_local(dt_utc: datetime, local_tz) -> str:
    """Format UTC datetime as local time string."""
    d = to_local(dt_utc, local_tz)
    return d.strftime("%a %Y-%m-%d %H:%M %Z")


def fmt_isoz(dt_utc: datetime) -> str:
    """Format UTC datetime as ISO 8601 with Zulu time."""
    return ensure_utc(dt_utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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
    if not local_tz:
        return naive.replace(tzinfo=timezone.utc)

    candidates = []
    for fold in (0, 1):
        aware = naive.replace(tzinfo=local_tz, fold=fold)
        back = aware.astimezone(timezone.utc).astimezone(local_tz)
        if back.replace(tzinfo=None) == naive:
            candidates.append(aware)
    if candidates:
        # Ambiguous time: choose the earlier UTC instant for determinism.
        best = min(candidates, key=lambda dt: dt.astimezone(timezone.utc))
        return best.astimezone(timezone.utc)

    # Non-existent time (spring forward): shift forward by 1 hour, then to next valid minute.
    cand = naive + timedelta(hours=1)
    for _ in range(180):
        for fold in (0, 1):
            aware = cand.replace(tzinfo=local_tz, fold=fold)
            back = aware.astimezone(timezone.utc).astimezone(local_tz)
            if back.replace(tzinfo=None) == cand:
                return aware.astimezone(timezone.utc)
        cand += timedelta(minutes=1)
    return naive.replace(tzinfo=local_tz, fold=0).astimezone(timezone.utc)
