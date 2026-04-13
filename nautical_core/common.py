from __future__ import annotations

import math
import os
import re
import sys


_INT_FLOATISH_RE = re.compile(r"^[+-]?\d+(?:\.0+)?$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def short_uuid(u: str | None) -> str:
    """Taskwarrior-style short uuid (first 8 hex)."""
    if not u or not isinstance(u, str):
        return ""
    s = u.strip().lower()
    if not s:
        return ""
    return s.split("-")[0] if "-" in s else s[:8]


def should_stamp_chain_id(task: dict) -> bool:
    """We stamp a chainID when task becomes/starts a nautical chain."""
    if not isinstance(task, dict):
        return False
    has_anchor = bool((task.get("anchor") or "").strip())
    has_anchor_file = bool((task.get("anchor_file") or "").strip())
    has_cp = bool((task.get("cp") or "").strip())
    already = bool((task.get("chainID") or "").strip())
    return (has_anchor or has_anchor_file or has_cp) and not already


def sanitize_text(v: str, max_len: int = 1024):
    """Remove control chars and clamp length for UDA safety."""
    if not isinstance(v, str):
        return v
    s = _CONTROL_CHARS_RE.sub("", v)
    if max_len > 0 and len(s) > max_len:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(f"[nautical] UDA value truncated from {len(s)} to {max_len} chars", file=sys.stderr)
            except Exception:
                pass
        s = s[:max_len]
    return s


def sanitize_task_strings(task: dict, max_len: int = 1024) -> None:
    """In-place sanitize of string values in a task payload."""
    if not isinstance(task, dict):
        return
    for k, v in list(task.items()):
        if isinstance(v, str):
            cleaned = sanitize_text(v, max_len=max_len)
            if cleaned != v and os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    print(f"[nautical] UDA field truncated: {k}", file=sys.stderr)
                except Exception:
                    pass
            task[k] = cleaned


def split_csv_tokens(spec: str) -> list[str]:
    return [t.strip() for t in str(spec or "").split(",") if t.strip()]


def split_csv_lower(spec: str) -> list[str]:
    return [t.lower() for t in split_csv_tokens(spec)]


def coerce_int(v, default=None):
    """Safely convert value to int, handling floats and strings."""
    try:
        if v is None:
            return default
        if isinstance(v, bool):
            return default
        if isinstance(v, int):
            return v if abs(v) <= (2**63 - 1) else default
        if isinstance(v, float):
            if not math.isfinite(v):
                return default
            iv = int(round(v))
            return iv if abs(iv) <= (2**63 - 1) else default
        s = str(v).strip()
        if _INT_FLOATISH_RE.fullmatch(s):
            iv = int(float(s))
            return iv if abs(iv) <= (2**63 - 1) else default
        iv = int(s)
        return iv if abs(iv) <= (2**63 - 1) else default
    except Exception:
        return default
