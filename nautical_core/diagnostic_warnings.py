from __future__ import annotations

import os
import sys
import time
from datetime import date


def warn_once_per_day(key: str, message: str, *, cache_dir: str, require_diag: bool) -> None:
    """Persist a tiny sentinel so we do not spam hook output."""
    if require_diag and os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        stamp_path = os.path.join(cache_dir, f".diag_{key}.stamp")

        today = date.today().isoformat()
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, "r", encoding="utf-8") as f:
                    if f.read().strip() == today:
                        return
            except Exception:
                pass

        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(today)
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(message, file=sys.stderr)
            except Exception:
                pass
    except Exception:
        pass


def warn_rate_limited_any(key: str, message: str, *, cache_dir: str, min_interval_s: float = 3600.0) -> None:
    """Emit a diagnostic warning at most once per min_interval_s (always on)."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        stamp_path = os.path.join(cache_dir, f".diag_{key}.stamp")
        now = time.time()
        last = None
        if os.path.exists(stamp_path):
            try:
                with open(stamp_path, "r", encoding="utf-8") as f:
                    raw = f.read().strip()
                last = float(raw) if raw else None
            except Exception:
                last = None
        if last is not None and (now - last) < float(min_interval_s or 0.0):
            return
        with open(stamp_path, "w", encoding="utf-8") as f:
            f.write(str(now))
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(message, file=sys.stderr)
            except Exception:
                pass
    except Exception:
        pass


def warn_missing_toml_parser(
    config_path: str,
    *,
    warn_once_per_day,
    warn_once_per_day_any,
) -> None:
    pyver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    warn_once_per_day_any(
        "missing_toml_parser_min",
        "[nautical] Config present but TOML parser unavailable; using defaults.",
    )
    msg = (
        "[nautical] Config detected but not loaded: TOML parser unavailable.\n"
        f"          Path: {config_path}\n"
        f"          Python: {pyver}\n"
        "          Fix: upgrade to Python 3.11+ (tomllib built-in) OR install tomli:\n"
        "               python3 -m pip install --user tomli\n"
        "          Tip: set NAUTICAL_DIAG=1 to print config search paths.\n"
    )
    warn_once_per_day("missing_toml_parser", msg)


def warn_toml_parse_error(
    config_path: str,
    err: Exception,
    *,
    warn_once_per_day,
    warn_once_per_day_any,
) -> None:
    warn_once_per_day_any(
        "toml_parse_error_min",
        "[nautical] Config parse failed; using defaults.",
    )
    msg = (
        "[nautical] Config file found but could not be parsed; defaults will be used.\n"
        f"          Path: {config_path}\n"
        f"          Error: {err}\n"
        "          Fix: validate TOML syntax, or run with NAUTICAL_DIAG=1 for more context.\n"
    )
    warn_once_per_day("toml_parse_error", msg)
