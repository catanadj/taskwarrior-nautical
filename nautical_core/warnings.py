from __future__ import annotations

import os
import sys
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
        if not require_diag or os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(message, file=sys.stderr)
            except Exception:
                pass
    except Exception:
        pass
