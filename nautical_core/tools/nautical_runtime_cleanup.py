#!/usr/bin/env python3
"""Safely inspect or clean inactive Nautical runtime releases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core import install_runtime  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA") or "~/.task")
    parser.add_argument("--keep", type=int, default=1, help="inactive releases to retain (default: 1)")
    parser.add_argument("--stale-after-seconds", type=float, default=86400.0)
    parser.add_argument("--apply", action="store_true", help="delete the planned artifacts")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = install_runtime.cleanup_runtime(
            Path(args.taskdata).expanduser(),
            keep_releases=max(0, args.keep),
            stale_after_seconds=max(0.0, args.stale_after_seconds),
            apply=args.apply,
        )
    except Exception as exc:
        result = {"status": "error", "error": str(exc)}
    if args.json:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"Nautical runtime cleanup: {result.get('status', 'error')}")
        print(f"Active release: {result.get('active_release') or 'none'}")
        print(f"Kept releases: {len(result.get('kept_releases') or [])}")
        print(f"Releases to remove: {len(result.get('remove_releases') or [])}")
        print(f"Abandoned paths to remove: {len(result.get('remove_abandoned') or [])}")
        if result.get("errors"):
            print("Errors:")
            for error in result["errors"]:
                print(f"  {error}")
    return 2 if result.get("status") == "error" or result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
