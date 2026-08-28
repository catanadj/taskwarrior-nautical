#!/usr/bin/env python3
"""Safely inspect or clean inactive Nautical runtime releases."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core import install_runtime  # noqa: E402
from nautical_core.operator_presentation import render_json_document  # noqa: E402
from nautical_core.operator_presentation import key_value_lines  # noqa: E402


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
        print(render_json_document(result))
    else:
        summary = {
            "Nautical runtime cleanup": result.get("status", "error"),
            "Active release": result.get("active_release") or "none",
            "Kept releases": len(result.get("kept_releases") or []),
            "Releases to remove": len(result.get("remove_releases") or []),
            "Abandoned paths to remove": len(result.get("remove_abandoned") or []),
        }
        print("\n".join(key_value_lines(summary)))
        if result.get("errors"):
            print("Errors:")
            for error in result["errors"]:
                print(f"  {error}")
    return 2 if result.get("status") == "error" or result.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
