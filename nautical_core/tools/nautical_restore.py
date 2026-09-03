#!/usr/bin/env python3
"""Validate or stage a local Nautical backup generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CORE_ROOT = Path(__file__).resolve().parents[1].parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from nautical_core.restore_service import restore_backup  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="verified backup generation directory")
    parser.add_argument("--target", default="", help="new or empty disposable restore directory")
    parser.add_argument("--apply", action="store_true", help="create the target after validation")
    parser.add_argument("--json", action="store_true", help="emit one JSON result (the default output)")
    args = parser.parse_args()
    try:
        report = restore_backup(
            Path(args.source),
            Path(args.target) if args.target else None,
            apply=bool(args.apply),
        )
        payload = report.to_dict()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        return 0 if report.status in {"validated", "restored"} else 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
