#!/usr/bin/env python3
"""Install or upgrade Nautical from a local release tree."""

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


_PLAN_LABELS = {
    "install": "Install",
    "upgrade": "Upgrade",
    "repair": "Repair",
    "reuse": "No changes",
}
_RESULT_LABELS = {
    "install": "Installed",
    "upgrade": "Upgraded",
    "repair": "Repaired",
    "reuse": "Already current",
}


def _render(payload: dict) -> None:
    operation = str(payload.get("operation") or "install")
    if payload.get("status") == "dry-run":
        print("Nautical install check: passed")
        print(f"Plan: {_PLAN_LABELS.get(operation, operation.replace('_', ' ').title())}")
    else:
        print("Nautical install: complete")
        print(f"Action: {_RESULT_LABELS.get(operation, operation.replace('_', ' ').title())}")
    print(f"Release: {payload['release_id']}")
    previous = str(payload.get("previous_release") or "")
    if previous and payload.get("status") == "dry-run":
        print(f"Current: {previous}")
    elif previous and previous != payload.get("active_release"):
        print(f"Previous: {previous}")
    print(f"Target: {payload['base']}")
    print(f"Hooks: {payload['hooks_dir']}")
    if payload.get("status") == "dry-run":
        print("Changes: none (dry run)")
    else:
        print(f"Launcher: {Path(payload['base']) / 'nautical'}")
        print("Validation: passed")
    if payload.get("migrated_legacy_core"):
        print(f"Legacy core backup: {payload.get('legacy_backup')}")
    for path in payload.get("migrated_configs") or []:
        print(f"Config preserved: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT), help="local release tree to install")
    parser.add_argument(
        "--taskdata",
        default=os.environ.get("TASKDATA") or "~/.task",
        help="Taskwarrior data directory (default: TASKDATA or ~/.task)",
    )
    parser.add_argument("--hooks-dir", default="", help="override the Taskwarrior hooks directory")
    parser.add_argument("--release-id", default="", help="optional stable release identifier")
    parser.add_argument("--dry-run", action="store_true", help="stage and validate without changing the installation")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()

    taskdata = Path(args.taskdata).expanduser()
    hooks_dir = Path(args.hooks_dir).expanduser() if args.hooks_dir else None
    try:
        payload = install_runtime.install_release(
            source=Path(args.source),
            taskdata=taskdata,
            hooks_dir=hooks_dir,
            release_id=args.release_id,
            dry_run=bool(args.dry_run),
        )
    except Exception as exc:
        payload = {
            "status": "error",
            "error": str(exc),
            "source": str(Path(args.source).expanduser()),
            "taskdata": str(taskdata),
        }
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        else:
            print(f"Nautical install failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        _render(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
