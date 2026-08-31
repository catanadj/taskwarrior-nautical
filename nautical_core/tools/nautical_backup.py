#!/usr/bin/env python3
"""Create a verified, local-only Nautical backup generation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

CORE_ROOT = Path(__file__).resolve().parents[1].parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from nautical_core.backup_service import (  # noqa: E402
    BackupExportError,
    BackupManifestError,
    backup_outbox_database,
    capture_taskwarrior_export,
    create_manifest,
    prune_backup_generations,
    publish_manifest,
)


def _outside_taskdata(taskdata: Path, destination: Path) -> None:
    source = taskdata.expanduser().resolve()
    target = destination.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise BackupExportError(f"backup destination already exists: {target}")
    if target.resolve() == source or source in target.resolve().parents:
        raise BackupExportError("backup destination must be outside Taskdata")


def create_backup(taskdata: Path, destination: Path, *, task_bin: str, timeout: float, keep: int = 2, prune: bool = False) -> dict[str, object]:
    """Capture portable task data and outbox into one atomically published directory."""
    taskdata = taskdata.expanduser().resolve()
    destination = destination.expanduser().absolute()
    _outside_taskdata(taskdata, destination)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        assert staging is not None
        task_export = capture_taskwarrior_export(taskdata, staging / "taskwarrior-export.json", task_bin=task_bin, timeout=timeout)
        outbox = backup_outbox_database(taskdata, staging / "lifecycle-outbox.db")
        manifest = create_manifest(
            staging,
            metadata={
                "taskdata": str(taskdata),
                "task_export_tasks": task_export.tasks,
                "outbox_quick_check": outbox.quick_check,
            },
        )
        publish_manifest(staging / "manifest.json", manifest)
        os.replace(staging, destination)
        staging = None
        retention: dict[str, object] = {"status": "not_requested", "keep": keep}
        if prune:
            result = prune_backup_generations(destination.parent, keep=keep)
            retention = {
                "status": "pruned",
                "keep": keep,
                "kept": list(result.kept),
                "removed": list(result.removed),
                "skipped": list(result.skipped),
            }
        return {
            "status": "created",
            "destination": str(destination),
            "manifest": str(destination / "manifest.json"),
            "task_export": {"tasks": task_export.tasks, "bytes": task_export.bytes, "sha256": task_export.sha256},
            "outbox": {"bytes": outbox.bytes, "quick_check": outbox.quick_check},
            "retention": retention,
        }
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # Keep the positional form for callers of the original developer tool,
    # while exposing named options for the offline runbook and shell scripts.
    parser.add_argument("destination_positional", nargs="?", help=argparse.SUPPRESS)
    parser.add_argument("--destination", dest="destination_option", help="new local backup directory")
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"))
    parser.add_argument("--task-bin", default="task")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--keep", type=int, default=2, help="verified generations to retain when --prune is used (default: 2)")
    parser.add_argument("--prune", action="store_true", help="prune older verified generations after a successful backup")
    parser.add_argument("--json", action="store_true", help="emit one JSON result")
    args = parser.parse_args()
    destination = args.destination_option or args.destination_positional
    if not destination:
        print(json.dumps({"status": "error", "error": "--destination is required"}, ensure_ascii=False))
        return 2
    try:
        result = create_backup(Path(args.taskdata), Path(destination), task_bin=args.task_bin, timeout=args.timeout, keep=args.keep, prune=args.prune)
    except (BackupExportError, BackupManifestError, OSError, ValueError) as exc:
        result = {"status": "error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
