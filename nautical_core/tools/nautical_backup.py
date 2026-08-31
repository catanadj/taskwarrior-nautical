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
    build_backup_metadata,
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


def _copy_resources(resources: list[tuple[str, Path]], taskdata: Path, staging: Path) -> None:
    names: set[str] = set()
    taskdata_resolved = taskdata.resolve()
    runtime_resolved = taskdata_resolved / ".nautical-runtime"
    for name, source in resources:
        if not name or name in {".", ".."} or "/" in name or "\\" in name or ".." in Path(name).parts:
            raise BackupExportError(f"resource name is unsafe: {name!r}")
        if name in names:
            raise BackupExportError(f"duplicate resource name: {name}")
        names.add(name)
        source = source.expanduser()
        if source.is_symlink() or not source.is_file():
            raise BackupExportError(f"resource is not a regular file: {source}")
        resolved = source.resolve()
        if resolved == taskdata_resolved or taskdata_resolved in resolved.parents:
            raise BackupExportError("resource source must be outside Taskdata")
        if resolved == runtime_resolved or runtime_resolved in resolved.parents:
            raise BackupExportError("resource source must be outside the managed runtime")
        target = staging / "resources" / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            shutil.copy2(resolved, target)
        except OSError as exc:
            raise BackupExportError(f"could not copy resource {name}: {exc}") from exc


def create_backup(taskdata: Path, destination: Path, *, task_bin: str, timeout: float, keep: int = 2, prune: bool = False, metadata: dict[str, object] | None = None, resources: list[tuple[str, Path]] | None = None) -> dict[str, object]:
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
        _copy_resources(resources or [], taskdata, staging)
        manifest = create_manifest(
            staging,
            metadata={
                **(metadata or {}),
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
    parser.add_argument("--active-release", default=None)
    parser.add_argument("--runtime-digest", default=None)
    parser.add_argument("--taskwarrior-version", default=None)
    parser.add_argument("--python-version", default=None)
    parser.add_argument("--timezone", default=None)
    parser.add_argument("--timezone-data-identity", default=None)
    parser.add_argument("--include", action="append", default=[], metavar="NAME=PATH", help="include an explicit regular resource file (repeatable)")
    parser.add_argument("--json", action="store_true", help="emit one JSON result")
    args = parser.parse_args()
    destination = args.destination_option or args.destination_positional
    if not destination:
        print(json.dumps({"status": "error", "error": "--destination is required"}, ensure_ascii=False))
        return 2
    try:
        provenance = build_backup_metadata(
            active_release=args.active_release,
            runtime_digest=args.runtime_digest,
            taskwarrior_version=args.taskwarrior_version,
            python_version=args.python_version,
            timezone=args.timezone,
            timezone_data_identity=args.timezone_data_identity,
        )
        includes: list[tuple[str, Path]] = []
        for value in args.include:
            if "=" not in value:
                raise BackupExportError(f"--include requires NAME=PATH: {value!r}")
            name, path = value.split("=", 1)
            includes.append((name, Path(path)))
        result = create_backup(Path(args.taskdata), Path(destination), task_bin=args.task_bin, timeout=args.timeout, keep=args.keep, prune=args.prune, metadata=provenance, resources=includes)
    except (BackupExportError, BackupManifestError, OSError, ValueError) as exc:
        result = {"status": "error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
