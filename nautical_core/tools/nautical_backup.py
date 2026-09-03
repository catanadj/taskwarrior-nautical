#!/usr/bin/env python3
"""Create a verified, local-only Nautical backup generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import tempfile
import time
import tomllib
from importlib import metadata as importlib_metadata

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
    runtime_dependency_versions,
)


def _outside_taskdata(taskdata: Path, destination: Path) -> None:
    source = taskdata.expanduser().resolve()
    target = destination.expanduser().absolute()
    if target.exists() or target.is_symlink():
        raise BackupExportError(f"backup destination already exists: {target}")
    if target.resolve() == source or source in target.resolve().parents:
        raise BackupExportError("backup destination must be outside Taskdata")


def _copy_resources(
    resources: list[tuple[str, Path]],
    taskdata: Path,
    staging: Path,
    *,
    allow_taskdata_owned: bool = False,
) -> None:
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
        if not allow_taskdata_owned and (resolved == taskdata_resolved or taskdata_resolved in resolved.parents):
            raise BackupExportError("resource source must be outside Taskdata")
        if resolved == runtime_resolved or runtime_resolved in resolved.parents:
            raise BackupExportError("resource source must be outside the managed runtime")
        target = staging / "resources" / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            shutil.copy2(resolved, target)
        except OSError as exc:
            raise BackupExportError(f"could not copy resource {name}: {exc}") from exc


def _active_config_path(taskdata: Path) -> Path | None:
    explicit = os.environ.get("NAUTICAL_CONFIG", "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    taskrc = os.environ.get("TASKRC", "").strip()
    if taskrc:
        directory = Path(taskrc).expanduser().resolve().parent
        candidates.extend((directory / "config-nautical.toml", directory / "nautical.toml"))
    candidates.extend((taskdata / "config-nautical.toml", taskdata / "nautical.toml"))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file() and not resolved.is_symlink():
            return resolved
    return None


def _default_resources(taskdata: Path, export_path: Path | None = None) -> list[tuple[str, Path]]:
    """Return Nautical-owned files required to reproduce the active install."""
    resources: list[tuple[str, Path]] = []
    config = _active_config_path(taskdata)
    if config is not None:
        resources.append((config.name, config))
        try:
            config_data = tomllib.loads(config.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise BackupExportError(f"active Nautical config cannot be read: {config}: {exc}") from exc
        directories: dict[str, Path] = {}
        for key, prefix in (("anchor_file_dir", "anchor"), ("omit_file_dir", "omit")):
            directory_value = config_data.get(key)
            if not isinstance(directory_value, str) or not directory_value.strip():
                continue
            directory = Path(directory_value).expanduser()
            if not directory.is_absolute():
                directory = config.parent / directory
            if not directory.is_dir() or directory.is_symlink():
                raise BackupExportError(f"configured {key} is unavailable: {directory}")
            directories[prefix] = directory

        def add_patterns(prefix: str, patterns: object) -> None:
            directory = directories.get(prefix)
            if directory is None:
                return
            values = patterns if isinstance(patterns, list) else [patterns]
            for pattern in values:
                if not isinstance(pattern, str) or pattern.strip().lower() in {"", "null", "none"}:
                    continue
                matches = sorted(directory.glob(pattern))
                if not matches:
                    raise BackupExportError(f"configured resource pattern has no match: {pattern}")
                for match in matches:
                    if match.is_file() and not match.is_symlink():
                        candidate = (f"{prefix}-{match.name}", match)
                        if candidate not in resources:
                            resources.append(candidate)

        for table in (config_data,):
            def collect(value: object) -> None:
                if not isinstance(value, dict):
                    return
                for field_name, field_value in value.items():
                    if field_name in {"anchor_file", "omit_file"}:
                        add_patterns("anchor" if field_name == "anchor_file" else "omit", field_value)
                    collect(field_value)

            collect(table)
        if export_path is not None:
            try:
                exported_rows = json.loads(export_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BackupExportError(f"Taskwarrior export cannot be inspected for resources: {exc}") from exc
            if isinstance(exported_rows, list):
                for row in exported_rows:
                    if isinstance(row, dict):
                        add_patterns("anchor", row.get("anchor_file"))
                        add_patterns("omit", row.get("omit_file"))
    uda = taskdata / "uda-nautical.conf"
    if uda.is_file() and not uda.is_symlink():
        resources.append((uda.name, uda))
    taskrc_value = os.environ.get("TASKRC", "").strip()
    taskrc = Path(taskrc_value).expanduser() if taskrc_value else Path("~/.taskrc").expanduser()
    if taskrc.is_file() and not taskrc.is_symlink():
        resources.append(("taskrc", taskrc))
    return resources


def _runtime_provenance(taskdata: Path) -> dict[str, str]:
    """Read managed-release identity without importing the live runtime."""
    result: dict[str, str] = {"python_version": sys.version.split()[0]}
    current = taskdata / ".nautical-runtime" / "current"
    manifest = current / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        return result
    try:
        value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return result
    if not isinstance(value, dict):
        return result
    for source, target in (("release_id", "active_release"), ("content_sha256", "runtime_digest")):
        item = value.get(source)
        if isinstance(item, str) and item.strip():
            result[target] = item.strip()
    return result


def _taskwarrior_version(task_bin: str) -> str | None:
    try:
        completed = subprocess.run(
            [task_bin, "--version"], capture_output=True, text=True, timeout=5.0, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    for token in text.replace("/", " ").split():
        if token and token[0].isdigit() and token.count(".") >= 1:
            return token.rstrip(")")
    return None


def _timezone_provenance(taskdata: Path) -> dict[str, str]:
    config = _active_config_path(taskdata)
    if config is None:
        return {}
    try:
        values = tomllib.loads(config.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    timezone = values.get("tz")
    if not isinstance(timezone, str) or not timezone.strip():
        return {}
    result = {"timezone": timezone.strip()}
    try:
        from importlib import metadata as package_metadata
        result["timezone_data_identity"] = f"tzdata:{package_metadata.version('tzdata')}"
        return result
    except package_metadata.PackageNotFoundError:
        pass
    try:
        from zoneinfo import TZPATH
        for directory in TZPATH:
            candidate = Path(directory) / timezone
            if candidate.is_file() and not candidate.is_symlink():
                result["timezone_data_identity"] = f"system:{hashlib.sha256(candidate.read_bytes()).hexdigest()}"
                break
    except OSError:
        pass
    return result


def _copy_verified_tree(source: Path, destination: Path, label: str) -> None:
    """Copy a managed tree while rejecting links and non-regular files."""
    if source.is_symlink() or not source.is_dir():
        raise BackupExportError(f"{label} is unavailable: {source}")
    for path in source.rglob("*"):
        relative = path.relative_to(source)
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise BackupExportError(f"{label} contains an unsafe entry: {relative}")
        target = destination / relative
        if path.is_dir():
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
        else:
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            try:
                shutil.copy2(path, target)
            except OSError as exc:
                raise BackupExportError(f"could not copy {label} entry {relative}: {exc}") from exc


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
        defaults = _default_resources(taskdata, staging / "taskwarrior-export.json")
        selected = [*defaults, *(resources or [])]
        names = [name for name, _source in selected]
        if len(names) != len(set(names)):
            raise BackupExportError("backup resources contain duplicate names")
        default_count = len(defaults)
        _copy_resources(selected[:default_count], taskdata, staging, allow_taskdata_owned=True)
        _copy_resources(selected[default_count:], taskdata, staging)
        current = taskdata / ".nautical-runtime" / "current"
        if current.exists() or current.is_symlink():
            release = current.resolve()
            _copy_verified_tree(release, staging / "runtime" / "releases" / release.name, "managed runtime")
        hooks = taskdata / "hooks"
        if hooks.exists() or hooks.is_symlink():
            _copy_verified_tree(hooks, staging / "hooks", "hook layout")
        manifest = create_manifest(
            staging,
            metadata={
                **(metadata or {}),
                "created_at": time.time(),
                "restore_tool_schema": 1,
                "taskdata": str(taskdata),
                "task_export_tasks": task_export.tasks,
                "outbox_quick_check": outbox.quick_check,
                "python_packages": runtime_dependency_versions(),
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
        includes: list[tuple[str, Path]] = []
        for value in args.include:
            if "=" not in value:
                raise BackupExportError(f"--include requires NAME=PATH: {value!r}")
            name, path = value.split("=", 1)
            includes.append((name, Path(path)))
        taskdata = Path(args.taskdata)
        auto = _runtime_provenance(taskdata)
        auto.update(_timezone_provenance(taskdata))
        version = _taskwarrior_version(args.task_bin)
        if version:
            auto["taskwarrior_version"] = version
        provenance = build_backup_metadata(
            active_release=args.active_release or auto.get("active_release"),
            runtime_digest=args.runtime_digest or auto.get("runtime_digest"),
            taskwarrior_version=args.taskwarrior_version or auto.get("taskwarrior_version"),
            python_version=args.python_version or auto.get("python_version"),
            timezone=args.timezone or auto.get("timezone"),
            timezone_data_identity=args.timezone_data_identity or auto.get("timezone_data_identity"),
        )
        result = create_backup(taskdata, Path(destination), task_bin=args.task_bin, timeout=args.timeout, keep=args.keep, prune=args.prune, metadata=provenance, resources=includes)
    except (BackupExportError, BackupManifestError, OSError, ValueError) as exc:
        result = {"status": "error", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
