"""Validate and stage a local Nautical backup without replacing live state."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
from typing import Any

from .backup_service import BackupManifestError, verify_manifest


class BackupRestoreError(RuntimeError):
    """Raised when a backup cannot be validated or safely restored."""


@dataclass(frozen=True, slots=True)
class RestoreReport:
    status: str
    source: str
    target: str | None = None
    tasks: int = 0
    checked: int = 0
    quick_check: str = ""
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source": self.source,
            "target": self.target,
            "tasks": self.tasks,
            "checked": self.checked,
            "quick_check": self.quick_check,
            "errors": list(self.errors),
        }


def _load_manifest(source: Path) -> dict[str, Any]:
    try:
        value = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BackupRestoreError(f"backup manifest is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise BackupRestoreError("backup manifest must be an object")
    return value


def _validate_export(source: Path) -> int:
    path = source / "taskwarrior-export.json"
    try:
        if path.is_symlink() or not path.is_file():
            raise BackupRestoreError("Taskwarrior export is missing or unsafe")
        value = json.loads(path.read_text(encoding="utf-8"))
    except BackupRestoreError:
        raise
    except (OSError, ValueError) as exc:
        raise BackupRestoreError(f"Taskwarrior export is unreadable: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise BackupRestoreError("Taskwarrior export must be a JSON array of objects")
    return len(value)


def _validate_outbox(source: Path) -> str:
    path = source / "lifecycle-outbox.db"
    if path.is_symlink() or not path.is_file():
        raise BackupRestoreError("lifecycle outbox backup is missing or unsafe")
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        result = str(connection.execute("PRAGMA quick_check").fetchone()[0])
    except (OSError, sqlite3.Error) as exc:
        raise BackupRestoreError(f"lifecycle outbox quick_check failed: {exc}") from exc
    finally:
        if connection is not None:
            connection.close()
    if result.lower() != "ok":
        raise BackupRestoreError(f"lifecycle outbox quick_check failed: {result}")
    return result


def validate_backup(source: Path) -> RestoreReport:
    """Validate a backup generation without creating or changing any files."""
    source = Path(source).expanduser().resolve()
    try:
        if not source.is_dir() or source.is_symlink():
            raise BackupRestoreError(f"backup source is not a directory: {source}")
        manifest = _load_manifest(source)
        verification = verify_manifest(source, manifest)
        if verification.status != "verified":
            raise BackupRestoreError("; ".join(verification.errors) or "backup checksum verification failed")
        tasks = _validate_export(source)
        quick_check = _validate_outbox(source)
        return RestoreReport("validated", str(source), tasks=tasks, checked=verification.checked, quick_check=quick_check)
    except (BackupRestoreError, BackupManifestError) as exc:
        return RestoreReport("rejected", str(source), errors=(str(exc),))


def restore_backup(source: Path, target: Path | None = None, *, apply: bool = False) -> RestoreReport:
    """Validate, then optionally stage a backup into a new disposable target.

    Without ``apply`` this function is strictly inspect-only.  An existing
    target must be an empty directory; live Taskdata is never overwritten.
    """
    report = validate_backup(source)
    if report.status != "validated" or target is None or not apply:
        return RestoreReport(report.status if target is None or not apply else "rejected", report.source, str(target) if target else None, report.tasks, report.checked, report.quick_check, report.errors)
    destination = Path(target).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        if not destination.is_dir() or any(destination.iterdir()):
            return RestoreReport("rejected", report.source, str(destination), errors=("restore target must be a new or empty directory",))
    parent = destination.parent
    parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{destination.name}.restore-", dir=parent))
    displaced: Path | None = None
    try:
        assert temporary is not None
        source_path = Path(report.source)
        shutil.copy2(source_path / "taskwarrior-export.json", temporary / "taskwarrior-export.json")
        state = temporary / ".nautical-state"
        state.mkdir(mode=0o700)
        shutil.copy2(source_path / "lifecycle-outbox.db", state / ".nautical_lifecycle_outbox.db")
        resources = source_path / "resources"
        if resources.exists():
            if resources.is_symlink() or not resources.is_dir():
                raise BackupRestoreError("backup resources directory is missing or unsafe")
            for resource in resources.iterdir():
                if resource.is_symlink() or not resource.is_file():
                    raise BackupRestoreError(f"backup resource is missing or unsafe: {resource.name}")
            shutil.copytree(resources, temporary / "resources")
        shutil.copy2(source_path / "manifest.json", temporary / "manifest.json")
        if destination.exists():
            displaced = Path(tempfile.mkdtemp(prefix=f".{destination.name}.previous-", dir=parent))
            displaced.rmdir()
            os.replace(destination, displaced)
        os.replace(temporary, destination)
        temporary = None
        if displaced is not None:
            displaced.rmdir()
            displaced = None
        return RestoreReport("restored", report.source, str(destination), report.tasks, report.checked, report.quick_check)
    except (BackupRestoreError, OSError, shutil.Error) as exc:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        if displaced is not None and not destination.exists():
            os.replace(displaced, destination)
        return RestoreReport("rejected", report.source, str(destination), errors=(f"restore could not be published: {exc}",))


__all__ = ("BackupRestoreError", "RestoreReport", "restore_backup", "validate_backup")
