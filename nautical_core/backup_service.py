"""Verified local backup inventory and manifest primitives.

This module deliberately does not copy or restore Taskwarrior data yet.  It
provides the integrity boundary used by those workflows: a versioned manifest
of regular local files, with path and digest validation and atomic publication.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from .task_command import failure_message, run_task_command


BACKUP_MANIFEST_SCHEMA = "nautical.backup"
BACKUP_MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PATH_BYTES = 4096
MAX_METADATA_KEYS = 128


def build_backup_metadata(
    *,
    active_release: str | None = None,
    runtime_digest: str | None = None,
    taskwarrior_version: str | None = None,
    python_version: str | None = None,
    timezone: str | None = None,
    timezone_data_identity: str | None = None,
) -> dict[str, Any]:
    """Build explicit, JSON-safe provenance metadata for a backup.

    Values are caller-supplied deliberately: creating a backup must not
    silently inspect or depend on the live installation's environment.
    """
    values = (
        ("active_release", active_release),
        ("runtime_digest", runtime_digest),
        ("taskwarrior_version", taskwarrior_version),
        ("python_version", python_version),
        ("timezone", timezone),
        ("timezone_data_identity", timezone_data_identity),
    )
    result: dict[str, Any] = {"metadata_schema": 1}
    for key, value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if normalized:
            result[key] = normalized
    json.dumps(result, ensure_ascii=False)
    return result


class BackupManifestError(ValueError):
    """Raised when an inventory or manifest violates the backup contract."""


class BackupExportError(RuntimeError):
    """Raised when a hooks-off Taskwarrior export cannot be captured safely."""


@dataclass(frozen=True, slots=True)
class BackupArtifact:
    """One regular file recorded relative to a backup root."""

    path: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        path = _safe_relative_path(self.path)
        if isinstance(self.size, bool) or self.size < 0:
            raise BackupManifestError(f"invalid artifact size: {path}")
        digest = str(self.sha256).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise BackupManifestError(f"invalid artifact digest: {path}")
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "sha256", digest)

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "bytes": self.size, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class BackupVerification:
    """Stable machine-readable result from manifest verification."""

    status: str
    root: str
    checked: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "root": self.root,
            "checked": self.checked,
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class BackupExport:
    """Metadata for one atomically captured portable Taskwarrior export."""

    status: str
    destination: str
    tasks: int
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class SQLiteBackup:
    """Metadata for one verified SQLite online backup."""

    status: str
    source: str
    destination: str
    bytes: int
    quick_check: str


@dataclass(frozen=True, slots=True)
class BackupRetention:
    """Result of pruning verified backup generations."""

    kept: tuple[str, ...]
    removed: tuple[str, ...]
    skipped: tuple[str, ...]


def _safe_relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > MAX_PATH_BYTES:
        raise BackupManifestError("manifest contains an invalid relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value.startswith("./"):
        raise BackupManifestError(f"manifest contains an unsafe path: {value!r}")
    normalized = path.as_posix()
    if normalized != value or normalized in ("", "."):
        raise BackupManifestError(f"manifest contains a non-canonical path: {value!r}")
    return normalized


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise BackupManifestError(f"cannot read backup artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact(root: Path, relative: str) -> BackupArtifact:
    relative = _safe_relative_path(relative)
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BackupManifestError(f"backup artifact is unavailable: {relative}") from exc
    if root not in resolved.parents or path.is_symlink() or not path.is_file():
        raise BackupManifestError(f"backup artifact is not a regular local file: {relative}")
    size = path.stat().st_size
    return BackupArtifact(relative, size, _digest(path))


def inventory(root: Path, files: Iterable[str] | None = None) -> tuple[BackupArtifact, ...]:
    """Hash regular files under *root*, rejecting symlink escapes."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise BackupManifestError(f"backup root is not a directory: {root}")
    if files is None:
        names: list[str] = []
        for path in root.rglob("*"):
            if path.is_symlink():
                raise BackupManifestError(f"backup tree contains a symlink: {path.relative_to(root)}")
            if path.is_file():
                names.append(path.relative_to(root).as_posix())
    else:
        names = list(files)
    if len(set(names)) != len(names):
        raise BackupManifestError("backup manifest contains duplicate artifact paths")
    return tuple(_artifact(root, name) for name in sorted(names))


def create_manifest(
    root: Path,
    *,
    files: Iterable[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a JSON-safe versioned manifest for local artifacts."""
    records = inventory(root, files)
    metadata_dict = dict(metadata or {})
    if len(metadata_dict) > MAX_METADATA_KEYS:
        raise BackupManifestError("backup metadata contains too many keys")
    try:
        json.dumps(metadata_dict, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise BackupManifestError(f"backup metadata is not JSON-serializable: {exc}") from exc
    return {
        "schema": BACKUP_MANIFEST_SCHEMA,
        "version": BACKUP_MANIFEST_VERSION,
        "metadata": metadata_dict,
        "files": [record.to_dict() for record in records],
    }


def publish_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    """Atomically publish a validated manifest without partial output."""
    encoded = json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if len(encoded.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise BackupManifestError("backup manifest exceeds the size limit")
    validate_manifest(manifest)
    destination = Path(path).expanduser()
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise BackupManifestError(f"could not publish backup manifest: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass


def validate_manifest(manifest: Mapping[str, Any]) -> tuple[BackupArtifact, ...]:
    """Validate structure and security properties without touching artifacts."""
    try:
        encoded = json.dumps(dict(manifest), ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise BackupManifestError(f"backup manifest is not JSON-serializable: {exc}") from exc
    if len(encoded.encode("utf-8")) > MAX_MANIFEST_BYTES:
        raise BackupManifestError("backup manifest exceeds the size limit")
    if not isinstance(manifest, Mapping) or manifest.get("schema") != BACKUP_MANIFEST_SCHEMA:
        raise BackupManifestError("unsupported backup manifest schema")
    if manifest.get("version") != BACKUP_MANIFEST_VERSION:
        raise BackupManifestError("unsupported backup manifest version")
    metadata = manifest.get("metadata", {})
    if not isinstance(metadata, Mapping) or len(metadata) > MAX_METADATA_KEYS:
        raise BackupManifestError("invalid backup metadata")
    files = manifest.get("files")
    if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
        raise BackupManifestError("backup manifest files must be a list")
    records: list[BackupArtifact] = []
    seen: set[str] = set()
    for value in files:
        if not isinstance(value, Mapping):
            raise BackupManifestError("backup manifest contains a non-object artifact")
        try:
            path = value["path"]
            size = value["bytes"]
            digest = value["sha256"]
            if not isinstance(path, str) or not isinstance(size, int) or isinstance(size, bool):
                raise BackupManifestError("backup manifest contains an invalid artifact")
            if not isinstance(digest, str):
                raise BackupManifestError("backup manifest contains an invalid artifact")
            record = BackupArtifact(path, size, digest)
        except (KeyError, TypeError, ValueError) as exc:
            raise BackupManifestError("backup manifest contains an invalid artifact") from exc
        if record.path in seen:
            raise BackupManifestError(f"backup manifest contains duplicate artifact: {record.path}")
        seen.add(record.path)
        records.append(record)
    return tuple(records)


def verify_manifest(root: Path, manifest: Mapping[str, Any]) -> BackupVerification:
    """Verify every recorded artifact and return errors instead of partial success."""
    try:
        records = validate_manifest(manifest)
        root_path = Path(root).expanduser().resolve()
        if not root_path.is_dir() or root_path.is_symlink():
            raise BackupManifestError(f"backup root is not a directory: {root_path}")
        for record in records:
            actual = _artifact(root_path, record.path)
            if actual.size != record.size or actual.sha256 != record.sha256:
                raise BackupManifestError(f"checksum mismatch: {record.path}")
        return BackupVerification("verified", str(root_path), len(records))
    except BackupManifestError as exc:
        return BackupVerification("rejected", str(Path(root).expanduser()), 0, (str(exc),))


def prune_backup_generations(root: Path, *, keep: int = 2) -> BackupRetention:
    """Remove old verified generation directories without deleting the last one.

    Invalid, incomplete, and unverified directories are retained for
    inspection rather than guessed at or silently removed.
    """
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 1:
        raise BackupManifestError("backup retention must keep at least one generation")
    root = Path(root).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise BackupManifestError(f"backup root is not a directory: {root}")
    valid: list[tuple[int, Path]] = []
    skipped: list[str] = []
    for candidate in root.iterdir():
        if candidate.is_symlink() or not candidate.is_dir():
            skipped.append(candidate.name)
            continue
        try:
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
            verification = verify_manifest(candidate, manifest)
            if verification.status != "verified":
                skipped.append(candidate.name)
                continue
            valid.append((candidate.stat().st_mtime_ns, candidate))
        except (OSError, TypeError, ValueError):
            skipped.append(candidate.name)
    valid.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    kept = [path for _mtime, path in valid[: int(keep)]]
    removed: list[str] = []
    for _mtime, candidate in valid[int(keep):]:
        try:
            shutil.rmtree(candidate)
        except OSError:
            skipped.append(candidate.name)
            continue
        removed.append(candidate.name)
    return BackupRetention(
        tuple(path.name for path in kept),
        tuple(removed),
        tuple(sorted(set(skipped))),
    )


def manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a validated manifest using Nautical's JSON contract."""
    validate_manifest(manifest)
    return json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True)


def capture_taskwarrior_export(
    taskdata: Path,
    destination: Path,
    *,
    task_bin: str = "task",
    timeout: float = 60.0,
) -> BackupExport:
    """Capture a validated hooks-off Taskwarrior JSON export atomically.

    The destination must be outside Taskdata and must not already exist.  This
    prevents an export from overwriting either live data or a prior artifact;
    callers can publish it into a generation directory after verification.
    """
    source = Path(taskdata).expanduser().resolve()
    target = Path(destination).expanduser().absolute()
    if not source.is_dir():
        raise BackupExportError(f"Taskdata is not a directory: {source}")
    if target.exists() or target.is_symlink():
        raise BackupExportError(f"export destination already exists: {target}")
    try:
        target_resolved = target.resolve()
    except OSError as exc:
        raise BackupExportError(f"export destination cannot be resolved: {target}") from exc
    if target_resolved == source or source in target_resolved.parents:
        raise BackupExportError("export destination must be outside Taskdata")
    env = dict(os.environ)
    env["TASKDATA"] = str(source)
    result = run_task_command(
        task_bin,
        ("rc.hooks=off", "rc.verbose=nothing", "export"),
        env=env,
        timeout=max(0.1, float(timeout)),
        purpose="offline backup Taskwarrior export",
    )
    if not result.ok:
        raise BackupExportError(failure_message(result, "Taskwarrior export"))
    try:
        payload = json.loads(result.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BackupExportError(f"Taskwarrior export is not valid JSON: {exc}") from exc
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise BackupExportError("Taskwarrior export must be a JSON array of objects")
    encoded = result.stdout.encode("utf-8")
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupExportError(f"could not prepare export destination: {exc}") from exc
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=f".{target.name}.", delete=False) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        temporary = None
    except OSError as exc:
        raise BackupExportError(f"could not publish Taskwarrior export: {exc}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except OSError:
                pass
    return BackupExport("captured", str(target), len(payload), len(encoded), hashlib.sha256(encoded).hexdigest())


def backup_outbox_database(taskdata: Path, destination: Path) -> SQLiteBackup:
    """Copy the lifecycle outbox with SQLite's online-backup API.

    The destination is a new standalone database outside Taskdata.  SQLite's
    backup API copies committed pages while handling a live WAL; callers still
    need to quiesce lifecycle mutation for a logically coordinated multi-file
    backup, which is intentionally outside this primitive.
    """
    source_root = Path(taskdata).expanduser().resolve()
    source = source_root / ".nautical-state" / ".nautical_lifecycle_outbox.db"
    target = Path(destination).expanduser().absolute()
    if not source_root.is_dir():
        raise BackupExportError(f"Taskdata is not a directory: {source_root}")
    if source.is_symlink() or not source.is_file():
        raise BackupExportError(f"lifecycle outbox database is unavailable: {source}")
    if target.exists() or target.is_symlink():
        raise BackupExportError(f"outbox backup destination already exists: {target}")
    try:
        target_resolved = target.resolve()
    except OSError as exc:
        raise BackupExportError(f"outbox backup destination cannot be resolved: {target}") from exc
    if target_resolved == source_root or source_root in target_resolved.parents:
        raise BackupExportError("outbox backup destination must be outside Taskdata")
    try:
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise BackupExportError(f"could not prepare outbox backup destination: {exc}") from exc
    source_conn: sqlite3.Connection | None = None
    target_conn: sqlite3.Connection | None = None
    try:
        source_conn = sqlite3.connect(str(source), uri=False)
        target_conn = sqlite3.connect(str(target), uri=False)
        source_conn.backup(target_conn)
        target_conn.commit()
        quick_check = str(target_conn.execute("PRAGMA quick_check").fetchone()[0])
        if quick_check.lower() != "ok":
            raise BackupExportError(f"outbox backup integrity check failed: {quick_check}")
    except BackupExportError:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    except (OSError, sqlite3.Error) as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise BackupExportError(f"could not back up lifecycle outbox: {exc}") from exc
    finally:
        if target_conn is not None:
            target_conn.close()
        if source_conn is not None:
            source_conn.close()
    try:
        size = target.stat().st_size
    except OSError as exc:
        try:
            target.unlink()
        except OSError:
            pass
        raise BackupExportError(f"outbox backup was not published: {exc}") from exc
    return SQLiteBackup("captured", str(source), str(target), size, quick_check)
