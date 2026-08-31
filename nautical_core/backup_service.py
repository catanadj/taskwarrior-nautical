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
import tempfile
from typing import Any, Iterable, Mapping, Sequence


BACKUP_MANIFEST_SCHEMA = "nautical.backup"
BACKUP_MANIFEST_VERSION = 1
MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_PATH_BYTES = 4096
MAX_METADATA_KEYS = 128


class BackupManifestError(ValueError):
    """Raised when an inventory or manifest violates the backup contract."""


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
            record = BackupArtifact(str(value["path"]), int(value["bytes"]), str(value["sha256"]))
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


def manifest_json(manifest: Mapping[str, Any]) -> str:
    """Serialize a validated manifest using Nautical's JSON contract."""
    validate_manifest(manifest)
    return json.dumps(dict(manifest), ensure_ascii=False, sort_keys=True)
