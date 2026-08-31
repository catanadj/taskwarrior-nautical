#!/usr/bin/env python3
"""Build and verify a self-contained, network-free Nautical recovery kit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
KIT_SCHEMA = 1
ROOT_FILES = ("nautical", "nautical_navigator.py", "on-add.nautical", "on-modify.nautical", "on-exit.nautical", "uda.conf", "requirements.txt", "config-nautical.toml")
KIT_TOOL = "dev_tools/nautical_offline_kit.py"
DOC_FILES = ("docs/getting-started/installation.md", "docs/operations/troubleshooting.md", "docs/operations/offline-readiness.md")


class KitError(RuntimeError):
    pass


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory() -> dict[str, Any]:
    versions: dict[str, str] = {}
    for name in ("astral", "rich", "prompt_toolkit", "python-dateutil"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    task = shutil.which("task")
    task_version = "unavailable"
    if task:
        try:
            result = subprocess.run([task, "--version"], capture_output=True, text=True, check=False, timeout=5)
            task_version = (result.stdout or result.stderr).strip().splitlines()[0]
        except (OSError, subprocess.SubprocessError):
            task_version = "unavailable"
    return {"python": sys.version.split()[0], "python_executable": sys.executable, "taskwarrior": task_version, "python_packages": versions, "timezone": os.environ.get("TZ", "") or "system"}


def _files(root: Path) -> list[Path]:
    paths = [root / name for name in ROOT_FILES]
    paths.extend((root / "nautical_core", root / KIT_TOOL))
    paths.extend(root / name for name in DOC_FILES)
    result: list[Path] = []
    for path in paths:
        if path.is_dir():
            result.extend(
                sorted(
                    item
                    for item in path.rglob("*")
                    if item.is_file()
                    and not item.is_symlink()
                    and ".nautical-cache" not in item.relative_to(root).parts
                    and "__pycache__" not in item.relative_to(root).parts
                    and item.suffix != ".pyc"
                )
            )
        elif path.is_file() and not path.is_symlink():
            result.append(path)
        else:
            raise KitError(f"required kit input is missing or unsafe: {path}")
    return sorted(result, key=lambda item: item.relative_to(root).as_posix())


def build(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise KitError(f"destination already exists: {destination}")
    if destination == source or source in destination.parents:
        raise KitError("kit destination must not be inside the source tree")
    inputs = _files(source)
    with tempfile.TemporaryDirectory(prefix="nautical-kit-", dir=str(destination.parent)) as temporary:
        staging = Path(temporary) / "kit"
        staging.mkdir(mode=0o700)
        for path in inputs:
            relative = path.relative_to(source)
            target = staging / relative
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copy2(path, target)
        records = [{"path": path.relative_to(source).as_posix(), "bytes": path.stat().st_size, "sha256": _digest(path)} for path in inputs]
        manifest = {"schema": KIT_SCHEMA, "inventory": _inventory(), "files": records}
        (staging / "kit-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        checksums = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
        (staging / "checksums.sha256").write_text(checksums, encoding="ascii")
        (staging / "OFFLINE-README.txt").write_text("Use the included local source with: ./nautical install --source .\nVerify first with: python3 dev_tools/nautical_offline_kit.py verify .\nNetworking is not required.\n", encoding="utf-8")
        os.replace(staging, destination)
    return {"status": "created", "kit": str(destination), "files": len(records), "manifest": str(destination / "kit-manifest.json")}


def verify(kit: Path) -> dict[str, Any]:
    kit = kit.expanduser().resolve()
    manifest_path = kit / "kit-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KitError(f"manifest is unreadable: {exc}") from exc
    if manifest.get("schema") != KIT_SCHEMA or not isinstance(manifest.get("files"), list):
        raise KitError("unsupported or malformed kit manifest")
    checked = 0
    for item in manifest["files"]:
        relative = item.get("path") if isinstance(item, dict) else None
        expected = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not isinstance(expected, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise KitError("manifest contains an unsafe file path")
        path = kit / relative
        if path.is_symlink() or not path.is_file() or kit not in path.resolve().parents:
            raise KitError(f"kit file is missing or escapes kit: {relative}")
        if _digest(path) != expected or path.stat().st_size != int(item.get("bytes", -1)):
            raise KitError(f"checksum mismatch: {relative}")
        checked += 1
    return {"status": "verified", "kit": str(kit), "files": checked, "manifest": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("destination")
    build_parser.add_argument("--source", default=str(ROOT))
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("kit")
    args = parser.parse_args()
    try:
        result = build(Path(args.source), Path(args.destination)) if args.command == "build" else verify(Path(args.kit))
    except KitError as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
