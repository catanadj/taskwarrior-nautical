#!/usr/bin/env python3
"""Install or upgrade Nautical from a local release tree."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core import install_runtime  # noqa: E402
from nautical_core.operator_presentation import render_json_document  # noqa: E402
from nautical_core.operator_presentation import key_value_lines  # noqa: E402


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


def _taskrc_path() -> Path:
    raw = os.environ.get("TASKRC", "").strip()
    return Path(raw).expanduser() if raw else Path.home() / ".taskrc"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent), text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _configure_udas(source: Path, taskdata: Path, *, dry_run: bool) -> dict[str, object]:
    """Install Nautical's UDA include without overwriting user customizations."""
    source_file = source / "uda.conf"
    if not source_file.is_file():
        raise RuntimeError(f"Nautical UDA definition is missing: {source_file}")
    uda_file = taskdata / "uda-nautical.conf"
    taskrc = _taskrc_path()
    include_line = f"include {uda_file}"
    uda_exists = uda_file.is_file()
    taskrc_exists = taskrc.is_file()
    taskrc_text = taskrc.read_text(encoding="utf-8") if taskrc_exists else ""
    included = any(line.strip() == include_line for line in taskrc_text.splitlines())
    actions: list[str] = []
    if not uda_exists:
        actions.append(f"create {uda_file}")
    if not included:
        actions.append(f"include {uda_file} in {taskrc}")
    if dry_run:
        return {
            "status": "planned" if actions else "current",
            "uda_file": str(uda_file),
            "taskrc": str(taskrc),
            "actions": actions,
        }
    taskdata.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not uda_exists:
        _atomic_write(uda_file, source_file.read_text(encoding="utf-8"))
    if not included:
        prefix = taskrc_text
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        _atomic_write(taskrc, f"{prefix}{include_line}\n")
    return {
        "status": "configured" if actions else "current",
        "uda_file": str(uda_file),
        "taskrc": str(taskrc),
        "actions": actions,
    }


def _render(payload: dict) -> None:
    operation = str(payload.get("operation") or "install")
    if payload.get("status") == "dry-run":
        print("Nautical install check: passed")
        print(f"Plan: {_PLAN_LABELS.get(operation, operation.replace('_', ' ').title())}")
    else:
        print("Nautical install: complete")
        print(f"Action: {_RESULT_LABELS.get(operation, operation.replace('_', ' ').title())}")
    stable = {"Release": payload["release_id"]}
    previous = str(payload.get("previous_release") or "")
    if previous and payload.get("status") == "dry-run":
        stable["Current"] = previous
    elif previous and previous != payload.get("active_release"):
        stable["Previous"] = previous
    stable.update({
        "Target": payload["base"],
        "Hooks": payload["hooks_dir"],
        "Command": payload.get("launcher_path") or Path(payload["base"]) / "nautical",
    })
    print("\n".join(key_value_lines(stable)))
    if payload.get("status") == "dry-run":
        print("Changes: none (dry run)")
    else:
        print(f"Launcher: {Path(payload['base']) / 'nautical'}")
        print("Validation: passed")
    uda = payload.get("uda_configuration")
    if isinstance(uda, dict):
        status = str(uda.get("status") or "unknown")
        print(f"UDAs: {status} ({uda.get('uda_file')})")
        for action in uda.get("actions") or []:
            print(f"UDA action: {action}")
    if payload.get("migrated_legacy_core"):
        print(f"Legacy core backup: {payload.get('legacy_backup')}")
    for path in payload.get("migrated_configs") or []:
        print(f"Config preserved: {path}")
    initialized = str(payload.get("initialized_config") or "")
    if initialized:
        print(f"Config initialized: {initialized}")
    planned_config = payload.get("config_initialization")
    if isinstance(planned_config, dict) and planned_config:
        print(f"Config to initialize: {planned_config.get('path')} (tz={planned_config.get('tz')})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(ROOT), help="local release tree to install")
    parser.add_argument(
        "--taskdata",
        default=os.environ.get("TASKDATA") or "~/.task",
        help="Taskwarrior data directory (default: TASKDATA or ~/.task)",
    )
    parser.add_argument("--hooks-dir", default="", help="override the Taskwarrior hooks directory")
    parser.add_argument(
        "--launcher-path",
        default="",
        help="user-facing command path (default: ~/.local/bin/nautical; set to $PREFIX/bin/nautical on Termux)",
    )
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
            launcher_path=(Path(args.launcher_path).expanduser() if args.launcher_path else install_runtime.default_launcher_path()),
            release_id=args.release_id,
            dry_run=bool(args.dry_run),
        )
        payload["uda_configuration"] = _configure_udas(
            Path(args.source).expanduser().resolve(),
            taskdata,
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
            print(render_json_document(payload))
        else:
            print(f"Nautical install failed: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(render_json_document(payload))
    else:
        _render(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
