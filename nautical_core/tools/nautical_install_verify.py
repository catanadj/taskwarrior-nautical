#!/usr/bin/env python3
"""Render a concise post-install report from Doctor's installation scope."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from nautical_core.operator_presentation import bounded_text, ordered_findings, render_json_document
from nautical_core.operator_findings import OperatorFinding


def _findings(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return canonical finding mappings, accepting direct legacy callers."""
    source = payload.get("operator_findings")
    if not isinstance(source, list):
        source = payload.get("findings") or []
    normalized: list[dict[str, Any]] = []
    for item in source:
        if not isinstance(item, dict):
            continue
        if "code" in item:
            normalized.append(item)
        else:
            normalized.append(OperatorFinding.from_doctor_mapping(item).to_dict())
    return normalized


def _items(payload: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    return [
        item for item in _findings(payload)
        if str(item.get("code") or "").startswith(prefix)
    ]


def _group_status(items: list[dict[str, Any]], *, empty_status: str = "failed") -> str:
    if not items:
        return empty_status
    if any(item.get("severity") == "error" for item in items):
        return "failed"
    if any(item.get("severity") == "warning" for item in items):
        return "attention"
    return "passed"


def build_report(
    payload: dict[str, Any],
    *,
    platform: str,
    launcher: Path,
) -> dict[str, Any]:
    findings = list(ordered_findings(_findings(payload)))
    checks = [
        {"name": "Platform", "status": "passed", "detail": platform},
        {"name": "Taskwarrior", "status": _group_status(_items(payload, "taskwarrior.")), "detail": "command available"},
        {"name": "Taskdata", "status": _group_status(_items(payload, "taskdata.")), "detail": str(payload.get("taskdata") or "")},
        {"name": "Runtime", "status": _group_status(_items(payload, "install.")), "detail": "managed release active"},
        {"name": "Hooks", "status": _group_status(_items(payload, "hook.")), "detail": "add, modify, and exit"},
        {"name": "Launcher", "status": "passed" if launcher.is_file() and os.access(launcher, os.X_OK) else "failed", "detail": str(launcher)},
        {
            "name": "UDAs",
            "status": _group_status(_items(payload, "uda."), empty_status="passed"),
            "detail": "Taskwarrior fields registered",
        },
        {"name": "Timezone", "status": _group_status(_items(payload, "config.timezone")), "detail": "explicit scheduling timezone"},
    ]

    required: list[dict[str, str]] = []
    optional: list[dict[str, str]] = []
    required_prefixes = ("integration.", "taskwarrior.", "taskdata.", "hook.", "uda.", "install.", "config.")
    optional_prefixes = ("navigator.", "astronomy.")
    seen: set[tuple[str, str]] = set()
    for item in findings:
        severity = str(item.get("severity") or "")
        if severity == "info":
            continue
        check_id = str(item.get("code") or "")
        if not check_id.startswith(required_prefixes + optional_prefixes):
            continue
        action = str(item.get("guidance") or item.get("message") or "Inspect this finding.").strip()
        key = (check_id, action)
        if key in seen:
            continue
        seen.add(key)
        record = {"id": check_id, "message": str(item.get("message") or ""), "action": action}
        if check_id.startswith(optional_prefixes) and severity != "error":
            optional.append(record)
        else:
            required.append(record)

    if shutil.which("nautical") is None and launcher.name == "nautical":
        optional.append({
            "id": "launcher.path",
            "message": "The Nautical launcher is installed but its directory is not on PATH.",
            "action": f'Add {launcher.parent} to PATH or invoke {launcher}.',
        })

    failed = any(check["status"] == "failed" for check in checks) or any(
        item.get("severity") == "error"
        and str(item.get("code") or "").startswith(required_prefixes + ("astronomy.",))
        for item in findings
    )
    # Optional environment hints must not make an otherwise valid install
    # appear unhealthy.  Only failed checks or required manual actions block.
    status = "failed" if failed else "attention" if required else "passed"
    return {
        "schema": "nautical.install.verification",
        "version": 1,
        "status": status,
        "checks": checks,
        "manual_actions": required,
        "optional_actions": optional,
    }


def render(report: dict[str, Any]) -> None:
    symbols = {"passed": "+", "attention": "!", "failed": "x"}
    color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
    styles = {
        "passed": "\033[32m" if color else "",
        "attention": "\033[33m" if color else "",
        "failed": "\033[31m" if color else "",
        "heading": "\033[1;36m" if color else "",
        "reset": "\033[0m" if color else "",
    }
    print(f"\n{styles['heading']}Post-install verification{styles['reset']}")
    for check in report.get("checks") or []:
        status = str(check.get("status") or "failed")
        style = styles.get(status, styles["failed"])
        print(f"  {style}{symbols.get(status, '?')}{styles['reset']} {bounded_text(check.get('name'), width=32)}: {bounded_text(check.get('detail'))}")
    manual = report.get("manual_actions") or []
    optional = report.get("optional_actions") or []
    if manual:
        print("\nManual action")
        for item in manual:
            print(f"  ! {bounded_text(item.get('action'))}")
    if optional:
        print("\nOptional")
        for item in optional:
            print(f"  ! {bounded_text(item.get('action'))}")
    status = str(report.get("status") or "failed")
    if status == "failed":
        print("\nInstallation verification failed. Resolve the required actions before using Nautical.")
    elif manual:
        print(f"\nInstallation completed; {len(manual)} manual action(s) remain.")
    elif optional:
        print("\nCore installation verified. Optional enhancements are listed above.")
    else:
        print("\nInstallation verified. Nautical is ready.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Doctor JSON payload")
    parser.add_argument("--platform", default="Linux")
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Doctor payload is not a JSON object")
        scope = str(payload.get("scope") or "full")
        if scope not in {"installation", "full"}:
            raise ValueError("Doctor payload has an unsupported scope")
        report = build_report(payload, platform=args.platform, launcher=Path(args.launcher).expanduser())
    except Exception as exc:
        print(f"Post-install verification could not be read: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(render_json_document(report))
    else:
        render(report)
    return 2 if report["status"] == "failed" else 1 if report["status"] == "attention" else 0


if __name__ == "__main__":
    raise SystemExit(main())
