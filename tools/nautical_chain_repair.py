#!/usr/bin/env python3
"""Repair deterministic prevLink/nextLink gaps inside Nautical chains."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core import chain_repair  # noqa: E402


def _run_task(task_bin: str, args: list[str], *, timeout: float = 60.0):
    return subprocess.run(
        [task_bin, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _export(task_bin: str) -> list[dict[str, Any]]:
    proc = _run_task(
        task_bin,
        ["rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", "chainID.not:", "export"],
        timeout=120.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "task export failed").strip())
    payload = json.loads((proc.stdout or "").strip() or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError("task export returned a non-list payload")
    return [row for row in payload if isinstance(row, dict)]


def _apply_repair(task_bin: str, repair: chain_repair.LinkRepair) -> None:
    proc = _run_task(
        task_bin,
        [
            "rc.hooks=off",
            "rc.confirmation=off",
            "rc.verbose=nothing",
            f"uuid:{repair.uuid}",
            "modify",
            f"{repair.field}:{repair.new}",
        ],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"failed to update {repair.short}").strip())


def _print_repair(repair: chain_repair.LinkRepair, *, applied: bool) -> None:
    old = repair.old or "-"
    suffix = " applied" if applied else ""
    print(f"repair:{suffix} {repair.short} chain {repair.chain_id} link {repair.link} {repair.field}: {old} -> {repair.new}")


def _print_issue(issue: chain_repair.ChainIssue) -> None:
    print(f"issue: {issue.chain_id} {issue.kind}: {issue.message}")
    for task in issue.tasks[:5]:
        print(
            "  "
            f"{task.get('short') or '????????'} link {task.get('link') or '-'} "
            f"prev:{task.get('prevLink') or '-'} next:{task.get('nextLink') or '-'} "
            f"{task.get('description') or ''}".rstrip()
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair deterministic Nautical chain link gaps.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    args = parser.parse_args(argv)

    tasks = _export(args.task_bin)
    repairs, issues = chain_repair.plan_chain_link_repairs(tasks)
    applied: list[dict[str, Any]] = []

    for issue in issues:
        if not args.json:
            _print_issue(issue)

    for repair in repairs:
        if args.apply:
            _apply_repair(args.task_bin, repair)
            applied.append(repair.__dict__)
        if not args.json:
            _print_repair(repair, applied=args.apply)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "repairs": len(repairs),
        "issues": len(issues),
        "applied": applied,
        "issue_details": [issue.__dict__ for issue in issues],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"summary: {summary['mode']}; repairs={summary['repairs']} issues={summary['issues']}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
