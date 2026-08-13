#!/usr/bin/env python3
"""Repair deterministic prevLink/nextLink gaps inside Nautical chains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import nautical_core as nautical_core_package  # noqa: E402
from nautical_core import chain_repair, task_command  # noqa: E402
from nautical_core.integration_context import (  # noqa: E402
    IntegrationAccess,
    IntegrationContext,
    build_operator_context,
)


def _run_task(command_prefix: tuple[str, ...], args: list[str], *, timeout: float = 60.0, read_only: bool = False):
    return task_command.run_task_command(
        command_prefix[0],
        [*command_prefix[1:], *args],
        timeout=timeout,
        retry_locks=read_only,
    )


def _export(command_prefix: tuple[str, ...]) -> list[dict[str, Any]]:
    proc = _run_task(
        command_prefix,
        ["rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", "chainID.not:", "export"],
        timeout=120.0,
        read_only=True,
    )
    payload = task_command.load_json_result(proc, "task export", empty=[])
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError("task export returned a non-list payload")
    return [row for row in payload if isinstance(row, dict)]


def _apply_repair(command_prefix: tuple[str, ...], repair: chain_repair.LinkRepair) -> None:
    proc = _run_task(
        command_prefix,
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
        raise RuntimeError(task_command.failure_message(proc, f"failed to update {repair.short}"))


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
        reason = str(task.get("reason") or "").strip()
        if reason:
            print(f"    why: {reason}")


def _failure(args: argparse.Namespace, stage: str, exc: Exception) -> int:
    reason = str(exc).strip() or type(exc).__name__
    if args.json:
        print(
            json.dumps(
                {
                    "mode": "apply" if args.apply else "dry-run",
                    "stage": stage,
                    "error": reason,
                    "repairs": 0,
                    "issues": 0,
                    "applied": [],
                    "issue_details": [],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"error: {stage.replace('_', ' ')}: {reason}", file=sys.stderr)
    return 1


def main(
    argv: list[str] | None = None,
    *,
    _integration_context: IntegrationContext | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Repair deterministic Nautical chain link gaps.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    args = parser.parse_args(argv)

    if _integration_context is None:
        try:
            _integration_context = build_operator_context(
                core=nautical_core_package,
                task_binary=args.task_bin,
                access=IntegrationAccess.MUTATION if args.apply else IntegrationAccess.READ_ONLY,
            )
        except Exception as exc:
            return _failure(args, "integration_context", exc)
    command_prefix = _integration_context.command_prefix
    try:
        tasks = _export(command_prefix)
    except Exception as exc:
        return _failure(args, "task_export", exc)
    repairs, issues = chain_repair.plan_chain_link_repairs(tasks)
    applied: list[dict[str, Any]] = []
    apply_error: dict[str, Any] | None = None

    for issue in issues:
        if not args.json:
            _print_issue(issue)

    for repair in repairs:
        if args.apply:
            try:
                _apply_repair(command_prefix, repair)
            except Exception as exc:
                apply_error = {
                    "repair": repair.__dict__,
                    "error": str(exc).strip() or type(exc).__name__,
                }
                if not args.json:
                    print(f"error: repair apply: {apply_error['error']}", file=sys.stderr)
                break
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
    if apply_error is not None:
        summary["error"] = apply_error
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        error_suffix = " errors=1" if apply_error is not None else ""
        print(f"summary: {summary['mode']}; repairs={summary['repairs']} issues={summary['issues']}{error_suffix}")
    return 1 if issues or apply_error is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
