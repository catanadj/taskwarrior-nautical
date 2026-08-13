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
from nautical_core import chain_repair  # noqa: E402
from nautical_core.integration_models import Absent, Found, Unavailable  # noqa: E402
from nautical_core.integration_models import (  # noqa: E402
    GuardTimestamp,
    GuardTimestampField,
    MetadataRepairPayload,
    MutationGuard,
    MutationOperation,
    MutationRequest,
    MutationOutcomeKind,
)
from nautical_core.lifecycle_models import recurrence_fingerprint  # noqa: E402
from nautical_core.task_read_repository import ALL_TASK_STATUSES, TaskReadRepository  # noqa: E402
from nautical_core.integration_context import IntegrationAccess  # noqa: E402
from nautical_core.taskwarrior_uow import (  # noqa: E402
    TaskwarriorUnitOfWork,
    build_operator_uow,
)
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService  # noqa: E402


def _export(repository: TaskReadRepository) -> list[dict[str, Any]]:
    repository.configure_commands(timeout=120.0, attempts=2, retry_delay=0.05)
    read = repository.broad_snapshot(
        identity="chain-repair",
        filters=("chainID.not:",),
        statuses=ALL_TASK_STATUSES,
        complete_chain_history=True,
    )
    if isinstance(read, Found):
        return [dict(row) for row in read.value.rows]
    if isinstance(read, Absent):
        return []
    if isinstance(read, Unavailable):
        raise RuntimeError(f"chain repair task read unavailable: {read.evidence.detail}")
    raise RuntimeError("chain repair task repository returned an invalid result")


def _apply_repair(unit_of_work: TaskwarriorUnitOfWork, repair: chain_repair.LinkRepair) -> None:
    read = unit_of_work.repository.by_uuid(repair.uuid, refresh=True)
    if isinstance(read, Unavailable):
        raise RuntimeError(read.evidence.detail or "chain repair guard read unavailable")
    if isinstance(read, Absent):
        raise RuntimeError("chain repair target is absent")
    if not isinstance(read, Found):
        raise RuntimeError("chain repair guard read was invalid")
    row = read.value
    raw_link = str(row.get("link") or "").strip()
    if not raw_link and repair.field == "link":
        # A missing link is the repair target, so the slot identity is the
        # only trustworthy link value available for the guard.  Other repair
        # fields still require an existing numeric link and fail closed.
        link = repair.link
    else:
        try:
            link = int(float(raw_link))
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("chain repair target has no numeric link") from exc
    modified = str(row.get("modified") or "").strip()
    if not modified:
        raise RuntimeError("chain repair target has no modified timestamp")
    guard = MutationGuard(
        task_uuid=repair.uuid,
        status=str(row.get("status") or ""),
        chain_id=str(row.get("chainID") or ""),
        link=link,
        recurrence_identity=recurrence_fingerprint(dict(row)),
        timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=unit_of_work.mutation_epoch,
        chain=str(row.get("chain") or "on"),
    )
    payload = MetadataRepairPayload.from_mapping(
        repair.uuid,
        {repair.field: repair.new},
        expected={repair.field: repair.old},
    )
    request = MutationRequest(MutationOperation.METADATA_REPAIR, guard, payload)
    outcome = TaskwarriorMutationService(unit_of_work).apply(request)
    if outcome.kind not in {MutationOutcomeKind.APPLIED, MutationOutcomeKind.ALREADY_APPLIED}:
        raise RuntimeError(outcome.reason or outcome.kind.value)


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
    _unit_of_work: TaskwarriorUnitOfWork | None = None,
) -> int:
    parser = argparse.ArgumentParser(description="Repair deterministic Nautical chain link gaps.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    args = parser.parse_args(argv)

    if _unit_of_work is None:
        try:
            _unit_of_work = build_operator_uow(
                core=nautical_core_package,
                task_binary=args.task_bin,
                access=IntegrationAccess.MUTATION if args.apply else IntegrationAccess.READ_ONLY,
            )
        except Exception as exc:
            return _failure(args, "integration_context", exc)
    try:
        tasks = _export(_unit_of_work.repository)
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
                _apply_repair(_unit_of_work, repair)
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
