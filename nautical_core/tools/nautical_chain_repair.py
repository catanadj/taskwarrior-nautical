#!/usr/bin/env python3
"""Repair deterministic prevLink/nextLink gaps inside Nautical chains."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import nautical_core as nautical_core_package  # noqa: E402
from nautical_core.chain_integrity_engine import ChainIntegrityEngine  # noqa: E402
from nautical_core.chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest  # noqa: E402
from nautical_core.lifecycle_outbox import LifecycleOutboxRepository  # noqa: E402
from nautical_core.integration_models import Absent, Found, Unavailable  # noqa: E402
from nautical_core.integration_models import (  # noqa: E402
    GuardTimestamp,
    GuardTimestampField,
    MetadataRepairPayload,
    MutationGuard,
    MutationOperation,
    MutationRequest,
)
from nautical_core.lifecycle_models import recurrence_fingerprint  # noqa: E402
from nautical_core.integration_context import IntegrationAccess  # noqa: E402
from nautical_core.taskwarrior_uow import (  # noqa: E402
    TaskwarriorUnitOfWork,
    build_operator_uow,
)
from nautical_core.taskwarrior_mutations import TaskwarriorMutationService  # noqa: E402


def _request_for_operation(unit_of_work: TaskwarriorUnitOfWork, operation: Any) -> MutationRequest:
    read = unit_of_work.repository.by_uuid(operation.target_uuid, refresh=True)
    if isinstance(read, Unavailable):
        raise RuntimeError(read.evidence.detail or "chain repair guard read unavailable")
    if isinstance(read, Absent):
        raise RuntimeError("chain repair target is absent")
    if not isinstance(read, Found):
        raise RuntimeError("chain repair guard read was invalid")
    row = read.value
    try:
        link = int(float(str(row.get("link") or "").strip()))
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("chain repair target has no numeric link") from exc
    modified = str(row.get("modified") or "").strip()
    if not modified:
        raise RuntimeError("chain repair target has no modified timestamp")
    guard = MutationGuard(
        task_uuid=operation.target_uuid,
        status=str(row.get("status") or ""),
        chain_id=str(row.get("chainID") or operation.chain_id),
        link=link,
        recurrence_identity=recurrence_fingerprint(dict(row)),
        timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
        expected_mutation_epoch=unit_of_work.mutation_epoch,
        chain=str(row.get("chain") or "on"),
    )
    payload = MetadataRepairPayload.from_mapping(
        operation.target_uuid,
        dict(operation.payload),
        expected={key: row.get(key) for key in operation.payload},
    )
    request = MutationRequest(MutationOperation.METADATA_REPAIR, guard, payload)
    return request


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
        config = _unit_of_work.context.configuration
        engine = ChainIntegrityEngine(
            ChainSnapshotService(_unit_of_work, configuration_fingerprint=config.fingerprint),
            configuration_fingerprint=config.fingerprint,
            schedule_fingerprint=config.scheduler_fingerprint,
        )
        result = engine.audit(
            IntegritySnapshotRequest.candidates(complete_chain_history=True),
            outbox_repository=LifecycleOutboxRepository(_unit_of_work.outbox.taskdata),
            mutation_epoch=_unit_of_work.mutation_epoch,
        )
        applied_result = result
        if args.apply and result.plans:
            applied_result = engine.apply(
                result,
                executor=TaskwarriorMutationService(_unit_of_work),
                request_factory=lambda operation: _request_for_operation(_unit_of_work, operation),
                outbox_repository=LifecycleOutboxRepository(_unit_of_work.outbox.taskdata),
                owner=f"chain-repair-{os.getpid()}",
            )
    except Exception as exc:
        return _failure(args, "integrity_audit", exc)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "repairs": len(result.plans),
        "issues": len(result.refusals),
        "applied": [item.__dict__ for item in applied_result.applications],
        "issue_details": [issue.__dict__ for issue in result.refusals],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"summary: {summary['mode']}; repairs={summary['repairs']} issues={summary['issues']}")
    return 1 if result.refusals or result.status.value in {"unavailable", "manual_review"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
