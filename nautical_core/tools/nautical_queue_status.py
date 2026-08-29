#!/usr/bin/env python3
"""Read-only lifecycle outbox inspector used by Nautical operators and doctor."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Any


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core.lifecycle_outbox import (  # noqa: E402
    OUTBOX_SCHEMA_VERSION,
    OUTBOX_ACK_RETENTION_SECONDS,
    LifecycleOutboxRepository,
)
from nautical_core.operator_models import OperatorFailure, OperatorV2Result, OperatorV2Status  # noqa: E402
from nautical_core.operator_models import OperatorLimits  # noqa: E402
from nautical_core.operator_context import OperatorInvocationBudget  # noqa: E402
from nautical_core.operator_presentation import ordered_records, render_result  # noqa: E402
from nautical_core.queue_status_service import QueueStatusService  # noqa: E402


_JSON_SCHEMA = "nautical.lifecycle_outbox_status"
_JSON_SCHEMA_VERSION = 1


def _status_payload(taskdata: Path, *, stale_after: float, limit: int) -> tuple[dict[str, Any], OperatorInvocationBudget]:
    budget = OperatorInvocationBudget(OperatorLimits(outbox_rows=max(1, limit)))
    return QueueStatusService().status_payload(taskdata, stale_after=stale_after, limit=limit, budget=budget), budget


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Nautical lifecycle outbox inspector")
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"))
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--limit", type=int, default=5, help="number of sample intents")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    parser.add_argument(
        "--prune-acknowledged",
        action="store_true",
        help="explicitly remove acknowledged intents older than the retention policy",
    )
    parser.add_argument("--retention-seconds", type=float, default=OUTBOX_ACK_RETENTION_SECONDS)
    parser.add_argument("--maintenance-limit", type=int, default=1000)
    parser.add_argument("--checkpoint", action="store_true", help="request a passive WAL checkpoint during pruning")
    args = parser.parse_args()
    maintenance: dict[str, Any] | None = None
    if args.prune_acknowledged:
        taskdata = Path(args.taskdata).expanduser().resolve()
        result = LifecycleOutboxRepository(taskdata).prune_acknowledged(
            retention_seconds=args.retention_seconds,
            limit=args.maintenance_limit,
            checkpoint=args.checkpoint,
        )
        maintenance = {
            "kind": result.kind.value,
            "ok": result.ok,
            "removed": result.removed,
            "retention_seconds": result.retention_seconds,
            "checkpoint": result.checkpoint,
            "reason": result.reason,
        }
    payload, budget = _status_payload(
        Path(args.taskdata),
        stale_after=max(0.0, float(args.stale_after_seconds)),
        limit=max(0, int(args.limit)),
    )
    if maintenance is not None:
        payload["maintenance"] = maintenance
        if not maintenance["ok"]:
            payload["issues"].append(f"outbox maintenance failed: {maintenance['reason'] or maintenance['kind']}")
            payload["status"] = "error"
    # Queue status retains its v1 ``warn`` spelling in the payload while the
    # shared v2 envelope uses the canonical attention status.
    result_status = OperatorV2Status.ATTENTION if payload["status"] == "warn" else OperatorV2Status(payload["status"])
    result_failure = None
    if result_status in {OperatorV2Status.ERROR, OperatorV2Status.UNAVAILABLE, OperatorV2Status.INVALID}:
        result_failure = OperatorFailure(
            code="queue_status_error",
            message=str((payload.get("issues") or ["queue status failed"])[0]),
        )
    operator_result: OperatorV2Result = OperatorV2Result(
        schema=_JSON_SCHEMA,
        operation="queue",
        status=result_status,
        payload={key: value for key, value in payload.items() if key not in {"schema", "status"}},
        failure=result_failure,
    )
    if args.json:
        print(render_result(operator_result, "json", budget=budget))
    else:
        outbox = payload["outbox"]
        print(render_result(operator_result, "text", budget=budget))
        print(f"status={payload['status']} taskdata={payload['taskdata']}")
        print(
            "outbox:"
            f" states={outbox.get('states', {})}"
            f" stale_claims={outbox.get('stale_claims', 0)}"
            f" max_attempts={outbox.get('max_attempts', 0)}"
        )
        retention = outbox.get("retention") or {}
        print(
            "retention:"
            f" acknowledged={retention.get('acknowledged', 0)}"
            f" eligible={retention.get('eligible', 0)}"
            f" oldest_age_s={retention.get('oldest_age_s', 0)}"
            f" policy_s={retention.get('retention_seconds', OUTBOX_ACK_RETENTION_SECONDS)}"
        )
        if maintenance is not None:
            print(
                "maintenance:"
                f" kind={maintenance['kind']} removed={maintenance['removed']}"
                f" checkpoint={maintenance['checkpoint']}"
            )
        schema = outbox["schema"]
        print(
            "schema:"
            f" status={schema.get('status')} version={schema.get('version')}"
            f" expected={schema.get('expected_version')} integrity={outbox.get('integrity')}"
        )
        for issue in payload["issues"]:
            print(f"issue: {issue}")
        for record in ordered_records(
            outbox.get("sample", []),
            keys=("state", "stage", "intent_id"),
        ):
            print(
                "intent:"
                f" id={record.get('intent_id')} state={record.get('state')}"
                f" stage={record.get('stage')} attempts={record.get('attempts')}"
                + (f" reason={record['reason']}" if record.get("reason") else "")
            )
    return 2 if payload["status"] == "error" else 1 if payload["status"] == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
