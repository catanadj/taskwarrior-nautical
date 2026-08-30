#!/usr/bin/env python3
"""Read-only inspection of lifecycle intents requiring operator review."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core.queue_status_service import QueueStatusService


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect lifecycle intents requiring manual review")
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"))
    parser.add_argument("--intent", help="inspect one exact intent ID")
    parser.add_argument("--task-bin", default=os.environ.get("NAUTICAL_TASK_BIN", "task"))
    parser.add_argument("--limit", type=int, default=100, help="maximum review intents to return")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--resolve-applied", action="store_true", help="acknowledge only high-confidence already-applied intents")
    parser.add_argument("--all", action="store_true", help="with --resolve-applied, assess every review intent")
    args = parser.parse_args()
    payload = QueueStatusService().review_payload(
        Path(args.taskdata), limit=max(0, args.limit), intent_id=args.intent, task_binary=args.task_bin
    )
    if args.resolve_applied:
        if not args.intent and not args.all:
            parser.error("--resolve-applied requires --intent or --all")
        candidates = payload.get("intents", []) if args.all else payload.get("intents", [])[:1]
        if args.all:
            candidates = []
            listing = QueueStatusService().review_payload(Path(args.taskdata), limit=max(1, args.limit))
            for item in listing.get("intents", []):
                detail = QueueStatusService().review_payload(Path(args.taskdata), intent_id=item.get("intent_id"), task_binary=args.task_bin)
                candidates.extend(detail.get("intents", []))
        resolved = []
        for item in candidates:
            assessment = item.get("assessment") or {}
            if assessment.get("status") != "already_applied":
                continue
            result = QueueStatusService().resolve_review(Path(args.taskdata), item["intent_id"], assessment["message"])
            if result.get("status") in {"resolved", "already_applied"}:
                resolved.append(item["intent_id"])
        payload["resolved"] = resolved
        payload["status"] = "resolved" if resolved else payload["status"]
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"review: {payload['status']} taskdata={payload['taskdata']}")
        for intent in payload["intents"]:
            plan = intent.get("plan") or {}
            guard = plan.get("parent_guard") or {}
            guard_time_field = "end" if guard.get("end") else "modified" if guard.get("modified") else "none"
            failure = intent.get("failure") or {}
            print(
                f"intent: id={intent.get('intent_id')} state={intent.get('state')} "
                f"event={plan.get('event', '')} action={plan.get('action', '')} "
                f"chain={plan.get('chainID', '')} parent={plan.get('parent_uuid', '')} "
                f"links={plan.get('source_link', '')}->{plan.get('target_link', '')} "
                f"reason={failure.get('message') or intent.get('reason', '')} "
                f"expected_{guard_time_field}={guard.get(guard_time_field, '-') or '-'}"
            )
            comparison = intent.get("guard_comparison") or {}
            if comparison:
                print(f"  guard_current: {comparison.get('status')}")
                for difference in comparison.get("differences", []):
                    print(
                        f"  changed: {difference.get('field')} "
                        f"expected={difference.get('expected', '-')!r} "
                        f"actual={difference.get('actual', '-')!r}"
                    )
                if comparison.get("reason"):
                    print(f"  guard_read: {comparison['reason']}")
            assessment = intent.get("assessment") or {}
            if assessment:
                print(
                    f"  assessment: {assessment.get('status')} "
                    f"confidence={assessment.get('confidence', 'unknown')}"
                )
                if assessment.get("message"):
                    print(f"  assessment_detail: {assessment['message']}")
                for option in assessment.get("options", []):
                    print(f"  option: {option}")
        if payload.get("failure"):
            print(f"failure: {payload['failure']['message']}")
    return 0 if payload["status"] in {"found", "empty", "resolved"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
