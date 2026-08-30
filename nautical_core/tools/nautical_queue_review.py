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
    parser.add_argument("--limit", type=int, default=100, help="maximum review intents to return")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args()
    payload = QueueStatusService().review_payload(
        Path(args.taskdata), limit=max(0, args.limit), intent_id=args.intent
    )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        print(f"review: {payload['status']} taskdata={payload['taskdata']}")
        for intent in payload["intents"]:
            plan = intent.get("plan") or {}
            failure = intent.get("failure") or {}
            print(
                f"intent: id={intent.get('intent_id')} state={intent.get('state')} "
                f"event={plan.get('event', '')} action={plan.get('action', '')} "
                f"chain={plan.get('chainID', '')} parent={plan.get('parent_uuid', '')} "
                f"links={plan.get('source_link', '')}->{plan.get('target_link', '')} "
                f"reason={failure.get('message') or intent.get('reason', '')}"
            )
        if payload.get("failure"):
            print(f"failure: {payload['failure']['message']}")
    return 0 if payload["status"] in {"found", "empty"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
