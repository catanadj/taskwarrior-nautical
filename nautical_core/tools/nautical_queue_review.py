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
from nautical_core.integration_context import IntegrationRuntime


_ACTION_LABELS = {
    "skip": "skip — leave unresolved",
    "resolve-applied": "resolve-applied — acknowledge only if already verified",
}


def _render_item(intent: dict, number: int) -> None:
    failure = intent.get("failure") or {}
    review_item = intent.get("review_item") or {}
    evidence = review_item.get("evidence") or {}
    if evidence:
        print(f"{number}. Chain {evidence.get('chain_id', '-')} · links {evidence.get('source_link', '-')} → {evidence.get('target_link', '-')}")
        context = review_item.get("task_context") or {}
        if context.get("when"):
            print(f"   When: {context['when']}")
        if context.get("description"):
            print(f"   Task: {context['description']}")
        print(f"   Problem: {evidence.get('reason') or failure.get('message') or intent.get('reason', 'manual review required')}")
        print(f"   Parent: {evidence.get('parent_uuid', '-')}   Child: {evidence.get('expected_child_uuid', '-')}")
        occupants = evidence.get("occupants") or []
        if occupants and tuple(occupants) != (evidence.get("parent_uuid"), evidence.get("expected_child_uuid")):
            print(f"   Occupants: {', '.join(occupants)}")
        if review_item.get("confirmation_available"):
            print("   Confirm: [redacted; use --json for action]")
        for option in review_item.get("actions", []):
            print(f"   Safe action: {_ACTION_LABELS.get(option, option)}")
    else:
        print(f"{number}. {intent.get('intent_id', '-')} — {failure.get('message') or intent.get('reason', 'manual review required')}")
    comparison = intent.get("guard_comparison") or {}
    if comparison and not str(intent.get("intent_id", "")).startswith("integrity:"):
        changed = [str(item.get("field")) for item in comparison.get("differences", []) if item.get("field")]
        if changed:
            print(f"   Guard: {comparison.get('status', 'changed')} ({', '.join(changed)})")
    assessment = intent.get("assessment") or {}
    if assessment:
        print(f"   Assessment: {assessment.get('message') or assessment.get('status', 'unknown')}")


def _render_payload(payload: dict) -> None:
    print(f"review: {payload['status']} taskdata={payload['taskdata']}")
    for number, intent in enumerate(payload.get("intents", []), 1):
        _render_item(intent, number)
    if payload.get("failure"):
        print(f"failure: {payload['failure']['message']}")


def _interactive_next(service: QueueStatusService, taskdata: Path, task_binary: str, limit: int, runtime: IntegrationRuntime) -> int:
    seen: set[str] = set()
    while True:
        listing = service.review_payload(taskdata, limit=max(1, limit), task_binary=task_binary, runtime=runtime)
        candidates = [item for item in listing.get("intents", []) if item.get("intent_id") not in seen]
        if not candidates:
            if not seen:
                _render_payload(listing)
            else:
                print("review: no more unresolved items")
            return 0
        selected = str(candidates[0].get("intent_id") or "")
        detail = service.review_payload(taskdata, limit=1, intent_id=selected, task_binary=task_binary, runtime=runtime)
        if detail.get("status") != "found" or not detail.get("intents"):
            seen.add(selected)
            continue
        _render_payload(detail)
        item = detail["intents"][0]
        token = str((item.get("review_item") or {}).get("confirmation_token") or "")
        print("   Choose: [s]kip  [r]esolve-applied  [n]ext  [q]uit")
        try:
            choice = input("> ").strip().lower()
        except EOFError:
            print()
            return 0
        if choice in {"q", "quit", "exit"}:
            return 0
        seen.add(selected)
        if choice in {"n", "next", ""}:
            continue
        action = {"s": "skip", "skip": "skip", "r": "resolve-applied", "resolve-applied": "resolve-applied"}.get(choice)
        if not action:
            print("Unrecognised choice; use s, r, n, or q.")
            continue
        result = service.apply_review_action(taskdata, selected, action, token, task_binary=task_binary, runtime=runtime)
        print(f"action: {result.get('status')} — {result.get('reason', '')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect lifecycle intents requiring manual review")
    parser.add_argument("--next", action="store_true", help="show one highest-priority review item")
    parser.add_argument("--taskdata", default=os.environ.get("TASKDATA", "~/.task"))
    parser.add_argument("--intent", help="inspect one exact intent ID")
    parser.add_argument("--task-bin", default=os.environ.get("NAUTICAL_TASK_BIN", "task"))
    parser.add_argument("--limit", type=int, default=100, help="maximum review intents to return")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--resolve-applied", action="store_true", help="acknowledge only high-confidence already-applied intents")
    parser.add_argument("--action", choices=("resolve-applied", "skip"), help="apply one guarded review action")
    parser.add_argument("--confirm", help="confirmation token printed with the review item")
    parser.add_argument("--all", action="store_true", help="with --resolve-applied, assess every review intent")
    args = parser.parse_args()
    if args.action and (not args.intent or not args.confirm):
        parser.error("--action requires --intent and --confirm")
    service = QueueStatusService()
    import nautical_core as core
    runtime = IntegrationRuntime.from_compatibility_facade(core)
    if args.next and args.intent:
        parser.error("--next cannot be combined with --intent")
    if args.next and not args.json and not args.action:
        return _interactive_next(service, Path(args.taskdata), args.task_bin, args.limit, runtime)
    payload = service.review_payload(
        Path(args.taskdata), limit=1 if args.next else max(0, args.limit),
        intent_id=args.intent, task_binary=args.task_bin, runtime=runtime,
    )
    if args.next and payload.get("intents"):
        selected = payload["intents"][0].get("intent_id")
        payload = service.review_payload(
            Path(args.taskdata), limit=1, intent_id=selected, task_binary=args.task_bin, runtime=runtime
        )
    if args.resolve_applied:
        if not args.intent and not args.all:
            parser.error("--resolve-applied requires --intent or --all")
        candidates = payload.get("intents", []) if args.all else payload.get("intents", [])[:1]
        if args.all:
            candidates = []
            listing = QueueStatusService().review_payload(Path(args.taskdata), limit=max(1, args.limit), runtime=runtime)
            for item in listing.get("intents", []):
                detail = QueueStatusService().review_payload(Path(args.taskdata), intent_id=item.get("intent_id"), task_binary=args.task_bin, runtime=runtime)
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
    if args.action:
        payload = service.apply_review_action(
            Path(args.taskdata), args.intent, args.action, args.confirm, task_binary=args.task_bin, runtime=runtime
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        _render_payload(payload)
    return 0 if payload["status"] in {"found", "empty", "resolved"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
