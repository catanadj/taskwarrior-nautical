#!/usr/bin/env python3
"""Versioned, read-only local query API for Nautical occurrences."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
MAX_REQUEST_BYTES = 1_048_576
MAX_REQUEST_DEPTH = 64
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import nautical_core as core  # noqa: E402
from nautical_core.integration_context import IntegrationAccess  # noqa: E402
from nautical_core.query_models import (  # noqa: E402
    OCCURRENCES_SCHEMA,
    CAPABILITIES_SCHEMA,
    NEXT_SCHEMA,
    NEXT_OPERATION,
    DEFAULT_MAX_FILE_SKIPS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_MAX_OCCURRENCES,
    DEFAULT_MAX_TOTAL_OCCURRENCES,
    DEFAULT_MAX_TASKS,
    HARD_MAX_FILE_SKIPS,
    HARD_MAX_ITERATIONS,
    HARD_MAX_OCCURRENCES,
    HARD_MAX_TOTAL_OCCURRENCES,
    HARD_MAX_TASKS,
    QUERY_API_VERSION,
    OccurrenceQueryRequest,
    QueryContractError,
)
from nautical_core.query_service import OccurrenceQueryService  # noqa: E402
from nautical_core.taskwarrior_uow import build_operator_uow  # noqa: E402

INTEGRITY_SCHEMA = "nautical.query.integrity"


def _error_payload(code: str, message: str, *, retryable: bool = False, operation: str = "occurrences") -> dict[str, Any]:
    return {
        "schema": NEXT_SCHEMA if operation == NEXT_OPERATION else OCCURRENCES_SCHEMA,
        "version": QUERY_API_VERSION,
        "operation": operation,
        "status": "invalid" if not retryable else "unavailable",
        "basis": "next" if operation == NEXT_OPERATION else "schedule",
        "timezone": None,
        "query": None,
        "configuration_fingerprint": None,
        "results": [],
        "failure": {
            "code": code,
            "message": str(message or code),
            "retryable": retryable,
            "task_uuid": None,
            "details": {},
        },
    }


def _capabilities_payload() -> dict[str, Any]:
    """Describe the public query surface without reading Taskwarrior data."""
    return {
        "schema": CAPABILITIES_SCHEMA,
        "version": QUERY_API_VERSION,
        "status": "ok",
        "operations": ["occurrences", "next", "integrity"],
        "next": {
            "basis": "read-only projected successor",
            "reference": "CP uses end when present, otherwise due/scheduled; anchors use due/scheduled",
            "evaluation": "Use at with an RFC 3339 timestamp for mode-aware daily progress.",
            "metadata": ["chain", "lifecycle", "lifecycle.daily_instances"],
            "mutates_taskwarrior": False,
        },
        "selectors": ["uuid", "chain_id", "all"],
        "omission_policies": ["exclude", "include", "report"],
        "timestamps": {
            "formats": ["ISO-8601 date", "RFC 3339 timestamp with explicit offset"],
            "date_semantics": "local calendar date in Nautical's configured timezone",
            "returned": ["local", "utc"],
        },
        "limits": {
            "defaults": {
                "tasks": DEFAULT_MAX_TASKS,
                "occurrences": DEFAULT_MAX_OCCURRENCES,
                "total_occurrences": DEFAULT_MAX_TOTAL_OCCURRENCES,
                "iterations": DEFAULT_MAX_ITERATIONS,
                "file_skips": DEFAULT_MAX_FILE_SKIPS,
            },
            "hard": {
                "tasks": HARD_MAX_TASKS,
                "occurrences": HARD_MAX_OCCURRENCES,
                "total_occurrences": HARD_MAX_TOTAL_OCCURRENCES,
                "iterations": HARD_MAX_ITERATIONS,
                "file_skips": HARD_MAX_FILE_SKIPS,
            },
        },
        "providers": {
            "astronomy": bool(importlib.util.find_spec("astral")),
        },
        "guide": {
            "intro": (
                "Nautical adds recurrence rules to Taskwarrior. An anchor describes "
                "calendar matches; cp (completion period) schedules the next task "
                "from completion or its end."
            ),
            "concepts": {
                "anchor": "Calendar recurrence, for example w:mon..fri@t=09:00.",
                "anchor_file": "Dates supplied by one or more files, optionally combined with an anchor.",
                "cp": "Completion-based recurrence, for example 1d or rand(11d..14d).",
                "chain": "Nautical links recurring tasks with chainID and link metadata.",
                "omissions": "Use exclude, include, or report to control omitted calendar matches.",
            },
            "quick_start": [
                "nautical query capabilities",
                "nautical query occurrences --uuid TASK_UUID --from 2026-08-24 --count 5",
                "nautical query occurrences --chain-id CHAIN_ID --from 2026-08-24 --count 10",
                "nautical query occurrences --all --from 2026-08-24 --to 2026-08-31",
                "nautical query next --uuid TASK_UUID --from 2026-08-24 --count 1",
                "nautical query next --uuid TASK_UUID --at 2026-08-24T15:00:00+03:00",
                "nautical query integrity --chain-id CHAIN_ID",
            ],
            "consumer_rule": (
                "Read the versioned JSON schema, statuses, and failure codes; do not "
                "import Nautical internals or parse human-readable panels."
            ),
            "task_range_rule": (
                "For task selectors, occurrences never precede the task's current due "
                "or scheduled reference; expression-only calendar expansion is a "
                "separate future operation."
            ),
        },
        "future_operations": ["inspect", "chains"],
    }


def _emit(payload: Mapping[str, Any], *, exit_code: int = 0) -> int:
    try:
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except BrokenPipeError:
        return 0
    return exit_code


def _diagnostic(message: str) -> None:
    if os.environ.get("NAUTICAL_DIAG") == "1":
        sys.stderr.write(f"[nautical] query: {message}\n")


def _integrity_payload(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    """Run a read-only integrity audit through the shared engine boundary."""
    from nautical_core.chain_integrity_engine import ChainIntegrityEngine
    from nautical_core.chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    selected = sum(bool(value) for value in (args.uuids, args.chain_id, args.all_tasks))
    if selected != 1:
        raise QueryContractError("integrity query requires exactly one of --uuid, --chain-id, or --all")
    if args.uuids:
        if len(args.uuids) != 1:
            raise QueryContractError("integrity query accepts one --uuid")
        request = IntegritySnapshotRequest.uuid(args.uuids[0], complete_chain_history=True)
    elif args.chain_id:
        request = IntegritySnapshotRequest.chain(args.chain_id)
    else:
        request = IntegritySnapshotRequest.candidates()
    unit_of_work = build_operator_uow(
        core=core,
        task_binary=shutil.which("task") or "task",
        env=os.environ,
        access=IntegrationAccess.READ_ONLY,
    )
    configuration = unit_of_work.context.configuration
    engine = ChainIntegrityEngine(
        ChainSnapshotService(unit_of_work, configuration_fingerprint=configuration.fingerprint),
        configuration_fingerprint=configuration.fingerprint,
        schedule_fingerprint=configuration.scheduler_fingerprint,
    )
    result = engine.audit(
        request,
        outbox_repository=LifecycleOutboxRepository(unit_of_work.outbox.taskdata),
        mutation_epoch=unit_of_work.mutation_epoch,
    )
    snapshot = result.snapshot
    payload = {
        "schema": INTEGRITY_SCHEMA,
        "version": 1,
        "operation": "integrity",
        "status": result.status.value,
        "configuration_fingerprint": configuration.fingerprint,
        "query": {"kind": request.kind.value, "chainID": request.chain_id or None, "uuid": request.task_uuid or None},
        "snapshot": snapshot.to_dict() if snapshot is not None else None,
        "findings": [finding.to_dict() for finding in result.findings],
        "plans": [plan.to_dict() for plan in result.plans],
        "refusals": [
            {"invariant_id": item.invariant_id, "reason_code": item.reason_code,
             "reason": item.reason, "snapshot_id": item.snapshot_id}
            for item in result.refusals
        ],
        "chain_statuses": [{"chainID": chain_id, "status": status.value} for chain_id, status in result.chain_statuses],
        "failure": {"message": result.reason} if result.reason else None,
    }
    return payload, 3 if result.status.value == "unavailable" else 0


def _decode_request(raw: str, source: str) -> Mapping[str, Any]:
    if len(raw.encode("utf-8")) > MAX_REQUEST_BYTES:
        raise QueryContractError(f"{source} exceeds the {MAX_REQUEST_BYTES}-byte limit")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise QueryContractError(f"{source} is not valid JSON: {exc}") from exc
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_REQUEST_DEPTH:
            raise QueryContractError(f"{source} exceeds the JSON nesting limit ({MAX_REQUEST_DEPTH})")
        if isinstance(item, Mapping):
            pending.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
    if not isinstance(value, Mapping):
        raise QueryContractError("query request must be a JSON object")
    return value


def _request_mapping(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.request is not None and args.request_file is not None:
        raise QueryContractError("use either --request or --request-file, not both")
    if args.request == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            raise QueryContractError("stdin request is empty")
        value = _decode_request(raw, "stdin")
    elif args.request is not None:
        value = _decode_request(args.request, "--request")
    elif args.request_file is not None:
        try:
            raw = Path(args.request_file).read_text(encoding="utf-8")
        except OSError as exc:
            raise QueryContractError(f"request file could not be read: {exc}") from exc
        value = _decode_request(raw, "request file")
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            raise QueryContractError("provide a JSON request through stdin, --request, or --request-file")
        value = _decode_request(raw, "stdin")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nautical's versioned read-only query API")
    parser.add_argument("operation", choices=("capabilities", "occurrences", "next", "integrity"), help="query operation")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--request", help="inline JSON request object, or '-' for stdin")
    source.add_argument("--request-file", help="path to a JSON request object")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--uuid", action="append", dest="uuids", help="task UUID or unambiguous prefix (repeatable)")
    selector.add_argument("--chain-id", dest="chain_id", help="chain identity")
    selector.add_argument("--all", action="store_true", dest="all_tasks", help="all active Nautical tasks")
    boundary = parser.add_mutually_exclusive_group()
    boundary.add_argument("--from", dest="start", help="inclusive local date or RFC 3339 timestamp")
    boundary.add_argument("--after", dest="after", help="exclusive local date or RFC 3339 timestamp")
    boundary.add_argument("--at", help="lifecycle evaluation timestamp for the next operation")
    parser.add_argument("--to", help="inclusive local date or RFC 3339 timestamp")
    parser.add_argument("--count", type=int, help="maximum number of occurrences per task")
    parser.add_argument("--max-total-occurrences", type=int, help="aggregate occurrence safety cap")
    parser.add_argument("--omissions", choices=("exclude", "include", "report"), dest="omission_policy", default="exclude")
    args = parser.parse_args(argv)
    if args.operation == "capabilities":
        return _emit(_capabilities_payload())
    if args.operation == "integrity":
        try:
            payload, exit_code = _integrity_payload(args)
            return _emit(payload, exit_code=exit_code)
        except QueryContractError as exc:
            _diagnostic(str(exc))
            return _emit({"schema": INTEGRITY_SCHEMA, "version": 1, "operation": "integrity",
                          "status": "invalid", "findings": [], "plans": [],
                          "failure": {"code": "invalid_request", "message": str(exc)}}, exit_code=2)
        except (OSError, RuntimeError, ValueError) as exc:
            _diagnostic(str(exc))
            return _emit({"schema": INTEGRITY_SCHEMA, "version": 1, "operation": "integrity",
                          "status": "unavailable", "findings": [], "plans": [],
                          "failure": {"code": "query_unavailable", "message": str(exc)}}, exit_code=3)
    try:
        flag_values = (
            args.uuids,
            args.chain_id,
            args.all_tasks,
            args.start,
            args.after,
            args.at,
            args.to,
            args.count,
            args.max_total_occurrences,
            args.omission_policy if args.omission_policy != "exclude" else None,
        )
        has_flags = any(value not in (None, False, []) for value in flag_values)
        if (args.request is not None or args.request_file is not None) and has_flags:
            raise QueryContractError("JSON request input cannot be combined with selector or range flags")
        if args.request is not None or args.request_file is not None:
            mapping = dict(_request_mapping(args))
        else:
            if args.uuids:
                selector_mapping = {"uuids": args.uuids}
            elif args.chain_id:
                selector_mapping = {"chainID": args.chain_id}
            elif args.all_tasks:
                selector_mapping = {"all_tasks": True}
            else:
                # Preserve stdin as the default transport when no flags were supplied.
                mapping = dict(_request_mapping(args))
                selector_mapping = None
            if selector_mapping is not None:
                start = args.after or args.start or args.at
                mapping = {
                    "selector": selector_mapping,
                    "from": start,
                    "to": args.to,
                    "count": args.count,
                    "start_inclusive": args.after is None,
                    "omission_policy": args.omission_policy,
                }
                if args.at is not None:
                    mapping.pop("from", None)
                    mapping["at"] = args.at
                if args.max_total_occurrences is not None:
                    mapping["max_total_occurrences"] = args.max_total_occurrences
        mapping.setdefault("operation", args.operation)
        request = OccurrenceQueryRequest.from_mapping(mapping)
        unit_of_work = build_operator_uow(
            core=core,
            task_binary=shutil.which("task") or "task",
            env=os.environ,
            access=IntegrationAccess.READ_ONLY,
        )
        service = OccurrenceQueryService(unit_of_work, core=core)
        response = service.query_next(request) if request.operation == NEXT_OPERATION else service.query(request)
        exit_code = 3 if response.status == "unavailable" else 2 if response.status == "invalid" else 0
        return _emit(response.to_dict(), exit_code=exit_code)
    except QueryContractError as exc:
        _diagnostic(str(exc))
        return _emit(_error_payload("invalid_request", str(exc), operation=args.operation), exit_code=2)
    except (OSError, RuntimeError, ValueError) as exc:
        _diagnostic(str(exc))
        return _emit(_error_payload("query_unavailable", str(exc), retryable=True, operation=args.operation), exit_code=3)


if __name__ == "__main__":
    raise SystemExit(main())
