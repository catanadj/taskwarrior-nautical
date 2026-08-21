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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import nautical_core as core  # noqa: E402
from nautical_core.integration_context import IntegrationAccess  # noqa: E402
from nautical_core.query_models import (  # noqa: E402
    OCCURRENCES_SCHEMA,
    CAPABILITIES_SCHEMA,
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


def _error_payload(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "schema": OCCURRENCES_SCHEMA,
        "version": QUERY_API_VERSION,
        "operation": "occurrences",
        "status": "invalid" if not retryable else "unavailable",
        "basis": "schedule",
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
        "operations": ["occurrences"],
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
        "future_operations": ["next", "inspect", "chains"],
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


def _request_mapping(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.request is not None and args.request_file is not None:
        raise QueryContractError("use either --request or --request-file, not both")
    if args.request == "-":
        raw = sys.stdin.read()
        if not raw.strip():
            raise QueryContractError("stdin request is empty")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise QueryContractError(f"stdin is not valid JSON: {exc}") from exc
    elif args.request is not None:
        try:
            value = json.loads(args.request)
        except json.JSONDecodeError as exc:
            raise QueryContractError(f"--request is not valid JSON: {exc}") from exc
    elif args.request_file is not None:
        try:
            value = json.loads(Path(args.request_file).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QueryContractError(f"request file could not be read as JSON: {exc}") from exc
    else:
        raw = sys.stdin.read()
        if not raw.strip():
            raise QueryContractError("provide a JSON request through stdin, --request, or --request-file")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise QueryContractError(f"stdin is not valid JSON: {exc}") from exc
    if not isinstance(value, Mapping):
        raise QueryContractError("query request must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nautical's versioned read-only query API")
    parser.add_argument("operation", choices=("capabilities", "occurrences"), help="query operation")
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
    parser.add_argument("--to", help="inclusive local date or RFC 3339 timestamp")
    parser.add_argument("--count", type=int, help="maximum number of occurrences per task")
    parser.add_argument("--max-total-occurrences", type=int, help="aggregate occurrence safety cap")
    parser.add_argument("--omissions", choices=("exclude", "include", "report"), dest="omission_policy", default="exclude")
    args = parser.parse_args(argv)
    if args.operation == "capabilities":
        return _emit(_capabilities_payload())
    try:
        flag_values = (
            args.uuids,
            args.chain_id,
            args.all_tasks,
            args.start,
            args.after,
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
                start = args.after or args.start
                mapping = {
                    "selector": selector_mapping,
                    "from": start,
                    "to": args.to,
                    "count": args.count,
                    "max_total_occurrences": args.max_total_occurrences,
                    "start_inclusive": args.after is None,
                    "omission_policy": args.omission_policy,
                }
        mapping.setdefault("operation", args.operation)
        request = OccurrenceQueryRequest.from_mapping(mapping)
        unit_of_work = build_operator_uow(
            core=core,
            task_binary=shutil.which("task") or "task",
            env=os.environ,
            access=IntegrationAccess.READ_ONLY,
        )
        response = OccurrenceQueryService(unit_of_work, core=core).query(request)
        exit_code = 3 if response.status == "unavailable" else 2 if response.status == "invalid" else 0
        return _emit(response.to_dict(), exit_code=exit_code)
    except QueryContractError as exc:
        _diagnostic(str(exc))
        return _emit(_error_payload("invalid_request", str(exc)), exit_code=2)
    except (OSError, RuntimeError, ValueError) as exc:
        _diagnostic(str(exc))
        return _emit(_error_payload("query_unavailable", str(exc), retryable=True), exit_code=3)


if __name__ == "__main__":
    raise SystemExit(main())
