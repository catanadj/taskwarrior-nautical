#!/usr/bin/env python3
"""Versioned, read-only local query API for Nautical occurrences."""

from __future__ import annotations

import argparse
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


def _emit(payload: Mapping[str, Any]) -> int:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    return 0


def _request_mapping(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.request is not None and args.request_file is not None:
        raise QueryContractError("use either --request or --request-file, not both")
    if args.request is not None:
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
    parser.add_argument("operation", choices=("occurrences",), help="query operation")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--request", help="inline JSON request object")
    source.add_argument("--request-file", help="path to a JSON request object")
    args = parser.parse_args(argv)
    try:
        mapping = dict(_request_mapping(args))
        mapping.setdefault("operation", args.operation)
        request = OccurrenceQueryRequest.from_mapping(mapping)
        unit_of_work = build_operator_uow(
            core=core,
            task_binary=shutil.which("task") or "task",
            env=os.environ,
            access=IntegrationAccess.READ_ONLY,
        )
        response = OccurrenceQueryService(unit_of_work, core=core).query(request)
        return _emit(response.to_dict())
    except QueryContractError as exc:
        return _emit(_error_payload("invalid_request", str(exc)))
    except (OSError, RuntimeError, ValueError) as exc:
        return _emit(_error_payload("query_unavailable", str(exc), retryable=True))


if __name__ == "__main__":
    raise SystemExit(main())
