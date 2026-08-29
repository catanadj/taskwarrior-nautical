"""Typed public document projection for the local query operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from .operator_models import OperatorFailure, OperatorV2Result, OperatorV2Status


def error_payload(code: str, message: str, *, retryable: bool = False, operation: str = "occurrences") -> dict[str, Any]:
    """Build a stable v1 query failure document for CLI transport errors."""
    return {
        "schema": "nautical.query.next" if operation == "next" else "nautical.query.occurrences",
        "version": 1,
        "operation": operation,
        "status": "invalid" if not retryable else "unavailable",
        "basis": "next" if operation == "next" else "schedule",
        "timezone": None,
        "query": None,
        "configuration_fingerprint": None,
        "results": [],
        "failure": {"code": code, "message": str(message or code), "retryable": retryable, "task_uuid": None, "details": {}},
    }


def to_operator_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a validated v1 query document to the canonical v2 envelope."""
    reserved = {"schema", "version", "operation", "status", "failure"}
    failure_value = payload.get("failure")
    failure = None
    if isinstance(failure_value, Mapping):
        failure = OperatorFailure(
            code=str(failure_value.get("code") or "query_failure"),
            message=str(failure_value.get("message") or "query failed"),
            retryable=bool(failure_value.get("retryable", False)),
            details=cast(Mapping[str, Any], failure_value.get("details"))
            if isinstance(failure_value.get("details"), Mapping)
            else {},
        )
    status = OperatorV2Status(str(payload.get("status") or "error"))
    if failure is None and status in {
        OperatorV2Status.INVALID,
        OperatorV2Status.UNAVAILABLE,
        OperatorV2Status.ERROR,
    }:
        failure = OperatorFailure(
            code="query_unavailable" if status is OperatorV2Status.UNAVAILABLE else "query_failure",
            message=str(payload.get("reason") or "operator result did not include failure evidence"),
            retryable=status is OperatorV2Status.UNAVAILABLE,
        )
    return OperatorV2Result(
        schema=str(payload.get("schema") or "nautical.query.unknown"),
        operation=str(payload.get("operation") or "query"),
        status=status,
        payload={key: value for key, value in payload.items() if key not in reserved},
        failure=failure,
    ).to_dict()


__all__ = ["error_payload", "to_operator_result"]
