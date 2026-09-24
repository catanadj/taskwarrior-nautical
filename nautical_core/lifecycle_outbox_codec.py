"""Pure serialization and transition helpers for the lifecycle outbox."""

from __future__ import annotations

import json
from typing import Any

from .lifecycle_models import ExecutionStage, LifecycleContractError, LifecyclePlan
_STAGE_ORDER = {
    ExecutionStage.PLANNED: 0,
    ExecutionStage.PERSISTED: 0,
    ExecutionStage.CHILD_PRESENT: 1,
    ExecutionStage.PARENT_LINKED: 2,
    ExecutionStage.VERIFIED: 3,
    ExecutionStage.FINALIZED: 4,
}


def plan_json(plan: LifecyclePlan) -> str:
    return json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def decode_plan(value: Any, *, error_type: type[Exception] = ValueError) -> LifecyclePlan:
    try:
        raw = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise error_type(f"invalid lifecycle plan JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise error_type("invalid lifecycle plan JSON: expected object")
    try:
        return LifecyclePlan.from_dict(raw)
    except (LifecycleContractError, TypeError, ValueError) as exc:
        raise error_type(f"invalid lifecycle plan: {exc}") from exc


def canonical_object_json(value: Any, *, field: str, error_type: type[Exception] = ValueError) -> str:
    try:
        decoded = json.loads(str(value or ""))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise error_type(f"invalid outbox {field} JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise error_type(f"invalid outbox {field} JSON: expected object")
    return json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def transition_allowed(current: ExecutionStage, target: ExecutionStage) -> bool:
    current_order = _STAGE_ORDER.get(current)
    target_order = _STAGE_ORDER.get(target)
    if current_order is None or target_order is None:
        return False
    return target_order == current_order or target_order == current_order + 1
