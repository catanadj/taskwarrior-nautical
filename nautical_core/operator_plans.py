"""Pure, evidence-bound operator planning contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Mapping

from .operator_models import CoverageKind, OperatorContractError, OperatorCoverage, OperatorRequest, OperatorScope


def _text(value: object, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OperatorContractError(f"plan {name} is required")
    return text


def _json_object(value: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Copy and validate a JSON-native mapping at the plan boundary."""
    result = dict(value)
    try:
        json.dumps(result, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise OperatorContractError(f"plan {name} must contain JSON-native values") from exc
    return result


@dataclass(frozen=True, slots=True)
class OperatorPlan:
    """Deterministic plan bound to one immutable observation basis."""

    action: str
    snapshot_id: str
    configuration_fingerprint: str
    scope: OperatorScope
    coverage: OperatorCoverage
    operations: tuple[Mapping[str, Any], ...] = ()
    immutable_inputs: Mapping[str, Any] = field(default_factory=dict)
    expected_guards: Mapping[str, Any] = field(default_factory=dict)
    expected_postconditions: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.scope, OperatorScope) or not isinstance(self.coverage, OperatorCoverage):
            raise OperatorContractError("plan scope and coverage must be typed")
        for value, name in ((self.immutable_inputs, "immutable_inputs"), (self.expected_guards, "expected_guards"), (self.expected_postconditions, "expected_postconditions")):
            if value is not None and not isinstance(value, Mapping):
                raise OperatorContractError(f"plan {name} must be an object")
        operations: list[Mapping[str, Any]] = []
        for index, operation in enumerate(self.operations):
            if not isinstance(operation, Mapping):
                raise OperatorContractError("plan operations must be objects")
            kind = str(operation.get("kind") or operation.get("action") or "").strip()
            if not kind:
                raise OperatorContractError(f"plan operation {index} requires a kind")
            operations.append(_json_object(operation, f"operation {index}"))
        object.__setattr__(self, "action", _text(self.action, "action"))
        object.__setattr__(self, "snapshot_id", _text(self.snapshot_id, "snapshot_id"))
        object.__setattr__(self, "configuration_fingerprint", _text(self.configuration_fingerprint, "configuration_fingerprint"))
        object.__setattr__(self, "operations", tuple(operations))
        object.__setattr__(self, "immutable_inputs", _json_object(self.immutable_inputs or {}, "immutable_inputs"))
        object.__setattr__(self, "expected_guards", _json_object(self.expected_guards or {}, "expected_guards"))
        object.__setattr__(self, "expected_postconditions", _json_object(self.expected_postconditions or {}, "expected_postconditions"))
        if self.coverage.snapshot_id and self.coverage.snapshot_id != self.snapshot_id:
            raise OperatorContractError("plan coverage belongs to a different snapshot")
        object.__setattr__(self, "reason", str(self.reason or "").strip())
        if self.action in {"apply", "repair", "mutate"} and self.coverage.kind is not CoverageKind.COMPLETE:
            raise OperatorContractError("effectful plan requires complete coverage")

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "snapshot_id": self.snapshot_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "scope": self.scope.to_dict(),
            "coverage": self.coverage.to_dict(),
            "operations": [dict(item) for item in self.operations],
            "immutable_inputs": dict(self.immutable_inputs),
            "expected_guards": dict(self.expected_guards),
            "expected_postconditions": dict(self.expected_postconditions),
            "reason": self.reason or None,
        }

    @property
    def fingerprint(self) -> str:
        """Return a stable identity for deterministic replay and comparison."""
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return "op1-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]

    def validate_for_request(self, request: OperatorRequest) -> None:
        """Reject applying a plan against a different validated request basis."""
        if not isinstance(request, OperatorRequest):
            raise OperatorContractError("plan validation requires an OperatorRequest")
        if request.scope != self.scope:
            raise OperatorContractError("plan scope differs from operator request")
        if not request.coverage.accepts(self.coverage):
            raise OperatorContractError("plan coverage does not satisfy operator request")
        if request.apply and self.action not in {"apply", "repair", "mutate"}:
            raise OperatorContractError("apply request cannot use a non-effectful plan")

    @property
    def is_noop(self) -> bool:
        """Whether this plan intentionally performs no operations."""
        return self.action in {"noop", "already_applied", "terminal"} or not self.operations

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorPlan":
        if not isinstance(value, Mapping):
            raise OperatorContractError("plan must be an object")
        operations = value.get("operations", ())
        if isinstance(operations, (str, bytes)) or not isinstance(operations, (list, tuple)):
            raise OperatorContractError("plan operations must be a list")
        return cls(
            action=value.get("action", ""),
            snapshot_id=value.get("snapshot_id", ""),
            configuration_fingerprint=value.get("configuration_fingerprint", ""),
            scope=OperatorScope.from_mapping(value.get("scope")),
            coverage=OperatorCoverage.from_mapping(value.get("coverage")),
            operations=tuple(operations),
            immutable_inputs=value.get("immutable_inputs", {}) or {},
            expected_guards=value.get("expected_guards", {}) or {},
            expected_postconditions=value.get("expected_postconditions", {}) or {},
            reason=value.get("reason", "") or "",
        )


__all__ = ["OperatorPlan"]
