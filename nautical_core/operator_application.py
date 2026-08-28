"""Pure authorization boundary for operator plan application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable
from collections.abc import Mapping

from .operator_models import OperatorContractError, OperatorOperation, OperatorRequest, OperatorResult
from .operator_plans import OperatorPlan
from .operator_domain_plans import DomainApplicationAuthorization


@dataclass(frozen=True, slots=True)
class ApplicationAuthorization:
    """Validated hand-off from planning to an effect owner."""

    plan: OperatorPlan
    request: OperatorRequest

    def __post_init__(self) -> None:
        if not isinstance(self.plan, OperatorPlan):
            raise OperatorContractError("application requires an OperatorPlan")
        if not isinstance(self.request, OperatorRequest):
            raise OperatorContractError("application requires an OperatorRequest")
        if not self.request.apply:
            raise OperatorContractError("read-only requests cannot reach the application boundary")
        self.plan.validate_for_request(self.request)
        if self.plan.is_noop:
            raise OperatorContractError("application requires an effectful plan")


@dataclass(frozen=True, slots=True)
class ApplicationReceipt:
    """Evidence returned after an authorized effect and its verification."""

    plan_fingerprint: str
    result: OperatorResult
    verified: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.result, OperatorResult):
            raise OperatorContractError("application receipt requires an OperatorResult")
        if not str(self.plan_fingerprint or "").strip():
            raise OperatorContractError("application receipt requires a plan fingerprint")
        if not isinstance(self.verified, bool) or not self.verified:
            raise OperatorContractError("application receipt must represent verified evidence")

    def to_dict(self) -> dict[str, object]:
        return {"plan_fingerprint": self.plan_fingerprint, "verified": self.verified, "result": self.result.to_dict()}


@runtime_checkable
class OperatorEffectOwner(Protocol):
    """Typed owner that performs an already-authorized effect."""

    def apply(self, authorization: ApplicationAuthorization) -> OperatorResult: ...


@runtime_checkable
class DomainEffectOwner(Protocol):
    """Typed owner for direct lifecycle or integrity domain plans."""

    def apply(self, authorization: DomainApplicationAuthorization) -> OperatorResult: ...


@runtime_checkable
class OperatorGuardVerifier(Protocol):
    """Owner of authoritative pre-mutation guard checks."""

    def verify(self, authorization: ApplicationAuthorization) -> None: ...


class MappingGuardVerifier:
    """Compare plan guards with a freshly acquired immutable evidence mapping."""

    def __init__(self, evidence: Callable[[], Mapping[str, object]]) -> None:
        self._evidence = evidence

    def verify(self, authorization: ApplicationAuthorization) -> None:
        current = self._evidence()
        if not isinstance(current, Mapping):
            raise OperatorContractError("guard evidence provider returned a non-object")
        expected = authorization.plan.expected_guards
        mismatches = [key for key, value in expected.items() if current.get(key) != value]
        if mismatches:
            names = ", ".join(sorted(str(key) for key in mismatches))
            raise OperatorContractError(f"application guards are stale: {names}")


class MappingPostconditionVerifier:
    """Compare expected plan postconditions with fresh external evidence."""

    def __init__(self, evidence: Callable[[], Mapping[str, object]]) -> None:
        self._evidence = evidence

    def verify(self, authorization: ApplicationAuthorization, result: OperatorResult) -> None:
        current = self._evidence()
        if not isinstance(current, Mapping):
            raise OperatorContractError("postcondition evidence provider returned a non-object")
        expected = authorization.plan.expected_postconditions
        mismatches = [key for key, value in expected.items() if current.get(key) != value]
        if mismatches:
            names = ", ".join(sorted(str(key) for key in mismatches))
            raise OperatorContractError(f"application postconditions not met: {names}")


@runtime_checkable
class OperatorPostconditionVerifier(Protocol):
    """Owner of authoritative post-mutation verification."""

    def verify(self, authorization: ApplicationAuthorization, result: OperatorResult) -> None: ...


class OperatorApplicationRegistry:
    """Resolve each plan action to exactly one effect owner."""

    def __init__(self, owners: Mapping[str, OperatorEffectOwner] | None = None) -> None:
        self._owners: dict[str, OperatorEffectOwner] = {}
        for action, owner in (owners or {}).items():
            self.register(action, owner)

    def register(self, action: str, owner: OperatorEffectOwner) -> None:
        key = str(action or "").strip()
        if not key:
            raise OperatorContractError("application owner action is required")
        if key in self._owners:
            raise OperatorContractError(f"application owner already registered for {key!r}")
        if not isinstance(owner, OperatorEffectOwner):
            raise OperatorContractError("application owner must provide apply()")
        self._owners[key] = owner

    def owner_for(self, plan: OperatorPlan) -> OperatorEffectOwner:
        if not isinstance(plan, OperatorPlan):
            raise OperatorContractError("application owner lookup requires an OperatorPlan")
        try:
            return self._owners[plan.action]
        except KeyError as exc:
            raise OperatorContractError(f"no application owner registered for {plan.action!r}") from exc

    def apply(
        self,
        plan: OperatorPlan,
        request: OperatorRequest,
        verifier: OperatorGuardVerifier,
        postcondition: OperatorPostconditionVerifier,
    ) -> OperatorResult:
        """Apply through the registered owner and both verification gates."""
        return apply_authorized(plan, request, verifier, self.owner_for(plan), postcondition)

    def apply_with_receipt(
        self,
        plan: OperatorPlan,
        request: OperatorRequest,
        verifier: OperatorGuardVerifier,
        postcondition: OperatorPostconditionVerifier,
    ) -> ApplicationReceipt:
        result = self.apply(plan, request, verifier, postcondition)
        return ApplicationReceipt(plan.fingerprint, result)


class DomainApplicationRegistry:
    """Resolve and invoke direct domain owners by explicit operation name."""

    def __init__(self, owners: Mapping[str, DomainEffectOwner] | None = None) -> None:
        self._owners: dict[str, DomainEffectOwner] = {}
        for operation, owner in (owners or {}).items():
            self.register(operation, owner)

    def register(self, operation: str, owner: DomainEffectOwner) -> None:
        key = str(operation or "").strip()
        if not key:
            raise OperatorContractError("domain owner operation is required")
        if key in self._owners:
            raise OperatorContractError(f"domain owner already registered for {key!r}")
        if not isinstance(owner, DomainEffectOwner):
            raise OperatorContractError("domain owner must provide apply()")
        self._owners[key] = owner

    def apply(self, operation: str, authorization: DomainApplicationAuthorization) -> OperatorResult:
        key = str(operation or "").strip()
        try:
            owner = self._owners[key]
        except KeyError as exc:
            raise OperatorContractError(f"no domain owner registered for {key!r}") from exc
        return owner.apply(authorization)


def authorize_application(plan: OperatorPlan, request: OperatorRequest) -> ApplicationAuthorization:
    """Validate the explicit apply hand-off without invoking any effect."""
    return ApplicationAuthorization(plan, request)


def apply_authorized(
    plan: OperatorPlan,
    request: OperatorRequest,
    verifier: OperatorGuardVerifier,
    owner: OperatorEffectOwner,
    postcondition: OperatorPostconditionVerifier,
) -> OperatorResult:
    """Authorize, guard, delegate, and verify before reporting an effect."""
    authorization = authorize_application(plan, request)
    if not isinstance(verifier, OperatorGuardVerifier):
        raise OperatorContractError("application guard verifier must provide verify()")
    verifier.verify(authorization)
    if not isinstance(owner, OperatorEffectOwner):
        raise OperatorContractError("application owner must provide apply()")
    result = owner.apply(authorization)
    if not isinstance(result, OperatorResult):
        raise OperatorContractError("application owner returned an untyped result")
    if result.operation not in {request.operation, OperatorOperation.APPLY}:
        raise OperatorContractError("application owner returned a result for a different operation")
    if not isinstance(postcondition, OperatorPostconditionVerifier):
        raise OperatorContractError("postcondition verifier must provide verify()")
    postcondition.verify(authorization, result)
    return result


__all__ = [
    "ApplicationAuthorization", "OperatorEffectOwner", "DomainEffectOwner", "OperatorGuardVerifier",
    "OperatorPostconditionVerifier", "OperatorApplicationRegistry",
    "DomainApplicationRegistry", "MappingGuardVerifier", "ApplicationReceipt",
    "MappingPostconditionVerifier",
    "apply_authorized", "authorize_application",
]
