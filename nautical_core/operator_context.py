"""Invocation-scoped context for the operator control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from collections import OrderedDict
from typing import Any, Protocol

from .integration_context import IntegrationContext
from .operator_models import OperatorContractError, OperatorLimits, OperatorRequest


class OperatorContextError(ValueError):
    """Raised when an operator invocation context is incomplete."""


class OperatorBudgetLedger(Protocol):
    """Narrow resource-accounting boundary for operator providers."""

    def consume(self, resource: str, amount: int = 1) -> bool: ...

    def usage(self, resource: str) -> int: ...

    def remaining(self, resource: str) -> int: ...


class OperatorOutputMode(str, Enum):
    JSON = "json"
    TEXT = "text"
    RICH = "rich"
    NAVIGATOR = "navigator"


@dataclass(frozen=True, slots=True)
class OperatorPresentationPolicy:
    """Immutable observer preferences; never consulted for domain decisions."""

    output: OperatorOutputMode = OperatorOutputMode.JSON
    diagnostics: bool = False

    def __post_init__(self) -> None:
        try:
            output = OperatorOutputMode(self.output)
        except (TypeError, ValueError) as exc:
            raise OperatorContextError("invalid operator output mode") from exc
        if not isinstance(self.diagnostics, bool):
            raise OperatorContextError("diagnostics policy must be boolean")
        object.__setattr__(self, "output", output)


@dataclass(slots=True)
class OperatorInvocationCache:
    """Bounded, invocation-local memoization with explicit reset semantics."""

    max_entries: int = 128
    _entries: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def __post_init__(self) -> None:
        if isinstance(self.max_entries, bool) or not isinstance(self.max_entries, int) or self.max_entries < 1:
            raise OperatorContextError("cache max_entries must be a positive integer")
        self._entries.clear()

    def get(self, key: str) -> Any | None:
        value = self._entries.get(str(key))
        if value is not None:
            self._entries.move_to_end(str(key))
        return value

    def put(self, key: str, value: Any) -> None:
        normalized = str(key)
        self._entries[normalized] = value
        self._entries.move_to_end(normalized)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def discard(self, key: str) -> bool:
        """Remove one projection and report whether it was present."""
        return self._entries.pop(str(key), None) is not None

    def clear_prefix(self, prefix: str) -> int:
        """Remove projections sharing a mutation-affected key prefix."""
        normalized = str(prefix)
        keys = [key for key in self._entries if key.startswith(normalized)]
        for key in keys:
            del self._entries[key]
        return len(keys)

    @property
    def size(self) -> int:
        return len(self._entries)


@dataclass(slots=True)
class OperatorInvocationBudget:
    """Invocation-local usage counters for resources with hard limits."""

    limits: OperatorLimits
    _usage: dict[str, int] = field(default_factory=dict)
    _effect_started: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.limits, OperatorLimits):
            raise OperatorContextError("operator budget requires typed limits")

    def consume(self, resource: str, amount: int = 1) -> bool:
        name = str(resource)
        try:
            limit = getattr(self.limits, name)
        except AttributeError as exc:
            raise OperatorContractError(f"unknown operator budget resource: {name}") from exc
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 1:
            raise OperatorContextError("operator budget amount must be positive")
        observed = self._usage.get(name, 0) + amount
        self._usage[name] = observed
        if self._effect_started:
            return True
        if observed > limit:
            self._usage[name] = observed - amount
            return False
        return True

    def begin_effect(self) -> None:
        """Mark the durable mutation boundary; later budget crossings cannot abort it."""
        self._effect_started = True

    @property
    def effect_started(self) -> bool:
        return self._effect_started

    def exceeded(self, resource: str) -> bool:
        name = str(resource)
        try:
            return self.usage(name) > int(getattr(self.limits, name))
        except AttributeError as exc:
            raise OperatorContractError(f"unknown operator budget resource: {name}") from exc

    def usage(self, resource: str) -> int:
        return self._usage.get(str(resource), 0)

    def remaining(self, resource: str) -> int:
        name = str(resource)
        try:
            return max(0, int(getattr(self.limits, name)) - self.usage(name))
        except AttributeError as exc:
            raise OperatorContractError(f"unknown operator budget resource: {name}") from exc

    def snapshot(self) -> dict[str, int]:
        return {name: self.usage(name) for name in self.limits.to_dict()}

    def observe_peak_memory(self, bytes_used: int) -> bool:
        """Record a measured peak and report whether the configured cap is exceeded."""
        if isinstance(bytes_used, bool) or not isinstance(bytes_used, int) or bytes_used < 0:
            raise OperatorContextError("peak memory observation must be a non-negative integer")
        current = self.usage("peak_memory_bytes")
        if bytes_used > current:
            self._usage["peak_memory_bytes"] = bytes_used
        return bytes_used <= self.limits.peak_memory_bytes


@dataclass(frozen=True, slots=True)
class OperatorInvocationContext:
    """Immutable shared state captured once for one operator invocation."""

    request: OperatorRequest
    integration: IntegrationContext
    captured_at: datetime
    mutation_epoch: str = "epoch-0"
    policy: OperatorPresentationPolicy = OperatorPresentationPolicy()
    cache: OperatorInvocationCache = field(default_factory=OperatorInvocationCache)
    budget: OperatorInvocationBudget | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, OperatorRequest):
            raise OperatorContextError("operator request is required")
        if not isinstance(self.integration, IntegrationContext):
            raise OperatorContextError("validated integration context is required")
        if not isinstance(self.policy, OperatorPresentationPolicy):
            raise OperatorContextError("operator presentation policy is invalid")
        if not isinstance(self.cache, OperatorInvocationCache):
            raise OperatorContextError("operator invocation cache is invalid")
        if self.budget is not None and not isinstance(self.budget, OperatorInvocationBudget):
            raise OperatorContextError("operator invocation budget is invalid")
        if self.request.apply and not self.integration.mutation_capable:
            raise OperatorContextError("apply request requires mutation-capable integration access")
        captured_at = self.captured_at
        if not isinstance(captured_at, datetime):
            raise OperatorContextError("captured_at must be a datetime")
        if captured_at.tzinfo is None or captured_at.utcoffset() is None:
            raise OperatorContextError("captured_at must be timezone-aware")
        epoch = str(self.mutation_epoch or "").strip()
        if not epoch:
            raise OperatorContextError("mutation_epoch is required")
        object.__setattr__(self, "captured_at", captured_at.astimezone(timezone.utc))
        object.__setattr__(self, "mutation_epoch", epoch)

    @classmethod
    def from_integration(
        cls,
        request: OperatorRequest,
        integration: IntegrationContext,
        *,
        captured_at: datetime | None = None,
        mutation_epoch: str = "epoch-0",
        policy: OperatorPresentationPolicy | None = None,
    ) -> "OperatorInvocationContext":
        """Bind an already-validated integration context without reloading it."""
        instant = captured_at
        if instant is None:
            instant = integration.clock.now_utc()
        return cls(
            request,
            integration,
            instant,
            mutation_epoch,
            policy or OperatorPresentationPolicy(),
            OperatorInvocationCache(max_entries=request.limits.cache_entries),
            OperatorInvocationBudget(request.limits),
        )

    @classmethod
    def from_unit_of_work(
        cls,
        request: OperatorRequest,
        unit_of_work: object,
        *,
        captured_at: datetime | None = None,
        policy: OperatorPresentationPolicy | None = None,
    ) -> "OperatorInvocationContext":
        """Bind one operator context to an existing Taskwarrior UoW."""
        integration = getattr(unit_of_work, "context", None)
        epoch = getattr(unit_of_work, "mutation_epoch", None)
        if not isinstance(integration, IntegrationContext):
            raise OperatorContextError("unit of work has no validated integration context")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, str)):
            raise OperatorContextError("unit of work has no valid mutation epoch")
        return cls.from_integration(
            request,
            integration,
            captured_at=captured_at,
            mutation_epoch=str(epoch),
            policy=policy,
        )

    @property
    def configuration_fingerprint(self) -> str:
        return self.integration.configuration.fingerprint

    @property
    def timezone_name(self) -> str:
        return self.integration.configuration.timezone_name

    @property
    def mutation_capable(self) -> bool:
        return self.integration.mutation_capable

    def assert_compatible(self, integration: IntegrationContext) -> None:
        """Reject configuration or timezone drift within this invocation."""
        if not isinstance(integration, IntegrationContext):
            raise OperatorContextError("validated integration context is required")
        if integration.configuration.fingerprint != self.configuration_fingerprint:
            raise OperatorContextError("configuration changed during operator invocation")
        if integration.configuration.timezone_name != self.timezone_name:
            raise OperatorContextError("timezone changed during operator invocation")
        if integration.local_timezone != self.integration.local_timezone:
            raise OperatorContextError("resolved timezone changed during operator invocation")
        if integration.taskdata != self.integration.taskdata:
            raise OperatorContextError("Taskdata changed during operator invocation")
        if integration.taskdata_source != self.integration.taskdata_source:
            raise OperatorContextError("Taskdata resolution source changed during operator invocation")
        if integration.command_prefix != self.integration.command_prefix:
            raise OperatorContextError("Taskwarrior command changed during operator invocation")

    def assert_epoch(self, mutation_epoch: str) -> None:
        """Reject reads or effects using a different unit-of-work epoch."""
        epoch = str(mutation_epoch or "").strip()
        if not epoch:
            raise OperatorContextError("mutation epoch is required")
        if epoch != self.mutation_epoch:
            raise OperatorContextError("mutation epoch changed during operator invocation")


__all__ = [
    "OperatorContextError", "OperatorOutputMode", "OperatorPresentationPolicy",
    "OperatorInvocationCache",
    "OperatorBudgetLedger",
    "OperatorInvocationBudget",
    "OperatorInvocationContext",
]
