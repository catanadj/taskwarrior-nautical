"""Immutable, invocation-scoped context for the hook workflow engine.

The context owns one clock sample and bounded caches.  It is intentionally
independent of hook composition roots so consumers can migrate without
retaining module-level state.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol
from .integration_context import IntegrationContext


class BusinessCalendar(Protocol):
    name: str

    def is_business_day(self, value) -> bool: ...


@dataclass(frozen=True, slots=True)
class SnapshotLease:
    """Identity and mutation epoch for evidence valid during one invocation."""

    source_identity: str
    mutation_epoch: int = 0

    def __post_init__(self) -> None:
        source = str(self.source_identity or "").strip()
        if not source:
            raise ValueError("snapshot lease requires a source identity")
        if isinstance(self.mutation_epoch, bool) or not isinstance(self.mutation_epoch, int) or self.mutation_epoch < 0:
            raise ValueError("snapshot lease mutation epoch must be non-negative")
        object.__setattr__(self, "source_identity", source)


class InvocationCache:
    """Small bounded cache that is cleared when an invocation closes."""

    __slots__ = ("_max_entries", "_values")

    def __init__(self, max_entries: int = 128) -> None:
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("invocation cache max_entries must be positive")
        self._max_entries = max_entries
        self._values: OrderedDict[object, object] = OrderedDict()

    def get(self, key: object, default: object = None) -> object:
        if key not in self._values:
            return default
        value = self._values.pop(key)
        self._values[key] = value
        return value

    def put(self, key: object, value: object) -> None:
        self._values.pop(key, None)
        self._values[key] = value
        while len(self._values) > self._max_entries:
            self._values.popitem(last=False)

    def clear(self) -> None:
        self._values.clear()

    def __len__(self) -> int:
        return len(self._values)


@dataclass(slots=True)
class InvocationCaches:
    """Named caches kept local to one workflow invocation."""

    max_entries: int = 128
    _stores: dict[str, InvocationCache] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def store(self, name: str) -> InvocationCache:
        if self._closed:
            raise RuntimeError("invocation caches are closed")
        key = str(name or "").strip()
        if not key:
            raise ValueError("invocation cache name is required")
        cache = self._stores.get(key)
        if cache is None:
            cache = InvocationCache(self.max_entries)
            self._stores[key] = cache
        return cache

    def clear(self) -> None:
        for cache in self._stores.values():
            cache.clear()
        self._stores.clear()
        self._closed = True

    def names(self) -> tuple[str, ...]:
        return tuple(self._stores)


@dataclass(frozen=True, slots=True)
class WorkflowInvocationContext:
    """One immutable clock/configuration/evidence scope for a hook call."""

    integration: IntegrationContext
    now_utc: datetime
    now_local: datetime
    configuration_lease: SnapshotLease
    task_lease: SnapshotLease
    business_calendar: BusinessCalendar | None = None
    caches: InvocationCaches = field(default_factory=InvocationCaches, compare=False)
    repository: object | None = field(default=None, compare=False)
    scheduler_session: object | None = field(default=None, compare=False)
    lifecycle_application: object | None = field(default=None, compare=False)
    closed: bool = field(default=False, init=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.integration, IntegrationContext):
            raise TypeError("workflow context requires an IntegrationContext")
        if self.now_utc.tzinfo is None or self.now_local.tzinfo is None:
            raise ValueError("workflow context clock values must be timezone-aware")
        utc = self.now_utc.astimezone(timezone.utc)
        local = utc.astimezone(self.integration.local_timezone)
        object.__setattr__(self, "now_utc", utc)
        object.__setattr__(self, "now_local", local)
        if not isinstance(self.configuration_lease, SnapshotLease) or not isinstance(self.task_lease, SnapshotLease):
            raise TypeError("workflow context leases are required")
        if self.business_calendar is not None and not callable(
            getattr(self.business_calendar, "is_business_day", None)
        ):
            raise TypeError("workflow business calendar is invalid")

    @classmethod
    def capture(
        cls,
        integration: IntegrationContext,
        *,
        configuration_lease: SnapshotLease,
        task_lease: SnapshotLease,
        business_calendar: BusinessCalendar | None = None,
        caches: InvocationCaches | None = None,
        repository: object | None = None,
        scheduler_session: object | None = None,
        lifecycle_application: object | None = None,
    ) -> "WorkflowInvocationContext":
        """Sample the clock exactly once and derive the local timestamp."""
        now_utc = integration.clock.now_utc()
        return cls(
            integration=integration,
            now_utc=now_utc,
            now_local=now_utc.astimezone(integration.local_timezone),
            configuration_lease=configuration_lease,
            task_lease=task_lease,
            business_calendar=business_calendar,
            caches=caches or InvocationCaches(),
            repository=repository,
            scheduler_session=scheduler_session,
            lifecycle_application=lifecycle_application,
        )

    @property
    def business_calendar_name(self) -> str:
        return str(getattr(self.business_calendar, "name", "") or "").strip()

    def close(self) -> None:
        """Clear invocation-local evidence and make the context unusable."""
        self.caches.clear()
        object.__setattr__(self, "closed", True)

    def __enter__(self) -> "WorkflowInvocationContext":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()
