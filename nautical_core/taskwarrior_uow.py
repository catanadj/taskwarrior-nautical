"""Invocation-scoped composition for Taskwarrior integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import os
from pathlib import Path
from typing import Any, Mapping

from .integration_context import DiagnosticEvent, DiagnosticsSink, IntegrationContext
from .integration_models import CommandFailureKind
from .taskwarrior_client import CommandObservation, TaskwarriorClient


class QueryScopeKind(str, Enum):
    BROAD = "broad"
    UUID = "uuid"
    CHAIN = "chain"
    CHILD_SLOT = "child_slot"
    PREDECESSOR = "predecessor"
    HISTORY = "history"
    VERIFICATION = "verification"


@dataclass(frozen=True, slots=True)
class QueryScope:
    kind: QueryScopeKind
    identity: str
    statuses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            kind = QueryScopeKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Taskwarrior query scope") from exc
        identity = str(self.identity or "").strip()
        if not identity:
            raise ValueError("Taskwarrior query scope requires an identity")
        statuses = tuple(sorted({str(item).strip().lower() for item in self.statuses if str(item).strip()}))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "statuses", statuses)


@dataclass(frozen=True, slots=True)
class SnapshotProvenance:
    scope: QueryScope
    mutation_epoch: int
    command_count: int


@dataclass(frozen=True, slots=True)
class CachedAuthoritativeRead:
    value: Any
    provenance: SnapshotProvenance


@dataclass(slots=True)
class InvocationReadCache:
    """Authoritative reads valid only within one mutation epoch."""

    _entries: dict[QueryScope, CachedAuthoritativeRead] = field(default_factory=dict)

    def get(self, scope: QueryScope, *, mutation_epoch: int) -> CachedAuthoritativeRead | None:
        entry = self._entries.get(scope)
        if entry is None or entry.provenance.mutation_epoch != mutation_epoch:
            return None
        return entry

    def put(
        self,
        scope: QueryScope,
        value: Any,
        *,
        mutation_epoch: int,
        command_count: int,
    ) -> CachedAuthoritativeRead:
        entry = CachedAuthoritativeRead(
            value,
            SnapshotProvenance(scope, mutation_epoch, command_count),
        )
        self._entries[scope] = entry
        return entry

    def invalidate(self, affected: tuple[QueryScope, ...] = ()) -> None:
        if not affected:
            self._entries.clear()
            return
        affected_set = set(affected)
        for scope in tuple(self._entries):
            if scope in affected_set:
                self._entries.pop(scope, None)

    @property
    def size(self) -> int:
        return len(self._entries)


@dataclass(frozen=True, slots=True)
class OutboxBinding:
    """Invocation-local handle to durable outbox ownership in Taskdata."""

    taskdata: Path

    def __post_init__(self) -> None:
        taskdata = Path(self.taskdata)
        if not taskdata.is_absolute():
            raise ValueError("outbox Taskdata path must be absolute")
        object.__setattr__(self, "taskdata", taskdata)


@dataclass(slots=True)
class CommandLedger:
    """Content-free per-invocation command metrics and budget observation."""

    context: IntegrationContext
    calls: int = 0
    attempts: int = 0
    duration: float = 0.0
    failures: int = 0
    by_purpose: dict[str, int] = field(default_factory=dict)
    budget_exceeded: bool = False
    _budget_reported: bool = False

    def observe(self, observation: CommandObservation) -> None:
        self.attempts += 1
        self.duration += observation.duration
        if observation.attempt == 1:
            self.calls += 1
            self.by_purpose[observation.purpose] = self.by_purpose.get(observation.purpose, 0) + 1
        if observation.kind is not CommandFailureKind.SUCCESS:
            self.failures += 1
        if self.calls > self.context.command_budget:
            self.budget_exceeded = True
            if not self._budget_reported:
                self.context.diagnostics.emit(DiagnosticEvent(
                    "command_budget",
                    f"invocation exceeded advisory command budget {self.context.command_budget}",
                ))
                self._budget_reported = True


@dataclass(slots=True)
class TaskwarriorUnitOfWork:
    """All mutable Taskwarrior integration state for one invocation."""

    context: IntegrationContext
    client: TaskwarriorClient
    reads: InvocationReadCache
    outbox: OutboxBinding
    commands: CommandLedger
    mutation_epoch: int = 0

    @classmethod
    def create(
        cls,
        context: IntegrationContext,
        *,
        env: Mapping[str, str] | None = None,
    ) -> "TaskwarriorUnitOfWork":
        if not isinstance(context, IntegrationContext):
            raise TypeError("Taskwarrior unit of work requires an IntegrationContext")
        ledger = CommandLedger(context)
        client = TaskwarriorClient(
            context.command_prefix,
            env=dict(os.environ if env is None else env),
            observer=ledger,
        )
        return cls(
            context=context,
            client=client,
            reads=InvocationReadCache(),
            outbox=OutboxBinding(context.taskdata),
            commands=ledger,
        )

    @property
    def diagnostics(self) -> DiagnosticsSink:
        return self.context.diagnostics

    def cached_read(self, scope: QueryScope) -> CachedAuthoritativeRead | None:
        return self.reads.get(scope, mutation_epoch=self.mutation_epoch)

    def cache_read(self, scope: QueryScope, value: Any) -> CachedAuthoritativeRead:
        return self.reads.put(
            scope,
            value,
            mutation_epoch=self.mutation_epoch,
            command_count=self.commands.calls,
        )

    def record_mutation(
        self,
        *,
        affected: tuple[QueryScope, ...] = (),
        uncertain: bool = False,
    ) -> int:
        """Advance freshness after any applied or uncertain external mutation."""
        del uncertain
        self.mutation_epoch += 1
        self.reads.invalidate(affected)
        return self.mutation_epoch


def build_taskwarrior_uow(
    context: IntegrationContext,
    *,
    env: Mapping[str, str] | None = None,
) -> TaskwarriorUnitOfWork:
    return TaskwarriorUnitOfWork.create(context, env=env)


__all__ = (
    "CachedAuthoritativeRead",
    "CommandLedger",
    "InvocationReadCache",
    "OutboxBinding",
    "QueryScope",
    "QueryScopeKind",
    "SnapshotProvenance",
    "TaskwarriorUnitOfWork",
    "build_taskwarrior_uow",
)
