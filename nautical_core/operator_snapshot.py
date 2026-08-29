"""Immutable snapshot contracts for the operator control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import hashlib
from typing import Any, Callable, Mapping, Protocol, TypeAlias, cast

from .operator_models import (
    CoverageRequirement,
    OperatorContractError,
    OperatorCoverage,
    OperatorLimits,
    OperatorFailure,
    OperatorScope,
    OperatorScopeKind,
    CoverageKind,
    _freeze_json_value,
    _json_value,
)
from .operator_context import OperatorInvocationContext
from .integration_models import Absent, Found, TaskRead, Unavailable
from .chain_integrity_models import ChainSnapshot


@dataclass(frozen=True, slots=True)
class SnapshotIndexes:
    """Stable identity indexes built once from one task snapshot."""

    task_uuids: tuple[str, ...] = ()
    chain_ids: tuple[str, ...] = ()
    links: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    recurrence_tasks: tuple[str, ...] = ()
    child_slots: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            values = tuple(dict.fromkeys(str(item).strip() for item in getattr(self, name) if str(item).strip()))
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in self.__dataclass_fields__}

    @classmethod
    def from_mapping(cls, value: object) -> "SnapshotIndexes":
        if not isinstance(value, Mapping):
            raise OperatorContractError("snapshot indexes must be an object")
        values: dict[str, tuple[str, ...]] = {}
        for name in cls.__dataclass_fields__:
            raw = value.get(name, ())
            if isinstance(raw, str) or not isinstance(raw, (list, tuple)):
                raise OperatorContractError(f"snapshot index {name} must be a list")
            values[name] = tuple(raw)
        return cls(**values)


@dataclass(frozen=True, slots=True)
class HydrationBatch:
    """Bounded set-read evidence for identities missing from a snapshot."""

    kind: str
    requested: tuple[str, ...]
    observed: tuple[str, ...] = ()
    limit: int = 100
    complete: bool = True

    def __post_init__(self) -> None:
        kind = str(self.kind or "").strip()
        if not kind:
            raise OperatorContractError("hydration kind is required")
        requested = tuple(dict.fromkeys(str(item).strip() for item in self.requested if str(item).strip()))
        observed = tuple(dict.fromkeys(str(item).strip() for item in self.observed if str(item).strip()))
        if isinstance(self.limit, bool) or not isinstance(self.limit, int) or self.limit < 1:
            raise OperatorContractError("hydration limit must be positive")
        if not isinstance(self.complete, bool):
            raise OperatorContractError("hydration complete must be boolean")
        if self.complete and not set(observed).issubset(set(requested)):
            raise OperatorContractError("complete hydration cannot contain unrequested identities")
        if self.complete and len(observed) > self.limit:
            raise OperatorContractError("complete hydration exceeds its limit")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "observed", observed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "requested": list(self.requested),
            "observed": list(self.observed),
            "limit": self.limit,
            "complete": self.complete,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "HydrationBatch":
        if not isinstance(value, Mapping):
            raise OperatorContractError("hydration batch must be an object")
        requested = value.get("requested", ())
        observed = value.get("observed", ())
        if isinstance(requested, str) or not isinstance(requested, (list, tuple)):
            raise OperatorContractError("hydration requested must be a list")
        if isinstance(observed, str) or not isinstance(observed, (list, tuple)):
            raise OperatorContractError("hydration observed must be a list")
        return cls(value.get("kind", ""), tuple(requested), tuple(observed), value.get("limit", 100), value.get("complete", True))


@dataclass(frozen=True, slots=True)
class SnapshotComponent:
    """Freshness evidence for one independently observed component."""

    name: str
    observed_at: datetime
    mutation_epoch: str
    coverage: OperatorCoverage | None = None

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        epoch = str(self.mutation_epoch or "").strip()
        if not name or not epoch:
            raise OperatorContractError("snapshot component name and mutation epoch are required")
        if not isinstance(self.observed_at, datetime) or self.observed_at.tzinfo is None or self.observed_at.utcoffset() is None:
            raise OperatorContractError("snapshot component observed_at must be timezone-aware")
        if self.coverage is not None and not isinstance(self.coverage, OperatorCoverage):
            raise OperatorContractError("snapshot component coverage is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "mutation_epoch", epoch)
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "observed_at": self.observed_at.isoformat().replace("+00:00", "Z"),
            "mutation_epoch": self.mutation_epoch,
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SnapshotComponent":
        if not isinstance(value, Mapping):
            raise OperatorContractError("snapshot component must be an object")
        raw_time = value.get("observed_at", "")
        if not isinstance(raw_time, str):
            raise OperatorContractError("snapshot component observed_at must be an RFC 3339 timestamp")
        try:
            observed_at = datetime.fromisoformat(raw_time.replace("Z", "+00:00"))
        except ValueError as exc:
            raise OperatorContractError("snapshot component observed_at must be an RFC 3339 timestamp") from exc
        raw_coverage = value.get("coverage")
        return cls(value.get("name", ""), observed_at, value.get("mutation_epoch", ""), None if raw_coverage is None else OperatorCoverage.from_mapping(raw_coverage))


@dataclass(frozen=True, slots=True)
class OperatorSnapshot:
    """One invocation-local observation basis with explicit provenance."""

    snapshot_id: str
    coverage: OperatorCoverage
    created_at: datetime
    mutation_epoch: str
    configuration_fingerprint: str
    components: Mapping[str, Any] = field(default_factory=dict)
    provider_manifest: Mapping[str, Any] = field(default_factory=dict)
    indexes: SnapshotIndexes = field(default_factory=SnapshotIndexes)
    hydration: tuple[HydrationBatch, ...] = ()
    component_evidence: tuple[SnapshotComponent, ...] = ()

    def __post_init__(self) -> None:
        snapshot_id = str(self.snapshot_id or "").strip()
        epoch = str(self.mutation_epoch or "").strip()
        fingerprint = str(self.configuration_fingerprint or "").strip()
        if not snapshot_id or not epoch or not fingerprint:
            raise OperatorContractError("snapshot identity, mutation epoch, and configuration are required")
        if not isinstance(self.coverage, OperatorCoverage):
            raise OperatorContractError("snapshot requires OperatorCoverage")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise OperatorContractError("snapshot created_at must be timezone-aware")
        if not isinstance(self.components, Mapping) or not isinstance(self.provider_manifest, Mapping):
            raise OperatorContractError("snapshot components and provider_manifest must be objects")
        if not isinstance(self.indexes, SnapshotIndexes):
            raise OperatorContractError("snapshot indexes must be SnapshotIndexes")
        hydration = tuple(self.hydration)
        if any(not isinstance(item, HydrationBatch) for item in hydration):
            raise OperatorContractError("snapshot hydration entries must be HydrationBatch")
        component_evidence = tuple(self.component_evidence)
        if any(not isinstance(item, SnapshotComponent) for item in component_evidence):
            raise OperatorContractError("snapshot component evidence entries are invalid")
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "mutation_epoch", epoch)
        object.__setattr__(self, "configuration_fingerprint", fingerprint)
        object.__setattr__(self, "created_at", self.created_at.astimezone(timezone.utc))
        object.__setattr__(self, "components", _freeze_json_value(self.components))
        object.__setattr__(self, "provider_manifest", _freeze_json_value(self.provider_manifest))
        object.__setattr__(self, "hydration", hydration)
        object.__setattr__(self, "component_evidence", component_evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "coverage": self.coverage.to_dict(),
            "created_at": self.created_at.isoformat().replace("+00:00", "Z"),
            "mutation_epoch": self.mutation_epoch,
            "configuration_fingerprint": self.configuration_fingerprint,
            "components": _json_value(self.components),
            "provider_manifest": _json_value(self.provider_manifest),
            "indexes": self.indexes.to_dict(),
            "hydration": [item.to_dict() for item in self.hydration],
            "component_evidence": [item.to_dict() for item in self.component_evidence],
        }

    def assert_consistent(self) -> None:
        """Reject component evidence captured across different mutation epochs."""
        mismatched = [
            item.name for item in self.component_evidence
            if item.mutation_epoch != self.mutation_epoch
        ]
        if mismatched:
            raise OperatorContractError(
                "snapshot contains mixed mutation epochs: " + ", ".join(sorted(mismatched))
            )

    @property
    def cacheable(self) -> bool:
        """Only authoritative or explicitly bounded evidence may be cached."""
        return self.coverage.kind.value != "unavailable"

    def assert_cacheable(self) -> None:
        if not self.cacheable:
            raise OperatorContractError("unavailable snapshot evidence must not be cached")

    def satisfies(self, requirement: CoverageRequirement) -> bool:
        """Return whether this snapshot meets a planner's evidence floor."""
        if not isinstance(requirement, CoverageRequirement):
            raise OperatorContractError("snapshot coverage check requires a typed requirement")
        return requirement.accepts(self.coverage)

    def assert_satisfies(self, requirement: CoverageRequirement) -> None:
        """Fail closed when a planner requests stronger evidence than observed."""
        if not self.satisfies(requirement):
            raise OperatorContractError(
                f"snapshot coverage {self.coverage.kind.value} does not satisfy {requirement.minimum.value}"
            )

    @classmethod
    def from_mapping(cls, value: object) -> "OperatorSnapshot":
        if not isinstance(value, Mapping):
            raise OperatorContractError("snapshot must be an object")
        raw_created = value.get("created_at", "")
        if not isinstance(raw_created, str):
            raise OperatorContractError("snapshot created_at must be an RFC 3339 timestamp")
        try:
            created_at = datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
        except ValueError as exc:
            raise OperatorContractError("snapshot created_at must be an RFC 3339 timestamp") from exc
        raw_hydration = value.get("hydration", ())
        if isinstance(raw_hydration, (str, bytes)) or not isinstance(raw_hydration, (list, tuple)):
            raise OperatorContractError("snapshot hydration must be a list")
        raw_components = value.get("component_evidence", ())
        if isinstance(raw_components, (str, bytes)) or not isinstance(raw_components, (list, tuple)):
            raise OperatorContractError("snapshot component_evidence must be a list")
        return cls(
            snapshot_id=value.get("snapshot_id", ""),
            coverage=OperatorCoverage.from_mapping(value.get("coverage")),
            created_at=created_at,
            mutation_epoch=value.get("mutation_epoch", ""),
            configuration_fingerprint=value.get("configuration_fingerprint", ""),
            components=value.get("components", {}),
            provider_manifest=value.get("provider_manifest", {}),
            indexes=SnapshotIndexes.from_mapping(value.get("indexes", {})),
            hydration=tuple(HydrationBatch.from_mapping(item) for item in raw_hydration),
            component_evidence=tuple(SnapshotComponent.from_mapping(item) for item in raw_components),
        )


@dataclass(frozen=True, slots=True)
class SnapshotReadRequest:
    """Validated read intent for constructing one authoritative snapshot."""

    scope: OperatorScope
    requirement: CoverageRequirement = field(default_factory=CoverageRequirement)
    limits: OperatorLimits = field(default_factory=OperatorLimits)
    refresh: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.scope, OperatorScope):
            raise OperatorContractError("snapshot read requires an explicit scope")
        if not isinstance(self.requirement, CoverageRequirement):
            raise OperatorContractError("snapshot read requires a coverage requirement")
        if not isinstance(self.limits, OperatorLimits):
            raise OperatorContractError("snapshot read requires typed limits")
        if not isinstance(self.refresh, bool):
            raise OperatorContractError("snapshot read refresh must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope.to_dict(),
            "requirement": self.requirement.to_dict(),
            "limits": self.limits.to_dict(),
            "refresh": self.refresh,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "SnapshotReadRequest":
        if not isinstance(value, Mapping):
            raise OperatorContractError("snapshot read request must be an object")
        return cls(
            scope=OperatorScope.from_mapping(value.get("scope")),
            requirement=CoverageRequirement.from_mapping(value.get("requirement", {})),
            limits=OperatorLimits.from_mapping(value.get("limits", {})),
            refresh=value.get("refresh", False),
        )


class SnapshotReader(Protocol):
    """Provider boundary for authoritative, bounded snapshot reads."""

    def read(
        self,
        context: OperatorInvocationContext,
        request: SnapshotReadRequest,
    ) -> "SnapshotReadResult":
        """Read one snapshot without broadening the requested scope."""
        ...


SnapshotReadResult: TypeAlias = OperatorSnapshot | OperatorFailure


class ChainSnapshotReader:
    """Adapt the existing integrity snapshot provider to operator requests."""

    def __init__(self, collector: Callable[[object], TaskRead[object]]) -> None:
        if not callable(collector):
            raise OperatorContractError("snapshot collector must be callable")
        self._collector = collector

    def read(
        self,
        context: OperatorInvocationContext,
        request: SnapshotReadRequest,
    ) -> SnapshotReadResult:
        if not isinstance(context, OperatorInvocationContext):
            raise OperatorContractError("snapshot read requires an operator context")
        if not isinstance(request, SnapshotReadRequest):
            raise OperatorContractError("snapshot read requires a typed request")
        cache_key = "operator-snapshot:" + json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
        if not request.refresh:
            cached = context.cache.get(cache_key)
            if isinstance(cached, OperatorSnapshot):
                return cached
        outcome = self.read_chain_snapshot(context, request)
        scope = request.scope
        if isinstance(outcome, OperatorFailure):
            return outcome
        snapshot = OperatorSnapshotAssembler.from_chain_snapshot(context, outcome)
        if not snapshot.satisfies(request.requirement):
            return OperatorFailure(
                "insufficient_snapshot_coverage",
                f"snapshot coverage {snapshot.coverage.kind.value} does not satisfy {request.requirement.minimum.value}",
                scope=scope,
                details={"snapshot_id": snapshot.snapshot_id, "source": snapshot.coverage.source},
            )
        observed_limits = (
            ("tasks", len(snapshot.indexes.task_uuids), request.limits.tasks),
            ("chains", len(snapshot.indexes.chain_ids), request.limits.chains),
            ("history_links", len(snapshot.indexes.links), request.limits.history_links),
        )
        for name, observed, limit in observed_limits:
            if observed > limit:
                return OperatorFailure(
                    "snapshot_limit_exceeded",
                    f"snapshot contains {observed} {name}, exceeding limit {limit}",
                    scope=scope,
                    details={"snapshot_id": snapshot.snapshot_id, "observed": observed, "limit": limit},
                )
        if snapshot.cacheable and not request.refresh:
            cache_key = "operator-snapshot:" + json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
            context.cache.put(cache_key, snapshot)
        return snapshot

    def read_chain_snapshot(
        self,
        context: OperatorInvocationContext,
        request: SnapshotReadRequest,
    ) -> ChainSnapshot | OperatorFailure:
        """Read the established chain snapshot without projecting its rows."""
        if not isinstance(context, OperatorInvocationContext):
            raise OperatorContractError("snapshot read requires an operator context")
        if not isinstance(request, SnapshotReadRequest):
            raise OperatorContractError("snapshot read requires a typed request")
        from .chain_snapshot import IntegritySnapshotRequest

        scope = request.scope
        if scope.kind in {OperatorScopeKind.CHAINS, OperatorScopeKind.UUIDS}:
            requested_identities = len(scope.values)
            if requested_identities > request.limits.hydration_identities:
                return OperatorFailure(
                    "snapshot_limit_exceeded",
                    f"snapshot requests {requested_identities} hydration identities, exceeding limit "
                    f"{request.limits.hydration_identities}",
                    scope=scope,
                    details={
                        "resource": "hydration_identities",
                        "observed": requested_identities,
                        "limit": request.limits.hydration_identities,
                    },
                )
        cache_key = "operator-source-snapshot:" + json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
        if not request.refresh:
            cached = context.cache.get(cache_key)
            if isinstance(cached, ChainSnapshot):
                return cached
        if scope.kind in {OperatorScopeKind.SYSTEM, OperatorScopeKind.ACTIVE_TASKS,
                          OperatorScopeKind.LIFECYCLE_CANDIDATES, OperatorScopeKind.INTEGRITY_CANDIDATES}:
            source_request = IntegritySnapshotRequest.candidates(
                complete_chain_history=request.requirement.minimum.value == "complete",
                refresh=request.refresh,
            )
        elif scope.kind is OperatorScopeKind.CHAIN and len(scope.values) == 1:
            source_request = IntegritySnapshotRequest.chain(
                scope.values[0],
                complete_chain_history=request.requirement.minimum.value == "complete",
                refresh=request.refresh,
            )
        elif scope.kind is OperatorScopeKind.UUID and len(scope.values) == 1:
            source_request = IntegritySnapshotRequest.uuid(
                scope.values[0],
                complete_chain_history=request.requirement.minimum.value == "complete",
                refresh=request.refresh,
            )
        elif scope.kind in {OperatorScopeKind.CHAINS, OperatorScopeKind.UUIDS}:
            complete = request.requirement.minimum.value == "complete"
            snapshots: list[ChainSnapshot] = []
            for value in scope.values:
                source_request = (
                    IntegritySnapshotRequest.chain(value, complete_chain_history=complete, refresh=request.refresh)
                    if scope.kind is OperatorScopeKind.CHAINS
                    else IntegritySnapshotRequest.uuid(value, complete_chain_history=complete, refresh=request.refresh)
                )
                if context.budget is not None and not context.budget.consume("taskwarrior_calls"):
                    return OperatorFailure(
                        "snapshot_limit_exceeded",
                        "Taskwarrior call budget exhausted before snapshot read",
                        scope=scope,
                        details={
                            "resource": "taskwarrior_calls",
                            "observed": context.budget.usage("taskwarrior_calls"),
                            "limit": request.limits.taskwarrior_calls,
                        },
                    )
                outcome = self._collector(source_request)
                if isinstance(outcome, Unavailable):
                    return OperatorFailure("snapshot_unavailable", outcome.evidence.detail, retryable=outcome.retryable, scope=scope)
                if isinstance(outcome, Absent):
                    continue
                if not isinstance(outcome, Found) or not isinstance(outcome.value, ChainSnapshot):
                    return OperatorFailure("invalid_snapshot_read", "snapshot provider returned an invalid result", scope=scope)
                snapshot = outcome.value
                identity_rows = tuple(
                    row for row in snapshot.rows
                    if (row.chain_id == value if scope.kind is OperatorScopeKind.CHAINS else row.task_uuid == value)
                )
                if any(
                    (row.chain_id != value if scope.kind is OperatorScopeKind.CHAINS else row.task_uuid != value)
                    for row in snapshot.rows
                ):
                    return OperatorFailure(
                        "invalid_snapshot_scope",
                        "snapshot provider returned rows outside the requested identity",
                        scope=scope,
                        details={"identity": value, "snapshot_id": snapshot.snapshot_id},
                    )
                snapshots.append(
                    ChainSnapshot(
                        snapshot.snapshot_id,
                        snapshot.coverage,
                        snapshot.source,
                        identity_rows,
                        snapshot.configuration_fingerprint,
                        snapshot.complete_chain_history,
                        snapshot.reason,
                    )
                )
            rows = tuple(row for snapshot in snapshots for row in snapshot.rows)
            digest = hashlib.sha256(":".join(snapshot.snapshot_id for snapshot in snapshots).encode()).hexdigest()[:16]
            from .chain_integrity_models import SnapshotCoverage
            return ChainSnapshot(
                "multi-" + digest,
                SnapshotCoverage.CHAIN if scope.kind is OperatorScopeKind.CHAINS else SnapshotCoverage.NARROW,
                "taskwarrior.authoritative_multi_read",
                rows,
                context.configuration_fingerprint,
                complete,
            )
        else:
            return OperatorFailure(
                "unsupported_snapshot_scope",
                f"authoritative snapshot reader does not support scope {scope.kind.value} with {len(scope.values)} value(s)",
                scope=scope,
            )
        if context.budget is not None and not context.budget.consume("taskwarrior_calls"):
            return OperatorFailure(
                "snapshot_limit_exceeded",
                "Taskwarrior call budget exhausted before snapshot read",
                scope=scope,
                details={
                    "resource": "taskwarrior_calls",
                    "observed": context.budget.usage("taskwarrior_calls"),
                    "limit": request.limits.taskwarrior_calls,
                },
            )
        outcome = self._collector(source_request)
        if isinstance(outcome, Found):
            if not hasattr(outcome.value, "coverage"):
                return OperatorFailure("invalid_snapshot", "snapshot provider returned an invalid value", scope=scope)
            if not request.refresh:
                context.cache.put(cache_key, outcome.value)
            return cast(ChainSnapshot, outcome.value)
        if isinstance(outcome, Unavailable):
            return OperatorFailure(
                "snapshot_unavailable",
                outcome.evidence.detail or "authoritative snapshot is unavailable",
                retryable=outcome.retryable,
                scope=scope,
                details={"query": outcome.query},
            )
        if isinstance(outcome, Absent):
            return OperatorFailure("snapshot_absent", outcome.reason, scope=scope, details={"query": outcome.query})
        return OperatorFailure("invalid_snapshot_read", "snapshot provider returned an invalid result", scope=scope)


class OperatorSnapshotSession:
    """Invocation-scoped coordinator for shared snapshot reads and invalidation."""

    def __init__(self, context: OperatorInvocationContext, reader: SnapshotReader) -> None:
        if not isinstance(context, OperatorInvocationContext):
            raise OperatorContractError("snapshot session requires an operator context")
        if not callable(getattr(reader, "read", None)):
            raise OperatorContractError("snapshot session requires a snapshot reader")
        self.context = context
        self.reader = reader

    def read(self, request: SnapshotReadRequest) -> SnapshotReadResult:
        return self.reader.read(self.context, request)

    def read_many(self, requests: tuple[SnapshotReadRequest, ...]) -> tuple[SnapshotReadResult, ...]:
        """Read independent scopes without allowing one failure to suppress others."""
        if not isinstance(requests, tuple):
            raise OperatorContractError("snapshot batch requests must be a tuple")
        if any(not isinstance(request, SnapshotReadRequest) for request in requests):
            raise OperatorContractError("snapshot batch contains an invalid request")
        return tuple(self.read(request) for request in requests)

    def invalidate(self, request: SnapshotReadRequest | None = None) -> int:
        if request is None:
            before = self.context.cache.size
            self.context.cache.clear_prefix("operator-snapshot:")
            self.context.cache.clear_prefix("operator-source-snapshot:")
            return before - self.context.cache.size
        encoded = json.dumps(request.to_dict(), sort_keys=True, separators=(",", ":"))
        removed = int(self.context.cache.discard("operator-snapshot:" + encoded))
        removed += int(self.context.cache.discard("operator-source-snapshot:" + encoded))
        return removed

    def invalidate_after_mutation(
        self,
        affected: tuple[SnapshotReadRequest, ...] = (),
        *,
        certain: bool,
    ) -> int:
        """Invalidate affected projections, or all evidence when uncertain."""
        if not isinstance(certain, bool):
            raise OperatorContractError("mutation certainty must be boolean")
        if not certain:
            return self.invalidate()
        total = 0
        for request in affected:
            if not isinstance(request, SnapshotReadRequest):
                raise OperatorContractError("affected snapshot entries must be typed requests")
            total += self.invalidate(request)
        return total


class OperatorSnapshotAssembler:
    """Validate and assemble one snapshot without performing I/O or effects."""

    @staticmethod
    def assemble(context: OperatorInvocationContext, snapshot: OperatorSnapshot) -> OperatorSnapshot:
        if not isinstance(context, OperatorInvocationContext):
            raise OperatorContractError("snapshot assembly requires an operator context")
        if not isinstance(snapshot, OperatorSnapshot):
            raise OperatorContractError("snapshot assembly requires an OperatorSnapshot")
        if snapshot.configuration_fingerprint != context.configuration_fingerprint:
            raise OperatorContractError("snapshot configuration differs from invocation context")
        if snapshot.mutation_epoch != context.mutation_epoch:
            raise OperatorContractError("snapshot mutation epoch differs from invocation context")
        snapshot.assert_consistent()
        return snapshot

    @staticmethod
    def from_chain_snapshot(
        context: OperatorInvocationContext,
        chain_snapshot: object,
    ) -> OperatorSnapshot:
        """Project an established chain snapshot into the operator envelope."""
        from .chain_integrity_models import ChainSnapshot, SnapshotCoverage

        if not isinstance(chain_snapshot, ChainSnapshot):
            raise OperatorContractError("chain snapshot source is invalid")
        source_fingerprint = str(chain_snapshot.configuration_fingerprint or "").strip()
        if source_fingerprint and source_fingerprint != context.configuration_fingerprint:
            raise OperatorContractError("chain snapshot configuration differs from invocation context")
        coverage_value = chain_snapshot.coverage.value
        if chain_snapshot.coverage is SnapshotCoverage.UNAVAILABLE:
            coverage = OperatorCoverage(CoverageKind.UNAVAILABLE, chain_snapshot.source, reason=chain_snapshot.reason)
        else:
            projected_kind = (
                CoverageKind.PARTIAL
                if chain_snapshot.coverage is SnapshotCoverage.TRUNCATED
                else CoverageKind.COMPLETE if chain_snapshot.complete_chain_history else CoverageKind.BOUNDED
            )
            coverage = OperatorCoverage(
                projected_kind,
                chain_snapshot.source,
                omitted_count=0,
                snapshot_id=chain_snapshot.snapshot_id,
                mutation_epoch=context.mutation_epoch,
            )
        nodes = chain_snapshot.rows
        indexes = SnapshotIndexes(
            task_uuids=tuple(node.task_uuid for node in nodes),
            chain_ids=tuple(node.chain_id for node in nodes),
            links=tuple(str(node.link) for node in nodes if node.link is not None),
            statuses=tuple(node.status for node in nodes),
            recurrence_tasks=tuple(node.task_uuid for node in nodes if node.field("anchor") or node.field("cp")),
            child_slots=tuple(f"{node.chain_id}:{node.link}" for node in nodes if node.chain_id and node.link is not None),
        )
        snapshot = OperatorSnapshot(
            snapshot_id=chain_snapshot.snapshot_id,
            coverage=coverage,
            created_at=context.captured_at,
            mutation_epoch=context.mutation_epoch,
            configuration_fingerprint=context.configuration_fingerprint,
            components={"tasks": len(nodes), "source_coverage": coverage_value},
            provider_manifest={"source": chain_snapshot.source},
            indexes=indexes,
            component_evidence=(SnapshotComponent(
                "chain",
                context.captured_at,
                context.mutation_epoch,
                coverage,
            ),),
        )
        return OperatorSnapshotAssembler.assemble(context, snapshot)


__all__ = [
    "SnapshotIndexes", "HydrationBatch", "SnapshotComponent", "OperatorSnapshot",
    "SnapshotReadRequest", "SnapshotReadResult", "SnapshotReader", "ChainSnapshotReader", "OperatorSnapshotSession",
    "OperatorSnapshotAssembler",
]
