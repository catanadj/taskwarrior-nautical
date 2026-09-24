"""Narrow claim/lease CAS port for lifecycle execution owners."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .lifecycle_outbox import LifecycleOutboxRecord, OutboxResult
from .lifecycle_outbox_operations import LifecycleExecutionOutboxPort
from .lifecycle_models import ExecutionStage


class LifecycleOutboxClaimPort(Protocol):
    def claim_batch(
        self, *, owner: str, lease_seconds: float, limit: int
    ) -> tuple[OutboxResult, tuple[LifecycleOutboxRecord, ...]]: ...

    def claim_intents(
        self, *, intent_ids: Sequence[str], owner: str, lease_seconds: float
    ) -> tuple[OutboxResult, Mapping[str, OutboxResult]]: ...

    def renew_lease(self, *, intent_id: str, owner: str, lease_seconds: float) -> OutboxResult: ...

    def renew_leases(
        self, *, intent_ids: Sequence[str], owner: str, lease_seconds: float
    ) -> tuple[OutboxResult, Mapping[str, OutboxResult]]: ...

    def advance_stages(
        self, *, stages: Mapping[str, ExecutionStage], owner: str
    ) -> tuple[OutboxResult, Mapping[str, OutboxResult]]: ...


class RepositoryClaimLeasePort:
    """Compatibility adapter exposing only repository claim/lease operations."""

    def __init__(self, repository: LifecycleExecutionOutboxPort) -> None:
        self._repository = repository

    def claim_batch(self, **kwargs: Any) -> tuple[OutboxResult, tuple[LifecycleOutboxRecord, ...]]:
        return self._repository.claim_batch(**kwargs)

    def claim_intents(self, **kwargs: Any) -> tuple[OutboxResult, Mapping[str, OutboxResult]]:
        return self._repository.claim_intents(**kwargs)

    def renew_lease(self, **kwargs: Any) -> OutboxResult:
        return self._repository.renew_lease(**kwargs)

    def renew_leases(self, **kwargs: Any) -> tuple[OutboxResult, Mapping[str, OutboxResult]]:
        return self._repository.renew_leases(**kwargs)

    def advance_stages(self, **kwargs: Any) -> tuple[OutboxResult, Mapping[str, OutboxResult]]:
        return self._repository.advance_stages(**kwargs)
