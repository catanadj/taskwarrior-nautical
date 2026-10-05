"""Narrow claim/lease CAS port for lifecycle execution owners."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .outbox import LifecycleOutboxRecord, OutboxResult
from .outbox_operations import LifecycleExecutionOutboxPort
from .models import ExecutionStage


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
    """Repository adapter exposing only claim/lease operations."""

    def __init__(self, repository: LifecycleExecutionOutboxPort) -> None:
        self._repository = repository

    def claim_batch(
        self, *, owner: str, lease_seconds: float, limit: int
    ) -> tuple[OutboxResult, tuple[LifecycleOutboxRecord, ...]]:
        return self._repository.claim_batch(
            owner=owner, lease_seconds=lease_seconds, limit=limit
        )

    def claim_intents(
        self, *, intent_ids: Sequence[str], owner: str, lease_seconds: float
    ) -> tuple[OutboxResult, Mapping[str, OutboxResult]]:
        return self._repository.claim_intents(
            intent_ids=intent_ids, owner=owner, lease_seconds=lease_seconds
        )

    def renew_lease(
        self, *, intent_id: str, owner: str, lease_seconds: float
    ) -> OutboxResult:
        return self._repository.renew_lease(
            intent_id=intent_id, owner=owner, lease_seconds=lease_seconds
        )

    def renew_leases(
        self, *, intent_ids: Sequence[str], owner: str, lease_seconds: float
    ) -> tuple[OutboxResult, Mapping[str, OutboxResult]]:
        return self._repository.renew_leases(
            intent_ids=intent_ids, owner=owner, lease_seconds=lease_seconds
        )

    def advance_stages(
        self, *, stages: Mapping[str, ExecutionStage], owner: str
    ) -> tuple[OutboxResult, Mapping[str, OutboxResult]]:
        return self._repository.advance_stages(stages=stages, owner=owner)
