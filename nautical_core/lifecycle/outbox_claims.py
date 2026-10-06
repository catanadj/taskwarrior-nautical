"""Narrow claim/lease CAS port for lifecycle execution owners."""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence

from .outbox import LifecycleOutboxRecord, OutboxResult
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
