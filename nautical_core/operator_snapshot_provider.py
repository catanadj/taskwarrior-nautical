"""Invocation-scoped authoritative snapshot provider for operator clients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest
from .taskwarrior_uow import TaskwarriorUnitOfWork


@dataclass(slots=True)
class OperatorSnapshotProvider:
    """Own one canonical chain snapshot reader for a validated invocation."""

    service: ChainSnapshotService

    @classmethod
    def for_unit_of_work(cls, unit_of_work: TaskwarriorUnitOfWork) -> "OperatorSnapshotProvider":
        configuration = unit_of_work.context.configuration
        return cls(ChainSnapshotService(
            unit_of_work,
            configuration_fingerprint=configuration.fingerprint,
        ))

    def collect(self, request: IntegritySnapshotRequest) -> Any:
        """Read one exact scope without broadening or reconstructing the reader."""
        return self.service.collect(request)


__all__ = ["OperatorSnapshotProvider"]
