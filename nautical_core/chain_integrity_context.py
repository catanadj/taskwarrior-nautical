"""Separated evidence sources used by chain integrity rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Mapping, Sequence

from .chain_graph import ChainGraph
from .lifecycle_outbox import LifecycleOutboxRecord


class OutboxCoverage(str, Enum):
    COMPLETE = "complete"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class OutboxSnapshot:
    """Immutable lifecycle-intent evidence, kept separate from task truth."""

    snapshot_id: str
    coverage: OutboxCoverage
    source: str
    records: tuple[LifecycleOutboxRecord, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        snapshot_id = str(self.snapshot_id or "").strip()
        source = str(self.source or "").strip()
        if not snapshot_id or not source:
            raise ValueError("outbox snapshot requires identity and source")
        coverage = OutboxCoverage(self.coverage)
        records = tuple(self.records)
        if any(not isinstance(record, LifecycleOutboxRecord) for record in records):
            raise TypeError("outbox snapshot records must be LifecycleOutboxRecord values")
        if len({record.intent_id for record in records}) != len(records):
            raise ValueError("outbox snapshot intent IDs must be unique")
        reason = str(self.reason or "").strip()
        if coverage is OutboxCoverage.UNAVAILABLE and not reason:
            raise ValueError("unavailable outbox snapshot requires a reason")
        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "records", tuple(sorted(records, key=lambda record: record.intent_id)))
        object.__setattr__(self, "reason", reason)

    @classmethod
    def from_records(cls, records: Sequence[LifecycleOutboxRecord], *, source: str = "sqlite.outbox") -> "OutboxSnapshot":
        ordered = tuple(sorted(records, key=lambda record: record.intent_id))
        payload = tuple(
            {
                "intent_id": record.intent_id,
                "state": record.state.value,
                "stage": record.stage.value,
                "configuration": record.configuration_fingerprint,
                "schedule": record.schedule_fingerprint,
                "plan": record.plan.to_dict(),
            }
            for record in ordered
        )
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()
        return cls("ois1-" + hashlib.sha256(encoded).hexdigest()[:24], OutboxCoverage.COMPLETE, source, ordered)

    @classmethod
    def unavailable(cls, reason: str, *, source: str = "sqlite.outbox") -> "OutboxSnapshot":
        digest = hashlib.sha256(str(reason).encode("utf-8")).hexdigest()[:24]
        return cls("ois1-unavailable-" + digest, OutboxCoverage.UNAVAILABLE, source, (), reason)

    def by_intent(self, intent_id: str) -> LifecycleOutboxRecord | None:
        wanted = str(intent_id or "").strip()
        for record in self.records:
            if record.intent_id == wanted:
                return record
        return None

    def for_chain(self, chain_id: str) -> tuple[LifecycleOutboxRecord, ...]:
        wanted = str(chain_id or "").strip()
        return tuple(record for record in self.records if record.plan.identity.chain_id == wanted)


@dataclass(frozen=True, slots=True)
class IntegrityContext:
    """One audit view combining independent, provenance-bearing evidence."""

    graph: ChainGraph
    outbox: OutboxSnapshot
    configuration_fingerprint: str = ""
    mutation_epoch: int = 0
    metadata: Mapping[str, str] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.graph, ChainGraph):
            raise TypeError("integrity context requires a ChainGraph")
        if not isinstance(self.outbox, OutboxSnapshot):
            raise TypeError("integrity context requires an OutboxSnapshot")
        fingerprint = str(self.configuration_fingerprint or "").strip()
        if isinstance(self.mutation_epoch, bool) or not isinstance(self.mutation_epoch, int) or self.mutation_epoch < 0:
            raise ValueError("integrity context mutation epoch must be non-negative")
        object.__setattr__(self, "configuration_fingerprint", fingerprint)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def snapshot_id(self) -> str:
        return self.graph.snapshot.snapshot_id

    @property
    def outbox_available(self) -> bool:
        return self.outbox.coverage is OutboxCoverage.COMPLETE


__all__ = ["IntegrityContext", "OutboxCoverage", "OutboxSnapshot"]
