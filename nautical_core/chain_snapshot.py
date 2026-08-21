"""Authoritative chain snapshots for the integrity engine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Protocol, Sequence

from .chain_integrity_models import ChainNode, ChainSnapshot, SnapshotCoverage
from .integration_models import (
    CommandFailureKind,
    FailureEvidence,
    Found,
    TaskRead,
    Unavailable,
)
from .task_read_repository import ALL_TASK_STATUSES, AuthoritativeTaskSnapshot


class _SnapshotRepository(Protocol):
    def broad_snapshot(
        self,
        *,
        identity: str,
        filters: Sequence[str],
        statuses: Sequence[str],
        complete_chain_history: bool = False,
        refresh: bool = False,
    ) -> TaskRead[AuthoritativeTaskSnapshot]: ...


class _SnapshotUnitOfWork(Protocol):
    repository: _SnapshotRepository
    mutation_epoch: int


class IntegritySnapshotKind(str, Enum):
    """The bounded export scopes understood by the integrity engine."""

    CANDIDATES = "candidates"
    CHAIN = "chain"


@dataclass(frozen=True, slots=True)
class IntegritySnapshotRequest:
    """A validated request for one authoritative chain observation."""

    kind: IntegritySnapshotKind
    chain_id: str = ""
    statuses: tuple[str, ...] = ALL_TASK_STATUSES
    complete_chain_history: bool = False
    refresh: bool = False

    def __post_init__(self) -> None:
        try:
            kind = IntegritySnapshotKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid integrity snapshot kind") from exc
        chain_id = str(self.chain_id or "").strip()
        if kind is IntegritySnapshotKind.CHAIN and not chain_id:
            raise ValueError("chain snapshot requires a chainID")
        statuses = tuple(sorted({str(status).strip().lower() for status in self.statuses if str(status).strip()}))
        if not statuses:
            raise ValueError("integrity snapshot requires statuses")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "complete_chain_history", bool(self.complete_chain_history))
        object.__setattr__(self, "refresh", bool(self.refresh))

    @classmethod
    def candidates(
        cls,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
        complete_chain_history: bool = False,
        refresh: bool = False,
    ) -> "IntegritySnapshotRequest":
        return cls(IntegritySnapshotKind.CANDIDATES, statuses=tuple(statuses),
                   complete_chain_history=complete_chain_history, refresh=refresh)

    @classmethod
    def chain(
        cls,
        chain_id: str,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
        complete_chain_history: bool = True,
        refresh: bool = False,
    ) -> "IntegritySnapshotRequest":
        return cls(IntegritySnapshotKind.CHAIN, chain_id=str(chain_id), statuses=tuple(statuses),
                   complete_chain_history=complete_chain_history, refresh=refresh)


def _snapshot_id(request: IntegritySnapshotRequest, rows: tuple[dict[str, object], ...], fingerprint: str) -> str:
    payload = {
        "request": {
            "kind": request.kind.value,
            "chain_id": request.chain_id,
            "statuses": request.statuses,
            "complete_chain_history": request.complete_chain_history,
        },
        "rows": rows,
        "configuration_fingerprint": fingerprint,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "cis1-" + hashlib.sha256(encoded).hexdigest()[:24]


class ChainSnapshotService:
    """Normalize authoritative Taskwarrior exports into integrity snapshots."""

    def __init__(self, unit_of_work: _SnapshotUnitOfWork, *, configuration_fingerprint: str = "") -> None:
        self._uow = unit_of_work
        self._repository = unit_of_work.repository
        self._configuration_fingerprint = str(configuration_fingerprint or "").strip()
        self._normalized: dict[tuple[int, IntegritySnapshotRequest], ChainSnapshot] = {}

    def collect(self, request: IntegritySnapshotRequest) -> TaskRead[ChainSnapshot]:
        if not isinstance(request, IntegritySnapshotRequest):
            raise TypeError("chain snapshot request must be an IntegritySnapshotRequest")
        cache_key = (self._uow.mutation_epoch, request)
        if not request.refresh and cache_key in self._normalized:
            return Found(self._normalized[cache_key], self._query(request))
        read = self._read(request)
        if isinstance(read, Unavailable):
            return read
        if not isinstance(read, Found):
            snapshot = self._empty_snapshot(request, read.reason)
            self._normalized[cache_key] = snapshot
            return Found(snapshot, self._query(request))
        try:
            rows = tuple(ChainNode.from_mapping(row) for row in read.value.rows)
        except (TypeError, ValueError) as exc:
            return Unavailable(self._query(request), self._invalid_response(read.value, str(exc)))
        raw_rows = tuple(dict(row) for row in read.value.rows)
        snapshot = ChainSnapshot(
            _snapshot_id(request, raw_rows, self._configuration_fingerprint),
            SnapshotCoverage.CHAIN if request.kind is IntegritySnapshotKind.CHAIN else SnapshotCoverage.CANDIDATES,
            "taskwarrior.authoritative_export",
            rows,
            self._configuration_fingerprint,
            request.complete_chain_history,
        )
        self._normalized[cache_key] = snapshot
        return Found(snapshot, self._query(request))

    def _read(self, request: IntegritySnapshotRequest) -> TaskRead[AuthoritativeTaskSnapshot]:
        if request.kind is IntegritySnapshotKind.CHAIN:
            filters: tuple[str, ...] = (f"chainID:{request.chain_id}",)
            identity = f"chain:{request.chain_id}"
        else:
            filters = ("chain:on",)
            identity = "chain:on"
        return self._repository.broad_snapshot(
            identity=identity,
            filters=filters,
            statuses=request.statuses,
            complete_chain_history=request.complete_chain_history,
            refresh=request.refresh,
        )

    @staticmethod
    def _query(request: IntegritySnapshotRequest) -> str:
        return f"integrity:{request.kind.value}:{request.chain_id or 'all'}"

    @staticmethod
    def _empty_snapshot(request: IntegritySnapshotRequest, reason: str) -> ChainSnapshot:
        return ChainSnapshot(
            "cis1-empty-" + hashlib.sha256(ChainSnapshotService._query(request).encode()).hexdigest()[:16],
            SnapshotCoverage.CHAIN if request.kind is IntegritySnapshotKind.CHAIN else SnapshotCoverage.CANDIDATES,
            "taskwarrior.authoritative_export",
            (),
            complete_chain_history=request.complete_chain_history,
            reason=reason,
        )

    @staticmethod
    def _invalid_response(snapshot: AuthoritativeTaskSnapshot, detail: str) -> FailureEvidence:
        result = snapshot.command_result
        return FailureEvidence(
            result.command,
            CommandFailureKind.INVALID_RESPONSE,
            result.returncode,
            result.attempt,
            result.duration,
            False,
            f"invalid chain row: {detail}",
        )
