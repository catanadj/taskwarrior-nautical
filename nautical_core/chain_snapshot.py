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
    UUID = "uuid"


@dataclass(frozen=True, slots=True)
class IntegritySnapshotRequest:
    """A validated request for one authoritative chain observation."""

    kind: IntegritySnapshotKind
    chain_id: str = ""
    task_uuid: str = ""
    statuses: tuple[str, ...] = ALL_TASK_STATUSES
    complete_chain_history: bool = False
    refresh: bool = False

    def __post_init__(self) -> None:
        try:
            kind = IntegritySnapshotKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid integrity snapshot kind") from exc
        chain_id = str(self.chain_id or "").strip()
        uuid_value = str(self.task_uuid or "").strip()
        if kind is IntegritySnapshotKind.CHAIN and not chain_id:
            raise ValueError("chain snapshot requires a chainID")
        if kind is IntegritySnapshotKind.UUID and not uuid_value:
            raise ValueError("UUID snapshot requires a task UUID")
        statuses = tuple(sorted({str(status).strip().lower() for status in self.statuses if str(status).strip()}))
        if not statuses:
            raise ValueError("integrity snapshot requires statuses")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "task_uuid", uuid_value)
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

    @classmethod
    def uuid(
        cls,
        uuid_value: str,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
        complete_chain_history: bool = False,
        refresh: bool = False,
    ) -> "IntegritySnapshotRequest":
        return cls(IntegritySnapshotKind.UUID, task_uuid=str(uuid_value), statuses=tuple(statuses),
                   complete_chain_history=complete_chain_history, refresh=refresh)


def _snapshot_id(request: IntegritySnapshotRequest, rows: tuple[dict[str, object], ...], fingerprint: str) -> str:
    payload = {
        "request": {
            "kind": request.kind.value,
            "chain_id": request.chain_id,
            "uuid": request.task_uuid,
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
        context = getattr(unit_of_work, "context", None)
        validated = getattr(getattr(context, "configuration", None), "fingerprint", "")
        if validated and self._configuration_fingerprint and validated != self._configuration_fingerprint:
            raise ValueError("integrity snapshot configuration fingerprint differs from invocation context")
        if not self._configuration_fingerprint and validated:
            self._configuration_fingerprint = str(validated).strip()
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
        if read.value.truncated:
            return Unavailable(self._query(request), self._invalid_response(
                read.value, "authoritative chain export was truncated",
            ))
        validation_error = self._validate_rows(request, read.value)
        if validation_error:
            return Unavailable(self._query(request), self._invalid_response(read.value, validation_error))
        snapshot = self.from_rows(request, read.value.rows, source="taskwarrior.authoritative_export")
        if isinstance(snapshot, Unavailable):
            return Unavailable(self._query(request), self._invalid_response(read.value, snapshot.reason))
        self._normalized[cache_key] = snapshot
        return Found(snapshot, self._query(request))

    def from_rows(
        self,
        request: IntegritySnapshotRequest,
        rows: Sequence[dict[str, object]],
        *,
        source: str,
        coverage: SnapshotCoverage | None = None,
    ) -> ChainSnapshot | Unavailable:
        """Build a validated snapshot from already-authoritative rows."""
        raw_rows = tuple(dict(row) for row in rows)
        invalid = self._validate_mapping_rows(request, raw_rows)
        if invalid:
            return Unavailable(self._query(request), FailureEvidence(
                TaskCommand(("task", "export"), "chain snapshot rows", 0.0),
                CommandFailureKind.INVALID_RESPONSE, 1, 1, 0.0, False, invalid,
            ))
        try:
            normalized = tuple(ChainNode.from_mapping(row) for row in raw_rows)
        except (TypeError, ValueError) as exc:
            return Unavailable(self._query(request), FailureEvidence(
                TaskCommand(("task", "export"), "chain snapshot rows", 0.0),
                CommandFailureKind.INVALID_RESPONSE, 1, 1, 0.0, False, str(exc),
            ))
        return ChainSnapshot(
            _snapshot_id(request, raw_rows, self._configuration_fingerprint),
            coverage or (SnapshotCoverage.CHAIN if request.kind is IntegritySnapshotKind.CHAIN else SnapshotCoverage.CANDIDATES),
            source,
            normalized,
            self._configuration_fingerprint,
            request.complete_chain_history,
        )

    @staticmethod
    def _validate_rows(
        request: IntegritySnapshotRequest,
        snapshot: AuthoritativeTaskSnapshot,
    ) -> str:
        """Reject impossible identity evidence before graph construction.

        A duplicate full UUID cannot represent two Taskwarrior tasks.  Letting
        it reach the graph makes short-link resolution and slot repair
        ambiguous, so the complete export is unavailable instead.  Narrow
        chain reads also must not silently return rows from another chain.
        """
        seen: set[str] = set()
        allowed_statuses = frozenset(request.statuses)
        for row in snapshot.rows:
            uuid_value = str(row.get("uuid") or "").strip().lower()
            if not uuid_value:
                return "chain export contains a row without a UUID"
            if uuid_value in seen:
                return f"chain export contains duplicate full UUID {uuid_value}"
            seen.add(uuid_value)
            status = str(row.get("status") or "").strip().lower()
            if not status:
                return f"chain export row {uuid_value} has no status"
            if status not in allowed_statuses:
                return f"chain export row {uuid_value} has status {status!r} outside requested scope"
            if request.kind is IntegritySnapshotKind.CHAIN:
                chain_id = str(row.get("chainID") or row.get("chain_id") or "").strip()
                if chain_id != request.chain_id:
                    return (
                        f"chain export returned row {uuid_value} with chainID "
                        f"{chain_id or '<empty>'}, expected {request.chain_id}"
                    )
        return ""

    @classmethod
    def _validate_mapping_rows(
        cls,
        request: IntegritySnapshotRequest,
        rows: Sequence[dict[str, object]],
    ) -> str:
        class _Rows:
            def __init__(self, values: Sequence[dict[str, object]]) -> None:
                self.rows = tuple(values)
        return cls._validate_rows(request, _Rows(rows))

    def _read(self, request: IntegritySnapshotRequest) -> TaskRead[AuthoritativeTaskSnapshot]:
        if request.kind is IntegritySnapshotKind.CHAIN:
            filters: tuple[str, ...] = (f"chainID:{request.chain_id}",)
            identity = f"chain:{request.chain_id}"
        elif request.kind is IntegritySnapshotKind.UUID:
            filters = (f"uuid:{request.task_uuid}",)
            identity = f"uuid:{request.task_uuid}"
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
        return f"integrity:{request.kind.value}:{request.chain_id or request.task_uuid or 'all'}"

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
