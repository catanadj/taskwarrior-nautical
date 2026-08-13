"""Typed, invocation-scoped Taskwarrior task reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
import json
from typing import TYPE_CHECKING, Any, Mapping, Sequence, TypeAlias

from .integration_models import (
    Absent,
    CommandFailureKind,
    FailureEvidence,
    Found,
    TaskCommandResult,
    TaskRead,
    Unavailable,
)
from .taskwarrior_uow import QueryScope, QueryScopeKind

if TYPE_CHECKING:
    from .taskwarrior_uow import TaskwarriorUnitOfWork


TaskRow: TypeAlias = Mapping[str, Any]
TaskSlot: TypeAlias = tuple[str, int]
ALL_TASK_STATUSES = ("completed", "deleted", "pending", "recurring", "waiting")
ACTIVE_TASK_STATUSES = ("pending", "waiting")
_RETRYABLE_READ_FAILURES = frozenset(
    {CommandFailureKind.TIMEOUT, CommandFailureKind.BUSY, CommandFailureKind.EXECUTION_FAILURE}
)


class TaskQueryKind(str, Enum):
    BROAD = "broad"
    UUID = "uuid"
    CHILD_SLOT = "child_slot"
    PREDECESSOR_SLOT = "predecessor_slot"
    CHAIN = "chain"
    ACTIVE_ROOTS = "active_roots"
    LIFECYCLE_CANDIDATES = "lifecycle_candidates"
    VERIFICATION = "verification"


@dataclass(frozen=True, slots=True)
class TaskSnapshotScope:
    """The exact state for which an exported snapshot is authoritative."""

    kind: TaskQueryKind
    identity: str
    statuses: tuple[str, ...]
    complete_chain_history: bool = False

    def __post_init__(self) -> None:
        try:
            kind = TaskQueryKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid task snapshot scope") from exc
        identity = str(self.identity or "").strip()
        if not identity:
            raise ValueError("task snapshot scope requires an identity")
        statuses = tuple(sorted({str(item).strip().lower() for item in self.statuses if str(item).strip()}))
        if not statuses:
            raise ValueError("task snapshot scope requires included statuses")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "statuses", statuses)
        object.__setattr__(self, "complete_chain_history", bool(self.complete_chain_history))


def _link_number(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(float(str(value)))
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _append_index(index: dict[Any, list[TaskRow]], key: Any, row: TaskRow) -> None:
    if key not in (None, "", ("", None)):
        index.setdefault(key, []).append(row)


def _freeze_index(index: dict[Any, list[TaskRow]]) -> Mapping[Any, tuple[TaskRow, ...]]:
    return MappingProxyType({key: tuple(values) for key, values in index.items()})


@dataclass(frozen=True, slots=True)
class AuthoritativeTaskSnapshot:
    """One parsed export plus indexes that preserve its authority scope."""

    scope: TaskSnapshotScope
    rows: tuple[TaskRow, ...]
    command_result: TaskCommandResult
    by_uuid: Mapping[str, tuple[TaskRow, ...]] = field(init=False, repr=False)
    by_short_uuid: Mapping[str, tuple[TaskRow, ...]] = field(init=False, repr=False)
    by_chain: Mapping[str, tuple[TaskRow, ...]] = field(init=False, repr=False)
    by_slot: Mapping[TaskSlot, tuple[TaskRow, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.scope, TaskSnapshotScope):
            raise TypeError("authoritative snapshot requires a TaskSnapshotScope")
        if not isinstance(self.command_result, TaskCommandResult) or not self.command_result.ok:
            raise ValueError("authoritative snapshot requires a successful command result")
        copied_rows: list[TaskRow] = []
        uuid_index: dict[str, list[TaskRow]] = {}
        short_index: dict[str, list[TaskRow]] = {}
        chain_index: dict[str, list[TaskRow]] = {}
        slot_index: dict[TaskSlot, list[TaskRow]] = {}
        for raw_row in self.rows:
            if not isinstance(raw_row, Mapping):
                raise ValueError("authoritative snapshot contains a non-object row")
            row: TaskRow = MappingProxyType(dict(raw_row))
            copied_rows.append(row)
            uuid_value = str(row.get("uuid") or "").strip().lower()
            chain_id = str(row.get("chainID") or "").strip()
            link = _link_number(row.get("link"))
            _append_index(uuid_index, uuid_value, row)
            _append_index(short_index, uuid_value[:8], row)
            _append_index(chain_index, chain_id, row)
            if chain_id and link is not None:
                _append_index(slot_index, (chain_id, link), row)
        object.__setattr__(self, "rows", tuple(copied_rows))
        object.__setattr__(self, "by_uuid", _freeze_index(uuid_index))
        object.__setattr__(self, "by_short_uuid", _freeze_index(short_index))
        object.__setattr__(self, "by_chain", _freeze_index(chain_index))
        object.__setattr__(self, "by_slot", _freeze_index(slot_index))

    def uuid_matches(self, uuid_value: str) -> tuple[TaskRow, ...]:
        identity = str(uuid_value or "").strip().lower()
        if not identity:
            return ()
        if len(identity) >= 32 or "-" in identity:
            return self.by_uuid.get(identity, ())
        if len(identity) == 8:
            return self.by_short_uuid.get(identity, ())
        return tuple(row for row in self.rows if str(row.get("uuid") or "").lower().startswith(identity))

    def chain_rows(self, chain_id: str) -> tuple[TaskRow, ...]:
        return self.by_chain.get(str(chain_id or "").strip(), ())

    def slot_rows(self, chain_id: str, link: int) -> tuple[TaskRow, ...]:
        return self.by_slot.get((str(chain_id or "").strip(), int(link)), ())


def _query_scope(scope: TaskSnapshotScope) -> QueryScope:
    kind = {
        TaskQueryKind.BROAD: QueryScopeKind.BROAD,
        TaskQueryKind.UUID: QueryScopeKind.UUID,
        TaskQueryKind.CHILD_SLOT: QueryScopeKind.CHILD_SLOT,
        TaskQueryKind.PREDECESSOR_SLOT: QueryScopeKind.PREDECESSOR,
        TaskQueryKind.CHAIN: QueryScopeKind.CHAIN,
        TaskQueryKind.ACTIVE_ROOTS: QueryScopeKind.BROAD,
        TaskQueryKind.LIFECYCLE_CANDIDATES: QueryScopeKind.BROAD,
        TaskQueryKind.VERIFICATION: QueryScopeKind.VERIFICATION,
    }[scope.kind]
    identity = f"{scope.identity}|history={int(scope.complete_chain_history)}"
    return QueryScope(kind, identity, scope.statuses)


def _status_filter(statuses: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(status).strip().lower() for status in statuses if str(status).strip()}))
    if len(normalized) == 1:
        return (f"status:{normalized[0]}",)
    tokens: list[str] = ["("]
    for index, status in enumerate(normalized):
        if index:
            tokens.append("or")
        tokens.append(f"status:{status}")
    tokens.append(")")
    return tuple(tokens)


class TaskReadRepository:
    """The sole typed task-query boundary for one unit of work."""

    def __init__(self, unit_of_work: "TaskwarriorUnitOfWork") -> None:
        self._uow = unit_of_work
        self._snapshots: list[tuple[int, AuthoritativeTaskSnapshot]] = []

    @staticmethod
    def _query_name(scope: TaskSnapshotScope) -> str:
        return f"{scope.kind.value}:{scope.identity}"

    @staticmethod
    def _failure(
        result: TaskCommandResult,
        query: str,
        *,
        kind: CommandFailureKind | None = None,
        detail: str = "",
    ) -> Unavailable:
        failure_kind = kind or result.kind
        evidence = FailureEvidence(
            result.command,
            failure_kind,
            result.returncode,
            result.attempt,
            result.duration,
            failure_kind in _RETRYABLE_READ_FAILURES,
            detail or result.stderr or result.stdout or failure_kind.value,
        )
        return Unavailable(query, evidence)

    def _cached(self, scope: TaskSnapshotScope) -> TaskRead[AuthoritativeTaskSnapshot] | None:
        cached = self._uow.cached_read(_query_scope(scope))
        if cached is None:
            return None
        value = cached.value
        if isinstance(value, (Found, Absent, Unavailable)):
            return value
        return None

    def _store(
        self,
        scope: TaskSnapshotScope,
        read: TaskRead[AuthoritativeTaskSnapshot],
    ) -> TaskRead[AuthoritativeTaskSnapshot]:
        self._uow.cache_read(_query_scope(scope), read)
        if isinstance(read, Found):
            self._snapshots.append((self._uow.mutation_epoch, read.value))
        return read

    def _export(
        self,
        scope: TaskSnapshotScope,
        filters: Sequence[str],
        *,
        empty_output_is_absent: bool,
        refresh: bool = False,
        timeout: float = 30.0,
        attempts: int = 2,
        use_tempfiles: bool = False,
    ) -> TaskRead[AuthoritativeTaskSnapshot]:
        query = self._query_name(scope)
        if not refresh:
            cached = self._cached(scope)
            if cached is not None:
                return cached
        result = self._uow.client.execute(
            (
                "rc.hooks=off",
                "rc.json.array=1",
                "rc.verbose=nothing",
                "rc.color=off",
                *filters,
                *_status_filter(scope.statuses),
                "export",
            ),
            purpose=f"task read {scope.kind.value}",
            timeout=timeout,
            attempts=attempts,
            retry_delay=0.05,
            use_tempfiles=use_tempfiles,
        )
        if result.kind is CommandFailureKind.ABSENT and empty_output_is_absent:
            return self._store(scope, Absent(query, "Taskwarrior authoritatively returned no matches"))
        if not result.ok:
            return self._store(scope, self._failure(result, query))
        raw = result.stdout.strip()
        if not raw:
            if empty_output_is_absent:
                return self._store(scope, Absent(query, "Taskwarrior authoritatively returned empty output"))
            return self._store(
                scope,
                self._failure(
                    result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail="Taskwarrior export returned empty output",
                ),
            )
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError) as exc:
            return self._store(
                scope,
                self._failure(
                    result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail=f"Taskwarrior export returned malformed JSON: {exc}",
                ),
            )
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            return self._store(
                scope,
                self._failure(
                    result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail="Taskwarrior export returned a non-array or non-object row",
                ),
            )
        unexpected_statuses = sorted(
            {
                str(row.get("status") or "").strip().lower()
                for row in payload
                if str(row.get("status") or "").strip().lower() not in scope.statuses
            }
        )
        if unexpected_statuses:
            return self._store(
                scope,
                self._failure(
                    result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail=f"Taskwarrior export returned statuses outside scope: {', '.join(unexpected_statuses)}",
                ),
            )
        snapshot = AuthoritativeTaskSnapshot(scope, tuple(payload), result)
        return self._store(scope, Found(snapshot, query))

    def _broad_for(
        self,
        statuses: Sequence[str],
        *,
        require_complete_history: bool = False,
    ) -> AuthoritativeTaskSnapshot | None:
        wanted = frozenset(str(status).strip().lower() for status in statuses)
        for epoch, snapshot in reversed(self._snapshots):
            if epoch != self._uow.mutation_epoch:
                continue
            if snapshot.scope.kind not in {
                TaskQueryKind.BROAD,
                TaskQueryKind.ACTIVE_ROOTS,
                TaskQueryKind.LIFECYCLE_CANDIDATES,
            }:
                continue
            if require_complete_history and not snapshot.scope.complete_chain_history:
                continue
            if wanted.issubset(snapshot.scope.statuses):
                return snapshot
        return None

    def _chain_authority_for(
        self,
        chain_id: str,
        statuses: Sequence[str],
        *,
        require_complete_history: bool,
    ) -> AuthoritativeTaskSnapshot | None:
        wanted = frozenset(str(status).strip().lower() for status in statuses)
        for epoch, snapshot in reversed(self._snapshots):
            if epoch != self._uow.mutation_epoch:
                continue
            if not wanted.issubset(snapshot.scope.statuses):
                continue
            if require_complete_history and not snapshot.scope.complete_chain_history:
                continue
            if snapshot.scope.kind is TaskQueryKind.CHAIN and snapshot.scope.identity == chain_id:
                return snapshot
            if snapshot.scope.kind in {
                TaskQueryKind.BROAD,
                TaskQueryKind.ACTIVE_ROOTS,
                TaskQueryKind.LIFECYCLE_CANDIDATES,
            }:
                return snapshot
        return None

    def broad_snapshot(
        self,
        *,
        identity: str,
        filters: Sequence[str],
        statuses: Sequence[str],
        complete_chain_history: bool = False,
        refresh: bool = False,
    ) -> TaskRead[AuthoritativeTaskSnapshot]:
        scope = TaskSnapshotScope(
            TaskQueryKind.BROAD,
            identity,
            tuple(statuses),
            complete_chain_history=complete_chain_history,
        )
        return self._export(scope, filters, empty_output_is_absent=True, refresh=refresh, use_tempfiles=True)

    def by_uuid(
        self,
        uuid_value: str,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
        refresh: bool = False,
        verification: bool = False,
    ) -> TaskRead[TaskRow]:
        identity = str(uuid_value or "").strip().lower()
        kind = TaskQueryKind.VERIFICATION if verification else TaskQueryKind.UUID
        scope = TaskSnapshotScope(kind, identity, tuple(statuses))
        query = self._query_name(scope)
        snapshot = None if refresh else self._broad_for(statuses)
        used_broad_snapshot = snapshot is not None
        if snapshot is None:
            read = self._export(scope, (f"uuid:{identity}",), empty_output_is_absent=True, refresh=refresh)
            if not isinstance(read, Found):
                return read
            snapshot = read.value
        matches = snapshot.uuid_matches(identity)
        if not used_broad_snapshot and len(matches) != len(snapshot.rows):
            return self._failure(
                snapshot.command_result,
                query,
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail=f"UUID export returned rows outside requested identity {identity}",
            )
        if not matches:
            if snapshot.rows and not used_broad_snapshot:
                return self._failure(
                    snapshot.command_result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail=f"UUID export returned a different task for {identity}",
                )
            return Absent(query, "authoritative snapshot contains no matching UUID")
        if len(matches) != 1:
            return self._failure(
                snapshot.command_result,
                query,
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail=f"UUID query returned {len(matches)} exact or prefix matches",
            )
        return Found(matches[0], query)

    def exact_child_slot(
        self,
        chain_id: str,
        link: int,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
        refresh: bool = False,
    ) -> TaskRead[TaskRow]:
        return self._slot_read(TaskQueryKind.CHILD_SLOT, chain_id, link, statuses=statuses, refresh=refresh)

    def predecessor_slot(
        self,
        chain_id: str,
        link: int,
        *,
        statuses: Sequence[str] = ("completed", "deleted"),
        refresh: bool = False,
    ) -> TaskRead[TaskRow]:
        return self._slot_read(
            TaskQueryKind.PREDECESSOR_SLOT,
            chain_id,
            link,
            statuses=statuses,
            refresh=refresh,
            require_complete_history=True,
        )

    def _slot_read(
        self,
        kind: TaskQueryKind,
        chain_id: str,
        link: int,
        *,
        statuses: Sequence[str],
        refresh: bool,
        require_complete_history: bool = False,
    ) -> TaskRead[TaskRow]:
        chain = str(chain_id or "").strip()
        link_no = int(link)
        scope = TaskSnapshotScope(kind, f"{chain}:{link_no}", tuple(statuses), require_complete_history)
        query = self._query_name(scope)
        snapshot = None if refresh else self._chain_authority_for(
            chain,
            statuses,
            require_complete_history=require_complete_history,
        )
        used_broad_snapshot = snapshot is not None
        if snapshot is None:
            read = self._export(
                scope,
                (f"chainID:{chain}", f"link:{link_no}"),
                empty_output_is_absent=True,
                refresh=refresh,
            )
            if not isinstance(read, Found):
                return read
            snapshot = read.value
        matches = snapshot.slot_rows(chain, link_no)
        if not used_broad_snapshot and len(matches) != len(snapshot.rows):
            return self._failure(
                snapshot.command_result,
                query,
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail=f"chain slot export returned rows outside requested slot {chain}:{link_no}",
            )
        if not matches:
            if snapshot.rows and not used_broad_snapshot:
                return self._failure(
                    snapshot.command_result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail=f"chain slot export returned a different slot for {chain}:{link_no}",
                )
            return Absent(query, "authoritative snapshot contains no matching chain slot")
        if len(matches) != 1:
            return self._failure(
                snapshot.command_result,
                query,
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail=f"exact chain slot returned {len(matches)} matches",
            )
        return Found(matches[0], query)

    def chain_snapshot(
        self,
        chain_id: str,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
        complete_history: bool = True,
        refresh: bool = False,
    ) -> TaskRead[tuple[TaskRow, ...]]:
        chain = str(chain_id or "").strip()
        scope = TaskSnapshotScope(TaskQueryKind.CHAIN, chain, tuple(statuses), complete_history)
        query = self._query_name(scope)
        snapshot = None if refresh else self._chain_authority_for(
            chain,
            statuses,
            require_complete_history=complete_history,
        )
        used_broad_snapshot = snapshot is not None
        if snapshot is None:
            read = self._export(scope, (f"chainID:{chain}",), empty_output_is_absent=True, refresh=refresh, use_tempfiles=True)
            if not isinstance(read, Found):
                return read
            snapshot = read.value
        rows = snapshot.chain_rows(chain)
        if not used_broad_snapshot and len(rows) != len(snapshot.rows):
            return self._failure(
                snapshot.command_result,
                query,
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail=f"chain export returned rows outside requested chain {chain}",
            )
        if not rows:
            if snapshot.rows and not used_broad_snapshot:
                return self._failure(
                    snapshot.command_result,
                    query,
                    kind=CommandFailureKind.INVALID_RESPONSE,
                    detail=f"chain export returned a different chain for {chain}",
                )
            return Absent(query, "authoritative snapshot contains no matching chain")
        if any(str(row.get("chainID") or "").strip() != chain for row in rows):
            return self._failure(
                snapshot.command_result,
                query,
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail="chain export returned a mismatched chainID",
            )
        return Found(rows, query)

    def active_recurrence_roots(
        self,
        *,
        refresh: bool = False,
    ) -> TaskRead[tuple[TaskRow, ...]]:
        scope = TaskSnapshotScope(TaskQueryKind.ACTIVE_ROOTS, "chain:on-link:1", ACTIVE_TASK_STATUSES)
        read = self._export(scope, ("chain:on", "link:1"), empty_output_is_absent=True, refresh=refresh)
        if not isinstance(read, Found):
            return read
        rows = tuple(
            row
            for row in read.value.rows
            if str(row.get("chain") or "").lower() == "on" and _link_number(row.get("link")) == 1
        )
        if len(rows) != len(read.value.rows):
            return self._failure(
                read.value.command_result,
                self._query_name(scope),
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail="active-root export returned a mismatched chain state or link",
            )
        return Found(rows, self._query_name(scope)) if rows else Absent(self._query_name(scope), "no active recurrence roots")

    def lifecycle_candidates(
        self,
        *,
        statuses: Sequence[str] = ("completed", "deleted", "pending"),
        refresh: bool = False,
    ) -> TaskRead[tuple[TaskRow, ...]]:
        scope = TaskSnapshotScope(TaskQueryKind.LIFECYCLE_CANDIDATES, "chain:on", tuple(statuses))
        read = self._export(scope, ("chain:on",), empty_output_is_absent=True, refresh=refresh, use_tempfiles=True)
        if not isinstance(read, Found):
            return read
        if any(str(row.get("chain") or "").strip().lower() != "on" for row in read.value.rows):
            return self._failure(
                read.value.command_result,
                self._query_name(scope),
                kind=CommandFailureKind.INVALID_RESPONSE,
                detail="lifecycle candidate export returned a task outside chain:on",
            )
        return Found(read.value.rows, self._query_name(scope)) if read.value.rows else Absent(self._query_name(scope), "no lifecycle candidates")

    def verification(
        self,
        uuid_value: str,
        *,
        statuses: Sequence[str] = ALL_TASK_STATUSES,
    ) -> TaskRead[TaskRow]:
        return self.by_uuid(uuid_value, statuses=statuses, refresh=True, verification=True)


__all__ = (
    "AuthoritativeTaskSnapshot",
    "ACTIVE_TASK_STATUSES",
    "ALL_TASK_STATUSES",
    "TaskReadRepository",
    "TaskQueryKind",
    "TaskRow",
    "TaskSlot",
    "TaskSnapshotScope",
)
