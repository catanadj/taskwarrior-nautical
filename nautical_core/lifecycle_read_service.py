"""Taskdata read orchestration for lifecycle operations.

The service deliberately knows nothing about hook globals or Taskwarrior's
process runner.  Those concerns are supplied as callbacks so completion,
reconcile, and future lifecycle consumers can share one read contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from collections.abc import Callable, Sequence
from functools import lru_cache
import threading
from typing import Any

TaskRow = dict[str, Any]
ReadQuery = Callable[[str, tuple[Any, ...]], Any]
ChainExport = Callable[[str, datetime | None, str | None, int], Sequence[TaskRow]]
ChainCache = Callable[[str], Sequence[TaskRow] | None]
TokenParser = Callable[[str | None], list[str] | None]
TokenMatcher = Callable[[TaskRow, str], bool]
CoerceInt = Callable[[Any, int | None], int | None]
Diagnostic = Callable[[str], None]
Counter = Callable[[str], None]


def chain_read_key(
    chain_id: str,
    since: datetime | None,
    extra: str | None,
    limit: int,
) -> tuple[Any, ...]:
    """Build the stable request-cache key for one chain read."""
    return (
        str(chain_id or ""),
        since.isoformat() if isinstance(since, datetime) else "",
        str(extra or ""),
        int(limit or 0),
    )


@lru_cache(maxsize=32)
def cached_chain_export(
    exporter: Callable[[str, datetime | None, str | None, int], Sequence[TaskRow]],
    chain_id: str,
    since_key: str,
    extra_key: str,
    limit: int,
) -> tuple[TaskRow, ...]:
    """Memoize one validated chain export by all scheduling parameters."""
    since = datetime.fromisoformat(since_key) if since_key else None
    rows = exporter(chain_id, since, extra_key or None, limit)
    return tuple(rows)


def clear_cached_chain_exports() -> None:
    """Clear the process-local export cache after a Taskwarrior mutation."""
    cached_chain_export.cache_clear()


@dataclass(frozen=True, slots=True)
class ChainIndexes:
    """Stable indexes used by completion lookups for one chain snapshot."""

    by_link: dict[int, list[TaskRow]]
    by_short: dict[str, TaskRow]
    by_uuid: dict[str, TaskRow]


@dataclass(slots=True)
class ChainCacheStore:
    """Request-scoped chain rows and indexes owned by the read service."""

    chain_id: str = ""
    rows: list[TaskRow] = field(default_factory=list)
    indexes: ChainIndexes | None = None
    lock: threading.RLock = field(default_factory=threading.RLock)

    def rows_for(self, chain_id: str) -> list[TaskRow] | None:
        with self.lock:
            if self.chain_id == chain_id and self.rows:
                return list(self.rows)
        return None

    def replace(self, chain_id: str, rows: Sequence[TaskRow], indexes: ChainIndexes) -> None:
        with self.lock:
            self.chain_id = str(chain_id or "")
            self.rows = [dict(row) for row in rows if isinstance(row, dict)]
            self.indexes = indexes


class LifecycleReadService:
    """Own chain snapshot filtering, indexing, and read-cache orchestration."""

    def __init__(
        self,
        *,
        coerce_int: CoerceInt,
        parse_extra_tokens: TokenParser,
        token_matcher: TokenMatcher,
        read_query_get: ReadQuery,
        chain_cache_get: ChainCache,
        export_chain_cached: ChainExport,
        max_chain_walk: int,
        diag: Diagnostic | None = None,
        record_stat: Counter | None = None,
        cache_store: ChainCacheStore | None = None,
        read_query_missing: object | None = None,
    ) -> None:
        self._coerce_int = coerce_int
        self._parse_extra_tokens = parse_extra_tokens
        self._token_matcher = token_matcher
        self._read_query_get = read_query_get
        self._chain_cache_get = chain_cache_get
        self._export_chain_cached = export_chain_cached
        self._max_chain_walk = max(1, int(max_chain_walk))
        self._diag = diag or (lambda _message: None)
        self._record_stat = record_stat or (lambda _name: None)
        self._cache_store = cache_store
        self._read_query_missing = read_query_missing

    def cached_chain_rows(self, chain_id: str) -> list[TaskRow] | None:
        """Return the service-owned chain cache, if seeded for this chain."""
        if self._cache_store is not None:
            return self._cache_store.rows_for(chain_id)
        return list(self._chain_cache_get(chain_id) or []) or None

    def lookup_short(self, short_uuid: str) -> tuple[TaskRow | None, str]:
        """Return a cached short UUID row and the cache's chain ID."""
        if self._cache_store is not None:
            with self._cache_store.lock:
                row = self._cache_store.indexes.by_short.get(short_uuid) if self._cache_store.indexes else None
                return (dict(row) if isinstance(row, dict) else None, self._cache_store.chain_id)
        return None, ""

    def lookup_uuid(self, uuid_value: str) -> TaskRow | None:
        """Return a cached full UUID row, if present."""
        if self._cache_store is not None:
            with self._cache_store.lock:
                row = self._cache_store.indexes.by_uuid.get(uuid_value) if self._cache_store.indexes else None
                return dict(row) if isinstance(row, dict) else None
        return None

    def cache_size(self, chain_id: str) -> int:
        """Return the number of cached rows for adaptive export timeouts."""
        if self._cache_store is not None:
            with self._cache_store.lock:
                return len(self._cache_store.rows) if self._cache_store.chain_id == chain_id else 0
        return len(self.cached_chain_rows(chain_id) or [])

    def seed_lookup_task(self, task: TaskRow, *, short_uuid: str) -> TaskRow:
        """Merge one exported task into the service-owned lookup indexes."""
        if self._cache_store is None:
            return dict(task)
        uuid_value = str(task.get("uuid") or "").strip()
        task_obj = dict(task)
        with self._cache_store.lock:
            existing = None
            if short_uuid and self._cache_store.indexes:
                existing = self._cache_store.indexes.by_short.get(short_uuid)
            if not isinstance(existing, dict) and uuid_value and self._cache_store.indexes:
                existing = self._cache_store.indexes.by_uuid.get(uuid_value)
            if isinstance(existing, dict):
                merged = dict(existing)
                merged.update(task)
                task_obj = merged
            by_short = dict(self._cache_store.indexes.by_short) if self._cache_store.indexes else {}
            by_uuid = dict(self._cache_store.indexes.by_uuid) if self._cache_store.indexes else {}
            if short_uuid:
                by_short[short_uuid] = task_obj
            if uuid_value:
                by_uuid[uuid_value] = task_obj
            existing_indexes = self._cache_store.indexes
            self._cache_store.indexes = ChainIndexes(
                by_link=existing_indexes.by_link if existing_indexes else {},
                by_short=by_short,
                by_uuid=by_uuid,
            )
        return task_obj

    def replace_chain_cache(self, chain_id: str, rows: Sequence[TaskRow]) -> ChainIndexes:
        """Replace cached chain rows and their indexes atomically."""
        stored_rows = [dict(row) for row in rows if isinstance(row, dict)]
        indexes = self.build_indexes(stored_rows)
        if self._cache_store is not None:
            self._cache_store.replace(chain_id, stored_rows, indexes)
        return indexes

    def collect_prev_two(
        self,
        current_task: TaskRow,
        *,
        get_chain_export: Callable[[str], list[TaskRow] | None],
        panel_chain_by_link: dict[int, list[TaskRow]] | None = None,
        panel_chain_snapshot_loaded: bool = False,
        chain_by_link: dict[int, list[TaskRow]] | None = None,
    ) -> list[TaskRow]:
        """Return up to two previous links from one authoritative chain read."""
        chain_id = str(current_task.get("chainID") or "").strip()
        if not chain_id:
            return []
        current_no = self._coerce_int(current_task.get("link"), None)
        if not current_no or current_no <= 1:
            return []

        def pick_best(candidates: list[TaskRow]) -> TaskRow | None:
            for status in ("pending", "completed", "deleted"):
                for task in candidates:
                    if str(task.get("status") or "").strip().lower() == status:
                        return task
            return candidates[0] if candidates else None

        chain_index = chain_by_link or panel_chain_by_link or {}
        if not chain_index and not panel_chain_snapshot_loaded:
            try:
                chain = get_chain_export(chain_id)
            except Exception:
                return []
            if not isinstance(chain, list):
                return []
            chain_index = self.build_indexes(chain).by_link

        previous: list[TaskRow] = []
        for wanted in (current_no - 2, current_no - 1):
            if wanted < 1:
                continue
            task = pick_best(chain_index.get(wanted, []))
            if task:
                previous.append(task)
        return previous

    def build_indexes(self, rows: Sequence[TaskRow]) -> ChainIndexes:
        """Build link, short UUID, and full UUID indexes in one pass."""
        by_link: dict[int, list[TaskRow]] = {}
        by_short: dict[str, TaskRow] = {}
        by_uuid: dict[str, TaskRow] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            link_no = self._coerce_int(row.get("link"), None)
            if link_no is not None:
                by_link.setdefault(link_no, []).append(row)
            uuid_value = row.get("uuid")
            if isinstance(uuid_value, str) and uuid_value:
                by_short[uuid_value[:8]] = row
                by_uuid[uuid_value] = row
        return ChainIndexes(by_link=by_link, by_short=by_short, by_uuid=by_uuid)

    def filter_rows(
        self,
        rows: Sequence[TaskRow],
        *,
        extra: str | None,
        limit: int | None,
    ) -> list[TaskRow] | None:
        """Apply the supported in-memory Taskwarrior predicates."""
        tokens = self._parse_extra_tokens(extra)
        if tokens is None:
            return None
        filtered = [row for row in rows if isinstance(row, dict)]
        for token in tokens:
            filtered = [row for row in filtered if self._token_matcher(row, token)]
        if isinstance(limit, int) and limit > 0:
            filtered = filtered[:limit]
        return filtered

    def filter_full_snapshot(
        self,
        rows: Sequence[TaskRow],
        *,
        extra: str | None,
        limit: int | None,
    ) -> list[TaskRow] | None:
        """Filter a full snapshot only when the predicate semantics are safe."""
        tokens = self._parse_extra_tokens(extra)
        if tokens is None:
            return None
        for token in tokens:
            if token.startswith("+"):
                continue
            key = token.split(":", 1)[0]
            base_key = key[:-4] if key.endswith(".not") else key
            if base_key not in {"chainID", "link", "id", "project", "status"}:
                return None
        return self.filter_rows(rows, extra=extra, limit=limit)

    def get_chain_export(
        self,
        chain_id: str,
        *,
        since: datetime | None = None,
        extra: str | None = None,
        read_query_missing: object | None = None,
        read_query_key: Callable[[str, datetime | None, str | None, int], tuple[Any, ...]] = chain_read_key,
    ) -> list[TaskRow] | None:
        """Read one chain using the request snapshot, run cache, or exporter."""
        missing = self._read_query_missing if read_query_missing is None else read_query_missing
        if not chain_id:
            return []
        if not since:
            full_snapshot = self._read_query_get(
                "chain", read_query_key(chain_id, None, None, 0)
            )
            if full_snapshot is not missing:
                if not isinstance(full_snapshot, list) or any(
                    not isinstance(row, dict) for row in full_snapshot
                ):
                    self._diag(f"cached chain read has invalid shape (chainID={chain_id})")
                    return None
                filtered = self.filter_full_snapshot(
                    full_snapshot,
                    extra=extra,
                    limit=self._max_chain_walk,
                )
                if filtered is not None:
                    self._record_stat("chain_snapshot_filter_hits")
                    return filtered
        cached_chain = self.cached_chain_rows(chain_id)
        if cached_chain and not since and not extra:
            return list(cached_chain)
        if cached_chain and not since:
            filtered = self.filter_rows(
                cached_chain,
                extra=extra,
                limit=self._max_chain_walk,
            )
            if filtered is not None:
                self._record_stat("chain_cache_filter_hits")
                return filtered
        try:
            rows = cached_chain_export(
                self._export_chain_cached,
                chain_id,
                since.isoformat() if isinstance(since, datetime) else "",
                str(extra or ""),
                self._max_chain_walk,
            )
        except RuntimeError as exc:
            self._diag(f"chain read unavailable (chainID={chain_id}): {exc}")
            return None
        return list(rows)

    def merge_spawned_child(
        self,
        rows: Sequence[TaskRow],
        *,
        parent_task: TaskRow,
        child_task: TaskRow,
        child_short: str,
        short_uuid: Callable[[str], str],
    ) -> list[TaskRow]:
        """Merge a spawned child into a previously read chain snapshot."""
        if not rows:
            return []
        parent_uuid = str(parent_task.get("uuid") or "").strip()
        parent_short = short_uuid(parent_uuid)
        child_uuid = str(child_task.get("uuid") or "").strip()
        merged: list[TaskRow] = []
        child_present = False
        for row in rows:
            row_obj = dict(row)
            row_uuid = str(row_obj.get("uuid") or "").strip()
            if parent_uuid and row_uuid == parent_uuid and child_short:
                row_obj["nextLink"] = child_short
            if child_uuid and row_uuid == child_uuid:
                child_present = True
                row_obj.update(child_task)
            merged.append(row_obj)
        if child_uuid and not child_present:
            child_obj = dict(child_task)
            if parent_short and not str(child_obj.get("prevLink") or "").strip():
                child_obj["prevLink"] = parent_short
            merged.append(child_obj)
        merged.sort(
            key=lambda task: (
                self._coerce_int(task.get("link"), 10**9),
                str(task.get("uuid") or ""),
            )
        )
        return merged


__all__ = [
    "ChainCacheStore",
    "ChainIndexes",
    "LifecycleReadService",
    "cached_chain_export",
    "chain_read_key",
    "clear_cached_chain_exports",
]
