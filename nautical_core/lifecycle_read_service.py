"""Taskdata read orchestration for lifecycle operations.

The service deliberately knows nothing about hook globals or Taskwarrior's
process runner.  Those concerns are supplied as callbacks so completion,
reconcile, and future lifecycle consumers can share one read contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

from .hook_support import LookupResult


TaskRow = dict[str, Any]
ReadQuery = Callable[[str, tuple[Any, ...]], Any]
ChainExport = Callable[[str, datetime | None, str | None, int], Sequence[TaskRow]]
ChainCache = Callable[[str], Sequence[TaskRow] | None]
BuildExportArgs = Callable[..., list[str] | None]
RunCheckedExport = Callable[[list[str], Any, float], "ChainReadResult"]
ReadQuerySet = Callable[[str, tuple[Any, ...], Any], None]
ReadQueryDelete = Callable[[str, tuple[Any, ...]], None]
TokenParser = Callable[[str | None], list[str] | None]
TokenMatcher = Callable[[TaskRow, str], bool]
CoerceInt = Callable[[Any, int | None], int | None]
Diagnostic = Callable[[str], None]
Counter = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class ChainReadResult:
    """Typed result for a chain export at the Taskwarrior boundary."""

    ok: bool
    rows: list[TaskRow]
    error: str = ""

    @classmethod
    def success(cls, rows: Sequence[TaskRow]) -> "ChainReadResult":
        return cls(True, list(rows), "")

    @classmethod
    def failure(cls, error: str) -> "ChainReadResult":
        return cls(False, [], str(error or "chain export unavailable"))


@dataclass(frozen=True, slots=True)
class ChainSnapshotResult:
    """Typed completion snapshot result, including its lookup coverage."""

    mode: str
    rows: list[TaskRow]
    loaded: bool
    chain_id: str = ""
    error: str = ""

    @property
    def is_unavailable(self) -> bool:
        return not self.loaded and bool(self.error)


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
        read_query_set: ReadQuerySet | None = None,
        read_query_delete: ReadQueryDelete | None = None,
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
        self._read_query_set = read_query_set or (lambda _kind, _key, _value: None)
        self._read_query_delete = read_query_delete or (lambda _kind, _key: None)

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

    def existing_next_lookup(
        self,
        parent_task: TaskRow,
        next_no: int,
        *,
        export_uuid_short_cached: Callable[[str], Any],
        get_chain_export: Callable[..., list[TaskRow] | None],
        snapshot_rows: list[TaskRow] | None = None,
        snapshot_loaded: bool = False,
    ) -> LookupResult:
        """Find an existing successor while preserving found/absent/unavailable."""
        if not isinstance(parent_task, dict):
            return LookupResult.unavailable("parent task is not an object")
        rows = [
            row
            for row in (snapshot_rows or [])
            if isinstance(row, dict)
            and str(row.get("link") or "").strip() == str(int(next_no))
            and str(row.get("status") or "").strip().lower() != "deleted"
        ]
        if rows:
            picked = self._pick_existing_next(rows)
            return LookupResult.found(picked) if picked else LookupResult.absent()

        next_ref = str(parent_task.get("nextLink") or "").strip()
        if next_ref:
            obj = export_uuid_short_cached(next_ref)
            if isinstance(obj, LookupResult):
                if obj.is_found or obj.is_unavailable:
                    return obj
                obj = None
            if isinstance(obj, dict) and str(obj.get("status") or "").strip().lower() != "deleted":
                return LookupResult.found(obj)

        chain_id = str(parent_task.get("chainID") or "").strip()
        if not chain_id or snapshot_loaded:
            return LookupResult.absent()
        try:
            rows = get_chain_export(
                chain_id,
                extra=f"link:{int(next_no)} status.not:deleted",
            )
        except Exception as exc:
            return LookupResult.unavailable(f"chain export failed: {exc}")
        if rows is None:
            return LookupResult.unavailable("chain export was unavailable")
        picked = self._pick_existing_next(rows)
        return LookupResult.found(picked) if picked else LookupResult.absent()

    @staticmethod
    def _pick_existing_next(rows: list[TaskRow]) -> TaskRow | None:
        for status in ("pending", "waiting", "completed"):
            for row in rows:
                if str(row.get("status") or "").strip().lower() == status:
                    return row
        return rows[0] if rows else None

    def completion_snapshot(
        self,
        chain_id: str,
        *,
        mode: str,
        links: Sequence[int] | None,
        load_snapshot: Callable[[str, list[int] | None], ChainReadResult],
        read_query_missing: object,
    ) -> ChainSnapshotResult:
        """Reuse or load one completion snapshot and promote full coverage."""
        normalized_links = None if links is None else tuple(sorted({int(link) for link in links if int(link) > 0}))
        snapshot_key = (str(chain_id), normalized_links)
        cached_snapshot = self._read_query_get("chain_snapshot", snapshot_key)
        if cached_snapshot is not read_query_missing:
            self._record_stat("chain_snapshot_hits")
            if not isinstance(cached_snapshot, list) or any(
                not isinstance(row, dict) for row in cached_snapshot
            ):
                self._read_query_delete("chain_snapshot", snapshot_key)
                return ChainSnapshotResult(
                    mode=mode,
                    rows=[],
                    loaded=False,
                    chain_id=str(chain_id),
                    error="cached completion snapshot has invalid shape",
                )
            return ChainSnapshotResult(
                mode=mode,
                rows=list(cached_snapshot),
                loaded=True,
                chain_id=str(chain_id),
            )

        self._record_stat("chain_snapshot_misses")
        result = load_snapshot(str(chain_id), None if normalized_links is None else list(normalized_links))
        if not result.ok:
            return ChainSnapshotResult(
                mode=mode,
                rows=[],
                loaded=False,
                chain_id=str(chain_id),
                error=result.error,
            )
        self._read_query_set("chain_snapshot", snapshot_key, result.rows)
        if normalized_links is None:
            self._read_query_set(
                "chain",
                chain_read_key(str(chain_id), None, None, 0),
                result.rows,
            )
        return ChainSnapshotResult(
            mode=mode,
            rows=list(result.rows),
            loaded=True,
            chain_id=str(chain_id),
        )

    def checked_export(
        self,
        chain_id: str,
        *,
        since: datetime | None,
        extra: str | None,
        env: Any,
        limit: int | None,
        build_args: BuildExportArgs,
        run_export: RunCheckedExport,
        timeout_for_chain: Callable[[str], float],
        read_query_missing: object,
    ) -> ChainReadResult:
        """Load one chain with typed failures and request-scoped reuse."""
        if not chain_id:
            return ChainReadResult.success([])
        args = build_args(chain_id, since=since, extra=extra, limit=limit)
        if args is None:
            return ChainReadResult.failure("invalid chain export filters")
        read_key = chain_read_key(chain_id, since, extra, int(limit or 0))
        if env is None:
            cached_read = self._read_query_get("chain", read_key)
            if cached_read is not read_query_missing:
                if isinstance(cached_read, list) and all(isinstance(row, dict) for row in cached_read):
                    return ChainReadResult.success(cached_read)
                return ChainReadResult.failure("cached chain export has invalid shape")
            if not since and not extra and int(limit or 0) > 0:
                full_key = chain_read_key(chain_id, None, None, 0)
                full_read = self._read_query_get("chain", full_key)
                if full_read is not read_query_missing:
                    if not isinstance(full_read, list) or not all(
                        isinstance(row, dict) for row in full_read
                    ):
                        return ChainReadResult.failure("cached chain export has invalid shape")
                    return ChainReadResult.success(full_read[: int(limit)])
        result = run_export(args, env, timeout_for_chain(chain_id))
        if result.ok and env is None:
            self._read_query_set("chain", read_key, result.rows)
        return result

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
        read_query_missing: object,
        read_query_key: Callable[[str, datetime | None, str | None, int], tuple[Any, ...]],
    ) -> list[TaskRow] | None:
        """Read one chain using the request snapshot, run cache, or exporter."""
        if not chain_id:
            return []
        if not since:
            full_snapshot = self._read_query_get(
                "chain", read_query_key(chain_id, None, None, 0)
            )
            if full_snapshot is not read_query_missing:
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
        cached_chain = self._chain_cache_get(chain_id)
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
    "ChainReadResult",
    "ChainSnapshotResult",
    "ChainIndexes",
    "LifecycleReadService",
    "cached_chain_export",
    "chain_read_key",
    "clear_cached_chain_exports",
]
