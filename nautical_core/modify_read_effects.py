"""Lifecycle chain-read composition for the typed on-modify workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .lifecycle.read_service import (
    ChainCacheStore,
    ChainSnapshotRepository,
    CoerceInt,
    Counter,
    Diagnostic,
    LifecycleReadService,
    ReadQuery,
    TokenMatcher,
    TokenParser,
)
from .integration_models import TaskCommandResult
from .task_models import TaskObservation


class ChainExportReader(Protocol):
    def get_chain_export(self, chain_id: str) -> list[TaskObservation] | None: ...


class TaskRowDecoder(Protocol):
    def __call__(
        self, row: Mapping[str, Any], *, source_query: str
    ) -> TaskObservation: ...


class TaskFieldReader(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...


@dataclass(frozen=True, slots=True)
class LifecycleReadCapabilities:
    """Explicit collaborators required to construct lifecycle read services."""

    coerce_int: CoerceInt
    parse_extra_tokens: TokenParser
    token_matcher: TokenMatcher
    read_query_get: ReadQuery
    read_query_missing: object
    max_chain_walk: int
    diag: Diagnostic
    record_stat: Counter
    cache_store: ChainCacheStore
    repository: ChainSnapshotRepository | None


@dataclass(frozen=True, slots=True)
class ExtraTokenPort:
    parse: Callable[[str | None], list[str] | None]


@dataclass(frozen=True, slots=True)
class ChainExportPort:
    service: ChainExportReader


@dataclass(frozen=True, slots=True)
class SeedLookupPorts:
    service: LifecycleReadService
    decode_row: TaskRowDecoder
    cache_set: Callable[[str, Any, Any], None]


@dataclass(frozen=True, slots=True)
class PreviousChainPorts:
    service: LifecycleReadService
    panel_chain_by_link: dict[int, list[TaskObservation]]
    panel_chain_snapshot_loaded: bool


@dataclass(frozen=True, slots=True)
class TwGetPorts:
    service: LifecycleReadService
    cache_get: Callable[[str, str], object]
    cache_set: Callable[[str, str, object], None]
    count: Callable[[str], None]
    diagnostic: Callable[[str], None]
    run_task: Callable[..., TaskCommandResult]
    command_prefix: Callable[[], list[str]]
    environment: Callable[[], dict[str, str]]


def _token_match(coerce_int: CoerceInt, task: TaskFieldReader, token: str) -> bool:
    if not hasattr(task, "get") or not isinstance(token, str) or not token:
        return False
    if token.startswith("+"):
        want = token[1:].strip().lower()
        tags = task.get("tags")
        return isinstance(tags, (list, tuple, set)) and want in {str(tag).strip().lower() for tag in tags}
    if ":" not in token:
        return False
    key, value = token.split(":", 1)
    negate = key.endswith(".not")
    if negate:
        key = key[:-4]
    actual = task.get(key)
    if key in {"link", "id"}:
        matched = str(coerce_int(actual, None) if actual is not None else "") == value
    else:
        matched = str(actual or "").strip().lower() == value.strip().lower()
    return (not matched) if negate else matched


def parse_extra_tokens(port: ExtraTokenPort, extra: str | None) -> list[str] | None:
    return port.parse(extra)


def seed_runtime_lookup_task(
    ports: SeedLookupPorts,
    payload: dict[str, Any] | None,
    *,
    lookup_short: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    uuid_str = str(payload.get("uuid") or "").strip()
    if not uuid_str:
        return None
    short = uuid_str[:8]
    observation = ports.decode_row(payload, source_query="on-modify lookup seed")
    task_obj = ports.service.seed_lookup_task(observation, short_uuid=short)
    requested_short = str(lookup_short or "").strip()
    if requested_short and requested_short != short:
        task_obj = ports.service.seed_lookup_task(task_obj, short_uuid=requested_short)
    entry = task_obj.get("entry")
    if short and entry:
        ports.cache_set("tw_get", f"{short}.entry", str(entry).strip())
    return task_obj.to_mapping()


def seed_runtime_lookup_tasks(
    ports: SeedLookupPorts, *tasks: dict[str, Any] | None
) -> None:
    for task in tasks:
        seed_runtime_lookup_task(ports, task)


def collect_prev_two(
    ports: PreviousChainPorts,
    current_task: TaskObservation,
    chain_by_link: dict[int, list[TaskObservation]] | None = None,
) -> list[TaskObservation]:
    from .integration_models import Absent, Found, Unavailable

    read = ports.service.collect_prev_two(
        current_task,
        get_chain_read=lambda chain_id: ports.service.get_chain_read(chain_id),
        panel_chain_by_link=ports.panel_chain_by_link,
        panel_chain_snapshot_loaded=ports.panel_chain_snapshot_loaded,
        chain_by_link=chain_by_link,
    )
    if isinstance(read, Unavailable):
        raise RuntimeError(read.evidence.detail or "lifecycle predecessor read unavailable")
    if isinstance(read, Absent):
        return []
    if not isinstance(read, Found):
        raise RuntimeError("lifecycle predecessor read returned an invalid result")
    return list(read.value)


def export_chain_required(port: ChainExportPort, seed_payload: dict[str, Any], env: Any = None) -> Any:
    chain_id = seed_payload.get("chainID")
    if not chain_id:
        raise RuntimeError("ChainID is required (legacy chain traversal removed). Run chainID backfill, then retry.")
    if env is not None:
        raise RuntimeError("chain reads must use the invocation Taskwarrior repository")
    rows = port.service.get_chain_export(chain_id)
    if rows is None:
        raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
    return rows


def tw_get_ports_for(host: Any) -> TwGetPorts:
    command = host._module("modify_command_effects")
    composition = host._module("modify_composition")
    return TwGetPorts(
        service=composition.lifecycle_read_service_for(host),
        cache_get=host._query_ctx_get,
        cache_set=host._query_ctx_set,
        count=host._diag_count,
        diagnostic=host._diag,
        run_task=lambda argv, **kwargs: command.run_task_result(
            command.command_ports_for(host), argv, **kwargs
        ),
        command_prefix=host._task_cmd_prefix,
        environment=lambda: host.os.environ.copy(),
    )


def tw_get_cached(ports: TwGetPorts, ref: str) -> str:
    """Return one cached Taskwarrior ``_get`` value for the current hook."""
    if ref.endswith(".entry"):
        short = ref[:-6].strip()
        cached, cache_chain_id = ports.service.lookup_short(short) if short else (None, "")
        if short and isinstance(cached, Mapping):
            ports.count("tw_get_cache_hits")
            return (str(cached.get("entry") or "")).strip()
        if short and cache_chain_id:
            ports.count("unexpected_cache_misses")
            ports.diagnostic(f"cache miss: _get {ref} (chainID={cache_chain_id})")
    cached_value = ports.cache_get("tw_get", ref)
    if isinstance(cached_value, str):
        ports.count("tw_get_cache_hits")
        return cached_value
    ports.count("tw_get_cache_misses")
    result = ports.run_task(
        ports.command_prefix() + ["rc.hooks=off", "rc.verbose=nothing", "_get", ref],
        env=ports.environment(),
        timeout=3.0,
        retries=2,
    )
    out = (result.stdout or "").strip() if result.ok else ""
    ports.cache_set("tw_get", ref, out or "")
    return out


__all__ = ("parse_extra_tokens", "SeedLookupPorts", "PreviousChainPorts", "seed_runtime_lookup_task", "seed_runtime_lookup_tasks", "collect_prev_two", "ChainExportPort", "export_chain_required", "TwGetPorts", "tw_get_ports_for", "tw_get_cached")
