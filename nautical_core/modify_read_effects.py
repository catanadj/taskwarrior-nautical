"""Lifecycle chain-read composition for the typed on-modify workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LifecycleReadCapabilities:
    """Explicit collaborators required to construct lifecycle read services."""

    coerce_int: Any
    parse_extra_tokens: Any
    token_matcher: Any
    read_query_get: Any
    read_query_missing: Any
    max_chain_walk: int
    diag: Any
    record_stat: Any
    cache_store: Any
    repository: Any


@dataclass(frozen=True, slots=True)
class ExtraTokenPort:
    parse: Any


@dataclass(frozen=True, slots=True)
class ChainExportPort:
    service: Any


@dataclass(frozen=True, slots=True)
class SeedLookupPorts:
    service: Any
    decode_row: Any
    cache_set: Any


@dataclass(frozen=True, slots=True)
class PreviousChainPorts:
    service: Any
    panel_chain_by_link: Any
    panel_chain_snapshot_loaded: Any


@dataclass(frozen=True, slots=True)
class TwGetPorts:
    service: Any
    cache_get: Any
    cache_set: Any
    count: Any
    diagnostic: Any
    run_task: Any
    command_prefix: Any
    environment: Any


def _token_match(coerce_int: Any, task: Any, token: str) -> bool:
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


def seed_runtime_lookup_task(ports: SeedLookupPorts, payload: dict[str, Any] | None, *, lookup_short: str | None = None) -> Any:
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


def seed_runtime_lookup_tasks(ports: SeedLookupPorts, *tasks: dict | None) -> None:
    for task in tasks:
        seed_runtime_lookup_task(ports, task)


def collect_prev_two(ports: PreviousChainPorts, current_task: dict[str, Any], chain_by_link: Any = None) -> Any:
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
    try:
        if ref.endswith(".entry"):
            short = ref[:-6].strip()
            cached, cache_chain_id = ports.service.lookup_short(short) if short else (None, "")
            if short and isinstance(cached, Mapping):
                ports.count("tw_get_cache_hits")
                return (str(cached.get("entry") or "")).strip()
            if short and cache_chain_id:
                ports.count("unexpected_cache_misses")
                ports.diagnostic(f"cache miss: _get {ref} (chainID={cache_chain_id})")
        cached = ports.cache_get("tw_get", ref)
        if isinstance(cached, str):
            ports.count("tw_get_cache_hits")
            return cached
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
    except Exception:
        return ""


__all__ = ("parse_extra_tokens", "SeedLookupPorts", "PreviousChainPorts", "seed_runtime_lookup_task", "seed_runtime_lookup_tasks", "collect_prev_two", "ChainExportPort", "export_chain_required", "TwGetPorts", "tw_get_ports_for", "tw_get_cached")
