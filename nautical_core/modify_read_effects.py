"""Lifecycle chain-read composition for the typed on-modify workflow."""

from __future__ import annotations

from typing import Any


def _token_match(core: Any, task: Any, token: str) -> bool:
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
        matched = str(core.coerce_int(actual, None) if actual is not None else "") == value
    else:
        matched = str(actual or "").strip().lower() == value.strip().lower()
    return (not matched) if negate else matched


def parse_extra_tokens(host: Any, extra: str | None) -> list[str] | None:
    return host._module("hook_support", required=False).parse_extra_tokens(extra)


def lifecycle_read_service(host: Any):
    state = host._modify_runtime_state()
    existing = getattr(state, "lifecycle_read_service", None)
    if existing is not None:
        repository = getattr(state, "task_repository", None)
        if repository is not None:
            bind_repository = getattr(existing, "bind_repository", None)
            if callable(bind_repository):
                bind_repository(repository)
        return existing
    module = host._module("lifecycle_read_service")
    if getattr(state, "chain_cache_store", None) is None:
        state.chain_cache_store = module.ChainCacheStore()
    service = module.LifecycleReadService(
        coerce_int=host.core.coerce_int,
        parse_extra_tokens=lambda extra: parse_extra_tokens(host, extra),
        token_matcher=lambda task, token: _token_match(host.core, task, token),
        read_query_get=host._read_query_get,
        chain_cache_get=lambda _chain_id: None,
        repository=getattr(state, "task_repository", None),
        max_chain_walk=host._MAX_CHAIN_WALK,
        diag=host._diag,
        record_stat=host._record_chain_snapshot_stat,
        cache_store=state.chain_cache_store,
        read_query_missing=host._READ_QUERY_MISSING,
    )
    state.lifecycle_read_service = service
    return service


def seed_runtime_lookup_task(host: Any, payload: dict | None, *, lookup_short: str | None = None):
    if not isinstance(payload, dict):
        return None
    uuid_str = str(payload.get("uuid") or "").strip()
    if not uuid_str:
        return None
    short = uuid_str[:8]
    service = lifecycle_read_service(host)
    observation = host._module("task_codec").DEFAULT_TASK_CODEC.decode_row(payload, source_query="on-modify lookup seed")
    task_obj = service.seed_lookup_task(observation, short_uuid=short)
    requested_short = str(lookup_short or "").strip()
    if requested_short and requested_short != short:
        task_obj = service.seed_lookup_task(task_obj, short_uuid=requested_short)
    entry = task_obj.get("entry")
    if short and entry:
        host._query_ctx_set("tw_get", f"{short}.entry", str(entry).strip())
    return task_obj.to_mapping()


def seed_runtime_lookup_tasks(host: Any, *tasks: dict | None) -> None:
    for task in tasks:
        seed_runtime_lookup_task(host, task)


def collect_prev_two(host: Any, current_task: dict, chain_by_link=None):
    from .integration_models import Absent, Found, Unavailable

    service = lifecycle_read_service(host)
    state = host._modify_runtime_state()
    read = service.collect_prev_two(
        current_task,
        get_chain_read=lambda chain_id: service.get_chain_read(chain_id),
        panel_chain_by_link=state.panel_chain_by_link,
        panel_chain_snapshot_loaded=state.panel_chain_snapshot_loaded,
        chain_by_link=chain_by_link,
    )
    if isinstance(read, Unavailable):
        raise RuntimeError(read.evidence.detail or "lifecycle predecessor read unavailable")
    if isinstance(read, Absent):
        return []
    if not isinstance(read, Found):
        raise RuntimeError("lifecycle predecessor read returned an invalid result")
    return list(read.value)


def export_chain_required(host: Any, seed_payload: dict, env=None):
    chain_id = seed_payload.get("chainID")
    if not chain_id:
        raise RuntimeError("ChainID is required (legacy chain traversal removed). Run chainID backfill, then retry.")
    if env is not None:
        raise RuntimeError("chain reads must use the invocation Taskwarrior repository")
    service = lifecycle_read_service(host)
    rows = service.get_chain_export(chain_id)
    if rows is None:
        raise RuntimeError(f"Chain export unavailable for chainID {chain_id}")
    return rows


__all__ = ("parse_extra_tokens", "lifecycle_read_service", "seed_runtime_lookup_task", "seed_runtime_lookup_tasks", "collect_prev_two", "export_chain_required")
