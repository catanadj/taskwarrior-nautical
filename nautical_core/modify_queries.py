from __future__ import annotations

from .task_models import TaskPayload
from .task_datetime import datetime_value, parser_for_host
from dataclasses import dataclass
from typing import Any

from .callback_ports import CallbackPort


@dataclass(frozen=True, slots=True)
class QueryPorts:
    root_uuid: CallbackPort
    tw_get_cached: CallbackPort
    dtparse: CallbackPort
    tolocal: CallbackPort
    cache_get: CallbackPort
    cache_set: CallbackPort
    diag_count: CallbackPort


def chain_root_and_age(
    task: TaskPayload,
    now_utc: Any,
    *,
    root_uuid_from: Any,
    tw_get_cached: Any,
    dtparse: Any,
    tolocal: Any,
) -> tuple[str, int | None]:
    try:
        root_short = root_uuid_from(task)
        age_days = None
        if root_short:
            root_entry = tw_get_cached(f"{root_short}.entry")
            entry_dt = dtparse(root_entry)
            if entry_dt:
                entry_local = tolocal(entry_dt).date()
                today_local = tolocal(now_utc).date()
                age_days = (today_local - entry_local).days
                if age_days < 0:
                    age_days = 0
        return root_short or "—", age_days
    except Exception:
        return "—", None


def format_root_and_age(task: TaskPayload, now_utc: Any, *, chain_root_and_age: Any) -> str:
    root_short, age_days = chain_root_and_age(task, now_utc)
    if not root_short or root_short == "—":
        return "—"
    if age_days is not None and age_days > 0:
        return f"{root_short} ▻ {age_days}d"
    return root_short


def cached_chain_root_and_age(ports: QueryPorts, task: TaskPayload, now_utc: Any) -> tuple[str, int | None]:
    """Resolve and cache chain root age within the current modify invocation."""
    try:
        cache_key = (ports.root_uuid(task), str(ports.tolocal(now_utc).date()))
    except Exception:
        cache_key = None
    if cache_key is not None:
        cached = ports.cache_get("chain_root_age", cache_key)
        if isinstance(cached, tuple) and len(cached) == 2:
            ports.diag_count("chain_root_age_cache_hits")
            return cached
        ports.diag_count("chain_root_age_cache_misses")
    result = chain_root_and_age(
        task,
        now_utc,
        root_uuid_from=ports.root_uuid,
        tw_get_cached=ports.tw_get_cached,
        dtparse=ports.dtparse,
        tolocal=ports.tolocal,
    )
    if cache_key is not None:
        ports.cache_set("chain_root_age", cache_key, result)
    return result


def cached_format_root_and_age(ports: QueryPorts, task: TaskPayload, now_utc: Any) -> str:
    """Format a cached chain root/age value for presentation consumers."""
    try:
        cache_key = (ports.root_uuid(task), str(ports.tolocal(now_utc).date()))
    except Exception:
        cache_key = None
    if cache_key is not None:
        cached = ports.cache_get("format_root_age", cache_key)
        if isinstance(cached, str):
            ports.diag_count("format_root_age_cache_hits")
            return cached
        ports.diag_count("format_root_age_cache_misses")
    result = format_root_and_age(
        task,
        now_utc,
        chain_root_and_age=lambda value, at: cached_chain_root_and_age(ports, value, at),
    )
    if cache_key is not None:
        ports.cache_set("format_root_age", cache_key, result)
    return result


def query_ports_for(host: Any) -> QueryPorts:
    return QueryPorts(
        root_uuid=host._module("modify_task_fields").root_uuid,
        tw_get_cached=lambda ref: host._module("modify_read_effects").tw_get_cached(
            host._module("modify_read_effects").tw_get_ports_for(host), ref
        ),
        dtparse=lambda value: datetime_value(parser_for_host(host), value),
        tolocal=host._tolocal,
        cache_get=host._query_ctx_get,
        cache_set=host._query_ctx_set,
        diag_count=host._diag_count,
    )
