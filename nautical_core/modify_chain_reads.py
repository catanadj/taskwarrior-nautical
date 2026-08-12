"""Compatibility adapters for lifecycle chain read decisions.

The production implementation lives in :mod:`lifecycle_read_service`.  This
module keeps the older function-shaped entry points available to tooling and
focused tests without retaining a second read algorithm.
"""

from __future__ import annotations

from .lifecycle_read_service import LifecycleReadService
from .hook_support import LookupResult


def _service(coerce_int) -> LifecycleReadService:
    return LifecycleReadService(
        coerce_int=coerce_int,
        parse_extra_tokens=lambda _extra: [],
        token_matcher=lambda _row, _token: True,
        read_query_get=lambda _kind, _key: object(),
        chain_cache_get=lambda _chain_id: None,
        export_chain_cached=lambda *_args: (),
        max_chain_walk=500,
    )


def collect_prev_two(
    current_task: dict,
    *,
    coerce_int,
    get_chain_export,
    panel_chain_by_link=None,
    panel_chain_snapshot_loaded: bool = False,
    chain_by_link: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """Delegate predecessor lookup to the lifecycle read service."""
    return _service(coerce_int).collect_prev_two(
        current_task,
        get_chain_export=get_chain_export,
        panel_chain_by_link=panel_chain_by_link,
        panel_chain_snapshot_loaded=panel_chain_snapshot_loaded,
        chain_by_link=chain_by_link,
    )


def existing_next_lookup(
    parent_task: dict,
    next_no: int,
    *,
    export_uuid_short_cached,
    get_chain_export,
    snapshot_rows: list[dict] | None = None,
    snapshot_loaded: bool = False,
) -> LookupResult:
    """Delegate successor lookup to the lifecycle read service."""
    return _service(lambda value, default=None: int(value) if str(value).isdigit() else default).existing_next_lookup(
        parent_task,
        next_no,
        export_uuid_short_cached=export_uuid_short_cached,
        get_chain_export=get_chain_export,
        snapshot_rows=snapshot_rows,
        snapshot_loaded=snapshot_loaded,
    )


__all__ = ("collect_prev_two", "existing_next_lookup")
