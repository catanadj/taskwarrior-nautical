from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CompletionFinalizeServices:
    build_and_spawn_child: Any
    seed_runtime_lookup_tasks: Any
    modify_chain_state: Any
    get_chain_export: Any
    build_chain_indexes: Any
    set_chain_cache: Any
    export_uuid_short_cached: Any
    merge_spawned_child_into_chain: Any
    chain_health_advice: Any
    chain_integrity_warnings: Any
    render_anchor_completion_feedback: Any
    render_cp_completion_feedback: Any
    print_task: Any
    diag_summary: Any
    show_analytics: bool
    check_integrity: bool
    analytics_style: str


def finalize_completion_modify(
    *,
    new: dict,
    ctx,
    computed,
    now_utc,
    need_chain: bool,
    chain_snapshot_loaded: bool,
    preloaded_chain: list[dict],
    preloaded_chain_by_link,
    preloaded_chain_by_short,
    chain_id: str,
    services: CompletionFinalizeServices,
) -> None:
    parent_short = ctx.parent_short
    base_no = ctx.base_no
    next_no = ctx.next_no
    kind = ctx.kind
    spawned = services.build_and_spawn_child(
        new,
        child_due=computed.child_due,
        child_field="scheduled" if isinstance(computed.meta, dict) and computed.meta.get("target_field") == "scheduled" else "due",
        next_no=next_no,
        parent_short=parent_short,
        kind=kind,
        cpmax=computed.cpmax,
        until_dt=computed.until_dt,
    )
    if spawned is None:
        return

    child = spawned.child
    services.seed_runtime_lookup_tasks(new, child)
    child_short = spawned.child_short
    stripped_attrs = spawned.stripped_attrs
    deferred_spawn = spawned.deferred_spawn
    spawn_intent_id = spawned.spawn_intent_id

    chain = list(preloaded_chain)
    chain_by_link = preloaded_chain_by_link
    chain_by_short = preloaded_chain_by_short
    if chain_id:
        try:
            if chain and spawned.verified and not deferred_spawn:
                chain = services.merge_spawned_child_into_chain(chain, new, child, child_short)
                chain_by_link, chain_by_short = services.build_chain_indexes(chain)
                services.set_chain_cache(chain_id, chain)
                services.export_uuid_short_cached.cache_clear()
            elif need_chain and not chain_snapshot_loaded:
                chain = services.get_chain_export(chain_id)
                if chain:
                    chain_by_link, chain_by_short = services.build_chain_indexes(chain)
                    services.set_chain_cache(chain_id, chain)
                    services.export_uuid_short_cached.cache_clear()
        except Exception:
            pass

    state = services.modify_chain_state()
    state.panel_chain_by_link = chain_by_link
    state.panel_chain_by_short = chain_by_short
    state.panel_chain_snapshot_loaded = True

    analytics_advice = None
    integrity_warnings = None
    if chain and services.show_analytics:
        try:
            analytics_advice = services.chain_health_advice(chain, kind, new, style=services.analytics_style)
        except Exception:
            analytics_advice = None
    if chain and services.check_integrity:
        try:
            integrity_warnings = services.chain_integrity_warnings(chain, expected_chain_id=chain_id)
        except Exception:
            integrity_warnings = None

    if kind in {"anchor", "anchor_file"}:
        services.render_anchor_completion_feedback(
            new=new,
            child=child,
            child_due=computed.child_due,
            child_short=child_short,
            next_no=next_no,
            parent_short=parent_short,
            cap_no=computed.cap_no,
            finals=computed.finals,
            now_utc=now_utc,
            until_dt=computed.until_dt,
            until_cap_no=computed.until_cap_no,
            dnf=computed.dnf,
            meta=computed.meta,
            stripped_attrs=stripped_attrs,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
            chain_by_short=chain_by_short,
            analytics_advice=analytics_advice,
            integrity_warnings=integrity_warnings,
            base_no=base_no,
        )
    else:
        services.render_cp_completion_feedback(
            new=new,
            child=child,
            child_due=computed.child_due,
            child_short=child_short,
            next_no=next_no,
            parent_short=parent_short,
            cap_no=computed.cap_no,
            finals=computed.finals,
            now_utc=now_utc,
            until_dt=computed.until_dt,
            until_cap_no=computed.until_cap_no,
            meta=computed.meta,
            deferred_spawn=deferred_spawn,
            spawn_intent_id=spawn_intent_id,
            chain_by_short=chain_by_short,
            analytics_advice=analytics_advice,
            integrity_warnings=integrity_warnings,
            base_no=base_no,
        )

    services.print_task(new)
    services.diag_summary()


__all__ = (
    "CompletionFinalizeServices",
    "finalize_completion_modify",
)
