from __future__ import annotations


def collect_prev_two(
    current_task: dict,
    *,
    coerce_int,
    get_chain_export,
    panel_chain_by_link=None,
    chain_by_link: dict[int, list[dict]] | None = None,
) -> list[dict]:
    """Return up to two previous tasks (older first) using chainID export only."""

    chain_id = (current_task.get("chainID") or "").strip()
    if not chain_id:
        return []

    cur_no = coerce_int(current_task.get("link"), None)
    if not cur_no or cur_no <= 1:
        return []

    def _pick_best(candidates: list[dict]) -> dict | None:
        if not candidates:
            return None
        for st in ("pending", "completed", "deleted"):
            for task in candidates:
                if (task.get("status") or "").strip().lower() == st:
                    return task
        return candidates[0]

    chain_index = chain_by_link
    if chain_index is None:
        if panel_chain_by_link:
            chain_index = panel_chain_by_link
        else:
            chain_index = {}
    if not chain_index:
        try:
            chain = get_chain_export(chain_id)
        except Exception:
            return []
        chain_index = {}
        for task in chain:
            link_no = coerce_int(task.get("link"), None)
            if link_no is None:
                continue
            chain_index.setdefault(link_no, []).append(task)

    prevs: list[dict] = []
    for want in (cur_no - 2, cur_no - 1):
        if want < 1:
            continue
        obj = _pick_best(chain_index.get(want, []))
        if obj:
            prevs.append(obj)
    return prevs


def existing_next_task(
    parent_task: dict,
    next_no: int,
    *,
    export_uuid_short_cached,
    get_chain_export,
) -> dict | None:
    """Return an existing next-link task for idempotent re-completion handling."""
    if not isinstance(parent_task, dict):
        return None

    next_ref = (parent_task.get("nextLink") or "").strip()
    if next_ref:
        obj = export_uuid_short_cached(next_ref)
        if isinstance(obj, dict) and (obj.get("status") or "").strip().lower() != "deleted":
            return obj

    chain_id = (parent_task.get("chainID") or parent_task.get("chainid") or "").strip()
    if not chain_id:
        return None
    try:
        rows = get_chain_export(chain_id, extra=f"link:{int(next_no)} status.not:deleted")
    except Exception:
        rows = []
    if not rows:
        return None

    for st in ("pending", "waiting", "completed"):
        for row in rows:
            if (row.get("status") or "").strip().lower() == st:
                return row
    return rows[0]
