from __future__ import annotations

from .hook_support import LookupResult


def collect_prev_two(
    current_task: dict,
    *,
    coerce_int,
    get_chain_export,
    panel_chain_by_link=None,
    panel_chain_snapshot_loaded: bool = False,
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
    if not chain_index and not panel_chain_snapshot_loaded:
        try:
            chain = get_chain_export(chain_id)
        except Exception:
            return []
        if not isinstance(chain, list):
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


def existing_next_lookup(
    parent_task: dict,
    next_no: int,
    *,
    export_uuid_short_cached,
    get_chain_export,
    snapshot_rows: list[dict] | None = None,
    snapshot_loaded: bool = False,
) -> LookupResult:
    """Return a tri-state result for the idempotent next-link lookup."""
    if not isinstance(parent_task, dict):
        return LookupResult.unavailable("parent task is not an object")

    rows = [
        row for row in (snapshot_rows or [])
        if isinstance(row, dict)
        and str(row.get("link") or "").strip() == str(int(next_no))
        and (row.get("status") or "").strip().lower() != "deleted"
    ]
    if rows:
        obj = _pick_existing_next(rows)
        return LookupResult.found(obj) if obj else LookupResult.absent()

    next_ref = (parent_task.get("nextLink") or "").strip()
    if next_ref:
        obj = export_uuid_short_cached(next_ref)
        if isinstance(obj, LookupResult):
            if obj.is_found:
                return obj
            if obj.is_unavailable:
                return obj
            obj = None
        if isinstance(obj, dict) and (obj.get("status") or "").strip().lower() != "deleted":
            return LookupResult.found(obj)

    chain_id = (parent_task.get("chainID") or "").strip()
    if not chain_id or snapshot_loaded:
        return LookupResult.absent()
    try:
        rows = get_chain_export(chain_id, extra=f"link:{int(next_no)} status.not:deleted")
    except Exception as exc:
        return LookupResult.unavailable(f"chain export failed: {exc}")
    if rows is None:
        return LookupResult.unavailable("chain export was unavailable")
    if not rows:
        return LookupResult.absent()

    obj = _pick_existing_next(rows)
    return LookupResult.found(obj) if obj else LookupResult.absent()


def _pick_existing_next(rows: list[dict]) -> dict | None:
    for st in ("pending", "waiting", "completed"):
        for row in rows:
            if (row.get("status") or "").strip().lower() == st:
                return row
    return rows[0] if rows else None
