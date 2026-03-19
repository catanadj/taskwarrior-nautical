from __future__ import annotations

import json
from typing import Any, Callable


def short_uuid(value: str, *, core: Any) -> str:
    if core is not None and hasattr(core, "short_uuid"):
        try:
            return str(core.short_uuid(value) or "")
        except Exception:
            pass
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    return raw.split("-")[0] if "-" in raw else raw[:8]


def export_uuid(
    uuid_str: str,
    *,
    hook_support: Any,
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout: float,
    retries: int,
    retry_delay: float,
    is_lock_error: Callable[[str], bool],
) -> dict[str, Any]:
    if hook_support is not None:
        return hook_support.export_uuid_status(
            run_task=run_task,
            task_cmd_prefix=task_cmd_prefix,
            uuid_str=uuid_str,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            is_lock_error=is_lock_error,
            tolerate_noisy_stdout=True,
        )
    if not uuid_str:
        return {"exists": False, "retryable": False, "err": "missing uuid", "obj": None}
    ok, out, err = run_task(
        task_cmd_prefix + [
            "rc.hooks=off",
            "rc.json.array=off",
            "rc.verbose=nothing",
            "rc.color=off",
            f"uuid:{uuid_str}",
            "export",
        ],
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    if not ok:
        return {"exists": False, "retryable": is_lock_error(err), "err": err or "", "obj": None}
    try:
        obj = json.loads(out.strip() or "{}")
        if obj.get("uuid"):
            return {"exists": True, "retryable": False, "err": "", "obj": obj}
        return {"exists": False, "retryable": False, "err": "not found", "obj": None}
    except Exception:
        if uuid_str in out:
            return {"exists": True, "retryable": False, "err": "", "obj": {"uuid": uuid_str}}
        return {"exists": False, "retryable": False, "err": "parse error", "obj": None}


def existing_equivalent_child(
    child: dict[str, Any],
    *,
    parent_uuid: str = "",
    task_cmd_prefix: list[str],
    run_task: Callable[..., tuple[bool, str, str]],
    timeout: float,
    retries: int,
    retry_delay: float,
    is_lock_error: Callable[[str], bool],
    short_uuid_fn: Callable[[str], str],
) -> dict[str, Any]:
    if not isinstance(child, dict):
        return {"exists": False, "retryable": False, "err": "missing child", "obj": None}
    chain_id = (child.get("chainID") or child.get("chainid") or "").strip()
    link_no = child.get("link")
    if not chain_id or link_no in (None, ""):
        return {"exists": False, "retryable": False, "err": "missing chain slot", "obj": None}
    try:
        link_token = str(int(link_no))
    except Exception:
        return {"exists": False, "retryable": False, "err": "invalid link", "obj": None}

    prev_link = (child.get("prevLink") or "").strip()
    if not prev_link and parent_uuid:
        prev_link = short_uuid_fn(parent_uuid)

    cmd = task_cmd_prefix + [
        "rc.hooks=off",
        "rc.json.array=1",
        "rc.verbose=nothing",
        "rc.color=off",
        f"chainID:{chain_id}",
        f"link:{link_token}",
    ]
    if prev_link:
        cmd.append(f"prevLink:{prev_link}")
    cmd.extend(["status.not:deleted", "export"])

    ok, out, err = run_task(
        cmd,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )
    if not ok:
        return {"exists": False, "retryable": is_lock_error(err), "err": err or "", "obj": None}
    try:
        rows = json.loads(out.strip() or "[]")
    except Exception:
        return {"exists": False, "retryable": False, "err": "parse error", "obj": None}
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return {"exists": False, "retryable": False, "err": "parse error", "obj": None}

    for wanted in ("pending", "waiting", "completed"):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if (row.get("status") or "").strip().lower() != wanted:
                continue
            if (row.get("uuid") or "").strip():
                return {"exists": True, "retryable": False, "err": "", "obj": row}
    for row in rows:
        if isinstance(row, dict) and (row.get("uuid") or "").strip():
            return {"exists": True, "retryable": False, "err": "", "obj": row}
    return {"exists": False, "retryable": False, "err": "not found", "obj": None}
