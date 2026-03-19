from __future__ import annotations

import json
from typing import Any, Callable


def import_child(
    obj: dict[str, Any],
    *,
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_import: float,
    is_lock_error: Callable[[str], bool],
    sleep: Callable[[float], None],
    random_uniform: Callable[[float, float], float],
) -> tuple[bool, str]:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    max_retries = 4
    last_err = ""
    for attempt in range(max_retries):
        ok, _out, err = run_task(
            task_cmd_prefix + ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            input_text=payload,
            timeout=timeout_import,
        )
        if ok:
            return True, ""
        last_err = err or ""
        if not is_lock_error(last_err):
            return False, last_err
        if attempt < max_retries - 1:
            base = 0.2 * (2 ** attempt)
            jitter = random_uniform(0.0, 0.1)
            sleep(base + jitter)
    return False, last_err


def parent_nextlink_state(
    parent_uuid: str,
    child_short: str,
    *,
    expected_prev: str | None,
    export_uuid: Callable[[str], dict[str, Any]],
) -> tuple[str, str]:
    if not parent_uuid or not child_short:
        return "invalid", "missing parent or child"
    res = export_uuid(parent_uuid)
    if res.get("retryable"):
        return "locked", "parent export locked"
    parent = res.get("obj") if isinstance(res, dict) else None
    if not parent:
        return "missing", "parent missing"
    current = (parent.get("nextLink") or "").strip()
    expected = (expected_prev or "").strip()
    if current == child_short:
        return "already", ""
    if expected:
        if current != expected:
            return "conflict", "parent nextLink changed"
    else:
        if current:
            return "conflict", "parent nextLink already set"
    return "ok", ""


def update_parent_nextlink(
    parent_uuid: str,
    child_short: str,
    *,
    expected_prev: str | None,
    lock_parent_nextlink: Callable[[str], Any],
    parent_nextlink_state_fn: Callable[[str, str, str | None], tuple[str, str]],
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_modify: float,
    retries_modify: int,
    retry_delay: float,
) -> tuple[bool, str]:
    if not parent_uuid or not child_short:
        return False, "missing parent or child"
    with lock_parent_nextlink(parent_uuid) as locked:
        if not locked:
            return False, "parent lock busy"
        state, msg = parent_nextlink_state_fn(parent_uuid, child_short, expected_prev)
        if state == "ok":
            ok, _out, err = run_task(
                task_cmd_prefix + [
                    "rc.hooks=off",
                    "rc.verbose=nothing",
                    f"uuid:{parent_uuid}",
                    "modify",
                    f"nextLink:{child_short}",
                ],
                timeout=timeout_modify,
                retries=retries_modify,
                retry_delay=retry_delay,
            )
            return ok, err or ""
        if state == "already":
            return True, ""
        return False, msg


def cleanup_orphan_child(
    child_uuid: str,
    *,
    spawn_intent_id: str = "",
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_modify: float,
    retries_modify: int,
    retry_delay: float,
    diag: Callable[[str], None],
) -> None:
    if not child_uuid:
        return
    ok, _out, err = run_task(
        task_cmd_prefix + [
            "rc.hooks=off",
            "rc.verbose=nothing",
            f"uuid:{child_uuid}",
            "modify",
            "status:deleted",
        ],
        timeout=timeout_modify,
        retries=retries_modify,
        retry_delay=retry_delay,
    )
    if not ok:
        if spawn_intent_id:
            diag(f"orphan cleanup failed (intent={spawn_intent_id} child={child_uuid[:8]}): {err}")
        else:
            diag(f"orphan cleanup failed (child={child_uuid[:8]}): {err}")
