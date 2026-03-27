from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nautical_core.exit_models import ExitEquivalentChildResult, ExitExportResult


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
) -> ExitExportResult:
    from nautical_core.exit_models import ExitExportResult

    if hook_support is not None:
        status = hook_support.export_uuid_status(
            run_task=run_task,
            task_cmd_prefix=task_cmd_prefix,
            uuid_str=uuid_str,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            is_lock_error=is_lock_error,
            tolerate_noisy_stdout=True,
        )
        return ExitExportResult(
            bool(getattr(status, "get", None) and status.get("exists")),
            bool(getattr(status, "get", None) and status.get("retryable")),
            str(status.get("err") or "") if getattr(status, "get", None) else "",
            status.get("obj") if getattr(status, "get", None) and isinstance(status.get("obj"), dict) else None,
        )
    if not uuid_str:
        return ExitExportResult(False, False, "missing uuid", None)
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
        return ExitExportResult(False, is_lock_error(err), err or "", None)
    try:
        obj = json.loads(out.strip() or "{}")
        if obj.get("uuid"):
            return ExitExportResult(True, False, "", obj)
        return ExitExportResult(False, False, "not found", None)
    except Exception:
        if uuid_str in out:
            return ExitExportResult(True, False, "", {"uuid": uuid_str})
        return ExitExportResult(False, False, "parse error", None)


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
) -> ExitEquivalentChildResult:
    from nautical_core.exit_models import ExitEquivalentChildResult

    if not isinstance(child, dict):
        return ExitEquivalentChildResult(False, False, "missing child", None)
    chain_id = (child.get("chainID") or child.get("chainid") or "").strip()
    link_no = child.get("link")
    if not chain_id or link_no in (None, ""):
        return ExitEquivalentChildResult(False, False, "missing chain slot", None)
    try:
        link_token = str(int(link_no))
    except Exception:
        return ExitEquivalentChildResult(False, False, "invalid link", None)

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
        return ExitEquivalentChildResult(False, is_lock_error(err), err or "", None)
    try:
        rows = json.loads(out.strip() or "[]")
    except Exception:
        return ExitEquivalentChildResult(False, False, "parse error", None)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return ExitEquivalentChildResult(False, False, "parse error", None)

    for wanted in ("pending", "waiting", "completed"):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if (row.get("status") or "").strip().lower() != wanted:
                continue
            if (row.get("uuid") or "").strip():
                return ExitEquivalentChildResult(True, False, "", row)
    for row in rows:
        if isinstance(row, dict) and (row.get("uuid") or "").strip():
            return ExitEquivalentChildResult(True, False, "", row)
    return ExitEquivalentChildResult(False, False, "not found", None)
