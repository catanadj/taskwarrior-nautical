from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nautical_core.exit_models import (
        ExitExportResult,
        ExitImportResult,
        ExitParentNextlinkStateResult,
        ExitParentUpdateResult,
    )


def _typed_result(run_task, cmd, *, input_text=None, timeout: float, retries: int = 1, retry_delay: float = 0.0):
    """Use one command-result adapter for typed and legacy runners."""
    from . import hook_support

    return hook_support.run_task_result(
        run_task=run_task,
        cmd=cmd,
        input_text=input_text,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )


def import_child(
    obj: dict[str, Any],
    *,
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_import: float,
    is_lock_error: Callable[[str], bool],
    sleep: Callable[[float], None],
    random_uniform: Callable[[float, float], float],
) -> ExitImportResult:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    max_retries = 4
    last_err = ""
    for attempt in range(max_retries):
        result = _typed_result(
            run_task,
            task_cmd_prefix + ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
            input_text=payload,
            timeout=timeout_import,
        )
        if result.ok:
            from nautical_core.exit_models import ExitImportResult
            return ExitImportResult(True, "")
        last_err = result.stderr or ""
        if result.kind != "lock_busy" and not is_lock_error(last_err):
            from nautical_core.exit_models import ExitImportResult
            return ExitImportResult(False, last_err)
        if attempt < max_retries - 1:
            base = 0.2 * (2 ** attempt)
            jitter = random_uniform(0.0, 0.1)
            sleep(base + jitter)
    from nautical_core.exit_models import ExitImportResult
    return ExitImportResult(False, last_err)


def import_children(
    children: list[dict[str, Any]],
    *,
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_import: float,
) -> ExitImportResult:
    """Import a bounded deterministic child batch in one Taskwarrior call."""
    from nautical_core.exit_models import ExitImportResult

    if not children:
        return ExitImportResult(True, "")
    payload = "".join(json.dumps(child, ensure_ascii=False, separators=(",", ":")) + "\n" for child in children)
    result = _typed_result(
        run_task,
        task_cmd_prefix + ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
        input_text=payload,
        timeout=timeout_import,
    )
    return ExitImportResult(bool(result.ok), result.stderr or "")


def parent_nextlink_state(
    parent_uuid: str,
    child_short: str,
    *,
    expected_prev: str | None,
    export_uuid: Callable[[str], ExitExportResult],
    parent_guard: dict[str, Any] | None = None,
    guard_mismatch_fn: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
) -> ExitParentNextlinkStateResult:
    if not parent_uuid or not child_short:
        from nautical_core.exit_models import ExitParentNextlinkStateResult
        return ExitParentNextlinkStateResult("invalid", "missing parent or child")
    res = export_uuid(parent_uuid)
    if res.retryable:
        from nautical_core.exit_models import ExitParentNextlinkStateResult
        return ExitParentNextlinkStateResult("locked", "parent export locked")
    parent = res.obj
    if not parent:
        from nautical_core.exit_models import ExitParentNextlinkStateResult
        return ExitParentNextlinkStateResult("missing", "parent missing")
    if parent_guard and guard_mismatch_fn is not None:
        mismatch = guard_mismatch_fn(parent, parent_guard)
        if mismatch:
            return ExitParentNextlinkStateResult("conflict", mismatch)
    current = (parent.get("nextLink") or "").strip()
    expected = (expected_prev or "").strip()
    if current == child_short:
        from nautical_core.exit_models import ExitParentNextlinkStateResult
        return ExitParentNextlinkStateResult("already", "")
    if expected:
        if current != expected:
            from nautical_core.exit_models import ExitParentNextlinkStateResult
            return ExitParentNextlinkStateResult("conflict", "parent nextLink changed")
    else:
        if current:
            from nautical_core.exit_models import ExitParentNextlinkStateResult
            return ExitParentNextlinkStateResult("conflict", "parent nextLink already set")
    from nautical_core.exit_models import ExitParentNextlinkStateResult
    return ExitParentNextlinkStateResult("ok", "")


def update_parent_nextlink(
    parent_uuid: str,
    child_short: str,
    *,
    expected_prev: str | None,
    lock_parent_nextlink: Callable[[str], Any],
    parent_nextlink_state_fn: Callable[[str, str, str | None], ExitParentNextlinkStateResult],
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_modify: float,
    retries_modify: int,
    retry_delay: float,
    parent_guard: dict[str, Any] | None = None,
    guard_mismatch_fn: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
) -> ExitParentUpdateResult:
    from nautical_core.exit_models import ExitParentUpdateResult

    if not parent_uuid or not child_short:
        return ExitParentUpdateResult(False, "missing parent or child")
    with lock_parent_nextlink(parent_uuid) as locked:
        if not locked:
            return ExitParentUpdateResult(False, "parent lock busy")
        if parent_guard is None and guard_mismatch_fn is None:
            state_res = parent_nextlink_state_fn(parent_uuid, child_short, expected_prev)
        else:
            state_res = parent_nextlink_state_fn(
                parent_uuid,
                child_short,
                expected_prev,
                parent_guard=parent_guard,
                guard_mismatch_fn=guard_mismatch_fn,
            )
        if state_res.state == "ok":
            # Keep the optimistic read and Taskwarrior mutation coupled. The
            # filesystem lock serializes Nautical writers, while this selector
            # also protects against an external Taskwarrior writer changing
            # nextLink between the export and modify commands.
            expected_filter = f"nextLink:{(expected_prev or '').strip()}"
            guard_filters: list[str] = []
            for field in ("status", "chain", "chainID", "link", "modified"):
                value = str((parent_guard or {}).get(field) or "").strip()
                if value:
                    guard_filters.append(f"{field}:{value}")
            result = _typed_result(
                run_task,
                task_cmd_prefix + [
                    "rc.hooks=off",
                    "rc.verbose=nothing",
                    f"uuid:{parent_uuid}",
                    expected_filter,
                    *guard_filters,
                    "modify",
                    f"nextLink:{child_short}",
                ],
                timeout=timeout_modify,
                retries=retries_modify,
                retry_delay=retry_delay,
            )
            if result.ok:
                return ExitParentUpdateResult(True, "", "ok")

            # Taskwarrior versions differ in how a selector no-match is
            # reported. Re-read once after a failed modify so a mutation that
            # landed despite a non-zero status is accepted, while stale or
            # conflicting parents remain fail-closed.
            try:
                post_state = parent_nextlink_state_fn(
                    parent_uuid,
                    child_short,
                    expected_prev,
                    parent_guard=parent_guard,
                    guard_mismatch_fn=guard_mismatch_fn,
                )
            except TypeError:
                post_state = None
            if post_state is not None:
                if post_state.state == "already":
                    return ExitParentUpdateResult(True, "", "already")
                if post_state.state in {"locked", "conflict", "missing", "invalid"}:
                    return ExitParentUpdateResult(False, post_state.err, post_state.state)
            return ExitParentUpdateResult(False, result.stderr or "parent update failed", "failed")
        if state_res.state == "already":
            return ExitParentUpdateResult(True, "", "already")
        return ExitParentUpdateResult(False, state_res.err, state_res.state)


def clear_parent_nextlink_if_matches(
    parent_uuid: str,
    child_short: str,
    *,
    lock_parent_nextlink: Callable[[str], Any],
    export_uuid: Callable[[str], ExitExportResult],
    run_task: Callable[..., tuple[bool, str, str]],
    task_cmd_prefix: list[str],
    timeout_modify: float,
    retries_modify: int,
    retry_delay: float,
) -> ExitParentUpdateResult:
    """Clear an optimistic parent link without overwriting a concurrent change."""
    from nautical_core.exit_models import ExitParentUpdateResult

    if not parent_uuid or not child_short:
        return ExitParentUpdateResult(False, "missing parent or child")
    with lock_parent_nextlink(parent_uuid) as locked:
        if not locked:
            return ExitParentUpdateResult(False, "parent lock busy")
        parent_res = export_uuid(parent_uuid)
        if parent_res.retryable:
            return ExitParentUpdateResult(False, parent_res.err or "parent export locked")
        parent = parent_res.obj
        if not parent:
            return ExitParentUpdateResult(True, "")
        current = str(parent.get("nextLink") or "").strip()
        if current != child_short:
            return ExitParentUpdateResult(True, "")
        result = _typed_result(
            run_task,
            task_cmd_prefix + [
                "rc.hooks=off",
                "rc.verbose=nothing",
                f"uuid:{parent_uuid}",
                f"nextLink:{child_short}",
                "modify",
                "nextLink:",
            ],
            timeout=timeout_modify,
            retries=retries_modify,
            retry_delay=retry_delay,
        )
        return ExitParentUpdateResult(result.ok, result.stderr or "")


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
    result = _typed_result(
        run_task,
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
    if not result.ok:
        if spawn_intent_id:
            diag(f"orphan cleanup failed (intent={spawn_intent_id} child={child_uuid[:8]}): {result.stderr}")
        else:
            diag(f"orphan cleanup failed (child={child_uuid[:8]}): {result.stderr}")
