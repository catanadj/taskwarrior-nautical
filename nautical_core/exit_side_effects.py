from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Callable

from .integration_models import CommandFailureKind

if TYPE_CHECKING:
    from nautical_core.exit_models import (
        ExitExportResult,
        ExitImportResult,
        ExitParentNextlinkStateCallback,
        ExitParentNextlinkStateResult,
        ExitParentUpdateResult,
    )
    from nautical_core.integration_models import TaskCommandResult


def _typed_result(run_task, cmd, *, input_text=None, timeout: float, retries: int = 1, retry_delay: float = 0.0):
    """Use one typed command-result callback."""
    from . import hook_support

    return hook_support.run_task_result(
        run_task=run_task,
        cmd=cmd,
        input_text=input_text,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
    )


def _retryable(result: TaskCommandResult) -> bool:
    return result.kind in {CommandFailureKind.BUSY, CommandFailureKind.TIMEOUT}


def import_child(
    obj: dict[str, Any],
    *,
    run_task: Callable[..., TaskCommandResult],
    task_cmd_prefix: list[str],
    timeout_import: float,
) -> ExitImportResult:
    payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
    result = _typed_result(
        run_task,
        task_cmd_prefix + ["rc.hooks=off", "rc.verbose=nothing", "import", "-"],
        input_text=payload,
        timeout=timeout_import,
        retries=4,
        retry_delay=0.2,
    )
    from nautical_core.exit_models import ExitImportResult
    if result.ok:
        return ExitImportResult(True, "")
    return ExitImportResult(False, result.stderr or result.kind.value, _retryable(result))


def import_children(
    children: list[dict[str, Any]],
    *,
    run_task: Callable[..., TaskCommandResult],
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
    return ExitImportResult(bool(result.ok), result.stderr or "", _retryable(result))


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
    parent_nextlink_state_fn: ExitParentNextlinkStateCallback,
    run_task: Callable[..., TaskCommandResult],
    task_cmd_prefix: list[str],
    timeout_modify: float,
    retries_modify: int,
    retry_delay: float,
    parent_guard: dict[str, Any] | None = None,
    guard_mismatch_fn: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
    parent_snapshot: dict[str, Any] | None = None,
) -> ExitParentUpdateResult:
    from nautical_core.exit_models import ExitParentUpdateResult

    if not parent_uuid or not child_short:
        return ExitParentUpdateResult(False, "missing parent or child")
    with lock_parent_nextlink(parent_uuid) as locked:
        if not locked:
            return ExitParentUpdateResult(False, "parent lock busy", retryable=True)
        if isinstance(parent_snapshot, dict):
            from nautical_core.exit_models import ExitParentNextlinkStateResult

            mismatch = ""
            if parent_guard is not None and guard_mismatch_fn is not None:
                mismatch = guard_mismatch_fn(parent_snapshot, parent_guard)
            if mismatch:
                state_res = ExitParentNextlinkStateResult("conflict", mismatch)
            else:
                current = str(parent_snapshot.get("nextLink") or "").strip()
                expected = str(expected_prev or "").strip()
                if current == child_short:
                    state_res = ExitParentNextlinkStateResult("already", "")
                elif (expected and current != expected) or (not expected and current):
                    state_res = ExitParentNextlinkStateResult("conflict", "parent nextLink changed")
                else:
                    state_res = ExitParentNextlinkStateResult("ok", "")
        elif parent_guard is None and guard_mismatch_fn is None:
            state_res = parent_nextlink_state_fn(parent_uuid, child_short, expected_prev)
        else:
            try:
                state_res = parent_nextlink_state_fn(
                    parent_uuid,
                    child_short,
                    expected_prev,
                    parent_guard=parent_guard,
                    guard_mismatch_fn=guard_mismatch_fn,
                )
            except TypeError:
                # Isolated fixtures may still provide the original narrow
                # callback; production services implement the typed protocol.
                state_res = parent_nextlink_state_fn(parent_uuid, child_short, expected_prev)
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
            post_state = None
            if parent_snapshot is None:
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
                    return ExitParentUpdateResult(
                        False,
                        post_state.err,
                        post_state.state,
                        post_state.state == "locked",
                    )
            return ExitParentUpdateResult(
                False,
                result.stderr or "parent update failed",
                "failed",
                _retryable(result),
            )
        if state_res.state == "already":
            return ExitParentUpdateResult(True, "", "already")
        return ExitParentUpdateResult(
            False,
            state_res.err,
            state_res.state,
            state_res.state == "locked",
        )


def clear_parent_nextlink_if_matches(
    parent_uuid: str,
    child_short: str,
    *,
    lock_parent_nextlink: Callable[[str], Any],
    export_uuid: Callable[[str], ExitExportResult],
    run_task: Callable[..., TaskCommandResult],
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
            return ExitParentUpdateResult(False, "parent lock busy", retryable=True)
        parent_res = export_uuid(parent_uuid)
        if parent_res.retryable:
            return ExitParentUpdateResult(
                False,
                parent_res.err or "parent export locked",
                retryable=True,
            )
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
        return ExitParentUpdateResult(
            result.ok,
            result.stderr or "",
            retryable=_retryable(result),
        )


def cleanup_orphan_child(
    child_uuid: str,
    *,
    spawn_intent_id: str = "",
    run_task: Callable[..., TaskCommandResult],
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


def cleanup_orphan_children(
    child_uuids: list[str],
    *,
    run_task: Callable[..., TaskCommandResult],
    task_cmd_prefix: list[str],
    timeout_modify: float,
    retries_modify: int,
    retry_delay: float,
) -> ExitImportResult:
    """Delete a bounded set of unlinked orphan children in one guarded command."""
    from nautical_core.exit_models import ExitImportResult

    uuids = []
    seen: set[str] = set()
    for value in child_uuids or []:
        token = str(value or "").strip()
        if token and token not in seen:
            seen.add(token)
            uuids.append(token)
    if not uuids:
        return ExitImportResult(True, "")
    selectors: list[str] = []
    for index, uuid_str in enumerate(uuids):
        if index:
            selectors.append("or")
        selectors.extend((f"uuid:{uuid_str}", "status.not:deleted"))
    result = _typed_result(
        run_task,
        task_cmd_prefix + [
            "rc.hooks=off",
            "rc.verbose=nothing",
            *selectors,
            "modify",
            "status:deleted",
        ],
        timeout=timeout_modify,
        retries=retries_modify,
        retry_delay=retry_delay,
    )
    return ExitImportResult(bool(result.ok), result.stderr or "", _retryable(result))
