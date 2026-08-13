"""Read-only parent-link classification for the on-exit adapter.

All Taskwarrior mutations are owned by ``taskwarrior_mutations``.  This
module intentionally contains no subprocess or mutation helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from .integration_models import Found, Unavailable

if TYPE_CHECKING:
    from nautical_core.exit_models import ExitParentNextlinkStateResult
    from nautical_core.integration_models import TaskRead


def parent_nextlink_state(
    parent_uuid: str,
    child_short: str,
    *,
    expected_prev: str | None,
    export_uuid: Callable[[str], TaskRead[dict[str, Any]]],
    parent_guard: dict[str, Any] | None = None,
    guard_mismatch_fn: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
) -> ExitParentNextlinkStateResult:
    """Classify the current parent pointer without acquiring mutation access."""
    from nautical_core.exit_models import ExitParentNextlinkStateResult

    if not parent_uuid or not child_short:
        return ExitParentNextlinkStateResult("invalid", "missing parent or child")
    result = export_uuid(parent_uuid)
    if isinstance(result, Unavailable):
        return ExitParentNextlinkStateResult("locked", result.evidence.detail or "parent export unavailable")
    parent = result.value if isinstance(result, Found) else None
    if not parent:
        return ExitParentNextlinkStateResult("missing", "parent missing")
    if parent_guard and guard_mismatch_fn is not None:
        mismatch = guard_mismatch_fn(parent, parent_guard)
        if mismatch:
            return ExitParentNextlinkStateResult("conflict", mismatch)
    current = str(parent.get("nextLink") or "").strip()
    expected = str(expected_prev or "").strip()
    if current == child_short:
        return ExitParentNextlinkStateResult("already", "")
    if expected:
        if current != expected:
            return ExitParentNextlinkStateResult("conflict", "parent nextLink changed")
    elif current:
        return ExitParentNextlinkStateResult("conflict", "parent nextLink already set")
    return ExitParentNextlinkStateResult("ok", "")
