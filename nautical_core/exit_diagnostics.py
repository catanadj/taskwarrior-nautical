"""Diagnostic formatting for the typed on-exit lifecycle drain."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any


def emit_outcome_diagnostics(
    outcomes: Iterable[Any],
    *,
    diagnostic: Callable[[str], None],
    limit: int,
) -> int:
    """Emit bounded per-intent reasons and return the suppressed count."""
    emitted = 0
    suppressed = 0
    for outcome in outcomes:
        reason = getattr(outcome, "reason", "")
        if not reason:
            continue
        if emitted < max(0, int(limit)):
            diagnostic(
                f"lifecycle intent {getattr(outcome, 'intent_id', '') or '(unstaged)'}: "
                f"{getattr(getattr(outcome, 'kind', None), 'value', 'unknown')}: {reason}"
            )
            emitted += 1
        else:
            suppressed += 1
    if suppressed:
        diagnostic(
            f"lifecycle diagnostics: suppressed {suppressed} additional intent results "
            f"(limit {max(0, int(limit))})"
        )
    return suppressed


def emit_drain_stats_diag(
    stats: Mapping[str, Any],
    *,
    startup_stats: Mapping[str, Any],
    diag_stats: Mapping[str, Any],
    diagnostic_block: Callable[..., None],
) -> None:
    """Render stable startup, drain, and Taskwarrior timing diagnostics."""
    startup = dict(startup_stats)
    if startup:
        diagnostic_block("on-exit startup", startup.items(), columns=2)
    drain_items = [
        ("entries_total", stats.get("entries_total", 0)),
        ("processed", stats.get("processed", 0)),
        ("errors", stats.get("errors", 0)),
        ("retry_released", stats.get("retry_released", 0)),
        ("manual_reviewed", stats.get("manual_reviewed", 0)),
        ("quarantined", stats.get("quarantined", 0)),
        ("conflicted", stats.get("conflicted", 0)),
        ("outbox_lock_failures", stats.get("outbox_lock_failures", 0)),
        ("diagnostics_suppressed", stats.get("diagnostics_suppressed", 0)),
        ("drain_ms", stats.get("drain_ms", 0)),
        ("presentation_ms", diag_stats.get("presentation_ms", 0)),
    ]
    diagnostic_block("on-exit drain", drain_items, columns=3)
    task_stats = {
        str(key): value
        for key, value in diag_stats.items()
        if str(key).startswith("run_task_calls")
        or str(key).startswith("run_task_failures")
        or str(key).startswith("run_task_seconds")
    }
    task_stats["run_task_seconds"] = round(float(task_stats.get("run_task_seconds", 0.0)), 4)
    diagnostic_block("on-exit task stats", task_stats.items(), columns=3)


def strict_feedback(stats: Any, *, enabled: bool) -> str | None:
    """Return the optional Taskwarrior-facing failure message for a drain."""
    if not enabled:
        return None
    errors = int(getattr(stats, "errors", 0) or 0)
    manual_reviewed = int(getattr(stats, "manual_reviewed", 0) or 0)
    outbox_lock_failures = int(getattr(stats, "outbox_lock_failures", 0) or 0)
    if not (errors or manual_reviewed or outbox_lock_failures):
        return None
    return (
        f"[nautical] on-exit: {manual_reviewed} manual-review intents, {errors} errors, "
        f"{outbox_lock_failures} outbox lock failures. Check nautical queue-status "
        "(set NAUTICAL_EXIT_STRICT=0 to disable)"
    )


__all__ = ("emit_outcome_diagnostics", "emit_drain_stats_diag", "strict_feedback")
