"""Optional on-exit progress presentation for typed lifecycle events."""

from __future__ import annotations

import os
import sys
import time
from typing import Any

from .lifecycle_models import LifecycleDrainProgress
from .operator_presentation import ProgressView


class ExitDrainProgress:
    """Render lifecycle drain events without participating in mutation."""

    def __init__(self, *, core: Any, diagnostic=None) -> None:
        self._core = core
        self._diagnostic = diagnostic
        self._progress: Any = None
        self._task_id: Any = None
        self._enabled = self._is_enabled()
        self._presentation_seconds = 0.0

    @property
    def presentation_ms(self) -> float:
        return round(self._presentation_seconds * 1000.0, 3)

    @staticmethod
    def _bound_label(value: object, *, limit: int = 72) -> str:
        text = str(value or "").replace("_", " ").strip()
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    def _is_enabled(self) -> bool:
        if not sys.stderr.isatty() or os.environ.get("TERM", "").strip().lower() == "dumb":
            return False
        raw = os.environ.get("NAUTICAL_EXIT_PROGRESS", "").strip().lower()
        if raw in {"0", "false", "no", "off"}:
            return False
        if raw in {"1", "true", "yes", "on"}:
            return True
        return bool(getattr(self._core, "EXIT_PROGRESS", True))

    def _start(self, total: int) -> None:
        if not self._enabled or total < 2 or self._progress is not None:
            return
        progress = None
        try:
            from rich.console import Console
            from rich.progress import (
                BarColumn,
                Progress,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            console = Console(file=sys.stderr, force_terminal=True)
            progress = Progress(
                TextColumn("[bold cyan]{task.description}[/]"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
                transient=True,
                refresh_per_second=8,
                redirect_stdout=False,
                redirect_stderr=False,
            )
            progress.start()
            self._progress = progress
            self._task_id = progress.add_task("⚓ Nautical drain", total=total)
        except Exception as exc:
            if self._diagnostic is not None:
                self._diagnostic(f"exit progress startup failed: {type(exc).__name__}: {exc}")
            if progress is not None:
                try:
                    progress.stop()
                except Exception:
                    pass
            self._progress = None
            self._task_id = None

    def on_event(self, event: object) -> None:
        started = time.perf_counter()
        try:
            view = ProgressView.from_event(event) if isinstance(event, LifecycleDrainProgress) else None
            if view is None:
                raise TypeError("drain progress event is not a LifecycleDrainProgress")
            stage = view.stage
            total = view.total
            self._start(total if stage == "claimed" else 0)
            if self._progress is None or self._task_id is None:
                return
            completed = view.completed
            detail = self._bound_label(view.label)
            description = "⚓ Nautical drain"
            if detail:
                description += f" · {detail}"
            self._progress.update(
                self._task_id,
                completed=completed,
                description=description,
                refresh=False,
            )
        except Exception as exc:
            if self._diagnostic is not None:
                self._diagnostic(f"exit progress update failed: {type(exc).__name__}: {exc}")
            return
        finally:
            self._presentation_seconds += time.perf_counter() - started

    def close(self) -> None:
        if self._progress is None:
            return
        try:
            self._progress.stop()
        except Exception as exc:
            if self._diagnostic is not None:
                self._diagnostic(f"exit progress shutdown failed: {type(exc).__name__}: {exc}")
            pass
        finally:
            self._progress = None
            self._task_id = None


def render_drain_failure_panel(core: Any, stats: dict[str, Any]) -> None:
    """Render actionable drain findings without changing lifecycle state."""
    if not isinstance(stats, dict) or core is None:
        return

    def count(key: str) -> int:
        try:
            return max(0, int(stats.get(key, 0) or 0))
        except Exception:
            return 0

    errors = count("errors")
    manual_reviewed = count("manual_reviewed")
    quarantined = count("quarantined")
    if not (errors or manual_reviewed or quarantined):
        return

    problems = []
    if manual_reviewed:
        problems.append(f"{manual_reviewed} manual-review intents")
    if quarantined:
        suffix = "" if quarantined == 1 else "s"
        problems.append(f"{quarantined} quarantined intent{suffix}")
    other_errors = max(0, errors - manual_reviewed - quarantined)
    if other_errors:
        suffix = "" if other_errors == 1 else "s"
        problems.append(f"{other_errors} other drain error{suffix}")

    rows = [("Action", "Run nautical queue-status"), ("Problems", "; ".join(problems) or f"{errors} drain errors")]
    if manual_reviewed or quarantined:
        rows.append(("Review", "Run nautical queue-status"))
    retry_released = count("retry_released")
    if retry_released:
        rows.append(("Retrying", str(retry_released)))
    outbox_lock_failures = count("outbox_lock_failures")
    if outbox_lock_failures:
        rows.append(("Lock events", str(outbox_lock_failures)))

    core.render_panel(
        "⚠ Nautical spawn drain failed",
        rows,
        kind="warning",
        panel_mode=core.PANEL_MODE,
        live_duration_ms=getattr(core, "LIVE_PANEL_DURATION_MS", 160),
        live_footer=getattr(core, "LIVE_PANEL_FOOTER", "NAUTICAL"),
        fast_color=core.FAST_COLOR,
        themes=core.panel_themes(),
        allow_line=True,
        label_width_min=6,
        label_width_max=14,
    )


__all__ = ("ExitDrainProgress", "render_drain_failure_panel")
