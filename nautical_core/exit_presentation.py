"""Optional on-exit progress presentation for typed lifecycle events."""

from __future__ import annotations

import os
import sys
import time
from typing import Any


class ExitDrainProgress:
    """Render lifecycle drain events without participating in mutation."""

    def __init__(self, *, core: Any) -> None:
        self._core = core
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
        except Exception:
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
            stage = str(getattr(getattr(event, "stage", None), "value", getattr(event, "stage", "")))
            total = max(0, int(getattr(event, "total", 0) or 0))
            self._start(total if stage == "claimed" else 0)
            if self._progress is None or self._task_id is None:
                return
            completed = max(0, int(getattr(event, "completed", 0) or 0))
            outcome = self._bound_label(getattr(event, "outcome", ""))
            detail = self._bound_label(getattr(event, "detail", ""))
            description = "⚓ Nautical drain"
            if detail:
                description += f" · {detail}"
            elif outcome:
                description += f" · {outcome}"
            self._progress.update(
                self._task_id,
                completed=completed,
                description=description,
                refresh=False,
            )
        except Exception:
            return
        finally:
            self._presentation_seconds += time.perf_counter() - started

    def close(self) -> None:
        if self._progress is None:
            return
        try:
            self._progress.stop()
        except Exception:
            pass
        finally:
            self._progress = None
            self._task_id = None


__all__ = ("ExitDrainProgress",)
