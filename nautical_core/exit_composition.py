"""Typed composition adapter for the on-exit hook router."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .hook_context import HookRuntimeContext
from .on_exit_models import ExitDrainStats


class ExitServices:
    """Bind hook presentation callbacks without owning lifecycle behavior."""

    def __init__(
        self,
        result_cls: type,
        *,
        redirect_stdout: Callable[[], None],
        drain_outbox: Callable[[HookRuntimeContext], ExitDrainStats],
        strict_feedback: Callable[[ExitDrainStats], str | None],
    ) -> None:
        self._result_cls = result_cls
        self._redirect_stdout = redirect_stdout
        self._drain_outbox = drain_outbox
        self._strict_feedback = strict_feedback

    def redirect_stdout(self) -> None:
        self._redirect_stdout()

    def drain_outbox(self, runtime: HookRuntimeContext) -> ExitDrainStats:
        return self._drain_outbox(runtime)

    def strict_feedback(self, stats: ExitDrainStats) -> str | None:
        return self._strict_feedback(stats)

    def result(self, *, exit_code: int, feedback_message: str | None, stats: ExitDrainStats):
        return self._result_cls(
            exit_code=exit_code,
            feedback_message=feedback_message,
            stats=stats,
        )


__all__ = ("ExitServices",)
