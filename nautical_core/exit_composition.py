"""Typed composition adapter for the on-exit hook router."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class ExitServices:
    """Bind hook presentation callbacks without owning lifecycle behavior."""

    def __init__(
        self,
        result_cls: type,
        *,
        redirect_stdout: Callable[[], None],
        drain_outbox: Callable[[Any], dict[str, Any]],
        strict_feedback: Callable[[dict[str, Any]], str | None],
    ) -> None:
        self._result_cls = result_cls
        self._redirect_stdout = redirect_stdout
        self._drain_outbox = drain_outbox
        self._strict_feedback = strict_feedback

    def redirect_stdout(self) -> None:
        self._redirect_stdout()

    def drain_outbox(self, unit_of_work: Any) -> dict[str, Any]:
        return self._drain_outbox(unit_of_work)

    def strict_feedback(self, stats: dict[str, Any]) -> str | None:
        return self._strict_feedback(stats)

    def result(self, *, exit_code: int, feedback_message: str | None, stats: dict[str, Any]):
        return self._result_cls(
            exit_code=exit_code,
            feedback_message=feedback_message,
            stats=stats,
        )


__all__ = ("ExitServices",)
