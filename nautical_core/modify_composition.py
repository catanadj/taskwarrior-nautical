"""Composition services for the on-modify hook."""

from __future__ import annotations

from typing import Any, Callable


class ModifyCompositionServices:
    """Bind on-modify effects to the hook's validated composition root."""

    typed_transition_handlers = True

    def __init__(self, host: Any, result_cls: Callable[..., Any]) -> None:
        self._host = host
        self._result_cls = result_cls

    def result(self, task, *, sanitize: bool):
        return self._result_cls(task=task, sanitize=sanitize)

    def has_nautical_fields(self, task):
        return self._host._task_has_nautical_fields(task, task)

    def load_core(self):
        self._host._load_core()

    def diag(self, message: str):
        self._host._diag(message)

    def fail_and_exit(self, title: str, message: str):
        self._host._fail_and_exit(title, message)

    def handle_non_completion(self, old, new, unit_of_work, transition=None):
        self._host._handle_non_completion_modify(
            old, new, unit_of_work, transition=transition
        )

    def handle_completion(self, old, new, unit_of_work, transition=None):
        return self._host._handle_completion_modify(
            old, new, unit_of_work, transition=transition
        )

    def handle_deleted(
        self, old, new, unit_of_work, transition=None, terminal_decision=None
    ):
        return self._host._handle_deleted_modify(
            old,
            new,
            unit_of_work,
            transition=transition,
            terminal_decision=terminal_decision,
        )


__all__ = ("ModifyCompositionServices",)
