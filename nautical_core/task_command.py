#!/usr/bin/env python3
"""Taskwarrior operator command facade and JSON decoding helpers."""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .integration_models import CommandFailureKind, TaskCommandResult
from .taskwarrior_client import TaskwarriorClient


class TaskCommandFailure(RuntimeError):
    """A failed Taskwarrior command with its typed boundary evidence."""

    def __init__(self, result: TaskCommandResult, operation: str):
        self.result = result
        self.operation = str(operation or "Taskwarrior command")
        super().__init__(failure_message(result, self.operation))

    @property
    def retryable(self) -> bool:
        return self.result.kind in {CommandFailureKind.BUSY, CommandFailureKind.TIMEOUT}


def run_command_once(
    argv: Sequence[str],
    *,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    purpose: str = "Taskwarrior command",
) -> TaskCommandResult:
    normalized = tuple(str(arg) for arg in argv)
    if not normalized:
        raise ValueError("Taskwarrior command is empty")
    return TaskwarriorClient(normalized[:1], env=env).execute(
        normalized[1:],
        purpose=purpose,
        timeout=timeout,
        input_text=input_text,
    )


def run_task_command(
    task_bin: str,
    args: Sequence[str],
    *,
    input_text: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    retry_locks: bool = False,
    retry_delay: float = 0.1,
    purpose: str = "Taskwarrior command",
) -> TaskCommandResult:
    """Execute an operator command through the single process boundary."""
    return TaskwarriorClient((task_bin,), env=env).execute(
        args,
        purpose=purpose,
        timeout=timeout,
        input_text=input_text,
        attempts=2 if retry_locks else 1,
        retry_delay=retry_delay,
    )


def failure_message(result: TaskCommandResult, operation: str) -> str:
    """Return a stable, actionable message for a failed command."""
    if result.kind is CommandFailureKind.TIMEOUT:
        return f"{operation} timed out after {result.command.timeout:g}s"
    if result.kind is CommandFailureKind.MISSING_BINARY:
        return f"Taskwarrior executable was not found: {result.command.argv[0]}"
    detail = str(result.stderr or result.stdout or "").strip()
    if detail:
        return detail
    return f"{operation} failed with exit code {result.returncode}"


def load_json_result(result: TaskCommandResult, operation: str, *, empty: Any) -> Any:
    """Decode JSON after the process boundary has established success."""
    if not result.ok:
        raise TaskCommandFailure(result, operation)
    raw = result.stdout.strip()
    if not raw:
        return empty
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{operation} returned invalid JSON: {exc}") from exc


__all__ = (
    "CommandFailureKind",
    "TaskCommandFailure",
    "TaskCommandResult",
    "failure_message",
    "load_json_result",
    "run_command_once",
    "run_task_command",
)
