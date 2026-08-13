"""Runtime facade for the single Taskwarrior process boundary."""

from __future__ import annotations

from typing import Mapping, Sequence

from .integration_models import CommandFailureKind, TaskCommandResult
from .taskwarrior_client import TaskwarriorClient


def run_task_result(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
    purpose: str = "Nautical hook command",
) -> TaskCommandResult:
    normalized = tuple(str(part) for part in cmd)
    if not normalized:
        raise ValueError("Taskwarrior command is empty")
    return TaskwarriorClient(normalized[:1], env=env).execute(
        normalized[1:],
        purpose=purpose,
        timeout=timeout,
        input_text=input_text,
        attempts=max(1, int(retries)),
        retry_delay=max(0.0, float(retry_delay)),
        use_tempfiles=use_tempfiles,
    )


def is_retryable_result(result: TaskCommandResult) -> bool:
    return result.kind in {CommandFailureKind.BUSY, CommandFailureKind.TIMEOUT}


__all__ = ("is_retryable_result", "run_task_result")
