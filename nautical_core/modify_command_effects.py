"""Typed Taskwarrior command effects for the on-modify hook."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .integration_models import TaskCommandResult


class DiagCounter(Protocol):
    def __call__(self, key: str, inc: float = 1) -> None: ...


class RunTaskRecorder(Protocol):
    def __call__(self, cmd: list[str], *, ok: bool, elapsed: float) -> None: ...


class TaskCommandExecutor(Protocol):
    def __call__(
        self,
        cmd: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
        timeout: float = 3.0,
        attempts: int = 2,
        retry_delay: float = 0.15,
        use_tempfiles: bool = False,
        purpose: str = "Nautical hook command",
    ) -> TaskCommandResult: ...


@dataclass(frozen=True, slots=True)
class CommandPorts:
    execute: TaskCommandExecutor
    purpose_bucket: Callable[[list[str]], str]
    diag_count: DiagCounter
    diag_record: RunTaskRecorder
    diag: Callable[[str], None]
    task_cmd_prefix: Callable[[], list[str]]


class CommandHost(Protocol):
    _run_task_diag_bucket: Callable[[list[str]], str]
    _diag_count: DiagCounter
    _diag_record_run_task: RunTaskRecorder
    _diag: Callable[[str], None]
    _task_cmd_prefix: Callable[[], list[str]]


def command_ports_for(host: CommandHost) -> CommandPorts:
    from .runtime_command import run_task_result as execute
    return CommandPorts(
        execute=execute,
        purpose_bucket=host._run_task_diag_bucket,
        diag_count=host._diag_count,
        diag_record=host._diag_record_run_task,
        diag=host._diag,
        task_cmd_prefix=host._task_cmd_prefix,
    )


def run_task_result(
    ports: CommandPorts,
    cmd: list[str],
    *,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    attempts: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
) -> TaskCommandResult:
    started = time.perf_counter()
    result = ports.execute(
        cmd,
        purpose=f"on-modify {ports.purpose_bucket(cmd)}",
        env=env,
        input_text=input_text,
        timeout=timeout,
        attempts=attempts,
        retry_delay=retry_delay,
        use_tempfiles=use_tempfiles,
    )
    elapsed = time.perf_counter() - started
    ports.diag_count("run_task_calls")
    ports.diag_count("run_task_seconds", elapsed)
    ports.diag_record(cmd, ok=result.ok, elapsed=elapsed)
    if not result.ok:
        ports.diag_count("run_task_failures")
    return result


def generate_child_uuid_candidate(
    ports: CommandPorts, env: Mapping[str, str]
) -> str:
    candidate = str(uuid.uuid4())
    while True:
        result = run_task_result(
            ports,
            ports.task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{candidate}", "count"],
            env=env,
            timeout=2.5,
            attempts=2,
        )
        if result.ok:
            if (result.stdout or "").strip() == "0":
                return candidate
            candidate = str(uuid.uuid4())
            continue
        ports.diag(f"uuid availability check failed (uuid={candidate[:8]}): {result.stderr.strip()}")
        return candidate


__all__ = ("CommandPorts", "command_ports_for", "run_task_result", "generate_child_uuid_candidate")
