"""Typed Taskwarrior command effects for the on-modify hook."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CommandPorts:
    execute: Any
    purpose_bucket: Any
    diag_count: Any
    diag_record: Any
    diag: Any
    task_cmd_prefix: Any


def command_ports_for(host: Any) -> CommandPorts:
    from .runtime_command import run_task_result as execute
    return CommandPorts(
        execute=execute,
        purpose_bucket=host._run_task_diag_bucket,
        diag_count=host._diag_count,
        diag_record=host._diag_record_run_task,
        diag=host._diag,
        task_cmd_prefix=host._task_cmd_prefix,
    )


def run_task_result(ports: CommandPorts, cmd: list[str], **kwargs: Any) -> Any:
    started = time.perf_counter()
    result = ports.execute(
        cmd,
        purpose=f"on-modify {ports.purpose_bucket(cmd)}",
        **kwargs,
    )
    elapsed = time.perf_counter() - started
    ports.diag_count("run_task_calls")
    ports.diag_count("run_task_seconds", elapsed)
    ports.diag_record(cmd, ok=result.ok, elapsed=elapsed)
    if not result.ok:
        ports.diag_count("run_task_failures")
    return result


def generate_child_uuid_candidate(ports: CommandPorts, env: dict) -> str:
    candidate = str(uuid.uuid4())
    while True:
        result = run_task_result(
            ports,
            ports.task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{candidate}", "count"],
            env=env,
            timeout=2.5,
            retries=2,
        )
        if result.ok:
            if (result.stdout or "").strip() == "0":
                return candidate
            candidate = str(uuid.uuid4())
            continue
        ports.diag(f"uuid availability check failed (uuid={candidate[:8]}): {result.stderr.strip()}")
        return candidate


__all__ = ("CommandPorts", "command_ports_for", "run_task_result", "generate_child_uuid_candidate")
