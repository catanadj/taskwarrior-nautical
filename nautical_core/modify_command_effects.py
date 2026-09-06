"""Typed Taskwarrior command effects for the on-modify hook."""

from __future__ import annotations

import time
import uuid
from typing import Any


def run_task_result(host: Any, cmd: list[str], **kwargs):
    from .runtime_command import run_task_result as execute

    started = time.perf_counter()
    result = execute(
        cmd,
        purpose=f"on-modify {host._run_task_diag_bucket(cmd)}",
        **kwargs,
    )
    elapsed = time.perf_counter() - started
    host._diag_count("run_task_calls")
    host._diag_count("run_task_seconds", elapsed)
    host._diag_record_run_task(cmd, ok=result.ok, elapsed=elapsed)
    if not result.ok:
        host._diag_count("run_task_failures")
    return result


def reserve_child_uuid(host: Any, env: dict) -> str:
    candidate = str(uuid.uuid4())
    while True:
        result = run_task_result(
            host,
            host._task_cmd_prefix() + ["rc.hooks=off", "rc.json.array=off", f"uuid:{candidate}", "count"],
            env=env,
            timeout=2.5,
            retries=2,
        )
        if result.ok:
            if (result.stdout or "").strip() == "0":
                return candidate
            candidate = str(uuid.uuid4())
            continue
        host._diag(f"uuid availability check failed (uuid={candidate[:8]}): {result.stderr.strip()}")
        return candidate


__all__ = ("run_task_result", "reserve_child_uuid")
