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


def task_text(host: Any, args, *, env=None) -> str:
    cache_key = None
    if env is None and host._task_args_cacheable(args):
        cache_key = tuple(str(value) for value in args)
        cached = host._query_ctx_get("task_text", cache_key)
        if isinstance(cached, str):
            host._diag_count("task_text_cache_hits")
            return cached
        host._diag_count("task_text_cache_misses")
    result = run_task_result(
        host,
        host._task_cmd_prefix() + ["rc.hooks=off"] + list(args),
        env=(env or host.os.environ.copy()),
        timeout=3.0,
        retries=2,
    )
    output = result.stdout or ""
    if cache_key is not None:
        host._query_ctx_set("task_text", cache_key, output)
    return output


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


__all__ = ("run_task_result", "task_text", "reserve_child_uuid")
