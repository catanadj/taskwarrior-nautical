from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Callable
from dataclasses import dataclass

try:
    from .task_codec import DEFAULT_TASK_CODEC
except ImportError:  # standalone hook bootstrap loader
    from nautical_core.task_codec import DEFAULT_TASK_CODEC  # type: ignore[no-redef]


@dataclass(slots=True)
class TaskHookResponse:
    task: dict[str, Any]
    sanitize: bool = False
    prof: Any | None = None


@dataclass(slots=True)
class ExitHookResponse:
    exit_code: int = 0
    feedback_message: str | None = None
    stats: dict[str, Any] | None = None


# Compatibility names retained for existing hook loaders and integrations.
HookJsonResult = TaskHookResponse
HookExitResult = ExitHookResponse
HookResponse = TaskHookResponse | ExitHookResponse


def emit_passthrough_json(task: Any) -> None:
    print(json.dumps(task if isinstance(task, dict) else {}, ensure_ascii=False), end='')
    try:
        sys.stdout.flush()
    except Exception:
        pass


def emit_task_json(task: dict[str, Any], *, sanitize: bool = False, core=None, prof=None) -> None:
    t_out = time.perf_counter()
    if sanitize and core is not None and getattr(core, 'SANITIZE_UDA', False):
        DEFAULT_TASK_CODEC.sanitize_task_mapping(task, max_len=core.SANITIZE_UDA_MAX_LEN)
    emit_passthrough_json(task)
    if prof is not None:
        prof.add_ms('stdout:emit', (time.perf_counter() - t_out) * 1000.0)


def read_stdin_text(max_bytes: int) -> tuple[bytes, str]:
    raw_bytes = sys.stdin.buffer.read(max_bytes + 1)
    raw_text = raw_bytes.decode("utf-8", errors="replace")
    return raw_bytes, raw_text


def decode_latest_task_from_raw(raw: str) -> dict | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    observations, _index, error = DEFAULT_TASK_CODEC.decode_leading_rows(
        raw,
        source_query="hook:panic-passthrough",
        max_objects=2,
    )
    if error or not observations:
        return None
    return observations[-1].to_mapping()


def redirect_stdout_to_devnull() -> None:
    try:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    except Exception:
        pass


def panic_passthrough(
    raw_input_text: str,
    parsed_task: Any,
    *,
    decode_latest_task_from_raw: Callable[[str], dict | None] | None = None,
) -> None:
    fallback: dict[str, Any] = {}
    task = parsed_task if isinstance(parsed_task, dict) else None
    if task is None and raw_input_text:
        if callable(decode_latest_task_from_raw):
            try:
                task = decode_latest_task_from_raw(raw_input_text)
            except Exception:
                task = None
        if task is None:
            try:
                task = globals()["decode_latest_task_from_raw"](raw_input_text)
            except Exception:
                task = None
    try:
        emit_passthrough_json(task if isinstance(task, dict) else fallback)
    except Exception:
        try:
            print('{}', end='')
        except Exception:
            pass
        try:
            sys.stdout.flush()
        except Exception:
            pass


def emit_json_result(result: TaskHookResponse, *, core=None) -> None:
    emit_task_json(result.task, sanitize=result.sanitize, core=core, prof=result.prof)


def emit_exit_result(result: ExitHookResponse, *, emit_exit_feedback, emit_stats_diag) -> int:
    stats = result.stats if isinstance(result.stats, dict) else {}
    emit_stats_diag(stats)
    if result.feedback_message:
        emit_exit_feedback(result.feedback_message)
    return int(result.exit_code)
