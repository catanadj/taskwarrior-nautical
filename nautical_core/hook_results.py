from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable
from dataclasses import dataclass


@dataclass(slots=True)
class HookJsonResult:
    task: dict[str, Any]
    sanitize: bool = False
    prof: Any | None = None


@dataclass(slots=True)
class HookExitResult:
    exit_code: int = 0
    feedback_message: str | None = None
    stats: dict[str, Any] | None = None


def emit_passthrough_json(task: Any) -> None:
    print(json.dumps(task if isinstance(task, dict) else {}, ensure_ascii=False), end='')
    try:
        sys.stdout.flush()
    except Exception:
        pass


def emit_task_json(task: dict[str, Any], *, sanitize: bool = False, core=None, prof=None) -> None:
    t_out = time.perf_counter()
    if sanitize and core is not None and getattr(core, 'SANITIZE_UDA', False):
        core.sanitize_task_strings(task, max_len=core.SANITIZE_UDA_MAX_LEN)
    emit_passthrough_json(task)
    if prof is not None:
        prof.add_ms('stdout:emit', (time.perf_counter() - t_out) * 1000.0)


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
                parsed = json.loads(raw_input_text.strip())
                if isinstance(parsed, dict):
                    task = parsed
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


def emit_json_result(result: HookJsonResult, *, core=None) -> None:
    emit_task_json(result.task, sanitize=result.sanitize, core=core, prof=result.prof)


def emit_exit_result(result: HookExitResult, *, emit_exit_feedback, emit_stats_diag) -> int:
    stats = result.stats if isinstance(result.stats, dict) else {}
    emit_stats_diag(stats)
    if result.feedback_message:
        emit_exit_feedback(result.feedback_message)
    return int(result.exit_code)
