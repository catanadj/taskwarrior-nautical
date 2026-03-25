from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable


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
