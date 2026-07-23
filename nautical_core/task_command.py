#!/usr/bin/env python3
"""Bounded Taskwarrior command execution for Nautical operator tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import subprocess
import time
from typing import Any, Sequence


_LOCK_MARKERS = (
    "database is locked",
    "unable to lock",
    "resource temporarily unavailable",
    "another task is running",
    "lock file",
    "lockfile",
    "locked by",
    "timeout",
)


@dataclass(frozen=True, slots=True)
class TaskCommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    kind: str
    attempts: int
    timeout: float

    @property
    def ok(self) -> bool:
        return self.kind == "ok"


def _text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


def _is_lock_error(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in _LOCK_MARKERS)


def run_task_command(
    task_bin: str,
    args: Sequence[str],
    *,
    input_text: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = 60.0,
    retry_locks: bool = False,
    retry_delay: float = 0.1,
) -> TaskCommandResult:
    """Run one Taskwarrior command and classify failures without raising."""
    argv = (task_bin, *(str(arg) for arg in args))
    max_attempts = 2 if retry_locks else 1
    result: TaskCommandResult | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            proc = subprocess.run(
                list(argv),
                input=input_text,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                env=dict(env) if env is not None else os.environ.copy(),
                timeout=timeout,
            )
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            if proc.returncode == 0:
                kind = "ok"
            elif _is_lock_error(stderr or stdout):
                kind = "lock_busy"
            else:
                kind = "nonzero"
            result = TaskCommandResult(argv, proc.returncode, stdout, stderr, kind, attempt, timeout)
        except subprocess.TimeoutExpired as exc:
            result = TaskCommandResult(
                argv,
                124,
                _text(exc.stdout),
                _text(exc.stderr),
                "timeout",
                attempt,
                timeout,
            )
        except FileNotFoundError:
            result = TaskCommandResult(argv, 127, "", "", "missing_binary", attempt, timeout)
        except OSError as exc:
            result = TaskCommandResult(argv, 126, "", str(exc), "exec_error", attempt, timeout)

        if result.kind != "lock_busy" or attempt >= max_attempts:
            return result
        time.sleep(max(0.0, retry_delay))

    assert result is not None
    return result


def failure_message(result: Any, operation: str) -> str:
    """Return a stable, actionable message for command-like results."""
    kind = str(getattr(result, "kind", "") or "")
    if kind == "timeout":
        timeout = float(getattr(result, "timeout", 0.0) or 0.0)
        return f"{operation} timed out after {timeout:g}s"
    if kind == "missing_binary":
        argv = getattr(result, "argv", ())
        binary = str(argv[0]) if argv else "task"
        return f"Taskwarrior executable was not found: {binary}"
    detail = str(getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()
    if detail:
        return detail
    returncode = int(getattr(result, "returncode", 1) or 1)
    return f"{operation} failed with exit code {returncode}"


def load_json_result(result: Any, operation: str, *, empty: Any) -> Any:
    """Validate success and decode a command's JSON stdout."""
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError(failure_message(result, operation))
    raw = str(getattr(result, "stdout", "") or "").strip()
    if not raw:
        return empty
    try:
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{operation} returned invalid JSON: {exc}") from exc
