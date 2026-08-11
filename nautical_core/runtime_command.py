"""Lazy Taskwarrior command execution for the Nautical runtime."""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
import time

from .task_command import TaskCommandResult, run_command_once


def _run_task_should_retry(attempt: int, retries: int) -> bool:
    return attempt < retries


def _run_task_retry_sleep(attempt: int, retry_delay: float) -> None:
    delay = retry_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, retry_delay) if retry_delay > 0 else 0.0
    time.sleep(delay + jitter)


def _run_task_failure_retryable(stderr: str) -> bool:
    """Compatibility helper for legacy callers; typed callers use ``kind``."""
    return str(stderr or "").strip().lower() == "timeout" or is_lock_error(stderr or "")


def _run_task_prepare_tempfiles(use_tempfiles: bool):
    out_path = err_path = None
    out_f = err_f = None
    if use_tempfiles:
        try:
            out_f = tempfile.NamedTemporaryFile(delete=False)
            err_f = tempfile.NamedTemporaryFile(delete=False)
            out_path = out_f.name
            err_path = err_f.name
        except Exception:
            try:
                if out_f is not None:
                    out_f.close()
                    if out_f.name:
                        os.unlink(out_f.name)
            except Exception:
                pass
            try:
                if err_f is not None:
                    err_f.close()
                    if err_f.name:
                        os.unlink(err_f.name)
            except Exception:
                pass
            out_f = err_f = None
            out_path = err_path = None
    return out_f, err_f, out_path, err_path


def _run_task_normalize_input(input_text, text_mode: bool):
    if not text_mode and isinstance(input_text, str):
        return input_text.encode("utf-8")
    if text_mode and isinstance(input_text, (bytes, bytearray)):
        return input_text.decode("utf-8", "replace")
    return input_text


def _run_task_collect_outputs(out_f, err_f, out_path, err_path, out, err):
    try:
        if out_f is not None:
            out_f.close()
        if err_f is not None:
            err_f.close()
    except Exception:
        pass
    if out_path:
        try:
            with open(out_path, "rb") as f:
                out = f.read().decode("utf-8", "replace")
        except Exception:
            out = ""
        try:
            os.unlink(out_path)
        except Exception:
            pass
    if err_path:
        try:
            with open(err_path, "rb") as f:
                err = f.read().decode("utf-8", "replace")
        except Exception:
            err = ""
        try:
            os.unlink(err_path)
        except Exception:
            pass
    return out, err


def _run_task_cleanup_paths(out_path: str | None, err_path: str | None) -> None:
    try:
        if out_path and os.path.exists(out_path):
            os.unlink(out_path)
        if err_path and os.path.exists(err_path):
            os.unlink(err_path)
    except Exception:
        pass


def _run_task_once_with_tempfiles(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    attempt: int = 1,
) -> TaskCommandResult:
    """Run one command attempt while retaining typed failure evidence."""
    env_map = env or os.environ.copy()
    argv = tuple(str(part) for part in cmd)
    out_f, err_f, out_path, err_path = _run_task_prepare_tempfiles(True)
    if out_f is None:
        normalized_input = _run_task_normalize_input(input_text, True)
        result = run_command_once(
            argv,
            env=env_map,
            input_text=normalized_input,
            timeout=timeout,
        )
        return TaskCommandResult(
            result.argv,
            result.returncode,
            result.stdout,
            result.stderr,
            result.kind,
            attempt,
            timeout,
        )

    proc = None
    try:
        normalized_input = _run_task_normalize_input(input_text, False)
        proc = subprocess.Popen(
            list(argv),
            stdin=subprocess.PIPE,
            stdout=out_f,
            stderr=err_f,
            text=False,
            close_fds=True,
            env=env_map,
        )
        try:
            out_bytes, err_bytes = proc.communicate(input=normalized_input, timeout=timeout)
            out, err = _run_task_collect_outputs(
                out_f, err_f, out_path, err_path, out_bytes, err_bytes
            )
            return TaskCommandResult(
                argv,
                int(proc.returncode or 0),
                out or "",
                err or "",
                "ok" if proc.returncode == 0 else (
                    "lock_busy" if is_lock_error(err or "") else "nonzero"
                ),
                attempt,
                timeout,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                out_bytes, err_bytes = proc.communicate(timeout=1.0)
            except Exception:
                out_bytes, err_bytes = b"", b""
            out, err = _run_task_collect_outputs(
                out_f, err_f, out_path, err_path, out_bytes, err_bytes
            )
            return TaskCommandResult(argv, 124, out or "", err or "timeout", "timeout", attempt, timeout)
    except FileNotFoundError as exc:
        _run_task_cleanup_paths(out_path, err_path)
        return TaskCommandResult(argv, 127, "", str(exc), "missing_binary", attempt, timeout)
    except OSError as exc:
        _run_task_cleanup_paths(out_path, err_path)
        return TaskCommandResult(argv, 126, "", str(exc), "exec_error", attempt, timeout)
    except Exception as exc:
        _run_task_cleanup_paths(out_path, err_path)
        return TaskCommandResult(argv, 1, "", str(exc), "exec_error", attempt, timeout)


def run_task_result(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
) -> TaskCommandResult:
    """Run a command and return the authoritative typed result."""
    env_map = env or os.environ.copy()
    attempts = max(1, int(retries))
    last: TaskCommandResult | None = None
    for attempt in range(1, attempts + 1):
        if use_tempfiles:
            result = _run_task_once_with_tempfiles(
                cmd,
                env=env_map,
                input_text=input_text,
                timeout=timeout,
                attempt=attempt,
            )
        else:
            raw = run_command_once(
                tuple(str(part) for part in cmd),
                env=env_map,
                input_text=input_text,
                timeout=timeout,
            )
            result = TaskCommandResult(
                raw.argv,
                raw.returncode,
                raw.stdout,
                raw.stderr,
                raw.kind,
                attempt,
                timeout,
            )
        last = result
        if result.ok or result.kind not in {"timeout", "lock_busy"} or attempt >= attempts:
            return result
        _run_task_retry_sleep(attempt, retry_delay)
    assert last is not None
    return last


def run_task(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
    use_tempfiles: bool = False,
) -> tuple[bool, str, str]:
    """Compatibility view of :func:`run_task_result` for external callers."""
    result = run_task_result(
        cmd,
        env=env,
        input_text=input_text,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        use_tempfiles=use_tempfiles,
    )
    compatibility_stderr = result.stderr
    if result.kind == "timeout" and not compatibility_stderr:
        compatibility_stderr = "timeout"
    return result.ok, result.stdout, compatibility_stderr


def is_lock_error(err: str) -> bool:
    """Check if stderr indicates a Taskwarrior/database lock error."""
    e = (err or "").lower()
    return (
        "database is locked" in e
        or "unable to lock" in e
        or "resource temporarily unavailable" in e
        or "another task is running" in e
        or "lock file" in e
        or "lockfile" in e
        or "locked by" in e
        or "timeout" in e
    )


__all__ = (
    "run_task",
    "run_task_result",
    "is_lock_error",
    "_run_task_should_retry",
    "_run_task_retry_sleep",
    "_run_task_prepare_tempfiles",
    "_run_task_normalize_input",
    "_run_task_collect_outputs",
    "_run_task_cleanup_paths",
)
