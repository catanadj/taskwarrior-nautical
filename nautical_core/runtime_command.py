"""Lazy Taskwarrior command execution for the Nautical runtime."""

from __future__ import annotations

import os
import random
import subprocess
import tempfile
import time

from .task_command import TaskCommandResult


def _run_task_should_retry(attempt: int, retries: int) -> bool:
    return attempt < retries


def _run_task_retry_sleep(attempt: int, retry_delay: float) -> None:
    delay = retry_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, retry_delay) if retry_delay > 0 else 0.0
    time.sleep(delay + jitter)


def _run_task_failure_retryable(stderr: str) -> bool:
    """Retry only transient lock/timeout failures, never ordinary errors."""
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
    """Run a subprocess; returns ``(ok, stdout, stderr)``."""
    env = env or os.environ.copy()
    last_out = ""
    last_err = ""
    attempts = max(1, int(retries))
    normalized_input = input_text
    for attempt in range(1, attempts + 1):
        out_f, err_f, out_path, err_path = None, None, None, None
        try:
            out_f, err_f, out_path, err_path = _run_task_prepare_tempfiles(use_tempfiles)
            text_mode = not bool(out_f)
            normalized_input = _run_task_normalize_input(input_text, text_mode)
            if out_f is None:
                from . import hook_support

                ok, out, err = hook_support.run_subprocess_once(
                    cmd,
                    env=env,
                    input_text=normalized_input,
                    timeout=timeout,
                )
                out, err = _run_task_collect_outputs(out_f, err_f, out_path, err_path, out, err)
                last_out = out or ""
                last_err = err or ""
                if ok:
                    return True, last_out, last_err
                if _run_task_should_retry(attempt, retries) and _run_task_failure_retryable(last_err):
                    _run_task_retry_sleep(attempt, retry_delay)
                    continue
                return False, last_out, last_err

            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=out_f,
                stderr=err_f,
                text=False,
                close_fds=True,
                env=env,
            )
            try:
                out_bytes, err_bytes = proc.communicate(input=normalized_input, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    out_bytes, err_bytes = proc.communicate(timeout=1.0)
                except Exception:
                    out_bytes, err_bytes = b"", b""
                out, err = _run_task_collect_outputs(
                    out_f, err_f, out_path, err_path, out_bytes, err_bytes
                )
                last_err = "timeout"
                if _run_task_should_retry(attempt, retries):
                    _run_task_retry_sleep(attempt, retry_delay)
                    continue
                return False, out or "", last_err
            out, err = _run_task_collect_outputs(
                out_f, err_f, out_path, err_path, out_bytes, err_bytes
            )
            last_out = out or ""
            last_err = err or ""
            if proc.returncode == 0:
                return True, last_out, last_err
            if _run_task_should_retry(attempt, retries) and _run_task_failure_retryable(last_err):
                _run_task_retry_sleep(attempt, retry_delay)
                continue
            return False, last_out, last_err
        except Exception as e:
            last_err = str(e)
            _run_task_cleanup_paths(out_path, err_path)
            if _run_task_should_retry(attempt, retries) and _run_task_failure_retryable(last_err):
                _run_task_retry_sleep(attempt, retry_delay)
                continue
            return False, last_out, last_err
    return False, last_out, last_err


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
    """Expose the runtime runner through the shared typed command boundary."""
    ok, stdout, stderr = run_task(
        cmd,
        env=env,
        input_text=input_text,
        timeout=timeout,
        retries=retries,
        retry_delay=retry_delay,
        use_tempfiles=use_tempfiles,
    )
    text = (stderr or stdout or "").lower()
    if ok:
        kind = "ok"
        returncode = 0
    elif "timeout" in text:
        kind = "timeout"
        returncode = 124
    elif "lock" in text:
        kind = "lock_busy"
        returncode = 1
    else:
        kind = "nonzero"
        returncode = 1
    return TaskCommandResult(
        tuple(str(part) for part in cmd),
        returncode,
        stdout or "",
        stderr or "",
        kind,
        max(1, int(retries or 1)),
        float(timeout),
    )


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
