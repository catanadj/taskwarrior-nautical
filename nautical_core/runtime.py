from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import time

from . import _normalized_abspath, _validated_user_dir


DIAG_LOG_REDACT_KEYS: frozenset[str] = frozenset(
    {"description", "annotation", "annotations", "note", "notes"}
)


def hook_arg_value(argv: list[str], keys: tuple[str, ...]) -> str:
    for tok in argv:
        s = str(tok or "").strip()
        if not s:
            continue
        for key in keys:
            for sep in (":", "="):
                prefix = f"{key}{sep}"
                if s.startswith(prefix):
                    val = s[len(prefix):].strip()
                    if val:
                        return val
    return ""


def resolve_task_data_context(
    *,
    argv: list[str] | None = None,
    env: dict | None = None,
    tw_dir: str | None = None,
) -> tuple[str, bool, str]:
    """
    Resolve Taskwarrior data directory context for hooks.

    Returns: (task_data_dir, use_rc_data_location, source)
      - task_data_dir: resolved directory path (user-expanded)
      - use_rc_data_location: True only when source is explicit (argv/env)
      - source: one of "argv", "env", "fallback"
    """
    args = list(argv if argv is not None else sys.argv[1:])
    env_map = env if env is not None else os.environ
    taskdata_env = str((env_map.get("TASKDATA") if hasattr(env_map, "get") else "") or "").strip()
    taskdata_arg = hook_arg_value(args, ("data", "data.location"))
    explicit = taskdata_arg or taskdata_env
    if explicit:
        source = "argv" if taskdata_arg else "env"
        safe_explicit = _validated_user_dir(
            str(explicit),
            label=("rc.data.location" if taskdata_arg else "TASKDATA"),
            trust_env="NAUTICAL_TRUST_TASKDATA_PATH",
            env_map=env_map,
        )
        if safe_explicit:
            return safe_explicit, True, source
    base = str(tw_dir or "~/.task")
    safe_fallback = _validated_user_dir(
        base,
        label="fallback task data dir",
        trust_env="NAUTICAL_TRUST_TASKDATA_PATH",
        env_map=env_map,
        warn_on_error=False,
    )
    return (safe_fallback or _normalized_abspath(base)), False, "fallback"


def _redact_dict(data: dict, redact_keys: frozenset) -> dict:
    out = {}
    for k, v in (data or {}).items():
        if k in redact_keys:
            out[k] = "[redacted]"
        else:
            out[k] = v
    return out


def diag_log_redact(msg: str, redact_keys: frozenset | None = None):
    """Redact sensitive keys from JSON msg for diagnostic logs."""
    keys = redact_keys or DIAG_LOG_REDACT_KEYS
    if isinstance(msg, dict):
        return _redact_dict(msg, keys)
    try:
        data = json.loads(msg)
        if isinstance(data, dict):
            for k in list(data.keys()):
                if k in keys:
                    data[k] = "[redacted]"
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        pass
    return msg


def _diag_log_path(data_dir: str | None = None) -> str:
    base = data_dir or os.environ.get("TASKDATA")
    if base:
        safe_base = _validated_user_dir(
            str(base),
            label="diag data dir",
            trust_env="NAUTICAL_TRUST_TASKDATA_PATH",
            warn_on_error=False,
        )
        if safe_base:
            return os.path.join(safe_base, ".nautical_diag.jsonl")
    safe_default = _validated_user_dir(
        "~/.task",
        label="diag fallback dir",
        trust_env="NAUTICAL_TRUST_TASKDATA_PATH",
        warn_on_error=False,
    )
    return os.path.join((safe_default or _normalized_abspath("~/.task")), ".nautical_diag.jsonl")


def diag_log(msg: str, hook_name: str, data_dir: str | None = None) -> None:
    """Append a JSONL diagnostic log entry (when NAUTICAL_DIAG_LOG=1)."""
    if os.environ.get("NAUTICAL_DIAG_LOG") != "1":
        return
    path = _diag_log_path(data_dir)
    max_bytes = int(os.environ.get("NAUTICAL_DIAG_LOG_MAX_BYTES") or 262144)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    except Exception:
        pass
    try:
        if max_bytes > 0 and os.path.exists(path):
            try:
                st = os.stat(path)
                if st.st_size > max_bytes:
                    overflow = path.replace(".jsonl", f".overflow.{int(time.time())}.jsonl")
                    os.replace(path, overflow)
            except Exception:
                pass
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
        except Exception:
            pass
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hook": hook_name,
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": os.getcwd(),
        }
        if data_dir:
            payload["data_dir"] = str(data_dir)
        if isinstance(msg, dict):
            red = diag_log_redact(msg)
            if isinstance(red, dict):
                payload["msg"] = str(red.get("msg") or red.get("message") or "")
                payload["data"] = red
            else:
                payload["msg"] = str(red)
        else:
            payload["msg"] = diag_log_redact(str(msg))
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


def diag(msg, hook_name: str = "nautical", data_dir: str | None = None) -> None:
    """Write diagnostics to stderr when NAUTICAL_DIAG=1 and append to diag log when NAUTICAL_DIAG_LOG=1."""
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {msg}\n")
        except Exception:
            pass
    diag_log(msg, hook_name, data_dir)


def _run_task_should_retry(attempt: int, retries: int) -> bool:
    return attempt < retries


def _run_task_retry_sleep(attempt: int, retry_delay: float) -> None:
    delay = retry_delay * (2 ** (attempt - 1))
    jitter = random.uniform(0.0, retry_delay) if retry_delay > 0 else 0.0
    time.sleep(delay + jitter)


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
    """Run a subprocess; returns (ok, stdout, stderr). Uses env or os.environ.copy()."""
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
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=(out_f if out_f is not None else subprocess.PIPE),
                stderr=(err_f if err_f is not None else subprocess.PIPE),
                text=text_mode,
                encoding=("utf-8" if text_mode else None),
                errors=("replace" if text_mode else None),
                close_fds=True,
                env=env,
            )
            try:
                out, err = proc.communicate(input=normalized_input, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    out, err = proc.communicate(timeout=1.0)
                except Exception:
                    out, err = "", ""
                out, err = _run_task_collect_outputs(out_f, err_f, out_path, err_path, out, err)
                last_err = "timeout"
                if _run_task_should_retry(attempt, retries):
                    _run_task_retry_sleep(attempt, retry_delay)
                    continue
                return False, out or "", last_err
            out, err = _run_task_collect_outputs(out_f, err_f, out_path, err_path, out, err)
            last_out = out or ""
            last_err = err or ""
            if proc.returncode == 0:
                return True, last_out, last_err
            if _run_task_should_retry(attempt, retries):
                _run_task_retry_sleep(attempt, retry_delay)
                continue
            return False, last_out, last_err
        except Exception as e:
            last_err = str(e)
            _run_task_cleanup_paths(out_path, err_path)
            if _run_task_should_retry(attempt, retries):
                _run_task_retry_sleep(attempt, retry_delay)
                continue
            return False, last_out, last_err
    return False, last_out, last_err


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
