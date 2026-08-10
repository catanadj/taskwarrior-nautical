from __future__ import annotations

import json
import os
import sys
import time

from . import _normalized_abspath, _validated_user_dir
from .hook_bootstrap import env_int


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
    max_bytes = env_int(
        "NAUTICAL_DIAG_LOG_MAX_BYTES",
        262144,
        min_value=0,
        max_value=100 * 1024 * 1024,
    )
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
    render = getattr(msg, "render", None)
    rendered = render() if callable(render) else str(msg)
    to_log_record = getattr(msg, "to_log_record", None)
    log_value = to_log_record() if callable(to_log_record) else msg
    if isinstance(log_value, dict):
        # Preserve structured fields while applying the existing redaction
        # policy to legacy JSON messages carried in the event text.
        log_value = dict(log_value)
        log_value["message"] = diag_log_redact(rendered)
    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {rendered}\n")
        except Exception:
            pass
    diag_log(log_value, hook_name, data_dir)


def _runtime_command_module():
    """Resolve command execution only when a runtime caller requests it."""
    from . import runtime_command

    return runtime_command


def run_task(*args, **kwargs):
    return _runtime_command_module().run_task(*args, **kwargs)


def is_lock_error(*args, **kwargs):
    return _runtime_command_module().is_lock_error(*args, **kwargs)


def _run_task_should_retry(*args, **kwargs):
    return _runtime_command_module()._run_task_should_retry(*args, **kwargs)


def _run_task_retry_sleep(*args, **kwargs):
    return _runtime_command_module()._run_task_retry_sleep(*args, **kwargs)


def _run_task_prepare_tempfiles(*args, **kwargs):
    return _runtime_command_module()._run_task_prepare_tempfiles(*args, **kwargs)


def _run_task_normalize_input(*args, **kwargs):
    return _runtime_command_module()._run_task_normalize_input(*args, **kwargs)


def _run_task_collect_outputs(*args, **kwargs):
    return _runtime_command_module()._run_task_collect_outputs(*args, **kwargs)


def _run_task_cleanup_paths(*args, **kwargs):
    return _runtime_command_module()._run_task_cleanup_paths(*args, **kwargs)
