from __future__ import annotations

from . import hook_support
from .task_models import TaskPayload


def task_text(args, *, run_task, task_cmd_prefix, env=None, timeout: float = 3.0, retries: int = 2, diag=None) -> str:
    env = env or {}
    result = hook_support.run_task_result(
        run_task=run_task,
        cmd=list(task_cmd_prefix) + ["rc.hooks=off"] + list(args),
        env=env,
        timeout=timeout,
        retries=retries,
    )
    if not result.ok and callable(diag):
        diag(f"task {' '.join(args)} failed: {(result.stderr or '').strip()}")
    return result.stdout or ""


def tw_get(ref: str, *, task_text) -> str:
    try:
        out = task_text(["rc.verbose=nothing", "_get", ref])
        return (out or "").strip()
    except Exception:
        return ""


def chain_root_and_age(
    task: TaskPayload,
    now_utc,
    *,
    root_uuid_from,
    tw_get_cached,
    dtparse,
    tolocal,
) -> tuple[str, int | None]:
    try:
        root_short = root_uuid_from(task)
        age_days = None
        if root_short:
            root_entry = tw_get_cached(f"{root_short}.entry")
            entry_dt = dtparse(root_entry)
            if entry_dt:
                entry_local = tolocal(entry_dt).date()
                today_local = tolocal(now_utc).date()
                age_days = (today_local - entry_local).days
                if age_days < 0:
                    age_days = 0
        return root_short or "—", age_days
    except Exception:
        return "—", None


def format_root_and_age(task: TaskPayload, now_utc, *, chain_root_and_age) -> str:
    root_short, age_days = chain_root_and_age(task, now_utc)
    if not root_short or root_short == "—":
        return "—"
    if age_days is not None and age_days > 0:
        return f"{root_short} ▻ {age_days}d"
    return root_short
