from __future__ import annotations

import json

from . import hook_support
from .modify_models import CompletionSnapshotResult


def export_completion_chain_snapshot(
    chain_id: str,
    links: list[int] | None,
    *,
    run_task,
    task_cmd_prefix,
    parse_export_array,
    diag=None,
    timeout: float = 3.0,
) -> CompletionSnapshotResult:
    if not chain_id:
        return CompletionSnapshotResult(False, [], "missing chain ID")
    clauses = [[f"chainID:{chain_id}", f"link:{link}"] for link in (links or [])]
    if not clauses:
        clauses = [[f"chainID:{chain_id}"]]
    args = list(task_cmd_prefix) + [
        "rc.hooks=off",
        "rc.json.array=1",
        "rc.verbose=nothing",
        "rc.color=off",
    ]
    for index, clause in enumerate(clauses):
        if index:
            args.append("or")
        args.extend(clause)
    args.append("export")
    result = hook_support.run_task_result(
        run_task=run_task,
        cmd=args,
        timeout=timeout,
        retries=1,
        use_tempfiles=not bool(links),
    )
    if not result.ok:
        if callable(diag):
            diag(f"completion chain snapshot failed: {(result.stderr or '').strip()}")
        return CompletionSnapshotResult(False, [], (result.stderr or "completion chain snapshot failed").strip())
    out = result.stdout
    if not (out or "").lstrip().startswith("["):
        if callable(diag):
            diag("completion chain snapshot returned malformed JSON")
        return CompletionSnapshotResult(False, [], "completion chain snapshot returned malformed JSON")
    try:
        # Validate the wire payload before invoking the compatibility parser.
        # The older parser intentionally returns [] on errors, which must not
        # be allowed to turn an unavailable snapshot into confirmed absence.
        payload = json.loads((out or "").strip())
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            raise ValueError("expected an array of task objects")
        rows = parse_export_array(out, diag=diag)
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("snapshot parser returned an invalid row payload")
    except Exception as exc:
        if callable(diag):
            diag(f"completion chain snapshot parse failed: {exc}")
        return CompletionSnapshotResult(False, [], str(exc))
    return CompletionSnapshotResult(True, rows)


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
    task: dict,
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


def format_root_and_age(task: dict, now_utc, *, chain_root_and_age) -> str:
    root_short, age_days = chain_root_and_age(task, now_utc)
    if not root_short or root_short == "—":
        return "—"
    if age_days is not None and age_days > 0:
        return f"{root_short} ▻ {age_days}d"
    return root_short


def export_chain_endpoint(
    chain_id: str,
    direction: str,
    *,
    run_task,
    task_cmd_prefix,
    parse_export_array,
    diag=None,
    timeout: float = 3.0,
    retries: int = 1,
):
    if not chain_id:
        return None
    sort_dir = "+" if direction == "first" else "-"
    args = list(task_cmd_prefix) + [
        "rc.hooks=off",
        "rc.json.array=on",
        "rc.verbose=nothing",
        f"chainID:{chain_id}",
        f"sort:link{sort_dir}",
        "limit:1",
        "export",
    ]
    result = hook_support.run_task_result(
        run_task=run_task,
        cmd=args,
        env=None,
        timeout=timeout,
        retries=retries,
    )
    if not result.ok:
        if callable(diag):
            diag(f"chain endpoint export failed ({direction}): {(result.stderr or '').strip()}")
        return None
    data = parse_export_array(result.stdout, diag=diag)
    return data[0] if data else None
