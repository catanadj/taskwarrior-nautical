#!/usr/bin/env python3
"""Repair Nautical chains missing a successor after completion or expiration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.machinery
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("NAUTICAL_CORE_PATH", str(BASE_DIR))

from nautical_core import queue_store, reconcile, safe_lock  # noqa: E402


_PARENT_LOCK_RETRIES = 600
_PARENT_LOCK_SLEEP_SECONDS = 0.1
_PARENT_LOCK_STALE_SECONDS = 300.0


def _candidate_on_modify_paths(explicit: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    for raw in (explicit, os.environ.get("NAUTICAL_ON_MODIFY_PATH")):
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            CORE_DIR / "hooks" / "modify_impl.py",
            BASE_DIR / "on-modify-nautical.py",
            BASE_DIR / "hooks" / "on-modify-nautical.py",
            BASE_DIR / "hooks" / "on-modify",
            CORE_DIR / "on-modify-nautical.py",
        ]
    )
    return candidates


def _modify_implementation_path(path: Path) -> Path:
    if path.name == "modify_impl.py":
        return path
    candidates = (
        path.parent / "nautical_core" / "hooks" / "modify_impl.py",
        path.parent.parent / "nautical_core" / "hooks" / "modify_impl.py",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), path)


def _run_task(task_bin: str, args: list[str], *, input_text: str | None = None, timeout: float = 60.0):
    return subprocess.run(
        [task_bin, *args],
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        env=os.environ.copy(),
    )


def _export(task_bin: str, filters: list[str], *, timeout: float = 120.0) -> list[dict[str, Any]]:
    proc = _run_task(
        task_bin,
        ["rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", *filters, "export"],
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "task export failed").strip())
    payload = json.loads((proc.stdout or "").strip() or "[]")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError("task export returned a non-list payload")
    return [row for row in payload if isinstance(row, dict)]


def _load_on_modify(hook_path: str | None = None):
    searched = _candidate_on_modify_paths(hook_path)
    path = next((candidate for candidate in searched if candidate.is_file()), None)
    if path is None:
        tried = ", ".join(str(candidate) for candidate in searched)
        raise RuntimeError(f"could not find on-modify hook; tried: {tried}")
    path = _modify_implementation_path(path)
    loader = importlib.machinery.SourceFileLoader("_nautical_reconcile_on_modify", str(path))
    spec = importlib.util.spec_from_loader("_nautical_reconcile_on_modify", loader)
    if spec is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    if hasattr(module, "_load_core"):
        module._load_core()
    return module


def _bind_hook_task_bin(hook: Any, task_bin: str) -> None:
    original_prefix = hook._task_cmd_prefix

    def _task_cmd_prefix() -> list[str]:
        prefix = list(original_prefix())
        if prefix:
            prefix[0] = task_bin
        return prefix

    hook._task_cmd_prefix = _task_cmd_prefix


def _candidate_rows(task_bin: str, hook: Any) -> list[dict[str, Any]]:
    completed = _export(task_bin, ["status:completed", "chain:on", "chainID.not:"])
    deleted = _export(task_bin, ["status:deleted", "chain:on", "chainID.not:"])
    rows = [row for row in completed if reconcile.is_orphan_completion_candidate(row)]
    rows.extend(
        row
        for row in deleted
        if reconcile.is_orphan_expiration_candidate(row, safe_parse_datetime=hook._safe_parse_datetime)
    )
    return rows


def _existing_children(task_bin: str, parent: dict[str, Any]) -> list[dict[str, Any]]:
    chain_id = str(parent.get("chainID") or "").strip()
    next_link = reconcile.int_or_default(parent.get("link"), 1) + 1
    if not chain_id:
        return []
    rows = _export(task_bin, [f"chainID:{chain_id}", f"link:{next_link}", "status.not:deleted"], timeout=30.0)
    if str(parent.get("status") or "").strip() == "deleted":
        rows.extend(
            _export(task_bin, [f"chainID:{chain_id}", f"link:{next_link}", "status:deleted"], timeout=30.0)
        )
    return rows


def _task_data_dir(task_bin: str) -> Path:
    raw = str(os.environ.get("TASKDATA") or "").strip()
    if not raw:
        proc = _run_task(
            task_bin,
            ["rc.hooks=off", "rc.verbose=nothing", "_get", "rc.data.location"],
            timeout=10.0,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "Taskwarrior data location lookup failed").strip())
        raw = str(proc.stdout or "").strip()
    if not raw:
        raise RuntimeError("Taskwarrior data location is empty")
    return Path(os.path.expandvars(raw)).expanduser().resolve()


@contextmanager
def _parent_apply_lock(taskdata: Path, parent_uuid: str):
    lock_path = queue_store.parent_nextlink_lock_path(taskdata, parent_uuid)
    with safe_lock(
        lock_path,
        retries=_PARENT_LOCK_RETRIES,
        sleep_base=_PARENT_LOCK_SLEEP_SECONDS,
        stale_after=_PARENT_LOCK_STALE_SECONDS,
    ) as acquired:
        yield acquired


def _fresh_parent(task_bin: str, parent: dict[str, Any]) -> dict[str, Any] | None:
    parent_uuid = str(parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    rows = _export(task_bin, [f"uuid:{parent_uuid}"], timeout=30.0)
    wanted = parent_uuid.lower()
    for row in rows:
        if str(row.get("uuid") or "").strip().lower() == wanted:
            return row
    return None


def _parent_guard_filters(parent: dict[str, Any]) -> list[str]:
    parent_uuid = str(parent.get("uuid") or "").strip()
    status = str(parent.get("status") or "").strip().lower()
    chain_id = str(parent.get("chainID") or "").strip()
    link = reconcile.int_or_default(parent.get("link"), 0)
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    if status not in {"completed", "deleted"}:
        raise RuntimeError("parent status is no longer reconcilable")
    if str(parent.get("chain") or "").strip().lower() != "on":
        raise RuntimeError("parent chain is no longer active")
    if not chain_id or link <= 0:
        raise RuntimeError("parent chain identity is incomplete")
    if str(parent.get("nextLink") or "").strip():
        raise RuntimeError("parent nextLink is already set")
    return [
        f"uuid:{parent_uuid}",
        f"status:{status}",
        "chain:on",
        f"chainID:{chain_id}",
        f"link:{link}",
        "nextLink:",
    ]


def _modify_parent_nextlink(task_bin: str, parent: dict[str, Any], child_short: str) -> None:
    filters = _parent_guard_filters(parent)
    proc = _run_task(
        task_bin,
        [
            "rc.hooks=off",
            "rc.confirmation=off",
            "rc.verbose=nothing",
            *filters,
            "modify",
            f"nextLink:{child_short}",
        ],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "parent nextLink update failed").strip())


def _disable_parent_chain(task_bin: str, parent: dict[str, Any]) -> None:
    filters = _parent_guard_filters(parent)
    proc = _run_task(
        task_bin,
        [
            "rc.hooks=off",
            "rc.confirmation=off",
            "rc.verbose=nothing",
            *filters,
            "modify",
            "chain:off",
        ],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "parent chain update failed").strip())


def _stale_plan(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "stale",
        parent,
        reconcile.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _refresh_plan(task_bin: str, hook: Any, original_parent: dict[str, Any]) -> reconcile.ReconcilePlan:
    parent = _fresh_parent(task_bin, original_parent)
    if parent is None:
        return _stale_plan(original_parent, "parent no longer exists")
    status = str(parent.get("status") or "").strip().lower()
    if status == "completed":
        candidate = reconcile.is_orphan_completion_candidate(parent)
    elif status == "deleted":
        candidate = reconcile.is_orphan_expiration_candidate(
            parent,
            safe_parse_datetime=hook._safe_parse_datetime,
        )
    else:
        candidate = False
    if not candidate:
        reason = (
            "parent nextLink already set"
            if str(parent.get("nextLink") or "").strip()
            else "parent no longer needs reconciliation"
        )
        return _stale_plan(parent, reason)
    return reconcile.build_reconcile_plan(
        parent,
        existing_children=_existing_children(task_bin, parent),
        hook=hook,
    )


def _apply_parent_atomic(
    task_bin: str,
    hook: Any,
    original_parent: dict[str, Any],
    *,
    taskdata: Path,
) -> tuple[reconcile.ReconcilePlan, str]:
    parent_uuid = str(original_parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    with _parent_apply_lock(taskdata, parent_uuid) as acquired:
        if not acquired:
            raise RuntimeError(f"parent reconcile lock busy: {reconcile.short_uuid(parent_uuid)}")
        plan = _refresh_plan(task_bin, hook, original_parent)
        if plan.action == "spawn":
            if not plan.child:
                raise RuntimeError("spawn plan has no child payload")
            child_short, _stripped = hook._spawn_child(plan.child, plan.parent)
            _modify_parent_nextlink(task_bin, plan.parent, child_short)
            return plan, child_short
        if plan.action == "backfill_nextlink":
            _modify_parent_nextlink(task_bin, plan.parent, plan.child_short)
            return plan, plan.child_short
        if plan.action == "legitimate_final":
            _disable_parent_chain(task_bin, plan.parent)
            return plan, "off"
        return plan, ""


def _fmt_parent(parent: dict[str, Any]) -> str:
    uuid = reconcile.short_uuid(parent.get("uuid")) or "????????"
    chain_id = str(parent.get("chainID") or "?")
    link = reconcile.int_or_default(parent.get("link"), 0)
    desc = str(parent.get("description") or "").strip()
    return f"{uuid} chain {chain_id} link {link}" + (f" · {desc}" if desc else "")


def _print_evidence(evidence: dict[str, Any], keys: tuple[str, ...]) -> None:
    for key in keys:
        value = evidence.get(key)
        if value in (None, ""):
            continue
        print(f"  {key.replace('_', ' ')}: {value}")


def _describe_plan(plan: reconcile.ReconcilePlan, *, hook: Any, fmt_dt_local=None) -> dict[str, Any]:
    evidence = reconcile.describe_plan(plan, fmt_dt_local=fmt_dt_local)
    child = plan.child if isinstance(plan.child, dict) else {}
    child_until = child.get("until")
    if not child_until:
        return evidence
    try:
        until_dt, until_err = hook._safe_parse_datetime(child_until)
    except Exception:
        return evidence
    if until_err or until_dt is None:
        return evidence

    if callable(fmt_dt_local):
        try:
            evidence["child_expires"] = str(fmt_dt_local(until_dt))
        except Exception:
            evidence["child_expires"] = str(child_until)
    else:
        evidence["child_expires"] = str(child_until)

    if plan.child_due is None:
        return evidence
    try:
        add_validation = hook.core._import_sibling("add_validation")
        carry = add_validation.describe_native_until_carry(
            until_dt,
            plan.child_due,
            to_local=hook.core.to_local,
        )
    except Exception:
        carry = None
    if carry:
        evidence["expiration"] = carry
    return evidence


def _print_plan(
    plan: reconcile.ReconcilePlan,
    evidence: dict[str, Any] | None = None,
    *,
    applied_short: str = "",
) -> None:
    parent = _fmt_parent(plan.parent)
    if evidence is None:
        evidence = reconcile.describe_plan(plan)
    if plan.action == "spawn":
        suffix = f" -> created {applied_short}" if applied_short else ""
        print(f"spawn: {parent}{suffix}")
        _print_evidence(evidence, ("reason", "kind", "next_link", "child_field", "child_target", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "backfill_nextlink":
        suffix = " (applied)" if applied_short else ""
        print(f"backfill nextLink: {parent}{suffix}")
        _print_evidence(evidence, ("reason", "next_link", "existing_child"))
    elif plan.action == "legitimate_final":
        suffix = " -> set chain:off" if applied_short else ""
        print(f"final: {parent} ({plan.reason}){suffix}")
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "stale":
        print(f"skip: {parent} ({plan.reason})")
    else:
        print(f"error: {parent} ({plan.reason})")
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair Nautical chains missing successors after completion or expiration.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--hook-path", default=None, help="Explicit on-modify hook path for non-standard installs.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    args = parser.parse_args(argv)

    hook = _load_on_modify(args.hook_path)
    _bind_hook_task_bin(hook, args.task_bin)
    fmt_dt_local = getattr(getattr(hook, "core", None), "fmt_dt_local", None)
    candidates = _candidate_rows(args.task_bin, hook)
    taskdata = _task_data_dir(args.task_bin) if args.apply else None
    plans: list[reconcile.ReconcilePlan] = []
    plan_evidence: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []

    for parent in candidates:
        applied_short = ""
        if args.apply:
            if taskdata is None:
                raise RuntimeError("Taskwarrior data location is unavailable")
            plan, applied_short = _apply_parent_atomic(
                args.task_bin,
                hook,
                parent,
                taskdata=taskdata,
            )
        else:
            plan = reconcile.build_reconcile_plan(
                parent,
                existing_children=_existing_children(args.task_bin, parent),
                hook=hook,
            )
        plans.append(plan)
        evidence = _describe_plan(plan, hook=hook, fmt_dt_local=fmt_dt_local)
        plan_evidence.append(evidence)
        if args.apply and applied_short:
            action = "disable_chain" if plan.action == "legitimate_final" else plan.action
            record = {
                "action": action,
                "parent": reconcile.short_uuid(plan.parent.get("uuid")),
            }
            if plan.action != "legitimate_final":
                record["child"] = applied_short
            applied.append(record)
        if not args.json:
            _print_plan(plan, evidence, applied_short=applied_short)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "candidates": len(candidates),
        "spawn": sum(1 for p in plans if p.action == "spawn"),
        "backfill_nextlink": sum(1 for p in plans if p.action == "backfill_nextlink"),
        "legitimate_final": sum(1 for p in plans if p.action == "legitimate_final"),
        "stale": sum(1 for p in plans if p.action == "stale"),
        "errors": sum(1 for p in plans if p.action == "error"),
        "plans": [
            {
                "action": plan.action,
                **evidence,
            }
            for plan, evidence in zip(plans, plan_evidence)
        ],
        "applied": applied,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(
            "summary: "
            f"{summary['mode']}; candidates={summary['candidates']} "
            f"spawn={summary['spawn']} backfill={summary['backfill_nextlink']} "
            f"final={summary['legitimate_final']} stale={summary['stale']} errors={summary['errors']}"
        )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
