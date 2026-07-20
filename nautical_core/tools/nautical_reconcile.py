#!/usr/bin/env python3
"""Repair Nautical chains missing a successor after completion or expiration."""

from __future__ import annotations

import argparse
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

from nautical_core import reconcile  # noqa: E402


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


def _modify_parent_nextlink(task_bin: str, parent: dict[str, Any], child_short: str) -> None:
    parent_uuid = str(parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    proc = _run_task(
        task_bin,
        ["rc.hooks=off", "rc.confirmation=off", "rc.verbose=nothing", f"uuid:{parent_uuid}", "modify", f"nextLink:{child_short}"],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "parent nextLink update failed").strip())


def _disable_parent_chain(task_bin: str, parent: dict[str, Any]) -> None:
    parent_uuid = str(parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    proc = _run_task(
        task_bin,
        ["rc.hooks=off", "rc.confirmation=off", "rc.verbose=nothing", f"uuid:{parent_uuid}", "modify", "chain:off"],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or "parent chain update failed").strip())


def _apply_spawn(task_bin: str, hook: Any, plan: reconcile.ReconcilePlan) -> str:
    if not plan.child:
        raise RuntimeError("spawn plan has no child payload")
    child_short, _stripped = hook._spawn_child(plan.child, plan.parent)
    _modify_parent_nextlink(task_bin, plan.parent, child_short)
    return child_short


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


def _print_plan(plan: reconcile.ReconcilePlan, *, applied_short: str = "", fmt_dt_local=None) -> None:
    parent = _fmt_parent(plan.parent)
    evidence = reconcile.describe_plan(plan, fmt_dt_local=fmt_dt_local)
    if plan.action == "spawn":
        suffix = f" -> created {applied_short}" if applied_short else ""
        print(f"spawn: {parent}{suffix}")
        _print_evidence(evidence, ("reason", "kind", "next_link", "child_field", "child_target", "child_due", "child_local"))
    elif plan.action == "backfill_nextlink":
        suffix = " (applied)" if applied_short else ""
        print(f"backfill nextLink: {parent}{suffix}")
        _print_evidence(evidence, ("reason", "next_link", "existing_child"))
    elif plan.action == "legitimate_final":
        suffix = " -> set chain:off" if applied_short else ""
        print(f"final: {parent} ({plan.reason}){suffix}")
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local"))
    else:
        print(f"error: {parent} ({plan.reason})")
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local"))


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
    plans: list[reconcile.ReconcilePlan] = []
    applied: list[dict[str, Any]] = []

    for parent in candidates:
        plan = reconcile.build_reconcile_plan(
            parent,
            existing_children=_existing_children(args.task_bin, parent),
            hook=hook,
        )
        plans.append(plan)
        if args.apply and plan.action == "spawn":
            child_short = _apply_spawn(args.task_bin, hook, plan)
            applied.append({"action": "spawn", "parent": reconcile.short_uuid(parent.get("uuid")), "child": child_short})
            if not args.json:
                _print_plan(plan, applied_short=child_short, fmt_dt_local=fmt_dt_local)
        elif args.apply and plan.action == "backfill_nextlink":
            _modify_parent_nextlink(args.task_bin, parent, plan.child_short)
            applied.append({"action": "backfill_nextlink", "parent": reconcile.short_uuid(parent.get("uuid")), "child": plan.child_short})
            if not args.json:
                _print_plan(plan, applied_short=plan.child_short, fmt_dt_local=fmt_dt_local)
        elif args.apply and plan.action == "legitimate_final":
            _disable_parent_chain(args.task_bin, parent)
            applied.append({"action": "disable_chain", "parent": reconcile.short_uuid(parent.get("uuid"))})
            if not args.json:
                _print_plan(plan, applied_short="off", fmt_dt_local=fmt_dt_local)
        elif not args.json:
            _print_plan(plan, fmt_dt_local=fmt_dt_local)

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "candidates": len(candidates),
        "spawn": sum(1 for p in plans if p.action == "spawn"),
        "backfill_nextlink": sum(1 for p in plans if p.action == "backfill_nextlink"),
        "legitimate_final": sum(1 for p in plans if p.action == "legitimate_final"),
        "errors": sum(1 for p in plans if p.action == "error"),
        "plans": [
            {
                "action": plan.action,
                **reconcile.describe_plan(plan, fmt_dt_local=fmt_dt_local),
            }
            for plan in plans
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
            f"final={summary['legitimate_final']} errors={summary['errors']}"
        )
    return 1 if summary["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
