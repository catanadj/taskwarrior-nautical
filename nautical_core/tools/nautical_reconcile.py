#!/usr/bin/env python3
"""Repair Nautical chains missing a successor after completion or expiration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.machinery
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("NAUTICAL_CORE_PATH", str(BASE_DIR))

from nautical_core import queue_store, reconcile, safe_lock, task_command  # noqa: E402


_PARENT_LOCK_RETRIES = 600
_PARENT_LOCK_SLEEP_SECONDS = 0.1
_PARENT_LOCK_STALE_SECONDS = 300.0
_DEFAULT_EXPIRATION_HOPS = 32
_MAX_EXPIRATION_HOPS = 1000
_RECONCILE_PROTOCOL = 1


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


def _run_task(
    task_bin: str,
    args: list[str],
    *,
    input_text: str | None = None,
    timeout: float = 60.0,
    read_only: bool = False,
):
    return task_command.run_task_command(
        task_bin,
        args,
        input_text=input_text,
        timeout=timeout,
        retry_locks=read_only,
    )


def _export(task_bin: str, filters: list[str], *, timeout: float = 120.0) -> list[dict[str, Any]]:
    proc = _run_task(
        task_bin,
        ["rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", *filters, "export"],
        timeout=timeout,
        read_only=True,
    )
    payload = task_command.load_json_result(proc, "task export", empty=[])
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


def _validate_hook_protocol(hook: Any) -> None:
    if not isinstance(hook, ModuleType):
        return
    protocol = getattr(hook, "NAUTICAL_RECONCILE_PROTOCOL", None)
    if protocol != _RECONCILE_PROTOCOL:
        raise RuntimeError(
            f"incompatible on-modify reconcile protocol {protocol!r}; "
            f"expected {_RECONCILE_PROTOCOL}"
        )
    required_hook = (
        "_task_cmd_prefix",
        "_safe_parse_datetime",
        "_compute_anchor_child_due",
        "_compute_cp_child_due",
        "_build_child_from_parent",
        "_spawn_child",
    )
    required_core = ("coerce_int", "to_local", "build_local_datetime", "fmt_isoz")
    missing = [name for name in required_hook if not callable(getattr(hook, name, None))]
    core = getattr(hook, "core", None)
    missing.extend(f"core.{name}" for name in required_core if not callable(getattr(core, name, None)))
    if missing:
        raise RuntimeError(f"on-modify reconcile protocol is missing: {', '.join(missing)}")


def _candidate_sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("chainID") or "").strip().casefold(),
        reconcile.int_or_default(row.get("link"), 0),
        str(row.get("status") or "").strip().casefold(),
        str(row.get("uuid") or "").strip().casefold(),
    )


def _candidate_rows(task_bin: str, hook: Any) -> list[dict[str, Any]]:
    completed = _export(task_bin, ["status:completed", "chain:on", "chainID.not:", "nextLink:"])
    deleted = _export(task_bin, ["status:deleted", "chain:on", "chainID.not:", "nextLink:"])
    rows = [row for row in completed if reconcile.is_orphan_completion_candidate(row)]
    rows.extend(row for row in deleted if reconcile.is_orphan_deleted_chain_candidate(row))
    return sorted(rows, key=_candidate_sort_key)


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


def _existing_children_for_plan(task_bin: str, parent: dict[str, Any], hook: Any) -> list[dict[str, Any]]:
    if str(parent.get("status") or "").strip() == "deleted":
        disposition, _reason = reconcile.deleted_chain_disposition(
            parent,
            safe_parse_datetime=hook._safe_parse_datetime,
        )
        if disposition != "expiration":
            return []
    return _existing_children(task_bin, parent)


def _expiration_hop_limit(value: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise argparse.ArgumentTypeError("expiration hop limit must be an integer") from exc
    if parsed < 1 or parsed > _MAX_EXPIRATION_HOPS:
        raise argparse.ArgumentTypeError(
            f"expiration hop limit must be between 1 and {_MAX_EXPIRATION_HOPS}"
        )
    return parsed


def _task_data_dir(task_bin: str) -> Path:
    raw = str(os.environ.get("TASKDATA") or "").strip()
    if not raw:
        proc = _run_task(
            task_bin,
            ["rc.hooks=off", "rc.verbose=nothing", "_get", "rc.data.location"],
            timeout=10.0,
            read_only=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(task_command.failure_message(proc, "Taskwarrior data location lookup"))
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
        raise RuntimeError(task_command.failure_message(proc, "parent nextLink update"))


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
        raise RuntimeError(task_command.failure_message(proc, "parent chain update"))


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
        candidate = reconcile.is_orphan_deleted_chain_candidate(parent)
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
        existing_children=_existing_children_for_plan(task_bin, parent, hook),
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
        if plan.action in {"legitimate_final", "manual_stop"}:
            _disable_parent_chain(task_bin, plan.parent)
            return plan, "off"
        return plan, ""


def _recovery_error(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "error",
        parent,
        reconcile.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _recovery_partial(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "partial",
        parent,
        reconcile.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _validate_recovery_child(parent: dict[str, Any], child: dict[str, Any]) -> str:
    _child_short, child_error = reconcile.resolve_existing_child(
        parent,
        [child],
        include_deleted=True,
    )
    return child_error


def _terminal_recovery_error(child: dict[str, Any], hook: Any, recovery_at: Any) -> str:
    if str(child.get("status") or "").strip().lower() != "pending":
        return ""
    until_raw = child.get("until")
    try:
        until_dt, until_err = hook._safe_parse_datetime(until_raw)
    except Exception:
        return "live recovery child native until could not be parsed"
    if until_err or until_dt is None:
        return f"live recovery child has no reliable native until: {until_err or 'missing until'}"

    target_field = "due" if child.get("due") else "scheduled"
    target_raw = child.get(target_field)
    try:
        target_dt, target_err = hook._safe_parse_datetime(target_raw)
    except Exception:
        return f"live recovery child {target_field} could not be parsed"
    if target_err or target_dt is None:
        return f"live recovery child has no reliable {target_field}: {target_err or f'missing {target_field}'}"
    try:
        if until_dt <= target_dt:
            return f"live recovery child native until is not later than its {target_field}"
        if until_dt <= recovery_at:
            return "live recovery child native until has already elapsed"
    except Exception:
        return "live recovery child timing could not be compared"
    return ""


def _next_recovery_child(
    task_bin: str,
    parent: dict[str, Any],
    child_short: str,
) -> dict[str, Any]:
    wanted = str(child_short or "").strip().lower()
    if not wanted:
        raise RuntimeError("recovery action did not identify its child")
    rows = _export(task_bin, [f"uuid:{wanted}"], timeout=30.0)
    matches = [
        row
        for row in rows
        if str(row.get("uuid") or "").strip().lower().startswith(wanted)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"recovery child {wanted} lookup returned {len(matches)} exact match(es)"
        )
    child = matches[0]
    validation_error = _validate_recovery_child(parent, child)
    if validation_error:
        raise RuntimeError(validation_error)
    return child


def _virtual_expired_child(
    plan: reconcile.ReconcilePlan,
    *,
    hook: Any,
    recovery_at: Any,
) -> tuple[dict[str, Any] | None, str]:
    child = dict(plan.child or {})
    until_raw = child.get("until")
    try:
        until_dt, until_err = hook._safe_parse_datetime(until_raw)
    except Exception:
        return None, "planned child expiration could not be parsed"
    if until_err or until_dt is None:
        return None, f"planned child has no reliable native until: {until_err or 'missing until'}"
    try:
        if until_dt > recovery_at:
            return None, ""
    except Exception:
        return None, "planned child expiration could not be compared with recovery time"

    child["status"] = "deleted"
    child["end"] = until_raw
    child["uuid"] = (
        f"dryrun-{str(child.get('chainID') or 'chain')}-"
        f"{reconcile.int_or_default(child.get('link'), plan.next_link)}"
    )
    child.pop("nextLink", None)
    validation_error = _validate_recovery_child(plan.parent, child)
    if validation_error:
        return None, validation_error
    return child, ""


def _reconcile_candidate(
    task_bin: str,
    hook: Any,
    parent: dict[str, Any],
    *,
    taskdata: Path | None,
    apply: bool,
    max_expiration_hops: int,
    recovery_at: Any,
) -> list[tuple[reconcile.ReconcilePlan, str]]:
    outcomes: list[tuple[reconcile.ReconcilePlan, str]] = []
    current = parent
    visited: set[tuple[str, int]] = set()
    expiration_hops = 0

    while True:
        slot = (
            str(current.get("chainID") or "").strip(),
            reconcile.int_or_default(current.get("link"), 0),
        )
        if slot in visited:
            outcomes.append((_recovery_error(current, "expiration recovery made no progress"), ""))
            break
        visited.add(slot)

        is_deleted = str(current.get("status") or "").strip().lower() == "deleted"
        if is_deleted and expiration_hops >= max_expiration_hops:
            outcomes.append(
                (
                    _recovery_partial(
                        current,
                        f"expiration recovery hop limit reached at {max_expiration_hops}; "
                        "rerun to continue or increase --max-expiration-hops",
                    ),
                    "",
                )
            )
            break

        if apply:
            if taskdata is None:
                raise RuntimeError("Taskwarrior data location is unavailable")
            try:
                plan, applied_short = _apply_parent_atomic(
                    task_bin,
                    hook,
                    current,
                    taskdata=taskdata,
                )
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                outcomes.append((_recovery_error(current, reason), ""))
                break
        else:
            plan = reconcile.build_reconcile_plan(
                current,
                existing_children=_existing_children_for_plan(task_bin, current, hook),
                hook=hook,
            )
            applied_short = ""
        outcomes.append((plan, applied_short))

        if not is_deleted or plan.action not in {"spawn", "backfill_nextlink"}:
            break
        expiration_hops += 1

        if apply or plan.action == "backfill_nextlink":
            child_short = applied_short or plan.child_short
            try:
                child = _next_recovery_child(task_bin, plan.parent, child_short)
            except Exception as exc:
                outcomes.append((_recovery_error(plan.parent, str(exc)), ""))
                break
        else:
            child, child_error = _virtual_expired_child(
                plan,
                hook=hook,
                recovery_at=recovery_at,
            )
            if child_error:
                outcomes.append((_recovery_error(plan.parent, child_error), ""))
                break
            if child is None:
                terminal_error = _terminal_recovery_error(
                    dict(plan.child or {}),
                    hook,
                    recovery_at,
                )
                if terminal_error:
                    outcomes.append((_recovery_error(plan.parent, terminal_error), ""))
                break

        terminal_error = _terminal_recovery_error(child, hook, recovery_at)
        if terminal_error:
            outcomes.append((_recovery_error(plan.parent, terminal_error), ""))
            break
        if not reconcile.is_orphan_deleted_chain_candidate(child):
            break
        current = child

    return outcomes


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
    elif plan.action == "manual_stop":
        suffix = " -> set chain:off" if applied_short else ""
        print(f"manual stop: {parent} ({plan.reason}){suffix}")
        _print_evidence(evidence, ("kind", "next_link"))
    elif plan.action == "stale":
        print(f"skip: {parent} ({plan.reason})")
    elif plan.action == "partial":
        print(f"partial: {parent} ({plan.reason})")
    else:
        print(f"error: {parent} ({plan.reason})")
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))


def _print_recovery_group(
    items: list[tuple[reconcile.ReconcilePlan, dict[str, Any], str]],
) -> None:
    first = items[0][0]
    last, evidence, applied_short = items[-1]
    hops = sum(1 for plan, _evidence, _applied in items if plan.action in {"spawn", "backfill_nextlink"})
    noun = "occurrence" if hops == 1 else "occurrences"
    print(f"recover: {_fmt_parent(first.parent)} -> advanced {hops} {noun}")
    if last.action in {"error", "partial", "legitimate_final", "manual_stop", "stale"}:
        print(f"  result: {last.action.replace('_', ' ')} ({last.reason})")
        return
    if applied_short:
        print(f"  child: {applied_short}")
    _print_evidence(evidence, ("next_link", "child_local", "child_due", "child_expires"))


def _startup_failure(args: Any, stage: str, exc: Exception) -> int:
    reason = str(exc).strip() or type(exc).__name__
    if args.json:
        payload: dict[str, Any] = {
            "mode": "apply" if args.apply else "dry-run",
            "status": "error",
            "stage": stage,
            "error": reason,
            "candidates": 0,
            "expiration_hops": 0,
            "recovered_chains": 0,
            "spawn": 0,
            "backfill_nextlink": 0,
            "legitimate_final": 0,
            "manual_stop": 0,
            "stale": 0,
            "partial": 0,
            "errors": 1,
            "plans": [],
            "applied": [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"error: {stage.replace('_', ' ')}: {reason}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair Nautical chains after hookless completion, expiration, or deletion.")
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--hook-path", default=None, help="Explicit on-modify hook path for non-standard installs.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    parser.add_argument("--verbose", action="store_true", help="Print every delayed-recovery hop.")
    parser.add_argument(
        "--max-expiration-hops",
        type=_expiration_hop_limit,
        default=_DEFAULT_EXPIRATION_HOPS,
        help=f"Maximum expired links recovered per chain (default: {_DEFAULT_EXPIRATION_HOPS}).",
    )
    args = parser.parse_args(argv)

    try:
        hook = _load_on_modify(args.hook_path)
    except Exception as exc:
        return _startup_failure(args, "hook_load", exc)
    try:
        _validate_hook_protocol(hook)
        _bind_hook_task_bin(hook, args.task_bin)
        fmt_dt_local = getattr(getattr(hook, "core", None), "fmt_dt_local", None)
        now_utc = getattr(getattr(hook, "core", None), "now_utc", None)
        recovery_at = now_utc() if callable(now_utc) else datetime.now(timezone.utc)
    except Exception as exc:
        return _startup_failure(args, "hook_protocol", exc)
    try:
        candidates = _candidate_rows(args.task_bin, hook)
    except Exception as exc:
        return _startup_failure(args, "candidate_export", exc)
    try:
        taskdata = _task_data_dir(args.task_bin) if args.apply else None
    except Exception as exc:
        return _startup_failure(args, "taskdata", exc)
    plans: list[reconcile.ReconcilePlan] = []
    plan_evidence: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    outcome_groups: list[list[tuple[reconcile.ReconcilePlan, str]]] = []
    processed_slots: set[tuple[str, int]] = set()

    for parent in candidates:
        parent_slot = (
            str(parent.get("chainID") or "").strip(),
            reconcile.int_or_default(parent.get("link"), 0),
        )
        if parent_slot in processed_slots:
            continue
        try:
            outcomes = _reconcile_candidate(
                args.task_bin,
                hook,
                parent,
                taskdata=taskdata,
                apply=args.apply,
                max_expiration_hops=args.max_expiration_hops,
                recovery_at=recovery_at,
            )
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            outcomes = [(_recovery_error(parent, reason), "")]
        outcome_groups.append(outcomes)
        rendered: list[tuple[reconcile.ReconcilePlan, dict[str, Any], str]] = []
        for plan, applied_short in outcomes:
            processed_slots.add(
                (
                    str(plan.parent.get("chainID") or "").strip(),
                    reconcile.int_or_default(plan.parent.get("link"), 0),
                )
            )
            plans.append(plan)
            evidence = _describe_plan(plan, hook=hook, fmt_dt_local=fmt_dt_local)
            plan_evidence.append(evidence)
            rendered.append((plan, evidence, applied_short))
            if args.apply and applied_short:
                disabling = plan.action in {"legitimate_final", "manual_stop"}
                action = "disable_chain" if disabling else plan.action
                record = {
                    "action": action,
                    "parent": reconcile.short_uuid(plan.parent.get("uuid")),
                }
                if not disabling:
                    record["child"] = applied_short
                applied.append(record)
        if not args.json:
            if args.verbose or len(rendered) <= 1:
                for plan, evidence, applied_short in rendered:
                    _print_plan(plan, evidence, applied_short=applied_short)
            else:
                _print_recovery_group(rendered)

    expiration_hops = sum(
        1
        for plan in plans
        if str(plan.parent.get("status") or "").strip() == "deleted"
        and plan.action in {"spawn", "backfill_nextlink"}
    )
    recovered_chains = sum(
        1
        for outcomes in outcome_groups
        if sum(
            1
            for plan, _applied in outcomes
            if str(plan.parent.get("status") or "").strip() == "deleted"
            and plan.action in {"spawn", "backfill_nextlink"}
        )
        > 1
        and all(plan.action not in {"error", "partial"} for plan, _applied in outcomes)
    )

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "candidates": len(candidates),
        "expiration_hops": expiration_hops,
        "recovered_chains": recovered_chains,
        "spawn": sum(1 for p in plans if p.action == "spawn"),
        "backfill_nextlink": sum(1 for p in plans if p.action == "backfill_nextlink"),
        "legitimate_final": sum(1 for p in plans if p.action == "legitimate_final"),
        "manual_stop": sum(1 for p in plans if p.action == "manual_stop"),
        "stale": sum(1 for p in plans if p.action == "stale"),
        "partial": sum(1 for p in plans if p.action == "partial"),
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
            f"expiration_hops={summary['expiration_hops']} recovered={summary['recovered_chains']} "
            f"final={summary['legitimate_final']} manual={summary['manual_stop']} "
            f"stale={summary['stale']} partial={summary['partial']} errors={summary['errors']}"
        )
    if summary["errors"]:
        return 1
    return 2 if summary["partial"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
