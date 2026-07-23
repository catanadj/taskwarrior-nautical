#!/usr/bin/env python3
"""Read-only installation and data diagnostics for Taskwarrior Nautical."""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tomllib
import zoneinfo
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

ZONEINFO_FACTORY: Callable[[str], Any] | None = getattr(zoneinfo, "ZoneInfo", None)
RICH_SPEC_FACTORY: Callable[[str], Any] = importlib.util.find_spec


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nautical_core import chain_repair, config_schema, install_runtime, reconcile  # noqa: E402
import nautical_core.runtime as runtime  # noqa: E402

REQUIRED_UDAS = {
    "cp": "string",
    "chain": "string",
    "anchor": "string",
    "anchor_file": "string",
    "anchor_mode": "string",
    "bc": "string",
    "omit": "string",
    "omit_file": "string",
    "chainMax": "numeric",
    "chainUntil": "date",
    "prevLink": "string",
    "nextLink": "string",
    "link": "numeric",
    "chainID": "string",
}
RECURRENCE_FIELDS = ("cp", "anchor", "anchor_file")
SEVERITY_RANK = {"ok": 0, "warn": 1, "error": 2}


def _finding(
    findings: list[dict[str, Any]],
    check_id: str,
    severity: str,
    message: str,
    *,
    fix: str = "",
    details: dict[str, Any] | None = None,
) -> None:
    item: dict[str, Any] = {"id": check_id, "severity": severity, "message": message}
    if fix:
        item["fix"] = fix
    if details:
        item["details"] = details
    findings.append(item)


def _run_task(task_bin: str, args: list[str], env: dict[str, str], timeout: float = 30.0):
    try:
        return subprocess.run(
            [task_bin, *args],
            text=True,
            capture_output=True,
            env=env,
            timeout=timeout,
        )
    except Exception as exc:
        return subprocess.CompletedProcess([task_bin, *args], 127, "", str(exc))


def _task_get(task_bin: str, key: str, env: dict[str, str]) -> tuple[bool, str]:
    proc = _run_task(task_bin, ["_get", key], env)
    return proc.returncode == 0, (proc.stdout or "").strip()


def _discover_taskdata(task_bin: str, env: dict[str, str]) -> str:
    ok, raw = _task_get(task_bin, "rc.data.location", env)
    return raw if ok else ""


def _task_export(task_bin: str, env: dict[str, str]) -> tuple[bool, list[dict[str, Any]], str]:
    proc = _run_task(
        task_bin,
        ["rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "rc.color=off", "export"],
        env,
        timeout=120.0,
    )
    if proc.returncode != 0:
        return False, [], (proc.stderr or proc.stdout or "task export failed").strip()
    try:
        payload = json.loads((proc.stdout or "").strip() or "[]")
    except Exception as exc:
        return False, [], f"task export returned invalid JSON: {exc}"
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        return False, [], "task export returned a non-list payload"
    return True, [row for row in payload if isinstance(row, dict)], ""


def _resolve_hooks_dir(task_bin: str, taskdata: Path, env: dict[str, str]) -> Path:
    ok, raw = _task_get(task_bin, "rc.hooks.location", env)
    if ok and raw:
        return Path(raw).expanduser().resolve()
    return (taskdata / "hooks").resolve()


def _hook_candidates(hooks_dir: Path, event: str) -> list[Path]:
    return install_runtime.hook_candidates(hooks_dir, event)


def _check_hook_installation(
    findings: list[dict[str, Any]],
    *,
    hooks_dir: Path,
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    validated: dict[str, dict[str, Any]] = {}
    for event in install_runtime.HOOK_RUNTIME_FILES:
        candidates = _hook_candidates(hooks_dir, event)
        active = [hook for hook in candidates if os.access(str(hook), os.X_OK)]
        if not candidates:
            _finding(
                findings,
                f"hook.{event}.missing",
                "error",
                f"No Nautical {event} hook was found in {hooks_dir}.",
                fix="Install the Nautical hook files and make them executable.",
            )
            continue
        if not active:
            _finding(
                findings,
                f"hook.{event}.inactive",
                "error",
                f"Nautical {event} hook is not executable: {candidates[0]}",
                fix=f"Run chmod +x {candidates[0]}",
                details={"hooks": [str(path) for path in candidates]},
            )
            continue
        if len(active) > 1:
            _finding(
                findings,
                f"hook.{event}.duplicate",
                "error",
                f"{len(active)} active Nautical {event} hooks were found; Taskwarrior may run all of them.",
                fix="Keep exactly one executable Nautical hook for this event.",
                details={"hooks": [str(path) for path in active]},
            )
            continue

        record, error, details = install_runtime.inspect_hook_runtime(active[0], event, env)
        if record is None:
            _finding(
                findings,
                f"hook.{event}.incompatible",
                "error",
                error,
                fix="Install the matching Nautical wrappers and nautical_core from the same release.",
                details=details,
            )
            continue
        validated[event] = record
        _finding(
            findings,
            f"hook.{event}",
            "ok",
            f"{event} hook and core are compatible: {active[0]}",
            details=details,
        )
    return validated


def _check_runtime(
    findings: list[dict[str, Any]],
    *,
    task_bin: str,
    taskdata: Path,
    env: dict[str, str],
) -> Path:
    proc = _run_task(task_bin, ["--version"], env)
    if proc.returncode != 0:
        _finding(
            findings,
            "taskwarrior.unavailable",
            "error",
            "Taskwarrior could not be executed.",
            fix="Install Taskwarrior or pass --task-bin.",
            details={"error": (proc.stderr or "").strip()},
        )
    else:
        _finding(
            findings,
            "taskwarrior.version",
            "ok",
            f"Taskwarrior {(proc.stdout or '').strip() or 'version detected'}.",
        )

    if not taskdata.exists():
        _finding(
            findings,
            "taskdata.missing",
            "error",
            f"Taskwarrior data directory does not exist: {taskdata}",
            fix="Check TASKDATA, TASKRC, or pass --taskdata.",
        )
    elif not taskdata.is_dir():
        _finding(findings, "taskdata.invalid", "error", f"Taskwarrior data path is not a directory: {taskdata}")
    else:
        writable = os.access(str(taskdata), os.R_OK | os.W_OK | os.X_OK)
        _finding(
            findings,
            "taskdata.access",
            "ok" if writable else "error",
            f"Taskwarrior data directory is {'accessible' if writable else 'not fully accessible'}: {taskdata}",
            fix="" if writable else "Correct ownership and directory permissions.",
        )
    return _resolve_hooks_dir(task_bin, taskdata, env)


def _check_hooks_and_udas(
    findings: list[dict[str, Any]],
    *,
    task_bin: str,
    hooks_dir: Path,
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    validated = _check_hook_installation(findings, hooks_dir=hooks_dir, env=env)
    for name, expected in REQUIRED_UDAS.items():
        ok, actual = _task_get(task_bin, f"rc.uda.{name}.type", env)
        if not ok or not actual:
            _finding(
                findings,
                f"uda.{name}.missing",
                "error",
                f"Required UDA '{name}' is not defined.",
                fix="Include Nautical's uda.conf from your Taskwarrior configuration.",
            )
        elif actual.lower() != expected:
            _finding(
                findings,
                f"uda.{name}.type",
                "error",
                f"UDA '{name}' has type '{actual}', expected '{expected}'.",
                fix=f"Set uda.{name}.type={expected}.",
            )
    return validated


def _check_managed_runtime(findings: list[dict[str, Any]], hooks_dir: Path) -> None:
    try:
        status = install_runtime.runtime_status(hooks_dir.parent)
    except Exception as exc:
        _finding(
            findings,
            "install.runtime",
            "error",
            "Managed Nautical runtime state could not be inspected.",
            fix="Run nautical install --dry-run, then reinstall from a valid local release.",
            details={"error": str(exc)},
        )
        return
    abandoned = list(status.get("abandoned") or [])
    if abandoned:
        _finding(
            findings,
            "install.runtime_abandoned",
            "warn",
            f"{len(abandoned)} abandoned install transaction path(s) were found.",
            fix="Confirm no install is running, then remove the listed .staging/.rollback paths.",
            details={"paths": abandoned},
        )
    if not status.get("managed"):
        return

    errors = list(status.get("errors") or [])
    if errors:
        _finding(
            findings,
            "install.runtime",
            "error",
            "Managed Nautical runtime is incomplete or has a broken active pointer.",
            fix="Reinstall from a valid local release; the installer will preserve the previous active release.",
            details={"errors": errors, **status},
        )
    else:
        release_id = str(status.get("active_release") or "unknown")
        _finding(
            findings,
            "install.runtime",
            "ok",
            f"Managed Nautical runtime is active: {release_id}.",
            details=status,
        )


def _config_candidates(taskdata: Path) -> list[Path]:
    explicit = os.environ.get("NAUTICAL_CONFIG", "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    candidates.extend(
        [
            taskdata / "nautical_core" / "config-nautical.toml",
            taskdata / "nautical_core" / "nautical.toml",
            Path("~/.config/nautical/config-nautical.toml").expanduser(),
            Path("~/.config/nautical/nautical.toml").expanduser(),
            taskdata / "config-nautical.toml",
            taskdata / "nautical.toml",
        ]
    )
    return candidates


def _check_config(findings: list[dict[str, Any]], taskdata: Path) -> None:
    config = next((path.resolve() for path in _config_candidates(taskdata) if path.is_file()), None)
    if config is None:
        _finding(
            findings,
            "config.missing",
            "warn",
            "No Nautical config file was found; built-in defaults will be used.",
        )
        return
    try:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
    except Exception as exc:
        _finding(
            findings,
            "config.invalid",
            "error",
            f"Nautical config cannot be parsed: {config}",
            fix="Correct the TOML syntax.",
            details={"error": str(exc)},
        )
        return
    _finding(findings, "config.loaded", "ok", f"Nautical config is valid: {config}")
    _check_config_schema(findings, data)
    _check_timezone(findings, data)
    _check_panel_config(findings, data)
    for key in ("anchor_file_dir", "omit_file_dir"):
        raw = str(data.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = config.parent / path
        valid = path.is_dir() and os.access(str(path), os.R_OK | os.X_OK)
        _finding(
            findings,
            f"config.{key}",
            "ok" if valid else "error",
            f"{key} {'is accessible' if valid else 'is not accessible'}: {path.resolve()}",
            fix="" if valid else f"Create or correct the configured {key} directory.",
        )


def _check_config_schema(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    for issue in config_schema.validate_config(data, skip_keys={"live_panel_duration_ms"}):
        kind = str(issue["kind"])
        key = str(issue["key"])
        details = {name: value for name, value in issue.items() if name not in {"kind", "message"}}
        if kind == "unknown":
            message = f"Unknown Nautical config key: {key}."
            fix = f"Remove or correct '{key}'."
        elif kind == "deprecated":
            message = f"Retired Nautical config key '{key}': {issue['message']}"
            fix = f"Remove '{key}' from the config."
        elif kind == "type":
            message = (
                f"Config key '{key}' has the wrong type; expected {issue['expected']} "
                f"and will use {issue['effective']!r}."
            )
            fix = f"Set '{key}' to a TOML {issue['expected']} value."
        elif kind == "range":
            message = (
                f"Config key '{key}' is outside its supported range; "
                f"{issue['effective']!r} will be used."
            )
            bounds = [
                text
                for text in (
                    f"at least {issue['min']}" if issue.get("min") is not None else "",
                    f"at most {issue['max']}" if issue.get("max") is not None else "",
                )
                if text
            ]
            fix = f"Set '{key}' to {' and '.join(bounds)}."
        else:
            message = (
                f"Config key '{key}' has unsupported value {issue['configured']!r}; "
                f"{issue['effective']!r} will be used."
            )
            fix = f"Set '{key}' to one of: {', '.join(issue['choices'])}."
        _finding(
            findings,
            f"config.schema.{kind}",
            "warn",
            message,
            fix=fix,
            details=details,
        )


def _check_timezone(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    tz_name = str(data.get("tz") or "Europe/Bucharest").strip() or "Europe/Bucharest"
    if ZONEINFO_FACTORY is None:
        _finding(
            findings,
            "config.timezone.unavailable",
            "warn",
            "Python zoneinfo support is unavailable; Nautical will use UTC fallback.",
            fix="Use Python 3.9+ with zoneinfo support, or install timezone support for your Python build.",
            details={"tz": tz_name},
        )
        return
    try:
        ZONEINFO_FACTORY(tz_name)
    except Exception as exc:
        _finding(
            findings,
            "config.timezone.invalid",
            "warn",
            f"Nautical timezone '{tz_name}' is not available; hooks will use UTC fallback.",
            fix="Install system tzdata, or on Termux/Python environments run: python3 -m pip install tzdata.",
            details={"tz": tz_name, "error": str(exc)},
        )
        return
    _finding(findings, "config.timezone", "ok", f"Nautical timezone is available: {tz_name}")


def _check_panel_config(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    mode = str(data.get("panel_mode") or "rich").strip().lower() or "rich"
    duration_spec = config_schema.CONFIG_SPECS["live_panel_duration_ms"]
    default_duration = int(duration_spec["default"])
    min_duration = int(duration_spec["min"])
    max_duration = int(duration_spec["max"])
    raw_duration = data.get("live_panel_duration_ms", default_duration)
    duration_valid = True
    try:
        configured_duration = int(str(raw_duration).strip())
    except Exception:
        configured_duration = default_duration
        duration_valid = False

    effective_duration = max(min_duration, min(max_duration, configured_duration))
    if not duration_valid:
        _finding(
            findings,
            "config.panel.duration.invalid",
            "warn",
            f"live_panel_duration_ms is invalid ({raw_duration!r}); "
            f"the effective duration is {default_duration} ms.",
            fix=f"Set live_panel_duration_ms to an integer from {min_duration} to {max_duration}.",
            details={"configured_duration_ms": raw_duration, "effective_duration_ms": default_duration},
        )
    elif effective_duration != configured_duration:
        _finding(
            findings,
            "config.panel.duration.clamped",
            "warn",
            f"live_panel_duration_ms is {configured_duration}; Nautical clamps it to {effective_duration} ms.",
            fix=f"Set live_panel_duration_ms to an integer from {min_duration} to {max_duration}.",
            details={
                "configured_duration_ms": configured_duration,
                "effective_duration_ms": effective_duration,
            },
        )

    if mode != "live":
        return

    try:
        rich_available = RICH_SPEC_FACTORY("rich") is not None
    except Exception:
        rich_available = False
    motion = "disabled" if effective_duration == 0 else f"{effective_duration} ms"
    rich_state = "available" if rich_available else "unavailable"
    _finding(
        findings,
        "config.panel.live",
        "ok",
        f"Live panels use {motion} effective duration; Rich is {rich_state}; non-TTY output uses static fallback.",
        details={
            "configured_duration_ms": raw_duration if not duration_valid else configured_duration,
            "effective_duration_ms": effective_duration,
            "rich_available": rich_available,
            "non_tty_fallback": "static",
        },
    )
    if not rich_available:
        _finding(
            findings,
            "config.panel.rich_missing",
            "warn",
            "panel_mode is live, but Rich is not installed; panels will use the static fallback.",
            fix="Run python3 -m pip install rich.",
        )


def _load_queue_status():
    path = TOOLS_DIR / "nautical_queue_status.py"
    spec = importlib.util.spec_from_file_location("_nautical_doctor_queue_status", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load queue status helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_queue(findings: list[dict[str, Any]], taskdata: Path, stale_after: float) -> dict[str, Any]:
    try:
        module = _load_queue_status()
        payload = module._status_payload(taskdata, stale_after=stale_after, limit=5)
    except Exception as exc:
        _finding(
            findings,
            "queue.unreadable",
            "error",
            "Nautical queue state could not be inspected.",
            details={"error": str(exc)},
        )
        return {}
    issues = payload.get("issues") or []
    _finding(
        findings,
        "queue.state",
        "warn" if issues else "ok",
        "Queue state has findings." if issues else "Queue and dead-letter state are clean.",
        fix="Run nautical queue-status for queue details." if issues else "",
        details={"issues": issues} if issues else None,
    )
    return payload


def _short_uuid(value: object) -> str:
    raw = str(value or "").strip().lower()
    return raw.split("-")[0] if "-" in raw else raw[:8]


def _task_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "uuid": str(row.get("uuid") or ""),
        "description": str(row.get("description") or ""),
        "status": str(row.get("status") or ""),
        "chainID": str(row.get("chainID") or ""),
        "link": row.get("link"),
    }


def _chain_repair_detail(repair: chain_repair.LinkRepair) -> dict[str, Any]:
    return {
        "task": repair.short,
        "chainID": repair.chain_id,
        "link": repair.link,
        "field": repair.field,
        "old": repair.old,
        "new": repair.new,
    }


def _existing_reconcile_children(rows: list[dict[str, Any]], parent: dict[str, Any]) -> list[dict[str, Any]]:
    chain_id = str(parent.get("chainID") or "").strip()
    next_link = reconcile.int_or_default(parent.get("link"), 1) + 1
    include_deleted = str(parent.get("status") or "").strip() == "deleted"
    return [
        row
        for row in rows
        if str(row.get("chainID") or "").strip() == chain_id
        and reconcile.int_or_default(row.get("link"), -1) == next_link
        and (include_deleted or str(row.get("status") or "").strip() != "deleted")
    ]


def _load_on_modify_hook_for_reconcile(implementation: Path | None):
    if implementation is None:
        raise RuntimeError("No validated Nautical on-modify implementation is available.")
    loader = importlib.machinery.SourceFileLoader("_nautical_doctor_on_modify", str(implementation))
    spec = importlib.util.spec_from_loader("_nautical_doctor_on_modify", loader)
    if spec is None:
        raise RuntimeError(f"could not load {implementation}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    if hasattr(module, "_load_core"):
        module._load_core()
    return module


def _check_reconcile_plans(
    findings: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    on_modify_impl: Path | None,
) -> None:
    completion_candidates = [row for row in rows if reconcile.is_orphan_completion_candidate(row)]
    deleted_candidates = [row for row in rows if reconcile.is_orphan_deleted_chain_candidate(row)]
    if not completion_candidates and not deleted_candidates:
        return

    hook = None
    plans: list[reconcile.ReconcilePlan] = []
    delayed_expiration_candidates: list[dict[str, Any]] = []
    unavailable = ""
    if deleted_candidates:
        try:
            hook = _load_on_modify_hook_for_reconcile(on_modify_impl)
        except Exception as exc:
            unavailable = str(exc)
    candidates = [*completion_candidates, *deleted_candidates]
    if not unavailable:
        for parent in candidates:
            existing_children = _existing_reconcile_children(rows, parent)
            if not existing_children and hook is None:
                try:
                    hook = _load_on_modify_hook_for_reconcile(on_modify_impl)
                except Exception as exc:
                    unavailable = str(exc)
                    break
            plan = reconcile.build_reconcile_plan(parent, existing_children=existing_children, hook=hook)
            plans.append(plan)
            if str(parent.get("status") or "").strip().lower() != "deleted":
                continue
            continues_through_deleted = any(
                str(child.get("status") or "").strip().lower() == "deleted"
                for child in existing_children
            )
            planned_until_elapsed = False
            if plan.action == "spawn" and isinstance(plan.child, dict) and hook is not None:
                try:
                    until_dt, until_err = hook._safe_parse_datetime(plan.child.get("until"))
                    now_utc = getattr(getattr(hook, "core", None), "now_utc", None)
                    planned_until_elapsed = (
                        not until_err
                        and until_dt is not None
                        and callable(now_utc)
                        and until_dt <= now_utc()
                    )
                except Exception:
                    planned_until_elapsed = False
            if continues_through_deleted or planned_until_elapsed:
                delayed_expiration_candidates.append(_task_detail(parent))

    if unavailable:
        _finding(
            findings,
            "chains.reconcile_unavailable",
            "warn",
            f"{len(candidates)} recurrence candidate(s) were found, but reconcile planning could not load on-modify.",
            fix="Run nautical reconcile for full diagnostics.",
            details={
                "error": unavailable,
                "candidates": [
                    _task_detail(row)
                    for row in candidates[:10]
                ],
            },
        )
        return

    if not plans:
        return

    action_counts: dict[str, int] = defaultdict(int)
    for plan in plans:
        action_counts[plan.action] += 1
    fmt_dt_local = getattr(getattr(hook, "core", None), "fmt_dt_local", None) if hook is not None else None
    _finding(
        findings,
        "chains.reconcile_available",
        "warn",
        f"{len(plans)} recurrence reconcile plan(s) are available.",
        fix="Run nautical reconcile --apply after reviewing the dry-run output.",
        details={
            "actions": dict(sorted(action_counts.items())),
            "delayed_expiration_candidates": delayed_expiration_candidates[:10],
            "delayed_recovery": (
                "nautical reconcile --apply will continue through expired successors "
                "up to its configured hop limit."
                if delayed_expiration_candidates
                else ""
            ),
            "plans": [
                {
                    "action": plan.action,
                    **reconcile.describe_plan(plan, fmt_dt_local=fmt_dt_local),
                }
                for plan in plans[:10]
            ],
        },
    )


def _check_chains(
    findings: list[dict[str, Any]],
    *,
    task_bin: str,
    env: dict[str, str],
    on_modify_impl: Path | None,
) -> dict[str, int]:
    ok, rows, err = _task_export(task_bin, env)
    if not ok:
        _finding(
            findings,
            "chains.export",
            "error",
            "Task data could not be exported for chain inspection.",
            details={"error": err},
        )
        return {"tasks": 0, "nautical_tasks": 0, "chains": 0}

    repairs, repair_issues = chain_repair.plan_chain_link_repairs(rows)
    if repairs:
        _finding(
            findings,
            "chains.repair_available",
            "warn",
            f"{len(repairs)} safe chain repair(s) are available.",
            fix="Run nautical chain-repair --apply after reviewing the dry-run output.",
            details={"repairs": [_chain_repair_detail(repair) for repair in repairs[:10]]},
        )
    if repair_issues:
        reason_counts: dict[str, int] = defaultdict(int)
        for issue in repair_issues:
            for task in issue.tasks:
                reason = str(task.get("reason") or issue.message or issue.kind).strip()
                reason_counts[reason] += 1
        _finding(
            findings,
            "chains.repair_review",
            "warn",
            f"{len(repair_issues)} chain repair issue(s) need review.",
            fix="Run nautical chain-repair and inspect the 'why:' lines.",
            details={
                "reasons": dict(sorted(reason_counts.items())),
                "issues": [
                    {
                        "kind": issue.kind,
                        "chainID": issue.chain_id,
                        "message": issue.message,
                        "tasks": issue.tasks[:5],
                    }
                    for issue in repair_issues[:10]
                ],
            },
        )
    _check_reconcile_plans(findings, rows=rows, on_modify_impl=on_modify_impl)

    nautical = [
        row
        for row in rows
        if any(str(row.get(field) or "").strip() for field in RECURRENCE_FIELDS)
        or str(row.get("chainID") or "").strip()
    ]
    by_short: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        short = _short_uuid(row.get("uuid"))
        if short:
            by_short[short].append(row)

    missing_chain = [row for row in nautical if any(row.get(field) for field in RECURRENCE_FIELDS) and not row.get("chainID")]
    if missing_chain:
        _finding(
            findings,
            "chains.missing_chainid",
            "error",
            f"{len(missing_chain)} Nautical task(s) are missing chainID.",
            fix="Run dev_tools/nautical_backfill_chainid.py, review its output, then retry.",
            details={"tasks": [_task_detail(row) for row in missing_chain[:10]]},
        )

    slots: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in nautical:
        if str(row.get("status") or "").lower() == "deleted":
            continue
        chain_id = str(row.get("chainID") or "").strip()
        try:
            link_no = int(row.get("link"))
        except Exception:
            continue
        if chain_id:
            slots[(chain_id, link_no)].append(row)
    duplicates = {slot: members for slot, members in slots.items() if len(members) > 1}
    if duplicates:
        sample = [
            {
                "chainID": chain_id,
                "link": link,
                "tasks": [_task_detail(row) for row in members],
            }
            for (chain_id, link), members in list(duplicates.items())[:10]
        ]
        _finding(
            findings,
            "chains.duplicate_slots",
            "error",
            f"{len(duplicates)} duplicate chain slot(s) were found.",
            fix="Inspect the duplicate tasks before deleting or merging anything.",
            details={"slots": sample},
        )

    dangling: list[dict[str, Any]] = []
    nonreciprocal: list[dict[str, Any]] = []
    for row in nautical:
        uuid = str(row.get("uuid") or "")
        current_short = _short_uuid(uuid)
        for field, reciprocal in (("prevLink", "nextLink"), ("nextLink", "prevLink")):
            token = _short_uuid(row.get(field))
            if not token:
                continue
            matches = by_short.get(token, [])
            if len(matches) != 1:
                dangling.append(
                    {
                        "task": _task_detail(row),
                        "field": field,
                        "target": token,
                        "matches": len(matches),
                    }
                )
                continue
            target = matches[0]
            if _short_uuid(target.get(reciprocal)) != current_short:
                nonreciprocal.append(
                    {
                        "task": _task_detail(row),
                        "field": field,
                        "target": _task_detail(target),
                        "expected_reciprocal": reciprocal,
                    }
                )
    if dangling:
        _finding(
            findings,
            "chains.dangling_links",
            "warn",
            f"{len(dangling)} unresolved chain link(s) were found.",
            details={"links": dangling[:10]},
        )
    if nonreciprocal:
        _finding(
            findings,
            "chains.nonreciprocal_links",
            "warn",
            f"{len(nonreciprocal)} non-reciprocal chain link(s) were found.",
            details={"links": nonreciprocal[:10]},
        )
    if not any(item["id"].startswith("chains.") and item["severity"] != "ok" for item in findings):
        _finding(findings, "chains.integrity", "ok", f"Chain integrity is clean across {len(nautical)} Nautical task(s).")

    return {
        "tasks": len(rows),
        "nautical_tasks": len(nautical),
        "chains": len({str(row.get("chainID") or "") for row in nautical if row.get("chainID")}),
    }


def _overall_status(findings: list[dict[str, Any]]) -> str:
    worst = max((SEVERITY_RANK.get(str(item.get("severity")), 0) for item in findings), default=0)
    return "error" if worst >= 2 else "warn" if worst == 1 else "ok"


def _format_task(task: dict[str, Any]) -> str:
    uuid = str(task.get("uuid") or "")
    short = _short_uuid(uuid) or "unknown"
    description = str(task.get("description") or "").strip() or "(no description)"
    parts = [f"{short} {description}"]
    chain_id = str(task.get("chainID") or "").strip()
    link = task.get("link")
    status = str(task.get("status") or "").strip()
    if chain_id:
        parts.append(f"chain={chain_id}")
    if link not in (None, ""):
        parts.append(f"link={link}")
    if status:
        parts.append(f"status={status}")
    return " | ".join(parts)


def _render_details(details: dict[str, Any]) -> None:
    error = str(details.get("error") or "").strip()
    if error:
        print(f"  Detail: {error}")
    for issue in details.get("issues") or []:
        if not isinstance(issue, dict):
            print(f"  Detail: {issue}")
            continue
        print(
            "  Issue: "
            f"{issue.get('kind') or '?'} chain={issue.get('chainID') or '?'} "
            f"{issue.get('message') or ''}".rstrip()
        )
        for task in issue.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            print(f"    Task: {_format_task(task)}")
            reason = str(task.get("reason") or "").strip()
            if reason:
                print(f"      Why: {reason}")
    for reason, count in (details.get("reasons") or {}).items():
        print(f"  Reason: {reason} ({count})")
    for repair in details.get("repairs") or []:
        if not isinstance(repair, dict):
            continue
        print(
            "  Repair: "
            f"{repair.get('task') or '?'} chain={repair.get('chainID') or '?'} "
            f"link={repair.get('link') or '?'} {repair.get('field') or '?'}: "
            f"{repair.get('old') or '-'} -> {repair.get('new') or '-'}"
        )
    for action, count in (details.get("actions") or {}).items():
        print(f"  Action: {action} ({count})")
    for plan in details.get("plans") or []:
        if not isinstance(plan, dict):
            continue
        print(
            "  Plan: "
            f"{plan.get('action') or '?'} parent={plan.get('parent') or '?'} "
            f"chain={plan.get('chainID') or '?'} next={plan.get('next_link') or '?'}"
        )
        reason = str(plan.get("reason") or "").strip()
        if reason:
            print(f"    Reason: {reason}")
        child = plan.get("existing_child") or plan.get("child_target") or plan.get("child_due")
        if child:
            print(f"    Child: {child}")
    for task in details.get("tasks") or []:
        if isinstance(task, dict):
            print(f"  Affected: {_format_task(task)}")
    for slot in details.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        print(f"  Slot: chain={slot.get('chainID') or '?'} link={slot.get('link')}")
        for task in slot.get("tasks") or []:
            if isinstance(task, dict):
                print(f"    Task: {_format_task(task)}")
    for link in details.get("links") or []:
        if not isinstance(link, dict):
            continue
        task = link.get("task")
        source = _format_task(task) if isinstance(task, dict) else "unknown task"
        target = link.get("target")
        target_text = _format_task(target) if isinstance(target, dict) else str(target or "?")
        field = str(link.get("field") or "link")
        matches = link.get("matches")
        suffix = f" ({matches} matches)" if matches is not None else ""
        print(f"  Affected: {source}")
        print(f"    {field} -> {target_text}{suffix}")


def _render_text(payload: dict[str, Any]) -> None:
    print(f"Nautical doctor: {payload['status']}")
    print(f"Taskdata: {payload['taskdata']}")
    timezone = _timezone_summary(payload.get("findings") or [])
    if timezone:
        print(f"Timezone: {timezone}")
    for section in ("error", "warn", "ok"):
        items = [item for item in payload["findings"] if item["severity"] == section]
        if not items:
            continue
        print(f"\n{section.upper()} ({len(items)})")
        for item in items:
            print(f"- [{item['id']}] {item['message']}")
            details = item.get("details")
            if isinstance(details, dict):
                _render_details(details)
            if item.get("fix"):
                print(f"  Fix: {item['fix']}")


def _timezone_summary(findings: list[dict[str, Any]]) -> str:
    for item in findings:
        if not isinstance(item, dict):
            continue
        check_id = item.get("id")
        if check_id == "config.timezone":
            return str(item.get("message") or "").replace("Nautical timezone is available: ", "")
        if check_id in {"config.timezone.invalid", "config.timezone.unavailable"}:
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            tz_name = str(details.get("tz") or "?")
            return f"{tz_name} unavailable; UTC fallback active"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskdata", default=None)
    parser.add_argument("--task-bin", default=shutil.which("task") or "task")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument("--stale-after-seconds", type=float, default=300.0)
    args = parser.parse_args()

    env = os.environ.copy()
    if args.taskdata:
        taskdata_raw, _, _ = runtime.resolve_task_data_context(
            argv=[f"data:{args.taskdata}"],
            env=env,
            tw_dir=args.taskdata,
        )
    else:
        taskdata_raw, _, _ = runtime.resolve_task_data_context(
            env=env,
            tw_dir=_discover_taskdata(args.task_bin, env) or None,
        )
    taskdata = Path(taskdata_raw).expanduser().resolve()
    env["TASKDATA"] = str(taskdata)
    findings: list[dict[str, Any]] = []

    hooks_dir = _check_runtime(
        findings,
        task_bin=args.task_bin,
        taskdata=taskdata,
        env=env,
    )
    hook_runtimes = _check_hooks_and_udas(
        findings,
        task_bin=args.task_bin,
        hooks_dir=hooks_dir,
        env=env,
    )
    _check_managed_runtime(findings, hooks_dir)
    _check_config(findings, taskdata)
    queue = _check_queue(findings, taskdata, max(0.0, args.stale_after_seconds))
    on_modify = hook_runtimes.get("on-modify") or {}
    counts = _check_chains(
        findings,
        task_bin=args.task_bin,
        env=env,
        on_modify_impl=on_modify.get("implementation"),
    )

    status = _overall_status(findings)
    payload = {
        "status": status,
        "taskdata": str(taskdata),
        "hooks_dir": str(hooks_dir),
        "counts": counts,
        "queue": queue,
        "findings": findings,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        _render_text(payload)
    return 2 if status == "error" else 1 if status == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
