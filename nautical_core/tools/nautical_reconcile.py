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
import time
from pathlib import Path
from types import ModuleType
from typing import Any


CORE_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = CORE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
os.environ.setdefault("NAUTICAL_CORE_PATH", str(BASE_DIR))

from nautical_core import queue_store, reconcile, safe_lock, task_command  # noqa: E402
from nautical_core.chain_generation import ChainGenerationService  # noqa: E402
from nautical_core.lifecycle_executor import (  # noqa: E402
    LifecycleTerminalExecutor,
    LifecycleTransitionExecutor,
    OperationResult,
    OperationState,
)
from nautical_core.lifecycle_models import (  # noqa: E402
    DeletionDisposition,
    LifecycleEvent,
    LifecyclePlan,
    LifecycleOutcomeKind,
    TaskSnapshot,
)
from nautical_core.lifecycle_planner import terminal_plan_for_snapshot  # noqa: E402
from nautical_core.reconcile_gateway import TaskwarriorMutationGateway  # noqa: E402
from nautical_core.timeutil import compare_datetimes  # noqa: E402


_PARENT_LOCK_RETRIES = 600
_PARENT_LOCK_SLEEP_SECONDS = 0.1
_PARENT_LOCK_STALE_SECONDS = 300.0
_RECONCILE_LOCK_STALE_SECONDS = 300.0
_DEFAULT_EXPIRATION_HOPS = 32
_MAX_EXPIRATION_HOPS = 1000
_RECONCILE_PROTOCOL = 2
_JSON_SCHEMA = "nautical.reconcile"
_JSON_SCHEMA_VERSION = 1
_EXPORT_STATS = {"calls": 0, "rows": 0, "seconds": 0.0, "slowest_seconds": 0.0, "snapshot_hits": 0}
_LOCK_STATS = {"reconcile_busy": 0, "parent_busy": 0}


class _ConfigurationDrift(RuntimeError):
    """Signal that this run must stop before applying under a new configuration."""


class _LifecycleRetryable(TimeoutError):
    """Signal that a lifecycle read/write should remain retryable."""


class _LifecycleManualReview(RuntimeError):
    """Signal that a lifecycle transition needs operator review."""


class _RecoveryLookupUnavailable(TimeoutError):
    """Signal that a narrow recovery-child read must be retried."""


class _PlanReadUnavailable(TimeoutError):
    """Signal that planning reads are unavailable and must not be treated as empty."""


class _ConfigurationVerification:
    """Validated configuration state used to gate reconcile mutations."""

    __slots__ = ("status", "reason")

    def __init__(self, status: str, reason: str = "") -> None:
        self.status = status
        self.reason = reason


class _ConfigurationReason(str):
    """String-compatible reason retaining the tri-state verification result."""

    status: str

    def __new__(cls, reason: str, status: str):
        value = str.__new__(cls, reason)
        value.status = status
        return value

_ANSI = {
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "reset": "\033[0m",
}


def _style(text: str, color: str, *, stream: Any = None) -> str:
    """Apply terminal color only for interactive, color-capable output."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") or not getattr(stream, "isatty", lambda: False)():
        return text
    return f"{_ANSI.get(color, '')}{text}{_ANSI['reset']}"


def _action_style(action: str) -> str:
    return {
        "spawn": "green",
        "backfill_nextlink": "cyan",
        "repair_until": "green",
        "legitimate_final": "yellow",
        "manual_stop": "yellow",
        "stale": "dim",
        "partial": "yellow",
        "error": "red",
        "repair_error": "red",
        "manual_review": "yellow",
    }.get(action, "cyan")


def _format_local_until(hook: Any, value: Any) -> str:
    """Render a repaired native-until target in configured local time when possible."""
    raw = str(value or "").strip()
    if not raw:
        return raw
    parser = getattr(hook, "safe_parse_datetime", None)
    if not callable(parser):
        parser = getattr(hook, "_safe_parse_datetime", None)
    formatter = getattr(getattr(hook, "core", None), "fmt_dt_local", None)
    if not callable(parser) or not callable(formatter):
        return raw
    try:
        parsed, error = parser(raw)
        if parsed is not None and not error:
            return str(formatter(parsed))
    except Exception:
        pass
    return raw


def _safe_parse_datetime(hook: Any, value: Any):
    # Heavy hook implementations keep this boundary private because they are
    # loaded as executable modules.  Prefer the public adapter when present,
    # then use the hook's typed private parser; never silently treat a missing
    # parser as a valid/absent timestamp.
    parser = getattr(hook, "safe_parse_datetime", None)
    if not callable(parser):
        parser = getattr(hook, "_safe_parse_datetime", None)
    if callable(parser):
        return parser(value)
    return None, "datetime parser unavailable"


def _stable_child_uuid(hook: Any, parent: dict[str, Any], child: dict[str, Any]) -> str:
    resolver = getattr(hook, "stable_child_uuid", None)
    return str(resolver(parent, child) or "") if callable(resolver) else ""


def _spawn_child(hook: Any, child: dict[str, Any], parent: dict[str, Any]) -> tuple[str, set[str]]:
    spawn = getattr(hook, "spawn_child", None)
    if not callable(spawn):
        raise RuntimeError("Taskwarrior mutation gateway does not provide child spawning")
    return spawn(child, parent)


def _candidate_on_modify_paths(explicit: str | None = None) -> list[Path]:
    candidates: list[Path] = []
    for raw in (explicit, os.environ.get("NAUTICAL_ON_MODIFY_PATH")):
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend(
        [
            CORE_DIR / "hooks" / "modify_impl.py",
            BASE_DIR / "on-modify.nautical",
            BASE_DIR / "hooks" / "on-modify.nautical",
            BASE_DIR / "on-modify-nautical.py",
            BASE_DIR / "hooks" / "on-modify-nautical.py",
            BASE_DIR / "hooks" / "on-modify",
            CORE_DIR / "on-modify.nautical",
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
    _EXPORT_STATS["calls"] += 1
    started = time.perf_counter()
    try:
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
        rows = [row for row in payload if isinstance(row, dict)]
        _EXPORT_STATS["rows"] += len(rows)
        return rows
    finally:
        elapsed = time.perf_counter() - started
        _EXPORT_STATS["seconds"] += elapsed
        _EXPORT_STATS["slowest_seconds"] = max(_EXPORT_STATS["slowest_seconds"], elapsed)


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


_DEFAULT_LOAD_ON_MODIFY = _load_on_modify


def _load_reconcile_runtime(task_bin: str, hook_path: str | None = None):
    """Load the lightweight operator runtime; legacy hooks remain opt-in."""
    if hook_path is not None or _load_on_modify is not _DEFAULT_LOAD_ON_MODIFY:
        return _load_on_modify(hook_path), True
    import nautical_core as core

    return TaskwarriorMutationGateway(core, task_bin=task_bin), False


def _bind_hook_task_bin(hook: Any, task_bin: str) -> None:
    original_prefix = getattr(hook, "task_cmd_prefix", None)
    if not callable(original_prefix):
        original_prefix = getattr(hook, "_task_cmd_prefix", None)
    if not callable(original_prefix):
        # Public gateway runtimes do not need a hook command prefix. Test and
        # embedded runtimes may expose only the historical private spelling;
        # absence of either is a protocol error, not a reason to fabricate a
        # Taskwarrior command.
        raise RuntimeError("on-modify reconcile protocol is missing task command prefix")

    def _task_cmd_prefix() -> list[str]:
        prefix = list(original_prefix())
        if prefix:
            prefix[0] = task_bin
        return prefix

    hook.task_cmd_prefix = _task_cmd_prefix


def _validate_hook_protocol(hook: Any) -> None:
    if not isinstance(hook, ModuleType):
        return
    protocol = getattr(hook, "NAUTICAL_RECONCILE_PROTOCOL", None)
    if protocol != _RECONCILE_PROTOCOL:
        raise RuntimeError(
            f"incompatible on-modify reconcile protocol {protocol!r}; "
            f"expected {_RECONCILE_PROTOCOL}"
        )
    required_core = (
        "coerce_int",
        "to_local",
        "utc_to_local_naive",
        "local_naive_to_utc",
        "build_local_datetime",
        "fmt_isoz",
    )
    missing: list[str] = []
    core = getattr(hook, "core", None)
    missing.extend(f"core.{name}" for name in required_core if not callable(getattr(core, name, None)))
    if missing:
        raise RuntimeError(f"on-modify reconcile protocol is missing: {', '.join(missing)}")


def _configuration_drift_reason(hook: Any) -> str:
    """Compatibility string for callers; failures are never treated as valid."""
    check = _configuration_verification(hook)
    if check.status == "valid":
        return ""
    return _ConfigurationReason(check.reason, check.status)


def _configuration_verification(hook: Any) -> _ConfigurationVerification:
    """Return valid, drifted, or unavailable configuration state."""
    core = getattr(hook, "core", None)
    checker = getattr(core, "configuration_drift", None)
    if not callable(checker):
        # Lightweight operator-test doubles predate the facade verifier. A
        # real imported Nautical module must never silently bypass this gate.
        if not isinstance(core, ModuleType):
            return _ConfigurationVerification("valid")
        return _ConfigurationVerification(
            "unavailable",
            "configuration verifier is unavailable; restart and rerun",
        )
    try:
        drift = checker()
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        return _ConfigurationVerification(
            "unavailable",
            f"configuration verification unavailable: {reason}; restart and rerun",
        )
    if not isinstance(drift, dict):
        return _ConfigurationVerification(
            "unavailable",
            "configuration verification returned an invalid result; restart and rerun",
        )
    if not drift.get("changed"):
        return _ConfigurationVerification("valid")
    source = str(drift.get("source") or "unknown")
    return _ConfigurationVerification(
        "drifted",
        f"configuration changed during reconcile (source: {source}); restart and rerun",
    )


def _configuration_state(hook: Any) -> tuple[str, str]:
    """Resolve the compatibility reason while retaining its tri-state status."""
    reason = _configuration_drift_reason(hook)
    return str(getattr(reason, "status", "drifted" if reason else "valid")), str(reason)


def _candidate_sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("chainID") or "").strip().casefold(),
        reconcile.int_or_default(row.get("link"), 0),
        str(row.get("status") or "").strip().casefold(),
        str(row.get("uuid") or "").strip().casefold(),
    )


class _ReconcileSnapshot:
    """Immutable read-phase views for active links and recovery candidates.

    Completed/deleted history is intentionally kept out of the active snapshot.
    Native-until repair reads a predecessor by chain/link only when an active
    row actually needs repair.
    """

    def __init__(self, task_bin: str):
        self.task_bin = task_bin
        self._active_rows: list[dict[str, Any]] | None = None
        self._candidate_rows: list[dict[str, Any]] | None = None

    def active_rows(self) -> list[dict[str, Any]]:
        if self._active_rows is None:
            self._active_rows = _export(
                self.task_bin,
                ["chain:on", "chainID.not:", "status.not:completed", "status.not:deleted"],
            )
        else:
            _EXPORT_STATS["snapshot_hits"] += 1
        return self._active_rows

    def candidate_rows(self) -> list[dict[str, Any]]:
        if self._candidate_rows is None:
            completed = _export(
                self.task_bin,
                ["status:completed", "chain:on", "chainID.not:", "nextLink:"],
            )
            deleted = _export(
                self.task_bin,
                ["status:deleted", "chain:on", "chainID.not:", "nextLink:"],
            )
            self._candidate_rows = completed + deleted
        else:
            _EXPORT_STATS["snapshot_hits"] += 1
        return self._candidate_rows


_READ_SNAPSHOT: _ReconcileSnapshot | None = None


def _candidate_rows(
    task_bin: str,
    hook: Any,
    *,
    snapshot: _ReconcileSnapshot | None = None,
) -> list[dict[str, Any]]:
    snapshot = snapshot or _READ_SNAPSHOT
    if snapshot is not None:
        rows = snapshot.candidate_rows()
        candidates = [
            row
            for row in rows
            if str(row.get("status") or "").strip().lower() == "completed"
            and reconcile.is_orphan_completion_candidate(row)
        ]
        candidates.extend(
            row
            for row in rows
            if str(row.get("status") or "").strip().lower() == "deleted"
            and reconcile.is_orphan_deleted_chain_candidate(row)
        )
        return sorted(candidates, key=_candidate_sort_key)
    completed = _export(task_bin, ["status:completed", "chain:on", "chainID.not:", "nextLink:"])
    deleted = _export(task_bin, ["status:deleted", "chain:on", "chainID.not:", "nextLink:"])
    rows = [row for row in completed if reconcile.is_orphan_completion_candidate(row)]
    rows.extend(row for row in deleted if reconcile.is_orphan_deleted_chain_candidate(row))
    return sorted(rows, key=_candidate_sort_key)


def _ambiguous_candidate_slots(rows: list[dict[str, Any]]) -> dict[tuple[str, int], str]:
    """Return candidate slots with more than one distinct parent identity."""
    grouped: dict[tuple[str, int], set[str]] = {}
    for row in rows:
        chain_id = str(row.get("chainID") or "").strip()
        link = reconcile.int_or_default(row.get("link"), 0)
        uuid = str(row.get("uuid") or "").strip().lower()
        if chain_id and link > 0 and uuid:
            grouped.setdefault((chain_id, link), set()).add(uuid)
    return {
        slot: (
            f"ambiguous candidate slot chain {slot[0]} link {slot[1]} "
            f"has {len(uuids)} distinct parent tasks"
        )
        for slot, uuids in grouped.items()
        if len(uuids) > 1
    }


def _active_chain_rows(
    task_bin: str,
    *,
    include_inactive: bool = False,
    snapshot: _ReconcileSnapshot | None = None,
) -> list[dict[str, Any]]:
    """Export live Nautical links for integrity checks, independently of recovery candidates."""
    rows = (
        snapshot.active_rows()
        if snapshot is not None
        else _export(task_bin, ["chain:on", "chainID.not:", "status.not:completed", "status.not:deleted"])
    )
    return sorted(
        (
            row
            for row in rows
            if str(row.get("status") or "").strip().lower() not in {"completed", "deleted"}
        ),
        key=_candidate_sort_key,
    )


def _native_until_guard_error(expected: dict[str, Any], fresh: dict[str, Any]) -> str | None:
    """Detect target or recurrence changes made after the audit export."""
    fields = (
        "uuid", "status", "chain", "chainID", "link", "due", "scheduled", "until",
        "anchor", "anchor_file", "cp", "chainMax", "chainUntil",
    )
    for field in fields:
        left = expected.get(field)
        right = fresh.get(field)
        if field == "link":
            left = reconcile.int_or_default(left, 0)
            right = reconcile.int_or_default(right, 0)
        else:
            left = str(left or "").strip()
            right = str(right or "").strip()
        if left != right:
            return f"native-until target changed ({field}: {left or '<empty>'} -> {right or '<empty>'})"
    return None


def _fresh_native_until_previous(task_bin: str, row: dict[str, Any]) -> dict[str, Any] | None:
    chain_id = str(row.get("chainID") or "").strip()
    link = reconcile.int_or_default(row.get("link"), 0)
    if not chain_id or link <= 1:
        return None
    rows = _export(task_bin, [f"chainID:{chain_id}", f"link:{link - 1}"], timeout=30.0)
    matches = [
        candidate
        for candidate in rows
        if str(candidate.get("chainID") or "").strip() == chain_id
        and reconcile.int_or_default(candidate.get("link"), 0) == link - 1
    ]
    return matches[0] if len(matches) == 1 else None


def _native_until_repairs(
    task_bin: str,
    hook: Any,
    *,
    apply: bool,
    taskdata: Path | None = None,
    snapshot: _ReconcileSnapshot | None = None,
    lease_held: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find invalid native windows and repair only those with a reliable predecessor."""
    active_rows = _active_chain_rows(
        task_bin,
        include_inactive=False,
        snapshot=snapshot or _READ_SNAPSHOT,
    )
    rows = active_rows
    by_chain_link = {
        (
            str(row.get("chainID") or "").strip(),
            reconcile.int_or_default(row.get("link"), 0),
        ): row
        for row in active_rows
    }
    repairs: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        reason = reconcile.invalid_native_until_reason(row, safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value))
        if not reason:
            continue
        chain_id = str(row.get("chainID") or "").strip()
        link = reconcile.int_or_default(row.get("link"), 0)
        previous = by_chain_link.get((chain_id, link - 1))
        if previous is None:
            # Historical predecessors are deliberately outside the active
            # snapshot; fetch only the predecessor needed by this invalid row.
            previous = _fresh_native_until_previous(task_bin, row)
        item = {
            "task": reconcile.short_uuid(row.get("uuid")),
            "chainID": chain_id,
            "link": link,
            "target": row.get("due") or row.get("scheduled"),
            "until": row.get("until"),
            "reason": reason,
        }
        repaired: str | None = None
        repair_error: str | None = None
        if previous is None:
            repair_error = "previous link is unavailable"
        else:
            previous_reason = reconcile.invalid_native_until_reason(
                previous,
                safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
            )
            if previous_reason:
                repair_error = f"previous link is invalid: {previous_reason}"
            else:
                kind = reconcile.recurrence_kind(row)
                repaired, repair_error = reconcile.repair_native_until_from_previous(
                    previous,
                    row,
                    kind=kind,
                    safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
                    fmt_isoz=hook.core.fmt_isoz,
                    utc_to_local_naive=hook.core.utc_to_local_naive,
                    local_naive_to_utc=hook.core.local_naive_to_utc,
                )
        if repair_error or not repaired:
            fallback, fallback_error = reconcile.fallback_native_until_at_day_end(
                row,
                    safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
                fmt_isoz=hook.core.fmt_isoz,
                utc_to_local_naive=hook.core.utc_to_local_naive,
                local_naive_to_utc=hook.core.local_naive_to_utc,
            )
            if fallback_error or not fallback:
                item["action"] = "manual_review"
                item["repair_error"] = fallback_error or repair_error or "could not calculate repaired until"
                repairs.append(item)
                continue
            repaired = fallback
            item["fallback"] = "local 23:00"
            item["reason"] = repair_error or item["reason"]
        item["action"] = "repair_until"
        item["new_until"] = repaired
        if apply:
            if taskdata is None:
                item["action"] = "repair_error"
                item["repair_error"] = "Taskwarrior data location is unavailable for native-until locking"
                errors.append(f"{item['task']} chain {chain_id} link {link}: {item['repair_error']}")
                repairs.append(item)
                continue
            with _reconcile_mutation_lock(taskdata, lease_held=lease_held) as reconcile_acquired:
                if not reconcile_acquired:
                    _LOCK_STATS["reconcile_busy"] += 1
                    item["action"] = "repair_error"
                    item["repair_error"] = "another reconcile apply is already running"
                else:
                    parent_lock = _parent_apply_lock(taskdata, str(row.get("uuid") or ""))
                    with parent_lock as acquired:
                        if not acquired:
                            _LOCK_STATS["parent_busy"] += 1
                            item["action"] = "repair_error"
                            item["repair_error"] = "native-until repair lock busy"
                        else:
                            fresh = _fresh_parent(task_bin, row)
                            guard_error = _native_until_guard_error(row, fresh) if fresh else "native-until target disappeared"
                            fresh_previous = _fresh_native_until_previous(task_bin, fresh or row)
                            if not guard_error:
                                if (previous is None) != (fresh_previous is None):
                                    guard_error = "native-until predecessor changed during repair"
                                elif previous is not None and fresh_previous is not None:
                                    guard_error = _native_until_guard_error(previous, fresh_previous)
                            if guard_error:
                                item["action"] = "repair_error"
                                item["repair_error"] = guard_error
                            else:
                                configuration_status, drift_reason = _configuration_state(hook)
                                if configuration_status != "valid":
                                    item["action"] = "manual_review"
                                    item["repair_error"] = drift_reason
                                    item["configuration_drift"] = True
                                    item["configuration_status"] = configuration_status
                                    repairs.append(item)
                                    return repairs, errors
                                try:
                                    _modify_native_until(task_bin, fresh, repaired)
                                    verified = _fresh_parent(task_bin, fresh)
                                    if verified is None or not _native_until_matches(verified, repaired, hook):
                                        actual = str((verified or {}).get("until") or "<missing>")
                                        item["action"] = "repair_error"
                                        item["repair_error"] = (
                                            f"native until repair verification failed (expected {repaired}; found {actual})"
                                        )
                                    else:
                                        item["applied"] = True
                                        by_chain_link[(chain_id, link)] = dict(verified)
                                except Exception as exc:
                                    item["action"] = "repair_error"
                                    item["repair_error"] = str(exc).strip() or type(exc).__name__
                if item.get("action") == "repair_error":
                    errors.append(f"{item['task']} chain {chain_id} link {link}: {item['repair_error']}")
        repairs.append(item)
    return repairs, errors


def _modify_native_until(task_bin: str, row: dict[str, Any], new_until: str) -> None:
    uuid = str(row.get("uuid") or "").strip()
    chain_id = str(row.get("chainID") or "").strip()
    link = reconcile.int_or_default(row.get("link"), 0)
    if not uuid or not chain_id or link <= 0:
        raise RuntimeError("native until repair lacks task identity")
    proc = _run_task(
        task_bin,
        [
            "rc.hooks=off",
            "rc.confirmation=off",
            "rc.verbose=nothing",
            f"uuid:{uuid}",
            "chain:on",
            f"chainID:{chain_id}",
            f"link:{link}",
            "modify",
            f"until:{new_until}",
        ],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(task_command.failure_message(proc, "native until repair"))


def _native_until_matches(fresh: dict[str, Any], expected: str, hook: Any) -> bool:
    """Compare native-until timestamps by instant, tolerating Taskwarrior formatting."""
    actual = str(fresh.get("until") or "").strip()
    if actual == str(expected or "").strip():
        return True
    try:
        actual_dt, actual_err = _safe_parse_datetime(hook, actual)
        expected_dt, expected_err = _safe_parse_datetime(hook, expected)
        return not actual_err and not expected_err and actual_dt is not None and actual_dt == expected_dt
    except Exception:
        return False


def _existing_children(task_bin: str, parent: dict[str, Any]) -> list[dict[str, Any]]:
    chain_id = str(parent.get("chainID") or "").strip()
    next_link = reconcile.int_or_default(parent.get("link"), 1) + 1
    if not chain_id:
        return []
    return _export(task_bin, [f"chainID:{chain_id}", f"link:{next_link}"], timeout=30.0)


def _existing_children_for_plan(task_bin: str, parent: dict[str, Any], hook: Any) -> list[dict[str, Any]]:
    if str(parent.get("status") or "").strip() == "deleted":
        evidence = reconcile.deleted_chain_disposition(
            parent,
            safe_parse_datetime=lambda value: _safe_parse_datetime(hook, value),
        )
        if evidence.disposition is not DeletionDisposition.EXPIRATION:
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


def _synchronize_taskdata_config(hook: Any, taskdata: Path | None) -> None:
    """Apply the validated config selected for the resolved Taskwarrior data directory."""
    if taskdata is None:
        return
    core = getattr(hook, "core", None)
    if core is None:
        raise RuntimeError("Nautical core is unavailable for configuration reload")
    reload_config = getattr(core, "reload_taskdata_config", None)
    if not callable(reload_config):
        # Lightweight fake cores used by operator-level unit tests do not
        # carry the facade API. A real imported Nautical module must provide it.
        if isinstance(core, ModuleType):
            raise RuntimeError("Nautical core does not provide validated configuration reload")
        return
    reload_config(taskdata)


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


@contextmanager
def _reconcile_apply_lock(taskdata: Path):
    """Serialize reconciler mutations without blocking a second invocation."""
    lock_path = queue_store.reconcile_lock_path(taskdata)
    with safe_lock(
        lock_path,
        retries=1,
        sleep_base=0.0,
        stale_after=_RECONCILE_LOCK_STALE_SECONDS,
    ) as acquired:
        yield acquired


@contextmanager
def _reconcile_mutation_lock(taskdata: Path, *, lease_held: bool):
    """Reuse the run lease when present, otherwise protect a direct mutation call."""
    if lease_held:
        yield True
        return
    with _reconcile_apply_lock(taskdata) as acquired:
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


def _is_legacy_root_without_link(parent: dict[str, Any]) -> bool:
    """Recognize only old root records whose link UDA was never stamped."""
    raw_link = parent.get("link")
    if raw_link is not None and str(raw_link).strip():
        return False
    parent_uuid = str(parent.get("uuid") or "").strip().lower()
    chain_id = str(parent.get("chainID") or "").strip().lower()
    if not parent_uuid or chain_id not in {parent_uuid, reconcile.short_uuid(parent_uuid).lower()}:
        return False
    return not str(parent.get("prevLink") or "").strip()


def _parent_identity_error(parent: dict[str, Any]) -> str:
    """Explain why a parent cannot be used as an atomic reconcile target."""
    chain_id = str(parent.get("chainID") or "").strip()
    if not chain_id:
        return "parent chainID is missing"
    if _is_legacy_root_without_link(parent):
        return ""

    raw_link = parent.get("link")
    if raw_link is None or not str(raw_link).strip():
        return "parent link is missing; run chain-repair --apply if the chain is deterministic"
    if isinstance(raw_link, bool):
        return f"parent link is invalid: {raw_link!r}"
    try:
        parsed_link = int(raw_link)
    except (TypeError, ValueError, OverflowError):
        return f"parent link is invalid: {raw_link!r}"
    if parsed_link <= 0:
        return f"parent link must be positive; got {parsed_link}"
    return ""


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
    identity_error = _parent_identity_error(parent)
    if identity_error:
        raise RuntimeError(identity_error)
    legacy_root = _is_legacy_root_without_link(parent)
    if str(parent.get("nextLink") or "").strip():
        raise RuntimeError("parent nextLink is already set")
    return [
        f"uuid:{parent_uuid}",
        f"status:{status}",
        "chain:on",
        f"chainID:{chain_id}",
        "link:" if legacy_root else f"link:{link}",
        "nextLink:",
    ]


def _modify_parent_nextlink(task_bin: str, parent: dict[str, Any], child_short: str) -> None:
    filters = _parent_guard_filters(parent)
    updates = ["link:1"] if _is_legacy_root_without_link(parent) else []
    updates.append(f"nextLink:{child_short}")
    proc = _run_task(
        task_bin,
        [
            "rc.hooks=off",
            "rc.confirmation=off",
            "rc.verbose=nothing",
            *filters,
            "modify",
            *updates,
        ],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(task_command.failure_message(proc, "parent nextLink update"))


def _disable_parent_chain(task_bin: str, parent: dict[str, Any]) -> None:
    filters = _parent_guard_filters(parent)
    updates = ["link:1"] if _is_legacy_root_without_link(parent) else []
    updates.append("chain:off")
    proc = _run_task(
        task_bin,
        [
            "rc.hooks=off",
            "rc.confirmation=off",
            "rc.verbose=nothing",
            *filters,
            "modify",
            *updates,
        ],
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(task_command.failure_message(proc, "parent chain update"))


def _verify_disabled_parent(task_bin: str, parent: dict[str, Any]) -> None:
    """Re-export a terminal parent before reporting chain disablement as applied."""
    fresh_parent = _fresh_parent(task_bin, parent)
    if fresh_parent is None:
        raise RuntimeError("post-apply verification could not re-export the disabled parent")
    if str(fresh_parent.get("chain") or "").strip().lower() != "off":
        shown = str(fresh_parent.get("chain") or "<empty>").strip() or "<empty>"
        raise RuntimeError(f"post-apply verification found parent chain {shown}; expected off")
    successor = str(fresh_parent.get("nextLink") or "").strip()
    if successor:
        raise RuntimeError(
            f"post-apply verification found successor {successor}; "
            "terminal chain must not remain spawnable"
        )


def _verify_applied_child(
    task_bin: str,
    parent: dict[str, Any],
    child_short: str,
    *,
    hook: Any = None,
    strict_uuid: bool = False,
) -> dict[str, Any]:
    """Re-export both sides of an apply before declaring the repair successful."""
    expected_child = str(child_short or "").strip().lower()
    if not expected_child:
        raise RuntimeError("post-apply verification has no child identity")
    fresh_parent = _fresh_parent(task_bin, parent)
    if fresh_parent is None:
        raise RuntimeError("post-apply verification could not re-export the parent")
    if str(fresh_parent.get("chainID") or "").strip() != str(parent.get("chainID") or "").strip():
        raise RuntimeError("post-apply verification found a changed parent chainID")
    linked_child = str(fresh_parent.get("nextLink") or "").strip().lower()
    if linked_child != expected_child and not linked_child.startswith(expected_child):
        shown = linked_child or "<empty>"
        raise RuntimeError(
            f"post-apply verification found parent nextLink {shown}; expected {child_short}"
        )
    rows = _existing_children(task_bin, fresh_parent)
    resolved, child_error = reconcile.resolve_existing_child(
        fresh_parent,
        rows,
        include_deleted=True,
    )
    if child_error:
        raise RuntimeError(f"post-apply child verification failed: {child_error}")
    if resolved.lower() != expected_child:
        shown = resolved or "<missing>"
        raise RuntimeError(
            f"post-apply child verification found {shown}; expected {child_short}"
        )
    matched = next(
        (
            row
            for row in rows
            if str(row.get("uuid") or "").strip().lower().startswith(expected_child)
        ),
        None,
    )
    if matched is None:
        raise RuntimeError("post-apply child verification could not identify the resolved child")
    if callable(getattr(hook, "stable_child_uuid", None)):
        expected_uuid = _stable_child_uuid(hook, fresh_parent, matched).strip().lower()
        actual_uuid = str(matched.get("uuid") or "").strip().lower()
        if strict_uuid and expected_uuid and actual_uuid != expected_uuid:
            raise RuntimeError(
                f"post-apply child UUID {actual_uuid[:8] or '<empty>'} "
                f"does not match deterministic slot identity {expected_uuid[:8]}"
            )
    return matched


def _stale_plan(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "stale",
        parent,
        reconcile.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _chain_generation_for_hook(hook: Any) -> ChainGenerationService:
    """Build the shared generator from configured core state only."""
    provided = getattr(hook, "chain_generation_service", None)
    if isinstance(provided, ChainGenerationService):
        return provided
    core = getattr(hook, "core", None)
    if core is None:
        raise RuntimeError("configured Nautical core is unavailable")
    if not callable(getattr(core, "parse_cp_sequence_tokens", None)):
        return ChainGenerationService.from_hook(hook)
    return ChainGenerationService.from_core(
        core,
        recurrence_update_udas=tuple(getattr(core, "RECURRENCE_UPDATE_UDAS", ()) or ()),
        debug_wait_sched=bool(getattr(core, "DEBUG_WAIT_SCHED", False)),
    )


def _refresh_plan(
    task_bin: str,
    hook: Any,
    original_parent: dict[str, Any],
    *,
    generation: ChainGenerationService | None = None,
) -> reconcile.ReconcilePlan:
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
    return _plan_for_parent(
        task_bin,
        hook,
        parent,
        generation=generation or _chain_generation_for_hook(hook),
    )


def _plan_for_parent(
    task_bin: str,
    hook: Any,
    parent: dict[str, Any],
    *,
    generation: ChainGenerationService | None = None,
) -> reconcile.ReconcilePlan:
    """Build the one reconcile plan used by both preview and apply paths."""
    configuration_status, configuration_reason = _configuration_state(hook)
    if configuration_status != "valid":
        raise _ConfigurationDrift(configuration_reason)
    try:
        existing_children = _existing_children_for_plan(task_bin, parent, hook)
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        raise _PlanReadUnavailable(f"reconcile child read unavailable: {reason}") from exc
    return reconcile.build_reconcile_plan(
        parent,
        existing_children=existing_children,
        hook=hook,
        generation=generation or _chain_generation_for_hook(hook),
    )


def _execute_reconcile_lifecycle_plan(
    task_bin: str,
    hook: Any,
    plan: reconcile.ReconcilePlan,
    *,
    verified_children: dict[str, dict[str, Any]] | None,
    label: str,
    strict_uuid: bool,
) -> str:
    """Execute and verify one reconcile spawn/backfill through the shared executor."""
    lifecycle_plan = getattr(plan, "lifecycle_plan", None)
    if not isinstance(lifecycle_plan, LifecyclePlan):
        raise RuntimeError(f"reconcile {label} plan has no typed lifecycle plan")
    services = _ReconcileLifecycleServices(task_bin, hook, plan.parent)
    outcome = LifecycleTransitionExecutor(services).execute(lifecycle_plan)
    if outcome.kind is LifecycleOutcomeKind.RETRYABLE:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] reconcile {label} retryable: {outcome.reason}", file=sys.stderr)
        raise _LifecycleRetryable(f"lifecycle {label} retryable: {outcome.reason}")
    if outcome.kind is not LifecycleOutcomeKind.APPLIED:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] reconcile {label} review: {outcome.reason}", file=sys.stderr)
        raise _LifecycleManualReview(f"lifecycle {label} requires review: {outcome.reason}")
    child_short = services.last_child_short or plan.child_short
    if not child_short:
        raise RuntimeError(f"lifecycle {label} produced no child identity")
    verified = _verify_applied_child(
        task_bin,
        plan.parent,
        child_short,
        hook=hook,
        strict_uuid=strict_uuid,
    )
    if verified_children is not None:
        verified_children[str(child_short).strip().lower()] = verified
    return child_short


def _terminal_lifecycle_plan(plan: reconcile.ReconcilePlan) -> LifecyclePlan:
    """Create the typed terminal plan for a reconcile final/manual decision."""
    parent = plan.parent
    if plan.action == "manual_stop":
        event = LifecycleEvent.MANUAL_DELETE
    elif str(parent.get("status") or "").strip().lower() == "deleted":
        event = LifecycleEvent.EXPIRE
    elif "chainmax" in str(plan.reason).replace(" ", "").lower():
        event = LifecycleEvent.CHAIN_MAX
    elif "chainuntil" in str(plan.reason).replace(" ", "").lower():
        event = LifecycleEvent.CHAIN_UNTIL
    else:
        event = LifecycleEvent.COMPLETE
    try:
        return terminal_plan_for_snapshot(
            TaskSnapshot.from_mapping(parent),
            event,
        )
    except Exception as exc:
        raise _LifecycleManualReview(
            f"terminal policy refused mutation: {str(exc).strip() or type(exc).__name__}"
        ) from exc


def _execute_reconcile_terminal_plan(
    task_bin: str,
    hook: Any,
    plan: reconcile.ReconcilePlan,
) -> str:
    """Apply a guarded terminal plan through the shared terminal executor."""
    lifecycle_plan = _terminal_lifecycle_plan(plan)
    services = _ReconcileLifecycleServices(task_bin, hook, plan.parent)
    outcome = LifecycleTerminalExecutor(services).execute(lifecycle_plan)
    if outcome.kind is LifecycleOutcomeKind.RETRYABLE:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] reconcile terminal retryable: {outcome.reason}", file=sys.stderr)
        raise _LifecycleRetryable(f"terminal transition retryable: {outcome.reason}")
    if outcome.kind is not LifecycleOutcomeKind.APPLIED:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] reconcile terminal review: {outcome.reason}", file=sys.stderr)
        raise _LifecycleManualReview(f"terminal transition requires review: {outcome.reason}")
    return "off"


def _apply_parent_atomic(
    task_bin: str,
    hook: Any,
    original_parent: dict[str, Any],
    *,
    taskdata: Path,
    lease_held: bool = False,
    verified_children: dict[str, dict[str, Any]] | None = None,
    generation: ChainGenerationService | None = None,
) -> tuple[reconcile.ReconcilePlan, str]:
    parent_uuid = str(original_parent.get("uuid") or "").strip()
    if not parent_uuid:
        raise RuntimeError("parent task has no UUID")
    with _reconcile_mutation_lock(taskdata, lease_held=lease_held) as reconcile_acquired:
        if not reconcile_acquired:
            _LOCK_STATS["reconcile_busy"] += 1
            raise RuntimeError("another reconcile apply is already running")
        with _parent_apply_lock(taskdata, parent_uuid) as acquired:
            if not acquired:
                _LOCK_STATS["parent_busy"] += 1
                raise RuntimeError(f"parent reconcile lock busy: {reconcile.short_uuid(parent_uuid)}")
            configuration_status, drift_reason = _configuration_state(hook)
            if configuration_status != "valid":
                raise _ConfigurationDrift(drift_reason)
            if generation is None:
                plan = _refresh_plan(task_bin, hook, original_parent)
            else:
                plan = _refresh_plan(
                    task_bin,
                    hook,
                    original_parent,
                    generation=generation,
                )
            if plan.action == "spawn":
                if not plan.child:
                    raise RuntimeError("spawn plan has no child payload")
                child_short = _execute_reconcile_lifecycle_plan(
                    task_bin,
                    hook,
                    plan,
                    verified_children=verified_children,
                    label="transition",
                    strict_uuid=True,
                )
                return plan, child_short
            if plan.action == "backfill_nextlink":
                child_short = _execute_reconcile_lifecycle_plan(
                    task_bin,
                    hook,
                    plan,
                    verified_children=verified_children,
                    label="backfill",
                    strict_uuid=False,
                )
                return plan, child_short
            if plan.action in {"legitimate_final", "manual_stop"}:
                return plan, _execute_reconcile_terminal_plan(task_bin, hook, plan)
            return plan, ""


def _recovery_error(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "error",
        parent,
        reconcile.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _recovery_terminal(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    """Classify an expired-but-still-pending child as resumable, not corrupt."""
    if reason.endswith("native until has already elapsed"):
        return _recovery_partial(
            parent,
            f"{reason}; wait for Taskwarrior to mark the child deleted, then rerun reconcile",
        )
    return _recovery_error(parent, reason)


def _recovery_partial(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "partial",
        parent,
        reconcile.int_or_default(parent.get("link"), 1) + 1,
        reason,
    )


def _recovery_manual_review(parent: dict[str, Any], reason: str) -> reconcile.ReconcilePlan:
    return reconcile.ReconcilePlan(
        "manual_review",
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
        until_dt, until_err = _safe_parse_datetime(hook, until_raw)
    except Exception:
        return "live recovery child native until could not be parsed"
    if until_err or until_dt is None:
        return f"live recovery child has no reliable native until: {until_err or 'missing until'}"

    target_field = "due" if child.get("due") else "scheduled"
    target_raw = child.get(target_field)
    try:
        target_dt, target_err = _safe_parse_datetime(hook, target_raw)
    except Exception:
        return f"live recovery child {target_field} could not be parsed"
    if target_err or target_dt is None:
        return f"live recovery child has no reliable {target_field}: {target_err or f'missing {target_field}'}"
    try:
        if compare_datetimes(until_dt, target_dt) <= 0:
            return f"live recovery child native until is not later than its {target_field}"
        if compare_datetimes(until_dt, recovery_at) <= 0:
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
    try:
        rows = _export(task_bin, [f"uuid:{wanted}"], timeout=30.0)
    except Exception as exc:
        reason = str(exc).strip() or type(exc).__name__
        raise _RecoveryLookupUnavailable(
            f"recovery child {wanted} lookup unavailable: {reason}"
        ) from exc
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
        until_dt, until_err = _safe_parse_datetime(hook, until_raw)
    except Exception:
        return None, "planned child expiration could not be parsed"
    if until_err or until_dt is None:
        return None, f"planned child has no reliable native until: {until_err or 'missing until'}"
    try:
        if compare_datetimes(until_dt, recovery_at) > 0:
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
    lease_held: bool = False,
    generation: ChainGenerationService | None = None,
) -> list[tuple[reconcile.ReconcilePlan, str]]:
    outcomes: list[tuple[reconcile.ReconcilePlan, str]] = []
    current = parent
    visited: set[tuple[str, int]] = set()
    expiration_hops = 0
    verified_children: dict[str, dict[str, Any]] = {}

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
                    lease_held=lease_held,
                    verified_children=verified_children,
                )
            except _ConfigurationDrift as exc:
                outcomes.append((_recovery_partial(current, str(exc)), ""))
                break
            except _LifecycleRetryable as exc:
                outcomes.append((_recovery_partial(current, str(exc)), ""))
                break
            except _LifecycleManualReview as exc:
                outcomes.append((_recovery_manual_review(current, str(exc)), ""))
                break
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                outcomes.append((_recovery_error(current, reason), ""))
                break
        else:
            try:
                plan = _plan_for_parent(
                    task_bin,
                    hook,
                    current,
                    generation=generation or _chain_generation_for_hook(hook),
                )
            except _ConfigurationDrift as exc:
                outcomes.append((_recovery_partial(current, str(exc)), ""))
                break
            except _PlanReadUnavailable as exc:
                outcomes.append((_recovery_partial(current, str(exc)), ""))
                break
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                outcomes.append((_recovery_error(current, reason), ""))
                break
            applied_short = ""
        outcomes.append((plan, applied_short))

        if not is_deleted or plan.action not in {"spawn", "backfill_nextlink"}:
            break
        expiration_hops += 1

        if apply or plan.action == "backfill_nextlink":
            child_short = applied_short or plan.child_short
            try:
                cached_child = verified_children.get(str(child_short or "").strip().lower())
                cached_terminal_error = (
                    _terminal_recovery_error(cached_child, hook, recovery_at)
                    if cached_child is not None
                    else ""
                )
                if (
                    cached_child is not None
                    and str(cached_child.get("status") or "").strip().lower() != "deleted"
                    and not cached_terminal_error
                ):
                    child = cached_child
                else:
                    child = _next_recovery_child(task_bin, plan.parent, child_short)
            except _RecoveryLookupUnavailable as exc:
                outcomes.append((_recovery_partial(plan.parent, str(exc)), ""))
                break
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
                    outcomes.append((_recovery_terminal(plan.parent, terminal_error), ""))
                break

        terminal_error = _terminal_recovery_error(child, hook, recovery_at)
        if terminal_error:
            outcomes.append((_recovery_terminal(plan.parent, terminal_error), ""))
            break
        if not reconcile.is_orphan_deleted_chain_candidate(child):
            break
        current = child

    return outcomes


class _ReconcileLifecycleServices:
    """Taskwarrior adapter for the shared lifecycle transition executor."""

    def __init__(self, task_bin: str, hook: Any, parent: dict[str, Any] | None = None):
        self.task_bin = task_bin
        self.hook = hook
        self.parent = dict(parent or {})
        self.imported = False
        self.last_child_short = ""

    @staticmethod
    def _result(state: OperationState, *, value: Any = None, reason: str = "") -> OperationResult:
        return OperationResult(state, value=value, reason=reason)

    @staticmethod
    def _link(value: Any) -> int | None:
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError, OverflowError):
            return None

    def validate_parent(self, plan: LifecyclePlan) -> OperationResult:
        try:
            parent = _fresh_parent(self.task_bin, {"uuid": plan.identity.parent_uuid})
        except Exception as exc:
            return self._result(OperationState.UNAVAILABLE, reason=f"parent export unavailable: {exc}")
        if parent is None:
            return self._result(OperationState.CONFLICT, reason="parent task is unavailable")
        guard = plan.parent_guard.to_dict()
        for field in ("status", "chain", "chainID"):
            expected = str(guard.get(field) or "").strip().casefold()
            actual = str(parent.get(field) or "").strip().casefold()
            if expected != actual:
                return self._result(
                    OperationState.CONFLICT,
                    reason=f"parent {field} changed (expected {expected or '-'}, found {actual or '-'})",
                )
        expected_link = self._link(guard.get("link"))
        actual_link = self._link(parent.get("link"))
        if expected_link is None or actual_link is None or expected_link != actual_link:
            return self._result(
                OperationState.CONFLICT,
                reason=f"parent link changed (expected {guard.get('link')}, found {parent.get('link')})",
            )
        expected_fingerprint = str(guard.get("recurrence_fingerprint") or "").strip()
        if expected_fingerprint:
            try:
                from nautical_core.lifecycle_models import recurrence_fingerprint

                actual_fingerprint = recurrence_fingerprint(parent)
            except Exception as exc:
                return self._result(OperationState.UNAVAILABLE, reason=f"parent fingerprint unavailable: {exc}")
            if actual_fingerprint != expected_fingerprint:
                return self._result(OperationState.CONFLICT, reason="parent recurrence inputs changed")
        return self._result(OperationState.APPLIED)

    def validate_terminal(self, plan: LifecyclePlan) -> OperationResult:
        try:
            parent = _fresh_parent(self.task_bin, {"uuid": plan.identity.parent_uuid})
        except Exception as exc:
            return self._result(OperationState.UNAVAILABLE, reason=f"terminal parent export unavailable: {exc}")
        if parent is None:
            return self._result(OperationState.UNAVAILABLE, reason="terminal parent export unavailable")
        if str(parent.get("nextLink") or "").strip():
            return self._result(
                OperationState.CONFLICT,
                reason="successor is already linked; retain it and review the terminal transition",
            )
        if plan.identity.event is not LifecycleEvent.MANUAL_DELETE:
            try:
                successors = _existing_children(self.task_bin, parent)
            except Exception as exc:
                return self._result(OperationState.UNAVAILABLE, reason=f"successor lookup unavailable: {exc}")
            if successors:
                return self._result(
                    OperationState.CONFLICT,
                    reason="successor is already persisted; retain it and review the terminal transition",
                )
        if str(parent.get("chain") or "").strip().lower() == "off":
            return self._result(OperationState.ALREADY)
        return self.validate_parent(plan)

    def disable_chain(self, plan: LifecyclePlan) -> OperationResult:
        try:
            parent = _fresh_parent(self.task_bin, {"uuid": plan.identity.parent_uuid})
            if parent is None:
                return self._result(OperationState.UNAVAILABLE, reason="terminal parent export unavailable")
            if str(parent.get("chain") or "").strip().lower() == "off":
                return self._result(OperationState.ALREADY)
            _disable_parent_chain(self.task_bin, parent)
            return self._result(OperationState.APPLIED)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            state = (
                OperationState.UNAVAILABLE
                if isinstance(exc, (TimeoutError, ConnectionError)) or "lock" in reason.lower()
                else OperationState.FAILED
            )
            return self._result(state, reason=f"chain disablement failed: {reason}")

    def verify_terminal(self, plan: LifecyclePlan) -> OperationResult:
        try:
            _verify_disabled_parent(self.task_bin, {"uuid": plan.identity.parent_uuid})
            return self._result(OperationState.APPLIED)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            state = (
                OperationState.UNAVAILABLE
                if isinstance(exc, (TimeoutError, ConnectionError)) or "lock" in reason.lower()
                else OperationState.CONFLICT
            )
            return self._result(state, reason=f"terminal chain verification failed: {reason}")

    def _rows_for_child(self, plan: LifecyclePlan) -> list[dict[str, Any]]:
        parent = {"chainID": plan.identity.chain_id, "link": plan.identity.source_link}
        return _existing_children(self.task_bin, parent)

    def find_equivalent_child(self, plan: LifecyclePlan) -> OperationResult:
        child = plan.child_dict()
        child_uuid = str(child.get("uuid") or "").strip().lower()
        try:
            rows = self._rows_for_child(plan)
        except Exception as exc:
            return self._result(OperationState.UNAVAILABLE, reason=f"child lookup unavailable: {exc}")
        for row in rows:
            row_uuid = str(row.get("uuid") or "").strip().lower()
            if child_uuid and row_uuid == child_uuid:
                return self._result(OperationState.FOUND, value=row)
            if (
                str(row.get("chainID") or "").strip() == plan.identity.chain_id
                and self._link(row.get("link")) == plan.identity.target_link
                and str(row.get("prevLink") or "").strip().lower()
                == str(plan.identity.parent_uuid or "").strip().lower()[:8]
            ):
                return self._result(OperationState.FOUND, value=row)
        return self._result(OperationState.ABSENT)

    def import_child(self, plan: LifecyclePlan) -> OperationResult:
        child = plan.child_dict()
        stable_uuid = _stable_child_uuid(self.hook, self.parent, child)
        if stable_uuid:
            child["uuid"] = stable_uuid
        try:
            child_short, _stripped = _spawn_child(self.hook, child, self.parent)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            state = (
                OperationState.UNAVAILABLE
                if isinstance(exc, (TimeoutError, ConnectionError)) or "lock" in reason.lower()
                else OperationState.FAILED
            )
            return self._result(state, reason=f"child import failed: {reason}")
        self.imported = True
        child["uuid"] = str(child.get("uuid") or child_short).strip()
        child["_reconcile_child_short"] = str(child_short or "").strip()
        return self._result(OperationState.APPLIED, value=child)

    def verify_child(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult:
        wanted_uuid = str(child.get("uuid") or "").strip()
        try:
            rows = self._rows_for_child(plan)
        except Exception as exc:
            return self._result(OperationState.UNAVAILABLE, reason=f"child verification unavailable: {exc}")
        wanted_uuid = wanted_uuid.lower()
        wanted_short = str(child.get("_reconcile_child_short") or "").strip().lower()
        for row in rows:
            actual = str(row.get("uuid") or "").strip().lower()
            if (wanted_uuid and actual == wanted_uuid) or (wanted_short and actual.startswith(wanted_short)):
                child.clear()
                child.update(row)
                return self._result(OperationState.APPLIED)
        if wanted_uuid and wanted_uuid.count("-") == 0:
            try:
                direct_rows = _export(self.task_bin, [f"uuid:{wanted_uuid}"], timeout=30.0)
            except Exception as exc:
                return self._result(OperationState.UNAVAILABLE, reason=f"child verification unavailable: {exc}")
            for row in direct_rows:
                actual = str(row.get("uuid") or "").strip().lower()
                if actual == wanted_uuid or actual.startswith(wanted_uuid):
                    child.clear()
                    child.update(row)
                    return self._result(OperationState.APPLIED)
        return self._result(OperationState.CONFLICT, reason="child postcondition is missing")

    def _child_short(self, plan: LifecyclePlan, child: dict[str, Any]) -> str:
        patch = plan.parent_patch_dict()
        value = str(
            patch.get("nextLink")
            or child.get("_reconcile_child_short")
            or str(child.get("uuid") or "")[:8]
        ).strip()
        self.last_child_short = value
        return value

    def apply_parent_patch(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult:
        child_short = self._child_short(plan, child)
        if not child_short:
            return self._result(OperationState.CONFLICT, reason="parent patch has no child identity")
        try:
            parent = _fresh_parent(self.task_bin, {"uuid": plan.identity.parent_uuid})
            if parent is None:
                return self._result(OperationState.UNAVAILABLE, reason="parent export unavailable")
            current = str(parent.get("nextLink") or "").strip().lower()
            if current == child_short.casefold():
                return self._result(OperationState.ALREADY)
            if current:
                return self._result(OperationState.CONFLICT, reason="parent nextLink already set")
            _modify_parent_nextlink(self.task_bin, parent, child_short)
            return self._result(OperationState.APPLIED)
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            state = (
                OperationState.UNAVAILABLE
                if isinstance(exc, (TimeoutError, ConnectionError)) or "lock" in reason.lower()
                else OperationState.FAILED
            )
            return self._result(state, reason=f"parent patch failed: {reason}")

    def verify_linkage(self, plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult:
        child_short = self._child_short(plan, child)
        try:
            parent = {"uuid": plan.identity.parent_uuid, "chainID": plan.identity.chain_id}
            verified = _verify_applied_child(
                self.task_bin,
                parent,
                child_short,
                hook=self.hook,
                strict_uuid=bool(str(child.get("uuid") or "").count("-")),
            )
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            state = OperationState.UNAVAILABLE if "lock" in reason.lower() else OperationState.CONFLICT
            return self._result(state, reason=f"parent linkage verification failed: {reason}")
        child.clear()
        child.update(verified)
        return self._result(OperationState.APPLIED)

    def compensate_child(self, _plan: LifecyclePlan, child: dict[str, Any]) -> OperationResult:
        child_uuid = str(child.get("uuid") or "").strip()
        if not child_uuid:
            return self._result(OperationState.FAILED, reason="imported child has no UUID for compensation")
        try:
            result = _run_task(
                self.task_bin,
                ["rc.hooks=off", "rc.confirmation=off", f"uuid:{child_uuid}", "delete"],
                timeout=30.0,
            )
            if result.returncode == 0:
                return self._result(OperationState.APPLIED)
            return self._result(OperationState.FAILED, reason=task_command.failure_message(result, "child compensation"))
        except Exception as exc:
            return self._result(OperationState.FAILED, reason=f"child compensation failed: {exc}")


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
        until_dt, until_err = _safe_parse_datetime(hook, child_until)
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
        print(_style(f"spawn: {parent}{suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("reason", "kind", "next_link", "child_field", "child_target", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "backfill_nextlink":
        suffix = " (applied)" if applied_short else ""
        print(_style(f"backfill nextLink: {parent}{suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("reason", "next_link", "existing_child"))
    elif plan.action == "legitimate_final":
        suffix = " -> set chain:off" if applied_short else ""
        label = "terminal" if reconcile.is_terminal_plan(plan) else "final"
        print(_style(f"{label}: {parent} ({plan.reason}){suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "manual_stop":
        suffix = " -> set chain:off" if applied_short else ""
        print(_style(f"manual stop: {parent} ({plan.reason}){suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("kind", "next_link"))
    elif plan.action == "stale":
        print(_style(f"skip: {parent} ({plan.reason})", _action_style(plan.action)))
    elif plan.action == "partial":
        print(_style(f"partial: {parent} ({plan.reason})", _action_style(plan.action)))
    else:
        print(_style(f"error: {parent} ({plan.reason})", _action_style("error")))
        _print_evidence(evidence, ("kind", "next_link", "child_due", "child_local", "child_expires", "expiration"))


def _print_recovery_group(
    items: list[tuple[reconcile.ReconcilePlan, dict[str, Any], str]],
) -> None:
    first = items[0][0]
    last, evidence, applied_short = items[-1]
    hops = sum(1 for plan, _evidence, _applied in items if plan.action in {"spawn", "backfill_nextlink"})
    noun = "occurrence" if hops == 1 else "occurrences"
    print(_style(f"recover: {_fmt_parent(first.parent)} -> advanced {hops} {noun}", "cyan"))
    if last.action in {"error", "partial", "legitimate_final", "manual_stop", "stale"}:
        result = "terminal" if reconcile.is_terminal_plan(last) else last.action.replace("_", " ")
        print(_style(f"  result: {result} ({last.reason})", _action_style(last.action)))
        return
    if applied_short:
        print(f"  child: {applied_short}")
    _print_evidence(evidence, ("next_link", "child_local", "child_due", "child_expires"))


def _startup_failure(args: Any, stage: str, exc: Exception) -> int:
    reason = str(exc).strip() or type(exc).__name__
    if args.json:
        configuration_status = "unavailable" if stage == "taskdata_config" else "valid"
        payload: dict[str, Any] = {
            "schema": _JSON_SCHEMA,
            "schema_version": _JSON_SCHEMA_VERSION,
            "mode": "apply" if args.apply else "dry-run",
            "status": "error",
            "stage": stage,
            "error": reason,
            "configuration_status": configuration_status,
            "configuration_drifted": 0,
            "configuration_drift": reason if configuration_status == "unavailable" else "",
            "candidates": 0,
            "expiration_hops": 0,
            "recovered_chains": 0,
            "spawn": 0,
            "backfill_nextlink": 0,
            "legitimate_final": 0,
            "terminal": 0,
            "manual_stop": 0,
            "stale": 0,
            "partial": 0,
            "native_until_manual_review": 0,
            "native_until_audit_skipped": 0,
            "errors": 1,
            "startup_errors": 1,
            "plan_errors": 0,
            "native_until_error_count": 0,
            "plans": [],
            "applied": [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_style(f"error: {stage.replace('_', ' ')}: {reason}", "red", stream=sys.stderr), file=sys.stderr)
    return 1


def main(
    argv: list[str] | None = None,
    *,
    _apply_lease_held: bool = False,
    _locked_taskdata: Path | None = None,
) -> int:
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
    _EXPORT_STATS.update(calls=0, rows=0, seconds=0.0, slowest_seconds=0.0, snapshot_hits=0)
    _LOCK_STATS.update(reconcile_busy=0, parent_busy=0)
    if args.apply and not _apply_lease_held:
        try:
            taskdata = _task_data_dir(args.task_bin)
        except Exception as exc:
            return _startup_failure(args, "taskdata", exc)
        with _reconcile_apply_lock(taskdata) as acquired:
            if not acquired:
                _LOCK_STATS["reconcile_busy"] += 1
                return _startup_failure(args, "apply_lock", RuntimeError("another reconcile apply is already running"))
            return main(argv, _apply_lease_held=True, _locked_taskdata=taskdata)

    try:
        hook, legacy_hook = _load_reconcile_runtime(args.task_bin, args.hook_path)
    except Exception as exc:
        return _startup_failure(args, "hook_load", exc)
    try:
        if legacy_hook:
            if isinstance(hook, ModuleType):
                # A real hook module is only a source of validated core
                # configuration here. Mutation and child generation belong to
                # the public operator gateway, not modify_impl internals.
                _validate_hook_protocol(hook)
                hook = TaskwarriorMutationGateway(hook.core, task_bin=args.task_bin)
                legacy_hook = False
            else:
                _bind_hook_task_bin(hook, args.task_bin)
        fmt_dt_local = getattr(getattr(hook, "core", None), "fmt_dt_local", None)
        now_utc = getattr(getattr(hook, "core", None), "now_utc", None)
        recovery_at = now_utc() if callable(now_utc) else datetime.now(timezone.utc)
    except Exception as exc:
        return _startup_failure(args, "hook_protocol" if legacy_hook else "runtime", exc)
    global _READ_SNAPSHOT
    snapshot = _ReconcileSnapshot(args.task_bin)
    _READ_SNAPSHOT = snapshot
    try:
        candidates = _candidate_rows(args.task_bin, hook)
    except Exception as exc:
        return _startup_failure(args, "candidate_export", exc)
    try:
        taskdata = _locked_taskdata if args.apply else None
        if args.apply and taskdata is None:
            taskdata = _task_data_dir(args.task_bin)
    except Exception as exc:
        return _startup_failure(args, "taskdata", exc)
    runtime_taskdata = taskdata
    if runtime_taskdata is None:
        try:
            if not str(os.environ.get("NAUTICAL_CONFIG") or "").strip():
                env_taskdata = str(os.environ.get("TASKDATA") or "").strip()
                if env_taskdata:
                    runtime_taskdata = Path(env_taskdata).expanduser().resolve()
                elif candidates and callable(getattr(getattr(hook, "core", None), "reload_taskdata_config", None)):
                    runtime_taskdata = _task_data_dir(args.task_bin)
        except Exception as exc:
            return _startup_failure(args, "taskdata_config", exc)
    try:
        _synchronize_taskdata_config(hook, runtime_taskdata)
    except Exception as exc:
        return _startup_failure(args, "taskdata_config", exc)
    try:
        generation = _chain_generation_for_hook(hook)
    except Exception as exc:
        return _startup_failure(args, "chain_generation", exc)
    configuration_status, configuration_drift_reason = _configuration_state(hook)
    native_until_audit_warning = ""
    if configuration_status != "valid":
        native_until_repairs: list[dict[str, Any]] = []
        native_until_errors: list[str] = []
    else:
        try:
            native_until_repairs, native_until_errors = _native_until_repairs(
                args.task_bin,
                hook,
                apply=args.apply,
                taskdata=taskdata,
                lease_held=_apply_lease_held,
            )
        except Exception as exc:
            # The integrity pass is supplementary; preserve normal recovery when its
            # independent export cannot run (for example while Taskwarrior is locked).
            native_until_repairs, native_until_errors = [], []
            native_until_audit_warning = str(exc).strip() or type(exc).__name__
            if not args.json:
                print(
                    _style(
                        f"warning: native-until audit skipped: {native_until_audit_warning}",
                        "yellow",
                        stream=sys.stderr,
                    ),
                    file=sys.stderr,
                )
    if configuration_status == "valid":
        blocked_item = next(
            (item for item in native_until_repairs if item.get("configuration_drift")),
            None,
        )
        if blocked_item is not None:
            configuration_drift_reason = str(blocked_item.get("repair_error") or "")
            configuration_status = str(blocked_item.get("configuration_status") or "drifted")
    if not args.json:
        for item in native_until_repairs:
            action = item.get("action") or "native_until"
            suffix = f" -> {_format_local_until(hook, item['new_until'])}" if item.get("new_until") else ""
            outcome = " (no change applied)" if action == "manual_review" else ""
            line = (
                f"native-until: {action:<13} {item.get('task') or '?'} "
                f"chain={item.get('chainID') or '?'} link={item.get('link') or '?'}"
                f"  {item.get('reason') or 'invalid native until'}{suffix}{outcome}"
            )
            print(_style(line, _action_style(action)))
        for error in native_until_errors:
            print(_style(f"error: native-until: {error}", "red", stream=sys.stderr), file=sys.stderr)
    plans: list[reconcile.ReconcilePlan] = []
    plan_evidence: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    outcome_groups: list[list[tuple[reconcile.ReconcilePlan, str]]] = []
    processed_slots: set[tuple[str, int]] = set()
    ambiguous_slots = _ambiguous_candidate_slots(candidates)

    for parent in candidates:
        if configuration_status == "valid":
            configuration_status, configuration_drift_reason = _configuration_state(hook)
        if configuration_status != "valid":
            break
        parent_slot = (
            str(parent.get("chainID") or "").strip(),
            reconcile.int_or_default(parent.get("link"), 0),
        )
        if parent_slot in processed_slots:
            continue
        if parent_slot in ambiguous_slots:
            outcomes = [(_recovery_error(parent, ambiguous_slots[parent_slot]), "")]
        else:
            try:
                outcomes = _reconcile_candidate(
                    args.task_bin,
                    hook,
                    parent,
                    taskdata=taskdata,
                    apply=args.apply,
                    max_expiration_hops=args.max_expiration_hops,
                    recovery_at=recovery_at,
                    lease_held=_apply_lease_held,
                    generation=generation,
                )
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                outcomes = [(_recovery_error(parent, reason), "")]
        outcome_groups.append(outcomes)
        if configuration_status == "valid":
            configuration_drift_reason = next(
                (
                    plan.reason
                    for plan, _applied in outcomes
                    if plan.action == "partial" and str(plan.reason).startswith("configuration ")
                ),
                "",
            )
            if configuration_drift_reason:
                configuration_status = (
                    "unavailable"
                    if (
                        "configuration verification unavailable" in configuration_drift_reason
                        or "configuration verifier is unavailable" in configuration_drift_reason
                    )
                    else "drifted"
                )
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
    native_until_manual_review = sum(
        1 for item in native_until_repairs if item.get("action") == "manual_review"
    )
    native_until_audit_skipped = int(bool(native_until_audit_warning))
    degraded = (
        any(plan.action == "partial" for plan in plans)
        or any(plan.action == "manual_review" for plan in plans)
        or native_until_manual_review > 0
        or native_until_audit_skipped > 0
        or bool(configuration_drift_reason)
    )
    plan_errors = sum(1 for plan in plans if plan.action == "error")
    native_until_error_count = len(native_until_errors)
    total_errors = plan_errors + native_until_error_count
    has_errors = total_errors > 0

    summary = {
        "schema": _JSON_SCHEMA,
        "schema_version": _JSON_SCHEMA_VERSION,
        "status": "error" if has_errors else "degraded" if degraded else "ok",
        "configuration_status": configuration_status,
        "configuration_drifted": int(configuration_status == "drifted"),
        "configuration_drift": configuration_drift_reason,
        "mode": "apply" if args.apply else "dry-run",
        "candidates": len(candidates),
        "expiration_hops": expiration_hops,
        "recovered_chains": recovered_chains,
        "spawn": sum(1 for p in plans if p.action == "spawn"),
        "backfill_nextlink": sum(1 for p in plans if p.action == "backfill_nextlink"),
        "legitimate_final": sum(1 for p in plans if p.action == "legitimate_final"),
        "terminal": sum(1 for p in plans if reconcile.is_terminal_plan(p)),
        "manual_stop": sum(1 for p in plans if p.action == "manual_stop"),
        "stale": sum(1 for p in plans if p.action == "stale"),
        "partial": sum(1 for p in plans if p.action == "partial"),
        "manual_review": sum(1 for p in plans if p.action == "manual_review"),
        "errors": total_errors,
        "startup_errors": 0,
        "plan_errors": plan_errors,
        "native_until_error_count": native_until_error_count,
        "native_until_manual_review": native_until_manual_review,
        "native_until_audit_skipped": native_until_audit_skipped,
        "export_calls": _EXPORT_STATS["calls"],
        "export_rows": _EXPORT_STATS["rows"],
        "export_seconds": round(_EXPORT_STATS["seconds"], 4),
        "slowest_export_seconds": round(_EXPORT_STATS["slowest_seconds"], 4),
        "snapshot_hits": _EXPORT_STATS["snapshot_hits"],
        "lock_contention": dict(_LOCK_STATS),
        "plans": [
            {
                "action": plan.action,
                **evidence,
            }
            for plan, evidence in zip(plans, plan_evidence)
        ],
        "applied": applied,
        "native_until_repairs": native_until_repairs,
        "native_until_errors": native_until_errors,
        "native_until_audit_warning": native_until_audit_warning,
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        summary_line = (
            "summary: "
            f"{summary['mode']}; candidates={summary['candidates']} "
            f"spawn={summary['spawn']} backfill={summary['backfill_nextlink']} "
            f"expiration_hops={summary['expiration_hops']} recovered={summary['recovered_chains']} "
            f"final={summary['legitimate_final']} manual={summary['manual_stop']} "
            f"terminal={summary['terminal']} "
            f"stale={summary['stale']} partial={summary['partial']} errors={summary['errors']}"
            f" plan_errors={summary['plan_errors']}"
            f" native_until_errors={summary['native_until_error_count']}"
            f" native_until={len(summary['native_until_repairs'])}"
            f" manual_review={summary['native_until_manual_review']}"
            f" audit_skipped={summary['native_until_audit_skipped']}"
            f" config={summary['configuration_status']}"
        )
        summary_color = "red" if has_errors else "yellow" if degraded else "green"
        print(_style(summary_line, summary_color))
        diagnostics_line = (
            "diagnostics: "
            f"exports={summary['export_calls']} rows={summary['export_rows']} "
            f"export_s={summary['export_seconds']:.4f} "
            f"slowest_export_s={summary['slowest_export_seconds']:.4f} "
            f"snapshot_hits={summary['snapshot_hits']} "
            f"lock_busy={sum(summary['lock_contention'].values())}"
        )
        print(_style(diagnostics_line, "dim"))
    if has_errors:
        return 1
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
