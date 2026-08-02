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
try:
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:  # Python 3.10 and earlier
    try:
        tomllib = importlib.import_module("tomli")
    except ModuleNotFoundError:
        tomllib = None
import zoneinfo
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
_RECONCILE_LOCK_STALE_SECONDS = 300.0
_DEFAULT_EXPIRATION_HOPS = 32
_MAX_EXPIRATION_HOPS = 1000
_RECONCILE_PROTOCOL = 1
_JSON_SCHEMA = "nautical.reconcile"
_JSON_SCHEMA_VERSION = 1


class _ConfigurationDrift(RuntimeError):
    """Signal that this run must stop before applying under a new configuration."""

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


def _configuration_drift_reason(hook: Any) -> str:
    try:
        drift = hook.core.configuration_drift()
    except Exception:
        return ""
    if not isinstance(drift, dict) or not drift.get("changed"):
        return ""
    source = str(drift.get("source") or "unknown")
    return f"configuration changed during reconcile (source: {source}); restart and rerun"


def _candidate_sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(row.get("chainID") or "").strip().casefold(),
        reconcile.int_or_default(row.get("link"), 0),
        str(row.get("status") or "").strip().casefold(),
        str(row.get("uuid") or "").strip().casefold(),
    )


class _ReconcileSnapshot:
    """Immutable read-phase chain export shared by candidate and audit scans."""

    def __init__(self, task_bin: str):
        self.task_bin = task_bin
        self._chain_rows: list[dict[str, Any]] | None = None

    def chain_rows(self) -> list[dict[str, Any]]:
        if self._chain_rows is None:
            self._chain_rows = _export(self.task_bin, ["chain:on", "chainID.not:"])
        return self._chain_rows


_READ_SNAPSHOT: _ReconcileSnapshot | None = None


def _candidate_rows(
    task_bin: str,
    hook: Any,
    *,
    snapshot: _ReconcileSnapshot | None = None,
) -> list[dict[str, Any]]:
    snapshot = snapshot or _READ_SNAPSHOT
    if snapshot is not None:
        rows = snapshot.chain_rows()
        if not rows:
            # Preserve compatibility with Taskwarrior wrappers that only honor
            # the older status-scoped export filters.
            completed = _export(task_bin, ["status:completed", "chain:on", "chainID.not:", "nextLink:"])
            deleted = _export(task_bin, ["status:deleted", "chain:on", "chainID.not:", "nextLink:"])
            candidates = [row for row in completed if reconcile.is_orphan_completion_candidate(row)]
            candidates.extend(row for row in deleted if reconcile.is_orphan_deleted_chain_candidate(row))
            return sorted(candidates, key=_candidate_sort_key)
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
    rows = snapshot.chain_rows() if snapshot is not None else _export(task_bin, ["chain:on", "chainID.not:"])
    return sorted(
        (
            row
            for row in rows
            if include_inactive
            or str(row.get("status") or "").strip().lower() not in {"completed", "deleted"}
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
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find invalid native windows and repair only those with a reliable predecessor."""
    all_rows = _active_chain_rows(
        task_bin,
        include_inactive=True,
        snapshot=snapshot or _READ_SNAPSHOT,
    )
    rows = [
        row
        for row in all_rows
        if str(row.get("status") or "").strip().lower() not in {"completed", "deleted"}
    ]
    by_chain_link = {
        (
            str(row.get("chainID") or "").strip(),
            reconcile.int_or_default(row.get("link"), 0),
        ): row
        for row in all_rows
    }
    repairs: list[dict[str, Any]] = []
    errors: list[str] = []
    for row in rows:
        reason = reconcile.invalid_native_until_reason(row, safe_parse_datetime=hook._safe_parse_datetime)
        if not reason:
            continue
        chain_id = str(row.get("chainID") or "").strip()
        link = reconcile.int_or_default(row.get("link"), 0)
        previous = by_chain_link.get((chain_id, link - 1))
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
                safe_parse_datetime=hook._safe_parse_datetime,
            )
            if previous_reason:
                repair_error = f"previous link is invalid: {previous_reason}"
            else:
                kind = reconcile.recurrence_kind(row)
                repaired, repair_error = reconcile.repair_native_until_from_previous(
                    previous,
                    row,
                    kind=kind,
                    safe_parse_datetime=hook._safe_parse_datetime,
                    fmt_isoz=hook.core.fmt_isoz,
                    utc_to_local_naive=getattr(hook, "_utc_to_local_naive"),
                    local_naive_to_utc=getattr(hook, "_local_naive_to_utc"),
                )
        if repair_error or not repaired:
            fallback, fallback_error = reconcile.fallback_native_until_at_day_end(
                row,
                safe_parse_datetime=hook._safe_parse_datetime,
                fmt_isoz=hook.core.fmt_isoz,
                utc_to_local_naive=getattr(hook, "_utc_to_local_naive"),
                local_naive_to_utc=getattr(hook, "_local_naive_to_utc"),
            )
            if fallback_error or not fallback:
                item["action"] = "manual_review"
                item["repair_error"] = fallback_error or repair_error or "could not calculate repaired until"
                errors.append(f"{item['task']} chain {chain_id} link {link}: {item['repair_error']}")
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
            with _reconcile_apply_lock(taskdata) as reconcile_acquired:
                if not reconcile_acquired:
                    item["action"] = "repair_error"
                    item["repair_error"] = "another reconcile apply is already running"
                else:
                    parent_lock = _parent_apply_lock(taskdata, str(row.get("uuid") or ""))
                    with parent_lock as acquired:
                        if not acquired:
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
                                drift_reason = _configuration_drift_reason(hook)
                                if drift_reason:
                                    item["action"] = "manual_review"
                                    item["repair_error"] = drift_reason
                                    item["configuration_drift"] = True
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
        actual_dt, actual_err = hook._safe_parse_datetime(actual)
        expected_dt, expected_err = hook._safe_parse_datetime(expected)
        return not actual_err and not expected_err and actual_dt is not None and actual_dt == expected_dt
    except Exception:
        return False


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


def _synchronize_taskdata_config(hook: Any, taskdata: Path | None) -> None:
    """Load installer-managed config when core imported too early.

    Reconcile imports the hook/core before it can ask Taskwarrior for its data
    directory. For custom TASKDATA installations that leaves the static core
    timezone and astronomy snapshots at their defaults even though the
    installed config is valid. Do not override an explicit NAUTICAL_CONFIG; it
    has already been selected (or deliberately rejected) by the normal loader.
    """
    if taskdata is None or tomllib is None or str(os.environ.get("NAUTICAL_CONFIG") or "").strip():
        return
    core = getattr(hook, "core", None)
    if core is None:
        return
    candidates = (
        taskdata / "config-nautical.toml",
        taskdata / "nautical.toml",
        taskdata / "nautical_core" / "config-nautical.toml",
        taskdata / "nautical_core" / "nautical.toml",
    )
    for path in candidates:
        try:
            if not path.is_file():
                continue
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        astronomy = data.get("astronomy") if isinstance(data, dict) else None
        if isinstance(data, dict) and "astronomy" in data and isinstance(astronomy, dict):
            core.ASTRONOMY_CONFIG = astronomy
        timezone_name = str(data.get("tz") or "").strip() if isinstance(data, dict) else ""
        if timezone_name:
            try:
                local_tz = zoneinfo.ZoneInfo(timezone_name)
            except Exception:
                local_tz = None
            if local_tz is not None:
                core.LOCAL_TZ_NAME = timezone_name
                core._LOCAL_TZ = local_tz
        core_config = getattr(core, "_core_config", None)
        if core_config is not None:
            if isinstance(data, dict) and "astronomy" in data and isinstance(astronomy, dict):
                core_config.ASTRONOMY_CONFIG = astronomy
            if timezone_name:
                core_config.LOCAL_TZ_NAME = timezone_name
        return


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


def _verify_applied_child(
    task_bin: str,
    parent: dict[str, Any],
    child_short: str,
    *,
    hook: Any = None,
    strict_uuid: bool = False,
) -> None:
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
    if callable(getattr(hook, "_stable_child_uuid", None)):
        matched = next(
            (
                row
                for row in rows
                if str(row.get("uuid") or "").strip().lower().startswith(expected_child)
            ),
            None,
        )
        if matched is not None:
            expected_uuid = str(hook._stable_child_uuid(fresh_parent, matched) or "").strip().lower()
            actual_uuid = str(matched.get("uuid") or "").strip().lower()
            if strict_uuid and expected_uuid and actual_uuid != expected_uuid:
                raise RuntimeError(
                    f"post-apply child UUID {actual_uuid[:8] or '<empty>'} "
                    f"does not match deterministic slot identity {expected_uuid[:8]}"
                )


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
    with _reconcile_apply_lock(taskdata) as reconcile_acquired:
        if not reconcile_acquired:
            raise RuntimeError("another reconcile apply is already running")
        with _parent_apply_lock(taskdata, parent_uuid) as acquired:
            if not acquired:
                raise RuntimeError(f"parent reconcile lock busy: {reconcile.short_uuid(parent_uuid)}")
            drift_reason = _configuration_drift_reason(hook)
            if drift_reason:
                raise _ConfigurationDrift(drift_reason)
            plan = _refresh_plan(task_bin, hook, original_parent)
            if plan.action == "spawn":
                if not plan.child:
                    raise RuntimeError("spawn plan has no child payload")
                child_short, _stripped = hook._spawn_child(plan.child, plan.parent)
                _modify_parent_nextlink(task_bin, plan.parent, child_short)
                _verify_applied_child(
                    task_bin,
                    plan.parent,
                    child_short,
                    hook=hook,
                    strict_uuid=True,
                )
                return plan, child_short
            if plan.action == "backfill_nextlink":
                _modify_parent_nextlink(task_bin, plan.parent, plan.child_short)
                _verify_applied_child(task_bin, plan.parent, plan.child_short, hook=hook)
                return plan, plan.child_short
            if plan.action in {"legitimate_final", "manual_stop"}:
                _disable_parent_chain(task_bin, plan.parent)
                _verify_disabled_parent(task_bin, plan.parent)
                return plan, "off"
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
            except _ConfigurationDrift as exc:
                outcomes.append((_recovery_partial(current, str(exc)), ""))
                break
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
        print(_style(f"spawn: {parent}{suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("reason", "kind", "next_link", "child_field", "child_target", "child_due", "child_local", "child_expires", "expiration"))
    elif plan.action == "backfill_nextlink":
        suffix = " (applied)" if applied_short else ""
        print(_style(f"backfill nextLink: {parent}{suffix}", _action_style(plan.action)))
        _print_evidence(evidence, ("reason", "next_link", "existing_child"))
    elif plan.action == "legitimate_final":
        suffix = " -> set chain:off" if applied_short else ""
        print(_style(f"final: {parent} ({plan.reason}){suffix}", _action_style(plan.action)))
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
        print(_style(f"  result: {last.action.replace('_', ' ')} ({last.reason})", _action_style(last.action)))
        return
    if applied_short:
        print(f"  child: {applied_short}")
    _print_evidence(evidence, ("next_link", "child_local", "child_due", "child_expires"))


def _startup_failure(args: Any, stage: str, exc: Exception) -> int:
    reason = str(exc).strip() or type(exc).__name__
    if args.json:
        payload: dict[str, Any] = {
            "schema": _JSON_SCHEMA,
            "schema_version": _JSON_SCHEMA_VERSION,
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
            "native_until_manual_review": 0,
            "native_until_audit_skipped": 0,
            "errors": 1,
            "plans": [],
            "applied": [],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(_style(f"error: {stage.replace('_', ' ')}: {reason}", "red", stream=sys.stderr), file=sys.stderr)
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
    global _READ_SNAPSHOT
    snapshot = _ReconcileSnapshot(args.task_bin)
    _READ_SNAPSHOT = snapshot
    try:
        candidates = _candidate_rows(args.task_bin, hook)
    except Exception as exc:
        return _startup_failure(args, "candidate_export", exc)
    try:
        taskdata = _task_data_dir(args.task_bin) if args.apply else None
    except Exception as exc:
        return _startup_failure(args, "taskdata", exc)
    runtime_taskdata = taskdata
    if runtime_taskdata is None:
        try:
            if not str(os.environ.get("NAUTICAL_CONFIG") or "").strip():
                env_taskdata = str(os.environ.get("TASKDATA") or "").strip()
                if env_taskdata:
                    runtime_taskdata = Path(env_taskdata).expanduser().resolve()
                elif not getattr(hook.core, "ASTRONOMY_CONFIG", {}):
                    runtime_taskdata = _task_data_dir(args.task_bin)
        except Exception:
            runtime_taskdata = None
    _synchronize_taskdata_config(hook, runtime_taskdata)
    configuration_drift_reason = _configuration_drift_reason(hook)
    native_until_audit_warning = ""
    if configuration_drift_reason:
        native_until_repairs: list[dict[str, Any]] = []
        native_until_errors: list[str] = []
    else:
        try:
            native_until_repairs, native_until_errors = _native_until_repairs(
                args.task_bin,
                hook,
                apply=args.apply,
                taskdata=taskdata,
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
    if not configuration_drift_reason:
        configuration_drift_reason = next(
            (
                str(item.get("repair_error") or "")
                for item in native_until_repairs
                if item.get("configuration_drift")
            ),
            "",
        )
    if not args.json:
        for item in native_until_repairs:
            action = item.get("action") or "native_until"
            suffix = f" -> {item['new_until']}" if item.get("new_until") else ""
            line = (
                f"native-until: {action:<13} {item.get('task') or '?'} "
                f"chain={item.get('chainID') or '?'} link={item.get('link') or '?'}"
                f"  {item.get('reason') or 'invalid native until'}{suffix}"
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
        if not configuration_drift_reason:
            configuration_drift_reason = _configuration_drift_reason(hook)
        if configuration_drift_reason:
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
                )
            except Exception as exc:
                reason = str(exc).strip() or type(exc).__name__
                outcomes = [(_recovery_error(parent, reason), "")]
        outcome_groups.append(outcomes)
        if not configuration_drift_reason:
            configuration_drift_reason = next(
                (
                    plan.reason
                    for plan, _applied in outcomes
                    if plan.action == "partial" and "configuration changed during reconcile" in plan.reason
                ),
                "",
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
        or native_until_manual_review > 0
        or native_until_audit_skipped > 0
        or bool(configuration_drift_reason)
    )
    has_errors = any(plan.action == "error" for plan in plans) or bool(native_until_errors)

    summary = {
        "schema": _JSON_SCHEMA,
        "schema_version": _JSON_SCHEMA_VERSION,
        "status": "error" if has_errors else "degraded" if degraded else "ok",
        "configuration_drifted": int(bool(configuration_drift_reason)),
        "configuration_drift": configuration_drift_reason,
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
        "native_until_manual_review": native_until_manual_review,
        "native_until_audit_skipped": native_until_audit_skipped,
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
            f"stale={summary['stale']} partial={summary['partial']} errors={summary['errors']}"
            f" native_until={len(summary['native_until_repairs'])}"
            f" manual_review={summary['native_until_manual_review']}"
            f" audit_skipped={summary['native_until_audit_skipped']}"
            f" config_drift={summary['configuration_drifted']}"
        )
        summary_color = "red" if has_errors else "yellow" if degraded else "green"
        print(_style(summary_line, summary_color))
    if has_errors:
        return 1
    return 2 if degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
