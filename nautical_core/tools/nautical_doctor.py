#!/usr/bin/env python3
"""Read-only installation and data diagnostics for Taskwarrior Nautical."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tomllib
import zoneinfo
from collections import defaultdict
from datetime import date, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

ZONEINFO_FACTORY: Callable[[str], Any] | None = getattr(zoneinfo, "ZoneInfo", None)
RICH_SPEC_FACTORY: Callable[[str], Any] = importlib.util.find_spec


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nautical_core as nautical_core_package  # noqa: E402
from nautical_core import astronomy, configuration_drift, config_schema, description_aliases, effective_config_snapshot, install_runtime  # noqa: E402
from nautical_core import chain_integrity_lifecycle as lifecycle  # noqa: E402
from nautical_core.integration_models import Absent, Found, Unavailable  # noqa: E402
from nautical_core.task_read_repository import ALL_TASK_STATUSES, TaskReadRepository  # noqa: E402
from nautical_core.chain_generation import ChainGenerationService  # noqa: E402
from nautical_core.integration_context import (  # noqa: E402
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork, build_operator_uow  # noqa: E402
from nautical_core.chain_snapshot import ChainSnapshotService, IntegritySnapshotRequest  # noqa: E402
from nautical_core.timeutil import compare_datetimes  # noqa: E402

_JSON_SCHEMA = "nautical.doctor"
_JSON_SCHEMA_VERSION = 1

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
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def _color_enabled(stream: Any = None) -> bool:
    """Use color only for an interactive terminal, unless explicitly forced."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    target = stream if stream is not None else sys.stdout
    try:
        return bool(target.isatty())
    except Exception:
        return False


def _paint(text: str, color: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


def _status_label(status: str, *, enabled: bool) -> str:
    labels = {"ok": "[OK]", "warn": "[WARN]", "error": "[FAIL]"}
    colors = {"ok": "green", "warn": "yellow", "error": "red"}
    normalized = str(status or "").strip().lower()
    label = labels.get(normalized, f"[{normalized.upper() or 'INFO'}]")
    return _paint(label, colors.get(normalized, "cyan"), enabled=enabled)


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


def _task_get(unit_of_work: TaskwarriorUnitOfWork, key: str) -> tuple[bool, str]:
    """Read a Taskwarrior setting through the invocation's shared client."""
    proc = unit_of_work.client.execute(
        ["_get", key],
        purpose="doctor Taskwarrior query",
        timeout=30.0,
        attempts=2,
        retry_delay=0.1,
    )
    return proc.ok, proc.stdout.strip()


def _task_export(unit_of_work: TaskwarriorUnitOfWork) -> tuple[bool, list[dict[str, Any]], str]:
    unit_of_work.repository.configure_commands(timeout=120.0, attempts=2, retry_delay=0.05)
    read = ChainSnapshotService(
        unit_of_work,
        configuration_fingerprint=unit_of_work.context.configuration.fingerprint,
    ).collect(IntegritySnapshotRequest.candidates(complete_chain_history=True))
    if isinstance(read, Found):
        return True, [node.to_dict() for node in read.value.rows], ""
    if isinstance(read, Absent):
        return True, [], ""
    if isinstance(read, Unavailable):
        return False, [], read.evidence.detail
    return False, [], "task repository returned an invalid result"


def _diagnostic_read_uow(
    taskdata: Path,
    task_bin: str,
    env: dict[str, str],
) -> TaskwarriorUnitOfWork:
    """Read Taskwarrior state without claiming valid scheduling configuration."""
    context = IntegrationContext(
        taskdata.resolve(),
        "doctor-recovery",
        (str(task_bin), f"rc.data.location={taskdata.resolve()}"),
        ValidatedNauticalConfiguration("doctor", "unavailable", "unavailable", "UTC", ()),
        timezone.utc,
        SilentDiagnostics(),
        SystemClock(),
        "doctor-diagnostic-read",
        256,
        IntegrationAccess.READ_ONLY,
    )
    return TaskwarriorUnitOfWork.create(context, env=env)


def _resolve_hooks_dir(unit_of_work: TaskwarriorUnitOfWork, taskdata: Path) -> Path:
    ok, raw = _task_get(unit_of_work, "rc.hooks.location")
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
    unit_of_work: TaskwarriorUnitOfWork,
    taskdata: Path,
) -> Path:
    proc = unit_of_work.client.execute(
        ["--version"],
        purpose="doctor Taskwarrior query",
        timeout=30.0,
        attempts=2,
        retry_delay=0.1,
    )
    if not proc.ok:
        _finding(
            findings,
            "taskwarrior.unavailable",
            "error",
            "Taskwarrior could not be executed.",
            fix="Install Taskwarrior or pass --task-bin.",
            details={
                "error": str(proc.stderr or proc.stdout or "").strip()
                or f"{proc.kind.value} (exit code {proc.returncode})"
            },
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
    return _resolve_hooks_dir(unit_of_work, taskdata)


def _check_hooks_and_udas(
    findings: list[dict[str, Any]],
    *,
    unit_of_work: TaskwarriorUnitOfWork,
    hooks_dir: Path,
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    validated = _check_hook_installation(findings, hooks_dir=hooks_dir, env=env)
    for name, expected in REQUIRED_UDAS.items():
        ok, actual = _task_get(unit_of_work, f"rc.uda.{name}.type")
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


def _check_managed_runtime(
    findings: list[dict[str, Any]],
    hooks_dir: Path,
    hook_runtimes: dict[str, dict[str, Any]] | None = None,
) -> None:
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
        manifest = status.get("manifest") if isinstance(status.get("manifest"), dict) else {}
        runtime_root = Path(str(status.get("runtime_root") or hooks_dir.parent)).expanduser()
        current_root = runtime_root / "current"
        provenance_errors: list[str] = []
        for event, record in (hook_runtimes or {}).items():
            implementation = record.get("implementation")
            if not implementation:
                provenance_errors.append(f"{event} implementation path is missing")
                continue
            try:
                Path(str(implementation)).resolve().relative_to(current_root.resolve())
            except Exception:
                provenance_errors.append(f"{event} implementation is outside the active release")
        provenance_details = {
            "release_id": release_id,
            "source": manifest.get("source", ""),
            "content_sha256": manifest.get("content_sha256", ""),
            "created_at": manifest.get("created_at", ""),
            "hook_impl_api": manifest.get("hook_impl_api", {}),
        }
        _finding(
            findings,
            "install.provenance",
            "error" if provenance_errors else "ok",
            "Active hooks resolve to one managed Nautical release."
            if not provenance_errors
            else "Active hooks do not share the managed Nautical release.",
            fix=("Reinstall Nautical so wrappers and nautical_core come from the same release." if provenance_errors else ""),
            details={**provenance_details, "errors": provenance_errors},
        )


def _config_candidates(taskdata: Path) -> list[Path]:
    explicit = os.environ.get("NAUTICAL_CONFIG", "").strip()
    candidates = [Path(explicit).expanduser()] if explicit else []
    taskrc = os.environ.get("TASKRC", "").strip()
    if taskrc:
        taskrc_dir = Path(taskrc).expanduser().resolve().parent
        candidates.extend(
            [
                taskrc_dir / "config-nautical.toml",
                taskrc_dir / "nautical.toml",
                taskrc_dir / ".task" / "config-nautical.toml",
                taskrc_dir / ".task" / "nautical.toml",
            ]
        )
    candidates.extend(
        [
            taskdata / "config-nautical.toml",
            taskdata / "nautical.toml",
            taskdata / "nautical_core" / "config-nautical.toml",
            taskdata / "nautical_core" / "nautical.toml",
            TOOLS_DIR.parent / "config-nautical.toml",
            TOOLS_DIR.parent / "nautical.toml",
            Path("~/.config/nautical/config-nautical.toml").expanduser(),
            Path("~/.config/nautical/nautical.toml").expanduser(),
            Path("~/.task/config-nautical.toml").expanduser(),
            Path("~/.task/nautical.toml").expanduser(),
        ]
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _check_config(findings: list[dict[str, Any]], taskdata: Path) -> None:
    config = next((path.resolve() for path in _config_candidates(taskdata) if path.is_file()), None)
    if config is None:
        _finding(
            findings,
            "config.missing",
            "warn",
            "No Nautical config file was found; built-in defaults will be used.",
        )
        _check_timezone(findings, {})
        _check_season_mode(findings, {})
        _check_astronomy(findings, {}, source_hint="defaults")
        _check_uda_aliases(findings, {})
        _check_config_drift(findings, "")
        _check_navigator_dependencies(findings, {})
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
    _check_uda_aliases(findings, data)
    _check_timezone(findings, data)
    _check_season_mode(findings, data)
    _check_astronomy(findings, data, source_hint=str(config))
    _check_config_drift(findings, str(config))
    _check_navigator_dependencies(findings, data)
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


def _check_uda_aliases(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Report whether description-based UDA aliases are active."""
    enabled = data.get("enable_uda_aliases") is True
    aliases = {alias: field for alias, field in description_aliases.ALIAS_TO_FIELD.items()}
    state = "enabled" if enabled else "disabled"
    _finding(
        findings,
        "config.uda_aliases",
        "ok",
        f"Description UDA aliases are {state}.",
        details={"enabled": enabled, "aliases": aliases, "clear_syntax": "alias:"},
    )


def _check_timezone(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    raw_tz = data.get("tz")
    tz_name = str(raw_tz or "UTC").strip() or "UTC"
    if not str(raw_tz or "").strip():
        _finding(
            findings,
            "config.timezone.missing",
            "warn",
            "No explicit Nautical timezone is configured; UTC fallback is active.",
            fix="Run Nautical install on a fresh target or set tz to an explicit IANA timezone in config-nautical.toml.",
            details={"tz": tz_name},
        )
        return
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


def _check_season_mode(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Report the active seasonal backend and preflight astronomical data."""
    snapshot = effective_config_snapshot()
    effective = snapshot.get("values") if isinstance(snapshot.get("values"), dict) else {}
    mode = str((data or {}).get("season_mode", effective.get("season_mode", "fixed")) or "fixed").strip().lower()
    hemisphere = str(
        (data or {}).get("season_hemisphere", effective.get("season_hemisphere", "north")) or "north"
    ).strip().lower()
    timezone_name = str((data or {}).get("tz", effective.get("tz", "UTC")) or "UTC").strip() or "UTC"
    if mode not in {"fixed", "astronomical"}:
        _finding(
            findings,
            "config.season_mode.invalid",
            "error",
            f"Unsupported seasonal boundary backend: {mode!r}.",
            fix="Set season_mode to 'fixed' or 'astronomical'.",
            details={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name},
        )
        return
    if mode == "fixed":
        _finding(
            findings,
            "config.season_mode",
            "ok",
            f"Seasonal boundaries use the fixed backend ({hemisphere} hemisphere).",
            details={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name},
        )
        return
    try:
        from nautical_core.astronomical_seasons import seasonal_events_utc

        events = seasonal_events_utc(date.today().year)
        if ZONEINFO_FACTORY is None:
            raise RuntimeError("zoneinfo support is unavailable")
        local_events = {
            name: event.astimezone(ZONEINFO_FACTORY(timezone_name)).date().isoformat()
            for name, event in events.items()
        }
    except Exception as exc:
        _finding(
            findings,
            "config.season_mode.astronomical_invalid",
            "error",
            f"Astronomical seasonal boundaries are unavailable for {date.today().year}: {exc}",
            fix="Verify timezone data and use a supported season year/backend, then rerun doctor.",
            details={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name},
        )
        return
    _finding(
        findings,
        "config.season_mode",
        "ok",
        f"Seasonal boundaries use astronomical transitions ({hemisphere} hemisphere, {timezone_name}).",
        details={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name, "events": local_events},
    )


def _check_astronomy(
    findings: list[dict[str, Any]],
    data: dict[str, Any],
    *,
    source_hint: str = "",
) -> None:
    """Check optional astronomy setup without touching Taskwarrior state."""
    snapshot = effective_config_snapshot()
    effective = snapshot.get("values") if isinstance(snapshot.get("values"), dict) else {}
    if isinstance(data, dict) and data:
        config = data.get("astronomy")
        effective_timezone = data.get("tz", effective.get("tz", "UTC"))
    else:
        config = effective.get("astronomy")
        effective_timezone = effective.get("tz", "UTC")
    result = astronomy.preflight(config)
    status = str(result.get("status") or "error")
    if status == "not_configured":
        _finding(
            findings,
            "astronomy.not_configured",
            "ok",
            "Astronomy is not configured; astronomical anchor times are disabled.",
            fix="Define [astronomy] locations only if using sunrise, sunset, moonrise, or moonset anchors.",
        )
        return
    severity = {"ok": "ok", "warning": "warn", "error": "error"}.get(status, "error")
    details = {key: value for key, value in result.items() if key not in {"status", "message"}}
    details["config_source"] = source_hint or snapshot.get("source", "unknown")
    details["effective_timezone"] = effective_timezone
    _finding(
        findings,
        "astronomy.preflight",
        severity,
        "Astronomy provider and location profile are usable."
        if status == "ok"
        else str(result.get("message") or "Astronomy preflight failed."),
        fix=(
            "Install astral in the active interpreter and verify the selected profile."
            if status == "error"
            else "Review the astronomy location and event availability before scheduling."
            if status == "warning"
            else ""
        ),
        details=details,
    )


def _check_config_drift(findings: list[dict[str, Any]], source_path: str) -> None:
    """Report runtime config drift when Doctor and core use the same source."""
    drift = configuration_drift()
    loaded_source = str(drift.get("source") or "")
    expected_source = os.path.abspath(str(source_path)) if source_path else "defaults"
    if loaded_source != expected_source:
        return
    if drift.get("changed"):
        _finding(
            findings,
            "config.drift",
            "warn",
            "The loaded Nautical configuration differs from the current file.",
            fix="Restart Navigator; Taskwarrior hooks will use the new configuration on their next invocation.",
            details={
                "source": loaded_source,
                "loaded_fingerprint": drift.get("loaded_fingerprint", ""),
                "current_fingerprint": drift.get("current_fingerprint", ""),
            },
        )
    else:
        _finding(
            findings,
            "config.drift",
            "ok",
            "Loaded Nautical configuration matches the current file.",
            details={"source": loaded_source, "fingerprint": drift.get("current_fingerprint", "")},
        )


def _check_navigator_dependencies(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Report dependencies required by the optional Navigator command."""
    def available(name: str) -> bool:
        try:
            return RICH_SPEC_FACTORY(name) is not None
        except Exception:
            return False

    required = {
        "rich": "formatted panels",
        "prompt_toolkit": "interactive chain selection",
        "dateutil": "datetime parsing",
    }
    missing = [name for name in required if not available(name)]
    if missing:
        _finding(
            findings,
            "navigator.dependencies",
            "warn",
            "Navigator dependencies are incomplete: " + ", ".join(missing) + ".",
            fix="Run python3 -m pip install -r requirements.txt.",
            details={"missing": missing, "python": sys.executable},
        )
    else:
        _finding(
            findings,
            "navigator.dependencies",
            "ok",
            "Navigator dependencies are available.",
            details={"python": sys.executable},
        )

    astronomy = data.get("astronomy")
    astronomy_configured = isinstance(astronomy, dict) and bool(astronomy.get("locations"))
    astral_available = available("astral")
    if astronomy_configured and not astral_available:
        _finding(
            findings,
            "navigator.astronomy_dependency",
            "warn",
            "Astronomy locations are configured, but Astral is not installed.",
            fix="Run python3 -m pip install -r requirements.txt.",
            details={"python": sys.executable},
        )


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


def _load_queue_status() -> ModuleType:
    path = TOOLS_DIR / "nautical_queue_status.py"
    spec = importlib.util.spec_from_file_location("_nautical_doctor_queue_status", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load queue status helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_lifecycle_outbox(findings: list[dict[str, Any]], taskdata: Path, stale_after: float) -> dict[str, Any]:
    try:
        module = _load_queue_status()
        payload = module._status_payload(taskdata, stale_after=stale_after, limit=5)
        if not isinstance(payload, dict):
            raise RuntimeError("lifecycle outbox status returned an invalid payload")
    except Exception as exc:
        _finding(
            findings,
            "outbox.unreadable",
            "error",
            "Nautical lifecycle outbox could not be inspected.",
            details={"error": str(exc)},
        )
        return {}
    outbox = payload.get("outbox") if isinstance(payload.get("outbox"), dict) else {}
    states = outbox.get("states") if isinstance(outbox.get("states"), dict) else {}
    quarantined = int(states.get("quarantined") or 0)
    if quarantined:
        _finding(
            findings,
            "outbox.poison_rows",
            "error",
            f"{quarantined} malformed lifecycle intent{'s' if quarantined != 1 else ''} quarantined.",
            fix="Inspect nautical queue-status and resolve the quarantined lifecycle intents.",
            details={"count": quarantined, "sample": outbox.get("sample") or []},
        )
    schema = outbox.get("schema") if isinstance(outbox.get("schema"), dict) else {}
    schema_status = str(schema.get("status") or "absent")
    if schema_status == "error":
        _finding(
            findings,
            "outbox.schema",
            "error",
            "Lifecycle outbox schema is incompatible with this Nautical runtime.",
            fix="Preserve the database, then upgrade Nautical or restore a compatible lifecycle outbox.",
            details=schema,
        )
    else:
        _finding(
            findings,
            "outbox.schema",
            "ok",
            (
                f"Lifecycle outbox schema v{schema.get('version')} is compatible."
                if schema_status == "ok"
                else "Lifecycle outbox has not been created yet."
            ),
            details=schema,
        )

    issues = payload.get("issues") or []
    outbox_status = str(payload.get("status") or "ok")
    _finding(
        findings,
        "outbox.state",
        "error" if outbox_status == "error" else ("warn" if issues else "ok"),
        "Lifecycle outbox has findings." if issues else "Lifecycle outbox is clean.",
        fix="Run nautical queue-status for lifecycle outbox details." if issues else "",
        details={"issues": issues} if issues else None,
    )
    retention = outbox.get("retention") if isinstance(outbox.get("retention"), dict) else {}
    eligible = int(retention.get("eligible") or 0)
    if eligible:
        _finding(
            findings,
            "outbox.retention",
            "warn",
            f"{eligible} acknowledged lifecycle intent{'s' if eligible != 1 else ''} exceed the retention policy.",
            fix="Run nautical queue-status --prune-acknowledged to remove only expired acknowledgements.",
            details=retention,
        )
    elif outbox.get("exists"):
        _finding(
            findings,
            "outbox.retention",
            "ok",
            "Lifecycle outbox retention is within policy.",
            details=retention,
        )
    return dict(payload)


_OBSOLETE_QUEUE_STATE_NAMES = (
    ".nautical_spawn_queue.jsonl",
    ".nautical_spawn_queue.processing.jsonl",
    ".nautical_spawn_queue.lock",
    ".nautical_queue.db",
    ".nautical_queue.db-wal",
    ".nautical_queue.db-shm",
    ".nautical_dead_letter.jsonl",
)


def _check_obsolete_queue_state(findings: list[dict[str, Any]], taskdata: Path) -> list[str]:
    """Report retired queue artifacts without reading or migrating them."""
    paths: list[str] = []
    roots = (taskdata, taskdata / ".nautical-state")
    for root in roots:
        for name in _OBSOLETE_QUEUE_STATE_NAMES:
            path = root / name
            if os.path.lexists(path):
                paths.append(str(path))
    if paths:
        _finding(
            findings,
            "outbox.obsolete_state",
            "warn",
            "Retired Nautical queue state was found; it is not used by this runtime.",
            fix=(
                "Back up any required records, then quarantine or remove the listed files; "
                "the lifecycle outbox is the only supported work store."
            ),
            details={"paths": sorted(set(paths))},
        )
    return sorted(set(paths))


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


def _existing_reconcile_children(rows: list[dict[str, Any]], parent: dict[str, Any]) -> list[dict[str, Any]]:
    chain_id = str(parent.get("chainID") or "").strip()
    next_link = lifecycle.int_or_default(parent.get("link"), 1) + 1
    include_deleted = str(parent.get("status") or "").strip() == "deleted"
    return [
        row
        for row in rows
        if str(row.get("chainID") or "").strip() == chain_id
        and lifecycle.int_or_default(row.get("link"), -1) == next_link
        and (include_deleted or str(row.get("status") or "").strip() != "deleted")
    ]


def _safe_parse_datetime(runtime: Any, value: Any):
    parser = getattr(runtime, "safe_parse_datetime", None) or getattr(runtime, "_safe_parse_datetime", None)
    if callable(parser):
        return parser(value)
    parser = getattr(runtime, "parse_dt_any", None)
    if not callable(parser):
        return None, "datetime parser unavailable"
    try:
        parsed = parser(value)
    except Exception as exc:
        return None, str(exc).strip() or type(exc).__name__
    return parsed, None if parsed is not None else f"unrecognized datetime: {value}"


def _check_reconcile_plans(
    findings: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
) -> None:
    completion_candidates = [row for row in rows if lifecycle.is_orphan_completion_candidate(row)]
    deleted_candidates = [row for row in rows if lifecycle.is_orphan_deleted_chain_candidate(row)]
    if not completion_candidates and not deleted_candidates:
        return

    hook = None
    generation = None
    plans: list[lifecycle.LifecycleRecoveryDecision] = []
    delayed_expiration_candidates: list[dict[str, Any]] = []
    unavailable = ""
    try:
        import nautical_core as core

        hook = core
        generation = ChainGenerationService.from_core(
            core,
            recurrence_update_udas=tuple(getattr(core, "RECURRENCE_UPDATE_UDAS", ()) or ()),
            debug_wait_sched=bool(getattr(core, "DEBUG_WAIT_SCHED", False)),
        )
    except Exception as exc:
        unavailable = str(exc)
    candidates = [*completion_candidates, *deleted_candidates]
    if not unavailable:
        for parent in candidates:
            existing_children = _existing_reconcile_children(rows, parent)
            plan = lifecycle.plan_recovery_decision(
                parent,
                existing_children=existing_children,
                hook=hook,
                generation=generation,
            )
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
                    until_dt, until_err = _safe_parse_datetime(hook, plan.child.get("until"))
                    now_utc = getattr(hook, "now_utc", None)
                    planned_until_elapsed = (
                        not until_err
                        and until_dt is not None
                        and callable(now_utc)
                        and compare_datetimes(until_dt, now_utc()) <= 0
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
                    **lifecycle.describe_plan(plan, fmt_dt_local=fmt_dt_local),
                }
                for plan in plans[:10]
            ],
        },
    )


def _check_chains(
    findings: list[dict[str, Any]],
    *,
    unit_of_work: TaskwarriorUnitOfWork | None,
) -> dict[str, int]:
    if unit_of_work is None:
        ok = False
        rows: list[dict[str, Any]] = []
        err = "validated integration context is unavailable"
    else:
        ok, rows, err = _task_export(unit_of_work)
    if not ok:
        _finding(
            findings,
            "chains.export",
            "error",
            "Task data could not be exported for chain inspection.",
            details={"error": err},
        )
        return {"tasks": 0, "nautical_tasks": 0, "chains": 0}

    from nautical_core.chain_integrity_engine import ChainIntegrityEngine
    from nautical_core.chain_snapshot import IntegritySnapshotRequest
    from nautical_core.integrity_report import doctor_findings
    from nautical_core.lifecycle_outbox import LifecycleOutboxRepository

    configuration = unit_of_work.context.configuration if unit_of_work is not None else None
    integrity = None
    if unit_of_work is not None and configuration is not None:
        integrity = ChainIntegrityEngine(
            ChainSnapshotService(unit_of_work, configuration_fingerprint=configuration.fingerprint),
            configuration_fingerprint=configuration.fingerprint,
            schedule_fingerprint=configuration.scheduler_fingerprint,
        ).audit(
            IntegritySnapshotRequest.candidates(complete_chain_history=True),
            outbox_repository=LifecycleOutboxRepository(unit_of_work.outbox.taskdata),
            mutation_epoch=unit_of_work.mutation_epoch,
        )
    if integrity is not None:
        findings.extend(doctor_findings(integrity))
    _check_reconcile_plans(findings, rows=rows)

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


def _render_details(details: dict[str, Any], *, stream: Any = None, enabled: bool = False) -> None:
    stream = stream if stream is not None else sys.stdout

    def write(line: str = "") -> None:
        print(line, file=stream)

    error = str(details.get("error") or "").strip()
    if error:
        write(f"  Detail: {error}")
    for issue in details.get("issues") or []:
        if not isinstance(issue, dict):
            write(f"  Detail: {issue}")
            continue
        write(
            "  Issue: "
            f"{issue.get('kind') or '?'} chain={issue.get('chainID') or '?'} "
            f"{issue.get('message') or ''}".rstrip()
        )
        for task in issue.get("tasks") or []:
            if not isinstance(task, dict):
                continue
            write(f"    Task: {_format_task(task)}")
            reason = str(task.get("reason") or "").strip()
            if reason:
                write(f"      Why: {reason}")
    for reason, count in (details.get("reasons") or {}).items():
        write(f"  Reason: {reason} ({count})")
    for repair in details.get("repairs") or []:
        if not isinstance(repair, dict):
            continue
        write(
            "  Repair: "
            f"{repair.get('task') or '?'} chain={repair.get('chainID') or '?'} "
            f"link={repair.get('link') or '?'} {repair.get('field') or '?'}: "
            f"{repair.get('old') or '-'} -> {repair.get('new') or '-'}"
        )
    for action, count in (details.get("actions") or {}).items():
        write(f"  Action: {action} ({count})")
    for plan in details.get("plans") or []:
        if not isinstance(plan, dict):
            continue
        write(
            "  Plan: "
            f"{plan.get('action') or '?'} parent={plan.get('parent') or '?'} "
            f"chain={plan.get('chainID') or '?'} next={plan.get('next_link') or '?'}"
        )
        reason = str(plan.get("reason") or "").strip()
        if reason:
            write(f"    Reason: {reason}")
        child = plan.get("existing_child") or plan.get("child_target") or plan.get("child_due")
        if child:
            write(f"    Child: {child}")
    for task in details.get("tasks") or []:
        if isinstance(task, dict):
            write(f"  Affected: {_format_task(task)}")
    for slot in details.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        write(f"  Slot: chain={slot.get('chainID') or '?'} link={slot.get('link')}")
        for task in slot.get("tasks") or []:
            if isinstance(task, dict):
                write(f"    Task: {_format_task(task)}")
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
        write(f"  Affected: {source}")
        write(f"    {field} -> {target_text}{suffix}")


def _render_text(payload: dict[str, Any], *, stream: Any = None) -> None:
    stream = stream if stream is not None else sys.stdout
    enabled = _color_enabled(stream)
    findings = [item for item in payload.get("findings") or [] if isinstance(item, dict)]
    counts = {
        severity: sum(1 for item in findings if item.get("severity") == severity)
        for severity in ("ok", "warn", "error")
    }
    status = str(payload.get("status") or "unknown")

    def write(line: str = "") -> None:
        print(line, file=stream)

    write(f"Nautical doctor: {_status_label(status, enabled=enabled)}")
    write(f"Taskdata: {payload.get('taskdata') or '?'}")
    ok_text = _paint(f"{counts['ok']} ok", "green", enabled=enabled)
    warn_text = _paint(f"{counts['warn']} warnings", "yellow", enabled=enabled)
    error_text = _paint(f"{counts['error']} failures", "red", enabled=enabled)
    write(
        "Checks: "
        f"{len(findings)} total | "
        f"{ok_text} | {warn_text} | {error_text}"
    )
    timezone = _timezone_summary(payload.get("findings") or [])
    if timezone:
        write(f"Timezone: {timezone}")
    for section in ("error", "warn", "ok"):
        items = [item for item in findings if item.get("severity") == section]
        if not items:
            continue
        heading = f"{_status_label(section, enabled=enabled)} {section.upper()} ({len(items)})"
        write(f"\n{heading}")
        for index, item in enumerate(items):
            if index:
                write()
            write(f"  {_status_label(section, enabled=enabled)} {item.get('id') or '?'}")
            write(f"    {item.get('message') or ''}")
            details = item.get("details")
            if isinstance(details, dict):
                _render_details(details, stream=stream, enabled=enabled)
            if item.get("fix"):
                write(f"    Fix: {_paint(str(item['fix']), 'yellow', enabled=enabled)}")


def _timezone_summary(findings: list[dict[str, Any]]) -> str:
    for item in findings:
        if not isinstance(item, dict):
            continue
        check_id = item.get("id")
        if check_id == "config.timezone":
            return str(item.get("message") or "").replace("Nautical timezone is available: ", "")
        if check_id in {"config.timezone.missing", "config.timezone.invalid", "config.timezone.unavailable"}:
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
    parser.add_argument("--clean-cache", action="store_true", help="prune expired and orphaned anchor cache files")
    args = parser.parse_args()

    env = os.environ.copy()
    findings: list[dict[str, Any]] = []
    unit_of_work = None
    try:
        unit_of_work = build_operator_uow(
            core=nautical_core_package,
            task_binary=args.task_bin,
            taskdata=args.taskdata,
            env=env,
            access=IntegrationAccess.READ_ONLY,
        )
    except Exception as exc:
        stage = str(getattr(exc, "stage", "context") or "context")
        failed_taskdata = getattr(exc, "taskdata", None)
        if stage not in {"configuration", "timezone"} or not isinstance(failed_taskdata, Path):
            message = f"Integration context unavailable: {exc}"
            if args.json:
                print(
                    json.dumps(
                        {
                            "schema": _JSON_SCHEMA,
                            "schema_version": _JSON_SCHEMA_VERSION,
                            "status": "error",
                            "findings": [
                                {"id": "integration.context", "severity": "error", "message": message}
                            ],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
            else:
                print(f"Nautical doctor: ERROR\nContext: {message}")
            return 1
        taskdata = failed_taskdata
        _finding(
            findings,
            "integration.context",
            "error",
            "The validated integration context could not be constructed.",
            fix="Correct the reported scheduling configuration before using Nautical hooks or mutations.",
            details={"error": str(exc)},
        )
        unit_of_work = _diagnostic_read_uow(taskdata, args.task_bin, env)
    else:
        taskdata = unit_of_work.context.taskdata
        args.task_bin = unit_of_work.context.command_prefix[0]
    env["TASKDATA"] = str(taskdata)

    hooks_dir = _check_runtime(
        findings,
        unit_of_work=unit_of_work,
        taskdata=taskdata,
    )
    hook_runtimes = _check_hooks_and_udas(
        findings,
        unit_of_work=unit_of_work,
        hooks_dir=hooks_dir,
        env=env,
    )
    _check_managed_runtime(findings, hooks_dir, hook_runtimes)
    _check_config(findings, taskdata)
    if args.clean_cache:
        gc_result = nautical_core_package.cache_gc()
        gc_errors = int(gc_result.get("errors", 0) or 0)
        _finding(
            findings,
            "cache.gc",
            "error" if gc_errors else "ok",
            "Anchor cache cleanup completed." if not gc_errors else "Anchor cache cleanup completed with errors.",
            details=gc_result,
        )
    outbox = _check_lifecycle_outbox(findings, taskdata, max(0.0, args.stale_after_seconds))
    obsolete_queue_state = _check_obsolete_queue_state(findings, taskdata)
    counts = _check_chains(
        findings,
        unit_of_work=unit_of_work,
    )

    status = _overall_status(findings)
    payload = {
        "schema": _JSON_SCHEMA,
        "schema_version": _JSON_SCHEMA_VERSION,
        "status": status,
        "taskdata": str(taskdata),
        "hooks_dir": str(hooks_dir),
        "counts": counts,
        "outbox": outbox,
        "obsolete_queue_state": obsolete_queue_state,
        "findings": findings,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    else:
        _render_text(payload)
    return 2 if status == "error" else 1 if status == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
