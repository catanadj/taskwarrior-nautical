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
from datetime import timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

ZONEINFO_FACTORY: Callable[[str], Any] | None = getattr(zoneinfo, "ZoneInfo", None)
RICH_SPEC_FACTORY: Callable[[str], Any] = importlib.util.find_spec


TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import nautical_core as nautical_core_package  # noqa: E402
except Exception as exc:  # configuration can fail while the package is importing
    # Keep the Doctor process contract intact even when startup configuration
    # cannot be loaded: callers still receive a structured, actionable result.
    if __name__ == "__main__":
        print(json.dumps({
            "schema": "nautical.doctor",
            "schema_version": 1,
            "status": "error",
            "findings": [{
                "id": "integration.startup",
                "severity": "error",
                "message": f"Nautical configuration could not be loaded: {exc}",
                "fix": "Correct the reported configuration, then rerun nautical doctor.",
            }],
        }, ensure_ascii=False, separators=(",", ":")))
        raise SystemExit(1)
    raise
from nautical_core.operator_presentation import finding_display, finding_status, group_findings_by_severity, ordered_findings, render_result  # noqa: E402
from nautical_core.operator_findings import FindingActionability, FindingSeverity, OperatorFinding  # noqa: E402
from nautical_core import astronomy, configuration_drift, config_schema, description_aliases, effective_config_snapshot, install_runtime  # noqa: E402
from nautical_core import chain_integrity_lifecycle as lifecycle  # noqa: E402
from nautical_core.operator_models import OperatorFailure, OperatorV2Result, OperatorV2Status  # noqa: E402
from nautical_core.integration_context import (  # noqa: E402
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork, build_operator_uow  # noqa: E402
from nautical_core.operator_control_plane import OperatorControlPlane  # noqa: E402
from nautical_core.operator_health_service import OperatorHealthService  # noqa: E402
from nautical_core.operator_application import DomainApplicationRegistry  # noqa: E402

_JSON_SCHEMA = "nautical.doctor"
_JSON_SCHEMA_VERSION = 1


def _v2_document(payload: dict[str, Any]) -> OperatorV2Result:
    """Build the shared v2 envelope while retaining Doctor's public fields."""
    raw_status = str(payload.get("status") or "error")
    # Preserve Doctor's v1 ``warn`` payload while using the canonical v2
    # attention status in the shared envelope.
    status = OperatorV2Status.ATTENTION if raw_status == "warn" else OperatorV2Status(raw_status)
    failure = None
    if status in {OperatorV2Status.ERROR, OperatorV2Status.UNAVAILABLE, OperatorV2Status.INVALID}:
        finding = next((item for item in payload.get("operator_findings", ()) if isinstance(item, dict)), {})
        failure = OperatorFailure(
            code=str(finding.get("code") or "doctor_error"),
            message=str(finding.get("message") or "Doctor reported an error"),
            details=finding.get("evidence") if isinstance(finding.get("evidence"), dict) else {},
        )
    return OperatorV2Result(
        schema=_JSON_SCHEMA,
        operation="diagnose",
        status=status,
        payload={key: value for key, value in payload.items() if key not in {"schema", "status"}},
        failure=failure,
    )

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
    # Validate every Doctor observation through the shared operator contract.
    # The v1 Doctor envelope is retained until its dedicated serialization pass.
    normalized_severity = (
        FindingSeverity.ERROR
        if severity == "error"
        else FindingSeverity.WARNING
        if severity == "warn"
        else FindingSeverity.INFO
    )
    canonical = OperatorFinding(
        code=check_id,
        domain=check_id.split(".", 1)[0] or "doctor",
        severity=normalized_severity,
        actionability=(
            FindingActionability.INFORMATIONAL
            if severity == "info" and not fix
            else FindingActionability.ACTIONABLE
        ),
        message=message,
        observed=details or {},
        guidance=fix or ("Inspect the reported evidence." if severity != "info" else ""),
    )
    findings.append(canonical.to_doctor_dict())


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


def _diagnostic_read_uow(
    taskdata: Path,
    task_bin: str,
    env: dict[str, str],
) -> TaskwarriorUnitOfWork:
    """Read Taskwarrior state without claiming valid scheduling configuration."""
    diagnostic_env = dict(env)
    diagnostic_env["TASKDATA"] = str(taskdata.resolve())
    context = IntegrationContext(
        taskdata.resolve(),
        "doctor-recovery",
        (str(task_bin),),
        ValidatedNauticalConfiguration("doctor", "unavailable", "unavailable", "UTC", ()),
        timezone.utc,
        SilentDiagnostics(),
        SystemClock(),
        "doctor-diagnostic-read",
        256,
        IntegrationAccess.READ_ONLY,
    )
    return TaskwarriorUnitOfWork.create(context, env=diagnostic_env)


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
        [],
        purpose="doctor Taskwarrior query",
        timeout=30.0,
        attempts=2,
        retry_delay=0.1,
    )
    # Minimal Taskwarrior shims may only support the version probe. Keep this
    # narrow compatibility path; real command failures remain reported.
    if not proc.ok:
        version_probe = unit_of_work.client.execute(
            ["--version"],
            purpose="doctor Taskwarrior version fallback",
            timeout=30.0,
            attempts=1,
        )
        if version_probe.ok:
            proc = version_probe
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
            "Taskwarrior command is available.",
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
    udas_valid = True
    for name, expected in REQUIRED_UDAS.items():
        ok, actual = _task_get(unit_of_work, f"rc.uda.{name}.type")
        if not ok or not actual:
            udas_valid = False
            _finding(
                findings,
                f"uda.{name}.missing",
                "error",
                f"Required UDA '{name}' is not defined.",
                fix="Include Nautical's uda.conf from your Taskwarrior configuration.",
            )
        elif actual.lower() != expected:
            udas_valid = False
            _finding(
                findings,
                f"uda.{name}.type",
                "error",
                f"UDA '{name}' has type '{actual}', expected '{expected}'.",
                fix=f"Set uda.{name}.type={expected}.",
            )
    if udas_valid:
        _finding(
            findings,
            "uda.registration",
            "ok",
            f"All {len(REQUIRED_UDAS)} required Nautical UDAs are registered.",
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
        findings.extend(item.to_doctor_dict() for item in OperatorHealthService.runtime_findings(
            {"managed": True, "errors": [str(exc)]}, hooks_dir.parent, hook_runtimes,
        ))
        return
    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.runtime_findings(
        status, hooks_dir.parent, hook_runtimes,
    ))


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
    findings.extend(
        item.to_doctor_dict()
        for item in OperatorHealthService.directory_findings(data, config.parent)
    )


def _check_config_schema(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    findings.extend(
        item.to_doctor_dict()
        for item in OperatorHealthService.configuration_schema_findings(data)
    )


def _check_uda_aliases(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Report whether description-based UDA aliases are active."""
    findings.extend(
        item.to_doctor_dict()
        for item in OperatorHealthService.uda_alias_findings(data)
    )


def _check_timezone(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    findings.extend(
        item.to_doctor_dict()
        for item in OperatorHealthService.timezone_findings(data, ZONEINFO_FACTORY)
    )


def _check_season_mode(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    snapshot = effective_config_snapshot()
    effective_value = snapshot.get("values")
    effective = effective_value if isinstance(effective_value, dict) else {}
    from nautical_core.astronomical_seasons import seasonal_events_utc
    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.season_findings(
        data, effective, ZONEINFO_FACTORY, seasonal_events_utc,
    ))


def _check_astronomy(
    findings: list[dict[str, Any]],
    data: dict[str, Any],
    *,
    source_hint: str = "",
) -> None:
    snapshot = effective_config_snapshot()
    effective_value = snapshot.get("values")
    effective = effective_value if isinstance(effective_value, dict) else {}
    config = data.get("astronomy") if isinstance(data, dict) and data else effective.get("astronomy")
    effective_timezone = data.get("tz", effective.get("tz", "UTC")) if isinstance(data, dict) and data else effective.get("tz", "UTC")
    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.astronomy_findings(
        config,
        effective_timezone=effective_timezone,
        source_hint=source_hint or snapshot.get("source", "unknown"),
        preflight=astronomy.preflight,
    ))


def _check_config_drift(findings: list[dict[str, Any]], source_path: str) -> None:
    """Report runtime config drift when Doctor and core use the same source."""
    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.configuration_drift_findings(
        source_path, configuration_drift,
    ))


def _check_navigator_dependencies(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Report dependencies required by the optional Navigator command."""
    def available(name: str) -> bool:
        try:
            return RICH_SPEC_FACTORY(name) is not None
        except Exception:
            return False

    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.navigator_dependency_findings(
        data, available, python_executable=sys.executable,
    ))


def _check_panel_config(findings: list[dict[str, Any]], data: dict[str, Any]) -> None:
    findings.extend(
        item.to_doctor_dict()
        for item in OperatorHealthService.panel_findings(data, RICH_SPEC_FACTORY)
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
    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.lifecycle_outbox_findings(payload))
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


def _check_chains(
    findings: list[dict[str, Any]],
    *,
    unit_of_work: TaskwarriorUnitOfWork | None,
) -> dict[str, int]:
    if unit_of_work is None:
        return {"tasks": 0, "nautical_tasks": 0, "chains": 0}
    configuration = unit_of_work.context.configuration
    control_plane = OperatorControlPlane.from_configuration(
        configuration,
        DomainApplicationRegistry(),
    )
    counts, chain_findings = control_plane.diagnose_chains(unit_of_work)
    findings.extend(chain_findings)
    return counts


def _format_task(task: dict[str, Any]) -> str:
    uuid = str(task.get("uuid") or "")
    short = lifecycle.short_uuid(uuid) or "unknown"
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

    chain_id = str(details.get("chainID") or details.get("chain_id") or "").strip()
    subjects = details.get("subjects") or details.get("subject_uuids") or ()
    if chain_id or subjects:
        subject_values = [str(item).strip() for item in subjects if str(item).strip()]
        suffix = ""
        if subject_values:
            shown = ",".join(value[:8] for value in subject_values[:8])
            suffix = f" tasks={shown}{'…' if len(subject_values) > 8 else ''}"
        write(f"  Affected: chain={chain_id or '?'}{suffix}")
    observed = details.get("observed")
    expected = details.get("expected")
    if observed:
        write(f"  Observed: {observed}")
    if expected:
        write(f"  Expected: {expected}")
    historical_count = details.get("historical_count")
    if historical_count:
        write(f"  Observations: {historical_count} completed-link finding(s)")
    detail_command = str(details.get("detail_command") or "").strip()
    if detail_command:
        write(f"  Details: {detail_command}")
    error = str(details.get("error") or "").strip()
    if error:
        write(f"  Detail: {error}")
    reason = str(details.get("reason") or "").strip()
    if reason and reason != error:
        write(f"  Detail: {reason}")
    for issue in details.get("issues") or []:
        if not isinstance(issue, dict):
            write(f"  Detail: {issue}")
            continue
        kind = issue.get("kind") or issue.get("invariant_id") or "?"
        chain_id = issue.get("chainID") or issue.get("chain_id") or "?"
        subjects = issue.get("subjects") or issue.get("subject_uuids") or ()
        subject_text = f" subjects={','.join(str(item)[:8] for item in subjects[:3])}" if subjects else ""
        write(
            "  Issue: "
            f"{kind} chain={chain_id}{subject_text} "
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


def _historical_summaries(findings: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Collapse completed-link evidence while leaving JSON output lossless."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in findings:
        if item.get("severity") != "info":
            continue
        details_value = item.get("details")
        details: dict[str, Any] = dict(details_value) if isinstance(details_value, dict) else {}
        if not details.get("historical"):
            continue
        chain_id = str(details.get("chainID") or "").strip() or "unassigned"
        invariant_id = str(details.get("invariant_id") or item.get("id") or "historical")
        observed_value = details.get("observed")
        observed: dict[str, Any] = dict(observed_value) if isinstance(observed_value, dict) else {}
        field = str(observed.get("field") or "").strip()
        key = (chain_id, invariant_id, field)
        group = groups.setdefault(key, {"count": 0, "subjects": set()})
        group["count"] += 1
        group["subjects"].update(str(value) for value in details.get("subjects") or () if value)

    summaries: list[dict[str, Any]] = []
    for (chain_id, invariant_id, field), group in sorted(groups.items()):
        count = int(group["count"])
        label = f" {field}" if field else ""
        summaries.append({
            "id": "chains.historical_summary",
            "severity": "info",
            "message": f"{count} completed-link{label} observation(s) retained for audit.",
            "fix": "No action is required; current pending-chain findings are reported separately.",
            "details": {
                "chainID": chain_id,
                "invariant_id": invariant_id,
                "historical_count": count,
                "subjects": sorted(group["subjects"])[:8],
                "detail_command": f"nautical query integrity --chain-id {chain_id}",
            },
        })
    return summaries


def _render_finding(item: object) -> dict[str, Any]:
    """Project a canonical finding into the renderer's detail-oriented view."""
    if not isinstance(item, dict):
        return {}
    if "code" not in item:
        return item
    severity = str(item.get("severity") or "info")
    severity = "warn" if severity == "warning" else severity
    evidence = item.get("evidence")
    details = dict(evidence) if isinstance(evidence, dict) else {}
    observed = item.get("observed")
    expected = item.get("expected")
    if isinstance(observed, dict):
        details["observed"] = observed
    if isinstance(expected, dict):
        details["expected"] = expected
    return {
        "id": item.get("code"),
        "severity": severity,
        "message": item.get("message"),
        "fix": item.get("guidance") or "",
        "details": details,
    }


def _render_text(payload: dict[str, Any], *, stream: Any = None) -> None:
    stream = stream if stream is not None else sys.stdout
    enabled = _color_enabled(stream)
    source_findings = payload.get("operator_findings") or []
    # Text is the operator-facing view: keep blocking/actionable observations
    # visible while leaving healthy inventory available through ``--json``.
    visible_source = [
        item for item in source_findings
        if isinstance(item, dict)
        and (
            str(item.get("severity") or "").lower() in {"error", "warning", "warn"}
            or str(item.get("actionability") or "").lower() in {"actionable", "blocking"}
            or (isinstance(item.get("evidence"), dict) and item["evidence"].get("historical"))
        )
    ]
    raw_findings = list(ordered_findings([_render_finding(item) for item in visible_source]))
    historical = [
        item for item in raw_findings
        if item.get("severity") == "info"
        and isinstance(item.get("details"), dict)
        and item["details"].get("historical")
    ]
    findings = [item for item in raw_findings if item not in historical]
    historical_summaries = _historical_summaries(historical)
    findings.extend(historical_summaries)
    grouped = group_findings_by_severity(findings)
    counts = {
        severity: len(grouped.get(severity, ()))
        for severity in ("ok", "warn", "error", "info")
    }
    status = str(payload.get("status") or "unknown")

    def write(line: str = "") -> None:
        print(line, file=stream)

    write(f"Nautical doctor: {_status_label(status, enabled=enabled)}")
    write(f"Taskdata: {payload.get('taskdata') or '?'}")
    ok_text = _paint(f"{counts['ok']} ok", "green", enabled=enabled)
    warn_text = _paint(f"{counts['warn']} warnings", "yellow", enabled=enabled)
    error_text = _paint(f"{counts['error']} failures", "red", enabled=enabled)
    info_text = _paint(
        f"{len(historical)} historical observations in {len(historical_summaries)} groups",
        "cyan",
        enabled=enabled,
    )
    write(
        "Checks: "
        f"{len(findings) - len(historical_summaries)} current | "
        f"{ok_text} | {warn_text} | {error_text} | {info_text}"
    )
    timezone = _timezone_summary([_render_finding(item) for item in source_findings])
    if timezone:
        write(f"Timezone: {timezone}")
    for section in ("error", "warn", "info", "ok"):
        items = list(grouped.get(section, ()))
        if not items:
            continue
        heading = f"{_status_label(section, enabled=enabled)} {section.upper()} ({len(items)})"
        write(f"\n{heading}")
        for index, item in enumerate(items):
            if index:
                write()
            code, message, guidance = finding_display(item)
            write(f"  {_status_label(section, enabled=enabled)} {code}")
            write(f"    {message}")
            details = item.get("details")
            if isinstance(details, dict):
                _render_details(details, stream=stream, enabled=enabled)
            if guidance:
                write(f"    Fix: {_paint(guidance, 'yellow', enabled=enabled)}")


def _timezone_summary(findings: list[dict[str, Any]]) -> str:
    for item in findings:
        if not isinstance(item, dict):
            continue
        check_id = item.get("id")
        if check_id == "config.timezone":
            return str(item.get("message") or "").replace("Nautical timezone is available: ", "")
        if check_id in {"config.timezone.missing", "config.timezone.invalid", "config.timezone.unavailable"}:
            details_value = item.get("details")
            details: dict[str, Any] = dict(details_value) if isinstance(details_value, dict) else {}
            tz_name = str(details.get("tz") or "?")
            return f"{tz_name} unavailable; UTC fallback active"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--taskdata", default=None)
    parser.add_argument("--task-bin", default=shutil.which("task") or "task")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    parser.add_argument(
        "--installation-only",
        action="store_true",
        help="check installation and configuration without auditing task chains or lifecycle state",
    )
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
                    render_result(_v2_document({
                            "schema": _JSON_SCHEMA,
                            "schema_version": _JSON_SCHEMA_VERSION,
                            "status": "error",
                            "findings": [
                                {"id": "integration.context", "severity": "error", "message": message}
                            ],
                        }), "json")
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
    if args.installation_only:
        outbox = {}
        obsolete_queue_state = []
        counts = {"tasks": 0, "nautical_tasks": 0, "chains": 0}
    else:
        outbox = _check_lifecycle_outbox(findings, taskdata, max(0.0, args.stale_after_seconds))
        obsolete_queue_state = _check_obsolete_queue_state(findings, taskdata)
        counts = _check_chains(
            findings,
            unit_of_work=unit_of_work,
        )

    typed_findings = tuple(
        OperatorFinding.from_mapping(item)
        for item in findings
        if isinstance(item, dict)
    )
    health = OperatorControlPlane.health_report(typed_findings)
    status = (
        "error"
        if health.status.value in {"error", "unavailable"}
        else "warn"
        if health.status.value in {"attention", "deferred", "partial"}
        else "ok"
    )
    payload = {
        "schema": _JSON_SCHEMA,
        "schema_version": _JSON_SCHEMA_VERSION,
        "status": status,
        "taskdata": str(taskdata),
        "hooks_dir": str(hooks_dir),
        "counts": counts,
        "outbox": outbox,
        "obsolete_queue_state": obsolete_queue_state,
        "scope": "installation" if args.installation_only else "full",
        "operator_findings": [item.to_dict() for item in health.findings],
    }
    if args.json:
        # Diagnostics may include timezone/provider objects supplied by an
        # optional backend. Keep the JSON boundary strict without allowing a
        # presentation-only value to make Doctor crash.
                print(render_result(_v2_document(payload), "json"))
    else:
        _render_text(payload)
    return 2 if status == "error" else 1 if status == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
