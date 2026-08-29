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
from typing import Any, Callable, Mapping, cast

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
from nautical_core.operator_context import OperatorInvocationBudget  # noqa: E402
from nautical_core.operator_models import OperatorLimits  # noqa: E402
from nautical_core.operator_findings import OperatorFinding, doctor_finding  # noqa: E402
from nautical_core import astronomy, configuration_drift, config_schema, description_aliases, effective_config_snapshot, install_runtime  # noqa: E402
from nautical_core.doctor_report import format_task, historical_summaries, render_finding, timezone_summary, to_operator_result  # noqa: E402
from nautical_core.integration_context import (  # noqa: E402
    IntegrationAccess,
    IntegrationContext,
    SilentDiagnostics,
    SystemClock,
    ValidatedNauticalConfiguration,
)
from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork, build_operator_uow  # noqa: E402
from nautical_core.operator_control_plane import OperatorControlPlane  # noqa: E402
from nautical_core.operator_health_service import ConfigurationDiagnosisRequest, OperatorHealthService, TaskwarriorDiagnosisRequest  # noqa: E402
from nautical_core.operator_application import DomainApplicationRegistry  # noqa: E402
from nautical_core.queue_status_service import QueueStatusService  # noqa: E402

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
    findings.append(doctor_finding(
        check_id, severity, message, guidance=fix, details=details,
    ).to_doctor_dict())


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
    task_error = ""
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
        task_error = str(proc.stderr or proc.stdout or "").strip() or f"{proc.kind.value} (exit code {proc.returncode})"
    report, hooks_dir = OperatorHealthService.diagnose_taskwarrior(TaskwarriorDiagnosisRequest(
        probe=lambda: (proc.ok, task_error),
        taskdata=taskdata,
        hooks_location=lambda: _task_get(unit_of_work, "rc.hooks.location")[1],
        default_hooks_dir=taskdata / "hooks",
    ))
    findings.extend(item.to_doctor_dict() for item in report.findings)
    return Path(str(hooks_dir))


def _check_hooks_and_udas(
    findings: list[dict[str, Any]],
    *,
    unit_of_work: TaskwarriorUnitOfWork,
    hooks_dir: Path,
    env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    typed, validated = OperatorHealthService.hook_installation_findings(
        hooks_dir, install_runtime.HOOK_RUNTIME_FILES, install_runtime.hook_candidates,
        install_runtime.inspect_hook_runtime, env,
    )
    findings.extend(item.to_doctor_dict() for item in typed)
    findings.extend(item.to_doctor_dict() for item in OperatorHealthService.uda_registration_findings(
        REQUIRED_UDAS,
        lambda name: _task_get(unit_of_work, f"rc.uda.{name}.type"),
    ))
    return validated


def _check_managed_runtime(
    findings: list[dict[str, Any]],
    hooks_dir: Path,
    hook_runtimes: dict[str, dict[str, Any]] | None = None,
) -> None:
    report = OperatorHealthService.diagnose_runtime(
        lambda: install_runtime.runtime_status(hooks_dir.parent),
        hooks_dir.parent,
        hook_runtimes,
    )
    findings.extend(item.to_doctor_dict() for item in report.findings)


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
            fix="Create config-nautical.toml or set NAUTICAL_CONFIG to a valid configuration file.",
        )
        from nautical_core.astronomical_seasons import seasonal_events_utc
        astronomy_preflight = cast(Callable[[object], dict[str, Any]], astronomy.preflight)
        report = OperatorHealthService.diagnose_configuration(ConfigurationDiagnosisRequest(
            {}, effective={}, config_dir=taskdata, timezone_factory=ZONEINFO_FACTORY,
            seasonal_events=seasonal_events_utc,
            astronomy_preflight=astronomy_preflight, source_path="defaults",
            drift_loader=configuration_drift, dependency_available=lambda name: RICH_SPEC_FACTORY(name) is not None,
            python_executable=sys.executable, rich_factory=RICH_SPEC_FACTORY,
        ))
        findings.extend(item.to_doctor_dict() for item in report.findings)
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
    snapshot = effective_config_snapshot()
    effective_value = snapshot.get("values")
    effective = effective_value if isinstance(effective_value, dict) else {}
    from nautical_core.astronomical_seasons import seasonal_events_utc
    astronomy_preflight = cast(Callable[[object], dict[str, Any]], astronomy.preflight)
    report = OperatorHealthService.diagnose_configuration(ConfigurationDiagnosisRequest(
        data, effective=effective, config_dir=config.parent, timezone_factory=ZONEINFO_FACTORY,
        seasonal_events=seasonal_events_utc, astronomy_preflight=astronomy_preflight,
        source_path=str(config), drift_loader=configuration_drift,
        dependency_available=lambda name: RICH_SPEC_FACTORY(name) is not None,
        python_executable=sys.executable, rich_factory=RICH_SPEC_FACTORY,
    ))
    findings.extend(item.to_doctor_dict() for item in report.findings)


def _check_lifecycle_outbox(findings: list[dict[str, Any]], taskdata: Path, stale_after: float) -> dict[str, Any]:
    try:
        payload = QueueStatusService().status_payload(taskdata, stale_after=stale_after, limit=5)
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
            write(f"    Task: {format_task(task)}")
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
            write(f"  Affected: {format_task(task)}")
    for slot in details.get("slots") or []:
        if not isinstance(slot, dict):
            continue
        write(f"  Slot: chain={slot.get('chainID') or '?'} link={slot.get('link')}")
        for task in slot.get("tasks") or []:
            if isinstance(task, dict):
                write(f"    Task: {format_task(task)}")
    for link in details.get("links") or []:
        if not isinstance(link, dict):
            continue
        task = link.get("task")
        source = format_task(task) if isinstance(task, dict) else "unknown task"
        target = link.get("target")
        target_text = format_task(target) if isinstance(target, dict) else str(target or "?")
        field = str(link.get("field") or "link")
        matches = link.get("matches")
        suffix = f" ({matches} matches)" if matches is not None else ""
        write(f"  Affected: {source}")
        write(f"    {field} -> {target_text}{suffix}")


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
    raw_findings = list(ordered_findings([render_finding(item) for item in visible_source]))
    historical = [
        item for item in raw_findings
        if item.get("severity") == "info"
        and isinstance(item.get("details"), dict)
        and item["details"].get("historical")
    ]
    findings = [item for item in raw_findings if item not in historical]
    historical_groups = historical_summaries(historical)
    findings.extend(historical_groups)
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
        f"{len(historical)} historical observations in {len(historical_groups)} groups",
        "cyan",
        enabled=enabled,
    )
    write(
        "Checks: "
        f"{len(findings) - len(historical_groups)} current | "
        f"{ok_text} | {warn_text} | {error_text} | {info_text}"
    )
    timezone = timezone_summary([render_finding(item) for item in source_findings])
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
    # Doctor audits the full task database, so use bounded full-system limits
    # rather than the smaller scoped-query defaults.
    budget = OperatorInvocationBudget(OperatorLimits(tasks=10_000, chains=1_000))

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
                    render_result(to_operator_result({
                            "schema": _JSON_SCHEMA,
                            "schema_version": _JSON_SCHEMA_VERSION,
                            "status": "error",
                            "findings": [
                                {"id": "integration.context", "severity": "error", "message": message}
                            ],
                        }), "json", budget=budget)
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
        obsolete_queue_state = sorted({
            str(root / name)
            for root in (taskdata, taskdata / ".nautical-state")
            for name in _OBSOLETE_QUEUE_STATE_NAMES
            if os.path.lexists(root / name)
        })
        findings.extend(item.to_doctor_dict() for item in OperatorHealthService.obsolete_queue_findings(
            taskdata, _OBSOLETE_QUEUE_STATE_NAMES,
        ))
        if unit_of_work is None:
            counts = {"tasks": 0, "nautical_tasks": 0, "chains": 0}
        else:
            control_plane = OperatorControlPlane.from_configuration(
                unit_of_work.context.configuration,
                DomainApplicationRegistry(),
            )
            counts, chain_findings = control_plane.diagnose_chains(unit_of_work, budget=budget)
            findings.extend(chain_findings)

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
                print(render_result(to_operator_result(payload), "json", budget=budget))
    else:
        _render_text(payload)
    return 2 if status == "error" else 1 if status == "warn" else 0


if __name__ == "__main__":
    raise SystemExit(main())
