"""Typed aggregation boundary for operator health observations."""

from __future__ import annotations

from dataclasses import dataclass
import os
import subprocess
import sqlite3
import json
import time
from typing import Any, Callable, Iterable, Mapping
from datetime import date, datetime
from pathlib import Path

from .operator_findings import (
    FindingActionability,
    FindingSeverity,
    OperatorFinding,
    deduplicate_findings,
    sort_findings,
    status_for_findings,
)
from .operator_models import OperatorStatus
from .config_schema import CONFIG_SPECS, validate_config
from .description_aliases import ALIAS_TO_FIELD


def _json_safe(value: object) -> object:
    """Normalize provider evidence before it enters the strict finding model."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, Path)):
        return value.isoformat() if isinstance(value, (datetime, date)) else str(value)
    zone_key = getattr(value, "key", None)
    if zone_key is not None:
        return str(zone_key)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class OperatorHealthReport:
    """Immutable health result assembled from typed findings."""

    findings: tuple[OperatorFinding, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(item, OperatorFinding) for item in self.findings):
            raise TypeError("health findings must be OperatorFinding values")
        normalized = sort_findings(deduplicate_findings(tuple(self.findings)))
        object.__setattr__(self, "findings", normalized)

    @property
    def status(self) -> OperatorStatus:
        return status_for_findings(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "findings": [finding.to_dict() for finding in self.findings],
        }


@dataclass(frozen=True, slots=True)
class ConfigurationDiagnosisRequest:
    """Validated inputs for one read-only configuration diagnosis."""

    data: dict[str, Any]
    effective: dict[str, Any]
    config_dir: object
    timezone_factory: Callable[[str], object] | None
    seasonal_events: Callable[[int], dict[str, Any]]
    astronomy_preflight: Callable[[object], dict[str, Any]]
    source_path: str
    drift_loader: Callable[[], dict[str, Any]]
    dependency_available: Callable[[str], bool]
    python_executable: str
    rich_factory: Callable[[str], object]

    def __post_init__(self) -> None:
        if not isinstance(self.data, dict) or not isinstance(self.effective, dict):
            raise TypeError("configuration diagnosis requires mapping data")
        for name in (
            "seasonal_events", "astronomy_preflight", "drift_loader",
            "dependency_available", "rich_factory",
        ):
            if not callable(getattr(self, name)):
                raise TypeError(f"configuration diagnosis requires callable {name}")


@dataclass(frozen=True, slots=True)
class TaskwarriorDiagnosisRequest:
    """Inputs for one read-only Taskwarrior invocation diagnosis."""

    probe: Callable[[], tuple[bool, str]]
    taskdata: object
    hooks_location: Callable[[], str]
    default_hooks_dir: object

    def __post_init__(self) -> None:
        for name in ("probe", "hooks_location"):
            if not callable(getattr(self, name)):
                raise TypeError(f"Taskwarrior diagnosis requires callable {name}")


class OperatorHealthService:
    """Aggregate health observations without performing I/O or mutations."""

    @staticmethod
    def report(findings: Iterable[OperatorFinding]) -> OperatorHealthReport:
        return OperatorHealthReport(tuple(findings))

    @staticmethod
    def storage_findings(
        paths: Mapping[str, object],
        *,
        statvfs_factory: Callable[[str], object] = os.statvfs,
    ) -> tuple[OperatorFinding, ...]:
        """Report read-only filesystem capacity for explicitly supplied paths."""
        findings: list[OperatorFinding] = []
        for label, raw_path in paths.items():
            path = str(raw_path)
            try:
                stats = statvfs_factory(path)
                block_size = int(getattr(stats, "f_frsize", 0) or getattr(stats, "f_bsize", 0))
                free_blocks = int(getattr(stats, "f_bavail"))
                total_blocks = int(getattr(stats, "f_blocks"))
                free_inodes = int(getattr(stats, "f_favail"))
                total_inodes = int(getattr(stats, "f_files"))
                if block_size < 1 or min(free_blocks, total_blocks, free_inodes, total_inodes) < 0:
                    raise ValueError("statvfs returned invalid capacity values")
                findings.append(OperatorFinding(
                    f"storage.{label}", "installation", FindingSeverity.INFO,
                    FindingActionability.INFORMATIONAL,
                    f"Storage capacity is readable for {label}: {path}.",
                    observed={
                        "path": path,
                        "free_bytes": free_blocks * block_size,
                        "total_bytes": total_blocks * block_size,
                        "free_inodes": free_inodes,
                        "total_inodes": total_inodes,
                    },
                ))
            except Exception as exc:
                findings.append(OperatorFinding(
                    f"storage.{label}", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING,
                    f"Storage capacity is unavailable for {label}: {path}.",
                    observed={"path": path, "error": str(exc)},
                    guidance=f"Make the {label} filesystem readable and retry deep Doctor.",
                ))
        return tuple(findings)

    @staticmethod
    def deep_identity_findings(
        runtime: Mapping[str, Any],
        task_binary: str,
        python_executable: str,
        *,
        digest_factory: Callable[[Path], str] | None = None,
        version_probe: Callable[[str], tuple[bool, str]] | None = None,
    ) -> tuple[OperatorFinding, ...]:
        """Verify active-release content and executable identities read-only."""
        findings: list[OperatorFinding] = []
        runtime_root = Path(str(runtime.get("runtime_root") or "")).expanduser()
        release_id = str(runtime.get("active_release") or "")
        manifest = runtime.get("manifest") if isinstance(runtime.get("manifest"), Mapping) else {}
        expected_digest = str(manifest.get("content_sha256") or "")
        release_path = runtime_root / "releases" / release_id if release_id else Path("")
        try:
            if digest_factory is None:
                from .install_runtime import source_digest
                digest_factory = source_digest
            if not release_id or not expected_digest or not release_path.is_dir():
                raise ValueError("active release or manifest digest is unavailable")
            actual_digest = digest_factory(release_path)
            if actual_digest != expected_digest:
                raise ValueError(f"digest mismatch (expected {expected_digest}, got {actual_digest})")
            findings.append(OperatorFinding(
                "install.release_digest", "installation", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                f"Active managed release content is verified: {release_id}.",
                observed={"release_id": release_id, "content_sha256": actual_digest, "path": str(release_path)},
            ))
        except Exception as exc:
            findings.append(OperatorFinding(
                "install.release_digest", "installation", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                "Active managed release content could not be verified.",
                observed={"release_id": release_id, "path": str(release_path), "error": str(exc)},
                guidance="Reinstall from a verified local kit or roll back to a retained release.",
            ))

        probe = version_probe or OperatorHealthService._probe_executable
        for label, executable, code in (("Taskwarrior", task_binary, "taskwarrior.identity"), ("Python", python_executable, "python.identity")):
            try:
                path = str(Path(executable).expanduser().resolve(strict=True))
                ok, version = probe(path)
                if not ok:
                    raise RuntimeError(version or "version probe failed")
                findings.append(OperatorFinding(
                    code, "installation", FindingSeverity.INFO,
                    FindingActionability.INFORMATIONAL,
                    f"{label} executable is usable: {path}.",
                    observed={"path": path, "version": version},
                ))
            except Exception as exc:
                findings.append(OperatorFinding(
                    code, "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING,
                    f"{label} executable identity could not be verified.",
                    observed={"path": str(executable), "error": str(exc)},
                    guidance=f"Verify the offline-kit {label} executable and retry deep Doctor.",
                ))
        return tuple(findings)

    @staticmethod
    def deep_resource_findings(
        timezone_name: str,
        resources: Mapping[str, object],
        *,
        timezone_factory: Callable[[str], object] | None,
    ) -> tuple[OperatorFinding, ...]:
        """Validate an explicit timezone and resource-path inventory read-only."""
        findings: list[OperatorFinding] = []
        try:
            if timezone_factory is None:
                raise RuntimeError("timezone support is unavailable")
            timezone_factory(str(timezone_name).strip())
            findings.append(OperatorFinding(
                "config.timezone.deep", "configuration", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                f"Configured timezone is available: {timezone_name}.",
                observed={"timezone": str(timezone_name)},
            ))
        except Exception as exc:
            findings.append(OperatorFinding(
                "config.timezone.deep", "configuration", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                f"Configured timezone is unavailable: {timezone_name}.",
                observed={"timezone": str(timezone_name), "error": str(exc)},
                guidance="Install timezone data or select an available IANA timezone, then retry deep Doctor.",
            ))
        for label, raw_path in resources.items():
            path = Path(str(raw_path)).expanduser()
            try:
                resolved = path.resolve(strict=True)
                if not (resolved.is_file() or resolved.is_dir()) or not os.access(str(resolved), os.R_OK):
                    raise OSError("resource is not readable")
                findings.append(OperatorFinding(
                    f"resource.{label}", "configuration", FindingSeverity.INFO,
                    FindingActionability.INFORMATIONAL,
                    f"Configured resource is readable: {label}.",
                    observed={"path": str(resolved), "kind": "directory" if resolved.is_dir() else "file"},
                ))
            except Exception as exc:
                findings.append(OperatorFinding(
                    f"resource.{label}", "configuration", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING,
                    f"Configured resource is unavailable: {label}.",
                    observed={"path": str(path), "error": str(exc)},
                    guidance=f"Create or correct the configured {label} resource, then retry deep Doctor.",
                ))
        return tuple(findings)

    @staticmethod
    def deep_local_state_findings(
        outbox_path: object,
        backup_root: object | None = None,
        *,
        quick_check: Callable[[Path], str] | None = None,
        backup_checker: Callable[[Path], bool] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> tuple[OperatorFinding, ...]:
        """Check outbox integrity and the newest optional backup generation."""
        findings: list[OperatorFinding] = []
        path = Path(str(outbox_path)).expanduser()
        try:
            checker = quick_check or OperatorHealthService._quick_check_sqlite
            result = checker(path)
            if result.lower() != "ok":
                raise RuntimeError(result)
            findings.append(OperatorFinding(
                "outbox.quick_check.deep", "lifecycle", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                "Lifecycle outbox integrity is verified.",
                observed={"path": str(path), "quick_check": result},
            ))
        except Exception as exc:
            findings.append(OperatorFinding(
                "outbox.quick_check.deep", "lifecycle", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                "Lifecycle outbox integrity could not be verified.",
                observed={"path": str(path), "error": str(exc)},
                guidance="Stop Nautical processes and restore or repair the outbox from a verified local backup.",
            ))
        if backup_root is None:
            return tuple(findings)
        root = Path(str(backup_root)).expanduser()
        try:
            generations = [item for item in root.iterdir() if item.is_dir() and not item.is_symlink()]
            if not generations:
                raise FileNotFoundError("no backup generations found")
            newest = max(generations, key=lambda item: (item.stat().st_mtime_ns, item.name))
            checker = backup_checker or OperatorHealthService._verify_backup_generation
            if not checker(newest):
                raise RuntimeError("manifest or artifact verification failed")
            manifest = json.loads((newest / "manifest.json").read_text(encoding="utf-8"))
            metadata = manifest.get("metadata") if isinstance(manifest, Mapping) else None
            if not isinstance(metadata, Mapping) or metadata.get("restore_tool_schema") != 1:
                raise RuntimeError("backup restore-tool schema is missing or unsupported")
            created_at = metadata.get("created_at")
            if isinstance(created_at, bool) or not isinstance(created_at, (int, float)):
                raise RuntimeError("backup creation timestamp is missing or invalid")
            age_seconds = max(0.0, float(clock()) - float(created_at))
            findings.append(OperatorFinding(
                "backup.newest.deep", "backup", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                "Newest configured backup generation is verified.",
                observed={"root": str(root), "generation": newest.name, "age_seconds": age_seconds, "restore_tool_schema": 1},
            ))
        except Exception as exc:
            findings.append(OperatorFinding(
                "backup.newest.deep", "backup", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                "Newest configured backup generation could not be verified.",
                observed={"root": str(root), "error": str(exc)},
                guidance="Create or select a verified local backup generation before relying on offline recovery.",
            ))
        return tuple(findings)

    @staticmethod
    def deep_clock_findings(
        runtime: Mapping[str, Any],
        backup_root: object | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> tuple[OperatorFinding, ...]:
        """Report only unambiguous clock-before-local-evidence anomalies."""
        now = float(clock())
        findings: list[OperatorFinding] = []
        release_manifest = runtime.get("manifest") if isinstance(runtime.get("manifest"), Mapping) else {}
        release_created = release_manifest.get("created_at")
        if isinstance(release_created, (int, float)) and not isinstance(release_created, bool) and now < float(release_created):
            findings.append(OperatorFinding(
                "time.before_release", "installation", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                "System clock predates the active Nautical release evidence.",
                observed={"now": now, "release_created_at": float(release_created)},
                guidance="Correct the system clock before scheduling or mutating recurrence state.",
            ))
        if backup_root is None:
            return tuple(findings)
        try:
            root = Path(str(backup_root)).expanduser()
            generations = [item for item in root.iterdir() if item.is_dir() and not item.is_symlink()]
            if not generations:
                return tuple(findings)
            newest = max(generations, key=lambda item: (item.stat().st_mtime_ns, item.name))
            manifest = json.loads((newest / "manifest.json").read_text(encoding="utf-8"))
            metadata = manifest.get("metadata") if isinstance(manifest, Mapping) else {}
            backup_created = metadata.get("created_at") if isinstance(metadata, Mapping) else None
            if isinstance(backup_created, (int, float)) and not isinstance(backup_created, bool) and now < float(backup_created):
                findings.append(OperatorFinding(
                    "time.before_backup", "backup", FindingSeverity.WARNING,
                    FindingActionability.ACTIONABLE,
                    "System clock predates the newest verified backup evidence.",
                    observed={"now": now, "backup_created_at": float(backup_created), "generation": newest.name},
                    guidance="Correct the system clock before relying on scheduling or recovery timestamps.",
                ))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return tuple(findings)

    @staticmethod
    def _quick_check_sqlite(path: Path) -> str:
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            return str(connection.execute("PRAGMA quick_check").fetchone()[0])
        finally:
            connection.close()

    @staticmethod
    def _verify_backup_generation(path: Path) -> bool:
        from .backup_service import verify_manifest
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return verify_manifest(path, manifest).status == "verified"

    @staticmethod
    def _probe_executable(path: str) -> tuple[bool, str]:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True, check=False, timeout=5)
        except (OSError, subprocess.SubprocessError) as exc:
            return False, str(exc)
        output = (result.stdout or result.stderr).strip()
        return result.returncode == 0, output

    @staticmethod
    def diagnose_configuration(request: ConfigurationDiagnosisRequest) -> OperatorHealthReport:
        """Evaluate one typed configuration diagnosis request."""
        findings = OperatorHealthService.configuration_findings(
            request.data,
            effective=request.effective,
            config_dir=request.config_dir,
            timezone_factory=request.timezone_factory,
            seasonal_events=request.seasonal_events,
            astronomy_preflight=request.astronomy_preflight,
            source_path=request.source_path,
            drift_loader=request.drift_loader,
            dependency_available=request.dependency_available,
            python_executable=request.python_executable,
            rich_factory=request.rich_factory,
        )
        return OperatorHealthService.report(findings)

    @staticmethod
    def configuration_findings(
        data: dict[str, Any],
        *,
        effective: dict[str, Any],
        config_dir: object,
        timezone_factory: Callable[[str], object] | None,
        seasonal_events: Callable[[int], dict[str, Any]],
        astronomy_preflight: Callable[[object], dict[str, Any]],
        source_path: str,
        drift_loader: Callable[[], dict[str, Any]],
        dependency_available: Callable[[str], bool],
        python_executable: str,
        rich_factory: Callable[[str], object],
    ) -> tuple[OperatorFinding, ...]:
        """Build all read-only configuration findings through one request."""
        findings: list[OperatorFinding] = []
        findings.extend(OperatorHealthService.configuration_schema_findings(data))
        findings.extend(OperatorHealthService.uda_alias_findings(data))
        findings.extend(OperatorHealthService.timezone_findings(data, timezone_factory))
        findings.extend(OperatorHealthService.season_findings(data, effective, timezone_factory, seasonal_events))
        timezone_name = str(data.get("tz", effective.get("tz", "UTC")) or "UTC")
        findings.extend(OperatorHealthService.astronomy_findings(
            data.get("astronomy") if data else effective.get("astronomy"),
            effective_timezone=timezone_name,
            source_hint=source_path,
            preflight=astronomy_preflight,
        ))
        findings.extend(OperatorHealthService.configuration_drift_findings(source_path, drift_loader))
        findings.extend(OperatorHealthService.navigator_dependency_findings(
            data, dependency_available, python_executable=python_executable,
        ))
        findings.extend(OperatorHealthService.panel_findings(data, rich_factory))
        findings.extend(OperatorHealthService.directory_findings(data, config_dir))
        return tuple(findings)

    @staticmethod
    def configuration_schema_findings(data: dict[str, Any]) -> tuple[OperatorFinding, ...]:
        """Validate configuration schema and return typed findings."""
        findings: list[OperatorFinding] = []
        for issue in validate_config(data, skip_keys={"live_panel_duration_ms"}):
            kind = str(issue["kind"])
            key = str(issue["key"])
            details = {name: value for name, value in issue.items() if name not in {"kind", "message"}}
            if kind == "unknown":
                message, guidance = f"Unknown Nautical config key: {key}.", f"Remove or correct '{key}'."
            elif kind == "deprecated":
                message = f"Retired Nautical config key '{key}': {issue['message']}"
                guidance = f"Remove '{key}' from the config."
            elif kind == "type":
                message = f"Config key '{key}' has the wrong type; expected {issue['expected']} and will use {issue['effective']!r}."
                guidance = f"Set '{key}' to a TOML {issue['expected']} value."
            elif kind == "range":
                message = f"Config key '{key}' is outside its supported range; {issue['effective']!r} will be used."
                bounds = [bound for bound in (
                    f"at least {issue['min']}" if issue.get("min") is not None else "",
                    f"at most {issue['max']}" if issue.get("max") is not None else "",
                ) if bound]
                guidance = f"Set '{key}' to {' and '.join(bounds)}."
            else:
                message = f"Config key '{key}' has unsupported value {issue['configured']!r}; {issue['effective']!r} will be used."
                guidance = f"Set '{key}' to one of: {', '.join(issue['choices'])}."
            findings.append(OperatorFinding(
                code=f"config.schema.{kind}", domain="configuration",
                severity=FindingSeverity.WARNING, actionability=FindingActionability.ACTIONABLE,
                message=message, observed=details, guidance=guidance,
            ))
        return tuple(findings)

    @staticmethod
    def uda_registration_findings(
        required: dict[str, str],
        read_type: Callable[[str], tuple[bool, str]],
    ) -> tuple[OperatorFinding, ...]:
        """Validate Taskwarrior UDA registration through a typed read callback."""
        result: list[OperatorFinding] = []
        valid = True
        for name, expected in required.items():
            ok, actual = read_type(name)
            if not ok or not actual:
                valid = False
                result.append(OperatorFinding(
                    f"uda.{name}.missing", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING,
                    f"Required UDA '{name}' is not defined.",
                    guidance="Include Nautical's uda.conf from your Taskwarrior configuration.",
                ))
            elif actual.lower() != expected:
                valid = False
                result.append(OperatorFinding(
                    f"uda.{name}.type", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING,
                    f"UDA '{name}' has type '{actual}', expected '{expected}'.",
                    guidance=f"Set uda.{name}.type={expected}.",
                ))
        if valid:
            result.append(OperatorFinding(
                "uda.registration", "installation", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                f"All {len(required)} required Nautical UDAs are registered.",
            ))
        return tuple(result)

    @staticmethod
    def taskdata_findings(
        task_available: bool,
        taskdata: object,
        *,
        task_error: str = "",
    ) -> tuple[OperatorFinding, ...]:
        """Classify Taskwarrior command and data-directory health."""
        from pathlib import Path
        import os
        result: list[OperatorFinding] = []
        if not task_available:
            result.append(OperatorFinding(
                "taskwarrior.unavailable", "installation", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                "Taskwarrior could not be executed.",
                observed={"error": task_error},
                guidance="Install Taskwarrior or pass --task-bin.",
            ))
        else:
            result.append(OperatorFinding(
                "taskwarrior.version", "installation", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                "Taskwarrior command is available.",
            ))
        path = Path(str(taskdata)).expanduser()
        if not path.exists():
            result.append(OperatorFinding(
                "taskdata.missing", "installation", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                f"Taskwarrior data directory does not exist: {path}",
                guidance="Check TASKDATA, TASKRC, or pass --taskdata.",
            ))
        elif not path.is_dir():
            result.append(OperatorFinding(
                "taskdata.invalid", "installation", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                f"Taskwarrior data path is not a directory: {path}",
            ))
        else:
            writable = os.access(str(path), os.R_OK | os.W_OK | os.X_OK)
            result.append(OperatorFinding(
                "taskdata.access", "installation",
                FindingSeverity.INFO if writable else FindingSeverity.ERROR,
                FindingActionability.INFORMATIONAL if writable else FindingActionability.BLOCKING,
                f"Taskwarrior data directory is {'accessible' if writable else 'not fully accessible'}: {path}",
                guidance="" if writable else "Correct ownership and directory permissions.",
            ))
        return tuple(result)

    @staticmethod
    def hook_installation_findings(
        hooks_dir: object,
        events: Iterable[str],
        candidates_for: Callable[[Any, str], Iterable[Any]],
        inspect_runtime: Callable[[Any, str, dict[str, str]], tuple[dict[str, Any] | None, str, dict[str, Any]]],
        env: dict[str, str],
    ) -> tuple[tuple[OperatorFinding, ...], dict[str, dict[str, Any]]]:
        """Validate hook layout and provenance through injected installers."""
        import os
        result: list[OperatorFinding] = []
        validated: dict[str, dict[str, Any]] = {}
        for event in events:
            candidates = list(candidates_for(hooks_dir, event))
            active = [path for path in candidates if os.access(str(path), os.X_OK)]
            if not candidates:
                result.append(OperatorFinding(
                    f"hook.{event}.missing", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING, f"No Nautical {event} hook was found in {hooks_dir}.",
                    guidance="Install the Nautical hook files and make them executable.",
                ))
                continue
            if not active:
                result.append(OperatorFinding(
                    f"hook.{event}.inactive", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING, f"Nautical {event} hook is not executable: {candidates[0]}",
                    observed={"hooks": [str(path) for path in candidates]},
                    guidance=f"Run chmod +x {candidates[0]}",
                ))
                continue
            if len(active) > 1:
                result.append(OperatorFinding(
                    f"hook.{event}.duplicate", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING,
                    f"{len(active)} active Nautical {event} hooks were found; Taskwarrior may run all of them.",
                    observed={"hooks": [str(path) for path in active]},
                    guidance="Keep exactly one executable Nautical hook for this event.",
                ))
                continue
            record, error, details = inspect_runtime(active[0], event, env)
            if record is None:
                result.append(OperatorFinding(
                    f"hook.{event}.incompatible", "installation", FindingSeverity.ERROR,
                    FindingActionability.BLOCKING, error,
                    observed=details,
                    guidance="Install the matching Nautical wrappers and nautical_core from the same release.",
                ))
                continue
            validated[event] = record
            result.append(OperatorFinding(
                f"hook.{event}", "installation", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                f"{event} hook and core are compatible: {active[0]}", observed=details,
            ))
        return tuple(result), validated

    @staticmethod
    def uda_alias_findings(data: dict[str, Any]) -> tuple[OperatorFinding, ...]:
        """Describe the opt-in description alias configuration."""
        enabled = data.get("enable_uda_aliases") is True
        state = "enabled" if enabled else "disabled"
        return (OperatorFinding(
            code="config.uda_aliases",
            domain="configuration",
            severity=FindingSeverity.INFO,
            actionability=FindingActionability.INFORMATIONAL,
            message=f"Description UDA aliases are {state}.",
            observed={
                "enabled": enabled,
                "aliases": dict(ALIAS_TO_FIELD),
                "clear_syntax": "alias:",
            },
        ),)

    @staticmethod
    def timezone_findings(
        data: dict[str, Any], zoneinfo_factory: Callable[[str], object] | None,
    ) -> tuple[OperatorFinding, ...]:
        """Validate the configured IANA timezone through an injected resolver."""
        raw_tz = data.get("tz")
        tz_name = str(raw_tz or "UTC").strip() or "UTC"
        if not str(raw_tz or "").strip():
            return (OperatorFinding(
                "config.timezone.missing", "configuration", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                "No explicit Nautical timezone is configured; UTC fallback is active.",
                observed={"tz": tz_name},
                guidance="Run Nautical install on a fresh target or set tz to an explicit IANA timezone in config-nautical.toml.",
            ),)
        if zoneinfo_factory is None:
            return (OperatorFinding(
                "config.timezone.unavailable", "configuration", FindingSeverity.WARNING,
                FindingActionability.RETRYABLE,
                "Python zoneinfo support is unavailable; Nautical will use UTC fallback.",
                observed={"tz": tz_name},
                guidance="Use Python 3.9+ with zoneinfo support, or install timezone support for your Python build.",
            ),)
        try:
            zoneinfo_factory(tz_name)
        except Exception as exc:
            return (OperatorFinding(
                "config.timezone.invalid", "configuration", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                f"Nautical timezone '{tz_name}' is not available; hooks will use UTC fallback.",
                observed={"tz": tz_name, "error": str(exc)},
                guidance="Install system tzdata, or on Termux/Python environments run: python3 -m pip install tzdata.",
            ),)
        return (OperatorFinding(
            "config.timezone", "configuration", FindingSeverity.INFO,
            FindingActionability.INFORMATIONAL,
            f"Nautical timezone is available: {tz_name}", observed={"tz": tz_name},
        ),)

    @staticmethod
    def panel_findings(
        data: dict[str, Any], rich_factory: Callable[[str], object],
    ) -> tuple[OperatorFinding, ...]:
        """Validate live-panel settings and optional Rich availability."""
        mode = str(data.get("panel_mode") or "rich").strip().lower() or "rich"
        spec = CONFIG_SPECS["live_panel_duration_ms"]
        default = int(spec["default"])
        minimum, maximum = int(spec["min"]), int(spec["max"])
        raw = data.get("live_panel_duration_ms", default)
        valid = True
        try:
            configured = int(str(raw).strip())
        except Exception:
            configured, valid = default, False
        effective = max(minimum, min(maximum, configured))
        result: list[OperatorFinding] = []
        if not valid:
            result.append(OperatorFinding(
                "config.panel.duration.invalid", "configuration", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                f"live_panel_duration_ms is invalid ({raw!r}); the effective duration is {default} ms.",
                observed={"configured_duration_ms": raw, "effective_duration_ms": default},
                guidance=f"Set live_panel_duration_ms to an integer from {minimum} to {maximum}.",
            ))
        elif effective != configured:
            result.append(OperatorFinding(
                "config.panel.duration.clamped", "configuration", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                f"live_panel_duration_ms is {configured}; Nautical clamps it to {effective} ms.",
                observed={"configured_duration_ms": configured, "effective_duration_ms": effective},
                guidance=f"Set live_panel_duration_ms to an integer from {minimum} to {maximum}.",
            ))
        if mode != "live":
            return tuple(result)
        try:
            rich_available = rich_factory("rich") is not None
        except Exception:
            rich_available = False
        motion = "disabled" if effective == 0 else f"{effective} ms"
        state = "available" if rich_available else "unavailable"
        result.append(OperatorFinding(
            "config.panel.live", "configuration", FindingSeverity.INFO,
            FindingActionability.INFORMATIONAL,
            f"Live panels use {motion} effective duration; Rich is {state}; non-TTY output uses static fallback.",
            observed={
                "configured_duration_ms": raw if not valid else configured,
                "effective_duration_ms": effective,
                "rich_available": rich_available,
                "non_tty_fallback": "static",
            },
        ))
        if not rich_available:
            result.append(OperatorFinding(
                "config.panel.rich_missing", "configuration", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                "panel_mode is live, but Rich is not installed; panels will use the static fallback.",
                guidance="Run python3 -m pip install rich.",
            ))
        return tuple(result)

    @staticmethod
    def directory_findings(data: dict[str, Any], config_dir: object) -> tuple[OperatorFinding, ...]:
        """Validate configured file-provider directories without performing writes."""
        from pathlib import Path
        import os
        import os
        base = Path(str(config_dir)).expanduser()
        result: list[OperatorFinding] = []
        for key in ("anchor_file_dir", "omit_file_dir"):
            raw = str(data.get(key) or "").strip()
            if not raw:
                continue
            path = Path(raw).expanduser()
            if not path.is_absolute():
                path = base / path
            resolved = path.resolve()
            valid = resolved.is_dir() and os.access(str(resolved), os.R_OK | os.X_OK)
            result.append(OperatorFinding(
                f"config.{key}", "configuration",
                FindingSeverity.INFO if valid else FindingSeverity.ERROR,
                FindingActionability.INFORMATIONAL if valid else FindingActionability.BLOCKING,
                f"{key} {'is accessible' if valid else 'is not accessible'}: {resolved}",
                guidance="" if valid else f"Create or correct the configured {key} directory.",
            ))
        return tuple(result)

    @staticmethod
    def astronomy_findings(
        config: object,
        *,
        effective_timezone: object,
        source_hint: str,
        preflight: Callable[[object], dict[str, Any]],
    ) -> tuple[OperatorFinding, ...]:
        """Project astronomy-provider preflight into typed findings."""
        result = preflight(config)
        status = str(result.get("status") or "error")
        if status == "not_configured":
            return (OperatorFinding(
                "astronomy.not_configured", "configuration", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                "Astronomy is not configured; astronomical anchor times are disabled.",
                guidance="Define [astronomy] locations only if using sunrise, sunset, moonrise, or moonset anchors.",
            ),)
        severity = FindingSeverity.INFO if status == "ok" else FindingSeverity.WARNING if status == "warning" else FindingSeverity.ERROR
        actionability = FindingActionability.INFORMATIONAL if status == "ok" else FindingActionability.ACTIONABLE
        details = {
            key: _json_safe(value)
            for key, value in result.items()
            if key not in {"status", "message"}
        }
        timezone_value = getattr(effective_timezone, "key", None) or str(effective_timezone)
        details.update({"config_source": source_hint, "effective_timezone": timezone_value})
        return (OperatorFinding(
            "astronomy.preflight", "configuration", severity, actionability,
            "Astronomy provider and location profile are usable." if status == "ok" else str(result.get("message") or "Astronomy preflight failed."),
            observed=details,
            guidance=("Install astral in the active interpreter and verify the selected profile." if status == "error" else "Review the astronomy location and event availability before scheduling." if status == "warning" else ""),
        ),)

    @staticmethod
    def configuration_drift_findings(
        source_path: str,
        drift_loader: Callable[[], dict[str, Any]],
    ) -> tuple[OperatorFinding, ...]:
        """Project runtime/file configuration drift into a typed finding."""
        from pathlib import Path
        drift = drift_loader()
        loaded_source = str(drift.get("source") or "")
        expected_source = str(Path(source_path).expanduser().resolve()) if source_path else "defaults"
        if loaded_source != expected_source:
            return ()
        if drift.get("changed"):
            return (OperatorFinding(
                "config.drift", "configuration", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                "The loaded Nautical configuration differs from the current file.",
                observed={
                    "source": loaded_source,
                    "loaded_fingerprint": drift.get("loaded_fingerprint", ""),
                    "current_fingerprint": drift.get("current_fingerprint", ""),
                },
                guidance="Restart Navigator; Taskwarrior hooks will use the new configuration on their next invocation.",
            ),)
        return (OperatorFinding(
            "config.drift", "configuration", FindingSeverity.INFO,
            FindingActionability.INFORMATIONAL,
            "Loaded Nautical configuration matches the current file.",
            observed={"source": loaded_source, "fingerprint": drift.get("current_fingerprint", "")},
        ),)

    @staticmethod
    def navigator_dependency_findings(
        data: dict[str, Any],
        available: Callable[[str], bool],
        *,
        python_executable: str,
    ) -> tuple[OperatorFinding, ...]:
        """Report optional Navigator and astronomy runtime dependencies."""
        required = ("rich", "prompt_toolkit", "dateutil")
        missing = [name for name in required if not available(name)]
        result: list[OperatorFinding] = []
        if missing:
            result.append(OperatorFinding(
                "navigator.dependencies", "installation", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                "Navigator dependencies are incomplete: " + ", ".join(missing) + ".",
                observed={"missing": missing, "python": python_executable},
                guidance="Run python3 -m pip install -r requirements.txt.",
            ))
        else:
            result.append(OperatorFinding(
                "navigator.dependencies", "installation", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                "Navigator dependencies are available.",
                observed={"python": python_executable},
            ))
        astronomy = data.get("astronomy")
        if isinstance(astronomy, dict) and astronomy.get("locations") and not available("astral"):
            result.append(OperatorFinding(
                "navigator.astronomy_dependency", "installation", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                "Astronomy locations are configured, but Astral is not installed.",
                observed={"python": python_executable},
                guidance="Run python3 -m pip install -r requirements.txt.",
            ))
        return tuple(result)

    @staticmethod
    def lifecycle_outbox_findings(payload: dict[str, Any]) -> tuple[OperatorFinding, ...]:
        """Project a queue-status payload into stable typed findings."""
        outbox_value = payload.get("outbox")
        outbox: dict[str, Any] = dict(outbox_value) if isinstance(outbox_value, dict) else {}
        states_value = outbox.get("states")
        states: dict[str, Any] = dict(states_value) if isinstance(states_value, dict) else {}
        result: list[OperatorFinding] = []
        quarantined = int(states.get("quarantined") or 0)
        if quarantined:
            result.append(OperatorFinding(
                "outbox.poison_rows", "outbox", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                f"{quarantined} malformed lifecycle intent{'s' if quarantined != 1 else ''} quarantined.",
                observed={"count": quarantined, "sample": outbox.get("sample") or []},
                guidance="Inspect nautical queue-status and resolve the quarantined lifecycle intents.",
            ))
        schema_value = outbox.get("schema")
        schema: dict[str, Any] = dict(schema_value) if isinstance(schema_value, dict) else {}
        schema_status = str(schema.get("status") or "absent")
        result.append(OperatorFinding(
            "outbox.schema", "outbox",
            FindingSeverity.ERROR if schema_status == "error" else FindingSeverity.INFO,
            FindingActionability.BLOCKING if schema_status == "error" else FindingActionability.INFORMATIONAL,
            "Lifecycle outbox schema is incompatible with this Nautical runtime." if schema_status == "error" else (
                f"Lifecycle outbox schema v{schema.get('version')} is compatible." if schema_status == "ok"
                else "Lifecycle outbox has not been created yet."
            ), observed=schema,
            guidance=("Preserve the database, then upgrade Nautical or restore a compatible lifecycle outbox." if schema_status == "error" else ""),
        ))
        issues = payload.get("issues") or []
        outbox_status = str(payload.get("status") or "ok")
        result.append(OperatorFinding(
            "outbox.state", "outbox",
            FindingSeverity.ERROR if outbox_status == "error" else FindingSeverity.WARNING if issues else FindingSeverity.INFO,
            FindingActionability.ACTIONABLE if issues or outbox_status == "error" else FindingActionability.INFORMATIONAL,
            "Lifecycle outbox has findings." if issues else "Lifecycle outbox is clean.",
            observed={"issues": issues} if issues else {},
            guidance="Run nautical queue-status for lifecycle outbox details." if issues else "",
        ))
        retention_value = outbox.get("retention")
        retention: dict[str, Any] = dict(retention_value) if isinstance(retention_value, dict) else {}
        eligible = int(retention.get("eligible") or 0)
        if eligible:
            result.append(OperatorFinding(
                "outbox.retention", "outbox", FindingSeverity.WARNING, FindingActionability.ACTIONABLE,
                f"{eligible} acknowledged lifecycle intent{'s' if eligible != 1 else ''} exceed the retention policy.",
                observed=retention,
                guidance="Run nautical queue-status --prune-acknowledged to remove only expired acknowledgements.",
            ))
        elif outbox.get("exists"):
            result.append(OperatorFinding(
                "outbox.retention", "outbox", FindingSeverity.INFO, FindingActionability.INFORMATIONAL,
                "Lifecycle outbox retention is within policy.", observed=retention,
            ))
        return tuple(result)

    @staticmethod
    def obsolete_queue_findings(
        taskdata: object,
        names: Iterable[str],
    ) -> tuple[OperatorFinding, ...]:
        """Report retired queue artifacts without reading or migrating them."""
        from pathlib import Path
        root = Path(str(taskdata)).expanduser()
        paths = sorted({str(root / name) for base in (root, root / ".nautical-state") for name in names if (base / name).exists()})
        if not paths:
            return ()
        return (OperatorFinding(
            "outbox.obsolete_state", "outbox", FindingSeverity.WARNING,
            FindingActionability.ACTIONABLE,
            "Retired Nautical queue state was found; it is not used by this runtime.",
            observed={"paths": paths},
            guidance=("Back up any required records, then quarantine or remove the listed files; "
                      "the lifecycle outbox is the only supported work store."),
        ),)

    @staticmethod
    def runtime_findings(
        status: dict[str, Any], runtime_root: object,
        hook_runtimes: dict[str, dict[str, Any]] | None = None,
    ) -> tuple[OperatorFinding, ...]:
        """Project managed-runtime status into typed installation findings."""
        from pathlib import Path
        result: list[OperatorFinding] = []
        abandoned = list(status.get("abandoned") or [])
        if abandoned:
            result.append(OperatorFinding(
                "install.runtime_abandoned", "installation", FindingSeverity.WARNING,
                FindingActionability.ACTIONABLE,
                f"{len(abandoned)} abandoned install transaction path(s) were found.",
                observed={"paths": abandoned},
                guidance="Confirm no install is running, then remove the listed .staging/.rollback paths.",
            ))
        if not status.get("managed"):
            return tuple(result)
        errors = list(status.get("errors") or [])
        if errors:
            result.append(OperatorFinding(
                "install.runtime", "installation", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                "Managed Nautical runtime is incomplete or has a broken active pointer.",
                observed={"errors": errors, **status},
                guidance="Reinstall from a valid local release; the installer will preserve the previous active release.",
            ))
            return tuple(result)
        release_id = str(status.get("active_release") or "unknown")
        result.append(OperatorFinding(
            "install.runtime", "installation", FindingSeverity.INFO,
            FindingActionability.INFORMATIONAL,
            f"Managed Nautical runtime is active: {release_id}.", observed=status,
        ))
        manifest_value = status.get("manifest")
        manifest: dict[str, Any] = manifest_value if isinstance(manifest_value, dict) else {}
        current_root = Path(str(status.get("runtime_root") or runtime_root)).expanduser() / "current"
        errors = []
        for event, record in (hook_runtimes or {}).items():
            implementation = record.get("implementation")
            if not implementation:
                errors.append(f"{event} implementation path is missing")
                continue
            try:
                Path(str(implementation)).resolve().relative_to(current_root.resolve())
            except Exception:
                errors.append(f"{event} implementation is outside the active release")
        evidence = {
            "release_id": release_id,
            "source": manifest.get("source", ""),
            "content_sha256": manifest.get("content_sha256", ""),
            "created_at": manifest.get("created_at", ""),
            "hook_impl_api": manifest.get("hook_impl_api", {}),
            "errors": errors,
        }
        result.append(OperatorFinding(
            "install.provenance", "installation",
            FindingSeverity.ERROR if errors else FindingSeverity.INFO,
            FindingActionability.BLOCKING if errors else FindingActionability.INFORMATIONAL,
            "Active hooks do not share the managed Nautical release." if errors else "Active hooks resolve to one managed Nautical release.",
            observed=evidence,
            guidance="Reinstall Nautical so wrappers and nautical_core come from the same release." if errors else "",
        ))
        return tuple(result)

    @staticmethod
    def diagnose_runtime(
        runtime_loader: Callable[[], dict[str, Any]],
        runtime_root: object,
        hook_runtimes: dict[str, dict[str, Any]] | None = None,
    ) -> OperatorHealthReport:
        """Acquire managed-runtime state and classify probe failures uniformly."""
        try:
            status = runtime_loader()
        except Exception as exc:
            status = {"managed": True, "errors": [str(exc)]}
        return OperatorHealthService.report(
            OperatorHealthService.runtime_findings(status, runtime_root, hook_runtimes)
        )

    @staticmethod
    def diagnose_taskwarrior(request: TaskwarriorDiagnosisRequest) -> tuple[OperatorHealthReport, object]:
        """Evaluate Taskwarrior availability and resolve its hooks directory."""
        from pathlib import Path
        import os
        available, error = request.probe()
        findings = list(OperatorHealthService.taskdata_findings(
            available, request.taskdata, task_error=error,
        ))
        configured = request.hooks_location().strip()
        hooks_dir = Path(configured).expanduser().resolve() if configured else Path(str(request.default_hooks_dir)).expanduser().resolve()
        return OperatorHealthService.report(findings), hooks_dir

    @staticmethod
    def season_findings(
        data: dict[str, Any], effective: dict[str, Any],
        zoneinfo_factory: Callable[[str], object] | None,
        events_provider: Callable[[int], dict[str, Any]],
        *, year: int | None = None,
    ) -> tuple[OperatorFinding, ...]:
        """Validate seasonal settings without changing effective-config semantics."""
        mode = str((data or {}).get("season_mode", effective.get("season_mode", "fixed")) or "fixed").strip().lower()
        hemisphere = str((data or {}).get("season_hemisphere", effective.get("season_hemisphere", "north")) or "north").strip().lower()
        timezone_name = str((data or {}).get("tz", effective.get("tz", "UTC")) or "UTC").strip() or "UTC"
        if mode not in {"fixed", "astronomical"}:
            return (OperatorFinding(
                "config.season_mode.invalid", "configuration", FindingSeverity.ERROR,
                FindingActionability.BLOCKING, f"Unsupported seasonal boundary backend: {mode!r}.",
                observed={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name},
                guidance="Set season_mode to 'fixed' or 'astronomical'.",
            ),)
        if mode == "fixed":
            return (OperatorFinding(
                "config.season_mode", "configuration", FindingSeverity.INFO,
                FindingActionability.INFORMATIONAL,
                f"Seasonal boundaries use the fixed backend ({hemisphere} hemisphere).",
                observed={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name},
            ),)
        event_year = date.today().year if year is None else year
        try:
            if zoneinfo_factory is None:
                raise RuntimeError("zoneinfo support is unavailable")
            events = events_provider(event_year)
            local_events = {
                name: event.astimezone(zoneinfo_factory(timezone_name)).date().isoformat()
                for name, event in events.items()
            }
        except Exception as exc:
            return (OperatorFinding(
                "config.season_mode.astronomical_invalid", "configuration", FindingSeverity.ERROR,
                FindingActionability.BLOCKING,
                f"Astronomical seasonal boundaries are unavailable for {event_year}: {exc}",
                observed={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name},
                guidance="Verify timezone data and use a supported season year/backend, then rerun doctor.",
            ),)
        return (OperatorFinding(
            "config.season_mode", "configuration", FindingSeverity.INFO,
            FindingActionability.INFORMATIONAL,
            f"Seasonal boundaries use astronomical transitions ({hemisphere} hemisphere, {timezone_name}).",
            observed={"mode": mode, "hemisphere": hemisphere, "timezone": timezone_name, "events": local_events},
        ),)


__all__ = ["OperatorHealthReport", "OperatorHealthService"]
