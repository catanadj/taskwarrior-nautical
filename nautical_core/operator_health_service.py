"""Typed aggregation boundary for operator health observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

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


class OperatorHealthService:
    """Aggregate health observations without performing I/O or mutations."""

    @staticmethod
    def report(findings: Iterable[OperatorFinding]) -> OperatorHealthReport:
        return OperatorHealthReport(tuple(findings))

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


__all__ = ["OperatorHealthReport", "OperatorHealthService"]
