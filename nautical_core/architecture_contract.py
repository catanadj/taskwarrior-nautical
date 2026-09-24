"""Static dependency-direction contract for the Nautical core package.

This module deliberately operates on source trees and never imports the
candidate package.  Deployment and CI can therefore validate a staged or
partially installed tree without triggering hook initialisation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DOMAIN = "domain/models"
RECURRENCE = "pure recurrence"
APPLICATION = "application services"
INTEGRATION = "integration adapters"
PRESENTATION = "presentation"
ENTRYPOINT = "hook/tool entry points"
COMPATIBILITY = "compatibility"

LAYERS = (DOMAIN, RECURRENCE, APPLICATION, INTEGRATION, PRESENTATION, ENTRYPOINT, COMPATIBILITY)

# Explicit ownership is preferable to a filename heuristic for the modules
# that form the package's architectural spine.  The fallback keeps the map
# complete as new domain modules are added.
_RECURRENCE_PREFIXES = (
    "anchor_", "astronom", "business_calendar", "cached_expansion",
    "compiled_schedule", "evaluation_session", "expansion_", "file_backed_dates",
    "file_source_expr", "monthly_", "moon_", "natural_language", "nth_monthly",
    "occurrence_", "omit_", "parser_", "position_selection", "precompute",
    "recurrence_", "schedule_", "scheduler_", "season_", "strict_validation",
    "time_", "timeutil", "tokenutil", "yearly_", "year_tokens",
)
_APPLICATION_NAMES = {
    "chain_generation", "chain_integrity_application", "chain_integrity_context",
    "chain_integrity_engine", "chain_integrity_lifecycle", "chain_integrity_recovery",
    "chain_invariants", "chain_repair_planner", "chain_snapshot", "doctor_report",
    "hook_engine", "hook_workflow_context", "integrity_audit_service",
    "integrity_operator_owner", "integrity_query_service", "lifecycle_application",
    "lifecycle_operator_owner", "lifecycle_planner", "lifecycle_read_service",
    "lifecycle_reconciliation", "modify_", "operator_application", "operator_control_plane",
    "operator_domain_planner", "operator_domain_plans", "operator_inspectors",
    "operator_snapshot", "operator_snapshot_provider", "query_service",
    "queue_status_service", "reconcile_operator_service", "reconcile_snapshot_service",
}
_INTEGRATION_PREFIXES = (
    "integration_", "taskwarrior", "task_command", "task_read_", "task_set_",
    "task_codec", "task_changes", "cache_", "backup_", "restore_", "install_",
    "lifecycle_outbox", "exit_probe", "runtime_command",
)
_PRESENTATION_PREFIXES = (
    "feedback_renderer", "calendar_feedback", "panel_", "operator_presentation", "exit_presentation",
    "query_report", "reconcile_report", "integrity_report", "installation_report", "ui",
)
_ENTRYPOINT_DIRS = {"hooks", "tools"}
_COMPATIBILITY_NAMES = {
    "__init__", "compat_api", "api_bindings", "parser_api", "scheduler_api",
    "cache_api", "time_api", "token_api", "quarter_api", "acf_api", "expansion_api",
    "business_calendar_api", "hint_builder_api", "linting_api", "configuration_facade", "cache_facade", "timezone_facade",
}
_INTEGRATION_NAMES = {"hook_context", "hook_runtime", "operator_health_service"}
_DOMAIN_NAMES = {"common", "hint_models", "task_models", "diagnostic_models"}


@dataclass(frozen=True)
class ImportReference:
    """One statically discovered import in a candidate source tree."""

    module: str
    line: int


@dataclass(frozen=True)
class ArchitectureViolation:
    """A dependency that violates the declared layer contract."""

    importing_file: str
    dependency: str
    layer: str
    rule: str
    line: int

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.importing_file,
            "dependency": self.dependency,
            "layer": self.layer,
            "rule": self.rule,
            "line": self.line,
            "message": (
                f"{self.importing_file}:{self.line} ({self.layer}) depends on "
                f"forbidden dependency {self.dependency}: {self.rule}"
            ),
        }


def module_layer(relative_path: str | Path) -> str:
    """Return the stable layer assigned to a package-relative Python path."""
    path = Path(relative_path)
    parts = path.parts
    if parts and parts[0] in _ENTRYPOINT_DIRS:
        return ENTRYPOINT
    stem = path.stem
    if stem in _COMPATIBILITY_NAMES:
        return COMPATIBILITY
    if stem in _INTEGRATION_NAMES:
        return INTEGRATION
    if stem in _DOMAIN_NAMES:
        return DOMAIN
    if parts and parts[0] == "tools":
        return ENTRYPOINT
    if stem in {"operator_models", "operator_findings", "diagnostic_models", "on_exit_models"}:
        return DOMAIN
    if stem in _PRESENTATION_PREFIXES or any(stem.startswith(p) for p in _PRESENTATION_PREFIXES):
        return PRESENTATION
    if stem in _APPLICATION_NAMES or any(stem.startswith(p) for p in _APPLICATION_NAMES if p.endswith("_")):
        return APPLICATION
    if stem in _INTEGRATION_PREFIXES or any(stem.startswith(p) for p in _INTEGRATION_PREFIXES):
        return INTEGRATION
    if stem in _RECURRENCE_PREFIXES or any(stem.startswith(p) for p in _RECURRENCE_PREFIXES):
        return RECURRENCE
    return DOMAIN


def module_layer_map(root: Path) -> dict[str, str]:
    """Return every package module and its assigned layer in sorted order."""
    package = root / "nautical_core"
    return {
        str(path.relative_to(root)): module_layer(path.relative_to(package))
        for path in sorted(package.rglob("*.py"))
    } if package.is_dir() else {}


def _imports(tree: ast.AST) -> Iterable[ImportReference]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield ImportReference(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield ImportReference(node.module, node.lineno)


def _is_forbidden(module: str, forbidden: tuple[str, ...]) -> bool:
    lowered = module.casefold()
    return any(
        module == prefix or module.startswith(prefix + ".") or prefix.casefold() in lowered
        for prefix in forbidden
    )


def _is_modify_composition_adapter(name: str) -> bool:
    return name.endswith(("_port_for", "_ports_for", "_services_for"))


def _hook_host_parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    arguments = (
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
    )
    return tuple(argument for argument in arguments if argument.arg == "host")


def _dynamic_hook_host_access(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Attribute | None:
    for candidate in ast.walk(node):
        if (
            isinstance(candidate, ast.Attribute)
            and isinstance(candidate.value, ast.Name)
            and candidate.value.id == "host"
            and candidate.attr in {"_module", "_read_query_get", "_READ_QUERY_MISSING", "core"}
        ):
            return candidate
    return None


def validate(root: Path) -> tuple[ArchitectureViolation, ...]:
    """Validate imports in ``root`` without importing any source module."""
    layers = module_layer_map(root)
    violations: list[ArchitectureViolation] = []
    forbidden_pure = (
        "nautical_core.hooks", "nautical_core.tools", "taskwarrior", "sqlite", "rich",
    )
    # The facade is a compatibility surface.  It may be consumed by tools,
    # hooks, and compatibility modules, but never by internal owners.
    facade_allowed = {ENTRYPOINT, COMPATIBILITY}

    for relative, layer in layers.items():
        path = root / relative
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        if path.name.startswith("modify_") and path.name.endswith("_effects.py"):
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                host_parameters = _hook_host_parameters(node)
                if host_parameters and not _is_modify_composition_adapter(node.name):
                    violations.append(ArchitectureViolation(
                        relative,
                        "hook-host",
                        layer,
                        (
                            f"modify effect operation {node.name} must consume explicit ports; "
                            "only named composition adapters may receive the hook host"
                        ),
                        host_parameters[0].lineno,
                    ))
                elif not _is_modify_composition_adapter(node.name):
                    dynamic_access = _dynamic_hook_host_access(node)
                    if dynamic_access is not None:
                        violations.append(ArchitectureViolation(
                            relative,
                            "hook-host",
                            layer,
                            (
                                f"modify effect operation {node.name} must not perform "
                                "dynamic hook-host lookup"
                            ),
                            dynamic_access.lineno,
                        ))
        for reference in _imports(tree):
            module = reference.module
            if layer in {DOMAIN, RECURRENCE} and _is_forbidden(module, forbidden_pure):
                violations.append(ArchitectureViolation(
                    relative, module, layer,
                    "domain and pure recurrence must remain independent of hooks, tools, Taskwarrior, SQLite, and Rich",
                    reference.line,
                ))
            if module == "nautical_core" and layer not in facade_allowed:
                violations.append(ArchitectureViolation(
                    relative, module, layer,
                    "internal production modules may not import the root facade",
                    reference.line,
                ))
            if module == "nautical_core.compat_api" and layer not in facade_allowed:
                violations.append(ArchitectureViolation(
                    relative, module, layer,
                    "primary production modules may not depend on the compatibility implementation",
                    reference.line,
                ))
    return tuple(sorted(violations, key=lambda item: (item.importing_file, item.line, item.dependency)))


def check(root: Path) -> list[dict[str, object]]:
    """Return deployment-sanity result records for the architecture contract."""
    violations = validate(root)
    return [{
        "kind": "architecture",
        "name": "dependency-direction",
        "ok": not violations,
        "message": "ok" if not violations else "; ".join(str(v.as_dict()["message"]) for v in violations),
        "layer_map": module_layer_map(root),
        "violations": [v.as_dict() for v in violations],
    }]


__all__ = (
    "APPLICATION", "COMPATIBILITY", "DOMAIN", "ENTRYPOINT", "INTEGRATION",
    "LAYERS", "PRESENTATION", "RECURRENCE", "ArchitectureViolation", "check",
    "module_layer", "module_layer_map", "validate",
)
