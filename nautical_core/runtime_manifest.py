"""Files and lazy modules required by each Nautical lifecycle hook."""

from __future__ import annotations


_SHARED_HOOK_MODULES = ("hook_runtime", "integration_context")

_INTEGRATION_FILES = (
    "integration_models.py",
    "task_read_repository.py",
    "taskwarrior_client.py",
    "taskwarrior_mutations.py",
    "taskwarrior_uow.py",
)


HOOK_LAZY_MODULES: dict[str, tuple[str, ...]] = {
    "on-add": (
        *_SHARED_HOOK_MODULES,
        "hook_support",
        "add_formatting",
        "add_validation",
        "astronomy_validation",
        "add_anchor_compute",
        "add_anchor_preview",
        "anchor_omit",
        "panel_diagnostics",
        "hook_context",
        "hook_results",
        "hook_engine",
        "modify_models",
        "task_codec",
        "task_models",
    ),
    "on-modify": (
        *_SHARED_HOOK_MODULES,
        "hook_support",
        "modify_queries",
        "lifecycle_read_service",
        "modify_spawn_prep",
        "modify_spawn",
        "chain_generation",
        "modify_ordinary",
        "modify_completion_preflight",
        "modify_completion_compute",
        "modify_completion_spawn",
        "modify_models",
        "task_codec",
        "task_models",
        "lifecycle_models",
        "lifecycle_planner",
        "chain_integrity_lifecycle",
        "lifecycle_application",
        "lifecycle_outbox",
        "modify_feedback",
        "modify_lifecycle",
        "modify_workflow",
        "modify_runtime",
        "modify_timeline",
        "modify_expiration",
        "modify_analytics",
        "anchor_omit",
        "add_anchor_compute",
        "panel_diagnostics",
        "hook_context",
        "hook_engine",
        "hook_results",
        "recurrence_evaluator",
        "modify_protocol",
        "modify_chain_summary",
        "modify_validation",
        "astronomy_validation",
        "modify_carry",
        "modify_carry_workflow",
        # These are imported directly in the completion path rather than via
        # HookModuleAccess, so keep them in the same deployment contract.
        "calendar_feedback",
        "modify_completion_flow",
    ),
    "on-exit": (
        *_SHARED_HOOK_MODULES,
        "integration_models",
        "taskwarrior_mutations",
        "hook_support",
        "lifecycle_application",
        "lifecycle_outbox",
        "exit_runtime",
        "hook_context",
        "hook_engine",
        "hook_results",
    ),
}


_HOOK_IMPL = {
    "on-add": "hooks/add_impl.py",
    "on-modify": "hooks/modify_impl.py",
    "on-exit": "hooks/exit_impl.py",
}

_HOOK_SUPPORT_FILES = {
    "on-add": ("hook_bootstrap.py", "hook_protocol.py", *_INTEGRATION_FILES),
    "on-modify": ("hook_bootstrap.py", "hook_protocol.py", *_INTEGRATION_FILES),
    "on-exit": ("hook_bootstrap.py", "config_support.py", "exit_probe.py", *_INTEGRATION_FILES),
}

HOOK_RUNTIME_FILES: dict[str, tuple[str, ...]] = {
    event: (
        impl,
        *_HOOK_SUPPORT_FILES[event],
        *(f"{module}.py" for module in modules),
    )
    for event, modules in HOOK_LAZY_MODULES.items()
    for impl in (_HOOK_IMPL[event],)
}

# Commands dispatched by the installed ``nautical`` launcher.  Hook smoke
# tests alone cannot prove these operator surfaces survived staging.
OPERATOR_RUNTIME_FILES = (
    "nautical",
    "nautical_navigator.py",
    "nautical_core/tools/nautical_install.py",
    "nautical_core/tools/nautical_runtime_cleanup.py",
    "nautical_core/tools/nautical_doctor.py",
    "nautical_core/tools/nautical_queue_status.py",
    "nautical_core/tools/nautical_reconcile.py",
    "nautical_core/reconcile_cli.py",
    "nautical_core/reconcile_report.py",
    "nautical_core/lifecycle_reconciliation.py",
    "nautical_core/integrity_report.py",
    "nautical_core/chain_integrity_recovery.py",
    "nautical_core/tools/nautical_query.py",
)

# Paths and symbols removed by the no-bridge lifecycle architecture. Deployment
# checks reject their reintroduction so a stale release cannot silently restore
# an alternate exit or reconcile owner.
REMOVED_RUNTIME_FILES = (
    "nautical_core/exit_drain_flow.py",
    "nautical_core/exit_entry_flow.py",
    "nautical_core/exit_models.py",
    "nautical_core/exit_side_effects.py",
)
REMOVED_RECONCILE_SYMBOLS = (
    "_is_legacy_root_without_link",
    "_validate_hook_protocol",
    "legacy_hook",
    "_RECONCILE_PROTOCOL",
)
# Operator commands compose the integrity/lifecycle services; they must not
# regain ownership by importing a hook implementation directly.
OPERATOR_FORBIDDEN_HOOK_IMPORTS = (
    "nautical_core.hooks",
    "nautical_core.hooks.add_impl",
    "nautical_core.hooks.modify_impl",
    "nautical_core.hooks.exit_impl",
    "nautical_core.hook_runtime",
)
PURE_INTEGRITY_MODULES = (
    "nautical_core/chain_graph.py",
    "nautical_core/chain_integrity_models.py",
    "nautical_core/chain_invariants.py",
    "nautical_core/chain_repair_planner.py",
)

# ``panel_colours`` is a core-facade lazy sibling rather than a hook
# ``_module()`` dependency, but it must still be present in staged releases.
for _event in HOOK_RUNTIME_FILES:
    HOOK_RUNTIME_FILES[_event] += ("panel_colours.py",)


__all__ = (
    "HOOK_LAZY_MODULES",
    "HOOK_RUNTIME_FILES",
    "OPERATOR_RUNTIME_FILES",
    "OPERATOR_FORBIDDEN_HOOK_IMPORTS",
    "PURE_INTEGRITY_MODULES",
)
