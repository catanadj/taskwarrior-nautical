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
        "add_anchor_compute",
        "add_anchor_preview",
        "anchor_omit",
        "panel_diagnostics",
        "hook_context",
        "hook_results",
        "hook_engine",
    ),
    "on-modify": (
        *_SHARED_HOOK_MODULES,
        "hook_support",
        "modify_queries",
        "lifecycle_read_service",
        "modify_spawn_prep",
        "chain_generation",
        "modify_ordinary",
        "modify_completion_preflight",
        "modify_completion_compute",
        "modify_completion_spawn",
        "modify_models",
        "lifecycle_models",
        "lifecycle_planner",
        "lifecycle_executor",
        "lifecycle_outbox",
        "modify_feedback",
        "modify_lifecycle",
        "modify_runtime",
        "modify_timeline",
        "modify_expiration",
        "modify_analytics",
        "anchor_omit",
        "add_anchor_compute",
        "panel_diagnostics",
        "queue_store",
        "queue_models",
        "reconcile",
        "hook_context",
        "hook_engine",
        "hook_results",
        "recurrence_evaluator",
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
        "exit_side_effects",
        "exit_entry_flow",
        "queue_store",
        "queue_models",
        "exit_models",
        "lifecycle_models",
        "lifecycle_planner",
        "lifecycle_executor",
        "lifecycle_outbox",
        "exit_runtime",
        "exit_drain_flow",
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

# ``panel_colours`` is a core-facade lazy sibling rather than a hook
# ``_module()`` dependency, but it must still be present in staged releases.
for _event in HOOK_RUNTIME_FILES:
    HOOK_RUNTIME_FILES[_event] += ("panel_colours.py",)


__all__ = ("HOOK_LAZY_MODULES", "HOOK_RUNTIME_FILES")
