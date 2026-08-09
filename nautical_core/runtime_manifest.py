"""Files and lazy modules required by each Nautical lifecycle hook."""

from __future__ import annotations


_SHARED_HOOK_MODULES = ("hook_runtime",)


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
        "modify_chain_reads",
        "modify_spawn_prep",
        "chain_generation",
        "modify_generation_compat",
        "modify_ordinary",
        "modify_completion_preflight",
        "modify_completion_compute",
        "modify_completion_spawn",
        "modify_models",
        "modify_feedback",
        "modify_lifecycle",
        "modify_runtime",
        "modify_timeline",
        "modify_expiration",
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
        "hook_support",
        "exit_queries",
        "exit_side_effects",
        "exit_entry_flow",
        "queue_store",
        "queue_models",
        "exit_models",
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
    "on-add": ("hook_bootstrap.py", "hook_protocol.py"),
    "on-modify": ("hook_bootstrap.py", "hook_protocol.py"),
    "on-exit": ("hook_bootstrap.py", "config_support.py", "exit_probe.py"),
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


__all__ = ("HOOK_LAZY_MODULES", "HOOK_RUNTIME_FILES")
