from __future__ import annotations

import importlib
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class HookModuleAccess:
    namespace: dict[str, Any]
    module_specs: dict[str, tuple[str, str, str, str]]
    errors: dict[str, str] = field(default_factory=dict)

    def load_named_module(self, name: str):
        cache_attr, failed_attr, _rel_name, import_name = self.module_specs[name]
        module = self.namespace.get(cache_attr)
        if module is not None:
            return module
        if self.namespace.get(failed_attr):
            return None
        try:
            module = importlib.import_module(import_name)
            self.namespace[cache_attr] = module
            return module
        except Exception as exc:
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            self.namespace[failed_attr] = True
            return None

    def require_loaded_module(self, module, rel_name: str, error: str = ""):
        if module is None:
            detail = f" ({error})" if error else ""
            raise RuntimeError(f"nautical_core/{rel_name} is required{detail}")
        return module

    def module(self, name: str, *, required: bool = True):
        module = self.load_named_module(name)
        if not required:
            return module
        rel_name = self.module_specs[name][2]
        return self.require_loaded_module(module, rel_name, self.errors.get(name, ""))


def build_hook_runtime_context(
    *,
    module_access: HookModuleAccess,
    hook_name: str,
    integration_context,
    hook_dir: str,
    profile_level: int = 0,
    import_ms: float | None = None,
):
    hook_context = module_access.module("hook_context")
    return hook_context.build_hook_runtime_context(
        hook_name=hook_name,
        integration=integration_context,
        hook_dir=hook_dir,
        profile_level=profile_level,
        import_ms=import_ms,
    )


def initialize_integration_context(
    *,
    module_access: HookModuleAccess,
    hook_bootstrap,
    core_base: Path,
    argv: tuple[str, ...],
    tw_dir: str,
    access: str,
):
    """Import core and construct the sole validated context for a full hook."""
    core, target, import_error = hook_bootstrap.import_core_package(core_base)
    if core is None:
        target_text = str(target or (core_base / "nautical_core" / "__init__.py"))
        if import_error is not None:
            raise RuntimeError(
                f"Failed to import nautical_core from {target_text}: "
                f"{type(import_error).__name__}: {import_error}"
            ) from import_error
        raise ModuleNotFoundError(
            "nautical_core package not found. Expected nautical_core/__init__.py "
            f"in ~/.task or NAUTICAL_CORE_PATH (resolved base: {core_base})"
        )
    context_module = module_access.module("integration_context")
    context = context_module.build_integration_context(
        core=core,
        argv=argv,
        env=os.environ,
        tw_dir=tw_dir,
        task_binary=os.environ.get("NAUTICAL_BENCH_TASK_BIN", "task"),
        access=context_module.IntegrationAccess(access),
    )
    return core, target, context
