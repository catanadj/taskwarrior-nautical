from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class HookModuleAccess:
    namespace: dict[str, Any]
    module_specs: dict[str, tuple[str, str, str, str]]

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
        except Exception:
            self.namespace[failed_attr] = True
            return None

    def require_loaded_module(self, module, rel_name: str):
        if module is None:
            raise RuntimeError(f"nautical_core/{rel_name} is required")
        return module

    def module(self, name: str, *, required: bool = True):
        module = self.load_named_module(name)
        if not required:
            return module
        rel_name = self.module_specs[name][2]
        return self.require_loaded_module(module, rel_name)


def build_hook_runtime_context(
    *,
    module_access: HookModuleAccess,
    hook_name: str,
    taskdata_dir: str,
    use_rc_data_location: bool,
    tw_dir: str,
    hook_dir: str,
    profile_level: int = 0,
    import_ms: float | None = None,
):
    hook_context = module_access.module("hook_context")
    return hook_context.build_hook_runtime_context(
        hook_name=hook_name,
        taskdata_dir=taskdata_dir,
        use_rc_data_location=use_rc_data_location,
        tw_dir=tw_dir,
        hook_dir=hook_dir,
        profile_level=profile_level,
        import_ms=import_ms,
    )
