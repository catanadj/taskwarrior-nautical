from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


_DEF_NAMES = (
    "ensure_utf8_stdio",
    "trusted_core_base",
    "core_target_from_base",
    "import_core_package",
    "resolve_task_data_context",
)


def _load_packaged_hook_bootstrap() -> ModuleType:
    target = Path(__file__).resolve().parent / "nautical_core" / "hook_bootstrap.py"
    spec = importlib.util.spec_from_file_location("nautical_core.hook_bootstrap", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load packaged hook_bootstrap from {target}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_PACKAGED = _load_packaged_hook_bootstrap()

for _name in _DEF_NAMES:
    globals()[_name] = getattr(_PACKAGED, _name)

__all__ = _DEF_NAMES
