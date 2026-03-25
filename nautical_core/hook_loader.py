from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping


def optional_sibling_module_target(base: Path, rel_name: str) -> Path | None:
    try:
        if base.is_file():
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                target = base.parent / rel_name
                return target if target.is_file() else None
            return None
    except Exception:
        return None
    target = base / "nautical_core" / rel_name
    return target if target.is_file() else None


def load_optional_sibling_module(
    cache_store: MutableMapping[str, Any],
    failed_store: MutableMapping[str, Any],
    *,
    cache_attr: str,
    failed_attr: str,
    rel_name: str,
    module_name: str,
    base: Path,
):
    module = cache_store.get(cache_attr)
    if module is not None:
        return module
    if failed_store.get(failed_attr):
        return None
    target = optional_sibling_module_target(base, rel_name)
    if not target:
        failed_store[failed_attr] = True
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            cache_store[cache_attr] = module
            return module
    except Exception:
        pass
    failed_store[failed_attr] = True
    return None


def load_named_module(
    name: str,
    specs: Mapping[str, tuple[str, str, str, str]],
    cache_store: MutableMapping[str, Any],
    failed_store: MutableMapping[str, Any],
    *,
    base: Path,
):
    cache_attr, failed_attr, rel_name, module_name = specs[name]
    return load_optional_sibling_module(
        cache_store,
        failed_store,
        cache_attr=cache_attr,
        failed_attr=failed_attr,
        rel_name=rel_name,
        module_name=module_name,
        base=base,
    )


def require_loaded_module(module, rel_name: str):
    if module is None:
        raise RuntimeError(f"nautical_core/{rel_name} is required")
    return module
