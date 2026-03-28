from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


def ensure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        try:
            if stream is not None:
                stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def trusted_core_base(default_base: Path, *, env: dict[str, str] | None = None, diag_enabled: bool = False) -> Path:
    env = env or os.environ
    raw = (env.get("NAUTICAL_CORE_PATH") or "").strip()
    if not raw:
        return default_base
    try:
        cand = Path(raw).expanduser().resolve()
    except Exception:
        return default_base
    if (env.get("NAUTICAL_TRUST_CORE_PATH") or "").strip().lower() in ("1", "true", "yes", "on"):
        return cand
    try:
        st = os.stat(cand)
        uid_fn = getattr(os, "getuid", None)
        if callable(uid_fn) and st.st_uid != uid_fn():
            raise PermissionError("owner mismatch")
        if (st.st_mode & 0o002) != 0:
            raise PermissionError("path is world-writable")
        return cand
    except Exception as exc:
        if diag_enabled:
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe NAUTICAL_CORE_PATH '{raw}': {exc}\n")
            except Exception:
                pass
        return default_base


def core_target_from_base(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                return base
            return None
    except Exception:
        return None
    pkg_init = base / "nautical_core" / "__init__.py"
    return pkg_init if pkg_init.is_file() else None


def import_core_package(base: Path) -> tuple[Any | None, Path | None, Exception | None]:
    target = core_target_from_base(base)
    if target is None:
        return None, None, None
    try:
        pkg_parent = str(target.parent.parent)
        if pkg_parent not in sys.path:
            sys.path.insert(0, pkg_parent)
        existing = sys.modules.get("nautical_core")
        try:
            existing_file = Path(str(getattr(existing, "__file__", ""))).resolve() if existing is not None else None
        except Exception:
            existing_file = None
        if existing is not None and existing_file == target.resolve():
            return existing, target, None
        if existing is not None:
            sys.modules.pop("nautical_core", None)
        importlib.invalidate_caches()
        module = importlib.import_module("nautical_core")
        return module, target, None
    except Exception as exc:
        return None, target, exc


def resolve_task_data_context(*, core: Any, core_import_error: Exception | None, core_import_target: Path | None, core_base: Path, tw_dir: str, argv: list[str], env: dict[str, str]) -> tuple[str, bool]:
    resolver = getattr(core, "resolve_task_data_context", None) if core is not None else None
    if not callable(resolver):
        if core is not None:
            raise RuntimeError("nautical_core.resolve_task_data_context is required")
        if core_import_error is not None:
            target = str(core_import_target or (core_base / "nautical_core" / "__init__.py"))
            raise RuntimeError(
                f"Failed to import nautical_core from {target}: "
                f"{type(core_import_error).__name__}: {core_import_error}"
            ) from core_import_error
        raise ModuleNotFoundError(
            f"nautical_core package not found. Expected nautical_core/__init__.py in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {core_base})"
        )
    data_dir, use_rc, _source = resolver(
        argv=argv,
        env=env,
        tw_dir=tw_dir,
    )
    return str(data_dir), bool(use_rc)
