#!/usr/bin/env python3
"""Thin on-exit launcher with an empty-queue fast path."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


HOOK_DIR = Path(__file__).parent
TW_DIR = HOOK_DIR.parent
_EXPECTED_IMPL_API = 1


def _load_bootstrap():
    try:
        import hook_bootstrap

        return hook_bootstrap
    except ModuleNotFoundError:
        pass

    candidates = [
        HOOK_DIR / "nautical_core" / "hook_bootstrap.py",
        TW_DIR / "nautical_core" / "hook_bootstrap.py",
    ]
    raw_core_path = (os.environ.get("NAUTICAL_CORE_PATH") or "").strip()
    if raw_core_path:
        try:
            core_path = Path(raw_core_path).expanduser()
            candidates.extend(
                (
                    core_path / "hook_bootstrap.py",
                    core_path / "nautical_core" / "hook_bootstrap.py",
                )
            )
        except Exception:
            pass

    for path in candidates:
        try:
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location("hook_bootstrap", path)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception:
            continue
    raise ModuleNotFoundError("could not locate nautical_core/hook_bootstrap.py")


hook_bootstrap = _load_bootstrap()
hook_bootstrap.ensure_utf8_stdio()


def _trusted_core_base(default_base: Path) -> Path:
    return hook_bootstrap.trusted_core_base(
        default_base,
        env=os.environ,
        diag_enabled=os.environ.get("NAUTICAL_DIAG") == "1",
    )


def _core_target_from_base(base: Path) -> Path | None:
    return hook_bootstrap.core_target_from_base(base)


_DEFAULT_CORE_BASE = HOOK_DIR if hook_bootstrap.core_target_from_base(HOOK_DIR) is not None else TW_DIR
_CORE_BASE = _trusted_core_base(_DEFAULT_CORE_BASE)


def _full_path_requested() -> bool:
    return (
        os.environ.get("NAUTICAL_DIAG") == "1"
        or os.environ.get("NAUTICAL_BENCH_FORCE_FULL") == "1"
    )


def _diagnose(message: str) -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        sys.stderr.write(f"[nautical] on-exit {message}\n")
    except Exception:
        pass


def _probe_exit_work():
    path_support, _path_support_path, path_support_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "config_support.py",
        "_nautical_exit_path_support_wrapper",
    )
    exit_probe, _exit_probe_path, exit_probe_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "exit_probe.py",
        "_nautical_exit_probe_wrapper",
    )
    if path_support is None or exit_probe is None:
        error = path_support_error or exit_probe_error
        return None, error or RuntimeError("exit probe helpers were not found")
    try:
        taskdata = hook_bootstrap.resolve_task_data_context_light(
            path_support=path_support,
            argv=sys.argv[1:],
            env=os.environ,
            tw_dir=str(TW_DIR),
        )
        if taskdata is None:
            return None, RuntimeError("taskdata context could not be resolved")
        return exit_probe.probe_exit_work(taskdata[0]), None
    except Exception as exc:
        return None, exc


def main() -> int:
    probe, probe_error = _probe_exit_work()
    if probe is not None and probe.definitely_empty and not _full_path_requested():
        return 0

    impl, impl_path, impl_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "hooks/exit_impl.py",
        "_nautical_on_exit_impl",
    )
    try:
        if impl is None:
            raise RuntimeError(f"implementation unavailable: {impl_error or impl_path or 'not found'}")
        api = getattr(impl, "HOOK_IMPL_API", None)
        if api != _EXPECTED_IMPL_API:
            raise RuntimeError(
                f"implementation API mismatch: expected {_EXPECTED_IMPL_API}, got {api!r}"
            )
        run_hook = getattr(impl, "run_hook", None)
        if not callable(run_hook):
            raise RuntimeError("implementation has no callable run_hook")
        return int(
            run_hook(
                argv=tuple(sys.argv[1:]),
                hook_dir=str(HOOK_DIR),
                core_base=str(_CORE_BASE),
            )
            or 0
        )
    except SystemExit:
        raise
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
        if probe_error is not None:
            detail += f"; probe: {type(probe_error).__name__}: {probe_error}"
        _diagnose(f"startup failure: {detail}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
