#!/usr/bin/env python3
"""Thin on-modify launcher with a plain-task fast path."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


HOOK_DIR = Path(__file__).parent
TW_DIR = HOOK_DIR.parent
_MAX_JSON_BYTES = 10 * 1024 * 1024
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


def _read_input() -> bytes:
    source = getattr(sys.stdin, "buffer", sys.stdin)
    raw = source.read(_MAX_JSON_BYTES + 1)
    return raw if isinstance(raw, bytes) else str(raw or "").encode("utf-8")


def _emit_fallback(protocol, probe) -> None:
    task = getattr(probe, "task", None) if probe is not None else None
    if protocol is not None:
        try:
            protocol.emit_passthrough_json(task)
            return
        except Exception:
            pass
    sys.stdout.write(json.dumps(task if isinstance(task, dict) else {}, ensure_ascii=False))
    sys.stdout.flush()


def _diagnose(message: str) -> None:
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    try:
        sys.stderr.write(f"[nautical] on-modify {message}\n")
    except Exception:
        pass


def _full_path_requested() -> bool:
    return (
        os.environ.get("NAUTICAL_DIAG") == "1"
        or (os.environ.get("NAUTICAL_PROFILE") or "0").strip() not in ("", "0")
        or os.environ.get("NAUTICAL_BENCH_FORCE_FULL") == "1"
    )


def main() -> int:
    raw_input = _read_input()
    protocol, _protocol_path, protocol_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "hook_protocol.py",
        "_nautical_hook_protocol_modify_wrapper",
    )
    probe = None
    if protocol is not None:
        try:
            probe = protocol.probe_on_modify(raw_input, max_bytes=_MAX_JSON_BYTES)
        except Exception as exc:
            protocol_error = exc

    if probe is not None and probe.valid:
        plain_fast_path = not probe.is_nautical and os.environ.get("NAUTICAL_BENCH_FORCE_FULL") != "1"
        ordinary_nautical_fast_path = False
        if probe.is_nautical and not _full_path_requested():
            try:
                classify_ordinary = getattr(protocol, "is_safe_nautical_ordinary_modify", None)
                ordinary_nautical_fast_path = bool(
                    callable(classify_ordinary) and classify_ordinary(probe.old, probe.new)
                )
            except Exception:
                ordinary_nautical_fast_path = False
        if plain_fast_path or ordinary_nautical_fast_path:
            protocol.emit_passthrough_json(probe.task)
            return 0

    impl, impl_path, impl_error = hook_bootstrap.load_core_helper_module(
        _CORE_BASE,
        "hooks/modify_impl.py",
        "_nautical_on_modify_impl",
    )
    try:
        if protocol is None:
            raise RuntimeError(f"protocol unavailable: {protocol_error or 'not found'}")
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
                raw_input=raw_input,
                argv=tuple(sys.argv[1:]),
                hook_dir=str(HOOK_DIR),
                core_base=str(_CORE_BASE),
            )
            or 0
        )
    except SystemExit:
        raise
    except Exception as exc:
        _diagnose(f"startup failure: {type(exc).__name__}: {exc}")
        _emit_fallback(protocol, probe)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
