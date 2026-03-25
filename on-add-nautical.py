#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
on-add-nautical.py (production)

Features:
- Weekly ranges require '..' (e.g., w:mon..fri).
- Auto-assign due when missing (anchor first, else cp).
- Panel logic:
    * If user set due -> First due = user's due; Next anchor shown separately.
    * If no due       -> First due = first anchor (or entry+cp) and due is auto-set.
- Optional warning if user's due isn't on an anchor day (constant toggle below).
- DST-safe datetime composition (local date+hh:mm -> UTC).
- Inclusive chainUntil behavior in preview.
"""

from __future__ import annotations

import sys, json, os, importlib, time, atexit, hashlib, random
import importlib.util
import re
from pathlib import Path
from datetime import timedelta, timezone, datetime
from functools import lru_cache
from contextlib import contextmanager
import subprocess
import tempfile
from collections import OrderedDict
from typing import Any
 # Ensure hook IO supports Unicode (emoji, symbols) in JSON output.
 # Python's json.dumps() defaults to ensure_ascii=True, which escapes non-ASCII
 # as "\\uXXXX". We prefer human-readable UTF-8 JSON for hook passthrough.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ========= User-togglable constants =========================================
NAUTICAL_HOOK_VERSION = "updateE-20260319"
ANCHOR_WARN = True  # If True, warn when a user-provided due is not on an anchor day
UPCOMING_PREVIEW = 5  # How many future dates to preview.
_PREVIEW_HARD_CAP = 100
_MAX_ITERATIONS = 2000
_MAX_PREVIEW_ITERATIONS = 750
_MAX_CHAIN_DURATION_YEARS = 5  # warn if chain extends this far
_MAX_JSON_BYTES = 10 * 1024 * 1024
# ============================================================================

# --------------------------------------------------------------------------------------
# Locate and import nautical_core (single fixed location: ~/.task)
# --------------------------------------------------------------------------------------
HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

def _trusted_core_base(default_base: Path) -> Path:
    raw = (os.environ.get("NAUTICAL_CORE_PATH") or "").strip()
    if not raw:
        return default_base
    try:
        cand = Path(raw).expanduser().resolve()
    except Exception:
        return default_base
    if (os.environ.get("NAUTICAL_TRUST_CORE_PATH") or "").strip().lower() in ("1", "true", "yes", "on"):
        return cand
    try:
        st = os.stat(cand)
        uid_fn = getattr(os, "getuid", None)
        if callable(uid_fn) and st.st_uid != uid_fn():
            raise PermissionError("owner mismatch")
        if (st.st_mode & 0o002) != 0:
            raise PermissionError("path is world-writable")
        return cand
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe NAUTICAL_CORE_PATH '{raw}': {e}\n")
            except Exception:
                pass
        return default_base

_CORE_BASE = _trusted_core_base(TW_DIR)


def _core_target_from_base(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "nautical_core.py":
                return base
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                return base
            return None
    except Exception:
        return None
    pyfile = base / "nautical_core.py"
    pkgini = base / "nautical_core" / "__init__.py"
    return pkgini if pkgini.is_file() else pyfile if pyfile.is_file() else None

# --- Optional micro-profiler (stderr-only; enable with NAUTICAL_PROFILE=1 or 2)
_PROFILE_LEVEL = int(os.environ.get('NAUTICAL_PROFILE', '0') or '0')
_IMPORT_T0 = time.perf_counter()

core = None
_CORE_READY = False
_CORE_IMPORT_ERROR: Exception | None = None
_CORE_IMPORT_TARGET: Path | None = None
_HOOK_LOADER = None
_HOOK_LOADER_LOAD_FAILED = False
_HOOK_SUPPORT = None
_HOOK_SUPPORT_LOAD_FAILED = False
_ADD_FORMATTING = None
_ADD_FORMATTING_LOAD_FAILED = False
_ADD_VALIDATION = None
_ADD_VALIDATION_LOAD_FAILED = False
_ADD_ANCHOR_COMPUTE = None
_ADD_ANCHOR_COMPUTE_LOAD_FAILED = False
_ADD_ANCHOR_PREVIEW = None
_ADD_ANCHOR_PREVIEW_LOAD_FAILED = False
_HOOK_CONTEXT = None
_HOOK_CONTEXT_LOAD_FAILED = False
_HOOK_RESULTS = None
_HOOK_RESULTS_LOAD_FAILED = False
_HOOK_ENGINE = None
_HOOK_ENGINE_LOAD_FAILED = False
_MODULE_SPECS = {
    "hook_support": (
        "_HOOK_SUPPORT",
        "_HOOK_SUPPORT_LOAD_FAILED",
        "hook_support.py",
        "nautical_hook_support",
    ),
    "add_formatting": (
        "_ADD_FORMATTING",
        "_ADD_FORMATTING_LOAD_FAILED",
        "add_formatting.py",
        "nautical_add_formatting",
    ),
    "add_validation": (
        "_ADD_VALIDATION",
        "_ADD_VALIDATION_LOAD_FAILED",
        "add_validation.py",
        "nautical_add_validation",
    ),
    "add_anchor_compute": (
        "_ADD_ANCHOR_COMPUTE",
        "_ADD_ANCHOR_COMPUTE_LOAD_FAILED",
        "add_anchor_compute.py",
        "nautical_add_anchor_compute",
    ),
    "add_anchor_preview": (
        "_ADD_ANCHOR_PREVIEW",
        "_ADD_ANCHOR_PREVIEW_LOAD_FAILED",
        "add_anchor_preview.py",
        "nautical_add_anchor_preview",
    ),
    "hook_context": (
        "_HOOK_CONTEXT",
        "_HOOK_CONTEXT_LOAD_FAILED",
        "hook_context.py",
        "nautical_hook_context",
    ),
    "hook_results": (
        "_HOOK_RESULTS",
        "_HOOK_RESULTS_LOAD_FAILED",
        "hook_results.py",
        "nautical_hook_results",
    ),
    "hook_engine": (
        "_HOOK_ENGINE",
        "_HOOK_ENGINE_LOAD_FAILED",
        "hook_engine.py",
        "nautical_hook_engine",
    ),
}
try:
    target = _core_target_from_base(_CORE_BASE)
    _CORE_IMPORT_TARGET = target
    if target:
        spec = importlib.util.spec_from_file_location("nautical_core", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_core"] = module
            spec.loader.exec_module(module)
            core = module
except Exception as e:
    core = None
    _CORE_IMPORT_ERROR = e

def _resolve_task_data_context() -> tuple[str, bool]:
    resolver = getattr(core, "resolve_task_data_context", None) if core is not None else None
    if not callable(resolver):
        if core is not None:
            raise RuntimeError("nautical_core.resolve_task_data_context is required")
        if _CORE_IMPORT_ERROR is not None:
            target = str(_CORE_IMPORT_TARGET or (_CORE_BASE / "nautical_core.py"))
            raise RuntimeError(
                f"Failed to import nautical_core from {target}: "
                f"{type(_CORE_IMPORT_ERROR).__name__}: {_CORE_IMPORT_ERROR}"
            ) from _CORE_IMPORT_ERROR
        raise ModuleNotFoundError(
            f"nautical_core.py not found. Expected in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {_CORE_BASE})"
        )
    data_dir, use_rc, _source = resolver(
        argv=sys.argv[1:],
        env=os.environ,
        tw_dir=str(TW_DIR),
    )
    return str(data_dir), bool(use_rc)


_TASKDATA_RAW, _USE_RC_DATA_LOCATION = _resolve_task_data_context()
TW_DATA_DIR = Path(_TASKDATA_RAW).expanduser()
_IMPORT_MS = None


def _hook_loader_target(base: Path) -> Path | None:
    try:
        if base.is_file():
            if base.name == "__init__.py" and base.parent.name == "nautical_core":
                target = base.parent / "hook_loader.py"
                return target if target.is_file() else None
            return None
    except Exception:
        return None
    target = base / "nautical_core" / "hook_loader.py"
    return target if target.is_file() else None


def _load_hook_loader():
    global _HOOK_LOADER, _HOOK_LOADER_LOAD_FAILED
    if _HOOK_LOADER is not None:
        return _HOOK_LOADER
    if _HOOK_LOADER_LOAD_FAILED:
        return None
    base = _CORE_IMPORT_TARGET or _CORE_BASE
    target = _hook_loader_target(base)
    if not target:
        _HOOK_LOADER_LOAD_FAILED = True
        return None
    try:
        spec = importlib.util.spec_from_file_location("nautical_hook_loader", target)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["nautical_hook_loader"] = module
            spec.loader.exec_module(module)
            _HOOK_LOADER = module
            return module
    except Exception:
        pass
    _HOOK_LOADER_LOAD_FAILED = True
    return None


def _load_named_module(name: str):
    hook_loader = _load_hook_loader()
    if hook_loader is None:
        return None
    return hook_loader.load_named_module(
        name,
        _MODULE_SPECS,
        globals(),
        globals(),
        base=_CORE_IMPORT_TARGET or _CORE_BASE,
    )


def _require_loaded_module(module, rel_name: str):
    hook_loader = _load_hook_loader()
    if hook_loader is not None:
        return hook_loader.require_loaded_module(module, rel_name)
    if module is None:
        raise RuntimeError(f"nautical_core/{rel_name} is required")
    return module


def _module(name: str, *, required: bool = True):
    module = _load_named_module(name)
    if not required:
        return module
    rel_name = _MODULE_SPECS[name][2]
    return _require_loaded_module(module, rel_name)

def _task_cmd_prefix() -> list[str]:
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.build_task_cmd_prefix(
            use_rc_data_location=_USE_RC_DATA_LOCATION,
            tw_data_dir=TW_DATA_DIR,
        )
    cmd = ["task"]
    if _USE_RC_DATA_LOCATION:
        cmd.append(f"rc.data.location={TW_DATA_DIR}")
    return cmd



def _load_core() -> None:
    global core, _IMPORT_MS, _MAX_JSON_BYTES, _CORE_READY
    if core is not None and _CORE_READY:
        return
    if core is None:
        base = _CORE_BASE
        target = _core_target_from_base(base)
        if target:
            spec = importlib.util.spec_from_file_location("nautical_core", target)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules["nautical_core"] = module
                spec.loader.exec_module(module)
                core = module
    if core is None:
        msg = (
            "nautical_core.py not found. Expected in ~/.task or NAUTICAL_CORE_PATH. "
            f"(resolved base: {_CORE_BASE})"
        )
        raise ModuleNotFoundError(msg)
    try:
        core._warn_once_per_day_any("core_path", f"[nautical] core loaded: {getattr(core, '__file__', 'unknown')}")
    except Exception:
        pass
    try:
        _MAX_JSON_BYTES = int(getattr(core, "MAX_JSON_BYTES", _MAX_JSON_BYTES))
    except Exception:
        pass
    try:
        global _CORE_SIG
        _CORE_SIG = _core_sig()
    except Exception:
        pass
    _IMPORT_MS = (time.perf_counter() - _IMPORT_T0) * 1000.0
    _CORE_READY = True

class _Profiler:
    """Minimal, low-risk profiler for hook execution (stderr-only)."""

    def __init__(self, level: int = 0, import_ms: float | None = None):
        self.level = int(level or 0)
        self.enabled = self.level > 0
        self.import_ms = float(import_ms) if import_ms is not None else None
        self._t0 = time.perf_counter()
        self._events: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._events.append((name, (time.perf_counter() - t0) * 1000.0))

    def add_ms(self, name: str, ms: float) -> None:
        if self.enabled:
            self._events.append((name, float(ms)))

    def emit(self) -> None:
        if not self.enabled:
            return
        total_ms = (time.perf_counter() - self._t0) * 1000.0
        lines = []
        lines.append(f"[NAUTICAL_PROFILE] total={total_ms:.1f}ms")
        if self.import_ms is not None:
            lines.append(f"  import_core={self.import_ms:.1f}ms")
        for name, ms in self._events:
            lines.append(f"  {name}={ms:.1f}ms")
        if self.level >= 2 and self._events:
            lines.append("  -- slowest --")
            for name, ms in sorted(self._events, key=lambda x: x[1], reverse=True)[:8]:
                lines.append(f"  {name}={ms:.1f}ms")
        sys.stderr.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
@lru_cache(maxsize=512)
def _parse_dt_any_cached(s: str):
    return core.parse_dt_any(s)


@lru_cache(maxsize=512)
def _fmt_dt_local_cached(dt):
    return core.fmt_dt_local(dt)


@lru_cache(maxsize=512)
def _to_local_cached(dt):
    return core.to_local(dt)


# ---- Optional cross-invocation disk cache for expensive anchor parsing (Termux-friendly) ----
# Default on; set NAUTICAL_DNF_DISK_CACHE=0 to disable.
_DNF_DISK_CACHE_ENABLED = (os.getenv("NAUTICAL_DNF_DISK_CACHE") or "1").strip().lower() in ("1", "true", "yes", "on")
_DNF_DISK_CACHE_PATH = HOOK_DIR / ".nautical_cache" / "dnf_cache.jsonl"
_DNF_DISK_CACHE: OrderedDict[str, Any] | None = None
_DNF_DISK_CACHE_DIRTY = False
_DNF_DISK_CACHE_MAX = 256
_DNF_DISK_CACHE_LOCK = _DNF_DISK_CACHE_PATH.with_suffix(".lock")
_DNF_DISK_CACHE_VERSION = 1
_DNF_DISK_CACHE_MAX_BYTES = 256 * 1024
_DNF_LOCK_RETRIES = 6
_DNF_LOCK_SLEEP_BASE = 0.03

def _core_sig() -> str:
    try:
        _load_core()
        st = Path(core.__file__).stat()
        return f"{st.st_mtime_ns}:{st.st_size}"
    except Exception:
        return "unknown"

_CORE_SIG = _core_sig()

def _dnf_cache_key(expr: str) -> str:
    # tie cache entries to the current core build
    return f"{_CORE_SIG}|{expr}"

@contextmanager
def _dnf_cache_lock():
    """Best-effort lock for disk cache access. Yields True if acquired."""
    try:
        _load_core()
    except Exception:
        yield False
        return
    with core.safe_lock(
        _DNF_DISK_CACHE_LOCK,
        retries=_DNF_LOCK_RETRIES,
        sleep_base=_DNF_LOCK_SLEEP_BASE,
        jitter=_DNF_LOCK_SLEEP_BASE,
        mkdir=True,
        stale_after=30.0,
    ) as acquired:
        yield acquired

def _dnf_cache_remove_oversized_file() -> tuple[bool, int | None]:
    try:
        st = _DNF_DISK_CACHE_PATH.stat()
        file_size = st.st_size
    except Exception:
        return False, None
    if file_size <= _DNF_DISK_CACHE_MAX_BYTES:
        return False, file_size
    _diag(f"DNF cache too large; resetting: {_DNF_DISK_CACHE_PATH}")
    try:
        _DNF_DISK_CACHE_PATH.unlink()
    except Exception:
        pass
    return True, file_size


def _dnf_cache_read_nonempty_lines() -> list[str]:
    with open(_DNF_DISK_CACHE_PATH, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _dnf_cache_split_header(lines: list[str]) -> tuple[list[str], bool]:
    soft_error = False
    data_lines = lines
    try:
        first_obj = json.loads(lines[0])
    except Exception:
        first_obj = None
    if isinstance(first_obj, dict) and "version" in first_obj:
        if int(first_obj.get("version") or 0) != _DNF_DISK_CACHE_VERSION:
            soft_error = True
        checksum = (first_obj.get("checksum") or "").strip()
        data_lines = lines[1:]
        if checksum:
            payload = "\n".join(data_lines)
            calc = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            if checksum != calc:
                soft_error = True
    return data_lines, soft_error


def _dnf_cache_extract_kv(obj: object) -> tuple[str | None, object]:
    key = None
    val = None
    if isinstance(obj, dict):
        if "key" in obj and "value" in obj:
            key = obj.get("key")
            val = obj.get("value")
        elif "k" in obj and "v" in obj:
            key = obj.get("k")
            val = obj.get("v")
    if key is None:
        return None, None
    return str(key), val


def _dnf_cache_ingest_lines(data_lines: list[str], cache: OrderedDict) -> tuple[bool, bool]:
    parsed_any = False
    soft_error = False
    for line in data_lines:
        try:
            obj = json.loads(line)
        except Exception:
            soft_error = True
            continue
        key, val = _dnf_cache_extract_kv(obj)
        if key is None:
            continue
        cache[key] = val
        parsed_any = True
    return parsed_any, soft_error


def _dnf_cache_quarantine_current() -> None:
    try:
        ts = int(time.time())
        bad = _DNF_DISK_CACHE_PATH.with_suffix(f".corrupt.{ts}.jsonl")
        os.replace(_DNF_DISK_CACHE_PATH, bad)
        _diag(f"DNF cache quarantined: {bad}")
    except Exception:
        pass


def _load_dnf_disk_cache() -> OrderedDict:
    global _DNF_DISK_CACHE, _DNF_DISK_CACHE_DIRTY
    if _DNF_DISK_CACHE is not None:
        return _DNF_DISK_CACHE
    _DNF_DISK_CACHE = OrderedDict()
    if not _DNF_DISK_CACHE_ENABLED:
        return _DNF_DISK_CACHE
    try:
        with _dnf_cache_lock() as locked:
            if not locked:
                return _DNF_DISK_CACHE
            _DNF_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            if _DNF_DISK_CACHE_PATH.exists():
                was_oversized, file_size = _dnf_cache_remove_oversized_file()
                if was_oversized:
                    return _DNF_DISK_CACHE
                lines = _dnf_cache_read_nonempty_lines()
                if not lines:
                    return _DNF_DISK_CACHE
                data_lines, soft_error = _dnf_cache_split_header(lines)
                parsed_any, ingest_soft_error = _dnf_cache_ingest_lines(data_lines, _DNF_DISK_CACHE)
                if ingest_soft_error:
                    soft_error = True
                if soft_error and parsed_any:
                    _DNF_DISK_CACHE_DIRTY = True
                if not parsed_any and (file_size or 0) > 0:
                    _dnf_cache_quarantine_current()
                    _DNF_DISK_CACHE = OrderedDict()
                    return _DNF_DISK_CACHE
    except Exception as e:
        _DNF_DISK_CACHE = OrderedDict()
    return _DNF_DISK_CACHE

def _save_dnf_disk_cache() -> None:
    global _DNF_DISK_CACHE_DIRTY
    if not (_DNF_DISK_CACHE_ENABLED and _DNF_DISK_CACHE_DIRTY and isinstance(_DNF_DISK_CACHE, OrderedDict)):
        return
    tmp = None
    try:
        with _dnf_cache_lock() as locked:
            if not locked:
                return
            _DNF_DISK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            # trim oldest
            while len(_DNF_DISK_CACHE) > _DNF_DISK_CACHE_MAX:
                _DNF_DISK_CACHE.popitem(last=False)
            fd, tmp = tempfile.mkstemp(
                dir=str(_DNF_DISK_CACHE_PATH.parent),
                prefix=".dnf_cache.",
                suffix=".tmp",
            )
            try:
                os.fchmod(fd, 0o600)
            except Exception:
                pass
            records = []
            for k, v in _DNF_DISK_CACHE.items():
                rec = {"key": k, "value": v}
                records.append(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
            payload = "\n".join(records)
            checksum = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                header = {"version": _DNF_DISK_CACHE_VERSION, "checksum": checksum}
                f.write(json.dumps(header, ensure_ascii=False, separators=(",", ":")) + "\n")
                if payload:
                    f.write(payload + "\n")
            os.replace(tmp, _DNF_DISK_CACHE_PATH)
            _DNF_DISK_CACHE_DIRTY = False
    except Exception:
        # cache write failures must never affect hook correctness
        pass
    finally:
        if tmp and os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except Exception:
                pass

def _save_dnf_disk_cache_signal_safe() -> None:
    """Best-effort cache save for atexit; avoid heavy work if shutting down."""
    try:
        if not _DNF_DISK_CACHE_DIRTY:
            return
        _save_dnf_disk_cache()
    except Exception:
        pass

if _DNF_DISK_CACHE_ENABLED:
    atexit.register(_save_dnf_disk_cache_signal_safe)

@lru_cache(maxsize=256)
def _validate_anchor_expr_cached(expr: str) -> list[list[dict]]:
    """
    Validate + parse anchor expression to DNF.

    Always caches in-memory (per invocation). Optionally caches across invocations
    when NAUTICAL_DNF_DISK_CACHE=1.
    """
    _load_core()
    global _DNF_DISK_CACHE_DIRTY
    if _DNF_DISK_CACHE_ENABLED:
        cache = _load_dnf_disk_cache()
        k = _dnf_cache_key(expr)
        if k in cache:
            cache.move_to_end(k)
            return cache[k]

    dnf = core.validate_anchor_expr_strict(expr)

    if _DNF_DISK_CACHE_ENABLED:
        cache = _load_dnf_disk_cache()
        k = _dnf_cache_key(expr)
        try:
            json.dumps(dnf, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            _diag("DNF cache skip: value not JSON-serializable")
            return dnf
        cache[k] = dnf
        cache.move_to_end(k)
        _DNF_DISK_CACHE_DIRTY = True

    return dnf

def _task_has_nautical_fields(task: dict) -> bool:
    if not isinstance(task, dict):
        return False
    for key in ("anchor", "anchor_mode", "cp", "chainID", "chainid", "chainMax", "chainUntil"):
        if (task.get(key) or "").strip():
            return True
    return False


def _human_delta(a, b, use_months_days=True):
    return core.humanize_delta(a, b, use_months_days=use_months_days)


def _short(u):
    try:
        s = str(u)
        return s[:8] if s else "—"
    except Exception:
        return "—"


def _root_uuid_from(task: dict) -> str | None:
    u = task.get("uuid")
    return str(u) if u else None


def _strip_quotes(s: str) -> str:
    s = (s or "").strip()
    return s[1:-1] if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"') else s

_PANEL_THEMES = {
    "preview_anchor": {"border": "light_sea_green", "title": "light_sea_green", "label": "cyan"},
    "preview_cp": {"border": "hot_pink", "title": "bright_green", "label": "green"},
    "error": {"border": "red", "title": "red", "label": "red"},
    "warning": {"border": "yellow", "title": "yellow", "label": "yellow"},
    "info": {"border": "white", "title": "white", "label": "white"},
}


def _panel(title, rows, kind: str = "info"):
    if core is None:
        try:
            _load_core()
        except Exception:
            try:
                sys.stderr.write(f"[nautical] {title}\n")
            except Exception:
                pass
            return
    core.render_panel(
        title,
        rows,
        kind=kind,
        panel_mode=core.PANEL_MODE,
        fast_color=core.FAST_COLOR,
        themes=_PANEL_THEMES,
        allow_line=False,
        label_width_min=6,
        label_width_max=28,
    )

def _format_anchor_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Compact layout for anchor preview."""
    add_formatting = _module("add_formatting")
    return add_formatting.format_anchor_rows(rows)


_DIAG_REDACT_KEYS = frozenset({"description", "annotation", "annotations", "note", "notes"})


def _diag_redact_msg(msg: object) -> str:
    raw = msg if isinstance(msg, str) else str(msg)
    redactor = getattr(core, "diag_log_redact", None) if core is not None else None
    if callable(redactor):
        try:
            red = redactor(raw)
            return red if isinstance(red, str) else str(red)
        except Exception:
            pass
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            for k in list(data.keys()):
                if k in _DIAG_REDACT_KEYS:
                    data[k] = "[redacted]"
            return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        pass
    return raw


def _diag(msg: str) -> None:
    safe_msg = _diag_redact_msg(msg)
    try:
        _load_core()
    except Exception:
        pass
    if core is not None:
        core.diag(safe_msg, "on-add", str(TW_DATA_DIR))
    elif os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            sys.stderr.write(f"[nautical] {safe_msg}\n")
        except Exception:
            pass


def _run_task(
    cmd: list[str],
    *,
    env: dict | None = None,
    input_text: str | None = None,
    timeout: float = 3.0,
    retries: int = 2,
    retry_delay: float = 0.15,
) -> tuple[bool, str, str]:
    load_err: Exception | None = None
    if core is None:
        try:
            _load_core()
        except Exception as e:
            load_err = e
    core_runner = getattr(core, "run_task", None) if core is not None else None
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        if load_err is not None and not callable(core_runner):
            _diag(f"core.run_task unavailable; falling back to subprocess: {load_err}")
        return hook_support.run_task(
            cmd,
            core_run_task=core_runner,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )
    if callable(core_runner):
        return core_runner(
            cmd,
            env=env,
            input_text=input_text,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )
    if load_err is not None:
        _diag(f"core.run_task unavailable; falling back to subprocess: {load_err}")
    env = env or os.environ.copy()
    proc = None
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            close_fds=True,
            env=env,
        )
        out, err = proc.communicate(input=input_text, timeout=timeout)
        return (proc.returncode == 0, out or "", err or "")
    except subprocess.TimeoutExpired:
        if proc is not None:
            proc.kill()
        try:
            out, err = proc.communicate(timeout=1.0) if proc is not None else ("", "")
        except Exception:
            out, err = "", ""
        return (False, out or "", "timeout")
    except Exception as e:
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass
        return (False, "", str(e))



def _format_cp_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Compact layout for classic cp preview."""
    add_formatting = _module("add_formatting")
    return add_formatting.format_cp_rows(rows)


def _fail_and_exit(title: str, msg: str) -> None:
    # Pretty panel -> stderr
    _panel(f"❌ {title}", [("Message", msg)], kind="error")
    sys.exit(1)


def _error_and_exit(msg_tuples):
    _panel("❌ Invalid Chain", msg_tuples, kind="error")
    sys.exit(1)

_RAW_INPUT_TEXT = ""
_PARSED_TASK = None

def _panic_passthrough() -> None:
    """Emit a valid fallback JSON task on unexpected errors."""
    hook_results = _module("hook_results")
    hook_results.panic_passthrough(_RAW_INPUT_TEXT, _PARSED_TASK)



# Local ISO string back to Taskwarrior (lets default-due adjusters run)
def _fmt_local_for_task(dt_utc):
    dl = core.to_local(dt_utc)
    return dl.strftime("%Y-%m-%dT%H:%M:%S")


# ------------------------------------------------------------------------------
# wait/scheduled preview helpers (panel only; no mutation on add)
# ------------------------------------------------------------------------------
def _fmt_td_no_seconds(td: timedelta) -> str:
    """Format timedelta as ±Dd HHh:MMm (seconds omitted)."""
    if not isinstance(td, timedelta):
        return "—"
    total_sec = int(td.total_seconds())
    sign = "-" if total_sec < 0 else "+"
    total_min = abs(total_sec) // 60
    days, rem = divmod(total_min, 60 * 24)
    hours, minutes = divmod(rem, 60)
    return f"{sign}{days}d {hours:02d}h:{minutes:02d}m"


def _append_wait_sched_rows(rows: list, task: dict, due_utc: datetime, auto_due: bool) -> None:
    """
    Append Scheduled/Wait lines to panel using LOCAL display, with UTC deltas to due.
    Also emit warnings when order due > scheduled > wait is violated.
    """
    if not (isinstance(rows, list) and isinstance(task, dict) and isinstance(due_utc, datetime)):
        return

    def _p(field: str) -> datetime | None:
        v = task.get(field)
        if not v:
            return None
        try:
            return core.parse_dt_any(v)
        except Exception:
            return None

    sched_dt = _p("scheduled")
    wait_dt = _p("wait")

    if sched_dt:
        rows.append((
            "Scheduled",
            f"[white]{core.fmt_dt_local(sched_dt)}[/]  [dim](Δ {_fmt_td_no_seconds(sched_dt - due_utc)})[/]"
        ))
    if wait_dt:
        rows.append((
            "Wait",
            f"[white]{core.fmt_dt_local(wait_dt)}[/]  [dim](Δ {_fmt_td_no_seconds(wait_dt - due_utc)})[/]"
        ))

    issues: list[str] = []
    if sched_dt and sched_dt > due_utc:
        issues.append(f"scheduled is after due by {_fmt_td_no_seconds(sched_dt - due_utc)}")
    if wait_dt and wait_dt > due_utc:
        issues.append(f"wait is after due by {_fmt_td_no_seconds(wait_dt - due_utc)}")
    if wait_dt and sched_dt and wait_dt > sched_dt:
        issues.append(f"wait is after scheduled by {_fmt_td_no_seconds(wait_dt - sched_dt)}")

    if issues:
        rows.append((
            "Warning",
            "[yellow]Wait/Sched: expected order due > scheduled > wait. " + "; ".join(issues) + ".[/]"
        ))
        if auto_due:
            rows.append((
                "Note",
                "[dim]This can happen when due is auto-assigned; adjust scheduled/wait if undesired.[/]"
            ))


# Helper to validate chainUntil is in the future
def _validate_until_not_past(until_dt, now_utc) -> tuple[bool, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_until_not_past(until_dt, now_utc, core=core)


# Helper to check if due is in the past (warning only)
def _check_due_in_past(due_dt, now_utc) -> tuple[bool, str | None]:
    add_validation = _module("add_validation")
    return add_validation.check_due_in_past(due_dt, now_utc, core=core)


# Helper to warn if chain extends unreasonably far
def _validate_chain_duration_reasonable(
    until_dt, now_utc, first_due, kind
) -> tuple[bool, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_chain_duration_reasonable(
        until_dt,
        now_utc,
        first_due,
        kind,
        max_chain_duration_years=_MAX_CHAIN_DURATION_YEARS,
    )


# Helper to validate cp/anchor not missing
def _validate_kind_not_conflicting(cp_str, anchor_str) -> tuple[bool, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_kind_not_conflicting(cp_str, anchor_str)


# Helper to validate chainMax > 0
def _validate_cpmax_positive(cpmax) -> tuple[bool, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_cpmax_positive(cpmax)


# Helper to safely parse with context
def _safe_parse_datetime(s, field_name) -> tuple[datetime | None, str | None]:
    add_validation = _module("add_validation")
    return add_validation.safe_parse_datetime(s, field_name, core=core, diag=_diag)


def _validate_no_legacy_colon_ranges(expr: str) -> tuple[bool, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_no_legacy_colon_ranges(expr)

def _safe_parse_duration(s, field_name) -> tuple[timedelta | None, str | None]:
    add_validation = _module("add_validation")
    return add_validation.safe_parse_duration(s, field_name, core=core, diag=_diag)


def _validate_anchor_syntax_strict(expr: str | list[list[dict]]) -> tuple[list[list[dict]] | None, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_anchor_syntax_strict(
        expr,
        validate_anchor_expr_cached=_validate_anchor_expr_cached,
        core=core,
        diag=_diag,
    )


def _validate_anchor_mode(mode_str) -> tuple[str, str | None]:
    add_validation = _module("add_validation")
    return add_validation.validate_anchor_mode(mode_str)


def _parse_extra_tokens(extra: str | None) -> list[str] | None:
    """Parse extra Taskwarrior filters in strict token form: key:value."""
    hook_support = _module("hook_support", required=False)
    if hook_support is not None:
        return hook_support.parse_extra_tokens(extra)
    if extra is None:
        return []
    if not isinstance(extra, str):
        return None
    s = extra.strip()
    if not s:
        return []
    out: list[str] = []
    for tok in s.split():
        if tok.startswith("+"):
            tag = tok[1:]
            if not tag or re.fullmatch(r"[A-Za-z0-9_.-]+", tag) is None:
                return None
            out.append(tok)
            continue
        if tok.startswith("-"):
            return None
        if ":" not in tok:
            return None
        key, value = tok.split(":", 1)
        if not key or not value:
            return None
        if re.fullmatch(r"[A-Za-z0-9_.-]+", key) is None:
            return None
        if re.fullmatch(r"[A-Za-z0-9_.:@%+,-]+", value) is None:
            return None
        out.append(f"{key}:{value}")
    return out

def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None) -> list[dict]:
    if not chain_id:
        return []
    hook_support = _module("hook_support", required=False)
    args = None
    if hook_support is not None:
        args = hook_support.build_chain_export_args(
            task_cmd_prefix=_task_cmd_prefix(),
            chain_id=chain_id,
            since=since,
            extra=extra,
            limit=None,
            parse_extra_tokens=_parse_extra_tokens,
            diag=_diag,
        )
    if args is None and hook_support is None:
        args = _task_cmd_prefix()
        args += ["rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
        if since:
            args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
        if extra:
            extra_tokens = _parse_extra_tokens(extra)
            if extra_tokens is None:
                _diag(f"tw_export_chain rejected extra: {extra!r}")
                return []
            args += extra_tokens
        args.append("export")
    if args is None:
        return []
    ok, out, err = _run_task(args, timeout=3.0, retries=2)
    if not ok:
        _diag(f"tw_export_chain failed (chainID={chain_id}): {err.strip()}")
        return []
    try:
        if hook_support is not None:
            return hook_support.parse_export_array(out, diag=_diag)
        data = json.loads(out.strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception as e:
        _diag(f"tw_export_chain JSON parse failed: {e}")
        return []


def _stamp_chain_id_on_add(task: dict) -> None:
    # [CHAINID] Stamp short root id on new chains (anchor/cp present, no existing chainID)
    try:
        if (task.get("anchor") or task.get("cp")) and not (task.get("chainID") or "").strip():
            task["chainID"] = core.short_uuid(task.get("uuid"))
    except Exception:
        # Never block task creation on bookkeeping
        pass


def _norm_t_mod(v):
    if v is None:
        return []
    if isinstance(v, tuple) and len(v) == 2:
        return [v]
    if isinstance(v, list):
        out = []
        for it in v:
            if isinstance(it, tuple) and len(it) == 2:
                out.append(it)
            elif isinstance(it, list) and len(it) == 2:
                out.append((int(it[0]), int(it[1])))
        return out
    if isinstance(v, str):
        out = []
        for part in [p.strip() for p in v.split(",") if p.strip()]:
            if len(part) == 5 and part[2] == ":" and part[:2].isdigit() and part[3:].isdigit():
                out.append((int(part[:2]), int(part[3:])))
        return out
    return []


def _anchor_step_once(dnf, prev_local_date, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_step_once(
        dnf, prev_local_date, interval_seed, seed_base, core=core
    )


def _anchor_term_fires_on_date(term, d, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_term_fires_on_date(
        term, d, interval_seed, seed_base, core=core
    )


def _anchor_expr_fires_on_date(dnf, d, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_expr_fires_on_date(
        dnf, d, interval_seed, seed_base, core=core
    )


def _anchor_times_for_date(dnf, d, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_times_for_date(
        dnf,
        d,
        interval_seed,
        seed_base,
        core=core,
        norm_t_mod=_norm_t_mod,
    )


def _anchor_pick_occurrence_local(dnf, ref_dt_local, inclusive: bool, fallback_hhmm, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_pick_occurrence_local(
        dnf,
        ref_dt_local,
        inclusive,
        fallback_hhmm,
        interval_seed,
        seed_base,
        core=core,
        norm_t_mod=_norm_t_mod,
    )


def _anchor_next_occurrence_after_local_dt(dnf, after_dt_local, fallback_hhmm, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_next_occurrence_after_local_dt(
        dnf,
        after_dt_local,
        fallback_hhmm,
        interval_seed,
        seed_base,
        core=core,
        norm_t_mod=_norm_t_mod,
    )


def _anchor_until_summary(dnf, until_dt, first_date_local, first_hhmm, interval_seed, seed_base):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_until_summary(
        dnf,
        until_dt,
        first_date_local,
        first_hhmm,
        interval_seed,
        seed_base,
        core=core,
        to_local_cached=_to_local_cached,
        max_preview_iterations=_MAX_PREVIEW_ITERATIONS,
        max_iterations=_MAX_ITERATIONS,
    )


def _anchor_build_preview(
    dnf,
    first_due_local_dt,
    preview_limit: int,
    until_dt,
    fallback_hhmm,
    interval_seed,
    seed_base,
):
    add_anchor_compute = _module("add_anchor_compute")
    return add_anchor_compute.anchor_build_preview(
        dnf,
        first_due_local_dt,
        preview_limit,
        until_dt,
        fallback_hhmm,
        interval_seed,
        seed_base,
        core=core,
        norm_t_mod=_norm_t_mod,
    )


def _cp_add_period_builder(td: timedelta):
    secs = int(td.total_seconds())
    preserve = secs % 86400 == 0

    def add_period(dt):
        if preserve:
            dl = core.to_local(dt)
            base = core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
            )
            return base.astimezone(timezone.utc)
        return (dt + td).replace(microsecond=0)

    return add_period


def _cp_until_summary(due_dt: datetime, until_dt: datetime | None, add_period) -> tuple[int | None, datetime | None]:
    if not until_dt:
        return None, None
    count = 0
    probe = due_dt
    last = None
    iterations = 0
    for _ in range(_MAX_PREVIEW_ITERATIONS):
        if iterations >= _MAX_ITERATIONS:
            break
        iterations += 1
        if probe > until_dt:
            break
        last = probe
        count += 1
        probe = add_period(probe)
    if count <= 0:
        return None, None
    return max(0, count - 1), last


def _cp_preview_lines(due_dt: datetime, until_dt: datetime | None, limit: int, add_period) -> list[str]:
    def _fmt(dt):
        return core.fmt_dt_local(dt)

    preview = []
    nxt = due_dt
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    for i in range(limit):
        nxt = add_period(nxt)
        if until_dt and nxt > until_dt:
            break
        color = colors[min(i, len(colors) - 1)]
        preview.append(f"[{color}]{_fmt(nxt)}[/{color}]")
    return preview


def _cp_limit_rows(
    rows: list[tuple[str, str]],
    *,
    cpmax: int,
    due_dt: datetime,
    until_dt: datetime | None,
    exact_until_count: int | None,
    final_until_dt: datetime | None,
    add_period,
    now_utc: datetime,
) -> None:
    def _fmt(dt):
        return core.fmt_dt_local(dt)

    lim_parts = []
    if cpmax and cpmax > 0:
        lim_parts.append(f"max [bold yellow]{cpmax}[/]")
        fmax = due_dt
        steps = max(0, cpmax - 1)
        for _ in range(steps):
            fmax = add_period(fmax)
        rows.append(
            (
                "Final (max)",
                f"[bright_magenta]{_fmt(fmax)}[/]  [dim]({_human_delta(now_utc, fmax, True)})[/]",
            )
        )

    if until_dt:
        lim_parts.append(f"until [bold yellow]{_fmt(until_dt)}[/]")
        if exact_until_count is not None:
            lim_parts.append(f"[white]{exact_until_count} more[/]")
        if final_until_dt:
            rows.append(
                (
                    "Final (until)",
                    f"[bright_magenta]{_fmt(final_until_dt)}[/]  [dim]({_human_delta(now_utc, final_until_dt, True)})[/]",
                )
            )

    if lim_parts:
        rows.append(("Limits", " [dim]|[/] ".join(lim_parts)))


def _handle_cp_preview_on_add(
    task: dict,
    cp_str: str,
    ch: str,
    now_utc: datetime,
    user_provided_due: bool,
    due_dt: datetime,
    until_dt: datetime | None,
) -> None:
    rows: list[tuple[str, str]] = []

    def _fmt(dt):
        return core.fmt_dt_local(dt)

    def _dt(s):
        return core.parse_dt_any(s)

    td, err = _safe_parse_duration(cp_str, "cp")
    if err:
        _error_and_exit([("Invalid cp", err)])
    if not td:
        _error_and_exit([("Invalid cp", f"Couldn't parse duration from '{cp_str}'")])

    if until_dt:
        is_reasonable, warn_msg = _validate_chain_duration_reasonable(
            until_dt, now_utc, now_utc + td if not user_provided_due else due_dt, "cp"
        )
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    add_period = _cp_add_period_builder(td)
    entry_dt = _dt(task.get("entry")) if task.get("entry") else now_utc

    if not user_provided_due:
        due_dt = add_period(entry_dt)
        task["due"] = _fmt_local_for_task(due_dt)
        rows.append(("[auto-due]", "Due was not set explicitly; assigned to entry+cp."))
    else:
        due_dt = _dt(task.get("due"))

    rows.append(("Period", f"[bold white]{cp_str}[/]"))
    rows.append(("First due", f"[bold bright_green]{_fmt(due_dt)}[/]"))

    # Scheduled/Wait preview (relative to First due)
    _append_wait_sched_rows(rows, task, due_dt, auto_due=(not user_provided_due))

    cpmax = core.coerce_int(task.get("chainMax"), 0)
    exact_until_count, final_until_dt = _cp_until_summary(due_dt, until_dt, add_period)

    allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
    allow_by_until = exact_until_count if exact_until_count is not None else 10**9
    limit = max(0, min(UPCOMING_PREVIEW, allow_by_max, allow_by_until, _PREVIEW_HARD_CAP))

    preview = _cp_preview_lines(due_dt, until_dt, limit, add_period)
    rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
    rows.append(("Delta", f"[bright_yellow]{_human_delta(now_utc, due_dt, False)}[/]"))

    _cp_limit_rows(
        rows,
        cpmax=cpmax,
        due_dt=due_dt,
        until_dt=until_dt,
        exact_until_count=exact_until_count,
        final_until_dt=final_until_dt,
        add_period=add_period,
        now_utc=now_utc,
    )

    rows.append(("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]"))
    rows = _format_cp_rows(rows)
    _panel(
        "⛓ Recurring Chain Preview",
        rows,
        kind="preview_cp",
    )
    if core.SANITIZE_UDA:
        core.sanitize_task_strings(task, max_len=core.SANITIZE_UDA_MAX_LEN)
    print(json.dumps(task, ensure_ascii=False), end="")
    sys.stdout.flush()


def _anchor_preview_prepare_dnf(task: dict[str, object], anchor_str: str, due_dt: datetime, rows: list[tuple[str, str]], prof) -> tuple[list[list[dict]], str]:
    add_anchor_preview = _module("add_anchor_preview")
    return add_anchor_preview.anchor_preview_prepare_dnf(
        task,
        anchor_str,
        due_dt,
        rows,
        prof,
        core=core,
        validate_anchor_syntax_strict=_validate_anchor_syntax_strict,
        validate_anchor_mode=_validate_anchor_mode,
        error_and_exit=_error_and_exit,
    )


def _anchor_preview_seed_context(task: dict, due_day, now_local: datetime, user_provided_due: bool):
    add_anchor_preview = _module("add_anchor_preview")
    return add_anchor_preview.anchor_preview_seed_context(
        task,
        due_day,
        now_local,
        user_provided_due,
        root_uuid_from=_root_uuid_from,
    )


def _anchor_preview_first_due(
    task: dict,
    dnf,
    *,
    now_local: datetime,
    due_dt: datetime,
    user_provided_due: bool,
    due_hhmm: tuple[int, int],
    interval_seed,
    seed_base: str,
    rows: list[tuple[str, str]],
    prof,
):
    add_anchor_preview = _module("add_anchor_preview")
    return add_anchor_preview.anchor_preview_first_due(
        task,
        dnf,
        now_local=now_local,
        due_dt=due_dt,
        user_provided_due=user_provided_due,
        due_hhmm=due_hhmm,
        interval_seed=interval_seed,
        seed_base=seed_base,
        rows=rows,
        prof=prof,
        core=core,
        to_local_cached=_to_local_cached,
        anchor_pick_occurrence_local=_anchor_pick_occurrence_local,
        error_and_exit=_error_and_exit,
        fmt_local_for_task=_fmt_local_for_task,
    )


def _anchor_preview_misaligned_due_warning(
    rows: list[tuple[str, str]],
    *,
    dnf,
    due_dt: datetime,
    interval_seed,
    seed_base: str,
) -> None:
    add_anchor_preview = _module("add_anchor_preview")
    add_anchor_preview.anchor_preview_misaligned_due_warning(
        rows,
        dnf=dnf,
        due_dt=due_dt,
        interval_seed=interval_seed,
        seed_base=seed_base,
        to_local_cached=_to_local_cached,
        anchor_step_once=_anchor_step_once,
    )


def _anchor_preview_lint_and_validate(anchor_str: str, prof) -> None:
    add_anchor_preview = _module("add_anchor_preview")
    add_anchor_preview.anchor_preview_lint_and_validate(
        anchor_str,
        prof,
        core=core,
        panel=_panel,
    )


def _anchor_preview_limit_rows(
    rows: list[tuple[str, str]],
    *,
    cpmax: int,
    until_dt: datetime | None,
    exact_until_count: int | None,
    final_until_dt: datetime | None,
    now_utc: datetime,
) -> None:
    add_anchor_preview = _module("add_anchor_preview")
    add_anchor_preview.anchor_preview_limit_rows(
        rows,
        cpmax=cpmax,
        until_dt=until_dt,
        exact_until_count=exact_until_count,
        final_until_dt=final_until_dt,
        now_utc=now_utc,
        core=core,
        human_delta=_human_delta,
    )


def _handle_anchor_preview_on_add(
    task: dict,
    anchor_str: str,
    ch: str,
    now_utc: datetime,
    now_local: datetime,
    user_provided_due: bool,
    due_dt: datetime,
    due_day,
    due_hhmm: tuple[int, int],
    until_dt: datetime | None,
    past_due_warning: str | None,
    prof,
) -> None:
    add_anchor_preview = _module("add_anchor_preview")
    add_anchor_preview.handle_anchor_preview_on_add(
        task=task,
        anchor_str=anchor_str,
        ch=ch,
        now_utc=now_utc,
        now_local=now_local,
        user_provided_due=user_provided_due,
        due_dt=due_dt,
        due_day=due_day,
        due_hhmm=due_hhmm,
        until_dt=until_dt,
        past_due_warning=past_due_warning,
        prof=prof,
        anchor_warn=ANCHOR_WARN,
        upcoming_preview=UPCOMING_PREVIEW,
        preview_hard_cap=_PREVIEW_HARD_CAP,
        core=core,
        root_uuid_from=_root_uuid_from,
        short=_short,
        validate_anchor_syntax_strict=_validate_anchor_syntax_strict,
        validate_anchor_mode=_validate_anchor_mode,
        validate_chain_duration_reasonable=_validate_chain_duration_reasonable,
        append_wait_sched_rows=_append_wait_sched_rows,
        anchor_step_once=_anchor_step_once,
        anchor_pick_occurrence_local=_anchor_pick_occurrence_local,
        anchor_until_summary=_anchor_until_summary,
        anchor_build_preview=_anchor_build_preview,
        to_local_cached=_to_local_cached,
        fmt_local_for_task=_fmt_local_for_task,
        format_anchor_rows=_format_anchor_rows,
        panel=_panel,
        emit_task_json=_emit_task_json,
        human_delta=_human_delta,
        error_and_exit=_error_and_exit,
    )


class _NoopProfiler:
    enabled = False

    @contextmanager
    def section(self, _name):
        yield

    def add_ms(self, _name, _ms):
        return None

    def emit(self):
        return None


def _build_profiler():
    prof = _NoopProfiler()
    if _PROFILE_LEVEL <= 0:
        return prof
    prof = _Profiler(level=_PROFILE_LEVEL, import_ms=_IMPORT_MS)
    if prof.enabled:
        atexit.register(prof.emit)
    return prof


def _read_on_add_task(prof) -> dict:
    with prof.section("read:stdin"):
        raw_bytes = sys.stdin.buffer.read(_MAX_JSON_BYTES + 1)
        raw_text = raw_bytes.decode("utf-8", errors="replace")
        global _RAW_INPUT_TEXT
        _RAW_INPUT_TEXT = raw_text
        if len(raw_bytes) > _MAX_JSON_BYTES:
            _fail_and_exit("Invalid input", f"on-add input exceeds {_MAX_JSON_BYTES} bytes")
        raw = raw_text.strip()
    if not raw:
        _fail_and_exit("Invalid input", "on-add must receive a single JSON task")
    try:
        with prof.section("parse:json"):
            task = json.loads(raw)
            global _PARSED_TASK
            _PARSED_TASK = task
            return task
    except Exception:
        _fail_and_exit("Invalid input", "on-add must receive a single JSON task")
    return {}


def _emit_task_json(task: dict, *, sanitize: bool = False, prof=None) -> None:
    hook_results = _module("hook_results")
    hook_results.emit_task_json(task, sanitize=sanitize, core=core, prof=prof)


def _build_hook_runtime_context():
    hook_context = _module("hook_context")
    return hook_context.build_hook_runtime_context(
        hook_name="on-add",
        taskdata_dir=str(TW_DATA_DIR),
        use_rc_data_location=_USE_RC_DATA_LOCATION,
        tw_dir=str(TW_DIR),
        hook_dir=str(HOOK_DIR),
        profile_level=_PROFILE_LEVEL,
        import_ms=_IMPORT_MS,
    )


def _build_on_add_context(task: dict, now_utc: datetime, now_local: datetime, *, prof=None):
    hook_context = _module("hook_context")
    _t_conf = time.perf_counter()
    try:
        return hook_context.build_on_add_context(
            task,
            now_utc,
            now_local,
            validate_kind_not_conflicting=_validate_kind_not_conflicting,
            kind_and_defaults_on_add=_kind_and_defaults_on_add,
            validate_chain_limits_on_add=_validate_chain_limits_on_add,
            due_context_on_add=_due_context_on_add,
        )
    except ValueError as exc:
        _error_and_exit([('Invalid chain config', str(exc))])
        raise
    finally:
        if prof is not None:
            prof.add_ms('validate:cp_vs_anchor', (time.perf_counter() - _t_conf) * 1000.0)


def _handle_anchor_preview_on_add_context(ctx, *, prof) -> None:
    _handle_anchor_preview_on_add(
        task=ctx.task,
        anchor_str=ctx.anchor_str,
        ch=ctx.chain_state,
        now_utc=ctx.now_utc,
        now_local=ctx.now_local,
        user_provided_due=ctx.user_provided_due,
        due_dt=ctx.due_dt,
        due_day=ctx.due_day,
        due_hhmm=ctx.due_hhmm,
        until_dt=ctx.until_dt,
        past_due_warning=ctx.past_due_warning,
        prof=prof,
    )


def _handle_cp_preview_on_add_context(ctx, *, prof) -> None:
    _handle_cp_preview_on_add(
        task=ctx.task,
        cp_str=ctx.cp_str,
        ch=ctx.chain_state,
        now_utc=ctx.now_utc,
        user_provided_due=ctx.user_provided_due,
        due_dt=ctx.due_dt,
        until_dt=ctx.until_dt,
    )


def _kind_and_defaults_on_add(task: dict, cp_str: str, anchor_str: str) -> tuple[str | None, str]:
    has_cp = bool(cp_str)
    has_anchor = bool(anchor_str)
    kind = "anchor" if has_anchor else ("cp" if has_cp else None)

    ch = (task.get("chain") or "").strip().lower()
    if (has_cp or has_anchor) and (not ch or ch == "off"):
        task["chain"] = "on"
        ch = "on"

    if has_cp or has_anchor:
        linked_already = bool((task.get("prevLink") or "").strip() or (task.get("nextLink") or "").strip())
        if not linked_already:
            link_no = core.coerce_int(task.get("link"), 0)
            if link_no <= 0:
                task["link"] = 1
    return kind, ch


def _validate_chain_limits_on_add(task: dict, now_utc: datetime) -> datetime | None:
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if cpmax:
        is_valid, err = _validate_cpmax_positive(cpmax)
        if not is_valid:
            _error_and_exit([("Invalid chainMax", err)])

    until_dt, err = _safe_parse_datetime(task.get("chainUntil"), "chainUntil")
    if err:
        _error_and_exit([("Invalid chainUntil", err)])
    if until_dt:
        is_valid, err = _validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            _error_and_exit([("Invalid chainUntil", err)])
    return until_dt


def _due_context_on_add(task: dict, now_utc: datetime) -> tuple[bool, datetime, str | None, object, tuple[int, int]]:
    user_provided_due = bool(task.get("due"))
    due_dt = None
    past_due_warning = None
    if user_provided_due:
        due_dt, err = _safe_parse_datetime(task.get("due"), "due")
        if err:
            _error_and_exit([("Invalid due", err)])
        is_past, warn_msg = _check_due_in_past(due_dt, now_utc)
        if is_past:
            past_due_warning = warn_msg
    if due_dt is None:
        due_dt = now_utc

    due_local = core.to_local(due_dt)
    due_day = due_local.date()
    due_hhmm = (due_local.hour, due_local.minute)
    return user_provided_due, due_dt, past_due_warning, due_day, due_hhmm


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    prof = _build_profiler()
    task = _read_on_add_task(prof)
    hook_engine = _module("hook_engine")
    hook_engine.handle_on_add(
        task,
        prof=prof,
        runtime=_build_hook_runtime_context(),
        core_ref=lambda: core,
        task_has_nautical_fields=_task_has_nautical_fields,
        load_core=_load_core,
        diag=_diag,
        fail_and_exit=_fail_and_exit,
        emit_task_json=_emit_task_json,
        build_on_add_context=_build_on_add_context,
        stamp_chain_id_on_add=_stamp_chain_id_on_add,
        handle_anchor_preview_on_add=_handle_anchor_preview_on_add_context,
        handle_cp_preview_on_add=_handle_cp_preview_on_add_context,
    )



# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                sys.stderr.write(f"[nautical] on-add unexpected error: {e}\n")
            except Exception:
                pass
        _panic_passthrough()
        raise SystemExit(1)
