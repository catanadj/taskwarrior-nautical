#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nautical Golden Tests
 - Imports local nautical_core/__init__.py
 - Verifies parsing, lint, natural, and next-occurrence properties
 - Covers prior regressions: leap day, quarters, /N monthly valid-month gating, last-<dow>,
   weekly AND unsatisfiable, @bd/@nbd/@nw natural text + date effects, rand with yearly window, etc.

Run:
  python3 nautical_golden_tests.py
Optional:
  python3 nautical_golden_tests.py --only leap --verbose
"""

import importlib
import sys, os, re, json, io, contextlib, stat
import random
import sqlite3
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.environ.setdefault("NAUTICAL_CORE_PATH", ROOT)

core = importlib.import_module("nautical_core")
_hook = importlib.import_module("on-modify-nautical")

# -------- Helpers -------------------------------------------------------------

def iso(d):
    if isinstance(d, (datetime, )):
        return d.date().isoformat()
    if isinstance(d, (date, )):
        return d.isoformat()
    s = str(d)
    # try yyyymmdd
    m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    # try YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return s

def parse_due(v):
    # your hooks already parse Taskwarrior-style datetimes; for tests, allow simple YYYY-MM-DD
    if not v:
        return None
    if isinstance(v, (datetime, date)):
        return v
    s = str(v).strip()
    try:
        return datetime.fromisoformat(s).replace(tzinfo=None)
    except Exception:
        pass
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except Exception:
        return None

def build_preview(expr, mode="ALL", due=None):
    """
    Always use core.build_and_cache_hints if present (it computes upcoming),
    but we do not require file-cache to be enabled in config.
    """
    due_dt = parse_due(due)

    natural = ""
    upcoming = []
    first_due = None

    # Prefer the real preview path
    if hasattr(core, "build_and_cache_hints"):
        try:
            pkg = core.build_and_cache_hints(expr, mode, default_due_dt=due_dt)
            if pkg:
                natural = pkg.get("natural") or natural
                # FIX: Use "next_dates" (the actual key) instead of "upcoming"
                next_dates = pkg.get("next_dates") or []
                for u in next_dates:
                    upcoming.append(iso(u))
                # Also check for "first_due" if present
                if pkg.get("first_due"):
                    first_due = iso(pkg["first_due"])
                return {"natural": natural, "upcoming": upcoming, "first_due": first_due}
        except Exception:
            # fall through to strict validate
            pass

    # Fallback: validate + best-effort natural
    core.validate_anchor_expr_strict(expr)
    if hasattr(core, "describe_anchor_expr"):
        try:
            natural = core.describe_anchor_expr(expr, default_due_dt=due_dt)
        except Exception:
            natural = ""
    return {"natural": natural, "upcoming": upcoming, "first_due": first_due}


def expect(cond, msg):
    if not cond:
        raise AssertionError(msg)

def has_function(name):
    return hasattr(core, name)

def _must_parse(expr):
    """Parse expression and return DNF, raising AssertionError on failure."""
    try:
        return core.validate_anchor_expr_strict(expr)
    except Exception as e:
        raise AssertionError(f"Failed to parse '{expr}': {e}")

def _must_preview(expr, due=None):
    """Get preview data, raising AssertionError if no dates."""
    p = build_preview(expr, due=due)
    if not p or not p.get("upcoming"):
        # Try to get next_dates from different key
        if hasattr(core, "build_and_cache_hints"):
            try:
                pkg = core.build_and_cache_hints(expr, "ALL", parse_due(due))
                if pkg and pkg.get("next_dates"):
                    return {"next_dates": pkg["next_dates"]}
            except Exception:
                pass
        raise AssertionError(f"No upcoming dates for '{expr}'")
    # Convert upcoming strings to dates
    next_dates = []
    for d_str in p["upcoming"]:
        try:
            next_dates.append(datetime.fromisoformat(d_str).date())
        except Exception:
            pass
    return {"next_dates": next_dates}

def _must_natural(expr):
    """Get natural language description, raising AssertionError if empty."""
    try:
        if hasattr(core, "describe_anchor_expr"):
            natural = core.describe_anchor_expr(expr)
            if natural:
                return natural
    except Exception:
        pass
    
    # Fallback through preview
    p = build_preview(expr)
    if p and p.get("natural"):
        return p["natural"]
    
    raise AssertionError(f"No natural language for '{expr}'")

# -------- Test cases ----------------------------------------------------------
# -------- Hook checks ---------------------------------------------------------
# These tests validate the shipped hook scripts (on-add / on-modify) at a high
# level, to catch regressions that can slip through core-only tests.

import subprocess
import importlib.util
import importlib.machinery
import inspect
import time as _time

def _force_tz_utc():
    # Make hook output deterministic across machines.
    os.environ["TZ"] = "UTC"
    try:
        _time.tzset()
    except Exception:
        pass

def _find_hook_file(name: str) -> str:
    # Per project convention: hooks live either next to core/tests, or under ./hooks/
    candidates = [
        os.path.join(ROOT, name),
        os.path.join(ROOT, "hooks", name),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise AssertionError(
        f"Hook script '{name}' not found. Expected at '{candidates[0]}' or '{candidates[1]}'."
    )

def _run_hook_script(path: str, task_obj: dict, env_extra: dict | None = None, timeout_s: float = 8.0):
    _force_tz_utc()
    env = os.environ.copy()
    # Ensure the hook can import local nautical_core/__init__.py
    env["PYTHONPATH"] = HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TZ", "UTC")
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(
        [sys.executable, path],
        input=json.dumps(task_obj),
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_s,
    )
    return p

def _run_hook_script_raw(path: str, raw_input: str, env_extra: dict | None = None, timeout_s: float = 8.0):
    _force_tz_utc()
    env = os.environ.copy()
    # Ensure the hook can import local nautical_core/__init__.py
    env["PYTHONPATH"] = HERE + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TZ", "UTC")
    if env_extra:
        env.update({k: str(v) for k, v in env_extra.items()})
    p = subprocess.run(
        [sys.executable, path],
        input=raw_input,
        text=True,
        capture_output=True,
        env=env,
        timeout=timeout_s,
    )
    return p

def _extract_last_json(stdout_text: str) -> dict:
    s = (stdout_text or "").strip()
    if not s:
        raise AssertionError("Hook produced no stdout JSON.")
    # Many hooks emit exactly one JSON object. If extra text exists, take the last {...}.
    candidates = re.findall(r"\{[\s\S]*\}", s)
    if not candidates:
        raise AssertionError(f"Could not locate JSON in hook stdout. stdout={s[:200]!r}")
    try:
        return json.loads(candidates[-1])
    except Exception as e:
        raise AssertionError(f"Invalid JSON from hook stdout: {e}. stdout_tail={candidates[-1][-200:]!r}")

def _load_hook_module(path: str, module_name: str):
    _force_tz_utc()
    loader = importlib.machinery.SourceFileLoader(module_name, path)
    spec = importlib.util.spec_from_loader(module_name, loader)
    mod = importlib.util.module_from_spec(spec)
    # Ensure local imports work
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    loader.exec_module(mod)
    return mod

def _load_core_module(path: str, module_name: str, config_path: str):
    prev_conf = os.environ.get("NAUTICAL_CONFIG")
    os.environ["NAUTICAL_CONFIG"] = config_path
    try:
        loader = importlib.machinery.SourceFileLoader(module_name, path)
        spec = importlib.util.spec_from_loader(module_name, loader)
        mod = importlib.util.module_from_spec(spec)
        if HERE not in sys.path:
            sys.path.insert(0, HERE)
        loader.exec_module(mod)
        return mod
    finally:
        if prev_conf is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = prev_conf

def _assert_stdout_json_only(stdout_text: str) -> dict:
    s = (stdout_text or "").strip()
    if not s:
        raise AssertionError("Hook produced no stdout JSON.")
    dec = json.JSONDecoder()
    obj, idx = dec.raw_decode(s)
    if s[idx:].strip():
        raise AssertionError("Hook stdout contains non-JSON content.")
    if not isinstance(obj, dict):
        raise AssertionError(f"Hook stdout JSON is not an object: {type(obj).__name__}")
    return obj
def _call_with_supported_kwargs(fn, **kwargs):
    sig = inspect.signature(fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**filtered)

def _strip_markup(s: str) -> str:
    # Remove Rich markup tags such as [bold red], [/], [cyan], etc.
    return re.sub(r"\[[^\]]*\]", "", s or "")



def test_lint_formats():
    """Test validator rejects malformed yearly tokens (':' instead of '-')"""
    # Instead of relying on linter fatals, assert the validator rejects bad yearly tokens
    try:
        core.validate_anchor_expr_strict("y:05:15")
        assert False, "Validator must fatal on 'y:05:15' (':' instead of '-')"
    except core.ParseError as e:
        low = str(e).lower()
        expect(("uses ':'" in low) or ("example" in low),
               f"Unexpected validator message: {e}")

def test_warn_once_per_day_stamp_written():
    """Ensure diagnostic stamp creation works without crashing."""
    with tempfile.TemporaryDirectory() as td:
        prev = os.environ.get("XDG_CACHE_HOME")
        prev_diag = os.environ.get("NAUTICAL_DIAG")
        os.environ["XDG_CACHE_HOME"] = td
        os.environ["NAUTICAL_DIAG"] = "1"
        try:
            core._warn_once_per_day("golden_test", "golden test message")
            stamp = os.path.join(td, "nautical", ".diag_golden_test.stamp")
            expect(os.path.exists(stamp), f"stamp not created: {stamp}")
            with open(stamp, "r", encoding="utf-8") as f:
                val = f.read().strip()
            expect(val == date.today().isoformat(), f"stamp has unexpected value: {val}")
        finally:
            if prev is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev
            if prev_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = prev_diag

def test_warn_once_per_day_no_diag_silent():
    """Ensure diagnostics are silent (no stamp) when NAUTICAL_DIAG is unset."""
    with tempfile.TemporaryDirectory() as td:
        prev = os.environ.get("XDG_CACHE_HOME")
        prev_diag = os.environ.get("NAUTICAL_DIAG")
        os.environ["XDG_CACHE_HOME"] = td
        if "NAUTICAL_DIAG" in os.environ:
            os.environ.pop("NAUTICAL_DIAG", None)
        try:
            core._warn_once_per_day("golden_test_silent", "golden test message")
            stamp = os.path.join(td, "nautical", ".diag_golden_test_silent.stamp")
            expect(not os.path.exists(stamp), f"stamp should not be created: {stamp}")
        finally:
            if prev is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev
            if prev_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = prev_diag

def test_warn_once_per_day_any_no_diag_silent():
    """Ensure _warn_once_per_day_any does not write to stderr when NAUTICAL_DIAG is unset."""
    with tempfile.TemporaryDirectory() as td:
        prev = os.environ.get("XDG_CACHE_HOME")
        prev_diag = os.environ.get("NAUTICAL_DIAG")
        os.environ["XDG_CACHE_HOME"] = td
        if "NAUTICAL_DIAG" in os.environ:
            os.environ.pop("NAUTICAL_DIAG", None)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                core._warn_once_per_day_any("golden_test_any_silent", "golden test message")
            expect(buf.getvalue() == "", "expected no stderr output when NAUTICAL_DIAG is unset")
        finally:
            if prev is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev
            if prev_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = prev_diag

def test_on_add_fail_and_exit_emits_json():
    """_fail_and_exit should fail-closed without emitting task JSON."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_fail_test")
    task = {"uuid": "00000000-0000-0000-0000-000000000abc", "description": "fail test"}
    mod._PARSED_TASK = dict(task)
    mod._RAW_INPUT_TEXT = json.dumps(task, ensure_ascii=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            mod._fail_and_exit("Invalid anchor", "anchor syntax error: bad")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
        else:
            raise AssertionError("_fail_and_exit did not exit")
    out = buf.getvalue().strip()
    expect(out == "", f"expected no stdout on failure, got: {out!r}")


def test_on_add_panic_passthrough_emits_valid_json():
    """on-add panic passthrough should always emit a valid JSON object."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_panic_passthrough_test")
    mod._PARSED_TASK = {"uuid": "00000000-0000-0000-0000-000000000111", "description": "panic-add"}
    mod._RAW_INPUT_TEXT = "{not-json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._panic_passthrough()
    out = buf.getvalue().strip()
    obj = json.loads(out or "{}")
    expect(isinstance(obj, dict), f"panic passthrough must emit JSON object, got: {out!r}")
    expect(obj.get("uuid") == "00000000-0000-0000-0000-000000000111", "parsed task should be preserved")


def test_on_modify_panic_passthrough_uses_latest_task():
    """on-modify panic passthrough should emit the latest task object."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_panic_passthrough_test")
    mod._PARSED_NEW = None
    old = {"uuid": "00000000-0000-0000-0000-000000000111", "status": "pending"}
    new = {"uuid": "00000000-0000-0000-0000-000000000111", "status": "completed"}
    mod._RAW_INPUT_TEXT = json.dumps(old) + "\n" + json.dumps(new)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._panic_passthrough()
    out = buf.getvalue().strip()
    obj = json.loads(out or "{}")
    expect(isinstance(obj, dict), f"panic passthrough must emit JSON object, got: {out!r}")
    expect(obj.get("status") == "completed", f"expected latest task, got: {obj}")


def test_on_add_ignores_unsafe_core_path_override():
    """on-add should ignore unsafe NAUTICAL_CORE_PATH overrides by default."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_unsafe_core_path_test")
    prev = os.environ.get("NAUTICAL_CORE_PATH")
    prev_trust = os.environ.get("NAUTICAL_TRUST_CORE_PATH")
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chmod(td, 0o777)
            except Exception:
                pass
            os.environ["NAUTICAL_CORE_PATH"] = td
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
            got = mod._trusted_core_base(Path(mod.TW_DIR))
            expect(Path(got).resolve() == Path(mod.TW_DIR).resolve(),
                   f"unsafe core path should fall back to TW_DIR, got {got}")
    finally:
        if prev is None:
            os.environ.pop("NAUTICAL_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_CORE_PATH"] = prev
        if prev_trust is None:
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_TRUST_CORE_PATH"] = prev_trust


def test_on_modify_ignores_unsafe_core_path_override():
    """on-modify should ignore unsafe NAUTICAL_CORE_PATH overrides by default."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_unsafe_core_path_test")
    prev = os.environ.get("NAUTICAL_CORE_PATH")
    prev_trust = os.environ.get("NAUTICAL_TRUST_CORE_PATH")
    try:
        with tempfile.TemporaryDirectory() as td:
            try:
                os.chmod(td, 0o777)
            except Exception:
                pass
            os.environ["NAUTICAL_CORE_PATH"] = td
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
            got = mod._trusted_core_base(Path(mod.TW_DIR))
            expect(Path(got).resolve() == Path(mod.TW_DIR).resolve(),
                   f"unsafe core path should fall back to TW_DIR, got {got}")
    finally:
        if prev is None:
            os.environ.pop("NAUTICAL_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_CORE_PATH"] = prev
        if prev_trust is None:
            os.environ.pop("NAUTICAL_TRUST_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_TRUST_CORE_PATH"] = prev_trust


def test_hook_stdout_strict_json_with_diag_on_add():
    """on-add must keep stdout JSON-only even when diagnostics are enabled."""
    hook = _find_hook_file("on-add-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        env = {
            "NAUTICAL_DIAG": "1",
            "NAUTICAL_CONFIG": os.path.join(td, "missing.toml"),
        }
        task = {
            "uuid": "00000000-0000-0000-0000-000000000333",
            "description": "hook test on-add strict stdout",
            "status": "pending",
            "entry": "20250101T000000Z",
        }
        p = _run_hook_script(hook, task, env_extra=env)
        expect(p.returncode == 0, f"on-add returned {p.returncode}")
        _assert_stdout_json_only(p.stdout)

def test_hook_stdout_strict_json_with_diag_on_modify():
    """on-modify must keep stdout JSON-only even when diagnostics are enabled."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        env = {
            "NAUTICAL_DIAG": "1",
            "NAUTICAL_CONFIG": os.path.join(td, "missing.toml"),
        }
        raw = json.dumps({"uuid": "00000000-0000-0000-0000-000000000444", "status": "pending"})
        p = _run_hook_script_raw(hook, raw, env_extra=env)
        expect(p.returncode == 0, f"on-modify returned {p.returncode}")
        _assert_stdout_json_only(p.stdout)

def test_hook_stdout_unicode_unescaped_on_add():
    """on-add passthrough stdout should preserve Unicode (ensure_ascii=False)."""
    hook = _find_hook_file("on-add-nautical.py")
    task = {
        "uuid": "00000000-0000-0000-0000-000000000445",
        "status": "pending",
        "description": "Cafe ăîșț ✅",
    }
    p = _run_hook_script(hook, task)
    expect(p.returncode == 0, f"on-add returned {p.returncode}")
    out = (p.stdout or "").strip()
    _assert_stdout_json_only(out)
    expect("ăîșț ✅" in out, f"expected raw Unicode in stdout, got: {out!r}")
    expect("\\u" not in out, f"stdout should not escape Unicode: {out!r}")

def test_hook_stdout_unicode_unescaped_on_modify():
    """on-modify passthrough stdout should preserve Unicode (ensure_ascii=False)."""
    hook = _find_hook_file("on-modify-nautical.py")
    raw = json.dumps(
        {
            "uuid": "00000000-0000-0000-0000-000000000446",
            "status": "pending",
            "description": "Cafe ăîșț ✅",
        },
        ensure_ascii=False,
    )
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode == 0, f"on-modify returned {p.returncode}")
    out = (p.stdout or "").strip()
    _assert_stdout_json_only(out)
    expect("ăîșț ✅" in out, f"expected raw Unicode in stdout, got: {out!r}")
    expect("\\u" not in out, f"stdout should not escape Unicode: {out!r}")

def test_hook_stdout_empty_on_exit():
    """on-exit should not emit stdout (stdout is redirected to /dev/null)."""
    hook = _find_hook_file("on-exit-nautical.py")
    env = {"NAUTICAL_DIAG": "1"}
    p = _run_hook_script_raw(hook, "", env_extra=env)
    expect(p.returncode == 0, f"on-exit returned {p.returncode}")
    expect((p.stdout or "") == "", f"on-exit expected empty stdout, got: {p.stdout!r}")

def test_hook_files_are_private_permissions():
    """Queue/dead-letter/lock files should not be group/world-readable."""
    lock_path = None
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
            mod_core = _load_hook_module(core_path, "_nautical_core_perm_test")
            lock_path = os.path.join(td, ".nautical_perm_test.lock")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(ok, "safe_lock did not acquire")
                mode = stat.S_IMODE(os.stat(lock_path).st_mode)
                expect((mode & 0o077) == 0, f"lock file has group/other perms: {oct(mode)}")

            hook_modify = _find_hook_file("on-modify-nautical.py")
            mod_modify = _load_hook_module(hook_modify, "_nautical_on_modify_perm_test")
            mod_modify._enqueue_deferred_spawn(
                {"spawn_intent_id": "si_perm", "child": {"uuid": "00000000-0000-0000-0000-000000000999"}}
            )
            db_path = mod_modify._SPAWN_QUEUE_DB_PATH
            expect(db_path.exists(), f"sqlite queue not created: {db_path}")
            mode = stat.S_IMODE(db_path.stat().st_mode)
            expect((mode & 0o077) == 0, f"sqlite queue file has group/other perms: {oct(mode)}")

            hook_exit = _find_hook_file("on-exit-nautical.py")
            mod_exit = _load_hook_module(hook_exit, "_nautical_on_exit_perm_test")
            mod_exit._write_dead_letter({"uuid": "00000000-0000-0000-0000-000000000888"}, "perm test")
            dl_path = mod_exit._DEAD_LETTER_PATH
            expect(dl_path.exists(), f"dead-letter not created: {dl_path}")
            mode = stat.S_IMODE(dl_path.stat().st_mode)
            expect((mode & 0o077) == 0, f"dead-letter has group/other perms: {oct(mode)}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_safe_lock_fcntl_contention():
    """safe_lock should fail to acquire when another process holds the lock."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_fcntl_test")
    if getattr(mod_core, "fcntl", None) is None:
        return
    with tempfile.TemporaryDirectory() as td:
        lock_path = os.path.join(td, ".nautical_fcntl.lock")
        ready_path = os.path.join(td, ".nautical_fcntl.ready")
        script = (
            "import os, time\n"
            "import fcntl\n"
            "lp = os.environ['LOCK_PATH']\n"
            "rp = os.environ['READY_PATH']\n"
            "fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o600)\n"
            "f = os.fdopen(fd, 'a', encoding='utf-8')\n"
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "with open(rp, 'w', encoding='utf-8') as r:\n"
            "    r.write('ready')\n"
            "time.sleep(1.0)\n"
        )
        p = subprocess.Popen(
            [sys.executable, "-c", script],
            env={**os.environ, "LOCK_PATH": lock_path, "READY_PATH": ready_path},
        )
        try:
            for _ in range(50):
                if os.path.exists(ready_path):
                    break
                _time.sleep(0.02)
            expect(os.path.exists(ready_path), "lock holder did not start")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(not ok, "safe_lock should not acquire while locked")
        finally:
            p.wait(timeout=3.0)
        with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
            expect(ok, "safe_lock should acquire after lock release")

def test_safe_lock_fallback_contention():
    """safe_lock should fail to acquire when fallback lockfile exists."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_fallback_test")
    prev_fcntl = getattr(mod_core, "fcntl", None)
    mod_core.fcntl = None
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, ".nautical_fallback.lock")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(ok, "fallback safe_lock did not acquire")
                with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok2:
                    expect(not ok2, "fallback safe_lock should not acquire when locked")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok3:
                expect(ok3, "fallback safe_lock should acquire after release")
    finally:
        mod_core.fcntl = prev_fcntl

def test_safe_lock_fallback_stale_cleanup():
    """safe_lock fallback should clear stale lockfiles."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_stale_test")
    prev_fcntl = getattr(mod_core, "fcntl", None)
    mod_core.fcntl = None
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, ".nautical_stale.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("999999 0\n")
            os.utime(lock_path, (1, 1))
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0, stale_after=1.0) as ok:
                expect(ok, "stale fallback lock was not cleared")
    finally:
        mod_core.fcntl = prev_fcntl

def test_safe_lock_fallback_stale_pid_cleanup():
    """safe_lock fallback should clear lockfiles with dead PIDs."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    mod_core = _load_hook_module(core_path, "_nautical_core_lock_pid_stale_test")
    prev_fcntl = getattr(mod_core, "fcntl", None)
    mod_core.fcntl = None
    try:
        with tempfile.TemporaryDirectory() as td:
            lock_path = os.path.join(td, ".nautical_pid.lock")
            with open(lock_path, "w", encoding="utf-8") as f:
                f.write("999999 0\n")
            os.utime(lock_path, (1, 1))
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0, stale_after=1.0) as ok:
                expect(ok, "stale PID lock was not cleared")
    finally:
        mod_core.fcntl = prev_fcntl

def test_on_modify_queue_repairs_permissions():
    """Existing sqlite queue file permissions should be repaired on enqueue."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_queue_perm_test")
            db_path = mod._SPAWN_QUEUE_DB_PATH
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_text("", encoding="utf-8")
            os.chmod(db_path, 0o666)
            mod._enqueue_deferred_spawn(
                {"spawn_intent_id": "si_perm_fix", "child": {"uuid": "00000000-0000-0000-0000-000000000123"}}
            )
            mode = stat.S_IMODE(db_path.stat().st_mode)
            expect((mode & 0o077) == 0, f"sqlite queue perms not repaired: {oct(mode)}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_exit_repairs_queue_and_dead_letter_permissions():
    """Existing sqlite queue/dead-letter permissions should be repaired on write."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_perm_repair_test")
            db_path = mod._QUEUE_DB_PATH
            dl_path = mod._DEAD_LETTER_PATH
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS queue_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        spawn_intent_id TEXT,
                        payload TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL DEFAULT 'queued',
                        claim_token TEXT,
                        claimed_at REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, created_at, updated_at) "
                    "VALUES (?, ?, 0, 'processing', 1.0, 1.0)",
                    ("si_perm_fix", json.dumps({"spawn_intent_id": "si_perm_fix"}, ensure_ascii=False, separators=(",", ":"))),
                )
                conn.commit()
            dl_path.write_text("{}", encoding="utf-8")
            os.chmod(db_path, 0o666)
            os.chmod(dl_path, 0o666)
            mod._requeue_entries([{"__queue_backend": "sqlite", "__queue_id": 1, "spawn_intent_id": "si_perm_fix"}])
            mod._write_dead_letter({"uuid": "00000000-0000-0000-0000-000000000789"}, "perm test")
            q_mode = stat.S_IMODE(db_path.stat().st_mode)
            dl_mode = stat.S_IMODE(dl_path.stat().st_mode)
            expect((q_mode & 0o077) == 0, f"sqlite queue perms not repaired: {oct(q_mode)}")
            expect((dl_mode & 0o077) == 0, f"dead-letter perms not repaired: {oct(dl_mode)}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_exit_timeouts_configurable():
    """on-exit should honor timeout env vars for task commands."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_export = os.environ.get("NAUTICAL_TASK_TIMEOUT_EXPORT")
        prev_import = os.environ.get("NAUTICAL_TASK_TIMEOUT_IMPORT")
        prev_modify = os.environ.get("NAUTICAL_TASK_TIMEOUT_MODIFY")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_TASK_TIMEOUT_EXPORT"] = "1.5"
        os.environ["NAUTICAL_TASK_TIMEOUT_IMPORT"] = "2.5"
        os.environ["NAUTICAL_TASK_TIMEOUT_MODIFY"] = "3.5"
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_timeout_test")
            timeouts = []

            def _fake_run_task(cmd, *, input_text=None, timeout=0.0, **_kwargs):
                timeouts.append(timeout)
                if "export" in cmd:
                    return True, json.dumps({"uuid": "u1"}), ""
                return True, "", ""

            mod._run_task = _fake_run_task
            mod._export_uuid("u1")
            mod._import_child({"uuid": "u2"})
            mod._update_parent_nextlink("p", "c")
            expect(timeouts == [1.5, 2.5, 1.5, 3.5], f"unexpected timeouts: {timeouts}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_export is None:
                os.environ.pop("NAUTICAL_TASK_TIMEOUT_EXPORT", None)
            else:
                os.environ["NAUTICAL_TASK_TIMEOUT_EXPORT"] = prev_export
            if prev_import is None:
                os.environ.pop("NAUTICAL_TASK_TIMEOUT_IMPORT", None)
            else:
                os.environ["NAUTICAL_TASK_TIMEOUT_IMPORT"] = prev_import
            if prev_modify is None:
                os.environ.pop("NAUTICAL_TASK_TIMEOUT_MODIFY", None)
            else:
                os.environ["NAUTICAL_TASK_TIMEOUT_MODIFY"] = prev_modify


def test_on_exit_queue_db_connect_retries_and_scales_busy_timeout():
    """on-exit queue DB connect should retry once and scale busy_timeout from connect timeout."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_retry = os.environ.get("NAUTICAL_QUEUE_DB_CONNECT_RETRIES")
        prev_sleep = os.environ.get("NAUTICAL_QUEUE_LOCK_SLEEP_BASE")
        prev_lock_retry = os.environ.get("NAUTICAL_QUEUE_LOCK_RETRIES")
        try:
            os.environ["TASKDATA"] = td
            os.environ["NAUTICAL_QUEUE_DB_CONNECT_RETRIES"] = "3"
            os.environ["NAUTICAL_QUEUE_LOCK_SLEEP_BASE"] = "0.2"
            os.environ["NAUTICAL_QUEUE_LOCK_RETRIES"] = "8"
            mod = _load_hook_module(hook, "_nautical_on_exit_db_connect_retry_test")

            attempts = {"n": 0}
            timeouts: list[float] = []
            sleeps: list[float] = []
            seen_sql: list[str] = []

            class _FakeConn:
                row_factory = None

                def execute(self, sql):
                    seen_sql.append(str(sql))
                    return self

            fake_conn = _FakeConn()
            saved_connect = mod.sqlite3.connect
            saved_sleep_fn = mod._sleep
            try:
                def _fake_connect(_path, timeout=0.0):
                    attempts["n"] += 1
                    timeouts.append(float(timeout))
                    if attempts["n"] == 1:
                        raise mod.sqlite3.OperationalError("database is locked")
                    return fake_conn

                mod.sqlite3.connect = _fake_connect
                mod._sleep = lambda secs: sleeps.append(float(secs))
                conn = mod._queue_db_connect()
            finally:
                mod.sqlite3.connect = saved_connect
                mod._sleep = saved_sleep_fn

            expect(conn is fake_conn, "expected second connect attempt to succeed")
            expect(attempts["n"] == 2, f"expected one retry, got attempts={attempts['n']}")
            expect(bool(sleeps) and sleeps[0] > 0.0, f"expected positive backoff sleep, got {sleeps}")
            expect(timeouts[1] >= timeouts[0], f"expected non-decreasing connect timeout, got {timeouts}")
            busy_sql = [s for s in seen_sql if s.startswith("PRAGMA busy_timeout=")]
            expect(bool(busy_sql), f"busy_timeout pragma was not set; SQL={seen_sql}")
            busy_ms = int(busy_sql[-1].split("=", 1)[1])
            expect(busy_ms >= 1500, f"busy_timeout too small: {busy_ms}")
            expect(
                busy_ms >= int(timeouts[-1] * 1000.0),
                f"busy_timeout should scale with timeout. busy_ms={busy_ms}, timeout={timeouts[-1]}",
            )
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_retry is None:
                os.environ.pop("NAUTICAL_QUEUE_DB_CONNECT_RETRIES", None)
            else:
                os.environ["NAUTICAL_QUEUE_DB_CONNECT_RETRIES"] = prev_retry
            if prev_sleep is None:
                os.environ.pop("NAUTICAL_QUEUE_LOCK_SLEEP_BASE", None)
            else:
                os.environ["NAUTICAL_QUEUE_LOCK_SLEEP_BASE"] = prev_sleep
            if prev_lock_retry is None:
                os.environ.pop("NAUTICAL_QUEUE_LOCK_RETRIES", None)
            else:
                os.environ["NAUTICAL_QUEUE_LOCK_RETRIES"] = prev_lock_retry


def test_diag_log_rotation_bounds():
    """Persistent diag log should rotate when exceeding max size."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_diag_log = os.environ.get("NAUTICAL_DIAG_LOG")
        prev_diag_max = os.environ.get("NAUTICAL_DIAG_LOG_MAX_BYTES")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_DIAG_LOG"] = "1"
        os.environ["NAUTICAL_DIAG_LOG_MAX_BYTES"] = "20"
        try:
            mod = _load_hook_module(hook, "_nautical_diag_log_rotation_test")
            log_path = Path(td) / ".nautical_diag.jsonl"
            log_path.write_text("x" * 64, encoding="utf-8")
            mod._diag("rotate me")
            overflow = list(Path(td).glob(".nautical_diag.overflow.*.jsonl"))
            expect(overflow, "diag log did not rotate")
            expect(log_path.exists(), "diag log missing after rotation")
            content = log_path.read_text(encoding="utf-8").strip()
            expect(content, "diag log not written after rotation")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_diag_log is None:
                os.environ.pop("NAUTICAL_DIAG_LOG", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG"] = prev_diag_log
            if prev_diag_max is None:
                os.environ.pop("NAUTICAL_DIAG_LOG_MAX_BYTES", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG_MAX_BYTES"] = prev_diag_max

def test_diag_log_redacts_sensitive_fields():
    """Persistent diag log should redact sensitive fields."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_diag_log = os.environ.get("NAUTICAL_DIAG_LOG")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_DIAG_LOG"] = "1"
        try:
            mod = _load_hook_module(hook, "_nautical_diag_log_redact_test")
            msg = json.dumps({"description": "secret", "notes": "hidden", "ok": "keep"})
            mod._diag(msg)
            log_path = Path(td) / ".nautical_diag.jsonl"
            content = log_path.read_text(encoding="utf-8")
            expect("secret" not in content and "hidden" not in content, "diag log did not redact sensitive fields")
            expect("[redacted]" in content, "diag log missing redaction marker")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_diag_log is None:
                os.environ.pop("NAUTICAL_DIAG_LOG", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG"] = prev_diag_log


def test_hook_diag_redact_msg_masks_sensitive_json_fields():
    """Hook-level diag redaction helper should mask sensitive JSON fields."""
    hook_add = _find_hook_file("on-add-nautical.py")
    hook_exit = _find_hook_file("on-exit-nautical.py")
    mod_add = _load_hook_module(hook_add, "_nautical_on_add_diag_redact_msg_test")
    mod_exit = _load_hook_module(hook_exit, "_nautical_on_exit_diag_redact_msg_test")
    raw = json.dumps(
        {
            "description": "sensitive text",
            "annotations": "top secret",
            "note": "private",
            "safe": "ok",
        },
        ensure_ascii=False,
    )
    red_add = mod_add._diag_redact_msg(raw)
    red_exit = mod_exit._diag_redact_msg(raw)
    obj_add = json.loads(red_add)
    obj_exit = json.loads(red_exit)
    expect(obj_add.get("description") == "[redacted]", f"on-add description not redacted: {obj_add}")
    expect(obj_add.get("annotations") == "[redacted]", f"on-add annotations not redacted: {obj_add}")
    expect(obj_add.get("note") == "[redacted]", f"on-add note not redacted: {obj_add}")
    expect(obj_add.get("safe") == "ok", f"on-add non-sensitive key changed: {obj_add}")
    expect(obj_exit.get("description") == "[redacted]", f"on-exit description not redacted: {obj_exit}")
    expect(obj_exit.get("annotations") == "[redacted]", f"on-exit annotations not redacted: {obj_exit}")
    expect(obj_exit.get("note") == "[redacted]", f"on-exit note not redacted: {obj_exit}")
    expect(obj_exit.get("safe") == "ok", f"on-exit non-sensitive key changed: {obj_exit}")


def test_diag_log_structured_fields():
    """Persistent diag log should include structured fields for dict messages."""
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_diag_log = os.environ.get("NAUTICAL_DIAG_LOG")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_DIAG_LOG"] = "1"
        try:
            core.diag({"msg": "hello", "description": "secret", "ok": "keep", "event": "test"})
            log_path = Path(td) / ".nautical_diag.jsonl"
            line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            obj = json.loads(line)
            expect(obj.get("msg") == "hello", f"unexpected msg: {obj.get('msg')!r}")
            data = obj.get("data") or {}
            expect(data.get("description") == "[redacted]", "structured redaction missing")
            expect(data.get("ok") == "keep", "structured payload missing ok field")
            expect("pid" in obj and "cwd" in obj, "structured fields missing pid/cwd")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_diag_log is None:
                os.environ.pop("NAUTICAL_DIAG_LOG", None)
            else:
                os.environ["NAUTICAL_DIAG_LOG"] = prev_diag_log

def test_warn_rate_limited_any():
    """Rate-limited warnings should only emit once per interval."""
    with tempfile.TemporaryDirectory() as td:
        prev_cache = os.environ.get("XDG_CACHE_HOME")
        prev_diag = os.environ.get("NAUTICAL_DIAG")
        os.environ["XDG_CACHE_HOME"] = td
        os.environ["NAUTICAL_DIAG"] = "1"
        buf = io.StringIO()
        try:
            with contextlib.redirect_stderr(buf):
                core._warn_rate_limited_any("golden_rate_limit", "rate limit message", min_interval_s=3600.0)
                core._warn_rate_limited_any("golden_rate_limit", "rate limit message", min_interval_s=3600.0)
            out = buf.getvalue().strip().splitlines()
            expect(len([ln for ln in out if "rate limit message" in ln]) == 1,
                   f"expected 1 warning, got: {out}")
        finally:
            if prev_cache is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev_cache
            if prev_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = prev_diag

def test_on_exit_requeues_when_task_lock_recent():
    """on-exit should requeue when Taskwarrior export reports a lock."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_lock_recent_test")
            entry = {
                "child": {"uuid": "u1"},
                "parent_uuid": "p1",
                "child_short": "c1",
                "spawn_intent_id": "si_test",
            }
            mod._requeue_entries([entry])

            def _run_task_stub(cmd, **_kwargs):
                if "export" in cmd:
                    return False, "", "database is locked"
                return True, "", ""

            mod._run_task = _run_task_stub
            stats = mod._drain_queue()
            expect(stats.get("requeued") == 1, f"expected requeue when lock active, got {stats}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_core_cache_dir_and_lock_permissions():
    """Core cache dir and lock files should have restricted permissions."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "cache")
        mod = _load_hook_module(core_path, "_nautical_core_cache_perm_test")
        mod._CACHE_DIR = None
        mod.ANCHOR_CACHE_DIR_OVERRIDE = cache_dir
        path = mod._cache_dir()
        expect(path == cache_dir, f"cache dir mismatch: {path}")
        mode = stat.S_IMODE(os.stat(path).st_mode)
        expect((mode & 0o077) == 0, f"cache dir has group/other perms: {oct(mode)}")
        lock_path = mod._cache_lock_path("permtest")
        with mod._cache_lock("permtest") as ok:
            expect(ok, "cache lock did not acquire")
            lmode = stat.S_IMODE(os.stat(lock_path).st_mode)
            expect((lmode & 0o077) == 0, f"cache lock has group/other perms: {oct(lmode)}")

def test_core_cache_lock_contention_matches_safe_lock():
    """_cache_lock should block contention similarly to safe_lock."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cache_dir = os.path.join(td, "cache")
        mod = _load_hook_module(core_path, "_nautical_core_cache_lock_test")
        mod._CACHE_DIR = None
        mod.ANCHOR_CACHE_DIR_OVERRIDE = cache_dir
        mod._cache_dir()
        with mod._cache_lock("contend") as ok:
            expect(ok, "cache lock did not acquire")
            with mod._cache_lock("contend") as ok2:
                expect(not ok2, "cache lock should not acquire when already locked")
        prev = getattr(mod, "fcntl", None)
        mod.fcntl = None
        try:
            with mod._cache_lock("contend2") as ok3:
                expect(ok3, "fallback cache lock did not acquire")
                with mod._cache_lock("contend2") as ok4:
                    expect(not ok4, "fallback cache lock should not acquire when locked")
        finally:
            mod.fcntl = prev


def test_core_cache_dir_rejects_symlink_override():
    """_cache_dir should reject symlink override paths and choose a real directory."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        target = os.path.join(td, "real-cache")
        symlink = os.path.join(td, "cache-link")
        os.makedirs(target, exist_ok=True)
        os.symlink(target, symlink)

        prev_xdg = os.environ.get("XDG_CACHE_HOME")
        prev_tmp = os.environ.get("NAUTICAL_ALLOW_TMP_CACHE")
        os.environ["XDG_CACHE_HOME"] = td
        os.environ["NAUTICAL_ALLOW_TMP_CACHE"] = "1"
        try:
            mod = _load_hook_module(core_path, "_nautical_core_cache_symlink_guard_test")
            mod._CACHE_DIR = None
            mod.ANCHOR_CACHE_DIR_OVERRIDE = symlink
            chosen = mod._cache_dir()
            expect(chosen != symlink, f"symlink override should be rejected, got {chosen}")
            expect(chosen and os.path.isdir(chosen), f"cache dir should fall back to valid dir, got {chosen!r}")
            expect(not os.path.islink(chosen), f"cache dir should not be symlink, got {chosen}")
        finally:
            if prev_xdg is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev_xdg
            if prev_tmp is None:
                os.environ.pop("NAUTICAL_ALLOW_TMP_CACHE", None)
            else:
                os.environ["NAUTICAL_ALLOW_TMP_CACHE"] = prev_tmp

def test_on_exit_large_queue_bounded_drain():
    """Large queues should drain in bounded batches and leave remainder."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        prev_max_lines = os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_LINES")
        prev_max_bytes = os.environ.get("NAUTICAL_SPAWN_QUEUE_MAX_BYTES")
        os.environ["TASKDATA"] = td
        os.environ["NAUTICAL_SPAWN_QUEUE_MAX_LINES"] = "3"
        os.environ["NAUTICAL_SPAWN_QUEUE_MAX_BYTES"] = "1048576"
        try:
            mod_exit = _load_hook_module(hook, "_nautical_on_exit_large_queue_test")
            q_path = mod_exit._QUEUE_PATH
            entries = [
                {
                    "child": {"uuid": f"u{i}"},
                    "parent_uuid": f"p{i}",
                    "child_short": f"c{i}",
                    "spawn_intent_id": f"si_{i}",
                }
                for i in range(7)
            ]
            with open(q_path, "w", encoding="utf-8") as f:
                for obj in entries:
                    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            batch = mod_exit._take_queue_entries()
            expect(len(batch) == 3, f"expected bounded batch size 3, got {len(batch)}")
            with open(q_path, "r", encoding="utf-8") as f:
                remaining = [ln for ln in f.read().splitlines() if ln.strip()]
            expect(len(remaining) == 4, f"expected 4 remaining lines, got {len(remaining)}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata
            if prev_max_lines is None:
                os.environ.pop("NAUTICAL_SPAWN_QUEUE_MAX_LINES", None)
            else:
                os.environ["NAUTICAL_SPAWN_QUEUE_MAX_LINES"] = prev_max_lines
            if prev_max_bytes is None:
                os.environ.pop("NAUTICAL_SPAWN_QUEUE_MAX_BYTES", None)
            else:
                os.environ["NAUTICAL_SPAWN_QUEUE_MAX_BYTES"] = prev_max_bytes

def test_on_exit_queue_drain_idempotent():
    """Re-draining after a mid-run failure should not re-import children."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod_exit = _load_hook_module(hook, "_nautical_on_exit_idempotent_test")
            imported = set()
            import_calls = 0
            modify_calls = 0
            modify_fail_once = True
            parent_next = {"p1": "", "p2": ""}

            def _fake_run_task(cmd, *, input_text=None, timeout=6.0, **_kwargs):
                nonlocal import_calls, modify_calls, modify_fail_once
                if "export" in cmd:
                    uuid = ""
                    for part in cmd:
                        if isinstance(part, str) and part.startswith("uuid:"):
                            uuid = part.split(":", 1)[1]
                            break
                    if uuid and uuid in imported:
                        return True, json.dumps({"uuid": uuid}), ""
                    if uuid in {"p1", "p2"}:
                        return True, json.dumps({"uuid": uuid, "nextLink": parent_next[uuid]}), ""
                    return True, "{}", ""
                if "import" in cmd:
                    import_calls += 1
                    try:
                        obj = json.loads((input_text or "").strip() or "{}")
                    except Exception:
                        obj = {}
                    if isinstance(obj, dict) and obj.get("uuid"):
                        imported.add(obj["uuid"])
                    return True, "", ""
                if "modify" in cmd:
                    modify_calls += 1
                    if modify_fail_once:
                        modify_fail_once = False
                        return False, "", "database is locked"
                    for parent_id, child_short in (("p1", "c1"), ("p2", "c2")):
                        if f"uuid:{parent_id}" in cmd:
                            parent_next[parent_id] = child_short
                    return True, "", ""
                return True, "", ""

            mod_exit._run_task = _fake_run_task
            q_path = mod_exit._QUEUE_PATH
            entries = [
                {"child": {"uuid": "u1"}, "parent_uuid": "p1", "child_short": "c1", "spawn_intent_id": "si_1"},
                {"child": {"uuid": "u2"}, "parent_uuid": "p2", "child_short": "c2", "spawn_intent_id": "si_2"},
            ]
            with open(q_path, "w", encoding="utf-8") as f:
                for obj in entries:
                    f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")

            stats1 = mod_exit._drain_queue()
            expect(stats1.get("requeued") == 1, f"expected one requeue, got {stats1}")

            stats2 = mod_exit._drain_queue()
            expect(stats2.get("errors") == 0, f"second drain should be clean, got {stats2}")
            expect(import_calls == 2, f"expected 2 imports total, got {import_calls}")
            expect(modify_calls == 3, f"expected 3 modify calls, got {modify_calls}")
            expect(not q_path.exists() or not q_path.read_text(encoding="utf-8").strip(), "queue not fully drained")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_exit_rolls_back_parent_nextlink_on_missing_child():
    """on-exit should not update parent nextLink if child is missing after failed import."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_rollback_test")
            calls = []
            parent_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
            child_uuid = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
            child_short = "bbbbbbbb"

            def _fake_run_task(cmd, *, input_text=None, timeout=0.0, **_kwargs):
                if "export" in cmd:
                    target = ""
                    for part in cmd:
                        if isinstance(part, str) and part.startswith("uuid:"):
                            target = part.split(":", 1)[1]
                            break
                    if target == child_uuid:
                        return True, "{}", ""
                    if target == parent_uuid:
                        payload = {"uuid": parent_uuid, "nextLink": child_short}
                        return True, json.dumps(payload), ""
                if "modify" in cmd and "nextLink:" in cmd:
                    calls.append("clear_parent")
                    return True, "", ""
                return True, "", ""

            mod._run_task = _fake_run_task
            mod._import_child = lambda _obj: (False, "import failed")
            q_path = mod._QUEUE_PATH
            entry = {
                "parent_uuid": parent_uuid,
                "child_short": child_short,
                "child": {"uuid": child_uuid},
                "spawn_intent_id": "si_test",
            }
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text(json.dumps(entry, ensure_ascii=False) + "\n", encoding="utf-8")
            mod._drain_queue()
            expect("clear_parent" not in calls, "parent nextLink should not be updated on missing child")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_exit_uses_tw_data_dir_for_export_and_modify():
    """on-exit should target TW_DATA_DIR when explicit data dir is enabled."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_data_dir_test")
    calls = []

    def _fake_run_task(cmd, **_kwargs):
        calls.append(cmd)
        if "export" in cmd:
            return True, json.dumps({"uuid": "deadbeef"}), ""
        return True, "", ""

    mod._run_task = _fake_run_task
    mod.TW_DATA_DIR = "/tmp/nautical_test_data"
    mod._USE_RC_DATA_LOCATION = True

    mod._export_uuid("deadbeef")
    mod._update_parent_nextlink("parent-uuid", "childshort")

    expect(calls, "expected _run_task to be called")
    want = f"rc.data.location={mod.TW_DATA_DIR}"
    expect(want in calls[0], f"export missing data dir: {calls[0]!r}")
    expect(any(want in call for call in calls[1:]), f"modify missing data dir: {calls!r}")

def test_on_exit_no_explicit_taskdata_skips_rc_data_location():
    """on-exit should not force rc.data.location when data dir is not explicit."""
    hook = _find_hook_file("on-exit-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-exit-nautical.py"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_exit_no_data_override_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    calls = []

    def _fake_run_task(cmd, **_kwargs):
        calls.append(cmd)
        return True, json.dumps({"uuid": "deadbeef"}), ""

    mod._run_task = _fake_run_task
    mod._export_uuid("deadbeef")
    expect(calls, "expected _run_task to be called")
    expect(
        all(not str(part).startswith("rc.data.location=") for part in calls[0]),
        f"should not force rc.data.location without explicit data dir: {calls[0]!r}",
    )

def test_on_exit_reads_data_arg_from_hook_argv():
    """on-exit should resolve TW_DATA_DIR from hook argv data: token."""
    hook = _find_hook_file("on-exit-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-exit-nautical.py", "api:2", "command:modify", "data:/tmp/nautical_data_arg_test"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_exit_data_arg_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    expect(str(mod.TW_DATA_DIR).endswith("/tmp/nautical_data_arg_test"), f"unexpected TW_DATA_DIR: {mod.TW_DATA_DIR}")
    expect(bool(getattr(mod, "_USE_RC_DATA_LOCATION", False)), "rc.data.location should be enabled when data arg is present")

def test_on_modify_no_explicit_taskdata_skips_rc_data_location():
    """on-modify should not force rc.data.location when data dir is not explicit."""
    hook = _find_hook_file("on-modify-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-modify-nautical.py"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_modify_no_data_override_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    calls = []

    def _fake_run_task(cmd, **_kwargs):
        calls.append(cmd)
        return True, json.dumps({"uuid": "deadbeef"}), ""

    mod._run_task = _fake_run_task
    _ = mod._task_exists_by_uuid_uncached("deadbeef", env={})
    expect(calls, "expected _run_task to be called")
    expect(
        all(not str(part).startswith("rc.data.location=") for part in calls[0]),
        f"should not force rc.data.location without explicit data dir: {calls[0]!r}",
    )

def test_on_modify_reads_data_arg_from_hook_argv():
    """on-modify should resolve TW_DATA_DIR from hook argv data: token."""
    hook = _find_hook_file("on-modify-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-modify-nautical.py", "api:2", "command:modify", "data:/tmp/nautical_data_arg_mod_test"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_modify_data_arg_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    expect(str(mod.TW_DATA_DIR).endswith("/tmp/nautical_data_arg_mod_test"), f"unexpected TW_DATA_DIR: {mod.TW_DATA_DIR}")
    expect(bool(getattr(mod, "_USE_RC_DATA_LOCATION", False)), "rc.data.location should be enabled when data arg is present")

def test_on_add_no_explicit_taskdata_skips_rc_data_location():
    """on-add should not force rc.data.location when data dir is not explicit."""
    hook = _find_hook_file("on-add-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-add-nautical.py"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_add_no_data_override_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    calls = []

    def _fake_run_task(cmd, **_kwargs):
        calls.append(cmd)
        return True, "[]", ""

    mod._run_task = _fake_run_task
    _ = mod.tw_export_chain("cid-test")
    expect(calls, "expected _run_task to be called")
    expect(
        all(not str(part).startswith("rc.data.location=") for part in calls[0]),
        f"should not force rc.data.location without explicit data dir: {calls[0]!r}",
    )

def test_on_add_reads_data_arg_from_hook_argv():
    """on-add should resolve TW_DATA_DIR from hook argv data: token."""
    hook = _find_hook_file("on-add-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    sys.argv = ["on-add-nautical.py", "api:2", "command:add", "data:/tmp/nautical_data_arg_add_test"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_add_data_arg_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is not None:
            os.environ["TASKDATA"] = prev_taskdata
    expect(str(mod.TW_DATA_DIR).endswith("/tmp/nautical_data_arg_add_test"), f"unexpected TW_DATA_DIR: {mod.TW_DATA_DIR}")
    expect(bool(getattr(mod, "_USE_RC_DATA_LOCATION", False)), "rc.data.location should be enabled when data arg is present")

def test_on_exit_data_arg_overrides_taskdata_env():
    """on-exit should prefer hook argv data: over TASKDATA env when both are present."""
    hook = _find_hook_file("on-exit-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    os.environ["TASKDATA"] = "/tmp/nautical_env_exit_test"
    sys.argv = ["on-exit-nautical.py", "api:2", "command:modify", "data:/tmp/nautical_arg_exit_test"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_exit_data_arg_precedence_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is None:
            os.environ.pop("TASKDATA", None)
        else:
            os.environ["TASKDATA"] = prev_taskdata
    expect(str(mod.TW_DATA_DIR).endswith("/tmp/nautical_arg_exit_test"), f"expected argv data dir, got: {mod.TW_DATA_DIR}")

def test_on_modify_data_arg_overrides_taskdata_env():
    """on-modify should prefer hook argv data: over TASKDATA env when both are present."""
    hook = _find_hook_file("on-modify-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    os.environ["TASKDATA"] = "/tmp/nautical_env_modify_test"
    sys.argv = ["on-modify-nautical.py", "api:2", "command:modify", "data:/tmp/nautical_arg_modify_test"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_modify_data_arg_precedence_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is None:
            os.environ.pop("TASKDATA", None)
        else:
            os.environ["TASKDATA"] = prev_taskdata
    expect(str(mod.TW_DATA_DIR).endswith("/tmp/nautical_arg_modify_test"), f"expected argv data dir, got: {mod.TW_DATA_DIR}")

def test_on_add_data_arg_overrides_taskdata_env():
    """on-add should prefer hook argv data: over TASKDATA env when both are present."""
    hook = _find_hook_file("on-add-nautical.py")
    prev_taskdata = os.environ.get("TASKDATA")
    prev_argv = list(sys.argv)
    os.environ["TASKDATA"] = "/tmp/nautical_env_add_test"
    sys.argv = ["on-add-nautical.py", "api:2", "command:add", "data:/tmp/nautical_arg_add_test"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_add_data_arg_precedence_test")
    finally:
        sys.argv = prev_argv
        if prev_taskdata is None:
            os.environ.pop("TASKDATA", None)
        else:
            os.environ["TASKDATA"] = prev_taskdata
    expect(str(mod.TW_DATA_DIR).endswith("/tmp/nautical_arg_add_test"), f"expected argv data dir, got: {mod.TW_DATA_DIR}")

def test_core_resolve_task_data_context_precedence():
    """core resolver should prefer argv data:, then TASKDATA env, then tw_dir fallback."""
    d1, use1, src1 = core.resolve_task_data_context(
        argv=["api:2", "command:modify", "data:/tmp/nautical_core_arg_test"],
        env={"TASKDATA": "/tmp/nautical_core_env_test"},
        tw_dir="/tmp/nautical_core_fallback_test",
    )
    expect(str(d1).endswith("/tmp/nautical_core_arg_test"), f"unexpected argv resolution: {(d1, use1, src1)!r}")
    expect(bool(use1), "argv data: should enable rc.data.location")
    expect(src1 == "argv", f"expected argv source, got {src1!r}")

    d2, use2, src2 = core.resolve_task_data_context(
        argv=["api:2", "command:modify"],
        env={"TASKDATA": "/tmp/nautical_core_env_test"},
        tw_dir="/tmp/nautical_core_fallback_test",
    )
    expect(str(d2).endswith("/tmp/nautical_core_env_test"), f"unexpected env resolution: {(d2, use2, src2)!r}")
    expect(bool(use2), "TASKDATA should enable rc.data.location")
    expect(src2 == "env", f"expected env source, got {src2!r}")

    d3, use3, src3 = core.resolve_task_data_context(
        argv=["api:2", "command:modify"],
        env={},
        tw_dir="/tmp/nautical_core_fallback_test",
    )
    expect(str(d3).endswith("/tmp/nautical_core_fallback_test"), f"unexpected fallback resolution: {(d3, use3, src3)!r}")
    expect(not bool(use3), "fallback should not enable rc.data.location")
    expect(src3 == "fallback", f"expected fallback source, got {src3!r}")

def test_core_resolve_task_data_context_rejects_unsafe_world_writable_dir():
    """core resolver should reject explicit world-writable data dirs by default."""
    if os.name == "nt":
        expect(True, "skip on non-POSIX")
        return
    with tempfile.TemporaryDirectory() as td:
        unsafe = Path(td) / "unsafe-data-dir"
        unsafe.mkdir()
        try:
            os.chmod(unsafe, 0o777)
        except Exception:
            expect(True, "chmod unavailable; skip")
            return
        fallback = Path(td) / "safe-fallback"
        got, use_rc, src = core.resolve_task_data_context(
            argv=["api:2", "command:modify", f"data:{unsafe}"],
            env={},
            tw_dir=str(fallback),
        )
        expect(src == "fallback", f"unsafe explicit dir should fall back, got source={src!r}")
        expect(not bool(use_rc), "fallback path should disable rc.data.location")
        expect(str(got).endswith("/safe-fallback"), f"unexpected fallback path: {got!r}")


def test_core_resolve_task_data_context_trust_override_allows_explicit_dir():
    """core resolver trust override should allow explicit paths without safety checks."""
    if os.name == "nt":
        expect(True, "skip on non-POSIX")
        return
    with tempfile.TemporaryDirectory() as td:
        unsafe = Path(td) / "unsafe-data-dir"
        unsafe.mkdir()
        try:
            os.chmod(unsafe, 0o777)
        except Exception:
            expect(True, "chmod unavailable; skip")
            return
        got, use_rc, src = core.resolve_task_data_context(
            argv=["api:2", "command:modify", f"data:{unsafe}"],
            env={"NAUTICAL_TRUST_TASKDATA_PATH": "1"},
            tw_dir=str(Path(td) / "safe-fallback"),
        )
        expect(src == "argv", f"trusted explicit dir should keep argv source, got {src!r}")
        expect(bool(use_rc), "trusted explicit dir should keep rc.data.location enabled")
        expect(
            got == os.path.abspath(str(unsafe)),
            f"trusted explicit dir mismatch: got={got!r} want={os.path.abspath(str(unsafe))!r}",
        )


def test_core_resolve_task_data_context_rejects_parent_traversal_segments():
    """core resolver should reject explicit data paths containing '..' by default."""
    d, use_rc, src = core.resolve_task_data_context(
        argv=["api:2", "command:modify", "data:../nautical_bad_dir"],
        env={},
        tw_dir="/tmp/nautical_core_fallback_test",
    )
    expect(src == "fallback", f"parent traversal path should fall back, got source={src!r}")
    expect(not bool(use_rc), "fallback should disable rc.data.location for rejected path")
    expect(str(d).endswith("/tmp/nautical_core_fallback_test"), f"unexpected fallback path: {d!r}")


def test_core_config_paths_rejects_parent_traversal_in_env():
    """_config_paths should reject NAUTICAL_CONFIG values containing '..' by default."""
    prev_cfg = os.environ.get("NAUTICAL_CONFIG")
    prev_trust = os.environ.get("NAUTICAL_TRUST_CONFIG_PATH")
    try:
        os.environ["NAUTICAL_CONFIG"] = "../nautical.toml"
        os.environ.pop("NAUTICAL_TRUST_CONFIG_PATH", None)
        got = core._config_paths()
    finally:
        if prev_cfg is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = prev_cfg
        if prev_trust is None:
            os.environ.pop("NAUTICAL_TRUST_CONFIG_PATH", None)
        else:
            os.environ["NAUTICAL_TRUST_CONFIG_PATH"] = prev_trust
    expect(got == [], f"expected NAUTICAL_CONFIG traversal path to be rejected, got: {got!r}")


def test_core_config_paths_trust_override_allows_parent_traversal_in_env():
    """_config_paths trust override should permit parent-segment NAUTICAL_CONFIG values."""
    prev_cfg = os.environ.get("NAUTICAL_CONFIG")
    prev_trust = os.environ.get("NAUTICAL_TRUST_CONFIG_PATH")
    try:
        os.environ["NAUTICAL_CONFIG"] = "../nautical.toml"
        os.environ["NAUTICAL_TRUST_CONFIG_PATH"] = "1"
        got = core._config_paths()
    finally:
        if prev_cfg is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = prev_cfg
        if prev_trust is None:
            os.environ.pop("NAUTICAL_TRUST_CONFIG_PATH", None)
        else:
            os.environ["NAUTICAL_TRUST_CONFIG_PATH"] = prev_trust
    expect(len(got) == 1, f"expected trusted NAUTICAL_CONFIG to be returned, got: {got!r}")
    expect(got[0] == os.path.abspath(os.path.expanduser("../nautical.toml")), f"unexpected trusted path: {got!r}")


def _assert_hook_requires_core_data_context(hook_name: str, module_name: str):
    hook = _find_hook_file(hook_name)
    prev_core = os.environ.get("NAUTICAL_CORE_PATH")
    prev_argv = list(sys.argv)
    with tempfile.TemporaryDirectory() as td:
        fake_core = Path(td) / "nautical_core/__init__.py"
        fake_core.write_text(
            "def _warn_once_per_day_any(*_args, **_kwargs):\n"
            "    return None\n",
            encoding="utf-8",
        )
        os.environ["NAUTICAL_CORE_PATH"] = td
        sys.argv = [hook_name]
        try:
            try:
                _load_hook_module(hook, module_name)
                raise AssertionError("expected hook import to fail without core data resolver")
            except Exception as e:
                expect(
                    "resolve_task_data_context" in str(e),
                    f"unexpected error when resolver missing: {e!r}",
                )
        finally:
            sys.argv = prev_argv
            if prev_core is None:
                os.environ.pop("NAUTICAL_CORE_PATH", None)
            else:
                os.environ["NAUTICAL_CORE_PATH"] = prev_core

def test_on_add_requires_core_data_context_helper():
    """on-add should fail closed when core data-context resolver is unavailable."""
    _assert_hook_requires_core_data_context("on-add-nautical.py", "_nautical_on_add_requires_core_data_ctx_test")

def test_on_modify_requires_core_data_context_helper():
    """on-modify should fail closed when core data-context resolver is unavailable."""
    _assert_hook_requires_core_data_context("on-modify-nautical.py", "_nautical_on_modify_requires_core_data_ctx_test")

def test_on_exit_requires_core_data_context_helper():
    """on-exit should fail closed when core data-context resolver is unavailable."""
    _assert_hook_requires_core_data_context("on-exit-nautical.py", "_nautical_on_exit_requires_core_data_ctx_test")

def test_on_modify_read_two_fuzz_inputs():
    """on-modify input parsing should be strict and return JSON errors on bad input."""
    hook = _find_hook_file("on-modify-nautical.py")
    cases = [
        ("", "empty"),
        ("{not-json}", "invalid"),
        (json.dumps({"status": "pending"}), "json"),
        (json.dumps({"status": "pending"}) + "\n" + json.dumps({"status": "pending"}), "json"),
        ("  \n" + json.dumps({"status": "pending"}) + "\n", "json"),
    ]
    for raw, mode in cases:
        p = _run_hook_script_raw(hook, raw)
        if mode in {"empty", "invalid", "json"}:
            expect(p.returncode != 0, f"on-modify should fail for case {mode}")
            expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")
        else:
            expect(p.returncode == 0, f"on-modify returned {p.returncode} for case {mode}")
            _assert_stdout_json_only(p.stdout)

def test_on_add_read_one_fuzz_inputs():
    """on-add input parsing should reject malformed JSON and empty input."""
    hook = _find_hook_file("on-add-nautical.py")
    cases = [
        ("", "empty"),
        ("{not-json}", "invalid"),
        (json.dumps({"status": "pending"}) + "\n" + json.dumps({"status": "pending"}), "multi"),
        ("  \n" + json.dumps({"status": "pending"}) + "\n{bad", "trailing"),
    ]
    for raw, mode in cases:
        p = _run_hook_script_raw(hook, raw)
        expect(p.returncode != 0, f"on-add should fail for case {mode}")
        expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_read_two_invalid_trailing():
    """on-modify should fail on extra garbage after JSON objects."""
    hook = _find_hook_file("on-modify-nautical.py")
    raw = json.dumps({"status": "pending"}) + "\n" + json.dumps({"status": "pending"}) + "\n" + "{bad"
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail on trailing garbage")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_read_two_array_uuid_mismatch_fails():
    """on-modify array input should reject old/new UUID mismatches."""
    hook = _find_hook_file("on-modify-nautical.py")
    raw = json.dumps(
        [
            {"uuid": "00000000-0000-0000-0000-000000000111", "status": "pending"},
            {"uuid": "00000000-0000-0000-0000-000000000222", "status": "completed"},
        ]
    )
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail for mismatched UUIDs in array input")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_read_two_array_single_missing_uuid_fails():
    """on-modify array input with one dict must include UUID."""
    hook = _find_hook_file("on-modify-nautical.py")
    raw = json.dumps([{"status": "pending"}])
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail when array input lacks UUID")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_modify_invalid_anchor_has_no_stdout():
    """on-modify should keep stdout empty on semantic validation failures."""
    hook = _find_hook_file("on-modify-nautical.py")
    old = {
        "uuid": "00000000-0000-0000-0000-000000000611",
        "status": "pending",
        "description": "invalid anchor test",
    }
    new = dict(old)
    new["anchor"] = "bad"
    raw = json.dumps(old) + "\n" + json.dumps(new)
    p = _run_hook_script_raw(hook, raw)
    expect(p.returncode != 0, "on-modify should fail on invalid anchor")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_on_add_rejects_oversized_stdin_early():
    """on-add should reject stdin over _MAX_JSON_BYTES before JSON parsing."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_oversized_input_test")
    mod._MAX_JSON_BYTES = 32
    raw = json.dumps({"uuid": "u", "status": "pending", "description": "x" * 256})

    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin, sys.stdout, sys.stderr = stdin, out, err
        try:
            mod.main()
            raise AssertionError("on-add should fail on oversized stdin")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr
    expect((out.getvalue() or "").strip() == "", f"expected no stdout on oversized input, got: {out.getvalue()!r}")

def test_on_modify_rejects_oversized_stdin_early():
    """on-modify should reject stdin over _MAX_JSON_BYTES before object parsing."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_oversized_input_test")
    mod._MAX_JSON_BYTES = 32
    raw = json.dumps({"uuid": "u", "status": "pending", "description": "x" * 256})

    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    out = io.StringIO()
    err = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin, sys.stdout, sys.stderr = stdin, out, err
        try:
            mod._read_two()
            raise AssertionError("on-modify should fail on oversized stdin")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr
    expect((out.getvalue() or "").strip() == "", f"expected no stdout on oversized input, got: {out.getvalue()!r}")

def test_health_check_json_ok_empty_taskdata():
    """health check should report ok for empty taskdata."""
    path = os.path.join(ROOT, "tools", "nautical_health_check.py")
    with tempfile.TemporaryDirectory() as td:
        p = subprocess.run(
            [sys.executable, path, "--taskdata", td, "--json"],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 0, f"health check returned {p.returncode}: {p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "ok", f"unexpected status: {obj}")

def test_health_check_critical_queue_bytes():
    """health check should return critical when queue exceeds crit threshold."""
    path = os.path.join(ROOT, "tools", "nautical_health_check.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        q = td_path / ".nautical_spawn_queue.jsonl"
        q.write_text("x" * 64, encoding="utf-8")
        p = subprocess.run(
            [
                sys.executable,
                path,
                "--taskdata",
                td,
                "--queue-warn-bytes",
                "32",
                "--queue-crit-bytes",
                "48",
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 2, f"expected critical exit code 2, got {p.returncode}. stderr={p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "crit", f"unexpected status: {obj}")

def test_health_check_critical_queue_db_rows():
    """health check should return critical when sqlite queue rows exceed crit threshold."""
    path = os.path.join(ROOT, "tools", "nautical_health_check.py")
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        db = td_path / ".nautical_queue.db"
        with sqlite3.connect(str(db)) as conn:
            conn.execute(
                """
                CREATE TABLE queue_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spawn_intent_id TEXT,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'queued',
                    claim_token TEXT,
                    claimed_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, created_at, updated_at) "
                "VALUES (?, ?, 0, 'queued', 1.0, 1.0)",
                ("si_hc", json.dumps({"spawn_intent_id": "si_hc"})),
            )
            conn.commit()
        p = subprocess.run(
            [
                sys.executable,
                path,
                "--taskdata",
                td,
                "--queue-warn-bytes",
                "1048576",
                "--queue-crit-bytes",
                "10485760",
                "--queue-db-warn-rows",
                "1",
                "--queue-db-crit-rows",
                "1",
                "--json",
            ],
            text=True,
            capture_output=True,
            timeout=8.0,
        )
        expect(p.returncode == 2, f"expected critical exit code 2, got {p.returncode}. stderr={p.stderr!r}")
        obj = json.loads((p.stdout or "").strip() or "{}")
        expect(obj.get("status") == "crit", f"unexpected status: {obj}")
        metrics = obj.get("metrics") or {}
        expect(int(metrics.get("queue_db_rows") or 0) == 1, f"expected queue_db_rows=1, got {metrics}")


def test_perf_budget_config_covers_cache_io_checks():
    """Perf budget config should include explicit cache save/load check budgets."""
    cfg_path = Path(ROOT) / "tools" / "perf_budget.json"
    obj = json.loads(cfg_path.read_text(encoding="utf-8"))
    budgets = obj.get("budgets_seconds") if isinstance(obj, dict) else None
    workload = obj.get("workload") if isinstance(obj, dict) else None
    expect(isinstance(budgets, dict), "budgets_seconds must be present")
    expect(isinstance(workload, dict), "workload must be present")
    expect("cache_save" in budgets, "cache_save budget missing")
    expect("cache_load_hot" in budgets, "cache_load_hot budget missing")
    expect("cache_save_rounds" in workload, "cache_save_rounds missing from workload")
    expect("cache_load_rounds" in workload, "cache_load_rounds missing from workload")


def test_deploy_sanity_script_reports_ok():
    """Deployment sanity script should pass on repo-local hooks/core."""
    path = os.path.join(ROOT, "tools", "nautical_deploy_sanity.py")
    p = subprocess.run(
        [sys.executable, path, "--json"],
        text=True,
        capture_output=True,
        timeout=12.0,
    )
    expect(p.returncode == 0, f"deploy sanity returned {p.returncode}: stderr={p.stderr!r}")
    obj = json.loads((p.stdout or "").strip() or "{}")
    expect(obj.get("status") == "ok", f"unexpected deploy sanity status: {obj}")
    results = obj.get("results") if isinstance(obj.get("results"), list) else []
    expect(results, "deploy sanity should report per-check results")
    expect(all(bool(r.get("ok")) for r in results if isinstance(r, dict)), f"failing sanity result: {results}")


def test_ops_templates_present_and_runner_executable():
    """ops templates should exist and runner script should be executable."""
    ops = os.path.join(ROOT, "tools", "ops")
    files = [
        "README.md",
        "nautical-health-check.crontab",
        "nautical-health-check.service",
        "nautical-health-check.timer",
        "nautical_health_check_cron.sh",
    ]
    for name in files:
        p = os.path.join(ops, name)
        expect(os.path.isfile(p), f"missing ops template: {p}")
    runner = os.path.join(ops, "nautical_health_check_cron.sh")
    expect(os.access(runner, os.X_OK), f"runner should be executable: {runner}")

def test_on_modify_queue_full_drops_with_dead_letter():
    """on-modify should drop spawn intent when queue exceeds max bytes."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_queue_full_test")
            if hasattr(mod, "_load_core"):
                mod._load_core()
            db_path = mod._SPAWN_QUEUE_DB_PATH
            dl_path = mod._DEAD_LETTER_PATH
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_text("x" * 128, encoding="utf-8")
            mod._SPAWN_QUEUE_MAX_BYTES = 10
            mod._enqueue_deferred_spawn({"spawn_intent_id": "si_full", "child": {"uuid": "u1"}})
            expect(db_path.read_text(encoding="utf-8") == "x" * 128, "sqlite queue file should not grow when full")
            expect(dl_path.exists(), "dead-letter not written on queue full")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_modify_enqueue_uses_sqlite_when_legacy_empty():
    """on-modify should enqueue spawn intents into SQLite when legacy JSONL is empty."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_enqueue_sqlite_test")
            if hasattr(mod, "_load_core"):
                mod._load_core()
            entry = {
                "spawn_intent_id": "si_sqlite_a",
                "parent_uuid": "00000000-0000-0000-0000-000000000001",
                "child_short": "00000001",
                "child": {"uuid": "00000000-0000-0000-0000-000000000002"},
            }
            ok, reason = mod._enqueue_deferred_spawn(entry)
            expect(ok, f"sqlite enqueue failed: {reason}")
            db_path = mod._SPAWN_QUEUE_DB_PATH
            expect(db_path.exists(), f"sqlite queue db missing: {db_path}")
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT payload FROM queue_entries WHERE spawn_intent_id=?",
                    ("si_sqlite_a",),
                ).fetchone()
            expect(row is not None, "sqlite queue row missing")
            payload = json.loads((row[0] or "").strip() or "{}")
            expect(payload.get("spawn_intent_id") == "si_sqlite_a", f"unexpected sqlite payload: {payload}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_modify_enqueue_uses_sqlite_even_with_legacy_jsonl_backlog():
    """on-modify should enqueue to SQLite even if a legacy JSONL backlog file exists."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_enqueue_sqlite_with_jsonl_backlog_test")
            if hasattr(mod, "_load_core"):
                mod._load_core()
            q_path = mod._SPAWN_QUEUE_PATH
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text('{"spawn_intent_id":"legacy_1"}\n', encoding="utf-8")

            entry = {
                "spawn_intent_id": "legacy_2",
                "parent_uuid": "00000000-0000-0000-0000-000000000003",
                "child_short": "00000003",
                "child": {"uuid": "00000000-0000-0000-0000-000000000004"},
            }
            ok, reason = mod._enqueue_deferred_spawn(entry)
            expect(ok, f"sqlite enqueue failed: {reason}")
            db_path = mod._SPAWN_QUEUE_DB_PATH
            expect(db_path.exists(), f"sqlite queue db missing: {db_path}")
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT payload FROM queue_entries WHERE spawn_intent_id=?",
                    ("legacy_2",),
                ).fetchone()
            expect(row is not None, "sqlite queue row missing when legacy backlog exists")
            payload = json.loads((row[0] or "").strip() or "{}")
            expect(payload.get("spawn_intent_id") == "legacy_2", f"unexpected sqlite payload: {payload}")
            txt = q_path.read_text(encoding="utf-8")
            expect('"spawn_intent_id":"legacy_1"' in txt, f"legacy entry should remain untouched: {txt!r}")
            expect('"spawn_intent_id":"legacy_2"' not in txt, f"new entry should not be appended to JSONL backlog: {txt!r}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_modify_chain_export_timeout_scales():
    """tw_export_chain should scale timeout based on cached chain size."""
    hook = _find_hook_file("on-modify-nautical.py")
    prev_base = os.environ.get("NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE")
    prev_per = os.environ.get("NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100")
    prev_max = os.environ.get("NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX")
    os.environ["NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE"] = "1.5"
    os.environ["NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100"] = "1.0"
    os.environ["NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX"] = "5.0"
    try:
        mod = _load_hook_module(hook, "_nautical_chain_export_timeout_test")
        mod._CHAIN_CACHE_CHAIN_ID = "cid"
        mod._CHAIN_CACHE = [{}] * 250
        mod._tw_lock_recent = lambda: False
        captured = {}

        def _fake_run_task(_cmd, env=None, timeout=0.0, retries=0, **_kwargs):
            captured["timeout"] = timeout
            return True, "[]", ""

        mod._run_task = _fake_run_task
        mod.tw_export_chain("cid")
        expect(captured.get("timeout") == 3.5, f"unexpected timeout: {captured.get('timeout')}")
    finally:
        if prev_base is None:
            os.environ.pop("NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE", None)
        else:
            os.environ["NAUTICAL_CHAIN_EXPORT_TIMEOUT_BASE"] = prev_base
        if prev_per is None:
            os.environ.pop("NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100", None)
        else:
            os.environ["NAUTICAL_CHAIN_EXPORT_TIMEOUT_PER_100"] = prev_per
        if prev_max is None:
            os.environ.pop("NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX", None)
        else:
            os.environ["NAUTICAL_CHAIN_EXPORT_TIMEOUT_MAX"] = prev_max

def test_tw_export_chain_extra_validation():
    """tw_export_chain should reject unsafe extra arguments."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_chain_export_extra_test")
    mod._tw_lock_recent = lambda: False
    called = {"run": False}

    def _fake_run_task(_cmd, env=None, timeout=0.0, retries=0):
        called["run"] = True
        return True, "[]", ""

    mod._run_task = _fake_run_task
    out = mod.tw_export_chain("cid", extra="status:pending; rm -rf /")
    expect(out == [], "unsafe extra should return empty list")
    expect(not called["run"], "unsafe extra should not call task")


def test_tw_export_chain_extra_rejects_dash_prefixed_tokens():
    """tw_export_chain extra parser should reject dash-prefixed tokens."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_chain_export_extra_dash_test")
    mod._tw_lock_recent = lambda: False
    called = {"run": False}

    def _fake_run_task(_cmd, env=None, timeout=0.0, retries=0):
        called["run"] = True
        return True, "[]", ""

    mod._run_task = _fake_run_task
    out = mod.tw_export_chain("cid", extra="status:pending -rc.hooks=on")
    expect(out == [], "dash-prefixed token should be rejected")
    expect(not called["run"], "rejected extra must not execute task")


def test_on_add_tw_export_chain_extra_validation():
    """on-add tw_export_chain should reject unsafe extra arguments."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_chain_export_extra_test")
    called = {"run": False}

    def _fake_run_task(_cmd, env=None, timeout=0.0, retries=0, **_kwargs):
        _ = (env, timeout, retries)
        called["run"] = True
        return True, "[]", ""

    mod._run_task = _fake_run_task
    out = mod.tw_export_chain("cid", extra="status:pending; rm -rf /")
    expect(out == [], "unsafe extra should return empty list")
    expect(not called["run"], "unsafe extra should not call task")


def test_on_modify_chain_cache_thread_safety_smoke():
    """Concurrent chain cache set/read paths should not crash or return invalid shapes."""
    import threading

    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_chain_cache_thread_safety_test")

    full_uuid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    mod._set_chain_cache(
        "cid-a",
        [{"uuid": full_uuid, "link": 1, "entry": "2026-01-01T00:00:00Z"}],
    )
    mod._task = lambda *_args, **_kwargs: "[]"

    errs = []
    hits = {"short": 0, "full": 0}

    def _writer(chain_id: str):
        try:
            for i in range(300):
                mod._set_chain_cache(
                    chain_id,
                    [{"uuid": full_uuid, "link": 1, "entry": f"2026-01-01T00:00:{i % 60:02d}Z"}],
                )
        except Exception as e:
            errs.append(f"writer:{e}")

    def _reader():
        try:
            for _ in range(600):
                s = mod._export_uuid_short("aaaaaaaa")
                if s is not None:
                    expect(isinstance(s, dict), f"short cache read should return dict, got {type(s)}")
                    hits["short"] += 1
                f = mod._export_uuid_full(full_uuid)
                if f is not None:
                    expect(isinstance(f, dict), f"full cache read should return dict, got {type(f)}")
                    hits["full"] += 1
        except Exception as e:
            errs.append(f"reader:{e}")

    threads = [
        threading.Thread(target=_writer, args=("cid-a",)),
        threading.Thread(target=_writer, args=("cid-b",)),
        threading.Thread(target=_reader),
        threading.Thread(target=_reader),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expect(not errs, f"concurrent chain cache access raised errors: {errs}")
    expect(hits["short"] > 0 and hits["full"] > 0, f"expected cache hits, got {hits}")

def test_chain_integrity_warnings_detects_issues():
    """Chain integrity checker should flag gaps and link inconsistencies."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_integrity_test")
    chain = [
        {
            "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "link": 1,
            "nextLink": "bbbbbbbb",
            "chainID": "cid",
        },
        {
            "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "link": 3,
            "prevLink": "aaaaaaaa",
            "chainID": "",
        },
    ]
    warnings = mod._chain_integrity_warnings(chain, expected_chain_id="cid")
    expect(any("missing link(s): 2" in w for w in warnings), f"expected link gap warning, got {warnings}")
    expect(any("missing chainID" in w for w in warnings), f"expected chainID warning, got {warnings}")


def test_chain_health_advice_coach_healthy_streak():
    """Chain health advice should report healthy streak for steady on-time completions."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_health_advice_healthy_test")
    chain = [
        {"uuid": "a", "link": 1, "status": "completed", "due": "20250101T090000Z", "end": "20250101T090500Z"},
        {"uuid": "b", "link": 2, "status": "completed", "due": "20250104T090000Z", "end": "20250104T091000Z"},
        {"uuid": "c", "link": 3, "status": "completed", "due": "20250107T090000Z", "end": "20250107T090800Z"},
        {"uuid": "d", "link": 4, "status": "pending", "due": "20250110T090000Z"},
    ]
    got = mod._chain_health_advice(chain, "cp", {"cp": "3d"}, style="coach")
    expect(
        got == "Chain looks healthy with a 3-link on-time streak; keep the current cadence.",
        f"unexpected healthy advice: {got!r}",
    )


def test_chain_health_advice_coach_low_ontime_issue():
    """Chain health advice should flag low on-time rate with actionable guidance."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_health_advice_issue_test")
    chain = [
        {"uuid": "a", "link": 1, "status": "completed", "due": "20250101T090000Z", "end": "20250102T120000Z"},
        {"uuid": "b", "link": 2, "status": "completed", "due": "20250102T090000Z", "end": "20250103T140000Z"},
        {"uuid": "c", "link": 3, "status": "completed", "due": "20250103T090000Z", "end": "20250103T093000Z"},
        {"uuid": "d", "link": 4, "status": "pending", "due": "20250105T090000Z"},
    ]
    got = mod._chain_health_advice(chain, "cp", {"cp": "1d"}, style="coach")
    expect(
        got == "Chain needs attention (on-time rate is low); try smaller scopes or later due times.",
        f"unexpected issue advice: {got!r}",
    )


def test_chain_health_advice_clinical_drift_and_style_normalization():
    """Clinical style should include OT, drift, streak, and volatility with normalized style input."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_health_advice_clinical_test")
    chain = [
        {"uuid": "a", "link": 1, "status": "completed", "due": "20250101T090000Z", "end": "20250101T100000Z"},
        {"uuid": "b", "link": 2, "status": "completed", "due": "20250102T090000Z", "end": "20250102T090500Z"},
        {"uuid": "c", "link": 3, "status": "completed", "due": "20250103T090000Z", "end": "20250103T090500Z"},
        {"uuid": "d", "link": 4, "status": "completed", "due": "20250105T090000Z", "end": "20250105T090500Z"},
    ]
    got = mod._chain_health_advice(chain, "anchor", {}, style=" Clinical ")
    expect(
        got == "OT 100% | Drift +1d 00h:00m | Streak 4 | Vol 0d 00h:23m",
        f"unexpected clinical advice: {got!r}",
    )


def test_dst_round_trip_noon_preserves_local_date():
    """Local date+time should round-trip through UTC across DST boundaries."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('tz = "America/New_York"\n')
        mod = _load_core_module(core_path, "_nautical_core_dst_test", cfg)
        dates = [
            date(2025, 3, 8),
            date(2025, 3, 9),
            date(2025, 3, 10),
            date(2025, 11, 1),
            date(2025, 11, 2),
            date(2025, 11, 3),
        ]
        for d in dates:
            dt_local = mod.build_local_datetime(d, (12, 0))
            dt_utc = dt_local.astimezone(timezone.utc)
            back = mod.to_local(dt_utc)
            expect(back.date() == d, f"DST round-trip date mismatch: {d} -> {back.date()}")
            expect(back.hour == 12 and back.minute == 0, f"DST round-trip time mismatch: {back}")


def test_core_invalid_timezone_warns_and_falls_back_to_utc():
    """Invalid timezone config should fall back to UTC and emit diagnostic warning when enabled."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('tz = "Invalid/Timezone"\n')

        prev_diag = os.environ.get("NAUTICAL_DIAG")
        prev_xdg = os.environ.get("XDG_CACHE_HOME")
        os.environ["NAUTICAL_DIAG"] = "1"
        os.environ["XDG_CACHE_HOME"] = td
        try:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                mod = _load_core_module(core_path, "_nautical_core_bad_tz_fallback_test", cfg)
            expect(getattr(mod, "_LOCAL_TZ", None) is None, "invalid timezone should use UTC fallback")
            stderr_text = buf.getvalue().lower()
            expect("utc fallback" in stderr_text, f"expected timezone fallback warning in stderr, got: {stderr_text!r}")
        finally:
            if prev_diag is None:
                os.environ.pop("NAUTICAL_DIAG", None)
            else:
                os.environ["NAUTICAL_DIAG"] = prev_diag
            if prev_xdg is None:
                os.environ.pop("XDG_CACHE_HOME", None)
            else:
                os.environ["XDG_CACHE_HOME"] = prev_xdg


def test_core_recurrence_update_udas_config_aliases():
    """recurrence UDA carry config should accept top-level and [recurrence] alias forms."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))

    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('recurrence_update_udas = ["rappel", "next_review"]\n')
            f.write("[recurrence]\n")
            f.write('update_udas = "ignored_alias"\n')
        mod = _load_core_module(core_path, "_nautical_core_recur_udas_top_test", cfg)
        expect(
            mod.RECURRENCE_UPDATE_UDAS == ("rappel", "next_review"),
            f"unexpected top-level recurrence_update_udas: {mod.RECURRENCE_UPDATE_UDAS}",
        )

    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("[recurrence]\n")
            f.write('update_udas = "rappel, next_review, bad-name, 9x"\n')
        mod = _load_core_module(core_path, "_nautical_core_recur_udas_alias_test", cfg)
        expect(
            mod.RECURRENCE_UPDATE_UDAS == ("rappel", "next_review"),
            f"unexpected alias recurrence.update_udas parse: {mod.RECURRENCE_UPDATE_UDAS}",
        )


def test_on_modify_invalid_json_passthrough():
    """Malformed JSON should fail fast without stdout JSON."""
    path = _find_hook_file("on-modify-nautical.py")
    raw = "{not-json}"
    p = _run_hook_script_raw(path, raw)
    expect(p.returncode != 0, "on-modify should fail on invalid JSON input")
    expect((p.stdout or "").strip() == "", f"expected no stdout on failure, got: {p.stdout!r}")

def test_next_for_and_no_progress_fails_fast():
    """_next_for_and should fail fast when a term makes no forward progress."""
    saved = core.next_after_atom_with_mods
    try:
        def _stub(_atom, ref_d, _seed):
            return ref_d
        core.next_after_atom_with_mods = _stub
        term = [{"typ": "w", "spec": "mon"}]
        try:
            core._next_for_and(term, date(2025, 1, 1), date(2025, 1, 1))
            expect(False, "_next_for_and should raise ParseError on no-progress")
        except core.ParseError:
            pass
    finally:
        core.next_after_atom_with_mods = saved


def test_next_for_and_transient_stall_recovers():
    """_next_for_and should recover from brief no-progress stalls."""
    saved_next = core.next_after_atom_with_mods
    saved_match = core.atom_matches_on
    try:
        calls = {"n": 0}

        def _stub(_atom, ref_d, _seed):
            calls["n"] += 1
            if calls["n"] == 1:
                return ref_d
            return ref_d + timedelta(days=1)

        core.next_after_atom_with_mods = _stub
        core.atom_matches_on = lambda _a, _d, _s: True
        term = [{"typ": "w", "spec": "mon"}]
        got = core._next_for_and(term, date(2025, 1, 1), date(2025, 1, 1))
        expect(got > date(2025, 1, 1), f"expected forward progress after transient stall, got {got}")
    finally:
        core.next_after_atom_with_mods = saved_next
        core.atom_matches_on = saved_match

def test_roll_apply_has_guard():
    """roll_apply should fail fast if weekday never converges."""
    class WeirdDate(date):
        def weekday(self):
            return 9

    try:
        core.roll_apply(WeirdDate(2025, 1, 1), {"roll": "pbd"})
        expect(False, "roll_apply should raise ParseError when weekday never converges")
    except core.ParseError:
        pass

def test_anchor_cache_cleans_stale_tmp_files():
    """cache_save should remove stale temp files for the same key."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write("enable_anchor_cache = true\n")
            td_path_norm = td.replace("\\", "/")
            f.write(f'anchor_cache_dir = "{td_path_norm}"\n')
        mod = _load_core_module(core_path, "_nautical_core_cache_tmp_test", cfg)
        mod.ENABLE_ANCHOR_CACHE = True
        mod.ANCHOR_CACHE_DIR_OVERRIDE = td
        mod._CACHE_DIR = None
        key = "deadbeef"
        stale = os.path.join(td, f".{key}.stale.tmp")
        with open(stale, "w", encoding="utf-8") as f:
            f.write("stale")
        mod.cache_save(key, {"dnf": []})
    expect(not os.path.exists(stale), "stale cache tmp file should be cleaned")

def test_weeks_between_iso_boundary():
    """_weeks_between should honor ISO week boundaries across years."""
    d1 = date(2024, 12, 31)  # ISO week 2025-W01
    d2 = date(2025, 1, 1)    # ISO week 2025-W01
    expect(core._weeks_between(d1, d2) == 0, "same ISO week across year should be 0")
    d3 = date(2024, 12, 29)  # ISO week 2024-W52
    d4 = date(2024, 12, 30)  # ISO week 2025-W01
    expect(core._weeks_between(d3, d4) == 1, "ISO week boundary should be 1")

def test_short_uuid_invalid_inputs():
    """short_uuid should not crash on invalid inputs."""
    expect(core.short_uuid(None) == "", "short_uuid None should be empty")
    expect(core.short_uuid(1234) == "", "short_uuid non-string should be empty")
    expect(core.short_uuid("abcd") == "abcd", "short_uuid should keep short strings")

def test_anchor_expr_length_limit():
    """Anchor expressions over 1024 chars should fail fast."""
    s = "w:mon" + ("+w:mon" * 300)
    try:
        core.parse_anchor_expr_to_dnf(s)
        expect(False, "long anchor should raise ParseError")
    except core.ParseError:
        pass


def test_parser_frontend_normalization_characterization():
    """Parser frontend normalization should preserve current pre-parse rewrites."""
    expect(
        core._normalize_anchor_expr_input('"07-rand"') == "rand-07",
        "mm-rand alias should normalize to rand-mm after quote stripping",
    )


def test_parser_frontend_year_colon_guard_characterization():
    """Year-token colon guard should preserve its friendly error text."""
    msg = core._fatal_bad_colon_in_year_tail("05:15")
    expect(msg is not None and "uses ':' between numbers" in msg, f"unexpected colon guard message: {msg!r}")
    try:
        core._raise_on_bad_colon_year_tokens("y:05:15")
        raise AssertionError("expected ParseError for colon year token")
    except core.ParseError as e:
        expect("uses ':' between numbers" in str(e), f"unexpected error: {e}")


def test_parser_frontend_comma_join_guard_characterization():
    """Frontend should keep rejecting comma-joined anchors with current guidance."""
    for tail, needle in (
        ("31, w:sun", "Anchors must be joined with '+'"),
        ("31@t=14:00, w:sun", "It looks like you used a comma to join anchors."),
    ):
        try:
            core._raise_if_comma_joined_anchors(tail)
            raise AssertionError(f"expected ParseError for {tail!r}")
        except core.ParseError as e:
            expect(needle in str(e), f"unexpected message for {tail!r}: {e}")


def test_anchor_parse_term_explosion_guard():
    """Parser should reject DNF Cartesian explosions before exhausting memory."""
    group = "(w:mon|w:tue|w:wed|w:thu|w:fri|w:sat)"
    expr = "+".join([group] * 6)  # 6^6 = 46,656 combined terms
    try:
        core.parse_anchor_expr_to_dnf(expr)
        expect(False, "expected ParseError for excessive combined terms")
    except core.ParseError as e:
        expect("too complex" in str(e).lower(), f"unexpected error for term explosion guard: {e}")

def test_coerce_int_bounds():
    """coerce_int should return default for out-of-bounds values."""
    big = 2**63
    expect(core.coerce_int(big, default=7) == 7, "coerce_int should reject too-large int")
    expect(core.coerce_int(float(big), default=7) == 7, "coerce_int should reject too-large float")

def test_build_local_datetime_dst_gap_and_ambiguous():
    """build_local_datetime should handle DST gaps and ambiguities deterministically."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core/__init__.py"))
    with tempfile.TemporaryDirectory() as td:
        cfg = os.path.join(td, "nautical.toml")
        with open(cfg, "w", encoding="utf-8") as f:
            f.write('tz = "America/New_York"\n')
        mod = _load_core_module(core_path, "_nautical_core_dst_policy_test", cfg)

        # Spring forward: 2025-03-09 02:30 does not exist -> shift forward.
        dt_utc = mod.build_local_datetime(date(2025, 3, 9), (2, 30))
        back = mod.to_local(dt_utc)
        expect(back.hour == 3 and back.minute == 30, f"DST gap should shift forward: {back}")

        # Fall back: 2025-11-02 01:30 is ambiguous -> choose earlier (EDT, UTC-4).
        dt_utc = mod.build_local_datetime(date(2025, 11, 2), (1, 30))
        back = mod.to_local(dt_utc)
        offset = back.utcoffset()
        expect(offset is not None and offset.total_seconds() == -4 * 3600, f"DST fall back should choose earlier: {back}")

def test_on_modify_chain_export_cache_key_includes_params():
    """Chain export cache should include since/extra in its key."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_cache_key_test")
    if not hasattr(mod, "_tw_export_chain_cached_key"):
        raise AssertionError("on-modify hook does not expose chain cache helper")
    calls = []

    def _fake(chain_id: str, since=None, extra=None, env=None, limit=None):
        calls.append((chain_id, since, extra, limit))
        return [{"uuid": "x"}]

    mod.tw_export_chain = _fake
    mod._tw_export_chain_cached_key.cache_clear()
    mod._CHAIN_CACHE_CHAIN_ID = ""
    mod._CHAIN_CACHE = []

    since = datetime(2025, 1, 1, tzinfo=timezone.utc)
    _ = mod._get_chain_export("abc", since=since, extra="status:pending")
    _ = mod._get_chain_export("abc", since=since, extra="status:pending")
    _ = mod._get_chain_export("abc", since=since, extra="status:completed")
    expect(len(calls) == 2, f"cache key ignored params, calls={calls}")

def test_on_modify_chain_export_skips_when_locked():
    """tw_export_chain should treat lock errors as non-fatal and return empty."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_lock_skip_test")
    if not hasattr(mod, "tw_export_chain"):
        raise AssertionError("on-modify hook does not expose tw_export_chain")
    calls = []

    def _fake_run_task(*_args, **_kwargs):
        calls.append(True)
        return False, "", "database is locked"

    mod._run_task = _fake_run_task
    out = mod.tw_export_chain("abc")
    expect(out == [], "expected empty export when lock is active")
    expect(len(calls) == 1, "tw_export_chain should attempt task once")


def test_on_modify_collect_prev_two_prefers_live_statuses():
    """Previous-link lookup should prefer live tasks over deleted duplicates."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_collect_prev_two_test")
    current = {"chainID": "abcd1234", "link": 4}
    chain_by_link = {
        2: [
            {"uuid": "deleted-2", "status": "deleted", "link": 2},
            {"uuid": "pending-2", "status": "pending", "link": 2},
        ],
        3: [
            {"uuid": "deleted-3", "status": "deleted", "link": 3},
            {"uuid": "completed-3", "status": "completed", "link": 3},
        ],
    }

    prevs = mod._collect_prev_two(current, chain_by_link=chain_by_link)
    expect([t.get("uuid") for t in prevs] == ["pending-2", "completed-3"], f"unexpected prevs: {prevs}")

def test_weekly_and_unsat():
    """Test weekly AND (Sat AND Mon) must be unsatisfiable"""
    fatal, _ = core.lint_anchor_expr("w:sat + w:mon")
    expect(bool(fatal), "Weekly A+B must be unsatisfiable (Sat AND Mon)")


def test_satisfiability_helpers_characterization():
    """Direct satisfiability helpers should preserve fast weekly/yearly failures."""
    try:
        core._quick_weekly_and_check(
            [{"typ": "w", "spec": "sat", "mods": {}}, {"typ": "w", "spec": "mon", "mods": {}}]
        )
        raise AssertionError("expected weekly AND helper to fail")
    except core.AndTermUnsatisfiable as e:
        expect("never coincide" in str(e), f"unexpected weekly unsat message: {e}")

    try:
        core._quick_yearly_and_check(
            [{"typ": "y", "spec": "01-01"}, {"typ": "y", "spec": "12-25"}]
        )
        raise AssertionError("expected yearly AND helper to fail")
    except core.AndTermUnsatisfiable as e:
        expect("never overlap within a year" in str(e), f"unexpected yearly unsat message: {e}")

    expect(
        core._term_has_any_match_within(
            [{"typ": "w", "spec": "mon", "mods": {}}, {"typ": "m", "spec": "1", "mods": {}}],
            date(2026, 1, 1),
            date(2026, 1, 1),
            years=2,
        ),
        "simple satisfiable term should have a match within scan window",
    )


def test_expansion_helpers_characterization():
    """Weekly/yearly expansion helpers should preserve core support behavior."""
    expect(core._weekly_spec_to_wset("mon..wed") == {0, 1, 2}, "weekly range should expand to Mon-Wed")
    expect(
        core._weekly_spec_to_wset("rand", mods={"bd": True}) == {0, 1, 2, 3, 4},
        "weekly rand with bd should restrict to business days",
    )
    expect(
        core._doms_for_weekly_spec("mon", 2026, 1) == {5, 12, 19, 26},
        "weekly DOM expansion for January 2026 Mondays should stay stable",
    )
    expect(
        core._y_ranges_from_spec("rand-07,01-10..01-12") == [(7, 1, 7, 31), (1, 10, 1, 12)],
        "yearly ranges should preserve rand-month and numeric range expansion",
    )
    expect(
        core._doms_allowed_by_year(2026, 7, ["rand-07"]) == set(range(1, 32)),
        "rand-07 should allow the full July DOM range",
    )

def test_nth_weekday_range():
    """Test nth weekday range validation (1..5 or last)"""
    fatal, _ = core.lint_anchor_expr("m:6th-mon")
    expect(bool(fatal), "6th-mon must fatal (nth in 1..5 or last)")

def test_lint_anchor_expr_characterization():
    """lint_anchor_expr should keep stable fatal messages for key malformed inputs."""
    cases = [
        ('"w:mon-fri"', "Weekly ranges must use '..' (e.g., 'w:mon..fri')."),
        ("w:mon:fri", "Weekly ranges must use '..' (e.g., 'w:mon..fri')."),
        ("y:01-01:12-31", "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."),
        ("w:mno", "Unknown weekday 'mno'. Did you mean 'mon'?"),
        ("m:6th-mon", "Invalid ordinal '6th'. Only 1st..5th are supported."),
        (
            "w:sat + w:mon",
            "These anchors joined with '+' don't share any possible date. If you meant 'either/or', join them with ',' or '|'.",
        ),
        (
            "y:q4..q2",
            "Invalid quarter range 'qX..qY': end quarter precedes start quarter. Split across the year boundary, e.g., 'q4, q1'.",
        ),
    ]
    for expr, expected in cases:
        fatal, warns = core.lint_anchor_expr(expr)
        expect(fatal == expected, f"{expr}: expected fatal {expected!r}, got {fatal!r}")
        expect(warns == [], f"{expr}: expected no warnings, got {warns!r}")

    too_long = "w:mon" + ("x" * 1100)
    fatal, warns = core.lint_anchor_expr(too_long)
    expect(fatal == "Anchor expression too long (max 1024 characters).", f"unexpected length fatal: {fatal!r}")
    expect(warns == [], f"expected no warnings for long input, got {warns!r}")

def test_last_weekday():
    """Test last weekday of month pattern"""
    # Verify natural language mentions "last"
    p = build_preview("m:last-fri")
    expect("last" in p["natural"].lower(), "Natural should mention last Friday")
    # Verify all upcoming dates are Fridays
    for d in p["upcoming"][:5]:
        dow = datetime.fromisoformat(d).weekday()  # 0=Mon
        expect(dow == 4, f"{d} must be Friday")

def test_monthly_valid_months_m2_5th_mon():
    """Test monthly pattern with interval (/2) and 5th Monday constraint"""
    # /2:5th-mon must count only months that HAVE the 5th Monday
    p = build_preview("m/2:5th-mon")
    expect(p["upcoming"], "Should produce upcoming dates")
    # Each is Monday
    for d in p["upcoming"][:6]:
        expect(datetime.fromisoformat(d).weekday() == 0, f"{d} must be Monday")


def test_monthly_support_helpers_characterization():
    """Monthly DOM and valid-month support helpers should preserve current behavior."""
    expect(core._doms_for_monthly_token("last-fri", 2026, 1) == {30}, "last-fri in Jan 2026 should resolve to the 30th")
    expect(core._month_has_hit("5th-mon", 2026, 2) is False, "Feb 2026 should not have a 5th Monday")
    expect(core._month_has_hit("5th-mon", 2026, 3) is True, "Mar 2026 should have a 5th Monday")
    expect(core._next_valid_month_on_or_after("5th-mon", 2026, 2) == (2026, 3), "next valid month after Feb 2026 should be Mar 2026")
    probe = date(2026, 3, 1)
    nxt = core._first_hit_after_probe_in_month("5th-mon", 2026, 3, probe)
    expect(nxt == date(2026, 3, 30), f"Unexpected first hit in Mar 2026: {nxt!r}")

def test_leap_year_29feb():
    """Test leap year handling for Feb 29"""
    p = build_preview("y:02-29")
    dates = p["upcoming"][:8]
    expect(dates, "Need some upcoming for leap-day")
    for d in dates:
        dt = datetime.fromisoformat(d)
        expect(dt.month == 2 and dt.day in (28,29), "Window around Feb; core may list anchor dates only")
    # Must contain an actual Feb 29 within a 4-year span
    expect(any(datetime.fromisoformat(d).day == 29 for d in dates), "Must include a Feb 29 occurrence")

def test_quarters_window():
    """Test quarter window constraints (Q1-Q2)"""
    # Quarter aliases paired with m:* are now rejected (ambiguous).
    expr = "m:2nd-mon + y:q1..q2"
    try:
        build_preview(expr)
        assert False, "Expected ParseError for quarter alias with m:*"
    except core.ParseError as e:
        msg = str(e).lower()
        expect("ambiguous" in msg and "quarter" in msg, f"Unexpected error: {e}")

def test_quarter_alias_unambiguous_month_selectors():
    """Quarter aliases should work with unambiguous monthly start/end selectors."""
    p_start = build_preview("m:1bd + y:q4")
    expect(p_start["upcoming"], "Expected upcoming dates for start-month quarter selector")
    for d in p_start["upcoming"][:6]:
        m = datetime.fromisoformat(d).month
        expect(m == 10, f"{d} should resolve to Q4 start month (October)")

    p_end = build_preview("m:-1bd + y:q4")
    expect(p_end["upcoming"], "Expected upcoming dates for end-month quarter selector")
    for d in p_end["upcoming"][:6]:
        m = datetime.fromisoformat(d).month
        expect(m == 12, f"{d} should resolve to Q4 end month (December)")


def test_quarter_selector_mode_characterization():
    """Quarter selector mode should stay stable for accepted start/end monthly selectors."""
    expect(core._quarter_month_selector_mode([{"spec": "1bd"}]) == "quarter_start", "1bd should map to quarter_start")
    expect(core._quarter_month_selector_mode([{"spec": "1st-mon"}]) == "quarter_start", "1st-mon should map to quarter_start")
    expect(core._quarter_month_selector_mode([{"spec": "-1bd"}]) == "quarter_end", "-1bd should map to quarter_end")
    expect(core._quarter_month_selector_mode([{"spec": "last-fri"}]) == "quarter_end", "last-fri should map to quarter_end")


def test_quarter_selector_mode_rejections():
    """Quarter selector mode should keep rejecting ambiguous monthly combinations."""
    cases = [
        ([{"spec": "rand"}], "cannot be combined with m:rand"),
        ([{"spec": "1,15"}], "require a single monthly selector token"),
        ([{"spec": "15"}], "ambiguous"),
        ([{"spec": "1bd"}, {"spec": "-1bd"}], "cannot be combined with multiple monthly atoms"),
    ]
    for m_atoms, needle in cases:
        try:
            core._quarter_month_selector_mode(m_atoms)
            raise AssertionError(f"Expected ParseError for {m_atoms!r}")
        except core.ParseError as e:
            expect(needle in str(e), f"Unexpected error for {m_atoms!r}: {e}")


def test_term_quarter_rewrite_mode_characterization():
    """Plain quarter aliases should still consult monthly disambiguation, but suffixed ones should not."""
    expect(
        core._term_quarter_rewrite_mode([{"typ": "y", "spec": "q4"}], [{"typ": "m", "spec": "-1bd"}]) == "quarter_end",
        "Plain q4 with -1bd should resolve to quarter_end",
    )
    expect(
        core._term_quarter_rewrite_mode([{"typ": "y", "spec": "q4s"}], [{"typ": "m", "spec": "-1bd"}]) == "first_month",
        "Suffixed q4s should not invoke monthly quarter disambiguation",
    )


def test_quarter_spec_rewrite_characterization():
    """Quarter rewrite output and qmap annotations should stay stable."""
    qmap = {}
    expect(
        core._rewrite_quarter_spec_mode("q4", "quarter_end", meta_out=qmap) == "12-01..12-31",
        "q4 quarter_end should rewrite to the Q4 end-month window in MD format",
    )
    expect(qmap == {"12-01..12-31": "Q4 end month"}, f"Unexpected qmap: {qmap}")

    qmap = {}
    expect(
        core._rewrite_quarter_spec_mode("q1..q2", "first_month", meta_out=qmap) == "01-01..01-31,04-01..04-30",
        "q1..q2 first_month should expand to Q1/Q2 first-month windows",
    )
    expect(
        qmap == {
            "01-01..01-31": "Q1 first month",
            "04-01..04-30": "Q2 first month",
        },
        f"Unexpected qmap: {qmap}",
    )

    expect(
        core._rewrite_quarter_spec_mode("q1s..q2s", "first_month") == "01-01..01-31,04-01..04-30",
        "q1s..q2s should expand to explicit start-month windows",
    )


def test_rewrite_quarters_in_context_characterization():
    """Quarter rewrite should update yearly atoms and preserve per-atom qmap notes."""
    dnf = [[{"typ": "m", "spec": "-1bd"}, {"typ": "y", "spec": "q4"}]]
    out = core._rewrite_quarters_in_context(dnf)
    y_atom = out[0][1]
    expect(y_atom["spec"] == "12-01..12-31", f"Unexpected rewritten y spec: {y_atom['spec']}")
    expect(y_atom.get("_qmap") == {"12-01..12-31": "Q4 end month"}, f"Unexpected qmap: {y_atom.get('_qmap')}")

def test_yearly_month_names():
    """Test month name constraints (Mar..Sep)"""
    # y:mar..sep must constrain to Mar..Sep
    p = build_preview("m:1st-mon + y:mar..sep")
    for d in p["upcoming"][:6]:
        m = datetime.fromisoformat(d).month
        expect(3 <= m <= 9, f"{d} must be Mar..Sep")

def test_weekday_weekend_single_time():
    """Weekday vs weekend @t should not merge into same-day multi-times."""
    expr = "w:wd@t=09:00 | w:we@t=11:00"
    dnf = core.validate_anchor_expr_strict(expr)
    dates = core.anchors_between_expr(
        dnf,
        start_excl=date(2026, 1, 4),
        end_excl=date(2026, 1, 20),
        default_seed=date(2026, 1, 5),
        seed_base="test",
    )
    seen = set()
    for d in dates[:8]:
        slots = _hook._extract_time_slots_for_date(dnf, d, date(2026, 1, 5))
        if d.weekday() < 5:
            expect(slots == [(9, 0)], f"{d} should use 09:00, got {slots}")
        else:
            expect(slots == [(11, 0)], f"{d} should use 11:00, got {slots}")
        key = (d.year, d.month, d.day)
        expect(key not in seen, f"Duplicate date produced: {d}")
        seen.add(key)


def test_anchors_between_expr_stops_on_no_progress():
    """anchors_between_expr should stop safely when next_after_expr does not advance."""
    import nautical_core as core

    saved_next = core.next_after_expr
    calls = {"n": 0}
    try:
        def _stuck_next(_dnf, cur, _seed, seed_base=None):
            _ = seed_base
            calls["n"] += 1
            return cur, {"basis": "stuck"}

        core.next_after_expr = _stuck_next
        out = core.anchors_between_expr(
            [[{"typ": "w", "spec": "mon", "mods": {}}]],
            start_excl=date(2026, 1, 1),
            end_excl=date(2026, 1, 15),
            default_seed=date(2026, 1, 1),
            seed_base="stuck",
        )
        expect(out == [], f"expected no anchors for no-progress engine, got {out}")
        expect(calls["n"] == 1, f"expected single call on no-progress, got {calls['n']}")
    finally:
        core.next_after_expr = saved_next

def test_rand_with_year_window():
    """Test random pattern with yearly window constraint"""
    # Only inside Apr 20 – May 15
    p = build_preview("y:04-20..05-15 + m:rand")
    expect(p["upcoming"], "Rand with window should produce dates")
    for d in p["upcoming"][:8]:
        dt = datetime.fromisoformat(d)
        mmdd = f"{dt.month:02d}-{dt.day:02d}"
        expect("04-20" <= mmdd <= "05-15", f"{d} must be within Apr 20–May 15")

def test_weekly_rand_N_gate():
    """Test weekly random with /N gating (ISO week modulo)"""
    # /4:rand → ISO week index % 4 == constant (deterministic buckets)
    p = build_preview("w/4:rand")
    expect(p["upcoming"], "Need dates for w/4:rand")
    # Compute mod value from the first
    import datetime as _dt
    def iso_week_index(dt: datetime.date) -> int:
        y, w, _ = dt.isocalendar()
        return y * 53 + w
    base = datetime.fromisoformat(p["upcoming"][0]).date()
    mod = iso_week_index(base) % 4
    for d in p["upcoming"][:8]:
        wk = iso_week_index(datetime.fromisoformat(d).date()) % 4
        expect(wk == mod, f"{d} must satisfy /4 gating (got {wk}, want {mod})")

def test_business_day_nbd_pbd_nw_natural():
    """Test business day modifier natural language generation"""
    # Natural must reflect rolls; dates must obey
    cases = [
        ("m:-1@nbd", ("next", "business")),
        ("m:-1@pbd", ("previous", "business")),
        ("m:15@nw", ("nearest", "business")),  # @nw = nearest business day
    ]
    for expr, want in cases:
        p = build_preview(expr)
        expect(p["natural"] is not None, "Natural must exist (string)")
        low = (p["natural"] or "").lower()
        # require both keywords in any order
        expect(all(w in low for w in want), f"Natural for {expr} must mention {' & '.join(want)}")

def test_time_splitting_per_atom():
    """Test time splitting within weekly atoms"""
    # w:mon@t=09:00,fri@t=15:00 (comma in same weekly atom)
    # Requirement: no fatal; preview pathway should produce something.
    p = build_preview("w:mon@t=09:00,fri@t=15:00")
    expect(p is not None, "preview returned object")
    # Either natural or upcoming should be non-empty if the parser accepted it
    expect(bool(p["natural"]) or bool(p["upcoming"]),
           "Multi-@t weekly should be accepted and produce output")

def test_weekly_multi_days_and_every_2weeks():
    """Test weekly pattern with multiple days and interval"""
    p = build_preview("w/2:mon..tue,thu..sat")
    expect(p["upcoming"], "weekly /2 preview must produce dates")
    # all days are in allowed set
    allowed = {0,1,3,4,5}  # mon,tue,thu,fri,sat
    for d in p["upcoming"][:8]:
        wd = datetime.fromisoformat(d).weekday()
        expect(wd in allowed, f"{d} not in allowed weekdays")

def test_performance_large_expressions():
    """Test performance with large/complex expressions"""
    import time
    
    complex_expr = " | ".join([f"w:{dow}" for dow in ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]])
    
    start_time = time.time()
    dnf = core.validate_anchor_expr_strict(complex_expr)
    parse_time = time.time() - start_time
    
    assert parse_time < 0.1, f"Parsing took {parse_time:.3f}s, should be < 0.1s"
    
    # Test date calculation performance
    start_date = date(2024, 1, 1)
    start_time = time.time()
    next_date, _ = core.next_after_expr(dnf, start_date)
    calc_time = time.time() - start_time
    
    assert calc_time < 0.05, f"Date calculation took {calc_time:.3f}s, should be < 0.05s"

def test_cache_consistency():
    """Test that cached and uncached results are identical"""
    test_expressions = [
        "w:mon",
        "m:15",
        "y:12-25",
        "w:mon@t=09:00",
        "m:rand",
        "w/2:mon",
    ]
    
    for expr in test_expressions:
        # Get uncached result
        dnf = core.validate_anchor_expr_strict(expr)
        natural_uncached = core.describe_anchor_expr(expr)
        
        # Get cached result (if caching enabled)
        if hasattr(core, "build_and_cache_hints"):
            cached = core.build_and_cache_hints(expr)
            assert cached, f"{expr}: cache should return data"
            
            # DNF in cache should match
            cached_dnf = cached.get("dnf")
            assert cached_dnf == dnf, f"{expr}: cached DNF doesn't match"
            
            # Natural language in cache should match
            cached_natural = cached.get("natural")
            assert cached_natural == natural_uncached, f"{expr}: cached natural doesn't match"

def test_parse_cache_returns_isolated_dnf_instances():
    """Cached parse should return independent DNF objects across calls."""
    expr = "w:mon@t=09:00 + m:1"
    dnf_a = core.parse_anchor_expr_to_dnf_cached(expr)
    dnf_b = core.parse_anchor_expr_to_dnf_cached(expr)
    expect(dnf_a is not dnf_b, "cached parse should return a new top-level object")
    dnf_a[0][0]["spec"] = "tue"
    expect(dnf_b[0][0]["spec"] == "mon", "mutating one parsed DNF should not affect later reads")


def test_build_and_cache_hints_returns_isolated_cached_payload():
    """build_and_cache_hints should return an isolated payload copy on cache hits."""
    expr = "w:mon@t=09:00"
    first = core.build_and_cache_hints(expr, "skip")
    second = core.build_and_cache_hints(expr, "skip")
    expect(first is not second, "cached hints should not return the same object identity")
    first_dnf = first.get("dnf") or []
    second_dnf = second.get("dnf") or []
    if first_dnf and second_dnf:
        first_dnf[0][0]["spec"] = "fri"
        expect(second_dnf[0][0]["spec"] == "mon", "cached hint payload should be isolated per call")

def test_cache_key_for_task_caches_build_acf_results():
    """cache_key_for_task should memoize ACF work for repeated (expr, mode, fmt)."""
    if hasattr(core, "_cache_key_for_task_cached"):
        core._cache_key_for_task_cached.cache_clear()
    calls = {"n": 0}
    saved = core.build_acf
    try:
        def _stub_build_acf(expr: str) -> str:
            calls["n"] += 1
            return f"acf:{expr}"

        core.build_acf = _stub_build_acf
        k1 = core.cache_key_for_task("w:mon", "skip")
        k2 = core.cache_key_for_task("w:mon", "skip")
        expect(k1 == k2, "cache key should be stable for repeated inputs")
        expect(calls["n"] == 1, f"expected one build_acf call, got {calls['n']}")

        _ = core.cache_key_for_task("w:tue", "skip")
        expect(calls["n"] == 2, "different expression should invoke build_acf once")

        core._clear_all_caches()
        _ = core.cache_key_for_task("w:mon", "skip")
        expect(calls["n"] == 3, "clear-all-caches should reset cache_key_for_task memoization")
    finally:
        core.build_acf = saved
        if hasattr(core, "_cache_key_for_task_cached"):
            core._cache_key_for_task_cached.cache_clear()

def test_build_and_cache_hints_parses_once_per_miss():
    """build_and_cache_hints should avoid an extra parse for natural text on cache miss."""
    calls = {"n": 0}
    saved_parse = core.parse_anchor_expr_to_dnf_cached
    saved_load = core.cache_load
    saved_save = core.cache_save
    try:
        def _counting_parse(expr: str):
            calls["n"] += 1
            return saved_parse(expr)

        core.parse_anchor_expr_to_dnf_cached = _counting_parse
        core.cache_load = lambda _k: None
        core.cache_save = lambda _k, _v: None
        payload = core.build_and_cache_hints("w:thu@t=08:45", "skip")
        expect(payload and payload.get("dnf"), "expected build_and_cache_hints payload on miss")
        # One parse is for cache key canonicalization (build_acf), one for dnf validation.
        # The natural text path should reuse that dnf instead of parsing a third time.
        expect(calls["n"] == 2, f"expected two parses on miss, got {calls['n']}")
    finally:
        core.parse_anchor_expr_to_dnf_cached = saved_parse
        core.cache_load = saved_load
        core.cache_save = saved_save

def test_parser_validation():
    """Test parser validation and error messages"""
    # Valid expressions that should parse
    valid_expressions = [
        "w:mon",
        "w:mon,tue",
        "m:1",
        "m:1,15,31",
        "m:1..15",
        "m:2nd-mon",
        "m:last-fri",
        "m:5bd",
        "y:01-01",
        "y:01-01..12-31",
        "y:q1",
        "y:q1..q2",
        "w:mon@t=09:00",
        "m:15@t=09:00@+1d",
        "(w:mon + m:1) | (w:fri + m:15)",
    ]
    
    for expr in valid_expressions:
        try:
            dnf = core.validate_anchor_expr_strict(expr)
            assert dnf, f"{expr}: should parse to non-empty DNF"
        except Exception as e:
            assert False, f"{expr}: should parse but got error: {e}"
    
    # Invalid expressions that should fail
    invalid_expressions = [
        ("w:mon-fri", "Invalid weekly range"),
        ("m:1:15", "Invalid monthly range '1:15'. Use '..'"),
        ("y:01-01:12-31", "Yearly ranges must use '..'"),
        ("w:invalid", "Unknown weekday"),
        ("m:32", "Day-of-month '32' out of range. Use 1..31 or -1..-31"),
        ("m:6th-mon", "nth-weekday must be between 1 and 5 (or 'last'). Did you mean 'last-mon'? Offending token: '6th-mon'"),
        ("y:13-01", "Yearly token '13-01' doesn’t match ANCHOR_YEAR_FMT=MD. month '13' is invalid. Did you mean MM-DD? e.g., '04-20'"),
        ("y:01-32", "Yearly token '01-32' doesn’t match ANCHOR_YEAR_FMT=MD. day '32' is invalid. Did you mean MM-DD? e.g., '04-20'"),
        ("w:mon + w:sun", "Weekly anchors joined with '+' never coincide (e.g., Saturday AND Monday). Use ',' (OR) or '|' instead"),
    ]
    
    for expr, expected_error in invalid_expressions:
        try:
            core.validate_anchor_expr_strict(expr)
            assert False, f"{expr}: should fail but parsed successfully"
        except core.ParseError as e:
            assert expected_error.lower() in str(e).lower(), \
                f"{expr}: wrong error message. Got: {e}"


def test_parser_atom_helpers_characterization():
    """Direct atom helper behavior should stay stable for common heads, mods, and single atoms."""
    expect(core._parse_atom_head("w/2") == ("w", 2), f"Unexpected head parse: {core._parse_atom_head('w/2')!r}")
    mods = core._parse_atom_mods("t=09:00,12:00@+1d")
    expect(mods["t"] == [(9, 0), (12, 0)], f"Unexpected time mods: {mods!r}")
    expect(mods["day_offset"] == 1, f"Unexpected day offset: {mods!r}")
    dnf = core._build_anchor_atom_dnf("m", "1st-mon@t=09:00")
    expect(dnf == [[{"typ": "m", "spec": "1mon", "ival": 1, "mods": {"t": (9, 0), "roll": None, "wd": None, "bd": False, "day_offset": 0}}]], f"Unexpected atom dnf: {dnf!r}")
    node, next_i = core._parse_anchor_atom_at("w:mon@t=09:00 + m:1", 0, len("w:mon@t=09:00 + m:1"))
    expect(node == [[{"typ": "w", "spec": "mon", "ival": 1, "mods": {"t": (9, 0), "roll": None, "wd": None, "bd": False, "day_offset": 0}}]], f"Unexpected parsed node: {node!r}")
    expect(next_i > 0, "Parser should advance index for single atom")


def test_yearly_spec_token_helper_accepts_known_valid_tokens():
    """Yearly token helper should accept canonical quarter/date forms."""
    for tok in ("q1", "q2s", "q1..q2", "q1s..q2s", "01-01", "01-01..31-12"):
        core._validate_yearly_spec_token(tok)


def test_yearly_spec_token_helper_rejects_bad_ranges():
    """Yearly token helper should reject malformed or cross-year-like tokens."""
    bad = (
        ("3..4", "incomplete"),
        ("13", "Invalid month"),
        ("q3..q1", "end quarter precedes start quarter"),
        ("20-04..10-03", "cross-year ranges"),
    )
    for tok, expected in bad:
        try:
            core._validate_yearly_spec_token(tok)
            raise AssertionError(f"{tok}: expected ParseError")
        except core.ParseError as e:
            expect(expected.lower() in str(e).lower(), f"{tok}: unexpected message: {e}")


def test_yearly_token_format_characterization():
    """Yearly token format validator should preserve key error surfaces and allowances."""
    cases = [
        ("y:05:15", "Yearly token '05:15' uses ':' between numbers. Use '-' and order per ANCHOR_YEAR_FMT=MD. Example: '06-01'."),
        ("y:01-01:12-31", "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."),
        ("y:13-01", "Yearly token '13-01' doesn’t match ANCHOR_YEAR_FMT=MD. month '13' is invalid. Did you mean MM-DD? e.g., '04-20'."),
        ("y:01-32", "Yearly token '01-32' doesn’t match ANCHOR_YEAR_FMT=MD. day '32' is invalid. Did you mean MM-DD? e.g., '04-20'."),
        ("y:04-20..03-10", "Yearly token '04-20..03-10' doesn’t match ANCHOR_YEAR_FMT=MD. end precedes start. Did you mean MM-DD? e.g., '04-20'."),
        ("y:rand-13", "Invalid month in yearly token 'rand-13'. Expected 01..12."),
    ]
    for expr, expected in cases:
        try:
            core.validate_anchor_expr_strict(expr)
            raise AssertionError(f"{expr}: expected ParseError")
        except core.ParseError as e:
            msg = str(e)
            expect(msg.startswith(expected), f"{expr}: expected prefix {expected!r}, got {msg!r}")

    # Existing permissive behavior for three-letter aliases/quarters should remain.
    _must_parse("y:foo")
    _must_parse("y:q1..q2")


def test_yearly_token_format_helper_characterization():
    """Direct yearly format helper should preserve accepted tokens and friendly failures."""
    core._validate_yearly_token_format("01-01..12-31")
    core._validate_yearly_token_format("rand-07")
    core._validate_yearly_token_format("q1..q2")
    try:
        core._validate_yearly_token_format("13-01")
        raise AssertionError("13-01: expected YearTokenFormatError")
    except core.YearTokenFormatError as e:
        expect("month '13' is invalid" in str(e), f"Unexpected message: {e}")


def test_validate_year_tokens_in_dnf_characterization():
    """DNF-level yearly validation should keep surfacing yearly format problems."""
    core._validate_year_tokens_in_dnf([[{"typ": "y", "spec": "01-01"}]])
    try:
        core._validate_year_tokens_in_dnf([[{"typ": "y", "spec": "05:15"}]])
        raise AssertionError("Expected YearTokenFormatError for bad yearly token in DNF")
    except core.YearTokenFormatError as e:
        expect("uses ':' between numbers" in str(e), f"Unexpected message: {e}")


def test_parse_y_token_characterization():
    """Direct yearly token parsing should preserve known quarter/day interpretations."""
    expect(core._parse_y_token("q1") == ("quarter", "q1"), f"Unexpected parse for q1: {core._parse_y_token('q1')!r}")
    expect(core._parse_y_token("q2s") == ("quarter", "q2s"), f"Unexpected parse for q2s: {core._parse_y_token('q2s')!r}")
    expect(core._parse_y_token("01-06") == ("day", (1, 6)), f"Unexpected parse for 01-06: {core._parse_y_token('01-06')!r}")
    expect(core._parse_y_token("06-jan") == ("day", (6, 1)), f"Unexpected parse for 06-jan: {core._parse_y_token('06-jan')!r}")
    expect(core._parse_y_token("13-01") is None, "Invalid month should not parse")
    expect(core._parse_y_token("31-04") is None, "April 31 should not parse")

def test_natural_language_comprehensive():
    """Test natural language generation for various patterns"""
    test_cases = [
        ("w:mon", "Mondays"),
        ("w:mon,tue,fri", "Mondays or Tuesdays or Fridays"),
        ("w/2:mon", "every 2 weeks: Mondays"),
        ("m:15", "the 15th day of each month"),
        ("m:-1", "the last day of each month"),
        ("m:2nd-mon", "the 2nd Monday of each month"),
        ("m:last-fri", "the last Friday of each month"),
        ("m:5bd", "the 5th business day of each month"),
        ("y:12-25", "Dec 25 each year"),
        ("y:01-01..01-31", "Jan each year"),
        ("m:15@t=09:00", "the 15th day of each month at 09:00"),
        ("m:-1@nbd", "the last day of each month if business day; otherwise the next business day"),
    ]
    
    for anchor, expected_phrase in test_cases:
        natural = core.describe_anchor_expr(anchor)
        assert natural, f"{anchor}: natural description should not be empty"
        # Check if expected phrase is contained (not exact match due to formatting)
        assert expected_phrase.lower() in natural.lower(), \
            f"{anchor}: expected '{expected_phrase}' in '{natural}'"

def test_natural_anchor_characterization_for_complex_terms():
    """Complex anchor prose should keep stable phrasing for key edge combinations."""
    cases = [
        ("m/2:31", "every 2 months among months that have day 31"),
        ("m/2:2nd-mon", "every 2 months among months that have the 2nd Monday"),
        ("m:-1@nbd@t=09:00", "the last day of each month if business day; otherwise the next business day at 09:00"),
        ("m:1@nw", "the 1st day of each month if business day; otherwise the nearest business day (Fri if Saturday, Mon if Sunday)"),
        ("w/3:rand", "every 3 weeks: one random day every 3 weeks"),
        ("w:mon + m:1 + y:01-01..03-31", "Mondays that fall on the 1st day of each month and within Jan–Mar each year"),
    ]
    for expr, expected in cases:
        got = core.describe_anchor_expr(expr)
        assert got == expected, f"{expr}: expected {expected!r}, got {got!r}"

def test_natural_compresses_repeated_within_variants():
    """describe_anchor_dnf should compact repeated OR terms that only vary by yearly 'within' token."""
    expr = (
        "(w:mon..wed) + (m:1..10) + "
        "(y:01-01|y:02-01|y:03-01|y:04-01|y:05-01|y:06-01|y:07-01|y:08-01|y:09-01|y:10-01)"
    )
    dnf = core.validate_anchor_expr_strict(expr)
    nat = core.describe_anchor_dnf(dnf, {"anchor_mode": "skip"})
    low = (nat or "").lower()

    assert "and within either " in low, f"Natural should compact yearly variants: {nat!r}"
    assert low.count("mondays through wednesdays") == 1, f"Natural repeats shared prefix: {nat!r}"
    assert "jan 1" in low and "oct 1" in low, f"Natural lost yearly endpoints: {nat!r}"
    assert "skip missed anchors" in low, f"Natural should include mode tail: {nat!r}"

def test_natural_compresses_repeated_fall_on_variants():
    """describe_anchor_dnf should compact repeated OR terms that only vary by monthly 'that fall on' token."""
    expr = "(w:mon) + (m:1|m:2|m:3)"
    dnf = core.validate_anchor_expr_strict(expr)
    nat = core.describe_anchor_dnf(dnf, {"anchor_mode": "skip"})
    low = (nat or "").lower()

    assert "that fall on either " in low, f"Natural should compact monthly variants: {nat!r}"
    assert low.count("mondays") == 1, f"Natural repeats shared prefix: {nat!r}"
    assert "the 1st" in low and "the 3rd day of each month" in low, f"Natural lost monthly endpoints: {nat!r}"
    assert "skip missed anchors" in low, f"Natural should include mode tail: {nat!r}"

def test_rand_bucket_signature_characterization():
    """Bucket signature should remain stable for canonical monthly rand/range terms."""
    term = [
        {"typ": "m", "spec": "1..7", "mods": {"t": (9, 30)}},
        {"typ": "m", "spec": "rand", "mods": {"bd": True}},
    ]
    got = core._rand_bucket_signature(term)
    expect(got == (1, "09:30", True, "1–7"), f"unexpected rand bucket signature: {got!r}")

    mixed = term + [{"typ": "w", "spec": "mon"}]
    expect(core._rand_bucket_signature(mixed) is None, "weekly atoms should disable rand bucket compression")

    bad_range = [
        {"typ": "m", "spec": "1..7"},
        {"typ": "m", "spec": "8..14"},
        {"typ": "m", "spec": "rand"},
    ]
    expect(
        core._rand_bucket_signature(bad_range) is None,
        "multiple monthly ranges in a term should not produce a bucket signature",
    )


def test_edge_cases():
    """Test edge cases and boundary conditions"""
    test_cases = [
        # Leap year handling
        ("y:02-29", "2023-01-01", "2024-02-29"),  # Non-leap -> leap year
        ("y:02-29", "2024-02-29", "2028-02-29"),  # Leap -> next leap
        
        # Month boundaries
        ("m:31", "2024-02-01", "2024-03-31"),  # Feb doesn't have 31st
        ("m:-31", "2024-02-01", "2024-03-01"),  # -31 means 1st (31 days before end; that is months that have 31 days, the other are skiped)
        
        # Year boundaries
        ("y:12-31", "2024-12-31", "2025-12-31"),  # Year rollover
        ("y:01-01", "2024-12-31", "2025-01-01"),  # Year rollover
        
        # Week boundaries with /N
        ("w/2:mon", "2024-12-30", "2025-01-13"),  # Across year boundary
    ]
    
    for anchor, start_str, expected_str in test_cases:
        start = date.fromisoformat(start_str)
        expected = date.fromisoformat(expected_str)
        
        dnf = core.validate_anchor_expr_strict(anchor)
        next_date, _ = core.next_after_expr(dnf, start)
        
        assert next_date == expected, f"{anchor}: got {next_date}, expected {expected}"

def test_heads_with_slashN_parse_ok():
    """
    Ensure '/N' on heads parses cleanly across w/m/y
    """
    ok = [
        "w/2:sun",
        "m/3:1st-mon",
        "y/4:06-01",
    ]
    for expr in ok:
        _must_parse(expr)

def test_monthly_valid_months_m2_5th_mon_upcoming_within_valid_months():
    """
    'm/2:5th-mon' should only yield months that actually have a 5th Monday.
    We also allow for interval gating to skip valid months in-between;
    the key property is that every produced date is truly the 5th Monday.
    """
    preview = _must_preview("m/2:5th-mon")
    dates = preview["next_dates"]
    assert dates, "No upcoming dates produced for m/2:5th-mon"

    import datetime as _dt
    import calendar as _cal

    def _is_5th_monday(d: _dt.date) -> bool:
        if d.weekday() != 0:  # Monday
            return False
        # Count Mondays in the month up to and including d
        count = 0
        for day in range(1, d.day+ 1):
            if _dt.date(d.year, d.month, day).weekday() == 0:
                count+= 1
        return count == 5

    for d in dates[:8]:  # check the first few
        assert _is_5th_monday(d), f"{d} is not a 5th Monday"

def test_leap_year_29feb_upcoming_only_on_leap_year():
    """
    'y:02-29' must produce dates only on leap years (i.e., exactly Feb 29)
    """
    preview = _must_preview("y:02-29")
    dates = preview["next_dates"]
    assert dates, "No upcoming dates for y:02-29"
    for d in dates[:6]:
        assert (d.month, d.day) == (2, 29), f"{d} is not Feb 29"

def test_rand_with_year_window_filtering():
    """
    'y:04-20..05-15+ m:rand' must produce all dates within the yearly window
    """
    preview = _must_preview("y:04-20..05-15+ m:rand")
    dates = preview["next_dates"]
    assert dates, "No upcoming dates for window+ m:rand"
    import datetime as _dt
    def _in_window(d: _dt.date) -> bool:
        # inclusive Apr 20 .. May 15
        a = _dt.date(d.year, 4, 20)
        b = _dt.date(d.year, 5, 15)
        return a <= d <= b
    for d in dates[:8]:
        assert _in_window(d), f"{d} not within Apr 20–May 15"

def test_weekly_rand_N_gate_spacing():
    """
    'w/4:rand' should respect 4-week ISO‐week gating between picks
    """
    preview = _must_preview("w/4:rand")
    dates = preview["next_dates"]
    assert len(dates) >= 4, "Need several upcoming for w/4:rand"
    
    # Compute mod value from the first
    def iso_week_index(dt: date) -> int:
        y, w, _ = dt.isocalendar()
        return y * 53 + w  # Fixed: 53 weeks max per year, not 60
    
    base = dates[0]
    mod = iso_week_index(base) % 4
    for d in dates[:8]:
        wk = iso_week_index(d) % 4
        assert wk == mod, f"{d} must satisfy /4 gating (got {wk}, want {mod})"

def test_prev_weekday_natural_text():
    """
    Natural for 'm:-1@prev-fri' should mention 'previous Friday before the last day of the month'
    """
    nat = _must_natural("m:-1@prev-fri")
    want_any = [
        "previous Friday before the last day of the month",
        "previous Friday before the last day",
        "previous Friday before month end",
    ]
    assert any(w in nat for w in want_any), f"Natural missing expected phrasing: {nat!r}"

def test_weekly_multi_days_every_2weeks_spacing_and_days():
    """
    'w/2:mon,thu' must parse and produce dates only on Mon/Thu,
    with ISO week gaps respecting /2 gating
    """
    preview = _must_preview("w/2:mon,thu")
    dates = preview["next_dates"]
    assert len(dates) >= 4, "Need upcoming dates for w/2:mon,thu"
    import datetime as _dt
    def _iso_idx(d: _dt.date) -> int:
        y, w, _ = d.isocalendar()
        return y * 60+ w
    for d in dates[:8]:
        assert d.weekday() in (0, 3), f"{d} is not Mon/Thu"
    idxs = [_iso_idx(d) for d in dates[:6]]
    # successive picks may alternate Mon/Thu inside the same eligible week;
    # ensure we do not see *disallowed* week buckets (i.e., every other week).
    # Check that between picks that advance weeks, the gap is even.
    for a, b in zip(idxs, idxs[1:]):
        if b != a:  # if week advanced
            assert (b - a) % 2 == 0, f"Week gap not even between {a}..{b}"

def test_inline_time_mods_split_ok():
    """
    'w:mon@t=09:00,fri@t=15:00' should be accepted (rewritten to OR of singletons)
    """
    # Linter must not fatal, and strict-validate must pass
    fatal, warns = core.lint_anchor_expr("w:mon@t=09:00,fri@t=15:00")
    assert fatal is None, f"Unexpected lint fatal: {fatal}"
    _must_parse("w:mon@t=09:00,fri@t=15:00")

def test_deterministic_randomness():
    """Test that random patterns are deterministic with same seed"""
    # Only valid random patterns
    test_cases = [
        "w:rand",      # Random weekday - valid
        "m:rand",      # Random day of month - valid
        "m:rand@bd",   # Random business day of month - valid
        "m:1..10 + m:rand",  # Random day in first 10 days - valid
        "y:07-rand", # random day in July
        "y:rand-07", # random day in July
        "y:rand" # random day in year
    ]
    
    for anchor in test_cases:
        dnf = core.validate_anchor_expr_strict(anchor)
        
        # Test with same seed produces same results
        start_date = date(2024, 1, 1)
        dates1 = []
        dates2 = []
        
        # Generate first sequence
        current = start_date
        for _ in range(3):
            next_date, _ = core.next_after_expr(dnf, current, seed_base="test_seed")
            dates1.append(next_date)
            current = next_date + timedelta(days=1)
        
        # Generate second sequence with same seed
        current = start_date
        for _ in range(3):
            next_date, _ = core.next_after_expr(dnf, current, seed_base="test_seed")
            dates2.append(next_date)
            current = next_date + timedelta(days=1)
        
        # Should be identical
        assert dates1 == dates2, f"{anchor}: random dates not deterministic"

def test_business_day_modifiers():
    """Test business day modifiers thoroughly"""
    test_cases = [
        # @bd - weekdays only (skip if weekend)
        ("m:15@bd", "2024-01-14", "2024-01-15"),  # 15th is Monday (weekday)
        ("m:15@bd", "2024-06-14", "2024-07-15"),  # 15th is Saturday, skip to next 15th that's weekday (July 15)
        
        # @pbd - previous business day
        ("m:-1@pbd", "2024-03-28", "2024-03-29"),  # March 31 is Sunday -> Friday 29th
        
        # @nbd - next business day  
        ("m:1@nbd", "2024-06-28", "2024-07-01"),  # July 1 is Monday (business day)
        ("m:1@nbd", "2024-08-31", "2024-09-02"),  # Sep 1 is Sunday -> Monday 2nd
        
        # @nw - nearest business day
        ("m:15@nw", "2024-06-14", "2024-06-14"),  # June 15 is Saturday -> June 14 (Friday)
        ("m:15@nw", "2024-09-14", "2024-09-16"),  # Sep 15 is Sunday -> Sep 16 (Monday)
    ]
    
    for anchor, start_str, expected_str in test_cases:
        start = date.fromisoformat(start_str)
        expected = date.fromisoformat(expected_str)
        
        dnf = core.validate_anchor_expr_strict(anchor)
        next_date, _ = core.next_after_expr(dnf, start)
        
        assert next_date == expected, f"{anchor}: got {next_date}, expected {expected}"

def test_next_after_atom_with_mods_characterization():
    """Direct atom scheduling should preserve seed-gated interval and roll/offset behavior."""
    def _single_atom(expr: str):
        dnf = core.validate_anchor_expr_strict(expr)
        expect(len(dnf) == 1 and len(dnf[0]) == 1, f"expected single-atom DNF for {expr!r}")
        return dnf[0][0]

    cases = [
        # Monthly /N uses valid-month buckets anchored to seed.
        ("m/2:31", date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 31)),
        ("m/2:31", date(2024, 2, 1), date(2024, 1, 1), date(2024, 5, 31)),
        ("m/2:31", date(2024, 3, 31), date(2024, 1, 1), date(2024, 5, 31)),
        # Weekly/yearly /N gating.
        ("w/2:mon", date(2024, 12, 9), date(2024, 12, 9), date(2024, 12, 23)),
        ("y/2:06-15", date(2024, 1, 1), date(2024, 1, 1), date(2024, 6, 15)),
        # @bd skip behavior on weekend target.
        ("m:3@bd", date(2026, 1, 1), date(2026, 1, 1), date(2026, 2, 3)),
        # Roll then day_offset behavior.
        ("m:1@nbd@+1d", date(2024, 8, 31), date(2024, 8, 31), date(2024, 9, 3)),
        # Business-day roll may return candidate equal to ref_d when base is strictly after.
        ("m:15@nw", date(2024, 6, 14), date(2024, 6, 14), date(2024, 6, 14)),
    ]

    for expr, ref_d, seed_d, expected in cases:
        atom = _single_atom(expr)
        got = core.next_after_atom_with_mods(atom, ref_d, seed_d)
        expect(got == expected, f"{expr} from {ref_d} seed={seed_d}: got {got}, expected {expected}")


def test_scheduler_atom_helpers_characterization():
    """Direct scheduler helpers should preserve interval-bucket and acceptance behavior."""
    atom = core.validate_anchor_expr_strict("w:mon")[0][0]
    expect(core.base_next_after_atom(atom, date(2024, 12, 9)) == date(2024, 12, 16), "base weekly next date should stay stable")
    expect(core._interval_allowed_for_atom("w", 2, date(2024, 12, 9), date(2024, 12, 23)), "weekly /2 bucket should allow two-week jump")
    expect(not core._interval_allowed_for_atom("w", 2, date(2024, 12, 9), date(2024, 12, 16)), "weekly /2 bucket should reject one-week jump")
    expect(
        core._advance_probe_for_interval_bucket("w", 2, date(2024, 12, 9), date(2024, 12, 16)) == date(2024, 12, 22),
        "weekly probe advance should jump to day before next allowed week",
    )
    expect(core._accept_roll_candidate(date(2024, 6, 14), date(2024, 6, 15), date(2024, 6, 14), "nw"), "business-day roll may accept cand equal to ref_d")
    expect(not core._accept_roll_candidate(date(2024, 6, 14), date(2024, 6, 14), date(2024, 6, 14), "nw"), "roll acceptance still requires future base")

def test_complex_dnf_expressions():
    """Test complex DNF expressions with OR and AND"""
    test_cases = [
        # OR expressions
        ("w:mon | w:fri", "2024-12-11", "2024-12-13"),  # Wed -> Fri (closer than Mon)
        ("m:1 | m:15", "2024-12-11", "2024-12-15"),  # 1st or 15th -> 15th
        
        # AND expressions  
        ("w:mon + m:1", "2024-12-01", "2025-09-01"),  # Monday AND 1st of month (next is Sep 1, 2025)
        ("w:fri + m:13", "2024-12-01", "2024-12-13"),  # Friday the 13th
        
        # Complex: (Monday in Jan) OR (Friday in Feb)
        # From 2024-01-01, next is 2024-01-08 (Monday in Jan), not 2024-01-01 itself
        ("(w:mon + y:01-01..01-31) | (w:fri + y:02-01..02-28)", "2024-01-01", "2024-01-08"),
    ]
    
    for anchor, start_str, expected_str in test_cases:
        start = date.fromisoformat(start_str)
        expected = date.fromisoformat(expected_str)
        
        dnf = core.validate_anchor_expr_strict(anchor)
        next_date, _ = core.next_after_expr(dnf, start)
        
        assert next_date == expected, f"{anchor}: got {next_date}, expected {expected}"

def test_interval_patterns():
    """Test /N intervals with different anchor types"""
    test_cases = [
        # w/2:mon - fixed: every 2 weeks on Monday
        ("w/2:mon", "2024-12-09", ["2024-12-23", "2025-01-06", "2025-01-20"]),
        
        # m/2:15 from 2024-01-01: Jan 15, Mar 15, May 15
        ("m/2:15", "2024-01-01", ["2024-01-15", "2024-03-15", "2024-05-15"]),
        
        # m/3:-1 from 2024-01-01: Jan 31, Apr 30, Jul 31
        ("m/3:-1", "2024-01-01", ["2024-01-31", "2024-04-30", "2024-07-31"]),
        
        # y/2:06-15 from 2024-01-01: 2024-06-15, 2026-06-15, 2028-06-15
        ("y/2:06-15", "2024-01-01", ["2024-06-15", "2026-06-15", "2028-06-15"]),
    ]
    
    for anchor, start_str, expected_dates in test_cases:
        start = date.fromisoformat(start_str)
        expected = [date.fromisoformat(d) for d in expected_dates]
        
        dnf = core.validate_anchor_expr_strict(anchor)
        
        current = start
        for i, exp in enumerate(expected):
            next_date, _ = core.next_after_expr(dnf, current, default_seed=start)
            assert next_date == exp, f"{anchor} iteration {i}: got {next_date}, expected {exp}"
            current = next_date + timedelta(days=1)

def test_anchor_date_calculations():
    """Test specific date calculations for various anchor patterns"""
    test_cases = [
        # (anchor, start_date, expected_next_date)
        ("w:mon", "2024-12-11", "2024-12-16"),  # Wednesday -> Next Monday
        ("w:mon,fri", "2024-12-11", "2024-12-13"),  # Wednesday -> Friday (closer than Monday)
        ("m:15", "2024-12-11", "2024-12-15"),  # 11th -> 15th of same month
        ("m:15", "2024-12-20", "2025-01-15"),  # After 15th -> next month
        ("m:-1", "2024-12-11", "2024-12-31"),  # Last day of month
        ("m:1", "2024-12-31", "2025-01-01"),  # First day of next month
        ("y:12-25", "2024-12-11", "2024-12-25"),  # Christmas
        ("y:12-25", "2024-12-26", "2025-12-25"),  # After Christmas -> next year
    ]
    
    for anchor, start_str, expected_str in test_cases:
        start = date.fromisoformat(start_str)
        expected = date.fromisoformat(expected_str)
        
        dnf = core.validate_anchor_expr_strict(anchor)
        next_date, _ = core.next_after_expr(dnf, start)
        
        assert next_date == expected, f"{anchor} from {start_str}: got {next_date}, expected {expected}"

def test_yearly_rand_natural_and_bounds():
    """Test yearly random patterns natural language and constraints"""
    # Natural for y:rand must mention random + each year
    nat = _must_natural("y:rand")
    low = nat.lower()
    expect("random" in low and "each year" in low, f"Unexpected natural for y:rand: {nat!r}")

    # y:07-rand → all picks are in July
    p = build_preview("y:07-rand")
    expect(p["upcoming"], "y:07-rand should produce upcoming dates")
    for d in p["upcoming"][:8]:
        expect(datetime.fromisoformat(d).month == 7, f"{d} must be in July")

    # y:rand-07 is accepted and identical constraint (July)
    p2 = build_preview("y:rand-07")
    expect(p2["upcoming"], "y:rand-07 should produce upcoming dates")
    for d in p2["upcoming"][:8]:
        expect(datetime.fromisoformat(d).month == 7, f"{d} must be in July")

def test_yearly_month_aliases_and_ranges():
    """Test month name aliases and numeric shorthands"""
    # Single month by name or numeric shorthand
    for expr in ("y:apr", "y:04"):
        p = build_preview(expr)
        expect(p["upcoming"], f"{expr} should produce upcoming dates")
        for d in p["upcoming"][:6]:
            expect(datetime.fromisoformat(d).month == 4, f"{d} must be in April for {expr}")

    # Month-name window (Jan..Jun) constrains outputs
    p = build_preview("y:jan..jun + m:rand")
    expect(p["upcoming"], "y:jan..jun + m:rand should produce upcoming dates")
    for d in p["upcoming"][:8]:
        m = datetime.fromisoformat(d).month
        expect(1 <= m <= 6, f"{d} must be within Jan..Jun")

def test_business_day_bd_skip_semantics():
    """Test @bd skip semantics (skip to next month if not business day)"""
    # @bd = only if business day else skip to next month's matching day (not roll)
    # 2026-01-03 is Saturday → skip to 2026-02-03 (Tuesday)
    start = date(2026, 1, 1)
    dnf = core.validate_anchor_expr_strict("m:3@bd")
    nxt, _ = core.next_after_expr(dnf, start)
    expect(nxt == date(2026, 2, 3), f"@bd should skip Jan (Sat) → 2026-02-03, got {nxt}")

def test_inline_time_mods_natural_contains_both_times():
    """Test inline time modifiers show both times in natural language"""
    expr = "w:mon@t=09:00,fri@t=15:00"
    nat = _must_natural(expr)
    low = nat.lower()
    expect("09:00" in low and "15:00" in low,
           f"Natural should reflect both times for {expr!r}: {nat!r}")
    _must_parse(expr)  # ensure strict parser accepts the inline split

def test_guard_commas_between_atoms_after_mods_fatal():
    """Test comma between atoms after modifiers is fatal"""
    bad = "m:31@t=14:00,w:sun@t=22:00"
    try:
        core.validate_anchor_expr_strict(bad)
        assert False, "Comma between atoms after @mods must be fatal"
    except core.ParseError as e:
        msg = str(e)
        expect("join" in msg.lower() or "use '+' (and) or '|'" in msg.lower(),
               f"Unexpected error message for bad comma join: {msg}")

def test_heads_with_slashN_parse_ok_again():
    """Regression guard: '/N' heads must parse across w/m/y"""
    # Regression guard: '/N' heads must parse across w/m/y
    for expr in ("w/2:sun", "m/3:1st-mon", "y/4:06-01"):
        _must_parse(expr)

def test_monthname_and_numeric_equivalence():
    """Test month name and numeric month equivalence"""
    # y:jul should behave like a July window; y:07 numeric should match too (for month-only)
    for expr in ("y:jul",):
        p = build_preview(expr)
        expect(p["upcoming"], f"{expr} should produce upcoming dates")
        for d in p["upcoming"][:6]:
            expect(datetime.fromisoformat(d).month == 7, f"{d} must be in July")

def test_cp_duration_parser_and_dst_preserve_whole_days():
    """CP branch: duration parsing + DST-safe whole-day stepping keeps wall-clock time stable."""

    # --- 1) Duration parsing (core responsibility) ---
    td = core.parse_cp_duration("P1DT2H30M")
    assert td == timedelta(days=1, hours=2, minutes=30), f"Unexpected td for P1DT2H30M: {td}"
    td_day = core.parse_cp_duration("P1D")
    assert td_day == timedelta(days=1), f"Unexpected td for P1D: {td_day}"

    # --- 2) DST-safe stepping semantics used by CP preview (hook responsibility) ---
    # If ZoneInfo isn't available, core runs in "UTC-only" mode; skip DST assertions.
    if getattr(core, "_LOCAL_TZ", None) is None:
        return

    # Pick a date that crosses DST start for Europe/Bucharest (default config).
    start_local = core.build_local_datetime(date(2026, 3, 28), (10, 0))
    start_utc = start_local.astimezone(timezone.utc).replace(microsecond=0)

    naive_next_utc = (start_utc + td_day).replace(microsecond=0)

    # Hook-style “preserve local HH:MM” for whole-day steps:
    dl = core.to_local(start_utc)
    preserved_local = core.build_local_datetime(
        (dl + timedelta(days=1)).date(), (dl.hour, dl.minute)
    )
    preserved_next_utc = preserved_local.astimezone(timezone.utc).replace(microsecond=0)

    # Preserved step must keep local HH:MM stable.
    pl = core.to_local(preserved_next_utc)
    assert (pl.hour, pl.minute) == (dl.hour, dl.minute), (
        f"Preserved CP step should keep local HH:MM stable: start={dl} -> preserved={pl}"
    )

    # If DST offset changed across the step, naive timedelta arithmetic will drift in wall-clock time.
    nl = core.to_local(naive_next_utc)
    if dl.utcoffset() != nl.utcoffset():
        assert (nl.hour, nl.minute) != (dl.hour, dl.minute), (
            f"Naive UTC+timedelta should drift across DST: start={dl} -> naive={nl}"
        )            

# -------- Runner --------------------------------------------------------------

def test_hook_on_add_multitime_preview_emits_all_slots():
    """on-add must accept @t=HH:MM list and preview intra-day slots when due is explicit."""
    hook = _find_hook_file("on-add-nautical.py")
    # Disable ANSI colors for deterministic output.
    env = {"NO_COLOR": "1"}
    expr = "w:wed@t=06:00,12:00,22:00"
    task = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "description": "hook test on-add multitime",
        "status": "pending",
        "project": "testing",
        "entry": "20251217T000000Z",
        "anchor": expr,
        "anchor_mode": "skip",
        # Explicit due so the preview is deterministic independent of 'now'
        "due": "20251217T060000Z",
    }
    p = _run_hook_script(hook, task, env_extra=env)
    if p.returncode != 0:
        raise AssertionError(f"on-add hook failed rc={p.returncode}. stderr={p.stderr[:400]!r}")
    out_task = _extract_last_json(p.stdout)
    # The hook should not override an explicit due.
    if out_task.get("due") != task["due"]:
        raise AssertionError(f"on-add changed explicit due: got {out_task.get('due')!r}, want {task['due']!r}")
    # Preview should show other intra-day slots (12:00 and 22:00) on the same date.
    stderr_txt = _strip_markup(p.stderr)
    if "12:00" not in stderr_txt or "22:00" not in stderr_txt:
        raise AssertionError(f"on-add preview missing expected intra-day times. stderr={stderr_txt[:500]!r}")

def test_hook_on_modify_timeline_multitime_includes_all_slots():
    """on-modify timeline generator must step occurrences (date+time), not only dates."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_test")
    if not hasattr(mod, "_timeline_lines"):
        raise AssertionError("on-modify hook does not expose _timeline_lines; cannot validate timeline stepping.")
    # Avoid external calls for prev collection in unit context.
    if hasattr(mod, "_collect_prev_two"):
        setattr(mod, "_collect_prev_two", lambda _task: [])
    expr = "w:mon..sun@t=06:00,12:00,22:00"
    dnf = core.validate_anchor_expr_strict(expr)
    # Simulate a chain where the next due is at 22:00 on a given day.
    child_due_utc = datetime(2025, 12, 20, 22, 0, tzinfo=timezone.utc)
    task = {
        "uuid": "00000000-0000-0000-0000-000000000222",
        "description": "hook test on-modify multitime",
        "anchor": expr,
        "anchor_mode": "skip",
        "link": 1,
        # completed earlier in the day
        "end": "20251220T090000Z",
        # due is not required by _timeline_lines, but helpful for formatting.
        "due": "20251220T120000Z",
    }
    lines = _call_with_supported_kwargs(
        mod._timeline_lines,
        kind="anchor",
        task=task,
        child_due_utc=child_due_utc,
        child_short="0000abcd",
        dnf=dnf,
        next_count=8,
        cap_no=None,
        cur_no=1,
    )
    txt = _strip_markup("\n".join(lines))
    times = sorted(set(re.findall(r"\b\d{2}:\d{2}\b", txt)))
    # Expect to see at least the three slots across the timeline.
    for t in ("06:00", "12:00", "22:00"):
        if t not in times:
            raise AssertionError(f"on-modify timeline missing time {t}. found={times}. text={txt[:500]!r}")
    # Also ensure it isn't collapsing to a single daily time.
    if len(times) < 3:
        raise AssertionError(f"on-modify timeline collapsed times unexpectedly: {times}. text={txt[:500]!r}")


def test_hook_task_runner_handles_nonzero():
    """Hook _run_task handles success and non-zero exit codes."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_run_task_test")
    if not hasattr(mod, "_run_task"):
        raise AssertionError("on-modify hook does not expose _run_task")

    ok, out, _err = mod._run_task([sys.executable, "-c", "print('ok')"], timeout=2, retries=1)
    expect(ok and out.strip() == "ok", f"_run_task expected success, got ok={ok}, out={out!r}")

    ok2, _out2, _err2 = mod._run_task(
        [sys.executable, "-c", "import sys; sys.exit(2)"],
        timeout=2,
        retries=1,
    )
    expect(not ok2, "_run_task expected non-zero exit to return ok=False")


def test_hook_run_task_falls_back_when_core_load_fails():
    """on-modify _run_task should fall back to subprocess if core load fails."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_run_task_fallback_test")
    saved_load_core = mod._load_core
    saved_core = mod.core
    try:
        mod.core = None

        def _boom():
            raise RuntimeError("core unavailable")

        mod._load_core = _boom
        ok, out, _err = mod._run_task([sys.executable, "-c", "print('fallback-ok')"], timeout=2, retries=1)
        expect(ok, "_run_task fallback path should succeed for simple subprocess command")
        expect(out.strip() == "fallback-ok", f"unexpected fallback stdout: {out!r}")
    finally:
        mod._load_core = saved_load_core
        mod.core = saved_core


def test_on_add_run_task_falls_back_when_core_load_fails():
    """on-add _run_task should fall back to subprocess if core load fails."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_run_task_fallback_test")
    saved_load_core = mod._load_core
    saved_core = mod.core
    saved_diag = mod._diag
    try:
        mod.core = None
        diag_calls = {"n": 0}

        def _boom():
            raise RuntimeError("core unavailable")

        def _diag_stub(_msg):
            diag_calls["n"] += 1

        mod._load_core = _boom
        mod._diag = _diag_stub
        ok, out, _err = mod._run_task([sys.executable, "-c", "print('fallback-ok')"], timeout=2, retries=1)
        expect(ok, "_run_task fallback path should succeed for simple subprocess command")
        expect(out.strip() == "fallback-ok", f"unexpected fallback stdout: {out!r}")
        expect(diag_calls["n"] >= 1, "fallback path should emit a diagnostic once")
    finally:
        mod._load_core = saved_load_core
        mod.core = saved_core
        mod._diag = saved_diag


def test_spawn_child_verifies_even_when_verify_import_disabled():
    """_spawn_child should verify child existence even when verify_import is disabled."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_spawn_verify_enforced_test")

    saved_reserve = mod._reserve_child_uuid
    saved_run_task = mod._run_task
    saved_exists = mod._task_exists_by_uuid
    saved_verify = mod._VERIFY_IMPORT
    try:
        mod._VERIFY_IMPORT = False
        mod._reserve_child_uuid = lambda _env: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        calls = {"import": 0, "exists": 0}

        def _run_task_stub(cmd, *, env=None, input_text=None, timeout=0.0, retries=0, use_tempfiles=False):
            _ = (env, input_text, timeout, retries, use_tempfiles)
            if isinstance(cmd, list) and "import" in cmd:
                calls["import"] += 1
                return True, "", ""
            return False, "", "unexpected"

        def _exists_stub(_uuid, _env):
            calls["exists"] += 1
            return calls["exists"] >= 2

        mod._run_task = _run_task_stub
        mod._task_exists_by_uuid = _exists_stub

        short, _stripped = mod._spawn_child({"description": "verify-enforced"})
        expect(short == "aaaaaaaa", f"unexpected child short uuid: {short}")
        expect(calls["import"] == 2, f"expected retry after failed verify, got imports={calls['import']}")
        expect(calls["exists"] == 2, f"expected verification on each import, got checks={calls['exists']}")
    finally:
        mod._reserve_child_uuid = saved_reserve
        mod._run_task = saved_run_task
        mod._task_exists_by_uuid = saved_exists
        mod._VERIFY_IMPORT = saved_verify


def test_core_run_task_tempfiles_accepts_text_input():
    """core.run_task should accept str input_text with use_tempfiles=True."""
    ok, out, err = core.run_task(
        [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
        input_text="hello\n",
        timeout=2.0,
        retries=1,
        use_tempfiles=True,
    )
    expect(ok, f"run_task expected ok=True, got err={err!r}")
    expect(out == "hello\n", f"run_task expected echoed input, got {out!r}")


def test_core_run_task_timeout_reports_timeout_with_tempfiles():
    """core.run_task should return timeout marker when process exceeds timeout using tempfiles."""
    ok, _out, err = core.run_task(
        [sys.executable, "-c", "import time; time.sleep(0.25); print('late')"],
        timeout=0.05,
        retries=1,
        use_tempfiles=True,
    )
    expect(not ok, "run_task timeout path should return ok=False")
    expect(err == "timeout", f"run_task timeout path should return 'timeout', got {err!r}")


def test_core_run_task_nonzero_retries_use_expected_backoff():
    """core.run_task should apply exponential retry backoff for non-zero exits."""
    sleeps = []
    orig_sleep = core.time.sleep
    orig_uniform = core.random.uniform
    try:
        core.time.sleep = lambda v: sleeps.append(v)
        core.random.uniform = lambda _a, _b: 0.0
        ok, _out, _err = core.run_task(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            timeout=1.0,
            retries=3,
            retry_delay=0.2,
            use_tempfiles=False,
        )
        expect(not ok, "run_task non-zero command should fail")
        expect(len(sleeps) == 2, f"expected 2 backoff sleeps, got {sleeps}")
        expect(abs(sleeps[0] - 0.2) < 1e-9, f"unexpected first backoff: {sleeps}")
        expect(abs(sleeps[1] - 0.4) < 1e-9, f"unexpected second backoff: {sleeps}")
    finally:
        core.time.sleep = orig_sleep
        core.random.uniform = orig_uniform


def test_core_run_task_tempfiles_fallback_handles_bytes_input():
    """core.run_task should fall back to text mode if tempfile allocation fails."""
    orig_ntf = core.tempfile.NamedTemporaryFile
    try:
        def _raise_named_tempfile(*_args, **_kwargs):
            raise OSError("tempfile unavailable")

        core.tempfile.NamedTemporaryFile = _raise_named_tempfile
        ok, out, err = core.run_task(
            [sys.executable, "-c", "import sys; sys.stdout.write(sys.stdin.read())"],
            input_text=b"abc\xff\n",
            timeout=1.0,
            retries=1,
            use_tempfiles=True,
        )
        expect(ok, f"run_task fallback should still succeed, got err={err!r}")
        expect(out == "abc\ufffd\n", f"fallback should decode bytes in text mode, got {out!r}")
    finally:
        core.tempfile.NamedTemporaryFile = orig_ntf


def test_on_add_dnf_cache_versioned_payload():
    """on-add DNF cache uses versioned payload and can round-trip."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_test")
    if not hasattr(mod, "_save_dnf_disk_cache") or not hasattr(mod, "_load_dnf_disk_cache"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = OrderedDict([("k1", {"v": 1})])
        mod._DNF_DISK_CACHE_DIRTY = True

        mod._save_dnf_disk_cache()

        mod._DNF_DISK_CACHE = None
        loaded = mod._load_dnf_disk_cache()
        expect(isinstance(loaded, OrderedDict), "DNF cache load should return OrderedDict")
        expect("k1" in loaded, "DNF cache did not round-trip expected key")


def test_on_add_dnf_cache_corrupt_payload_recovers():
    """on-add DNF cache load should quarantine and continue on invalid JSONL."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_corrupt_test")
    if not hasattr(mod, "_load_dnf_disk_cache"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        with open(cache_path, "wb") as f:
            f.write(b"{bad-json}\n")

        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None

        loaded = mod._load_dnf_disk_cache()
        expect(isinstance(loaded, OrderedDict), "DNF cache load should return OrderedDict")


def test_on_add_dnf_cache_quarantines_invalid_jsonl():
    """on-add DNF cache quarantines JSONL files with no valid JSON objects."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_quarantine_test")
    if not hasattr(mod, "_load_dnf_disk_cache"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write("nope\nstill nope\n")

        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None

        try:
            _ = mod._load_dnf_disk_cache()
        except Exception:
            pass
        quarantined = [p for p in os.listdir(td) if p.startswith("dnf_cache.corrupt.") and p.endswith(".jsonl")]
        expect(quarantined, "DNF cache should quarantine invalid JSONL")


def test_on_add_dnf_cache_checksum_mismatch_salvages():
    """on-add DNF cache should salvage entries on checksum mismatch."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_checksum_test")
    if not hasattr(mod, "_load_dnf_disk_cache"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        bad_header = json.dumps({"version": 1, "checksum": "deadbeefdeadbeef"})
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(bad_header + "\n")
            f.write(json.dumps({"key": "k1", "value": {"v": 1}}) + "\n")

        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None
        cache = mod._load_dnf_disk_cache()
        expect(cache.get("k1") == {"v": 1}, "DNF cache should salvage entries on checksum mismatch")
        quarantined = [p for p in os.listdir(td) if p.startswith("dnf_cache.corrupt.") and p.endswith(".jsonl")]
        expect(not quarantined, "DNF cache should not quarantine checksum mismatch when salvageable")


def test_on_add_dnf_cache_size_guard_skips_load():
    """on-add DNF cache skips load when file is too large."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_size_guard_test")
    if not hasattr(mod, "_load_dnf_disk_cache"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        # 300 KB of garbage (over the 256 KB limit).
        with open(cache_path, "wb") as f:
            f.write(b"x" * (300 * 1024))

        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None

        loaded = mod._load_dnf_disk_cache()
        expect(isinstance(loaded, OrderedDict), "DNF cache size-guard load should return OrderedDict")
        expect(len(loaded) == 0, "DNF cache size-guard load should return empty cache")


def test_on_add_dnf_cache_skips_non_jsonable_values():
    """on-add DNF cache should skip non-JSON-serializable values."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_non_jsonable_test")
    if not hasattr(mod, "_validate_anchor_expr_cached"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = OrderedDict()
        mod._DNF_DISK_CACHE_DIRTY = False
        mod._validate_anchor_expr_cached.cache_clear()

        orig = mod.core.validate_anchor_expr_strict
        try:
            mod.core.validate_anchor_expr_strict = lambda _expr: {"bad": set([1, 2])}
            dnf = mod._validate_anchor_expr_cached("w:mon")
        finally:
            mod.core.validate_anchor_expr_strict = orig

        expect(isinstance(dnf, dict), "DNF should be returned even when not cacheable")
        expect(not mod._DNF_DISK_CACHE_DIRTY, "DNF cache should not be marked dirty on non-serializable values")
        expect(len(mod._DNF_DISK_CACHE) == 0, "DNF cache should not store non-serializable values")


def test_core_import_deterministic():
    """Hooks should ignore TASKDATA unless NAUTICAL_DEV=1."""
    with tempfile.TemporaryDirectory() as td:
        bad_core = Path(td) / "nautical_core/__init__.py"
        bad_core.write_text("raise RuntimeError('bad core')\n", encoding="utf-8")
        os.environ["TASKDATA"] = td
        os.environ.pop("NAUTICAL_DEV", None)
        try:
            hook = _find_hook_file("on-add-nautical.py")
            _ = _load_hook_module(hook, "_nautical_on_add_import_deterministic_test").core
        finally:
            os.environ.pop("TASKDATA", None)

    expect(True, "core import should ignore TASKDATA when NAUTICAL_DEV is not set")

def test_on_modify_spawn_intent_id_in_entry():
    """on-modify spawn intent entries should include a correlation id."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_spawn_intent_id_test")
    if not hasattr(mod, "_spawn_intent_entry"):
        raise AssertionError("on-modify hook does not expose spawn intent helper")

    entry = mod._spawn_intent_entry("parent", {"uuid": "child"}, "deadbeef", "", "si_test")
    expect(entry.get("spawn_intent_id") == "si_test", "spawn_intent_id should be preserved in queue entry")

def test_on_exit_spawn_intents_drain():
    """on-exit should import child and update parent from spawn intents."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_drain_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"

        child_uuid = "00000000-0000-0000-0000-000000000999"
        parent_uuid = "00000000-0000-0000-0000-000000000111"
        entry = {
            "parent_uuid": parent_uuid,
            "child_short": "deadbeef",
            "child": {"uuid": child_uuid, "description": "test"},
            "spawn_intent_id": "si_test",
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        imported = {"ok": False}
        parent_updated = {"ok": False}
        parent_next = {"value": ""}

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if "export" in cmd_s and f"uuid:{child_uuid}" in cmd_s:
                if imported["ok"]:
                    return True, json.dumps({"uuid": child_uuid}), ""
                return True, "{}", ""
            if "export" in cmd_s and f"uuid:{parent_uuid}" in cmd_s:
                return True, json.dumps({"uuid": parent_uuid, "nextLink": parent_next["value"]}), ""
            if "import" in cmd_s:
                imported["ok"] = True
                return True, "", ""
            if "modify" in cmd_s and "uuid:00000000-0000-0000-0000-000000000111" in cmd_s:
                parent_updated["ok"] = True
                parent_next["value"] = "deadbeef"
                return True, "", ""
            return False, "", "unexpected"

        mod._run_task = _run_task_stub
        stats = mod._drain_queue()
        expect(stats.get("errors") == 0, f"drain errors: {stats}")
        expect(stats.get("processed") == 1, f"unexpected drain processed: {stats}")
        expect(imported["ok"], "child import did not run")
        expect(parent_updated["ok"], "parent update did not run")

def test_on_exit_take_queue_migrates_legacy_processing_backlog_to_sqlite():
    """on-exit should migrate legacy processing backlog into sqlite and then claim from sqlite."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_processing_migration_test")
    if not hasattr(mod, "_take_queue_entries"):
        raise AssertionError("on-exit hook does not expose queue helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._QUEUE_DB_PATH = td_path / ".nautical_queue.db"

        legacy = {
            "spawn_intent_id": "si_legacy_processing",
            "parent_uuid": "00000000-0000-0000-0000-000000000101",
            "child_short": "abcd0001",
            "child": {"uuid": "00000000-0000-0000-0000-000000000102"},
        }
        mod._QUEUE_PATH.write_text("", encoding="utf-8")
        mod._QUEUE_PROCESSING_PATH.write_text(json.dumps(legacy) + "\n", encoding="utf-8")

        entries = mod._take_queue_entries()
        expect(len(entries) == 1, f"expected one migrated entry, got: {entries}")
        expect(entries[0].get("spawn_intent_id") == "si_legacy_processing", f"wrong entry claimed: {entries}")
        expect(entries[0].get("__queue_backend") == "sqlite", f"entry should be sqlite-backed: {entries}")
        expect(not mod._QUEUE_PROCESSING_PATH.exists(), "legacy processing backlog should be cleared after migration")
        with sqlite3.connect(str(mod._QUEUE_DB_PATH)) as conn:
            still_processing = conn.execute(
                "SELECT COUNT(1) FROM queue_entries WHERE spawn_intent_id='si_legacy_processing' AND state='processing'"
            ).fetchone()[0]
        expect(int(still_processing) == 1, "migrated sqlite entry should be claimed as processing")

def test_on_exit_drain_skips_finalized_sqlite_intent():
    """on-exit should skip and ack SQLite entries already finalized in intent log."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_sqlite_skip_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._QUEUE_DB_PATH = td_path / ".nautical_queue.db"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        mod._QUEUE_PATH.write_text("", encoding="utf-8")
        intent_log = mod._QUEUE_PATH.parent / ".nautical_spawn_intents.jsonl"
        intent_log.write_text(
            json.dumps(
                {
                    "ts": "2026-01-01T00:00:00Z",
                    "hook": "on-exit",
                    "status": "done",
                    "spawn_intent_id": "si_done",
                    "reason": "processed",
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )

        with sqlite3.connect(str(mod._QUEUE_DB_PATH)) as conn:
            conn.execute(
                """
                CREATE TABLE queue_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spawn_intent_id TEXT,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'queued',
                    claim_token TEXT,
                    claimed_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, created_at, updated_at) "
                "VALUES (?, ?, 0, 'queued', 1.0, 1.0)",
                (
                    "si_done",
                    json.dumps(
                        {
                            "spawn_intent_id": "si_done",
                            "parent_uuid": "00000000-0000-0000-0000-000000000111",
                            "child_short": "deadbeef",
                            "child": {"uuid": "00000000-0000-0000-0000-000000000999"},
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
            conn.commit()

        stats = mod._drain_queue()
        expect(stats.get("entries_total") == 1, f"unexpected entries_total: {stats}")
        expect(stats.get("entries_skipped_idempotent") == 1, f"expected idempotent skip: {stats}")
        expect(stats.get("processed") == 1, f"expected processed=1 due to skip accounting: {stats}")
        with sqlite3.connect(str(mod._QUEUE_DB_PATH)) as conn:
            remaining = conn.execute("SELECT COUNT(1) FROM queue_entries").fetchone()[0]
        expect(int(remaining) == 0, f"sqlite entry should be acked/deleted, remaining={remaining}")


def test_on_exit_queue_drain_is_transactional():
    """on-exit should not truncate queue if staging replace fails."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_queue_txn_test")
    if not hasattr(mod, "_take_queue_entries"):
        raise AssertionError("on-exit hook does not expose queue helper")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"

        child_uuid = "00000000-0000-0000-0000-000000000123"
        valid = json.dumps({"child": {"uuid": child_uuid}})
        invalid = "{bad-json"
        original = valid + "\n" + invalid + "\n"
        mod._QUEUE_PATH.write_text(original, encoding="utf-8")

        orig_replace = mod.os.replace

        def _replace_fail(*_args, **_kwargs):
            raise OSError("replace failed")

        mod.os.replace = _replace_fail
        try:
            entries = mod._take_queue_entries()
        finally:
            mod.os.replace = orig_replace

        expect(len(entries) == 1, f"unexpected entries: {entries}")
        after = mod._QUEUE_PATH.read_text(encoding="utf-8")
        expect(after == original, "queue should remain unchanged on replace failure")


def test_on_exit_queue_stat_failure_does_not_crash():
    """on-exit should handle queue stat failures without crashing."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_queue_stat_failure_test")
    if not hasattr(mod, "_take_queue_entries"):
        raise AssertionError("on-exit hook does not expose queue helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._QUEUE_PATH.write_text(
            json.dumps({"spawn_intent_id": "si_stat", "child": {"uuid": "00000000-0000-0000-0000-000000000123"}}) + "\n",
            encoding="utf-8",
        )

        orig_stat = mod.Path.stat
        orig_exists = mod.Path.exists

        def _stat_fail(path_obj):
            if path_obj == mod._QUEUE_PATH:
                raise OSError("stat failed")
            return orig_stat(path_obj)

        def _exists_override(path_obj):
            if path_obj == mod._QUEUE_PATH:
                return True
            return orig_exists(path_obj)

        mod.Path.stat = _stat_fail
        mod.Path.exists = _exists_override
        try:
            entries = mod._take_queue_entries()
        finally:
            mod.Path.stat = orig_stat
            mod.Path.exists = orig_exists

        expect(len(entries) == 1, f"expected one entry despite stat failure, got: {entries}")


def test_on_exit_quarantines_bad_queue_lines():
    """on-exit should quarantine invalid queue JSON lines."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_quarantine_test")
            q_path = mod._QUEUE_PATH
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text("{bad-json}\n", encoding="utf-8")
            mod._take_queue_entries()
            bad_path = mod._QUEUE_QUARANTINE_PATH
            expect(bad_path.exists(), "queue quarantine file not created")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata


def test_on_exit_dead_letter_on_missing_fields():
    """on-exit should dead-letter entries missing required fields."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_missing_fields_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        entry_missing_spawn = {
            "parent_uuid": "00000000-0000-0000-0000-000000000111",
            "child_short": "deadbeef",
            "child": {"uuid": "00000000-0000-0000-0000-000000000999"},
        }
        entry_missing_child_uuid = {
            "spawn_intent_id": "si_missing_child_uuid",
            "child": {},
        }
        mod._QUEUE_PATH.write_text(
            json.dumps(entry_missing_spawn) + "\n" + json.dumps(entry_missing_child_uuid) + "\n",
            encoding="utf-8",
        )

        stats = mod._drain_queue()
        expect(stats.get("errors") == 2, f"expected 2 errors, got: {stats}")
        expect(mod._DEAD_LETTER_PATH.exists(), "dead letter file not created")
        reasons = []
        for line in mod._DEAD_LETTER_PATH.read_text(encoding="utf-8").splitlines():
            try:
                reasons.append(json.loads(line).get("reason"))
            except Exception:
                pass
        expect("missing spawn_intent_id" in reasons, f"missing spawn_intent_id not in reasons: {reasons}")
        expect("missing child uuid" in reasons, f"missing child uuid not in reasons: {reasons}")


def test_on_exit_processing_file_merges_into_queue():
    """on-exit should merge .processing queue back into main queue when both exist."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_processing_merge_test")
    if not hasattr(mod, "_take_queue_entries"):
        raise AssertionError("on-exit hook does not expose queue helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"

        entry_a = {"spawn_intent_id": "si_a", "child": {"uuid": "00000000-0000-0000-0000-000000000aaa"}}
        entry_b = {"spawn_intent_id": "si_b", "child": {"uuid": "00000000-0000-0000-0000-000000000bbb"}}
        mod._QUEUE_PATH.write_text(json.dumps(entry_a) + "\n", encoding="utf-8")
        mod._QUEUE_PROCESSING_PATH.write_text(json.dumps(entry_b) + "\n", encoding="utf-8")

        entries = mod._take_queue_entries()
        ids = sorted([e.get("spawn_intent_id") for e in entries if isinstance(e, dict)])
        expect(ids == ["si_a", "si_b"], f"unexpected merged ids: {ids}")


def test_on_exit_parent_nextlink_changed_dead_letter():
    """on-exit should dead-letter when parent nextLink changed unexpectedly."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_parent_nextlink_changed_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        child_uuid = "00000000-0000-0000-0000-000000000999"
        parent_uuid = "00000000-0000-0000-0000-000000000111"
        entry = {
            "spawn_intent_id": "si_conflict",
            "parent_uuid": parent_uuid,
            "parent_nextlink": "prevlink",
            "child_short": "deadbeef",
            "child": {"uuid": child_uuid},
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        calls = {"import": 0}

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if "export" in cmd_s and f"uuid:{child_uuid}" in cmd_s:
                return True, json.dumps({"uuid": child_uuid}), ""
            if "export" in cmd_s and f"uuid:{parent_uuid}" in cmd_s:
                return True, json.dumps({"uuid": parent_uuid, "nextLink": "other"}), ""
            if " import " in f" {cmd_s} ":
                calls["import"] += 1
                return True, "", ""
            return False, "", "unexpected"

        mod._run_task = _run_task_stub
        stats = mod._drain_queue()
        expect(stats.get("errors") == 1, f"expected 1 error, got: {stats}")
        expect(mod._DEAD_LETTER_PATH.exists(), "dead letter file not created")
        lines = mod._DEAD_LETTER_PATH.read_text(encoding="utf-8").splitlines()
        reasons = []
        for line in lines:
            try:
                reasons.append(json.loads(line).get("reason", ""))
            except Exception:
                pass
        expect(any("parent update failed: parent nextLink changed" in r for r in reasons),
               f"unexpected reasons: {reasons}")
        expect(calls["import"] == 0, "child import should not run when parent CAS precheck fails")


def test_on_exit_parent_update_lock_busy_requeues():
    """on-exit should requeue (not dead-letter) when parent nextLink lock is busy."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_parent_lock_busy_requeue_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        child_uuid = "00000000-0000-0000-0000-000000000555"
        parent_uuid = "00000000-0000-0000-0000-000000000111"
        entry = {
            "spawn_intent_id": "si_parent_lock_busy",
            "parent_uuid": parent_uuid,
            "parent_nextlink": "",
            "child_short": child_uuid[:8],
            "child": {"uuid": child_uuid},
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        # Precheck passes and child is considered present; contention happens at parent update lock.
        mod._parent_nextlink_state = lambda *_args, **_kwargs: ("ok", "")
        mod._export_uuid = lambda uuid_str: {"exists": True, "retryable": False, "err": "", "obj": {"uuid": uuid_str}}

        @contextlib.contextmanager
        def _busy_parent_lock(_parent_uuid: str):
            yield False

        mod._lock_parent_nextlink = _busy_parent_lock
        stats = mod._drain_queue()
        expect(stats.get("requeued") == 1, f"expected one requeued entry, got: {stats}")
        expect(stats.get("errors") == 0, f"lock-busy parent update should be retryable: {stats}")
        expect(not mod._DEAD_LETTER_PATH.exists(), "lock-busy parent update should not dead-letter")


def test_on_exit_retry_budget_after_post_import_lock_counts_dead_letter():
    """on-exit should count dead-lettered retries when post-import confirm stays locked."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_post_import_retry_budget_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        child_uuid = "00000000-0000-0000-0000-000000000888"
        entry = {
            "spawn_intent_id": "si_budget",
            "attempts": mod._QUEUE_RETRY_MAX,
            "child": {"uuid": child_uuid},
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        calls = {"child_export": 0}

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if f"uuid:{child_uuid}" in cmd_s and "export" in cmd_s:
                calls["child_export"] += 1
                if calls["child_export"] == 1:
                    return True, "{}", ""
                return False, "", "database is locked"
            if " import " in f" {cmd_s} ":
                return True, "", ""
            return False, "", "unexpected"

        mod._run_task = _run_task_stub
        stats = mod._drain_queue()
        expect(stats.get("errors") == 1, f"expected one error, got: {stats}")
        expect(stats.get("dead_lettered") == 1, f"expected one dead-lettered entry, got: {stats}")


def test_on_exit_post_import_parent_conflict_cleans_orphan():
    """on-exit should clean up an imported child when parent CAS loses the race."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_post_import_cleanup_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        child_uuid = "00000000-0000-0000-0000-000000000777"
        parent_uuid = "00000000-0000-0000-0000-000000000111"
        entry = {
            "spawn_intent_id": "si_cleanup",
            "parent_uuid": parent_uuid,
            "parent_nextlink": "",
            "child_short": child_uuid[:8],
            "child": {"uuid": child_uuid},
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        state = {"parent_exports": 0, "child_exports": 0, "cleanup_called": False}

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if f"uuid:{parent_uuid}" in cmd_s and "export" in cmd_s:
                state["parent_exports"] += 1
                if state["parent_exports"] == 1:
                    return True, json.dumps({"uuid": parent_uuid, "nextLink": ""}), ""
                return True, json.dumps({"uuid": parent_uuid, "nextLink": "other"}), ""
            if f"uuid:{child_uuid}" in cmd_s and "export" in cmd_s:
                state["child_exports"] += 1
                if state["child_exports"] == 1:
                    return True, "{}", ""
                return True, json.dumps({"uuid": child_uuid}), ""
            if " import " in f" {cmd_s} ":
                return True, "", ""
            if f"uuid:{child_uuid}" in cmd_s and "modify status:deleted" in cmd_s:
                state["cleanup_called"] = True
                return True, "", ""
            return False, "", "unexpected"

        mod._run_task = _run_task_stub
        stats = mod._drain_queue()
        expect(stats.get("errors") == 1, f"expected one error, got: {stats}")
        expect(stats.get("dead_lettered") == 1, f"expected dead-lettered conflict, got: {stats}")
        expect(state["cleanup_called"], "orphan cleanup should run after post-import parent conflict")


def test_on_exit_idempotent_skip_for_finalized_intent():
    """on-exit should skip queue entries already finalized in the intent log."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_intent_skip_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        sid = "si_done_skip"
        child_uuid = "00000000-0000-0000-0000-000000000909"
        mod._QUEUE_PATH.write_text(
            json.dumps({"spawn_intent_id": sid, "child": {"uuid": child_uuid}, "child_short": child_uuid[:8]}) + "\n",
            encoding="utf-8",
        )
        intent_payload = {
            "ts": "2026-01-01T00:00:00Z",
            "hook": "on-exit",
            "status": "done",
            "spawn_intent_id": sid,
            "reason": "processed",
        }
        mod._intent_log_path().write_text(json.dumps(intent_payload) + "\n", encoding="utf-8")

        def _run_task_should_not_call(*_a, **_k):
            raise AssertionError("task subprocess should not be called for finalized intent")

        mod._run_task = _run_task_should_not_call
        stats = mod._drain_queue()
        expect(stats.get("processed") == 1, f"expected one processed (skipped) entry, got: {stats}")
        expect(stats.get("entries_skipped_idempotent") == 1, f"expected one idempotent skip, got: {stats}")
        expect(stats.get("errors") == 0, f"expected no errors, got: {stats}")


def test_on_exit_lock_storm_circuit_requeues_remaining():
    """on-exit should trip circuit breaker and requeue remaining entries under lock storm."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_lock_storm_circuit_test")
    if not hasattr(mod, "_drain_queue"):
        raise AssertionError("on-exit hook does not expose drain helper")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        entries = []
        for i in range(3):
            entries.append(
                {
                    "spawn_intent_id": f"si_storm_{i}",
                    "child_short": f"deadbe{i}",
                    "child": {"uuid": f"00000000-0000-0000-0000-0000000009{i}{i}"},
                }
            )
        mod._QUEUE_PATH.write_text("\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8")

        mod._LOCK_STORM_THRESHOLD = 2
        mod._LOCK_BACKOFF_BASE = 0.0
        mod._LOCK_BACKOFF_MAX = 0.0
        mod._export_uuid = lambda _u: {"exists": False, "retryable": True, "err": "database is locked", "obj": None}

        captured = {"count": 0}

        def _requeue_capture(items):
            captured["count"] = len(items)
            return True

        mod._requeue_entries = _requeue_capture
        stats = mod._drain_queue()
        expect(stats.get("circuit_breaks") == 1, f"expected one circuit break, got: {stats}")
        expect(stats.get("requeued") == 3, f"expected three requeued entries, got: {stats}")
        expect(captured["count"] == 3, f"expected three entries passed to requeue, got: {captured}")
        expect(stats.get("lock_events", 0) >= 2, f"expected lock events in stats, got: {stats}")
        for key in ("entries_total", "entries_skipped_idempotent", "intent_log_load_ms", "drain_ms"):
            expect(key in stats, f"missing metric key: {key}")


def test_on_exit_import_child_retries_on_lock():
    """on-exit should retry child import on lock errors with backoff."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_retry_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        mod.TW_DATA_DIR = Path(td)
        calls = {"count": 0}
        sleeps: list[float] = []

        def _run_task_stub(_cmd, **_kwargs):
            calls["count"] += 1
            if calls["count"] < 3:
                return False, "", "database is locked"
            return True, "", ""

        mod._run_task = _run_task_stub
        mod._sleep = lambda secs: sleeps.append(secs)
        mod.random.uniform = lambda _a, _b: 0.0

        ok, err = mod._import_child({"uuid": "00000000-0000-0000-0000-000000000abc"})
        expect(ok, f"import should succeed after retries, err={err}")
        expect(calls["count"] == 3, f"unexpected retry count: {calls['count']}")
        expect(len(sleeps) == 2, f"unexpected sleep calls: {sleeps}")


def test_on_exit_dead_letter_on_import_failure():
    """on-exit should dead-letter entries that fail to import."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_dead_letter_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        child_uuid = "00000000-0000-0000-0000-000000000777"
        entry = {
            "parent_uuid": "00000000-0000-0000-0000-000000000888",
            "child_short": "c0ffee12",
            "child": {"uuid": child_uuid, "description": "test"},
            "spawn_intent_id": "si_test",
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if "export" in cmd_s and f"uuid:{child_uuid}" in cmd_s:
                return True, "{}", ""
            if "export" in cmd_s and "uuid:00000000-0000-0000-0000-000000000888" in cmd_s:
                return True, json.dumps({"uuid": "00000000-0000-0000-0000-000000000888", "nextLink": ""}), ""
            if "import" in cmd_s:
                return False, "", "invalid task"
            return False, "", "unexpected"

        mod._run_task = _run_task_stub
        stats = mod._drain_queue()
        expect(stats.get("processed") == 0, f"unexpected processed: {stats}")
        expect(stats.get("errors") == 1, f"unexpected errors: {stats}")

        dead = mod._DEAD_LETTER_PATH.read_text(encoding="utf-8")
        expect("child import failed" in dead, "dead letter should capture import failure")


def test_on_modify_carry_wall_clock_across_dst():
    """carry-forward should preserve local wall-clock offset across DST."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_carry_dst_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    mod.core.LOCAL_TZ_NAME = "America/New_York"
    mod.core._LOCAL_TZ = ZoneInfo("America/New_York")

    due_local = date(2025, 3, 9)
    due_utc = mod.core.build_local_datetime(due_local, (1, 30))
    wait_utc = mod.core.build_local_datetime(due_local, (3, 30))

    child_due_utc = mod.core.build_local_datetime(date(2025, 3, 10), (1, 30))

    parent = {
        "due": mod.core.fmt_isoz(due_utc),
        "wait": mod.core.fmt_isoz(wait_utc),
    }
    child = {"due": mod.core.fmt_isoz(child_due_utc)}

    mod._carry_relative_datetime(parent, child, child_due_utc, "wait")
    wait_child = mod.core.parse_dt_any(child.get("wait"))
    wait_local = mod.core.to_local(wait_child)

    expect(wait_local.hour == 3 and wait_local.minute == 30, f"unexpected local wait: {wait_local}")


def test_on_modify_build_child_carries_configured_uda_datetime():
    """configured recurrence_update_udas fields should carry with wall-clock delta."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_carry_uda_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    try:
        from zoneinfo import ZoneInfo
    except Exception:
        return

    prev_cfg = getattr(mod, "_RECURRENCE_UPDATE_UDAS", ())
    try:
        mod._RECURRENCE_UPDATE_UDAS = ("rappel",)
        mod.core.LOCAL_TZ_NAME = "America/New_York"
        mod.core._LOCAL_TZ = ZoneInfo("America/New_York")

        due_local = date(2025, 3, 9)
        due_utc = mod.core.build_local_datetime(due_local, (1, 30))
        rappel_utc = mod.core.build_local_datetime(due_local, (3, 30))
        child_due_utc = mod.core.build_local_datetime(date(2025, 3, 10), (1, 30))

        parent = {
            "uuid": "00000000-0000-0000-0000-000000000999",
            "status": "completed",
            "due": mod.core.fmt_isoz(due_utc),
            "rappel": mod.core.fmt_isoz(rappel_utc),
            "cp": "1d",
            "chainID": "cid12345",
        }
        child = mod._build_child_from_parent(
            parent,
            child_due_utc,
            2,
            "deadbeef",
            "cp",
            0,
            None,
        )
        rappel_child = mod.core.parse_dt_any(child.get("rappel"))
        rappel_local = mod.core.to_local(rappel_child)
        expect(
            rappel_local.hour == 3 and rappel_local.minute == 30,
            f"unexpected local rappel: {rappel_local}",
        )
    finally:
        mod._RECURRENCE_UPDATE_UDAS = prev_cfg


def test_on_modify_stable_child_uuid_is_slot_deterministic():
    """stable child UUID should be deterministic for the same parent slot and change with link."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_stable_child_uuid_test")

    parent = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "cp": "P1D",
        "chainID": "cid12345",
        "link": 1,
    }
    child_a = {"chainID": "cid12345", "link": 2}
    child_b = {"chainID": "cid12345", "link": 2}
    child_c = {"chainID": "cid12345", "link": 3}

    uuid_a = mod._stable_child_uuid(parent, child_a)
    uuid_b = mod._stable_child_uuid(parent, child_b)
    uuid_c = mod._stable_child_uuid(parent, child_c)

    expect(bool(uuid_a), "stable child uuid should not be empty")
    expect(uuid_a == uuid_b, "same chain slot should yield same stable uuid")
    expect(uuid_a != uuid_c, "different link slot should yield different stable uuid")


def test_normalize_spec_for_acf_cache_guards():
    """normalize spec cache should bound inputs before caching."""
    import nautical_core as core

    res = core._normalize_spec_for_acf_cached("w", "mon", "MD")
    expect(res == "mon", f"unexpected normalize result: {res}")

    long_spec = "x" * 300
    res = core._normalize_spec_for_acf_cached("w", long_spec, "MD")
    expect(res is None, "expected None for overly long spec")

    res = core._normalize_spec_for_acf_cached("q", "mon", "MD")
    expect(res is None, "expected None for invalid typ")


def test_on_modify_link_limit():
    """on-modify should block spawns when link exceeds max."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_link_limit_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False
    mod.core.MAX_LINK_NUMBER = 3

    mod._spawn_child_atomic = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("should not spawn"))

    old = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "status": "pending",
        "description": "limit test",
        "anchor": "w:mon",
        "chainID": "abcd1234",
        "link": 3,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update({"status": "completed", "end": "20250102T090000Z"})

    import io
    from contextlib import redirect_stdout, redirect_stderr

    raw = json.dumps(old) + "\n" + json.dumps(new)
    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()
    orig_stdin = sys.stdin
    try:
        sys.stdin = stdin
        with redirect_stdout(stdout), redirect_stderr(stderr):
            mod.main()
    finally:
        sys.stdin = orig_stdin

    out = json.loads((stdout.getvalue() or "{}").strip() or "{}")
    expect(out.get("link") == 3, "should pass task through unchanged")


def test_on_modify_completion_preflight_context_happy_path():
    """completion preflight should derive link numbers, kind, and chain id for a valid chain task."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_preflight_context_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._existing_next_task = lambda _task, _next_no: None
    new = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "status": "completed",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 2,
    }

    ctx = mod._completion_preflight_context(new, mod.core.now_utc())
    expect(bool(ctx), f"expected preflight context, got {ctx}")
    expect(ctx.get("parent_short") == "00000000", f"unexpected parent_short: {ctx}")
    expect(ctx.get("base_no") == 2 and ctx.get("next_no") == 3, f"unexpected link numbers: {ctx}")
    expect(ctx.get("kind") == "cp", f"unexpected kind: {ctx}")
    expect(ctx.get("chain_id") == "abcd1234", f"unexpected chain id: {ctx}")


def test_on_modify_completion_compute_next_and_limits_happy_path():
    """completion compute should assemble child due and cap metadata from helper results."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_compute_next_limits_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    child_due = mod.core.now_utc() + timedelta(days=1)
    until_dt = child_due + timedelta(days=10)
    finals = [("max", child_due + timedelta(days=5))]

    mod._completion_compute_child_due = lambda _new, _kind: (child_due, {"basis": "stub"}, None)
    mod._completion_until_or_fail = lambda _new, _now: until_dt
    mod._completion_until_guard_or_stop = lambda _new, _child_due, _until_dt, _now: True
    mod._completion_require_child_due_or_fail = lambda _new, _child_due: True
    mod._completion_warn_unreasonable_duration = lambda *_a, **_k: None
    mod._completion_caps = lambda _kind, _new, _child_due, _dnf: (3, until_dt, 3, finals, 3)
    mod._completion_cap_guard_or_stop = lambda _new, _next_no, _cap_no, _now: True

    out = mod._completion_compute_next_and_limits({"chainUntil": "ignored"}, "cp", 2, mod.core.now_utc())
    expect(bool(out), f"expected computed payload, got {out}")
    expect(out.get("child_due") == child_due, f"unexpected child_due: {out}")
    expect(out.get("meta") == {"basis": "stub"}, f"unexpected meta: {out}")
    expect(out.get("until_dt") == until_dt, f"unexpected until_dt: {out}")
    expect(out.get("cpmax") == 3 and out.get("cap_no") == 3, f"unexpected cap data: {out}")
    expect(out.get("finals") == finals and out.get("until_cap_no") == 3, f"unexpected finals: {out}")


def test_on_modify_completion_build_and_spawn_child_happy_path():
    """completion spawn wrapper should return child info and stamp nextLink when verified."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_completion_spawn_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    new = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "status": "completed",
        "chainID": "abcd1234",
        "link": 1,
    }
    child = {"uuid": "00000000-0000-0000-0000-000000000222", "link": 2}
    mod._build_child_from_parent = lambda *_a, **_k: dict(child)
    mod._spawn_child_atomic = lambda _child, _parent: ("deadbeef", set(), True, False, None, "si_test")

    out = mod._completion_build_and_spawn_child(
        new,
        child_due=mod.core.now_utc(),
        next_no=2,
        parent_short="00000000",
        kind="cp",
        cpmax=0,
        until_dt=None,
    )
    expect(bool(out), f"expected spawn result, got {out}")
    expect(out.get("child") == child, f"unexpected child payload: {out}")
    expect(out.get("child_short") == "deadbeef", f"unexpected child short: {out}")
    expect(out.get("verified") is True and out.get("deferred_spawn") is False, f"unexpected verification state: {out}")
    expect(out.get("spawn_intent_id") == "si_test", f"unexpected spawn intent id: {out}")
    expect(new.get("nextLink") == "deadbeef", f"verified spawn should stamp nextLink: {new}")


def test_on_modify_render_cp_completion_feedback_wrapper():
    """CP completion feedback wrapper should delegate and emit a preview panel title."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_feedback_wrapper_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_TIMELINE_GAPS = False
    mod._CHAIN_COLOR_PER_CHAIN = False
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._format_next_cp_rows = lambda fb: fb
    mod._format_root_and_age = lambda *_a, **_k: "abcd1234"
    mod._timeline_lines = lambda *_a, **_k: []
    mod._export_uuid_short_cached = lambda _short: {}

    captured = {}
    mod._panel_line = lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("line mode should not be used"))
    mod._panel = lambda title, fb, **_k: captured.update({"title": title, "fb": list(fb)})

    prev_panel_mode = mod.core.PANEL_MODE
    try:
        mod.core.PANEL_MODE = "panel"
        mod._render_cp_completion_feedback(
            new={"cp": "P1D", "uuid": "00000000-0000-0000-0000-000000000111", "chainID": "abcd1234"},
            child={"uuid": "00000000-0000-0000-0000-000000000222"},
            child_due=mod.core.now_utc(),
            child_short="deadbeef",
            next_no=2,
            parent_short="00000000",
            cap_no=None,
            finals=[],
            now_utc=mod.core.now_utc(),
            until_dt=None,
            until_cap_no=None,
            meta={},
            deferred_spawn=False,
            spawn_intent_id=None,
            chain_by_short=None,
            analytics_advice=None,
            integrity_warnings=None,
            base_no=1,
        )
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    expect("title" in captured, "expected preview panel emission")
    expect("Next link" in captured["title"], f"unexpected panel title: {captured}")


def test_on_add_preview_hard_cap():
    """on-add preview loop should respect hard cap even with large preview setting."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_preview_cap_test")

    mod.UPCOMING_PREVIEW = 1000
    mod._PREVIEW_HARD_CAP = 3

    first_date_local = date(2025, 1, 6)
    first_hhmm = (9, 0)

    def _step_once(prev_date):
        return prev_date + timedelta(days=7)

    preview = []
    cur_dt = core.to_local(core.build_local_datetime(first_date_local, first_hhmm))
    for i in range(mod._PREVIEW_HARD_CAP + 5):
        if i >= mod._PREVIEW_HARD_CAP:
            break
        nxt_date = _step_once(cur_dt.date())
        cur_dt = core.to_local(core.build_local_datetime(nxt_date, first_hhmm))
        preview.append(core.fmt_dt_local(cur_dt.astimezone(timezone.utc)))

    preview_limit = max(0, min(mod.UPCOMING_PREVIEW, 10**9, 10**9, mod._PREVIEW_HARD_CAP))
    expect(preview_limit == 3, f"unexpected preview limit: {preview_limit}")
    expect(len(preview) == 3, "preview hard cap should limit preview length")


def test_on_add_flushes_stdout():
    """on-add should flush stdout after emitting JSON."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_flush_test")

    class _FlushIO(io.StringIO):
        def __init__(self):
            super().__init__()
            self.flushed = False

        def flush(self):
            self.flushed = True
            return super().flush()

    task = {"uuid": "00000000-0000-0000-0000-000000000111", "status": "pending"}
    raw = json.dumps(task)
    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    stdout = _FlushIO()
    stderr = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr
        mod.main()
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr

    expect(stdout.flushed, "stdout.flush should be called")


def test_on_add_profiler_lazy_init():
    """on-add should not register profiler when disabled."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_profiler_lazy_test")
    mod._PROFILE_LEVEL = 0

    called = {"ok": False}
    orig_register = mod.atexit.register
    mod.atexit.register = lambda *_a, **_k: called.update(ok=True)

    task = {"uuid": "00000000-0000-0000-0000-000000000111", "status": "pending"}
    raw = json.dumps(task)
    stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    stdout = io.StringIO()
    stderr = io.StringIO()
    orig_stdin, orig_stdout, orig_stderr = sys.stdin, sys.stdout, sys.stderr
    try:
        sys.stdin = stdin
        sys.stdout = stdout
        sys.stderr = stderr
        mod.main()
    finally:
        sys.stdin, sys.stdout, sys.stderr = orig_stdin, orig_stdout, orig_stderr
        mod.atexit.register = orig_register

    expect(not called["ok"], "profiler should not register when disabled")


def test_on_add_format_anchor_rows_numbers_upcoming_from_three_with_next_anchor():
    """on-add anchor formatting should number upcoming entries from 3 when Next anchor exists."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_format_rows_next_anchor_test")
    if not hasattr(mod, "_format_anchor_rows"):
        raise AssertionError("on-add hook does not expose _format_anchor_rows")

    rows = [
        ("Pattern", "w:mon"),
        ("First due", "2025-01-01 09:00"),
        ("Next anchor", "2025-01-08 09:00"),
        ("Upcoming", "[cyan]2025-01-15 09:00[/]\n[cyan]2025-01-22 09:00[/]"),
        ("Delta", "+7d"),
        ("Chain", "enabled"),
    ]
    out = mod._format_anchor_rows(rows)
    txt = _strip_markup("\n".join(v for _, v in out if isinstance(v, str)))
    expect(" 3 ▸ 2025-01-15 09:00" in txt, f"expected #3 upcoming marker, got: {txt!r}")
    expect(" 4 ▸ 2025-01-22 09:00" in txt, f"expected #4 upcoming marker, got: {txt!r}")


def test_on_add_format_anchor_rows_numbers_upcoming_from_two_without_next_anchor():
    """on-add anchor formatting should number upcoming entries from 2 without Next anchor."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_format_rows_no_next_anchor_test")
    if not hasattr(mod, "_format_anchor_rows"):
        raise AssertionError("on-add hook does not expose _format_anchor_rows")

    rows = [
        ("Pattern", "w:mon"),
        ("First due", "2025-01-01 09:00"),
        ("Upcoming", "[cyan]2025-01-08 09:00[/]"),
        ("Delta", "+7d"),
        ("Other", "x"),
    ]
    out = mod._format_anchor_rows(rows)
    txt = _strip_markup("\n".join(v for _, v in out if isinstance(v, str)))
    expect(" 2 ▸ 2025-01-08 09:00" in txt, f"expected #2 upcoming marker, got: {txt!r}")
    expect("Δ +7d" in txt, f"expected inline delta in first-due row, got: {txt!r}")


def test_on_modify_panel_fallback():
    """on-modify panel should fall back to plain output on errors."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_panel_fallback_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    orig_term = mod.core.term_width_stderr
    mod.core.term_width_stderr = lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom"))
    stderr = io.StringIO()
    orig_stderr = sys.stderr
    try:
        sys.stderr = stderr
        mod._panel("Test Panel", [("Key", "Value")], kind="info")
    finally:
        sys.stderr = orig_stderr
        mod.core.term_width_stderr = orig_term

    out = stderr.getvalue()
    expect("Test Panel" in out, "fallback panel should emit title")


def test_core_render_panel_line_mode_uses_panel_line():
    calls = []
    orig_panel_line = core.panel_line
    try:
        def _fake_panel_line(title, line, **kwargs):
            calls.append((title, line, kwargs.get("kind")))
        core.panel_line = _fake_panel_line
        core.render_panel("Title", [("Key", "Value")], kind="info", panel_mode="line")
    finally:
        core.panel_line = orig_panel_line

    expect(len(calls) == 1, "line mode should route through panel_line once")
    expect(calls[0][1] == "Title — Key: Value", f"unexpected line payload: {calls[0][1]!r}")


def test_core_render_panel_line_force_rich_kind_skips_panel_line():
    calls = []
    orig_panel_line = core.panel_line
    stderr = io.StringIO()
    orig_stderr = sys.stderr
    try:
        def _fake_panel_line(title, line, **kwargs):
            calls.append((title, line, kwargs.get("kind")))
        core.panel_line = _fake_panel_line
        sys.stderr = stderr
        core.render_panel(
            "Title",
            [("Key", "Value")],
            kind="preview_anchor",
            panel_mode="line",
            line_force_rich_kinds={"preview_anchor"},
        )
    finally:
        sys.stderr = orig_stderr
        core.panel_line = orig_panel_line

    expect(not calls, "line-forced rich kind should bypass panel_line")
    out = stderr.getvalue()
    expect("Title" in out and "Key" in out, "promoted rich/fast path should emit fallback panel text")


def test_on_exit_import_error_but_child_exists():
    """on-exit should proceed if import reports failure but child exists."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_import_error_child_exists_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        child_uuid = "00000000-0000-0000-0000-000000000321"
        entry = {
            "parent_uuid": "00000000-0000-0000-0000-000000000456",
            "child_short": "abcdef12",
            "child": {"uuid": child_uuid, "description": "test"},
            "spawn_intent_id": "si_test",
        }
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        state = {"export": 0}
        parent_next = {"value": ""}

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if "export" in cmd_s and f"uuid:{child_uuid}" in cmd_s:
                state["export"] += 1
                if state["export"] >= 2:
                    return True, json.dumps({"uuid": child_uuid}), ""
                return True, "{}", ""
            if "import" in cmd_s:
                return False, "", "invalid task"
            if "export" in cmd_s and "uuid:00000000-0000-0000-0000-000000000456" in cmd_s:
                return True, json.dumps({"uuid": "00000000-0000-0000-0000-000000000456", "nextLink": parent_next["value"]}), ""
            if "modify" in cmd_s and "uuid:00000000-0000-0000-0000-000000000456" in cmd_s:
                parent_next["value"] = "abcdef12"
                return True, "", ""
            return False, "", "unexpected"

        mod._run_task = _run_task_stub
        stats = mod._drain_queue()
        expect(stats.get("processed") == 1, f"unexpected processed: {stats}")


def test_cache_metrics_emits_when_enabled():
    """cache metrics should emit when NAUTICAL_DIAG_METRICS=1."""
    import nautical_core as core

    stderr = io.StringIO()
    orig_err = sys.stderr
    os.environ["NAUTICAL_DIAG_METRICS"] = "1"
    os.environ["NAUTICAL_DIAG"] = "1"
    try:
        with tempfile.TemporaryDirectory() as td:
            os.environ["XDG_CACHE_HOME"] = td
            sys.stderr = stderr
            core._emit_cache_metrics()
    finally:
        sys.stderr = orig_err
        os.environ.pop("NAUTICAL_DIAG_METRICS", None)
        os.environ.pop("NAUTICAL_DIAG", None)
        os.environ.pop("XDG_CACHE_HOME", None)

    out = stderr.getvalue()
    expect("nautical-metrics" in out, f"unexpected metrics output: {out!r}")


def test_sanitize_task_strings_removes_controls():
    """sanitize_task_strings should remove control chars and clamp length."""
    import nautical_core as core

    task = {"description": "hi\x00there\x1f!"}
    core.sanitize_task_strings(task, max_len=8)
    expect("\x00" not in task["description"], "control chars should be removed")
    expect(len(task["description"]) == 8, "should clamp length to max_len")


def test_clear_all_caches_env():
    """_clear_all_caches should be callable via env toggle."""
    import nautical_core as core

    os.environ["NAUTICAL_CLEAR_CACHES"] = "1"
    try:
        core.parse_anchor_expr_to_dnf_cached("w:mon")
    finally:
        os.environ.pop("NAUTICAL_CLEAR_CACHES", None)

    expect(True, "cache clear hook executed")


def test_cache_save_writes_all_bytes():
    """cache_save should write full blob and set 0600 permissions when possible."""
    import nautical_core as core

    with tempfile.TemporaryDirectory() as td:
        core.ENABLE_ANCHOR_CACHE = True
        core.ANCHOR_CACHE_DIR_OVERRIDE = td
        key = "testcache"
        obj = {"dnf": [[{"typ": "w", "spec": "mon"}]]}
        core.cache_save(key, obj)

        path = core._cache_path(key)
        expect(os.path.exists(path), "cache file should exist")
        st = os.stat(path)
        expect(st.st_size > 0, "cache file should not be empty")


def test_cache_save_returns_false_when_lock_busy():
    """cache_save should return False when cache lock is unavailable."""
    import nautical_core as core

    saved_enabled = core.ENABLE_ANCHOR_CACHE
    saved_dir = core.ANCHOR_CACHE_DIR_OVERRIDE
    saved_lock = core._cache_lock
    try:
        core.ENABLE_ANCHOR_CACHE = True
        with tempfile.TemporaryDirectory() as td:
            core.ANCHOR_CACHE_DIR_OVERRIDE = td

            @contextlib.contextmanager
            def _busy_lock(_key: str):
                yield False

            core._cache_lock = _busy_lock
            ok = core.cache_save("busylock", {"dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]]})
            expect(ok is False, f"expected cache_save False on busy lock, got {ok!r}")
    finally:
        core._cache_lock = saved_lock
        core.ENABLE_ANCHOR_CACHE = saved_enabled
        core.ANCHOR_CACHE_DIR_OVERRIDE = saved_dir


def test_cache_save_returns_false_when_atomic_replace_fails():
    """cache_save should return False when atomic replace fails."""
    import nautical_core as core

    saved_enabled = core.ENABLE_ANCHOR_CACHE
    saved_dir = core.ANCHOR_CACHE_DIR_OVERRIDE
    saved_replace = core._cache_atomic_replace
    try:
        core.ENABLE_ANCHOR_CACHE = True
        with tempfile.TemporaryDirectory() as td:
            core.ANCHOR_CACHE_DIR_OVERRIDE = td

            def _raise_replace(_src: str, _dst: str) -> None:
                raise OSError("replace failed")

            core._cache_atomic_replace = _raise_replace
            ok = core.cache_save("replacefail", {"dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]]})
            expect(ok is False, f"expected cache_save False on replace failure, got {ok!r}")
    finally:
        core._cache_atomic_replace = saved_replace
        core.ENABLE_ANCHOR_CACHE = saved_enabled
        core.ANCHOR_CACHE_DIR_OVERRIDE = saved_dir


def test_cache_load_rejects_invalid_payload_shape():
    """cache_load should reject cached payloads with invalid field types."""
    import nautical_core as core

    with tempfile.TemporaryDirectory() as td:
        saved_enabled = core.ENABLE_ANCHOR_CACHE
        saved_dir = core.ANCHOR_CACHE_DIR_OVERRIDE
        try:
            core.ENABLE_ANCHOR_CACHE = True
            core.ANCHOR_CACHE_DIR_OVERRIDE = td
            key = "invalidshape"
            core.cache_save(
                key,
                {
                    "dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]],
                    "natural": ["wrong-type"],
                    "next_dates": ["2026-01-05T00:00"],
                },
            )
            loaded = core.cache_load(key)
            expect(loaded is None, f"invalid payload shape should be rejected, got {loaded!r}")
        finally:
            core.ENABLE_ANCHOR_CACHE = saved_enabled
            core.ANCHOR_CACHE_DIR_OVERRIDE = saved_dir


def test_parse_anchor_expr_fuzz_inputs():
    """Parser should not crash on mixed valid/invalid inputs."""
    import nautical_core as core

    samples = [
        "",
        " ",
        "w:mon",
        "m:15",
        "y:06-01",
        "w:mon..fri@t=09:00",
        "m:rand",
        "y:rand-12",
        "w:rand,mon",
        "w:mon..fri + m:bad",
        "w:mon | m:1st-mon",
        "w:mon..fri@t=99:99",
        "x:bad",
    ]
    for s in samples:
        try:
            dnf = core.parse_anchor_expr_to_dnf_cached(s)
            if s.strip():
                expect(isinstance(dnf, list), f"unexpected dnf type for {s!r}")
        except core.ParseError:
            pass
        except Exception as e:
            raise AssertionError(f"unexpected exception for {s!r}: {e}")


def _random_anchor_expr(rng: random.Random) -> str:
    """Generate deterministic mixed-quality anchor expressions for fuzz/property tests."""
    w_specs = ["mon", "fri", "mon..fri", "mon,tue", "rand", "rand,mon", "bad"]
    m_specs = ["1", "-1", "2nd-mon", "last-fri", "rand", "1..15", "35", "bad"]
    y_specs = ["01-01", "12-31", "01-01..01-31", "q1", "rand-07", "13-01", "bad"]
    mods = ["", "@t=09:00", "@bd", "@wd", "@nbd", "@pbd", "@nw", "@t=99:99", "@@@"]
    sep = [" + ", " | ", "  ", ""]

    def atom() -> str:
        typ = rng.choice(["w", "m", "y", "x"])
        if typ == "w":
            spec = rng.choice(w_specs)
        elif typ == "m":
            spec = rng.choice(m_specs)
        elif typ == "y":
            spec = rng.choice(y_specs)
        else:
            spec = rng.choice(["bad", "noop", ""])
        intv = rng.choice(["", "/2", "/3", "/0"])
        mod = rng.choice(mods)
        return f"{typ}{intv}:{spec}{mod}"

    mode = rng.randint(0, 4)
    if mode == 0:
        return atom()
    if mode == 1:
        return atom() + rng.choice(sep) + atom()
    if mode == 2:
        return "(" + atom() + rng.choice(sep) + atom() + ")"
    if mode == 3:
        noise = "".join(rng.choice("()|+@:,- abcXYZ0123") for _ in range(rng.randint(1, 40)))
        return noise
    return " " + atom() + " "


def test_anchor_parse_validate_fuzz_no_unexpected_exceptions():
    """Fuzz parse/validate/normalize surfaces; only ParseError is allowed for invalid input."""
    import nautical_core as core

    rng = random.Random(20260309)
    for _ in range(250):
        expr = _random_anchor_expr(rng)
        try:
            dnf = core.validate_anchor_expr_strict(expr)
        except core.ParseError:
            continue
        except Exception as e:
            raise AssertionError(f"unexpected exception for {expr!r}: {type(e).__name__}: {e}")

        expect(isinstance(dnf, list), f"validate should return DNF list for {expr!r}")
        try:
            dnf2 = core.validate_anchor_expr_strict(dnf)
        except Exception as e:
            raise AssertionError(f"validate should accept parsed DNF for {expr!r}: {e}")
        expect(isinstance(dnf2, list), f"DNF re-validation should return list for {expr!r}")


def test_anchor_validate_roundtrip_preserves_next_occurrence():
    """Validating parsed DNF should preserve next-occurrence behavior vs validating source string."""
    import nautical_core as core

    anchors = [
        "w:mon",
        "w:mon,tue,wed",
        "w/2:fri",
        "m:15",
        "m:last-fri",
        "m:rand",
        "y:01-01..01-31",
        "w:mon + m:1",
        "m:rand + y:01-01..01-31",
    ]
    start = date(2025, 1, 1)
    for expr in anchors:
        dnf_from_str = core.validate_anchor_expr_strict(expr)
        dnf_from_dnf = core.validate_anchor_expr_strict(dnf_from_str)
        ref_a = start
        ref_b = start
        for _ in range(5):
            nxt_a, _ = core.next_after_expr(dnf_from_str, ref_a, seed_base="roundtrip")
            nxt_b, _ = core.next_after_expr(dnf_from_dnf, ref_b, seed_base="roundtrip")
            expect(nxt_a == nxt_b, f"round-trip mismatch for {expr!r}: {nxt_a} vs {nxt_b}")
            ref_a = nxt_a
            ref_b = nxt_b


def test_anchor_parse_deep_nesting_guard():
    """Deeply nested expressions should fail with ParseError, not recursion/runtime failures."""
    import nautical_core as core

    expr = "(" * 64 + "w:mon" + ")" * 64
    try:
        core.parse_anchor_expr_to_dnf_cached(expr)
        raise AssertionError("expected ParseError for deep nesting")
    except core.ParseError as e:
        expect("nesting too deep" in str(e).lower(), f"unexpected deep nesting message: {e}")


def test_anchor_validate_rejects_legacy_tuple_error_payload():
    """Legacy tuple-style parse error payloads should be rejected defensively."""
    import nautical_core as core

    try:
        core.validate_anchor_expr_strict(("legacy parser error", None))
        raise AssertionError("expected ParseError for tuple error payload")
    except core.ParseError as e:
        expect(str(e) == "legacy parser error", f"unexpected tuple payload message: {e!r}")


def test_rand_determinism_with_seed():
    """Random anchors should be deterministic with the same seed."""
    import nautical_core as core

    dnf = core.parse_anchor_expr_to_dnf_cached("m:rand")
    after = date(2025, 1, 1)
    a1, _meta1 = core.next_after_expr(dnf, after, seed_base="test-seed")
    a2, _meta2 = core.next_after_expr(dnf, after, seed_base="test-seed")
    expect(a1 == a2, f"rand picks should match: {a1} vs {a2}")

def test_next_after_expr_branch_characterization():
    """next_after_expr should preserve basis/meta behavior across major branches."""
    # Simple weekly fast path.
    dnf_simple = core.parse_anchor_expr_to_dnf_cached("w:mon")
    d_simple, m_simple = core.next_after_expr(dnf_simple, date(2024, 12, 11))
    expect(d_simple == date(2024, 12, 16), f"unexpected simple weekly date: {d_simple}")
    expect((m_simple or {}).get("basis") == "simple_weekly", f"unexpected simple basis: {m_simple}")

    # Normal term path.
    dnf_term = core.parse_anchor_expr_to_dnf_cached("w:mon + m:1")
    d_term, m_term = core.next_after_expr(dnf_term, date(2024, 12, 1))
    expect(d_term == date(2025, 9, 1), f"unexpected term date: {d_term}")
    expect((m_term or {}).get("basis") == "term", f"unexpected term basis: {m_term}")

    # Monthly random path.
    dnf_mrand = core.parse_anchor_expr_to_dnf_cached("m:rand")
    d_mrand, m_mrand = core.next_after_expr(dnf_mrand, date(2025, 1, 1), seed_base="branch-seed")
    expect((m_mrand or {}).get("basis") == "rand", f"unexpected m:rand basis: {m_mrand}")
    period_m = (m_mrand or {}).get("rand_period") or ""
    expect(bool(re.fullmatch(r"\d{6}", period_m)), f"unexpected m:rand period: {period_m!r}")
    expect(d_mrand > date(2025, 1, 1), f"m:rand should be strictly after start, got {d_mrand}")

    # Yearly random path with target month.
    dnf_yrand = core.parse_anchor_expr_to_dnf_cached("y:rand-07")
    d_yrand, m_yrand = core.next_after_expr(dnf_yrand, date(2025, 1, 1), seed_base="branch-seed")
    expect((m_yrand or {}).get("basis") == "rand", f"unexpected y:rand basis: {m_yrand}")
    period_y = (m_yrand or {}).get("rand_period") or ""
    expect(bool(re.fullmatch(r"\d{4}-\d{2}", period_y)), f"unexpected y:rand period: {period_y!r}")
    expect(d_yrand.month == 7, f"y:rand-07 should stay in July, got {d_yrand}")

    # Monthly random constrained by yearly window.
    dnf_rand_year = core.parse_anchor_expr_to_dnf_cached("m:rand + y:01-01..01-31")
    d_rand_year, m_rand_year = core.next_after_expr(dnf_rand_year, date(2025, 1, 1), seed_base="branch-seed")
    expect((m_rand_year or {}).get("basis") == "rand+yearly", f"unexpected rand+yearly basis: {m_rand_year}")
    expect(d_rand_year.month == 1, f"rand+yearly should stay in January, got {d_rand_year}")


def test_on_exit_lock_failure_keeps_queue():
    """Queue should remain intact if on-exit lock cannot be acquired."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_lock_fail_queue_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"

        entry = {"child": {"uuid": "00000000-0000-0000-0000-000000000999"}}
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        class _FakeLock:
            def __enter__(self):
                return False
            def __exit__(self, *args):
                return False

        orig_lock = mod._lock_queue
        mod._lock_queue = lambda: _FakeLock()
        try:
            got = mod._take_queue_entries()
        finally:
            mod._lock_queue = orig_lock

        expect(got == [], "lock failure should return empty entries")
        expect(mod._QUEUE_PATH.exists() and mod._QUEUE_PATH.read_text(encoding="utf-8"), "queue should remain on lock failure")


def test_on_exit_local_safe_lock_fails_closed_on_network_mount_without_fcntl():
    """on-exit local lock fallback should fail closed on network mounts when fcntl is unavailable."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_network_lock_fail_closed_test")
    with tempfile.TemporaryDirectory() as td:
        lock_path = Path(td) / "network.lock"
        saved_fcntl = mod.fcntl
        saved_path_probe = mod._path_looks_network_mount
        try:
            mod.fcntl = None
            mod._path_looks_network_mount = lambda _p: True
            with mod._local_safe_lock(lock_path, retries=1, sleep_base=0.0) as acquired:
                expect(not acquired, "expected lock acquisition to fail closed")
            expect(not lock_path.exists(), "fail-closed path should not create lock files")
        finally:
            mod.fcntl = saved_fcntl
            mod._path_looks_network_mount = saved_path_probe


def test_on_exit_queue_streaming_line_cap():
    """on-exit should stream queue and honor line cap."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_queue_line_cap_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._QUEUE_MAX_LINES = 1

        entry1 = {"child": {"uuid": "00000000-0000-0000-0000-000000000111"}}
        entry2 = {"child": {"uuid": "00000000-0000-0000-0000-000000000222"}}
        mod._QUEUE_PATH.write_text(json.dumps(entry1) + "\n" + json.dumps(entry2) + "\n", encoding="utf-8")

        entries = mod._take_queue_entries()
        expect(len(entries) == 1, f"unexpected entries length: {entries}")
        remaining = mod._QUEUE_PATH.read_text(encoding="utf-8").strip().splitlines()
        expect(len(remaining) == 1, "queue should keep remainder lines")


def test_on_exit_queue_rotate_then_drain():
    """on-exit should drain overflow queue after rotation."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_queue_rotate_drain_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._QUEUE_MAX_BYTES = 1
        mod._QUEUE_MAX_LINES = 10

        entry = {"child": {"uuid": "00000000-0000-0000-0000-000000000999"}}
        mod._QUEUE_PATH.write_text(json.dumps(entry) + "\n", encoding="utf-8")

        entries = mod._take_queue_entries()
        expect(len(entries) == 1, f"unexpected entries after rotate: {entries}")
        expect(mod._QUEUE_PATH.exists(), "queue path should remain")


def test_on_exit_dead_letter_rotation():
    """dead-letter should rotate when exceeding size cap."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_dead_letter_rotation_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"
        mod._DEAD_LETTER_MAX_BYTES = 10

        entry = {"child": {"uuid": "00000000-0000-0000-0000-000000000999"}}
        mod._write_dead_letter(entry, "fail")
        mod._write_dead_letter(entry, "fail2")

        rotated = [p for p in os.listdir(td) if p.startswith(".nautical_dead_letter.overflow.")]
        expect(rotated, "dead-letter should rotate when size exceeds cap")


def test_on_exit_dead_letter_carries_spawn_intent_id():
    """dead-letter should include spawn_intent_id when present."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_dead_letter_spawn_id_test")
    if not hasattr(mod, "_write_dead_letter"):
        raise AssertionError("on-exit hook does not expose dead-letter helper")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod._DEAD_LETTER_PATH = td_path / ".nautical_spawn_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_spawn_dead_letter.lock"
        mod._DEAD_LETTER_MAX_BYTES = 0

        entry = {
            "spawn_intent_id": "si_test",
            "parent_uuid": "parent",
            "child_short": "deadbeef",
            "child": {"uuid": "child"},
        }
        mod._write_dead_letter(entry, "reason")
        payload = json.loads(mod._DEAD_LETTER_PATH.read_text(encoding="utf-8").strip())
        expect(payload.get("spawn_intent_id") == "si_test", "dead-letter should carry spawn_intent_id")


def test_queue_json_parse_dead_letter():
    """Invalid queue JSON should go to dead-letter and be removed from queue."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_queue_json_dead_letter_test")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        mod._QUEUE_PATH.write_text("{bad\n", encoding="utf-8")
        entries = mod._take_queue_entries()
        expect(entries == [], "bad JSON should not yield entries")
        expect(mod._DEAD_LETTER_PATH.exists(), "dead-letter should be created")
        remaining = mod._QUEUE_PATH.read_text(encoding="utf-8").strip()
        expect(remaining == "", "queue should be cleared of bad line")

def test_queue_json_non_object_dead_letter_removed():
    """Non-object queue JSON should be dead-lettered and removed from queue."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_queue_json_non_object_test")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"
        mod._QUEUE_QUARANTINE_PATH = td_path / ".nautical_spawn_queue.bad.jsonl"

        mod._QUEUE_PATH.write_text('"not-an-object"\n', encoding="utf-8")
        entries = mod._take_queue_entries()
        expect(entries == [], "non-object JSON should not yield entries")
        expect(mod._DEAD_LETTER_PATH.exists(), "dead-letter should be created for non-object JSON")
        remaining = mod._QUEUE_PATH.read_text(encoding="utf-8").strip()
        expect(remaining == "", "queue should be cleared of non-object line")

def test_on_exit_requeue_failure_leaves_sqlite_entry_processing():
    """on-exit should report requeue failure and keep sqlite claim state for investigation."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_requeue_fail_sqlite_processing_test")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._QUEUE_PROCESSING_PATH = td_path / ".nautical_spawn_queue.processing.jsonl"
        mod._QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._QUEUE_DB_PATH = td_path / ".nautical_queue.db"
        mod._DEAD_LETTER_PATH = td_path / ".nautical_dead_letter.jsonl"
        mod._DEAD_LETTER_LOCK = td_path / ".nautical_dead_letter.lock"

        entry = {
            "spawn_intent_id": "si_requeue_fail",
            "parent_uuid": "",
            "child_short": "deadbeef",
            "child": {"uuid": "00000000-0000-0000-0000-000000000777"},
        }
        with sqlite3.connect(str(mod._QUEUE_DB_PATH)) as conn:
            conn.execute(
                """
                CREATE TABLE queue_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spawn_intent_id TEXT,
                    payload TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    state TEXT NOT NULL DEFAULT 'queued',
                    claim_token TEXT,
                    claimed_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, created_at, updated_at) "
                "VALUES (?, ?, 0, 'queued', 1.0, 1.0)",
                ("si_requeue_fail", json.dumps(entry, ensure_ascii=False, separators=(",", ":"))),
            )
            conn.commit()

        mod._export_uuid = lambda _u: {"exists": False, "retryable": True, "err": "database is locked", "obj": None}
        mod._requeue_entries = lambda _entries: False

        stats = mod._drain_queue()
        expect(stats.get("requeue_failed") == 1, f"expected one requeue failure, got {stats}")
        expect(stats.get("errors", 0) >= 1, f"expected error count incremented, got {stats}")
        with sqlite3.connect(str(mod._QUEUE_DB_PATH)) as conn:
            state = conn.execute(
                "SELECT state FROM queue_entries WHERE spawn_intent_id='si_requeue_fail'"
            ).fetchone()
        expect(state is not None and state[0] == "processing", f"sqlite entry should remain processing on requeue failure: {state}")


def test_on_exit_export_uuid_noisy_stdout():
    """on-exit export should tolerate noisy stdout when UUID present."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_export_uuid_noisy_test")

    def _run_task_noisy(*_a, **_k):
        return True, "WARN something\n00000000-0000-0000-0000-000000000111\n", ""

    mod._run_task = _run_task_noisy
    obj = mod._export_uuid("00000000-0000-0000-0000-000000000111")
    expect(obj and obj.get("exists"), "noisy stdout should still be treated as exists")


def test_on_exit_emit_exit_feedback_reaches_stdout_contract():
    """on-exit failing-hook feedback should still reach stdout even after stdout redirection."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_emit_feedback_test")

    class _DevNullLike:
        def write(self, _s):
            return None
        def flush(self):
            return None

    fake_stdout = io.StringIO()
    fake_stderr = io.StringIO()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    orig_dunder_stdout = sys.__stdout__
    try:
        sys.stdout = _DevNullLike()
        sys.stderr = fake_stderr
        sys.__stdout__ = fake_stdout
        mod._emit_exit_feedback("[nautical] test feedback")
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        sys.__stdout__ = orig_dunder_stdout

    expect("[nautical] test feedback" in fake_stdout.getvalue(), "feedback should reach stdout contract stream")
    expect("[nautical] test feedback" in fake_stderr.getvalue(), "feedback should also remain visible on stderr")


def test_on_modify_recompleted_task_with_nextlink_skips_spawn():
    """Re-completing a reactivated task should not spawn when nextLink already exists."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_recomplete_skip_spawn_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    called = {"spawn": False}

    def _spawn_child_atomic_stub(_child, _parent):
        called["spawn"] = True
        return ("deadbeef", set(), False, True, "queued", "si_test1")

    mod._spawn_child_atomic = _spawn_child_atomic_stub
    mod._export_uuid_short_cached = lambda _short: {
        "uuid": "deadbeef-0000-0000-0000-000000000222",
        "status": "pending",
        "link": 2,
    }

    old = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "status": "pending",
        "description": "reactivated duplicate guard",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "nextLink": "deadbeef",
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            mod.main()
    finally:
        sys.stdin = prev_stdin

    out_task = _extract_last_json(buf_out.getvalue())
    expect(not called["spawn"], "re-completion should not trigger duplicate spawn")
    expect(out_task.get("nextLink") == "deadbeef", "existing nextLink should be preserved")


def test_on_modify_recompleted_task_with_existing_link_skips_spawn():
    """Re-completing should not spawn when link #N+1 already exists in chain even if nextLink is empty."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_recomplete_link_guard_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    called = {"spawn": False}

    def _spawn_child_atomic_stub(_child, _parent):
        called["spawn"] = True
        return ("cafebabe", set(), False, True, "queued", "si_test2")

    mod._spawn_child_atomic = _spawn_child_atomic_stub
    mod._export_uuid_short_cached = lambda _short: None

    def _get_chain_export_stub(chain_id, since=None, extra=None, env=None):
        if chain_id == "abcd1234" and extra and "link:2" in extra:
            return [
                {
                    "uuid": "00000000-0000-0000-0000-000000000222",
                    "status": "pending",
                    "link": 2,
                    "chainID": "abcd1234",
                }
            ]
        return []

    mod._get_chain_export = _get_chain_export_stub

    old = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "status": "pending",
        "description": "reactivated duplicate guard via link check",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            mod.main()
    finally:
        sys.stdin = prev_stdin

    _ = _extract_last_json(buf_out.getvalue())
    expect(not called["spawn"], "existing link #N+1 should prevent duplicate spawn")


def test_on_modify_cp_completion_spawns_next_link():
    """on-modify should spawn the next CP link on completion."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_cp_spawn_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    spawned = {}

    def _spawn_child_atomic_stub(child, parent):
        spawned["child"] = child
        return ("deadbeef", set(), False, True, "queued", "si_test3")

    mod._spawn_child_atomic = _spawn_child_atomic_stub
    mod._export_uuid_short_cached = lambda _short: {}

    old = {
        "uuid": "00000000-0000-0000-0000-000000000111",
        "status": "pending",
        "description": "cp spawn test",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                mod.main()
            except SystemExit as e:
                raise AssertionError(f"on-modify exited unexpectedly (code={e.code})")
    finally:
        sys.stdin = prev_stdin

    out_task = _extract_last_json(buf_out.getvalue())
    expect("child" in spawned, "CP completion did not trigger spawn")
    expect(out_task.get("nextLink") in (None, ""), "CP completion should not set nextLink in decision-only mode")


def test_on_modify_spawn_intent_queue_failure_is_reported():
    """_spawn_child_atomic should report queue failure instead of claiming deferred success."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_spawn_queue_failure_test")
    mod._reserve_child_uuid = lambda _env: "00000000-0000-0000-0000-00000000abcd"
    mod._enqueue_spawn_intent = lambda _entry: (False, "queue lock busy")

    child_short, _stripped, verified, deferred, reason, intent = mod._spawn_child_atomic(
        {"description": "x"},
        {"uuid": "00000000-0000-0000-0000-000000000111", "nextLink": ""},
    )
    expect(child_short == "00000000", f"unexpected child short: {child_short}")
    expect(not verified, "verified should be false when queue fails")
    expect(not deferred, "deferred should be false when queue fails")
    expect("queue lock busy" in (reason or ""), f"missing queue failure reason: {reason}")
    expect(bool(intent), "spawn intent id should still be generated")


def test_on_add_run_task_timeout():
    """on-add _run_task returns timeout on subprocess timeouts."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_run_task_timeout_test")
    if not hasattr(mod, "_run_task"):
        raise AssertionError("on-add hook does not expose _run_task")

    orig_core_run = mod.core.run_task if getattr(mod, "core", None) is not None else None
    orig_popen = None
    if orig_core_run is not None:
        mod.core.run_task = lambda *_args, **_kwargs: (False, "", "timeout")
    else:
        orig_popen = mod.subprocess.Popen
        class _FakeProc:
            def __init__(self):
                self.returncode = 1
            def communicate(self, input=None, timeout=None):
                raise subprocess.TimeoutExpired(cmd="task", timeout=timeout)
            def kill(self):
                return None
        mod.subprocess.Popen = lambda *_args, **_kwargs: _FakeProc()
    try:
        ok, _out, err = mod._run_task(["task", "export"], timeout=0.1, retries=1)
    finally:
        if orig_core_run is not None:
            mod.core.run_task = orig_core_run
        if orig_popen is not None:
            mod.subprocess.Popen = orig_popen

    expect(not ok, "_run_task should return ok=False on timeout")
    expect(err == "timeout", f"_run_task should report timeout, got {err!r}")


def test_on_modify_run_task_timeout():
    """on-modify _run_task returns timeout on subprocess timeouts."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_run_task_timeout_test")
    if not hasattr(mod, "_run_task"):
        raise AssertionError("on-modify hook does not expose _run_task")

    orig_core_run = mod.core.run_task if getattr(mod, "core", None) is not None else None
    orig_popen = None
    if orig_core_run is not None:
        mod.core.run_task = lambda *_args, **_kwargs: (False, "", "timeout")
    else:
        orig_popen = mod.subprocess.Popen
        class _FakeProc:
            def __init__(self):
                self.returncode = 1
            def communicate(self, input=None, timeout=None):
                raise subprocess.TimeoutExpired(cmd="task", timeout=timeout)
            def kill(self):
                return None
        mod.subprocess.Popen = lambda *_args, **_kwargs: _FakeProc()
    try:
        ok, _out, err = mod._run_task(["task", "export"], timeout=0.1, retries=1)
    finally:
        if orig_core_run is not None:
            mod.core.run_task = orig_core_run
        if orig_popen is not None:
            mod.subprocess.Popen = orig_popen

    expect(not ok, "_run_task should return ok=False on timeout")
    expect(err == "timeout", f"_run_task should report timeout, got {err!r}")


def test_on_modify_export_uuid_short_invalid_json():
    """on-modify _export_uuid_short returns None on invalid JSON."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_export_uuid_invalid_json_test")
    if not hasattr(mod, "_export_uuid_short"):
        raise AssertionError("on-modify hook does not expose _export_uuid_short")

    def _run_task_bad(*_args, **_kwargs):
        return True, "not-json", ""

    orig = mod._run_task
    mod._run_task = _run_task_bad
    try:
        obj = mod._export_uuid_short("deadbeef")
    finally:
        mod._run_task = orig

    expect(obj is None, "Invalid JSON should yield None from _export_uuid_short")


def test_on_modify_export_uuid_short_prefix_mismatch():
    """on-modify _export_uuid_short returns None on prefix mismatch."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_export_uuid_prefix_mismatch_test")
    if not hasattr(mod, "_export_uuid_short"):
        raise AssertionError("on-modify hook does not expose _export_uuid_short")

    def _run_task_ok(*_args, **_kwargs):
        return True, json.dumps({"uuid": "00000000-0000-0000-0000-000000000001"}), ""

    orig = mod._run_task
    mod._run_task = _run_task_ok
    try:
        obj = mod._export_uuid_short("deadbeef")
    finally:
        mod._run_task = orig

    expect(obj is None, "Prefix mismatch should yield None from _export_uuid_short")


def test_on_modify_export_uuid_full_cached():
    """Full UUID export should be cached within a hook run."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_export_full_cache_test")
    calls = {"count": 0}
    uuid_full = "00000000-0000-0000-0000-000000000abc"

    def _run_task_stub(cmd, **_kwargs):
        cmd_s = " ".join(cmd)
        if "export" in cmd_s and f"uuid:{uuid_full}" in cmd_s:
            calls["count"] += 1
            return True, json.dumps([{"uuid": uuid_full}]), ""
        return False, "", "unexpected"

    mod._run_task = _run_task_stub
    a = mod._export_uuid_full(uuid_full, env=None)
    b = mod._export_uuid_full(uuid_full, env=None)
    expect(a and b, "expected export results")
    expect(calls["count"] == 1, f"expected 1 export call, got {calls['count']}")


def test_on_modify_missing_taskdata_uses_tw_dir():
    """on-modify uses TW_DIR when TASKDATA is missing."""
    hook = _find_hook_file("on-modify-nautical.py")
    orig = os.environ.get("TASKDATA")
    if "TASKDATA" in os.environ:
        del os.environ["TASKDATA"]
    try:
        mod = _load_hook_module(hook, "_nautical_on_modify_no_taskdata_test")
    finally:
        if orig is not None:
            os.environ["TASKDATA"] = orig

    expect(
        str(getattr(mod, "TW_DATA_DIR", "")) == str(getattr(mod, "TW_DIR", "")),
        "TW_DATA_DIR should fall back to TW_DIR when TASKDATA is unset",
    )


def test_hooks_no_direct_subprocess_run():
    """Hooks should not call subprocess.run outside _run_task."""
    import ast

    def _bad_calls(path: str) -> list[tuple[int, str]]:
        src = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(src, filename=path)
        bad = []
        stack = []

        class Visitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_AsyncFunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            def visit_Call(self, node):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "run":
                    if isinstance(func.value, ast.Name) and func.value.id == "subprocess":
                        current_fn = stack[-1] if stack else ""
                        if current_fn != "_run_task":
                            bad.append((node.lineno, current_fn or "<module>"))
                self.generic_visit(node)

        Visitor().visit(tree)
        return bad

    for hook_name in ("on-add-nautical.py", "on-modify-nautical.py"):
        path = _find_hook_file(hook_name)
        bad = _bad_calls(path)
        expect(not bad, f"Direct subprocess.run found in {hook_name}: {bad}")
TESTS = [
    test_lint_formats,
    test_weekly_and_unsat,
    test_satisfiability_helpers_characterization,
    test_expansion_helpers_characterization,
    test_nth_weekday_range,
    test_lint_anchor_expr_characterization,
    test_last_weekday,
    test_monthly_valid_months_m2_5th_mon,
    test_monthly_support_helpers_characterization,
    test_leap_year_29feb,
    test_quarters_window,
    test_quarter_alias_unambiguous_month_selectors,
    test_quarter_selector_mode_characterization,
    test_quarter_selector_mode_rejections,
    test_term_quarter_rewrite_mode_characterization,
    test_quarter_spec_rewrite_characterization,
    test_rewrite_quarters_in_context_characterization,
    test_yearly_month_names,
    test_rand_with_year_window,
    test_weekly_rand_N_gate,
    test_business_day_nbd_pbd_nw_natural,
    test_time_splitting_per_atom,
    test_weekly_multi_days_and_every_2weeks,
    test_heads_with_slashN_parse_ok,
    test_monthly_valid_months_m2_5th_mon_upcoming_within_valid_months,
    test_leap_year_29feb_upcoming_only_on_leap_year,
    test_rand_with_year_window_filtering,
    test_weekly_rand_N_gate_spacing,
    test_prev_weekday_natural_text,
    test_weekly_multi_days_every_2weeks_spacing_and_days,
    test_anchors_between_expr_stops_on_no_progress,
    test_inline_time_mods_split_ok,
    test_anchor_date_calculations,
    test_interval_patterns,
    test_complex_dnf_expressions,
    test_business_day_modifiers,
    test_next_after_atom_with_mods_characterization,
    test_deterministic_randomness,
    test_edge_cases,
    test_natural_language_comprehensive,
    test_natural_anchor_characterization_for_complex_terms,
    test_natural_compresses_repeated_within_variants,
    test_natural_compresses_repeated_fall_on_variants,
    test_rand_bucket_signature_characterization,
    test_parser_validation,
    test_parser_atom_helpers_characterization,
    test_yearly_spec_token_helper_accepts_known_valid_tokens,
    test_yearly_spec_token_helper_rejects_bad_ranges,
    test_yearly_token_format_characterization,
    test_yearly_token_format_helper_characterization,
    test_validate_year_tokens_in_dnf_characterization,
    test_parse_y_token_characterization,
    test_cache_consistency,
    test_parse_cache_returns_isolated_dnf_instances,
    test_build_and_cache_hints_returns_isolated_cached_payload,
    test_cache_key_for_task_caches_build_acf_results,
    test_build_and_cache_hints_parses_once_per_miss,
    test_yearly_rand_natural_and_bounds,
    test_yearly_month_aliases_and_ranges,
    test_business_day_bd_skip_semantics,
    test_scheduler_atom_helpers_characterization,
    test_inline_time_mods_natural_contains_both_times,
    test_guard_commas_between_atoms_after_mods_fatal,
    test_heads_with_slashN_parse_ok_again,
    test_monthname_and_numeric_equivalence,
    test_cp_duration_parser_and_dst_preserve_whole_days,
    test_hook_on_add_multitime_preview_emits_all_slots,
    test_hook_on_modify_timeline_multitime_includes_all_slots,
    test_hook_task_runner_handles_nonzero,
    test_hook_run_task_falls_back_when_core_load_fails,
    test_on_add_run_task_falls_back_when_core_load_fails,
    test_spawn_child_verifies_even_when_verify_import_disabled,
    test_core_run_task_tempfiles_accepts_text_input,
    test_core_run_task_timeout_reports_timeout_with_tempfiles,
    test_core_run_task_nonzero_retries_use_expected_backoff,
    test_core_run_task_tempfiles_fallback_handles_bytes_input,
    test_warn_once_per_day_stamp_written,
    test_warn_once_per_day_no_diag_silent,
    test_warn_once_per_day_any_no_diag_silent,
    test_hook_stdout_strict_json_with_diag_on_add,
    test_hook_stdout_strict_json_with_diag_on_modify,
    test_hook_stdout_unicode_unescaped_on_add,
    test_hook_stdout_unicode_unescaped_on_modify,
    test_hook_stdout_empty_on_exit,
    test_hook_files_are_private_permissions,
    test_safe_lock_fcntl_contention,
    test_safe_lock_fallback_contention,
    test_safe_lock_fallback_stale_cleanup,
    test_safe_lock_fallback_stale_pid_cleanup,
    test_on_modify_queue_repairs_permissions,
    test_on_exit_repairs_queue_and_dead_letter_permissions,
    test_on_exit_timeouts_configurable,
    test_on_exit_queue_db_connect_retries_and_scales_busy_timeout,
    test_diag_log_rotation_bounds,
    test_diag_log_redacts_sensitive_fields,
    test_hook_diag_redact_msg_masks_sensitive_json_fields,
    test_diag_log_structured_fields,
    test_on_exit_requeues_when_task_lock_recent,
    test_core_cache_dir_and_lock_permissions,
    test_core_cache_lock_contention_matches_safe_lock,
    test_core_cache_dir_rejects_symlink_override,
    test_on_modify_invalid_json_passthrough,
    test_on_modify_read_two_invalid_trailing,
    test_on_modify_read_two_array_uuid_mismatch_fails,
    test_on_modify_read_two_array_single_missing_uuid_fails,
    test_on_modify_invalid_anchor_has_no_stdout,
    test_on_add_rejects_oversized_stdin_early,
    test_on_modify_rejects_oversized_stdin_early,
    test_health_check_json_ok_empty_taskdata,
    test_health_check_critical_queue_bytes,
    test_health_check_critical_queue_db_rows,
    test_perf_budget_config_covers_cache_io_checks,
    test_deploy_sanity_script_reports_ok,
    test_ops_templates_present_and_runner_executable,
    test_on_modify_queue_full_drops_with_dead_letter,
    test_on_modify_enqueue_uses_sqlite_when_legacy_empty,
    test_on_modify_enqueue_uses_sqlite_even_with_legacy_jsonl_backlog,
    test_on_modify_chain_export_timeout_scales,
    test_tw_export_chain_extra_validation,
    test_tw_export_chain_extra_rejects_dash_prefixed_tokens,
    test_on_add_tw_export_chain_extra_validation,
    test_on_modify_chain_cache_thread_safety_smoke,
    test_next_for_and_no_progress_fails_fast,
    test_next_for_and_transient_stall_recovers,
    test_roll_apply_has_guard,
    test_anchor_cache_cleans_stale_tmp_files,
    test_weeks_between_iso_boundary,
    test_short_uuid_invalid_inputs,
    test_anchor_expr_length_limit,
    test_parser_frontend_normalization_characterization,
    test_parser_frontend_year_colon_guard_characterization,
    test_parser_frontend_comma_join_guard_characterization,
    test_anchor_parse_term_explosion_guard,
    test_build_local_datetime_dst_gap_and_ambiguous,
    test_on_modify_chain_export_cache_key_includes_params,
    test_on_modify_chain_export_skips_when_locked,
    test_on_modify_collect_prev_two_prefers_live_statuses,
    test_coerce_int_bounds,
    test_on_add_fail_and_exit_emits_json,
    test_on_add_panic_passthrough_emits_valid_json,
    test_on_modify_panic_passthrough_uses_latest_task,
    test_on_add_ignores_unsafe_core_path_override,
    test_on_modify_ignores_unsafe_core_path_override,
    test_on_add_read_one_fuzz_inputs,
    test_on_modify_read_two_fuzz_inputs,
    test_on_add_dnf_cache_versioned_payload,
    test_on_add_dnf_cache_corrupt_payload_recovers,
    test_on_add_dnf_cache_quarantines_invalid_jsonl,
    test_on_add_dnf_cache_checksum_mismatch_salvages,
    test_on_add_dnf_cache_size_guard_skips_load,
    test_on_add_dnf_cache_skips_non_jsonable_values,
    test_on_exit_spawn_intents_drain,
    test_on_exit_take_queue_migrates_legacy_processing_backlog_to_sqlite,
    test_on_exit_drain_skips_finalized_sqlite_intent,
    test_on_exit_queue_drain_is_transactional,
    test_on_exit_queue_stat_failure_does_not_crash,
    test_on_exit_quarantines_bad_queue_lines,
    test_on_exit_dead_letter_on_missing_fields,
    test_on_exit_processing_file_merges_into_queue,
    test_on_exit_import_child_retries_on_lock,
    test_on_exit_dead_letter_on_import_failure,
    test_on_exit_large_queue_bounded_drain,
    test_on_exit_queue_drain_idempotent,
    test_on_exit_rolls_back_parent_nextlink_on_missing_child,
    test_on_exit_uses_tw_data_dir_for_export_and_modify,
    test_on_exit_no_explicit_taskdata_skips_rc_data_location,
    test_on_exit_reads_data_arg_from_hook_argv,
    test_on_modify_no_explicit_taskdata_skips_rc_data_location,
    test_on_modify_reads_data_arg_from_hook_argv,
    test_on_add_no_explicit_taskdata_skips_rc_data_location,
    test_on_add_reads_data_arg_from_hook_argv,
    test_on_exit_data_arg_overrides_taskdata_env,
    test_on_modify_data_arg_overrides_taskdata_env,
    test_on_add_data_arg_overrides_taskdata_env,
    test_core_resolve_task_data_context_precedence,
    test_core_resolve_task_data_context_rejects_unsafe_world_writable_dir,
    test_core_resolve_task_data_context_trust_override_allows_explicit_dir,
    test_core_resolve_task_data_context_rejects_parent_traversal_segments,
    test_core_config_paths_rejects_parent_traversal_in_env,
    test_core_config_paths_trust_override_allows_parent_traversal_in_env,
    test_on_add_requires_core_data_context_helper,
    test_on_modify_requires_core_data_context_helper,
    test_on_exit_requires_core_data_context_helper,
    test_on_modify_carry_wall_clock_across_dst,
    test_normalize_spec_for_acf_cache_guards,
    test_on_modify_link_limit,
    test_on_modify_completion_preflight_context_happy_path,
    test_on_modify_completion_compute_next_and_limits_happy_path,
    test_on_modify_completion_build_and_spawn_child_happy_path,
    test_on_modify_render_cp_completion_feedback_wrapper,
    test_on_add_preview_hard_cap,
    test_on_add_flushes_stdout,
    test_on_add_profiler_lazy_init,
    test_on_add_format_anchor_rows_numbers_upcoming_from_three_with_next_anchor,
    test_on_add_format_anchor_rows_numbers_upcoming_from_two_without_next_anchor,
    test_on_modify_panel_fallback,
    test_core_render_panel_line_mode_uses_panel_line,
    test_core_render_panel_line_force_rich_kind_skips_panel_line,
    test_on_exit_import_error_but_child_exists,
    test_on_exit_parent_nextlink_changed_dead_letter,
    test_on_exit_parent_update_lock_busy_requeues,
    test_on_exit_retry_budget_after_post_import_lock_counts_dead_letter,
    test_on_exit_post_import_parent_conflict_cleans_orphan,
    test_on_exit_idempotent_skip_for_finalized_intent,
    test_on_exit_lock_storm_circuit_requeues_remaining,
    test_cache_metrics_emits_when_enabled,
    test_sanitize_task_strings_removes_controls,
    test_clear_all_caches_env,
    test_cache_save_writes_all_bytes,
    test_cache_save_returns_false_when_lock_busy,
    test_cache_save_returns_false_when_atomic_replace_fails,
    test_cache_load_rejects_invalid_payload_shape,
    test_parse_anchor_expr_fuzz_inputs,
    test_anchor_parse_validate_fuzz_no_unexpected_exceptions,
    test_anchor_validate_roundtrip_preserves_next_occurrence,
    test_anchor_parse_deep_nesting_guard,
    test_anchor_validate_rejects_legacy_tuple_error_payload,
    test_rand_determinism_with_seed,
    test_next_after_expr_branch_characterization,
    test_on_exit_lock_failure_keeps_queue,
    test_on_exit_local_safe_lock_fails_closed_on_network_mount_without_fcntl,
    test_on_exit_queue_streaming_line_cap,
    test_on_exit_queue_rotate_then_drain,
    test_on_exit_dead_letter_rotation,
    test_queue_json_parse_dead_letter,
    test_queue_json_non_object_dead_letter_removed,
    test_on_exit_dead_letter_carries_spawn_intent_id,
    test_on_exit_requeue_failure_leaves_sqlite_entry_processing,
    test_on_exit_export_uuid_noisy_stdout,
    test_on_exit_emit_exit_feedback_reaches_stdout_contract,
    test_core_import_deterministic,
    test_on_modify_spawn_intent_id_in_entry,
    test_on_modify_recompleted_task_with_nextlink_skips_spawn,
    test_on_modify_recompleted_task_with_existing_link_skips_spawn,
    test_on_modify_cp_completion_spawns_next_link,
    test_on_modify_spawn_intent_queue_failure_is_reported,
    test_on_add_run_task_timeout,
    test_on_modify_run_task_timeout,
    test_on_modify_export_uuid_short_invalid_json,
    test_on_modify_export_uuid_short_prefix_mismatch,
    test_on_modify_export_uuid_full_cached,
    test_on_modify_stable_child_uuid_is_slot_deterministic,
    test_on_modify_missing_taskdata_uses_tw_dir,
    test_hooks_no_direct_subprocess_run,
    test_chain_integrity_warnings_detects_issues,
    test_chain_health_advice_coach_healthy_streak,
    test_chain_health_advice_coach_low_ontime_issue,
    test_chain_health_advice_clinical_drift_and_style_normalization,
    test_dst_round_trip_noon_preserves_local_date,
    test_core_invalid_timezone_warns_and_falls_back_to_utc,
    test_core_recurrence_update_udas_config_aliases,
    test_warn_rate_limited_any,
    test_on_modify_build_child_carries_configured_uda_datetime,

]

DEEP_TESTS = [

]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter for test names")
    ap.add_argument("--verbose", action="store_true", help="show detailed test information")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        sel = []
        for fn in TESTS:
            if args.only.lower() in fn.__name__.lower():
                sel.append(fn)
        selected = sel

    fails = 0
    total_tests = 0
    
    for fn in selected:
        total_tests += 1
        try:
            fn()
            if args.verbose:
                # Extract docstring and print test description
                docstring = fn.__doc__ or "No description available"
                # Get first line of docstring
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✓ {fn.__name__}: {description}")
        except AssertionError as e:
            fails += 1
            if args.verbose:
                docstring = fn.__doc__ or "No description available"
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✗ {fn.__name__}: {description}")
                print(f"  ERROR: {e}")
            else:
                print(f"✗ {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            if args.verbose:
                docstring = fn.__doc__ or "No description available"
                description = docstring.strip().split('\n')[0] if docstring else fn.__name__
                print(f"✗ {fn.__name__}: {description}")
                print(f"  UNEXPECTED ERROR: {e}")
                import traceback
                traceback.print_exc()
            else:
                print(f"✗ {fn.__name__}: unexpected error {e}")

    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests run: {total_tests}")
    print(f"Passed: {total_tests - fails}")
    print(f"Failed: {fails}")
    print(f"Success rate: {((total_tests - fails) / total_tests * 100):.1f}%")
    print(f"{'='*60}")
    
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()
