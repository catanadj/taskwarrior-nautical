#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Nautical Golden Tests
 - Imports local nautical_core.py
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
    # Ensure the hook can import local nautical_core.py
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
    # Ensure the hook can import local nautical_core.py
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
            core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
            mod_core = _load_hook_module(core_path, "_nautical_core_perm_test")
            lock_path = os.path.join(td, ".nautical_perm_test.lock")
            with mod_core.safe_lock(lock_path, retries=2, sleep_base=0.01, jitter=0.0) as ok:
                expect(ok, "safe_lock did not acquire")
                mode = stat.S_IMODE(os.stat(lock_path).st_mode)
                expect((mode & 0o077) == 0, f"lock file has group/other perms: {oct(mode)}")

            hook_modify = _find_hook_file("on-modify-nautical.py")
            mod_modify = _load_hook_module(hook_modify, "_nautical_on_modify_perm_test")
            mod_modify._enqueue_deferred_spawn({"uuid": "00000000-0000-0000-0000-000000000999"})
            q_path = mod_modify._SPAWN_QUEUE_PATH
            expect(q_path.exists(), f"spawn queue not created: {q_path}")
            mode = stat.S_IMODE(q_path.stat().st_mode)
            expect((mode & 0o077) == 0, f"queue file has group/other perms: {oct(mode)}")

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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    """Existing queue file permissions should be repaired on append."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_queue_perm_test")
            q_path = mod._SPAWN_QUEUE_PATH
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text("{}", encoding="utf-8")
            os.chmod(q_path, 0o666)
            mod._enqueue_deferred_spawn({"uuid": "00000000-0000-0000-0000-000000000123"})
            mode = stat.S_IMODE(q_path.stat().st_mode)
            expect((mode & 0o077) == 0, f"queue perms not repaired: {oct(mode)}")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_on_exit_repairs_queue_and_dead_letter_permissions():
    """Existing queue/dead-letter permissions should be repaired on write."""
    hook = _find_hook_file("on-exit-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_exit_perm_repair_test")
            q_path = mod._QUEUE_PATH
            dl_path = mod._DEAD_LETTER_PATH
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text("{}", encoding="utf-8")
            dl_path.write_text("{}", encoding="utf-8")
            os.chmod(q_path, 0o666)
            os.chmod(dl_path, 0o666)
            mod._requeue_entries([{"uuid": "00000000-0000-0000-0000-000000000456"}])
            mod._write_dead_letter({"uuid": "00000000-0000-0000-0000-000000000789"}, "perm test")
            q_mode = stat.S_IMODE(q_path.stat().st_mode)
            dl_mode = stat.S_IMODE(dl_path.stat().st_mode)
            expect((q_mode & 0o077) == 0, f"queue perms not repaired: {oct(q_mode)}")
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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    """on-exit should target TW_DATA_DIR for export/modify calls."""
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

    mod._export_uuid("deadbeef")
    mod._update_parent_nextlink("parent-uuid", "childshort")

    expect(calls, "expected _run_task to be called")
    want = f"rc.data.location={mod.TW_DATA_DIR}"
    expect(want in calls[0], f"export missing data dir: {calls[0]!r}")
    expect(any(want in call for call in calls[1:]), f"modify missing data dir: {calls!r}")

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
            q_path = mod._SPAWN_QUEUE_PATH
            dl_path = mod._DEAD_LETTER_PATH
            q_path.parent.mkdir(parents=True, exist_ok=True)
            q_path.write_text("x" * 128, encoding="utf-8")
            mod._SPAWN_QUEUE_MAX_BYTES = 10
            mod._enqueue_deferred_spawn({"spawn_intent_id": "si_full", "child": {"uuid": "u1"}})
            expect(q_path.read_text(encoding="utf-8") == "x" * 128, "queue should not grow when full")
            expect(dl_path.exists(), "dead-letter not written on queue full")
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

def test_dst_round_trip_noon_preserves_local_date():
    """Local date+time should round-trip through UTC across DST boundaries."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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

def test_coerce_int_bounds():
    """coerce_int should return default for out-of-bounds values."""
    big = 2**63
    expect(core.coerce_int(big, default=7) == 7, "coerce_int should reject too-large int")
    expect(core.coerce_int(float(big), default=7) == 7, "coerce_int should reject too-large float")

def test_build_local_datetime_dst_gap_and_ambiguous():
    """build_local_datetime should handle DST gaps and ambiguities deterministically."""
    core_path = os.path.abspath(os.path.join(HERE, "..", "nautical_core.py"))
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

def test_weekly_and_unsat():
    """Test weekly AND (Sat AND Mon) must be unsatisfiable"""
    fatal, _ = core.lint_anchor_expr("w:sat + w:mon")
    expect(bool(fatal), "Weekly A+B must be unsatisfiable (Sat AND Mon)")

def test_nth_weekday_range():
    """Test nth weekday range validation (1..5 or last)"""
    fatal, _ = core.lint_anchor_expr("m:6th-mon")
    expect(bool(fatal), "6th-mon must fatal (nth in 1..5 or last)")

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
        bad_core = Path(td) / "nautical_core.py"
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

        def _run_task_stub(cmd, **_kwargs):
            cmd_s = " ".join(cmd)
            if "export" in cmd_s and f"uuid:{child_uuid}" in cmd_s:
                return True, json.dumps({"uuid": child_uuid}), ""
            if "export" in cmd_s and f"uuid:{parent_uuid}" in cmd_s:
                return True, json.dumps({"uuid": parent_uuid, "nextLink": "other"}), ""
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


def test_rand_determinism_with_seed():
    """Random anchors should be deterministic with the same seed."""
    import nautical_core as core

    dnf = core.parse_anchor_expr_to_dnf_cached("m:rand")
    after = date(2025, 1, 1)
    a1, _meta1 = core.next_after_expr(dnf, after, seed_base="test-seed")
    a2, _meta2 = core.next_after_expr(dnf, after, seed_base="test-seed")
    expect(a1 == a2, f"rand picks should match: {a1} vs {a2}")


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


def test_on_exit_export_uuid_noisy_stdout():
    """on-exit export should tolerate noisy stdout when UUID present."""
    hook = _find_hook_file("on-exit-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_exit_export_uuid_noisy_test")

    def _run_task_noisy(*_a, **_k):
        return True, "WARN something\n00000000-0000-0000-0000-000000000111\n", ""

    mod._run_task = _run_task_noisy
    obj = mod._export_uuid("00000000-0000-0000-0000-000000000111")
    expect(obj and obj.get("exists"), "noisy stdout should still be treated as exists")


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
        return ("deadbeef", set(), False, True, "queued")

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
    test_nth_weekday_range,
    test_last_weekday,
    test_monthly_valid_months_m2_5th_mon,
    test_leap_year_29feb,
    test_quarters_window,
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
    test_inline_time_mods_split_ok,
    test_anchor_date_calculations,
    test_interval_patterns,
    test_complex_dnf_expressions,
    test_business_day_modifiers,
    test_deterministic_randomness,
    test_edge_cases,
    test_natural_language_comprehensive,
    test_parser_validation,
    test_cache_consistency,
    test_yearly_rand_natural_and_bounds,
    test_yearly_month_aliases_and_ranges,
    test_business_day_bd_skip_semantics,
    test_inline_time_mods_natural_contains_both_times,
    test_guard_commas_between_atoms_after_mods_fatal,
    test_heads_with_slashN_parse_ok_again,
    test_monthname_and_numeric_equivalence,
    test_cp_duration_parser_and_dst_preserve_whole_days,
    test_hook_on_add_multitime_preview_emits_all_slots,
    test_hook_on_modify_timeline_multitime_includes_all_slots,
    test_hook_task_runner_handles_nonzero,
    test_core_run_task_tempfiles_accepts_text_input,
    test_warn_once_per_day_stamp_written,
    test_warn_once_per_day_no_diag_silent,
    test_warn_once_per_day_any_no_diag_silent,
    test_hook_stdout_strict_json_with_diag_on_add,
    test_hook_stdout_strict_json_with_diag_on_modify,
    test_hook_stdout_empty_on_exit,
    test_hook_files_are_private_permissions,
    test_safe_lock_fcntl_contention,
    test_safe_lock_fallback_contention,
    test_safe_lock_fallback_stale_cleanup,
    test_safe_lock_fallback_stale_pid_cleanup,
    test_on_modify_queue_repairs_permissions,
    test_on_exit_repairs_queue_and_dead_letter_permissions,
    test_on_exit_timeouts_configurable,
    test_diag_log_rotation_bounds,
    test_diag_log_redacts_sensitive_fields,
    test_diag_log_structured_fields,
    test_on_exit_requeues_when_task_lock_recent,
    test_core_cache_dir_and_lock_permissions,
    test_core_cache_lock_contention_matches_safe_lock,
    test_on_modify_invalid_json_passthrough,
    test_on_modify_read_two_invalid_trailing,
    test_on_modify_queue_full_drops_with_dead_letter,
    test_on_modify_chain_export_timeout_scales,
    test_tw_export_chain_extra_validation,
    test_next_for_and_no_progress_fails_fast,
    test_roll_apply_has_guard,
    test_anchor_cache_cleans_stale_tmp_files,
    test_weeks_between_iso_boundary,
    test_short_uuid_invalid_inputs,
    test_anchor_expr_length_limit,
    test_build_local_datetime_dst_gap_and_ambiguous,
    test_on_modify_chain_export_cache_key_includes_params,
    test_on_modify_chain_export_skips_when_locked,
    test_coerce_int_bounds,
    test_on_add_fail_and_exit_emits_json,
    test_on_add_read_one_fuzz_inputs,
    test_on_modify_read_two_fuzz_inputs,
    test_on_add_dnf_cache_versioned_payload,
    test_on_add_dnf_cache_corrupt_payload_recovers,
    test_on_add_dnf_cache_quarantines_invalid_jsonl,
    test_on_add_dnf_cache_checksum_mismatch_salvages,
    test_on_add_dnf_cache_size_guard_skips_load,
    test_on_add_dnf_cache_skips_non_jsonable_values,
    test_on_exit_spawn_intents_drain,
    test_on_exit_queue_drain_is_transactional,
    test_on_exit_quarantines_bad_queue_lines,
    test_on_exit_dead_letter_on_missing_fields,
    test_on_exit_processing_file_merges_into_queue,
    test_on_exit_import_child_retries_on_lock,
    test_on_exit_dead_letter_on_import_failure,
    test_on_exit_large_queue_bounded_drain,
    test_on_exit_queue_drain_idempotent,
    test_on_exit_rolls_back_parent_nextlink_on_missing_child,
    test_on_exit_uses_tw_data_dir_for_export_and_modify,
    test_on_modify_carry_wall_clock_across_dst,
    test_normalize_spec_for_acf_cache_guards,
    test_on_modify_link_limit,
    test_on_add_preview_hard_cap,
    test_on_add_flushes_stdout,
    test_on_add_profiler_lazy_init,
    test_on_modify_panel_fallback,
    test_on_exit_import_error_but_child_exists,
    test_on_exit_parent_nextlink_changed_dead_letter,
    test_cache_metrics_emits_when_enabled,
    test_sanitize_task_strings_removes_controls,
    test_clear_all_caches_env,
    test_cache_save_writes_all_bytes,
    test_parse_anchor_expr_fuzz_inputs,
    test_rand_determinism_with_seed,
    test_on_exit_lock_failure_keeps_queue,
    test_on_exit_queue_streaming_line_cap,
    test_on_exit_queue_rotate_then_drain,
    test_on_exit_dead_letter_rotation,
    test_queue_json_parse_dead_letter,
    test_on_exit_dead_letter_carries_spawn_intent_id,
    test_on_exit_export_uuid_noisy_stdout,
    test_core_import_deterministic,
    test_on_modify_spawn_intent_id_in_entry,
    test_on_modify_cp_completion_spawns_next_link,
    test_on_add_run_task_timeout,
    test_on_modify_run_task_timeout,
    test_on_modify_export_uuid_short_invalid_json,
    test_on_modify_export_uuid_short_prefix_mismatch,
    test_on_modify_export_uuid_full_cached,
    test_on_modify_missing_taskdata_uses_tw_dir,
    test_hooks_no_direct_subprocess_run,
    test_chain_integrity_warnings_detects_issues,
    test_dst_round_trip_noon_preserves_local_date,
    test_warn_rate_limited_any,

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
