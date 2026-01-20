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
import sys, os, re, json, io, contextlib
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

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

def test_on_add_fail_and_exit_emits_json():
    """_fail_and_exit should emit JSON to stdout for hook error paths."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_fail_test")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            mod._fail_and_exit("Invalid anchor", "anchor syntax error: bad")
        except SystemExit as e:
            expect(e.code == 1, f"unexpected exit code: {e.code}")
        else:
            raise AssertionError("_fail_and_exit did not exit")
    out = buf.getvalue().strip()
    obj = json.loads(out)
    expect(obj.get("error") == "Invalid anchor", f"unexpected error field: {obj!r}")
    expect("anchor syntax error" in (obj.get("message") or ""), f"unexpected message field: {obj!r}")

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

def test_on_modify_read_two_fuzz_inputs():
    """on-modify input parsing should be tolerant and never crash."""
    hook = _find_hook_file("on-modify-nautical.py")
    cases = [
        ("", "empty"),
        ("{not-json}", "passthrough"),
        (json.dumps({"status": "pending"}), "json"),
        (json.dumps({"status": "pending"}) + "\n" + json.dumps({"status": "pending"}), "json"),
        ("  \n" + json.dumps({"status": "pending"}) + "\n", "json"),
    ]
    for raw, mode in cases:
        p = _run_hook_script_raw(hook, raw)
        expect(p.returncode == 0, f"on-modify returned {p.returncode} for case {mode}")
        if mode == "empty":
            expect((p.stdout or "") == "", "empty input should return empty stdout")
        elif mode == "passthrough":
            expect((p.stdout or "").strip() == raw.strip(), "invalid input should pass through raw")
        else:
            _assert_stdout_json_only(p.stdout)

def test_spawn_queue_recovers_partial_payload():
    """Deferred spawn queue should recover valid JSON objects from mixed/partial payloads."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_queue_test")
            qpath = mod._SPAWN_QUEUE_PATH
            payload = '{"uuid":"aaaaaaaa"}\\n{bad\\n{"uuid":"bbbbbbbb"}'
            qpath.write_text(payload, encoding="utf-8")
            out = mod._take_deferred_spawn_payload()
            objs = [json.loads(ln) for ln in (out or "").splitlines() if ln.strip()]
            uuids = {o.get("uuid") for o in objs}
            expect("aaaaaaaa" in uuids and "bbbbbbbb" in uuids, f"unexpected queue uuids: {uuids}")
            expect(qpath.read_text(encoding="utf-8") == "", "queue should be truncated after recovery")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

def test_spawn_queue_invalid_payload_preserved():
    """Deferred spawn queue should avoid truncation if nothing parses."""
    hook = _find_hook_file("on-modify-nautical.py")
    with tempfile.TemporaryDirectory() as td:
        prev_taskdata = os.environ.get("TASKDATA")
        os.environ["TASKDATA"] = td
        try:
            mod = _load_hook_module(hook, "_nautical_on_modify_queue_test2")
            qpath = mod._SPAWN_QUEUE_PATH
            payload = "not-json\n"
            qpath.write_text(payload, encoding="utf-8")
            out = mod._take_deferred_spawn_payload()
            expect((out or "").strip() == "", "queue should return empty payload for invalid data")
            expect(qpath.read_text(encoding="utf-8") == payload, "queue should be preserved on parse failure")
        finally:
            if prev_taskdata is None:
                os.environ.pop("TASKDATA", None)
            else:
                os.environ["TASKDATA"] = prev_taskdata

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
    """Malformed JSON should not crash on-modify; it should pass through raw input."""
    path = _find_hook_file("on-modify-nautical.py")
    raw = "{not-json}"
    p = _run_hook_script_raw(path, raw)
    expect(p.returncode == 0, f"on-modify returned {p.returncode}")
    expect((p.stdout or "").strip() == raw.strip(), "on-modify did not pass through raw input")

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
        mod._DNF_DISK_CACHE_PATH_LEGACY = Path(os.path.join(td, "dnf_cache.pkl"))
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = OrderedDict([("k1", {"v": 1})])
        mod._DNF_DISK_CACHE_DIRTY = True

        mod._save_dnf_disk_cache()

        mod._DNF_DISK_CACHE = None
        loaded = mod._load_dnf_disk_cache()
        expect(isinstance(loaded, OrderedDict), "DNF cache load should return OrderedDict")
        expect("k1" in loaded, "DNF cache did not round-trip expected key")


def test_on_add_dnf_cache_corrupt_payload_recovers():
    """on-add DNF cache load recovers from corrupt pickle without raising."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_cache_corrupt_test")
    if not hasattr(mod, "_load_dnf_disk_cache"):
        raise AssertionError("on-add hook does not expose DNF cache helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cache_path = os.path.join(td, "dnf_cache.jsonl")
        with open(cache_path, "wb") as f:
            f.write(b"not a pickle")

        mod._DNF_DISK_CACHE_PATH = Path(cache_path)
        mod._DNF_DISK_CACHE_LOCK = Path(cache_path).with_suffix(".lock")
        mod._DNF_DISK_CACHE_PATH_LEGACY = Path(os.path.join(td, "dnf_cache.pkl"))
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None

        loaded = mod._load_dnf_disk_cache()
        expect(isinstance(loaded, OrderedDict), "DNF cache corrupt load should return OrderedDict")
        expect(len(loaded) == 0, "DNF cache corrupt load should return empty cache")


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
        mod._DNF_DISK_CACHE_PATH_LEGACY = Path(os.path.join(td, "dnf_cache.pkl"))
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None

        _ = mod._load_dnf_disk_cache()
        quarantined = [p for p in os.listdir(td) if p.startswith("dnf_cache.corrupt.") and p.endswith(".jsonl")]
        expect(quarantined, "DNF cache should quarantine invalid JSONL")


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
        mod._DNF_DISK_CACHE_PATH_LEGACY = Path(os.path.join(td, "dnf_cache.pkl"))
        mod._DNF_DISK_CACHE_ENABLED = True
        mod._DNF_DISK_CACHE = None

        loaded = mod._load_dnf_disk_cache()
        expect(isinstance(loaded, OrderedDict), "DNF cache size-guard load should return OrderedDict")
        expect(len(loaded) == 0, "DNF cache size-guard load should return empty cache")


def test_on_modify_spawn_deferred_then_drain():
    """on-modify deferred spawn drains queue after lock-style timeout."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_defer_drain_test")
    if not hasattr(mod, "_spawn_child_atomic") or not hasattr(mod, "_drain_deferred_spawn_queue"):
        raise AssertionError("on-modify hook does not expose spawn/defer helpers")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod.TW_DATA_DIR = td_path
        mod._SPAWN_QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._SPAWN_QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"
        mod._SPAWN_QUEUE_KICK = td_path / ".nautical_spawn_queue.kick"

        def _run_task_timeout(cmd, **kwargs):
            return False, "", "timeout"

        mod._run_task = _run_task_timeout

        try:
            mod._spawn_child_atomic({"description": "test"}, {"uuid": "parent"})
            raise AssertionError("Expected _SpawnDeferred due to timeout")
        except Exception as e:
            if not isinstance(e, mod._SpawnDeferred):
                raise

        expect(mod._SPAWN_QUEUE_PATH.exists(), "Deferred spawn queue file missing")

        def _run_task_ok(cmd, **kwargs):
            return True, "", ""

        mod._run_task = _run_task_ok
        rc = mod._drain_deferred_spawn_queue()
        expect(rc == 0, f"Drain should succeed, got rc={rc}")

        if mod._SPAWN_QUEUE_PATH.exists():
            txt = mod._SPAWN_QUEUE_PATH.read_text(encoding="utf-8").strip()
            expect(txt == "", "Drain should empty the deferred spawn queue")


def test_on_modify_spawn_queue_batch_limit():
    """on-modify queue drain batches when the queue is large."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_queue_batch_test")
    if not hasattr(mod, "_take_deferred_spawn_payload"):
        raise AssertionError("on-modify hook does not expose _take_deferred_spawn_payload")

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        mod._SPAWN_QUEUE_PATH = td_path / ".nautical_spawn_queue.jsonl"
        mod._SPAWN_QUEUE_LOCK = td_path / ".nautical_spawn_queue.lock"

        mod._SPAWN_QUEUE_MAX_BYTES = 1
        mod._SPAWN_QUEUE_DRAIN_MAX_ITEMS = 2

        items = [{"uuid": f"{i:08d}"} for i in range(5)]
        mod._SPAWN_QUEUE_PATH.write_text(
            "\n".join(json.dumps(o) for o in items) + "\n",
            encoding="utf-8",
        )

        payload = mod._take_deferred_spawn_payload()
        taken = [ln for ln in payload.splitlines() if ln.strip()]
        expect(len(taken) == 2, f"Expected 2 items taken, got {len(taken)}")

        remaining = mod._SPAWN_QUEUE_PATH.read_text(encoding="utf-8").strip().splitlines()
        remaining = [ln for ln in remaining if ln.strip()]
        expect(len(remaining) == 3, f"Expected 3 items remaining, got {len(remaining)}")


def test_on_add_run_task_timeout():
    """on-add _run_task returns timeout on subprocess timeouts."""
    hook = _find_hook_file("on-add-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_add_run_task_timeout_test")
    if not hasattr(mod, "_run_task"):
        raise AssertionError("on-add hook does not expose _run_task")

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="task", timeout=1)

    orig = mod.subprocess.run
    mod.subprocess.run = _raise_timeout
    try:
        ok, _out, err = mod._run_task(["task", "export"], timeout=0.1, retries=1)
    finally:
        mod.subprocess.run = orig

    expect(not ok, "_run_task should return ok=False on timeout")
    expect(err == "timeout", f"_run_task should report timeout, got {err!r}")


def test_on_modify_run_task_timeout():
    """on-modify _run_task returns timeout on subprocess timeouts."""
    hook = _find_hook_file("on-modify-nautical.py")
    mod = _load_hook_module(hook, "_nautical_on_modify_run_task_timeout_test")
    if not hasattr(mod, "_run_task"):
        raise AssertionError("on-modify hook does not expose _run_task")

    def _raise_timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="task", timeout=1)

    orig = mod.subprocess.run
    mod.subprocess.run = _raise_timeout
    try:
        ok, _out, err = mod._run_task(["task", "export"], timeout=0.1, retries=1)
    finally:
        mod.subprocess.run = orig

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
    test_warn_once_per_day_stamp_written,
    test_warn_once_per_day_no_diag_silent,
    test_hook_stdout_strict_json_with_diag_on_add,
    test_hook_stdout_strict_json_with_diag_on_modify,
    test_on_modify_invalid_json_passthrough,
    test_on_add_fail_and_exit_emits_json,
    test_on_modify_read_two_fuzz_inputs,
    test_on_add_dnf_cache_versioned_payload,
    test_on_add_dnf_cache_corrupt_payload_recovers,
    test_on_add_dnf_cache_quarantines_invalid_jsonl,
    test_on_add_dnf_cache_size_guard_skips_load,
    test_on_modify_spawn_deferred_then_drain,
    test_on_modify_spawn_queue_batch_limit,
    test_spawn_queue_recovers_partial_payload,
    test_spawn_queue_invalid_payload_preserved,
    test_on_add_run_task_timeout,
    test_on_modify_run_task_timeout,
    test_on_modify_export_uuid_short_invalid_json,
    test_on_modify_missing_taskdata_uses_tw_dir,
    test_hooks_no_direct_subprocess_run,
    test_chain_integrity_warnings_detects_issues,
    test_dst_round_trip_noon_preserves_local_date,

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
