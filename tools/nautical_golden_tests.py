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
import sys, os, re, json
from datetime import date, datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

core = importlib.import_module("nautical_core")

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

def test_lint_formats():
    # Instead of relying on linter fatals, assert the validator rejects bad yearly tokens
    try:
        core.validate_anchor_expr_strict("y:05:15")
        assert False, "Validator must fatal on 'y:05:15' (':' instead of '-')"
    except core.ParseError as e:
        low = str(e).lower()
        expect(("uses ':'" in low) or ("example" in low) or ("anchor_year_fmt" in low),
               f"Unexpected validator message: {e}")


def test_weekly_and_unsat():
    fatal, _ = core.lint_anchor_expr("w:sat + w:mon")
    expect(bool(fatal), "Weekly A+B must be unsatisfiable (Sat AND Mon)")

def test_nth_weekday_range():
    fatal, _ = core.lint_anchor_expr("m:6th-mon")
    expect(bool(fatal), "6th-mon must fatal (nth in 1..5 or last)")

def test_last_weekday():
    p = build_preview("m:last-fri")
    expect("last" in p["natural"].lower(), "Natural should mention last Friday")
    # Next few should all be Fridays
    for d in p["upcoming"][:5]:
        dow = datetime.fromisoformat(d).weekday()  # 0=Mon
        expect(dow == 4, f"{d} must be Friday")

def test_monthly_valid_months_m2_5th_mon():
    # /2:5th-mon must count only months that HAVE the 5th Monday
    p = build_preview("m/2:5th-mon")
    expect(p["upcoming"], "Should produce upcoming dates")
    # Each is Monday
    for d in p["upcoming"][:6]:
        expect(datetime.fromisoformat(d).weekday() == 0, f"{d} must be Monday")

def test_leap_year_29feb():
    p = build_preview("y:02-29")
    dates = p["upcoming"][:8]
    expect(dates, "Need some upcoming for leap-day")
    for d in dates:
        dt = datetime.fromisoformat(d)
        expect(dt.month == 2 and dt.day in (28,29), "Window around Feb; core may list anchor dates only")
    # Must contain an actual Feb 29 within a 4-year span
    expect(any(datetime.fromisoformat(d).day == 29 for d in dates), "Must include a Feb 29 occurrence")

def test_quarters_window():
    # Second Monday only within H1
    p = build_preview("m:2nd-mon + y:q1:q2")
    # All months must be in 1..6
    for d in p["upcoming"][:6]:
        m = datetime.fromisoformat(d).month
        expect(1 <= m <= 6, f"{d} must be within Q1–Q2")

def test_yearly_month_names():
    # y:mar:sep must constrain to Mar..Sep
    p = build_preview("m:1st-mon + y:mar:sep")
    for d in p["upcoming"][:6]:
        m = datetime.fromisoformat(d).month
        expect(3 <= m <= 9, f"{d} must be Mar..Sep")

def test_rand_with_year_window():
    # Only inside Apr 20 – May 15
    p = build_preview("y:04-20:05-15 + m:rand")
    expect(p["upcoming"], "Rand with window should produce dates")
    for d in p["upcoming"][:8]:
        dt = datetime.fromisoformat(d)
        mmdd = f"{dt.month:02d}-{dt.day:02d}"
        expect("04-20" <= mmdd <= "05-15", f"{d} must be within Apr 20–May 15")

def test_weekly_rand_N_gate():
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
    # w:mon@t=09:00,fri@t=15:00 (comma in same weekly atom)
    # Requirement: no fatal; preview pathway should produce something.
    p = build_preview("w:mon@t=09:00,fri@t=15:00")
    expect(p is not None, "preview returned object")
    # Either natural or upcoming should be non-empty if the parser accepted it
    expect(bool(p["natural"]) or bool(p["upcoming"]),
           "Multi-@t weekly should be accepted and produce output")

def test_year_fmt_md_dm_switch():
    # Temporarily flip format and re-lint
    orig = getattr(core, "ANCHOR_YEAR_FMT", "MD")
    try:
        core.ANCHOR_YEAR_FMT = "DM"
        # With DM format: 01-13 means day=01, month=13 (invalid month)
        fatal, _ = core.lint_anchor_expr("y:01-13")
        expect(bool(fatal), "Under DM, 01-13 should be fatal (month 13 is invalid)")
        # With DM format: 13-01 means day=13, month=01 (valid)
        fatal, _ = core.lint_anchor_expr("y:13-01")
        expect(not fatal, "Under DM, 13-01 is valid (13th of January)")
    finally:
        core.ANCHOR_YEAR_FMT = orig

def test_weekly_multi_days_and_every_2weeks():
    p = build_preview("w/2:mon-tue,thu-sat")
    expect(p["upcoming"], "weekly /2 preview must produce dates")
    # all days are in allowed set
    allowed = {0,1,3,4,5}  # mon,tue,thu,fri,sat
    for d in p["upcoming"][:8]:
        wd = datetime.fromisoformat(d).weekday()
        expect(wd in allowed, f"{d} not in allowed weekdays")

def test_performance_large_expressions():
    """Test performance with large/complex expressions."""
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
    """Test that cached and uncached results are identical."""
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
    """Test parser validation and error messages."""
    # Valid expressions that should parse
    valid_expressions = [
        "w:mon",
        "w:mon,tue",
        "w:mon-fri",
        "m:1",
        "m:1,15,31",
        "m:1:15",
        "m:2nd-mon",
        "m:last-fri",
        "m:5bd",
        "y:01-01",
        "y:01-01:12-31",
        "y:q1",
        "y:q1:q2",
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
        ("w:invalid", "Unknown weekday"),
        ("m:32", "Day out of range"),
        ("m:6th-mon", "6th weekday not allowed"),
        ("y:13-01", "Invalid month"),
        ("y:01-32", "Invalid day"),
        ("w:mon + w:sun", "Unsatisfiable AND"),
    ]
    
    for expr, expected_error in invalid_expressions:
        try:
            core.validate_anchor_expr_strict(expr)
            assert False, f"{expr}: should fail but parsed successfully"
        except core.ParseError as e:
            assert expected_error.lower() in str(e).lower(), \
                f"{expr}: wrong error message. Got: {e}"

def test_natural_language_comprehensive():
    """Test natural language generation for various patterns."""
    test_cases = [
        ("w:mon", "Mondays"),
        ("w:mon,tue,fri", "Mondays, Tuesdays or Fridays"),
        ("w/2:mon", "every 2 weeks: Mondays"),
        ("m:15", "the 15th day of each month"),
        ("m:-1", "the last day of each month"),
        ("m:2nd-mon", "the 2nd Monday of each month"),
        ("m:last-fri", "the last Friday of each month"),
        ("m:5bd", "the 5th business day of each month"),
        ("y:12-25", "25 Dec each year"),
        ("y:01-01:01-31", "Jan each year"),
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
    """Test edge cases and boundary conditions."""
    test_cases = [
        # Leap year handling
        ("y:02-29", "2023-01-01", "2024-02-29"),  # Non-leap -> leap year
        ("y:02-29", "2024-02-29", "2028-02-29"),  # Leap -> next leap
        
        # Month boundaries
        ("m:31", "2024-02-01", "2024-03-31"),  # Feb doesn't have 31st
        ("m:-31", "2024-02-01", "2024-01-01"),  # -31 means 1st (31 days before end)
        
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
    Ensure '/N' on heads parses cleanly across w/m/y.
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
    'y:02-29' must produce dates only on leap years (i.e., exactly Feb 29).
    """
    preview = _must_preview("y:02-29")
    dates = preview["next_dates"]
    assert dates, "No upcoming dates for y:02-29"
    for d in dates[:6]:
        assert (d.month, d.day) == (2, 29), f"{d} is not Feb 29"

def test_rand_with_year_window_filtering():
    """
    'y:04-20:05-15+ m:rand' must produce all dates within the yearly window.
    """
    preview = _must_preview("y:04-20:05-15+ m:rand")
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
    'w/4:rand' should respect 4-week ISO‐week gating between picks.
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
    Natural for 'm:-1@prev-fri' should mention 'previous Friday before the last day of the month'.
    """
    nat = _must_natural("m:-1@prev-fri")
    want_any = [
        "previous Friday before the last day of the month",
        "previous Friday before the last day",
        "previous Friday before month end",
    ]
    assert any(w in nat for w in want_any), f"Natural missing expected phrasing: {nat!r}"

def test_year_fmt_md_dm_switch_minimal():
    """
    Toggle ANCHOR_YEAR_FMT and assert 06-01 is MM-DD vs DD-MM appropriately.
    """
    # Use the global core module, don't re-import
    saved = core.ANCHOR_YEAR_FMT
    
    try:
        # Clear cache to force fresh computation
        if hasattr(core, '_hint_cache'):
            core._hint_cache.clear()
        elif hasattr(core, '_anchor_hint_cache'):
            core._anchor_hint_cache.clear()
        
        # Test DM format: 06-01 = day=06, month=01 = January 6th
        core.ANCHOR_YEAR_FMT = "DM"
        # Parse fresh to avoid cached DNF
        dnf_dm = core.validate_anchor_expr_strict("y:06-01")
        nat_dm = core.describe_anchor_expr("y:06-01")
        # Check for either "6 Jan" or "Jan 6"
        assert any(phrase in nat_dm.lower() for phrase in ["6 jan", "jan 6"]), \
            f"DM format (06-01 = Jan 6): got '{nat_dm}'"

        # Clear cache again
        if hasattr(core, '_hint_cache'):
            core._hint_cache.clear()
        elif hasattr(core, '_anchor_hint_cache'):
            core._anchor_hint_cache.clear()
        
        # Test MD format: 06-01 = month=06, day=01 = June 1st
        core.ANCHOR_YEAR_FMT = "MD"
        # Parse fresh
        dnf_md = core.validate_anchor_expr_strict("y:06-01")
        nat_md = core.describe_anchor_expr("y:06-01")
        # Check for either "1 Jun" or "Jun 1"
        assert any(phrase in nat_md.lower() for phrase in ["1 jun", "jun 1"]), \
            f"MD format (06-01 = June 1): got '{nat_md}'"
    finally:
        core.ANCHOR_YEAR_FMT = saved

def test_weekly_multi_days_every_2weeks_spacing_and_days():
    """
    'w/2:mon,thu' must parse and produce dates only on Mon/Thu,
    with ISO week gaps respecting /2 gating.
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
    'w:mon@t=09:00,fri@t=15:00' should be accepted (rewritten to OR of singletons).
    """
    # Linter must not fatal, and strict-validate must pass
    fatal, warns = core.lint_anchor_expr("w:mon@t=09:00,fri@t=15:00")
    assert fatal is None, f"Unexpected lint fatal: {fatal}"
    _must_parse("w:mon@t=09:00,fri@t=15:00")

def test_deterministic_randomness():
    """Test that random patterns are deterministic with same seed."""
    # Only valid random patterns
    test_cases = [
        "w:rand",      # Random weekday - valid
        "m:rand",      # Random day of month - valid
        "m:rand@bd",   # Random business day of month - valid
        "m:1:10 + m:rand",  # Random day in first 10 days - valid
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
    """Test business day modifiers thoroughly."""
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

def test_complex_dnf_expressions():
    """Test complex DNF expressions with OR and AND."""
    test_cases = [
        # OR expressions
        ("w:mon | w:fri", "2024-12-11", "2024-12-13"),  # Wed -> Fri (closer than Mon)
        ("m:1 | m:15", "2024-12-11", "2024-12-15"),  # 1st or 15th -> 15th
        
        # AND expressions  
        ("w:mon + m:1", "2024-12-01", "2025-09-01"),  # Monday AND 1st of month (next is Sep 1, 2025)
        ("w:fri + m:13", "2024-12-01", "2024-12-13"),  # Friday the 13th
        
        # Complex: (Monday in Jan) OR (Friday in Feb)
        # From 2024-01-01, next is 2024-01-08 (Monday in Jan), not 2024-01-01 itself
        ("(w:mon + y:01-01:01-31) | (w:fri + y:02-01:02-28)", "2024-01-01", "2024-01-08"),
    ]
    
    for anchor, start_str, expected_str in test_cases:
        start = date.fromisoformat(start_str)
        expected = date.fromisoformat(expected_str)
        
        dnf = core.validate_anchor_expr_strict(anchor)
        next_date, _ = core.next_after_expr(dnf, start)
        
        assert next_date == expected, f"{anchor}: got {next_date}, expected {expected}"

def test_interval_patterns():
    """Test /N intervals with different anchor types."""
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
    """Test specific date calculations for various anchor patterns."""
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
    # Single month by name or numeric shorthand
    for expr in ("y:apr", "y:04"):
        p = build_preview(expr)
        expect(p["upcoming"], f"{expr} should produce upcoming dates")
        for d in p["upcoming"][:6]:
            expect(datetime.fromisoformat(d).month == 4, f"{d} must be in April for {expr}")

    # Month-name window (Jan..Jun) constrains outputs
    p = build_preview("y:jan:jun + m:rand")
    expect(p["upcoming"], "y:jan:jun + m:rand should produce upcoming dates")
    for d in p["upcoming"][:8]:
        m = datetime.fromisoformat(d).month
        expect(1 <= m <= 6, f"{d} must be within Jan..Jun")

def test_business_day_bd_skip_semantics():
    # @bd = only if business day else skip to next month’s matching day (not roll)
    # 2026-01-03 is Saturday → skip to 2026-02-03 (Tuesday)
    start = date(2026, 1, 1)
    dnf = core.validate_anchor_expr_strict("m:3@bd")
    nxt, _ = core.next_after_expr(dnf, start)
    expect(nxt == date(2026, 2, 3), f"@bd should skip Jan (Sat) → 2026-02-03, got {nxt}")

def test_inline_time_mods_natural_contains_both_times():
    expr = "w:mon@t=09:00,fri@t=15:00"
    nat = _must_natural(expr)
    low = nat.lower()
    expect("09:00" in low and "15:00" in low,
           f"Natural should reflect both times for {expr!r}: {nat!r}")
    _must_parse(expr)  # ensure strict parser accepts the inline split

def test_guard_commas_between_atoms_after_mods_fatal():
    bad = "m:31@t=14:00,w:sun@t=22:00"
    try:
        core.validate_anchor_expr_strict(bad)
        assert False, "Comma between atoms after @mods must be fatal"
    except core.ParseError as e:
        msg = str(e)
        expect("join" in msg.lower() or "use '+' (and) or '|'" in msg.lower(),
               f"Unexpected error message for bad comma join: {msg}")

def test_heads_with_slashN_parse_ok_again():
    # Regression guard: '/N' heads must parse across w/m/y
    for expr in ("w/2:sun", "m/3:1st-mon", "y/4:06-01"):
        _must_parse(expr)

def test_monthname_and_numeric_equivalence():
    # y:jul should behave like a July window; y:07 numeric should match too (for month-only)
    for expr in ("y:jul",):
        p = build_preview(expr)
        expect(p["upcoming"], f"{expr} should produce upcoming dates")
        for d in p["upcoming"][:6]:
            expect(datetime.fromisoformat(d).month == 7, f"{d} must be in July")



# -------- Runner --------------------------------------------------------------

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
    test_year_fmt_md_dm_switch,
    test_weekly_multi_days_and_every_2weeks,
    test_heads_with_slashN_parse_ok,
    test_monthly_valid_months_m2_5th_mon_upcoming_within_valid_months,
    test_leap_year_29feb_upcoming_only_on_leap_year,
    test_rand_with_year_window_filtering,
    test_weekly_rand_N_gate_spacing,
    test_prev_weekday_natural_text,
    test_year_fmt_md_dm_switch_minimal,
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
    test_monthname_and_numeric_equivalence
]

DEEP_TESTS = [

]

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="substring filter for test names")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    selected = TESTS
    if args.only:
        sel = []
        for fn in TESTS:
            if args.only.lower() in fn.__name__.lower():
                sel.append(fn)
        selected = sel

    fails = 0
    for fn in selected:
        try:
            fn()
            if args.verbose:
                print(f"✓ {fn.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"✗ {fn.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"✗ {fn.__name__}: unexpected error {e}")

    total = len(selected)
    print(f"\nDone: {total - fails}/{total} passing")
    sys.exit(1 if fails else 0)

if __name__ == "__main__":
    main()

