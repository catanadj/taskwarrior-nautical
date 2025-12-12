#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
on-add-nautical.py (production)

Features:
- Reject legacy weekly colon ranges on add (use w:mon-fri).
- Auto-assign due when missing (anchor first, else cp).
- Panel logic:
    * If user set due -> First due = user's due; Next anchor shown separately.
    * If no due       -> First due = first anchor (or entry+cp) and due is auto-set.
- Optional warning if user's due isn't on an anchor day (constant toggle below).
- DST-safe datetime composition (local date+hh:mm -> UTC).
- Inclusive chainUntil behavior in preview.
"""

from __future__ import annotations

import sys, json, os, importlib
from pathlib import Path
from datetime import timedelta, timezone, datetime
from functools import lru_cache
import shlex, subprocess

# ========= User-togglable constants =========================================
ANCHOR_WARN = True  # If True, warn when a user-provided due is not on an anchor day
UPCOMING_PREVIEW = 5  # How many future dates to preview.
_MAX_ITERATIONS = 2000
_MAX_PREVIEW_ITERATIONS = 2000
_MAX_CHAIN_DURATION_YEARS = 5  # warn if chain extends this far
# ============================================================================

# --------------------------------------------------------------------------------------
# Locate and import nautical_core (looks in ~/.task/hooks, ~/.task, $TASKDATA, $TASKRC)
# --------------------------------------------------------------------------------------
HOOK_DIR = Path(__file__).resolve().parent
TW_DIR = HOOK_DIR.parent

_candidates: list[Path] = []


def _add(p):
    if not p:
        return
    p = Path(p).expanduser().resolve()
    if p not in _candidates:
        _candidates.append(p)


_add(HOOK_DIR)  # dev copy next to hook
_add(TW_DIR)  # ~/.task
if os.environ.get("TASKDATA"):
    _add(os.environ["TASKDATA"])
if os.environ.get("TASKRC"):
    _add(Path(os.environ["TASKRC"]).parent)
_add(Path.home() / ".task")

core = None
for base in _candidates:
    pyfile = base / "nautical_core.py"
    pkgini = base / "nautical_core" / "__init__.py"
    if pyfile.is_file():
        sys.path.insert(0, str(base))
        core = importlib.import_module("nautical_core")
        break
    if pkgini.is_file():
        sys.path.insert(0, str(base))
        core = importlib.import_module("nautical_core")
        break

if core is None:
    msg = "nautical_core.py not found. Looked in:\n  " + "\n  ".join(
        str(p) for p in _candidates
    )
    raise ModuleNotFoundError(msg)


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


@lru_cache(maxsize=256)
def _validate_anchor_expr_cached(expr: str):
    return core.validate_anchor_expr_strict(expr)


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

def _panel(title, rows, kind: str = "info"):
    """
    Small Rich helper to render 2-column panels in a consistent style.

    `rows` is a list of (label, value) pairs.
    - label can be `None` to create a spacer row.
    - Some labels like "Warning", "Note", "Error" get stronger styling automatically.
    - `kind` selects a colour theme (preview_anchor, preview_cp, error, warning, info).
    """
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception:
        # Fallback: plain text on stderr
        sys.stderr.write(title + "\n")
        for k, v in rows:
            if k is None:
                sys.stderr.write(f"{v}\n")
            else:
                sys.stderr.write(f"{k}: {v}\n")
        return

    THEMES = {
        "preview_anchor": {
            "border": "light_sea_green",
            "title": "light_sea_green",
            "label": "cyan",
        },
        "preview_cp": {
            "border": "hot_pink",
            "title": "bright_green",
            "label": "green",
        },
        "error": {"border": "red", "title": "red", "label": "red"},
        "warning": {"border": "yellow", "title": "yellow", "label": "yellow"},
        "info": {"border": "blue", "title": "cyan", "label": "cyan"},
    }
    theme = THEMES.get(kind, THEMES["info"])

    console = Console(file=sys.stderr, force_terminal=True)
    t = Table.grid(padding=(0, 1), expand=False)
    t.add_column(style=f"bold {theme['label']}", no_wrap=True, justify="right")
    t.add_column(style="white")

    for k, v in rows:
        if k is None:
            # spacer / section-break line
            t.add_row("", v or "")
            continue

        label_text = Text(str(k))
        lk = str(k).lower()
        if "warning" in lk:
            label_text.stylize("bold yellow")
        elif "error" in lk:
            label_text.stylize("bold red")
        elif "note" in lk:
            label_text.stylize("italic cyan")

        t.add_row(label_text, v)

    console.print(
        Panel(
            t,
            title=Text(title, style=f"bold {theme['title']}"),
            border_style=theme["border"],
            expand=False,
            padding=(0, 1),
        )
    )


def _format_anchor_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Compact layout for anchor preview .
    """
    # Upcoming numbering to match the *chain link* index:
    #   link #1 → First due
    #   link #2 → Next anchor (if present)
    #   link #3+→ Upcoming list
    has_next_anchor = any(k == "Next anchor" for k, _ in rows)
    upcoming_start = 3 if has_next_anchor else 2

    # Extract delta row; later we inline it into "First due"
    delta_text = None
    for k, v in rows:
        if k == "Delta":
            delta_text = v
            break

    config_keys = {"Pattern", "Natural"}
    schedule_keys = {"First due", "Next anchor", "[auto-due]", "Upcoming"}
    limits_keys = {"Limits", "Final (until)"}

    config: list[tuple[str, str]] = []
    schedule: list[tuple[str, str]] = []
    limits: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    rand: list[tuple[str, str]] = []
    chain: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        # Skip standalone Delta row; we embed it into "First due"
        if k == "Delta":
            continue

        if k == "First due" and delta_text:
            v = f"{v}  [dim](Δ {delta_text})[/]"

        lk = (str(k).lower() if k is not None else "")

        if k in config_keys:
            config.append((k, v))
        elif k in schedule_keys:
            if k == "Upcoming" and v and v != "[dim]–[/]":
                # Add correct link numbers: 3,4,... if Next anchor present; else 2,3,...
                lines = v.splitlines()
                new_lines = []
                idx = upcoming_start
                for line in lines:
                    new_lines.append(f"[dim]{idx:>2} ▸[/] {line}")
                    idx += 1
                v = "\n".join(new_lines)
            schedule.append((k, v))
        elif k in limits_keys:
            limits.append((k, v))
        elif lk.startswith("warning") or lk.startswith("note"):
            warnings.append((k, v))
        elif k == "Rand":
            rand.append((k, v))
        elif k == "Chain":
            chain.append((k, v))
        else:
            others.append((k, v))

    # Any unknown rows → CONFIG so they don't vanish
    config.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(config)
    _add(schedule)
    _add(limits)
    _add(warnings)
    _add(rand)
    _add(chain)

    return out or rows




def _format_cp_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    """Compact layout for classic cp preview.
    """
    # For cp chains, First due is link #1, Upcoming are 2,3,...
    upcoming_start = 2

    # Extract delta row; later we inline it into "First due"
    delta_text = None
    for k, v in rows:
        if k == "Delta":
            delta_text = v
            break

    config_keys = {"Period"}
    schedule_keys = {"First due", "[auto-due]", "Upcoming"}
    limits_keys = {"Limits", "Final (max)", "Final (until)"}

    config: list[tuple[str, str]] = []
    schedule: list[tuple[str, str]] = []
    limits: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    chain: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        if k == "Delta":
            continue

        if k == "First due" and delta_text:
            v = f"{v}  [dim](Δ {delta_text})[/]"

        lk = (str(k).lower() if k is not None else "")

        if k in config_keys:
            config.append((k, v))
        elif k in schedule_keys:
            if k == "Upcoming" and v and v != "[dim]–[/]":
                lines = v.splitlines()
                new_lines = []
                idx = upcoming_start
                for line in lines:
                    new_lines.append(f"[dim]{idx:>2} ▸[/] {line}")
                    idx += 1
                v = "\n".join(new_lines)
            schedule.append((k, v))
        elif k in limits_keys:
            limits.append((k, v))
        elif lk.startswith("warning") or lk.startswith("note"):
            warnings.append((k, v))
        elif k == "Chain":
            chain.append((k, v))
        else:
            others.append((k, v))

    config.extend(others)

    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]):
        nonlocal out
        if not group:
            return
        if out:
            out.append((None, ""))  # spacer line
        out.extend(group)

    _add(config)
    _add(schedule)
    _add(limits)
    _add(warnings)
    _add(chain)

    return out or rows


def _fail_and_exit(title: str, msg: str) -> None:
    # Pretty panel -> stderr
    _panel(f"❌ {title}", [("Message", msg)], kind="error")
    # Minimal feedback -> stdout (what Task expects when a hook fails)
    print(f"{title}: {msg}")
    sys.exit(1)


def _error_and_exit(msg_tuples):
    _panel("❌ Invalid Chain", msg_tuples, kind="error")
    sys.exit(1)



# Local ISO string back to Taskwarrior (lets default-due adjusters run)
def _fmt_local_for_task(dt_utc):
    dl = core.to_local(dt_utc)
    return dl.strftime("%Y-%m-%dT%H:%M:%S")


# Helper to validate chainUntil is in the future
def _validate_until_not_past(until_dt, now_utc) -> tuple[bool, str | None]:
    """Check if chainUntil is in the past. Returns (is_valid, error_msg)."""
    if not until_dt:
        return (True, None)

    grace = timedelta(minutes=1)
    if until_dt < (now_utc - grace):
        past_by = now_utc - until_dt
        past_s = core.humanize_delta(until_dt, now_utc, use_months_days=False)
        return (False, f"chainUntil is in the past (was {past_s} ago)")

    return (True, None)


# Helper to check if due is in the past (warning only)
def _check_due_in_past(due_dt, now_utc) -> tuple[bool, str | None]:
    """
    Check if user-provided due is in the past.
    Returns (is_past, warning_msg).
    This is a warning only - allows backlog tasks.
    """
    if not due_dt:
        return (False, None)

    grace = timedelta(minutes=1)
    if due_dt < (now_utc - grace):
        ago = now_utc - due_dt
        ago_s = core.humanize_delta(due_dt, now_utc, use_months_days=False)
        return (True, f"Due date is in the past ({ago_s} ago).")

    return (False, None)


# Helper to warn if chain extends unreasonably far
def _validate_chain_duration_reasonable(
    until_dt, now_utc, first_due, kind
) -> tuple[bool, str | None]:
    """
    Warn if chain extends too far into future.
    Returns (is_reasonable, warning_msg).
    """
    if not until_dt:
        return (True, None)

    span = until_dt - now_utc
    years = span.days / 365.25

    if years > _MAX_CHAIN_DURATION_YEARS:
        return (False, f"Chain extends {years:.1f} years into future.")

    return (True, None)


# Helper to validate cp/anchor not missing
def _validate_kind_not_conflicting(cp_str, anchor_str) -> tuple[bool, str | None]:
    """
    Check if both cp and anchor are set (invalid).
    Returns (is_valid, error_msg).
    """
    has_cp = bool((cp_str or "").strip())
    has_anchor = bool((anchor_str or "").strip())

    if has_cp and has_anchor:
        return (False, "Cannot set both 'cp' and 'anchor'. Choose one.")

    return (True, None)


# Helper to validate chainMax > 0
def _validate_cpmax_positive(cpmax) -> tuple[bool, str | None]:
    """Check if chainMax is valid (positive). Returns (is_valid, error_msg)."""
    if cpmax <= 0:
        return (False, "chainMax must be > 0")

    return (True, None)


# Helper to safely parse with context
def _safe_parse_datetime(s, field_name) -> tuple[datetime | None, str | None]:
    """
    Parse datetime with error context.
    Returns (datetime, error_msg).
    """
    if not s:
        return (None, None)

    try:
        dt = core.parse_dt_any(s)
        if dt is None:
            return (None, f"{field_name}: Unrecognized datetime format '{s}'")
        return (dt, None)
    except ValueError as e:
        return (None, f"{field_name}: {str(e)}")
    except Exception as e:
        return (None, f"{field_name}: Unexpected parsing error: {str(e)}")


def _validate_no_legacy_colon_ranges(expr: str) -> tuple[bool, str | None]:
    """
    Reject legacy weekly colon ranges (e.g., 'mon:wed:fri').
    Users should use w:mon-fri instead.
    Returns (is_valid, error_msg).
    """
    if not expr:
        return (True, None)
    
    expr = expr.strip()
    
    # Check for colon-separated day patterns (legacy format)
    # Patterns like: mon:wed:fri, mon:wed, tue:thu
    day_names = {"mon", "tue", "wed", "thu", "fri", "sat", "sun"}
    
    # Split by spaces first to handle DNF terms
    terms = expr.split()
    for term in terms:
        # Remove optional parentheses for DNF
        clean_term = term.strip("()")
        
        # Check if this looks like a colon-separated day list
        if ":" in clean_term:
            parts = clean_term.split(":")
            # Check if all parts look like day abbreviations
            if len(parts) >= 2 and all(p.lower() in day_names for p in parts):
                legacy_example = ":".join(parts)
                new_format = f"w:{parts[0]}-{parts[-1]}"
                return (
                    False,
                    f"Legacy colon range '{legacy_example}' detected. "
                    f"Use '{new_format}' instead (e.g., '{new_format}' for '{legacy_example}')."
                )
    
    return (True, None)

def _safe_parse_duration(s, field_name) -> tuple[timedelta | None, str | None]:
    """
    Parse duration with error context.
    Returns (timedelta, error_msg).
    """
    if not s:
        return (None, None)

    try:
        td = core.parse_cp_duration(s)
        if td is None:
            return (
                None,
                f"{field_name}: Invalid duration format '{s}' (expected: 3d, 2w, 1h, etc.)",
            )
        return (td, None)
    except ValueError as e:
        return (None, f"{field_name}: {str(e)}")
    except Exception as e:
        return (None, f"{field_name}: Unexpected parsing error: {str(e)}")


def _validate_anchor_syntax_strict(expr) -> tuple[bool, str | None]:
    """
    Strictly validate an anchor. Accepts string or DNF.
    Returns (True, None) on success, (False, message) on failure.
    """
    try:
        core.validate_anchor_expr_strict(expr)
        return True, None
    except Exception as e:
        return False, str(e)


def _validate_anchor_mode(mode_str) -> tuple[str, str | None]:
    """
    Validate and normalize anchor_mode. Returns (normalized_mode, error_msg).
    """
    mode = (mode_str or "skip").strip().lower()

    if mode not in ("skip", "all", "flex"):
        return (
            "skip",
            f"anchor_mode must be 'skip', 'all', or 'flex' (got '{mode}'). Defaulting to 'skip'.",
        )

    return (mode, None)

def tw_export_chain(chain_id: str, since: datetime | None = None, extra: str | None = None) -> list[dict]:
    if not chain_id:
        return []
    args = ["task", "rc.hooks=off", "rc.json.array=on", "rc.verbose=nothing", f"chainID:{chain_id}"]
    if since:
        args.append(f"modified.after:{since.strftime('%Y-%m-%dT%H:%M:%S')}")
    if extra:
        args += shlex.split(extra)
    args.append("export")
    try:
        out = subprocess.check_output(args, text=True)
        data = json.loads(out.strip() or "[]")
        return data if isinstance(data, list) else [data]
    except Exception:
        return []
# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    raw = sys.stdin.read().strip()
    if not raw:
        print("", end="")
        return
    try:
        task = json.loads(raw)
    except Exception:
        _error_and_exit([("Invalid input", "on-add must receive a single JSON task")])

    now_utc = core.now_utc()
    now_local = core.to_local(now_utc)

    def _fmt(dt):
        return core.fmt_dt_local(dt)

    def _dt(s):
        return core.parse_dt_any(s)

    def _tol(dt):
        return core.to_local(dt)

    def _build_local(d, hmm):
        return core.build_local_datetime(d, hmm).astimezone(timezone.utc)

    # ========== EDGE CASE 1: Both cp and anchor set ==========
    cp_str = (task.get("cp") or "").strip()
    anchor_str = (task.get("anchor") or "").strip()
    is_valid, err = _validate_kind_not_conflicting(cp_str, anchor_str)
    if not is_valid:
        _error_and_exit([("Invalid chain config", err)])

    # Normalize
    has_cp = bool(cp_str)
    has_anchor = bool(anchor_str)
    kind = "anchor" if has_anchor else ("cp" if has_cp else None)

    # Default enable chain if cp/anchor present (respect explicit 'off')
    ch = (task.get("chain") or "").strip().lower()
    if (has_cp or has_anchor) and (not ch or ch == "off"):
        task["chain"] = "on"
        ch = "on"
        # Transparent notice to the user about the auto-enable:
        # _panel("Enabled chain", [("Reason", "cp/anchor present on add → set chain:on")], style="yellow")

    # If nothing chain-related, just pass through
    if not kind:
        print(json.dumps(task), end="")
        return

    # ========== EDGE CASE 2: chainMax validation ==========
    cpmax = core.coerce_int(task.get("chainMax"), 0)
    if cpmax:
        is_valid, err = _validate_cpmax_positive(cpmax)
        if not is_valid:
            _error_and_exit([("Invalid chainMax", err)])

    # ========== EDGE CASE 3: Parse and validate chainUntil ==========
    until_dt, err = _safe_parse_datetime(task.get("chainUntil"), "chainUntil")
    if err:
        _error_and_exit([("Invalid chainUntil", err)])

    if until_dt:
        is_valid, err = _validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            _error_and_exit([("Invalid chainUntil", err)])

    # Due seed (may be overwritten by auto-due)
    user_provided_due = bool(task.get("due"))
    due_dt = None
    past_due_warning = None
    if user_provided_due:
        due_dt, err = _safe_parse_datetime(task.get("due"), "due")
        if err:
            _error_and_exit([("Invalid due", err)])

        # ========== EDGE CASE 4: User due in past (warning only) ==========
        is_past, warn_msg = _check_due_in_past(due_dt, now_utc)
        if is_past:
            past_due_warning = warn_msg

    if due_dt is None:
        due_dt = now_utc

    due_local = _tol(due_dt)
    due_day = due_local.date()
    due_hhmm = (due_local.hour, due_local.minute)

    rows = []

    # [CHAINID] Stamp short root id on new chains (anchor/cp present, no existing chainID)
    try:
        if core.ENABLE_CHAIN_ID and ((task.get("anchor") or task.get("cp")) and not (task.get("chainID") or "").strip()):
            task["chainID"] = core.short_uuid(task.get("uuid"))
    except Exception:
        # Never block task creation on bookkeeping
        pass
    # ==================================================================================
    # ANCHOR PREVIEW
    # ==================================================================================
    if kind == "anchor":
        # ========== EDGE CASE 5: Invalid anchor expression ==========
        # ========== EDGE CASE 5a: Check for legacy colon ranges ==========
        is_valid, err = _validate_no_legacy_colon_ranges(anchor_str)
        if not is_valid:
            _error_and_exit([("Invalid anchor", err)])
        
        # ========== EDGE CASE 5b: Invalid anchor expression ==========
        is_valid, err = _validate_anchor_syntax_strict(anchor_str)
        if not is_valid:
            _error_and_exit([("Invalid anchor", err)])

        anchor_mode = ((task.get("anchor_mode") or "").strip().upper() or "ALL")
        if core.ENABLE_ANCHOR_CACHE:
            pkg = core.build_and_cache_hints(anchor_str, anchor_mode, default_due_dt=task.get("due"))
            natural = pkg.get("natural") or core.describe_anchor_expr(anchor_str, default_due_dt=task.get("due"))
            dnf = pkg.get("dnf")  # if you need it for the panel
        else:
            natural = core.describe_anchor_expr(anchor_str, default_due_dt=task.get("due"))
            dnf = core.validate_anchor_expr_strict(anchor_str)

        # ========== EDGE CASE 6: Invalid anchor_mode ==========
        mode, warn_msg = _validate_anchor_mode(task.get("anchor_mode"))
        task["anchor_mode"] = mode
        if warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

        # Safe validate
        try:
            dnf = _validate_anchor_expr_cached(anchor_str)
        except Exception as e:
            _fail_and_exit("Invalid anchor", f"anchor syntax error: {e}")


        tag = {
            "skip": "[bold bright_cyan]SKIP[/]",
            "all": "[bold yellow]ALL[/]",
            "flex": "[bold magenta]FLEX[/]",
        }.get(mode, "[bold bright_cyan]SKIP[/]")
        rows.append(("Pattern", f"[white]{anchor_str}[/]  {tag}"))
        try:
            rows.append(("Natural", f"[white]{core.describe_anchor_dnf(dnf, task)}[/]"))
        except Exception:
            pass

        base_local_date = due_day if user_provided_due else now_local.date()
        seed_base = _root_uuid_from(task) or "preview"

        # Fix: use a stable default_seed for /N gating
        interval_seed = base_local_date  # or first anchor day if you prefer

        def step_once(prev_local_date):
            try:
                nxt_date, _ = core.next_after_expr(
                    dnf,
                    prev_local_date,
                    default_seed=interval_seed,  # <-- fixed seed for the whole chain
                    seed_base=seed_base,
                )
                return nxt_date
            except Exception:
                return None


        first_date_local = step_once(base_local_date - timedelta(days=1))

        # ========== EDGE CASE 7: Anchor pattern doesn't match (no future dates) ==========
        if not first_date_local:
            _error_and_exit(
                [
                    (
                        "anchor pattern",
                        "No matching anchor dates found. Pattern may be invalid or too restrictive.",
                    )
                ]
            )

        first_hhmm = core.pick_hhmm_from_dnf_for_date(
            dnf, first_date_local, first_date_local
        ) or (due_hhmm if user_provided_due else (9, 0))
        first_due_local_dt = core.build_local_datetime(first_date_local, first_hhmm)
        first_due_utc = first_due_local_dt.astimezone(timezone.utc)

        if user_provided_due:
            display_first_due_utc = due_dt
            rows.append(
                ("First due", f"[bold bright_green]{_fmt(display_first_due_utc)}[/]")
            )
            rows.append(("Next anchor", f"[white]{_fmt(first_due_utc)}[/]"))
        else:
            display_first_due_utc = first_due_utc
            rows.append(
                ("First due", f"[bold bright_green]{_fmt(display_first_due_utc)}[/]")
            )
            task["due"] = _fmt_local_for_task(first_due_utc)
            rows.append(
                (
                    "[auto-due]",
                    "Due date was not explicitly set; assigned to first anchor match.",
                )
            )

        # Show past due warning if applicable
        if past_due_warning:
            rows.append(("Warning", f"[yellow]{past_due_warning}[/]"))

        if user_provided_due and ANCHOR_WARN:
            due_local_date = _to_local_cached(due_dt).date()
            first_after_due_date = step_once(due_local_date - timedelta(days=1))
            is_on_anchor_day = first_after_due_date == due_local_date
            if not is_on_anchor_day:
                rows.append(
                    (
                        "Note",
                        "[italic yellow]Your due is not an anchor day; chain follows anchors."
                        " To align, set due to a matching anchor day or omit due to auto-assign.[/]",
                    )
                )

        # ========== EDGE CASE 8: Chain duration warning ==========
        if until_dt:
            is_reasonable, warn_msg = _validate_chain_duration_reasonable(
                until_dt, now_utc, first_due_utc, "anchor"
            )
            if not is_reasonable and warn_msg:
                rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

        cpmax = core.coerce_int(task.get("chainMax"), 0)

        exact_until_count = None
        final_until_dt = None
        if until_dt:
            end_day = _to_local_cached(until_dt).date()
            count = 0
            prev = first_date_local - timedelta(days=1)
            last = None
            iterations = 0
            for _ in range(_MAX_PREVIEW_ITERATIONS):
                if iterations >= _MAX_ITERATIONS:
                    break
                iterations += 1

                nxt = step_once(prev)
                if not nxt or nxt > end_day:
                    break
                count += 1
                last = nxt
                prev = nxt
            exact_until_count = max(0, count - 1)
            if last:
                final_hhmm = (
                    core.pick_hhmm_from_dnf_for_date(dnf, last, first_date_local)
                    or first_hhmm
                )
                final_until_dt = core.build_local_datetime(last, final_hhmm).astimezone(
                    timezone.utc
                )

        allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
        allow_by_until = exact_until_count if exact_until_count is not None else 10**9

        # Lint for *hints only*; do not fail on linter
        _, warns = core.lint_anchor_expr(anchor_str)
        for w in warns:
            _panel("ℹ️  Lint", [("Hint", w)], kind="note")

        # Validator is the single source of truth
        core.validate_anchor_expr_strict(anchor_str)



        preview_limit = max(0, min(UPCOMING_PREVIEW, allow_by_max, allow_by_until))

        preview = []
        colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
        cur_date = first_date_local
        last_date = None
        for i in range(preview_limit):
            nxt_date = step_once(cur_date)
            if not nxt_date:
                break
            if last_date is not None and nxt_date <= last_date:
                cur_date = last_date + timedelta(days=1)
                continue

            this_hhmm = (
                core.pick_hhmm_from_dnf_for_date(dnf, nxt_date, first_date_local)
                or first_hhmm
            )
            dt_local = core.build_local_datetime(nxt_date, this_hhmm)
            dt_utc = dt_local.astimezone(timezone.utc)

            if until_dt and dt_utc > until_dt:
                break

            color = colors[min(i, len(colors) - 1)]
            preview.append(
                f"[{color}]{dt_local.strftime('%a %Y-%m-%d %H:%M %Z')}[/{color}]"
            )
            last_date = nxt_date
            cur_date = nxt_date

        rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))
        rows.append(
            (
                "Delta",
                f"[bright_yellow]{_human_delta(now_utc, display_first_due_utc, core.expr_has_m_or_y(dnf))}[/]",
            )
        )

        lim_parts = []
        if cpmax and cpmax > 0:
            lim_parts.append(f"max [bold yellow]{cpmax}[/]")
        if until_dt:
            lim_parts.append(f"until [bold yellow]{_fmt(until_dt)}[/]")
            if exact_until_count is not None:
                lim_parts.append(f"[white]{exact_until_count} more[/]")
        if lim_parts:
            rows.append(("Limits", " [dim]|[/] ".join(lim_parts)))
        if final_until_dt:
            rows.append(
                (
                    "Final (until)",
                    f"[bright_magenta]{_fmt(final_until_dt)}[/]  [dim]({_human_delta(now_utc, final_until_dt, True)})[/]",
                )
            )

        if "rand" in anchor_str.lower():
            base = _short(_root_uuid_from(task))
            rows.append(
                (
                    "Rand",
                    f"[dim italic]Preview uses provisional seed; final picks are chain-bound to {base}[/]",
                )
            )

        rows.append(
            (
                "Chain",
                "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]",
            )
        )
        rows = _format_anchor_rows(rows)
        _panel(
            "⚓︎ Anchor Preview",
            rows,
            kind="preview_anchor",
        )

        print(json.dumps(task), end="")
        return


    # ==================================================================================
    # CLASSIC CP PREVIEW
    # ==================================================================================
    td, err = _safe_parse_duration(cp_str, "cp")
    if err:
        _error_and_exit([("Invalid cp", err)])
    if not td:
        _error_and_exit([("Invalid cp", f"Couldn't parse duration from '{cp_str}'")])

    # ========== EDGE CASE 9: Chain duration warning for cp ==========
    if until_dt:
        is_reasonable, warn_msg = _validate_chain_duration_reasonable(
            until_dt, now_utc, now_utc + td if not user_provided_due else due_dt, "cp"
        )
        if not is_reasonable and warn_msg:
            rows.append(("Warning", f"[yellow]{warn_msg}[/]"))

    secs = int(td.total_seconds())
    preserve = secs % 86400 == 0

    entry_dt = _dt(task.get("entry")) if task.get("entry") else now_utc

    def add_period(dt):
        if preserve:
            dl = core.to_local(dt)
            base = core.build_local_datetime(
                (dl + timedelta(days=int(secs // 86400))).date(), (dl.hour, dl.minute)
            )
            return base.astimezone(timezone.utc)
        else:
            return (dt + td).replace(microsecond=0)

    if not user_provided_due:
        due_dt = add_period(entry_dt)
        task["due"] = _fmt_local_for_task(due_dt)
        rows.append(("[auto-due]", "Due was not set explicitly; assigned to entry+cp."))
    else:
        due_dt = _dt(task.get("due"))

    rows.append(("Period", f"[bold white]{cp_str}[/]"))
    rows.append(("First due", f"[bold bright_green]{_fmt(due_dt)}[/]"))

    cpmax = core.coerce_int(task.get("chainMax"), 0)

    exact_until_count = None
    final_until_dt = None
    if until_dt:
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
        if count > 0:
            exact_until_count = max(0, count - 1)
            final_until_dt = last

    allow_by_max = (cpmax - 1) if (cpmax and cpmax > 0) else 10**9
    allow_by_until = exact_until_count if exact_until_count is not None else 10**9
    limit = max(0, min(UPCOMING_PREVIEW, allow_by_max, allow_by_until))

    preview, nxt = [], due_dt
    colors = ["bright_cyan", "cyan", "bright_blue", "blue", "bright_black"]
    for i in range(limit):
        nxt = add_period(nxt)
        if until_dt and nxt > until_dt:
            break
        color = colors[min(i, len(colors) - 1)]
        preview.append(f"[{color}]{_fmt(nxt)}[/{color}]")
    rows.append(("Upcoming", "\n".join(preview) if preview else "[dim]–[/]"))

    rows.append(("Delta", f"[bright_yellow]{_human_delta(now_utc, due_dt, False)}[/]"))

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

    rows.append(
        ("Chain", "[bold green]enabled[/]" if ch == "on" else "[bold red]disabled[/]")
    )
    rows = _format_cp_rows(rows)
    _panel(
        "⛓ Recurring Chain Preview",
        rows,
        kind="preview_cp",
    )
    print(json.dumps(task), end="")



# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
