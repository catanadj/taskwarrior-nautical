#!/usr/bin/env python3
"""
Taskwarrior Chain Analyzer — Pro Edition

What’s new:
- Integrates cp_anchor analysis (Pattern, Natural, First/Next anchors, projected future anchors).
- Fuzzy search across ALL chained tasks (any status), plus a Chain picker (Active vs Finished).
- Builds chains by walking prevLink; shows a finished-chain brief (totals, span, cadence, anchor adherence).
- Calendar now shows completed dates (green) AND upcoming link dates (blue) and pending due dates (yellow).
- Graceful fallback if nautical_core.py isn't available: cp analysis still works; anchors show a helpful note.

Original base (ref): Enhanced Taskwarrior Chain Analyzer.  # Cited in chat
"""

from __future__ import annotations

import json
import sys
import subprocess
import datetime
import argparse
import calendar
import statistics
import os
import importlib
import importlib.machinery
import importlib.util
import shutil
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from dateutil import parser as date_parser, tz
from datetime import date, timedelta
from collections import defaultdict, deque

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.align import Align
from rich import box

from prompt_toolkit import prompt
from prompt_toolkit.completion import FuzzyCompleter, WordCompleter
try:
    import asciichartpy as asciichart  
except Exception:
    asciichart = None



MAX_CAL_FUTURE_STEPS = 2000  # safety cap

def _first_of_month(d: date) -> date:
    return date(d.year, d.month, 1)

def _add_months(d: date, n: int) -> date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return date(y, m, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Constants / styling
# ──────────────────────────────────────────────────────────────────────────────
console = Console()
UTC_ZONE = tz.tzutc()
LOCAL_ZONE = tz.tzlocal()

COLORS = {
    'primary': 'bright_cyan',
    'secondary': 'bright_blue',
    'success': 'green',
    'warning': 'bright_yellow',
    'error': 'bright_red',
    'muted': 'grey58',
    'accent': 'bright_magenta',
    'future': 'bright_blue',
    'pending': 'yellow',
}

FIELD_COLORS = {
    'status': 'bright_green',
    'priority': 'bright_yellow',
    'due': 'bright_red',
    'wait': 'blue',
    'scheduled': 'green',
    'description': 'bright_cyan',
    'project': 'magenta',
    'tags': 'bright_blue',
    'depends': 'yellow',
    'start': 'green',
    'end': 'bright_green',
    'annotations': 'bright_magenta',
    'area': 'cyan',
    'cp': 'bright_white',
    'duration': 'bright_white'
}

CHANGE_ICONS = {
    'added': '✨',
    'removed': '🗑️',
    'modified': '📝',
    'completed': '✅',
    'started': '🚀',
    'waiting': '⏳',
    'deleted': '❌'
}

# Calendar preview lengths
UPCOMING_MAX = 12  # project up to 12 next occurrences

# Try to import nautical_core (used by hooks); analyzer runs even if missing
def _find_nautical_core() -> Optional[Any]:
    # Search common locations (same approach as hooks)
    candidates: List[Path] = []
    def _add(p):
        if not p: return
        p = Path(p).expanduser()
        if p not in candidates:
            candidates.append(p)
    hook_dir = Path(__file__).resolve().parent
    _add(hook_dir)                     # alongside this script
    _add(hook_dir.parent)              # ~/.task
    if os.environ.get("TASKDATA"):
        _add(os.environ["TASKDATA"])
    if os.environ.get("TASKRC"):
        _add(Path(os.environ["TASKRC"]).parent)
    _add(Path.home() / ".task")

    for base in candidates:
        pyfile = base / "nautical_core.py"
        pkgini = base / "nautical_core" / "__init__.py"
        if pyfile.is_file() or pkgini.is_file():
            sys.path.insert(0, str(base))
            try:
                return importlib.import_module("nautical_core")
            except Exception:
                pass
    return None

core = _find_nautical_core()

# ──────────────────────────────────────────────────────────────────────────────
# Navigator helpers (explain/validate/self-check)
# ──────────────────────────────────────────────────────────────────────────────
def _diag_enabled() -> bool:
    return os.environ.get("NAUTICAL_DIAG") == "1"

def _emit_check(status: str, label: str, detail: str) -> None:
    color = {
        "OK": COLORS["success"],
        "WARN": COLORS["warning"],
        "FAIL": COLORS["error"],
    }.get(status, COLORS["muted"])
    console.print(f"[{color}]{status:>4}[/] {label}: {detail}")

def _print_diag_report(hook_dir: Path | None) -> None:
    console.print("\n[bold]Diagnostics[/bold]")
    console.print(f"nautical_core={getattr(core, '__file__', 'unknown')}")
    if hook_dir:
        console.print(f"hook_dir={hook_dir}")
    for k in (
        "TASKRC",
        "TASKDATA",
        "NAUTICAL_DNF_DISK_CACHE",
        "NAUTICAL_PROFILE",
    ):
        v = os.environ.get(k)
        if v is not None:
            console.print(f"env.{k}={v}")
    try:
        console.print(f"core.enable_anchor_cache={getattr(core, 'ENABLE_ANCHOR_CACHE', None)}")
        console.print(f"core.anchor_cache_dir={getattr(core, 'ANCHOR_CACHE_DIR_OVERRIDE', '')}")
        console.print(f"core.anchor_cache_ttl={getattr(core, 'ANCHOR_CACHE_TTL', None)}")
    except Exception:
        pass

def _load_hook_module(path: Path, name: str) -> tuple[bool, str]:
    try:
        loader = importlib.machinery.SourceFileLoader(name, str(path))
        spec = importlib.util.spec_from_loader(name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return True, "import ok"
    except Exception as e:
        return False, str(e)

def _find_hook_dir() -> Path | None:
    candidates: list[Path] = []
    def _add(p):
        if not p:
            return
        p = Path(p).expanduser()
        if p not in candidates:
            candidates.append(p)
    script_dir = Path(__file__).resolve().parent
    _add(script_dir.parent)
    if os.environ.get("TASKDATA"):
        _add(os.environ["TASKDATA"])
    if os.environ.get("TASKRC"):
        _add(Path(os.environ["TASKRC"]).parent)
    _add(Path.home() / ".task")
    for base in candidates:
        on_add = base / "on-add-nautical.py"
        on_mod = base / "on-modify-nautical.py"
        if on_add.exists() or on_mod.exists():
            return base
    return None

def _self_check() -> int:
    console.print("[bold]Nautical self-check[/bold]")
    ok = True

    task_bin = shutil.which("task")
    if task_bin:
        _emit_check("OK", "task", task_bin)
    else:
        ok = False
        _emit_check("FAIL", "task", "not found in PATH")

    cfg_paths = []
    if hasattr(core, "_config_paths"):
        try:
            cfg_paths = core._config_paths()
        except Exception:
            cfg_paths = []
    cfg_existing = [p for p in cfg_paths if os.path.exists(p)]
    if cfg_existing:
        cfg_path = cfg_existing[0]
        detail = f"found {cfg_path}"
        if hasattr(core, "_read_toml"):
            try:
                data = core._read_toml(cfg_path)
                if not data:
                    detail += " (empty or parse error)"
                    _emit_check("WARN", "config", detail)
                else:
                    _emit_check("OK", "config", detail)
            except Exception as e:
                _emit_check("WARN", "config", f"{cfg_path} (parse error: {e})")
        else:
            _emit_check("OK", "config", detail)
    else:
        _emit_check("WARN", "config", "no config file found; defaults in use")

    hook_dir = _find_hook_dir()
    if hook_dir:
        on_add = hook_dir / "on-add-nautical.py"
        on_mod = hook_dir / "on-modify-nautical.py"
        if on_add.exists():
            ok_add, msg = _load_hook_module(on_add, "_nautical_on_add_check")
            _emit_check("OK" if ok_add else "FAIL", "on-add hook", msg)
            ok = ok and ok_add
        else:
            ok = False
            _emit_check("FAIL", "on-add hook", f"missing: {on_add}")

        if on_mod.exists():
            ok_mod, msg = _load_hook_module(on_mod, "_nautical_on_mod_check")
            _emit_check("OK" if ok_mod else "FAIL", "on-modify hook", msg)
            ok = ok and ok_mod
        else:
            ok = False
            _emit_check("FAIL", "on-modify hook", f"missing: {on_mod}")
    else:
        ok = False
        _emit_check("FAIL", "hooks", "could not locate hook directory")

    dnf_cache_enabled = (os.environ.get("NAUTICAL_DNF_DISK_CACHE") or "1").strip().lower() in ("1", "true", "yes", "on")
    if hook_dir:
        cache_path = hook_dir / ".nautical_cache" / "dnf_cache.jsonl"
        if dnf_cache_enabled:
            if cache_path.exists():
                try:
                    size = cache_path.stat().st_size
                except Exception:
                    size = 0
                valid = 0
                total = 0
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            total += 1
                            try:
                                obj = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(obj, dict) and "version" in obj:
                                continue
                            if isinstance(obj, dict) and ("key" in obj or "k" in obj):
                                valid += 1
                except Exception as e:
                    _emit_check("WARN", "dnf cache", f"read error: {e}")
                if total == 0:
                    _emit_check("WARN", "dnf cache", f"empty or unreadable ({cache_path})")
                elif valid == 0:
                    _emit_check("WARN", "dnf cache", f"no valid records ({cache_path})")
                else:
                    _emit_check("OK", "dnf cache", f"{valid} entries, {size} bytes")
            else:
                _emit_check("WARN", "dnf cache", f"enabled but missing: {cache_path}")
        else:
            _emit_check("OK", "dnf cache", "disabled")

    if getattr(core, "ENABLE_ANCHOR_CACHE", False):
        _emit_check("OK", "core anchor cache", "enabled (config)")
    else:
        _emit_check("WARN", "core anchor cache", "disabled (config)")

    if _diag_enabled():
        _print_diag_report(hook_dir)

    return 0 if ok else 1

def _anchor_preview(expr: str, count: int = 5) -> tuple[str, list[str]]:
    natural = ""
    next_dates: list[str] = []
    dnf = None
    default_seed = None
    try:
        dnf = core.validate_anchor_expr_strict(expr)
        default_seed = core.to_local(core.now_utc()).date()
    except Exception:
        dnf = None

    if not natural and hasattr(core, "describe_anchor_expr"):
        try:
            natural = core.describe_anchor_expr(expr)
        except Exception:
            natural = ""

    if dnf:
        after_date = core.to_local(core.now_utc()).date()
        seed = after_date
        for _ in range(count):
            nxt, _meta = core.next_after_expr(dnf, after_date, default_seed=seed, seed_base="preview")
            if not nxt:
                break
            hhmm = core.pick_hhmm_from_dnf_for_date(dnf, nxt, seed)
                if hhmm:
                    dt_utc = core.build_local_datetime(nxt, hhmm)
                    next_dates.append(core.fmt_dt_local(dt_utc))
            else:
                next_dates.append(str(nxt))
            after_date = nxt

    return natural, next_dates[:count]

def _anchor_explain(expr: str) -> int:
    try:
        core.validate_anchor_expr_strict(expr)
    except Exception as e:
        console.print(f"[{COLORS['error']}]Invalid anchor:[/] {e}")
        return 1

    natural, next_dates = _anchor_preview(expr, count=5)
    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_row("Expression", expr)
    table.add_row("Natural", natural or "—")
    if next_dates:
        nxt = "\n".join([f"{i}. {d}" for i, d in enumerate(next_dates, 1)])
        table.add_row("Next", nxt)
    console.print(Panel(table, title="Anchor explain", border_style=COLORS["secondary"], expand=False))
    return 0

def _taskdata_dir() -> Path:
    td = os.environ.get("TASKDATA")
    if td:
        return Path(td).expanduser()
    return Path.home() / ".task"

def _is_valid_spawn_entry(entry: dict) -> bool:
    if not isinstance(entry, dict):
        return False
    spawn_intent_id = (entry.get("spawn_intent_id") or "").strip()
    if not spawn_intent_id:
        return False
    child = entry.get("child")
    if not isinstance(child, dict):
        return False
    child_uuid = (child.get("uuid") or "").strip()
    if not child_uuid:
        return False
    return True

def _recover_dead_letter(*, dry_run: bool, prune: bool, limit: int | None) -> int:
    td = _taskdata_dir()
    dead_path = td / ".nautical_dead_letter.jsonl"
    queue_path = td / ".nautical_spawn_queue.jsonl"
    lock_dead = td / ".nautical_dead_letter.lock"
    lock_queue = td / ".nautical_spawn_queue.lock"

    if not dead_path.exists():
        console.print(f"[{COLORS['warning']}]No dead-letter file found at {dead_path}[/]")
        return 0

    lock_fn = getattr(core, "safe_lock", None) if core else None
    if lock_fn is None:
        def _noop_lock(_path, **_kwargs):
            from contextlib import contextmanager
            @contextmanager
            def _cm():
                yield True
            return _cm()
        lock_fn = _noop_lock

    lines: list[str] = []
    recovered_entries: list[dict] = []
    recovered_line_indexes: list[int] = []
    skipped = 0
    invalid = 0

    with lock_fn(lock_dead, retries=3, sleep_base=0.05, jitter=0.05, mkdir=True, stale_after=30.0) as ok:
        if not ok:
            console.print(f"[{COLORS['warning']}]Dead-letter lock busy; try again later.[/]")
            return 1
        try:
            lines = [ln.rstrip("\n") for ln in dead_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception as e:
            console.print(f"[{COLORS['error']}]Failed to read dead-letter file: {e}[/]")
            return 1

    for idx, line in enumerate(lines):
        if limit is not None and len(recovered_entries) >= limit:
            break
        try:
            obj = json.loads(line)
        except Exception:
            skipped += 1
            continue
        entry = obj.get("entry") if isinstance(obj, dict) else None
        if isinstance(entry, dict) and _is_valid_spawn_entry(entry):
            recovered_entries.append(entry)
            recovered_line_indexes.append(idx)
        else:
            invalid += 1

    if dry_run:
        console.print(
            f"[{COLORS['secondary']}]Recoverable entries: {len(recovered_entries)} "
            f"(invalid: {invalid}, skipped: {skipped})[/]"
        )
        return 0

    if recovered_entries:
        with lock_fn(lock_queue, retries=3, sleep_base=0.05, jitter=0.05, mkdir=True, stale_after=30.0) as ok:
            if not ok:
                console.print(f"[{COLORS['warning']}]Queue lock busy; try again later.[/]")
                return 1
            try:
                queue_path.parent.mkdir(parents=True, exist_ok=True)
                with open(queue_path, "a", encoding="utf-8") as f:
                    for entry in recovered_entries:
                        f.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            except Exception as e:
                console.print(f"[{COLORS['error']}]Failed to append to spawn queue: {e}[/]")
                return 1

    if prune and recovered_line_indexes:
        recovered_set = set(recovered_line_indexes)
        with lock_fn(lock_dead, retries=3, sleep_base=0.05, jitter=0.05, mkdir=True, stale_after=30.0) as ok:
            if not ok:
                console.print(f"[{COLORS['warning']}]Dead-letter lock busy; prune skipped.[/]")
            else:
                try:
                    current = [ln for ln in dead_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    kept = [ln for i, ln in enumerate(current) if i not in recovered_set]
                    dead_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
                except Exception as e:
                    console.print(f"[{COLORS['warning']}]Failed to prune dead-letter file: {e}[/]")

    console.print(
        f"[{COLORS['success']}]Recovered {len(recovered_entries)} entries[/]"
        + (f" (invalid: {invalid})" if invalid else "")
        + (f" (skipped: {skipped})" if skipped else "")
    )
    return 0

def _validate_anchor(expr: str) -> int:
    try:
        core.validate_anchor_expr_strict(expr)
        console.print(f"[{COLORS['success']}]OK[/] anchor: valid")
        return 0
    except Exception as e:
        console.print(f"[{COLORS['error']}]FAIL[/] anchor: {e}")
        return 1

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TaskChange:
    field: str
    change_type: str
    from_value: Any = None
    to_value: Any = None
    added_items: List = None
    removed_items: List = None
    delta: Optional[str] = None
    group: int = 99


# ──────────────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────────────
class TaskAnalyzer:
    """Main analyzer with anchor/cp introspection and chain discovery."""

    def __init__(self):
        self._task_cache: Dict[int, Dict] = {}
        self._uuid_cache: Dict[str, Dict] = {}
        self._children: Dict[str, List[Dict]] = {}  # prev_uuid -> [children]

    # ── Task retrieval ───────────────────────────────────────────────────────
    @lru_cache(maxsize=1)
    def get_all_chained_tasks(self) -> List[Dict]:
        """Fetch all chained tasks (any status)."""
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            spinner = progress.add_task("Loading chained tasks…", total=None)
            try:
                # includes completed/waiting/pending/etc.
                result = subprocess.run(["task", "chain:on", "export", "all"], capture_output=True, text=True, check=True)
                tasks = json.loads(result.stdout)

                for t in tasks:
                    if t.get("id"):
                        self._task_cache[int(t["id"])] = t
                    if t.get("uuid"):
                        self._uuid_cache[t["uuid"]] = t
                    prev = t.get("prevLink")
                    if prev:
                        self._children.setdefault(prev, []).append(t)

                progress.update(spinner, description=f"✅ Loaded {len(tasks)} chained tasks")
                return tasks

            except subprocess.CalledProcessError as e:
                console.print(f"[{COLORS['error']}]Error: Failed to retrieve tasks (exit {e.returncode})[/]")
                if e.stderr:
                    console.print(f"[{COLORS['muted']}]{e.stderr.strip()}[/]")
                sys.exit(1)
            except json.JSONDecodeError as e:
                console.print(f"[{COLORS['error']}]JSON parse error: {e}[/]")
                sys.exit(1)

    # ── Chain discovery ──────────────────────────────────────────────────────
    def _is_root(self, task: Dict) -> bool:
        """A root has no prevLink (or prev not present in our uuid set)."""
        prev = task.get("prevLink")
        return not prev or prev not in self._uuid_cache

    def build_all_chains(self) -> List[List[Dict]]:
        """
        Build chains strictly from real prev->curr edges across ALL tasks.
        No grouping by link or root. Each task appears in at most one chain.
        At forks: the earliest child stays on the main path; others start new chains.
        """
        self._build_indexes()
        by_uuid, children, indeg = self._build_global_graph()
        if not by_uuid:
            return []

        visited = set()
        chains: List[List[Dict]] = []

        # Start from heads (in-degree 0), oldest first
        heads = [by_uuid[u] for u, d in indeg.items() if d == 0 and u in by_uuid]
        heads.sort(key=self._entry_key)

        pending = heads[:]  # queue of starting points for new chains

        def walk_from(start_task: Dict):
            chain = []
            cur = start_task
            while cur and cur.get("uuid") not in visited:
                chain.append(cur)
                visited.add(cur.get("uuid"))
                kids = sorted(children.get(cur.get("uuid"), []), key=self._entry_key)
                if not kids:
                    break
                # enqueue secondary branches (without duplicating nodes)
                for k in kids[1:]:
                    if k.get("uuid") not in visited:
                        pending.append(k)
                # continue along earliest child
                nxt = kids[0]
                if nxt.get("uuid") in visited:
                    break
                cur = nxt
            return chain

        # Walk main heads
        while pending:
            start = pending.pop(0)
            if start.get("uuid") in visited:
                continue
            chains.append(walk_from(start))

        # Cover any remaining nodes not reachable from a head (cycles/breaks)
        leftovers = [t for u, t in by_uuid.items() if u not in visited]
        leftovers.sort(key=self._entry_key)
        for t in leftovers:
            if t.get("uuid") in visited:
                continue
            chains.append(walk_from(t))

        # Ensure each chain is oldest→newest by entry
        chains = [sorted(ch, key=self._entry_key) for ch in chains]

        # Active chains first (pending/waiting), then finished; stable by tail desc
        def is_active(ch): return ch[-1].get("status") in ("pending", "waiting")
        def tail_desc(ch): return (ch[-1].get("description") or "").lower()

        chains.sort(key=lambda ch: (0 if is_active(ch) else 1, tail_desc(ch)))
        return chains

    def compute_scheduled_days(chain: List[dict]) -> Set[date]:
        """
        Generate scheduled occurrences (as LOCAL dates) for the chain, covering a broad
        window so rendering can choose up to 12 months cleanly.

        Rules:
        - If anchor is present → use your strict parser + next_after_expr
        - Else, if classic cp → step by cp duration
        - Enforce chainUntil (hard stop for future dates)
        - Enforce chainMax only on FUTURE occurrences (left from current link)
        """
        if not chain:
            return set()
        last = chain[-1]

        # --- basic parsed fields
        anchor_expr = (last.get("cp_anchor") or "").strip()
        cp_period   = (last.get("cp") or "").strip()
        cpmax       = core.coerce_int(last.get("chainMax"), 0) or 0
        cur_link    = core.coerce_int(last.get("link"), 1) or 1
        left        = max(cpmax - cur_link, 0) if cpmax else None
        until_utc   = core.parse_dt_any(last.get("chainUntil")) if last.get("chainUntil") else None
        now_local   = core.to_local(core.now_utc()).date()

        # --- completion window seed (so we cover the past where completions exist)
        comp_days: Set[date] = set()
        for t in chain:
            e = t.get("end")
            if e:
                edt = core.parse_dt_any(e)
                if edt:
                    comp_days.add(core.to_local(edt).date())

        # pick a broad window: 6 months before earliest completion (or today) → up to 18 months ahead
        if comp_days:
            window_start = _first_of_month(min(comp_days))  # earliest completion month
        else:
            window_start = _first_of_month(now_local)
        window_start = _add_months(window_start, -6)
        window_end   = _add_months(window_start, 24)  # generous; rendering will cap to 12 months

        scheduled: Set[date] = set()

        if anchor_expr:
            # ---------- ANCHOR ----------
            try:
                dnf = core.validate_anchor_expr_strict(anchor_expr)
            except Exception:
                return scheduled

            # choose seed: prefer 'due'; else compute next from window_start
            due_dt = core.parse_dt_any(last.get("due")) if last.get("due") else None
            if due_dt:
                seed_day = core.to_local(due_dt).date()
                clock = (core.to_local(due_dt).hour, core.to_local(due_dt).minute)
            else:
                seed_day, hhmm = core.next_after_expr(dnf, window_start, default_seed=window_start)
                clock = hhmm or (core.DEFAULT_DUE_HOUR, 0)

            # walk forward until window_end / chainUntil / chainMax (future)
            cur = seed_day
            prev = None
            future_taken = 0
            for _ in range(MAX_CAL_FUTURE_STEPS):
                nxt, _ = core.next_after_expr(dnf, cur, default_seed=seed_day)
                if prev is not None and nxt <= prev:
                    cur = prev + timedelta(days=1); continue
                prev = nxt
                # hard stop by window_end
                if nxt > window_end:
                    break
                # chainUntil check
                if until_utc:
                    dt_utc = core.build_local_datetime(nxt, clock)
                    if dt_utc > until_utc:
                        break
                # chainMax enforcement only for future occurrences
                if left is not None and nxt >= now_local:
                    if future_taken >= left:
                        break
                    future_taken += 1

                if nxt >= window_start:
                    scheduled.add(nxt)
                cur = nxt

        elif cp_period:
            # ---------- CLASSIC CP ----------
            td = core.parse_cp_duration(cp_period)
            if not td:
                return scheduled

            # start from earliest known 'due' in the chain, else last 'due', else now
            dues = []
            for t in chain:
                if t.get("due"):
                    dt = core.parse_dt_any(t["due"])
                    if dt:
                        dues.append(dt)
            base = min(dues) if dues else (core.parse_dt_any(last.get("due")) or core.now_utc())
            base = base.replace(microsecond=0)

            # step forward
            cur_dt = base
            future_taken = 0
            for _ in range(MAX_CAL_FUTURE_STEPS):
                cur_dt = cur_dt + td
                if until_utc and cur_dt > until_utc:
                    break
                dloc = core.to_local(cur_dt).date()
                if dloc > window_end:
                    break
                if dloc >= window_start:
                    # chainMax only for future
                    if left is not None and dloc >= now_local:
                        if future_taken >= left:
                            break
                        future_taken += 1
                    scheduled.add(dloc)
        return scheduled


    def render_status_calendar(done_days: Set[date], scheduled_days: Set[date]) -> None:
        """
        Renders up to 12 months, 3 per row, coloring:
        - DONE (blue bg, white fg)        : days in done_days
        - MISSED (red bg, white fg)       : scheduled < today AND not in done_days
        - FUTURE (pink bg, black fg)      : scheduled >= today AND not in done_days
        - Other in-month days (dim grey)  : light numbers
        - Out-of-month days               : plain spaces

        Span:
        - natural: first completion month → last scheduled month
        - if span > 12 months, center a 12-month window around *today*.
        """
        from rich.panel import Panel
        from rich.text import Text
        from rich import box

        if not done_days and not scheduled_days:
            console.print(Panel(Text("No completions or scheduled dates.", style="bright_black"),
                                title="Calendar", border_style="bright_black"))
            return

        today = core.to_local(core.now_utc()).date()

        # ----- choose span
        anchors = []
        if done_days:
            anchors.append(min(done_days))
        if scheduled_days:
            anchors.append(min(scheduled_days))
        earliest = min(anchors)
        latest   = max([max(done_days) if done_days else earliest,
                        max(scheduled_days) if scheduled_days else earliest])

        start_month = _first_of_month(earliest)
        end_month   = _first_of_month(latest)

        # Build full sequence and cap / center on today if > 12 months
        months: List[Tuple[int,int]] = []
        y, m = start_month.year, start_month.month
        total = 0
        while True:
            months.append((y, m))
            if (y, m) == (end_month.year, end_month.month):
                break
            m += 1
            if m == 13:
                m = 1; y += 1
            total += 1
            if total > 240:  # hard safety
                break

        if len(months) > 12:
            tgt = (today.year, today.month)
            # find closest index to today
            idx = 0
            for i, ym in enumerate(months):
                if ym == tgt:
                    idx = i; break
                if ym > tgt:
                    idx = max(0, i - 1); break
            start_idx = max(0, idx - 6)
            end_idx   = min(len(months), start_idx + 12)
            if end_idx - start_idx < 12:
                start_idx = max(0, end_idx - 12)
            months = months[start_idx:end_idx]

        # ----- cell painter
        BLUE_BG   = "\x1b[44m\x1b[97m"
        RED_BG    = "\x1b[41m\x1b[97m"
        PINK_BG   = "\x1b[45m\x1b[30m"   # pinkish (magenta) with black text
        DIM_FG    = "\x1b[90m"
        RESET     = "\x1b[0m"

        def cell_color(d: date) -> str:
            if d in done_days:
                return f"{BLUE_BG}{d.day:2d}{RESET}"
            if d in scheduled_days:
                if d < today:
                    return f"{RED_BG}{d.day:2d}{RESET}"     # missed
                else:
                    return f"{PINK_BG}{d.day:2d}{RESET}"    # future
            # non-scheduled in-month
            return f"{DIM_FG}{d.day:2d}{RESET}"

        # ----- build month blocks (8 fixed lines each)
        cal = calendar.Calendar(firstweekday=calendar.MONDAY)

        def month_block(y: int, m: int) -> List[str]:
            lines: List[str] = []
            title = f"{calendar.month_name[m]} {y}".center(20)
            lines.append(title)
            lines.append("Mo Tu We Th Fr Sa Su")
            weeks = cal.monthdayscalendar(y, m)
            while len(weeks) < 6:
                weeks.append([0,0,0,0,0,0,0])
            for wk in weeks:
                parts = []
                for d in wk:
                    if d == 0:
                        parts.append("  ")
                    else:
                        parts.append(cell_color(date(y, m, d)))
                lines.append(" ".join(parts))
            return lines  # 8 lines, width ≈ 20 (ANSI has zero width)

        # ----- stitch 3 per row
        rows: List[str] = []
        for i in range(0, len(months), 3):
            trio = months[i:i+3]
            blocks = [month_block(yy, mm) for (yy, mm) in trio]
            while len(blocks) < 3:
                blocks.append([" " * 20] * 8)
            for r in range(8):
                rows.append(f"{blocks[0][r]}  {blocks[1][r]}  {blocks[2][r]}")
            if i + 3 < len(months):
                rows.append("")  # blank line between groups

        # legend
        legend = (
            f"{BLUE_BG}  {RESET} Done   "
            f"{RED_BG}  {RESET} Missed   "
            f"{PINK_BG}  {RESET} Upcoming"
        )
        block = legend + "\n" + "\n".join(rows)

        console.print(
            Panel(
                Text.from_ansi(block),
                title="Calendar",
                border_style="bright_black",
                box=box.SIMPLE,
                padding=(0,1),
            ),
            no_wrap=True  # <- keep columns intact
        )


    def _build_indexes(self) -> None:
        """Index all chained tasks by full and short UUIDs."""
        _ = self.get_all_chained_tasks()
        self._uuid_by_short = {}
        for u in list(self._uuid_cache.keys()):
            self._uuid_by_short[u[:8]] = self._uuid_by_short.get(u[:8], u)

    def _render_line_plot(self, x_labels: List[str], series: Dict[str, List[Optional[float]]], height: int = 10, width: int = 60) -> Panel:
        """
        Render a simple terminal line chart:
        - Y values are floats (e.g., hours of Δ vs CP)
        - Missing points (None) create gaps
        - Baseline (0) is drawn as a horizontal axis
        """
        from rich.text import Text

        # Determine scale (symmetric about 0, with a small margin)
        all_vals = []
        for vals in series.values():
            all_vals += [abs(v) for v in vals if v is not None]
        ymax = max(all_vals) if all_vals else 1.0
        ymin, ymax = -ymax, ymax
        if ymax == 0:
            ymax = 1.0
            ymin = -1.0

        # Sanity sizes
        height = max(6, height)
        # Make width at least number of points (so every point maps to a column)
        width = max(len(x_labels), width)

        # X mapping: 0..n-1 spread across width-1
        n = len(x_labels)
        if n < 2:
            n = 2
        x_map = [round(i * (width - 1) / (n - 1)) for i in range(len(x_labels))]

        # Prepare canvas (rows of chars)
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Convert y value to row index (0 at top)
        def y_to_row(y: float) -> int:
            # row 0 => ymax ; row height-1 => ymin
            pos = 0 if ymax == ymin else (ymax - y) / (ymax - ymin)
            return max(0, min(height - 1, int(round(pos * (height - 1)))))

        # Baseline (y=0)
        zero_row = y_to_row(0.0)
        for x in range(width):
            grid[zero_row][x] = "─"

        # Colors for series
        s_colors = {
            "Due Δ": "cyan",
            "End Δ": "magenta",
        }

        # Plot each series as a polyline of '•' with simple connectors '─'
        for name, vals in series.items():
            color = s_colors.get(name, "white")
            prev = None  # (x_col, y_row)
            for idx, v in enumerate(vals):
                if v is None:
                    prev = None
                    continue
                x_col = x_map[idx]
                y_row = y_to_row(float(v))

                # point
                grid[y_row][x_col] = "•"

                # connector from prev horizontally (only if same row and no gap)
                if prev is not None:
                    px, py = prev
                    if py == y_row and x_col > px + 1:
                        for x in range(px + 1, x_col):
                            if grid[py][x] == " ":
                                grid[py][x] = "─"
                prev = (x_col, y_row)

            # Colorize this series' glyphs
            for r in range(height):
                for c in range(width):
                    ch = grid[r][c]
                    if ch in ("•", "─"):  # baseline is also '─', but we recolor later
                        pass  # we colorize at print time

        # Build Rich Text lines with color
        lines = []
        for r in range(height):
            t = Text()
            for c in range(width):
                ch = grid[r][c]
                # baseline color:
                if r == zero_row and ch == "─":
                    t.append(ch, style="grey50")
                # Due Δ point/segment:
                elif ch == "•":
                    # We can't track per-pixel ownership without more bookkeeping;
                    # color the point based on proximity to the nearest non-None series point.
                    t.append(ch, style=s_colors.get("Due Δ"))
                elif ch == "─":
                    # leave neutral unless it's baseline; series horizontal segments will be colored by legend context
                    t.append(ch, style="white")
                else:
                    t.append(ch)
            lines.append(t)

        # X labels (compact: show first, middle, last)
        xlab = Text()
        if x_labels:
            lbls = [" " * width]
            lines.append(Text(""))  # spacer
            # show index labels (first / mid / last)
            marks = [(0, x_labels[0])]
            if len(x_labels) > 2:
                marks.append((len(x_labels)//2, x_labels[len(x_labels)//2]))
            marks.append((len(x_labels)-1, x_labels[-1]))
            # draw underneath with crude placement
            label_row = [" " for _ in range(width)]
            for i, s in marks:
                col = x_map[i]
                s = str(s)
                start = max(0, min(width-1, col - len(s)//2))
                for j, ch in enumerate(s):
                    pos = start + j
                    if pos < width:
                        label_row[pos] = ch
            lines.append(Text("".join(label_row), style="grey70"))

        # Legend
        legend = Table.grid(padding=(0,2))
        legend.add_column(style="white"); legend.add_column(style="white")
        legend.add_row(Text("Baseline", style="grey50"), "0 (on schedule)")
        legend.add_row(Text("Due Δ", style="cyan"), "early (−) / late (+)")
        legend.add_row(Text("End Δ", style="magenta"), "early (−) / late (+)")

        from rich.console import Group
        return Panel(Align.left(Group(*lines)),
                    title="⏱️ CP Performance vs Baseline", border_style=COLORS["secondary"], padding=(1,1))



    def _build_global_graph(self) -> tuple[dict, dict, dict]:
        """
        Build a single graph over ALL chained tasks:
        by_uuid: uuid -> task
        children: parent_uuid -> [child tasks]
        indeg: uuid -> in-degree
        Only add edges where child's prevLink resolves to a parent present in by_uuid.
        """
        tasks = self.get_all_chained_tasks()
        by_uuid = {t.get("uuid"): t for t in tasks if t.get("uuid")}
        children = defaultdict(list)
        indeg = defaultdict(int)

        for t in tasks:
            u = t.get("uuid")
            if u:
                indeg.setdefault(u, 0)
            p = self._resolve_prev_uuid(t.get("prevLink"))
            if p and p in by_uuid and u:
                children[p].append(t)
                indeg[u] = indeg.get(u, 0) + 1

        # keep for finished-chain detection
        self._children = children
        return by_uuid, children, indeg

    def _resolve_prev_uuid(self, prev: str | None) -> str | None:
        """Resolve prevLink (full or short) to a full UUID present in our cache."""
        if not prev:
            return None
        if prev in self._uuid_cache:
            return prev
        key = prev[:8]
        full = self._uuid_by_short.get(key)
        return full if full and full in self._uuid_cache else None

    def _entry_key(self, t: Dict) -> tuple:
        """Sort key: entry then uuid for stability."""
        return (t.get("entry") or "", t.get("uuid") or "")

    def _ultimate_root_uuid(self, t: Dict) -> str | None:
        """Walk prevLink (with prefix resolution) to find the root uuid."""
        seen = set()
        cur = t
        while True:
            pu = self._resolve_prev_uuid(cur.get("prevLink"))
            if not pu or pu in seen or pu not in self._uuid_cache:
                break
            seen.add(pu)
            cur = self._uuid_cache[pu]
        return cur.get("uuid")

    def _chain_key(self, t: Dict) -> str:
        """
        Stable chain identifier:
        1) If a link UDA exists, use it: 'link' or 'chainedLink'
        2) Else use the ultimate root uuid (via prevLink, prefix-safe)
        3) Else fall back to this task's uuid (singleton)
        """
        link = t.get("link") or t.get("chainedLink")
        if link:
            return str(link)
        root = self._ultimate_root_uuid(t)
        return root or t.get("uuid") or f"singleton:{id(t)}"

    def build_chain_from_tasks(self, start_id: int) -> List[Dict]:
        """Backtrack using prevLink; return chain oldest→newest."""
        _ = self.get_all_chained_tasks()
        if start_id not in self._task_cache:
            console.print(f"[{COLORS['error']}]Task ID {start_id} not found in chained tasks[/]")
            sys.exit(1)

        chain: List[Dict] = []
        current = self._task_cache[start_id]
        # walk back to root
        while current:
            chain.append(current)
            prev_uuid = current.get("prevLink")
            if not prev_uuid:
                break
            nxt = self._uuid_cache.get(prev_uuid)
            if not nxt:
                # prefix match fallback
                for u, t in self._uuid_cache.items():
                    if u.startswith(prev_uuid):
                        nxt = t
                        break
            if not nxt:
                break
            current = nxt
        chain = list(reversed(chain))

        # then try walking forward deterministically like build_all_chains does
        tail = chain[-1]
        while True:
            kids = self._children.get(tail.get("uuid"), [])
            if not kids:
                break
            kids_sorted = sorted(kids, key=lambda k: (k.get("entry") or "", k.get("uuid") or ""))
            nxt = kids_sorted[0]
            if nxt in chain:
                break
            chain.append(nxt)
            tail = nxt

        return chain

    # -- Classic CP Performance vs Baseline ------------------------------------------------------------
    def _parse_cp_seconds(self, cp_str: str) -> Optional[int]:
        """Return CP period in seconds (prefer core.parse_cp_duration; fallback Nd/Nw/Nm/Ny/Nh)."""
        cp = (cp_str or "").strip()
        if not cp:
            return None
        try:
            if core and hasattr(core, "parse_cp_duration"):
                td = core.parse_cp_duration(cp)
                if td:
                    return int(td.total_seconds())
        except Exception:
            pass
        import re, datetime as _dt
        m = re.match(r"^\s*(\d+)\s*([dwmyh])\s*$", cp, re.I)
        if not m:
            return None
        n = int(m.group(1)); u = m.group(2).lower()
        mult = {'d':86400,'w':604800,'h':3600}.get(u)
        if u == 'm': mult = 30*86400
        if u == 'y': mult = 365*86400
        return n * mult if mult else None

    def _fmt_delta_short(self, seconds: int) -> str:
        if seconds == 0:
            return "±0m"
        sign = "+" if seconds > 0 else "-"
        s = abs(int(seconds))
        d, r = divmod(s, 86400)
        h, r = divmod(r, 3600)
        m, _ = divmod(r, 60)
        parts = []
        if d: parts.append(f"{d}d")
        if h: parts.append(f"{h}h")
        if not d and not h and m: parts.append(f"{m}m")
        if not parts: parts.append("0m")
        return sign + " ".join(parts)


    def _render_delta_bar(self, delta_s: int, max_abs_s: int, width: int = 40, early_color="bright_cyan", late_color="bright_red"):
        """Return a Rich Text bar centered at zero (left=early, right=late)."""
        from rich.text import Text
        if width < 10: width = 10
        half = width // 2
        zero = "│"  # center axis
        t = Text()
        # Avoid division by zero
        if max_abs_s <= 0:
            t.append(" " * half)
            t.append(zero, style="grey50")
            t.append(" " * half)
            return t
        scale = max_abs_s / half  # seconds per cell
        cells = min(half, int(round(abs(delta_s) / scale)))
        # left pad
        if delta_s < 0:
            t.append(" " * (half - cells))
            t.append("█" * cells, style=early_color)
            t.append(zero, style="grey50")
            t.append(" " * half)
        elif delta_s > 0:
            t.append(" " * half)
            t.append(zero, style="grey50")
            t.append("█" * cells, style=late_color)
            t.append(" " * (half - cells))
        else:
            t.append(" " * half)
            t.append(zero, style="grey50")
            t.append(" " * half)
        return t

    # ─────────────────────────────────────────────────────────────────────────────
    # asciichart renderer (with graceful fallback)
    # ─────────────────────────────────────────────────────────────────────────────
    def _legend_ansi_line(items: list[tuple[str, str]]) -> str:
        # items = [(label, ansi_color_code), ...]
        parts = []
        for label, ansi in items:
            parts.append(f"{ansi}──\x1b[0m {label}")
        return "  ".join(parts)

    def _fallback_dot_plot(series_list: list[list[float]], height: int = 12) -> str:
        """Very small multi-series dot plot (•), gaps allowed via None."""
        if not series_list:
            return ""
        n = max(len(s) for s in series_list)
        # pad to length n
        padded = []
        for s in series_list:
            s = list(s)
            if len(s) < n:
                s = s + [s[-1] if s and s[-1] is not None else 0.0] * (n - len(s))
            padded.append(s)
        vals = [v for s in padded for v in s if v is not None]
        lo, hi = (min(vals), max(vals)) if vals else (-1.0, 1.0)
        if hi == lo:
            hi, lo = hi + 1.0, lo - 1.0
        rows = []
        # map y -> row (0=top, height-1=bottom)
        def y2r(y: float) -> int:
            pos = (hi - y) / (hi - lo)
            return max(0, min(height - 1, int(round(pos * (height - 1)))))
        # precompute rows for each series
        bands = []
        for s in padded:
            bands.append([None if v is None else y2r(float(v)) for v in s])
        colors = ["\x1b[90m", "\x1b[36m", "\x1b[35m"]  # grey, cyan, magenta
        for r in range(height):
            line = []
            for x in range(n):
                ch = " "
                for si, b in enumerate(bands):
                    if b[x] is not None and b[x] == r:
                        ch = f"{colors[si % len(colors)]}•\x1b[0m"
                        break
                line.append(ch)
            rows.append("".join(line))
        return "\n".join(rows)

    def _render_asciichart_block(lines: dict[str, list[float]], height: int = 12, title: str = "CP Performance vs Baseline") -> Panel:
        """
        lines: {"Baseline": [...], "Due Δ": [...], "End Δ": [...]}
        All lists must be the same length. Values are floats (hours). Gaps allowed with None.
        """
        # Ensure same length and minimal presence
        labels = ["Baseline", "Due Δ", "End Δ"]
        data = [lines.get(k, []) for k in labels]
        if not data or not any(data):
            from rich.text import Text
            return Panel(Text("No data"), title=title, border_style="bright_black")

        # Replace Nones for asciichart if present; we’ll try real library first.
        # (asciichartpy doesn't support None; we’ll fallback if it errors)
        cleaned = []
        for series in data:
            if not series:
                cleaned.append([])
                continue
            cleaned.append([0.0 if v is None else float(v) for v in series])

        legend = _legend_ansi_line([
            ("Baseline", "\x1b[38;5;46m"),   # grey
            ("Due Δ",    "\x1b[36m"),   # cyan
            ("End Δ",    "\x1b[35m"),   # magenta
        ])

        plot_body = None
        if asciichart:
            try:
                ac_colors = [
                    getattr(asciichart, "lightgray", getattr(asciichart, "default", None)),
                    getattr(asciichart, "cyan", None),
                    getattr(asciichart, "magenta", None),
                ]
                cfg = {"height": height, "colors": ac_colors}
                plot_body = asciichart.plot(cleaned, cfg)
            except Exception:
                plot_body = None

        if plot_body is None:
            # fallback that keeps gaps if the real lib isn't available
            plot_body = _fallback_dot_plot([lines["Baseline"], lines["Due Δ"], lines["End Δ"]], height=height)

        from rich.text import Text
        from rich import box
        block = legend + "\n" + plot_body
        return Panel(Text.from_ansi(block), title=title, border_style="bright_black", box=box.SIMPLE, padding=(0,1))


    def _cp_performance_panel(self, chain: List[Dict]) -> None:
        """
        Render a compact 'CP Performance vs Baseline' panel using an ASCII line chart.
        - 3 lines:
            • Baseline (0)      → grey
            • Due Δ (hours)     → cyan
            • End  Δ (hours)    → magenta
        - Works without asciichartpy installed (uses a tiny dot-plot fallback).
        """
        # -----------------------------
        # 1) Build the time series
        # -----------------------------
        def parse_dt(s: Optional[str]) -> Optional[datetime.datetime]:
            if not s:
                return None
            try:
                return core.parse_dt_any(s)
            except Exception:
                return None

        def deltas_in_seconds(tasks: List[Dict], field: str) -> List[Optional[float]]:
            """
            For consecutive tasks, compute delta(current[field] - prev[field]) in seconds.
            If either side missing/unparseable, returns None for that slot.
            Output length = len(tasks)-1 (one value per link step).
            """
            out: List[Optional[float]] = []
            for i in range(1, len(tasks)):
                prev = parse_dt(tasks[i-1].get(field))
                cur  = parse_dt(tasks[i].get(field))
                if prev and cur:
                    out.append((cur - prev).total_seconds())
                else:
                    out.append(None)
            return out

        if not chain or len(chain) < 2:
            console.print(Panel(Text("Not enough data for performance chart.", style="bright_black"),
                                title="⏱️ CP Performance vs Baseline",
                                border_style="bright_black"))
            return

        # Compute seconds deltas between consecutive items
        due_series_sec = deltas_in_seconds(chain, "due")
        end_series_sec = deltas_in_seconds(chain, "end")

        # Convert to hours for plotting; keep None as gaps
        def to_hours(v: Optional[float]) -> Optional[float]:
            return (v / 3600.0) if v is not None else None

        due_hours = [to_hours(v) for v in due_series_sec]
        end_hours = [to_hours(v) for v in end_series_sec]

        # Baseline = 0 line with the same length as the longest series
        n = max(len(due_hours), len(end_hours))
        baseline = [0.0] * n

        # Pad shorter series with None so all three have equal length
        if len(due_hours) < n:
            due_hours = due_hours + [None] * (n - len(due_hours))
        if len(end_hours) < n:
            end_hours = end_hours + [None] * (n - len(end_hours))

        # -----------------------------
        # 2) Render with asciichart (or fallback)
        # -----------------------------
        # Try to import asciichartpy if available
        try:
            import asciichartpy as asciichart  # pip install asciichartpy
        except Exception:
            asciichart = None  # fallback below

        def legend_line(items: List[Tuple[str, str]]) -> str:
            # items: [(label, ansi_color)]
            parts = [f"{ansi}──\x1b[0m {label}" for label, ansi in items]
            return "  ".join(parts)

        def dot_plot_fallback(series_list: List[List[Optional[float]]], height: int = 12) -> str:
            """
            Simple multi-series dot plot (•). Keeps gaps where values are None.
            Ensures deterministic width (one column per point).
            """
            if not series_list:
                return ""
            n_cols = max(len(s) for s in series_list)
            # Pad to equal length
            padded: List[List[Optional[float]]] = []
            for s in series_list:
                s2 = list(s)
                if len(s2) < n_cols:
                    s2 = s2 + [None] * (n_cols - len(s2))
                padded.append(s2)

            # Find min/max ignoring None
            vals = [v for s in padded for v in s if v is not None]
            if not vals:
                vals = [0.0]
            lo, hi = min(vals), max(vals)
            if hi == lo:
                hi, lo = hi + 1.0, lo - 1.0

            # Map y->row (0 top, height-1 bottom)
            def y2r(y: float) -> int:
                pos = (hi - y) / (hi - lo)
                return max(0, min(height - 1, int(round(pos * (height - 1)))))

            bands: List[List[Optional[int]]] = []
            for s in padded:
                bands.append([None if v is None else y2r(float(v)) for v in s])

            # grey, cyan, magenta
            colors = ["\x1b[90m", "\x1b[36m", "\x1b[35m"]
            rows: List[str] = []
            for r in range(height):
                line = []
                for x in range(n_cols):
                    ch = " "
                    for si, b in enumerate(bands):
                        if b[x] is not None and b[x] == r:
                            ch = f"{colors[si % len(colors)]}•\x1b[0m"
                            break
                    line.append(ch)
                rows.append("".join(line))
            return "\n".join(rows)

        # Configure colors and legend
        legend = legend_line([
            ("Baseline", "\x1b[90m"),  # grey
            ("Due Δ",    "\x1b[36m"),  # cyan
            ("End Δ",    "\x1b[35m"),  # magenta
        ])

        height = 12
        plot_body: Optional[str] = None

        if asciichart:
            try:
                # asciichartpy doesn't support None → replace gaps with previous value (or 0)
                def fill_gaps(s: List[Optional[float]]) -> List[float]:
                    out: List[float] = []
                    last = 0.0
                    for v in s:
                        if v is None:
                            out.append(last)
                        else:
                            last = float(v)
                            out.append(last)
                    return out

                data_for_lib = [
                    fill_gaps(baseline),
                    fill_gaps(due_hours),
                    fill_gaps(end_hours),
                ]
                cfg = {
                    "height": height,
                    "colors": [
                        getattr(asciichart, "lightgray", getattr(asciichart, "default", None)),
                        getattr(asciichart, "cyan", None),
                        getattr(asciichart, "magenta", None),
                    ],
                }
                plot_body = asciichart.plot(data_for_lib, cfg)
            except Exception:
                plot_body = None

        if plot_body is None:
            plot_body = dot_plot_fallback([baseline, due_hours, end_hours], height=height)

        block = legend + "\n" + plot_body

        # -----------------------------
        # 3) Print panel + small stats
        # -----------------------------
        # Basic stats for context (ignore None)
        def stats(xs: List[Optional[float]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
            vals = [x for x in xs if x is not None]
            if not vals:
                return None, None, None
            avg = sum(vals) / len(vals)
            return min(vals), max(vals), avg

        due_min, due_max, due_avg = stats(due_hours)
        end_min, end_max, end_avg = stats(end_hours)

        from rich import box
        from rich.text import Text
        chart_panel = Panel(Text.from_ansi(block),
                            title="⏱️ CP Performance vs Baseline",
                            border_style="bright_black",
                            box=box.SIMPLE,
                            padding=(0, 1))
        console.print(chart_panel)

        # Optional: quick numeric summary table
        tbl = Table(show_header=True, header_style="bold", border_style="bright_black", box=box.SIMPLE_HEAVY)
        tbl.add_column("Series", justify="left")
        tbl.add_column("Min (h)", justify="right")
        tbl.add_column("Max (h)", justify="right")
        tbl.add_column("Avg (h)", justify="right")

        def f(v: Optional[float]) -> str:
            return f"{v:.2f}" if v is not None else "—"

        tbl.add_row("Due Δ", f(due_min), f(due_max), f(due_avg))
        tbl.add_row("End Δ", f(end_min), f(end_max), f(end_avg))
        console.print(tbl)



    def _task_link_number(self, t: Dict, fallback_index_1_based: int) -> int:
        """
        Return the chain link number for a task.
        Prefers UDAs 'link' or 'chainedLink'; falls back to the 1-based index we pass in.
        """
        v = t.get("link")
        if v is None:
            v = t.get("chainedLink")
        try:
            return int(v)
        except Exception:
            return int(fallback_index_1_based)

    def _should_use_vertical_plot(self, n_points: int) -> bool:
        """
        Decide whether to use the vertical plot (better for mobile/narrow terminals).
        - Force with env ANALYZER_VERTICAL=1
        - Auto if width is small or points won't fit comfortably in a line chart.

        """
        # explicit CLI override
        if hasattr(self, "force_vertical"):
            return bool(self.force_vertical)

        if str(os.environ.get("ANALYZER_VERTICAL", "0")).strip() in ("1", "true", "on", "yes", "y"):
            return True
        try:
            width = console.size.width
        except Exception:
            import shutil
            width = shutil.get_terminal_size((80, 20)).columns
        # Heuristics: if fewer than ~70 columns OR too many points for the default line plot width
        return width < 90 or n_points > (width - 20)


    def _render_vertical_stem_plot(
        self,
        x_labels: list[str],                        # '#1', '#2', …
        series: dict[str, list[Optional[float]]],   # hours
        height: int = 12,
        col_width: int = 2,                         # 2 chars per column keeps it tight
    ) -> Panel:
        from rich.text import Text

        # Determine symmetric y-range around 0
        all_vals = []
        for vals in series.values():
            all_vals += [abs(v) for v in vals if v is not None]
        ymax = max(all_vals) if all_vals else 1.0
        ymin, ymax = -ymax, ymax
        if ymax == 0:
            ymax = 1.0
            ymin = -1.0

        height = max(6, height)
        n = len(x_labels)
        if n == 0:
            return Panel(Text(""), title="⏱️ CP Performance (vertical)", border_style=COLORS["secondary"])

        # Map y (hours) to grid row
        def y_to_row(y: float) -> int:
            pos = 0 if ymax == ymin else (ymax - y) / (ymax - ymin)
            return max(0, min(height - 1, int(round(pos * (height - 1)))))

        zero_row = y_to_row(0.0)

        # Build empty grid (height rows, n cols * col_width)
        width = n * col_width
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Draw baseline (horizontal) at zero
        for c in range(width):
            grid[zero_row][c] = "─"

        # Colors
        s_colors = {"Due Δ": "cyan", "End Δ": "magenta"}
        s_glyphs = {"Due Δ": "│",     "End Δ": "│"}       # stems
        s_point  = {"Due Δ": "●",     "End Δ": "◆"}       # tip symbol

        # Draw each series as vertical stems per column
        for name, vals in series.items():
            color = s_colors.get(name, "white")
            stem  = s_glyphs.get(name, "│")
            tip   = s_point.get(name, "●")

            for i, v in enumerate(vals):
                if v is None:
                    continue
                col_start = i * col_width
                col_mid = col_start + (col_width // 2)
                y_row = y_to_row(float(v))

                # Draw stem from baseline to y_row
                if y_row < zero_row:
                    r0, r1 = y_row, zero_row
                else:
                    r0, r1 = zero_row, y_row

                for r in range(r0, r1 + 1):
                    grid[r][col_mid] = stem

                # Draw tip
                grid[y_row][col_mid] = tip

        # Convert to Rich Text lines with coloring
        lines = []
        for r in range(height):
            t = Text()
            for c in range(width):
                ch = grid[r][c]
                # Baseline
                if r == zero_row and ch == "─":
                    t.append(ch, style="grey50")
                # Series stems/tips — try to color by proximity (heuristic)
                elif ch in ("│", "●", "◆"):
                    # choose color by checking which series has a point near this row in this column (rough)
                    # we’ll simply color both series glyphs consistently:
                    if ch == "◆":
                        t.append(ch, style=s_colors["End Δ"])
                    elif ch == "●":
                        t.append(ch, style=s_colors["Due Δ"])
                    else:
                        # a neutral stem; keep white to avoid over-coloring
                        t.append(ch, style="white")
                else:
                    t.append(ch)
            lines.append(t)

        # Label row: print every Kth label for readability
        label_row = [" " for _ in range(width)]
        step = 1
        if n > 30:
            step = 2
        if n > 60:
            step = 3
        if n > 90:
            step = max(1, n // 30)  # ~<= 30 visible labels

        for i, lab in enumerate(x_labels):
            if i % step != 0:
                continue
            s = lab
            col_start = i * col_width
            col_mid = col_start + (col_width // 2)
            start = max(0, min(width - 1, col_mid - len(s)//2))
            for j, ch in enumerate(s):
                pos = start + j
                if pos < width:
                    label_row[pos] = ch
        lines.append(Text("".join(label_row), style="grey70"))

        # Legend
        legend = Table.grid(padding=(0,2))
        legend.add_column(style="white"); legend.add_column(style="white")
        legend.add_row(Text("Baseline", style="grey50"), "0 (on schedule)")
        legend.add_row(Text("Due Δ", style="cyan"), "early (−) / late (+)")
        legend.add_row(Text("End Δ", style="magenta"), "early (−) / late (+)")

        from rich.console import Group
        return Panel(Align.left(Group(*lines)),
                    title="⏱️ CP Performance vs Baseline (vertical)",
                    border_style=COLORS["secondary"],
                    padding=(1,1))



    # ── Time / formatting helpers ────────────────────────────────────────────

    def _context_datetime_for_task(self, t: Dict) -> str:
        """Pick a contextual timestamp for the changes table: prefer due, else end, else start, else entry."""
        for key in ("due", "end", "start", "entry"):
            val = t.get(key)
            if val:
                return self.format_local_time(val, "%Y-%m-%d")
        return "—"

    @staticmethod
    @lru_cache(maxsize=128)
    def convert_to_local(utc_time_str: str) -> Optional[datetime.datetime]:
        if not utc_time_str:
            return None
        try:
            utc_time = date_parser.parse(utc_time_str)
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=UTC_ZONE)
            return utc_time.astimezone(LOCAL_ZONE)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def format_local_time(time_obj, fmt: str = "%Y-%m-%d %H:%M") -> str:
        if not time_obj:
            return "N/A"
        if isinstance(time_obj, str):
            time_obj = TaskAnalyzer.convert_to_local(time_obj)
            if not time_obj:
                return "N/A"
        return time_obj.astimezone(LOCAL_ZONE).strftime(fmt)

    @staticmethod
    def parse_duration(duration_str: str) -> int:
        """Parse ISO-8601 duration like P1DT2H3M into seconds."""
        if not duration_str or not duration_str.startswith('P'):
            return 0
        import re
        seconds = 0
        parts = duration_str[1:].replace('T', '')
        for value, unit in re.findall(r'(\d+)([DHMS])', parts):
            seconds += int(value) * {'D': 86400, 'H': 3600, 'M': 60, 'S': 1}.get(unit, 0)
        return seconds

    @staticmethod
    def format_timedelta(seconds: int) -> str:
        if seconds == 0:
            return "0s"
        days, rem = divmod(int(seconds), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts = []
        if days: parts.append(f"{days}d")
        if hours: parts.append(f"{hours}h")
        if minutes: parts.append(f"{minutes}m")
        if secs and not parts: parts.append(f"{secs}s")
        return " ".join(parts) if parts else "0s"

    @staticmethod
    def _seconds_between(prev_str: Optional[str], cur_str: Optional[str]) -> Optional[int]:
        if not prev_str or not cur_str:
            return None
        try:
            prev = TaskAnalyzer.convert_to_local(prev_str)
            cur = TaskAnalyzer.convert_to_local(cur_str)
            if not prev or not cur:
                return None
            return int((cur - prev).total_seconds())
        except Exception:
            return None

    @staticmethod
    def _humanize_delta_seconds(seconds: int) -> str:
        if seconds == 0:
            return "±0m"
        sign = "+" if seconds > 0 else "-"
        s = abs(int(seconds))
        d, r = divmod(s, 86400)
        h, r = divmod(r, 3600)
        m, _ = divmod(r, 60)
        parts: List[str] = []
        if d: parts.append(f"{d}d")
        if h and len(parts) < 2: parts.append(f"{h}h")
        if not parts and m: parts.append(f"{m}m")
        if not parts: parts.append("0m")
        return sign + " ".join(parts)

    @staticmethod
    def _priority_arrow(prev: Optional[str], cur: Optional[str]) -> str:
        order = {"L": 0, "M": 1, "H": 2}
        if prev in order and cur in order:
            if order[cur] > order[prev]:
                return " ↑"
            if order[cur] < order[prev]:
                return " ↓"
        return ""

    @staticmethod
    def _list_diff(prev, cur):
        prev = set(prev or [])
        cur = set(cur or [])
        added = sorted(cur - prev)
        removed = sorted(prev - cur)
        return added, removed

    # ── Change detection ─────────────────────────────────────────────────────
    def detect_meaningful_changes(self, prev_task: Optional[Dict], current_task: Dict) -> List[TaskChange]:
        if not prev_task:
            return []

        # Removed 'due' and 'end' because the CP Performance panel now visualizes them.
        meaningful_fields = {
            "description", "status", "priority", "project", "area",
            "wait", "scheduled", "until", "start",
            "duration", "tags", "depends", "annotations", "chainMax", "chainUntil", "anchor_mode", "cp_anchor"
        }

        changes: List[TaskChange] = []
        for field in meaningful_fields:
            prev_val = prev_task.get(field)
            curr_val = current_task.get(field)
            if prev_val == curr_val or (not prev_val and not curr_val):
                continue
            ch = self._analyze_field_change(field, prev_val, curr_val)
            if ch:
                changes.append(ch)
        return changes


    def _analyze_field_change(self, field: str, prev_val: Any, curr_val: Any) -> Optional[TaskChange]:
        change_type = self._get_change_type(prev_val, curr_val)

        if field in ["due", "wait", "scheduled", "until", "start", "end"]:
            prev_disp = self.format_local_time(prev_val, "%m/%d %H:%M") if prev_val else None
            cur_disp  = self.format_local_time(curr_val, "%m/%d %H:%M") if curr_val else None
            delta_s   = self._seconds_between(prev_val, curr_val)
            delta_txt = self._humanize_delta_seconds(delta_s) if delta_s is not None else None
            if prev_disp != cur_disp:
                return TaskChange(field=field, change_type=change_type,
                                  from_value=prev_disp, to_value=cur_disp,
                                  delta=delta_txt, group=10)
            return None

        if field == "priority":
            prev_disp = prev_val or ""
            cur_disp  = curr_val or ""
            if prev_disp != cur_disp:
                return TaskChange(field=field, change_type=change_type,
                                  from_value=prev_disp, to_value=cur_disp + self._priority_arrow(prev_disp, cur_disp),
                                  group=20)
            return None

        if field == "status":
            prev_disp = (str(prev_val).title() if prev_val else None)
            cur_disp  = (str(curr_val).title() if curr_val else None)
            if prev_disp != cur_disp:
                return TaskChange(field=field, change_type=change_type,
                                  from_value=prev_disp, to_value=cur_disp, group=20)
            return None

        if field == "duration":
            prev_s = self.parse_duration(prev_val) if prev_val else 0
            cur_s  = self.parse_duration(curr_val) if curr_val else 0
            if prev_s != cur_s:
                return TaskChange(field=field, change_type=change_type,
                                  from_value=self.format_timedelta(prev_s) if prev_s else None,
                                  to_value=self.format_timedelta(cur_s) if cur_s else None,
                                  delta=self._humanize_delta_seconds(cur_s - prev_s), group=30)
            return None

        if field in ["tags", "depends"]:
            added, removed = self._list_diff(prev_val, curr_val)
            if added or removed:
                return TaskChange(field=field, change_type="modified",
                                  added_items=added, removed_items=removed, group=40)
            return None

        if field == "annotations":
            return self._handle_annotations_change(prev_val, curr_val)

        prev_display = str(prev_val) if prev_val is not None else None
        curr_display = str(curr_val) if curr_val is not None else None
        if prev_display != curr_display:
            return TaskChange(field=field, change_type=change_type,
                              from_value=prev_display, to_value=curr_display, group=90)
        return None

    @staticmethod
    def _get_change_type(prev_val: Any, curr_val: Any) -> str:
        if prev_val is None and curr_val is not None:
            return "added"
        elif prev_val is not None and curr_val is None:
            return "removed"
        else:
            return "modified"

    def _handle_annotations_change(self, prev_val: Any, curr_val: Any) -> Optional[TaskChange]:
        prev_annots = prev_val if isinstance(prev_val, list) else []
        curr_annots = curr_val if isinstance(curr_val, list) else []
        added = [a for a in curr_annots if a not in prev_annots]
        removed = [a for a in prev_annots if a not in curr_annots]
        if added or removed:
            return TaskChange(field="annotations", change_type="modified",
                              added_items=added, removed_items=removed, group=50)
        return None

    # ── Anchor / CP analysis helpers ─────────────────────────────────────────
    def _anchor_summary(self, task: Dict) -> Optional[Tuple[str, str]]:
        """Return (pattern, natural) if cp_anchor present."""
        expr = (task.get("cp_anchor") or "").strip()
        if not expr:
            return None
        natural = None
        pat = expr
        if core:
            try:
                dnf = core.validate_anchor_expr_strict(expr)
                natural = core.describe_anchor_dnf(dnf, task)
            except Exception:
                natural = None
        return (pat, natural)

    def _project_anchor_dates(
        self,
        task: Dict,
        limit: int = UPCOMING_MAX,
        start_from_date: Optional[date] = None,  # local date; project strictly AFTER this date
    ) -> List[datetime.datetime]:
        """
        Project the next anchor datetimes in LOCAL TZ, strictly after `start_from_date`
        (exclusive). If start_from_date is None, we fall back to *today*.
        """
        expr = (task.get("cp_anchor") or "").strip()
        if not expr or not core:
            return []

        try:
            dnf = core.validate_anchor_expr_strict(expr)
        except Exception:
            return []

        # Base local date
        if start_from_date is None:
            now_utc = core.now_utc() if hasattr(core, "now_utc") else datetime.datetime.now(tz=UTC_ZONE)
            now_loc = core.to_local(now_utc) if hasattr(core, "to_local") else now_utc.astimezone(LOCAL_ZONE)
            start_from_date = now_loc.date()

        # step produces the next matching *date* after prev_date
        def step(prev_date: date):
            nxt_date, _ = core.next_after_expr(
                dnf, prev_date, default_seed=prev_date, seed_base=task.get("uuid") or "analyzer"
            )
            return nxt_date

        # First date strictly AFTER start_from_date
        first_date = step(start_from_date)
        if not first_date:
            return []

        # Time-of-day: from pattern if available; otherwise keep due’s HH:MM (if any), else 09:00
        due_local_dt = self.convert_to_local(task.get("due") or "")
        default_hhmm = (due_local_dt.hour, due_local_dt.minute) if due_local_dt else (9, 0)
        first_hhmm = core.pick_hhmm_from_dnf_for_date(dnf, first_date, first_date) or default_hhmm

        out = []
        cur_date, cur_hhmm = first_date, first_hhmm
        for _ in range(limit):
            dt_utc = core.build_local_datetime(cur_date, cur_hhmm)
            out.append(core.to_local(dt_utc))

            # next anchor
            nxt_date = step(cur_date)
            if not nxt_date:
                break
            cur_hhmm = core.pick_hhmm_from_dnf_for_date(dnf, nxt_date, first_date) or cur_hhmm
            cur_date = nxt_date

        return out


    def _project_cp_dates(
        self,
        task: Dict,
        limit: int = UPCOMING_MAX,
        start_from_dt_local: Optional[datetime.datetime] = None,  # local datetime; project strictly AFTER this datetime
    ) -> List[datetime.datetime]:
        """
        Project next cp-based occurrences strictly AFTER `start_from_dt_local`.
        If None, we fall back to now().
        """
        period = (task.get("cp") or "").strip()
        if not period:
            return []

        # duration
        if core and hasattr(core, "parse_cp_duration"):
            td = core.parse_cp_duration(period)
            if not td:
                return []
        else:
            import re
            m = re.match(r"^\s*(\d+)\s*([dwmyh])\s*$", period, re.I)
            if not m:
                return []
            n = int(m.group(1)); u = m.group(2).lower()
            mult = {'d':86400,'w':604800,'h':3600}.get(u)
            if u == 'm': mult = 30*86400
            if u == 'y': mult = 365*86400
            td = datetime.timedelta(seconds=n*mult)

        # base local datetime
        if start_from_dt_local is None:
            start_from_dt_local = datetime.datetime.now(tz=LOCAL_ZONE)

        # First occurrence after base
        occurrences = []
        base = start_from_dt_local
        for _ in range(limit):
            base = (base + td).replace(microsecond=0)
            occurrences.append(base)
        return occurrences


    def _due_is_anchor_day(self, due_dt_utc: str, task: Dict) -> Optional[bool]:
        """Check if a due (UTC string) falls on an anchor day (local date)."""
        if not core:
            return None
        expr = (task.get("cp_anchor") or "").strip()
        if not expr:
            return None
        try:
            dnf = core.validate_anchor_expr_strict(expr)
        except Exception:
            return None
        due_local = self.convert_to_local(due_dt_utc)
        if not due_local:
            return None
        due_day = due_local.date()
        def step(prev_date: date):
            nxt_date, _ = core.next_after_expr(dnf, prev_date, default_seed=prev_date, seed_base=task.get("uuid") or "analyzer")
            return nxt_date
        first_on_or_after = step(due_day - timedelta(days=1))
        return first_on_or_after == due_day

    def _months_per_row(self) -> int:
        try:
            width = console.size.width
        except Exception:
            import shutil
            width = shutil.get_terminal_size((80, 20)).columns
        if width >= 120:  # wide → 3 per row
            return 3
        if width >= 90:   # medium → 2 per row
            return 2
        return 1          # narrow → 1 per row


    # ── Rendering ────────────────────────────────────────────────────────────
    def format_changes_beautifully(self, changes: List[TaskChange]) -> Text:
        if not changes:
            return Text("→ No changes", style=COLORS['muted'])
        changes = sorted(changes, key=lambda c: (c.group, c.field))

        out = Text()
        for idx, ch in enumerate(changes):
            if idx: out.append("\n")
            icon = CHANGE_ICONS.get(ch.change_type, "•")
            field_color = FIELD_COLORS.get(ch.field, "white")
            out.append(f"{icon} ", style=COLORS['accent'])
            out.append(ch.field, style=field_color)
            out.append(": ")

            if ch.field in ("tags", "depends"):
                plus = [f"+{x}" for x in (ch.added_items or [])]
                minus = [f"-{x}" for x in (ch.removed_items or [])]
                body = " ".join(plus + minus) if (plus or minus) else "—"
                out.append(body, style="bold white"); continue

            if ch.field == "annotations":
                self._format_annotation_changes(out, ch); continue

            value_style = "bold bright_green" if ch.field == "end" else ("bright_red" if ch.field == "due" else "bold white")
            if ch.change_type == "added":
                out.append("set to ", style=COLORS['muted'])
                out.append(str(ch.to_value or "—"), style="bold white")
            elif ch.change_type == "removed":
                out.append("removed ", style=COLORS['muted'])
                out.append(str(ch.from_value or "—"), style="bold white")
            else:
                if ch.from_value is not None:
                    out.append(str(ch.from_value), style=value_style)
                    out.append(" → ", style="white")
                out.append(str(ch.to_value or "—"), style=value_style)
            if ch.delta:
                out.append(f"  ({ch.delta})", style=COLORS['secondary'])
        return out

    def _format_annotation_changes(self, result: Text, change: TaskChange):
        a_count = len(change.added_items or [])
        r_count = len(change.removed_items or [])
        head = []
        if a_count: head.append(f"+{a_count}")
        if r_count: head.append(f"-{r_count}")
        if head:
            result.append("notes ", style=COLORS['muted'])
            result.append("/".join(head), style="bold white")
        else:
            result.append("notes —", style=COLORS['muted'])

        if change.added_items:
            try:
                newest = max(
                    change.added_items,
                    key=lambda a: self.convert_to_local(a.get("entry") or "") or datetime.datetime.min
                )
            except Exception:
                newest = change.added_items[-1]
            when = self.format_local_time(newest.get("entry"), "%m/%d %H:%M")
            txt = (newest.get("description") or "").strip()
            if txt:
                preview = (txt[:60] + "…") if len(txt) > 60 else txt
                result.append(f"  | latest: [{when}] {preview}", style=COLORS['secondary'])

    # Calendar helpers
    def _create_month_table(self, cal, year: int, month: int, completed: Set[date], upcoming: Set[date], pending_due: Set[date]) -> Table:
        table = Table(show_header=True, box=None, padding=(0, 1))
        for day in ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]:
            table.add_column(day, justify="center", min_width=3, style=COLORS['muted'])
        for week in cal.monthdays2calendar(year, month):
            row = []
            for day_num, _weekday in week:
                if day_num == 0:
                    row.append("   ")
                else:
                    d = date(year, month, day_num)
                    if d in completed:
                        row.append(f"[{COLORS['success']}]{day_num:2d}[/{COLORS['success']}]")
                    elif d in pending_due:
                        row.append(f"[{COLORS['pending']}]{day_num:2d}[/{COLORS['pending']}]")
                    elif d in upcoming:
                        row.append(f"[{COLORS['future']}]{day_num:2d}[/{COLORS['future']}]")
                    else:
                        row.append(f"[{COLORS['muted']}]{day_num:2d}[/{COLORS['muted']}]")
            table.add_row(*row)
        return table

    def create_enhanced_calendar(self, completed_dates: List[date], upcoming_dates: List[date], pending_due_dates: List[date]) -> Panel:
        if not (completed_dates or upcoming_dates or pending_due_dates):
            empty = Align.center("📅 No chain activity")
            return Panel(empty, title="📅 Calendar", border_style=COLORS['warning'], padding=(0,1), expand=False)

        all_dates = sorted(set(completed_dates + upcoming_dates + pending_due_dates))
        first_date, last_date = all_dates[0], all_dates[-1]
        cal = calendar.TextCalendar(calendar.MONDAY)

        cset, uset, pset = set(completed_dates), set(upcoming_dates), set(pending_due_dates)

        panels = []
        for year in range(first_date.year, last_date.year + 1):
            start_m = first_date.month if year == first_date.year else 1
            end_m = last_date.month if year == last_date.year else 12
            for month in range(start_m, end_m + 1):
                mtable = self._create_month_table(cal, year, month, cset, uset, pset)
                panels.append(Panel(mtable, title=f"{calendar.month_name[month]} {year}",
                                    border_style=COLORS['secondary'], padding=(0,1), expand=False))

        mpr = self._months_per_row()
        rows = []
        for i in range(0, len(panels), mpr):
            # tight layout: no equal widths, no expand, small padding between months
            rows.append(Columns(panels[i:i+mpr], equal=False, expand=False, padding=1))

        summary = f"✅ {len(cset)} done • 🔵 {len(uset)} upcoming • 🟡 {len(pset)} pending"
        return Panel(Align.left(Columns(rows, equal=False, expand=False, padding=0)),
                    title=summary, border_style=COLORS['primary'], padding=(0,1), expand=False)



    # ── Baseline / Table / Status ────────────────────────────────────────────
    def _format_status_text(self, status: str) -> Text:
        styles = {
            "Completed": (COLORS['success'], "✅"),
            "Pending":   (COLORS['warning'], "⏳"),
            "Waiting":   (COLORS['secondary'], "🔵"),
            "Deleted":   (COLORS['error'], "❌"),
            "Recurring": (COLORS['accent'], "🔄")
        }
        style, icon = styles.get(status, ("white", "•"))
        return Text(f"{icon} {status}", style=style)

    def _format_status(self, status: str) -> Text:
        styles = {
            "completed": f"✅ {COLORS['success']}",
            "pending":   f"⏳ {COLORS['warning']}",
            "waiting":   f"🔵 {COLORS['secondary']}",
            "deleted":   f"❌ {COLORS['error']}",
            "recurring": f"🔄 {COLORS['accent']}"
        }
        style = styles.get(status, f"• {COLORS['muted']}")
        return Text(status.title(), style=style)

    def _display_baseline_task(self, baseline_task: Dict, is_truncated: bool = False):
        details = Table(show_header=False, box=None, padding=(0, 2))
        details.add_column("Field", style=f"bold {COLORS['accent']}", width=15)
        details.add_column("Value", overflow="fold")

        fields = [
            ("UUID", baseline_task.get("uuid", "N/A")[:8]),
            ("Description", baseline_task.get("description", "N/A")),
            ("Status", baseline_task.get("status", "N/A").title()),
            ("Project", baseline_task.get("project", "N/A")),
            ("Priority", baseline_task.get("priority", "N/A")),
            ("Link Period", baseline_task.get("cp", "N/A")),
            ("Created", self.format_local_time(baseline_task.get("entry"), "%Y-%m-%d %H:%M")),
        ]

        # Anchor summary if present
        anchor = self._anchor_summary(baseline_task)
        if anchor:
            pat, nat = anchor
            fields.append(("Pattern", pat))
            if nat:
                fields.append(("Natural", nat))

        for label, field in [("Due", "due"), ("Scheduled", "scheduled"), ("Wait", "wait"),
                             ("Started", "start"), ("Completed", "end")]:
            if baseline_task.get(field):
                fields.append((label, self.format_local_time(baseline_task[field], "%Y-%m-%d %H:%M")))

        if baseline_task.get("tags"):
            fields.append(("Tags", ", ".join(sorted(baseline_task["tags"]))))

        if baseline_task.get("depends"):
            fields.append(("Depends", ", ".join(baseline_task["depends"])))

        if baseline_task.get("duration"):
            dur = self.format_timedelta(self.parse_duration(baseline_task["duration"]))
            fields.append(("Duration", dur))

        if baseline_task.get("annotations"):
            n = len(baseline_task["annotations"])
            fields.append(("Annotations", f"{n} annotation{'s' if n != 1 else ''}"))

        for field, value in fields:
            if field.lower() in FIELD_COLORS:
                field_text = Text(field, style=FIELD_COLORS[field.lower()])
            else:
                field_text = Text(field, style=COLORS['muted'])

            if field == "Status":
                value_text = self._format_status_text(value)
            elif field in ["Due", "Scheduled", "Wait", "Started", "Completed"] and value != "N/A":
                value_text = Text(value, style=COLORS['secondary'])
            elif field == "Priority" and value != "N/A":
                pr_colors = {"H": COLORS['error'], "M": COLORS['warning'], "L": COLORS['success']}
                value_text = Text(value, style=pr_colors.get(value, "white"))
            else:
                value_text = Text(str(value), style="white")

            details.add_row(field_text, value_text)

        title = "📋 Baseline Task Details"
        if is_truncated:
            title += " (showing last tasks in chain)"

        console.print(Panel(details, title=title, border_style=COLORS['primary'], padding=(1, 2)))
        console.print()

    def _display_chain_table(self, chain: List[Dict]):
        # Keep only rows that *actually* have non-timing changes after our filters
        rows = [
            (t.get("uuid", "")[:8], self._context_datetime_for_task(t), t.get("meaningful_changes", []))
            for t in chain
        ]
        rows = [(u, d, ch) for (u, d, ch) in rows if ch]  # drop "no changes"

        table = Table(
            show_header=True,
            header_style=f"bold {COLORS['primary']}",
            border_style=COLORS['secondary'],
            box=box.ROUNDED
        )
        table.add_column("UUID", style=f"bold {COLORS['accent']}", width=8)
        table.add_column("Date", style="white", width=17, no_wrap=True)
        table.add_column("Changes", overflow="fold", min_width=40)

        if not rows:
            table.add_row("—", "—", Text("No non-timing changes", style=COLORS["muted"]))
            console.print(table)
            return

        for short_uuid, when_str, changes in rows:
            pretty = self.format_changes_beautifully(changes)
            table.add_row(short_uuid, when_str, pretty)

        console.print(table)




    def _get_completion_dates(self, chain: List[Dict]) -> List[date]:
        dates: List[date] = []
        for task in chain:
            if task.get("status") == "completed" and task.get("end"):
                local_time = self.convert_to_local(task["end"])
                if local_time:
                    dates.append(local_time.date())
        return sorted(set(dates))

    def _get_pending_due_dates(self, chain: List[Dict]) -> List[date]:
        dates: List[date] = []
        for task in chain:
            if task.get("status") in ("pending","waiting") and task.get("due"):
                local_time = self.convert_to_local(task["due"])
                if local_time:
                    dates.append(local_time.date())
        return sorted(set(dates))

    def _finished_chain_brief(self, chain: List[Dict]):
        """Compact in-depth summary when a chain is finished."""
        total = len(chain)
        completed = [t for t in chain if t.get("status") == "completed" and t.get("end")]
        if not completed:
            return

        first_end = self.convert_to_local(completed[0]["end"])
        last_end  = self.convert_to_local(completed[-1]["end"])
        span_days = (last_end.date() - first_end.date()).days if first_end and last_end else 0

        # Cadence based on end-to-end gaps (in days)
        gaps = []
        for i in range(1, len(completed)):
            a = self.convert_to_local(completed[i-1]["end"])
            b = self.convert_to_local(completed[i]["end"])
            if a and b:
                gaps.append((b - a).total_seconds() / 86400.0)

        med = statistics.median(gaps) if gaps else 0
        mean = statistics.mean(gaps) if gaps else 0

        # Anchor adherence (% completed tasks whose due fell on an anchor day)
        adherence_txt = "N/A"
        if any((t.get("cp_anchor") for t in chain)) and core:
            checks = []
            for t in completed:
                due = t.get("due")
                if due:
                    ok = self._due_is_anchor_day(due, t)
                    if ok is not None:
                        checks.append(1 if ok else 0)
            if checks:
                adherence_txt = f"{100.0 * sum(checks)/len(checks):.0f}% on-anchor"

        grid = Table.grid(padding=(0,2))
        grid.add_column(style="bold white")
        grid.add_column(style="white")
        grid.add_row("Total links", str(total))
        grid.add_row("Completed",   str(len(completed)))
        if first_end and last_end:
            grid.add_row("First→Last", f"{first_end.date()} → {last_end.date()}  ({span_days} days)")
        grid.add_row("Cadence", f"median {med:.1f}d • mean {mean:.1f}d")
        grid.add_row("Anchor adherence", adherence_txt)

        console.print(Panel(grid, title="📦 Finished Chain Summary", border_style=COLORS['accent'], expand=False))

    # ── Interactive helpers ──────────────────────────────────────────────────
    def select_chain_interactively(self) -> List[Dict]:
        """One fuzzy choice per chain (tail desc). Collisions resolved to newest tail."""
        chains = self.build_all_chains()

        from prompt_toolkit import prompt
        from prompt_toolkit.completion import FuzzyCompleter, WordCompleter

        # Map base label -> list of chain indexes sharing same text
        bucket = defaultdict(list)
        for i, ch in enumerate(chains):
            tail = ch[-1]
            label = f"[{tail.get('status','unknown')}] {tail.get('description','<no description>')}"
            bucket[label].append(i)

        # Keep newest tail when labels collide
        def tail_when(idx: int):
            tail = chains[idx][-1]
            when = (tail.get("end") or tail.get("due") or tail.get("entry") or "")
            return (when, tail.get("uuid") or "")

        labels = []
        label_to_index = {}
        for label, idxs in bucket.items():
            winner = sorted(idxs, key=tail_when, reverse=True)[0]
            labels.append(label)
            label_to_index[label] = winner

        # Active first
        def is_active(idx): return chains[idx][-1].get("status") in ("pending", "waiting")
        labels.sort(key=lambda lbl: (0 if is_active(label_to_index[lbl]) else 1, lbl.lower()))

        completer = FuzzyCompleter(WordCompleter(labels, match_middle=True))
        console.print(Panel("🔗 Pick a chain (one entry per chain; fuzzy search).",
                            title="Chain Selection", border_style=COLORS['primary']))
        while True:
            try:
                sel = prompt("❯ ", completer=completer).strip()
                if sel in label_to_index:
                    return chains[label_to_index[sel]]
                console.print(f"[{COLORS['error']}]Not found. Try again.[/]")
            except KeyboardInterrupt:
                console.print(f"\n[{COLORS['warning']}]Cancelled by user[/]")
                sys.exit(0)



    def select_task_interactively(self) -> int:
        tasks = self.get_all_chained_tasks()
        choices: Dict[str, int] = {}
        for task in tasks:
            if task.get("id"):
                desc = task.get("description", "<no description>")
                status = task.get("status", "unknown")
                display = f"[{status}] {desc}"
                choices[display] = int(task["id"])

        completer = FuzzyCompleter(WordCompleter(list(choices.keys()), match_middle=True))
        console.print(Panel("🔍 Type to search for a task (fuzzy matching enabled)", title="Task Selection", border_style=COLORS['primary']))
        while True:
            try:
                selection = prompt("❯ ", completer=completer).strip()
                if selection in choices:
                    return choices[selection]
                console.print(f"[{COLORS['error']}]Task not found. Please try again.[/]")
            except KeyboardInterrupt:
                console.print(f"\n[{COLORS['warning']}]Cancelled by user[/]")
                sys.exit(0)

    def get_count_interactively(self) -> Optional[int]:
        while True:
            try:
                inp = console.input("📊 How many recent tasks to show? (default: 10, 'all' for everything): ").strip()
                if not inp:
                    return 10
                if inp.lower() in ["all", "a"]:
                    return None
                count = int(inp)
                if count > 0:
                    return count
                console.print(f"[{COLORS['error']}]Please enter a positive number[/]")
            except ValueError:
                console.print(f"[{COLORS['error']}]Enter a number or 'all'[/]")
            except KeyboardInterrupt:
                console.print(f"\n[{COLORS['warning']}]Cancelled by user[/]")
                sys.exit(0)

    # ── Orchestration ────────────────────────────────────────────────────────
    def analyze_chain(self, chain: List[Dict], count: Optional[int] = None):
        console.print(Panel(Align.center("🔗 Taskwarrior Chain Analyzer (Pro)"),
                            style=f"bold {COLORS['primary']}", border_style=COLORS['primary'], expand=False))

        _ = self.get_all_chained_tasks()

        full_chain = chain
        display_chain = full_chain if count is None else full_chain[-count:]

        if display_chain:
            self._display_baseline_task(display_chain[0], len(full_chain) > len(display_chain))

        is_chain_active = bool(full_chain and full_chain[-1].get("status") in ("pending", "waiting"))

        for i, task in enumerate(display_chain):
            if i > 0:
                prev_task = display_chain[i - 1]
                changes = self.detect_meaningful_changes(prev_task, task)

                # Hide the cosmetic tail flip when chain is active
                is_last_displayed = (i == len(display_chain) - 1)
                if is_chain_active and is_last_displayed:
                    changes = [
                        ch for ch in changes
                        if not (
                            ch.field == "status"
                            and (str(ch.from_value or "").lower().startswith("completed"))
                            and (str(ch.to_value or "").lower().startswith(("pending", "waiting")))
                        )
                    ]

                task["meaningful_changes"] = changes
            else:
                task["meaningful_changes"] = []

        # calendar data:
        completed_dates = self._get_completion_dates(full_chain)
        pending_due_dates = self._get_pending_due_dates(full_chain)

        # upcoming link dates: cp_anchor preferred; else cp
        tail = full_chain[-1] if full_chain else None
        upcoming_local_datetimes: List[datetime.datetime] = []
        is_active = False

        if tail:
            is_active = tail.get("status") in ("pending", "waiting")

            if is_active:
                due_local_dt = self.convert_to_local(tail.get("due") or "")
                # For anchors: start strictly AFTER the due *date* (so overdue tasks show next anchors right away)
                if (tail.get("cp_anchor") or "").strip() and core:
                    start_from_date = due_local_dt.date() if due_local_dt else None
                    upcoming_local_datetimes = self._project_anchor_dates(
                        tail, limit=UPCOMING_MAX, start_from_date=start_from_date
                    )
                # For cp: start strictly AFTER the due *datetime*
                elif (tail.get("cp") or "").strip():
                    start_from_dt_local = due_local_dt if due_local_dt else None
                    upcoming_local_datetimes = self._project_cp_dates(
                        tail, limit=UPCOMING_MAX, start_from_dt_local=start_from_dt_local
                    )



        upcoming_dates = sorted(set(dt.date() for dt in upcoming_local_datetimes))
        console.print(self.create_enhanced_calendar(completed_dates, upcoming_dates, pending_due_dates))

        # cp_anchor panel (if present)
        anchor_info = self._anchor_summary(tail) if tail else None
        if anchor_info:
            pat, nat = anchor_info
            rows = [("Pattern", pat)]
            if nat:
                rows.append(("Natural", nat))
            if is_active and tail.get("due"):
                rows.append(("Current due", self.format_local_time(tail.get("due"), "%a %Y-%m-%d %H:%M %Z")))

            if core and is_active and upcoming_local_datetimes:
                rows.append((
                    "Next",
                    "\n".join(
                        dt.strftime("%a %Y-%m-%d %H:%M %Z")
                        for dt in upcoming_local_datetimes[:min(5, len(upcoming_local_datetimes))]
                    )
                ))
            elif not core:
                rows.append(("Note", "Anchors present but chained_core.py not found — anchor projection disabled."))

            console.print()  # simple blank line spacer
            self._panel("⛵ Anchor Analysis", rows, "bright_cyan", "bright_cyan")


        # chain table + summary
        self._display_chain_table(display_chain)
        # Show CP performance chart for classic CP chains (no anchors)
        tail = full_chain[-1] if full_chain else None
        if tail and (tail.get("cp") or "").strip() and not (tail.get("cp_anchor") or "").strip():
            self._cp_performance_panel(full_chain)

        # finished-chain brief
        if tail and tail.get("status") == "completed":
            if not self._children.get(tail.get("uuid"), []):
                self._finished_chain_brief(full_chain)
        self._display_summary(len(full_chain), len(display_chain), completed_dates)


    def _panel(self, title, rows, style="cyan", border_style="blue"):
        try:
            from rich.table import Table as RTable
            from rich.panel import Panel as RPanel
            from rich.text import Text
            t = RTable.grid(padding=(0, 2), expand=False)
            t.add_column(style="bold cyan", no_wrap=True, justify="right")
            t.add_column(style="white")
            for k, v in rows:
                t.add_row(f"{k}", v if isinstance(v, str) else str(v))
            console.print(RPanel(t, title=Text(title, style=f"bold {style}"),
                                 border_style=border_style, expand=False, padding=(0, 1)))
        except Exception:
            console.print(title)
            for k, v in rows: console.print(f"{k}: {v}")

    def _display_summary(self, total_tasks: int, displayed_tasks: int, completion_dates: List[date]):
        items = [
            f"📊 Total chain length: {total_tasks}",
            f"👁️  Displayed tasks: {displayed_tasks}",
            f"✅ Completed days: {len(completion_dates)}"
        ]
        if completion_dates:
            items.append(f"📅 Date span: {(max(completion_dates) - min(completion_dates)).days} days")

        console.print(Panel(Align.center(" • ".join(items)), style=COLORS['secondary'],
                            border_style=COLORS['muted'], expand=False))

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Taskwarrior Chain Analyzer — Pro Edition",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--explain", metavar="ANCHOR", help="Explain an anchor expression")
    parser.add_argument("--validate", metavar="ANCHOR", help="Validate an anchor expression")
    parser.add_argument("--self-check", action="store_true", help="Run self-check diagnostics")
    parser.add_argument("--id", type=int, help="Analyze a chain starting from a specific task ID")
    parser.add_argument("--mode", choices=["chain", "task"], default="chain",
                        help="Pick a whole chain (default) or start from a task")
    parser.add_argument("-c", "--count", default="all", help="Number of recent tasks to display")
    parser.add_argument("--vertical", action="store_true",
                        help="Force vertical timing chart (good for Termux/narrow terminals)")
    parser.add_argument("--horizontal", action="store_true",
                        help="Force horizontal timing chart")
    parser.add_argument(
        "--recover-dead-letter",
        action="store_true",
        help="Requeue dead-lettered spawn intents into the spawn queue",
    )
    parser.add_argument(
        "--recover-dry-run",
        action="store_true",
        help="Report recoverable dead-letter entries without writing files",
    )
    parser.add_argument(
        "--recover-prune",
        action="store_true",
        help="Remove recovered lines from the dead-letter file",
    )
    parser.add_argument(
        "--recover-limit",
        type=int,
        default=None,
        help="Recover at most N entries from dead-letter",
    )


    args = parser.parse_args()
    if not core:
        console.print(f"[{COLORS['error']}]Error: nautical_core.py not found.[/]")
        sys.exit(1)

    if args.self_check or args.explain or args.validate or args.recover_dead_letter:
        code = 0
        if args.self_check:
            code = max(code, _self_check())
        if args.validate:
            code = max(code, _validate_anchor(args.validate))
        if args.explain:
            code = max(code, _anchor_explain(args.explain))
        if args.recover_dead_letter:
            code = max(
                code,
                _recover_dead_letter(
                    dry_run=args.recover_dry_run,
                    prune=args.recover_prune,
                    limit=args.recover_limit,
                ),
            )
        sys.exit(code)

    analyzer = TaskAnalyzer()
    if args.vertical:
        analyzer.force_vertical = True
    elif args.horizontal:
        analyzer.force_vertical = False


    try:
        if args.mode == "task":
            if args.id is None:
                task_id = analyzer.select_task_interactively()
                count = analyzer.get_count_interactively()
            else:
                task_id = args.id
                count = None if args.count == "all" else int(args.count)
            chain = analyzer.build_chain_from_tasks(task_id)
            analyzer.analyze_chain(chain, count)

        else:  # mode == "chain"
            if args.id is None:
                chain = analyzer.select_chain_interactively()
                count = analyzer.get_count_interactively()
            else:
                # when given an id, still resolve full chain first
                chain = analyzer.build_chain_from_tasks(args.id)
                count = None if args.count == "all" else int(args.count)
            analyzer.analyze_chain(chain, count)

    except KeyboardInterrupt:
        console.print(f"\n[{COLORS['warning']}]Operation cancelled[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[{COLORS['error']}]Error: {e}[/]")
        sys.exit(1)


if __name__ == '__main__':
    main()
