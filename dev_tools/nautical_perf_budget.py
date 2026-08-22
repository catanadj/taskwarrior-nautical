#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic performance budget checks for core anchor paths.

Usage:
  python3 dev_tools/nautical_perf_budget.py
  python3 dev_tools/nautical_perf_budget.py --enforce
  python3 dev_tools/nautical_perf_budget.py --json --enforce
  python3 dev_tools/nautical_perf_budget.py --extended --json
  python3 dev_tools/nautical_perf_budget.py --extended --slow-device --workflows-only --json
  python3 dev_tools/nautical_perf_budget.py --budget-file dev_tools/perf_budget.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

core = importlib.import_module("nautical_core")
install_runtime = importlib.import_module("nautical_core.install_runtime")
lifecycle_outbox = importlib.import_module("nautical_core.lifecycle_outbox")
IMPORT_PROFILES: dict[str, int] = {}


def _load_budget_config(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to read budget file '{path}': {e}")
    if not isinstance(data, dict):
        raise RuntimeError(f"Budget file '{path}' must contain a JSON object.")
    workload = data.get("workload")
    budgets = data.get("budgets_seconds")
    if not isinstance(workload, dict) or not isinstance(budgets, dict):
        raise RuntimeError("Budget file requires 'workload' and 'budgets_seconds' objects.")
    return data


def _clear_caches() -> None:
    try:
        core._clear_all_caches()
    except Exception:
        pass


def _bench_parse_validate(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.validate_anchor_expr_strict(expr)
    return time.perf_counter() - t0


def _bench_describe_expr(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.describe_anchor_expr(expr)
    return time.perf_counter() - t0


def _bench_next_after(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    dnfs = [core.validate_anchor_expr_strict(expr) for expr in exprs]
    ref = date(2026, 1, 1)
    t0 = time.perf_counter()
    for _ in range(rounds):
        for dnf in dnfs:
            core.next_after_expr(dnf, ref)
    return time.perf_counter() - t0


def _bench_build_hints(exprs: list[str], rounds: int, *, mode: str = "warm") -> float:
    """Measure hint construction with an explicit persistent-cache state."""
    with _perf_cache_context():
        saved_load = core.cache_load
        saved_save = core.cache_save
        counts = {"hits": 0, "misses": 0, "saves": 0}

        def counted_load(*args, **kwargs):
            value = saved_load(*args, **kwargs)
            counts["hits" if value is not None else "misses"] += 1
            return value

        def counted_save(*args, **kwargs):
            result = saved_save(*args, **kwargs)
            if result:
                counts["saves"] += 1
            return result

        core.cache_load = counted_load
        core.cache_save = counted_save
        try:
            if mode == "cold":
                root = Path(core.ANCHOR_CACHE_DIR_OVERRIDE)
                started = time.perf_counter()
                for sample_index in range(max(1, rounds)):
                    core.ANCHOR_CACHE_DIR_OVERRIDE = str(root / f"cold-{sample_index}")
                    core._CACHE_DIR = None
                    _clear_caches()
                    for expr in exprs:
                        core.build_and_cache_hints(expr, "skip")
                if counts["hits"] or not counts["misses"]:
                    raise RuntimeError(f"cold hint benchmark observed unexpected cache state: {counts}")
                return time.perf_counter() - started
            if mode != "warm":
                raise ValueError(f"unknown hint benchmark mode: {mode}")
            core.cache_load = saved_load
            core.cache_save = saved_save
            _clear_caches()
            for expr in exprs:
                core.build_and_cache_hints(expr, "skip")
            core.cache_load = counted_load
            core.cache_save = counted_save
            counts = {"hits": 0, "misses": 0, "saves": 0}
            _clear_caches()
            t0 = time.perf_counter()
            for _ in range(max(1, rounds)):
                for expr in exprs:
                    core.build_and_cache_hints(expr, "skip")
            if counts["misses"] or not counts["hits"] or counts["saves"]:
                raise RuntimeError(f"warm hint benchmark observed unexpected cache state: {counts}")
            return time.perf_counter() - t0
        finally:
            core.cache_load = saved_load
            core.cache_save = saved_save


def _bench_cache_key_hot(exprs: list[str], rounds: int) -> float:
    _clear_caches()
    t0 = time.perf_counter()
    for _ in range(rounds):
        for expr in exprs:
            core.cache_key_for_task(expr, "skip")
    return time.perf_counter() - t0


@contextmanager
def _perf_cache_context():
    """Isolate cache I/O benchmarks from user cache directories."""
    saved_enable = bool(getattr(core, "ENABLE_ANCHOR_CACHE", False))
    saved_override = str(getattr(core, "ANCHOR_CACHE_DIR_OVERRIDE", "") or "")
    saved_cache_dir = getattr(core, "_CACHE_DIR", None)
    saved_ttl = int(getattr(core, "ANCHOR_CACHE_TTL", 0) or 0)
    with tempfile.TemporaryDirectory(prefix="nautical-perf-cache-") as td:
        try:
            core.ENABLE_ANCHOR_CACHE = True
            core.ANCHOR_CACHE_DIR_OVERRIDE = td
            core.ANCHOR_CACHE_TTL = 0
            core._CACHE_DIR = None
            _clear_caches()
            yield td
        finally:
            core.ENABLE_ANCHOR_CACHE = saved_enable
            core.ANCHOR_CACHE_DIR_OVERRIDE = saved_override
            core.ANCHOR_CACHE_TTL = saved_ttl
            core._CACHE_DIR = saved_cache_dir
            _clear_caches()


def _cache_payload(expr: str, idx: int) -> dict:
    return {
        "natural": expr,
        "next_dates": ["2026-01-01", "2026-01-08", "2026-01-15"],
        "meta": {"i": idx},
        # Keep payload shape aligned with cache schema checks.
        "dnf": [[{"typ": "w", "spec": "mon", "mods": {}}]],
    }


def _bench_cache_save(exprs: list[str], rounds: int) -> float:
    with _perf_cache_context():
        keys = [f"perf-save-{i}" for i in range(max(1, len(exprs)))]
        t0 = time.perf_counter()
        idx = 0
        for _ in range(rounds):
            for i, expr in enumerate(exprs):
                payload = _cache_payload(expr, idx)
                if not core.cache_save(keys[i], payload):
                    raise RuntimeError("cache_save benchmark write failed")
                idx += 1
        return time.perf_counter() - t0


def _bench_cache_load_hot(exprs: list[str], rounds: int) -> float:
    with _perf_cache_context():
        keys = [f"perf-load-{i}" for i in range(max(1, len(exprs)))]
        for i, expr in enumerate(exprs):
            if not core.cache_save(keys[i], _cache_payload(expr, i)):
                raise RuntimeError("cache_load benchmark setup write failed")
        _clear_caches()
        t0 = time.perf_counter()
        for _ in range(rounds):
            for key in keys:
                obj = core.cache_load(key)
                if not isinstance(obj, dict):
                    raise RuntimeError("cache_load benchmark read failed")
        return time.perf_counter() - t0


def _bench_outbox_schema_hot(rounds: int) -> float:
    with tempfile.TemporaryDirectory(prefix="nautical-perf-outbox-") as td:
        repository = lifecycle_outbox.LifecycleOutboxRepository(Path(td))
        if not repository.open().ok:
            raise RuntimeError("outbox schema benchmark setup failed")
        t0 = time.perf_counter()
        for _ in range(rounds):
            if not repository.open().ok:
                raise RuntimeError("outbox schema hot open failed")
        return time.perf_counter() - t0


def _bench_outbox_schema_cold(rounds: int) -> float:
    """Measure lifecycle outbox initialization across fresh Python processes."""
    with tempfile.TemporaryDirectory(prefix="nautical-perf-outbox-cold-") as td:
        script = (
            "from pathlib import Path; import sys; "
            "from nautical_core.lifecycle_outbox import LifecycleOutboxRepository; "
            "result = LifecycleOutboxRepository(Path(sys.argv[1])).open(); "
            "raise SystemExit(0 if result.ok else result.reason)"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
        )
        started = time.perf_counter()
        for _ in range(max(1, rounds)):
            proc = subprocess.run(
                [sys.executable, "-c", script, td],
                cwd=str(ROOT),
                env=env,
                text=True,
                capture_output=True,
                timeout=30.0,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"cold outbox initialization failed: {proc.stderr.strip()}")
        return time.perf_counter() - started


def _bench_cold_import(kind: str, rounds: int) -> float:
    """Measure a fresh-process import without reusing this benchmark process."""
    if kind == "core":
        script = "import nautical_core; print('__NAUTICAL_IMPORT_COUNT__=' + str(len(__import__('sys').modules)))"
    elif kind == "modify_impl":
        script = (
            "import importlib.util, sys; "
            "spec = importlib.util.spec_from_file_location('perf_modify_impl', sys.argv[1]); "
            "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "print('__NAUTICAL_IMPORT_COUNT__=' + str(len(sys.modules)))"
        )
    else:
        raise ValueError(f"unknown cold import benchmark kind: {kind}")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(ROOT), env.get("PYTHONPATH", "")) if part
    )
    command = [sys.executable, "-c", script]
    if kind == "modify_impl":
        command.append(str(ROOT / "nautical_core" / "hooks" / "modify_impl.py"))
    started = time.perf_counter()
    for _ in range(max(1, rounds)):
        proc = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=30.0,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"cold {kind} import failed: {proc.stderr.strip()}")
        for line in reversed((proc.stdout or "").splitlines()):
            if line.startswith("__NAUTICAL_IMPORT_COUNT__="):
                try:
                    IMPORT_PROFILES[kind] = int(line.split("=", 1)[1])
                except ValueError:
                    pass
                break
    return time.perf_counter() - started


def _bench_anchor_file_provider(rounds: int) -> float:
    """Exercise cached anchor-file expansion and successor lookup."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-file-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        for index in range(365):
            item_date = date(2026, 1, 1) + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        provider = anchor_files.AnchorFileOccurrenceProvider(
            "calendar.csv@t=09:00",
            td,
            (9, 0),
        )
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        started = time.perf_counter()
        for index in range(max(1, rounds)):
            after = datetime(2026, 1, 1, 8, 0) + timedelta(days=index % 364)
            if provider.next_after(after, build_local_datetime=build, to_local=identity) is None:
                raise RuntimeError("anchor-file provider benchmark unexpectedly exhausted")
        return time.perf_counter() - started


def _bench_anchor_file_batch_provider(rounds: int) -> float:
    """Measure provider-owned finite batch generation on a warm file cache."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    occurrence_provider = importlib.import_module("nautical_core.occurrence_provider")
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-batch-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        for index in range(365):
            item_date = date(2026, 1, 1) + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        provider = anchor_files.AnchorFileOccurrenceProvider("calendar.csv@t=09:00", td, (9, 0))
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        started = time.perf_counter()
        for index in range(max(1, rounds)):
            after = datetime(2026, 1, 1, 8, 0) + timedelta(days=index % 350)
            batch = occurrence_provider.collect_after(
                provider,
                after,
                limit=5,
                build_local_datetime=build,
                to_local=identity,
                require_contract=True,
            )
            if not batch:
                raise RuntimeError("anchor-file batch benchmark unexpectedly returned no occurrences")
        return time.perf_counter() - started


def _bench_large_anchor_file_provider(
    rounds: int,
    *,
    row_count: int = 5000,
    mode: str = "hot",
    business_day_only: bool = False,
) -> float:
    """Exercise large file-backed calendars and their cached lookup cursors."""
    anchor_files = importlib.import_module("nautical_core.anchor_files")
    business_calendar = importlib.import_module("nautical_core.business_calendar")
    row_count = max(1000, int(row_count))
    with tempfile.TemporaryDirectory(prefix="nautical-perf-anchor-large-") as td:
        path = Path(td) / "calendar.csv"
        rows = ["date,description"]
        first_day = date(2020, 1, 1)
        for index in range(row_count):
            item_date = first_day + timedelta(days=index)
            rows.append(f"{item_date.isoformat()},event-{index}")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        suffix = "@bd" if business_day_only else ""
        provider_name = f"calendar.csv{suffix}"
        calendar = business_calendar.DEFAULT_BUSINESS_CALENDAR if business_day_only else None
        build = lambda day, hhmm: datetime(day.year, day.month, day.day, *hhmm)
        identity = lambda value: value
        query_indexes = list(range(0, row_count - 2, max(1, row_count // 37)))
        if mode == "nonmonotonic":
            query_indexes = query_indexes[::2] + list(reversed(query_indexes[1::2]))
        started = time.perf_counter()
        provider = None
        if mode != "cold":
            provider = anchor_files.AnchorFileOccurrenceProvider(
                provider_name,
                td,
                (9, 0),
                business_calendar=calendar,
            )
            # Exclude setup from the hot lookup measurement.
            if provider.next_after(
                datetime.combine(first_day - timedelta(days=1), datetime.min.time()),
                build_local_datetime=build,
                to_local=identity,
            ) is None:
                raise RuntimeError("large anchor-file provider failed to load its first occurrence")
        for index in range(max(1, int(rounds))):
            for query_index in query_indexes:
                current = provider
                if mode == "cold":
                    current = anchor_files.AnchorFileOccurrenceProvider(
                        provider_name,
                        td,
                        (9, 0),
                        business_calendar=calendar,
                    )
                after = datetime.combine(
                    first_day + timedelta(days=query_index),
                    datetime.min.time(),
                )
                occurrence = current.next_after(
                    after,
                    build_local_datetime=build,
                    to_local=identity,
                )
                if occurrence is None:
                    raise RuntimeError(
                        f"large anchor-file provider exhausted unexpectedly ({mode}, query={query_index})"
                    )
                if business_day_only and occurrence.day.weekday() >= 5:
                    raise RuntimeError("business-day anchor-file benchmark returned a weekend")
        return time.perf_counter() - started


def _bench_business_calendar_omissions(rounds: int) -> float:
    """Measure recurrence selection with a large explicit omission set."""
    business_calendar = importlib.import_module("nautical_core.business_calendar")
    dnf = core.validate_anchor_expr_strict("m:1..31@bd")
    first_day = date(2026, 1, 1)
    calendar_days = frozenset(first_day + timedelta(days=index) for index in range(366))
    omitted = frozenset(
        item
        for item in calendar_days
        if item.day not in {1, 15} or item.weekday() >= 5
    )
    calendar = business_calendar.ConfiguredBusinessCalendar(
        name="perf-omissions",
        fingerprint="perf-omissions-v1",
        anchor_dates=calendar_days,
        omit_dates=omitted,
        _anchor_matches=lambda _value: False,
        _omit_matches=lambda _value: False,
    )
    started = time.perf_counter()
    for index in range(max(1, int(rounds))):
        result, _meta = core.next_after_expr(
            dnf,
            first_day + timedelta(days=index % 300),
            seed_base="perf-omissions",
            business_calendar=calendar,
        )
        if result is None or result.day not in {1, 15} or result.weekday() >= 5:
            raise RuntimeError("large omission-set benchmark selected an omitted date")
    return time.perf_counter() - started


def _bench_astronomy_provider(rounds: int, *, event: str) -> float | None:
    """Measure deterministic astronomy resolution when Astral is installed."""
    if importlib.util.find_spec("astral") is None:
        return None
    astronomy = importlib.import_module("nautical_core.astronomy")
    config = {
        "default_location": "perf",
        "locations": {
            "perf": {
                "latitude": 40.7128,
                "longitude": -74.0060,
                "elevation": 10,
                "timezone": "UTC",
            }
        },
    }
    started = time.perf_counter()
    for index in range(max(1, int(rounds))):
        reference = date(2026, 1, 1) + timedelta(days=index % 60)
        phase_day = astronomy.resolve_phase_date(
            "full",
            reference,
            config=config,
            horizon_days=60,
        )
        resolved = astronomy.resolve_event(event, phase_day, config=config)
        if resolved.tzinfo is None:
            raise RuntimeError("astronomy benchmark returned a naive datetime")
    return time.perf_counter() - started


def _bench_native_until_reconcile(rounds: int, *, apply: bool) -> float:
    """Measure native-until audit/repair on independent valid fixtures."""
    with tempfile.TemporaryDirectory(prefix="nautical-perf-native-until-") as td:
        root = Path(td)
        config_path = root / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\npanel_mode = "minimal"\n', encoding="utf-8")
        taskrc_path = root / "taskrc"
        taskrc_path.write_text(
            "uda.chainID.type=string\n"
            "uda.chain.type=string\n"
            "uda.link.type=numeric\n"
            "uda.prevLink.type=string\n"
            "uda.nextLink.type=string\n"
            "uda.cp.type=string\n"
            "uda.anchor.type=string\n"
            "uda.anchor_mode.type=string\n",
            encoding="utf-8",
        )
        base_env = dict(os.environ)
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                # The benchmark deliberately supplies isolated temporary
                # Taskdata directories; allow the launcher to use them.
                "NAUTICAL_TRUST_TASKDATA_PATH": "1",
                "TASKRC": str(taskrc_path),
                "TZ": "UTC",
            }
        )
        started = time.perf_counter()
        for sample_index in range(max(1, int(rounds))):
            taskdata = root / f"native-until-{sample_index}"
            taskdata.mkdir()
            parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"native-until/{sample_index}/parent"))
            child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"native-until/{sample_index}/child"))
            rows = [
                {
                    "uuid": parent_uuid,
                    # Native-until predecessor reads intentionally inspect
                    # completed/deleted history, so model the prior link as
                    # completed rather than leaving it active.
                    "status": "completed",
                    "description": "Native-until benchmark predecessor",
                    "cp": "P1D",
                    "chain": "on",
                    "chainID": f"native-until-{sample_index}",
                    "link": 1,
                    "nextLink": child_uuid[:8],
                    "due": "20270101T090000Z",
                    "until": "20270101T200000Z",
                },
                {
                    "uuid": child_uuid,
                    "status": "pending",
                    "description": "Native-until benchmark invalid child",
                    "cp": "P1D",
                    "chain": "on",
                    "chainID": f"native-until-{sample_index}",
                    "link": 2,
                    "prevLink": parent_uuid[:8],
                    "due": "20270102T090000Z",
                    "until": "20270101T200000Z",
                },
            ]
            env = dict(base_env, TASKDATA=str(taskdata))
            imported = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                text=True,
                capture_output=True,
                env=env,
                timeout=30.0,
            )
            if imported.returncode != 0:
                raise RuntimeError(f"native-until fixture import failed: {(imported.stderr or imported.stdout).strip()}")
            command = [sys.executable, str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"), "--json"]
            if apply:
                command.append("--apply")
            repaired = subprocess.run(
                command,
                text=True,
                capture_output=True,
                env=env,
                timeout=30.0,
            )
            if repaired.returncode != 0:
                raise RuntimeError(f"native-until reconcile failed: {(repaired.stderr or repaired.stdout).strip()}")
            try:
                summary = json.loads(repaired.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("native-until reconcile returned invalid JSON") from exc
            repairs = summary.get("native_until_repairs") if isinstance(summary, dict) else None
            if not isinstance(repairs, list) or len(repairs) != 1:
                raise RuntimeError("native-until benchmark did not inspect exactly one invalid child")
            item = repairs[0]
            if item.get("action") != "repair_until" or bool(item.get("applied")) != apply:
                raise RuntimeError(f"native-until benchmark did not take the expected path: {item}")
            if apply:
                exported = subprocess.run(
                    [
                        "task",
                        "rc.hooks=off",
                        "rc.json.array=1",
                        "rc.verbose=nothing",
                        "rc.color=off",
                        f"uuid:{child_uuid}",
                        "export",
                    ],
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=30.0,
                )
                if exported.returncode != 0:
                    raise RuntimeError("native-until verification export failed")
                rows_after = json.loads(exported.stdout or "[]")
                if not isinstance(rows_after, list) or len(rows_after) != 1:
                    raise RuntimeError("native-until verification export returned an unexpected task count")
                if str(rows_after[0].get("until") or "") != "20270102T200000Z":
                    raise RuntimeError("native-until apply benchmark did not persist the repaired endpoint")
        return time.perf_counter() - started


def _measure(name: str, fn, repeats: int) -> dict:
    samples = []
    # Warmup once for interpreter/cache stabilization.
    _ = fn()
    for _ in range(max(1, repeats)):
        samples.append(float(fn()))
    samples = sorted(samples)
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": statistics.median(samples),
        "max_s": samples[-1],
    }


def _strict_json_object(raw: str) -> dict:
    text = (raw or "").strip()
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(text)
    except Exception as exc:
        raise RuntimeError(f"hook stdout is not valid JSON: {exc}") from exc
    if text[end:].strip() or not isinstance(obj, dict):
        raise RuntimeError("hook stdout must contain exactly one JSON object")
    return obj


def _run_hook_timed(
    hook_path: Path,
    *,
    input_text: str,
    env: dict[str, str],
    expected_task: dict | None,
) -> float:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(hook_path)],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=15.0,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(
            f"{hook_path.name} failed with exit {proc.returncode}: "
            f"{(proc.stderr or proc.stdout or '').strip()}"
        )
    if expected_task is None:
        if (proc.stdout or "").strip():
            raise RuntimeError(f"{hook_path.name} wrote unexpected stdout")
    else:
        actual = _strict_json_object(proc.stdout or "")
        if actual != expected_task:
            raise RuntimeError(f"{hook_path.name} changed the plain passthrough task")
    return elapsed


def _init_empty_outbox(taskdata: Path) -> None:
    result = lifecycle_outbox.LifecycleOutboxRepository(taskdata).open()
    if not result.ok:
        raise RuntimeError(f"outbox benchmark setup failed: {result.reason or result.kind.value}")


def _measure_hook_fast_path(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    max_ratio: float,
) -> dict:
    fast_env = dict(base_env)
    fast_env.pop("NAUTICAL_BENCH_FORCE_FULL", None)
    full_env = dict(base_env)
    full_env["NAUTICAL_BENCH_FORCE_FULL"] = "1"

    _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
    _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)

    fast_samples: list[float] = []
    full_samples: list[float] = []
    for index in range(max(1, int(repeats))):
        if index % 2 == 0:
            fast_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
            )
            full_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)
            )
        else:
            full_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=full_env, expected_task=expected_task)
            )
            fast_samples.append(
                _run_hook_timed(hook_path, input_text=input_text, env=fast_env, expected_task=expected_task)
            )

    fast_samples.sort()
    full_samples.sort()
    fast_median = float(statistics.median(fast_samples))
    full_median = float(statistics.median(full_samples))
    ratio = fast_median / full_median if full_median > 0.0 else 1.0
    return {
        "name": name,
        "samples_s": fast_samples,
        "min_s": fast_samples[0],
        "median_s": fast_median,
        "max_s": fast_samples[-1],
        "full_samples_s": full_samples,
        "full_median_s": full_median,
        "fast_to_full_ratio": ratio,
        "max_ratio": float(max_ratio),
        "budget_s": 0.0,
        "pass": ratio <= float(max_ratio),
    }


def _measure_managed_hook_latency(
    name: str,
    hook_path: Path,
    *,
    input_text: str,
    expected_task: dict | None,
    base_env: dict[str, str],
    repeats: int,
    baseline_median_s: float,
    max_ratio: float,
) -> dict:
    env = dict(base_env)
    env.pop("NAUTICAL_CORE_PATH", None)
    env.pop("NAUTICAL_TRUST_CORE_PATH", None)
    env.pop("NAUTICAL_BENCH_FORCE_FULL", None)

    _run_hook_timed(hook_path, input_text=input_text, env=env, expected_task=expected_task)
    samples = sorted(
        _run_hook_timed(hook_path, input_text=input_text, env=env, expected_task=expected_task)
        for _ in range(max(1, int(repeats)))
    )
    median_s = float(statistics.median(samples))
    ratio = median_s / baseline_median_s if baseline_median_s > 0.0 else 1.0
    return {
        "name": name,
        "samples_s": samples,
        "min_s": samples[0],
        "median_s": median_s,
        "max_s": samples[-1],
        "baseline_median_s": float(baseline_median_s),
        "managed_to_source_ratio": ratio,
        "max_ratio": float(max_ratio),
        "budget_s": 0.0,
        "pass": ratio <= float(max_ratio),
    }


def _bench_hook_fast_paths(cfg: dict) -> dict[str, dict]:
    hook_cfg = cfg.get("hook_fast_path")
    if not isinstance(hook_cfg, dict) or not hook_cfg.get("enabled", True):
        return {}
    repeats = max(1, int(hook_cfg.get("repeats", 7)))
    max_ratios = hook_cfg.get("max_ratio") if isinstance(hook_cfg.get("max_ratio"), dict) else {}
    managed_max_ratio = float(hook_cfg.get("managed_layout_max_ratio", 1.5))

    plain = {
        "uuid": "11111111-1111-1111-1111-111111111111",
        "description": "plain hook latency",
        "status": "pending",
        "entry": "20260101T000000Z",
        "modified": "20260101T000000Z",
    }
    modified = dict(plain, modified="20260101T000001Z")

    with tempfile.TemporaryDirectory(prefix="nautical-hook-perf-") as td:
        temp_root = Path(td)
        config_path = temp_root / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\npanel_mode = "minimal"\n', encoding="utf-8")
        base_env = os.environ.copy()
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "TZ": "UTC",
            }
        )
        for key in ("NAUTICAL_DIAG", "NAUTICAL_DIAG_LOG", "NAUTICAL_PROFILE"):
            base_env.pop(key, None)

        cases = []
        add_data = temp_root / "add-data"
        add_data.mkdir()
        cases.append(
            (
                "hook_plain_add",
                ROOT / "on-add.nautical",
                json.dumps(plain, ensure_ascii=False),
                plain,
                add_data,
            )
        )
        modify_data = temp_root / "modify-data"
        modify_data.mkdir()
        cases.append(
            (
                "hook_plain_modify",
                ROOT / "on-modify.nautical",
                json.dumps(plain, ensure_ascii=False) + "\n" + json.dumps(modified, ensure_ascii=False),
                modified,
                modify_data,
            )
        )
        nautical_old = dict(
            plain,
            cp="P1D",
            chain="on",
            chainID="abcd1234",
            link=3,
            due="20270101T090000Z",
        )
        nautical_modified = dict(nautical_old, description="ordinary Nautical edit", modified="20260101T000001Z")
        cases.append(
            (
                "hook_nautical_ordinary_modify",
                ROOT / "on-modify.nautical",
                json.dumps(nautical_old, ensure_ascii=False) + "\n" + json.dumps(nautical_modified, ensure_ascii=False),
                nautical_modified,
                modify_data,
            )
        )
        exit_data = temp_root / "exit-data"
        exit_data.mkdir()
        _init_empty_outbox(exit_data)
        cases.append(("hook_empty_exit", ROOT / "on-exit.nautical", "", None, exit_data))

        results = {}
        for name, hook_path, input_text, expected_task, taskdata in cases:
            env = dict(base_env)
            env["TASKDATA"] = str(taskdata)
            ratio_budget = float(max_ratios.get(name, 0.8))
            results[name] = _measure_hook_fast_path(
                name,
                hook_path,
                input_text=input_text,
                expected_task=expected_task,
                base_env=env,
                repeats=repeats,
                max_ratio=ratio_budget,
            )

        managed_data = temp_root / "managed-data"
        install_runtime.install_release(
            source=ROOT,
            taskdata=managed_data,
            release_id="perf-managed",
            smoke=False,
        )
        _init_empty_outbox(managed_data)
        managed_env = dict(base_env)
        managed_env["TASKDATA"] = str(managed_data)
        for name, source_hook, input_text, expected_task, _taskdata in cases:
            managed_name = f"managed_{name}"
            results[managed_name] = _measure_managed_hook_latency(
                managed_name,
                managed_data / "hooks" / source_hook.name,
                input_text=input_text,
                expected_task=expected_task,
                base_env=managed_env,
                repeats=repeats,
                baseline_median_s=float(results[name]["median_s"]),
                max_ratio=managed_max_ratio,
            )
        return results


def _run_workflow_hook(hook_path: Path, *, input_text: str, env: dict[str, str], expect_output: bool) -> float:
    elapsed, _result, _stderr = _run_workflow_hook_result(
        hook_path,
        input_text=input_text,
        env=env,
        expect_output=expect_output,
    )
    return elapsed


def _run_workflow_hook_result(
    hook_path: Path,
    *,
    input_text: str,
    env: dict[str, str],
    expect_output: bool,
) -> tuple[float, dict | None, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(hook_path)],
        input=input_text,
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    elapsed = time.perf_counter() - started
    if proc.returncode != 0:
        raise RuntimeError(f"{hook_path.name} workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
    result = None
    if expect_output:
        result = _strict_json_object(proc.stdout or "")
    elif (proc.stdout or "").strip():
        raise RuntimeError(f"{hook_path.name} workflow wrote unexpected stdout")
    return elapsed, result, proc.stderr or ""


def _measure_workflow(name: str, samples: list[float], budget: float) -> dict:
    ordered = sorted(float(value) for value in samples)
    median = float(statistics.median(ordered))
    return {
        "name": name,
        "samples_s": ordered,
        "min_s": ordered[0],
        "median_s": median,
        "max_s": ordered[-1],
        "budget_s": float(budget),
        "pass": float(budget) <= 0.0 or median <= float(budget),
    }


def _read_exit_task_call_stats(path: Path) -> dict[str, int]:
    """Read benchmark-only command counters without enabling hook diagnostics."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"queue drain benchmark stats were unavailable: {exc}") from exc
    stats = payload.get("task_stats") if isinstance(payload, dict) else None
    if not isinstance(stats, dict):
        raise RuntimeError("queue drain benchmark stats did not contain task_stats")
    return {str(key): int(value) for key, value in stats.items() if str(key).startswith("run_task_calls")}


def _apply_task_call_budgets(result: dict, samples: list[dict[str, int]], budget: dict) -> None:
    """Attach and enforce per-workflow Taskwarrior call-count budgets."""
    if not isinstance(budget, dict) or not samples:
        return
    maxima = {
        key: max(int(sample.get(key, 0)) for sample in samples)
        for key in budget
        if all(isinstance(sample, dict) for sample in samples)
    }
    checks = {
        key: {
            "max_observed": value,
            "budget": int(budget[key]),
            "pass": value <= int(budget[key]),
        }
        for key, value in maxima.items()
    }
    result["task_call_budget"] = checks
    result["pass"] = bool(result.get("pass", True)) and all(item["pass"] for item in checks.values())


def _workflow_outbox_pending(taskdata: Path) -> list[dict]:
    """Read active lifecycle outbox records for benchmark mutation assertions."""
    result, status = lifecycle_outbox.LifecycleOutboxRepository(taskdata).status(limit=100)
    if not result.ok:
        raise RuntimeError(f"workflow outbox status failed: {result.reason or result.kind.value}")
    return [
        record
        for record in status.get("records", [])
        if isinstance(record, dict) and record.get("state") in {"ready", "claimed", "retry"}
    ]


def _stage_workflow_plans(
    taskdata: Path,
    plans: list,
    *,
    configuration_fingerprint: str,
    schedule_fingerprint: str,
) -> None:
    repository = lifecycle_outbox.LifecycleOutboxRepository(taskdata)
    for plan in plans:
        result = repository.enqueue(
            plan,
            configuration_fingerprint=configuration_fingerprint,
            schedule_fingerprint=schedule_fingerprint,
        )
        if not result.ok:
            raise RuntimeError(f"workflow outbox enqueue failed: {result.reason or result.kind.value}")


def _bind_workflow_plans_to_parents(plans: list, rows: list[dict]) -> list:
    """Bind benchmark plans to the guards Taskwarrior assigned on import."""
    from nautical_core.lifecycle_models import LifecyclePlan, ParentGuard, recurrence_fingerprint

    by_uuid = {str(row.get("uuid") or "").strip(): row for row in rows if isinstance(row, dict)}
    bound = []
    for plan in plans:
        parent = by_uuid.get(plan.identity.parent_uuid)
        if parent is None:
            raise RuntimeError(f"outbox fixture parent is missing: {plan.identity.parent_uuid}")
        guard = ParentGuard(
            status=str(parent.get("status") or ""),
            chain=str(parent.get("chain") or ""),
            chain_id=str(parent.get("chainID") or ""),
            link=int(parent.get("link") or 0),
            recurrence_fingerprint=recurrence_fingerprint(parent),
            modified=str(parent.get("modified") or ""),
        )
        bound.append(
            LifecyclePlan.from_mappings(
                identity=plan.identity,
                action=plan.action,
                parent_guard=guard,
                child_payload=plan.child_dict(),
                parent_patch=plan.parent_patch_dict(),
                expected_postconditions=plan.expected_postconditions,
                max_attempts=plan.max_attempts,
                stage=plan.stage,
            )
        )
    return bound


def _outbox_lifecycle_fixture(prefix: str, sample_index: int, count: int = 8) -> tuple[list[dict], list]:
    """Create independent typed lifecycle plans for durable outbox recovery tests."""
    from nautical_core.lifecycle_models import (
        ExecutionStage,
        LifecycleAction,
        LifecycleEvent,
        LifecycleIdentity,
        LifecyclePlan,
        ParentGuard,
    )

    parents: list[dict] = []
    plans: list = []
    for index in range(count):
        parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{prefix}/{sample_index}/parent/{index}"))
        child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/{prefix}/{sample_index}/child/{index}"))
        parent_link = index + 1
        child_link = parent_link + 1
        chain_id = f"queue-{prefix}-chain-{index}"
        parent = {
            "uuid": parent_uuid,
            "status": "completed",
            "description": f"Queue {prefix} benchmark parent",
            "chain": "on",
            "chainID": chain_id,
            "link": str(parent_link),
            "cp": "P1D",
            "due": "20260101T090000Z",
        }
        child = {
            "uuid": child_uuid,
            "status": "pending",
            "description": f"Queue {prefix} benchmark child",
            "chain": "on",
            "chainID": chain_id,
            "link": child_link,
            "prevLink": parent_uuid[:8],
            "cp": "P1D",
            "due": "20260102T090000Z",
        }
        guard = {"status": "completed", "chain": "on", "chainID": chain_id, "link": str(parent_link)}
        plan = LifecyclePlan.from_mappings(
            identity=LifecycleIdentity(
                chain_id, parent_uuid, parent_link, child_link, LifecycleEvent.COMPLETE
            ),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard.from_mapping(guard),
            child_payload=child,
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
            stage=ExecutionStage.PERSISTED,
        )
        parents.append(parent)
        plans.append(plan)
    return parents, plans


def _merge_task_call_stats(*stats: dict[str, int]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in stats:
        for key, value in item.items():
            merged[key] = merged.get(key, 0) + int(value)
    return merged


def _completion_fixture(kind: str, sample_index: int, *, nonfinal: bool, mode: str) -> dict:
    """Build independent, deterministic identities for one completion sample."""
    key = f"nautical-perf/{kind}/{mode}/{sample_index}"
    parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, key + "/parent"))
    chain_id = f"{kind}-perf-{mode}-{sample_index:04d}"
    limit = 2 if nonfinal else 1
    if kind == "cp":
        return {
            "uuid": parent_uuid,
            "status": "pending",
            "description": f"CP completion benchmark {mode} {sample_index}",
            "cp": "P1D",
            "chain": "on",
            "chainID": chain_id,
            "link": 1,
            "chainMax": limit,
            "due": "20260101T090000Z",
            "modified": "20260101T090000Z",
        }
    return {
        "uuid": parent_uuid,
        "status": "pending",
        "description": f"Anchor completion benchmark {mode} {sample_index}",
        "anchor": "w:mon@t=09:00",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": chain_id,
        "link": 1,
        "chainMax": limit,
        "due": "20260105T090000Z",
        "modified": "20260105T090000Z",
    }


def _import_existing_completion_child(parent: dict, *, env: dict[str, str]) -> None:
    """Seed an existing next link for the idempotent completion benchmark."""
    child_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, str(parent["uuid"]) + "/child"))
    child = {
        "uuid": child_uuid,
        "status": "pending",
        "description": parent["description"],
        "chain": "on",
        "chainID": parent["chainID"],
        # Keep link textual so Taskwarrior's JSON export compares exactly with
        # the completion lookup's requested link number.
        "link": "2",
        "prevLink": str(parent["uuid"])[:8],
        "due": "20260102T090000Z" if parent.get("cp") else "20260112T090000Z",
    }
    if parent.get("cp"):
        child["cp"] = parent["cp"]
    else:
        child["anchor"] = parent["anchor"]
        child["anchor_mode"] = parent["anchor_mode"]
    proc = subprocess.run(
        ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
        input=json.dumps(child, ensure_ascii=False) + "\n",
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"idempotent completion fixture import failed: {(proc.stderr or proc.stdout or '').strip()}")


def _import_workflow_rows(rows: Sequence[dict], *, env: dict[str, str]) -> None:
    """Seed synthetic lifecycle parents before exercising full hooks.

    Full completion paths intentionally perform authoritative Taskwarrior
    reads.  Keeping the fixture import explicit makes the benchmark measure
    lifecycle work rather than the fail-closed response to an empty data dir.
    """
    proc = subprocess.run(
        ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
        input="".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        text=True,
        capture_output=True,
        env=env,
        timeout=30.0,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"workflow parent fixture import failed: {(proc.stderr or proc.stdout or '').strip()}")


def _bench_expensive_workflows(cfg: dict) -> dict[str, dict]:
    """Exercise completion, queue-drain, and reconcile paths in isolation."""
    workflow_cfg = cfg.get("workflow_perf")
    if not isinstance(workflow_cfg, dict) or not workflow_cfg.get("enabled", True):
        return {}
    repeats = max(1, int(workflow_cfg.get("repeats", 3)))
    budgets = workflow_cfg.get("budgets_seconds") if isinstance(workflow_cfg.get("budgets_seconds"), dict) else {}
    with tempfile.TemporaryDirectory(prefix="nautical-workflow-perf-") as td:
        root = Path(td)
        real_task = shutil.which("task")
        if not real_task:
            raise RuntimeError("Taskwarrior executable was not found for workflow benchmark")
        task_wrapper = root / "task-benchmark-wrapper.py"
        task_wrapper.write_text(
            "#!/usr/bin/env python3\n"
            "import os, subprocess, sys\n"
            f"real = {real_task!r}\n"
            "args = sys.argv[1:]\n"
            "mode = os.environ.get('NAUTICAL_BENCH_FAIL_MODE', '')\n"
            "if mode == 'fail-export' and 'export' in args:\n"
            "    sys.stderr.write('database is locked\\n')\n"
            "    raise SystemExit(1)\n"
            "if mode == 'partial-import' and 'import' in args:\n"
            "    lines = sys.stdin.read().splitlines()\n"
            "    if lines:\n"
            "        done = subprocess.run([real, *args], input=lines[0] + '\\n', text=True, capture_output=True)\n"
            "        sys.stdout.write(done.stdout or '')\n"
            "        sys.stderr.write(done.stderr or '')\n"
            "        if done.returncode:\n"
            "            raise SystemExit(done.returncode)\n"
            "    sys.stderr.write('database is locked after partial import\\n')\n"
            "    raise SystemExit(1)\n"
            "raise SystemExit(subprocess.run([real, *args], text=True).returncode)\n",
            encoding="utf-8",
        )
        task_wrapper.chmod(0o700)
        config_path = root / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\npanel_mode = "minimal"\n', encoding="utf-8")
        taskrc_path = root / "taskrc"
        taskrc_path.write_text(
            "uda.chainID.type=string\n"
            "uda.chain.type=string\n"
            "uda.link.type=numeric\n"
            "uda.prevLink.type=string\n"
            "uda.nextLink.type=string\n"
            "uda.cp.type=string\n"
            "uda.anchor.type=string\n"
            "uda.anchor_mode.type=string\n",
            encoding="utf-8",
        )
        base_env = os.environ.copy()
        base_env.update(
            {
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": str(ROOT),
                "NAUTICAL_TRUST_CONFIG_PATH": "1",
                "NAUTICAL_TRUST_CORE_PATH": "1",
                "NAUTICAL_TRUST_TASKDATA_PATH": "1",
                "TASKRC": str(taskrc_path),
                "TZ": "UTC",
            }
        )
        fingerprint_taskdata = root / "fingerprint-probe"
        fingerprint_taskdata.mkdir()
        fingerprint_probe = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json, os, nautical_core as core; "
                    "core.reload_taskdata_config(os.environ['TASKDATA']); "
                    "print(json.dumps({'configuration': core.effective_config_fingerprint(), "
                    "'schedule': core.scheduler_config_fingerprint()}))"
                ),
            ],
            text=True,
            capture_output=True,
            env=dict(base_env, TASKDATA=str(fingerprint_taskdata)),
            timeout=30.0,
        )
        if fingerprint_probe.returncode != 0:
            raise RuntimeError(
                "workflow fingerprint probe failed: "
                f"{(fingerprint_probe.stderr or fingerprint_probe.stdout or '').strip()}"
            )
        try:
            workflow_fingerprints = json.loads(fingerprint_probe.stdout or "{}")
            config_fingerprint = str(workflow_fingerprints["configuration"])
            schedule_fingerprint = str(workflow_fingerprints["schedule"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("workflow fingerprint probe returned invalid JSON") from exc
        for key in ("NAUTICAL_DIAG", "NAUTICAL_DIAG_LOG", "NAUTICAL_PROFILE"):
            base_env.pop(key, None)

        completion_cases = (
            ("workflow_cp_completion", "cp", False),
            ("workflow_cp_completion_nonfinal", "cp", True),
            ("workflow_anchor_completion", "anchor", False),
            ("workflow_anchor_completion_nonfinal", "anchor", True),
        )
        results: dict[str, dict] = {}

        ordinary_samples = []
        for sample_index in range(repeats):
            old = _completion_fixture("cp", sample_index, nonfinal=True, mode="ordinary")
            new = dict(old, description=f"Ordinary Nautical edit {sample_index}")
            taskdata = root / f"ordinary-modify-{sample_index}"
            taskdata.mkdir()
            env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1")
            _import_workflow_rows((old,), env=env)
            elapsed, result, _stderr = _run_workflow_hook_result(
                ROOT / "on-modify.nautical",
                input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                env=env,
                expect_output=True,
            )
            if result != new or _workflow_outbox_pending(taskdata):
                raise RuntimeError("workflow_ordinary_modify changed the task or staged work")
            ordinary_samples.append(elapsed)
        results["workflow_ordinary_modify"] = _measure_workflow(
            "workflow_ordinary_modify",
            ordinary_samples,
            float(budgets.get("workflow_ordinary_modify", 2.0)),
        )

        expiration_samples = []
        for sample_index in range(repeats):
            key = f"nautical-perf/expiration/{sample_index}"
            parent_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, key + "/parent"))
            old = {
                "uuid": parent_uuid,
                "status": "pending",
                "description": f"Expiration recovery benchmark {sample_index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": f"expiration-perf-{sample_index:04d}",
                "link": 1,
                "due": "20260101T090000Z",
                "until": "20260101T200000Z",
                # Lifecycle plans require the Taskwarrior modified guard;
                # preserve it in the synthetic hook snapshot just as an
                # exported task would.
                "modified": "20260101T090000Z",
            }
            new = dict(old, status="deleted", end="20260102T090000Z")
            taskdata = root / f"expiration-recovery-{sample_index}"
            taskdata.mkdir()
            env = dict(base_env, TASKDATA=str(taskdata), NAUTICAL_BENCH_FORCE_FULL="1")
            _import_workflow_rows((old,), env=env)
            elapsed, result, _stderr = _run_workflow_hook_result(
                ROOT / "on-modify.nautical",
                input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                env=env,
                expect_output=True,
            )
            staged = _workflow_outbox_pending(taskdata)
            if not isinstance(result, dict) or result.get("chain") != "on" or len(staged) != 1:
                raise RuntimeError("workflow_expiration_recovery did not stage exactly one successor")
            expiration_samples.append(elapsed)
        results["workflow_expiration_recovery"] = _measure_workflow(
            "workflow_expiration_recovery",
            expiration_samples,
            float(budgets.get("workflow_expiration_recovery", 2.0)),
        )

        for name, kind, nonfinal in completion_cases:
            fresh_samples = []
            for sample_index in range(repeats):
                old = _completion_fixture(kind, sample_index, nonfinal=nonfinal, mode="fresh")
                new = dict(old, status="completed", end="20260101T100000Z" if kind == "cp" else "20260105T100000Z")
                taskdata = root / f"{name}-fresh-{sample_index}"
                taskdata.mkdir()
                env = dict(base_env, TASKDATA=str(taskdata))
                _import_workflow_rows((old,), env=env)
                elapsed, result, _stderr = _run_workflow_hook_result(
                    ROOT / "on-modify.nautical",
                    input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                    env=env,
                    expect_output=True,
                )
                if not isinstance(result, dict):
                    raise RuntimeError(f"{name} fresh sample returned no task object")
                queued = _workflow_outbox_pending(taskdata)
                if nonfinal:
                    if result.get("chain") != "on" or len(queued) != 1:
                        raise RuntimeError(f"{name} fresh sample did not queue exactly one child")
                    if queued[0].get("stage") != "planned":
                        raise RuntimeError(f"{name} fresh sample staged an invalid lifecycle record")
                elif result.get("chain") != "off" or queued:
                    raise RuntimeError(f"{name} final sample did not complete without a successor")
                fresh_samples.append(elapsed)
            results[name] = _measure_workflow(name, fresh_samples, float(budgets.get(name, 2.0)))

            if nonfinal:
                idem_name = f"{name}_idempotent"
                idem_samples = []
                for sample_index in range(repeats):
                    old = _completion_fixture(kind, sample_index, nonfinal=True, mode="idempotent")
                    new = dict(old, status="completed", end="20260101T100000Z" if kind == "cp" else "20260105T100000Z")
                    taskdata = root / f"{idem_name}-{sample_index}"
                    taskdata.mkdir()
                    env = dict(base_env, TASKDATA=str(taskdata))
                    _import_workflow_rows((old,), env=env)
                    _import_existing_completion_child(old, env=env)
                    elapsed, result, stderr = _run_workflow_hook_result(
                        ROOT / "on-modify.nautical",
                        input_text=json.dumps(old, ensure_ascii=False) + "\n" + json.dumps(new, ensure_ascii=False),
                        env=env,
                        expect_output=True,
                    )
                    if not isinstance(result, dict) or result.get("chain") != "on":
                        raise RuntimeError(f"{idem_name} sample changed the completed parent unexpectedly")
                    if _workflow_outbox_pending(taskdata):
                        raise RuntimeError(f"{idem_name} sample staged a duplicate child")
                    if "Spawn skipped" not in stderr:
                        raise RuntimeError(f"{idem_name} sample did not report the existing next link")
                    idem_samples.append(elapsed)
                results[idem_name] = _measure_workflow(
                    idem_name,
                    idem_samples,
                    float(budgets.get(idem_name, 2.0)),
                )

        queue_samples = []
        queue_call_stats: list[dict[str, int]] = []
        queue_idempotent_samples = []
        queue_idempotent_call_stats: list[dict[str, int]] = []
        queue_partial_samples = []
        queue_partial_first_samples = []
        queue_partial_recovery_samples = []
        queue_partial_call_stats: list[dict[str, int]] = []
        for sample_index in range(repeats):
            queue_data = root / f"populated-queue-{sample_index}"
            _init_empty_outbox(queue_data)
            stats_path = queue_data / "on-exit-task-stats.json"
            queue_env = dict(base_env, TASKDATA=str(queue_data), NAUTICAL_BENCH_STATS_FILE=str(stats_path))
            queue_env["NAUTICAL_BENCH_TASK_BIN"] = str(task_wrapper)
            if os.environ.get("NAUTICAL_DIAG") == "1":
                queue_env["NAUTICAL_DIAG"] = "1"
            parents, queue_plans = _outbox_lifecycle_fixture("queue", sample_index)
            parent_uuids = {str(parent["uuid"]) for parent in parents}
            child_uuids = {str(plan.child_dict()["uuid"]) for plan in queue_plans}

            import_proc = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(parent, ensure_ascii=False) + "\n" for parent in parents),
                text=True,
                capture_output=True,
                env=queue_env,
                timeout=30.0,
            )
            if import_proc.returncode != 0:
                raise RuntimeError(
                    "queue drain parent fixture import failed: "
                    f"{(import_proc.stderr or import_proc.stdout or '').strip()}"
                )
            preload_probe = subprocess.run(
                [
                    "task", f"rc.data.location={queue_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on",
                    "(", "status:completed", "or", "status:deleted", "or", "status:pending", "or",
                    "status:waiting", ")", "export",
                ],
                text=True,
                capture_output=True,
                env=queue_env,
                timeout=30.0,
            )
            preload_rows = json.loads(preload_probe.stdout or "[]")
            if (
                preload_probe.returncode != 0
                or not isinstance(preload_rows, list)
                or {str(row.get("uuid") or "") for row in preload_rows if isinstance(row, dict)} != parent_uuids
            ):
                raise RuntimeError(
                    "queue drain benchmark parent preflight was not visible to Taskwarrior: "
                    f"{(preload_probe.stderr or preload_probe.stdout or '').strip()}"
                )
            queue_plans = _bind_workflow_plans_to_parents(queue_plans, preload_rows)
            _stage_workflow_plans(
                queue_data,
                queue_plans,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
            )

            queue_elapsed, _queue_result, _queue_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical",
                input_text="",
                env=queue_env,
                expect_output=False,
            )
            queue_samples.append(queue_elapsed)
            queue_call_stats.append(_read_exit_task_call_stats(stats_path))

            if _workflow_outbox_pending(queue_data):
                raise RuntimeError(
                    "outbox drain benchmark left active intents after successful processing: "
                    f"{_workflow_outbox_pending(queue_data)!r}; stderr={_queue_stderr.strip()!r}"
                )

            # Acknowledged outbox records are terminal and must not trigger
            # another Taskwarrior read or mutation on a replay drain.
            try:
                stats_path.unlink()
            except FileNotFoundError:
                pass
            idem_elapsed, _idem_result, _idem_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical",
                input_text="",
                env=dict(queue_env, NAUTICAL_BENCH_FORCE_FULL="1"),
                expect_output=False,
            )
            queue_idempotent_samples.append(idem_elapsed)
            queue_idempotent_call_stats.append(_read_exit_task_call_stats(stats_path))
            if _workflow_outbox_pending(queue_data):
                raise RuntimeError(
                    "idempotent outbox drain left active intents: "
                    f"{_workflow_outbox_pending(queue_data)!r}; stderr={_idem_stderr.strip()!r}"
                )
            export_proc = subprocess.run(
                [
                    "task",
                    "rc.hooks=off",
                    "rc.json.array=1",
                    "rc.verbose=nothing",
                    "rc.color=off",
                    "chain:on",
                    "export",
                ],
                text=True,
                capture_output=True,
                env=queue_env,
                timeout=30.0,
            )
            if export_proc.returncode != 0:
                raise RuntimeError(
                    "queue drain benchmark export failed: "
                    f"{(export_proc.stderr or export_proc.stdout or '').strip()}"
                )
            try:
                exported = json.loads(export_proc.stdout or "[]")
            except json.JSONDecodeError as exc:
                raise RuntimeError("queue drain benchmark export was not valid JSON") from exc
            if not isinstance(exported, list) or len(exported) != 16:
                raise RuntimeError(
                    "queue drain benchmark did not retain 8 parents and import 8 children: "
                    f"{len(exported) if isinstance(exported, list) else type(exported).__name__} tasks; "
                    f"outbox={lifecycle_outbox.LifecycleOutboxRepository(queue_data).status(limit=20)[1]!r}"
                )
            children = [
                row
                for row in exported
                if isinstance(row, dict) and str(row.get("uuid") or "") in child_uuids
            ]
            if len(children) != 8 or any(not str(row.get("prevLink") or "").strip() for row in children):
                raise RuntimeError("queue drain benchmark did not import/link all child tasks")
            parents_after = [
                row
                for row in exported
                if isinstance(row, dict) and str(row.get("uuid") or "") in parent_uuids
            ]
            if len(parents_after) != 8 or any(not str(row.get("nextLink") or "").strip() for row in parents_after):
                raise RuntimeError("queue drain benchmark did not update all parent nextLink values")

            # Exercise a partial batch import: the runner imports only the
            # first child and then fails. The next invocation must requeue,
            # discover that child, import the remaining seven, and converge.
            partial_data = root / f"partial-queue-{sample_index}"
            _init_empty_outbox(partial_data)
            partial_stats_path = partial_data / "on-exit-task-stats.json"
            partial_env = dict(base_env, TASKDATA=str(partial_data), NAUTICAL_BENCH_STATS_FILE=str(partial_stats_path))
            partial_env["NAUTICAL_BENCH_TASK_BIN"] = str(task_wrapper)
            partial_parents, partial_plans = _outbox_lifecycle_fixture("partial", sample_index)
            partial_import_proc = subprocess.run(
                ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
                input="".join(json.dumps(parent, ensure_ascii=False) + "\n" for parent in partial_parents),
                text=True,
                capture_output=True,
                env=partial_env,
                timeout=30.0,
            )
            if partial_import_proc.returncode != 0:
                raise RuntimeError("partial queue parent fixture import failed")
            partial_parent_probe = subprocess.run(
                ["task", f"rc.data.location={partial_data}", "rc.hooks=off", "rc.json.array=1", "rc.verbose=nothing", "chain:on", "export"],
                text=True,
                capture_output=True,
                env=partial_env,
                timeout=30.0,
            )
            if partial_parent_probe.returncode != 0:
                raise RuntimeError("partial queue parent verification export failed")
            partial_plans = _bind_workflow_plans_to_parents(
                partial_plans,
                json.loads(partial_parent_probe.stdout or "[]"),
            )
            _stage_workflow_plans(
                partial_data,
                partial_plans,
                configuration_fingerprint=config_fingerprint,
                schedule_fingerprint=schedule_fingerprint,
            )
            partial_env["NAUTICAL_BENCH_FAIL_MODE"] = "partial-import"
            try:
                partial_stats_path.unlink()
            except FileNotFoundError:
                pass
            partial_t0 = time.perf_counter()
            _first_partial_elapsed, _first_partial_result, first_partial_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical", input_text="", env=partial_env, expect_output=False
            )
            first_partial_elapsed = time.perf_counter() - partial_t0
            first_partial_stats = _read_exit_task_call_stats(partial_stats_path)
            pending_after_partial = _workflow_outbox_pending(partial_data)
            if not 1 <= len(pending_after_partial) <= 8:
                raise RuntimeError(
                    "partial import did not requeue all lifecycle intents: "
                    f"{pending_after_partial!r}; stderr={first_partial_stderr.strip()!r}"
                )
            partial_env.pop("NAUTICAL_BENCH_FAIL_MODE", None)
            try:
                partial_stats_path.unlink()
            except FileNotFoundError:
                pass
            recovery_t0 = time.perf_counter()
            _second_partial_elapsed, _second_partial_result, second_partial_stderr = _run_workflow_hook_result(
                ROOT / "on-exit.nautical", input_text="", env=partial_env, expect_output=False
            )
            recovery_elapsed = time.perf_counter() - recovery_t0
            second_partial_stats = _read_exit_task_call_stats(partial_stats_path)
            if _workflow_outbox_pending(partial_data):
                raise RuntimeError(
                    "partial import recovery left active lifecycle intents: "
                    f"{_workflow_outbox_pending(partial_data)!r}; stderr={second_partial_stderr.strip()!r}"
                )
            queue_partial_first_samples.append(first_partial_elapsed)
            queue_partial_recovery_samples.append(recovery_elapsed)
            queue_partial_samples.append(first_partial_elapsed + recovery_elapsed)
            queue_partial_call_stats.append(_merge_task_call_stats(first_partial_stats, second_partial_stats))
        queue_result = _measure_workflow(
            "workflow_queue_drain", queue_samples,
            float(budgets.get("workflow_queue_drain", 3.0)),
        )
        queue_result["task_call_stats"] = queue_call_stats
        call_budgets = workflow_cfg.get("task_call_budgets")
        if isinstance(call_budgets, dict):
            _apply_task_call_budgets(
                queue_result,
                queue_call_stats,
                call_budgets.get("workflow_queue_drain", {}),
            )
        results["workflow_queue_drain"] = queue_result
        queue_idempotent_result = _measure_workflow(
            "workflow_queue_drain_idempotent",
            queue_idempotent_samples,
            float(budgets.get("workflow_queue_drain_idempotent", 3.0)),
        )
        queue_idempotent_result["task_call_stats"] = queue_idempotent_call_stats
        if isinstance(call_budgets, dict):
            _apply_task_call_budgets(
                queue_idempotent_result,
                queue_idempotent_call_stats,
                call_budgets.get("workflow_queue_drain_idempotent", {}),
            )
        results["workflow_queue_drain_idempotent"] = queue_idempotent_result
        queue_partial_result = _measure_workflow(
            "workflow_queue_drain_partial_recovery",
            queue_partial_samples,
            float(budgets.get("workflow_queue_drain_partial_recovery", 6.0)),
        )
        queue_partial_result["first_attempt_samples_s"] = sorted(queue_partial_first_samples)
        queue_partial_result["first_attempt_median_s"] = float(statistics.median(queue_partial_first_samples))
        queue_partial_result["recovery_samples_s"] = sorted(queue_partial_recovery_samples)
        queue_partial_result["recovery_median_s"] = float(statistics.median(queue_partial_recovery_samples))
        queue_partial_result["task_call_stats"] = queue_partial_call_stats
        if isinstance(call_budgets, dict):
            _apply_task_call_budgets(
                queue_partial_result,
                queue_partial_call_stats,
                call_budgets.get("workflow_queue_drain_partial_recovery", {}),
            )
        results["workflow_queue_drain_partial_recovery"] = queue_partial_result

        reconcile_data = root / "reconcile"
        reconcile_data.mkdir()
        reconcile_env = dict(base_env, TASKDATA=str(reconcile_data))
        history_rows = max(1, int(workflow_cfg.get("reconcile_history_rows", 256)))
        reconcile_tasks = []
        for link in range(1, history_rows + 1):
            task = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile/{link}")),
                "status": "completed",
                "description": f"Reconcile performance benchmark {link}",
                "cp": "P1D",
                "chain": "on",
                "chainID": "reconcile-perf-chain",
                "link": link,
                "due": f"202601{min(link, 28):02d}T090000Z",
            }
            if link > 1:
                task["prevLink"] = reconcile_tasks[-1]["uuid"][:8]
            if link < history_rows:
                task["nextLink"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile/{link + 1}"))[:8]
            reconcile_tasks.append(task)
        import_proc = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in reconcile_tasks),
            text=True,
            capture_output=True,
            env=reconcile_env,
            timeout=30.0,
        )
        if import_proc.returncode != 0:
            raise RuntimeError(f"reconcile fixture import failed: {(import_proc.stderr or import_proc.stdout or '').strip()}")
        reconcile_cmd = [sys.executable, str(ROOT / "nautical_core" / "tools" / "nautical_reconcile.py"), "--json"]
        reconcile_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=reconcile_env, timeout=30.0)
            if proc.returncode != 0:
                raise RuntimeError(f"reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
            report = json.loads(proc.stdout or "{}")
            if (
                not isinstance(report, dict)
                or int(report.get("export_calls", 0)) != 1
                or int(report.get("export_rows", 0)) != history_rows
            ):
                raise RuntimeError("healthy reconcile workflow did not use exactly one broad snapshot")
            reconcile_samples.append(time.perf_counter() - started)
        results["workflow_reconcile"] = _measure_workflow(
            "workflow_reconcile", reconcile_samples, float(budgets.get("workflow_reconcile", 3.0))
        )

        # Keep an empty audit as a first-class workload.  This catches startup,
        # snapshot, and report overhead without accidentally measuring stale
        # queue cleanup or a failed Taskwarrior export.
        empty_data = root / "reconcile-empty"
        empty_data.mkdir()
        empty_env = dict(base_env, TASKDATA=str(empty_data))
        empty_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=empty_env, timeout=30.0)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"empty reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("empty reconcile workflow returned invalid JSON") from exc
            if not isinstance(report, dict) or report.get("schema") != "nautical.reconcile":
                raise RuntimeError("empty reconcile workflow returned an invalid report")
            if int(report.get("export_calls", 0)) != 1:
                raise RuntimeError("empty reconcile workflow did not use exactly one broad snapshot")
            empty_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_empty"] = _measure_workflow(
            "workflow_reconcile_empty",
            empty_samples,
            float(budgets.get("workflow_reconcile_empty", budgets.get("workflow_reconcile", 3.0))),
        )

        # Candidate-heavy audits must prove that the benchmark contains
        # actionable integrity evidence.  A zero-candidate run would only
        # measure the healthy path again.
        candidate_data = root / "reconcile-candidates"
        candidate_data.mkdir()
        candidate_env = dict(base_env, TASKDATA=str(candidate_data))
        candidate_count = max(1, int(workflow_cfg.get("reconcile_candidate_chains", 32)))
        candidate_tasks = [
            {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-candidate/{index}")),
                "status": "completed",
                "description": f"Reconcile candidate benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": f"reconcile-candidate-{index}",
                "link": 1,
                "due": "20260101T090000Z",
            }
            for index in range(candidate_count)
        ]
        candidate_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in candidate_tasks),
            text=True,
            capture_output=True,
            env=candidate_env,
            timeout=30.0,
        )
        if candidate_import.returncode != 0:
            raise RuntimeError(
                "reconcile candidate fixture import failed: "
                f"{(candidate_import.stderr or candidate_import.stdout or '').strip()}"
            )
        candidate_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=candidate_env, timeout=30.0)
            if proc.returncode not in (0, 1, 2):
                raise RuntimeError(
                    f"candidate reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("candidate reconcile workflow returned invalid JSON") from exc
            summary = report if isinstance(report, dict) else {}
            if int(summary.get("candidates", 0)) <= 0 and not summary.get("plans"):
                raise RuntimeError("candidate reconcile workflow produced no integrity candidates or plans")
            candidate_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_candidates"] = _measure_workflow(
            "workflow_reconcile_candidates",
            candidate_samples,
            float(budgets.get("workflow_reconcile_candidates", budgets.get("workflow_reconcile", 3.0))),
        )

        long_data = root / "reconcile-long-history"
        long_data.mkdir()
        long_env = dict(base_env, TASKDATA=str(long_data))
        long_count = max(history_rows, int(workflow_cfg.get("reconcile_long_history_rows", 2048)))
        long_tasks = []
        for link in range(1, long_count + 1):
            task = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-long/{link}")),
                "status": "completed",
                "description": f"Long reconcile benchmark {link}",
                "cp": "P1D",
                "chain": "on",
                "chainID": "reconcile-long-chain",
                "link": link,
                "due": f"202601{min(link, 28):02d}T090000Z",
            }
            if link > 1:
                task["prevLink"] = long_tasks[-1]["uuid"][:8]
            if link < long_count:
                task["nextLink"] = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-long/{link + 1}"))[:8]
            long_tasks.append(task)
        long_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in long_tasks),
            text=True,
            capture_output=True,
            env=long_env,
            timeout=60.0,
        )
        if long_import.returncode != 0:
            raise RuntimeError(
                "long reconcile fixture import failed: "
                f"{(long_import.stderr or long_import.stdout or '').strip()}"
            )
        long_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=long_env, timeout=60.0)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"long reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("long reconcile workflow returned invalid JSON") from exc
            if not isinstance(report, dict) or report.get("schema") != "nautical.reconcile":
                raise RuntimeError("long reconcile workflow returned an invalid report")
            if int(report.get("export_calls", 0)) != 1 or int(report.get("export_rows", 0)) != long_count:
                raise RuntimeError("long reconcile workflow exceeded its single-snapshot row budget")
            long_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_long_history"] = _measure_workflow(
            "workflow_reconcile_long_history",
            long_samples,
            float(budgets.get("workflow_reconcile_long_history", budgets.get("workflow_reconcile", 3.0))),
        )

        corrupt_data = root / "reconcile-corrupted"
        corrupt_data.mkdir()
        corrupt_env = dict(base_env, TASKDATA=str(corrupt_data))
        corrupt_tasks = []
        corrupt_count = max(1, int(workflow_cfg.get("reconcile_corrupted_chains", 16)))
        for index in range(corrupt_count):
            chain_id = f"reconcile-corrupt-{index}"
            first_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-corrupt/{index}/first"))
            second_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-corrupt/{index}/second"))
            common = {
                "status": "pending",
                "description": f"Corrupted reconcile benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": chain_id,
                "link": 1,
                "due": "20260101T090000Z",
            }
            corrupt_tasks.extend([
                dict(common, uuid=first_uuid, nextLink=second_uuid[:8]),
                dict(common, uuid=second_uuid, prevLink=first_uuid[:8]),
            ])
        corrupt_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in corrupt_tasks),
            text=True,
            capture_output=True,
            env=corrupt_env,
            timeout=30.0,
        )
        if corrupt_import.returncode != 0:
            raise RuntimeError(
                "corrupted reconcile fixture import failed: "
                f"{(corrupt_import.stderr or corrupt_import.stdout or '').strip()}"
            )
        corrupt_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=corrupt_env, timeout=30.0)
            if proc.returncode not in (0, 1, 2):
                raise RuntimeError(
                    f"corrupted reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}"
                )
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("corrupted reconcile workflow returned invalid JSON") from exc
            audit = report.get("integrity_audit") if isinstance(report, dict) else None
            if not isinstance(audit, dict) or not audit.get("findings"):
                raise RuntimeError("corrupted reconcile workflow hid its integrity findings")
            corrupt_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_corrupted"] = _measure_workflow(
            "workflow_reconcile_corrupted",
            corrupt_samples,
            float(budgets.get("workflow_reconcile_corrupted", budgets.get("workflow_reconcile", 3.0))),
        )

        mixed_data = root / "reconcile-mixed"
        mixed_data.mkdir()
        mixed_env = dict(base_env, TASKDATA=str(mixed_data))
        mixed_tasks = []
        mixed_healthy_count = max(2, int(workflow_cfg.get("reconcile_mixed_healthy_rows", 8)))
        healthy_rows = []
        for link in range(1, mixed_healthy_count + 1):
            row = {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/healthy/{link}")),
                "status": "completed",
                "description": f"Mixed healthy benchmark {link}",
                "cp": "P1D",
                "chain": "on",
                "chainID": "reconcile-mixed-healthy",
                "link": link,
                "due": "20260101T090000Z",
            }
            if healthy_rows:
                row["prevLink"] = healthy_rows[-1]["uuid"][:8]
            healthy_rows.append(row)
        for index, row in enumerate(healthy_rows):
            if index + 1 < len(healthy_rows):
                row["nextLink"] = healthy_rows[index + 1]["uuid"][:8]
        mixed_tasks.extend(healthy_rows)
        for index in range(max(1, int(workflow_cfg.get("reconcile_mixed_candidates", 8)))):
            mixed_tasks.append({
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/candidate/{index}")),
                "status": "completed",
                "description": f"Mixed candidate benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": f"reconcile-mixed-candidate-{index}",
                "link": 1,
                "due": "20260101T090000Z",
            })
        for index in range(max(1, int(workflow_cfg.get("reconcile_mixed_corrupted", 4)))):
            chain_id = f"reconcile-mixed-corrupt-{index}"
            first_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/corrupt/{index}/first"))
            second_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/reconcile-mixed/corrupt/{index}/second"))
            common = {
                "status": "pending",
                "description": f"Mixed corrupted benchmark {index}",
                "cp": "P1D",
                "chain": "on",
                "chainID": chain_id,
                "link": 1,
                "due": "20260101T090000Z",
            }
            mixed_tasks.extend([
                dict(common, uuid=first_uuid, nextLink=second_uuid[:8]),
                dict(common, uuid=second_uuid, prevLink=first_uuid[:8]),
            ])
        mixed_import = subprocess.run(
            ["task", "rc.hooks=off", "rc.verbose=nothing", "import"],
            input="".join(json.dumps(task, ensure_ascii=False) + "\n" for task in mixed_tasks),
            text=True,
            capture_output=True,
            env=mixed_env,
            timeout=30.0,
        )
        if mixed_import.returncode != 0:
            raise RuntimeError(
                "mixed reconcile fixture import failed: "
                f"{(mixed_import.stderr or mixed_import.stdout or '').strip()}"
            )
        mixed_samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            proc = subprocess.run(reconcile_cmd, text=True, capture_output=True, env=mixed_env, timeout=30.0)
            if proc.returncode not in (0, 1, 2):
                raise RuntimeError(f"mixed reconcile workflow failed: {(proc.stderr or proc.stdout or '').strip()}")
            try:
                report = json.loads(proc.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("mixed reconcile workflow returned invalid JSON") from exc
            audit = report.get("integrity_audit") if isinstance(report, dict) else None
            if not isinstance(audit, dict) or not audit.get("findings") or int(report.get("candidates", 0)) <= 0:
                raise RuntimeError("mixed reconcile workflow did not preserve both candidate and integrity evidence")
            mixed_samples.append(time.perf_counter() - started)
        results["workflow_reconcile_mixed"] = _measure_workflow(
            "workflow_reconcile_mixed",
            mixed_samples,
            float(budgets.get("workflow_reconcile_mixed", budgets.get("workflow_reconcile", 3.0))),
        )
        return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-file", default=str(HERE / "perf_budget.json"))
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON summary")
    ap.add_argument("--enforce", action="store_true", help="fail non-zero if any budget is exceeded")
    ap.add_argument(
        "--extended",
        action="store_true",
        help="run large-file, astronomy, omission, and native-until benchmarks",
    )
    ap.add_argument(
        "--slow-device",
        action="store_true",
        help="use the slower-device budgets for hint and extended benchmarks",
    )
    ap.add_argument(
        "--workflows-only",
        action="store_true",
        help="run only expensive completion, queue, and reconcile workflows",
    )
    args = ap.parse_args()

    cfg = _load_budget_config(Path(args.budget_file))
    workload = cfg["workload"]
    budgets = cfg["budgets_seconds"]
    if args.slow_device and isinstance(cfg.get("slow_device_budgets_seconds"), dict):
        budgets = {**budgets, **cfg["slow_device_budgets_seconds"]}
    exprs = [str(x) for x in workload.get("expressions", []) if str(x).strip()]
    if not exprs:
        raise RuntimeError("No expressions defined in workload.expressions")

    repeats = int(workload.get("repeats", 5))
    parse_rounds = int(workload.get("parse_validate_rounds", 220))
    describe_rounds = int(workload.get("describe_expr_rounds", 220))
    next_after_rounds = int(workload.get("next_after_rounds", 220))
    hints_rounds = int(workload.get("build_hints_rounds", 180))
    hints_cold_rounds = max(1, int(workload.get("build_hints_cold_rounds", 1)))
    hints_warm_rounds = max(1, int(workload.get("build_hints_warm_rounds", hints_rounds)))
    cache_key_rounds = int(workload.get("cache_key_rounds", 2500))
    cache_save_rounds = int(workload.get("cache_save_rounds", 120))
    cache_load_rounds = int(workload.get("cache_load_rounds", 300))
    outbox_schema_hot_rounds = int(workload.get("outbox_schema_hot_rounds", 1000))
    outbox_schema_cold_rounds = int(workload.get("outbox_schema_cold_rounds", 3))
    anchor_file_rounds = int(workload.get("anchor_file_rounds", 300))
    cold_import_rounds = int(workload.get("cold_import_rounds", 3))

    checks = [
        ("cold_core_import", lambda: _bench_cold_import("core", cold_import_rounds), repeats),
        (
            "cold_modify_impl_import",
            lambda: _bench_cold_import("modify_impl", cold_import_rounds),
            repeats,
        ),
        ("parse_validate", lambda: _bench_parse_validate(exprs, parse_rounds), repeats),
        ("describe_expr", lambda: _bench_describe_expr(exprs, describe_rounds), repeats),
        ("next_after", lambda: _bench_next_after(exprs, next_after_rounds), repeats),
        (
            "build_hints_cold",
            lambda: _bench_build_hints(exprs, hints_cold_rounds, mode="cold"),
            repeats,
        ),
        (
            "build_hints_warm",
            lambda: _bench_build_hints(exprs, hints_warm_rounds, mode="warm"),
            repeats,
        ),
        ("cache_key_hot", lambda: _bench_cache_key_hot(exprs, cache_key_rounds), repeats),
        ("cache_save", lambda: _bench_cache_save(exprs, cache_save_rounds), repeats),
        ("cache_load_hot", lambda: _bench_cache_load_hot(exprs, cache_load_rounds), repeats),
        ("outbox_schema_hot", lambda: _bench_outbox_schema_hot(outbox_schema_hot_rounds), repeats),
        ("outbox_schema_cold", lambda: _bench_outbox_schema_cold(outbox_schema_cold_rounds), repeats),
        ("anchor_file_provider", lambda: _bench_anchor_file_provider(anchor_file_rounds), repeats),
        ("anchor_file_batch_provider", lambda: _bench_anchor_file_batch_provider(anchor_file_rounds), repeats),
    ]
    if args.workflows_only:
        checks = []

    extended = cfg.get("extended_workload")
    if args.extended and isinstance(extended, dict) and extended.get("enabled", True):
        extended_repeats = max(1, int(extended.get("repeats", 2)))
        extended_rows = max(1000, int(extended.get("anchor_file_rows", 5000)))
        extended_rounds = max(1, int(extended.get("rounds", 8)))
        extended_budgets_key = "slow_device_budgets_seconds" if args.slow_device else "budgets_seconds"
        extended_budgets = extended.get(extended_budgets_key)
        if not isinstance(extended_budgets, dict):
            extended_budgets = extended.get("budgets_seconds")
        if not isinstance(extended_budgets, dict):
            extended_budgets = {}

        checks.extend(
            [
                (
                    "anchor_file_large_cold",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="cold",
                    ),
                    extended_repeats,
                ),
                (
                    "anchor_file_large_hot",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="hot",
                    ),
                    extended_repeats,
                ),
                (
                    "anchor_file_nonmonotonic",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="nonmonotonic",
                    ),
                    extended_repeats,
                ),
                (
                    "anchor_file_business_day_omissions",
                    lambda: _bench_large_anchor_file_provider(
                        extended_rounds,
                        row_count=extended_rows,
                        mode="hot",
                        business_day_only=True,
                    ),
                    extended_repeats,
                ),
                (
                    "business_calendar_large_omissions",
                    lambda: _bench_business_calendar_omissions(extended_rounds * 40),
                    extended_repeats,
                ),
                (
                    "native_until_reconcile_dry_run",
                    lambda: _bench_native_until_reconcile(extended_repeats, apply=False),
                    1,
                ),
                (
                    "native_until_reconcile_apply",
                    lambda: _bench_native_until_reconcile(extended_repeats, apply=True),
                    1,
                ),
            ]
        )
        if importlib.util.find_spec("astral") is not None:
            checks.extend(
                [
                    (
                        "astronomy_anchor_add",
                        lambda: _bench_astronomy_provider(extended_rounds, event="sunrise") or 0.0,
                        extended_repeats,
                    ),
                    (
                        "astronomy_anchor_completion",
                        lambda: _bench_astronomy_provider(extended_rounds, event="moonrise") or 0.0,
                        extended_repeats,
                    ),
                ]
            )

    seasonal = cfg.get("seasonal_workload")
    if not args.workflows_only and isinstance(seasonal, dict):
        seasonal_exprs = [
            str(value)
            for value in seasonal.get("expressions", [])
            if str(value).strip()
        ]
        if seasonal_exprs:
            seasonal_repeats = max(1, int(seasonal.get("repeats", 3)))
            seasonal_parse_rounds = max(1, int(seasonal.get("parse_validate_rounds", 100)))
            seasonal_next_rounds = max(1, int(seasonal.get("next_after_rounds", 100)))
            seasonal_hint_rounds = max(1, int(seasonal.get("build_hints_rounds", 1)))
            seasonal_cold_rounds = max(1, int(seasonal.get("build_hints_cold_rounds", 1)))
            seasonal_warm_rounds = max(1, int(seasonal.get("build_hints_warm_rounds", seasonal_hint_rounds)))
            checks.extend(
                [
                    (
                        "seasonal_parse_validate",
                        lambda: _bench_parse_validate(
                            seasonal_exprs,
                            seasonal_parse_rounds,
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_next_after",
                        lambda: _bench_next_after(
                            seasonal_exprs,
                            seasonal_next_rounds,
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_build_hints_cold",
                        lambda: _bench_build_hints(
                            seasonal_exprs,
                            seasonal_cold_rounds,
                            mode="cold",
                        ),
                        seasonal_repeats,
                    ),
                    (
                        "seasonal_build_hints_warm",
                        lambda: _bench_build_hints(
                            seasonal_exprs,
                            seasonal_warm_rounds,
                            mode="warm",
                        ),
                        seasonal_repeats,
                    ),
                ]
            )

    results = {}
    failures = []
    for name, fn, check_repeats in checks:
        r = _measure(name, fn, check_repeats)
        extended_budgets = {}
        if args.extended and isinstance(cfg.get("extended_workload"), dict):
            profile_key = "slow_device_budgets_seconds" if args.slow_device else "budgets_seconds"
            candidate = cfg["extended_workload"].get(profile_key)
            if isinstance(candidate, dict):
                extended_budgets = candidate
        budget = float(extended_budgets.get(name, budgets.get(name, 0.0)))
        r["budget_s"] = budget
        if name == "cold_core_import":
            r["module_count"] = IMPORT_PROFILES.get("core", 0)
        elif name == "cold_modify_impl_import":
            r["module_count"] = IMPORT_PROFILES.get("modify_impl", 0)
        r["pass"] = (budget <= 0.0) or (r["median_s"] <= budget)
        results[name] = r
        if args.enforce and not r["pass"]:
            failures.append(name)

    hook_results = {} if args.workflows_only else _bench_hook_fast_paths(cfg)
    for name, result in hook_results.items():
        results[name] = result
        if args.enforce and not result["pass"]:
            failures.append(name)

    workflow_results = _bench_expensive_workflows(cfg)
    for name, result in workflow_results.items():
        results[name] = result
        if args.enforce and not result["pass"]:
            failures.append(name)

    summary = {
        "budget_file": str(Path(args.budget_file).resolve()),
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "cwd": os.getcwd(),
        "results": results,
        "enforced": bool(args.enforce),
        "ok": len(failures) == 0,
        "failed_checks": failures,
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), indent=2))
    else:
        print("Nautical Performance Budget")
        print(f"Budget file: {summary['budget_file']}")
        for name, r in results.items():
            status = "OK" if r["pass"] else "FAIL"
            if "fast_to_full_ratio" in r:
                print(
                    f"- {name}: fast={r['median_s']:.4f}s full={r['full_median_s']:.4f}s "
                    f"ratio={r['fast_to_full_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            elif "managed_to_source_ratio" in r:
                print(
                    f"- {name}: managed={r['median_s']:.4f}s source={r['baseline_median_s']:.4f}s "
                    f"ratio={r['managed_to_source_ratio']:.3f} max_ratio={r['max_ratio']:.3f} => {status}"
                )
            else:
                print(
                    f"- {name}: median={r['median_s']:.4f}s "
                    f"(min={r['min_s']:.4f}s, max={r['max_s']:.4f}s) "
                    f"budget={r['budget_s']:.4f}s => {status}"
                )
        if args.enforce:
            print("Enforced:", "PASS" if summary["ok"] else f"FAIL ({', '.join(failures)})")

    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
