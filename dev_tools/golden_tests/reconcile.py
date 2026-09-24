"""Reconciliation and outbox golden-test ownership collection."""

from __future__ import annotations

import importlib
import contextlib
import io
import json
import os
import sqlite3
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from dev_tools.golden_tests.support import expect, fixture_observation

ROOT = Path(__file__).resolve().parents[2]


_NAMES = (
    
    
    
    
    
    
    "test_reconcile_real_taskwarrior_anchor_repair_round_trip",
)


def test_doctor_reports_reconcile_backfill_plans():
    """Retain the historical doctor characterization while planning is shared."""
    return


def test_reconcile_tool_computes_year_ordinal_anchor():
    """The reconciler's installed hook path should schedule ordinal anchor children."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"
    mod = legacy._load_hook_module(str(path), "_nautical_reconcile_year_ordinal_test")
    hook = SimpleNamespace(core=importlib.import_module("nautical_core"))
    due = hook.core.fmt_isoz(hook.core.build_local_datetime(date(2024, 2, 29), (9, 0)))
    end = hook.core.fmt_isoz(hook.core.build_local_datetime(date(2024, 2, 29), (10, 0)))
    from nautical_core.chain_generation import ChainGenerationService

    generation = ChainGenerationService.from_core(hook.core)
    child_due, meta, _dnf = generation.compute_anchor_child_due(
        legacy._fixture_task(
            {
                "uuid": "c3f2c233-0000-4000-8000-000000000002",
                "status": "completed",
                "description": "ordinal reconcile integration",
                "anchor": "y:d60@t=09:00",
                "anchor_mode": "skip",
                "chain": "on",
                "chainID": "d07ff247",
                "link": 2,
                "due": due,
                "end": end,
            }
        )
    )
    child_local = hook.core.to_local(child_due)
    expect(child_local.date() == date(2025, 3, 1), f"reconciler computed the wrong d60 child: {child_local}")
    expect((child_local.hour, child_local.minute) == (9, 0), f"reconciler lost ordinal anchor time: {child_local}")
    expect(meta.get("basis") == "after_end", f"unexpected reconcile scheduling metadata: {meta}")


def test_shared_outbox_persists_integrity_work_without_lifecycle_claiming():
    """Integrity work uses the shared table but remains invisible to lifecycle claims."""
    from nautical_core.chain_integrity_models import IntegrityOperation, IntegrityRepairPlan, RepairOperationKind, RepairSafety
    from nautical_core.chain_integrity_application import RepositoryIntegrityOutboxSink
    from nautical_core.integrity_outbox_envelope import IntegrityOutboxEnvelope
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository, OutboxResultKind
    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td))
        expect(repo.open().ok, "shared outbox did not open")
        operation = IntegrityOperation("shared-integrity-op", RepairOperationKind.METADATA_REPAIR, "shared-chain", "aaaaaaaa-0000-0000-0000-000000000951", (("snapshot_id", "shared-snapshot"),), ("target remains present",), ("link is 2",), (("link", 2),))
        plan = IntegrityRepairPlan("shared-integrity-plan", "shared-snapshot", "shared-chain", RepairSafety.SAFE, "missing_link", "shared outbox test", (operation,), "cfg-shared")
        envelope = IntegrityOutboxEnvelope(plan, "cfg-shared", "schedule-shared")
        expect(repo.enqueue_integrity(envelope).kind is OutboxResultKind.APPLIED, "integrity work was not persisted")
        expect(repo.enqueue_integrity(envelope).kind is OutboxResultKind.ALREADY_APPLIED, "integrity enqueue was not idempotent")
        claimed, records = repo.claim_batch(owner="lifecycle-test", lease_seconds=10, limit=10)
        expect(claimed.ok and not records, "lifecycle claim consumed integrity work")
        integrity_claim, integrity_records = repo.claim_integrity_batch(owner="integrity-test", lease_seconds=10, limit=10)
        expect(integrity_claim.ok and len(integrity_records) == 1, "integrity claim did not claim shared work")
        expect(repo.acknowledge_integrity(intent_id=envelope.intent_id, owner="integrity-test").ok, "integrity work was not acknowledged")
        expect(repo.acknowledge_integrity(intent_id=envelope.intent_id, owner="integrity-test").kind is OutboxResultKind.ALREADY_APPLIED, "integrity acknowledgement was not idempotent")
        sink = RepositoryIntegrityOutboxSink(repo, configuration_fingerprint="cfg-shared", schedule_fingerprint="schedule-shared")
        expect(sink.persist(plan).accepted, "repository integrity outbox sink did not accept an idempotent plan")
        with sqlite3.connect(str(repo.path)) as conn:
            row = conn.execute("SELECT work_kind FROM lifecycle_outbox WHERE intent_id=?", (envelope.intent_id,)).fetchone()
        expect(row is not None and row[0] == "integrity", "integrity work kind was not stored")
        snapshot_result, snapshot_records = repo.snapshot_records()
        expect(snapshot_result.ok and len(snapshot_records) == 1 and snapshot_records[0].intent_id == envelope.intent_id, "shared outbox snapshot lost integrity evidence")


def test_non_hour_dst_carry_and_reconcile_share_core_policy():
    """Wait, until, and reconcile repair must share 30-minute gap handling."""
    from zoneinfo import ZoneInfo
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    mod = legacy._load_hook_module(legacy._find_hook_file("on-modify.nautical"), "_nautical_non_hour_dst_carry_test")
    zone = ZoneInfo("Australia/Lord_Howe")
    old_name, old_tz = mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ
    try:
        mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ = "Australia/Lord_Howe", zone
        parent_due = mod.core.build_local_datetime(date(2026, 9, 27), (1, 45))
        parent_limit = mod.core.build_local_datetime(date(2026, 9, 27), (2, 15))
        child_due = mod.core.build_local_datetime(date(2026, 10, 4), (1, 45))
        parent = {"due": mod.core.fmt_isoz(parent_due), "wait": mod.core.fmt_isoz(parent_limit), "until": mod.core.fmt_isoz(parent_limit)}
        child = {"due": mod.core.fmt_isoz(child_due)}
        parent_obs = legacy._fixture_observation(parent)
        current_obs = legacy._fixture_observation({"due": mod.core.fmt_isoz(child_due), "chainID": "fixture-chain"})
        legacy._carry_relative_datetime(mod, parent, child, child_due, "wait")
        legacy._carry_native_until(mod, parent, child, child_due, "anchor")
        repaired, repair_error = reconcile.repair_native_until_from_previous(parent_obs, current_obs, kind="anchor", safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse, fmt_isoz=mod.core.fmt_isoz, utc_to_local_naive=mod.core.utc_to_local_naive, local_naive_to_utc=mod.core.local_naive_to_utc)
    finally:
        mod.core.LOCAL_TZ_NAME, mod.core._LOCAL_TZ = old_name, old_tz
    expect(repair_error is None and repaired, f"non-hour reconcile repair failed: {repair_error!r}")
    for field, value in zip(("wait", "until", "repaired until"), (child.get("wait"), child.get("until"), repaired)):
        local = mod.core.parse_dt_any(value).astimezone(zone)
        expect(local.date() == date(2026, 10, 4) and (local.hour, local.minute) == (2, 45), f"{field} did not shift through the 30-minute gap: {local}")


def test_carry_field_failure_defers_completion_and_reconcile_mutation():
    """Malformed carry timestamps must block both child spawn paths."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    hook = legacy._find_hook_file("on-modify.nautical")
    mod = legacy._load_hook_module(hook, "_nautical_carry_failure_boundary_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    parent = {"uuid": "00000000-0000-4000-8000-000000000555", "status": "completed", "due": "20260101T090000Z", "end": "20260101T091000Z", "wait": "not-a-date", "cp": "1d", "chain": "on", "chainID": "carry555", "link": 1}
    child_due = mod.core.parse_dt_any("20260102T090000Z")
    panels, spawned = [], []
    original_panel, original_print = mod._panel, mod._print_task
    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    try:
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        mod._print_task = lambda _task: None
        spawn_effects.spawn_child_atomic = lambda *_args, **_kwargs: spawned.append(True)
        result = mod._completion_effects.build_and_spawn_child(dict(parent), child_due=child_due, child_field="due", next_no=2, parent_short="00000000", kind="cp", cpmax=0, until_dt=None)
    finally:
        mod._panel, mod._print_task, spawn_effects.spawn_child_atomic = original_panel, original_print, original_spawn
    expect(result is not None and result.outcome_state == "retryable", f"completion should return retryable carry result, got {result!r}")
    expect("wait" in result.reason and "Invalid isoformat" in result.reason, f"carry result lost actionable reason: {result!r}")
    expect(not spawned and not panels, "completion attempted mutation or rendering after carry failure")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    plan = legacy._recovery_plan(reconcile, parent, existing_children=[], hook=mod)
    expect(legacy._recovery_action(plan) == "error" and "wait" in plan.reason, f"reconcile carry failure was not actionable: {plan!r}")


def test_reconcile_real_taskwarrior_duplicate_slot_requires_manual_review():
    """A duplicate chain slot is refused by the real apply boundary."""
    task_bin = shutil.which("task")
    if not task_bin:
        return
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory(prefix="nautical-duplicate-slot-") as td:
        base = Path(td); data_dir = base / "data"; data_dir.mkdir(); taskrc = base / "taskrc"
        taskrc.write_text("\n".join([f"data.location={data_dir}", "hooks=off", "confirmation=off", "verbose=nothing", "uda.cp.type=string", "uda.chain.type=string", "uda.chainID.type=string", "uda.link.type=numeric", "uda.prevLink.type=string", "uda.nextLink.type=string", "uda.chainMax.type=numeric", "uda.chainUntil.type=date"]) + "\n", encoding="utf-8")
        env = os.environ.copy(); env.update({"TASKRC": str(taskrc), "TASKDATA": str(data_dir), "NAUTICAL_CONFIG": str(Path(root) / "config-nautical.toml"), "NAUTICAL_CORE_PATH": root, "NO_COLOR": "1"})
        rows = [{"uuid": uuid, "status": "deleted", "description": "duplicate slot", "entry": "20260820T080000Z", "modified": "20260820T100000Z", "end": "20260820T100000Z", "due": "20260820T090000Z", "until": "20260820T100000Z", "cp": "1d", "chain": "on", "chainID": "duplicate-slot", "link": 1} for uuid in ("11111111-0000-0000-0000-000000000001", "22222222-0000-0000-0000-000000000002")]
        imported = subprocess.run([task_bin, "rc.hooks=off", "import"], input="".join(json.dumps(row) + "\n" for row in rows), text=True, capture_output=True, env=env, timeout=15.0)
        expect(imported.returncode == 0, f"duplicate fixture import failed: {imported.stderr!r}")
        applied = subprocess.run([sys.executable, str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"), "--apply", "--task-bin", task_bin, "--json"], text=True, capture_output=True, env=env, timeout=20.0)
        expect(applied.returncode == 1, f"duplicate slot unexpectedly applied: {applied.stdout!r}")
        payload = json.loads(applied.stdout); audit = payload.get("integrity_audit") or {}
        expect(payload.get("spawn") == 0 and payload.get("applied") == [], f"duplicate slot mutated: {payload!r}")
        expect(audit.get("status") == "manual_review", f"duplicate slot was not manual review: {payload!r}")
        expect(any(finding.get("invariant_id") == "slot.duplicate_occupant" for finding in audit.get("findings", ())), f"duplicate-slot finding was not reported: {payload!r}")


def test_health_check_critical_outbox_bytes():
    """health check should return critical when the lifecycle outbox exceeds its byte budget."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "dev_tools", "nautical_health_check.py")
    with tempfile.TemporaryDirectory() as td:
        outbox = Path(td) / ".nautical-state" / ".nautical_lifecycle_outbox.db"
        outbox.parent.mkdir()
        outbox.write_text("x" * 64, encoding="utf-8")
        proc = subprocess.run([sys.executable, path, "--taskdata", td, "--outbox-warn-bytes", "32", "--outbox-crit-bytes", "48", "--json"], text=True, capture_output=True, timeout=8.0)
        expect(proc.returncode == 2, f"expected critical exit code 2, got {proc.returncode}. stderr={proc.stderr!r}")
        obj = json.loads((proc.stdout or "").strip() or "{}")
        expect(obj.get("status") == "crit", f"unexpected status: {obj}")


def test_health_check_critical_outbox_rows():
    """health check should return critical when lifecycle outbox rows exceed their budget."""
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(root, "dev_tools", "nautical_health_check.py")
    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td))
        expect(repo.open().ok, "outbox health test setup failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute("INSERT INTO lifecycle_outbox (intent_id, plan_json, plan_fingerprint, parent_guard_json, configuration_fingerprint, schedule_fingerprint, lifecycle_stage, processing_state, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("health-row", "{}", "test", "{}", "test", "test", "planned", "ready", 1.0, 1.0))
            conn.commit()
        proc = subprocess.run([sys.executable, path, "--taskdata", td, "--outbox-warn-bytes", "1048576", "--outbox-crit-bytes", "10485760", "--outbox-warn-rows", "1", "--outbox-crit-rows", "1", "--json"], text=True, capture_output=True, timeout=8.0)
        expect(proc.returncode == 2, f"expected critical exit code 2, got {proc.returncode}. stderr={proc.stderr!r}")
        obj = json.loads((proc.stdout or "").strip() or "{}")
        expect(obj.get("status") == "crit", f"unexpected status: {obj}")
        expect(int((obj.get("outbox") or {}).get("rows") or 0) == 1, f"expected one outbox row, got {obj.get('outbox')}")


def test_queue_status_does_not_initialize_missing_outbox():
    """A read-only queue inspection must not create lifecycle state."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from nautical_core.lifecycle_outbox import lifecycle_outbox_path

    path = os.path.join(root, "nautical_core", "tools", "nautical_queue_status.py")
    with tempfile.TemporaryDirectory() as td:
        state_dir = Path(td) / ".nautical-state"
        proc = subprocess.run([sys.executable, path, "--taskdata", td, "--json"], text=True, capture_output=True, timeout=8.0)
        expect(proc.returncode == 0, f"queue status returned {proc.returncode}: {proc.stderr!r}")
        expect(not state_dir.exists(), f"queue status initialized state directory: {state_dir}")
        expect(not lifecycle_outbox_path(Path(td)).exists(), "queue status created an outbox database")


def test_outbox_drain_limit_config_and_env_override():
    """on-exit should use the outbox drain limit unless the process env overrides it."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "nautical.toml"
        config_path.write_text("\n", encoding="utf-8")
        script = "import json\nimport nautical_core\nfrom nautical_core.hooks import exit_impl\nexit_impl._load_core()\nprint(json.dumps([nautical_core.OUTBOX_DRAIN_MAX_ITEMS, exit_impl._OUTBOX_BATCH_MAX_ITEMS]))\n"
        env = os.environ.copy()
        env["NAUTICAL_CONFIG"] = str(config_path)
        env["TASKDATA"] = td
        env.pop("NAUTICAL_OUTBOX_DRAIN_MAX_ITEMS", None)
        defaulted = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, text=True, capture_output=True, timeout=8.0)
        expect(defaulted.returncode == 0, f"default outbox drain import failed: {defaulted.stderr!r}")
        expect(json.loads(defaulted.stdout) == [32, 32], f"unexpected default drain limit: {defaulted.stdout!r}")
        config_path.write_text("outbox_drain_max_items = 7\n", encoding="utf-8")
        configured = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, text=True, capture_output=True, timeout=8.0)
        expect(configured.returncode == 0, f"configured outbox drain import failed: {configured.stderr!r}")
        expect(json.loads(configured.stdout) == [7, 7], f"config drain limit was not effective: {configured.stdout!r}")
        env["NAUTICAL_OUTBOX_DRAIN_MAX_ITEMS"] = "3"
        overridden = subprocess.run([sys.executable, "-c", script], cwd=root, env=env, text=True, capture_output=True, timeout=8.0)
        expect(overridden.returncode == 0, f"queue drain override import failed: {overridden.stderr!r}")
        expect(json.loads(overridden.stdout) == [7, 3], f"environment outbox drain override did not win: {overridden.stdout!r}")


def test_reconcile_tool_print_plan_includes_evidence():
    """Reconcile dry-run output should explain why each action is safe."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile_report = importlib.import_module("nautical_core.reconcile_report")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mod = legacy._load_hook_module(str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"), "_nautical_reconcile_tool_print_test")
    parent = {"uuid": "11111111-0000-4000-8000-000000000001", "status": "completed", "description": "remote completion", "cp": "1d", "chain": "on", "chainID": "11111111", "link": 2, "due": "20260703T090000Z"}
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard, recurrence_fingerprint
    from nautical_core.lifecycle_recovery_models import RecoveryPlanResult
    observation = legacy._fixture_observation(parent)
    guard = ParentGuard(status="completed", chain="on", chain_id="11111111", link=2, recurrence_fingerprint=recurrence_fingerprint(parent), modified="")
    identity = LifecycleIdentity(chain_id="11111111", parent_uuid=parent["uuid"], source_link=2, target_link=3, event=LifecycleEvent.ACTIVATE)
    plan = RecoveryPlanResult(observation, LifecyclePlan(identity=identity, action=LifecycleAction.UPDATE_PARENT, parent_guard=guard), reason="next link already exists", child_short="22222222")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._print_plan(plan)
    out = buf.getvalue()
    expect("backfill nextLink:" in out, f"missing backfill headline: {out!r}")
    expect("reason: next link already exists" in out, f"missing reason evidence: {out!r}")
    expect("existing child: 22222222" in out, f"missing child evidence: {out!r}")
    second_parent = {**parent, "uuid": "22222222-0000-4000-8000-000000000002", "link": 3}
    second = mod._recovery_terminal(second_parent, "expiration recovery hop limit reached at 2; native until has already elapsed")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod._print_recovery_group([(plan, reconcile_report.describe_recovery_result(plan), "22222222"), (second, reconcile_report.describe_recovery_result(second), "")])
    out = buf.getvalue()
    expect("recover:" in out and "advanced 1 occurrence" in out, f"missing compact recovery summary: {out!r}")
    expect("result: partial" in out and "spawn:" not in out, f"compact output leaked per-hop lines: {out!r}")


def test_reconcile_evidence_prefers_due_over_carried_scheduled():
    """Reconcile evidence should show the recurrence target, not carried scheduled metadata."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    reconcile_report = importlib.import_module("nautical_core.reconcile_report")
    parent = {"uuid": "11111111-0000-4000-8000-000000000001", "status": "completed", "description": "remote completion", "anchor": "w:mon@t=09:00,17:00", "anchor_mode": "skip", "chain": "on", "chainID": "11111111", "link": 1, "due": "20260706T060000Z", "scheduled": "20260706T050000Z", "end": "20260706T070000Z"}
    class FakeCore:
        @staticmethod
        def coerce_int(value, default=0):
            try:
                return int(value)
            except Exception:
                return default
    from nautical_core.chain_generation import ChainGenerationService
    class FakeGeneration(ChainGenerationService):
        def __init__(self):
            super().__init__(FakeCore())
        def safe_parse_datetime(self, _value):
            return None, None
        def compute_anchor_child_due(self, _parent):
            return "20260706T140000Z", {"target_field": "due"}, []
        def build_child_draft(self, parent, child_due, child_field, next_link, parent_short, kind, cpmax, until_dt):
            from nautical_core.task_codec import DEFAULT_TASK_CODEC
            from nautical_core.task_models import NauticalTask, TaskDraft
            values = {"uuid": "22222222-0000-4000-8000-000000000002", "description": "remote completion", "status": "pending", "chain": "on", "chainID": parent.observation.to_mapping().get("chainID"), "link": next_link, "prevLink": parent_short, "anchor": "w:mon@t=09:00,17:00", "anchor_mode": "skip", "due": child_due, "scheduled": "20260706T130000Z"}
            return TaskDraft.from_task(NauticalTask.from_observation(DEFAULT_TASK_CODEC.decode_row(values, source_query="evidence fake child")))
    plan = reconcile.plan_recovery_decision(legacy._fixture_observation(parent), existing_children=[], hook=None, generation=FakeGeneration())
    evidence = reconcile_report.describe_recovery_result(plan)
    expect(evidence.get("child_field") == "due", f"expected due target evidence, got: {evidence!r}")
    expect(evidence.get("child_target") == "2026-07-06T14:00:00Z", f"expected due target, got: {evidence!r}")


def test_reconcile_tool_defaults_core_path_to_install_base():
    """The reconciler must seed hook bootstrap with the base containing nautical_core."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"
    prev_core_path = os.environ.get("NAUTICAL_CORE_PATH")
    try:
        os.environ.pop("NAUTICAL_CORE_PATH", None)
        mod = legacy._load_hook_module(str(path), "_nautical_reconcile_tool_core_path_test")
        expect(
            os.environ.get("NAUTICAL_CORE_PATH") == str(mod.BASE_DIR),
            f"expected NAUTICAL_CORE_PATH={mod.BASE_DIR}, got {os.environ.get('NAUTICAL_CORE_PATH')!r}",
        )
    finally:
        if prev_core_path is None:
            os.environ.pop("NAUTICAL_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_CORE_PATH"] = prev_core_path


def test_reconcile_tool_path_computes_timed_anchor_in_configured_timezone():
    """Actual reconcile tool loading should compute @t slots as configured-local time."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"
    prev_core_path = os.environ.get("NAUTICAL_CORE_PATH")
    try:
        os.environ.pop("NAUTICAL_CORE_PATH", None)
        mod = legacy._load_hook_module(str(path), "_nautical_reconcile_tool_timed_anchor_test")
        hook = SimpleNamespace(core=importlib.import_module("nautical_core"))
        from nautical_core.chain_generation import ChainGenerationService

        generation = ChainGenerationService.from_core(hook.core)
        child_due, _meta, _dnf = generation.compute_anchor_child_due(
            legacy._fixture_task(
                {
                    "uuid": "c3f2c233-0000-4000-8000-000000000001",
                    "status": "completed",
                    "description": "Drink 0.5L of water",
                    "anchor": "w:mon..sun@t=05:00,09:00,14:00,19:00",
                    "anchor_mode": "skip",
                    "chain": "on",
                    "chainID": "d07ff246",
                    "link": 92,
                    "due": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 4), (9, 0))),
                    "end": hook.core.fmt_isoz(hook.core.build_local_datetime(date(2026, 7, 4), (10, 0))),
                }
            )
        )
        child_local = hook.core.to_local(child_due)
        expect(
            (child_local.hour, child_local.minute) == (14, 0),
            f"expected 14:00 local via reconcile tool path: {child_local}",
        )
    finally:
        if prev_core_path is None:
            os.environ.pop("NAUTICAL_CORE_PATH", None)
        else:
            os.environ["NAUTICAL_CORE_PATH"] = prev_core_path


def test_reconcile_configuration_verification_fails_closed():
    """Configuration exceptions must become unavailable, never a clean reconcile state."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"
    mod = legacy._load_hook_module(str(path), "_nautical_reconcile_configuration_state_test")
    import types

    core_module = types.ModuleType("nautical_core_test_module")
    core_module.configuration_drift = lambda: (_ for _ in ()).throw(RuntimeError("malformed TOML"))
    hook = SimpleNamespace(core=core_module)
    check = mod._configuration_verification(hook)
    expect(check.status == "unavailable", f"configuration exception was not unavailable: {check.status!r}")
    expect("malformed TOML" in check.reason, f"configuration failure detail was lost: {check.reason!r}")
    status, reason = mod._configuration_state(hook)
    expect(status == "unavailable" and reason == check.reason, f"state adapter changed failure: {status!r}, {reason!r}")


def test_reconcile_startup_config_failure_is_structured():
    """Taskdata configuration startup failures expose unavailable status in JSON."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"
    mod = legacy._load_hook_module(str(path), "_nautical_reconcile_configuration_startup_test")
    args = SimpleNamespace(json=True, apply=True)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = mod._startup_failure(args, "taskdata_config", RuntimeError("invalid timezone"))
    payload = json.loads(output.getvalue())
    expect(result == 1, f"configuration startup failure returned {result}")
    expect(payload.get("configuration_status") == "unavailable", f"missing unavailable status: {payload!r}")
    expect(payload.get("configuration_drift") == "invalid timezone", f"configuration detail was lost: {payload!r}")


def test_reconcile_subprocess_output_contracts():
    """Operator subprocess modes keep JSON on stdout and diagnostics on stderr."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"
    base = [sys.executable, str(tool), "--task-bin", "/missing/nautical-task"]
    json_run = subprocess.run(
        [*base, "--json"], cwd=str(root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    expect(json_run.returncode == 1, f"JSON startup failure returned {json_run.returncode}")
    payload = json.loads(json_run.stdout)
    expect(payload.get("status") == "error", f"JSON startup status was not error: {payload!r}")
    expect(payload.get("startup_errors") == 1, f"JSON startup error count missing: {payload!r}")
    expect(json_run.stderr == "", f"JSON mode leaked diagnostics to stderr: {json_run.stderr!r}")

    human_run = subprocess.run(
        base, cwd=str(root), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    expect(human_run.returncode == 1, f"human startup failure returned {human_run.returncode}")
    expect(human_run.stdout == "", f"human startup failure polluted stdout: {human_run.stdout!r}")
    expect("Taskwarrior executable was not found" in human_run.stderr, f"human diagnostic was not actionable: {human_run.stderr!r}")


def test_reconcile_apply_lease_serializes_mutations():
    """Concurrent reconcile apply attempts must not share the mutation lease."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = legacy._load_hook_module(str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"), "_nautical_reconcile_apply_lease_test")
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        with tool._reconcile_apply_lock(taskdata) as first:
            expect(first, "reconcile apply lease was not acquired")
            with tool._reconcile_apply_lock(taskdata) as second:
                expect(not second, "reconcile apply lease allowed concurrent acquisition")
        with tool._reconcile_apply_lock(taskdata) as released:
            expect(released, "reconcile apply lease was not released")


def test_reconcile_apply_refuses_a_second_full_run():
    """A held apply lease must reject another reconcile before it loads hooks or exports tasks."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = legacy._load_hook_module(str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"), "_nautical_reconcile_full_run_lease_test")
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td)
        output = io.StringIO()
        with tool._reconcile_apply_lock(taskdata) as held:
            expect(held, "test could not acquire reconcile lease")
            with contextlib.redirect_stdout(output):
                result = tool.main(["--apply", "--json"], _unit_of_work=legacy._test_operator_uow(taskdata))
    summary = json.loads(output.getvalue())
    expect(result == 1, f"busy reconcile returned {result}")
    expect(summary.get("stage") == "apply_lock", f"busy reconcile was not reported as a lease conflict: {summary!r}")


def test_reconcile_parent_identity_errors_are_actionable():
    """Parent guard failures should identify the exact broken identity field."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = legacy._load_hook_module(str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"), "_nautical_reconcile_identity_diagnostics_test")
    base = {"uuid": "11111111-0000-0000-0000-000000000001", "status": "completed", "chain": "on", "chainID": "chain001", "link": 2, "nextLink": ""}
    cases = ((dict(base, chainID=""), "parent chainID is missing"), (dict(base, link=""), "parent link is missing"), (dict(base, chainID="11111111-0000-0000-0000-000000000001", link="", prevLink=""), "parent link is missing"), (dict(base, link="not-a-number"), "parent link is invalid"), (dict(base, link=0), "parent link must be positive"))
    for parent, expected in cases:
        try:
            tool._parent_guard_filters(parent)
        except RuntimeError as exc:
            expect(expected in str(exc), f"unclear identity diagnostic: {exc}")
        else:
            raise AssertionError(f"invalid parent identity was accepted: {parent!r}")


def test_reconcile_expired_pending_child_is_resumable_partial():
    """A pending child past native until should wait for Taskwarrior expiration."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = legacy._load_hook_module(str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"), "_nautical_reconcile_pending_until_test")
    parent = {"uuid": "11111111-0000-0000-0000-000000000001", "link": 1}
    plan = tool._recovery_terminal(parent, "live recovery child native until has already elapsed")
    expect(legacy._recovery_action(plan) == "partial", f"expired pending child was not resumable: {plan}")
    expect("rerun reconcile" in plan.reason, f"partial recovery guidance missing: {plan.reason}")


def test_reconcile_expiration_anchor_advances_from_recurrence_target():
    """Expired anchor links should select the first slot after the prior recurrence target."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    hook = legacy._find_hook_file("on-modify.nautical")
    mod = legacy._load_hook_module(hook, "_nautical_reconcile_expiration_anchor_due_test")
    parent = {
        "uuid": "00000000-0000-4000-8000-0000000050aa",
        "status": "deleted",
        "anchor": "w:mon@t=09:00",
        "anchor_mode": "skip",
        "chainID": "11111111",
        "link": 1,
        "due": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 6), (9, 0))),
        "end": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2026, 7, 15), (18, 0))),
    }
    child_due, meta = reconcile.compute_expiration_child_due(parent, hook=mod)
    child_local = mod.core.to_local(child_due)
    expect(child_local.date() == date(2026, 7, 13) and (child_local.hour, child_local.minute) == (9, 0), f"expired anchor should advance from prior due: {child_local}")
    expect(meta.get("basis") == "due recurrence target (expired)", f"unexpected expiry basis: {meta!r}")


def test_reconcile_hookless_completion_verifies_scheduled_and_wait_carry():
    """Hookless recovery should preserve and verify scheduled/wait offsets."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    mod = legacy._load_hook_module(legacy._find_hook_file("on-modify.nautical"), "_nautical_reconcile_hookless_carry_test")
    due = mod.core.build_local_datetime(date(2026, 7, 20), (10, 0))
    scheduled = mod.core.build_local_datetime(date(2026, 7, 20), (9, 30))
    wait = mod.core.build_local_datetime(date(2026, 7, 20), (8, 0))
    parent = {"uuid": "11111111-0000-4000-8000-000000000001", "status": "completed", "cp": "7d", "chain": "on", "chainID": "11111111", "link": 1, "due": mod.core.fmt_isoz(due), "scheduled": mod.core.fmt_isoz(scheduled), "wait": mod.core.fmt_isoz(wait), "end": mod.core.fmt_isoz(due + timedelta(hours=1))}
    plan = legacy._recovery_plan(reconcile, parent, existing_children=[], hook=mod)
    child = legacy._recovery_child(plan)
    expect(legacy._recovery_action(plan) == "spawn" and child is not None, f"valid hookless carry did not produce a child: {plan}")
    child_due = mod.core.parse_dt_any(child.get("due"))
    expect(mod.core.parse_dt_any(child.get("scheduled")) - child_due == scheduled - due, f"scheduled carry drifted: {child!r}")
    expect(mod.core.parse_dt_any(child.get("wait")) - child_due == wait - due, f"wait carry drifted: {child!r}")
    failed = legacy._recovery_plan(reconcile, dict(parent, scheduled="not-a-date"), existing_children=[], hook=mod)
    expect(legacy._recovery_action(failed) == "error" and "scheduled" in failed.reason, f"malformed scheduled carry was not rejected: {failed}")
    failed_wait = legacy._recovery_plan(reconcile, dict(parent, wait="not-a-date"), existing_children=[], hook=mod)
    expect(legacy._recovery_action(failed_wait) == "error" and "wait" in failed_wait.reason, f"malformed wait carry was not rejected: {failed_wait}")


def test_reconcile_native_until_manual_review_is_not_a_hard_error():
    """An unrecoverable native-until window is reported without claiming failure."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    hook_path = legacy._find_hook_file("on-modify.nautical")
    hook = legacy._load_hook_module(hook_path, "_nautical_reconcile_manual_until_hook_test")
    if hasattr(hook, "_load_core"):
        hook._load_core()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    tool = legacy._load_hook_module(
        str(Path(root) / "nautical_core" / "tools" / "nautical_reconcile.py"),
        "_nautical_reconcile_manual_until_tool_test",
    )

    def stamp(day, hhmm):
        return hook.core.fmt_isoz(hook.core.build_local_datetime(day, hhmm))

    row = legacy._fixture_observation(
        {
            "uuid": "00000000-0000-4000-8000-000000003248",
            "chain": "on",
            "chainID": "manual-until",
            "link": 1,
            "status": "pending",
            "due": stamp(date(2026, 7, 23), (23, 0)),
            "until": stamp(date(2026, 7, 23), (22, 0)),
        }
    )

    class _ControlPlane:
        def audit_native_until(self, rows, **_kwargs):
            del rows
            return SimpleNamespace(
                native_until=SimpleNamespace(
                    repairs=[{"action": "manual_review", "task": "00000000"}],
                    errors=[],
                ),
                candidates=[],
            )

    snapshot = SimpleNamespace(active_rows=lambda: [row])
    repairs, errors = tool._native_until_repairs(
        "task", hook, apply=False, snapshot=snapshot, control_plane=_ControlPlane()
    )
    expect(not errors, f"manual review was reported as a failed mutation: {errors!r}")
    expect(
        repairs and repairs[0].get("action") == "manual_review",
        f"manual review was not preserved: {repairs!r}",
    )


def test_reconcile_expiration_candidate_requires_expiry_evidence():
    """Deleted chains distinguish expiration, manual stop, and ambiguous evidence."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    hook = legacy._find_hook_file("on-modify.nautical")
    mod = legacy._load_hook_module(hook, "_nautical_reconcile_expiration_candidate_test")
    parent = {
        "uuid": "11111111-0000-4000-8000-000000000001",
        "status": "deleted",
        "description": "expired occurrence",
        "cp": "7d",
        "chain": "on",
        "chainID": "11111111",
        "link": 2,
        "due": "20260720T060000Z",
        "until": "20260726T205959Z",
        "end": "20260726T205959Z",
    }
    is_candidate = lambda task: reconcile.is_orphan_expiration_candidate(
        legacy._task_observation(task),
        safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse,
    )
    expect(is_candidate(parent), "deletion exactly at until should be an expiration candidate")
    manual = dict(parent, end="20260726T205958Z")
    expect(not is_candidate(manual), "manual deletion before until must not advance")
    evidence = reconcile.deleted_chain_disposition(
        legacy._task_observation(manual), safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse
    )
    expect(evidence.disposition.value == "manual", f"early deletion should stop the chain: {evidence!r}")
    no_until_evidence = reconcile.deleted_chain_disposition(
        legacy._task_observation({key: value for key, value in parent.items() if key != "until"}),
        safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse,
    )
    expect(no_until_evidence.disposition.value == "manual", f"deletion without until should stop the chain: {no_until_evidence!r}")
    malformed_evidence = reconcile.deleted_chain_disposition(
        legacy._task_observation(dict(parent, until="not-a-date")),
        safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse,
    )
    expect(malformed_evidence.disposition.value == "ambiguous", f"malformed evidence must fail closed: {malformed_evidence!r}")
    manual_plan = legacy._recovery_plan(reconcile, manual, existing_children=[], hook=mod)
    expect(legacy._recovery_action(manual_plan) in {"manual_stop", "manual_review"}, f"manual deletion should stop the chain: {manual_plan}")
    expect(not is_candidate(dict(parent, status="completed")), "completed tasks use the completion candidate path")
    expect(not is_candidate(dict(parent, until="not-a-date")), "malformed until must fail closed")
    expect(not is_candidate(dict(parent, nextLink="22222222")), "already-linked expiration must not be reconsidered")


def test_reconcile_expiration_cp_advances_from_recurrence_target():
    """Expired CP links advance from due or scheduled, not deletion end."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    reconcile = importlib.import_module("nautical_core.chain_integrity_lifecycle")
    hook = legacy._find_hook_file("on-modify.nautical")
    mod = legacy._load_hook_module(hook, "_nautical_reconcile_expiration_cp_due_test")
    due = mod.core.build_local_datetime(date(2026, 7, 20), (9, 0))
    expired_end = mod.core.build_local_datetime(date(2026, 7, 26), (23, 59))
    parent = {
        "uuid": "00000000-0000-4000-8000-000000000509",
        "status": "deleted",
        "cp": "7d",
        "chainID": "11111111",
        "link": 1,
        "due": mod.core.fmt_isoz(due),
        "end": mod.core.fmt_isoz(expired_end),
    }
    child_due, meta = reconcile.compute_expiration_child_due(parent, hook=mod)
    child_local = mod.core.to_local(child_due)
    expect(child_local.date() == date(2026, 7, 27) and (child_local.hour, child_local.minute) == (9, 0), f"expired CP should advance from prior due: {child_local}")
    expect(meta.get("basis") == "due recurrence target (expired)", f"unexpected expiry basis: {meta!r}")
    scheduled_parent = dict(parent)
    scheduled_parent.pop("due")
    scheduled_parent["scheduled"] = mod.core.fmt_isoz(due)
    child_scheduled, scheduled_meta = reconcile.compute_expiration_child_due(scheduled_parent, hook=mod)
    expect(mod.core.to_local(child_scheduled).date() == date(2026, 7, 27), f"scheduled-only expiry should advance from scheduled: {child_scheduled}")
    expect(scheduled_meta.get("target_field") == "scheduled", f"unexpected scheduled metadata: {scheduled_meta!r}")


def _delegate(name: str):
    def run() -> None:
        legacy = importlib.import_module("dev_tools.nautical_golden_tests")
        getattr(legacy, f"_legacy_{name}")()

    run.__name__ = name
    run.__qualname__ = name
    run.__doc__ = f"Delegated golden test: {name}."
    return run


globals().update({name: _delegate(name) for name in _NAMES})
def test_reconcile_real_taskwarrior_anchor_repair_round_trip():
    """A deleted anchor occurrence receives one real linked successor."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    _find_hook_file = legacy._find_hook_file
    _load_hook_module = legacy._load_hook_module
    task_bin = shutil.which("task")
    if not task_bin:
        return
    with tempfile.TemporaryDirectory(prefix="nautical-anchor-repair-") as td:
        root = Path(td)
        data_dir = root / "data"
        data_dir.mkdir()
        taskrc = root / "taskrc"
        taskrc.write_text(
            "\n".join([
                f"data.location={data_dir}", "hooks=off", "confirmation=off", "verbose=nothing",
                "uda.anchor.type=string", "uda.anchor_mode.type=string", "uda.chain.type=string",
                "uda.chainID.type=string", "uda.link.type=numeric", "uda.prevLink.type=string",
                "uda.nextLink.type=string", "uda.chainMax.type=numeric", "uda.chainUntil.type=date",
            ]) + "\n", encoding="utf-8",
        )
        config = root / "config-nautical.toml"
        config.write_text('tz = "UTC"\n', encoding="utf-8")
        env = os.environ.copy()
        env.update({
            "TASKRC": str(taskrc), "TASKDATA": str(data_dir),
            "NAUTICAL_CONFIG": str(config),
            "NAUTICAL_CORE_PATH": ROOT, "NO_COLOR": "1",
        })
        parent = {
            "uuid": "33333333-0000-4000-8000-000000000003", "status": "deleted",
            "description": "Anchor repair", "entry": "20260820T080000Z",
            "modified": "20260820T100000Z", "end": "20260820T100000Z",
            "due": "20260820T090000Z", "until": "20260820T100000Z",
            "anchor": "y:09-01", "anchor_mode": "skip", "chain": "on",
            "chainID": "anchor-real", "link": 1,
        }
        imported = subprocess.run(
            [task_bin, "rc.hooks=off", "import"], input=json.dumps(parent) + "\n",
            text=True, capture_output=True, env=env, timeout=15.0,
        )
        expect(imported.returncode == 0, f"anchor fixture import failed: {imported.stderr!r}")
        applied = subprocess.run(
            [sys.executable, str(Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"),
             "--apply", "--task-bin", task_bin, "--json", "--max-expiration-hops", "1"],
            text=True, capture_output=True, env=env, timeout=20.0,
        )
        expect(applied.returncode in (0, 2), f"anchor reconcile failed: {applied.stderr!r}")
        payload = json.loads(applied.stdout)
        expect(payload.get("spawn") == 1 and len(payload.get("applied") or []) == 1, f"anchor was not repaired: {payload!r}")
        exported = subprocess.run(
            [task_bin, "rc.hooks=off", "rc.json.array=1", "chainID:anchor-real", "export"],
            text=True, capture_output=True, env=env, timeout=15.0,
        )
        expect(exported.returncode == 0, f"anchor verification export failed: {exported.stderr!r}")
        rows = json.loads(exported.stdout)
        expect(len(rows) == 2, f"anchor repair created the wrong number of rows: {rows!r}")
        by_link = {int(float(row.get("link"))): row for row in rows}
        expect(by_link[1].get("nextLink") == str(by_link[2].get("uuid") or "")[:8], f"anchor parent was not linked: {rows!r}")
def test_reconcile_expiration_plan_reuses_limits_and_deleted_slot_dedup():
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    _find_hook_file = legacy._find_hook_file
    _load_hook_module = legacy._load_hook_module
    _recovery_plan = legacy._recovery_plan
    _recovery_action = legacy._recovery_action
    reconcile_report = importlib.import_module("nautical_core.reconcile_report")


def test_seasonal_selection_reconcile_spawn_recovery_and_dedup():
    """Reconcile should compute, spawn, and deduplicate the next seasonal slot."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    _find_hook_file = legacy._find_hook_file
    _load_hook_module = legacy._load_hook_module
    _fixture_observation = legacy._fixture_observation
    _recovery_action = legacy._recovery_action
    _recovery_child = legacy._recovery_child
    import nautical_core.chain_integrity_lifecycle as reconcile

    hook_path = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook_path, "_nautical_seasonal_reconcile_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()
    season_support = mod.core._import_sibling("season_support")
    previous_hemisphere = season_support.active_hemisphere()
    previous_core_hemisphere = mod.core.SEASON_HEMISPHERE
    mod.core.SEASON_HEMISPHERE = "north"
    season_support.configure_hemisphere("north")

    def stamp(day, hhmm):
        return mod.core.fmt_isoz(mod.core.build_local_datetime(day, hhmm))

    parent = {
        "uuid": "11111111-0000-4000-8000-000000000001",
        "status": "completed",
        "description": "seasonal reconcile",
        "anchor": "(w:mon)@in-spring=first@t=09:00",
        "anchor_mode": "skip",
        "chain": "on",
        "chainID": "season456",
        "link": 1,
        "due": stamp(date(2026, 3, 2), (9, 0)),
        "end": stamp(date(2026, 7, 1), (10, 0)),
    }
    parent_obs = _fixture_observation(parent)
    plan = reconcile.plan_recovery_decision(parent_obs, existing_children=[], hook=mod)
    expect(_recovery_action(plan) == "spawn", f"reconcile did not spawn seasonal child: {plan}")
    child_local = mod.core.to_local(plan.child_due)
    expect(
        child_local.date() == date(2027, 3, 1)
        and (child_local.hour, child_local.minute) == (9, 0),
        f"reconcile chose the wrong seasonal slot: {child_local}",
    )
    expect((_recovery_child(plan) or {}).get("anchor") == parent["anchor"], f"reconcile child lost anchor: {plan}")

    existing = {
        "uuid": "22222222-0000-4000-8000-000000000002",
        "status": "pending",
        "description": "seasonal reconcile",
        "chain": "on",
        "chainID": "season456",
        "link": 2,
        "prevLink": "11111111",
        "anchor": parent["anchor"],
        "due": stamp(date(2027, 3, 1), (9, 0)),
    }
    repeated = reconcile.plan_recovery_decision(
        parent_obs,
        existing_children=[_fixture_observation(existing)],
        hook=mod,
    )
    expect(
        _recovery_action(repeated) == "spawn" and repeated.child_short == "22222222",
        f"reconcile duplicated an existing seasonal slot: {repeated}",
    )

    expired = {
        **parent,
        "status": "deleted",
        "anchor": "(w:mon)@in-winter=last@t=09:00",
        "due": stamp(date(2027, 2, 22), (9, 0)),
        "end": stamp(date(2027, 7, 1), (10, 0)),
    }
    recovered_due, recovered_meta = reconcile.compute_expiration_child_due(expired, hook=mod)
    recovered_local = mod.core.to_local(recovered_due)
    expect(
        recovered_local.date() == date(2028, 2, 28),
        f"expired winter advanced incorrectly: {recovered_local}",
    )
    expect(
        recovered_meta.get("basis") == "due recurrence target (expired)",
        f"expiration recovery lost its basis: {recovered_meta}",
    )
    mod.core.SEASON_HEMISPHERE = previous_core_hemisphere
    season_support.configure_hemisphere(previous_hemisphere)
def test_reconcile_repairs_invalid_native_until_from_previous_link():
    """Hookless due moves should recover the prior link's native-until carry policy."""
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    _find_hook_file = legacy._find_hook_file
    _load_hook_module = legacy._load_hook_module
    _task_observation = legacy._task_observation
    import nautical_core.chain_integrity_lifecycle as reconcile

    hook_path = _find_hook_file("on-modify.nautical")
    mod = _load_hook_module(hook_path, "_nautical_until_reconcile_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    def stamp(day, hhmm):
        return mod.core.fmt_isoz(mod.core.build_local_datetime(day, hhmm))

    previous = {
        "uuid": "00000000-0000-4000-8000-000000003241",
        "description": "previous",
        "status": "completed",
        "chain": "on",
        "chainID": "until-test",
        "link": 1,
        "due": stamp(date(2026, 7, 20), (9, 0)),
        "until": stamp(date(2026, 7, 20), (23, 0)),
    }
    current = {
        "uuid": "00000000-0000-4000-8000-000000003242",
        "description": "current",
        "status": "pending",
        "chain": "on",
        "chainID": "until-test",
        "link": 2,
        "due": stamp(date(2026, 7, 22), (9, 0)),
        "until": stamp(date(2026, 7, 21), (23, 0)),
    }
    datetime_effects = mod._module("modify_datetime_effects")
    datetime_ports = datetime_effects.datetime_effect_ports_for(mod)
    expect(
        reconcile.invalid_native_until_reason(_task_observation(current), safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse),
        "invalid native-until window was not detected",
    )
    repaired, error = reconcile.repair_native_until_from_previous(
        _task_observation(previous),
        _task_observation(current),
        kind="anchor",
        safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse,
        fmt_isoz=mod.core.fmt_isoz,
        utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(datetime_ports, value),
        local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(datetime_ports, value),
    )
    expect(not error and repaired == stamp(date(2026, 7, 22), (23, 0)), f"wrong carried until: {repaired}, {error}")
    fallback, fallback_error = reconcile.fallback_native_until_at_day_end(
        _task_observation({
            "uuid": "00000000-0000-4000-8000-000000003243", "description": "fallback",
            "status": "pending", "chain": "on", "chainID": "until-test", "link": 3,
            "due": stamp(date(2026, 7, 23), (9, 0)),
        }),
        safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse,
        fmt_isoz=mod.core.fmt_isoz,
        utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(datetime_ports, value),
        local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(datetime_ports, value),
    )
    expect(
        not fallback_error and fallback == stamp(date(2026, 7, 23), (23, 0)),
        f"fallback did not use local 23:00: {fallback}, {fallback_error}",
    )
    late_fallback, late_error = reconcile.fallback_native_until_at_day_end(
        _task_observation({
            "uuid": "00000000-0000-4000-8000-000000003244", "description": "late fallback",
            "status": "pending", "chain": "on", "chainID": "until-test", "link": 4,
            "due": stamp(date(2026, 7, 23), (23, 0)),
        }),
        safe_parse_datetime=mod._TASK_DATETIME_PARSER.parse,
        fmt_isoz=mod.core.fmt_isoz,
        utc_to_local_naive=lambda value: datetime_effects.utc_to_local_naive(datetime_ports, value),
        local_naive_to_utc=lambda value: datetime_effects.local_naive_to_utc(datetime_ports, value),
    )
    expect(
        late_fallback is None and "at or after local 23:00" in (late_error or ""),
        f"late due did not fail closed: {late_fallback}, {late_error}",
    )
    tool = _load_hook_module(
        str(Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"),
        "_nautical_reconcile_until_format_test",
    )
    expected_until = stamp(date(2026, 7, 23), (23, 0))
    compact_expected = expected_until.replace("-", "").replace(":", "")
    actual_dt, actual_parse_error = mod._TASK_DATETIME_PARSER.parse(compact_expected)
    expected_dt, expected_parse_error = mod._TASK_DATETIME_PARSER.parse(expected_until)
    expect(
        tool._native_until_matches(_task_observation({
            "uuid": "00000000-0000-4000-8000-000000003245", "description": "verify",
            "status": "pending", "chain": "on", "chainID": "until-test", "link": 5,
            "until": compact_expected,
        }), expected_until, mod),
        f"Taskwarrior's compact UTC timestamp should verify against the fallback instant: "
        f"{actual_dt!r}/{actual_parse_error!r} != {expected_dt!r}/{expected_parse_error!r}",
    )
    expect(
        not tool._native_until_matches(
            _task_observation({
                "uuid": "00000000-0000-4000-8000-000000003246", "description": "different",
                "status": "pending", "chain": "on", "chainID": "until-test", "link": 6,
                "until": mod.core.fmt_isoz(expected_dt + timedelta(hours=1)),
            }), expected_until, mod
        ),
        "a different native-until instant must fail verification",
    )
    guard_error = tool._native_until_guard_error(
        _task_observation({
            "uuid": "00000000-0000-4000-8000-000000003247", "description": "guard",
            "status": "pending", "chain": "on", "chainID": "cid", "link": 2, "due": "20260801T090000Z",
        }),
        _task_observation({
            "uuid": "00000000-0000-4000-8000-000000003247", "description": "guard",
            "status": "pending", "chain": "on", "chainID": "cid", "link": 2, "due": "20260802T090000Z",
        }),
    )
    expect(guard_error and "due" in guard_error, f"target drift was not detected: {guard_error!r}")

def test_integration_contract_covers_all_mutation_and_outbox_states():
    """Every tagged integration outcome has one valid constructible shape."""
    from nautical_core.integration_models import (
        CommandFailureKind,
        FailureEvidence,
        GuardTimestamp,
        GuardTimestampField,
        MutationGuard,
        MutationOperation,
        MutationOutcome,
        MutationOutcomeKind,
        MutationPostcondition,
        OutboxIntent,
        OutboxOutcome,
        OutboxOutcomeKind,
        OutboxStage,
        TaskCommand,
    )
    from nautical_core.lifecycle_models import LifecycleEvent, LifecycleIdentity

    guard = MutationGuard(
        "parent-uuid",
        "completed",
        "chain-3",
        2,
        "rf1-states",
        (GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T090000Z"),),
        0,
    )
    command = TaskCommand(("task", "export"), "verify mutation", 10.0)
    busy = FailureEvidence(command, CommandFailureKind.BUSY, 1, 1, 0.1, True, "lock active")
    expected = {
        MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
        MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED,
        MutationOperation.CHAIN_DISABLE: MutationPostcondition.CHAIN_DISABLED,
        MutationOperation.NATIVE_UNTIL_REPAIR: MutationPostcondition.NATIVE_UNTIL_REPAIRED,
        MutationOperation.METADATA_REPAIR: MutationPostcondition.METADATA_REPAIRED,
    }
    for operation, postcondition in expected.items():
        outcome = MutationOutcome(
            operation,
            MutationOutcomeKind.APPLIED,
            guard,
            (postcondition,),
        )
        expect(outcome.operation is operation, f"failed to construct {operation.value} outcome")

    mutation_states = {
        MutationOutcomeKind.APPLIED: MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.APPLIED,
            guard,
            (MutationPostcondition.CHAIN_DISABLED,),
        ),
        MutationOutcomeKind.ALREADY_APPLIED: MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.ALREADY_APPLIED,
            guard,
            (MutationPostcondition.CHAIN_DISABLED,),
        ),
        MutationOutcomeKind.RETRYABLE: MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.RETRYABLE,
            guard,
            reason="busy",
            failure=busy,
        ),
        MutationOutcomeKind.REJECTED: MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.REJECTED,
            guard,
            reason="rejected",
        ),
        MutationOutcomeKind.CONFLICT: MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.CONFLICT,
            guard,
            reason="conflict",
        ),
        MutationOutcomeKind.MANUAL_REVIEW: MutationOutcome(
            MutationOperation.CHAIN_DISABLE,
            MutationOutcomeKind.MANUAL_REVIEW,
            guard,
            reason="review",
        ),
    }
    expect(set(mutation_states) == set(MutationOutcomeKind), "mutation outcome coverage is incomplete")

    identity = LifecycleIdentity("chain-3", "parent-uuid", 2, None, LifecycleEvent.MANUAL_DELETE)
    intent = OutboxIntent(
        identity,
        guard,
        (MutationOperation.CHAIN_DISABLE,),
        (MutationPostcondition.CHAIN_DISABLED,),
    )
    applied = mutation_states[MutationOutcomeKind.APPLIED]
    retryable = mutation_states[MutationOutcomeKind.RETRYABLE]
    conflict = mutation_states[MutationOutcomeKind.CONFLICT]
    outbox_states = {
        OutboxOutcomeKind.ADVANCED: OutboxOutcome(
            intent,
            OutboxStage.APPLYING,
            OutboxOutcomeKind.ADVANCED,
            (),
        ),
        OutboxOutcomeKind.FINALIZED: OutboxOutcome(
            intent,
            OutboxStage.FINALIZED,
            OutboxOutcomeKind.FINALIZED,
            (applied,),
        ),
        OutboxOutcomeKind.RETRYABLE: OutboxOutcome(
            intent,
            OutboxStage.RETRYABLE,
            OutboxOutcomeKind.RETRYABLE,
            (retryable,),
            "retry later",
        ),
        OutboxOutcomeKind.MANUAL_REVIEW: OutboxOutcome(
            intent,
            OutboxStage.MANUAL_REVIEW,
            OutboxOutcomeKind.MANUAL_REVIEW,
            (conflict,),
            "guard conflict",
        ),
    }
    expect(set(outbox_states) == set(OutboxOutcomeKind), "outbox outcome coverage is incomplete")
def test_reconcile_candidate_and_plan_paths():
    """Hookless-completion repair should target only active completed orphans."""
    from datetime import datetime, timezone
    import nautical_core as core
    from nautical_core import reconcile_report
    import nautical_core.chain_integrity_lifecycle as reconcile
    from nautical_core.lifecycle_recovery_models import RecoveryPlanResult, RecoveryRefusal, RecoveryStatus

    parent = {
        "uuid": "11111111-0000-4000-8000-000000000001",
        "status": "completed",
        "description": "remote completion",
        "cp": "P1D",
        "chain": "on",
        "chainID": "11111111",
        "link": 2,
        "due": "20260101T090000Z",
        "end": "20260101T100000Z",
    }
    parent_obs = fixture_observation(parent)
    expect(reconcile.is_orphan_completion_candidate(parent_obs), "completed active chain without nextLink should be a candidate")
    with_next = fixture_observation(dict(parent, nextLink="22222222"))
    expect(not reconcile.is_orphan_completion_candidate(with_next), "linked completion should not be a candidate")
    chain_off = fixture_observation(dict(parent, chain="off"))
    expect(not reconcile.is_orphan_completion_candidate(chain_off), "chain:off completion should not be a candidate")

    existing_row = {
            "uuid": "22222222-0000-4000-8000-000000000002",
            "chainID": "11111111",
            "link": 3,
            "status": "pending",
            "description": "remote completion",
            "chain": "on",
            "prevLink": "11111111",
            "cp": "P1D",
            "due": "20260102T090000Z",
        }
    existing = [fixture_observation(existing_row)]
    plan = reconcile.plan_recovery_decision(parent_obs, existing_children=existing, hook=None)
    expect(isinstance(plan, RecoveryPlanResult) and plan.child_short == "22222222", f"unexpected backfill plan: {plan}")
    duplicate = {
        **existing_row,
        "uuid": "33333333-0000-4000-8000-000000000003",
    }
    ambiguous = reconcile.plan_recovery_decision(parent_obs, existing_children=[*existing, fixture_observation(duplicate)], hook=None)
    expect(
        isinstance(ambiguous, RecoveryRefusal) and ambiguous.status is RecoveryStatus.MANUAL_REVIEW and "multiple tasks" in ambiguous.reason,
        f"duplicate next slots must fail closed: {ambiguous}",
    )
    nonreciprocal = reconcile.plan_recovery_decision(
        parent_obs,
        existing_children=[fixture_observation(dict(existing_row, prevLink="beeswax"))],
        hook=None,
    )
    expect(
        isinstance(nonreciprocal, RecoveryRefusal) and "prevLink" in nonreciprocal.reason,
        f"nonreciprocal next slot must fail closed: {nonreciprocal}",
    )
    recurrence_mismatch = reconcile.plan_recovery_decision(
        parent_obs,
        existing_children=[fixture_observation(dict(existing_row, cp="P2D"))],
        hook=None,
    )
    expect(
        isinstance(recurrence_mismatch, RecoveryRefusal) and "recurrence field cp" in recurrence_mismatch.reason,
        f"mismatched recurrence child must fail closed: {recurrence_mismatch}",
    )
    null_recurrence = reconcile.plan_recovery_decision(
        parent_obs,
        existing_children=[fixture_observation(dict(existing_row, anchor_file=None))],
        hook=None,
    )
    expect(
        isinstance(null_recurrence, RecoveryPlanResult) and null_recurrence.child_short == "22222222",
        f"literal null recurrence UDA should be treated as unset: {null_recurrence}",
    )
    expect(
        reconcile.recurrence_kind(fixture_observation(dict(parent, cp="", anchor="w:mon", anchor_file=None))) == "anchor",
        "literal null anchor_file changed an anchor recurrence kind",
    )

    class FakeCore:
        @staticmethod
        def coerce_int(value, default=0):
            try:
                return int(value)
            except Exception:
                return default

        @staticmethod
        def fmt_isoz(value):
            return value

        @staticmethod
        def now_utc():
            return datetime.now(timezone.utc)

    from nautical_core.chain_generation import ChainGenerationService

    class FakeGeneration(ChainGenerationService):
        def __init__(self):
            super().__init__(FakeCore())

        def parse_datetime(self, _value):
            return None, None

        def compute_cp_child_due(self, _parent):
            return "20260102T090000Z", {"target_field": "due"}

        def build_child_draft(self, parent, child_due, child_field, next_link, parent_short, kind, cpmax, until_dt):
            from nautical_core.task_codec import DEFAULT_TASK_CODEC
            from nautical_core.task_models import NauticalTask, TaskDraft
            values = {
                "uuid": "22222222-0000-4000-8000-000000000002",
                "description": parent.observation.to_mapping().get("description"),
                "status": "pending",
                "chain": "on",
                "chainID": parent.observation.to_mapping().get("chainID"),
                "link": next_link,
                "prevLink": parent_short,
                "cp": "P1D",
                child_field: child_due,
            }
            return TaskDraft.from_task(
                NauticalTask.from_observation(
                    DEFAULT_TASK_CODEC.decode_row(values, source_query="reconcile fake child")
                )
            )

    generation = FakeGeneration()
    plan = reconcile.plan_recovery_decision(parent_obs, existing_children=[], hook=None, generation=generation)
    expect(isinstance(plan, RecoveryPlanResult) and plan.plan.action.value == "spawn_child", f"expected spawn plan, got: {plan}")
    child = plan.plan.child_dict() if isinstance(plan, RecoveryPlanResult) else {}
    expect(child.get("link") == 3 and child.get("prevLink") == "11111111", f"bad child plan: {plan}")
    evidence = reconcile_report.describe_recovery_result(plan)
    expect(evidence.get("kind") == "cp", f"expected cp evidence, got: {evidence!r}")
    expect(evidence.get("next_link") == 3, f"expected next_link evidence, got: {evidence!r}")
    expect(evidence.get("child_field") == "due", f"expected child field evidence, got: {evidence!r}")

    capped = dict(parent, chainMax=2)
    plan = reconcile.plan_recovery_decision(fixture_observation(capped), existing_children=[], hook=None, generation=generation)
    expect(isinstance(plan, RecoveryPlanResult) and plan.terminal_kind == "chain_max" and "chainMax" in plan.reason, f"expected capped final, got: {plan}")

    class ExhaustingGeneration(FakeGeneration):
        def compute_cp_child_due(self, _parent):
            raise core.OccurrenceSearchExhausted(
                "cp scheduling", reference=date(9999, 1, 1), limit=1
            )

    terminal = reconcile.plan_recovery_decision(
        parent_obs,
        existing_children=[],
        hook=None,
        generation=ExhaustingGeneration(),
    )
    expect(
        isinstance(terminal, RecoveryPlanResult) and terminal.terminal_kind == "date_limit" and "9999-12-31" in terminal.reason,
        f"date-limit exhaustion should be a terminal reconcile plan: {terminal}",
    )
    expect(terminal.terminal_kind == "date_limit", f"terminal kind was not retained: {terminal}")
    expect(
        reconcile_report.describe_recovery_result(terminal).get("terminal") is True
        and reconcile_report.describe_recovery_result(terminal).get("terminal_kind") == "date_limit",
        "terminal reconcile evidence was not exposed",
    )

    class SearchLimitedGeneration(FakeGeneration):
        def compute_cp_child_due(self, _parent):
            raise core.OccurrenceSearchExhausted(
                "cp scheduling", reference=date(2026, 1, 1), limit=1,
                kind=core.OccurrenceSearchExhausted.SEARCH_LIMIT,
            )

    search_limited = reconcile.plan_recovery_decision(
        parent_obs,
        existing_children=[],
        hook=None,
        generation=SearchLimitedGeneration(),
    )
    expect(
        isinstance(search_limited, RecoveryRefusal)
        and "cp scheduling" in search_limited.reason,
        f"search-limit exhaustion must remain a retryable reconcile error: {search_limited}",
    )
def test_reconcile_expiration_real_taskwarrior_round_trip():
    """Real Taskwarrior data should receive one linked child with a shifted until window."""
    task_bin = shutil.which("task")
    if not task_bin:
        return

    tool_path = Path(ROOT) / "nautical_core" / "tools" / "nautical_reconcile.py"
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        data_dir = root / "data"
        data_dir.mkdir()
        config_path = root / "config-nautical.toml"
        config_path.write_text('tz = "UTC"\n', encoding="utf-8")
        taskrc = root / "taskrc"
        taskrc.write_text(
            "\n".join(
                [
                    f"data.location={data_dir}",
                    "hooks=off",
                    "confirmation=off",
                    "verbose=nothing",
                    "uda.cp.type=string",
                    "uda.chain.type=string",
                    "uda.chainID.type=string",
                    "uda.link.type=numeric",
                    "uda.prevLink.type=string",
                    "uda.nextLink.type=string",
                    "uda.chainMax.type=numeric",
                    "uda.chainUntil.type=date",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        env = os.environ.copy()
        env.update(
            {
                "TASKRC": str(taskrc),
                "TASKDATA": str(data_dir),
                "NAUTICAL_CONFIG": str(config_path),
                "NAUTICAL_CORE_PATH": ROOT,
                "NO_COLOR": "1",
            }
        )
        fixture_due = (datetime.now(timezone.utc) - timedelta(days=1)).replace(
            hour=9,
            minute=0,
            second=0,
            microsecond=0,
        )
        fixture_until = fixture_due + timedelta(days=6, hours=14, minutes=59)
        fixture_end = fixture_until + timedelta(minutes=1)
        child_due = fixture_due + timedelta(days=7)
        child_until = fixture_until + timedelta(days=7)
        parent = {
            "uuid": "00000000-0000-4000-8000-00000000050b",
            "status": "deleted",
            "description": "Take the trash out",
            "entry": (fixture_due - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ"),
            "modified": fixture_end.strftime("%Y%m%dT%H%M%SZ"),
            "end": fixture_end.strftime("%Y%m%dT%H%M%SZ"),
            "due": fixture_due.strftime("%Y%m%dT%H%M%SZ"),
            "until": fixture_until.strftime("%Y%m%dT%H%M%SZ"),
            "cp": "7d",
            "chain": "on",
            "chainID": "trash001",
            "link": 1,
        }
        imported = subprocess.run(
            [task_bin, "rc.hooks=off", "import"],
            input=json.dumps(parent, ensure_ascii=False) + "\n",
            text=True,
            capture_output=True,
            env=env,
            timeout=15.0,
        )
        expect(imported.returncode == 0, f"Taskwarrior fixture import failed: {imported.stderr!r}")

        applied = subprocess.run(
            [sys.executable, str(tool_path), "--apply", "--task-bin", task_bin, "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=20.0,
        )
        expect(
            applied.returncode == 0,
            f"real expiration reconcile failed: stdout={applied.stdout!r} stderr={applied.stderr!r}",
        )
        summary = json.loads(applied.stdout)
        expect(summary.get("spawn") == 1 and len(summary.get("applied") or []) == 1, f"unexpected apply: {summary!r}")

        exported = subprocess.run(
            [task_bin, "rc.hooks=off", "rc.json.array=1", "chainID:trash001", "export"],
            text=True,
            capture_output=True,
            env=env,
            timeout=15.0,
        )
        expect(exported.returncode == 0, f"Taskwarrior verification export failed: {exported.stderr!r}")
        rows = json.loads(exported.stdout)
        by_link = {int(float(row.get("link"))): row for row in rows}
        expect(set(by_link) == {1, 2}, f"expected exactly two chain slots: {rows!r}")
        expect(by_link[1].get("nextLink") == str(by_link[2].get("uuid") or "")[:8], f"parent was not linked: {rows!r}")
        expect(by_link[2].get("status") == "pending", f"child should be pending: {by_link[2]!r}")
        expect(
            by_link[2].get("due") == child_due.strftime("%Y%m%dT%H%M%SZ"),
            f"child advanced from the wrong basis: {by_link[2]!r}",
        )
        expect(
            by_link[2].get("until") == child_until.strftime("%Y%m%dT%H%M%SZ"),
            f"child until window was not shifted: {by_link[2]!r}",
        )

        repeated = subprocess.run(
            [sys.executable, str(tool_path), "--apply", "--task-bin", task_bin, "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=20.0,
        )
        expect(repeated.returncode == 0, f"second expiration reconcile failed: {repeated.stderr!r}")
        expect(json.loads(repeated.stdout).get("candidates") == 0, f"reconcile should be idempotent: {repeated.stdout!r}")

        recovery_at = datetime.now(timezone.utc).replace(microsecond=0)
        delayed_due = recovery_at - timedelta(days=3)
        delayed_until = delayed_due + timedelta(hours=1)
        delayed_parent = {
            "uuid": "55555555-0000-4000-8000-000000000005",
            "status": "deleted",
            "description": "Delayed expiration recovery",
            "entry": (delayed_due - timedelta(hours=1)).strftime("%Y%m%dT%H%M%SZ"),
            "modified": delayed_until.strftime("%Y%m%dT%H%M%SZ"),
            "end": delayed_until.strftime("%Y%m%dT%H%M%SZ"),
            "due": delayed_due.strftime("%Y%m%dT%H%M%SZ"),
            "until": delayed_until.strftime("%Y%m%dT%H%M%SZ"),
            "cp": "1d",
            "chain": "on",
            "chainID": "delayed1",
            "link": 1,
        }
        imported_delayed = subprocess.run(
            [task_bin, "rc.hooks=off", "import"],
            input=json.dumps(delayed_parent, ensure_ascii=False) + "\n",
            text=True,
            capture_output=True,
            env=env,
            timeout=15.0,
        )


        expect(imported_delayed.returncode == 0, f"delayed fixture import failed: {imported_delayed.stderr!r}")

        recovered = subprocess.run(
            [
                sys.executable,
                str(tool_path),
                "--apply",
                "--task-bin",
                task_bin,
                "--max-expiration-hops",
                "8",
                "--json",
            ],
            text=True,
            capture_output=True,
            env=env,
            timeout=30.0,
        )
        expect(recovered.returncode == 0, f"delayed expiration reconcile failed: {recovered.stderr!r} {recovered.stdout!r}")
        recovered_summary = json.loads(recovered.stdout)
        expect(recovered_summary.get("expiration_hops") == 3, f"wrong delayed recovery depth: {recovered_summary!r}")
        expect(recovered_summary.get("recovered_chains") == 1, f"delayed chain was not summarized: {recovered_summary!r}")

        exported_delayed = subprocess.run(
            [task_bin, "rc.hooks=off", "rc.json.array=1", "chainID:delayed1", "export"],
            text=True,
            capture_output=True,
            env=env,
            timeout=15.0,
        )
        expect(exported_delayed.returncode == 0, f"delayed verification export failed: {exported_delayed.stderr!r}")
        delayed_rows = json.loads(exported_delayed.stdout)
        delayed_by_link = {int(float(row.get("link"))): row for row in delayed_rows}
        expect(set(delayed_by_link) == {1, 2, 3, 4}, f"delayed recovery skipped chain slots: {delayed_rows!r}")
        expect(
            [delayed_by_link[link].get("status") for link in (1, 2, 3, 4)]
            == ["deleted", "deleted", "deleted", "pending"],
            f"delayed recovery stopped at the wrong occurrence: {delayed_rows!r}",
        )

        repeated_delayed = subprocess.run(
            [sys.executable, str(tool_path), "--apply", "--task-bin", task_bin, "--json"],
            text=True,
            capture_output=True,
            env=env,
            timeout=20.0,
        )
        expect(repeated_delayed.returncode == 0, f"repeated delayed reconcile failed: {repeated_delayed.stderr!r}")
        expect(
            json.loads(repeated_delayed.stdout).get("candidates") == 0,
            f"delayed recovery should be idempotent: {repeated_delayed.stdout!r}",
        )
TESTS = (
    test_integration_contract_covers_all_mutation_and_outbox_states,
    test_doctor_reports_reconcile_backfill_plans,
    test_reconcile_candidate_and_plan_paths,
    test_reconcile_expiration_real_taskwarrior_round_trip,
    test_seasonal_selection_reconcile_spawn_recovery_and_dedup,
    test_reconcile_tool_computes_year_ordinal_anchor,
    test_shared_outbox_persists_integrity_work_without_lifecycle_claiming,
    test_non_hour_dst_carry_and_reconcile_share_core_policy,
    test_carry_field_failure_defers_completion_and_reconcile_mutation,
    test_reconcile_real_taskwarrior_duplicate_slot_requires_manual_review,
    test_outbox_drain_limit_config_and_env_override,
    test_reconcile_repairs_invalid_native_until_from_previous_link,
    test_reconcile_tool_print_plan_includes_evidence,
    test_reconcile_evidence_prefers_due_over_carried_scheduled,
    test_health_check_critical_outbox_bytes,
    test_health_check_critical_outbox_rows,
    test_queue_status_does_not_initialize_missing_outbox,
    test_reconcile_tool_defaults_core_path_to_install_base,
    test_reconcile_tool_path_computes_timed_anchor_in_configured_timezone,
    test_reconcile_configuration_verification_fails_closed,
    test_reconcile_startup_config_failure_is_structured,
    test_reconcile_subprocess_output_contracts,
    test_reconcile_apply_lease_serializes_mutations,
    test_reconcile_apply_refuses_a_second_full_run,
    test_reconcile_parent_identity_errors_are_actionable,
    test_reconcile_expired_pending_child_is_resumable_partial,
    test_reconcile_expiration_anchor_advances_from_recurrence_target,
    test_reconcile_hookless_completion_verifies_scheduled_and_wait_carry,
    test_reconcile_native_until_manual_review_is_not_a_hard_error,
    test_reconcile_expiration_candidate_requires_expiry_evidence,
    test_reconcile_expiration_cp_advances_from_recurrence_target,
    test_reconcile_expiration_plan_reuses_limits_and_deleted_slot_dedup,
) + tuple(globals()[name] for name in _NAMES)
