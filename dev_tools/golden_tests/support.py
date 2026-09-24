"""Shared assertions for golden-test domain collections."""

from __future__ import annotations

import contextlib
import json
import inspect
import importlib.util
import importlib.machinery
import sys
import sqlite3
import subprocess
import sys
import os
import re
import time as _time
from datetime import date, datetime
from pathlib import Path


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@contextlib.contextmanager
def test_term(value: str):
    """Run terminal-sensitive tests with an explicit TERM and restore it."""
    previous = os.environ.get("TERM")
    os.environ["TERM"] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TERM", None)
        else:
            os.environ["TERM"] = previous


def iso(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = str(value)
    match = re.match(r"^(\d{4})(\d{2})(\d{2})$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return text


def parse_due(value):
    if not value:
        return None
    if isinstance(value, (datetime, date)):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except Exception:
        pass
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except Exception:
        return None


def astral_test_available() -> bool:
    """Require Astral only in the CI astronomy matrix; keep local tests optional."""
    try:
        from astral import Observer, moon, sun  # noqa: F401
        return True
    except ImportError:
        if os.environ.get("NAUTICAL_REQUIRE_ASTRAL") == "1":
            raise AssertionError("Astral is required for this test job; install requirements.txt")
        return False


def typed_command_result(cmd, ok: bool, stdout: str = "", stderr: str = ""):
    """Build typed command evidence for isolated hook tests."""
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult

    kind = CommandFailureKind.SUCCESS if ok else CommandFailureKind.REJECTED
    return TaskCommandResult(
        TaskCommand(tuple(str(part) for part in cmd), "isolated hook test", 1.0),
        0 if ok else 1,
        str(stdout or ""),
        str(stderr or ""),
        kind,
        1,
        0.0,
    )


def task_observations(rows):
    """Decode fixture rows through the same boundary as Taskwarrior exports."""
    from nautical_core.task_codec import DEFAULT_TASK_CODEC

    return tuple(DEFAULT_TASK_CODEC.decode_row(row, source_query="golden fixture") for row in rows)


def task_observation(row):
    return task_observations((row,))[0]


def task_snapshot(row):
    """Build lifecycle snapshots through the observation boundary."""
    from nautical_core.lifecycle_models import TaskSnapshot

    return TaskSnapshot.from_observation(task_observations((row,))[0])


def chain_node(row):
    """Build an integrity node through the production observation boundary."""
    from nautical_core.chain_integrity_models import ChainNode

    return ChainNode.from_observation(task_observation(row))


def task_draft(row):
    """Build a typed child fixture without using a mapping seam."""
    from datetime import datetime
    from nautical_core.task_models import NauticalTask, TaskDraft

    normalized = {
        key: value.isoformat().replace("+00:00", "Z") if isinstance(value, datetime) else value
        for key, value in row.items()
    }
    observation = task_observation(normalized)
    task = NauticalTask.from_observation(observation)
    target_field = "due" if task.temporal.due is not None else "scheduled"
    target = task.temporal.due or task.temporal.scheduled
    if target is None:
        raise AssertionError("typed child fixture requires a target")
    excluded = {
        "id", "uuid", "status", "modified", "end", "chainID", "link", "prevLink", "nextLink",
        "description", "chain", "anchor", "anchor_file", "anchor_mode", "cp", "omit", "omit_file",
        "bc", "chainMax", "chainUntil", "due", "scheduled",
    }
    values = observation.to_mapping()
    return TaskDraft(
        identity=task.identity,
        description=task.description,
        recurrence=task.recurrence,
        target=target,
        fields={key: value for key, value in values.items() if key not in excluded},
        target_field=target_field,
    )


def recovery_action(result):
    """Project typed recovery results for characterization assertions."""
    from nautical_core.lifecycle_recovery_models import RecoveryPlanResult

    if not isinstance(result, RecoveryPlanResult):
        return result.status.value
    return {
        "spawn_child": "spawn",
        "update_parent": "backfill_nextlink",
        "finalize_chain": "legitimate_final",
        "disable_chain": "manual_stop",
    }.get(result.plan.action.value, result.plan.action.value)


def recovery_child(result):
    from nautical_core.lifecycle_recovery_models import RecoveryPlanResult

    return result.plan.child_dict() if isinstance(result, RecoveryPlanResult) else None


def recovery_plan(reconcile, parent, **kwargs):
    if "existing_children" in kwargs:
        kwargs["existing_children"] = [fixture_observation(child) for child in kwargs["existing_children"]]
    return reconcile.plan_recovery_decision(fixture_observation(parent), **kwargs)


def must_parse(expr):
    import nautical_core

    try:
        return nautical_core.validate_anchor_expr_strict(expr)
    except Exception as error:
        raise AssertionError(f"Failed to parse '{expr}': {error}")


def new_lifecycle_read_service():
    import nautical_core

    read_service = nautical_core._import_sibling("lifecycle_read_service")
    missing = object()
    return read_service.LifecycleReadService(
        coerce_int=nautical_core.coerce_int,
        parse_extra_tokens=lambda _extra: [],
        token_matcher=lambda _row, _token: True,
        read_query_get=lambda _kind, _key: missing,
        chain_cache_get=lambda _chain_id: None,
        repository=object(),
        max_chain_walk=500,
        read_query_missing=missing,
    )


def scheduler_for_fixture(task, *, context=None):
    from nautical_core.scheduler_service import SchedulerService

    return SchedulerService.from_observation(fixture_observation(task, context=context), context=context)


def evaluator_for_fixture(task, *, context=None, timezone_value=None, timezone=None, **context_kwargs):
    from nautical_core.recurrence_evaluator import RecurrenceEvaluator

    if context is None:
        from datetime import timezone as _timezone
        from nautical_core.recurrence_context import RecurrenceContext

        context = RecurrenceContext(
            chain_id=str(dict(task).get("chainID") or ""),
            timezone=timezone_value or timezone or _timezone.utc,
            **context_kwargs,
        )
    return RecurrenceEvaluator.from_observation(fixture_observation(task, context=context), context=context)


def seed_sqlite_queue(db_path, entries):
    items = entries if isinstance(entries, list) else [entries]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_path)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS queue_entries (id INTEGER PRIMARY KEY AUTOINCREMENT, spawn_intent_id TEXT, payload TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL DEFAULT 'queued', claim_token TEXT, claimed_at REAL, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
        for item in items:
            if not isinstance(item, dict):
                continue
            payload = dict(item)
            state = str(payload.pop("__queue_state", "queued") or "queued")
            claim_token = payload.pop("__claim_token", None)
            claimed_at = payload.pop("__claimed_at", None)
            created_at = float(payload.pop("__created_at", 1.0) or 1.0)
            updated_at = float(payload.pop("__updated_at", created_at) or created_at)
            try:
                attempts = int(payload.get("attempts") or 0)
            except Exception:
                attempts = 0
            connection.execute("INSERT INTO queue_entries (spawn_intent_id, payload, attempts, state, claim_token, claimed_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (str(payload.get("spawn_intent_id") or "").strip() or None, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), attempts, state, claim_token, claimed_at, created_at, updated_at))
        connection.commit()


def build_preview(expr, mode="ALL", due=None):
    import nautical_core

    due_dt = parse_due(due)
    natural = ""
    upcoming = []
    first_due = None
    if hasattr(nautical_core, "build_and_cache_hints"):
        try:
            package = nautical_core.build_and_cache_hints(expr, mode, default_due_dt=due_dt)
            if package:
                natural = package.get("natural") or natural
                upcoming = [iso(item) for item in package.get("next_dates") or []]
                if package.get("first_due"):
                    first_due = iso(package["first_due"])
                return {"natural": natural, "upcoming": upcoming, "first_due": first_due}
        except Exception:
            pass
    nautical_core.validate_anchor_expr_strict(expr)
    if hasattr(nautical_core, "describe_anchor_expr"):
        try:
            natural = nautical_core.describe_anchor_expr(expr, default_due_dt=due_dt)
        except Exception:
            natural = ""
    return {"natural": natural, "upcoming": upcoming, "first_due": first_due}


def must_preview(expr, due=None):
    package = build_preview(expr, due=due)
    if not package or not package.get("upcoming"):
        raise AssertionError(f"No upcoming dates for '{expr}'")
    from datetime import datetime

    return {"next_dates": [datetime.fromisoformat(value).date() for value in package["upcoming"]]}


def must_natural(expr):
    import nautical_core

    try:
        if hasattr(nautical_core, "describe_anchor_expr"):
            natural = nautical_core.describe_anchor_expr(expr)
            if natural:
                return natural
    except Exception:
        pass
    package = build_preview(expr)
    if package and package.get("natural"):
        return package["natural"]
    raise AssertionError(f"No natural language for '{expr}'")


def run_hook_script(path: str, task_obj: dict, env_extra: dict | None = None, timeout_s: float = 8.0):
    force_tz_utc()
    env = os.environ.copy()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TZ", "UTC")
    if env_extra:
        env.update({key: str(value) for key, value in env_extra.items()})
    return subprocess.run([sys.executable, path], input=json.dumps(canonical_hook_fixture(task_obj)), text=True, capture_output=True, env=env, timeout=timeout_s)


def run_hook_script_raw(path: str, raw_input: str, env_extra: dict | None = None, timeout_s: float = 8.0):
    force_tz_utc()
    env = os.environ.copy()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TZ", "UTC")
    if env_extra:
        env.update({key: str(value) for key, value in env_extra.items()})
    return subprocess.run([sys.executable, path], input=raw_input, text=True, capture_output=True, env=env, timeout=timeout_s)


def modify_effect(hook, name, *args, **kwargs):
    effects = importlib.import_module("nautical_core.modify_composition_adapters")
    name = {"expiration_services": "expiration_services_for"}.get(name, name)
    return getattr(effects, name)(hook, *args, **kwargs)


def fixture_observation(task, *, context=None):
    """Decode a compact fixture through the typed observation boundary."""
    from nautical_core.task_models import TaskObservation

    if isinstance(task, TaskObservation):
        return task
    values = dict(task)
    recurrence_fixture = any(values.get(field) for field in ("cp", "anchor", "anchor_file"))
    if context is not None and not values.get("chainID"):
        values["chainID"] = context.chain_id
    if recurrence_fixture and not values.get("chainID"):
        values["chainID"] = "fixture-chain"
    if values.get("chainID"):
        values.setdefault("uuid", "00000000-0000-4000-8000-000000000001")
        values.setdefault("description", "typed fixture task")
        values.setdefault("status", "pending")
        values.setdefault("link", 1)
    return task_observation(values)


def fixture_task(task, *, context=None):
    from nautical_core.task_models import NauticalTask

    values = dict(task)
    values.setdefault("uuid", "00000000-0000-4000-8000-000000000001")
    values.setdefault("status", "pending")
    values.setdefault("chainID", context.chain_id if context is not None else "fixture-chain")
    values.setdefault("link", 1)
    values.setdefault("chain", "on")
    if not any(values.get(field) for field in ("cp", "anchor", "anchor_file")):
        values["cp"] = "P1D"
    return NauticalTask.from_observation(fixture_observation(values, context=context))


def plan_from_values(**kwargs):
    from nautical_core.lifecycle_models import LifecyclePlan, _freeze_pairs

    child_payload = kwargs.pop("child_payload", None)
    parent_patch = kwargs.pop("parent_patch", None)
    if child_payload is not None and getattr(kwargs.get("action"), "value", kwargs.get("action")) == "spawn_child":
        child_payload = dict(child_payload)
        child_payload.setdefault("description", "typed lifecycle child")
        child_payload.setdefault("status", "pending")
        child_payload.setdefault("chain", "on")
        child_payload.setdefault("anchor", "")
        child_payload.setdefault("anchor_file", "")
        child_payload.setdefault("omit", "")
        child_payload.setdefault("omit_file", "")
        child_payload.setdefault("anchor_mode", "skip")
        child_payload.setdefault("cp", "1d")
        child_payload.setdefault("due", "2026-01-02T00:00:00Z")
    return LifecyclePlan(
        **kwargs,
        child_payload=_freeze_pairs(child_payload),
        parent_patch=_freeze_pairs(parent_patch),
    )


def child_payload_from_values(payload, *, parent_uuid):
    from nautical_core.integration_models import ChildImportPayload, _coerce_payload_link, _freeze_pairs

    target_link = _coerce_payload_link(payload.get("link"))
    if target_link is None:
        raise ValueError("test child payload requires an integer link")
    return ChildImportPayload(
        parent_uuid,
        str(payload.get("uuid") or ""),
        str(payload.get("chainID") or ""),
        target_link,
        _freeze_pairs(payload),
    )


def metadata_payload_from_values(task_uuid, updates, *, expected=None):
    from nautical_core.integration_models import MetadataRepairPayload, _freeze_pairs

    return MetadataRepairPayload(task_uuid, _freeze_pairs(updates), _freeze_pairs(expected or {}))


def has_function(name):
    import nautical_core

    return hasattr(nautical_core, name)


def force_tz_utc():
    os.environ["TZ"] = "UTC"
    try:
        _time.tzset()
    except Exception:
        pass


def extract_last_json(stdout_text: str) -> dict:
    text = (stdout_text or "").strip()
    if not text:
        raise AssertionError("Hook produced no stdout JSON.")
    candidates = re.findall(r"\{[\s\S]*\}", text)
    if not candidates:
        raise AssertionError(f"Could not locate JSON in hook stdout. stdout={text[:200]!r}")
    try:
        return json.loads(candidates[-1])
    except Exception as error:
        raise AssertionError(f"Invalid JSON from hook stdout: {error}. stdout_tail={candidates[-1][-200:]!r}")


def assert_stdout_json_only(stdout_text: str) -> dict:
    text = (stdout_text or "").strip()
    if not text:
        raise AssertionError("Hook produced no stdout JSON.")
    obj, index = json.JSONDecoder().raw_decode(text)
    if text[index:].strip():
        raise AssertionError("Hook stdout contains non-JSON content.")
    if not isinstance(obj, dict):
        raise AssertionError(f"Hook stdout JSON is not an object: {type(obj).__name__}")
    return obj


def call_with_supported_kwargs(fn, **kwargs):
    signature = inspect.signature(fn)
    accepts_var_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    filtered = dict(kwargs) if accepts_var_kwargs else {k: v for k, v in kwargs.items() if k in signature.parameters}
    for key in ("task", "new", "old", "parent", "child"):
        value = filtered.get(key)
        if isinstance(value, dict) and {"anchor", "anchor_file", "cp", "chainID"}.intersection(value):
            value = dict(value)
            value.setdefault("uuid", "00000000-0000-4000-8000-000000000701")
            value.setdefault("status", "pending")
            value.setdefault("link", 1)
            value.setdefault("chainID", "fixture-domain")
            value.setdefault("chain", "on")
            filtered[key] = value
    return fn(**filtered)


def strip_markup(value: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", value or "")


def doctor_findings(payload):
    normalized = []
    for item in payload.get("operator_findings") or []:
        if not isinstance(item, dict):
            continue
        evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        details = dict(evidence)
        if isinstance(item.get("observed"), dict) and item["observed"]:
            details["observed"] = item["observed"]
        if isinstance(item.get("expected"), dict) and item["expected"]:
            details["expected"] = item["expected"]
        normalized.append({"id": item.get("code"), "severity": "warn" if item.get("severity") == "warning" else "ok" if item.get("severity") == "info" else item.get("severity"), "message": item.get("message"), "fix": item.get("guidance") or "", "details": details})
    return normalized


def doctor_hook_installation(mod, findings, *, hooks_dir, env):
    typed, validated = mod.OperatorHealthService.hook_installation_findings(hooks_dir, mod.install_runtime.HOOK_RUNTIME_FILES, mod.install_runtime.hook_candidates, mod.install_runtime.inspect_hook_runtime, env)
    findings.extend(item.to_doctor_dict() for item in typed)
    return validated


def doctor_obsolete_queue_state(mod, findings, taskdata):
    paths = sorted(str(root / name) for root in (taskdata, taskdata / ".nautical-state") for name in mod._OBSOLETE_QUEUE_STATE_NAMES if os.path.lexists(root / name))
    findings.extend(item.to_doctor_dict() for item in mod.OperatorHealthService.obsolete_queue_findings(taskdata, mod._OBSOLETE_QUEUE_STATE_NAMES))
    return paths


_OPERATOR_TEMPORARIES = []


def test_operator_uow(taskdata=None):
    from datetime import timezone
    from tempfile import TemporaryDirectory
    from nautical_core.integration_context import IntegrationAccess, IntegrationContext, SilentDiagnostics, SystemClock, ValidatedNauticalConfiguration
    from nautical_core.taskwarrior_uow import TaskwarriorUnitOfWork

    if taskdata is None:
        temporary = TemporaryDirectory(prefix="nautical-test-taskdata-")
        _OPERATOR_TEMPORARIES.append(temporary)
        taskdata = temporary.name
    context = IntegrationContext(Path(taskdata).resolve(), "test", ("task",), ValidatedNauticalConfiguration("test", "config", "scheduler", "UTC", ()), timezone.utc, SilentDiagnostics(), SystemClock(), "test-operator", 256, IntegrationAccess.MUTATION)
    return TaskwarriorUnitOfWork.create(context, env={})


def load_core_module(path: str, module_name: str, config_path: str):
    previous = os.environ.get("NAUTICAL_CONFIG")
    os.environ["NAUTICAL_CONFIG"] = config_path
    try:
        spec = importlib.util.spec_from_file_location(module_name, path, submodule_search_locations=[os.path.dirname(path)])
        if spec is None or spec.loader is None:
            raise ImportError(f"could not create package spec for {path}")
        module = importlib.util.module_from_spec(spec)
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if root not in sys.path:
            sys.path.insert(0, root)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        refresh = getattr(module, "_refresh_facade_config_exports", None)
        if callable(refresh):
            try:
                refresh()
            except Exception:
                pass
        return module
    finally:
        if previous is None:
            os.environ.pop("NAUTICAL_CONFIG", None)
        else:
            os.environ["NAUTICAL_CONFIG"] = previous


def load_hook_protocol_module(module_name: str):
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return legacy._load_hook_module(os.path.join(root, "nautical_core", "hook_protocol.py"), module_name)


def load_exit_probe_module(module_name: str):
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return legacy._load_hook_module(os.path.join(root, "nautical_core", "exit_probe.py"), module_name)


def load_hook_module(path: str, module_name: str):
    legacy = importlib.import_module("dev_tools.nautical_golden_tests")
    force_tz_utc()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if os.path.basename(path) in {"on-add.nautical", "on-modify.nautical", "on-exit.nautical"}:
        path = os.path.join(root, "nautical_core", "hooks", {"on-add.nautical": "add_impl.py", "on-modify.nautical": "modify_impl.py", "on-exit.nautical": "exit_impl.py"}[os.path.basename(path)])
    is_core_package = os.path.basename(path) == "__init__.py" and os.path.basename(os.path.dirname(path)) == "nautical_core"
    if is_core_package:
        spec = importlib.util.spec_from_file_location(module_name, path, submodule_search_locations=[os.path.dirname(path)])
    else:
        spec = importlib.util.spec_from_loader(module_name, importlib.machinery.SourceFileLoader(module_name, path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    if root not in sys.path:
        sys.path.insert(0, root)
    if is_core_package:
        sys.modules[module_name] = module
    spec.loader.exec_module(module)
    load_core = getattr(module, "_load_core", None)
    if callable(load_core) and os.path.basename(path) in {"add_impl.py", "modify_impl.py", "exit_impl.py"}:
        load_core()
    if os.path.basename(path) == "modify_impl.py":
        module._completion_effects = legacy._BoundCompletionEffects(module)
        module._transition_effects = legacy._BoundTransitionEffects(module)
        module._presentation_effects = legacy._BoundPresentationEffects(module)
        module._diagnostics_effects = legacy._BoundDiagnosticsEffects(module)
        schedule_effects = importlib.import_module("nautical_core.modify_schedule_effects")
        cp_ports = schedule_effects.cp_completion_ports_for(module)
        anchor_ports = schedule_effects.anchor_completion_ports_for(module)
        module._estimate_cp_final_by_max = lambda task, due: schedule_effects.estimate_cp_final_by_max(cp_ports, task, due)
        module._estimate_anchor_final_by_max = lambda task, due, dnf: schedule_effects.estimate_anchor_final_by_max(anchor_ports, task, due, dnf)
        module._cap_from_until_cp = lambda task, due: schedule_effects.cap_from_until_cp(cp_ports, task, due)
        module._cap_from_until_anchor = lambda task, due, dnf: schedule_effects.cap_from_until_anchor(anchor_ports, task, due, dnf)
        def timeline_lines(kind, task, child_due_utc, child_short, dnf, **kwargs):
            override = getattr(module, "_collect_prev_two", None)
            if callable(override):
                kwargs["_collect_prev_two_override"] = override
            return module._presentation_effects.timeline_lines(kind, task, child_due_utc, child_short, dnf, **kwargs)
        module._timeline_lines = timeline_lines
    return module


def find_hook_file(name: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [os.path.join(root, name), os.path.join(root, "hooks", name)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise AssertionError(f"Hook script '{name}' not found. Expected at '{candidates[0]}' or '{candidates[1]}'.")


def canonical_hook_fixture(task_obj: dict) -> dict:
    """Fill structural defaults required by typed Nautical hook inputs."""
    values = dict(task_obj)
    recurrence_fields = {"anchor", "anchor_file", "cp", "chain", "chainID", "link"}
    if not recurrence_fields.intersection(values):
        return values
    uuid_value = str(values.get("uuid") or "")
    generated_identity = len(uuid_value) != 36 or uuid_value.count("-") != 4
    if generated_identity:
        values["uuid"] = "00000000-0000-4000-8000-000000000001"
    values.setdefault("status", "pending")
    values.setdefault("link", 1)
    if generated_identity:
        values.setdefault("chainID", "fixture-chain")
    values.setdefault("chain", "on")
    return values


def generation_service(hook):
    from nautical_core.chain_generation import ChainGenerationService

    return ChainGenerationService.from_hook(hook)


def compute_anchor_child_due(hook, parent):
    from nautical_core.task_models import NauticalTask

    return generation_service(hook).compute_anchor_child_due(NauticalTask.from_observation(fixture_observation(parent)))


def compute_cp_child_due(hook, parent):
    from nautical_core.task_models import NauticalTask

    return generation_service(hook).compute_cp_child_due(NauticalTask.from_observation(fixture_observation(parent)))


def carry_relative_datetime(hook, parent, child, child_due, field, **kwargs):
    return generation_service(hook).carry_relative_datetime(
        parent if hasattr(parent, "observation") else fixture_task(parent), child, child_due, field,
        parent_anchor_field=kwargs.pop("parent_anchor_field", "due"),
        child_anchor_field=kwargs.pop("child_anchor_field", "due"), **kwargs,
    )


def carry_native_until(hook, parent, child, child_due, kind, **kwargs):
    return generation_service(hook).carry_native_until(
        parent if hasattr(parent, "observation") else fixture_task(parent), child, child_due, kind,
        parent_anchor_field=kwargs.pop("parent_anchor_field", "due"),
        child_anchor_field=kwargs.pop("child_anchor_field", "due"), **kwargs,
    )


def build_child_draft_for_test(hook, parent, child_due, child_field, next_link, parent_short, kind, cpmax, until_dt):
    from nautical_core.task_models import NauticalTask

    return generation_service(hook).build_child_draft(
        NauticalTask.from_observation(fixture_observation(parent)),
        child_due, child_field, next_link, parent_short, kind, cpmax, until_dt,
    ).to_mapping()


def found_task(row):
    from nautical_core.integration_models import Found

    return Found(row, "isolated test read")


def absent_task(reason: str = "not found"):
    from nautical_core.integration_models import Absent

    return Absent("isolated test read", reason)


def unavailable_task(reason: str = "database is locked"):
    from nautical_core.integration_models import CommandFailureKind, FailureEvidence, TaskCommand, Unavailable

    command = TaskCommand(("task", "export"), "isolated test read", 1.0)
    evidence = FailureEvidence(command, CommandFailureKind.BUSY, 1, 1, 0.0, True, reason)
    return Unavailable("isolated test read", evidence)
