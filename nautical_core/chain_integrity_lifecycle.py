from __future__ import annotations

from datetime import datetime, time, timedelta
from dataclasses import dataclass
from typing import Any

from nautical_core import astronomy, native_until
from nautical_core.chain_generation import ChainGenerationService
from nautical_core.timeutil import compare_datetimes
from nautical_core.scheduler_service import SchedulerService
from nautical_core.scheduler_models import OccurrenceSearchExhausted, occurrence_exhaustion_message
from nautical_core.task_codec import TaskCodec
from nautical_core.lifecycle_models import (
    LifecycleAction,
    LifecycleEvent,
    LifecycleIdentity,
    LifecycleRecoveryDecision,
    LifecyclePlan,
    ParentGuard,
    TaskSnapshot,
    recurrence_fingerprint,
)
from nautical_core.lifecycle_planner import (
    LifecyclePreflight,
    RecurrenceCandidate,
    expiration_candidate,
    plan_expiration_successor,
    plan_candidate_successor,
)
from nautical_core.task_models import FieldPresence, NauticalTask, TaskDraft, TaskObservation, TaskPayload
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.lifecycle_models import DeletionDisposition, DeletionEvidence


RECURRENCE_FIELDS = ("anchor", "anchor_file", "cp")


def _child_draft(child: TaskPayload) -> TaskDraft:
    task = NauticalTask.from_observation(
        DEFAULT_TASK_CODEC.decode_row(child, source_query="reconcile child draft")
    )
    return TaskDraft.from_task(task)


def _recurrence_field_text(value: object) -> str:
    """Normalize Taskwarrior's literal null UDA sentinel as an unset value."""
    return TaskCodec.normalize_text(value)


def _generation_service(hook: Any = None) -> ChainGenerationService:
    """Resolve the shared generator without requiring an on-modify module."""
    if isinstance(hook, ChainGenerationService):
        return hook
    if hook is not None:
        return ChainGenerationService.from_hook(hook)
    import nautical_core as core

    return ChainGenerationService.from_core(
        core,
        recurrence_update_udas=tuple(getattr(core, "RECURRENCE_UPDATE_UDAS", ()) or ()),
        debug_wait_sched=bool(getattr(core, "DEBUG_WAIT_SCHED", False)),
    )


def scheduling_error_message(exc: BaseException) -> str:
    """Keep astronomy failures actionable in dry-run and apply plans."""
    return astronomy.scheduling_error_message(exc)


def is_terminal_plan(plan: LifecycleRecoveryDecision) -> bool:
    """Return whether a final plan ended at the representable date boundary."""
    return plan.action == "legitimate_final" and plan.terminal_kind == "date_limit"


def short_uuid(value: object) -> str:
    raw = str(value or "").strip()
    return raw[:8] if raw else ""


def int_or_default(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if not isinstance(value, (str, bytes, int, float)):
        return default
    try:
        return int(value)
    except Exception:
        return default


def is_nautical_recurrence(task: TaskObservation) -> bool:
    return any(str(_observation_value(task, field) or "").strip() for field in RECURRENCE_FIELDS)


def _is_unlinked_active_chain(task: TaskObservation) -> bool:
    if str(_observation_value(task, "chain") or "").strip().lower() != "on":
        return False
    if not str(_observation_value(task, "chainID") or "").strip():
        return False
    if str(_observation_value(task, "nextLink") or "").strip():
        return False
    if not is_nautical_recurrence(task):
        return False
    return True


def is_orphan_completion_candidate(task: TaskObservation) -> bool:
    return str(_observation_value(task, "status") or "").strip() == "completed" and _is_unlinked_active_chain(task)


def is_orphan_deleted_chain_candidate(task: TaskObservation) -> bool:
    return str(_observation_value(task, "status") or "").strip() == "deleted" and _is_unlinked_active_chain(task)


def deleted_chain_disposition(
    task: TaskObservation,
    *,
    safe_parse_datetime: Any,
) -> DeletionEvidence:
    """Classify an unlinked deleted chain as expiration, manual stop, or ambiguous."""
    if not is_orphan_deleted_chain_candidate(task):
        return DeletionEvidence(DeletionDisposition.NOT_APPLICABLE)
    until_raw = _observation_value(task, "until")
    end_raw = _observation_value(task, "end")
    if not str(until_raw or "").strip():
        return DeletionEvidence(DeletionDisposition.MANUAL, "deleted without native until")
    try:
        until_dt, until_err = safe_parse_datetime(until_raw)
        end_dt, end_err = safe_parse_datetime(end_raw)
    except Exception:
        return DeletionEvidence(
            DeletionDisposition.AMBIGUOUS,
            "deleted task has no reliable native-until expiration evidence",
        )
    if until_err or end_err or until_dt is None or end_dt is None:
        return DeletionEvidence(
            DeletionDisposition.AMBIGUOUS,
            "deleted task has no reliable native-until expiration evidence",
        )
    try:
        if compare_datetimes(until_dt, end_dt) <= 0:
            return DeletionEvidence(DeletionDisposition.EXPIRATION, "native until elapsed")
        return DeletionEvidence(DeletionDisposition.MANUAL, "deleted before native until")
    except Exception:
        return DeletionEvidence(
            DeletionDisposition.AMBIGUOUS,
            "deleted task has no reliable native-until expiration evidence",
        )


def is_orphan_expiration_candidate(task: TaskObservation, *, safe_parse_datetime: Any) -> bool:
    """Return whether a deleted link has strong evidence of native until expiration."""
    evidence = deleted_chain_disposition(
        task,
        safe_parse_datetime=safe_parse_datetime,
    )
    return evidence.disposition is DeletionDisposition.EXPIRATION


def compute_expiration_child_due(
    parent: TaskPayload, *, hook: Any = None, generation: ChainGenerationService | None = None
) -> tuple[Any, dict[str, Any]]:
    """Compute the next recurrence target after an expired link without mutating it."""
    generation = generation or _generation_service(hook)
    candidate = expiration_candidate(
        TaskSnapshot.from_observation(DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile expiration")),
        generation=generation,
    )
    return candidate.child_due, dict(candidate.metadata)


def _observation_value(task: TaskObservation, field: str) -> object:
    state = task.field(field)
    if state.presence is FieldPresence.ABSENT:
        return None
    return state.raw_value()


def native_until_target_field(task: TaskObservation) -> str:
    """Return the recurrence target field used by a task."""
    return "due" if _observation_value(task, "due") else "scheduled"


def invalid_relative_carry_reason(
    parent: TaskObservation,
    child: TaskPayload,
    *,
    child_field: str,
    hook: Any = None,
    generation: ChainGenerationService | None = None,
) -> str | None:
    """Verify that scheduled/wait retain their local offset from the recurrence target."""
    generation = generation or _generation_service(hook)
    core = generation.core
    utc_to_local_naive = getattr(core, "utc_to_local_naive", None)
    if not callable(getattr(core, "parse_dt_any", None)) or not callable(utc_to_local_naive):
        return None
    parent_field = native_until_target_field(parent)
    fields = ["wait"]
    if child_field != "scheduled":
        fields.append("scheduled")
    for field in fields:
        parent_value_raw = _observation_value(parent, field)
        if not parent_value_raw:
            continue
        if not child.get(field):
            return f"{field} carry is missing from the reconciled child"
        try:
            parent_target = core.parse_dt_any(_observation_value(parent, parent_field))
            parent_value = core.parse_dt_any(parent_value_raw)
            child_target = core.parse_dt_any(child.get(child_field))
            child_value = core.parse_dt_any(child.get(field))
            if not all((parent_target, parent_value, child_target, child_value)):
                return f"{field} carry contains an unparseable timestamp"
            parent_delta = utc_to_local_naive(parent_value) - utc_to_local_naive(parent_target)
            child_delta = utc_to_local_naive(child_value) - utc_to_local_naive(child_target)
        except Exception as exc:
            return f"{field} carry could not be verified: {exc}"
        if child_delta != parent_delta:
            return f"{field} carry changed its recurrence-target offset"
    return None


def invalid_native_until_reason(
    task: TaskObservation,
    *,
    safe_parse_datetime: Any,
) -> str | None:
    """Describe an invalid native expiration window, if one is present."""
    until_raw = _observation_value(task, "until")
    target_field = native_until_target_field(task)
    target_raw = _observation_value(task, target_field)
    if not until_raw or not target_raw:
        return None
    until_dt, until_err = safe_parse_datetime(until_raw)
    target_dt, target_err = safe_parse_datetime(target_raw)
    if until_err or target_err or until_dt is None or target_dt is None:
        return "native until or recurrence target is not parseable"
    if compare_datetimes(until_dt, target_dt) <= 0:
        return f"native until is not later than {target_field}"
    return None


def repair_native_until_from_previous(
    previous: TaskObservation,
    current: TaskObservation,
    *,
    kind: str,
    safe_parse_datetime: Any,
    fmt_isoz: Any,
    utc_to_local_naive: Any,
    local_naive_to_utc: Any,
) -> tuple[str | None, str | None]:
    """Carry the previous link's native expiration policy onto the current target."""
    parent_field = native_until_target_field(previous)
    child_field = native_until_target_field(current)
    parent_target, parent_target_err = safe_parse_datetime(_observation_value(previous, parent_field))
    parent_until, parent_until_err = safe_parse_datetime(_observation_value(previous, "until"))
    child_target, child_target_err = safe_parse_datetime(_observation_value(current, child_field))
    if parent_target_err or parent_until_err or child_target_err:
        return None, "previous link lacks parseable target/until state"
    if not all((parent_target, parent_until, child_target)):
        return None, "previous link lacks target or native until"
    try:
        repaired = native_until.carry(
            parent_target,
            parent_until,
            child_target,
            kind,
            utc_to_local_naive=utc_to_local_naive,
            local_naive_to_utc=local_naive_to_utc,
        )
    except native_until.NativeUntilCarryError as exc:
        return None, str(exc)
    return fmt_isoz(repaired), None


def fallback_native_until_at_day_end(
    current: TaskObservation,
    *,
    safe_parse_datetime: Any,
    fmt_isoz: Any,
    utc_to_local_naive: Any,
    local_naive_to_utc: Any,
) -> tuple[str | None, str | None]:
    """Use local 23:00 when a prior link cannot provide an expiration policy."""
    target_field = native_until_target_field(current)
    target, target_err = safe_parse_datetime(_observation_value(current, target_field))
    if target_err or target is None:
        return None, f"cannot infer native until without a parseable {target_field}"
    try:
        target_local = utc_to_local_naive(target)
        fallback_local = datetime.combine(target_local.date(), time(23, 0))
        if fallback_local <= target_local:
            return None, f"cannot infer native until: {target_field} is at or after local 23:00"
        return fmt_isoz(local_naive_to_utc(fallback_local)), None
    except Exception:
        return None, "cannot infer native until at local 23:00"


def _child_recurrence_mismatch(parent: TaskPayload, child: TaskPayload) -> str:
    """Return a mismatch when a candidate child carries a different recurrence."""
    if not any(_recurrence_field_text(child.get(field)) for field in RECURRENCE_FIELDS):
        return ""
    for field in RECURRENCE_FIELDS:
        parent_value = _recurrence_field_text(parent.get(field))
        child_value = _recurrence_field_text(child.get(field))
        if child_value and child_value != parent_value:
            expected = parent_value or "<empty>"
            actual = child_value or "<empty>"
            return f"recurrence field {field} is {actual}; expected {expected}"
    return ""


def resolve_existing_child(
    parent: TaskObservation,
    rows: list[TaskObservation] | tuple[TaskObservation, ...],
    *,
    include_deleted: bool = False,
) -> tuple[str, str]:
    if not isinstance(parent, TaskObservation) or any(not isinstance(row, TaskObservation) for row in rows):
        raise TypeError("child resolution requires typed task observations")
    chain_id = str(_observation_value(parent, "chainID") or "").strip()
    next_link = int_or_default(_observation_value(parent, "link"), 1) + 1
    matches: dict[str, TaskObservation] = {}
    for row in rows:
        if str(_observation_value(row, "chainID") or "").strip() != chain_id:
            continue
        if int_or_default(_observation_value(row, "link"), -1) != next_link:
            continue
        if not include_deleted and str(_observation_value(row, "status") or "").strip() == "deleted":
            continue
        child_uuid = str(_observation_value(row, "uuid") or "").strip()
        if len(child_uuid) < 8:
            return "", f"next slot #{next_link} contains a task without a valid UUID"
        matches[child_uuid.lower()] = row

    if not matches:
        return "", ""
    if len(matches) > 1:
        children = ", ".join(sorted(short_uuid(uuid) for uuid in matches))
        return "", f"next slot #{next_link} contains multiple tasks: {children}"

    child_uuid, child = next(iter(matches.items()))
    parent_uuid = str(_observation_value(parent, "uuid") or "").strip().lower()
    parent_short = short_uuid(parent_uuid)
    if len(parent_uuid) < 8 or not parent_short:
        return "", "parent task has no valid UUID for reciprocal link validation"
    prev_link = str(_observation_value(child, "prevLink") or "").strip().lower()
    if prev_link not in {parent_short, parent_uuid}:
        shown = prev_link or "<empty>"
        return (
            "",
            f"next slot #{next_link} child {short_uuid(child_uuid)} has "
            f"prevLink {shown}; expected {parent_short}",
        )
    recurrence_error = _child_recurrence_mismatch(parent.to_mapping(), child.to_mapping())
    if recurrence_error:
        return (
            "",
            f"next slot #{next_link} child {short_uuid(child_uuid)} has "
            f"{recurrence_error}",
        )
    return short_uuid(child_uuid), ""


def recurrence_kind(task: TaskObservation) -> str:
    # Recovery only needs the recurrence family to carry native-until policy;
    # full schedule compilation belongs to the scheduler service.
    if TaskCodec.normalize_text(_observation_value(task, "anchor")):
        return "anchor"
    if TaskCodec.normalize_text(_observation_value(task, "anchor_file")):
        return "anchor_file"
    return "cp"


def describe_plan(plan: LifecycleRecoveryDecision, *, fmt_dt_local: Any = None) -> dict[str, Any]:
    parent = plan.parent
    parent_values = parent.to_mapping() if isinstance(parent, TaskObservation) else parent
    parent_observation = (
        parent
        if isinstance(parent, TaskObservation)
        else DEFAULT_TASK_CODEC.decode_row(parent_values, source_query="lifecycle plan description")
    )
    if plan.action == "manual_stop":
        trigger = "manual_deletion"
    elif str(parent_values.get("status") or "").strip() == "deleted":
        trigger = "expiration"
    else:
        trigger = "completion"
    evidence: dict[str, Any] = {
        "parent": short_uuid(parent_values.get("uuid")),
        "chainID": str(parent_values.get("chainID") or ""),
        "parent_link": int_or_default(parent_values.get("link"), 0),
        "next_link": plan.next_link,
        "kind": recurrence_kind(parent_observation),
        "trigger": trigger,
        "reason": plan.reason,
    }
    if is_terminal_plan(plan):
        evidence["terminal"] = True
        evidence["terminal_kind"] = plan.terminal_kind
    if plan.child_due is not None:
        evidence["child_due"] = str(plan.child_due)
        if callable(fmt_dt_local):
            try:
                evidence["child_local"] = str(fmt_dt_local(plan.child_due))
            except Exception:
                pass
    if plan.child_short:
        evidence["existing_child"] = plan.child_short
    if plan.child:
        field = "due" if "due" in plan.child else "scheduled" if "scheduled" in plan.child else "due"
        evidence["child_field"] = field
        if plan.child.get(field) is not None:
            evidence["child_target"] = str(plan.child.get(field))
    return evidence


def _build_expiration_child_with_day_end(
    parent: TaskPayload,
    *,
    child_due: Any,
    child_field: str,
    next_link: int,
    parent_short: str,
    kind: str,
    cpmax: int,
    until_dt: Any,
    hook: Any,
    generation: ChainGenerationService | None = None,
) -> dict[str, Any]:
    generation = generation or _generation_service(hook)
    target_raw = parent.get("due") or parent.get("scheduled")
    target_dt, target_err = generation.safe_parse_datetime(target_raw)
    if target_err or target_dt is None:
        raise ValueError(target_err or "expired recurrence has no due or scheduled timestamp")
    target_local = generation.core.to_local(target_dt)
    fallback_until = generation.core.build_local_datetime(target_local.date(), (23, 59)) + timedelta(seconds=59)
    fallback_parent = dict(parent)
    fallback_parent["until"] = generation.core.fmt_isoz(fallback_until)
    return generation.build_child_draft(
        NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(fallback_parent, source_query="reconcile expiration fallback")
        ),
        child_due,
        child_field,
        next_link,
        parent_short,
        kind,
        cpmax,
        until_dt,
    ).to_mapping()


def _plan_recovery_decision_unscoped(
    parent: TaskPayload,
    *,
    existing_children: list[TaskObservation] | tuple[TaskObservation, ...],
    hook: Any,
    generation: ChainGenerationService | None = None,
) -> LifecycleRecoveryDecision:
    generation = generation or _generation_service(hook)
    observation = DEFAULT_TASK_CODEC.decode_row(parent, source_query="reconcile recovery")
    try:
        operational_parent = NauticalTask.from_observation(observation)
    except (TypeError, ValueError) as exc:
        return LifecycleRecoveryDecision(
            "error",
            observation,
            int_or_default(parent.get("link"), 1) + 1,
            f"parent task validation failed: {exc}",
        )
    decision_parent = observation
    link = int_or_default(parent.get("link"), 1)
    next_link = link + 1
    is_expiration = str(parent.get("status") or "").strip() == "deleted"
    if is_expiration:
        evidence = deleted_chain_disposition(
            operational_parent.observation,
            safe_parse_datetime=generation.safe_parse_datetime,
        )
        if evidence.disposition is DeletionDisposition.MANUAL:
            return LifecycleRecoveryDecision("manual_stop", decision_parent, next_link, evidence.reason)
        if evidence.disposition is not DeletionDisposition.EXPIRATION:
            return LifecycleRecoveryDecision(
                "error",
                decision_parent,
                next_link,
                evidence.reason or "deleted task has no reliable native-until expiration evidence",
            )

    child_short, child_error = resolve_existing_child(
        decision_parent,
        existing_children,
        include_deleted=is_expiration,
    )
    if child_error:
        return LifecycleRecoveryDecision("error", decision_parent, next_link, child_error)
    if child_short:
        existing_child = next(
            (
                row
                for row in existing_children
                if str(_observation_value(row, "uuid") or "").strip().lower().startswith(child_short.lower())
            ),
            None,
        )
        if not isinstance(existing_child, TaskObservation):
            return LifecycleRecoveryDecision(
                "error",
                decision_parent,
                next_link,
                "existing successor identity could not be loaded",
                child_short=child_short,
            )
        try:
            guard = ParentGuard(
                status=str(parent.get("status") or "pending"),
                chain=str(parent.get("chain") or "on"),
                chain_id=str(parent.get("chainID") or ""),
                link=link,
                recurrence_fingerprint=recurrence_fingerprint(parent),
                modified=str(parent.get("modified") or ""),
            )
            identity = LifecycleIdentity(
                chain_id=guard.chain_id,
                parent_uuid=str(parent.get("uuid") or ""),
                source_link=guard.link,
                target_link=next_link,
                event=LifecycleEvent.EXPIRE if is_expiration else LifecycleEvent.COMPLETE,
            )
            lifecycle_plan = LifecyclePlan.from_draft(
                identity=identity,
                action=LifecycleAction.SPAWN_CHILD,
                parent_guard=guard,
                draft=_child_draft(existing_child.to_mapping()),
                parent_patch={"nextLink": child_short},
                expected_postconditions=("child_present", "parent_linked", "verified"),
            )
        except Exception as exc:
            return LifecycleRecoveryDecision(
                "error",
                decision_parent,
                next_link,
                f"failed to build successor recovery plan: {scheduling_error_message(exc)}",
                child_short=child_short,
            )
        return LifecycleRecoveryDecision(
            "backfill_nextlink",
            decision_parent,
            next_link,
            "next link already exists",
            child_short=child_short,
            child=existing_child.to_mapping(),
            lifecycle_plan=lifecycle_plan,
        )

    try:
        evaluator = SchedulerService.from_task(
            NauticalTask.from_observation(observation)
        ).session.evaluator
        kind = evaluator.kind or "cp"
        limits = evaluator.limits
        until_dt = limits.chain_until
        until_err = None
        cpmax = limits.chain_max or 0
    except ValueError:
        # Keep incomplete legacy rows classifiable; normal validation below
        # still reports malformed chain limits through the hook boundary.
        evaluator = None
        kind = recurrence_kind(operational_parent)
        until_dt, until_err = generation.safe_parse_datetime(parent.get("chainUntil"))
        cpmax = generation.core.coerce_int(parent.get("chainMax"), 0)
    if until_err:
        return LifecycleRecoveryDecision("error", decision_parent, next_link, f"invalid chainUntil: {until_err}")

    if cpmax and next_link > cpmax:
        return LifecycleRecoveryDecision(
            "legitimate_final", decision_parent, next_link, "reached chainMax", terminal_kind="chain_max",
        )

    try:
        if is_expiration:
            expiration = expiration_candidate(
                TaskSnapshot.from_observation(observation),
                generation=generation,
            )
            child_due = expiration.child_due
            meta = dict(expiration.metadata)
        elif kind in {"anchor", "anchor_file"}:
            child_due, meta, _dnf = generation.compute_anchor_child_due(operational_parent)
        else:
            child_due, meta = generation.compute_cp_child_due(operational_parent)
    except Exception as exc:
        if isinstance(exc, OccurrenceSearchExhausted) and exc.is_date_limit:
            return LifecycleRecoveryDecision(
                "legitimate_final",
                decision_parent,
                next_link,
                occurrence_exhaustion_message(exc),
                terminal_kind=exc.kind,
            )
        return LifecycleRecoveryDecision("error", decision_parent, next_link, scheduling_error_message(exc))

    if not child_due:
        return LifecycleRecoveryDecision("error", decision_parent, next_link, "could not compute next recurrence timestamp")
    if until_dt and compare_datetimes(child_due, until_dt) > 0:
        return LifecycleRecoveryDecision(
            "legitimate_final", decision_parent, next_link, "reached chainUntil",
            child_due=child_due, terminal_kind="chain_until",
        )

    child_field = "scheduled" if isinstance(meta, dict) and meta.get("target_field") == "scheduled" else "due"
    parent_short = short_uuid(parent.get("uuid"))
    lifecycle_plan = None
    try:
        candidate = RecurrenceCandidate(
            child_due=child_due,
            metadata=tuple(sorted(dict(meta or {}).items())),
            dnf=None,
            until=until_dt,
        )
        planner_kwargs = {
            "generation": generation,
            "validated_configuration": {"scheduler_fingerprint": "reconcile"},
            "compare_datetimes": compare_datetimes,
            "preflight": LifecyclePreflight.from_context(
                base_link=link,
                next_link=next_link,
                kind=kind,
                chain_id=parent.get("chainID"),
            ),
            "carry_validator": lambda snapshot, candidate_child, _candidate: invalid_relative_carry_reason(
                snapshot.observation,
                dict(candidate_child),
                child_field=child_field,
                generation=generation,
            ),
        }
        lifecycle_plan = (
            plan_expiration_successor(
                TaskSnapshot.from_observation(observation),
                **planner_kwargs,
            )
            if is_expiration
            else plan_candidate_successor(
                TaskSnapshot.from_observation(observation),
                LifecycleEvent.COMPLETE,
                candidate,
                **planner_kwargs,
            )
        )
        if lifecycle_plan.action is LifecycleAction.FINALIZE_CHAIN:
            return LifecycleRecoveryDecision(
                "legitimate_final",
                decision_parent,
                next_link,
                "reached lifecycle successor limit",
                child_due=child_due,
            )
        child = lifecycle_plan.child_dict()
    except Exception as exc:
        underlying = exc
        while getattr(underlying, "__cause__", None) is not None:
            underlying = underlying.__cause__
        carry_conflict = (
            isinstance(underlying, native_until.NativeUntilCarryError)
            and underlying.code == native_until.CARRY_CONFLICT
        )
        if is_expiration and kind in {"anchor", "anchor_file"} and carry_conflict:
            try:
                child = _build_expiration_child_with_day_end(
                    parent,
                    child_due=child_due,
                    child_field=child_field,
                    next_link=next_link,
                    parent_short=parent_short,
                    kind=kind,
                    cpmax=cpmax,
                    until_dt=until_dt,
                    hook=hook,
                    generation=generation,
                )
            except Exception as fallback_exc:
                return LifecycleRecoveryDecision("error", decision_parent, next_link, f"failed to build child: {scheduling_error_message(fallback_exc)}", child_due=child_due)
        else:
            return LifecycleRecoveryDecision("error", decision_parent, next_link, f"failed to build child: {scheduling_error_message(exc)}", child_due=child_due)
    if lifecycle_plan is None:
        try:
            guard = ParentGuard(
                status=str(parent.get("status") or "pending"),
                chain=str(parent.get("chain") or "on"),
                chain_id=str(parent.get("chainID") or ""),
                link=int(next_link - 1),
                recurrence_fingerprint=recurrence_fingerprint(parent),
                modified=str(parent.get("modified") or ""),
            )
            identity = LifecycleIdentity(
                chain_id=guard.chain_id,
                parent_uuid=str(parent.get("uuid") or ""),
                source_link=guard.link,
                target_link=next_link,
                event=LifecycleEvent.EXPIRE if is_expiration else LifecycleEvent.COMPLETE,
            )
            lifecycle_plan = LifecyclePlan.from_draft(
                identity=identity,
                action=LifecycleAction.SPAWN_CHILD,
                parent_guard=guard,
                draft=_child_draft(child),
                parent_patch={},
                expected_postconditions=("child_present", "parent_linked", "verified"),
            )
        except Exception as exc:
            return LifecycleRecoveryDecision(
                "error",
                decision_parent,
                next_link,
                f"failed to build lifecycle plan: {scheduling_error_message(exc)}",
                child_due=child_due,
            )
    reason = "expired link missing next link" if is_expiration else "missing next link"
    return LifecycleRecoveryDecision(
        "spawn",
        decision_parent,
        next_link,
        reason,
        child=child,
        child_due=child_due,
        lifecycle_plan=lifecycle_plan,
    )


def plan_recovery_decision(
    parent: TaskObservation,
    *,
    existing_children: list[TaskObservation] | tuple[TaskObservation, ...],
    hook: Any,
    generation: ChainGenerationService | None = None,
) -> LifecycleRecoveryDecision:
    """Build one plan inside the parent task's business-calendar context."""
    if not isinstance(parent, TaskObservation):
        raise TypeError("recovery planning requires a TaskObservation parent")
    parent_values = parent.to_mapping()
    generation = generation or _generation_service(hook)
    core = generation.core
    use_task_calendar = getattr(core, "use_task_business_calendar", None)
    if not callable(use_task_calendar):
        return _plan_recovery_decision_unscoped(
            parent_values,
            existing_children=existing_children,
            hook=hook,
            generation=generation,
        )

    next_link = int_or_default(parent_values.get("link"), 1) + 1
    try:
        calendar_context = use_task_calendar(parent_values)
    except Exception as exc:
        return LifecycleRecoveryDecision(
            "error",
            parent,
            next_link,
            f"invalid business calendar: {exc}",
        )
    with calendar_context:
        return _plan_recovery_decision_unscoped(
            parent_values,
            existing_children=existing_children,
            hook=hook,
            generation=generation,
        )
