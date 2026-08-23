from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

from nautical_core.modify_models import (
    CapFromUntilAnchorCallback,
    CapFromUntilCpCallback,
    CoerceIntCallback,
    ComputeAnchorChildDueCallback,
    ComputeCpChildDueCallback,
    CompletionComputeResult,
    CompletionLifecycleDiagnostic,
    CompletionLifecycleResult,
    CompletionComputeServices,
    DatetimeParserCallback,
    EndChainSummaryCallback,
    EstimateAnchorFinalCallback,
    EstimateCpFinalCallback,
    PanelCallback,
    PrintTaskCallback,
    SafeParseDatetimeCallback,
    DiagnosticCallback,
    ValidateChainDurationCallback,
    ValidateUntilCallback,
)
from nautical_core.scheduler_models import (
    OccurrenceSearchExhausted,
    occurrence_exhaustion_message,
)
from nautical_core.timeutil import compare_datetimes
from nautical_core.lifecycle_models import LifecycleEvent
from nautical_core.modify_lifecycle import apply_terminal_transition
from nautical_core.task_codec import DEFAULT_TASK_CODEC
from nautical_core.task_models import NauticalTask


def _terminal_diagnostic(new: dict[str, Any], next_no: int, failure_kind: str) -> CompletionLifecycleDiagnostic:
    try:
        raw_link = new.get("link")
        parent_link = int(raw_link) if isinstance(raw_link, (str, int, float)) else None
    except (TypeError, ValueError):
        parent_link = None
    chain_id = str(new.get("chainID") or "").strip()
    return CompletionLifecycleDiagnostic(
        transition_id=f"{chain_id}:{parent_link}->{next_no}",
        chain_id=chain_id,
        parent_link=parent_link,
        child_link=next_no,
        stage="compute",
        failure_kind=failure_kind,
    )


def completion_compute_child_due(
    new: dict[str, Any],
    kind: str,
    *,
    compute_anchor_child_due: ComputeAnchorChildDueCallback,
    compute_cp_child_due: ComputeCpChildDueCallback,
    panel: PanelCallback,
    print_task: PrintTaskCallback,
    diag: DiagnosticCallback | None = None,
    on_terminal: Any | None = None,
) -> tuple[Any, Any, Any] | None:
    try:
        if kind in {"anchor", "anchor_file"}:
            child_due, meta, dnf = compute_anchor_child_due(new)
            if kind == "anchor_file" and dnf is None:
                dnf = []
        else:
            child_due, meta = compute_cp_child_due(new)
            dnf = None
        return child_due, meta, dnf
    except OccurrenceSearchExhausted as exc:
        if callable(on_terminal):
            on_terminal(exc)
        else:
            panel(
                "⛔ Chain error",
                [("Scheduler", occurrence_exhaustion_message(exc))],
                kind="error",
            )
            print_task(new)
        return None
    except ValueError as exc:
        panel(
            "⛔ Chain error",
            [("Reason", f"Invalid task field: {str(exc)}")],
            kind="error",
        )
        print_task(new)
        return None
    except Exception as exc:
        if callable(diag):
            diag(f"compute next due failed: {exc}")
        panel(
            "⛔ Chain error",
            [("Reason", "Could not compute next recurrence timestamp")],
            kind="error",
        )
        print_task(new)
        return None


def completion_until_or_fail(
    new: dict[str, Any],
    now_utc: Any,
    *,
    safe_parse_datetime: SafeParseDatetimeCallback,
    validate_until_not_past: ValidateUntilCallback,
    panel: PanelCallback,
    print_task: PrintTaskCallback,
) -> datetime | None | bool:
    until_dt, err = safe_parse_datetime(new.get("chainUntil"))
    if err:
        panel("⛔ Chain error", [("Reason", f"Invalid chainUntil: {err}")], kind="error")
        print_task(new)
        return False

    if until_dt:
        is_valid, err_msg = validate_until_not_past(until_dt, now_utc)
        if not is_valid:
            panel(
                "⛔ Chain error",
                [("Reason", f"Invalid chainUntil: {err_msg}")],
                kind="error",
            )
            print_task(new)
            return False
    return until_dt


def completion_until_guard_or_stop(
    new: dict[str, Any],
    child_due: Any,
    until_dt: Any,
    now_utc: Any,
    *,
    end_chain_summary: EndChainSummaryCallback,
    print_task: PrintTaskCallback,
) -> bool:
    if until_dt and compare_datetimes(child_due, until_dt) > 0:
        end_chain_summary(new, "Reached 'until' limit", now_utc)
        apply_terminal_transition(new, LifecycleEvent.CHAIN_UNTIL)
        print_task(new)
        return False
    return True


def completion_require_child_due_or_fail(
    new: dict[str, Any],
    child_due: Any,
    *,
    panel: PanelCallback,
    print_task: PrintTaskCallback,
) -> bool:
    if child_due:
        return True
    panel(
        "⛔ Chain error",
        [("Reason", "Could not compute next recurrence timestamp (no end date on parent)")],
        kind="error",
    )
    print_task(new)
    return False


def completion_warn_unreasonable_duration(
    new: dict[str, Any],
    child_due: Any,
    until_dt: Any,
    now_utc: Any,
    *,
    validate_chain_duration_reasonable: ValidateChainDurationCallback,
    panel: PanelCallback,
) -> None:
    if not until_dt:
        return
    is_reasonable, warn_msg = validate_chain_duration_reasonable(child_due, until_dt, now_utc)
    if warn_msg and not is_reasonable:
        panel("⚠ Chain duration warning", [("Warning", warn_msg)], kind="warning")


def completion_caps(
    kind: str,
    new: dict[str, Any],
    child_due: Any,
    dnf: Any,
    *,
    coerce_int: CoerceIntCallback,
    dtparse: DatetimeParserCallback,
    estimate_cp_final_by_max: EstimateCpFinalCallback,
    estimate_anchor_final_by_max: EstimateAnchorFinalCallback,
    cap_from_until_cp: CapFromUntilCpCallback,
    cap_from_until_anchor: CapFromUntilAnchorCallback,
) -> tuple[int, datetime | None, int | None, list[tuple[str, Any]], int | None]:
    cpmax = coerce_int(new.get("chainMax"), 0)
    until_dt = dtparse(new.get("chainUntil"))
    cap_no = cpmax if cpmax else None
    finals = []

    if kind == "cp" and cpmax:
        try:
            fmax = estimate_cp_final_by_max(new, child_due)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass
    if kind in {"anchor", "anchor_file"} and cpmax:
        try:
            fmax = estimate_anchor_final_by_max(new, child_due, dnf)
            if fmax:
                finals.append(("max", fmax))
        except Exception:
            pass

    until_cap_no = None
    if until_dt:
        if kind == "cp":
            u_no, u_dt = cap_from_until_cp(new, child_due)
        else:
            u_no, u_dt = cap_from_until_anchor(new, child_due, dnf)
        if u_no:
            until_cap_no = u_no
            cap_no = min(cap_no, u_no) if cap_no else u_no
        if u_dt:
            finals.append(("until", u_dt))
    return cpmax, until_dt, cap_no, finals, until_cap_no


def cap_from_until_cp(
    task: dict[str, Any],
    next_due_utc: Any,
    *,
    parse_datetime: DatetimeParserCallback,
    parse_cp_sequence_tokens: Callable[[str], Any],
    coerce_int: CoerceIntCallback,
    sequence_period_for_link: Callable[[Any, str, int, str], Any],
    add_period: Callable[[Any, Any], Any],
    max_iterations: int,
) -> tuple[int | None, Any]:
    """Return the final CP link and due date permitted by chainUntil."""
    until = parse_datetime(task.get("chainUntil"))
    if not until:
        return None, None
    cp_str = task.get("cp") or ""
    tokens = parse_cp_sequence_tokens(cp_str)
    if not tokens:
        return None, None
    current_link = coerce_int(task.get("link"), 1)
    next_link = current_link + 1
    due = next_due_utc
    last_link = None
    last_due = None
    iterations = 0
    while due and compare_datetimes(due, until) <= 0 and iterations < max_iterations:
        iterations += 1
        last_link, last_due = next_link, due
        period = sequence_period_for_link(
            tokens,
            cp_str,
            next_link,
            str(task.get("chainID") or "").strip(),
        )
        due = add_period(due, period)
        next_link += 1
    return last_link, last_due


def cap_from_until_anchor(
    task: dict[str, Any],
    next_due_utc: Any,
    dnf: Any,
    *,
    parse_datetime: Any,
    coerce_int: Any,
    recurrence_seed_base: Any,
    to_local_cached: Any,
    safe_parse_datetime: Any,
    anchor_file_fallback_hhmm: Any,
    omit_dnf_from_parent: Any,
    recurrence_evaluator_for_task: Any,
    anchor_file_provider_for: Any,
    anchor_included_occurrences: Any,
    compare_datetimes: Any,
    max_iterations: int,
) -> tuple[int | None, Any]:
    """Return the final anchor link and due date permitted by ``chainUntil``."""
    until_utc = parse_datetime(task.get("chainUntil"))
    if not until_utc:
        return None, None

    current_link = coerce_int(task.get("link"), 1)
    seed_base = recurrence_seed_base(task)
    next_local = to_local_cached(next_due_utc)
    until_local = to_local_cached(until_utc)
    due0, _ = safe_parse_datetime(task.get("due"))
    default_seed = to_local_cached(due0 or next_due_utc).date()
    fallback_hhmm = anchor_file_fallback_hhmm(task, next_local)
    _omit_expr, omit_dnf = omit_dnf_from_parent(task)
    scheduler = recurrence_evaluator_for_task(task)._default_next_occurrence_after_local_dt
    anchor_file = (task.get("anchor_file") or "").strip()
    anchor_file_provider = None
    if anchor_file:
        anchor_file_provider = anchor_file_provider_for(
            anchor_file,
            fallback_hhmm=fallback_hhmm,
            seed_base=seed_base,
        )

    count = 0
    last_hit = None
    cursor = next_local
    iterations = 0

    while iterations < max_iterations and compare_datetimes(cursor, until_local) <= 0:
        iterations += 1
        count += 1
        last_hit = cursor
        if anchor_file:
            future = anchor_included_occurrences(
                task,
                after_local_dt=cursor,
                inclusive=False,
                limit=2,
                fallback_hhmm=fallback_hhmm,
                omit_dnf=omit_dnf,
                seed_base=seed_base,
                default_seed_date=default_seed,
                dnf=dnf,
                anchor_file_provider=anchor_file_provider,
            )
            cursor = future[0] if future else None
        else:
            cursor = scheduler(
                dnf,
                cursor,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                fallback_hhmm=fallback_hhmm,
            )
        if cursor is None:
            break

    if cursor is not None and compare_datetimes(cursor, until_local) <= 0 and iterations >= max_iterations:
        raise ValueError(
            f"Anchor chainUntil projection exceeded {max_iterations} occurrences; "
            "narrow chainUntil or use a larger recurrence interval."
        )
    if count == 0 or last_hit is None:
        return None, None
    return current_link + count, last_hit.astimezone(timezone.utc)


def estimate_cp_final_by_max(
    task: dict[str, Any],
    next_due_utc: Any,
    *,
    coerce_int: Any,
    parse_cp_sequence_tokens: Any,
    sequence_period_for_link: Any,
    add_period: Any,
    max_iterations: int,
    diagnostic: Any | None = None,
) -> Any:
    """Estimate the final CP due date permitted by ``chainMax``."""
    chain_max = coerce_int(task.get("chainMax"), 0)
    if not chain_max:
        return None
    current_link = coerce_int(task.get("link"), 1)
    if current_link >= chain_max:
        return None

    cp_str = task.get("cp") or ""
    tokens = parse_cp_sequence_tokens(cp_str)
    if not tokens:
        return None

    future_due = next_due_utc
    future_link = current_link + 1
    iterations = 0
    while future_link < chain_max:
        iterations += 1
        if iterations > max_iterations:
            if callable(diagnostic):
                diagnostic(
                    f"chainMax forecast stopped after {max_iterations} occurrences; "
                    "final date is unavailable"
                )
            return None
        period = sequence_period_for_link(
            tokens,
            cp_str,
            future_link,
            str(task.get("chainID") or "").strip(),
        )
        future_link += 1
        future_due = add_period(future_due, period)
    return future_due


def estimate_anchor_final_by_max(
    task: dict[str, Any],
    next_due_utc: Any,
    dnf: Any,
    *,
    coerce_int: Any,
    recurrence_seed_base: Any,
    to_local_cached: Any,
    safe_parse_datetime: Any,
    anchor_file_fallback_hhmm: Any,
    omit_dnf_from_parent: Any,
    recurrence_evaluator_for_task: Any,
    anchor_file_provider_for: Any,
    anchor_included_occurrences: Any,
    diagnostic: Any | None = None,
    max_iterations: int,
) -> Any:
    """Estimate the final anchor due date permitted by ``chainMax``."""
    chain_max = coerce_int(task.get("chainMax"), 0)
    if not chain_max:
        return None
    current_link = coerce_int(task.get("link"), 1)
    if current_link >= chain_max:
        return None

    seed_base = recurrence_seed_base(task)
    next_local = to_local_cached(next_due_utc)
    due0, _ = safe_parse_datetime(task.get("due"))
    default_seed = to_local_cached(due0 or next_due_utc).date()
    fallback_hhmm = anchor_file_fallback_hhmm(task, next_local)
    _omit_expr, omit_dnf = omit_dnf_from_parent(task)
    scheduler = recurrence_evaluator_for_task(task)._default_next_occurrence_after_local_dt
    anchor_file = (task.get("anchor_file") or "").strip()
    anchor_file_provider = None
    if anchor_file:
        anchor_file_provider = anchor_file_provider_for(
            anchor_file,
            fallback_hhmm=fallback_hhmm,
            seed_base=seed_base,
        )

    future_link = current_link + 1
    future_local = next_local
    iterations = 0
    while future_link < chain_max:
        iterations += 1
        if iterations > max_iterations:
            if callable(diagnostic):
                diagnostic(
                    f"chainMax forecast stopped after {max_iterations} occurrences; "
                    "final date is unavailable"
                )
            return None
        if anchor_file:
            future = anchor_included_occurrences(
                task,
                after_local_dt=future_local,
                inclusive=False,
                limit=2,
                fallback_hhmm=fallback_hhmm,
                omit_dnf=omit_dnf,
                seed_base=seed_base,
                default_seed_date=default_seed,
                dnf=dnf,
                anchor_file_provider=anchor_file_provider,
            )
            future_local = future[0] if future else None
        else:
            future_local = scheduler(
                dnf,
                future_local,
                default_seed_date=default_seed,
                seed_base=seed_base,
                omit_dnf=omit_dnf,
                fallback_hhmm=fallback_hhmm,
            )
        if future_local is None:
            return None
        future_link += 1
    return future_local.astimezone(timezone.utc)


def first_recurrence_target(
    task: dict[str, Any],
    source: str,
    *,
    parse_datetime: Any,
    format_datetime: Any,
    generation_service: Any,
) -> Any:
    """Compute the first projected target used by recurrence-update panels."""
    target_field = "due" if task.get("due") else "scheduled" if task.get("scheduled") else ""
    if not target_field:
        return None
    target = parse_datetime(task.get(target_field))
    if not target:
        return None
    parent = dict(task)
    parent["end"] = format_datetime(target)
    try:
        generation = generation_service()
        typed_parent = NauticalTask.from_observation(
            DEFAULT_TASK_CODEC.decode_row(parent, source_query="completion recurrence target")
        )
        if source in {"anchor", "anchor_file"}:
            result = generation.compute_anchor_child_due(typed_parent)
        else:
            result = generation.compute_cp_child_due(typed_parent)
        return result[0] if result else None
    except Exception:
        return None


def completion_cap_guard_or_stop(
    new: dict[str, Any],
    next_no: int,
    cap_no: int | None,
    now_utc: Any,
    *,
    end_chain_summary: EndChainSummaryCallback,
    print_task: PrintTaskCallback,
) -> bool:
    if cap_no and next_no > cap_no:
        end_chain_summary(new, f"Reached cap #{cap_no}", now_utc, current_task=new)
        apply_terminal_transition(new, LifecycleEvent.CHAIN_MAX)
        print_task(new)
        return False
    return True


def attach_lifecycle_plan(
    new: dict[str, Any],
    computed: CompletionComputeResult,
    next_no: int,
    now_utc: Any,
    *,
    preflight: Any | None,
    generation: Any,
    scheduler_fingerprint: str,
    compare_datetimes: Any,
    invalid_relative_carry_reason: Any,
    lifecycle_planner: Any,
    lifecycle_models: Any,
    modify_models: Any,
    end_chain_summary: EndChainSummaryCallback,
    ensure_terminal_chain_off: Any,
    panel: PanelCallback,
    print_task: PrintTaskCallback,
    diag: DiagnosticCallback,
) -> CompletionComputeResult | CompletionLifecycleResult:
    """Attach the shared lifecycle successor plan to a computed result."""
    try:
        candidate = lifecycle_planner.RecurrenceCandidate(
            child_due=computed.child_due,
            metadata=tuple(sorted(dict(computed.meta or {}).items())),
            dnf=computed.dnf,
            until=computed.until_dt,
        )
        plan = lifecycle_planner.plan_candidate_successor(
            lifecycle_models.TaskSnapshot.from_observation(
                DEFAULT_TASK_CODEC.decode_row(new, source_query="modify completion")
            ),
            lifecycle_models.LifecycleEvent.COMPLETE,
            candidate,
            generation=generation,
            validated_configuration={"scheduler_fingerprint": scheduler_fingerprint},
            compare_datetimes=compare_datetimes,
            preflight=(
                lifecycle_planner.LifecyclePreflight.from_context(
                    base_link=preflight.base_no,
                    next_link=preflight.next_no,
                    kind=preflight.kind,
                    chain_id=preflight.chain_id,
                )
                if preflight is not None
                else None
            ),
            carry_validator=lambda snapshot, candidate_child, _candidate: invalid_relative_carry_reason(
                snapshot.to_dict(),
                dict(candidate_child),
                child_field=str(computed.meta.get("target_field") or "due"),
                generation=generation,
            ),
        )
        if plan.action is lifecycle_models.LifecycleAction.FINALIZE_CHAIN:
            end_chain_summary(new, "Reached lifecycle successor limit", now_utc)
            ensure_terminal_chain_off(new, "complete")
            print_task(new)
            return modify_models.CompletionLifecycleResult(
                state="terminal",
                reason="successor limit reached",
                diagnostic=modify_models.CompletionLifecycleDiagnostic(
                    transition_id=f"{str(new.get('chainID') or '').strip()}:{new.get('link')}->{next_no}",
                    chain_id=str(new.get("chainID") or "").strip(),
                    parent_link=int(str(new.get("link"))) if str(new.get("link") or "").isdigit() else None,
                    child_link=next_no,
                    stage="plan",
                    failure_kind="successor_limit",
                ),
            )
        computed.lifecycle_plan = plan
        computed.planned_child = plan.child_dict()
    except Exception as exc:
        diag(f"lifecycle planner failed: {type(exc).__name__}: {exc}")
        panel("⛓ Chain error", [("Reason", str(exc) or "Could not construct a lifecycle successor plan")], kind="error")
        print_task(new)
        return modify_models.CompletionLifecycleResult(
            state="retryable",
            reason=str(exc).strip() or "Could not construct a lifecycle successor plan",
            diagnostic=modify_models.CompletionLifecycleDiagnostic(
                transition_id=f"{str(new.get('chainID') or '').strip()}:{new.get('link')}->{next_no}",
                chain_id=str(new.get("chainID") or "").strip(),
                parent_link=int(str(new.get("link"))) if str(new.get("link") or "").isdigit() else None,
                child_link=next_no,
                stage="plan",
                failure_kind="planner_error",
            ),
        )
    return computed


def completion_compute_next_and_limits(
    new: dict[str, Any],
    kind: str,
    next_no: int,
    now_utc: Any,
    *,
    services: CompletionComputeServices,
) -> CompletionComputeResult | CompletionLifecycleResult | None:
    completion_compute_child_due = services.completion_compute_child_due
    completion_until_or_fail = services.completion_until_or_fail
    completion_until_guard_or_stop = services.completion_until_guard_or_stop
    completion_require_child_due_or_fail = services.completion_require_child_due_or_fail
    completion_warn_unreasonable_duration = services.completion_warn_unreasonable_duration
    completion_caps = services.completion_caps
    completion_cap_guard_or_stop = services.completion_cap_guard_or_stop
    computed = completion_compute_child_due(new, kind)
    if computed is None:
        if str(new.get("chain") or "").strip().lower() == "off":
            return CompletionLifecycleResult(
                state="terminal",
                reason="recurrence scheduler reached a terminal boundary",
                diagnostic=_terminal_diagnostic(new, next_no, "scheduler_exhausted"),
            )
        return CompletionLifecycleResult(
            state="retryable",
            reason="could not compute next recurrence timestamp",
            diagnostic=_terminal_diagnostic(new, next_no, "scheduler_error"),
        )
    child_due, meta, dnf = computed

    until_dt = completion_until_or_fail(new, now_utc)
    if until_dt is False:
        return CompletionLifecycleResult(
            state="retryable",
            reason="chainUntil validation failed",
            diagnostic=_terminal_diagnostic(new, next_no, "chain_until_validation"),
        )

    if not completion_until_guard_or_stop(new, child_due, until_dt, now_utc):
        if str(new.get("chain") or "").strip().lower() == "off":
            return CompletionLifecycleResult(
                state="terminal",
                reason="chainUntil boundary reached",
                diagnostic=_terminal_diagnostic(new, next_no, "chain_until"),
            )
        return CompletionLifecycleResult(
            state="retryable",
            reason="chainUntil boundary prevented successor creation",
            diagnostic=_terminal_diagnostic(new, next_no, "chain_until_guard"),
        )

    if not completion_require_child_due_or_fail(new, child_due):
        return CompletionLifecycleResult(
            state="retryable",
            reason="child recurrence timestamp is unavailable",
            diagnostic=_terminal_diagnostic(new, next_no, "missing_child_due"),
        )

    completion_warn_unreasonable_duration(new, child_due, until_dt, now_utc)
    cpmax, until_dt, cap_no, finals, until_cap_no = completion_caps(kind, new, child_due, dnf)

    if not completion_cap_guard_or_stop(new, next_no, cap_no, now_utc):
        if str(new.get("chain") or "").strip().lower() == "off":
            return CompletionLifecycleResult(
                state="terminal",
                reason="successor limit reached",
                diagnostic=_terminal_diagnostic(new, next_no, "successor_limit"),
            )
        return CompletionLifecycleResult(
            state="retryable",
            reason="successor limit prevented child creation",
            diagnostic=_terminal_diagnostic(new, next_no, "successor_limit_guard"),
        )

    return CompletionComputeResult(
        child_due=child_due,
        meta=meta,
        dnf=dnf,
        until_dt=until_dt,
        cpmax=cpmax,
        cap_no=cap_no,
        finals=finals,
        until_cap_no=until_cap_no,
    )
