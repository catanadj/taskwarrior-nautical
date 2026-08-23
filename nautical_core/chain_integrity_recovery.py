"""Typed recovery evidence for the chain-integrity engine.

This module deliberately contains no Taskwarrior or UI code.  It turns a
snapshot and predecessor lookup into native-until repair candidates; mutation
and presentation are performed by the outer application services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable

from . import chain_integrity_lifecycle as lifecycle
from .native_until_integrity import NativeUntilAudit, audit_result
from .integration_models import (
    GuardTimestamp,
    GuardTimestampField,
    MutationGuard,
    MutationOperation,
    MutationRequest,
    NativeUntilRepairPayload,
)
from .lifecycle_models import recurrence_fingerprint
from .task_models import FieldPresence, TaskObservation


def _observation_value(row: TaskObservation, name: str) -> object:
    state = row.field(name)
    if state.presence is FieldPresence.ABSENT:
        return None
    return getattr(state.value, "value", state.value)


@dataclass(frozen=True, slots=True)
class NativeUntilRepairCandidate:
    row: TaskObservation
    previous: TaskObservation | None
    item: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecoveryAudit:
    native_until: NativeUntilAudit
    candidates: tuple[NativeUntilRepairCandidate, ...] = ()


class IntegrityRecoveryService:
    """Build typed recovery evidence from one authoritative chain snapshot."""

    def __init__(self, *, child_lookup: Callable[[str, int], dict[str, Any] | None] | None = None) -> None:
        self._child_lookup = child_lookup

    def existing_children(self, parent: dict[str, Any]) -> list[dict[str, Any]]:
        """Resolve the single successor slot without owning repository I/O."""
        chain_id = str(parent.get("chainID") or "").strip()
        next_link = lifecycle.int_or_default(parent.get("link"), 1) + 1
        if not chain_id or self._child_lookup is None:
            return []
        value = self._child_lookup(chain_id, next_link)
        return [dict(value)] if value is not None else []

    @staticmethod
    def native_until_request(
        row: dict[str, Any],
        new_until: str,
        *,
        mutation_epoch: int,
    ) -> MutationRequest:
        """Build the guarded native-until mutation without Taskwarrior I/O."""
        uuid = str(row.get("uuid") or "").strip()
        chain_id = str(row.get("chainID") or "").strip()
        link = lifecycle.int_or_default(row.get("link"), 0)
        modified = str(row.get("modified") or "").strip()
        expected_until = str(row.get("until") or "").strip()
        if not uuid or not chain_id or link <= 0 or not modified or not expected_until:
            raise ValueError("native until repair lacks task identity")
        guard = MutationGuard(
            task_uuid=uuid,
            status=str(row.get("status") or ""),
            chain_id=chain_id,
            link=link,
            recurrence_identity=recurrence_fingerprint(row),
            timestamps=(GuardTimestamp(GuardTimestampField.MODIFIED, modified),),
            expected_mutation_epoch=mutation_epoch,
            chain=str(row.get("chain") or "on"),
        )
        return MutationRequest(
            MutationOperation.NATIVE_UNTIL_REPAIR,
            guard,
            NativeUntilRepairPayload(uuid, expected_until, str(new_until)),
        )

    @staticmethod
    def candidate_sort_key(row: TaskObservation) -> tuple[str, int, str, str]:
        return (
            str(_observation_value(row, "chainID") or "").strip().casefold(),
            lifecycle.int_or_default(_observation_value(row, "link"), 0),
            str(_observation_value(row, "status") or "").strip().casefold(),
            str(_observation_value(row, "uuid") or "").strip().casefold(),
        )

    @staticmethod
    def ambiguous_candidate_slots(rows: Iterable[TaskObservation]) -> dict[tuple[str, int], str]:
        grouped: dict[tuple[str, int], set[str]] = {}
        for row in rows:
            chain_id = str(_observation_value(row, "chainID") or "").strip()
            link = lifecycle.int_or_default(_observation_value(row, "link"), 0)
            uuid = str(_observation_value(row, "uuid") or "").strip().lower()
            if chain_id and link > 0 and uuid:
                grouped.setdefault((chain_id, link), set()).add(uuid)
        return {
            slot: (
                f"ambiguous candidate slot chain {slot[0]} link {slot[1]} "
                f"has {len(uuids)} distinct parent tasks"
            )
            for slot, uuids in grouped.items()
            if len(uuids) > 1
        }

    def audit_native_until(
        self,
        rows: Iterable[dict[str, Any]],
        *,
        predecessor: Callable[[dict[str, Any]], dict[str, Any] | None],
        safe_parse_datetime: Callable[[Any], tuple[Any, str | None]],
        fmt_isoz: Callable[[Any], str],
        utc_to_local_naive: Callable[[Any], Any],
        local_naive_to_utc: Callable[[Any], Any],
    ) -> RecoveryAudit:
        materialized = tuple(dict(row) for row in rows)
        by_chain_link = {
            (str(row.get("chainID") or "").strip(), lifecycle.int_or_default(row.get("link"), 0)): row
            for row in materialized
        }
        repairs: list[dict[str, Any]] = []
        errors: list[str] = []
        candidates: list[NativeUntilRepairCandidate] = []
        for row in materialized:
            reason = lifecycle.invalid_native_until_reason(row, safe_parse_datetime=safe_parse_datetime)
            if not reason:
                continue
            chain_id = str(row.get("chainID") or "").strip()
            link = lifecycle.int_or_default(row.get("link"), 0)
            previous = by_chain_link.get((chain_id, link - 1)) or predecessor(row)
            item: dict[str, Any] = {
                "task": lifecycle.short_uuid(row.get("uuid")),
                "chainID": chain_id,
                "link": link,
                "target": row.get("due") or row.get("scheduled"),
                "until": row.get("until"),
                "reason": reason,
            }
            repaired: str | None = None
            repair_error: str | None = None
            if previous is None:
                repair_error = "previous link is unavailable"
            else:
                previous_reason = lifecycle.invalid_native_until_reason(
                    previous, safe_parse_datetime=safe_parse_datetime,
                )
                if previous_reason:
                    repair_error = f"previous link is invalid: {previous_reason}"
                else:
                    repaired, repair_error = lifecycle.repair_native_until_from_previous(
                        previous,
                        row,
                        kind=lifecycle.recurrence_kind(row),
                        safe_parse_datetime=safe_parse_datetime,
                        fmt_isoz=fmt_isoz,
                        utc_to_local_naive=utc_to_local_naive,
                        local_naive_to_utc=local_naive_to_utc,
                    )
            if repair_error or not repaired:
                fallback, fallback_error = lifecycle.fallback_native_until_at_day_end(
                    row,
                    safe_parse_datetime=safe_parse_datetime,
                    fmt_isoz=fmt_isoz,
                    utc_to_local_naive=utc_to_local_naive,
                    local_naive_to_utc=local_naive_to_utc,
                )
                if fallback_error or not fallback:
                    item["action"] = "manual_review"
                    item["repair_error"] = fallback_error or repair_error or "could not calculate repaired until"
                    repairs.append(item)
                    continue
                repaired = fallback
                item["fallback"] = "local 23:00"
                item["reason"] = repair_error or item["reason"]
            item["action"] = "repair_until"
            item["new_until"] = repaired
            repairs.append(item)
            candidates.append(NativeUntilRepairCandidate(dict(row), dict(previous) if previous else None, item))
        for item in repairs:
            if item.get("action") == "repair_error":
                errors.append(
                    f"{item.get('task', '')} chain {item.get('chainID', '')} link {item.get('link', '')}: "
                    f"{item.get('repair_error', 'native-until repair failed')}"
                )
        return RecoveryAudit(audit_result(repairs, errors), tuple(candidates))

    def apply_native_until_candidate(
        self,
        row: dict[str, Any],
        previous: dict[str, Any] | None,
        item: dict[str, Any],
        *,
        repaired: str,
        taskdata: Any,
        lease_held: bool,
        mutation_lock: Callable[[Any, bool], Any],
        parent_lock: Callable[[str], Any],
        refresh_parent: Callable[[dict[str, Any]], dict[str, Any] | None],
        refresh_previous: Callable[[dict[str, Any]], dict[str, Any] | None],
        guard_error: Callable[[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None], str | None],
        configuration: Callable[[], tuple[str, str]],
        mutate: Callable[[dict[str, Any], str], None],
        verify: Callable[[dict[str, Any] | None, str], bool],
        on_lock_busy: Callable[[str], None],
    ) -> str | None:
        """Apply one candidate with ordered locks and fail-closed guards."""
        if taskdata is None:
            item["action"] = "repair_error"
            item["repair_error"] = "Taskwarrior data location is unavailable for native-until locking"
            return item["repair_error"]
        with mutation_lock(taskdata, lease_held) as reconcile_acquired:
            if not reconcile_acquired:
                on_lock_busy("reconcile")
                item["action"] = "repair_error"
                item["repair_error"] = "another reconcile apply is already running"
            else:
                with parent_lock(str(row.get("uuid") or "")) as acquired:
                    if not acquired:
                        on_lock_busy("parent")
                        item["action"] = "repair_error"
                        item["repair_error"] = "native-until repair lock busy"
                    else:
                        fresh = refresh_parent(row)
                        fresh_previous = refresh_previous(fresh or row)
                        reason = guard_error(row, fresh, fresh_previous)
                        if reason:
                            item["action"] = "repair_error"
                            item["repair_error"] = reason
                        else:
                            status, detail = configuration()
                            if status != "valid":
                                item["action"] = "manual_review"
                                item["repair_error"] = detail
                                item["configuration_drift"] = True
                                item["configuration_status"] = status
                                return None
                            try:
                                if fresh is None:
                                    raise RuntimeError("native-until target disappeared")
                                mutate(fresh, repaired)
                                verified = refresh_parent(fresh)
                                if not verify(verified, repaired):
                                    actual = str((verified or {}).get("until") or "<missing>")
                                    item["action"] = "repair_error"
                                    item["repair_error"] = (
                                        f"native until repair verification failed (expected {repaired}; found {actual})"
                                    )
                                else:
                                    item["applied"] = True
                            except Exception as exc:
                                item["action"] = "repair_error"
                                item["repair_error"] = str(exc).strip() or type(exc).__name__
        return item.get("repair_error") if item.get("action") == "repair_error" else None


__all__ = ("IntegrityRecoveryService", "NativeUntilRepairCandidate", "RecoveryAudit")
