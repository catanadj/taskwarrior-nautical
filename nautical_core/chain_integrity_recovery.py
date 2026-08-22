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


@dataclass(frozen=True, slots=True)
class NativeUntilRepairCandidate:
    row: dict[str, Any]
    previous: dict[str, Any] | None
    item: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RecoveryAudit:
    native_until: NativeUntilAudit
    candidates: tuple[NativeUntilRepairCandidate, ...] = ()


class IntegrityRecoveryService:
    """Build typed recovery evidence from one authoritative chain snapshot."""

    @staticmethod
    def candidate_sort_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
        return (
            str(row.get("chainID") or "").strip().casefold(),
            lifecycle.int_or_default(row.get("link"), 0),
            str(row.get("status") or "").strip().casefold(),
            str(row.get("uuid") or "").strip().casefold(),
        )

    @staticmethod
    def ambiguous_candidate_slots(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, int], str]:
        grouped: dict[tuple[str, int], set[str]] = {}
        for row in rows:
            chain_id = str(row.get("chainID") or "").strip()
            link = lifecycle.int_or_default(row.get("link"), 0)
            uuid = str(row.get("uuid") or "").strip().lower()
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


__all__ = ("IntegrityRecoveryService", "NativeUntilRepairCandidate", "RecoveryAudit")
