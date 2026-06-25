from __future__ import annotations

from dataclasses import dataclass
from typing import Any


RECURRENCE_FIELDS = ("anchor", "anchor_file", "cp")


@dataclass(frozen=True)
class LinkRepair:
    uuid: str
    short: str
    chain_id: str
    link: int
    field: str
    old: str
    new: str


@dataclass(frozen=True)
class ChainIssue:
    kind: str
    chain_id: str
    message: str
    tasks: list[dict[str, Any]]


def short_uuid(value: object) -> str:
    raw = str(value or "").strip()
    return raw[:8] if raw else ""


def int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (str, bytes, int, float)):
        return None
    try:
        return int(value)
    except Exception:
        return None


def is_nautical_chain_task(task: dict[str, Any]) -> bool:
    if str(task.get("status") or "").strip() == "deleted":
        return False
    if str(task.get("chainID") or "").strip():
        return True
    return any(str(task.get(field) or "").strip() for field in RECURRENCE_FIELDS)


def task_summary(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "uuid": str(task.get("uuid") or ""),
        "short": short_uuid(task.get("uuid")),
        "status": str(task.get("status") or ""),
        "chainID": str(task.get("chainID") or ""),
        "link": task.get("link"),
        "prevLink": str(task.get("prevLink") or ""),
        "nextLink": str(task.get("nextLink") or ""),
        "description": str(task.get("description") or ""),
    }


def group_by_chain_id(tasks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        if not is_nautical_chain_task(task):
            continue
        chain_id = str(task.get("chainID") or "").strip()
        if not chain_id:
            continue
        groups.setdefault(chain_id, []).append(task)
    return groups


def plan_chain_link_repairs(tasks: list[dict[str, Any]]) -> tuple[list[LinkRepair], list[ChainIssue]]:
    repairs: list[LinkRepair] = []
    issues: list[ChainIssue] = []

    for chain_id, members in sorted(group_by_chain_id(tasks).items()):
        slots: dict[int, list[dict[str, Any]]] = {}
        missing_link: list[dict[str, Any]] = []
        for task in members:
            link = int_or_none(task.get("link"))
            if link is None:
                missing_link.append(task)
                continue
            slots.setdefault(link, []).append(task)

        if missing_link:
            issues.append(
                ChainIssue(
                    "missing_link",
                    chain_id,
                    f"{len(missing_link)} task(s) are missing a numeric link",
                    [task_summary(task) for task in missing_link],
                )
            )

        duplicates = {link: rows for link, rows in slots.items() if len(rows) > 1}
        for link, rows in sorted(duplicates.items()):
            issues.append(
                ChainIssue(
                    "duplicate_slot",
                    chain_id,
                    f"link {link} has {len(rows)} tasks; skipped",
                    [task_summary(task) for task in rows],
                )
            )

        unique_slots = {link: rows[0] for link, rows in slots.items() if len(rows) == 1}
        for link in sorted(unique_slots):
            current = unique_slots[link]
            nxt = unique_slots.get(link + 1)
            if nxt is None:
                continue
            current_uuid = str(current.get("uuid") or "").strip()
            next_uuid = str(nxt.get("uuid") or "").strip()
            current_short = short_uuid(current_uuid)
            next_short = short_uuid(next_uuid)
            if not current_uuid or not next_uuid or not current_short or not next_short:
                continue

            current_next = str(current.get("nextLink") or "").strip()
            if current_next != next_short:
                repairs.append(
                    LinkRepair(current_uuid, current_short, chain_id, link, "nextLink", current_next, next_short)
                )

            next_prev = str(nxt.get("prevLink") or "").strip()
            if next_prev != current_short:
                repairs.append(
                    LinkRepair(next_uuid, next_short, chain_id, link + 1, "prevLink", next_prev, current_short)
                )

    return repairs, issues
