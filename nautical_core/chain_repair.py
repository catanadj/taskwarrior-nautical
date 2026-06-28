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


def task_summary(task: dict[str, Any], *, reason: str = "") -> dict[str, Any]:
    summary = {
        "uuid": str(task.get("uuid") or ""),
        "short": short_uuid(task.get("uuid")),
        "status": str(task.get("status") or ""),
        "chainID": str(task.get("chainID") or ""),
        "link": task.get("link"),
        "prevLink": str(task.get("prevLink") or ""),
        "nextLink": str(task.get("nextLink") or ""),
        "description": str(task.get("description") or ""),
    }
    if reason:
        summary["reason"] = reason
    return summary


def _short_index(tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any] | None]:
    index: dict[str, dict[str, Any] | None] = {}
    for task in tasks:
        short = short_uuid(task.get("uuid"))
        if not short:
            continue
        if short in index:
            index[short] = None
        else:
            index[short] = task
    return index


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


def _link_evidence(short: str, by_short: dict[str, dict[str, Any] | None], field: str) -> tuple[int | None, str]:
    if short not in by_short:
        return None, f"{field} target {short} was not found in this chain"
    task = by_short.get(short)
    if task is None:
        return None, f"{field} target {short} is ambiguous"
    link = int_or_none(task.get("link"))
    if link is None:
        return None, f"{field} target {short} has no numeric link"
    return link, ""


def _infer_missing_link(task: dict[str, Any], by_short: dict[str, dict[str, Any] | None]) -> tuple[int | None, str]:
    candidates: list[int] = []
    skipped: list[str] = []

    prev_short = str(task.get("prevLink") or "").strip()
    if prev_short:
        prev_link, reason = _link_evidence(prev_short, by_short, "prevLink")
        if prev_link is not None:
            candidates.append(prev_link + 1)
        elif reason:
            skipped.append(reason)

    next_short = str(task.get("nextLink") or "").strip()
    if next_short:
        next_link, reason = _link_evidence(next_short, by_short, "nextLink")
        if next_link is not None:
            candidates.append(next_link - 1)
        elif reason:
            skipped.append(reason)

    if not candidates:
        return None, "; ".join(skipped) or "no usable prevLink/nextLink evidence"
    inferred = candidates[0]
    if inferred < 1:
        return None, f"adjacent pointers imply invalid link {inferred}"
    if any(candidate != inferred for candidate in candidates):
        return None, f"prevLink/nextLink imply different links: {', '.join(str(candidate) for candidate in candidates)}"
    return inferred, ""


def _infer_single_root_link(chain_id: str, members: list[dict[str, Any]], missing_link: list[dict[str, Any]]) -> tuple[int | None, str]:
    if len(members) != 1 or len(missing_link) != 1:
        return None, ""
    task = missing_link[0]
    uuid = str(task.get("uuid") or "").strip()
    if short_uuid(uuid) != chain_id:
        return None, "single-task chain is not rooted at its own chainID"
    if str(task.get("prevLink") or "").strip():
        return None, "single-task chain has prevLink, so it does not look like root link 1"
    return 1, ""


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

        if missing_link:
            by_short = _short_index(members)
            tasks_by_inferred: dict[int, list[dict[str, Any]]] = {}
            unresolved: list[tuple[dict[str, Any], str]] = []
            single_root_link, single_root_reason = _infer_single_root_link(chain_id, members, missing_link)

            for task in missing_link:
                if single_root_link is not None:
                    inferred, reason = single_root_link, ""
                elif single_root_reason:
                    inferred, reason = None, single_root_reason
                else:
                    inferred, reason = _infer_missing_link(task, by_short)
                uuid = str(task.get("uuid") or "").strip()
                if inferred is None or not uuid:
                    unresolved.append((task, reason or "missing task UUID"))
                    continue
                tasks_by_inferred.setdefault(inferred, []).append(task)

            for inferred, rows in sorted(tasks_by_inferred.items()):
                if inferred in slots:
                    unresolved.extend((task, f"inferred link {inferred} is already occupied") for task in rows)
                    continue
                if len(rows) > 1:
                    unresolved.extend((task, f"{len(rows)} missing-link tasks infer the same link {inferred}") for task in rows)
                    continue
                task = rows[0]
                uuid = str(task.get("uuid") or "").strip()
                short = short_uuid(uuid)
                if uuid and short:
                    repairs.append(LinkRepair(uuid, short, chain_id, inferred, "link", str(task.get("link") or ""), str(inferred)))

            if unresolved:
                issues.append(
                    ChainIssue(
                        "missing_link",
                        chain_id,
                        f"{len(unresolved)} task(s) are missing a numeric link",
                        [task_summary(task, reason=reason) for task, reason in unresolved],
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
