from __future__ import annotations


def completion_link_numbers_or_fail(
    new: dict,
    *,
    coerce_int,
    max_link_number: int,
    panel,
    print_task,
) -> tuple[int, int] | None:
    base_no = coerce_int(new.get("link"), 1)
    if base_no < 1 or base_no > max_link_number:
        panel(
            "⛔ Link number invalid",
            [("Reason", f"Link number {base_no} is outside 1..{max_link_number}.")],
            kind="error",
        )
        print_task(new)
        return None
    next_no = base_no + 1
    if next_no > max_link_number:
        panel(
            "⛔ Link limit exceeded",
            [("Reason", f"Link number {next_no} exceeds max_link_number={max_link_number}.")],
            kind="error",
        )
        print_task(new)
        return None
    return base_no, next_no


def completion_kind_or_stop(
    new: dict,
    now_utc,
    *,
    panel,
    print_task,
    end_chain_summary,
) -> str | None:
    raw_ch = (new.get("chain") or "").strip().lower()
    has_anchor = bool((new.get("anchor") or "").strip())
    has_cp = bool((new.get("cp") or "").strip())
    effective_on = (raw_ch == "on") or (raw_ch == "" and (has_anchor or has_cp))
    if not effective_on:
        if has_anchor or has_cp:
            panel(
                "Chain disabled (chain:off) — no next link will be spawned.",
                [],
                kind="disabled",
            )
            print_task(new)
            end_chain_summary(new, "Manual stop.", now_utc)
        else:
            print_task(new)
        return None

    kind = "anchor" if has_anchor else ("cp" if has_cp else None)
    if not kind:
        print_task(new)
        return None
    return kind


def completion_chain_id_or_fail(new: dict, *, panel, print_task) -> str | None:
    chain_id = (new.get("chainID") or new.get("chainid") or "").strip()
    if chain_id:
        return chain_id
    panel(
        "⛔ ChainID missing",
        [
            ("Reason", "ChainID is required in v3+ and legacy link-walk is removed."),
            ("Fix", "Run tools/nautical_backfill_chainid.py, then retry."),
        ],
        kind="error",
    )
    print_task(new)
    return None


def completion_existing_next_or_fail(
    new: dict,
    next_no: int,
    *,
    existing_next_task,
    short,
    panel,
    print_task,
) -> bool:
    existing_next = existing_next_task(new, next_no)
    if not existing_next:
        return True
    ex_uuid = (existing_next.get("uuid") or "").strip()
    ex_short = short(ex_uuid)
    ex_status = ((existing_next.get("status") or "").strip() or "unknown").lower()
    panel(
        "ℹ Spawn skipped",
        [
            ("Reason", "Next link already exists for this completed task."),
            ("Existing", f"#{next_no} {ex_short} ({ex_status})"),
        ],
        kind="note",
    )
    print_task(new)
    return False


def completion_preflight_context(
    new: dict,
    now_utc,
    *,
    short,
    completion_link_numbers_or_fail,
    completion_kind_or_stop,
    completion_chain_id_or_fail,
    completion_existing_next_or_fail,
) -> dict | None:
    parent_short = short(new.get("uuid"))
    nums = completion_link_numbers_or_fail(new)
    if nums is None:
        return None
    base_no, next_no = nums

    kind = completion_kind_or_stop(new, now_utc)
    if not kind:
        return None

    chain_id = completion_chain_id_or_fail(new)
    if not chain_id:
        return None

    if not completion_existing_next_or_fail(new, next_no):
        return None

    return {
        "parent_short": parent_short,
        "base_no": base_no,
        "next_no": next_no,
        "kind": kind,
        "chain_id": chain_id,
    }
