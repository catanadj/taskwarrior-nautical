from __future__ import annotations


def _anchor_has_next_anchor(rows: list[tuple[str, str]]) -> bool:
    return any(k == "Next anchor" for k, _ in rows)


def _anchor_delta_text(rows: list[tuple[str, str]]) -> str | None:
    for k, v in rows:
        if k == "Delta":
            return v
    return None


def _anchor_upcoming_numbered(v: str, start_idx: int) -> str:
    if not v or v == "[dim]–[/]":
        return v
    lines = v.splitlines()
    new_lines = []
    idx = start_idx
    for line in lines:
        new_lines.append(f"[dim]{idx:>2} ▸[/] {line}")
        idx += 1
    return "\n".join(new_lines)


def _anchor_classify_rows(
    rows: list[tuple[str, str]],
    *,
    delta_text: str | None,
    upcoming_start: int,
) -> dict[str, list[tuple[str, str]]]:
    config_keys = {"Pattern", "Natural"}
    schedule_keys = {"First due", "Next anchor", "Scheduled", "Wait", "[auto-due]", "Upcoming"}
    limits_keys = {"Limits", "Final (until)"}
    grouped: dict[str, list[tuple[str, str]]] = {
        "config": [],
        "schedule": [],
        "limits": [],
        "warnings": [],
        "rand": [],
        "chain": [],
        "others": [],
    }

    for k, v in rows:
        if k == "Delta":
            continue
        if k == "First due" and delta_text:
            v = f"{v}  [dim](Δ {delta_text})[/]"

        lk = (str(k).lower() if k is not None else "")
        if k in config_keys:
            grouped["config"].append((k, v))
            continue
        if k in schedule_keys:
            if k == "Upcoming":
                v = _anchor_upcoming_numbered(v, upcoming_start)
            grouped["schedule"].append((k, v))
            continue
        if k in limits_keys:
            grouped["limits"].append((k, v))
            continue
        if lk.startswith("warning") or lk.startswith("note"):
            grouped["warnings"].append((k, v))
            continue
        if k == "Rand":
            grouped["rand"].append((k, v))
            continue
        if k == "Chain":
            grouped["chain"].append((k, v))
            continue
        grouped["others"].append((k, v))
    return grouped


def _anchor_compose_rows(grouped: dict[str, list[tuple[str, str]]]) -> list[tuple[str | None, str]]:
    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]) -> None:
        if not group:
            return
        if out:
            out.append((None, ""))
        out.extend(group)

    grouped["config"].extend(grouped["others"])
    _add(grouped["config"])
    _add(grouped["schedule"])
    _add(grouped["limits"])
    _add(grouped["warnings"])
    _add(grouped["rand"])
    _add(grouped["chain"])
    return out


def format_anchor_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    has_next_anchor = _anchor_has_next_anchor(rows)
    upcoming_start = 3 if has_next_anchor else 2
    delta_text = _anchor_delta_text(rows)
    grouped = _anchor_classify_rows(
        rows,
        delta_text=delta_text,
        upcoming_start=upcoming_start,
    )
    out = _anchor_compose_rows(grouped)
    return out or rows


def format_cp_rows(rows: list[tuple[str, str]]) -> list[tuple[str | None, str]]:
    upcoming_start = 2
    delta_text = None
    for k, v in rows:
        if k == "Delta":
            delta_text = v
            break

    config_keys = {"Period"}
    schedule_keys = {"First due", "Scheduled", "Wait", "[auto-due]", "Upcoming"}
    limits_keys = {"Limits", "Final (max)", "Final (until)"}

    config: list[tuple[str, str]] = []
    schedule: list[tuple[str, str]] = []
    limits: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    chain: list[tuple[str, str]] = []
    others: list[tuple[str, str]] = []

    for k, v in rows:
        if k == "Delta":
            continue

        if k == "First due" and delta_text:
            v = f"{v}  [dim](Δ {delta_text})[/]"

        lk = (str(k).lower() if k is not None else "")

        if k in config_keys:
            config.append((k, v))
        elif k in schedule_keys:
            if k == "Upcoming" and v and v != "[dim]–[/]":
                lines = v.splitlines()
                new_lines = []
                idx = upcoming_start
                for line in lines:
                    new_lines.append(f"[dim]{idx:>2} ▸[/] {line}")
                    idx += 1
                v = "\n".join(new_lines)
            schedule.append((k, v))
        elif k in limits_keys:
            limits.append((k, v))
        elif lk.startswith("warning") or lk.startswith("note"):
            warnings.append((k, v))
        elif k == "Chain":
            chain.append((k, v))
        else:
            others.append((k, v))

    config.extend(others)
    out: list[tuple[str | None, str]] = []

    def _add(group: list[tuple[str, str]]) -> None:
        if not group:
            return
        if out:
            out.append((None, ""))
        out.extend(group)

    _add(config)
    _add(schedule)
    _add(limits)
    _add(warnings)
    _add(chain)
    return out or rows
