"""Short Nautical UDA directives embedded in a task description."""

from __future__ import annotations

import re


ALIAS_TO_FIELD = {
    "a": "anchor",
    "af": "anchor_file",
    "am": "anchor_mode",
    "o": "omit",
    "of": "omit_file",
    "cm": "chainMax",
    "cu": "chainUntil",
}
_ALIAS_RE = re.compile(r"(?<!\S)(a|af|am|o|of|cm|cu):", re.IGNORECASE)
_ANCHOR_VALUE_RE = re.compile(r"^(?:-|\(|@|(?:w|m|y|d|bd|moon|in-)[^\s:]*:)", re.IGNORECASE)
_CHAIN_UNTIL_RE = re.compile(r"^(?:-|today|tomorrow|eow|eom|eoy|sow|som|\d)", re.IGNORECASE)


def _alias_value_is_plausible(field: str, value: str) -> bool:
    """Reject obvious prose collisions before aliases reach full validation."""
    if value == "-":
        return True
    if field in {"anchor", "omit"}:
        return bool(_ANCHOR_VALUE_RE.match(value))
    if field == "anchor_mode":
        return value.lower() in {"skip", "all", "flex"}
    if field == "chainMax":
        return value.isdigit() and int(value) > 0
    if field == "chainUntil":
        return bool(_CHAIN_UNTIL_RE.match(value))
    return True


def parse_description_aliases(description: object) -> tuple[str, dict[str, str]]:
    """Extract a trailing block of short UDA directives from a description.

    Values must begin immediately after the colon. This avoids treating normal
    prose such as ``a: book`` as a directive while still allowing expressions
    containing spaces, for example ``a:(w:mon | w:fri) am:all``.
    """
    text = str(description or "")
    matches = list(_ALIAS_RE.finditer(text))
    if not matches:
        return text, {}

    for start_index, start_match in enumerate(matches):
        values: dict[str, str] = {}
        valid = True
        for index, match in enumerate(matches[start_index:], start=start_index):
            value_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            value = text[match.end():value_end].strip()
            alias = match.group(1).lower()
            field = ALIAS_TO_FIELD[alias]
            if not value:
                raise ValueError(f"Description alias '{alias}:' requires a value.")
            if text[match.end():match.end() + 1].isspace():
                valid = False
                break
            if not _alias_value_is_plausible(field, value):
                valid = False
                break
            if field in values:
                raise ValueError(f"Description alias '{alias}:' was specified more than once.")
            values[field] = value
        if valid and values:
            return text[:start_match.start()].rstrip(), values

    return text, {}


def apply_description_aliases(task: dict, previous: dict | None = None) -> bool:
    """Apply parsed aliases to a task, returning whether any were found.

    On modification, an alias may replace a canonical value when that value
    was unchanged from the previous task. ``alias:-`` explicitly clears it.
    """
    description = task.get("description")
    if not isinstance(description, str) or not description:
        return False
    clean_description, fields = parse_description_aliases(description)
    if not fields:
        return False
    for field, value in fields.items():
        current = task.get(field)
        old = previous.get(field) if isinstance(previous, dict) else None
        changed_since_previous = previous is not None and current != old
        if changed_since_previous and current not in (None, "") and str(current) != value:
            raise ValueError(f"{field} is already set to a different value")
        if value == "-":
            if changed_since_previous:
                raise ValueError(f"{field} was changed separately; cannot clear it with an alias")
            task.pop(field, None)
        else:
            if previous is None and current not in (None, "") and str(current) != value:
                raise ValueError(f"{field} is already set to a different value")
            task[field] = value
    task["description"] = clean_description
    return True


__all__ = ("ALIAS_TO_FIELD", "apply_description_aliases", "parse_description_aliases")
