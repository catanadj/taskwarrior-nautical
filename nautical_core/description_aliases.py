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
            if field in values:
                raise ValueError(f"Description alias '{alias}:' was specified more than once.")
            values[field] = value
        if valid and values:
            return text[:start_match.start()].rstrip(), values

    return text, {}


__all__ = ("ALIAS_TO_FIELD", "parse_description_aliases")
