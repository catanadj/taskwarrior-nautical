from __future__ import annotations


def normalize_anchor_expr_input(
    s: str,
    *,
    unwrap_quotes,
    rewrite_weekly_multi_time_atoms,
    re_mod,
    parse_error_cls,
) -> str:
    s = unwrap_quotes(s or "").strip()
    if len(s) > 1024:
        raise parse_error_cls("Anchor expression too long (max 1024 characters).")
    s = re_mod.sub(r"\b(\d{2})-rand\b", r"rand-\1", s)
    s = rewrite_weekly_multi_time_atoms(s)
    return s


def fatal_bad_colon_in_year_tail(
    tail: str,
    *,
    split_csv_tokens,
    re_mod,
    yearfmt,
) -> str | None:
    head = tail.split("@", 1)[0]
    for tok in split_csv_tokens(head):
        if re_mod.fullmatch(r"\d{2}:\d{2}(?::\d{2}:\d{2})?", tok):
            fmt = yearfmt()
            example = "06-01" if fmt == "MD" else "01-06"
            return (
                f"Yearly token '{tok}' uses ':' between numbers. "
                f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{example}'."
            )
        if ":" in tok:
            return "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
    return None


def raise_on_bad_colon_year_tokens(
    s: str,
    *,
    re_mod,
    fatal_bad_colon_in_year_tail,
    parse_error_cls,
) -> None:
    for match in re_mod.finditer(r"\by\s*(?:/\d+)?\s*:", s):
        j = match.end()
        k = j
        while k < len(s) and s[k] not in "+|)":
            k += 1
        tail = s[j:k]
        fatal_msg = fatal_bad_colon_in_year_tail(tail)
        if fatal_msg:
            raise parse_error_cls(fatal_msg)


def skip_ws_pos(s: str, i: int, n: int) -> int:
    while i < n and s[i].isspace():
        i += 1
    return i


def raise_if_comma_joined_anchors(full_tail: str, *, re_mod, parse_error_cls) -> None:
    if re_mod.search(r"@[^)]*?,\s*(?:w|m|y)(?:/|:)", full_tail):
        raise parse_error_cls(
            "It looks like you used a comma to join anchors. "
            "Use '+' (AND) or '|' (OR), e.g. 'm:31@t=14:00 | w:sun@t=22:00'."
        )
    if re_mod.search(r",\s*(?:w|m|y)(?:/|:)", full_tail):
        raise parse_error_cls(
            "Anchors must be joined with '+' (AND) or '|' (OR). "
            "For example: 'm:31 + w:sun' or 'm:31 | w:sun'."
        )
