from __future__ import annotations


def split_top_level(expr: str, delim: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == delim and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def normalize_grouped_list_filters(s: str) -> str:
    terms = split_top_level(s, "|")
    out_terms: list[str] = []
    for term in terms:
        parts = split_top_level(term, "+")
        if len(parts) <= 1:
            out_terms.append(term.strip())
            continue
        new_parts: list[str] = []
        for part in parts:
            p = part.strip()
            if "," in p and not (p.startswith("(") and p.endswith(")")):
                p = f"({p})"
            new_parts.append(p)
        out_terms.append(" + ".join(new_parts))
    return " | ".join(t for t in out_terms if t)


def rewrite_weekly_multi_time_atoms(s: str, *, split_csv_tokens, re_mod) -> str:
    """
    Rewrite patterns like:
        w:mon@t=09:00,fri@t=15:00
    into:
        w:mon@t=09:00 | w:fri@t=15:00

    Rules:
      - Only triggers inside a single weekly atom (starts with 'w:').
      - Splits on top-level commas, but keeps each token's @t with it.
      - Leaves existing '|' and '+' structure intact.
    """
    out = []
    i, n = 0, len(s)

    def flush_atom(prefix: str, body: str):
        parts = split_csv_tokens(body)
        if len(parts) <= 1:
            out.append(prefix + body)
            return
        pat = re_mod.compile(r"^(mon|tue|wed|thu|fri|sat|sun)(@t=\d{2}:\d{2})?$", re_mod.I)
        if all(pat.match(p) for p in parts):
            out.append(" | ".join(f"{prefix}{p}" for p in parts))
        else:
            out.append(prefix + body)

    while i < n:
        if s[i] == "w":
            colon = -1
            k = i + 1
            while k < n:
                if s[k] == ":":
                    colon = k
                    break
                if s[k] in "|+)(":
                    break
                k += 1
            if colon == -1:
                out.append(s[i])
                i += 1
                continue
            j = colon + 1
            depth = 0
            while j < n:
                c = s[j]
                if c == "(":
                    depth += 1
                elif c == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif depth == 0 and c in "|+":
                    break
                j += 1
            flush_atom(s[i:colon + 1], s[colon + 1:j])
            i = j
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


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
    s = normalize_grouped_list_filters(s)
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
