from __future__ import annotations


def parse_y_token(
    tok: str,
    fmt: str,
    *,
    quarters,
    months,
    y_token_re,
    re_mod,
):
    """Parse yearly token (e.g., '15-02' or 'q1')."""
    tok = tok.strip().lower()
    if tok in quarters:
        return ("quarter", tok)
    match = re_mod.fullmatch(r"q([1-4])([sme])", tok)
    if match:
        return ("quarter", tok)
    match = y_token_re.match(tok)
    if not match:
        return None
    a, b = match.group(1), match.group(2)
    if b.isalpha():
        if b not in months:
            return None
        b = months[b]
    else:
        b = int(b)
    a = int(a)
    if fmt == "DM":
        d, mn = a, b
    else:
        mn, d = a, b
    if not (1 <= mn <= 12):
        return None
    if mn in (4, 6, 9, 11) and d == 31:
        return None
    if mn == 2 and d > 29:
        return None
    if not (1 <= d <= 31):
        return None
    return ("day", (mn, d))
