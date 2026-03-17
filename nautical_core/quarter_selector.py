from __future__ import annotations


MONTH_SELECTOR_MAX_LEN = 64


def quarter_atom_spec(atom: dict) -> str:
    return (atom.get("spec") or atom.get("value") or "").lower()


def has_quarter_tokens(spec: str, *, split_csv_lower, re_mod) -> bool:
    for tok in split_csv_lower(spec):
        if re_mod.fullmatch(r"q[1-4][sme]?(?:\.\.q[1-4][sme]?)?", tok):
            return True
    return False


def has_plain_quarter_tokens(spec: str, *, split_csv_lower, re_mod) -> bool:
    for tok in split_csv_lower(spec):
        if re_mod.fullmatch(r"q[1-4](?:\.\.q[1-4])?", tok):
            return True
    return False


def is_negative_ascii_int(tok: str) -> bool:
    if len(tok) < 2 or tok[0] != "-":
        return False
    rest = tok[1:]
    return rest.isascii() and rest.isdigit()


def is_start_month_selector(
    tok: str,
    *,
    parse_error_cls,
    safe_match,
    nth_weekday_re,
) -> bool:
    t = (tok or "").strip().lower()
    if len(t) > MONTH_SELECTOR_MAX_LEN:
        raise parse_error_cls("Monthly selector too long (max 64 characters).")
    if t in ("1", "1bd"):
        return True
    match = safe_match(nth_weekday_re, t, max_len=MONTH_SELECTOR_MAX_LEN)
    if not match:
        return False
    n_raw = (match.group(1) or "").lower()
    return n_raw in ("1", "1st")


def is_end_month_selector(
    tok: str,
    *,
    parse_error_cls,
    safe_match,
    nth_weekday_re,
    bd_re,
) -> bool:
    t = (tok or "").strip().lower()
    if len(t) > MONTH_SELECTOR_MAX_LEN:
        raise parse_error_cls("Monthly selector too long (max 64 characters).")
    if is_negative_ascii_int(t):
        return True
    match_bd = safe_match(bd_re, t, max_len=MONTH_SELECTOR_MAX_LEN)
    if match_bd and int(match_bd.group(1)) < 0:
        return True
    match = safe_match(nth_weekday_re, t, max_len=MONTH_SELECTOR_MAX_LEN)
    if not match:
        return False
    n_raw = (match.group(1) or "").lower()
    return n_raw == "last" or n_raw.startswith("-")


def quarter_month_selector_mode(
    m_atoms: list[dict],
    *,
    parse_error_cls,
    expand_monthly_aliases,
    split_csv_tokens,
    is_start_month_selector,
    is_end_month_selector,
) -> str:
    if len(m_atoms) != 1:
        raise parse_error_cls(
            "Quarter aliases (y:q1..q4) cannot be combined with multiple monthly atoms in the same term. "
            "Use a single m:* selector or replace y:q* with explicit months (e.g. y:oct..dec)."
        )
    mspec = (m_atoms[0].get("spec") or m_atoms[0].get("value") or "").strip().lower()
    mspec = expand_monthly_aliases(mspec)
    if mspec == "rand":
        raise parse_error_cls(
            "Quarter aliases (y:q1..q4) cannot be combined with m:rand. "
            "Use explicit months if you need randomness within a quarter-like window."
        )
    mtoks = split_csv_tokens(mspec)
    if len(mtoks) != 1:
        raise parse_error_cls(
            "Quarter aliases (y:q1..q4) require a single monthly selector token when used with m:*. "
            "Examples: m:1bd + y:q4 (start month) OR m:-1bd + y:q4 (end month). "
            "If you meant a specific month of the quarter, use y:q4s/y:q4m/y:q4e."
        )

    mtok = mtoks[0]
    if is_end_month_selector(mtok):
        return "quarter_end"
    if is_start_month_selector(mtok):
        return "quarter_start"
    raise parse_error_cls(
        "Quarter aliases (y:q1..q4) paired with m:* are ambiguous here. "
        "Use y:qNs/y:qNm/y:qNe to target start/mid/end month, or use explicit months (e.g. y:oct..dec)."
    )


def term_quarter_rewrite_mode(
    y_atoms: list[dict],
    m_atoms: list[dict],
    *,
    quarter_atom_spec,
    has_plain_quarter_tokens,
    quarter_month_selector_mode,
) -> str:
    if not m_atoms:
        return "first_month"
    has_plain_quarters = any(has_plain_quarter_tokens(quarter_atom_spec(atom)) for atom in y_atoms)
    if not has_plain_quarters:
        return "first_month"
    return quarter_month_selector_mode(m_atoms)
