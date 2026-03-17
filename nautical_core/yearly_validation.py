from __future__ import annotations

import re


YEARLY_MONTH_MAX = {
    1: 31,
    2: 29,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31,
}

YEARLY_QUARTER_RE = re.compile(r"^q[1-4][sme]?$")
YEARLY_QUARTER_RANGE_RE = re.compile(r"^(q[1-4])([sme])?\.\.(q[1-4])([sme])?$")
YEARLY_MONTH_ONLY_RE = re.compile(r"^\d{1,2}$")
YEARLY_MONTH_RANGE_ONLY_RE = re.compile(r"^\d{1,2}\.\.\d{1,2}$")
YEARLY_NON_PADDED_DM_RE = re.compile(r"^\d{1,2}-\d{1,2}(?:\.\.\d{1,2}-\d{1,2})?$")
YEARLY_PADDED_DM_RE = re.compile(
    r"^(?P<d1>\d{2})-(?P<m1>\d{2})(?:\.\.(?P<d2>\d{2})-(?P<m2>\d{2}))?$"
)


def yearly_pair_from_fmt(a: int, b: int, fmt: str) -> tuple[int, int]:
    return (b, a) if fmt == "MD" else (a, b)


def yearly_mmdd_error(mm: int, dd: int) -> str | None:
    if not (1 <= mm <= 12):
        return f"month '{mm:02d}' is invalid"
    if not (1 <= dd <= 31):
        return f"day '{dd:02d}' is invalid"
    return None


def validate_yearly_token_allowlist(tok: str, fmt: str, *, year_token_format_error_cls) -> None:
    s = tok

    if s == "rand" or re.fullmatch(r"rand-\d{2}", s):
        return

    if re.fullmatch(r"\d{2}:\d{2}(?::\d{2}:\d{2})?", s):
        example = "06-01" if fmt == "MD" else "01-06"
        raise year_token_format_error_cls(
            f"Yearly token '{tok}' uses ':' between numbers. "
            f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{example}'."
        )

    if re.fullmatch(r"\d{2}-\d{2}(?:\.\.\d{2}-\d{2})?", s):
        return

    if re.fullmatch(r"(?:[a-z]{3}|\d{2})\.\.(?:[a-z]{3}|\d{2})", s):
        return
    if re.fullmatch(r"[a-z]{3}", s):
        return
    if re.fullmatch(r"q[1-4](?:\.\.q[1-4])?", s):
        return

    raise year_token_format_error_cls(
        f"Unknown yearly token '{tok}'. Expected day-month, month alias, or quarter."
    )


def validate_yearly_token_detailed(tok: str, fmt: str, *, year_token_format_error_cls) -> tuple[str, str] | None:
    s = tok.strip().lower()

    if s == "rand":
        return None

    m_randm = re.fullmatch(r"rand-(\d{2})", s)
    if m_randm:
        mm = int(m_randm.group(1))
        if 1 <= mm <= 12:
            return None
        raise year_token_format_error_cls(f"Invalid month in yearly token '{tok}'. Expected 01..12.")

    m = re.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", s)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        d1, m1 = yearly_pair_from_fmt(a, b, fmt)
        err = yearly_mmdd_error(m1, d1)
        if err:
            return tok, err
        if m.group(3):
            c, d = int(m.group(3)), int(m.group(4))
            d2, m2 = yearly_pair_from_fmt(c, d, fmt)
            err2 = yearly_mmdd_error(m2, d2)
            if err2:
                return tok, err2
            if (m2, d2) < (m1, d1):
                return tok, "end precedes start"
        return None

    m_col1 = re.fullmatch(r"(\d{2}):(\d{2})$", s)
    m_col2 = re.fullmatch(r"(\d{2}):(\d{2}):(\d{2}):(\d{2})$", s)
    if m_col1 or m_col2:
        if m_col1:
            a, b = int(m_col1.group(1)), int(m_col1.group(2))
            ex = f"{a:02d}-{b:02d}" if fmt == "MD" else f"{b:02d}-{a:02d}"
        else:
            a, b, c, d = map(int, m_col2.groups())
            ex = (
                f"{a:02d}-{b:02d}..{c:02d}-{d:02d}"
                if fmt == "MD"
                else f"{b:02d}-{a:02d}..{d:02d}-{c:02d}"
            )
        raise year_token_format_error_cls(
            f"Yearly token '{tok}' uses ':' between numbers. "
            f"Use '-' and order per ANCHOR_YEAR_FMT={fmt}. Example: '{ex}'."
        )

    if ":" in s:
        raise year_token_format_error_cls(
            "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
        )

    if any(ch.isdigit() for ch in s) and any(ch in s for ch in "-:"):
        ex = "MM-DD" if fmt == "MD" else "DD-MM"
        raise year_token_format_error_cls(
            f"Yearly token '{tok}' doesn’t match ANCHOR_YEAR_FMT={fmt}. "
            f"Expected {ex} or {ex}..{ex}."
        )

    return None


def validate_yearly_token_format(
    spec: str,
    *,
    yearfmt,
    split_csv_lower,
    year_token_format_error_cls,
) -> None:
    fmt = yearfmt()
    if not spec:
        return

    tokens = split_csv_lower(spec)
    for tok in tokens:
        validate_yearly_token_allowlist(tok, fmt, year_token_format_error_cls=year_token_format_error_cls)

    bad = None
    for tok in tokens:
        bad = validate_yearly_token_detailed(tok, fmt, year_token_format_error_cls=year_token_format_error_cls)
        if bad:
            break

    if bad:
        tok, reason = bad
        sug = (
            "Did you mean MM-DD? e.g., '04-20'."
            if fmt == "MD"
            else "Did you mean DD-MM? e.g., '20-04'."
        )
        raise year_token_format_error_cls(
            f"Yearly token '{tok}' doesn’t match ANCHOR_YEAR_FMT={fmt}. {reason}. {sug}"
        )


def validate_year_tokens_in_dnf(
    dnf,
    *,
    validate_yearly_token_format,
) -> None:
    for term in dnf:
        for atom in term:
            if (atom.get("typ") or "").lower() == "y":
                spec = (atom.get("spec") or "").strip()
                validate_yearly_token_format(spec)


def validate_yearly_token(
    tok: str,
    *,
    quarters,
    parse_y_token,
    parse_error_cls,
) -> None:
    tok = tok.strip().lower()
    if tok in quarters or re.fullmatch(r"q[1-4][sme]", tok):
        return
    if ":" in tok:
        raise parse_error_cls(
            f"Invalid yearly range '{tok}'. Use '..' (e.g., 'y:07-01..07-31', 'y:q1..q2')."
        )
    if ".." in tok:
        a, b = tok.split("..", 1)
        pa = parse_y_token(a)
        pb = parse_y_token(b)
        if not pa or not pb:
            raise parse_error_cls(f"Invalid yearly range '{tok}'")
        return
    p = parse_y_token(tok)
    if not p:
        raise parse_error_cls(f"Unknown yearly token '{tok}'")


def yearly_last_day(mm: int) -> int:
    return YEARLY_MONTH_MAX.get(mm, 31)


def yearly_check_day_month(
    dd: int,
    mm: int,
    label: str,
    tok: str,
    *,
    parse_error_cls,
    month_full,
) -> None:
    if mm < 1 or mm > 12:
        raise parse_error_cls(
            f"Invalid month '{mm:02d}' in '{tok}' ({label}). Months must be 01..12."
        )
    maxd = yearly_last_day(mm)
    if dd < 1 or dd > maxd:
        near = maxd if dd > maxd else 1
        hint = f" {month_full[mm]} has {maxd} days."
        sug1 = f"{near:02d}-{mm:02d}"
        sug2 = f"01-{mm:02d}..{maxd:02d}-{mm:02d}"
        raise parse_error_cls(
            f"Invalid day '{dd:02d}' for month '{mm:02d}' in '{tok}' ({label}).{hint} "
            f"Try '{sug1}' or '{sug2}'."
        )


def validate_yearly_spec_token(
    tok: str,
    *,
    parse_error_cls,
    month_full,
) -> None:
    if ":" in tok:
        raise parse_error_cls(
            "Yearly ranges must use '..' (e.g., '01-01..12-31', 'q1..q2')."
        )
    if YEARLY_QUARTER_RE.fullmatch(tok):
        return

    m = YEARLY_QUARTER_RANGE_RE.fullmatch(tok)
    if m:
        q_from = int(m.group(1)[1])
        q_to = int(m.group(3)[1])
        suf_from = m.group(2) or ""
        suf_to = m.group(4) or ""
        if suf_from != suf_to:
            raise parse_error_cls(
                f"Invalid quarter range '{tok}': suffixes must match "
                "(use q1s..q2s or q1..q2)."
            )
        if q_to < q_from:
            raise parse_error_cls(
                f"Invalid quarter range '{tok}': end quarter precedes start quarter. "
                f"Split across the year boundary, e.g., 'q{q_from}, q{q_to}'."
            )
        return

    if YEARLY_MONTH_ONLY_RE.match(tok):
        mm = int(tok)
        if not (1 <= mm <= 12):
            raise parse_error_cls(
                f"Invalid month '{tok}'. Months must be 01..12. "
                f"Try '{mm:02d}' or the full-month form '01-{mm:02d}..{yearly_last_day(mm):02d}-{mm:02d}'."
            )
        raise parse_error_cls(
            f"Yearly token '{tok}' is incomplete. Did you mean the full month? "
            f"Try '01-{mm:02d}..{yearly_last_day(mm):02d}-{mm:02d}'."
        )

    if YEARLY_MONTH_RANGE_ONLY_RE.match(tok):
        m1, m2 = (int(x) for x in tok.split("..", 1))
        if not (1 <= m1 <= 12 and 1 <= m2 <= 12):
            raise parse_error_cls(
                f"Invalid month range '{tok}'. Months must be 01..12. "
                f"Try '01-{m1:02d}..{yearly_last_day(m2):02d}-{m2:02d}'."
            )
        if m2 < m1:
            left = f"01-{m1:02d}..31-12"
            right = f"01-01..{yearly_last_day(m2):02d}-{m2:02d}"
            raise parse_error_cls(
                f"Invalid month range '{tok}': end month is before start month. "
                f"Split across years, e.g., '{left}, {right}'."
            )
        raise parse_error_cls(
            f"Yearly token '{tok}' is incomplete. Did you mean a full multi-month range? "
            f"Try '01-{m1:02d}..{yearly_last_day(m2):02d}-{m2:02d}'."
        )

    if YEARLY_NON_PADDED_DM_RE.match(tok) and not YEARLY_PADDED_DM_RE.match(tok):
        pieces = re.split(r"-|\\.\\.", tok)
        padded = "..".join(
            [f"{int(pieces[0]):02d}-{int(pieces[1]):02d}"]
            + ([f"{int(pieces[2]):02d}-{int(pieces[3]):02d}"] if len(pieces) == 4 else [])
        )
        raise parse_error_cls(
            f"Invalid yearly token '{tok}'. Use zero-padded 'DD-MM' or 'DD-MM..DD-MM'. "
            f"Try '{padded}'."
        )

    m = YEARLY_PADDED_DM_RE.match(tok)
    if not m:
        raise parse_error_cls(
            f"Unknown yearly token '{tok}'. Expected 'DD-MM', 'DD-MM..DD-MM', "
            f"or quarter aliases 'q1..q4'/'q1s/q1m/q1e' (e.g., 'q1', 'q1s', 'q1..q2')."
        )

    d1 = int(m.group("d1"))
    m1 = int(m.group("m1"))
    d2g = m.group("d2")
    m2g = m.group("m2")

    if d2g is None and m2g is None:
        yearly_check_day_month(d1, m1, "single day", tok, parse_error_cls=parse_error_cls, month_full=month_full)
        return

    d2 = int(d2g)
    m2 = int(m2g)
    yearly_check_day_month(d1, m1, "range start", tok, parse_error_cls=parse_error_cls, month_full=month_full)
    yearly_check_day_month(d2, m2, "range end", tok, parse_error_cls=parse_error_cls, month_full=month_full)
    if (m2, d2) < (m1, d1):
        left = f"{d1:02d}-{m1:02d}..31-12"
        right = f"01-01..{d2:02d}-{m2:02d}"
        raise parse_error_cls(
            f"Invalid range '{tok}': start must be on/before end; cross-year ranges "
            f"aren't supported. Try splitting: '{left}, {right}'."
        )


def validate_yearly_spec(
    spec: str,
    *,
    split_csv_lower,
    validate_yearly_spec_token,
    parse_error_cls,
) -> None:
    toks = split_csv_lower(spec)
    if not toks:
        raise parse_error_cls("Empty yearly spec")
    for tok in toks:
        validate_yearly_spec_token(tok)
