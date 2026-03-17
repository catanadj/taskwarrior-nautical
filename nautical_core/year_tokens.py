from __future__ import annotations

from . import common as _common
from . import tokenutil as _tokenutil


def year_full_months_span_token(m1: int, m2: int, *, tok_range) -> str:
    """Full span across months [m1..m2], respecting the caller's year format."""
    return tok_range(1, int(m1), 31, int(m2))


def rewrite_month_names_to_ranges(spec: str, *, tok_range) -> str:
    if not spec:
        return spec

    out = []
    for raw in _common.split_csv_tokens(spec):
        s = raw.lower()
        if ".." in s:
            a, b = [x.strip() for x in s.split("..", 1)]
            if a in _tokenutil.MONTH_ALIAS and b in _tokenutil.MONTH_ALIAS:
                m1, m2 = _tokenutil.MONTH_ALIAS[a], _tokenutil.MONTH_ALIAS[b]
                if m1 <= m2:
                    out.append(tok_range(1, m1, _tokenutil.static_month_last_day(m2), m2))
                else:
                    out.append(raw)
                continue
        if s in _tokenutil.MONTH_ALIAS:
            mm = _tokenutil.MONTH_ALIAS[s]
            out.append(tok_range(1, mm, _tokenutil.static_month_last_day(mm), mm))
            continue
        out.append(raw)

    seen, dedup = set(), []
    for tok in out:
        if tok not in seen:
            dedup.append(tok)
            seen.add(tok)
    return ",".join(dedup)


def year_full_month_range_token(mm: int, *, tok_range) -> str:
    return tok_range(1, int(mm), 31, int(mm))


def rewrite_year_month_aliases_in_context(dnf: list[list[dict]], *, tok_range) -> list[list[dict]]:
    """
    In-place normalize yearly specs that are pure month references into
    full-month numeric ranges that the yearly gate understands.
    """
    for term in dnf:
        for atom in term:
            if (atom.get("typ") or atom.get("type") or "").lower() != "y":
                continue
            spec = (atom.get("spec") or atom.get("value") or "").strip().lower()
            if not spec:
                continue

            new_tokens: list[str] = []
            changed = False
            for tok in _common.split_csv_tokens(spec):
                if ".." in tok and "-" not in tok:
                    left, right = [x.strip() for x in tok.split("..", 1)]
                    m1, m2 = _tokenutil.mon_to_int(left), _tokenutil.mon_to_int(right)
                    if m1 and m2:
                        new_tokens.append(tok_range(1, m1, 31, m2))
                        changed = True
                        continue

                m_single = _tokenutil.mon_to_int(tok)
                if m_single:
                    new_tokens.append(year_full_month_range_token(m_single, tok_range=tok_range))
                    changed = True
                    continue

                new_tokens.append(tok)

            if changed:
                atom["spec"] = ",".join(new_tokens)

    return dnf
