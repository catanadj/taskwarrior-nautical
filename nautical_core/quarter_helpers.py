from __future__ import annotations


def yearly_tokens(term, *, split_csv_tokens):
    out = []
    for atom in term:
        if (atom.get("typ") or atom.get("type") or "").lower() == "y":
            spec = (atom.get("spec") or atom.get("value") or "").lower()
            out.extend(split_csv_tokens(spec))
    return out


def monthly_tokens(term, *, split_csv_tokens):
    out = []
    for atom in term:
        if (atom.get("typ") or atom.get("type") or "").lower() == "m":
            spec = (atom.get("spec") or atom.get("value") or "").lower()
            out.extend(split_csv_tokens(spec))
    return out


def quarters_from_tokens(y_toks, *, token_rev: dict[str, int]):
    qs = []
    for tok in y_toks:
        q = token_rev.get(tok)
        if q:
            qs.append(q)
    return sorted(set(qs))


def format_quarter_set(qs):
    """Return ('each quarter' | 'Q2' | 'Q1–Q2' | 'Q1 and Q3')."""
    if not qs:
        return None
    if qs == [1, 2, 3, 4]:
        return "each quarter"
    if max(qs) - min(qs) + 1 == len(qs):
        if len(qs) == 2:
            return f"Q{qs[0]}–Q{qs[1]}"
        return ", ".join(f"Q{x}" for x in qs[:-1]) + f" and Q{qs[-1]}"
    if len(qs) == 1:
        return f"Q{qs[0]}"
    return ", ".join(f"Q{x}" for x in qs[:-1]) + f" and Q{qs[-1]}"
