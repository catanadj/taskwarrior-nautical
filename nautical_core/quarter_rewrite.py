from __future__ import annotations


def rewrite_quarter_spec_mode(
    spec: str,
    mode: str,
    *,
    meta_out: dict | None = None,
    split_csv_lower,
    tok_range,
    static_month_last_day,
    quarter_pos_month: dict[int, dict[str, int]],
    re_mod,
) -> str:
    """
    Rewrite q1..q4 tokens in a yearly spec into concrete y:* tokens.
    """
    if not spec:
        return spec

    qmap: dict[str, str] = {}

    def first_month_window(q: int) -> str:
        mm = {1: 1, 2: 4, 3: 7, 4: 10}[q]
        return tok_range(1, mm, static_month_last_day(mm), mm)

    def last_month_window(q: int) -> str:
        mm = {1: 3, 2: 6, 3: 9, 4: 12}[q]
        return tok_range(1, mm, static_month_last_day(mm), mm)

    def mid_month_window(q: int) -> str:
        mm = {1: 2, 2: 5, 3: 8, 4: 11}[q]
        return tok_range(1, mm, static_month_last_day(mm), mm)

    def pos_month_window(q: int, pos: str) -> str | None:
        mm = quarter_pos_month.get(q, {}).get(pos)
        if not mm:
            return None
        return tok_range(1, mm, static_month_last_day(mm), mm)

    def emit(q: int) -> str:
        if mode == "quarter_end":
            return last_month_window(q)
        if mode == "quarter_mid":
            return mid_month_window(q)
        if mode == "quarter_window":
            start = {1: 1, 2: 4, 3: 7, 4: 10}[q]
            end = {1: 3, 2: 6, 3: 9, 4: 12}[q]
            return tok_range(1, start, 31, end)
        return first_month_window(q)

    def note(q: int) -> str:
        if mode == "quarter_end":
            return f"Q{q} end month"
        if mode == "quarter_mid":
            return f"Q{q} mid month"
        if mode == "quarter_start":
            return f"Q{q} start month"
        if mode == "quarter_window":
            return f"Q{q} window"
        return f"Q{q} first month"

    out = []
    toks = split_csv_lower(spec)

    for tok in toks:
        match = re_mod.fullmatch(r"q([1-4])([sme])", tok)
        if match:
            q = int(match.group(1))
            pos = match.group(2)
            window = pos_month_window(q, pos)
            if window:
                out.append(window)
                pos_note = {"s": "start", "m": "mid", "e": "end"}[pos]
                qmap[window] = f"Q{q} {pos_note} month"
            continue

        match = re_mod.fullmatch(r"q([1-4])([sme])\.\.q([1-4])([sme])", tok)
        if match:
            qa, qb = int(match.group(1)), int(match.group(3))
            posa, posb = match.group(2), match.group(4)
            if posa != posb:
                out.append(tok)
                continue
            if qa > qb:
                out.append(tok)
            else:
                for q in range(qa, qb + 1):
                    window = pos_month_window(q, posa)
                    if window:
                        out.append(window)
                        pos_note = {"s": "start", "m": "mid", "e": "end"}[posa]
                        qmap[window] = f"Q{q} {pos_note} month"
            continue

        match = re_mod.fullmatch(r"q([1-4])", tok)
        if match:
            q = int(match.group(1))
            window = emit(q)
            out.append(window)
            qmap[window] = note(q)
            continue

        match = re_mod.fullmatch(r"q([1-4])\.\.q([1-4])", tok)
        if match:
            qa, qb = int(match.group(1)), int(match.group(2))
            if qa > qb:
                out.append(tok)
            else:
                for q in range(qa, qb + 1):
                    window = emit(q)
                    out.append(window)
                    qmap[window] = note(q)
            continue

        out.append(tok)

    seen, dedup = set(), []
    for tok in out:
        if tok not in seen:
            dedup.append(tok)
            seen.add(tok)

    if meta_out is not None and qmap:
        meta_out.update(qmap)

    return ",".join(dedup)


def rewrite_quarter_year_atoms(
    y_atoms: list[dict],
    mode: str,
    *,
    quarter_atom_spec,
    has_quarter_tokens,
    rewrite_quarter_spec_mode,
) -> None:
    for atom in y_atoms:
        spec = quarter_atom_spec(atom)
        if not has_quarter_tokens(spec):
            continue
        qmeta: dict[str, str] = {}
        atom["spec"] = rewrite_quarter_spec_mode(spec, mode, meta_out=qmeta)
        if qmeta:
            atom["_qmap"] = qmeta


def rewrite_quarters_in_context(
    dnf,
    *,
    has_quarter_tokens,
    quarter_atom_spec,
    term_quarter_rewrite_mode,
    rewrite_quarter_year_atoms,
):
    for term in dnf:
        y_atoms = [a for a in term if (a.get("typ") or a.get("type") or "").lower() == "y"]
        if not y_atoms:
            continue
        if not any(has_quarter_tokens(quarter_atom_spec(a)) for a in y_atoms):
            continue
        m_atoms = [a for a in term if (a.get("typ") or a.get("type") or "").lower() == "m"]
        mode = term_quarter_rewrite_mode(y_atoms, m_atoms)
        rewrite_quarter_year_atoms(y_atoms, mode)
    return dnf
