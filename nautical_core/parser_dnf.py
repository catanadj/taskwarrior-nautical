from __future__ import annotations

from . import position_selection


def _group_has_date_modifiers(mods: dict) -> bool:
    return bool(
        mods.get("roll")
        or mods.get("wd") is not None
        or mods.get("bd")
        or int(mods.get("day_offset", 0) or 0)
        or int(mods.get("business_day_offset", 0) or 0)
    )


def _apply_group_modifiers(res, mods: dict, *, parse_error_cls) -> None:
    has_date_modifiers = _group_has_date_modifiers(mods)
    if has_date_modifiers and any(len(term) != 1 for term in res):
        raise parse_error_cls(
            "Grouped date modifiers require OR-only branches. "
            "Attach date modifiers to individual atoms in groups containing '+'."
        )
    for term in res:
        for atom in term:
            atom_mods = atom.setdefault("mods", {})
            if mods.get("t"):
                if atom_mods.get("t"):
                    raise parse_error_cls(
                        "Cannot apply a grouped @t modifier because the group already has a timed term."
                    )
                tval = mods["t"]
                atom_mods["t"] = list(tval) if isinstance(tval, list) else tval
            if not has_date_modifiers:
                continue
            if _group_has_date_modifiers(atom_mods):
                raise parse_error_cls(
                    "Cannot combine grouped date modifiers with date modifiers already inside the group."
                )
            if mods.get("roll"):
                atom_mods["roll"] = mods["roll"]
                atom_mods["wd"] = mods.get("wd")
            if mods.get("bd"):
                atom_mods["bd"] = True
            atom_mods["day_offset"] = int(mods.get("day_offset", 0) or 0)
            atom_mods["business_day_offset"] = int(mods.get("business_day_offset", 0) or 0)


def parse_anchor_expr_to_dnf(
    s: str,
    *,
    normalize_anchor_expr_input,
    raise_on_bad_colon_year_tokens,
    parse_anchor_atom_at,
    parse_atom_mods,
    skip_ws_pos,
    rewrite_quarters_in_context,
    rewrite_year_month_aliases_in_context,
    validate_year_tokens_in_dnf,
    validate_and_terms_satisfiable,
    max_anchor_dnf_terms: int,
    parse_error_cls,
    today,
):
    s = normalize_anchor_expr_input(s)
    raise_on_bad_colon_year_tokens(s)

    i = 0
    n = len(s)

    def parse_atom():
        nonlocal i
        node, i = parse_anchor_atom_at(s, i, n)
        return node

    def parse_factor(depth: int = 0):
        nonlocal i
        if depth > 50:
            raise parse_error_cls("Expression nesting too deep")
        i = skip_ws_pos(s, i, n)
        if i < n and s[i] == "(":
            i += 1
            res = parse_expr(depth + 1)
            i = skip_ws_pos(s, i, n)
            if i >= n or s[i] != ")":
                raise parse_error_cls("Unclosed '('")
            i += 1
            i = skip_ws_pos(s, i, n)
            if i < n and s[i] == "@":
                start = i
                i += 1
                while i < n:
                    if s[i] in "|)":
                        break
                    if s[i] == "+" and s[i - 1] != "@":
                        break
                    i += 1
                mods_text = s[start:i].strip()
                if not mods_text.strip("@").strip():
                    raise parse_error_cls("Grouped modifier is empty.")
                try:
                    selection = position_selection.parse_group_selection_modifier(mods_text)
                except ValueError as exc:
                    raise parse_error_cls(str(exc)) from None
                if selection is not None:
                    scope, positions, remaining_modifiers = selection
                    mods = parse_atom_mods(remaining_modifiers)
                    node = {
                        "kind": "select",
                        "scope": scope,
                        "positions": positions,
                        "expr": res,
                        "mods": mods,
                    }
                    try:
                        normalized = position_selection.validate_public_selection_node(node)
                    except ValueError as exc:
                        raise parse_error_cls(str(exc)) from None
                    return [[normalized]]
                mods = parse_atom_mods(mods_text)
                _apply_group_modifiers(res, mods, parse_error_cls=parse_error_cls)
            return res
        return parse_atom()

    def and_merge(a_terms, b_terms):
        out = []
        for ta in a_terms:
            for tb in b_terms:
                out.append(ta + tb)
                if len(out) > max_anchor_dnf_terms:
                    raise parse_error_cls(
                        f"Expression too complex: more than {max_anchor_dnf_terms} combined terms."
                    )
        return out

    def parse_term(depth: int = 0):
        nonlocal i
        left = parse_factor(depth)
        while True:
            pos = i
            i = skip_ws_pos(s, i, n)
            if i >= n or s[i] != "+" or (i > 0 and s[i - 1] == "@"):
                i = pos
                break
            i += 1
            right = parse_factor(depth)
            left = and_merge(left, right)
        return left

    def parse_expr(depth: int = 0):
        nonlocal i
        left = parse_term(depth)
        while True:
            pos = i
            i = skip_ws_pos(s, i, n)
            if i >= n or s[i] != "|":
                i = pos
                break
            i += 1
            right = parse_term(depth)
            if len(left) + len(right) > max_anchor_dnf_terms:
                raise parse_error_cls(
                    f"Expression too complex: more than {max_anchor_dnf_terms} OR terms."
                )
            left = left + right
        return left

    res = parse_expr(0)
    i = skip_ws_pos(s, i, n)
    if i != n:
        raise parse_error_cls("Unexpected trailing characters")
    dnf = rewrite_quarters_in_context(res)
    dnf = rewrite_year_month_aliases_in_context(dnf)
    validate_year_tokens_in_dnf(dnf)
    validate_and_terms_satisfiable(dnf, ref_d=today())
    return dnf
