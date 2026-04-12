from __future__ import annotations


def parse_anchor_expr_to_dnf(
    s: str,
    *,
    normalize_anchor_expr_input,
    raise_on_bad_colon_year_tokens,
    parse_anchor_atom_at,
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
