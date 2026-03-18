from __future__ import annotations


def normalize_anchor_input_to_dnf(expr, *, parse_anchor_expr_to_dnf_cached, parse_error_cls):
    """Normalize user input to parsed DNF, preserving current error messages."""
    if isinstance(expr, str):
        s = (expr or "").strip()
        if not s:
            raise parse_error_cls("Empty anchor expression.")
        try:
            dnf = parse_anchor_expr_to_dnf_cached(s)
        except parse_error_cls as exc:
            raise parse_error_cls(f"{exc} (expr: {s})")
    elif isinstance(expr, (list, tuple)):
        dnf = expr
    else:
        raise parse_error_cls(
            f"Invalid anchor type {type(expr).__name__}; expected string or parsed DNF."
        )

    # Defensive compatibility for legacy tuple-style parser errors.
    if isinstance(dnf, tuple) and len(dnf) == 2 and isinstance(dnf[0], str):
        raise parse_error_cls(dnf[0])
    return dnf


def assert_dnf_structure_strict(dnf, *, is_atom_like, parse_error_cls) -> None:
    if not isinstance(dnf, (list, tuple)):
        raise parse_error_cls("Internal error: DNF must be a list of terms.")
    for term in dnf:
        if not isinstance(term, (list, tuple)):
            raise parse_error_cls("Internal error: each term must be a list of atoms.")
        for atom in term:
            if not isinstance(atom, dict):
                raise parse_error_cls("Internal error: each atom must be a dict.")
            if not is_atom_like(atom):
                raise parse_error_cls("Internal error: atom missing required fields (typ/spec/mods).")


def validate_anchor_atom_strict(
    atom: dict,
    *,
    validate_weekly_spec,
    validate_monthly_spec,
    active_mod_keys,
    validate_yearly_token_format,
    parse_error_cls,
) -> None:
    typ = (atom.get("typ") or atom.get("type") or "").lower()
    spec = (atom.get("spec") or atom.get("value") or "").lower()
    interval = int(atom.get("ival") or atom.get("intv") or 1)
    mods = atom.get("mods") or {}
    active = None

    if typ == "w":
        validate_weekly_spec(spec)
        return

    if typ == "m":
        if spec == "rand":
            active = active_mod_keys(mods)
            bad = [key for key in active if key not in ("t", "bd", "wd")]
            if bad:
                raise parse_error_cls(f"m:rand does not support @{', '.join(bad)}")
            if interval < 1:
                raise parse_error_cls("Monthly interval (/N) must be >= 1")
        else:
            validate_monthly_spec(spec)
        return

    if typ == "y":
        if spec == "rand" or spec.startswith("rand-"):
            if spec.startswith("rand-"):
                try:
                    mm = int(spec.split("-", 1)[1])
                except Exception:
                    raise parse_error_cls(f"Invalid token 'y:{spec}'")
                if not (1 <= mm <= 12):
                    raise parse_error_cls(f"Invalid month in y:{spec}")
            if active is None:
                active = active_mod_keys(mods)
            bad = [key for key in active if key not in ("t", "bd", "wd")]
            if bad:
                raise parse_error_cls(f"y:{spec} does not support @{', '.join(bad)}")
        else:
            validate_yearly_token_format(spec)
        return

    raise parse_error_cls(f"Unknown anchor type '{typ}'")


def validate_anchor_dnf_atoms_strict(dnf, *, validate_anchor_atom_strict) -> None:
    for term in dnf:
        for atom in term:
            validate_anchor_atom_strict(atom)


def validate_anchor_expr_strict(
    expr,
    *,
    normalize_anchor_input_to_dnf,
    assert_dnf_structure_strict,
    validate_anchor_dnf_atoms_strict,
):
    """
    Validate an anchor expression. Accepts:
      - str  (e.g., "w/2:sun + m:1st-mon"), parsed to DNF
      - DNF  (list[list[dict]]), already parsed

    Returns the normalized DNF on success; raises ParseError on failure.
    """
    dnf = normalize_anchor_input_to_dnf(expr)
    assert_dnf_structure_strict(dnf)
    validate_anchor_dnf_atoms_strict(dnf)
    return dnf
