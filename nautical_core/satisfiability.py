from __future__ import annotations

from datetime import timedelta
import re


def weekday_set_from_weekly_atom(atom, *, weekly_spec_to_wset) -> set[int]:
    if (atom.get("typ") or "").lower() != "w":
        return set()
    spec = atom.get("spec") or ""
    mods = atom.get("mods") or {}
    return weekly_spec_to_wset(spec, mods=mods)


def md_pairs_from_yearly_spec(spec: str, *, expand_yearly_cached, leap_year_for_checks: int) -> set[tuple[int, int]]:
    if not spec:
        return set()
    try:
        dates = expand_yearly_cached(spec, leap_year_for_checks)
    except Exception:
        return set()
    return {(d.month, d.day) for d in dates}


def quick_weekly_and_check(term: list[dict], *, weekday_set_from_weekly_atom, and_term_unsatisfiable_cls) -> None:
    w_sets = [
        weekday_set_from_weekly_atom(atom)
        for atom in term
        if (atom.get("typ") or "").lower() == "w"
    ]
    if len(w_sets) >= 2:
        inter = set.intersection(*w_sets) if all(w_sets) else set()
        if not inter:
            raise and_term_unsatisfiable_cls(
                "Weekly anchors joined with '+' never coincide (e.g., Saturday AND Monday). "
                "Use ',' (OR) or '|' instead."
            )


def quick_yearly_and_check(term: list[dict], *, md_pairs_from_yearly_spec, and_term_unsatisfiable_cls) -> None:
    y_atoms = [a for a in term if (a.get("typ") or "").lower() == "y"]
    if len(y_atoms) < 2:
        return
    md_sets = []
    for atom in y_atoms:
        spec = (atom.get("spec") or "").strip().lower()
        s = md_pairs_from_yearly_spec(spec)
        if s:
            md_sets.append(s)
    if len(md_sets) >= 2:
        inter = set.intersection(*md_sets)
        if not inter:
            joined = ", ".join((atom.get("spec") or "").strip().lower() for atom in y_atoms)
            raise and_term_unsatisfiable_cls(
                "Yearly anchors joined with '+' never overlap within a year. "
                f"If you intended 'either/or', join them with commas: y:{joined}"
            )


def term_has_any_match_within(
    term: list[dict],
    start,
    seed,
    *,
    atom_matches_on,
    years: int = 8,
) -> bool:
    def matches_or_flexible(atom, d):
        typ = (atom.get("typ") or "").lower()
        spec = (atom.get("spec") or "").lower()
        if typ in ("w", "m") and "rand" in spec:
            return True
        return atom_matches_on(atom, d, seed)

    limit = start + timedelta(days=366 * max(1, years))
    d = start
    while d <= limit:
        if all(matches_or_flexible(atom, d) for atom in term):
            return True
        d += timedelta(days=1)
    return False


def validate_and_terms_satisfiable(
    dnf: list[list[dict]],
    ref_d,
    *,
    quick_weekly_and_check,
    quick_yearly_and_check,
    term_has_any_match_within,
    normalize_spec_for_acf,
    month_from_alias,
    and_term_unsatisfiable_cls,
) -> None:
    seed = ref_d
    for term in dnf:
        if len(term) < 2:
            continue
        quick_weekly_and_check(term)
        quick_yearly_and_check(term)
        if not term_has_any_match_within(term, ref_d, seed, years=8):
            pieces = []
            for atom in term:
                typ = (atom.get("typ") or "").lower()
                spec = (atom.get("spec") or "").strip()
                if typ in ("w", "m"):
                    try:
                        spec = normalize_spec_for_acf(typ, spec) or spec
                    except Exception:
                        pass
                if typ == "m" and spec:
                    m = re.fullmatch(r"([a-z]{3,9}|\d{2})(\.\.([a-z]{3,9}|\d{2}))?", spec.lower())
                    if m and month_from_alias(m.group(1)) is not None:
                        right = m.group(3)
                        if right is None or month_from_alias(right) is not None:
                            typ = "y"
                if typ:
                    pieces.append(f"{typ}:{spec}" if spec else typ)
            hint = ", ".join(pieces)
            raise and_term_unsatisfiable_cls(
                "These anchors joined with '+' don't share any possible date. "
                "If you meant 'either/or', join them with ',' (OR) or use '|'. "
                f"Example: {hint.replace(' + ', ', ')}"
            )
