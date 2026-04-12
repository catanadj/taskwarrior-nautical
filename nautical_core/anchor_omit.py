from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable


_OMIT_TIMED_ERROR = "omit does not support time modifiers (@t). Omit rules are date-based only."


def combine_omit_state(*, omit_dnf=None, omit_dates=None, omit_descriptions=None):
    dates = frozenset(omit_dates or [])
    descriptions = dict(omit_descriptions or {})
    if not omit_dnf and not dates and not descriptions:
        return None
    if not dates and not descriptions:
        return omit_dnf
    return {"dnf": omit_dnf, "dates": dates, "descriptions": descriptions}


def _split_omit_state(omit_state):
    if not omit_state:
        return None, frozenset(), {}
    if isinstance(omit_state, dict):
        return (
            omit_state.get("dnf"),
            frozenset(omit_state.get("dates") or []),
            dict(omit_state.get("descriptions") or {}),
        )
    return omit_state, frozenset(), {}


def _split_top_level(expr: str, delim: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")" and depth > 0:
            depth -= 1
        if ch == delim and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts


def normalize_omit_expr(expr: str) -> str:
    s = str(expr or "").strip()
    if not s:
        return s
    terms = _split_top_level(s, "|")
    out_terms: list[str] = []
    for term in terms:
        parts = _split_top_level(term, "+")
        if len(parts) <= 1:
            out_terms.append(term.strip())
            continue
        new_parts: list[str] = []
        for part in parts:
            p = part.strip()
            if "," in p and not (p.startswith("(") and p.endswith(")")):
                p = f"({p})"
            new_parts.append(p)
        out_terms.append(" + ".join(new_parts))
    return " | ".join(t for t in out_terms if t)


def validate_omit_expr_strict(
    expr: str | list[list[dict[str, Any]]],
    *,
    validate_anchor_expr_cached: Callable[[str | list[list[dict[str, Any]]]], list[list[dict[str, Any]]]],
) -> list[list[dict[str, Any]]]:
    if isinstance(expr, str):
        expr = normalize_omit_expr(expr)
    dnf = validate_anchor_expr_cached(expr)
    for term in dnf:
        for atom in term:
            mods = atom.get("mods") or {}
            if mods.get("t"):
                raise ValueError(_OMIT_TIMED_ERROR)
    return dnf


def omit_expr_fires_on_date(
    omit_dnf,
    d,
    default_seed,
    seed_base,
    *,
    core: Any,
) -> bool:
    omit_expr_dnf, omit_dates, _omit_descriptions = _split_omit_state(omit_dnf)
    if d in omit_dates:
        return True
    if not omit_expr_dnf:
        return False
    try:
        nxt, _ = core.next_after_expr(
            omit_expr_dnf,
            d - timedelta(days=1),
            default_seed=default_seed,
            seed_base=seed_base,
        )
        return nxt == d
    except Exception:
        return False


def omit_description_for_date(omit_dnf, d) -> str | None:
    _omit_expr_dnf, _omit_dates, omit_descriptions = _split_omit_state(omit_dnf)
    text = str(omit_descriptions.get(d) or "").strip()
    return text or None


def next_after_expr_with_omit(
    dnf,
    after_date,
    default_seed=None,
    seed_base=None,
    *,
    omit_dnf=None,
    core: Any,
    max_skip_iterations: int = 512,
):
    omit_expr_dnf, omit_dates, _omit_descriptions = _split_omit_state(omit_dnf)
    if not omit_expr_dnf and not omit_dates:
        return core.next_after_expr(
            dnf,
            after_date,
            default_seed=default_seed,
            seed_base=seed_base,
        )

    probe = after_date
    for _ in range(max_skip_iterations):
        nxt, meta = core.next_after_expr(
            dnf,
            probe,
            default_seed=default_seed,
            seed_base=seed_base,
        )
        if nxt is None:
            return (None, meta)
        if nxt <= probe:
            raise ValueError("No valid anchor occurrences found after applying omit rules.")
        if not omit_expr_fires_on_date(
            omit_dnf,
            nxt,
            default_seed,
            seed_base,
            core=core,
        ):
            return (nxt, meta)
        probe = nxt

    raise ValueError("No valid anchor occurrences found after applying omit rules.")
