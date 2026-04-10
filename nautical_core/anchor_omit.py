from __future__ import annotations

from datetime import timedelta
from typing import Any, Callable


_OMIT_TIMED_ERROR = "omit does not support time modifiers (@t). Omit rules are date-based only."


def validate_omit_expr_strict(
    expr: str | list[list[dict[str, Any]]],
    *,
    validate_anchor_expr_cached: Callable[[str | list[list[dict[str, Any]]]], list[list[dict[str, Any]]]],
) -> list[list[dict[str, Any]]]:
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
    if not omit_dnf:
        return False
    try:
        nxt, _ = core.next_after_expr(
            omit_dnf,
            d - timedelta(days=1),
            default_seed=default_seed,
            seed_base=seed_base,
        )
        return nxt == d
    except Exception:
        return False


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
    if not omit_dnf:
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

