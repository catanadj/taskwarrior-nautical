"""Core-bound candidate enumeration for recurrence ranges."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    """Bind candidate enumeration to one core facade instance."""
    core = namespace if namespace is not None else vars(module)

    def anchors_between_large_range(
        dnf,
        start_excl,
        end_excl,
        default_seed,
        seed_base=None,
        scheduler_service=None,
    ):
        return core["_precompute"].anchors_between_large_range(
            dnf,
            start_excl,
            end_excl,
            default_seed,
            seed_base=seed_base,
            until_count_cap=core["UNTIL_COUNT_CAP"],
            next_after_expr=core["next_after_expr"],
            scheduler_service=scheduler_service,
        )

    def anchors_between_expr(
        dnf,
        start_excl,
        end_excl,
        default_seed,
        seed_base=None,
        *,
        scheduler_service=None,
    ):
        return core["_precompute"].anchors_between_expr(
            dnf,
            start_excl,
            end_excl,
            default_seed,
            seed_base=seed_base,
            until_count_cap=core["UNTIL_COUNT_CAP"],
            next_after_expr=core["next_after_expr"],
            anchors_between_large_range=anchors_between_large_range,
            warn_once_per_day=core["_warn_once_per_day"],
            os_mod=core["os"],
            scheduler_service=scheduler_service,
        )

    return SimpleNamespace(
        anchors_between_large_range=anchors_between_large_range,
        anchors_between_expr=anchors_between_expr,
    )


__all__ = ("for_core",)
