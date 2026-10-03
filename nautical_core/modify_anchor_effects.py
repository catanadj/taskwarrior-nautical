"""Anchor omission-state assembly for the typed on-modify workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .anchor_omit import OmitState


OmitDNF = list[list[dict[str, Any]]]


class OmitStateCombiner(Protocol):
    def __call__(
        self,
        *,
        omit_dnf: OmitDNF | None = None,
        omit_dates: frozenset[date] | None = None,
        omit_descriptions: Mapping[date, str] | None = None,
    ) -> OmitState | None: ...


@dataclass(frozen=True, slots=True)
class OmitPorts:
    validate_omit: Callable[[str], OmitDNF]
    load_omit_file_data: Callable[
        [str | None, str | None], tuple[frozenset[date], dict[date, str]]
    ]
    omit_file_dir: str
    combine_omit_state: OmitStateCombiner


def omit_ports_for(host: Any) -> OmitPorts:
    omit_files = host.core._import_sibling("omit_files")
    return OmitPorts(
        validate_omit=host._validate_omit_expr_cached,
        load_omit_file_data=omit_files.load_omit_file_data,
        omit_file_dir=getattr(host.core, "OMIT_FILE_DIR", ""),
        combine_omit_state=host._module("anchor_omit").combine_omit_state,
    )


def omit_dnf_from_parent(
    ports: OmitPorts, task_mapping: dict[str, Any]
) -> tuple[str, OmitState | None]:
    expr_str = (task_mapping.get("omit") or "").strip()
    omit_file = (task_mapping.get("omit_file") or "").strip()
    omit_dnf = None
    omit_dates: frozenset[date] = frozenset()
    omit_descriptions: dict[date, str] = {}
    if expr_str:
        try:
            omit_dnf = ports.validate_omit(expr_str)
        except Exception as exc:
            raise ValueError(f"Invalid omit expression '{expr_str}': {exc}") from exc
    if omit_file:
        try:
            omit_dates, omit_descriptions = ports.load_omit_file_data(
                omit_file, ports.omit_file_dir
            )
        except Exception as exc:
            raise ValueError(f"Invalid omit_file '{omit_file}': {exc}") from exc
    if not omit_dnf and not omit_dates and not omit_descriptions:
        return "", None
    return expr_str, ports.combine_omit_state(
        omit_dnf=omit_dnf,
        omit_dates=omit_dates,
        omit_descriptions=omit_descriptions,
    )


__all__ = ("OmitPorts", "omit_ports_for", "omit_dnf_from_parent")
