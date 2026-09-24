"""Typed payload contracts produced by the anchor-hint builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from .parsing.parser_models import AnchorDNF


class HintMetaCfg(TypedDict, total=False):
    fmt: str
    salt: str
    tz: str
    hol: str
    bc: str


class HintMeta(TypedDict, total=False):
    created: int
    cfg: HintMetaCfg


class HintPerYear(TypedDict, total=False):
    est: int
    first: str
    last: str


class HintLimits(TypedDict, total=False):
    stop: str
    max_left: int
    until: str


class AnchorHintsPayload(TypedDict, total=False):
    meta: HintMeta
    dnf: "AnchorDNF"
    natural: str
    next_dates: list[str]
    per_year: HintPerYear
    limits: HintLimits
    rand_preview: list[str]


__all__ = (
    "HintMetaCfg",
    "HintMeta",
    "HintPerYear",
    "HintLimits",
    "AnchorHintsPayload",
)
