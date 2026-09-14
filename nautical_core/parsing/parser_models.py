"""Public parser types and exceptions shared by the core facade."""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict


class AnchorMods(TypedDict, total=False):
    t: str
    bd: bool
    wd: bool
    pbd: int
    nbd: int
    nw: bool
    day_offset: int
    business_day_offset: int


class AnchorAtom(TypedDict, total=False):
    typ: str
    type: str
    spec: str
    value: str
    interval: int
    mods: AnchorMods


AnchorTerm: TypeAlias = list[AnchorAtom]
AnchorDNF: TypeAlias = list[AnchorTerm]
AnchorValidationResult: TypeAlias = tuple[AnchorDNF | None, str | None]


class ParseError(Exception):
    """Raised when an anchor expression cannot be parsed or validated."""


class YearTokenFormatError(ParseError):
    """Raised for malformed yearly anchor tokens."""


class AndTermUnsatisfiable(ParseError):
    """Raised when an AND term has no possible calendar match."""


__all__ = (
    "AnchorMods",
    "AnchorAtom",
    "AnchorTerm",
    "AnchorDNF",
    "AnchorValidationResult",
    "ParseError",
    "YearTokenFormatError",
    "AndTermUnsatisfiable",
)
