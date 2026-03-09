#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared type aliases used by hooks and core."""

from __future__ import annotations

from typing import Any, TypeAlias, TypedDict


class AnchorMods(TypedDict, total=False):
    t: str
    bd: bool
    wd: bool
    pbd: int
    nbd: int
    nw: bool


class AnchorAtom(TypedDict, total=False):
    typ: str
    type: str
    spec: str
    value: str
    interval: int
    mods: AnchorMods


AnchorTerm: TypeAlias = list[AnchorAtom]
AnchorDNF: TypeAlias = list[AnchorTerm]
TaskDict: TypeAlias = dict[str, Any]
AnchorValidationResult: TypeAlias = tuple[AnchorDNF | None, str | None]


class HintMetaCfg(TypedDict, total=False):
    fmt: str
    salt: str
    tz: str
    hol: str


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
    dnf: AnchorDNF
    natural: str
    next_dates: list[str]
    per_year: HintPerYear
    limits: HintLimits
    rand_preview: list[str]
