"""Typed public document boundary for the Doctor operator."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class DoctorReport(dict[str, Any]):
    """Validated Doctor document with lossless forward-compatible fields."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if not isinstance(document, Mapping):
            raise ValueError("Doctor report must be an object")
        if document.get("schema") != "nautical.doctor":
            raise ValueError("invalid Doctor report schema")
        if document.get("schema_version") != 1:
            raise ValueError("unsupported Doctor report version")
        if document.get("version") != 2 or document.get("operation") != "diagnose":
            raise ValueError("invalid Doctor report envelope")
        if str(document.get("status") or "") not in {
            "ok", "attention", "repairable", "deferred", "manual_review",
            "unavailable", "partial", "error",
        }:
            raise ValueError("invalid Doctor report status")
        super().__init__(document)

    @classmethod
    def from_mapping(cls, value: object) -> "DoctorReport":
        if not isinstance(value, Mapping):
            raise ValueError("Doctor report must be an object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


__all__ = ["DoctorReport"]
