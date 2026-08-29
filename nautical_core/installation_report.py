"""Typed public document boundary for post-install verification."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class InstallationVerificationReport(dict[str, Any]):
    """Validated installation report with lossless extension fields."""

    def __init__(self, document: Mapping[str, Any]) -> None:
        if not isinstance(document, Mapping):
            raise ValueError("installation report must be an object")
        if document.get("schema") != "nautical.install.verification":
            raise ValueError("invalid installation report schema")
        if document.get("version") != 1:
            raise ValueError("unsupported installation report version")
        if str(document.get("status") or "") not in {"passed", "attention", "failed"}:
            raise ValueError("invalid installation report status")
        for field in ("checks", "manual_actions", "optional_actions"):
            value = document.get(field)
            if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
                raise ValueError(f"installation report {field} must be a list of objects")
        super().__init__(document)

    @classmethod
    def from_mapping(cls, value: object) -> "InstallationVerificationReport":
        if not isinstance(value, Mapping):
            raise ValueError("installation report must be an object")
        return cls(value)

    def to_dict(self) -> dict[str, Any]:
        return dict(self)


__all__ = ["InstallationVerificationReport"]
