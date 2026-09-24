"""Shared callable service-port types for composition boundaries."""

from __future__ import annotations

from typing import Any, Protocol


class CallbackPort(Protocol):
    """A callable collaborator supplied by an owning composition root."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
