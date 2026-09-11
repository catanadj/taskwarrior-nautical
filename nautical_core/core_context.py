"""Explicit per-instance runtime context for isolated core services."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ParserDependencies:
    """Immutable parser-owned snapshot translated at the compatibility edge."""

    values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "ParserDependencies":
        return cls(MappingProxyType(dict(values)))

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


@dataclass(frozen=True, slots=True)
class SchedulerDependencies:
    """Immutable scheduler-owned snapshot translated at the API boundary."""

    values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SchedulerDependencies":
        return cls(MappingProxyType(dict(values)))

    def __getitem__(self, name: str) -> Any:
        return self.values[name]

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)


@dataclass(frozen=True, slots=True)
class CoreContext:
    """Own one core namespace and its sibling-module loader.

    The context is intentionally small: it provides an explicit hand-off point
    for parser, scheduler, and cache services without changing legacy callers.
    """

    namespace: dict[str, Any]
    import_sibling: Any
    source_file: str | None = None

    @classmethod
    def from_core(cls, module: Any, *, namespace: Mapping[str, Any] | None = None) -> "CoreContext":
        values = namespace if namespace is not None else vars(module)
        loader = values.get("_import_sibling", getattr(module, "_import_sibling", None))
        if not callable(loader):
            raise TypeError("core context requires a callable sibling-module loader")
        return cls(dict(values), loader, getattr(module, "__file__", None))

    def get(self, name: str, default: Any = None) -> Any:
        return self.namespace.get(name, default)

    def require(self, name: str) -> Any:
        try:
            return self.namespace[name]
        except KeyError as exc:
            raise RuntimeError(f"core context is missing required binding: {name}") from exc

    def __getattr__(self, name: str) -> Any:
        """Expose legacy ``module._name`` reads through the owned namespace."""
        try:
            return self.namespace[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
