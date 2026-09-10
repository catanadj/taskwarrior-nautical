"""Immutable runtime bindings returned by core-bound API factories."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


def core_namespace(
    module: Any,
    namespace: Mapping[str, Any] | None,
    context: Any,
    owner: str,
) -> dict[str, Any]:
    """Resolve one factory's explicit core source with a stable error."""
    if context is not None:
        return context.namespace
    if namespace is not None:
        return namespace if isinstance(namespace, dict) else dict(namespace)
    if module is None:
        raise TypeError(f"{owner}.for_core requires a module, namespace, or CoreContext")
    return vars(module)


@dataclass(frozen=True)
class ApiBinding:
    """A read-only, explicitly owned set of core API members.

    API modules expose many small functions whose exact signatures belong to
    their owning modules.  This container supplies the common boundary
    contract without allowing callers to add or replace members at runtime.
    ``from_mapping`` copies its input, so a caller cannot mutate a binding by
    retaining the source dictionary.
    """

    _members: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, members: Mapping[str, Any]) -> "ApiBinding":
        return cls(MappingProxyType(dict(members)))

    @classmethod
    def from_kwargs(cls, **members: Any) -> "ApiBinding":
        """Build a binding from the named members of one API factory."""
        return cls.from_mapping(members)

    def __getattr__(self, name: str) -> Any:
        try:
            return self._members[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._members))

    def keys(self):
        """Return the immutable binding member names for introspection."""
        return self._members.keys()


__all__ = ("ApiBinding", "core_namespace")
