"""Public natural-language description entry points for the core facade."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any):
    """Create description APIs bound to one core module instance."""
    return SimpleNamespace(
        describe_anchor_expr=module._describe_anchor_expr_impl,
        describe_anchor_dnf=module._describe_anchor_dnf_impl,
    )


__all__ = ("for_core",)
