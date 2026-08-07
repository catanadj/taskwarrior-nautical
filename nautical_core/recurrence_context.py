"""Shared identity and runtime context for recurrence evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class RecurrenceContext:
    """Immutable context shared by recurrence resolvers.

    ``chain_id`` is the stable identity used by deterministic recurrence
    features.  The remaining fields are optional during migration; callers
    can adopt the context incrementally without changing existing APIs.
    """

    chain_id: str
    timezone: Any | None = None
    business_calendar: Any | None = None
    astronomy_config: Mapping[str, Any] | None = None
    anchor_file_dir: str = ""
    namespace: str = "nautical"

    def __post_init__(self) -> None:
        chain_id = str(self.chain_id or "").strip()
        if not chain_id:
            raise ValueError("Recurrence context requires a chain ID.")
        object.__setattr__(self, "chain_id", chain_id)

    @property
    def seed_base(self) -> str:
        """Compatibility view for the existing seed-based resolver APIs."""
        return self.chain_id

    @classmethod
    def from_task(
        cls,
        task: Mapping[str, Any],
        *,
        fallback_chain_id: str | None = None,
        **kwargs: Any,
    ) -> "RecurrenceContext":
        """Build context, with any non-chain fallback requiring explicit opt-in."""
        chain_id = task.get("chainID") or fallback_chain_id or ""
        return cls(chain_id=str(chain_id), **kwargs)


__all__ = ("RecurrenceContext",)
