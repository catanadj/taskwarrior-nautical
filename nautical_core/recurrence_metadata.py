"""Small, shared metadata operations for recurrence atoms and periods."""

from __future__ import annotations

from datetime import date


def active_mod_keys(mods: dict) -> set:
    """Return modifier keys that carry an effective value."""
    return {
        key
        for key, value in (mods or {}).items()
        if value not in (None, False, 0, 0.0, "", [])
    }


def atom_type(atom) -> str:
    return (atom.get("typ") or atom.get("type") or "").lower()


def atom_spec(atom) -> str:
    return str(atom.get("spec") or atom.get("value") or "").lower()


def atom_mods(atom) -> dict:
    return atom.get("mods") or {}


def atom_interval(atom) -> int:
    try:
        return int(atom.get("ival") or atom.get("intv") or 1)
    except Exception:
        return 1


def months_since(seed_local: date, year: int, month: int) -> int:
    """Return the signed number of calendar months from ``seed_local``."""
    return (year - seed_local.year) * 12 + (month - seed_local.month)


__all__ = (
    "active_mod_keys",
    "atom_type",
    "atom_spec",
    "atom_mods",
    "atom_interval",
    "months_since",
)
