"""Argument contract for the reconcile operator front end."""

from __future__ import annotations

import argparse
from collections.abc import Callable


def build_parser(
    *,
    expiration_hop_limit: Callable[[str], int],
    default_expiration_hops: int,
    max_expiration_hops: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair Nautical chains after hookless completion, expiration, or deletion."
    )
    parser.add_argument("--apply", action="store_true", help="Apply repairs. Default is dry-run.")
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    parser.add_argument("--verbose", action="store_true", help="Print every delayed-recovery hop.")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--chain-id", help="Restrict audit and recovery to one chainID.")
    scope.add_argument("--uuid", help="Restrict audit and recovery to one task UUID.")
    parser.add_argument(
        "--no-housekeeping",
        action="store_true",
        help="Skip opportunistic lifecycle outbox housekeeping for this run.",
    )
    parser.add_argument(
        "--max-expiration-hops",
        type=expiration_hop_limit,
        default=default_expiration_hops,
        help=f"Maximum expired links recovered per chain (default: {default_expiration_hops}; max: {max_expiration_hops}).",
    )
    return parser


__all__ = ["build_parser"]
