"""Argument contract for the reconcile operator front end."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from collections.abc import Callable


@dataclass(slots=True)
class ReconcileRequest:
    """Normalized reconcile operation submitted by the CLI front end."""

    apply: bool
    dry_run: bool
    task_bin: str
    json: bool
    verbose: bool
    full_audit: bool
    chain_id: str | None
    uuid: str | None
    no_housekeeping: bool
    max_expiration_hops: int

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ReconcileRequest":
        request = cls(
            apply=bool(args.apply),
            dry_run=bool(args.dry_run),
            task_bin=str(args.task_bin),
            json=bool(args.json),
            verbose=bool(args.verbose),
            full_audit=bool(args.full_audit),
            chain_id=args.chain_id,
            uuid=args.uuid,
            no_housekeeping=bool(args.no_housekeeping),
            max_expiration_hops=int(args.max_expiration_hops),
        )
        if request.apply and request.dry_run:
            raise ValueError("reconcile request cannot apply and dry-run together")
        if request.chain_id and request.uuid:
            raise ValueError("reconcile request cannot combine chainID and UUID scope")
        return request


def build_parser(
    *,
    expiration_hop_limit: Callable[[str], int],
    default_expiration_hops: int,
    max_expiration_hops: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair Nautical chains after hookless completion, expiration, or deletion."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Apply repairs.")
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview repairs without mutating Taskwarrior (the default).",
    )
    parser.add_argument("--task-bin", default="task", help="Taskwarrior binary to execute.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary.")
    parser.add_argument("--verbose", action="store_true", help="Print every delayed-recovery hop.")
    parser.add_argument(
        "--full-audit",
        action="store_true",
        help="Export complete chain history for deep validation; default audits active tips and unresolved terminals.",
    )
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


__all__ = ["ReconcileRequest", "build_parser"]
