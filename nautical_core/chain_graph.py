"""Pure immutable graph construction for chain integrity observations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, TypeVar

from .chain_integrity_models import (
    ChainNode,
    ChainReference,
    ChainSnapshot,
    IntegrityContractError,
    ReferenceState,
    SnapshotCoverage,
)


IndexKey = TypeVar("IndexKey", str, tuple[str, int])


def _index(values: Mapping[IndexKey, list[ChainNode]]) -> Mapping[IndexKey, tuple[ChainNode, ...]]:
    return MappingProxyType({key: tuple(items) for key, items in values.items()})


def _ordered(nodes: tuple[ChainNode, ...]) -> tuple[ChainNode, ...]:
    return tuple(sorted(nodes, key=lambda node: (node.chain_id, node.link is None, node.link or 0, node.task_uuid)))


@dataclass(frozen=True, slots=True)
class ChainGraph:
    """Deterministic, read-only indexes and edge evidence for one snapshot."""

    snapshot: ChainSnapshot
    nodes: tuple[ChainNode, ...]
    by_uuid: Mapping[str, tuple[ChainNode, ...]]
    by_short_uuid: Mapping[str, tuple[ChainNode, ...]]
    by_chain: Mapping[str, tuple[ChainNode, ...]]
    by_slot: Mapping[tuple[str, int], tuple[ChainNode, ...]]
    by_status: Mapping[str, tuple[ChainNode, ...]]
    references: Mapping[tuple[str, str], ChainReference]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ChainSnapshot):
            raise IntegrityContractError("chain graph requires a chain snapshot")
        nodes = tuple(self.nodes)
        if any(not isinstance(node, ChainNode) for node in nodes):
            raise IntegrityContractError("chain graph nodes must be ChainNode values")
        object.__setattr__(self, "nodes", _ordered(nodes))
        for name in ("by_uuid", "by_short_uuid", "by_chain", "by_slot", "by_status", "references"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise IntegrityContractError(f"chain graph {name} index must be a mapping")
            if name == "references":
                normalized = dict(value)
            else:
                normalized = {key: tuple(items) for key, items in value.items()}
            object.__setattr__(self, name, MappingProxyType(normalized))

    @classmethod
    def from_snapshot(cls, snapshot: ChainSnapshot) -> "ChainGraph":
        if not isinstance(snapshot, ChainSnapshot):
            raise TypeError("chain graph requires a ChainSnapshot")
        if snapshot.coverage is SnapshotCoverage.UNAVAILABLE:
            raise IntegrityContractError("unavailable snapshots cannot construct a chain graph")
        nodes = _ordered(snapshot.rows)
        by_uuid: dict[str, list[ChainNode]] = {}
        by_short: dict[str, list[ChainNode]] = {}
        by_chain: dict[str, list[ChainNode]] = {}
        by_slot: dict[tuple[str, int], list[ChainNode]] = {}
        by_status: dict[str, list[ChainNode]] = {}
        for node in nodes:
            uuid = node.task_uuid.lower()
            by_uuid.setdefault(uuid, []).append(node)
            by_short.setdefault(uuid[:8], []).append(node)
            if node.chain_id:
                by_chain.setdefault(node.chain_id, []).append(node)
                if node.link is not None:
                    by_slot.setdefault((node.chain_id, node.link), []).append(node)
            by_status.setdefault(node.status, []).append(node)
        references: dict[tuple[str, str], ChainReference] = {}
        for node in nodes:
            for field in ("prevLink", "nextLink"):
                references[(node.task_uuid, field)] = cls._resolve_reference(node, field, by_uuid, by_short)
        return cls(
            snapshot,
            nodes,
            _index(by_uuid),
            _index(by_short),
            _index(by_chain),
            _index(by_slot),
            _index(by_status),
            MappingProxyType(references),
        )

    @staticmethod
    def _resolve_reference(
        node: ChainNode,
        field: str,
        by_uuid: Mapping[str, list[ChainNode]],
        by_short: Mapping[str, list[ChainNode]],
    ) -> ChainReference:
        token = node.reference_token(field)
        if not token:
            return ChainReference(field, "", ReferenceState.ABSENT, reason="reference is empty")
        matches = list(by_uuid.get(token, ()))
        if not matches:
            if len(token) == 8:
                matches = list(by_short.get(token, ()))
            else:
                matches = [candidate for uuid, values in by_uuid.items() if uuid.startswith(token) for candidate in values]
        if len(matches) > 1:
            return ChainReference(field, token, ReferenceState.AMBIGUOUS, reason="reference matches multiple nodes")
        if not matches:
            return ChainReference(field, token, ReferenceState.OUTSIDE_COVERAGE, reason="target is outside snapshot coverage")
        target = matches[0]
        return ChainReference(field, token, ReferenceState.RESOLVED, target.task_uuid, target.link)

    def uuid_matches(self, value: str) -> tuple[ChainNode, ...]:
        token = str(value or "").strip().lower()
        return self.by_uuid.get(token, self.by_short_uuid.get(token, ()))

    def chain_nodes(self, chain_id: str) -> tuple[ChainNode, ...]:
        return self.by_chain.get(str(chain_id or "").strip(), ())

    def slot_nodes(self, chain_id: str, link: int) -> tuple[ChainNode, ...]:
        return self.by_slot.get((str(chain_id or "").strip(), int(link)), ())

    def status_nodes(self, status: str) -> tuple[ChainNode, ...]:
        return self.by_status.get(str(status or "").strip().lower(), ())

    def lifecycle_nodes(self, intent: str) -> tuple[ChainNode, ...]:
        """Return nodes with the requested semantic lifecycle intent."""
        token = str(intent or "").strip().lower()
        return tuple(node for node in self.nodes if node.lifecycle_intent.value == token)

    def roots(self, chain_id: str = "") -> tuple[ChainNode, ...]:
        """Return chain roots whose predecessor is explicitly absent."""
        nodes = self.chain_nodes(chain_id) if str(chain_id or "").strip() else self.nodes
        return tuple(
            node for node in nodes
            if not node.reference_token("prevLink")
        )

    def tips(self, chain_id: str = "") -> tuple[ChainNode, ...]:
        """Return chain tips whose successor is explicitly absent."""
        nodes = self.chain_nodes(chain_id) if str(chain_id or "").strip() else self.nodes
        return tuple(
            node for node in nodes
            if not node.reference_token("nextLink")
        )

    def orphan_candidates(self) -> tuple[ChainNode, ...]:
        """Return nodes with an unresolved non-empty edge reference."""
        return tuple(
            node for node in self.nodes
            if any(
                self.reference(node.task_uuid, field).state in {
                    ReferenceState.OUTSIDE_COVERAGE,
                    ReferenceState.AMBIGUOUS,
                    ReferenceState.UNAVAILABLE,
                }
                for field in ("prevLink", "nextLink")
            )
        )

    def referenced_children(self) -> tuple[ChainNode, ...]:
        """Return resolved successor targets, de-duplicated deterministically."""
        uuids = {
            reference.target_uuid
            for reference in self.references.values()
            if reference.field == "nextLink" and reference.state is ReferenceState.RESOLVED
        }
        return tuple(node for node in self.nodes if node.task_uuid in uuids)

    def reference(self, task_uuid: str, field: str) -> ChainReference:
        normalized_field = str(field or "").strip()
        if normalized_field not in {"prevLink", "nextLink"}:
            raise ValueError("chain reference field must be prevLink or nextLink")
        return self.references.get(
            (str(task_uuid or "").strip(), normalized_field),
            ChainReference(normalized_field, "", ReferenceState.UNAVAILABLE, reason="node is outside graph"),
        )

    def to_dict(self) -> dict[str, object]:
        """Return deterministic diagnostic evidence for this immutable graph.

        This is an invocation-local representation for tests and structured
        diagnostics, not a persistence format.  Nodes and references are
        sorted independently so equivalent exports produce identical data
        regardless of Taskwarrior's row ordering.
        """
        nodes = tuple(node.to_dict() for node in self.nodes)
        references = tuple(
            {
                "task_uuid": task_uuid,
                "field": field,
                "token": reference.token,
                "state": reference.state.value,
                "target_uuid": reference.target_uuid,
                "target_link": reference.target_link,
                "reason": reference.reason,
            }
            for (task_uuid, field), reference in sorted(self.references.items())
        )
        return {
            "snapshot_id": self.snapshot.snapshot_id,
            "coverage": self.snapshot.coverage.value,
            "source": self.snapshot.source,
            "configuration_fingerprint": self.snapshot.configuration_fingerprint,
            "complete_chain_history": self.snapshot.complete_chain_history,
            "nodes": nodes,
            "references": references,
        }
