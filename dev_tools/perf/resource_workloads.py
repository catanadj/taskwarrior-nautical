"""Task-model and resource benchmark workloads."""

from __future__ import annotations

import time
import json
import tracemalloc
import uuid
from typing import Any, Sequence


def task_codec(task_codec: Any, rounds: int) -> float:
    """Measure typed task decoding across representative payload sizes."""
    codec = task_codec.DEFAULT_TASK_CODEC
    base = {
        "uuid": "00000000-0000-4000-8000-000000000001",
        "description": "codec benchmark", "status": "pending", "chain": "on",
        "chainID": "codec-perf", "link": 1, "anchor": "w:mon",
        "due": "20260824T090000Z", "entry": "20260820T090000Z",
    }
    large = {
        **base,
        "uuid": "00000000-0000-4000-8000-000000000002",
        "annotations": [{"entry": "20260820T090000Z", "description": "x" * 256} for _ in range(64)],
        "tags": [f"tag-{index}" for index in range(64)],
        "depends": [f"00000000-0000-4000-8000-{index:012d}" for index in range(32)],
        "custom": {"nested": [{"index": index, "value": "v" * 128} for index in range(32)]},
    }
    malformed_exports = ("{not-json", "[] trailing", '{"uuid":"missing-array"}')
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        for row in (base, large):
            observation = codec.decode_row(row, source_query="perf:codec")
            if observation.field("uuid").presence.value != "value":
                raise RuntimeError("codec benchmark lost task identity")
        for text in malformed_exports:
            try:
                codec.decode_export(text, source_query="perf:codec")
            except task_codec.TaskCodecError:
                continue
            raise RuntimeError("codec benchmark accepted malformed export")
    return time.perf_counter() - started


def task_immutability(task_codec: Any, rounds: int) -> float:
    """Verify frozen task fields survive source mutation without copies."""
    source = {
        "uuid": "00000000-0000-4000-8000-000000000003", "description": "immutable benchmark",
        "status": "pending", "chainID": "immutable-perf", "link": 1,
        "custom": {"nested": [{"value": "original"}, {"value": "stable"}]},
    }
    observation = task_codec.DEFAULT_TASK_CODEC.decode_row(source, source_query="perf:immutability")
    frozen = observation.arbitrary["custom"]
    source["custom"]["nested"][0]["value"] = "mutated"
    if observation.arbitrary["custom"] != frozen or "mutated" in repr(observation.arbitrary["custom"]):
        raise RuntimeError("immutable observation changed after source mutation")
    if observation.arbitrary["custom"] is not frozen:
        raise RuntimeError("immutable arbitrary field was rebuilt during access")
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        for _ in range(1000):
            if observation.arbitrary["custom"] is not frozen:
                raise RuntimeError("immutable field access returned a new value")
            if observation.field("description").value != "immutable benchmark":
                raise RuntimeError("immutable scalar field changed")
    return time.perf_counter() - started


def task_snapshot_reuse(task_codec: Any, rounds: int, row_count: int = 1000) -> float:
    """Measure indexed and graph reuse after one decode of a broad snapshot."""
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult
    from nautical_core.task_read_repository import AuthoritativeTaskSnapshot, TaskQueryKind, TaskSnapshotScope
    from nautical_core.chain_graph import ChainGraph
    from nautical_core.chain_integrity_models import ChainNode, ChainSnapshot, SnapshotCoverage
    from nautical_core.chain_invariants import evaluate_invariants

    class CountingCodec:
        def __init__(self):
            self.decode_count = 0

        def decode_row(self, row, *, source_query, **kwargs):
            self.decode_count += 1
            return task_codec.DEFAULT_TASK_CODEC.decode_row(row, source_query=source_query, **kwargs)

    rows = [
        {
            "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/snapshot/{index}")),
            "description": f"snapshot row {index}", "status": "pending", "chain": "on",
            "chainID": f"snapshot-chain-{index // 10}", "link": (index % 10) + 1,
            "anchor": "w:mon", "due": "20260824T090000Z",
        }
        for index in range(max(1, int(row_count)))
    ]
    codec = CountingCodec()
    observations = tuple(codec.decode_row(row, source_query="perf:snapshot") for row in rows)
    expected_decode_count = len(rows)
    command = TaskCommand(("task", "export"), "perf snapshot", 1.0)
    result = TaskCommandResult(command, 0, "[]", "", CommandFailureKind.SUCCESS, 1, 0.0)
    scope = TaskSnapshotScope(TaskQueryKind.BROAD, "perf-snapshot", ("pending",))
    snapshot = AuthoritativeTaskSnapshot(scope, observations, result)
    graph_snapshot = ChainSnapshot(
        "perf-snapshot-graph", SnapshotCoverage.COMPLETE, "perf.snapshot",
        tuple(ChainNode.from_observation(row) for row in observations),
        complete_chain_history=True,
    )
    graph = ChainGraph.from_snapshot(graph_snapshot)
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        for index in range(0, len(observations), max(1, len(observations) // 20)):
            row = observations[index]
            uuid_value = row.field("uuid").value
            chain_value = row.field("chainID").value
            link_value = row.field("link").value
            if not snapshot.uuid_matches(str(getattr(uuid_value, "value", uuid_value))):
                raise RuntimeError("snapshot UUID index lost a decoded row")
            if not snapshot.chain_rows(str(getattr(chain_value, "value", chain_value))):
                raise RuntimeError("snapshot chain index lost a decoded row")
            if not snapshot.slot_rows(str(getattr(chain_value, "value", chain_value)), int(getattr(link_value, "value", link_value))):
                raise RuntimeError("snapshot slot index lost a decoded row")
        findings = evaluate_invariants(graph)
        if findings:
            raise RuntimeError(f"snapshot graph reuse produced findings: {findings[0].reason_code}")
        observation_by_uuid = {
            str(getattr(row.field("uuid").value, "value", row.field("uuid").value)): row
            for row in observations
        }
        if any(node.observation is not observation_by_uuid.get(node.task_uuid) for node in graph.nodes):
            raise RuntimeError("chain graph did not retain decoded observations")
        if codec.decode_count != expected_decode_count:
            raise RuntimeError("downstream snapshot consumers decoded a row more than once")
    if any(snapshot.uuid_matches(str(getattr(row.field("uuid").value, "value", "")))[0] is not row for row in observations):
        raise RuntimeError("snapshot indexes did not reuse immutable observations")
    return time.perf_counter() - started


def task_resource_limits(task_codec: Any, rounds: int) -> float:
    """Measure bounded nested freezing and the existing protocol size guard."""
    from nautical_core.hook_protocol import MAX_JSON_BYTES, probe_on_add

    nested = {"level": [{"value": "x" * 64, "items": [index, index + 1]} for index in range(32)]}
    row = {
        "uuid": "00000000-0000-4000-8000-000000000004", "description": "resource limit benchmark",
        "status": "pending", "chainID": "resource-perf", "link": 1, "nested": nested,
    }
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) >= MAX_JSON_BYTES:
        raise RuntimeError("resource benchmark fixture unexpectedly exceeds the protocol limit")
    observation = task_codec.DEFAULT_TASK_CODEC.decode_row(row, source_query="perf:resource-limits")
    oversize = b"{" + b'"description":"' + b"x" * MAX_JSON_BYTES + b'"}'
    if probe_on_add(oversize, max_bytes=MAX_JSON_BYTES).failure is None:
        raise RuntimeError("protocol accepted an oversized hook payload")
    started = time.perf_counter()
    for _ in range(max(1, int(rounds))):
        if not observation.arbitrary.get("nested"):
            raise RuntimeError("bounded nested arbitrary field was lost")
        if len(encoded.encode("utf-8")) >= MAX_JSON_BYTES:
            raise RuntimeError("resource fixture crossed the configured input limit")
    return time.perf_counter() - started


def task_snapshot_memory(task_codec: Any, counts: Sequence[int]) -> float:
    """Measure peak memory while decoding and indexing bounded snapshots."""
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult
    from nautical_core.task_read_repository import AuthoritativeTaskSnapshot, TaskQueryKind, TaskSnapshotScope

    command = TaskCommand(("task", "export"), "perf snapshot memory", 1.0)
    result = TaskCommandResult(command, 0, "[]", "", CommandFailureKind.SUCCESS, 1, 0.0)
    measurements: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for requested in counts:
        row_count = max(1, int(requested))
        rows = [
            {
                "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"nautical-perf/memory/{index}")),
                "description": f"memory row {index}", "status": "pending", "chain": "on",
                "chainID": f"memory-chain-{index // 10}", "link": (index % 10) + 1,
                "anchor": "w:mon", "due": "20260824T090000Z",
            }
            for index in range(row_count)
        ]
        tracemalloc.start()
        observations = tuple(task_codec.DEFAULT_TASK_CODEC.decode_row(row, source_query="perf:memory") for row in rows)
        scope = TaskSnapshotScope(TaskQueryKind.BROAD, f"perf-memory-{row_count}", ("pending",))
        snapshot = AuthoritativeTaskSnapshot(scope, observations, result)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if len(snapshot.rows) != row_count or len(snapshot.by_uuid) != row_count:
            raise RuntimeError("memory benchmark snapshot was truncated")
        measurements[str(row_count)] = {"current_bytes": int(current), "peak_bytes": int(peak)}
    return time.perf_counter() - started
