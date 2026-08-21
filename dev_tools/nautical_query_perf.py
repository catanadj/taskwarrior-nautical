#!/usr/bin/env python3
"""Small performance budget for Nautical's read-only query API.

The default workload is deterministic and does not require a live Taskwarrior
database.  Pass ``--uuid`` to additionally measure a real query invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nautical_core as core
from nautical_core.integration_context import IntegrationAccess
from nautical_core.integration_models import Found
from nautical_core.query_models import OccurrenceQueryRequest
from nautical_core.query_service import OccurrenceQueryService


TASK = {
    "uuid": "00000000-0000-0000-0000-000000000099",
    "chainID": "query-perf",
    "link": 1,
    "description": "query benchmark",
    "anchor": "w:mon..sun@t=04:30,12:30",
    "anchor_mode": "skip",
    "due": "20260824T013000Z",
}
TASK_BATCH = dict(TASK, uuid="00000000-0000-0000-0000-000000000100", link=2)


class _Repository:
    def __init__(self, rows: tuple[dict[str, object], ...] = (TASK,)) -> None:
        self.calls: list[str] = []
        self.rows = rows

    def by_uuid(self, value, **kwargs):
        del kwargs
        self.calls.append("uuid")
        return Found(TASK, f"uuid:{value}")

    def chain_snapshot(self, value, **kwargs):
        del kwargs
        self.calls.append("chain")
        return Found(_Snapshot(self.rows), f"chain:{value}")

    def broad_snapshot(self, **kwargs):
        del kwargs
        self.calls.append("broad")
        return Found(_Snapshot(self.rows), "broad:query")


class _Snapshot:
    def __init__(self, rows: tuple[dict[str, object], ...]) -> None:
        self.rows = rows

    def uuid_matches(self, value: str) -> tuple[dict[str, object], ...]:
        return tuple(row for row in self.rows if str(row.get("uuid")) == value)

    def __iter__(self):
        return iter(self.rows)


def _service() -> OccurrenceQueryService:
    repository = _Repository()
    uow = SimpleNamespace(
        context=SimpleNamespace(
            access=IntegrationAccess.READ_ONLY,
            local_timezone=timezone(timedelta(hours=3)),
            configuration=SimpleNamespace(fingerprint="query-perf"),
        ),
        repository=repository,
    )
    return OccurrenceQueryService(uow, core=core)


def _read_call_baseline() -> dict[str, int]:
    """Measure the repository reads required by each public selector shape."""
    repository = _Repository((TASK, TASK_BATCH))
    uow = SimpleNamespace(
        context=SimpleNamespace(
            access=IntegrationAccess.READ_ONLY,
            local_timezone=timezone(timedelta(hours=3)),
            configuration=SimpleNamespace(fingerprint="query-perf"),
        ),
        repository=repository,
    )
    service = OccurrenceQueryService(uow, core=core)
    requests = {
        "uuid": {"selector": {"uuids": [TASK["uuid"]]}},
        "chain": {"selector": {"chain_id": TASK["chainID"]}},
        "all_active": {"selector": {"all_tasks": True}},
        "batch_uuid": {"selector": {"uuids": [TASK["uuid"], TASK_BATCH["uuid"]]}},
    }
    for payload in requests.values():
        payload.update({"from": "2026-08-24", "to": "2026-08-24", "count": 1})
    counts: dict[str, int] = {}
    for name, payload in requests.items():
        before = len(repository.calls)
        service.query(OccurrenceQueryRequest.from_mapping(payload))
        counts[name] = len(repository.calls) - before
    # The synthetic repository represents one Taskwarrior export per read.
    return counts


def _request(operation: str = "occurrences") -> OccurrenceQueryRequest:
    return OccurrenceQueryRequest.from_mapping(
        {
            "operation": operation,
            "selector": {"uuids": [TASK["uuid"]]},
            "from": "2026-08-24",
            "to": "2026-08-24",
            "count": 2,
        }
    )


def _samples(fn, count: int) -> list[float]:
    values = []
    for _ in range(max(1, count)):
        started = time.perf_counter()
        fn()
        values.append(time.perf_counter() - started)
    return values


def _summary(values: list[float]) -> dict[str, object]:
    ordered = sorted(values)
    return {
        "samples_s": ordered,
        "min_s": ordered[0],
        "median_s": statistics.median(ordered),
        "max_s": ordered[-1],
    }


def _real_query(uuid_value: str, taskdata: str | None) -> dict[str, object]:
    env = dict(os.environ)
    if taskdata:
        env["TASKDATA"] = str(Path(taskdata).expanduser())
    command = [
        sys.executable,
        str(ROOT / "nautical"),
        "query",
        "occurrences",
        "--uuid",
        uuid_value,
        "--from",
        "2026-08-24",
        "--count",
        "1",
    ]
    started = time.perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, env=env, check=False)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_s": elapsed,
        "exit_code": result.returncode,
        "status": (json.loads(result.stdout).get("status") if result.stdout.strip() else "no_json"),
        "stderr": result.stderr.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Nautical's local query API")
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--uuid", help="also benchmark one real Taskwarrior UUID")
    parser.add_argument("--taskdata", help="TASKDATA directory for the real query")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)
    service = _service()
    occurrence_request = _request()
    next_request = _request("next")
    results: dict[str, object] = {
        "capability_process_cold": _summary(
            _samples(
                lambda: subprocess.run(
                    [sys.executable, str(ROOT / "nautical"), "query", "capabilities"],
                    cwd=str(ROOT),
                    text=True,
                    capture_output=True,
                    check=False,
                ),
                args.samples,
            )
        ),
        "service_occurrences_warm": _summary(
            _samples(lambda: service.query(occurrence_request), args.samples)
        ),
        "service_next_warm": _summary(
            _samples(lambda: service.query_next(next_request), args.samples)
        ),
        "repository_read_calls": _read_call_baseline(),
    }
    if args.uuid:
        results["real_taskwarrior_occurrence"] = _real_query(args.uuid, args.taskdata)
    payload = {
        "schema": "nautical.query.performance",
        "version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "samples": max(1, args.samples),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for name, value in results.items():
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
