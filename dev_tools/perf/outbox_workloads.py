"""Lifecycle/outbox benchmark workloads with explicit dependencies."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any


def schema_hot(lifecycle_outbox: Any, rounds: int) -> float:
    with tempfile.TemporaryDirectory(prefix="nautical-perf-outbox-") as td:
        repository = lifecycle_outbox.LifecycleOutboxRepository(Path(td))
        if not repository.open().ok:
            raise RuntimeError("outbox schema benchmark setup failed")
        started = time.perf_counter()
        for _ in range(rounds):
            if not repository.open().ok:
                raise RuntimeError("outbox schema hot open failed")
        return time.perf_counter() - started


def schema_cold(root: Path, rounds: int) -> float:
    """Measure lifecycle outbox initialization across fresh Python processes."""
    with tempfile.TemporaryDirectory(prefix="nautical-perf-outbox-cold-") as td:
        script = (
            "from pathlib import Path; import sys; "
            "from nautical_core.lifecycle_outbox import LifecycleOutboxRepository; "
            "result = LifecycleOutboxRepository(Path(sys.argv[1])).open(); "
            "raise SystemExit(0 if result.ok else result.reason)"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(root), env.get("PYTHONPATH", "")) if part
        )
        started = time.perf_counter()
        for _ in range(max(1, rounds)):
            proc = subprocess.run(
                [sys.executable, "-c", script, td],
                cwd=str(root),
                env=env,
                text=True,
                capture_output=True,
                timeout=30.0,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"cold outbox initialization failed: {proc.stderr.strip()}")
        return time.perf_counter() - started


def lifecycle_staging(
    lifecycle_outbox: Any,
    init_empty_outbox: Any,
    outbox_lifecycle_fixture: Any,
    workflow_outbox_pending: Any,
) -> float:
    """Measure one guarded lifecycle plan crossing the durable staging boundary."""
    from nautical_core.lifecycle_application import LifecycleApplicationService

    with tempfile.TemporaryDirectory(prefix="nautical-perf-lifecycle-stage-") as td:
        taskdata = Path(td)
        init_empty_outbox(taskdata)
        _parents, plans = outbox_lifecycle_fixture("stage", 0, count=1)
        plans[0] = replace(plans[0], parent_guard=replace(plans[0].parent_guard, modified="20260829T000000Z"))
        repository = lifecycle_outbox.LifecycleOutboxRepository(taskdata)
        service = LifecycleApplicationService(outbox=repository, owner="perf-stage")
        started = time.perf_counter()
        outcome = service.stage(plans[0], configuration_fingerprint="perf-config", schedule_fingerprint="perf-schedule")
        if outcome.kind.value not in {"applied", "staged", "already_applied"}:
            raise RuntimeError(f"lifecycle staging stage returned an unexpected outcome: {outcome!r}")
        pending = workflow_outbox_pending(taskdata)
        if len(pending) != 1 or pending[0].get("state") not in {"ready", "claimed", "retry"}:
            raise RuntimeError(f"lifecycle staging stage did not leave one durable intent: {pending!r}")
        return time.perf_counter() - started
