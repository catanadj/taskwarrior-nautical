"""Lifecycle and durable-outbox golden-test ownership boundary.

The implementation bodies remain in the legacy runner during this staged
extraction.  These wrappers move registration and execution ownership into a
domain collection without duplicating lifecycle helpers.
"""

from __future__ import annotations

import importlib
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import tempfile
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from dev_tools.golden_tests.support import (
    child_payload_from_values,
    build_child_draft_for_test,
    extract_last_json,
    expect,
    find_hook_file,
    load_hook_module,
    metadata_payload_from_values,
    modify_effect,
    plan_from_values,
    task_draft,
    task_observation,
    test_operator_uow,
)
from tests.support.lifecycle_execution import LifecycleExecutionFixture

ROOT = Path(__file__).resolve().parents[2]


def test_lifecycle_configuration_drift_blocks_mutation():
    """A plan persisted under one configuration cannot mutate under another."""
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import LifecycleApplicationService, LifecycleApplicationOutcomeKind

    class _Uow:
        mutation_epoch = 0

    class _Mutations:
        def __init__(self):
            self.calls = 0

        def apply(self, _request):
            self.calls += 1
            raise AssertionError("configuration drift must block mutation")

    parent_uuid = "00000000-0000-4000-8000-000000000851"
    child_uuid = "00000000-0000-4000-8000-000000000852"
    plan = plan_from_values(
        identity=LifecycleIdentity("cfg-drift", parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
        action=LifecycleAction.SPAWN_CHILD,
        parent_guard=ParentGuard("completed", "on", "cfg-drift", 1, "rf-cfg-drift", "20260101T000000Z"),
        child_payload={"uuid": child_uuid, "chainID": "cfg-drift", "link": 2, "prevLink": parent_uuid[:8]},
        parent_patch={"nextLink": child_uuid[:8]},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )
    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td))
        mutations = _Mutations()
        adapter = LifecycleExecutionFixture(mutations)
        service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter, execution=adapter, outbox=outbox, owner="cfg-drift")
        staged = service.stage(plan, configuration_fingerprint="cfg-before", schedule_fingerprint="sch")
        expect(staged.ok, f"configuration-drift plan did not stage: {staged}")
        result = service.drain(limit=1, configuration_fingerprint="cfg-after", schedule_fingerprint="sch")
        expect(len(result.outcomes) == 1, f"expected one drift outcome: {result.outcomes}")
        outcome = result.outcomes[0]
        expect(outcome.kind is LifecycleApplicationOutcomeKind.MANUAL_REVIEW, f"drift was not rejected: {outcome}")
        expect("configuration" in outcome.reason.lower(), f"drift reason was not actionable: {outcome.reason}")
        expect(mutations.calls == 0, "configuration drift reached the mutation gateway")
        _, status = outbox.status()
        expect(status["states"].get("manual_review") == 1, f"drift was not durably recorded: {status}")


def test_lifecycle_application_staging_only_service_rejects_execution():
    """A service without mutation dependencies may stage but not execute."""
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import LifecycleApplicationService, LifecycleApplicationError

    with tempfile.TemporaryDirectory() as td:
        service = LifecycleApplicationService(outbox=_LifecycleOutboxRepository(Path(td)), owner="on-modify")
        guard = ParentGuard("completed", "on", "chain-s12e", 1, "rf1-s12e", "20260101T000000Z")
        identity = LifecycleIdentity("chain-s12e", "00000000-0000-4000-8000-000000000601", 1, 2, LifecycleEvent.COMPLETE)
        plan = plan_from_values(identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard, child_payload={"uuid": "00000000-0000-4000-8000-000000000602", "chainID": "chain-s12e", "link": 2, "prevLink": "00000000"}, parent_patch={"nextLink": "00000000"}, expected_postconditions=("child_present", "parent_linked", "verified"))
        staged = service.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(staged.ok, f"staging-only service failed to stage: {staged}")
        raised = False
        try:
            service.drain(limit=5, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        except LifecycleApplicationError:
            raised = True
        expect(raised, "drain() on staging-only service must raise LifecycleApplicationError")

def test_lifecycle_outbox_two_process_claims_are_exclusive():
    """Two independent drain workers cannot claim the same lifecycle intent."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    parent_uuid = "00000000-0000-4000-8000-000000000911"
    child_uuid = "00000000-0000-4000-8000-000000000912"
    plan = plan_from_values(
        identity=LifecycleIdentity("claim-race", parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
        action=LifecycleAction.SPAWN_CHILD,
        parent_guard=ParentGuard("completed", "on", "claim-race", 1, "rf-claim-race", "20260101T000000Z"),
        child_payload={"uuid": child_uuid, "chainID": "claim-race", "link": 2, "prevLink": parent_uuid[:8]},
        parent_patch={"nextLink": child_uuid[:8]},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )
    worker = (
        "import json, sys; from pathlib import Path; "
        "from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository; "
        "repo = _LifecycleOutboxRepository(Path(sys.argv[1]), connect_timeout=1.0); "
        "result, records = repo.claim_batch(owner=sys.argv[2], lease_seconds=5.0, limit=1); "
        "print(json.dumps({'kind': result.kind.value, 'count': len(records)}), flush=True)"
    )
    with tempfile.TemporaryDirectory(prefix="nautical-claim-race-") as td:
        repo = _LifecycleOutboxRepository(Path(td))
        staged = repo.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(staged.ok, f"claim race fixture did not stage: {staged}")
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", worker, td, f"worker-{idx}"],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for idx in range(2)
        ]
        results = [process.communicate(timeout=10) for process in processes]
        payloads = [json.loads(stdout.strip()) for _process, (stdout, _stderr) in zip(processes, results)]
        expect(sorted(item["count"] for item in payloads) == [0, 1], f"claim race duplicated work: {payloads}")


def test_lifecycle_queue_and_reconcile_claims_are_exclusive():
    """FIFO drain and exact reconcile claims cannot own one intent together."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    parent_uuid = "00000000-0000-4000-8000-000000000921"
    child_uuid = "00000000-0000-4000-8000-000000000922"
    plan = plan_from_values(
        identity=LifecycleIdentity("cross-owner-race", parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
        action=LifecycleAction.SPAWN_CHILD,
        parent_guard=ParentGuard("completed", "on", "cross-owner-race", 1, "rf-cross-owner", "20260101T000000Z"),
        child_payload={"uuid": child_uuid, "chainID": "cross-owner-race", "link": 2, "prevLink": parent_uuid[:8]},
        parent_patch={"nextLink": child_uuid[:8]},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )
    worker = (
        "import json, sys; from pathlib import Path; "
        "from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository; "
        "repo = _LifecycleOutboxRepository(Path(sys.argv[1]), connect_timeout=1.0); "
        "batch = repo.claim_batch(owner=sys.argv[2], lease_seconds=5.0, limit=1) if sys.argv[3] == 'queue' else None; "
        "result = batch[0] if batch is not None else repo.claim_intent(owner=sys.argv[2], lease_seconds=5.0, intent_id=sys.argv[4]); "
        "count = len(batch[1]) if batch is not None else int(result.ok); "
        "print(json.dumps({'kind': result.kind.value, 'count': count}), flush=True)"
    )

    def run_race(modes):
        with tempfile.TemporaryDirectory(prefix="nautical-cross-owner-") as td:
            repo = _LifecycleOutboxRepository(Path(td))
            staged = repo.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            expect(staged.ok and staged.record is not None, f"cross-owner fixture did not stage: {staged}")
            intent_id = staged.record.intent_id
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", worker, td, f"owner-{idx}", mode, intent_id],
                    cwd=ROOT,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                for idx, mode in enumerate(modes)
            ]
            results = [process.communicate(timeout=10) for process in processes]
            payloads = [json.loads(stdout.strip()) for _process, (stdout, _stderr) in zip(processes, results)]
            return payloads

    queue_race = run_race(("queue", "exact"))
    exact_race = run_race(("exact", "exact"))
    expect(sum(item["count"] for item in queue_race) == 1, f"queue/reconcile claim race was not exclusive: {queue_race}")
    expect(sum(item["count"] for item in exact_race) == 1, f"reconcile/reconcile claim race was not exclusive: {exact_race}")


def test_lifecycle_stale_owner_lease_is_reclaimed_by_next_process():
    """An expired owner cannot retain a claim; the next process can recover it."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository, OutboxResultKind

    parent_uuid = "00000000-0000-4000-8000-000000000931"
    child_uuid = "00000000-0000-4000-8000-000000000932"
    plan = plan_from_values(
        identity=LifecycleIdentity("stale-owner", parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
        action=LifecycleAction.SPAWN_CHILD,
        parent_guard=ParentGuard("completed", "on", "stale-owner", 1, "rf-stale-owner", "20260101T000000Z"),
        child_payload={"uuid": child_uuid, "chainID": "stale-owner", "link": 2, "prevLink": parent_uuid[:8]},
        parent_patch={"nextLink": child_uuid[:8]},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )
    with tempfile.TemporaryDirectory(prefix="nautical-stale-owner-") as td:
        repo = _LifecycleOutboxRepository(Path(td))
        staged = repo.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(staged.ok, f"stale-owner fixture did not stage: {staged}")
        first, records = repo.claim_batch(owner="stale-owner", lease_seconds=0.05, limit=1)
        expect(first.ok and len(records) == 1, f"stale owner did not claim fixture: {first}, {records}")
        time.sleep(0.08)
        worker = (
            "import sys; from pathlib import Path; "
            "from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository; "
            "repo = _LifecycleOutboxRepository(Path(sys.argv[1]), connect_timeout=1.0); "
            "result = repo.claim_intent(owner='replacement', lease_seconds=5.0, intent_id=sys.argv[2]); "
            "print(result.kind.value, flush=True); raise SystemExit(0 if result.ok else 1)"
        )
        process = subprocess.run(
            [sys.executable, "-c", worker, td, staged.record.intent_id],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        expect(process.returncode == 0 and process.stdout.strip() == OutboxResultKind.APPLIED.value,
               f"replacement process could not reclaim stale owner: {process.stdout!r} {process.stderr!r}")





_NAMES = (
    "test_on_modify_staged_plan_carries_parent_guard_and_stable_intent_id",
)


def _delegate(name: str):
    def run() -> None:
        legacy = importlib.import_module("dev_tools.nautical_golden_tests")
        getattr(legacy, f"_legacy_{name}")()

    run.__name__ = name
    run.__qualname__ = name
    run.__doc__ = f"Lifecycle domain test delegated to the staged legacy implementation: {name}."
    return run


globals().update({name: _delegate(name) for name in _NAMES})
TESTS = (
    test_lifecycle_configuration_drift_blocks_mutation,
        test_lifecycle_application_staging_only_service_rejects_execution,
    test_lifecycle_outbox_two_process_claims_are_exclusive,
    test_lifecycle_queue_and_reconcile_claims_are_exclusive,
    test_lifecycle_stale_owner_lease_is_reclaimed_by_next_process,
) + tuple(globals()[name] for name in _NAMES)


def test_lifecycle_application_renews_batch_leases_before_mutation():
    """A slow batched import must not proceed to parent linking after expiry."""
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import LifecycleApplicationService, LifecycleApplicationOutcomeKind
    from nautical_core.integration_models import MutationOperation, MutationOutcome, MutationOutcomeKind, MutationPostcondition
    now = [100.0]
    class _Scripted:
        def __init__(self): self.calls = []
        def apply(self, request):
            self.calls.append(request.operation)
            postcondition = {MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED, MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED}.get(request.operation)
            return MutationOutcome(request.operation, MutationOutcomeKind.APPLIED, request.guard, (postcondition,) if postcondition else ())
    class _Uow: mutation_epoch = 0
    class _SlowExecution(LifecycleExecutionFixture):
        def apply_lifecycle_children_unverified(self, requests):
            outcomes = super().apply_lifecycle_children_unverified(requests); now[0] += 2.0; return outcomes
    def make_plan(parent_uuid, child_uuid, chain_id):
        guard = ParentGuard("completed", "on", chain_id, 1, f"rf-{chain_id}", "20260101T000000Z")
        identity = LifecycleIdentity(chain_id, parent_uuid, 1, 2, LifecycleEvent.COMPLETE)
        return plan_from_values(identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard, child_payload={"uuid": child_uuid, "chainID": chain_id, "link": 2, "prevLink": parent_uuid[:8]}, parent_patch={"nextLink": child_uuid[:8]}, expected_postconditions=("child_present", "parent_linked", "verified"))
    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td), clock=lambda: now[0]); mutations = _Scripted(); adapter = _SlowExecution(mutations)
        service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter, execution=adapter, outbox=outbox, owner="slow-batch", lease_seconds=1.0)
        service.stage(make_plan("00000000-0000-4000-8000-000000000601", "00000000-0000-4000-8000-000000000602", "chain-s3a"), configuration_fingerprint="cfg", schedule_fingerprint="sch")
        service.stage(make_plan("00000000-0000-4000-8000-000000000603", "00000000-0000-4000-8000-000000000604", "chain-s3b"), configuration_fingerprint="cfg", schedule_fingerprint="sch")
        result = service.drain(limit=2, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(len(result.outcomes) == 2 and all(item.kind is LifecycleApplicationOutcomeKind.MANUAL_REVIEW for item in result.outcomes), f"expired batch lease was not rejected: {result.outcomes}")
        expect(mutations.calls == [MutationOperation.CHILD_IMPORT, MutationOperation.CHILD_IMPORT], f"parent mutation ran after lease expiry: {mutations.calls}")


TESTS = TESTS + (test_lifecycle_application_renews_batch_leases_before_mutation,)

def test_lifecycle_application_outbox_faults_are_retryable():
    """Outbox persist, claim, and manual-review faults never appear durable."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import LifecycleApplicationService, LifecycleApplicationOutcomeKind

    class _Uow:
        mutation_epoch = 0

    parent_uuid = "00000000-0000-4000-8000-000000000801"
    child_uuid = "00000000-0000-4000-8000-000000000802"
    guard = ParentGuard("completed", "on", "fault-outbox", 1, "rf-fault", "20260101T000000Z")
    plan = plan_from_values(
        identity=LifecycleIdentity("fault-outbox", parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
        action=LifecycleAction.SPAWN_CHILD,
        parent_guard=guard,
        child_payload={"uuid": child_uuid, "chainID": "fault-outbox", "link": 2, "prevLink": parent_uuid[:8]},
        parent_patch={"nextLink": child_uuid[:8]},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )

    class _FailingOutbox(_LifecycleOutboxRepository):
        def enqueue(self, *args, **kwargs):
            raise OSError("disk full")
        def claim_batch(self, **kwargs):
            raise OSError("database locked")

    mutations = LifecycleExecutionFixture(object())
    with tempfile.TemporaryDirectory() as td:
        outbox = _FailingOutbox(Path(td))
        service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=mutations, execution=mutations, outbox=outbox, owner="fault")
        staged = service.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(staged.kind is LifecycleApplicationOutcomeKind.RETRYABLE and "disk full" in staged.reason,
               f"enqueue failure was not retryable: {staged}")
        drained = service.drain(limit=1, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(drained.claim.kind.value == "retryable" and "database locked" in drained.claim.reason,
               f"claim failure was not retryable: {drained.claim}")

    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td))
        record = outbox.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch").record
        service = LifecycleApplicationService(unit_of_work=_Uow(), outbox=outbox, owner="fault")
        outbox.manual_review = lambda **kwargs: (_ for _ in ()).throw(OSError("manual review write failed"))
        review = service._manual_review(record, "simulated invalid intent")
        expect(review.kind is LifecycleApplicationOutcomeKind.RETRYABLE and "manual review write failed" in review.reason,
               f"manual-review persistence failure was not retryable: {review}")


def test_lifecycle_shuffled_process_drains_converge_to_same_outbox_state():
    """Repeated worker processes converge despite shuffled staging order."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

    plans = []
    for index in range(3):
        parent_uuid = f"00000000-0000-4000-8000-00000000094{index}"
        child_uuid = f"00000000-0000-4000-8000-00000000095{index}"
        chain_id = f"shuffle-{index}"
        plans.append(plan_from_values(
            identity=LifecycleIdentity(chain_id, parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", chain_id, 1, f"rf-{chain_id}", "20260101T000000Z"),
            child_payload={"uuid": child_uuid, "chainID": chain_id, "link": 2, "prevLink": parent_uuid[:8]},
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        ))
    worker = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository\n"
        "repo = _LifecycleOutboxRepository(Path(sys.argv[1]))\n"
        "claimed = []\n"
        "while True:\n"
        "    result, records = repo.claim_batch(owner='convergence-worker', lease_seconds=5.0, limit=2)\n"
        "    if not records:\n"
        "        break\n"
        "    for record in records:\n"
        "        claimed.append(record.intent_id)\n"
        "        repo.acknowledge(intent_id=record.intent_id, owner='convergence-worker')\n"
        "_, status = repo.status()\n"
        "print(json.dumps({'claimed': sorted(claimed), 'states': status['states']}, sort_keys=True), flush=True)"
    )

    def run(order):
        with tempfile.TemporaryDirectory(prefix="nautical-convergence-") as td:
            repo = _LifecycleOutboxRepository(Path(td))
            for plan in order:
                staged = repo.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
                expect(staged.ok, f"convergence fixture did not stage: {staged}")
            process = subprocess.run(
                [sys.executable, "-c", worker, td], cwd=ROOT, text=True, capture_output=True, check=False,
            )
            expect(process.returncode == 0, f"convergence worker failed: {process.stderr!r}")
            return json.loads(process.stdout.strip())

    forward = run(plans)
    reverse = run(tuple(reversed(plans)))
    expect(forward == reverse, f"shuffled process drains diverged: {forward} != {reverse}")


TESTS = (
    test_lifecycle_configuration_drift_blocks_mutation,
    test_lifecycle_application_staging_only_service_rejects_execution,
    test_lifecycle_outbox_two_process_claims_are_exclusive,
    test_lifecycle_queue_and_reconcile_claims_are_exclusive,
    test_lifecycle_stale_owner_lease_is_reclaimed_by_next_process,
    test_lifecycle_application_outbox_faults_are_retryable,
    test_lifecycle_shuffled_process_drains_converge_to_same_outbox_state,
) + tuple(globals()[name] for name in _NAMES)

def test_lifecycle_application_idempotency_and_duplicate_staging():
    """Staging the same plan twice is idempotent; draining an already-applied
    intent produces already_applied and draining an empty outbox is a no-op."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import (
        LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard,
    )
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import (
        LifecycleApplicationService, LifecycleApplicationOutcomeKind,
    )
    from nautical_core.integration_models import (
        MutationOperation, MutationOutcome, MutationOutcomeKind, MutationPostcondition,
    )

    class _Scripted:
        def __init__(self, script):
            self.script = list(script)
        def apply(self, request):
            item = self.script.pop(0)
            pc = {MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
                  MutationOperation.PARENT_LINK:  MutationPostcondition.PARENT_LINKED}.get(request.operation)
            return MutationOutcome(request.operation, item, request.guard, (pc,) if item is MutationOutcomeKind.APPLIED and pc else ())

    class _Uow:
        mutation_epoch = 0

    guard = ParentGuard("completed", "on", "chain-s12d", 1, "rf1-s12d", "20260101T000000Z")
    identity = LifecycleIdentity("chain-s12d", "00000000-0000-4000-8000-000000000401", 1, 2, LifecycleEvent.COMPLETE)
    plan = plan_from_values(
        identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard,
        child_payload={"uuid": "00000000-0000-4000-8000-000000000402", "chainID": "chain-s12d", "link": 2, "prevLink": "00000000"},
        parent_patch={"nextLink": "00000000"},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )

    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td))
        mutations = _Scripted([MutationOutcomeKind.APPLIED, MutationOutcomeKind.APPLIED])
        adapter = LifecycleExecutionFixture(mutations)
        service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter, execution=adapter,
                                               outbox=outbox, owner="test")
        r1 = service.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        r2 = service.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(r1.kind is LifecycleApplicationOutcomeKind.APPLIED, f"first stage failed: {r1}")
        expect(r2.kind is LifecycleApplicationOutcomeKind.ALREADY_APPLIED, f"duplicate stage not idempotent: {r2}")
        _, status = outbox.status()
        expect(len(status["records"]) == 1, f"duplicate staging created a second record: {status}")

        # drain once -> applied
        d1 = service.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(d1.outcomes[0].ok, f"first drain failed: {d1.outcomes[0]}")
        # drain again -> empty (acknowledged, not claimed again)
        d2 = service.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(len(d2.outcomes) == 0, f"second drain should find nothing: {d2.outcomes}")


def test_lifecycle_application_execute_staged_targets_exact_intent():
    """execute_staged claims only the named intent and leaves unrelated queued
    work completely untouched."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import (
        LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard,
    )
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import (
        LifecycleApplicationService, LifecycleApplicationOutcomeKind,
    )
    from nautical_core.integration_models import (
        MutationOperation, MutationOutcome, MutationOutcomeKind, MutationPostcondition,
    )

    class _Scripted:
        def __init__(self, script):
            self.script = list(script)
            self.calls = []
        def apply(self, request):
            self.calls.append(request.operation)
            item = self.script.pop(0)
            pc = {MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
                  MutationOperation.PARENT_LINK:  MutationPostcondition.PARENT_LINKED}.get(request.operation)
            return MutationOutcome(request.operation, item, request.guard, (pc,) if item is MutationOutcomeKind.APPLIED and pc else ())

    class _Uow:
        mutation_epoch = 0

    def _plan(parent_uuid, child_uuid, chain_id):
        guard = ParentGuard("completed", "on", chain_id, 1, f"rf1-{chain_id}", "20260101T000000Z")
        identity = LifecycleIdentity(chain_id, parent_uuid, 1, 2, LifecycleEvent.COMPLETE)
        return plan_from_values(
            identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard,
            child_payload={"uuid": child_uuid, "chainID": chain_id, "link": 2, "prevLink": parent_uuid[:8]},
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )

    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td))
        mutations = _Scripted([MutationOutcomeKind.APPLIED, MutationOutcomeKind.APPLIED])
        adapter = LifecycleExecutionFixture(mutations)
        service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter, execution=adapter, outbox=outbox, owner="reconcile")

        other_plan = _plan("00000000-0000-4000-8000-000000000501", "00000000-0000-4000-8000-000000000502", "chain-other-s12")
        my_plan    = _plan("00000000-0000-4000-8000-000000000503", "00000000-0000-4000-8000-000000000504", "chain-mine-s12")

        service.stage(other_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        service.stage(my_plan,   configuration_fingerprint="cfg", schedule_fingerprint="sch")

        outcome = service.execute_staged(my_plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(outcome.kind is LifecycleApplicationOutcomeKind.APPLIED, f"execute_staged failed: {outcome}")
        expect(outcome.intent_id == my_plan.identity.idempotency_key, "wrong intent executed")

        _, status = outbox.status()
        other_row = next(r for r in status["records"] if r["intent_id"] == other_plan.identity.idempotency_key)
        expect(other_row["state"] == "ready" and other_row["stage"] == "planned",
               f"unrelated intent was disturbed: {other_row}")


TESTS = (
    test_lifecycle_configuration_drift_blocks_mutation,
    test_lifecycle_application_staging_only_service_rejects_execution,
    test_lifecycle_outbox_two_process_claims_are_exclusive,
    test_lifecycle_queue_and_reconcile_claims_are_exclusive,
    test_lifecycle_stale_owner_lease_is_reclaimed_by_next_process,
    test_lifecycle_application_outbox_faults_are_retryable,
    test_lifecycle_shuffled_process_drains_converge_to_same_outbox_state,
    test_lifecycle_application_idempotency_and_duplicate_staging,
    test_lifecycle_application_execute_staged_targets_exact_intent,
) + tuple(globals()[name] for name in _NAMES)

def test_lifecycle_application_happy_path_real_stack():
    """stage + drain produces an applied outcome and mutates Taskwarrior state."""
    import json, tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import (
        LifecycleAction, LifecycleDrainStage, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard,
    )
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import (
        LifecycleApplicationService, LifecycleApplicationOutcomeKind,
    )
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService
    from nautical_core.integration_models import (
        Absent, CommandFailureKind, Found, TaskCommand, TaskCommandResult,
    )

    class _Repo:
        def __init__(self, rows):
            self.rows = dict(rows)
            self.set_calls = 0
            self.broad_calls = 0
        def by_uuid(self, u, *, refresh=False):
            r = self.rows.get(str(u).lower())
            if r is None:
                return Absent(f"uuid:{u}", "not found")
            return Found(task_observation(r), f"uuid:{u}")
        def broad_snapshot(self, *, identity, **_kwargs):
            self.broad_calls += 1
            rows = self.rows
            class _Snapshot:
                def uuid_matches(self, value):
                    row = rows.get(str(value).lower())
                    return () if row is None else (task_observation(row),)
            return Found(_Snapshot(), identity)
        def read_uuid_set(self, request):
            from nautical_core.task_set_reads import SetReadResult, SetReadStatus
            self.set_calls += 1
            found = {
                identity: task_observation(self.rows[identity])
                for identity in request.uuids
                if identity in self.rows
            }
            return SetReadResult(
                SetReadStatus.COMPLETE,
                request.uuids,
                found=found,
                absent=tuple(identity for identity in request.uuids if identity not in found),
                complete_for_requested_identities=True,
            )

    class _Client:
        def __init__(self, repo):
            self.repo = repo
        def execute(self, args, *, purpose, timeout, input_text=None, attempts=1):
            args = list(args)
            command = TaskCommand(("task", *args), purpose, timeout, input_text)
            if "import" in args:
                for line in (input_text or "{}").splitlines():
                    row = json.loads(line)
                    self.repo.rows[str(row["uuid"]).lower()] = row
            elif "modify" in args:
                uuid_token = next((a for a in args if a.startswith("uuid:")), "")
                target = uuid_token.split(":", 1)[1].lower() if uuid_token else None
                if target and target in self.repo.rows:
                    for token in args[args.index("modify") + 1:]:
                        k, v = token.split(":", 1)
                        self.repo.rows[target][k] = v
            return TaskCommandResult(command, 0, "", "", CommandFailureKind.SUCCESS, 1, 0.01)

    class _Uow:
        def __init__(self, rows):
            self.repository = _Repo(rows)
            self.client = _Client(self.repository)
            self.mutation_epoch = 0
        def record_mutation(self, *, uncertain=False):
            self.mutation_epoch += 1
            return self.mutation_epoch

    from nautical_core.lifecycle_models import recurrence_fingerprint as _rfp
    parent_uuid = "00000000-0000-4000-8000-000000000101"
    child_uuid  = "00000000-0000-4000-8000-000000000102"
    parent_uuid_2 = "00000000-0000-4000-8000-000000000201"
    child_uuid_2 = "00000000-0000-4000-8000-000000000202"
    parent = {"uuid": parent_uuid, "status": "completed", "chain": "on",
              "chainID": "chain-s12", "link": 1, "modified": "20260101T000000Z", "cp": "1d"}
    parent_2 = {"uuid": parent_uuid_2, "status": "completed", "chain": "on",
                "chainID": "chain-s12-2", "link": 1, "modified": "20260101T000000Z", "cp": "1d"}
    uow = _Uow({parent_uuid: parent, parent_uuid_2: parent_2})
    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td))
        mutations = TaskwarriorMutationService(uow)
        service = LifecycleApplicationService(
            unit_of_work=uow, mutations=mutations, execution=mutations, outbox=outbox, owner="test-s12",
        )
        guard = ParentGuard("completed", "on", "chain-s12", 1, _rfp(parent), "20260101T000000Z")
        identity = LifecycleIdentity("chain-s12", parent_uuid, 1, 2, LifecycleEvent.COMPLETE)
        plan = LifecyclePlan.from_draft(
            identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard,
            draft=task_draft({
                "uuid": child_uuid,
                "description": "child",
                "chainID": "chain-s12",
                "link": 2,
                "prevLink": parent_uuid[:8],
                "status": "pending",
                "chain": "on",
                "cp": "1d",
                "due": "20260824T090000Z",
            }),
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )
        guard_2 = ParentGuard("completed", "on", "chain-s12-2", 1, _rfp(parent_2), "20260101T000000Z")
        identity_2 = LifecycleIdentity("chain-s12-2", parent_uuid_2, 1, 2, LifecycleEvent.COMPLETE)
        plan_2 = LifecyclePlan.from_draft(
            identity=identity_2, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard_2,
            draft=task_draft({
                "uuid": child_uuid_2,
                "description": "child 2",
                "chainID": "chain-s12-2",
                "link": 2,
                "prevLink": parent_uuid_2[:8],
                "status": "pending",
                "chain": "on",
                "cp": "1d",
                "due": "20260824T100000Z",
            }),
            parent_patch={"nextLink": child_uuid_2[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )
        staged = service.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(staged.ok, f"stage failed: {staged}")
        staged_2 = service.stage(plan_2, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(staged_2.ok, f"second stage failed: {staged_2}")
        progress = []
        result = service.drain(
            limit=10,
            configuration_fingerprint="cfg",
            schedule_fingerprint="sch",
            progress=progress.append,
        )
        expect(len(result.outcomes) == 2, f"expected 2 outcomes: {result.outcomes}")
        expect(all(item.kind is LifecycleApplicationOutcomeKind.APPLIED for item in result.outcomes), f"outcomes: {result.outcomes}")
        expect(progress[0].stage is LifecycleDrainStage.CLAIMED, f"missing claimed progress: {progress}")
        expect(progress[-1].stage is LifecycleDrainStage.COMPLETE, f"missing final progress: {progress}")
        processing = [event.completed for event in progress if event.stage is LifecycleDrainStage.PROCESSING]
        expect(processing == list(range(1, 13)), f"drain did not advance per lifecycle action: {progress}")
        expect(progress[-1].completed == 12 and progress[-1].total == 12, f"invalid final progress: {progress[-1]}")
        expect(child_uuid.lower() in uow.repository.rows, "child was not imported into task store")
        expect(child_uuid_2.lower() in uow.repository.rows, "second child was not imported into task store")
        expect(uow.repository.rows[parent_uuid]["nextLink"] == child_uuid[:8], "parent nextLink not set")
        expect(uow.repository.rows[parent_uuid_2]["nextLink"] == child_uuid_2[:8], "second parent nextLink not set")
        expect(
            uow.repository.set_calls == 3,
            "multi-plan drain must use exactly one set read per authoritative phase "
            f"(preflight, child verification, parent verification), got {uow.repository.set_calls}",
        )
        expect(uow.repository.broad_calls == 0, f"drain used broad history exports: {uow.repository.broad_calls}")
def test_lifecycle_application_crash_at_each_stage_resumes_without_remutation():
    """A crash at each stage boundary resumes from the correct next step."""
    from nautical_core.lifecycle_models import LifecycleAction, LifecycleEvent, LifecycleIdentity, LifecyclePlan, ParentGuard, ExecutionStage
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import LifecycleApplicationService, LifecycleApplicationOutcomeKind
    from nautical_core.integration_models import MutationOperation, MutationOutcome, MutationOutcomeKind, MutationPostcondition, FailureEvidence, CommandFailureKind, TaskCommand

    class _Scripted:
        def __init__(self, script):
            self.script = list(script)
            self.calls = []
        def apply(self, request):
            self.calls.append(request.operation)
            if not self.script:
                raise AssertionError(f"unexpected mutation call: {request.operation}")
            item = self.script.pop(0)
            pc = {MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
                  MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED}.get(request.operation)
            if item is MutationOutcomeKind.RETRYABLE:
                evidence = FailureEvidence(TaskCommand(("task", "modify"), "test mutation", 5.0), CommandFailureKind.TIMEOUT, -1, 1, 0.1, True, "simulated timeout")
                return MutationOutcome(request.operation, item, request.guard, (), "simulated retryable", evidence)
            return MutationOutcome(request.operation, item, request.guard, (pc,) if item is MutationOutcomeKind.APPLIED and pc else (), "" if item is MutationOutcomeKind.APPLIED else "simulated failure")

    class _Uow:
        mutation_epoch = 0

    def make_plan(parent_uuid, child_uuid):
        guard = ParentGuard("completed", "on", "chain-s12b", 1, "rf1-s12b", "20260101T000000Z")
        identity = LifecycleIdentity("chain-s12b", parent_uuid, 1, 2, LifecycleEvent.COMPLETE)
        return LifecyclePlan.from_draft(identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard,
            draft=task_draft({"uuid": child_uuid, "description": "crash recovery child", "status": "pending", "chain": "on", "chainID": "chain-s12b", "link": 2, "prevLink": parent_uuid[:8], "cp": "1d", "due": "20260102T000000Z"}),
            parent_patch={"nextLink": child_uuid[:8]}, expected_postconditions=("child_present", "parent_linked", "verified"))

    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td)); uow = _Uow()
        m1 = _Scripted([MutationOutcomeKind.APPLIED, MutationOutcomeKind.RETRYABLE]); adapter1 = LifecycleExecutionFixture(m1)
        svc1 = LifecycleApplicationService(unit_of_work=uow, mutations=adapter1, execution=adapter1, outbox=outbox, owner="owner-a", lease_seconds=0.2)
        plan = make_plan("00000000-0000-4000-8000-000000000201", "00000000-0000-4000-8000-000000000202")
        svc1.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        d1 = svc1.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(d1.outcomes[0].kind is LifecycleApplicationOutcomeKind.RETRYABLE, f"expected retryable: {d1.outcomes[0]}")
        _, status = outbox.status(); expect(status["records"][0]["stage"] == "child_present", "stage must be child_present after partial failure")
        time.sleep(0.3)
        m2 = _Scripted([MutationOutcomeKind.APPLIED]); adapter2 = LifecycleExecutionFixture(m2)
        svc2 = LifecycleApplicationService(unit_of_work=uow, mutations=adapter2, execution=adapter2, outbox=outbox, owner="owner-b", lease_seconds=30)
        d2 = svc2.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(d2.outcomes[0].ok, f"resume at link failed: {d2.outcomes[0]}"); expect(m2.calls == [MutationOperation.PARENT_LINK], f"child_import was repeated: {m2.calls}")

    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td)); plan = make_plan("00000000-0000-4000-8000-000000000203", "00000000-0000-4000-8000-000000000204")
        staged = outbox.enqueue(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        outbox.claim_intent(owner="owner-a", lease_seconds=1.0, intent_id=staged.record.intent_id)
        child = outbox.advance_stage(intent_id=staged.record.intent_id, owner="owner-a", stage=ExecutionStage.CHILD_PRESENT)
        parent = outbox.advance_stage(intent_id=staged.record.intent_id, owner="owner-a", stage=ExecutionStage.PARENT_LINKED)
        expect(child.ok and parent.ok, "crash fixture could not persist both completed stages")
        time.sleep(1.1)
        m3 = _Scripted([]); adapter3 = LifecycleExecutionFixture(m3)
        svc3 = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter3, execution=adapter3, outbox=outbox, owner="owner-b", lease_seconds=30)
        d3 = svc3.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(d3.outcomes[0].ok, f"resume at parent_linked should succeed without remutation: {d3.outcomes[0]}"); expect(m3.calls == [], f"unexpected mutations: {m3.calls}")


def test_lifecycle_application_stage_failure_matrix_resumes_idempotently():
    """Each persisted spawn boundary can fail once and resume without unsafe duplication."""
    from nautical_core.lifecycle_models import ExecutionStage, LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository, OutboxResult, OutboxResultKind
    from nautical_core.lifecycle_application import LifecycleApplicationService, LifecycleApplicationOutcomeKind
    from nautical_core.integration_models import MutationOperation, MutationOutcome, MutationOutcomeKind, MutationPostcondition

    class _Uow:
        mutation_epoch = 0

    class _Mutations:
        def __init__(self):
            self.calls = []

        def apply(self, request):
            self.calls.append(request.operation)
            postcondition = {
                MutationOperation.CHILD_IMPORT: MutationPostcondition.CHILD_IMPORTED,
                MutationOperation.PARENT_LINK: MutationPostcondition.PARENT_LINKED,
            }.get(request.operation)
            prior = sum(1 for operation in self.calls if operation is request.operation)
            kind = MutationOutcomeKind.ALREADY_APPLIED if prior > 1 else MutationOutcomeKind.APPLIED
            return MutationOutcome(request.operation, kind, request.guard,
                                   (postcondition,) if postcondition else (),
                                   "already present" if kind is MutationOutcomeKind.ALREADY_APPLIED else "")

    class _FailingOutbox(_LifecycleOutboxRepository):
        def __init__(self, path, *, fail_stage=None, fail_ack=False):
            super().__init__(path)
            self.fail_stage = fail_stage
            self.fail_ack = fail_ack

        def advance_stage(self, *, intent_id, owner, stage):
            if self.fail_stage is stage:
                self.fail_stage = None
                return OutboxResult(OutboxResultKind.RETRYABLE, reason=f"injected {stage.value} persistence failure")
            return super().advance_stage(intent_id=intent_id, owner=owner, stage=stage)

        def acknowledge(self, *, intent_id, owner):
            if self.fail_ack:
                self.fail_ack = False
                return OutboxResult(OutboxResultKind.RETRYABLE, reason="injected acknowledgement failure")
            return super().acknowledge(intent_id=intent_id, owner=owner)

    parent_uuid = "00000000-0000-4000-8000-000000000901"
    child_uuid = "00000000-0000-4000-8000-000000000902"
    plan = plan_from_values(
        identity=LifecycleIdentity("stage-matrix", parent_uuid, 1, 2, LifecycleEvent.COMPLETE),
        action=LifecycleAction.SPAWN_CHILD,
        parent_guard=ParentGuard("completed", "on", "stage-matrix", 1, "rf-stage-matrix", "20260101T000000Z"),
        child_payload={"uuid": child_uuid, "chainID": "stage-matrix", "link": 2, "prevLink": parent_uuid[:8]},
        parent_patch={"nextLink": child_uuid[:8]},
        expected_postconditions=("child_present", "parent_linked", "verified"),
    )

    cases = (("child-stage", ExecutionStage.CHILD_PRESENT, False),
             ("parent-stage", ExecutionStage.PARENT_LINKED, False),
             ("verified-stage", ExecutionStage.VERIFIED, False),
             ("acknowledgement", None, True))
    for label, fail_stage, fail_ack in cases:
        with tempfile.TemporaryDirectory(prefix=f"nautical-stage-{label}-") as td:
            outbox = _FailingOutbox(Path(td), fail_stage=fail_stage, fail_ack=fail_ack)
            mutations = _Mutations()
            adapter = LifecycleExecutionFixture(mutations)
            service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter, execution=adapter,
                                                  outbox=outbox, owner=f"stage-{label}", lease_seconds=1.0)
            staged = service.stage(plan, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            expect(staged.ok, f"{label}: staging failed: {staged}")
            first = service.drain(limit=1, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            expect(first.outcomes and first.outcomes[0].kind is LifecycleApplicationOutcomeKind.RETRYABLE,
                   f"{label}: injected failure was not retryable: {first.outcomes}")
            time.sleep(1.05)
            second = service.drain(limit=1, configuration_fingerprint="cfg", schedule_fingerprint="sch")
            if not second.outcomes:
                _, retry_status = outbox.status()
                raise AssertionError(f"{label}: retry did not claim the intent: {retry_status}")
            expect(second.outcomes[0].kind is LifecycleApplicationOutcomeKind.APPLIED,
                   f"{label}: retry did not converge: {second.outcomes}")
            _, status = outbox.status()
            expect(status["states"].get("acknowledged") == 1, f"{label}: intent was not acknowledged: {status}")
            expect(mutations.calls.count(MutationOperation.CHILD_IMPORT) <= 2,
                   f"{label}: child mutation was repeated unsafely: {mutations.calls}")


def test_lifecycle_application_conflict_and_retry_budget_outcomes():
    """Conflicts surface as manual_review; retryable failures that exhaust the
    budget quarantine the intent rather than looping."""
    import tempfile
    from pathlib import Path
    from nautical_core.lifecycle_models import (
        LifecycleAction, LifecycleEvent, LifecycleIdentity, ParentGuard,
    )
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.lifecycle_application import (
        LifecycleApplicationService, LifecycleApplicationOutcomeKind,
    )
    from nautical_core.integration_models import (
        MutationOutcome, MutationOutcomeKind,
    )

    from nautical_core.integration_models import FailureEvidence, CommandFailureKind, TaskCommand as _TC_s12c

    class _Scripted:
        def __init__(self, script):
            self.script = list(script)
        def apply(self, request):
            item = self.script.pop(0)
            if item is MutationOutcomeKind.RETRYABLE:
                cmd = _TC_s12c(("task", "modify"), "test", 5.0)
                ev = FailureEvidence(cmd, CommandFailureKind.TIMEOUT, -1, 1, 0.1, True, "simulated timeout")
                return MutationOutcome(request.operation, item, request.guard, (), "simulated", ev)
            return MutationOutcome(request.operation, item, request.guard, (), "simulated" if item is not MutationOutcomeKind.APPLIED else "")

    class _Uow:
        mutation_epoch = 0

    def _plan(parent_uuid, child_uuid, max_attempts=3):
        guard = ParentGuard("completed", "on", "chain-s12c", 1, "rf1-s12c", "20260101T000000Z")
        identity = LifecycleIdentity("chain-s12c", parent_uuid, 1, 2, LifecycleEvent.COMPLETE)
        return plan_from_values(
            identity=identity, action=LifecycleAction.SPAWN_CHILD, parent_guard=guard,
            child_payload={"uuid": child_uuid, "chainID": "chain-s12c", "link": 2, "prevLink": parent_uuid[:8]},
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
            max_attempts=max_attempts,
        )

    # Conflict -> manual_review, durably recorded
    with tempfile.TemporaryDirectory() as td:
        outbox = _LifecycleOutboxRepository(Path(td))
        mutations = _Scripted([MutationOutcomeKind.CONFLICT])
        adapter = LifecycleExecutionFixture(mutations)
        service = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter, execution=adapter,
                                               outbox=outbox, owner="test")
        p = _plan("00000000-0000-4000-8000-000000000301", "00000000-0000-4000-8000-000000000302")
        service.stage(p, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        result = service.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(result.outcomes[0].kind is LifecycleApplicationOutcomeKind.MANUAL_REVIEW,
               f"conflict should surface as manual_review: {result.outcomes[0]}")
        _, status = outbox.status()
        expect(status["states"].get("manual_review") == 1, f"conflict was not durably recorded: {status}")

    # Retryable at budget exhaustion -> quarantined, not infinite loop
    with tempfile.TemporaryDirectory() as td:
        outbox2 = _LifecycleOutboxRepository(Path(td))
        mutations2 = _Scripted([MutationOutcomeKind.RETRYABLE])
        adapter2 = LifecycleExecutionFixture(mutations2)
        service2 = LifecycleApplicationService(unit_of_work=_Uow(), mutations=adapter2, execution=adapter2,
                                                outbox=outbox2, owner="test")
        p2 = _plan("00000000-0000-4000-8000-000000000303", "00000000-0000-4000-8000-000000000304", max_attempts=1)
        service2.stage(p2, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        result2 = service2.drain(limit=10, configuration_fingerprint="cfg", schedule_fingerprint="sch")
        expect(result2.outcomes[0].kind is LifecycleApplicationOutcomeKind.QUARANTINED,
               f"budget exhaustion should quarantine: {result2.outcomes[0]}")
        _, status2 = outbox2.status()
        expect(status2["states"].get("quarantined") == 1, f"record was not quarantined: {status2}")


TESTS = (
    test_lifecycle_configuration_drift_blocks_mutation,
    test_lifecycle_application_staging_only_service_rejects_execution,
    test_lifecycle_outbox_two_process_claims_are_exclusive,
    test_lifecycle_queue_and_reconcile_claims_are_exclusive,
    test_lifecycle_stale_owner_lease_is_reclaimed_by_next_process,
    test_lifecycle_application_outbox_faults_are_retryable,
    test_lifecycle_shuffled_process_drains_converge_to_same_outbox_state,
    test_lifecycle_application_idempotency_and_duplicate_staging,
    test_lifecycle_application_execute_staged_targets_exact_intent,
    test_lifecycle_application_conflict_and_retry_budget_outcomes,
    test_lifecycle_application_renews_batch_leases_before_mutation,
    test_lifecycle_application_stage_failure_matrix_resumes_idempotently,
    test_lifecycle_application_crash_at_each_stage_resumes_without_remutation,
    test_lifecycle_application_happy_path_real_stack,
) + tuple(globals()[name] for name in _NAMES)
def test_taskwarrior_mutation_service_is_guarded_idempotent_and_fail_closed():
    """Named mutations re-read, verify, classify replay, and preserve failures."""
    from nautical_core.integration_models import (
        Absent,
        ChainDisablePayload,
        ChildCompensationPayload,
        CommandFailureKind,
        FailureEvidence,
        Found,
        GuardTimestamp,
        GuardTimestampField,
        MutationGuard,
        MutationOperation,
        MutationOutcomeKind,
        MutationRequest,
        NativeUntilRepairPayload,
        ParentLinkClearPayload,
        ParentLinkPayload,
        TaskCommand,
        TaskCommandResult,
        Unavailable,
    )
    from nautical_core.lifecycle_models import recurrence_fingerprint
    from nautical_core.task_codec import DEFAULT_TASK_CODEC
    from nautical_core.taskwarrior_mutations import TaskwarriorMutationService

    parent_uuid = "00000000-0000-4000-8000-000000000924"
    child_uuid = "00000000-0000-4000-8000-000000000925"
    parent = {
        "uuid": parent_uuid,
        "status": "completed",
        "chain": "on",
        "chainID": "chain-service",
        "link": 7,
        "modified": "20260813T100000Z",
        "anchor": "w:mon",
        "cp": "1d",
    }

    class Repo:
        def __init__(self):
            self.rows = {parent_uuid: parent}
            self.unavailable = False

        def by_uuid(self, uuid_value, *, refresh=False):
            del refresh
            command = TaskCommand(("task", "export"), "test read", 1.0)
            if self.unavailable:
                evidence = FailureEvidence(command, CommandFailureKind.BUSY, 1, 1, 0.01, True, "lock active")
                return Unavailable(f"uuid:{uuid_value}", evidence)
            row = self.rows.get(str(uuid_value).lower())
            if row is None:
                return Absent(f"uuid:{uuid_value}", "not present")
            return Found(
                DEFAULT_TASK_CODEC.decode_row(row, source_query=f"uuid:{uuid_value}"),
                f"uuid:{uuid_value}",
            )

        def exact_child_slot(
            self,
            chain_id,
            link,
            *,
            statuses=(),
            expected_prev_link="",
            complete_chain_history=False,
            refresh=False,
        ):
            del complete_chain_history, refresh
            wanted_statuses = {str(value).lower() for value in statuses}
            for row in self.rows.values():
                if str(row.get("chainID") or "") != str(chain_id):
                    continue
                try:
                    if int(float(row.get("link"))) != int(link):
                        continue
                except (TypeError, ValueError):
                    continue
                if wanted_statuses and str(row.get("status") or "").lower() not in wanted_statuses:
                    continue
                if expected_prev_link and str(row.get("prevLink") or "") != str(expected_prev_link):
                    continue
                return Found(
                    DEFAULT_TASK_CODEC.decode_row(row, source_query=f"chainID:{chain_id} link:{link}"),
                    f"chainID:{chain_id} link:{link}",
                )
            return Absent(f"chainID:{chain_id} link:{link}", "not present")

    class Client:
        def __init__(self, repo):
            self.repo = repo
            self.calls = []

        def execute(self, args, *, purpose, timeout, input_text=None, attempts=1):
            del attempts
            args = list(args)
            self.calls.append((args, purpose))
            command = TaskCommand(("task", *args), purpose, timeout, input_text)
            if "import" in args:
                for line in (input_text or "{}").splitlines():
                    row = json.loads(line)
                    self.repo.rows[str(row["uuid"]).lower()] = row
            elif "delete" in args:
                uuid_token = next((item for item in args if item.startswith("uuid:")), "")
                self.repo.rows.pop(uuid_token.split(":", 1)[1].lower(), None)
            else:
                update_at = args.index("modify") + 1
                uuid_token = next((item for item in args if item.startswith("uuid:")), "")
                target_uuid = uuid_token.split(":", 1)[1].lower() if uuid_token else parent_uuid
                target = self.repo.rows[target_uuid]
                for token in args[update_at:]:
                    key, value = token.split(":", 1)
                    if key == "until" and "-" in value:
                        value = datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%Y%m%dT%H%M%SZ")
                    target[key] = value
            return TaskCommandResult(command, 0, "", "", CommandFailureKind.SUCCESS, 1, 0.01)

    class Uow:
        def __init__(self):
            self.repository = Repo()
            self.client = Client(self.repository)
            self.mutation_epoch = 0
            self.context = type("Context", (), {"mutation_capable": True})()

        def record_mutation(self, *, uncertain=False):
            del uncertain
            self.mutation_epoch += 1
            return self.mutation_epoch

    uow = Uow()

    def request(operation, payload, epoch, *, chain="on"):
        return MutationRequest(
            operation,
            MutationGuard(
                parent_uuid,
                "completed",
                "chain-service",
                7,
                recurrence_fingerprint(parent),
                (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
                epoch,
                chain,
            ),
            payload,
        )

    service = TaskwarriorMutationService(uow)
    uow.repository.rows.pop(parent_uuid)
    deleted = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 0))
    expect(deleted.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE}, f"deleted parent was applied: {deleted}")
    expect(not uow.client.calls, "deleted parent reached the mutation command")
    uow.repository.rows[parent_uuid] = parent

    child = child_payload_from_values(
        {
            "uuid": child_uuid,
            "chainID": "chain-service",
            "link": 8,
            "prevLink": parent_uuid[:8],
            "description": "service child",
            "status": "pending",
            "chain": "on",
            "modified": "20260813T100000Z",
            "anchor": "w:mon",
            "cp": "1d",
        },
        parent_uuid=parent_uuid,
    )
    imported = service.apply(request(MutationOperation.CHILD_IMPORT, child, 0))
    expect(imported.kind is MutationOutcomeKind.APPLIED, f"child import was not applied: {imported}")
    uow.repository.rows[child_uuid]["link"] = 8.0
    replay = service.apply(request(MutationOperation.CHILD_IMPORT, child, 1))
    expect(replay.kind is MutationOutcomeKind.ALREADY_APPLIED, f"numeric child link replay was not normalized: {replay}")
    link_payload = ParentLinkPayload(parent_uuid, child_uuid[:8])

    # Child identity replacement race: an existing UUID with changed chain
    # identity or UUID payload is not treated as the requested child.
    for field, value in (("uuid", "00000000-0000-4000-8000-000000000926"), ("chainID", "user-child-chain"), ("link", 99), ("prevLink", "user-edit")):
        original = uow.repository.rows[child_uuid].get(field)
        uow.repository.rows[child_uuid][field] = value
        calls_before = len(uow.client.calls)
        raced_child_identity = service.apply(request(MutationOperation.CHILD_IMPORT, child, 1))
        expect(raced_child_identity.kind in {
            MutationOutcomeKind.CONFLICT,
            MutationOutcomeKind.RETRYABLE,
            MutationOutcomeKind.MANUAL_REVIEW,
        },
               f"user edit of child identity {field} was not rejected: {raced_child_identity}")
        expect(len(uow.client.calls) == calls_before, f"child identity {field} race reached Taskwarrior")
        if original is None:
            uow.repository.rows[child_uuid].pop(field, None)
        else:
            uow.repository.rows[child_uuid][field] = original

    linked = service.apply(request(MutationOperation.PARENT_LINK, link_payload, 1))
    expect(linked.kind is MutationOutcomeKind.APPLIED, f"parent link was not applied: {linked}")
    # Taskwarrior updates ``modified`` when the link succeeds.  Recovery can
    # therefore see the desired nextLink with a newer timestamp before the
    # outbox stage was persisted; that state must converge idempotently.
    parent["modified"] = "20260813T100001Z"
    recovered = service.apply(
        MutationRequest(
            MutationOperation.PARENT_LINK,
            MutationGuard(
                parent_uuid,
                "completed",
                "chain-service",
                7,
                recurrence_fingerprint(parent),
                (GuardTimestamp(GuardTimestampField.MODIFIED, "20260813T100000Z"),),
                2,
                "on",
            ),
            link_payload,
        )
    )
    expect(
        recovered.kind is MutationOutcomeKind.ALREADY_APPLIED,
        f"parent link recovery did not converge after modified changed: {recovered}",
    )
    cleared = service.apply(
        MutationRequest(
            MutationOperation.PARENT_LINK_CLEAR,
            MutationGuard(
                parent_uuid,
                "completed",
                "chain-service",
                7,
                recurrence_fingerprint(parent),
                (GuardTimestamp(GuardTimestampField.MODIFIED, parent["modified"]),),
                2,
                "on",
            ),
            ParentLinkClearPayload(parent_uuid, child_uuid[:8]),
        )
    )
    expect(cleared.kind is MutationOutcomeKind.APPLIED, f"parent link clear was not applied: {cleared}")
    disabled = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 3))
    expect(disabled.kind is MutationOutcomeKind.APPLIED, f"chain disablement was not applied: {disabled}")
    replay = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 4))
    expect(replay.kind is MutationOutcomeKind.ALREADY_APPLIED, f"chain replay was not idempotent: {replay}")
    parent["until"] = "20260813T200000Z"
    native = service.apply(
        request(
            MutationOperation.NATIVE_UNTIL_REPAIR,
            NativeUntilRepairPayload(parent_uuid, parent["until"], "20260814T200000Z"),
            4,
            chain="off",
        )
    )
    expect(native.kind is MutationOutcomeKind.APPLIED, f"native-until repair was not applied: {native}")
    metadata = service.apply(
        request(
            MutationOperation.METADATA_REPAIR,
            metadata_payload_from_values(parent_uuid, {"chainMax": "5"}, expected={"chainMax": ""}),
            5,
            chain="off",
        )
    )
    expect(metadata.kind is MutationOutcomeKind.APPLIED, f"metadata repair was not applied: {metadata}")
    child_row = uow.repository.rows[child_uuid]
    child_guard = MutationGuard(
        child_uuid,
        "pending",
        "chain-service",
        8,
        recurrence_fingerprint(child_row),
        (GuardTimestamp(GuardTimestampField.MODIFIED, child_row["modified"]),),
        6,
        "on",
    )
    for field, value in (("status", "completed"), ("modified", "20260813T100002Z")):
        original = child_row.get(field)
        child_row[field] = value
        calls_before = len(uow.client.calls)
        raced_child = service.apply(
            MutationRequest(MutationOperation.CHILD_COMPENSATION, child_guard, ChildCompensationPayload(child_uuid))
        )
        expect(raced_child.kind in {MutationOutcomeKind.CONFLICT, MutationOutcomeKind.RETRYABLE},
               f"user edit of child {field} was not rejected: {raced_child}")
        expect(len(uow.client.calls) == calls_before, f"child {field} race reached Taskwarrior")
        if original is None:
            child_row.pop(field, None)
        else:
            child_row[field] = original
    compensated = service.apply(
        MutationRequest(
            MutationOperation.CHILD_COMPENSATION,
            child_guard,
            ChildCompensationPayload(child_uuid),
        )
    )
    expect(compensated.kind is MutationOutcomeKind.APPLIED, f"child compensation was not applied: {compensated}")
    replay_compensation = service.apply(
        MutationRequest(
            MutationOperation.CHILD_COMPENSATION,
            MutationGuard(
                child_uuid,
                "pending",
                "chain-service",
                8,
                child_guard.recurrence_identity,
                child_guard.timestamps,
                7,
                "on",
            ),
            ChildCompensationPayload(child_uuid),
        )
    )
    expect(
        replay_compensation.kind is MutationOutcomeKind.ALREADY_APPLIED,
        f"child compensation replay was not idempotent: {replay_compensation}",
    )
    modify_calls = [args for args, purpose in uow.client.calls if "modify" in args]
    expect(modify_calls, "lifecycle mutation test did not exercise a modify command")
    expect(
        all(args and args[0] == "rc.hooks=off" for args in modify_calls),
        f"lifecycle modify commands must disable hooks: {modify_calls}",
    )
    uow.repository.unavailable = True
    unavailable = service.apply(request(MutationOperation.CHAIN_DISABLE, ChainDisablePayload(parent_uuid), 7))
    expect(unavailable.kind is MutationOutcomeKind.RETRYABLE, f"unavailable guard was not retryable: {unavailable}")
    expect(len(uow.client.calls) == 7, f"unexpected Taskwarrior mutation count: {uow.client.calls}")



def test_lifecycle_outbox_persists_typed_plans_and_recovers_claims():
    """The durable outbox owns immutable plans, leases, stages, and poison rows."""
    import threading

    from nautical_core.lifecycle_models import (
        LifecycleAction,
        LifecycleEvent,
        LifecycleIdentity,
        LifecyclePlan,
        ParentGuard,
    )
    from nautical_core.lifecycle_outbox import (
        _LifecycleOutboxRepository,
        OutboxFailure,
        OutboxProcessingState,
        OutboxResultKind,
    )

    now = [1000.0]

    def clock():
        return now[0]

    def plan_for(
        link: int,
        *,
        legacy_null_anchor_file: bool = False,
        child_entry: str = "",
        child_description: str = "",
        numeric_variant: bool = False,
    ) -> LifecyclePlan:
        parent_uuid = f"00000000-0000-4000-8000-{link:012d}"
        child_uuid = f"10000000-0000-4000-8000-{link:012d}"
        child_payload = {
            "uuid": child_uuid,
            "description": child_description or "outbox child",
            "status": "pending",
            "chain": "on",
            "chainID": "outbox-chain",
            "link": link + 1,
            "prevLink": parent_uuid[:8],
            "cp": "1d",
            "due": "20260824T090000Z",
            "numeric_metadata": {"slot": link + 1},
        }
        if legacy_null_anchor_file:
            child_payload["anchor_file"] = None
        if numeric_variant:
            child_payload["link"] = float(link + 1)
            child_payload["numeric_metadata"] = {"slot": float(link + 1)}
        if child_entry:
            child_payload["entry"] = child_entry
        if child_description:
            child_payload["description"] = child_description
        return LifecyclePlan.from_draft(
            identity=LifecycleIdentity("outbox-chain", parent_uuid, link, link + 1, LifecycleEvent.COMPLETE),
            action=LifecycleAction.SPAWN_CHILD,
            parent_guard=ParentGuard("completed", "on", "outbox-chain", link, "rf1-test"),
            draft=task_draft(child_payload),
            parent_patch={"nextLink": child_uuid[:8]},
            expected_postconditions=("child_present", "parent_linked", "verified"),
        )

    with tempfile.TemporaryDirectory() as td:
        repo = _LifecycleOutboxRepository(Path(td), clock=clock)
        plan = plan_for(1)
        first = repo.enqueue(plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(first.kind is OutboxResultKind.APPLIED, f"outbox enqueue failed: {first}")
        # Verify durable plan decoding and claiming from a fresh interpreter,
        # rather than only reopening the repository in-process.
        restart_plan = plan_for(30)
        restart = repo.enqueue(restart_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(restart.ok, "process-restart lifecycle intent could not be staged")
        restart_script = """
import json
import sys
from pathlib import Path
from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository

repository = _LifecycleOutboxRepository(Path(sys.argv[1]))
result = repository.claim_intent(
    owner="fresh-process",
    lease_seconds=5,
    intent_id=sys.argv[2],
)
if not result.ok or result.record is None:
    raise SystemExit(f"fresh process could not claim lifecycle intent: {result!r}")
record = result.record
print(json.dumps({"semantic_key": record.plan.semantic_key(), "stage": record.stage.value}))
"""
        restart_process = subprocess.run(
            [sys.executable, "-c", restart_script, td, restart_plan.identity.idempotency_key],
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": ROOT},
            capture_output=True,
            text=True,
            check=False,
        )
        expect(
            restart_process.returncode == 0,
            f"fresh lifecycle process failed: {restart_process.stderr!r}",
        )
        try:
            restart_evidence = json.loads(restart_process.stdout)
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"fresh lifecycle process returned invalid evidence: {restart_process.stdout!r}"
            ) from exc
        expect(
            restart_evidence == {"semantic_key": restart_plan.semantic_key(), "stage": "planned"},
            f"fresh lifecycle process changed the persisted plan: {restart_evidence!r}",
        )
        duplicate = repo.enqueue(plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(duplicate.kind is OutboxResultKind.ALREADY_APPLIED, "outbox duplicate enqueue was not idempotent")
        fingerprint_drift = repo.enqueue(plan, configuration_fingerprint="cf2", schedule_fingerprint="sf1")
        expect(fingerprint_drift.kind is OutboxResultKind.ALREADY_APPLIED, "queued intent was blocked by fingerprint drift")
        conflict = repo.enqueue(
            plan_for(1, child_description="different immutable child"),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(conflict.kind is OutboxResultKind.CONFLICT, "outbox accepted divergent immutable intent")

        claimed, records = repo.claim_batch(owner="first-worker", lease_seconds=5, limit=10)
        expect(claimed.ok and len(records) == 1, f"outbox claim failed: {claimed}, {records}")
        intent_id = records[0].intent_id
        child = repo.advance_stage(intent_id=intent_id, owner="first-worker", stage="child_present")
        expect(child.ok, f"outbox child stage failed: {child}")
        released = repo.release_retry(
            intent_id=intent_id,
            owner="first-worker",
            failure=OutboxFailure("task_busy", "Taskwarrior lock active"),
        )
        expect(released.ok, f"outbox retry release failed: {released}")
        now[0] += 1
        reclaimed, records = repo.claim_batch(owner="second-worker", lease_seconds=5, limit=10)
        expect(reclaimed.ok and len(records) == 1, f"outbox retry claim failed: {reclaimed}, {records}")
        record = records[0]
        expect(record.stage.value == "child_present", "outbox retry lost verified child progress")
        expect(record.attempts == 2, "outbox retry did not increment attempts")
        expect(repo.advance_stage(intent_id=intent_id, owner="second-worker", stage="parent_linked").ok, "outbox parent stage failed")
        expect(repo.advance_stage(intent_id=intent_id, owner="second-worker", stage="verified").ok, "outbox verification stage failed")
        expect(repo.acknowledge(intent_id=intent_id, owner="second-worker").ok, "outbox acknowledgement failed")
        replay = repo.enqueue(plan, configuration_fingerprint="new-config", schedule_fingerprint="new-schedule")
        expect(replay.kind is OutboxResultKind.ALREADY_APPLIED, "acknowledged intent was blocked by fingerprint drift")

        legacy = plan_for(2, legacy_null_anchor_file=True)
        expect(
            repo.enqueue(legacy, configuration_fingerprint="cf1", schedule_fingerprint="sf1").kind
            is OutboxResultKind.APPLIED,
            "legacy null lifecycle intent could not be staged",
        )
        converged = repo.enqueue(plan_for(2), configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(
            converged.kind is OutboxResultKind.ALREADY_APPLIED,
            "legacy null lifecycle intent was treated as an immutable conflict",
        )
        old_entry = repo.enqueue(
            plan_for(3, child_entry="20260820T200000Z"),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(old_entry.kind is OutboxResultKind.APPLIED, "volatile-entry lifecycle intent could not be staged")
        new_entry = repo.enqueue(
            plan_for(3, child_entry="20260820T210000Z"),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(new_entry.kind is OutboxResultKind.ALREADY_APPLIED, "entry timestamp caused a lifecycle conflict")
        numeric = repo.enqueue(
            plan_for(11),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(numeric.kind is OutboxResultKind.APPLIED, "numeric lifecycle intent could not be staged")
        numeric_variant = repo.enqueue(
            plan_for(11, numeric_variant=True),
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(
            numeric_variant.kind is OutboxResultKind.ALREADY_APPLIED,
            "numeric JSON representation caused a lifecycle conflict",
        )
        expect(numeric.record is not None, "numeric lifecycle intent lost its durable record")
        numeric_claim = repo.claim_intent(
            owner="numeric-cleanup",
            lease_seconds=5,
            intent_id=numeric.record.intent_id,
        )
        expect(numeric_claim.ok, "numeric lifecycle intent cleanup claim failed")
        expect(
            repo.manual_review(
                intent_id=numeric.record.intent_id,
                owner="numeric-cleanup",
                failure=OutboxFailure("test_cleanup", "numeric representation test complete"),
            ).ok,
            "numeric lifecycle intent cleanup failed",
        )
        manual_plan = plan_for(20)
        manual_staged = repo.enqueue(manual_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(manual_staged.ok and manual_staged.record is not None, "manual intent enqueue failed")
        manual_claim = repo.claim_intent(
            owner="manual-owner", lease_seconds=5, intent_id=manual_staged.record.intent_id
        )
        expect(manual_claim.ok and manual_claim.record is not None, "manual intent claim failed")
        expect(
            repo.manual_review(
                intent_id=manual_staged.record.intent_id,
                owner="manual-owner",
                failure=OutboxFailure("mutation_conflict", "postcondition does not match"),
            ).ok,
            "manual intent setup failed",
        )
        reopened = repo.enqueue(manual_plan, configuration_fingerprint="new-config", schedule_fingerprint="sf1")
        expect(reopened.kind is OutboxResultKind.APPLIED, "known stale postcondition review was not reopened")
        expect(reopened.record is not None and reopened.record.state is OutboxProcessingState.RETRY, "reopened intent was not retryable")
        cleanup_claim = repo.claim_intent(owner="cleanup-owner", lease_seconds=5, intent_id=manual_staged.record.intent_id)
        expect(cleanup_claim.ok, "reopened intent cleanup claim failed")
        for stage in ("child_present", "parent_linked", "verified"):
            expect(
                repo.advance_stage(intent_id=manual_staged.record.intent_id, owner="cleanup-owner", stage=stage).ok,
                f"reopened intent cleanup could not reach {stage}",
            )
        expect(repo.acknowledge(intent_id=manual_staged.record.intent_id, owner="cleanup-owner").ok, "reopened intent cleanup failed")

        rejected_plan = plan_for(22)
        rejected_staged = repo.enqueue(rejected_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1")
        expect(rejected_staged.ok and rejected_staged.record is not None, "rejected intent enqueue failed")
        rejected_claim = repo.claim_intent(
            owner="rejected-owner",
            lease_seconds=5,
            intent_id=rejected_staged.record.intent_id,
        )
        expect(rejected_claim.ok, "rejected intent claim failed")
        expect(
            repo.manual_review(
                intent_id=rejected_staged.record.intent_id,
                owner="rejected-owner",
                failure=OutboxFailure("mutation_rejected", "parent link command failed"),
            ).ok,
            "rejected intent review setup failed",
        )
        reopened_rejected = repo.enqueue(
            rejected_plan,
            configuration_fingerprint="cf1",
            schedule_fingerprint="sf1",
        )
        expect(reopened_rejected.kind is OutboxResultKind.APPLIED, "rejected mutation intent was not reopened")
        expect(reopened_rejected.record is not None, "reopened rejected intent lost its durable record")
        rejected_cleanup = repo.claim_intent(
            owner="rejected-cleanup",
            lease_seconds=5,
            intent_id=rejected_staged.record.intent_id,
        )
        expect(rejected_cleanup.ok, "reopened rejected intent cleanup claim failed")
        expect(
            repo.manual_review(
                intent_id=rejected_staged.record.intent_id,
                owner="rejected-cleanup",
                failure=OutboxFailure("test_cleanup", "rejected mutation test complete"),
            ).ok,
            "reopened rejected intent cleanup failed",
        )

        stage_sequences = (
            (),
            ("child_present",),
            ("child_present", "parent_linked"),
            ("child_present", "parent_linked", "verified"),
        )
        # Use IDs that do not overlap the legacy/numeric fixtures above.
        for link, prior_stages in enumerate(stage_sequences, start=30):
            staged_plan = plan_for(link)
            expect(
                repo.enqueue(staged_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok,
                f"stage recovery enqueue failed for link {link}",
            )
            claimed, records = repo.claim_batch(owner=f"stalled-{link}", lease_seconds=5, limit=1)
            expect(claimed.ok and len(records) == 1, f"stage recovery claim failed for link {link}: {claimed}")
            for stage in prior_stages:
                expect(
                    repo.advance_stage(
                        intent_id=records[0].intent_id,
                        owner=f"stalled-{link}",
                        stage=stage,
                    ).ok,
                    f"stage recovery could not persist {stage}",
                )
            now[0] += 6
            recovered, records = repo.claim_batch(owner=f"recovered-{link}", lease_seconds=5, limit=1)
            expect(recovered.ok and len(records) == 1, f"expired lease was not recovered for link {link}")
            expected_stage = prior_stages[-1] if prior_stages else "planned"
            expect(records[0].stage.value == expected_stage, f"recovery lost {expected_stage} progress")
            expect(
                repo.manual_review(
                    intent_id=records[0].intent_id,
                    owner=f"recovered-{link}",
                    failure=OutboxFailure("test_cleanup", "stage recovery test complete"),
                ).ok,
                f"stage recovery cleanup failed for link {link}",
            )

        concurrent_plan = plan_for(7)
        expect(
            repo.enqueue(concurrent_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok,
            "concurrent claim enqueue failed",
        )
        claim_barrier = threading.Barrier(2)
        concurrent_claims: list[tuple[bool, tuple[object, ...]]] = []
        claims_lock = threading.Lock()

        def claim_once(worker: str) -> None:
            claim_barrier.wait()
            result = _LifecycleOutboxRepository(Path(td), clock=clock).claim_intent(
                owner=worker,
                lease_seconds=5,
                intent_id=concurrent_plan.identity.idempotency_key,
            )
            with claims_lock:
                concurrent_claims.append((result.ok, (result.record,) if result.record is not None else ()))

        workers = [threading.Thread(target=claim_once, args=(f"concurrent-{index}",)) for index in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5)
        claimed_records = [record for ok, records in concurrent_claims if ok for record in records]
        expect(len(concurrent_claims) == 2 and len(claimed_records) == 1, "concurrent outbox claim was not exclusive")

        poison_plan = plan_for(9)
        expect(repo.enqueue(poison_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok, "poison test enqueue failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute("UPDATE lifecycle_outbox SET plan_json='{' WHERE intent_id=?", (poison_plan.identity.idempotency_key,))
        now[0] += 1
        claimed = repo.claim_intent(
            owner="poison-worker", lease_seconds=5,
            intent_id=poison_plan.identity.idempotency_key,
        )
        expect(not claimed.ok and claimed.record is None, f"poison outbox row was claimed: {claimed}")
        status_result, status = repo.status()
        expect(status_result.ok, f"outbox status failed: {status_result}")
        expect(
            status.get("states", {}).get(OutboxProcessingState.QUARANTINED.value) == 1,
            f"poison row was not quarantined: {status}",
        )

        tampered_plan = plan_for(10)
        expect(repo.enqueue(tampered_plan, configuration_fingerprint="cf1", schedule_fingerprint="sf1").ok, "guard test enqueue failed")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute(
                "UPDATE lifecycle_outbox SET parent_guard_json=? WHERE intent_id=?",
                ('{"chain":"off"}', tampered_plan.identity.idempotency_key),
            )
        claimed = repo.claim_intent(
            owner="integrity-worker", lease_seconds=5,
            intent_id=tampered_plan.identity.idempotency_key,
        )
        expect(not claimed.ok and claimed.record is None, f"tampered immutable row was claimed: {claimed}")
        status_result, status = repo.status()
        expect(status_result.ok, f"outbox integrity status failed: {status_result}")
        expect(
            status.get("states", {}).get(OutboxProcessingState.QUARANTINED.value) == 2,
            f"tampered immutable row was not quarantined: {status}",
        )
        reasons = [record.get("reason", "") for record in status.get("records", [])]
        expect(any("parent guard differs" in reason for reason in reasons), f"integrity reason was not preserved: {status}")



def test_lifecycle_outbox_initialization_is_concurrent_and_rejects_unknown_schema():
    """First-open races are bounded, WAL-backed, and never silently downgrade schema."""
    import threading

    from nautical_core.lifecycle_outbox import (
        _LifecycleOutboxRepository,
        OUTBOX_LEGACY_SCHEMA_VERSION,
        OUTBOX_SCHEMA_VERSION,
        OutboxResultKind,
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        worker = (
            "import sys; from pathlib import Path; "
            "from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository; "
            "result = _LifecycleOutboxRepository(Path(sys.argv[1]), connect_timeout=0.5).open(); "
            "print(result.kind.value, flush=True); raise SystemExit(0 if result.ok else 1)"
        )
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", worker, str(root)],
                cwd=ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        process_results = [process.communicate(timeout=10) for process in processes]
        expect(
            all(process.returncode == 0 and stdout.strip() == "applied" for process, (stdout, _stderr) in zip(processes, process_results)),
            f"concurrent process outbox initialization failed: {process_results}",
        )
        barrier = threading.Barrier(4)
        outcomes = []
        outcomes_lock = threading.Lock()

        def open_repository() -> None:
            barrier.wait()
            result = _LifecycleOutboxRepository(root, connect_timeout=0.1).open()
            with outcomes_lock:
                outcomes.append(result)

        threads = [threading.Thread(target=open_repository) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        expect(len(outcomes) == 4 and all(result.ok for result in outcomes), f"concurrent outbox initialization failed: {outcomes}")

        repo = _LifecycleOutboxRepository(root)
        traced_sql: list[str] = []
        original_connect = repo._connect

        def traced_connect():
            conn = original_connect()
            conn.set_trace_callback(traced_sql.append)
            return conn

        repo._connect = traced_connect
        reopened = repo.open()
        expect(reopened.ok, f"reopening an adopted outbox failed: {reopened}")
        expect(
            not any("PRAGMA journal_mode=WAL" in statement for statement in traced_sql),
            f"reopening an adopted outbox renegotiated WAL: {traced_sql!r}",
        )
        with sqlite3.connect(str(repo.path)) as conn:
            journal_mode = str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower()
            expect(journal_mode == "wal", f"outbox did not retain WAL journal mode: {journal_mode}")
            conn.execute("ALTER TABLE lifecycle_outbox DROP COLUMN work_kind")
            conn.execute(f"PRAGMA user_version={OUTBOX_LEGACY_SCHEMA_VERSION}")
        migrated = repo.open()
        expect(migrated.ok, f"legacy outbox schema did not migrate: {migrated}")
        with sqlite3.connect(str(repo.path)) as conn:
            columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(lifecycle_outbox)")}
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        expect("work_kind" in columns and version == OUTBOX_SCHEMA_VERSION, "outbox schema v2 migration incomplete")
        with sqlite3.connect(str(repo.path)) as conn:
            conn.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION + 1}")
        rejected = repo.open()
        expect(rejected.kind is OutboxResultKind.REJECTED, f"future outbox schema was accepted: {rejected}")
        expect("newer than supported" in rejected.reason, f"future schema rejection was not actionable: {rejected}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        path = root / ".nautical-state" / ".nautical_lifecycle_outbox.db"
        path.parent.mkdir()
        path.write_bytes(b"not a sqlite database")
        rejected = _LifecycleOutboxRepository(root).open()
        expect(rejected.kind is OutboxResultKind.REJECTED, f"corrupt outbox database was accepted: {rejected}")



def test_queue_claim_quarantines_poison_rows_and_queue_status_reports_them():
    """Quarantined lifecycle intents remain visible to operator diagnostics."""
    from nautical_core.lifecycle_outbox import _LifecycleOutboxRepository
    from nautical_core.tools import nautical_doctor
    from nautical_core.tools import nautical_queue_status

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        repository = _LifecycleOutboxRepository(root)
        expect(repository.open().ok, "lifecycle outbox did not initialize")
        with sqlite3.connect(str(repository.path)) as conn:
            conn.execute(
                "INSERT INTO lifecycle_outbox "
                "(intent_id, plan_json, plan_fingerprint, parent_guard_json, configuration_fingerprint, "
                "schedule_fingerprint, lifecycle_stage, processing_state, attempts, failure_json, created_at, updated_at) "
                "VALUES (?, '{}', 'poison', '{}', 'cf', 'sf', 'planned', 'quarantined', 1, ?, 1.0, 1.0)",
                ("outbox-poison", json.dumps({"code": "poison_row", "message": "invalid lifecycle plan JSON"})),
            )

        summary, _budget = nautical_queue_status._status_payload(root, stale_after=300.0, limit=5)
        issues = summary.get("issues", [])
        outbox = summary.get("outbox", {})
        expect(outbox.get("states", {}).get("quarantined") == 1, f"outbox status missed quarantined row: {summary!r}")
        expect(
            any("quarantined" in issue for issue in issues),
            f"queue status missed poison issue: {issues!r}",
        )
        findings = []
        nautical_doctor._check_lifecycle_outbox(findings, root, 300.0)
        poison_finding = next((item for item in findings if item.get("id") == "outbox.poison_rows"), None)
        expect(poison_finding and poison_finding.get("severity") == "error", f"doctor missed poison row: {findings!r}")



def test_on_modify_spawn_intent_queue_failure_is_reported():
    """_spawn_child_atomic should report queue failure instead of claiming deferred success."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_spawn_queue_failure_test")
    command_effects = mod._module("modify_command_effects")
    command_effects.reserve_child_uuid = lambda _host, _env: "00000000-0000-4000-8000-00000000abcd"
    spawn_effects = mod._module("modify_spawn_effects")
    original_enqueue = spawn_effects.enqueue_spawn_intent
    spawn_effects.enqueue_spawn_intent = lambda _host, _entry: (False, "queue lock busy")

    spawn_ports = spawn_effects.spawn_child_ports_for(mod)
    child_short, _stripped, verified, deferred, reason, intent = spawn_effects.spawn_child_atomic(spawn_ports,
        {
            "uuid": "00000000-0000-4000-8000-000000000999",
            "description": "x",
            "status": "pending",
            "chainID": "abcd1234",
            "link": 2,
            "cp": "1d",
            "anchor_mode": "skip",
            "due": "20260824T090000Z",
        },
        {
            "uuid": "00000000-0000-4000-8000-000000000111",
            "chainID": "abcd1234",
            "chain": "on",
            "link": 1,
            "status": "completed",
            "nextLink": "",
        },
    )
    spawn_effects.enqueue_spawn_intent = original_enqueue
    expect(len(child_short) == 8 and all(ch in "0123456789abcdef" for ch in child_short.lower()), f"unexpected child short: {child_short}")
    expect(not verified, "verified should be false when queue fails")
    expect(not deferred, "deferred should be false when queue fails")
    expect("queue lock busy" in (reason or ""), f"missing queue failure reason: {reason}")
    expect(bool(intent), "spawn intent id should still be generated")



TESTS = TESTS + (
    test_taskwarrior_mutation_service_is_guarded_idempotent_and_fail_closed,
    test_lifecycle_outbox_persists_typed_plans_and_recovers_claims,
    test_lifecycle_outbox_initialization_is_concurrent_and_rejects_unknown_schema,
    test_queue_claim_quarantines_poison_rows_and_queue_status_reports_them,
    test_on_modify_spawn_intent_queue_failure_is_reported,
)
def test_on_modify_completion_helper_returns_finalized_lifecycle_result():
    """The hook helper must expose the typed result returned by finalization."""
    import nautical_core as core

    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_completion_result_boundary_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    models = core._import_sibling("modify_models")
    expected = models.CompletionLifecycleResult(state="applied", child_short="child123")
    ctx = SimpleNamespace(
        parent_short="parent01",
        base_no=1,
        next_no=2,
        kind="anchor",
        chain_id="chain01",
        chain_snapshot=SimpleNamespace(rows=[], mode="next", loaded=False),
    )
    computed = SimpleNamespace(
        child_due=core.now_utc(),
        meta={"target_field": "due"},
        dnf=[],
        until_dt=None,
        cpmax=0,
        cap_no=None,
        finals=[],
        until_cap_no=None,
    )
    fake_flow = SimpleNamespace(
        CompletionFlowServices=lambda **kwargs: kwargs,
        CompletionFinalizeServices=lambda **kwargs: kwargs,
        finalize_completion_modify=lambda **_kwargs: expected,
        handle_completion_modify=lambda *_args, **_kwargs: expected,
    )
    validation = mod._module("modify_validation_effects")
    original = {
        "validate_cp": validation.validate_cp,
        "preserve_cp": mod._transition_effects.preserve_cp_relative_offsets_on_due_change,
        "preserve_until": mod._transition_effects.preserve_native_until_on_target_change,
        "validate_until": mod._module("modify_validation_effects").validate_native_until,
        "validate_slots": mod._module("modify_validation_effects").validate_native_until_slots,
        "preflight": mod._completion_effects.preflight_context,
        "compute": mod._completion_effects.compute_next_and_limits,
        "import_module": mod.importlib.import_module,
    }
    try:
        validation.validate_cp = lambda *_a, **_k: None
        mod._transition_effects.preserve_cp_relative_offsets_on_due_change = lambda *_a, **_k: None
        mod._transition_effects.preserve_native_until_on_target_change = lambda *_a, **_k: None
        mod._module("modify_validation_effects").validate_native_until = lambda *_a, **_k: None
        mod._module("modify_validation_effects").validate_native_until_slots = lambda *_a, **_k: None
        mod._completion_effects.preflight_context = lambda *_a, **_k: ctx
        mod._completion_effects.compute_next_and_limits = lambda *_a, **_k: computed

        def fake_import(name):
            if name == "nautical_core.modify_completion_flow":
                return fake_flow
            return original["import_module"](name)

        mod.importlib.import_module = fake_import
        result = modify_effect(mod, "handle_completion",
            {"uuid": "parent", "status": "pending"},
            {"uuid": "parent", "status": "completed"},
            test_operator_uow(),
        )
    finally:
        validation.validate_cp = original["validate_cp"]
        mod._transition_effects.preserve_cp_relative_offsets_on_due_change = original["preserve_cp"]
        mod._transition_effects.preserve_native_until_on_target_change = original["preserve_until"]
        mod._module("modify_validation_effects").validate_native_until = original["validate_until"]
        mod._module("modify_validation_effects").validate_native_until_slots = original["validate_slots"]
        mod._completion_effects.preflight_context = original["preflight"]
        mod._completion_effects.compute_next_and_limits = original["compute"]
        mod.importlib.import_module = original["import_module"]

    expect(result is expected, f"completion helper dropped finalized result: {result!r}")

def test_on_modify_completion_build_and_spawn_child_happy_path():
    """completion spawn wrapper should return child info and stamp nextLink when verified."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_completion_spawn_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    new = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "completed",
        "chainID": "abcd1234",
        "link": 1,
        "cp": "P1D",
    }
    child = {"uuid": "00000000-0000-4000-8000-000000000222", "link": 2}
    from nautical_core.chain_generation import ChainGenerationService

    class StubGeneration(ChainGenerationService):
        def build_child_draft(self, parent, child_due, child_field, next_link_no, *_args, **_kwargs):
            return task_draft({
                **child,
                "description": "typed child fixture",
                "chain": "on",
                "status": "pending",
                "chainID": parent.observation.to_mapping()["chainID"],
                "link": next_link_no,
                "cp": "P1D",
                "anchor_mode": "skip",
                child_field: child_due,
            })

    generation_effects = mod._module("modify_generation_effects")
    original_generation = generation_effects.chain_generation_service
    generation_effects.chain_generation_service = lambda _host: StubGeneration.from_core(mod.core)
    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, _child, _parent, **_kwargs: ("beeswax", set(), True, False, None, "si_test")
    try:
        out = mod._completion_effects.build_and_spawn_child(
            new,
            child_due=mod.core.now_utc(),
            child_field="due",
            next_no=2,
            parent_short="00000000",
            kind="cp",
            cpmax=0,
            until_dt=None,
        )
    finally:
        generation_effects.chain_generation_service = original_generation
        spawn_effects.spawn_child_atomic = original_spawn
    expect(bool(out), f"expected spawn result, got {out}")
    expect(out.child.get("uuid") == child["uuid"], f"unexpected child payload: {out}")
    expect(out.child.get("link") == 2, f"typed child lost link: {out}")
    expect(out.child_short == "beeswax", f"unexpected child short: {out}")
    expect(out.verified is True and out.deferred_spawn is False, f"unexpected verification state: {out}")
    expect(out.spawn_intent_id == "si_test", f"unexpected spawn intent id: {out}")
    expect(new.get("nextLink") == "beeswax", f"verified spawn should stamp nextLink: {new}")


def test_on_modify_completion_spawn_exception_is_retryable_with_reason():
    """A spawn command exception must remain typed and actionable for finalization."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_completion_spawn_exception_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent = {
        "uuid": "00000000-0000-4000-8000-000000000121",
        "status": "completed",
        "chainID": "spawn121",
        "link": 1,
        "cp": "P1D",
    }
    child = {"uuid": "00000000-0000-4000-8000-000000000122", "link": 2}
    from nautical_core.chain_generation import ChainGenerationService

    class StubGeneration(ChainGenerationService):
        def build_child_draft(self, parent, child_due, child_field, next_link_no, *_args, **_kwargs):
            return task_draft({
                **child,
                "description": "typed child fixture",
                "chain": "on",
                "status": "pending",
                "chainID": parent.observation.to_mapping()["chainID"],
                "link": next_link_no,
                "cp": "P1D",
                "anchor_mode": "skip",
                child_field: child_due,
            })

    generation_effects = mod._module("modify_generation_effects")
    original_generation = generation_effects.chain_generation_service
    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    original_panel = mod._panel
    original_print = mod._print_task
    panels = []
    try:
        generation_effects.chain_generation_service = lambda _host: StubGeneration.from_core(mod.core)
        spawn_effects.spawn_child_atomic = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Taskwarrior lock busy"))
        mod._panel = lambda title, rows, *, kind=None: panels.append((title, list(rows), kind))
        mod._print_task = lambda _task: None
        result = mod._completion_effects.build_and_spawn_child(
            parent,
            child_due=mod.core.now_utc(),
            child_field="due",
            next_no=2,
            parent_short="00000000",
            kind="cp",
            cpmax=0,
            until_dt=None,
        )
    finally:
        generation_effects.chain_generation_service = original_generation
        spawn_effects.spawn_child_atomic = original_spawn
        mod._panel = original_panel
        mod._print_task = original_print

    expect(result is not None and result.outcome_state == "retryable", f"spawn exception lost typed state: {result!r}")
    expect("Taskwarrior lock busy" in result.reason, f"spawn exception lost reason: {result!r}")
    expect(not panels, f"spawn helper should not render before finalization: {panels!r}")




def test_on_modify_build_child_scheduled_only_keeps_due_unset_and_carries_wait():
    """scheduled-only child spawn should carry relative dates from scheduled."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_build_child_sched_only_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    parent = {
        "uuid": "00000000-0000-4000-8000-000000000333",
        "status": "completed",
        "link": 1,
        "scheduled": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 1), (9, 0))),
        "wait": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 1), (7, 0))),
        "until": mod.core.fmt_isoz(mod.core.build_local_datetime(date(2025, 1, 2), (17, 0))),
        "cp": "1d",
        "chainID": "cid_sched",
    }
    child_due = mod.core.build_local_datetime(date(2025, 1, 2), (9, 0))
    child = build_child_draft_for_test(mod,
        parent,
        child_due,
        "scheduled",
        2,
        "beef",
        "cp",
        0,
        None,
    )
    expect(not child.get("due"), f"scheduled-only child should not get due: {child}")
    expect(child.get("scheduled") == mod.core.fmt_isoz(child_due), f"unexpected child scheduled: {child}")
    wait_local = mod.core.to_local(mod.core.parse_dt_any(child.get("wait")))
    expect((wait_local.hour, wait_local.minute) == (7, 0), f"unexpected carried wait: {wait_local}")
    until_local = mod.core.to_local(mod.core.parse_dt_any(child.get("until")))
    expect(
        until_local.date() == date(2025, 1, 3) and (until_local.hour, until_local.minute) == (17, 0),
        f"unexpected carried until: {until_local}",
    )

def test_on_modify_cp_completion_spawns_next_link():
    """on-modify should spawn the next CP link on completion."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_cp_spawn_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    spawned = {}

    def _spawn_child_atomic_stub(child, parent):
        spawned["child"] = child
        return ("beeswax", set(), False, True, "queued", "si_test3")

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, child, parent, **_kwargs: _spawn_child_atomic_stub(child, parent)
    modify_models = mod._module("modify_models")
    mod._completion_effects.chain_snapshot = lambda chain_id, _base, _next: modify_models.CompletionChainSnapshot(
        mode="next", rows=[], loaded=True, chain_id=str(chain_id)
    )
    mod._completion_effects.existing_next_or_fail = lambda *_a, **_k: True
    # A confirmed empty chain is distinct from an unavailable Taskwarrior
    # export; keep this spawn-path test deterministic and network-free.
    mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export = lambda *_a, **_k: []

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "cp spawn test",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            try:
                mod.main()
            except SystemExit as e:
                raise AssertionError(f"on-modify exited unexpectedly (code={e.code})")
    finally:
        sys.stdin = prev_stdin
        spawn_effects.spawn_child_atomic = original_spawn

    out_task = extract_last_json(buf_out.getvalue())
    expect("child" in spawned, "CP completion did not trigger spawn")
    expect(out_task.get("nextLink") in (None, ""), "CP completion should not set nextLink in decision-only mode")

TESTS = TESTS + (
    test_on_modify_completion_build_and_spawn_child_happy_path,
    test_on_modify_completion_spawn_exception_is_retryable_with_reason,
    test_on_modify_build_child_scheduled_only_keeps_due_unset_and_carries_wait,
    test_on_modify_cp_completion_spawns_next_link,
    test_on_modify_completion_helper_returns_finalized_lifecycle_result,
)


def test_on_modify_completion_chain_snapshot_modes_and_query():
    """Completion presentation modes share one authoritative chain read."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_completion_snapshot_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    from nautical_core.integration_models import Found

    saved = (mod.core.PANEL_MODE, mod._SHOW_ANALYTICS, mod._CHECK_CHAIN_INTEGRITY)
    calls = []
    repository = SimpleNamespace(
        chain_snapshot=lambda chain_id: calls.append(chain_id) or Found(tuple(), "chain snapshot")
    )

    try:
        mod._SHOW_ANALYTICS = False
        mod._CHECK_CHAIN_INTEGRITY = False
        mod.core.PANEL_MODE = "line"
        next_only = mod._completion_effects.chain_snapshot("cid", 5, 6, repository)
        expect(next_only.mode == "next" and next_only.loaded, f"unexpected line snapshot: {next_only}")

        mod.core.PANEL_MODE = "rich"
        recent = mod._completion_effects.chain_snapshot("cid", 5, 6, repository)
        expect(recent.mode == "recent" and recent.loaded, f"unexpected recent snapshot: {recent}")

        mod._CHECK_CHAIN_INTEGRITY = True
        full = mod._completion_effects.chain_snapshot("cid", 5, 6, repository)
        expect(full.mode == "full" and full.loaded, f"unexpected full snapshot: {full}")
        expect(calls == ["cid", "cid", "cid"], f"completion bypassed repository chain reads: {calls!r}")
    finally:
        mod.core.PANEL_MODE, mod._SHOW_ANALYTICS, mod._CHECK_CHAIN_INTEGRITY = saved

def test_on_modify_recompleted_task_with_nextlink_skips_spawn():
    """Re-completing a reactivated task should not spawn when nextLink already exists."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_recomplete_skip_spawn_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    called = {"spawn": False}

    def _spawn_child_atomic_stub(_child, _parent):
        called["spawn"] = True
        return ("beeswax", set(), False, True, "queued", "si_test1")

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, child, parent, **_kwargs: _spawn_child_atomic_stub(child, parent)

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "reactivated duplicate guard",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "nextLink": "beeswax",
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            mod.main()
    finally:
        sys.stdin = prev_stdin
        spawn_effects.spawn_child_atomic = original_spawn

    out_task = extract_last_json(buf_out.getvalue())
    spawn_effects.spawn_child_atomic = original_spawn
    expect(not called["spawn"], "re-completion should not trigger duplicate spawn")
    expect(out_task.get("nextLink") == "beeswax", "existing nextLink should be preserved")

def test_on_modify_recompleted_task_with_existing_link_skips_spawn():
    """Re-completing should not spawn when link #N+1 already exists in chain even if nextLink is empty."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_recomplete_link_guard_test")
    mod._SHOW_TIMELINE_GAPS = False
    mod._SHOW_ANALYTICS = False
    mod._CHECK_CHAIN_INTEGRITY = False

    called = {"spawn": False}

    def _spawn_child_atomic_stub(_child, _parent):
        called["spawn"] = True
        return ("cafebabe", set(), False, True, "queued", "si_test2")

    spawn_effects = mod._module("modify_spawn_effects")
    original_spawn = spawn_effects.spawn_child_atomic
    spawn_effects.spawn_child_atomic = lambda _ports, child, parent, **_kwargs: _spawn_child_atomic_stub(child, parent)
    modify_models = mod._module("modify_models")
    mod._completion_effects.chain_snapshot = lambda chain_id, _base, _next: modify_models.CompletionChainSnapshot(
        mode="recent", rows=[], loaded=False, chain_id=str(chain_id)
    )
    def _existing_next_guard(task, *_args, **_kwargs):
        mod._print_task(task)
        return False

    mod._completion_effects.existing_next_or_fail = _existing_next_guard

    def _get_chain_export_stub(chain_id, since=None, extra=None, env=None):
        if chain_id == "abcd1234" and extra and "link:2" in extra:
            return [
                {
                    "uuid": "00000000-0000-4000-8000-000000000222",
                    "status": "pending",
                    "link": 2,
                    "chainID": "abcd1234",
                }
            ]
        return []

    mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export = _get_chain_export_stub

    old = {
        "uuid": "00000000-0000-4000-8000-000000000111",
        "status": "pending",
        "description": "reactivated duplicate guard via link check",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update(
        {
            "status": "completed",
            "end": "20250102T090000Z",
        }
    )

    raw = json.dumps(old) + "\n" + json.dumps(new) + "\n"
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    buf_in = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
    prev_stdin = sys.stdin
    try:
        sys.stdin = buf_in
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            mod.main()
    finally:
        sys.stdin = prev_stdin
        spawn_effects.spawn_child_atomic = original_spawn

    _ = extract_last_json(buf_out.getvalue())
    expect(not called["spawn"], "existing link #N+1 should prevent duplicate spawn")

def test_on_modify_completion_reuses_single_chain_export_when_chain_needed():
    """on-modify should reuse one full-chain export across preflight and later feedback prep when chain context is needed."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_on_modify_single_chain_export_test")
    if hasattr(mod, "_load_core"):
        mod._load_core()

    mod._SHOW_ANALYTICS = True
    mod._SHOW_TIMELINE_GAPS = False
    mod._CHECK_CHAIN_INTEGRITY = False
    prev_panel_mode = mod.core.PANEL_MODE
    mod.core.PANEL_MODE = "text"

    now_utc = mod.core.now_utc()
    child_due = now_utc + timedelta(days=1)
    export_calls = {"count": 0}
    parent_uuid = "00000000-0000-4000-8000-000000000111"
    child_uuid = "00000000-0000-4000-8000-000000000222"

    chain_rows = [
        {
            "uuid": parent_uuid,
            "status": "completed",
            "description": "cp spawn test",
            "cp": "P1D",
            "chainID": "abcd1234",
            "chain": "on",
            "link": 1,
            "due": "20250101T090000Z",
            "entry": "2025-01-01T09:00:00Z",
            "nextLink": "",
        }
    ]

    modify_models = mod._module("modify_models")
    mod._completion_effects.compute_next_and_limits = lambda *_a, **_k: modify_models.CompletionComputeResult(
        child_due=child_due,
        meta={},
        dnf=None,
        until_dt=None,
        cpmax=0,
        cap_no=None,
        finals=[],
        until_cap_no=None,
    )
    mod._completion_effects.build_and_spawn_child = lambda *_a, **_k: modify_models.CompletionSpawnResult(
        child={
            "uuid": child_uuid,
            "status": "pending",
            "description": "next cp",
            "chainID": "abcd1234",
            "link": 2,
            "prevLink": parent_uuid[:8],
            "due": mod.core.fmt_isoz(child_due),
        },
        child_short=child_uuid[:8],
        stripped_attrs=[],
        verified=True,
        deferred_spawn=False,
        spawn_intent_id=None,
    )
    mod._presentation_effects.render_cp_completion_feedback = lambda **_k: None
    mod._diagnostics_effects.chain_health_advice = lambda *_a, **_k: None
    mod._append_next_wait_sched_rows = lambda *_a, **_k: None
    mod._module("lifecycle_read_service").clear_cached_chain_exports()
    mod._reset_modify_runtime_state()

    old = {
        "uuid": parent_uuid,
        "status": "pending",
        "description": "cp spawn test",
        "cp": "P1D",
        "chainID": "abcd1234",
        "link": 1,
        "due": "20250101T090000Z",
    }
    new = dict(old)
    new.update({"status": "completed", "end": "20250102T090000Z"})

    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult

    uow = test_operator_uow()

    class Client:
        def execute(self, args, *, purpose, timeout, **_kwargs):
            export_calls["count"] += 1
            command = TaskCommand(("task", *args), purpose, timeout)
            return TaskCommandResult(
                command,
                0,
                json.dumps(chain_rows),
                "",
                CommandFailureKind.SUCCESS,
                1,
                0.001,
            )

    uow.client = Client()

    try:
        modify_effect(mod, "handle_completion", old, new, uow)
    finally:
        mod.core.PANEL_MODE = prev_panel_mode

    expect(export_calls["count"] == 1, f"expected one underlying chain export, got {export_calls}")

def test_on_modify_completion_snapshot_reuses_full_chain_read():
    """A completion chain snapshot should satisfy the exact child-slot read."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_completion_snapshot_reuse_test")
    mod._reset_modify_runtime_state()
    saved_analytics = mod._SHOW_ANALYTICS
    mod._SHOW_ANALYTICS = True
    from nautical_core.integration_models import CommandFailureKind, Found, TaskCommand, TaskCommandResult

    uow = test_operator_uow()
    calls = {"count": 0}

    class Client:
        def execute(self, args, *, purpose, timeout, **_kwargs):
            calls["count"] += 1
            rows = [{
                "uuid": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "chainID": "reuse01",
                "link": 2,
                "chain": "on",
                "status": "pending",
            }]
            command = TaskCommand(("task", *args), purpose, timeout)
            return TaskCommandResult(command, 0, json.dumps(rows), "", CommandFailureKind.SUCCESS, 1, 0.001)

    uow.client = Client()
    try:
        snapshot = mod._completion_effects.chain_snapshot("reuse01", 1, 2, uow.repository)
        expect(snapshot.loaded and snapshot.coverage == "full", f"unexpected full snapshot: {snapshot!r}")
        reused = uow.repository.exact_child_slot("reuse01", 2)
        expect(isinstance(reused, Found), f"full snapshot did not satisfy child-slot read: {reused!r}")
        expect(calls["count"] == 1, f"full snapshot was exported more than once: {calls}")
    finally:
        mod._SHOW_ANALYTICS = saved_analytics
        mod._reset_modify_runtime_state()

TESTS = TESTS + (
    test_on_modify_completion_chain_snapshot_modes_and_query,
    test_on_modify_recompleted_task_with_nextlink_skips_spawn,
    test_on_modify_recompleted_task_with_existing_link_skips_spawn,
    test_on_modify_completion_reuses_single_chain_export_when_chain_needed,
    test_on_modify_completion_snapshot_reuses_full_chain_read,
)


def test_on_modify_lifecycle_export_reuses_completion_chain_snapshot():
    """Lifecycle filtering and completion presentation share one chain export."""
    hook = find_hook_file("on-modify.nautical")
    mod = load_hook_module(hook, "_nautical_lifecycle_export_reuse_test")
    mod._reset_modify_runtime_state()
    saved_analytics = mod._SHOW_ANALYTICS
    mod._SHOW_ANALYTICS = True
    from nautical_core.integration_models import CommandFailureKind, TaskCommand, TaskCommandResult

    uow = test_operator_uow()
    calls = {"count": 0}

    class Client:
        def execute(self, args, *, purpose, timeout, **_kwargs):
            calls["count"] += 1
            rows = [{
                "uuid": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "chainID": "reuse02",
                "link": 2,
                "chain": "on",
                "status": "pending",
            }]
            command = TaskCommand(("task", *args), purpose, timeout)
            return TaskCommandResult(command, 0, json.dumps(rows), "", CommandFailureKind.SUCCESS, 1, 0.001)

    uow.client = Client()
    mod._modify_runtime_state().task_repository = uow.repository
    try:
        rows = mod._module("modify_composition").lifecycle_read_service_for(mod).get_chain_export("reuse02")
        expect(len(rows) == 1, f"lifecycle chain export returned unexpected rows: {rows!r}")
        snapshot = mod._completion_effects.chain_snapshot("reuse02", 1, 2, uow.repository)
        expect(snapshot.loaded and snapshot.rows, f"completion snapshot did not reuse chain rows: {snapshot!r}")
        expect(calls["count"] == 1, f"lifecycle and completion repeated chain export: {calls}")
    finally:
        mod._SHOW_ANALYTICS = saved_analytics
        mod._reset_modify_runtime_state()

TESTS = TESTS + (
    test_on_modify_lifecycle_export_reuses_completion_chain_snapshot,
)
