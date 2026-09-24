"""Lifecycle and durable-outbox golden-test ownership boundary.

The implementation bodies remain in the legacy runner during this staged
extraction.  These wrappers move registration and execution ownership into a
domain collection without duplicating lifecycle helpers.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import tempfile
import sys
import time
from pathlib import Path

from dev_tools.golden_tests.support import expect, plan_from_values, task_draft, task_observation
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
