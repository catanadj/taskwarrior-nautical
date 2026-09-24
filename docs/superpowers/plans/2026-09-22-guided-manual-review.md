# Guided Manual-Review Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a Nautical-owned, evidence-backed command that presents manual-review items one at a time and offers only guarded, applicable actions.

**Architecture:** Reuse the lifecycle outbox, typed query/read services, chain graph, and existing repair planner. Add a typed review projection between those services and the CLI; action execution goes through the existing lifecycle/integrity mutation gateways, never raw Taskwarrior commands.

**Tech Stack:** Python 3, `unittest`, existing Nautical typed models/services, argparse CLI, JSON output.

**Spec:** `docs/superpowers/specs/2026-09-22-guided-manual-review-design.md`

## Global Constraints

- Default mode performs no writes.
- Hook stdout remains strict JSON; diagnostics remain stderr-only and opt-in.
- Every mutation is guarded by current parent/slot evidence.
- Ambiguous or stale evidence remains `manual_review`.
- No new required UDA is introduced.
- No automatic deletion of duplicate or orphan tasks.

## Review Focus

- Duplicate slot with two historical occupants: show both occupants and offer no branch-selection action.
- Already-applied successor: offer only idempotent acknowledgement.
- Stale parent guard: refuse the selected action and refresh the review item.
- Missing/unavailable chain snapshot: return a bounded retryable/manual result without raw exports.
- Interrupted action: rerunning the same action must converge without duplicate imports or links.

### Task 1: Typed review-item projection

**Files:**
- Create: `nautical_core/manual_review_models.py`
- Modify: `nautical_core/queue_status_service.py`
- Test: `tests/test_manual_review_models.py`

**Interfaces:**
- Produce `ManualReviewItem`, `ManualReviewAction`, and `ManualReviewEvidence` immutable models.
- `ManualReviewItem.to_dict()` returns stable JSON-native evidence and action descriptions.
- `QueueStatusService.build_review_item(record, *, task_binary)` returns one typed item or a typed unavailable result.

- [ ] Write tests for stable serialization, action filtering, duplicate-slot ambiguity, and unavailable evidence.
- [ ] Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_manual_review_models -v` and verify the new tests fail because the models do not exist.
- [ ] Implement the immutable models and the projection using existing outbox records, `TaskwarriorClient`, `TaskCodec`, `ChainGraph`, and `ChainRepairPlanner`.
- [ ] Keep raw task rows internal to the projection; expose UUID prefixes, links, statuses, and structured reasons only.
- [ ] Re-run the focused tests and verify they pass.
- [ ] Run `git diff --check`.

### Task 2: Fresh chain evidence and explanation

**Files:**
- Modify: `nautical_core/queue_status_service.py`
- Test: `tests/test_queue_review.py`

**Interfaces:**
- Extend `review_payload()` with `--next`-compatible ordering and one-item evidence.
- Preserve existing `review_payload()` JSON compatibility while adding `review_item` fields.

- [ ] Add failing tests for a duplicate-slot record, an already-applied successor, and a missing parent.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Build fresh chain snapshots through the typed read service, evaluate invariants, and attach planner refusal/plan evidence.
- [ ] Ensure duplicate slots produce `manual_review` with both occupant UUIDs and no automatic branch action.
- [ ] Re-run focused tests.

### Task 3: Guided CLI navigation

**Files:**
- Modify: `nautical_core/tools/nautical_queue_review.py`
- Modify: `nautical` command dispatch/help if required by the existing launcher.
- Test: `tests/test_queue_review.py`
- Test: `tests/test_operator_process_contract.py`

**Interfaces:**
- Add `--next` to select the highest-priority unresolved item.
- Add `--json` output containing exactly one `review_item` plus evidence/actions.
- Keep existing `--intent`, listing, and `--resolve-applied` behavior compatible.

- [ ] Add failing CLI tests for `--next`, empty queue, and JSON output.
- [ ] Implement deterministic priority ordering: manual review severity, stale age, then intent ID.
- [ ] Render concise human output with situation, evidence, and numbered available actions.
- [ ] Keep diagnostics off stdout and preserve strict JSON behavior.
- [ ] Re-run CLI-focused tests.

### Task 4: Guarded action dispatch

**Files:**
- Modify: `nautical_core/queue_status_service.py`
- Modify: `nautical_core/tools/nautical_queue_review.py`
- Test: `tests/test_queue_review.py`
- Test: `tests/test_lifecycle_failure_injection.py`

**Interfaces:**
- Add `QueueStatusService.apply_review_action(taskdata, intent_id, action, confirmation, task_binary)`.
- Accepted actions initially: `resolve-applied`, `retry`, `replan`, `quarantine`, `skip`.
- Return structured `{status, reason, review_item}` results.

- [ ] Add failing tests for invalid action, missing confirmation, stale confirmation, idempotent resolve, and retry after interruption.
- [ ] Implement confirmation tokens derived from the displayed intent/evidence fingerprint.
- [ ] Re-read parent and chain evidence immediately before mutation.
- [ ] Route `resolve-applied` and retry/replan through existing outbox/lifecycle services.
- [ ] Make `quarantine` a durable manual-review state transition; make `skip` non-mutating.
- [ ] Refuse branch acceptance until a planner proof exists.
- [ ] Re-run lifecycle failure-injection tests.

### Task 5: End-to-end verification and documentation

**Files:**
- Modify: `tests/test_golden_registry_integrity.py` if the test registry requires the new contracts.
- Modify: `Manual.md` or the existing operator documentation location.
- Test: `tests/test_queue_review.py`

- [ ] Add an end-to-end fixture covering duplicate-slot analysis, display, stale confirmation, and safe resolution.
- [ ] Document `nautical review --next`, `--json`, and explicit action application.
- [ ] Run focused review tests, then the complete suite:
  `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q`
- [ ] Run `git diff --check` and the deployment sanity check.
- [ ] Report any skipped or environment-dependent tests explicitly.

