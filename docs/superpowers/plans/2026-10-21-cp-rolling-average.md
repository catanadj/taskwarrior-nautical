# Rolling-average completion periods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `cp:avg(x)` with a safe one-day bootstrap and history-derived rolling intervals while preserving all existing CP behavior.

**Architecture:** Extend the CP parser with a typed average token. Inject an authoritative chain-history reader into `ChainGenerationService`, where a single resolver computes the bootstrap or rolling interval from adjacent completed links. Completion and reconcile already share this service, so both paths use the same result.

**Tech Stack:** Python, `datetime`, existing Taskwarrior authoritative repository, unittest contract tests, MkDocs documentation.

**Spec:** `docs/superpowers/specs/2026-10-21-cp-rolling-average-design.md`

## Global Constraints

- Fixed/random/jittered/sequenced CP behavior must remain unchanged.
- Hook stdout remains strict Taskwarrior JSON; diagnostics remain stderr-only under `NAUTICAL_DIAG=1`.
- History failures must fail closed through the existing manual-review/error path.
- Use `end` timestamps and adjacent numeric links only.

## Review Focus

- A fixed `cp:3d` without due still defaults to entry plus three days.
- `cp:avg(x)` without due seeds the current hook time plus one day.
- One completed interval bootstraps safely; later intervals change the rolling average deterministically.
- Missing, malformed, duplicate, or non-adjacent chain history never produces a guessed due.
- Add preview and modify feedback describe average mode without pretending to know a future interval.

### Task 1: Parse and validate average CP tokens

**Files:**
- Modify: `nautical_core/cp_parser.py`
- Modify: `nautical_core/add_validation.py`
- Modify: `nautical_core/add_preview_composition.py`
- Test: `tests/recurrence/test_cp_sequence_contracts.py`
- Test: existing add validation/preview contract tests

- [ ] Add a typed `avg` token carrying the positive window and make `avg(0)`, negative, non-integer, and malformed forms invalid.
- [ ] Verify parser and validation tests fail before implementation and pass after it.
- [ ] Keep average tokens out of fixed-duration arithmetic helpers unless explicitly handled.

### Task 2: Add authoritative rolling interval resolution

**Files:**
- Modify: `nautical_core/chain_generation.py`
- Modify: `nautical_core/modify_generation_effects.py` and its factory wiring
- Modify: generation construction sites that already have a `TaskReadRepository`
- Test: `tests/test_chain_generation_recovery_contract.py`

- [ ] Inject a narrow chain-history reader contract, not a hook/module global.
- [ ] Add failing tests for one-day bootstrap, one measured interval, latest-window averaging, link gaps, malformed `end`, and unavailable history.
- [ ] Implement a resolver that selects adjacent completed links by numeric `link`, averages exact seconds, and raises the existing safe generation error for unusable history.
- [ ] Return metadata identifying `cp:avg(x)`, bootstrap versus history basis, sample count, and window.

### Task 3: Preserve add and completion behavior

**Files:**
- Modify: `nautical_core/add_preview_composition.py` and add composition/default handling
- Modify: `nautical_core/modify_completion_effects.py` only where average metadata/error routing requires it
- Test: add-hook and completion contract tests

- [ ] Seed missing due with `entry + 1 day` only for average mode.
- [ ] Leave fixed CP's existing `entry + cp` default untouched.
- [ ] Verify completion creates the child from the shared generation result and malformed history follows manual review.

### Task 4: Reconcile parity and documentation

**Files:**
- Modify: reconcile construction/wiring where repository injection is required
- Modify: `docs/guides/completion-periods.md`
- Modify: `docs/guides/recipes.md` if gallery coverage is appropriate
- Test: reconcile contract tests and full suite

- [ ] Verify reconcile and completion produce identical average child due values from the same chain snapshot.
- [ ] Document syntax, bootstrap behavior, sample/window semantics, and safety failures.
- [ ] Run targeted tests, mypy, and the full test suite.

