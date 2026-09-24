# Facade Owner Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove internal parser, scheduler, and cache dependence on the mutable `nautical_core` facade while preserving public API behavior.

**Architecture:** Compatibility wrappers remain in `nautical_core/__init__.py`; each API composition root constructs an immutable owner dependency bundle. Owner functions consume explicit collaborators and direct sibling modules, with no runtime service-locator lookup.

**Tech Stack:** Python 3.11+, dataclasses, typing.Protocol, unittest, mypy.

**Spec:** Approved architectural design in the 2026-09-15 conversation.

## Global Constraints

- Preserve public facade signatures and Unicode-safe hook behavior.
- Keep dependency construction at `for_core()`/`CoreContext` boundaries.
- Do not add compatibility bridges inside owner implementations.
- Every migration slice must have focused tests and pass `git diff --check`.

### Task 1: Parser owner dependency closure

**Files:**
- Modify: `nautical_core/parser_api.py`
- Test: `tests/test_architecture_contract.py`, `tests/test_parser_contracts.py`

**Interfaces:**
- Produce an immutable parser owner bundle containing parser modules, preset tables, validators, and error types.
- Keep `for_core(...).parse_anchor_expr_to_dnf` and strict-validation signatures unchanged.

- [ ] Add a direct-owner test that constructs parser dependencies without root-facade attribute reads.
- [x] Move parser DNF closure construction to consume explicit parser-owner collaborators.
- [x] Keep legacy module-level functions as thin compatibility adapters.
- [x] Run parser and architecture tests.
- [x] Commit parser owner/caller migration slices (`cc5aeda`, `7ce4326`, `2d64f8b`, `32e6cb4`, `a665f23`).

### Task 2: Scheduler owner dependency closure

**Files:**
- Modify: `nautical_core/scheduler_api.py`
- Test: `tests/test_scheduler_api_contract.py`, `tests/recurrence/test_scheduler_cross_path_conformance.py`

**Interfaces:**
- Produce immutable atom, modifier, and runtime dependency bundles.
- Preserve all public scheduler wrappers and explicit business-calendar overrides.

- [ ] Add direct-owner tests with a minimal dependency bundle.
- [ ] Replace module namespace reads in scheduler closures with bundle fields.
- [ ] Verify active calendar, astronomy, random selection, and interval behavior.
- [ ] Run scheduler contract and recurrence conformance tests.
- [ ] Commit the scheduler slice.

### Task 3: Cache owner dependency closure

**Files:**
- Modify: `nautical_core/cache_api.py`
- Test: `tests/test_cache_api_contract.py`, `tests/test_cache_locking.py`

**Interfaces:**
- Produce an immutable cache runtime bundle for filesystem, clock, locking, serializer, and memory state.
- Preserve cache key, TTL, locking, and atomic replacement behavior.

- [ ] Add a direct-owner construction test with explicit fake filesystem/clock/lock collaborators.
- [ ] Remove fallback reads from the mutable root namespace in cache operations.
- [ ] Verify lock failure, stale-lock recovery, corruption rejection, and TTL behavior.
- [ ] Run cache contract and locking tests.
- [ ] Commit the cache slice.

### Task 4: Enforce the boundary and parity

**Files:**
- Modify: `nautical_core/architecture_contract.py`
- Test: `tests/test_architecture_contract.py`, `tests/test_typed_api_bindings.py`

- [ ] Extend the static validator to reject owner reads through `module`/`core` namespaces except in compatibility adapters.
- [ ] Add facade-parity tests covering representative parser, scheduler, and cache calls.
- [ ] Run compilation, focused tests, full unittest discovery, mypy, golden, black-box, and deployment sanity.
- [ ] Review the diff for accidental public API changes and record evidence in the WP17 checklist.
- [ ] Commit the boundary-enforcement slice.

## Verification Commands

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_architecture_contract tests.test_parser_contracts tests.test_scheduler_api_contract tests.test_cache_api_contract -q
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest discover -s tests -q
python3 -m mypy --config-file mypy.ini
git diff --check
```
