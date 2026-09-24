# WP5 Compatibility Facade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Inventory and progressively shrink the root compatibility facade without breaking the 130-name public surface or installed-runtime imports.

**Architecture:** Treat `compat_api.PUBLIC_EXPORTS` as the frozen public contract. Assign each name a canonical owner, preserve lazy forwarding and monkeypatch seams, and migrate one cohesive owner group at a time. The root facade remains metadata and forwarding only; implementation stays in focused API modules.

**Tech Stack:** Python 3.11, lazy imports, typed `ApiBinding`, unittest, mypy, deployment sanity.

**Spec:** `checklists/REPOSITORY_UPGRADE_REFACTOR_CHECKLIST.md`, Work Package 5.

## Global Constraints

- Preserve all 130 public names and current signatures until an explicit deprecation decision.
- Preserve installed hook/runtime imports, lazy loading, cache attributes, and monkeypatch points.
- Do not remove zero-use aliases based only on internal search results.
- Migrate one cohesive ownership group per change.
- Keep domain and recurrence modules independent of the root facade and integration dependencies.

## Review Focus

- A public name imported before its owner module is loaded must still resolve lazily; test cold imports.
- A caller monkeypatching a compatibility alias must affect the documented owner seam; test alias assignment.
- Signature and cache-control attributes must remain stable through forwarding; test `inspect.signature` and cache APIs.
- Installed runtime copies must expose the same imports as the source tree; run deployment/import checks.
- Domain and recurrence modules must not import the facade; run architecture-contract tests after each group.

### Task 1: Freeze the public surface

**Files:**
- Create: `tests/test_public_surface_snapshot.py`
- Create: `docs/superpowers/wp5-public-surface.md`
- Modify: `checklists/REPOSITORY_UPGRADE_REFACTOR_CHECKLIST.md`
- Modify: `.superpowers/sdd/2026-09-21-wp5-compatibility-facade/progress.md`

Record the 130 export names, current signatures for callable exports, model/type owners, compatibility aliases, and cache attributes. Add tests that import the facade cold, resolve every name, and assert the snapshot remains stable.

### Task 2: Assign canonical owners

**Files:**
- Modify: `nautical_core/compat_api.py`
- Modify: `nautical_core/api_bindings.py`
- Test: `tests/test_public_surface_snapshot.py`, `tests/test_typed_api_bindings.py`

Add a typed owner registry mapping every export to its canonical module and classification (`public`, `installed_runtime`, `test_seam`, or `legacy_alias`). Keep forwarding behavior unchanged and fail tests if an export lacks an owner.

### Task 3: Migrate one cohesive API group

**Files:**
- Modify: one selected `*_api.py` owner and `nautical_core/compat_api.py`
- Test: focused owner contract plus public-surface snapshot

Select the smallest group with a complete owner (configuration/cache or parser). Move implementation only if the owner already exposes an equivalent factory; retain lazy aliases and assignment behavior. Verify cold import, signatures, monkeypatching, and architecture boundaries.

### Task 4: Installed-runtime compatibility gate

Run the public-surface tests, typed binding tests, architecture tests, parser owner tests, deployment sanity, and strict mypy before selecting the next group. Do not remove forwarding modules until the installed-layout checks pass.

### Task 5: Deprecation decision record

Document the compatibility window and removal rule for each alias. No alias removal occurs in WP5 until its owner, importer references, installed layout, and release notes are covered.
