# Parser Domain Migration Implementation Plan

> **For agentic workers:** Execute this plan inline with focused test checkpoints.

**Goal:** Establish a navigable `nautical_core/parsing/` domain while preserving legacy import paths and runtime loading.

**Architecture:** Move parser implementation modules into a package and leave compatibility shims at their historical paths. Update lazy manifests and package-relative imports together, then verify both source-tree and installed-layout imports.

**Tech Stack:** Python 3, unittest, Taskwarrior hook loader, desloppify.

**Spec:** The approved parser-domain migration slice from the remediation checklist.

## Global Constraints

- Preserve strict JSON hook stdout and existing public import names.
- Keep `ensure_ascii=False` behavior unchanged.
- Do not add generated `.desloppify/` state to product commits.
- Keep all commits local; no push.

### Task 1: Map parser dependencies and add package boundary

**Files:**
- Create: `nautical_core/parsing/__init__.py`
- Test: `tests/test_parser_domain_imports.py`

- [x] Add import-contract tests for canonical and compatibility module paths.
- [x] Run the new tests and confirm the canonical path is importable after migration.
- [x] Define package exports without changing behavior.

### Task 2: Move parser implementations with compatibility shims

**Files:**
- Create: `nautical_core/parsing/parser_atoms.py`, `parser_dnf.py`, `parser_models.py`, `parser_support_api.py`, `parser_frontend.py`
- Modify: historical `nautical_core/parser_*.py` modules as shims.
- Modify: `nautical_core/runtime_manifest.py`, `nautical_core/__init__.py`, and direct callers.

- [x] Move implementation ownership to the package and preserve old imports through shims.
- [x] Update string-based lazy module names and relative imports.
- [x] Run parser, hook-loading, and installed-layout tests.

### Task 3: Verify and document the boundary

**Files:**
- Modify: `tests/test_parser_domain_imports.py`
- Modify: `checklists/DESLOPPIFY_REVIEW_REMEDIATION_CHECKLIST.md`

- [x] Run the full unittest suite, compilation, and diff checks.
- [ ] Run `desloppify scan --path .` with `local-archive` excluded after the queue is drained.
- [x] Resolve the package-organization finding with migration evidence.
