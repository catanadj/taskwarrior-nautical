# High-ROI Health and Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the three highest-value health areas with one bounded installer vertical slice: direct tests, explicit snapshot types, and private backup permissions.

**Architecture:** Keep filesystem transaction ownership in `nautical_core/install_filesystem.py`. Add typed snapshot contracts at that boundary, test the existing atomic operations directly, and ensure installer backups cannot inherit permissive source or directory permissions. No public import paths or runtime workflow contracts change.

**Tech Stack:** Python 3.11, `unittest`, `pathlib`, `TypedDict`, POSIX file modes.

**Spec:** User-approved ROI priorities: test health, type safety, and production security hardening.

## Global Constraints

- Hook stdout remains strict JSON; diagnostics stay on stderr only under `NAUTICAL_DIAG=1`.
- Preserve `ensure_ascii=False` for JSON output.
- Keep edits small and avoid touching untracked files.
- Preserve public behavior and existing installer transaction semantics.

## Review Focus

- A backup directory must be created privately even when the caller did not pre-create it.
- A copied backup must be owner-readable/writable only, regardless of source permissions.
- Missing, symlink, regular-file, and invalid managed paths must retain their current snapshot/restore behavior.
- Atomic helpers must remove temporary files when the underlying operation raises.
- Typed snapshot values must remain compatible with existing callers and mypy.

### Task 1: Direct installer filesystem coverage

**Files:**
- Create: `tests/test_install_filesystem.py`
- Test: `tests/test_install_filesystem.py`

**Interfaces:**
- Consumes: `InstallLock`, `atomic_symlink`, `atomic_copy`, `atomic_write_text`, `snapshot_file`, `restore_file`, `pointer_snapshot`, `restore_pointer`.
- Produces: direct regression coverage for installer transaction behavior.

- [x] Write tests for lock serialization, atomic writes/copies/symlinks, file snapshots, pointer snapshots, and restore behavior.
- [x] Run the new test module and confirm it exercises the real module without mocks for filesystem operations.
- [ ] Commit only after the production hardening in Task 2 is green.

### Task 2: Typed and private installer snapshots

**Files:**
- Modify: `nautical_core/install_filesystem.py`
- Test: `tests/test_install_filesystem.py`

**Interfaces:**
- Consumes: existing snapshot dictionaries and restore functions.
- Produces: `FileSnapshot` and `PointerSnapshot` TypedDict contracts; `snapshot_file` returns a typed union and `restore_file` accepts it.

- [x] Add typed snapshot definitions without changing serialized keys.
- [x] Make `snapshot_file` create `backup_dir` with mode `0700` and force copied backup files to mode `0600`.
- [x] Add tests that source files with permissive modes still produce private backups.
- [x] Run the focused tests and mypy for `nautical_core`.

### Task 3: Verification and scanner evidence

**Files:**
- Modify: none beyond Tasks 1–2.

**Interfaces:**
- Consumes: the typed installer boundary and its direct tests.
- Produces: evidence for test health, type safety, and security improvements.

- [x] Run `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest tests.test_install_filesystem -q`.
- [x] Run the full repository unit suite with an explicit repository test root.
- [ ] Run the golden suite and mypy.
- [x] Run `git diff --check` and record the scanner delta; do not resolve unrelated findings without evidence.
