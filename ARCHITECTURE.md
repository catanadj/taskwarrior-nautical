# Nautical Architecture Map

This document is an internal map of the current code structure.

It is not a user manual. It exists to answer:

- which file owns which behavior
- how the three hooks are divided
- where shared recurrence logic lives
- where to make changes safely

## Runtime Shape

Nautical still installs as:

- `on-add-nautical.py`
- `on-modify-nautical.py`
- `on-exit-nautical.py`
- `nautical_core/`

The hooks are the public runtime entrypoints.
The `nautical_core/` package contains shared services and internal modules.

## Hook Responsibilities

### `on-add-nautical.py`

Owns:

- reading the new task from stdin
- validating add-time recurrence settings
- previewing the initial chain / anchor schedule
- auto-assigning due when appropriate
- emitting the final task JSON to stdout

Internal split:

- formatting: [`nautical_core/add_formatting.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/add_formatting.py)
- validation/preflight: [`nautical_core/add_validation.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/add_validation.py)
- anchor date stepping / preview compute: [`nautical_core/add_anchor_compute.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/add_anchor_compute.py)
- anchor preview orchestration: [`nautical_core/add_anchor_preview.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/add_anchor_preview.py)

### `on-modify-nautical.py`

Owns:

- handling status transitions, especially completion
- deciding whether a next child should be created
- spawning or deferring child creation
- rendering completion feedback and summaries
- non-completion validation and chain updates

Internal split:

- read-only queries: [`nautical_core/modify_queries.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_queries.py)
- chain reads: [`nautical_core/modify_chain_reads.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_chain_reads.py)
- child payload prep: [`nautical_core/modify_spawn_prep.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_spawn_prep.py)
- completion preflight: [`nautical_core/modify_completion_preflight.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_completion_preflight.py)
- completion compute: [`nautical_core/modify_completion_compute.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_completion_compute.py)
- completion spawn wrapper: [`nautical_core/modify_completion_spawn.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_completion_spawn.py)
- completion feedback rendering: [`nautical_core/modify_feedback.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_feedback.py)
- completion timeline rendering: [`nautical_core/modify_timeline.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/modify_timeline.py)

### `on-exit-nautical.py`

Owns:

- draining deferred spawn intents after Taskwarrior releases its lock
- importing queued child tasks
- reconciling parent `nextLink`
- dead-letter / retry behavior
- queue backend orchestration

Internal split:

- read-only query helpers: [`nautical_core/exit_queries.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/exit_queries.py)
- import / parent-update side effects: [`nautical_core/exit_side_effects.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/exit_side_effects.py)
- per-entry decision flow: [`nautical_core/exit_entry_flow.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/exit_entry_flow.py)

The queue storage and drain backend still mostly live in `on-exit-nautical.py`.

## Shared Core Layers

### Core Facade

[`nautical_core/__init__.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/__init__.py) is now primarily a compatibility facade and shared API surface.

It still exports the names the hooks rely on, but most major subsystems have been split into dedicated modules.

### Runtime / Config / UI

- runtime + subprocess + taskdata resolution: [`nautical_core/runtime.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/runtime.py)
- config loading and typed access: [`nautical_core/config_support.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/config_support.py)
- warnings and rate-limited notices: [`nautical_core/warnings.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/warnings.py)
- UI / panel helpers: [`nautical_core/ui.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/ui.py)
- generic hook transport helpers: [`nautical_core/hook_support.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/hook_support.py)

### Parsing and Validation

- token normalization: [`nautical_core/tokenutil.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/tokenutil.py)
- yearly token rewrites: [`nautical_core/year_tokens.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/year_tokens.py)
- nth-weekday monthly helpers: [`nautical_core/nth_monthly.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/nth_monthly.py)
- quarter selector/rewrite support:
  - [`nautical_core/quarter_helpers.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/quarter_helpers.py)
  - [`nautical_core/quarter_selector.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/quarter_selector.py)
  - [`nautical_core/quarter_rewrite.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/quarter_rewrite.py)
- yearly parsing/validation:
  - [`nautical_core/yearly_parse.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/yearly_parse.py)
  - [`nautical_core/yearly_validation.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/yearly_validation.py)
- parser frontend / atoms / DNF driver:
  - [`nautical_core/parser_frontend.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/parser_frontend.py)
  - [`nautical_core/parser_atoms.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/parser_atoms.py)
  - [`nautical_core/parser_dnf.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/parser_dnf.py)
- strict validation + linting:
  - [`nautical_core/strict_validation.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/strict_validation.py)
  - [`nautical_core/linting.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/linting.py)

### Scheduling and Expansion

- expansion helpers: [`nautical_core/expansion_support.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/expansion_support.py)
- monthly support: [`nautical_core/monthly_support.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/monthly_support.py)
- cached expansion: [`nautical_core/cached_expansion.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/cached_expansion.py)
- atom scheduler: [`nautical_core/scheduler_atom.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/scheduler_atom.py)
- term/expression scheduler: [`nautical_core/scheduler_expr.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/scheduler_expr.py)
- schedule utilities: [`nautical_core/schedule_utils.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/schedule_utils.py)
- satisfiability checks: [`nautical_core/satisfiability.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/satisfiability.py)

### Precompute / Cache / ACF

- preview and hint precompute: [`nautical_core/precompute.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/precompute.py)
- cache paths and cache helpers:
  - [`nautical_core/cache_support.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/cache_support.py)
  - [`nautical_core/cache_locking.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/cache_locking.py)
  - [`nautical_core/cache_payload.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/cache_payload.py)
- ACF normalization/canonicalization: [`nautical_core/acf_support.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/acf_support.py)

### Natural Language

- anchor description / prose rendering: [`nautical_core/natural_language.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/natural_language.py)

## Utility Modules

- generic pure helpers: [`nautical_core/common.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/common.py)
- date arithmetic: [`nautical_core/dates.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/dates.py)
- time helpers: [`nautical_core/timeutil.py`](/home/pooK/.sku/DB/.files/TaskWarrior_tools/github/taskwarrior-nautical/nautical_core/timeutil.py)

## Safe Change Guidelines

If you want the safest edit location:

- recurrence parsing: change parser/validation modules first, not the hooks
- scheduling behavior: change scheduler modules first, not preview wrappers
- add-time preview presentation: change `add_*` modules first
- completion behavior: change `modify_*` modules first
- drain/import behavior: change `exit_*` modules first

Avoid starting in the hooks unless the change is truly about:

- hook IO
- hook orchestration
- top-level taskwarrior command flow

## Current Refactor Status

Phase 1:

- split the former giant core implementation into focused internal modules

Phase 2:

- refactored `on-add` into real submodules
- consolidated `on-modify` loader/fallback structure
- split `on-exit` into query / side-effect / entry-flow layers

Remaining larger concentrations of complexity:

- `on-modify-nautical.py`
- `on-exit-nautical.py` queue/storage plumbing
- `nautical_core/__init__.py` compatibility facade
