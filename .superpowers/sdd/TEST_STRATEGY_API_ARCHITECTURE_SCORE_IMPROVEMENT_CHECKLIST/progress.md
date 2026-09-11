# SDD ledger — plan: checklists/TEST_STRATEGY_API_ARCHITECTURE_SCORE_IMPROVEMENT_CHECKLIST.md

## Setup

- Workspace: `.superpowers/sdd/TEST_STRATEGY_API_ARCHITECTURE_SCORE_IMPROVEMENT_CHECKLIST`
- Branch: `desloppify/review-remediation`
- Existing worktree changes are user-owned remediation; do not reset, stash, or
  fold unrelated files into task commits.
- No reachable design specification was supplied beyond the checklist and its
  referenced desloppify evidence; the checklist is the binding plan.

## Preflight plan review

| Task | Self-consistency check | Result |
|---|---|---|
| 1 Direct natural-language tests | Test file, direct owners, and migrated registry cases align. | Clear |
| 2 On-add route matrix | Subprocess fixture, JSON assertions, and route list align. | Clear |
| 3 Direct test migration | Waves and coverage gates use the same direct-test boundary. | Clear |
| 4 Golden decomposition | Registry removal and discovery requirements are consistent. | Clear |
| 5 Coverage ratchets | Development-only coverage and fast-gate requirements are compatible. | Clear |
| 6 Operator result | One typed result and presentation-only serialization are consistent. | Clear |
| 7 Datetime port | Canonical parser tuple and field-aware decoration are consistent. | Clear |
| 8 Typed API bindings | Frozen bindings and compatibility aliases have distinct roles. | Clear |
| 9 Architecture contract | Layer direction and forbidden-import tests reinforce one another. | Clear |
| 10 CoreContext removal | Parser, scheduler, and cache each retain a compatibility factory. | Clear |
| 11 Hook-host removal | Composition owns services while hook entry points retain protocol duties. | Clear |
| 12 Lifecycle capability port | Complete execution is required before drain; stage-only remains supported. | Clear |
| 13 Presentation contexts | Focused contexts replace wide bags without adding pass-through layers. | Clear |
| 14 Security adjudication | Evidence-based classification precedes any production security edit. | Clear |
| 15 Facade boundary | Public compatibility remains while internal owners stop importing the facade. | Clear |
| 16 Type checking | Type gates cover every new boundary introduced by Tasks 6–15. | Clear |
| 17 Final gate | Full verification and fresh blind review occur after implementation. | Clear |

### Shared-file and shared-interface review

| Tasks | Shared surface | Finding and ruling |
|---|---|---|
| 1, 3, 4 | Natural-language tests and golden registry | Migrate behavior once; Task 1 establishes direct contracts, Task 3 expands coverage, and Task 4 removes migrated registry entries. No duplication is permitted. |
| 2, 3 | Hook tests | Task 2 owns executable route behavior; Task 3 may add direct owner tests but must not duplicate subprocess cases. |
| 3, 5 | Test coverage gates | Task 3 supplies behavioral coverage; Task 5 measures and ratchets it. The ratchet cannot replace missing behavior assertions. |
| 6, 8 | Operator/API result types | Task 6 defines the result contract; Task 8 may expose it through typed bindings but cannot create a second result shape. |
| 7, 8 | Datetime parser exposure | Task 7 owns parsing semantics; Task 8 only types and publishes the stable port. |
| 8, 10, 15 | Facade and API factories | Task 8 types bindings, Task 10 removes service-locator use, and Task 15 limits compatibility imports. Each preserves the public boundary. |
| 9, 10, 15, 16 | Import direction and type gates | Task 9 defines the rule, Task 10/15 migrate callers, and Task 16 enforces types. No task may weaken the architecture rule to pass typing. |
| 11, 12 | Hook composition and lifecycle execution | Task 11 removes host reach-through; Task 12 supplies a complete lifecycle port at that new composition boundary. |
| 11, 13 | Modify presentation services | Task 11 removes hook state access; Task 13 narrows renderer collaborators after that boundary is explicit. |
| 12, 17 | Lifecycle safety | Task 12 preserves guards, limits, and postconditions; Task 17 verifies interruption and recovery. |
| 13, 17 | Renderer output | Task 13 preserves output contracts; Task 17 verifies black-box and Unicode behavior. |
| 14, 17 | Security evidence | Task 14 adjudicates signals and adds fixes/tests only for genuine defects; Task 17 verifies the resulting trust boundaries. |
| 15, 16 | Compatibility and annotations | Task 15 keeps documented aliases; Task 16 checks the migrated internals without forcing false precision on Taskwarrior payloads. |

### Preflight rulings

- No contradictory task pair was found.
- Ruling: preserve the existing dirty worktree and isolate each new logical
  change in a focused local commit — this avoids destroying user remediation;
  the cost if wrong is a more fragmented history, which can be squashed at
  final integration.
- Ruling: execute Tasks 1–5 before API and architecture rewrites — direct
  behavioral evidence is the safety net for later boundary changes; the cost
  if wrong is slower early progress, not behavior divergence.

## Task status

- Task 1: complete (commits `c48e07e..b0b5673`, scoped re-review clean)
- Ruling: preserve the pre-existing golden-test rewrites that were already in
  the shared dirty worktree before Task 1. Rewriting history to split them now
  risks discarding user-owned remediation; the cost if wrong is a less focused
  Task 1 commit, which must be audited in the final branch review.
- Task 1 evidence: direct suite 5/5; natural golden slice 14/15 with one
  unrelated pre-existing hook-preview failure; no production behavior changed.
- Task 2: complete (commits `7ed1715..3e41f57`, three scoped review rounds clean)
- Task 2 evidence: focused hook/route suite 10/10; golden `hook_on_add` slice
  41/41; no production behavior retained from the review fix.
- Ruling: split Work Package 3 into independently reviewed Waves A, B, and C;
  each has distinct owners and test gates, so one monolithic coverage change
  would be difficult to review and safely revert. The cost if wrong is extra
  review bookkeeping, not behavioral scope.
- Task 3A: decomposed after usage-limit failure; cache, parser, scheduler, and
  calendar sub-waves will be reviewed independently.
- Task 3A-cache: pending (direct cache API contracts)
- Task 3A-cache: complete (commits `b593f2f..ff3254e`, review clean)
- Task 3A-cache evidence: focused suite 6/6; full discovery 532 tests with one
  unrelated recurrence-terminal failure; cache golden slice 46/50 with four
  unrelated dynamic-module loading failures.
- Task 3A-parser: complete (commit `951dba7`, scoped review approved)
- Task 3A-parser evidence: focused parser/domain suite 9/9; parser golden
  characterization 3/3; no production changes. Residual full golden failure
  is the pre-existing lazy parser import characterization.
- Task 3A-scheduler: pending (direct scheduler atom/API contracts)
- Task 3A-scheduler: in progress (base `951dba7`; brief `task-3a-scheduler-brief.md`)
- Task 3A-scheduler: complete (commits `29a2c84..952d55c`, scoped re-review approved)
- Task 3A-scheduler evidence: focused suite 9/9; scheduler characterization 2/2;
  golden scheduler slice 31/32 with one pre-existing lazy-import failure;
  tests-only changes.
- Task 3A-calendar: in progress (base `952d55c`; brief `task-3a-calendar-brief.md`)
- Task 3A-calendar: complete (commits `b1b7fa9..e2c1dd4`, scoped re-review approved)
- Task 3A-calendar evidence: focused suite 7/7; tests-only changes; positive
  configured membership and shift-limit exhaustion explicitly covered.
- Task 3B-add-preview: in progress (base `e2c1dd4`; brief `task-3b-add-preview-brief.md`)
- Task 3B-add-preview: complete (commits `2ce23d3..81cde95`, scoped re-review approved)
- Task 3B-add-preview evidence: focused suite 13/13; relevant on-add golden
  slice 3/3; tests-only changes.
- Task 3B-chain: pending (chain-generation and recovery contracts)
- Task 3B-chain: in progress (base `81cde95`; brief `task-3b-chain-brief.md`)
- Task 3B-chain: complete (commit `79b7f7d`, scoped re-review approved)
- Task 3B-chain evidence: focused suite 11/11; chain golden slice 71/72 with
  one unrelated pre-existing displacement-renderer failure; tests-only.
- Task 3B-config-time: pending (business calendar config and time utility contracts)
- Task 3B-config-time: in progress (base `79b7f7d`; brief `task-3b-config-time-brief.md`)
- Task 3B-config-time: complete (commits `9901533..848db3d`, scoped re-review approved)
- Task 3B-config-time evidence: focused suite 7/7; calendar golden slice
  29/30 with one unrelated displacement-renderer failure; tests-only.
- Task 3C-operator-commands: pending (query, doctor, reconcile contracts)
- Task 3C-operator-commands: in progress (base `848db3d`; brief `task-3c-operator-commands-brief.md`)
- Task 3C-operator-commands: complete (commits `41ce8c6..e6ee077`, scoped re-review approved)
- Task 3C-operator-commands evidence: focused suite 7/7; operator golden slice
  10/10; tests-only changes.
- Task 3C-deploy-reliability: pending (deployment sanity and reliability smoke contracts)
- Task 3C-deploy-reliability: in progress (base `e6ee077`; brief `task-3c-deploy-reliability-brief.md`)
- Task 3C-deploy-reliability: complete (commit `9d2b059`, scoped review approved)
- Task 3C-deploy-reliability evidence: focused suite 10/10; deployment golden
  slice 4/5 with one pre-existing dirty-checkout direct-json failure;
  tests-only changes.
- Task 3C-stress: pending (stress campaign profile/validation contracts)
- Task 3C-stress: in progress (base `9d2b059`; brief `task-3c-stress-brief.md`)
- Task 3C-stress: complete (commit `a34eb76`, scoped re-review approved)
- Task 3C-stress evidence: focused suite 8/8; relevant golden test 1/1;
  full discovery 609 tests with one unrelated recurrence-terminal failure;
  tests-only changes.
- Wave A-C direct-test migration: implementation slices complete; wave gate and
  fresh scan remain pending.
- Task 4-golden-integrity: in progress (base `a34eb76`; brief `task-4-golden-integrity-brief.md`)
- Task 4-golden-integrity: complete (commit `becd6dd`, scoped review approved)
- Task 4-golden-integrity evidence: registry tests 3/3; full discovery 612
  tests with one known recurrence-terminal failure; golden registry snapshot
  995 top-level / 983 registered / 12 explicitly retired.
- Task 6-operator-result: in progress (commits `97d2011..936b9e2`; review pending)
- Task 6-operator-result: complete (commits `97d2011..17eb325`, scoped re-review approved)
- Task 6-operator-result evidence: lifecycle + operator process suites 30/30;
  relevant reconcile/operator golden slice 42/42; mypy targets and diff checks
  pass; clean typed result boundary with no orphan helpers.
- Task 7-datetime-port: in progress (base `17eb325`)
- Task 7-datetime-port: complete (commits `bc0d7fd..dfdb316`, scoped re-review approved)
- Task 7-datetime-port evidence: focused regression suite 53/53; reconcile,
  transition, spawn, and operator paths pass; 11-file mypy check and diff
  checks pass. Ruling: commit contains adjacent lifecycle edits from the
  shared dirty worktree; preserve and audit them in final review.
- Task 8-typed-api-bindings: in progress (base `dfdb316`)
- Task 8-typed-api-bindings: complete (commits `22c9a5b..fcd1913`, scoped re-review approved)
- Task 8-typed-api-bindings evidence: all 13 factories use frozen
  `ApiBinding`; canonical parser cache fingerprints and runtime manifest are
  covered; public surface is pinned at 130 unique wildcard-importable names;
  focused API/cache/parser suites 41/41 and 16-file mypy checks pass.
- Task 9-architecture-contract: in progress (base `fcd1913`; brief `task-9-architecture-contract-brief.md`)
- Task 9-architecture-contract: complete (commit `7dc93ac`, scoped review approved)
- Task 9-architecture-contract evidence: explicit seven-layer AST dependency
  contract; deployment sanity and type-check CI wiring; invalid-fixture tests
  report importer, dependency, layer, and line; architecture scan 0 violations;
  focused architecture/deployment tests 14/14 and lifecycle boundary test 1/1.
  Full discovery 631 tests with the known pre-existing recurrence-terminal
  failure; deployment sanity retains the unrelated dirty-worktree direct-json
  finding.
- Task 10-core-context: in progress (commits `4dbb278`, `51f553f`, `5433586`)
- Task 10-core-context evidence: parser, scheduler, and cache factories now
  snapshot inputs into immutable dependency objects; cache directory and lock
  state are per-binding. Focused parser/scheduler/cache suite 29/29 passing.
  Remaining: owner-specific dependency objects and removal of residual
  string-keyed lookups.
- Task 10-core-context gate: combined parser, scheduler, cache, compatibility,
  and architecture suite 38/38 passing; compilation and diff checks clean.
- Task 10-core-context cache follow-up: cache garbage collection now uses the
  binding-local lock collaborator as well; cache contracts remain 10/10.
- Task 10-core-context cache state: `CacheState` now owns mutable memory
  entries and limits separately from immutable `CacheDependencies`; combined
  WP10 gate remains green at 39/39 focused tests.
- Task 10-core-context parser owner slice: commit `d9bcad1` injects an
  immutable `ParserOwnerDependencies` record into the pure DNF parser call;
  construction remains lazy and parser/compatibility tests pass 16/16.
- Task 10-core-context scheduler owner slices: commits `1d2beaa` and the
  pending interval-dependency commit isolate atom and interval collaborators;
  scheduler and exhaustion tests pass 10/10.
- Task 10-core-context scheduler composition slice: commit pending for
  `SchedulerModifierDependencies`; modifier, selection, and calendar-bound
  collaborators are now passed as one immutable owner record. Scheduler and
  exhaustion tests remain 14/14 passing.
- Task 10-core-context cutover audit: focused scheduler/cache/architecture
  gate 23/23 passing, but no-primary-lookup gate is not yet satisfied;
  `scheduler_api.py` retains 83 and `cache_api.py` retains 77 facade lookups.
  These are tracked as the remaining owner-specific migration, not waived.
- Task 10-core-context lookup migration: scheduler and cache factory-local
  namespaces now use immutable `deps` throughout; both files report zero
  primary `core[...]`/`core.get(...)` lookups. Combined gate 40/40 passing.
  Parser remains the final owner-specific lookup migration.
- Task 10-core-context: complete (commits `4dbb278`–`7e8221c`). Parser,
  scheduler, and cache owner boundaries now use immutable dependency snapshots;
  all three primary APIs report zero `core[...]`/`core.get(...)` lookups.
  Final cutover suite 41/41 passing and architecture scan 0 violations.
- Task 10-core-context polish: added executable no-primary-lookup regression
  coverage plus scheduler/cache snapshot immutability tests; polish suite
  28/28 passing. Broad `Any`-to-Protocol typing remains optional future work.
- Task 11-hook-host-isolation: in progress. Removed lifecycle-read repository
  rebinding and introduced frozen `LifecycleReadCapabilities`; focused modify,
  lifecycle, and operator tests remain green at 31/31.
- Task 11 capability narrowing: completion flow now receives
  `seed_runtime_lookup_tasks` directly from `ModifyRuntimeServices`, removing
  one capability-bag/host reach-through. Focused suite remains 31/31.
- Task 11 route callback migration: completion flow now receives the lifecycle
  read service directly from `ModifyRuntimeServices`; focused suite remains
  31/31. Additional diagnostic, presentation, validation, and scheduling
  callbacks remain to be migrated.
- Task 11 diagnostic/presentation/validation migration: completion flow now
  receives diagnostics, renderers, task printing, transition, and validation
  callbacks from `ModifyRuntimeServices`; focused suite remains 31/31.
- Task 11 scheduling migration: completion flow now receives
  `compute_next_and_limits` through `ModifyRuntimeServices`; focused suite
  remains 31/31. Remaining host parameters in lower-level route helpers still
  require a separate staged migration before WP11 is complete.
- Task 11 lower-level helper slice: removed the unused host parameter from
  `modify_schedule_effects.recurrence_seed_base` and updated all callers;
  scheduler/modify/operator tests pass 40/40.
- Task 11 focused port extraction: `cp_add_period` now consumes immutable
  `SchedulePorts` instead of a host object; direct port coverage added and the
  focused slice passes 15/15.
- Task 11 scheduling port follow-up: `sequence_period_for_link` now consumes
  an explicit `SequencePorts` collaborator; combined modify/scheduler/operator
  slice passes 42/42.
- Task 11 datetime port extraction: comparison is now supplied through frozen
  `DatetimePorts` across scheduling, validation, and completion helpers;
  focused modify/operator suite passes 33/33.
- Task 11 validation port extraction: `chain_duration_reasonable` now consumes
  frozen `DurationPorts`; completion wiring and modify validation tests pass
  33/33.
- Task 11 until-validation extraction: `until_not_past` now consumes frozen
  `UntilPorts`; completion and validation wiring remain green at 33/33.
- Task 11 presentation callback follow-up: remaining `on_time_delta` callers
  now receive `HumanDeltaPort`; diagnostics/presentation regression suite
  remains green at 33/33.
- Task 11 validation presentation port: `anchor_mode` now consumes frozen
  `AnchorModePorts` for warnings; modify validation/operator tests pass 33/33.
- Task 11 shared validation port: anchor/omit validation now consumes frozen
  `SharedValidationPorts`; transition and modify validation tests remain 33/33.
- Task 11 CP validation port: `validate_cp` now consumes frozen
  `CPValidationPorts`; focused validation/transition/operator tests pass 32/32.
- Task 11 read port: optional-token parsing now consumes frozen
  `ExtraTokenPort`; lifecycle/modify/operator tests pass 33/33.
- Task 11 token matcher port: lifecycle token matching now receives only the
  integer-coercion callable; focused lifecycle/modify/operator tests pass
  33/33.
- Task 11 chain-limit validation batch: `validate_chain_limits` now consumes
  frozen `ChainLimitPorts`; modify/transition/operator tests pass 33/33.
- Task 11 scheduling batch 1: occurrence projection now consumes explicit
  `OccurrencePorts`; modify diagnostics/presentation/operator tests pass
  33/33.
- Task 11 diagnostics batch 2: diagnostic datetime parsing now consumes frozen
  `DatetimeValuePort`; diagnostics/isolation/operator tests pass 31/31.
- Task 11 native-until validation batch: `validate_native_until` now consumes
  frozen `NativeUntilPorts`; composition and modify validation tests pass
  33/33.
- Task 11 native-until slot batch: calendar-slot validation now consumes
  frozen `NativeUntilSlotPorts`; modify validation/composition tests pass
  33/33.
- Task 11 diagnostics batch: chain health and integrity analysis now consume
  explicit frozen `AnalyticsPorts`; focused modify/operator tests pass 33/33.
- Task 11 completion snapshot batch: chain snapshot reads now consume frozen
  `SnapshotPorts`, isolating repository and model dependencies; focused tests
  pass 33/33.
- Task 11 time-slot batch: anchor slot normalization now consumes frozen
  `TimeSlotPorts`; focused modify/operator tests pass 33/33.
- Task 11 omission-state batch: parent omit expression/file loading now consumes
  frozen `OmitPorts`; completion, workflow, carry, isolation, and validation
  tests pass 44/44.
- Task 11 command boundary batch: Taskwarrior subprocess execution and UUID
  availability checks now consume frozen `CommandPorts`; completion/workflow,
  carry, isolation, and validation tests pass 44/44.
