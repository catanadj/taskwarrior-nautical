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
- Task 11 generation boundary batch: chain-generation service caching now
  consumes frozen `GenerationPorts`; all affected completion/workflow and
  isolation tests pass 44/44.
- Task 11 query boundary batch: chain-root/age caching and formatting now
  consume frozen `QueryPorts`; completion/workflow and isolation tests pass
  44/44.
- Task 11 spawn identity batch: lifecycle parent/child identity derivation now
  consumes frozen `SpawnIdentityPorts`; completion/workflow/isolation tests
  pass 44/44.
- Task 11 carry rejection batch: native-until carry error reporting now
  consumes frozen `NativeCarryPorts`; completion/workflow/isolation tests pass
  44/44.
- Task 11 scheduler callback batch: scheduler service/evaluator construction
  now consumes frozen `SchedulerPorts`; focused lifecycle tests pass 44/44.
- Task 11 chain export batch: required chain reads now consume explicit
  `ChainExportPort`; diagnostics and lifecycle tests pass 45/45.
- Task 11 regression pass: compilation, focused lifecycle suites, and diff
  checks are clean after the three-pass migration loop.
- Task 11 seed lookup batch: runtime lookup seeding now consumes frozen
  `SeedLookupPorts`; isolation/workflow/completion tests pass 30/30.
- Task 11 predecessor batch: previous-chain collection now consumes frozen
  `PreviousChainPorts`; diagnostics/workflow/isolation tests pass 31/31.
- Task 11 three-pass regression: full focused lifecycle/operator slice passes
  71/71 with clean diff checks.
- Task 11 anchor projection batch: included-occurrence lookup now consumes
  `AnchorOccurrencePorts`; scheduling/completion/isolation tests pass 30/30.
- Task 11 spawn staging batch: lifecycle outbox staging now consumes explicit
  `SpawnIntentPorts`; completion/workflow/isolation tests pass 30/30.
- Task 11 five-pass verification: compiled schedule/spawn routes and reran the
  focused migration suites successfully.
- Task 11 completion preflight batch: link validation, chain identity, and
  existing-next checks now consume frozen `CompletionPreflightPorts`; focused
  completion/workflow/isolation tests pass 30/30.
- Task 11 native carry batch: native-until target preservation now consumes
  frozen `NativePreservePorts`; workflow/carry/isolation tests pass 40/40.
- Task 11 hook-host architecture gate: added a fixture-backed AST contract that
  rejects `host` parameters on modify effect operations while permitting only
  explicitly named composition constructors (`*_port_for`, `*_ports_for`,
  `*_services_for`, and `*_for_host`). The contract's focused fixtures pass; the repository
  gate intentionally remains red with 47 operations until the remaining
  presentation, transition, completion, diagnostic, read, schedule, spawn,
  validation, format, and route migrations finish.
- Task 11 schedule/spawn/format batch: schedule cap/estimate operations now
  consume `CPCompletionPorts` or `AnchorCompletionPorts`; atomic spawn and UUID
  derivation consume `SpawnChildPorts` or `ChildUuidPorts`; line previews
  consume `LinePreviewPorts`. The architecture gate reports zero hook-host
  violations in these three owners; compilation, 26 focused tests, and diff
  checks pass. Combined completion verification awaits reconciliation of the
  concurrent `CompletionComputePorts` caller migration.
- Task 11 golden adapter cutover: `_BoundCompletionEffects` now constructs the
  explicit preflight, compute, spawn, and snapshot ports required by migrated
  operations. Golden schedule and spawn callers no longer pass a hook host,
  and lifecycle-read fixtures use the explicit for-host composition adapter.
  Six focused schedule/spawn/completion golden cases pass; legacy-call scans,
  compilation, architecture checks for all three owners, and diff checks pass.
- Task 11 transition/carry cutover: completion recurrence validation now
  consumes frozen `CompletionValidationPorts`; CP and native carry ports are
  assembled once at the composition root, and `modify_transition_effects` has
  no hook-host or dynamic-module dependency. Focused workflow, carry,
  completion, isolation, and operator tests pass 69/69.
- Task 11 capability-bag cutover: removed `ModifyRuntimeServices.capabilities`
  and replaced it with operation-scoped non-completion, completion, and
  deletion capability sets. The isolation architecture gate and combined
  presentation/lifecycle slice pass 76/76; remaining AST findings are tracked
  for subsequent adapter extraction.
- Task 11 completion compute/guard cutover: next-occurrence computation now
  consumes frozen `CompletionComputePorts` and `CompletionLifecyclePlanPorts`;
  the completion-kind guard reuses `CompletionPreflightPorts`, and host-aware
  assembly occurs once in the named composition adapter. Focused completion,
  workflow, carry, and operator tests pass 66/66.
- Task 11 completion feedback/preflight/spawn cutover: feedback callbacks are
  named composition ports, while preflight context and child spawning consume
  frozen `CompletionPreflightContextPorts` and `CompletionSpawnPorts`.
  `modify_completion_effects.py` now has zero hook-host architecture
  violations; focused completion/workflow/carry/operator tests pass 66/66.
- Task 11 remaining hook-host finding batch: renamed the remaining hook-facing
  route/presentation adapters to explicit `*_for_host` boundaries, converted
  anchor and omit validation to frozen ports, and repaired native-until carry
  adapter parsing so it passes parsed datetimes (including conflict evidence).
  Updated the golden test-only bindings to the renamed boundaries and current
  scheduler port signature. Architecture contract reports 0 hook-host
  violations; focused unit slice passes 96/96 and the route-focused on-modify
  golden slice passes 30/30. Full unittest discovery is 651/652; the sole
  failure is the unrelated `collect_after_cursor` terminal-evidence test. The
  broader `--only on_modify` golden slice is still not clean (116/143), so WP11
  remains in progress; the effect-adjacent host adapters still need a final
  composition-boundary review and the remaining golden contracts need repair.
- Task 11 final cutover and verification: centralized hook-aware route and
  presentation assembly in `modify_composition_adapters.py`, removed the
  former `modify_effects.py` wrapper, and built `LifecycleReadService` at the
  composition root. Fixed the UI-port caller, command-port execution, direct
  datetime comparison imports, and timeline omission evidence exposed by the
  complete golden run. The architecture contract reports zero hook-host
  violations. Fresh verification: on-modify golden tests 143/143; focused
  architecture/hook/lifecycle tests 33/33; deployment sanity and compilation
  pass; `git diff --check` is clean. Full discovery runs 652 tests with one
  unrelated failure in `test_recurrence_cursor_terminal.CursorTerminalEvidenceTests.test_collect_after_cursor_preserves_terminal_evidence`
  (terminal evidence is dropped by `collect_after_cursor`). WP11 is complete;
  that cursor-terminal behavior remains outside this work package.
- Task 11 hardening pass: removed the generic `module_loader` capability from
  `TimelineServices`, which had allowed the renderer to request hook modules
  after composition. The composition adapter now supplies only typed omission
  predicate/description callbacks; a focused regression test proves the generic
  loader is absent. The test was observed failing before the cutover and passes
  after it. Fresh verification: focused architecture/hook/lifecycle tests
  34/34, on-modify golden tests 143/143, deployment sanity passes, and
  `git diff --check` is clean.
- Full-suite cursor-terminal defect fix: `RecurrenceEvaluator.collect_after_cursor`
  previously reduced `OccurrenceCursor` to its datetime and then rebuilt an
  `OccurrenceBatch` without the source batch's terminal evidence. It now passes
  the cursor contract intact to `collect_after` and carries terminal evidence
  into the returned batch; datetime-based calls retain the prior strict-after
  default. The focused cursor suite passes 2/2, cursor/parity golden tests 5/5,
  and full unittest discovery passes 653/653.
- Task 12 lifecycle execution boundary: replaced the dependency-derived optional
  capability bag with an explicit `LifecycleExecutionPort`, sorted composition-
  time validation, and a deterministic stage-only execution error before
  session setup or claims. On-exit and reconcile pass the concrete mutation
  service; on-modify remains intentionally stage-only under Taskwarrior's lock.
  Golden test doubles now use a dedicated complete-port fixture, with the batch
  lease-expiry regression updated to exercise the batched path. No installed-
  layout legacy caller was found, so no dynamic compatibility adapter was
  added. Verification: capability 5/5, lifecycle failure injection 23/23,
  effect boundary 13/13, lifecycle golden 44/44, and full unit discovery
  658/658.
- Task 12 hardening review: found that a complete batch execution port could
  still be composed with a non-null but incomplete direct mutation gateway.
  The application service calls `apply` and guarded child compensation outside
  the batch port, so it now validates both methods at construction and reports
  missing names before any claim. Added a red/green contract test; updated the
  dedicated golden fixture to satisfy both explicit contracts. Fresh checks:
  capability/effect tests 19/19, failure-injection 23/23, lifecycle golden
  44/44, full unit discovery 659/659, deployment sanity, compilation, and
  `git diff --check` all pass.
- Task 12 outbox hardening: execution-capable composition now validates the
  single enqueue operation, invocation session, and all atomic wave-storage
  methods. Drain and reconcile paths call those methods directly; failed bulk
  renewal/stage/ack operations remain visible instead of silently downgrading
  to per-intent updates. Updated the persistence-fault golden fixture to use a
  complete real-repository contract and replaced an execution test's partial
  placeholder with a real outbox. Fresh verification: focused capability,
  failure-injection, and effect-boundary tests 43/43; full unittest discovery
  660/660; lifecycle golden tests 44/44; deploy sanity, compilation, and
  `git diff --check` pass.
- Task 12 final contract review: the first outbox gate covered wave/session
  operations but omitted direct single-record claim, stage, lease, retry, and
  review methods used by one-record drains and recovery. Expanded composition
  validation to the full directly-used execution surface; stage-only
  composition now also rejects an outbox without `enqueue`. Added focused
  regression cases for both gaps and updated the package-12 contract text.
  Fresh verification: focused lifecycle/effect tests 45/45, full unit discovery
  662/662, lifecycle golden 44/44, deployment sanity, compilation, and
  `git diff --check` all pass.
- Task 13 presentation contexts: replaced `AnchorPreviewServices` with
  `AnchorExpressionPreviewServices`, removed `core` from both public preview
  handlers and all four focused preview/timeline contexts, and split timeline
  projection from formatting. Timeline projection now receives the configured
  evaluator and scheduler callbacks directly; the add-preview collector no
  longer carries legacy evaluator/callback fallback arguments. The wall-clock
  fixture helper was confined to `dev_tools/legacy_preview_adapter.py`.
  `modify_presentation_effects.py` was reviewed and remains focused on
  chain-style ports; service assembly belongs at the composition adapters.
  Added a runtime usage matrix for expression preview, anchor-file preview, CP
  timeline, and anchor timeline, plus Unicode renderer coverage.
  Verification: focused required/context suites 49/49; full unit discovery
  667/667; selected preview/timeline/timezone golden slice 112/112; deployment
  sanity, compilation, and `git diff --check` pass. No commit was created;
  changes remain local in the active branch.

## WP4 reconciliation — 2026-09-13

- Current evidence supersedes the earlier WP4 wording: parser front-end and
  parser-atom direct tests, scheduler exhaustion/long-interval tests, cache
  contracts, on-add route contracts, chain graph/planner tests, and broad
  lifecycle/operator/backup/restore unit suites now exist. The historical
  description of scheduler/cache as merely underway and backup/restore as
  unstarted is stale.
- Registry snapshot now records 920 top-level functions, 908 registered, 12
  explicit retirements, and 0 duplicate registrations. Lifecycle model and
  operator v2 contract migrations, plus lifecycle planner/terminal policy and
  operator presentation contracts, plus hook-protocol and TaskDocument direct
  contracts, plus query model and in-process command contracts, bring the WP4
  migration total to 76 golden functions.
- WP4 remains incomplete. Parser validation/front-end/atom,
  scheduler/occurrence and cache, on-add route, lifecycle/operator, and
  backup/restore direct suites now exist, but a related unit suite does not by
  itself complete a domain migration. The remaining golden scenarios require
  classification, direct-contract migrations where justified, per-domain
  unit/golden inventory and retained-acceptance rationale, plus normal and
  shuffled golden-run evidence. WP17 remains deferred.
- Task 4 parser-validation batch: complete. The valid/invalid parser matrix,
  yearly-token error surfaces, and yearly-format helper contracts now run in
  `tests/recurrence/test_yearly_token_migration.py`; no production behavior
  changed and no commit was created. Brief:
  `task-4-parser-validation-brief.md`.
- Task 4 scheduler cross-path batch: complete. The ordinary, sparse, interval,
  AND/OR, omission, seeded-random, and overnight matrix now runs in
  `tests/recurrence/test_scheduler_cross_path_conformance.py`; next, collect,
  preview, and range results agree, streams remain monotonic, and repeated
  collection is deterministic. No production behavior changed and no commit
  was created. Brief: `task-4-scheduler-cross-path-brief.md`.
- Task 4 scheduler stream hardening: complete. Collection and bounded-range
  paths now compare complete signature streams; preview remains a single-result
  contract. Scoped review found no concrete defects. Focused/shuffled scheduler
  tests, scheduler golden slice 26/26, registry integrity, and `git diff
  --check` passed. Report: `task-4-scheduler-stream-hardening-report.md`.
- Task 4 lifecycle/operator pure-contract batch: complete after scoped review.
  Four direct-contract goldens now have equivalent `unittest` coverage; the
  reviewer approved and the full normal golden runner passed 933/933. Focused
  direct/registry tests passed 23/23, lifecycle slice 40/40, operator slice
  7/7, and `git diff --check` passed. WP4 classification, domain inventory,
  retained-acceptance rationale, full shuffled run, and final suite evidence
  remain open; WP17 is deferred.
- Task 4 hook-protocol and TaskDocument batch: direct `unittest` contracts now
  cover the twelve migrated pure golden functions. The golden runner retains
  isolated-load and executable/bootstrap/output/permission acceptance cases.
  No production behavior changed and no commit was created. Brief:
  `task-4-hook-protocol-and-task-document-brief.md`.
- Task 4 query model and command-contract batch: four direct query golden
  contracts now run in `tests/test_query_models.py` and
  `tests/test_query_command_contracts.py`. The golden runner retains query
  process, installed-layout, and concurrent-Taskdata acceptance coverage.
  Focused direct/registry tests passed 7/7, the retained query slice passed
  4/4, and `git diff --check` passed. No production behavior changed and no
  commit was created. Report:
  `task-4-query-model-and-command-contracts-report.md`.
- Task 4 query-service direct-contract batch: nine pure read-only occurrence
  and `next` service contracts now run in `tests/test_query_service_contracts.py`.
  They cover scheduler parity, omission/cap behavior, typed absent/unavailable
  reads, selector filtering and batching, per-task validation, due-bounded and
  CP projection, chain bounds, and daily skip-mode evidence. Query process,
  installed-layout, and concurrent-Taskdata acceptance cases remain golden.
  Registry snapshot: 920 top-level / 908 registered / 12 retired / 0 duplicate;
  WP4 migrated total: 76. No production behavior changed and no commit was
  created. Base: `1c7b9ef6ccd2341d87786d82e34b559dbc62dcea`. Implementation:
  added the direct module; removed only the nine mapped golden definitions and
  registry entries; updated registry integrity, inventory, and report.
  Evidence: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python3 -m unittest
  tests.test_query_service_contracts tests.test_golden_registry_integrity -v`
  passed 12/12 in 0.898s; `PYTHONDONTWRITEBYTECODE=1 python3
  dev_tools/nautical_golden_tests.py --only query_process_boundary --only
  operator_processes_concurrent_contracts --only query_installed_layout
  --verbose` passed 3/3; `git diff --check` exited 0 with no output.
  Scoped review initially treated pre-existing, already-reviewed query-model
  changes as part of this task because the shared worktree is uncommitted. A
  corrected scope artifact and re-review verified the earlier model-contract
  replacement and the 929/917-to-920/908 nine-test delta; both findings were
  marked addressed, with no new issue. Controller assertion-level review then
  found that the UUID batching test did not pin one broad snapshot and the
  `next` projection test did not assert source-row immutability. Reopened for a
  focused hardening fix before task completion.
- Task 4-query-service fix round 1/5: added an exact one-snapshot assertion and
  made per-UUID reads fail; asserted that anchor and CP source mappings remain
  unchanged after `query_next`. Direct and registry suites pass 12/12 in
  0.935s; `git diff --check` is clean. No production or registry changes.
- Task 4-query-service: complete (task review findings verified as addressed;
  controller assertion hardening passes; no commit was created).
- Task 4 lifecycle planner/helper batch: five pure candidate-planning,
  completion/reconcile parity, expiration-basis, recurrence-matrix, and
  idempotent terminal-patch contracts now run in
  `tests/test_lifecycle_terminal_plans.py`. The golden definitions and registry
  entries were removed; lifecycle runtime and Taskwarrior round-trip cases
  remain in the acceptance runner. Registry snapshot is now 915 top-level /
  903 registered / 12 retired / 0 duplicate; cumulative WP4 migrations: 81.
  Focused direct and registry tests passed 21/21; retained real-stack and
  Taskwarrior round-trip golden cases passed 2/2; golden module compilation and
  `git diff --check` passed. Report:
  `task-4-lifecycle-planner-direct-contracts-report.md`. No production behavior
  changed and no commit was created.
- Task 4 lifecycle read-service batch: three pure index/merge, safe full-snapshot
  filtering, and chain-cache contracts now run in
  `tests/test_lifecycle_read_service.py`. The snapshot contract uses a fake
  repository that fails if the fallback read occurs; this replaces the former
  disconnected exporter counter. Registry snapshot is now 912 top-level / 900
  registered / 12 retired / 0 duplicate; cumulative WP4 migrations: 84.
  Focused direct tests passed 4/4. Full unittest discovery and normal/shuffled
  golden gates must be rerun after the remaining WP4 reconciliation work.
  Report: `task-4-lifecycle-read-service-report.md`. No production behavior
  changed and no commit was created.
- Whole-suite checkpoint before lifecycle read-service migration: 759 unittest
  tests passed in 77.579s; normal golden 903/903 and seeded shuffle
  `20260811` passed 903/903. These gates predate the latest three-case migration
  and do not close the still-open WP4 classification/domain-inventory work.
- WP4 classification audit probe: AST boundary-marker scan found 91 of 912
  top-level tests with obvious subprocess/Taskwarrior/process-style markers;
  the other 821 were only review candidates, not certified unit tests. This
  heuristic is not accepted as final classification. WP17 remains deferred.
- Task 4 chain-integrity model/snapshot batch: two pure contracts moved to
  `tests/test_chain_integrity_models.py`, covering incomplete repair identity,
  dependency/frozen-plan validation, authoritative cache epoch behavior, and
  fail-closed malformed/truncated/duplicate/mismatched snapshots. Registry is
  now 910 top-level / 898 registered / 12 retired / 0 duplicate; cumulative
  WP4 migrations: 86. Focused direct tests pass 2/2. Report:
  `task-4-chain-integrity-models-report.md`. Whole-suite gates are pending after
  this and later migrations; no production behavior changed and no commit was
  created.
- Task 4 chain-integrity engine/invariant batch: five pure audit, report parity,
  bounded hydration, ownership-registry, and invariant-rule contracts now run
  in `tests/test_chain_integrity_engine.py`. Registry is 905 top-level / 893
  registered / 12 retired / 0 duplicate; cumulative WP4 migrations: 91.
  Focused direct suite passed 5/5 before the final invariant addition. Rerun it
  with registry validation; whole-suite gates remain pending. Report:
  `task-4-chain-integrity-engine-report.md`. No production behavior changed or
  commit created.
- WP4 chain-integrity follow-up added direct checks for outbox/graph provenance,
  acknowledged lifecycle postconditions, and integrity-application refusal and
  delegation. Together with the model/snapshot and engine/invariant migrations,
  13 pure chain-integrity golden cases have been removed. Current registry is
  902 top-level / 890 registered / 12 retired / 0 duplicate; cumulative WP4
  direct migrations: 94. Focused chain-integrity direct/registry tests passed
  13/13. Whole-suite gates will be rerun after further classification and
  migration; WP4 remains incomplete and WP17 deferred.
- Task 4 task-domain model/codec batch: seven pure typed read, observation,
  projection, presentation view, codec, hook-framing, and draft/patch contracts
  moved to `tests/test_task_domain_models.py`. Registry is now 895 top-level /
  883 registered / 12 retired / 0 duplicate; cumulative WP4 direct migrations:
  101. The focused model/lifecycle/integrity/registry group passed 42/42;
  golden module compilation and `git diff --check` passed. Whole-suite gates
  remain pending after the ongoing WP4 classification/migration work. Report:
  `task-4-task-domain-models-report.md`. No production behavior changed or
  commit created.
- Task 4 task-read repository batch: nine snapshot, set-read, cache/fallback,
  malformed-output, mutation-epoch, and domain-read contracts now run in
  `tests/test_task_read_repository_contracts.py`. Taskwarrior command execution
  remains at the golden acceptance boundary. Registry is 886 top-level / 874
  registered / 12 retired / 0 duplicate; cumulative WP4 migrations: 110.
  Focused model/lifecycle/integrity/registry suite passed 51/51.
- Task 4 unit-of-work batch: three in-memory read-cache scope, explicit broad
  coverage, and invocation isolation contracts now run in
  `tests/test_taskwarrior_uow_contracts.py`; real command/retry/timeout and
  budget diagnostics remain golden. Registry is 883 top-level / 871 registered
  / 12 retired / 0 duplicate; cumulative WP4 migrations: 113. Focused suite
  passed 54/54; golden module compiles and `git diff --check` passes. Reports:
  `task-4-task-read-snapshot-contracts-report.md` and
  `task-4-taskwarrior-uow-report.md`. No production behavior changed and no
  commit was created.
- WP4 full-gate checkpoint after 113 migrations: standard unittest discovery
  passed 791 tests in 54.524s; normal and shuffled golden runs both passed
  871/871 with shuffle seed `20260811`. Registry remains 883 top-level / 871
  registered / 12 retired / 0 duplicate. `git diff --check` and golden module
  compilation pass. Scenario-level direct/acceptance classification and
  per-domain runtime/count inventory remain open, so WP4 is not ready to close
  and WP17 remains deferred.
- Current AST marker scan over the 883 top-level golden functions: 58 have
  direct subprocess-call markers, 259 dynamically load hook/core modules, 10
  mention repository/task-I/O methods, and 251 use temporary filesystem or
  Taskdata fixtures. Markers overlap and only guide manual review; they do not
  certify acceptance classification.
- Navigator direct-view batch: ten metadata/query parity, stable task metadata,
  calendar, summary, chain choice, change row, detail, projection, trace, and
  aggregate serialization contracts now run in
  `tests/test_navigator_view_models.py`; their golden definitions and registry
  entries are removed. Registry-integrity coverage now distinguishes the 12
  retained-unregistered characterization functions from the 10 migrated and
  deleted functions. Registry is 873 top-level / 861 registered / 12 retained
  characterization / 10 migrated-and-removed / 0 duplicate. The new direct
  module passes 10/10; registry tests pass; remaining Navigator golden filter
  passes 20/20. Full current gates: unittest discovery 802 tests in 54.061s,
  normal golden 861/861, shuffle seed `20260811` 861/861, and `git diff
  --check`. Navigator direct module measured 0.119s; remaining name-filtered
  Navigator golden slice 0.846s. WP4 still remains open: full per-case
  classification and authoritative per-domain inventory are not yet complete.
- CP recurrence batch: four direct contracts now cover duration and sequence
  parsing, repeat/link boundaries, deterministic random/jitter selection and
  chain scope, and local wall-clock preservation over a DST transition in
  `tests/recurrence/test_cp_sequence_contracts.py`. Their four golden functions
  and registrations were removed; the on-add/on-modify cross-path comparison
  remains acceptance/conformance coverage. Registry is 869 top-level / 857
  registered / 12 retained-unregistered characterizations / 14 migrated and
  removed / 0 duplicates. Direct CP tests pass 4/4; direct-plus-registry tests
  pass 8/8; the CP name-filtered golden slice passed 27/27. Fresh full gates
  then passed: unittest discovery 806 in 57.147s, normal golden 857/857, and
  seeded shuffle `20260811` 857/857. Golden module and migrated tests compile;
  `git diff --check` passes. Cumulative WP4 direct migrations: 127. WP4
  classification and authoritative per-domain inventory remain open; WP17
  remains deferred.
- Astronomy/year-ordinal direct-contract batch: four deterministic astronomy
  grammar/configuration/phase-math contracts moved to
  `tests/test_astronomy_contracts.py`. Nine year-day and ISO-week ordinal
  validation, expansion, scheduling, interval, random/omit, documented-example,
  and serialization cases moved into `tests/recurrence/test_yearly_token_migration.py`.
  Provider-dependent astronomy smoke and hook/timeline ordinal integration
  remain in golden acceptance. Current registry is 856 top-level / 844
  registered / 12 retained-unregistered characterizations / 27 migrated and
  removed / 0 duplicates. Focused suites and registry integrity pass 23/23;
  the remaining ordinal-filter golden slice passes 2/2. Cumulative direct
  migrations: 140. Fresh full gates then passed: unittest discovery 819 tests
  in 54.749s; normal and shuffled golden runs 844/844 with seed `20260811`;
  golden and changed test modules compile; `git diff --check` passes. WP4
  classification and per-domain inventory remain open, WP17 deferred.
- File-backed recurrence batch: basename safety for anchor/omit files, omit CSV
  header lookup/date deduplication/description mapping, and anchor-file
  time/offset/window parsing now run in `tests/test_file_backed_contracts.py`.
  Eight golden parser/input cases were removed; file occurrence expansion,
  provider ordering/cursor, and Taskwarrior-facing recurrence behavior remain
  golden. Registry is 848 top-level / 836 registered / 12 retained-unregistered
  characterizations / 35 additional direct migrations this cycle / 0
  duplicates. Direct plus registry tests pass 8/8, and selected retained
  acceptance cases pass 2/2. Cumulative direct migrations: 148. Fresh full
  gates pass: unittest discovery 823 tests in 54.944s; normal and shuffled
  golden 836/836 with seed `20260811`; golden and migrated test modules compile;
  `git diff --check` passes. WP4 classification and per-domain inventory
  remain open, WP17 deferred.
- Seasonal direct-contract batch: twelve deterministic fixed/astronomical
  calendar, selector parsing, ACF/cache round-trip, generic and scoped
  scheduling, modifier-boundary, overflow, and semantic-guard scenarios now
  run from `tests/recurrence/test_season_calendar_contracts.py`. The twelve
  corresponding golden definitions and registrations were removed; the
  remaining provider and hook/on-add/on-modify/reconcile integration cases
  remain acceptance coverage. Registry: 836 top-level / 824 registered / 12
  retired / 47 newly migrated in this WP4 cycle / 0 duplicates; cumulative WP4
  direct migrations: 160. Focused direct-plus-registry tests pass 16/16, the
  remaining seasonal golden slice passes 5/5, full unittest discovery passes
  835 in 55.140s, and normal plus shuffled golden runs pass 824/824 with seed
  `20260811`. Python compilation passes. `git diff --check` still reports
  trailing whitespace in unrelated pre-existing hunks of the already-dirty
  golden module; this batch did not alter those lines. WP4 classification and
  authoritative per-domain inventory remain open; WP17 remains deferred.
- Business-calendar direct-contract batch: three characterization contracts
  for default weekday rolls/offsets/ordinals, normalized immutable calendar
  definitions, and unstable-rule/unmatched-file rejection now run from
  `tests/test_business_calendar_contract.py`; the old golden definitions and
  registrations were removed. The remaining scheduler, file, displacement,
  hook, Taskdata/TOML, and reconcile scenarios remain golden acceptance.
  Current registry: 833 top-level / 821 registered / 12 retired / 50 newly
  migrated this WP4 cycle / 0 duplicates; cumulative WP4 direct migrations:
  163. Focused direct-plus-registry tests pass 12/12, the broad business-
  calendar golden filter passes 18/18, full unittest discovery passes 836 in
  59.270s, and normal plus seeded-shuffle golden runs pass 821/821 with seed
  `20260811`. Python compilation passes. `git diff --check` still reports
  trailing whitespace in unrelated pre-existing hunks of the dirty golden
  module. WP4 classification and per-domain inventory remain open; WP17 stays
  deferred.
- Business-calendar follow-up migrated four more behavior-level contracts:
  custom calendar flow through scheduler/filter/roll/offset/ordinal/random
  selection, rules/files resolution and omission precedence, injected-calendar
  anchor/omit modifiers, and displacement capture. Their golden definitions
  and registrations were removed; direct tests plus registry integrity pass
  16/16. Current registry: 829 top-level / 817 registered / 12 retired / 54
  newly migrated in this WP4 cycle / 0 duplicates; cumulative WP4 direct
  migrations: 167. Fresh full gates pass: unittest discovery 840 tests in
  55.908s, normal golden 817/817, and deterministic shuffled golden (seed
  `20260811`) 817/817. Changed Python modules compile. Whole-worktree
  `git diff --check` still reports trailing whitespace in unrelated
  pre-existing hunks of the dirty golden module. WP4 classification and
  per-domain inventory remain open; WP17 remains deferred.
- File-backed/provider contract batch moved seven deterministic anchor-file
  input/provider behaviors into `tests/test_file_backed_contracts.py` and nine
  typed-provider/collector behaviors into
  `tests/test_occurrence_provider_contracts.py`. The corresponding 16 golden
  copies and registrations are removed; the remaining file-backed golden
  slices are cross-provider, DST integration, and hook acceptance checks.
  Registry is 813 top-level / 801 registered / 12 retired / 70 newly migrated
  in this WP4 cycle / 0 duplicates; cumulative direct migrations: 183. Focused
  direct-plus-registry coverage passes (file-backed 15/15; provider 13/13),
  anchor-file acceptance 43/43, provider acceptance 6/6, and collection
  acceptance 1/1. Fresh full gates: unittest discovery 856 tests in 55.207s;
  normal golden 801/801; deterministic shuffled golden (seed `20260811`)
  801/801. Changed Python modules compile. Whole-worktree `git diff --check`
  still flags trailing whitespace in unrelated pre-existing changed hunks of
  the golden module. WP4 case classification and per-domain inventory remain
  open; WP17 stays deferred.
- Occurrence-provider follow-up migrated four additional direct contracts:
  ordinary lazy-anchor typed values, DST-fold backward-progress rejection,
  anchor-file cursor-cache parity against fresh lookups, and provider batch
  generation against repeated strict `next_after` calls. The old batch test
  compared two invocations of the same batch path; the direct version now uses
  an independent repeated-successor reference. Focused direct-plus-registry
  tests pass 17/17; remaining anchor-file and provider golden slices pass 41/41
  and 4/4. Registry is 809 top-level / 797 registered / 12 retired / 74 newly
  migrated in this WP4 cycle / 0 duplicates; cumulative direct migrations:
  187. Fresh full gates: unittest discovery 860 tests in 55.323s, normal golden
  797/797, and shuffled golden (seed `20260811`) 797/797. Python compilation
  passes. Whole-worktree `git diff --check` still reports unrelated trailing
  whitespace in pre-existing changed hunks of the golden module. WP4
  classification and per-domain inventory remain open; WP17 stays deferred.
- Time-window and parser-owner batch moved fourteen DNF/ACF, random-window,
  grouped/composable schedule, cache-shape, bounded-language, quarter-rewrite,
  and description-alias contracts into the standard parser-owner and
  time-window suites. The natural-language time-window case had been in the
  explicit retired set; it is now a live direct test, so retired cases drop
  from 12 to 11. Focused direct plus registry tests pass 29/29; remaining
  time-window acceptance passes 11/11, composable-schedule filter has no
  acceptance cases, and quarter preview cases pass 2/2. Registry is 778
  top-level / 767 registered / 11 retired / 105 newly migrated in this WP4
  cycle / 0 duplicates; cumulative direct migrations: 218. Full gates are
  pending refresh after this batch. WP4 case classification and per-domain
  inventory remain open; WP17 stays deferred.
- Completed direct parser-boundary migration for the remaining deterministic
  time-window/ACF behavior: partition and overnight metadata, seeded-random
  identity/context, hour-only lists, composable/grouped schedules and offsets,
  natural-language bounds, contradictory timing rejection, and cached DNF
  shape validation. These fourteen cases now live in
  `tests/test_parser_owner_api_contracts.py`; hook, DST, process-stability,
  and Navigator cases remain golden. Focused parser/time-window/registry suite
  passes 29/29; remaining time-window acceptance passes 11/11 and quarter
  preview acceptance passes 2/2. Registry is 778 top-level / 767 registered /
  11 retired / 105 newly migrated in this WP4 cycle / 0 duplicates; cumulative
  direct migrations: 218. Full gates pass: unittest discovery 882 tests in
  71.108s, normal golden 767/767, and shuffled golden (seed `20260811`)
  767/767. Changed Python modules compile. Whole-worktree `git diff --check`
  still flags trailing whitespace in unrelated pre-existing changed hunks of
  the golden module. WP4 classification/per-domain inventory remain open;
  WP17 stays deferred.
- Parser/scheduler expression contract continuation migrated 65 additional
  golden contracts into standard unittest discovery. Direct suites now cover
  parser diagnostics/limits, grouped modifier distribution and rejection,
  yearly token parsing, date and interval boundaries, business-day semantics,
  leap/fifth-weekday scheduling, random/counted-random identity and bounds,
  and typed exhaustion. Current registry: 713 top-level / 702 registered / 11
  explicitly retired / 170 newly migrated in this WP4 cycle / 0 duplicates;
  cumulative WP4 direct migrations: 283. Fresh full gates pass: unittest
  discovery 931 tests in 61.659s, normal golden 702/702, and deterministic
  shuffled golden (seed `20260811`) 702/702. Changed Python modules compile,
  and registry integrity passes as part of discovery. WP4 remains incomplete:
  the remaining 702 registered scenarios still need case-level
  direct-contract-versus-acceptance classification and per-domain
  count/runtime inventory. WP17 remains deferred.
- Parser fuzz/normalization follow-up moved the seeded malformed-input,
  DNF-round-trip, normalization/cache-isolation, parser-depth/error-payload,
  random-repeat, branch-metadata, and expression-characterization cases into
  `tests/recurrence/test_parser_fuzz_contracts.py`. The two matching golden
  registrations were removed after direct tests reproduced normalized DNF,
  natural text, deterministic date sequences, and random-date constraints.
  Registry is now 704 top-level / 693 registered / 11 retired / 179 newly
  migrated in this WP4 cycle / 0 duplicates; cumulative WP4 direct migrations:
  292. Full gates pass: unittest discovery 940 tests in 67.945s, normal golden
  693/693, and shuffled golden (seed `20260811`) 693/693. Changed Python files
  compile and registry integrity passes. WP4's case-level classification and
  per-domain count/runtime inventory remain open; WP17 remains deferred.
- Completed a further six narrow direct-contract migrations: ISO-week boundary
  counting, invalid/short UUID handling, integer bounds, modified-atom
  interval/roll/offset dates, parser satisfiability helper errors, and monthly
  month-alias guidance. Added them to normally discovered owner tests and
  removed the corresponding golden functions/registrations. A shuffled run
  exposed an existing isolate defect in the omit-file combined-load golden:
  it relied on config exports being initialized by earlier cases. The fixture
  now synchronizes config before overriding the temporary omit directory; the
  test passes alone and in shuffled order. Registry is 698 top-level / 687
  registered / 11 retired / 185 newly migrated this WP4 cycle / 0 duplicates;
  cumulative WP4 direct migrations: 298. Full gates pass: unittest discovery
  945 tests in 66.037s, normal golden 687/687, and shuffled golden (seed
  `20260811`) 687/687. WP4 case classification and per-domain count/runtime
  inventory remain open; WP17 remains deferred.
- Moved the three daily diagnostic-warning contracts to
  `tests/test_diagnostic_warnings_contract.py`, exercising the warning owner
  with an injected cache directory and deterministic stderr/environment
  assertions; removed their golden duplicates. Focused direct tests and registry
  integrity pass. Current registry: 695 top-level / 684 registered / 11 retired
  / 188 newly migrated this WP4 cycle / 0 duplicates; cumulative WP4 direct
  migrations: 301. Full gates pass: unittest discovery 948 tests in 66.096s,
  normal golden 684/684, and shuffled golden (seed `20260811`) 684/684. Changed
  Python modules compile and registry integrity passes. The remaining cases
  still require evidence-backed classification.
- Migrated the cache/precompute consistency matrix for six parser forms from
  the golden runner into `tests/test_precompute_contract.py`, exercising the
  production precompute owner with direct parser and natural-language bindings.
  Focused precompute tests and registry integrity pass. Registry is 694
  top-level / 683 registered / 11 retired / 189 newly migrated this WP4 cycle /
  0 duplicates; cumulative WP4 direct migrations: 302. Full gates pass:
  unittest discovery 949 tests in 66.161s, normal golden 683/683, and shuffled
  golden (seed `20260811`) 683/683. Changed Python modules compile and registry
  integrity passes. Case-level classification remains open.
- Migrated the positive-integral `chainMax` parsing boundary to
  `tests/test_add_validation_contract.py` and removed its duplicate golden
  case. Focused direct validation and registry integrity pass. Registry is now
  693 top-level / 682 registered / 11 retired / 190 newly migrated in this WP4
  cycle / 0 duplicates; cumulative WP4 direct migrations: 303. Full gates are
  pending. Remaining scenarios still need case-level classification.
- Migrated three config-path traversal/security cases into
  `tests/test_config_path_security_contracts.py`, exercising runtime Taskdata
  resolution and config path policy directly; removed the matching golden
  cases. Focused tests and registry integrity pass. Registry: 690 top-level /
  679 registered / 11 retired / 193 newly migrated this WP4 cycle / 0
  duplicates; cumulative WP4 direct migrations: 306. Full gates pass: unittest
  discovery 953 tests in 68.461s, normal golden 679/679, and shuffled golden
  (seed `20260811`) 679/679. Restored the neighboring hook integration-context
  helper after the case deletion boundary accidentally included it; adjusted
  its accepted fail-closed error text to include the current runtime-port
  wording, verified all three dependent hook cases, and reran both golden orders.
  Case-level classification of the remaining registry remains open.
- Added direct Taskdata resolver tests for argv/environment/fallback precedence,
  unsafe world-writable directories, and the explicit trust override. Removed
  the three matching golden cases; preserved the neighboring hook integration
  helper. Focused direct tests, the three dependent hook golden tests, and
  registry integrity pass. Registry is 687 top-level / 676 registered / 11
  retired / 196 newly migrated this WP4 cycle / 0 duplicates; cumulative WP4
  direct migrations: 309. Full gates pass: unittest discovery 955 tests in
  66.229s, normal golden 676/676, and shuffled golden (seed `20260811`) 676/676.
  Registry integrity and focused config/hook tests pass. Case-level
  classification of the remaining golden registry remains open.
- Power-failure recovery checkpoint: active branch `desloppify/review-remediation`
  and dirty worktree were preserved. Migrated 19 pure cursor, typed-outcome,
  scheduler-service/session, range, trace, shuffled-session, hint-failure, and
  terminal-evidence golden cases into direct unittest contracts. Added full
  direct scheduler runtime coverage without production changes, including a
  generated recurrence matrix and context-sensitive parity helper cases.
  Focused gates: 30/30 before the final parity additions and cross-path/registry
  gates 8/8 afterward; full unittest
  discovery: 976 tests in 77.671s; normal and shuffled golden runs: 657/657
  each (seed `20260811`). Registry is 668 top-level / 657 registered / 11
  retired / 215 newly migrated in this WP4 cycle / 0 duplicates; cumulative
  WP4 direct migrations: 328. Broader evaluator-owner, operational parity,
  provider, and remaining registry scenarios still need individual
  classification; WP4 is not complete.
- Continued WP4 case-level review in the anchor-file provider and inclusion
  domains. Fourteen isolated contracts (lazy projection, expansion caching, cursor
  reuse/indexing, DST ordering/fold metadata, duplicate/overnight descriptions,
  retry after load failure, and incomparable datetimes) now run as direct
  unittest contracts; golden duplicates were removed and registry
  allowlist/counts updated. Focused owner suites plus registry integrity pass
  31/31. An intermediate golden run overlapped source edits and was discarded;
  fresh stable and shuffled runs are recorded in the following verification
  entry. Remaining golden cases still require individual decision and
  rationale; WP4 is incomplete.
- Verification completed for the current state: standard discovery passes
  990/990; normal golden passes 643/643; seeded shuffled golden passes 643/643
  (`20260811`). Current registry is 654 top-level / 643 registered / 11
  retired / 229 direct-migrated allowlist entries, with zero duplicate
  registrations. Cumulative WP4 direct migrations: 342. The run-overlap
  failure was only an invalid verification race; reruns after edits all pass.
  WP4 remains incomplete because the remaining 643 registered cases still need
  individual classification and appropriate migration.
- Resumed case-by-case WP4 work. Migrated the evaluator/chain-generation time
  form parity case and split the combined DST-gap/business-calendar parity case
  into three direct methods in `tests/recurrence/test_scheduler_cross_path_conformance.py`.
  The initial parity fixture failed because it captured the timezone before
  lazy configuration had initialized; initialize the core timezone first, then
  pass the same captured timezone and local cursors to both paths. Focused
  cross-path plus registry tests pass 12/12. Registry is 649 top-level / 637
  registered / 12 explicitly retired / 234 migrated this WP4 cycle / 0
  duplicates; cumulative direct migrations: 347. Full unit and golden gates
  are pending. The remaining 637 golden functions still require individual
  classification; no production behavior was changed.
- The three recurrence-identity characterization wrappers were migrated to
  `tests/test_recurrence_identity_contracts.py` as direct tests of their owning
  modules. Focused recurrence-identity, cross-path, and registry checks pass
  15/15. Current registry is 646 top-level / 634 registered / 12 explicitly
  retired / 237 migrated in this WP4 cycle / 0 duplicates; cumulative direct
  migrations: 350. Complete gates now pass: standard discovery 999 tests in
  67.423s; normal golden 634/634; shuffled golden 634/634 (seed `20260811`).
  Case-level classification remains open for 634 registered golden scenarios.
- Migrated two random-time-window golden cases. DST slot uniqueness now uses
  direct `time_windows` and explicit `timeutil` timezone contracts;
  unsupported composition guidance and anchor-file random-window
  canonicalization are direct parser/file-spec tests. The process-stability
  scenario remains golden because it explicitly checks cross-process behavior.
  Focused time-window, parser-owner, file-backed, and registry checks pass.
  Current inventory is 644 top-level / 632 registered / 12 explicitly retired /
  239 direct-migrated entries / 0 duplicates; cumulative WP4 migrations: 352.
  Full gates pass: standard discovery 1002 tests in 67.185s, normal golden
  632/632, and shuffled golden 632/632 (seed `20260811`).
- Continued WP4 positional-selection classification. Eleven owner-level parser,
  evaluator, period-bound, capacity, advice, and cache cases plus six public
  parser/scheduler contracts now run in
  `tests/recurrence/test_position_selection_contracts.py`. Removed only their
  duplicate golden wrappers; on-add hook, modify-completion, and timeline
  integrations remain at the acceptance boundary. Current registry: 627
  top-level / 615 registered / 12 explicitly retired / 256 migrated in this
  WP4 cycle / 0 duplicate registrations; cumulative WP4 direct migrations:
  369. Focused direct and registry checks pass 21/21. Full verification passes:
  unittest discovery 1016 tests in 75.373s; normal golden 615/615; seeded
  shuffled golden 615/615 (seed `20260811`). Python compilation passes. Scoped
  `git diff --check` still reports trailing whitespace at unrelated existing
  modified lines 10321, 10323, 10327, 10329, 10335, and 15744 in the already-
  dirty golden module; these lines were not part of this migration.
- Continued WP4 with six pure omission-parser/evaluator/scheduler contracts and
  one file-source grammar/parser contract. Direct coverage now lives in
  `tests/recurrence/test_omit_contracts.py` and
  `tests/test_file_backed_contracts.py`; filesystem-dependent omit-file and
  hook/Taskwarrior acceptance scenarios remain at their appropriate boundaries.
  Registry is now 620 top-level / 608 registered / 12 explicitly retired / 263
  migrated in this WP4 cycle / 0 duplicates; cumulative WP4 direct migrations:
  376. Focused recurrence, file-backed, and registry suites pass 25/25. Full
  verification passes: unittest discovery 1023 tests in 75.517s; normal golden
  608/608; seeded shuffled golden 608/608 (seed `20260811`). Python compilation
  passes for the changed direct tests, registry test, and golden module. Scoped
  whitespace check still finds the previously noted unrelated trailing-space
  hunks in the already-dirty golden file.
- Migrated three pure native-until presentation/validation/carry contracts to
  `tests/test_native_until_contracts.py`; process/hook preview and modify-flow
  decisions remain golden. Current registry: 617 top-level / 605 registered /
  12 explicitly retired / 266 migrated in this WP4 cycle / 0 duplicates;
  cumulative WP4 direct migrations: 379. Focused native-until and registry
  tests pass 7/7. Full gates are pending.
- Moved three shared-time contracts into `tests/test_config_time_contract.py`:
  scheduling/completion comparator identity, repeated-hour comparison plus
  provider alias behavior, and Pacific/Apia full-day-gap resolution. Current
  registry: 614 top-level / 602 registered / 12 explicitly retired / 269
  migrated in this WP4 cycle / 0 duplicates; cumulative WP4 direct migrations:
  382. Focused time/native-until/registry tests pass 14/14. Full verification
  passes: unittest discovery 1029 tests in 83.740s, normal golden 602/602, and
  seeded shuffled golden 602/602 (seed `20260811`). No production behavior
  changed.
- Migrated direct lifecycle recurrence-fingerprint invariance and two runtime
  configuration contracts (snapshot provenance/isolation, warm fingerprint
  cache behavior). The config-fingerprint cross-process/cache-key scenario
  remains golden. Current registry: 611 top-level / 599 registered / 12
  explicitly retired / 272 migrated this WP4 cycle / 0 duplicates; cumulative
  WP4 direct migrations: 385. Focused lifecycle/configuration/registry tests
  pass 25/25. Full gates are pending for this latest batch.
- Migrated the recurrence-spec normalization/context contract into
  `tests/recurrence/test_scheduler_runtime_contracts.py` and bounded numeric
  environment parsing into `tests/test_hook_bootstrap.py`. The hook-level
  malformed-environment behavior remains golden. Current registry: 609
  top-level / 597 registered / 12 explicitly retired / 274 migrated in this WP4
  cycle / 0 duplicates; cumulative WP4 direct migrations: 387. Focused hook,
  recurrence-runtime, and registry checks pass 21/21. Full verification passes:
  unittest discovery 1034 tests in 77.052s; normal golden 597/597; seeded
  shuffled golden 597/597 (seed `20260811`).
- Split the last-Friday golden case: its natural-language phrase was already
  covered directly, so only the distinct deterministic five-occurrence weekday
  check moved to `tests/recurrence/test_scheduler_runtime_contracts.py`. Moved
  the missing explicit-config warning check to
  `tests/test_structured_failure_boundaries.py`. Current registry: 607
  top-level / 595 registered / 12 explicitly retired / 276 migrated this WP4
  cycle / 0 duplicates; cumulative WP4 direct migrations: 389. Focused
  recurrence, structured-failure, and registry tests pass 20/20. Full gates are
  clean: unittest discovery 1036 tests in 74.921s, normal golden 595/595, and
  seeded shuffled golden 595/595 (seed `20260811`).
- Migrated two runtime ownership invariants into
  `tests/test_architecture_contract.py`: panel-colour manifest inclusion and
  absence of removed exit-flow modules from both the source tree and runtime
  manifests. Current registry: 605 top-level / 593 registered / 12 explicitly
  retired / 278 migrated in this WP4 cycle / 0 duplicates; cumulative WP4 direct
  migrations: 391. Focused architecture and registry tests pass 17/17. Full
  verification passes: unittest discovery 1038 tests in 74.692s; normal golden
  593/593; seeded shuffled golden 593/593 (seed `20260811`).
- Migrated the on-add UTC-fallback warning policy and panel diagnostics for
  empty/unmatched file-backed sources into existing direct owner suites. Current
  registry: 603 top-level / 591 registered / 12 explicitly retired / 280
  migrated in this WP4 cycle / 0 duplicates; cumulative WP4 direct migrations:
  393. Focused preview, structured-failure, and registry tests pass 29/29. Full
  verification passes: unittest discovery 1040 tests in 74.930s; normal golden
  591/591; seeded shuffled golden 591/591 (seed `20260811`).
- Migrated the Doctor UDA-alias configuration contract into
  `tests/test_doctor_configuration_contract.py`. The test covers enabled
  alias messaging/clear syntax and missing-config default-off behavior with
  isolated environment settings. Registry integrity now expects 602 top-level
  / 590 registered / 12 retired / 281 WP4 migrations / 0 duplicates; cumulative
  WP4 direct migrations: 394. Focused and full verification are pending.
- Continued the Doctor cluster migration: configuration-schema findings and
  live-panel policy now have direct configuration contracts; timezone summary,
  compact large-history output, and historical finding aggregation now have
  direct presentation contracts. Registry integrity expects 597 top-level /
  585 registered / 12 retired / 286 WP4 migrations / 0 duplicates; cumulative
  direct migrations: 399. Focused direct tests pass; full verification pending.
- Moved six more pure Doctor configuration-service contracts into direct tests:
  timezone missing/unavailable states, astronomy preflight, seasonal event
  projection/invalid mode, active-source configuration drift, and missing
  Navigator dependencies. Kept separate subprocess config-edit/removal drift
  golden because it verifies process-lifetime behavior. Current registry:
  591 top-level / 579 registered / 12 retired / 292 WP4 migrations / 0
  duplicates; cumulative direct migrations: 405.
- Full verification after the Doctor migration passes: unittest discovery
  1051/1051 in 75.070s; normal golden 579/579; seeded shuffled golden 579/579
  (seed `20260811`). Python compilation and scoped `git diff --check` also pass.
  The Doctor cases retained in golden exercise actual CLI/Taskwarrior/install
  or process-lifetime boundaries. Per-case WP4 inventory remains incomplete.
- Migrated `test_operator_context_discovers_taskdata_once` into
  `tests/test_integration_runtime_ports.py`, where the injected Taskwarrior
  executable and configuration callbacks verify single discovery/reload and
  preserve the configuration failure stage. Moved the presentation import
  boundary assertion into `tests/test_architecture_contract.py`. Registry is
  now 589 top-level / 577 registered / 12 retired / 294 WP4 migrations / 0
  duplicates; cumulative direct migrations: 407. Focused checks pass; full
  gates are pending.
- Moved bounded-versus-full lifecycle candidate query construction into
  `tests/test_task_read_repository_contracts.py`; the injected client asserts
  only query filters and avoids golden-runner coupling. Registry now stands at
  588 top-level / 576 registered / 12 retired / 295 WP4 migrations / 0
  duplicates; cumulative direct migrations: 408. Focused verification pending.
- Moved four completion-analytics contracts to
  `tests/test_modify_analytics_contracts.py`: chain-gap/missing-identity
  warnings, healthy streak, low on-time guidance, and clinical drift/style
  normalization. Direct tests pass using the hook's 3,600-second tolerance.
  Registry: 584 top-level / 572 registered / 12 retired / 299 WP4 migrations /
  0 duplicates; cumulative direct migrations: 412. Full gates pending.
- Moved the public facade export/signature assertions into typed API tests;
  standard golden-registry integrity replaces the golden self-registration
  validator. Query invalid-request and compact Unicode/budget serialization
  contracts now live under query command tests; integrity report parity lives
  under `tests/test_integrity_report_contract.py`. Registry: 579 top-level /
  567 registered / 12 retired / 304 WP4 migrations / 0 duplicates; cumulative
  direct migrations: 417. Focused tests pass; full gates pending.
- Migrated TaskCommand failure classification and opt-in lock retries into
  `tests/test_task_command_contracts.py`, retaining subprocess coverage while
  testing the command wrapper directly. Registry: 577 top-level / 565
  registered / 12 retired / 306 WP4 migrations / 0 duplicates; cumulative
  direct migrations: 419. Focused tests pass; full gates pending.
