# Composable Recurrence Policy Grammar

Status: exploratory design reference; not implemented and not a commitment to
change Nautical's public syntax.

This document proposes a normalized recurrence-policy model for a possible
future Nautical. It preserves the existing real-world recurrence vision while
making generation, omission, adjustment, lifecycle, and exceptions independent
concepts that can be composed safely.

The examples use an illustrative block syntax because it exposes the model
clearly. It is not a final grammar proposal. Existing Taskwarrior fields such as
`cp`, `anchor`, `omit`, `chainMax`, and `chainUntil` could remain the normal
user interface and compile into this model. Complex policies could be exposed
through named presets instead of requiring long command-line expressions.

## Motivation

Nautical already supports two strong recurrence families:

- completion-relative periods through `cp`;
- calendar-relative schedules through anchors, files, omissions, business
  calendars, positional selectors, deterministic randomness, and astronomical
  times.

The main limitation is no longer the number of schedule atoms. It is that some
useful policies belong to separate engines or fields and cannot be freely
combined.

For example, a real-world policy might be:

> Fourteen days after completion, choose the next working Monday, exclude
> company holidays, set the due time to 09:00, allow at most two unfinished
> occurrences, retry failures tomorrow, and stop after twelve successful
> completions.

No single conventional RRULE expresses that policy. Nautical has many of the
individual capabilities, but not one orthogonal model that composes all of
them.

The proposed design treats recurrence as a deterministic policy pipeline:

```text
origin
    -> generate candidate occurrences
    -> add inclusion sources
    -> remove excluded candidates
    -> select within candidate sets
    -> adjust selected targets
    -> materialize task occurrences
    -> advance according to outcomes
    -> terminate at explicit boundaries
```

Overrides and schedule revisions are layered over this pipeline rather than
silently rewriting historical meaning.

## Design Goals

- Preserve Nautical's current capabilities and concise forms.
- Compose completion-relative and calendar-relative scheduling.
- Distinguish skipping an occurrence from moving an occurrence.
- Make intentional overlapping occurrences possible without admitting
  accidental duplicates.
- Give completion, skip, failure, expiration, cancellation, and deletion
  explicit lifecycle meanings.
- Make one-occurrence and this-and-future edits auditable.
- Keep schedule calculation pure, deterministic, bounded, and explainable.
- Give every occurrence a stable logical identity independent of its current
  Taskwarrior UUID or adjusted due time.
- Record enough schedule and dependency provenance to reproduce decisions.
- Fail closed when required calendars, files, timezones, or configuration are
  unavailable or inconsistent.

## Non-Goals

- This proposal does not select a final surface syntax.
- It does not require replacing existing Nautical UDAs.
- It does not make arbitrary executable user code part of recurrence.
- It does not make renderers, hooks, or Taskwarrior commands responsible for
  schedule decisions.
- It does not weaken deterministic successor identity, lifecycle durability,
  authoritative reads, mutation guards, or postcondition verification.
- It does not require materializing all future occurrences.

## Normalized Policy Shape

An illustrative normalized schedule is:

```text
schedule {
    origin
    generate
    include
    exclude
    select
    adjust
    materialize
    advance
    terminate
}
```

Not every schedule needs every clause. Missing clauses receive explicit,
versioned defaults during compilation so runtime behavior never depends on an
implicit parser convention.

Each clause owns one question:

| Clause | Question |
| --- | --- |
| `origin` | From which instant, target, or event does calculation begin? |
| `generate` | Which ordered candidate occurrences exist? |
| `include` | Which additional candidate sources join the base stream? |
| `exclude` | Which unadjusted candidates are omitted entirely? |
| `select` | Which candidates are chosen from a bounded candidate set? |
| `adjust` | How is a chosen target deliberately moved or timed? |
| `materialize` | When and how many Taskwarrior tasks become visible? |
| `advance` | Which outcome causes which lifecycle transition? |
| `terminate` | Under which conditions does the series stop? |

The compiled model should be a typed immutable structure, not a dictionary of
loosely related strings.

## Evaluation Semantics

Textual clause order must not determine behavior. The compiler normalizes a
schedule into the canonical phases below.

1. Resolve a schedule revision and its portable dependencies.
2. Determine the calculation origin.
3. Generate a bounded ordered set or stream of raw candidates.
4. Union any additional inclusion sources.
5. Deduplicate candidates by logical occurrence slot.
6. Remove candidates matched by exclusion generators.
7. Apply selection within each declared selection window.
8. Apply ordered target adjustments.
9. Resolve target collisions according to an explicit collision policy.
10. Apply materialization admission and open-occurrence limits.
11. Produce a pure plan with explanation evidence.
12. Delegate authorized effects to the established lifecycle owner.

An exclusion and an adjustment have deliberately different meanings:

- `exclude holiday` means the holiday occurrence does not exist.
- `adjust roll_forward until business_day` means the occurrence exists but is
  placed on a different date.

If an adjusted target must avoid holidays, that condition belongs inside the
roll predicate:

```text
adjust roll_forward until all {
    business_day @work
    not calendar @company-holidays
}
```

It should not depend on a later, hidden reapplication of the ordinary
`exclude` phase.

## Origin

The origin identifies the reference used to calculate raw candidates.

```text
origin completion
origin previous_target
origin first_target
origin fixed 2026-09-01T09:00:00+03:00
origin occurrence 17
```

Typical meanings are:

- `completion`: calculate relative to the outcome time of the current
  occurrence;
- `previous_target`: advance from the scheduled target rather than when work
  happened;
- `first_target`: retain a stable series epoch;
- `fixed`: use an explicit versioned epoch;
- `occurrence`: use a stable logical occurrence as the revision boundary.

The clock policy must be explicit when durations interact with local time:

```text
origin completion clock exact_elapsed
origin completion clock preserve_wall
```

## Candidate Generation

Generators produce deterministic ordered candidate streams or bounded sets.

```text
generate every 3d
generate cycle { 3d, 20d, 7d }
generate weekday { mon, wed, fri }
generate monthly day { 1, 15, last }
generate yearly range 12-24..12-31
generate dates @team-events
generate rrule "FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=-1"
```

RRULE is one possible generator, not the top-level recurrence model. This
allows Nautical-native completion sequences, file-backed sources, random
selection, business calendars, and astronomical schedules to coexist with
standard rules without reducing everything to RRULE.

Multiple generators may retain branch-specific attributes:

```text
generate any {
    weekday mon at 09:00
    weekday fri at 15:00
}
```

This produces Monday at 09:00 and Friday at 15:00, not the Cartesian product
of both weekdays and both times.

## Inclusion Sources

`include` adds candidate streams to the base generator.

```text
generate weekday mon

include any {
    dates @company-events
    yearly 05-05 at 12:00
}
```

Inclusions are unioned and then deduplicated by logical slot. A collision
policy determines whether two semantically distinct sources landing on the
same target are merged, retained separately, or sent to review.

## Exclusion Sources

`exclude` removes raw candidate occurrences. Both recurring and explicit
sources are allowed.

```text
exclude any {
    weekday wed
    yearly range 12-24..12-31
    dates @company-holidays
    calendar @factory-shutdowns
}
```

Exclusions are date-based unless a clause explicitly declares time-level
matching. A date exclusion removes every time generated on that local date:

```text
exclude date 2026-12-25
exclude instant 2026-12-25T09:00:00+03:00
```

This preserves the useful distinction already present in Nautical: omitting a
date normally removes all of that date's scheduled times.

## Selection

Selection chooses candidates from a declared bounded window.

```text
select first in month
select last in quarter
select { 1st, 3rd, last } in year
select random 2 in week seed occurrence_slot
select first on_or_after candidate
```

Random selection must remain reproducible. Its seed includes the algorithm
version, stable series identity, schedule revision, logical occurrence slot,
normalized candidate set, and configured salt.

Selectors must prove their candidate window is finite and bounded before
evaluation. Unsupported unbounded combinations fail validation rather than
searching indefinitely.

## Adjustment

Adjustments deliberately transform a selected target.

```text
adjust set_time 09:00
adjust offset -2d
adjust offset +1 business_day @work
adjust preserve_wall_clock
adjust exact_elapsed
```

Rolling is explicit:

```text
adjust roll_forward until business_day @work
adjust roll_backward until business_day @work
adjust nearest until business_day @work tie forward
```

A constrained roll may combine predicates:

```text
adjust roll_forward until all {
    weekday mon
    business_day @work
    not calendar @company-holidays
}
```

Every adjustment contributes a before/after step to the explanation trace.

## Materialization

Materialization controls when logical occurrences become Taskwarrior tasks.

```text
materialize one_active
materialize all_due
materialize lookahead 3
materialize horizon 7d
materialize maximum_open 2
```

Policies may be combined where their meaning is defined:

```text
materialize {
    horizon 7d
    maximum_open 4
}
```

Each materialized occurrence has a deterministic slot identity. Intentional
overlap therefore does not require abandoning duplicate protection.

Materialization is admission policy only. It does not change the underlying
candidate stream or silently mark unfinished work complete.

## Outcome And Advancement

Advancement maps an explicit occurrence outcome to a lifecycle decision.

```text
advance {
    completed -> next
    skipped   -> next count=true
    failed    -> retry after 1d maximum=3
    expired   -> next
    cancelled -> stop
    deleted   -> stop
}
```

The outcome vocabulary must be small, typed, and independent of presentation.
Taskwarrior transitions are translated into outcomes at one validated
boundary. Ambiguous deletion or unavailable evidence never becomes a guessed
outcome.

An advancement action may be:

```text
next
retry after 1d
defer until 2026-10-01T09:00:00+03:00
stop
manual_review
```

Effects remain durable, guarded, idempotent, and verified through Nautical's
lifecycle application owner.

## Termination

Termination is separate from candidate generation and advancement.

```text
terminate occurrence_count 12
terminate successful_count 12
terminate target_after 2027-12-31T23:59:59+03:00
terminate after 3 consecutive_failures
terminate manual_only
```

Several conditions use explicit combination policy:

```text
terminate first_of {
    occurrence_count 12
    target_after 2027-12-31T23:59:59+03:00
}
```

Arbitrary executable predicates are excluded. Every termination rule must be
portable, deterministic, bounded, serializable, and explainable.

## Occurrence Identity

Task UUID, series identity, and logical occurrence identity are different
concepts.

- The series ID identifies the enduring recurrence.
- The schedule revision identifies the rules effective for a range of
  occurrences.
- The occurrence slot identifies one logical obligation before adjustment or
  Taskwarrior materialization.
- The Taskwarrior UUID identifies one stored task representing that slot.

A logical slot might be derived from canonical inputs such as:

```text
series_id
schedule_revision
generator_branch
generator_position
raw_candidate
```

Moving an occurrence must not change its slot identity. Reapplying the same
plan must derive the same task identity. Two genuinely separate occurrences
that happen to share a timestamp must retain different slots unless an
explicit collision policy merges them.

## Collision Policy

Adjustments and inclusion unions can make distinct logical occurrences land on
the same target.

```text
collision merge
collision keep_separate
collision manual_review
```

The default should be chosen conservatively. Silent merging can lose a real
obligation, while silent duplication can create unwanted work. A compiled
schedule should reject combinations whose collision semantics are undefined.

## Schedule Revisions

Changing a series creates a new immutable schedule revision rather than
rewriting the meaning of completed occurrences.

```text
revise series "176f5c68" {
    effective_from occurrence 17

    replace {
        generate weekday tue
        adjust set_time 10:00
    }
}
```

The resulting history is explicit:

```text
revision 1 -> occurrences 1..16
revision 2 -> occurrences 17..
```

Each occurrence records the revision that generated it. Navigator, query, and
reconcile can then explain historical targets using their original policy.

## Per-Occurrence Overrides

An override changes one logical occurrence without changing the base schedule.

Move one occurrence:

```text
override occurrence {
    series "176f5c68"
    original_target 2026-12-24T09:00:00+03:00
    move_to 2026-12-23T14:00:00+03:00
    reason "office closes early"
}
```

Cancel one occurrence but retain its position in the series:

```text
override occurrence {
    series "176f5c68"
    original_target 2026-12-31T09:00:00+03:00
    cancel
    count true
}
```

Overrides are immutable records with deterministic identity, scope, reason,
creation evidence, and supersession history. They do not mutate the canonical
base grammar silently.

## Examples

### Basic Completion Period

Current form:

```text
cp:3d
```

Normalized form:

```text
schedule {
    origin completion clock preserve_wall
    generate every 3d
    materialize one_active
    advance completed -> next
}
```

An exact elapsed period is explicit:

```text
schedule {
    origin completion clock exact_elapsed
    generate every 28h
    materialize one_active
}
```

### Calendar Recurrence With Branch-Specific Times

Current form:

```text
anchor:"w:mon@t=09:00,fri@t=15:00"
```

Normalized form:

```text
schedule {
    origin previous_target

    generate any {
        weekday mon at 09:00
        weekday fri at 15:00
    }

    materialize one_active
    advance completed -> next
}
```

### Calendar Inclusion And Omission

Current form:

```text
anchor:"w:mon..fri"
omit:"w:wed | y:12-24..12-31"
omit_file:"company-holidays.csv"
```

Normalized form:

```text
schedule {
    origin previous_target
    generate weekdays mon..fri

    exclude any {
        weekday wed
        yearly range 12-24..12-31
        dates @company-holidays
    }

    select first after previous_target
    materialize one_active
}
```

### Completion Followed By Calendar Placement

Fourteen days after completion, choose the next non-holiday working Monday at
09:00:

```text
schedule {
    origin completion
    generate offset 14d

    adjust roll_forward until all {
        weekday mon
        business_day @work
        not calendar @company-holidays
    }

    adjust set_time 09:00
    materialize one_active
}
```

An illustrative compact form could be:

```text
recur:"completion +14d; next working mon; omit @company-holidays; at 09:00"
```

### Monthly Roll Policy

The fifteenth of each month, rolled forward when it is not a working day:

```text
schedule {
    origin previous_target
    generate monthly day 15
    adjust roll_forward until business_day @work
    adjust set_time 09:00
}
```

This differs from omission. The monthly obligation still exists; only its
placement moves.

### Controlled Overlap

Create daily 08:00 and 20:00 medication tasks even if an earlier dose remains
unfinished:

```text
schedule {
    origin fixed 2026-09-01T08:00:00+03:00

    generate daily at {
        08:00
        20:00
    }

    materialize {
        horizon 2d
        maximum_open 4
    }

    collision keep_separate
}
```

### Outcome-Sensitive Advancement

```text
schedule {
    origin previous_target
    generate daily at 09:00

    advance {
        completed -> next
        skipped   -> next count=true
        failed    -> retry after 1d maximum=3
        expired   -> next
        deleted   -> stop
    }
}
```

### Complex Inspection Policy

An inspection is due six weeks after completion, placed on the next working
Tuesday outside factory shutdowns, at 07:30. At most two may remain open.
Failures retry after three days, and the series stops after twenty successful
inspections.

```text
schedule {
    origin completion
    generate offset 6w

    adjust roll_forward until all {
        weekday tue
        business_day @factory
        not calendar @factory-shutdowns
        not yearly range 12-24..01-02
    }

    adjust set_time 07:30
    materialize maximum_open 2

    advance {
        completed -> next count_success=true
        failed    -> retry after 3d
        skipped   -> next count_success=false
        expired   -> manual_review
    }

    terminate successful_count 20
}
```

## Compatibility With Existing Nautical Syntax

Existing syntax can compile into the normalized policy without changing the
ordinary user workflow.

```text
cp:3d
    -> origin(completion)
    -> generate(period=3d)

cp:"3d,20d,7d"
    -> origin(completion)
    -> generate(period_cycle=[3d,20d,7d])

anchor:"w:mon,wed"
    -> origin(previous_target)
    -> generate(weekday=[mon,wed])

anchor_file:"events.csv"
    -> include(dates=@events)

omit:"y:12-24..12-31"
    -> exclude(yearly_range=12-24..12-31)

omit_file:"holidays.csv"
    -> exclude(dates=@holidays)

anchor_mode:skip
    -> select(first_after=now)

anchor_mode:all
    -> select(first_after=previous_target)

chainMax:12
    -> terminate(occurrence_count=12)

chainUntil:2027-12-31
    -> terminate(target_after=2027-12-31)
```

`anchor_mode:flex` would compile into a versioned one-shot selection override
followed by the ordinary `all` policy, rather than remaining a hidden mutation.

## Named Presets

Simple schedules should remain concise. Complex policies are better stored as
named, reviewable configuration.

An illustrative TOML representation is:

```toml
[schedules.factory_inspection]
origin = "completion"
generate = "6w"
roll_forward_until = ["w:tue", "business:@factory"]
exclude = ["@factory-shutdowns", "y:12-24..01-02"]
time = "07:30"
maximum_open = 2
terminate_successful_count = 20
```

Task creation remains short:

```text
task add "Factory inspection" schedule:@factory_inspection
```

The preset must compile to a self-contained normalized revision with content
fingerprints. Runtime calculation does not reinterpret an unversioned preset
silently.

## Explanation Trace

Every pure evaluation returns a bounded explanation trace alongside its plan.

For the hybrid completion example:

```text
Completion                       2026-09-01 16:20 +03:00
Add fourteen days                2026-09-15 16:20 +03:00
Next Monday                      2026-09-21
Business calendar @work          eligible
Company holiday                  excluded
Next qualifying Monday           2026-09-28
Set task time                    09:00
Final target                     2026-09-28 09:00 +03:00
Schedule revision                4
Occurrence slot                  18
```

The trace contains policy identifiers and bounded facts, not private task
descriptions. Navigator, query, dry-run, application verification, and repair
must derive the same result from the same snapshot and revision.

## Portable Dependencies And Provenance

A compiled schedule revision records or fingerprints every input that may
change a decision:

- normalized policy schema and grammar version;
- timezone identifier and timezone-data version where available;
- business-calendar definitions;
- file-backed inclusion and exclusion content;
- anchor and omission presets;
- deterministic random algorithm version and salt identity;
- astronomy profile and relevant dependency version;
- schedule revision and override set;
- materialization, collision, outcome, and termination policies.

An unavailable dependency is not an empty candidate source. Effectful planning
fails closed until authoritative evidence is restored.

Completed occurrences retain their schedule revision and logical slot. This
makes historical decisions explainable after presets, files, or configuration
change.

## Architecture Boundary

The normalized recurrence model belongs in a pure domain package.

```text
surface syntax / existing UDAs / named preset
        -> parser and validator
        -> immutable normalized schedule revision
        -> pure candidate evaluator
        -> immutable occurrence plan and explanation
        -> lifecycle authorization and durable application
        -> authoritative postcondition verification
```

The parser performs no Taskwarrior, filesystem, SQLite, or rendering effects.
Dependency providers resolve trusted files, calendars, timezones, and
astronomy data before pure evaluation. The evaluator consumes immutable
snapshots only.

The recurrence engine decides what should happen. Existing lifecycle and
Taskwarrior integration owners decide whether and how an authorized effect is
performed safely.

## Safety And Reliability Requirements

- All normalized models, revisions, overrides, plans, and explanation facts
  are deeply immutable.
- Every search and selector has explicit candidate and iteration bounds.
- Every external dependency has typed available, unavailable, malformed,
  partial, and stale outcomes.
- Schedule revisions and overrides have stable canonical fingerprints.
- Dry-run and apply consume the same plan; apply adds authorization, guarded
  delegation, refresh, and verification only.
- A mutation never proceeds from partial or unavailable schedule evidence.
- Certain mutation invalidates affected evidence; uncertain mutation
  invalidates the entire invocation snapshot.
- Materialization is idempotent per logical occurrence slot.
- Concurrent devices derive the same slot and task identity from the same
  revision and evidence.
- Conflicting schedule revisions, overrides, or portable dependencies fail to
  a typed conflict or manual-review result.
- No SQLite transaction remains open across a Taskwarrior call.
- Hook stdout remains strict JSON with `ensure_ascii=False`; diagnostics use
  stderr only when `NAUTICAL_DIAG=1`.

## Suggested Introduction Strategy

This design can be introduced internally without immediately adding public
syntax.

1. Define the immutable normalized model and canonical encoding.
2. Compile existing `cp`, anchor, omission, mode, and termination fields into
   it without changing behavior.
3. Prove conformance against current golden schedules and lifecycle plans.
4. Move Navigator and query explanation to normalized traces.
5. Add schedule revisions and occurrence-slot identity.
6. Add first-class one-occurrence overrides.
7. Add hybrid completion/calendar composition.
8. Add bounded materialization policies and intentional overlap.
9. Add richer outcome policies only after Taskwarrior transition semantics are
   explicitly mapped and tested.
10. Consider a compact public grammar only after the normalized contract is
    stable.

At every stage, existing surface syntax remains valid and no old/new lifecycle
owner runs in parallel for the same task.

## Open Design Questions

- What is the canonical occurrence-slot identity across schedule revisions?
- Should the default collision policy be `manual_review` or a narrowly defined
  deterministic merge?
- Which outcome vocabulary can be represented faithfully through Taskwarrior
  status transitions and UDAs?
- How should skip, cancel, expire, and delete affect occurrence counts and
  successful-completion counts?
- Which materialization policies are safe with Taskwarrior synchronization and
  dependency semantics?
- How are portable calendar dependencies distributed without turning local
  file paths into task data?
- Should a schedule revision embed normalized dependency content or only
  content-address it?
- How should timezone-data version differences be detected and presented?
- Which adjustments revalidate eligibility, and how is that behavior made
  explicit rather than order-dependent?
- How should overlapping occurrences inherit dependencies and annotations?
- Which current grammar constructs should remain surface sugar, and which
  should become named presets?
- What is the bounded migration and rollback model for existing chains?

## Summary

The proposal does not seek a larger collection of special-case recurrence
tokens. It seeks a small set of typed, composable policies:

```text
generate
include
exclude
select
adjust
materialize
advance
terminate
override
revise
```

The most important artifact is the immutable normalized schedule revision and
its explanation trace. Surface syntax is secondary.

Such a model would allow Nautical to retain its independent real-world
recurrence vision while making new capabilities easier to reason about,
combine, test, synchronize, recover, and explain.
