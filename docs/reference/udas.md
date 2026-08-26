# Taskwarrior UDAs

The installer places Nautical UDA definitions in a managed Taskwarrior include.
These are the supported recurrence and lineage fields.

| UDA | Type | Ownership | Purpose |
| --- | --- | --- | --- |
| `cp` | string | user | Completion period or sequence |
| `anchor` | string | user | Calendar expression |
| `anchor_file` | string | user | File-backed inclusion expression |
| `anchor_mode` | string | user | `skip`, `all`, or `flex`; default `skip` |
| `bc` | string | user | Named business calendar |
| `omit` | string | user | Date exclusion expression |
| `omit_file` | string | user | File-backed exclusions |
| `chainMax` | numeric | user | Maximum link count |
| `chainUntil` | date | user | Final eligible recurrence target |
| `chain` | string | user/lifecycle | `on` or `off`; default `off` |
| `chainID` | string | lifecycle | Stable recurrence identity |
| `link` | numeric | lifecycle | One-based chain position |
| `prevLink` | string/UUID | lifecycle | Previous task UUID |
| `nextLink` | string/UUID | lifecycle | Next task UUID |

Taskwarrior 3.4.2 can use UUID type for `prevLink` and `nextLink`; string remains
portable across supported 3.x installations.

## Ownership rules

User recurrence inputs may be changed on a pending task, subject to transition
validation. Lineage fields must not be manually cleared or changed. Nautical
requires a complete chain identity before planning recurrence.

`cp` and anchor recurrence are mutually exclusive. Anchor recurrence may use
any combination of `anchor`, `anchor_file`, `omit`, and `omit_file`.

## Description aliases

Aliases are an opt-in convenience for task descriptions:

```toml
enable_uda_aliases = true
```

| Alias | UDA |
| --- | --- |
| `a:` | `anchor` |
| `af:` | `anchor_file` |
| `am:` | `anchor_mode` |
| `o:` | `omit` |
| `of:` | `omit_file` |
| `cm:` | `chainMax` |
| `cu:` | `chainUntil` |

```bash
task add 'Morning review a:w:mon@t=9 am:skip'
```

Aliases must be a trailing description block and the value begins immediately
after the colon. Clear a field with an empty alias such as `a:`; the dash form
is rejected. Nautical preserves an existing human description when a modify
command contains aliases only.

Use canonical UDA arguments in scripts and integrations. Aliases are intended
for interactive entry.
