# Configuration

Nautical reads the first existing valid configuration candidate. Set
`NAUTICAL_DIAG=1` to print the exact search order for the current invocation.

## Recommended starting file

```toml
# Calendar identity and timezone.
wrand_salt = "nautical|wrand|v4"
tz = "Europe/Bucharest"              # IANA timezone
season_hemisphere = "north"          # north or south
season_mode = "fixed"                # fixed or astronomical

# Trusted file-backed calendar directories; empty disables file resolution.
anchor_file_dir = ""
omit_file_dir = ""

# Parsing and cache behavior.
enable_anchor_cache = true
enable_uda_aliases = false
anchor_cache_dir = ""                 # automatic location when empty
anchor_cache_ttl = 0                   # seconds; 0 never expires by age

# Hook presentation and bounded work.
chain_color_per_chain = false
show_timeline_gaps = true
show_analytics = false
analytics_style = "clinical"          # clinical or coach
analytics_ontime_tol_secs = 14400
check_chain_integrity = false
debug_wait_sched = false
recurrence_update_udas = []
panel_mode = "rich"                    # rich/live/fast/line/compact/minimal/text
live_panel_duration_ms = 160           # 0..1000 total reveal budget
live_panel_footer = "NAUTICAL"
exit_progress = true
fast_color = true
outbox_drain_max_items = 32
max_chain_walk = 500

# Safety limits.
max_anchor_iterations = 128            # 32..1024
max_link_number = 10000
sanitize_uda = false
sanitize_uda_max_len = 1024            # 64..4096
max_json_bytes = 10485760              # 1 KiB..100 MiB
cache_ttl_secs = 3600
cache_load_mem_max = 128               # 16..4096
cache_load_mem_ttl = 300               # 0..86400 seconds

# Tables must follow top-level settings.
[astronomy]
default_location = "home"

[astronomy.locations.home]
latitude = 44.4268
longitude = 26.1025
elevation = 75
timezone = "Europe/Bucharest"

[anchor_presets]
payday = "m:15,-1bd"
workout = "w:mon,wed,fri"

[omit_presets]
holidays = "y:12-24..12-31"

[business_calendar.work]
anchor = "w:mon..fri"
omit = ["y:01-01", "y:12-25"]
```

TOML tables remain active until the next table. Keep all top-level settings
before `[astronomy]`, presets, and business calendars.

## Discovery

`NAUTICAL_CONFIG` selects one explicit file. Otherwise Nautical considers
`config-nautical.toml` and `nautical.toml` under, in order:

1. directories derived from `TASKRC`;
2. the effective `TASKDATA` directory;
3. the active Nautical core directory;
4. `$XDG_CONFIG_HOME/nautical` when set;
5. `~/.config/nautical`;
6. `~/.task`.

Unsafe, malformed, or scheduling-invalid discovered configuration is reported
and blocks schedule-dependent mutation. Nautical does not silently replace it
with defaults for a Nautical task.

## Presets

```bash
task add "Pay bills" anchor:"@payday"
task add "Workout except holidays" anchor:"@workout" omit:"@holidays"
```

Anchor and omission presets have separate namespaces. Preset and calendar
content contributes to the effective schedule fingerprint.

## Deprecated keys

Remove `holiday_region`, `verify_import`, `spawn_queue_max_bytes`, and
`spawn_queue_drain_max_items`. The lifecycle outbox replaced the JSON spawn
queue; use `outbox_drain_max_items` for the current drain bound.
