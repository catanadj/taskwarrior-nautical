# Installation

## Requirements

Nautical requires:

- Taskwarrior 3.x (the current compatibility target is 3.4.2);
- Python 3.11 or newer;
- Git and `curl` for the bootstrap installer;
- the Python packages in `requirements.txt`.

The installer detects ordinary Linux and Termux layouts. It installs a managed
runtime under Taskwarrior's data directory, registers Nautical's UDAs, installs
the three hooks, and exposes the `nautical` command.

## Install or upgrade

```bash
curl -fsSL https://raw.githubusercontent.com/catanadj/taskwarrior-nautical/main/bootstrap.sh | bash
```

The bootstrap asks before installing missing Python requirements. It then runs
the validated installer and a post-install verification. Existing Nautical
configuration is preserved.

For an audited installation, download the release first, inspect
`bootstrap.sh`, and run it locally.

## Verify

```bash
nautical doctor --installation-only
```

The verification should identify:

- the Taskwarrior executable and data directory;
- one active managed Nautical release;
- compatible add, modify, and exit hooks;
- registered UDAs;
- an explicit scheduling timezone;
- optional astronomy dependencies when configured.

Run the full read-only audit after installation:

```bash
nautical doctor
```

## Local checkout installation

From a Nautical checkout:

```bash
./nautical install
```

Useful installer options include:

```bash
./nautical install --dry-run
./nautical install --taskdata /path/to/taskdata
./nautical install --launcher-path /path/to/bin/nautical
./nautical install --json
```

## Configuration

The installer works with defaults, but calendar scheduling should use an
explicit IANA timezone. Copy `config-nautical.toml` into a supported config
location and set at least:

```toml
tz = "Europe/Bucharest"
```

See [Configuration](../reference/configuration.md) for discovery order,
validation rules, and every supported setting.

## Upgrade behavior

Re-running the bootstrap installs a content-addressed release, validates it,
switches the active runtime atomically, and updates the public launcher. Old
managed releases are eligible for bounded runtime cleanup.

After an upgrade, run:

```bash
nautical doctor --installation-only
```

If the launcher still resolves to an old path, inspect it with:

```bash
command -v nautical
type -a nautical
```

Then follow [Troubleshooting](../operations/troubleshooting.md).
