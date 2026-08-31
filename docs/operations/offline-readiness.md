# Offline Readiness

Nautical can be installed and repaired without network access when a local
offline kit is prepared in advance. A kit contains the exact runtime source,
configuration and UDA inputs, documentation, a dependency inventory, and
checksums.

## Build And Verify A Kit

From a Nautical checkout, choose a destination outside the checkout:

```bash
python3 dev_tools/nautical_offline_kit.py build /path/to/nautical-kit
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit
```

The builder does not contact Git, download packages, or modify Taskwarrior.
It refuses to overwrite an existing destination. Verify the kit with the
ordinary Python interpreter before taking the device offline.

## Local Install

After copying the verified kit to a device, verify it again and install from
its root:

```bash
python3 dev_tools/nautical_offline_kit.py verify /path/to/nautical-kit
./nautical install --source /path/to/nautical-kit --dry-run
./nautical install --source /path/to/nautical-kit
nautical doctor --installation-only
```

Use a disposable `TASKDATA` directory for repair drills. The kit does not
include user Taskwarrior data, lifecycle state, or private resource files;
back those up separately before an offline recovery operation.
