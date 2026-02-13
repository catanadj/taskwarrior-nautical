# Nautical Ops Templates

Files in this directory:

- `nautical_health_check_cron.sh`: runner script used by cron/systemd.
- `nautical-health-check.crontab`: cron line template.
- `nautical-health-check.service`: systemd service template.
- `nautical-health-check.timer`: systemd timer template.

## Preconditions

1. Set in your shell/session where hooks run:

```bash
export NAUTICAL_DIAG_LOG=1
# optional safety-over-speed mode
export NAUTICAL_DURABLE_QUEUE=1
# keep strict mode off during bake-in
export NAUTICAL_EXIT_STRICT=0
```

2. Confirm health check works:

```bash
python3 tools/nautical_health_check.py --taskdata "$HOME/.task" --json
```

## Cron installation

1. Replace `REPO_DIR` in `nautical-health-check.crontab` with your absolute repo path.
2. Add the cron line via `crontab -e`.

## systemd installation (user units)

1. Replace `REPO_DIR` in `nautical-health-check.service` with your absolute repo path.
2. Install units:

```bash
mkdir -p "$HOME/.config/systemd/user"
cp tools/ops/nautical-health-check.service "$HOME/.config/systemd/user/"
cp tools/ops/nautical-health-check.timer "$HOME/.config/systemd/user/"
systemctl --user daemon-reload
systemctl --user enable --now nautical-health-check.timer
systemctl --user status nautical-health-check.timer
```

3. Inspect logs:

```bash
journalctl --user -u nautical-health-check.service -n 50 --no-pager
```

## Promote strict mode

After stable bake-in (no sustained dead-letter growth, low lock failures), set:

```bash
export NAUTICAL_EXIT_STRICT=1
```

`nautical_health_check_cron.sh` exits with:

- `0` healthy
- `1` warning
- `2` critical
