"""Installer-focused golden tests."""

from __future__ import annotations

from pathlib import Path
import tempfile

from nautical_core import install_runtime


def test_installer_initializes_explicit_timezone_config():
    """Fresh installs should write an explicit detected timezone."""
    previous = install_runtime.detect_local_timezone
    install_runtime.detect_local_timezone = lambda: "UTC"
    try:
        with tempfile.TemporaryDirectory() as temporary:
            taskdata = Path(temporary) / "taskdata"
            result = install_runtime.install_release(
                source=Path(__file__).resolve().parents[2],
                taskdata=taskdata,
                release_id="timezone-config",
                smoke=False,
            )
            config = taskdata / "config-nautical.toml"
            if result.get("initialized_config") != str(config):
                raise AssertionError(f"fresh config was not reported: {result!r}")
            expected = '# Nautical timezone detected during installation.\ntz = "UTC"\n'
            if config.read_text(encoding="utf-8") != expected:
                raise AssertionError("fresh config content was wrong")
    finally:
        install_runtime.detect_local_timezone = previous


TESTS = (test_installer_initializes_explicit_timezone_config,)
