"""Installer-focused golden tests."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace

import nautical_core.install_runtime as install_runtime
from dev_tools.golden_tests.support import expect


ROOT = Path(__file__).resolve().parents[2]


def test_installer_dry_run_fresh_install_and_idempotent_reinstall():
    """Local installs should validate before mutation and safely reuse identical releases."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        dry = install_runtime.install_release(
            source=ROOT,
            taskdata=taskdata,
            release_id="dry-run",
            dry_run=True,
            smoke=False,
        )
        expect(dry.get("status") == "dry-run", f"unexpected dry-run result: {dry!r}")
        expect(dry.get("operation") == "install", f"fresh dry-run did not plan an install: {dry!r}")
        expect(dry.get("changed") is False, f"dry-run reported target changes: {dry!r}")
        expect(not taskdata.exists(), f"dry-run mutated the target: {taskdata}")
        installed = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="release-one", smoke=False,
        )
        expect(installed.get("status") == "installed", f"fresh install failed: {installed!r}")
        expect(installed.get("operation") == "install", f"fresh install action was unclear: {installed!r}")
        expect(installed.get("changed") is True, f"fresh install did not report its change: {installed!r}")
        expect(installed.get("active_release") == "release-one", f"active release was not reported: {installed!r}")
        current = taskdata / ".nautical-runtime/current"
        wrapper = taskdata / "hooks/on-modify.nautical"
        expect(current.is_symlink(), "current pointer was not installed")
        expect(os.readlink(current) == "releases/release-one", "wrong active release")
        expect((taskdata / "nautical_core").is_symlink(), "stable core path was not installed")
        expect(
            install_runtime.validate_installed(taskdata, taskdata / "hooks", smoke=True)
            == {"on-add": 1, "on-modify": 1, "on-exit": 1},
            "installed hook layout did not pass post-install validation",
        )
        status = install_runtime.runtime_status(taskdata)
        expect(status.get("active_release") == "release-one", f"runtime status is wrong: {status!r}")
        expect(not status.get("errors"), f"fresh runtime status has errors: {status!r}")
        current_inode = os.lstat(current).st_ino
        wrapper_stamp = wrapper.stat().st_mtime_ns
        repeated = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="release-one", smoke=False,
        )
        expect(repeated.get("reused_release") is True, f"identical reinstall was not reused: {repeated!r}")
        expect(repeated.get("operation") == "reuse", f"identical reinstall was not a no-op: {repeated!r}")
        expect(repeated.get("changed") is False, f"identical reinstall reported changes: {repeated!r}")
        expect(os.lstat(current).st_ino == current_inode, "same-release install rewrote the active pointer")
        expect(wrapper.stat().st_mtime_ns == wrapper_stamp, "same-release install rewrote a valid wrapper")
        wrapper.unlink()
        repaired = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="release-one", smoke=False,
        )
        expect(repaired.get("operation") == "repair", f"damaged active release was not repaired: {repaired!r}")
        expect(repaired.get("changed") is True and wrapper.is_file(), f"repair did not restore the wrapper: {repaired!r}")
        launcher = taskdata / "nautical"
        launcher.write_text("#!/usr/bin/env python3\n# stale launcher\n", encoding="utf-8")
        launcher.chmod(0o755)
        repaired = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="release-one", smoke=False,
        )
        expect(repaired.get("operation") == "repair", f"stale launcher was not repaired: {repaired!r}")
        expect(launcher.read_bytes() == (ROOT / "nautical").read_bytes(), "repair retained a stale launcher")
        command_path = Path(td) / "bin" / "nautical"
        command_install = install_runtime.install_release(
            source=ROOT,
            taskdata=Path(td) / "command-taskdata",
            release_id="command-release",
            smoke=False,
            launcher_path=command_path,
        )
        expect(command_install.get("operation") == "install", f"command install failed: {command_install!r}")
        expect(command_path.is_symlink(), "user command launcher was not created as a symlink")
        expect(command_path.resolve() == (Path(td) / "command-taskdata" / "nautical").resolve(), "command launcher points to the wrong target")
        command_path.unlink()
        command_repair = install_runtime.install_release(
            source=ROOT,
            taskdata=Path(td) / "command-taskdata",
            release_id="command-release",
            smoke=False,
            launcher_path=command_path,
        )
        expect(command_repair.get("operation") == "repair", f"missing command launcher was not repaired: {command_repair!r}")
        expect(command_path.is_symlink(), "repair did not recreate the user command launcher")


def test_installer_navigator_dependency_failure_is_actionable():
    """Navigator smoke failures should identify missing requirements without a traceback."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "nautical_navigator.py").write_text("# test navigator\n", encoding="utf-8")
        original_run = install_runtime.subprocess.run
        install_runtime.subprocess.run = lambda *_args, **_kwargs: SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "Traceback (most recent call last):\n"
                "  File 'nautical_navigator.py', line 1, in <module>\n"
                "ModuleNotFoundError: No module named 'rich'\n"
            ),
        )
        try:
            install_runtime._smoke_navigator(root)
        except install_runtime.InstallError as exc:
            message = str(exc)
            expect("missing Python module 'rich'" in message, f"missing dependency was not identified: {message}")
            expect("requirements.txt" in message, f"dependency remedy is not actionable: {message}")
            expect("Traceback" not in message, f"dependency failure leaked a traceback: {message}")
        else:
            raise AssertionError("missing Navigator dependency did not fail validation")
        finally:
            install_runtime.subprocess.run = original_run


def test_installer_upgrade_rollback_restores_active_runtime():
    """A failed upgrade should restore its pointer and every managed wrapper."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        install_runtime.install_release(source=ROOT, taskdata=taskdata, release_id="release-one", smoke=False)
        current = taskdata / ".nautical-runtime/current"
        wrapper = taskdata / "hooks/on-modify.nautical"
        pointer_before = os.readlink(current)
        wrapper_before = wrapper.read_bytes()
        try:
            install_runtime.install_release(
                source=ROOT, taskdata=taskdata, release_id="release-two", smoke=False,
                _fail_after="after_wrappers",
            )
            raise AssertionError("injected upgrade failure should have raised")
        except install_runtime.InstallError as exc:
            expect("injected failure" in str(exc), f"unexpected rollback error: {exc}")
        expect(os.readlink(current) == pointer_before, "failed upgrade did not restore current")
        expect(wrapper.read_bytes() == wrapper_before, "failed upgrade did not restore wrapper")
        expect(not install_runtime.runtime_status(taskdata).get("errors"), "rollback left a broken managed runtime")
        retained = taskdata / ".nautical-runtime/releases/release-one"
        retained_plan = install_runtime.install_release(
            source=retained, taskdata=taskdata, release_id="release-one", dry_run=True, smoke=False,
        )
        expect(retained_plan.get("status") == "dry-run", f"retained release was not selectable: {retained_plan!r}")
        expect(retained_plan.get("previous_release") == "release-one", f"rollback selected an unexpected release: {retained_plan!r}")
        planned = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="release-two", dry_run=True, smoke=False,
        )
        expect(planned.get("operation") == "upgrade", f"upgrade dry-run was not identified: {planned!r}")
        expect(planned.get("previous_release") == "release-one", f"upgrade plan lost prior release: {planned!r}")
        expect(os.readlink(current) == pointer_before, "upgrade dry-run changed the active release")
        upgraded = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="release-two", smoke=False,
        )
        expect(upgraded.get("status") == "installed", f"upgrade failed: {upgraded!r}")
        expect(upgraded.get("operation") == "upgrade", f"upgrade action was unclear: {upgraded!r}")
        expect(upgraded.get("previous_release") == "release-one", f"upgrade lost prior release: {upgraded!r}")
        expect(upgraded.get("active_release") == "release-two", f"upgrade lost active release: {upgraded!r}")
        expect(os.readlink(current) == "releases/release-two", "upgrade did not atomically select release two")
        expect((taskdata / ".nautical-runtime/releases/release-one").is_dir(), "previous release was removed")


def test_installer_migrates_legacy_core_and_rolls_back_first_switch():
    """Legacy migration should preserve config and restore the directory on failure."""
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        legacy = taskdata / "nautical_core"
        legacy.mkdir(parents=True)
        (legacy / "legacy-marker").write_text("keep", encoding="utf-8")
        (legacy / "config-nautical.toml").write_text('tz = "UTC"\n', encoding="utf-8")
        result = install_runtime.install_release(
            source=ROOT, taskdata=taskdata, release_id="migrated", smoke=False,
        )
        backup = Path(str(result.get("legacy_backup") or ""))
        expect(result.get("migrated_legacy_core") is True, f"legacy migration was not reported: {result!r}")
        expect((backup / "legacy-marker").read_text(encoding="utf-8") == "keep", "legacy backup lost data")
        expect((taskdata / "config-nautical.toml").read_text(encoding="utf-8") == 'tz = "UTC"\n', "config was not preserved")
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        legacy = taskdata / "nautical_core"
        legacy.mkdir(parents=True)
        (legacy / "legacy-marker").write_text("restore", encoding="utf-8")
        (legacy / "config-nautical.toml").write_text('tz = "UTC"\n', encoding="utf-8")
        try:
            install_runtime.install_release(
                source=ROOT, taskdata=taskdata, release_id="rollback", smoke=False,
                _fail_after="after_pointer",
            )
            raise AssertionError("injected migration failure should have raised")
        except install_runtime.InstallError:
            pass
        expect(legacy.is_dir() and not legacy.is_symlink(), "rollback did not restore legacy core directory")
        expect((legacy / "legacy-marker").read_text(encoding="utf-8") == "restore", "rollback lost legacy data")
        expect(not (taskdata / "config-nautical.toml").exists(), "rollback left a migrated config copy")
        expect(install_runtime.runtime_status(taskdata).get("managed") is False, "failed first install looks active")


def test_installer_lock_and_duplicate_hook_guards():
    """Concurrent installs and pre-existing duplicate Nautical hooks should fail closed."""
    with tempfile.TemporaryDirectory() as td:
        lock_path = Path(td) / "install.lock"
        with install_runtime._InstallLock(lock_path):
            try:
                with install_runtime._InstallLock(lock_path):
                    raise AssertionError("second installer acquired an active lock")
            except install_runtime.InstallError as exc:
                expect("already running" in str(exc), f"unexpected lock error: {exc}")
    with tempfile.TemporaryDirectory() as td:
        taskdata = Path(td) / "taskdata"
        hooks = taskdata / "hooks"
        hooks.mkdir(parents=True)
        duplicate = hooks / "on-add-custom"
        shutil.copy2(ROOT / "on-add.nautical", duplicate)
        duplicate.chmod(0o755)
        try:
            install_runtime.install_release(source=ROOT, taskdata=taskdata, release_id="blocked", smoke=False)
            raise AssertionError("duplicate active hook should block installation")
        except install_runtime.InstallError as exc:
            expect("duplicate execution" in str(exc), f"unexpected duplicate-hook error: {exc}")


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


TESTS = (
    test_installer_dry_run_fresh_install_and_idempotent_reinstall,
    test_installer_navigator_dependency_failure_is_actionable,
    test_installer_upgrade_rollback_restores_active_runtime,
    test_installer_migrates_legacy_core_and_rolls_back_first_switch,
    test_installer_lock_and_duplicate_hook_guards,
    test_installer_initializes_explicit_timezone_config,
)
