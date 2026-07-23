from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from nautical_core import hook_bootstrap

try:
    import fcntl
except Exception:
    fcntl = None


HOOK_FILES = {
    "on-add": "on-add-nautical.py",
    "on-modify": "on-modify-nautical.py",
    "on-exit": "on-exit-nautical.py",
}
HOOK_RUNTIME_FILES = {
    "on-add": ("hooks/add_impl.py", "hook_bootstrap.py", "hook_protocol.py"),
    "on-modify": ("hooks/modify_impl.py", "hook_bootstrap.py", "hook_protocol.py"),
    "on-exit": ("hooks/exit_impl.py", "hook_bootstrap.py", "config_support.py", "exit_probe.py"),
}
MANAGED_ROOT_FILES = ("nautical", "nautical_navigator.py")
_COPY_IGNORE = ("__pycache__", "*.pyc", "*.pyo", ".nautical-cache", ".nautical_cache")
_MAX_HOOK_SOURCE_BYTES = 2 * 1024 * 1024
_RELEASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class InstallError(RuntimeError):
    pass


def python_contract(path: Path) -> tuple[dict[str, int], set[str], str]:
    try:
        if path.stat().st_size > _MAX_HOOK_SOURCE_BYTES:
            return {}, set(), f"source exceeds {_MAX_HOOK_SOURCE_BYTES} bytes"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except Exception as exc:
        return {}, set(), f"{type(exc).__name__}: {exc}"

    constants: dict[str, int] = {}
    functions: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
            continue
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, int):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                constants[target.id] = int(value.value)
    return constants, functions, ""


def hook_candidates(hooks_dir: Path, event: str) -> list[Path]:
    try:
        candidates = sorted(path for path in hooks_dir.glob(f"{event}*") if path.is_file())
    except Exception:
        return []
    nautical: list[Path] = []
    for path in candidates:
        if "nautical" in path.name.lower():
            nautical.append(path)
            continue
        constants, _functions, _error = python_contract(path)
        if "_EXPECTED_IMPL_API" in constants or "HOOK_IMPL_API" in constants:
            nautical.append(path)
    return nautical


def inspect_hook_runtime(
    hook: Path,
    event: str,
    env: dict[str, str],
) -> tuple[dict[str, Any] | None, str, dict[str, Any]]:
    wrapper_constants, _wrapper_functions, wrapper_error = python_contract(hook)
    expected_api = wrapper_constants.get("_EXPECTED_IMPL_API")
    if wrapper_error or expected_api is None:
        reason = wrapper_error or "wrapper does not declare _EXPECTED_IMPL_API"
        return None, f"Nautical {event} wrapper is not compatible: {reason}.", {
            "hook": str(hook),
            "error": reason,
        }

    hook_dir = hook.parent
    default_base = hook_dir if hook_bootstrap.core_target_from_base(hook_dir) is not None else hook_dir.parent
    core_base = hook_bootstrap.trusted_core_base(default_base, env=env)
    core_init = hook_bootstrap.core_target_from_base(core_base)
    if core_init is None:
        return None, f"Nautical {event} hook cannot resolve nautical_core.", {
            "hook": str(hook),
            "core_base": str(core_base),
        }

    core_dir = core_init.parent
    required = HOOK_RUNTIME_FILES[event]
    missing = [name for name in required if not (core_dir / name).is_file()]
    if missing:
        return None, f"Nautical {event} runtime is incomplete.", {
            "hook": str(hook),
            "core": str(core_init),
            "missing": missing,
        }

    impl_path = core_dir / required[0]
    impl_constants, impl_functions, impl_error = python_contract(impl_path)
    actual_api = impl_constants.get("HOOK_IMPL_API")
    if impl_error or actual_api is None or "run_hook" not in impl_functions:
        reason = impl_error or "implementation must declare HOOK_IMPL_API and run_hook"
        return None, f"Nautical {event} implementation contract is invalid: {reason}.", {
            "hook": str(hook),
            "implementation": str(impl_path),
            "error": reason,
        }
    if actual_api != expected_api:
        return None, (
            f"Nautical {event} wrapper/core API mismatch: "
            f"wrapper expects {expected_api}, implementation provides {actual_api}."
        ), {
            "hook": str(hook),
            "implementation": str(impl_path),
            "expected_api": expected_api,
            "actual_api": actual_api,
        }
    return {
        "hook": hook,
        "core": core_init,
        "implementation": impl_path,
        "api": actual_api,
    }, "", {
        "hook": str(hook),
        "core": str(core_init),
        "implementation": str(impl_path),
        "api": actual_api,
    }


def _runtime_source_files(source: Path) -> list[Path]:
    files = [source / name for name in (*HOOK_FILES.values(), *MANAGED_ROOT_FILES)]
    files.extend(
        path
        for path in (source / "nautical_core").rglob("*")
        if path.is_file()
        and not any(part in {"__pycache__", ".nautical-cache", ".nautical_cache"} for part in path.parts)
        and path.suffix not in {".pyc", ".pyo"}
    )
    return sorted(files, key=lambda path: str(path.relative_to(source)))


def source_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in _runtime_source_files(source):
        if not path.is_file():
            raise InstallError(f"required runtime file is missing: {path}")
        rel = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(rel).to_bytes(4, "big"))
        digest.update(rel)
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def _copy_release(source: Path, stage: Path) -> None:
    stage.mkdir(mode=0o700, parents=True)
    for name in (*HOOK_FILES.values(), *MANAGED_ROOT_FILES):
        src = source / name
        if not src.is_file():
            raise InstallError(f"required runtime file is missing: {src}")
        shutil.copy2(src, stage / name)
    shutil.copytree(
        source / "nautical_core",
        stage / "nautical_core",
        ignore=shutil.ignore_patterns(*_COPY_IGNORE),
    )
    for name in (*HOOK_FILES.values(), "nautical"):
        (stage / name).chmod(0o755)


def _strict_json_object(stdout_text: str) -> bool:
    text = str(stdout_text or "").strip()
    if not text:
        return False
    try:
        value, offset = json.JSONDecoder().raw_decode(text)
    except Exception:
        return False
    return isinstance(value, dict) and not text[offset:].strip()


def _smoke_hooks(
    hooks: dict[str, Path],
    *,
    core_base: Path | None,
) -> None:
    plain = {
        "uuid": "11111111-1111-1111-1111-111111111111",
        "description": "Nautical installation validation",
        "status": "pending",
        "entry": "20260101T000000Z",
        "modified": "20260101T000000Z",
    }
    modified = dict(plain)
    modified["modified"] = "20260101T000001Z"
    with tempfile.TemporaryDirectory(prefix="nautical-install-smoke-") as td:
        env = os.environ.copy()
        env.update({
            "NAUTICAL_BENCH_FORCE_FULL": "1",
            "TASKDATA": td,
            "XDG_CACHE_HOME": str(Path(td) / "cache"),
            "TZ": "UTC",
        })
        if core_base is None:
            env.pop("NAUTICAL_CORE_PATH", None)
            env.pop("NAUTICAL_TRUST_CORE_PATH", None)
        else:
            env["NAUTICAL_CORE_PATH"] = str(core_base)
            env["NAUTICAL_TRUST_CORE_PATH"] = "1"
        env.pop("NAUTICAL_DIAG", None)
        env.pop("NAUTICAL_DIAG_LOG", None)
        cases = (
            ("on-add", json.dumps(plain), True),
            ("on-modify", json.dumps(plain) + "\n" + json.dumps(modified), True),
            ("on-exit", "", False),
        )
        for event, raw_input, expect_json in cases:
            hook = hooks[event]
            try:
                proc = subprocess.run(
                    [sys.executable, str(hook)],
                    input=raw_input,
                    text=True,
                    capture_output=True,
                    env=env,
                    timeout=12.0,
                )
            except Exception as exc:
                raise InstallError(f"{event} validation could not run: {exc}") from exc
            if proc.returncode != 0:
                error = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
                raise InstallError(f"{event} validation failed: {error}")
            if expect_json and not _strict_json_object(proc.stdout):
                raise InstallError(f"{event} validation did not emit one JSON object")
            if not expect_json and (proc.stdout or "").strip():
                raise InstallError(f"{event} validation wrote unexpected stdout")


def validate_release(root: Path, *, smoke: bool = True) -> dict[str, int]:
    for path in root.rglob("*.py"):
        try:
            compile(path.read_bytes(), str(path), "exec")
        except Exception as exc:
            raise InstallError(f"Python validation failed for {path}: {exc}") from exc

    env = {"NAUTICAL_CORE_PATH": str(root), "NAUTICAL_TRUST_CORE_PATH": "1"}
    apis: dict[str, int] = {}
    for event, name in HOOK_FILES.items():
        record, error, _details = inspect_hook_runtime(root / name, event, env)
        if record is None:
            raise InstallError(error)
        apis[event] = int(record["api"])
    if smoke:
        _smoke_hooks(
            {event: root / name for event, name in HOOK_FILES.items()},
            core_base=root,
        )
    return apis


def validate_installed(base: Path, hooks_dir: Path, *, smoke: bool = True) -> dict[str, int]:
    apis: dict[str, int] = {}
    for event, name in HOOK_FILES.items():
        hook = hooks_dir / name
        record, error, _details = inspect_hook_runtime(hook, event, {"NAUTICAL_CORE_PATH": ""})
        if record is None:
            raise InstallError(error)
        apis[event] = int(record["api"])
    if smoke:
        _smoke_hooks(
            {event: hooks_dir / name for event, name in HOOK_FILES.items()},
            core_base=None,
        )
    launcher = base / "nautical"
    if not (launcher.is_file() and os.access(str(launcher), os.X_OK)):
        raise InstallError(f"installed launcher is missing or not executable: {launcher}")
    return apis


class _InstallLock:
    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.fallback_fd: int | None = None

    def __enter__(self):
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if fcntl is not None:
            self.handle = self.path.open("a+", encoding="utf-8")
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                self.handle.close()
                self.handle = None
                raise InstallError("another Nautical installation is already running") from exc
            self.handle.seek(0)
            self.handle.truncate()
            self.handle.write(f"{os.getpid()}\n")
            self.handle.flush()
            return self
        try:
            self.fallback_fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(self.fallback_fd, f"{os.getpid()}\n".encode("ascii"))
        except FileExistsError as exc:
            raise InstallError("another Nautical installation is already running") from exc
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        if self.handle is not None:
            try:
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            finally:
                self.handle.close()
        if self.fallback_fd is not None:
            os.close(self.fallback_fd)
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass


def _lexists(path: Path) -> bool:
    return os.path.lexists(str(path))


def _atomic_symlink(target: str, path: Path) -> None:
    temp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        os.symlink(target, temp)
        os.replace(temp, path)
    finally:
        if _lexists(temp):
            temp.unlink()


def _atomic_copy(source: Path, target: Path, *, executable: bool = False) -> None:
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = target.parent / f".{target.name}.tmp-{uuid.uuid4().hex}"
    try:
        shutil.copy2(source, temp)
        if executable:
            temp.chmod(0o755)
        os.replace(temp, target)
    finally:
        if _lexists(temp):
            temp.unlink()


def _snapshot_file(path: Path, backup_dir: Path) -> dict[str, Any]:
    if not _lexists(path):
        return {"kind": "missing"}
    if path.is_symlink():
        return {"kind": "symlink", "target": os.readlink(path)}
    if not path.is_file():
        raise InstallError(f"managed install path is not a file or symlink: {path}")
    backup = backup_dir / path.name
    shutil.copy2(path, backup)
    return {"kind": "file", "backup": str(backup)}


def _restore_file(path: Path, snapshot: dict[str, Any]) -> None:
    kind = snapshot["kind"]
    if kind == "missing":
        if _lexists(path):
            path.unlink()
    elif kind == "symlink":
        _atomic_symlink(str(snapshot["target"]), path)
    else:
        _atomic_copy(Path(str(snapshot["backup"])), path, executable=os.access(str(snapshot["backup"]), os.X_OK))


def _pointer_snapshot(path: Path) -> dict[str, Any]:
    if not _lexists(path):
        return {"kind": "missing"}
    if not path.is_symlink():
        raise InstallError(f"managed runtime pointer is not a symlink: {path}")
    return {"kind": "symlink", "target": os.readlink(path)}


def _restore_pointer(path: Path, snapshot: dict[str, Any]) -> None:
    if snapshot["kind"] == "missing":
        if _lexists(path):
            path.unlink()
    else:
        _atomic_symlink(str(snapshot["target"]), path)


def _hook_conflicts(hooks_dir: Path) -> list[str]:
    conflicts = []
    for event, canonical_name in HOOK_FILES.items():
        canonical = hooks_dir / canonical_name
        for hook in hook_candidates(hooks_dir, event):
            if hook != canonical and os.access(str(hook), os.X_OK):
                conflicts.append(str(hook))
    return sorted(set(conflicts))


def _verify_running_wrappers(hooks_dir: Path, new_apis: dict[str, int]) -> None:
    for event, name in HOOK_FILES.items():
        hook = hooks_dir / name
        if not (hook.is_file() and os.access(str(hook), os.X_OK)):
            continue
        constants, _functions, error = python_contract(hook)
        expected = constants.get("_EXPECTED_IMPL_API")
        if error or expected is None:
            raise InstallError(
                f"cannot prove upgrade compatibility for active wrapper {hook}; "
                "replace or disable it before installing"
            )
        if expected != new_apis[event]:
            raise InstallError(
                f"active {event} wrapper expects API {expected}, "
                f"but the new core provides {new_apis[event]}"
            )


def _write_manifest(
    stage: Path,
    *,
    release_id: str,
    digest: str,
    source: Path,
    apis: dict[str, int],
) -> None:
    payload = {
        "schema": 1,
        "release_id": release_id,
        "content_sha256": digest,
        "created_at": int(time.time()),
        "source": str(source),
        "hook_impl_api": apis,
    }
    (stage / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _release_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise InstallError(f"release manifest is unreadable at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InstallError(f"release manifest is invalid at {path}")
    return value


def _active_release_name(current: Path, releases_dir: Path) -> str:
    if not current.is_symlink():
        return ""
    try:
        resolved = current.resolve(strict=True)
        resolved.relative_to(releases_dir.resolve())
        return resolved.name
    except Exception:
        return ""


def _target_has_nautical(base: Path, hooks_dir: Path, current: Path) -> bool:
    paths = [
        current,
        base / "nautical_core",
        *(base / name for name in MANAGED_ROOT_FILES),
        *(hooks_dir / name for name in HOOK_FILES.values()),
    ]
    return any(_lexists(path) for path in paths)


def _target_plan(
    *,
    base: Path,
    hooks_dir: Path,
    releases_dir: Path,
    current: Path,
    release_id: str,
    digest: str,
    apis: dict[str, int],
    smoke: bool,
) -> dict[str, Any]:
    conflicts = _hook_conflicts(hooks_dir)
    if conflicts:
        raise InstallError(
            "other active Nautical hooks would cause duplicate execution: "
            + ", ".join(conflicts)
        )
    _verify_running_wrappers(hooks_dir, apis)

    release_dir = releases_dir / release_id
    release_exists = release_dir.exists()
    if release_exists:
        manifest = _release_manifest(release_dir)
        if str(manifest.get("content_sha256") or "") != digest:
            raise InstallError(f"release ID already exists with different content: {release_id}")
        if source_digest(release_dir) != digest:
            raise InstallError(f"release content does not match its manifest: {release_id}")
        validate_release(release_dir, smoke=smoke)

    previous_release = _active_release_name(current, releases_dir)
    has_existing = _target_has_nautical(base, hooks_dir, current)
    operation = "upgrade" if has_existing else "install"
    if previous_release == release_id and release_exists:
        try:
            validate_installed(base, hooks_dir, smoke=smoke)
        except InstallError:
            operation = "repair"
        else:
            operation = "reuse"
    return {
        "operation": operation,
        "previous_release": previous_release,
        "release_exists": release_exists,
    }


def install_release(
    *,
    source: Path,
    taskdata: Path,
    hooks_dir: Path | None = None,
    release_id: str = "",
    dry_run: bool = False,
    smoke: bool = True,
    _fail_after: str = "",
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    taskdata = taskdata.expanduser().resolve()
    hooks_dir = (hooks_dir or (taskdata / "hooks")).expanduser().resolve()
    base = hooks_dir.parent
    if not source.is_dir():
        raise InstallError(f"source directory does not exist: {source}")
    digest = source_digest(source)
    release_id = release_id.strip() or f"r-{digest[:12]}"
    if not _RELEASE_ID_RE.fullmatch(release_id):
        raise InstallError("release ID must use 1-64 letters, numbers, dots, underscores, or hyphens")

    runtime_root = base / ".nautical-runtime"
    releases_dir = runtime_root / "releases"
    current = runtime_root / "current"
    core_link = base / "nautical_core"
    lock_path = runtime_root / "install.lock"

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="nautical-install-stage-") as td:
            stage = Path(td) / release_id
            _copy_release(source, stage)
            apis = validate_release(stage, smoke=smoke)
        plan = _target_plan(
            base=base,
            hooks_dir=hooks_dir,
            releases_dir=releases_dir,
            current=current,
            release_id=release_id,
            digest=digest,
            apis=apis,
            smoke=smoke,
        )
        return {
            "status": "dry-run",
            "operation": plan["operation"],
            "changed": False,
            "previous_release": plan["previous_release"],
            "active_release": plan["previous_release"],
            "release_id": release_id,
            "content_sha256": digest,
            "source": str(source),
            "base": str(base),
            "hooks_dir": str(hooks_dir),
            "hook_impl_api": apis,
        }

    hooks_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    runtime_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    releases_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    with _InstallLock(lock_path):
        token = uuid.uuid4().hex
        stage = runtime_root / f".staging-{token}"
        rollback_dir = runtime_root / f".rollback-{token}"
        rollback_dir.mkdir(mode=0o700)
        release_dir = releases_dir / release_id
        reused_release = False
        legacy_core: Path | None = None
        migrated_configs: list[str] = []
        pointer_before: dict[str, Any] | None = None
        core_before: dict[str, Any] | None = None
        file_snapshots: dict[Path, dict[str, Any]] = {}
        switched = False
        try:
            _copy_release(source, stage)
            apis = validate_release(stage, smoke=smoke)
            _write_manifest(stage, release_id=release_id, digest=digest, source=source, apis=apis)
            plan = _target_plan(
                base=base,
                hooks_dir=hooks_dir,
                releases_dir=releases_dir,
                current=current,
                release_id=release_id,
                digest=digest,
                apis=apis,
                smoke=smoke,
            )

            if plan["release_exists"]:
                reused_release = True
                shutil.rmtree(stage)
            else:
                os.replace(stage, release_dir)
            if _fail_after == "after_release":
                raise InstallError("injected failure after release staging")

            if plan["operation"] == "reuse":
                shutil.rmtree(rollback_dir, ignore_errors=True)
                return {
                    "status": "installed",
                    "operation": "reuse",
                    "changed": False,
                    "previous_release": plan["previous_release"],
                    "active_release": release_id,
                    "release_id": release_id,
                    "content_sha256": digest,
                    "source": str(source),
                    "base": str(base),
                    "hooks_dir": str(hooks_dir),
                    "current": str(current),
                    "reused_release": True,
                    "migrated_legacy_core": False,
                    "legacy_backup": "",
                    "migrated_configs": [],
                    "hook_impl_api": apis,
                }

            pointer_before = _pointer_snapshot(current)
            switched = True
            if _lexists(core_link) and not core_link.is_symlink():
                if not core_link.is_dir():
                    raise InstallError(f"existing nautical_core path is not a directory or symlink: {core_link}")
                legacy_core = rollback_dir / "legacy-nautical_core"
                os.replace(core_link, legacy_core)
                core_before = {"kind": "legacy"}
                for config_name in ("config-nautical.toml", "nautical.toml"):
                    legacy_config = legacy_core / config_name
                    target_config = taskdata / config_name
                    if legacy_config.is_file() and not _lexists(target_config):
                        file_snapshots[target_config] = _snapshot_file(target_config, rollback_dir)
                        _atomic_copy(legacy_config, target_config)
                        migrated_configs.append(str(target_config))
            elif core_link.is_symlink():
                core_before = {"kind": "symlink", "target": os.readlink(core_link)}
            else:
                core_before = {"kind": "missing"}

            _atomic_symlink(f"releases/{release_id}", current)
            _atomic_symlink(".nautical-runtime/current/nautical_core", core_link)
            if _fail_after == "after_pointer":
                raise InstallError("injected failure after pointer switch")

            for name in (*HOOK_FILES.values(), *MANAGED_ROOT_FILES):
                target = (hooks_dir / name) if name in HOOK_FILES.values() else (base / name)
                file_snapshots[target] = _snapshot_file(target, rollback_dir)
                _atomic_copy(release_dir / name, target, executable=(name != "nautical_navigator.py"))
            if _fail_after == "after_wrappers":
                raise InstallError("injected failure after wrapper install")
            if _fail_after == "before_postcheck":
                raise InstallError("injected failure before post-install validation")

            installed_apis = validate_installed(base, hooks_dir, smoke=smoke)
            if installed_apis != apis:
                raise InstallError("post-install hook API validation changed unexpectedly")

            migrated_backup = ""
            if legacy_core is not None and legacy_core.exists():
                backup_root = runtime_root / "backups" / f"{int(time.time())}-{release_id}"
                backup_root.mkdir(mode=0o700, parents=True)
                final_legacy = backup_root / "nautical_core"
                os.replace(legacy_core, final_legacy)
                migrated_backup = str(final_legacy)
            shutil.rmtree(rollback_dir, ignore_errors=True)
            return {
                "status": "installed",
                "operation": plan["operation"],
                "changed": True,
                "previous_release": plan["previous_release"],
                "active_release": release_id,
                "release_id": release_id,
                "content_sha256": digest,
                "source": str(source),
                "base": str(base),
                "hooks_dir": str(hooks_dir),
                "current": str(current),
                "reused_release": reused_release,
                "migrated_legacy_core": bool(migrated_backup),
                "legacy_backup": migrated_backup,
                "migrated_configs": migrated_configs,
                "hook_impl_api": apis,
            }
        except Exception:
            if switched:
                for target, snapshot in reversed(list(file_snapshots.items())):
                    _restore_file(target, snapshot)
                if pointer_before is not None:
                    _restore_pointer(current, pointer_before)
                if core_before is not None:
                    if core_before["kind"] == "legacy":
                        if _lexists(core_link):
                            core_link.unlink()
                        if legacy_core is not None and legacy_core.exists():
                            os.replace(legacy_core, core_link)
                    elif core_before["kind"] == "symlink":
                        _atomic_symlink(str(core_before["target"]), core_link)
                    elif _lexists(core_link):
                        core_link.unlink()
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)
            if rollback_dir.exists():
                shutil.rmtree(rollback_dir, ignore_errors=True)
            raise


def runtime_status(base: Path) -> dict[str, Any]:
    base = base.expanduser().resolve()
    runtime_root = base / ".nautical-runtime"
    if not runtime_root.exists():
        return {"managed": False, "base": str(base)}

    current = runtime_root / "current"
    releases = runtime_root / "releases"
    core_link = base / "nautical_core"
    try:
        staging = sorted(
            str(path)
            for path in runtime_root.iterdir()
            if path.name.startswith((".staging-", ".rollback-"))
        )
    except Exception as exc:
        return {
            "managed": True,
            "base": str(base),
            "runtime_root": str(runtime_root),
            "active_release": "",
            "manifest": {},
            "abandoned": [],
            "errors": [f"runtime directory is unreadable: {exc}"],
        }
    managed = _lexists(current) or core_link.is_symlink()
    if not managed:
        return {
            "managed": False,
            "base": str(base),
            "runtime_root": str(runtime_root),
            "abandoned": staging,
        }
    errors: list[str] = []
    active_release = ""
    manifest: dict[str, Any] = {}
    if not current.is_symlink():
        errors.append("current runtime pointer is missing or is not a symlink")
    else:
        try:
            resolved = current.resolve(strict=True)
            resolved.relative_to(releases.resolve())
            active_release = resolved.name
            manifest = _release_manifest(resolved)
        except Exception as exc:
            errors.append(f"current runtime pointer is invalid: {exc}")

    if not core_link.is_symlink():
        errors.append("stable nautical_core path is missing or is not a symlink")
    else:
        try:
            core_resolved = core_link.resolve(strict=True)
            current_core = (current / "nautical_core").resolve(strict=True)
            if core_resolved != current_core:
                errors.append("stable nautical_core link does not use the active runtime")
        except Exception as exc:
            errors.append(f"stable nautical_core link is invalid: {exc}")

    return {
        "managed": managed,
        "base": str(base),
        "runtime_root": str(runtime_root),
        "active_release": active_release,
        "manifest": manifest,
        "abandoned": staging,
        "errors": errors,
    }
