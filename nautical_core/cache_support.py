from __future__ import annotations

import hashlib
import os
import stat
import sys
import tempfile


def nautical_cache_dir(*, validated_user_dir) -> str:
    base_raw = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    safe_base = validated_user_dir(
        base_raw,
        label="XDG_CACHE_HOME",
        trust_env="NAUTICAL_TRUST_CACHE_PATH",
        warn_on_error=False,
    )
    base = safe_base or os.path.expanduser("~/.cache")
    return os.path.join(base, "nautical")


def ensure_cache_dir(path: str) -> bool:
    try:
        os.makedirs(path, mode=0o700, exist_ok=True)
        st_before = os.lstat(path)
        if stat.S_ISLNK(st_before.st_mode) or (not stat.S_ISDIR(st_before.st_mode)):
            return False
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(path, flags)
        try:
            st_fd = os.fstat(fd)
            if not stat.S_ISDIR(st_fd.st_mode):
                return False
            if (st_before.st_dev, st_before.st_ino) != (st_fd.st_dev, st_fd.st_ino):
                return False
            try:
                os.fchmod(fd, 0o700)
            except Exception:
                try:
                    os.chmod(path, 0o700)
                except Exception:
                    pass
        finally:
            os.close(fd)
        return os.access(path, os.W_OK)
    except Exception:
        return False


def select_cache_dir(
    *,
    anchor_cache_dir_override: str,
    nautical_cache_dir_path: str,
    validated_user_dir,
) -> str:
    candidates = []
    if anchor_cache_dir_override:
        override = validated_user_dir(
            anchor_cache_dir_override,
            label="anchor_cache_dir",
            trust_env="NAUTICAL_TRUST_CACHE_PATH",
        )
        if override:
            candidates.append(override)

    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, ".nautical-cache"))

    taskdata = os.environ.get("TASKDATA")
    if taskdata:
        safe_taskdata = validated_user_dir(
            taskdata,
            label="TASKDATA",
            trust_env="NAUTICAL_TRUST_TASKDATA_PATH",
        )
        if safe_taskdata:
            candidates.append(os.path.join(safe_taskdata, ".nautical-cache"))

    safe_default_cache = validated_user_dir(
        nautical_cache_dir_path,
        label="XDG cache dir",
        trust_env="NAUTICAL_TRUST_CACHE_PATH",
    )
    if safe_default_cache:
        candidates.append(safe_default_cache)
    if os.environ.get("NAUTICAL_ALLOW_TMP_CACHE") == "1":
        candidates.append(os.path.join(tempfile.gettempdir(), "nautical-cache"))

    for p in candidates:
        if not p:
            continue
        try:
            if ensure_cache_dir(p):
                return p
        except Exception:
            continue

    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            print("[nautical] Anchor cache disabled: no writable cache dir found.", file=sys.stderr)
        except Exception:
            pass
    return ""


def cache_key(
    acf: str,
    anchor_mode: str,
    *,
    anchor_year_fmt: str,
    wrand_salt: str,
    local_tz_name: str,
    holiday_region: str,
) -> str:
    payload = "|".join(
        [
            acf,
            anchor_mode or "",
            anchor_year_fmt,
            wrand_salt,
            local_tz_name,
            holiday_region,
            "nautical-cache|v1",
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_path(base: str, key: str) -> str:
    if not base:
        return ""
    return os.path.join(base, f"{key}.jsonz")


def cache_lock_path(base: str, key: str) -> str:
    if not base:
        return ""
    return os.path.join(base, f".{key}.lock")
