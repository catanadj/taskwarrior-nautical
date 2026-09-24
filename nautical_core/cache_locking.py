from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Protocol


class TimePort(Protocol):
    def time(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class RandomPort(Protocol):
    def uniform(self, start: float, end: float) -> float: ...


class FcntlPort(Protocol):
    LOCK_EX: int
    LOCK_NB: int
    LOCK_UN: int

    def flock(self, file_descriptor: int, operation: int) -> None: ...


def cache_dir(
    current_cache_dir: Any,
    *,
    anchor_cache_dir_override: Any,
    nautical_cache_dir_path: Any,
    validated_user_dir: Any,
    select_cache_dir: Any,
) -> Any:
    if current_cache_dir is not None:
        return current_cache_dir
    return select_cache_dir(
        anchor_cache_dir_override=anchor_cache_dir_override,
        nautical_cache_dir_path=nautical_cache_dir_path,
        validated_user_dir=validated_user_dir,
    )


def safe_lock_sleep_once(
    sleep_base: float,
    jitter: float,
    *,
    time_mod: TimePort,
    random_mod: RandomPort,
) -> None:
    try:
        delay = float(sleep_base or 0.0)
    except Exception:
        delay = 0.0
    if jitter:
        try:
            delay += random_mod.uniform(0.0, float(jitter))
        except Exception:
            pass
    if delay > 0:
        time_mod.sleep(delay)


def safe_lock_ensure_parent(path_str: str, mkdir: bool, *, os_mod: Any) -> None:
    if not mkdir:
        return
    try:
        parent = os_mod.path.dirname(path_str)
        if parent:
            os_mod.makedirs(parent, exist_ok=True)
    except Exception:
        pass


def safe_lock_age(path_str: str, *, time_mod: TimePort, os_mod: Any) -> float | None:
    try:
        with open(path_str, "r", encoding="utf-8") as fh:
            head = fh.read(64)
        parts = head.strip().split()
        if len(parts) >= 2:
            return time_mod.time() - float(parts[1])
    except Exception:
        pass
    try:
        st = os_mod.stat(path_str)
        return time_mod.time() - float(st.st_mtime)
    except Exception:
        return None


def safe_lock_stale_pid(
    path_str: str,
    stale_after: float | None,
    *,
    time_mod: TimePort,
    os_mod: Any,
) -> bool:
    try:
        with open(path_str, "r", encoding="utf-8") as fh:
            head = fh.read(64)
        parts = head.strip().split()
        pid_str = parts[0] if parts else ""
        pid = int(pid_str)
        if pid <= 0:
            return True
        if stale_after is not None and len(parts) >= 2:
            try:
                age = time_mod.time() - float(parts[1])
                if age < float(stale_after):
                    return False
            except Exception:
                pass
        try:
            os_mod.kill(pid, 0)
            return False
        except PermissionError:
            return False
        except ProcessLookupError:
            return True
        except Exception:
            return False
    except Exception:
        return False


@contextmanager
def safe_lock_fcntl_context(
    path_str: str,
    *,
    tries: int,
    sleep_base: float,
    jitter: float,
    mode: int,
    mkdir: bool,
    safe_lock_ensure_parent: Any,
    safe_lock_sleep_once: Any,
    fcntl_mod: FcntlPort,
    os_mod: Any,
) -> Any:
    lf = None
    acquired = False
    safe_lock_ensure_parent(path_str, mkdir)
    try:
        fd = os_mod.open(path_str, os_mod.O_CREAT | os_mod.O_RDWR, mode)
        try:
            os_mod.fchmod(fd, mode)
        except Exception:
            pass
        lf = os_mod.fdopen(fd, "a", encoding="utf-8")
        for _ in range(tries):
            try:
                fcntl_mod.flock(lf.fileno(), fcntl_mod.LOCK_EX | fcntl_mod.LOCK_NB)
                acquired = True
                break
            except Exception:
                safe_lock_sleep_once(sleep_base, jitter)
    except Exception:
        lf = None
    try:
        yield acquired
    finally:
        try:
            if acquired and lf is not None:
                fcntl_mod.flock(lf.fileno(), fcntl_mod.LOCK_UN)
        except Exception:
            pass
        try:
            if lf is not None:
                lf.close()
        except Exception:
            pass


@contextmanager
def safe_lock_excl_context(
    path_str: str,
    *,
    tries: int,
    sleep_base: float,
    jitter: float,
    mode: int,
    mkdir: bool,
    stale_after: float | None,
    safe_lock_ensure_parent: Any,
    safe_lock_stale_pid: Any,
    safe_lock_age: Any,
    safe_lock_sleep_once: Any,
    os_mod: Any,
    time_mod: TimePort,
) -> Any:
    fd = None
    acquired = False
    for _ in range(tries):
        safe_lock_ensure_parent(path_str, mkdir)
        try:
            fd = os_mod.open(path_str, os_mod.O_CREAT | os_mod.O_EXCL | os_mod.O_WRONLY, mode)
            try:
                os_mod.fchmod(fd, mode)
            except Exception:
                pass
            try:
                payload = f"{os_mod.getpid()} {int(time_mod.time())}\n"
                os_mod.write(fd, payload.encode("ascii", "replace"))
            except Exception:
                pass
            acquired = True
            break
        except FileExistsError:
            pid_stale = safe_lock_stale_pid(path_str, stale_after)
            age_stale = False
            if stale_after is not None:
                age = safe_lock_age(path_str)
                if age is not None and age >= float(stale_after):
                    age_stale = True
            if pid_stale and age_stale:
                try:
                    os_mod.unlink(path_str)
                except Exception:
                    pass
            else:
                safe_lock_sleep_once(sleep_base, jitter)
        except Exception:
            break
    try:
        yield acquired
    finally:
        try:
            if acquired and fd is not None:
                os_mod.close(fd)
        except Exception:
            pass
        try:
            if acquired and fd is not None:
                os_mod.unlink(path_str)
        except Exception:
            pass


@contextmanager
def safe_lock(
    path: Any,
    *,
    retries: int = 6,
    sleep_base: float = 0.05,
    jitter: float = 0.0,
    mode: int = 0o600,
    mkdir: bool = True,
    stale_after: float | None = 60.0,
    fcntl_mod: FcntlPort | None,
    os_mod: Any,
    time_mod: TimePort,
    random_mod: RandomPort,
) -> Any:
    path_str = str(path) if path else ""
    if not path_str:
        yield False
        return

    tries = max(1, int(retries or 0))

    _ensure_parent = lambda p, m: safe_lock_ensure_parent(p, m, os_mod=os_mod)
    _sleep_once = lambda base, jit: safe_lock_sleep_once(base, jit, time_mod=time_mod, random_mod=random_mod)
    _age = lambda p: safe_lock_age(p, time_mod=time_mod, os_mod=os_mod)
    _stale_pid = lambda p, s: safe_lock_stale_pid(p, s, time_mod=time_mod, os_mod=os_mod)

    if fcntl_mod is not None:
        with safe_lock_fcntl_context(
            path_str,
            tries=tries,
            sleep_base=sleep_base,
            jitter=jitter,
            mode=mode,
            mkdir=mkdir,
            safe_lock_ensure_parent=_ensure_parent,
            safe_lock_sleep_once=_sleep_once,
            fcntl_mod=fcntl_mod,
            os_mod=os_mod,
        ) as acquired:
            yield acquired
        return

    with safe_lock_excl_context(
        path_str,
        tries=tries,
        sleep_base=sleep_base,
        jitter=jitter,
        mode=mode,
        mkdir=mkdir,
        stale_after=stale_after,
        safe_lock_ensure_parent=_ensure_parent,
        safe_lock_stale_pid=_stale_pid,
        safe_lock_age=_age,
        safe_lock_sleep_once=_sleep_once,
        os_mod=os_mod,
        time_mod=time_mod,
    ) as acquired:
        yield acquired


@contextmanager
def cache_lock(
    key: str,
    *,
    cache_lock_path: Any,
    safe_lock: Any,
    cache_lock_retries: int,
    cache_lock_sleep_base: float,
    cache_lock_jitter: float,
    cache_lock_stale_after: float,
) -> Any:
    lock_path = cache_lock_path(key)
    if not lock_path:
        yield False
        return
    with safe_lock(
        lock_path,
        retries=cache_lock_retries,
        sleep_base=cache_lock_sleep_base,
        jitter=cache_lock_jitter,
        mode=0o600,
        mkdir=True,
        stale_after=cache_lock_stale_after,
    ) as acquired:
        yield acquired


def bind_locking(
    *,
    cache_lock_path: Any,
    retries: int,
    sleep_base: float,
    jitter: float,
    stale_after: float,
    fcntl_mod: FcntlPort | None,
    os_mod: Any,
    time_mod: TimePort,
    random_mod: RandomPort,
) -> Any:
    """Bind lock dependencies once for one core facade instance."""
    def bound_safe_lock(path: Any, **kwargs: Any) -> Any:
        return safe_lock(
            path,
            fcntl_mod=fcntl_mod,
            os_mod=os_mod,
            time_mod=time_mod,
            random_mod=random_mod,
            **kwargs,
        )

    def bound_cache_lock(key: Any) -> Any:
        return cache_lock(
            key,
            cache_lock_path=cache_lock_path,
            safe_lock=bound_safe_lock,
            cache_lock_retries=retries,
            cache_lock_sleep_base=sleep_base,
            cache_lock_jitter=jitter,
            cache_lock_stale_after=stale_after,
        )

    return SimpleNamespace(safe_lock=bound_safe_lock, cache_lock=bound_cache_lock)
