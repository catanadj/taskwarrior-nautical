"""Storage and filesystem-safety golden tests."""

from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

from dev_tools.golden_tests.support import expect
from nautical_core.cache_locking import safe_lock


def test_safe_lock_fcntl_contention():
    """safe_lock should refuse contention from another process."""
    if fcntl is None:
        return
    with tempfile.TemporaryDirectory() as temporary:
        lock_path = Path(temporary) / ".nautical_fcntl.lock"
        ready_path = Path(temporary) / ".nautical_fcntl.ready"
        script = (
            "import os, time\n"
            "import fcntl\n"
            "lp = os.environ['LOCK_PATH']\n"
            "rp = os.environ['READY_PATH']\n"
            "fd = os.open(lp, os.O_CREAT | os.O_RDWR, 0o600)\n"
            "f = os.fdopen(fd, 'a', encoding='utf-8')\n"
            "fcntl.flock(f.fileno(), fcntl.LOCK_EX)\n"
            "with open(rp, 'w', encoding='utf-8') as r:\n"
            "    r.write('ready')\n"
            "time.sleep(1.0)\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            env={
                **os.environ,
                "LOCK_PATH": str(lock_path),
                "READY_PATH": str(ready_path),
            },
        )
        try:
            for _ in range(50):
                if ready_path.exists():
                    break
                time.sleep(0.02)
            expect(ready_path.exists(), "lock holder did not start")
            with safe_lock(
                lock_path,
                retries=2,
                sleep_base=0.01,
                jitter=0.0,
                fcntl_mod=fcntl,
                os_mod=os,
                time_mod=time,
                random_mod=random,
            ) as acquired:
                expect(not acquired, "safe_lock should not acquire while locked")
        finally:
            process.wait(timeout=3.0)
        with safe_lock(
            lock_path,
            retries=2,
            sleep_base=0.01,
            jitter=0.0,
            fcntl_mod=fcntl,
            os_mod=os,
            time_mod=time,
            random_mod=random,
        ) as acquired:
            expect(acquired, "safe_lock should acquire after lock release")


TESTS = (test_safe_lock_fcntl_contention,)
