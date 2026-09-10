"""Direct behavioral contracts for the core-bound cache API."""

from __future__ import annotations

from collections import OrderedDict
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
from nautical_core import cache_api


class _Clock:
    def __init__(self) -> None:
        self.now = 1_700_000_000.0

    def time(self) -> float:
        return self.now

    def time_ns(self) -> int:
        return int(self.now * 1_000_000_000)

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class CacheApiContractTests(unittest.TestCase):
    _namespaces: list[dict] = []

    def _binding(self, root: Path, *, config: list[str] | None = None):
        config = config if config is not None else ["config-a"]
        namespace = vars(core).copy()
        namespace.update(
            _CACHE_DIR=str(root),
            _CACHE_LOAD_MEM=OrderedDict(),
            _CACHE_LOAD_MEM_MAX=8,
            _CACHE_LOAD_MEM_TTL=300,
            ANCHOR_CACHE_DIR_OVERRIDE=str(root),
            ENABLE_ANCHOR_CACHE=True,
            ANCHOR_CACHE_TTL=0,
            _CACHE_LOCK_RETRIES=2,
            _CACHE_LOCK_SLEEP_BASE=0,
            _CACHE_LOCK_JITTER=0,
            _CACHE_LOCK_STALE_AFTER=300,
            scheduler_config_fingerprint=lambda: config[0],
            effective_config_fingerprint=lambda: config[0],
            time=_Clock(),
            random=__import__("random"),
            os=os,
            json=json,
        )
        namespace["_import_sibling"] = core._import_sibling
        binding = cache_api.for_core(namespace=namespace, module=core)
        namespace["_cache_lock"] = binding._cache_lock
        self._namespaces.append(namespace)
        return binding

    def test_cache_miss_then_hit_returns_stable_copies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td))
            value = {"natural": "café", "next_dates": ["2026-09-10"]}
            self.assertIsNone(binding.cache_load("key"))
            self.assertTrue(binding.cache_save("key", value))
            loaded = binding.cache_load("key")
            self.assertEqual(loaded, value)
            self.assertIsNot(loaded, value)
            loaded["next_dates"].append("2026-09-11")
            self.assertEqual(binding.cache_load("key"), value)

    def test_corrupt_payload_is_quarantined_and_next_read_is_clean_miss(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            path = Path(binding._cache_path("broken"))
            path.write_bytes(b"not valid cache data")
            self.assertIsNone(binding.cache_load("broken"))
            self.assertFalse(path.exists())
            quarantined = list(root.glob("broken.jsonz.bad.*"))
            self.assertEqual(len(quarantined), 1)
            self.assertEqual(quarantined[0].read_bytes(), b"not valid cache data")
            self.assertIsNone(binding.cache_load("broken"))

    def test_lock_refusal_is_retry_safe_and_release_allows_save(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            lock_path = Path(binding._cache_lock_path("locked"))
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertFalse(binding.cache_save("locked", {"natural": "held"}))
                self.assertTrue(lock_path.exists())
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
            self.assertTrue(binding.cache_save("locked", {"natural": "released"}))

    def test_unicode_payload_is_written_unescaped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            self.assertTrue(binding.cache_save("unicode", {"natural": "東京—café"}))
            encoded = Path(binding._cache_path("unicode")).read_bytes()
            decoded = __import__("zlib").decompress(__import__("base64").b85decode(encoded))
            self.assertIn("東京".encode("utf-8"), decoded)

    def test_configuration_and_calendar_fingerprints_select_distinct_keys(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            config = ["config-a"]
            binding = self._binding(Path(td), config=config)
            first = binding.cache_key_for_task("m:1", "next", "calendar-a")
            self.assertEqual(first, binding.cache_key_for_task("m:1", "next", "calendar-a"))
            config[0] = "config-b"
            second = binding.cache_key_for_task("m:1", "next", "calendar-a")
            self.assertNotEqual(first, second)
            self.assertNotEqual(second, binding.cache_key_for_task("m:1", "next", "calendar-b"))

    def test_instances_do_not_share_memory_entries_or_locks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = self._binding(root)
            second = self._binding(root)
            self.assertTrue(first.cache_save("private", {"natural": "one"}))
            self.assertEqual(first.cache_load("private"), {"natural": "one"})
            self.assertIsNot(self._namespaces[-1]["_CACHE_LOAD_MEM"], self._namespaces[-2]["_CACHE_LOAD_MEM"])
            self.assertIn("private", self._namespaces[-2]["_CACHE_LOAD_MEM"])
            self.assertNotIn("private", self._namespaces[-1]["_CACHE_LOAD_MEM"])
            self.assertEqual(first._cache_lock_path("private"), second._cache_lock_path("private"))
            with first.safe_lock(first._cache_lock_path("private"), retries=1, sleep_base=0) as held:
                self.assertTrue(held)
                with second.safe_lock(second._cache_lock_path("private"), retries=1, sleep_base=0) as competing:
                    self.assertFalse(competing)

    def test_semantic_fingerprint_tracks_canonical_parser_sources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            seen: list[str] = []

            def parser_stat(parser_mtime: int):
                def stat(path):
                    path = str(path)
                    seen.append(path)
                    if path.endswith("parsing/parser_dnf.py"):
                        return SimpleNamespace(st_mtime_ns=parser_mtime, st_size=1)
                    return SimpleNamespace(st_mtime_ns=1, st_size=1)

                return stat

            with patch.object(cache_api.os, "stat", side_effect=parser_stat(1)):
                before = self._binding(Path(td))._cache_semantic_fingerprint()
            with patch.object(cache_api.os, "stat", side_effect=parser_stat(2)):
                after = self._binding(Path(td))._cache_semantic_fingerprint()

            self.assertTrue(any(path.endswith("parsing/parser_dnf.py") for path in seen))
            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
