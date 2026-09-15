"""Direct behavioral contracts for the core-bound cache API."""

from __future__ import annotations

from collections import OrderedDict
import fcntl
import io
import json
import os
from pathlib import Path
import stat
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
from nautical_core import cache_api
from nautical_core import cache_support


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

    def _binding(
        self,
        root: Path,
        *,
        config: list[str] | None = None,
        build_acf=None,
        atomic_replace=None,
        semantic_fingerprint=None,
        ttl: int = 0,
        time_mod=None,
    ):
        config = config if config is not None else ["config-a"]
        namespace = vars(core).copy()
        namespace.update(
            _CACHE_DIR=str(root),
            _CACHE_LOAD_MEM=OrderedDict(),
            _CACHE_LOAD_MEM_MAX=8,
            _CACHE_LOAD_MEM_TTL=300,
            ANCHOR_CACHE_DIR_OVERRIDE=str(root),
            ENABLE_ANCHOR_CACHE=True,
            ANCHOR_CACHE_TTL=ttl,
            _CACHE_LOCK_RETRIES=2,
            _CACHE_LOCK_SLEEP_BASE=0,
            _CACHE_LOCK_JITTER=0,
            _CACHE_LOCK_STALE_AFTER=300,
            scheduler_config_fingerprint=lambda: config[0],
            effective_config_fingerprint=lambda: config[0],
            time=time_mod or _Clock(),
            random=__import__("random"),
            os=os,
            json=json,
        )
        if build_acf is not None:
            namespace["build_acf"] = build_acf
        if atomic_replace is not None:
            namespace["_cache_atomic_replace"] = atomic_replace
        if semantic_fingerprint is not None:
            namespace["_cache_semantic_fingerprint"] = semantic_fingerprint
        namespace["_import_sibling"] = core._import_sibling
        binding = cache_api.for_core(namespace=namespace, module=core)
        namespace["_cache_lock"] = binding._cache_lock
        self._namespaces.append(namespace)
        return binding

    def test_cache_location_selection_prefers_safe_install_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            override = str(root / "shared-cache")
            taskdata = str(root / "taskdata")
            default = str(root / "xdg" / "nautical")
            checkout = str(Path(cache_support.__file__).resolve().parent / ".nautical-cache")

            with patch.dict(os.environ, {"TASKDATA": taskdata}, clear=False):
                os.environ.pop("NAUTICAL_ALLOW_TMP_CACHE", None)
                with patch.object(cache_support, "ensure_cache_dir", side_effect=lambda path: path == override):
                    self.assertEqual(cache_support.select_cache_dir(
                        anchor_cache_dir_override=override,
                        nautical_cache_dir_path=default,
                        validated_user_dir=lambda path, **_kwargs: path,
                    ), override)

                with patch.object(cache_support, "ensure_cache_dir", side_effect=lambda path: path == checkout):
                    self.assertEqual(cache_support.select_cache_dir(
                        anchor_cache_dir_override="",
                        nautical_cache_dir_path=default,
                        validated_user_dir=lambda path, **_kwargs: path,
                    ), checkout)

                managed = str(root / "taskdata" / ".nautical-cache")
                with patch.object(cache_support, "ensure_cache_dir", side_effect=lambda path: path == managed):
                    self.assertEqual(cache_support.select_cache_dir(
                        anchor_cache_dir_override="",
                        nautical_cache_dir_path=default,
                        validated_user_dir=lambda path, **_kwargs: path,
                    ), managed)

                with patch.object(cache_support, "ensure_cache_dir", side_effect=lambda path: path == default):
                    self.assertEqual(cache_support.select_cache_dir(
                        anchor_cache_dir_override="",
                        nautical_cache_dir_path=default,
                        validated_user_dir=lambda path, **_kwargs: path,
                    ), default)

    def test_clear_cache_environment_toggle_invokes_global_clear(self) -> None:
        with (
            patch.dict(os.environ, {"NAUTICAL_CLEAR_CACHES": "1"}),
            patch.object(core, "_clear_all_caches") as clear_all,
        ):
            core.parse_anchor_expr_to_dnf_cached("w:mon")

        clear_all.assert_called_once_with()

    def test_cache_miss_then_hit_returns_stable_copies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td))
            value = {"natural": "café", "next_dates": ["2026-09-10"]}
            self.assertIsNone(binding.cache_load("key"))
            self.assertTrue(binding.cache_save("key", value))
            self.assertEqual(
                stat.S_IMODE(Path(binding._cache_path("key")).stat().st_mode),
                0o600,
            )
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

    def test_explicit_gc_removes_quarantined_cache_payloads(self) -> None:
        import base64
        import zlib

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            malformed_shape = base64.b85encode(
                zlib.compress(json.dumps({"dnf": "invalid"}).encode("utf-8"))
            )
            for key, payload in (("broken", b"not-a-cache"), ("invalid", malformed_shape)):
                Path(binding._cache_path(key)).write_bytes(payload)
                self.assertIsNone(binding.cache_load(key))

            quarantined = list(root.glob("*.jsonz.bad.*"))
            self.assertEqual(len(quarantined), 2)
            result = binding.cache_gc(stale_tmp_age=0)

            self.assertGreaterEqual(result["temporary"], 2)
            self.assertEqual(list(root.glob("*.jsonz.bad.*")), [])

    def test_reader_retries_when_file_generation_changes_during_read(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            old_value = {"dnf": [[{"typ": "w", "spec": "mon"}]]}
            new_value = {"dnf": [[{"typ": "w", "spec": "fri"}]]}
            self.assertTrue(binding.cache_save("stable-read", old_value))
            self.assertTrue(binding.cache_save("replacement", new_value))
            path = Path(binding._cache_path("stable-read"))
            replacement = Path(binding._cache_path("replacement"))
            self._namespaces[-1]["_CACHE_LOAD_MEM"].clear()

            original_stat = os.stat
            calls = 0

            def stat_with_publish(target, *args, **kwargs):
                nonlocal calls
                if os.fspath(target) == os.fspath(path):
                    calls += 1
                    if calls == 2:
                        replacement.replace(path)
                return original_stat(target, *args, **kwargs)

            with patch.object(cache_api.os, "stat", side_effect=stat_with_publish):
                loaded = binding.cache_load("stable-read")

            self.assertEqual(loaded, new_value, f"reader did not observe replacement after {calls} stats")
            self.assertGreaterEqual(calls, 4)

    def test_save_removes_stale_temporary_files_for_same_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            stale = root / ".beeswax.stale.tmp"
            stale.write_text("partial write", encoding="utf-8")

            self.assertTrue(binding.cache_save("beeswax", {"natural": "Mondays"}))
            self.assertFalse(stale.exists())

    def test_explicit_gc_prunes_expired_overflow_and_stale_temp_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            clock = _Clock()
            clock.now = time.time()
            binding = self._binding(root, ttl=1, time_mod=clock)
            expired = root / "expired.jsonz"
            fresh_old = root / "fresh-old.jsonz"
            fresh_new = root / "fresh-new.jsonz"
            stale_tmp = root / ".orphan.tmp"
            unrelated = root / "notes.txt"
            for path in (expired, fresh_old, fresh_new, stale_tmp, unrelated):
                path.write_bytes(b"fixture")
            os.utime(expired, (clock.now - 10, clock.now - 10))
            os.utime(fresh_old, (clock.now - 0.5, clock.now - 0.5))
            os.utime(stale_tmp, (clock.now - 10, clock.now - 10))

            result = binding.cache_gc(max_entries=1, stale_tmp_age=1)

            self.assertEqual(result["expired"], 1)
            self.assertEqual(result["overflow"], 1)
            self.assertEqual(result["temporary"], 1)
            self.assertTrue(fresh_new.exists())
            self.assertFalse(expired.exists())
            self.assertFalse(fresh_old.exists())
            self.assertFalse(stale_tmp.exists())
            self.assertTrue(unrelated.exists())

    def test_cache_metrics_are_emitted_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            stderr = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {
                        "NAUTICAL_DIAG": "1",
                        "NAUTICAL_DIAG_METRICS": "1",
                        "XDG_CACHE_HOME": td,
                    },
                ),
                patch("sys.stderr", stderr),
            ):
                core._emit_cache_metrics()

            self.assertIn("nautical-metrics", stderr.getvalue())

            stderr.seek(0)
            stderr.truncate(0)
            with (
                patch.dict(
                    os.environ,
                    {"NAUTICAL_DIAG": "1", "NAUTICAL_DIAG_METRICS": ""},
                ),
                patch("sys.stderr", stderr),
            ):
                core._emit_cache_metrics()
            self.assertEqual(stderr.getvalue(), "")

    def test_unsupported_schema_and_invalid_shape_are_quarantined(self) -> None:
        import base64
        import zlib

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            invalid_payloads = {
                "missing-schema": {"dnf": []},
                "future-schema": {"_nautical_cache_version": 99, "dnf": []},
                "invalid-shape": {"_nautical_cache_version": 2, "dnf": "not-dnf"},
            }
            for key, payload in invalid_payloads.items():
                encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                Path(binding._cache_path(key)).write_bytes(
                    base64.b85encode(zlib.compress(encoded))
                )

            for key in invalid_payloads:
                with self.subTest(key=key):
                    self.assertIsNone(binding.cache_load(key))
                    self.assertEqual(len(list(root.glob(f"{key}.jsonz.bad.*"))), 1)

    def test_atomic_replace_failure_returns_false_without_publishing(self) -> None:
        def fail_replace(_source: str, _target: str) -> None:
            raise OSError("simulated replace failure")

        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td), atomic_replace=fail_replace)
            self.assertFalse(binding.cache_save("replace-failure", {"natural": "Mondays"}))
            self.assertFalse(Path(binding._cache_path("replace-failure")).exists())

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

    def test_semantic_fingerprint_changes_hint_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            fingerprint = ["semantic-test-a"]
            binding = self._binding(
                Path(td), semantic_fingerprint=lambda: fingerprint[0]
            )

            first = binding.cache_key_for_task("w:mon", "skip")
            fingerprint[0] = "semantic-test-b"
            second = binding.cache_key_for_task("w:mon", "skip")

            self.assertNotEqual(first, second)

    def test_task_key_memoizes_acf_work_until_its_binding_cache_is_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            calls: list[str] = []
            binding = self._binding(
                Path(td),
                build_acf=lambda expression: calls.append(expression) or f"acf:{expression}",
            )

            first = binding.cache_key_for_task("w:mon", "skip", "calendar")
            repeated = binding.cache_key_for_task("w:mon", "skip", "calendar")
            self.assertEqual(first, repeated)
            self.assertEqual(calls, ["w:mon"])

            binding.cache_key_for_task("w:tue", "skip", "calendar")
            self.assertEqual(calls, ["w:mon", "w:tue"])

            binding._cache_key_for_task_cached.cache_clear()
            binding.cache_key_for_task("w:mon", "skip", "calendar")
            self.assertEqual(calls, ["w:mon", "w:tue", "w:mon"])

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

    def test_dnf_fingerprint_tracks_parser_atoms_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td))

            def stat_for(mtime: int):
                def stat(path):
                    path = str(path)
                    if path.endswith("parsing/parser_atoms.py"):
                        return SimpleNamespace(st_mtime_ns=mtime, st_size=1)
                    return SimpleNamespace(st_mtime_ns=1, st_size=1)

                return stat

            with patch.object(cache_api.os, "stat", side_effect=stat_for(1)):
                before = binding._dnf_cache_fingerprint()
            with patch.object(cache_api.os, "stat", side_effect=stat_for(2)):
                after = self._binding(Path(td))._dnf_cache_fingerprint()

            self.assertNotEqual(before, after)

    def test_dnf_fingerprint_tracks_parser_frontend_source(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td))

            def stat_for(mtime: int):
                def stat(path):
                    path = str(path)
                    if path.endswith("parsing/parser_frontend.py"):
                        return SimpleNamespace(st_mtime_ns=mtime, st_size=1)
                    return SimpleNamespace(st_mtime_ns=1, st_size=1)

                return stat

            with patch.object(cache_api.os, "stat", side_effect=stat_for(1)):
                before = binding._dnf_cache_fingerprint()
            with patch.object(cache_api.os, "stat", side_effect=stat_for(2)):
                after = self._binding(Path(td))._dnf_cache_fingerprint()

            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
