"""Direct behavioral contracts for the core-bound cache API."""

from __future__ import annotations

from collections import OrderedDict
import builtins
from dataclasses import fields, is_dataclass
import fcntl
import importlib
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Iterator, get_type_hints
from types import SimpleNamespace
from unittest.mock import patch

import nautical_core as core
import nautical_core.cache_api as cache_api
import nautical_core.cache_facade as cache_facade
import nautical_core.cache_locking as cache_locking
import nautical_core.cache_payload as cache_payload
import nautical_core.cache_ports as cache_ports
import nautical_core.cache_support as cache_support
from nautical_core.core_context import CacheState


def _import_core_sibling(name: str):
    return importlib.import_module(f"nautical_core.{name}")


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
    def test_cache_directory_validation_uses_a_shared_callback_contract(self) -> None:
        for owner in (cache_support.nautical_cache_dir, cache_support.select_cache_dir):
            with self.subTest(owner=owner.__name__):
                hints = get_type_hints(owner)
                self.assertIs(hints["validated_user_dir"], cache_support.ValidatedUserDir)

    def test_cache_directory_binding_uses_a_typed_selector_contract(self) -> None:
        hints = get_type_hints(cache_locking.cache_dir)
        self.assertEqual(hints["current_cache_dir"], str | None)
        self.assertIs(hints["validated_user_dir"], cache_support.ValidatedUserDir)
        self.assertIs(hints["select_cache_dir"], cache_locking.CacheDirectorySelector)
        self.assertEqual(hints["return"], str)

    def test_cache_lock_contexts_have_typed_inputs_and_yields(self) -> None:
        safe_hints = get_type_hints(cache_locking.safe_lock)
        self.assertIs(safe_hints["path"], object)
        self.assertEqual(safe_hints["return"], Iterator[bool])
        cache_hints = get_type_hints(cache_locking.cache_lock)
        self.assertEqual(cache_hints["cache_lock_path"], cache_locking.Callable[[str], str])
        self.assertIs(cache_hints["safe_lock"], cache_locking.BoundSafeLock)
        self.assertEqual(cache_hints["return"], Iterator[bool])

    def test_cache_locking_callbacks_have_concrete_context_contracts(self) -> None:
        fcntl_hints = get_type_hints(cache_locking.safe_lock_fcntl_context)
        self.assertEqual(
            fcntl_hints["safe_lock_ensure_parent"],
            cache_locking.Callable[[str, bool], None],
        )
        self.assertEqual(
            fcntl_hints["safe_lock_sleep_once"],
            cache_locking.Callable[[float, float], None],
        )
        self.assertEqual(fcntl_hints["return"], Iterator[bool])

        exclusive_hints = get_type_hints(cache_locking.safe_lock_excl_context)
        self.assertEqual(
            exclusive_hints["safe_lock_stale_pid"],
            cache_locking.Callable[[str, float | None], bool],
        )
        self.assertEqual(
            exclusive_hints["safe_lock_age"],
            cache_locking.Callable[[str], float | None],
        )
        self.assertEqual(exclusive_hints["return"], Iterator[bool])

        binding_hints = get_type_hints(cache_locking.bind_locking)
        self.assertEqual(
            binding_hints["cache_lock_path"],
            cache_locking.Callable[[str], str],
        )
        self.assertIs(binding_hints["return"], cache_locking.BoundLocking)

    def test_cache_facade_uses_narrow_cache_capabilities(self) -> None:
        metrics_hints = get_type_hints(cache_facade.emit_metrics)
        clear_hints = get_type_hints(cache_facade.clear_all)
        self.assertEqual(
            metrics_hints["caches"],
            cache_facade.Iterable[tuple[str, cache_facade.CacheInfoPort]],
        )
        self.assertEqual(metrics_hints["warn_once"], cache_facade.Callable[[str, str], None])
        self.assertIs(clear_hints["memory_cache"], cache_facade.MemoryCacheClearPort)
        self.assertEqual(
            clear_hints["caches"],
            cache_facade.Iterable[cache_facade.CacheClearPort],
        )
        self.assertIs(clear_hints["position_selection"], cache_facade.PositionSelectionClearPort)
        self.assertIs(clear_hints["selection_matcher"], cache_facade.CacheClearPort)

    def test_cache_runtime_overrides_have_concrete_callback_contracts(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertEqual(hints["atomic_replace_override"], Callable[[str, str], None] | None)
        self.assertEqual(hints["clone_payload_override"], Callable[[dict], dict] | None)
        self.assertEqual(hints["normalize_dnf_override"], Callable[[object], object] | None)
        self.assertEqual(hints["payload_shape_override"], Callable[[object], bool] | None)
        self.assertEqual(hints["semantic_fingerprint_override"], Callable[[], str] | None)
        self.assertEqual(
            hints["cache_lock_override"],
            Callable[[str], ContextManager[bool]] | None,
        )
        self.assertEqual(hints["is_dnf_like_override"], Callable[[object], bool] | None)

    def test_cache_runtime_clock_and_random_ports_are_narrow(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertIs(hints["clock"], cache_ports.CacheClockPort)
        self.assertIs(hints["random"], cache_ports.RandomPort)

    def test_cache_runtime_filesystem_uses_a_direct_use_capability(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertIs(hints["filesystem"], cache_ports.FilesystemPort)

    def test_cache_runtime_fcntl_uses_the_locking_capability(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertEqual(
            hints["fcntl"],
            cache_ports.FcntlPort | None,
        )

    def test_cache_io_owners_share_filesystem_and_clock_capabilities(self) -> None:
        for owner in (cache_payload.cache_load, cache_payload.cache_save, cache_payload.cache_gc):
            hints = get_type_hints(owner)
            with self.subTest(owner=owner.__name__):
                self.assertIs(hints["os_mod"], cache_ports.FilesystemPort)
                if "time_mod" in hints:
                    self.assertIs(hints["time_mod"], cache_ports.ClockPort)

        lock_hints = get_type_hints(cache_locking.safe_lock)
        self.assertIs(lock_hints["os_mod"], cache_ports.FilesystemPort)
        self.assertIs(lock_hints["time_mod"], cache_ports.ClockPort)
        self.assertIs(lock_hints["random_mod"], cache_ports.RandomPort)
        self.assertEqual(lock_hints["fcntl_mod"], cache_ports.FcntlPort | None)

    def test_cache_runtime_json_uses_its_consumed_operations(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertIs(hints["json"], cache_ports.JsonPort)

    def test_cache_runtime_compression_uses_its_consumed_operations(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertIs(hints["compression"], cache_ports.CompressionPort)

    def test_cache_runtime_encoding_and_temporary_file_ports_are_narrow(self) -> None:
        hints = get_type_hints(cache_api.CacheRuntimeDependencies)
        self.assertIs(hints["base64"], cache_ports.Base64Port)
        self.assertIs(hints["tempfile"], cache_ports.TemporaryFilePort)

    def test_cache_payload_consumes_shared_io_port_contracts(self) -> None:
        load_hints = get_type_hints(cache_payload.cache_load)
        save_hints = get_type_hints(cache_payload.cache_save)
        self.assertIs(load_hints["json_mod"], cache_ports.JsonPort)
        self.assertIs(load_hints["zlib_mod"], cache_ports.CompressionPort)
        self.assertIs(load_hints["base64_mod"], cache_ports.Base64Port)
        self.assertIs(save_hints["json_mod"], cache_ports.JsonPort)
        self.assertIs(save_hints["zlib_mod"], cache_ports.CompressionPort)
        self.assertIs(save_hints["base64_mod"], cache_ports.Base64Port)
        self.assertIs(save_hints["tempfile_mod"], cache_ports.TemporaryFilePort)

    def test_payload_load_and_save_share_the_cache_state_model(self) -> None:
        load_signature = get_type_hints(cache_payload.cache_load)
        save_signature = get_type_hints(cache_payload.cache_save)
        self.assertIs(load_signature.get("cache_state"), CacheState)
        self.assertNotIn("cache_load_mem", load_signature)
        self.assertNotIn("cache_load_mem_ttl", load_signature)
        self.assertNotIn("cache_load_mem_max", load_signature)
        self.assertIs(save_signature.get("cache_state"), CacheState)
        self.assertNotIn("cache_load_mem", save_signature)

    def test_cache_shape_predicates_accept_untrusted_objects(self) -> None:
        for predicate, argument in (
            (cache_payload.is_atom_like, "atom"),
            (cache_payload.is_selection_like, "value"),
            (cache_payload.is_factor_like, "value"),
        ):
            with self.subTest(predicate=predicate.__name__):
                self.assertEqual(get_type_hints(predicate)[argument], object)

        hints = get_type_hints(cache_payload.is_dnf_like)
        self.assertEqual(hints["dnf"], object)
        self.assertEqual(hints["is_atom_like"], Callable[[object], bool])

    def test_cache_payload_shape_gate_rejects_non_mapping_json(self) -> None:
        hints = get_type_hints(cache_payload.cache_payload_shape_ok)
        self.assertIs(hints["obj"], object)
        self.assertFalse(
            cache_payload.cache_payload_shape_ok(
                ["not", "a", "cache object"],
                is_dnf_like=lambda _value: True,
            )
        )

    def test_bounded_decompress_uses_a_typed_zlib_port(self) -> None:
        hints = get_type_hints(cache_payload._bounded_decompress)
        self.assertIs(hints["zlib_mod"], cache_ports.CompressionPort)

    def test_atomic_replace_uses_a_typed_filesystem_port(self) -> None:
        hints = get_type_hints(cache_payload.cache_atomic_replace)
        self.assertIs(hints["os_mod"], cache_ports.AtomicReplacePort)

    def test_cache_load_and_save_callbacks_have_explicit_callable_contracts(self) -> None:
        load_hints = get_type_hints(cache_payload.cache_load)
        self.assertEqual(load_hints["cache_path"], Callable[[str], str])
        self.assertEqual(load_hints["clone_cache_payload"], Callable[[dict], dict])
        self.assertEqual(load_hints["normalize_dnf_cached"], Callable[[object], object])
        self.assertEqual(load_hints["cache_payload_shape_ok"], Callable[[object], bool])
        self.assertEqual(load_hints["diag"], Callable[[str], None])

        save_hints = get_type_hints(cache_payload.cache_save)
        self.assertEqual(save_hints["cache_path"], Callable[[str], str])
        self.assertEqual(save_hints["cache_dir"], Callable[[], str])
        self.assertEqual(save_hints["cache_lock"], Callable[[str], ContextManager[bool]])
        self.assertEqual(save_hints["diag"], Callable[[str], None])
        self.assertEqual(save_hints["cache_atomic_replace"], Callable[[str, str], None])

    def test_cache_gc_declares_typed_locking_callbacks(self) -> None:
        hints = get_type_hints(cache_payload.cache_gc)
        self.assertEqual(hints["cache_lock"], Callable[[str], ContextManager[bool]])
        self.assertEqual(hints["stale_lock_check"], Callable[[str, float], bool])
        self.assertIs(hints["return"], cache_payload.CacheGcResult)
        self.assertEqual(
            get_type_hints(hints["return"]),
            {
                "removed": int,
                "bytes": int,
                "temporary": int,
                "expired": int,
                "overflow": int,
                "locks_removed": int,
                "locks_skipped": int,
                "errors": int,
            },
        )

    def test_cached_task_key_uses_typed_acf_and_cache_key_builders(self) -> None:
        hints = get_type_hints(cache_payload.cache_key_for_task_cached)
        self.assertEqual(hints["build_acf"], Callable[[str], str])
        self.assertIs(hints["cache_key"], cache_payload._CacheKeyCallback)

    def test_cache_payload_shape_validator_declares_typed_dnf_callback(self) -> None:
        self.assertEqual(
            get_type_hints(cache_payload.cache_payload_shape_ok)["is_dnf_like"],
            Callable[[object], bool],
        )

    _namespaces: list[dict] = []

    def test_bound_locking_is_a_named_immutable_dependency_contract(self) -> None:
        bound = cache_locking.bind_locking(
            cache_lock_path=lambda _key: "",
            retries=1,
            sleep_base=0.0,
            jitter=0.0,
            stale_after=60.0,
            fcntl_mod=fcntl,
            os_mod=os,
            time_mod=_Clock(),
            random_mod=SimpleNamespace(uniform=lambda _start, _end: 0.0),
        )

        self.assertTrue(is_dataclass(bound))
        self.assertTrue(type(bound).__dataclass_params__.frozen)
        self.assertEqual(
            tuple(field.name for field in fields(bound)),
            ("safe_lock", "cache_lock"),
        )

    def test_cache_lock_parent_setup_does_not_hide_unexpected_errors(self) -> None:
        class BrokenPath:
            @staticmethod
            def dirname(_path: str) -> str:
                raise RuntimeError("path adapter invariant failed")

        with self.assertRaisesRegex(RuntimeError, "path adapter invariant failed"):
            cache_locking.safe_lock_ensure_parent(
                "cache/lock", True, os_mod=SimpleNamespace(path=BrokenPath())
            )

    def test_cache_lock_delay_does_not_hide_unexpected_jitter_errors(self) -> None:
        class BrokenRandom:
            @staticmethod
            def uniform(_start: float, _end: float) -> float:
                raise RuntimeError("random adapter invariant failed")

        with self.assertRaisesRegex(RuntimeError, "random adapter invariant failed"):
            cache_locking.safe_lock_sleep_once(
                0.1, 0.1, time_mod=_Clock(), random_mod=BrokenRandom()
            )

    def test_runtime_context_uses_explicit_filesystem_clock_and_lock(self) -> None:
        fake_filesystem = SimpleNamespace()
        fake_clock = _Clock()
        fake_random = SimpleNamespace(uniform=lambda _start, _end: 0.0)
        fake_lock = SimpleNamespace(LOCK_EX=1, LOCK_NB=2, LOCK_UN=4, flock=lambda *_args: None)
        siblings = {
            "cache_support": cache_support,
            "cache_locking": cache_locking,
            "cache_payload": cache_payload,
        }
        values = {
            "_CACHE_LOAD_MEM": OrderedDict(),
            "_CACHE_LOAD_MEM_MAX": 8,
            "_CACHE_LOAD_MEM_TTL": 300,
            "os": fake_filesystem,
            "time": fake_clock,
            "random": fake_random,
            "fcntl": fake_lock,
            "json": json,
            "zlib": __import__("zlib"),
            "base64": __import__("base64"),
        }
        context = cache_api.CoreContext(
            namespace=values,
            import_sibling=lambda name: siblings[name],
        )

        try:
            runtime = cache_api._binding_context(None, namespace=None, context=context).runtime
        except AttributeError as exc:
            self.fail(f"cache binding does not expose an explicit runtime bundle: {exc}")

        self.assertIs(runtime.filesystem, fake_filesystem)
        self.assertIs(runtime.clock, fake_clock)
        self.assertIs(runtime.random, fake_random)
        self.assertIs(runtime.fcntl, fake_lock)

    def test_fcntl_loader_does_not_hide_unexpected_import_errors(self) -> None:
        real_import = builtins.__import__

        def fail_fcntl_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "fcntl":
                raise RuntimeError("fcntl loader invariant failed")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with patch("builtins.__import__", side_effect=fail_fcntl_import):
                with self.assertRaisesRegex(RuntimeError, "fcntl loader invariant failed"):
                    importlib.reload(cache_api)
        finally:
            importlib.reload(cache_api)

    def _binding(
        self,
        root: Path,
        *,
        config: list[str] | None = None,
        build_acf=None,
        atomic_replace=None,
        semantic_fingerprint=None,
        import_sibling: Any = None,
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
        namespace["_import_sibling"] = import_sibling or _import_core_sibling
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
            patch.object(cache_facade, "clear_all") as clear_all,
        ):
            core.parse_anchor_expr_to_dnf_cached("w:mon")

        clear_all.assert_called_once()

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

            with patch.object(os, "stat", side_effect=stat_with_publish):
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

    def test_cache_gc_does_not_hide_unexpected_stat_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / "candidate.jsonz"
            candidate.write_bytes(b"cache")
            binding = self._binding(root)
            real_stat = os.stat

            def fail_candidate_stat(path, *args, **kwargs):
                if os.fspath(path) == str(candidate):
                    raise RuntimeError("filesystem stat invariant failed")
                return real_stat(path, *args, **kwargs)

            with patch.object(os, "stat", side_effect=fail_candidate_stat):
                with self.assertRaisesRegex(RuntimeError, "filesystem stat invariant failed"):
                    binding.cache_gc()

    def test_cache_gc_does_not_hide_unexpected_unlink_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            candidate = root / ".orphan.tmp"
            candidate.write_bytes(b"partial")
            binding = self._binding(root)
            real_unlink = os.unlink

            def fail_candidate_unlink(path, *args, **kwargs):
                if os.fspath(path) == str(candidate):
                    raise RuntimeError("filesystem unlink invariant failed")
                return real_unlink(path, *args, **kwargs)

            with patch.object(os, "unlink", side_effect=fail_candidate_unlink):
                with self.assertRaisesRegex(RuntimeError, "filesystem unlink invariant failed"):
                    binding.cache_gc(stale_tmp_age=0)

    def test_cache_metrics_are_emitted_only_when_enabled(self) -> None:
        from functools import lru_cache

        @lru_cache
        def cached(value: str) -> str:
            return value

        cached("w:mon")

        def emit_metrics() -> None:
            cache_facade.emit_metrics(
                (("normalize_acf", cached),),
                lambda _key, message: print(message, file=sys.stderr),
            )

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
                emit_metrics()

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
                emit_metrics()
            self.assertEqual(stderr.getvalue(), "")

    def test_cache_metrics_continue_after_one_cache_info_failure(self) -> None:
        class BrokenCache:
            @staticmethod
            def cache_info() -> str:
                raise RuntimeError("optional metrics failure")

        messages: list[str] = []
        with patch.dict(os.environ, {"NAUTICAL_DIAG_METRICS": "1"}):
            cache_facade.emit_metrics(
                (("broken", BrokenCache()), ("healthy", SimpleNamespace(cache_info=lambda: "hits=3"))),
                lambda _key, message: messages.append(message),
            )

        self.assertEqual(len(messages), 1)
        self.assertIn("healthy: hits=3", messages[0])
        self.assertNotIn("broken", messages[0])

    def test_cache_clear_continues_after_one_cache_clear_failure(self) -> None:
        calls: list[str] = []

        class Cache:
            def __init__(self, name: str, broken: bool = False) -> None:
                self.name = name
                self.broken = broken

            def cache_clear(self) -> None:
                calls.append(self.name)
                if self.broken:
                    raise RuntimeError("cache clear failed")

        memory_cache = SimpleNamespace(clear=lambda: calls.append("memory"))
        position = SimpleNamespace(clear_candidate_cache=lambda: calls.append("position"))
        matcher = Cache("matcher")

        cache_facade.clear_all(
            memory_cache,
            (Cache("broken", broken=True), Cache("healthy")),
            position_selection=position,
            selection_matcher=matcher,
        )

        self.assertEqual(calls, ["memory", "broken", "healthy", "position", "matcher"])

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

    def test_quarantine_does_not_hide_unexpected_replace_errors(self) -> None:
        import base64
        import zlib

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            key = "quarantine-runtime-error"
            cache_path = Path(binding._cache_path(key))
            encoded = json.dumps(
                {"_nautical_cache_version": 99, "dnf": []}, separators=(",", ":")
            ).encode("utf-8")
            cache_path.write_bytes(base64.b85encode(zlib.compress(encoded)))
            real_replace = os.replace

            def fail_quarantine_replace(source, target, *args, **kwargs):
                if ".bad." in os.fspath(target):
                    raise RuntimeError("quarantine adapter invariant failed")
                return real_replace(source, target, *args, **kwargs)

            with patch.object(os, "replace", side_effect=fail_quarantine_replace):
                with self.assertRaisesRegex(RuntimeError, "quarantine adapter invariant failed"):
                    binding.cache_load(key)

    def test_atomic_replace_failure_returns_false_without_publishing(self) -> None:
        def fail_replace(_source: str, _target: str) -> None:
            raise OSError("simulated replace failure")

        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td), atomic_replace=fail_replace)
            self.assertFalse(binding.cache_save("replace-failure", {"natural": "Mondays"}))
            self.assertFalse(Path(binding._cache_path("replace-failure")).exists())

    def test_cache_save_does_not_hide_unexpected_temporary_cleanup_errors(self) -> None:
        def fail_replace(_source: str, _target: str) -> None:
            raise OSError("simulated replace failure")

        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td), atomic_replace=fail_replace)
            real_unlink = os.unlink

            def fail_temporary_unlink(path, *args, **kwargs):
                if str(path).endswith(".tmp"):
                    raise RuntimeError("temporary cleanup invariant failed")
                return real_unlink(path, *args, **kwargs)

            with patch.object(os, "unlink", side_effect=fail_temporary_unlink):
                with self.assertRaisesRegex(RuntimeError, "temporary cleanup invariant failed"):
                    binding.cache_save("cleanup-failure", {"natural": "Mondays"})

    def test_cache_save_surfaces_unexpected_chmod_error_and_cleans_resources(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cache_path = root / "anchor.jsonz"

            class BrokenFilesystem:
                environ = os.environ
                path = os.path

                def __init__(self) -> None:
                    self.file_descriptor: int | None = None

                @staticmethod
                def listdir(path: str) -> list[str]:
                    return os.listdir(path)

                def fchmod(self, file_descriptor: int, _mode: int) -> None:
                    self.file_descriptor = file_descriptor
                    raise RuntimeError("cache chmod invariant failed")

                @staticmethod
                def write(file_descriptor: int, data: bytes) -> int:
                    return os.write(file_descriptor, data)

                @staticmethod
                def close(file_descriptor: int) -> None:
                    os.close(file_descriptor)

                @staticmethod
                def unlink(path: str) -> None:
                    os.unlink(path)

            filesystem = BrokenFilesystem()
            with self.assertRaisesRegex(RuntimeError, "cache chmod invariant failed"):
                cache_payload.cache_save(
                    "chmod-failure",
                    {"natural": "Mondays"},
                    enable_anchor_cache=True,
                    json_mod=json,
                    zlib_mod=__import__("zlib"),
                    base64_mod=__import__("base64"),
                    cache_path=lambda _key: str(cache_path),
                    cache_dir=lambda: td,
                    cache_lock=lambda _key: nullcontext(True),
                    diag=lambda _message: None,
                    os_mod=filesystem,
                    tempfile_mod=tempfile,
                    cache_atomic_replace=lambda source, target: os.replace(source, target),
                    cache_state=CacheState(memory=OrderedDict(), max_entries=8, ttl=300),
                )

            self.assertIsNotNone(filesystem.file_descriptor)
            assert filesystem.file_descriptor is not None
            with self.assertRaises(OSError):
                os.fstat(filesystem.file_descriptor)
            self.assertEqual(list(root.glob(".*.tmp")), [])
            self.assertFalse(cache_path.exists())

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

            with binding._cache_lock("contend") as acquired:
                self.assertTrue(acquired)
                with binding._cache_lock("contend") as competing:
                    self.assertFalse(competing)

    def test_cache_lock_uses_owner_fallback_when_fcntl_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            namespace = vars(core).copy()
            namespace.update(
                _CACHE_DIR=td,
                _CACHE_LOAD_MEM=OrderedDict(),
                fcntl=None,
                os=os,
                time=_Clock(),
                random=__import__("random"),
            )
            binding = cache_api.for_core(namespace=namespace, module=core)

            with binding._cache_lock("fallback") as acquired:
                self.assertTrue(acquired)
                with binding._cache_lock("fallback") as competing:
                    self.assertFalse(competing)
            with binding._cache_lock("fallback") as reacquired:
                self.assertTrue(reacquired)

    def test_fallback_lock_recovers_dead_stale_pid_but_not_a_live_pid(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "fallback.lock"
            clock = _Clock()
            lock_path.write_text("999999 0\n", encoding="ascii")

            with cache_locking.safe_lock(
                lock_path,
                retries=2,
                sleep_base=0,
                stale_after=1,
                fcntl_mod=None,
                os_mod=os,
                time_mod=clock,
                random_mod=SimpleNamespace(uniform=lambda _start, _end: 0.0),
            ) as acquired:
                self.assertTrue(acquired)

            lock_path.write_text(f"{os.getpid()} 0\n", encoding="ascii")
            with cache_locking.safe_lock(
                lock_path,
                retries=1,
                sleep_base=0,
                stale_after=1,
                fcntl_mod=None,
                os_mod=os,
                time_mod=clock,
                random_mod=SimpleNamespace(uniform=lambda _start, _end: 0.0),
            ) as acquired:
                self.assertFalse(acquired)

    def test_fallback_lock_age_does_not_hide_unexpected_clock_errors(self) -> None:
        class BrokenClock:
            def time(self) -> float:
                raise RuntimeError("clock invariant failed")

            def sleep(self, _seconds: float) -> None:
                return None

        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "fallback.lock"
            lock_path.write_text("123 0\n", encoding="ascii")

            with self.assertRaisesRegex(RuntimeError, "clock invariant failed"):
                cache_locking.safe_lock_age(str(lock_path), time_mod=BrokenClock(), os_mod=os)

    def test_stale_pid_check_does_not_hide_unexpected_clock_errors(self) -> None:
        class BrokenClock:
            def time(self) -> float:
                raise RuntimeError("clock invariant failed")

            def sleep(self, _seconds: float) -> None:
                return None

        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "fallback.lock"
            lock_path.write_text("123 0\n", encoding="ascii")

            with self.assertRaisesRegex(RuntimeError, "clock invariant failed"):
                cache_locking.safe_lock_stale_pid(
                    str(lock_path), 1, time_mod=BrokenClock(), os_mod=os
                )

    def test_stale_pid_check_does_not_hide_unexpected_process_adapter_errors(self) -> None:
        class BrokenProcessAdapter:
            @staticmethod
            def kill(_pid: int, _signal: int) -> None:
                raise RuntimeError("process adapter invariant failed")

        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "fallback.lock"
            lock_path.write_text("123 0\n", encoding="ascii")

            with self.assertRaisesRegex(RuntimeError, "process adapter invariant failed"):
                cache_locking.safe_lock_stale_pid(
                    str(lock_path), None, time_mod=_Clock(), os_mod=BrokenProcessAdapter()
                )

    def test_exclusive_lock_does_not_hide_unexpected_open_errors(self) -> None:
        class BrokenFilesystem:
            O_CREAT = 1
            O_EXCL = 2
            O_WRONLY = 4

            @staticmethod
            def open(_path: str, _flags: int, _mode: int) -> int:
                raise RuntimeError("filesystem adapter invariant failed")

        with self.assertRaisesRegex(RuntimeError, "filesystem adapter invariant failed"):
            with cache_locking.safe_lock(
                "lock",
                retries=1,
                mkdir=False,
                fcntl_mod=None,
                os_mod=BrokenFilesystem(),
                time_mod=_Clock(),
                random_mod=SimpleNamespace(uniform=lambda _start, _end: 0.0),
            ):
                self.fail("an unexpected open error must not yield a lock result")

    def test_stale_lock_cleanup_does_not_hide_unexpected_unlink_errors(self) -> None:
        class BrokenFilesystem:
            O_CREAT = 1
            O_EXCL = 2
            O_WRONLY = 4

            @staticmethod
            def open(_path: str, _flags: int, _mode: int) -> int:
                raise FileExistsError("lock already exists")

            @staticmethod
            def unlink(_path: str) -> None:
                raise RuntimeError("filesystem cleanup invariant failed")

        with self.assertRaisesRegex(RuntimeError, "filesystem cleanup invariant failed"):
            with cache_locking.safe_lock_excl_context(
                "lock",
                tries=1,
                sleep_base=0,
                jitter=0,
                mode=0o600,
                mkdir=False,
                stale_after=1,
                safe_lock_ensure_parent=lambda _path, _mkdir: None,
                safe_lock_stale_pid=lambda _path, _stale_after: True,
                safe_lock_age=lambda _path: 2,
                safe_lock_sleep_once=lambda _base, _jitter: None,
                os_mod=BrokenFilesystem(),
                time_mod=_Clock(),
            ):
                self.fail("an unexpected stale-lock cleanup error must propagate")

    def test_fcntl_lock_surfaces_fchmod_errors_and_closes_open_descriptor(self) -> None:
        class BrokenFilesystem:
            O_CREAT = os.O_CREAT
            O_RDWR = os.O_RDWR

            def __init__(self) -> None:
                self.file_descriptor: int | None = None

            def open(self, path: str, flags: int, mode: int) -> int:
                self.file_descriptor = os.open(path, flags, mode)
                return self.file_descriptor

            @staticmethod
            def fchmod(_file_descriptor: int, _mode: int) -> None:
                raise RuntimeError("chmod adapter invariant failed")

            @staticmethod
            def fdopen(file_descriptor: int, mode: str, *, encoding: str):
                return os.fdopen(file_descriptor, mode, encoding=encoding)

            @staticmethod
            def close(file_descriptor: int) -> None:
                os.close(file_descriptor)

        with tempfile.TemporaryDirectory() as td:
            lock_path = Path(td) / "fcntl.lock"
            filesystem = BrokenFilesystem()

            with self.assertRaisesRegex(RuntimeError, "chmod adapter invariant failed"):
                with cache_locking.safe_lock_fcntl_context(
                    str(lock_path),
                    tries=1,
                    sleep_base=0,
                    jitter=0,
                    mode=0o600,
                    mkdir=False,
                    safe_lock_ensure_parent=lambda _path, _mkdir: None,
                    safe_lock_sleep_once=lambda _base, _jitter: None,
                    fcntl_mod=fcntl,
                    os_mod=filesystem,
                ):
                    self.fail("unexpected chmod failure must not yield a lock result")

            self.assertIsNotNone(filesystem.file_descriptor)
            assert filesystem.file_descriptor is not None
            with self.assertRaises(OSError):
                os.fstat(filesystem.file_descriptor)

    def test_exclusive_lock_cleans_partial_file_after_unexpected_fchmod_error(self) -> None:
        class BrokenFilesystem:
            O_CREAT = 1
            O_EXCL = 2
            O_WRONLY = 4

            def __init__(self) -> None:
                self.closed = False
                self.unlinked = False

            @staticmethod
            def open(_path: str, _flags: int, _mode: int) -> int:
                return 19

            @staticmethod
            def fchmod(_file_descriptor: int, _mode: int) -> None:
                raise RuntimeError("chmod adapter invariant failed")

            def close(self, _file_descriptor: int) -> None:
                self.closed = True

            def unlink(self, _path: str) -> None:
                self.unlinked = True

        filesystem = BrokenFilesystem()
        with self.assertRaisesRegex(RuntimeError, "chmod adapter invariant failed"):
            with cache_locking.safe_lock_excl_context(
                "lock",
                tries=1,
                sleep_base=0,
                jitter=0,
                mode=0o600,
                mkdir=False,
                stale_after=1,
                safe_lock_ensure_parent=lambda _path, _mkdir: None,
                safe_lock_stale_pid=lambda _path, _stale_after: False,
                safe_lock_age=lambda _path: None,
                safe_lock_sleep_once=lambda _base, _jitter: None,
                os_mod=filesystem,
                time_mod=_Clock(),
            ):
                self.fail("unexpected chmod failure must not yield a lock result")

        self.assertTrue(filesystem.closed)
        self.assertTrue(filesystem.unlinked)

    def test_cache_directory_and_lock_permissions_are_private(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary) / "cache"
            namespace = vars(core).copy()
            namespace.update(
                _CACHE_DIR=None,
                _CACHE_LOAD_MEM=OrderedDict(),
                ANCHOR_CACHE_DIR_OVERRIDE=str(cache_dir),
                _CACHE_LOCK_RETRIES=2,
                _CACHE_LOCK_SLEEP_BASE=0,
                _CACHE_LOCK_JITTER=0,
                _CACHE_LOCK_STALE_AFTER=300,
                os=os,
                time=_Clock(),
                random=__import__("random"),
                _import_sibling=_import_core_sibling,
            )
            binding = cache_api.for_core(namespace=namespace, module=core)
            with patch.dict(os.environ, {"NAUTICAL_TRUST_CACHE_PATH": "1"}):
                self.assertEqual(Path(binding._cache_dir()), cache_dir)
                cache_mode = stat.S_IMODE(cache_dir.stat().st_mode)
                self.assertEqual(cache_mode & 0o077, 0)

                lock_path = Path(binding._cache_lock_path("permissions"))
                with binding._cache_lock("permissions") as acquired:
                    self.assertTrue(acquired)
                    lock_mode = stat.S_IMODE(lock_path.stat().st_mode)
                self.assertEqual(lock_mode & 0o077, 0)

    def test_cache_directory_setup_does_not_hide_unexpected_filesystem_errors(self) -> None:
        with patch.object(
            cache_support.os,
            "makedirs",
            side_effect=RuntimeError("filesystem adapter invariant failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "filesystem adapter invariant failed"):
                cache_support.ensure_cache_dir("/cache")

    def test_cache_directory_permission_fallback_does_not_hide_unexpected_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with (
                patch.object(cache_support.os, "fchmod", side_effect=OSError("chmod unavailable")),
                patch.object(
                    cache_support.os,
                    "chmod",
                    side_effect=RuntimeError("permission adapter invariant failed"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "permission adapter invariant failed"):
                    cache_support.ensure_cache_dir(td)

    def test_cache_payload_shape_validator_does_not_hide_runtime_errors(self) -> None:
        def broken_validator(_value: object) -> bool:
            raise RuntimeError("payload validator invariant failed")

        with self.assertRaisesRegex(RuntimeError, "payload validator invariant failed"):
            cache_payload.cache_payload_shape_ok({"dnf": []}, is_dnf_like=broken_validator)

    def test_cache_directory_selection_rejects_symlink_override(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "real-cache"
            symlink = root / "cache-link"
            target.mkdir()
            symlink.symlink_to(target, target_is_directory=True)
            cache_dir = root / "xdg" / "nautical"
            namespace = vars(core).copy()
            namespace.update(
                _CACHE_DIR=None,
                _CACHE_LOAD_MEM=OrderedDict(),
                ANCHOR_CACHE_DIR_OVERRIDE=str(symlink),
                os=os,
                time=_Clock(),
                random=__import__("random"),
                _import_sibling=_import_core_sibling,
            )
            binding = cache_api.for_core(namespace=namespace, module=core)
            ensure_cache_dir = cache_support.ensure_cache_dir

            def ensure_temporary_cache_dir(path: str) -> bool:
                if not Path(path).is_relative_to(root):
                    return False
                return ensure_cache_dir(path)

            with (
                patch.dict(
                    os.environ,
                    {
                        "NAUTICAL_TRUST_CACHE_PATH": "1",
                        "XDG_CACHE_HOME": str(root / "xdg"),
                        "TASKDATA": "",
                        "NAUTICAL_ALLOW_TMP_CACHE": "",
                    },
                ),
                patch.object(
                    cache_support,
                    "ensure_cache_dir",
                    side_effect=ensure_temporary_cache_dir,
                ),
            ):
                self.assertEqual(Path(binding._cache_dir()), cache_dir)
                self.assertTrue(cache_dir.is_dir())
                self.assertFalse(cache_dir.is_symlink())

    def test_unexpected_fcntl_error_propagates_and_closes_lock_file(self) -> None:
        class BrokenFcntl:
            LOCK_EX = fcntl.LOCK_EX
            LOCK_NB = fcntl.LOCK_NB
            LOCK_UN = fcntl.LOCK_UN

            def __init__(self) -> None:
                self.file_descriptor: int | None = None

            def flock(self, file_descriptor: int, _operation: int) -> None:
                self.file_descriptor = file_descriptor
                raise OSError("simulated flock I/O failure")

        with tempfile.TemporaryDirectory() as td:
            lock_driver = BrokenFcntl()
            namespace = vars(core).copy()
            namespace.update(
                _CACHE_DIR=td,
                _CACHE_LOAD_MEM=OrderedDict(),
                fcntl=lock_driver,
                os=os,
                time=_Clock(),
                random=__import__("random"),
            )
            binding = cache_api.for_core(namespace=namespace, module=core)

            with self.assertRaisesRegex(OSError, "simulated flock I/O failure"):
                with binding._cache_lock("broken"):
                    self.fail("a failed flock must not yield an acquired lock")

            self.assertIsNotNone(lock_driver.file_descriptor)
            with self.assertRaises(OSError):
                os.fstat(lock_driver.file_descriptor)

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

    def test_semantic_fingerprint_does_not_hide_unexpected_stat_errors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td))
            real_stat = os.stat

            def fail_semantic_stat(
                path: str, *args: Any, **kwargs: Any
            ) -> os.stat_result:
                if str(path).endswith("/scheduler_api.py"):
                    raise RuntimeError("semantic fingerprint stat invariant failed")
                return real_stat(path, *args, **kwargs)

            with patch.object(os, "stat", side_effect=fail_semantic_stat):
                with self.assertRaisesRegex(RuntimeError, "semantic fingerprint stat invariant failed"):
                    binding._cache_semantic_fingerprint()

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

    def test_task_cache_key_does_not_hide_unexpected_acf_builder_errors(self) -> None:
        def broken_acf(_expression: str) -> str:
            raise RuntimeError("ACF builder invariant failed")

        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td), build_acf=broken_acf)

            with self.assertRaisesRegex(RuntimeError, "ACF builder invariant failed"):
                binding.cache_key_for_task("w:mon", "skip", "calendar")

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

            with patch.object(os, "stat", side_effect=parser_stat(1)):
                before = self._binding(Path(td))._cache_semantic_fingerprint()
            with patch.object(os, "stat", side_effect=parser_stat(2)):
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

            with patch.object(os, "stat", side_effect=stat_for(1)):
                before = binding._dnf_cache_fingerprint()
            with patch.object(os, "stat", side_effect=stat_for(2)):
                after = self._binding(Path(td))._dnf_cache_fingerprint()

            self.assertNotEqual(before, after)

    def test_dnf_cache_uses_central_api_and_fingerprints_parser(self) -> None:
        dnf = [[
            {
                "typ": "w",
                "spec": "mon",
                "ival": 1,
                "mods": {
                    "t": None,
                    "roll": None,
                    "wd": None,
                    "bd": False,
                    "day_offset": 0,
                    "business_day_offset": 0,
                },
            }
        ]]
        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td))

            self.assertTrue(binding._dnf_cache_save("w:mon", dnf))
            self.assertEqual(binding._dnf_cache_load("w:mon"), dnf)
            fingerprint = binding._dnf_cache_fingerprint()
            self.assertIn("parser=", fingerprint)
            self.assertIn("schema:", fingerprint)
            self.assertIn("release:", fingerprint)
            cache_path = Path(binding._cache_path(binding._dnf_cache_key("w:mon")))
            self.assertEqual(cache_path.suffix, ".jsonz")
            self.assertTrue(cache_path.exists())

    def test_dnf_cache_quarantines_invalid_central_payload(self) -> None:
        dnf_key_payload = {"kind": "anchor-dnf", "dnf": "invalid DNF"}
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binding = self._binding(root)
            key = binding._dnf_cache_key("w:mon")
            cache_path = Path(binding._cache_path(key))

            self.assertTrue(binding.cache_save(key, dnf_key_payload))
            self.assertIsNone(binding._dnf_cache_load("w:mon"))
            self.assertFalse(cache_path.exists())
            self.assertEqual(len(list(root.glob(cache_path.name + ".bad.*"))), 1)

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

            with patch.object(os, "stat", side_effect=stat_for(1)):
                before = binding._dnf_cache_fingerprint()
            with patch.object(os, "stat", side_effect=stat_for(2)):
                after = self._binding(Path(td))._dnf_cache_fingerprint()

            self.assertNotEqual(before, after)

    def test_dnf_fingerprint_does_not_hide_unexpected_import_errors(self) -> None:
        def broken_sibling(name: str):
            if name == "parsing.parser_atoms":
                raise RuntimeError("parser import invariant failed")
            return _import_core_sibling(name)

        with tempfile.TemporaryDirectory() as td:
            binding = self._binding(Path(td), import_sibling=broken_sibling)

            with self.assertRaisesRegex(RuntimeError, "parser import invariant failed"):
                binding._dnf_cache_fingerprint()


if __name__ == "__main__":
    unittest.main()
