import base64
from collections import OrderedDict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
import zlib

from nautical_core.cache_payload import (
    MAX_CACHE_DECODED_BYTES,
    MAX_CACHE_FILE_BYTES,
    _bounded_decompress,
    cache_load,
    cache_save,
)


class CacheResourceLimitTests(unittest.TestCase):
    def _load(self, root: Path, blob: bytes):
        path = root / "entry.cache"
        path.write_bytes(blob)
        return cache_load(
            "key", enable_anchor_cache=True, cache_path=lambda _key: str(path),
            anchor_cache_ttl=0, time_mod=time, cache_load_mem=OrderedDict(),
            cache_load_mem_ttl=0, clone_cache_payload=lambda value: value,
            normalize_dnf_cached=lambda value: value, cache_payload_shape_ok=lambda _value: True,
            cache_load_mem_max=4, diag=lambda _message: None, os_mod=os,
            json_mod=json, zlib_mod=zlib, base64_mod=base64,
        )

    def test_encoded_file_ceiling_is_rejected_before_decode(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(self._load(Path(td), b"x" * (MAX_CACHE_FILE_BYTES + 1)))

    def test_decoded_payload_ceiling_is_bounded(self):
        payload = b"a" * (MAX_CACHE_DECODED_BYTES + 1)
        encoded = base64.b85encode(zlib.compress(payload, 9))
        with self.assertRaisesRegex(ValueError, "decoded cache payload exceeds"):
            _bounded_decompress(base64.b85decode(encoded), zlib, MAX_CACHE_DECODED_BYTES)

    def test_decoded_payload_at_exact_limit_is_allowed(self):
        payload = b"a" * MAX_CACHE_DECODED_BYTES
        encoded = zlib.compress(payload, 9)
        self.assertEqual(_bounded_decompress(encoded, zlib, MAX_CACHE_DECODED_BYTES), payload)

    def test_truncated_and_corrupt_streams_are_cache_misses(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            valid = base64.b85encode(zlib.compress(b"{}", 9))
            self.assertIsNone(self._load(root, valid[:-2]))
            self.assertIsNone(self._load(root, b"not-a-cache"))

    def test_save_declines_decoded_payload_above_same_ceiling(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            @contextmanager
            def lock(_key):
                yield True

            saved = cache_save(
                "key", {"natural": "x" * MAX_CACHE_DECODED_BYTES},
                enable_anchor_cache=True, json_mod=json, zlib_mod=zlib,
                base64_mod=base64, cache_path=lambda _key: str(root / "entry.cache"),
                cache_dir=lambda: str(root), cache_lock=lock, diag=lambda _message: None,
                os_mod=os, tempfile_mod=tempfile, cache_atomic_replace=os.replace,
                cache_load_mem=OrderedDict(),
            )
            self.assertFalse(saved)
            self.assertFalse((root / "entry.cache").exists())


if __name__ == "__main__":
    unittest.main()
