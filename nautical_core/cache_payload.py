from __future__ import annotations


def is_atom_like(atom) -> bool:
    if not isinstance(atom, dict):
        return False
    typ = atom.get("typ") or atom.get("type")
    spec = atom.get("spec") or atom.get("value")
    if not isinstance(typ, str) or not typ:
        return False
    if spec is None:
        return False
    mods = atom.get("mods")
    if mods is not None and not isinstance(mods, dict):
        return False
    return True


def is_dnf_like(dnf, *, is_atom_like) -> bool:
    if not isinstance(dnf, list):
        return False
    for term in dnf:
        if not isinstance(term, list):
            return False
        for atom in term:
            if not is_atom_like(atom):
                return False
    return True


def clone_mod_value(value):
    if isinstance(value, tuple):
        return tuple(clone_mod_value(item) for item in value)
    if isinstance(value, list):
        return [clone_mod_value(item) for item in value]
    if isinstance(value, dict):
        return {key: clone_mod_value(item) for key, item in value.items()}
    return value


def clone_mods(mods):
    if not isinstance(mods, dict):
        return {}
    out = {}
    for key, value in mods.items():
        if isinstance(value, tuple):
            out[key] = tuple(clone_mod_value(item) for item in value)
        elif isinstance(value, list):
            out[key] = [clone_mod_value(item) for item in value]
        elif isinstance(value, dict):
            out[key] = {inner_key: clone_mod_value(inner_value) for inner_key, inner_value in value.items()}
        else:
            out[key] = value
    return out


def clone_atom(atom):
    if not isinstance(atom, dict):
        return atom
    out = {}
    for key, value in atom.items():
        if key == "mods":
            out["mods"] = clone_mods(value)
        elif isinstance(value, tuple):
            out[key] = tuple(clone_mod_value(item) for item in value)
        elif isinstance(value, list):
            out[key] = [clone_mod_value(item) for item in value]
        elif isinstance(value, dict):
            out[key] = {inner_key: clone_mod_value(inner_value) for inner_key, inner_value in value.items()}
        else:
            out[key] = value
    return out


def clone_dnf(dnf):
    if not isinstance(dnf, list):
        return dnf
    out = []
    for term in dnf:
        if isinstance(term, list):
            out.append([clone_atom(atom) for atom in term])
        else:
            out.append(term)
    return out


def clone_cache_payload(obj: dict) -> dict:
    if not isinstance(obj, dict):
        return obj
    out = {}
    for key, value in obj.items():
        if key == "dnf":
            out[key] = clone_dnf(value)
        elif isinstance(value, list):
            out[key] = list(value)
        elif isinstance(value, dict):
            inner = {}
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, (dict, list, tuple)):
                    inner[inner_key] = clone_mod_value(inner_value)
                else:
                    inner[inner_key] = inner_value
            out[key] = inner
        elif isinstance(value, (dict, list, tuple)):
            out[key] = clone_mod_value(value)
        else:
            out[key] = value
    return out


def normalize_dnf_cached(dnf):
    if not isinstance(dnf, (list, tuple)):
        return dnf
    for term in dnf:
        if not isinstance(term, (list, tuple)):
            continue
        for atom in term:
            if not isinstance(atom, dict):
                continue
            mods = atom.get("mods")
            if not isinstance(mods, dict):
                continue
            tval = mods.get("t")
            if isinstance(tval, list):
                if len(tval) == 2 and all(isinstance(x, int) for x in tval):
                    mods["t"] = (tval[0], tval[1])
                elif tval and all(
                    isinstance(x, list) and len(x) == 2 and all(isinstance(y, int) for y in x)
                    for x in tval
                ):
                    mods["t"] = [(x[0], x[1]) for x in tval]
    return dnf


def cache_payload_shape_ok(obj: dict, *, is_dnf_like) -> bool:
    try:
        if "dnf" in obj and not is_dnf_like(obj.get("dnf")):
            return False
        natural = obj.get("natural")
        if natural is not None and not isinstance(natural, str):
            return False
        next_dates = obj.get("next_dates")
        if next_dates is not None:
            if not isinstance(next_dates, list):
                return False
            for item in next_dates:
                if not isinstance(item, str):
                    return False
        meta = obj.get("meta")
        if meta is not None and not isinstance(meta, dict):
            return False
        per_year = obj.get("per_year")
        if per_year is not None and not isinstance(per_year, dict):
            return False
        limits = obj.get("limits")
        if limits is not None and not isinstance(limits, dict):
            return False
    except Exception:
        return False
    return True


def cache_atomic_replace(src: str, dst: str, *, os_mod) -> None:
    try:
        os_mod.replace(src, dst)
        return
    except OSError:
        if os_mod.name != "nt":
            raise
    try:
        import ctypes

        flags = 0x1 | 0x8
        ok = ctypes.windll.kernel32.MoveFileExW(str(src), str(dst), flags)
        if ok:
            return
        err = ctypes.GetLastError()
        raise OSError(err, "MoveFileExW failed")
    except Exception:
        raise


def cache_load(
    key: str,
    *,
    enable_anchor_cache: bool,
    cache_path,
    anchor_cache_ttl: int,
    time_mod,
    cache_load_mem,
    cache_load_mem_ttl: int,
    clone_cache_payload,
    normalize_dnf_cached,
    cache_payload_shape_ok,
    cache_load_mem_max: int,
    diag,
    os_mod,
    json_mod,
    zlib_mod,
    base64_mod,
):
    if not enable_anchor_cache:
        return None
    path = cache_path(key)
    if not path:
        return None
    try:
        st = os_mod.stat(path)
        if anchor_cache_ttl and (time_mod.time() - st.st_mtime) > anchor_cache_ttl:
            return None
        stamp = (int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))), int(st.st_size))
        now = time_mod.time()
        if cache_load_mem_ttl > 0 and cache_load_mem:
            expired = [
                memo_key
                for memo_key, (_mt, _sz, _obj, loaded_at) in cache_load_mem.items()
                if (now - loaded_at) > cache_load_mem_ttl
            ]
            for memo_key in expired:
                cache_load_mem.pop(memo_key, None)
        memo = cache_load_mem.get(key)
        if memo and memo[0] == stamp[0] and memo[1] == stamp[1]:
            if cache_load_mem_ttl <= 0 or (now - memo[3]) <= cache_load_mem_ttl:
                cache_load_mem.move_to_end(key)
                return clone_cache_payload(memo[2])
            cache_load_mem.pop(key, None)
        with open(path, "rb") as fh:
            blob = fh.read()
        data = zlib_mod.decompress(base64_mod.b85decode(blob))
        obj = json_mod.loads(data.decode("utf-8"))
        if isinstance(obj, dict) and "dnf" in obj:
            obj["dnf"] = normalize_dnf_cached(obj.get("dnf"))
        if isinstance(obj, dict):
            if not cache_payload_shape_ok(obj):
                if os_mod.environ.get("NAUTICAL_DIAG") == "1":
                    diag(f"cache_load rejected invalid payload shape for key={key}")
                return None
            cache_load_mem[key] = (stamp[0], stamp[1], obj, now)
            cache_load_mem.move_to_end(key)
            if len(cache_load_mem) > cache_load_mem_max:
                cache_load_mem.popitem(last=False)
            return clone_cache_payload(obj)
        return None
    except (OSError, ValueError, json_mod.JSONDecodeError, zlib_mod.error) as exc:
        if os_mod.environ.get("NAUTICAL_DIAG") == "1":
            diag(f"cache_load failed: {exc}")
        return None


def cache_save(
    key: str,
    obj: dict,
    *,
    enable_anchor_cache: bool,
    json_mod,
    zlib_mod,
    base64_mod,
    cache_path,
    cache_dir,
    cache_lock,
    diag,
    os_mod,
    tempfile_mod,
    cache_atomic_replace,
    cache_load_mem,
):
    if not enable_anchor_cache:
        return False
    data = json_mod.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    blob = base64_mod.b85encode(zlib_mod.compress(data, 9))
    path = cache_path(key)
    if not path:
        return False
    tmpf = None
    ok_saved = False
    try:
        base = cache_dir()
        if not base:
            return False
        with cache_lock(key) as locked:
            if not locked:
                if os_mod.environ.get("NAUTICAL_DIAG") == "1":
                    diag(f"cache_save lock busy for key={key}")
                return False
            try:
                for name in os_mod.listdir(base):
                    if name.startswith(f".{key}.") and name.endswith(".tmp"):
                        try:
                            os_mod.unlink(os_mod.path.join(base, name))
                        except Exception:
                            pass
            except Exception:
                pass
            fd, tmpf = tempfile_mod.mkstemp(dir=base, prefix=f".{key}.", suffix=".tmp")
            try:
                os_mod.fchmod(fd, 0o600)
            except Exception:
                pass
            try:
                written = 0
                while written < len(blob):
                    n = os_mod.write(fd, blob[written:])
                    if n == 0:
                        raise OSError("write returned 0")
                    written += n
            finally:
                try:
                    os_mod.close(fd)
                except Exception:
                    pass
            cache_atomic_replace(tmpf, path)
            ok_saved = True
    except (OSError, ValueError, json_mod.JSONDecodeError, zlib_mod.error) as exc:
        if os_mod.environ.get("NAUTICAL_DIAG") == "1":
            diag(f"cache_save failed: {exc}")
    finally:
        cache_load_mem.pop(key, None)
        if tmpf and os_mod.path.exists(tmpf):
            try:
                os_mod.unlink(tmpf)
            except Exception:
                pass
    return ok_saved


def cache_key_for_task_cached(anchor_expr: str, anchor_mode: str, fmt: str, *, build_acf, cache_key) -> str:
    _ = fmt
    try:
        acf = build_acf(anchor_expr)
    except Exception:
        acf = (anchor_expr or "").strip()
    return cache_key(acf, anchor_mode or "")
