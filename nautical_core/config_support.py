from __future__ import annotations

import copy
import os
import re
import sys


_UDA_ATTR_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def env_flag_true(name: str, env_map: dict | None = None) -> bool:
    src = env_map if env_map is not None else os.environ
    try:
        raw = src.get(name, "") if hasattr(src, "get") else ""
    except Exception:
        raw = ""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def path_input_error(path_value: str) -> str | None:
    raw = str(path_value or "").strip()
    if not raw:
        return "empty path"
    if "\x00" in raw:
        return "NUL byte in path"
    parts = raw.replace("\\", "/").split("/")
    if any(part == ".." for part in parts):
        return "parent traversal ('..') is not allowed"
    return None


def normalized_abspath(path_value: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path_value or "").strip()))


def nearest_existing_dir(path_value: str) -> str | None:
    cur = normalized_abspath(path_value)
    while cur:
        if os.path.isdir(cur):
            return cur
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            return None
        cur = parent
    return None


def world_writable_without_sticky(mode: int) -> bool:
    return bool(mode & 0o002) and not bool(mode & 0o1000)


def path_safety_error(path_value: str, *, expect_dir: bool = True) -> str | None:
    ap = normalized_abspath(path_value)
    if not ap:
        return "empty path"
    try:
        target_exists = os.path.exists(ap)
        if target_exists:
            if expect_dir and not os.path.isdir(ap):
                return "not a directory"
            probe = ap
        else:
            probe = nearest_existing_dir(ap)
            if not probe:
                return "no existing parent directory"

        st = os.stat(probe)
        if target_exists:
            uid_fn = getattr(os, "getuid", None)
            if callable(uid_fn):
                uid = uid_fn()
                if st.st_uid != uid:
                    return "owner mismatch"
        if world_writable_without_sticky(st.st_mode):
            return "world-writable path without sticky bit"
        if expect_dir:
            if not os.access(probe, os.W_OK | os.X_OK):
                return "path is not writable/searchable"
        else:
            if target_exists and not os.path.isdir(ap):
                if not os.access(ap, os.R_OK):
                    return "path is not readable"
            elif not os.access(probe, os.W_OK | os.X_OK):
                return "parent path is not writable/searchable"
    except Exception as exc:
        return str(exc)
    return None


def validated_user_dir(
    path_value: str,
    *,
    label: str,
    trust_env: str = "",
    env_map: dict | None = None,
    warn_on_error: bool = True,
) -> str:
    raw = str(path_value or "").strip()
    in_err = path_input_error(raw)
    if in_err:
        if warn_on_error and env_flag_true("NAUTICAL_DIAG", env_map):
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe {label} '{raw}': {in_err}\n")
            except Exception:
                pass
        return ""
    ap = normalized_abspath(raw)
    if trust_env and env_flag_true(trust_env, env_map):
        return ap
    err = path_safety_error(ap, expect_dir=True)
    if err:
        if warn_on_error and env_flag_true("NAUTICAL_DIAG", env_map):
            try:
                sys.stderr.write(f"[nautical] Ignoring unsafe {label} '{path_value}': {err}\n")
            except Exception:
                pass
        return ""
    return ap


def read_toml(
    path: str,
    *,
    tomllib_mod,
    warn_missing_toml_parser,
    warn_toml_parse_error,
):
    try:
        if not path or not os.path.exists(path):
            return {}
    except Exception:
        return {}

    env_path = os.environ.get("NAUTICAL_CONFIG") or ""
    env_abs = os.path.abspath(os.path.expanduser(env_path)) if env_path else ""
    is_env_path = bool(env_abs and path == env_abs)
    trust_config_path = env_flag_true("NAUTICAL_TRUST_CONFIG_PATH")

    if not trust_config_path:
        in_err = path_input_error(path)
        if in_err:
            if os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    sys.stderr.write(f"[nautical] Ignoring unsafe config path '{path}': {in_err}\n")
                except Exception:
                    pass
            return {}
        safety_err = path_safety_error(path, expect_dir=False)
        if safety_err:
            if os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    sys.stderr.write(f"[nautical] Ignoring unsafe config path '{path}': {safety_err}\n")
                except Exception:
                    pass
            return {}

    if tomllib_mod is None:
        if is_env_path:
            raise RuntimeError(
                f"NAUTICAL_CONFIG is set but TOML parser is unavailable for {path}. "
                "Install tomli or upgrade to Python 3.11+."
            )
        warn_missing_toml_parser(path)
        return {}

    try:
        with open(path, "rb") as fh:
            return tomllib_mod.load(fh) or {}
    except Exception as exc:
        if is_env_path:
            raise RuntimeError(f"NAUTICAL_CONFIG parse failed for {path}: {exc}")
        if os.environ.get("NAUTICAL_DIAG") == "1":
            print(f"[nautical] Failed to parse TOML: {path}: {exc}", file=sys.stderr)
        else:
            warn_toml_parse_error(path, exc)
        return {}


def warn_env_config_missing(env_path: str, *, warn_once_per_day_any) -> None:
    warn_once_per_day_any(
        "config_missing",
        "[nautical] NAUTICAL_CONFIG path missing; using defaults.",
    )
    if os.environ.get("NAUTICAL_DIAG") != "1":
        return
    ap = os.path.abspath(os.path.expanduser(env_path))
    print(
        "[nautical] NAUTICAL_CONFIG is set but the file is missing or invalid; defaults will be used.\n"
        f"          Resolved path: {ap}\n"
        "          Fix: create the file at that path or update NAUTICAL_CONFIG.\n",
        file=sys.stderr,
    )


def normalize_keys(data: dict) -> dict:
    out = {}
    for key, value in (data or {}).items():
        out[str(key).strip().lower()] = value
    return out


def config_paths(*, warn_env_config_missing) -> list[str]:
    env_path = os.environ.get("NAUTICAL_CONFIG")
    if env_path:
        raw_env = str(env_path).strip()
        in_err = path_input_error(raw_env)
        if in_err and not env_flag_true("NAUTICAL_TRUST_CONFIG_PATH"):
            if os.environ.get("NAUTICAL_DIAG") == "1":
                try:
                    sys.stderr.write(f"[nautical] Ignoring unsafe NAUTICAL_CONFIG '{raw_env}': {in_err}\n")
                except Exception:
                    pass
            return []
        ap = os.path.abspath(os.path.expanduser(raw_env))
        if (not os.path.exists(ap)) or os.path.isdir(ap):
            warn_env_config_missing(env_path)
        return [ap]

    def _dedup(paths: list[str]) -> list[str]:
        seen = set()
        out = []
        for path in paths:
            if not path:
                continue
            ap = os.path.abspath(os.path.expanduser(path))
            if ap in seen:
                continue
            seen.add(ap)
            out.append(ap)
        return out

    def _candidates_in_dir(dirname: str) -> list[str]:
        dirname = os.path.abspath(os.path.expanduser(dirname))
        return [
            os.path.join(dirname, "config-nautical.toml"),
            os.path.join(dirname, "nautical.toml"),
        ]

    paths: list[str] = []
    trc = os.environ.get("TASKRC")
    if trc:
        trc_abs = os.path.abspath(os.path.expanduser(trc))
        trc_dir = os.path.dirname(trc_abs)
        paths.extend(_candidates_in_dir(trc_dir))
        paths.extend(_candidates_in_dir(os.path.join(trc_dir, ".task")))
        if os.path.basename(trc_dir) == ".task":
            paths.extend(_candidates_in_dir(trc_dir))

    moddir = os.path.dirname(os.path.abspath(__file__))
    paths.extend(_candidates_in_dir(moddir))

    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        paths.extend(_candidates_in_dir(os.path.join(xdg, "nautical")))
    paths.extend(_candidates_in_dir(os.path.expanduser("~/.config/nautical")))
    paths.extend(_candidates_in_dir(os.path.expanduser("~/.task")))

    out = _dedup(paths)

    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            print("[nautical] Config search order:", file=sys.stderr)
            for path in out:
                print(f"  - {path}", file=sys.stderr)
        except Exception:
            pass

    return out


def load_config(
    *,
    defaults: dict,
    config_paths,
    read_toml,
    normalize_keys,
):
    cfg = dict(defaults)
    chosen = None

    paths = config_paths()
    for path in paths:
        data = read_toml(path)
        if data:
            cfg.update(normalize_keys(data))
            chosen = path
            break

    if os.environ.get("NAUTICAL_DIAG") == "1":
        try:
            if chosen:
                print(f"[nautical] Using config: {chosen}", file=sys.stderr)
            else:
                print("[nautical] No config file found; using defaults.", file=sys.stderr)
                print("[nautical] Search order:", file=sys.stderr)
                for path in paths:
                    print(f"  - {path}", file=sys.stderr)
        except Exception:
            pass

    cfg["wrand_salt"] = str(cfg.get("wrand_salt") or defaults["wrand_salt"])
    cfg["tz"] = str(cfg.get("tz") or defaults["tz"])
    cfg["holiday_region"] = str(cfg.get("holiday_region") or "")
    cfg["omit_file_dir"] = str(cfg.get("omit_file_dir") or "")
    if cfg.get("recurrence_update_udas") is None:
        rec = cfg.get("recurrence")
        if isinstance(rec, dict):
            rec_norm = normalize_keys(rec)
            if rec_norm.get("update_udas") is not None:
                cfg["recurrence_update_udas"] = rec_norm.get("update_udas")
    if cfg.get("recurrence_update_udas") is None and cfg.get("recurrence.update_udas") is not None:
        cfg["recurrence_update_udas"] = cfg.get("recurrence.update_udas")
    return cfg


def get_config(conf_cache, *, load_config):
    if conf_cache is None:
        conf_cache = copy.deepcopy(load_config())
    return copy.deepcopy(conf_cache), conf_cache


def conf_raw(conf: dict, key: str):
    return conf.get(key)


def conf_str(conf: dict, key: str, default: str) -> str:
    value = conf_raw(conf, key)
    if value is None:
        return str(default)
    text = str(value).strip()
    return text if text else str(default)


def conf_int(
    conf: dict,
    key: str,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    value = conf_raw(conf, key)
    try:
        out = int(str(value).strip())
    except Exception:
        out = int(default)
    if min_value is not None and out < min_value:
        out = int(min_value)
    if max_value is not None and out > max_value:
        out = int(max_value)
    return out


def conf_bool(
    conf: dict,
    key: str,
    default: bool = False,
    true_values: set[str] | None = None,
    false_values: set[str] | None = None,
) -> bool:
    value = conf_raw(conf, key)
    if value is None:
        return bool(default)
    text = str(value).strip().lower()
    if not text:
        return bool(default)
    if true_values and text in true_values:
        return True
    if false_values and text in false_values:
        return False
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off", "none"):
        return False
    return bool(default)


def conf_csv_or_list(conf: dict, key: str, default: list[str] | None = None, lower: bool = False) -> list[str]:
    value = conf_raw(conf, key)
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        raw_items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    out: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        text = str(item).strip()
        if not text:
            continue
        if lower:
            text = text.lower()
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out if out else list(default or [])


def conf_uda_field_list(conf: dict, key: str) -> list[str]:
    fields = conf_csv_or_list(conf, key, default=[], lower=True)
    out: list[str] = []
    for field in fields:
        if _UDA_ATTR_NAME_RE.fullmatch(field):
            out.append(field)
            continue
        if os.environ.get("NAUTICAL_DIAG") == "1":
            try:
                print(f"[nautical] Ignoring invalid UDA field in {key}: {field!r}", file=sys.stderr)
            except Exception:
                pass
    return out


def trueish(v, default=False):
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")
