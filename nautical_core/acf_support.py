from __future__ import annotations


def atom_sort_key(x: dict, *, json_mod) -> tuple:
    sj = json_mod.dumps(x.get("s"), separators=(",", ":"), sort_keys=True)
    mj = json_mod.dumps(x.get("m"), separators=(",", ":"), sort_keys=True)
    return (x.get("t", ""), int(x.get("i", 1) or 1), sj, mj)


def acf_unpack(packed: str, *, base64_mod, zlib_mod, json_mod) -> dict:
    raw = base64_mod.b85decode(packed.encode("ascii"))
    return json_mod.loads(zlib_mod.decompress(raw).decode("utf-8"))


def build_acf(
    expr: str,
    *,
    parse_anchor_expr_to_dnf_cached,
    coerce_int,
    normalize_spec_for_acf,
    mods_to_acf,
    atom_sort_key,
    json_mod,
    zlib_mod,
    base64_mod,
    hashlib_mod,
    acf_checksum_len: int,
) -> str:
    if not expr or not expr.strip():
        return ""

    try:
        dnf = parse_anchor_expr_to_dnf_cached(expr)
    except Exception:
        return "!PARSE_ERROR"

    terms = []
    for term in dnf:
        atoms = []
        for atom in term:
            typ = (atom.get("typ") or "").lower()
            ival = int(coerce_int(atom.get("ival"), 1) or 1)
            spec = atom.get("spec") or ""
            mods = atom.get("mods") or {}

            norm_spec = normalize_spec_for_acf(typ, spec)
            if norm_spec is None:
                continue

            atom_obj = {
                "t": typ,
                "s": norm_spec,
                "m": mods_to_acf(mods),
            }
            if ival != 1:
                atom_obj["i"] = ival
            atoms.append(atom_obj)

        if atoms:
            atoms.sort(key=atom_sort_key)
            terms.append(atoms)

    if not terms:
        return ""

    terms.sort(key=lambda x: json_mod.dumps(x, sort_keys=True))
    structure = {"terms": terms}
    json_str = json_mod.dumps(structure, separators=(",", ":"), sort_keys=True)
    compressed = zlib_mod.compress(json_str.encode(), level=9)
    packed = base64_mod.b85encode(compressed).decode("ascii")
    checksum = hashlib_mod.sha256(packed.encode()).hexdigest()[:acf_checksum_len]
    return f"{checksum}:{packed}"


def normalize_spec_for_acf_uncached(
    typ: str,
    spec: str,
    *,
    expand_weekly_aliases,
    split_csv_tokens,
    normalize_weekday,
    expand_monthly_aliases,
    re_mod,
    year_pair,
):
    spec = (spec or "").strip().lower()

    if typ == "w":
        spec = expand_weekly_aliases(spec)
        tokens = []
        for token in split_csv_tokens(spec):
            if not token:
                continue
            if ".." in token:
                start, end = token.split("..", 1)
                s1 = normalize_weekday(start)
                s2 = normalize_weekday(end)
                if s1 and s2:
                    tokens.append(f"{s1}..{s2}")
            else:
                normalized = normalize_weekday(token)
                if normalized:
                    tokens.append(normalized)
        if not tokens:
            return None
        ranges = sorted([token for token in tokens if ".." in token])
        singles = sorted([token for token in tokens if ".." not in token])
        return ",".join(ranges + singles)

    if typ == "m":
        spec = expand_monthly_aliases(spec)
        toks = []
        for token in split_csv_tokens(spec):
            if not token:
                continue
            if ".." in token:
                left, right = [x.strip() for x in token.split("..", 1)]
                if left and right:
                    toks.append(f"{left}..{right}")
                else:
                    toks.append(token)
            else:
                toks.append(token)
        if not toks:
            return None
        ranges = sorted([token for token in toks if ".." in token])
        singles = sorted([token for token in toks if ".." not in token])
        return ",".join(ranges + singles)

    if typ == "y":
        out = []
        for token in split_csv_tokens(spec):
            match = re_mod.fullmatch(r"(\d{2})-(\d{2})(?:\.\.(\d{2})-(\d{2}))?$", token)
            if not match:
                out.append(token)
                continue
            a, b = int(match.group(1)), int(match.group(2))
            d1, m1 = year_pair(a, b)
            if match.group(3):
                c, d = int(match.group(3)), int(match.group(4))
                d2, m2 = year_pair(c, d)
                out.append({"m": m1, "d": d1, "to": {"m": m2, "d": d2}})
            else:
                out.append({"m": m1, "d": d1})
        return out

    return None


def normalize_spec_for_acf(typ: str, spec: str, *, normalize_spec_for_acf_cached, clone_mod_value):
    res = normalize_spec_for_acf_cached((typ or "").lower(), spec or "")
    if isinstance(res, (list, dict)):
        return clone_mod_value(res)
    return res


def is_valid_acf(acf_str: str, *, hashlib_mod, acf_checksum_len: int, acf_unpack) -> bool:
    if not acf_str:
        return False
    parts = acf_str.split(":", 2)
    if len(parts) == 3 and parts[0].startswith("c"):
        _, checksum, payload = parts
    else:
        if ":" not in acf_str:
            return False
        checksum, payload = acf_str.split(":", 1)

    if len(checksum) != acf_checksum_len:
        return False
    if hashlib_mod.sha256(payload.encode()).hexdigest()[:acf_checksum_len] != checksum:
        return False
    try:
        obj = acf_unpack(payload)
        return bool(obj and "terms" in obj)
    except Exception:
        return False


def acf_to_original_format(
    acf_str: str,
    *,
    is_valid_acf,
    acf_unpack,
    acf_spec_to_string,
    acf_mods_to_string,
) -> str:
    if not is_valid_acf(acf_str):
        return ""
    parts = acf_str.split(":", 2)
    if len(parts) == 3 and parts[0].startswith("c"):
        packed = parts[2]
    else:
        packed = acf_str.split(":", 1)[1]
    obj = acf_unpack(packed)
    if not obj:
        return ""

    terms_str = []
    for term in obj.get("terms", []):
        atoms_str = []
        for atom in term:
            typ = atom["t"]
            spec = atom["s"]
            ival = atom.get("i", 1)
            mods = atom.get("m", {})
            spec_str = acf_spec_to_string(typ, spec)
            atom_str = f"{typ}"
            if ival != 1:
                atom_str += f"/{ival}"
            atom_str += f":{spec_str}"
            if mods:
                mods_str = acf_mods_to_string(mods)
                if mods_str:
                    atom_str += mods_str
            atoms_str.append(atom_str)
        terms_str.append("+".join(sorted(atoms_str)))

    return " | ".join(sorted(terms_str))


def mods_to_acf(mods: dict, *, hhmm_re) -> dict:
    out: dict[str, object] = {}
    if not mods:
        return out
    tval = mods.get("t")
    if tval:
        if isinstance(tval, tuple):
            out["t"] = f"{tval[0]:02d}:{tval[1]:02d}"
        elif isinstance(tval, str) and hhmm_re.fullmatch(tval):
            out["t"] = tval
    roll = mods.get("roll")
    if roll in ("pbd", "nbd", "nw", "next-wd", "prev-wd"):
        out["roll"] = roll
    if mods.get("bd"):
        out["bd"] = True
    if isinstance(mods.get("wd"), int):
        out["wd"] = int(mods["wd"])
    off = int(mods.get("day_offset") or 0)
    if off:
        out["+d"] = off
    return out


def acf_mods_to_string(m: dict, *, wd_abbr) -> str:
    parts = []
    if m.get("t"):
        parts.append(f"@t={m['t']}")
    roll = m.get("roll")
    if roll in ("pbd", "nbd", "nw"):
        parts.append(f"@{roll}")
    elif roll in ("next-wd", "prev-wd"):
        wd = m.get("wd")
        wd_s = wd_abbr[wd] if isinstance(wd, int) and 0 <= wd < 7 else None
        if wd_s:
            parts.append(f"@{roll.split('-')[0]}-{wd_s}")
    if m.get("bd"):
        parts.append("@bd")
    if isinstance(m.get("+d"), int) and m["+d"]:
        parts.append(f"@{m['+d']:+d}d")
    return "".join(parts)


def acf_spec_to_string(typ: str, spec, *, tok, tok_range) -> str:
    if typ == "y" and isinstance(spec, list):
        out = []
        for item in spec:
            if isinstance(item, dict) and "m" in item and "d" in item:
                d1, m1 = item["d"], item["m"]
                if "to" in item and item["to"]:
                    d2, m2 = item["to"]["d"], item["to"]["m"]
                    out.append(tok_range(d1, m1, d2, m2))
                else:
                    out.append(tok(d1, m1))
            else:
                out.append(str(item))
        return ",".join(out)
    return str(spec)
