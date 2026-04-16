from __future__ import annotations
import re


_HOUR_PAD_RE = re.compile(r"^(\d):(\d{2})(?::\d{2})?$")


def parse_hhmm(s: str, *, hhmm_re):
    match = hhmm_re.match(s)
    if not match:
        return None
    return (int(match.group(1)), int(match.group(2)))


def _time_padding_hint(tok: str) -> str | None:
    text = (tok or "").strip()
    match = _HOUR_PAD_RE.match(text)
    if not match:
        return None
    return f"Time '{text}' needs a leading zero. Use '0{match.group(1)}:{match.group(2)}'."


def parse_atom_head(head: str, *, re_mod, parse_error_cls) -> tuple[str, int]:
    h = (head or "").strip().lower()
    match = re_mod.fullmatch(r"(w|m|y)(?:/(\d{1,3}))?$", h)
    if not match:
        raise parse_error_cls(
            f"Invalid anchor head '{head}'. Expected 'w', 'm', or 'y' with optional '/N', "
            "e.g., 'w/2', 'm/3', 'y/4'."
        )
    typ = match.group(1)
    ival = int(match.group(2) or 1)
    if ival < 1:
        ival = 1
    if ival > 100:
        ival = 100
    return typ, ival


def parse_atom_mods(
    mods_str: str,
    *,
    split_csv_tokens,
    parse_hhmm,
    next_prev_wd_re,
    weekdays,
    day_offset_re,
    parse_error_cls,
):
    mods = {"t": None, "roll": None, "wd": None, "bd": False, "day_offset": 0}
    if not mods_str:
        return mods

    def parse_time_list(v: str):
        parts = split_csv_tokens(v)
        if not parts:
            return None
        out = []
        seen = set()
        for p in parts:
            hhmm = parse_hhmm(p)
            if not hhmm:
                hint = _time_padding_hint(p)
                if hint:
                    raise parse_error_cls(hint)
                raise parse_error_cls(f"Invalid time in @t=HH:MM[,HH:MM...]: '{p}'")
            if hhmm not in seen:
                out.append(hhmm)
                seen.add(hhmm)
        return out

    for raw in mods_str.split("@"):
        tok = raw.strip().lower()
        if not tok:
            continue
        if tok in ("nw", "pbd", "nbd"):
            mods["roll"] = tok
            continue
        if tok == "bd":
            mods["bd"] = True
            continue
        match = next_prev_wd_re.match(tok)
        if match:
            mods["roll"] = f"{match.group(1)}-wd"
            mods["wd"] = weekdays[match.group(2)]
            continue
        if tok.startswith("t="):
            if mods["t"] is not None:
                raise parse_error_cls(
                    "Duplicate '@t=' modifier. Use a single '@t=HH:MM,HH:MM,...' list."
                )
            tval = tok.split("=", 1)[1].strip()
            tlist = parse_time_list(tval)
            if not tlist:
                hint = _time_padding_hint(tval)
                if hint:
                    raise parse_error_cls(hint)
                raise parse_error_cls(f"Invalid time in @t=HH:MM[,HH:MM...]: '{tok}'")
            mods["t"] = tlist[0] if len(tlist) == 1 else tlist
            continue
        match = day_offset_re.match(tok)
        if match:
            mods["day_offset"] += int(match.group(1))
            continue
        raise parse_error_cls(f"Unknown modifier '@{tok}'")
    return mods


def normalize_monthly_ordinal_spec(spec: str, *, re_mod) -> str:
    def ord_norm(match):
        return f"{match.group(1)}{match.group(3).lower()}"

    return re_mod.sub(
        r"\b([1-5])\s*(st|nd|rd|th)\s*-\s*(mon|tue|wed|thu|fri|sat|sun)\b",
        ord_norm,
        spec,
        flags=re_mod.IGNORECASE,
    )


def build_anchor_atom_dnf(
    head: str,
    full_tail: str,
    *,
    parse_atom_head,
    parse_group_with_inline_mods,
    normalize_monthly_ordinal_spec,
    split_csv_lower,
    parse_atom_mods,
):
    typ, ival = parse_atom_head(head)
    tlo = (typ or "").lower()

    dnf_or = parse_group_with_inline_mods(tlo, ival, full_tail, "")
    if dnf_or is not None:
        return dnf_or

    spec, mods_str = (full_tail.split("@", 1) + [""])[:2]
    if tlo == "m":
        spec = normalize_monthly_ordinal_spec(spec)

    if tlo == "w":
        toks = split_csv_lower(spec)
        if "rand" in toks and len(toks) > 1:
            mods = parse_atom_mods(mods_str)
            return [[{"typ": "w", "spec": t, "ival": ival, "mods": mods}] for t in toks]

    mods = parse_atom_mods(mods_str)
    return [[{"typ": tlo, "spec": spec.strip().lower(), "ival": ival, "mods": mods}]]


def parse_anchor_atom_at(
    s: str,
    i: int,
    n: int,
    *,
    skip_ws_pos,
    raise_if_comma_joined_anchors,
    build_anchor_atom_dnf,
    parse_error_cls,
):
    i = skip_ws_pos(s, i, n)
    start = i
    while i < n and s[i] not in ":()+|":
        i += 1
    head = s[start:i].strip()

    if i >= n or s[i] != ":":
        raise parse_error_cls("Expected ':' after anchor type. Example 'w:mon', 'm:-1', 'y:06-01'")
    i += 1

    start = i
    while i < n:
        ch = s[i]
        if ch in ")|":
            break
        if ch == "+" and not (i > start and s[i - 1] == "@"):
            break
        i += 1

    full_tail = s[start:i].strip()
    raise_if_comma_joined_anchors(full_tail)
    return build_anchor_atom_dnf(head, full_tail), i
