"""Core-bound parser frontend and strict validation adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any


def for_core(module: Any, *, namespace: dict[str, Any] | None = None):
    core = namespace if namespace is not None else vars(module)
    ttl_lru_cache = core["_ttl_lru_cache"]

    def parse_hhmm(value: str):
        return core["_parser_atoms"].parse_hhmm(value, hhmm_re=core["_hhmm_re"])

    def parse_atom_head(head: str):
        return core["_parser_atoms"].parse_atom_head(
            head,
            re_mod=core["re"],
            parse_error_cls=core["ParseError"],
        )

    def parse_atom_mods(mods_str: str):
        return core["_parser_atoms"].parse_atom_mods(
            mods_str,
            split_csv_tokens=core["_split_csv_tokens"],
            parse_hhmm=parse_hhmm,
            next_prev_wd_re=core["_next_prev_wd_re"],
            weekdays=core["_WEEKDAYS"],
            day_offset_re=core["_day_offset_re"],
            parse_error_cls=core["ParseError"],
        )

    @ttl_lru_cache(maxsize=512)
    def parse_y_token_cached(tok: str, fmt: str):
        return core["_yearly_parse"].parse_y_token(
            tok,
            fmt,
            quarters=core["_QUARTERS"],
            months=core["_MONTHS"],
            y_token_re=core["_y_token_re"],
            re_mod=core["re"],
        )

    def parse_y_token(tok: str):
        return parse_y_token_cached(tok, core["_yearfmt"]())

    def fatal_bad_colon_in_year_tail(tail: str):
        return core["_parser_frontend"].fatal_bad_colon_in_year_tail(
            tail,
            split_csv_tokens=core["_split_csv_tokens"],
            re_mod=core["re"],
            yearfmt=core["_yearfmt"],
        )

    def raise_on_bad_colon_year_tokens(value: str) -> None:
        core["_parser_frontend"].raise_on_bad_colon_year_tokens(
            value,
            re_mod=core["re"],
            fatal_bad_colon_in_year_tail=fatal_bad_colon_in_year_tail,
            parse_error_cls=core["ParseError"],
        )

    def skip_ws_pos(value: str, index: int, length: int) -> int:
        return core["_parser_frontend"].skip_ws_pos(value, index, length)

    def raise_if_comma_joined_anchors(full_tail: str) -> None:
        core["_parser_frontend"].raise_if_comma_joined_anchors(
            full_tail,
            re_mod=core["re"],
            parse_error_cls=core["ParseError"],
        )

    @ttl_lru_cache(maxsize=256)
    def parse_anchor_expr_to_dnf_cached_obj(s: str, fmt: str):
        return core["parse_anchor_expr_to_dnf"](s)

    def parse_anchor_expr_to_dnf_cached_impl(s: str):
        if not s:
            return []
        key = core["_unwrap_quotes"](s or "").strip()
        if not key:
            return []
        result = core["_clone_dnf"](
            parse_anchor_expr_to_dnf_cached_obj(key, core["_yearfmt"]())
        )
        core["_emit_cache_metrics"]()
        if core["os"].environ.get("NAUTICAL_CLEAR_CACHES") == "1":
            core["_clear_all_caches"]()
        return result

    def validate_weekly_spec(spec: str):
        value = core["_expand_weekly_aliases"](spec)
        tokens = core["_split_csv_lower"](value)
        if not tokens:
            raise core["ParseError"](
                f"Weekly spec is empty. Examples: '{core['_CANON_WEEKLY_RANGE_EX']}', "
                f"'{core['_CANON_WEEKLY_LIST_EX']}'."
            )
        random_tokens = [
            token for token in tokens
            if core["re"].fullmatch(r"(?:rand|[1-9]\d{0,2}rand)", token)
        ]
        if random_tokens:
            if len(tokens) > 1:
                raise core["ParseError"]("A weekly random selector cannot be combined with explicit weekdays in the same list. Use '+' to constrain its candidate pool or '|' for a separate branch.")
            count = core["_cached_expansion"].random_count_from_spec(random_tokens[0]) or 1
            if count > 7:
                raise core["ParseError"]("Weekly random count cannot exceed 7 days.")
            return
        for token in tokens:
            if core["re"].fullmatch(r"w-?\d+", token):
                canonical_week = int(token[1:])
                raise core["ParseError"](
                    "ISO week numbers belong to yearly anchors. "
                    f"Use 'y:w{canonical_week}' instead of 'w:{token}'."
                )
            if "-" in token or ":" in token:
                raise core["ParseError"](
                    f"Invalid weekly range '{token}'. Use '..' (e.g., '{core['_CANON_WEEKLY_RANGE_EX'] if '_CANON_WEEKLY_RANGE_EX' in core else 'w:mon..fri'}')."
                )
            if ".." in token:
                left, right = token.split("..", 1)
                if left not in core["_WEEKDAYS"] or right not in core["_WEEKDAYS"]:
                    raise core["ParseError"](f"Unknown weekday in range '{token}'.")
            elif token not in core["_WEEKDAYS"]:
                raise core["ParseError"](f"Unknown weekday token '{token}'.")

    def validate_monthly_spec(spec: str):
        value = core["_expand_monthly_aliases"](spec)
        tokens = core["_split_csv_lower"](value)
        if not tokens:
            raise core["ParseError"]("Empty monthly spec")
        for token in tokens:
            random_count = core["_cached_expansion"].random_count_from_spec(token)
            if random_count is not None:
                if random_count > 31:
                    raise core["ParseError"]("Monthly random count cannot exceed 31 days.")
                continue
            if core["_int_like_re"].fullmatch(token):
                number = int(token)
                if number == 0:
                    raise core["ParseError"](
                        "Day-of-month 0 is not allowed. Use 1..31 or negative -1..-31 (e.g., -1 for last day)."
                    )
                if abs(number) > 31:
                    raise core["ParseError"](
                        f"Day-of-month '{token}' out of range. Use 1..31 or -1..-31."
                    )
                continue
            if ".." in token:
                left, right = token.split("..", 1)
                if not (core["_int_like_re"].fullmatch(left) and core["_int_like_re"].fullmatch(right)):
                    raise core["ParseError"](f"Invalid monthly range '{token}'.")
                if any(int(part) == 0 or abs(int(part)) > 31 for part in (left, right)):
                    raise core["ParseError"](f"Monthly range '{token}' out of bounds.")
                continue
            if ":" in token:
                raise core["ParseError"](f"Invalid monthly range '{token}'. Use '..'.")
            match = core["_nth_weekday_re"].match(token)
            if match:
                raw = match.group(1)
                if raw == "last":
                    continue
                number_text = core["re"].sub(r"(st|nd|rd|th)$", "", raw)
                try:
                    number = int(number_text)
                except Exception:
                    raise core["ParseError"](
                        f"Invalid nth-weekday number '{raw}' in token '{token}'. "
                        "Use 1..5 or 'last' (e.g., '2nd-mon', 'last-fri')."
                    )
                if number == 0 or abs(number) > 5:
                    suggestion = f"last-{match.group(2)}" if number > 5 else None
                    message = "nth-weekday must be between 1 and 5 (or 'last')."
                    if suggestion:
                        message += f" Did you mean '{suggestion}'?"
                    raise core["ParseError"](f"{message} Offending token: '{token}'.")
                continue
            match = core["_bd_re"].match(token)
            if match:
                number = int(match.group(1))
                if number == 0 or abs(number) > 31:
                    raise core["ParseError"](f"Business-day index '{number}' out of range.")
                continue
            if core["_month_from_alias"](token) is not None:
                raise core["ParseError"](
                    f"Unknown monthly token '{token}'. Month names belong to yearly anchors. "
                    f"Use 'y:{token}' for specific months, or monthly selectors like "
                    "'m:15', 'm:last-fri', or 'm:5bd'."
                )
            raise core["ParseError"](
                f"Unknown monthly token '{token}'. Examples: "
                "'15', '-1', '1..7', '-3..-1', '2nd-mon', 'last-fri', '5bd'."
            )

    def split_inline_items_respecting_t_lists(s: str) -> list[str]:
        """Split comma-list items while preserving commas in ``@t=`` lists."""
        if not s:
            return []
        out: list[str] = []
        buf: list[str] = []
        in_t_value = False
        i, n = 0, len(s)

        def flush() -> None:
            token = "".join(buf).strip()
            if token:
                out.append(token)
            buf.clear()

        while i < n:
            ch = s[i]
            if ch == "@":
                in_t_value = s[i:i + 3].lower() == "@t="
                buf.append(ch)
                i += 1
                continue
            if ch == ",":
                if in_t_value:
                    j = i + 1
                    while j < n and s[j].isspace():
                        j += 1
                    if j < n and s[j] == ",":
                        buf.append(ch)
                        i += 1
                        continue
                    k = j
                    while k < n and s[k] != ",":
                        k += 1
                    nxt = s[j:k].strip()
                    if nxt and "@" not in nxt and (
                        core["_astronomy"].is_event_name(nxt)
                        or ((not nxt[0].isalpha()) and nxt[0] not in "-(|&")
                    ):
                        buf.append(ch)
                        i += 1
                        continue
                    flush()
                    in_t_value = False
                    i += 1
                    continue
                flush()
                i += 1
                continue
            buf.append(ch)
            i += 1
        flush()
        return out

    def parse_group_with_inline_mods(typ: str, ival: int, spec: str, outer_mods_str: str):
        tokens = [
            token.strip()
            for token in split_inline_items_respecting_t_lists(str(spec or ""))
            if token.strip()
        ]
        if len(tokens) < 2 or not any("@" in token for token in tokens):
            return None
        if outer_mods_str.strip():
            raise core["ParseError"](
                "Cannot mix group-level modifiers (after ':') with per-item modifiers in the same list. "
                "Choose one style: either 'w:mon@t=09:00,fri@t=15:00' (per-item) "
                "or 'w:mon,fri@t=09:00,15:00' (group)."
            )
        at_idxs = [i for i, token in enumerate(tokens) if "@" in token]
        if len(at_idxs) == 1 and at_idxs[0] == len(tokens) - 1:
            return None
        or_terms = []
        for token in tokens:
            if "@" in token:
                item_spec, item_mods_str = token.split("@", 1)
            else:
                item_spec, item_mods_str = token, ""
            item_spec = item_spec.strip().lower()
            item_mods = core["_parse_atom_mods"](item_mods_str.strip())
            if item_mods_str.strip() and (
                item_mods.get("roll")
                or item_mods.get("wd") is not None
                or item_mods.get("bd")
                or (item_mods.get("day_offset") or 0) != 0
                or (item_mods.get("business_day_offset") or 0) != 0
            ):
                raise core["ParseError"](
                    "Inline per-item modifiers in comma-lists only support '@t=HH:MM[,HH:MM...]'. "
                    "For other modifiers (e.g. '@bd'), use group style like 'w:mon,tue@bd@t=09:00,12:00' "
                    "or explicit OR terms with '|', e.g. '(w:mon@t=09:00) | (w:tue@bd@t=12:00)'."
                )
            or_terms.append([{"typ": typ, "spec": item_spec, "ival": ival, "mods": item_mods}])
        return or_terms

    def rewrite_weekly_multi_time_atoms(s: str) -> str:
        return core["_parser_frontend"].rewrite_weekly_multi_time_atoms(
            s,
            split_csv_tokens=core["_split_csv_tokens"],
            re_mod=core["re"],
        )

    return SimpleNamespace(
        _parse_hhmm=parse_hhmm,
        _parse_atom_head=parse_atom_head,
        _parse_atom_mods=parse_atom_mods,
        _parse_y_token_cached=parse_y_token_cached,
        _parse_y_token=parse_y_token,
        _fatal_bad_colon_in_year_tail=fatal_bad_colon_in_year_tail,
        _raise_on_bad_colon_year_tokens=raise_on_bad_colon_year_tokens,
        _skip_ws_pos=skip_ws_pos,
        _raise_if_comma_joined_anchors=raise_if_comma_joined_anchors,
        _parse_anchor_expr_to_dnf_cached_obj=parse_anchor_expr_to_dnf_cached_obj,
        _parse_anchor_expr_to_dnf_cached_impl=parse_anchor_expr_to_dnf_cached_impl,
        _validate_weekly_spec=validate_weekly_spec,
        _validate_monthly_spec=validate_monthly_spec,
        _split_inline_items_respecting_t_lists=split_inline_items_respecting_t_lists,
        _parse_group_with_inline_mods=parse_group_with_inline_mods,
        _rewrite_weekly_multi_time_atoms=rewrite_weekly_multi_time_atoms,
    )


__all__ = ("for_core",)
