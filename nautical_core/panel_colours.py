from __future__ import annotations

import hashlib
from functools import lru_cache


_ANCHOR_PALETTE = """
bright_cyan bright_green cyan cyan1 cyan2 cyan3 turquoise2 medium_turquoise
dark_turquoise cyan3 deep_sky_blue1 sky_blue1 dodger_blue1 steel_blue1
spring_green3 sea_green2 green1 green3 green4 green_yellow dark_sea_green
navy_blue dark_blue cornflower_blue royal_blue1 dodger_blue2 dodger_blue3
deep_sky_blue2 deep_sky_blue3 deep_sky_blue4 light_sky_blue1 light_sky_blue3
steel_blue steel_blue3 slate_blue1 slate_blue3 light_slate_blue aquamarine1
aquamarine3 turquoise4 pale_turquoise1 pale_turquoise4 dark_cyan light_cyan1
light_cyan3 dark_slate_gray1 dark_slate_gray2 dark_slate_gray3 light_sea_green
cadet_blue spring_green1 spring_green2 spring_green4 sea_green1 sea_green3
pale_green1 pale_green3 medium_spring_green dark_sea_green1 dark_sea_green2
dark_sea_green3 dark_sea_green4 honeydew2
""".split()

_CP_PALETTE = """
orange_red1 light_salmon1 light_pink3 light_coral dark_orange orange3 gold3
indian_red1 light_coral salmon1 deep_pink3 hot_pink medium_violet_red red red1
red3 pale_violet_red1 orchid orchid1 dark_violet bright_magenta magenta magenta1
magenta2 magenta3 medium_orchid medium_orchid1 medium_orchid3 orchid2 plum1
plum2 plum3 plum4 violet thistle1 thistle3 bright_yellow yellow yellow1 yellow2
yellow3 gold1 dark_goldenrod dark_orange3 orange1 orange4 sandy_brown tan
navajo_white1 navajo_white3 wheat1 khaki1 khaki3 cornsilk1 light_goldenrod1
light_goldenrod2 light_goldenrod3 light_yellow3 bright_red dark_red indian_red
misty_rose1 misty_rose3 pink1 pink3 light_pink1 light_pink4 rosy_brown
""".split()


@lru_cache(maxsize=512)
def chain_colour_root(kind: str, root_identity: str) -> str:
    """Return the deterministic display colour for a recurrence chain."""
    palette = _CP_PALETTE if kind == "cp" else _ANCHOR_PALETTE
    identity = str(root_identity or "").strip().lower()
    if not identity:
        return palette[0]
    digest = hashlib.sha256(
        f"nautical-chain-colour-v2|{kind}|{identity}".encode("utf-8")
    ).digest()
    return palette[int.from_bytes(digest[:8], "big") % len(palette)]
