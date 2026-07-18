"""PLO preflop position logic (in-position / out-of-position).

Single source of truth for hero's IP/OOP standing -- the CSV ``Relative
Position`` column (Layer 8) and the Layer 6 SOLVER DATA block read the same
value so they can't drift. Ports :mod:`pipeline.preflop.position` to the PLO
6-max seat set (``LJ, HJ, CO, BU, SB, BB``).
"""

from __future__ import annotations

from pipeline.plo.fact_extractor import PloFacts

# Postflop action order: SB acts first ... BU (button) acts last. A higher rank
# = acts later postflop = in position. NOTE there is deliberately NO
# blind-vs-blind exception (July 2026 bugfix): at a ring table the BvB SB
# still acts FIRST on every postflop street -- the BB has position. "The SB
# is the dealer" is true only at a literal 2-player table (heads-up format),
# which no pack is. Mirrors pipeline/preflop/position.py; tests pin the
# ring-table rule.
# One combined dict covers BOTH packs: only the RELATIVE order of two seats
# is ever compared, the 6-max subset (LJ < HJ < CO < BU) keeps its order
# inside the 9-max ladder, and BU/BTN (the same seat in the two packs'
# dialects) never co-occur at one table.
_POSTFLOP_RANK: dict[str, int] = {
    "SB": 0,
    "BB": 1,
    "UTG": 2,
    "UTG+1": 3,
    "UTG+2": 4,
    "LJ": 5,
    "HJ": 6,
    "CO": 7,
    "BU": 8,
    "BTN": 8,
}

# The dealer seat in either dialect (6-max "BU", 9-max "BTN").
_BUTTON_SEATS = frozenset({"BU", "BTN"})


def ip_oop_positions(hero_pos: str, villain_pos: str) -> tuple[str, str]:
    """Return ``(ip_position, oop_position)`` for a 2-way preflop spot.

    The seat with the higher postflop rank (acts later) is in position.
    Blind-vs-blind follows the same rule: the BB acts after the SB on every
    postflop street, so the BB is IP (no special case -- see module comment).
    """
    hero_rank = _POSTFLOP_RANK.get(hero_pos, -1)
    villain_rank = _POSTFLOP_RANK.get(villain_pos, -1)
    if hero_rank > villain_rank:
        return (hero_pos, villain_pos)
    return (villain_pos, hero_pos)


def hero_relative_position(facts: PloFacts) -> str:
    """Hero's IP/OOP standing as ``"In Position"`` / ``"Out of Position"``.

    With a villain: hero is IP iff it acts last postflop. On open spots (no
    villain) the opener is in position only when nobody behind acts later
    postflop -- true only for the Button (an SB first-in open still has the
    BB in position behind it).
    """
    hero = facts.spot.node.actor
    if facts.villain_stats is None:
        return "In Position" if hero in _BUTTON_SEATS else "Out of Position"
    ip_pos, _oop = ip_oop_positions(hero, facts.villain_stats.seat)
    return "In Position" if hero == ip_pos else "Out of Position"


# Preflop seat buckets per table size (pack-internal seat codes). 6-max: the
# pack's LJ is its FIRST seat (displayed UTG) -> early. 9-max: the three
# UTG-family seats are early, the real LJ/HJ are middle, CO/BTN late.
_BUCKETS_BY_TABLE: dict[int, dict[str, frozenset[str]]] = {
    6: {
        "early": frozenset({"LJ"}),
        "middle": frozenset({"HJ"}),
        "late": frozenset({"CO", "BU"}),
    },
    9: {
        "early": frozenset({"UTG", "UTG+1", "UTG+2"}),
        "middle": frozenset({"LJ", "HJ"}),
        "late": frozenset({"CO", "BTN"}),
    },
}


def position_bucket(seat: str, *, table_size: int = 6) -> str:
    """``'early'`` / ``'middle'`` / ``'late'`` / ``'sb'`` / ``'bb'`` for a
    pack seat code. Shared by the concept tagger and the skill tagger so the
    two can't drift; table-size aware because the 6-max pack's LJ is its
    UTG-equivalent while the 9-max LJ is a genuine middle seat."""
    if seat == "SB":
        return "sb"
    if seat == "BB":
        return "bb"
    buckets = _BUCKETS_BY_TABLE[table_size]
    for name in ("early", "middle", "late"):
        if seat in buckets[name]:
            return name
    msg = f"unknown seat {seat!r} for a {table_size}-max table"
    raise ValueError(msg)


__all__ = ["hero_relative_position", "ip_oop_positions", "position_bucket"]
