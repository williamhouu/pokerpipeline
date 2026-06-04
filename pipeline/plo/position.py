"""PLO preflop position logic (in-position / out-of-position).

Single source of truth for hero's IP/OOP standing -- the CSV ``Relative
Position`` column (Layer 8) and the Layer 6 SOLVER DATA block read the same
value so they can't drift. Ports :mod:`pipeline.preflop.position` to the PLO
6-max seat set (``LJ, HJ, CO, BU, SB, BB``).
"""

from __future__ import annotations

from pipeline.plo.fact_extractor import PloFacts

# Postflop action order: SB acts first ... BU (button) acts last. A higher rank
# = acts later postflop = in position. Blind-vs-blind is the exception: with
# only SB + BB live, the SB is the dealer and acts LAST postflop, so SB is IP.
_POSTFLOP_RANK: dict[str, int] = {
    "SB": 0,
    "BB": 1,
    "LJ": 2,
    "HJ": 3,
    "CO": 4,
    "BU": 5,
}


def ip_oop_positions(hero_pos: str, villain_pos: str) -> tuple[str, str]:
    """Return ``(ip_position, oop_position)`` for a 2-way preflop spot.

    BvB (only SB + BB): SB acts last postflop, so SB is IP. Otherwise the seat
    with the higher postflop rank (acts later) is in position.
    """
    if {hero_pos, villain_pos} == {"SB", "BB"}:
        return ("SB", "BB")
    hero_rank = _POSTFLOP_RANK.get(hero_pos, -1)
    villain_rank = _POSTFLOP_RANK.get(villain_pos, -1)
    if hero_rank > villain_rank:
        return (hero_pos, villain_pos)
    return (villain_pos, hero_pos)


def hero_relative_position(facts: PloFacts) -> str:
    """Hero's IP/OOP standing as ``"In Position"`` / ``"Out of Position"``.

    With a villain: hero is IP iff it acts last postflop (the BvB-SB exception
    included). On open spots (no villain) the opener is in position only when
    nobody behind acts later -- true for the Button, and for the SB in a BvB
    open (the SB is the dealer).
    """
    hero = facts.spot.node.actor
    if facts.villain_stats is None:
        return "In Position" if hero in ("BU", "SB") else "Out of Position"
    ip_pos, _oop = ip_oop_positions(hero, facts.villain_stats.seat)
    return "In Position" if hero == ip_pos else "Out of Position"


__all__ = ["hero_relative_position", "ip_oop_positions"]
