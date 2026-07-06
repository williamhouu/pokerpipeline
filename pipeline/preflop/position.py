"""Shared preflop position logic (in-position / out-of-position).

Single source of truth so the two places that need hero's IP/OOP standing
-- the CSV ``Relative Position`` column (Layer 8) and the Layer 6 prompt's
SOLVER DATA block -- can't drift apart. (Previously the LLM had to *infer*
position and could get the blind-vs-blind case wrong; now it's handed the
same value the CSV uses.)
"""

from __future__ import annotations

from pipeline.preflop.fact_extractor import PreflopFacts

# Postflop action order: SB acts first ... BTN acts last. A higher rank means
# the seat acts LATER postflop = is in position. NOTE there is deliberately NO
# blind-vs-blind exception (July 2026 bugfix): at a ring table (6/8/9-max) the
# BvB SB still acts FIRST on every postflop street -- the BB has position.
# "The SB is the dealer" is true only at a literal 2-player table (heads-up
# format), which none of the packs are. An earlier version special-cased BvB
# to SB-is-IP and shipped wrong Relative Position / position prose / position
# skills on every BvB SB row; tests now pin the ring-table rule.
_POSTFLOP_RANK: dict[str, int] = {
    "SB": 0, "BB": 1, "UTG": 2, "UTG+1": 3, "UTG+2": 4,
    "LJ": 5, "HJ": 6, "CO": 7, "BTN": 8,
}


def ip_oop_positions(hero_pos: str, villain_pos: str) -> tuple[str, str]:
    """Return ``(ip_position, oop_position)`` for a 2-way preflop spot.

    The seat with the higher postflop rank (acts later) is in position.
    Blind-vs-blind follows the same rule: the BB acts after the SB postflop,
    so the BB is IP (no special case -- see the module comment).
    """
    hero_rank = _POSTFLOP_RANK.get(hero_pos, -1)
    villain_rank = _POSTFLOP_RANK.get(villain_pos, -1)
    if hero_rank > villain_rank:
        return (hero_pos, villain_pos)
    return (villain_pos, hero_pos)


def hero_relative_position(facts: PreflopFacts) -> str:
    """Hero's IP/OOP standing as ``"In Position"`` / ``"Out of Position"``.

    With a villain: hero is IP iff it acts last postflop. On open spots (no
    villain yet) the opener is in position only when nobody behind acts
    later postflop -- true only for the BTN. Every other open (including
    the SB first-in, where the BB is behind preflop AND in position
    postflop) leaves a seat acting after hero.
    """
    hero = facts.spot.node.actor
    if facts.villain_stats is None:
        return "In Position" if hero == "BTN" else "Out of Position"
    ip_pos, _oop_pos = ip_oop_positions(hero, facts.villain_stats.position)
    return "In Position" if hero == ip_pos else "Out of Position"


__all__ = ["hero_relative_position", "ip_oop_positions"]
