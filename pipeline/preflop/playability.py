"""Deterministic postflop-playability profile of a starting hand.

The LLM keeps getting hand playability wrong when it free-styles it (e.g. it
called A9s's straight equity "wheelish", which is false -- A9s is disconnected,
its only straight is the bare ace playing a wheel, same as any ace). This
module computes, with zero poker judgement left to the model, exactly what a
169-class starting hand can make after the flop:

  * flush potential (nut / non-nut / none), from suitedness + whether it holds
    an ace,
  * straight connectivity (open-ender ... disconnected), from the rank gap with
    ace-low handled,
  * whether it makes the Broadway or the wheel straight specifically,
  * whether it makes sets (pocket pair),
  * kicker quality + domination risk when it pairs its top card.

Everything is a pure function of the 169-class label. The explanation
generator puts this in the SOLVER DATA block as ``postflop_playability`` and the
prompt instructs the model to describe playability ONLY from these facts -- it
is never allowed to name a draw or straight type the facts do not list. This is
the architectural fix for the "wheelish straights" failure: compute the truth,
let the LLM only phrase it.
"""
from __future__ import annotations

from typing import Any

# rank char -> high value (ace high). Ace also plays low (=1) for the wheel.
_RANK_VAL: dict[str, int] = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def _parse_class(hand_class: str) -> tuple[str, str, str] | None:
    """('A', '9', 'suited') for 'A9s'; ('7', '7', 'pair') for '77'.

    Returns None if the label is not a recognizable 169-class.
    """
    hc = hand_class.strip()
    if len(hc) == 2 and hc[0] in _RANK_VAL and hc[1] in _RANK_VAL and hc[0] == hc[1]:
        return hc[0], hc[1], "pair"
    if len(hc) == 3 and hc[0] in _RANK_VAL and hc[1] in _RANK_VAL and hc[2] in "so":
        return hc[0], hc[1], ("suited" if hc[2] == "s" else "offsuit")
    return None


def hand_playability(hand_class: str) -> dict[str, Any]:
    """Deterministic postflop-playability facts for a 169-class label.

    See module docstring. Returns a JSON-ready dict for the SOLVER DATA block.
    For an unrecognized label, returns ``{"recognized": False}`` so callers can
    skip it rather than emit garbage.
    """
    parsed = _parse_class(hand_class)
    if parsed is None:
        return {"recognized": False}
    r1, r2, kind = parsed
    v1, v2 = _RANK_VAL[r1], _RANK_VAL[r2]
    hi, lo = max(v1, v2), min(v1, v2)
    suited = kind == "suited"
    is_pair = kind == "pair"
    has_ace = "A" in (r1, r2)

    # --- flush --------------------------------------------------------------
    if is_pair or not suited:
        flush = "none"
    elif has_ace:
        flush = "nut_flush"          # any Axs is the nut flush
    else:
        flush = "non_nut_flush"      # Kxs and below can be out-flushed

    # --- straight -----------------------------------------------------------
    makes_wheel = False
    makes_broadway = False
    if is_pair:
        straight = "none"            # a pair makes sets, not straights
    else:
        span = hi - lo               # ace-high span
        non_ace = lo if hi == 14 else hi
        if has_ace:
            span = min(span, non_ace - 1)   # ace plays low: A=1
            if non_ace <= 5:                # A2..A5 make A-2-3-4-5
                makes_wheel = True
        if lo >= 10:                 # both cards T..A -> can make A-K-Q-J-T
            makes_broadway = True
        if span >= 5:
            straight = "disconnected"       # the two cards can never share a straight
        elif span == 1:
            straight = "open_ender"
        elif span == 2:
            straight = "one_gapper"
        elif span == 3:
            straight = "two_gapper"
        else:                        # span == 4
            straight = "three_gapper"

    # --- sets + top-pair kicker / domination --------------------------------
    makes_sets = is_pair
    top_pair_kicker = "n/a"
    domination = "none"
    if not is_pair:
        top_char = r1 if v1 >= v2 else r2
        kicker = lo                  # the lower card kicks when you pair the top
        if top_char == "A":
            if kicker >= 13:
                top_pair_kicker = "top"        # AK
            elif kicker >= 11:
                top_pair_kicker = "strong"     # AQ, AJ
            elif kicker == 10:
                top_pair_kicker = "playable"   # AT
            else:
                top_pair_kicker = "weak"       # A9..A2
                domination = "ace_weak_kicker"
        elif top_char == "K":
            if kicker >= 12:
                top_pair_kicker = "strong"     # KQ
            elif kicker >= 10:
                top_pair_kicker = "playable"   # KJ, KT
            else:
                top_pair_kicker = "weak"       # K9..K2
                domination = "king_weak_kicker"

    # --- the deterministic "where value comes from" headline ----------------
    primary_value: list[str] = []
    if flush == "nut_flush":
        primary_value.append("the nut flush")
    elif flush == "non_nut_flush":
        primary_value.append("a flush (not the nuts)")
    if makes_broadway:
        primary_value.append("the nut straight (Broadway)")
    elif makes_wheel:
        primary_value.append("the wheel")
    elif straight in ("open_ender", "one_gapper"):
        primary_value.append("straights")
    if makes_sets:
        primary_value.append("sets")
    if not primary_value:
        # no draw, no pair: value is just high-card / pairing the board
        primary_value.append("top pair and high-card strength")

    return {
        "recognized": True,
        "suited": suited,
        "pocket_pair": is_pair,
        "flush_potential": flush,
        "straight_potential": straight,
        "makes_broadway_straight": makes_broadway,
        "makes_wheel_straight": makes_wheel,
        "makes_sets": makes_sets,
        "top_pair_kicker": top_pair_kicker,
        "domination_risk": domination,
        "primary_value": primary_value,
    }
