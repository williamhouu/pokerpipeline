"""Helpers for expanding Ryan-pack preflop range files into PioSolver's
canonical 1326-combo weight vector.

Ryan's range files (`ranges/ryan_preflop_tree/.../<position>/<filename>.txt`)
hold one weight per of the 169 preflop hand classes:
`AA:1.0,A2s:0.621,A2o:0.0,A3s:1.0,...`. PioSolver's UPI `set_range OOP/IP`
command takes 1326 combo-level floats, in the order PioSolver enumerates
combinations of 2 cards from 52. This module bridges the two.

Pio's combo enumeration (verified via `show_hand_order()` against the
PioSolver Edge 3 build at C:\\PioSOLVER\\PioSOLVER3-edge.exe):

  * Card index: `rank_index * 4 + suit_index`, where ranks are 2..A (0..12)
    and suits are c,d,h,s (0..3). So 2c=0, 2d=1, 2h=2, 2s=3, 3c=4, ..., As=51.
  * Combo position: for a combo with card_a_idx > card_b_idx,
    `position = card_a_idx * (card_a_idx - 1) // 2 + card_b_idx`.
    This is the standard C(52, 2) enumeration walking outer card from low
    to high and inner card from low to one-less-than-outer.

This module is import-cheap (no Pio, no I/O at import time).
"""
from __future__ import annotations

from pathlib import Path

CARD_COUNT = 52
HAND_COUNT = 1326                              # C(52, 2)

# Pio's card order: rank slow, suit fast, ranks low-to-high, suits c d h s.
_RANKS = "23456789TJQKA"                       # ascending; index 0..12
_SUITS = "cdhs"                                # ascending; index 0..3


def card_label(card_idx: int) -> str:
    """Two-char card label (e.g. '2c', 'As') from Pio's card index."""
    if not 0 <= card_idx < CARD_COUNT:
        raise ValueError(f"card_idx out of range: {card_idx}")
    return _RANKS[card_idx // 4] + _SUITS[card_idx % 4]


def combo_label(position: int) -> str:
    """The 4-character combo label at Pio's hand-order position
    (e.g. position 0 -> '2d2c'). The higher-indexed card comes first."""
    card_a_idx, card_b_idx = combo_cards(position)
    return card_label(card_a_idx) + card_label(card_b_idx)


def combo_cards(position: int) -> tuple[int, int]:
    """Decode a Pio hand-order position into its two card indices.

    Returns `(card_a_idx, card_b_idx)` with `card_a_idx > card_b_idx`.
    """
    if not 0 <= position < HAND_COUNT:
        raise ValueError(f"position out of range: {position}")
    # Largest a such that a*(a-1)/2 <= position.
    a = 1
    while a * (a - 1) // 2 <= position:
        a += 1
    a -= 1
    b = position - a * (a - 1) // 2
    return a, b


def combo_to_hand_class(card_a_idx: int, card_b_idx: int) -> str:
    """The 169-hand-class label (e.g. 'AA', 'AKs', 'AKo') for a card pair."""
    rank_a, suit_a = divmod(card_a_idx, 4)
    rank_b, suit_b = divmod(card_b_idx, 4)
    if rank_a == rank_b:
        return _RANKS[rank_a] + _RANKS[rank_b]
    # Higher rank first in the label.
    if rank_a > rank_b:
        high_rank, low_rank = rank_a, rank_b
    else:
        high_rank, low_rank = rank_b, rank_a
    suited = "s" if suit_a == suit_b else "o"
    return _RANKS[high_rank] + _RANKS[low_rank] + suited


def parse_range_file(path: Path | str) -> dict[str, float]:
    """Parse a Ryan-pack range file into a `{hand_class: weight}` dict.

    The file format is a single line of `Hand:weight` pairs separated by
    commas. Whitespace is tolerated. Expected to carry all 169 hand classes;
    a sanity check enforces that count.
    """
    text = Path(path).read_text(encoding="utf-8").strip()
    out: dict[str, float] = {}
    for pair in text.split(","):
        pair = pair.strip()
        if not pair:
            continue
        hand, weight = pair.split(":")
        out[hand.strip()] = float(weight)
    if len(out) != 169:
        raise ValueError(
            f"expected 169 hand classes in {path}, got {len(out)}. The pack "
            f"format assumes all 169 entries are present (zero-weight ones "
            f"included). Mismatches likely mean a corrupted or truncated file.")
    return out


def expand_to_combo_weights(hand_class_weights: dict[str, float]) -> list[float]:
    """Expand a 169-hand-class weight dict to 1326 combo weights in Pio's
    canonical hand-order. Every combo gets the weight of its hand class
    (a class's combos all share its weight)."""
    weights = [0.0] * HAND_COUNT
    for position in range(HAND_COUNT):
        a, b = combo_cards(position)
        cls = combo_to_hand_class(a, b)
        try:
            weights[position] = hand_class_weights[cls]
        except KeyError as exc:
            raise KeyError(
                f"hand class {cls!r} not in input dict (position={position}, "
                f"combo={combo_label(position)}); range file likely missing "
                f"this class") from exc
    return weights


def format_set_range_line(side: str, weights: list[float]) -> str:
    """Format a `set_range OOP|IP <1326 floats>` UPI command line.

    PioSolver Edge accepts plain floats; we use 6 decimals to preserve the
    full precision Ryan's pack carries while keeping the line size sane.
    Trailing zeros are stripped to match what Pio's GUI tree-builder writes.
    """
    if side not in ("OOP", "IP"):
        raise ValueError(f"side must be 'OOP' or 'IP', got {side!r}")
    if len(weights) != HAND_COUNT:
        raise ValueError(
            f"expected {HAND_COUNT} weights, got {len(weights)}")
    return "set_range " + side + " " + " ".join(_format_weight(w) for w in weights)


def _format_weight(w: float) -> str:
    """Compact float formatting: trim trailing zeros after the decimal."""
    if w == 0:
        return "0"
    if w == 1:
        return "1"
    s = f"{w:.6f}".rstrip("0").rstrip(".")
    return s or "0"


__all__ = [
    "CARD_COUNT",
    "HAND_COUNT",
    "card_label",
    "combo_cards",
    "combo_label",
    "combo_to_hand_class",
    "expand_to_combo_weights",
    "format_set_range_line",
    "parse_range_file",
]
