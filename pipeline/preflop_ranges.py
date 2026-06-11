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


def combo_str_to_hand_class(combo: str) -> str:
    """Hand-class label for a 4-character combo string like 'AhKc' or 'AhAd'."""
    rank_a = _RANKS.index(combo[0])
    rank_b = _RANKS.index(combo[2])
    suit_a = _SUITS.index(combo[1])
    suit_b = _SUITS.index(combo[3])
    return combo_to_hand_class(rank_a * 4 + suit_a, rank_b * 4 + suit_b)


# Ryan-pack canonical pivot order for the 169 hand classes: A row first, then
# 2 row, 3 row, ..., K row. For each pivot, list the pair, then every
# remaining rank suited+offsuit (higher-rank shown first in the label).
# Verified against the first 100 chars of the BB-call-vs-BTN-open pack file:
#   AA:0.0,A2s:1.0,A2o:0.0,A3s:1.0,...,AKo:0.0,22:1.0,32s:1.0,32o:0.0,...
_RYAN_PIVOT_ORDER = "A23456789TJQK"


def canonical_169_hand_classes() -> list[str]:
    """The 169 preflop hand-class labels in Ryan's pack canonical ordering.

    Used by Layer 8's ip_range / oop_range CSV columns (May 2026 Ryan ask)
    so the serialized order matches the team's existing preflop pack format
    -- enables future UI to consume both interchangeably without re-sorting.

    Total = 25 (A row) + 23 (2) + 21 (3) + ... + 1 (K) = 169.
    """
    rank_values = {r: i for i, r in enumerate(_RANKS)}
    classes: list[str] = []
    for i, pivot in enumerate(_RYAN_PIVOT_ORDER):
        classes.append(pivot + pivot)            # pair first
        for other in _RYAN_PIVOT_ORDER[i + 1:]:
            high = pivot if rank_values[pivot] > rank_values[other] else other
            low = other if high == pivot else pivot
            classes.append(high + low + "s")
            classes.append(high + low + "o")
    return classes


def aggregate_combo_range_to_classes(combo_range: dict[str, float],
                                      board: list[str]) -> dict[str, float]:
    """Aggregate a combo-level range to 169 preflop hand-class weights.

    `combo_range` maps 4-char combo strings (e.g. 'AhAd', 'AhKc') to per-combo
    weights in [0, 1]. `board` is the list of community cards to filter out
    board-blocked combos.

    For each of the 169 hand classes, the result is the MEAN weight over the
    unblocked combos of that class that appear in `combo_range`. Classes with
    no unblocked combos in the range map to 0.0 -- typically because every
    combo of the class shares a card with the board, or because the player's
    range simply doesn't include any combo of that class.

    The 169-entry assertion validates downstream UI assumptions: the output
    is always a complete 169-entry dict, even when most classes are 0.0.
    """
    board_cards = set(board)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for combo, weight in combo_range.items():
        # Skip combos that share a card with the board (impossible holdings).
        if combo[:2] in board_cards or combo[2:] in board_cards:
            continue
        cls = combo_str_to_hand_class(combo)
        sums[cls] = sums.get(cls, 0.0) + float(weight)
        counts[cls] = counts.get(cls, 0) + 1
    out: dict[str, float] = {}
    for cls in canonical_169_hand_classes():
        c = counts.get(cls, 0)
        out[cls] = sums[cls] / c if c > 0 else 0.0
    if len(out) != 169:
        raise AssertionError(
            f"hand-class aggregation produced {len(out)} entries; expected 169")
    return out


def format_hand_class_range(class_weights: dict[str, float]) -> str:
    """Serialise a 169-class weight dict in Ryan's pack format.

    Output is a single line: 'AA:0.0,A2s:1.0,A2o:0.0,A3s:1.0,...' -- 169
    comma-separated `Hand:weight` pairs in canonical Ryan-pack order. Weights
    are formatted compactly (trailing zeros stripped), matching the pack files.
    Used by Layer 8's ip_range / oop_range CSV columns; consumers can either
    parse it back via `parse_range_file` or render directly into a UI grid.
    """
    if len(class_weights) != 169:
        raise ValueError(
            f"format_hand_class_range expected 169 entries, got {len(class_weights)}")
    parts = []
    for cls in canonical_169_hand_classes():
        if cls not in class_weights:
            raise KeyError(
                f"class {cls!r} missing from input dict -- aggregator must "
                f"emit the full 169-entry canonical set, including zeros.")
        parts.append(f"{cls}:{_format_weight(class_weights[cls])}")
    return ",".join(parts)


def parse_range_file(path: Path | str) -> dict[str, float]:
    """Parse a preflop range file into a `{hand_class: weight}` dict.

    Dispatches on the file extension:

      * ``.rng`` -- a MonkerSolver-export node file (the NLHE 9-max pack);
        the weight column of :func:`parse_monker_rng_file`.
      * anything else -- Ryan-pack PioViewer format: a single line of
        `Hand:weight` pairs separated by commas. Whitespace is tolerated.

    Both formats carry all 169 hand classes (zero-weight ones included);
    a sanity check enforces that count. The weights have the same
    semantics in both packs: the JOINT probability that the hand reaches
    the node AND takes this action (the spot sampler normalises by
    presence downstream).
    """
    path = Path(path)
    if path.suffix == ".rng":
        return {hand: p for hand, (p, _ev) in parse_monker_rng_file(path).items()}
    text = path.read_text(encoding="utf-8").strip()
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


def parse_monker_rng_file(path: Path | str) -> dict[str, tuple[float, float]]:
    """Parse a MonkerSolver-export NLHE `.rng` node file.

    Format (see ``docs/nlhe9_pack_notes.md``): ASCII text, two lines per
    hand class, 169 classes per file::

        AA
        1.0;8313.56063199635
        A2s
        0.546;0.017402295110777916
        ...

    Line 1 of each pair is the hand-class label (the same 169 labels as
    the Ryan pack: ``AA``, ``A2s``, ``A2o``...), line 2 is
    ``<weight>;<ev>``. Unlike the PLO pack's `.rng` files (16,432
    pattern lines, order-implicit), these are self-labeling -- the label
    is read, not inferred from line position.

    Returns:
        ``{hand_class: (weight, ev_raw)}`` for all 169 classes. ``weight``
        is the joint reach-and-take-this-action probability in [0, 1].
        ``ev_raw`` is Monker's EV in milli-big-blinds measured from the
        start of the hand (verified June 2026, see
        ``docs/nlhe9_pack_notes.md`` -- NOT the PLO pack's milli-small-
        blinds). Because the baseline is per-hand constant, EV gaps
        between actions are ``(ev_a - ev_b) / 1000`` in bb. A missing EV
        field (``"0.5;"`` or a bare ``"0.5"`` -- the export drops the
        field on some deep low-traffic nodes) parses as 0.0, mirroring
        the PLO reader's tolerance.

    Raises:
        ValueError: wrong line count, a duplicate or non-canonical label
            set, or unparseable floats.
    """
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
             if ln.strip()]
    expected = 2 * 169
    if len(lines) != expected:
        raise ValueError(
            f"{path.name}: expected {expected} non-empty lines "
            f"(169 label + weight;ev pairs), got {len(lines)} -- not a "
            "169-class NLHE Monker .rng")
    out: dict[str, tuple[float, float]] = {}
    for i in range(169):
        label = lines[2 * i]
        payload = lines[2 * i + 1]
        p_str, _sep, ev_str = payload.partition(";")
        try:
            weight = float(p_str)
            ev = float(ev_str) if ev_str else 0.0
        except ValueError as exc:
            raise ValueError(
                f"{path.name}: unparseable weight/EV pair {payload!r} "
                f"for hand {label!r}") from exc
        if label in out:
            raise ValueError(f"{path.name}: duplicate hand label {label!r}")
        out[label] = (weight, ev)
    if set(out) != set(canonical_169_hand_classes()):
        missing = sorted(set(canonical_169_hand_classes()) - set(out))[:5]
        extra = sorted(set(out) - set(canonical_169_hand_classes()))[:5]
        raise ValueError(
            f"{path.name}: label set is not the canonical 169 hand classes "
            f"(missing e.g. {missing}, unexpected e.g. {extra})")
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
    "aggregate_combo_range_to_classes",
    "canonical_169_hand_classes",
    "card_label",
    "combo_cards",
    "combo_label",
    "combo_str_to_hand_class",
    "combo_to_hand_class",
    "expand_to_combo_weights",
    "format_hand_class_range",
    "format_set_range_line",
    "parse_monker_rng_file",
    "parse_range_file",
]
