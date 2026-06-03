"""PLO hand model -- structural classification of a 4-card preflop hand.

NLHE collapses to 169 classes (the 13x13 grid); PLO has 270,725 combos and
no clean equivalent, so we don't enumerate -- we *label*. This module turns
four hole cards into the structural facts PLO reasoning actually keys on:
suit pattern, pair pattern, connectedness / wrap potential, and nut-flush /
high-card quality, plus a tunable strength bucket and a human-readable
descriptor.

It is preflop-only (no board), pure (cards in, label out), and -- per the
project decision to drop PLO range *display* -- exists to feed concept tags,
the difficulty hand-axis, and prose, NOT a chart. It needs no preflop pack,
so it is buildable before the pack lands.

Design mirrors NLHE conventions: the strength bucket is a categorical key
(like ``difficulty.HAND_CLASS_EASE``'s keys) that a future PLO difficulty
axis maps to an ease value. The scoring constants below are starting values
-- tune against graded output, the same way NLHE's thresholds are tuned.
The objective structural fields (suit/pair/connectedness/flags) are exact;
only ``strength`` is heuristic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from pipeline.cards import BROADWAY, card_suit, parse_card, rank_value

_PLO_HAND_LEN = 4

# Two hole cards can share a straight only if their ranks are within this
# span (a 5-card straight, using exactly 2 hole cards). A card farther than
# this from every other hole card is a "dangler" -- dead for straights.
_STRAIGHT_REACH = 4

_RANK_NAME: dict[int, str] = {
    14: "ace", 13: "king", 12: "queen", 11: "jack", 10: "ten",
    9: "nine", 8: "eight", 7: "seven", 6: "six", 5: "five",
    4: "four", 3: "three", 2: "two",
}


@dataclass(frozen=True)
class PloHandClass:
    """Structural label for a 4-card PLO preflop hand.

    All fields except ``strength`` (and the prose ``descriptor``) are exact,
    objective properties of the four cards.
    """

    cards: tuple[str, str, str, str]
    # double_suited | single_suited | rainbow | three_suited | monotone
    suit_pattern: str
    # unpaired | one_pair | two_pair | trips | quads
    pair_pattern: str
    pair_ranks: tuple[int, ...]  # paired rank values, high -> low (AA -> (14,))
    distinct_ranks: tuple[int, ...]  # distinct rank values, high -> low
    # rundown | one_gapper | two_gapper | connected | disconnected
    connectedness: str
    span: int  # best-ace rank spread among the distinct ranks
    has_dangler: bool
    has_ace: bool
    suited_ace: bool  # an ace sharing a suit with another card (nut-flush)
    broadway_count: int  # cards of rank T..A
    double_suited: bool
    wrap_potential: bool  # unpaired + tightly connected -> flops big wraps
    strength: str  # premium | strong | medium | marginal | weak | trash
    descriptor: str  # human-readable, for prose


# --- normalisation --------------------------------------------------------
def _normalize(hand: object) -> tuple[str, str, str, str]:
    """Parse a PLO hand into four distinct, rank-sorted cards.

    Accepts an iterable of four card tokens or an 8-char combo string
    (``"AhKhQsJs"``). Sorted high rank -> low (suit as a stable tiebreak).
    """
    if isinstance(hand, str):
        text = hand.strip().replace("10", "T")
        tokens = text.split() if " " in text else [text[i : i + 2] for i in range(0, len(text), 2)]
    else:
        tokens = [str(t) for t in hand]
    cards = [parse_card(t) for t in tokens]
    if len(cards) != _PLO_HAND_LEN:
        msg = f"PLO hand needs exactly 4 cards, got {len(cards)}: {cards!r}"
        raise ValueError(msg)
    if len(set(cards)) != _PLO_HAND_LEN:
        msg = f"PLO hand has duplicate cards: {cards!r}"
        raise ValueError(msg)
    ordered = sorted(cards, key=lambda c: (rank_value(c), card_suit(c)), reverse=True)
    return (ordered[0], ordered[1], ordered[2], ordered[3])


# --- suit / pair structure ------------------------------------------------
def _suit_pattern(cards: tuple[str, ...]) -> str:
    counts = sorted(Counter(card_suit(c) for c in cards).values(), reverse=True)
    return {
        (2, 2): "double_suited",
        (2, 1, 1): "single_suited",
        (1, 1, 1, 1): "rainbow",
        (3, 1): "three_suited",
        (4,): "monotone",
    }[tuple(counts)]


def _pair_structure(cards: tuple[str, ...]) -> tuple[str, tuple[int, ...]]:
    rank_counts = Counter(rank_value(c) for c in cards)
    paired = tuple(sorted((r for r, n in rank_counts.items() if n >= 2), reverse=True))
    counts = sorted(rank_counts.values(), reverse=True)
    pattern = {
        (1, 1, 1, 1): "unpaired",
        (2, 1, 1): "one_pair",
        (2, 2): "two_pair",
        (3, 1): "trips",
        (4,): "quads",
    }[tuple(counts)]
    return pattern, paired


# --- connectedness / danglers (ace plays high or low) ---------------------
def _layout_metrics(distinct: list[int]) -> tuple[int, bool]:
    """``(total_gap, has_dangler)`` for one ace-placement of the distinct ranks.

    ``total_gap`` = missing ranks between the lowest and highest distinct
    card. A dangler is a card whose nearest neighbour is more than
    :data:`_STRAIGHT_REACH` ranks away.
    """
    s = sorted(distinct)
    total_gap = (s[-1] - s[0]) - (len(s) - 1)
    has_dangler = any(
        min(abs(v - o) for j, o in enumerate(s) if j != i) > _STRAIGHT_REACH
        for i, v in enumerate(s)
    ) if len(s) > 1 else True
    return total_gap, has_dangler


def _connectedness(
    distinct_high: tuple[int, ...], pair_pattern: str
) -> tuple[str, int, bool]:
    """Classify straight/wrap shape. Returns ``(category, span, has_dangler)``.

    Evaluates ace-high and (if an ace is present) ace-low, keeping whichever
    layout is better connected.
    """
    layouts = [list(distinct_high)]
    if 14 in distinct_high:
        layouts.append([1 if r == 14 else r for r in distinct_high])

    best_gap, best_dangler, best_span = None, True, 0
    for layout in layouts:
        gap, dangler = _layout_metrics(layout)
        if best_gap is None or (gap, dangler) < (best_gap, best_dangler):
            best_gap, best_dangler = gap, dangler
            s = sorted(layout)
            best_span = s[-1] - s[0]
    assert best_gap is not None

    if pair_pattern == "unpaired":
        if best_gap == 0:
            category = "rundown"
        elif best_gap == 1:
            category = "one_gapper"
        elif best_gap == 2:
            category = "two_gapper"
        else:
            category = "disconnected" if best_dangler else "connected"
    else:
        category = "disconnected" if best_dangler else "connected"
    return category, best_span, best_dangler


# --- nut / high-card flags -------------------------------------------------
def _suited_ace(cards: tuple[str, ...]) -> bool:
    """True if an ace shares its suit with another card (nut-flush potential)."""
    return any(
        rank_value(a) == 14 and any(card_suit(a) == card_suit(b) for b in cards if b != a)
        for a in cards
    )


# --- strength heuristic (TUNABLE) -----------------------------------------
# Starting values. Each rule is a separate, documented contribution so the
# weighting is easy to read and retune against graded output.
_SUIT_SCORE = {
    "double_suited": 2.0,
    "single_suited": 1.0,
    "rainbow": -0.5,
    "three_suited": -0.5,
    "monotone": -1.0,
}
_CONNECT_SCORE = {
    "rundown": 2.0,
    "one_gapper": 1.0,
    "two_gapper": 0.5,
    "connected": 0.0,
    "disconnected": -1.0,
}
_STRENGTH_BANDS = [  # (min_score, label), checked high -> low
    (5.0, "premium"),
    (3.0, "strong"),
    (1.0, "medium"),
    (-0.5, "marginal"),
    (-1.5, "weak"),
]
_LOW_CARD_CEILING = 8  # all cards <= this -> non-nut everything


def _pair_score(pair_pattern: str, pair_ranks: tuple[int, ...]) -> float:
    if pair_pattern == "one_pair":
        top = pair_ranks[0]
        return 2.0 if top >= 12 else 1.0 if top >= 9 else 0.0
    if pair_pattern == "two_pair":
        top = pair_ranks[0]
        return 1.5 if top >= 12 else 0.75 if top >= 9 else 0.25
    if pair_pattern == "trips":
        return -2.0
    if pair_pattern == "quads":
        return -3.0
    return 0.0  # unpaired


def _strength(
    suit_pattern: str,
    pair_pattern: str,
    pair_ranks: tuple[int, ...],
    connectedness: str,
    distinct_ranks: tuple[int, ...],
    broadway_count: int,
    has_ace: bool,
    suited_ace: bool,
) -> str:
    score = _SUIT_SCORE[suit_pattern] + _CONNECT_SCORE[connectedness]
    score += _pair_score(pair_pattern, pair_ranks)
    if broadway_count == _PLO_HAND_LEN:
        score += 1.0
    if suited_ace:
        score += 1.0
    elif has_ace:
        score += 0.25
    if max(distinct_ranks) <= _LOW_CARD_CEILING:
        score -= 1.5
    for threshold, label in _STRENGTH_BANDS:
        if score >= threshold:
            return label
    return "trash"


# --- descriptor (prose) ----------------------------------------------------
_SUIT_WORD = {
    "double_suited": "double-suited",
    "single_suited": "single-suited",
    "rainbow": "rainbow",
    "three_suited": "three-suited",
    "monotone": "monotone",
}


def _descriptor(
    suit_pattern: str,
    pair_pattern: str,
    pair_ranks: tuple[int, ...],
    distinct_ranks: tuple[int, ...],
    connectedness: str,
    broadway_count: int,
    has_dangler: bool,
) -> str:
    suit_word = _SUIT_WORD[suit_pattern]
    high_name = _RANK_NAME[max(distinct_ranks)]

    if pair_pattern == "quads":
        core = f"quad {_RANK_NAME[pair_ranks[0]]}s"
    elif pair_pattern == "trips":
        core = f"trip {_RANK_NAME[pair_ranks[0]]}s"
    elif pair_pattern == "two_pair":
        core = f"{_RANK_NAME[pair_ranks[0]]}s and {_RANK_NAME[pair_ranks[1]]}s"
    elif pair_pattern == "one_pair":
        core = f"{_RANK_NAME[pair_ranks[0]]}s"
    elif connectedness == "rundown" and broadway_count == _PLO_HAND_LEN:
        core = "broadway rundown"
    else:
        shape = {
            "rundown": "rundown",
            "one_gapper": "one-gapper",
            "two_gapper": "two-gapper",
            "connected": "connected",
            "disconnected": "disconnected",
        }[connectedness]
        core = f"{high_name}-high {shape}"

    text = f"{suit_word} {core}"
    if has_dangler and pair_pattern in {"one_pair", "two_pair"}:
        text += " with a dangler"
    return text


# --- public entry point ----------------------------------------------------
def classify_plo_hand(hand: object) -> PloHandClass:
    """Classify a 4-card PLO preflop hand.

    ``hand`` is an iterable of four card tokens or an 8-char combo string.
    """
    cards = _normalize(hand)
    suit_pattern = _suit_pattern(cards)
    pair_pattern, pair_ranks = _pair_structure(cards)
    distinct_ranks = tuple(sorted({rank_value(c) for c in cards}, reverse=True))
    connectedness, span, has_dangler = _connectedness(distinct_ranks, pair_pattern)

    has_ace = 14 in distinct_ranks
    suited_ace = _suited_ace(cards)
    broadway_count = sum(1 for c in cards if rank_value(c) in BROADWAY)
    double_suited = suit_pattern == "double_suited"
    wrap_potential = pair_pattern == "unpaired" and connectedness in {
        "rundown",
        "one_gapper",
    }
    strength = _strength(
        suit_pattern,
        pair_pattern,
        pair_ranks,
        connectedness,
        distinct_ranks,
        broadway_count,
        has_ace,
        suited_ace,
    )
    descriptor = _descriptor(
        suit_pattern,
        pair_pattern,
        pair_ranks,
        distinct_ranks,
        connectedness,
        broadway_count,
        has_dangler,
    )
    return PloHandClass(
        cards=cards,
        suit_pattern=suit_pattern,
        pair_pattern=pair_pattern,
        pair_ranks=pair_ranks,
        distinct_ranks=distinct_ranks,
        connectedness=connectedness,
        span=span,
        has_dangler=has_dangler,
        has_ace=has_ace,
        suited_ace=suited_ace,
        broadway_count=broadway_count,
        double_suited=double_suited,
        wrap_potential=wrap_potential,
        strength=strength,
        descriptor=descriptor,
    )
