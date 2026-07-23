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

from pipeline.cards import (
    BROADWAY,
    RANK_VALUE,
    card_rank,
    card_suit,
    parse_card,
    rank_value,
)

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
def _layout_metrics(
    distinct: list[int], unpaired: frozenset[int]
) -> tuple[int, bool, bool]:
    """``(total_gap, far_rank, has_dangler)`` for one ace-placement.

    ``total_gap`` = missing ranks between the lowest and highest distinct
    card. ``far_rank`` is True when ANY rank's nearest neighbour is more than
    :data:`_STRAIGHT_REACH` away (drives the connected/disconnected category).
    ``has_dangler`` restricts that to UNPAIRED ranks (high-ace values): a
    paired rank is never a dangler -- its value is the pair (set-mining), not
    connectivity -- so double-paired hands like KK44 and quads have none,
    while QQJ2's deuce still is one. Distances are still measured against ALL
    ranks (a side card sitting next to the pair rank is connected to it).
    """
    s = sorted(distinct)
    total_gap = (s[-1] - s[0]) - (len(s) - 1)
    if len(s) <= 1:
        # Quads: keep the historical "disconnected" category (no straight
        # coordination), but all ranks are paired so there is no dangler.
        return total_gap, True, False

    def _is_far(i: int, v: int) -> bool:
        return min(abs(v - o) for j, o in enumerate(s) if j != i) > _STRAIGHT_REACH

    far_flags = [(_orig_rank(v) in unpaired, _is_far(i, v)) for i, v in enumerate(s)]
    far_rank = any(far for _, far in far_flags)
    has_dangler = any(far and is_unpaired for is_unpaired, far in far_flags)
    return total_gap, far_rank, has_dangler


def _orig_rank(layout_value: int) -> int:
    """Map an ace-low layout value back to the high-ace rank (1 -> 14)."""
    return 14 if layout_value == 1 else layout_value


def _connectedness(
    distinct_high: tuple[int, ...],
    pair_pattern: str,
    pair_ranks: tuple[int, ...],
) -> tuple[str, int, bool]:
    """Classify straight/wrap shape. Returns ``(category, span, has_dangler)``.

    Evaluates ace-high and (if an ace is present) ace-low, keeping whichever
    layout is better connected. The category keys on raw rank distance (KK44
    IS disconnected as a straight shape); the dangler flag additionally
    requires the far card to be unpaired (KK44 has NO dangler -- the fours
    are a set-mining pair, not a dead card).
    """
    unpaired = frozenset(set(distinct_high) - set(pair_ranks))
    layouts = [list(distinct_high)]
    if 14 in distinct_high:
        layouts.append([1 if r == 14 else r for r in distinct_high])

    best_gap, best_far, best_dangler, best_span = None, True, True, 0
    for layout in layouts:
        gap, far_rank, dangler = _layout_metrics(layout, unpaired)
        if best_gap is None or (gap, far_rank) < (best_gap, best_far):
            best_gap, best_far, best_dangler = gap, far_rank, dangler
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
            category = "disconnected" if best_far else "connected"
    else:
        category = "disconnected" if best_far else "connected"
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
    connectedness, span, has_dangler = _connectedness(
        distinct_ranks, pair_pattern, pair_ranks
    )

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


# --- flush nut-ranking (deterministic; feeds the Layer 6 SOLVER DATA) ------
_SUIT_CHAR_WORD = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}
# Plain text suit symbols for the Show-the-math labels ("A♦ 8♦"). Deliberately
# NOT the emoji-variant forms the Question prose uses -- panel labels render
# in Streamlit markdown and the CSV, where the bare glyphs are cleaner.
_SUIT_CHAR_SYMBOL = {"c": "♣", "d": "♦", "h": "♥", "s": "♠"}
# Label for a flush whose HIGHEST card is this rank. Ace = the nut flush, king =
# second-nut (it becomes the nut when the ace of that suit is on the board),
# queen = third, jack = fourth; anything lower is a weak (non-nut) flush.
# Only rankings players actually name: A/K/Q-high. Everything from the jack
# down is just "weak" -- the old J="fourth-nut" vs T="weak" split put a fake
# category cliff between adjacent ranks, and the LLM dramatized it into "the
# diamonds are a real flush, the clubs are a backup".
_FLUSH_NUT_LABEL: dict[str, str] = {
    "A": "nut",
    "K": "second-nut",
    "Q": "third-nut",
}


@dataclass(frozen=True)
class FlushSuit:
    """A suit hero holds 2+ cards in, and how nutted its flush would be."""

    suit: str  # 'd'
    suit_word: str  # 'diamonds'
    high_rank: str  # 'K'
    nut_label: str  # 'second-nut' (or 'weak' below the jack)
    is_nut: bool  # True only for an ace-high (true nut) flush


def flush_suits(hand: object) -> tuple[FlushSuit, ...]:
    """Each suit hero holds 2+ cards in, with its flush's nut ranking.

    A PLO flush uses two hole cards, so only a suit with 2+ of hero's cards has
    flush potential. The ranking keys off the highest held card: ace = the nut
    flush, king = second-nut, queen = third, jack = fourth, lower = weak.
    Returned high-card first. A rainbow / single-suited-with-no-pair hand may
    return an empty tuple (no two cards share a suit).
    """
    cards = _normalize(hand)
    by_suit: dict[str, list[str]] = {}
    for card in cards:
        by_suit.setdefault(card_suit(card), []).append(card)
    out: list[FlushSuit] = []
    for suit, suit_cards in by_suit.items():
        if len(suit_cards) < 2:  # noqa: PLR2004 -- a flush needs two hole cards
            continue
        high_rank = card_rank(max(suit_cards, key=rank_value))
        out.append(
            FlushSuit(
                suit=suit,
                suit_word=_SUIT_CHAR_WORD[suit],
                high_rank=high_rank,
                nut_label=_FLUSH_NUT_LABEL.get(high_rank, "weak"),
                is_nut=high_rank == "A",
            )
        )
    return tuple(sorted(out, key=lambda f: RANK_VALUE[f.high_rank], reverse=True))


_CLOSE_SUIT_GAP = 2  # weak suits within two ranks are "close in strength"


def _an(rank: str) -> str:
    """The article for ``<rank>-high`` ("an A-high flush", "a K-high flush")."""
    return "an" if rank in {"A", "8"} else "a"


def describe_flush_potential(hand: object) -> str:
    """One-line flush nut-ranking for the SOLVER DATA block.

    Draw-tense by construction: "can make at best a J-high flush" -- preflop
    there is no made flush, and "gives you a real flush" prose came straight
    from the old noun phrasing. When every suit is weak (J-high or below) the
    line says no suit makes the nut flush and, when they sit within two ranks
    of each other, that they are close in strength -- so the LLM cannot rank
    one weak suit far above another. Real rankings (A/K/Q-high) keep their
    labels: that hierarchy is true and worth teaching.

    E.g. ``"diamonds can make at best a J-high flush (weak), clubs can make
    at best a T-high flush (weak). Neither suit makes the nut flush, and the
    two are close in strength"`` or ``"none (no two cards share a suit)"``.
    """
    suits = flush_suits(hand)
    if not suits:
        return "none (no two cards share a suit)"
    parts = ", ".join(
        f"{f.suit_word} can make at best {_an(f.high_rank)} {f.high_rank}-high "
        f"flush ({f.nut_label})"
        for f in suits
    )
    if all(f.nut_label == "weak" for f in suits):
        if len(suits) == 1:
            return f"{parts}. Not the nut flush"
        gap = RANK_VALUE[suits[0].high_rank] - RANK_VALUE[suits[-1].high_rank]
        close = (
            ", and the two are close in strength" if gap <= _CLOSE_SUIT_GAP else ""
        )
        return f"{parts}. Neither suit makes the nut flush{close}"
    return parts


# --- card redundancy (trips / quads in hand; feeds the SOLVER DATA) ---------
_RANK_WORD_PLURAL = {
    14: "aces", 13: "kings", 12: "queens", 11: "jacks", 10: "tens",
    9: "nines", 8: "eights", 7: "sevens", 6: "sixes", 5: "fives",
    4: "fours", 3: "threes", 2: "deuces",
}


def describe_card_redundancy(hand: object) -> str | None:
    """Deterministic redundant-card count for trips/quads hands, or ``None``.

    A PLO hand uses exactly two hole cards, so a pair is the most that
    duplicated ranks can ever contribute. Three of a rank = exactly ONE
    redundant card (the pair uses two; the third adds no new two-card combo
    and only removes the hand's own set/full-house outs). Four of a rank =
    two redundant cards and no set outs at all. Stated here so Layer 6
    reports the count instead of doing the arithmetic itself (which produced
    "two of your three aces are dead" / "the fourth ace" style miscounts).
    Returns ``None`` for hands with no triplicated rank -- omit from the data
    block rather than stating a non-fact.
    """
    cards = _normalize(hand)
    rank_counts = Counter(rank_value(c) for c in cards)
    for rank, count in rank_counts.items():
        word = _RANK_WORD_PLURAL[rank]
        singular = word[:-1] if rank != 6 else "six"  # "sixes" -> "six"
        if count == 3:  # noqa: PLR2004
            return (
                f"you hold three {word} but a pair is the most two hole cards "
                f"can make, so exactly ONE of the three is redundant dead "
                f"weight. The hand plays as a bare pair of {word}, and only "
                f"one {singular} is left in the deck for sets or full houses."
            )
        if count == 4:  # noqa: PLR2004
            return (
                f"you hold all four {word}: two are pure dead weight (a pair "
                f"is the most two hole cards can make), and no {singular} "
                f"remains in the deck -- this hand can never flop a set."
            )
    return None


def flush_ceiling_stat_entries(hand: object) -> list[dict[str, str]]:
    """Show-the-math rows: one per suit hero can flush in, plus a no-flush row.

    The panel form of :func:`describe_flush_potential` (idea 1 of the July-16
    panel proposal): each suit hero holds 2+ cards in gets its own
    ``{key, label, value, note}`` entry stating the hard ceiling of any flush
    that suit can make. Vocabulary comes from the SAME :func:`flush_suits`
    rows that build the SOLVER DATA ``flush_potential`` line, so the panel
    and the data block can never disagree.

    Accuracy rules (each pinned by tests):

    * Every claim is about HERO'S OWN CEILING ("the best diamond flush this
      hand can make is K-high"), never about unbeatability -- a straight
      flush outranks even the nut flush, so "no one can beat it" style
      wording is deliberately absent.
    * A hand with no two cards of one suit (rainbow, and every trips/quads
      hand without a suited side card) emits the single "no flush possible"
      row: a PLO flush must use exactly two hole cards of the suit.
    * A three-suited/monotone suit shows only its TWO HIGHEST cards in the
      label -- the pair the best flush actually uses; the extra card is the
      dead-weight row's story (:func:`dead_weight_stat_entries`).
    """
    cards = _normalize(hand)
    suits = flush_suits(cards)
    if not suits:
        return [
            {
                "key": "flush_ceiling",
                "label": "Flush ceiling",
                # "NA" + plain-English note (July 2026, user wording ask).
                "value": "NA",
                "note": (
                    "You have a rainbow hand. Every card is a different "
                    "suit, and a flush needs two of your own cards in the "
                    "same suit, so this hand can never make one."
                ),
            }
        ]
    entries: list[dict[str, str]] = []
    for f in suits:
        held = sorted(
            (c for c in cards if card_suit(c) == f.suit),
            key=rank_value,
            reverse=True,
        )[:2]
        shown = " ".join(
            card_rank(c) + _SUIT_CHAR_SYMBOL[f.suit] for c in held
        )
        sym = _SUIT_CHAR_SYMBOL[f.suit]
        if f.is_nut:
            value = "nut flush possible"
            note = (
                f"You hold the A{sym}, so the best {f.suit_word} flush this "
                "hand can make is ace-high: the nut flush."
            )
        elif f.nut_label == "second-nut":
            value = "second-nut flush at best"
            note = (
                f"You do not hold the A{sym}, so the best {f.suit_word} "
                "flush this hand can make is K-high: the second-nut flush."
            )
        elif f.nut_label == "third-nut":
            value = "third-nut flush at best"
            note = (
                f"Your highest {f.suit_word[:-1]} is the Q{sym}, so the "
                f"best {f.suit_word} flush this hand can make is Q-high: "
                "the third-nut flush."
            )
        else:
            value = f"{f.high_rank}-high flush at best (weak)"
            note = (
                f"The best {f.suit_word} flush this hand can make is "
                f"{f.high_rank}-high, well below the nut flush. This is "
                "second-best-flush territory, the classic spot where "
                "non-nut suits lose a big pot."
            )
        entries.append(
            {
                "key": f"flush_ceiling_{f.suit_word}",
                "label": f"Flush ceiling: {f.suit_word} ({shown})",
                "value": value,
                "note": note,
            }
        )
    return entries


def dead_weight_stat_entries(hand: object) -> list[dict[str, str]]:
    """Show-the-math row for redundant cards: at most ONE entry, often none.

    The panel form of :func:`describe_card_redundancy` /
    :func:`describe_suit_redundancy` (idea 2 of the July-16 panel proposal).
    The note reuses the EXACT describe_* sentence already shipped in SOLVER
    DATA and validated in prose, single source of truth for the wording.

    Accuracy rules (each pinned by tests):

    * Rank and suit redundancy are mutually exclusive BY CONSTRUCTION
      (trips occupy three different suits, quads four), so this never
      emits two rows.
    * A plain pair (or two pair) emits NOTHING: a pair is fully playable
      (sets, full houses), not dead weight.
    * No specific card is named as "the dead one". For trips the three
      same-rank cards are interchangeable rank-wise but NOT suit-wise (a
      Q♥ sharing hearts with a 4th heart carries flush value its siblings
      don't), and a suit's lowest card can still hold rank value (it may
      pair the 4th card) -- naming one card would be false precision, so
      the row counts the redundancy instead.
    """
    cards = _normalize(hand)
    rank_counts = Counter(rank_value(c) for c in cards)
    trip_rank = next((r for r, n in rank_counts.items() if n >= 3), None)  # noqa: PLR2004
    if trip_rank is not None:
        note = describe_card_redundancy(cards)
        if note is None:  # pragma: no cover -- unreachable, defensive only
            return []
        # Panel captions stand alone, so sentence-case the reused SOLVER
        # DATA wording (which follows a field name there) at display time.
        note = note[:1].upper() + note[1:]
        count = rank_counts[trip_rank]
        word = _RANK_WORD_PLURAL[trip_rank]
        return [
            {
                "key": "dead_weight",
                "label": (
                    f"Dead weight: three {word}"
                    if count == 3  # noqa: PLR2004
                    else f"Dead weight: four {word}"
                ),
                "value": (
                    "one card is redundant"
                    if count == 3  # noqa: PLR2004
                    else "two cards are redundant, no set possible"
                ),
                "note": note,
            }
        ]
    suit_counts = Counter(card_suit(c) for c in cards)
    heavy_suit = next((s for s, n in suit_counts.items() if n >= 3), None)  # noqa: PLR2004
    if heavy_suit is not None:
        note = describe_suit_redundancy(cards)
        if note is None:  # pragma: no cover -- unreachable, defensive only
            return []
        note = note[:1].upper() + note[1:]
        count = suit_counts[heavy_suit]
        word = _SUIT_CHAR_WORD[heavy_suit]
        return [
            {
                "key": "dead_weight",
                "label": (
                    f"Dead weight: three {word}"
                    if count == 3  # noqa: PLR2004
                    else f"Dead weight: four {word}"
                ),
                "value": (
                    "one card adds no flush value"
                    if count == 3  # noqa: PLR2004
                    else "two cards add no flush value"
                ),
                "note": note,
            }
        ]
    return []


def describe_suit_redundancy(hand: object) -> str | None:
    """Deterministic dead-suit-card note for three-suited/monotone hands.

    A flush only ever uses two hole cards, so a third (or fourth) card of one
    suit is a dead card that also removes the hand's own flush outs from the
    deck. Stated here so Layer 6 reports the real structural flaw instead of
    inventing one (e.g. calling a connected card "a dangler"). ``None`` for
    rainbow / single-suited / double-suited hands -- no suit redundancy to
    report. Mutually exclusive with :func:`describe_card_redundancy` (trips
    occupy three different suits, quads four).
    """
    cards = _normalize(hand)
    suit_counts = Counter(card_suit(c) for c in cards)
    for suit, count in suit_counts.items():
        word = _SUIT_CHAR_WORD[suit]
        if count == 3:  # noqa: PLR2004
            return (
                f"you hold three {word} but a flush only ever uses two hole "
                f"cards, so the third {word[:-1]} is a dead card that also "
                f"removes one of your own flush outs from the deck."
            )
        if count == 4:  # noqa: PLR2004
            return (
                f"all four of your cards are {word} but a flush only ever "
                f"uses two hole cards, so two are dead cards, and they remove "
                f"two of your own flush outs from the deck."
            )
    return None
