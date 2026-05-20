"""Hand class classification (Fact Extractor / Layer 5).

A pure function of hero's hole cards and the community board -- no solver data,
no LLM. Produces the `hand_class` field of the Layer 5 data block: the made-hand
category, any draws, the strength bucket, and a single composite label.

    >>> classify_hand("Ah Ad", "8c 7c 6d")
    {'made_hand': 'overpair', 'draws': [], 'strength_bucket': 'strong', 'label': 'overpair_no_draws'}

See docs/engineering_brief.docx, "Hand Class": 24 made-hand categories, 7 draw
types, 6 strength buckets. The brief states its thresholds are starting points
to tune against the gold explanations; where a judgement call was needed it is
marked `v1:` inline.

Scope: this classifier is postflop -- it requires a 3-5 card board. Preflop
hand labelling (premium_pair_AA, suited_connector_67s, ...) is a separate,
simpler mapping and is intentionally not handled here.
"""
from __future__ import annotations

import itertools
from collections import Counter

from pipeline.cards import card_suit, parse_board, parse_hole, rank_value
from pipeline.fact_extractor.board_texture import classify_board

# The 24 made-hand categories, strongest to weakest (brief, "Made-hand categories").
MADE_HAND_CATEGORIES = (
    "straight_flush", "quads",
    "full_house_set_plus_board", "full_house_trips_plus_pocket",
    "flush_nut", "flush_second_nut", "flush_weak",
    "straight_nut", "straight_weak",
    "set", "trips",
    "two_pair_top", "two_pair_top_and_bottom", "two_pair_mid",
    "overpair", "top_pair_top_kicker", "top_pair_good_kicker", "top_pair_weak_kicker",
    "pocket_pair_below_overcards", "second_pair", "third_pair", "bottom_pair",
    "ace_high", "no_pair_air",
)

# The 7 draw types (brief, "Draw types"). Orthogonal to the made-hand category.
DRAW_TYPES = (
    "flush_draw_nut", "flush_draw_weak",
    "straight_draw_open_ended", "gutshot",
    "combo_draw",
    "backdoor_flush_draw", "backdoor_straight_draw",
)

# The 6 strength buckets (brief, "Strength buckets"; CLAUDE.md spells the third
# 'medium'). Concept-tag rules in Layer 5 look up hands by bucket.
STRENGTH_BUCKETS = ("premium", "strong", "medium", "vulnerable", "marginal", "air")

# made_hand -> strength bucket. v1: the brief's bucket table omits `flush_weak`;
# a made flush still beats every non-flush, so it is placed in `strong`.
_BUCKET = {
    "straight_flush": "premium", "quads": "premium",
    "full_house_set_plus_board": "premium", "full_house_trips_plus_pocket": "premium",
    "flush_nut": "premium", "straight_nut": "premium", "set": "premium",
    "trips": "premium", "two_pair_top": "premium",
    "flush_second_nut": "strong", "flush_weak": "strong", "straight_weak": "strong",
    "two_pair_top_and_bottom": "strong", "overpair": "strong",
    "top_pair_top_kicker": "strong",
    "top_pair_good_kicker": "medium", "two_pair_mid": "medium",
    "pocket_pair_below_overcards": "medium",
    "top_pair_weak_kicker": "vulnerable", "second_pair": "vulnerable",
    "third_pair": "marginal", "bottom_pair": "marginal", "ace_high": "marginal",
    "no_pair_air": "air",
}

# Headline-draw selection for the composite label: most significant first.
_DRAW_PRIORITY = (
    "combo_draw", "flush_draw_nut", "straight_draw_open_ended",
    "flush_draw_weak", "gutshot", "backdoor_flush_draw", "backdoor_straight_draw",
)
# Friendly token each draw contributes to the label (e.g. top_pair_with_flush_draw).
_DRAW_IN_LABEL = {
    "combo_draw": "combo_draw",
    "flush_draw_nut": "flush_draw", "flush_draw_weak": "flush_draw",
    "straight_draw_open_ended": "straight_draw", "gutshot": "gutshot",
    "backdoor_flush_draw": "backdoor_flush_draw",
    "backdoor_straight_draw": "backdoor_straight_draw",
}


# --- low-level poker primitives ----------------------------------------------
def _value_set(values) -> set[int]:
    """Rank values as a set, with the ace also counted low (for wheel straights)."""
    s = set(values)
    if 14 in s:
        s.add(1)
    return s


def _straight_top(values) -> int | None:
    """Highest top-card of a 5-card straight within these rank values; None if none."""
    present = _value_set(values)
    for top in range(14, 4, -1):                  # T=14..5; T=5 is the wheel
        if all((top - i) in present for i in range(5)):
            return top
    return None


def _max_board_straight_top(board_values) -> int | None:
    """Highest straight top-card any holding can make on this board.

    A straight with top T is reachable if the board supplies >= 3 of its five
    ranks (an opponent fills the other two). The highest such T is the nut
    straight -- hero is on the nut straight iff hero's straight reaches it.
    """
    present = _value_set(board_values)
    for top in range(14, 4, -1):
        if len({top - i for i in range(5)} & present) >= 3:
            return top
    return None


def _flush_suit(cards) -> str | None:
    """The suit with 5+ cards (a made flush), or None. At most one with <=7 cards."""
    for suit, n in Counter(card_suit(c) for c in cards).items():
        if n >= 5:
            return suit
    return None


def _flush_strength(hero_suit_values, board_suit_values) -> str:
    """Rank a (made or drawing) flush as 'nut' / 'second' / 'weak'.

    An opponent only outranks hero by holding a higher card of the suit that is
    not already on the shared board. Count those: 0 -> nut, 1 -> second.
    v1: with no card of the suit hero holds (a board flush), call it 'weak'.
    """
    if not hero_suit_values:
        return "weak"
    hero_best = max(hero_suit_values)
    on_board = set(board_suit_values)
    outranking = sum(1 for r in range(hero_best + 1, 15) if r not in on_board)
    return {0: "nut", 1: "second"}.get(outranking, "weak")


# --- made-hand classification ------------------------------------------------
def _two_pair_category(pair_ranks, board_distinct) -> str:
    """Place a two-pair hand among two_pair_top / _top_and_bottom / _mid."""
    top_two = sorted(pair_ranks, reverse=True)[:2]
    try:
        idx = sorted(board_distinct.index(r) for r in top_two)
    except ValueError:
        # v1: a paired rank is not a board rank (pocket pair + a paired board).
        return "two_pair_mid"
    if idx == [0, 1]:
        return "two_pair_top"
    if idx[0] == 0 and idx[1] == len(board_distinct) - 1:
        return "two_pair_top_and_bottom"
    return "two_pair_mid"


def _top_pair_kicker(pair_rank, kicker) -> str:
    """Sub-classify a top pair by kicker (brief: good = T+ but not top)."""
    best_kicker = 14 if pair_rank != 14 else 13
    if kicker == best_kicker:
        return "top_pair_top_kicker"
    if kicker >= 10:
        return "top_pair_good_kicker"
    return "top_pair_weak_kicker"


def _one_pair_category(pair_rank, hole_values, board_values, board_distinct) -> str | None:
    """Classify a one-pair hand. None means the pair sits wholly on the board."""
    if hole_values[0] == hole_values[1]:          # hero holds a pocket pair
        if pair_rank > max(board_values):
            return "overpair"
        return "pocket_pair_below_overcards"
    if pair_rank not in hole_values:              # pair is entirely on the board
        return None
    kicker = hole_values[1] if hole_values[0] == pair_rank else hole_values[0]
    pos = board_distinct.index(pair_rank)
    if pos == 0:
        return _top_pair_kicker(pair_rank, kicker)
    if pos == len(board_distinct) - 1:
        return "bottom_pair"
    if pos == 1:
        return "second_pair"
    return "third_pair"


def _classify_made_hand(hole, board) -> str:
    """The single best made-hand category for hero's 5-7 cards."""
    cards = hole + board
    values = [rank_value(c) for c in cards]
    hole_values = [rank_value(c) for c in hole]
    board_values = [rank_value(c) for c in board]
    rank_count = Counter(values)
    board_count = Counter(board_values)
    board_distinct = sorted(set(board_values), reverse=True)
    flush_suit = _flush_suit(cards)

    # 1. Straight flush.
    if flush_suit is not None:
        suited = [rank_value(c) for c in cards if card_suit(c) == flush_suit]
        if _straight_top(suited) is not None:
            return "straight_flush"

    # 2. Quads.
    if any(n >= 4 for n in rank_count.values()):
        return "quads"

    # 3. Full house. v1: split by whether the trips sit wholly on the board.
    trips = sorted((r for r, n in rank_count.items() if n >= 3), reverse=True)
    pairs_all = sorted((r for r, n in rank_count.items() if n >= 2), reverse=True)
    if trips and (len(trips) >= 2 or any(r != trips[0] for r in pairs_all)):
        if board_count.get(trips[0], 0) >= 3:
            return "full_house_trips_plus_pocket"
        return "full_house_set_plus_board"

    # 4. Flush.
    if flush_suit is not None:
        hero_s = [rank_value(c) for c in hole if card_suit(c) == flush_suit]
        board_s = [rank_value(c) for c in board if card_suit(c) == flush_suit]
        return {"nut": "flush_nut", "second": "flush_second_nut",
                "weak": "flush_weak"}[_flush_strength(hero_s, board_s)]

    # 5. Straight.
    straight_top = _straight_top(values)
    if straight_top is not None:
        nut_top = _max_board_straight_top(board_values)
        if nut_top is None or straight_top >= nut_top:
            return "straight_nut"
        return "straight_weak"

    # 6. Three of a kind -- set (hero's pocket pair) vs trips (board pair).
    if trips:
        trip = trips[0]
        if hole_values[0] == hole_values[1] == trip:
            return "set"
        if board_count.get(trip, 0) == 2:
            return "trips"
        # else: trips sit wholly on the board -- hero contributed nothing; fall
        # through and classify hero by their own cards.

    # 7. Two pair.
    pair_ranks = [r for r, n in rank_count.items() if n >= 2]
    if len(pair_ranks) >= 2:
        return _two_pair_category(pair_ranks, board_distinct)

    # 8. One pair.
    if len(pair_ranks) == 1:
        category = _one_pair_category(pair_ranks[0], hole_values,
                                      board_values, board_distinct)
        if category is not None:
            return category

    # 9. High card.
    return "ace_high" if 14 in hole_values else "no_pair_air"


# --- draw classification -----------------------------------------------------
def _flush_draw(hole, board) -> str | None:
    """A 4-card flush hero has a piece of -> 'flush_draw_nut' / '_weak', else None."""
    for suit, n in Counter(card_suit(c) for c in hole + board).items():
        if n != 4:
            continue
        hero_s = [rank_value(c) for c in hole if card_suit(c) == suit]
        if not hero_s:                            # board four-flush, not hero's draw
            continue
        board_s = [rank_value(c) for c in board if card_suit(c) == suit]
        strength = _flush_strength(hero_s, board_s)
        return "flush_draw_nut" if strength == "nut" else "flush_draw_weak"
    return None


def _straight_draw(hole, board) -> str | None:
    """A one-card straight draw hero is part of -> open-ended / gutshot, else None.

    Open-ended is read off the number of distinct completing ranks: 2+ outs
    ranks behaves as open-ended (true OESD or a double gutshot), exactly 1 is a
    gutshot. Hero must hold a card inside the drawing window.
    """
    values = [rank_value(c) for c in hole + board]
    if _straight_top(values) is not None:
        return None                               # already a made straight
    present = _value_set(values)
    hole_window = _value_set(rank_value(c) for c in hole)
    completing: set[int] = set()
    for top in range(14, 4, -1):
        window = {top - i for i in range(5)}
        if len(window & present) == 4 and window & hole_window:
            completing |= window - present        # the single missing rank
    if not completing:
        return None
    return "straight_draw_open_ended" if len(completing) >= 2 else "gutshot"


def _backdoor_flush(hole, board) -> bool:
    """Hero holds a piece of a 3-card flush (needs running cards)."""
    hero_suits = {card_suit(c) for c in hole}
    for suit, n in Counter(card_suit(c) for c in hole + board).items():
        if n == 3 and suit in hero_suits:
            return True
    return False


def _backdoor_straight(hole, board) -> bool:
    """Hero holds a piece of 3 distinct ranks inside a 5-rank span."""
    values = sorted(_value_set(rank_value(c) for c in hole + board))
    hole_window = _value_set(rank_value(c) for c in hole)
    for combo in itertools.combinations(values, 3):
        if combo[-1] - combo[0] <= 4 and set(combo) & hole_window:
            return True
    return False


def _classify_draws(hole, board, made_hand) -> list[str]:
    """Every draw hero holds. Draws are orthogonal to the made-hand category.

    The list carries each atomic draw; when a flush draw and a straight draw
    co-occur, 'combo_draw' is also added as a summary tag (so both filtering
    keys work). Backdoor draws are flop-only and reported only when no stronger
    draw of that kind exists.
    """
    if len(board) >= 5:                           # river: the hand is complete
        return []

    has_made_flush = made_hand in ("flush_nut", "flush_second_nut",
                                   "flush_weak", "straight_flush")
    has_made_straight = made_hand in ("straight_nut", "straight_weak", "straight_flush")
    flush = None if has_made_flush else _flush_draw(hole, board)
    straight = None if has_made_straight else _straight_draw(hole, board)

    draws: list[str] = []
    if flush and straight:
        draws.append("combo_draw")
    if flush:
        draws.append(flush)
    if straight:
        draws.append(straight)

    if len(board) == 3:                           # backdoor draws exist only on the flop
        if not has_made_flush and not flush and _backdoor_flush(hole, board):
            draws.append("backdoor_flush_draw")
        if not has_made_straight and not straight and _backdoor_straight(hole, board):
            draws.append("backdoor_straight_draw")
    return draws


# --- strength bucket & label -------------------------------------------------
def _strength_bucket(made_hand, hole, board) -> str:
    """Map the made hand to its strength bucket, with one contextual refinement."""
    bucket = _BUCKET[made_hand]
    # v1: the brief routes a "weak overpair on a wet board" into `vulnerable`.
    # Weak overpair := pair of jacks or lower; wet := composite wet/very_wet/dynamic.
    if made_hand == "overpair":
        pair_rank = rank_value(hole[0])
        wet = classify_board(board)["composite"] in ("wet", "very_wet", "dynamic")
        if pair_rank <= 11 and wet:
            return "vulnerable"
    return bucket


def _label(made_hand, draws) -> str:
    """The composite label: '<made_hand>_no_draws' or '<made_hand>_with_<draw>'.

    Backdoor (runner-runner) draws stay in the `draws` list but never headline
    the label -- they are too thin to be the story, and the brief's label
    examples only ever feature primary draws.
    """
    primary = [d for d in draws
               if d not in ("backdoor_flush_draw", "backdoor_straight_draw")]
    if not primary:
        return f"{made_hand}_no_draws"
    headline = min(primary, key=_DRAW_PRIORITY.index)
    return f"{made_hand}_with_{_DRAW_IN_LABEL[headline]}"


def classify_hand(hole, board) -> dict:
    """Classify hero's hand on a board.

    `hole` is hero's two cards (list, 'AhKs', or 'Ah Ks'); `board` is 3-5
    community cards (list or space-separated string). Returns the `hand_class`
    dict: made_hand, draws, strength_bucket, label.
    """
    hole_cards = parse_hole(hole)
    board_cards = parse_board(board)
    clash = set(hole_cards) & set(board_cards)
    if clash:
        raise ValueError(f"hole cards also on the board: {sorted(clash)}")

    made_hand = _classify_made_hand(hole_cards, board_cards)
    draws = _classify_draws(hole_cards, board_cards, made_hand)
    return {
        "made_hand": made_hand,
        "draws": draws,
        "strength_bucket": _strength_bucket(made_hand, hole_cards, board_cards),
        "label": _label(made_hand, draws),
    }
