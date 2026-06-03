"""PLO concept-tag library -- the hand-structure tags.

Each tag is a pure boolean ``def tag(hand: PloHandClass) -> bool``, mirroring
the NLHE tagger (``pipeline.preflop.concept_tags``). Crucially, every tag
reads a *field* off the already-classified :class:`PloHandClass` rather than
re-deriving structure from the cards -- the NLHE tagger's bugs came from each
tag re-parsing the ``hand_class`` string, so here the hand model is the
single source of truth and the tags are a thin, named boolean layer over it.

Scope: this module holds the **hand-only** concept tags -- the ones that are
PLO-specific and need no preflop pack (suitedness, pairing, connectedness /
wrap, nut potential, strength). They are buildable and exhaustively testable
now, before the pack lands.

The rest of the full PLO concept-tag catalog is **game-agnostic** and waits
on the PloFacts / pack layer (it mirrors NLHE almost exactly):

  * Position context (5): early / middle / late / SB / BB        [from node]
  * Decision context (7): open / facing_single_raise / facing_3bet /
    facing_4bet_plus / squeeze / bvb / multiway                  [from history]
  * Strategy shape (5): mixed / near_pure / aggressive / passive / fold
                                                                 [from solver]
  * Equity context (4): equity_dominant / favorite / coinflip / dominated
                                              [4-card equity vs villain range]
  * Range dynamics (3): hero / villain / equal range advantage  [range equity]
  * Stack depth (3): short / standard / deep                    [pack metadata]
  * Villain-relative blockers (2): blocks_villain_nut_flush / _value [facts]

When PloFacts exists, ``compute_plo_concept_tags(facts)`` will call
:func:`compute_plo_hand_tags(facts.hand_class)` here and add those.

Tag groups and their invariants (verified exhaustively over all 270,725
combos in ``scripts/plo_tag_simulation.py`` and sampled in the test suite):
exactly one SUIT tag, one PAIR tag, and one CONNECTEDNESS tag fire per hand;
the rest are additive feature flags.
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.plo.hand_model import PloHandClass

TagFn = Callable[[PloHandClass], bool]

_BROADWAY_HEAVY_MIN = 3
_LOW_CARD_CEILING = 8
_ACE = 14
_KING = 13


# --- Suit structure (partition: exactly one fires) ------------------------
def double_suited(h: PloHandClass) -> bool:
    """Two suits, two cards each -- two flush draws, the best suit shape."""
    return h.suit_pattern == "double_suited"


def single_suited(h: PloHandClass) -> bool:
    """Exactly two cards of one suit -- a single flush draw."""
    return h.suit_pattern == "single_suited"


def rainbow(h: PloHandClass) -> bool:
    """Four distinct suits -- no flush potential at all."""
    return h.suit_pattern == "rainbow"


def three_suited(h: PloHandClass) -> bool:
    """Three of one suit -- the third suited card is partly dead (weak)."""
    return h.suit_pattern == "three_suited"


def monotone(h: PloHandClass) -> bool:
    """All four one suit -- three cards dead for flushes (very weak)."""
    return h.suit_pattern == "monotone"


# --- Pair structure (partition: exactly one fires) ------------------------
def unpaired_hand(h: PloHandClass) -> bool:
    """No pair -- all four ranks distinct (the rundown / wrap family)."""
    return h.pair_pattern == "unpaired"


def single_pair(h: PloHandClass) -> bool:
    """Exactly one pocket pair plus two other ranks."""
    return h.pair_pattern == "one_pair"


def double_paired(h: PloHandClass) -> bool:
    """Two pocket pairs (e.g. KKQQ) -- two set-mining ranks + redraws."""
    return h.pair_pattern == "two_pair"


def trips_in_hand(h: PloHandClass) -> bool:
    """Three of a kind in hand -- two of the three are dead (bad)."""
    return h.pair_pattern == "trips"


def quads_in_hand(h: PloHandClass) -> bool:
    """Four of a kind in hand -- three cards dead (the worst shape)."""
    return h.pair_pattern == "quads"


# --- Pair quality (additive flags) ----------------------------------------
def pocket_aces(h: PloHandClass) -> bool:
    """Hand contains a pair of aces (AAxx) -- the premium PLO category."""
    return _ACE in h.pair_ranks


def pocket_kings(h: PloHandClass) -> bool:
    """Hand contains a pair of kings (KKxx)."""
    return _KING in h.pair_ranks


# --- Connectedness (partition: exactly one fires) -------------------------
def rundown(h: PloHandClass) -> bool:
    """Four consecutive ranks (e.g. JT98, or A234 with the ace low)."""
    return h.connectedness == "rundown"


def one_gap_rundown(h: PloHandClass) -> bool:
    """Four ranks spanning one gap (e.g. JT97)."""
    return h.connectedness == "one_gapper"


def two_gap_rundown(h: PloHandClass) -> bool:
    """Four ranks spanning two gaps (e.g. JT86)."""
    return h.connectedness == "two_gapper"


def connected_hand(h: PloHandClass) -> bool:
    """Loosely connected (or a paired hand with no dangler)."""
    return h.connectedness == "connected"


def disconnected_hand(h: PloHandClass) -> bool:
    """At least one card is too far from the others to make a straight."""
    return h.connectedness == "disconnected"


# --- Connectedness features (additive flags) ------------------------------
def wrap_potential(h: PloHandClass) -> bool:
    """Unpaired and tightly connected -- flops big wrap draws."""
    return h.wrap_potential


def has_dangler(h: PloHandClass) -> bool:
    """One card is disconnected from the rest -- a partly wasted card."""
    return h.has_dangler


def broadway_rundown(h: PloHandClass) -> bool:
    """A rundown made entirely of broadway cards (AKQJ, KQJT) -- nut-heavy."""
    return h.connectedness == "rundown" and h.broadway_count == 4


# --- Nut / high-card quality (additive flags) -----------------------------
def nut_flush_potential(h: PloHandClass) -> bool:
    """An ace sharing a suit with another card -- can make the nut flush."""
    return h.suited_ace


def bare_ace(h: PloHandClass) -> bool:
    """An ace with no suit backup -- a blocker, not a nut-flush maker."""
    return h.has_ace and not h.suited_ace


def all_broadway(h: PloHandClass) -> bool:
    """All four cards are broadway (T-A) -- makes the nut straights."""
    return h.broadway_count == 4


def broadway_heavy(h: PloHandClass) -> bool:
    """Three or more broadway cards -- mostly high, nutted holdings."""
    return h.broadway_count >= _BROADWAY_HEAVY_MIN


def low_cards(h: PloHandClass) -> bool:
    """Every card is an eight or lower -- non-nut everything."""
    return max(h.distinct_ranks) <= _LOW_CARD_CEILING


# --- Strength summary (additive flags) ------------------------------------
def premium_hand(h: PloHandClass) -> bool:
    """The hand model's top strength bucket (tunable heuristic)."""
    return h.strength == "premium"


def trash_hand(h: PloHandClass) -> bool:
    """The hand model's bottom strength bucket (tunable heuristic)."""
    return h.strength == "trash"


# --- registry + aggregator ------------------------------------------------
# Order = CSV readability: suit, pair, pair-quality, connectedness,
# connect-features, nut/high-card, strength.
SUIT_TAGS: tuple[TagFn, ...] = (
    double_suited, single_suited, rainbow, three_suited, monotone,
)
PAIR_TAGS: tuple[TagFn, ...] = (
    unpaired_hand, single_pair, double_paired, trips_in_hand, quads_in_hand,
)
CONNECTEDNESS_TAGS: tuple[TagFn, ...] = (
    rundown, one_gap_rundown, two_gap_rundown, connected_hand, disconnected_hand,
)

_HAND_TAG_REGISTRY: tuple[TagFn, ...] = (
    *SUIT_TAGS,
    *PAIR_TAGS,
    pocket_aces, pocket_kings,
    *CONNECTEDNESS_TAGS,
    wrap_potential, has_dangler, broadway_rundown,
    nut_flush_potential, bare_ace, all_broadway, broadway_heavy, low_cards,
    premium_hand, trash_hand,
)


def compute_plo_hand_tags(hand: PloHandClass) -> list[str]:
    """Firing hand-structure tag names for a classified PLO hand, in order.

    Each tag in :data:`_HAND_TAG_REGISTRY` is called once; its function name
    is the canonical tag label. The full PLO concept-tag set (with position /
    decision / equity / range tags) joins these once the PloFacts layer lands.
    """
    return [fn.__name__ for fn in _HAND_TAG_REGISTRY if fn(hand)]


__all__ = [
    "CONNECTEDNESS_TAGS",
    "PAIR_TAGS",
    "SUIT_TAGS",
    "TagFn",
    "all_broadway",
    "bare_ace",
    "broadway_heavy",
    "broadway_rundown",
    "compute_plo_hand_tags",
    "connected_hand",
    "disconnected_hand",
    "double_paired",
    "double_suited",
    "low_cards",
    "monotone",
    "nut_flush_potential",
    "one_gap_rundown",
    "pocket_aces",
    "pocket_kings",
    "premium_hand",
    "quads_in_hand",
    "rainbow",
    "rundown",
    "single_pair",
    "single_suited",
    "three_suited",
    "trash_hand",
    "trips_in_hand",
    "two_gap_rundown",
    "unpaired_hand",
    "wrap_potential",
]
