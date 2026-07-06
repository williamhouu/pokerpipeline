"""Preflop concept-tag library.

Each tag is a pure Python boolean function ``def tag_name(facts: PreflopFacts)
-> bool``. The aggregator :func:`compute_concept_tags` collects the firing
tags for a spot.

Per the brief's "the LLM never thinks about poker" principle, every tag is
deterministic Python -- never an LLM call, never a heuristic that requires
strategic judgement. The LLM later reads the firing tags as part of the
SOLVER DATA block and uses them to frame the explanation; Layer 7 validates
the explanation mentions the relevant tags.

Categories (38 tags total):

  Position context           (5)  early/middle/late/SB/BB
  Decision context           (7)  open / facing single raise / facing 3bet /
                                  facing 4bet+ / squeeze / BvB / multiway
  Stack depth                (3)  short / standard / deep
  Hand strength              (8)  premium pair / medium pair / small pair /
                                  premium unpaired / suited broadway /
                                  suited connector / suited ace / unconnected
  Strategy shape             (5)  mixed / near-pure / aggressive / passive / fold
  Equity context             (4)  dominant / favorite / coinflip / dominated
  Blockers                   (3)  ace / king / blocks villain top value
  Range dynamics             (3)  hero advantage / villain advantage / equal

Adding new tags: define a function in this module, then add it to the
:data:`_TAG_REGISTRY` tuple. The aggregator picks it up automatically. Tag
function names are the canonical tag names (lowercase, underscored); the
aggregator reads ``fn.__name__`` to label them.
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import PreflopActionType

# Type alias for a tag function. Each one takes a PreflopFacts and returns
# True iff the concept applies to this spot.
TagFn = Callable[[PreflopFacts], bool]


# --- helpers reused across tag functions ------------------------------------
def _raise_count(facts: PreflopFacts) -> int:
    """Number of raises (incl. all-ins) in the prior action history."""
    return sum(
        1
        for a in facts.spot.node.history_before
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )


def _calls_after_last_raise(facts: PreflopFacts) -> int:
    """Number of calls between the latest raise and hero's decision."""
    count = 0
    seen_raise = False
    for a in facts.spot.node.history_before:
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            seen_raise = True
            count = 0  # reset on a new raise
        elif seen_raise and a.action_type is PreflopActionType.CALL:
            count += 1
    return count


def _non_fold_actor_count(facts: PreflopFacts) -> int:
    """Players still in the pot at hero's decision point (incl. hero).

    A position counts only if its LAST action in the history is a non-fold
    -- a player who entered the pot and then folded (e.g. opened, then
    folded to a squeeze) is out of the hand, even though their dead money
    remains; that was the bug behind multiway_pot firing on heads-up
    "HJ opens, BTN calls, SB squeezes, HJ folds, BTN decides" spots.
    Hero always counts; set semantics dedupe hero's own prior action
    (e.g. an open before later facing a 3-bet) against the explicit add.
    Mirrors ``format_writer._active_villain_actions``; keep in sync with
    its node-level twin :func:`pipeline.preflop.batch.active_player_count`.

    Examples:
        * HJ opens, CO 3-bets, HJ decides -> {HJ, CO} = 2 (heads-up).
        * BTN opens, BB calls, hero is SB to act -> {BTN, BB, SB} = 3
          (squeeze opportunity, truly multiway).
        * UTG to act first (open) -> {UTG} = 1 (hero alone).
        * HJ opens, BTN calls, SB squeezes, HJ folds, BTN decides ->
          {SB, BTN} = 2 (heads-up again -- HJ is out).
    """
    last_action: dict[str, PreflopActionType] = {}
    for a in facts.spot.node.history_before:
        last_action[a.position] = a.action_type
    active_positions = {
        position
        for position, action_type in last_action.items()
        if action_type is not PreflopActionType.FOLD
    }
    active_positions.add(facts.spot.node.actor)
    return len(active_positions)


def _hand_class(facts: PreflopFacts) -> str:
    return facts.spot.hero_hand_class


def _hero_cards(facts: PreflopFacts) -> tuple[str, str]:
    """Hero's two cards as (first-rank, second-rank) -- e.g. ('A', 'K')."""
    combo = facts.spot.hero_card_combo
    return (combo[0], combo[2])


# --- Position context (5) ----------------------------------------------------
_EARLY_POSITIONS = frozenset({"UTG", "UTG+1", "UTG+2"})
_MIDDLE_POSITIONS = frozenset({"LJ", "HJ"})
_LATE_POSITIONS = frozenset({"CO", "BTN"})


def early_position(facts: PreflopFacts) -> bool:
    """Hero is in UTG / UTG+1 / UTG+2."""
    return facts.spot.node.actor in _EARLY_POSITIONS


def middle_position(facts: PreflopFacts) -> bool:
    """Hero is in the Lojack or Hijack."""
    return facts.spot.node.actor in _MIDDLE_POSITIONS


def late_position(facts: PreflopFacts) -> bool:
    """Hero is in the Cutoff or on the Button."""
    return facts.spot.node.actor in _LATE_POSITIONS


def small_blind(facts: PreflopFacts) -> bool:
    """Hero is in the Small Blind."""
    return facts.spot.node.actor == "SB"


def big_blind(facts: PreflopFacts) -> bool:
    """Hero is in the Big Blind."""
    return facts.spot.node.actor == "BB"


# --- Decision context (7) ----------------------------------------------------
def open_decision(facts: PreflopFacts) -> bool:
    """No prior raise -- hero deciding whether to open the action."""
    return _raise_count(facts) == 0


def facing_single_raise(facts: PreflopFacts) -> bool:
    """Exactly one prior raise AND no caller between raiser and hero. Hero
    is responding to an open."""
    return _raise_count(facts) == 1 and _calls_after_last_raise(facts) == 0


def facing_3bet(facts: PreflopFacts) -> bool:
    """Exactly two prior raises -- hero opened (or 3-bet) and got re-raised."""
    return _raise_count(facts) == 2


def facing_4bet_plus(facts: PreflopFacts) -> bool:
    """Three or more prior raises -- hero is in a 4-bet+ spot."""
    return _raise_count(facts) >= 3


def squeeze_opportunity(facts: PreflopFacts) -> bool:
    """One prior raise AND at least one caller between raiser and hero.
    Hero raising here is a squeeze."""
    return _raise_count(facts) == 1 and _calls_after_last_raise(facts) >= 1


def bvb_spot(facts: PreflopFacts) -> bool:
    """Only SB and BB still in the hand at the decision point. Hero is in
    a blind-vs-blind situation -- either SB opens or BB defends an SB
    open."""
    history = facts.spot.node.history_before
    # All non-blind positions must be folded for it to be BvB. Deliberately
    # ANY non-fold action (not last-action like _non_fold_actor_count): a
    # non-blind who entered the pot and later folded still shaped the ranges
    # and left dead money, so it's not a pure blind battle.
    non_blinds_in_history = [a for a in history if a.position not in ("SB", "BB")]
    if any(a.action_type is not PreflopActionType.FOLD for a in non_blinds_in_history):
        return False
    # Hero must be SB or BB.
    return facts.spot.node.actor in ("SB", "BB")


def multiway_pot(facts: PreflopFacts) -> bool:
    """3+ players still in the pot at hero's decision point."""
    return _non_fold_actor_count(facts) >= 3


# --- Stack depth (3) ---------------------------------------------------------
# Stack depth signals come from the pack metadata, not the facts directly.
# Ryan's pack is fixed at 100bb; the 9-max future pack might have different
# depths. The tag reads the pack's stack_depth_bb via the node's
# pack-derived metadata. For now we hard-code "standard" for the Ryan pack
# since we don't have a stack-depth field on PreflopFacts -- a TODO when
# multi-stack packs arrive.
def short_stack(facts: PreflopFacts) -> bool:
    """Effective stack < 40bb. v1: always False for the Ryan pack (100bb).
    TODO(Phase B): read stack depth from pack metadata when multi-stack
    packs land."""
    return False


def standard_stack(facts: PreflopFacts) -> bool:
    """Effective stack in 40-150bb. v1: always True for the Ryan pack
    (which is 100bb)."""
    return True


def deep_stack(facts: PreflopFacts) -> bool:
    """Effective stack > 150bb. v1: always False for the Ryan pack."""
    return False


# --- Hand strength (8) -------------------------------------------------------
_PREMIUM_PAIRS = frozenset({"AA", "KK", "QQ"})
_MEDIUM_PAIRS = frozenset({"JJ", "TT", "99"})
_SMALL_PAIRS = frozenset({"88", "77", "66", "55", "44", "33", "22"})
_PREMIUM_UNPAIRED = frozenset({"AKs", "AKo", "AQs", "AQo"})
_SUITED_BROADWAY = frozenset({"KQs", "KJs", "KTs", "QJs", "QTs", "JTs"})


def premium_pair(facts: PreflopFacts) -> bool:
    """Hero holds AA, KK, or QQ -- top-tier pocket pairs."""
    return _hand_class(facts) in _PREMIUM_PAIRS


def medium_pair(facts: PreflopFacts) -> bool:
    """Hero holds JJ, TT, or 99 -- pairs that play very differently
    against value (raise) vs. capped (call/set-mine) ranges."""
    return _hand_class(facts) in _MEDIUM_PAIRS


def small_pair(facts: PreflopFacts) -> bool:
    """Hero holds 88-22 -- set-mining range, rarely overpair preflop."""
    return _hand_class(facts) in _SMALL_PAIRS


def premium_unpaired(facts: PreflopFacts) -> bool:
    """AK or AQ, suited or offsuit -- the elite unpaired hands."""
    return _hand_class(facts) in _PREMIUM_UNPAIRED


def suited_broadway(facts: PreflopFacts) -> bool:
    """Suited two-broadway combinations: KQs, KJs, KTs, QJs, QTs, JTs."""
    return _hand_class(facts) in _SUITED_BROADWAY


def suited_connector(facts: PreflopFacts) -> bool:
    """Consecutive same-suit hands (98s, 87s, 76s, etc.). Excludes the
    broadway-down suited connectors (JTs, T9s) -- those have specific
    classifications above. The smallest in this tag is 54s; broadest is 98s."""
    hc = _hand_class(facts)
    if len(hc) != 3 or hc[2] != "s":
        return False
    # Pull the two rank chars and convert to rank index.
    _RANK_ORDER = "23456789TJQKA"
    if hc[0] not in _RANK_ORDER or hc[1] not in _RANK_ORDER:
        return False
    high = _RANK_ORDER.index(hc[0])
    low = _RANK_ORDER.index(hc[1])
    # Consecutive: high - low == 1. Cap at 98s (so JTs etc. don't double-tag
    # with suited_broadway).
    if high - low != 1:
        return False
    return high <= _RANK_ORDER.index("9")  # 98s and below


def suited_ace(facts: PreflopFacts) -> bool:
    """Axs hands that aren't already premium_unpaired (AKs, AQs).
    Includes A2s-AJs."""
    hc = _hand_class(facts)
    if len(hc) != 3 or hc[2] != "s" or hc[0] != "A":
        return False
    return hc not in _PREMIUM_UNPAIRED  # exclude AKs / AQs


def unconnected_offsuit(facts: PreflopFacts) -> bool:
    """Offsuit hand with rank gap >= 2 and not a premium unpaired or
    broadway-class hand -- the genuine "weak hand" tag."""
    hc = _hand_class(facts)
    if len(hc) != 3 or hc[2] != "o":
        return False
    if hc in _PREMIUM_UNPAIRED:
        return False
    _RANK_ORDER = "23456789TJQKA"
    if hc[0] not in _RANK_ORDER or hc[1] not in _RANK_ORDER:
        return False
    high = _RANK_ORDER.index(hc[0])
    low = _RANK_ORDER.index(hc[1])
    return (high - low) >= 2


# --- Strategy shape (5) ------------------------------------------------------
def mixed_strategy(facts: PreflopFacts) -> bool:
    """Dominant action's conditional frequency is 55-95% -- a real mix."""
    return 0.55 <= facts.spot.dominant_frequency < 0.95


def near_pure_strategy(facts: PreflopFacts) -> bool:
    """Dominant action's conditional frequency is >= 95% -- effectively pure."""
    return facts.spot.dominant_frequency >= 0.95


_AGGRESSIVE_PIO_VERBS = ("Raise", "AllIn")


def dominant_is_aggressive(facts: PreflopFacts) -> bool:
    """Hero's dominant action is a Raise or All-in (i.e., adds chips,
    raises the bet level)."""
    label = facts.spot.dominant_action
    return any(label.startswith(verb) for verb in _AGGRESSIVE_PIO_VERBS)


def dominant_is_passive(facts: PreflopFacts) -> bool:
    """Hero's dominant action is a Call (no aggression, no fold)."""
    return facts.spot.dominant_action == "Call"


def dominant_is_fold(facts: PreflopFacts) -> bool:
    """Hero's dominant action is Fold."""
    return facts.spot.dominant_action == "Fold"


# --- Equity context (4) ------------------------------------------------------
def equity_dominant(facts: PreflopFacts) -> bool:
    """Hero's hand has > 70% equity vs villain's range."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and eq > 0.70


def equity_favorite(facts: PreflopFacts) -> bool:
    """Hero's hand has 55-70% equity vs villain's range."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and 0.55 <= eq <= 0.70


def coinflip(facts: PreflopFacts) -> bool:
    """Hero's equity vs villain is 45-55% -- close to a coin flip."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and 0.45 <= eq < 0.55


def dominated(facts: PreflopFacts) -> bool:
    """Hero's equity vs villain is < 35% -- hero is dominated."""
    eq = facts.hero_equity_vs_villain
    return eq is not None and eq < 0.35


# A villain range this tight (pct of dealt hands) or tighter is
# dominator-heavy -- packed with big aces and pairs that outkick a weak ace --
# which is what turns a weak ace/king into a reverse-implied-odds hand rather
# than a speculative one. Against wider ranges those same hands have POSITIVE
# implied odds, so the tag must stay off there.
_RIO_TIGHT_RANGE_MAX_PCT = 20.0
# Facing a RE-RAISE (3-bet or later) the width proxy breaks down: a 3-bet
# range is value-tilted toward exactly the hands that dominate a weak ace/king
# no matter how wide it is (a 25% blind-vs-blind 3-bet range is still packed
# with better Ax/Kx). So re-raise spots get a wider cutoff, sized to cover
# real 3-bet ranges (BvB runs ~20-28%) while still excluding degenerate
# super-wide lines. July 2026, from the KTo-SB-vs-BB-3bet QC miss.
_RIO_RERAISE_RANGE_MAX_PCT = 30.0


def _facing_all_in(facts: PreflopFacts) -> bool:
    """True when the action hero faces (the latest aggression) is an all-in."""
    for a in reversed(facts.spot.node.history_before):
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN):
            return a.action_type is PreflopActionType.ALL_IN
    return False


def reverse_implied_odds(facts: PreflopFacts) -> bool:
    """Dominated-pair-prone hand facing a dominator-heavy range.

    Fires for a weak ace (A2s-AJs, or a weak A-high offsuit) or a weak K-high
    offsuit when villain's range is dominator-heavy: these flop second-best
    top pairs that pay off bigger hands -- the textbook reverse-implied-odds
    spot. The ace/king makes a top pair that is routinely outkicked, so you
    win small and lose big. Two qualifying range shapes:

      * a TIGHT range (<= 20% of hands, e.g. A4s vs a tight UTG open) --
        against a WIDE open the same hand is a fine speculative holding with
        POSITIVE implied odds, so width gates the tag; and
      * a RE-RAISE range (hero faces a 3-bet or later) up to 30% wide --
        re-raise ranges are value-tilted toward dominators at any realistic
        width (the KTo SB-vs-BB-3bet fold). Gated to spots whose dominant
        action IS a fold so this wider path can never hand the LLM an RIO
        frame against a correct call/raise (mirrors the skill layer's
        fold-archetype gate; the ``dominant_is_fold`` tag sets precedent
        for decision-aware tags).

    Never fires facing an ALL-IN: with no postflop play left there are no
    implied odds in either direction -- that fold is pure pot odds.

    Mutually exclusive with positive implied odds BY CONSTRUCTION: pocket
    pairs (set value) and suited connectors (nut straights/flushes) are
    neither weak-ace nor weak-K-offsuit hands, so they never fire here; and it
    never fires on the ``call_for_implied_odds`` archetype.
    """
    if facts.archetype == "call_for_implied_odds":
        return False  # a hand can't be reverse- AND positive-implied-odds
    if _facing_all_in(facts):
        return False  # no postflop play left -> no reverse implied odds
    stats = facts.villain_stats
    width = getattr(stats, "pct_of_dealt_hands", None) if stats else None
    if width is None:
        return False  # no villain to be dominated by
    facing_reraise = _raise_count(facts) >= 2  # noqa: PLR2004
    if facing_reraise:
        dominant_fold = facts.spot.dominant_action.startswith("Fold")
        if width > _RIO_RERAISE_RANGE_MAX_PCT or not dominant_fold:
            return False
    elif width > _RIO_TIGHT_RANGE_MAX_PCT:
        return False  # vs a single raise, wide range = positive implied odds
    high = _hand_class(facts)[0]
    weak_ace = suited_ace(facts) or (unconnected_offsuit(facts) and high == "A")
    weak_king = unconnected_offsuit(facts) and high == "K"
    return weak_ace or weak_king


# --- Blockers (3) ------------------------------------------------------------
def ace_blocker(facts: PreflopFacts) -> bool:
    """Hero holds at least one Ace -- removes AA / AK / etc. from villain's
    value range. The classic 4-bet bluff signal."""
    ranks = _hero_cards(facts)
    return "A" in ranks


def king_blocker(facts: PreflopFacts) -> bool:
    """Hero holds at least one King -- secondary blocker for value combos
    like KK / AK / KQ."""
    ranks = _hero_cards(facts)
    return "K" in ranks


def blocks_villain_top_value(facts: PreflopFacts) -> bool:
    """Hero's specific cards remove at least one combo from villain's top
    value range. Reads :attr:`PreflopFacts.blockers` -- a dict computed by
    fact_extractor of {hand_class: combos_blocked}. Tag fires if the dict
    has any non-zero entry."""
    return bool(facts.blockers) and any(v > 0 for v in facts.blockers.values())


# --- Range dynamics (3) ------------------------------------------------------
def hero_range_advantage(facts: PreflopFacts) -> bool:
    """Hero's range equity vs villain's range >= 53% -- meaningful edge."""
    re_ = facts.hero_range_equity_vs_villain
    return re_ is not None and re_ >= 0.53


def villain_range_advantage(facts: PreflopFacts) -> bool:
    """Villain's range has the edge: hero_range_equity_vs_villain <= 47%."""
    re_ = facts.hero_range_equity_vs_villain
    return re_ is not None and re_ <= 0.47


def roughly_equal_ranges(facts: PreflopFacts) -> bool:
    """Range equity is within 47-53% -- no meaningful range advantage
    either way."""
    re_ = facts.hero_range_equity_vs_villain
    return re_ is not None and 0.47 < re_ < 0.53


# --- registry + aggregator ---------------------------------------------------
# Order matters for the CSV column readability: tags emitted in this order
# in the output. Position first (always relevant), then decision context,
# then strategy shape, then hand-strength, then equity, blockers, range
# dynamics, stack depth last (currently always "standard" so not
# information-rich).
_TAG_REGISTRY: tuple[TagFn, ...] = (
    # Position
    early_position,
    middle_position,
    late_position,
    small_blind,
    big_blind,
    # Decision context
    open_decision,
    facing_single_raise,
    facing_3bet,
    facing_4bet_plus,
    squeeze_opportunity,
    bvb_spot,
    multiway_pot,
    # Strategy shape
    mixed_strategy,
    near_pure_strategy,
    dominant_is_aggressive,
    dominant_is_passive,
    dominant_is_fold,
    # Hand strength
    premium_pair,
    medium_pair,
    small_pair,
    premium_unpaired,
    suited_broadway,
    suited_connector,
    suited_ace,
    unconnected_offsuit,
    # Equity
    equity_dominant,
    equity_favorite,
    coinflip,
    dominated,
    reverse_implied_odds,
    # Blockers
    ace_blocker,
    king_blocker,
    blocks_villain_top_value,
    # Range dynamics
    hero_range_advantage,
    villain_range_advantage,
    roughly_equal_ranges,
    # Stack depth (last because Ryan-pack v1 is always 100bb)
    short_stack,
    standard_stack,
    deep_stack,
)


def compute_concept_tags(facts: PreflopFacts) -> list[str]:
    """Return the firing tag names for this spot, in registry order.

    Each tag in :data:`_TAG_REGISTRY` is called once; the function name
    is used as the tag's canonical label. Tag functions never raise --
    they must be pure boolean queries. The output is the subset of tag
    names that returned True, deterministic for any given PreflopFacts.

    Used by:
      - Layer 6: tags appear in the SOLVER DATA block so the LLM has
        strategic context when writing the explanation.
      - Layer 8: rendered into the ``concept_tags`` CSV column.
      - Phase 3: mapped to user-facing skills in the app.
    """
    return [fn.__name__ for fn in _TAG_REGISTRY if fn(facts)]


__all__ = [
    "TagFn",
    "compute_concept_tags",
    # Position
    "early_position",
    "middle_position",
    "late_position",
    "small_blind",
    "big_blind",
    # Decision context
    "open_decision",
    "facing_single_raise",
    "facing_3bet",
    "facing_4bet_plus",
    "squeeze_opportunity",
    "bvb_spot",
    "multiway_pot",
    # Stack depth
    "short_stack",
    "standard_stack",
    "deep_stack",
    # Hand strength
    "premium_pair",
    "medium_pair",
    "small_pair",
    "premium_unpaired",
    "suited_broadway",
    "suited_connector",
    "suited_ace",
    "unconnected_offsuit",
    # Strategy shape
    "mixed_strategy",
    "near_pure_strategy",
    "dominant_is_aggressive",
    "dominant_is_passive",
    "dominant_is_fold",
    # Equity
    "equity_dominant",
    "equity_favorite",
    "coinflip",
    "dominated",
    "reverse_implied_odds",
    # Blockers
    "ace_blocker",
    "king_blocker",
    "blocks_villain_top_value",
    # Range dynamics
    "hero_range_advantage",
    "villain_range_advantage",
    "roughly_equal_ranges",
]
