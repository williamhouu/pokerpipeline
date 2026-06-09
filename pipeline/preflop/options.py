"""Deterministic option selection for preflop questions.

Per the brief's "the LLM never thinks about poker" principle, the option
strings and ``correct_answer`` for a preflop question can be computed
entirely from solver data -- no LLM judgement required. This module
implements that computation.

Three styles, matching the admin panel's "Answer option style" radio:

* :func:`build_options_basic`  -- bare action labels (``"Fold"``,
  ``"Call"``, ``"Raise 60%"``). Used when the strategy is dominant
  enough that frequency framing would be misleading.
* :func:`build_options_gto`    -- Always/Mostly template
  (``"Always Call"``, ``"Mostly Call"``, ``"Mostly Fold"``,
  ``"Always Fold"``). Used when the strategy is mixed and the
  player needs to recognise both actions are part of the right answer.
* :func:`build_options_auto`   -- picks one of the above based on the
  dominant-action frequency (binary if >= 80%, gto otherwise).

The public entry :func:`build_options` is a thin dispatcher. Each
builder returns a ``(options, correct_answer)`` tuple where
``options`` is 1-4 strings and ``correct_answer`` is one of them
exactly.

The ``correct_answer`` for GTO style uses the deterministic prefix
mapping shared with the postflop path
(:func:`pipeline.explanation_generator.frequency_to_verb_prefix`):

  * dominant freq >= 95% -> ``"Always <action>"``
  * 5% <= freq < 95%     -> ``"Mostly <action>"``
  * freq < 5%            -> action is essentially not played

For 3+ action mixes, GTO style emits composite labels of the form
``"Mostly call, sometimes raise"`` instead of standalone "Sometimes X"
labels (which were banned per Ryan's Apr-2026 review as ambiguous).

This module replaces the LLM's responsibility for picking options.
The LLM in Layer 6 now writes only the ``answer_explanation`` prose;
the four option strings and the correct answer are computed here and
passed in.
"""

from __future__ import annotations

from pipeline.explanation_generator import frequency_to_verb_prefix
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.grammars.types import PreflopActionType

# Frequency threshold above which a non-played action is dropped from the
# option set. Below 5% the action is essentially noise (Pio occasionally
# mixes in 0.1% raise frequencies that aren't strategic content).
_MIN_MEANINGFUL_FREQ = 0.05

# Raise-level verb table for canonicalising Pio action labels. The Ryan
# pack labels raises as "Raise 60%" / "Raise 182%" / etc -- the % is the
# pack's internal "percent of pot" sizing token, NOT a player-facing
# frequency. Options shown to players should use the brief's preflop
# verb conventions: 1st raise = "Open" (or "Raise" depending on house
# style), 2nd = "3-bet", 3rd = "4-bet", 4th = "5-bet", 5+ = "Raise".
#
# We use "Raise" for raise_level=1 in OPTION columns rather than "Open"
# because "Open" reads weirdly in a multiple-choice context ("Open" vs
# "Fold"). The Question column's prose still uses "opens to $X" per
# the brief's verb table for that block.
_RAISE_LEVEL_OPTION_VERB: dict[int, str] = {
    1: "Raise",
    2: "3-bet",
    3: "4-bet",
    4: "5-bet",
}

# Frequency threshold above which the spot's dominant action is treated as
# "clearly best" for auto-pick: GTO framing below, basic above. Mirrors the
# threshold in :func:`pipeline.preflop.explanation_generator.
# _detect_option_style_preflop` so the two surfaces agree.
_BINARY_ACTION_FREQ_THRESHOLD = 0.80

# Action aggression order. Used to sort option labels so the player
# reads them as a spectrum from "most conservative" (Fold) to "most
# committed" (All-in), regardless of which action is dominant.
# Lower = less aggressive. The Fold-first ordering rule (originally a
# special case) is just the general "least-aggressive-first" rule
# applied to a Fold/<dominant> pair.
_ACTION_AGGRESSION: dict[str, int] = {
    "fold": 0,
    "check": 1,
    "call": 2,
    "raise": 3,
    "3-bet": 4,
    "4-bet": 5,
    "5-bet": 6,
    "all-in": 7,
}


def _action_aggression(label: str) -> int:
    """Return the aggression rank for an action label. Unknown -> 99."""
    if not label:
        return 99
    first_word = label.split()[0].lower().strip(".,;:!?\"'()[]")
    return _ACTION_AGGRESSION.get(first_word, 99)


# Threshold above which "Always <action>" is a plausible correct
# answer rather than a strict-wrong distractor. Below this, the
# multiple-choice template should not include "Always {dominant}" --
# it's dead air against the composite "Mostly {dominant}, sometimes X"
# alternatives that ALSO start with the dominant verb. Tracks
# frequency_to_verb_prefix's >=95% boundary so a spot whose correct
# answer starts with "Always" will always include that option.
_PURE_STRATEGY_THRESHOLD = 0.95

# The canonical answer-style identifiers. The admin panel's radio choices
# map to these via :data:`ANSWER_STYLE_FROM_RADIO_LABEL`.
ANSWER_STYLES: tuple[str, ...] = ("basic", "gto", "auto")

# Mapping from the admin panel's radio-button display strings to the
# canonical answer-style identifiers above. Kept here (not in the admin
# panel) so the orchestrator + CLI scripts can use the same mapping.
ANSWER_STYLE_FROM_RADIO_LABEL: dict[str, str] = {
    "Basic (fold/call/raise)": "basic",
    "GTO (always/mostly)": "gto",
    "Sizing (33%/75%/150%) — coming soon": "basic",  # falls back; preflop has no sizing axis
    "Auto-pick": "auto",
}


# --- internal helpers --------------------------------------------------------
def _meaningful_actions(facts: PreflopFacts) -> list[tuple[str, float]]:
    """Pio's actions at this spot, ordered by descending frequency,
    filtered to those played at least :data:`_MIN_MEANINGFUL_FREQ`."""
    ranked = sorted(
        facts.spot.action_frequencies.items(),
        key=lambda kv: -kv[1],
    )
    return [(label, freq) for label, freq in ranked if freq >= _MIN_MEANINGFUL_FREQ]


def _hero_raise_level(facts: PreflopFacts) -> int:
    """How many raises hero's prospective raise would be the (1+N)-th of.

    Counts raises + all-ins in ``history_before``. Hero's raise would
    cap that count; the verb for hero's option is keyed by the result.
    """
    return 1 + sum(
        1
        for a in facts.spot.node.history_before
        if a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
    )


def is_check_spot(facts: PreflopFacts) -> bool:
    """True when hero faces no bet and the no-raise action is a *check*.

    Preflop the BB has already posted the big blind, so when the action
    reaches them with no raise outstanding (everyone folded or
    limped/completed), the to-call is 0: their only options are to check
    or to raise -- never to "call". The Ryan pack still files the no-raise
    action under the ``"Call"`` label, so we relabel it ``"Check"`` for the
    player-facing options, the correct answer, and the SOLVER DATA block.

    Only the BB can ever check preflop (the SB still owes the blind
    difference, which makes their no-raise action a genuine
    completion/call), so the guard keys off ``actor == "BB"`` with no
    raise/all-in in the history.
    """
    node = facts.spot.node
    if node.actor != "BB":
        return False
    return not any(
        a.action_type in (PreflopActionType.RAISE, PreflopActionType.ALL_IN)
        for a in node.history_before
    )


def canonicalize_action_label(
    label: str, *, raise_level: int, check_spot: bool = False
) -> str:
    """Convert a Pio action label into a player-facing option string.

    The Ryan pack labels actions as ``"Fold"``, ``"Call"``, ``"AllIn"``,
    and ``"Raise <pct>%"`` where ``<pct>`` is the internal sizing token.
    Players seeing ``"Raise 182%"`` read it as a frequency, not a
    sizing token -- so we map raises to the brief's preflop verb
    table based on ``raise_level``.

    Args:
        label: The Pio action label.
        raise_level: How many raises hero's option would be the (n)-th
            of (1 = open, 2 = 3-bet, 3 = 4-bet, 4 = 5-bet).
        check_spot: When ``True`` (see :func:`is_check_spot`), hero is the
            BB facing no bet, so the ``"Call"`` label becomes ``"Check"``.

    Returns:
        ``"Fold"`` / ``"Call"`` (or ``"Check"`` in a check spot) /
        ``"All-in"`` / verb-by-level for raises.
    """
    if label.startswith("Raise"):
        return _RAISE_LEVEL_OPTION_VERB.get(raise_level, "Raise")
    if label == "AllIn":
        return "All-in"
    if check_spot and label == "Call":
        return "Check"
    return label  # Fold / Call / Check


def canonicalize_strategy(
    facts: PreflopFacts,
) -> dict[str, float]:
    """Return ``{canonical_label: freq}`` summing duplicate keys.

    When hero's tree has multiple raise sizes at the same node (rare in
    the Ryan pack but possible), the canonicalisation collapses them
    into a single ``"Raise"``/``"3-bet"``/etc entry whose frequency is
    the sum of the originals. For single-raise-size spots (the common
    case) this is a 1:1 relabeling.

    In a check spot (hero is the BB facing no bet, see
    :func:`is_check_spot`) the no-raise ``"Call"`` entry is relabeled
    ``"Check"`` so every downstream surface -- options, correct answer,
    SOLVER DATA, the CSV, validators -- agrees on "Check".
    """
    raise_level = _hero_raise_level(facts)
    check_spot = is_check_spot(facts)
    out: dict[str, float] = {}
    for raw_label, freq in facts.spot.action_frequencies.items():
        canon = canonicalize_action_label(
            raw_label, raise_level=raise_level, check_spot=check_spot
        )
        out[canon] = out.get(canon, 0.0) + freq
    return out


def _canonical_dominant(facts: PreflopFacts) -> str:
    """The canonical label for ``facts.spot.dominant_action``."""
    return canonicalize_action_label(
        facts.spot.dominant_action,
        raise_level=_hero_raise_level(facts),
        check_spot=is_check_spot(facts),
    )


def _pick_gto_secondary(
    facts: PreflopFacts,
    dominant_label: str,
) -> str | None:
    """Pick the ``B`` action for the 2-action GTO template.

    Rule:
      1. Among non-dominant canonical actions, pick the highest-frequency.
      2. If multiple are tied at the highest frequency, prefer ``"Fold"``
         (so a "Call 100% / Fold 0% / Raise 0%" spot picks Fold as the
         most pedagogically-useful "wrong answer" alternative).
      3. If no non-dominant actions exist (Pio's tree only has one action
         at this node -- shouldn't happen for spots passing the
         worthiness filter), return None and the caller falls back to
         the Basic style.
    """
    canonical = canonicalize_strategy(facts)
    candidates = {
        label: freq for label, freq in canonical.items() if label != dominant_label
    }
    if not candidates:
        return None
    max_freq = max(candidates.values())
    tied_at_max = [label for label, freq in candidates.items() if freq == max_freq]
    if len(tied_at_max) > 1 and "Fold" in tied_at_max:
        return "Fold"
    return tied_at_max[0]


# --- the three builders ------------------------------------------------------
def _meaningful_canonical_actions(
    facts: PreflopFacts,
) -> list[tuple[str, float]]:
    """``_meaningful_actions`` plus label canonicalisation.

    Returns canonical-labeled ``(label, freq)`` pairs ordered by
    descending frequency, filtered to >= :data:`_MIN_MEANINGFUL_FREQ`.
    Multiple raw raise labels (rare; the Ryan pack uses one per node)
    collapse into a single canonical entry whose frequency is the sum.
    """
    canonical_strategy = canonicalize_strategy(facts)
    ranked = sorted(canonical_strategy.items(), key=lambda kv: -kv[1])
    return [(label, freq) for label, freq in ranked if freq >= _MIN_MEANINGFUL_FREQ]


def build_options_basic(
    facts: PreflopFacts,
) -> tuple[list[str], str]:
    """Bare action labels, one option per meaningfully-played action.

    Returns up to 4 options ordered by descending frequency. Labels are
    canonicalised: ``"Raise 60%"`` -> ``"Raise"`` (or ``"3-bet"`` /
    ``"4-bet"`` / ``"5-bet"`` depending on prior-raise count); ``"AllIn"``
    -> ``"All-in"``. The ``correct_answer`` is the canonical label of
    the dominant action.

    Empty / degenerate strategies fall through to a 1-option set
    containing the canonical dominant action, so the row stays valid.
    """
    canonical = canonicalize_strategy(facts)
    canonical_correct = _canonical_dominant(facts)
    if not canonical:
        return [canonical_correct], canonical_correct

    # Sort by descending frequency for the cap step.
    ordered = sorted(canonical.items(), key=lambda kv: -kv[1])

    # Step 2: 4-option cap with Fold-protection rule.
    if len(ordered) <= 4:
        kept = {label for label, _ in ordered}
    else:
        fold_freq = canonical.get("Fold", 0.0)
        if fold_freq == 0.0:
            # Fold played 0% AND there are 5+ alternatives -- drop Fold.
            kept = {label for label, _ in ordered if label != "Fold"}
            # Take top 4 of what remains.
            top_4 = [label for label, _ in ordered if label != "Fold"][:4]
            kept = set(top_4)
        else:
            # Fold protected because Pio actually plays it sometimes.
            kept_non_fold = [label for label, _ in ordered if label != "Fold"][:3]
            kept = {"Fold", *kept_non_fold}

    # Step 3: Fold first when present, remaining by descending frequency.
    if "Fold" in kept:
        remaining_by_freq = [
            label for label, _ in ordered if label in kept and label != "Fold"
        ]
        ordered_options = ["Fold", *remaining_by_freq]
    else:
        ordered_options = [label for label, _ in ordered if label in kept]

    # Defensive: dominant must be in options. The cap rule only drops Fold
    # or low-frequency actions, never the dominant (which has rank 1 by
    # frequency), so this is belt-and-suspenders.
    if canonical_correct not in ordered_options:
        ordered_options = [
            canonical_correct,
            *(opt for opt in ordered_options if opt != canonical_correct),
        ][:4]

    return ordered_options, canonical_correct


def build_options_gto(
    facts: PreflopFacts,
) -> tuple[list[str], str]:
    """Always/Mostly 4-option spectrum -- unified for all preflop spots.

    Picks Pio's top 2 actions by frequency, sorts them by aggression
    (less-aggressive first), and emits the 4-option spectrum::

        Always <less>, Mostly <less>, Mostly <more>, Always <more>

    Reads as a spectrum from "most conservative" (pure less-aggressive)
    to "most committed" (pure more-aggressive). The two middle
    "Mostly" rungs represent mixed strategies leaning each direction.

    The correct_answer is ``"<prefix> <dominant>"`` where prefix is
    ``frequency_to_verb_prefix(dominant_freq)``:

      * dominant freq >= 95% -> ``"Always {dominant}"``
      * 5% <= dominant freq < 95% -> ``"Mostly {dominant}"``

    Both correct-answer forms are always present in the spectrum, so
    the player picks between the pure-or-mixed framing in each
    direction.

    For 3+ action mixes (e.g. Call 70%, 4-bet 15%, All-in 15%) the
    third+ actions are NOT surfaced as separate options. The lower-
    frequency secondaries get folded into the LLM's explanation prose
    via the SOLVER DATA block; the option set only tests "which
    direction dominates, and is it pure or mixed?". This dropped the
    old composite-label template (``"Mostly X, sometimes Y"``) -- it
    was hard to evaluate without a meta-elimination shortcut, and
    composite labels overloaded both the player and the validators
    with frequency-precision questions that weren't the lesson.

    Single-action edge case (one action played at 100%): falls back
    to the 2-action template against a secondary picked from Pio's
    tree (no frequency filter) so the player still gets real choices
    to rule out.
    """
    meaningful = _meaningful_canonical_actions(facts)
    if not meaningful:
        return build_options_basic(facts)

    dominant_label, dominant_freq = meaningful[0]
    prefix = frequency_to_verb_prefix(dominant_freq)
    if prefix == "":
        # Dominant played at < 5% -- bizarre edge case. Fall through to basic.
        return build_options_basic(facts)
    correct = f"{prefix} {dominant_label}"

    # Pick the secondary action. When 2+ actions are >= 5%, take the
    # most-frequent non-dominant. When only one action is meaningful
    # (a near-pure spot, e.g. Fold 100%), reach into Pio's tree-
    # offered actions for a non-frequency-filtered alternative.
    if len(meaningful) >= 2:
        secondary_label = meaningful[1][0]
    else:
        picked = _pick_gto_secondary(facts, dominant_label)
        if picked is None:
            # Pio's tree only offers one action -- worthiness filter
            # would normally drop this, but fall back to basic just
            # in case it slips through.
            return build_options_basic(facts)
        secondary_label = picked

    # Order by action aggression: less-aggressive first. The 4-option
    # block reads as a spectrum (Fold-leaning -> commit-leaning).
    less, more = sorted(
        [dominant_label, secondary_label], key=_action_aggression
    )
    options = [
        f"Always {less}",
        f"Mostly {less}",
        f"Mostly {more}",
        f"Always {more}",
    ]
    return options, correct


def build_options_auto(
    facts: PreflopFacts,
) -> tuple[list[str], str]:
    """Pick basic vs gto based on dominant-action frequency.

    Dominant freq >= 80%: spot is clearly answered, basic framing reads
    cleanest (just the action names).
    Dominant freq <  80%: spot is meaningfully mixed, GTO framing
    surfaces both actions so the player learns the mix.
    """
    if facts.spot.dominant_frequency >= _BINARY_ACTION_FREQ_THRESHOLD:
        return build_options_basic(facts)
    return build_options_gto(facts)


# --- the dispatcher ---------------------------------------------------------
def build_options(
    facts: PreflopFacts,
    *,
    style: str = "auto",
) -> tuple[list[str], str]:
    """Compute ``(options, correct_answer)`` for one preflop spot.

    Args:
        facts: The Layer 5 preflop data block.
        style: One of :data:`ANSWER_STYLES` (``"basic"``, ``"gto"``,
            ``"auto"``). Default ``"auto"``.

    Returns:
        ``(options, correct_answer)``: a list of 1-4 option strings
        (the first is always the dominant action's framing) and a
        single ``correct_answer`` string that equals exactly one
        member of ``options``.

    Raises:
        ValueError: if ``style`` isn't in :data:`ANSWER_STYLES`.
    """
    if style == "basic":
        return build_options_basic(facts)
    if style == "gto":
        return build_options_gto(facts)
    if style == "auto":
        return build_options_auto(facts)
    raise ValueError(f"unknown answer style {style!r}; expected one of {ANSWER_STYLES}")


__all__ = [
    "ANSWER_STYLES",
    "ANSWER_STYLE_FROM_RADIO_LABEL",
    "build_options",
    "build_options_auto",
    "build_options_basic",
    "build_options_gto",
    "canonicalize_action_label",
    "canonicalize_strategy",
    "is_check_spot",
]
