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


def canonicalize_action_label(label: str, *, raise_level: int) -> str:
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

    Returns:
        ``"Fold"`` / ``"Call"`` / ``"All-in"`` / verb-by-level for raises.
    """
    if label.startswith("Raise"):
        return _RAISE_LEVEL_OPTION_VERB.get(raise_level, "Raise")
    if label == "AllIn":
        return "All-in"
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
    """
    raise_level = _hero_raise_level(facts)
    out: dict[str, float] = {}
    for raw_label, freq in facts.spot.action_frequencies.items():
        canon = canonicalize_action_label(raw_label, raise_level=raise_level)
        out[canon] = out.get(canon, 0.0) + freq
    return out


def _canonical_dominant(facts: PreflopFacts) -> str:
    """The canonical label for ``facts.spot.dominant_action``."""
    return canonicalize_action_label(
        facts.spot.dominant_action,
        raise_level=_hero_raise_level(facts),
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
    """Always/Mostly template per the brief's GTO framing.

    Two-action mix (the common case):
      ``"Always <L>", "Mostly <L>", "Mostly <M>", "Always <M>"``
      where L is the LESS aggressive action and M is the more
      aggressive one -- regardless of which is dominant. Reads as a
      spectrum from most-conservative to most-committed. (Generalises
      the old Fold-first special case: Fold is just the least
      aggressive action, so it falls out naturally.)

    Three-action+ mix:
      ``[Always Fold OR Mostly Fold, sometimes <dominant>]`` first
      (an anti-strategy distractor), then ``"Mostly <dominant>,
      sometimes <secondary>"`` labels covering each non-dominant
      action played at >= 5%, ordered by aggression of the secondary.
      The leading ``"Always <dominant>"`` (the old design) is dropped
      because for mixed spots (dominant freq < 95%) it's dead air
      against the composites -- any halfway-thoughtful player
      eliminates it instantly.

    Single-action (one action played at 100%):
      Falls back to the 2-action template against a secondary picked
      from Pio's tree (so the player gets real choices to rule out).

    The correct_answer is ``"<prefix> <A>"`` where prefix is
    ``frequency_to_verb_prefix(dominant_freq)`` -- the same
    deterministic mapping the LLM was being instructed to follow
    in the previous Layer-6-picks-options architecture.
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

    # --- 3+ meaningful actions: composite-label template ---------------------
    if len(meaningful) >= 3:
        return _build_options_gto_composite(
            dominant_label, dominant_freq,
            secondary_labels=[label for label, _ in meaningful[1:]],
            correct=correct,
        )

    # --- 1 or 2 meaningful actions: 4-option 2-action template --------------
    # For 1-meaningful (near-pure spot), Pio plays one action almost
    # entirely; we still emit the full 4-option template so the player
    # gets real choices to rule out. The B action is picked from Pio's
    # other tree-offered actions (no frequency filter).
    secondary_label = _pick_gto_secondary(facts, dominant_label)
    if secondary_label is None:
        # Pio's tree only offers one action at this node. Degenerate
        # (the worthiness filter normally excludes such spots) -- fall
        # back to basic rather than emit a 1-option set.
        return build_options_basic(facts)

    # Order by action aggression: less-aggressive first. This makes the
    # 4-option block read as a spectrum (Fold-leaning -> commit-leaning).
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


def _build_options_gto_composite(
    dominant_label: str,
    dominant_freq: float,
    *,
    secondary_labels: list[str],
    correct: str,
) -> tuple[list[str], str]:
    """3+ action template: drop "Always {dominant}", use anti-strategy distractor.

    Layout (in order):
      [0] Anti-strategy distractor -- "Always Fold" if Fold isn't a
          secondary, else "Mostly Fold, sometimes {dominant}".
          (Skipped if the dominant action IS Fold, since both forms
          would clash with the dominant-anchored composites below.)
      [1] "Mostly {dominant}, sometimes {secondary[0]}"   <- correct
      [2] "Mostly {dominant}, sometimes {secondary[1]}"
      [3] "Mostly {dominant}, sometimes {secondary[2]}" (if exists)

    Secondaries are sorted by aggression (lesser-action first) so the
    composite alternatives also read as a spectrum.

    When dominant freq >= 95%, the spot is essentially pure; correct
    will be "Always {dominant}" rather than a composite. In that case
    the dominant-anchored composites are wrong anyway, so this template
    still works -- the user can pick "Always {dominant}" from one of
    the slots... wait, it's not there. So for the rare pure-with-tiny-
    mix-in case, this template hides the correct answer. Fall back.
    """
    correct_uses_always = correct.startswith("Always ")
    if correct_uses_always:
        # Rare: dominant >= 95% with 2+ other actions >= 5%. The
        # composite template can't represent "Always {dominant}".
        # Fall back to the basic builder so the correct answer is
        # in the option set.
        return build_options_basic_for_correct(
            dominant_label, secondary_labels, correct
        )

    # Sort secondaries by aggression so composites read in a spectrum.
    sorted_secondaries = sorted(secondary_labels, key=_action_aggression)
    composites = [
        f"Mostly {dominant_label}, sometimes {sec}"
        for sec in sorted_secondaries
    ]
    correct_composite = f"Mostly {dominant_label}, sometimes {secondary_labels[0]}"
    if correct_composite not in composites:
        # Defensive: should not happen since secondaries is the source
        # of sorted_secondaries. Fall through to basic.
        return build_options_basic_for_correct(
            dominant_label, secondary_labels, correct
        )

    # Pick an anti-strategy distractor. The lead option of the option
    # set -- the "you're being too tight" alternative the player
    # weighs against the dominant-anchored composites.
    distractor: str | None
    if dominant_label == "Fold":
        # Dominant is already Fold -- a Fold-flavored distractor would
        # clash. Skip the distractor; 3-option set is fine.
        distractor = None
    elif "Fold" not in secondary_labels:
        # Fold isn't in the strategy at all -> "Always Fold" is a clean
        # anti-strategy distractor: extreme conservative play.
        distractor = "Always Fold"
    else:
        # Fold is already a secondary (one of the composites is
        # "Mostly {dominant}, sometimes Fold"). Pick the overfold
        # composite as the distractor: "Mostly Fold, sometimes {dom}".
        distractor = f"Mostly Fold, sometimes {dominant_label}"
        # Remove the existing "Mostly {dominant}, sometimes Fold" since
        # the overfold version is its mirror image and we don't need both.
        composites = [c for c in composites if c != f"Mostly {dominant_label}, sometimes Fold"]
        # Re-include the dominant-anchored Fold composite if dropping
        # it would lose the correct answer.
        if correct_composite not in composites:
            composites.insert(0, correct_composite)

    options: list[str] = []
    if distractor is not None:
        options.append(distractor)
    options.extend(composites)
    options = options[:4]

    if correct_composite not in options:
        # Defensive: padding overflow. Swap last for correct.
        options[-1] = correct_composite

    return options, correct_composite


def build_options_basic_for_correct(
    dominant_label: str,
    secondary_labels: list[str],
    correct: str,
) -> tuple[list[str], str]:
    """Fallback used by GTO composite when 'Always {dominant}' is correct.

    Builds a Fold/Call/Raise-style option set that includes the
    correct "Always {dominant}" answer alongside basic action labels
    for the other secondaries. Rare branch -- only fires when the
    dominant action is at >= 95% but the spot still has >= 3 actions
    above the 5% threshold. The basic builder would also fit here,
    but its correct_answer is just the bare action label whereas this
    one preserves the "Always" prefix.
    """
    options: list[str] = [correct]  # "Always {dominant}"
    for label in [dominant_label, *secondary_labels]:
        if label not in options:
            options.append(label)
    options = options[:4]
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
]
