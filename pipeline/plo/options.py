"""Deterministic option selection for PLO preflop questions.

Per "the LLM never thinks about poker", the four option strings and the
``correct_answer`` are computed entirely from the solver strategy -- no LLM. The
LLM (Layer 6) writes only the explanation prose; the options come from here.

Ports :mod:`pipeline.preflop.options`. Three styles:

* :func:`build_options_basic` -- bare action labels (``"Fold"``, ``"Call"``,
  ``"Raise"`` / ``"3-bet"`` / ...).
* :func:`build_options_gto` -- the Always/Mostly spectrum for mixed spots.
* :func:`build_options_auto` -- basic when the dominant action is >= 80%, gto
  otherwise.

PLO action labels come from the node options (``"Fold"``, ``"Call"``,
``"Raise 100%"``, ``"All-in"``, ``"Min-raise"``); raises canonicalise to the
preflop bet-level verb (Raise / 3-bet / 4-bet / 5-bet) by how many raises came
before. Every raise in this pack is pot-sized, so the ``%`` sizing token is
dropped from the player-facing label.
"""

from __future__ import annotations

from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.pack import PloActionType

_MIN_MEANINGFUL_FREQ = 0.05
_BINARY_ACTION_FREQ_THRESHOLD = 0.80
_PURE_STRATEGY_PREFIX = 0.95
_MAX_OPTIONS = 4
_MAX_NON_FOLD = 3

_AGGRESSIVE = {
    PloActionType.RAISE,
    PloActionType.MIN_RAISE,
    PloActionType.ALL_IN,
}

_RAISE_LEVEL_OPTION_VERB: dict[int, str] = {
    1: "Raise",
    2: "3-bet",
    3: "4-bet",
    4: "5-bet",
}

# Aggression order: read options from most conservative (Fold) to most
# committed (All-in), regardless of which is dominant.
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

ANSWER_STYLES: tuple[str, ...] = ("basic", "gto", "auto")


def _freq_prefix(freq: float) -> str:
    """Frequency -> ``"Always"`` (>=95%) / ``"Mostly"`` (>=5%) / ``""``."""
    if freq >= _PURE_STRATEGY_PREFIX:
        return "Always"
    if freq >= _MIN_MEANINGFUL_FREQ:
        return "Mostly"
    return ""


def _action_aggression(label: str) -> int:
    if not label:
        return 99
    first_word = label.split()[0].lower().strip(".,;:!?\"'()[]")
    return _ACTION_AGGRESSION.get(first_word, 99)


def _hero_raise_level(facts: PloFacts) -> int:
    """How many raises hero's prospective raise would be the (1+N)-th of."""
    return 1 + sum(
        1 for a in facts.spot.node.history_before if a.action in _AGGRESSIVE
    )


def is_check_spot(facts: PloFacts) -> bool:
    """True when hero faces no bet and the no-raise action is a *check*.

    Preflop the BB has already posted the big blind, so when the action reaches
    them with no raise outstanding (everyone folded or limped/completed), the
    to-call is 0: their only options are to check or to raise -- never to
    "call". The solver still files the no-raise action under the ``"Call"``
    label (the pack's call token), so we relabel it ``"Check"`` for the
    player-facing options, the correct answer, and the SOLVER DATA block.

    Only the BB can ever check preflop (the SB still owes the blind difference,
    which makes their no-raise action a genuine completion/call), so the guard
    keys off ``actor == "BB"`` with no raise/all-in in the history.
    """
    node = facts.spot.node
    if node.actor != "BB":
        return False
    return not any(a.action in _AGGRESSIVE for a in node.history_before)


def canonicalize_action_label(
    label: str, *, raise_level: int, check_spot: bool = False
) -> str:
    """Convert a node action label into a player-facing option string.

    ``"Raise 100%"`` / ``"Min-raise"`` -> the bet-level verb (Raise / 3-bet /
    4-bet / 5-bet); ``"All-in"`` stays; ``"Fold"`` stays. ``"Call"`` stays --
    except in a check spot (``check_spot=True``, see :func:`is_check_spot`),
    where the BB faces no bet and ``"Call"`` becomes ``"Check"``.
    """
    if label.startswith("Raise") or label == "Min-raise":
        return _RAISE_LEVEL_OPTION_VERB.get(raise_level, "Raise")
    if check_spot and label == "Call":
        return "Check"
    return label  # Fold / Call / All-in


def canonicalize_strategy(facts: PloFacts) -> dict[str, float]:
    """``{canonical_label: freq}`` summing duplicate labels (e.g. two raise
    sizes collapsing into one ``"3-bet"`` entry)."""
    raise_level = _hero_raise_level(facts)
    check_spot = is_check_spot(facts)
    out: dict[str, float] = {}
    for raw_label, freq in facts.spot.action_frequencies.items():
        canon = canonicalize_action_label(
            raw_label, raise_level=raise_level, check_spot=check_spot
        )
        out[canon] = out.get(canon, 0.0) + freq
    return out


def _canonical_dominant(facts: PloFacts) -> str:
    return canonicalize_action_label(
        facts.spot.dominant_action,
        raise_level=_hero_raise_level(facts),
        check_spot=is_check_spot(facts),
    )


def _pick_gto_secondary(facts: PloFacts, dominant_label: str) -> str | None:
    """The ``B`` action for the 2-action GTO template: highest-frequency
    non-dominant action, preferring Fold on ties; None if none exists."""
    canonical = canonicalize_strategy(facts)
    candidates = {lbl: f for lbl, f in canonical.items() if lbl != dominant_label}
    if not candidates:
        return None
    max_freq = max(candidates.values())
    tied = [lbl for lbl, f in candidates.items() if f == max_freq]
    if len(tied) > 1 and "Fold" in tied:
        return "Fold"
    return tied[0]


def _meaningful_canonical_actions(facts: PloFacts) -> list[tuple[str, float]]:
    """Canonical ``(label, freq)`` pairs, descending freq, >= 5%."""
    canonical = canonicalize_strategy(facts)
    ranked = sorted(canonical.items(), key=lambda kv: -kv[1])
    return [(lbl, f) for lbl, f in ranked if f >= _MIN_MEANINGFUL_FREQ]


def build_options_basic(facts: PloFacts) -> tuple[list[str], str]:
    """Bare action labels, one per meaningfully-played action (<= 4),
    Fold first. ``correct_answer`` is the canonical dominant action."""
    canonical = canonicalize_strategy(facts)
    canonical_correct = _canonical_dominant(facts)
    if not canonical:
        return [canonical_correct], canonical_correct

    ordered = sorted(canonical.items(), key=lambda kv: -kv[1])

    if len(ordered) <= _MAX_OPTIONS:
        kept = {lbl for lbl, _ in ordered}
    elif canonical.get("Fold", 0.0) == 0.0:
        # Fold played 0% AND 5+ alternatives -> drop Fold, take the top 4.
        kept = {lbl for lbl, _ in [o for o in ordered if o[0] != "Fold"][:_MAX_OPTIONS]}
    else:
        # Fold protected (Pio actually plays it): keep Fold + top 3 others.
        kept_non_fold = [lbl for lbl, _ in ordered if lbl != "Fold"][:_MAX_NON_FOLD]
        kept = {"Fold", *kept_non_fold}

    if "Fold" in kept:
        remaining = [lbl for lbl, _ in ordered if lbl in kept and lbl != "Fold"]
        ordered_options = ["Fold", *remaining]
    else:
        ordered_options = [lbl for lbl, _ in ordered if lbl in kept]

    if canonical_correct not in ordered_options:
        ordered_options = [
            canonical_correct,
            *(o for o in ordered_options if o != canonical_correct),
        ][:_MAX_OPTIONS]

    return ordered_options, canonical_correct


def build_options_gto(facts: PloFacts) -> tuple[list[str], str]:
    """Always/Mostly 4-option spectrum from the top two actions::

        Always <less>, Mostly <less>, Mostly <more>, Always <more>

    sorted less-aggressive first. ``correct_answer`` is
    ``"<Always|Mostly> <dominant>"``. Falls back to basic on degenerate input.
    """
    meaningful = _meaningful_canonical_actions(facts)
    if not meaningful:
        return build_options_basic(facts)

    dominant_label, dominant_freq = meaningful[0]
    prefix = _freq_prefix(dominant_freq)
    if prefix == "":
        return build_options_basic(facts)
    correct = f"{prefix} {dominant_label}"

    if len(meaningful) >= 2:  # noqa: PLR2004
        secondary_label = meaningful[1][0]
    else:
        picked = _pick_gto_secondary(facts, dominant_label)
        if picked is None:
            return build_options_basic(facts)
        secondary_label = picked

    less, more = sorted([dominant_label, secondary_label], key=_action_aggression)
    options = [
        f"Always {less}",
        f"Mostly {less}",
        f"Mostly {more}",
        f"Always {more}",
    ]
    return options, correct


def build_options_auto(facts: PloFacts) -> tuple[list[str], str]:
    """Basic when the dominant action is >= 80%, gto otherwise."""
    if facts.spot.dominant_frequency >= _BINARY_ACTION_FREQ_THRESHOLD:
        return build_options_basic(facts)
    return build_options_gto(facts)


def build_options(facts: PloFacts, *, style: str = "auto") -> tuple[list[str], str]:
    """Compute ``(options, correct_answer)`` for one PLO spot.

    ``options`` is 1-4 strings; ``correct_answer`` equals exactly one of them.
    Raises ``ValueError`` on an unknown style.
    """
    if style == "basic":
        return build_options_basic(facts)
    if style == "gto":
        return build_options_gto(facts)
    if style == "auto":
        return build_options_auto(facts)
    msg = f"unknown answer style {style!r}; expected one of {ANSWER_STYLES}"
    raise ValueError(msg)


__all__ = [
    "ANSWER_STYLES",
    "build_options",
    "build_options_auto",
    "build_options_basic",
    "build_options_gto",
    "canonicalize_action_label",
    "canonicalize_strategy",
    "is_check_spot",
]
