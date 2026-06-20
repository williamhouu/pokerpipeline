"""Deterministic neutral-credit scoring for multiple-choice answer options.

Shared by all three question pipelines (NLHE preflop, PLO preflop, postflop).
A pure leaf utility: it imports nothing from any pipeline, so the
self-contained postflop path can use it under its import contract.

The product grades an answer into three buckets instead of two: a player who
picks the single best option gets FULL credit, a player who picks an option
that is "close enough" to be defensible gets NEUTRAL credit (held harmless,
no penalty), and everyone else made a mistake. This module computes which
options earn the neutral bucket. The LLM is never involved -- the options and
their frequencies come straight from the solver.

## The 20-point rule

Every option makes a frequency claim, and earns credit when the hand's real
solver frequency for that action is within 20 percentage points of the claim:

  * ``"Always X"`` claims X is played 100% of the time -> creditable when the
    real frequency of X is >= 80% (within 20 of 100). So "Always Call" is only
    a fair answer when the hand really does almost always call.
  * ``"Mostly X"`` claims X is a real part of the mix     -> real freq >= 20%.
  * a bare ``"X"`` action label (basic style) makes the same "real part of the
    mix" claim                                            -> real freq >= 20%.

The single best-matching option is the full-credit ``correct_answer`` (chosen
upstream by the option builder); every OTHER creditable option earns neutral
credit; the rest are mistakes. Worked examples (Call dominant):

  * Call 65 / Raise 35, correct "Mostly Call": "Mostly Raise" is NEUTRAL
    (Raise 35% >= 20%); "Always Call" is a MISTAKE (Call 65% < 80%, so "always"
    overstates a clearly mixed spot); "Always Raise" is a mistake.
  * Call 85 / Raise 15, correct "Mostly Call": "Always Call" is NEUTRAL
    (Call 85% >= 80%, basically right); both raise rungs are mistakes
    (Raise 15% < 20%, the 15% line gets no credit).

Why frequency and not EV: a solver only mixes two actions when their EVs are
~equal, so the EV gap is ~0 for *every* mix (85/15 and 65/35 alike). EV
therefore can't tell a thin sliver from a real toss-up; the frequency split
can. The cutoff is frequency-based on purpose.
"""

from __future__ import annotations

from collections.abc import Mapping

# A claim is "close enough" when the real frequency is within 20 points of it.
ALWAYS_CLAIM_MIN_FREQ = 0.80  # "Always X" claims 100% -> real X must be >= 80%
MOSTLY_CLAIM_MIN_FREQ = 0.20  # "Mostly X" / bare "X"  -> real X must be >= 20%

# Floating-point slack so a frequency that is exactly on a boundary (e.g. an
# 80/20 spot whose 0.80 arrived as 0.79999998 from a weight/presence divide)
# still counts. Far smaller than any meaningful frequency difference.
_FREQ_EPSILON = 1e-9

_ALWAYS_PREFIX = "Always "
_MOSTLY_PREFIX = "Mostly "


def _action_and_floor(option: str) -> tuple[str, float]:
    """Parse an option string into ``(action_label, min_frequency)``.

    ``"Always Call"`` -> ``("Call", 0.80)``; ``"Mostly Raise"`` -> ``("Raise",
    0.20)``; a bare ``"Fold"`` -> ``("Fold", 0.20)``. The returned action label
    is exactly what the strategy mix is keyed by, so the caller's frequency map
    looks it up directly (canonical preflop labels like ``"3-bet"`` / ``"All-in"``
    or postflop labels like ``"Bet 75%"`` both round-trip).
    """
    if option.startswith(_ALWAYS_PREFIX):
        return option[len(_ALWAYS_PREFIX) :], ALWAYS_CLAIM_MIN_FREQ
    if option.startswith(_MOSTLY_PREFIX):
        return option[len(_MOSTLY_PREFIX) :], MOSTLY_CLAIM_MIN_FREQ
    return option, MOSTLY_CLAIM_MIN_FREQ


def neutral_credit_options(
    options: list[str],
    correct_answer: str,
    action_frequencies: Mapping[str, float],
) -> list[str]:
    """Return the options that earn neutral credit, in their input order.

    Args:
        options: The displayed option strings (empties are ignored).
        correct_answer: The full-credit option -- excluded from the result.
        action_frequencies: Maps each action label to the hand's real solver
            frequency (the SAME labels the options are built from). For preflop
            this is ``canonicalize_strategy(facts)``; for postflop it is
            ``spot.action_frequencies``.

    Returns:
        The subset of ``options`` (other than ``correct_answer``) whose action
        clears the 20-point floor for its "Always"/"Mostly"/bare claim. Pure
        and deterministic -- no LLM, no randomness.
    """
    neutral: list[str] = []
    for option in options:
        if not option or option == correct_answer:
            continue
        action, floor = _action_and_floor(option)
        if action_frequencies.get(action, 0.0) >= floor - _FREQ_EPSILON:
            neutral.append(option)
    return neutral


def format_neutral_credit(neutral: list[str]) -> str:
    """The CSV cell value for the ``neutral_credit`` column.

    Comma-separated option strings in order, or ``""`` when no option earns
    neutral credit (the common case for a clear spot).
    """
    return ", ".join(neutral)


__all__ = [
    "ALWAYS_CLAIM_MIN_FREQ",
    "MOSTLY_CLAIM_MIN_FREQ",
    "format_neutral_credit",
    "neutral_credit_options",
]
