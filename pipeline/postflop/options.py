"""Build the four multiple-choice answer options for a postflop spot.

Deterministic, no LLM: the options and the correct answer are derived from
the node's available actions + hero's dominant action. Layer 6 is *handed*
these and must reproduce the correct one exactly (the validator enforces
``correct_answer in options``).

Two shapes, matching the team's existing format:

* **Binary spots** (exactly two actions, e.g. Check/Bet or Fold/Call): the
  Always/Mostly spectrum the gold questions use --
  ``["Always {less}", "Mostly {less}", "Mostly {more}", "Always {more}"]``
  ordered least-aggressive first. The correct answer is ``"Always {dom}"``
  when the dominant action is ~pure (>=95%) else ``"Mostly {dom}"``.
* **Multi-action spots** (3+ actions, e.g. Check / Bet 33% / Bet 75%, or
  Fold / Call / Raise): one plain option per action, ordered by aggression;
  the correct answer is the dominant action's plain label.

In both cases ``correct_answer`` is guaranteed to be one of the returned
options verbatim.
"""

from __future__ import annotations

from pipeline.postflop.solve import NodeAction
from pipeline.postflop.spot_sampler import PostflopSpot

# Aggression ordering of the canonical verbs (least -> most aggressive).
_VERB_RANK = {"fold": 0, "check": 1, "call": 2, "bet": 3, "raise": 4}

# Dominant frequency at/above which the correct answer is phrased "Always X"
# rather than "Mostly X" (binary spots only). "Always" is reserved for a
# literally-pure (100%) action -- a worthy spot tops out at 99% dominant, so its
# correct answer is "Mostly X" and "Always X" is at most a neutral near-miss
# (June 2026; mirrors the preflop frequency_to_verb_prefix change).
PURE_THRESHOLD = 0.9999

# Above this dominant frequency, "auto" style uses plain labels (the spot is
# clearly answered, so the Always/Mostly spectrum is overkill); below, it uses
# the spectrum. Mirrors the preflop auto threshold.
_AUTO_BASIC_THRESHOLD = 0.80

# The three answer-option styles (parity with preflop):
#   * "basic" -- plain action labels (Check / Bet 33% / Fold / Call / Raise to X).
#   * "gto"   -- the Always/Mostly spectrum for a 2-action spot; a 3+-size spot
#                can't be spectrum'd, so it falls back to plain labels.
#   * "auto"  -- basic when one action is clearly dominant (>= 80%), else gto.
ANSWER_STYLES: tuple[str, ...] = ("basic", "gto", "auto")

# Admin radio label -> canonical style (kept here so the CLI + admin agree).
ANSWER_STYLE_FROM_RADIO_LABEL: dict[str, str] = {
    "Basic (plain labels)": "basic",
    "GTO (always/mostly)": "gto",
    "Auto-pick": "auto",
}


def _aggression_key(action: NodeAction) -> tuple[int, float]:
    """Sort key: verb rank, then bet size (so Bet 33% precedes Bet 75%)."""
    size = action.pot_fraction if action.pot_fraction is not None else (
        action.to_bb if action.to_bb is not None else 0.0
    )
    return (_VERB_RANK.get(action.verb, 9), size)


def _plain_options(
    actions: list[NodeAction], labels: list[str], dominant: str
) -> tuple[list[str], str]:
    """Plain action labels (one per action, <= 4, aggression-ordered); the
    correct answer is the dominant action's plain label."""
    options = labels[:4]
    if dominant not in options:
        options = [dominant] + [lbl for lbl in labels if lbl != dominant][:3]
        options = sorted(
            options,
            key=lambda lbl: _aggression_key(next(a for a in actions if a.label == lbl)),
        )
    return options, dominant


def _spectrum_options(
    spot: PostflopSpot, labels: list[str], dominant: str
) -> tuple[list[str], str]:
    """The Always/Mostly 4-rung spectrum for a 2-action spot. ``correct`` is
    ``"Always {dom}"`` only when the dominant action is literally pure, else
    ``"Mostly {dom}"`` (June 2026 rule)."""
    less, more = labels[0], labels[1]
    options = [f"Always {less}", f"Mostly {less}", f"Mostly {more}", f"Always {more}"]
    prefix = "Always" if spot.dominant_frequency >= PURE_THRESHOLD else "Mostly"
    correct = f"{prefix} {dominant}"
    if correct not in options:  # defensive: dominant must be one of the two
        correct = f"Mostly {dominant}" if f"Mostly {dominant}" in options else options[0]
    return options, correct


def build_options(
    spot: PostflopSpot, *, style: str = "auto"
) -> tuple[list[str], str]:
    """Return ``(options, correct_answer)`` for ``spot`` in the given style.

    ``style`` is one of :data:`ANSWER_STYLES` (``"basic"`` / ``"gto"`` /
    ``"auto"``). ``options`` has up to four entries (empty slots omitted; the
    format writer maps them onto option_1..option_4); ``correct_answer`` is
    always exactly one of them.

    Raises:
        ValueError: if the node exposes no actions (a malformed solve), or
            ``style`` isn't recognised.
    """
    actions = sorted(spot.node.actions, key=_aggression_key)
    if not actions:
        raise ValueError(f"node {spot.node.node_id} has no actions to build options")
    labels = [a.label for a in actions]
    dominant = spot.dominant_action

    resolved = style
    if resolved == "auto":
        resolved = "basic" if spot.dominant_frequency >= _AUTO_BASIC_THRESHOLD else "gto"
    if resolved not in ("basic", "gto"):
        raise ValueError(f"unknown answer style {style!r}; expected one of {ANSWER_STYLES}")

    # The Always/Mostly spectrum applies only to a 2-action spot; a multi-size
    # spot (Check / Bet 33% / Bet 75% / ...) can't be spectrum'd, so gto falls
    # back to plain labels there.
    if resolved == "gto" and len(actions) == 2:  # noqa: PLR2004
        return _spectrum_options(spot, labels, dominant)
    return _plain_options(actions, labels, dominant)


__all__ = [
    "ANSWER_STYLES",
    "ANSWER_STYLE_FROM_RADIO_LABEL",
    "PURE_THRESHOLD",
    "build_options",
]
