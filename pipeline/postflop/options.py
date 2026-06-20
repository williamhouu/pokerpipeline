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


def _aggression_key(action: NodeAction) -> tuple[int, float]:
    """Sort key: verb rank, then bet size (so Bet 33% precedes Bet 75%)."""
    size = action.pot_fraction if action.pot_fraction is not None else (
        action.to_bb if action.to_bb is not None else 0.0
    )
    return (_VERB_RANK.get(action.verb, 9), size)


def build_options(spot: PostflopSpot) -> tuple[list[str], str]:
    """Return ``(options, correct_answer)`` for ``spot``.

    ``options`` has up to four entries (empty slots are simply omitted; the
    format writer maps them onto option_1..option_4). ``correct_answer`` is
    always exactly one of ``options``.

    Raises:
        ValueError: if the node exposes no actions (a malformed solve).
    """
    actions = sorted(spot.node.actions, key=_aggression_key)
    if not actions:
        raise ValueError(f"node {spot.node.node_id} has no actions to build options")

    labels = [a.label for a in actions]
    dominant = spot.dominant_action

    if len(actions) == 2:
        less, more = labels[0], labels[1]
        options = [
            f"Always {less}",
            f"Mostly {less}",
            f"Mostly {more}",
            f"Always {more}",
        ]
        prefix = "Always" if spot.dominant_frequency >= PURE_THRESHOLD else "Mostly"
        correct = f"{prefix} {dominant}"
        # Defensive: the dominant must be one of the two node actions.
        if correct not in options:
            correct = f"Mostly {dominant}" if f"Mostly {dominant}" in options else options[0]
        return options, correct

    # 3+ actions: plain labels, one per action, capped at four (keeping the
    # dominant if we must drop any).
    options = labels[:4]
    if dominant not in options:
        options = [dominant] + [lbl for lbl in labels if lbl != dominant][:3]
        # restore aggression order
        options = sorted(options, key=lambda lbl: _aggression_key(
            next(a for a in actions if a.label == lbl)
        ))
    return options, dominant


__all__ = ["PURE_THRESHOLD", "build_options"]
