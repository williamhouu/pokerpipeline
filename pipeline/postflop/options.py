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

# Near-binary collapse: a verb taken BELOW this frequency (summed over its sizes)
# is a GTO-balancing sliver, not a real option. When a 3+-verb spot (e.g. a
# Fold/Call/Raise facing-bet decision) has exactly TWO verbs at/above this, the
# slivers are dropped and the two live verbs are spectrum'd -- so a 60/38/2
# Fold/Call/Raise reads as the Fold-vs-Call binary it really is, instead of
# falling back to plain labels.
_NEAR_BINARY_DROP_FREQ = 0.05

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


def _verb_frequencies(spot: PostflopSpot) -> dict[str, float]:
    """Each verb's total solver frequency at this spot (summed over its sizes)."""
    freqs: dict[str, float] = {}
    for a in spot.node.actions:
        freqs[a.verb] = freqs.get(a.verb, 0.0) + spot.action_frequencies.get(a.label, 0.0)
    return freqs


def _live_verbs(spot: PostflopSpot) -> list[str]:
    """The verbs taken at/above the near-binary threshold (the real options);
    verbs below it are GTO-balancing slivers."""
    return [v for v, f in _verb_frequencies(spot).items() if f >= _NEAR_BINARY_DROP_FREQ]


def _best_ev_alternative_verb(
    spot: PostflopSpot, dominant_verb: str
) -> str | None:
    """The highest-EV verb OTHER than the dominant one, from the spot's
    per-action EVs (hand-specific when the solve carries combo EVs, else
    the range-mean). A multi-size verb is ranked by its best size. None
    when the solve ships no EVs -- the caller keeps its old fallback."""
    from pipeline.postflop.spot_sampler import (  # noqa: PLC0415 - cycle guard
        spot_action_evs_bb,
    )

    evs = spot_action_evs_bb(spot)
    if not evs:
        return None
    verb_of = {a.label: a.verb for a in spot.node.actions}
    best_verb: str | None = None
    best_ev: float | None = None
    for label, ev in evs.items():
        verb = verb_of.get(label)
        if verb is None or verb == dominant_verb:
            continue
        if best_ev is None or ev > best_ev:
            best_verb, best_ev = verb, ev
    return best_verb


def _collapsed_spectrum_options(
    spot: PostflopSpot, verbs: list[str] | None = None
) -> tuple[list[str], str]:
    """Collapse a spot onto a binary spectrum over TWO verbs.

    Used two ways: a 2-verb MULTI-size spot (Check + Bet 33%/50%/67% -> the
    Check-vs-Bet spectrum), and a near-binary 3+-verb spot where only two verbs
    are live (``verbs`` names them, e.g. Fold + Call when Raise is a 2% sliver).
    The bet size is dropped from the OPTION (the LLM still receives the real size
    in its data block, and ~99% of hands commit to a single size, so the option
    loses essentially nothing). The correct answer is the dominant verb FAMILY's
    claim -- "Always Bet" when the family is ~pure, else "Mostly Bet".
    """
    actions = sorted(spot.node.actions, key=_aggression_key)
    by_verb: dict[str, list[NodeAction]] = {}
    for a in actions:
        by_verb.setdefault(a.verb, []).append(a)
    chosen = verbs if verbs is not None else list(by_verb)
    passive_v, aggr_v = sorted(chosen, key=lambda v: _VERB_RANK.get(v, 9))[:2]

    def _family(verb: str) -> str:
        # A single action keeps its own label; a multi-size verb collapses to
        # the capitalised verb ("Bet" / "Raise").
        acts = by_verb[verb]
        return acts[0].label if len(acts) == 1 else verb.capitalize()

    passive, aggr = _family(passive_v), _family(aggr_v)
    options = [f"Always {passive}", f"Mostly {passive}", f"Mostly {aggr}", f"Always {aggr}"]
    aggr_freq = sum(spot.action_frequencies.get(a.label, 0.0) for a in by_verb[aggr_v])
    pass_freq = sum(spot.action_frequencies.get(a.label, 0.0) for a in by_verb[passive_v])
    dom, dom_freq = (aggr, aggr_freq) if aggr_freq >= pass_freq else (passive, pass_freq)
    prefix = "Always" if dom_freq >= PURE_THRESHOLD else "Mostly"
    return options, f"{prefix} {dom}"


def frequencies_for_options(
    action_frequencies: dict[str, float], options: list[str]
) -> dict[str, float]:
    """Solver frequency for each option's action, summing collapsed families.

    ``neutral_credit_options`` looks up each option (stripped of its
    ``Always``/``Mostly`` prefix) in a frequency map. For plain / single-size
    options that is just ``action_frequencies``; for a COLLAPSED family option
    (``"Bet"`` / ``"Raise"`` with the size dropped) the matching key is absent,
    so we sum every size of that verb (``"Bet"`` -> ``"Bet 33%" + "Bet 50%"``).
    Returns ``{action_label: frequency}`` keyed the way neutral-credit expects.
    """
    out: dict[str, float] = {}
    for opt in options:
        if not opt:
            continue
        action = opt
        for prefix in ("Always ", "Mostly "):
            if action.startswith(prefix):
                action = action[len(prefix):]
                break
        if action in out:
            continue
        if action in action_frequencies:
            out[action] = action_frequencies[action]
        else:  # collapsed verb family -> sum every size of that verb
            out[action] = sum(
                f for lbl, f in action_frequencies.items()
                if lbl == action or lbl.startswith(action + " ")
            )
    return out


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

    # The Always/Mostly spectrum: a 2-ACTION spot uses the two action labels; a
    # multi-SIZE spot with only two action TYPES (Check + Bet 33%/50%/67%)
    # COLLAPSES to its two verbs (Check vs Bet) and spectrums those -- a hand
    # commits to one size ~99% of the time, so the dropped size costs nothing
    # (the LLM still gets the real size in its data block).
    if resolved == "gto":
        if len(actions) == 2:  # noqa: PLR2004
            return _spectrum_options(spot, labels, dominant)
        n_verbs = len({a.verb for a in actions})
        # Multi-size collapse (Check vs Bet) is EXPLICIT-gto only; "auto" keeps
        # plain size labels on multi-size spots (its long-standing behaviour),
        # so turning on gto is a deliberate "I want the spectrum everywhere".
        if style == "gto" and n_verbs == 2:  # noqa: PLR2004
            return _collapsed_spectrum_options(spot)
        # Near-binary collapse: a 3+-VERB spot (Fold/Call/Raise) whose third+
        # verbs are GTO slivers is really a 2-verb decision -- spectrum the two
        # LIVE verbs instead of plain labels. Gated to 3+ verbs so it does NOT
        # touch a 2-verb multi-size spot (handled above, explicit-gto-only).
        # Applies under auto-resolved gto (a mixed facing-bet spot is the common
        # case), so it is NOT gated to explicit gto.
        if n_verbs >= 3:  # noqa: PLR2004
            live = _live_verbs(spot)
            if len(live) == 2:  # noqa: PLR2004
                return _collapsed_spectrum_options(spot, verbs=live)
            if len(live) == 1:
                # Pure / near-pure 3+-verb spot: every alternative sits at
                # ~0% frequency, so frequency can't rank them. STANDING RULE
                # (user, July 2026, mirrors preflop's EV-first secondary --
                # see pipeline.preflop.options._pick_gto_secondary): the
                # wrong-answer option is the SECOND-BEST action BY EV, the
                # genuinely most tempting mistake. Do not order any other
                # heuristic ahead of the EV ranking. Falls through to plain
                # labels only when the solve ships no per-action EVs.
                alt = _best_ev_alternative_verb(spot, live[0])
                if alt is not None:
                    return _collapsed_spectrum_options(
                        spot, verbs=[live[0], alt]
                    )
    return _plain_options(actions, labels, dominant)


__all__ = [
    "ANSWER_STYLES",
    "ANSWER_STYLE_FROM_RADIO_LABEL",
    "PURE_THRESHOLD",
    "build_options",
    "frequencies_for_options",
]
