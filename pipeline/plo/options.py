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
# "Always X" is the correct answer only for a literally-pure (100%) action;
# every mixed spot (even 99/1) is "Mostly X" correct, with "Always X" a neutral
# near-miss. Worthy spots top out at 99%, so they're always "Mostly". (June 2026.)
_PURE_STRATEGY_PREFIX = 0.9999
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


def answer_qualifier(dominant_freq: float) -> str:
    """The ``Always``/``Mostly`` qualifier the GTO option path renders for a
    dominant action at this solver frequency.

    SINGLE SOURCE OF TRUTH: delegates to :func:`_freq_prefix` -- the exact
    mapping :func:`build_options_gto` applies to the correct answer -- so a
    consumer (e.g. the 🎛️ fully-balanced qualifier axis) can never drift
    from the rendered option prefix. Returns ``""`` only for a dominant
    below 5% (never a worthy spot).
    """
    return _freq_prefix(dominant_freq)


def qualifier_axis_active(style: str) -> bool:
    """True when this answer style can render Always/Mostly qualifiers.

    ``gto`` always renders them; ``auto`` may (it falls to the GTO spectrum
    on mixed spots). ``basic`` never does, so the balanced qualifier axis
    must stay OFF there -- including it would change basic-style selection
    for a distinction the player can never see.
    """
    return style in ("gto", "auto")


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


def integer_percentages(strategy: dict[str, float]) -> dict[str, int]:
    """Largest-remainder integer percents summing to exactly 100.

    THE single allocation for every percentage surface of a strategy (the CSV
    ``action_frequencies`` column AND the SOLVER DATA ``action_strategy``
    block). They previously used two rounding paths (largest-remainder vs
    naive ``round()``), which disagree on exact-.5 boundaries -- a Call
    98.5/Fold 1.5 mix showed the player "Call: 99%" while the LLM (and the
    claim checker judging its prose) read "Call: 98%". Same fact, one number.
    Ties in remainder go to the higher-frequency action (stable sort), which
    preserves the CSV column's historical byte-exact output.
    """
    by_freq = sorted(strategy.items(), key=lambda kv: -kv[1])
    floors = [(label, int(v * 100), (v * 100) % 1) for label, v in by_freq]
    deficit = 100 - sum(floor for _, floor, _ in floors)
    bumps = {
        i
        for i, _ in sorted(enumerate(floors), key=lambda kv: -kv[1][2])[: max(deficit, 0)]
    }
    out = {
        label: floor + (1 if i in bumps else 0)
        for i, (label, floor, _) in enumerate(floors)
    }
    # HONESTY CLAMP (July 2026, user rule): never display 100%/0% unless
    # literally true. A 99.5/0.5 mix used to round to "Check: 100%, Raise: 0%"
    # while the correct answer stayed "Mostly Check" (the Always qualifier
    # needs a literally-pure action, _PURE_STRATEGY_PREFIX) -- the display
    # contradicted the answer key. Same purity test as the qualifier, so the
    # two surfaces can never disagree: a present-but-mixed action shows at
    # least 1, a mixed dominant at most 99; sums stay exactly 100.
    present_floor = 1.0 - _PURE_STRATEGY_PREFIX
    exact = dict(strategy)
    for label in out:
        if out[label] == 0 and exact[label] > present_floor:
            out[label] = 1
        elif out[label] == 100 and exact[label] < _PURE_STRATEGY_PREFIX:
            out[label] = 99
    overflow = sum(out.values()) - 100
    for label, _v in (by_freq if overflow > 0 else reversed(by_freq)):
        if overflow == 0:
            break
        if overflow > 0 and out[label] > 1 and exact[label] < _PURE_STRATEGY_PREFIX:
            take = min(overflow, out[label] - 1)
            out[label] -= take
            overflow -= take
        elif overflow < 0 and 0 < out[label] < 99:
            give = min(-overflow, 99 - out[label])
            out[label] += give
            overflow += give
    if overflow < 0:
        # Residual clamp deficit (Aug 2026 bugfix, found by the chart
        # export): a dominant in [0.9998, 0.9999) with EVERY other label a
        # sub-0.0001 sliver leaves out = {dominant: 99, others: 0} -- the
        # loop above finds nothing in (0, 99) to give the point to, and the
        # function returned a sum of 99. The sum-to-100 contract is hard
        # (the CSV cross-check audits it); showing 1 for a genuinely-taken
        # sliver is house-sanctioned (the clamp's own present-action rule),
        # so give the deficit to the largest present-but-zero labels.
        # overflow is -1 by construction (only one label can be clamped
        # from 100), but loop defensively.
        for label, v in by_freq:
            if overflow == 0:
                break
            if out[label] == 0 and v > 0.0:
                out[label] += 1
                overflow += 1
    if overflow > 0:
        # Mirror residual (same Aug 2026 wave): caller shares are usually
        # ratios (mass / total), so the exact values can sum to 1 + 1ulp.
        # At the purity boundary that yields a dominant that KEEPS 100
        # (exact >= _PURE_STRATEGY_PREFIX) while a sliver still clears the
        # present-floor promote to 1 -- sum 101, and the loop above cannot
        # take from a pure dominant. Demote the smallest promoted slivers
        # back to 0: the dominant's exact says "Always", so 100/0 is the
        # display consistent with the qualifier.
        for label, _v in reversed(by_freq):
            if overflow == 0:
                break
            if out[label] == 1:
                out[label] = 0
                overflow -= 1
    # INVARIANT: every return path sums to exactly 100 (aggregate asserts,
    # the batch cross-check, and the chart exports all rely on it).
    return out


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
    non-dominant action, preferring Fold on ties; None if none exists.

    When several non-fold actions tie at ~0% (a near-pure FOLD spot), break the
    tie by LEAST aggression rather than dict order -- the old ``tied[0]`` could
    surface All-in as a nonsense "Always All-in" option. (Mirrors the preflop
    fix; preflop additionally uses per-action EVs, which PLO's raw-labelled
    ``ev_by_action`` doesn't map cleanly onto here -- a possible refinement.)"""
    canonical = canonicalize_strategy(facts)
    candidates = {lbl: f for lbl, f in canonical.items() if lbl != dominant_label}
    if not candidates:
        return None
    max_freq = max(candidates.values())
    tied = [lbl for lbl, f in candidates.items() if f == max_freq]
    if len(tied) == 1:
        return tied[0]
    if "Fold" in tied:
        return "Fold"
    return min(tied, key=_action_aggression)


def _meaningful_canonical_actions(facts: PloFacts) -> list[tuple[str, float]]:
    """Canonical ``(label, freq)`` pairs, descending freq, >= 5%."""
    canonical = canonicalize_strategy(facts)
    ranked = sorted(canonical.items(), key=lambda kv: -kv[1])
    return [(lbl, f) for lbl, f in ranked if f >= _MIN_MEANINGFUL_FREQ]


def build_options_basic(facts: PloFacts) -> tuple[list[str], str]:
    """Bare action labels, one per meaningfully-played action (<= 4),
    displayed least-aggressive first. ``correct_answer`` is the canonical
    dominant action.

    Selection (WHICH actions make the cut when there are 5+) is by solver
    frequency; the DISPLAY order is always the aggression ladder
    (Fold < Check < Call < Raise < 3-bet < ... < All-in) -- team standing
    rule (July 2026): an option row must read least -> most aggressive, so
    "Fold · 4-bet · Call" (frequency order) can never ship.
    """
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

    # The correct answer always makes the cut (displaces the least-frequent
    # kept alternative when the menu is full).
    if canonical_correct not in kept:
        if len(kept) >= _MAX_OPTIONS:
            drop = next(
                lbl for lbl, _ in reversed(ordered) if lbl in kept and lbl != "Fold"
            )
            kept.discard(drop)
        kept.add(canonical_correct)

    # Display order: the aggression ladder, never frequency (label as the
    # tie-break keeps unknown labels deterministic).
    ordered_options = sorted(kept, key=lambda lbl: (_action_aggression(lbl), lbl))
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
    "answer_qualifier",
    "build_options",
    "build_options_auto",
    "build_options_basic",
    "build_options_gto",
    "canonicalize_action_label",
    "canonicalize_strategy",
    "is_check_spot",
    "qualifier_axis_active",
]
