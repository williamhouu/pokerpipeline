"""Deterministic "why this action" factor breakdown for a preflop spot.

A prototype of the app's "Why this action?" view: a list of human-readable
FACTORS, each leaning toward one of the spot's two main actions, plus a
one-line summary. Every factor is computed by plain Python from the
solver's own facts (the concept tags, the archetype, equity, position) --
the LLM never touches it, same architecture rule as the rest of the
pipeline.

## What this DOES and does NOT claim

- **Factors**: the firing concept tags + a few scalar facts. These are
  already deterministic and validated, so the factor LIST is solid.
- **Direction**: each factor leans "aggressive" (toward the more
  aggressive of the spot's two main actions) or "cautious" (toward the
  less aggressive one). These directions are conservative, well-understood
  poker truths (a premium hand wants to build the pot; a dominated hand
  wants to fold), and they are tied to the solver-derived ARCHETYPE where
  possible. We do NOT claim solver-exact magnitudes -- ``strength`` is a
  coarse weak/moderate/strong bucket, the same honest stance as
  :mod:`pipeline.preflop.exploit` ("direction is deterministic, magnitude
  is not"). A future iteration can fit real per-factor weights on the
  generated solver corpus (each spot is labelled with the solver's action
  + every tag), which is the rigorous way to get the bar lengths.

The point of surfacing this deterministically -- rather than letting an
LLM narrate "why" -- is that a heuristic that can contradict the solver is
exactly the failure mode the pipeline exists to prevent. So the rules here
stay close to the tags and lean only where the direction is robust.

Two builders share one rule set:

* :func:`compute_why_factors` -- from a populated ``PreflopFacts`` (used at
  generation time / when facts are in hand).
* :func:`why_factors_from_row` -- from a written CSV row (used by the admin
  panel, which only has the row). Both feed the same :data:`_RULES`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.preflop.fact_extractor import PreflopFacts

# Aggression rank of an action label -- mirrors options._ACTION_AGGRESSION,
# kept local so the two main actions can be ordered into more/less aggressive.
_AGGRESSION: dict[str, int] = {
    "fold": 0,
    "check": 1,
    "call": 2,
    "raise": 3,
    "3-bet": 4,
    "4-bet": 5,
    "5-bet": 6,
    "all-in": 7,
}


def _aggression(label: str) -> int:
    if not label:
        return -1
    return _AGGRESSION.get(label.split()[0].lower().strip(".,;:!?\"'()[]"), 99)


# Archetype -> a one-line "primary reason". The archetype is the solver's
# computed role for the hand and always names the answer (open_for_value =>
# the raise, fold_dominated => the fold, call_for_implied_odds => the call),
# so the "Solver's read" factor always favors the dominant action -- it ties
# the headline straight to the solver-derived frame, not a guess.
_ARCHETYPE_REASON: dict[str, str] = {
    "open_for_value": "a clear opening hand worth raising for value",
    "3bet_for_value": "strong enough to 3-bet for value",
    "4bet_for_value": "premium enough to 4-bet for value",
    "5bet_for_value": "a top hand happy to get it in",
    "squeeze_for_value": "a strong squeeze for value over the callers",
    "all_in_for_value": "strong enough to stack off",
    "call_for_value": "strong, but flatting keeps villain's range wide",
    "3bet_as_bluff": "a 3-bet bluff using fold equity and blockers",
    "4bet_as_bluff": "a 4-bet bluff leaning on blockers",
    "5bet_as_bluff": "a 5-bet bluff",
    "squeeze_as_bluff": "a squeeze bluff over the open and callers",
    "all_in_as_bluff": "a jam bluff with fold equity",
    "fold_dominated": "dominated by the strong part of villain's range",
    "fold_pot_odds": "the price isn't enough for this hand's equity",
    "fold_outranged": "outranged in this spot",
    "call_for_implied_odds": "a speculative call playing for implied odds",
    "sb_complete": "completing cheaply from the small blind",
    "call_allin": "a pot-odds call of the jam",
    "bb_check": "a free check in an unraised pot",
}


@dataclass(frozen=True)
class WhyInputs:
    """The deterministic signals the factor rules read.

    Built either from ``PreflopFacts`` or from a CSV row, so the rules are
    written once. ``in_position`` is ``None`` when unknown (e.g. an open
    spot with no villain to be in/out of position against).
    """

    dominant_action: str           # canonical label of the solver's top action
    alternative_action: str        # canonical label of the main alternative
    tags: frozenset[str]           # firing concept tags
    archetype: str
    hero_equity: float | None      # vs villain's range, in [0, 1]
    break_even_equity: float | None  # pot-odds threshold to call, in [0, 1]
    in_position: bool | None       # True = IP, False = OOP, None = unknown


@dataclass(frozen=True)
class WhyFactor:
    """One factor and which action it argues for."""

    label: str       # short name, e.g. "Premium hand"
    favors: str      # the actual action label this factor leans toward
    strength: int    # 1 = weak, 2 = moderate, 3 = strong (coarse, not exact)
    detail: str      # one-line plain-English reason


@dataclass(frozen=True)
class WhyBreakdown:
    """The full breakdown for one spot."""

    answer: str                     # the solver's dominant action
    alternative: str                # the main alternative action
    factors: tuple[WhyFactor, ...]  # each favors `answer` or `alternative`
    summary: str                    # deterministic one/two-line summary


# A rule reads the inputs and returns (label, lean, strength, detail) or None.
# `lean` is "aggressive" or "cautious"; it is resolved to the spot's actual
# more/less-aggressive action in `compute_breakdown`.
_RawFactor = tuple[str, str, int, str]
_Rule = Callable[[WhyInputs], "_RawFactor | None"]


def _has(inp: WhyInputs, *tags: str) -> bool:
    return any(t in inp.tags for t in tags)


# --- the factor rules --------------------------------------------------------
def _rule_archetype(inp: WhyInputs) -> _RawFactor | None:
    """The strategic frame the solver assigned -- the headline reason.

    Always favors the answer (lean ``"answer"``): the archetype is derived
    from the dominant action, so it can't argue against it.
    """
    reason = _ARCHETYPE_REASON.get(inp.archetype)
    if reason is None:
        return None
    return ("Solver's read", "answer", 3, f"This plays as {reason}.")


def _rule_premium(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "premium_pair", "premium_unpaired"):
        return (
            "Premium hand", "aggressive", 3,
            "A premium holding wants to build the pot, not play it small.",
        )
    return None


def _rule_dominated(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "dominated"):
        return (
            "Dominated", "cautious", 3,
            "Villain's range is weighted toward hands that have you crushed.",
        )
    return None


def _rule_equity_edge(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "equity_dominant"):
        return (
            "Big equity edge", "aggressive", 2,
            "You're a clear equity favorite over villain's range.",
        )
    if _has(inp, "coinflip"):
        return (
            "Roughly a coinflip", "cautious", 1,
            "You're about even on equity, so there's little edge to press.",
        )
    return None


def _rule_blockers(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "blocks_villain_top_value"):
        return (
            "Key blockers", "aggressive", 2,
            "Your cards remove some of villain's strongest continues, "
            "adding fold equity.",
        )
    return None


def _rule_speculative(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "suited_connector", "small_pair", "suited_ace"):
        return (
            "Speculative hand", "continue", 1,
            "A hand that plays for implied odds prefers to see a flop cheaply "
            "rather than bloat the pot or give up.",
        )
    return None


def _rule_position(inp: WhyInputs) -> _RawFactor | None:
    if inp.in_position is True:
        return (
            "In position", "aggressive", 1,
            "Acting last postflop lets you apply pressure and realize equity "
            "better.",
        )
    if inp.in_position is False:
        return (
            "Out of position", "cautious", 1,
            "Playing the rest of the hand first makes it harder to realize "
            "equity, so lean toward the lower-variance line.",
        )
    return None


def _rule_range_advantage(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "hero_range_advantage"):
        return (
            "Range advantage", "aggressive", 1,
            "Your range is stronger here, which supports the aggressive line.",
        )
    if _has(inp, "villain_range_advantage"):
        return (
            "Range disadvantage", "cautious", 1,
            "Villain's range is stronger here, so press less.",
        )
    return None


def _rule_stack(inp: WhyInputs) -> _RawFactor | None:
    if _has(inp, "short_stack"):
        return (
            "Short stack", "aggressive", 2,
            "Shallow stacks favor a polarized raise-or-fold game over flatting.",
        )
    if _has(inp, "deep_stack"):
        return (
            "Deep stack", "cautious", 1,
            "Deep stacks reward seeing flops with hands that can stack villain.",
        )
    return None


def _rule_pot_odds(inp: WhyInputs) -> _RawFactor | None:
    if inp.hero_equity is None or inp.break_even_equity is None:
        return None
    margin = inp.hero_equity - inp.break_even_equity
    if margin >= 0.04:
        return (
            "Good raw price", "continue", 2,
            f"You need about {inp.break_even_equity:.0%} and hold roughly "
            f"{inp.hero_equity:.0%} raw, so the pot odds look tempting "
            "(before equity realization).",
        )
    if margin <= -0.04:
        return (
            "Bad price", "fold", 2,
            f"You need about {inp.break_even_equity:.0%} but hold only roughly "
            f"{inp.hero_equity:.0%}.",
        )
    return None


_RULES: tuple[_Rule, ...] = (
    _rule_archetype,
    _rule_premium,
    _rule_dominated,
    _rule_equity_edge,
    _rule_blockers,
    _rule_speculative,
    _rule_position,
    _rule_range_advantage,
    _rule_stack,
    _rule_pot_odds,
)


# --- assembly ----------------------------------------------------------------
def _is_call(label: str) -> bool:
    return label.split()[0].lower() in ("call", "check") if label else False


def compute_breakdown(inp: WhyInputs) -> WhyBreakdown:
    """Run the rules and assemble the breakdown from a :class:`WhyInputs`.

    Each fired rule's abstract ``lean`` is resolved to a real option among
    the spot's two main actions:

      * ``"answer"``     -> the solver's dominant action (the archetype).
      * ``"aggressive"`` -> the more aggressive of the two.
      * ``"cautious"``   -> the less aggressive of the two.
      * ``"continue"``   -> the call/check action specifically (a good price
        or a speculative hand wants to *continue*, which is "call" whether
        the alternative is a raise or a fold). Dropped if neither action is
        a call/check.
      * ``"fold"``       -> the fold action specifically (a bad price wants
        to fold). Dropped if neither action is a fold.

    The last two keep price/speculative factors from pointing at the wrong
    option on a fold-vs-call axis, where the abstract more/less-aggressive
    poles would otherwise flip them.
    """
    answer = inp.dominant_action
    alt = inp.alternative_action or answer
    more_aggressive = answer if _aggression(answer) >= _aggression(alt) else alt
    less_aggressive = alt if more_aggressive == answer else answer
    call_action = next((a for a in (answer, alt) if _is_call(a)), None)
    fold_action = next(
        (a for a in (answer, alt) if a.split()[:1] == ["Fold"]), None
    )

    factors: list[WhyFactor] = []
    for rule in _RULES:
        raw = rule(inp)
        if raw is None:
            continue
        label, lean, strength, detail = raw
        if lean == "answer":
            favors: str | None = answer
        elif lean == "aggressive":
            favors = more_aggressive
        elif lean == "cautious":
            favors = less_aggressive
        elif lean == "continue":
            favors = call_action
        else:  # "fold"
            favors = fold_action
        if favors is None:  # the targeted action isn't on offer -> drop it
            continue
        factors.append(WhyFactor(label, favors, strength, detail))

    # Stable, useful order: strongest first, factors for the answer before
    # factors for the alternative (so the supporting case reads first).
    factors.sort(key=lambda f: (f.favors != answer, -f.strength, f.label))
    return WhyBreakdown(
        answer=answer,
        alternative=alt,
        factors=tuple(factors),
        summary=_summary(answer, alt, factors),
    )


def _summary(answer: str, alt: str, factors: list[WhyFactor]) -> str:
    """A deterministic one/two-line summary, screenshot-style."""
    for_answer = [f for f in factors if f.favors == answer]
    against = [f for f in factors if f.favors != answer]
    if not for_answer:
        return f"The solver's answer is {answer}."
    lead = ", ".join(f.label.lower() for f in for_answer[:2])
    out = f"{lead.capitalize()} push toward {answer}."
    if against and alt != answer:
        out += f" The main pull toward {alt} is {against[0].label.lower()}."
    return out


def render_why_text(breakdown: WhyBreakdown) -> str:
    """A plain-text rendering of the breakdown -- "written all out".

    Groups factors by which action they argue for, lists each with a coarse
    strength marker, and closes with the summary + the honest caveat. This
    is what the admin dropdown shows; the app will build its own UI from the
    structured :class:`WhyBreakdown`.
    """
    marks = {1: "·", 2: "··", 3: "···"}
    for_answer = [f for f in breakdown.factors if f.favors == breakdown.answer]
    against = [f for f in breakdown.factors if f.favors != breakdown.answer]
    lines: list[str] = []
    head = f"Why {breakdown.answer}"
    if breakdown.alternative != breakdown.answer:
        head += f" (vs {breakdown.alternative})"
    lines.append(head)
    if for_answer:
        lines.append(f"\nArguing for {breakdown.answer}:")
        lines += [
            f"  {marks[f.strength]} {f.label} — {f.detail}" for f in for_answer
        ]
    if against:
        lines.append(f"\nArguing for {breakdown.alternative}:")
        lines += [
            f"  {marks[f.strength]} {f.label} — {f.detail}" for f in against
        ]
    lines.append(f"\nSummary: {breakdown.summary}")
    lines.append(
        "\n(Directional heuristic: factors + lean are deterministic from the "
        "solver's facts; the strengths are coarse, not solver-exact "
        "magnitudes.)"
    )
    return "\n".join(lines)


# --- builders: from facts, and from a CSV row --------------------------------
def compute_why_factors(facts: PreflopFacts) -> WhyBreakdown:
    """Build the breakdown from a populated ``PreflopFacts``.

    Pulls the dominant + alternative actions from the canonical strategy,
    the firing tags from the concept tagger, and equity / position from the
    facts. Imports are local to keep this module free of heavy deps when
    only the pure rule engine is needed (e.g. the ``from_row`` path).
    """
    from pipeline.preflop.concept_tags import compute_concept_tags
    from pipeline.preflop.options import canonicalize_strategy
    from pipeline.preflop.position import hero_relative_position

    strategy = canonicalize_strategy(facts)
    ranked = sorted(strategy.items(), key=lambda kv: -kv[1])
    dominant = ranked[0][0] if ranked else ""
    alt = ranked[1][0] if len(ranked) > 1 else dominant

    rel = hero_relative_position(facts)
    in_pos = (
        True if rel == "In Position" else False if rel == "Out of Position" else None
    )
    inp = WhyInputs(
        dominant_action=dominant,
        alternative_action=alt,
        tags=frozenset(compute_concept_tags(facts)),
        archetype=facts.archetype,
        hero_equity=facts.hero_equity_vs_villain,
        break_even_equity=facts.break_even_equity,
        in_position=in_pos,
    )
    return compute_breakdown(inp)


def _pct_cell_to_float(cell: str) -> float | None:
    """Parse a "47%" CSV cell into 0.47; blank/garbage -> None."""
    cell = (cell or "").strip().rstrip("%").strip()
    if not cell:
        return None
    try:
        return float(cell) / 100.0
    except ValueError:
        return None


def why_factors_from_row(row: dict[str, str]) -> WhyBreakdown:
    """Build the breakdown from a written CSV row (the admin-panel path).

    Reads the same signals out of the row's columns -- ``action_frequencies``
    for the two main actions, ``concept_tags`` for the firing tags,
    ``archetype`` / ``hero_equity`` / ``pot_odds`` / ``Relative Position``
    for the rest -- so it agrees with :func:`compute_why_factors` without
    re-running the solver.
    """
    # action_frequencies is "Label: NN%, Label: NN%, ..." in descending freq.
    freqs = [
        p.rsplit(":", 1)[0].strip()
        for p in (row.get("action_frequencies") or "").split(",")
        if ":" in p
    ]
    dominant = freqs[0] if freqs else (row.get("Correct Answer") or "")
    alt = freqs[1] if len(freqs) > 1 else dominant
    tags = frozenset(
        t.strip() for t in (row.get("concept_tags") or "").split(",") if t.strip()
    )
    rel = (row.get("Relative Position") or "").strip()
    in_pos = (
        True if rel == "In Position" else False if rel == "Out of Position" else None
    )
    inp = WhyInputs(
        dominant_action=dominant,
        alternative_action=alt,
        tags=tags,
        archetype=(row.get("archetype") or "").strip(),
        hero_equity=_pct_cell_to_float(row.get("hero_equity", "")),
        break_even_equity=_pct_cell_to_float(row.get("pot_odds", "")),
        in_position=in_pos,
    )
    return compute_breakdown(inp)


__all__ = [
    "WhyBreakdown",
    "WhyFactor",
    "WhyInputs",
    "compute_breakdown",
    "compute_why_factors",
    "render_why_text",
    "why_factors_from_row",
]
