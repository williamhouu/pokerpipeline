"""Deterministic range examples: hands that lean toward the runner-up action.

Layer-5/6 fact ("the LLM never thinks about poker"): when the solver's answer
is, say, Fold at 95%, the instructive contrast is WHICH hands in hero's own
range at this node genuinely prefer the runner-up action (the 5% Call). That
is real solver data, each hand's action mix at the node, so it is computed
here and handed to the LLM as a quotable list. A prompt that names contrast
hands without this fact would have to invent them, which is exactly the
failure mode the pipeline exists to prevent.

Selection is deliberately the borderline band: hands leaning clearly toward
the other action but NOT pure (a hand that always takes it is obvious and
teaches nothing -- and hero's own hand can never qualify, since its dominant
action is by definition the other one). Rendering is suit-pattern-light
("KQJT double-suited"), never specific suited cards, so the card-fabrication
audit (which rejects rank+suit-emoji mentions outside hero's hand) is never
triggered by quoting these.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from pipeline.plo.hand_model import classify_plo_hand
from pipeline.plo.hand_order import HAND_COUNT, cards_at
from pipeline.plo.options import (
    _hero_raise_level,
    canonicalize_action_label,
    is_check_spot,
)
from pipeline.plo.spot_sampler import _cached_node_values

if TYPE_CHECKING:
    from pipeline.plo.fact_extractor import PloFacts
    from pipeline.plo.node_enumerator import PloDecisionNode

# The instructive borderline: leaning clearly toward the other action ...
_LEAN_MIN = 0.45
# ... but not so hard that the example is obvious ("100% frequency" hands out).
_LEAN_MAX = 0.90
# The example must actually show up at this node with real weight.
_MIN_PRESENCE = 0.05
_MAX_EXAMPLES = 3
# Above this lean the qualifier reads "mostly <verb>", below it "often <verb>".
_MOSTLY_FLOOR = 0.60

_RANK_DESC = "AKQJT98765432"
_SUIT_WORD = {
    "double_suited": "double-suited",
    "single_suited": "single-suited",
    "rainbow": "rainbow",
    "three_suited": "three-suited",
    "monotone": "monotone",
}
_VERB = {
    "Fold": "folds",
    "Call": "calls",
    "Check": "checks",
    "Raise": "raises",
    "3-bet": "3-bets",
    "4-bet": "4-bets",
    "5-bet": "5-bets",
    "All-in": "moves all-in",
}


def render_hand_class(cards: tuple[str, ...]) -> str:
    """Compact player-facing name: ranks high to low + suit pattern.

    ("Kc", "Kd", "4c", "4d") -> "KK44 double-suited". No specific suits, so
    quoting it in prose can never trip the card-fabrication audit.
    """
    ranks = sorted((c[0].upper() for c in cards), key=_RANK_DESC.index)
    return f"{''.join(ranks)} {_SUIT_WORD[classify_plo_hand(cards).suit_pattern]}"


def format_examples(
    rows: Iterable[tuple[float, float, tuple[str, ...]]], verb: str
) -> list[str]:
    """Render the example strings from ``(presence, lean_freq, cards)`` rows.

    Band-filters to the borderline (lean in [0.45, 0.90], presence >= 5%),
    scores by presence * lean, dedupes by rendered name (suit-isomorphic
    neighbours collapse), caps at 3. The qualifier keeps the mixing honest:
    "mostly <verb>" / "often <verb>", never "always".
    """
    scored = sorted(
        (
            r
            for r in rows
            if r[0] >= _MIN_PRESENCE and _LEAN_MIN <= r[1] <= _LEAN_MAX
        ),
        key=lambda r: -(r[0] * r[1]),
    )
    out: list[str] = []
    seen: set[str] = set()
    for _presence, lean, cards in scored:
        name = render_hand_class(cards)
        if name in seen:
            continue
        seen.add(name)
        qual = "mostly" if lean >= _MOSTLY_FLOOR else "often"
        out.append(f"{name} ({qual} {verb})")
        if len(out) == _MAX_EXAMPLES:
            break
    return out


# Per-(node, option) memo: a batch's round-robin revisits the same node for
# many hero hands, and the 16k-hand walk below is the expensive part.
_memo: dict[tuple[str, str], tuple[str, ...]] = {}
_MEMO_CAP = 256


def hands_leaning_to_option(
    node: PloDecisionNode, raw_label: str, verb: str
) -> tuple[str, ...]:
    """Example hands at ``node`` leaning toward the ``raw_label`` action.

    Walks every hand's (presence, conditional frequency) for the option --
    the same numbers :func:`pipeline.plo.spot_sampler.sample_plo_spot` uses,
    via the same .rng cache -- and renders the borderline band.
    """
    key = (node.node_id, raw_label)
    cached = _memo.get(key)
    if cached is not None:
        return cached

    values = {opt.label: _cached_node_values(str(opt.path)) for opt in node.actions}
    target = values.get(raw_label)
    if target is None:
        result: tuple[str, ...] = ()
    else:

        def _rows() -> Iterable[tuple[float, float, tuple[str, ...]]]:
            for i in range(HAND_COUNT):
                presence = sum(v[i][0] for v in values.values())
                if presence < _MIN_PRESENCE:
                    continue
                lean = target[i][0] / presence
                if _LEAN_MIN <= lean <= _LEAN_MAX:
                    yield presence, lean, tuple(cards_at(i))

        result = tuple(format_examples(_rows(), verb))

    if len(_memo) >= _MEMO_CAP:
        _memo.clear()
    _memo[key] = result
    return result


def leaning_examples_for_spot(facts: PloFacts) -> dict[str, object] | None:
    """The data-block fact for one spot, or ``None`` when nothing qualifies.

    Shape: ``{"action": "<display label>", "hands": ["KQJT double-suited
    (mostly calls)", ...]}`` -- the runner-up option by hero's own frequencies
    plus up to 3 borderline hands from hero's range that lean toward it.
    """
    freqs = facts.spot.action_frequencies
    if len(freqs) < 2:  # noqa: PLR2004 -- no second option, no contrast
        return None
    dominant_raw = max(freqs, key=lambda label: freqs[label])
    rest = {label: f for label, f in freqs.items() if label != dominant_raw}
    if not rest:
        return None
    runner_raw = max(rest, key=lambda label: rest[label])
    display = canonicalize_action_label(
        runner_raw,
        raise_level=_hero_raise_level(facts),
        check_spot=is_check_spot(facts),
    )
    verb = _VERB.get(display, f"takes the {display.lower()}")
    hands = hands_leaning_to_option(facts.spot.node, runner_raw, verb)
    if not hands:
        return None
    return {"action": display, "hands": list(hands)}


__all__ = [
    "format_examples",
    "hands_leaning_to_option",
    "leaning_examples_for_spot",
    "render_hand_class",
]
