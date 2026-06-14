"""Deterministic range examples: hands that lean toward the runner-up action.

NLHE port of :mod:`pipeline.plo.range_examples` ("the LLM never thinks about
poker"): when the solver's answer is, say, Fold at 93%, the instructive
contrast is WHICH hand classes in hero's own range at this node genuinely
prefer the runner-up action. Each class's action mix at the node is real
pack data, so it is computed here and handed to the LLM as a quotable list;
a prompt that names contrast hands without this fact would have to invent
them.

Selection is the borderline band: leaning clearly toward the other action
but NOT pure (a hand that always takes it is obvious and teaches nothing --
and hero's own hand can never qualify, since its dominant action is by
definition the other one). NLHE rendering is just the 169-class name
("AJo", "KQs", "77"), already player-facing and free of the suit-pattern
words that made the PLO version interact with the shape-claim audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.preflop.options import (
    _hero_raise_level,
    canonicalize_action_label,
    is_check_spot,
)
from pipeline.preflop.spot_sampler import _cached_parse_range_file
from pipeline.preflop_ranges import canonical_169_hand_classes

if TYPE_CHECKING:
    from pipeline.preflop.fact_extractor import PreflopFacts
    from pipeline.preflop.node_enumerator import PreflopDecisionNode

# The instructive borderline: leaning clearly toward the other action ...
_LEAN_MIN = 0.45
# ... but not so hard that the example is obvious ("100% frequency" hands out).
_LEAN_MAX = 0.90
# The example must actually show up at this node with real weight.
_MIN_PRESENCE = 0.05
_MAX_EXAMPLES = 3
# Above this lean the qualifier reads "mostly <verb>", below it "often <verb>".
_MOSTLY_FLOOR = 0.60
# Group floor: a BUCKET is only listed as "preferring the other action" when a
# real majority of it does (>= 50%), so "ace-broadways (call 46%)" -- which
# actually folds 54% -- never reads as preferring to call. (June 2026.)
_GROUP_LEAN_MIN = 0.50

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


def format_examples(
    rows: list[tuple[float, float, str]], verb: str
) -> list[str]:
    """Render example strings from ``(presence, lean_freq, hand_class)`` rows.

    Band-filters to the borderline (lean in [0.45, 0.90], presence >= 5%),
    scores by presence * lean, caps at 3. The qualifier keeps the mixing
    honest: "mostly <verb>" / "often <verb>", never "always".
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
    for _presence, lean, hand_class in scored[:_MAX_EXAMPLES]:
        qual = "mostly" if lean >= _MOSTLY_FLOOR else "often"
        out.append(f"{hand_class} ({qual} {verb})")
    return out


# Per-(node, option) memo: a batch revisits the same node for many hero
# hands, and the 169-class walk below re-reads each option's range file.
_memo: dict[tuple[tuple[str, ...], str], tuple[str, ...]] = {}
_MEMO_CAP = 256


def hands_leaning_to_option(
    node: PreflopDecisionNode, raw_label: str, verb: str
) -> tuple[str, ...]:
    """Example hand classes at ``node`` leaning toward ``raw_label``.

    Walks every class's (presence, conditional frequency) for the option --
    the same numbers :func:`pipeline.preflop.spot_sampler.sample_spot` uses,
    via the same range-file cache -- and renders the borderline band.
    """
    paths = tuple(str(opt.range_file.path) for opt in node.actions)
    key = (paths, raw_label)
    cached = _memo.get(key)
    if cached is not None:
        return cached

    weights = {opt.label: _cached_parse_range_file(str(opt.range_file.path))
               for opt in node.actions}
    target = weights.get(raw_label)
    if target is None:
        result: tuple[str, ...] = ()
    else:
        rows: list[tuple[float, float, str]] = []
        for hand_class in canonical_169_hand_classes():
            presence = sum(w.get(hand_class, 0.0) for w in weights.values())
            if presence < _MIN_PRESENCE:
                continue
            lean = target.get(hand_class, 0.0) / presence
            if _LEAN_MIN <= lean <= _LEAN_MAX:
                rows.append((presence, lean, hand_class))
        result = tuple(format_examples(rows, verb))

    if len(_memo) >= _MEMO_CAP:
        _memo.clear()
    _memo[key] = result
    return result


def leaning_groups_to_option(
    node: PreflopDecisionNode,
    raw_label: str,
    action_word: str,
    skip_bucket: str | None = None,
) -> tuple[str, ...]:
    """Hand-class BUCKETS in hero's range leaning toward ``raw_label``, each
    with the bucket's GROUP frequency for that action ("wheel aces (fold 64%)").

    ``skip_bucket`` drops one bucket (hero's OWN) so the contrast never
    describes the hand-type hero is holding -- "weak offsuit aces 3-bet" makes
    no sense as a contrast when hero's A5o is itself a weak offsuit ace that
    calls.

    Avoids the over-generalization trap of tagging one hand with its bucket: a
    bucket is described as a group ONLY when 2+ of its classes are present; a
    lone present class names the specific hand ("A8s (fold 71%)"). The ``other``
    catch-all is skipped (no meaningful type to describe). The group frequency
    is combo-weighted over the bucket's present classes (AKo's 12 combos count
    3x a suited hand's), so it is the bucket's true average -- "your wheel aces
    fold 64%" is then a measured fact, not extrapolated from one example.
    """
    from pipeline.preflop.hand_categories import (  # noqa: PLC0415
        CATEGORY_LABEL_PLURAL,
        OTHER,
        categorize_hand_class,
        combos_in_class,
    )

    weights = {opt.label: _cached_parse_range_file(str(opt.range_file.path))
               for opt in node.actions}
    target = weights.get(raw_label)
    if target is None:
        return ()
    # Per bucket: combos taking raw_label, combos present, and each present
    # member class with its own lean (for the single-member guardrail).
    runner: dict[str, float] = {}
    present: dict[str, float] = {}
    members: dict[str, list[tuple[str, float]]] = {}
    for hand_class in canonical_169_hand_classes():
        presence = sum(w.get(hand_class, 0.0) for w in weights.values())
        if presence < _MIN_PRESENCE:
            continue
        bucket = categorize_hand_class(hand_class)
        if bucket == OTHER or bucket == skip_bucket:
            continue  # junk, or hero's OWN bucket (would contradict the hand)
        combos = combos_in_class(hand_class)
        runner[bucket] = runner.get(bucket, 0.0) + combos * target.get(hand_class, 0.0)
        present[bucket] = present.get(bucket, 0.0) + combos * presence
        members.setdefault(bucket, []).append(
            (hand_class, target.get(hand_class, 0.0) / presence)
        )

    rows: list[tuple[float, str, int]] = []  # (presence_score, label, pct)
    for bucket, present_combos in present.items():
        if present_combos <= 0:
            continue
        agg = runner[bucket] / present_combos
        if not (_GROUP_LEAN_MIN <= agg <= _LEAN_MAX):
            continue
        if len(members[bucket]) >= 2:
            label = CATEGORY_LABEL_PLURAL.get(bucket, bucket.replace("_", " "))
            pct = round(agg * 100)
        else:  # one class present -> name the hand (agg == that hand's lean)
            label, lean = members[bucket][0]
            pct = round(lean * 100)
        rows.append((present_combos, label, pct))
    rows.sort(key=lambda r: -r[0])
    return tuple(
        f"{label} ({action_word} {pct}%)" for _score, label, pct in rows[:_MAX_EXAMPLES]
    )


def leaning_examples_for_spot(facts: PreflopFacts) -> dict[str, object] | None:
    """The data-block fact for one spot, or ``None`` when nothing qualifies.

    Shape: ``{"action": "<display label>", "hands": ["wheel aces (fold 64%)",
    "A8s (fold 71%)", ...]}`` -- the runner-up option by hero's own
    frequencies, plus up to 3 hand-class BUCKETS in hero's range that lean
    toward it, each with the bucket's group frequency. Buckets with a single
    present class name that hand instead (no false generalization).
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
    # Drop hero's OWN bucket: a contrast like "weak offsuit aces (3-bet 60%)"
    # contradicts the hand in question when hero holds A5o (a weak offsuit ace
    # that calls). The contrast is about OTHER hand-types in the range.
    from pipeline.preflop.hand_categories import (  # noqa: PLC0415
        categorize_hand_class,
    )

    hero_bucket = categorize_hand_class(facts.spot.hero_hand_class)
    hands = leaning_groups_to_option(
        facts.spot.node, runner_raw, display.lower(), skip_bucket=hero_bucket
    )
    if not hands:
        return None
    return {"action": display, "hands": list(hands)}


__all__ = [
    "format_examples",
    "hands_leaning_to_option",
    "leaning_examples_for_spot",
]
