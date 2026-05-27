"""Layer 6 (preflop edition): Explanation Generator for preflop spots.

The preflop sibling of ``pipeline.explanation_generator``. Same role: take a
fully populated data block (here ``PreflopFacts`` instead of postflop's
``SpotData``) and ask the LLM to narrate the solver-derived facts in the
team's coaching voice. Output is the same six CSV columns -- option_1..4,
correct_answer, answer_explanation -- packaged in the shared
``GeneratedExplanation`` dataclass that Layer 8 consumes for both paths.

Per the brief's "the LLM never thinks about poker" principle, the model is
not allowed to reason strategically here either: the data block carries
hero's hand class, the per-action solver frequencies, villain identity
+ range stats, hero's equity vs villain's range, hero's range-vs-range
equity, blockers, and the precomputed strategic archetype. The LLM
translates those facts into prose.

What differs from postflop:

  * **Voice rules** drop all board-texture references; the hand-class
    naming convention is the 169-class system (``AKo``, ``T9s``, ``77``)
    rather than postflop's made-hand + draw breakdown.
  * **Archetype catalog** uses the 16 preflop archetypes from
    :func:`pipeline.preflop.fact_extractor.classify_archetype`
    (``open_for_value``, ``3bet_for_value``, ``squeeze_as_bluff``, ...).
  * **Option style detection** has no "sizing" branch -- preflop options
    embed their size in the label (``"Raise 308%"``), so the LLM emits the
    full label rather than a separate size string.
  * **Gold examples** are pulled from
    :func:`pipeline.preflop.gold_examples.load_preflop_gold_examples`,
    which filters the shared xlsx pool down to the preflop subset.
  * **No Layer 7 audit validators** wired in step 7a -- the postflop
    validators are bound to ``DecisionData`` (which preflop doesn't use),
    and we don't yet have a preflop audit corpus to know which strategic
    failures to defend against. Phase B will add preflop-specific
    validators after the first review batch surfaces concrete failure
    modes.

Usage::

    from pipeline.preflop.explanation_generator import generate_preflop_explanation
    explanation = generate_preflop_explanation(preflop_facts)

The Anthropic SDK call uses prompt caching on the system block and the
gold-example block, mirroring the postflop entry point -- the system
prompt is identical across every preflop generation, and the gold-example
block is identical within a batch.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pipeline.explanation_generator import (
    BANNED_LITERAL_PHRASES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ExplanationValidationError,
    GeneratedExplanation,
    frequency_to_verb_prefix,
    parse_response,
)
from pipeline.preflop.fact_extractor import PreflopFacts
from pipeline.preflop.gold_examples import load_preflop_gold_examples
from pipeline.preflop.grammars.types import PreflopActionType

logger = logging.getLogger(__name__)

# How many preflop gold examples to ship in the cached prompt block. Matches
# the postflop count (8). The preflop subset of the xlsx is smaller than the
# postflop subset, so the loader may return fewer -- we cap at whatever the
# loader produces.
GOLD_EXAMPLE_COUNT = 8

# Frequency threshold above which the spot's dominant action is treated as
# "clearly best" -- no Always/Mostly prefix needed. Mirrors the postflop
# threshold so the two paths share the same boundary between binary_action
# and frequency styles.
_BINARY_ACTION_FREQ_THRESHOLD = 0.80


# --- voice rules (preflop edition) -------------------------------------------
# Mirror the postflop ruleset's structure but strip board-texture language and
# add preflop-specific rules (range terminology, hand-class notation, position
# action history). Each rule's lead clause is asserted by the tests so future
# wording tweaks don't silently break the system prompt.
VOICE_RULES_PREFLOP: tuple[str, ...] = (
    # 1. Verdict first
    "Open with the verdict. The first sentence states the correct action "
    'plainly ("The best play is to call.", "You should 3-bet here.", '
    '"This is a clear fold."). Do not bury the answer.',
    # 2. Second-person coaching
    'Address the reader directly in second person ("you", "your hand", '
    '"you\'re holding"). Never refer to "the player" or "hero" -- the '
    "student IS hero.",
    # 3. Structure: verdict -> reasoning -> optional exploit caveat
    "Structure the body as: (a) verdict, (b) the reasoning (your hand "
    "class, position, prior action, villain's range, equity), (c) an "
    "optional one-line exploit note when the spot is population-sensitive. "
    "Skip (c) if it doesn't apply.",
    # 4. Length: 2-5 sentences
    "Keep the explanation between 2 and 5 sentences. Tight is better than "
    "thorough. If you can say it in 3 sentences, say it in 3.",
    # 5. Concrete naming (positions, hand classes -- NO board features)
    "Name things concretely: positions by abbreviation (UTG, HJ, CO, BTN, "
    "SB, BB), hands by 169-class label (AKo, JTs, 77, A2s). Never reference "
    '"the flop", "the turn", "the river", or any community cards -- '
    "the decision is preflop, no board has come yet.",
    # 6. Plain-language solver facts
    'Translate solver numbers into plain language. Say "BTN opens roughly '
    'the top half of hands here" rather than "BTN\'s range is 49.8% of '
    'combos"; say "your hand has good equity against this 3-bet range" '
    'rather than "hero_equity_vs_villain = 0.47".',
    # 7. No em dashes, no semicolons
    "Never use an em dash. Never use a semicolon. Use periods, commas, and "
    "short sentences. Do not use any phrase from the BANNED PHRASES list.",
    # 8. Confident instructive tone
    "Write with the confidence of a coach who has solved the spot. Do not "
    'hedge with "it might be", "perhaps", "in theory". State what '
    "the right play is and why, plainly.",
    # 9. Suit emojis for specific card references (matches postflop rule 9)
    "When referencing a SPECIFIC card or combo in prose, use the suit "
    "emoji form: A♠️, K❤️, T♦️, "
    "9♣️ (rank letter followed by the suit emoji, no space). "
    "Never use plain-text suit letters like 'Kh', 'Ad', or 'AsKs' in the "
    "answer explanation -- those are internal solver notation, not voice. "
    "Hand-class labels (AKo, JTs, 77) stay as plain text -- the emoji "
    "rule only applies when you actually name specific cards.",
    # 10. Name specific villain hand classes (preflop analogue of postflop rule 10)
    "When the explanation discusses villain's range, name 2 to 3 SPECIFIC "
    "hand classes from `villain_stats.top_combos` in the SOLVER DATA "
    "block. Use the 169-class labels (e.g. \"villain's 3-bet range here is "
    'dense with AA, KK, and AKs at full weight, plus some QQ and AKo"). '
    "Do not describe villain's range abstractly with only generic phrases "
    'like "value hands" or "top of range" -- anchor abstract phrases '
    "to actual hand classes from the data block.",
)


# --- preflop archetype catalog -----------------------------------------------
# Mirrors :func:`pipeline.preflop.fact_extractor.classify_archetype`. Adding a
# new archetype to the classifier requires adding its frame guidance here so
# the LLM has voice instructions for it. Each line is a single sentence the
# LLM can read fast in the system prompt.
PREFLOP_ARCHETYPE_GUIDANCE: dict[str, str] = {
    "open_for_value": (
        "Hero is first to act with a strong opening hand. Frame the "
        "explanation around hand strength + position: the hand is a clear "
        "open at this depth and position."
    ),
    "fold_outranged": (
        "Hero is first to act with a hand outside the opening range from "
        "this position. Frame the explanation around position discipline: "
        "the hand plays poorly out of position multi-way and isn't a profit "
        "open."
    ),
    "fold_dominated": (
        "Hero is facing a raise with a hand dominated by villain's range. "
        "Frame the explanation around domination: villain's continuing "
        "range crushes hero's hand class, and the implied odds don't "
        "compensate."
    ),
    "fold_pot_odds": (
        "Hero is facing a raise where the price is wrong despite some "
        "equity. Frame the explanation around pot odds + reverse implied "
        "odds: hero has enough raw equity but not enough realized equity "
        "out of position or against a polarized range."
    ),
    "call_for_value": (
        "Hero is calling a raise with a strong-but-not-3bet-worthy hand. "
        "Frame the explanation around playing a strong hand in position "
        "without bloating the pot vs a tight raising range."
    ),
    "call_for_implied_odds": (
        "Hero is calling a raise with a speculative hand whose value comes "
        "from hitting big on later streets. Frame the explanation around "
        "implied odds + position: the hand is a -EV preflop call that "
        "becomes profitable through realizing equity postflop."
    ),
    "all_in_for_value": (
        "Hero is shoving (or calling all-in) with a hand that wants action "
        "from villain's calling range. Rare preflop, usually short-stack "
        "or 5-bet pots. Frame around stack depth + value extraction."
    ),
    "all_in_as_bluff": (
        "Hero is shoving with fold equity rather than equity. Frame around "
        "polarization: hero's range here is nuts + bluff shoves, and the "
        "specific hand is one of the bluff shoves."
    ),
    "3bet_for_value": (
        "Hero is 3-betting a raise with a premium hand that dominates "
        "villain's continuing range. Frame the explanation around hand "
        "strength + isolating villain: 3-betting builds the pot with a "
        "hand that crushes the call-down range."
    ),
    "3bet_as_bluff": (
        "Hero is 3-betting with a hand that doesn't beat villain's value "
        "but has fold equity + good blockers + playability when called. "
        "Frame around polarization: 3-bet bluffs use ace-blockers, "
        "suited connectors, or hands one rank below the value 3-bet range."
    ),
    "4bet_for_value": (
        "Hero is 4-betting a 3-bet with a premium hand that dominates "
        "villain's 5-bet range. Frame around hand strength: only the very "
        "top of hero's range can 4-bet for stacks."
    ),
    "4bet_as_bluff": (
        "Hero is 4-betting with a hand that has fold equity vs villain's "
        "3-bet range. Frame around blockers (ace-blockers reduce villain's "
        "AA/AK combos) and game-theory balance with the value 4-bets."
    ),
    "5bet_for_value": (
        "Hero is 5-betting (or shoving over a 4-bet) with a premium hand. "
        "Frame around stack depth + hand strength: this is a value "
        "all-in, hero wants villain's continuing range to call."
    ),
    "5bet_as_bluff": (
        "Hero is shoving a 5-bet as a bluff. Frame around blockers + "
        "polarization: the hand has key blockers (e.g. an ace) and the "
        "shove either folds out villain's mid-strength 4-bet range or "
        "runs into AA/KK."
    ),
    "squeeze_for_value": (
        "Hero is raising over an open + at least one call, with a strong "
        "hand. Frame around dead money + isolation: the open + call(s) "
        "added dead money to the pot, and hero's hand wants to play heads-"
        "up against villain's continuing range."
    ),
    "squeeze_as_bluff": (
        "Hero is squeezing with a hand that has fold equity rather than "
        "showdown value. Frame around dead money + polarization: the open + "
        "call(s) create attractive dead money, hero's hand has blockers + "
        "playability when called."
    ),
}


# --- option-style detection (preflop) ----------------------------------------
def _detect_option_style_preflop(facts: PreflopFacts) -> str:
    """Pick the option style for a preflop spot.

    Returns one of:

      * ``"frequency"`` -- the spot is meaningfully mixed (the dominant
        action's frequency is below ``_BINARY_ACTION_FREQ_THRESHOLD``).
        Options use the Always/Mostly template with full action labels
        (e.g. ``"Always Fold"``, ``"Mostly Raise 308%"``).
      * ``"binary_action"`` -- a single action dominates. Options are the
        full action labels Pio offers (``"Fold"``, ``"Call"``, ``"Raise
        308%"``); no prefix.

    No "sizing" branch -- preflop options embed their size in the action
    label itself (multiple raise sizes at the same node show up as
    distinct labels), so the LLM emits whatever label Pio supplies.
    """
    if facts.spot.dominant_frequency >= _BINARY_ACTION_FREQ_THRESHOLD:
        return "binary_action"
    return "frequency"


def _expected_correct_prefix_preflop(facts: PreflopFacts) -> str | None:
    """The deterministic correct_answer prefix for a frequency-style spot.

    Returns None for binary_action spots (bare action label, no prefix).
    Mirrors ``pipeline.explanation_generator._expected_correct_prefix``
    but operates on PreflopFacts.
    """
    if _detect_option_style_preflop(facts) != "frequency":
        return None
    return frequency_to_verb_prefix(facts.spot.dominant_frequency)


# --- prompt assembly ---------------------------------------------------------
def _format_voice_rules() -> str:
    return "\n".join(f"{i + 1}. {rule}" for i, rule in enumerate(VOICE_RULES_PREFLOP))


def _format_banned_phrases() -> str:
    return ", ".join(f'"{p}"' for p in BANNED_LITERAL_PHRASES)


def _format_archetype_catalog() -> str:
    parts = [
        f"  - {name}: {guide}" for name, guide in PREFLOP_ARCHETYPE_GUIDANCE.items()
    ]
    return "\n".join(parts)


def build_preflop_system_prompt() -> str:
    """The static preflop system prompt: voice rules, archetype catalog,
    banned phrases, frequency-prefix rule, output schema.

    Stable across every preflop generation, so the SDK call marks it
    cacheable.
    """
    return (
        "You are the explanation generator for a poker training app. You "
        "translate solver-derived facts into short, on-voice coaching "
        "explanations for PREFLOP decisions. You never reason about poker "
        "yourself: every strategic claim in your output must be supported "
        "by a field in the SOLVER DATA block the user gives you.\n\n"
        "VOICE RULES (all ten apply to every output):\n"
        f"{_format_voice_rules()}\n\n"
        "STRATEGIC ARCHETYPES. The data block carries a `archetype` field. "
        "It is the STRATEGIC FRAME your explanation must be built around -- "
        "the action itself is in `dominant_action`. The 16 preflop "
        "archetypes and the frame each one demands:\n"
        f"{_format_archetype_catalog()}\n\n"
        f"BANNED PHRASES (never appear in any output field): {_format_banned_phrases()}.\n\n"
        "DETERMINISTIC FREQUENCY PREFIX MAPPING: When a frequency-style "
        "option-style instruction specifies a required prefix for "
        "correct_answer, use it exactly. The prefix is computed in Python "
        "from Pio's frequency of the dominant action:\n"
        '  >= 95%   -> "Always"\n'
        '  5-95%    -> "Mostly"\n'
        "  < 5%     -> the action is not played; do not emit a prefix.\n"
        'Standalone "Sometimes X" and "Rarely X" labels are BANNED '
        "(per Apr-2026 review: ambiguous to players). Use composite labels "
        'like "Mostly call, sometimes raise" only when explicitly '
        "instructed by the option-style block.\n\n"
        "OUTPUT FORMAT: respond with a single JSON object and nothing else. "
        "No prose before or after, no markdown fences. The object has "
        "exactly these six keys, all string-valued:\n"
        "  option_1, option_2, option_3, option_4, correct_answer, "
        "answer_explanation\n"
        'Empty options must be empty strings (""), not null. The '
        "correct_answer string must equal one of option_1..option_4 "
        "character-for-character; the consumer pairs them by exact string "
        "match."
    )


def _option_style_instruction_preflop(style: str, facts: PreflopFacts) -> str:
    """The natural-language option-style instruction for a preflop spot."""
    action_labels = sorted(facts.spot.action_frequencies.keys())
    labels_repr = ", ".join(repr(label) for label in action_labels)
    if style == "binary_action":
        return (
            "OPTION STYLE: binary action. The decision has a clear best "
            "action (dominant frequency >= 80%). Each option must be one "
            f"of Pio's offered action labels: {labels_repr}. Use 2 to 4 "
            "options matching the actions the solver offers; leave unused "
            "options as empty strings. correct_answer is the bare action "
            "label of the dominant action -- no prefix."
        )
    # Frequency style.
    prefix = _expected_correct_prefix_preflop(facts) or "Mostly"
    dominant = facts.spot.dominant_action
    # The two highest-frequency action labels Pio plays at >= 5%.
    ranked = sorted(facts.spot.action_frequencies.items(), key=lambda kv: -kv[1])
    meaningful = [(label, f) for label, f in ranked if f >= 0.05]
    if len(meaningful) >= 2:
        verb_a, verb_b = meaningful[0][0], meaningful[1][0]
    else:
        # Degenerate: only one action has meaningful frequency. Fall back to
        # the binary_action shape -- shouldn't reach here because the style
        # detector already routed those, but defensive.
        verb_a, verb_b = dominant, ""
    return (
        "OPTION STYLE: frequency. The solver mixes between two actions. "
        'Use exactly four options in this template: "Always <action A>", '
        '"Mostly <action A>", "Mostly <action B>", "Always <action '
        'B>", where action A is the action Pio plays more often. Both '
        'middle options use "Mostly" -- standalone "Sometimes X" / '
        '"Rarely X" labels are banned.\n\n'
        f'HARD CONSTRAINT -- correct_answer must equal exactly "{prefix} '
        f"{verb_a}\" where {verb_a!r} is Pio's dominant action label "
        f"verbatim. The prefix {prefix!r} is computed deterministically by "
        "Python from Pio's frequency; do NOT substitute your own "
        f"judgement. Action A = {verb_a!r}; action B = {verb_b!r}. The "
        "option set MUST include both of these labels; do not drop either "
        "in favour of a more elegant template."
    )


def _question_framing_preflop(facts: PreflopFacts) -> str:
    """The plain-English description of what's being decided at this spot.

    Mirrors the postflop framing but with preflop-specific fields: hand
    class, prior preflop action history, archetype.
    """
    spot = facts.spot
    node = spot.node
    hero_pos = node.actor
    villain_pos = (
        facts.villain_stats.position if facts.villain_stats else "no specific villain"
    )
    actions = (
        ", ".join(sorted(spot.action_frequencies.keys())) or "(no actions recorded)"
    )
    # Render prior history as "UTG opens 60%, HJ folds, CO raises 77%".
    history_parts: list[str] = []
    for entry in node.history_before:
        if entry.action_type is PreflopActionType.RAISE:
            verb = f"raises {entry.raise_size_pct:g}%"
        elif entry.action_type is PreflopActionType.CALL:
            verb = "calls"
        elif entry.action_type is PreflopActionType.FOLD:
            verb = "folds"
        elif entry.action_type is PreflopActionType.ALL_IN:
            verb = "shoves all-in"
        else:  # pragma: no cover
            verb = entry.action_type.value
        history_parts.append(f"{entry.position} {verb}")
    history_str = ", ".join(history_parts) or "no prior action (hero is first to act)"

    framing = (
        f"Stage: preflop. Hero ({hero_pos}) is holding "
        f"{spot.hero_hand_class} ({spot.hero_card_combo}). Prior action: "
        f"{history_str}. Decision-point villain: {villain_pos}. "
        f"Available actions in the solver: {actions}. The solver-correct "
        f'action is "{spot.dominant_action}" '
        f"(frequency {spot.dominant_frequency:.0%}). The explanation must "
        f"justify exactly that action."
    )
    archetype = facts.archetype
    if archetype and archetype in PREFLOP_ARCHETYPE_GUIDANCE:
        framing += (
            f"\n\nRECOMMENDED-ACTION ARCHETYPE: {archetype}.\n"
            f"{PREFLOP_ARCHETYPE_GUIDANCE[archetype]}"
        )
    return framing


def _compute_concept_tags_for_prompt(facts: PreflopFacts) -> list[str]:
    """The firing concept tags for this spot. Imported lazily so the
    concept-tag module isn't required for postflop test fixtures."""
    from pipeline.preflop.concept_tags import compute_concept_tags  # noqa: PLC0415

    return compute_concept_tags(facts)


def _trim_facts_for_prompt(facts: PreflopFacts) -> dict[str, Any]:
    """A compact JSON-ready view of PreflopFacts for the user prompt.

    Strips internal solver objects (the full PreflopDecisionNode + all its
    range-file paths) and zero/empty fields to keep the prompt size sane.
    What the LLM sees:

      * hand class + specific card combo (string forms only)
      * per-action frequencies (the strategy)
      * dominant action + frequency
      * villain identity + range stats (combo count, top hand classes)
      * hero equity vs villain range
      * hero's range equity vs villain range
      * blocker counts grouped by hand class
      * archetype label
    """
    spot = facts.spot
    out: dict[str, Any] = {
        "hand_class": spot.hero_hand_class,
        "hero_card_combo": spot.hero_card_combo,
        "actor": spot.node.actor,
        "action_frequencies": {
            label: round(freq, 4)
            for label, freq in spot.action_frequencies.items()
            if freq > 0
        },
        "dominant_action": spot.dominant_action,
        "dominant_frequency": round(spot.dominant_frequency, 4),
        "archetype": facts.archetype,
        # Concept tags from pipeline.preflop.concept_tags -- the firing
        # tags for this spot. Layer 6 uses them to anchor the explanation
        # in specific strategic concepts (e.g. "ace_blocker" makes
        # 3-bet bluff prose more concrete).
        "concept_tags": _compute_concept_tags_for_prompt(facts),
    }
    if facts.villain_stats is not None:
        v = facts.villain_stats
        out["villain_stats"] = {
            "position": v.position,
            "action_label": v.action_label,
            "weighted_combo_count": round(v.weighted_combo_count, 2),
            "pct_of_dealt_hands": round(v.pct_of_dealt_hands, 2),
            "top_combos": [
                {"hand_class": hand_class, "weight": round(weight, 4)}
                for hand_class, weight in v.top_combos
            ],
        }
    if facts.hero_equity_vs_villain is not None:
        out["hero_equity_vs_villain"] = round(facts.hero_equity_vs_villain, 4)
        out["hero_equity_runouts_used"] = facts.hero_equity_runouts_used
    if facts.hero_range_equity_vs_villain is not None:
        out["hero_range_equity_vs_villain"] = round(
            facts.hero_range_equity_vs_villain, 4
        )
    if facts.blockers:
        # Sort by count desc for readability; the LLM tends to cite the
        # top blockers first.
        out["blockers"] = dict(sorted(facts.blockers.items(), key=lambda kv: -kv[1]))
    # Render the prior history as a list of structured dicts -- the LLM
    # gets the framing string for prose and the structured form for any
    # references it wants to make to specific actors.
    out["history_before"] = [asdict(entry) for entry in spot.node.history_before]
    return out


def _format_gold_examples(
    examples: list[dict[str, Any]] | tuple[dict[str, Any], ...],
) -> str:
    """Render the gold-example block for the user prompt."""
    parts = []
    for index, ex in enumerate(examples, start=1):
        options = [
            ex.get("option 1", ""),
            ex.get("option 2", ""),
            ex.get("option 3", ""),
            ex.get("option 4", ""),
        ]
        parts.append(
            f"--- GOLD EXAMPLE {index} (preflop) ---\n"
            f"Question: {ex.get('Question', '')}\n"
            f"Options: {options}\n"
            f"Correct Answer: {ex.get('Correct Answer', '')}\n"
            f"Answer Explanation:\n{ex.get('Answer Explanation', '')}"
        )
    return "\n\n".join(parts)


def build_preflop_user_prompt(
    facts: PreflopFacts,
    gold_examples: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    style: str,
) -> str:
    """Full per-call user prompt: gold examples + framing + solver data.

    Public flat form -- the SDK call uses ``_build_messages_payload`` which
    splits it into a cached gold-example block and a live block, but tests
    and ad-hoc callers can use this combined form.
    """
    return (
        "Read these gold examples to lock in the voice. Imitate the cadence, "
        "verdict-first structure, and concrete naming. Do not copy phrasing.\n\n"
        f"{_format_gold_examples(gold_examples)}\n\n"
        "=== NEW QUESTION TO WRITE ===\n\n"
        f"{_question_framing_preflop(facts)}\n\n"
        f"{_option_style_instruction_preflop(style, facts)}\n\n"
        "SOLVER DATA (every strategic claim in your output must trace back "
        "to a field below; do not invent equity numbers or range claims):\n"
        f"{json.dumps(_trim_facts_for_prompt(facts), indent=2, default=str)}\n\n"
        "Now write the JSON object."
    )


def _build_messages_payload(
    facts: PreflopFacts,
    gold_examples: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    style: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The ``(system, messages)`` payload for ``client.messages.create``.

    Splits the user prompt into a cached gold-example block and a live
    block so the Anthropic prompt cache hits across thousands of
    generations.
    """
    system = [
        {
            "type": "text",
            "text": build_preflop_system_prompt(),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    cached_block = (
        "Read these gold examples to lock in the voice. Imitate the cadence, "
        "verdict-first structure, and concrete naming. Do not copy phrasing.\n\n"
        f"{_format_gold_examples(gold_examples)}"
    )
    live_block = (
        "\n\n=== NEW QUESTION TO WRITE ===\n\n"
        f"{_question_framing_preflop(facts)}\n\n"
        f"{_option_style_instruction_preflop(style, facts)}\n\n"
        "SOLVER DATA (every strategic claim in your output must trace back "
        "to a field below; do not invent equity numbers or range claims):\n"
        f"{json.dumps(_trim_facts_for_prompt(facts), indent=2, default=str)}\n\n"
        "Now write the JSON object."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_block,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": live_block},
            ],
        }
    ]
    return system, messages


# --- response validation (structural; Layer 7 audit deferred to Phase B) -----
def _validate(explanation: GeneratedExplanation) -> str | None:
    """Return None if structurally valid, else a short error message.

    Same shape as the postflop checker -- the four invariants the parser
    can't enforce on its own:

      * at least one non-empty option
      * correct_answer non-empty
      * correct_answer matches one of the options exactly
      * answer_explanation non-empty

    TODO(phase-b): add preflop-specific Layer 7 audit validators (verb
    correctness, archetype-consistency, frequency-prefix correctness)
    once the first review batch surfaces concrete failure modes. Layer 7
    for postflop is bound to ``DecisionData`` which preflop doesn't use,
    so the wiring needs preflop-shape variants. Skipped in step 7a.
    """
    options = explanation.options()
    if not options:
        return "the response had no non-empty options"
    if not explanation.correct_answer:
        return "correct_answer was empty"
    if explanation.correct_answer not in options:
        return (
            f"correct_answer {explanation.correct_answer!r} did not "
            f"match any of the four options exactly: {options!r}"
        )
    if not explanation.answer_explanation.strip():
        return "answer_explanation was empty"
    return None


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of an Anthropic Messages response.

    Tolerates both the SDK's content-block list and a raw string (handy for
    mocking in tests). Mirrors the postflop helper -- redefined locally
    rather than imported so this module stays decoupled from the
    grandfathered postflop file's private API.
    """
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if content is None:
        raise ExplanationValidationError(
            f"Anthropic response had no content: {response!r}"
        )
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            return str(text)
        if isinstance(block, dict) and "text" in block:
            return str(block["text"])
    raise ExplanationValidationError(
        f"no text block in Anthropic response: {content!r}"
    )


# --- main entry point --------------------------------------------------------
def generate_preflop_explanation(
    facts: PreflopFacts,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gold_examples: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    max_retries: int = 1,
) -> GeneratedExplanation:
    """Run Layer 6 (preflop) on a PreflopFacts and return the six CSV columns.

    Reads ``ANTHROPIC_API_KEY`` from the environment when constructing the
    default client. Pass a mock ``client`` (any object with
    ``messages.create``) for tests. ``gold_examples`` defaults to the
    preflop subset of the team's xlsx gold pool; pass a list for tests
    to avoid hitting the file.

    Retries once with a corrective message when the response fails
    structural validation (most commonly: ``correct_answer`` not matching
    any ``option_N`` exactly). A second failure raises
    ``ExplanationValidationError`` so the caller can route the question
    to human review.
    """
    if client is None:
        from anthropic import Anthropic  # lazy -- runtime dep

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if gold_examples is None:
        gold_examples = list(load_preflop_gold_examples())[:GOLD_EXAMPLE_COUNT]

    style = _detect_option_style_preflop(facts)
    system, messages = _build_messages_payload(facts, gold_examples, style)

    last_error: str | None = None
    for _ in range(max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = _extract_text(response)
        try:
            explanation = parse_response(text)
            error = _validate(explanation)
            if error is None:
                return explanation
            last_error = error
        except ExplanationValidationError as exc:
            last_error = str(exc)

        # One retry with a corrective assistant + user turn. Append, don't
        # restart -- keeps the gold-example cache block warm.
        messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": "That response failed validation: "
                f"{last_error}. Re-emit the JSON object, correcting only that "
                "issue. Keep the same options and explanation otherwise.",
            },
        ]

    raise ExplanationValidationError(
        f"Layer 6 (preflop) failed after {max_retries + 1} attempts; "
        f"last error: {last_error}. Spot routed to human review."
    )


# --- system-prompt override loader -------------------------------------------
# The admin panel's prompt editor saves edited prompts to a file under
# admin_panel/prompts/. If that file exists, it overrides the built-in
# :func:`build_preflop_system_prompt` output -- so prompt iteration doesn't
# require code changes. Reset = delete the file.
_PROMPT_OVERRIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "admin_panel"
    / "prompts"
    / "preflop_system.txt"
)


def load_preflop_system_prompt() -> str:
    """The active preflop system prompt -- override file if present, built-in
    default otherwise.

    The override mechanism lets the admin panel save edited prompts to
    ``admin_panel/prompts/preflop_system.txt``; this function checks for
    that file at every call. Reset to default = delete the override.

    Cached at module level would defeat the prompt-editor workflow (the
    user expects edits to take effect on the next generation), so this
    is intentionally NOT cached -- the file read happens per call.
    For batch runs the cost is negligible (one stat + maybe one read
    per batch start; the Anthropic prompt cache still applies to the
    resulting string since the content is identical across spots).
    """
    if _PROMPT_OVERRIDE_PATH.is_file():
        return _PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8")
    return build_preflop_system_prompt()


# --- new: explanation-only path (Layer 6 with deterministic options) --------
# Once :mod:`pipeline.preflop.options` computes the option strings + correct
# answer in pure Python, the LLM's job collapses to writing the explanation
# prose. The prompt is smaller (no option-style instruction, smaller output
# schema) and there's no retry-on-mismatch loop -- correct_answer can never
# disagree with the options when both come from the same deterministic
# source.
def _explanation_only_user_prompt(
    facts: PreflopFacts,
    gold_examples: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    options: list[str],
    correct_answer: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (system, messages) for the explanation-only path.

    Splits the user prompt into a cached gold-example block and a live
    block (same cache strategy as the full-generation path).
    """
    system = [
        {
            "type": "text",
            "text": load_preflop_system_prompt(),
            "cache_control": {"type": "ephemeral"},
        }
    ]
    cached_block = (
        "Read these gold examples to lock in the voice. Imitate the cadence, "
        "verdict-first structure, and concrete naming. Do not copy phrasing.\n\n"
        f"{_format_gold_examples(gold_examples)}"
    )
    # Format the option set for the LLM: one bullet per option, plus the
    # correct_answer called out separately so the model knows what to justify.
    options_block = "\n".join(f"  * {opt}" for opt in options)
    live_block = (
        "\n\n=== NEW QUESTION TO WRITE ===\n\n"
        f"{_question_framing_preflop(facts)}\n\n"
        "OPTIONS (already chosen by the deterministic option-selection module; "
        "do not invent or modify):\n"
        f"{options_block}\n\n"
        f"CORRECT ANSWER (also pre-chosen, verbatim): {correct_answer!r}\n\n"
        "SOLVER DATA (every strategic claim in your explanation must trace "
        "back to a field below; do not invent equity numbers or range "
        "claims):\n"
        f"{json.dumps(_trim_facts_for_prompt(facts), indent=2, default=str)}\n\n"
        "Your job: write ONLY the answer_explanation field justifying why the "
        "correct answer is correct. Respond with a single JSON object: "
        '{"answer_explanation": "<your prose>"} -- no other keys, no markdown '
        "fences, no prose outside the JSON. The explanation must follow "
        "every voice rule above (2-5 sentences, second person, verdict-first, "
        "suit emojis for specific cards, no em dashes or semicolons, etc.)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_block,
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": live_block},
            ],
        }
    ]
    return system, messages


def _parse_explanation_only_response(text: str) -> str:
    """Pull the answer_explanation string out of a single-field JSON response.

    Tolerates accidental code fences + leading prose, same as
    :func:`pipeline.explanation_generator.parse_response`.
    """
    import re  # noqa: PLC0415

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ExplanationValidationError(
                f"no JSON object in LLM response: {text!r}"
            )
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExplanationValidationError(
            f"LLM response was not valid JSON: {exc}"
        ) from exc
    explanation = data.get("answer_explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ExplanationValidationError(
            f"LLM response missing or empty answer_explanation: {data!r}"
        )
    return explanation


def generate_preflop_answer_explanation(
    facts: PreflopFacts,
    options: list[str],
    correct_answer: str,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    gold_examples: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
    max_retries: int = 1,
) -> GeneratedExplanation:
    """Generate ONLY the answer_explanation prose for a preflop spot.

    The option strings and ``correct_answer`` are computed beforehand by
    :mod:`pipeline.preflop.options` -- they're inputs to this function,
    not outputs. The LLM writes only the explanation prose, justifying
    why the supplied ``correct_answer`` is correct.

    Returns a fully-populated :class:`GeneratedExplanation` with the
    supplied options padded into the four CSV slots, the supplied
    correct_answer verbatim, and the LLM's prose in answer_explanation.

    Args:
        facts: The Layer 5 preflop data block.
        options: 1-4 option strings, computed by build_options.
        correct_answer: The correct option string, equal to one of
            ``options`` exactly.
        client: Optional Anthropic client (defaults to env-key lazy build).
        model: Model id (e.g. ``"claude-opus-4-7"``).
        temperature, max_tokens: Sampling controls.
        gold_examples: Optional gold-example pool override.
        max_retries: Retry count on parse failures. Default 1.

    Raises:
        ValueError: if ``correct_answer`` is not in ``options``.
        ExplanationValidationError: if every attempt produces invalid
            JSON / missing explanation field.
    """
    if correct_answer not in options:
        raise ValueError(
            f"correct_answer {correct_answer!r} not in options {options!r}"
        )
    if client is None:
        from anthropic import Anthropic  # lazy -- runtime dep

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if gold_examples is None:
        gold_examples = list(load_preflop_gold_examples())[:GOLD_EXAMPLE_COUNT]

    system, messages = _explanation_only_user_prompt(
        facts, gold_examples, options, correct_answer
    )

    last_error: str | None = None
    for _ in range(max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = _extract_text(response)
        try:
            explanation_prose = _parse_explanation_only_response(text)
            # Pad options to 4 for the CSV row.
            padded = (list(options) + ["", "", "", ""])[:4]
            return GeneratedExplanation(
                option_1=padded[0],
                option_2=padded[1],
                option_3=padded[2],
                option_4=padded[3],
                correct_answer=correct_answer,
                answer_explanation=explanation_prose,
            )
        except ExplanationValidationError as exc:
            last_error = str(exc)

        # One retry with a corrective turn.
        messages = messages + [
            {"role": "assistant", "content": text},
            {
                "role": "user",
                "content": (
                    f"That response failed validation: {last_error}. "
                    "Re-emit the JSON object with exactly one key "
                    "answer_explanation."
                ),
            },
        ]

    raise ExplanationValidationError(
        f"Layer 6 (preflop, explanation-only) failed after "
        f"{max_retries + 1} attempts; last error: {last_error}. "
        f"Spot routed to human review."
    )


__all__ = [
    "GOLD_EXAMPLE_COUNT",
    "PREFLOP_ARCHETYPE_GUIDANCE",
    "VOICE_RULES_PREFLOP",
    "build_preflop_system_prompt",
    "build_preflop_user_prompt",
    "generate_preflop_answer_explanation",
    "generate_preflop_explanation",
    "load_preflop_system_prompt",
]
