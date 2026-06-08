"""PLO Layer 6 -- the ONLY layer that calls an external LLM.

Writes the ``answer_explanation`` prose for one PLO preflop spot. The four
options and the correct answer are computed deterministically
(:mod:`pipeline.plo.options`); the LLM only writes the words, grounded in a
SOLVER DATA block built from :class:`~pipeline.plo.fact_extractor.PloFacts`. It
never decides strategy -- the solver already did.

Reuses the vetted NLHE voice rules + banned-phrase list. Only the two rules that
hardcode Hold'em hand notation (the 169-class label) are adapted for 4-card PLO;
everything else is copied verbatim, including rule 7 (no em dashes / semicolons).

Ships WITHOUT few-shot examples by default -- the voice rules carry the voice,
and no examples avoids templating every question into one mold. The ``examples``
seam (:mod:`pipeline.plo.gold_examples`) is wired so they drop in later with no
code change.

Em dashes are GUARANTEED out three ways: a voice rule forbids them, a validator
flags any banned phrase, and a deterministic strip rewrites any that slip
through (a soft prompt rule alone is not enough at scale).
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from typing import Any

from pipeline.action_history import format_card
from pipeline.explanation_generator import (
    BANNED_LITERAL_PHRASES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    ExplanationValidationError,
    GeneratedExplanation,
    call_messages_create,
)
from pipeline.plo.action_history import format_plo_action_history
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.gold_examples import load_plo_gold_examples
from pipeline.plo.options import canonicalize_strategy
from pipeline.plo.position import hero_relative_position
from pipeline.plo.spot_tags import compute_plo_concept_tags

logger = logging.getLogger(__name__)

UsageCallback = Callable[[str, int, int, int, int], None]
_PERCENT = 100


# --- voice rules (PLO) -----------------------------------------------------
# Copied verbatim from VOICE_RULES_PREFLOP except rules 5 and 10, which hardcode
# the Hold'em 169-class label and are adapted for 4-card PLO (see the two
# ADAPTED comments). Lead clauses are asserted in tests so wording tweaks can't
# silently drop a rule.
VOICE_RULES_PLO: tuple[str, ...] = (
    "Open with the verdict. The first sentence states the correct action "
    'plainly ("The best play is to call.", "You should 3-bet here.", '
    '"This is a clear fold."). Do not bury the answer.',
    'Address the reader directly in second person ("you", "your hand", '
    '"you\'re holding"). Never refer to "the player" or "hero" -- the '
    "student IS hero.",
    "Structure the body as: (a) verdict, (b) the reasoning (your hand's "
    "shape, position, prior action, villain's range, equity), (c) an optional "
    "one-line exploit note when the spot is population-sensitive. Skip (c) if "
    "it doesn't apply. Never misstate the solver's own numbers: the SOLVER "
    "DATA action_strategy is the GTO baseline, so do not call the GTO line 'a "
    "fold' or claim the solver mixes an action it shows at 0%.",
    "Be as long as the spot genuinely needs and no longer. Lead tight and get "
    "to the point, but if a spot has real nuance, take the room. Do not pad to "
    "hit a length, and do not cram a complex spot into one line.",
    # 5. ADAPTED for PLO: 4-card hands have no 169-class label; positions use BU.
    "Name things concretely: positions by abbreviation (LJ, HJ, CO, BU, SB, "
    "BB), and your hand by its SHAPE (a double-suited rundown, aces with a "
    "broadway side card, a bare ace, a dangler). PLO hands have four cards and "
    "no two-card class label, so never write Hold'em notation like AKs or 77. "
    'Never reference "the flop", "the turn", "the river", or any community '
    "cards -- the decision is preflop, no board has come yet.",
    'Translate solver numbers into plain language. Say "the Button opens '
    'roughly the top fifth of hands here" rather than "open range = 18% of '
    'combos"; say "your hand runs about even against this 3-bet range" rather '
    'than "hero_equity_vs_villain = 0.48".',
    "Never use an em dash. Never use a semicolon. Use periods, commas, and "
    "short sentences. Do not use any phrase from the BANNED PHRASES list.",
    "Write with the confidence of a coach who has solved the spot. Do not "
    'hedge with "it might be", "perhaps", "in theory". State what the right '
    "play is and why, plainly.",
    "When referencing a SPECIFIC card in prose, use the suit emoji form: A♠️, "
    "K❤️, T♦️, 9♣️ (rank letter then suit emoji, no space). Never use "
    "plain-text suit letters like 'Kh' or 'Ad' -- those are internal solver "
    "notation, not voice.",
    # 10. ADAPTED for PLO: describe villain's range by TYPE, not by 4-card combos.
    "When discussing villain's range, describe it by TYPE and size, anchored "
    "to the SOLVER DATA (e.g. \"a tight 3-bet range, mostly premium pairs and "
    'big double-suited broadway hands"). Do not invent specific four-card '
    "holdings, and do not leave the range as a vague \"value hands\" with no "
    "shape.",
    # 11. Clean final draft -- no "thinking out loud" / self-correction.
    "Write a clean, final draft, as if already edited. Never think out loud or "
    'revise yourself mid-sentence: do not use "actually", "wait", "I mean", or '
    'a trailing "..." to fix a word you just wrote. State each fact once and '
    "correctly. The SOLVER DATA your_hand field gives the exact four cards, so "
    "get every card and suit right the first time. Do not narrate the hand "
    'card-by-card ("connectivity from the ten down to the three"); name the '
    "shape and the cards that matter, then move on.",
    # 12. Frame blockers correctly for a PREFLOP decision.
    "Frame blockers as preflop CARD REMOVAL. An ace mainly removes villain's "
    "AA and other premium aces, shrinking his value range -- that is the "
    "point, and you do not need to name the ace's suit to make it. A suited "
    "ace also blocks the nut flush, but preflop that is a minor, secondary "
    "factor: mention it briefly if at all, never as the main reason, and do "
    "not claim a single suited card meaningfully cuts into his range before any "
    "board has come.",
)

# Concise strategic frame per PLO archetype (one line the LLM reads fast).
PLO_ARCHETYPE_GUIDANCE: dict[str, str] = {
    "open_for_value": "First in with a hand worth opening; frame around its shape and position.",
    "open_fold": "First in with a hand below the opening threshold; frame around position discipline.",
    "fold_dominated": "Facing aggression with a hand too weak or non-nut to continue; frame around the missing equity and playability.",
    "fold_pot_odds": "A playable hand that is still priced out; frame around pot odds, realization, and reverse implied odds.",
    "call_for_value": "Calling with a strong hand that prefers a controlled pot to a re-raise; frame around strength and position.",
    "call_for_implied_odds": "Calling a well-shaped speculative hand that realizes equity postflop; frame around playability, position, implied odds, and a nut-vs-non-nut caution where it applies.",
    "call_allin": "Calling a jam. No streets remain, so it is a pure pot-odds and raw-equity call. Do NOT mention implied odds, draws, or postflop play.",
    "3bet_for_value": "Re-raising for value or strong playability. If the equity is close, say so plainly and lean on position and nut potential rather than claiming you are ahead.",
    "3bet_as_bluff": "A light 3-bet whose value is fold equity and playability, not raw strength; frame around pressure and position.",
    "4bet_for_value": "Re-raising a 3-bet with a premium hand; frame around big-pair construction, and respect that aces sit at the top of villain's range when your equity is only mid-40s.",
    "4bet_as_bluff": "A light 4-bet leaning on fold equity and blockers, not raw strength.",
    "5bet_for_value": "A stack-committing 5-bet with a premium; frame around the top of ranges and the price.",
    "5bet_as_bluff": "A light 5-bet jam leaning on fold equity and blockers.",
    "squeeze_for_value": "Raising over an open plus caller(s) with a strong hand; frame around isolating and the dead money.",
    "squeeze_as_bluff": "A light squeeze leaning on the dead money and fold equity.",
    "all_in_for_value": "A preflop jam for value; frame around stack depth and getting it in ahead.",
    "all_in_as_bluff": "A preflop jam leaning on fold equity.",
    "unclassified": "Frame plainly from the equity, position, and the action_strategy.",
}


# --- prompt construction ---------------------------------------------------
def _format_voice_rules() -> str:
    return "\n".join(f"{i + 1}. {r}" for i, r in enumerate(VOICE_RULES_PLO))


def _format_banned_phrases() -> str:
    return ", ".join(f'"{p}"' for p in BANNED_LITERAL_PHRASES)


def build_plo_system_prompt(*, examples: tuple[dict[str, Any], ...] = ()) -> str:
    """The PLO Layer 6 system prompt. ``examples`` is empty by default."""
    parts = [
        "You are an expert Pot-Limit Omaha coach writing the answer "
        "explanation for a single preflop decision. You receive a SOLVER DATA "
        "block containing the correct action and the facts behind it. You "
        "write ONLY the explanation prose that justifies the correct action. "
        "You never decide strategy yourself -- the solver already did -- you "
        "put its conclusion into a coach's words, grounded only in the data "
        "given.\n",
        "VOICE RULES (every rule applies to every output):\n"
        + _format_voice_rules()
        + "\n",
        f"BANNED PHRASES (never appear in the output): {_format_banned_phrases()}.\n",
        "STRATEGIC FRAMES by archetype:\n"
        + "\n".join(f"- {a}: {g}" for a, g in PLO_ARCHETYPE_GUIDANCE.items())
        + "\n",
    ]
    if examples:
        parts.append(
            "EXAMPLES of the target voice (match the STRUCTURE and tone, vary "
            "your openings, do not copy content):\n"
            + "\n\n".join(
                f"Q: {e.get('question', '')}\nA: {e.get('answer_explanation', '')}"
                for e in examples
            )
            + "\n"
        )
    parts.append(
        'Output a single JSON object with exactly one key "answer_explanation" '
        "whose value is the explanation string. No other keys, no preamble, no "
        "code fence."
    )
    return "\n".join(parts)


def _pct(value: float | None) -> int | None:
    return None if value is None else round(value * _PERCENT)


def build_solver_data(
    facts: PloFacts,
    options: list[str],
    correct_answer: str,
    *,
    include_skills: bool = False,
) -> dict[str, Any]:
    """The SOLVER DATA block fed to the LLM -- the facts behind the decision.

    ``include_skills`` adds the tagged user-facing skills as a
    ``skills_this_spot_tests`` field -- OFF by default; it exists so the admin
    Compare page can A/B "does naming the skills help the prose?" head-to-head.
    """
    from pipeline.plo.skill_tagger import compute_plo_skills  # noqa: PLC0415
    hand = facts.hand_class
    villain = facts.villain_stats
    data: dict[str, Any] = {
        "situation": format_plo_action_history(facts, display_in_bb=True),
        "your_hand": " ".join(format_card(c) for c in facts.spot.hero_cards),
        "your_hand_shape": f"{hand.descriptor} ({hand.strength})",
        "your_position": hero_relative_position(facts).lower(),
        "options": options,
        "correct_action": correct_answer,
        "action_strategy": {
            label: f"{round(freq * _PERCENT)}%"
            for label, freq in canonicalize_strategy(facts).items()
        },
        "strategic_frame": (
            f"{facts.archetype}: "
            f"{PLO_ARCHETYPE_GUIDANCE.get(facts.archetype, '')}"
        ),
        "concept_tags": compute_plo_concept_tags(facts),
        "ev_gap_bb": (
            round(facts.ev_gap_bb, 2) if facts.ev_gap_bb is not None else None
        ),
    }
    eq = _pct(facts.hero_equity_vs_villain)
    if eq is not None:
        data["your_hand_equity_vs_villain_range_pct"] = eq
    range_eq = _pct(facts.hero_range_equity_vs_villain)
    if range_eq is not None:
        data["your_range_equity_vs_villain_range_pct"] = range_eq
    if villain is not None:
        data["villain"] = {
            "seat": villain.seat,
            "action": villain.action_label,
            "range_pct_of_all_hands": round(villain.pct_of_dealt_hands),
        }
    if include_skills:
        data["skills_this_spot_tests"] = compute_plo_skills(facts)
    return data


def build_plo_user_prompt(
    facts: PloFacts,
    options: list[str],
    correct_answer: str,
    *,
    include_skills: bool = False,
) -> str:
    """The per-question USER message: the SOLVER DATA block + the ask.

    This is exactly what the LLM receives for one spot (the system prompt is
    separate). Public so the admin Prompt page can show it -- the live path and
    the preview build the prompt the same way, so they can't drift.
    ``include_skills`` adds the tagged skills to the data (Compare A/B seam).
    """
    block = json.dumps(
        build_solver_data(facts, options, correct_answer, include_skills=include_skills),
        indent=2,
    )
    return (
        f"SOLVER DATA:\n{block}\n\n"
        f'Write the answer_explanation justifying why "{correct_answer}" is the '
        'correct action. Output JSON {"answer_explanation": "..."}.'
    )


# --- post-processing + validation -----------------------------------------
# Punctuation we auto-rewrite (a guarantee, since the rule alone leaks at scale).
_AUTO_REWRITE = {"—": ", ", "–": ", ", ";": ","}
# Phrase bans that should trigger a regeneration rather than a silent rewrite.
_BANNED_PHRASES = tuple(
    p for p in BANNED_LITERAL_PHRASES if p not in _AUTO_REWRITE
)


def _normalize(text: str) -> str:
    """Tidy prose AND guarantee no em/en dashes or semicolons reach the CSV."""
    for bad, good in _AUTO_REWRITE.items():
        text = text.replace(bad, good)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\s+,", ",", text)  # " ," -> ","
    text = re.sub(r",\s*,", ",", text)  # ",," -> ","
    text = re.sub(r"[ \t]{2,}", " ", text)  # collapse runs of spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _banned_present(text: str) -> list[str]:
    low = text.lower()
    return [p for p in _BANNED_PHRASES if p.lower() in low]


# A specific card named in prose (rank + suit emoji) that isn't one of hero's
# four is a fabrication -- e.g. "K<heart>" when the hand holds K<diamond>. PLO
# suits are load-bearing (double-suited), so the LLM occasionally misstates one;
# this deterministic audit rejects it and forces a corrective retry. (NLHE
# rarely hits this: its hands abstract to a suit-light 2-card class like "KQo".)
_PROSE_SUIT = {"♠": "s", "♥": "h", "❤": "h", "♦": "d", "♣": "c"}
_PROSE_CARD_RE = re.compile(r"([2-9TJQKA])\s?([♠♥❤♦♣])")


def _fabricated_cards(prose: str, hero_cards: tuple[str, ...]) -> list[str]:
    """Specific cards named in ``prose`` that aren't among hero's four.

    Only flags explicit ``<rank><suit-emoji>`` mentions (the voice-rule form);
    vague references ("your ace", "the king") are never flagged, so there are
    no false positives. Preflop there is no board and villain's range is named
    by type, so every concrete card should be one of hero's own.
    """
    held = {(c[0].upper(), c[1].lower()) for c in hero_cards}
    out: list[str] = []
    for rank, suit_char in _PROSE_CARD_RE.findall(prose):
        suit = _PROSE_SUIT.get(suit_char)
        token = f"{rank}{suit_char}"
        if suit is not None and (rank.upper(), suit) not in held and token not in out:
            out.append(token)
    return out


def _parse(text: str) -> str:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ExplanationValidationError(f"no JSON object in response: {text!r}")
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ExplanationValidationError(f"response was not valid JSON: {exc}") from exc
    prose = data.get("answer_explanation")
    if not isinstance(prose, str) or not prose.strip():
        raise ExplanationValidationError(f"missing/empty answer_explanation: {data!r}")
    return _normalize(prose)


def _extract_text(response: Any) -> str:
    content = getattr(response, "content", None)
    if content:
        text = getattr(content[0], "text", None)
        if isinstance(text, str) and text.strip():
            return text
    raise ExplanationValidationError(f"no text in LLM response: {response!r}")


def _extract_usage(response: Any) -> tuple[int, int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0, 0, 0)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        int(getattr(usage, "cache_read_input_tokens", 0) or 0),
    )


# --- the entry point -------------------------------------------------------
def generate_plo_answer_explanation(
    facts: PloFacts,
    options: list[str],
    correct_answer: str,
    *,
    client: Any = None,
    examples: tuple[dict[str, Any], ...] | None = None,
    system_prompt: str | None = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 1,
    include_skills: bool = False,
    usage_callback: UsageCallback | None = None,
) -> GeneratedExplanation:
    """Generate the answer_explanation prose for one PLO spot.

    Returns a :class:`GeneratedExplanation` with the supplied options/correct
    answer and the LLM's prose (normalized so no em dash / semicolon survives).
    Pass a mock ``client`` in tests. ``examples=None`` loads the (currently
    empty) PLO gold pool; pass a tuple to override.

    ``system_prompt`` overrides the built-in system prompt verbatim -- the
    admin panel's prompt library passes an edited full prompt here (the same
    seam as NLHE). When None, the built-in :func:`build_plo_system_prompt` is
    used (with ``examples``).

    Raises ``ValueError`` if ``correct_answer`` not in ``options``, or
    ``ExplanationValidationError`` if every attempt fails validation.
    """
    if correct_answer not in options:
        msg = f"correct_answer {correct_answer!r} not in options {options!r}"
        raise ValueError(msg)
    if client is None:
        from anthropic import Anthropic  # noqa: PLC0415 -- runtime dep

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    if examples is None:
        examples = load_plo_gold_examples()

    system = (
        system_prompt
        if system_prompt is not None
        else build_plo_system_prompt(examples=examples)
    )
    messages = [
        {
            "role": "user",
            "content": build_plo_user_prompt(
                facts, options, correct_answer, include_skills=include_skills
            ),
        }
    ]
    padded = (list(options) + ["", "", "", ""])[:4]

    last_error = ""
    last_text = ""
    for _ in range(max_retries + 1):
        response = call_messages_create(
            client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        if usage_callback is not None:
            usage_callback(model, *_extract_usage(response))
        last_text = _extract_text(response)
        try:
            prose = _parse(last_text)
            banned = _banned_present(prose)
            if banned:
                msg = f"used banned phrase(s): {banned}"
                raise ExplanationValidationError(msg)
            fabricated = _fabricated_cards(prose, facts.spot.hero_cards)
            if fabricated:
                hand = " ".join(format_card(c) for c in facts.spot.hero_cards)
                msg = (
                    f"named card(s) not in your hand: {', '.join(fabricated)}. "
                    f"Your exact hand is {hand} -- use only those four cards, "
                    "and prefer describing the shape over reciting cards."
                )
                raise ExplanationValidationError(msg)
            return GeneratedExplanation(
                option_1=padded[0],
                option_2=padded[1],
                option_3=padded[2],
                option_4=padded[3],
                correct_answer=correct_answer,
                answer_explanation=prose,
            )
        except ExplanationValidationError as exc:
            last_error = str(exc)
        messages = [
            *messages,
            {"role": "assistant", "content": last_text},
            {
                "role": "user",
                "content": (
                    f"That failed validation: {last_error}. Re-emit the JSON "
                    'object with exactly one key "answer_explanation", obeying '
                    "every voice rule (no em dashes, no semicolons)."
                ),
            },
        ]

    raise ExplanationValidationError(
        f"PLO Layer 6 failed after {max_retries + 1} attempts; last error: "
        f"{last_error}. Spot routed to human review.",
        last_attempt_text=last_text,
    )


__all__ = [
    "PLO_ARCHETYPE_GUIDANCE",
    "VOICE_RULES_PLO",
    "build_plo_system_prompt",
    "build_plo_user_prompt",
    "build_solver_data",
    "generate_plo_answer_explanation",
]
