"""Layer 6 (postflop): the ONLY layer that calls an external LLM.

Given a fully-resolved :class:`~pipeline.postflop.facts.PostflopFacts` block
plus the deterministically-built options and the correct answer, this turns
the data into a 2-5 sentence coaching explanation in the team's voice. The
LLM writes ONLY the prose; the options and the correct answer come from the
solver (Layer 4 / the option builder), so the model can never change the
strategy -- exactly the boundary the whole project defends.

Two paths:

* :func:`placeholder_explanation` -- a deterministic, no-API explanation built
  from the fact block. The batch uses it for dry runs (and tests use it), so
  the entire pipeline runs end-to-end with no ``ANTHROPIC_API_KEY``.
* :func:`generate_postflop_explanation` -- the real Anthropic call (via the
  shared :func:`pipeline.explanation_generator.call_messages_create`), with
  one corrective retry against the hard validators before raising
  :class:`~pipeline.explanation_generator.ExplanationValidationError` for
  Layer 7 to route to human review.

The system prompt + archetype guidance live here; prompt tuning against gold
postflop examples is a documented follow-up (see the package README).
"""

from __future__ import annotations

from collections.abc import Callable

from pipeline.explanation_generator import (
    ExplanationValidationError,
    GeneratedExplanation,
    call_messages_create,
)
from pipeline.postflop.action_history import format_question
from pipeline.postflop.facts import PostflopFacts
from pipeline.postflop.solve import PostflopSolve
from pipeline.postflop.validators import run_postflop_audit_validators

DEFAULT_MODEL = "claude-opus-4-7"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_MAX_TOKENS = 700

UsageCallback = Callable[[object], None]

# The strategic frame for each postflop archetype -- the LLM is told exactly
# one of these (chosen deterministically by the fact extractor) and writes
# toward it. Keeps the model from inventing the "why".
POSTFLOP_ARCHETYPE_GUIDANCE: dict[str, str] = {
    "value_bet": "Bet for value: a strong made hand that wants worse hands to "
                 "call. Frame around getting value from villain's weaker range.",
    "bluff": "A bet with little showdown value that needs villain to fold. "
             "Frame around fold equity, not made-hand strength.",
    "semibluff": "A bet with a strong draw: fold equity now plus equity when "
                 "called. Frame around the draw and the two ways to win.",
    "protection_bet": "Bet a vulnerable made hand to deny equity. Frame around "
                      "charging draws / overcards on a board that can turn bad.",
    "pot_control_check": "Check a medium-strength hand to keep the pot small "
                         "and realize equity. Frame around pot control, not weakness.",
    "trap_check": "Check a very strong hand to induce or trap. Frame around "
                  "letting villain bluff or catch up cheaply.",
    "defensive_check": "Check a weak hand with little to gain by betting. Frame "
                       "around giving up the betting lead, not bluffing into a "
                       "stronger range.",
    "bluff_catch": "Call to catch bluffs with a hand that beats villain's bluffs "
                   "but loses to value. Frame around villain's bluff frequency "
                   "and the price, not your own hand strength.",
    "call_with_showdown": "Call with a strong hand that prefers not to raise. "
                          "Frame around keeping villain's bluffs in.",
    "call_drawing": "Call with a draw getting the right price/implied odds. "
                    "Frame around the price and what completes your hand.",
    "value_raise": "Raise a strong hand for value. Frame around building the pot "
                   "against villain's worse continues.",
    "bluff_raise": "Raise as a bluff/semibluff. Frame around fold equity and "
                   "the hands you represent.",
    "fold_to_pressure": "Fold to a bet you can't profitably continue against. "
                        "Frame around being beaten by villain's betting range / "
                        "the wrong price.",
    "unclassified": "Explain the solver's chosen action plainly from the data.",
}

# The voice rules + hard constraints, kept terse. The brief's full voice rules
# live in the gold examples; this is the postflop-specific instruction.
POSTFLOP_SYSTEM_PROMPT = """\
You are writing the answer explanation for a single postflop poker training \
question, in the voice of an expert coach. You will be given a fully-resolved \
SOLVER DATA block and the CORRECT answer. Write ONLY the explanation prose.

Rules:
1. Open with the verdict (the correct action), then give the reasoning.
2. Coach the reader as "you"; refer to the opponent by position.
3. 2-5 sentences. Be concrete and confident; no hedging filler.
4. Use ONLY the facts in the SOLVER DATA block. Never invent equities, \
frequencies, or cards. If you cite a number, use the one given.
5. You may name your own two cards and the board cards (with suit emojis); \
never name a specific card you were not given.
6. NO em dashes, NO semicolons, no corporate/template phrasing.
7. Write toward the STRATEGIC FRAME you are given; do not re-decide the spot.
Return only the explanation text, no preamble, no headings.
"""


def _humanize(token: str) -> str:
    """'top_pair_top_kicker' -> 'top pair, top kicker' (readable, for prose)."""
    return token.replace("_", " ")


def build_solver_data_block(facts: PostflopFacts) -> str:
    """The structured fact block the LLM reads (the data, not the prose)."""
    f = facts
    lines = [
        f"STREET: {f.street}",
        f"BOARD: {' '.join(f.board)}",
        f"HERO: {f.hero_position} ({'in position' if f.hero_in_position else 'out of position'})"
        f", {'the preflop aggressor' if f.hero_is_preflop_aggressor else 'not the preflop aggressor'}",
        f"VILLAIN: {f.villain_position}",
        f"HERO HAND: {f.spot.hero_combo}  ({_humanize(f.hand_label)})",
        f"HAND STRENGTH BUCKET: {f.strength_bucket}",
        f"BOARD TEXTURE: {f.board_texture.get('composite', '')} "
        f"({f.board_texture.get('suit_distribution', '')}, "
        f"{f.board_texture.get('connectedness', '')}, "
        f"{f.board_texture.get('pair_status', '')})",
        f"HERO EQUITY vs villain range: {f.hero_equity_vs_villain * 100:.0f}%",
        f"POT: {f.pot_bb:g}bb   EFFECTIVE STACK: {f.spot.node.effective_stack_bb:g}bb"
        f"   SPR: {f.spr:.1f}",
    ]
    if f.break_even_equity is not None:
        lines.append(
            f"FACING A BET: must call {f.to_call_bb:g}bb; break-even equity "
            f"{f.break_even_equity * 100:.0f}%"
        )
    lines.extend([
        f"CORRECT ACTION: {f.dominant_action} (solver frequency "
        f"{f.dominant_frequency:.0%})",
        f"STRATEGIC FRAME ({f.archetype}): "
        f"{POSTFLOP_ARCHETYPE_GUIDANCE.get(f.archetype, '')}",
        f"CONCEPT TAGS: {', '.join(f.concept_tags)}",
    ])
    return "\n".join(lines)


def _assemble(options: list[str], correct: str, prose: str) -> GeneratedExplanation:
    """Build the 6-column dataclass from our options + the LLM's prose."""
    padded = (options + ["", "", "", ""])[:4]
    return GeneratedExplanation(
        option_1=padded[0],
        option_2=padded[1],
        option_3=padded[2],
        option_4=padded[3],
        correct_answer=correct,
        answer_explanation=prose.strip(),
    )


def placeholder_explanation(
    facts: PostflopFacts, options: list[str], correct: str
) -> GeneratedExplanation:
    """A deterministic, no-API explanation built from the fact block.

    Used for dry runs and tests so the whole pipeline runs without an API key.
    Deliberately plain (and free of suit-emoji cards) so it always passes the
    hard validators; it is NOT meant to read like the gold coaching voice.
    """
    f = facts
    verb = f.dominant_verb or "play"
    eq = round(f.hero_equity_vs_villain * 100)
    frame = POSTFLOP_ARCHETYPE_GUIDANCE.get(f.archetype, "").split(".")[0].lower()
    prose = (
        f"The solver's play is to {verb} here. "
        f"With {_humanize(f.hand_label)} you have about {eq}% equity against "
        f"{f.villain_position}'s range. "
        f"This is a {_humanize(f.archetype)} spot: {frame}."
    )
    return _assemble(options, correct, prose)


def generate_postflop_explanation(
    facts: PostflopFacts,
    options: list[str],
    correct: str,
    solve: PostflopSolve,
    *,
    client: object,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str | None = None,
    usage_callback: UsageCallback | None = None,
) -> GeneratedExplanation:
    """Generate one postflop explanation via the Anthropic API + validate.

    The LLM writes only the prose; ``options`` and ``correct`` are fixed. One
    corrective retry against the hard validators, then
    :class:`ExplanationValidationError` (Layer 7 routes that to human review).

    Args:
        facts: the resolved fact block.
        options / correct: the deterministically-built answer set.
        solve: parent solve (for the question narrative).
        client: an Anthropic client (``client.messages.create``). Required.
    """
    if client is None:
        raise ValueError(
            "generate_postflop_explanation needs an Anthropic client; use "
            "placeholder_explanation for dry runs."
        )
    system = system_prompt or POSTFLOP_SYSTEM_PROMPT
    question = format_question(facts.spot, solve)
    data_block = build_solver_data_block(facts)
    base_user = (
        f"QUESTION:\n{question}\n\n"
        f"OPTIONS: {options}\n"
        f"CORRECT ANSWER: {correct}\n\n"
        f"SOLVER DATA:\n{data_block}\n\n"
        "Write the answer explanation."
    )

    feedback = ""
    last_prose = ""
    for _attempt in range(2):
        user = base_user if not feedback else f"{base_user}\n\nFIX: {feedback}"
        response = call_messages_create(
            client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        if usage_callback is not None and getattr(response, "usage", None) is not None:
            usage_callback(response.usage)
        last_prose = _response_text(response)
        generated = _assemble(options, correct, last_prose)
        result = run_postflop_audit_validators(generated, facts)
        if result.is_valid:
            return generated
        feedback = result.error_message

    raise ExplanationValidationError(
        f"postflop explanation failed validation after one retry: {feedback}",
        last_attempt_text=last_prose,
        last_attempt_candidate=_assemble(options, correct, last_prose),
    )


def _response_text(response: object) -> str:
    """Pull the text out of an Anthropic messages response (or a test stub)."""
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "".join(parts)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TEMPERATURE",
    "POSTFLOP_ARCHETYPE_GUIDANCE",
    "POSTFLOP_SYSTEM_PROMPT",
    "build_solver_data_block",
    "generate_postflop_explanation",
    "placeholder_explanation",
]
