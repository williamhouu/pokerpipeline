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
from pathlib import Path

from pipeline.bb_display import round_to_half_bb
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
    "fold_to_pressure": "Fold to pressure you can't profitably continue against. "
                        "Frame around being beaten by villain's betting/raising "
                        "range. If hero's equity actually CLEARS the break-even "
                        "price, do NOT claim the price is unmet -- the fold is "
                        "because the range is too strong or the spot is close, so "
                        "say that honestly rather than reversing the price.",
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
8. Range vs range: if you say who is ahead as a RANGE on this board, or who \
holds the nutted hands, state it exactly as the RANGE ADVANTAGE and NUT \
ADVANTAGE facts say. Never compute or reverse who has the edge, and do not \
claim a range advantage the data does not give you.
9. Blockers: discuss blockers ONLY when a BLOCKERS line is present, and only as \
it states (whether you mainly block villain's value or their bluffs). If there \
is no BLOCKERS line, do not mention blockers at all. Never invent which specific \
combos you remove or reverse value vs bluffs.
10. Equity source: when you say WHERE your equity comes from, follow the \
CURRENTLY AHEAD line. A made hand that beats much of villain's range has \
SHOWDOWN equity (it is ahead of their weaker and air hands right now) -- do NOT \
attribute its equity to draws, outs, backdoors, or "spiking" a card unless you \
actually hold a draw (a DRAWS line is present). A hand that is behind most of \
villain's range but holds a real draw gets its equity from improving. If a hand \
is ahead now yet the answer is to fold or check, frame it as a realization or \
reverse-implied-odds problem (you cannot profitably continue), not as having low \
equity. When the CURRENTLY AHEAD line shows a chop share (TIES), treat it honestly: equity that comes from chopping is pot-splitting, not chances to win. A hand that mostly chops and otherwise loses is NOT "drawing dead", it is "mostly chopping"; never describe chop-heavy equity as winning chances.
11. Bet-size wording: never echo a raw solver size label as the instruction \
("You should bet 53%" is wrong). State the action so it matches the correct \
option's wording, and express any size in natural poker language with the real \
amount where given: "bet about half the pot (4bb)", "a small third-pot bet", \
"an overbet". Rough map: ~33% = about a third of the pot, ~50-60% = about half \
the pot, ~66-80% = about two-thirds pot, ~100% = a pot-sized bet, 110%+ = an \
overbet.
Return only the explanation text, no preamble, no headings.
"""


# Chop share worth surfacing on the CURRENTLY AHEAD line (and in the math
# panel). Below this, ties are rounding noise; above it, the LLM must not
# call the hand's equity "winning chances" (team, July 2026: a 48%-equity
# river that was almost all chops read as "drawing dead").
_TIE_SIGNIFICANT = 0.05


def _tie_clause(f: PostflopFacts) -> str:
    """', and TIES (chops) with N%' when the chop share is significant."""
    tied = getattr(f, "currently_tied_pct", 0.0) or 0.0
    if tied < _TIE_SIGNIFICANT:
        return ""
    return f", and TIES (chops the pot) with {tied * 100:.0f}%,"


# Admin-editable override for the postflop system prompt. The Prompt page saves
# the user's edits here (gitignored, like the preflop one); generation reads the
# override if present, else the built-in default. Reset = delete the file.
# Simpler than preflop's: postflop's archetype guidance reaches the LLM per-spot
# in the SOLVER DATA block (not baked into the system prompt), so there's no
# catalog to resync -- a plain file read suffices.
_POSTFLOP_PROMPT_OVERRIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "admin_panel"
    / "prompts"
    / "postflop_system.txt"
)


def load_postflop_system_prompt() -> str:
    """The active postflop system prompt: the admin override file if present,
    else :data:`POSTFLOP_SYSTEM_PROMPT`. Not cached (edits take effect on the
    next run); reset to default = delete the override file."""
    if _POSTFLOP_PROMPT_OVERRIDE_PATH.is_file():
        return _POSTFLOP_PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8")
    return POSTFLOP_SYSTEM_PROMPT


def _humanize(token: str) -> str:
    """'top_pair_top_kicker' -> 'top pair, top kicker' (readable, for prose)."""
    return token.replace("_", " ")


# Render the resolved range/nut-advantage verdict for the data block. "hero" /
# "villain" become "you" / "villain" so the LLM can quote it verbatim.
_ADVANTAGE_PHRASE = {
    "hero": "you (hero)",
    "villain": "villain",
    "even": "roughly even (neither player)",
}


def _advantage_phrase(label: str) -> str:
    return _ADVANTAGE_PHRASE.get(label, "roughly even (neither player)")


# Precise, unambiguous names for each draw type the hand classifier emits. The
# composite hand_label flattens an open-ended draw to "...with straight draw",
# which the LLM has mislabeled "open-ended" on a gutshot (and vice versa); the
# DRAWS line in the data block surfaces the exact type so the prose can't guess.
_DRAW_LABELS = {
    "straight_draw_open_ended": "open-ended straight draw",
    "gutshot": "gutshot (one card completes the straight)",
    "flush_draw_nut": "nut flush draw",
    "flush_draw_weak": "weak flush draw",
    "combo_draw": "combo draw (flush draw plus straight draw)",
}


def build_solver_data_block(facts: PostflopFacts) -> str:
    """The structured fact block the LLM reads (the data, not the prose)."""
    f = facts
    # On a facing-bet node villain_range IS the betting range, so name it that.
    ahead_ref = (
        f"the hands {f.villain_position} is betting"
        if f.spot.node.is_facing_bet
        else f"{f.villain_position}'s range"
    )
    lines = [
        f"STREET: {f.street}",
        f"BOARD: {' '.join(f.board)}",
        f"HERO: {f.hero_position} ({'in position' if f.hero_in_position else 'out of position'})"
        f", {'the preflop aggressor' if f.hero_is_preflop_aggressor else 'not the preflop aggressor'}",
        f"VILLAIN: {f.villain_position}",
        f"HERO HAND: {f.spot.hero_combo}  ({_humanize(f.hand_label)})",
        f"HAND STRENGTH BUCKET: {f.strength_bucket}",
        *(
            [f"DRAWS: {', '.join(_DRAW_LABELS.get(d, _humanize(d)) for d in f.draws)}"]
            if f.draws
            else []
        ),
        f"BOARD TEXTURE: {f.board_texture.get('composite', '')} "
        f"({f.board_texture.get('suit_distribution', '')}, "
        f"{f.board_texture.get('connectedness', '')}, "
        f"{f.board_texture.get('pair_status', '')})",
        f"HERO EQUITY vs villain range: {f.hero_equity_vs_villain * 100:.0f}%",
        # Where the equity comes from (exact showdown comparison, resolved in
        # Python). A made hand AHEAD of much of villain's range has SHOWDOWN
        # equity (it beats their weaker/air hands now) -- NOT draw equity. The
        # LLM must use this and not mis-attribute a made hand's equity to draws.
        f"CURRENTLY AHEAD: your hand beats {f.currently_ahead_pct * 100:.0f}%{_tie_clause(f)} of "
        f"{ahead_ref} at showdown right now (behind "
        f"{f.currently_behind_pct * 100:.0f}%)",
        # Node-level "who is ahead on this board" verdicts, resolved in Python
        # (the LLM mirrors them; it must never compute or reverse them). The
        # number is the supporting evidence for the stated verdict.
        f"RANGE ADVANTAGE: {_advantage_phrase(f.range_advantage)} "
        f"(your range ~{f.hero_range_equity * 100:.0f}% equity vs villain's "
        f"range on this board)",
        f"NUT ADVANTAGE: {_advantage_phrase(f.nut_advantage)} "
        f"(strong made hands, two pair or better: you {f.hero_nut_share * 100:.0f}%, "
        f"{f.villain_position} {f.villain_nut_share * 100:.0f}%)",
        # Blocker value/bluff decomposition -- only when there's a real effect
        # (resolved in Python). The LLM must mirror this, never invent blockers.
        *(
            [
                f"BLOCKERS: you remove ~{f.blocked_value_pct * 100:.0f}% of "
                f"{f.villain_position}'s value combos and ~{f.blocked_bluff_pct * 100:.0f}% "
                f"of their bluff combos -- you mainly block their "
                f"{'VALUE' if f.blocker_effect == 'value' else 'BLUFFS'}"
            ]
            if f.blocker_effect != "neutral"
            else []
        ),
        # bb amounts snapped to the 0.5bb display grid so the model writes clean
        # numbers; SPR / break-even % come from the EXACT amounts (a ratio / a
        # percentage, no "ugly bb" problem). See pipeline.bb_display.
        f"POT: {round_to_half_bb(f.pot_bb):g}bb   "
        f"EFFECTIVE STACK: {round_to_half_bb(f.spot.node.effective_stack_bb):g}bb"
        f"   SPR: {f.spr:.1f}",
    ]
    if f.break_even_equity is not None:
        lines.append(
            f"FACING A BET: must call {round_to_half_bb(f.to_call_bb):g}bb; "
            f"break-even equity {f.break_even_equity * 100:.0f}%"
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
    "load_postflop_system_prompt",
    "placeholder_explanation",
]
