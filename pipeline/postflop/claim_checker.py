"""Layer 7 (postflop): a second LLM pass that audits one explanation.

The postflop analogue of :mod:`pipeline.preflop.claim_checker`. After the
explanation is written, we send it back to the model with one job: go through
it claim by claim and flag any poker assertion that is WRONG or any prose that
would confuse a thoughtful student, while leaving correct coaching and style
alone. It returns a structured list of problems; it does NOT rewrite anything
(that is the reviser's job).

WHY postflop needs its own checker: the deterministic SOLVER DATA block is
always right and the mechanical validators catch specific error types (banned
phrases / invented cards / garbled glyphs). But the prose around the facts can
still slip in a subtle poker error no regex anticipates, and postflop's error
surface is wider than preflop's: a misread board, a miscounted draw, a made
hand called something it is not, and -- the brief's flagship LLM failure --
the RANGE ADVANTAGE assigned to the wrong player. Catching an arbitrary subtle
postflop error needs poker judgement, which a focused second pass supplies.

It is a SOFT signal (flag for human review / feed the reviser), never a hard
gate, and it FAILS OPEN: a checker call that errors or returns unparseable
output becomes "no issues", so a checker malfunction never blocks a good
explanation. One extra LLM call per explanation, so it is opt-in (a batch flag).

Self-contained on purpose: like the rest of ``pipeline.postflop`` it imports
only the shared leaf utilities (``call_messages_create`` / ``_extract_text``),
never another pipeline's module. The JSON column format it emits is identical
to the preflop checker's so the admin Review panel reads both with one parser.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from pipeline.explanation_generator import _extract_text, call_messages_create
from pipeline.postflop.explanation_generator import DEFAULT_MODEL

# The checker's output is a short JSON list, so it needs little room.
_CHECKER_MAX_TOKENS = 1024

_CHECKER_CALIBRATION = (
    "\nCALIBRATION -- WHAT NOT TO FLAG:\n"
    "- Flag ONLY claims that CONTRADICT the SOLVER DATA or the QUESTION's "
    "action history. Style, tone, emphasis, and word choice are out of "
    "scope.\n"
    "- If you find yourself writing 'minor', 'slightly', 'defensible', "
    "'imprecise but', or 'technically', do NOT flag it -- that is the "
    "signal the claim is acceptable.\n"
    "- A number that matches the data within rounding is correct; do not "
    "flag how it is framed unless the framing reverses its meaning.\n"
)

POSTFLOP_CHECKER_SYSTEM_PROMPT = (
    "You are a sharp No-Limit Hold'em poker editor reviewing ONE answer "
    "explanation for a single POSTFLOP spot (a flop, turn, or river decision). "
    "You are given the QUESTION (the exact spot, with the FULL action history "
    "that led to this decision), a SOLVER DATA block of ground-truth facts, and "
    "the ANSWER EXPLANATION written about them. Your main job is to flag prose a "
    "thoughtful student would find CONFUSING, MISLEADING, or SELF-CONTRADICTORY, "
    "plus any outright poker error. You do not rewrite anything.\n\n"
    "THE QUESTION IS THE SOURCE OF TRUTH FOR THE ACTION SEQUENCE. It spells out "
    "every action on every street -- who opened, bet, checked, called, raised, "
    "or led, and for how much. The SOLVER DATA block does NOT repeat the line; "
    "it only names the preflop aggressor and the current street. So NEVER flag a "
    "reference to the line as 'invented' or 'not in the data' just because the "
    "data block omits it -- VERIFY it against the QUESTION, and flag it only if "
    "it actually contradicts the QUESTION. A donk-lead, a check-raise, or a probe "
    "by the player who is NOT the preflop aggressor is a normal line and will be "
    "in the QUESTION even though the data block names the other player the "
    "aggressor. (Inventing a specific RANGE composition the facts don't support "
    "is still flaggable -- but judge that against the RANGE / NUT lines, not the "
    "action sequence.)\n\n"
    "PRIORITY 1 -- CLARITY AND INTERNAL COHERENCE (the main job). The "
    "explanation is fed the solver's numbers and almost always quotes them "
    "correctly, so do NOT hunt for tiny data mismatches. Flag the exact phrase "
    "when:\n"
    "  (a) Equity is framed against the price backwards. The FACING A BET line "
    "gives a break-even price; HERO EQUITY is the hand's share. If the prose "
    "says the price is not met / hero is 'priced out' while the stated equity "
    "is actually ABOVE the break-even number (or vice versa), flag it. A fold "
    "with the price met should say the range is too strong or the spot is "
    "close, NOT that the price is unmet.\n"
    "  (b) The stated reasoning does not support the verdict, or a sentence "
    "undercuts the conclusion without resolving it, so a reader thinks 'wait, "
    "that sounds like a call, but the answer is fold'.\n"
    "  (c) Two parts contradict each other, or the same fact is described two "
    "different ways.\n"
    "  (d) A relationship is stated backwards, OR a clause jams two different "
    "ideas together so its subject no longer fits its predicate, OR a key "
    "sentence is simply hard to follow and a careful reader must re-read it to "
    "parse. Example to flag: 'the hands that continue either beat you or fold out "
    "the bluffs you already crush' -- hands that CALL do not fold anything out; "
    "folding out worse hands is what BETTING does, so the clause is incoherent.\n"
    "  (e) A future or conditional outcome on a later street is stated as a "
    "certainty ('you win when the flush comes' as if guaranteed) instead of a "
    "possibility.\n\n"
    "PRIORITY 2 -- CLEAR POSTFLOP POKER ERRORS (rarer, flag these too):\n"
    "  * RANGE ADVANTAGE / NUT ADVANTAGE assigned to the WRONG player. The data "
    "block resolves who is ahead as a range and who holds the nutted hands; if "
    "the prose says hero has the range or nut advantage when the fact gives it "
    "to villain (or 'roughly even'), or vice versa, flag it. This is the single "
    "most common error -- check it whenever the prose discusses who the board "
    "favors.\n"
    "  * A made hand called something it is not (a single pair described as two "
    "pair or a set, a draw described as a made hand), or a hand 'beating' a "
    "holding it actually loses to.\n"
    "  * A draw mislabeled: the DRAWS line names the exact type, so a gutshot "
    "called open-ended, or a non-nut draw called 'the nuts', is an error. If "
    "the prose names a draw type absent from the DRAWS line, flag it.\n"
    "  * Blocker claims that CONTRADICT or go beyond the BLOCKERS fact. The data "
    "block has a BLOCKERS line ONLY when hero meaningfully removes villain's "
    "value or bluffs, and it says WHICH. Flag prose that reverses it (claims "
    "'you block their bluffs' when the fact says value, or vice versa), that "
    "discusses blockers when there is NO BLOCKERS line, or that invents a "
    "specific combo count. Correct value/bluff blocker talk that matches the "
    "fact is fine.\n"
    "  * An equity number, or the words flip / coinflip / race, pinned to ONE "
    "named villain hand instead of a range. ICM talk in a cash game. A bet size "
    "named that the CORRECT ACTION did not use, or an argument that a different "
    "size/line is better than the solver's.\n"
    "  * An action-mix description that disagrees with the solver frequency "
    "(calling a 60/40 mix a clear/automatic decision). Do NOT flag "
    "always/mostly/never wording on a near-pure action -- those are the "
    "question's option buckets, so a 95%+ action stated as 'always' is correct. "
    "Do NOT read a bet size ('a 33% bet', '67% pot') as an equity figure.\n"
    "  * Equity mis-attributed to draws when it is showdown equity. The "
    "CURRENTLY AHEAD line says what share of villain's range hero's hand already "
    "beats. If hero beats a large share now and there is NO DRAWS line, but the "
    "prose pins hero's equity on draws, outs, backdoors, or 'spiking' a card, "
    "flag it -- a made hand ahead of villain's air has SHOWDOWN equity, not draw "
    "equity. (A real draw, named on a DRAWS line, may of course be credited.)\n\n"
    "BE CONSERVATIVE. Flag a phrase ONLY if you are confident a careful reader "
    "would actually trip on it or be misled. Every entry asserts the prose is "
    "genuinely confusing or wrong, so if something is merely terse, stylistic, "
    "slightly imprecise-but-defensible, or a matter of taste, OMIT it. Do NOT "
    "flag correct coaching, emphasis, or length.\n\n"
    "Return a single JSON object: {\"issues\": [{\"claim\": \"<the exact phrase "
    "from the explanation>\", \"problem\": \"<one sentence on why it confuses "
    "the reader or is wrong>\"}]}. If nothing is wrong, return {\"issues\": []}. "
    "Output only the JSON, with no preamble and no code fence."
    + _CHECKER_CALIBRATION
)


@dataclass(frozen=True)
class ClaimIssue:
    """One flagged claim from the checker."""

    claim: str      # the exact phrase the checker pulled from the explanation
    problem: str    # one sentence on why it is wrong


@dataclass(frozen=True)
class ClaimCheckResult:
    """The checker's verdict for one explanation."""

    passed: bool                       # True = no issues found
    issues: tuple[ClaimIssue, ...]     # the flagged claims (empty when passed)
    raw: str = ""                      # raw checker output, kept for debugging


def build_checker_user_prompt(
    explanation: str, solver_data_block: str, question: str = ""
) -> str:
    """The per-call user message: the QUESTION (line), the data block, the explanation.

    Postflop's data block is already a formatted string (from
    :func:`pipeline.postflop.explanation_generator.build_solver_data_block`),
    so it is embedded as-is rather than re-serialised from a dict. ``question``
    is the deterministic action-history narrative -- the SAME string the reader
    and the writer see. It is the source of truth for the LINE, so the checker
    stops false-flagging true references to earlier-street action (a donk-lead, a
    check-raise) that the data block never restates. Empty -> omitted (the writer
    always passes it; only older direct callers leave it blank).
    """
    head = (
        f"QUESTION (the spot + full action history):\n{question}\n\n"
        if question.strip()
        else ""
    )
    return (
        f"{head}"
        "SOLVER DATA:\n"
        f"{solver_data_block}\n\n"
        "ANSWER EXPLANATION:\n"
        f"{explanation}\n"
    )


def parse_checker_response(text: str) -> ClaimCheckResult:
    """Parse the checker's JSON into a result. Fails OPEN: unparseable output
    becomes a clean pass (with the raw text kept), so a checker malfunction
    never blocks a good explanation."""
    cleaned = text.strip()
    if cleaned.startswith("```"):                       # tolerate a code fence
        cleaned = cleaned.strip("`").strip()
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return ClaimCheckResult(passed=True, issues=(), raw=text)
    raw_issues = data.get("issues", []) if isinstance(data, dict) else []
    issues = tuple(
        ClaimIssue(
            claim=str(d.get("claim", "")).strip(),
            problem=str(d.get("problem", "")).strip(),
        )
        for d in raw_issues
        if isinstance(d, dict) and (d.get("claim") or d.get("problem"))
    )
    return ClaimCheckResult(passed=not issues, issues=issues, raw=text)


def check_postflop_claims(
    explanation: str,
    solver_data_block: str,
    client: object,
    *,
    question: str = "",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    system_prompt: str = POSTFLOP_CHECKER_SYSTEM_PROMPT,
) -> ClaimCheckResult:
    """Audit one postflop explanation against its SOLVER DATA block via a
    checker LLM call. Returns a structured verdict (see module docstring). Fails
    open. ``question`` is the action-history narrative (the line); pass it so the
    checker verifies line references against the truth instead of flagging them
    as invented. ``system_prompt`` defaults to
    :data:`POSTFLOP_CHECKER_SYSTEM_PROMPT` but the admin panel can pass an edited
    version so it is tunable.
    """
    if client is None:
        import os  # noqa: PLC0415

        from anthropic import Anthropic  # noqa: PLC0415

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user = build_checker_user_prompt(explanation, solver_data_block, question)
    response = call_messages_create(
        client,
        model=model,
        max_tokens=_CHECKER_MAX_TOKENS,
        temperature=temperature,
        system=system_prompt or POSTFLOP_CHECKER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    return parse_checker_response(_extract_text(response))


def claim_check_to_json(result: ClaimCheckResult) -> str:
    """Serialize a result for the ``claim_check`` CSV column.

    Always returns a JSON list (``"[]"`` when clean, a list of issues
    otherwise) so a row that WAS checked is distinguishable from one that was
    not (the batch leaves the column ``""`` when the checker didn't run). The
    shape matches the preflop checker's, so the admin Review panel reads both.
    """
    return json.dumps(
        [{"claim": i.claim, "problem": i.problem} for i in result.issues],
        separators=(",", ":"),
    )


__all__ = [
    "POSTFLOP_CHECKER_SYSTEM_PROMPT",
    "ClaimCheckResult",
    "ClaimIssue",
    "build_checker_user_prompt",
    "check_postflop_claims",
    "claim_check_to_json",
    "parse_checker_response",
]
