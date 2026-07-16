"""Layer 7 claim checker -- a second LLM pass that audits one explanation.

WHAT IT IS (in plain terms): after the explanation is written, we send it back
to the model with exactly one job -- go through it claim by claim and flag any
poker assertion that is WRONG, while leaving correct coaching and style alone.
It returns a structured list of problems; it does NOT rewrite anything.

WHY: the deterministic facts in the SOLVER DATA block are always right, and the
mechanical validators catch specific error types (blocker / terminology / suit
/ position). But the LLM-written prose around the facts can still slip in a
small, novel poker error that no mechanical rule anticipates -- the "90 percent
perfect, one tiny wrong thing" failure (e.g. "AK pays off your set", "wheelish
straights", reversed blocker logic, ICM talk in a cash game). Catching an
*arbitrary* subtle poker error needs poker judgement, which a second focused
pass supplies where a regex cannot. This is the brief's Layer 7 strategy
checker.

HOW IT IS MEANT TO BE USED: as a SOFT signal first (flag for human review), not
a hard reject -- the checker is itself an LLM with its own error rate, so we
calibrate it on real batches (what does it catch, how many false positives)
before trusting it to gate. It FAILS OPEN: if the checker call errors or returns
unparseable output, the result is "no issues", so a checker malfunction never
blocks a good explanation.

It costs one extra LLM call per explanation, so wiring it into generation should
be opt-in (a flag on the batch), not on by default.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pipeline.explanation_generator import (
    DEFAULT_MODEL,
    _extract_text,
    call_messages_create,
)

# The checker's output is a short JSON list, so it needs little room.
_CHECKER_MAX_TOKENS = 1024

CHECKER_SYSTEM_PROMPT = (
    "You are a sharp No-Limit Hold'em poker editor reviewing ONE answer "
    "explanation for a single preflop spot. You are given a SOLVER DATA block "
    "of ground-truth facts and the ANSWER EXPLANATION written about them. Your "
    "main job is to flag prose a thoughtful student would find CONFUSING, "
    "MISLEADING, or SELF-CONTRADICTORY -- plus any outright poker error. You "
    "do not rewrite anything.\n\n"
    "PRIORITY 1 -- CLARITY AND INTERNAL COHERENCE (this is the main job). The "
    "explanation is fed the solver's numbers and almost always quotes them "
    "correctly, so do NOT hunt for tiny data mismatches. What you ARE looking "
    "for is prose that confuses or misleads the reader, or whose own logic "
    "does not hang together. Flag the exact phrase when:\n"
    "  (a) A number is framed to imply the OPPOSITE of what it says. Flagship "
    "example: 'you only have about 41% equity' while the same explanation says "
    "you need about 36% to call -- 41% is MORE than 36%, so the word 'only' "
    "wrongly implies the price is bad. (The real reason to fold may be poor "
    "realization, but the equity-vs-price sentence as written argues for a "
    "call while the verdict is fold.) Any time the stated equity beats the "
    "stated price but the prose treats it as insufficient (or vice versa), "
    "flag it.\n"
    "  (b) The stated reasoning does not support the verdict, or a sentence "
    "undercuts the conclusion without resolving it, so a reader is left "
    "thinking 'wait, that sounds like a call, but the answer is fold'.\n"
    "  (c) Two parts of the explanation contradict each other, or the same "
    "fact is described two different ways.\n"
    "  (d) A cause-and-effect or comparison is stated backwards (a position, "
    "blocker, or domination relationship that is the reverse of the truth), or "
    "a key sentence is simply hard to follow.\n"
    "  (e) A CONDITIONAL or FUTURE outcome is stated as if it were CERTAIN. "
    "Flagship: 'you would be playing a 3-bet pot multiway' or 'you'll be "
    "three-way'. Check `dominant_action` and "
    "`your_call_or_fold_closes_the_action`. If the recommended action is fold, "
    "hero plays NO pot at all, so 'you would be playing / you'd be in' a pot is "
    "a hypothetical that must read 'if you called...', not a statement of fact. "
    "And if `your_call_or_fold_closes_the_action` is false (players are STILL "
    "TO ACT, per `still_to_act_after_you`), the pot going multiway also depends "
    "on those players continuing, so 'would' overstates it -- the honest word "
    "is 'could' or 'if'. Flag the phrase whenever an unsettled result (the pot "
    "going multiway, a player still to act continuing, hitting a card on a "
    "later street) is written as a certainty instead of a possibility.\n\n"
    "PRIORITY 2 -- CLEAR POKER ERRORS (flag these too, but they are rarer). A "
    "non-pair 'paying off' a set, reversed domination, a disconnected hand "
    "called a straight draw, ICM in a cash game, an equity number or the word "
    "coinflip pinned to ONE named hand instead of a whole range, a stated "
    "position that is backwards, or an action-mix description that disagrees "
    "with action_frequencies (e.g. calling it '3-bet or fold' when the hand "
    "also calls a meaningful share). Do NOT flag 'always'/'mostly'/'never' "
    "wording on a near-pure action -- those are the question's option buckets, "
    "so an action at, say, 98 percent stated as 'always' is correct.\n\n"
    "BE CONSERVATIVE. Flag a phrase ONLY if you are confident a careful reader "
    "would actually trip on it or be misled. Every entry you return asserts "
    "the prose is genuinely confusing or wrong, so if something is merely "
    "terse, stylistic, slightly imprecise-but-defensible, or a matter of "
    "taste, OMIT it -- never include a hedged or borderline entry. Do NOT flag "
    "correct coaching, emphasis, or length, and do NOT flag two numbers that "
    "actually agree (one in eight is about 12 percent).\n\n"
    "Return a single JSON object: {\"issues\": [{\"claim\": \"<the exact "
    "phrase from the explanation>\", \"problem\": \"<one sentence on why it "
    "confuses the reader or is wrong>\"}]}. If nothing is wrong, return "
    "{\"issues\": []}. Output only the JSON, with no preamble and no code "
    "fence."
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


def build_checker_user_prompt(explanation: str, solver_data: dict[str, Any]) -> str:
    """The per-call user message: the data block, then the explanation."""
    return (
        "SOLVER DATA:\n"
        f"{json.dumps(solver_data, indent=2, default=str)}\n\n"
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


def check_explanation_claims(
    explanation: str,
    solver_data: dict[str, Any],
    client: Any,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    system_prompt: str = CHECKER_SYSTEM_PROMPT,
    usage_callback: Any = None,
) -> ClaimCheckResult:
    """Audit one explanation against its SOLVER DATA block via a checker LLM
    call. Returns a structured verdict (see module docstring). Fails open.

    ``system_prompt`` defaults to :data:`CHECKER_SYSTEM_PROMPT` but the admin
    panel can pass an edited version so the checker prompt is tunable like the
    explanation prompts.

    ``usage_callback`` (July 2026 spend-logger fix): the preflop 5-arg
    convention ``(model, in, out, cache_creation, cache_read)``, same as the
    reviser. Before this, every checker call (the gate, the best-of-2 passes,
    the final audit) burned tokens INVISIBLY -- the lifetime-spend ledger
    counted generation + reviser only, so audited batches under-reported by
    roughly the checker's share. Any new LLM call site MUST report usage.
    """
    # The admin panel drives the batch with client=None (generation lazily
    # creates its own client; the checker must too, or it dies on a None
    # client and the claim_check column comes back empty -- the checker
    # silently does nothing). Mirror the generation pattern exactly.
    if client is None:
        import os  # noqa: PLC0415

        from anthropic import Anthropic  # noqa: PLC0415

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user = build_checker_user_prompt(explanation, solver_data)
    response = call_messages_create(
        client,
        model=model,
        max_tokens=_CHECKER_MAX_TOKENS,
        temperature=temperature,
        system=system_prompt or CHECKER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    if usage_callback is not None:
        from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
            _extract_usage,
        )

        usage_callback(model, *_extract_usage(response))
    return parse_checker_response(_extract_text(response))


def claim_check_to_json(result: ClaimCheckResult) -> str:
    """Serialize a result for the ``claim_check`` CSV column.

    Always returns a JSON list (``"[]"`` when clean, a list of issues
    otherwise) so a row that WAS checked is distinguishable from one that was
    not (the batch leaves the column ``""`` when the checker didn't run).
    """
    return json.dumps(
        [{"claim": i.claim, "problem": i.problem} for i in result.issues],
        separators=(",", ":"),
    )


def parse_claim_check(cell: str) -> list[dict[str, str]]:
    """Read a ``claim_check`` cell back into issue dicts (app side). Tolerant:
    blank/malformed -> empty list (panel hides itself / shows 'clean')."""
    if not cell or not cell.strip():
        return []
    try:
        data = json.loads(cell)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


__all__ = [
    "CHECKER_SYSTEM_PROMPT",
    "ClaimCheckResult",
    "ClaimIssue",
    "build_checker_user_prompt",
    "check_explanation_claims",
    "claim_check_to_json",
    "parse_checker_response",
    "parse_claim_check",
]
