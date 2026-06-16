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
    "You are a meticulous No-Limit Hold'em fact-checker. You are given a "
    "SOLVER DATA block of ground-truth facts for one preflop spot and an "
    "ANSWER EXPLANATION written about that spot. Your only job is to find poker "
    "claims in the explanation that are WRONG. You do not rewrite anything.\n\n"
    "Go through the explanation claim by claim. Flag a claim ONLY if it is one "
    "of these:\n"
    "  1. It CONTRADICTS the SOLVER DATA block: a number, action frequency, "
    "position, range, blocker, or label that disagrees with the data.\n"
    "  2. It is POKER-INCORRECT in general. Examples: claiming a non-pair like "
    "AK pays off your set (sets get paid by overpairs, not unpaired hands), "
    "reversing which hand dominates which, calling a disconnected hand a "
    "straight draw, mentioning ICM in a cash game, or attaching an equity "
    "number or the word coinflip to ONE named hand instead of a whole range.\n"
    "  3. It states a SPECIFIC fact that is neither in the data block nor a "
    "generally-true poker truth: an invented number, range, blocker, or read.\n"
    "  4. It MISDESCRIBES THE ACTION MIX or is INTERNALLY INCONSISTENT. Before "
    "finishing, explicitly compare how the explanation FRAMES the strategy "
    "against the action_frequencies numbers. Flag the exact phrase when they "
    "disagree. Concrete checks: (a) if the prose calls the spot 'polarized', a "
    "'3-bet-or-fold' spot, or says there is no flatting/calling range, but "
    "action_frequencies has a Call (or flat) frequency above ~10 percent, that "
    "is wrong; (b) if the prose calls the hand a 'bluff', or says it folds / "
    "gives up / is 'the weakest stuff', but action_frequencies shows Fold at or "
    "near 0 percent, that is wrong; (c) if two sentences in the explanation "
    "contradict each other. Do NOT flag 'always' / 'mostly' / 'never' wording on "
    "a near-pure action -- those are the question's option buckets, so an action "
    "at, say, 98 percent stated as 'always' is correct, not mixed.\n\n"
    "Be conservative: include a claim ONLY if you are CONFIDENT it is clearly "
    "wrong. Anything you list is an assertion that it IS wrong, so if you would "
    "call an entry minor, borderline, imprecise-but-ok, or 'not really wrong', "
    "OMIT it entirely -- never include a hedged entry. Do NOT flag correct "
    "general coaching, style, emphasis, length, or word choice when the "
    "underlying claim is right (e.g. calling a real blocker a 'key' blocker is "
    "fine). Do NOT flag two numbers that actually agree (one in eight is about "
    "12 percent). Do NOT flag whether a hand is in villain's range unless "
    "villain_stats clearly shows it is not. DO flag clear errors: a position "
    "that contradicts hero_position, a false domination claim, an impossible or "
    "absent blocker, an equity number or flip/coinflip pinned on one named "
    "hand, a number that disagrees with the data, or a strategy/action-mix "
    "description that disagrees with action_frequencies (e.g. calling it "
    "'3-bet or fold' when the hand also calls a meaningful share).\n\n"
    "Return a single JSON object: {\"issues\": [{\"claim\": \"<the exact "
    "phrase from the explanation>\", \"problem\": \"<one sentence on why it is "
    "wrong>\"}]}. If nothing is wrong, return {\"issues\": []}. Output only the "
    "JSON, with no preamble and no code fence."
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
) -> ClaimCheckResult:
    """Audit one explanation against its SOLVER DATA block via a checker LLM
    call. Returns a structured verdict (see module docstring). Fails open.

    ``system_prompt`` defaults to :data:`CHECKER_SYSTEM_PROMPT` but the admin
    panel can pass an edited version so the checker prompt is tunable like the
    explanation prompts.
    """
    user = build_checker_user_prompt(explanation, solver_data)
    response = call_messages_create(
        client,
        model=model,
        max_tokens=_CHECKER_MAX_TOKENS,
        temperature=temperature,
        system=system_prompt or CHECKER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
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
