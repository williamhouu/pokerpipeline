"""Layer 7 claim checker for PLO -- a second LLM pass auditing one explanation.

The PLO port of :mod:`pipeline.preflop.claim_checker` (July 2026), kept
self-contained like every PLO module (no cross-pipeline imports; only the
shared ``pipeline.explanation_generator`` leaf). Same contract as NLHE:

* FLAG-ONLY -- it returns a structured list of problems and rewrites nothing;
  flags land in the ``claim_check`` column + the meta record for human review.
* FAILS OPEN -- an errored call or unparseable output is a clean pass, so a
  checker malfunction never blocks a good explanation.
* OPT-IN -- one extra LLM call per question (``run_claim_checker`` on the
  batch), off by default.

The system prompt carries the calibration lessons the NLHE checkers earned
the hard way (June-July 2026): verify against the SOLVER DATA block only (no
independent poker theorizing), flag only what would genuinely mislead a
reader (no style/data-mismatch hunting), and never flag the Always/Mostly
option-bucket wording. PLO-specific bullets target the four-card failure
modes the deterministic shape validator can't fully cover: equity or
"flipping" pinned to one named hand, reversed suit-dominance, dangler talk
contradicting ``your_hand_shape``, and certainty about multiway pots that
are still unsettled.
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

_CHECKER_MAX_TOKENS = 1024

PLO_CHECKER_SYSTEM_PROMPT = (
    "You are a sharp Pot-Limit Omaha poker editor reviewing ONE answer "
    "explanation for a single PREFLOP spot. You are given a SOLVER DATA block "
    "of ground-truth facts (the four hero cards, the deterministic hand-shape "
    "description, the action strategy percentages, equity numbers when "
    "present, and the villain's range width) and the ANSWER EXPLANATION "
    "written about them. Your job is to flag prose a thoughtful student would "
    "find CONFUSING, MISLEADING, or SELF-CONTRADICTORY -- plus any outright "
    "poker error. You do not rewrite anything.\n\n"
    "PRIORITY 1 -- CLARITY AND INTERNAL COHERENCE. The explanation is fed the "
    "solver's numbers and almost always quotes them correctly, so do NOT hunt "
    "for tiny data mismatches. Flag the exact phrase when:\n"
    "  (a) A number is framed to imply the OPPOSITE of what it says (equity "
    "described as insufficient when it beats the stated price, or vice "
    "versa).\n"
    "  (b) The stated reasoning does not support the verdict, so a reader is "
    "left thinking the opposite action sounds better without that tension "
    "being resolved.\n"
    "  (c) Two parts of the explanation contradict each other, or the same "
    "fact is described two different ways.\n"
    "  (d) A CONDITIONAL or FUTURE outcome is stated as certain: if the "
    "recommended action is fold, hero plays no pot at all ('you would be "
    "playing a bloated pot' must read 'if you called...'); a pot only goes "
    "multiway if the remaining players actually continue.\n\n"
    "PRIORITY 2 -- CLEAR PLO ERRORS (rarer; the deterministic validators "
    "already catch fabricated cards and hand-shape claims, so only flag what "
    "they cannot see):\n"
    "  (a) An equity number or the word 'flip'/'coinflip' pinned to ONE named "
    "villain hand instead of the villain's whole range.\n"
    "  (b) Suit-dominance stated backwards (e.g. calling a king-high suit "
    "'the nut flush draw', or treating hero's ace-high suit as dominated).\n"
    "  (c) Prose that contradicts the block's `your_hand_shape`, "
    "`flush_potential`, `card_redundancy`, or `suit_redundancy` fields -- "
    "e.g. praising 'four connected cards working together' when the shape "
    "says there is a dangler, or touting set-mining with a redundancy note "
    "saying trips leave only one out.\n"
    "  (d) An action-mix description that disagrees with `action_strategy` "
    "(calling a spot '3-bet or fold' when the hand also calls a meaningful "
    "share).\n"
    "  (e) NLHE logic imported into PLO: treating a bare pair as a strong "
    "made hand preflop, blocker claims about single cards deciding ranges, "
    "or ICM talk in a cash game.\n\n"
    "Do NOT flag 'always'/'mostly'/'never' wording that mirrors the "
    "question's option buckets (an action at 98 percent stated as the main "
    "play is correct). BE CONSERVATIVE: flag a phrase ONLY if you are "
    "confident a careful reader would trip on it or be misled; if something "
    "is terse, stylistic, or imprecise-but-defensible, OMIT it. Never emit "
    "an entry whose problem amounts to 'this is fine' or 'no real issue' -- "
    "if a phrase is acceptable, it does not belong in the list at all.\n\n"
    "Return a single JSON object: {\"issues\": [{\"claim\": \"<the exact "
    "phrase from the explanation>\", \"problem\": \"<one sentence on why it "
    "confuses the reader or is wrong>\"}]}. If nothing is wrong, return "
    "{\"issues\": []}. Output only the JSON, with no preamble and no code "
    "fence."
)


@dataclass(frozen=True)
class PloClaimIssue:
    """One flagged claim from the checker."""

    claim: str
    problem: str


@dataclass(frozen=True)
class PloClaimCheckResult:
    """The checker's verdict for one explanation."""

    passed: bool
    issues: tuple[PloClaimIssue, ...]
    raw: str = ""


def build_checker_user_prompt(explanation: str, solver_data: dict[str, Any]) -> str:
    """The per-call user message: the data block, then the explanation."""
    return (
        "SOLVER DATA:\n"
        f"{json.dumps(solver_data, indent=2, default=str)}\n\n"
        "ANSWER EXPLANATION:\n"
        f"{explanation}\n"
    )


def parse_checker_response(text: str) -> PloClaimCheckResult:
    """Parse the checker's JSON. Fails OPEN: unparseable output becomes a
    clean pass (raw kept for debugging)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned[:4].lower() == "json":
            cleaned = cleaned[4:].strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return PloClaimCheckResult(passed=True, issues=(), raw=text)
    raw_issues = data.get("issues", []) if isinstance(data, dict) else []
    issues = tuple(
        PloClaimIssue(
            claim=str(d.get("claim", "")).strip(),
            problem=str(d.get("problem", "")).strip(),
        )
        for d in raw_issues
        if isinstance(d, dict) and (d.get("claim") or d.get("problem"))
    )
    # Deterministic guard (July 2026): the prompt forbids 'no real issue'
    # entries, but the model still occasionally narrates one ("...but the
    # phrasing is acceptable. Not a real issue."). A self-retracted entry is
    # NOT a finding -- shipping it flags a clean row and highlights a
    # non-problem on Review -- so drop it here where the rule is enforceable.
    issues = tuple(i for i in issues if not _self_retracted(i.problem))
    return PloClaimCheckResult(passed=not issues, issues=issues, raw=text)


#: Explicit self-retractions only -- softer phrases ("is acceptable",
#: "is fine") also appear inside GENUINE findings that quote the prose they
#: are refuting, so matching them would drop real flags.
_RETRACTION_MARKERS = (
    "not a real issue",
    "no real issue",
    "not an issue",
    "not a genuine issue",
)


def _self_retracted(problem: str) -> bool:
    """True when the issue's own problem text talks itself out of the flag."""
    low = problem.lower()
    return any(m in low for m in _RETRACTION_MARKERS)


def check_plo_explanation_claims(
    explanation: str,
    solver_data: dict[str, Any],
    client: Any,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    system_prompt: str = PLO_CHECKER_SYSTEM_PROMPT,
    usage_callback: Any = None,
) -> PloClaimCheckResult:
    """Audit one PLO explanation against its SOLVER DATA block. Fails open.

    Mirrors the NLHE checker's client handling: a ``None`` client (the admin
    path) lazily creates one, so the checker never silently no-ops.

    ``usage_callback`` (July 2026 spend-logger fix): the PLO 5-arg convention
    ``(model, in, out, cache_creation, cache_read)``, same as generation, so
    the checker's tokens reach the admin cost readout + lifetime ledger. Any
    new LLM call site MUST report usage."""
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
        system=system_prompt or PLO_CHECKER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user}],
    )
    if usage_callback is not None:
        from pipeline.plo.explanation_generator import _extract_usage  # noqa: PLC0415

        usage_callback(model, *_extract_usage(response))
    return parse_checker_response(_extract_text(response))


def claim_check_to_json(result: PloClaimCheckResult) -> str:
    """Serialize for the ``claim_check`` CSV column: always a JSON list
    (``"[]"`` when clean) so a checked row is distinguishable from an
    unchecked one (``""``)."""
    return json.dumps(
        [{"claim": i.claim, "problem": i.problem} for i in result.issues],
        separators=(",", ":"),
    )


def parse_claim_check(cell: str) -> list[dict[str, str]]:
    """Read a ``claim_check`` cell back into issue dicts (app side).
    Tolerant: blank/malformed -> empty list."""
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
    "PLO_CHECKER_SYSTEM_PROMPT",
    "PloClaimCheckResult",
    "PloClaimIssue",
    "build_checker_user_prompt",
    "check_plo_explanation_claims",
    "claim_check_to_json",
    "parse_checker_response",
    "parse_claim_check",
]
