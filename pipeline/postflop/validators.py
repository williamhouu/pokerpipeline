"""Layer 7 (postflop): deterministic post-LLM validators.

All checks here are pure Python over the LLM's
:class:`~pipeline.explanation_generator.GeneratedExplanation` and the
:class:`~pipeline.postflop.facts.PostflopFacts` data block -- no second LLM
call. Two tiers, mirroring the preflop split:

* **Hard** (:func:`run_postflop_audit_validators`): a failure rejects the
  generation; the batch retries once with the failure message as corrective
  feedback, then routes to human review.
* **Soft** (:func:`run_postflop_soft_validators`): never reject -- return
  warning strings that mark the row ``flagged`` for a human to glance at.
  Tuned for precision (a flag should mean "a human should actually look").

Postflop differs from preflop in one important way for card checks: there IS
a board, so prose may legitimately name board cards as well as hero's hole
cards. :func:`validate_card_suit_consistency` allows both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.explanation_generator import BANNED_LITERAL_PHRASES, GeneratedExplanation
from pipeline.postflop.facts import PostflopFacts

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# A specific card in prose: rank immediately followed by a suit emoji.
_CARD_EMOJI = re.compile(r"([AKQJT2-9])([♠♥♦♣❤])")
_SUIT_LETTER = {"♠": "s", "♥": "h", "❤": "h", "♦": "d", "♣": "c"}
_BANNED_PUNCT = tuple(re.compile(p) for p in (r"—", r";", r"–"))


@dataclass(frozen=True)
class PostflopValidationResult:
    """Outcome of one hard validator (composable)."""

    is_valid: bool
    error_message: str = ""

    @classmethod
    def ok(cls) -> PostflopValidationResult:
        return cls(is_valid=True)

    @classmethod
    def fail(cls, message: str) -> PostflopValidationResult:
        return cls(is_valid=False, error_message=message)


# --- hard validators --------------------------------------------------------
def validate_correct_answer(
    generated: GeneratedExplanation,
    facts: PostflopFacts,  # noqa: ARG001 -- signature uniformity
) -> PostflopValidationResult:
    """``correct_answer`` must equal one of the (non-empty) options exactly."""
    options = generated.options()
    if generated.correct_answer not in options:
        return PostflopValidationResult.fail(
            f"correct_answer {generated.correct_answer!r} is not one of the "
            f"options {options!r}; it must match an option verbatim."
        )
    return PostflopValidationResult.ok()


def validate_banned_phrases(
    generated: GeneratedExplanation,
    facts: PostflopFacts,  # noqa: ARG001
) -> PostflopValidationResult:
    """Ban em dashes, semicolons, and the team's literal-phrase blocklist."""
    text = generated.answer_explanation or ""
    if not text:
        return PostflopValidationResult.ok()
    hits: list[str] = []
    for pattern in _BANNED_PUNCT:
        if pattern.search(text):
            hits.append(f"banned punctuation {pattern.pattern!r}")
    low = text.lower()
    for phrase in BANNED_LITERAL_PHRASES:
        if isinstance(phrase, str) and len(phrase) > 1 and phrase.lower() in low:
            hits.append(f"banned phrase {phrase!r}")
    if hits:
        return PostflopValidationResult.fail(
            "explanation uses banned punctuation or phrases: "
            + "; ".join(hits)
            + ". Rewrite in the team's voice (no em dashes, no semicolons, no "
            "template/corporate phrasing)."
        )
    return PostflopValidationResult.ok()


def validate_card_suit_consistency(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> PostflopValidationResult:
    """Specific cards named in prose must be hero's hole cards or the board.

    Postflop the legal specific cards are hero's two cards PLUS every board
    card (the LLM may reference the flop/turn/river). Anything else is an
    invented card -- the postflop analogue of preflop's "your K❤️" bug.
    """
    text = generated.answer_explanation or ""
    if not text:
        return PostflopValidationResult.ok()
    hero = {
        (facts.spot.hero_combo[i].upper(), facts.spot.hero_combo[i + 1].lower())
        for i in range(0, len(facts.spot.hero_combo) - 1, 2)
    }
    board = {(c[0].upper(), c[1].lower()) for c in facts.board}
    allowed = hero | board
    offending = sorted({
        f"{rank}{_SUIT_LETTER[suit]}"
        for rank, suit in _CARD_EMOJI.findall(text)
        if (rank, _SUIT_LETTER[suit]) not in allowed
    })
    if offending:
        return PostflopValidationResult.fail(
            "the explanation names specific cards that are neither yours nor "
            "on the board: " + ", ".join(offending)
            + f". Your cards are {facts.spot.hero_combo}; the board is "
            f"{' '.join(facts.board)}. Only name those cards."
        )
    return PostflopValidationResult.ok()


def run_postflop_audit_validators(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> PostflopValidationResult:
    """Run every hard validator in series; return the first failure or ok."""
    for check in (
        validate_correct_answer,
        validate_banned_phrases,
        validate_card_suit_consistency,
    ):
        result = check(generated, facts)
        if not result.is_valid:
            return result
    return PostflopValidationResult.ok()


# --- soft validators (flag, never reject) -----------------------------------
# Coarse action buckets: fold / passive (check, call) / aggressive (bet,
# raise). We flag only a clear conflict, never a sizing/phrasing nuance.
_VERB_BUCKET = {
    "fold": "fold", "check": "passive", "call": "passive",
    "bet": "aggressive", "raise": "aggressive",
}
_BUCKET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfold(?:s|ed|ing)?\b", re.I), "fold"),
    (re.compile(r"\b(?:bet|bets|betting|raise|raises|raising)\b", re.I), "aggressive"),
    (re.compile(r"\b(?:check|checks|checking|call|calls|calling)\b", re.I), "passive"),
)
_PCT_FIGURE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
_EQUITY_CUES = ("equity", "favorite", "favourite")
_NUMBER_TOLERANCE_PCT = 10.0


def soft_validate_verdict_vs_answer(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> list[str]:
    """Warn when the opening verdict's action conflicts with the answer.

    The first sentence is the verdict. We collect the action buckets it names
    (fold / passive / aggressive). If the dominant action's bucket is among
    them, pass; otherwise the verdict talks about a different line than the
    answer -- flag for review. Same high-precision design as the preflop soft
    check (a hedge that names the answer passes).
    """
    text = generated.answer_explanation or ""
    dominant_bucket = _VERB_BUCKET.get(facts.dominant_verb)
    if not text or dominant_bucket is None:
        return []
    first = _SENTENCE_SPLIT.split(text)[0]
    present = {bucket for pat, bucket in _BUCKET_PATTERNS if pat.search(first)}
    if not present or dominant_bucket in present:
        return []
    return [
        f"the opening verdict does not mention the answer action "
        f"({facts.dominant_action!r}, a {dominant_bucket} play). Review "
        "whether the first sentence states the right action."
    ]


def soft_validate_equity_vs_data(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> list[str]:
    """Warn when a cited equity % clearly contradicts the data block.

    Postflop v1 is heads-up, so hero_equity_vs_villain is the reference for the
    hand's equity. But a number "in an equity context" can legitimately be EITHER
    hero's equity OR the pot-odds **break-even** price -- well-written bluff-catch
    prose says "you only need 20% equity to continue and you have 35%", citing
    both. So a figure that matches either reference (within tolerance) is fine;
    only a number matching NEITHER is flagged as possibly invented.
    """
    text = generated.answer_explanation or ""
    if not text:
        return []
    target = facts.hero_equity_vs_villain * 100.0
    break_even = (
        facts.break_even_equity * 100.0 if facts.break_even_equity is not None else None
    )
    for m in _PCT_FIGURE.finditer(text):
        value = float(m.group(1))
        if value <= 0 or value >= 100:
            continue
        window = text[max(0, m.start() - 55): m.end() + 30].lower()
        if not any(cue in window for cue in _EQUITY_CUES):
            continue
        if "range" in window and "your range" in window:
            continue  # a range-equity claim, a different figure
        if abs(value - target) <= _NUMBER_TOLERANCE_PCT:
            continue  # matches hero's equity
        if break_even is not None and abs(value - break_even) <= _NUMBER_TOLERANCE_PCT:
            continue  # matches the pot-odds break-even price ("need X% to call")
        be_note = f", break-even {break_even:.0f}%" if break_even is not None else ""
        return [
            f"prose cites {value:g}% equity, but the data block has hero "
            f"equity {target:.0f}%{be_note}. Review the number (the LLM may "
            "have invented it)."
        ]
    return []


def run_postflop_soft_validators(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> list[str]:
    """Run every soft validator; return all warnings (empty == clean)."""
    warnings: list[str] = []
    for check in (soft_validate_verdict_vs_answer, soft_validate_equity_vs_data):
        warnings.extend(check(generated, facts))
    return warnings


__all__ = [
    "PostflopValidationResult",
    "run_postflop_audit_validators",
    "run_postflop_soft_validators",
    "soft_validate_equity_vs_data",
    "soft_validate_verdict_vs_answer",
    "validate_banned_phrases",
    "validate_card_suit_consistency",
    "validate_correct_answer",
]
