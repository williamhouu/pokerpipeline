"""PLO explanation validators -- the reusable deterministic audit stack.

The PLO port of :mod:`pipeline.preflop.validators` (July 2026), kept
self-contained like every PLO module. Two tiers, same contract as NLHE:

* **Hard** (:func:`run_plo_audit_validators`) -- the SAME deterministic checks
  the Layer-6 generation retry loop applies (banned phrases, list formatting,
  fabricated cards, hand-shape claims), packaged so the Layer-7 reviser can
  re-validate a rewrite before it ships. A rewrite that breaks a hard rule is
  DISCARDED by the reviser and the original kept -- this module is that
  deterministic floor. The check implementations live in
  :mod:`pipeline.plo.explanation_generator` (single source of truth; the
  generation loop and this runner can never drift apart) and are re-exposed
  here behind the ``PloValidationResult`` interface.

* **Soft** (:func:`run_plo_soft_validators`) -- flag-for-review only, never a
  rejection. v1 is the position-wording check: the PLO claim checker's first
  live calibration (July 2026) found position REVERSALS were the #1 real
  failure mode ("you'll be out of position" on an in-position hero), and that
  contradiction is checkable deterministically against
  :func:`pipeline.plo.position.hero_relative_position` -- the same fact the
  SOLVER DATA block carries. Ported from the NLHE
  ``soft_validate_position_words`` with the PLO seat vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.explanation_generator import (
    GeneratedExplanation,
    find_internal_xml_tag,
)
from pipeline.plo.explanation_generator import (
    _banned_present,
    _fabricated_cards,
    _list_formatting_error,
    _shape_claim_errors,
    _terminology_errors,
)
from pipeline.plo.fact_extractor import PloFacts
from pipeline.plo.position import hero_relative_position
from pipeline.plo.range_examples import leaning_examples_for_spot


@dataclass(frozen=True)
class PloValidationResult:
    """Outcome of a validator run (mirrors ``PreflopValidationResult``)."""

    is_valid: bool
    error_message: str = ""

    @classmethod
    def ok(cls) -> PloValidationResult:
        return cls(is_valid=True)

    @classmethod
    def fail(cls, message: str) -> PloValidationResult:
        return cls(is_valid=False, error_message=message)


def _shape_exempt_phrases(facts: PloFacts) -> tuple[str, ...]:
    """Hand names quoted from the range-examples fact, exempted from the
    shape-claim audit -- EXACTLY as the generation loop computes them, so a
    rewrite is judged by the same rules as the first draft."""
    leaning = leaning_examples_for_spot(facts)
    if leaning is None:
        return ()
    hands = [str(h) for h in leaning.get("hands", [])]
    return tuple(hands) + tuple(h.split(" (")[0] for h in hands)


def validate_banned_phrases(
    generated: GeneratedExplanation, facts: PloFacts  # noqa: ARG001
) -> PloValidationResult:
    """Reject the brief's banned phrases (the regeneration-worthy subset)."""
    banned = _banned_present(generated.answer_explanation or "")
    if banned:
        return PloValidationResult.fail(f"used banned phrase(s): {banned}")
    return PloValidationResult.ok()


def validate_no_list_formatting(
    generated: GeneratedExplanation, facts: PloFacts  # noqa: ARG001
) -> PloValidationResult:
    """Reject bulleted/numbered-list formatting in the explanation prose."""
    error = _list_formatting_error(generated.answer_explanation or "")
    if error:
        return PloValidationResult.fail(error)
    return PloValidationResult.ok()


def validate_no_internal_xml(
    generated: GeneratedExplanation, facts: PloFacts  # noqa: ARG001
) -> PloValidationResult:
    """Reject internal/XML-style tags leaked into the prose (e.g. <thinking>).

    Guards the Opus 5 thinking-disabled failure mode (July 2026); prose never
    legitimately contains angle-bracket tags. Shared detector:
    :func:`pipeline.explanation_generator.find_internal_xml_tag`.
    """
    tag = find_internal_xml_tag(generated.answer_explanation or "")
    if tag is not None:
        return PloValidationResult.fail(
            f"explanation contains an internal tag {tag!r}. Do not include "
            "internal or system XML tags in your response -- return only the "
            "explanation prose."
        )
    return PloValidationResult.ok()


def validate_card_fabrication(
    generated: GeneratedExplanation, facts: PloFacts
) -> PloValidationResult:
    """Reject a specific card named in prose that isn't one of hero's four."""
    fabricated = _fabricated_cards(
        generated.answer_explanation or "", facts.spot.hero_cards
    )
    if fabricated:
        hand = " ".join(facts.spot.hero_cards)
        return PloValidationResult.fail(
            f"named card(s) not in your hand: {', '.join(fabricated)}. "
            f"Your exact hand is {hand} -- use only those four cards, "
            "and prefer describing the shape over reciting cards."
        )
    return PloValidationResult.ok()


def validate_terminology(
    generated: GeneratedExplanation, facts: PloFacts
) -> PloValidationResult:
    """Reject preflop-terminology misuse the action history refutes (v1:
    'limp' language when every call in the hand came after a raise)."""
    errors = _terminology_errors(generated.answer_explanation or "", facts)
    if errors:
        return PloValidationResult.fail(f"the explanation {errors[0]}")
    return PloValidationResult.ok()


def validate_shape_claims(
    generated: GeneratedExplanation, facts: PloFacts
) -> PloValidationResult:
    """Reject hero-hand shape claims the deterministic classifier says are
    false (invented danglers, wrong suit pattern, made-flush-preflop, ...)."""
    errors = _shape_claim_errors(
        generated.answer_explanation or "",
        facts.hand_class,
        exempt_phrases=_shape_exempt_phrases(facts),
    )
    if errors:
        return PloValidationResult.fail(
            "the explanation misstates the hand: "
            f"{'; '.join(errors)}. The hand is actually "
            f"'{facts.hand_class.descriptor}'. Describe the shape "
            "using only the your_hand_shape, card_redundancy, and "
            "suit_redundancy facts, and flush strength using the "
            "flush_potential fact's own wording (potential to make "
            "a flush, never a made flush, never a ranking the fact "
            "does not state)."
        )
    return PloValidationResult.ok()


def run_plo_audit_validators(
    generated: GeneratedExplanation,
    facts: PloFacts,
    *,
    allow_list_formatting: bool = False,
) -> PloValidationResult:
    """Run every hard validator in series; return the first failure or ok.

    The reviser calls this on every rewrite (mirroring the NLHE
    ``run_preflop_audit_validators`` contract): a rewrite that fails here is
    discarded (with one corrective retry fed the ``error_message``), so an
    auto-fix can never ship prose the generation loop itself would have
    rejected.

    ``allow_list_formatting``: pass True when the batch's system prompt is a
    factor-list prompt (``prompt_sanctions_lists``) -- there "- " lines ARE
    the requested voice, and the no-list rule must not reject them.
    """
    checks = [
        validate_banned_phrases,
        validate_no_list_formatting,
        validate_no_internal_xml,
        validate_card_fabrication,
        validate_terminology,
        validate_shape_claims,
    ]
    if allow_list_formatting:
        checks.remove(validate_no_list_formatting)
    for check in checks:
        result = check(generated, facts)
        if not result.is_valid:
            return result
    return PloValidationResult.ok()


# --- soft validators (flag, never fail) --------------------------------------
# Position phrases + subject cues, ported from the NLHE soft position check
# with the PLO 6-max seat vocabulary. A phrase only contradicts the position
# fact when it describes HERO: "the button calls in position" is about the
# villain and must not flag. Subjectless verdict sentences read as hero-bound.
_POS_IN = re.compile(r"\bin position\b", re.I)
_POS_OUT = re.compile(r"\bout of position\b", re.I)
_HERO_SUBJECT = re.compile(r"\b(?:you|your|yourself)\b", re.I)
_OTHER_SUBJECT = re.compile(
    r"\b(?:villain|opponent|the raiser|the opener|the limper|small blind|"
    r"big blind|under the gun|lojack|hijack|cutoff|button|UTG\+?[12]?|"
    r"LJ|HJ|CO|BU|BTN|SB|BB)\b",
    re.I,
)
# A negation right before the phrase ("not in position", "isn't in position")
# AGREES with an OOP hero, so it must not flag.
_POS_NEGATED = re.compile(r"(?:\bnot|n't)\s+$", re.I)
# A counterfactual sentence ("if you were closer to heads up in position
# you'd have a case") describes a DIFFERENT spot on purpose -- never flag it.
# Live false positive from the first 9-max batch (July 2026).
_POS_HYPOTHETICAL = re.compile(
    r"\b(?:if you|if we|had you|were you|would be|you'd be)\b", re.I
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _position_claim_is_hero_bound(before: str) -> bool:
    """Whether a position phrase describes hero, judged from its subject slot
    (the sentence text preceding the phrase)."""
    if _HERO_SUBJECT.search(before):
        return True
    return not _OTHER_SUBJECT.search(before)


def soft_validate_position_words(
    generated: GeneratedExplanation, facts: PloFacts
) -> list[str]:
    """Warn when a HERO-bound position word contradicts the position fact.

    SOFT on purpose: a contradiction is usually a real reversal (the PLO
    claim checker's #1 live catch), but can be a legitimate
    multiway-realization note ("you'd be out of position against the callers
    behind"), so it flags for review instead of failing.
    """
    text = generated.answer_explanation or ""
    if not text:
        return []

    relative = hero_relative_position(facts)
    if relative == "In Position":
        phrase_re, claim = _POS_OUT, "out of position"
    elif relative == "Out of Position":
        phrase_re, claim = _POS_IN, "in position"
    else:
        return []

    for sentence in _SENTENCE_SPLIT.split(text):
        if _POS_HYPOTHETICAL.search(sentence):
            continue  # counterfactual: describes a different spot on purpose
        for m in phrase_re.finditer(sentence):
            before = sentence[: m.start()]
            if _POS_NEGATED.search(before):
                continue
            if _position_claim_is_hero_bound(before):
                return [
                    f"prose says '{claim}' about hero, but hero is "
                    f"{relative} vs the villain. Review (may be a legit note "
                    "about players still to act behind)."
                ]
    return []


def run_plo_soft_validators(
    generated: GeneratedExplanation, facts: PloFacts
) -> list[str]:
    """Run every soft validator; return all warnings (empty = clean).

    Soft warnings never fail a generation and never trigger a retry. The
    batch driver marks warned rows ``validation_status='flagged'`` and
    records the warnings in the meta sidecar so the PLO Review page surfaces
    them for a human.
    """
    warnings: list[str] = []
    for check in (soft_validate_position_words,):
        warnings.extend(check(generated, facts))
    return warnings


__all__ = [
    "PloValidationResult",
    "run_plo_audit_validators",
    "run_plo_soft_validators",
    "soft_validate_position_words",
    "validate_banned_phrases",
    "validate_card_fabrication",
    "validate_no_internal_xml",
    "validate_no_list_formatting",
    "validate_shape_claims",
]
