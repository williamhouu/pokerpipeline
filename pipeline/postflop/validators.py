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

# The four valid suit glyphs (with/without the heavy-heart variant). A rank
# followed by ANY OTHER symbol/emoji glyph is a garble (e.g. "9" + Taurus
# instead of "9" + spade) -- the bug validate_card_suit_consistency misses,
# because its regex only matches a *valid* suit. We scan the symbol ranges that
# cover the likely garbles (Misc Symbols / Dingbats and the emoji planes); all
# four real suits live in that first range and are whitelisted, so they pass.
_VALID_SUIT_GLYPHS = frozenset("♠♥♦♣❤")
_RANK_CHARS = frozenset("AKQJT98765432")


def _is_symbol_glyph(ch: str) -> bool:
    o = ord(ch)
    return 0x2600 <= o <= 0x27BF or 0x1F000 <= o <= 0x1FAFF


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


def validate_no_garbled_card_glyphs(
    generated: GeneratedExplanation,
    facts: PostflopFacts,  # noqa: ARG001
) -> PostflopValidationResult:
    """Reject a rank glyph followed by a NON-suit symbol/emoji ("9" + Taurus).

    The LLM occasionally garbles a card's suit emoji into an unrelated symbol;
    :func:`validate_card_suit_consistency` can't catch it (its card regex only
    matches the four valid suits, so a garbled glyph is invisible to it). This
    is the dedicated backstop -- more cards are named on the turn/river, so the
    garble surface grows. A rank directly followed by ♠/♥/♦/♣ (a real card) is
    fine; a rank directly followed by any other symbol glyph is the garble."""
    text = generated.answer_explanation or ""
    bad: list[str] = []
    for i in range(len(text) - 1):
        nxt = text[i + 1]
        if (
            text[i] in _RANK_CHARS
            and nxt not in _VALID_SUIT_GLYPHS
            and _is_symbol_glyph(nxt)
        ):
            bad.append(text[i] + nxt)
    if bad:
        return PostflopValidationResult.fail(
            "the explanation has garbled card glyph(s) (a rank followed by a "
            "non-suit symbol): " + ", ".join(sorted(set(bad)))
            + ". A card's suit must be one of ♠️ ♥️ ♦️ ♣️ right after the rank "
            "(e.g. 9♠️), never another emoji or symbol."
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
        validate_no_garbled_card_glyphs,
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
# A percentage immediately followed by one of these is a BET SIZE ("a 33% bet",
# "67% pot", "a small 33% size"), not an equity claim. Postflop prose constantly
# puts bet sizes next to the word "equity", so without this the equity check
# flags the size itself (observed on the factor-list prompt).
_SIZE_WORD_AFTER = re.compile(
    r"^[\s-]*(?:size|sizing|sized|pot|bet|stab|c-?bet|donk|barrel|overbet)\b", re.I
)
# Sizing context just BEFORE a percentage also marks it a bet size, e.g.
# "sizing up to 50% or 67%" (the figure follows "siz..." or another "NN% or").
_SIZE_CONTEXT_BEFORE = re.compile(r"siz|over-?bet|\d\s*%\s*(?:or|and|/|,)", re.I)
# A percentage that is a DERIVED GAP, not an equity claim: "the missing 2%",
# "2% short", "2% shy of the price". These are correct (equity-minus-break-even)
# and must not be read as an invented equity figure.
_GAP_WORDS = ("missing", "short", "shy")
# A percentage that is an ACTION FREQUENCY, not equity: "folds 66% of the time",
# "mix at 66% check", "66% of the time", "66% check". Postflop prose cites the
# solver's frequencies constantly, often a sentence away from the word "equity".
_FREQ_AFTER = re.compile(r"^[\s,.-]*(?:of the time|check|bet|call|fold|rais|mix)", re.I)
_FREQ_BEFORE = re.compile(
    r"(?:of the time|mix\w*|fold\w*|call\w*|check\w*|rais\w*|bet\w*)\s*$", re.I
)
_SENTENCE_BOUNDS = (".", "!", "?", "\n")


def _sentence_bounded(text: str, start: int, end: int) -> str:
    """The text around ``text[start:end]``, clipped so it does not cross a
    sentence boundary on either side. Keeps a bet size in one sentence from
    borrowing an "equity" cue out of the neighbouring sentence."""
    pre = text[max(0, start - 60):start]
    for sep in _SENTENCE_BOUNDS:
        idx = pre.rfind(sep)
        if idx != -1:
            pre = pre[idx + 1:]
    post = text[end:end + 40]
    cut = len(post)
    for sep in _SENTENCE_BOUNDS:
        idx = post.find(sep)
        if idx != -1:
            cut = min(cut, idx)
    return pre + text[start:end] + post[:cut]


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
        if _SIZE_WORD_AFTER.match(text[m.end():m.end() + 16]):
            continue  # a bet size ("33% bet" / "67% pot"), not an equity claim
        if _SIZE_CONTEXT_BEFORE.search(text[max(0, m.start() - 30):m.start()]):
            continue  # a bet size ("sizing up to 50% or 67%"), not equity
        gap_ctx = text[max(0, m.start() - 14):m.end() + 10].lower()
        if any(w in gap_ctx for w in _GAP_WORDS):
            continue  # a derived gap ("the missing 2%", "2% short"), not equity
        if _FREQ_AFTER.match(text[m.end():m.end() + 14]) or _FREQ_BEFORE.search(
            text[max(0, m.start() - 14):m.start()]
        ):
            continue  # an action frequency ("66% of the time", "mix at 66%"), not equity
        window = _sentence_bounded(text, m.start(), m.end()).lower()
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


_BLOCK_RE = re.compile(r"block", re.I)


def soft_validate_blocker_direction(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> list[str]:
    """Flag a REVERSED blocker-direction claim vs the BLOCKERS fact.

    The data resolves whether hero mainly blocks villain's VALUE or their
    BLUFFS. If the prose claims the opposite, the bluff-catch logic is backwards
    (the brief's reversed-blocker failure mode). High precision: only flags when
    a "block ... value/bluff" claim clearly contradicts a non-neutral verdict.
    """
    text = generated.answer_explanation or ""
    if not text or facts.blocker_effect == "neutral":
        return []
    low = text.lower()
    claims_value = claims_bluffs = False
    for m in _BLOCK_RE.finditer(low):
        window = low[m.start():m.end() + 40]
        has_value, has_bluff = "value" in window, "bluff" in window
        if has_value and not has_bluff:
            claims_value = True
        elif has_bluff and not has_value:
            claims_bluffs = True
    if facts.blocker_effect == "value" and claims_bluffs and not claims_value:
        return [
            "prose says you block villain's BLUFFS, but the data says you mainly "
            "block their VALUE. Review the blocker direction (this reverses the "
            "bluff-catch logic)."
        ]
    if facts.blocker_effect == "bluffs" and claims_value and not claims_bluffs:
        return [
            "prose says you block villain's VALUE, but the data says you mainly "
            "block their BLUFFS. Review the blocker direction (this reverses the "
            "bluff-catch logic)."
        ]
    return []


# A bare percent-command like "bet 53%" (the solver's raw action label) with
# no "of the pot" anchor. "bet 53% of the pot" reads fine and is not flagged.
_RAW_PERCENT_SIZE_RE = re.compile(
    r"\b(?:bet|bets|betting|raise|raises|raising)\s+\d{1,3}\s*%(?!\s*of\b)",
    re.IGNORECASE,
)


def soft_validate_raw_percent_size(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> list[str]:
    """Flag prose that echoes a raw solver size label ("You should bet 53%").

    TEAM RULE (July 2026): sizes read in natural poker language ("bet about
    half the pot (4bb)"), never as a bare percent command -- especially now
    that the GTO answer options are size-free. Flag-only (a reviewer decides);
    "53% of the pot" is acceptable wording and not flagged.
    """
    text = generated.answer_explanation or ""
    m = _RAW_PERCENT_SIZE_RE.search(text)
    if m:
        return [
            f"prose echoes a raw solver size label ({m.group(0)!r}). Sizes "
            "should read in natural language with the real amount, e.g. "
            "\"bet about half the pot (4bb)\"."
        ]
    return []


def run_postflop_soft_validators(
    generated: GeneratedExplanation,
    facts: PostflopFacts,
) -> list[str]:
    """Run every soft validator; return all warnings (empty == clean)."""
    warnings: list[str] = []
    for check in (
        soft_validate_verdict_vs_answer,
        soft_validate_equity_vs_data,
        soft_validate_blocker_direction,
        soft_validate_raw_percent_size,
    ):
        warnings.extend(check(generated, facts))
    return warnings


__all__ = [
    "PostflopValidationResult",
    "run_postflop_audit_validators",
    "run_postflop_soft_validators",
    "soft_validate_blocker_direction",
    "soft_validate_equity_vs_data",
    "soft_validate_raw_percent_size",
    "soft_validate_verdict_vs_answer",
    "validate_banned_phrases",
    "validate_card_suit_consistency",
    "validate_correct_answer",
    "validate_no_garbled_card_glyphs",
]
