"""Preflop-entry questions derived from a postflop ``.db`` solve.

A postflop solve has NO preflop decision tree -- it starts at the flop. But it
carries the **flop-entry ranges with real frequencies** (see
``PostflopSolve.preflop_entry_ranges``): how often each combo took the action
that started the hand -- the IP player's OPEN frequency, the OOP player's
CALL-vs-open frequency. That is enough to synthesise the ONE preflop decision
that created this hand:

* the OOP caller facing the open -- "call vs fold" at the solver frequency, a
  genuine mix (e.g. ``22`` continuing ~83%);
* the IP opener -- "open vs fold" (mostly trivial: opens ~always).

This is deliberately a **small, honest** question type. The ceiling (settled in
the design notes) is real: there are NO preflop EVs, the ranges are solve
INPUTS rather than solved decisions, and this single-raised-pot line has no
3-bet / 4-bet branch. So a preflop-entry question is NOT a substitute for a
real standalone preflop question (use the preflop range-pack pipeline for
those). It IS the right source for:

* the **preflop leg of a play-through** (the entry decision that started THIS
  hand, self-consistent with the postflop streets), and
* a **"generate preflop standalone"** option off the same solve.

Self-contained, like the rest of :mod:`pipeline.postflop`: it reuses only pure
leaf utilities (``combo_str_to_hand_class``, the shared explanation-generator
leaf, ``build_chat_context``, ``neutral_credit``) and emits the SAME
:data:`pipeline.postflop.format_writer.POSTFLOP_CSV_COLUMNS` so a preflop-entry
row drops straight into a postflop batch / the Review page, distinguished only
by ``Hand Stage == "Preflop"`` and an empty board.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pipeline.bb_display import round_to_half_bb
from pipeline.chat_context import StrategyEntry, build_chat_context
from pipeline.explanation_generator import (
    ExplanationValidationError,
    GeneratedExplanation,
    call_messages_create,
)
from pipeline.neutral_credit import format_neutral_credit, neutral_credit_options
from pipeline.postflop.action_history import build_context_line
from pipeline.postflop.format_writer import _PREFLOP_POT_TYPE, _stack_depth_bucket
from pipeline.postflop.animation_script import build_preflop_animation_script
from pipeline.postflop.solve import PostflopSolve
from pipeline.postflop.stat_notes import StatNote, stat_notes_to_json
from pipeline.preflop_ranges import combo_str_to_hand_class

# Standard cash blinds in bb (this line is BTN-open vs BB; the SB folds and its
# posted half-blind is dead money). A future 3-bet-pot / blind-vs-blind solve
# would parameterise these from its own preflop line.
_SB_BB = 0.5
_BB_BB = 1.0

# The two entry actions this single-raised-pot line exposes, with the strategic
# frame the LLM writes toward (chosen deterministically, never by the model).
ENTRY_ARCHETYPE_GUIDANCE: dict[str, str] = {
    "defend_vs_open": "Defend the big blind by calling: you are closing the "
                      "action with a price, getting to see a flop in position-"
                      "aware terms. Frame around the pot odds and realising "
                      "equity with a hand too good to fold but not strong "
                      "enough to raise.",
    "open_for_value": "Open (raise first in) to take the initiative with a hand "
                      "worth playing. Frame around hand strength and position, "
                      "not a price.",
}

# Suit letter -> the app's suit word (matches preflop/postflop app-table tokens).
_SUIT_WORD = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}
# Suit letter -> emoji (for the question prose; mirrors action_history cards).
_SUIT_EMOJI = {"s": "♠️", "h": "❤️", "d": "♦️", "c": "♣️"}

_SEAT_INTRO = {
    "BTN": "on the Button", "SB": "in the Small Blind", "BB": "in the Big Blind",
    "CO": "in the Cutoff", "HJ": "in the Hijack", "LJ": "in the Lojack",
    "UTG": "under the gun", "UTG+1": "in UTG+1", "UTG+2": "in UTG+2",
}
_SEAT_SUBJECT = {
    "BTN": "the Button", "SB": "the Small Blind", "BB": "the Big Blind",
    "CO": "the Cutoff", "HJ": "the Hijack", "LJ": "the Lojack",
    "UTG": "UTG", "UTG+1": "UTG+1", "UTG+2": "UTG+2",
}


# --- facts ------------------------------------------------------------------
@dataclass(frozen=True)
class PreflopEntryFacts:
    """The resolved facts for one preflop-entry decision. Layer 6's only input.

    Everything is solver-derived (the entry-range weight) or deterministic
    geometry. The LLM turns this into prose; it never recomputes any of it.
    """

    solve: PostflopSolve
    hero_position: str
    villain_position: str
    hero_in_position: bool
    hero_combo: str
    hand_class: str  # 169-class label, e.g. "22", "JTs", "AQo"
    entry_verb: str  # "call" (defender) | "open" (opener)
    continue_freq: float  # solver frequency of the entry action, [0,1]
    open_to_bb: float  # the open size faced (defender) / made (opener)
    pot_bb: float  # pot the hero is deciding into (before the hero acts)
    to_call_bb: float  # chips hero adds to continue (0 for the opener)
    break_even_equity: float | None  # pot-odds price (defender only)
    archetype: str
    concept_tags: list[str] = field(default_factory=list)

    @property
    def fold_freq(self) -> float:
        return max(0.0, 1.0 - self.continue_freq)

    @property
    def difficulty(self) -> int:
        """A frequency-only difficulty on the brief's 500-3000 Elo scale.

        A pure entry (always call / always open) is easy; a close mix is hard.
        Mirrors the brief's MVP formula on the dominant frequency (here the
        more-frequent of continue / fold). Floored at 500 (the documented
        "easiest"), matching pipeline.postflop.difficulty._HARD_FLOOR."""
        dom = max(self.continue_freq, self.fold_freq)
        raw = 3000.0 - ((dom - 0.55) / 0.40) * 2500.0
        return int(round(min(3000.0, max(500.0, raw))))


# --- ranges / spot enumeration ---------------------------------------------
def _entry_verb(solve: PostflopSolve, position: str) -> str:
    """'open' for the IP opener, 'call' for the OOP defender (this SRP line)."""
    return "open" if position == solve.ip_position else "call"


def _open_to_bb(solve: PostflopSolve) -> float:
    """The preflop open size in bb, from the solve's preflop summary."""
    for step in solve.preflop_summary:
        if step.verb in ("open", "raise") and step.to_bb is not None:
            return float(step.to_bb)
    return 2.5  # sane default if the summary lacks the size


def _representative_combo(hand_class: str, entry_range: dict[str, float]) -> str | None:
    """The lexicographically-first concrete combo of ``hand_class`` present in
    ``entry_range`` (deterministic). ``None`` if the class isn't represented."""
    combos = sorted(
        c for c in entry_range if combo_str_to_hand_class(c) == hand_class
    )
    return combos[0] if combos else None


def build_preflop_entry_facts(
    solve: PostflopSolve, hero_position: str, hero_combo: str
) -> PreflopEntryFacts:
    """Resolve the preflop-entry facts for one (hero seat, concrete combo).

    ``hero_combo`` is a 4-char combo (e.g. ``"2c2d"``); its entry frequency is
    read from ``solve.preflop_entry_ranges[hero_position]`` (the combo's own
    weight). Raises ``KeyError`` if the seat has no entry-range data.
    """
    entry_range = dict(solve.preflop_entry_ranges[hero_position])
    verb = _entry_verb(solve, hero_position)
    open_to = _open_to_bb(solve)
    continue_freq = max(0.0, min(1.0, float(entry_range.get(hero_combo, 0.0))))
    hero_in_position = hero_position == solve.ip_position

    if verb == "call":  # defender facing the open: a real price
        to_call = open_to - _BB_BB
        pot_before = open_to + _BB_BB + _SB_BB  # open + posted BB + dead SB
        break_even = to_call / (pot_before + to_call) if (pot_before + to_call) else None
        pot_bb = pot_before
        archetype = "defend_vs_open"
    else:  # opener (raise first in): no price, just open-or-fold
        to_call = 0.0
        pot_bb = _SB_BB + _BB_BB  # the blinds in front
        break_even = None
        archetype = "open_for_value"

    tags = _entry_concept_tags(verb, hero_in_position, continue_freq, hero_combo)
    return PreflopEntryFacts(
        solve=solve,
        hero_position=hero_position,
        villain_position=solve.ip_position if hero_position == solve.oop_position else solve.oop_position,
        hero_in_position=hero_in_position,
        hero_combo=hero_combo,
        hand_class=combo_str_to_hand_class(hero_combo),
        entry_verb=verb,
        continue_freq=continue_freq,
        open_to_bb=open_to,
        pot_bb=pot_bb,
        to_call_bb=to_call,
        break_even_equity=break_even,
        archetype=archetype,
        concept_tags=tags,
    )


def _entry_concept_tags(
    verb: str, hero_in_position: bool, continue_freq: float, combo: str
) -> list[str]:
    """A small, deterministic concept-tag set for a preflop-entry spot."""
    tags = ["preflop_decision"]
    if verb == "call":
        tags += ["facing_an_open", "blind_defense", "pot_odds"]
    else:
        tags += ["open_raise_first_in"]
    tags.append("in_position" if hero_in_position else "out_of_position")
    if 0.05 < continue_freq < 0.95:  # noqa: PLR2004
        tags.append("mixed_frequency")
    r1, r2 = combo[0], combo[2]
    if r1 == r2:
        tags.append("pocket_pair")
    elif combo[1] == combo[3]:
        tags.append("suited")
    else:
        tags.append("offsuit")
    return tags


def enumerate_preflop_entry_facts(
    solve: PostflopSolve,
    *,
    heroes: tuple[str, ...] = (),
) -> list[PreflopEntryFacts]:
    """One preflop-entry facts object per 169-class with entry data, both seats.

    Iterates each requested hero seat (default: both) and each 169 hand-class
    that has a representative combo in that seat's entry range, in a
    deterministic order. Each class is represented by its first concrete combo.
    """
    seats = tuple(heroes) or solve.positions
    out: list[PreflopEntryFacts] = []
    for seat in seats:
        entry_range = dict(solve.preflop_entry_ranges.get(seat, {}))
        if not entry_range:
            continue
        seen: set[str] = set()
        for combo in sorted(entry_range):
            hc = combo_str_to_hand_class(combo)
            if hc in seen:
                continue
            seen.add(hc)
            rep = _representative_combo(hc, entry_range)
            if rep is None:
                continue
            out.append(build_preflop_entry_facts(solve, seat, rep))
    return out


def preflop_entry_is_worthy(
    facts: PreflopEntryFacts, *, min_frequency: float, max_frequency: float
) -> bool:
    """Worthy when the entry action is taken at a genuinely-mixed frequency.

    Same frequency-window idea as the postflop / preflop worthiness gate: a pure
    (always / never) entry is a trivial question, a mix is interesting. The
    play-through path bypasses this (it always shows the entry that started the
    hand, mixed or not)."""
    return min_frequency <= facts.continue_freq <= max_frequency


# Hands that, facing a SINGLE open, are call-or-3bet and essentially NEVER fold.
# A postflop solve only carries the FLAT-CALL frequency (no 3-bet branch), so for
# these the standalone call-vs-fold question misrepresents the decision: the
# non-call mass is a 3-bet, not a fold, but we have no data to show that. They are
# excluded from STANDALONE preflop-entry questions (#6B). Play-through legs KEEP
# them (``as_played``: the hand really did call to reach this flop -- a fact, not
# a strategy claim). The proper fix is sourcing preflop ranges from a dedicated
# preflop solve that carries the call/3-bet/fold split.
PREMIUM_NEVER_FOLD_CLASSES = frozenset({"AA", "KK", "QQ", "JJ", "AKs", "AKo", "AQs"})


def standalone_entry_is_reliable(facts: PreflopEntryFacts) -> bool:
    """Whether a STANDALONE preflop-entry question fairly represents the decision.

    False for a DEFENDER (caller) holding a premium that 3-bets rather than folds
    -- the call/fold binary would mislabel its non-call mass as a fold. The
    OPENER's open/fold framing is always reliable (premiums open, junk folds; no
    hidden third action), so only the caller is gated. See
    :data:`PREMIUM_NEVER_FOLD_CLASSES`."""
    if facts.entry_verb != "call":
        return True
    return facts.hand_class not in PREMIUM_NEVER_FOLD_CLASSES


# --- options ----------------------------------------------------------------
def _continue_label(facts: PreflopEntryFacts, *, display_in_bb: bool) -> str:
    """The plain label for the continue action: 'Call' / 'Open to 2.5bb'."""
    if facts.entry_verb == "call":
        return "Call"
    amt = _amount(facts.open_to_bb, solve=facts.solve, display_in_bb=display_in_bb)
    return f"Open to {amt}"


def build_preflop_entry_options(
    facts: PreflopEntryFacts, *, style: str = "auto", display_in_bb: bool = True,
    as_played: bool = False,
) -> tuple[list[str], str]:
    """The four answer options + the correct answer for a preflop-entry spot.

    A binary continue/fold decision: ``basic`` -> plain ``[continue, Fold]``;
    ``gto`` -> the Always/Mostly spectrum; ``auto`` -> basic when the dominant
    action is clearly dominant (>=80%), else the spectrum (mirrors the postflop
    auto rule).

    ``as_played`` (the play-through preflop LEG): the hand reached this flop by
    taking the entry action, so the correct answer is ALWAYS that action (Call /
    Open), never Fold -- even when the combo's flat-call frequency is low because
    it mostly takes another line (a 3-bet) outside this solve's data. Without
    this, a heavy-3-bet hand whose call-weight is < 50% would be mislabelled
    "Mostly Fold", contradicting the fact that it is sitting on the flop."""
    cont = _continue_label(facts, display_in_bb=display_in_bb)
    dom_freq = max(facts.continue_freq, facts.fold_freq)
    resolved = style
    if resolved == "auto":
        resolved = "basic" if dom_freq >= 0.80 else "gto"  # noqa: PLR2004

    # as_played forces the CORRECT answer to the continue side (the hand reached
    # the flop by taking it), but still HONOURS the requested option SHAPE so the
    # play-through preflop leg matches the postflop legs' style (a GTO batch shows
    # the Always/Mostly spectrum here too, not a stray basic Call/Fold).
    if resolved == "basic":
        options = [cont, "Fold"]
        correct = cont if (as_played or facts.continue_freq >= facts.fold_freq) else "Fold"
        return options, correct

    # gto spectrum (Fold least aggressive -> continue most aggressive).
    options = ["Always Fold", "Mostly Fold", f"Mostly {cont}", f"Always {cont}"]
    if as_played or facts.continue_freq >= facts.fold_freq:
        dom, freq = cont, facts.continue_freq  # always the continue side when as_played
    else:
        dom, freq = "Fold", facts.fold_freq
    prefix = "Always" if freq >= 0.9999 else "Mostly"  # noqa: PLR2004
    correct = f"{prefix} {dom}"
    if correct not in options:  # defensive
        correct = f"Mostly {dom}"
    return options, correct


def _option_frequencies(facts: PreflopEntryFacts, cont_label: str) -> dict[str, float]:
    """Solver frequency per option action (continue vs fold) for neutral credit."""
    return {cont_label: facts.continue_freq, "Fold": facts.fold_freq}


# --- amount / card rendering ------------------------------------------------
def _amount(amount_bb: float, *, solve: PostflopSolve, display_in_bb: bool) -> str:
    """Render a bb amount as '2.5bb' (0.5bb grid) or '$5' / '$4.32'."""
    if display_in_bb:
        return f"{round_to_half_bb(amount_bb):g}bb"
    d = amount_bb * solve.bb_in_dollars
    return f"${round(d):g}" if abs(d - round(d)) < 0.005 else f"${d:.2f}"  # noqa: PLR2004


def _cards_emoji(combo: str) -> str:
    return "".join(f"{combo[i]}{_SUIT_EMOJI[combo[i + 1].lower()]}" for i in (0, 2))


def _cards_app(combo: str) -> str:
    return ", ".join(f"{combo[i]}-{_SUIT_WORD[combo[i + 1].lower()]}" for i in (0, 2))


def _subject(position: str, hero: str, *, cap: bool) -> str:
    if position == hero:
        return "You" if cap else "you"
    phrase = _SEAT_SUBJECT.get(position, position)
    return phrase[0].upper() + phrase[1:] if cap else phrase


# --- question prose ---------------------------------------------------------
def format_preflop_entry_question(
    facts: PreflopEntryFacts, *, display_in_bb: bool = True
) -> str:
    """The deterministic question narrative for a preflop-entry decision.

    Mirrors the postflop ``format_question`` voice (hero seat + cards, then the
    action up to the decision) but stops preflop -- no board has come."""
    hero = facts.hero_position
    lines = [f"You're {_SEAT_INTRO.get(hero, hero)} with {_cards_emoji(facts.hero_combo)}."]
    open_amt = _amount(facts.open_to_bb, solve=facts.solve, display_in_bb=display_in_bb)
    if facts.entry_verb == "call":
        opener = _subject(facts.villain_position, hero, cap=True)
        lines.append(f"{opener} opens to {open_amt} and it folds to you.")
    else:
        lines.append("It folds to you.")
    return "\n".join(lines)


# --- explanation (Layer 6) --------------------------------------------------
PREFLOP_ENTRY_SYSTEM_PROMPT = """\
You are writing the answer explanation for a single PREFLOP poker training \
question, in the voice of an expert coach. You are given a fully-resolved \
SOLVER DATA block and the CORRECT answer. Write ONLY the explanation prose.

Rules:
1. Open with the verdict (the correct action), then give the reasoning.
2. Coach the reader as "you"; refer to the opponent by position.
3. 2-4 sentences. Be concrete and confident; no hedging filler.
4. Use ONLY the facts in the SOLVER DATA block. Never invent equities, \
frequencies, ranges, or cards. If you cite a number, use the one given.
5. This is the PREFLOP entry decision (open, or call vs an open). Frame a CALL \
around the pot-odds price and realising equity; frame an OPEN around hand \
strength and position. Do NOT discuss any flop, turn, or river -- no board has \
come. Do NOT claim the non-call part of the frequency is a fold (some hands \
take a different line such as a 3-bet, which is outside this data); speak only \
to the call frequency you are given.
6. You may name your own two cards (with suit emojis); never name a card you \
were not given.
7. NO em dashes, NO semicolons, no corporate/template phrasing.
8. This is a solve INPUT range, so do not claim exact post-flop equities or \
EVs you were not given. Return only the explanation text, no preamble.
"""


# Admin-editable override for the preflop-entry system prompt (gitignored, like
# the postflop one). Generation reads the override if present, else the built-in.
# The preflop-entry leg of a play-through reads DIFFERENTLY from the postflop legs
# precisely because it uses THIS prompt, not the postflop system prompt -- so it
# gets its own editor on the Prompt page. Reset = delete the file.
_PREFLOP_ENTRY_PROMPT_OVERRIDE_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "admin_panel"
    / "prompts"
    / "preflop_entry_system.txt"
)


def load_preflop_entry_system_prompt() -> str:
    """The active preflop-entry system prompt: the admin override file if present,
    else :data:`PREFLOP_ENTRY_SYSTEM_PROMPT`. Not cached (edits take effect on the
    next run); reset to default = delete the override file."""
    if _PREFLOP_ENTRY_PROMPT_OVERRIDE_PATH.is_file():
        return _PREFLOP_ENTRY_PROMPT_OVERRIDE_PATH.read_text(encoding="utf-8")
    return PREFLOP_ENTRY_SYSTEM_PROMPT


def build_preflop_entry_data_block(
    facts: PreflopEntryFacts, *, display_in_bb: bool = True
) -> str:
    """The structured fact block the LLM reads for a preflop-entry spot."""
    f = facts
    lines = [
        "STREET: preflop",
        f"HERO: {f.hero_position} "
        f"({'in position' if f.hero_in_position else 'out of position'} postflop)",
        f"VILLAIN: {f.villain_position}",
        f"HERO HAND: {f.hero_combo} ({f.hand_class})",
        f"ACTION: {'facing an open to ' + _amount(f.open_to_bb, solve=f.solve, display_in_bb=display_in_bb) if f.entry_verb == 'call' else 'first in (raise-or-fold)'}",
        f"POT BEFORE YOU ACT: {round_to_half_bb(f.pot_bb):g}bb",
    ]
    if f.break_even_equity is not None:
        lines.append(
            f"TO CONTINUE: call {round_to_half_bb(f.to_call_bb):g}bb; "
            f"break-even equity {f.break_even_equity * 100:.0f}%"
        )
    verb = "call" if f.entry_verb == "call" else "open"
    lines += [
        f"CORRECT ACTION: {verb} (solver frequency {f.continue_freq:.0%})",
        f"STRATEGIC FRAME ({f.archetype}): {ENTRY_ARCHETYPE_GUIDANCE.get(f.archetype, '')}",
        f"CONCEPT TAGS: {', '.join(f.concept_tags)}",
    ]
    return "\n".join(lines)


def placeholder_preflop_entry_explanation(
    facts: PreflopEntryFacts, options: list[str], correct: str,
    *, display_in_bb: bool = True, as_played: bool = False,
) -> GeneratedExplanation:
    """A deterministic, no-API explanation (dry runs / tests).

    ``as_played`` (play-through leg): frame the entry as what this hand did to
    reach the flop, not as a best-play claim (the combo may mostly take another
    line whose data we don't have)."""
    f = facts
    if f.entry_verb == "call":
        price = (
            f" You are getting about {f.break_even_equity * 100:.0f}% break-even on the call."
            if f.break_even_equity is not None else ""
        )
        if as_played:
            prose = (
                f"To reach this flop, {f.hand_class} called the open from the "
                f"{f.hero_position}.{price} The solver flats it about "
                f"{f.continue_freq:.0%} of the time in this spot."
            )
        else:
            prose = (
                f"The solver defends {f.hand_class} by calling about {f.continue_freq:.0%} "
                f"of the time from the {f.hero_position}.{price} The price and your "
                f"position make this a clear flat-call to see a flop."
            )
    else:
        prose = (
            f"The solver opens {f.hand_class} about {f.continue_freq:.0%} of the time "
            f"from the {f.hero_position}, raising first in to take the initiative."
        )
    return _assemble(options, correct, prose)


def _assemble(options: list[str], correct: str, prose: str) -> GeneratedExplanation:
    padded = (options + ["", "", "", ""])[:4]
    return GeneratedExplanation(
        option_1=padded[0], option_2=padded[1], option_3=padded[2],
        option_4=padded[3], correct_answer=correct,
        answer_explanation=prose.strip(),
    )


def generate_preflop_entry_explanation(
    facts: PreflopEntryFacts, options: list[str], correct: str,
    *, client: object, model: str, temperature: float = 0.3, max_tokens: int = 500,
    system_prompt: str | None = None, display_in_bb: bool = True,
    usage_callback=None, as_played: bool = False,
) -> GeneratedExplanation:
    """Generate one preflop-entry explanation via the Anthropic API.

    The LLM writes only the prose; ``options`` / ``correct`` are fixed. A light
    validity check (the correct answer must be one of the options, which the
    assembler guarantees) -- the heavy hard-validators are postflop-specific, so
    here we only re-assert the answer-key contract and ship."""
    if client is None:
        raise ValueError("generate_preflop_entry_explanation needs an Anthropic client")
    system = system_prompt or PREFLOP_ENTRY_SYSTEM_PROMPT
    question = format_preflop_entry_question(facts, display_in_bb=display_in_bb)
    data_block = build_preflop_entry_data_block(facts, display_in_bb=display_in_bb)
    played_note = (
        "\n\nNOTE: this is one leg of a full hand the player is replaying. This "
        "hand reached the flop by taking the entry action, so frame it as what "
        "the hand did to get here (and how standard that is at the given "
        "frequency), not as a fold."
        if as_played else ""
    )
    user = (
        f"QUESTION:\n{question}\n\nOPTIONS: {options}\nCORRECT ANSWER: {correct}\n\n"
        f"SOLVER DATA:\n{data_block}{played_note}\n\nWrite the answer explanation."
    )
    response = call_messages_create(
        client, model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    if usage_callback is not None and getattr(response, "usage", None) is not None:
        usage_callback(response.usage)
    prose = _response_text(response)
    if correct not in (options):  # the assembler guarantees this; defensive
        raise ExplanationValidationError(
            "preflop-entry correct answer not in options",
            last_attempt_text=prose,
        )
    return _assemble(options, correct, prose)


def _response_text(response: object) -> str:
    content = getattr(response, "content", None)
    if content is None:
        return str(response)
    parts = []
    for block in content:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "".join(parts)


# --- app-table tokens (preflop spec, mirrored) ------------------------------
def _entry_app_table(facts: PreflopEntryFacts, *, display_in_bb: bool) -> dict[str, str]:
    """The 7 app table-state tokens for a preflop-entry decision.

    Mirrors the team's preflop table-state spec (POS-$stack[-$wager-action]) for
    exactly the two entry decisions this single-raised-pot line exposes: the BB
    defender facing the open, and the opener first-in. Built natively (postflop
    stays self-contained) from the same bb amounts the prose uses."""
    f = facts
    solve = f.solve
    stack_bb = solve.effective_stack_bb

    def money(amount_bb: float) -> str:
        return _amount(amount_bb, solve=solve, display_in_bb=display_in_bb)

    def remaining(invested_bb: float) -> str:
        rem = stack_bb - invested_bb
        if display_in_bb:
            return f"{round_to_half_bb(rem):g}bb"
        return "$" + f"{round(rem * solve.bb_in_dollars):g}"

    hero = f.hero_position
    villain = f.villain_position
    seats: list[tuple[float, str]] = []
    if f.entry_verb == "call":
        # Hero (BB) has posted the big blind, deciding to call the open.
        user_seat = f"{hero}-{remaining(_BB_BB)}-{money(_BB_BB)}"
        # The opener sits with its raise; the SB posted and folded (dead).
        seats.append((f.open_to_bb, f"{villain}-{remaining(f.open_to_bb)}-{money(f.open_to_bb)}-raise"))
        seats.append((_SB_BB, f"SB-{remaining(_SB_BB)}-{money(_SB_BB)}-FOLD"))
        pot_bb = f.open_to_bb + _BB_BB + _SB_BB
    else:
        # Hero is first to act (opener); only the blinds are in.
        user_seat = f"{hero}-{remaining(0.0)}"
        seats.append((_SB_BB, f"SB-{remaining(_SB_BB)}-{money(_SB_BB)}"))
        seats.append((_BB_BB, f"BB-{remaining(_BB_BB)}-{money(_BB_BB)}"))
        pot_bb = _SB_BB + _BB_BB
    seats.sort(key=lambda e: e[0])
    pot = (f"{round_to_half_bb(pot_bb):g}bb" if display_in_bb
           else "$" + f"{round(pot_bb * solve.bb_in_dollars):g}")
    default_stack = (f"{round_to_half_bb(stack_bb):g}bb" if display_in_bb
                     else "$" + f"{round(stack_bb * solve.bb_in_dollars):g}")
    return {
        "user_seat": user_seat,
        "user_cards": _cards_app(f.hero_combo),
        "cards_on_table": "",  # preflop: no board
        "table_size": str(solve.table_size),
        "default_stack": default_stack,
        "seats": ", ".join(s for _a, s in seats),
        "pot": pot,
    }


# --- skills -----------------------------------------------------------------
def compute_preflop_entry_skills(facts: PreflopEntryFacts) -> list[str]:
    """User-facing skill labels for a preflop-entry spot (canonical catalog
    names; deterministic, never the LLM)."""
    skills: list[str] = []
    if facts.entry_verb == "call":
        skills.append("Blind Defense")
        if facts.break_even_equity is not None:
            skills.append("Pot Odds")
    else:
        skills.append("Preflop Hand Selection")
    return skills


# --- CSV row ----------------------------------------------------------------
def build_preflop_entry_row(
    facts: PreflopEntryFacts,
    explanation: GeneratedExplanation,
    number: int,
    *,
    validation_status: str = "draft",
    display_in_bb: bool = True,
    hand_id: str = "",
    sequence_index: int | str = "",
    sequence_total: int | str = "",
    as_played: bool = False,
) -> dict[str, str]:
    """Build one POSTFLOP_CSV_COLUMNS row for a preflop-entry question.

    Emits the same schema as :func:`pipeline.postflop.format_writer.
    build_postflop_row` so a preflop-entry row merges into a postflop batch and
    renders on the Review page; ``Hand Stage`` is ``"Preflop"`` and board-only
    columns are empty. ``hand_id`` / ``sequence_index`` tag it as the first leg
    of a play-through (Option B).

    ``as_played`` (play-through leg): the answer was forced to the continue side,
    so frequency-based neutral credit on the FOLD side would contradict it -- a
    play-through leg gets NO neutral credit (it's a "what this hand did" fact)."""
    f = facts
    cont_label = _continue_label(f, display_in_bb=display_in_bb)
    table = _entry_app_table(f, display_in_bb=display_in_bb)
    opts = (explanation.options() + ["", "", "", ""])[:4]
    pot_odds = (
        f"{f.break_even_equity * 100:.0f}%" if f.break_even_equity is not None else ""
    )
    neutral_list = [] if as_played else neutral_credit_options(
        explanation.options(), explanation.correct_answer,
        _option_frequencies(f, cont_label),
    )
    freqs = {cont_label: f.continue_freq, "Fold": f.fold_freq}
    action_freq_str = ", ".join(
        f"{lbl}: {fr * 100:.0f}%" for lbl, fr in sorted(freqs.items(), key=lambda kv: -kv[1])
        if fr >= 0.005  # noqa: PLR2004
    )
    skills = compute_preflop_entry_skills(f)
    chat = _preflop_entry_chat_context(
        f, explanation, cont_label, neutral=neutral_list, skills=skills,
        display_in_bb=display_in_bb,
    )
    return {
        "No": str(number),
        "hand_id": hand_id,
        "sequence_index": str(sequence_index),
        "sequence_total": str(sequence_total),
        "hand_difficulty": "",  # stamped by the full-hand driver
        "Hand Stage": "Preflop",
        "Context": build_context_line(f.solve, display_in_bb=display_in_bb),
        "User Seat": table["user_seat"],
        "User Cards": table["user_cards"],
        "Cards on Table": table["cards_on_table"],
        "Table Size": table["table_size"],
        "Default Stack": table["default_stack"],
        "Seats": table["seats"],
        "POT": table["pot"],
        "Question": format_preflop_entry_question(f, display_in_bb=display_in_bb),
        "Question Type": "Hand scenario question",  # sentence case, no period (July 2026)
        "option 1": opts[0],
        "option 2": opts[1],
        "option 3": opts[2],
        "option 4": opts[3],
        "Correct Answer": explanation.correct_answer,
        "neutral_credit": format_neutral_credit(neutral_list),
        "Answer Explanation": explanation.answer_explanation,
        "Cash/Tourney": "Tournament" if f.solve.game_format == "tournament" else "Cash",
        "Live or Online": f.solve.live_or_online,
        "Relative Position": "In Position" if f.hero_in_position else "Out of Position",
        # Shared-schema classification columns (June 2026 unification) so the
        # entry leg's first 41 columns line up with every other row. The pot type
        # = raises in the solve's preflop line (open=1 -> single raised pot).
        "Preflop Pot Type": _PREFLOP_POT_TYPE.get(
            sum(1 for st in f.solve.preflop_summary
                if st.verb in ("open", "raise", "3bet", "4bet", "5bet")),
            "Multi-raised pot",
        ),
        "Pot Participant": "Heads-Up" if len(f.solve.positions) <= 2 else "Multi-Way",  # noqa: PLR2004
        "Stack Depth": _stack_depth_bucket(f.solve.effective_stack_bb),
        # No exploit guidance on a preflop-ENTRY leg: its archetype vocabulary is
        # preflop's, and the postflop exploit engine is keyed on postflop
        # archetypes. Column present (schema parity), value blank.
        "exploit_notes": "",
        "Position Matchup": f"{f.hero_position}_vs_{f.villain_position}",
        "Difficulty Rating": str(f.difficulty),
        "action_frequencies": action_freq_str,
        "action_ev_bb": "",  # no preflop EVs in a postflop solve (the ceiling)
        "hero_equity": "",   # no all-in equity surfaced (solve input, not a decision)
        "range_equity": "",
        "pot_odds": pot_odds,
        "spr": "",
        # "Show the math" panel: a preflop-entry leg only has the price (a
        # postflop solve carries no preflop equity), so just the pot-odds row --
        # same JSON shape the app parses on every other question.
        "stat_notes": stat_notes_to_json(
            [
                StatNote(
                    "pot_odds", "Pot odds", pot_odds,
                    f"Your pot odds here are {pot_odds}.",
                )
            ]
            if pot_odds
            else []
        ),
        "concept_tags": ", ".join(f.concept_tags),
        "skills": ", ".join(skills),
        "archetype": f.archetype,
        "board_texture": "",
        # No ranges: a postflop solve has no ACCURATE preflop range (only the
        # flat-call frequency, missing the 3-bet/fold split). Leave empty rather
        # than show a half-true range. (A dedicated preflop solve would fill this.)
        "ranges": "",
        "easy_freq": "",
        "easy_ev": "",
        "easy_concept": "",
        "easy_hand": "",
        "Notes": "Auto-generated by poker-pipeline (preflop-entry from postflop solve).",
        "solver_reference": f"{f.solve.source_reference}/preflop/{f.hero_position}/{f.hero_combo}",
        "validation_status": validation_status,
        "chat_context": chat,
        "claim_check": "",
        # The app's animation timeline: blinds + folds up to hero's preflop
        # decision (see pipeline/postflop/animation_script.py).
        "animation_script": build_preflop_animation_script(
            facts.solve, facts.hero_position,
        ),
    }


def _preflop_entry_chat_context(
    facts: PreflopEntryFacts, explanation: GeneratedExplanation, cont_label: str,
    *, neutral: list[str], skills: list[str], display_in_bb: bool,
) -> str:
    """The per-question chatbot JSON blob for a preflop-entry spot."""
    f = facts
    strategy = [
        StrategyEntry(action=cont_label, frequency_pct=f.continue_freq * 100.0, ev_bb=None),
        StrategyEntry(action="Fold", frequency_pct=f.fold_freq * 100.0, ev_bb=None),
    ]
    key_facts: dict[str, object] = {
        "pot_before_you_act_bb": round(f.pot_bb, 1),
        "entry_frequency_pct": round(f.continue_freq * 100),
    }
    if f.break_even_equity is not None:
        key_facts["break_even_equity_pct"] = round(f.break_even_equity * 100)
        key_facts["to_call_bb"] = round(f.to_call_bb, 1)
    return build_chat_context(
        pipeline="postflop",
        situation=format_preflop_entry_question(f, display_in_bb=display_in_bb),
        hero_hand=_cards_emoji(f.hero_combo),
        hand_summary=f"{f.hand_class} (preflop entry decision)",
        recommended_action=explanation.correct_answer,
        also_acceptable=neutral,
        full_strategy=strategy,
        key_facts=key_facts,
        villain={"seat": f.villain_position},
        strategic_frame=f"{f.archetype}: {ENTRY_ARCHETYPE_GUIDANCE.get(f.archetype, '')}",
        concept_tags=f.concept_tags,
        skills_tested=skills,
        difficulty=f.difficulty,
        coaching_answer=explanation.answer_explanation,
    )


__all__ = [
    "ENTRY_ARCHETYPE_GUIDANCE",
    "PREFLOP_ENTRY_SYSTEM_PROMPT",
    "PreflopEntryFacts",
    "build_preflop_entry_data_block",
    "build_preflop_entry_facts",
    "build_preflop_entry_options",
    "build_preflop_entry_row",
    "compute_preflop_entry_skills",
    "enumerate_preflop_entry_facts",
    "PREMIUM_NEVER_FOLD_CLASSES",
    "format_preflop_entry_question",
    "generate_preflop_entry_explanation",
    "load_preflop_entry_system_prompt",
    "placeholder_preflop_entry_explanation",
    "preflop_entry_is_worthy",
    "standalone_entry_is_reliable",
]
