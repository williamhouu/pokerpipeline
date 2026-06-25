"""Layer 8 helper (postflop): the Runout app's table-state CSV tokens.

The preflop pipeline emits the app's poker-table render tokens (seats + chips
+ actions) from :mod:`pipeline.preflop.app_table_format`; this is the postflop
counterpart. Postflop is where the board exists, so ``Cards on Table`` is
finally non-empty -- the whole reason this port matters.

Columns produced (the team's template B-H):

    User Seat       POS-$<remaining>[-$<amount>-<action>]
    User Cards      RANK-suitword, RANK-suitword
    Cards on Table  RANK-suitword, ...        (the board: 3-5 cards)
    Table Size      bare int (6 / 9)
    Default Stack   $<n> (cash $) / <n>BB (bb display)
    Seats           comma-separated villain tokens (same grammar as User Seat)
    POT             $<n> / <n>BB  (INCLUDING any unmatched bet hero faces)

It is built natively from the postflop facts + solve -- the resolved positions,
the preflop line, the per-node multi-street action history, and the board --
NOT by re-parsing prose. The chip amounts come from the SAME bb-denominated IR
the question prose renders from (``solve.bb_in_dollars`` for the dollar path),
so the table tokens never disagree with the Question column or the pot.

How this differs from the *preflop* engine (confirmed against the team's
``docs/output_format_examples.xlsx`` postflop rows, e.g. ``BTN-$41.25`` /
``SB-$29.65-$11.60-bet``):

  * **Remaining stacks keep cents.** Preflop rounds remaining to whole dollars;
    the postflop sample rows show ``$41.25`` / ``$29.65`` -- postflop bet sizes
    create fractional stacks, so we render the exact (trimmed) value.
  * **Both players are always shown.** Preflop omits non-blind folders; at a
    postflop decision both players are by definition still in the hand, so the
    villain seat always renders (bare ``POS-$rem`` when they have not acted on
    the decision street yet).
  * **Per-player remaining is reconstructed by a betting walk** across the
    preflop line + every postflop street, since the IR only carries the
    aggregate min-stack-behind, not per-seat investment.

Heads-up for v1 (the IR is two-player), but the seat loop is written over all
non-hero positions so it does not need rewriting if the IR grows multiway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotations only -- avoid importing the heavy facts module
    from pipeline.postflop.facts import PostflopFacts
    from pipeline.postflop.solve import PostflopSolve

# Suit letter (as stored in the IR's combo / board cards, e.g. "Js") -> the
# app's suit word.
_SUIT_WORD = {"s": "spades", "h": "hearts", "d": "diamonds", "c": "clubs"}

# Preflop verbs that put chips in as a raise-TO amount (the rest -- call / check
# / fold -- don't set a new commit level).
_PREFLOP_RAISE_VERBS = frozenset({"open", "raise", "3-bet", "4-bet", "5-bet"})


def _trim_num(n: float) -> str:
    """A number as the app renders it: integers lose the decimal ("100"),
    fractions keep up to 2dp with trailing zeros trimmed ("97.5", "41.25")."""
    n = round(n, 2)
    if n == int(n):
        return str(int(n))
    return f"{n:.2f}".rstrip("0").rstrip(".")


def _card_token(card: str) -> str:
    """One card ``"Js"`` -> ``"J-spades"`` (ten stays the ``T`` token)."""
    return f"{card[0]}-{_SUIT_WORD[card[1].lower()]}"


def _cards_tokens(cards: tuple[str, ...] | list[str]) -> str:
    """A card sequence -> ``"2-clubs, J-spades, 7-spades"`` (empty if none)."""
    return ", ".join(_card_token(c) for c in cards)


def _make_fmt(*, display_in_bb: bool, bb_in_dollars: float):
    """An amount formatter: a bb amount -> ``"$41.25"`` (dollars) or ``"97.5BB"``
    (bb display). Zero renders as ``"$0"`` / ``"0BB"`` (used for a check seat)."""

    def _fmt(amount_bb: float) -> str:
        if display_in_bb:
            return f"{_trim_num(amount_bb)}BB"
        return "$" + _trim_num(amount_bb * bb_in_dollars)

    return _fmt


def _preflop_committed(solve: PostflopSolve) -> dict[str, float]:
    """Each still-in player's total chips committed PREFLOP, from the solve's
    preflop summary.

    A raise/open/3-bet sets that seat's level; a call matches the current top
    level. (Heads-up SRP: both end at the open size.) Limped pots -- where a
    seat checks its blind option -- are not in the v1 solves; they would need
    the posted blind seeded here.
    """
    committed: dict[str, float] = {
        solve.oop_position: 0.0,
        solve.ip_position: 0.0,
    }
    for step in solve.preflop_summary:
        pos = step.position
        if pos not in committed:
            continue
        if step.verb in _PREFLOP_RAISE_VERBS and step.to_bb is not None:
            committed[pos] = step.to_bb
        elif step.verb == "call":
            committed[pos] = max(committed.values())
        # check / fold: no new chips for stack accounting.
    return committed


def _seat_states(facts: PostflopFacts, solve: PostflopSolve) -> dict[str, dict]:
    """Reconstruct each still-in player's seat state at the decision.

    Returns ``{position: {"remaining": bb, "street_amount": bb, "verb": str|None,
    "is_check": bool}}`` where ``remaining`` is chips behind (stack minus every
    chip committed across all streets), ``street_amount`` is the chips that
    seat has in front on the DECISION street (0 if they've not acted / checked),
    and ``verb`` is their last action on the decision street (``None`` if they
    have not acted on it yet).

    The pot the hero plays for comes straight from the IR (``facts.pot_bb``);
    this walk exists only to split per-player remaining + this-street wagers,
    which the IR's aggregate min-behind can't give.
    """
    positions = (solve.oop_position, solve.ip_position)
    total = dict(_preflop_committed(solve))  # cumulative across all streets
    decision_street = facts.street

    # Walk the postflop history street by street. A completed street ends
    # matched (both equal) or checked-through (both 0), so its end-of-street
    # commit is that street's per-player contribution -> fold into the total.
    # The DECISION street's (partial) commit is the chips currently in front.
    street_amount: dict[str, float] = {p: 0.0 for p in positions}
    last_verb: dict[str, str | None] = {p: None for p in positions}
    cur_street: str | None = None
    for step in facts.spot.node.history:
        if step.street != cur_street:
            if cur_street is not None:
                for p in positions:
                    total[p] += street_amount[p]
            street_amount = {p: 0.0 for p in positions}
            last_verb = {p: None for p in positions}
            cur_street = step.street
        pos = step.position
        if pos not in street_amount:
            continue
        if step.verb in ("bet", "raise") and step.to_bb is not None:
            street_amount[pos] = step.to_bb  # bet OF / raise TO == this-street total
        elif step.verb == "call":
            street_amount[pos] = max(street_amount.values())
        # check: stays 0. fold: not possible for a still-in player pre-decision.
        last_verb[pos] = step.verb

    # The final street walked is the decision street ONLY if it carried
    # pre-decision action. When hero is first to act on the decision street
    # (e.g. OOP on a fresh turn), there is no decision-street group -> both
    # in-front amounts are 0 and the last completed street was already folded
    # into the total above. Either way, fold the final group into the total too.
    if cur_street is not None:
        for p in positions:
            total[p] += street_amount[p]
    if cur_street != decision_street:
        # Hero opens the decision street: nothing is in front yet.
        street_amount = {p: 0.0 for p in positions}
        last_verb = {p: None for p in positions}

    return {
        p: {
            "remaining": solve.effective_stack_bb - total[p],
            "street_amount": street_amount[p],
            "verb": last_verb[p],
            "is_check": last_verb[p] == "check",
        }
        for p in positions
    }


def build_postflop_app_table_columns(
    facts: PostflopFacts,
    solve: PostflopSolve,
    *,
    display_in_bb: bool = True,
) -> dict[str, str]:
    """Build the table-state CSV columns for one postflop spot.

    Returns ``user_seat`` / ``user_cards`` / ``cards_on_table`` / ``table_size``
    / ``default_stack`` / ``seats`` / ``pot`` -- each a ready-to-write app token
    string. Amounts render in big blinds (``display_in_bb=True``) or in dollars
    (``solve.bb_in_dollars``), matching the Question prose.
    """
    fmt = _make_fmt(display_in_bb=display_in_bb, bb_in_dollars=solve.bb_in_dollars)
    states = _seat_states(facts, solve)
    hero = facts.hero_position

    # --- hero seat (User Seat) -------------------------------------------
    hs = states[hero]
    user_seat = f"{hero}-{fmt(hs['remaining'])}"
    # Hero's own last action shows only for a wager (bet/raise) they made this
    # street and are now facing a re-raise on -- a hero check leaves nothing in
    # front (mirrors the preflop engine, which never suffixes a hero check).
    if hs["verb"] in ("bet", "raise") and hs["street_amount"] > 0:
        user_seat += f"-{fmt(hs['street_amount'])}-{hs['verb']}"

    # --- villain seat(s) (Seats), ascending by chips in front -------------
    entries: list[tuple[float, str]] = []
    for pos in (solve.oop_position, solve.ip_position):
        if pos == hero:
            continue
        st = states[pos]
        seat = f"{pos}-{fmt(st['remaining'])}"
        if st["is_check"]:
            seat += f"-{fmt(0)}-check"
        elif st["verb"] is not None and st["street_amount"] > 0:
            seat += f"-{fmt(st['street_amount'])}-{st['verb']}"
        # else: still to act on this street -> bare POS-$remaining.
        entries.append((st["street_amount"], seat))
    entries.sort(key=lambda e: e[0])
    seats = ", ".join(seat for _amt, seat in entries)

    return {
        "user_seat": user_seat,
        "user_cards": _cards_tokens([facts.spot.hero_combo[:2], facts.spot.hero_combo[2:]]),
        "cards_on_table": _cards_tokens(facts.board),
        "table_size": str(solve.table_size),
        "default_stack": fmt(solve.effective_stack_bb),
        "seats": seats,
        "pot": fmt(facts.pot_bb),
    }


__all__ = ["build_postflop_app_table_columns"]
