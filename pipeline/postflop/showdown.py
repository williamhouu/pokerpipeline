"""Showdown resolution for full-hand play-throughs (July 2026).

After the user answers the FINAL question of a play-through, the app plays
a closing sequence: hero's correct action, the villain's (possibly
invented) response, card reveals, and the pot push. This module builds that
sequence into the final leg's ``animation_script`` as the ``resolution``
object (format version 2) -- see :mod:`pipeline.postflop.animation_script`
for the base grammar.

THE VINDICATION RULE (team, July 2026). The revealed villain hand is chosen
so the CORRECT final decision is visibly right:

* correct **Call**  -> villain shows a WEAKER hand (hero wins the pot);
* correct **Fold**  -> villain shows a STRONGER hand ("good fold");
* correct **value Bet/Raise** -> villain calls with a WEAKER hand;
* correct **bluff Bet/Raise** -> villain folds a STRONGER hand (shown);
* correct **Check** (closing) -> weaker if any (hero wins), else stronger
  (the check lost the minimum).

Deliberately curated, not sampled from the raw outcome distribution -- a
teaching choice the team made knowingly (a correct call never gets
"punished" by a cooler). It stays honest where it matters: the revealed
hand is drawn from the villain's REAL reach-weighted range at the final
node (a hand that genuinely played his exact line), only the flattering
slice of it. Ties never vindicate and are excluded. When no vindicating
hand exists in the range, NO resolution is attached (never invent a combo
the range doesn't hold).

INVENTED RESPONSES obey the no-unrealistic-all-in rule: a villain response
is only ever a call or a fold.

Hands whose final decision does not actually end the hand (a flop/turn
call or check with betting still live) get no resolution -- closure there
would mean inventing whole streets. Endings covered: any-street FOLD, a
CALL that closes the hand (river, or an all-in on any street -- the
remaining board is dealt inside the resolution), river BET/RAISE (villain
response invented), and a river CHECK that closes the action.

Deterministic: the villain hand and any runout are drawn from an RNG
seeded by (hand_id, node_id, hero combo), so regeneration and the
re-verifier reproduce the same reveal byte-for-byte.

Winner metadata (July 2026, additive -- version stays 2): each ``reveal``
carries ``best_five`` (the exact five cards making the hand, for the app
to highlight); the ``win`` event carries ``reason`` ("showdown" | "fold"),
``hand_label`` (showdown wins only -- on a fold win the winner's hand may
never have been shown), and the winner's ``stack_bb`` AFTER collecting the
pot (the renderer-does-zero-arithmetic invariant).
"""

from __future__ import annotations

import json
import random
import zlib
from itertools import combinations

from pipeline.action_history import format_card
from pipeline.animation import AnimationTable
from pipeline.fact_extractor.equity import rank_hand
from pipeline.fact_extractor.hand_class import classify_hand
from pipeline.postflop.animation_script import (
    _table,
    _walk_postflop,
    _walk_preflop,
)
from pipeline.postflop.solve import PostflopNode, PostflopSolve

# A five-card board has 5 cards; endings before that need a runout.
_FULL_BOARD = 5
_RANKS = "23456789TJQKA"
_SUITS = "cdhs"

# Prose for the reveal labels, from classify_hand's made-hand tokens.
# Fallback: the token with underscores spaced (kept readable for new tokens).
_MADE_HAND_PROSE = {
    "straight_flush": "a straight flush",
    "quads": "four of a kind",
    "full_house": "a full house",
    "full_house_set_plus_board": "a full house",
    "full_house_trips_plus_pocket": "a full house",
    "flush_nut": "the nut flush",
    "flush_second_nut": "a flush",
    "flush_weak": "a flush",
    "flush": "a flush",
    "straight_nut": "the nut straight",
    "straight": "a straight",
    "straight_weak": "a straight",
    "set": "a set",
    "trips": "three of a kind",
    "two_pair_top": "two pair",
    "two_pair_top_and_bottom": "two pair",
    "two_pair_mid": "two pair",
    "two_pair": "two pair",
    "overpair": "an overpair",
    "pocket_pair_below_overcards": "an underpair",
    "top_pair_top_kicker": "top pair, top kicker",
    "top_pair_good_kicker": "top pair",
    "top_pair_weak_kicker": "top pair, weak kicker",
    "second_pair": "second pair",
    "third_pair": "third pair",
    "bottom_pair": "bottom pair",
    "middle_pair": "middle pair",
    "weak_pair": "a weak pair",
    "underpair": "an underpair",
    "one_pair": "a pair",
    "ace_high": "ace high",
    "no_pair_air": "high card",
    "air": "high card",
}


def _possessive(label: str) -> str:
    """'a straight' -> 'straight' for 'Your straight wins the pot'."""
    for art in ("a ", "an ", "the "):
        if label.startswith(art):
            return label[len(art):]
    return label


_SEAT_SUBJECT = {
    "BTN": "the Button", "SB": "the Small Blind", "BB": "the Big Blind",
    "CO": "the Cutoff", "HJ": "the Hijack", "LJ": "the Lojack",
    "UTG": "UTG", "UTG+1": "UTG+1", "UTG+2": "UTG+2",
}


def _hand_prose(cards: list[str], board: list[str]) -> str:
    token = classify_hand(cards, board)["made_hand"]
    return _MADE_HAND_PROSE.get(token, token.replace("_", " "))


def _best_five(cards: list[str], board: list[str]) -> list[str]:
    """The exact five cards that make the hand (for the app to highlight).

    Max over all 5-card subsets of hole+board by the shared evaluator;
    deterministic (fixed input order, first max wins ties, e.g. a board
    quads where the kicker choice is equivalent)."""
    seven = cards + board
    if len(seven) <= 5:  # noqa: PLR2004 -- a pre-river fold ending
        return list(seven)
    return list(max(combinations(seven, 5), key=lambda c: rank_hand(list(c))))


def _cards_emoji(cards: list[str]) -> str:
    return "".join(format_card(c) for c in cards)


def _seed(hand_id: str, node_id: str, combo: str) -> int:
    return zlib.crc32(f"{hand_id}|{node_id}|{combo}".encode())


def _normalized_answer(correct_answer: str) -> str:
    ans = correct_answer.strip()
    for prefix in ("Always ", "Mostly "):
        if ans.startswith(prefix):
            ans = ans[len(prefix):]
    return ans


def _correct_verb(node: PostflopNode, correct_answer: str) -> tuple[str, float | None]:
    """(verb, to_bb) for the correct answer's action at this node."""
    ans = _normalized_answer(correct_answer)
    action = node.action_by_label(ans)
    if action is not None:
        return action.verb, action.to_bb
    lowered = ans.lower()
    for verb in ("check", "call", "fold"):
        if lowered.startswith(verb):
            return verb, None
    if lowered.startswith(("bet", "all-in")):
        return ("raise" if node.is_facing_bet else "bet"), None
    if lowered.startswith("raise"):
        return "raise", None
    return "", None


def _slices(
    hero_combo: str, villain_range, board: list[str],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """(hero_beats, beats_hero) villain combos + reach weights on ``board``.

    Exact rank comparison (no runouts); ties excluded from BOTH slices (a
    chop vindicates nothing). Combos colliding with hero/board are skipped.
    """
    hero_cards = [hero_combo[:2], hero_combo[2:]]
    blocked = set(hero_cards) | set(board)
    hero_rank = rank_hand(hero_cards + board)
    weaker: list[tuple[str, float]] = []
    stronger: list[tuple[str, float]] = []
    for combo, weight in villain_range.items():
        if weight <= 0:
            continue
        cards = [combo[:2], combo[2:]]
        if set(cards) & blocked:
            continue
        v_rank = rank_hand(cards + board)
        if hero_rank > v_rank:
            weaker.append((combo, float(weight)))
        elif v_rank > hero_rank:
            stronger.append((combo, float(weight)))
    return weaker, stronger


def _sample(slice_: list[tuple[str, float]], rng: random.Random) -> str | None:
    if not slice_:
        return None
    combos = [c for c, _ in slice_]
    weights = [w for _, w in slice_]
    return rng.choices(combos, weights=weights, k=1)[0]


def _runout(board: list[str], blocked: set[str], rng: random.Random) -> list[str]:
    """The remaining board cards to five, dealt seeded from the live deck."""
    deck = sorted(
        r + s for r in _RANKS for s in _SUITS
        if r + s not in blocked and r + s not in board
    )
    return rng.sample(deck, _FULL_BOARD - len(board))


def build_showdown_resolution(
    node: PostflopNode,
    solve: PostflopSolve,
    *,
    hero_combo: str,
    correct_answer: str,
    hand_id: str,
) -> dict | None:
    """The ``resolution`` object for a final leg, or None (no honest ending).

    Replays the hand's chip walk to the decision (so every resolution amount
    continues the exact pot/stack state the main timeline ends on), then
    emits: hero's correct action, the villain's response where one must be
    invented (call-or-fold only), any remaining board cards, the reveals,
    and the pot push.
    """
    verb, to_bb = _correct_verb(node, correct_answer)
    if verb in ("bet", "raise") and to_bb is None:
        return None  # can't animate a wager whose size the node doesn't state
    facing_all_in = bool(node.history) and node.history[-1].all_in
    street = node.street
    ends_hand = (
        verb == "fold"
        or (verb == "call" and (street == "river" or facing_all_in))
        or (verb in ("bet", "raise") and street == "river")
        or (verb == "check" and street == "river" and node.actor == solve.ip_position)
    )
    if not ends_hand or not verb:
        return None

    hero = node.actor
    villain = node.villain
    rng = random.Random(_seed(hand_id, node.node_id, hero_combo))
    board = list(node.board)
    hero_cards = [hero_combo[:2], hero_combo[2:]]

    # Rebuild the table state to the decision so resolution amounts continue
    # the main timeline's exact pot/stacks (same walk, same numbers).
    table: AnimationTable = _table(solve)
    _walk_preflop(table, solve, stop_before_step=None)
    _walk_postflop(table, node)
    table.events = []  # the resolution numbers its own events from 1

    # Runout FIRST for an all-in call ending: vindication is judged on the
    # FINAL board (the call must visibly win), and the villain sample must
    # not collide with the dealt cards.
    runout: list[str] = []
    if verb == "call" and facing_all_in and len(board) < _FULL_BOARD:
        runout = _runout(board, set(hero_cards), rng)
    final_board = board + runout

    weaker, stronger = _slices(hero_combo, node.villain_range, final_board)

    villain_folds = False
    if verb == "call":
        pick, vindicates = _sample(weaker, rng), "Call"
    elif verb == "fold":
        pick, vindicates = _sample(stronger, rng), "Fold"
    elif verb == "check":
        pick = _sample(weaker, rng) or _sample(stronger, rng)
        vindicates = "Check"
    else:  # bet / raise: value gets paid by worse, a bluff folds out better
        total_w = sum(w for _, w in weaker)
        total_s = sum(w for _, w in stronger)
        if total_w >= total_s and weaker:
            pick, vindicates = _sample(weaker, rng), correct_answer
        else:
            pick, vindicates = _sample(stronger, rng), correct_answer
            villain_folds = True
    if pick is None:
        return None  # the range holds no vindicating hand: attach nothing
    villain_cards = [pick[:2], pick[2:]]

    # --- the resolution events, continuing the main walk's chip state ------
    if verb == "fold":
        table.emit("fold", seat=hero)
    elif verb == "call":
        table.call(hero)
    elif verb in ("bet", "raise") and to_bb is not None:
        table.wager_to(hero, verb, float(to_bb))
    elif verb == "check":
        table.emit("check", seat=hero)

    if verb in ("bet", "raise"):
        if villain_folds:
            table.emit("fold", seat=villain)
        else:
            table.call(villain)

    for i, card in enumerate(runout):
        table.emit(
            "deal", street=("turn", "river")[len(board) + i - 3], cards=[card],
        )

    hero_shows = verb != "fold" and not villain_folds
    villain_label = _hand_prose(villain_cards, final_board)
    table.emit(
        "reveal", seat=villain, cards=list(villain_cards),
        hand_label=villain_label,
        best_five=_best_five(villain_cards, final_board),
        **({"folded": True} if villain_folds else {}),
    )
    hero_label = _hand_prose(hero_cards, final_board)
    if hero_shows:
        table.emit(
            "reveal", seat=hero, cards=hero_cards, hand_label=hero_label,
            best_five=_best_five(hero_cards, final_board),
        )

    if verb == "fold":
        winner = villain
    elif villain_folds:
        winner = hero
    else:
        winner = hero if rank_hand(hero_cards + final_board) > rank_hand(
            villain_cards + final_board
        ) else villain
    # The win event carries everything the app's pot-push + result banner
    # needs, so the renderer does zero inference: WHY the seat won
    # ("showdown" | "fold"), the winning hand's label at a showdown (never
    # on a fold win -- the winner's hand may not have been shown), and the
    # winner's stack AFTER collecting the pot (the money invariant: every
    # chip event states the resulting state).
    won_at_showdown = verb != "fold" and not villain_folds
    win_fields: dict = {"reason": "showdown" if won_at_showdown else "fold"}
    if won_at_showdown:
        win_fields["hand_label"] = hero_label if winner == hero else villain_label
    win_event = table.emit("win", seat=winner, **win_fields)
    table.stacks[winner] += table.pot
    table.money(win_event, pot_bb=table.pot, stack_bb=table.stacks[winner])

    # --- the one-line summary for the app's result screen -------------------
    v_subj = _SEAT_SUBJECT.get(villain, villain)
    v_str = f"{v_subj[0].upper()}{v_subj[1:]} shows {_cards_emoji(villain_cards)} ({villain_label})"
    if verb == "fold":
        summary = f"{v_str}. Folding saved you money."
    elif villain_folds:
        summary = (
            f"{v_subj[0].upper()}{v_subj[1:]} folds {_cards_emoji(villain_cards)} "
            f"({villain_label}). Your bet takes the pot."
        )
    elif winner == hero:
        summary = f"{v_str}. Your {_possessive(hero_label)} wins the pot."
    else:
        summary = f"{v_str}. Checking behind lost the minimum."

    return {
        "vindicates": vindicates,
        "villain_seat": villain,
        "villain_cards": list(villain_cards),
        "summary": summary,
        "events": table.events,
    }


def attach_showdown_resolution(
    row: dict,
    *,
    node: PostflopNode,
    solve: PostflopSolve,
    hero_combo: str,
    correct_answer: str,
    hand_id: str,
) -> dict | None:
    """Attach the resolution to ``row['animation_script']`` (version 2).

    Mutates the row in place; returns the resolution (for the meta record)
    or None when no honest ending exists (row left untouched at version 1).
    """
    resolution = build_showdown_resolution(
        node, solve, hero_combo=hero_combo,
        correct_answer=correct_answer, hand_id=hand_id,
    )
    if resolution is None:
        return None
    payload = json.loads(row["animation_script"])
    payload["version"] = 2
    payload["resolution"] = resolution
    row["animation_script"] = json.dumps(payload, separators=(",", ":"))
    return resolution


__all__ = ["attach_showdown_resolution", "build_showdown_resolution"]
