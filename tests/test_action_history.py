"""Tests for pipeline.action_history.

Run directly (`python tests/test_action_history.py`) or under pytest. The core
coverage is all 10 worked examples from docs/engineering_brief.docx, "Action
History & Context Format Specification" -- each is checked as a full-string
match for both the context block and the action history block.

Suit emoji are built from explicit codepoints (suit + U+FE0F; hearts U+2764)
so the test pins the exact notation the brief specifies.

NOTE on pot figures: the brief's printed pots agree with this module on 7 of
the 10 examples. On examples 6, 7, and 8 the brief's hand-written pot is off
(153 vs 152, 475 vs 485, 133 vs 125) -- arithmetic slips in the doc, with no
consistent rule. This module sums blinds + antes + every committed chip, which
the other 7 examples confirm; the expected values below use that correct math.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.action_history import (                       # noqa: E402
    format_action_history, format_card, format_context, format_hand,
)

S = "♠️"   # spades
H = "❤️"   # heart
D = "♦️"   # diamonds
C = "♣️"   # clubs


# Each example: the input hand dict, expected context, expected action history.
EXAMPLES = [
    (
        "1: BB defense vs Button open, flop spot",
        {
            "stakes": {"sb": 1, "bb": 2}, "format": "cash", "venue": "live",
            "table_size": 6, "effective_stack": 200,
            "hero_position": "BB", "hero_cards": ["8h", "7h"],
            "preflop_actions": [("BTN", "open", 6), ("SB", "fold"), ("BB", "call")],
            "board": {"flop": ["9h", "6c", "2h"], "turn": None, "river": None},
            "flop_actions": [("BB", "check"), ("BTN", "bet", 7)],
            "turn_actions": [], "river_actions": [],
        },
        "$1/$2 Live cash. 6-handed. $200 effective stacks.",
        f"You're in the Big Blind with 8{H}7{H}.\n"
        f"The Button opens to $6. You call.\n\n"
        f"Flop ($13): 9{H}6{C}2{H}\n"
        f"You check. The Button bets $7.",
    ),
    (
        "2: BTN open vs BB call, turn check",
        {
            "stakes": {"sb": 1, "bb": 2}, "format": "cash", "venue": "online",
            "table_size": 6, "effective_stack": 200,
            "hero_position": "BTN", "hero_cards": ["As", "Ac"],
            "preflop_actions": [("BTN", "open", 5), ("BB", "call")],
            "board": {"flop": ["Js", "Jh", "7c"], "turn": "5h", "river": None},
            "flop_actions": [("BB", "check"), ("BTN", "bet", 4), ("BB", "call")],
            "turn_actions": [("BB", "check")], "river_actions": [],
        },
        "$1/$2 Online cash. 6-handed. $200 effective stacks.",
        f"You're on the Button with A{S}A{C}.\n"
        f"You open to $5. The Big Blind calls.\n\n"
        f"Flop ($11): J{S}J{H}7{C}\n"
        f"The Big Blind checks. You bet $4. The Big Blind calls.\n\n"
        f"Turn ($19): 5{H}\n"
        f"The Big Blind checks.",
    ),
    (
        "3: BTN open facing 3-bet (preflop only)",
        {
            "stakes": {"sb": 1, "bb": 2}, "format": "cash", "venue": "live",
            "table_size": 6, "effective_stack": 200,
            "hero_position": "BTN", "hero_cards": ["Ah", "Kh"],
            "preflop_actions": [("BTN", "open", 6), ("BB", "3-bet", 20)],
            "board": {"flop": None, "turn": None, "river": None},
        },
        "$1/$2 Live cash. 6-handed. $200 effective stacks.",
        f"You're on the Button with A{H}K{H}.\n"
        f"You open to $6. The Big Blind 3-bets to $20.",
    ),
    (
        "4: 9-handed, UTG opens, hero in UTG+2",
        {
            "stakes": {"sb": 2, "bb": 5}, "format": "cash", "venue": "live",
            "table_size": 9, "effective_stack": 500,
            "hero_position": "UTG+2", "hero_cards": ["Qs", "Qc"],
            "preflop_actions": [("UTG", "open", 15)],
            "board": {"flop": None, "turn": None, "river": None},
        },
        "$2/$5 Live cash. 9-handed. $500 effective stacks.",
        f"You're UTG+2 with Q{S}Q{C}.\n"
        f"UTG opens to $15.",
    ),
    (
        "5: Multi-way to the flop, one folds postflop",
        {
            "stakes": {"sb": 1, "bb": 2}, "format": "cash", "venue": "online",
            "table_size": 6, "effective_stack": 200,
            "hero_position": "BB", "hero_cards": ["8h", "7h"],
            "preflop_actions": [("HJ", "open", 6), ("BTN", "call"), ("BB", "call")],
            "board": {"flop": ["Ah", "Tc", "5d"], "turn": "4s", "river": None},
            "flop_actions": [("BB", "check"), ("HJ", "bet", 10),
                             ("BTN", "fold"), ("BB", "call")],
            "turn_actions": [("BB", "check"), ("HJ", "bet", 25)],
            "river_actions": [],
        },
        "$1/$2 Online cash. 6-handed. $200 effective stacks.",
        f"You're in the Big Blind with 8{H}7{H}.\n"
        f"The Hijack opens to $6. The Button calls. You call.\n\n"
        f"Flop ($19): A{H}T{C}5{D}\n"
        f"You check. The Hijack bets $10. The Button folds. You call.\n\n"
        f"Turn ($39): 4{S}\n"
        f"You check. The Hijack bets $25.",
    ),
    (
        "6: 3-bet pot, three-way to the flop",
        {
            "stakes": {"sb": 2, "bb": 5}, "format": "cash", "venue": "online",
            "table_size": 6, "effective_stack": 500,
            "hero_position": "BTN", "hero_cards": ["As", "Ks"],
            "preflop_actions": [("UTG", "open", 15), ("BTN", "3-bet", 50),
                                ("BB", "call"), ("UTG", "call")],
            "board": {"flop": ["Kh", "7c", "4h"], "turn": None, "river": None},
            "flop_actions": [("BB", "check"), ("UTG", "check"), ("BTN", "bet", 75),
                             ("BB", "fold"), ("UTG", "call")],
            "turn_actions": [], "river_actions": [],
        },
        "$2/$5 Online cash. 6-handed. $500 effective stacks.",
        # Brief prints "Flop ($153)"; correct sum is $152 (SB 2 + 50*3).
        f"You're on the Button with A{S}K{S}.\n"
        f"UTG opens to $15. You 3-bet to $50. The Big Blind calls. UTG calls.\n\n"
        f"Flop ($152): K{H}7{C}4{H}\n"
        f"The Big Blind checks. UTG checks. You bet $75. "
        f"The Big Blind folds. UTG calls.",
    ),
    (
        "7: 9-handed, multiway flop after a squeeze",
        {
            "stakes": {"sb": 5, "bb": 10}, "format": "cash", "venue": "online",
            "table_size": 9, "effective_stack": 1000,
            "hero_position": "BB", "hero_cards": ["As", "Ac"],
            "preflop_actions": [("UTG+1", "open", 30), ("LJ", "call"),
                                ("CO", "call"), ("BB", "3-bet", 150),
                                ("UTG+1", "call"), ("LJ", "call")],
            "board": {"flop": ["Js", "Jh", "7c"], "turn": None, "river": None},
            "flop_actions": [("BB", "bet", 200), ("UTG+1", "fold"), ("LJ", "call")],
            "turn_actions": [], "river_actions": [],
        },
        "$5/$10 Online cash. 9-handed. $1,000 effective stacks.",
        # Brief prints "Flop ($475)"; correct sum is $485 (SB 5 + 150*3 + CO 30).
        f"You're in the Big Blind with A{S}A{C}.\n"
        f"UTG+1 opens to $30. The Lojack calls. The Cutoff calls. "
        f"You 3-bet to $150. UTG+1 calls. The Lojack calls.\n\n"
        f"Flop ($485): J{S}J{H}7{C}\n"
        f"You bet $200. UTG+1 folds. The Lojack calls.",
    ),
    (
        "8: 4-way pot, two fold postflop",
        {
            "stakes": {"sb": 1, "bb": 2}, "format": "cash", "venue": "online",
            "table_size": 6, "effective_stack": 200,
            "hero_position": "BB", "hero_cards": ["5s", "5c"],
            "preflop_actions": [("UTG", "open", 6), ("CO", "call"),
                                ("BTN", "call"), ("BB", "call")],
            "board": {"flop": ["9h", "5h", "2c"], "turn": "Kd", "river": None},
            "flop_actions": [("BB", "bet", 15), ("UTG", "raise", 50),
                             ("CO", "fold"), ("BTN", "fold"), ("BB", "call")],
            "turn_actions": [], "river_actions": [],
        },
        "$1/$2 Online cash. 6-handed. $200 effective stacks.",
        # Brief prints "Turn ($133)"; correct sum is $125 ($25 + flop 50*2).
        f"You're in the Big Blind with 5{S}5{C}.\n"
        f"UTG opens to $6. The Cutoff calls. The Button calls. You call.\n\n"
        f"Flop ($25): 9{H}5{H}2{C}\n"
        f"You bet $15. UTG raises to $50. The Cutoff folds. "
        f"The Button folds. You call.\n\n"
        f"Turn ($125): K{D}",
    ),
    (
        "9: Tournament, ICM final table",
        {
            "stakes": {"sb": 0.5, "bb": 1}, "format": "tournament",
            "buy_in": 1500, "stage": "Final table", "ante": 0,
            "table_size": 7, "effective_stack": 25,
            "hero_position": "BTN", "hero_cards": ["As", "Qh"],
            "preflop_actions": [("CO", "open", 2), ("BTN", "all-in", 25)],
            "board": {"flop": None, "turn": None, "river": None},
        },
        "$1,500 Final table tournament. 7-handed. 25bb effective stacks.",
        f"You're on the Button with A{S}Q{H}.\n"
        f"The Cutoff opens to 2bb. You move all-in for 25bb.",
    ),
    (
        "10: Limped pot, hero raises behind",
        {
            "stakes": {"sb": 1, "bb": 2}, "format": "cash", "venue": "live",
            "table_size": 9, "effective_stack": 300,
            "hero_position": "BTN", "hero_cards": ["As", "Ks"],
            "preflop_actions": [("UTG", "limp"), ("UTG+2", "limp"),
                                ("BTN", "open", 15), ("BB", "call"),
                                ("UTG", "call"), ("UTG+2", "call")],
            "board": {"flop": ["Ah", "9c", "6s"], "turn": None, "river": None},
            "flop_actions": [("BB", "check"), ("UTG", "check"),
                             ("UTG+2", "bet", 25), ("BTN", "raise", 80)],
            "turn_actions": [], "river_actions": [],
        },
        "$1/$2 Live cash. 9-handed. $300 effective stacks.",
        f"You're on the Button with A{S}K{S}.\n"
        f"UTG limps. UTG+2 limps. You open to $15. The Big Blind calls. "
        f"UTG calls. UTG+2 calls.\n\n"
        f"Flop ($61): A{H}9{C}6{S}\n"
        f"The Big Blind checks. UTG checks. UTG+2 bets $25. You raise to $80.",
    ),
]


def test_worked_examples_context():
    for name, hand, expected, _ in EXAMPLES:
        got = format_context(hand)
        assert got == expected, f"Example {name}\n  expected: {expected!r}\n  got:      {got!r}"


def test_worked_examples_action_history():
    for name, hand, _, expected in EXAMPLES:
        got = format_action_history(hand)
        assert got == expected, f"Example {name}\n  expected:\n{expected}\n  got:\n{got}"


def test_format_hand_returns_both_blocks():
    name, hand, ctx, ah = EXAMPLES[0]
    assert format_hand(hand) == (ctx, ah)


def test_card_formatting():
    assert format_card("As") == f"A{S}"
    assert format_card("Th") == f"T{H}"
    assert format_card("2d") == f"2{D}"
    assert format_card("9c") == f"9{C}"
    assert format_card("kh") == f"K{H}"          # normalised from lowercase


def test_validation_rejects_bad_input():
    base = EXAMPLES[0][1]

    dup = dict(base, hero_cards=["8h", "9h"],
               board={"flop": ["8h", "6c", "2h"], "turn": None, "river": None})
    bad_seat = dict(base, table_size=6, hero_position="LJ")  # LJ needs 7+ handed
    for label, hand in (("duplicate cards", dup), ("invalid seat", bad_seat)):
        try:
            format_action_history(hand)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {label}")


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
