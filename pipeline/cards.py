"""Card and rank primitives shared across the pipeline.

A card is a 2-character string: a rank from '23456789TJQKA' followed by a
lowercase suit from 'cdhs' -- e.g. 'As', 'Th', '7c'. This matches the notation
PioSolver uses and the scripts in scripts/ already use.
"""
from __future__ import annotations

RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUE = {r: i + 2 for i, r in enumerate(RANKS)}   # '2' -> 2 ... 'A' -> 14
BROADWAY = frozenset({10, 11, 12, 13, 14})             # T, J, Q, K, A


def parse_card(token: str) -> str:
    """Normalise a card to '<RANK><suit>' form, e.g. 'aH' -> 'Ah'. Raises on junk."""
    t = token.strip().replace("10", "T")
    if len(t) != 2:
        raise ValueError(f"not a card: {token!r}")
    rank, suit = t[0].upper(), t[1].lower()
    if rank not in RANKS or suit not in SUITS:
        raise ValueError(f"not a card: {token!r}")
    return rank + suit


def card_rank(card: str) -> str:
    """The rank character of a card, e.g. 'Ah' -> 'A'."""
    return card[0]


def card_suit(card: str) -> str:
    """The suit character of a card, e.g. 'Ah' -> 'h'."""
    return card[1]


def rank_value(card: str) -> int:
    """The numeric rank of a card, 2..14 (ace high)."""
    return RANK_VALUE[card[0]]


def parse_board(board) -> list[str]:
    """Validate and normalise a board.

    Accepts a list of card tokens or a space-separated string. Enforces 3-5
    distinct cards. Returns the normalised card list.
    """
    if isinstance(board, str):
        board = board.split()
    cards = [parse_card(c) for c in board]
    if not 3 <= len(cards) <= 5:
        raise ValueError(f"board must be 3-5 cards, got {len(cards)}: {board}")
    if len(set(cards)) != len(cards):
        raise ValueError(f"board has duplicate cards: {board}")
    return cards


def parse_hole(hole) -> list[str]:
    """Validate and normalise hero's two hole cards.

    Accepts a list of card tokens, a space-separated string ('Ah Ks'), or a
    bare 4-character string ('AhKs'). Enforces exactly 2 distinct cards.
    Returns the normalised card list.
    """
    if isinstance(hole, str):
        text = hole.strip().replace("10", "T")
        hole = text.split() if " " in text else (
            [text[:2], text[2:]] if len(text) == 4 else [text])
    cards = [parse_card(c) for c in hole]
    if len(cards) != 2:
        raise ValueError(f"hole must be exactly 2 cards, got {len(cards)}: {hole}")
    if cards[0] == cards[1]:
        raise ValueError(f"hole has duplicate cards: {hole}")
    return cards
