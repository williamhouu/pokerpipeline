"""Razor's-edge detection: pure hands sitting on a range boundary.

A 100%-frequency spot is trivial for the SOLVER, but hard for a HUMAN when
the hand sits right at the edge of the range -- its NEIGHBOR in the 169
grid does the opposite. "ATo always folds to this 3-bet while AJo always
calls" is exactly the cutoff knowledge training is for, yet the default
difficulty rating (70% frequency+EV weight) scores such spots Easy. This
module detects the boundary deterministically from the node's own strategy
(no equity sim, no LLM) so the difficulty layer can floor these spots into
the Medium/Hard bands, mirroring the opt-in trap-aware mechanism.

Three neighbor relations are checked, all at the SAME decision node:

  * kicker neighbors -- same high card and suitedness, kicker one rank
    away (ATo vs AJo / A9o);
  * the suited/offsuit twin -- ATs vs ATo (a pure suitedness lesson);
  * adjacent pairs -- TT vs JJ / 99.

A neighbor counts as "opposite" when it actually reaches the node
(presence above the worthiness floor) and its DOMINANT action is a
different action class (fold / call / raise / all-in -- two raise sizes do
not make a boundary). More opposite neighbors = a more exceptional hand
(three opposites means the hand is an ISLAND doing something none of its
neighbors do, usually a blocker story) = harder, so the floor is GRADED
by the count rather than pinned to one number.

Pure functions over an existing PreflopSpot; the only data read is the
node's per-class action weights via ``sample_spot``.
"""

from __future__ import annotations

from pipeline.preflop.question_extractor import MIN_PRESENCE
from pipeline.preflop.spot_sampler import PreflopSpot, sample_spot

_RANKS = "23456789TJQKA"
_RANK_IDX = {r: i for i, r in enumerate(_RANKS)}

# Graded difficulty floors by opposite-neighbor count. One opposite
# neighbor = a plain boundary hand (upper-Medium/low-Hard); two = the hand
# is boxed in from multiple directions; three or more = an island doing
# something none of its neighbors do (the hardest, usually blocker-driven).
RAZOR_FLOOR_BY_COUNT: dict[int, int] = {1: 2000, 2: 2300}
RAZOR_FLOOR_MAX: int = 2600  # 3+ opposite neighbors


def razor_floor_for_count(n_opposite: int) -> int | None:
    """The graded difficulty floor for a spot with this many opposite
    neighbors. None when there is no boundary (0 neighbors)."""
    if n_opposite <= 0:
        return None
    return RAZOR_FLOOR_BY_COUNT.get(n_opposite, RAZOR_FLOOR_MAX)


def _action_class(action_label: str) -> str:
    """Collapse an action label to its class -- two raise SIZES are the
    same decision direction, not a range boundary."""
    if action_label.startswith("Fold"):
        return "fold"
    if action_label.startswith("Call") or action_label.startswith("Check"):
        return "call"
    if action_label.startswith("AllIn"):
        return "allin"
    return "raise"


def neighbor_classes(hand_class: str) -> list[str]:
    """The grid neighbors of a 169-class: kicker one step either way (same
    suitedness), the suited/offsuit twin, and for pairs the adjacent pairs.
    Collisions (a kicker stepping onto the high card, ranks off the grid)
    are skipped rather than mapped across categories."""
    out: list[str] = []
    if len(hand_class) == 2 and hand_class[0] == hand_class[1]:  # a pair
        idx = _RANK_IDX.get(hand_class[0])
        if idx is None:
            return []
        for step in (1, -1):
            j = idx + step
            if 0 <= j < len(_RANKS):
                out.append(_RANKS[j] * 2)
        return out
    if len(hand_class) != 3 or hand_class[2] not in "so":
        return []
    hi, lo, suffix = hand_class[0], hand_class[1], hand_class[2]
    hi_idx, lo_idx = _RANK_IDX.get(hi), _RANK_IDX.get(lo)
    if hi_idx is None or lo_idx is None:
        return []
    for step in (1, -1):
        j = lo_idx + step
        # Stay on the grid and below the high card (no category crossing).
        if 0 <= j < len(_RANKS) and j != hi_idx and j < hi_idx:
            out.append(f"{hi}{_RANKS[j]}{suffix}")
    out.append(f"{hi}{lo}{'o' if suffix == 's' else 's'}")  # the twin
    return out


def find_opposite_neighbors(
    spot: PreflopSpot,
    *,
    min_presence: float = MIN_PRESENCE,
) -> list[tuple[str, str]]:
    """Neighbors of ``spot``'s hand whose dominant action at the SAME node
    is a different action class.

    Returns ``[(neighbor_class, neighbor_dominant_action), ...]`` -- empty
    when the hand is interior to its region (no boundary). Neighbors that
    do not reach the node (presence below ``min_presence``) are ignored:
    a hand that is never here cannot mark a boundary.
    """
    hero_class = _action_class(spot.dominant_action)
    out: list[tuple[str, str]] = []
    for nc in neighbor_classes(spot.hero_hand_class):
        neighbor = sample_spot(spot.node, nc)
        if neighbor.presence < min_presence or not neighbor.dominant_action:
            continue
        if _action_class(neighbor.dominant_action) != hero_class:
            out.append((nc, neighbor.dominant_action))
    return out


__all__ = [
    "RAZOR_FLOOR_BY_COUNT",
    "RAZOR_FLOOR_MAX",
    "find_opposite_neighbors",
    "neighbor_classes",
    "razor_floor_for_count",
]
