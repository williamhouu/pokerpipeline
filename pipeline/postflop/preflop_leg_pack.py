"""Pack-backed preflop legs for full-hand play-throughs (July 2026).

The entry-derived preflop leg (:mod:`pipeline.postflop.preflop_entry`) is
honest but capped: a postflop solve carries no preflop EVs, no 3-bet
branch, no preflop ranges -- so the leg ships without the Show-the-math
panel, the ranges grid, per-action EVs, domination facts, or the 4-axis
difficulty. This module lifts that ceiling by sourcing the SAME preflop
decision from the closest-matching PREFLOP RANGE PACK and building the leg
with the full preflop pipeline (facts, EVs, GTO options under the
EV-secondary rule, stat_notes, ranges JSON, skills, difficulty with
trap/razor, validators).

SANCTIONED CROSS-PIPELINE EXCEPTION. The postflop package's rule is
"import no other pipeline's batch/facts/validators/writer" so postflop work
can't disturb preflop. A full-hand question is inherently a COMPOSITION of
the two pipelines, so this ONE module is the seam: every preflop import is
lazy (inside functions), the package still imports and tests without any
pack on disk, and nothing else in ``pipeline/postflop`` may import from
``pipeline.preflop``.

MATCHING, not guessing. A pack leg is only used when the pack provably
describes THIS hand's preflop reality; otherwise the caller falls back to
the entry-derived leg (SRP) or drops the leg (multi-raise). Three gates:

1. **Geometry** -- same table size, same effective stack, and EVERY raise
   size on the solve's preflop line within :data:`OPEN_SIZE_TOLERANCE_BB`
   of the pack's (the open AND, in a 3-bet pot, the 3-bet). A mismatched
   size would make the preflop leg's pot math contradict the postflop
   legs of the SAME hand.
2. **Line** -- the pack contains a node for EVERY decision in the solve's
   ``preflop_summary``, with every non-line seat folding: SRP = the
   opener's first-in node + the defender's facing-the-open node; 3-bet
   pot = those two PLUS the opener's facing-the-3-bet node. Matched
   generically from the summary, so deeper lines extend the same way.
3. **Coherence** -- the pack's dominant action for the hero's hand must
   match what the hand actually did at THAT step (the opener opened, the
   3-bettor raised, the caller called). A play-through advances along the
   as-played line, and the established design keeps each leg's correct
   answer consistent with it; a hand whose pack strategy contradicts the
   line keeps the entry leg (SRP) or drops the preflop leg (multi-raise
   -- the entry weights cannot express a raise-or-call-or-fold decision,
   so there is nothing honest to fall back to).

Pack preference among geometry matches: ``*_IMPROVED`` packs first, then
lexicographic -- deterministic, so batches stay byte-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline.postflop.animation_script import build_preflop_animation_script
from pipeline.postflop.format_writer import POSTFLOP_ROW_COLUMNS
from pipeline.postflop.preflop_entry import _open_to_bb
from pipeline.provenance import build_notes, parse_notes
from pipeline.postflop.solve import PostflopSolve

logger = logging.getLogger(__name__)

# The pack's rendered open size must sit within this of the solve's derived
# open size (packs quantize sizes to a 0.5bb display grid, so allow just over
# half a step).
OPEN_SIZE_TOLERANCE_BB = 0.26


@dataclass(frozen=True)
class PackLineStep:
    """One preflop decision of the solve's line, resolved to its pack node.

    ``step_index`` is the position in ``solve.preflop_summary``;
    ``as_played_prefix`` is what the hand actually did there ("Raise" for an
    open / 3-bet, "Call" for a flat) -- the coherence gate matches the pack's
    dominant action against it. ``size_bb`` is the raise-to size (None for a
    call)."""

    step_index: int
    position: str
    node: Any                 # the pack's decision node for this step
    as_played_prefix: str     # "Raise" | "Call"
    size_bb: float | None


@dataclass(frozen=True)
class PackLegSource:
    """A verified pack + one resolved node per decision of the solve's line."""

    pack: Any                 # PreflopPack (typed loosely: lazy import)
    pack_id: str
    steps: tuple[PackLineStep, ...]   # summary order (open, [3-bet,] defend)
    open_size_bb: float

    # Back-compat views of the SRP pair (steps[0] is always the opener's
    # first-in node, steps[1] the defender's facing-the-open node).
    @property
    def opener_node(self) -> Any:
        return self.steps[0].node

    @property
    def defender_node(self) -> Any:
        return self.steps[1].node

    def steps_for(self, hero_position: str) -> tuple[PackLineStep, ...]:
        """The hero's decisions on this line, in order (the 3-bet-pot opener
        has two: the open and the call of the 3-bet)."""
        return tuple(s for s in self.steps if s.position == hero_position)

    def step_at(self, hero_position: str, step_index: int | None) -> PackLineStep:
        """The hero's step, by summary index when given, else the hero's
        UNIQUE step (raises if ambiguous -- multi-step heroes must say which)."""
        mine = self.steps_for(hero_position)
        if step_index is not None:
            for s in mine:
                if s.step_index == step_index:
                    return s
            raise KeyError(f"{hero_position} has no line step {step_index}")
        if len(mine) != 1:
            raise KeyError(
                f"{hero_position} acts {len(mine)} times on this line; "
                "pass step_index"
            )
        return mine[0]


def find_pack_leg_source(
    solve: PostflopSolve, ranges_root: Path | str,
    *, packs: list | None = None,
) -> PackLegSource | None:
    """The closest preflop pack that provably matches this solve's preflop
    line, or None (caller falls back to entry-derived legs).

    Deterministic: geometry candidates are ordered IMPROVED-first then by
    pack id, and the first full match wins. ``packs`` overrides discovery
    (tests inject unregistered fixture packs).
    """
    try:
        from pipeline.preflop.grammars.types import (  # noqa: PLC0415
            PreflopActionType,
        )
        from pipeline.preflop.node_enumerator import (  # noqa: PLC0415
            enumerate_nodes,
        )
        from pipeline.preflop.pack import (  # noqa: PLC0415
            all_packs,
            discover_packs,
        )
    except Exception as exc:  # noqa: BLE001 - preflop pipeline unavailable
        logger.warning("pack legs unavailable (preflop import failed): %s", exc)
        return None

    if packs is None:
        # The registry is process-global and discover_packs refuses to
        # re-register (the admin panel / a prior call may already have
        # discovered) -- reuse what's registered, discover only when empty.
        try:
            packs = list(all_packs()) or discover_packs(Path(ranges_root))
        except Exception as exc:  # noqa: BLE001
            logger.warning("pack legs unavailable (discovery failed): %s", exc)
            return None

    # The solve's preflop line as (position, is_raise, to_bb) decisions, in
    # order. Every step of the heads-up-to-the-flop summary is a decision we
    # want a pack node for (open / [3-bet / call-the-3-bet]); anything the
    # matcher can't express (a limped or unparseable line) fails generically.
    raise_verbs = ("open", "raise", "3-bet", "4-bet", "5-bet")
    line = [
        (st.position, st.verb in raise_verbs, st.to_bb)
        for st in solve.preflop_summary
    ]
    if not line or not line[0][1] or any(
        not is_raise and to_bb for _, is_raise, to_bb in line
    ):
        logger.info("pack legs: %s has no matchable raise-first line", solve.solve_id)
        return None
    solve_raise_sizes = [to_bb for _, is_raise, to_bb in line if is_raise]
    if any(s is None for s in solve_raise_sizes):
        logger.info("pack legs: %s line lacks raise sizes; skipping", solve.solve_id)
        return None

    stack = round(solve.effective_stack_bb)
    candidates = [
        p for p in packs
        if p.table_size == solve.table_size
        and round(p.stack_depth_bb) == stack
    ]
    candidates.sort(key=lambda p: (not p.pack_id.endswith("_IMPROVED"), p.pack_id))

    for pack in candidates:
        try:
            nodes = enumerate_nodes([pack])
        except Exception as exc:  # noqa: BLE001 - a broken pack must not kill legs
            logger.warning("pack %s enumeration failed: %s", pack.pack_id, exc)
            continue

        def _matches_line_prefix(node, k: int) -> bool:
            """Node is ``line[k]``'s decision: the actor matches, the
            history's NON-FOLD actions are exactly the line's first ``k``
            steps (position + raise-vs-call), and every other seat folded.
            Sizes are verified separately via resolve (unit-safe)."""
            if node.actor != line[k][0]:
                return False
            hist = node.history_before
            acted = [
                a for a in hist if a.action_type is not PreflopActionType.FOLD
            ]
            if len(acted) != k:
                return False
            for a, (pos, is_raise, _size) in zip(acted, line[:k]):
                if a.position != pos:
                    return False
                if is_raise and a.action_type is not PreflopActionType.RAISE:
                    return False
                if not is_raise and a.action_type is not PreflopActionType.CALL:
                    return False
            return True

        step_nodes: list[Any] = []
        for k in range(len(line)):
            found = [n for n in nodes if _matches_line_prefix(n, k)]
            if len(found) != 1:
                # 0 = the pack lacks this decision; >1 = several raise sizes
                # reach it and the structural match can't disambiguate --
                # resolve each candidate's sizes and keep the one matching
                # the solve's line.
                found = [
                    n for n in found
                    if _sizes_match(n, pack, solve_raise_sizes, partial=True)
                ]
            if len(found) != 1:
                step_nodes = []
                break
            step_nodes.append(found[0])
        if not step_nodes:
            continue

        # Geometry: EVERY raise size on the line (resolved on the deepest
        # node, whose history contains them all) within tolerance.
        if not _sizes_match(step_nodes[-1], pack, solve_raise_sizes):
            logger.info(
                "pack %s line found but raise sizes differ from solve %s; skipping",
                pack.pack_id, solve.solve_id,
            )
            continue

        steps = tuple(
            PackLineStep(
                step_index=k,
                position=line[k][0],
                node=step_nodes[k],
                as_played_prefix="Raise" if line[k][1] else "Call",
                size_bb=line[k][2] if line[k][1] else None,
            )
            for k in range(len(line))
        )
        logger.info(
            "pack legs: %s matches %s (%s)",
            pack.pack_id, solve.solve_id,
            ", ".join(f"{s.position} {s.as_played_prefix}"
                      + (f" {s.size_bb:g}bb" if s.size_bb else "")
                      for s in steps),
        )
        return PackLegSource(
            pack=pack, pack_id=pack.pack_id, steps=steps,
            open_size_bb=float(solve_raise_sizes[0]),
        )
    return None


def _sizes_match(
    node, pack, solve_raise_sizes: list, *, partial: bool = False,
) -> bool:
    """Whether the raise sizes in ``node``'s history (resolved to bb,
    unit-safe across pack grammars) match the solve's, within
    :data:`OPEN_SIZE_TOLERANCE_BB`. ``partial``: the node may sit mid-line,
    so only compare the sizes present so far."""
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )

    try:
        resolved = resolve_preflop_history(node.history_before, pack)
    except Exception:  # noqa: BLE001 - a broken node never matches
        return False
    pack_sizes = [s for s in resolved.sizes_bb if s is not None]
    expected = solve_raise_sizes[: len(pack_sizes)] if partial else solve_raise_sizes
    if len(pack_sizes) != len(expected):
        return False
    return all(
        abs(float(p) - float(e)) <= OPEN_SIZE_TOLERANCE_BB
        for p, e in zip(pack_sizes, expected)
    )


def compute_pack_leg_difficulty(
    source: PackLegSource, hero_position: str, hero_combo: str,
    solve: PostflopSolve, *,
    equity_runouts: int,
    trap_difficulty: bool = False,
    razor_difficulty: bool = False,
    step_index: int | None = None,
) -> int | None:
    """The pack leg's 4-axis difficulty WITHOUT generating anything (used by
    the hand-difficulty pre-pass). None when the leg wouldn't use the pack
    (coherence gate) -- the caller falls back to the entry difficulty (SRP)
    or drops the leg (multi-raise). ``step_index`` picks the hero's decision
    on a multi-step line (see :meth:`PackLegSource.step_at`)."""
    built = _build_pack_facts(
        source, hero_position, hero_combo, solve,
        equity_runouts=equity_runouts, step_index=step_index,
    )
    if built is None:
        return None
    _facts, difficulty = _facts_difficulty(
        built, trap_difficulty=trap_difficulty,
        razor_difficulty=razor_difficulty,
    )
    return difficulty.score


def _build_pack_facts(
    source: PackLegSource, hero_position: str, hero_combo: str,
    solve: PostflopSolve, *, equity_runouts: int,
    step_index: int | None = None,
    terminal_fold: bool = False,
    terminal_raise: bool = False,
):
    """Sample + fully enrich the preflop facts for one of the hero's leg
    decisions, or None when the coherence gate fails (pack dominant != the
    as-played action at that step).

    ``terminal_fold=True`` builds a PREFLOP-ENDING leg (the hand stops on a
    correct fold): the coherence gate flips from "matches the as-played
    continuation" to "the dominant action IS Fold" -- there is no as-played
    action after a fold, so matching one would be meaningless."""
    from dataclasses import replace  # noqa: PLC0415

    from pipeline.preflop.ev_engine import (  # noqa: PLC0415
        compute_ev_gap_bb,
        compute_price_geometry,
    )
    from pipeline.preflop.spot_sampler import sample_spot  # noqa: PLC0415
    from pipeline.preflop.batch import (  # noqa: PLC0415
        ev_gap_from_action_evs,
    )
    from pipeline.preflop.fact_extractor import extract_facts  # noqa: PLC0415
    from pipeline.preflop_ranges import (  # noqa: PLC0415
        combo_str_to_hand_class,
    )

    step = source.step_at(hero_position, step_index)
    node = step.node
    as_played = step.as_played_prefix
    hand_class = combo_str_to_hand_class(hero_combo)
    spot = _sampled_pack_spot(node, hand_class, source.pack, combo=hero_combo)
    if spot is None:  # artifact-material jam mix: the leg must never be asked
        return None
    # Coherence gate: pack dominant must be the as-played FAMILY ("Raise" /
    # "Call"; an all-in is its own token, so a mostly-jam hand never passes
    # as a sized raise). Size-level matching isn't needed: these pack nodes
    # carry one sized raise each, and the line matcher already verified that
    # size against the solve's. A terminal-fold leg instead REQUIRES the
    # dominant action to be Fold (the hand ends here by design).
    if terminal_fold:
        if not spot.dominant_action.startswith("Fold"):
            return None
    elif terminal_raise:
        # Raise-ending leg: the correct action is the sized re-raise (never
        # AllIn -- the artifact-jam rule keeps jams out of invented lines).
        if not spot.dominant_action.startswith("Raise"):
            return None
    elif not spot.dominant_action.startswith(as_played):
        return None
    facts = extract_facts(spot, source.pack, equity_runouts=equity_runouts)
    _pot, _call, _be = compute_price_geometry(facts, source.pack)
    facts = replace(
        facts,
        break_even_equity=_be,
        price_pot_bb=_pot,
        price_call_bb=_call,
        rake_pct=source.pack.rake_pct or 0.0,
    )
    ev_gap = ev_gap_from_action_evs(facts, source.pack)
    if ev_gap is None:
        ev_gap = compute_ev_gap_bb(facts, source.pack)
    return replace(facts, ev_gap_bb=ev_gap)


def _facts_difficulty(
    facts, *, trap_difficulty: bool, razor_difficulty: bool,
):
    """The batch driver's difficulty computation, mirrored exactly (near-pure
    EV credit + the opt-in trap/razor floors)."""
    from pipeline.preflop.batch import (  # noqa: PLC0415
        _NEAR_PURE_DOMINANT_FREQ,
        _NEAR_PURE_EV_CREDIT_BB,
    )
    from pipeline.preflop.difficulty import compute_difficulty  # noqa: PLC0415

    ev_for_difficulty = facts.ev_gap_bb
    if facts.spot.dominant_frequency >= _NEAR_PURE_DOMINANT_FREQ:
        ev_for_difficulty = _NEAR_PURE_EV_CREDIT_BB
    return facts, compute_difficulty(
        facts, ev_gap_bb=ev_for_difficulty,
        apply_trap_bump=trap_difficulty,
        apply_razor_bump=razor_difficulty,
    )


# --- preflop-ENDING hands (balanced-lengths generator, July 2026) -----------
# A quarter of production hands end preflop: hero's correct action at their
# (deepest) preflop decision is Fold, the hand stops there, and -- when the
# fold is TO a raise -- the raiser reveals a clearly-stronger starting hand
# (the same vindication idea as the postflop showdown module). First-in folds
# (open-folding the Button) get no reveal: nobody bet, the blinds just take
# it, so the leg ships without a resolution (the documented normal case).

_RANKS_DESC = "AKQJT98765432"
_SUITS = "cdhs"
_RANK_WORDS = {
    "A": "aces", "K": "kings", "Q": "queens", "J": "jacks", "T": "tens",
    "9": "nines", "8": "eights", "7": "sevens", "6": "sixes", "5": "fives",
    "4": "fours", "3": "threes", "2": "deuces",
}
_RANK_NAME = {
    "A": "ace", "K": "king", "Q": "queen", "J": "jack", "T": "ten",
    "9": "nine", "8": "eight", "7": "seven", "6": "six", "5": "five",
    "4": "four", "3": "three", "2": "deuce",
}
# The revealed hand must be a CLEAR favourite over hero's exact combo --
# coinflips don't vindicate a fold.
_PREFLOP_VINDICATION_EQUITY = 0.55
_PREFLOP_EQUITY_SAMPLES = 200


def _class_combos(hand_class: str) -> list[str]:
    """Concrete combos for a 169-grid class, deterministic order."""
    r1, r2 = hand_class[0], hand_class[1]
    if len(hand_class) == 2:  # pair
        return [
            f"{r1}{a}{r2}{b}"
            for i, a in enumerate(_SUITS) for b in _SUITS[i + 1:]
        ]
    if hand_class.endswith("s"):
        return [f"{r1}{s}{r2}{s}" for s in _SUITS]
    return [f"{r1}{a}{r2}{b}" for a in _SUITS for b in _SUITS if a != b]


def _preflop_hand_label(combo: str) -> str:
    """Plain-English starting-hand name for the reveal caption."""
    r1, r2, suited = combo[0], combo[2], combo[1] == combo[3]
    if r1 == r2:
        return f"a pair of {_RANK_WORDS[r1]}"
    hi, lo = sorted((r1, r2), key=_RANKS_DESC.index)
    tail = " suited" if suited else " offsuit"
    return f"{_RANK_NAME[hi]}-{_RANK_NAME[lo]}{tail}"


def _sampled_pack_spot(node, hand_class: str, pack, *, combo: str | None = None):
    """``sample_spot`` + the deep-pack ARTIFACT-STRIP (July 2026).

    On packs whose all-ins are tree artifacts (``not pack_allins_realistic``)
    the spot's trace AllIn dust is stripped + renormalised (so pack-leg
    options, qualifiers, and the coherence gates all see the real mix), and
    a MATERIAL jam mix returns ``None`` -- the leg must never be asked.
    Realistic short-stack packs pass through untouched. The audit
    re-verifier rebuilds legs through this same helper, so rows stay
    byte-identical."""
    from pipeline.preflop.pack import pack_allins_realistic  # noqa: PLC0415
    from pipeline.preflop.spot_sampler import (  # noqa: PLC0415
        sample_spot,
        strip_artifact_allins,
    )

    spot = sample_spot(node, hand_class, combo=combo)
    if pack_allins_realistic(pack):
        return spot
    spot = strip_artifact_allins(spot)
    return None if spot.artifact_material else spot


def fold_ender_hand_classes(
    source: PackLegSource, hero_position: str, *,
    step_index: int | None = None,
    min_frequency: float, max_frequency: float,
) -> list[str]:
    """Hand classes whose dominant action at hero's step is FOLD, within the
    worthiness window -- the candidate pool for preflop-ending hands.
    Deterministic (sorted)."""
    step = source.step_at(hero_position, step_index)
    out: list[str] = []
    for hi_i, hi in enumerate(_RANKS_DESC):
        for lo_i, lo in enumerate(_RANKS_DESC):
            if lo_i < hi_i:
                continue
            classes = [hi + lo] if hi == lo else [hi + lo + "s", hi + lo + "o"]
            for hand_class in classes:
                try:
                    spot = _sampled_pack_spot(step.node, hand_class, source.pack)
                except (KeyError, ValueError):
                    continue
                if (
                    spot is not None
                    and spot.dominant_action.startswith("Fold")
                    and min_frequency <= spot.dominant_frequency <= max_frequency
                ):
                    out.append(hand_class)
    return out


def combo_for_hand_class(hand_class: str, rng) -> str:
    """A concrete combo for a class, seeded-deterministic."""
    return rng.choice(_class_combos(hand_class))


def raise_ender_hand_classes(
    source: PackLegSource, hero_position: str, *,
    step_index: int | None = None,
    min_frequency: float, max_frequency: float,
) -> list[str]:
    """Hand classes whose dominant action at hero's step is the sized
    RE-RAISE (never AllIn -- the artifact-jam rule), within the worthiness
    window -- the candidate pool for raise-ending hands (a correct 3-bet /
    4-bet that ends the hand). Deterministic (sorted grid order)."""
    step = source.step_at(hero_position, step_index)
    out: list[str] = []
    for hi_i, hi in enumerate(_RANKS_DESC):
        for lo_i, lo in enumerate(_RANKS_DESC):
            if lo_i < hi_i:
                continue
            classes = [hi + lo] if hi == lo else [hi + lo + "s", hi + lo + "o"]
            for hand_class in classes:
                try:
                    spot = _sampled_pack_spot(step.node, hand_class, source.pack)
                except (KeyError, ValueError):
                    continue
                if (
                    spot is not None
                    and spot.dominant_action.startswith("Raise")
                    and min_frequency <= spot.dominant_frequency <= max_frequency
                ):
                    out.append(hand_class)
    return out


def _find_response_node(source: PackLegSource, hero_step) -> Any | None:
    """The villain's decision node FACING hero's raise: the pack node whose
    ``history_before`` equals hero's node history plus hero's sized raise.
    None when the pack tree doesn't extend past hero's raise."""
    from pipeline.preflop.grammars.types import PreflopActionType  # noqa: PLC0415
    from pipeline.preflop.node_enumerator import enumerate_nodes  # noqa: PLC0415

    hero_node = hero_step.node
    raise_opt = next(
        (a for a in hero_node.actions
         if a.action_type is PreflopActionType.RAISE),
        None,
    )
    if raise_opt is None:
        return None
    from pipeline.preflop.grammars.types import ParsedAction  # noqa: PLC0415

    want = tuple(hero_node.history_before) + (ParsedAction(
        position=hero_step.position,
        action_type=PreflopActionType.RAISE,
        raise_size_pct=raise_opt.raise_size_pct,
    ),)
    for node in enumerate_nodes([source.pack]):
        if tuple(node.history_before) == want:
            return node
    return None


def build_preflop_raise_resolution(
    source: PackLegSource,
    hero_position: str,
    hero_combo: str,
    solve: PostflopSolve,
    *,
    hand_id: str,
    step_index: int | None = None,
) -> dict | None:
    """The ``resolution`` for a raise-ENDING hand: hero's correct 3-bet /
    4-bet, the villain's REAL fold (sampled from the pack's response node,
    weighted by their reach into the pot times their fold frequency facing
    the raise), the folded hand revealed, and the pot pushed to hero.

    Call-or-fold rule: only the FOLD slice is shown (no postflop solve
    exists for the raised pot, so a call cannot be played out honestly).
    None when the pack has no response node or villain never folds.
    """
    import random as _random  # noqa: PLC0415
    import zlib  # noqa: PLC0415

    from pipeline.action_history import format_card  # noqa: PLC0415
    from pipeline.postflop.animation_script import (  # noqa: PLC0415
        _table,
        _walk_preflop,
    )
    from pipeline.preflop.action_history import (  # noqa: PLC0415
        resolve_preflop_history,
    )
    from pipeline.preflop.grammars.types import PreflopActionType  # noqa: PLC0415
    from pipeline.preflop.grammars.types import ParsedAction  # noqa: PLC0415
    from pipeline.preflop.spot_sampler import sample_spot  # noqa: PLC0415

    hero_step = source.step_at(hero_position, step_index)
    hero_idx = source.steps.index(hero_step)
    prior_raises = [
        s for s in source.steps[:hero_idx] if s.as_played_prefix == "Raise"
    ]
    response_node = _find_response_node(source, hero_step)
    if response_node is None:
        return None
    villain = response_node.actor
    hero_cards = [hero_combo[:2], hero_combo[2:]]
    rng = _random.Random(
        zlib.crc32(f"{hand_id}|preflop_raise|{hero_combo}".encode())
    )

    # Villain's reach node: where THEY last acted before hero's raise (the
    # opener's open / the 3-bettor's 3-bet). Weight = reach freq x fold freq.
    villain_prior = next(
        (s for s in reversed(source.steps[:hero_idx]) if s.position == villain),
        None,
    )
    candidates: list[tuple[str, float]] = []
    for hi_i, hi in enumerate(_RANKS_DESC):
        for lo_i, lo in enumerate(_RANKS_DESC):
            if lo_i < hi_i:
                continue
            classes = [hi + lo] if hi == lo else [hi + lo + "s", hi + lo + "o"]
            for hand_class in classes:
                try:
                    rspot = sample_spot(response_node, hand_class)
                except (KeyError, ValueError):
                    continue
                fold_freq = rspot.action_frequencies.get("Fold", 0.0)
                if fold_freq <= 0.0:
                    continue
                reach = 1.0
                if villain_prior is not None:
                    try:
                        vspot = sample_spot(villain_prior.node, hand_class)
                    except (KeyError, ValueError):
                        continue
                    reach = sum(
                        f for a, f in vspot.action_frequencies.items()
                        if a.startswith(villain_prior.as_played_prefix)
                    )
                weight = reach * fold_freq
                if weight <= 0.0:
                    continue
                for c in _class_combos(hand_class):
                    if not set([c[:2], c[2:]]) & set(hero_cards):
                        candidates.append((c, weight))
    if not candidates:
        return None
    pick = rng.choices(
        [c for c, _ in candidates], weights=[w for _, w in candidates], k=1,
    )[0]
    villain_cards = [pick[:2], pick[2:]]

    # Hero's raise-TO size in bb: resolve the history INCLUDING hero's raise
    # with the same machinery the prose uses (pack rounding honoured).
    raise_opt = next(
        a for a in hero_step.node.actions
        if a.action_type is PreflopActionType.RAISE
    )
    resolved = resolve_preflop_history(
        tuple(hero_step.node.history_before) + (ParsedAction(
            position=hero_position,
            action_type=PreflopActionType.RAISE,
            raise_size_pct=raise_opt.raise_size_pct,
        ),),
        source.pack,
    )
    raise_to_bb = resolved.sizes_bb[-1]

    table = _table(solve)
    _walk_preflop(table, solve, stop_before_step=hero_step.step_index)
    table.events = []
    table.wager_to(hero_position, "raise", float(raise_to_bb))
    table.emit("fold", seat=villain)
    label = _preflop_hand_label(pick)
    table.emit(
        "reveal", seat=villain, cards=list(villain_cards), hand_label=label,
        best_five=list(villain_cards), folded=True,
    )
    win_event = table.emit("win", seat=hero_position, reason="fold")
    table.stacks[hero_position] += table.pot
    table.money(win_event, pot_bb=table.pot,
                stack_bb=table.stacks[hero_position])
    cards_emoji = "".join(format_card(c) for c in villain_cards)
    seat_subj = {
        "BTN": "The Button", "SB": "The Small Blind", "BB": "The Big Blind",
        "CO": "The Cutoff", "HJ": "The Hijack", "LJ": "The Lojack",
    }.get(villain, villain)
    raise_word = {1: "3-bet", 2: "4-bet"}.get(len(prior_raises), "raise")
    return {
        "vindicates": f"{raise_word.capitalize()}",
        "villain_seat": villain,
        "villain_cards": list(villain_cards),
        "summary": (
            f"{seat_subj} folds {cards_emoji} ({label}). "
            f"Your {raise_word} takes the pot."
        ),
        "events": table.events,
    }


def build_preflop_fold_resolution(
    source: PackLegSource,
    hero_position: str,
    hero_combo: str,
    solve: PostflopSolve,
    *,
    hand_id: str,
    step_index: int | None = None,
) -> dict | None:
    """The ``resolution`` object for a preflop-ending fold, or None.

    Vindication rule (mirrors :mod:`pipeline.postflop.showdown`): the seat
    whose raise hero folds to reveals a hand drawn from their REAL raising
    range at that node, restricted to combos that are a clear preflop
    favourite over hero's exact hand (equity >= 55%, seeded MC). First-in
    folds (no raise before hero) return None -- nobody to reveal.
    """
    import random as _random  # noqa: PLC0415
    import zlib  # noqa: PLC0415

    from pipeline.action_history import format_card  # noqa: PLC0415
    from pipeline.postflop.animation_script import (  # noqa: PLC0415
        _table,
        _walk_preflop,
    )
    from pipeline.preflop.equity import preflop_hand_equity  # noqa: PLC0415
    from pipeline.preflop.spot_sampler import sample_spot  # noqa: PLC0415

    hero_step = source.step_at(hero_position, step_index)
    hero_idx = source.steps.index(hero_step)
    aggressors = [
        s for s in source.steps[:hero_idx] if s.as_played_prefix == "Raise"
    ]
    if not aggressors:
        return None  # first-in fold: the blinds take it, nothing to reveal
    villain_step = aggressors[-1]
    villain = villain_step.position
    hero_cards = [hero_combo[:2], hero_combo[2:]]
    rng = _random.Random(
        zlib.crc32(f"{hand_id}|preflop_fold|{hero_combo}".encode())
    )

    # Villain's raising range at their node: per-class raise frequency ->
    # concrete combos (hero's cards blocked), gated to clear favourites.
    candidates: list[tuple[str, float]] = []
    for hi_i, hi in enumerate(_RANKS_DESC):
        for lo_i, lo in enumerate(_RANKS_DESC):
            if lo_i < hi_i:
                continue
            classes = [hi + lo] if hi == lo else [hi + lo + "s", hi + lo + "o"]
            for hand_class in classes:
                try:
                    vspot = sample_spot(villain_step.node, hand_class)
                except (KeyError, ValueError):
                    continue
                freq = sum(
                    f for a, f in vspot.action_frequencies.items()
                    if a.startswith(("Raise", "AllIn", "3-bet", "4-bet"))
                ) if hasattr(vspot, "action_frequencies") else (
                    vspot.dominant_frequency
                    if vspot.dominant_action.startswith("Raise") else 0.0
                )
                if freq <= 0.0:
                    continue
                # class-level pre-gate with a representative combo
                rep = _class_combos(hand_class)[0]
                if set([rep[:2], rep[2:]]) & set(hero_cards):
                    rep = next(
                        (c for c in _class_combos(hand_class)
                         if not set([c[:2], c[2:]]) & set(hero_cards)),
                        None,
                    )
                if rep is None:
                    continue
                eq = preflop_hand_equity(
                    [rep[:2], rep[2:]], hero_cards,
                    n_samples=_PREFLOP_EQUITY_SAMPLES,
                    rng=_random.Random(
                        zlib.crc32(f"{hand_id}|{hand_class}".encode())
                    ),
                )
                if eq < _PREFLOP_VINDICATION_EQUITY:
                    continue
                for c in _class_combos(hand_class):
                    if not set([c[:2], c[2:]]) & set(hero_cards):
                        candidates.append((c, freq))
    if not candidates:
        return None
    combos = [c for c, _ in candidates]
    weights = [w for _, w in candidates]
    pick = rng.choices(combos, weights=weights, k=1)[0]
    villain_cards = [pick[:2], pick[2:]]

    # Rebuild the chip walk to hero's decision so the resolution continues
    # the leg's exact pot/stacks (same numbers as the main timeline; the
    # step's own step_index is the authoritative preflop_summary position,
    # exactly what the leg's animation used).
    table = _table(solve)
    _walk_preflop(table, solve, stop_before_step=hero_step.step_index)
    # drop the terminal decision event; the resolution numbers from 1
    table.events = []
    table.emit("fold", seat=hero_position)
    label = _preflop_hand_label(pick)
    table.emit(
        "reveal", seat=villain, cards=list(villain_cards), hand_label=label,
        best_five=list(villain_cards),
    )
    win_event = table.emit("win", seat=villain, reason="fold")
    table.stacks[villain] += table.pot
    table.money(win_event, pot_bb=table.pot, stack_bb=table.stacks[villain])
    cards_emoji = "".join(format_card(c) for c in villain_cards)
    seat_subj = {
        "BTN": "The Button", "SB": "The Small Blind", "BB": "The Big Blind",
        "CO": "The Cutoff", "HJ": "The Hijack", "LJ": "The Lojack",
    }.get(villain, villain)
    return {
        "vindicates": "Fold",
        "villain_seat": villain,
        "villain_cards": list(villain_cards),
        "summary": (
            f"{seat_subj} shows {cards_emoji} ({label}). "
            "Folding saved you money."
        ),
        "events": table.events,
    }


def build_pack_preflop_leg_row(
    source: PackLegSource,
    hero_position: str,
    hero_combo: str,
    solve: PostflopSolve,
    *,
    number: int,
    hand_id: str,
    sequence_index: int,
    sequence_total: int,
    use_placeholder: bool,
    client: object | None,
    model: str,
    temperature: float,
    max_tokens: int,
    answer_style: str,
    display_in_bb: bool,
    equity_runouts: int,
    trap_difficulty: bool = False,
    razor_difficulty: bool = False,
    system_prompt: str | None = None,
    usage_cb=None,
    prebuilt=None,
    step_index: int | None = None,
    terminal_fold: bool = False,
    terminal_raise: bool = False,
    run_claim_checker: bool = False,
    revise_pass: bool = False,
    final_audit: bool = False,
    second_rewrite: bool = False,
) -> tuple[dict[str, str] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Build the pack-backed preflop leg row (POSTFLOP schema).

    ``system_prompt`` overrides the PREFLOP system prompt for the
    explanation call (this leg is written by the preflop pipeline's
    generator, not the postflop or preflop-entry one). None = the active
    preflop prompt (override file or built-in default). ``step_index``
    picks the hero's decision on a multi-step line (the 3-bet-pot opener
    has two legs: the open and the call of the 3-bet); None = the hero's
    unique step (SRP).

    Returns ``(row, meta_record, failure)``. ``(None, None, None)`` means
    "pack not applicable for this hand" (coherence gate) and the caller
    falls back to the entry-derived leg (SRP) or drops the leg
    (multi-raise). A generation failure returns a failure record like the
    other leg builders.
    """
    from pipeline.preflop.batch import _placeholder_explanation  # noqa: PLC0415
    from pipeline.preflop.explanation_generator import (  # noqa: PLC0415
        generate_preflop_answer_explanation,
    )
    from pipeline.explanation_generator import (  # noqa: PLC0415
        ExplanationValidationError,
    )
    from pipeline.preflop.format_writer import build_preflop_row  # noqa: PLC0415
    from pipeline.preflop.options import build_options  # noqa: PLC0415
    from pipeline.preflop.validators import (  # noqa: PLC0415
        run_preflop_soft_validators,
    )

    if prebuilt is not None:
        # The hand-difficulty pre-pass already paid the equity sim and the
        # difficulty computation; reuse both.
        facts, difficulty = prebuilt
    else:
        built = _build_pack_facts(
            source, hero_position, hero_combo, solve,
            equity_runouts=equity_runouts, step_index=step_index,
            terminal_fold=terminal_fold, terminal_raise=terminal_raise,
        )
        if built is None:
            return None, None, None
        facts, difficulty = _facts_difficulty(
            built, trap_difficulty=trap_difficulty,
            razor_difficulty=razor_difficulty,
        )

    options, correct = build_options(
        facts, style=answer_style, pack=source.pack,
    )
    # USAGE-CALLBACK ARITY INVARIANT: the POSTFLOP batch's counter takes ONE
    # usage OBJECT (`_usage(response.usage)`), but the PREFLOP generator
    # reports `(model, in_t, out_t, cache_c, cache_r)` -- five positionals.
    # This cross-pipeline seam MUST adapt between the two conventions, or the
    # first real (non-dry-run) pack leg dies with a TypeError that no dry-run
    # test can see (the June-2026 revise-pass bug was this same class).
    # `in_t`/`out_t` are read off response.usage, so the adapter hands over
    # exactly the numbers the postflop counter would have read itself.
    pre_usage_cb = None
    if usage_cb is not None:
        def pre_usage_cb(
            _model: str, in_t: int, out_t: int, cache_c: int, cache_r: int
        ) -> None:
            # Cache tokens forwarded too (July 2026): the preflop generator
            # prompt-caches, so dropping cache_c/cache_r here made pack-leg
            # cache spend invisible to the batch totals (THE USAGE RULE).
            usage_cb(SimpleNamespace(
                input_tokens=in_t, output_tokens=out_t,
                cache_creation_input_tokens=cache_c,
                cache_read_input_tokens=cache_r,
            ))

    try:
        if use_placeholder:
            explanation = _placeholder_explanation(options, correct)
        else:
            explanation = generate_preflop_answer_explanation(
                facts, options, correct, client=client, model=model,
                temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt,
                usage_callback=pre_usage_cb,
            )
    except ExplanationValidationError as exc:
        return None, None, {
            "node_id": facts.spot.node.node_id, "hero_combo": hero_combo,
            "hand_id": hand_id, "error_message": str(exc),
            "attempt_text": exc.last_attempt_text,
        }

    # --- Layer-7 audit on the PACK preflop leg (July 2026) ----------------
    # Mirrors the preflop batch's flow with the PREFLOP checker prompt (the
    # postflop checker's failure catalogue doesn't fit a preflop decision;
    # this leg carries the full preflop SOLVER DATA block the preflop
    # checker expects). Flag-only records issues; revise_pass rewrites
    # flagged prose (re-validated by the preflop hard validators; a rewrite
    # that breaks one is discarded and the original ships); final_audit
    # re-checks the kept rewrite. All fail open -- an audit error never
    # drops a leg.
    claim_issues: list[str] = []
    revise_record: dict[str, Any] | None = None
    if not use_placeholder and client is not None and (run_claim_checker or revise_pass):
        from pipeline.preflop.batch import _safe_claim_check  # noqa: PLC0415
        from pipeline.preflop.claim_checker import (  # noqa: PLC0415
            CHECKER_SYSTEM_PROMPT,
        )
        from pipeline.preflop.reviser import revise_explanation  # noqa: PLC0415

        cc = _safe_claim_check(
            explanation.answer_explanation, facts, client, model=model,
            system_prompt=CHECKER_SYSTEM_PROMPT,
            node_id=facts.spot.node.node_id,
            usage_callback=pre_usage_cb,
        )
        gate_issues = (
            [f"{i.claim} -- {i.problem}" for i in cc.issues]
            if cc is not None else []
        )
        if revise_pass:
            if not gate_issues:
                revise_record = {"status": "clean", "gate_issues": []}
            else:
                original_prose = explanation.answer_explanation
                try:
                    rev = revise_explanation(
                        explanation, facts, issues=gate_issues,
                        client=client, model=model, temperature=temperature,
                        max_tokens=max_tokens, system_prompt=system_prompt,
                        usage_callback=pre_usage_cb,
                    )
                except Exception as exc:  # noqa: BLE001 - never drop a leg
                    logger.warning(
                        "pack leg reviser failed for %s: %s",
                        facts.spot.node.node_id, exc,
                    )
                    rev = None
                if rev is not None and rev.changed:
                    explanation = rev.explanation  # ship the rewrite
                    revise_record = {
                        "status": "fixed",
                        "gate_issues": gate_issues,
                        "original_explanation": original_prose,
                        "revised_explanation": rev.explanation.answer_explanation,
                    }
                    if final_audit:
                        cc4 = _safe_claim_check(
                            explanation.answer_explanation, facts, client,
                            model=model, system_prompt=CHECKER_SYSTEM_PROMPT,
                            node_id=facts.spot.node.node_id,
                            usage_callback=pre_usage_cb,
                        )
                        if cc4 is not None:
                            revise_record["final_audit_issues"] = [
                                f"{i.claim} -- {i.problem}" for i in cc4.issues
                            ]
                        # SECOND REWRITE ROUND (July 2026, strict-clean):
                        # same bounded extra round as the postflop legs
                        # (pipeline.postflop.layer7) -- revise vs the
                        # final-audit issues, then re-audit. One round only;
                        # a discarded/unchanged second rewrite keeps round
                        # 1's text and flags.
                        fa_issues = list(
                            revise_record.get("final_audit_issues") or []
                        )
                        if second_rewrite and fa_issues:
                            try:
                                rev2 = revise_explanation(
                                    explanation, facts, issues=fa_issues,
                                    client=client, model=model,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    system_prompt=system_prompt,
                                    usage_callback=pre_usage_cb,
                                )
                            except Exception as exc:  # noqa: BLE001
                                logger.warning(
                                    "pack leg second-round reviser failed"
                                    " for %s: %s",
                                    facts.spot.node.node_id, exc,
                                )
                                rev2 = None
                            second_rec: dict[str, Any] = {
                                "issues_before": fa_issues,
                            }
                            if rev2 is not None and rev2.changed:
                                explanation = rev2.explanation
                                second_rec["status"] = "fixed"
                                revise_record["revised_explanation"] = (
                                    rev2.explanation.answer_explanation
                                )
                                cc5 = _safe_claim_check(
                                    explanation.answer_explanation, facts,
                                    client, model=model,
                                    system_prompt=CHECKER_SYSTEM_PROMPT,
                                    node_id=facts.spot.node.node_id,
                                    usage_callback=pre_usage_cb,
                                )
                                revise_record["final_audit_issues"] = (
                                    [
                                        f"{i.claim} -- {i.problem}"
                                        for i in cc5.issues
                                    ]
                                    if cc5 is not None else []
                                )
                            else:
                                second_rec["status"] = (
                                    "discarded"
                                    if rev2 is not None
                                    and getattr(rev2, "rejected_reason", "")
                                    else "unchanged"
                                )
                                second_rec["rejected_reason"] = (
                                    getattr(rev2, "rejected_reason", "")
                                    if rev2 is not None
                                    else "the reviser call failed"
                                )
                            revise_record["second_rewrite"] = second_rec
                else:
                    reason = (
                        getattr(rev, "rejected_reason", "") if rev
                        else "the reviser call failed"
                    )
                    revise_record = {
                        "status": "discarded" if reason else "unchanged",
                        "gate_issues": gate_issues,
                        "rejected_reason": reason,
                        "original_explanation": original_prose,
                    }
        else:
            claim_issues = gate_issues

    soft_warnings = (
        [] if use_placeholder else run_preflop_soft_validators(explanation, facts)
    )
    preflop_row = build_preflop_row(
        facts, explanation,
        pack=source.pack,
        difficulty=difficulty,
        number=number,
        stakes_bb_dollars=solve.bb_in_dollars,
        live_or_online=solve.live_or_online,
        game_format=solve.game_format,
        display_in_bb=display_in_bb,
        validation_status="flagged" if soft_warnings else "draft",
    )

    # Adapt the preflop-schema row onto the postflop schema (the two share
    # the 41-column prefix by NAME); postflop-only columns get the same
    # values the entry-derived leg uses; the hand's Context stays the
    # SOLVE's so every leg of one hand reads the same game header.
    from pipeline.postflop.action_history import build_context_line  # noqa: PLC0415

    row = {col: preflop_row.get(col, "") for col in POSTFLOP_ROW_COLUMNS}
    # Reuse the preflop writer's enriched Notes parts (chart / situation /
    # node reference), swapping only the provenance sentence below.
    _pre = parse_notes(preflop_row.get("Notes", ""))
    row.update({
        "No": str(number),
        "hand_id": hand_id,
        "sequence_index": str(sequence_index),
        "sequence_total": str(sequence_total),
        "Hand Stage": "Preflop",
        "Context": build_context_line(solve, display_in_bb=display_in_bb),
        "Cards on Table": "",
        "board_texture": "",
        "exploit_notes": preflop_row.get("exploit_notes", ""),
        # Keep the enriched chart/situation/node the preflop writer built
        # (so the Node: field still holds this leg's node reference), but swap
        # the provenance sentence to name the source pack. July 2026.
        "Notes": build_notes(
            "Auto-generated by poker-pipeline (full-hand preflop leg from "
            f"pack {source.pack_id}).",
            chart=_pre.chart,
            situation=_pre.situation,
            node_ref=_pre.node_ref,
        ),
        # The app's animation timeline: blinds + folds + the raises before
        # THIS decision of the line (step-aware: the 3-bet-pot opener's
        # second leg animates through the 3-bet before pausing).
        "animation_script": build_preflop_animation_script(
            solve, hero_position,
            step_index=source.step_at(hero_position, step_index).step_index,
        ),
    })
    record = {
        "node_id": facts.spot.node.node_id,
        "hero_combo": hero_combo,
        "street": "preflop",
        "hero_position": hero_position,
        "as_played": not (terminal_fold or terminal_raise),
        "terminal_fold": terminal_fold,
        "terminal_raise": terminal_raise,
        "preflop_leg_source": "pack",
        # Which decision of the preflop line this leg asks (index into the
        # solve's preflop_summary; the audit re-verifier joins on it to
        # rebuild the right leg when a hero acts twice, e.g. a 3-bet pot).
        "preflop_step_index": source.step_at(hero_position, step_index).step_index,
        "pack_id": source.pack_id,
        "hand_id": hand_id,
        "sequence_index": sequence_index,
        "sequence_total": sequence_total,
        "correct_answer": correct,
        "options": options,
        "archetype": facts.archetype,
        "difficulty": difficulty.score,
    }
    if soft_warnings:
        record["validator_warnings"] = soft_warnings
    if claim_issues:
        record["claim_check_issues"] = claim_issues
    if revise_record is not None:
        record["revise"] = revise_record
    # Unresolved audit findings mark the row flagged, like the batch drivers.
    unresolved = list(claim_issues)
    if revise_record is not None and revise_record.get("status") in (
        "discarded", "unchanged",
    ):
        unresolved += list(revise_record.get("gate_issues") or [])
    unresolved += list((revise_record or {}).get("final_audit_issues") or [])
    if unresolved:
        row["validation_status"] = "flagged"
    return row, record, None


def run_full_hand_cross_check(
    rows: list[dict], records: list[dict],
) -> dict[int, list[str]]:
    """The deterministic batch cross-check over a full-hand batch's rows.

    Reuses :mod:`pipeline.preflop.batch_cross_check` (zero-LLM,
    first-principles row verification: position claims, skills hygiene,
    domination direction, frequency sums, difficulty bands, the GTO
    second-best-by-EV rule) -- its checks key off row/record SHAPE and skip
    what a row does not carry, so the preflop pack legs get the full set
    and the postflop legs the applicable subset. Lives HERE because this
    module is the one sanctioned pipeline.preflop import seam. Fails open:
    an unavailable checker returns {} rather than blocking a batch."""
    try:
        from pipeline.preflop.batch_cross_check import (  # noqa: PLC0415
            cross_check_batch,
        )
    except Exception as exc:  # noqa: BLE001 - checker unavailable, fail open
        logger.warning("full-hand cross-check unavailable: %s", exc)
        return {}
    # POSTFLOP records carry solver_data as the RENDERED PROSE BLOCK (a
    # string); the preflop checker expects the structured dict (preflop
    # pack-leg records have it). Normalize so the string-shaped rows simply
    # skip the solver-data checks instead of crashing the whole pass.
    safe_records = [
        {
            **r,
            "solver_data": (
                r.get("solver_data")
                if isinstance(r.get("solver_data"), dict) else {}
            ),
        }
        for r in records
    ]
    return cross_check_batch(rows, safe_records)


__all__ = [
    "OPEN_SIZE_TOLERANCE_BB",
    "PackLegSource",
    "PackLineStep",
    "build_pack_preflop_leg_row",
    "compute_pack_leg_difficulty",
    "find_pack_leg_source",
    "run_full_hand_cross_check",
]
