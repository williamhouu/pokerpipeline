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
the entry-derived leg. Three gates:

1. **Geometry** -- same table size, same effective stack, and the pack's
   open size at this line within :data:`OPEN_SIZE_TOLERANCE_BB` of the
   solve's (derived from the flop pot). A mismatched open would make the
   preflop leg's pot math contradict the postflop legs of the SAME hand.
2. **Line** -- the pack contains the exact nodes: the opener's
   first-in decision and the defender's facing-the-open decision, with
   every earlier seat folding (this SRP line is what the postflop solves
   model).
3. **Coherence** -- the pack's dominant action for the hero's hand must
   match what the hand actually did (the opener opened, the defender
   called). A play-through advances along the as-played line, and the
   established design keeps the entry leg's correct answer consistent
   with it; when the pack says the hand mostly 3-bets, this hand's story
   would contradict the answer, so that hand keeps the entry-derived leg.

Pack preference among geometry matches: ``*_IMPROVED`` packs first, then
lexicographic -- deterministic, so batches stay byte-identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pipeline.postflop.format_writer import POSTFLOP_CSV_COLUMNS
from pipeline.postflop.preflop_entry import _open_to_bb
from pipeline.postflop.solve import PostflopSolve

logger = logging.getLogger(__name__)

# The pack's rendered open size must sit within this of the solve's derived
# open size (packs quantize sizes to a 0.5bb display grid, so allow just over
# half a step).
OPEN_SIZE_TOLERANCE_BB = 0.26


@dataclass(frozen=True)
class PackLegSource:
    """A verified pack + the two preflop nodes matching the solve's line."""

    pack: Any                 # PreflopPack (typed loosely: lazy import)
    pack_id: str
    opener_node: Any          # the IP seat's first-in decision node
    defender_node: Any        # the OOP seat's facing-the-open decision node
    open_size_bb: float


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
        from pipeline.preflop.action_history import (  # noqa: PLC0415
            resolve_preflop_history,
        )
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

    solve_open = _open_to_bb(solve)
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
        opener = defender = None
        for node in nodes:
            hist = node.history_before
            if node.actor == solve.ip_position and all(
                a.action_type is PreflopActionType.FOLD for a in hist
            ):
                opener = node
            elif node.actor == solve.oop_position:
                raises = [
                    a for a in hist
                    if a.action_type in (PreflopActionType.RAISE,
                                         PreflopActionType.ALL_IN)
                ]
                others_folded = all(
                    a.action_type is PreflopActionType.FOLD
                    for a in hist
                    if a not in raises
                )
                if (len(raises) == 1
                        and raises[0].position == solve.ip_position
                        and raises[0].action_type is PreflopActionType.RAISE
                        and others_folded):
                    defender = node
        if opener is None or defender is None:
            continue
        try:
            resolved = resolve_preflop_history(defender.history_before, pack)
            pack_open = next(
                (s for s in reversed(resolved.sizes_bb) if s is not None),
                None,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pack %s size walk failed: %s", pack.pack_id, exc)
            continue
        if pack_open is None or abs(pack_open - solve_open) > OPEN_SIZE_TOLERANCE_BB:
            logger.info(
                "pack %s line found but open size %s != solve %s; skipping",
                pack.pack_id, pack_open, solve_open,
            )
            continue
        logger.info(
            "pack legs: %s matches %s (open %.1fbb)",
            pack.pack_id, solve.solve_id, pack_open,
        )
        return PackLegSource(
            pack=pack, pack_id=pack.pack_id, opener_node=opener,
            defender_node=defender, open_size_bb=pack_open,
        )
    return None


def compute_pack_leg_difficulty(
    source: PackLegSource, hero_position: str, hero_combo: str,
    solve: PostflopSolve, *,
    equity_runouts: int,
    trap_difficulty: bool = False,
    razor_difficulty: bool = False,
) -> int | None:
    """The pack leg's 4-axis difficulty WITHOUT generating anything (used by
    the hand-difficulty pre-pass). None when the leg wouldn't use the pack
    (coherence gate) -- the caller falls back to the entry difficulty."""
    built = _build_pack_facts(
        source, hero_position, hero_combo, solve,
        equity_runouts=equity_runouts,
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
):
    """Sample + fully enrich the preflop facts for the hero's leg, or None
    when the coherence gate fails (pack dominant != as-played action)."""
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

    node = (
        source.opener_node
        if hero_position == solve.ip_position
        else source.defender_node
    )
    as_played = "Raise" if hero_position == solve.ip_position else "Call"
    hand_class = combo_str_to_hand_class(hero_combo)
    spot = sample_spot(node, hand_class, combo=hero_combo)
    if not spot.dominant_action.startswith(as_played):
        return None  # coherence gate: pack disagrees with the as-played line
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
) -> tuple[dict[str, str] | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Build the pack-backed preflop leg row (POSTFLOP schema).

    ``system_prompt`` overrides the PREFLOP system prompt for the
    explanation call (this leg is written by the preflop pipeline's
    generator, not the postflop or preflop-entry one). None = the active
    preflop prompt (override file or built-in default).

    Returns ``(row, meta_record, failure)``. ``(None, None, None)`` means
    "pack not applicable for this hand" (coherence gate) and the caller
    falls back to the entry-derived leg. A generation failure returns a
    failure record like the other leg builders.
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
            equity_runouts=equity_runouts,
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
            _model: str, in_t: int, out_t: int, _cache_c: int, _cache_r: int
        ) -> None:
            usage_cb(SimpleNamespace(input_tokens=in_t, output_tokens=out_t))

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

    row = {col: preflop_row.get(col, "") for col in POSTFLOP_CSV_COLUMNS}
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
        "Notes": (
            "Auto-generated by poker-pipeline (full-hand preflop leg from "
            f"pack {source.pack_id})."
        ),
    })
    record = {
        "node_id": facts.spot.node.node_id,
        "hero_combo": hero_combo,
        "street": "preflop",
        "hero_position": hero_position,
        "as_played": True,
        "preflop_leg_source": "pack",
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
    return row, record, None


__all__ = [
    "OPEN_SIZE_TOLERANCE_BB",
    "PackLegSource",
    "build_pack_preflop_leg_row",
    "compute_pack_leg_difficulty",
    "find_pack_leg_source",
]
