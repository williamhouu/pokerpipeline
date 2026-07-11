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
from pipeline.postflop.format_writer import POSTFLOP_CSV_COLUMNS
from pipeline.postflop.preflop_entry import _open_to_bb
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
):
    """Sample + fully enrich the preflop facts for one of the hero's leg
    decisions, or None when the coherence gate fails (pack dominant != the
    as-played action at that step)."""
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
    spot = sample_spot(node, hand_class, combo=hero_combo)
    # Coherence gate: pack dominant must be the as-played FAMILY ("Raise" /
    # "Call"; an all-in is its own token, so a mostly-jam hand never passes
    # as a sized raise). Size-level matching isn't needed: these pack nodes
    # carry one sized raise each, and the line matcher already verified that
    # size against the solve's.
    if not spot.dominant_action.startswith(as_played):
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
    run_claim_checker: bool = False,
    revise_pass: bool = False,
    final_audit: bool = False,
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
                        )
                        if cc4 is not None:
                            revise_record["final_audit_issues"] = [
                                f"{i.claim} -- {i.problem}" for i in cc4.issues
                            ]
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
        "as_played": True,
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
