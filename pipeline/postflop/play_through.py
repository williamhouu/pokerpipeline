"""Full-hand "play-through" assembler: linked, ordered question sequences.

A play-through is one HAND a user answers street by street -- preflop entry,
then the flop / turn / river decisions on a single connected line -- advancing
regardless of right or wrong (linear, no branching). The app plays it by
GROUPING on ``hand_id`` and ORDERING on ``sequence_index`` (Option B; see
:data:`pipeline.postflop.format_writer.POSTFLOP_CSV_COLUMNS`).

This module turns a :class:`~pipeline.postflop.solve.PostflopSolve` into
:class:`PlayThroughHand` objects. The batch driver then renders each leg into a
CSV row carrying the shared ``hand_id`` and its 1-based ``sequence_index``.

How a hand is assembled
-----------------------
* **Connectivity is by (board, history), not node id.** Node A is an
  ancestor-or-self of node B on the same line exactly when A's board AND A's
  action history are prefixes of B's. This is adapter-agnostic (it works on the
  ``.db`` grammar and on the hand-built fixtures) and never confuses two lines
  that merely share a prefix -- the action history records what actually
  happened, so a divergence shows up as a non-prefix.
* **One hand = one hero + one combo down one line.** Seeded from the worthy
  spots (so every hand has at least one genuinely interesting decision), each
  hand anchors at the deepest seed and collects every node on that line where
  the hero acts with that same combo -- the hero's decision at each street --
  plus the preflop entry that started the hand. Deeper seeds (turn / river)
  yield longer, multi-street play-throughs.
* **Hero-first, villain-frame architected.** ``include_villain`` additionally
  emits the VILLAIN's own coherent line on the same runout as a SECOND hand
  (its own ``hand_id``), so one tree can produce 8+ ordered questions. Built
  hero-only by default.

Determinism: seeds are processed in a fixed (depth, node id, combo) order and
``hand_id`` is a content hash, so the same solve yields byte-identical hands
(the batch's byte-identical-CSV guarantee depends on it). ``variety_seed``
(July 2026) keeps that guarantee PER SEED while fixing the "every batch is
the same pocket fours" repetition: the alphabetical tie-break puts digit
combos (4d4c...) at the front of every batch, so small repeated batches
re-served the same few hands. A seeded shuffle WITHIN each depth group
re-orders the candidates (same seed = same batch; the batch meta records
the seed); the cross-depth deepest-first order is the dedup invariant and
is never disturbed.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from itertools import groupby

from pipeline.postflop.preflop_entry import (
    PreflopEntryFacts,
    build_preflop_entry_facts,
)
from pipeline.postflop.solve import PostflopNode, PostflopSolve
from pipeline.postflop.spot_sampler import PostflopSpot, sample_spot

_STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


@dataclass(frozen=True)
class HandLeg:
    """One question in a play-through, in order.

    Three kinds: ``postflop`` (``spot`` set), ``preflop_entry`` (``entry_facts``
    set; the SRP open-or-defend entry), and ``preflop_line`` (``step_index``
    set; one decision of a MULTI-RAISE preflop line, e.g. the open / the 3-bet
    / the call in a 3-bet pot -- built exclusively from a matching preflop
    range pack by the batch driver, because the postflop solve's entry weights
    cannot express a raise-or-call-or-fold decision). ``street`` and
    ``node_id`` (synthetic for preflop legs) are convenience copies.
    """

    kind: str  # "preflop_entry" | "preflop_line" | "postflop"
    street: str
    node_id: str
    spot: PostflopSpot | None = None
    entry_facts: PreflopEntryFacts | None = None
    # Index into solve.preflop_summary for kind="preflop_line".
    step_index: int | None = None


@dataclass(frozen=True)
class PlayThroughHand:
    """One hand's ordered legs, sharing a ``hand_id``.

    ``sequence_index`` for leg ``i`` is ``i + 1``; ``total`` is ``len(legs)``.
    """

    hand_id: str
    hero: str
    hero_combo: str
    frame: str  # "hero" | "villain" -- which player's decisions this hand asks
    legs: tuple[HandLeg, ...]

    @property
    def total(self) -> int:
        return len(self.legs)


def _is_prefix(short, long) -> bool:
    short, long = tuple(short), tuple(long)
    return len(short) <= len(long) and long[: len(short)] == short


def _on_line(ancestor: PostflopNode, descendant: PostflopNode) -> bool:
    """True when ``ancestor`` is an ancestor-or-self of ``descendant`` -- its
    board AND action history are both prefixes of the descendant's."""
    return _is_prefix(ancestor.board, descendant.board) and _is_prefix(
        ancestor.history, descendant.history
    )


def _depth(node: PostflopNode) -> tuple[int, int]:
    """Sort/seed depth: (board length, history length). Deeper = later street /
    more action, so a river facing-bet node outranks the flop root."""
    return (len(node.board), len(node.history))


def _variety_order(ordered_seeds: list, variety_seed: int | None) -> list:
    """Re-order depth-sorted seeds for batch-to-batch variety.

    ``None`` = the legacy fixed order (byte-identical batches). With a seed,
    shuffle WITHIN each depth group only. INVARIANT: the cross-depth
    deepest-first order must hold -- the duplicate-hand suppression in
    :func:`assemble_hands` relies on a combo's LONGEST line being processed
    before its shorter prefixes; a global shuffle would keep truncated hands
    and drop the full play-throughs. Within one depth that ordering carries
    no meaning (it was an alphabetical tie-break, which is exactly what made
    every small batch open with the same digit-first combos), so a seeded
    permutation there is safe and deterministic per seed.
    """
    if variety_seed is None:
        return ordered_seeds
    rng = random.Random(variety_seed)
    out: list = []
    for _, group in groupby(ordered_seeds, key=lambda s: _depth(s.node)):
        bucket = list(group)
        rng.shuffle(bucket)
        out.extend(bucket)
    return out


def balanced_hand_mix(hands: list, limit: int) -> list:
    """Round-robin ``hands`` across (hero seat, final street, hero-hand
    strength bucket) and keep the first ``limit``.

    The variety seed shuffles uniformly, so a SMALL batch can still come out
    clustered (three BTN river hands, all air). Bucketing by who plays the
    hand, how deep it runs, and how strong the hero's hand class is spreads
    a batch across genuinely different stories. Deterministic: buckets are
    visited in first-appearance order of the (already seeded-shuffled)
    input, and within a bucket the input order is preserved -- same input =
    same mix. Cheap by design (hand-class lookup only, no equity sims), so
    it can run over a several-times-oversized candidate pool BEFORE any
    facts are computed."""
    from pipeline.postflop.spot_selection import (  # noqa: PLC0415 - cycle guard
        spot_strength_bucket,
    )

    if limit <= 0 or len(hands) <= limit:
        return list(hands[:limit] if limit > 0 else [])
    buckets: dict[tuple, list] = {}
    for hand in hands:
        deepest = hand.legs[-1]  # legs end with the deepest postflop decision
        strength = (
            spot_strength_bucket(deepest.spot) if deepest.spot is not None else ""
        )
        key = (hand.hero, deepest.street, strength)
        buckets.setdefault(key, []).append(hand)
    out: list = []
    queues = list(buckets.values())
    while len(out) < limit and queues:
        queues = [q for q in queues if q]
        for q in queues:
            if len(out) >= limit:
                break
            out.append(q.pop(0))
    return out


def _solve_tag(solve: PostflopSolve) -> str:
    """A short, filesystem-safe tag for the hand_id prefix (cosmetic; the hash
    carries uniqueness)."""
    raw = solve.solve_id or solve.source_reference or "solve"
    return re.sub(r"[^A-Za-z0-9]+", "", raw)[:18] or "solve"


def _hand_id(solve: PostflopSolve, hero: str, combo: str, anchor: PostflopNode, frame: str) -> str:
    """A deterministic, globally-unique, readable id shared by a hand's legs.

    Readable prefix (solve / flop / runout / combo) for humans; an 8-char
    content hash over the full line (source ref + anchor node + combo + frame)
    guarantees global uniqueness with no random UUID, so the CSV stays
    byte-identical."""
    flopstem = "".join(solve.flop)
    runout = "".join(anchor.board[3:])  # turn + river cards, if any
    raw = f"{solve.source_reference}|{frame}|{hero}|{combo}|{anchor.node_id}|{tuple(anchor.board)}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    parts = [_solve_tag(solve), flopstem]
    if runout:
        parts.append(runout)
    parts += [combo, digest]
    return "_".join(parts)


def _line_nodes(
    solve: PostflopSolve, anchor: PostflopNode, actor: str, combo: str
) -> list[PostflopNode]:
    """Every node on ``anchor``'s line where ``actor`` decides with ``combo``,
    ordered shallow -> deep (the player's decision at each street)."""
    out = [
        n
        for n in solve.nodes.values()
        if n.actor == actor
        and combo in n.strategy
        and _on_line(n, anchor)
    ]
    out.sort(key=lambda n: (_depth(n), n.node_id))
    return out


def _villain_combo_for_line(
    solve: PostflopSolve, villain_nodes: list[PostflopNode]
) -> str | None:
    """A single villain combo present in EVERY villain-decision node on the line.

    For the villain-frame hand: one combo must flow through all of the villain's
    decisions. Among combos present in all of them, pick the highest total reach
    (a typical villain holding), tie-broken by sorted combo for determinism.
    ``None`` when no combo is common to every villain node."""
    if not villain_nodes:
        return None
    common: set[str] | None = None
    for n in villain_nodes:
        combos = set(n.strategy)
        common = combos if common is None else (common & combos)
    if not common:
        return None
    deepest = villain_nodes[-1]
    return max(
        sorted(common),
        key=lambda c: deepest.hero_range.get(c, 0.0),
    )


def _build_legs(
    solve: PostflopSolve, hero: str, combo: str, nodes: list[PostflopNode],
    *, include_preflop: bool,
) -> list[HandLeg]:
    """The ordered legs for one hero+combo: optional preflop leg(s), then each
    postflop decision node.

    SRP: one ``preflop_entry`` leg from the solve's entry weights. MULTI-RAISE
    (3-bet pot): one ``preflop_line`` leg per summary step the hero acted on
    (the opener gets TWO: the open and the call of the 3-bet) -- placeholders
    the batch driver fills from a matching preflop pack; the entry weights are
    NOT used (they can't express a raise-or-call-or-fold decision)."""
    legs: list[HandLeg] = []
    n_raises = sum(
        1 for st in solve.preflop_summary
        if st.verb in ("open", "raise", "3-bet", "4-bet", "5-bet")
    )
    if include_preflop and n_raises > 1:
        for i, step in enumerate(solve.preflop_summary):
            if step.position == hero:
                legs.append(HandLeg(
                    kind="preflop_line", street="preflop",
                    node_id=f"preflop_step:{i}", step_index=i,
                ))
    elif include_preflop and combo in solve.preflop_entry_ranges.get(hero, {}):
        entry = build_preflop_entry_facts(solve, hero, combo)
        legs.append(HandLeg(kind="preflop_entry", street="preflop", node_id="",
                            entry_facts=entry))
    for n in nodes:
        # FORCED-MOVE GUARD (July 2026): a node offering fewer than two
        # actions is not a decision -- solve trees truncate deep lines to
        # check-only, and emitting such a leg ships a one-option "question"
        # (and pays an LLM call to explain a forced move). Skip it: the next
        # leg's Question prose narrates the whole line, so the play-through
        # reads continuously. The hand's SEED decision always survives (the
        # worthiness window requires a mixed strategy = 2+ actions).
        if len(n.actions) < 2:
            continue
        legs.append(HandLeg(kind="postflop", street=n.street, node_id=n.node_id,
                            spot=sample_spot(n, combo)))
    return legs


def assemble_hands(
    solve: PostflopSolve,
    *,
    seeds: list[PostflopSpot],
    heroes: tuple[str, ...] = (),
    max_hands: int | None = None,
    include_preflop: bool = True,
    include_villain: bool = False,
    variety_seed: int | None = None,
) -> list[PlayThroughHand]:
    """Assemble play-through hands from ``solve``, seeded by worthy ``seeds``.

    Args:
        solve: the (ideally line-closed) solve -- each seed's decision-node
            ancestors must be present for the connected line to resolve. Load a
            ``.db`` with ``include_ancestors=True``.
        seeds: worthy postflop spots that anchor the hands (so each hand has at
            least one interesting decision). Processed deepest-first; a seed
            already absorbed into a deeper hand's line is skipped.
        heroes: restrict to these hero seats (empty = both).
        max_hands: stop after this many hands (``None`` = all).
        include_preflop: prepend the preflop-entry leg (default True).
        include_villain: also emit the villain's coherent line on the same
            runout as a separate hand (the "flip the frame" path; architected,
            off by default).
        variety_seed: ``None`` = the legacy fixed order (identical hands every
            batch); an int = a per-seed deterministic shuffle within each depth
            group, so different seeds pick different hands (see
            :func:`_variety_order`).

    Returns:
        A deterministic (per ``variety_seed``) list of :class:`PlayThroughHand`.
    """
    seats = tuple(heroes) or tuple(solve.positions)
    ordered_seeds = sorted(
        (s for s in seeds if s.node.actor in seats),
        key=lambda s: (-_depth(s.node)[0], -_depth(s.node)[1], s.node.node_id, s.hero_combo),
    )
    ordered_seeds = _variety_order(ordered_seeds, variety_seed)

    # consumed holds every (node_id, combo) already absorbed into a built hand.
    # A seed is SKIPPED when ANY node on its line is already consumed -- not just
    # its own node. This is what stops near-duplicate hands: many deep runouts of
    # the SAME combo share the same shallow ancestors (the preflop entry + the
    # flop decision), so without this you get one hand per runout, all identical
    # until the river. Processing deepest-first, the longest line for a (hero,
    # combo) wins and every shorter/sibling line through a shared node is dropped
    # -> one play-through per (hero, combo) [+ per genuinely distinct early line].
    consumed: set[tuple[str, str]] = set()
    hands: list[PlayThroughHand] = []
    for seed in ordered_seeds:
        hero = seed.node.actor
        combo = seed.hero_combo
        line = _line_nodes(solve, seed.node, hero, combo)
        if not line:
            continue
        if any((n.node_id, combo) in consumed for n in line):
            continue  # this line shares ancestry with an already-built hand
        for n in line:
            consumed.add((n.node_id, combo))
        legs = _build_legs(solve, hero, combo, line, include_preflop=include_preflop)
        anchor = line[-1]
        hands.append(PlayThroughHand(
            hand_id=_hand_id(solve, hero, combo, anchor, "hero"),
            hero=hero, hero_combo=combo, frame="hero", legs=tuple(legs),
        ))

        if include_villain:
            villain = seed.node.villain
            v_nodes = [
                n for n in solve.nodes.values()
                if n.actor == villain and _on_line(n, anchor)
            ]
            v_nodes.sort(key=lambda n: (_depth(n), n.node_id))
            v_combo = _villain_combo_for_line(solve, v_nodes)
            # Same overlap-dedup for the villain frame (don't repeat a villain
            # line already emitted from another hero hand on the same runout).
            if v_combo is not None and not any(
                (n.node_id, v_combo) in consumed for n in v_nodes
            ):
                for n in v_nodes:
                    consumed.add((n.node_id, v_combo))
                v_legs = _build_legs(
                    solve, villain, v_combo, v_nodes, include_preflop=include_preflop
                )
                if v_legs:
                    hands.append(PlayThroughHand(
                        hand_id=_hand_id(solve, villain, v_combo, anchor, "villain"),
                        hero=villain, hero_combo=v_combo, frame="villain",
                        legs=tuple(v_legs),
                    ))

        if max_hands is not None and len(hands) >= max_hands:
            break
    return hands


__all__ = ["HandLeg", "PlayThroughHand", "assemble_hands"]
