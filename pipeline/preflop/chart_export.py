"""SolverPack chart export -- the app team's native preflop-chart JSON.

Emits the NLH MTT 8-max preflop packs (``monker_mtt8_*bb``) AND the NLHE
CASH 6-max packs (the PioViewer-format Ryan 100bb pack + the Monker
20/30bb short-stack packs) in the Runout chart engine's ``SolverPack``
shape, ONE file per pack, per the app developer's spec
(``mtt-chart-export-spec.md``, "Option B"). The exported tree powers the
interactive open -> 3-bet -> 4-bet drill, so the one hard requirement is
TREE LINKS: a child node's map key is ``parentKey + "." + act.c``.

For dot-token grammars (``monker_nlhe``) that falls straight out of the
file naming (a node's key is the dot-joined history tokens of its
range-file stems; the root is the empty string ``""``). The ``ryan_pack``
grammar names files as underscore-joined ``<Position>_<Action>`` pairs
instead (``UTG_60%_HJ_Call_..._BTN_Fold``); its ``c`` codes are the bare
action tokens (``Fold``/``Call``/``AI``/``60%`` -- unique among siblings,
one actor per node) and its path key is the dot-joined history ACTION
tokens, dropping the redundant position tokens (seat rotation is
deterministic given the action sequence, exactly the property the Monker
naming relies on). The ``gto_preflop_8max`` grammar (the 8-max LIVE cash
packs) names files as underscore-joined ``<Seat>-<Code>`` tokens
(``UTG-F_UTG+1-R3_..._SB-C``); its ``c`` codes are the bare action CODES
(``F``/``C``/``A``/``R<bb>``), with any ``.`` in a fractional raise size
swapped for ``-`` (``R38.5`` -> ``R38-5``) because the spec's tree-link
scheme reserves ``.`` as the path separator; seats are dropped from path
keys for the same rotation-determinism reason as ryan_pack.

Pure functions only -- no Streamlit, no I/O beyond reading the pack's range
files -- so everything is browserless-testable (fix-durability rule). The
thin CLI lives in ``scripts/export_chart_packs.py``.

Format contract highlights (see the spec + the export README for the rest):

* ``acts`` order is fold, call, raise(s) ascending by ``to``, allin(s) last.
  The 169-cell ``pure`` string and every ``mix`` row index into this array,
  so the ordering is part of the data contract, not cosmetics.
* ``pure`` is EXACTLY 169 chars over the spec's grid (RANKS ``A..2``,
  ``idx = r*13 + c``; pair on the diagonal, suited above, offsuit below).
  Hands absent from every range file at a node are pure fold.
* ``mix`` rows are integer percentages, one per act, summing to EXACTLY
  100 via largest-remainder (mirrors the house allocation in
  ``pipeline/plo/options.py:integer_percentages`` -- mirrored rather than
  imported so ``pipeline.preflop`` takes no ``pipeline.plo`` dependency).
* All sizes are decoded through the shared resolution walk
  (``resolve_preflop_history`` / ``format_writer._normalize_action``), never
  hardcoded; the 1bb BB ante is already inside ``pot`` (never re-added).
"""

from __future__ import annotations

import json
import re
import string
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from pipeline.action_history import preflop_order
from pipeline.preflop.action_history import resolve_preflop_history

# SEAM NOTE: _normalize_action is format_writer's private (label, to_bb)
# mapper for a node option. Importing it here (an in-repo consumer) keeps the
# chart's raise/all-in sizes single-sourced with the CSV `ranges` column and
# the Question prose -- they can never disagree. If format_writer ever
# renames it, the cross-check test in tests/test_chart_export.py fails first.
from pipeline.preflop.format_writer import _normalize_action
from pipeline.preflop.grammars.types import PreflopActionType
from pipeline.preflop.node_enumerator import (
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.pack import PreflopPack
from pipeline.preflop_ranges import parse_range_file

# The seven MTT 8-max depths this exporter ships (informational default for
# the CLI; any monker_nlhe pack with dot-token stems exports fine).
CHART_EXPORT_PACK_IDS: tuple[str, ...] = tuple(
    f"monker_mtt8_{d}bb" for d in (10, 15, 20, 30, 50, 75, 300)
)

# The NLHE CASH 6-max packs the CLI's --cash flag exports (the Ryan
# PioViewer 100bb pack + the Monker ladder). INVARIANT: exactly ONE pack
# per depth -- monker_nlhe_6max_100bb is deliberately ABSENT because the
# Ryan pack owns the nlh-6max-100bb chart id (it is the solve production
# questions generate from, so app charts always match question content).
CASH_CHART_EXPORT_PACK_IDS: tuple[str, ...] = (
    "ryan_preflop_tree_6max_100bb",
    "monker_nlhe_6max_20bb",
    "monker_nlhe_6max_30bb",
    "monker_nlhe_6max_40bb",
    "monker_nlhe_6max_50bb",
    "monker_nlhe_6max_70bb",
    "monker_nlhe_6max_150bb",
    "monker_nlhe_6max_200bb",
)

# The 8-max LIVE cash packs the CLI's --cash8 flag exports (the
# gto_preflop_8max grammar). INVARIANT: the IMPROVED variants ONLY -- they
# are the packs production questions generate from, so the app charts must
# match question content; the non-IMPROVED originals are deliberately
# absent (same one-pack-per-depth rule as the 6-max list above).
CASH8_CHART_EXPORT_PACK_IDS: tuple[str, ...] = (
    "preflop_8max_100bb_IMPROVED",
    "preflop_8max_200bb_IMPROVED",
    "preflop_8max_300bb_IMPROVED",
)

# Spec note 1: the app grid's seat-label enum. Every exported seat (order[]
# and node pos) MUST be from this set -- fail loudly, never invent a mapping.
_APP_SEAT_ENUM = frozenset(
    ("UTG", "UTG+1", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB")
)

# Rake caps live in the pack DESCRIPTIONS (e.g. "rake 4%/0.3bb cap"), same
# house pattern as pipeline/plo/pack.py:rake_pct_from_note -- the percent
# itself is the structured rake_pct field.
_RAKE_CAP_RE = re.compile(r"rake\s*\d+(?:\.\d+)?%\s*/\s*(\d+(?:\.\d+)?)\s*bb")

# Spec grid rank order (index 0..12). idx = r*13 + c; r==c pair, c>r suited,
# c<r offsuit. NOT the pack's canonical file order -- this is the app grid.
GRID_RANKS = "AKQJT98765432"

# acts[] display/index order: fold, call, raise(s), allin(s). Part of the
# data contract -- `pure`/`mix` index into the sorted array.
_KIND_ORDER = {"fold": 0, "call": 1, "raise": 2, "allin": 3}

# base-36 digits for the `pure` string ("1" = acts[1], "a" = acts[10]).
_BASE36 = string.digits + string.ascii_lowercase


def grid_hand(idx: int) -> str:
    """The spec grid's hand label at cell ``idx`` (0..168).

    ``idx = r*13 + c`` over :data:`GRID_RANKS`: diagonal = pair ("AA" at 0),
    above the diagonal (c > r) = suited ("AKs" at 1), below (c < r) =
    offsuit ("AKo" at 13). Labels match the pack files' 169 class names
    (high rank first + s/o), so range weights key directly.
    """
    if not 0 <= idx <= 168:  # noqa: PLR2004
        raise ValueError(f"grid index out of range: {idx}")
    r, c = divmod(idx, 13)
    if r == c:
        return GRID_RANKS[r] * 2
    if c > r:
        return GRID_RANKS[r] + GRID_RANKS[c] + "s"
    return GRID_RANKS[c] + GRID_RANKS[r] + "o"


def grid_index(hand: str) -> int:
    """Inverse of :func:`grid_hand` -- the cell index of a hand label."""
    hi, lo = GRID_RANKS.index(hand[0]), GRID_RANKS.index(hand[1])
    if hand[0] == hand[1]:
        return hi * 13 + hi
    if hand.endswith("s"):
        return hi * 13 + lo
    return lo * 13 + hi


def largest_remainder_percentages(freqs: Sequence[float]) -> list[int]:
    """Integer percentages summing to EXACTLY 100, by largest remainder.

    Mirrors the house allocation (``pipeline/plo/options.py:
    integer_percentages``): floor everything, then hand the deficit to the
    largest fractional remainders, ties going to the higher-frequency entry.
    Input need not be normalised -- entries are scaled by their sum.
    """
    total = sum(freqs)
    if total <= 0:
        raise ValueError("largest_remainder_percentages: all-zero input")
    scaled = [f / total * 100.0 for f in freqs]
    floors = [int(s) for s in scaled]
    remainders = [s - fl for s, fl in zip(scaled, floors)]
    deficit = 100 - sum(floors)
    order = sorted(
        range(len(scaled)), key=lambda i: (-remainders[i], -scaled[i], i)
    )
    for i in order[: max(deficit, 0)]:
        floors[i] += 1
    return floors


def _num(x: float) -> int | float:
    """Compact JSON number: integral floats as ints, else 2dp rounding."""
    r = round(float(x), 2)
    return int(r) if r == int(r) else r


def rake_label(pack: PreflopPack) -> str:
    """The SolverPack ``rake`` display label, built from pack data.

    ``rake_pct`` gives the percent; the bb cap (when the pack has one) is
    parsed from the registered description's ``rake <pct>%/<cap>bb``
    convention. ``"none"`` when the pack carries no rake (the MTT packs).
    Never hardcoded per pack -- register the data, the label follows.
    """
    if not pack.rake_pct:
        return "none"
    label = f"{_num(pack.rake_pct * 100)}%"
    m = _RAKE_CAP_RE.search(pack.description or "")
    if m:
        label += f" cap {_num(float(m.group(1)))}bb"
    return label


def _ryan_stem_action_tokens(stem: str) -> list[str]:
    """A ryan_pack file stem's ACTION tokens (every 2nd underscore token).

    Stems are ``<Position>_<Action>`` pairs; positions are redundant with
    the action sequence (deterministic rotation), so path keys / c codes
    use the action tokens alone -- the exact analog of the Monker naming.
    """
    tokens = stem.split("_")
    if len(tokens) < 2 or len(tokens) % 2 != 0:
        raise AssertionError(f"ryan_pack stem has odd token count: {stem!r}")
    return tokens[1::2]


def _gto8_stem_action_codes(stem: str) -> list[str]:
    """A gto_preflop_8max stem's action CODES, dot-sanitized for c use.

    Stems are underscore-joined ``<Seat>-<Code>`` tokens; the seat is
    redundant with the action sequence (deterministic rotation, the same
    property ryan_pack relies on), so path keys / ``c`` codes use the bare
    codes. Fractional raise sizes carry a ``.`` (``R38.5``) which the
    spec's tree-link scheme reserves as the path separator, so it becomes
    ``-`` (``R38-5``) -- injective over this grammar's code alphabet
    (codes never contain ``-`` natively), hence still sibling-unique.
    """
    codes = []
    for token in stem.split("_"):
        seat, _, code = token.partition("-")
        if not code:
            raise AssertionError(
                f"gto_preflop_8max stem token {token!r} has no action code"
            )
        codes.append(code.replace(".", "-"))
    return codes


def _act_c_code(stem: str, grammar_name: str | None) -> str:
    """The act's ``c`` code from its range-file stem, per grammar.

    INVARIANT (spec tree-link contract): a ``c`` code must never contain a
    ``.`` -- the validator recovers the parent key via ``rpartition(".")``,
    so a dotted code would break connectivity. Dot-token grammars can't
    produce one by construction; ryan_pack percent tokens are asserted.
    """
    if grammar_name == "ryan_pack":
        code = _ryan_stem_action_tokens(stem)[-1]
        if "." in code:
            raise AssertionError(
                f"ryan_pack act code {code!r} contains '.' -- the tree-link "
                "key scheme cannot represent it"
            )
        return code
    if grammar_name == "gto_preflop_8max":
        # Sanitizer guarantees no "." (fractional sizes become R38-5).
        return _gto8_stem_action_codes(stem)[-1]
    return stem.split(".")[-1]


def node_path_key(
    node: PreflopDecisionNode, grammar_name: str | None = None
) -> str:
    """The node's SolverPack map key: its dot-joined history action tokens.

    Derived from the option range-file stems (stem = history tokens + the
    action's own token; ryan_pack stems interleave redundant position
    tokens, dropped here). INVARIANT (spec tree-link contract): every
    option of a node must share the same stem prefix, and a child node's
    key is then ``parentKey + "." + act.c`` by construction of the file
    naming. Root (first-in, empty history) is the empty string ``""``.
    """
    if grammar_name == "ryan_pack":
        prefixes = {
            ".".join(_ryan_stem_action_tokens(opt.range_file.path.stem)[:-1])
            for opt in node.actions
        }
    elif grammar_name == "gto_preflop_8max":
        prefixes = {
            ".".join(_gto8_stem_action_codes(opt.range_file.path.stem)[:-1])
            for opt in node.actions
        }
    else:
        prefixes = {
            ".".join(opt.range_file.path.stem.split(".")[:-1])
            for opt in node.actions
        }
    if len(prefixes) != 1:
        raise AssertionError(
            f"node {node.node_id}: option file stems disagree on the history "
            f"prefix ({sorted(prefixes)}) -- the pathKey/tree-link contract "
            "is broken"
        )
    key = prefixes.pop()
    if len(node.history_before) != (len(key.split(".")) if key else 0):
        raise AssertionError(
            f"node {node.node_id}: history length != prefix token count "
            f"for key {key!r}"
        )
    return key


def export_node(
    node: PreflopDecisionNode, pack: PreflopPack
) -> tuple[str, dict[str, Any]]:
    """One decision node -> its ``(pathKey, SolverPack node object)``.

    Sizes come from the shared resolution walk (``resolve_preflop_history``
    + ``_normalize_action``), so ``pot`` includes the BB ante and every
    raise-TO matches what the question prose / `ranges` column would say.
    """
    state = resolve_preflop_history(node.history_before, pack)

    # --- acts: normalise, size, and order fold/call/raise-asc/allin ------
    entries: list[tuple[int, float, str, dict[str, Any], Path]] = []
    for opt in node.actions:
        c_code = _act_c_code(opt.range_file.path.stem, pack.grammar_name)
        kind, to_bb = _normalize_action(opt, node.actor, state, pack)
        if kind == "fold":
            to: float = 0.0
        elif kind == "call":
            # Call-TO convention: the total the caller matches (the current
            # bet), NOT the increment -- documented in the export README.
            to = state.high_bet_bb
        else:
            to = float(to_bb or 0.0)
        if kind == "allin" and to != float(pack.stack_depth_bb):
            # resolve_preflop_history commits the full effective stack for
            # every jam (BB included -- the ante is dead money on top, not
            # part of the stack); keep the export consistent with it.
            raise AssertionError(
                f"node {node.node_id}: all-in to {to} != effective stack "
                f"{pack.stack_depth_bb}"
            )
        entries.append(
            (
                _KIND_ORDER[kind],
                to,
                c_code,
                {"c": c_code, "k": kind, "to": _num(to)},
                opt.range_file.path,
            )
        )
    entries.sort(key=lambda e: (e[0], e[1], e[2]))
    acts = [e[3] for e in entries]
    if len(acts) > len(_BASE36):
        raise AssertionError(f"node {node.node_id}: too many acts to encode")

    # --- per-hand strategy -> pure / mix ---------------------------------
    # Range weights are the JOINT reach-and-take probability; the chart
    # shows the mix CONDITIONAL on reaching, so each hand is normalised by
    # its own total. Mixed-ness is judged on the EXACT parsed weights.
    weights = [parse_range_file(e[4]) for e in entries]
    fold_index = next(
        (i for i, a in enumerate(acts) if a["k"] == "fold"), None
    )
    pure_chars: list[str] = []
    mix: dict[str, list[int]] = {}
    for idx in range(169):
        hand = grid_hand(idx)
        freqs = [w.get(hand, 0.0) for w in weights]
        live = [i for i, f in enumerate(freqs) if f > 0.0]
        if not live:
            # Absent from every file = the hand never reaches this node =
            # it folded out earlier -> pure fold (spec: every cell resolves).
            # Some vendor nodes ship NO fold file at all (an empty fold
            # range is simply not exported -- e.g. the 40bb BTN, having
            # flatted an open, never folds to the SB squeeze). There the
            # menu has no fold act, and an absent hand is REACH-ZERO: it
            # cannot arrive here holding this hand, so the cell is
            # display-only. Assign the LEAST-AGGRESSIVE act on the menu
            # (acts are ordered fold < call < raises < all-in) so the
            # encoding stays total without inventing a fold the solver
            # does not offer.
            pure_chars.append(
                _BASE36[fold_index if fold_index is not None else 0]
            )
        elif len(live) == 1:
            pure_chars.append(_BASE36[live[0]])
        else:
            pure_chars.append(".")
            mix[str(idx)] = largest_remainder_percentages(freqs)
    pure = "".join(pure_chars)
    if len(pure) != 169:  # noqa: PLR2004
        raise AssertionError(f"node {node.node_id}: pure length {len(pure)}")

    return node_path_key(node, pack.grammar_name), {
        "pos": node.actor,
        "bl": state.raise_level,
        "pot": _num(state.pot_bb),
        "bet": _num(state.high_bet_bb),
        "acts": acts,
        "pure": pure,
        "mix": mix,
    }


def contracted_key(
    stem_key: str,
    node: PreflopDecisionNode,
    existing_stem_keys: frozenset[str] | set[str],
) -> str:
    """The node's FINAL map key: its stem key with forced folds contracted.

    Some packs (the 8-max cash conversions) materialise a forced fold as
    HISTORY only: e.g. after the SB 3-bets a BTN open, the BB has no
    decision node in the pack (no ``..._BB-C``/``BB-R`` files exist) and
    the BB's fold appears solely inside the opener-response filenames. The
    stem-derived key would then reference a parent node that does not
    exist, breaking the spec's tree-link walk. This pass SPLICES such an
    edge: a FOLD token whose would-be decision node is absent from the
    pack is dropped from the key, so the walker steps straight from the
    3-bet act to the opener's response (the app shows the responder's
    ``pos``; the folded blind stays in ``pot`` via the shared resolution
    walk). Only fold edges may be contracted -- a missing parent on any
    other action is a real hole and is left in place for
    :func:`validate_solver_pack` to reject loudly. Packs whose every
    decision is materialised (the MTT + 6-max exports) are untouched:
    every prefix exists, so the key comes back verbatim.
    """
    if not stem_key:
        return ""
    tokens = stem_key.split(".")
    history = node.history_before
    if len(tokens) != len(history):
        raise AssertionError(
            f"node {node.node_id}: key tokens != history length"
        )
    kept: list[str] = []
    for i, (token, action) in enumerate(zip(tokens, history)):
        parent = ".".join(tokens[:i])
        if (
            action.action_type is PreflopActionType.FOLD
            and parent
            and parent not in existing_stem_keys
        ):
            continue
        kept.append(token)
    return ".".join(kept)


def pack_export_id(pack: PreflopPack) -> str:
    """The SolverPack ``id``, e.g. ``"nlh-8max-15bb"`` / ``"nlh-6max-100bb"``.

    INVARIANT (chart-id namespace): exactly ONE chart id per
    (table, depth, format). The MTT 8-max exports shipped first and own
    the bare ``nlh-8max-<depth>bb`` names, so 8-max CASH packs carry a
    ``-cash-`` infix (``nlh-8max-cash-100bb``) -- without it the cash
    100/200/300bb files would silently overwrite the MTT depths of the
    same name. 6-max stays bare (no 6-max MTT exports exist, and the
    bare 6-max ids already shipped to the app team).
    """
    infix = (
        "cash-"
        if pack.table_size == 8 and pack.game_format != "tournament"  # noqa: PLR2004
        else ""
    )
    return f"nlh-{pack.table_size}max-{infix}{int(pack.stack_depth_bb)}bb"


def build_solver_pack(
    pack: PreflopPack,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Export one full pack to the SolverPack dict (Option B, one per depth).

    Enumerates every decision node, exports each, validates the result
    loudly (:func:`validate_solver_pack`), and returns the finished object
    ready for compact JSON serialisation.
    """
    order = list(preflop_order(pack.table_size))
    if pack.table_size == 8 and order != [  # noqa: PLR2004
        "UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB",
    ]:
        # Spec note 1: the 8-max enum has NO UTG+2 -- fail loudly rather
        # than ship seat labels the app grid won't match.
        raise AssertionError(f"unexpected 8-max seat order: {order}")
    if pack.table_size == 6 and order != [  # noqa: PLR2004
        "UTG", "HJ", "CO", "BTN", "SB", "BB",
    ]:
        # 6-max rotation, verified empirically against the packs' parsers
        # (first-in actors at increasing fold depths).
        raise AssertionError(f"unexpected 6-max seat order: {order}")
    bad_seats = [s for s in order if s not in _APP_SEAT_ENUM]
    if bad_seats:
        # Spec note 1: NEVER invent a seat mapping -- stop and report.
        raise AssertionError(
            f"seats {bad_seats} not in the app's allowed enum "
            f"{sorted(_APP_SEAT_ENUM)}"
        )
    if pack.game_format != "tournament" and float(pack.ante_bb) != 0.0:
        # Cash packs carry no ante anywhere (header AND every node's pot,
        # which resolve_preflop_history builds from the same ante_bb field).
        raise AssertionError(
            f"cash pack {pack.pack_id} has ante_bb={pack.ante_bb}"
        )

    all_nodes = enumerate_nodes([pack])
    exported: list[tuple[PreflopDecisionNode, str, dict[str, Any]]] = []
    stem_keys: set[str] = set()
    for i, node in enumerate(all_nodes):
        key, obj = export_node(node, pack)
        if key in stem_keys:
            raise AssertionError(f"duplicate node key {key!r}")
        stem_keys.add(key)
        exported.append((node, key, obj))
        if progress is not None:
            progress(i + 1, len(all_nodes))

    # Forced-fold splice (no-op for fully-materialised packs) -- see
    # contracted_key. Runs after the full stem-key set is known.
    nodes_map: dict[str, dict[str, Any]] = {}
    for node, key, obj in exported:
        final_key = contracted_key(key, node, stem_keys)
        if final_key in nodes_map:
            raise AssertionError(f"duplicate node key {final_key!r}")
        nodes_map[final_key] = obj

    # Deterministic file layout: root first, then keys sorted.
    ordered = dict(
        sorted(nodes_map.items(), key=lambda kv: (kv[0] != "", kv[0]))
    )
    solver_pack: dict[str, Any] = {
        "id": pack_export_id(pack),
        "format": "MTT" if pack.game_format == "tournament" else "Cash",
        "table": f"{pack.table_size}-Max",
        "stack": int(pack.stack_depth_bb),
        "rake": rake_label(pack),
        "ante": _num(pack.ante_bb),
        "blinds": [_num(pack.sb_to_bb_ratio), 1],
        "order": order,
        "decisionNodes": len(ordered),
        "nodes": ordered,
    }
    validate_solver_pack(solver_pack)
    return solver_pack


def validate_solver_pack(solver_pack: dict[str, Any]) -> None:
    """Loud structural self-checks on a finished SolverPack object.

    Raises AssertionError on the first violation: root presence/actor,
    169-char ``pure`` strings, ``mix`` rows (length == len(acts), sum ==
    exactly 100, cell marked "." in ``pure``), act ordering, and full tree
    connectivity (every non-root key = an existing parent key + "." + one
    of that parent's act ``c`` codes).
    """
    nodes = solver_pack["nodes"]
    order = solver_pack["order"]
    if "" not in nodes:
        raise AssertionError("root node (empty-string key) missing")
    if nodes[""]["pos"] != order[0]:
        raise AssertionError(
            f"root pos {nodes['']['pos']!r} != first seat {order[0]!r}"
        )
    if solver_pack["decisionNodes"] != len(nodes):
        raise AssertionError("decisionNodes != len(nodes)")
    for key, node in nodes.items():
        acts = node["acts"]
        if node["pos"] not in order:
            raise AssertionError(f"{key!r}: pos {node['pos']!r} not in order")
        kinds = [_KIND_ORDER[a["k"]] for a in acts]
        if kinds != sorted(kinds):
            raise AssertionError(f"{key!r}: acts out of kind order")
        raise_tos = [a["to"] for a in acts if a["k"] == "raise"]
        if raise_tos != sorted(raise_tos):
            raise AssertionError(f"{key!r}: raises not ascending by to")
        pure = node["pure"]
        if len(pure) != 169:  # noqa: PLR2004
            raise AssertionError(f"{key!r}: pure length {len(pure)}")
        for ch in set(pure) - {"."}:
            if _BASE36.index(ch) >= len(acts):
                raise AssertionError(f"{key!r}: pure char {ch!r} out of range")
        for cell, row in node["mix"].items():
            idx = int(cell)
            if not 0 <= idx <= 168:  # noqa: PLR2004
                raise AssertionError(f"{key!r}: mix cell {cell} out of range")
            if pure[idx] != ".":
                raise AssertionError(f"{key!r}: mix cell {cell} not '.' in pure")
            if len(row) != len(acts):
                raise AssertionError(f"{key!r}: mix row {cell} length mismatch")
            if sum(row) != 100:  # noqa: PLR2004
                raise AssertionError(
                    f"{key!r}: mix row {cell} sums to {sum(row)}"
                )
        if key:
            parent_key, _, c_code = key.rpartition(".")
            parent = nodes.get(parent_key)
            if parent is None:
                raise AssertionError(f"{key!r}: parent {parent_key!r} missing")
            if c_code not in {a["c"] for a in parent["acts"]}:
                raise AssertionError(
                    f"{key!r}: reached by {c_code!r}, not an act of its parent"
                )


def write_solver_pack_json(solver_pack: dict[str, Any], path: Path) -> int:
    """Serialise compactly (machine file: no pretty-printing) -> bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(solver_pack, fh, separators=(",", ":"), ensure_ascii=True)
    return path.stat().st_size
