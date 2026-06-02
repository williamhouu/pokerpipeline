"""Tests for pipeline.preflop.batch (end-to-end orchestrator).

Builds a tiny synthetic preflop pack on disk in tmp_path, runs the
orchestrator with a mock Anthropic client (or dry-run), and asserts on
the resulting CSV + BatchResult metadata.

Most tests use dry-run to skip the LLM entirely -- the LLM-path is
covered by ``tests/test_preflop_explanation_generator.py``. The two
generation-loop tests that exercise the LLM call use a mock client.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.format_writer import CSV_COLUMNS  # noqa: E402
from pipeline.preflop.batch import (  # noqa: E402
    ACTION_CONTEXTS,
    BatchResult,
    _build_batch_meta,
    collect_worthy_spots,
    filter_nodes,
    generate_preflop_batch,
    node_action_context,
)
from pipeline.preflop.grammars.types import (  # noqa: E402
    ParsedAction,
    PreflopActionType,
)
from pipeline.preflop.node_enumerator import (  # noqa: E402
    PreflopDecisionNode,
    enumerate_nodes,
)
from pipeline.preflop.pack import (  # noqa: E402
    PreflopPack,
    clear_registry,
    register_pack,
)
from pipeline.preflop_ranges import canonical_169_hand_classes  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_registry() -> Any:
    """The pack registry is module-global; clean before + after every test
    so registrations from one test don't leak into the next."""
    clear_registry()
    yield
    clear_registry()


# --- fixture builders -------------------------------------------------------
def _write_full_range(path: Path, weights: dict[str, float]) -> None:
    """Write a range file with all 169 entries; unlisted classes default
    to 0.0. parse_range_file requires every class to be present."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = ",".join(
        f"{cls}:{weights.get(cls, 0.0)}" for cls in canonical_169_hand_classes()
    )
    path.write_text(line)


def _build_open_only_pack(tmp_path: Path) -> PreflopPack:
    """One node: UTG decision with Fold + Raise 60% actions.

    A5s and 77 are mixed (60% / 70% open) so they pass the worthiness
    filter; everything else is either pure-open (top of range) or
    pure-fold (bottom of range).
    """
    pack_root = tmp_path / "test_pack"
    utg = pack_root / "UTG"

    classes = canonical_169_hand_classes()
    pure_open = {"AA", "KK", "QQ", "AKs", "AKo", "AQs", "AQo"}
    raise_weights = {c: (1.0 if c in pure_open else 0.0) for c in classes}
    fold_weights = {c: (1.0 - raise_weights[c]) for c in classes}
    # Two mixed hand classes -- the orchestrator's worthiness filter
    # should accept these but reject the pure ones.
    raise_weights["A5s"] = 0.60
    fold_weights["A5s"] = 0.40
    raise_weights["77"] = 0.70
    fold_weights["77"] = 0.30

    _write_full_range(utg / "UTG_60%.txt", raise_weights)
    _write_full_range(utg / "UTG_Fold.txt", fold_weights)

    pack = PreflopPack(
        pack_id="test_open_pack",
        root_path=pack_root,
        grammar_name="ryan_pack",
        table_size=6,
        stack_depth_bb=100,
        open_size_bb=2.5,
        description="test open-only pack",
    )
    register_pack(pack)
    return pack


def _mock_client(responses: list[str]) -> SimpleNamespace:
    """Fake Anthropic-shaped client. Each ``messages.create`` call pops
    one response off the queue."""
    queue = list(responses)
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        text = queue.pop(0)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])

    return SimpleNamespace(
        messages=SimpleNamespace(create=create),
        _calls=calls,
    )


# --- node_action_context -----------------------------------------------------
def _node_with_history(history: tuple[ParsedAction, ...]) -> PreflopDecisionNode:
    return PreflopDecisionNode(
        pack_id="t",
        actor="X",
        history_before=history,
        actions=(),
    )


def test_node_action_context_opening_empty_history() -> None:
    assert node_action_context(_node_with_history(())) == "Opening"


def test_node_action_context_opening_all_folds() -> None:
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("HJ", PreflopActionType.FOLD),
    )
    assert node_action_context(_node_with_history(history)) == "Opening"


def test_node_action_context_facing_single_raise() -> None:
    history = (
        ParsedAction("UTG", PreflopActionType.FOLD),
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
    )
    assert node_action_context(_node_with_history(history)) == "Facing single raise"


def test_node_action_context_facing_3bet() -> None:
    history = (
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("BB", PreflopActionType.RAISE, 182.0),
    )
    assert node_action_context(_node_with_history(history)) == "Facing 3-bet"


def test_node_action_context_facing_4bet() -> None:
    history = (
        ParsedAction("BTN", PreflopActionType.RAISE, 60.0),
        ParsedAction("BB", PreflopActionType.RAISE, 182.0),
        ParsedAction("BTN", PreflopActionType.RAISE, 50.0),
    )
    assert node_action_context(_node_with_history(history)) == "Facing 4-bet+"


def test_node_action_context_after_calls() -> None:
    """An open followed by a caller before hero acts = squeeze spot."""
    history = (
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.CALL),
    )
    assert node_action_context(_node_with_history(history)) == "After call(s)"


def test_action_contexts_constant_matches_ui_options() -> None:
    """The orchestrator and the admin panel must share the same context
    vocabulary -- if a new context label is added, both have to know it."""
    assert ACTION_CONTEXTS == (
        "Opening",
        "Facing single raise",
        "Facing 3-bet",
        "Facing 4-bet+",
        "After call(s)",
    )


# --- filter_nodes -----------------------------------------------------------
def test_filter_nodes_by_position(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    nodes = enumerate_nodes([pack])
    # Only one node (UTG) in our test pack.
    assert len(nodes) == 1
    # Position filter keeps it.
    assert len(filter_nodes(nodes, hero_positions=["UTG"], action_contexts=None)) == 1
    # Position filter rejects it.
    assert len(filter_nodes(nodes, hero_positions=["BTN"], action_contexts=None)) == 0


def test_filter_nodes_empty_filters_keep_all(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    nodes = enumerate_nodes([pack])
    assert filter_nodes(nodes, hero_positions=None, action_contexts=None) == list(nodes)
    assert filter_nodes(nodes, hero_positions=[], action_contexts=[]) == list(nodes)


def test_filter_nodes_action_context(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    nodes = enumerate_nodes([pack])
    # UTG decision with empty history -> Opening context.
    assert (
        len(filter_nodes(nodes, hero_positions=None, action_contexts=["Opening"])) == 1
    )
    assert (
        len(filter_nodes(nodes, hero_positions=None, action_contexts=["Facing 3-bet"]))
        == 0
    )


# --- collect_worthy_spots ----------------------------------------------------
def test_collect_worthy_spots_drops_pure_actions(tmp_path: Path) -> None:
    """A5s (60/40) and 77 (70/30) are worthy; pure-fold and pure-open
    hands are filtered out by the freq window."""
    pack = _build_open_only_pack(tmp_path)
    nodes = enumerate_nodes([pack])
    worthy = collect_worthy_spots(nodes)
    hand_classes = sorted(spot.hero_hand_class for spot, _ in worthy)
    # Only the two mixed hands pass the default 55-95% window.
    assert hand_classes == ["77", "A5s"]


def test_collect_worthy_spots_freq_window_override(tmp_path: Path) -> None:
    """A narrower window (50-70%) excludes 77 (which is at 70%) but
    still keeps A5s (60%). Verifies the freq kwargs plumb through."""
    pack = _build_open_only_pack(tmp_path)
    nodes = enumerate_nodes([pack])
    worthy = collect_worthy_spots(
        nodes,
        min_frequency=0.50,
        max_frequency=0.69,
    )
    hand_classes = [spot.hero_hand_class for spot, _ in worthy]
    assert hand_classes == ["A5s"]


# --- generate_preflop_batch: dry-run path -----------------------------------
def test_dry_run_writes_csv_with_placeholder_explanations(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"

    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        dry_run=True,
        random_seed=42,
    )

    # Two worthy spots in this pack, sample tries 10 but only 2 exist.
    assert result.worthy_spots_available == 2
    assert result.questions_attempted == 2
    assert result.questions_written == 2
    assert result.failures == []
    assert result.output_path == out

    with open(out, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert header == CSV_COLUMNS
    assert len(rows) == 2

    # Hand Stage is "Preflop", Cards on Table is empty.
    stage_idx = CSV_COLUMNS.index("Hand Stage")
    board_idx = CSV_COLUMNS.index("Cards on Table")
    assert all(r[stage_idx] == "Preflop" for r in rows)
    assert all(r[board_idx] == "" for r in rows)

    # Dry-run marker appears in answer_explanation.
    ans_idx = CSV_COLUMNS.index("Answer Explanation")
    assert all("[dry-run placeholder" in r[ans_idx] for r in rows)


def test_dry_run_no_worthy_spots_returns_empty_result(tmp_path: Path) -> None:
    """When no worthy spots match the filters, no CSV is written and
    questions_written is 0 -- the orchestrator doesn't crash."""
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=5,
        action_contexts=["Facing 3-bet"],  # no 3-bet nodes in this pack
        dry_run=True,
    )
    assert result.questions_written == 0
    assert result.questions_attempted == 0
    assert result.worthy_spots_available == 0
    assert result.output_path is None
    assert not out.exists()


def test_difficulty_band_excludes_all_when_min_above_ceiling(
    tmp_path: Path,
) -> None:
    """A min_difficulty above the algorithm's hard ceiling rejects every
    spot -- no rows, and difficulty_filtered_out counts the rejections."""
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        min_difficulty=3201,  # above DIFFICULTY_MAX (3200)
        dry_run=True,
        random_seed=42,
    )
    assert result.worthy_spots_available == 2
    assert result.questions_written == 0
    assert result.questions_attempted == 0
    assert result.difficulty_filtered_out == 2
    assert result.output_path is None
    assert not out.exists()


def test_difficulty_band_excludes_all_when_max_below_floor(
    tmp_path: Path,
) -> None:
    """A max_difficulty below the hard floor rejects every spot too."""
    pack = _build_open_only_pack(tmp_path)
    result = generate_preflop_batch(
        pack=pack,
        output_path=tmp_path / "out.csv",
        total_questions=10,
        max_difficulty=399,  # below DIFFICULTY_MIN (400)
        dry_run=True,
        random_seed=42,
    )
    assert result.questions_written == 0
    assert result.difficulty_filtered_out == 2


def test_default_band_filters_nothing(tmp_path: Path) -> None:
    """The default full band is a no-op filter: every worthy spot is kept
    and difficulty_filtered_out stays at 0."""
    pack = _build_open_only_pack(tmp_path)
    result = generate_preflop_batch(
        pack=pack,
        output_path=tmp_path / "out.csv",
        total_questions=10,
        dry_run=True,
        random_seed=42,
    )
    assert result.questions_written == 2
    assert result.difficulty_filtered_out == 0


def test_min_ev_gap_gate_passes_raise_spots_through(tmp_path: Path) -> None:
    """The min-EV-gap gate only filters spots the EV engine could score.
    Open/raise spots have ev_gap=None, so even an impossibly high gate
    leaves them in (otherwise the gate would wipe out every open spot)."""
    pack = _build_open_only_pack(tmp_path)  # open spots -> ev_gap is None
    result = generate_preflop_batch(
        pack=pack,
        output_path=tmp_path / "out.csv",
        total_questions=10,
        min_ev_gap_bb=99.0,  # impossibly high; raise spots still pass
        dry_run=True,
        random_seed=42,
    )
    assert result.questions_written == 2
    assert result.difficulty_filtered_out == 0


def test_random_seed_makes_sampling_deterministic(tmp_path: Path) -> None:
    """Same seed -> same row order in the CSV."""
    pack = _build_open_only_pack(tmp_path)

    def _run(seed: int) -> list[str]:
        out = tmp_path / f"out_{seed}.csv"
        generate_preflop_batch(
            pack=pack,
            output_path=out,
            total_questions=2,
            dry_run=True,
            random_seed=seed,
        )
        with open(out, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            next(reader)  # header
            hand_col_idx = CSV_COLUMNS.index("User Cards")
            return [r[hand_col_idx] for r in reader]

    assert _run(7) == _run(7)


# --- generate_preflop_batch: progress callback ------------------------------
def test_progress_callback_invoked_per_spot(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"

    calls: list[tuple[str, int, int]] = []

    def cb(msg: str, current: int, total: int) -> None:
        calls.append((msg, current, total))

    generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        dry_run=True,
        progress_callback=cb,
        random_seed=1,
    )
    # Two worthy spots -> two callback invocations.
    assert len(calls) == 2
    for index, (msg, current, total) in enumerate(calls):
        assert "Generating question" in msg
        assert current == index
        assert total == 2


# --- generate_preflop_batch: LLM path with mock client ----------------------
def test_llm_path_writes_csv_with_real_explanation(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"

    # Two worthy spots -> two LLM calls. Both return the same JSON shape
    # (the validator only checks that correct_answer matches an option).
    response = (
        '{"option_1": "Fold", "option_2": "Raise 60%", "option_3": "", '
        '"option_4": "", "correct_answer": "Raise 60%", '
        '"answer_explanation": "Open this hand for value."}'
    )
    client = _mock_client([response, response])

    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        client=client,
        dry_run=False,
        random_seed=0,
    )
    assert result.questions_written == 2
    assert result.failures == []
    # One LLM call per question.
    assert len(client._calls) == 2

    with open(out, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        next(reader)
        rows = list(reader)
    ans_idx = CSV_COLUMNS.index("Answer Explanation")
    assert all(r[ans_idx] == "Open this hand for value." for r in rows)


def test_per_spot_failure_does_not_abort_batch(tmp_path: Path) -> None:
    """One LLM response is malformed; the orchestrator records the
    failure, moves on, and the other spot's row still ships."""
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"

    bad = "this is not JSON at all"
    good = (
        '{"option_1": "Fold", "option_2": "Raise 60%", "option_3": "", '
        '"option_4": "", "correct_answer": "Raise 60%", '
        '"answer_explanation": "Real prose."}'
    )
    # Layer 6 retries once; both attempts return the same bad payload for
    # the first spot, then both attempts return the good payload for
    # the second.
    client = _mock_client([bad, bad, good, good])

    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        client=client,
        dry_run=False,
        random_seed=0,
    )
    # Exactly one spot succeeded, one failed.
    assert result.questions_attempted == 2
    assert result.questions_written == 1
    assert len(result.failures) == 1


def test_batch_result_is_frozen_and_carries_intermediate_counts(tmp_path: Path) -> None:
    """BatchResult exposes nodes_after_filter and worthy_spots_available
    so the admin panel can show 'X nodes match filters, Y worthy spots'
    as a sanity check before/after generation."""
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=1,
        dry_run=True,
        random_seed=0,
    )
    assert isinstance(result, BatchResult)
    assert result.nodes_after_filter == 1
    assert result.worthy_spots_available == 2
    # Frozen: attribute assignment raises FrozenInstanceError
    # (subclass of AttributeError).
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        result.questions_written = 99  # type: ignore[misc]


# --- prompt-workshop: meta sidecar -----------------------------------------
def test_build_batch_meta_snapshots_prompt_and_records() -> None:
    records: list[dict[str, object]] = [
        {
            "node_id": "n1",
            "hand_class": "AKs",
            "framing": "f",
            "options": ["Fold", "Call"],
            "correct_answer": "Call",
            "solver_data": {"x": 1},
            "live_block": "live",
        },
    ]
    meta = _build_batch_meta(
        prompt_name="My Prompt",
        system_prompt="THE SYSTEM PROMPT",
        gold_block="GOLD",
        model="claude-opus-4-7",
        temperature=0.3,
        seed=42,
        dry_run=False,
        prompt_records=records,
    )
    assert meta["prompt_name"] == "My Prompt"
    assert meta["prompt_text"] == "THE SYSTEM PROMPT"
    # Snapshot sha lets a later edit/rename of the prompt stay unambiguous.
    assert meta["prompt_sha"] == hashlib.sha256(b"THE SYSTEM PROMPT").hexdigest()
    assert meta["gold_block"] == "GOLD"
    assert meta["model"] == "claude-opus-4-7"
    assert meta["questions"] == records


def test_build_batch_meta_blanks_model_on_dry_run() -> None:
    meta = _build_batch_meta(
        prompt_name="",
        system_prompt="s",
        gold_block="g",
        model="claude-opus-4-7",
        temperature=0.0,
        seed=None,
        dry_run=True,
        prompt_records=[],
    )
    # Dry-run didn't call the model, so the id is blanked (matches model_used).
    assert meta["model"] == ""
    assert meta["dry_run"] is True


def test_dry_run_writes_meta_sidecar_with_prompt_tag_and_inputs(
    tmp_path: Path,
) -> None:
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        dry_run=True,
        random_seed=42,
        system_prompt="WORKSHOP PROMPT UNDER TEST",
        prompt_name="Workshop v1",
    )
    # The sidecar sits next to the CSV and is surfaced on the result.
    expected_meta = out.with_suffix(".meta.json")
    assert result.meta_path == expected_meta
    assert expected_meta.is_file()
    assert result.prompt_name == "Workshop v1"

    meta = json.loads(expected_meta.read_text(encoding="utf-8"))
    assert meta["prompt_name"] == "Workshop v1"
    # The run was built against the custom prompt, snapshotted verbatim.
    assert meta["prompt_text"] == "WORKSHOP PROMPT UNDER TEST"
    assert meta["dry_run"] is True
    # One meta question per CSV row, captured in row order.
    assert len(meta["questions"]) == result.questions_written == 2
    q0 = meta["questions"][0]
    assert set(q0) >= {
        "node_id",
        "hand_class",
        "framing",
        "options",
        "correct_answer",
        "solver_data",
        "live_block",
    }
    assert isinstance(q0["solver_data"], dict)
    assert isinstance(q0["options"], list)
    assert "SOLVER DATA" in q0["live_block"]


def test_no_rows_writes_no_meta(tmp_path: Path) -> None:
    pack = _build_open_only_pack(tmp_path)
    out = tmp_path / "out.csv"
    result = generate_preflop_batch(
        pack=pack,
        output_path=out,
        total_questions=10,
        action_contexts=["Facing 4-bet+"],  # open-only pack has none
        dry_run=True,
        random_seed=42,
    )
    assert result.questions_written == 0
    assert result.meta_path is None
    assert not out.with_suffix(".meta.json").exists()


def test_node_is_unconverged_guard_against_real_pack() -> None:
    """Convergence guard: a clean RFI node passes, but the deep multiway jam
    tail (AA folding / premium inversions) is overwhelmingly flagged."""
    ranges = Path(__file__).resolve().parent.parent / "ranges"
    if not ranges.is_dir():
        pytest.skip("ranges/ not present locally")
    from pipeline.preflop.batch import node_is_unconverged
    from pipeline.preflop.grammars.types import PreflopActionType as PT
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import clear_registry, discover_packs

    clear_registry()
    packs = discover_packs(ranges)
    if not packs:
        pytest.skip("Ryan pack not present under ranges/")
    nodes = enumerate_nodes(packs)
    # A clean UTG RFI node (first to act) is converged -> not flagged.
    utg = next(n for n in nodes if n.actor == "UTG" and len(n.history_before) == 0)
    assert node_is_unconverged(utg) is False
    # The deep multiway facing-jam tail is unconverged -> the guard fires on
    # the large majority of it (we measured ~88% AA-folding).
    jam = [
        n
        for n in nodes
        if n.actor == "UTG"
        and any(a.action_type is PT.ALL_IN for a in n.history_before)
        and len(n.history_before) >= 6
    ]
    assert jam, "expected deep multiway jam nodes in the pack"
    flagged = sum(1 for n in jam if node_is_unconverged(n))
    assert flagged > 0.5 * len(jam)


def test_filter_nodes_by_player_count() -> None:
    """player_counts keeps only nodes with that many players still in."""
    from pipeline.preflop.batch import active_player_count, filter_nodes
    from pipeline.preflop.grammars.types import ParsedAction
    from pipeline.preflop.grammars.types import PreflopActionType as PT
    from pipeline.preflop.node_enumerator import PreflopDecisionNode

    def _node(actor: str, history: tuple) -> PreflopDecisionNode:
        return PreflopDecisionNode(
            pack_id="t", actor=actor, history_before=history, actions=()
        )

    open_node = _node("UTG", ())                                    # {UTG} = 1
    heads_up = _node("BB", (ParsedAction("BTN", PT.RAISE, 60.0),))   # {BTN,BB} = 2
    three_way = _node(
        "SB", (ParsedAction("HJ", PT.RAISE, 60.0), ParsedAction("CO", PT.CALL))
    )                                                               # {HJ,CO,SB} = 3
    assert active_player_count(open_node) == 1
    assert active_player_count(heads_up) == 2
    assert active_player_count(three_way) == 3

    nodes = [open_node, heads_up, three_way]
    # Only heads-up.
    assert filter_nodes(
        nodes, hero_positions=None, action_contexts=None, player_counts={2}
    ) == [heads_up]
    # None = no filter (everything).
    assert filter_nodes(
        nodes, hero_positions=None, action_contexts=None, player_counts=None
    ) == nodes
    # Open + three-way.
    kept = filter_nodes(
        nodes, hero_positions=None, action_contexts=None, player_counts={1, 3}
    )
    assert kept == [open_node, three_way]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
