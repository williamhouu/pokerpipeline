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

from pipeline.preflop.format_writer import PREFLOP_CSV_COLUMNS  # noqa: E402
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


def test_node_action_context_after_one_call() -> None:
    """An open + a single live flat-caller before hero = a tame squeeze spot."""
    history = (
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.CALL),
    )
    assert node_action_context(_node_with_history(history)) == "After one call"


def test_node_action_context_after_multiple_calls() -> None:
    """Two or more live flat-callers = the multiway bucket."""
    history = (
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.CALL),
        ParsedAction("BTN", PreflopActionType.CALL),
    )
    assert node_action_context(_node_with_history(history)) == "After multiple calls"


def test_node_action_context_3bet_pot_with_caller_is_facing_3bet() -> None:
    """Raise level wins over an earlier flat (June 2026 reorder): opened,
    flat-called, then 3-bet (a squeeze) is 'Facing 3-bet', NOT an after-call
    bucket. The flat doesn't hide the raise hero faces."""
    history = (
        ParsedAction("HJ", PreflopActionType.RAISE, 60.0),
        ParsedAction("CO", PreflopActionType.CALL),
        ParsedAction("BTN", PreflopActionType.RAISE, 165.0),  # 3-bet squeeze
    )
    assert node_action_context(_node_with_history(history)) == "Facing 3-bet"


def test_node_action_context_4bet_pot_with_caller_is_facing_4bet() -> None:
    """A 4-bet pot that contains an early flat is still 'Facing 4-bet+' -- the
    reorder stops it hiding under 'After one call' (the real #1/#5/#7 bug)."""
    history = (
        ParsedAction("UTG", PreflopActionType.RAISE, 60.0),
        ParsedAction("UTG+2", PreflopActionType.CALL),
        ParsedAction("LJ", PreflopActionType.RAISE, 180.0),   # 3-bet
        ParsedAction("SB", PreflopActionType.RAISE, 460.0),   # 4-bet
    )
    assert node_action_context(_node_with_history(history)) == "Facing 4-bet+"


def test_node_is_unconverged_flags_uniform_default(tmp_path: Path) -> None:
    """A node where most reaching hands split equally across every action
    (the solver's unrefined default) is flagged -- even when AA behaves, so
    this isolates the uniform-default tell from the AA canary."""
    from pipeline.preflop.batch import node_is_unconverged
    from pipeline.preflop.grammars.types import ParsedRangeFile
    from pipeline.preflop.node_enumerator import (
        PreflopActionOption,
        PreflopDecisionNode,
    )
    from pipeline.preflop_ranges import canonical_169_hand_classes

    hands = canonical_169_hand_classes()
    # AA continues 100% (so the AA canary is satisfied); every OTHER hand is
    # 1/3 in each of fold/call/raise -- the uniform default.
    w = {"Fold": {}, "Call": {}, "Raise": {}}
    for h in hands:
        if h == "AA":
            w["Fold"][h], w["Call"][h], w["Raise"][h] = 0.0, 1.0, 0.0
        else:
            w["Fold"][h] = w["Call"][h] = w["Raise"][h] = 1 / 3
    opts = []
    for label, act in (
        ("Fold", PreflopActionType.FOLD),
        ("Call", PreflopActionType.CALL),
        ("Raise", PreflopActionType.RAISE),
    ):
        p = tmp_path / f"{label}.txt"
        p.write_text(
            ",".join(f"{h}:{w[label][h]:.4f}" for h in hands), encoding="utf-8"
        )
        rf = ParsedRangeFile(
            pack_id="t", path=p, actor="BTN", actor_action=act,
            actor_raise_size_pct=None, action_history=(),
        )
        opts.append(
            PreflopActionOption(action_type=act, raise_size_pct=None, range_file=rf)
        )
    node = PreflopDecisionNode(
        pack_id="t", actor="BTN", history_before=(), actions=tuple(opts)
    )
    assert node_is_unconverged(node) is True


# --- EV-coherence guard (spot_mix_incoherent) -------------------------------
def _patch_mix(monkeypatch, *, evs, freqs) -> None:
    """Patch the two readers spot_mix_incoherent lazily imports at call time,
    so the gate's threshold logic can be tested without a real pack/EVs."""
    monkeypatch.setattr(
        "pipeline.preflop.format_writer.action_evs_bb", lambda f, p: evs
    )
    monkeypatch.setattr(
        "pipeline.preflop.options.canonicalize_strategy", lambda f: freqs
    )


def test_spot_mix_incoherent_flags_large_ev_spread(monkeypatch) -> None:
    """The 82o case: a 26%-played action ~15bb worse than the fold it mostly
    takes is unconverged-node noise, not a real mixed strategy."""
    from pipeline.preflop.batch import spot_mix_incoherent

    _patch_mix(
        monkeypatch,
        evs={"Fold": -1.0, "Call": -16.1},
        freqs={"Fold": 0.737, "Call": 0.263},
    )
    assert spot_mix_incoherent(object(), object()) is True


def test_spot_mix_incoherent_passes_indifferent_mix(monkeypatch) -> None:
    """The Q9o case: two actions at equal EV is exactly what GTO mixing is."""
    from pipeline.preflop.batch import spot_mix_incoherent

    _patch_mix(
        monkeypatch,
        evs={"3-bet": -0.96, "Call": -0.96},
        freqs={"3-bet": 0.805, "Call": 0.195},
    )
    assert spot_mix_incoherent(object(), object()) is False


def test_spot_mix_incoherent_ignores_sub_threshold_sliver(monkeypatch) -> None:
    """The AJs case: a ~1.5% sliver on a worse action doesn't make a near-pure
    fold a 'mix' -- only one action clears the 10% frequency floor."""
    from pipeline.preflop.batch import spot_mix_incoherent

    _patch_mix(
        monkeypatch,
        evs={"Fold": -12.0, "Call": -35.0},
        freqs={"Fold": 0.985, "Call": 0.015},
    )
    assert spot_mix_incoherent(object(), object()) is False


def test_spot_mix_incoherent_passes_without_ev_data(monkeypatch) -> None:
    """Weight-only packs (no per-action EVs) can't be judged -- fail open."""
    from pipeline.preflop.batch import spot_mix_incoherent

    _patch_mix(monkeypatch, evs=None, freqs={"Fold": 0.6, "Call": 0.4})
    assert spot_mix_incoherent(object(), object()) is False


def test_spot_mix_incoherent_threshold_boundary(monkeypatch) -> None:
    """A small EV spread (slightly imperfect convergence) passes; a clearly
    non-indifferent spread fails."""
    from pipeline.preflop.batch import spot_mix_incoherent

    _patch_mix(
        monkeypatch,
        evs={"Call": -0.5, "Raise": -3.0},  # 2.5bb spread (< 3.0 default)
        freqs={"Call": 0.5, "Raise": 0.5},
    )
    assert spot_mix_incoherent(object(), object()) is False
    _patch_mix(
        monkeypatch,
        evs={"Call": -0.5, "Raise": -5.5},  # 5.0bb spread
        freqs={"Call": 0.5, "Raise": 0.5},
    )
    assert spot_mix_incoherent(object(), object()) is True


# --- ev_gap_from_action_evs (EV gap from the solver's own per-action EVs) ----
def test_ev_gap_from_action_evs_uses_solver_evs(monkeypatch) -> None:
    """The QQ case: dominant-by-frequency EV minus 2nd-most-frequent EV, from
    the solver's OWN per-action EVs -- 0.11bb, NOT the analytic 12.55bb that
    used to skew difficulty on multiway 3-bet pots."""
    from pipeline.preflop.batch import ev_gap_from_action_evs

    _patch_mix(
        monkeypatch,
        evs={"Fold": -4.00, "Call": -3.89, "All-in": -5.77},
        freqs={"Fold": 0.567, "Call": 0.421, "All-in": 0.006},
    )
    gap = ev_gap_from_action_evs(object(), object())
    assert gap is not None
    assert abs(gap - 0.11) < 1e-6


def test_ev_gap_from_action_evs_none_without_file_evs(monkeypatch) -> None:
    """EV-less packs (the Ryan pack) -> None, so the caller falls back to the
    analytic engine."""
    from pipeline.preflop.batch import ev_gap_from_action_evs

    _patch_mix(monkeypatch, evs=None, freqs={"Fold": 0.6, "Call": 0.4})
    assert ev_gap_from_action_evs(object(), object()) is None


def test_action_contexts_constant_matches_ui_options() -> None:
    """The orchestrator and the admin panel must share the same context
    vocabulary -- if a new context label is added, both have to know it."""
    assert ACTION_CONTEXTS == (
        "Opening",
        "Facing single raise",
        "Facing 3-bet",
        "Facing 4-bet+",
        "After one call",
        "After multiple calls",
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
    assert header == PREFLOP_CSV_COLUMNS
    assert len(rows) == 2

    # Hand Stage is "Preflop", Cards on Table is empty.
    stage_idx = PREFLOP_CSV_COLUMNS.index("Hand Stage")
    board_idx = PREFLOP_CSV_COLUMNS.index("Cards on Table")
    assert all(r[stage_idx] == "Preflop" for r in rows)
    assert all(r[board_idx] == "" for r in rows)

    # Dry-run marker appears in answer_explanation.
    ans_idx = PREFLOP_CSV_COLUMNS.index("Answer Explanation")
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
            hand_col_idx = PREFLOP_CSV_COLUMNS.index("User Cards")
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
    ans_idx = PREFLOP_CSV_COLUMNS.index("Answer Explanation")
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
    # The malformed response never parsed into a candidate, so there's no
    # rebuilt row -- this spot is viewable on Review but not one-click
    # promotable (you'd re-generate it). See the next test for the parsed case.
    assert result.failures[0].row is None


def test_build_failure_builds_row_from_rejected_candidate(tmp_path: Path) -> None:
    """When the rejected attempt PARSED (a candidate exists), the failure
    carries the COMPLETE CSV row built from that exact explanation -- the
    'keep the exact explanation' path the Review page promotes."""
    from dataclasses import asdict

    from pipeline.preflop.batch import _build_failure
    from pipeline.preflop.difficulty import compute_difficulty
    from pipeline.preflop.explanation_generator import (
        ExplanationValidationError,
        GeneratedExplanation,
    )
    from pipeline.preflop.fact_extractor import extract_facts
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.options import build_options
    from pipeline.preflop.spot_sampler import sample_spot

    pack = _build_open_only_pack(tmp_path)
    node = enumerate_nodes([pack])[0]
    spot = sample_spot(node, "AA")
    facts = extract_facts(spot, pack, equity_runouts=20)
    difficulty = compute_difficulty(facts)
    options, correct = build_options(facts)
    padded = (list(options) + ["", "", "", ""])[:4]
    candidate = GeneratedExplanation(
        option_1=padded[0],
        option_2=padded[1],
        option_3=padded[2],
        option_4=padded[3],
        correct_answer=correct,
        answer_explanation="Kept-verbatim coaching prose.",
    )
    exc = ExplanationValidationError(
        "rejected by a content validator",
        last_attempt_text="{...}",
        last_attempt_candidate=candidate,
    )
    failure = _build_failure(
        spot, facts, options, correct, exc,
        pack=pack, stakes_bb_dollars=0.5, live_or_online="Online",
        game_format="cash", difficulty=difficulty, display_in_bb=False,
    )
    assert failure.row is not None
    assert failure.row["Answer Explanation"] == "Kept-verbatim coaching prose."
    assert failure.row["validation_status"] == "needs_review"
    # Serializes for the meta sidecar exactly as the batch persists it.
    json.loads(json.dumps(asdict(failure), default=str))


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
    # Counters default to {} when the caller doesn't pass them.
    assert meta["counters"] == {}


def test_build_batch_meta_records_counters() -> None:
    """Outcome counters land in the meta sidecar (June 2026 audit gap:
    gate SETTINGS were recorded but the skip counts were UI-only, so a
    re-audit couldn't confirm the gates fired from the meta alone)."""
    counters = {
        "rare_line_filtered_out": 3,
        "rare_premise_filtered_out": 6,
        "questions_written": 5,
        "soft_flagged_rows": 1,
    }
    meta = _build_batch_meta(
        prompt_name="P",
        system_prompt="S",
        gold_block="G",
        model="m",
        temperature=0.3,
        seed=None,
        dry_run=True,
        prompt_records=[],
        counters=counters,
    )
    assert meta["counters"] == counters


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


def _write_canary_rng(
    path: Path, overrides: dict[str, tuple[float, float]]
) -> None:
    """A structurally valid 169-class Monker .rng with per-hand overrides."""
    from pipeline.preflop_ranges import canonical_169_hand_classes

    lines: list[str] = []
    for cls in canonical_169_hand_classes():
        p, ev = overrides.get(cls, (0.0, 0.0))
        lines.append(cls)
        lines.append(f"{p};{ev}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _canary_node(tmp_path: Path, files: dict[str, dict[str, tuple[float, float]]]):
    """Build one decision node from Monker stems -> per-hand override maps."""
    from pipeline.preflop.node_enumerator import enumerate_nodes
    from pipeline.preflop.pack import PreflopPack

    for stem, overrides in files.items():
        _write_canary_rng(tmp_path / f"{stem}.rng", overrides)
    pack = PreflopPack(
        pack_id="canary_test",
        root_path=tmp_path,
        grammar_name="monker_nlhe",
        table_size=9,
        stack_depth_bb=100,
        open_size_bb=4.0,
        file_glob="*.rng",
    )
    (node,) = enumerate_nodes([pack])
    return node


# SB limped earlier, faces the BB's iso-raise: AA's presence at this node
# is small (AA almost never limps) -- the canary must judge the
# CONDITIONAL strategy, not the raw joint mass.
_LIMP_WAR = "0.0.0.0.0.0.0.1.40100"


def test_canary_passes_converged_hero_acted_before_node(tmp_path: Path) -> None:
    """AA reaches at 0.2 (it rarely limped) but continues 100% of the time
    it is here -> converged, NOT flagged. The pre-June-2026 raw-sum version
    compared 0.2 to ~1 and flagged this (and ~80% of every hero-acted-before
    node on the 9-max pack), silently gutting After-call(s) generation."""
    from pipeline.preflop.batch import node_is_unconverged

    node = _canary_node(
        tmp_path,
        {
            f"{_LIMP_WAR}.0": {"AA": (0.0, 0.0), "KK": (0.0, 0.0)},
            f"{_LIMP_WAR}.1": {"AA": (0.05, 0.0), "KK": (0.1, 0.0)},
            f"{_LIMP_WAR}.40100": {"AA": (0.15, 0.0), "KK": (0.1, 0.0)},
        },
    )
    assert node_is_unconverged(node) is False


def test_canary_flags_aa_folding_conditionally(tmp_path: Path) -> None:
    """AA reaches at 0.2 but folds half of that -> unconverged, flagged."""
    from pipeline.preflop.batch import node_is_unconverged

    node = _canary_node(
        tmp_path,
        {
            f"{_LIMP_WAR}.0": {"AA": (0.1, 0.0)},
            f"{_LIMP_WAR}.1": {"AA": (0.05, 0.0)},
            f"{_LIMP_WAR}.40100": {"AA": (0.05, 0.0)},
        },
    )
    assert node_is_unconverged(node) is True


def test_canary_skips_near_zero_presence(tmp_path: Path) -> None:
    """AA effectively never reaches (presence 0.001) -> nothing to judge;
    the spot-level presence filter owns these, not the canary."""
    from pipeline.preflop.batch import node_is_unconverged

    node = _canary_node(
        tmp_path,
        {
            f"{_LIMP_WAR}.0": {"AA": (0.001, 0.0)},
            f"{_LIMP_WAR}.1": {"72o": (0.5, 0.0)},
            f"{_LIMP_WAR}.40100": {},
        },
    )
    assert node_is_unconverged(node) is False


def test_canary_flags_conditional_premium_inversion(tmp_path: Path) -> None:
    """KK continuing less often than QQ (conditionally) is an inversion."""
    from pipeline.preflop.batch import node_is_unconverged

    node = _canary_node(
        tmp_path,
        {
            f"{_LIMP_WAR}.0": {"KK": (0.2, 0.0), "QQ": (0.05, 0.0)},
            f"{_LIMP_WAR}.1": {"AA": (1.0, 0.0), "KK": (0.4, 0.0), "QQ": (0.5, 0.0)},
            f"{_LIMP_WAR}.40100": {"KK": (0.4, 0.0), "QQ": (0.45, 0.0)},
        },
    )
    # AA: presence 1.0, continues 1.0 (passes). KK: 0.8/1.0 = 0.80
    # continue; QQ: 0.95/1.0 = 0.95 -> KK < QQ - 0.10 -> flagged.
    assert node_is_unconverged(node) is True


def test_node_is_unconverged_guard_against_real_packs() -> None:
    """Convergence guard on the real packs: clean RFI nodes pass, the deep
    jam tail is flagged at a real-but-not-wholesale rate (the conditional
    canary fires on true AA-misbehavior, not on low reach), and a known
    garbage 9-max node (AA folding half its mass facing a jam) is caught."""
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
        pytest.skip("no real pack present under ranges/")
    for pack in packs:
        nodes = enumerate_nodes([pack])
        # A clean UTG RFI node (first to act) is converged -> not flagged.
        utg = next(
            n for n in nodes if n.actor == "UTG" and len(n.history_before) == 0
        )
        assert node_is_unconverged(utg) is False, pack.pack_id
        # Deep facing-jam tail: flagged meaningfully but not wholesale
        # (measured 8-9% on the 100bb Ryan + 9-max packs with the
        # conditional canary). The rate band is calibrated to those deep
        # 100bb trees; the short-stack 6-max packs (20/30bb) are a different
        # regime -- their push/fold-ish tails converge tightly and AA never
        # misbehaves facing a jam, so the canary legitimately stays quiet
        # (0 flagged is correct, not a miss). Only assert the band where it
        # was calibrated; the clean-RFI check above still covers every pack.
        jam = [
            n
            for n in nodes
            if n.actor == "UTG"
            and any(a.action_type is PT.ALL_IN for a in n.history_before)
            and len(n.history_before) >= 6
        ]
        if jam and pack.stack_depth_bb >= 100:
            flagged = sum(1 for n in jam if node_is_unconverged(n))
            assert 0.02 * len(jam) <= flagged <= 0.6 * len(jam), (
                f"{pack.pack_id}: {flagged}/{len(jam)}"
            )
        if pack.grammar_name == "monker_nlhe":
            # The audit's known-garbage node: 5-way limp/jam fest where the
            # BB chart shows AA folding ~50% -- must be caught.
            bad = next(
                (
                    n
                    for n in nodes
                    if n.actor == "BB"
                    and n.actions[0].range_file.path.stem.startswith(
                        "40120.40084.40046.3.1.1.1.1"
                    )
                ),
                None,
            )
            if bad is not None:
                assert node_is_unconverged(bad) is True


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
    # HJ opened then folded to the squeeze -> out of the hand: {SB,BTN} = 2.
    entrant_folded = _node(
        "BTN",
        (
            ParsedAction("HJ", PT.RAISE, 60.0),
            ParsedAction("BTN", PT.CALL),
            ParsedAction("SB", PT.RAISE, 200.0),
            ParsedAction("HJ", PT.FOLD),
        ),
    )
    assert active_player_count(open_node) == 1
    assert active_player_count(heads_up) == 2
    assert active_player_count(three_way) == 3
    assert active_player_count(entrant_folded) == 2

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


# --- _incoming_villain_line_pct: limped-pot premise gate (June 2026) ---------
def test_incoming_villain_line_pct_measures_limp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A limped pot (no raiser) gets the completer's call frequency measured.

    identify_villain only sees raises, so a rare limp (the SB completes
    ~0.1% on the 9-max pack) used to slip the premise gate and produce a
    bb-vs-limp question built on a near-never action."""
    from pipeline.preflop import batch as B
    from pipeline.preflop.spot_sampler import PreflopSpot

    folds = tuple(
        ParsedAction(p, PreflopActionType.FOLD)
        for p in ("UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN")
    )
    limper_hist = folds  # everything before the SB acts
    hero_hist = folds + (ParsedAction("SB", PreflopActionType.CALL),)
    hero_node = PreflopDecisionNode(
        pack_id="t", actor="BB", history_before=hero_hist, actions=()
    )
    limper_node = PreflopDecisionNode(
        pack_id="t", actor="SB", history_before=limper_hist, actions=()
    )
    node_index = {("SB", limper_hist): limper_node}

    # No raiser in the line.
    monkeypatch.setattr(B, "identify_villain", lambda n: None)
    # SB limps 52s half the time, raise-or-folds everything else.
    fake_spots = [
        PreflopSpot(
            node=limper_node, hero_hand_class="52s", hero_card_combo="5s2s",
            action_frequencies={"Call": 0.5, "Raise 60%": 0.0, "Fold": 0.5},
            dominant_action="Fold", dominant_frequency=0.5, presence=1.0,
        ),
        PreflopSpot(
            node=limper_node, hero_hand_class="AA", hero_card_combo="AsAh",
            action_frequencies={"Call": 0.0, "Raise 60%": 1.0, "Fold": 0.0},
            dominant_action="Raise 60%", dominant_frequency=1.0, presence=1.0,
        ),
    ]
    monkeypatch.setattr(B, "enumerate_spots_for_node", lambda n: fake_spots)

    pct = B._incoming_villain_line_pct(hero_node, node_index, pack=None)
    # 52s: presence 1.0 * call 0.5 * 4 combos = 2.0 ; AA: 0 -> total 2.0.
    assert pct == pytest.approx(100.0 * 2.0 / 1326.0, abs=1e-6)
    assert pct < 0.25  # below the default floor -> the spot is gated


def test_incoming_villain_line_pct_delegates_to_raiser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When there IS a raiser, delegate to _villain_line_pct unchanged."""
    from pipeline.preflop import batch as B

    node = PreflopDecisionNode(
        pack_id="t", actor="BB",
        history_before=(ParsedAction("BTN", PreflopActionType.RAISE, 60.0),),
        actions=(),
    )
    monkeypatch.setattr(B, "identify_villain", lambda n: n.history_before[-1])
    monkeypatch.setattr(B, "_villain_line_pct", lambda n, p: 7.5)
    assert B._incoming_villain_line_pct(node, {}, pack=None) == 7.5


def test_snap_facts_to_pure_displays_near_pure_as_100() -> None:
    """A 96-99% spot is shown as 100% (pure freqs, EV note dropped), with the
    real mix returned for the Review/Compare note. Below the threshold the
    spot is untouched."""
    from pipeline.preflop.batch import _snap_facts_to_pure
    from pipeline.preflop.fact_extractor import PreflopFacts
    from pipeline.preflop.spot_sampler import PreflopSpot

    node = PreflopDecisionNode(
        pack_id="t", actor="BB", history_before=(), actions=()
    )
    spot = PreflopSpot(
        node=node, hero_hand_class="K3s", hero_card_combo="Ks3s",
        action_frequencies={"Fold": 0.97, "Call": 0.03},
        dominant_action="Fold", dominant_frequency=0.97, presence=1.0,
    )
    facts = PreflopFacts(spot=spot, ev_gap_bb=0.04)

    # At/above threshold -> snapped to pure; real mix returned; EV note dropped.
    snapped, real = _snap_facts_to_pure(facts, spot, 0.96)
    assert isinstance(snapped, PreflopFacts)
    assert real == {"Fold": 0.97, "Call": 0.03}
    assert snapped.spot.action_frequencies == {"Fold": 1.0, "Call": 0.0}
    assert snapped.spot.dominant_frequency == 1.0
    assert snapped.spot.dominant_action == "Fold"
    assert snapped.ev_gap_bb is None

    # Below threshold -> untouched (same object), no real freqs.
    same, none_freqs = _snap_facts_to_pure(facts, spot, 0.99)
    assert none_freqs is None
    assert same is facts


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
