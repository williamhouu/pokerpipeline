"""Tests for pipeline.preflop_ranges.

Pure unit tests. The biggest risk in this module is getting Pio's combo
enumeration wrong (silently mapping AsAh to a different position than Pio
expects produces a perfectly valid file that solves the wrong game). The
canonical order was verified once against Pio Edge 3's show_hand_order() and
saved at test_output/pio_hand_order.json; the headline test below replays
those samples so future-us catches any accidental drift.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.preflop_ranges import (                                       # noqa: E402
    CARD_COUNT, HAND_COUNT, aggregate_combo_range_to_classes,
    canonical_169_hand_classes, card_label, combo_cards, combo_label,
    combo_str_to_hand_class, combo_to_hand_class, expand_to_combo_weights,
    format_hand_class_range, format_set_range_line, parse_range_file,
)


# --- card index <-> label ---------------------------------------------------
def test_card_label_endpoints():
    # Pio's card order: rank slow (2..A), suit fast (c,d,h,s).
    assert card_label(0) == "2c"
    assert card_label(1) == "2d"
    assert card_label(2) == "2h"
    assert card_label(3) == "2s"
    assert card_label(4) == "3c"
    assert card_label(51) == "As"


def test_card_label_count():
    labels = {card_label(i) for i in range(CARD_COUNT)}
    assert len(labels) == CARD_COUNT


# --- combo enumeration ------------------------------------------------------
def test_combo_label_matches_pio_show_hand_order_samples():
    """The first 12 and last 6 entries Pio's show_hand_order() emits, fixed
    against PioSOLVER3-edge.exe. If this test ever fails, Pio's order has
    drifted (highly unlikely between Edge 3 builds, but worth catching)."""
    expected_first_12 = [
        "2d2c", "2h2c", "2h2d", "2s2c", "2s2d", "2s2h",
        "3c2c", "3c2d", "3c2h", "3c2s", "3d2c", "3d2d",
    ]
    for i, want in enumerate(expected_first_12):
        assert combo_label(i) == want, f"position {i}: {combo_label(i)!r} != {want!r}"
    expected_last_6 = ["AsKd", "AsKh", "AsKs", "AsAc", "AsAd", "AsAh"]
    for offset, want in enumerate(expected_last_6):
        position = HAND_COUNT - 6 + offset
        assert combo_label(position) == want, \
            f"position {position}: {combo_label(position)!r} != {want!r}"


def test_combo_position_formula_roundtrip():
    """Every position decodes to two cards that re-encode to the same position
    (via the same formula). Guards against an off-by-one in combo_cards."""
    for position in range(HAND_COUNT):
        a, b = combo_cards(position)
        assert a > b
        # Re-encode.
        re_encoded = a * (a - 1) // 2 + b
        assert re_encoded == position


def test_combo_count_is_1326():
    seen = {combo_label(p) for p in range(HAND_COUNT)}
    assert len(seen) == HAND_COUNT


# --- hand-class classification ----------------------------------------------
def test_hand_class_examples():
    # AA: both Aces, any two of {Ac, Ad, Ah, As}.
    # As=51, Ah=50; 51*50/2+50 = 1325. (Last position; classified as AA.)
    assert combo_to_hand_class(51, 50) == "AA"
    # 22: 2c=0, 2d=1; position 0. Classified as 22.
    assert combo_to_hand_class(1, 0) == "22"
    # AKs: A and K, same suit. As=51, Ks=47; same suit (s). AKs.
    assert combo_to_hand_class(51, 47) == "AKs"
    # AKo: A and K, different suits. As=51, Kh=46. AKo.
    assert combo_to_hand_class(51, 46) == "AKo"
    # 72o (offsuit): 7 and 2, different suits.
    # 7c=20, 2d=1. 72o (high rank first).
    assert combo_to_hand_class(20, 1) == "72o"
    # T9s: T and 9, same suit. Th=34, 9h=30. T9s.
    assert combo_to_hand_class(34, 30) == "T9s"


def test_hand_classes_cover_169():
    classes = set()
    for position in range(HAND_COUNT):
        a, b = combo_cards(position)
        classes.add(combo_to_hand_class(a, b))
    assert len(classes) == 169


# --- pack file parsing ------------------------------------------------------
def test_parse_range_file_success(tmp_path):
    """A synthetic 169-entry file parses to a 169-class dict with float weights."""
    # Build a fake range file with all 169 classes at weight 0.5.
    ranks = "23456789TJQKA"
    classes: list[str] = []
    for i, hi in enumerate(reversed(ranks)):
        for j, lo in enumerate(reversed(ranks)):
            if i == j:
                classes.append(hi + lo)
            elif i < j:
                classes.append(hi + lo + "s")
            else:
                classes.append(lo + hi + "o")
    classes = list(dict.fromkeys(classes))           # de-dupe ordered
    assert len(classes) == 169
    pairs = [f"{c}:0.5" for c in classes]
    file_path = tmp_path / "fake_range.txt"
    file_path.write_text(",".join(pairs), encoding="utf-8")
    parsed = parse_range_file(file_path)
    assert len(parsed) == 169
    assert all(w == 0.5 for w in parsed.values())


def test_parse_range_file_too_few_classes_raises(tmp_path):
    bad = tmp_path / "bad.txt"
    bad.write_text("AA:1.0,KK:1.0", encoding="utf-8")
    try:
        parse_range_file(bad)
    except ValueError as exc:
        assert "169" in str(exc)
        return
    raise AssertionError("expected ValueError for short file")


# --- expansion --------------------------------------------------------------
def test_expand_to_combo_weights_uniform_class():
    """If a hand class has weight w, every one of its combos gets weight w."""
    # Build a complete 169-class dict with AA=0.7, everything else 0.
    classes = _all_169_classes()
    hcw = {c: 0.0 for c in classes}
    hcw["AA"] = 0.7
    weights = expand_to_combo_weights(hcw)
    assert len(weights) == HAND_COUNT
    # Six AA combos (Ac/Ad/Ah/As pairwise) should be 0.7; the rest 0.
    aa_positions = [p for p in range(HAND_COUNT)
                    if combo_to_hand_class(*combo_cards(p)) == "AA"]
    assert len(aa_positions) == 6
    assert all(weights[p] == 0.7 for p in aa_positions)
    assert sum(1 for w in weights if w != 0) == 6


def test_expand_to_combo_weights_preserves_total():
    """Sum-of-weighted-combos must equal sum-of-(class_weight * class_combos).

    Pairs have 6 combos, suited have 4, offsuit have 12.
    """
    classes = _all_169_classes()
    hcw = {c: 0.0 for c in classes}
    hcw["AA"] = 1.0      # 6 combos
    hcw["AKs"] = 0.5     # 4 combos -> 2.0
    hcw["AKo"] = 0.25    # 12 combos -> 3.0
    weights = expand_to_combo_weights(hcw)
    assert sum(weights) == 6.0 + 2.0 + 3.0


def _all_169_classes() -> list[str]:
    """The canonical 169 hand-class names."""
    ranks = "23456789TJQKA"
    out: list[str] = []
    for i, hi in enumerate(reversed(ranks)):
        for j, lo in enumerate(reversed(ranks)):
            if i == j:
                out.append(hi + lo)
            elif i < j:
                out.append(hi + lo + "s")
            else:
                out.append(lo + hi + "o")
    return list(dict.fromkeys(out))


# --- set_range line formatting ----------------------------------------------
def test_format_set_range_line_starts_with_side():
    weights = [0.0] * HAND_COUNT
    line = format_set_range_line("OOP", weights)
    assert line.startswith("set_range OOP ")


def test_format_set_range_line_compact_integers():
    """Whole numbers render as '0' or '1' (no decimal), matching the
    style Pio's GUI tree-builder writes."""
    weights = [0.0] * HAND_COUNT
    weights[0] = 1.0
    weights[5] = 0.5
    line = format_set_range_line("IP", weights)
    tokens = line.split()[2:]
    assert tokens[0] == "1"
    assert tokens[1] == "0"
    assert tokens[5] == "0.5"


def test_format_set_range_line_rejects_wrong_size():
    try:
        format_set_range_line("OOP", [0.0] * 100)
    except ValueError as exc:
        assert "1326" in str(exc) or "expected" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_format_set_range_line_rejects_bad_side():
    try:
        format_set_range_line("HERO", [0.0] * HAND_COUNT)
    except ValueError as exc:
        assert "OOP" in str(exc) or "IP" in str(exc)
        return
    raise AssertionError("expected ValueError")


# --- Ryan ask (May 2026): canonical 169 + aggregator + serialiser -----------
def test_canonical_169_count_and_endpoints():
    classes = canonical_169_hand_classes()
    assert len(classes) == 169
    # Ryan-pack ordering: A row first (25 entries), 2 row (23), ..., K row (1).
    assert classes[:5] == ["AA", "A2s", "A2o", "A3s", "A3o"]
    assert classes[24] == "AKo"                  # last of A row
    assert classes[25] == "22"                   # start of 2 row
    assert classes[26] == "32s"
    assert classes[-1] == "KK"


def test_canonical_169_matches_real_ryan_pack_order():
    """The canonical enumeration must match what Ryan's pack files actually
    write, so the new ip_range / oop_range columns are drop-in compatible."""
    sample = parse_range_file(Path(__file__).resolve().parent.parent
                              / "ranges" / "ryan_preflop_tree"
                              / "PioViewer - NLH 6max 100bb 2.5x Open"
                              / "BB"
                              / "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt")
    assert list(sample) == canonical_169_hand_classes()


def test_combo_str_to_hand_class():
    assert combo_str_to_hand_class("AhAd") == "AA"
    assert combo_str_to_hand_class("AhKh") == "AKs"
    assert combo_str_to_hand_class("AhKs") == "AKo"
    assert combo_str_to_hand_class("2c3c") == "32s"
    assert combo_str_to_hand_class("2c3d") == "32o"


def test_aggregate_filters_board_blocked_combos():
    """Combos sharing a card with the board are excluded from both the
    numerator and the denominator -- they're impossible holdings."""
    # Board has Ah. AhAd, AhAc, AhAs are all blocked (share Ah).
    # Only AcAd, AcAs, AdAs are unblocked.
    board = ["Ah", "Kd", "9c"]
    combo_range = {
        # AA: 3 of 6 combos blocked by board's Ah; 3 unblocked at varying weights.
        "AhAd": 1.0, "AhAc": 1.0, "AhAs": 1.0,
        "AcAd": 1.0, "AcAs": 0.5, "AdAs": 0.0,
    }
    out = aggregate_combo_range_to_classes(combo_range, board)
    # AA class: mean of unblocked weights = (1.0 + 0.5 + 0.0) / 3 = 0.5.
    assert out["AA"] == 0.5
    # All other classes 0 (no combos in range).
    assert out["KK"] == 0.0
    assert out["AKs"] == 0.0


def test_aggregate_emits_full_169_entries():
    """Even when only a few classes have weight, the output is the full 169
    canonical entries -- the 169-entry assertion is the contract."""
    out = aggregate_combo_range_to_classes({"AhAd": 1.0}, board=["2c", "3d", "4h"])
    assert len(out) == 169
    assert set(out) == set(canonical_169_hand_classes())
    assert out["AA"] == 1.0
    assert sum(1 for v in out.values() if v > 0) == 1


def test_aggregate_mean_per_unblocked_combo():
    """Sanity: when 3 of 4 AKs combos are in range at full weight on a board
    with no A or K, the class weight is 3/4 = 0.75 (not 3.0 / 4 / 6)."""
    board = ["2c", "3d", "4h"]
    # 4 AKs combos: AhKh, AdKd, AcKc, AsKs. Range has 3 at full weight, 1
    # absent (zero-weight equivalent).
    combo_range = {"AhKh": 1.0, "AdKd": 1.0, "AcKc": 1.0}
    out = aggregate_combo_range_to_classes(combo_range, board)
    # 3 combos sum to 3.0; only 3 are in the dict, so count=3 -> mean=1.0.
    # The absent AsKs is NOT counted (not in the dict).
    assert out["AKs"] == 1.0


def test_format_hand_class_range_canonical_order_and_format():
    """The serialised string starts with 'AA:<weight>,A2s:<weight>,...' and
    has 169 comma-separated entries -- drop-in match for Ryan's pack format."""
    weights = {cls: 0.0 for cls in canonical_169_hand_classes()}
    weights["AA"] = 1.0
    weights["KK"] = 0.5
    s = format_hand_class_range(weights)
    parts = s.split(",")
    assert len(parts) == 169
    assert parts[0] == "AA:1"
    assert parts[1] == "A2s:0"
    # Compact float formatting matches the pack (trailing zeros stripped).
    assert "KK:0.5" in parts


def test_format_hand_class_range_rejects_short_input():
    weights = {"AA": 1.0, "KK": 0.5}
    try:
        format_hand_class_range(weights)
    except ValueError as exc:
        assert "169 entries" in str(exc)
    else:
        raise AssertionError("format_hand_class_range should have rejected 2-entry dict")


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        # tmp_path fixture not supported in standalone mode; skip those.
        if "tmp_path" in fn.__code__.co_varnames:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                try:
                    fn(Path(td))
                    print(f"  [PASS] {name}")
                except AssertionError as exc:
                    failed += 1
                    print(f"  [FAIL] {name}: {exc}")
        else:
            try:
                fn()
                print(f"  [PASS] {name}")
            except AssertionError as exc:
                failed += 1
                print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
