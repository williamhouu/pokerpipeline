"""Tests for pipeline.scenario_config (Layer 1).

Run directly (`python tests/test_scenario_config.py`) or under pytest. Covers
the registry lookup, the registered btn_vs_bb_srp_2cJs7s scenario's field
values, and the spot_to_hand bridge into the action-history renderer.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.action_history import format_action_history             # noqa: E402
from pipeline.fact_extractor.spot_data import (                       # noqa: E402
    DecisionData, SpotData, SpotMetadata,
)
from pipeline.scenario_config import (                                # noqa: E402
    COMMON_STAKE_LEVELS_BB_DOLLARS, SCENARIOS, ScenarioConfig, get_scenario,
    round_to_nearest_increment, scale_scenario, spot_to_hand,
)


def _spot(action_sequence, board, hero_cards=("Ah", "Kh"),
          hero_position="BB", villain_position="BTN",
          big_blind_chips=87.75, pot_bb=5.5) -> SpotData:
    """A populated SpotData fixture sized to the btn_vs_bb test solve."""
    street = {3: "flop", 4: "turn", 5: "river"}[len(board)]
    return SpotData(
        SpotMetadata(
            street=street, hero_position=hero_position,
            villain_position=villain_position,
            position_dynamic=f"{hero_position}_vs_{villain_position}",
            hero_is_preflop_raiser=(hero_position == "BTN"),
            game_format="cash", preflop_raise_count=1,
            stack_depth_bb=100.0, effective_stack_bb=97.5,
            hero_cards=hero_cards, board=board,
            action_sequence=action_sequence,
            big_blind_chips=big_blind_chips, pot_bb=pot_bb,
        ),
        decision_data=DecisionData(correct_action="check"),
    )


# --- registry lookup ---------------------------------------------------------
def test_btn_vs_bb_srp_registered():
    """The Tier 1 test solve is registered with exactly the user-specified fields."""
    s = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    assert s.format == "Cash 6-max"
    assert s.stakes == "$0.25/$0.50"
    assert s.table_size == 6
    assert s.default_stack_bb == 100
    assert s.default_stack_dollars == 50.00
    assert s.live_or_online == "Online"
    assert s.preflop_action == "BTN open 2.5bb, BB call"
    assert s.oop_position == "BB" and s.ip_position == "BTN"
    assert s.game_format == "cash" and s.venue == "online"
    # Derived context string -- matches sample row 1's online-cash format.
    # Whole-dollar stack drops the trailing '.00' per Ryan-feedback Fix 1
    # (May 2026); a non-integer stack would still render as e.g. "Stacks $52.50".
    assert s.context == "6-Handed, $0.25/$0.50, Stacks $50"
    # $/bb = 50/100 = 0.5
    assert s.dollars_per_bb == 0.5


def test_get_scenario_accepts_path_and_string():
    """Lookup works from a Path, a path string, or a bare key -- stem is the key."""
    expected = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    assert get_scenario("btn_vs_bb_srp_2cJs7s") is expected
    assert get_scenario("btn_vs_bb_srp_2cJs7s.cfr") is expected
    assert get_scenario(Path("test_solves/btn_vs_bb_srp_2cJs7s.cfr")) is expected
    assert get_scenario(Path("/some/other/dir/btn_vs_bb_srp_2cJs7s.cfr")) is expected


def test_get_scenario_unknown_raises_clear_error():
    """An unregistered solve produces a KeyError naming the registered solves."""
    try:
        get_scenario("not_a_real_solve")
    except KeyError as exc:
        message = str(exc)
        assert "not_a_real_solve" in message
        assert "btn_vs_bb_srp_2cJs7s" in message
        assert "pipeline/scenario_config.py" in message
        return
    raise AssertionError("expected KeyError")


# --- ScenarioConfig validation ----------------------------------------------
def test_invalid_game_format_rejected():
    try:
        ScenarioConfig(cfr_key="x", format="x", stakes="x", live_or_online="Online",
                       preflop_action="x", game_format="zynga", stakes_sb=0.25,
                       stakes_bb=0.50, table_size=6, default_stack_bb=100,
                       default_stack_dollars=50.0, venue="online",
                       oop_position="BB", ip_position="BTN")
    except ValueError as exc:
        assert "game_format" in str(exc)
        return
    raise AssertionError("expected ValueError")


def test_oop_ip_must_differ():
    try:
        ScenarioConfig(cfr_key="x", format="x", stakes="x", live_or_online="Online",
                       preflop_action="x", game_format="cash", stakes_sb=0.25,
                       stakes_bb=0.50, table_size=6, default_stack_bb=100,
                       default_stack_dollars=50.0, venue="online",
                       oop_position="BB", ip_position="BB")
    except ValueError as exc:
        assert "must differ" in str(exc)
        return
    raise AssertionError("expected ValueError")


# --- round_to_nearest_increment ---------------------------------------------
def test_round_to_nearest_increment_ryan_feedback_examples():
    """The exact examples Ryan listed in his Apr-2026 V6 review."""
    sb = 0.25
    assert round_to_nearest_increment(1.85, sb) == 1.75
    assert round_to_nearest_increment(5.23, sb) == 5.25
    assert round_to_nearest_increment(12.15, sb) == 12.25
    assert round_to_nearest_increment(26.26, sb) == 26.25


def test_round_to_nearest_increment_below_increment():
    """Sub-SB amounts round to either 0 or one increment, depending on side."""
    assert round_to_nearest_increment(0.10, 0.25) == 0.0
    assert round_to_nearest_increment(0.13, 0.25) == 0.25
    # The conversion in _convert_postflop clamps the result to at least one
    # increment, so a sub-SB raw amount still renders as a real wager.


def test_round_to_nearest_increment_rejects_nonpositive_increment():
    try:
        round_to_nearest_increment(1.85, 0)
    except ValueError as exc:
        assert "increment" in str(exc).lower()
        return
    raise AssertionError("expected ValueError")


def test_convert_postflop_rounds_dollars_to_sb():
    """The Ryan-feedback example values from his V6 review, threaded through
    the production conversion pipeline. Chip amounts chosen so the raw
    $-conversion lands on each of the four examples Ryan listed."""
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # chips_per_bb = 87.75. dollars_per_bb = 0.50. So
    #   chips * (0.50 / 87.75) = dollars.
    # Pick chips so raw dollars match Ryan's examples.
    cases = [
        (324.7, 1.85, 1.75),    # 1.85 -> 1.75
        (917.85, 5.23, 5.25),   # 5.23 -> 5.25
        (2132.3, 12.15, 12.25), # 12.15 -> 12.25
        (4608.4, 26.26, 26.25), # 26.26 -> 26.25
    ]
    for chips, raw, snapped in cases:
        # Verify our chip choices reproduce the raw dollar amounts to 2 decimals.
        derived_raw = chips / 87.75 * 0.50
        assert abs(derived_raw - raw) < 0.01, f"{chips} chips -> {derived_raw} != {raw}"
        # Then verify the conversion snaps to the expected SB multiple.
        from pipeline.scenario_config import _convert_postflop
        out = _convert_postflop([("IP", f"bet {chips}")], scenario, 87.75)
        assert out[0] == ("BTN", "bet", snapped), \
            f"{chips} chips: expected snap to {snapped}, got {out[0]}"


# --- spot_to_hand bridge -----------------------------------------------------
def test_spot_to_hand_basic_shape():
    """spot_to_hand produces a dict format_action_history can render -- no error."""
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # A simple flop-only line: hero (BB=OOP) checks, BTN (IP) bets 41 chips.
    action_sequence = [("OOP", "check"), ("IP", "bet 41")]
    spot = _spot(action_sequence, board=["2c", "Js", "7s"])
    hand = spot_to_hand(spot, scenario)
    assert hand["stakes"] == {"sb": 0.25, "bb": 0.50}
    assert hand["format"] == "cash"
    assert hand["venue"] == "online"
    assert hand["table_size"] == 6
    assert hand["effective_stack"] == 50.00
    assert hand["hero_position"] == "BB"
    assert hand["preflop_actions"] == [("BTN", "open", 1.25), ("BB", "call")]
    assert hand["board"] == {"flop": ["2c", "Js", "7s"], "turn": None, "river": None}
    # OOP -> BB, IP -> BTN, label parsed and chips converted to dollars.
    # 41 chips / 87.75 chips/bb * $0.50/bb = $0.234, rounded to nearest
    # small blind ($0.25) per Ryan's Apr-2026 feedback so the prose reads
    # like a real wager. The smallest-rounded-up clamp also kicks in so a
    # below-SB bet still renders as exactly one SB.
    assert hand["flop_actions"] == [("BB", "check"), ("BTN", "bet", 0.25)]
    assert hand["turn_actions"] == []
    assert hand["river_actions"] == []


def test_spot_to_hand_river_line_renders_end_to_end():
    """The full hand dict feeds into format_action_history without error and
    produces the team's voice ('You're in the Big Blind with A<heart>K<heart>.')."""
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # Multi-street river line.
    action_sequence = [
        ("OOP", "check"), ("IP", "check"),         # flop check-check
        ("deal", "8h"),                            # turn deal
        ("OOP", "check"), ("IP", "bet 38"),        # turn cb
        ("OOP", "call"),
        ("deal", "As"),                            # river deal
        ("OOP", "check"),                          # river check (hero's spot)
    ]
    spot = _spot(action_sequence,
                 board=["2c", "Js", "7s", "8h", "As"], hero_cards=("Ah", "Kh"))
    hand = spot_to_hand(spot, scenario)
    rendered = format_action_history(hand)
    assert rendered.startswith("You're in the Big Blind with A")
    # Postflop action lines reference dollar amounts. A bet of 38 chips
    # is $0.22 raw -> snaps to $0.25 (one SB) after the Ryan-feedback
    # round-to-nearest-SB step.
    assert "$0.25" in rendered, rendered
    # Three street sections appear (flop / turn / river).
    assert "Flop" in rendered and "Turn" in rendered and "River" in rendered


def test_spot_to_hand_uses_ip_when_hero_is_ip():
    """When hero is the IP side (BTN), 'You' refers to BTN in the rendered text."""
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # IP-side hero -- BTN is hero, so OOP actions are villain (BB).
    spot = _spot([("OOP", "check"), ("IP", "bet 41")],
                 board=["2c", "Js", "7s"],
                 hero_cards=("Ah", "Kh"),
                 hero_position="BTN", villain_position="BB")
    hand = spot_to_hand(spot, scenario)
    rendered = format_action_history(hand)
    assert rendered.startswith("You're on the Button with A")
    # The Big Blind acted first on the flop -- with a check.
    assert "The Big Blind checks." in rendered
    # Hero (you) bet on the flop.
    assert "You bet" in rendered


def test_spot_to_hand_requires_big_blind_chips():
    """A spot built without big_blind_chips would silently divide by zero --
    spot_to_hand catches that with a clear error."""
    scenario = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # Force big_blind_chips to 0 by bypassing the validator.
    spot = _spot([("OOP", "check")], board=["2c", "Js", "7s"])
    object.__setattr__(spot.spot_metadata, "big_blind_chips", 0.0)
    try:
        spot_to_hand(spot, scenario)
    except ValueError as exc:
        assert "big_blind_chips" in str(exc)
        return
    raise AssertionError("expected ValueError")


# --- scale_scenario ---------------------------------------------------------
def test_scale_scenario_basic_5_10():
    """$0.25/$0.50 default scaled to $5/$10. Ratio is 20x."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    scaled = scale_scenario(base, target_bb_dollars=10.0)
    assert scaled.stakes_sb == 5.0
    assert scaled.stakes_bb == 10.0
    assert scaled.default_stack_dollars == 1000.0   # 100bb * $10
    assert scaled.default_stack_bb == 100            # unchanged (stack depth in bb)
    assert scaled.stakes == "$5/$10"
    assert scaled.context == "6-Handed, $5/$10, Stacks $1,000"


def test_scale_scenario_preflop_actions_scaled():
    """The dollar amounts inside preflop_actions tuples get rescaled too."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    # base has preflop_actions=(("BTN", "open", 1.25), ("BB", "call"))
    scaled = scale_scenario(base, target_bb_dollars=2.0)   # 4x ratio
    actions = list(scaled.preflop_actions)
    assert actions[0] == ("BTN", "open", 5.0)    # 1.25 * 4 = 5.0
    assert actions[1] == ("BB", "call")           # 2-tuple unchanged


def test_scale_scenario_dollars_per_bb_property_updates():
    """The derived `dollars_per_bb` property reflects the scaled stack."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    scaled = scale_scenario(base, target_bb_dollars=2.0)
    assert scaled.dollars_per_bb == 2.0           # was 0.50


def test_scale_scenario_preserves_non_dollar_fields():
    """cfr_key, positions, table_size, game_format, etc. unchanged."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    scaled = scale_scenario(base, target_bb_dollars=10.0)
    assert scaled.cfr_key == base.cfr_key
    assert scaled.oop_position == base.oop_position
    assert scaled.ip_position == base.ip_position
    assert scaled.table_size == base.table_size
    assert scaled.game_format == base.game_format
    assert scaled.preflop_action == base.preflop_action
    assert scaled.live_or_online == base.live_or_online


def test_scale_scenario_identity_scaling():
    """Scaling to the same BB returns equivalent values (no math drift)."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    same = scale_scenario(base, target_bb_dollars=base.stakes_bb)
    assert same.stakes_sb == base.stakes_sb
    assert same.stakes_bb == base.stakes_bb
    assert same.default_stack_dollars == base.default_stack_dollars
    assert same.stakes == base.stakes


def test_scale_scenario_micro_stakes_rounding():
    """Scaling down to micro stakes — preflop bet amounts round to cents."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    scaled = scale_scenario(base, target_bb_dollars=0.10)   # 0.2x ratio
    assert scaled.stakes_sb == 0.05
    assert scaled.stakes_bb == 0.10
    assert scaled.default_stack_dollars == 10.0   # 100bb * $0.10
    # preflop_actions: 1.25 * 0.2 = 0.25
    actions = list(scaled.preflop_actions)
    assert actions[0] == ("BTN", "open", 0.25)


def test_scale_scenario_high_stakes():
    """$50/$100 stakes — biggest in COMMON_STAKE_LEVELS_BB_DOLLARS."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    scaled = scale_scenario(base, target_bb_dollars=100.0)
    assert scaled.stakes_sb == 50.0
    assert scaled.stakes_bb == 100.0
    assert scaled.default_stack_dollars == 10_000.0   # 100bb * $100
    assert scaled.stakes == "$50/$100"


def test_scale_scenario_rejects_nonpositive():
    """Zero or negative target raises ValueError with a clear message."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    for bad in (0, -1, -0.50):
        try:
            scale_scenario(base, target_bb_dollars=bad)
        except ValueError as exc:
            assert "target_bb_dollars" in str(exc)
            continue
        raise AssertionError(
            f"expected ValueError for target_bb_dollars={bad}"
        )


def test_scale_scenario_returns_frozen_dataclass():
    """The returned config is still frozen (no accidental mutation)."""
    base = SCENARIOS["btn_vs_bb_srp_2cJs7s"]
    scaled = scale_scenario(base, target_bb_dollars=10.0)
    try:
        scaled.stakes_bb = 5.0  # type: ignore[misc]
    except dataclasses_FrozenInstanceError:
        return
    raise AssertionError("expected FrozenInstanceError")


# Imported lazily so the import doesn't shadow the dataclasses module in the
# rest of the file.
from dataclasses import FrozenInstanceError as dataclasses_FrozenInstanceError  # noqa: E402


def test_common_stake_levels_constant():
    """The advertised stake levels are all positive, sorted, and unique."""
    levels = COMMON_STAKE_LEVELS_BB_DOLLARS
    assert len(levels) >= 5                        # enough for a real dropdown
    assert all(level > 0 for level in levels)
    assert list(levels) == sorted(levels)          # ascending order
    assert len(set(levels)) == len(levels)         # unique
    # Tier-1 default is in the list.
    assert 0.50 in levels


if __name__ == "__main__":
    suite = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failed = 0
    for name, fn in suite:
        try:
            fn()
            print(f"  [PASS] {name}")
        except AssertionError as exc:
            failed += 1
            print(f"  [FAIL] {name}: {exc}")
    print(f"\n{len(suite) - failed}/{len(suite)} tests passed")
    sys.exit(1 if failed else 0)
