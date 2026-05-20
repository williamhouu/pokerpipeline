"""Concept tag registry and the compute_tags entry point (Layer 5).

TAG_REGISTRY maps each of the 42 concept-tag names (Sections A-G of
docs/engineering_brief.docx, "Concept Tag Library Specification") to its tag
function. compute_tags runs every tag against a SpotData and returns the names
that fire -- the concept_tags list for the spot's data block.

Section H (feel_based_decision, exploitative_deviation, icm_pressure_spot) is
intentionally excluded: the brief defers it ("ignore these at first") because
those tags need composite rules, population data, or an ICM model.

Adding a tag is two lines: write the function in its section module, add it to
TAG_REGISTRY. Each tag stays a pure, independently unit-tested function.
"""
from __future__ import annotations

from collections.abc import Callable

from pipeline.fact_extractor.concept_tags.section_a_range import (
    villain_capped, villain_linear, villain_polarized, villain_uncapped,
)
from pipeline.fact_extractor.concept_tags.section_b_decision_class import (
    bluff_spot, bluffcatch_spot, commitment_threshold_decision,
    equity_denial_spot, float_call_spot, merged_value_spot, pot_control_spot,
    protection_bet_spot, thin_value_spot,
)
from pipeline.fact_extractor.concept_tags.section_c_action_types import (
    check_raise_spot, donk_bet_spot, facing_check_raise_spot, facing_donk_spot,
    facing_overbet_spot, facing_probe_spot, overbet_spot, probe_bet_spot,
)
from pipeline.fact_extractor.concept_tags.section_d_blockers import (
    blocks_bluffs_unblocks_value, blocks_value, blocks_value_unblocks_bluffs,
    no_blocker_effects,
)
from pipeline.fact_extractor.concept_tags.section_e_range_advantage import (
    nut_advantage_hero, nut_advantage_villain, range_advantage_hero,
    range_advantage_villain,
)
from pipeline.fact_extractor.concept_tags.section_f_equity_math import (
    equity_over_realized, equity_under_realized, implied_odds_call,
    mdf_defense_threshold, reverse_implied_odds_call,
)
from pipeline.fact_extractor.concept_tags.section_g_pot_context import (
    blind_defense_spot, blind_vs_blind_pot, four_bet_pot, multiway_pot,
    short_stack_tournament, single_raised_pot, squeeze_pot, three_bet_pot,
)
from pipeline.fact_extractor.spot_data import SpotData

# Tag name -> tag function, in the brief's section order (A-G). compute_tags
# iterates this dict, so the order here is the order of the emitted tag list.
# "3bet_pot" / "4bet_pot" are not valid identifiers, hence the three_bet_pot /
# four_bet_pot functions behind those names.
TAG_REGISTRY: dict[str, Callable[[SpotData], bool]] = {
    # Section A -- Range Characterization
    "villain_capped": villain_capped,
    "villain_uncapped": villain_uncapped,
    "villain_polarized": villain_polarized,
    "villain_linear": villain_linear,
    # Section B -- Decision Class
    "thin_value_spot": thin_value_spot,
    "bluffcatch_spot": bluffcatch_spot,
    "protection_bet_spot": protection_bet_spot,
    "merged_value_spot": merged_value_spot,
    "pot_control_spot": pot_control_spot,
    "equity_denial_spot": equity_denial_spot,
    "bluff_spot": bluff_spot,
    "commitment_threshold_decision": commitment_threshold_decision,
    "float_call_spot": float_call_spot,
    # Section C -- Postflop Action-Type Spots
    "check_raise_spot": check_raise_spot,
    "facing_check_raise_spot": facing_check_raise_spot,
    "donk_bet_spot": donk_bet_spot,
    "facing_donk_spot": facing_donk_spot,
    "probe_bet_spot": probe_bet_spot,
    "facing_probe_spot": facing_probe_spot,
    "overbet_spot": overbet_spot,
    "facing_overbet_spot": facing_overbet_spot,
    # Section D -- Blocker Effects
    "blocks_value_unblocks_bluffs": blocks_value_unblocks_bluffs,
    "blocks_bluffs_unblocks_value": blocks_bluffs_unblocks_value,
    "blocks_value": blocks_value,
    "no_blocker_effects": no_blocker_effects,
    # Section E -- Range Advantage
    "range_advantage_hero": range_advantage_hero,
    "range_advantage_villain": range_advantage_villain,
    "nut_advantage_hero": nut_advantage_hero,
    "nut_advantage_villain": nut_advantage_villain,
    # Section F -- Equity & Math Concepts
    "equity_under_realized": equity_under_realized,
    "equity_over_realized": equity_over_realized,
    "implied_odds_call": implied_odds_call,
    "reverse_implied_odds_call": reverse_implied_odds_call,
    "mdf_defense_threshold": mdf_defense_threshold,
    # Section G -- Pot and Action Context
    "3bet_pot": three_bet_pot,
    "4bet_pot": four_bet_pot,
    "single_raised_pot": single_raised_pot,
    "squeeze_pot": squeeze_pot,
    "multiway_pot": multiway_pot,
    "blind_vs_blind_pot": blind_vs_blind_pot,
    "blind_defense_spot": blind_defense_spot,
    "short_stack_tournament": short_stack_tournament,
}


def compute_tags(spot_data: SpotData) -> list[str]:
    """Run every concept tag over a spot; return the names that fire.

    The result is the `concept_tags` list for the spot's data block, ordered by
    the brief's sections A-G.
    """
    return [name for name, tag_fn in TAG_REGISTRY.items() if tag_fn(spot_data)]
