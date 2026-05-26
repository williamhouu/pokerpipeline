"""Preflop question generation.

Subpackage for preflop-question generation, which reads from preflop range
packs (e.g. Ryan's 6-max 100bb 2.5x Open pack) and produces SpotData objects
the existing Layer 6 (explanation_generator) can write explanations for.

Architecture (each module is a step):

  * ``pack``                -- the ``PreflopPack`` dataclass and registry.
                                One pack per (vendor, table_size, stack_depth)
                                tuple. Future packs (9-max, MTT stacks, etc.)
                                register here too.
  * ``grammars/``           -- one filename-grammar parser per pack format.
                                Decouples "Ryan's UTG_60%_HJ_Fold..." from
                                future vendors' conventions.
  * ``node_enumerator``     -- walks one or more packs, returns the catalog
                                of distinct preflop decision nodes.
  * ``spot_sampler``        -- (node, hero hand) -> SpotData for Layer 6.
  * ``fact_extractor``      -- per-action frequencies, range shape, etc.
  * ``question_extractor``  -- worthiness + difficulty filter (frequency
                                window; no EV-gap filter pre-equity-engine).

See docs/ryan_range_pack_index.md for the Ryan pack's filename grammar.
"""

from pipeline.preflop.pack import (
    KNOWN_PACK_SIGNATURES,
    PreflopPack,
    all_packs,
    discover_packs,
    get_pack,
    register_pack,
)

__all__ = [
    "KNOWN_PACK_SIGNATURES",
    "PreflopPack",
    "all_packs",
    "discover_packs",
    "get_pack",
    "register_pack",
]
