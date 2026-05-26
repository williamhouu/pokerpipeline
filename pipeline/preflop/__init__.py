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
  * ``gold_examples``       -- filters the shared xlsx gold pool down to
                                preflop-only rows for Layer 6's prompt.
  * ``explanation_generator`` -- Layer 6 preflop branch: turns a
                                ``PreflopFacts`` into the six LLM-written
                                CSV columns. Sibling of
                                ``pipeline.explanation_generator``.
  * ``format_writer``       -- Layer 8 preflop branch: turns a
                                ``PreflopFacts`` + ``GeneratedExplanation``
                                into a 38-column CSV row. Sibling of
                                ``pipeline.format_writer``.

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
