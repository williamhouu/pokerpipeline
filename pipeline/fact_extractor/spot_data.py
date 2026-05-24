"""SpotData -- the structured data block every concept-tag function consumes.

This is the typed Layer 5 data block from docs/engineering_brief.docx
("Concept Tag Library Specification" -> "Data block format"). The fact
extractor populates a SpotData; each of the 42 concept-tag functions reads
fields off it and returns a bool, and `compute_tags` collects the firing tags.

    SpotData
      .spot_metadata       SpotMetadata        street, stacks, SPR, pot context
      .decision_data       DecisionData        options, EVs, strategy, line
      .equity_data         EquityData          equity, pot odds, EQR, MDF
      .range_data          RangeData           ranges, shapes, blockers, counts
      .hand_class          HandClass | None    from pipeline.fact_extractor.hand_class
      .board_texture       BoardTexture | None  from pipeline.fact_extractor.board_texture
      .population_baseline PopulationBaseline   baseline range for capped/uncapped
      .concept_tags        list[str]            filled by the tagger

Design notes:
  * Numeric thresholds belong in the tag rules, never here -- this module only
    defines the shape and validates it.
  * Fields carry both raw solver numbers and pre-computed strategic aggregates,
    per the brief ("both raw solver numbers AND strategic conclusions").
  * Sub-dataclasses validate their own fields on construction; SpotData
    serialises to/from a plain dict (the JSON data block) losslessly.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from pipeline.cards import parse_card
from pipeline.fact_extractor.board_texture import classify_board
from pipeline.fact_extractor.hand_class import classify_hand

STREETS = ("preflop", "flop", "turn", "river")
GAME_FORMATS = ("cash", "tournament")


# --- validation helpers ------------------------------------------------------
def _unit(name: str, value: float) -> None:
    """Require a value in [0, 1] (an equity, frequency, or percentage)."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _nonneg(name: str, value: float) -> None:
    """Require a value >= 0 (a stack, count, SPR, or pot fraction)."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")


def _unit_map(name: str, mapping: dict) -> None:
    """Require every value of a frequency map to lie in [0, 1]."""
    for key, value in mapping.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{key!r}] must be in [0, 1], got {value!r}")


# --- range primitives --------------------------------------------------------
@dataclass(frozen=True)
class Combo:
    """One hand combination in a range: its two cards, its weight in the range
    (0-1), and its equity vs the opponent's continuing range (0-1)."""

    cards: tuple[str, str]
    weight: float = 1.0
    equity: float = 0.0

    def __post_init__(self) -> None:
        cards = tuple(parse_card(c) for c in self.cards)
        if len(cards) != 2:
            raise ValueError(f"a combo has exactly 2 cards, got {self.cards!r}")
        if cards[0] == cards[1]:
            raise ValueError(f"a combo's two cards must differ: {self.cards!r}")
        object.__setattr__(self, "cards", cards)        # frozen -> setattr bypass
        _unit("Combo.weight", self.weight)
        _unit("Combo.equity", self.equity)

    @classmethod
    def from_dict(cls, data: dict) -> "Combo":
        return cls(cards=tuple(data["cards"]),
                   weight=data.get("weight", 1.0),
                   equity=data.get("equity", 0.0))


# --- references to the already-built Layer 5 descriptive modules -------------
@dataclass
class HandClass:
    """Hero's hand class -- the output shape of pipeline.fact_extractor.hand_class."""

    made_hand: str
    draws: list[str] = field(default_factory=list)
    strength_bucket: str = ""
    label: str = ""

    @classmethod
    def from_cards(cls, hole, board) -> "HandClass":
        """Classify hero's hand by calling the hand_class module."""
        return cls(**classify_hand(hole, board))

    @classmethod
    def from_dict(cls, data: dict) -> "HandClass":
        return cls(made_hand=data["made_hand"], draws=list(data.get("draws", [])),
                   strength_bucket=data.get("strength_bucket", ""),
                   label=data.get("label", ""))


@dataclass
class BoardTexture:
    """The board texture -- the output shape of pipeline.fact_extractor.board_texture."""

    suit_distribution: str
    pair_status: str
    connectedness: str
    rank_distribution: str
    composite: str

    @classmethod
    def from_board(cls, board) -> "BoardTexture":
        """Classify the board by calling the board_texture module."""
        return cls(**classify_board(board))

    @classmethod
    def from_dict(cls, data: dict) -> "BoardTexture":
        return cls(**data)


# --- population baseline -----------------------------------------------------
@dataclass
class PopulationBaseline:
    """The standard solver-derived range shape for a position + action context.

    Used by villain_capped / villain_uncapped (Section A). Real baselines are
    not wired yet: an unpopulated baseline (`populated=False`) is the default,
    and tags must fall back gracefully when they get one.
    """

    position_dynamic: str = ""
    action_context: str = ""
    top_5pct_combos: float = 0.0
    populated: bool = False

    def __post_init__(self) -> None:
        _nonneg("PopulationBaseline.top_5pct_combos", self.top_5pct_combos)

    @classmethod
    def from_dict(cls, data: dict) -> "PopulationBaseline":
        return cls(**data)


def lookup_population_baseline(position_dynamic: str = "",
                               action_context: str = "") -> PopulationBaseline:
    """Resolve the population baseline for a position + action context.

    Real baselines (standard solver-derived range shapes, indexed by position
    and action context, computed once from solver outputs) are not wired yet;
    until then this returns an empty, unpopulated baseline.
    """
    return PopulationBaseline(position_dynamic=position_dynamic,
                              action_context=action_context, populated=False)


# --- the four data-block sections --------------------------------------------
@dataclass
class SpotMetadata:
    """Structural facts about the spot: street, stacks, SPR, pot context."""

    street: str                                          # one of STREETS
    stack_depth_bb: float = 100.0
    effective_stack_bb: float = 100.0                    # short_stack_tournament
    spr: float = 0.0
    position_dynamic: str = ""                           # e.g. "BB_vs_BTN"
    hero_position: str = ""                              # e.g. "BB" (blind tags)
    villain_position: str = ""                           # blind_vs_blind_pot
    hero_in_position: bool = False                       # donk / pot_control
    hero_is_preflop_raiser: bool = False                 # donk / facing_donk
    game_format: str = "cash"                            # one of GAME_FORMATS
    preflop_raise_count: int = 0                         # SRP / 3bet / 4bet pot
    had_cold_caller: bool = False                        # squeeze_pot
    active_players_on_flop: int = 2                      # multiway_pot
    parent_node_id: str = ""                             # hand-replay metadata
    action_to_reach: str = ""                            # hand-replay metadata
    node_id: str = ""                                    # solver node id (solver_reference)
    hero_cards: tuple = ()                               # hero's two hole cards
    board: list = field(default_factory=list)            # community cards
    # Rendering context for the deterministic action-history block (Layer 8).
    # Populated by extract_facts; tag functions don't read these.
    action_sequence: list = field(default_factory=list)  # full (actor, label) path
    big_blind_chips: float = 1.0                         # chips per bb in this solve
    pot_bb: float = 0.0                                  # pot at the decision in bb
    # Ryan-feedback Fix 5 (May 2026): per-street aggressor history. Excludes
    # preflop (postflop solves don't carry preflop action). Values per entry:
    # "hero" / "villain" / "check" (no bets that street). One entry per
    # completed prior street; current-street entry not included.
    aggression_history: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hero_cards = tuple(self.hero_cards)
        self.board = list(self.board)
        self.action_sequence = list(self.action_sequence)
        if self.big_blind_chips <= 0:
            raise ValueError(f"big_blind_chips must be > 0, "
                             f"got {self.big_blind_chips!r}")
        _nonneg("SpotMetadata.pot_bb", self.pot_bb)
        if self.street not in STREETS:
            raise ValueError(f"street must be one of {STREETS}, got {self.street!r}")
        if self.game_format not in GAME_FORMATS:
            raise ValueError(f"game_format must be one of {GAME_FORMATS}, "
                             f"got {self.game_format!r}")
        for name in ("stack_depth_bb", "effective_stack_bb", "spr"):
            _nonneg(f"SpotMetadata.{name}", getattr(self, name))
        if self.preflop_raise_count < 0:
            raise ValueError("preflop_raise_count must be >= 0, "
                             f"got {self.preflop_raise_count}")
        if self.active_players_on_flop < 2:
            raise ValueError("active_players_on_flop must be >= 2, "
                             f"got {self.active_players_on_flop}")

    @classmethod
    def from_dict(cls, data: dict) -> "SpotMetadata":
        return cls(**data)


@dataclass
class DecisionData:
    """The decision: available options, EVs, strategy, and the action line."""

    options: list[str] = field(default_factory=list)
    # range_mean_evs_per_action: PioSolver's range-weighted mean EV at each
    # action's CHILD node, keyed by canonical verb ("call", "bet", ...). Values
    # are bb-denominated. This is the AGGREGATE across hero's range -- not the
    # specific hero combo's EV. The May-2026 audit caught the LLM citing these
    # as combo-specific ("Ks3s has EV +27bb"); the explicit name + a doc-block
    # in Layer 6's prompt make the aggregate framing unambiguous.
    range_mean_evs_per_action: dict[str, float] = field(default_factory=dict)
    hero_combo_strategy: dict[str, float] = field(default_factory=dict)
    range_aggregate_strategy: dict[str, float] = field(default_factory=dict)
    correct_action: str = ""
    ev_gap_bb: float = 0.0
    ev_loss_per_action_bb: dict[str, float] = field(default_factory=dict)
    decision_class: str = ""                             # optional precomputed
    # Bet sizing -- overbet / merged_value / facing_overbet (Section C / B).
    option_pot_fractions: dict[str, float] = field(default_factory=dict)
    facing_bet_pot_fraction: float | None = None
    # Action line -- the action-type spots (Section C). Each entry is
    # (actor, action), actor in {"hero", "villain"}.
    street_actions: list[tuple[str, str]] = field(default_factory=list)
    prior_street_actions: list[tuple[str, str]] = field(default_factory=list)
    # Commitment -- commitment_threshold_decision (Section B).
    post_decision_spr: float = 0.0
    stack_committed_fraction: float = 0.0
    # Decision-class signals -- the remaining Section B tags.
    hero_has_later_street_value: bool = True             # bluffcatch_spot
    estimated_fold_equity: float = 0.0                   # bluff_spot
    estimated_showdown_value: float = 0.0                # bluff_spot
    check_ev_vs_villain_range: float = 0.0               # equity_denial_spot
    bet_ev_vs_calling_range: float = 0.0                 # equity_denial_spot
    float_line_is_positive: bool = False                 # float_call_spot
    # Ryan-feedback Fix 5 (May 2026): the strategic archetype of the correct
    # action -- one of the 13 archetypes Layer 6 uses to frame the explanation.
    # See pipeline.fact_extractor.archetypes.RECOMMENDED_ACTION_ARCHETYPES.
    recommended_action_archetype: str = ""

    def __post_init__(self) -> None:
        _unit_map("hero_combo_strategy", self.hero_combo_strategy)
        _unit_map("range_aggregate_strategy", self.range_aggregate_strategy)
        for key, value in self.option_pot_fractions.items():
            _nonneg(f"option_pot_fractions[{key!r}]", value)
        if self.facing_bet_pot_fraction is not None:
            _nonneg("facing_bet_pot_fraction", self.facing_bet_pot_fraction)
        _nonneg("post_decision_spr", self.post_decision_spr)
        _unit("stack_committed_fraction", self.stack_committed_fraction)
        self.street_actions = [tuple(a) for a in self.street_actions]
        self.prior_street_actions = [tuple(a) for a in self.prior_street_actions]
        for action in self.street_actions + self.prior_street_actions:
            if len(action) != 2:
                raise ValueError("each action entry must be (actor, action), "
                                 f"got {action!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionData":
        return cls(**data)


@dataclass
class EquityData:
    """Equity and pot-math facts -- equity, pot odds, realization, MDF."""

    hero_raw_equity_vs_continuing: float = 0.0
    hero_raw_equity_vs_calling: float = 0.0              # thin_value_spot
    pot_odds_required: float = 0.0
    mdf: float = 0.0                                     # mdf_defense_threshold
    equity_realization_ratio: float = 1.0               # EQR; 1.0 = neutral
    equity_realization_gap: float = 0.0

    def __post_init__(self) -> None:
        for name in ("hero_raw_equity_vs_continuing", "hero_raw_equity_vs_calling",
                     "pot_odds_required", "mdf"):
            _unit(f"EquityData.{name}", getattr(self, name))
        _nonneg("EquityData.equity_realization_ratio", self.equity_realization_ratio)
        if not -1.0 <= self.equity_realization_gap <= 1.0:
            raise ValueError("equity_realization_gap must be in [-1, 1], "
                             f"got {self.equity_realization_gap!r}")

    @classmethod
    def from_dict(cls, data: dict) -> "EquityData":
        return cls(**data)


@dataclass
class RangeData:
    """Range facts -- the raw ranges plus pre-computed strategic aggregates."""

    # Raw ranges (combo-level). Empty until the solver output is wired in.
    villain_range: list[Combo] = field(default_factory=list)
    hero_range: list[Combo] = field(default_factory=list)
    # Range shape -- villain_polarized / villain_linear (Section A).
    villain_range_shape: str = ""
    # Combo counts -- blocker math context (Section D).
    villain_value_combos: float = 0.0
    villain_bluff_combos: float = 0.0
    # Blocker percentages -- Section D tags read these directly.
    hero_blocks_value_pct: float = 0.0
    hero_blocks_bluffs_pct: float = 0.0
    # Range advantage -- Section E tags.
    hero_total_equity: float = 0.0
    villain_total_equity: float = 0.0
    hero_strong_hand_count: float = 0.0
    villain_strong_hand_count: float = 0.0
    hero_top_5pct_combos: float = 0.0
    villain_top_5pct_combos: float = 0.0
    # Decision-class context -- Section B tags.
    villain_draw_equity_pct: float = 0.0                 # protection_bet_spot
    villain_calling_range_worse_pct: float = 0.0         # thin_value_spot
    villain_calling_range_has_worse_made_hands: bool = False   # merged_value_spot
    villain_calling_range_has_draws: bool = False              # merged_value_spot
    # Ryan-feedback Fix 4 (May 2026): top hand classes in villain's continuing
    # range, ranked by total_weight * strength_bucket_score, with 2-3 example
    # combos per class formatted in the emoji notation Layer 6's prose uses.
    # Layer 6 reads this to name 2-3 specific villain combos in answer_
    # explanation instead of describing villain's range abstractly. Each entry:
    #   {
    #     "hand_class_label": "set_no_draws",
    #     "bucket": "premium",
    #     "total_weight": 3.5,
    #     "combo_count": 6,
    #     "example_combos": ["K♠️K♣️", ...],
    #   }
    villain_top_value_combos: list[dict] = field(default_factory=list)
    # Ryan-feedback Fix 5 (May 2026): hero's range disposition at this node.
    # "capped" -- hero has very few nut-tier combos in range (cannot have
    # premium hands here); "uncapped" -- hero's range contains nut combos;
    # "polarized" -- hero's range is mostly nuts + mostly air, little middle.
    # Layer 6 uses this to frame whether hero's range supports a value-bet,
    # a bluff_catch, etc.
    hero_range_disposition: str = ""

    def __post_init__(self) -> None:
        self.villain_range = [c if isinstance(c, Combo) else Combo.from_dict(c)
                              for c in self.villain_range]
        self.hero_range = [c if isinstance(c, Combo) else Combo.from_dict(c)
                           for c in self.hero_range]
        for name in ("hero_blocks_value_pct", "hero_blocks_bluffs_pct",
                     "hero_total_equity", "villain_total_equity",
                     "villain_draw_equity_pct", "villain_calling_range_worse_pct"):
            _unit(f"RangeData.{name}", getattr(self, name))
        for name in ("villain_value_combos", "villain_bluff_combos",
                     "hero_strong_hand_count", "villain_strong_hand_count",
                     "hero_top_5pct_combos", "villain_top_5pct_combos"):
            _nonneg(f"RangeData.{name}", getattr(self, name))

    @classmethod
    def from_dict(cls, data: dict) -> "RangeData":
        return cls(**data)


# --- the data block ----------------------------------------------------------
@dataclass
class SpotData:
    """The full Layer 5 data block: input to every concept-tag function."""

    spot_metadata: SpotMetadata
    decision_data: DecisionData = field(default_factory=DecisionData)
    equity_data: EquityData = field(default_factory=EquityData)
    range_data: RangeData = field(default_factory=RangeData)
    hand_class: HandClass | None = None                  # None for preflop spots
    board_texture: BoardTexture | None = None            # None for preflop spots
    population_baseline: PopulationBaseline = field(default_factory=PopulationBaseline)
    concept_tags: list[str] = field(default_factory=list)   # filled by the tagger

    def __post_init__(self) -> None:
        sections = {"spot_metadata": SpotMetadata, "decision_data": DecisionData,
                    "equity_data": EquityData, "range_data": RangeData,
                    "population_baseline": PopulationBaseline}
        for name, expected in sections.items():
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"SpotData.{name} must be a {expected.__name__}")
        if self.hand_class is not None and not isinstance(self.hand_class, HandClass):
            raise TypeError("SpotData.hand_class must be a HandClass or None")
        if self.board_texture is not None and not isinstance(self.board_texture,
                                                             BoardTexture):
            raise TypeError("SpotData.board_texture must be a BoardTexture or None")
        if not all(isinstance(tag, str) for tag in self.concept_tags):
            raise ValueError("concept_tags must be a list of strings")

    # -- serialization --------------------------------------------------------
    def to_dict(self) -> dict:
        """The data block as a plain (JSON-ready) nested dict."""
        return asdict(self)

    def to_json(self, **kwargs) -> str:
        """The data block as a JSON string."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict) -> "SpotData":
        """Rebuild a SpotData from a data-block dict (the inverse of to_dict)."""
        hand_class = data.get("hand_class")
        board_texture = data.get("board_texture")
        return cls(
            spot_metadata=SpotMetadata.from_dict(data["spot_metadata"]),
            decision_data=DecisionData.from_dict(data.get("decision_data", {})),
            equity_data=EquityData.from_dict(data.get("equity_data", {})),
            range_data=RangeData.from_dict(data.get("range_data", {})),
            hand_class=HandClass.from_dict(hand_class) if hand_class else None,
            board_texture=(BoardTexture.from_dict(board_texture)
                           if board_texture else None),
            population_baseline=PopulationBaseline.from_dict(
                data.get("population_baseline", {})),
            concept_tags=list(data.get("concept_tags", [])),
        )

    @classmethod
    def from_json(cls, text: str) -> "SpotData":
        """Rebuild a SpotData from a JSON data block."""
        return cls.from_dict(json.loads(text))
