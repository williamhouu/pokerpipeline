"""Layer 1: Scenario Config Registry.

A solved `.cfr` file only carries the postflop game tree -- it doesn't know the
preflop action line that produced it, the table size, the blind level, or
whether the table is online or live. Without those facts, Layer 8 cannot render
the Context, Question, Table Size, Default Stack, Live or Online, Seats, or
POT columns from real data; they have to be left as `[TBD]` placeholders.

This module is the registry: one `ScenarioConfig` per solve, keyed by the
`.cfr` filename (without the extension). Layer 1 looks the scenario up;
`spot_to_hand` translates a populated SpotData + its scenario into the hand
dict that `pipeline.action_history.format_action_history` consumes, so the
deterministic action-history block can be written into the Question column.

Adding a new solve is a registry edit: register one `ScenarioConfig` here and
the pipeline picks it up the next time the solve is generated against.

See docs/engineering_brief.docx, "Layer 1: Spot Generator", for the scenario
concept; this module is the minimal Phase-1 implementation -- one scenario
per solve. Wider tiers (multiway, tournament) extend the dataclass; the
registry mechanism does not change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from pipeline.fact_extractor.spot_data import SpotData


@dataclass(frozen=True)
class ScenarioConfig:
    """Per-solve metadata that the postflop `.cfr` file does not carry.

    Fields fall into three groups:

      * Display strings -- the exact text the CSV columns expect. Stored
        pre-formatted so the format_writer doesn't have to re-derive them.
      * Parsed numerics -- the same facts in machine form, for the action
        history renderer (which composes its own strings from sb/bb,
        preflop_actions, etc.).
      * Solve geometry -- which seat is OOP / IP in the solver tree, so the
        path sampler knows what to label OOP/IP actions as.

    Frozen so it can be reused freely across spots without accidental mutation.
    """

    # Registry key -- the `.cfr` filename without the extension.
    cfr_key: str

    # Display strings (CSV columns).
    format: str                                  # "Cash 6-max"
    stakes: str                                  # "$0.25/$0.50"
    live_or_online: str                          # "Online" | "Live" | "Not specified"
    preflop_action: str                          # "BTN open 2.5bb, BB call"

    # Parsed numerics (renderer + math).
    game_format: str                             # "cash" | "tournament"
    stakes_sb: float                             # 0.25
    stakes_bb: float                             # 0.50
    table_size: int                              # 6
    default_stack_bb: int                        # 100
    default_stack_dollars: float                 # 50.00
    venue: str                                   # "online" | "live" -- lower for context

    # Solve geometry.
    oop_position: str                            # seat label for the OOP side, e.g. "BB"
    ip_position: str                             # seat label for the IP side, e.g. "BTN"
    preflop_actions: tuple = field(default=())   # rendered preflop line, e.g.
                                                  # (("BTN", "open", 1.25), ("BB", "call"))

    # Derived strings, computed in __post_init__.
    context: str = field(init=False)             # "6-Handed, $0.25/$0.50, Stacks $50.00"

    def __post_init__(self) -> None:
        if self.game_format not in ("cash", "tournament"):
            raise ValueError(f"game_format must be 'cash' or 'tournament', "
                             f"got {self.game_format!r}")
        if self.venue not in ("online", "live"):
            raise ValueError(f"venue must be 'online' or 'live', got {self.venue!r}")
        if not 6 <= self.table_size <= 9:
            raise ValueError(f"table_size must be 6-9, got {self.table_size}")
        if self.oop_position == self.ip_position:
            raise ValueError("oop_position and ip_position must differ")
        if self.dollars_per_bb <= 0:
            raise ValueError("default_stack_dollars / default_stack_bb must be > 0")
        # Match the team's online-cash Context format from the .xlsx sample row 1:
        # "6-Handed, $0.25/$0.50, Stacks $50.00".
        object.__setattr__(
            self, "context",
            f"{self.table_size}-Handed, {self.stakes}, "
            f"Stacks ${self.default_stack_dollars:.2f}",
        )

    @property
    def dollars_per_bb(self) -> float:
        """Display-cash value of one big blind."""
        return self.default_stack_dollars / self.default_stack_bb


# --- the registry -----------------------------------------------------------
def _srp_scenario_template(*, cfr_key: str, preflop_action: str,
                           oop_position: str, ip_position: str,
                           preflop_actions: tuple) -> "ScenarioConfig":
    """Helper: build a Cash6max 100bb online SRP ScenarioConfig used as a
    TEMPLATE -- batch_demo_v6 clones it per actual .cfr via dataclasses.replace.
    All Tier-1 SRP scenarios share the same stakes/table/stack metadata;
    only positions, the prose action, and the preflop_actions tuple differ.
    """
    return ScenarioConfig(
        cfr_key=cfr_key,
        format="Cash 6-max", stakes="$0.25/$0.50",
        live_or_online="Online", preflop_action=preflop_action,
        game_format="cash", stakes_sb=0.25, stakes_bb=0.50, table_size=6,
        default_stack_bb=100, default_stack_dollars=50.00, venue="online",
        oop_position=oop_position, ip_position=ip_position,
        preflop_actions=preflop_actions,
    )


SCENARIOS: dict[str, ScenarioConfig] = {
    "btn_vs_bb_srp_2cJs7s": ScenarioConfig(
        cfr_key="btn_vs_bb_srp_2cJs7s",
        format="Cash 6-max",
        stakes="$0.25/$0.50",
        live_or_online="Online",
        preflop_action="BTN open 2.5bb, BB call",
        game_format="cash",
        stakes_sb=0.25,
        stakes_bb=0.50,
        table_size=6,
        default_stack_bb=100,
        default_stack_dollars=50.00,
        venue="online",
        oop_position="BB",
        ip_position="BTN",
        # BTN opens 2.5bb ($1.25); SB folds (implicit -- only the surviving
        # seats appear in preflop_actions); BB calls.
        preflop_actions=(("BTN", "open", 1.25), ("BB", "call")),
    ),
    # Scenario 2 (May 2026): CO opens vs BB call. The "_template" suffix
    # signals this entry is cloned by batch_demo_v6 per actual .cfr via
    # dataclasses.replace (the cfr_key field is overridden at clone time).
    "co_vs_bb_srp_template": _srp_scenario_template(
        cfr_key="co_vs_bb_srp_template",
        preflop_action="CO open 2.5bb, BB call",
        oop_position="BB", ip_position="CO",
        preflop_actions=(("CO", "open", 1.25), ("BB", "call")),
    ),
    # Scenario 4 (May 2026): SB opens vs BB call (BvB). Postflop order is
    # SB->BB so SB is the OOP side. preflop_actions includes only the
    # surviving seats; UTG/HJ/CO/BTN all fold before action reaches SB.
    "sb_vs_bb_srp_template": _srp_scenario_template(
        cfr_key="sb_vs_bb_srp_template",
        preflop_action="SB open 2.5bb, BB call",
        oop_position="SB", ip_position="BB",
        preflop_actions=(("SB", "open", 1.25), ("BB", "call")),
    ),
    # Scenario 5 (May 2026): HJ opens vs BB call (SRP). preflop_actions
    # includes only the surviving seats; UTG folds before HJ, and CO/BTN/SB
    # fold after HJ's open.
    "hj_vs_bb_srp_template": _srp_scenario_template(
        cfr_key="hj_vs_bb_srp_template",
        preflop_action="HJ open 2.5bb, BB call",
        oop_position="BB", ip_position="HJ",
        preflop_actions=(("HJ", "open", 1.25), ("BB", "call")),
    ),
    # Scenario 3 (May 2026): BTN opens vs SB call (SRP, thin). Postflop
    # order is SB -> BTN, so SB is OOP. BB folds before action returns,
    # so only BTN and SB are surviving seats in preflop_actions.
    "btn_vs_sb_srp_template": _srp_scenario_template(
        cfr_key="btn_vs_sb_srp_template",
        preflop_action="BTN open 2.5bb, SB call",
        oop_position="SB", ip_position="BTN",
        preflop_actions=(("BTN", "open", 1.25), ("SB", "call")),
    ),
}


def _cfr_key(cfr_path: Path | str) -> str:
    """The registry key for a `.cfr` path or string -- the filename stem."""
    return Path(cfr_path).stem


def get_scenario(cfr_path: Path | str) -> ScenarioConfig:
    """Look up the scenario registered for a `.cfr` solve.

    Accepts a Path, a path string, or a bare scenario key. Raises KeyError
    with a clear list of what *is* registered if the key is unknown.
    """
    key = _cfr_key(cfr_path)
    try:
        return SCENARIOS[key]
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIOS)) or "(none)"
        raise KeyError(
            f"no scenario registered for {key!r}. Known scenarios: {known}. "
            f"Register one in pipeline/scenario_config.py:SCENARIOS before "
            f"running the pipeline against this solve."
        ) from exc


# --- spot -> action-history hand dict ---------------------------------------
def _split_segments(action_sequence: Iterable[tuple[str, str]]) -> list[list[tuple[str, str]]]:
    """Split a path-sampler action sequence into per-street segments.

    `("deal", card)` entries are the boundaries between streets. For a
    postflop-rooted solve (every Tier 1 solve) the returned list is
    `[flop_actions, turn_actions, river_actions]`: PioSolver's tree root sits
    at the flop, so the first segment is the flop, not preflop. Preflop
    actions come from the scenario, not the solve. Absent streets are dropped.
    """
    segments: list[list[tuple[str, str]]] = [[]]
    for actor, label in action_sequence:
        if actor == "deal":
            segments.append([])
        else:
            segments[-1].append((actor, label))
    return segments


def round_to_nearest_increment(amount: float, increment: float) -> float:
    """Round `amount` to the nearest multiple of `increment`.

    Used to snap converted bet/raise dollar amounts in the action-history
    prose to the nearest small blind so chip-derived numbers like
    `$1.85 / $5.23 / $12.15` render as `$1.75 / $5.25 / $12.25` -- the
    sizes a real player at $0.25/$0.50 would actually wager. Underlying
    EV math stays in chips; only the displayed prose is rounded.

    Banker's-rounding-free: ties round to the nearest even multiple of
    `increment` would be a surprise here, so we use Python's `round` on
    the multiple-count which already does banker's rounding -- acceptable
    because the inputs are not exact halfway points in practice. If a
    later spec wants away-from-zero rounding we can swap.
    """
    if increment <= 0:
        raise ValueError(f"increment must be > 0, got {increment!r}")
    return round(amount / increment) * increment


def _convert_postflop(entries: list[tuple[str, str]],
                      scenario: ScenarioConfig,
                      chips_per_bb: float) -> list[tuple]:
    """Translate one street of OOP/IP/<chip-amount> entries into the
    (position, verb, dollars) form `pipeline.action_history` expects.

    Bet/raise amounts come through PioSolver in chips and are converted to
    cash dollars using
        chips -> bb:   chips / chips_per_bb
        bb    -> $:    scenario.dollars_per_bb
    The final dollar amount is rounded to the nearest small blind so the
    action-history prose reads like a real player's wager (Ryan-feedback
    item #1, Apr 2026). Cent-precision rounding is the last-resort fallback
    when `stakes_sb` isn't set on the scenario.
    """
    converted: list[tuple] = []
    sb = scenario.stakes_sb if getattr(scenario, "stakes_sb", 0) > 0 else 0.01
    for actor, label in entries:
        position = (scenario.oop_position if actor == "OOP"
                    else scenario.ip_position)
        parts = label.split()
        verb = parts[0]
        if len(parts) == 2:
            chips = float(parts[1])
            raw_dollars = chips / chips_per_bb * scenario.dollars_per_bb
            amount = round_to_nearest_increment(raw_dollars, sb)
            # action_history validates amounts > 0; clamp very-tiny rounding to one
            # increment (so the smallest non-zero wager is one SB, not a fractional
            # cent that the formatter would render as $0.00).
            amount = max(amount, sb)
            converted.append((position, verb, amount))
        else:
            converted.append((position, verb))
    return converted


def spot_to_hand(spot_data: SpotData, scenario: ScenarioConfig) -> dict:
    """Turn a populated SpotData + its scenario into an action_history hand dict.

    The returned dict is the input shape `pipeline.action_history.format_hand`
    / `format_action_history` / `format_context` consume. This is the bridge
    between Layer 5 (structured spot data) and the deterministic hand renderer.

    Requires `spot_data.spot_metadata` to carry `action_sequence`,
    `big_blind_chips`, and `pot_bb` -- these are populated by
    `pipeline.fact_extractor.extract_facts` for every spot.
    """
    meta = spot_data.spot_metadata
    if meta.big_blind_chips <= 0:
        raise ValueError(
            f"spot has big_blind_chips={meta.big_blind_chips!r}; was extract_facts "
            f"called with `big_blind=...` matching the solve?"
        )

    segments = _split_segments(meta.action_sequence)
    # Postflop solve: segments[0] = flop, [1] = turn, [2] = river. Pad to 3
    # so we can index uniformly below.
    while len(segments) < 3:
        segments.append([])

    board = list(meta.board)
    flop = board[:3] if len(board) >= 3 else None
    turn = board[3] if len(board) >= 4 else None
    river = board[4] if len(board) >= 5 else None

    return {
        "stakes": {"sb": scenario.stakes_sb, "bb": scenario.stakes_bb},
        "format": scenario.game_format,
        "venue": scenario.venue,
        "table_size": scenario.table_size,
        # For cash the action_history context formatter wants dollars; for
        # tournament it wants bb. Tier 1 is cash-only -- the tournament branch
        # will need scenario.default_stack_bb once added.
        "effective_stack": (scenario.default_stack_dollars
                            if scenario.game_format == "cash"
                            else scenario.default_stack_bb),
        "hero_position": meta.hero_position,
        "hero_cards": list(meta.hero_cards),
        "preflop_actions": list(scenario.preflop_actions),
        "board": {"flop": flop, "turn": turn, "river": river},
        "flop_actions": _convert_postflop(segments[0], scenario, meta.big_blind_chips),
        "turn_actions": _convert_postflop(segments[1], scenario, meta.big_blind_chips),
        "river_actions": _convert_postflop(segments[2], scenario, meta.big_blind_chips),
    }


__all__ = [
    "ScenarioConfig",
    "SCENARIOS",
    "get_scenario",
    "round_to_nearest_increment",
    "spot_to_hand",
]
