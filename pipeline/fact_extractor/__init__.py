"""Layer 5 -- Fact Extractor.

Turns solver output into the structured data block every explanation is built
from: hand class, board texture, equity math, and concept tags. All strategic
reasoning lives here, in deterministic Python -- never in the LLM.

`extract_facts` is the top-level entry point: it takes a Path Sampler
SpotContext and orchestrates the whole layer -- hand class, board texture,
equity / range / blocker extraction, and the concept tagger -- into one fully
populated SpotData.

See docs/engineering_brief.docx, "Layer 5: Fact Extractor".
"""
from __future__ import annotations

from pipeline.fact_extractor.board_texture import classify_board  # noqa: F401
from pipeline.fact_extractor.concept_tags.registry import compute_tags
from pipeline.fact_extractor.equity import equity_vs_range
from pipeline.fact_extractor.equity_range_blockers import (
    compute_equity_data, compute_range_data,
)
from pipeline.fact_extractor.hand_class import classify_hand  # noqa: F401
from pipeline.fact_extractor.spot_data import (
    BoardTexture, DecisionData, HandClass, SpotData, SpotMetadata,
)


def _canonical(label: str) -> str:
    """Action label -> canonical verb: 'bet 36' -> 'bet', 'check' -> 'check'."""
    return label.split()[0]


def _amount(label: str) -> float | None:
    """The chip amount embedded in an action label, if any ('bet 36' -> 36)."""
    parts = label.split()
    if len(parts) == 2 and parts[1].replace(".", "", 1).isdigit():
        return float(parts[1])
    return None


def _street_segments(action_sequence):
    """Split an action sequence into per-street segments at the card deals."""
    segments: list[list] = [[]]
    for actor, label in action_sequence:
        if actor == "deal":
            segments.append([])
        else:
            segments[-1].append((actor, label))
    return segments


def _build_decision_data(spot_context, big_blind: float) -> DecisionData:
    """SpotContext -> DecisionData: options, strategy, EVs, and the action line.

    Solver EVs (PioSolver's range-weighted mean of net-from-baseline chips at
    each action's CHILD node -- see the EV-gap convention block below) are
    divided by `big_blind` so the data block is bb-denominated. For a 100bb
    solve `big_blind == effective_stack / 100`; callers in `scripts/` derive it
    from `client.show_effective_stack()` so this stays exact across stack depths.
    `ev_gap_bb` is the bb-denominated gap between the best and second-best
    action's child-mean EV -- Layer 4's question-worthiness signal.
    """
    node = spot_context.node
    actions = spot_context.actions
    hero_side = "OOP" if node.hero_is_oop else "IP"

    segments = _street_segments(node.action_sequence)
    convert = lambda entries: [("hero" if actor == hero_side else "villain",
                                _canonical(label)) for actor, label in entries]

    option_fractions = {}
    for action in actions:
        amount = _amount(action.label)
        if amount is not None and node.pot > 0:
            option_fractions[_canonical(action.label)] = amount / node.pot

    facing = None
    if node.amount_to_call > 0:
        pot_before = node.pot - node.amount_to_call
        if pot_before > 0:
            facing = node.amount_to_call / pot_before

    # correct_action: the verb of the MAX-FREQUENCY action. At a converged
    # equilibrium -- which PioSolver Edge reaches at <0.5% pot exploitability
    # per the brief -- every mixed action has the same EV, so max-frequency
    # equals max-EV. The 5-row audit (May 2026) confirmed empirically: every
    # spot had pio_top_freq == pio_top_ev. If this pipeline runs against a
    # non-equilibrium solve in the future, the freq/EV equivalence may not
    # hold and this line should be revisited.
    correct = (_canonical(max(actions, key=lambda a: a.frequency).label)
               if actions else "")
    # ev_gap_bb convention (verified end-to-end against the BTN-vs-BB SRP
    # 2c Js 7s test solve, May 2026 -- the investigation that produced this
    # file's docstrings):
    #   * `a.ev` is the range-weighted mean EV at the CHILD node reached by
    #     action `a`, in chips. PioSolver's `calc_ev` returns net-from-baseline
    #     chips per combo; `path_sampler._weighted_mean` weights those by hero's
    #     range at the decision node. Mean is over the child, NOT the parent --
    #     so the gap means "this action's downstream value vs that action's
    #     downstream value", which is the only framing that makes the 0.5bb
    #     threshold in `question_extractor.MIN_EV_GAP_BB` meaningful.
    #   * `big_blind` is `effective_stack / 100` (every Tier 1 solve is 100bb
    #     deep); dividing chips by `big_blind` gives a bb figure.
    #   * `ev_gap_bb` is best-minus-second over ACTIONS, not combos. (Per-combo
    #     spreads can be huge even when the range-mean gap is small -- e.g. an
    #     AKo combo and a 22 combo in the same range will swing wildly on a
    #     single river card while the action's range-EV barely moves.)
    evs_bb = sorted((a.ev / big_blind for a in actions), reverse=True)
    ev_gap_bb = evs_bb[0] - evs_bb[1] if len(evs_bb) >= 2 else 0.0
    return DecisionData(
        options=[a.label for a in actions],
        range_mean_evs_per_action={_canonical(a.label): a.ev / big_blind
                                   for a in actions},
        range_aggregate_strategy={_canonical(a.label): max(0.0, min(1.0, a.frequency))
                                  for a in actions},
        correct_action=correct,
        ev_gap_bb=ev_gap_bb,
        option_pot_fractions=option_fractions,
        facing_bet_pot_fraction=facing,
        street_actions=convert(segments[-1]),
        prior_street_actions=convert(segments[-2]) if len(segments) >= 2 else [],
    )


def _build_spot_metadata(spot_context, scenario: dict, big_blind: float,
                         hero_hand: str) -> SpotMetadata:
    """SpotContext -> SpotMetadata, with `scenario` supplying the facts a
    postflop solve cannot carry (preflop raise count, game format, ...)."""
    node = spot_context.node
    fields = dict(
        street=node.street,
        stack_depth_bb=node.effective_stack / big_blind,
        effective_stack_bb=node.effective_stack / big_blind,
        spr=node.effective_stack / node.pot if node.pot > 0 else 0.0,
        position_dynamic=f"{node.hero_position}_vs_{node.villain_position}",
        hero_position=node.hero_position,
        villain_position=node.villain_position,
        hero_in_position=not node.hero_is_oop,
        parent_node_id=node.parent_node_id,
        action_to_reach=node.action_to_reach,
        node_id=node.node_id,
        hero_cards=(hero_hand[:2], hero_hand[2:]),
        board=list(node.board),
        # Rendering context for the deterministic action-history block.
        action_sequence=list(node.action_sequence),
        big_blind_chips=big_blind,
        pot_bb=node.pot / big_blind,
    )
    fields.update(scenario)                         # caller-supplied overrides
    return SpotMetadata(**fields)


def extract_facts(spot_context, *, hero_hand: str | None = None,
                  scenario: dict | None = None,
                  big_blind: float = 1.0) -> SpotData:
    """Build a fully populated SpotData from a Path Sampler SpotContext.

    Orchestrates the whole of Layer 5: hand class, board texture, the equity /
    range / blocker extraction, and the concept tagger. `hero_hand` defaults to
    hero's most likely combo at the spot; `scenario` supplies metadata the
    solve does not carry (e.g. preflop_raise_count); `big_blind` is the chip
    value of one big blind, used to express EVs and stacks in bb. Real callers
    pass `client.show_effective_stack() / 100` (every Tier 1 solve is 100bb
    deep); leave the default 1.0 only when keeping raw solver chips on purpose
    -- e.g. in unit tests with synthetic SpotContexts. See
    `_build_decision_data` for the full EV-gap convention.

    The equity pass runs once here and is shared by the equity_data and
    range_data builders.
    """
    node = spot_context.node
    board = node.board
    if hero_hand is None:
        playable = {c: w for c, w in spot_context.hero_range.items()
                    if not ({c[:2], c[2:]} & set(board))}
        if not playable:
            raise ValueError("spot_context has no playable hero combos")
        hero_hand = max(playable, key=playable.get)

    hero_equity, villain_combo_equity = equity_vs_range(
        [hero_hand[:2], hero_hand[2:]], spot_context.villain_range, board)

    spot = SpotData(
        spot_metadata=_build_spot_metadata(spot_context, scenario or {},
                                           big_blind, hero_hand),
        decision_data=_build_decision_data(spot_context, big_blind),
        equity_data=compute_equity_data(spot_context, hero_hand, hero_equity),
        range_data=compute_range_data(spot_context, hero_hand, villain_combo_equity),
        hand_class=(HandClass.from_cards(hero_hand, board)
                    if len(board) >= 3 else None),
        board_texture=(BoardTexture.from_board(board)
                       if len(board) >= 3 else None),
    )
    spot.concept_tags = compute_tags(spot)
    return spot
