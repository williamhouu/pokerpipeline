"""Layer 2: Tree Resolver -- solver-side scenario specifications.

Where `pipeline.scenario_config.ScenarioConfig` carries the *display* metadata
that Layers 5-8 use to render Context, Question, Stakes, etc., this module
carries the *solver* metadata Layer 2 uses to drive PioSolver Edge: ranges,
bet sizings, accuracy target, the pot-and-stack geometry the postflop solve
starts from.

The two are deliberately separated:

  * ScenarioConfig is keyed by `.cfr` filename stem (one entry per solve).
    It's the small set of facts the postflop solve cannot carry but that
    Layers 5-8 need at render time.
  * SolverSpec is keyed by SCENARIO name (one entry per (positions, pot type,
    stack depth) tuple -- spans many `.cfr` files, one per flop). It's the
    larger set of facts the batch solver needs to *produce* the solves.

The expected end state, once Layer 2 ships its first batch of solves:

  SolverSpec('Cash6max_100bb_BTN_open_BB_call')
      x flop_sets.STANDARD_25_FLOPS
      -> 25 .cfr files at solves/Cash6max_100bb_BTN_open_BB_call/<flop>.cfr

Each of those .cfr files needs a ScenarioConfig for Layers 5-8 -- today that
mapping is hand-authored in scenario_config.py. A follow-up commit will
auto-derive ScenarioConfig from SolverSpec + flop so the registry doesn't
grow linearly with solve count.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --- range placeholders -----------------------------------------------------
# These are Phase-0 placeholders chosen to roughly match PioSolver Edge's
# default 100bb 6-max ranges. They will be reviewed by Ryan against the
# team's preferred preflop solver output before Tier 1 production solves.
#
# Format: PioSolver range string (subset notation, comma-separated). The
# alternative is a path to a .txt file holding a 13x13 weighted grid; pass
# either; the batch solver dispatches on whether the string looks like a path.
#
# Source / sanity-check:
#   * BTN open ~45% of hands at 100bb 6-max: pairs, broadway suited+offsuit,
#     suited connectors+gappers, broadway-x suited, all suited aces, K/Q
#     wheels.
#   * BB call vs BTN 2.5bb is ~25% (the calling portion only; 3-bets are
#     excluded from a "call" range and live in a separate solve).
BTN_OPEN_100BB_PLACEHOLDER = (
    "22+,"
    "A2s+,K2s+,Q4s+,J6s+,T6s+,96s+,85s+,75s+,64s+,53s+,43s,"
    "A2o+,K9o+,Q9o+,J9o+,T9o"
)
BB_CALL_VS_BTN_OPEN_PLACEHOLDER = (
    "22-TT,"
    "A2s-A9s,K2s-K9s,Q5s-Q9s,J7s-J9s,T7s-T9s,96s-97s,86s-87s,75s-76s,65s,54s,"
    "A2o-A9o,KTo,QTo,JTo"
)


# --- the solver-side spec ---------------------------------------------------
@dataclass(frozen=True)
class SolverSpec:
    """Everything PioSolver Edge needs to set up + solve a postflop tree.

    Frozen so it can be shared across many solves in a batch run without any
    chance of accidental mutation between iterations.

    Two ways to drive a solve:

      * **Template-driven (recommended).** `pio_template_path` points at one
        of Pio's shipped `.txt` templates under `C:\\PioSOLVER\\TreeBuilding\\`.
        Layer 2 reads the template line-by-line, substitutes the target
        flop into the `set_board` line, and issues each command via UPI.
        The template encodes ranges, sizings, and (via `add_line`) the
        complete betting tree -- this matches Pio's GUI tree-builder output
        exactly and reproduces the hand-solved test `.cfr`. See the May-2026
        UPI findings doc for why this is the correct architecture.

      * **Programmatic (fallback).** If `pio_template_path` is None, Layer 2
        builds the tree from scratch via individual UPI commands using the
        spec's range/sizing fields. Not currently supported for Tier 1
        (the precise UPI dialect for tree construction varies across Pio
        builds); keep template-driven for production.

    Fields:
      * `name` -- registry key; also the directory name under `solves/`.
      * `format` -- "cash" or "tournament" (Tier 1 is cash only).
      * `stack_bb` -- starting stack in bb at the start of the hand
        (BEFORE preflop investment). Display field; the actual solver-stack
        comes from `starting_postflop_stack_chips`.
      * `oop_position` / `ip_position` -- e.g. "BB" / "BTN". Pio doesn't
        care about position labels; these are recorded for downstream
        rendering and for cache-path readability.
      * `oop_range` / `ip_range` -- DOCUMENTARY in template-driven mode --
        the actual range weights come from the template. Kept on the spec
        so the registry remains self-describing without opening the
        template file. Pio range string OR a path to a .txt range file.
      * `pot_after_preflop_chips` -- the carried pot at the flop root, in
        chips. Must match the template's `#Pot#` value when template-driven.
      * `starting_postflop_stack_chips` -- effective stack at the flop root.
        Must match the template's `#EffectiveStacks#` value.
      * `bet_sizes_oop_pct` / `bet_sizes_ip_pct` / `raise_sizes_pct` --
        documentary in template-driven mode (actual sizings come from the
        template's `#FlopConfig.BetSize#` etc. fields).
      * `accuracy_target_chips` -- solve exploitability target in chips
        (~0.5% of pot per the brief). Templates don't carry this -- Layer 2
        issues `set_accuracy <chips>` after loading the template.
      * `iso_suits` / `iso_board` -- documentary; the template's
        `set_isomorphism` line dictates the actual setting at solve time.
      * `bb_in_chips` -- chip value of 1bb at this stack scale.
      * `pio_template_path` -- path to the Pio tree template. Can be an
        absolute Windows path (Pio's shipped templates under
        `C:\\PioSOLVER\\TreeBuilding\\`) or a path relative to the repo
        root (custom templates under `templates/`). For Tier 1 the canonical
        Ryan-ranges template lives at
        `templates/Cash6max_100bb_BTN_open_BB_call_ryan_ranges.txt`,
        generated by `scripts/build_ryan_ranges_template.py`.
      * `preflop_action_description` -- human-readable preflop line, used
        in log messages and (eventually) when synthesising a ScenarioConfig.
      * `using_ryan_ranges` -- True iff the OOP/IP ranges in the template
        were derived from Ryan's preflop range pack (the canonical Tier-1
        source per docs/ryan_range_pack_index.md) rather than from Pio's
        shipped placeholder weights or hand-tuned strings. Documentary:
        callers (batch driver, downstream solves) can use it to log which
        range source produced a given solve.
    """

    name: str
    format: str
    stack_bb: int
    oop_position: str
    ip_position: str
    oop_range: str
    ip_range: str
    pot_after_preflop_chips: int
    starting_postflop_stack_chips: int
    bet_sizes_oop_pct: tuple[int, ...]
    bet_sizes_ip_pct: tuple[int, ...]
    raise_sizes_pct: tuple[int, ...]
    accuracy_target_chips: float
    bb_in_chips: int
    iso_suits: bool = False
    iso_board: bool = False
    preflop_action_description: str = ""
    pio_template_path: str = ""                  # template-driven solve source
    using_ryan_ranges: bool = False              # documentary -- see field docstring

    def __post_init__(self) -> None:
        if self.format not in ("cash", "tournament"):
            raise ValueError(f"format must be 'cash' or 'tournament', "
                             f"got {self.format!r}")
        if self.stack_bb <= 0:
            raise ValueError(f"stack_bb must be > 0, got {self.stack_bb}")
        if self.oop_position == self.ip_position:
            raise ValueError("oop_position and ip_position must differ")
        if self.pot_after_preflop_chips <= 0:
            raise ValueError("pot_after_preflop_chips must be > 0, "
                             f"got {self.pot_after_preflop_chips}")
        if self.starting_postflop_stack_chips <= 0:
            raise ValueError(
                "starting_postflop_stack_chips must be > 0, "
                f"got {self.starting_postflop_stack_chips}")
        if self.bb_in_chips <= 0:
            raise ValueError(f"bb_in_chips must be > 0, got {self.bb_in_chips}")
        if not self.bet_sizes_oop_pct or not self.bet_sizes_ip_pct:
            raise ValueError("at least one bet size required for each side; "
                             "the brief specifies a minimum of 2 per street")
        for name in ("bet_sizes_oop_pct", "bet_sizes_ip_pct", "raise_sizes_pct"):
            for size in getattr(self, name):
                if size <= 0 or size > 1000:
                    raise ValueError(
                        f"{name} entries must be 1-1000 (% of pot), got {size}")
        if self.accuracy_target_chips <= 0:
            raise ValueError("accuracy_target_chips must be > 0, "
                             f"got {self.accuracy_target_chips}")
        # Cross-consistency: pot + 2*stack should equal the total hand pot if
        # both players were all-in -- sanity-check that the geometry is
        # plausible (allowing for ~bb of dead money).
        max_total = (self.pot_after_preflop_chips
                     + 2 * self.starting_postflop_stack_chips)
        full_stack_chips = self.stack_bb * self.bb_in_chips
        if max_total < full_stack_chips:
            raise ValueError(
                f"geometry inconsistent: pot + 2*stack = {max_total} chips "
                f"but stack_bb * bb_in_chips = {full_stack_chips}")

    @property
    def cache_dir_name(self) -> str:
        """Directory under `solves/` for this scenario's .cfr files."""
        return self.name

    def range_is_file(self, side: str) -> bool:
        """Whether the OOP/IP range field is a filesystem path (vs an
        inline Pio range string). Path detection: ends in .txt or .rng
        AND resolves to an existing file. Relative paths are resolved
        against the repo root, so the result is independent of the
        caller's CWD.
        """
        candidate = self.oop_range if side == "OOP" else self.ip_range
        path = Path(candidate)
        if path.suffix.lower() not in (".txt", ".rng"):
            return False
        if path.is_absolute():
            return path.is_file()
        # Relative -- resolve against repo root (= parent of `pipeline/`).
        return (Path(__file__).resolve().parent.parent / path).is_file()


# --- the registry -----------------------------------------------------------
# Naming convention for `name`:
#   <Format><TableSize>_<StackBB>_<OOPpos>_<preflop_action_summary>
# Filesystem-safe (no spaces, no special chars beyond underscore). The same
# name becomes the cache directory under solves/.
def _srp_spec(*, name: str, oop_position: str, ip_position: str,
              oop_range: str, ip_range: str, template_basename: str,
              preflop_action_description: str) -> "SolverSpec":
    """Helper: build a Cash6max 100bb SRP SolverSpec. All Tier-1 SRP
    scenarios share the same chip geometry (pot=55, eff=975) and bet
    sizings (drawn from PioSolver's shipped 2bpot-full.txt template);
    only ranges, positions, and the action-description prose differ.

    `template_basename` is the filename (no path) under `templates/`;
    `oop_range` / `ip_range` are repo-relative paths into Ryan's pack.
    """
    return SolverSpec(
        name=name, format="cash", stack_bb=100,
        oop_position=oop_position, ip_position=ip_position,
        oop_range=oop_range, ip_range=ip_range,
        pot_after_preflop_chips=55,
        starting_postflop_stack_chips=975,
        bb_in_chips=10,
        bet_sizes_oop_pct=(65,), bet_sizes_ip_pct=(65,),
        raise_sizes_pct=(52,),
        accuracy_target_chips=0.28,
        iso_suits=True, iso_board=False,
        preflop_action_description=preflop_action_description,
        pio_template_path=f"templates/{template_basename}",
        using_ryan_ranges=True,
    )


def _3bp_spec(*, name: str, oop_position: str, ip_position: str,
              oop_range: str, ip_range: str, template_basename: str,
              preflop_action_description: str) -> "SolverSpec":
    """Helper: build a Cash6max 100bb 3-bet pot SolverSpec. All Tier-1
    3-bp scenarios (6-10) share the same chip geometry (pot=180, eff=910)
    drawn from Pio's shipped 100bb/3bpot-full.txt (also committed to the
    repo at templates/3bpot-full.txt); only ranges, positions, and the
    preflop action prose differ.

    The pack's 3-bet size token varies by actor (77%, 150%, 155%, 182%)
    but we model all 3bp scenarios with one canonical postflop geometry
    -- same shared-chassis precedent the SRP scenarios use (see
    docs/ryan_range_pack_index.md "Sizing convention" for the open
    question still owed to Ryan).

    Accuracy 0.9 chips ~= 0.5% of the 180-chip pot per the brief.
    """
    return SolverSpec(
        name=name, format="cash", stack_bb=100,
        oop_position=oop_position, ip_position=ip_position,
        oop_range=oop_range, ip_range=ip_range,
        pot_after_preflop_chips=180,
        starting_postflop_stack_chips=910,
        bb_in_chips=10,
        bet_sizes_oop_pct=(52,), bet_sizes_ip_pct=(52,),
        raise_sizes_pct=(45,),
        accuracy_target_chips=0.9,
        iso_suits=True, iso_board=False,
        preflop_action_description=preflop_action_description,
        pio_template_path=f"templates/{template_basename}",
        using_ryan_ranges=True,
    )


def _4bp_spec(*, name: str, oop_position: str, ip_position: str,
              oop_range: str, ip_range: str, template_basename: str,
              preflop_action_description: str) -> "SolverSpec":
    """Helper: build a Cash6max 100bb 4-bet pot SolverSpec. All Tier-1
    4-bp scenarios (11-14) share the same chip geometry (pot=500,
    eff=1000) drawn from Pio's HUspots/4betpot.txt (committed to the
    repo at templates/4bpot-full.txt).

    Geometry caveat: Pio's HU 4bp template was built for a deeper-stacked
    4-bp situation than a real 6-max 100bb 4-bp post-call (which has
    pot~44bb / eff~78bb, SPR~1.8 vs Pio's pot=50bb / eff=100bb, SPR=2).
    Downstream layers that convert solve-chip amounts to display dollars
    via bb_in_chips=10 will produce numbers consistent with a 125bb HU
    cash 4-bp rather than a 100bb 6-max 4-bp. The pack ranges are still
    correct for 100bb 6-max 4-bp; only the postflop pot/stack geometry
    differs. Same shared-chassis precedent as SRP and 3bp (one canonical
    geometry, ranges differ per scenario); see commit on 4bpot-full.txt
    for the original geometry choice.

    Accuracy 2.5 chips ~= 0.5% of the 500-chip pot per the brief.
    """
    return SolverSpec(
        name=name, format="cash", stack_bb=100,
        oop_position=oop_position, ip_position=ip_position,
        oop_range=oop_range, ip_range=ip_range,
        pot_after_preflop_chips=500,
        starting_postflop_stack_chips=1000,
        bb_in_chips=10,
        bet_sizes_oop_pct=(65,), bet_sizes_ip_pct=(65,),
        raise_sizes_pct=(300,),  # template's '3x' raise size = 300% pot
        accuracy_target_chips=2.5,
        iso_suits=True, iso_board=False,
        preflop_action_description=preflop_action_description,
        pio_template_path=f"templates/{template_basename}",
        using_ryan_ranges=True,
    )


SOLVER_SPECS: dict[str, SolverSpec] = {
    "Cash6max_100bb_BTN_open_BB_call": SolverSpec(
        name="Cash6max_100bb_BTN_open_BB_call",
        format="cash",
        stack_bb=100,
        oop_position="BB",
        ip_position="BTN",
        # Documentary -- the template's `set_range OOP <1326 weights>` supplies
        # the actual range, expanded from Ryan's pack files. The repo-relative
        # paths recorded here are the authoritative source for that expansion;
        # `scripts/build_ryan_ranges_template.py` reads them and regenerates
        # the template if the source weights change.
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%.txt"),
        # Pio's `2bpot-full.txt` template chip scale (#Pot#55 #EffectiveStacks#975).
        # 1bb ~= 10 chips at this scale (55 = 5.5bb, 975 = ~100bb). Different
        # from a naive 90 chips/bb derivation -- Pio's template uses its own
        # canonical scale, and the existing hand-solved test_solves/btn_vs_bb_srp_2cJs7s.cfr
        # confirms it via show_effective_stack() = 975.
        pot_after_preflop_chips=55,
        starting_postflop_stack_chips=975,
        bb_in_chips=10,
        # Documentary -- the template's `add_line` sequences encode the
        # actual bet/raise chip amounts used at solve time. The %s recorded
        # here match the template's `#FlopConfig.BetSize#65` etc. fields.
        bet_sizes_oop_pct=(65,),                  # flop default; turn/river vary
        bet_sizes_ip_pct=(65,),
        raise_sizes_pct=(52,),                    # template's #*.RaiseSize#52
        # 0.28 chips = ~0.5% of the 55-chip pot, the brief's accuracy target.
        accuracy_target_chips=0.28,
        iso_suits=True,                           # template uses set_isomorphism 1 0
        iso_board=False,
        preflop_action_description="BTN opens 2.5bb, SB folds, BB calls",
        # Repo-local template -- clone of PioSolver's shipped 2bpot-full.txt
        # with the `set_range OOP/IP` lines replaced by 1326-combo weights
        # expanded from Ryan's pack (the two files referenced in oop_range /
        # ip_range above). Regenerate via:
        #   python scripts/build_ryan_ranges_template.py
        # See test_output/upi_findings.md for why template-driven solves are
        # the right primitive vs. constructing the tree from scratch.
        pio_template_path=(
            "templates/Cash6max_100bb_BTN_open_BB_call_ryan_ranges.txt"),
        using_ryan_ranges=True,
    ),
    # Scenario 2 (May 2026): CO opens vs BB call (SRP). Same chip geometry as
    # BTN-vs-BB; ranges from docs/ryan_range_pack_index.md scenario #2.
    "Cash6max_100bb_CO_open_BB_call": _srp_spec(
        name="Cash6max_100bb_CO_open_BB_call",
        oop_position="BB", ip_position="CO",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_Fold_CO_60%_BTN_Fold_SB_Fold_BB_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/CO/"
                  "UTG_Fold_HJ_Fold_CO_60%.txt"),
        template_basename="Cash6max_100bb_CO_open_BB_call_ryan_ranges.txt",
        preflop_action_description=(
            "UTG and HJ fold, CO opens 2.5bb, BTN and SB fold, BB calls"),
    ),
    # Scenario 4 (May 2026): SB opens vs BB call (BvB SRP). Postflop order
    # is SB->BB so SB is OOP at the flop. The pack's SB-open node labels
    # itself "76%" but we model it as 2.5x to keep the shared 5.5bb-pot
    # Tier-1 geometry; see docs/ryan_range_pack_index.md "Sizing convention"
    # for the open question still owed to Ryan.
    "Cash6max_100bb_SB_open_BB_call": _srp_spec(
        name="Cash6max_100bb_SB_open_BB_call",
        oop_position="SB", ip_position="BB",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/SB/"
                   "UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76%.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                  "UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76%_BB_Call.txt"),
        template_basename="Cash6max_100bb_SB_open_BB_call_ryan_ranges.txt",
        preflop_action_description=(
            "UTG, HJ, CO, BTN fold, SB opens 2.5bb, BB calls (BvB)"),
    ),
    # Scenario 5 (May 2026): HJ opens vs BB call (SRP). Ranges from
    # docs/ryan_range_pack_index.md scenario #5.
    "Cash6max_100bb_HJ_open_BB_call": _srp_spec(
        name="Cash6max_100bb_HJ_open_BB_call",
        oop_position="BB", ip_position="HJ",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/HJ/"
                  "UTG_Fold_HJ_60%.txt"),
        template_basename="Cash6max_100bb_HJ_open_BB_call_ryan_ranges.txt",
        preflop_action_description=(
            "UTG folds, HJ opens 2.5bb, CO, BTN, SB fold, BB calls"),
    ),
    # Scenario 3 (May 2026): BTN opens vs SB call (SRP, thin SB-flat).
    # Postflop order: SB -> BTN, so SB is OOP. Per
    # docs/ryan_range_pack_index.md Open Questions A: SB only flats with
    # ~6% of hands vs BTN open (the bulk of SB's defense is 3-bet,
    # see Scenario 9). Wired for completeness; Ryan will decide whether
    # to keep it in production.
    "Cash6max_100bb_BTN_open_SB_call": _srp_spec(
        name="Cash6max_100bb_BTN_open_SB_call",
        oop_position="SB", ip_position="BTN",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/SB/"
                   "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%.txt"),
        template_basename="Cash6max_100bb_BTN_open_SB_call_ryan_ranges.txt",
        preflop_action_description=(
            "UTG, HJ, CO fold, BTN opens 2.5bb, SB calls, BB folds"),
    ),
    # Scenario 6 (May 2026): BTN opens, BB 3-bets, BTN calls (3BP).
    # Pack uses BB 3-bet token '182%'. Ranges from
    # docs/ryan_range_pack_index.md scenario #6.
    "Cash6max_100bb_BTN_open_BB_3bet_BTN_call": _3bp_spec(
        name="Cash6max_100bb_BTN_open_BB_3bet_BTN_call",
        oop_position="BB", ip_position="BTN",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_Call.txt"),
        template_basename=(
            "Cash6max_100bb_BTN_open_BB_3bet_BTN_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG, HJ, CO fold, BTN opens 2.5bb, SB folds, BB 3-bets, BTN calls"),
    ),
    # Scenario 7 (May 2026): CO opens, BTN 3-bets, CO calls (3BP). Pack
    # uses BTN 3-bet token '77%' -- significantly smaller than BB's 182%
    # 3-bets because IP doesn't need the same fold-equity pressure.
    # Ranges from docs/ryan_range_pack_index.md scenario #7.
    "Cash6max_100bb_CO_open_BTN_3bet_CO_call": _3bp_spec(
        name="Cash6max_100bb_CO_open_BTN_3bet_CO_call",
        oop_position="CO", ip_position="BTN",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/CO/"
                   "UTG_Fold_HJ_Fold_CO_60%_BTN_77%_SB_Fold_BB_Fold_CO_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_60%_BTN_77%.txt"),
        template_basename=(
            "Cash6max_100bb_CO_open_BTN_3bet_CO_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG, HJ fold, CO opens 2.5bb, BTN 3-bets, SB and BB fold, CO calls"),
    ),
    # Scenario 8 (May 2026): HJ opens, BB 3-bets, HJ calls (3BP). Pack
    # uses BB 3-bet token '182%'; ranges tighter than vs BTN because HJ
    # opens tighter. From docs/ryan_range_pack_index.md scenario #8.
    "Cash6max_100bb_HJ_open_BB_3bet_HJ_call": _3bp_spec(
        name="Cash6max_100bb_HJ_open_BB_3bet_HJ_call",
        oop_position="BB", ip_position="HJ",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/HJ/"
                  "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%_HJ_Call.txt"),
        template_basename=(
            "Cash6max_100bb_HJ_open_BB_3bet_HJ_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG folds, HJ opens 2.5bb, CO, BTN, SB fold, BB 3-bets, HJ calls"),
    ),
    # Scenario 9 (May 2026): BTN opens, SB 3-bets, BTN calls (3BP). Pack
    # uses SB 3-bet token '150%'. SB is OOP at the flop (postflop order
    # SB->BTN). BB folds during round 2 before BTN's response (so the
    # range file path includes the BB_Fold token). From docs scenario #9.
    "Cash6max_100bb_BTN_open_SB_3bet_BTN_call": _3bp_spec(
        name="Cash6max_100bb_BTN_open_SB_3bet_BTN_call",
        oop_position="SB", ip_position="BTN",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/SB/"
                   "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_150%.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_150%_BB_Fold_BTN_Call.txt"),
        template_basename=(
            "Cash6max_100bb_BTN_open_SB_3bet_BTN_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG, HJ, CO fold, BTN opens 2.5bb, SB 3-bets, BB folds, BTN calls"),
    ),
    # Scenario 10 (May 2026): UTG opens, BB 3-bets, UTG calls (3BP).
    # Pack uses BB 3-bet token '155%'. Tightest BB 3-bet range, since
    # UTG opens tightest. From docs/ryan_range_pack_index.md scenario #10.
    "Cash6max_100bb_UTG_open_BB_3bet_UTG_call": _3bp_spec(
        name="Cash6max_100bb_UTG_open_BB_3bet_UTG_call",
        oop_position="BB", ip_position="UTG",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/UTG/"
                  "UTG_60%_HJ_Fold_CO_Fold_BTN_Fold_SB_Fold_BB_155%_UTG_Call.txt"),
        template_basename=(
            "Cash6max_100bb_UTG_open_BB_3bet_UTG_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG opens 2.5bb, HJ, CO, BTN, SB fold, BB 3-bets, UTG calls"),
    ),
    # Scenario 11 (May 2026): BTN opens, BB 3-bets, BTN 4-bets, BB calls
    # (4BP). Pack uses BTN 4-bet token '50%'. BB's call range (6.4%) is
    # WIDER than BTN's 4-bet range (3.3%) because BB's 3-bet contains
    # semi-bluffs that defend by calling vs a 4-bet rather than folding.
    # From docs/ryan_range_pack_index.md scenario #11.
    "Cash6max_100bb_BTN_open_BB_3bet_BTN_4bet_BB_call": _4bp_spec(
        name="Cash6max_100bb_BTN_open_BB_3bet_BTN_4bet_BB_call",
        oop_position="BB", ip_position="BTN",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_50%_BB_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_182%_BTN_50%.txt"),
        template_basename=(
            "Cash6max_100bb_BTN_open_BB_3bet_BTN_4bet_BB_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG, HJ, CO fold, BTN opens 2.5bb, SB folds, "
            "BB 3-bets, BTN 4-bets, BB calls"),
    ),
    # Scenario 12 (May 2026): CO opens, BTN 3-bets, CO 4-bets, BTN calls
    # (4BP). Pack uses CO 4-bet token '95%' (vs '50%' for BTN/HJ/UTG
    # 4-bets) -- notably larger. From docs scenario #12.
    "Cash6max_100bb_CO_open_BTN_3bet_CO_4bet_BTN_call": _4bp_spec(
        name="Cash6max_100bb_CO_open_BTN_3bet_CO_4bet_BTN_call",
        oop_position="CO", ip_position="BTN",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/CO/"
                   "UTG_Fold_HJ_Fold_CO_60%_BTN_77%_SB_Fold_BB_Fold_CO_95%.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/BTN/"
                  "UTG_Fold_HJ_Fold_CO_60%_BTN_77%_SB_Fold_BB_Fold_CO_95%_BTN_Call.txt"),
        template_basename=(
            "Cash6max_100bb_CO_open_BTN_3bet_CO_4bet_BTN_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG, HJ fold, CO opens 2.5bb, BTN 3-bets, "
            "SB and BB fold, CO 4-bets, BTN calls"),
    ),
    # Scenario 13 (May 2026): HJ opens, BB 3-bets, HJ 4-bets, BB calls
    # (4BP). Pack uses HJ 4-bet token '50%'. Extremely tight HJ 4-bet
    # range (no full-weight hand class other than AA). From docs #13.
    "Cash6max_100bb_HJ_open_BB_3bet_HJ_4bet_BB_call": _4bp_spec(
        name="Cash6max_100bb_HJ_open_BB_3bet_HJ_4bet_BB_call",
        oop_position="BB", ip_position="HJ",
        oop_range=("ranges/ryan_preflop_tree/"
                   "PioViewer - NLH 6max 100bb 2.5x Open/BB/"
                   "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%_HJ_50%_BB_Call.txt"),
        ip_range=("ranges/ryan_preflop_tree/"
                  "PioViewer - NLH 6max 100bb 2.5x Open/HJ/"
                  "UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_182%_HJ_50%.txt"),
        template_basename=(
            "Cash6max_100bb_HJ_open_BB_3bet_HJ_4bet_BB_call_ryan_ranges.txt"),
        preflop_action_description=(
            "UTG folds, HJ opens 2.5bb, CO, BTN, SB fold, "
            "BB 3-bets, HJ 4-bets, BB calls"),
    ),
}


def get_solver_spec(name: str) -> SolverSpec:
    """Look up a solver spec by name. Clear error if missing."""
    try:
        return SOLVER_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(SOLVER_SPECS)) or "(none)"
        raise KeyError(
            f"no solver spec registered for {name!r}. Known specs: {known}. "
            f"Register one in pipeline/scenario_spec.py:SOLVER_SPECS first."
        ) from exc


__all__ = [
    "SolverSpec",
    "SOLVER_SPECS",
    "get_solver_spec",
    "BTN_OPEN_100BB_PLACEHOLDER",
    "BB_CALL_VS_BTN_OPEN_PLACEHOLDER",
]
