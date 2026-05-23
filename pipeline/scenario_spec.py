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
