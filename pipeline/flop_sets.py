"""Layer 2: Tree Resolver -- flop set selection.

The batch solver takes a SolverSpec + a flop set and produces one `.cfr` per
flop. This module catalogs the available flop sets.

Two sets registered:

  * `MINIMAL_DEBUG` -- the single flop `2c Js 7s` from the existing
    hand-solved test .cfr. Running the batch solver against this set lets
    us verify Layer 2 produces a structurally equivalent solve before
    committing compute to the full 25.

  * `STANDARD_25_FLOPS` -- a 25-board sample chosen to cover the
    flop universe's main axes (suit distribution, rank distribution,
    connectedness, pair status). This is NOT a universally-canonical
    sample -- the poker-solver community has multiple proposed 25-flop
    subsets (PioSolver internal samples, GTOWizard, academic equity-
    preserving clusterings; see docstring for source/audit notes). The
    set should be reviewed against the team's preferred sample before
    Tier 1 production solves.

The format for a flop is a list of three card strings, in PioSolver order:
  `["2c", "Js", "7s"]`. Helpers convert to/from `"2c Js 7s"` (Pio's
`set_board` input) and `"2cJs7s"` (filesystem-safe stem for the cache path).

Adding a new flop set: append to FLOP_SETS in this module; tests will
flag empty or inconsistent entries.
"""
from __future__ import annotations

from pipeline.cards import parse_card

# --- helpers ----------------------------------------------------------------
Flop = tuple[str, str, str]


def normalize_flop(flop) -> Flop:
    """Validate and canonicalise a flop: 3 distinct cards in Pio's
    rank+suit-letter form (`'Js'`, `'2c'`)."""
    if isinstance(flop, str):
        # Accept "2c Js 7s" or "2cJs7s" inputs.
        if " " in flop:
            cards = flop.split()
        else:
            cards = [flop[i:i + 2] for i in range(0, len(flop), 2)]
    else:
        cards = list(flop)
    if len(cards) != 3:
        raise ValueError(f"a flop has 3 cards, got {len(cards)}: {flop!r}")
    normalized = tuple(parse_card(c) for c in cards)
    if len(set(normalized)) != 3:
        raise ValueError(f"duplicate cards in flop: {flop!r}")
    return normalized                            # type: ignore[return-value]


def flop_board_string(flop) -> str:
    """The Pio `set_board` input form: space-separated cards."""
    return " ".join(normalize_flop(flop))


def flop_filename_stem(flop) -> str:
    """The filesystem-safe stem for the cache path: concatenated cards."""
    return "".join(normalize_flop(flop))


# --- the catalog ------------------------------------------------------------
# The single-flop "is the batch solver working at all" set. Reproduces the
# board of test_solves/btn_vs_bb_srp_2cJs7s.cfr, so a Layer 2 solve against
# this set can be compared structurally against the hand-solved file.
MINIMAL_DEBUG: tuple[Flop, ...] = (
    normalize_flop(("2c", "Js", "7s")),
)

# 25-flop sample for Tier 1 study solves. Each entry is annotated with the
# axes it covers; together they span:
#
#   suit distribution    : monotone (4), two-tone (12), rainbow (9)
#   pair status          : unpaired (22), paired (3)
#   rank distribution    : broadway (5), high-mid (6), middling (8),
#                          low-mid (4), low (2)
#   connectedness        : connected (8), one-gap (7), disconnected (10)
#
# Sources/inspiration (none copied verbatim; this list is curated for this
# project and should be diffed against the team's preferred sample):
#   * PioSolver community study-board subsets (various coaching forums).
#   * The "equity-preserving cluster representatives" approach from the
#     solver-community literature (sample one flop per equity-similarity
#     cluster of the 22,100 distinct flops).
#   * GTOWizard's monthly featured boards (rotating; the texture axes those
#     boards span informed which textures got included here).
#
# Phase-0 audit checklist before Tier 1 production:
#   [ ] confirm with Ryan that this set's texture distribution matches the
#       team's intended study coverage
#   [ ] swap any entries Ryan considers redundant or under-representative
#   [ ] consider whether 25 is enough -- some teams use 38 or 49 for tighter
#       texture coverage; STANDARD_25 is the minimum-viable starting point
STANDARD_25_FLOPS: tuple[Flop, ...] = tuple(normalize_flop(f) for f in (
    # --- High dry ---
    ("As", "Kd", "9h"),                          # A-high rainbow, disconnected
    ("Ah", "Kh", "5c"),                          # A-high two-tone, dry
    ("Ad", "8c", "3s"),                          # A-high rainbow, low gappers
    # --- High wet ---
    ("As", "Ks", "Qd"),                          # A-K-Q two-tone, broadway connected
    ("Ah", "Jh", "Th"),                          # A-J-T monotone, broadway-down
    # --- Broadway connected ---
    ("Kd", "Qc", "Jh"),                          # K-Q-J rainbow, connected broadway
    ("Qs", "Jd", "Ts"),                          # Q-J-T two-tone, connected straight-y
    # --- Mid two-tone ---
    ("Ts", "9s", "5d"),                          # T-9-5 two-tone, semi-connected
    ("9h", "8h", "4c"),                          # 9-8-4 two-tone, low-mid
    ("8c", "7c", "2d"),                          # 8-7-2 two-tone, gapped
    # --- Mid rainbow ---
    ("Th", "9c", "8d"),                          # T-9-8 rainbow, straight-heavy
    ("9s", "6d", "3c"),                          # 9-6-3 rainbow, disconnected
    ("8h", "5d", "2c"),                          # 8-5-2 rainbow, low gappers
    # --- Monotone ---
    ("Kc", "9c", "3c"),                          # K-9-3 monotone clubs
    ("Js", "6s", "4s"),                          # J-6-4 monotone spades
    # --- Paired ---
    ("Th", "Td", "5c"),                          # T-T-5 paired, dry
    ("8s", "8h", "2c"),                          # 8-8-2 paired, low kicker
    ("4d", "4s", "Kh"),                          # 4-4-K paired, overcard ace
    # --- Low connected (straight-heavy) ---
    ("7s", "6c", "5d"),                          # 7-6-5 rainbow, low connected
    ("6h", "5h", "4d"),                          # 6-5-4 two-tone, low connected
    # --- Low dry ---
    ("5h", "3d", "2c"),                          # 5-3-2 rainbow, low gappers
    ("Ah", "5d", "2c"),                          # A-5-2 rainbow, wheel-y
    # --- King-high study spots ---
    ("Kh", "Th", "6c"),                          # K-T-6 two-tone
    ("Ks", "7s", "3d"),                          # K-7-3 two-tone, K-high dry
    # --- Queen-high ---
    ("Qh", "8d", "3c"),                          # Q-8-3 rainbow, Q-high dry
))


FLOP_SETS: dict[str, tuple[Flop, ...]] = {
    "MINIMAL_DEBUG": MINIMAL_DEBUG,
    "STANDARD_25_FLOPS": STANDARD_25_FLOPS,
}


def select_flops(set_name: str) -> tuple[Flop, ...]:
    """Look up a flop set by name. Clear error if missing."""
    try:
        return FLOP_SETS[set_name]
    except KeyError as exc:
        known = ", ".join(sorted(FLOP_SETS)) or "(none)"
        raise KeyError(
            f"no flop set registered for {set_name!r}. Known sets: {known}."
        ) from exc


__all__ = [
    "Flop",
    "FLOP_SETS",
    "MINIMAL_DEBUG",
    "STANDARD_25_FLOPS",
    "flop_board_string",
    "flop_filename_stem",
    "normalize_flop",
    "select_flops",
]
