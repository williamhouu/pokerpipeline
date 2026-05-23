"""Build a Ryan-ranges template for one of the registered scenarios.

For each scenario, this script:

  1. Reads PioSolver's shipped `2bpot-full.txt` template (the structural
     chassis: pot/stack geometry, bet sizes, full `add_line` tree, etc.).
  2. Reads the scenario's OOP/IP range files from Ryan's preflop pack at
     `ranges/ryan_preflop_tree/...`. Filenames come from the registry
     below, which mirrors the per-scenario mapping in
     `docs/ryan_range_pack_index.md`.
  3. Expands each 169-hand-class file to a 1326-combo weight vector in
     PioSolver's canonical hand-order via `pipeline.preflop_ranges`.
  4. Replaces the template's two `set_range` lines with the expanded vectors,
     leaves every other line intact (sizings, `add_line` tree, build_tree).
  5. Writes the result to
     `templates/Cash6max_100bb_<scenario>_ryan_ranges.txt`.

This is a one-time build step per scenario. Re-run only when Ryan ships an
updated pack, the source template changes, or a new scenario is added.

Usage:
    # Default scenario (BTN_open_BB_call) for backward compat:
    python scripts/build_ryan_ranges_template.py

    # New scenarios -- one of the keys in SCENARIO_REGISTRY below:
    python scripts/build_ryan_ranges_template.py --scenario CO_open_BB_call
    python scripts/build_ryan_ranges_template.py --scenario SB_open_BB_call
    python scripts/build_ryan_ranges_template.py --scenario HJ_open_BB_call

    # Dry-check: exit non-zero if the existing template would change.
    python scripts/build_ryan_ranges_template.py --scenario CO_open_BB_call --check
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.preflop_ranges import (                                       # noqa: E402
    expand_to_combo_weights, format_set_range_line, parse_range_file,
)

SOURCE_TEMPLATE = Path(r"C:\PioSOLVER\TreeBuilding\100bb\2bpot-full.txt")
RANGE_PACK_ROOT = (REPO_ROOT / "ranges" / "ryan_preflop_tree"
                   / "PioViewer - NLH 6max 100bb 2.5x Open")
TEMPLATES_DIR = REPO_ROOT / "templates"


# --- per-scenario registry --------------------------------------------------
@dataclass(frozen=True)
class ScenarioBuild:
    """Inputs for building one scenario's Ryan-ranges template.

    Fields:
      * key -- the scenario name; becomes the template filename suffix and
        the SolverSpec name (after the `Cash6max_100bb_` prefix).
      * description -- one-line headline for the header comment in the
        generated template; readable by humans opening the .txt.
      * oop_position / ip_position -- documentary; used only to label the
        header block. Pio doesn't care about position names.
      * oop_range_path / ip_range_path -- pack-relative paths to the two
        Ryan-pack `.txt` files. Resolved against `RANGE_PACK_ROOT` at
        build time.
      * oop_label / ip_label -- short label for the header block, e.g.
        "BB call vs CO" / "CO RFI".
    """

    key: str
    description: str
    oop_position: str
    ip_position: str
    oop_range_path: str
    ip_range_path: str
    oop_label: str
    ip_label: str


SCENARIO_REGISTRY: dict[str, ScenarioBuild] = {
    "BTN_open_BB_call": ScenarioBuild(
        key="BTN_open_BB_call",
        description="Cash 6-max 100bb BTN-vs-BB SRP",
        oop_position="BB", ip_position="BTN",
        oop_range_path="BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%_SB_Fold_BB_Call.txt",
        ip_range_path="BTN/UTG_Fold_HJ_Fold_CO_Fold_BTN_60%.txt",
        oop_label="BB call vs BTN open",
        ip_label="BTN RFI",
    ),
    "CO_open_BB_call": ScenarioBuild(
        key="CO_open_BB_call",
        description="Cash 6-max 100bb CO-vs-BB SRP",
        oop_position="BB", ip_position="CO",
        oop_range_path="BB/UTG_Fold_HJ_Fold_CO_60%_BTN_Fold_SB_Fold_BB_Call.txt",
        ip_range_path="CO/UTG_Fold_HJ_Fold_CO_60%.txt",
        oop_label="BB call vs CO open",
        ip_label="CO RFI",
    ),
    "SB_open_BB_call": ScenarioBuild(
        # BvB: postflop order is SB->BB so SB is OOP at the flop.
        key="SB_open_BB_call",
        description="Cash 6-max 100bb SB-vs-BB BvB SRP",
        oop_position="SB", ip_position="BB",
        oop_range_path="SB/UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76%.txt",
        ip_range_path="BB/UTG_Fold_HJ_Fold_CO_Fold_BTN_Fold_SB_76%_BB_Call.txt",
        oop_label="SB RFI (BvB)",
        ip_label="BB call vs SB open",
    ),
    "HJ_open_BB_call": ScenarioBuild(
        key="HJ_open_BB_call",
        description="Cash 6-max 100bb HJ-vs-BB SRP",
        oop_position="BB", ip_position="HJ",
        oop_range_path="BB/UTG_Fold_HJ_60%_CO_Fold_BTN_Fold_SB_Fold_BB_Call.txt",
        ip_range_path="HJ/UTG_Fold_HJ_60%.txt",
        oop_label="BB call vs HJ open",
        ip_label="HJ RFI",
    ),
}


def _header_for(scenario: ScenarioBuild) -> str:
    return (
        f"# {scenario.description} -- ranges from Ryan's preflop pack.\n"
        f"#\n"
        f"# Generated by scripts/build_ryan_ranges_template.py from:\n"
        f"#   OOP ({scenario.oop_label}): ranges/ryan_preflop_tree/.../{scenario.oop_range_path}\n"
        f"#   IP  ({scenario.ip_label}): ranges/ryan_preflop_tree/.../{scenario.ip_range_path}\n"
        f"#\n"
        f"# Sizings (add_line entries, #FlopConfig.BetSize, etc.) and the build_tree\n"
        f"# command are cloned verbatim from PioSolver's shipped 2bpot-full.txt.\n"
    )


def _output_path(scenario: ScenarioBuild) -> Path:
    return (TEMPLATES_DIR
            / f"Cash6max_100bb_{scenario.key}_ryan_ranges.txt")


def build(scenario_key: str, *, check_only: bool = False) -> int:
    try:
        scenario = SCENARIO_REGISTRY[scenario_key]
    except KeyError:
        print(f"ERROR: unknown scenario {scenario_key!r}. "
              f"Known: {sorted(SCENARIO_REGISTRY)}", file=sys.stderr)
        return 2

    oop_path = RANGE_PACK_ROOT / scenario.oop_range_path
    ip_path = RANGE_PACK_ROOT / scenario.ip_range_path
    if not SOURCE_TEMPLATE.is_file():
        print(f"ERROR: source template not found: {SOURCE_TEMPLATE}",
              file=sys.stderr)
        return 2
    for label, path in (("OOP", oop_path), ("IP", ip_path)):
        if not path.is_file():
            print(f"ERROR: Ryan {label} range file not found: {path}",
                  file=sys.stderr)
            return 2

    print(f"Scenario             : {scenario.key} ({scenario.description})")
    print(f"Source template      : {SOURCE_TEMPLATE}")
    print(f"OOP range ({scenario.oop_position}):   "
          f"{oop_path.relative_to(REPO_ROOT)}")
    print(f"IP  range ({scenario.ip_position}):   "
          f"{ip_path.relative_to(REPO_ROOT)}")
    print()

    oop_classes = parse_range_file(oop_path)
    ip_classes = parse_range_file(ip_path)
    oop_weights = expand_to_combo_weights(oop_classes)
    ip_weights = expand_to_combo_weights(ip_classes)
    print(f"OOP ({scenario.oop_label:<22s}): {sum(oop_weights):>6.1f} "
          f"weighted combos ({100 * sum(oop_weights) / 1326:.1f}% of all hands)")
    print(f"IP  ({scenario.ip_label:<22s}): {sum(ip_weights):>6.1f} "
          f"weighted combos ({100 * sum(ip_weights) / 1326:.1f}% of all hands)")

    source_lines = SOURCE_TEMPLATE.read_text(encoding="utf-8").splitlines()
    new_oop_line = format_set_range_line("OOP", oop_weights)
    new_ip_line = format_set_range_line("IP", ip_weights)

    output_lines: list[str] = _header_for(scenario).splitlines()
    oop_replaced = ip_replaced = False
    for line in source_lines:
        stripped = line.lstrip()
        if stripped.startswith("set_range OOP "):
            output_lines.append(new_oop_line)
            oop_replaced = True
        elif stripped.startswith("set_range IP "):
            output_lines.append(new_ip_line)
            ip_replaced = True
        else:
            output_lines.append(line)
    if not oop_replaced or not ip_replaced:
        print(f"ERROR: source template missing set_range OOP/IP lines "
              f"(oop_found={oop_replaced}, ip_found={ip_replaced})",
              file=sys.stderr)
        return 3

    new_content = "\n".join(output_lines) + "\n"
    output = _output_path(scenario)

    if check_only:
        if not output.is_file():
            print(f"\ncheck: output template missing at {output}")
            return 1
        current = output.read_text(encoding="utf-8")
        if current == new_content:
            print(f"\ncheck: {output.relative_to(REPO_ROOT)} is up to date.")
            return 0
        print(f"\ncheck: {output.relative_to(REPO_ROOT)} is OUT OF DATE; "
              f"re-run without --check to regenerate.")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(new_content, encoding="utf-8")
    print(f"\nwrote: {output.relative_to(REPO_ROOT)} "
          f"({len(new_content):,} bytes)")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default="BTN_open_BB_call",
                        choices=sorted(SCENARIO_REGISTRY),
                        help="scenario key (default BTN_open_BB_call)")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if the existing output template "
                             "would change; do not write anything")
    args = parser.parse_args(argv)
    return build(args.scenario, check_only=args.check)


if __name__ == "__main__":
    sys.exit(main())
