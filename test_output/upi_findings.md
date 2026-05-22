# PioSolver Edge UPI write-side findings (Layer 2 verification)

Captured during the first real-solve attempt on MINIMAL_DEBUG. The Layer 2
architecture (scenario_spec, flop_sets, batch_solver, CLI) is correct; the
specific UPI command sequence used by `_configure_tree` needed adjustment
once Pio's actual write-side dialect was observed.

## What works (verified on PioSolver Edge 3)

| UPI verb | Example | Notes |
|---|---|---|
| `set_isomorphism` | `set_isomorphism 0 0` | Suits/board iso flags |
| `set_eff_stack` | `set_eff_stack 975` | Postflop chip stack |
| `set_pot` | `set_pot 0 0 55` | 3-tuple: invested OOP / IP / carried |
| `set_board` | `set_board 2cJs7s` | **No spaces.** Spaces → `incorrect or missing argument` |
| `set_range OOP` | `set_range OOP <1326 weights>` | Expects 1326 space-separated floats (combo-level, not 169 hand-level) |
| `set_range IP` | `set_range IP <1326 weights>` | Same |
| `set_bet_sizes OOP` | `set_bet_sizes OOP 33,75` | Plural verb; comma-separated %s |
| `is_tree_present` | `is_tree_present` | Returns `false` / `true` |
| `show_tree_params` | (no args) | Returns board / pot / bet_sizes / donk_bet config |
| `show_settings` | (no args) | Returns accuracy / accuracy_mode / thread_no / step |
| `show_memory` | (no args) | Returns total + available physical memory |

## What doesn't (and why)

Failed during probing — these names DON'T exist on Edge 3:

- `set_range_oop` / `set_range_ip`  → use `set_range OOP …` / `set_range IP …`
- `set_bet_size` (singular) → use `set_bet_sizes`
- `set_raise_size` / `set_raise_sizes` / `set_oop_raise_size` → **no equivalent verb found**
- `build_tree_postflop` / `init_tree` / `calc_tree` / `solve` → none recognised
- `help` / `list_commands` / `?` / `show_help` → no built-in help

## The actual write-side workflow

PioSolver Edge's GUI tree-builder doesn't expose a single high-level
"configure the tree" verb. Instead it produces a **tree-template `.txt`
file** under `C:\PioSOLVER\TreeBuilding\<stacks>\` (e.g.
`100bb/2bpot-full.txt`) whose bottom half is **literal UPI commands** the
GUI reads and issues line-by-line:

```
# metadata header lines start with '#'
#Type#NoLimit
#Range0#AA:0.1,KK:0.1,QQ:0.1,…       <-- human-readable range string
#Board#2c Js 7s                       <-- space-separated here, but…
#Pot#55
#EffectiveStacks#975
#FlopConfig.BetSize#65                <-- as % of pot
…
# blank line separates header from UPI commands
set_range OOP 0.9 0.9 0.9 … (1326 floats)
set_range IP   1   1   1 … (1326 floats)
set_board 2cJs7s                      <-- no spaces in the UPI form
set_eff_stack 975
set_isomorphism 1 0
set_pot 0 0 55
clear_lines
add_line 0 0 0 0 0 41 112 257 553 975
add_line 0 0 0 0 41 112 257 553 975
…  (~100 add_line entries, each a legal chip-amount sequence)
build_tree
```

So the right write-side architecture is:

1. **Use Pio's shipped templates as the source of truth.** They encode the
   exact UPI dialect Pio expects, including the chip-scale conventions
   (`#Pot#55` for the BTN-vs-BB SRP 100bb scenario rather than the 495
   chips a naive 90-chips-per-bb conversion would suggest).
2. **For each (scenario, flop), the only line that varies is `set_board`.**
   Ranges, pot, stack, sizings, add_lines all stay the same across flops
   in a scenario.
3. **Drive Pio by reading the template + issuing each non-comment line as
   a UPI command**, with the `set_board` line swapped to the target flop.
4. **After `build_tree`, add the run-side commands** Pio's template doesn't
   include: `set_accuracy <chips>`, `go`, `wait_for_solver`, `dump_tree
   <output.cfr>`.

## Chip-scale correction

The hand-solved `test_solves/btn_vs_bb_srp_2cJs7s.cfr` reports
`show_effective_stack() = 975` and matches Pio's `2bpot-full.txt` template
exactly. So the canonical Tier 1 chip scale is:

| Field | Value |
|---|---|
| `bb_in_chips` | ~10 (template uses 10; actual is 9.55 since pot=55 = 5.5bb) |
| `starting_postflop_stack_chips` | 975 |
| `pot_after_preflop_chips` | 55 |
| `accuracy_target_chips` | ~0.28 (0.5% of 55) |

The pre-pivot SolverSpec used 90 chips/bb (so pot=495, stack=8775) — that
shape works mathematically but produces a `.cfr` with a different
chip scale than the existing hand-solved file, making structural diffs
needlessly noisy. Switching to Pio's native scale lets the first real
batch solve be directly comparable to the hand-solved file.

## Templates available out of the box

In `C:\PioSOLVER\TreeBuilding\100bb\`:

```
2bpot-full+airiver.txt        2-bet pot, full tree, +all-in river
2bpot-full.txt                2-bet pot, full tree                <-- Tier 1 SRP
2bpot-nodonkbet.txt           2-bet pot, no donk bets
2bpot-oldstyle+donkbet.txt    older tree style + donk bets
2bpot-oldstyle.txt            older tree style
3bpot-full+airiver.txt        3-bet pot, full tree, +all-in river
3bpot-full.txt                3-bet pot, full tree                <-- Tier 1 3-bet pot
6maxBTNvsBB-3bet.txt          BTN vs BB 3-bet specific
SBvsBB100bb.txt               SB vs BB 100bb
SbvsBB100bb+ai.txt            SB vs BB 100bb with all-in
```

`2bpot-full.txt` is the closest match to our `Cash6max_100bb_BTN_open_BB_call`
scenario — it IS that scenario. The template's ranges aren't exactly Ryan's
preferred ranges yet (placeholders per the spec), but the geometry and
sizing tree are correct.

## Notes for the audit follow-up commit

- The current `pipeline/batch_solver.py:_configure_tree` issues commands
  from the spec's bet sizings; replace with a template-driven approach.
- The current `pipeline/scenario_spec.py` chip values are 90×bb; correct
  them to match Pio's template (10×bb for SRP postflop at 100bb).
- Add a `pio_template_path` field to SolverSpec referencing the template.
- `set_raise_size` has no UPI equivalent on Edge 3 — raise sizes are
  encoded into the `add_line` sequences directly. The spec's
  `raise_sizes_pct` field is documentary; the actual raise amounts come
  from the template.
