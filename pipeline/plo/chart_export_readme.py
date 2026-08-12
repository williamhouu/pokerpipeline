"""The PLO chart export's README text (written to the export root by the CLI).

Kept as a module constant so tests can pin format-contract statements without
running an export. User-facing text: plain English, no em dashes, and the
solve vendor is never named.
"""

from __future__ import annotations

README_MD = """# PLO Preflop Chart Packs (6-Max)

One directory per pack. Each pack is a complete preflop decision tree for one
game type and stack depth, built for a filterable hand LIST UI plus a
"check any hand" lookup (PLO has 16,432 suit-isomorphic hand classes, so
there is no 13x13 grid like Holdem).

## File layout

```
<pack_id>/
    index.json              pack header + the full node tree (small, load first)
    nodes/<node>.json.gz    one hand list per decision node (lazy-load on demand)
```

`index.json` is plain JSON. The per-node hand lists are gzip-compressed JSON
(they are large and extremely compressible). For R2 / S3 style hosting either
upload the `.gz` bytes as-is with `Content-Encoding: gzip` (browsers and
`fetch()` decompress transparently) or decompress at upload time; the file
name maps 1:1 to the node key (see below). `index.json` stays small on
purpose so the app can load it eagerly and fetch node files lazily.

## index.json

Header fields:

- `pack_id`: stable id, e.g. `plo_6max_100bb`, `plo_mtt_6max_25bb`
- `game`: always `"PLO"`
- `format`: `"Cash"` or `"MTT"`
- `table`: `"6-Max"`
- `stack_bb`: effective stack in big blinds
- `rake`: human-readable rake label (e.g. `"5% up to 1bb"`, `"no rake (MTT)"`)
- `ante_bb`: present on MTT packs only (value 1). The 1bb big-blind ante is
  ALREADY included in every `pot_bb` and every pot-limit raise size in the
  tree. Never add it again.
- `seats`: the preflop acting order, display names: UTG, HJ, CO, BTN, SB, BB
- `hand_class_count`: 16432
- `node_count`: number of nodes in `nodes`
- `format_version`: 1
- `nodes`: the tree, keyed by node path key

## The tree walk contract

- A node's map key is the dot-joined action tokens of the history that
  reaches it. The ROOT (first decision, UTG first to act) is the empty
  string `""`.
- A child's key is `parentKey + "." + act.c` (for root children, just
  `act.c`).
- An absent child key means the walk ends there: either the line is not in
  the solve, or no hand ever reaches it (zero-frequency branches are pruned;
  the parent still lists the act, typically at an aggregate share of 0).

Per node:

- `pos`: the acting seat, display name (`UTG`, `HJ`, `CO`, `BTN`, `SB`, `BB`)
- `bl`: raise level so far (0 = unopened pot, 1 = facing an open,
  2 = facing a 3-bet, ...)
- `pot_bb`: the pot at the moment of decision, in big blinds (blinds and,
  on MTT packs, the 1bb ante included)
- `to_call_bb`: the unmatched amount in front of the actor (0 when nothing
  to call; for a first-in non-blind seat this is mechanically 1, the blind
  to match, even where the tree offers no limp)
- `acts`: the available actions, ordered least to most aggressive
  (fold, call, raises ascending by size, all-in last):
  - `c`: the action token (the child key suffix)
  - `k`: one of `fold | call | raise | allin`
  - `to_bb`: for raises and all-ins, the raise-TO total in big blinds
    (pot-limit resolved); for calls, the total the caller matches; 0 for
    folds
- `aggregate`: `{action label: integer percent}` summing to exactly 100.
  COMBO-WEIGHTED share of the acting player's whole reaching range that
  takes each action (each hand class weighted by its concrete combo count
  and by how often it reaches this node). A share can be 0 (present in the
  tree at negligible frequency).
- `hands_file`: relative path to this node's hand list

Action labels (the keys of `aggregate` and of every hand `mix`) are:
`Fold`, `Call`, `Raise <pct>%` (pot-percentage raise, e.g. `Raise 100%` is
the pot raise), `Min-raise`, `All-in`. They are unique within a node.

## nodes/<node>.json.gz

The file name is the node key with dots replaced by underscores; the root is
`root.json.gz`.

```
{
  "node": "<key>",
  "pack": "<pack_id>",
  "acts": ["Fold", "Call", ...],          // same order as index acts
  "hands": [ <entry>, ... ]               // canonical class order
}
```

Per hand entry:

- `h`: the canonical hand class string, exactly as used across the pack
  (e.g. `"(AK)(AK)"`, `"AA(2A)"`, `"KQJT"`). Cards inside one pair of
  parentheses share a suit; every card outside any group is in its own suit.
  Ranks read `AKQJT98765432` (use `T`, never `10`).
- `n`: how many concrete 4-card combos this class represents (1 to 24; all
  classes together cover the 270,725 dealable PLO hands)
- `b`: display bucket (see below)
- `s`: suit partition: `double-suited | single-suited | rainbow`
- `f`: structural facts (exact, from the pipeline's hand model, so chart
  tags always match question prose):
  - `pair`: `unpaired | one_pair | two_pair | trips | quads`
  - `suit`: `double_suited | single_suited | three_suited | monotone |
    rainbow` (the exact 5-way pattern; `s` above is the 3-way display
    partition)
  - `conn`: `rundown | one_gapper | two_gapper | connected | disconnected`
- `m`: `{action label: integer percent}`, summing to exactly 100, largest
  remainder rounding. CONDITIONAL on reaching this node: "of the times this
  hand gets here, how often it takes each action". Only actions the hand
  actually takes appear (a rounded-down sliver can appear as 0).

### The IN-RANGE rule

A node's file lists ONLY the hand classes that actually reach that spot
(strategy weight above zero in at least one of the node's action files,
folding included: a hand that arrives and then folds IS in range at the
node). A hand absent from a node's file does not reach that spot; the app
should show "not in range" for it. At the root every class is listed, so the
root file doubles as the full class catalog.

## Check-any-hand canonicalization

To look up four user-entered cards, map them to the canonical class string
with a one-time table built from the root file:

```
canonical_key(cards):                      # cards like ["Ah","Kh","As","Kd"]
    parsed = [(rank_value(c), suit_of(c)) for c in cards]   # A=14 .. 2=2
    best = None
    for perm in all 24 permutations of the four suit labels:
        mapped = sort([(rank, perm[suit]) for (rank, suit) in parsed])
        best = min(best, mapped)
    return best                            # suit-isomorphism invariant

build_lookup(root_hands):                  # once, from root.json.gz
    table = {}
    for entry in root_hands:
        cards = parse_class_string(entry.h)   # parens = shared suit; each
                                              # ungrouped card = a new suit
        table[canonical_key(cards)] = entry.h
    return table

lookup(user_cards) = table[canonical_key(user_cards)]
```

Two hands map to the same key exactly when they are the same up to suit
relabeling, which is exactly the class granularity of the pack. Normalize
`10` to `T` and reject duplicate cards before keying. Then check the hand at
any node by `h` (absent = not in range).

## Buckets (`b`), first match wins

1. `AAxx`: two or more aces (AAKK and AAAx land here)
2. `KKxx`: two or more kings
3. `QQ-TT`: highest paired rank is Q, J or T (QQJJ lands here)
4. `Two pair`: two distinct paired ranks (both 99 or below by this point)
5. `Low pair`: exactly one paired rank, 99 or below (trips and quads of a
   low rank land here too: the pair is what two hole cards can play)
6. `Rundown`: four unpaired cards with at most ONE missing rank in total
   across the hand, trying the ace both high and low (KQJT, JT98, T987,
   J987, A234, AKQJ). Exception: an all-broadway one-gapper is NOT a
   Rundown (it falls to Broadway).
7. `Broadway`: all four cards T or higher, unpaired, not already a Rundown
   (AKQT, AKJT, AQJT)
8. `Dangler`: three coordinated cards plus one disconnected card. Exact
   rule, ace tried high and low: exactly one card sits more than 4 ranks
   from its nearest other card, and the remaining three span at most 4
   ranks (KQJ2, AKQ2, JT92). 4 ranks is the widest gap at which two hole
   cards can still share a five-card straight.
9. `Other`: everything else (two-gappers like J986, doubly disconnected
   hands like KQ72, and so on)

## Suits (`s`)

- `double-suited`: two suits, two cards each (two possible flush suits)
- `rainbow`: four different suits (no flush possible)
- `single-suited`: everything else, i.e. exactly one suit with 2+ cards.
  This includes three-of-one-suit and monotone hands: a flush uses exactly
  two hole cards, so they still have just one flush suit. The exact 5-way
  pattern is in `f.suit` if the UI wants to badge them differently.

## Companion per-class file

A separate single-file export, `plo-charts-data.json`, carries the app's
phase-1 PER-CLASS chart data (the `pack -> depth -> heroSeat -> nodeKey ->
class -> [fold, call, raise]` skeleton from the app's Charts spec). It is
derived from EXACTLY the same pack data as this per-hand tree: same nodes,
same range files, same pot-limit resolution; the class triples are the
combo-weighted aggregates of the per-hand mixes in `nodes/*.json.gz`. This
directory is the per-hand layer that replaces the class lookup later with
no UI change. Note the two exports use different class taxonomies on
purpose: this directory's `b` bucket is the hand-list filter taxonomy;
the per-class file uses the app engine's own classifier.

## Numbers

- All amounts are in big blinds, pot-limit resolved from the blinds (SB 0.5,
  BB 1) plus, on MTT packs, the 1bb big-blind ante. Amounts are exact
  (rounded to 4 decimals), NOT snapped to a display grid.
- Every `aggregate` and every `m` sums to exactly 100.
- MTT packs are chip-EV solves with a big-blind ante and no rake. Cash packs
  are raked (see each pack's `rake` label).
"""
