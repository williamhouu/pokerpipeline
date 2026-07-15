# `animation_script` column: format spec for the app team

*Written July 2026. Format versions covered: 1 and 2. Real sample payloads
(pretty-printed, generated from an actual v7 solve batch) are in
`docs/animation_script_samples/`.*

## What it is

Every question row in the CSV now ends with an `animation_script` column:
one self-contained JSON blob describing the full chip-and-card timeline of
the hand up to the moment the question is asked. The app should animate the
table from this blob and from nothing else. Do not parse the Question prose;
the blob is built from the same resolved amounts as the prose and the seat
tokens, so the two always agree, and the blob carries things the prose
deliberately skips (blind posts, every preflop fold in order).

It is emitted by all four writers (standalone postflop, full-hand
play-throughs, standalone NLHE preflop, PLO), always as the LAST column,
with one identical grammar. The only differences by pipeline are which event
types actually appear (a preflop question has no `deal` events) and whether
dollar fields are present (cash yes, tournament no).

Guarantees you can build against:

- **Self-contained.** No other column is needed to render the animation.
- **Renderer does zero arithmetic.** Every chip event carries the pot and
  the actor's remaining stack AFTER the event.
- **Deterministic.** Regenerating a batch reproduces the same JSON byte for
  byte, and our audit tooling re-verifies the column on every batch.
- **Compact JSON** (no whitespace) in the CSV cell. The samples in the
  samples folder are pretty-printed for readability only.

## Top-level shape

```json
{
  "version": 1,
  "table_size": 8,
  "seats": ["UTG", "UTG+1", "LJ", "HJ", "CO", "BTN", "SB", "BB"],
  "hero_seat": "BTN",
  "sb_bb": 0.5,
  "bb_bb": 1.0,
  "starting_stack_bb": 200.0,
  "bb_dollars": 2.0,
  "events": [ ... ]
}
```

| Field | Type | Notes |
|---|---|---|
| `version` | int | `1` = base timeline. `2` = same, plus a `resolution` object (see below). Treat unknown future versions as "render what you understand". |
| `table_size` | int | Number of seats dealt in. |
| `seats` | string[] | All seats in preflop acting order (first entry acts first preflop). Use it to lay out the table and to know every player's starting position. |
| `hero_seat` | string | The seat the user is playing. Always one of `seats`. |
| `sb_bb`, `bb_bb` | number | Blind sizes in bb (0.5 / 1.0 today). |
| `starting_stack_bb` | number | Every player's starting stack in bb. |
| `bb_dollars` | number | Dollar value of 1bb. **Cash games only; absent on tournaments.** Its presence is also your signal for whether `*_dollars` twins exist on events. |
| `events` | object[] | The ordered timeline. Ends at a `decision` event. |

## Events

Every event has `"i"` (1-based sequence number) and `"type"`. Types:

### Chip events: `post`, `raise`, `bet`, `call`

```json
{"i": 8, "type": "raise", "seat": "BTN",
 "amount_bb": 3.0, "to_bb": 3.0, "pot_bb": 4.5, "stack_bb": 197.0,
 "amount_dollars": 6.0, "to_dollars": 6.0, "pot_dollars": 9.0, "stack_dollars": 394.0}
```

| Field | Meaning |
|---|---|
| `seat` | Who acts. |
| `amount_bb` | Chips ADDED to the pot by this event (the amount that slides in). |
| `to_bb` | The actor's total in front for this street after the event. Present on `post`, `raise`, `bet`; absent on `call` (a call just matches). |
| `pot_bb` | Pot AFTER the event. |
| `stack_bb` | The actor's remaining stack AFTER the event. |
| `all_in` | `true` when the wager or call is all-in. Absent otherwise. |
| `*_dollars` | Dollar twin of each bb field. Cash games only. |

`post` = a blind. `bet` = first wager on a postflop street. `raise` =
a preflop open/3-bet/4-bet or a postflop raise. `call` = matching a wager.

Amounts are exact 2-decimal numerics (a 33% pot bet can be `2.14`).
If the app wants prettier numbers, display-rounding (we use a 0.5bb grid
elsewhere) is the renderer's choice; the JSON keeps the truth.

### `fold`

```json
{"i": 3, "type": "fold", "seat": "UTG"}
```

One event PER fold, never a grouped list, and the order is meaningful: in a
3-bet pot the SB's fold comes after the open, not with the early-position
folds. If you want one visual sweep for consecutive folds, coalesce adjacent
fold events at render time.

### `check`

```json
{"i": 12, "type": "check", "seat": "BB"}
```

### `deal`

```json
{"i": 11, "type": "deal", "street": "flop", "cards": ["Kd", "7s", "3s"]}
```

Marks a street being dealt. `street` is `flop` / `turn` / `river`; the flop
carries 3 cards, turn and river carry 1. Card notation is rank + suit
letter: ranks `2-9`, `T`, `J`, `Q`, `K`, `A`; suits `c` `d` `h` `s`.
Betting on the new street starts fresh after a deal (that is why `to_bb`
resets). Preflop-only questions have no `deal` events.

### `decision` (the terminal event of the main timeline)

```json
{"i": 19, "type": "decision", "seat": "BTN", "street": "river"}
```

The timeline stops here: this is the seat and street the question asks
about, and where the app should pause the animation and show the options.
`seat` always equals `hero_seat`.

There is deliberately NO showdown in the main timeline. The question ends at
the decision, and the opponent holds a range, not a specific hand. The one
sanctioned exception is the version-2 `resolution` below.

## Version 2: the `resolution` object (full-hand final legs only)

Full-hand play-throughs are linked rows sharing a `hand_id`, ordered by
`sequence_index` (preflop leg first, river leg last). On the LAST leg of a
hand, when the hand's correct final action genuinely ends the hand, the blob
is `"version": 2` and carries a `resolution`: the closing sequence to play
AFTER the user answers.

```json
"resolution": {
  "vindicates": "Call",
  "villain_seat": "BB",
  "villain_cards": ["Qh", "9h"],
  "summary": "The Big Blind shows Q❤️9❤️ (high card). Your ace high wins the pot.",
  "events": [
    {"i": 1, "type": "call", "seat": "BTN", "amount_bb": 5.72, "pot_bb": 22.22, ...},
    {"i": 2, "type": "reveal", "seat": "BB", "cards": ["Qh", "9h"], "hand_label": "high card"},
    {"i": 3, "type": "reveal", "seat": "BTN", "cards": ["Ac", "4c"], "hand_label": "ace high"},
    {"i": 4, "type": "win", "seat": "BTN", "pot_bb": 22.22, "pot_dollars": 44.44}
  ]
}
```

| Field | Meaning |
|---|---|
| `vindicates` | The correct answer this resolution proves right (matches the row's Correct Answer). |
| `villain_seat`, `villain_cards` | The revealed opponent hand. |
| `summary` | A ready-made one-liner for the result screen. Suit emojis included. Use it as-is if you want zero text work. |
| `events` | The closing timeline. **Numbered from 1 again**, and its pot/stack numbers CONTINUE exactly from where the main timeline's decision event left off. |

Resolution events reuse the same grammar plus two new types:

- **`reveal`**: `seat`, `cards`, `hand_label` (plain-English made-hand
  name, e.g. `"top pair"`, `"a flush"`), and `best_five` (the exact five
  cards that make the hand, drawn from hole cards + board; highlight them
  on the table when showing the reveal). May carry `"folded": true` when
  the opponent folds to hero's bet but we still show what they folded.
- **`win`**: the pot push plus everything the result banner needs, so you
  never infer:
  - `seat`: who wins the pot.
  - `reason`: `"showdown"` (cards decided it) or `"fold"` (someone folded).
  - `hand_label`: the WINNING hand's name, e.g. `"a full house"` for a
    banner like "You win with a full house". It belongs to whoever `seat`
    names — usually the hero, but on a losing closing check it is the
    OPPONENT'S hand (banner reads from `summary`: "Checking behind lost
    the minimum"). Present on showdown wins ONLY; a fold win has no
    showdown hand, use the `summary` line instead.
  - `pot_bb` (and `pot_dollars` on cash): the pot being pushed.
  - `stack_bb` (and `stack_dollars`): the winner's stack AFTER collecting
    the pot; set their stack display to this, same as any chip event.

The sequence in `events` is always: hero's correct action, then (only when
hero bet or raised) the opponent's invented response (a call or a fold,
never a raise), then any remaining board cards as `deal` events (an all-in
call before the river deals out the rest of the board inside the
resolution), then the reveal(s), then `win`. The opponent's hand is always
revealed, with `"folded": true` when they folded it, so the user sees the
bluff worked or the fold was right. Hero's hand is revealed only when the
hand actually reaches showdown (hero did not fold and the opponent did not
fold).

How the revealed hand is chosen (so you can trust it, not so you need to do
anything): it is sampled deterministically from the opponent's REAL solver
range for the exact line they played, restricted to the slice that proves
the correct answer right (call beats a weaker hand, fold ducked a stronger
one, value bet gets paid by worse, bluff folds out better). Ties never
qualify. When no such hand exists in the range, no resolution is attached
and the row stays version 1, so a final leg being version 1 is normal, not
an error.

## What appears where

| Row kind | version | deal events | resolution |
|---|---|---|---|
| Standalone NLHE preflop question | 1 | never | never |
| Standalone PLO preflop question | 1 | never | never |
| Standalone postflop question | 1 | yes | never |
| Full-hand leg (not last, or no honest ending) | 1 | preflop leg: no; postflop legs: yes | no |
| Full-hand FINAL leg with an honest ending | 2 | yes | yes |

Notes for full hands specifically:

- Group rows by `hand_id`, order by `sequence_index`. Each leg's blob
  replays the whole hand from the blinds up to THAT leg's decision, so every
  leg is independently renderable from scratch. If you want to fast-forward
  instead: drop the previous leg's terminal `decision` event, and what
  remains is an exact prefix of the next leg's event list (the decision is
  replaced by the action the hand actually took); play only the new tail.
  This prefix property is verified in our test suite.
- In a 3-bet pot the opener gets TWO preflop legs (the open decision, then
  the facing-a-3-bet decision). The second leg's timeline includes the open
  and the 3-bet, ending at the new decision. See
  `animation_script_samples/3bp_leg2_preflop_facing_3bet.json`.

## Endings: every case the result screen must handle

1. **Showdown win/loss** (`win.reason == "showdown"`): both `reveal`
   events present, `win.hand_label` names the winning hand. Banner
   pattern: "You win with a full house" / use `resolution.summary` as the
   ready-made line (it covers the losing-check case too: "Checking behind
   lost the minimum").
2. **Hero's bluff works** (`win.reason == "fold"`, winner is hero): the
   opponent's reveal has `"folded": true` -- show their surrendered hand
   face-up (that is the teaching moment), then push the pot to hero. No
   `win.hand_label` (hero never showed).
3. **Hero's good fold** (`win.reason == "fold"`, winner is the opponent):
   the opponent's stronger hand is revealed, pot goes to them, and the
   `summary` says folding saved money. Hero never reveals.
4. **All-in call before the river**: the resolution contains the
   remaining `deal` events (turn/river) BEFORE the reveals -- animate the
   runout, then the reveals, then the pot push.
5. **No resolution at all** (final leg still `"version": 1`): a normal,
   expected case (roughly: the correct final action does not end the hand,
   or no honest vindicating hand exists in the opponent's real range). End
   on the decision + answer explanation; no reveal, no pot push. Do not
   invent an ending.
6. **Ties/chops never appear** (excluded by design), so a split-pot
   animation is not needed.
7. **The user answered WRONG**: the resolution still plays the CORRECT
   action (that is the design -- the closing sequence demonstrates why the
   right answer is right; the explanation text handles the user's
   mistake). There is no alternate timeline per wrong option.
8. Only FULL-HAND final legs carry resolutions. Standalone questions
   (preflop, PLO, standalone postflop) always end at the decision.
9. **Preflop fold ending** (balanced-lengths batches): a hand can end on
   its FIRST question -- hero correctly folds preflop. When the fold is to
   a raise, the resolution reveals the raiser's stronger starting hand
   (`best_five` is just their two cards; no board exists). A first-in fold
   (open-folding) has NO resolution -- the blinds take it; end on the
   explanation.

## Do not reveal the hand's length up front

Balanced batches deliberately mix hands ending preflop, on the flop, on
the turn, and on the river in EQUAL shares, so a user can never reason
"there are more questions, therefore folding is wrong". That only works if
the UI hides the total: do not show "Question 2 of 5" before the hand is
over. Show progress as the street ("Flop decision") or reveal the count
after the hand ends. `sequence_total` is in the CSV for grouping -- treat
it as internal.

## Sample files

All in `docs/animation_script_samples/`, generated from a real 8-max 200bb
BTN-vs-BB v7 solve (dry-run batch, so the prose is placeholder but every
number and card is real):

| File | Shows |
|---|---|
| `srp_leg1_preflop.json` | Preflop leg: blinds, five folds, decision at the BTN. Simplest possible timeline. |
| `srp_leg2_flop.json` | Flop leg: same preflop line completed, flop dealt, decision. |
| `srp_leg3_turn.json` | Turn leg: adds flop checks, turn deal, a turn bet and call. |
| `srp_leg4_river_with_resolution.json` | Final river leg, version 2: facing a river bet, resolution = hero calls, both reveal, hero wins. |
| `srp_final_fold_resolution.json` | Final leg where the correct answer is Fold: opponent reveals the stronger hand, opponent wins. |
| `3bp_leg2_preflop_facing_3bet.json` | The 3-bet-pot opener's second preflop leg (open, 3-bet, decision back on the opener). |

## Renderer checklist

1. Parse the cell as JSON; branch nothing on pipeline, only on `version`
   and event `type`.
2. Seat players from `seats` with `starting_stack_bb` each; highlight
   `hero_seat`.
3. Play `events` in order. Chip events: move `amount_bb` in, set the pot
   display to `pot_bb` and the actor's stack display to `stack_bb` (never
   compute these yourself). Show dollars instead of bb by using the
   `*_dollars` twins when present.
4. At `decision`, pause and present the question.
5. Version 2 only, after the user answers: play `resolution.events` the
   same way, then show `resolution.summary`.
6. Ignore unknown fields and unknown event types gracefully; the format is
   versioned and may grow.
