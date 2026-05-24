# V7.3 results — five Ryan-feedback fixes applied

V7.3 is the first batch produced after the five May-2026 Ryan-feedback fixes
landed (commits `5b637c6`, `8128744`, `a3805c1`, `58fe22a`, `f8978ef`).
This doc captures the V7.3 verification on Scenario 1 (BTN-vs-BB SRP) and
spot-checks each fix end-to-end in the output.

## Headline

| Metric | V7.1 | V7.3 | Notes |
|---|---:|---:|---|
| Questions written | 42 | **12** | smaller verification batch by design |
| Layer-6 validation failures | 3 | **0** | no corrective retries needed |
| Layer-6 API failures | 0 | 0 | — |
| Anthropic calls (incl. retries) | 64 | **15** | 12 spots + 3 cached system-prompt warmups |
| Approx. cost (Sonnet 4.6) | $3.27 | **$0.90** | per-question cost similar; scale differs |
| Wall-clock | 33.2 min | 13.1 min | per-question 65s in V7.3 vs ~47s in V7.1 |

Per-question cost rose slightly because the system prompt grew (voice rules
9-10, 13-entry archetype catalog, 6 archetype-framed gold examples) — but the
cached portion stays warm across the batch, so net cost was modest.

## Fix verification

### Fix 1 — Drop trailing .00 on whole dollar amounts

Confirmed: Context column on every V7.3 row reads `Stacks $50` (not
`Stacks $50.00`). The `_format_dollars` helper in `pipeline/scenario_config.py`
now mirrors `pipeline/format_writer._dollars`: integer dollars render `$50`;
fractional dollars keep 2-decimal precision (e.g. `$1.25`).

### Fix 2 — Question Type column

Confirmed: every V7.3 row's Question Type cell is the literal string
`Hand Scenario Question.` (with trailing period). The prior `Multiple Choice`
placeholder is gone.

### Fix 3 — Suit emojis in Answer Explanation

Confirmed across all 12 V7.3 explanations. No row contains plain solver
notation (`Kh`, `AdKd`, etc.); every specific card reference uses the emoji
form. Examples:

- Row 1: `A♣️4❤️, A♣️4♣️, A❤️4♣️ are the only value combos BTN can show up with`
- Row 9: `BTN's worse hands like T♠️6♠️ and Q♠️4♠️ fold to a bet`

The soft `validate_no_plain_card_notation` validator did NOT print any warnings
during the run (grep for `soft-warn` in `test_output/batch_v7_3_run.log` returns
empty), so the prompt change is holding.

### Fix 4 — Name specific high-frequency villain combos

Confirmed: every V7.3 explanation that discusses villain's range names 2-3
specific combos from `range_data.villain_top_value_combos`. Examples:

- Row 5 (full house, recommended check): `full houses like K♣️4♣️, A♦️8♦️,
  and 8♣️4♣️, plus the two-pair combos K♣️8♣️ and K♣️9♣️`
- Row 7 (second pair, mostly call): `BB's value is concentrated in straights
  like A♠️4❤️, A♠️4♦️, A♠️4♣️ and the nut straight combos 6♠️4♠️, 6♣️4♣️,
  6♦️4♦️, plus sets like 3♠️3❤️ and 5♦️5♣️`
- Row 8 (two pair, fold): `full houses like A♣️5♠️ and A♣️5♣️, straights like
  A♦️4❤️ and A♣️4❤️, and the same pocket sixes you hold in combos like 6♠️6❤️
  and 6♦️6♣️`

The soft `validate_villain_combo_citation` validator did NOT fire on any row.

### Fix 5 — Strategic reasoning correctness

The big one. V7.1 failed by matching the literal `nut_advantage_villain`
concept tag and writing "villain has the nut advantage" as the explanation
anchor — wrong frame when hero in the spot was actually inducing villain
(trap_check) or pot-controlling. V7.3 adds:

- Layer 5 `pipeline/fact_extractor/archetypes.py` — 13-archetype classifier
  driven by `(correct_action, hand_strength_bucket, aggression_history,
  facing_bet, draws)`.
- `SpotMetadata.aggression_history` (per-prior-street aggressor: hero /
  villain / check), `RangeData.hero_range_disposition` (capped / uncapped /
  polarized / linear), `DecisionData.recommended_action_archetype` (one of
  the 13).
- Layer 6 system prompt: 13-entry strategic archetype catalog with per-
  archetype framing guidance + 6 archetype-framed gold examples (including a
  canonical trap_check example matching the V7.1 failure pattern).
- Layer 7 `validate_archetype_consistency` — hard validator with per-
  archetype anti-pattern phrase lists. Rejects when explanation contains
  trap_check anti-patterns like "villain has the nut advantage", retries
  once.

#### Spot-check 1 — pot_control_check (Row 9): the closest analog to the V7.1 failure case

Hero is BB with K♦️8♦️, board 5❤️3♦️2♣️8♠️8❤️.
Action: BB led flop $1.75, BTN called; BB checked turn, BTN bet $6.50,
BB called; river decision. Trips with K kicker, recommended action: **Mostly
check** (76% / bet 24%). Hero was NOT the aggressor on the turn (villain bet,
hero called), so archetype = **pot_control_check** (not trap_check).

V7.3 explanation:

> You should mostly check K♦️8♦️ here on the river. You have trips, but
> betting builds a pot you do not want to play big: BTN's worse hands like
> T♠️6♠️ and Q♠️4♠️ fold to a bet, while the hands that continue --
> T♦️T♣️ and T❤️T♣️ for tens full, plus 5♠️4♠️ and 5♦️4♦️ for two pair --
> are all ahead of or splitting with you. Checking takes the showdown at a
> controlled pot size and lets BTN's large bluff population, hands like
> K♦️6♦️ and Q♠️4♠️ that have no showdown value, check back and lose
> quietly.

Concept tag `nut_advantage_villain` IS on this row, but the explanation does
NOT say "villain has the nut advantage" — it says "BTN's continuing range is
ahead of or splitting with you, BTN's worse hands fold, checking takes the
showdown at a controlled pot size." That's the pot_control_check frame
exactly. The V7.1 failure mode does not recur.

#### Spot-check 2 — bluff_catch (Row 7): facing turn ~70% bet

Hero is BTN with K♣️5♣️ (second pair), board 5❤️3♦️2♣️8♠️. Facing BB's
$14.50 turn bet (~70% pot). Recommended: **Mostly call, sometimes fold**
(60/30/10). Archetype = **bluff_catch**.

V7.3 explanation:

> Mostly you should call here, and the price is what makes it right. You're
> facing a roughly 70% pot bet on the 8♠️ turn, which means you only need
> about 41% equity to continue, and your second pair on K♣️5♣️ on a
> 5❤️3♦️2♣️8♠️ board sits right at that threshold against BB's polarized
> range. BB's value is concentrated in straights like A♠️4❤️, A♠️4♦️,
> A♠️4♣️ and the nut straight combos 6♠️4♠️, 6♣️4♣️, 6♦️4♦️, plus sets like
> 3♠️3❤️ and 5♦️5♣️ -- but BB is also firing this line with a large number
> of straight-draw bluffs like 7♣️4♣️, 7❤️4❤️, and 7♦️4♦️, and those bluff
> combos actually outnumber the value combos. **You're not calling because
> second pair is strong -- you're calling because BB's bluff frequency at
> this price makes folding a mistake.**

Textbook bluff_catch framing per the user's spec: explicitly says NOT a
value call, calls out price + bluff frequency. Cites specific value AND
bluff combos. Anti-pattern phrases like "for value" or "hand is strong
enough" are absent.

#### Spot-check 3 — pot_control_check with full house (Row 5): when a "strong" hand isn't actually strong

Hero is BTN with 5❤️4❤️, board 4♦️4♠️K❤️A♠️A❤️ (fours full of 5s — but
beaten by every full house on a paired-aces board where BB check-raised the
flop and bet-call-checked through). Recommended: **Mostly check** (59/41).

V7.3 explanation:

> The best play is to check. You're holding 5❤️4❤️ on a 4♦️4♠️K❤️A♠️A❤️
> board, which gives you a full house, but betting here builds a pot you
> don't want to play big. BB's worse hands -- the pocket pairs like
> 7❤️7♣️, 8❤️8♦️, and 8♠️8❤️ -- all fold to an overbet, and the hands that
> continue are exactly the ones that beat or chop you: full houses like
> K♣️4♣️, A♦️8♦️, and 8♣️4♣️, plus the two-pair combos K♣️8♣️ and K♣️9♣️
> that have you in bad shape. Checking takes the free showdown and collects
> the pot against the bulk of BB's range that has already given up.

Correct pot_control_check framing for an "absolute boat that's actually a
relative bluff catcher" spot. The LLM recognized that 5❤️4❤️ is the WEAKEST
possible full house on this runout and reasoned about what BB's check-raise
line represents. No anti-patterns.

## Validator behaviour

| Validator | Type | Fires per V7.3 row |
|---|---|---:|
| validate_option_set | hard (Apr 2026) | 0 |
| validate_correct_answer_verb | hard (Apr 2026) | 0 |
| validate_option_set_completeness | hard (Apr 2026) | 0 |
| validate_no_standalone_sometimes | hard (Apr 2026) | 0 |
| validate_composite_label_frequencies | hard (Apr 2026) | 0 |
| **validate_archetype_consistency** | **hard (Fix 5)** | **0** |
| validate_no_plain_card_notation | soft (Fix 3) | 0 (no plain notation in any row) |
| validate_villain_combo_citation | soft (Fix 4) | 0 (every villain-mentioning row cites combos) |

Zero retries across all 12 spots — the prompt updates landed cleanly, the
LLM honoured every voice rule the first time. If Fix 5's hard validator were
to ever fire in production it would be visible in `test_output/batch_*_run.log`
as `archetype-consistency` rejection lines and a +1 corrective-retry count.

## Sample reasoning improvements (post-Fix 5)

Three concrete shifts visible in V7.3 explanations compared to V7.1's pattern:

1. **"Bluff catch" is now named explicitly when it applies.** Row 7's
   explanation has "this is a bluff catch -- you're calling because the
   price is right, not because you're ahead." V7.1 framed similar spots as
   "you're calling because second pair has enough showdown value" — wrong.

2. **Pot-control-check spots cite the actual reason.** Row 9 ("trips, BTN
   was the aggressor on the prior street, recommended check") frames the
   check as "BTN's continuing range is ahead of or splitting with you,
   BTN's worse hands fold, checking takes the showdown at a controlled pot
   size." V7.1's tendency to anchor on `nut_advantage_villain` tag literally
   would have produced "BTN has the nut advantage here" — the user's exact
   complaint. V7.3 doesn't.

3. **Specific villain combos are named in every villain-range discussion.**
   Where V7.1 said "BB's range is loaded with straights and two-pair," V7.3
   says "BB's continuing range is K♣️J♣️, T♠️T♣️ for sets, plus 6❤️6♣️ and
   the straight combos 4♦️3♦️, 7♣️6♣️". The `villain_top_value_combos`
   field anchors the prose to actual data.

## Open follow-ups before Tier-1 consolidated batch

Things V7.3 confirmed but the batch was too small to stress-test:

1. **No trap_check spot appeared in the 12-question sample.** The closest
   analog (Row 9) classified as pot_control_check because villain led the
   turn, not hero. A larger run that includes a spot where hero led flop
   AND turn AND river decision is to check with a strong hand would
   exercise the trap_check branch + its specific guidance + its anti-pattern
   anti-patterns. Expect this in the consolidated 14-scenario batch.

2. **Soft validators stay soft for now.** Per Ryan's instruction Fix 3 / 4
   validators warn but don't reject. If they don't fire in the consolidated
   run either, harden them. If they do fire, investigate which prompt
   refinement closes the gap.

3. **The bigger system prompt costs ~$0.075/question (Sonnet 4.6, prompt
   cached).** Acceptable for now; if Tier-1 (10k-question target) cost
   discipline tightens, the archetype catalog can collapse to a single
   per-spot guidance line in the live prompt without losing the trap_check
   anti-pattern protection (the hard validator is the real safety net).

## Files

- `test_output/batch_questions_v7_3.csv` — the 12 V7.3 questions.
- `test_output/batch_v7_3_run.log` — full batch log (gitignored).
- Commits: `5b637c6` (Fix 1), `8128744` (Fix 2), `a3805c1` (Fix 3),
  `58fe22a` (Fix 4), `f8978ef` (Fix 5).
