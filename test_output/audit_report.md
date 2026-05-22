# Trust-chain audit: strategic content in v3 batch output

**Scope:** verify that strategic decisions (`correct_answer`, action frequencies, EVs)
come from PioSolver and not from the LLM, before sending any v3 output to the team.

**Method:** static code review of `pipeline/explanation_generator.py` and its upstream
producers in Layer 5; then a live spot-check of 5 rows from
`test_output/batch_30_questions_v3.csv`, walking the same node in the test
`.cfr` to compare raw Pio numbers against the CSV.

**Headline:** the **strategic decision** (which action is best at the spot) is
correctly Python-derived from the solver, with no LLM input — verified across all
5 spot-checks. The **framing around that decision** (which two actions to template,
"Mostly" vs "Always", attribution of range-mean EVs to specific combos) is
LLM-controlled and shows real defects today, including one row (Row 1) where the
LLM offered a call-vs-raise frame for a strategy that is mechanically call-vs-fold.
None of these defects is caught by the existing validator. Layer 7 will need to.

---

## (a) Part 1 — Code-level findings

### Q1. Is `correct_answer` pre-computed in Python from Pio, or does the LLM pick it?

**Partial.** The correct *action verb* (call / fold / bet / check / raise) is
Python-derived from Pio; the *option-string* the CSV column ultimately holds
(e.g. "Mostly call") is LLM-mapped from that verb. No Python check confirms
that the LLM's mapping is correct.

The flow:

1. **Pio →** path sampler reads `calc_ev(child)` and the range-weighted strategy
   for each child of the decision node, populating
   `path_sampler.ActionOption{label, frequency, ev}`
   (`pipeline/path_sampler.py:263-265`).
2. **Python decides the verb.** Fact extractor picks the **max-frequency**
   action (note: frequency, not EV — see Q1.5 below), canonicalises its label
   to a bare verb, and stores it on the data block:
   ```python
   # pipeline/fact_extractor/__init__.py:83
   correct = (_canonical(max(actions, key=lambda a: a.frequency).label)
              if actions else "")
   # ...
   correct_action=correct,
   ```
   `_canonical("bet 36")` returns `"bet"`. So `decision_data.correct_action` is
   a bare verb at this point.
3. **Verb is injected into the prompt** verbatim:
   ```python
   # pipeline/explanation_generator.py:267-274
   return (
       f"Stage: {meta.street}. Hero ({hero}) is deciding against "
       f"{villain}. Available actions in the solver: {actions}. The "
       f"solver-correct action is \"{decision.correct_action}\" "
       f"(frequency dominant). The explanation must justify exactly "
       f"that action."
   )
   ```
4. **LLM generates the option strings + picks `correct_answer`.** It receives
   "the solver-correct action is `call`" and must produce four `option_N`
   strings (in the style Python chose — see Q4) plus a `correct_answer` string
   that equals one of them exactly.
5. **Validator (Layer 6's own retry, not Layer 7) checks only**
   `correct_answer in options` (`pipeline/explanation_generator.py:422-424`).
   It does **not** verify that `correct_answer`'s verb matches
   `decision.correct_action`.

So an LLM that produced
`options=["Always raise", "Mostly raise", "Mostly fold", "Always fold"]` and
`correct_answer="Mostly raise"` for a spot where Python said `correct_action="call"`
would pass Layer 6's validator. The "the LLM never thinks about poker" principle
is enforced at the *verb* level but not at the *option-string* level.

### Q1.5. Frequency vs EV — minor finding

Line 83 picks the **max-frequency** action, not the max-EV one. At a converged
equilibrium (which Pio reaches solving to <0.5% pot exploitability per
CLAUDE.md) every mixed action has the same EV, so max-freq and max-EV
coincide. The 5 spot-checks confirm this empirically: `pio_top_freq ==
pio_top_ev` for every row. But the docstring on `_build_decision_data` still
calls this "best vs second-best **action**" — worth noting that "best" here
means "most-played at equilibrium," not "highest-EV by inspection," in case a
future non-equilibrium solve violates the equivalence.

### Q2. Exact code path (recap)

```
PioSolver UPI calc_ev + show_strategy + show_range
    │
    ▼  path_sampler.PathSampler.build_spot_context (line 234-272)
ActionOption(label, frequency=range-weighted, ev=range-weighted-at-child)
    │
    ▼  fact_extractor.__init__._build_decision_data (line 52-94)
DecisionData.correct_action = _canonical(max-frequency action's label)   ← VERB
DecisionData.ev_gap_bb       = (top.ev - second.ev) / big_blind          ← CHIPS→BB
DecisionData.range_aggregate_strategy = {verb: freq, …}
DecisionData.options                  = [a.label for a in actions]
    │
    ▼  explanation_generator._question_framing (line 262-275)
String: "The solver-correct action is \"{verb}\" (frequency dominant)."
    │
    ▼  explanation_generator._build_messages_payload (line 342-377)
Anthropic Messages API call (system+gold cached, live block per-spot)
    │
    ▼  LLM emits JSON: {option_1, option_2, option_3, option_4, correct_answer, …}
    │
    ▼  parse_response + _validate (line 415-427)
Checks: correct_answer ∈ {option_1..option_4}.  Does NOT check verb match.
```

### Q3. Are option labels Python-computed or LLM-invented?

**LLM picks the strings; Python provides the raw material.** The LLM sees in
the SOLVER DATA block:
- `decision_data.options` = raw Pio labels (`["fold", "raise 608", "call"]`)
- `decision_data.range_aggregate_strategy` = `{"fold": 0.34, "raise": 0.001, "call": 0.66}`

…then writes four option strings per the style template Python chose (Q4
below). **No Python validation enforces that the LLM's option set ⊆ Pio's
actually-played actions.** Row 1 of the spot-check shows this matters: Pio's
strategy is call (65.6%) + fold (34.4%), with raise at 0.08% (essentially
never), yet the LLM produced options `["Always call", "Mostly call", "Mostly
raise", "Always raise"]` — omitting fold entirely and presenting the "wrong"
alternative as raise. The verb the LLM picked for `correct_answer` ("Mostly
call") still matches Python's `correct_action = "call"`, so Layer 6's
validator was happy, but the option set is mechanically misleading.

### Q4. Where is the Pio-frequency → option-style-label mapping?

**There is no deterministic Python mapping for `Always` vs `Mostly`.** The
LLM picks. Concretely:

- `_detect_option_style` (lines 147-169) selects **between** three styles
  using deterministic Python rules:
  - **`sizing`**: ≥ 2 distinct bet-size fractions in `option_pot_fractions`
  - **`frequency`**: any top action below 80% frequency
  - **`binary_action`**: otherwise (single dominant action ≥ 80%)
- `_option_style_instruction` (lines 172-203) gives the LLM **templates** to
  fill, e.g. for frequency style: *"Use exactly four options in this template:
  'Always <action A>', 'Mostly <action A>', 'Mostly <action B>', 'Always
  <action B>', where action A is the action the solver picks more often."*
- **No threshold is specified for which template slot to use as
  `correct_answer`.** A spot with freq 0.57 and a spot with freq 0.77 both get
  the same templates; the LLM decides whether "Mostly" or "Always" applies.

Observed in the 5 spot-checks:

| Pio top-freq | Style chosen | LLM's correct_answer |
|---:|---|---|
| 0.6557 | frequency | "Mostly call" |
| 0.9438 | binary_action | "Call" (no Mostly/Always distinction) |
| 0.5708 | frequency | "Mostly check" |
| 0.7653 | frequency | "Mostly bet" |
| 0.7310 | frequency | "Mostly bet" |

The LLM never picked "Always X" in this sample; everything frequency-style got
"Mostly". The 0.80 boundary between frequency and binary_action styles in
Python is the *only* deterministic frequency-derived label decision.

---

## (b) Part 2 — Live spot-check, 5 rows vs Pio raw

Test solve: `test_solves/btn_vs_bb_srp_2cJs7s.cfr`. Pio root effective stack =
975 chips → `big_blind = 9.75 chips/bb`.

### Summary table

| Row | Street | Hero | Pio top-freq action | Pio top-EV action | Pio ev_gap (bb) | CSV ev_gap (bb) | CSV correct_answer | Verb match | Gap match (±0.01) | Options OK |
|---:|---|---|---|---|---:|---:|---|:---:|:---:|:---:|
| 1  | Turn  | BTN (IP)  | `c` (0.6557) | `c` (-51.51 chips) | **2.826** | **2.83** | "Mostly call" | ✓ | ✓ | **✗** *(see notes)* |
| 6  | Turn  | BTN (IP)  | `c` (0.9438) | `c` (-435.53 chips) | **7.843** | **7.84** | "Call" | ✓ | ✓ | ✓ |
| 8  | Turn  | BB (OOP)  | `c` (0.5708) | `c` (+570.29 chips) | **1.349** | **1.35** | "Mostly check" | ✓ | ✓ | ✓ |
| 14 | River | BB (OOP)  | `b354` (0.7653) | `b354` (+87.84 chips) | **0.583** | **0.58** | "Mostly bet" | ✓ | ✓ | ✓ |
| 18 | River | BTN (IP)  | `b975` (0.7310) | `b975` (+261.12 chips) | **9.603** | **9.60** | "Mostly bet" | ✓ | ✓ | ✓ |

`Verb match`: does the verb of `correct_answer` ("call", "bet", …) match
Pio's max-freq action? `Gap match`: does the CSV's `ev_gap_bb` equal Pio's
`(best_ev - second_ev) / big_blind` to ±0.01 bb?

**ev_gap math is bit-exact (5/5).** The conversion convention pinned in
`pipeline/fact_extractor/__init__.py` and the test `test_ev_gap_bb_convention`
holds in production — no surprises.

**Verb mapping is correct (5/5).** Pio's max-freq action equals Pio's max-EV
action in every row (equilibrium intact), and the LLM's `correct_answer`'s
verb matches it in every row.

### Row 1 — significant LLM framing defect

Pio's strategy at `r:0:c:b36:c:8h:c:b125:b284` (BTN facing a check-raise on
the turn with pocket 2s, set on `2c Js 7s 8h`):

| Action | freq | EV (chips) | EV (bb) |
|---|---:|---:|---:|
| `b608` (re-raise) | **0.0008** | -79.06 | -8.11 |
| `c` (call) | 0.6557 | -51.51 | -5.28 |
| `f` (fold) | **0.3435** | -125.00 | -12.82 |

The real strategy is **call (66%) vs fold (34%)**. Raise is played 8 times in
10,000 — essentially never. Yet the CSV options are:

```
option 1: Always call
option 2: Mostly call
option 3: Mostly raise
option 4: Always raise
```

Fold is **absent**. A test-taker reads this as "the question is whether to
call or raise" — when mechanically it is "whether to call or fold."

The explanation text doubles down on the wrong frame:

> "The best play is to mostly call and only rarely raise. … Raising folds out
> all those draws, which make up a large portion of BB's range, and you want
> them to keep putting money in while they're behind. … the EV of calling is
> meaningfully better than raising, so trapping here is clearly correct."

Every appearance of "raise" should be "fold". The Python data block sent to
the LLM contains the correct strategy (`range_aggregate_strategy` includes
`fold: 0.3435`), so this is not a data error — it is an LLM judgement error
in choosing which two actions to template. No validator catches it.

### Row 6 — binary-action style, mostly correct, one minor prose nit

Strategy: call (94.4%) vs fold (5.6%), 9-of-10-played call. Options
`["Call", "Fold"]` correctly reflect Pio. Explanation citation matches:

> "Your range as a whole calls here at an overwhelming rate" — true; freq is 0.94.

Minor: explanation describes the board as "two-tone." The board at this spot is
`2c Js 7s 8h` — three suits (clubs, spades, hearts), not two-tone strictly.
Cosmetic; doesn't change the conclusion.

### Row 8 — solid all-around

Strategy: check (57.1%) vs bet 975/jam (42.9%). Options correctly cast as
check-vs-bet. EV math (check +58.49 bb, bet +57.14 bb) → gap 1.35 bb ✓.

Explanation cites verifiable facts:
- "set of sevens on this turn" ✓ (hero 7h 7c, board 2c Js 7s 8h)
- "action went bet, raise, re-raise, four-bet, call" ✓
  (action_sequence `b36 b102 b237 b512 c`)
- "Checking is actually slightly higher EV than betting" ✓
  (+58.49 vs +57.14)
- "SPR is under half a pot" — verifiable from `spot_metadata.spr` in the
  data block; not independently re-checked but qualitatively plausible at
  this stack depth after 4 raises.

No defects.

### Row 14 — solid; one citation worth flagging

Strategy: bet 354 (76.5%) vs check (23.5%). Options correctly cast as
bet-vs-check.

Explanation citation:
- "betting is still correct roughly 75% of the time" ✓
  (Pio freq is 0.7653 — explanation rounds to 75%)
- "overbet of around 115% pot" — derivable from
  `decision_data.option_pot_fractions["bet"]`. Not re-checked against raw chip
  math here; flagged as "trust the field."

### Row 18 — Pio numbers cited correctly, but attribution is misleading

Strategy: bet 975 (73.1%) vs check (26.9%). Options correctly cast as
bet-vs-check.

Explanation:

> "Your range as a whole bets this river about 73% of the time, and **Ks3s
> specifically** has an EV of around 27 big blinds when you bet versus only
> 17 big blinds when you check, a gap of nearly 10 big blinds."

Pio's range-weighted EVs: bet +26.78 bb, check +17.18 bb. The LLM rounded
these to 27 and 17 — **numerically accurate** to the data block. But it
attributed them to **"Ks3s specifically"** — implying these are the
hero-combo EVs.

They are not. The field the LLM read is `decision_data.hero_combo_evs` =
`{"bet": 26.78, "check": 17.18}`. Despite the field's name, it is
populated as **range-weighted aggregate EV per action**, not per-combo. See
`pipeline/fact_extractor/__init__.py:85`:
```python
hero_combo_evs={_canonical(a.label): a.ev / big_blind for a in actions},
```
where `a.ev` is the *range*-weighted mean from `path_sampler._weighted_mean`.

So the numbers are real Pio facts but the explanation attributes the
*range-mean* as if it were *combo-specific*. For a hand like Ks3s (the nut
flush on this board), the actual per-combo EV is presumably even higher than
the range mean (the range includes weaker combos that bet for protection /
bluff-balance). The direction of the misframe is benign here, but the
attribution is still wrong.

This is a **field-naming bug feeding an LLM framing bug**. The fix is two-
sided: rename the field to something like `range_mean_evs_per_action`, and
either provide a real per-combo EV map or instruct the LLM not to attribute
the aggregate to a specific combo.

---

## (c) Summary verdict

### Trust chain — what's intact

- **Strategic decision (which action is correct):** Python-derived from Pio's
  max-frequency action; passed to the LLM verbatim as a constraint; verified
  matching in 5/5 spot-checks.
- **EV gap math:** bit-exact match between CSV and Pio raw numbers in 5/5
  spot-checks. The `(best - second) / big_blind` convention is correctly
  applied at scale.
- **Option style choice (sizing / frequency / binary_action):** decided in
  Python from solver signals; no LLM input.

### Trust chain — what leaks LLM judgement

1. **Which two actions to template (frequency style).** The LLM picks. Row 1
   shows this can go wrong: the LLM offered a call-vs-raise dichotomy for a
   call-vs-fold strategy. The data block contained the right facts; the LLM
   misread them.
2. **"Always" vs "Mostly" label.** Pure LLM judgement, no Python threshold.
   No deterministic mapping from Pio's freq to which template slot is correct.
3. **`correct_answer` option-string mapping.** Python provides the verb; the
   LLM picks which of its four option strings to label as `correct_answer`.
   Layer 6's validator only checks `correct_answer ∈ options`, not that the
   verb embedded in `correct_answer` matches `decision.correct_action`.
4. **Option set is not validated against Pio's available actions.** An LLM
   that invents an option (or omits a real one — Row 1) passes validation
   silently.
5. **`hero_combo_evs` is range-mean, not per-combo, despite the name.** Row 18
   shows the LLM honoured the misleading name and attributed range-mean
   numbers to a specific combo.

### Verdict

**Strategic correctness of the *decision*: solid.** The LLM never picks which
action is right — Python does, from Pio's frequency, and the LLM is told what
to justify. All 5 spot-checks confirm.

**Strategic correctness of the *frame around* the decision: fragile.** Of 5
rows, 1 has a fundamentally wrong dichotomy (Row 1: call vs raise instead of
call vs fold), 1 has a real-numbers-but-misattributed citation (Row 18: range
mean labelled as combo-specific), and 1 has a cosmetic board mis-description
(Row 6: "two-tone" on a three-suit turn). The remaining 2 (rows 8, 14) are
clean.

**Is the v3 output safe to send to the team?** Mostly yes for the *decision*;
**no for the *teaching*** without manual review of every option set and every
cited number. The 1/5 rate of seriously-misframing options matches the brief's
expectation that "30–50% [will be] rejected on first pass" by Layer 7 — but
Layer 7 doesn't exist yet, so today these reach the reviewer un-flagged.

### Items for follow-up commits (NOT done in this audit)

The audit does not propose code changes — it surfaces them so they can be
addressed deliberately. In priority order:

1. **Layer 7 option-set checker.** Verify that every `option_N` string's
   embedded verb appears in `decision_data.range_aggregate_strategy` with
   non-trivial frequency (say ≥ 5%). Would have caught Row 1.
2. **Layer 7 correct_answer verb-match check.** Confirm the verb embedded in
   `correct_answer` matches `decision.correct_action`. Tighter than the
   current `correct_answer ∈ options` check.
3. **Deterministic "Always" vs "Mostly" threshold in Python.** Compute the
   correct label slot in Python from the Pio freq (e.g. ≥ 90% = "Always",
   55-90% = "Mostly") and pass it to the LLM as a constraint instead of a
   template choice.
4. **Rename `DecisionData.hero_combo_evs` → `range_mean_evs_per_action`** and
   either provide an actual per-combo EV field (from
   `pio.calc_ev[hero_hand_index]`) or update the LLM prompt to prohibit
   per-combo attribution of range-mean values. Would have caught Row 18.
5. **Question 1.5 nit:** add a comment to `_build_decision_data` line 83
   noting the freq/EV equivalence assumption (equilibrium-only) and the
   docstring should say "max-frequency" not "best/second-best."

---

**Audit performed:** 2026-05-21. **Solve:** `btn_vs_bb_srp_2cJs7s.cfr`.
**Pipeline state at audit:** master @ `5d86ea5`. **CSV under audit:**
`test_output/batch_30_questions_v3.csv` (20 rows). **Audit raw data:**
`test_output/audit_spotcheck.json`.
