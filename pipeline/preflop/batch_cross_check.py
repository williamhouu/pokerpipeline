"""Deterministic post-batch fact cross-checks (July 2026).

Runs automatically at the end of every preflop batch, on the ARTIFACTS as
written (the CSV rows + meta question records), and verifies them against
poker facts derived from FIRST PRINCIPLES -- deliberately NOT by calling
the same pipeline code that produced them, so an internally-consistent bug
still fails. This is the machine replacement for the LLM "sanity audit"'s
fact categories: when both were calibrated on a real batch, these checks
had zero false positives while the LLM pass had zero true ones.

Checks per row:
  1. ``Relative Position`` recomputed from the seats alone, using a seat
     order DUPLICATED here on purpose (do not "deduplicate" it into
     pipeline.preflop.position -- independence is the point; the July 2026
     blind-vs-blind bug lived in that module and every consumer agreed
     with it).
  2. In/Out of Position Play skills consistent with the seats.
  3. Blind-vs-blind skill hygiene (Blind vs. Blind Play present, Blind
     Defense absent).
  4. Domination lists: direction of every entry re-derived (guards the
     bucket wiring and serialization; the classifier itself is guarded by
     the ground-truth test tier), and an EMPTY dominated_by challenged
     against villain's most-common hands.
  5. Difficulty Rating inside the batch's requested band.
  6. ``action_frequencies`` summing to ~100%.
  7. No Reverse Implied Odds skill on an all-in spot (no postflop play).
  8. Hero-subject position claims in the prose vs the seat-derived truth.

Pure functions; no I/O. The batch driver attaches findings to the meta
question records (``cross_check_issues``) + a ``cross_check_problems``
counter, and the Review page renders them under a distinct badge. A thin
CLI for re-checking any existing batch lives in
``scripts/cross_check_preflop_batch.py``.
"""

from __future__ import annotations

import re

from pipeline.preflop.domination import classify_matchup

# Ring-table postflop action order, from first principles (see module
# docstring for why this is NOT imported from pipeline.preflop.position).
_POSTFLOP_ORDER = ["SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "CO", "BTN"]

# Hero-subject position phrases and the standing they claim.
_PROSE_POSITION_CLAIMS: tuple[tuple[str, str], ...] = (
    ("you're in position", "In Position"),
    ("you are in position", "In Position"),
    ("you'll be in position", "In Position"),
    ("you're out of position", "Out of Position"),
    ("you are out of position", "Out of Position"),
    ("you'll be out of position", "Out of Position"),
)


def expected_relative_position(hero: str, villain: str | None) -> str | None:
    """Hero's IP/OOP standing from the seats alone (ring table).

    None when the hero seat is unknown. With no villain (an open spot),
    only the BTN is guaranteed position on everyone behind.
    """
    if hero not in _POSTFLOP_ORDER:
        return None
    if villain is None or villain not in _POSTFLOP_ORDER:
        return "In Position" if hero == "BTN" else "Out of Position"
    return (
        "In Position"
        if _POSTFLOP_ORDER.index(hero) > _POSTFLOP_ORDER.index(villain)
        else "Out of Position"
    )


def cross_check_row(
    row: dict,
    question_record: dict,
    *,
    min_difficulty: int = 400,
    max_difficulty: int = 3200,
) -> list[str]:
    """All findings for one written CSV row + its meta question record."""
    issues: list[str] = []
    matchup = (row.get("Position Matchup") or "").split("_vs_")
    hero = matchup[0] if matchup and matchup[0] else ""
    villain = matchup[1] if len(matchup) > 1 else None
    skills = [s.strip() for s in (row.get("skills") or "").split(",") if s.strip()]
    sd = question_record.get("solver_data") or {}
    hand = question_record.get("hand_class") or sd.get("hand_class") or ""

    # 1. Relative Position vs seats.
    expected = expected_relative_position(hero, villain)
    got = row.get("Relative Position") or ""
    if expected and got and got != expected:
        issues.append(
            f"Relative Position is {got!r} but the seats ({hero} vs "
            f"{villain or 'open'}) imply {expected!r}"
        )

    # 2. Position skills vs seats.
    if expected == "In Position" and "Out of Position Play" in skills:
        issues.append(
            f"skills tag 'Out of Position Play' but {hero} has position "
            f"on {villain}"
        )
    if expected == "Out of Position" and "In Position Play" in skills:
        issues.append(
            f"skills tag 'In Position Play' but {hero} is out of position "
            f"vs {villain or 'the field'}"
        )

    # 3. BvB skill hygiene -- only when the pot is ACTUALLY blind-vs-blind:
    # folded TO the blinds, no non-blind seat ever put chips in. The MTT
    # packs' limpy multiway lines produce SB-vs-BB *matchups* in pots that
    # merely COLLAPSED to the blinds (e.g. a BB squeeze folding out the
    # field) -- those are squeeze/defense spots, not "Blind vs. Blind Play"
    # (Aug 2026). Participants are read from the prior_action bet-level
    # narration (a non-blind seat with any verb other than "folds" was in
    # the pot) plus the live-hands list.
    others = sd.get("other_players_still_in_hand") or []
    non_blind_live = any(
        str(p).split(" ")[0] not in ("SB", "BB") for p in others
    )
    non_blind_participant = False
    for clause in str(sd.get("prior_action") or "").split(","):
        words = clause.strip().split()
        if len(words) >= 2 and words[0] not in ("SB", "BB") and words[1] != "folds":
            non_blind_participant = True
            break
    if (
        {hero, villain or ""} == {"SB", "BB"}
        and not non_blind_live
        and not non_blind_participant
    ):
        if "Blind vs. Blind Play" not in skills:
            issues.append("blind-vs-blind spot missing the 'Blind vs. "
                          "Blind Play' skill")
        if "Blind Defense" in skills:
            issues.append("blind-vs-blind spot wrongly tagged 'Blind "
                          "Defense' (BvB has its own skill)")

    # 4. Domination lists: direction + empty-list challenge.
    dom = sd.get("domination_vs_villain_range") or {}
    if hand:
        for cls in dom.get("dominated_by", []):
            verdict = classify_matchup(hand, cls)
            if verdict != "dominates_you":
                issues.append(
                    f"dominated_by lists {cls}, but {cls} vs {hand} "
                    f"classifies as {verdict}"
                )
        for cls in dom.get("you_dominate", []):
            verdict = classify_matchup(hand, cls)
            if verdict != "you_dominate":
                issues.append(
                    f"you_dominate lists {cls}, but {cls} vs {hand} "
                    f"classifies as {verdict}"
                )
        likely = [
            h.get("hand_class") if isinstance(h, dict) else str(h)
            for h in (sd.get("villain_stats") or {}).get(
                "most_common_combos", []
            )
        ]
        if likely and dom and not dom.get("dominated_by"):
            missed = [
                h for h in likely
                if h and classify_matchup(hand, h) == "dominates_you"
            ]
            if missed:
                issues.append(
                    f"dominated_by is EMPTY but villain's most common "
                    f"hands include dominators of {hand}: {missed}"
                )

    # 5. Difficulty band membership.
    try:
        score = int(row.get("Difficulty Rating") or 0)
        if not (min_difficulty <= score <= max_difficulty):
            issues.append(
                f"Difficulty Rating {score} is outside the batch's "
                f"requested band [{min_difficulty}, {max_difficulty}]"
            )
    except ValueError:
        issues.append("Difficulty Rating is not an integer")

    # 6. Frequencies sum to ~100%.
    freqs = re.findall(r":\s*(\d+)%", row.get("action_frequencies") or "")
    if freqs:
        total = sum(int(f) for f in freqs)
        if not (97 <= total <= 103):  # noqa: PLR2004 - rounding slack
            issues.append(f"action_frequencies sum to {total}%")

    # 7. No RIO skill on an all-in spot.
    question_text = (row.get("Question") or "").lower()
    if "all-in" in question_text and "Reverse Implied Odds" in skills:
        issues.append(
            "'Reverse Implied Odds' tagged on an all-in spot (no postflop "
            "play means no implied odds in either direction)"
        )

    # 8. Hero-subject prose position claims vs the seats.
    prose = (row.get("Answer Explanation") or "").lower()
    for phrase, claim in _PROSE_POSITION_CLAIMS:
        if phrase in prose and expected and claim != expected:
            issues.append(
                f"prose says {phrase!r} but the seats imply {expected}"
            )

    # 9. GTO options on a literal-pure row must pair the correct action
    #    with the SECOND-BEST action BY EV (standing user rule, July 2026:
    #    the wrong-answer option is the genuinely most tempting mistake;
    #    at a pure spot frequency carries no signal, so EV ranks). This
    #    check exists because the rule was once implemented example-scoped
    #    and quietly missed pure spots -- now every shipped batch is
    #    audited for the rule itself. Tolerance 0.05bb so an EV tie can
    #    legitimately go either way.
    issues.extend(_check_gto_secondary_by_ev(row))

    # 10. Option order = the aggression ladder (standing user rule, July
    #     2026: the option row reads least -> most aggressive; frequency
    #     order once shipped "Fold · 4-bet · Call"). Deliberately its own
    #     ladder map, not an import from the options module -- independence
    #     is this file's point.
    issues.extend(_check_option_aggression_order(row))

    return issues


# Independent aggression ladder for check 10 (mirrors, but does not import,
# the options modules' _ACTION_AGGRESSION). "Always X" / "Mostly X" GTO
# labels rank by their action word.
_AGGRESSION_LADDER: dict[str, int] = {
    "fold": 0, "check": 1, "call": 2, "raise": 3,
    "3-bet": 4, "4-bet": 5, "5-bet": 6, "all-in": 7,
}


def _option_aggression(label: str) -> int | None:
    words = label.strip().lower().split()
    if not words:
        return None
    if words[0] in ("always", "mostly") and len(words) > 1:
        words = words[1:]
    return _AGGRESSION_LADDER.get(words[0].strip(".,;:!?\"'()[]"))


def _check_option_aggression_order(row: dict) -> list[str]:
    """Options in the written row must be non-decreasing in aggression."""
    labels = [
        (row.get(f"option {i}") or "").strip()
        for i in (1, 2, 3, 4)
    ]
    ranks = [
        (lbl, _option_aggression(lbl)) for lbl in labels if lbl
    ]
    known = [(lbl, r) for lbl, r in ranks if r is not None]
    for (a_lbl, a), (b_lbl, b) in zip(known, known[1:], strict=False):
        if a > b:
            return [
                f"options out of aggression order: {a_lbl!r} before {b_lbl!r} "
                "(the row must read least -> most aggressive)"
            ]
    return []


def _parse_labeled_floats(cell: str) -> dict[str, float]:
    """``"Call: +2.87, Fold: +0.00"`` -> {"Call": 2.87, "Fold": 0.0}."""
    out: dict[str, float] = {}
    for part in (cell or "").split(","):
        label, _, value = part.partition(":")
        try:
            out[label.strip()] = float(value.strip().rstrip("%"))
        except ValueError:
            continue
    return out


# Mirrors the artifact all-in rule (pack_allins_realistic): at stacks deeper
# than this, an all-in is a tree artifact that never surfaces as an option, so
# the EV-secondary audit must not demand it as the wrong-answer either (July
# 23 2026: 200bb pack legs false-flagged because the stripped All-in's pack
# EV beat Call's). Rows whose stack can't be parsed (dollar-denominated cash)
# are treated as deep -- every such pack in the pipeline is 100bb+.
_ALLIN_REALISTIC_MAX_BB = 40.0


def _row_stack_bb(row: dict) -> float | None:
    m = re.search(r"([\d.]+)\s*BB\b", row.get("Default Stack") or "", re.I)
    return float(m.group(1)) if m else None


def _check_gto_secondary_by_ev(row: dict) -> list[str]:
    """Check 9 (see cross_check_row): the GTO secondary is EV-ranked."""
    options = [row.get(f"option {i}") or "" for i in (1, 2, 3, 4)]
    filled = [o for o in options if o]
    correct = row.get("Correct Answer") or ""
    is_gto = (
        correct.startswith(("Always ", "Mostly "))
        and filled
        and all(o.startswith(("Always ", "Mostly ")) for o in filled)
    )
    if not is_gto:
        return []
    correct_verb = correct.split(" ", 1)[1]
    alt_verbs = {o.split(" ", 1)[1] for o in filled} - {correct_verb}
    if len(alt_verbs) != 1:
        return []  # not the 2-action spectrum shape
    secondary = next(iter(alt_verbs))
    freqs = _parse_labeled_floats(row.get("action_frequencies") or "")
    if freqs.get(correct_verb) != 100.0:  # noqa: PLR2004
        return []  # only literal-pure spots force the EV tie-break
    evs = _parse_labeled_floats(row.get("action_ev_bb") or "")
    candidates = {a: v for a, v in evs.items() if a != correct_verb}
    stack = _row_stack_bb(row)
    if stack is None or stack > _ALLIN_REALISTIC_MAX_BB:
        # Deep stacks: the artifact rule strips All-in from every option
        # surface, so it cannot be the required EV secondary.
        candidates = {
            a: v for a, v in candidates.items()
            if a.lower().replace("-", "").replace(" ", "") != "allin"
        }
    if not candidates or secondary not in candidates:
        return []  # EV-less pack, or the secondary's EV isn't published
    best = max(candidates, key=lambda a: candidates[a])
    if candidates[best] - candidates[secondary] > 0.05:  # noqa: PLR2004
        return [
            f"GTO secondary is {secondary} ({candidates[secondary]:+.2f}bb) "
            f"but {best} is the better alternative "
            f"({candidates[best]:+.2f}bb) -- the wrong-answer option must "
            f"be the second-best action by EV"
        ]
    return []


def cross_check_batch(
    csv_rows: list[dict],
    question_records: list[dict],
    *,
    min_difficulty: int = 400,
    max_difficulty: int = 3200,
) -> dict[int, list[str]]:
    """Cross-check every row; returns {row_index: findings} for rows with
    findings (empty dict = fully clean). Rows and records must be in the
    same (CSV) order -- the batch driver guarantees this."""
    out: dict[int, list[str]] = {}
    for i, (row, rec) in enumerate(zip(csv_rows, question_records)):
        found = cross_check_row(
            row, rec,
            min_difficulty=min_difficulty, max_difficulty=max_difficulty,
        )
        if found:
            out[i] = found
    return out


__all__ = [
    "cross_check_batch",
    "cross_check_row",
    "expected_relative_position",
]
