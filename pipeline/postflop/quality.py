"""Solve-quality / node-reach gate for postflop (the convergence guard).

The postflop analog of the preflop ``node_is_unconverged`` check. Postflop needs
this MORE than preflop because (a) we ingest third-party ``.db`` solves of
unknown quality and (b) turn/river have 100k+ nodes that get down-sampled, so a
node can be barely reached or never properly trained. Generating a question off
such a node teaches noise.

A node is flagged as low quality when either:

* **Low reach** -- too few hero OR villain combos actually reach it (a
  rarely-reached / down-sample-stranded node), or
* **Uniform-strategy artifact** -- nearly all combos play the *identical mixed*
  strategy. A converged solve gives hand-specific mixes; one mix shared across
  ~every hand is an untrained default (e.g. every hand "bets 50%"). A uniform
  PURE action (every hand bets) is legitimate (a clear c-bet) and is NOT flagged.

Conservative thresholds by design: it should only catch genuine junk, never a
real (if simple) node. Returns a human-readable reason so the batch can log
exactly why a node was skipped. Pure + deterministic.
"""

from __future__ import annotations

from collections import Counter

# A combo "reaches" a node if its reach weight clears this floor.
_MIN_WEIGHT = 0.01
# Minimum combos (hero / villain) that must reach the node.
_MIN_HERO_COMBOS = 6
_MIN_VILLAIN_COMBOS = 6
# Uniform-strategy check: only meaningful with enough combos; flags when one
# identical NON-PURE mix is shared by more than this fraction of combos.
_UNIFORM_MIN_COMBOS = 10
_UNIFORM_MAX_FRAC = 0.90
# A mix counts as "pure" (legit, not a uniform-default artifact) when its top
# action is at least this frequent.
_PURE_FREQ = 0.99


def node_quality_issue(node) -> str | None:
    """Return a reason string if ``node`` looks like junk, else ``None``.

    See the module docstring for the two checks. ``None`` == the node passes.
    """
    hero_live = sum(1 for w in node.hero_range.values() if w >= _MIN_WEIGHT)
    if hero_live < _MIN_HERO_COMBOS:
        return (
            f"only {hero_live} hero combos reach this node "
            "(rarely reached / down-sample-stranded)"
        )
    villain_live = sum(1 for w in node.villain_range.values() if w >= _MIN_WEIGHT)
    if villain_live < _MIN_VILLAIN_COMBOS:
        return f"only {villain_live} villain combos reach this node"

    # Uniform-strategy artifact: signature each combo's mix (rounded) and see if
    # one identical non-pure mix dominates.
    sigs: Counter[tuple] = Counter()
    total = 0
    for mix in node.strategy.values():
        total += 1
        sig = tuple(
            sorted((lbl, round(fr, 2)) for lbl, fr in mix.items() if fr > 0.005)
        )
        sigs[sig] += 1
    if total >= _UNIFORM_MIN_COMBOS:
        sig, count = sigs.most_common(1)[0]
        modal_frac = count / total
        top_freq = max((fr for _, fr in sig), default=1.0)
        is_pure = len(sig) <= 1 or top_freq >= _PURE_FREQ
        if modal_frac > _UNIFORM_MAX_FRAC and not is_pure:
            return (
                f"{modal_frac:.0%} of combos play an identical mixed strategy "
                "(untrained / default node)"
            )
    return None


__all__ = ["node_quality_issue"]
