"""Shared artifact-strip math (July 2026, team standing rule).

Deep-stack solver trees offer "bet/jam your whole stack" branches because the
TREE includes them, not because anyone plays a many-times-pot jam. The line
and answer gates already refuse to BUILD on such actions; this leaf covers
the remaining surface -- the action's presence in a question's OWN strategy
mix and option menu. Each pipeline supplies its own artifact test (postflop:
:func:`pipeline.postflop.premise.artifact_allin_action_labels`; preflop:
``pack_allins_realistic`` -- a deep pack's ``AllIn`` files); the math here is
the shared verdict:

* artifact mass >= :data:`ARTIFACT_MATERIALITY`: MATERIAL. Mixing is
  EV-parity, so the solver genuinely wants the jam sometimes -- the spot's
  real strategy needs a line we refuse to show, and it must NEVER be asked
  (standalone, seed, final leg, or mid-hand question; narrated only).
* below it: a convergence-sliver TRACE (the same <=3-5% dust the
  pack-improvement passes snap to zero). The artifact labels are STRIPPED
  from the mix (removed outright, even at zero mass, so an option menu built
  from the keys can never show them) and the rest renormalised -- "Always"
  qualifiers then require a literal 100% POST-strip.

Shared leaf (like ``bb_display`` / ``trap_grading``): both pipelines import
it; it imports nothing from either.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

# Lowered from 0.20 on July 14 at the user's push: a 90/10 call/jam mix means
# EV(jam) ~= EV(call) (mixing IS indifference), so even 10% is strategic
# signal, not noise. 5% is the convergence-sliver convention the
# IMPROVED-pack snapping already uses.
ARTIFACT_MATERIALITY = 0.05


def strip_artifact_mass(
    freqs: Mapping[str, float], artifact_labels: Iterable[str]
) -> tuple[dict[str, float], float, bool]:
    """Return ``(stripped_freqs, artifact_mass, material)`` for one spot.

    ``freqs`` is the spot's (already conditional/normalised) action mix;
    ``artifact_labels`` the labels judged artifacts by the caller's realism
    test. Material (mass >= :data:`ARTIFACT_MATERIALITY`) returns the mix
    UNCHANGED -- honest frequencies on a spot that must never be asked.
    Otherwise every artifact label is removed (zero-mass ones too) and the
    remaining mass renormalised to sum ~1.
    """
    labels = set(artifact_labels)
    present = [label for label in labels if label in freqs]
    if not present:
        return dict(freqs), 0.0, False
    mass = float(sum(freqs[label] for label in present))
    if mass >= ARTIFACT_MATERIALITY:
        return dict(freqs), mass, True
    out = {label: f for label, f in freqs.items() if label not in labels}
    remaining = sum(out.values())
    if remaining > 0:
        out = {label: f / remaining for label, f in out.items()}
    return out, mass, False


__all__ = ["ARTIFACT_MATERIALITY", "strip_artifact_mass"]
