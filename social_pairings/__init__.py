"""Social MDPs for two pick-and-place robots that share their objects.

    from social_pairings import Config, run_pairing, six_object_scene
    known = Config(partner_object="known")      # each robot knows which object the other wants
    result = run_pairing(six_object_scene(), ("hinder", "cooperation"), weights=(2, 2),
                         cfg=known)
    result["done_at"]        # (None, None): the hinderer keeps its partner's object and
                             # has no goal of its own; the cooperator delivers the
                             # hinderer's object anyway, at 5 (both at level 2)

Nobody is told its partner's social goal, nor which object it wants: each robot infers both,
and its partner's destination, from what its partner does (``Config``'s defaults since
2026-10-08 and 2026-10-09; ``Config(**TOLD)`` tells each robot its partner's goal and object,
as until then).  With every one of the six objects a candidate, the six-object scene needs
about 13 GiB; ``partner_object="known"``, as above, or ``max_objects=3`` make it small.

The belief update is Eqs. 5-6 as the paper prints them, kept as the authors' solver keeps
them (since 2026-10-30): two posteriors about each robot -- its physical goal from its
level-1 hypothesis policies averaged over the social posterior (Eq. 6;
``Config.physical_likelihood="level0"``: from its level-0 policies, as the authors' code),
in two stages by its hands; its social goal from its level-1 hypothesis policies averaged
over the physical posterior (Eq. 5) -- the idle filter over both, Eq. 5's transition term,
and nothing else (``models``; the README, "The belief update of 2026-10-30").

A level-2 cooperator cooperates with a partner it takes to be cooperative, competitive or
exchanging, and minds its own goal against any other (since 2026-10-11,
``Config.cooperate_with``; ``"all"`` is Table I as printed, as in ``TOLD``).  Every robot
with a social goal reasons at level 2 unless a level is asked for, a hinderer included
(since 2026-10-11, ``Config.level_free_level``; 1 in ``TOLD``).  A hinderer has no goal of
its own: its reward is its moves' costs and -w r(g_j) (since 2026-10-14,
``Config.hinder_own_goal``; True, the hinderer of before, in ``TOLD``).

Modules: ``goals`` (Table I, hinder, the readings of competition), ``config`` (every option),
``scenes``, ``grid`` / ``hands`` / ``world`` (the world), ``kernels`` / ``models`` (the
solver), ``rollout``, ``pairing`` (the entry point), ``hindrance`` (did the hinderer
hinder?), ``render`` (the annotated GIFs), ``serialize`` (JSON).
"""
from .config import (AUTOMATIC_HANDS, COOPERATE_WITH, EVERY_PARTNER, HYPOTHESES, PAPER_GOALS,
                     TOLD, Config, SolverFailure)
from .goals import SOCIAL_GOALS
from .pairing import default_levels, levels_used, run_baseline, run_pairing
from .scenes import Scene, exchange_scene, six_object_scene, two_room_scene

__all__ = ["Config", "SolverFailure", "TOLD", "AUTOMATIC_HANDS", "PAPER_GOALS", "HYPOTHESES",
           "SOCIAL_GOALS", "COOPERATE_WITH", "EVERY_PARTNER",
           "Scene", "run_pairing", "run_baseline", "default_levels", "levels_used",
           "six_object_scene", "two_room_scene", "exchange_scene"]
