"""The public entry point: roll out one pairing of social goals in one scene."""
from __future__ import annotations

from dataclasses import asdict
from typing import Optional, Sequence

import numpy as np

from . import goals as G
from .config import Config
from .rollout import build_world, rollout
from .scenes import Scene

__all__ = ["run_pairing", "run_baseline", "default_levels", "levels_used"]


def _effective(goals: Sequence, levels: Sequence[int],
               weights: Optional[Sequence[float]]) -> list:
    """Each robot's social goal as it counts: none at level 0 or with weight 0."""
    goals = [G.parse_goal(g) for g in goals]
    return [goal if int(levels[i]) >= 1 and (weights is None or weights[i] > 0.0) else G.NONE
            for i, goal in enumerate(goals)]


def default_levels(goals: Sequence, cfg: Optional[Config] = None,
                   weights: Optional[Sequence[float]] = None) -> tuple:
    """The level each robot reasons at when none is asked for (``levels=None``): 2, the
    level of the paper's experiments, for every robot with a social goal (since 2026-10-11,
    the user: "Whenever I say run it, always assume both are level 2") -- except
    ``Config.physical_level`` (0 by default) for a robot with a physical goal only.  A robot
    whose social term is built on goal rewards alone, with no partner reward ``R_j`` in it,
    reasons at ``Config.level_free_level``: 2 by default, 1 until 2026-10-11 (``TOLD``):

    * ``hinder``;
    * ``competition`` in its race forms (``Config.competition="race_then_hinder"`` and
      ``"race"``; the default reading, ``"rival"``, reasons at level 2 either way);
    * ``conflict`` read as a task conflict (``Config.conflict="task"``, the default since
      2026-11-01: ``-w r(g_j)`` at every level; as Table I prints it, ``"table"``, it holds
      ``-R_j`` above level 1 and its default level is 2).

    Why 1 until then: Table I has two columns.  Terms in ``r(g_j)`` are its column l = 1, given to a
    robot that takes its partner to pursue its physical goal and nothing else; at level 1
    a hinderer and a competitor of a race form do what their reward says, and a partner one
    level up models them exactly.  A competitor may be asked for level 2 (``levels=(2, 2)``):
    its term is the same, and it then predicts its partner by the partner's social goal --
    which makes it act on what it expects of its partner before its own goal is done (the
    README, "A competitor at level 2").  A hinderer could not until 2026-10-11
    (:func:`levels_used`); since then it reasons at the level asked, 2 by default.  Under
    the two readings of the table (``"table"``, ``"table_hostile"``) a competitor's term
    holds ``R_j`` above level 1, and its default level is 2.  So is a rival's
    (``"rival"``), whose term holds ``R_j`` against a cooperative partner -- whoever its
    partner is: against a rival its term is a race at either level, and level 2 is then
    the level at which it predicts its partner by the partner's social goal.

    A physical goal only (``"none"``): level 0 by default since 2026-10-08 -- a plain MDP,
    no social stance, nothing inferred; 2 until then (``Config(physical_level=2)``).

    ``weights``: a robot with weight 0 has no social goal whatever goal it was given, and
    defaults to ``Config.physical_level`` as well.  Without ``weights`` every goal counts."""
    cfg = Config() if cfg is None else cfg
    effective = _effective(goals, (1, 1), weights)
    return tuple(cfg.physical_level if goal == G.NONE
                 else cfg.level_free_level if G.level_free(goal, competition=cfg.competition,
                                                           conflict=cfg.conflict)
                 else 2
                 for goal in effective)


def levels_used(goals: Sequence, levels: Sequence[int], cfg: Optional[Config] = None,
                weights: Optional[Sequence[float]] = None) -> tuple:
    """The level each robot actually reasons at when ``levels`` are asked for: those levels
    (since 2026-10-11).  With ``Config.level_free_level=1`` (``TOLD``, the rule until then) a
    hinderer is held to level 1 at most (asked for level 0 it has no social goal, and stays
    at level 0).

    A robot without a social goal still reasons at its level (it predicts its partner); so
    does a robot whose weight is 0, which is a robot without a social goal whatever goal it
    was given.  A competitor reasons at the level asked, under every reading of competition
    (its *default* level under a race form is ``Config.level_free_level``, 2 since
    2026-10-11 and 1 until then: :func:`default_levels`).

    Why a hinderer was held: rolled out at level 2 against a level-2 competitor of the table,
    told its partner's goal, it never took the object (the README, "What hindering rests on",
    line 3).  Since 2026-10-11 it reasons at the level asked, as every other robot does.

    Changed on 2026-10-06: until then a competitor of a race form was held to level 1 as
    well, so ``levels=(2, 2)``, ``(2, 1)`` or ``(1, 2)`` with such a competitor meant level 1
    for it; they now mean what they say.  For the behaviour of before, ask for level 1 in
    the competitor's place (``(1, 1)`` where ``(2, 1)`` was asked with the competitor first);
    leaving ``levels`` out gives what ``(2, 2)`` gave.

    ``weights``: a robot with weight 0 (or at level 0) has no social goal and is not held
    to level 1.  Without ``weights`` every goal counts.  ``cfg``: ``Config.level_free_level``
    decides whether a hinderer is held (1) or not (2, the default)."""
    cfg = Config() if cfg is None else cfg
    levels = [int(level) for level in levels]
    effective = _effective(goals, levels, weights)
    held = cfg.level_free_level == 1
    return tuple(min(level, 1) if goal == G.HINDER and held else level
                 for goal, level in zip(effective, levels))


def run_pairing(scene: Scene, goals: Sequence, *, weights: Sequence[float] = (1.0, 1.0),
                levels: Optional[Sequence[int]] = None, cfg: Optional[Config] = None,
                n_steps: int = 60) -> dict:
    """Roll out two robots with the given social goals.

    scene : the :class:`Scene` -- with hand-off cells and a room per robot, if it has them
        (``Scene.handoff_cells``, ``Scene.rooms``; :func:`social_pairings.exchange_scene`).
    goals : one social goal per robot -- "cooperation", "conflict", "competition",
        "coercion", "exchange", "hinder", or None / "none" (see :mod:`social_pairings.goals`).
    weights : each robot's weight on its social term; 1 is the paper's Table I.  ``hinder``
        and a competitor that races are not recursive, so their weight can be as large as
        you like; for the recursive goals a level-2 robot's term holds its partner's, and
        the weights multiply.
    levels : each robot's reasoning level (0, 1 or 2).  None: :func:`default_levels` -- 2
        for every robot with a social goal, a hinderer and a competitor of a race form
        included (since 2026-10-11; 1 for those two until then, and under ``TOLD``), and
        ``Config.physical_level`` (0) for a physical goal only.  Levels asked for are the
        levels used (:func:`levels_used`; with ``Config.level_free_level=1`` a hinderer
        stays at level 1 at most, as until 2026-10-11).  (Until 2026-10-06 a competitor of
        a race form was held to level 1 whatever was asked.  To get it back, ask for level 1
        in its place.)
    cfg : the :class:`Config` (defaults: greedy actions, hand-overs by consent, the
        partner's destination and social goal inferred, competition read as a rival;
        ``Config(**TOLD)``: each robot told its partner's social goal, as until 2026-10-08).
    n_steps : the most transitions to roll out; the rollout stops earlier once both goals
        are done, or on a deadlock.

    Returns a dict (everything JSON-serialisable through :func:`social_pairings.serialize.to_json`):

    ``done_at`` (2,)      transition at which each robot's goal was done, or None (always
                          None for a robot without a goal of its own, ``own_goal``)
    ``own_goal`` (2,)     False for a hinderer without a goal of its own
                          (``Config.hinder_own_goal=False``, the default)
    ``stopped_by``        "done", "deadlock" or "cap"
    ``ledger`` [2]        per robot: counts of moves, waits, hand actions (by kind), pick-ups,
                          hand-overs, ...
    ``events``            (transition, what, robot, object, other robot); what: "picked up",
                          "put back", "put down" (on a hand-off cell), "handed over", "set
                          down", "set down in vain", "handed over in vain" (an object the
                          partner does not take: only where the partner's object is inferred)
    ``trace`` [steps]     per transition: positions, object statuses, Q values (total and
                          physical), the immediate social reward of each action, beliefs
                          (``object_belief``: which object the partner wants, where that is
                          inferred; ``final_object_belief``, the last)
    ``goals``, ``levels`` (asked for, or the defaults), ``levels_used``, ``weights``,
    ``social_terms``, ``config``
    ``solver``            residuals and iteration counts; ``numerically_accepted``
    """
    cfg = Config() if cfg is None else cfg
    if len(goals) != 2 or len(weights) != 2 or (levels is not None and len(levels) != 2):
        raise ValueError("need one social goal, weight and level per robot")
    goal_idx = tuple(G.parse_goal(g) for g in goals)
    weights = tuple(float(x) for x in weights)
    if any(not np.isfinite(x) or x < 0.0 for x in weights):
        raise ValueError(f"weights must be finite and >= 0, got {weights!r}")
    if levels is None:
        levels = default_levels(goal_idx, cfg, weights)
    levels = tuple(int(level) for level in levels)
    if any(level not in (0, 1, 2) for level in levels):
        raise ValueError(f"levels must be 0, 1 or 2, got {levels!r}")
    if isinstance(n_steps, bool) or not isinstance(n_steps, (int, np.integer)) or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    used = levels_used(goal_idx, levels, cfg, weights)
    world, goal_ids, solo_steps, scenery = build_world(
        scene.grid, scene.starts, scene.pickups, scene.destinations, scene.candidate_cells,
        scene.objects, cfg, handoff_cells=scene.handoff_cells, rooms=scene.rooms)
    result = rollout(world, goal_ids, solo_steps, used, goal_idx, weights, cfg, int(n_steps),
                     scenery)
    effective = tuple(G.parse_goal(name) for name in result["social_goals"])
    result.update(
        goals=tuple(G.goal_name(g) for g in goal_idx),
        levels=levels, levels_used=used,
        social_terms=tuple(G.describe(effective[i], used[i],
                                      effective[1 - i] if cfg.partner_social_goal == "known"
                                      else None, competition=cfg.competition,
                                      exchange_level1=cfg.exchange_level1,
                                      cooperate_with=G.cooperation_partners(cfg.cooperate_with),
                                      hinder_own_goal=cfg.hinder_own_goal,
                                      conflict=cfg.conflict)
                           for i in (0, 1)),
        config=asdict(cfg))
    result["scene"].update(name=scene.name, grid=np.isfinite(scene.grid).astype(int),
                           pickups=scene.pickups, resolution=cfg.resolution)
    return result


def run_baseline(scene: Scene, *, cfg: Optional[Config] = None, n_steps: int = 60) -> dict:
    """The same scene with no social goals (two level-0 robots): the yardstick against
    which hindering and helping are measured."""
    return run_pairing(scene, (None, None), levels=(0, 0), cfg=cfg, n_steps=n_steps)
