"""The solver against brute force: every model of levels 0, 1 and 2 on a 3 x 3 map.

For each pairing the engine's models are solved once (to 1e-7) with arbitrary sharp beliefs,
and every array the decisions rest on is compared with ``reference.Brute``, which writes the
same models out state by state from the documented rules: the partner's action distribution
(Eq. 4), the expected rewards (Eq. 2 with the social term of Table I / hinder / the race
forms of competition / a rival), the values and Q (Eq. 3) and the partner policies (Eq. 7).

The slower comparisons run only with SOCIAL_PAIRINGS_SLOW=1.
"""
import os
import unittest

import numpy as np

from social_pairings import HYPOTHESES
from social_pairings import AUTOMATIC_HANDS
from social_pairings import Config as PackageConfig
from social_pairings import goals as G
from social_pairings.constants import GONE, HELD0, HELD1, HOME
from social_pairings.models import NO_GOAL_TWIN, NO_TERM, SPOILER_TWIN, SocialMDP
from social_pairings.util import configure_threads

from .reference import (PASS, Brute, level1_term, level2_race, make_world, states_of,
                        table_l2)
from .told import Config

SLOW = os.environ.get("SOCIAL_PAIRINGS_SLOW", "") not in ("", "0")
VERBOSE = os.environ.get("SOCIAL_PAIRINGS_VERBOSE", "") not in ("", "0")

# Largest deviation allowed per level and quantity (R, V and Q times the size of the
# rewards).  The models are solved to 1e-7, but the engine stores rewards and policies in
# single precision: values are then good to ~1e-5 and softmax policies (tau = 0.2) to ~1e-4,
# and a level-2 model inherits its partner distribution from the level-1 policies.
LIMITS = {1: dict(P=2e-5, R=2e-5, V=2e-4, Q=2e-4, psi=2e-4),
          2: dict(P=2e-4, R=2e-4, V=2e-3, Q=2e-3)}

TINY = dict(grid=np.ones((3, 3)), starts=((0, 1), (2, 1)), pickups=((2, 2), (0, 0)),
            destinations=((0, 2), (2, 0)), candidate_cells=((1, 0),))


def tight(**options) -> Config:
    """The paper's constants on a 1 m grid, solved to |V - V*| <= 1e-7: what is left is the
    difference between the models, not the solver's tolerance.  The defaults of before
    2026-10-08 (``told``)."""
    options.setdefault("value_tol", 1e-7)
    return Config(num_threads=2, resolution=1.0, **options)


def tight_now(**options) -> PackageConfig:
    """The same with the package's defaults of now: every partner goal inferred over seven
    social goals, the unweighted copy, a rival, the idle hypothesis (since 2026-10-08 noon;
    the levels are passed explicitly here, so a physical goal only is at the level asked)
    -- but the rule of the hands until 2026-10-18 (``AUTOMATIC_HANDS``; the rule of now is
    ``tight_explicit``, tested in ``test_explicit.py``)."""
    options.setdefault("value_tol", 1e-7)
    return PackageConfig(num_threads=2, resolution=1.0, **{**AUTOMATIC_HANDS, **options})


def tight_explicit(**options) -> PackageConfig:
    """The package's defaults of now, the rule of the hands of 2026-10-18 included: nothing
    automatic, a pick-up of three transitions costing three moves' worth on the first, the
    other hand actions two."""
    options.setdefault("value_tol", 1e-7)
    return PackageConfig(num_threads=2, resolution=1.0, **options)


def hostage(w):
    """Mid-episode: R0 holds R1's object and stands next to R1."""
    return (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HOME, HELD0], [0, 0]))


def crossed(w):
    """Mid-episode: each robot holds the other's object."""
    return (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HELD1, HELD0], [0, 0]))


def one_done(w):
    """Late in a race: R0's goal is done, R1 carries its own object one step from its
    destination."""
    return (int(w.cell_id[(0, 2)]), int(w.cell_id[(2, 1)]), w.block_of([GONE, HELD1], [1, 0]))


def other_done(w):
    """Late in a race, the other way round: R1's goal is done, R0 carries its own object
    one step from its destination."""
    return (int(w.cell_id[(0, 1)]), int(w.cell_id[(2, 0)]), w.block_of([HELD0, GONE], [0, 1]))


LYING = 4                                    # status of an object on the first hand-off cell


def at_the_doorway(w):
    """``PASS``: each robot holds the other's object; R0 stands in the doorway, R1 next to
    it in its own room."""
    return (int(w.cell_id[(2, 0)]), int(w.cell_id[(3, 0)]), w.block_of([HELD1, HELD0], [0, 0]))


def put_down(w):
    """``PASS``: R0 has put R1's object down in the doorway and stepped back; R1 stands
    next to the doorway with R0's object in its hands."""
    return (int(w.cell_id[(1, 0)]), int(w.cell_id[(3, 0)]), w.block_of([HELD1, LYING], [0, 0]))


def both_lying(w):
    """``PASS``: both objects lie in the doorway, R1 stands on them, R0 next to it."""
    return (int(w.cell_id[(1, 0)]), int(w.cell_id[(2, 0)]), w.block_of([LYING, LYING], [0, 0]))


# A candidate destination on the place of R0's object: a robot standing there with the object
# has no way left to go under that hypothesis (D = 0) and has not finished.
ON_A_PLACE = dict(TINY, candidate_cells=((2, 2),))


def solved(scene, cfg, levels, goals, weights, state=None, seed=0):
    """The engine's models of one pairing, solved once at ``state`` with random sharp beliefs
    (and idle probabilities) in every slot."""
    w, goal_ids = make_world(scene, cfg)
    mdp = SocialMDP(w, goal_ids, levels, tuple(G.parse_goal(g) for g in goals), weights, cfg,
                    configure_threads(cfg.num_threads))
    mdp.allocate()
    info = mdp.solve_level0()
    state = w.initial if state is None else state(w)
    rng = np.random.default_rng(seed)
    for j in (0, 1):                 # random sharp posteriors about each robot (and idle)
        b = mdp.about[j]
        b.log_phys = np.log(rng.dirichlet(np.full(b.log_phys.size, 0.7)))
        if b.log_social is not None:
            b.log_social = np.log(rng.dirichlet(np.full(b.log_social.size, 0.7)))
        b.idle = float(rng.uniform(0.0, 0.5)) if cfg.idle_hypothesis else 0.0
    mdp.plan(state)
    mdp.init_models()
    for depth in (1, 2):
        mdp.inputs(depth)
        mdp.solve(depth)
    return w, goal_ids, mdp, state, info


class ModelsAgainstBruteForce(unittest.TestCase):
    def compare(self, cfg, levels, goals, weights=(1.0, 1.0), state=None, seed=0, scene=TINY):
        w, gid, mdp, state, info = solved(scene, cfg, levels, goals, weights, state, seed)
        bf = Brute(w, cfg)
        # physical goals: an object and a destination (with the partner's object known,
        # a robot's own object to each destination)
        K, L0, L1, L2 = mdp.PG, mdp.L0, mdp.L1, mdp.L2
        goals_of = [[(int(mdp.pg_obj[i, g]), int(mdp.pg_dst[i, g])) for g in range(K)]
                    for i in (0, 1)]
        names = tuple(G.goal_name(g) for g in mdp.goals)
        exchange = cfg.exchange_share if cfg.exchange_level1 != "none" else 0.0
        unweighted = cfg.exchange_level1 == "copy_unweighted"
        spoiler = not getattr(cfg, "hinder_own_goal", True)   # a hinderer without its own goal
        sigma = 1.0 if cfg.social_cost else 0.0
        # the size of the rewards: a level-1 term holds one weight, a level-2 term both --
        # but for an exchanger's, whose partner's copy is not multiplied by a weight
        scale = {1: max(1.0, max(weights)), 2: max(1.0, max(weights), weights[0] * weights[1])}
        if unweighted and all(g in ("exchange", "none") for g in goals):
            scale[2] = scale[1]
        self.assertTrue(info["ok"] and L1.ok[L1.active].all() and L2.ok[L2.active].all(),
                        "a solve missed its Bellman-residual target")

        def hypotheses_about(robot: int) -> tuple:
            """The social goals the partner of ``robot`` holds possible for it."""
            if cfg.partner_social_goal == "known":
                return (names[robot],)
            return tuple(cfg.social_hypotheses)

        def term1(robot: int, goal: str) -> tuple:
            """The level-1 term of ``robot`` with social goal ``goal``.  With known partner
            goals it is told its partner's (which only a rival asks for)."""
            partner = names[1 - robot] if cfg.partner_social_goal == "known" else None
            return level1_term(goal, weights[robot], exchange, cfg.competition, partner,
                               unweighted, spoiler)

        def worst(a, b, where=None) -> float:
            d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
            return float((d if where is None else d[where]).max())

        def worst_q(depth, row, robot, Q, states, s0, seed_) -> float:
            rng = np.random.default_rng(seed_)
            out = 0.0
            picks = rng.choice(states, size=min(200, states.size), replace=False).tolist()
            for s in [s0] + picks:
                q = mdp.q_values((depth, row), int(s), robot)
                self.assertTrue((np.isfinite(q) == np.isfinite(Q[s])).all(),
                                "the legal actions differ")
                fin = np.isfinite(q)
                out = max(out, worst(q[fin], Q[s][fin]))
            return out

        # ---- level 0: 2 K plain MDPs ----------------------------------------------------
        psis = {}
        every = states_of(w, mdp.reachable)
        for o in (0, 1):
            own = states_of(w, mdp.level0_blocks[o])
            for k in range(K):
                V, _, psi = bf.level0(o, goals_of[o][k][1], goals_of[o][k][0])
                psis[(o, k)] = psi
                self.assertLess(worst(L0.V[o * K + k][own], V[own]), 2e-5)
                self.assertLess(worst(L0.psi[o * K + k][every], psi[every]), 2e-5)

        # ---- level 1: every slot (robot, destination), every model in it ---------------
        found = {1: dict(P=0.0, R=0.0, V=0.0, Q=0.0, psi=0.0), 2: dict(P=0.0, R=0.0, V=0.0, Q=0.0)}
        s0 = w.sid(state)
        hyp_psi = {}
        n1 = 0
        for (a, depth, k), slot in mdp.slots.items():
            if depth != 1:
                continue
            expected = set()             # the level-1 terms this slot must hold, by Table I
            if levels[1 - a] == 2:
                expected |= {term1(a, t) for t in hypotheses_about(a)}
            if levels[a] == 1 and k == mdp.own_goal[a]:
                twin = SPOILER_TWIN if (spoiler and names[a] == "hinder") else NO_TERM
                expected |= {term1(a, names[a])} | ({twin} if cfg.physical_twin else set())
            self.assertEqual(set(slot.model_keys), expected)
            belief = mdp.planning_weights(slot)
            idle = np.full(K, mdp.about[1 - a].idle)
            for row, key in zip(slot.model_rows.tolist(), slot.model_keys):
                extra = {}
                V, Q, P, R, legal, psi = bf.level1(
                    a, goals_of[a][k][1], key, belief, idle,
                    [psis[(1 - a, kk)] for kk in range(K)], False,
                    goals_of[a][k][0], goals_of[1 - a], extra=extra)
                hyp_psi[(a, k, key)] = psi
                n1 += 1
                if mdp.root[a] == (1, row):      # the robot's own model: its value by term
                    self.check_terms(mdp, a, bf, extra, scale[1], 1)
                states = states_of(w, mdp.reach_of(a, k, state[2]))
                seen = found[1]
                seen["P"] = max(seen["P"], worst(L1.P[slot.p_row][states], P[states]))
                seen["R"] = max(seen["R"], worst(L1.R[row][states], R[states], legal[states]))
                solved_states = states_of(w, mdp.solved_blocks(L1, row))
                if solved_states.size:
                    seen["V"] = max(seen["V"], worst(L1.V[row][solved_states], V[solved_states]))
                if L1.psi_row[row] >= 0:
                    seen["psi"] = max(seen["psi"],
                                      worst(L1.psi[L1.psi_row[row]][states], psi[states]))
                seen["Q"] = max(seen["Q"], worst_q(1, row, a, Q, states, s0, row))

        # ---- level 2: the root and its physical twin -----------------------------------
        n2 = 0
        for (a, depth, g), slot in mdp.slots.items():
            if depth != 2:
                continue
            p = 1 - a
            hyp_names = hypotheses_about(p)
            self.assertEqual(tuple(G.goal_name(t) for t in mdp.hyp_goals[a]), hyp_names)
            # Eq. 4 as printed: P(g_j) P(xi_ji), the product of the two posteriors
            phys, social = mdp.about[p].phys(), mdp.about[p].social()
            belief = np.outer(phys, social).reshape(-1)
            self.assertLess(float(np.abs(belief - mdp.planning_weights(slot)).max()), 1e-12)
            hyps = []
            for h, (k, t) in enumerate(zip(slot.hyp_goal.tolist(), slot.hyp_type.tolist())):
                coeff = term1(p, hyp_names[t])
                hyps.append((k, t, belief[h], mdp.about[p].idle, hyp_psi[(p, k, coeff)], coeff))
            # the partner's (every level-1 model's) belief about robot a: about[a]
            bel_me = [mdp.about[a].phys() for k in range(K)]
            idle_me = [np.full(K, mdp.about[a].idle) for k in range(K)]
            for row, goal in zip(slot.model_rows.tolist(), slot.model_keys):
                mine = "no goal" if goal == NO_GOAL_TWIN else G.goal_name(goal)

                def race(t, mine=mine):          # its race term against social goal t
                    return level2_race(mine, weights[a], cfg.competition, hyp_names[t])

                def term(t, same, mine=mine, race=race):
                    if race(t) != 0.0:           # a racer has nothing of the column l > 1
                        return (0.0, 0.0, 0.0)
                    return table_l2(mine, hyp_names[t], same, cfg.exchange_share, weights[a],
                                    cfg.competition, cfg.cooperate_with, spoiler)

                physical = all(
                    race(t) == 0.0 and term(t, same) == (0.0, 0.0, 0.0)
                    for t in range(len(hyp_names)) for same in (False, True))
                # the twin of a robot without a goal of its own has no term on its partner
                social = not physical and mine != "no goal"
                extra = {}
                V, Q, P, R, legal = bf.level2(
                    a, goals_of[a][g][1], term, hyps, bel_me, idle_me,
                    [psis[(a, kk)] for kk in range(K)], sigma, False, physical, race,
                    goals_of[a][g][0], goals_of[p], goals_of[a], g, social=social,
                    extra=extra, condition=cfg.child_conditioning)
                n2 += 1
                if mdp.root[a] == (2, row):      # the robot's own model: its value by term
                    self.check_terms(mdp, a, bf, extra, scale[2], 2)
                states = states_of(w, mdp.reach_of(a, g, state[2]))
                seen = found[2]
                seen["P"] = max(seen["P"], worst(L2.P[slot.p_row][states], P[states]))
                seen["R"] = max(seen["R"], worst(L2.R[row][states], R[states], legal[states]))
                solved_states = states_of(w, mdp.solved_blocks(L2, row))
                if solved_states.size:
                    seen["V"] = max(seen["V"], worst(L2.V[row][solved_states], V[solved_states]))
                seen["Q"] = max(seen["Q"], worst_q(2, row, a, Q, states, s0, 1000 + row))

        if VERBOSE:
            print(f"\n    {goals} levels {levels} weights {weights}: {n1} level-1 and {n2} "
                  f"level-2 models; largest deviations from brute force: "
                  + "; ".join(f"level {lv} " + ", ".join(f"{k} {v:.1e}" for k, v in d.items())
                              for lv, d in found.items() if (n1, n2)[lv - 1]))
        for level, limits in LIMITS.items():
            for name, limit in limits.items():
                size = scale[level] if name in ("R", "V", "Q") else 1.0
                self.assertLess(found[level][name], limit * size,
                                f"level {level}: {name} differs from brute force")
        return n1, n2

    def check_terms(self, mdp, a: int, bf, extra: dict, size: float, level: int) -> None:
        """``SocialMDP.value_components`` of robot a's root against the reference: under the
        engine's greedy policy, the discounted sum of each term of the reward apart
        (``Brute.evaluate`` on the reference's own rewards and transitions) is the engine's,
        the terms add up to the engine's V, and the terms the engine leaves out pay nothing
        (or, a hinderer without a goal of its own, cancel).

        Each term is held to the level's V limit: the engine's and the reference's terms are
        both exact solves under the same policy, but on rewards and transitions that may differ
        by the level's P and R limits (a level-2 model's come from the partner's level-1
        policies, solved iteratively), as V does. That the terms add up to the engine's own V
        is checked apart, at 1e-3 of the scale."""
        got = mdp.value_components(a)
        states = np.flatnonzero(got["policy"] >= 0)
        self.assertGreater(states.size, 0)
        want = bf.evaluate(got["policy"], extra["parts"], extra["trip"], extra["const"], states)
        names = got["names"]
        for k, name in enumerate(names):
            d = float(np.abs(got["parts"][k][states] - want[name][states]).max())
            self.assertLess(d, LIMITS[level]["V"] * size,
                            f"term {name} of robot {a} differs from brute force")
        left_out = [name for name in want if name not in names]
        gone = sum(want[name][states] for name in left_out) if left_out else np.zeros(states.size)
        self.assertLess(float(np.abs(gone).max()), 1e-6 * size, f"{left_out} pay something")
        total = got["parts"][:, states].sum(axis=0)
        self.assertLess(float(np.abs(total - got["total"][states]).max()), 1e-3 * size)
        self.assertLess(got["residual"], 1e-3 * size)

    # ---- the hinder pairings ------------------------------------------------------------
    def test_hinder_against_a_level_2_competitor(self):
        """The pairing of the bug report: R0 hinders at level 1, R1 competes at level 2 as
        Table I prints it and models R0 as a hinderer for each of its K possible
        destinations.  (By default ``run_pairing`` now has a competitor race a hinderer, at
        level 1.)"""
        n1, n2 = self.compare(tight(competition="table"), (1, 2), ("hinder", "competition"),
                              (2.0, 1.5))
        self.assertEqual((n1, n2), (3 + 1, 2))       # K hypotheses + R0's twin; root + twin

    def test_hinder_holding_the_hostage_against_a_level_2_cooperator(self):
        self.compare(tight(), (1, 2), ("hinder", "cooperation"), (3.0, 2.0), state=hostage,
                     seed=1)

    def test_hinder_against_a_partner_without_social_goal(self):
        self.compare(tight(), (1, 0), ("hinder", "none"), (30.0, 1.0), state=hostage, seed=2)

    # ---- competition as a race -----------------------------------------------------------
    # tight() reads it as "race_then_hinder" (told.py): once its own goal is done a competitor is
    # paid 2 r(g_i) - r(g_j).  "race": r(g_i).
    def test_two_racing_competitors(self):
        """Level 1 against level 1: each loses w rho while only its partner's goal is done,
        and once its own is done it is paid for it and for its lead over its partner."""
        self.assertEqual(tight().competition, "race_then_hinder")
        n1, n2 = self.compare(tight(), (1, 1), ("competition", "competition"), (2.0, 30.0),
                              seed=8)
        self.assertEqual((n1, n2), (4, 0))           # two roots and their twins

    def test_competitors_that_stop_once_their_goal_is_done(self):
        """``competition="race"``: nothing about the partner once the own goal is done.
        Two such robots before and after one goal is done; a level-2 cooperator whose +R_j
        holds such a racer's term."""
        cfg = tight(competition="race")
        self.compare(cfg, (1, 1), ("competition", "competition"), (2.0, 30.0), seed=8)
        self.compare(cfg, (1, 1), ("competition", "competition"), (30.0, 0.8),
                     state=one_done, seed=9)
        self.compare(cfg, (1, 2), ("competition", "cooperation"), (30.0, 1.0),
                     state=crossed, seed=20)
        self.compare(cfg, (2, 1), ("cooperation", "competition"), (1.5, 2.0),
                     state=one_done, seed=21)

    def test_racing_competitors_once_one_goal_is_done(self):
        self.compare(tight(), (1, 1), ("competition", "competition"), (30.0, 0.8),
                     state=one_done, seed=9)

    def test_once_its_goal_is_done_a_competitor_is_a_hinderer(self):
        """``race_then_hinder``: where its own goal is done a competitor is paid
        ``r(g_i) + w (2 r(g_i) - r(g_j))`` with ``r(g_i) = rho``, and a hinderer
        ``r(g_i) - w r(g_j)``: the same but for the constant ``2 w rho``.  So its rewards
        are a hinderer's plus ``2 w rho``, its values and Q a hinderer's plus
        ``2 w rho / (1 - gamma)``, in every state and for every action, and it chooses what
        a hinderer chooses.  Under ``"race"`` it has no term on its partner there, and the
        same comparison fails.

        At level 1, and at level 2: there the competitor's term is the same term, and the
        hinderer it is compared with is a level-2 one (``SocialMDP`` builds it;
        ``run_pairing`` holds a hinderer to level 1)."""
        def model(level, goal, weight, **options):
            cfg = tight(**options)
            w, gid, mdp, state, _ = solved(TINY, cfg, (level, 0), (goal, "none"),
                                           (weight, 1.0), state=one_done, seed=40)
            depth, row = mdp.root[0]
            self.assertEqual(depth, level)
            bank = mdp.bank(depth)
            blocks = mdp.reach_of(0, gid[0], state[2])
            self.assertTrue((w.done[blocks, 0] != 0).all())        # its goal stays done
            states = states_of(w, blocks)
            q = np.stack([mdp.q_values((depth, row), int(s), 0) for s in states])   # (S, 10)
            slot = mdp.slots[(0, depth, gid[0])]
            return cfg, states, q, bank.V[row][states], bank.R[row][states], slot

        for level in (1, 2):
            for weight in (1.0, 30.0):
                cfg, states, q_h, v_h, r_h, slot_h = model(level, "hinder", weight)
                _, states_c, q_c, v_c, r_c, slot_c = model(level, "competition", weight)
                np.testing.assert_array_equal(states, states_c)
                self.assertIs(type(slot_h), type(slot_c))
                legal = np.isfinite(q_h)                                       # (S, 10)
                np.testing.assert_array_equal(legal, np.isfinite(q_c))
                self.assertGreater(states.size, 100)
                step = 2.0 * weight * cfg.rho
                ahead = step / (1.0 - cfg.gamma)
                tol = LIMITS[level]["Q"] * max(1.0, weight)
                off = dict(R=float(np.abs(r_c[legal] - r_h[legal] - step).max()),
                           V=float(np.abs(v_c - v_h - ahead).max()),
                           Q=float(np.abs(q_c[legal] - q_h[legal] - ahead).max()))
                self.assertLess(off["R"], LIMITS[level]["R"] * weight)
                self.assertLess(off["V"], tol)
                self.assertLess(off["Q"], tol)
                # ... and so the same choice, wherever a hinderer's is not a toss-up
                best = np.where(legal, q_h, -np.inf)
                order = np.sort(best, axis=1)
                clear = order[:, -1] - order[:, -2] > 4.0 * tol
                self.assertGreater(int(clear.sum()), 50)
                np.testing.assert_array_equal(
                    np.where(legal, q_c, -np.inf).argmax(axis=1)[clear],
                    best.argmax(axis=1)[clear])
                # the hinderer does hinder there: its values depend on where its partner is
                self.assertGreater(np.ptp(v_h), 10.0 * tol)

                v_r = model(level, "competition", weight, competition="race")[3]
                self.assertGreater(np.ptp(v_r - v_h), 10.0 * tol)   # "race": not a hinderer's
                if VERBOSE:
                    print(f"\n    level {level}, weight {weight:g}: {states.size} states with "
                          f"R0's goal done, {int(legal.sum())} legal (state, action); "
                          f"competitor minus hinderer: R = {step:g}, V = Q = {ahead:g}, off by "
                          "at most " + ", ".join(f"{k} {v:.1e}" for k, v in off.items())
                          + f"; same choice in all {int(clear.sum())} states where the "
                          f"hinderer's is clear; under \"race\" V - V_hinder varies by "
                          f"{np.ptp(v_r - v_h):.3g}")

    # ---- a competitor of a race form at level 2 -------------------------------------------
    # Its term is the one of level 1, on goal rewards (under every hypothesis about its
    # partner by that hypothesis' destination); its partner is predicted by the partner's
    # level-1 models, which hold the partner's own social term.
    def test_two_level_2_racers(self):
        n1, n2 = self.compare(tight(), (2, 2), ("competition", "competition"), (2.0, 3.0),
                              seed=50)
        self.assertEqual((n1, n2), (6, 4))     # K hypotheses about each; two roots and twins
        # R0 done and ahead, R1 behind
        self.compare(tight(), (2, 2), ("competition", "competition"), (30.0, 0.8),
                     state=one_done, seed=51)

    def test_a_level_2_racer_and_a_level_2_cooperator(self):
        """The racer's partner hypotheses are level-1 cooperators; the cooperator's +R_j
        holds the term of a level-1 racer."""
        self.compare(tight(), (2, 2), ("competition", "cooperation"), (30.0, 1.0),
                     state=crossed, seed=54)
        self.compare(tight(), (2, 2), ("cooperation", "competition"), (1.5, 2.0),
                     state=one_done, seed=55)    # the cooperator is done, the racer behind

    def test_a_level_2_racer_against_a_hinderer_that_holds_its_object(self):
        self.compare(tight(), (2, 1), ("competition", "hinder"), (2.0, 1.5), state=crossed,
                     seed=57)

    def test_the_race_form_at_level_2(self):
        """``competition="race"``: once its own goal is done, nothing about its partner."""
        self.compare(tight(competition="race"), (2, 2), ("competition", "cooperation"),
                     (30.0, 1.0), state=one_done, seed=60)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_more_level_2_racers(self):
        """The racer behind at level 2; from the start against a cooperator; ahead of a
        cooperator; with its partner's object in its hands against a partner without social
        goal; two of the race form; a partner whose social goal it has to infer; a scene
        in which an object lies on a candidate destination."""
        self.compare(tight(), (2, 2), ("competition", "competition"), (0.8, 30.0),
                     state=other_done, seed=52)
        self.compare(tight(), (2, 2), ("competition", "cooperation"), (2.0, 1.0), seed=53)
        self.compare(tight(), (2, 2), ("cooperation", "competition"), (1.5, 2.0),
                     state=other_done, seed=56)
        self.compare(tight(), (2, 0), ("competition", "none"), (3.0, 1.0), state=hostage,
                     seed=58)
        self.compare(tight(competition="race"), (2, 2), ("competition", "competition"),
                     (2.0, 3.0), seed=59)
        cfg = tight(partner_social_goal="inferred",
                    social_hypotheses=("cooperation", "competition", "none"))
        self.compare(cfg, (2, 2), ("competition", "cooperation"), (2.0, 1.0), state=crossed,
                     seed=61)
        self.compare(tight(), (2, 2), ("competition", "competition"), (2.0, 3.0), seed=62,
                     scene=ON_A_PLACE)

    def test_a_level_2_robot_without_social_goal_models_the_racer(self):
        self.compare(tight(), (2, 1), ("none", "competition"), (1.0, 3.0), seed=10)

    def test_level_2_conflict_against_a_racing_competitor(self):
        """The conflict robot's term is -R_j, and R_j holds the competitor's race term: what
        the competitor takes itself to have won or lost."""
        self.compare(tight(), (2, 1), ("conflict", "competition"), (1.5, 2.0), seed=11)
        self.compare(tight(), (2, 1), ("conflict", "competition"), (1.5, 2.0), state=one_done,
                     seed=12)

    def test_a_goal_is_finished_by_its_object_not_by_a_reward_of_rho(self):
        """Where an object's place is a candidate destination a robot standing on it earns
        rho without having finished anything: the race term must not move."""
        self.compare(tight(), (1, 1), ("competition", "competition"), (2.0, 3.0), seed=17,
                     scene=ON_A_PLACE)
        self.compare(tight(), (2, 1), ("conflict", "competition"), (1.5, 2.0), seed=19,
                     scene=ON_A_PLACE)

    def test_competition_as_table_one_prints_it(self):
        self.compare(tight(competition="table"), (1, 1), ("competition", "competition"),
                     (2.0, 2.0), seed=13)

    def test_a_level_2_cooperator_models_the_racer(self):
        """A competitor races a cooperator too.  The cooperator's term is +R_j, and R_j holds
        the racer's term: the cooperator is paid once its partner has finished, loses for
        as long as it is itself there first -- and, once its partner is done, loses what
        its partner takes it to have reached of its own goal."""
        self.compare(tight(), (1, 2), ("competition", "cooperation"), (2.0, 1.0), seed=14)
        self.compare(tight(), (1, 2), ("competition", "cooperation"), (30.0, 1.0),
                     state=crossed, seed=20)
        self.compare(tight(), (2, 1), ("cooperation", "competition"), (1.5, 2.0),
                     state=one_done, seed=21)      # the cooperator is done, the racer behind

    def test_competition_against_a_cooperator_as_table_one_prints_it(self):
        """Level 2 against level 2: the table's competition against a cooperative partner
        is +R_j, the cell that makes a competitor deliver for its partner."""
        n1, n2 = self.compare(tight(competition="table"), (2, 2),
                              ("competition", "cooperation"), (0.8, 0.8), state=crossed,
                              seed=14)
        self.assertEqual(n2, 4)                      # two level-2 roots and their twins

    # ---- the table with the competitor's cell against a helpful partner turned -----------
    # ``competition="table_hostile"``: -R_j where Table I prints +R_j.  compare() takes the
    # coefficients from reference.TABLE_L2_HOSTILE.
    def test_the_hostile_table_against_a_cooperator(self):
        """Level 2 against level 2, each with the other's object in its hands: the turned
        cell, -R_j against a cooperative partner."""
        n1, n2 = self.compare(tight(competition="table_hostile"), (2, 2),
                              ("competition", "cooperation"), (0.8, 0.8), state=crossed,
                              seed=14)
        self.assertEqual(n2, 4)                      # two level-2 roots and their twins
        self.compare(tight(competition="table_hostile"), (2, 2),
                     ("cooperation", "competition"), (1.5, 2.0), seed=70)

    def test_under_the_hostile_table_a_competitor_is_a_conflict_robot(self):
        """With that cell turned the competition row of Table I is its conflict row, but
        for the cell against a coercing partner with the robot's own destination; the
        column l = 1 is the same entry already.  So against a cooperative, an exchanging
        and a competing partner the pairing is the conflict pairing: every model of either
        robot is the same array -- rewards, values and Q, bit for bit.  Against a coercing
        partner it is not.  Under the table as printed the competitor's models differ from
        a conflict robot's against the two partners the table has it help, and its
        partner's do not (a partner models it at level 1, where nothing was turned).

        All of this is with known partner goals.  With an inferred one the competitor's
        term mixes the cells of every hypothesis, the untouched coercion cell among them:
        its models are then not a conflict robot's, and are again once coercion is not
        among the hypotheses."""
        def models(goals, weights, **options):
            cfg = tight(**options)
            w, gid, mdp, state, _ = solved(TINY, cfg, (2, 2), goals, weights, state=crossed,
                                           seed=14)
            out = []
            for robot in (0, 1):
                depth, row = mdp.root[robot]
                self.assertEqual(depth, 2)
                bank = mdp.bank(depth)
                states = states_of(w, mdp.reach_of(robot, gid[robot], state[2]))
                q = np.stack([mdp.q_values((depth, row), int(s), robot) for s in states])
                out.append((bank.R[row][states].copy(), bank.V[row][states].copy(), q))
            return out                               # per robot: R (S, 10), V (S,), Q (S, 10)

        def same(a, b) -> bool:
            return all(np.array_equal(x, y, equal_nan=True) for x, y in zip(a, b))

        for weights in ((0.8, 0.8), (30.0, 1.0)):
            for partner in ("cooperation", "exchange", "competition", "coercion"):
                for seat in (0, 1):
                    with self.subTest(weights=weights, partner=partner, seat=seat):
                        other = "conflict" if partner == "competition" else partner
                        goals, twin = ("competition", partner), ("conflict", other)
                        if seat == 1:
                            goals, twin = goals[::-1], twin[::-1]
                        hostile = models(goals, weights, competition="table_hostile")
                        conflict = models(twin, weights)
                        printed = models(goals, weights, competition="table")
                        self.assertGreater(hostile[0][1].size, 500)       # states compared
                        # the partner's models: the same under the table as printed
                        self.assertTrue(same(hostile[1 - seat], printed[1 - seat]))
                        if partner == "coercion":
                            self.assertFalse(same(hostile[seat], conflict[seat]))
                            self.assertTrue(same(hostile[seat], printed[seat]))
                            continue
                        for robot in (0, 1):
                            self.assertTrue(same(hostile[robot], conflict[robot]), robot)
                        # the competitor's: only where the cell was turned
                        self.assertEqual(same(hostile[seat], printed[seat]),
                                         partner == "competition")
        # an inferred partner goal: the same Config for both pairings, so that the
        # hypothesis "competition" means the same thing in both
        inferred = dict(competition="table_hostile", partner_social_goal="inferred")
        goals, twin = ("competition", "cooperation"), ("conflict", "cooperation")
        with self.subTest(partner_social_goal="inferred"):
            hostile = models(goals, (0.8, 0.8), **inferred)
            conflict = models(twin, (0.8, 0.8), **inferred)
            self.assertFalse(same(hostile[0], conflict[0]))        # the coercion hypothesis
            self.assertTrue(same(hostile[1], conflict[1]))         # the cooperator's models
            without = dict(inferred, social_hypotheses=("cooperation", "conflict",
                                                        "competition", "exchange", "none"))
            hostile, conflict = models(goals, (0.8, 0.8), **without), models(
                twin, (0.8, 0.8), **without)
            for robot in (0, 1):
                self.assertTrue(same(hostile[robot], conflict[robot]), robot)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_more_of_the_hostile_table(self):
        """The other turned cell (an exchanging partner); the cell that asks for the
        destinations (a coercing partner: +R_j if it is taken to have the robot's own
        destination, as printed); a partner whose social goal has to be inferred, so that
        turned and untouched cells stand side by side in one term."""
        cfg = tight(competition="table_hostile")
        self.compare(cfg, (2, 2), ("competition", "exchange"), (2.0, 1.0), seed=71)
        self.compare(cfg, (2, 2), ("competition", "coercion"), (2.0, 1.5), state=crossed,
                     seed=72)
        self.compare(cfg, (2, 2), ("competition", "competition"), (2.0, 2.0), seed=73)
        self.compare(cfg, (1, 2), ("hinder", "competition"), (2.0, 1.5), seed=74)
        cfg = tight(competition="table_hostile", partner_social_goal="inferred")
        self.compare(cfg, (2, 2), ("competition", "cooperation"), (2.0, 1.0), state=crossed,
                     seed=75)

    # ---- a rival: a race against every partner but a cooperative one ----------------------
    # ``competition="rival"``.  compare() takes a robot's level-1 term by the partner it is
    # told to face (reference.level1_term) and its level-2 term hypothesis by hypothesis: the
    # race term (reference.level2_race) or the cell of the hostile table.
    def test_two_rivals(self):
        """Level 2 against level 2: each races, and takes the other for a level-1 robot
        that races.  (With R0 done and R1 behind: test_more_rivals.)"""
        cfg = tight(competition="rival")
        n1, n2 = self.compare(cfg, (2, 2), ("competition", "competition"), (2.0, 3.0), seed=80)
        self.assertEqual((n1, n2), (6, 4))     # K hypotheses about each; two roots and twins

    def test_a_rival_against_a_cooperator(self):
        """The rival's term is -R_j, the cell of the hostile table; the cooperator's +R_j
        holds the term of a level-1 rival that is told it faces a helper, -r(g_j)."""
        cfg = tight(competition="rival")
        self.compare(cfg, (2, 2), ("competition", "cooperation"), (0.8, 0.8), state=crossed,
                     seed=82)
        self.compare(cfg, (2, 2), ("cooperation", "competition"), (1.5, 2.0), seed=83)

    def test_a_rival_is_the_competitor_of_one_of_two_other_readings(self):
        """With known partner goals a rival is, by whom it faces, the competitor of
        "table_hostile" (a cooperative partner) or of "race" (anybody else, an exchanging
        partner included): the model each robot acts by is the same array -- rewards,
        values and Q on every state it can reach, bit for bit -- and is not the other
        reading's.  At levels (2, 2) and (1, 2); a cooperator with weight 0 is no helper.
        (Six cases; 23 with the slow tests, which add levels (2, 1), a conflict partner, a
        partner without social goal and a cooperator at level 0.)"""
        def models(goals, weights, levels, **options):
            cfg = tight(**options)
            w, gid, mdp, state, _ = solved(TINY, cfg, levels, goals, weights, state=crossed,
                                           seed=14)
            out = []
            for robot in (0, 1):
                depth, row = mdp.root[robot]
                if depth == 0:                       # a plain MDP: no social term in it
                    continue
                bank = mdp.bank(depth)
                states = states_of(w, mdp.reach_of(robot, gid[robot], state[2]))
                q = np.stack([mdp.q_values((depth, row), int(s), robot) for s in states])
                out.append((bank.R[row][states].copy(), bank.V[row][states].copy(), q))
            return out                               # per robot: R (S, 10), V (S,), Q (S, 10)

        def same(a, b) -> bool:
            return len(a) == len(b) and all(np.array_equal(x, y, equal_nan=True)
                                            for one, two in zip(a, b) for x, y in zip(one, two))

        cases = [(("competition", "cooperation"), (0.8, 0.8), (2, 2)),
                 (("competition", "exchange"), (0.8, 0.8), (2, 2)),
                 (("competition", "competition"), (0.8, 0.8), (2, 2)),
                 (("competition", "cooperation"), (0.8, 0.8), (1, 2)),
                 (("cooperation", "competition"), (1.0, 30.0), (2, 2)),
                 (("competition", "cooperation"), (2.0, 0.0), (2, 2))]       # no helper
        if SLOW:
            partners = ("cooperation", "exchange", "competition", "conflict", "none")
            cases += [(("competition", partner), (30.0, 1.0), (2, 2)) for partner in partners]
            cases += [(("competition", partner), (0.8, 0.8), (2, 2))
                      for partner in ("conflict", "none")]
            cases += [(("competition", partner), (0.8, 0.8), levels)
                      for partner in partners for levels in ((1, 2), (2, 1))
                      if (partner, levels) != ("cooperation", (1, 2))]
            cases += [(("competition", "cooperation"), (2.0, 1.0), (2, 0))]  # a level-0 one
        self.assertEqual(len(cases), 23 if SLOW else 6)
        for goals, weights, levels in cases:
            with self.subTest(goals=goals, weights=weights, levels=levels):
                helped = any(goals[i] == "competition" and goals[1 - i] == "cooperation"
                             and weights[1 - i] > 0.0 and levels[1 - i] > 0 for i in (0, 1))
                twin, other = (("table_hostile", "race") if helped
                               else ("race", "table_hostile"))
                rival = models(goals, weights, levels, competition="rival")
                self.assertGreater(rival[0][1].size, 500)                  # states compared
                self.assertTrue(same(rival, models(goals, weights, levels, competition=twin)))
                self.assertFalse(same(rival, models(goals, weights, levels,
                                                    competition=other)))

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_more_rivals(self):
        """Two rivals with R0 done and R1 behind; a level-1 rival that is told it faces a
        helper (its root is the table's entry, and the level-2 cooperator models it
        exactly); an exchanging partner (no helper: a race); a coercing partner (a race: no
        cell asks for the destinations); a hinderer that holds its object; a partner
        without social goal; a cooperator with weight 0; and partners whose social goal
        has to be inferred -- a rival's term then mixes the race term and -R_j over its
        hypotheses, and a level-1 rival that is not told whom it faces races."""
        cfg = tight(competition="rival")
        self.compare(cfg, (2, 2), ("competition", "competition"), (30.0, 0.8), state=one_done,
                     seed=81)
        self.compare(cfg, (1, 2), ("competition", "cooperation"), (2.0, 1.0), seed=84)
        self.compare(cfg, (2, 2), ("competition", "exchange"), (2.0, 1.0), seed=85)
        self.compare(cfg, (2, 2), ("competition", "coercion"), (2.0, 1.5), state=crossed,
                     seed=86)
        self.compare(cfg, (2, 1), ("competition", "hinder"), (2.0, 1.5), state=crossed,
                     seed=87)
        self.compare(cfg, (2, 0), ("competition", "none"), (3.0, 1.0), state=hostage, seed=88)
        self.compare(cfg, (2, 2), ("competition", "cooperation"), (2.0, 0.0), seed=89)
        self.compare(cfg, (2, 2), ("competition", "competition"), (0.8, 30.0),
                     state=other_done, seed=90)
        cfg = tight(competition="rival", partner_social_goal="inferred")
        self.compare(cfg, (2, 2), ("competition", "cooperation"), (2.0, 1.0), state=crossed,
                     seed=91)
        self.compare(cfg, (2, 2), ("competition", "competition"), (2.0, 3.0), state=one_done,
                     seed=92)
        cfg = tight(competition="rival", partner_social_goal="inferred",
                    social_hypotheses=("cooperation", "competition", "none"))
        self.compare(cfg, (2, 1), ("cooperation", "competition"), (1.0, 2.0), seed=93)

    # ---- hand-off cells and rooms: an object reaches its owner through the other's hands --
    def test_an_exchange_through_a_doorway(self):
        """Two rooms, a hand-off cell in the doorway, and nobody finishes a partner's task
        for it: at the start, and with each robot holding the other's object, R0 in the
        doorway -- where "place" puts the object down -- and R1 next to it, where "give"
        hands over to a partner with free hands (R0 may give from the doorway as well)."""
        cfg = tight(set_down_for_partner=False)
        n1, n2 = self.compare(cfg, (2, 2), ("exchange", "exchange"), (2.0, 3.0), scene=PASS,
                              seed=101)
        self.assertEqual((n1, n2), (2 * 2, 2 * 2))   # K hypotheses per robot; roots and twins
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (30.0, 30.0), state=at_the_doorway,
                     scene=PASS, seed=102)

    def test_an_exchange_by_putting_down_and_picking_up(self):
        """The same with ``hand_over=False``: an object changes hands only over the hand-off
        cell."""
        cfg = tight(set_down_for_partner=False, hand_over=False)
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (30.0, 30.0), state=at_the_doorway,
                     scene=PASS, seed=103)
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (1.0, 2.0), state=put_down,
                     scene=PASS, seed=104)

    def test_rooms_at_level_1_and_a_partner_without_social_goal(self):
        cfg = tight(set_down_for_partner=False, hand_over=False)
        self.compare(cfg, (1, 2), ("cooperation", "none"), (3.0, 1.0), state=both_lying,
                     scene=PASS, seed=105)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_more_of_rooms_and_hand_off_cells(self):
        """The older rules and an inferred partner goal in the rooms; set-downs for the
        partner allowed there; cooperation and conflict."""
        self.compare(tight(), (2, 2), ("cooperation", "cooperation"), (2.0, 2.0), scene=PASS,
                     state=put_down, seed=106)
        self.compare(tight(set_down_for_partner=False, exchange_level1="copy"), (2, 2),
                     ("exchange", "exchange"), (2.0, 2.0), scene=PASS, state=both_lying,
                     seed=107)
        self.compare(tight(set_down_for_partner=False, forced_swap=True, held_reward="carrier",
                           hands_full_reward="none"), (2, 1), ("conflict", "cooperation"),
                     (1.0, 2.0), scene=PASS, state=at_the_doorway, seed=108)
        cfg = tight(set_down_for_partner=False, partner_social_goal="inferred",
                    idle_hypothesis=True, social_hypotheses=("exchange", "cooperation", "none"))
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (2.0, 2.0), scene=PASS,
                     state=put_down, seed=109)

    # ---- every social goal inferred, and what an exchanger is at level 1 -------------------
    def test_an_inferred_exchange_with_a_copy_that_does_not_grow_with_the_weight(self):
        """``exchange_level1="copy_unweighted"`` with the partner's social goal inferred over
        the five goals of the paper.  Unequal weights: the copy of a level-1 exchanger is n
        for both robots, not n times its weight (the reference is told so; with the
        weighted copy it would be 7.5 and 0.5 here)."""
        cfg = tight(set_down_for_partner=False, hand_over=False,
                    partner_social_goal="inferred", exchange_level1="copy_unweighted")
        n1, n2 = self.compare(cfg, (2, 2), ("exchange", "exchange"), (30.0, 2.0), scene=PASS,
                              seed=114)
        # per robot: K destinations x the five kinds of partner; its own root and twin
        self.assertEqual((n1, n2), (2 * 2 * 5, 2 * 2))

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_more_of_the_inferred_exchange(self):
        """At 30 / 30 with each robot holding the other's object at the doorway; with an
        object lying in the doorway; with hand-overs allowed; the weighted copy next to it;
        and a partner with weight 0."""
        cfg = tight(set_down_for_partner=False, hand_over=False,
                    partner_social_goal="inferred", exchange_level1="copy_unweighted")
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (30.0, 30.0), scene=PASS,
                     state=at_the_doorway, seed=115)
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (2.0, 3.0), scene=PASS,
                     state=put_down, seed=116)
        cfg = tight(set_down_for_partner=False, partner_social_goal="inferred",
                    exchange_level1="copy_unweighted")
        self.compare(cfg, (2, 2), ("exchange", "cooperation"), (1.0, 5.0), scene=PASS,
                     state=both_lying, seed=117)
        cfg = tight(set_down_for_partner=False, hand_over=False,
                    partner_social_goal="inferred", exchange_level1="copy")
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (5.0, 2.0), scene=PASS, seed=118)
        # a partner with weight 0: whatever goal it is taken to have, it has no social term
        # and no copy, so the five hypotheses about it are one model
        cfg = tight(set_down_for_partner=False, hand_over=False,
                    partner_social_goal="inferred", exchange_level1="copy_unweighted")
        n1, _ = self.compare(cfg, (2, 2), ("exchange", "exchange"), (3.0, 0.0), scene=PASS,
                             seed=121)
        self.assertEqual(n1, 2 * 5 + 2 * 1)

    def test_a_level_1_exchanger_with_the_unweighted_copy(self):
        """Told its partner's goal, a level-1 exchanger then has the term n r(g_j); a
        robot with weight 0 has none."""
        cfg = tight(set_down_for_partner=False, hand_over=False,
                    exchange_level1="copy_unweighted")
        self.compare(cfg, (1, 2), ("exchange", "cooperation"), (3.0, 1.0), scene=PASS,
                     state=at_the_doorway, seed=119)
        self.compare(cfg, (1, 1), ("exchange", "exchange"), (3.0, 0.0), scene=PASS, seed=120)

    # ---- hand-overs only: nobody finishes its partner's task -----------------------------
    def test_hand_overs_only_once_a_goal_is_done(self):
        """``set_down_for_partner=False``.  A model sets its own robot's object down and
        nobody else's, so the blocks in which a robot's goal is done are reached by that
        robot's models only -- and its partner's policies are read there all the same.
        (Until 2026-10-07 they had not been computed there, and a level-2 robot stopped one
        step short of its destination: ``SocialMDP.reach_of``.)"""
        cfg = tight(set_down_for_partner=False)
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (2.0, 2.0), seed=110)
        self.compare(cfg, (2, 2), ("cooperation", "conflict"), (1.0, 2.0), state=crossed,
                     seed=111)
        self.compare(cfg, (2, 1), ("cooperation", "competition"), (2.0, 1.0), state=one_done,
                     seed=112)

    def test_every_model_is_solved_wherever_any_model_gets(self):
        """With hand-overs only the two robots' own dynamics reach different blocks; every
        model is solved, and every partner policy defined, on all of them."""
        cfg = tight(set_down_for_partner=False)
        w, gid, mdp, state, _ = solved(TINY, cfg, (2, 2), ("exchange", "exchange"), (2.0, 2.0),
                                       seed=113)
        K = w.K
        own = [set(w.reach((mdp.tabs[a * K + gid[a]], mdp.tabs[2 * K + a * K + gid[a]]),
                           state[2], (True, True)).tolist()) for a in (0, 1)]
        self.assertNotEqual(own[0], own[1])
        everywhere = set(mdp.reach_of(0, gid[0], state[2]).tolist())
        self.assertEqual(everywhere, set(mdp.reach_of(1, gid[1], state[2]).tolist()))
        self.assertTrue(own[0] | own[1] <= everywhere)
        self.assertTrue(any(w.done[b, 0] and not w.done[b, 1] for b in everywhere))
        self.assertTrue(any(w.done[b, 1] and not w.done[b, 0] for b in everywhere))
        states = states_of(w, sorted(everywhere))
        L1 = mdp.L1
        rows = [r for r in range(L1.M) if L1.psi_row[r] >= 0 and L1.active[r]]
        self.assertEqual(len(rows), 2 * K)
        for r in rows:                           # a distribution over actions at every state
            total = L1.psi[L1.psi_row[r]][states].astype(np.float64).sum(axis=1)
            self.assertLess(float(np.abs(total - 1.0).max()), 1e-5)
        for bank in (mdp.L1, mdp.L2):
            for g in range(bank.G):
                total = bank.P[g][states].sum(axis=1)
                self.assertLess(float(np.abs(total - 1.0).max()), 1e-9)

    def test_reachable_blocks_are_cached_by_what_a_table_holds(self):
        """Not by which array it is: the cache key once held ``id(table)``, and a table in
        a recycled array was handed the blocks of the table that had been there before."""
        cfg = tight(set_down_for_partner=False)
        w, gid = make_world(TINY, cfg)
        first, second = w.model_tab(0, gid[0]), w.model_tab(1, gid[1])
        table = first.copy()
        blocks_first = w.reach((table,), w.block0, (True, True)).copy()
        table[:] = second                        # the same array, another table
        blocks_second = w.reach((table,), w.block0, (True, True))
        fresh, _ = make_world(TINY, cfg)
        expected = fresh.reach((fresh.model_tab(1, gid[1]),), fresh.block0, (True, True))
        self.assertTrue(np.array_equal(blocks_second, expected))
        self.assertFalse(np.array_equal(blocks_first, blocks_second))

    # ---- the defaults since 2026-10-08 -------------------------------------------------
    def test_the_defaults_of_now(self):
        """``Config``'s own defaults: every partner goal inferred over seven social goals
        (the paper's five, none and hinder), the unweighted copy, a rival, a hinderer
        without a goal of its own (2026-10-14: -w r(g_j) - r(g_i), no longer conflict's
        -w r(g_j) at level 1): seven models per destination for seven goals.  A rival that
        judges whom it faces by its belief next to an exchanger, in the rooms with the
        doorway; a cooperator next to a rival at 30 / 30 with each holding the other's
        object."""
        cfg = tight_now(set_down_for_partner=False, hand_over=False)
        self.assertEqual((cfg.partner_social_goal, cfg.social_hypotheses, cfg.competition,
                          cfg.exchange_level1, cfg.partner_object),
                         ("inferred", HYPOTHESES, "rival", "copy_unweighted", "inferred"))
        n1, n2 = self.compare(cfg, (2, 2), ("competition", "exchange"), (2.0, 3.0), scene=PASS,
                              seed=131)
        # per robot: M x K physical goals (2 objects, 2 destinations) x 7 terms; root, twin
        self.assertEqual((n1, n2), (2 * (2 * 2 * 7), 2 * 2))
        known = tight_now(set_down_for_partner=False, hand_over=False, partner_object="known")
        n1, n2 = self.compare(known, (2, 2), ("competition", "exchange"), (2.0, 3.0),
                              scene=PASS, seed=131)
        self.assertEqual((n1, n2), (2 * (2 * 7), 2 * 2))   # per robot: K x 7
        self.compare(cfg, (2, 2), ("cooperation", "competition"), (30.0, 30.0), scene=PASS,
                     state=at_the_doorway, seed=132)

    def test_a_cooperator_that_cooperates_with_three_goals(self):
        """``Config.cooperate_with``'s default (2026-10-11): a level-2 cooperator next to a
        level-1 hinderer, which it may take for a conflict partner, a hinderer, a coercer or a
        robot without social goal (no term) or for a cooperative, competitive or exchanging
        one (Table I's cell); and with only a hinderer and a robot without social goal
        cooperated with.  The reference leaves the cell out by the partner's name."""
        cfg = tight_now()
        self.assertEqual(cfg.cooperate_with, ("cooperation", "competition", "exchange"))
        self.compare(cfg, (1, 2), ("hinder", "cooperation"), (1.0, 1.0), seed=137)
        self.compare(tight_now(cooperate_with=("hinder", "none")), (1, 2),
                     ("hinder", "cooperation"), (2.0, 1.0), state=hostage, seed=138)

    def test_a_hinderer_at_level_2(self):
        """Since 2026-10-11 a hinderer reasons at level 2 unless a level is asked for
        (``Config.level_free_level``): its term is -w r(g_j) (since 2026-10-14 without a goal
        of its own as well: -r(g_i)), and it predicts its partner by the partner's level-1
        models -- seven goals held possible, the partner's object inferred.  Next to a
        level-2 cooperator, on the open map; and the hinderer of before, with its own goal."""
        cfg = tight_now()
        self.assertEqual((cfg.level_free_level, cfg.hinder_own_goal), (2, False))
        self.compare(cfg, (2, 2), ("hinder", "cooperation"), (1.0, 1.0), seed=139)
        self.compare(tight_now(hinder_own_goal=True), (2, 2), ("hinder", "cooperation"),
                     (1.0, 1.0), seed=139)

    def test_swaps_and_steals(self):
        """``Config.swap_on_place`` and ``Config.steal`` (2026-10-12): a level-1 hinderer
        holding its partner's object next to a level-2 cooperator, each told the other's
        goal -- the partner may take the object out of its hands."""
        self.compare(tight(swap_on_place=True, steal=True), (1, 2), ("hinder", "cooperation"),
                     (2.0, 2.0), state=hostage, seed=141)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_swaps_and_steals_at_level_2_inferring(self):
        """The same options with both robots at level 2 and every goal and object inferred:
        coercion against hinder, each holding the other's object."""
        self.compare(tight_now(swap_on_place=True, steal=True), (2, 2), ("coercion", "hinder"),
                     (1.0, 1.0), state=crossed, seed=142)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_more_of_the_defaults_of_now(self):
        """Two robots without social goals (whose partners hold seven goals possible for
        them), a hinderer at level 1 next to a cooperator, two rivals once one is done,
        coercion and conflict -- on the open map, where a robot may set its partner's
        object down on the partner's destination."""
        cfg = tight_now()
        self.compare(cfg, (2, 2), ("none", "none"), (30.0, 30.0), seed=133)
        self.compare(cfg, (1, 2), ("hinder", "cooperation"), (2.0, 2.0), state=hostage,
                     seed=134)
        self.compare(cfg, (2, 2), ("competition", "competition"), (30.0, 1.0), state=one_done,
                     seed=135)
        self.compare(cfg, (2, 2), ("coercion", "conflict"), (1.0, 2.0), state=crossed,
                     seed=136)

    # ---- a robot whose hands are full of its partner's object ---------------------------
    def test_no_goal_reward_with_full_hands(self):
        """``hands_full_reward="none"``: the hostage costs its holder its own goal reward."""
        cfg = tight(hands_full_reward="none")
        self.compare(cfg, (1, 2), ("hinder", "cooperation"), (3.0, 2.0), state=hostage, seed=15)
        self.compare(cfg, (1, 1), ("competition", "competition"), (2.0, 2.0), state=crossed,
                     seed=16)

    # ---- Table I at levels 1 and 2 -------------------------------------------------------
    def test_level_1_cooperation_and_conflict(self):
        self.compare(tight(), (1, 1), ("cooperation", "conflict"), (1.0, 1.0), seed=3)

    def test_level_2_conflict_against_cooperation(self):
        self.compare(tight(), (2, 2), ("conflict", "cooperation"), (0.8, 0.8), state=crossed,
                     seed=4)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_level_2_exchange_with_inferred_goals_and_the_older_rules(self):
        """Everything the default profile switches off: the partner's social goal inferred
        over five hypotheses, the idle mode, the partner's cost in R_j, the reward that
        follows a carried object, forced swaps."""
        cfg = tight(partner_social_goal="inferred", idle_hypothesis=True, social_cost=True,
                    held_reward="carrier", forced_swap=True)
        self.compare(cfg, (2, 2), ("exchange", "exchange"), (1.0, 1.0), state=crossed, seed=5)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_level_2_coercion_and_competition_with_marginal_beliefs(self):
        cfg = tight(partner_social_goal="inferred", social_belief="marginals",
                    exchange_level1="copy", exchange_share=0.6, competition="table",
                    social_hypotheses=("cooperation", "competition", "coercion", "exchange",
                                       "none"))
        self.compare(cfg, (2, 2), ("coercion", "competition"), (1.0, 2.0), seed=6)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_a_level_2_cooperator_that_has_to_infer_the_racer(self):
        """The partner's social goal is not known: the cooperator holds three hypotheses
        about it, one of them a race."""
        cfg = tight(partner_social_goal="inferred",
                    social_hypotheses=("cooperation", "competition", "none"))
        self.compare(cfg, (2, 1), ("cooperation", "competition"), (1.0, 2.0), seed=22)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_hinder_at_level_2_is_still_minus_the_partners_goal_reward(self):
        """A hinderer rolled out at level 2 (rollout() called directly) keeps -w r(g_j)."""
        self.compare(tight(competition="table"), (2, 2), ("hinder", "competition"), (2.0, 2.0),
                     state=hostage, seed=7)


if __name__ == "__main__":
    unittest.main()
