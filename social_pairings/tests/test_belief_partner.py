"""Whom the belief planner takes its partner to be (Config.belief_partner, 2026-11-05
evening): the hypotheses of the planning robot's own posteriors about its partner instead of
its true goals -- the world of a hypothesis, the partner's root under it, the hypotheses
kept and their weights, a decision and a rollout.
"""
import json
import unittest

import numpy as np

from social_pairings import goals as G
from social_pairings.beliefkernels import choice_tables
from social_pairings.beliefplan import INFERRED
from social_pairings.config import ADDED_FIELDS, Config
from social_pairings.pairing import run_pairing
from social_pairings.scenes import Scene
from social_pairings.serialize import to_json
from social_pairings.tests.test_belief_plan import PAIR, planner
from social_pairings.tests.test_models import TINY, tight_explicit


def point_mass(mdp, r, goal, social):
    """Robot r's posteriors about its partner a point mass on (goal, social goal)."""
    p = 1 - r
    b = mdp.about[p]
    lp = np.full(mdp.PG, -np.inf)
    lp[goal] = 0.0
    types = [int(t) for t in mdp.hyp_goals[r]]
    ls = np.full(len(types), -np.inf)
    ls[types.index(int(social))] = 0.0
    b.log_phys, b.log_social = lp, ls


class Options(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(Config().belief_partner, "inferred")
        self.assertEqual(ADDED_FIELDS["belief_partner"], "oracle")   # the records of before
        for bad in (dict(belief_partner="truth"), dict(belief_partner_mass=1.0),
                    dict(belief_partner_mass=-0.1), dict(belief_partner_max=0),
                    dict(belief_partner_max=1.5), dict(belief_partner_max=True)):
            with self.assertRaises(ValueError):
                Config(**bad)


class Worlds(unittest.TestCase):
    def test_the_world_of_the_true_goals_is_the_world(self):
        w, mdp, st, pl, ok, lo, hi = planner()
        H = pl._hyp(mdp, 1, int(mdp.own_goal[0]), int(mdp.goals[0]), 1.0, None, False)
        np.testing.assert_array_equal(H.tab, w.true_tab)
        for a, b in zip(H.events, pl.events[1]):
            np.testing.assert_array_equal(a, b)

    def test_every_hypothesis_has_a_world(self):
        """Every (physical goal, social goal) the robot holds possible -- the partner's goal
        object the robot's own included -- has a hands table and a model."""
        w, mdp, st, pl, ok, lo, hi = planner()
        for g in range(mdp.PG):
            for t in mdp.hyp_goals[1]:
                H = pl._hyp(mdp, 1, g, int(t), 1.0, None, False)
                self.assertEqual(H.tab.shape, w.true_tab.shape)
                self.assertEqual(H.slot.group, g)
                self.assertEqual(H.over["var"], 0 * mdp.PG + g)

    def test_the_partner_with_the_robots_own_goal(self):
        """A partner taken to want the robot's own object on the robot's own destination: one
        set-down there finishes both goals -- no successor with one goal done."""
        w, mdp, st, pl, ok, lo, hi = planner()
        p, r = 0, 1
        mine = [g for g in range(mdp.PG) if int(mdp.pg_obj[p][g]) == r
                and int(w.dest_site[int(mdp.pg_dst[p][g])]) == int(w.site_of[w.true_dest[r]])]
        self.assertEqual(len(mine), 1)
        H = pl._hyp(mdp, r, mine[0], G.CONFLICT, 1.0, None, False)
        one = w.done[:, 0] != w.done[:, 1]
        self.assertFalse(one[H.tab].any())
        self.assertTrue((H.tab == w.T).any())          # and both goals can be done
        other = [g for g in range(mdp.PG) if int(mdp.pg_obj[p][g]) == r and g != mine[0]]
        H2 = pl._hyp(mdp, r, other[0], G.CONFLICT, 1.0, None, False)
        self.assertTrue(one[H2.tab].any())             # elsewhere it takes the object for itself

    def test_the_partner_under_its_true_goals_is_as_solved(self):
        """The partner's root re-solved under the hypothesis of its true goals, at the
        belief it holds: the root the rollout solved (to the solver's tolerance), the same
        choices."""
        w, mdp, st, pl, ok, lo, hi = planner()
        H = pl._hyp(mdp, 1, int(mdp.own_goal[0]), int(mdp.goals[0]), 1.0, None, False)
        Q0, _, _ = pl._partner_tables(mdp, 1, None, ok, lo, hi, None)          # the oracle
        Q1, _, stats = pl._partner_tables(mdp, 1, None, ok, lo, hi, None, H)    # re-solved
        fin = np.isfinite(Q0)
        np.testing.assert_array_equal(fin, np.isfinite(Q1))
        np.testing.assert_allclose(Q1[fin], Q0[fin], atol=1e-3, rtol=0)
        b0, a0, b1, a1 = (np.empty(w.S, dtype=np.int8) for _ in range(4))
        choice_tables(Q0, pl.margin, b0, a0)
        choice_tables(Q1, pl.margin, b1, a1)
        s = w.sid(st)
        self.assertEqual(int(b0[s]), int(b1[s]))
        self.assertEqual(int(a0[s]), int(a1[s]))


class Hypotheses(unittest.TestCase):
    def test_kept_and_weighted(self):
        w, mdp, st, pl, ok, lo, hi = planner(belief_partner="inferred", belief_partner_max=3,
                                             belief_partner_mass=0.05)
        p = 0
        types = [int(t) for t in mdp.hyp_goals[1]]
        b = mdp.about[p]
        phys = np.zeros(mdp.PG)
        g0, g1 = 0, 1                       # two physical goals
        phys[g0], phys[g1] = 0.7, 0.3
        soc = np.zeros(len(types))
        soc[types.index(G.HINDER)] = 0.6
        soc[types.index(G.CONFLICT)] = 0.4
        with np.errstate(divide="ignore"):
            b.log_phys, b.log_social = np.log(phys), np.log(soc)
        hyps, covered = pl._hypotheses(mdp, 1)
        same_object = int(mdp.pg_obj[p][g0]) == int(mdp.pg_obj[p][g1])
        # the goal-less hinderer: one hypothesis per object, its most likely destination
        hinder = [H for H in hyps if H.social == G.HINDER]
        self.assertEqual(len(hinder), 1 if same_object else 2)
        self.assertEqual(hinder[0].goal, g0)
        self.assertAlmostEqual(sum(H.weight for H in hyps), 1.0)
        self.assertLessEqual(len(hyps), 3)
        masses = [sum(m for _, _, m in H.members) for H in hyps]
        self.assertEqual(masses, sorted(masses, reverse=True))
        self.assertAlmostEqual(covered, sum(masses))
        for H, m in zip(hyps, masses):
            self.assertAlmostEqual(H.weight, m / sum(masses))

    def test_one_hypothesis(self):
        w, mdp, st, pl, ok, lo, hi = planner(belief_partner="inferred", belief_partner_max=1)
        hyps, covered = pl._hypotheses(mdp, 1)
        self.assertEqual(len(hyps), 1)
        self.assertEqual(hyps[0].weight, 1.0)
        self.assertFalse(hyps[0].oracle)


class Decisions(unittest.TestCase):
    def test_a_point_mass_on_the_truth_is_the_oracle(self):
        """Robot r sure of its partner's true goals: the decision of the inferred partner is
        the oracle's (the partner re-solved, so to the solver's tolerance)."""
        out = {}
        for who in ("oracle", "inferred"):
            w, mdp, st, pl, ok, lo, hi = planner(belief_partner=who, belief_horizon=5,
                                                 belief_points=6, belief_refinements=1)
            point_mass(mdp, 1, int(mdp.own_goal[0]), int(mdp.goals[0]))
            q = pl.decide(mdp, 0, st, (None, None))[1]
            out[who] = (q, pl.decisions[-1])
        (q0, r0), (q1, r1) = out["oracle"], out["inferred"]
        fin = np.isfinite(q0)
        np.testing.assert_array_equal(fin, np.isfinite(q1))
        np.testing.assert_allclose(q1[fin], q0[fin], atol=1e-3, rtol=0)
        self.assertEqual(r0["partner_predicted"], r1["partner_predicted"])
        self.assertEqual(len(r1["hypotheses"]), 1)
        self.assertEqual(r1["response"], INFERRED)

    def test_the_decision_is_the_weighted_plans(self):
        """Two hypotheses: the decision's Q is their posterior-weighted Q."""
        w, mdp, st, pl, ok, lo, hi = planner(belief_partner="inferred", belief_planning="frozen",
                                             belief_horizon=3)
        p = 0
        types = [int(t) for t in mdp.hyp_goals[1]]
        g = int(mdp.own_goal[p])
        lp = np.full(mdp.PG, -np.inf)
        lp[g] = 0.0
        ls = np.full(len(types), -np.inf)
        ls[types.index(G.HINDER)], ls[types.index(G.COOPERATION)] = np.log(0.7), np.log(0.3)
        mdp.about[p].log_phys, mdp.about[p].log_social = lp, ls
        q = pl.decide(mdp, 0, st, (None, None))[1]
        rec = pl.decisions[-1]
        hs = rec["hypotheses"]
        self.assertEqual([h["social"] for h in hs], ["hinder", "cooperation"])
        np.testing.assert_allclose([h["weight"] for h in hs], [0.7, 0.3])
        fin = np.isfinite(q)
        want = 0.7 * np.asarray(hs[0]["q"]) + 0.3 * np.asarray(hs[1]["q"])
        np.testing.assert_allclose(q[fin], want[fin])
        pred = rec["partner_prediction"]
        self.assertAlmostEqual(sum(pred.values()), 1.0)


class Rollout(unittest.TestCase):
    def test_tiny_rollout_inferred(self):
        scene = Scene(name="tiny", **TINY)
        cfg = tight_explicit(partner_object="inferred", inertia=1.0, swap_on_place=True,
                             steal=True, belief_planning="updated", objective="task",
                             belief_horizon=5, belief_points=5, belief_refinements=1,
                             belief_partner_max=2, value_maps=False, belief_detail=False)
        res = run_pairing(scene, PAIR, levels=(2, 2), cfg=cfg, n_steps=4)
        bp = res["belief_planner"]
        self.assertEqual(bp["partner"], "inferred")
        self.assertEqual(bp["response"], INFERRED)
        for d, row in zip(bp["decisions"], res["trace"]):
            self.assertEqual(d["chosen"], row["action"][1])
            self.assertAlmostEqual(sum(h["weight"] for h in d["hypotheses"]), 1.0)
            self.assertLessEqual(len(d["hypotheses"]), 2)
            self.assertAlmostEqual(sum(d["partner_prediction"].values()), 1.0)
        json.dumps(to_json(res))


if __name__ == "__main__":
    unittest.main()
