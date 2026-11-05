"""Goal-less physical hypotheses (Config.goalless_hypotheses, 2026-11-05 night; the user: "I
think the problem is that we are trying to model hinder, but hinder has no goal.  I think we
should add in the goal posterior and object posterior no object and no destination") and the
level-3 hinderer (Config.belief_planning_goals with "hinder"; "We may also want to make a
hindering agent level 3 so its able to understand the social much better").

The option; the hypothesis space (object M "no object", destination K "no destination", their
product); the extended world tables (reward 0, never finished); every goal-less hypothesis the
same model (its tables, one slot, one row); a goal-less level-0 model stays; Eq. 5's
transition term on goal-less hypotheses; a hinderer without a goal of its own has none in the
world either; the posteriors' sizes; rollouts with the hinderer and the coercer at level 3.
"""
import json
import unittest

import numpy as np

from social_pairings import goals as G
from social_pairings.config import ADDED_FIELDS, Config
from social_pairings.constants import GIVE, N_ACTIONS, WAIT
from social_pairings.pairing import run_pairing
from social_pairings.scenes import Scene
from social_pairings.serialize import to_json
from social_pairings.tests.test_models import TINY, solved, tight_explicit

PAIR = ("hinder", "coercion")      # R0 hinders (no goal of its own), R1 coerces


def model(goalless=True, pair=PAIR, **options):
    cfg = tight_explicit(partner_object=options.pop("partner_object", "inferred"),
                         inertia=1.0, swap_on_place=True, steal=True,
                         goalless_hypotheses=goalless, **options)
    w, gid, mdp, st, _ = solved(TINY, cfg, (2, 2), pair, (1.0, 1.0), None, seed=5)
    return w, mdp, st


class Options(unittest.TestCase):
    def test_option(self):
        self.assertFalse(Config().goalless_hypotheses)          # the paper's space by default
        self.assertFalse(ADDED_FIELDS["goalless_hypotheses"])
        with self.assertRaises(ValueError):
            Config(goalless_hypotheses=True, physical_likelihood="cut")
        self.assertEqual(Config().belief_planning_goals, ("coercion",))
        Config(belief_planning="updated", belief_planning_goals=("coercion", "hinder"))


class Space(unittest.TestCase):
    def test_the_hypotheses(self):
        w, mdp, st = model()
        M, K = w.M, w.K
        self.assertEqual((mdp.n_obj, mdp.n_dst, mdp.PG), (M + 1, K + 1, (M + 1) * (K + 1)))
        for i in (0, 1):
            pairs = set(zip(mdp.pg_obj[i].tolist(), mdp.pg_dst[i].tolist()))
            self.assertEqual(pairs, {(q, k) for q in range(M + 1) for k in range(K + 1)})
            less = (mdp.pg_obj[i] == M) | (mdp.pg_dst[i] == K)
            np.testing.assert_array_equal(mdp.goal_less[i], less)
            self.assertEqual(int(less.sum()), M + K + 1)
            self.assertEqual(mdp.goalless_rep[i], int(np.flatnonzero(less)[0]))
            np.testing.assert_array_equal(mdp.pg_acc[i], np.where(less, M, mdp.pg_obj[i]))
        # the coercer keeps its goal; the hinderer without a goal of its own has none
        self.assertTrue(mdp.goal_less[0, mdp.own_goal[0]])
        self.assertFalse(mdp.goal_less[1, mdp.own_goal[1]])
        self.assertEqual(mdp.goal_less_robot, (True, False))
        # without the option: the paper's space
        w0, mdp0, _ = model(goalless=False)
        self.assertEqual(mdp0.PG, w0.M * w0.K)
        self.assertFalse(mdp0.goal_less.any())
        self.assertEqual(mdp0.goalless_rep, (-1, -1))
        self.assertEqual(w0.goal_less_robots, (False, False))

    def test_known_objects(self):
        w, mdp, st = model(partner_object="known")
        self.assertEqual(mdp.PG, w.K + 1)
        np.testing.assert_array_equal(mdp.goal_less[0], np.arange(w.K + 1) == w.K)

    def test_the_extended_tables(self):
        """Object M and destination K: reward 0 and never finished, in every class and block;
        the real indices as the world's own tables."""
        w, mdp, st = model()
        M, K = w.M, w.K
        np.testing.assert_array_equal(w.rate_cls_x[:, :, :K], w.rate_cls)
        np.testing.assert_array_equal(w.fin_cls_x[:, :, :K], w.fin_cls)
        self.assertTrue((w.rate_cls_x[:, :, K] == 0.0).all())
        self.assertTrue((w.fin_cls_x[:, :, K] == 0).all())
        np.testing.assert_array_equal(w.cls_obj_x[:, :, :M], w.cls_obj)
        self.assertTrue((w.cls_obj_x[:, :, M] == w.zero_cls).all())
        self.assertTrue((w.rate_cls_x[w.zero_cls] == 0.0).all())
        self.assertTrue((w.fin_cls_x[w.zero_cls] == 0).all())
        self.assertEqual(int(w.dest_x[K]), -1)
        self.assertEqual(int(w.dest_site_x[K]), 0)


class OneModel(unittest.TestCase):
    def test_every_goalless_hypothesis_is_one_model(self):
        """The dynamics tables of every goal-less hypothesis are the same (no site, no goal
        object), and the bank holds one level-1 slot for them, whose rows every goal-less
        hypothesis reads."""
        w, mdp, st = model()
        PG = mdp.PG
        for o in (0, 1):
            rep = mdp.goalless_rep[o]
            for g in np.flatnonzero(mdp.goal_less[o]).tolist():
                for base in (0, 2 * PG, 4 * PG):
                    np.testing.assert_array_equal(mdp.tabs[base + o * PG + g],
                                                  mdp.tabs[base + o * PG + rep])
                for t in range(len(mdp.hyp_goals[1 - o])):
                    self.assertEqual(mdp.hypothesis_row(o, g, t),
                                     mdp.hypothesis_row(o, rep, t))
            groups = {k for (a, d, k) in mdp.slots if a == o and d == 1}
            want = {g for g in range(PG) if not mdp.goal_less[o, g]} | {rep}
            self.assertEqual(groups, want)

    def test_a_goalless_level0_model_stays(self):
        """No goal reward anywhere: the level-0 model's best action is to stay, its value
        about the cost of staying forever."""
        w, mdp, st = model()
        s = w.sid(st)
        for o in (0, 1):
            g = mdp.goalless_rep[o]
            q = mdp.q_values((0, o * mdp.PG + g), s, o)
            fin = np.isfinite(q)
            self.assertEqual(int(np.argmax(np.where(fin, q, -np.inf))), WAIT)
            stay_forever = -float(w.cfg.costs[WAIT]) / (1.0 - w.cfg.gamma)
            self.assertLess(abs(float(q[WAIT]) - stay_forever), 1e-3 * abs(stay_forever))


class TransitionTerm(unittest.TestCase):
    def test_goalless_hypotheses_under_eq5(self):
        """Finishing rules every goal-less hypothesis out; a hand-over the partner took rules
        them out (they take nothing), one it did not take keeps them."""
        w, mdp, st = model()
        j, i = 1, 0                                 # about robot 1 (it has a goal)
        s = w.sid(st)
        less = mdp.goal_less[j]
        acts = (WAIT, WAIT)
        done = dict(done_now=(False, True), failed=(False, False))
        keep = mdp.transition_term(j, s, acts, st, done)
        self.assertFalse(keep[less].any())
        quiet = dict(done_now=(False, False), failed=(False, False))
        self.assertTrue(mdp.transition_term(j, s, acts, st, quiet).all())


class World(unittest.TestCase):
    def test_the_hinderer_has_no_goal_in_the_world(self):
        w, mdp, st = model()
        self.assertEqual(w.goal_less_robots, (True, False))
        self.assertEqual(w.true_gs[0], 0)
        self.assertEqual(w.true_acc, (-1, 1))
        self.assertGreater(w.true_gs[1], 0)
        w0, mdp0, _ = model(goalless=False)
        self.assertEqual(w0.true_acc, (0, 1))
        self.assertGreater(w0.true_gs[0], 0)

    def test_beliefs_sizes(self):
        w, mdp, st = model()
        for i in (0, 1):
            ob, db = mdp.object_belief(i), mdp.goal_belief(i)
            self.assertEqual((ob.size, db.size), (w.M + 1, w.K + 1))
            self.assertAlmostEqual(float(ob.sum()), 1.0, places=12)
            self.assertAlmostEqual(float(db.sum()), 1.0, places=12)
            lo, ld = mdp.factors(1 - i, mdp.about[1 - i].log_phys)
            np.testing.assert_allclose(np.exp(lo), ob, atol=1e-12)
            np.testing.assert_allclose(np.exp(ld), db, atol=1e-12)


class Rollouts(unittest.TestCase):
    def test_tiny_rollout(self):
        """Six transitions: the coercer reads the goal-less hinderer as having no object more
        than the prior says; the records are whole."""
        scene = Scene(name="tiny", **TINY)
        cfg = tight_explicit(partner_object="inferred", inertia=1.0, swap_on_place=True,
                             steal=True, goalless_hypotheses=True, value_maps=False,
                             belief_detail=True)
        res = run_pairing(scene, PAIR, levels=(2, 2), cfg=cfg, n_steps=6)
        ob = np.asarray(res["final_object_belief"][1], dtype=float)
        self.assertEqual(ob.size, 3)
        self.assertGreater(float(ob[-1]), 1.0 / 3.0)      # "no object" for the hinderer
        json.dumps(to_json(res))

    def test_level3_hinderer(self):
        """Both robots plan over their partner's belief and read it at level 2 (the hinderer
        a level-3 agent too): decisions of both in the record, every shared update level 2."""
        scene = Scene(name="tiny", **TINY)
        cfg = tight_explicit(partner_object="inferred", inertia=1.0, swap_on_place=True,
                             steal=True, goalless_hypotheses=True, value_maps=False,
                             belief_detail=True, belief_planning="updated", objective="task",
                             belief_horizon=4, belief_points=4, belief_refinements=0,
                             belief_partner_max=3, belief_inference="level2",
                             belief_planning_goals=("coercion", "hinder"))
        res = run_pairing(scene, PAIR, levels=(2, 2), cfg=cfg, n_steps=3)
        bp = res["belief_planner"]
        self.assertEqual(list(bp["robots"]), [0, 1])
        robots = sorted({d["robot"] for d in bp["decisions"]})
        self.assertEqual(robots, [0, 1])
        for row in res["trace"]:
            for r in (0, 1):
                self.assertEqual(row["belief_update"][r]["models"]["level"], 2)
        json.dumps(to_json(res))


if __name__ == "__main__":
    unittest.main()
