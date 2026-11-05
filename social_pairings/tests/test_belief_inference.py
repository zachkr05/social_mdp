"""The coercer reading its partner at level 2 (Config.belief_inference="level2", 2026-11-05
evening: a level-3 agent): the option; the override of the hypothesis policies in Eqs. 5-6;
the level-2 policy of the true hypothesis is the partner's root as the rollout solved it; in
a rollout the partner's action is the true hypothesis' argmax at every transition (it is
never penalised) and the record says what was solved.
"""
import json
import unittest

import numpy as np

from social_pairings import goals as G
from social_pairings.config import ADDED_FIELDS, Config
from social_pairings.constants import N_ACTIONS
from social_pairings.pairing import run_pairing
from social_pairings.scenes import Scene
from social_pairings.serialize import to_json
from social_pairings.tests.test_belief_plan import PAIR, planner
from social_pairings.tests.test_models import TINY, tight_explicit
from social_pairings.util import log_softmax


class Options(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(Config().belief_inference, "level1")
        self.assertEqual(ADDED_FIELDS["belief_inference"], "level1")
        with self.assertRaises(ValueError):
            Config(belief_inference="level3")
        with self.assertRaises(ValueError):                 # needs the belief planner
            Config(belief_inference="level2")
        with self.assertRaises(ValueError):                 # ... and Eq. 6 as printed
            Config(belief_inference="level2", belief_planning="updated",
                   physical_likelihood="level0")
        Config(belief_inference="level2", belief_planning="updated")


class Override(unittest.TestCase):
    def test_the_override_is_read(self):
        """With an override set, Eqs. 5-6 read the given policies (lapse applied), and the
        record says level 2."""
        w, mdp, st, pl, ok, lo, hi = planner()
        p, r = 0, 1
        s = w.sid(st)
        T = len(mdp.hyp_goals[r])
        rng = np.random.default_rng(0)
        q = rng.normal(size=(mdp.PG, T, N_ACTIONS))
        lp = np.stack([[log_softmax(q[g, t]) for t in range(T)] for g in range(mdp.PG)])
        mdp.likelihood_override = {p: (q, lp)}
        detail = {}
        a = 4
        out = mdp.hypothesis_log_policies(p, s, a, mdp.about[p].log_phys, detail)
        np.testing.assert_allclose(out, lp[:, :, a])
        self.assertEqual(detail["models"]["level"], 2)
        mdp.likelihood_override = {}
        out1 = mdp.hypothesis_log_policies(p, s, a, mdp.about[p].log_phys)
        self.assertFalse(np.allclose(out, out1))


class Level2(unittest.TestCase):
    def test_the_true_hypothesis_is_the_root_as_solved(self):
        """The level-2 policy of the partner under the hypothesis of its true goals, at the
        belief it holds: log softmax(Q / tau) of its root as the rollout solved it."""
        w, mdp, st, pl, ok, lo, hi = planner(belief_inference="level2")
        p, r = 0, 1
        s = w.sid(st)
        q_all, lp_all = pl.log_policies(mdp, r, s)
        g_star, t_star = int(mdp.own_goal[p]), list(mdp.hyp_goals[r]).index(int(mdp.goals[p]))
        q_root = mdp.q_values(mdp.root[p], s, p)
        fin = np.isfinite(q_root)
        np.testing.assert_array_equal(fin, np.isfinite(q_all[g_star, t_star]))
        np.testing.assert_allclose(q_all[g_star, t_star][fin], q_root[fin], atol=2e-3, rtol=0)
        want = log_softmax(q_root / mdp.inference_taus[r])
        np.testing.assert_allclose(lp_all[g_star, t_star][fin], want[fin], atol=5e-2, rtol=0)
        # every hypothesis the posteriors hold possible was solved
        held = np.isfinite(mdp.about[p].log_phys)[:, None] & np.isfinite(mdp.about[p].log_social)[None, :]
        np.testing.assert_array_equal(np.isfinite(lp_all).any(axis=-1), held)
        self.assertEqual(pl.inference_log[-1]["hypotheses"], int(held.sum()))

    def test_tiny_rollout_level2(self):
        """A rollout with the level-2 read: the partner's action is the argmax of the true
        hypothesis' level-2 policy at every transition (the partner acts by that model,
        greedily; no inertia), the override is cleared after each update, the record holds
        the solves."""
        scene = Scene(name="tiny", **TINY)
        cfg = tight_explicit(partner_object="inferred", inertia=0.0, swap_on_place=True,
                             steal=True, belief_planning="updated", objective="task",
                             belief_horizon=4, belief_points=4, belief_refinements=0,
                             belief_partner_max=2, belief_inference="level2",
                             value_maps=False, belief_detail=True)
        res = run_pairing(scene, PAIR, levels=(2, 2), cfg=cfg, n_steps=4)
        bp = res["belief_planner"]
        self.assertEqual(bp["inference"], "level2")
        self.assertEqual(len(bp["inference_log"]), len(res["trace"]))
        p, r = 0, 1
        for row, log in zip(res["trace"], bp["inference_log"]):
            self.assertGreaterEqual(log["solves"], 0)
            upd = row["belief_update"][r]                 # robot r's update about its partner
            self.assertEqual(upd["models"]["level"], 2)
            a_p = row["action"][p]
            q = np.asarray(upd["models"]["q"], dtype=float)   # (PG, T, 10)
            social = upd["hypotheses"]["social"]
            t_star = social.index(G.goal_name(G.HINDER))
            # under the true hypothesis (the partner's own goal, its social goal) its action
            # is the model's argmax
            for g in range(q.shape[0]):
                if (int(upd["hypotheses"]["object"][g]) == p
                        and not np.isnan(q[g, t_star]).all()):
                    qt = np.where(np.isnan(q[g, t_star]), -np.inf, q[g, t_star])
                    if np.isclose(qt[a_p], qt.max(), atol=5e-3):   # (re-solved: tolerance)
                        break
            else:
                self.fail(f"the partner's action {a_p} is no true hypothesis' argmax")
        json.dumps(to_json(res))


if __name__ == "__main__":
    unittest.main()
