"""The decoy feint (Config.decoy_feint, social_pairings/feint.py, 2026-11-04): the path to the
decoy, the look-ahead through the partner's own belief update, and the end valued afresh.
"""
import unittest

import numpy as np

from social_pairings import goals as G
from social_pairings.constants import PICK
from social_pairings.feint import DecoyFeint
from social_pairings.tests.test_models import TINY, hostage, solved, tight_explicit
from social_pairings.util import configure_threads

PAIR = ("hinder", "coercion")      # R0 hinders, R1 coerces; R0 holds R1's object (hostage)


def standoff(**options):
    cfg = tight_explicit(partner_object="inferred", decoy_feint=True, **options)
    w, gid, mdp, state, _ = solved(TINY, cfg, (2, 2), PAIR, (1.0, 1.0), hostage, seed=5)
    args = (w, gid, (2, 2), tuple(G.parse_goal(g) for g in PAIR), (1.0, 1.0), cfg,
            configure_threads(cfg.num_threads))
    feint = DecoyFeint(mdp, args)
    s = w.sid(state)
    q = [mdp.q_values(mdp.root[i], s, i) for i in (0, 1)]
    status = [int(v) for v in w.status[int(state[2])]]
    return w, mdp, state, feint, q, status


class Feint(unittest.TestCase):
    def test_who_feints(self):
        w, mdp, state, feint, q, status = standoff()
        self.assertEqual(feint.robots, (1,))          # the coercer, not the hinderer
        self.assertTrue(feint.free(w, status, 0, 1))  # R0's own object lies at its place
        self.assertFalse(feint.free(w, status, 1, 1))  # R1's is in R0's hands

    def test_the_first_move(self):
        w, mdp, state, feint, q, status = standoff()
        nb = w.nb_by[1]
        start, goal = int(state[1]), feint.cell_of(w, status, state, [None, None], 0)
        a, to_go = feint.first_move(w, 1, start, goal, int(state[0]), q[1])
        self.assertIsNotNone(a)
        self.assertTrue(np.isfinite(q[1][a]))
        self.assertEqual(to_go, 1)                    # (2, 1) -> (2, 2), next to it
        self.assertEqual(int(nb[start, a]), goal)
        self.assertEqual(feint.first_move(w, 1, goal, goal, int(state[0]), q[1]), (None, 0))

    def test_the_look_ahead(self):
        w, mdp, state, feint, q, status = standoff()
        before = list(mdp.about)
        rec = feint.evaluate(mdp, 0, state, status, [None, None], 0, 1, q[1])
        self.assertIsNotNone(rec)
        self.assertTrue(all(a is b for a, b in zip(mdp.about, before)))   # beliefs restored
        moves = [a for a, _ in rec["steps"]]
        self.assertEqual(moves.count(PICK), w.pick_steps if hasattr(w, "pick_steps")
                         else mdp.cfg.pick_steps)
        self.assertEqual(rec["transitions"], len(moves))
        for key in ("p_now", "p_end"):
            self.assertTrue(0.0 <= rec[key] <= 1.0)
        for key in ("v_now", "option_return", "v_end", "q_option", "gain"):
            self.assertTrue(np.isfinite(rec[key]), key)
        self.assertAlmostEqual(rec["gain"], rec["q_option"] - rec["v_now"], places=9)
        self.assertEqual(feint.log[-1], rec)

    def test_the_end_re_solved_where_the_belief_moves(self):
        w, mdp, state, feint, q, status = standoff(feint_min_shift=0.0)
        rec = feint.evaluate(mdp, 0, state, status, [None, None], 0, 1, q[1])
        self.assertTrue(rec["end_solved"])
        self.assertIsNotNone(feint.evaluator)

    def test_the_evaluator_is_the_model(self):
        """Given the partner's belief as it is, the second model's value at the state is the
        first's: the end of an option is valued by the same models, re-solved."""
        w, mdp, state, feint, q, status = standoff()
        v = feint.end_value(mdp, state, mdp.about[1], 1)
        finite = np.isfinite(q[1])
        self.assertAlmostEqual(v, float(q[1][finite].max()), places=5)


if __name__ == "__main__":
    unittest.main()
