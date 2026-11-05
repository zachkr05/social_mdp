"""The decoy feint (Config.decoy_feint, social_pairings/feint.py, 2026-11-04): the path to the
decoy, the look-ahead through the partner's own belief update, and the end valued afresh.
"""
import unittest

import numpy as np

from social_pairings import goals as G
from social_pairings.constants import HELD0, HELD1, HOME, PICK, PLACE, SWAP, WAIT
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

    def test_guarded(self):
        """The robot's object lying free is guarded where the partner is no farther from it
        than the robot (round the partner)."""
        w, mdp, state, feint, q, status = standoff()
        b = w.block_of([HOME, HOME], [0, 0])
        free = [int(v) for v in w.status[b]]
        at = lambda c: int(w.cell_id[c])               # noqa: E731
        near = (at((0, 1)), at((2, 1)), b)             # R0 next to R1's object at (0, 0)
        far = (at((2, 0)), at((0, 1)), b)              # R1 next to it, R0 far
        self.assertTrue(feint.guarded(w, near, free, [None, None], 1))
        self.assertFalse(feint.guarded(w, far, free, [None, None], 1))
        self.assertFalse(feint.guarded(w, state, status, [None, None], 1))   # in R0's hands

    def test_the_lure_target(self):
        """The candidate destination farthest, for the partner, from the robot's route back
        (its object's cell and its destination), not the robot's own destination."""
        w, mdp, state, feint, q, status = standoff()
        at = lambda c: int(w.cell_id[c])               # noqa: E731
        b = w.block_of([HELD1, HOME], [0, 0])          # R1 has the decoy, its object at home
        st = (at((2, 2)), at((2, 1)), b)
        stat = [int(v) for v in w.status[b]]
        self.assertEqual(int(w.true_dest[1]), at((2, 0)))
        self.assertEqual(feint.lure_target(w, st, stat, [None, None], 1), at((0, 2)))

    def test_the_head_start(self):
        """> 0 where the partner, a transition late, reaches no cell of the robot's way back
        (to its object, the swap, to its destination) before the robot does."""
        w, mdp, state, feint, q, status = standoff()
        at = lambda c: int(w.cell_id[c])               # noqa: E731
        b = w.block_of([HELD1, HOME], [0, 0])
        stat = [int(v) for v in w.status[b]]
        far = (at((0, 2)), at((0, 1)), b)              # R1 next to its object, R0 behind it
        near = (at((2, 2)), at((0, 1)), b)             # R0 near R1's destination (2, 0)
        self.assertEqual([x for x, _ in feint.route(w, far, stat, [None, None], 1)],
                         [at((0, 0)), at((1, 0)), at((2, 0))])
        self.assertGreater(feint.head_start(w, far, stat, [None, None], 1), 0)
        self.assertLessEqual(feint.head_start(w, near, stat, [None, None], 1), 0)

    def test_the_phases(self):
        """Luring until the head start; then to its object and the swap on it; then
        delivering; luring again where the partner picks the object up meanwhile; the
        decoy lost ends it."""
        w, mdp, state, feint, q, status = standoff(swap_on_place=True)
        at = lambda c: int(w.cell_id[c])               # noqa: E731

        def follow(cells, objects, t=1):
            st = (at(cells[0]), at(cells[1]), w.block_of(objects, [0, 0]))
            stat = [int(v) for v in w.status[st[2]]]
            qq = [mdp.q_values(mdp.root[i], w.sid(st), i) for i in (0, 1)]
            return feint._decide(mdp, t, st, stat, [None, None], qq, 1, WAIT), qq

        feint.option[1] = dict(decoy=0, since=0, lures=0)
        feint._phase(0, 1, "luring", 50, lure=at((0, 2)))
        a, _ = follow(((2, 2), (0, 1)), [HELD1, HOME])     # R0 near its destination: lure on
        self.assertEqual(feint.option[1]["phase"], "luring")
        self.assertEqual(int(w.nb_by[1][at((0, 1)), a]), at((0, 2)))
        a, qq = follow(((0, 2), (0, 1)), [HELD1, HOME])    # R0 behind: turn back
        self.assertEqual(feint.option[1]["phase"], "to its object")
        self.assertEqual(int(w.nb_by[1][at((0, 1)), a]), at((0, 0)))
        a, qq = follow(((0, 2), (0, 0)), [HELD1, HOME])    # on it: the swap
        self.assertEqual(a, SWAP if np.isfinite(qq[1][SWAP]) else WAIT)
        a, qq = follow(((0, 2), (0, 0)), [HOME, HELD1])    # its object in hand: deliver
        self.assertEqual(feint.option[1]["phase"], "delivering")
        self.assertEqual(int(w.nb_by[1][at((0, 0)), a]), at((1, 0)))
        a, qq = follow(((0, 2), (2, 0)), [HOME, HELD1])    # on its destination: set it down
        self.assertEqual(a, PLACE if np.isfinite(qq[1][PLACE]) else WAIT)
        feint._phase(2, 1, "to its object", 50)            # the partner takes it back: lure
        follow(((0, 2), (1, 1)), [HELD1, HELD0])
        self.assertEqual(feint.option[1]["phase"], "luring")
        follow(((0, 2), (1, 1)), [HOME, HOME])             # the decoy lost
        self.assertIsNone(feint.option[1])
        self.assertEqual(feint.events[-1][2:], ("ended", "the decoy lost"))

    def test_waiting_for_the_way(self):
        """The partner on the next cell of the shortest way: wait feint_wait transitions,
        then go round."""
        w, mdp, state, feint, q, status = standoff()
        at = lambda c: int(w.cell_id[c])               # noqa: E731
        b = w.block_of([HOME, HOME], [0, 0])
        st = (at((1, 1)), at((2, 1)), b)               # R0 between R1 and (0, 1)
        qq = mdp.q_values(mdp.root[1], w.sid(st), 1)
        feint.option[1] = dict(decoy=0, since=0, lures=0)
        feint._phase(0, 1, "to its object", 50)
        acts = [feint._go(mdp, st, at((0, 1)), 1, qq) for _ in range(feint.cfg.feint_wait + 1)]
        self.assertEqual(acts[:-1], [WAIT] * feint.cfg.feint_wait)
        self.assertIn(int(w.nb_by[1][at((2, 1)), acts[-1]]), (at((2, 0)), at((2, 2))))

    def test_the_evaluator_is_the_model(self):
        """Given the partner's belief as it is, the second model's value at the state is the
        first's: the end of an option is valued by the same models, re-solved."""
        w, mdp, state, feint, q, status = standoff()
        v = feint.end_value(mdp, state, mdp.about[1], 1)
        finite = np.isfinite(q[1])
        self.assertAlmostEqual(v, float(q[1][finite].max()), places=5)


if __name__ == "__main__":
    unittest.main()
