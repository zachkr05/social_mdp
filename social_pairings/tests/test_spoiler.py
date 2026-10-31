"""A hinderer without a goal of its own (``Config.hinder_own_goal=False``, the default since
2026-10-14; the user: "a hindering agent would just want to minimize the opposing agent's
task, they don't really care about their task").  Its reward is its moves' costs and
``-w r(g_j)``: in the terms of goals.py ``b = -1`` at level 1 and ``beta = -1`` above it, not
multiplied by the weight, which takes the ``r(g_i)`` of Eq. 2 away exactly.

Checked: the terms (levels 1 and 2, several weights) and the options' bookkeeping; the
engine against the brute force of ``tests/reference.py`` (which states the spoiler's term
itself) with the hinderer at level 1 -- alone against a robot without social goal, and
modelled by a level-2 cooperator that knows it, whose ``+R_j`` holds the spoiler's reward;
the twin a record compares it with (its moves' costs, nothing else); and a rollout's record
(no goal of its own: never done, and the run is over when its partner's goal is).  The
level-2 spoiler with every goal inferred is compared with the brute force in
``test_models.py`` (``coercion`` / ``hinder`` with swaps and taking), whose harness reads
``Config.hinder_own_goal``.
"""
import unittest

import numpy as np

from social_pairings import Scene
from social_pairings import goals as G
from social_pairings.config import ADDED_FIELDS, TOLD, written_with
from social_pairings.constants import WAIT
from social_pairings.models import NO_GOAL_TWIN, SPOILER_TWIN

from . import test_models as TM
from .reference import level1_term, states_of, table_l2
from .hands_before import Config, run_pairing
from .test_models import TINY, hostage, solved, tight, tight_now


class Terms(unittest.TestCase):
    def test_the_terms(self):
        for w in (0.5, 1.0, 3.0, 30.0):
            self.assertEqual(G.level1_term(G.HINDER, weight=w, hinder_own_goal=False),
                             (-w, -1.0, 0.0))
            self.assertEqual(G.level1_term(G.HINDER, weight=w, hinder_own_goal=True),
                             (-w, 0.0, 0.0))
            for t in range(len(G.SOCIAL_GOALS)):
                for same in (False, True):
                    self.assertEqual(G.level2_term(G.HINDER, t, same, weight=w,
                                                   hinder_own_goal=False),
                                     (0.0, -1.0, -w, 0.0))
                    self.assertEqual(G.level2_term(G.HINDER, t, same, weight=w),
                                     (0.0, 0.0, -w, 0.0))
            # the reference states the same, apart from the engine
            self.assertEqual(level1_term("hinder", w, spoiler=True), (-w, -1.0, 0.0))
            self.assertEqual(table_l2("hinder", "cooperation", False, 0.0, w, spoiler=True),
                             (0.0, -1.0, -w))
        # nothing else changes with the option
        for goal in range(len(G.SOCIAL_GOALS)):
            if goal == G.HINDER:
                continue
            self.assertEqual(G.level1_term(goal, weight=2.0, hinder_own_goal=False),
                             G.level1_term(goal, weight=2.0))
            for t in range(len(G.SOCIAL_GOALS)):
                self.assertEqual(G.level2_term(goal, t, False, weight=2.0, hinder_own_goal=False),
                                 G.level2_term(goal, t, False, weight=2.0))
        self.assertIn("no goal of its own", G.describe(G.HINDER, 2, hinder_own_goal=False))

    def test_the_option(self):
        self.assertFalse(Config().hinder_own_goal)
        self.assertTrue(Config(**TOLD).hinder_own_goal)
        self.assertTrue(ADDED_FIELDS["hinder_own_goal"])
        self.assertTrue(written_with({})["hinder_own_goal"])      # records of before
        with self.assertRaises(ValueError):
            Config(hinder_own_goal="no")


class AgainstBruteForce(unittest.TestCase):
    """The brute force of test_models.py, with the hinderer a spoiler."""

    def setUp(self):
        self.bf = TM.ModelsAgainstBruteForce("test_hinder_against_a_partner_without_social_goal")

    def test_spoiler_against_a_partner_without_social_goal(self):
        n1, n2 = self.bf.compare(tight(hinder_own_goal=False), (1, 0), ("hinder", "none"),
                                 (30.0, 1.0), state=hostage, seed=2)
        self.assertEqual((n1, n2), (2, 0))        # its root and its twin (the moves' costs)

    def test_spoiler_modelled_by_a_level_2_cooperator(self):
        self.bf.compare(tight(hinder_own_goal=False), (1, 2), ("hinder", "cooperation"),
                        (3.0, 2.0), state=hostage, seed=1)

    def test_spoiler_at_weight_one_against_a_level_2_competitor(self):
        self.bf.compare(tight(hinder_own_goal=False, competition="table"), (1, 2),
                        ("hinder", "competition"), (1.0, 1.5))


class Twin(unittest.TestCase):
    def test_the_twin_pays_for_its_moves_only(self):
        """A record splits a robot's Q into what its twin -- the same robot without its
        social term -- makes of each action and the rest.  A spoiler without its social term
        has nothing to do: its twin's Q differ by the actions' costs only, at level 1 and at
        level 2."""
        for levels in ((1, 0), (2, 2)):
            cfg = tight_now(hinder_own_goal=False) if levels == (2, 2) else \
                tight(hinder_own_goal=False)
            w, gid, mdp, state, _ = solved(TINY, cfg, levels, ("hinder", "none"), (1.0, 1.0),
                                           state=hostage, seed=4)
            self.assertEqual(mdp.spoiler, (True, False))
            depth, row = mdp.twin[0]
            bank = mdp.bank(depth)
            self.assertEqual(depth, levels[0])
            key = SPOILER_TWIN if depth == 1 else NO_GOAL_TWIN
            slot = [sl for (a, d, _), sl in mdp.slots.items() if a == 0 and d == depth][0]
            self.assertIn(key, slot.model_keys)
            rng = np.random.default_rng(0)
            every = states_of(w, mdp.solved_blocks(bank, row))     # the states it is solved on
            states = rng.choice(every, min(60, every.size), replace=False)
            for s in [w.sid(state)] + states.tolist():
                q = mdp.q_values((depth, row), int(s), 0)
                legal = np.isfinite(q)
                cost = w.cost_c[divmod(int(s), w.NN)[0], 0]
                want = -(cost - cost[WAIT])
                np.testing.assert_allclose((q - q[WAIT])[legal], want[legal], atol=1e-6)


class Record(unittest.TestCase):
    def test_a_rollout(self):
        """No goal of its own: the spoiler is never done, books no goal reward, and the run
        is over once its partner's goal is done (or stalls, or runs out of transitions)."""
        scene = Scene(name="tiny", **TINY)
        cfg = Config(num_threads=2, resolution=1.0)
        r = run_pairing(scene, ("none", "hinder"), levels=(2, 2), cfg=cfg, n_steps=14)
        self.assertEqual(tuple(r["own_goal"]), (True, False))
        self.assertIsNone(r["done_at"][1])
        self.assertEqual(r["ledger"][1]["physical_reward"], 0.0)
        self.assertIn("no goal of its own", r["social_terms"][1])
        self.assertEqual(r["completed"], r["done_at"][0] is not None)
        if r["done_at"][0] is not None:
            self.assertEqual(r["stopped_by"], "done")
        old = run_pairing(scene, ("none", "hinder"), levels=(2, 2),
                          cfg=Config(num_threads=2, resolution=1.0, hinder_own_goal=True),
                          n_steps=14)
        self.assertEqual(tuple(old["own_goal"]), (True, True))


if __name__ == "__main__":
    unittest.main()
