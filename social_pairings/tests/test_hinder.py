"""Does a hinderer hinder?  The regression test of the bug this package was written to fix.

On the six-object scene a ``hinder`` robot must take its partner's object within its first
transitions and never let go of it -- at small and at large weights, against a cooperative,
a competitive and a plainly physical partner -- so that the partner never finishes, without
picking things up and putting them back, pacing, or playing with the spare objects.
"""
import unittest

import numpy as np

from social_pairings import six_object_scene
from social_pairings import goals as G
from social_pairings.constants import N_ACTIONS, PICK
from social_pairings.hindrance import hindrance_report

from .told import Config, default_levels, levels_used, run_baseline, run_pairing

SCENE = six_object_scene()
WEIGHTS = (0.8, 1.0, 2.0, 30.0)
STEPS = 40


class HinderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(num_threads=2)
        cls.baseline = run_baseline(SCENE, cfg=cls.cfg, n_steps=STEPS)

    def hinder(self, partner_goal, weight, **options):
        cfg = Config(num_threads=2, **options)
        result = run_pairing(SCENE, ("hinder", partner_goal), weights=(weight, weight),
                             cfg=cfg, n_steps=STEPS)
        return result, hindrance_report(result, self.baseline, hinderer=0)

    def assert_hinders(self, partner_goal, weight):
        result, report = self.hinder(partner_goal, weight)
        self.assertTrue(result["numerically_accepted"])
        # effect: the partner finished alone at transition 11 and now never does
        self.assertEqual(report["baseline_done_at"], 11)
        self.assertIsNone(result["done_at"][1])
        self.assertTrue(report["prevented"] and report["hindered"])
        self.assertEqual(report["return_denied"], 1.0)
        # conduct: takes the object at once, keeps it to the end, and does nothing else odd
        self.assertEqual(report["mechanism"], "holds the partner's object")
        self.assertTrue(report["purposeful"])
        self.assertLessEqual(report["took_object_at"], 3)
        self.assertEqual(report["releases"], 0)
        self.assertEqual(result["holds"][0], 1)
        taken = report["took_object_at"]
        self.assertTrue(all(row["next_holds"][0] == 1 for row in result["trace"][taken - 1:]),
                        "the object left the hinderer's hands")
        conduct = report["conduct"][0]
        self.assertEqual((conduct["put_back_cycles"], conduct["spare_pickups"],
                          conduct["reversals"]), (0, 0, 0))
        return result, report

    def test_baseline_both_robots_finish(self):
        self.assertEqual(tuple(self.baseline["done_at"]), (9, 11))
        self.assertEqual(self.baseline["stopped_by"], "done")

    def test_against_a_cooperative_partner(self):
        for weight in WEIGHTS:
            with self.subTest(weight=weight):
                result, _ = self.assert_hinders("cooperation", weight)
                # the cooperator still delivers the hinderer's object for it
                self.assertIsNotNone(result["done_at"][0])
                self.assertEqual(result["ledger"][0]["delivered_by"], 1)

    def test_against_a_competitive_partner(self):
        for weight in WEIGHTS:
            with self.subTest(weight=weight):
                result, report = self.assert_hinders("competition", weight)
                self.assertEqual(tuple(result["done_at"]), (None, None))
                # the competitor's own goal is never done, so under greedy actions it does
                # not take the hinderer's object in turn; and it does not pace either
                self.assertEqual(tuple(result["holds"]), (1, -1))
                conduct = report["conduct"][1]
                self.assertEqual((conduct["put_back_cycles"], conduct["reversals"]), (0, 0))

    def test_against_a_competitor_as_table_one_prints_it(self):
        """``competition="table"``: the competitor answers in kind and holds the hinderer's
        object in turn."""
        for weight in (1.0, 30.0):
            with self.subTest(weight=weight):
                result, report = self.hinder("competition", weight, competition="table")
                self.assertTrue(report["prevented"])
                self.assertEqual(tuple(result["levels_used"]), (1, 2))
                self.assertEqual(tuple(result["holds"]), (1, 0))

    def test_against_a_partner_without_social_goal(self):
        for weight in WEIGHTS:
            with self.subTest(weight=weight):
                self.assert_hinders("none", weight)

    def test_a_hinderer_reasons_at_level_1(self):
        """Its term is Table I's conflict entry for l = 1; its level-2 partner models it
        exactly."""
        self.assertEqual(levels_used((G.HINDER, G.CONFLICT), (2, 2)), (1, 2))
        self.assertEqual(levels_used((G.COOPERATION, G.HINDER), (2, 2)), (2, 1))
        self.assertEqual(levels_used((G.HINDER, G.NONE), (0, 2)), (0, 2))
        # with weight 0 it has no social goal: it reasons at the level asked for
        self.assertEqual(levels_used((G.HINDER, G.COOPERATION), (2, 2), weights=(0.0, 1.0)),
                         (2, 2))
        self.assertEqual(levels_used((G.HINDER, G.COOPERATION), (2, 2), weights=(0.1, 0.0)),
                         (1, 2))
        # ... and level 1 is its level when none is asked for
        self.assertEqual(default_levels((G.HINDER, G.CONFLICT)), (1, 2))
        self.assertEqual(default_levels((G.COOPERATION, G.HINDER)), (2, 1))
        self.assertEqual(default_levels((G.HINDER, G.COOPERATION), weights=(0.0, 1.0)), (2, 2))
        result, _ = self.hinder("cooperation", 1.0)
        self.assertEqual(tuple(result["levels"]), (1, 2))
        self.assertEqual(tuple(result["levels_used"]), (1, 2))
        self.assertIn("-w r(g_j)", result["social_terms"][0])

    def test_taking_the_object_is_what_the_social_term_pays_for(self):
        """In the transition in which it picks the object up, the social part of the
        hinderer's Q (Q minus the Q of its twin without a social term) favours "pick" over
        every other action, although physically "pick" is a detour; and from then on its
        social reward is exactly zero: the partner's goal reward is 0 while its object is
        held."""
        result, report = self.hinder("none", 2.0)
        taken = report["took_object_at"]
        row = result["trace"][taken - 1]
        self.assertEqual(row["action"][0], PICK)
        q, q_physical = np.asarray(row["q"][0]), np.asarray(row["q_physical"][0])
        legal = np.isfinite(q)
        social = np.full(N_ACTIONS, -np.inf)
        social[legal] = q[legal] - q_physical[legal]
        self.assertEqual(int(np.argmax(social)), PICK)
        self.assertLess(q_physical[PICK], np.max(q_physical[legal]))
        for row in result["trace"][taken:]:
            self.assertAlmostEqual(float(row["social_reward"][0][row["action"][0]]), 0.0,
                                   places=6)

    def test_with_forced_swaps_nothing_can_be_held(self):
        """The older rule (a robot handed its own object must take it and give up what it
        holds): a partner that takes the hinderer's object in turn -- the competitor of
        Table I as printed -- gets its own back with one "give", so the same hinderer no
        longer hinders with purpose.  This is the first of the reasons the previous
        implementation looked erratic."""
        result, report = self.hinder("competition", 30.0, forced_swap=True,
                                     competition="table")
        lost = report["releases"] > 0 or any(
            what == "handed over" and obj == 1 for _, what, _, obj, _ in result["events"])
        self.assertTrue(lost or not report["purposeful"])


if __name__ == "__main__":
    unittest.main()
