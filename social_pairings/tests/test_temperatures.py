"""A tau per robot (``Config.robot_temperatures``, 2026-11-01).

Robot i reads its partner at its own tau: the Eq. 7 policies it predicts the partner by and
the likelihood of its belief update about the partner.  So every model of robot o is
softmaxed at its partner's tau -- also inside robot o's own level-2 model, where the
partner's level-1 models read robot o -- and one tau for both is the rule of before.

Checked here: the option itself (two equal taus are the one-tau record); the level-0
policies and the physical likelihood of each robot against the one-tau engine at its
partner's tau (a level-0 model's values do not depend on tau, so they must match exactly);
every array of levels 0-2 against the brute-force reference (``reference.Brute``, which
softmaxes a model of robot o at ``tau_of(1 - o)``) with two different taus, at the user's
pairing (competition against coercion); and a rollout's record.
"""
import dataclasses
import unittest

import numpy as np

from social_pairings import Config
from social_pairings.config import written_with
from social_pairings.models import SocialMDP
from social_pairings.serialize import to_json
from social_pairings import goals as G
from social_pairings.util import configure_threads

from .reference import make_world
from .test_models import TINY, tight_now

TAUS = (0.1, 0.6)                       # agent_1's, agent_2's: far apart on purpose


def level0(cfg, goals=("competition", "coercion")):
    """The engine's level-0 models of a pairing on the tiny map, solved."""
    w, goal_ids = make_world(TINY, cfg)
    mdp = SocialMDP(w, goal_ids, (2, 2), tuple(G.parse_goal(g) for g in goals), (1.0, 1.0),
                    cfg, configure_threads(cfg.num_threads))
    mdp.allocate()
    mdp.solve_level0()
    return w, mdp


class Option(unittest.TestCase):
    def test_two_equal_taus_are_one(self):
        self.assertEqual(Config(robot_temperatures=(0.3, 0.3)), Config(temperature=0.3))
        cfg = Config(robot_temperatures=[0.1, 0.5])
        self.assertEqual(cfg.robot_temperatures, (0.1, 0.5))
        self.assertEqual((cfg.temperature_of(0), cfg.temperature_of(1)), (0.1, 0.5))
        self.assertAlmostEqual(cfg.tau_of(1) / cfg.tau_of(0), 5.0)
        self.assertAlmostEqual(cfg.tau_of(0), 0.1 * float(cfg.costs[0]))
        with self.assertRaises(ValueError):       # one tau asked of two robots with two
            _ = cfg.tau
        self.assertEqual(Config().tau, Config().tau_of(1))

    def test_bad_values(self):
        for bad in ((0.1,), (0.1, 0.2, 0.3), (0.1, 0.0), (0.1, -1.0), (0.1, float("inf")), "ab"):
            with self.assertRaises(ValueError):
                Config(robot_temperatures=bad)

    def test_records_of_before_had_one_tau(self):
        self.assertIsNone(written_with({})["robot_temperatures"])
        mine = to_json(dataclasses.asdict(Config(robot_temperatures=TAUS)))
        self.assertEqual(mine["robot_temperatures"], list(TAUS))


class AgainstOneTau(unittest.TestCase):
    """Robot o's models are read at its partner's tau: with taus (a, b), robot 0's level-0
    policies and the likelihood of its actions are those of the one-tau engine at b, robot
    1's at a."""

    @classmethod
    def setUpClass(cls):
        cls.w, cls.both = level0(tight_now(robot_temperatures=TAUS))
        cls.one = [level0(tight_now(temperature=t))[1] for t in TAUS]

    def test_level_0_policies(self):
        K = self.both.PG
        for o in (0, 1):
            rows = slice(o * K, (o + 1) * K)
            read_by = self.one[1 - o]                # robot o is read at its partner's tau
            self.assertTrue(np.array_equal(self.both.L0.psi[rows], read_by.L0.psi[rows]))
            other = self.one[o]
            self.assertGreater(float(np.abs(self.both.L0.psi[rows] - other.L0.psi[rows]).max()),
                               1e-2, "the two taus should give different policies")

    def test_the_likelihood_of_the_physical_posterior(self):
        w = self.w
        s = w.sid(w.initial)
        for j in (0, 1):
            for a in range(4):
                got = self.both.physical_log_likelihood(j, s, a)
                want = self.one[1 - j].physical_log_likelihood(j, s, a)
                self.assertTrue(np.allclose(got, want, equal_nan=True))

    def test_a_sampled_robot_draws_at_its_own_tau(self):
        q = np.array([0.0, -0.5, -1.0, -np.inf, 0.2, -2.0, -np.inf, -np.inf, -np.inf, -np.inf])
        for i in (0, 1):
            got = self.both.execution_policy(i, self.w.initial, q)
            want = self.one[i].execution_policy(i, self.w.initial, q)
            self.assertTrue(np.allclose(got, want))


class AgainstBruteForce(unittest.TestCase):
    def test_competition_against_coercion_with_a_tau_each(self):
        """Every array of levels 0-2 against the reference, agent_1 a competitor reading its
        partner at 0.1 and agent_2 a coercer reading its partner at 0.6 (and the other way
        round), both at level 2, everything inferred."""
        from .test_models import ModelsAgainstBruteForce
        case = ModelsAgainstBruteForce(methodName="compare")
        for taus in (TAUS, TAUS[::-1]):
            case.compare(tight_now(robot_temperatures=taus), (2, 2),
                         ("competition", "coercion"), (1.0, 1.0), seed=211)


class Record(unittest.TestCase):
    def test_a_rollout_keeps_both_taus(self):
        import contextlib
        import io

        from social_pairings import run_pairing
        from social_pairings.scenes import Scene
        scene = Scene(name="tiny_taus", grid=TINY["grid"], starts=TINY["starts"],
                      pickups=TINY["pickups"], destinations=TINY["destinations"],
                      candidate_cells=TINY["candidate_cells"])
        cfg = Config(num_threads=2, resolution=1.0, robot_temperatures=TAUS, verbose=True)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            result = run_pairing(scene, ("competition", "coercion"), levels=(2, 2), cfg=cfg,
                                 n_steps=2)
        self.assertEqual(tuple(result["config"]["robot_temperatures"]), TAUS)
        self.assertIn(f"tau={cfg.tau_of(0):g}/{cfg.tau_of(1):g}", printed.getvalue(),
                      "the description names both taus")


if __name__ == "__main__":
    unittest.main()
