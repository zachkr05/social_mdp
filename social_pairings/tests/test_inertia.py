"""Inertia (``Config.inertia``, 2026-10-14; the user: keep the previous action unless another
beats it by about one move cost): a robot does not undo its last action -- step back to the
cell it came from, or undo what its hands just did (place what it picked up, pick up what it
placed, give back what it took, take back what it gave, swap back) -- unless undoing beats
every other action by more than ``inertia`` x c(move).

Checked: the rule on Q values of the runs of 2026-10-14 (the hinderer's put-back at
transition 5 of none / hinder, a dodge), its edge cases and that with nothing to undo or a
margin of 0 it is Algorithm 1's argmax; what undoes what; the option's bookkeeping; and two
rollouts, greedy and sampled, against the rule's invariants (no undoing within the margin,
the flags and the ledger agree, and without inertia the rollout of before).
"""
import unittest

import numpy as np

from social_pairings import Scene
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.constants import GIVE, N_ACTIONS, PICK, PLACE, SWAP, TAKE, WAIT
from social_pairings.util import OPPOSITE_MOVE, argmax_action, choose_with_inertia, undo_of

from .hands_before import Config, run_pairing
from .test_models import TINY

NEG = -np.inf


class Rule(unittest.TestCase):
    def test_what_undoes_what(self):
        for a in range(4):
            self.assertEqual(undo_of(a, True, False), OPPOSITE_MOVE[a])
            self.assertEqual(OPPOSITE_MOVE[OPPOSITE_MOVE[a]], a)
            self.assertIsNone(undo_of(a, False, False))          # a blocked move
        self.assertEqual(OPPOSITE_MOVE, (2, 3, 0, 1))             # +y <-> -y, +x <-> -x
        self.assertIsNone(undo_of(WAIT, False, False))
        for a, b in ((PICK, PLACE), (PLACE, PICK), (TAKE, GIVE), (GIVE, TAKE), (SWAP, SWAP)):
            self.assertEqual(undo_of(a, False, True), b)          # picked up: placing it ...
            self.assertIsNone(undo_of(a, False, False))           # a hand action that did nothing

    def test_the_put_back_of_none_hinder(self):
        """Transition 5 of none / hinder (2026-10-14): the hinderer has just picked up its
        partner's object; putting it back beat waiting by 0.051."""
        q = np.array([-32.797, NEG, -32.783, NEG, -31.893, NEG, -31.842, NEG, NEG, NEG])
        self.assertEqual(argmax_action(q), PLACE)
        self.assertEqual(choose_with_inertia(q, PLACE, 0.5), (WAIT, True))
        self.assertEqual(choose_with_inertia(q, PLACE, 0.04), (PLACE, False))   # below the gap
        self.assertEqual(choose_with_inertia(q, None, 0.5), (PLACE, False))     # nothing to undo
        self.assertEqual(choose_with_inertia(q, PLACE, 0.0), (PLACE, False))

    def test_a_dodge_is_kept(self):
        """Transition 22 of none / hinder: stepping away from the partner is worth about 10
        more than standing still.  Right after the opposite step it still steps away: by the
        other way out (0.002 worse) where there is one, and back where there is not."""
        q = np.array([NEG, -11.380, -11.378, -22.202, -21.752, NEG, NEG, NEG, -22.425, NEG])
        self.assertEqual(choose_with_inertia(q, 2, 0.5), (1, True))          # the other way
        self.assertEqual(choose_with_inertia(q, 1, 0.5), (2, False))        # not the undo
        q[1] = NEG                                                           # no other way
        self.assertEqual(choose_with_inertia(q, 2, 0.5), (2, False))

    def test_edges(self):
        q = np.array([NEG, NEG, NEG, NEG, NEG, NEG, 1.0, NEG, NEG, NEG])
        self.assertEqual(choose_with_inertia(q, PLACE, 0.5), (PLACE, False))  # all it can do
        q = np.array([1.0, 0.75, NEG, NEG, 0.25, NEG, NEG, NEG, NEG, NEG])
        self.assertEqual(choose_with_inertia(q, 0, 0.25), (1, True))         # exactly the margin
        self.assertEqual(choose_with_inertia(q, 0, 0.24), (0, False))
        rng = np.random.default_rng(0)
        for _ in range(500):                                   # margin 0: the argmax itself
            q = rng.normal(size=N_ACTIONS)
            q[rng.random(N_ACTIONS) < 0.3] = NEG
            if not np.isfinite(q).any():
                continue
            q[rng.integers(N_ACTIONS)] = q[np.isfinite(q)].max()       # ties
            for undo in (None, 0, 1, 2, 3, PICK, PLACE, TAKE, GIVE, SWAP):
                self.assertEqual(choose_with_inertia(q, undo, 0.0), (argmax_action(q), False))

    def test_the_option(self):
        self.assertEqual(Config().inertia, 0.0)
        self.assertEqual(ADDED_FIELDS["inertia"], 0.0)
        self.assertEqual(written_with({})["inertia"], 0.0)          # records of before
        self.assertAlmostEqual(Config(inertia=1.0).inertia_margin, 0.5)       # 0.5 m cells
        self.assertAlmostEqual(Config(inertia=1.0, resolution=1.0).inertia_margin, 1.0)
        for bad in (-0.1, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                Config(inertia=bad)


class Rollouts(unittest.TestCase):
    scene = Scene(name="tiny", **TINY)

    def run_it(self, **options):
        cfg = Config(num_threads=2, resolution=1.0, **options)
        return cfg, run_pairing(self.scene, ("none", "hinder"), levels=(2, 2), cfg=cfg,
                                n_steps=14)

    def check(self, cfg, r):
        """The rule's invariants along a rollout."""
        margin, flags = cfg.inertia_margin, 0
        rows = r["trace"]
        for prev, row in zip([None] + rows[:-1], rows):
            for i in (0, 1):
                q = np.array([np.nan if v is None else v for v in row["q"][i]], dtype=float)
                q[~np.isfinite(q)] = -np.inf
                a = int(row["action"][i])
                held = bool(row["inertia"][i])
                flags += held
                undo = None
                if prev is not None:
                    moved = tuple(prev["next_pos"][i]) != tuple(prev["pos"][i])
                    changed = prev["holds"][i] != prev["next_holds"][i]
                    undo = undo_of(int(prev["action"][i]), moved, changed)
                if margin <= 0.0 or undo is None:
                    self.assertFalse(held)
                    continue
                rest = q.copy()
                rest[undo] = -np.inf
                if cfg.action_selection == "argmax":
                    if a == undo and np.isfinite(rest).any():          # undone: worth it
                        self.assertGreater(q[undo] - rest.max(), margin - 1e-9)
                    if held:
                        self.assertEqual(argmax_action(q), undo)
                        self.assertEqual(a, argmax_action(rest))
                        self.assertLessEqual(q[undo] - q[a], margin + 1e-9)
                    else:
                        self.assertEqual(a, choose_with_inertia(q, undo, margin)[0])
                else:
                    self.assertEqual(held, argmax_action(q) == undo and a != undo)
        self.assertEqual(flags, sum(led["inertia"] for led in r["ledger"]))
        return flags

    def test_without_inertia_the_rollout_of_before(self):
        cfg, r = self.run_it()
        self.assertEqual(self.check(cfg, r), 0)
        cfg0, r0 = self.run_it(inertia=0.0)
        self.assertEqual([row["action"] for row in r["trace"]],
                         [row["action"] for row in r0["trace"]])

    def test_greedy(self):
        """On this map the hinderer picks its partner's object up and puts it back at once;
        with inertia it waits in between (twice), and holds it from transition 7 on.  A
        large margin: no undoing at all but where it is the only legal action."""
        cfg, r = self.run_it(inertia=1.0)
        self.assertGreater(self.check(cfg, r), 0)
        cfg, r = self.run_it(inertia=100.0)
        self.assertGreater(self.check(cfg, r), 0)

    def test_sampled(self):
        cfg, r = self.run_it(inertia=1.0, action_selection="sample", seed=3)
        self.assertGreater(self.check(cfg, r), 0)


if __name__ == "__main__":
    unittest.main()
