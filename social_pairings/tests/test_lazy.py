"""Lazy hypotheses (2026-11-04): ``Config.lazy_mass``, ``lazy_period`` (models.SocialMDP.lazy_rows).

A level-2 robot's hypothesis models about its partner that carry little mass in its two
posteriors are not re-solved every transition.  Checked on the tiny scene: the option and
the records of before; which rows are skipped (their mass at most ``lazy_mass``, the
lightest first, every model solved at the first transition and at least every
``lazy_period`` transitions, the roots and twins always); that a skipped model keeps its
values and policy; that ``lazy_mass=0`` is the rule of before, transition for transition;
and that a rollout with the option records what it skipped and stays close to the exact
one.
"""
import unittest

import numpy as np

from social_pairings import Config, run_pairing
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.scenes import Scene

from .test_models import TINY, solved, tight_explicit


def transitions(mdp, w, n, record=None):
    """n transitions of the engine's loop (plan, inputs, solve, act greedily, observe);
    record(t, mdp) is called after each plan."""
    state = w.initial
    for t in range(n):
        mdp.plan(state)
        if record is not None:
            record(t, mdp)
        if t == 0:
            mdp.init_models()
        for depth in (1, 2):
            mdp.inputs(depth)
            mdp.solve(depth)
        s = w.sid(state)
        actions = tuple(int(np.argmax(mdp.q_values(mdp.root[i], s, i))) for i in (0, 1))
        nxt = w.unsid(w.step(s, actions))
        status = tuple(int(x) for x in w.status[state[2]])
        done = tuple(int(x) for x in w.done[state[2]])
        new_status, new_done, failed, handed, _ = w.real_step(status, done, state, nxt, actions,
                                                              detail=True)
        event = dict(done_now=tuple(bool(new_done[i]) and not done[i] for i in (0, 1)),
                     failed=failed, handed_in_vain=handed,
                     held=tuple(int(w.hold[state[2], i]) for i in (0, 1)))
        mdp.observe(actions, s, nxt, event)
        state = nxt
    return state


def fresh(cfg, seed=41):
    w, gid, mdp, _, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"), (1.0, 1.0),
                               seed=seed)
    mdp = type(mdp)(w, gid, (2, 2), mdp.goals, mdp.weights, cfg, 2)
    mdp.allocate()
    mdp.solve_level0()
    return w, mdp


class Options(unittest.TestCase):
    def test_defaults_and_records_of_before(self):
        self.assertEqual(Config().lazy_mass, 0.0)
        self.assertEqual(Config().lazy_period, 5)
        self.assertEqual(ADDED_FIELDS["lazy_mass"], 0.0)
        self.assertEqual(ADDED_FIELDS["lazy_period"], 5)
        self.assertEqual(written_with({})["lazy_mass"], 0.0)
        self.assertEqual(written_with({})["lazy_period"], 5)
        for bad in (dict(lazy_mass=1.0), dict(lazy_mass=-0.1), dict(lazy_mass=True),
                    dict(lazy_period=0), dict(lazy_period=2.5)):
            with self.assertRaises(ValueError):
                Config(**bad)
        Config(lazy_mass=0.5, lazy_period=1)


class Rows(unittest.TestCase):
    def test_what_is_skipped(self):
        cfg = tight_explicit(value_tol=1e-5, value_maps=False, lazy_mass=0.3, lazy_period=3)
        w, mdp = fresh(cfg)
        L1 = mdp.L1
        hyp_rows = set(np.flatnonzero(L1.psi_row >= 0).tolist())
        seen = []

        def record(t, mdp):
            solved_now = set(np.flatnonzero(L1.active).tolist())
            seen.append((t, solved_now, mdp.lazy_info, [b.copy() for b in mdp.about]))

        transitions(mdp, w, 7, record)
        # the first transition solves every hypothesis model
        self.assertTrue(hyp_rows <= seen[0][1])
        self.assertEqual(seen[0][2][0][0] + seen[0][2][1][0], 0)
        skipped_somewhere = False
        for t, solved_now, info, about in seen:
            # when each hypothesis model was last solved before this plan
            prev = {r: max((tt for tt, solved_then, _, _ in seen[:t] if r in solved_then),
                           default=-1) for r in hyp_rows}
            due = {r for r in hyp_rows if prev[r] < 0 or t - prev[r] >= cfg.lazy_period}
            # the mass of what was skipped, per robot, is at most lazy_mass; the lightest
            # models go first; a model that is due is never skipped
            for a in (0, 1):
                phys, social = about[a].phys(), about[a].social()
                hyp = mdp.hyp_goals[1 - a]
                weight = {}
                for k in range(mdp.PG):
                    if not np.isfinite(about[a].log_phys[k]):
                        continue
                    for tt in range(len(hyp)):
                        r = mdp.hypothesis_row(a, k, tt)
                        weight[r] = weight.get(r, 0.0) + phys[k] * social[tt]
                skipped = [r for r in weight if r not in solved_now]
                kept = [r for r in weight if r in solved_now]
                mass = sum(weight[r] for r in skipped)
                self.assertLessEqual(mass, cfg.lazy_mass + 1e-12)
                self.assertEqual(len(skipped), info[a][0])
                self.assertAlmostEqual(mass, info[a][1], places=12)
                self.assertEqual(len(kept), info[a][2])
                for r in skipped:
                    self.assertNotIn(r, due, (t, r))
                if skipped:
                    light = max(weight[r] for r in skipped)
                    for r in kept:
                        if weight[r] < light:
                            self.assertIn(r, due, (t, r))
                skipped_somewhere |= bool(skipped)
            # every model is solved at least every lazy_period transitions
            for r in hyp_rows:
                last = max((tt for tt, solved_then, _, _ in seen[:t + 1] if r in solved_then),
                           default=-1)
                self.assertGreaterEqual(last, t - cfg.lazy_period + 1, (t, r))
        self.assertTrue(skipped_somewhere, "nothing was ever skipped at lazy_mass 0.3")
        # the roots and twins (depth 2) are solved every transition
        self.assertTrue(mdp.L2.active.any())

    def test_a_skipped_model_keeps_its_values_and_policy(self):
        cfg = tight_explicit(value_tol=1e-5, value_maps=False, lazy_mass=0.3, lazy_period=4)
        w, mdp = fresh(cfg)
        L1 = mdp.L1
        before = {}
        checked = 0

        def record(t, mdp):
            # at plan t: the values are those of solve t - 1's end, and L1.active the rows
            # that solve t will solve.  A row that solve t - 1 skipped must have kept, through
            # that solve, the values and policy it had at plan t - 1.
            nonlocal checked
            for r, (V, psi, solved_then) in before.items():
                if not solved_then:
                    np.testing.assert_array_equal(L1.V[r], V)
                    np.testing.assert_array_equal(L1.psi[L1.psi_row[r]], psi)
                    checked += 1
            before.clear()
            for r in np.flatnonzero(L1.psi_row >= 0).tolist():
                before[r] = (L1.V[r].copy(), L1.psi[L1.psi_row[r]].copy(), bool(L1.active[r]))

        transitions(mdp, w, 6, record)
        self.assertGreater(checked, 0)

    def test_zero_is_the_rule_of_before(self):
        exact = tight_explicit(value_tol=1e-5, value_maps=False)
        lazy0 = tight_explicit(value_tol=1e-5, value_maps=False, lazy_mass=0.0, lazy_period=2)
        states = []
        for cfg in (exact, lazy0):
            w, mdp = fresh(cfg)
            active = []
            transitions(mdp, w, 5, lambda t, m: active.append(m.L1.active.copy()))
            states.append((active, [b.phys() for b in mdp.about], mdp.L1.V.copy()))
        for a, b in zip(*[s[0] for s in states]):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(states[0][1], states[1][1]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(states[0][2], states[1][2])


class Rollouts(unittest.TestCase):
    def test_a_rollout_records_what_it_skipped(self):
        scene = Scene(name="tiny", **{k: v for k, v in TINY.items()})
        base = dict(num_threads=2, resolution=1.0, value_tol=1e-5, value_maps=False,
                    deadlock_window=6, livelock_window=0)
        exact = run_pairing(scene, ("cooperation", "competition"), levels=(2, 2),
                            cfg=Config(**base), n_steps=8)
        lazy = run_pairing(scene, ("cooperation", "competition"), levels=(2, 2),
                           cfg=Config(**base, lazy_mass=0.2, lazy_period=3), n_steps=8)
        self.assertNotIn("lazy", exact["trace"][0])
        skipped = [row["lazy"] for row in lazy["trace"]]
        self.assertEqual(skipped[0][0][0] + skipped[0][1][0], 0)       # all solved at first
        self.assertTrue(all(len(x) == 2 and len(x[0]) == 3 for x in skipped))
        self.assertTrue(any(x[0][0] + x[1][0] > 0 for x in skipped), "nothing skipped")
        self.assertTrue(all(x[0][1] <= 0.2 + 1e-12 and x[1][1] <= 0.2 + 1e-12 for x in skipped))
        # the same first actions (the first transition is exact; what follows may differ
        # by what the stale policies move, a little)
        self.assertEqual(lazy["trace"][0]["action"], exact["trace"][0]["action"])
        self.assertEqual(lazy["config"]["lazy_mass"], 0.2)


if __name__ == "__main__":
    unittest.main()
