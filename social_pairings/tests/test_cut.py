"""The partner's subgoals and the protected ("cut") likelihood of its physical goal
(``Config.physical_likelihood="cut"``, 2026-11-04; :mod:`social_pairings.subgoals`).

The user: "Ok let's go ahead and implement this and push to a new branch", of the
formalization's Eq. 25, L(g) = pi N_g(a) + (1 - pi) S(a), shown on the hall_crossing record.
Checked here on the tiny scene: the option; the subgoals (which are possible where, their
policies); the engine's update against the reference update of test_beliefs.py fed with
Eq. 25 computed by hand from the engine's own pieces (the "none" models' policies, the
subgoals' policies, the predictive subgoal belief); an act that only a subgoal explains moves
no physical hypothesis against another; every likelihood ratio lies between 1 and the
ratio of the own-task models'; without subgoals the likelihood is the "none" models'; the
subgoal dynamics; the record.
"""
import unittest

import numpy as np

from social_pairings import Config
from social_pairings import goals as G
from social_pairings.constants import HELD1, HOME, PLACE, WAIT
from social_pairings.rollout import rollout

from .reference import make_world
from .test_beliefs import engine_likelihoods, event_at, reference_update
from .test_models import TINY, solved, tight_explicit


def cut_reference(mdp, j, state, a):
    """Eq. 25 by hand: (PG,) pi N_g(a) + (1 - pi) S(a), and (pi, S, N)."""
    w = mdp.w
    s = w.sid(state)
    old = mdp.about[j]
    _, lik1, _ = engine_likelihoods(mdp, j, s, a)
    nav = lik1[:, mdp.none_index[j]]
    sg = mdp.subgoals[j]
    kappa = sg.policies(state, mdp.legal(j, True, s), np.asarray(w.cost_c[state[2], j], float))
    omega = np.einsum("g,t,gtz->tz", old.phys(), old.social(), old.options)
    pi = omega[:, 0].sum()
    S = (omega[:, 1:] * kappa[:, a][None, :]).sum() / (1.0 - pi) if pi < 1.0 else 0.0
    return pi * nav + (1.0 - pi) * S, pi, S, nav, lik1


class Cut(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit(physical_likelihood="cut")
        cls.w, cls.gid, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                       ("cooperation", "competition"),
                                                       (1.0, 1.0), seed=21)
        cls.base = [b.copy() for b in cls.mdp.about]

    def setUp(self):
        self.mdp.about = [b.copy() for b in self.base]

    def holding(self):
        w = self.w
        return (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HOME, HELD1], [0, 0]))

    def open_moves(self, state, j=1):
        q = self.mdp.q_values(self.mdp.root[j], self.w.sid(state), j)
        return [a for a in range(5) if np.isfinite(q[a])]

    def random_subgoals(self, j, rng):
        """A random subgoal belief, each social goal on the subgoals it allows."""
        mdp = self.mdp
        sg = mdp.subgoals[j]
        P = np.zeros_like(mdp.about[j].options)
        for t in range(len(sg.hyp_goals)):
            mask = np.concatenate([[True], sg.allowed[t]])
            v = rng.dirichlet(np.full(int(mask.sum()), 0.8), size=P.shape[0])
            P[:, t, mask] = v
        mdp.about[j].options = P

    def actions_at(self, state, j=1):
        return [tuple(WAIT if r != j else a for r in (0, 1)) for a in self.open_moves(state, j)]

    def test_the_option(self):
        self.assertEqual(Config().physical_likelihood, "level1")     # Eq. 6 as printed
        Config(physical_likelihood="cut")
        with self.assertRaises(ValueError):                            # its own task: "none"
            Config(physical_likelihood="cut", social_hypotheses=("cooperation", "conflict"))
        with self.assertRaises(ValueError):
            Config(physical_likelihood="cut", partner_social_goal="known")
        for name, bad in (("option_temperature", 0.0), ("option_detour", -1.0),
                          ("option_start", 1.5), ("option_switch", -0.1),
                          ("option_keepaway_end", 2.0), ("option_start", True)):
            with self.assertRaises(ValueError, msg=name):
                Config(**{name: bad})
        self.assertIsNone(solved(TINY, tight_explicit(), (2, 2), ("cooperation", "competition"),
                                 (1.0, 1.0), seed=21)[2].about[0].options)

    def test_the_subgoals(self):
        mdp, w = self.mdp, self.w
        sg = mdp.subgoals[1]
        kinds = sg.kinds
        start, held = self.start, self.holding()
        f0, f1 = sg.feasible(start), sg.feasible(held)
        for n, kind in enumerate(kinds):        # hands empty: fetch; holding: the others
            self.assertEqual(bool(f0[n]), kind in ("fetch", "keepaway"), (kind, "empty"))
            if kind in ("deliver", "handoff", "meet", "keepaway"):
                self.assertTrue(f1[n], (kind, "holding"))
            else:
                self.assertFalse(f1[n], (kind, "holding"))
        for state in (start, held):
            s = w.sid(state)
            legal = mdp.legal(1, True, s)
            kappa = sg.policies(state, legal, np.asarray(w.cost_c[state[2], 1], float))
            feas = sg.feasible(state)
            np.testing.assert_allclose(kappa.sum(1), feas.astype(float), atol=1e-12)
            self.assertEqual(float(kappa[:, ~legal].sum()), 0.0)
        # helping goals hold the helping subgoals, hostile ones keeping away, the rest none
        for t, goal in enumerate(sg.hyp_goals):
            for n, kind in enumerate(kinds):
                want = (kind == "keepaway") == (goal in (G.CONFLICT, G.COMPETITION, G.HINDER))
                want = want and goal not in (G.NONE, G.COERCION)
                self.assertEqual(bool(sg.allowed[t, n]), want, (G.goal_name(goal), kind))

    def test_against_the_reference(self):
        """Eq. 25 in Eq. 6's place, the rest of the update as without it (the idle mode, the
        transition term, the two posteriors apart, Eq. 5)."""
        mdp, w = self.mdp, self.w
        rng = np.random.default_rng(3)
        n = 0
        for state in (self.start, self.holding()):
            for actions in self.actions_at(state):
                self.random_subgoals(1, rng)
                s = w.sid(state)
                nxt = w.unsid(w.step(s, actions))
                event = event_at(w, state)
                old = mdp.about[1]
                L, pi, S, nav, lik1 = cut_reference(mdp, 1, state, actions[1])
                self.assertTrue(0.0 < pi < 1.0)
                keep = mdp.transition_term(1, s, actions, nxt, event)
                l_idle = float(np.exp(mdp._log_idle(1, s, actions[1])))
                want = reference_update(old.phys(), old.social(), old.idle, L, lik1, l_idle,
                                        keep, int(w.hold[state[2], 1]) < 0, mdp.pg_obj[1],
                                        mdp.cfg.idle_hazard, True, True, mdp.pg_dst[1])
                got = mdp.updated_belief(1, s, actions, nxt, event)
                np.testing.assert_allclose(got.phys(), want[0], atol=1e-12)
                np.testing.assert_allclose(got.social(), want[1], atol=1e-12)
                self.assertAlmostEqual(got.idle, want[2], places=12)
                np.testing.assert_allclose(got.options.sum(-1), 1.0, atol=1e-12)
                n += 1
        self.assertGreaterEqual(n, 6)

    def test_an_act_for_the_partner_moves_no_physical_goal(self):
        """Robot 1, holding its object on a candidate that is not its destination, sets it
        down there for robot 0, the subgoal belief all on that set-down: every physical goal
        explains it alike, and the destination posterior is the prior's but for what the
        transition term rules out."""
        mdp, w = self.mdp, self.w
        sg = mdp.subgoals[1]
        other = [k for k in range(w.K) if int(w.dest[k]) != int(w.true_dest[1])
                 and int(w.dest[k]) != int(w.cell_id[(0, 2)])]
        if not other:
            self.skipTest("no second candidate on this map")
        c = other[0]
        state = (int(w.cell_id[(0, 2)]), int(w.dest[c]), w.block_of([HOME, HELD1], [0, 0]))
        s = w.sid(state)
        actions = (WAIT, PLACE)
        self.assertTrue(mdp.legal(1, True, s)[PLACE])
        old = mdp.about[1]
        t_coop = sg.hyp_goals.index(G.COOPERATION)
        old.log_social = np.log(np.eye(len(sg.hyp_goals))[t_coop] + 1e-300)
        P = np.zeros_like(old.options)
        P[:, :, 0] = 1.0
        P[:, t_coop, 0] = 0.0
        P[:, t_coop, 1 + sg.items.index(("deliver", c))] = 1.0
        old.options = P
        L, pi, S, _, _ = cut_reference(mdp, 1, state, PLACE)
        self.assertLess(pi, 1e-12)
        np.testing.assert_allclose(L, S, rtol=1e-12)
        nxt = w.unsid(w.step(s, actions))
        event = event_at(w, state)
        keep = mdp.transition_term(1, s, actions, nxt, event)
        got = mdp.updated_belief(1, s, actions, nxt, event)
        prior = old.phys() * keep
        prior = prior / prior.sum()
        _, want_d = mdp.factors(1, np.log(np.maximum(prior, 1e-300)))
        _, got_d = mdp.factors(1, got.log_phys)
        np.testing.assert_allclose(np.exp(got_d), np.exp(want_d), atol=1e-9)

    def test_ratios_between_one_and_the_own_task(self):
        mdp = self.mdp
        rng = np.random.default_rng(5)
        n = 0
        for state in (self.start, self.holding()):
            for actions in self.actions_at(state):
                self.random_subgoals(1, rng)
                L, pi, S, nav, _ = cut_reference(mdp, 1, state, actions[1])
                for g1 in range(mdp.PG):
                    for g2 in range(mdp.PG):
                        if nav[g2] <= 0.0 or L[g2] <= 0.0:
                            continue
                        r, rn = L[g1] / L[g2], nav[g1] / nav[g2]
                        self.assertGreaterEqual(r, min(1.0, rn) * (1 - 1e-12))
                        self.assertLessEqual(r, max(1.0, rn) * (1 + 1e-12))
                        n += 1
        self.assertGreater(n, 50)

    def test_without_subgoals_it_is_the_none_models(self):
        """No subgoal ever starting (option_start = 0): every likelihood is robot j's
        "none" model's, transition after transition."""
        cfg = tight_explicit(physical_likelihood="cut", option_start=0.0)
        w, _, mdp, start, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"),
                                     (1.0, 1.0), seed=21)
        state = start
        for _ in range(3):
            s = w.sid(state)
            actions = tuple(int(np.argmax(mdp.q_values(mdp.root[i], s, i))) for i in (0, 1))
            nxt = w.unsid(w.step(s, actions))
            event = event_at(w, state)
            for j in (0, 1):
                if w.busy[state[2], j]:
                    continue
                old = mdp.about[j]
                self.assertTrue(np.all(old.options[:, :, 0] == 1.0))
                _, lik1, l_idle = engine_likelihoods(mdp, j, s, actions[j])
                keep = mdp.transition_term(j, s, actions, nxt, event)
                want = reference_update(old.phys(), old.social(), old.idle,
                                        lik1[:, mdp.none_index[j]], lik1, l_idle, keep,
                                        int(w.hold[state[2], j]) < 0, mdp.pg_obj[j],
                                        cfg.idle_hazard, True, True, mdp.pg_dst[j])
                got = mdp.updated_belief(j, s, actions, nxt, event)
                np.testing.assert_allclose(got.phys(), want[0], atol=1e-12)
                mdp.about[j] = got
            state = nxt

    def test_the_dynamics(self):
        mdp, w = self.mdp, self.w
        sg = mdp.subgoals[1]
        state = self.holding()
        reader = mdp.about[0].phys()
        chi = sg.prior(state, mdp.pg_obj[1], mdp.pg_dst[1], reader, mdp.pg_obj[0], mdp.pg_dst[0])
        P = np.zeros_like(mdp.about[1].options)
        P[:, :, 0] = 1.0
        out = sg.predict(P, chi, np.zeros(sg.L, dtype=bool))
        can = chi.sum(-1) > 0
        np.testing.assert_allclose(out[:, :, 1:].sum(-1),
                                   np.where(can, self.cfg.option_start, 0.0), atol=1e-12)
        np.testing.assert_allclose(chi.sum(-1), can.astype(float), atol=1e-12)
        # a set-down on its cell ends "deliver" there; others go on but for an interruption
        c = [k for k in range(w.K)][0]
        n = sg.items.index(("deliver", c))
        P = np.zeros_like(P)
        P[:, :, 1 + n] = 1.0
        here = (state[0], int(w.dest[c]), state[2])
        ended = sg.acted(here, PLACE)
        self.assertTrue(ended[n])
        out = sg.predict(P, np.zeros_like(chi), ended)
        np.testing.assert_allclose(out[:, :, 0], 1.0, atol=1e-12)
        out = sg.predict(P, np.zeros_like(chi), np.zeros(sg.L, dtype=bool))
        np.testing.assert_allclose(out[:, :, 1 + n], 1.0 - self.cfg.option_switch, atol=1e-12)

    def test_the_record(self):
        cfg = tight_explicit(physical_likelihood="cut", value_maps=False, deadlock_window=6,
                             livelock_window=0)
        w, goal_ids = make_world(TINY, cfg)
        result = rollout(w, goal_ids, (6, 6), (2, 2),
                         tuple(G.parse_goal(g) for g in ("cooperation", "competition")),
                         (1.0, 1.0), cfg, 4)
        rows = [r for r in result["trace"] if "cut" in r["belief_update"][0]]
        self.assertTrue(rows)
        for row in result["trace"]:
            for i in (0, 1):
                self.assertAlmostEqual(float(np.sum(row["belief"][i])), 1.0)
            for d in row["belief_update"]:
                if "cut" in d:
                    self.assertTrue(0.0 <= d["cut"]["pi"] <= 1.0)
                    self.assertIn("subgoal_belief", d)


if __name__ == "__main__":
    unittest.main()
