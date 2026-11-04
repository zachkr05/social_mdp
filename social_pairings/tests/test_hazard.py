"""The partner's goal may change (2026-11-02): Config.goal_hazard, social_hazard, hand_hazard.

Baker, Saxe & Tenenbaum (2009), "Action understanding as inverse planning", Cognition, model
M2: the hidden goal is redrawn from the prior with probability h at every step, so after each
update a posterior is mixed with the uniform over the hypotheses it still holds possible,
``b <- (1 - h) b + h / |support|``.  Here: the physical posterior in the factor the
partner's next action speaks to (under ``Config.belief_stages``), the social posterior whole,
and, at the partner's hand actions, the factor the action speaks to mixed BEFORE its
evidence (``hand_hazard``).  0 (the default) is the update of before.

The engine's update is checked against the reference of test_beliefs.py with the mixing
written out on plain probabilities; the mixing itself against its definition; and a few
transitions of the tiny scene against the reference.
"""
import unittest

import numpy as np

from social_pairings import Config
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.constants import GIVE, HELD0, HOME, PICK, PLACE, WAIT
from social_pairings.models import mix_uniform

from .test_beliefs import engine_likelihoods, event_at, marginals, product, reference_update
from .test_models import TINY, solved, tight_explicit

HAZARDS = dict(goal_hazard=0.05, social_hazard=0.03, hand_hazard=0.5)


def mix(p, h):
    """(1 - h) p + h u, u uniform over p's support (p > 0), on probabilities."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    sup = p > 0.0
    out = np.zeros_like(p)
    out[sup] = (1.0 - h) * p[sup] + h / sup.sum()
    return out


def factor_mix(p, h, objects, obj, stages=True, factored=False, dst=None):
    """The physical posterior p (PG,) mixed in one factor: the object (``objects``; the
    destination given each object kept), the destination given each object (its mass
    kept), or -- without the stages -- whole.  factored (``Config.belief_factored``, the two
    posteriors apart): the object's posterior or the destination's mixed (both without the
    stages), and the belief their product."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    if h <= 0.0:
        return p
    if factored:
        po, pd = marginals(p, np.asarray(obj), np.asarray(dst))
        if objects or not stages:
            po = mix(po, h)
        if not objects or not stages:
            pd = mix(pd, h)
        return product(po, pd, np.asarray(obj), np.asarray(dst))
    if not stages:
        return mix(p, h)
    out = p.copy()
    groups = np.unique(obj)
    mass = np.array([p[obj == q].sum() for q in groups])
    if objects:
        new = mix(mass, h)
        for q, m, n in zip(groups, mass, new):
            if m > 0.0:
                out[obj == q] = p[obj == q] / m * n
    else:
        for q, m in zip(groups, mass):
            if m > 0.0:
                out[obj == q] = mix(p[obj == q] / m, h) * m
    return out


def reference_with_hazards(old, lik0, lik1, l_idle, keep, empty, holds_next, hand, obj, cfg,
                           stages=True, dst=None):
    """reference_update with the hazards: the hand hazard on the prior before the evidence,
    then the update, then goal_hazard and social_hazard after it."""
    fac = cfg.belief_factored
    prior = old.phys()
    if hand and cfg.hand_hazard > 0.0:
        prior = factor_mix(prior, cfg.hand_hazard, empty, obj, stages, fac, dst)
    phys, social, idle = reference_update(prior, old.social(), old.idle, lik0, lik1, l_idle,
                                          keep, empty, obj, cfg.idle_hazard, stages, fac, dst)
    phys = factor_mix(phys, cfg.goal_hazard, not holds_next, obj, stages, fac, dst)
    if social is not None and cfg.social_hazard > 0.0:
        social = mix(social, cfg.social_hazard)
    return phys, social, idle


class Options(unittest.TestCase):
    def test_defaults_and_records_of_before(self):
        cfg = Config()
        for name in ("goal_hazard", "social_hazard", "hand_hazard"):
            self.assertEqual(getattr(cfg, name), 0.0)          # the update of before
            self.assertEqual(ADDED_FIELDS[name], 0.0)
            self.assertEqual(written_with({})[name], 0.0)      # a record of before: none
        Config(goal_hazard=1.0, social_hazard=0.0, hand_hazard=1.0)
        for name in ("goal_hazard", "social_hazard", "hand_hazard"):
            for bad in (-0.01, 1.01, float("nan"), True, "0.1", None):
                with self.assertRaises(ValueError, msg=f"{name}={bad!r}"):
                    Config(**{name: bad})


class Mixing(unittest.TestCase):
    def test_mix_uniform(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            p = rng.dirichlet(np.full(7, 0.3))
            p[rng.random(7) < 0.3] = 0.0                       # some ruled out
            if p.sum() == 0.0:
                continue
            p = p / p.sum()
            with np.errstate(divide="ignore"):
                logp = np.log(p)
            for h in (0.0, 0.02, 0.5, 1.0):
                got = np.exp(mix_uniform(logp, h))
                np.testing.assert_allclose(got, mix(p, h), atol=1e-14)
                self.assertAlmostEqual(got.sum(), 1.0)
                n = int((p > 0).sum())
                self.assertTrue(np.all(got[p > 0] >= h / n - 1e-15))   # the floor
                self.assertTrue(np.all(got[p == 0] == 0.0))            # ruled out stays out
            np.testing.assert_allclose(np.exp(mix_uniform(logp, 1.0))[p > 0],
                                       1.0 / (p > 0).sum())             # h = 1: the heuristic H
            np.testing.assert_allclose(np.exp(mix_uniform(logp + 3.0, 0.1)), mix(p, 0.1),
                                       atol=1e-14)                      # normalised first

    def test_the_two_posteriors_apart(self):
        """The default since 2026-11-04 (``belief_factored``): the hazard mixes the object's
        posterior or the destination's, the other kept, and the belief is their product."""
        cfg = tight_explicit(**HAZARDS)
        self.assertTrue(cfg.belief_factored)
        w, _, mdp, _, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"), (1.0, 1.0),
                                 seed=5)
        for j in (0, 1):
            obj, dst = mdp.pg_obj[j], mdp.pg_dst[j]
            po, pd = marginals(mdp.about[j].phys(), obj, dst)
            pd[0] = 0.0                                         # one destination ruled out
            pd = pd / pd.sum()
            p = product(po, pd, obj, dst)
            with np.errstate(divide="ignore"):
                logp = np.log(p)
            for h in (0.1, 1.0):
                for objects in (True, False):
                    got = np.exp(mdp.hazard_phys(j, logp, h, objects=objects))
                    np.testing.assert_allclose(got, factor_mix(p, h, objects, obj, True, True,
                                                               dst), atol=1e-14)
                    go, gd = marginals(got, obj, dst)
                    if objects:
                        np.testing.assert_allclose(gd, pd, atol=1e-12)     # kept
                        np.testing.assert_allclose(go, mix(po, h), atol=1e-12)
                    else:
                        np.testing.assert_allclose(go, po, atol=1e-12)
                        np.testing.assert_allclose(gd, mix(pd, h), atol=1e-12)
                        self.assertTrue(np.all(gd[1:] >= h / (len(gd) - 1) - 1e-12))
                    self.assertEqual(gd[0], 0.0)                  # ruled out stays out
                    np.testing.assert_allclose(got, product(go, gd, obj, dst), atol=1e-14)

    def test_the_factors(self):
        """One joint posterior (``belief_factored=False``, the rule until 2026-11-04)."""
        cfg = tight_explicit(belief_factored=False, **HAZARDS)
        w, _, mdp, _, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"), (1.0, 1.0),
                                 seed=5)
        for j in (0, 1):
            obj = mdp.pg_obj[j]
            p = mdp.about[j].phys()
            p[0] = 0.0                                          # one hypothesis ruled out
            p = p / p.sum()
            with np.errstate(divide="ignore"):
                logp = np.log(p)
            for h in (0.1, 1.0):
                by_object = np.exp(mdp.hazard_phys(j, logp, h, objects=True))
                np.testing.assert_allclose(by_object, factor_mix(p, h, True, obj), atol=1e-14)
                by_destination = np.exp(mdp.hazard_phys(j, logp, h, objects=False))
                np.testing.assert_allclose(by_destination, factor_mix(p, h, False, obj),
                                           atol=1e-14)
                self.assertEqual(by_object[0], 0.0)
                self.assertEqual(by_destination[0], 0.0)
                for q in np.unique(obj):
                    sel = obj == q
                    # objects: the destination given the object kept; destinations: the mass
                    np.testing.assert_allclose(by_object[sel] / by_object[sel].sum(),
                                               p[sel] / p[sel].sum(), atol=1e-12)
                    self.assertAlmostEqual(by_destination[sel].sum(), p[sel].sum(), places=12)
                masses = np.array([by_object[obj == q].sum() for q in np.unique(obj)])
                self.assertTrue(np.all(masses >= h / len(masses) - 1e-12))
            np.testing.assert_array_equal(mdp.hazard_phys(j, logp, 0.0, objects=True), logp)


class Update(unittest.TestCase):
    """The engine's update with every hazard on, against the reference."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit(**HAZARDS)
        cls.w, _, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "competition"), (1.0, 1.0),
                                                 seed=21)

    def check(self, state, actions, stages=True) -> int:
        w, mdp = self.w, self.mdp
        s = w.sid(state)
        nxt = w.unsid(w.step(s, actions))
        event = event_at(w, state)
        checked = 0
        for j in (0, 1):
            if w.busy[state[2], j]:
                continue
            a = int(actions[j])
            if not np.isfinite(mdp.q_values(mdp.root[j], s, j)[a]):
                continue                                         # not a legal action here
            old = mdp.about[j]
            lik0, lik1, l_idle = engine_likelihoods(mdp, j, s, a)
            keep = mdp.transition_term(j, s, actions, nxt, event)
            holds_next = int(w.hold[nxt[2], j]) >= 0 or bool(w.busy[nxt[2], j])
            want = reference_with_hazards(old, lik0, lik1, l_idle, keep,
                                          int(w.hold[state[2], j]) < 0, holds_next, a >= PICK,
                                          mdp.pg_obj[j], self.cfg, stages, mdp.pg_dst[j])
            got = mdp.updated_belief(j, s, actions, nxt, event)
            np.testing.assert_allclose(got.phys(), want[0], atol=1e-12)
            np.testing.assert_allclose(got.social(), want[1], atol=1e-12)
            self.assertAlmostEqual(got.idle, want[2], places=12)
            # the floors: the factor the next action speaks to, and the social posterior
            obj = mdp.pg_obj[j]
            if not holds_next:
                masses = np.array([got.phys()[obj == q].sum() for q in np.unique(obj)])
                masses = masses[masses > 0.0]
                self.assertTrue(np.all(masses >= self.cfg.goal_hazard / len(masses) - 1e-12))
            social = got.social()[got.social() > 0.0]           # (a goal ruled out stays out)
            self.assertTrue(np.all(social >= self.cfg.social_hazard / social.size - 1e-12))
            checked += 1
        return checked

    def test_moves_with_empty_hands(self):
        n = sum(self.check(self.start, (a0, a1)) for a0 in range(5) for a1 in range(5))
        self.assertGreater(n, 0)

    def test_a_pick_up(self):
        """A pick-up: the hand hazard on the belief about the object, before its evidence;
        the lift then makes the next update one with the object in hand."""
        w, mdp = self.w, self.mdp
        state = (int(w.cell_id[(2, 2)]), int(w.cell_id[(0, 1)]), w.block_of([HOME, HOME], [0, 0]))
        s = w.sid(state)
        self.assertTrue(np.isfinite(mdp.q_values(mdp.root[0], s, 0)[PICK]))
        nxt = w.unsid(w.step(s, (PICK, WAIT)))
        self.assertTrue(w.busy[nxt[2], 0] or int(w.hold[nxt[2], 0]) >= 0)
        self.assertEqual(self.check(state, (PICK, WAIT)), 2)
        # against no hand hazard: the pick-up's evidence on a prior that forgot half of it
        old = mdp.about[0]
        got = mdp.updated_belief(0, s, (PICK, WAIT), nxt, event_at(w, state))
        lik0, lik1, l_idle = engine_likelihoods(mdp, 0, s, PICK)
        keep = mdp.transition_term(0, s, (PICK, WAIT), nxt, event_at(w, state))
        fac = self.cfg.belief_factored
        plain = reference_update(old.phys(), old.social(), old.idle, lik0, lik1, l_idle, keep,
                                 True, mdp.pg_obj[0], self.cfg.idle_hazard, True, fac,
                                 mdp.pg_dst[0])[0]
        self.assertGreater(np.abs(got.phys() - factor_mix(plain, self.cfg.goal_hazard, False,
                                                          mdp.pg_obj[0], True, fac,
                                                          mdp.pg_dst[0])).max(), 1e-6)

    def test_with_an_object_in_hand(self):
        w = self.w
        holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]),
                   w.block_of([HOME, HELD0], [0, 0]))
        n = sum(self.check(holding, (a0, a1)) for a0 in (0, 1, 2, 3, WAIT, PLACE, GIVE)
                for a1 in (0, WAIT, PICK))
        self.assertGreater(n, 0)

    def test_without_the_stages(self):
        cfg = tight_explicit(belief_stages=False, **HAZARDS)
        self.cfg = cfg
        self.w, _, self.mdp, self.start, _ = solved(TINY, cfg, (2, 2),
                                                    ("cooperation", "competition"), (1.0, 1.0),
                                                    seed=22)
        try:
            for a0 in range(5):
                self.check(self.start, (a0, WAIT), stages=False)
        finally:
            type(self).setUpClass()
            self.cfg = type(self).cfg

    def test_lifting_tells_nothing_and_nothing_changes(self):
        w, mdp = self.w, self.mdp
        lifting = (int(w.cell_id[(2, 2)]), int(w.cell_id[(0, 0)]),
                   w.block_of([w.lift0, HOME], [0, 0]))
        s = w.sid(lifting)
        old = mdp.about[0]
        got = mdp.updated_belief(0, s, (PICK, WAIT), w.unsid(w.step(s, (PICK, WAIT))),
                                 event_at(w, lifting))
        np.testing.assert_array_equal(got.log_phys, old.log_phys)
        np.testing.assert_array_equal(got.log_social, old.log_social)


class Recovery(unittest.TestCase):
    """What the hazard buys, on the arithmetic of one hypothesis against the rest (the
    numbers given the user, 2026-11-02): each surprising move multiplies it by 1/20,000, each
    move in its favour by 20."""

    @staticmethod
    def moves_back(h, surprising, n=4, start=0.98, against=1 / 20000, favour=20.0):
        p = np.full(n, (1.0 - start) / (n - 1))
        p[0] = start
        for _ in range(surprising):
            p = p * np.r_[against, np.ones(n - 1)]
            p = mix(p, h) if h > 0 else p / p.sum()
        for k in range(1, 100):
            p = p * np.r_[favour, np.ones(n - 1)]
            p = mix(p, h) if h > 0 else p / p.sum()
            if p[0] > 0.9:
                return k
        return None

    def test_the_damage_does_not_pile_up(self):
        self.assertEqual(self.moves_back(0.0, 1), 3)
        self.assertEqual(self.moves_back(0.0, 3), 10)       # as many as it took, and more
        self.assertEqual(self.moves_back(0.02, 1), 3)
        self.assertEqual(self.moves_back(0.02, 3), 3)       # the same few however deep
        self.assertEqual(self.moves_back(0.02, 10), 3)

    def test_an_unsupported_belief_fades(self):
        p = np.array([0.62, 0.38 / 3, 0.38 / 3, 0.38 / 3])
        for _ in range(35):
            p = mix(p, 0.02)
        self.assertLess(p[0], 0.25 + (0.62 - 0.25) / 2 + 0.01)   # about halfway in 35 steps


class Rollout(unittest.TestCase):
    def test_transitions_against_the_reference(self):
        cfg = tight_explicit(value_tol=1e-5, value_maps=False, deadlock_window=6,
                             livelock_window=0, **HAZARDS)
        w, gid, mdp, _, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"),
                                   (1.0, 1.0), seed=41)
        mdp = type(mdp)(w, gid, (2, 2), mdp.goals, mdp.weights, cfg, 2)
        mdp.allocate()
        mdp.solve_level0()
        state = w.initial
        hands = 0
        for t in range(8):
            mdp.plan(state)
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
            new_status, new_done, failed, handed, _ = w.real_step(status, done, state, nxt,
                                                                  actions, detail=True)
            event = dict(done_now=tuple(bool(new_done[i]) and not done[i] for i in (0, 1)),
                         failed=failed, handed_in_vain=handed,
                         held=tuple(int(w.hold[state[2], i]) for i in (0, 1)))
            want = []
            for j in (0, 1):
                old = mdp.about[j]
                if w.busy[state[2], j]:
                    want.append((old.phys(), old.social(), old.idle))
                    continue
                a = actions[j]
                hands += int(a >= PICK)
                lik0, lik1, l_idle = engine_likelihoods(mdp, j, s, a)
                keep = mdp.transition_term(j, s, actions, nxt, event)
                holds_next = int(w.hold[nxt[2], j]) >= 0 or bool(w.busy[nxt[2], j])
                want.append(reference_with_hazards(old, lik0, lik1, l_idle, keep,
                                                   int(w.hold[state[2], j]) < 0, holds_next,
                                                   a >= PICK, mdp.pg_obj[j], cfg, True,
                                                   mdp.pg_dst[j]))
            mdp.observe(actions, s, nxt, event)
            for j in (0, 1):
                np.testing.assert_allclose(mdp.about[j].phys(), want[j][0], atol=1e-10)
                np.testing.assert_allclose(mdp.about[j].social(), want[j][1], atol=1e-10)
            state = nxt
        self.assertGreater(hands, 0, "no hand action in the transitions checked")


if __name__ == "__main__":
    unittest.main()
