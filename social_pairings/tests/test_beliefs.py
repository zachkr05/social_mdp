"""The belief update of 2026-10-30: Eqs. 5-6 as the authors' solver keeps them.

Two posteriors about each robot (``SocialMDP.about``): the physical one from the robot's
level-0 policies (Eq. 6, the authors' physical-goal estimation), staged by the robot's hands
(``Config.belief_stages``); the social one from the robot's level-1 hypothesis policies
averaged over the physical posterior (Eq. 5, the authors' social-goal estimation); one idle
probability as a two-mode filter over both (``Config.idle_hypothesis``); and Eq. 5's
transition term for the hands' goal-dependent outcomes (``Config.condition_on_placement``).
Nothing else: the lapse, the carry and park tables, the "outcome" observation and the
separate inference temperature of before are gone (``Config`` raises on them).

The engine's update is checked against :func:`reference_update`, the same arithmetic written
out from the equations on plain probabilities, from the engine's own likelihoods (Eq. 7's
softmax of the hypothesis models' Q, which test_models.py checks against brute force); the
transition term against the hands' rule case by case; and a rollout of the tiny scene
against the reference transition by transition.
"""
import unittest

import numpy as np

from social_pairings import Config
from social_pairings.config import BELIEF_DEVICES, TOLD
from social_pairings.constants import GIVE, HELD0, HELD1, HOME, PICK, PLACE, WAIT
from social_pairings.rollout import rollout

from .reference import make_world
from .test_models import TINY, solved, tight_explicit, tight_now

QUIET = dict(done_now=(False, False), failed=(False, False), handed_in_vain=(False, False))


def reference_update(prior_phys, prior_social, idle, lik0, lik1, l_idle, keep, empty, obj,
                     hazard, stages=True):
    """The update of one transition on probabilities (not logs), from the equations.

    prior_phys (PG,), prior_social (nT,) or None, idle: the beliefs as they were.
    lik0 (PG,): psi^0_g(a | s) of each physical hypothesis; lik1 (PG, nT): psi^1_(g,t)(a | s)
    of each level-1 hypothesis; l_idle: the idle mode's probability of the action; keep
    (PG,) bool: Eq. 5's transition term; empty: the robot's hands were empty (the stage);
    obj (PG,): the object of each physical hypothesis.  -> (phys, social, idle)."""
    prior_phys = np.asarray(prior_phys, dtype=float)
    lik0 = np.asarray(lik0, dtype=float)
    L = lik0.copy()
    if stages and empty:                 # the destination averaged out within each object
        for q in np.unique(obj):
            sel = obj == q
            mass = prior_phys[sel].sum()
            if mass > 0.0:
                L[sel] = (prior_phys[sel] * lik0[sel]).sum() / mass
    mix = (1.0 - idle) * L + idle * l_idle
    post = prior_phys * mix * keep
    explained = post.sum() > 0.0
    if not explained:
        post = prior_phys * keep
    if post.sum() > 0.0:
        if stages and not empty:         # object by object, to the prior's mass of each
            out = np.zeros_like(post)
            for q in np.unique(obj):
                sel = obj == q
                if post[sel].sum() > 0.0 and prior_phys[sel].sum() > 0.0:
                    out[sel] = post[sel] / post[sel].sum() * prior_phys[sel].sum()
            phys = out / out.sum()
        else:
            phys = post / post.sum()
    else:
        phys = prior_phys
    # the prior over the physical goals the transition term keeps: what the idle mode and the
    # social goals are judged against (the term is the same under every social goal)
    kept = prior_phys * keep
    kept = kept / kept.sum() if kept.sum() > 0.0 else prior_phys
    q1 = idle                            # the mode's posterior ...
    if explained:
        active = float((kept * L).sum())
        total = (1.0 - idle) * active + idle * l_idle
        q1 = idle * l_idle / total if total > 0.0 else idle
    new_idle = q1 * (1.0 - hazard) + (1.0 - q1) * hazard   # ... then the switch, every step
    social = None
    if prior_social is not None:
        Lt = np.asarray([(kept * np.asarray(lik1)[:, t]).sum()
                         for t in range(len(prior_social))])
        post_t = np.asarray(prior_social) * ((1.0 - idle) * Lt + idle * l_idle)
        social = post_t / post_t.sum() if post_t.sum() > 0.0 else np.asarray(prior_social)
    return phys, social, new_idle


def engine_likelihoods(mdp, j, s, a):
    """The engine's likelihoods of robot j's action a at joint state s, on probabilities:
    lik0 (PG,), lik1 (PG, nT) (zero rows for the hypotheses ruled out), l_idle.  Under
    ``Config.physical_likelihood="level1"`` lik0 is lik1 averaged over the social posterior
    (Eq. 6 as printed)."""
    lik0 = np.exp(mdp.physical_log_likelihood(j, s, a))
    nT = len(mdp.hyp_goals[1 - j])
    lik1 = np.zeros((mdp.PG, nT))
    for g in range(mdp.PG):
        if not np.isfinite(mdp.about[j].log_phys[g]):
            continue
        for t in range(nT):
            r = mdp.hypothesis_row(j, g, t)
            lik1[g, t] = np.exp(mdp._log_policy((1, r), s, j))[a]
    if mdp.cfg.physical_likelihood == "level1" and mdp.about[j].log_social is not None:
        lik0 = lik1 @ mdp.about[j].social()
    l_idle = float(np.exp(mdp._log_idle(j, s, a)))
    return lik0, lik1, l_idle


def event_at(w, state) -> dict:
    return dict(QUIET, held=tuple(int(w.hold[state[2], i]) for i in (0, 1)))


class Options(unittest.TestCase):
    def test_the_devices_of_before_are_gone(self):
        cfg = Config()
        for name, value in BELIEF_DEVICES.items():
            self.assertEqual(getattr(cfg, name), value)
        for bad in ("joint", "level2", None):
            with self.assertRaises(ValueError):
                Config(physical_likelihood=bad)
        self.assertEqual(cfg.physical_likelihood, "level1")   # Eq. 6 as printed
        for name, bad in (("belief_lapse", 0.05), ("partner_delivers", True),
                          ("partner_parks", True), ("partner_observed", "outcome"),
                          ("inference_temperature", 0.5), ("social_belief", "joint"),
                          ("child_conditioning", True)):
            with self.assertRaises(ValueError, msg=name):
                Config(**{name: bad})
        Config(**TOLD)                           # the defaults of before still build
        self.assertTrue(cfg.idle_hypothesis and cfg.belief_stages and cfg.condition_on_placement)
        self.assertEqual(cfg.tau_inference, cfg.tau)


class Structure(unittest.TestCase):
    """Two posteriors per robot, the product as the planning weights."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit()
        cls.w, _, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "competition"), (1.0, 1.0),
                                                 seed=11)

    def test_priors(self):
        w, _, mdp, _, _ = solved(TINY, self.cfg, (2, 2), ("cooperation", "competition"),
                                 (1.0, 1.0), seed=12)
        fresh = type(mdp)(w, mdp.goal_ids, (2, 2), mdp.goals, mdp.weights, self.cfg, 2)
        for j in (0, 1):
            b = fresh.about[j]
            np.testing.assert_allclose(b.phys(), np.full(mdp.PG, 1.0 / mdp.PG))
            np.testing.assert_allclose(b.social(), np.full(7, 1.0 / 7))
            self.assertEqual(b.idle, self.cfg.idle_hazard)
        fresh1 = type(mdp)(w, mdp.goal_ids, (1, 2), mdp.goals, mdp.weights, self.cfg, 2)
        self.assertIsNone(fresh1.about[1].log_social)      # robot 0 at level 1: no social belief
        self.assertIsNotNone(fresh1.about[0].log_social)
        fresh0 = type(mdp)(w, mdp.goal_ids, (0, 2), mdp.goals, mdp.weights, self.cfg, 2)
        np.testing.assert_allclose(fresh0.goal_belief(0), np.full(w.K, 1.0 / w.K))
        self.assertIsNone(fresh0.social_belief(0))
        self.assertIsNone(fresh0.idle_belief(0))

    def test_planning_weights_are_the_product(self):
        mdp = self.mdp
        for a in (0, 1):
            slot = mdp.root_slot(a)
            b = mdp.about[1 - a]
            want = np.outer(b.phys(), b.social()).reshape(-1)
            np.testing.assert_allclose(mdp.planning_weights(slot), want)
            for k, t, wt in zip(slot.hyp_goal, slot.hyp_type, want):
                self.assertAlmostEqual(wt, b.phys()[k] * b.social()[t])
            for g in range(mdp.PG):             # the level-1 slots: the physical posterior
                np.testing.assert_allclose(mdp.planning_weights(mdp.slots[(a, 1, g)]), b.phys())

    def test_the_level_1_models_of_a_robot_share_its_partner_distribution(self):
        mdp = self.mdp
        for a in (0, 1):
            rows = [mdp.slots[(a, 1, g)].p_row for g in range(mdp.PG)]
            self.assertEqual(len(set(rows)), 1)
        self.assertEqual(mdp.L1.G, 2)
        self.assertEqual(mdp.L2.G, 2)

    def test_marginals(self):
        mdp, w = self.mdp, self.w
        for i in (0, 1):
            b = mdp.about[1 - i].phys()
            np.testing.assert_allclose(mdp.goal_belief(i), np.bincount(mdp.pg_dst[1 - i],
                                                                     weights=b, minlength=w.K))
            np.testing.assert_allclose(mdp.object_belief(i), np.bincount(mdp.pg_obj[1 - i],
                                                                       weights=b, minlength=w.M))
            self.assertAlmostEqual(mdp.social_belief(i).sum(), 1.0)
            self.assertEqual(mdp.idle_belief(i), mdp.about[1 - i].idle)


class Update(unittest.TestCase):
    """The engine's update against the reference, from random beliefs, at several states."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit()
        cls.w, _, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "competition"), (1.0, 1.0),
                                                 seed=21)

    def check(self, state, actions, stages=True):
        w, mdp = self.w, self.mdp
        s = w.sid(state)
        nxt = w.unsid(w.step(s, actions))
        event = event_at(w, state)
        for j in (0, 1):
            if w.busy[state[2], j]:
                continue
            a = int(actions[j])
            if not np.isfinite(mdp.q_values(mdp.root[j], s, j)[a]):
                continue                 # not a legal action here
            old = mdp.about[j]
            lik0, lik1, l_idle = engine_likelihoods(mdp, j, s, a)
            keep = mdp.transition_term(j, s, actions, nxt, event)
            want = reference_update(old.phys(), old.social(), old.idle, lik0, lik1, l_idle,
                                    keep, int(w.hold[state[2], j]) < 0, mdp.pg_obj[j],
                                    self.cfg.idle_hazard, stages)
            got = mdp.updated_belief(j, s, actions, nxt, event)
            np.testing.assert_allclose(got.phys(), want[0], atol=1e-12)
            np.testing.assert_allclose(got.social(), want[1], atol=1e-12)
            self.assertAlmostEqual(got.idle, want[2], places=12)
            self.assertAlmostEqual(got.phys().sum(), 1.0)
            self.assertAlmostEqual(got.social().sum(), 1.0)
        return True

    def test_at_the_start(self):
        for a0 in range(5):
            for a1 in range(5):
                self.check(self.start, (a0, a1))

    def test_with_an_object_in_hand(self):
        w = self.w
        holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HOME, HELD0], [0, 0]))
        for a0 in (0, 1, 2, 3, WAIT, PLACE, GIVE):
            for a1 in (0, WAIT, PICK):
                self.check(holding, (a0, a1))

    def test_without_the_stages(self):
        cfg = tight_explicit(belief_stages=False)
        self.w, _, self.mdp, self.start, _ = solved(TINY, cfg, (2, 2),
                                                    ("cooperation", "competition"), (1.0, 1.0),
                                                    seed=22)
        try:
            for a0 in range(5):
                self.check(self.start, (a0, WAIT), stages=False)
        finally:
            self.setUpClass()

    def test_the_level_0_likelihood_of_the_authors_code(self):
        """``physical_likelihood="level0"``: the physical likelihood is the partner's level-0
        policies (the authors' physical-goal estimator), not Eq. 6 as printed -- the
        level-1 hypothesis policies averaged over the social posterior, the default; the
        social update is the same under both."""
        cfg = tight_explicit(physical_likelihood="level0")
        self.w, _, self.mdp, self.start, _ = solved(TINY, cfg, (2, 2),
                                                    ("cooperation", "competition"), (1.0, 1.0),
                                                    seed=23)
        try:
            w, mdp = self.w, self.mdp
            s = w.sid(self.start)
            for a0 in range(5):
                self.check(self.start, (a0, WAIT))
            holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]),
                       w.block_of([HOME, HELD0], [0, 0]))
            s = w.sid(holding)
            differ = 0.0
            for a0 in (0, 1, 2, 3, WAIT, PLACE, GIVE):
                self.check(holding, (a0, WAIT))
                lik0, lik1, _ = engine_likelihoods(mdp, 0, s, a0)
                self.assertTrue(np.allclose(lik0, np.exp(mdp.physical_log_likelihood(0, s, a0))))
                differ = max(differ, float(np.abs(lik0 - lik1 @ mdp.about[0].social()).max()))
            self.assertGreater(differ, 1e-6)     # not Eq. 6 as printed
        finally:
            self.setUpClass()

    def test_the_stage_rule(self):
        """Hands empty: the belief about the destination given the object does not move.
        Holding: the belief about the object does not move."""
        w, mdp = self.w, self.mdp
        s = w.sid(self.start)
        for a0 in range(4):
            old = mdp.about[0]
            got = mdp.updated_belief(0, s, (a0, WAIT), w.unsid(w.step(s, (a0, WAIT))),
                                     event_at(w, self.start))
            for q in range(w.M):
                sel = mdp.pg_obj[0] == q
                before, after = old.phys()[sel], got.phys()[sel]
                np.testing.assert_allclose(after / after.sum(), before / before.sum(), atol=1e-12)
        holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HELD0, HOME], [0, 0]))
        s = w.sid(holding)
        for a0 in range(4):
            if not np.isfinite(mdp.q_values(mdp.root[0], s, 0)[a0]):
                continue
            old = mdp.about[0]
            got = mdp.updated_belief(0, s, (a0, WAIT), w.unsid(w.step(s, (a0, WAIT))),
                                     event_at(w, holding))
            for q in range(w.M):
                sel = mdp.pg_obj[0] == q
                self.assertAlmostEqual(got.phys()[sel].sum(), old.phys()[sel].sum(), places=12)

    def test_the_physical_posterior_uses_level_0_only(self):
        """The social goal does not enter Eq. 6: the physical posterior is the same whatever
        the social posterior."""
        w, mdp = self.w, self.mdp
        s = w.sid(self.start)
        old = mdp.about[1].copy()
        got_a = mdp.updated_belief(1, s, (WAIT, 1), w.unsid(w.step(s, (WAIT, 1))),
                                   event_at(w, self.start))
        mdp.about[1].log_social = np.log(np.full(7, 1.0 / 7))
        got_b = mdp.updated_belief(1, s, (WAIT, 1), w.unsid(w.step(s, (WAIT, 1))),
                                   event_at(w, self.start))
        mdp.about[1] = old
        np.testing.assert_allclose(got_a.phys(), got_b.phys(), atol=1e-12)
        self.assertAlmostEqual(got_a.idle, got_b.idle, places=12)

    def test_lifting_tells_nothing(self):
        w, mdp = self.w, self.mdp
        lifting = (int(w.cell_id[(2, 2)]), int(w.cell_id[(0, 0)]),
                   w.block_of([w.lift0, HOME], [0, 0]))        # robot 0 one step into a lift
        self.assertTrue(w.busy[lifting[2], 0])
        s = w.sid(lifting)
        old = mdp.about[0]
        got = mdp.updated_belief(0, s, (PICK, WAIT), w.unsid(w.step(s, (PICK, WAIT))),
                                 event_at(w, lifting))
        np.testing.assert_array_equal(got.log_phys, old.log_phys)
        np.testing.assert_array_equal(got.log_social, old.log_social)
        self.assertEqual(got.idle, old.idle)


class TransitionTerm(unittest.TestCase):
    """Eq. 5's T term: what the hands reveal, case by case."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit()
        cls.w, _, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "competition"), (1.0, 1.0),
                                                 seed=31)

    def test_a_robot_that_finishes_reveals_its_goal(self):
        w, mdp = self.w, self.mdp
        obj, dst = mdp.pg_obj[1], mdp.pg_dst[1]
        state = (int(w.cell_id[(0, 2)]), int(w.true_dest[1]), w.block_of([HOME, HELD1], [0, 0]))
        s = w.sid(state)
        actions = (WAIT, PLACE)
        nxt = w.unsid(w.step(s, actions))
        self.assertTrue(w.done[nxt[2], 1])
        event = dict(QUIET, done_now=(False, True), held=(-1, 1))
        keep = mdp.transition_term(1, s, actions, nxt, event)
        want = (obj == 1) & (w.dest[dst] == int(w.true_dest[1]))
        np.testing.assert_array_equal(keep, want)
        self.assertEqual(int(want.sum()), 1)
        got = mdp.updated_belief(1, s, actions, nxt, event)
        np.testing.assert_allclose(got.phys(), want.astype(float))

    def test_a_set_down_of_the_own_object_that_did_nothing(self):
        w, mdp = self.w, self.mdp
        obj, dst = mdp.pg_obj[1], mdp.pg_dst[1]
        other = [k for k in range(w.K) if int(w.dest[k]) != int(w.true_dest[1])
                 and int(w.dest[k]) != int(w.cell_id[(0, 2)])]
        if not other:
            self.skipTest("no second candidate on this map")
        cell = int(w.dest[other[0]])
        state = (int(w.cell_id[(0, 2)]), cell, w.block_of([HOME, HELD1], [0, 0]))
        s = w.sid(state)
        actions = (WAIT, PLACE)
        nxt = w.unsid(w.step(s, actions))
        self.assertEqual(int(w.hold[nxt[2], 1]), 1)             # still in its hands
        keep = mdp.transition_term(1, s, actions, nxt, dict(QUIET, held=(-1, 1)))
        np.testing.assert_array_equal(keep, ~((obj == 1) & (w.dest[dst] == cell)))

    def test_a_set_down_for_the_partner_that_failed(self):
        w, mdp = self.w, self.mdp
        obj, dst = mdp.pg_obj[1], mdp.pg_dst[1]
        other = [k for k in range(w.K) if int(w.dest[k]) != int(w.true_dest[1])]
        cell = int(w.dest[other[0]])
        # robot 0 holds robot 1's object on a candidate that is not robot 1's destination
        state = (cell, int(w.cell_id[(2, 0)]) if int(w.cell_id[(2, 0)]) != cell
                 else int(w.cell_id[(2, 1)]), w.block_of([HOME, HELD0], [0, 0]))
        s = w.sid(state)
        actions = (PLACE, WAIT)
        nxt = w.unsid(w.step(s, actions))
        self.assertEqual(int(w.hold[nxt[2], 0]), 1)
        event = dict(QUIET, failed=(True, False), held=(1, -1))
        keep = mdp.transition_term(1, s, actions, nxt, event)
        np.testing.assert_array_equal(keep, ~((obj == 1) & (w.dest[dst] == cell)))
        # ... and one that counted: robot 1's goal is (its object, that cell)
        state = (int(w.true_dest[1]), int(w.cell_id[(2, 0)]) if int(w.cell_id[(2, 0)]) !=
                 int(w.true_dest[1]) else int(w.cell_id[(2, 1)]), w.block_of([HOME, HELD0], [0, 0]))
        s = w.sid(state)
        nxt = w.unsid(w.step(s, actions))
        self.assertTrue(w.done[nxt[2], 1])
        keep = mdp.transition_term(1, s, actions, nxt, dict(QUIET, done_now=(False, True),
                                                             held=(1, -1)))
        np.testing.assert_array_equal(keep, (obj == 1) & (w.dest[dst] == int(w.true_dest[1])))

    def test_a_hand_over(self):
        w, mdp = self.w, self.mdp
        obj = mdp.pg_obj[1]
        # robot 0 holds robot 1's object next to robot 1: a hand-over robot 1 takes
        state = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HOME, HELD0], [0, 0]))
        s = w.sid(state)
        actions = (GIVE, WAIT)
        nxt = w.unsid(w.step(s, actions))
        self.assertEqual(int(w.hold[nxt[2], 1]), 1)
        keep = mdp.transition_term(1, s, actions, nxt, dict(QUIET, held=(1, -1)))
        np.testing.assert_array_equal(keep, obj == 1)
        # robot 0 holds its own object: robot 1 does not take it (handed over in vain)
        state = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HELD0, HOME], [0, 0]))
        s = w.sid(state)
        nxt = w.unsid(w.step(s, actions))
        self.assertEqual(int(w.hold[nxt[2], 0]), 0)
        keep = mdp.transition_term(1, s, actions, nxt,
                                   dict(QUIET, handed_in_vain=(True, False), held=(0, -1)))
        np.testing.assert_array_equal(keep, obj != 0)

    def test_without_the_term(self):
        cfg = tight_explicit(condition_on_placement=False)
        w, _, mdp, start, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"),
                                     (1.0, 1.0), seed=32)
        state = (int(w.cell_id[(0, 2)]), int(w.true_dest[1]), w.block_of([HOME, HELD1], [0, 0]))
        s = w.sid(state)
        actions = (WAIT, PLACE)
        nxt = w.unsid(w.step(s, actions))
        got = mdp.updated_belief(1, s, actions, nxt, dict(QUIET, done_now=(False, True),
                                                           held=(-1, 1)))
        self.assertGreater(int((got.phys() > 0.0).sum()), 1)


class Rollout(unittest.TestCase):
    """A rollout of the tiny scene, every posterior of every transition against the
    reference, and the record's beliefs consistent with the engine's."""

    def test_rollout_against_the_reference(self):
        cfg = tight_now(value_tol=1e-5, value_maps=False, deadlock_window=6, livelock_window=0)
        w, gid, mdp, state, _ = solved(TINY, cfg, (2, 2), ("cooperation", "competition"),
                                       (1.0, 1.0), seed=41)
        # a fresh engine, rolled out by hand for a few transitions with the engine's choices
        mdp = type(mdp)(w, gid, (2, 2), mdp.goals, mdp.weights, cfg, 2)
        mdp.allocate()
        mdp.solve_level0()
        state = w.initial
        for t in range(6):
            mdp.plan(state)
            if t == 0:
                mdp.init_models()
            for depth in (1, 2):
                mdp.inputs(depth)
                mdp.solve(depth)
            s = w.sid(state)
            actions = tuple(int(np.argmax(mdp.q_values(mdp.root[i], s, i))) for i in (0, 1))
            nxt = w.unsid(w.step(s, actions))
            status, done = tuple(int(x) for x in w.status[state[2]]), tuple(int(x) for x in
                                                                           w.done[state[2]])
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
                lik0, lik1, l_idle = engine_likelihoods(mdp, j, s, actions[j])
                keep = mdp.transition_term(j, s, actions, nxt, event)
                want.append(reference_update(old.phys(), old.social(), old.idle, lik0, lik1,
                                             l_idle, keep, int(w.hold[state[2], j]) < 0,
                                             mdp.pg_obj[j], cfg.idle_hazard))
            mdp.observe(actions, s, nxt, event)
            for j in (0, 1):
                np.testing.assert_allclose(mdp.about[j].phys(), want[j][0], atol=1e-10)
                np.testing.assert_allclose(mdp.about[j].social(), want[j][1], atol=1e-10)
                self.assertAlmostEqual(mdp.about[j].idle, want[j][2], places=10)
            state = nxt

    def test_the_record(self):
        cfg = tight_now(value_tol=1e-4, value_maps=False, deadlock_window=5, livelock_window=0)
        w, goal_ids = make_world(TINY, cfg)
        from social_pairings import goals as G
        result = rollout(w, goal_ids, (6, 6), (2, 2),
                         tuple(G.parse_goal(g) for g in ("cooperation", "competition")),
                         (1.0, 1.0), cfg, 4)
        for row in result["trace"]:
            for i in (0, 1):
                self.assertAlmostEqual(float(np.sum(row["belief"][i])), 1.0)
                self.assertAlmostEqual(float(np.sum(row["object_belief"][i])), 1.0)
                self.assertAlmostEqual(float(np.sum(row["social_belief"][i])), 1.0)
                self.assertTrue(0.0 <= row["idle_belief"][i] <= 1.0)
            self.assertNotIn("seen_as", row)


if __name__ == "__main__":
    unittest.main()
