"""The belief update of 2026-10-30: Eqs. 5-6 as the authors' solver keeps them.

Two posteriors about each robot (``SocialMDP.about``): the physical one from the robot's
level-0 policies (Eq. 6, the authors' physical-goal estimation), staged by the robot's hands
(``Config.belief_stages``); the social one from the robot's level-1 hypothesis policies
averaged over the physical posterior (Eq. 5, the authors' social-goal estimation); one idle
probability as a two-mode filter over both (``Config.idle_hypothesis``); and Eq. 5's
transition term for the hands' goal-dependent outcomes (``Config.condition_on_placement``).
Nothing else: the carry and park tables and the "outcome" observation of before are gone
(``Config`` raises on them); the lapse and the separate inference temperature of before are
options again since 2026-11-04, off by default (classes ``Lapse``, ``InferenceTemperature``).

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
from social_pairings.util import log_softmax

from .reference import make_world
from .test_models import TINY, solved, tight_explicit, tight_now

QUIET = dict(done_now=(False, False), failed=(False, False), handed_in_vain=(False, False))


def marginals(p, obj, dst):
    """The two posteriors of a belief over (object, destination) hypotheses: its masses by
    object and by destination, on probabilities."""
    p = np.asarray(p, dtype=float)
    po = np.array([p[obj == q].sum() for q in range(int(obj.max()) + 1)])
    pd = np.array([p[dst == d].sum() for d in range(int(dst.max()) + 1)])
    return po, pd


def product(po, pd, obj, dst):
    """b(o, d) = b_O(o) b_D(d) over the hypotheses, normalised."""
    out = np.asarray(po)[obj] * np.asarray(pd)[dst]
    return out / out.sum()


def reference_update(prior_phys, prior_social, idle, lik0, lik1, l_idle, keep, empty, obj,
                     hazard, stages=True, factored=False, dst=None):
    """The update of one transition on probabilities (not logs), from the equations.

    prior_phys (PG,), prior_social (nT,) or None, idle: the beliefs as they were.
    lik0 (PG,): psi^0_g(a | s) of each physical hypothesis; lik1 (PG, nT): psi^1_(g,t)(a | s)
    of each level-1 hypothesis; l_idle: the idle mode's probability of the action; keep
    (PG,) bool: Eq. 5's transition term; empty: the robot's hands were empty (the stage);
    obj (PG,), dst (PG,): the object and the destination of each physical hypothesis.
    factored (``Config.belief_factored``): the two posteriors apart -- the one that learns
    in this stage (the object's with empty hands, the destination's holding; both without
    the stages) is the marginal of prior x likelihood x transition term, the other the
    marginal of prior x transition term, and the belief their product.  -> (phys, social,
    idle)."""
    prior_phys = np.asarray(prior_phys, dtype=float)
    lik0 = np.asarray(lik0, dtype=float)
    L = lik0.copy()
    if stages and empty and not factored:    # the destination averaged out within each object
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
    if post.sum() > 0.0 and factored:
        obj, dst = np.asarray(obj), np.asarray(dst)
        po_l, pd_l = marginals(post, obj, dst)
        po_h, pd_h = marginals(prior_phys * keep, obj, dst)
        po = po_l if (empty or not stages) else po_h
        pd = pd_l if (not empty or not stages) else pd_h
        phys = product(po / po.sum(), pd / pd.sum(), obj, dst)
    elif post.sum() > 0.0:
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


def engine_likelihoods(mdp, j, s, a, lapse=None):
    """The engine's likelihoods of robot j's action a at joint state s, on probabilities:
    lik0 (PG,), lik1 (PG, nT) (zero rows for the hypotheses ruled out), l_idle.  Under
    ``Config.physical_likelihood="level1"`` lik0 is lik1 averaged over the social posterior
    (Eq. 6 as printed).  Each model's policy is computed here from its Q, by hand: Eq. 7's
    softmax at the tau robot 1 - j reads robot j at, or at ``Config.inference_temperature``
    where set; and the lapse (``Config.belief_lapse``, or ``lapse``) applied to it,
    (1 - e) psi(a) + e / |A|, A the actions the model allows."""
    e = float(mdp.cfg.belief_lapse if lapse is None else lapse)
    cfg = mdp.cfg
    scale = float(cfg.costs[0]) if cfg.temperature_unit == "move_cost" else 1.0
    tau = (mdp.taus[1 - j] if cfg.inference_temperature is None
           else float(cfg.inference_temperature) * scale)

    def lik_of(model):
        lp = log_softmax(mdp.q_values(model, s, j) / tau)       # Eq. 7 at that tau, no lapse
        allowed = np.isfinite(lp)
        if not allowed[a]:
            return 0.0
        return (1.0 - e) * float(np.exp(lp[a])) + e / int(allowed.sum())

    lik0 = np.array([lik_of((0, j * mdp.PG + g)) for g in range(mdp.PG)])
    nT = len(mdp.hyp_goals[1 - j])
    lik1 = np.zeros((mdp.PG, nT))
    for g in range(mdp.PG):
        if not np.isfinite(mdp.about[j].log_phys[g]):
            continue
        for t in range(nT):
            r = mdp.hypothesis_row(j, g, t)
            lik1[g, t] = lik_of((1, r))
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
        for name, bad in (("partner_delivers", True),
                          ("partner_parks", True), ("partner_observed", "outcome"),
                          ("social_belief", "joint"), ("child_conditioning", True)):
            with self.assertRaises(ValueError, msg=name):
                Config(**{name: bad})
        # the lapse and the inference temperature of then are options again (2026-11-04),
        # off by default (classes Lapse, InferenceTemperature)
        self.assertNotIn("belief_lapse", BELIEF_DEVICES)
        self.assertNotIn("inference_temperature", BELIEF_DEVICES)
        self.assertEqual(cfg.belief_lapse, 0.0)
        self.assertIsNone(cfg.inference_temperature)
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
                                    self.cfg.idle_hazard, stages, mdp.cfg.belief_factored,
                                    mdp.pg_dst[j])
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
        """The two posteriors apart (the default since 2026-11-04): hands empty, the belief
        about the destination does not move; holding, the belief about the object does not
        move; and the belief is their product.  (One joint posterior, ``belief_factored=
        False``: hands empty, the destination given the object does not move.)"""
        w, mdp = self.w, self.mdp
        self.assertTrue(mdp.cfg.belief_factored)
        obj, dst = mdp.pg_obj[0], mdp.pg_dst[0]
        s = w.sid(self.start)
        moved = 0.0
        for a0 in range(4):
            old = mdp.about[0]
            got = mdp.updated_belief(0, s, (a0, WAIT), w.unsid(w.step(s, (a0, WAIT))),
                                     event_at(w, self.start))
            po0, pd0 = marginals(old.phys(), obj, dst)
            po1, pd1 = marginals(got.phys(), obj, dst)
            np.testing.assert_allclose(pd1, pd0, atol=1e-12)          # the goal's: frozen
            np.testing.assert_allclose(got.phys(), product(po1, pd1, obj, dst), atol=1e-12)
            moved = max(moved, float(np.abs(po1 - po0).max()))
        self.assertGreater(moved, 1e-6)                            # the object's learns
        holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]), w.block_of([HELD0, HOME], [0, 0]))
        s = w.sid(holding)
        for a0 in range(4):
            if not np.isfinite(mdp.q_values(mdp.root[0], s, 0)[a0]):
                continue
            old = mdp.about[0]
            got = mdp.updated_belief(0, s, (a0, WAIT), w.unsid(w.step(s, (a0, WAIT))),
                                     event_at(w, holding))
            for q in range(w.M):                                    # the object's: frozen
                sel = mdp.pg_obj[0] == q
                self.assertAlmostEqual(got.phys()[sel].sum(), old.phys()[sel].sum(), places=12)
            po1, pd1 = marginals(got.phys(), obj, dst)
            np.testing.assert_allclose(got.phys(), product(po1, pd1, obj, dst), atol=1e-12)

    def test_the_stage_rule_of_one_joint_posterior(self):
        """``belief_factored=False`` (the rule until 2026-11-04): hands empty, the belief
        about the destination given each object does not move; holding, the object's."""
        cfg = tight_explicit(belief_factored=False)
        self.w, _, self.mdp, self.start, _ = solved(TINY, cfg, (2, 2),
                                                    ("cooperation", "competition"), (1.0, 1.0),
                                                    seed=24)
        try:
            w, mdp = self.w, self.mdp
            s = w.sid(self.start)
            for a0 in range(4):
                self.check(self.start, (a0, WAIT))
                old = mdp.about[0]
                got = mdp.updated_belief(0, s, (a0, WAIT), w.unsid(w.step(s, (a0, WAIT))),
                                         event_at(w, self.start))
                for q in range(w.M):
                    sel = mdp.pg_obj[0] == q
                    before, after = old.phys()[sel], got.phys()[sel]
                    np.testing.assert_allclose(after / after.sum(), before / before.sum(),
                                               atol=1e-12)
        finally:
            self.setUpClass()

    def test_the_goal_follows_the_object_that_explains_the_move(self):
        """The case of 2026-11-04 (head_on_blocked_goal, hinder / cooperation, tau 0.1): the
        hinderer sure at 0.8 that red wants O1, red carrying O2 toward G1.  Under "O1" no
        destination explains the step, the one least at odds the most (here G2); under "O2"
        G1 does.  One joint posterior believes the destination under each object apart, so
        G2 wins with the 0.8; the two posteriors apart weigh each destination by how well
        the objects together explain the step, and G1 wins."""
        w, mdp = self.w, self.mdp
        obj, dst = mdp.pg_obj[0], mdp.pg_dst[0]
        K = w.K
        po = np.array([0.8, 0.2])
        pd = np.full(K, 1.0 / K)
        prior = np.log(product(po, pd, obj, dst))
        lik = np.full(mdp.PG, 1e-4)
        lik[(obj == 0) & (dst == 1)] = 3e-4          # O1: every destination at odds, d = 1 least
        lik[(obj == 1) & (dst == 0)] = 0.9           # O2: d = 0 explains the step
        lik[(obj == 1) & (dst == 1)] = 0.05
        post = prior + np.log(lik)
        hard = prior.copy()
        got = np.exp(mdp.factored_posterior(0, post, hard, empty=False))
        _, pd_new = marginals(got, obj, dst)
        self.assertEqual(int(np.argmax(pd_new)), 0)
        po_new, _ = marginals(got, obj, dst)
        np.testing.assert_allclose(po_new, po, atol=1e-12)         # the object's frozen
        joint = np.exp(mdp._renormalize_stage(0, post, prior, False))
        _, pd_joint = marginals(joint, obj, dst)
        self.assertEqual(int(np.argmax(pd_joint)), 1)

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


class Lapse(unittest.TestCase):
    """The lapse (``Config.belief_lapse``, 2026-11-04; the user: "lets go ahead an implement
    this: A lapse: psi~ = (1 - e) psi + e / |A| ... It caps what one odd move can cost"):
    the option, the lapsed policies, the update against the reference (which applies the
    lapse by hand to each model's policy), and the bound on what one move can do."""

    E = 0.05

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit(belief_lapse=cls.E)
        cls.w, _, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "competition"), (1.0, 1.0),
                                                 seed=21)

    def test_the_option(self):
        self.assertEqual(Config().belief_lapse, 0.0)       # Eqs. 5-6 as printed by default
        Config(belief_lapse=0.05)
        Config(belief_lapse=0.0)
        for bad in (1.0, -0.01, 1.5, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Config(belief_lapse=bad)

    def test_the_lapsed_policies(self):
        """(1 - e) psi + e / |A| over the actions the model allows, -inf elsewhere: still a
        distribution, and never below e / |A| where an action is allowed."""
        w, mdp, e = self.w, self.mdp, self.E
        s = w.sid(self.start)
        seen = 0
        for j in (0, 1):
            for g in range(mdp.PG):
                r = mdp.hypothesis_row(j, g, 0)
                lp = mdp._log_policy((1, r), s, j)
                got = mdp.lapsed(lp)
                allowed = np.isfinite(lp)
                np.testing.assert_array_equal(np.isfinite(got), allowed)
                n = int(allowed.sum())
                want = (1.0 - e) * np.exp(lp[allowed]) + e / n
                np.testing.assert_allclose(np.exp(got[allowed]), want, rtol=1e-12)
                self.assertAlmostEqual(float(np.exp(got[allowed]).sum()), 1.0, places=12)
                self.assertGreaterEqual(float(np.exp(got[allowed]).min()), e / n * (1 - 1e-12))
                seen += 1
        self.assertGreater(seen, 0)
        # without the lapse the policy as it is
        mdp0 = solved(TINY, tight_explicit(), (2, 2), ("cooperation", "competition"),
                      (1.0, 1.0), seed=21)[2]
        lp = mdp0._log_policy((1, mdp0.hypothesis_row(0, 0, 0)), s, 0)
        np.testing.assert_array_equal(mdp0.lapsed(lp), lp)

    def check(self, state, actions):
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
                continue                 # not a legal action here
            old = mdp.about[j]
            lik0, lik1, l_idle = engine_likelihoods(mdp, j, s, a)        # lapsed by hand
            keep = mdp.transition_term(j, s, actions, nxt, event)
            want = reference_update(old.phys(), old.social(), old.idle, lik0, lik1, l_idle,
                                    keep, int(w.hold[state[2], j]) < 0, mdp.pg_obj[j],
                                    self.cfg.idle_hazard, True, mdp.cfg.belief_factored,
                                    mdp.pg_dst[j])
            got = mdp.updated_belief(j, s, actions, nxt, event)
            np.testing.assert_allclose(got.phys(), want[0], atol=1e-12)
            np.testing.assert_allclose(got.social(), want[1], atol=1e-12)
            self.assertAlmostEqual(got.idle, want[2], places=12)
            # one move moves no hypothesis by more than |A| / e against another: every
            # hypothesis explains a move at least (1 - idle) e / 10, none more than 1 (a move
            # is open to every model; the transition term rules nothing out at a move) -- in
            # the posterior that learns (the object's or the destination's) and the social one
            if a < PICK and keep.all():
                obj, dst = mdp.pg_obj[j], mdp.pg_dst[j]
                pairs = list(zip(marginals(got.phys(), obj, dst), marginals(old.phys(), obj, dst)))
                for new_b, old_b in pairs + [(got.social(), old.social())]:
                    held = old_b > 1e-300
                    ratio = new_b[held] / old_b[held]
                    self.assertGreaterEqual(ratio.min() / ratio.max(),
                                            (1.0 - old.idle) * self.E / 10 * (1 - 1e-9))
            checked += 1
        return checked

    def test_against_the_reference(self):
        w = self.w
        n = sum(self.check(self.start, (a0, a1)) for a0 in range(5) for a1 in range(5))
        holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]),
                   w.block_of([HOME, HELD0], [0, 0]))
        n += sum(self.check(holding, (a0, a1)) for a0 in (0, 1, 2, 3, WAIT, PLACE, GIVE)
                 for a1 in (0, WAIT, PICK))
        self.assertGreater(n, 40)

    def test_what_one_detour_costs(self):
        """Of the moves at the start, the one some hypothesis explains least, against the
        hypothesis that explains it best: with the lapse never less than e / |A| (here
        |A| <= 10); without it, what the softmax gives (less, where a hypothesis rates the
        move badly)."""
        w, mdp = self.w, self.mdp
        mdp0 = solved(TINY, tight_explicit(), (2, 2), ("cooperation", "competition"),
                      (1.0, 1.0), seed=21)[2]
        s = w.sid(self.start)
        worst = {}
        for name, m in (("lapse", mdp), ("none", mdp0)):
            low = 1.0
            for a in range(5):
                if not np.isfinite(m.q_values(m.root[1], s, 1)[a]):
                    continue
                lik = np.exp(m.hypothesis_log_policies(1, s, a, m.about[1].log_phys))
                lik = lik[np.isfinite(m.about[1].log_phys)]
                low = min(low, float(lik.min() / lik.max()))
            worst[name] = low
        self.assertGreaterEqual(worst["lapse"], self.E / 10 * (1 - 1e-9))
        self.assertLess(worst["none"], self.E / 10)        # some hypothesis rates a move badly
        self.assertGreater(worst["lapse"], worst["none"])


class InferenceTemperature(unittest.TestCase):
    """A temperature of the belief update's own (``Config.inference_temperature``,
    2026-11-04; the user: "Ok lets implement this: Tempered likelihood P^eta, eta = tau /
    tau_inf = 0.2 (tau_inf = 0.5 c(move) ...)"): the option, the likelihood's policy (the
    softmax at tau_inf, psi^eta renormalised), the update against the reference, and the
    robots' own models untouched."""

    T = 0.5

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_explicit(inference_temperature=cls.T)
        cls.w, _, cls.mdp, cls.start, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "competition"), (1.0, 1.0),
                                                 seed=21)
        cls.cfg0 = tight_explicit()
        cls.mdp0 = solved(TINY, cls.cfg0, (2, 2), ("cooperation", "competition"), (1.0, 1.0),
                          seed=21)[2]

    def test_the_option(self):
        self.assertIsNone(Config().inference_temperature)  # Eq. 7's tau by default
        cfg = Config(temperature=0.1, inference_temperature=0.5)
        self.assertAlmostEqual(cfg.tau, 0.1 * cfg.costs[0])
        self.assertAlmostEqual(cfg.tau_inference, 0.5 * cfg.costs[0])
        self.assertAlmostEqual(cfg.inference_tau_of(1), 0.5 * cfg.costs[0])
        two = Config(robot_temperatures=(0.1, 0.3))
        self.assertEqual((two.inference_tau_of(0), two.inference_tau_of(1)),
                         (two.tau_of(0), two.tau_of(1)))            # None: each robot's own
        two = Config(robot_temperatures=(0.1, 0.3), inference_temperature=0.5)
        self.assertAlmostEqual(two.inference_tau_of(0), two.inference_tau_of(1))
        for bad in (0.0, -0.5, float("inf"), float("nan")):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Config(inference_temperature=bad)

    def test_the_likelihood_policy(self):
        """softmax(Q / tau_inf) = psi^eta / sum psi^eta, psi Eq. 7's at tau, eta = tau /
        tau_inf; the planning taus as they were."""
        w, mdp, mdp0 = self.w, self.mdp, self.mdp0
        s = w.sid(self.start)
        tau_inf = self.T * float(self.cfg.costs[0])
        self.assertEqual(mdp.taus, mdp0.taus)
        np.testing.assert_array_equal(mdp.read_inv_tau, mdp0.read_inv_tau)
        self.assertEqual(mdp.inference_taus, (tau_inf, tau_inf))
        self.assertEqual(mdp0.inference_taus, mdp0.taus)
        for j in (0, 1):
            eta = mdp.taus[1 - j] / tau_inf
            for g in range(mdp.PG):
                for model in ((0, j * mdp.PG + g), (1, mdp.hypothesis_row(j, g, 0))):
                    q = mdp.q_values(model, s, j)
                    got = mdp._log_policy(model, s, j)
                    np.testing.assert_allclose(got, log_softmax(q / tau_inf), atol=1e-12)
                    psi = np.exp(mdp0._log_policy(model, s, j))               # at Eq. 7's tau
                    want = psi ** eta / (psi ** eta).sum()
                    np.testing.assert_allclose(np.exp(got), want, atol=1e-10)

    def test_the_robots_own_models_are_untouched(self):
        """Only the update reads the models at tau_inf: solved from the same beliefs, every
        value table is what it is without the option."""
        mdp, mdp0 = self.mdp, self.mdp0
        for bank, bank0 in ((mdp.L0, mdp0.L0), (mdp.L1, mdp0.L1), (mdp.L2, mdp0.L2)):
            np.testing.assert_allclose(np.asarray(bank.V), np.asarray(bank0.V), atol=1e-6)
            np.testing.assert_allclose(np.asarray(bank.psi), np.asarray(bank0.psi), atol=1e-6)

    def check(self, mdp, state, actions):
        w = self.w
        s = w.sid(state)
        nxt = w.unsid(w.step(s, actions))
        event = event_at(w, state)
        checked = 0
        for j in (0, 1):
            if w.busy[state[2], j]:
                continue
            a = int(actions[j])
            if not np.isfinite(mdp.q_values(mdp.root[j], s, j)[a]):
                continue
            old = mdp.about[j]
            lik0, lik1, l_idle = engine_likelihoods(mdp, j, s, a)     # at tau_inf, by hand
            keep = mdp.transition_term(j, s, actions, nxt, event)
            want = reference_update(old.phys(), old.social(), old.idle, lik0, lik1, l_idle,
                                    keep, int(w.hold[state[2], j]) < 0, mdp.pg_obj[j],
                                    mdp.cfg.idle_hazard, True, mdp.cfg.belief_factored,
                                    mdp.pg_dst[j])
            got = mdp.updated_belief(j, s, actions, nxt, event)
            np.testing.assert_allclose(got.phys(), want[0], atol=1e-12)
            np.testing.assert_allclose(got.social(), want[1], atol=1e-12)
            self.assertAlmostEqual(got.idle, want[2], places=12)
            checked += 1
        return checked

    def test_against_the_reference(self):
        w = self.w
        holding = (int(w.cell_id[(1, 1)]), int(w.cell_id[(2, 1)]),
                   w.block_of([HOME, HELD0], [0, 0]))
        n = sum(self.check(self.mdp, self.start, (a0, a1)) for a0 in range(5) for a1 in range(5))
        n += sum(self.check(self.mdp, holding, (a0, a1)) for a0 in (0, 1, 2, 3, WAIT, PLACE, GIVE)
                 for a1 in (0, WAIT, PICK))
        self.assertGreater(n, 40)

    def test_with_the_lapse(self):
        """Both options at once: the lapse on the policy at tau_inf."""
        cfg = tight_explicit(inference_temperature=self.T, belief_lapse=0.05)
        mdp = solved(TINY, cfg, (2, 2), ("cooperation", "competition"), (1.0, 1.0), seed=21)[2]
        n = sum(self.check(mdp, self.start, (a0, a1)) for a0 in range(5) for a1 in range(5))
        self.assertGreater(n, 20)

    def test_softer_than_eq_7(self):
        """Of the moves at the start, what the hypothesis that explains a move least makes of
        it against the one that explains it best: less extreme at tau_inf."""
        w, mdp, mdp0 = self.w, self.mdp, self.mdp0
        s = w.sid(self.start)
        worst = []
        for m in (mdp, mdp0):
            low = 1.0
            for a in range(5):
                if not np.isfinite(m.q_values(m.root[1], s, 1)[a]):
                    continue
                lik = np.exp(m.hypothesis_log_policies(1, s, a, m.about[1].log_phys))
                lik = lik[np.isfinite(m.about[1].log_phys)]
                low = min(low, float(lik.min() / lik.max()))
            worst.append(low)
        self.assertGreater(worst[0], worst[1])


class Rollout(unittest.TestCase):
    """A rollout of the tiny scene, every posterior of every transition against the
    reference, and the record's beliefs consistent with the engine's."""

    def test_rollout_against_the_reference(self):
        self.roll_against_the_reference(
            tight_now(value_tol=1e-5, value_maps=False, deadlock_window=6, livelock_window=0))

    def test_rollout_against_the_reference_with_a_lapse(self):
        """The same under the lapse (``Config.belief_lapse``, 2026-11-04), which the
        reference applies by hand to every model's policy (engine_likelihoods)."""
        self.roll_against_the_reference(
            tight_now(value_tol=1e-5, value_maps=False, deadlock_window=6, livelock_window=0,
                      belief_lapse=0.05))

    def test_rollout_against_the_reference_softened(self):
        """... and with the likelihood's own temperature (``Config.inference_temperature``,
        2026-11-04) as well as the lapse."""
        self.roll_against_the_reference(
            tight_now(value_tol=1e-5, value_maps=False, deadlock_window=6, livelock_window=0,
                      belief_lapse=0.05, inference_temperature=0.5))

    def roll_against_the_reference(self, cfg):
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
                                             mdp.pg_obj[j], cfg.idle_hazard, True,
                                             cfg.belief_factored, mdp.pg_dst[j]))
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
