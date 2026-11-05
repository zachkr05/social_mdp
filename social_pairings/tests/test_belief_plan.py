"""Planning over the partner's belief (Config.belief_planning, social_pairings/beliefplan.py,
2026-11-05) and the coercer's task objective (Config.objective): the replica of the partner's
update against SocialMDP.updated_belief, the Q tables against q_values, the partner's choice
with its inertia, the reward, the timing of the plan, a rollout.
"""
import json
import unittest

import numpy as np

from social_pairings.beliefkernels import NO_UNDO, choice_tables, undo_code
from social_pairings.beliefplan import BeliefPlanner
from social_pairings.config import Config
from social_pairings.constants import HOME, N_ACTIONS, PICK
from social_pairings.pairing import run_pairing
from social_pairings.scenes import Scene
from social_pairings.serialize import to_json
from social_pairings.tests.test_models import TINY, hostage, solved, tight_explicit
from social_pairings.util import block_items, choose_with_inertia, undo_of

PAIR = ("hinder", "coercion")      # R0 hinders, R1 coerces; R0 holds R1's object (hostage)


def planner(state=hostage, seed=5, **options):
    options.setdefault("belief_planning", "updated")
    options.setdefault("belief_partner", "oracle")
    cfg = tight_explicit(partner_object="inferred", inertia=1.0, swap_on_place=True, steal=True,
                         **options)
    w, gid, mdp, st, _ = solved(TINY, cfg, (2, 2), PAIR, (1.0, 1.0), state, seed=seed)
    pl = BeliefPlanner(mdp, 30)
    blocks = mdp.reach_of(1, mdp.own_goal[1], int(st[2]))
    ok = np.zeros_like(w.state_ok)
    ok[blocks] = w.state_ok[blocks]
    _, lo, hi = block_items(w.NN, blocks, 0)
    return w, mdp, st, pl, ok, lo, hi


def valid_states(w, ok, n, seed=0):
    states = [s for s in range(w.S) if ok[s // w.NN, s - (s // w.NN) * w.NN]]
    rng = np.random.default_rng(seed)
    return [int(s) for s in rng.choice(states, size=min(n, len(states)), replace=False)]


def event_of(w, s, act):
    """The belief update's event of a transition, as the rollout makes it (real_step)."""
    st = w.unsid(s)
    nxt = w.unsid(w.step(s, act))
    status = tuple(int(v) for v in w.status[st[2]])
    done = tuple(int(v) for v in w.done[st[2]])
    _, nd, failed, handed, _ = w.real_step(status, done, st, nxt, act, detail=True)
    return nxt, dict(done_now=tuple(bool(nd[i]) and not done[i] for i in (0, 1)), failed=failed,
                     handed_in_vain=handed, held=tuple(int(w.hold[st[2], i]) for i in (0, 1)))


def close(a, b):
    """The largest difference of two log beliefs (-inf where both are), or inf where their
    supports differ."""
    if a is None or b is None:
        return 0.0 if a is None and b is None else np.inf
    fa, fb = np.isfinite(a), np.isfinite(b)
    if (fa != fb).any():
        return np.inf
    return float(np.abs(a[fa] - b[fa]).max()) if fa.any() else 0.0


class Options(unittest.TestCase):
    def test_validation(self):
        with self.assertRaises(ValueError):
            Config(objective="complete")                      # the planner's only
        with self.assertRaises(ValueError):
            Config(belief_planning="updated", physical_likelihood="cut")
        with self.assertRaises(ValueError):
            Config(belief_planning="updated", belief_window=3)
        with self.assertRaises(ValueError):
            Config(belief_planning="frozen", decoy_feint=True)
        with self.assertRaises(ValueError):
            Config(belief_planning="frozen", action_selection="sample")
        with self.assertRaises(ValueError):
            Config(belief_planning="maybe")
        Config(belief_planning="frozen", physical_likelihood="cut")   # frozen needs no update
        Config(objective="complete", belief_planning="frozen")

    def test_undo_code(self):
        for a in range(N_ACTIONS):
            for moved in (False, True):
                for changed in (False, True):
                    want = undo_of(a, moved, changed)
                    self.assertEqual(undo_code(a, moved, changed),
                                     NO_UNDO if want is None else want)


class TaskObjective(unittest.TestCase):
    """objective="task": the partner's estimate of the coercer's goal reward inside R_j is its
    true goal reward -- against a goal-less hinderer R = (1 + w + w w_j) r(g) - c."""

    def rewards(self, objective):
        cfg = tight_explicit(partner_object="inferred", partner_social_goal="known",
                             objective=objective)
        w, gid, mdp, state, _ = solved(TINY, cfg, (2, 2), PAIR, (1.0, 1.0), hostage, seed=3)
        out = []
        s = w.sid(state)
        for a1 in range(N_ACTIONS):
            act = (4, a1)
            nxt = w.unsid(w.step(s, act))
            own = w.rate(nxt, 1, gid[1])
            cost = float(w.cost_c[state[2], 1, a1])
            out.append((mdp.modeled_reward(1, s, act, nxt, own - cost), own, cost, nxt, mdp))
        return out

    def test_task(self):
        for got, own, cost, _, _ in self.rewards("task"):
            self.assertAlmostEqual(got, 3.0 * own - cost, places=9)

    def test_table(self):
        """The default is Table I: (1 + w) r(g) + w w_j r~(g) - c, r~ under the partner's
        belief about the coercer."""
        for got, own, cost, nxt, mdp in self.rewards("table"):
            w = mdp.w
            est = w.goal_rates(nxt, 1, mdp.pg_obj[1], mdp.pg_dst[1], estimate=True)
            rbar = float((mdp.about[1].phys() * est).sum())
            self.assertAlmostEqual(got, 2.0 * own + rbar - cost, places=9)

    def test_the_reward_tables_of_the_model(self):
        """The root's expected rewards (the solver's R) read the true goal under "task"."""
        cfg = tight_explicit(partner_object="inferred", partner_social_goal="known",
                             objective="task")
        w, gid, mdp, state, _ = solved(TINY, cfg, (2, 2), PAIR, (1.0, 1.0), hostage, seed=3)
        nested = mdp.nested_phys(1)
        self.assertEqual(nested[mdp.own_goal[1]], 1.0)
        self.assertEqual(nested.sum(), 1.0)
        self.assertFalse(mdp.task_objective(0))        # the hinderer's reward as it was


class Tables(unittest.TestCase):
    def test_q_tables_are_q_values(self):
        w, mdp, st, pl, ok, lo, hi = planner()
        Q, _, _ = pl._partner_tables(mdp, 1, None, ok, lo, hi, None)
        L1 = mdp.L1
        row = mdp.hypothesis_row(1, int(np.argmax(mdp.about[1].log_phys)), 0)
        Qh = pl._q(mdp, L1, row, L1.R, L1.P, L1.V, row, int(L1.p_of[row]), L1.pd, L1.ho, ok,
                   lo, hi)
        for s in valid_states(w, ok, 400):
            for got, model, robot in ((Q, mdp.root[0], 0), (Qh, (1, row), 1)):
                want = mdp.q_values(model, s, robot)
                self.assertTrue((np.isfinite(want) == np.isfinite(got[s])).all())
                fin = np.isfinite(want)
                self.assertLess(np.abs(want[fin] - got[s][fin]).max(initial=0.0), 1e-9)

    def test_the_partner_choice_with_inertia(self):
        w, mdp, st, pl, ok, lo, hi = planner()
        Q, _, _ = pl._partner_tables(mdp, 1, None, ok, lo, hi, None)
        best = np.empty(w.S, dtype=np.int8)
        alt = np.empty(w.S, dtype=np.int8)
        margin = mdp.cfg.inertia_margin
        choice_tables(Q, margin, best, alt)
        for s in valid_states(w, ok, 300):
            q = mdp.q_values(mdp.root[0], s, 0)
            for u in [None] + list(range(N_ACTIONS)):
                want, _ = choose_with_inertia(q, u, margin)
                code = NO_UNDO if u is None else u
                got = alt[s] if code == best[s] else best[s]
                self.assertEqual(int(got), want)

    def test_the_partner_re_solved_at_its_own_belief_is_as_solved(self):
        """Re-solving the partner's root on scratch arrays at the belief it holds gives the
        rollout's Q."""
        w, mdp, st, pl, ok, lo, hi = planner()
        b = mdp.about[1]
        Q0, _, _ = pl._partner_tables(mdp, 1, None, ok, lo, hi, None)
        Q1, _, stats = pl._partner_tables(mdp, 1, (b.log_phys, b.log_social, b.idle), ok, lo,
                                          hi, None)
        self.assertTrue(stats[3] > 0.5)                # converged
        fin = np.isfinite(Q0)
        self.assertTrue((fin == np.isfinite(Q1)).all())
        self.assertLess(np.abs(Q0[fin] - Q1[fin]).max(), 1e-4)


class Replica(unittest.TestCase):
    """beliefkernels.belief_update is SocialMDP.updated_belief (the partner's own update)."""

    def compare(self, n=150, state=hostage, seed=5, **options):
        w, mdp, st, pl, ok, lo, hi = planner(state=state, seed=seed, **options)
        LL = pl._likelihoods(mdp, 1, ok, lo, hi)
        b = mdp.about[1]
        checked = 0
        for s in valid_states(w, ok, n) + [w.sid(st)]:
            qp = mdp.q_values(mdp.root[0], s, 0)
            for a1 in np.flatnonzero(pl.legal[1][s]).tolist():
                for a0 in np.flatnonzero(np.isfinite(qp)).tolist():
                    act = (a0, a1)
                    nxt, event = event_of(w, s, act)
                    ref = mdp.updated_belief(1, s, act, nxt, event)
                    lp, ls, q = pl.step_belief(mdp, 1, LL, (b.log_phys, b.log_social, b.idle),
                                               s, a1, a0)
                    self.assertLess(close(ref.log_phys, lp), 1e-9, (s, act))
                    self.assertLess(close(ref.log_social, ls), 1e-9, (s, act))
                    self.assertAlmostEqual(ref.idle, q, places=12)
                    checked += 1
        self.assertGreater(checked, 100)

    def test_default(self):
        self.compare()

    def test_hazards(self):
        self.compare(goal_hazard=0.02, social_hazard=0.01, hand_hazard=0.1)

    def test_lapse_power_tau(self):
        self.compare(belief_lapse=0.05, likelihood_power=0.7, inference_temperature=0.5)

    def test_joint(self):
        self.compare(belief_factored=False)
        self.compare(belief_factored=False, belief_stages=False, goal_hazard=0.05)

    def test_no_idle(self):
        self.compare(idle_hypothesis=False)

    def test_the_first_pick_counts_the_lift_does_not(self):
        """The advisor's point A: the free choice to pick an object up is evidence about which
        object the robot wants (its hands empty before it); the forced steps of the lift are
        not (nothing is learnt while it lifts)."""
        w, mdp, st, pl, ok, lo, hi = planner()
        LL = pl._likelihoods(mdp, 1, ok, lo, hi)
        b = mdp.about[1]
        b0 = (b.log_phys, b.log_social, b.idle)
        at = lambda c: int(w.cell_id[c])                   # noqa: E731
        free = w.block_of([HOME, HOME], [0, 0])            # R0's object (O1) lies at (2, 2)
        s = w.sid((at((0, 1)), at((2, 2)), free))          # R1 on it, hands empty
        self.assertTrue(pl.legal[1][s][PICK])
        lp, _, _ = pl.step_belief(mdp, 1, LL, b0, s, PICK, 4)
        lo0, _ = mdp.factors(1, b.log_phys)
        lo1, _ = mdp.factors(1, lp)
        self.assertGreater(np.exp(lo1[0]), np.exp(lo0[0]))   # it wants O1, more than before
        s2 = int(w.step(s, (4, PICK)))
        c2 = s2 // w.NN
        self.assertTrue(w.busy[c2, 1])                     # lifting now
        lp2, ls2, q2 = pl.step_belief(mdp, 1, LL, (lp, b.log_social, b.idle), s2, PICK, 4)
        self.assertEqual(close(lp2, lp), 0.0)              # the lift's own steps: nothing
        self.assertEqual(close(ls2, b.log_social), 0.0)


class Decision(unittest.TestCase):
    def test_a_decision(self):
        w, mdp, st, pl, ok, lo, hi = planner(belief_horizon=6, belief_points=8)
        undo = (None, None)
        q = pl.decide(mdp, 0, st, undo)[1]
        rec = pl.decisions[-1]
        self.assertTrue(np.isfinite(q).any())
        self.assertTrue((np.isfinite(q) == pl.legal[1][w.sid(st)]).all())
        want, _ = choose_with_inertia(mdp.q_values(mdp.root[0], w.sid(st), 0), None,
                                      mdp.cfg.inertia_margin)
        self.assertEqual(rec["partner_predicted"], want)   # the partner's choice, exactly
        fin = np.isfinite(q)
        np.testing.assert_allclose(rec["delta_belief"][fin], (q - rec["q_hold"])[fin])
        self.assertGreaterEqual(rec["points"], 1)
        self.assertLessEqual(rec["points"], 8)
        json.dumps(to_json(pl.record()))

    def test_frozen_horizon_one_is_the_tail(self):
        """Frozen, one transition: R + gamma V_tail(s'), the tail robot r's best answer to the
        partner's greedy choice at the belief it holds."""
        w, mdp, st, pl, ok, lo, hi = planner(belief_planning="frozen", belief_horizon=1)
        q = pl.decide(mdp, 0, st, (None, None))[1]
        self.assertTrue(np.isfinite(q).any())
        rec = pl.decisions[-1]
        np.testing.assert_allclose(q[np.isfinite(q)], rec["q_frozen"][np.isfinite(q)])


class StealRobots(unittest.TestCase):
    """Config.steal_robots: with (True, False) the first robot takes what its partner holds,
    the second does not."""

    def test_only_the_first_takes(self):
        from social_pairings.constants import TAKE
        from social_pairings.tests.reference import make_world
        for robots, r1_takes in (((True, True), True), ((True, False), False)):
            cfg = tight_explicit(steal=True, steal_robots=robots)
            w, gid = make_world(TINY, cfg)
            st = hostage(w)                       # R0 holds R1's object, next to R1
            s = w.sid(st)
            nxt = w.unsid(w.step(s, (4, TAKE)))
            self.assertEqual(int(w.hold[nxt[2], 1]) == 1, r1_takes)
            ok = w.hand_ok(False)[1, st[2]]
            site = int(w.site_of[st[1]])
            self.assertEqual(bool((ok[site * 2 + 1] >> (TAKE - PICK)) & 1), r1_takes)


class Rollout(unittest.TestCase):
    def test_tiny_rollout(self):
        scene = Scene(name="tiny", **TINY)
        for mode in ("frozen", "updated"):
            cfg = tight_explicit(partner_object="inferred", inertia=1.0, swap_on_place=True,
                                 steal=True, belief_planning=mode, objective="task",
                                 belief_horizon=6, belief_points=6, belief_refinements=1,
                                 value_maps=False, belief_detail=False, belief_partner="oracle")
            res = run_pairing(scene, PAIR, levels=(2, 2), cfg=cfg, n_steps=5)
            bp = res["belief_planner"]
            self.assertEqual(bp["robots"], [1])
            self.assertEqual(bp["mode"], mode)
            for d, row in zip(bp["decisions"], res["trace"]):
                self.assertEqual(d["chosen"], row["action"][1])
                self.assertTrue(d["partner_as_predicted"])     # the oracle: always right
                self.assertIsNotNone(row["belief_plan"][1])
            json.dumps(to_json(res))


if __name__ == "__main__":
    unittest.main()
