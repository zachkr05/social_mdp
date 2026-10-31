"""Which object the partner wants, inferred (``Config.partner_object``, since 2026-10-09).

The user: "Modify it such that the pickup objects are always inferred as per the paper" --
every object of the scene a candidate for the partner's goal, the robot's own included, and
spare objects too.  A physical goal is an object and a destination, and a robot's estimate
of its partner's starts uniform over all of them (Sec. III-B, Eq. 6).  The engine is checked
against brute force where the models are (``test_models``: the defaults of now; the hands
rule: ``test_hands``); here is what is new besides: the option, the world, the prior, the
record, what an act for the partner that did not count tells, the partner's estimate of a
robot's reward once the robot's own goal is done, the memory guard and the figure.  The
level-2 comparison on the open map only with SOCIAL_PAIRINGS_SLOW=1.
"""
import os
import unittest

import numpy as np

from social_pairings import TOLD, Scene, SolverFailure, six_object_scene
from social_pairings import goals as G
from social_pairings.constants import GIVE, GONE, HELD0, HOME, PLACE, WAIT
from social_pairings.models import SocialMDP
from social_pairings.render import _callout, _Canvas, event_text, frames_of
from social_pairings.rollout import build_world
from social_pairings.util import configure_threads, memory_limit_bytes

from . import test_models as TM
from .hands_before import Config, run_pairing
from .reference import Brute, make_world
from .test_render import disagreements

SLOW = os.environ.get("SOCIAL_PAIRINGS_SLOW", "") not in ("", "0")
NAMES = ("agent_1", "agent_2")
TINY = TM.TINY                                   # 3 x 3, candidates G0 (1, 0), G1 (0, 2), G2 (2, 0)
TINY_SCENE = Scene(name="tiny", **TINY)
SPARE = dict(TINY, objects=((1, 2),))            # ... with a spare object O3 on (1, 2)


def tiny(**options) -> Config:
    """The package's defaults on the 1 m grid of the tiny map."""
    return Config(num_threads=2, resolution=1.0, **options)


def at(w, x0, x1, status, done=(0, 0)) -> tuple:
    return int(w.cell_id[x0]), int(w.cell_id[x1]), w.block_of(list(status), list(done))


class TheOption(unittest.TestCase):
    def test_inferred_by_default_known_when_told(self):
        self.assertEqual(Config().partner_object, "inferred")
        self.assertEqual(Config(**TOLD).partner_object, "known")
        with self.assertRaises(ValueError):
            Config(partner_object="guessed")


class EveryObjectIsACandidate(unittest.TestCase):
    def world(self, scene, **options):
        cfg = tiny(**options)
        w, goal_ids, _, scenery = build_world(
            scene["grid"], scene["starts"], scene["pickups"], scene["destinations"],
            scene["candidate_cells"], scene.get("objects", "goal"), cfg)
        return w, goal_ids, scenery, cfg

    def test_a_spare_object_is_in_the_model(self):
        """Inferred, a spare object may be what the partner wants: it is in the model (and
        may be picked up); known, it is scenery unless handle_spares."""
        w, _, scenery, _ = self.world(SPARE)
        self.assertEqual((w.M, scenery), (3, ()))
        w, _, scenery, _ = self.world(SPARE, partner_object="known")
        self.assertEqual((w.M, scenery), (2, ((1, 2),)))

    def test_objects_left_out_by_max_objects_are_scenery(self):
        scene = six_object_scene()
        cfg = Config(num_threads=2, max_objects=3)
        w, _, _, scenery = build_world(scene.grid, scene.starts, scene.pickups,
                                       scene.destinations, scene.candidate_cells, scene.objects,
                                       cfg)
        self.assertEqual((w.M, len(scenery)), (3, 3))
        self.assertEqual(set(w.objects[2:]) | set(scenery), set(scene.object_cells[2:]))
        self.assertEqual(tuple(w.objects[:2]), scene.pickups)

    def test_the_prior(self):
        """PG = M K physical goals, g = q K + k; a robot's own goal is its own object to its
        destination; the level-2 belief starts uniform over (object, destination, social
        goal), so each marginal is uniform."""
        w, goal_ids, _, cfg = self.world(SPARE)
        mdp = SocialMDP(w, goal_ids, (2, 2), (G.COOPERATION, G.COMPETITION), (1.0, 1.0), cfg,
                        configure_threads(2))
        M, K, n = w.M, w.K, len(cfg.social_hypotheses)
        self.assertEqual((M, K, mdp.PG), (3, 3, 9))
        for i in (0, 1):
            self.assertEqual(list(zip(mdp.pg_obj[i].tolist(), mdp.pg_dst[i].tolist())),
                             [(q, k) for q in range(M) for k in range(K)])
            g = mdp.own_goal[i]
            self.assertEqual((int(mdp.pg_obj[i, g]), int(mdp.pg_dst[i, g])), (i, goal_ids[i]))
            slot = mdp.root_slot(i)
            self.assertEqual(slot.hyp_goal.size, M * K * n)
            self.assertTrue(np.allclose(mdp.planning_weights(slot), 1.0 / (M * K * n)))
            self.assertTrue(np.allclose(mdp.object_belief(i), 1.0 / M))
            self.assertTrue(np.allclose(mdp.goal_belief(i), 1.0 / K))
        # known: the partner's own object to each destination
        w, goal_ids, _, cfg = self.world(TINY, partner_object="known")
        mdp = SocialMDP(w, goal_ids, (2, 2), (G.COOPERATION, G.COMPETITION), (1.0, 1.0), cfg,
                        configure_threads(2))
        self.assertEqual(mdp.PG, w.K)
        self.assertEqual(mdp.pg_obj.tolist(), [[0] * w.K, [1] * w.K])
        self.assertTrue(np.allclose(mdp.object_belief(0), [0.0, 1.0]))


class TheRecord(unittest.TestCase):
    """Two cooperators of Table I (``cooperate_with="all"``, the default until 2026-10-11) on
    the tiny map.  Told which object the other wants, each fetches the other's object and
    sets it down on the other's destination (10 / 6; 7 / 5 with "use", until 2026-10-16).
    Inferring it, with the two posteriors of 2026-10-17 (15 / 19): agent_2 picks agent_1's
    object up at 4 and, after two set-downs in vain, sets it down on agent_1's destination
    at 15; agent_1 picks agent_2's object up at 2 and puts it back at 6, and agent_2
    delivers its own (19).  Both end up knowing which object the other wants.  (With one
    posterior, 2026-10-16: 18 / never, each sure that the other wants agent_1's object --
    agent_2 rightly, agent_1 wrongly; 12 / 21 when this was written.)

    Under the update of 2026-10-30 (Eqs. 5-6 and nothing else; the README, "The belief
    update of 2026-10-30"), inferring it: 10 / 6 -- agent_1 picks agent_2's object up at 2,
    sets it down in vain at 4 and on agent_2's destination at 6, then fetches its own (8)
    and delivers it (10); agent_2 fetches nothing, and both end up knowing the other's
    object.  Told the object: 22 / 17 -- each takes the other's object (2, 5) and, neither
    learning the other's destination from a partner that only ever acts for it, tries wrong
    candidates (8, 10, 11) until agent_1's set-down counts (17); agent_2 hands agent_1 its
    object (18) and agent_1 delivers it (22)."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(TINY_SCENE, ("cooperation", "cooperation"), weights=(1.0, 1.0),
                                 cfg=tiny(cooperate_with="all"), n_steps=25)
        cls.known = run_pairing(TINY_SCENE, ("cooperation", "cooperation"), weights=(1.0, 1.0),
                                cfg=tiny(partner_object="known", cooperate_with="all"),
                                n_steps=25)

    def test_the_object_beliefs(self):
        r = self.result
        self.assertEqual(r["config"]["partner_object"], "inferred")
        self.assertEqual(r["done_at"], (10, 6))
        self.assertEqual([e[:4] for e in r["events"]],
                         [(2, "picked up", 0, 1), (4, "set down in vain", 0, 1),
                          (6, "set down", 0, 1), (8, "picked up", 0, 0), (10, "set down", 0, 0)])
        M = r["scene"]["modelled_objects"]
        for row in r["trace"]:
            for key in ("object_belief", "next_object_belief"):
                self.assertEqual(len(row[key]), 2)
                for b in row[key]:
                    self.assertEqual(len(b), M)
                    self.assertAlmostEqual(float(np.sum(b)), 1.0, places=9)
        first = r["trace"][0]["object_belief"]
        self.assertTrue(np.allclose(first, 1.0 / M))
        final = r["final_object_belief"]
        self.assertGreater(final[0][1], 0.99)    # agent_1: agent_2 wants O2
        self.assertGreater(final[1][0], 0.99)    # agent_2: agent_1 wants O1

    def test_a_record_with_known_objects_has_none(self):
        k = self.known
        self.assertEqual(k["done_at"], (22, 17))
        self.assertEqual([e[:4] for e in k["events"]],
                         [(2, "picked up", 0, 1), (5, "picked up", 1, 0),
                          (8, "set down in vain", 1, 0), (10, "set down in vain", 0, 1),
                          (11, "set down in vain", 1, 0), (17, "set down", 0, 1),
                          (18, "handed over", 1, 0), (22, "set down", 0, 0)])
        self.assertNotIn("object_belief", k["trace"][0])
        self.assertNotIn("final_object_belief", k)

    def test_the_figure(self):
        """The belief panel holds both: each robot's belief about which object its partner
        wants (O1, O2, the partner's own outlined), then about its destination (G0..G2, the
        true one outlined); the bars are the record's."""
        r = self.result
        canvas = _Canvas(r, NAMES, "relative", "", 25, 3)
        try:
            self.assertTrue(canvas.story.objects_inferred and canvas.story.inferred)
            M, K = canvas.story.n_objects, canvas.K
            self.assertEqual((M, K), (2, 3))
            frames = frames_of(r)
            truth = r["scene"]["goal_ids"]
            for f in (0, 4, len(frames) - 1):
                canvas.draw(f)
                bars = {}
                for patch in canvas.ax_belief.patches:
                    centre = patch.get_x() + 0.5 * patch.get_width()
                    x = int(round(centre))
                    bars[0 if centre < x else 1, x] = (
                        patch.get_height(), patch.get_edgecolor()[3] > 0
                        and tuple(patch.get_edgecolor()[:3]) == (0.0, 0.0, 0.0))
                self.assertEqual(len(bars), 2 * (M + K))
                for i in (0, 1):
                    for q in range(M):
                        height, outlined = bars[i, q]
                        self.assertAlmostEqual(height, frames[f]["object_belief"][i][q],
                                               places=12)
                        self.assertEqual(outlined, q == 1 - i)
                    for k in range(K):
                        height, outlined = bars[i, M + 1 + k]
                        self.assertAlmostEqual(height, frames[f]["belief"][i][k], places=12)
                        self.assertEqual(outlined, k == truth[1 - i])
            canvas.fig.canvas.draw()
            renderer = canvas.fig.canvas.get_renderer()
            self.assertEqual([t.get_text() for t in canvas.ax_belief.get_xticklabels()],
                             ["O1", "O2", "G0", "G1", "G2"])
            left, right = canvas.ax_belief.get_window_extent(), canvas.ax_social.get_window_extent()
            self.assertLess(left.x1, right.x0)
            for ax in (canvas.ax_belief, canvas.ax_social):
                ticks = [t.get_window_extent(renderer) for t in ax.get_xticklabels()
                         if t.get_text()]
                for a, b in zip(ticks, ticks[1:]):
                    self.assertLess(a.x1, b.x0)              # no tick label runs into the next
                title = ax._left_title.get_window_extent(renderer)
                self.assertTrue(title.width > 0
                                and title.x1 <= ax.get_window_extent().x1 + 1.0)
            self.assertEqual(canvas.ax_belief.get_title(loc="left"),
                             "belief about the partner's object and destination")
            legend = canvas.ax_belief.get_legend().get_window_extent(renderer)
            self.assertTrue(left.x0 <= legend.x0 and legend.x1 <= right.x1)
            self.assertEqual(disagreements(r), [])
        finally:
            canvas.close()
        canvas = _Canvas(self.known, NAMES, "relative", "", 25, 3)
        try:
            self.assertFalse(canvas.story.objects_inferred)
            canvas.draw(2)
            self.assertEqual([t.get_text() for t in canvas.ax_belief.get_xticklabels()],
                             ["G0", "G1", "G2"])
        finally:
            canvas.close()


class TwoSelectiveCooperators(unittest.TestCase):
    """The same two cooperators under the default of 2026-10-11: each cooperates only with a
    partner it takes to be cooperative, competitive or exchanging.  Told which object the
    other wants, they do what Table I's cooperators do (10 / 6): agent_1 takes agent_2's
    object at transition 2, sets it down in vain on G0 at 4 and on agent_2's destination at
    6; agent_2 picks agent_1's object up at 7 and sets it down on agent_1's destination at
    10.  (With "use", until 2026-10-16: agent_1 handed it over at 4, agent_2 picked agent_1's
    object up and put it back, and agent_1 fetched it itself, 14 / 4.)  Inferring it, with
    the two posteriors of 2026-10-17 (21 / 19): agent_1 takes agent_2's object at 2 and
    puts it back at 7; agent_2 takes agent_1's at 4, sets it down in vain at 7, puts it back
    at 11, and each fetches its own (19 / 21).  Both end up knowing the other's object;
    agent_2 takes agent_1 for a cooperator (0.85), agent_1 takes agent_2 for a competitor
    (0.86).  (With one posterior, 2026-10-16: 7 / never -- agent_2 delivered agent_1's
    object at 7, each was sure the other wants agent_1's object, and agent_1 took agent_2
    for a conflict partner nearly as much as for a cooperator; pinned when this was
    written: 7 / 8.)

    Under the update of 2026-10-30 (Eqs. 5-6 and nothing else), told the object: 8 / 7 --
    each steps toward the other's object and picks it up (2, 4); agent_1's second step, away
    from agent_2's object toward its own, is one that only the hypotheses killed by its
    first would have explained, and agent_2 has it as a hinderer from then on (the README,
    "What pure Bayes at Eq. 7's tau is like"); at 6, adjacent, each hands the other its
    object (a cooperator that takes its partner for a hinderer has no term about the
    partner's goal and lets the foreign object go the cheapest way), and each delivers its
    own (7, 8), each sure the other is a hinderer.  Inferring it: 7 / 6 -- each fetches its
    own object (4, 5) and delivers it (6, 7); both end up knowing the other's object, and
    neither, having seen no act for itself, is sure of the other's social goal."""

    @classmethod
    def setUpClass(cls):
        pair = ("cooperation", "cooperation")
        cls.known = run_pairing(TINY_SCENE, pair, weights=(1.0, 1.0),
                                cfg=tiny(partner_object="known"), n_steps=25)
        cls.inferred = run_pairing(TINY_SCENE, pair, weights=(1.0, 1.0), cfg=tiny(),
                                   n_steps=25)

    def test_told_which_object(self):
        r = self.known
        self.assertEqual(tuple(r["config"]["cooperate_with"]),
                         ("cooperation", "competition", "exchange"))
        self.assertEqual(r["done_at"], (8, 7))
        self.assertEqual([e[:4] for e in r["events"]],
                         [(2, "picked up", 1, 0), (4, "picked up", 0, 1), (6, "handed over", 1, 0),
                          (6, "handed over", 0, 1), (7, "set down", 1, 1), (8, "set down", 0, 0)])
        social = r["final_social_belief"]
        self.assertGreater(social[0][G.HINDER], 0.99)
        self.assertGreater(social[1][G.HINDER], 0.99)

    def test_inferring_it(self):
        r = self.inferred
        self.assertEqual(r["done_at"], (7, 6))
        self.assertEqual([e[:4] for e in r["events"]],
                         [(4, "picked up", 1, 1), (5, "picked up", 0, 0), (6, "set down", 1, 1),
                          (7, "set down", 0, 0)])
        final = r["final_object_belief"]
        self.assertGreater(final[0][1], 0.99)    # agent_1: agent_2 wants O2
        self.assertGreater(final[1][0], 0.99)    # agent_2: agent_1 wants O1
        social = r["final_social_belief"]        # no act for the partner: nobody is sure
        self.assertLess(max(social[0]), 0.5)
        self.assertLess(max(social[1]), 0.5)

    def test_the_record_says_whom_it_cooperates_with(self):
        for term in self.inferred["social_terms"]:
            self.assertIn("[cooperates with cooperation, competition, exchange only]", term)


class WhatAnActInVainTells(unittest.TestCase):
    """A robot that does not know which object its partner wants may offer the wrong one.
    The hands rule: nothing happens.  ``real_step`` tells it apart, and the robot's belief
    rules out exactly what the act showed the partner does not want."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = TM.tight_now()
        cls.w, cls.goal_ids = make_world(TINY, cls.cfg)

    def test_a_hand_over_of_another_object(self):
        """agent_1 holds its own object next to agent_2 and gives it: agent_2 takes only its
        own."""
        w = self.w
        state = at(w, (1, 1), (2, 1), (HELD0, HOME))
        nxt = w.unsid(w.step(w.sid(state), (GIVE, WAIT)))
        out = w.real_step((HELD0, HOME), (0, 0), state, nxt, (GIVE, WAIT), detail=True)
        self.assertEqual(out, ((HELD0, HOME), (0, 0), (False, False), (True, False),
                               (False, False)))          # ... and took nothing (Config.steal)
        self.assertEqual(w.real_step((HELD0, HOME), (0, 0), state, nxt, (GIVE, WAIT)),
                         ((HELD0, HOME), (0, 0), (False, False)))

    def test_a_set_down_of_another_object(self):
        """agent_1 holds its own object on agent_2's destination (2, 0) and places it: a
        set-down for agent_2 that does not count -- the place is right, the object not."""
        w = self.w
        state = at(w, (2, 0), (0, 1), (HELD0, HOME))
        out = w.real_step((HELD0, HOME), (0, 0), state, state, (PLACE, WAIT), detail=True)
        self.assertEqual(out, ((HELD0, HOME), (0, 0), (True, False), (False, False),
                               (False, False)))

    def test_what_each_rules_out(self):
        """Eq. 5's transition term (``SocialMDP.transition_term``): what a set-down for the
        partner that failed, a hand-over in vain and the partner's finishing rule out of the
        hypotheses about the partner's physical goal."""
        w = self.w
        state = at(w, (2, 0), (0, 1), (HELD0, HOME))
        _, _, mdp, state, _ = TM.solved(TINY, self.cfg, (2, 2), ("cooperation", "cooperation"),
                                        (1.0, 1.0), state=lambda w_: state, seed=21)
        s = w.sid(state)
        objs, dest = mdp.pg_obj[1], w.dest[mdp.pg_dst[1]]   # agent_1's hypotheses about agent_2
        quiet = dict(done_now=(False, False), failed=(False, False),
                     handed_in_vain=(False, False), held=(0, -1))
        cases = ((dict(quiet, failed=(True, False)), (PLACE, WAIT),
                  (objs == 0) & (dest == state[0])),
                 (dict(quiet, handed_in_vain=(True, False)), (GIVE, WAIT), objs == 0),
                 (dict(quiet, done_now=(False, True)), (WAIT, WAIT),
                  (objs != 1) | (dest != w.true_dest[1])))
        for event, actions, ruled_out in cases:
            keep = mdp.transition_term(1, s, actions, state, event)
            self.assertTrue(ruled_out.any() and not ruled_out.all())
            np.testing.assert_array_equal(keep, ~ruled_out)
            post = mdp.updated_belief(1, s, actions, state, event)
            self.assertTrue(np.isneginf(post.log_phys[ruled_out]).all())
            self.assertTrue(np.isfinite(post.log_phys[~ruled_out]).all())
            self.assertAlmostEqual(float(post.phys().sum()), 1.0, places=9)
        keep = mdp.transition_term(1, s, (WAIT, WAIT), state, quiet)   # nothing told
        self.assertTrue(keep.all())


class ThePartnersEstimate(unittest.TestCase):
    """In a level-2 robot's own model its done flag is its own object delivered.  Its
    partner's level-1 models estimate its goal reward under each goal they hold possible
    for it (Table I's R_j); under "it wants another object" the flag says nothing -- that
    object's whereabouts do (``ObjectWorld.cls_est``).  As the robot's own hypothetical
    goal, in its own model, the flag is that goal done (``cls_obj``)."""

    def test_the_classes(self):
        cfg = TM.tight_now()
        w, _ = make_world(TINY, cfg)
        bf = Brute(w, cfg)
        state = at(w, (1, 1), (0, 1), (HOME, GONE), (0, 1))     # agent_2's goal done
        n, b = (state[0], state[1]), state[2]
        rho = cfg.rho
        for k in range(w.K):
            # its own object: done, either way
            self.assertEqual(w.goal_rates(state, 1, [1], [k])[0], rho)
            self.assertEqual(w.goal_rates(state, 1, [1], [k], estimate=True)[0], rho)
            # agent_1's object O1 as agent_2's goal in a model of agent_2 with that goal
            self.assertEqual(w.goal_rates(state, 1, [0], [k])[0], rho)
            self.assertTrue(w.goals_finished(state, 1, [0], [k])[0])
            # ... as agent_1 estimates it in agent_2's own model: O1 still lies at its place
            est = w.goal_rates(state, 1, [0], [k], estimate=True)[0]
            self.assertAlmostEqual(est, bf.rate(b, 1, n, k, 0, estimate=True), places=12)
            self.assertLess(est, rho)
            self.assertFalse(w.goals_finished(state, 1, [0], [k], estimate=True)[0])
        # the robot's own object and its partner's: the same classes before the flag
        self.assertTrue((w.cls_est[:, 0, 0] == w.cls_obj[:, 0, 0]).all())
        undone = w.done[:, 1] == 0
        self.assertTrue((w.cls_est[undone, 1, 0] == w.cls_obj[undone, 1, 0]).all())

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_against_brute_force(self):
        """Two cooperators on the open tiny map, agent_2 done: every level-1 and level-2
        quantity against the brute force of reference.py."""
        TM.ModelsAgainstBruteForce.compare(self, TM.tight_now(), (2, 2),
                                           ("cooperation", "cooperation"), (1.0, 1.0),
                                           state=TM.other_done, seed=137)


class TheMemoryGuard(unittest.TestCase):
    def test_a_model_too_big_fails_at_once(self):
        """Before its first table is built, with what to do about it."""
        with self.assertRaises(SolverFailure) as caught:
            run_pairing(TINY_SCENE, ("cooperation", "cooperation"),
                        cfg=tiny(max_memory_gib=1e-5), n_steps=2)
        self.assertIn("max_objects", str(caught.exception))
        self.assertIn('partner_object="known"', str(caught.exception))

    def test_the_memory_there_is(self):
        n = memory_limit_bytes()
        self.assertTrue(n is None or n > 0)


class EventsInWords(unittest.TestCase):
    def test_acts_in_vain(self):
        self.assertEqual(event_text(("handed over in vain", 0, 0, 1), NAMES),
                         "agent_1 tried to hand O1 to agent_2 (not agent_2's object)")
        self.assertEqual(event_text(("handed over in vain", 0, 2, 1), NAMES, "present",
                                    subject=False),
                         "tries to hand O3 to agent_2 (not agent_2's object)")
        place = ("set down in vain", 1, 1, 0)
        self.assertEqual(event_text(place, NAMES),
                         "agent_2 tried to place O2 (not agent_1's destination)")
        self.assertEqual(event_text(place, NAMES, why="object"),
                         "agent_2 tried to place O2 (not agent_1's object)")
        self.assertEqual(event_text(place, NAMES, why="both"),
                         "agent_2 tried to place O2 (neither agent_1's object nor its "
                         "destination)")
        self.assertEqual(_callout([("handed over in vain", 0, 0, 1)], NAMES),
                         "FAILED TO HAND OVER O1\nagent_1 → agent_2\nnot agent_2's object")
        self.assertEqual(_callout([place], NAMES, why=["object"]),
                         "FAILED TO PLACE O2\nby agent_2\nnot agent_1's object")


if __name__ == "__main__":
    unittest.main()
