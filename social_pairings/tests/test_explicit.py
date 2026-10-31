"""Nothing automatic (2026-10-18): a robot's own object is picked up by "pick" and set down
by "place" like any other; a pick-up takes ``Config.pick_steps`` = 3 transitions and costs
three moves' worth on the first (``pick_cost_rate``), the other hand actions two
(``use_cost_rate``); swap stays one transition.

The user: "Lets also make the pickup cost 3 and take 3 timesteps, Lets make dropping an
object cost 2, any sort of action besides movements should therefore cost 2"; "lets make it
such that nothing is automatic. Lets make it so you must move to the cell then select the
pickup dropoff etc action"; "3 at the start then 0 and 0"; "Swap should be its own action
i.e it should be one timestep and it should cost 2".

What this means for the world (``ObjectWorld``, ``hands.py``): the object statuses of a
lift in progress, a lifter that may only "pick" (``busy``), what is left of a robot's way
counting the transitions of its hands, a robot finishing by placing its object on its
destination -- in its own model as in its partner's (``model_tab``: the partner's own
set-down on any candidate cell) -- and the rule of before kept under
``Config(**AUTOMATIC_HANDS)``, which the tests written before run under
(``hands_before.py``, ``told.py``).
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from social_pairings import AUTOMATIC_HANDS, TOLD, Config, Scene, run_pairing
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.constants import (GIVE, GONE, HELD0, HELD1, HOME, PICK, PLACE, SWAP, TAKE,
                                       WAIT)
from social_pairings.hands import configurations, lift_stage, lifter
from social_pairings.render import _Story, event_text, render_videos
from social_pairings.serialize import load_result, save_result

from .reference import make_world
from .test_models import TINY, solved, tight_explicit, tight_now

UP, RIGHT, DOWN, LEFT = 0, 1, 2, 3           # +y, +x, -y, -x
NAMES = ("agent_1", "agent_2")
# R0 starts at (0, 1); its object O1 lies at (2, 2) and belongs on (0, 2).  R1 starts at
# (2, 1); its object O2 lies at (0, 0) and belongs on (2, 0).  Candidate G0 (1, 0) is free.


def config(**options) -> Config:
    return Config(num_threads=2, **options)


class Options(unittest.TestCase):
    def test_the_defaults_of_now(self):
        c = Config()
        self.assertEqual((c.automatic_hands, c.pick_steps, c.pick_cost_rate, c.use_cost_rate),
                         (False, 3, 3.0, 2.0))
        self.assertEqual(c.move_cost_rate, 1.0)      # a move; "stay" stays at 0.1 of it

    def test_the_rule_of_before(self):
        c = Config(**AUTOMATIC_HANDS)
        self.assertEqual((c.automatic_hands, c.pick_steps, c.pick_cost_rate, c.use_cost_rate),
                         (True, 1, 1.0, 1.0))
        self.assertTrue(Config(**TOLD).automatic_hands)
        for key, value in AUTOMATIC_HANDS.items():
            if key != "use_cost_rate":           # a field records have always had
                self.assertEqual(ADDED_FIELDS[key], value)
        self.assertTrue(written_with({})["automatic_hands"])    # a record of before
        self.assertEqual(written_with({})["pick_steps"], 1)

    def test_validation(self):
        with self.assertRaises(ValueError):
            Config(pick_steps=3, automatic_hands=True)
        for bad in (0, 10, 2.0, True):
            with self.assertRaises(ValueError):
                Config(pick_steps=bad)
        with self.assertRaises(ValueError):
            Config(pick_cost_rate=0.0)
        Config(pick_steps=1)                      # instant, by "pick" all the same
        Config(pick_steps=1, automatic_hands=True)


class Statuses(unittest.TestCase):
    """The object statuses of a lift: two per robot after the lying statuses."""

    def test_configurations(self):
        self.assertEqual(len(configurations(2, 4)), 20)                  # as before
        self.assertEqual(len(configurations(2, 8, -1, 4, 2)), 60)
        self.assertEqual(len(configurations(2, 10, 0, 6, 2)), 77)        # with swaps
        for st in configurations(2, 8, -1, 4, 2):
            lifting = [(v - 4) // 2 for v in st[:2] if 4 <= v < 8]
            self.assertEqual(len(lifting), len(set(lifting)))            # one lift a robot
            for i in (0, 1):                                             # with free hands
                self.assertFalse(i in lifting and (HELD0 + i) in st[:2])
        self.assertEqual(lifter(5, 4, 2), 0)
        self.assertEqual(lift_stage(5, 4, 2), 2)
        self.assertEqual(lifter(6, 4, 2), 1)
        self.assertEqual(lift_stage(7, 4, 2), 2)
        self.assertEqual(lifter(3, 4, 2), -1)
        self.assertEqual(lifter(8, 4, 2), -1)
        self.assertEqual(lifter(5, 4, 0), -1)

    def test_the_world(self):
        w, _ = make_world(TINY, config())
        self.assertEqual((w.lift0, w.nstage, w.n_status, w.B), (4, 2, 8, 60))
        self.assertFalse(w.auto)
        b = w.block_of([4, HOME], [0, 0])          # R0 one step into lifting O1
        self.assertEqual((w.lift[b, 0], w.stage[b, 0], w.busy[b, 0], w.busy[b, 1]), (0, 1, 1, 0))
        self.assertEqual(w.hold[b, 0], -1)
        self.assertEqual(w.object_cells([4, HOME], (w.cell_id[(2, 2)], 0, b)), ((2, 2), (0, 0)))
        self.assertIn("being lifted by R0, step 1 of 3", w.status_name(4))
        # the joint states solved: in a lifting block, the lifter where the object can lie
        self.assertEqual(int(w.state_ok.sum()), 1704)                 # 4,860 states in all
        self.assertEqual(int(w.pair_ok.sum()) * w.B, 60 * 72)
        x0, x1 = np.divmod(np.arange(w.NN), w.N)
        ok = w.state_ok[b] != 0
        self.assertTrue((x0[ok] == w.home[0]).all())                   # R0 on O1's place
        self.assertEqual(int(ok.sum()), w.N - 1)
        lying = w.block_of([HOME, HOME], [0, 0])
        np.testing.assert_array_equal(w.state_ok[lying] != 0, w.pair_ok)
        old, _ = make_world(TINY, config(**AUTOMATIC_HANDS))
        self.assertEqual((old.lift0, old.nstage, old.n_status, old.B), (4, 0, 4, 20))
        self.assertTrue(old.auto and not old.busy.any())
        np.testing.assert_array_equal(old.state_ok != 0, np.tile(old.pair_ok, (old.B, 1)))


class Hands(unittest.TestCase):
    """The rule, on the real objects (``ObjectWorld.real_step``) and the actions offered."""

    options: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.cfg = config(**cls.options)
        cls.w, cls.goal_ids = make_world(TINY, cls.cfg)

    def state(self, cells, status, done=(0, 0)):
        w = self.w
        return (int(w.cell_id[cells[0]]), int(w.cell_id[cells[1]]), w.block_of(status, done))

    def after(self, state, *joint_actions):
        s = self.w.sid(state)
        for act in joint_actions:
            s = self.w.step(s, act)
        return self.w.unsid(s)

    def objects(self, state) -> tuple:
        return tuple(int(v) for v in self.w.status[state[2]])

    def done(self, state) -> tuple:
        return tuple(int(v) for v in self.w.done[state[2]])

    def hands(self, state, robot: int, social: bool = True) -> int:
        w = self.w
        pair = state[0] * w.N + state[1]
        return int(w.hand_ok(social)[robot, state[2], w.uctx[robot, pair]])

    def test_nothing_on_arriving(self):
        """R0 arriving on its object's cell takes nothing; arriving on its destination with
        its object sets nothing down."""
        s = self.state(((1, 2), (2, 1)), [HOME, HOME])
        s = self.after(s, (RIGHT, WAIT))
        self.assertEqual(self.w.cell(s[0]), (2, 2))
        self.assertEqual(self.objects(s), (HOME, HOME))
        s = self.state(((1, 2), (2, 1)), [HELD0, HOME])
        s = self.after(s, (LEFT, WAIT))
        self.assertEqual(self.w.cell(s[0]), (0, 2))
        self.assertEqual((self.objects(s), self.done(s)), ((HELD0, HOME), (0, 0)))

    def test_a_pick_up_takes_three_transitions(self):
        s = self.state(((2, 2), (2, 1)), [HOME, HOME])
        self.assertEqual(self.hands(s, 0), 1 << (PICK - PICK))
        s = self.after(s, (PICK, WAIT))
        self.assertEqual(self.objects(s), (4, HOME))               # lifting, step 1
        self.assertEqual(self.hands(s, 0), 1 << (PICK - PICK))     # "pick" alone, and
        self.assertTrue(self.w.busy[s[2], 0])                      # no move (busy)
        s = self.after(s, (PICK, WAIT))
        self.assertEqual(self.objects(s), (5, HOME))               # step 2
        s = self.after(s, (PICK, WAIT))
        self.assertEqual(self.objects(s), (HELD0, HOME))           # in its hands
        self.assertFalse(self.w.busy[s[2], 0])

    def test_the_costs(self):
        w, c = self.w, self.cfg
        move = c.move_cost_rate * c.dt
        lying = w.block_of([HOME, HOME], [0, 0])
        lifting = w.block_of([4, HOME], [0, 0])
        held = w.block_of([HELD0, HOME], [0, 0])
        self.assertAlmostEqual(w.cost_c[lying, 0, PICK], 3 * move)
        self.assertAlmostEqual(w.cost_c[lifting, 0, PICK], 0.0)
        self.assertAlmostEqual(w.cost_c[w.block_of([5, HOME], [0, 0]), 0, PICK], 0.0)
        for a in (PLACE, TAKE, GIVE, SWAP):
            self.assertAlmostEqual(w.cost_c[held, 0, a], 2 * move)
        np.testing.assert_allclose(w.cost_c[lying, 0, :5], c.costs)      # moves and stay
        self.assertAlmostEqual(w.cost_c[lying, 0, WAIT], 0.1 * move)

    def test_place_on_its_destination_finishes(self):
        s = self.state(((0, 2), (2, 1)), [HELD0, HOME])
        self.assertTrue(self.hands(s, 0, social=False) & (1 << (PLACE - PICK)))
        s = self.after(s, (PLACE, WAIT))
        self.assertEqual((self.objects(s), self.done(s)), ((GONE, HOME), (1, 0)))

    def test_place_elsewhere_does_nothing(self):
        """Its own object on a candidate cell that is not its destination: nothing happens
        -- and the record calls it a set-down in vain, as before: with the partner's object
        inferred, a partner that wanted that object there would have taken it, and the
        belief update rules that hypothesis out."""
        s = self.state(((1, 0), (2, 1)), [HELD0, HOME])
        s2 = self.after(s, (PLACE, WAIT))
        self.assertEqual((self.objects(s2), self.done(s2)), ((HELD0, HOME), (0, 0)))
        st, dn, failed = self.w.real_step([HELD0, HOME], [0, 0], s, s2, (PLACE, WAIT))
        self.assertEqual((st, dn, failed), ((HELD0, HOME), (0, 0), (True, False)))

    def test_the_partner_cannot_take_from_a_lifter_nor_hand_it_anything(self):
        """R1 next to R0, which is lifting O2 (R1's object): no take; R1 holding O1 (R0's
        object): no give."""
        cfg = config(steal=True)
        w, _ = make_world(TINY, cfg)
        b = w.block_of([HOME, 6 + 0], [0, 0])    # R1... no: R0 lifting O2, step 1 (code 4)
        b = w.block_of([HOME, 4], [0, 0])
        s = (int(w.cell_id[(0, 0)]), int(w.cell_id[(1, 0)]), b)
        pair = s[0] * w.N + s[1]
        self.assertFalse(w.hand_ok(True)[1, b, w.uctx[1, pair]] & (1 << (TAKE - PICK)))
        nxt = w.unsid(w.step(w.sid(s), (PICK, TAKE)))
        self.assertEqual(tuple(w.status[nxt[2]]), (HOME, 5))
        b = w.block_of([HELD1, 4], [0, 0])
        s = (int(w.cell_id[(0, 0)]), int(w.cell_id[(1, 0)]), b)
        pair = s[0] * w.N + s[1]
        self.assertFalse(w.hand_ok(True)[1, b, w.uctx[1, pair]] & (1 << (GIVE - PICK)))
        nxt = w.unsid(w.step(w.sid(s), (PICK, GIVE)))
        self.assertEqual(tuple(w.status[nxt[2]]), (HELD1, 5))

    def test_swap_is_one_transition(self):
        cfg = config(swap_on_place=True)
        w, _ = make_world(TINY, cfg)
        # R0 holds O2 on O1's place (2, 2): swap -> O1 in its hands at once, O2 left there
        b = w.block_of([HOME, HELD0], [0, 0])
        s = (int(w.cell_id[(2, 2)]), int(w.cell_id[(1, 1)]), b)
        nxt = w.unsid(w.step(w.sid(s), (SWAP, WAIT)))
        self.assertEqual(tuple(w.status[nxt[2]]), (HELD0, 4 + 0))
        self.assertAlmostEqual(w.cost_c[b, 0, SWAP], 2 * cfg.move_cost_rate * cfg.dt)

    def test_the_rule_of_before_still(self):
        w, _ = make_world(TINY, config(**AUTOMATIC_HANDS))
        s = (int(w.cell_id[(1, 2)]), int(w.cell_id[(2, 1)]), w.block_of([HOME, HOME], [0, 0]))
        nxt = w.unsid(w.step(w.sid(s), (RIGHT, WAIT)))
        self.assertEqual(tuple(w.status[nxt[2]]), (HELD0, HOME))        # taken on arriving
        s = (int(w.cell_id[(1, 2)]), int(w.cell_id[(2, 1)]), w.block_of([HELD0, HOME], [0, 0]))
        nxt = w.unsid(w.step(w.sid(s), (LEFT, WAIT)))
        self.assertEqual((tuple(w.status[nxt[2]]), tuple(w.done[nxt[2]])), ((GONE, HOME), (1, 0)))


class WaysLeft(unittest.TestCase):
    """What is left of a robot's way counts the transitions of its hands: the lift ahead and
    the one set-down, so that done is the one state worth rho and placing beats hovering."""

    def test_distances(self):
        w, _ = make_world(TINY, config())

        def cell(c) -> int:
            return int(w.cell_id[c])

        lying = w.block_of([HOME, HOME], [0, 0])
        held = w.block_of([HELD0, HOME], [0, 0])
        goal = 1                                  # G1 (0, 2) is R0's destination
        self.assertEqual(int(w.dest[goal]), cell((0, 2)))
        # at its object's cell: lift 3 + 2 moves + 1 set-down
        self.assertEqual(w.dist_left((cell((2, 2)), cell((1, 1)), lying), 0, goal), 6)
        # holding it on the destination: the set-down; done: 0
        self.assertEqual(w.dist_left((cell((0, 2)), cell((1, 1)), held), 0, goal), 1)
        self.assertLess(w.rate((cell((0, 2)), cell((1, 1)), held), 0, goal), w.cfg.rho)
        for stage, code in ((1, 4), (2, 5)):
            b = w.block_of([code, HOME], [0, 0])
            self.assertEqual(w.dist_left((cell((2, 2)), cell((1, 1)), b), 0, goal),
                             (3 - stage) + 2 + 1)
        # the rule of before: moves alone, and holding it there is finished
        old, _ = make_world(TINY, config(**AUTOMATIC_HANDS))
        self.assertEqual(old.dist_left((cell((2, 2)), cell((1, 1)), old.block_of([HOME, HOME], [0, 0])), 0, goal), 2)  # noqa: E501
        self.assertEqual(old.dist_left((cell((0, 2)), cell((1, 1)), old.block_of([HELD0, HOME], [0, 0])), 0, goal), 0)  # noqa: E501
        self.assertTrue(old.finished((cell((0, 2)), cell((1, 1)), old.block_of([HELD0, HOME], [0, 0])), 0)[goal])  # noqa: E501
        self.assertFalse(w.finished((cell((0, 2)), cell((1, 1)), held), 0)[goal])


class AgainstBruteForce(unittest.TestCase):
    """The engine against the reference (``reference.Ref`` / ``Brute``, which have the rule
    of now written out apart), at levels 1 and 2, with the lifts, the explicit set-down and
    the new costs -- and a robot's own model finishing by "place" on its destination, its
    partner's models of it the same.  (``ModelsAgainstBruteForce.compare`` does the
    comparing, on a case of its own -- not by inheritance, which would run its tests
    again.)"""

    def compare(self, cfg, levels, goals, seed):
        # (imported here: at the module's top pytest would collect that case's tests again)
        from .test_models import ModelsAgainstBruteForce
        case = ModelsAgainstBruteForce("compare")
        return case.compare(cfg, levels, goals, (1.0, 1.0), seed=seed)

    def test_level_1(self):
        cfg = tight_explicit()
        self.assertFalse(cfg.automatic_hands)
        self.compare(cfg, (1, 1), ("cooperation", "conflict"), 3)

    def test_level_2(self):
        self.compare(tight_explicit(), (2, 2), ("cooperation", "hinder"), 4)

    def test_level_2_with_swaps_and_steals(self):
        self.compare(tight_explicit(swap_on_place=True, steal=True), (2, 2),
                     ("competition", "cooperation"), 5)

    def test_the_rule_of_before_is_what_the_other_tests_run_under(self):
        self.assertTrue(tight_now().automatic_hands)
        self.assertEqual(tight_now().pick_steps, 1)


class Rollouts(unittest.TestCase):
    scene = Scene(name="tiny", **TINY)

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(cls.scene, ("cooperation", "competition"), weights=(1.0, 1.0),
                                 cfg=Config(num_threads=2, resolution=1.0), n_steps=16)

    def test_both_finish_by_their_own_hands(self):
        """Each robot walks to its object, lifts it over three transitions, carries it to
        its destination and places it: done at 9 and 9."""
        r = self.result
        self.assertEqual((tuple(r["done_at"]), r["stopped_by"]), ((9, 9), "done"))
        A = r["actions"]
        acts = [[A[a] for a in row["action"]] for row in r["trace"]]
        self.assertEqual([a[0] for a in acts[3:6]], ["pick", "pick", "pick"])
        self.assertEqual(acts[8], ["place", "place"])
        self.assertEqual([tuple(e) for e in r["events"]],
                         [(4, "lifting", 0, 0, None), (4, "lifting", 1, 1, None),
                          (6, "picked up", 0, 0, None), (6, "picked up", 1, 1, None),
                          (9, "set down", 0, 0, 0), (9, "set down", 1, 1, 1)])
        self.assertEqual([tuple(row["status"]) for row in r["trace"][3:7]],
                         [(0, 0), (4, 6), (5, 7), (1, 2)])
        for i in (0, 1):
            self.assertEqual((r["ledger"][i]["lifts"], r["ledger"][i]["pickups"]), (1, 1))
            self.assertEqual(r["ledger"][i]["by_hand_action"],
                             dict(pick=3, place=1, take=0, give=0, swap=0))
            self.assertEqual(r["ledger"][i]["picked_at"], 6)
        # the steps of a lift are forced: the one legal action there
        row = r["trace"][4]
        for i in (0, 1):
            q = np.asarray([-np.inf if v is None else v for v in row["q"][i]])
            self.assertEqual(np.isfinite(q).sum(), 1)
            self.assertTrue(np.isfinite(q[PICK]))
        # the partner's lift is no evidence: beliefs as they were
        for key in ("object_belief", "belief", "social_belief"):
            np.testing.assert_allclose(r["trace"][4][key], r["trace"][5][key])

    def test_the_record_and_the_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(Path(folder) / "run.json", self.result)
            loaded = load_result(path)
            self.assertFalse(written_with(loaded["config"])["automatic_hands"])
            story = _Story(loaded, NAMES)
            self.assertEqual((story.lift0, story.nstage, story.lift_steps), (4, 2, 3))
            self.assertEqual((story.lifter(4), story.lift_stage(4)), (0, 1))
            self.assertEqual((story.lifter(7), story.lift_stage(7)), (1, 2))
            self.assertEqual(story.lifter(1), -1)
            self.assertIn("being lifted by it (2/3)", story.goal(5, 0)[3])
            paths = render_videos(loaded, Path(folder) / "run", names=NAMES, fps=3.0,
                                  title="test", dpi=60)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertGreater(p.stat().st_size, 5000)
        self.assertEqual(event_text(("lifting", 1, 0, None), NAMES), "agent_2 began to lift O1")
        self.assertEqual(event_text(("lifting", 0, 1, None), NAMES, tense="present",
                                    subject=False), "begins to lift O2")

    def test_a_hinderer_lifts_the_object_it_guards(self):
        """A hinderer standing on the cooperator's object lifts it and keeps it; the
        cooperator reads the lift as evidence of the object (the pick), nothing more."""
        r = run_pairing(self.scene, ("cooperation", "hinder"), weights=(1.0, 1.0),
                        cfg=Config(num_threads=2, resolution=1.0, value_maps=False), n_steps=12)
        kinds = [(e[1], e[2], e[3]) for e in r["events"]]
        self.assertIn(("lifting", 1, 0), kinds)
        self.assertIn(("picked up", 1, 0), kinds)
        self.assertEqual(tuple(r["done_at"]), (None, None))
        t = next(e[0] for e in r["events"] if e[1] == "lifting")
        before = r["trace"][t - 1]["object_belief"][0]
        after = r["trace"][t]["object_belief"][0]
        self.assertGreater(after[0], before[0])


class Solved(unittest.TestCase):
    """A robot's own model: holding its object on its destination it places it (the one
    state worth rho is done)."""

    def test_placing_beats_hovering(self):
        w, _, mdp, state, _ = solved(TINY, tight_explicit(), (1, 1),
                                     ("cooperation", "conflict"), (1.0, 1.0), seed=3)
        s = w.sid((int(w.cell_id[(0, 2)]), int(w.cell_id[(1, 1)]),
                   w.block_of([HELD0, HOME], [0, 0])))
        q = mdp.q_values(mdp.root[0], s, 0)
        self.assertEqual(int(np.argmax(np.where(np.isfinite(q), q, -np.inf))), PLACE)
        self.assertGreater(q[PLACE], q[WAIT])            # (by a step's worth: it can place
        cfg = w.cfg                                      # next step too)
        hovering = (w.rate(w.unsid(s), 0, mdp.goal_ids[0]) - cfg.costs[WAIT]) / (1 - cfg.gamma)
        self.assertGreater(q[PLACE] - hovering, 10.0)    # holding it there for ever


if __name__ == "__main__":
    unittest.main()
