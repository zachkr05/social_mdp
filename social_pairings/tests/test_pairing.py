"""The entry point: what ``run_pairing`` takes, returns and refuses.

The slower runs (the exchange scene at every weight) only with SOCIAL_PAIRINGS_SLOW=1.
"""
import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from social_pairings import Scene, SolverFailure, exchange_scene, six_object_scene, two_room_scene
from social_pairings import goals as G
from social_pairings.constants import ACTION_NAMES, GIVE, GONE, HELD0, HELD1, HOME, PICK
from social_pairings.render import frames_of, render_gif
from social_pairings.rollout import _events_of, _Run, build_world
from social_pairings.serialize import (USE_ACTIONS, action_names, load_result,
                                       save_result, to_json)

from .told import Config, default_levels, levels_used, run_baseline, run_pairing

CFG = Config(num_threads=2)
SLOW = os.environ.get("SOCIAL_PAIRINGS_SLOW", "") not in ("", "0")


def actions(result) -> list:
    return [tuple(row["action"]) for row in result["trace"]]


class RecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = six_object_scene()
        cls.result = run_pairing(cls.scene, ("hinder", "cooperation"), weights=(2.0, 2.0),
                                 cfg=CFG, n_steps=12)

    def test_what_the_record_holds(self):
        r = self.result
        for key in ("steps", "stopped_by", "completed", "done_at", "ledger", "trace", "events",
                    "deadlock", "status", "holds", "object_cells", "final_pos", "final_belief",
                    "goals", "levels", "levels_used", "weights", "social_terms", "config",
                    "scene", "solver", "numerically_accepted", "wall_time_s"):
            self.assertIn(key, r)
        self.assertEqual(r["goals"], ("hinder", "cooperation"))
        # no levels asked for: the defaults (of TOLD: level_free_level=1), 1 for the hinderer
        # and 2 for its partner
        self.assertEqual((r["levels"], r["levels_used"]), ((1, 2), (1, 2)))
        self.assertEqual(r["weights"], (2.0, 2.0))
        self.assertEqual(r["steps"], len(r["trace"]))
        self.assertTrue(r["numerically_accepted"])
        self.assertLessEqual(r["solver"]["residual_max"], r["solver"]["residual_target"])

    def test_spare_objects_are_listed_but_not_modelled(self):
        r = self.result
        self.assertEqual(len(r["scene"]["objects"]), 6)
        self.assertEqual(r["scene"]["modelled_objects"], 2)
        self.assertEqual(r["scene"]["objects"][:2], self.scene.pickups)
        for row in r["trace"]:
            self.assertEqual(len(row["status"]), 6)
            self.assertEqual(tuple(row["status"][2:]), (HOME,) * 4)
            self.assertEqual(tuple(row["object_cells"][2:]), tuple(r["scene"]["objects"][2:]))

    def test_the_trace_is_one_continuous_story(self):
        r = self.result
        walls = np.asarray(r["scene"]["grid"]) == 0
        K = len(r["scene"]["candidates"])
        rows = r["trace"]
        self.assertTrue(np.allclose(rows[0]["belief"][1], 1.0 / K))
        for t, row in enumerate(rows):
            self.assertEqual(row["step"], t + 1)
            for key in ("pos", "next_pos"):
                self.assertNotEqual(tuple(row[key][0]), tuple(row[key][1]))
                self.assertFalse(any(walls[tuple(p)] for p in row[key]))
            for i in (0, 1):
                self.assertTrue(np.isfinite(row["q"][i][row["action"][i]]))
                self.assertAlmostEqual(float(np.sum(row["belief"][i])), 1.0)
            for prefix in ("", "next_"):
                cells, status = row[prefix + "object_cells"], row[prefix + "status"]
                pos = row["pos" if not prefix else "next_pos"]
                for k, (cell, st) in enumerate(zip(cells, status)):
                    if st == HOME:
                        self.assertEqual(tuple(cell), tuple(r["scene"]["objects"][k]))
                    elif st in (HELD0, HELD1):
                        self.assertEqual(tuple(cell), tuple(pos[st - 1]))
                    else:
                        self.assertEqual(st, GONE)
                        self.assertIn(tuple(cell), [tuple(d) for d in r["scene"]["destinations"]])
            if t + 1 < len(rows):
                for key in ("pos", "status", "holds", "object_cells"):
                    self.assertEqual(row["next_" + key], rows[t + 1][key])
        self.assertTrue(np.allclose(rows[-1]["next_belief"], r["final_belief"]))

    def test_events_match_the_ledger(self):
        r = self.result
        picked = [e for e in r["events"] if e[1] == "picked up"]
        self.assertEqual(sum(led["pickups"] for led in r["ledger"]), len(picked))
        self.assertEqual(r["done_at"], tuple(led["done_at"] for led in r["ledger"]))

    def test_json_round_trip_and_rendering_from_the_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(Path(folder) / "run.json", self.result)
            loaded = load_result(path)
            self.assertEqual(loaded["done_at"], list(self.result["done_at"]))
            self.assertEqual([tuple(row["action"]) for row in loaded["trace"]],
                             actions(self.result))
            json.dumps(to_json(self.result), allow_nan=False)        # strict JSON: no inf / nan
            short = dict(loaded, trace=loaded["trace"][:3])
            gif = render_gif(short, Path(folder) / "run.gif", title="test")
            self.assertGreater(gif.stat().st_size, 10000)
            self.assertEqual(len(frames_of(short)), 4)               # 3 transitions + the end

    def test_the_record_names_its_actions(self):
        """Since 2026-10-16 a record says what its action numbers are; one without (written
        before) was rolled out with "use", and is not drawn."""
        self.assertEqual(tuple(self.result["actions"]), ACTION_NAMES)
        with tempfile.TemporaryDirectory() as folder:
            loaded = load_result(save_result(Path(folder) / "run.json", self.result))
            self.assertEqual(action_names(loaded), ACTION_NAMES)
            before = {k: v for k, v in loaded.items() if k != "actions"}
            self.assertEqual(action_names(before), USE_ACTIONS)
            with self.assertRaises(ValueError):
                render_gif(dict(before, trace=before["trace"][:2]), Path(folder) / "x.gif")


class BehaviourOfTheEntryPointTests(unittest.TestCase):
    def test_the_same_call_gives_the_same_rollout(self):
        scene = two_room_scene()
        a = run_pairing(scene, ("conflict", "cooperation"), cfg=CFG, n_steps=8)
        b = run_pairing(scene, ("conflict", "cooperation"), cfg=CFG, n_steps=8)
        self.assertEqual(actions(a), actions(b))
        self.assertTrue(all(np.array_equal(x["q"][i], y["q"][i], equal_nan=True)
                            for x, y in zip(a["trace"], b["trace"]) for i in (0, 1)))

    def test_scenery_does_not_change_what_the_robots_do(self):
        """Six objects with four of them scenery roll out exactly as the two own objects."""
        two = run_pairing(two_room_scene(), ("hinder", "competition"), cfg=CFG, n_steps=8)
        six = run_pairing(six_object_scene(), ("hinder", "competition"), cfg=CFG, n_steps=8)
        self.assertEqual(actions(two), actions(six))
        self.assertTrue(all(np.array_equal(x["q"][i], y["q"][i], equal_nan=True)
                            for x, y in zip(two["trace"], six["trace"]) for i in (0, 1)))

    def test_baseline_is_two_level_0_robots_without_social_goals(self):
        base = run_baseline(two_room_scene(), cfg=CFG)
        self.assertEqual((base["goals"], base["levels_used"]), (("none", "none"), (0, 0)))
        self.assertEqual(base["social_terms"], ("no social term", "no social term"))
        self.assertTrue(base["completed"])

    def test_weight_zero_switches_a_social_goal_off(self):
        scene = two_room_scene()
        off = run_pairing(scene, ("cooperation", "conflict"), weights=(0.0, 0.0), cfg=CFG,
                          n_steps=8)
        none = run_pairing(scene, (None, None), cfg=CFG, n_steps=8)
        self.assertEqual(off["social_goals"], ("none", "none"))
        self.assertEqual(actions(off), actions(none))

    def test_levels_asked_for_and_levels_by_default(self):
        """Without ``levels`` a hinderer and a competitor that races reason at level 1 and
        everybody else at level 2.  Levels asked for are used as asked -- a competitor at
        level 2 -- but for a hinderer, which stays at level 1."""
        scene = two_room_scene()
        for goals, default in ((("competition", "cooperation"), (1, 2)),
                               (("cooperation", "competition"), (2, 1)),
                               (("hinder", "competition"), (1, 1)),
                               (("cooperation", "conflict"), (2, 2)),
                               ((None, "exchange"), (2, 2))):
            with self.subTest(goals=goals):
                self.assertEqual(default_levels(goals), default)
                self.assertEqual(default_levels(goals, CFG, (1.0, 1.0)), default)
                r = run_pairing(scene, goals, cfg=CFG, n_steps=2)
                self.assertEqual((r["levels"], r["levels_used"]), (default, default))
        self.assertEqual(default_levels(("competition", "hinder"), weights=(0.0, 0.0)), (2, 2))
        asked = run_pairing(scene, ("competition", "cooperation"), levels=(2, 2), cfg=CFG,
                            n_steps=2)
        self.assertEqual((asked["levels"], asked["levels_used"]), ((2, 2), (2, 2)))
        held = run_pairing(scene, ("hinder", "competition"), levels=(2, 2), cfg=CFG, n_steps=2)
        self.assertEqual((held["levels"], held["levels_used"]), ((2, 2), (1, 2)))
        self.assertEqual(levels_used(("hinder", "competition"), (2, 2)), (1, 2))
        low = run_pairing(scene, ("competition", "cooperation"), levels=(1, 1), cfg=CFG,
                          n_steps=2)
        self.assertEqual((low["levels"], low["levels_used"]), ((1, 1), (1, 1)))

    def test_weight_zero_switches_hinder_and_competition_off_too(self):
        """The two goals whose level is 1 by default: with weight 0 they are no goals, and
        the robot reasons at level 2, as a robot without social goal does."""
        scene = two_room_scene()
        none = run_pairing(scene, (None, "cooperation"), weights=(0.0, 1.0), cfg=CFG,
                           n_steps=10)
        self.assertEqual(none["levels_used"], (2, 2))
        for goal in ("hinder", "competition"):
            with self.subTest(goal=goal):
                off = run_pairing(scene, (goal, "cooperation"), weights=(0.0, 1.0), cfg=CFG,
                                  n_steps=10)
                self.assertEqual(off["social_goals"], ("none", "cooperation"))
                self.assertEqual(off["levels_used"], (2, 2))
                self.assertEqual(off["social_terms"][0], "no social term")
                self.assertEqual(actions(off), actions(none))
                for row, other in zip(off["trace"], none["trace"]):
                    for i in (0, 1):
                        np.testing.assert_array_equal(row["q"][i], other["q"][i])

    def test_a_swap_that_finishes_both_goals(self):
        """Each robot stands on its own destination with the other's object in its hands;
        both give.  The objects change hands and are set down by their owners:
        with own-object goals O1 cannot have finished R1's goal, nor O2 R0's."""
        cfg = Config(num_threads=2, resolution=1.0)
        w, _, solo, _ = build_world(np.ones((3, 3)), ((0, 0), (1, 0)), ((2, 2), (0, 2)),
                                    ((0, 0), (1, 0)), (), "goal", cfg)
        old_status, old_done = (HELD1, HELD0), (0, 0)
        block = w.block_of(old_status, old_done)
        state = (int(w.cell_id[(0, 0)]), int(w.cell_id[(1, 0)]), block)
        nxt = w.unsid(w.step(w.sid(state), (GIVE, GIVE)))
        new_status, new_done, failed = w.real_step(old_status, old_done, state, nxt,
                                                   (GIVE, GIVE))
        self.assertEqual((tuple(new_status), tuple(new_done)), ((GONE, GONE), (1, 1)))
        events = _events_of(w, _Run(w, solo), old_status, new_status, nxt, (True, True),
                            failed, block)
        self.assertEqual(events, [("handed over", 1, 0, 0), ("set down", 0, 0, 0),
                                  ("handed over", 0, 1, 1), ("set down", 1, 1, 1)])

    def test_sampled_actions_are_reproducible_by_seed(self):
        scene = two_room_scene()
        runs = [run_pairing(scene, ("cooperation", "cooperation"), n_steps=6,
                            cfg=Config(num_threads=2, action_selection="sample", seed=seed))
                for seed in (3, 3)]
        self.assertEqual(actions(runs[0]), actions(runs[1]))

    def test_what_is_refused(self):
        scene = two_room_scene()
        for kwargs in (dict(goals=("barter", "none")), dict(goals=("hinder",)),
                       dict(goals=("hinder", "none"), weights=(-1.0, 1.0)),
                       dict(goals=("hinder", "none"), weights=(float("nan"), 1.0)),
                       dict(goals=("hinder", "none"), levels=(3, 1)),
                       dict(goals=("hinder", "none"), n_steps=0)):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                run_pairing(scene, kwargs.pop("goals"), cfg=CFG, **kwargs)
        with self.assertRaises(ValueError):
            Config(action_selection="greedy")
        walled_in = Scene(grid=scene.grid, starts=scene.starts, pickups=scene.pickups,
                          destinations=scene.destinations, candidate_cells=((0, 0),))
        with self.assertRaises(SolverFailure):
            run_baseline(walled_in, cfg=CFG)


class ExchangeSceneTests(unittest.TestCase):
    """Each robot in its own room, its object in the other one, a hand-off cell in the
    doorway, and nobody finishes its partner's task for it."""

    RULE = dict(set_down_for_partner=False, hand_over=False)

    @classmethod
    def setUpClass(cls):
        cls.scene = exchange_scene()
        cls.put_down = run_pairing(cls.scene, ("exchange", "exchange"), weights=(30.0, 30.0),
                                   cfg=Config(num_threads=2, **cls.RULE), n_steps=40)

    def events(self, result) -> list:
        return [tuple(e[:4]) for e in result["events"]]

    def test_the_scene(self):
        scene = self.scene
        door = (4, 2)
        self.assertEqual(scene.handoff_cells, (door,))
        left, right = (set(room) for room in scene.rooms)
        self.assertEqual(left & right, {door})
        for i, room in enumerate((left, right)):
            self.assertIn(scene.starts[i], room)
            self.assertIn(scene.destinations[i], room)       # its destination: in its room
            self.assertNotIn(scene.pickups[i], room)         # its object: in the other one
            self.assertIn(scene.pickups[1 - i], room)
        json.dumps(to_json(scene.to_json()))
        open_door = exchange_scene(rooms=False, handoff=False)
        self.assertEqual((open_door.rooms, open_door.handoff_cells), (None, ()))

    def test_nobody_finishes_alone(self):
        base = run_baseline(self.scene, cfg=Config(num_threads=2, **self.RULE), n_steps=40)
        self.assertEqual((tuple(base["done_at"]), base["solo_steps"]),
                         ((None, None), (None, None)))
        self.assertEqual(base["events"], [])

    def test_two_exchangers_put_down_and_pick_up(self):
        """Each takes its partner's object (2), carries it to the doorway and puts it down
        there (7, 10); each picks its own object up from the doorway (12, 13) and sets it
        down on its own destination (18, 17)."""
        r = self.put_down
        self.assertEqual(self.events(r), [
            (2, "picked up", 1, 0), (2, "picked up", 0, 1), (7, "put down", 0, 1),
            (10, "put down", 1, 0), (12, "picked up", 0, 0), (13, "picked up", 1, 1),
            (17, "set down", 1, 1), (18, "set down", 0, 0)])
        self.assertEqual((tuple(r["done_at"]), r["stopped_by"]), ((18, 17), "done"))
        self.assertEqual([led["put_downs"] for led in r["ledger"]], [1, 1])
        self.assertEqual([led["handed_over"] for led in r["ledger"]], [0, 0])
        self.assertEqual([led["delivered_by"] for led in r["ledger"]], [0, 1])
        self.assertTrue(r["numerically_accepted"])
        self.assertEqual(tuple(tuple(c) for c in r["scene"]["handoff_cells"]), ((4, 2),))

    def test_the_exchange_at_a_small_weight(self):
        """The same exchange at weights 1 / 1; agent_2 takes its own object in the visit to
        the doorway in which it puts agent_1's down (11), and is done at 15."""
        r = run_pairing(self.scene, ("exchange", "exchange"), weights=(1.0, 1.0),
                        cfg=Config(num_threads=2, **self.RULE), n_steps=40)
        self.assertEqual(self.events(r), [
            (2, "picked up", 1, 0), (2, "picked up", 0, 1), (7, "put down", 0, 1),
            (10, "put down", 1, 0), (11, "picked up", 1, 1), (12, "picked up", 0, 0),
            (15, "set down", 1, 1), (18, "set down", 0, 0)])
        self.assertEqual(tuple(r["done_at"]), (18, 15))
        self.assertEqual(r["trace"][10]["action"][1], PICK)      # by its hands, not on arriving

    def test_nobody_leaves_its_room_and_objects_lie_where_the_record_says(self):
        r = self.put_down
        rooms = [set(tuple(c) for c in room) for room in r["scene"]["rooms"]]
        lying = 0
        for row in r["trace"]:
            for i in (0, 1):
                self.assertIn(tuple(row["pos"][i]), rooms[i])
                self.assertIn(tuple(row["next_pos"][i]), rooms[i])
            for k in (0, 1):
                if row["next_status"][k] >= 4:               # on the hand-off cell
                    lying += 1
                    self.assertEqual(tuple(row["next_object_cells"][k]), (4, 2))
        self.assertGreater(lying, 0)

    def test_with_hand_overs_allowed_the_exchange_is_still_put_down_and_pick_up(self):
        """A robot standing in the doorway puts its partner's object down, whatever
        ``Config.hand_over`` says: nothing is handed over from a hand-off cell.  From next
        to the doorway a robot may hand over to a partner standing in it with free hands;
        agent_2 tries once (transition 8), as agent_1 steps out of the doorway."""
        r = run_pairing(self.scene, ("exchange", "exchange"), weights=(30.0, 30.0),
                        cfg=Config(num_threads=2, set_down_for_partner=False), n_steps=40)
        self.assertEqual(self.events(r), [
            (2, "picked up", 1, 0), (2, "picked up", 0, 1), (7, "put down", 0, 1),
            (10, "put down", 1, 0), (11, "picked up", 1, 1), (12, "picked up", 0, 0),
            (15, "set down", 1, 1), (18, "set down", 0, 0)])
        self.assertEqual(tuple(r["done_at"]), (18, 15))
        self.assertEqual([led["handed_over"] for led in r["ledger"]], [0, 0])
        tried = r["trace"][7]
        self.assertEqual((tried["action"][1], tuple(tried["events"])), (GIVE, ()))
        self.assertEqual((tuple(tried["pos"][0]), tuple(tried["next_pos"][0])), ((4, 2), (3, 2)))

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_the_exchange_at_every_weight(self):
        """Weights 0.2 to 100, put down and pick up and with hand-overs allowed: both goals
        are done in every run, each object is put down in the doorway once (7, 10) and
        nothing is handed over."""
        for hand_over, late in ((False, {30.0: (18, 17), 100.0: (23, 19)}),
                                (True, {100.0: (23, 19)})):
            cfg = Config(num_threads=2, set_down_for_partner=False, hand_over=hand_over)
            for w in (0.2, 0.5, 0.8, 1.0, 2.0, 5.0, 10.0, 30.0, 100.0):
                with self.subTest(hand_over=hand_over, weight=w):
                    r = run_pairing(self.scene, ("exchange", "exchange"), weights=(w, w),
                                    cfg=cfg, n_steps=60)
                    self.assertEqual(tuple(r["done_at"]), late.get(w, (18, 15)))
                    self.assertEqual([e for e in self.events(r) if e[1] == "put down"],
                                     [(7, "put down", 0, 1), (10, "put down", 1, 0)])
                    self.assertEqual([led["handed_over"] for led in r["ledger"]], [0, 0])

    def test_two_rivals_trade_through_the_doorway_as_well(self):
        """The scene does not single out the exchange goal.  At weights 30 / 30 two rivals
        put down and pick up as two exchangers do: each holds what the other needs, and a
        rival that puts its partner's object down in the doorway can take its own from
        there.  At 1 / 1 they do not move."""
        cfg = Config(num_threads=2, competition="rival", **self.RULE)
        pair = ("competition", "competition")
        r = run_pairing(self.scene, pair, weights=(30.0, 30.0), cfg=cfg, n_steps=40)
        self.assertEqual((tuple(r["levels_used"]), tuple(r["done_at"])), ((2, 2), (18, 15)))
        self.assertEqual([e for e in self.events(r) if e[1] == "put down"],
                         [(7, "put down", 0, 1), (10, "put down", 1, 0)])
        r = run_pairing(self.scene, pair, weights=(1.0, 1.0), cfg=cfg, n_steps=40)
        self.assertEqual((tuple(r["done_at"]), r["events"]), ((None, None), []))

    def test_without_two_exchangers_nothing_is_exchanged(self):
        """Two robots without social goals at level 2, and an exchanger next to a robot
        without social goal: nobody touches an object (Table I gives exchange a term only
        against an exchanging partner).  A cooperator next to a robot without social goal
        gives and gets nothing back.  Two cooperators wait for each other."""
        cfg = Config(num_threads=2, **self.RULE)
        for goals, levels in ((("none", "none"), (2, 2)), (("exchange", "none"), None),
                              (("cooperation", "cooperation"), None)):
            with self.subTest(goals=goals):
                r = run_pairing(self.scene, goals, weights=(30.0, 30.0), levels=levels, cfg=cfg,
                                n_steps=40)
                self.assertEqual((tuple(r["done_at"]), r["events"]), ((None, None), []))
                self.assertEqual([led["moves"] for led in r["ledger"]], [0, 0])
        r = run_pairing(self.scene, ("cooperation", "none"), weights=(30.0, 30.0), cfg=cfg,
                        n_steps=40)
        self.assertEqual(self.events(r), [(2, "picked up", 0, 1), (7, "put down", 0, 1),
                                          (8, "picked up", 1, 1), (12, "set down", 1, 1)])
        self.assertEqual(tuple(r["done_at"]), (None, 12))

    def test_hand_overs_only_roll_out_the_same_every_time(self):
        """Until 2026-10-07 they did not: the blocks a model is solved on came from a cache
        keyed by the address of a temporary array."""
        cfg = Config(num_threads=2, set_down_for_partner=False)
        scene = exchange_scene(rooms=False, handoff=False)
        runs = [run_pairing(scene, ("exchange", "exchange"), weights=(w, w), cfg=cfg, n_steps=12)
                for w in (1.0, 30.0, 1.0, 30.0)]
        for a, b in ((runs[0], runs[2]), (runs[1], runs[3])):
            self.assertEqual(actions(a), actions(b))
            self.assertTrue(all(np.array_equal(x["q"][i], y["q"][i], equal_nan=True)
                                for x, y in zip(a["trace"], b["trace"]) for i in (0, 1)))

    def test_with_hand_overs_only_a_level_2_robot_still_finishes_its_own_goal(self):
        """Until 2026-10-07 it stopped one step short of its destination: where its own
        goal was done it read partner policies that had never been computed."""
        r = run_pairing(six_object_scene(), ("competition", "competition"),
                        weights=(30.0, 30.0), levels=(2, 2), n_steps=40,
                        cfg=Config(num_threads=2, competition="rival",
                                   set_down_for_partner=False))
        self.assertEqual(tuple(r["done_at"]), (9, 11))

    def test_what_a_scene_refuses(self):
        scene = self.scene
        described = dict(scene.__dict__)
        cfg = Config(num_threads=2, **self.RULE)
        for change in (dict(handoff_cells=(scene.pickups[0],)),
                       dict(handoff_cells=(scene.destinations[1],))):
            with self.subTest(**change), self.assertRaises(ValueError):
                run_baseline(Scene(**{**described, **change}), cfg=cfg)
        outside = dict(starts=((5, 2), scene.starts[1]))             # agent_1 in agent_2's room
        with self.assertRaises(SolverFailure):
            run_baseline(Scene(**{**described, **outside}), cfg=cfg)
        with self.assertRaises(ValueError):
            Scene(**{**described, "rooms": (scene.rooms[0],)})


class InferredExchangeTests(unittest.TestCase):
    """The exchange scene with NOBODY told its partner's social goal
    (``partner_social_goal="inferred"``: a uniform prior over the five goals of the paper,
    updated from what the partner does), put down and pick up.  The hypothesis "competition"
    is the rival of the scripts (``competition="rival"``) unless a test says otherwise.  A
    robot is still given its partner's weight and level."""

    RULE = dict(set_down_for_partner=False, hand_over=False, partner_social_goal="inferred",
                competition="rival")
    EXCHANGE = [(2, "picked up", 1, 0), (2, "picked up", 0, 1), (7, "put down", 0, 1),
                (10, "put down", 1, 0), (11, "picked up", 1, 1), (12, "picked up", 0, 0),
                (15, "set down", 1, 1), (18, "set down", 0, 0)]

    @classmethod
    def setUpClass(cls):
        cls.scene = exchange_scene()
        cls.cfg = Config(num_threads=2, exchange_level1="copy_unweighted", **cls.RULE)
        cls.result = run_pairing(cls.scene, ("exchange", "exchange"), weights=(30.0, 30.0),
                                 cfg=cls.cfg, n_steps=40)

    def events(self, result) -> list:
        return [tuple(e[:4]) for e in result["events"]]

    def belief_in(self, result, goal: int) -> list:
        """Each robot's belief that its partner has ``goal``, before every transition and
        at the end -> two lists."""
        rows = [row["social_belief"] for row in result["trace"]]
        rows.append(result["final_social_belief"])
        return [[float(np.asarray(b[i], dtype=float)[goal]) for b in rows] for i in (0, 1)]

    def test_two_exchangers_that_have_to_infer_each_other_exchange(self):
        """With the text's copy of the partner's goal as what an exchanger is at level 1,
        not multiplied by the weight: each picks its partner's object up (2), puts it down
        in the doorway (7, 10), takes its own from there (12, 11) and sets it down on its
        own destination (18, 15)."""
        r = self.result
        self.assertEqual(self.events(r), self.EXCHANGE)
        self.assertEqual((tuple(r["done_at"]), r["stopped_by"]), ((18, 15), "done"))
        self.assertEqual((r["config"]["partner_social_goal"], tuple(r["levels_used"])),
                         ("inferred", (2, 2)))
        self.assertTrue(r["numerically_accepted"])
        self.assertIn("depends on the partner's social goal", r["social_terms"][0])

    def test_what_each_comes_to_believe(self):
        """Both start from 1/5 on each goal.  The partner's help can be read as cooperation
        or as exchange and as nothing else; neither robot's belief in "exchange" ever dies,
        which is what keeps its own term alive.  At the end agent_1 is sure that agent_2
        exchanges; agent_2 still takes agent_1 for a cooperator rather than an exchanger."""
        r = self.result
        exchange, cooperation = self.belief_in(r, G.EXCHANGE), self.belief_in(r, G.COOPERATION)
        for i in (0, 1):
            first = np.asarray(r["trace"][0]["social_belief"][i], dtype=float)
            hyp = [G.parse_goal(name) for name in r["config"]["social_hypotheses"]]
            self.assertTrue(np.allclose(first[hyp], 0.2) and np.isclose(first.sum(), 1.0))
            self.assertGreater(min(exchange[i]), 0.02)
            helpful = np.asarray(exchange[i]) + np.asarray(cooperation[i])
            self.assertGreater(float(helpful[3:].min()), 0.99)   # from the third transition
        self.assertGreater(exchange[0][-1], 0.95)
        self.assertTrue(0.1 < exchange[1][-1] < 0.3 and cooperation[1][-1] > 0.7)

    def test_under_table_one_as_printed_a_helper_is_read_as_a_cooperator(self):
        """``exchange_level1="none"``: at level 1 an exchanger does nothing for its partner,
        so the robot that is helped reads its helper as a cooperator -- against which the
        exchange row is N/A.  At 30 / 30 each takes the other's object (3), both hold
        "cooperation" from then on, and nobody finishes."""
        r = run_pairing(self.scene, ("exchange", "exchange"), weights=(30.0, 30.0),
                        cfg=Config(num_threads=2, **self.RULE), n_steps=40)
        self.assertEqual(sorted(self.events(r)),
                         [(3, "picked up", 0, 1), (3, "picked up", 1, 0)])
        self.assertEqual(tuple(r["done_at"]), (None, None))
        cooperation, exchange = self.belief_in(r, G.COOPERATION), self.belief_in(r, G.EXCHANGE)
        for i in (0, 1):
            self.assertGreater(min(cooperation[i][3:]), 0.8)
            self.assertLess(max(exchange[i][3:]), 0.06)

    def test_a_copy_that_grows_with_the_weight_makes_both_wait(self):
        """``exchange_level1="copy"`` at 30 / 30: the level-1 exchanger of each robot's
        model is so eager a helper that the robot waits for it: each takes one step and
        nobody touches an object.  At 2 / 2 the two exchange."""
        cfg = Config(num_threads=2, exchange_level1="copy", **self.RULE)
        r = run_pairing(self.scene, ("exchange", "exchange"), weights=(30.0, 30.0), cfg=cfg,
                        n_steps=40)
        self.assertEqual((tuple(r["done_at"]), r["events"]), ((None, None), []))
        self.assertEqual([led["moves"] for led in r["ledger"]], [1, 1])
        r = run_pairing(self.scene, ("exchange", "exchange"), weights=(2.0, 2.0), cfg=cfg,
                        n_steps=40)
        self.assertEqual(tuple(r["done_at"]), (18, 15))

    def test_at_the_weight_of_the_paper_the_two_copies_are_one(self):
        runs = [run_pairing(self.scene, ("exchange", "exchange"), weights=(1.0, 1.0),
                            cfg=Config(num_threads=2, exchange_level1=mode, **self.RULE),
                            n_steps=40) for mode in ("copy", "copy_unweighted")]
        self.assertEqual(actions(runs[0]), actions(runs[1]))
        self.assertEqual(tuple(runs[0]["done_at"]), (19, 16))
        self.assertTrue(all(np.array_equal(x["q"][i], y["q"][i], equal_nan=True)
                            for x, y in zip(runs[0]["trace"], runs[1]["trace"])
                            for i in (0, 1)))

    def test_the_reading_of_competition_among_the_hypotheses(self):
        """What a level-1 competitor is changes one of the five hypotheses.  At 30 / 30 the
        exchange is the same, event for event, with Table I's competitor and with
        ``"race_then_hinder"`` among them."""
        for reading in ("table", "race_then_hinder"):
            with self.subTest(competition=reading):
                r = run_pairing(self.scene, ("exchange", "exchange"), weights=(30.0, 30.0),
                                cfg=dataclasses.replace(self.cfg, competition=reading),
                                n_steps=40)
                self.assertEqual(self.events(r), self.EXCHANGE)

    def test_other_pairings_with_inferred_goals(self):
        """Two robots without social goals do not touch an object (each takes two or
        three steps and waits).  An exchanger next to a hinderer takes its partner's
        object and does not put it down for it.  Two cooperators, each unsure of the
        other, give one after the other."""
        cfg = self.cfg
        r = run_pairing(self.scene, ("none", "none"), weights=(30.0, 30.0), levels=(2, 2),
                        cfg=cfg, n_steps=40)
        self.assertEqual((tuple(r["done_at"]), r["events"]), ((None, None), []))
        self.assertEqual([led["moves"] for led in r["ledger"]], [2, 3])
        r = run_pairing(self.scene, ("exchange", "hinder"), weights=(30.0, 30.0), cfg=cfg,
                        n_steps=40)
        self.assertEqual((tuple(r["done_at"]), self.events(r)),
                         ((None, None), [(4, "picked up", 0, 1)]))
        r = run_pairing(self.scene, ("cooperation", "cooperation"), weights=(30.0, 30.0),
                        cfg=cfg, n_steps=40)
        self.assertEqual(tuple(r["done_at"]), (14, 27))
        self.assertEqual([e[0] for e in self.events(r) if e[1] == "put down"], [7, 22])

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_table_one_as_printed_at_every_weight(self):
        """``exchange_level1="none"``, put down and pick up: whoever is helped first stops
        helping.  From 0.5 to 10 one robot puts its partner's object down in the doorway
        and gets nothing back; at 30 neither gives; only at 100 are both goals done.  At
        2 / 2 the robot that was helped is sure that its helper is a cooperator."""
        done = {0.2: (None, None), 0.5: (19, None), 0.8: (18, None), 1.0: (15, None),
                2.0: (None, 14), 5.0: (None, 18), 10.0: (None, 18), 30.0: (None, None),
                100.0: (22, 19)}
        cfg = Config(num_threads=2, **self.RULE)
        for w, expected in done.items():
            with self.subTest(weight=w):
                r = run_pairing(self.scene, ("exchange", "exchange"), weights=(w, w), cfg=cfg,
                                n_steps=60)
                self.assertEqual(tuple(r["done_at"]), expected)
                if w == 2.0:
                    self.assertEqual([led["put_downs"] for led in r["ledger"]], [1, 0])
                    self.assertGreater(self.belief_in(r, G.COOPERATION)[1][-1], 0.99)

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_the_inferred_exchange_at_every_weight(self):
        """Put down and pick up: both goals done at every weight from 0.5 to 100, each
        object put down in the doorway once and nothing handed over; at 0.2 nobody touches
        an object.  With hand-overs allowed: 14 / 13 at every weight from 0.2 to 100.  With
        Table I's competitor among the hypotheses instead of the rival: from 0.8."""
        done = {0.2: (None, None), 0.5: (19, 16), 0.8: (19, 16), 1.0: (19, 16),
                2.0: (18, 15), 5.0: (18, 15), 10.0: (18, 15), 30.0: (18, 15),
                100.0: (17, 14)}
        # agent_1's first step at w = 2 is ahead of a wait by 0.0009 in Q, the size of the
        # engine's difference from brute force; with a wait the two finish one transition later
        also = {2.0: (19, 16)}
        for w, expected in done.items():
            with self.subTest(weight=w):
                r = run_pairing(self.scene, ("exchange", "exchange"), weights=(w, w),
                                cfg=self.cfg, n_steps=60)
                self.assertIn(tuple(r["done_at"]), (expected, also.get(w, expected)))
                self.assertEqual([led["put_downs"] for led in r["ledger"]],
                                 [0, 0] if expected == (None, None) else [1, 1])
                self.assertEqual([led["handed_over"] for led in r["ledger"]], [0, 0])
        cfg = dataclasses.replace(self.cfg, hand_over=True)
        for w in done:
            with self.subTest(weight=w, hand_over=True):
                r = run_pairing(self.scene, ("exchange", "exchange"), weights=(w, w), cfg=cfg,
                                n_steps=60)
                self.assertEqual(tuple(r["done_at"]), (14, 13))
                self.assertEqual([e for e in self.events(r) if e[1] != "picked up"],
                                 [(7, "put down", 0, 1), (8, "handed over", 1, 0),
                                  (13, "set down", 1, 1), (14, "set down", 0, 0)])
        cfg = dataclasses.replace(self.cfg, competition="table")
        table = {0.5: (None, None), 0.8: (20, 17), 1.0: (20, 17), 2.0: (19, 18),
                 5.0: (18, 15), 30.0: (18, 15), 100.0: (17, 14)}
        for w, expected in table.items():
            with self.subTest(weight=w, competition="table"):
                r = run_pairing(self.scene, ("exchange", "exchange"), weights=(w, w), cfg=cfg,
                                n_steps=60)
                self.assertEqual(tuple(r["done_at"]), expected)


class SceneTests(unittest.TestCase):
    def test_candidates_and_objects(self):
        scene = six_object_scene()
        self.assertEqual(len(scene.candidates), 6)
        self.assertTrue(set(scene.destinations) <= set(scene.candidates))
        self.assertEqual(scene.object_cells[:2], scene.pickups)
        self.assertEqual(len(scene.object_cells), 6)
        self.assertEqual(len(two_room_scene().object_cells), 2)
        described = scene.to_json()
        self.assertEqual(np.asarray(described["grid"]).shape, scene.grid.shape)
        json.dumps(to_json(described))

    def test_each_robot_starts_next_to_its_partners_object(self):
        scene = two_room_scene()
        for i in (0, 1):
            start, theirs = scene.starts[i], scene.pickups[1 - i]
            self.assertEqual(abs(start[0] - theirs[0]) + abs(start[1] - theirs[1]), 1)


if __name__ == "__main__":
    unittest.main()
