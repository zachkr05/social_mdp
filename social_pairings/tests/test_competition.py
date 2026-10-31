"""Do competitors compete?  The regression test of the five readings of competition.

``Config.competition``:

* "race_then_hinder" (the default until 2026-10-08, under which these tests run:
  ``told.py``): a competitor looks after its own goal; once that is done -- by itself or,
  for it, by its partner -- it works against its partner.
* "race": the same until its own goal is done, and nothing about its partner afterwards: it
  never helps, and it stops.
* "table": Table I as printed.  At l = 1 its entry is the conflict entry, ``-r(g_j)``: on
  the six-object scene two such robots take each other's object and neither goal is ever
  done.  Against a cooperative partner it is ``+R_j``: the competitor delivers for it.
* "table_hostile": the table with that cell turned to ``-R_j``.  Against a cooperator the
  competitor is then a conflict robot: it takes the cooperator's object and keeps it
  (:class:`HostileTableTests`).
* "rival": a race against every partner but a cooperative one, against which it is the
  competitor of "table_hostile".  Two rivals each do their own task, where two competitors
  of the table at level 2 never do -- at w = 30 they step to and fro and never touch an
  object (:class:`RivalTests`).

A competitor of a race form reasons at level 1 unless level 2 is asked for
(:class:`CompetitionTests`: the default levels).  At level 2 its term is the same, but it
is told its partner's social goal and acts on what it expects of its partner before its own
goal is done (:class:`LevelTwoCompetitorTests`).
"""
import os
import unittest

import numpy as np

from social_pairings import Scene, six_object_scene, two_room_scene
from social_pairings import goals as G
from social_pairings.constants import GIVE
from social_pairings.hindrance import conduct

from .told import Config, default_levels, levels_used, run_baseline, run_pairing

SLOW = os.environ.get("SOCIAL_PAIRINGS_SLOW", "") not in ("", "0")

SCENE = six_object_scene()
WEIGHTS = (0.8, 1.0, 2.0, 30.0)
STEPS = 40
RACES = ("race_then_hinder", "race")
OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}              # of a move: +y / -y, +x / -x


def own_deliveries(result) -> list:
    """(transition, robot) of every object a robot set down for its own goal."""
    return [(t, robot) for t, what, robot, obj, other in result["events"]
            if what == "set down" and obj == robot == other]


def reversals_until(result, robot: int, last: int) -> int:
    """How often the robot took back its previous move in transitions 1..last."""
    moves = [row["action"][robot] for row in result["trace"][:last]
             if row["action"][robot] < 4]
    return sum(1 for a, b in zip(moves, moves[1:]) if OPPOSITE[a] == b)


def events(result) -> list:
    return [tuple(e) for e in result["events"]]


def same_rollout(a, b) -> bool:
    """Are two records one rollout?  Every action, every event, the finishes, the ending,
    what is in the hands -- and every Q value of every transition, bit for bit."""
    if (len(a["trace"]) != len(b["trace"]) or events(a) != events(b)
            or tuple(a["done_at"]) != tuple(b["done_at"]) or a["stopped_by"] != b["stopped_by"]
            or tuple(a["holds"]) != tuple(b["holds"])):
        return False
    return all(tuple(x["action"]) == tuple(y["action"])
               and np.array_equal(np.asarray(x["q"], dtype=float),          # (2, 6)
                                  np.asarray(y["q"], dtype=float), equal_nan=True)
               for x, y in zip(a["trace"], b["trace"]))


def same_ledgers(a, b) -> bool:
    """Is every entry of each robot's ledger the same in two records: what it did, and
    what it was paid (the social term read off the transitions that came about)?"""
    return all(sorted((k, repr(v)) for k, v in x.items())
               == sorted((k, repr(v)) for k, v in y.items())
               for x, y in zip(a["ledger"], b["ledger"]))


def far_room() -> Scene:
    """The two-room map with R1's destination moved to the far room: alone R1 finishes 8
    transitions after R0 (9 / 17)."""
    return Scene(**{**two_room_scene().__dict__, "destinations": ((7, 1), (5, 5))})


def one_cell_door() -> Scene:
    """The two-room map with a doorway one cell wide: one robot has to let the other
    through."""
    grid = two_room_scene().grid.copy()
    grid[4, 3] = np.inf
    return Scene(**{**two_room_scene().__dict__, "grid": grid})


class CompetitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(num_threads=2)
        cls.baseline = run_baseline(SCENE, cfg=cls.cfg, n_steps=STEPS)

    def compete(self, weight, scene=SCENE, partner="competition", **options):
        cfg = Config(num_threads=2, **options)
        return run_pairing(scene, ("competition", partner), weights=(weight, weight), cfg=cfg,
                           n_steps=STEPS)

    def assert_races(self, result, done_at):
        self.assertTrue(result["numerically_accepted"])
        self.assertEqual(result["stopped_by"], "done")
        self.assertEqual(tuple(result["done_at"]), tuple(done_at))
        # each robot picked up its own object and nothing else, and set it down itself
        picked = sorted((robot, obj) for _, what, robot, obj, _ in result["events"]
                        if what == "picked up")
        self.assertEqual(picked, [(0, 0), (1, 1)])
        self.assertEqual(sorted(own_deliveries(result)),
                         sorted((t, i) for i, t in enumerate(done_at)))
        for i in (0, 1):
            self.assertEqual(result["ledger"][i]["delivered_by"], i)
            self.assertEqual(conduct(result, i)["wasted_hand_actions"], 0)   # no wasted one,
            self.assertLessEqual(reversals_until(result, i, done_at[i]), 1)    # no pacing

    # ---- two competitors ------------------------------------------------------------------
    def test_two_competitors_each_do_their_own_task(self):
        """As fast as two robots without social goals (9 and 11 transitions), whatever the
        weight, under both race forms."""
        self.assertEqual(tuple(self.baseline["done_at"]), (9, 11))
        for mode in RACES:
            for weight in WEIGHTS:
                with self.subTest(competition=mode, weight=weight):
                    result = self.compete(weight, competition=mode)
                    self.assert_races(result, (9, 11))
                    self.assertEqual(tuple(result["levels_used"]), (1, 1))
                    self.assertIn("[race, then hinder]" if mode == "race_then_hinder"
                                  else "[race form]", result["social_terms"][0])

    def test_against_a_partner_without_social_goal(self):
        for mode in RACES:
            for weight in (1.0, 30.0):
                with self.subTest(competition=mode, weight=weight):
                    self.assert_races(self.compete(weight, partner="none", competition=mode),
                                      (9, 11))

    def test_once_done_a_competitor_goes_for_its_partner(self):
        """Two competitors.  R0 is done at transition 9 and R1 two transitions later.
        "race_then_hinder": R0 does not stay on its destination -- it sets off for the
        other room, where R1 is: two moves, to the cell two steps along the way to the
        doorway (up and left in either order).  "race": it stays."""
        home = tuple(SCENE.destinations[0])
        self.assertEqual(home, (7, 1))
        for mode, cell in (("race_then_hinder", (6, 2)), ("race", home)):
            result = self.compete(1.0, competition=mode)
            self.assertEqual(tuple(result["done_at"]), (9, 11))
            trace = result["trace"]
            self.assertEqual(tuple(trace[8]["next_pos"][0]), home)        # done: on its goal
            self.assertEqual(tuple(trace[10]["next_pos"][0]), cell, mode)
            moved = [row["action"][0] < 4 for row in trace[9:11]]
            self.assertEqual(moved, [mode == "race_then_hinder"] * 2)

    def test_the_robot_far_behind_still_races(self):
        """R1's destination moved to the far room: alone it finishes 8 transitions after R0.
        Under both race forms it goes for its own object and never touches R0's.
        "race": it loses w rho in each transition from 10 to 16 and finishes at 17.
        "race_then_hinder": R0, done at 9, comes back and stands in its way, and R1 never
        gets through (it still has its object in its hands at the step limit)."""
        far = far_room()
        base = run_baseline(far, cfg=self.cfg, n_steps=STEPS)
        self.assertEqual(tuple(base["done_at"]), (9, 17))
        weight = 30.0
        result = self.compete(weight, scene=far, competition="race")
        self.assert_races(result, (9, 17))
        for row in result["trace"]:
            if 10 <= row["step"] <= 16:
                self.assertAlmostEqual(float(row["social_reward"][1][row["action"][1]]),
                                       -weight * self.cfg.rho, places=4)
        for weight in (1.0, 30.0):
            with self.subTest(competition="race_then_hinder", weight=weight):
                result = self.compete(weight, scene=far)
                self.assertTrue(result["numerically_accepted"])
                self.assertEqual(tuple(result["done_at"]), (9, None))
                self.assertEqual(result["stopped_by"], "cap")
                self.assertEqual(events(result), [(7, "picked up", 0, 0, None),
                                                  (9, "set down", 0, 0, 0),
                                                  (9, "picked up", 1, 1, None)])
                self.assertEqual(tuple(result["holds"]), (-1, 1))
                self.assertGreaterEqual(result["ledger"][1]["blocked_by_partner"], 5)
                self.assertGreater(result["ledger"][0]["moves"], 20)     # R0 keeps moving
                for row in result["trace"][9:]:          # R1 loses w rho for being behind
                    self.assertAlmostEqual(float(row["social_reward"][1][row["action"][1]]),
                                           -weight * self.cfg.rho, places=4)

    def test_the_race_term_in_the_rewards(self):
        """The expected social reward of the action taken, transition by transition.  Zero
        while nobody has finished.  R0 lands on its destination in transition 9.  R1 loses
        w rho in transition 10 (R0 is done, it is not) and is paid in transition 11, in
        which it finishes; in transition 9 it does not know R0's destination and expects a
        part of the loss.
        "race": R0 is paid w rho from transition 9 on.  "race_then_hinder": its lead on
        top, w (rho - r(g_1)) as it takes R1's goal reward to be, so between w rho and
        2 w rho while R1 is on its way and w rho once both are done."""
        weight = 2.0
        prize = weight * self.cfg.rho
        for mode in RACES:
            with self.subTest(competition=mode):
                result = self.compete(weight, competition=mode)
                self.assertEqual(tuple(result["done_at"]), (9, 11))
                self.assertEqual(result["steps"], 12)
                paid = {i: {row["step"]: float(row["social_reward"][i][row["action"][i]])
                            for row in result["trace"]} for i in (0, 1)}
                for i in (0, 1):
                    for t in range(1, 9):
                        self.assertAlmostEqual(paid[i][t], 0.0, places=5, msg=(i, t))
                self.assertTrue(-prize < paid[1][9] < 0.0, paid[1][9])
                self.assertAlmostEqual(paid[1][10], -prize, places=5)
                self.assertAlmostEqual(paid[1][11], prize, places=5)
                self.assertAlmostEqual(paid[1][12], prize, places=5)
                self.assertAlmostEqual(paid[0][12], prize, places=5)       # both are done
                # What each robot was paid in the end (the ledger, from the states that
                # came about): R1 lost the prize in 9 and 10 and won it in 11 and 12.
                self.assertAlmostEqual(result["ledger"][1]["social_reward"], 0.0, places=3)
                if mode == "race":
                    for t in (9, 10, 11):
                        self.assertAlmostEqual(paid[0][t], prize, places=5)
                    self.assertAlmostEqual(result["ledger"][0]["social_reward"], 4 * prize,
                                           places=4)
                else:
                    for t in (9, 10, 11):
                        self.assertTrue(prize < paid[0][t] < 2 * prize, (t, paid[0][t]))
                    self.assertTrue(paid[0][9] > paid[0][10] > paid[0][11], paid[0])
                    self.assertTrue(4 * prize < result["ledger"][0]["social_reward"]
                                    < 6 * prize)

    def test_which_level_a_competitor_reasons_at(self):
        """A race form: level 1 when no level is asked for, whoever its partner is; the
        level asked for otherwise.  The two readings of the table: level 2 by default."""
        comp, coop = G.COMPETITION, G.COOPERATION
        for mode in RACES:
            cfg = Config(competition=mode)
            self.assertEqual(default_levels((comp, comp), cfg), (1, 1))
            self.assertEqual(default_levels((comp, G.NONE), cfg), (1, 2))
            self.assertEqual(default_levels((G.HINDER, comp), cfg), (1, 1))
            self.assertEqual(default_levels((G.CONFLICT, comp), cfg), (2, 1))
            # whoever its partner is, and whether or not the partner's social goal is known
            self.assertEqual(default_levels((comp, coop), cfg), (1, 2))
            self.assertEqual(default_levels((comp, G.EXCHANGE), cfg), (1, 2))
            self.assertEqual(default_levels((comp, G.COERCION), cfg), (1, 2))
            self.assertEqual(default_levels((comp, coop), Config(
                competition=mode, partner_social_goal="inferred")), (1, 2))
            # with weight 0 a competitor has no social goal
            self.assertEqual(default_levels((comp, coop), cfg, weights=(0.0, 1.0)), (2, 2))
            # asked for, a level is the level used: a competitor at level 2 ...
            self.assertEqual(levels_used((comp, comp), (2, 2), cfg), (2, 2))
            self.assertEqual(levels_used((comp, coop), (2, 2), cfg), (2, 2))
            self.assertEqual(levels_used((comp, comp), (1, 2), cfg), (1, 2))
            self.assertEqual(levels_used((comp, coop), (0, 2), cfg), (0, 2))
            # ... but not a hinderer
            self.assertEqual(levels_used((G.HINDER, comp), (2, 2), cfg), (1, 2))
        self.assertEqual(default_levels((comp, comp)), (1, 1))          # the default
        self.assertEqual(default_levels(("competition", "cooperation")), (1, 2))
        # the table, as printed and with the cell against a cooperator turned -- and a
        # rival, whoever its partner is and whether or not it is told
        for mode in ("table", "table_hostile", "rival"):
            table = Config(competition=mode)
            self.assertEqual(default_levels((comp, comp), table), (2, 2))
            self.assertEqual(default_levels((comp, coop), table), (2, 2))
            self.assertEqual(default_levels((comp, G.NONE), table), (2, 2))
            self.assertEqual(default_levels((G.HINDER, comp), table), (1, 2))
            self.assertEqual(default_levels((comp, coop), Config(
                competition=mode, partner_social_goal="inferred")), (2, 2))
            self.assertEqual(levels_used((comp, comp), (2, 2), table), (2, 2))
            self.assertEqual(levels_used((comp, coop), (1, 2), table), (1, 2))
            self.assertEqual(levels_used((comp, comp), (1, 1), table), (1, 1))

    def test_the_table_as_printed_is_a_stand_off(self):
        """``competition="table"`` at level 1: each takes the other's object, at once, and
        keeps it."""
        result = run_pairing(SCENE, ("competition", "competition"), weights=(1.0, 1.0),
                             levels=(1, 1), cfg=Config(num_threads=2, competition="table"),
                             n_steps=STEPS)
        self.assertEqual(tuple(result["done_at"]), (None, None))
        self.assertEqual(tuple(result["holds"]), (1, 0))
        taken = sorted((robot, obj, t) for t, what, robot, obj, _ in result["events"]
                       if what == "picked up")
        self.assertEqual([(robot, obj) for robot, obj, _ in taken], [(0, 1), (1, 0)])
        self.assertTrue(all(t <= 3 for _, _, t in taken))

    # ---- a competitor and a cooperator: the two race forms and the table as printed ----------
    def test_against_a_cooperator_it_takes_its_object_once_its_own_goal_is_done(self):
        """The default, "race_then_hinder".  The cooperator picks the competitor's object up
        at transition 2.  The competitor, two cells on its way to it, turns round and goes
        back to its start, next to the cooperator's object.  Once the cooperator has set
        the competitor's object down on the competitor's destination, the competitor picks
        the cooperator's object up -- one or two transitions later, not before -- and
        keeps it: the cooperator never finishes.  (At w = 1 and 2 the cooperator tries one
        wrong destination first.)

        Which transition the competitor turns in, and whether it takes the object one or
        two transitions after its goal is done, rests on differences in Q of 1e-3 at small
        weights; the test pins them only at w = 30, where they are 0.3.

        At w = 0.8, since 2026-10-16 (five hand actions; until then as above, 5 / never):
        the cooperator picks the competitor's object up at 7 and puts it back at 14, each
        fetches its own (19 / 25), and the competitor never touches the cooperator's."""
        course = {1.0: (10, [6]), 2.0: (10, [6]), 30.0: (5, [])}
        for weight, (placed, in_vain) in course.items():
            with self.subTest(weight=weight):
                result = self.compete(weight, partner="cooperation")
                self.assertTrue(result["numerically_accepted"])
                self.assertEqual(tuple(result["levels_used"]), (1, 2))
                self.assertIn("[race, then hinder]", result["social_terms"][0])
                self.assertEqual(tuple(result["done_at"]), (placed, None))
                self.assertEqual(result["stopped_by"], "deadlock")
                taken = [t for t, what, robot, obj, _ in result["events"]
                         if what == "picked up" and (robot, obj) == (0, 1)]
                self.assertEqual(len(taken), 1)
                self.assertTrue(placed < taken[0] <= placed + 2, (placed, taken))
                self.assertEqual(events(result),
                                 [(2, "picked up", 1, 0, None)]
                                 + [(t, "set down in vain", 1, 0, 0) for t in in_vain]
                                 + [(placed, "set down", 1, 0, 0),
                                    (taken[0], "picked up", 0, 1, None)])
                self.assertEqual(tuple(result["holds"]), (1, -1))        # ... to the end
                mine = result["ledger"][0]
                self.assertEqual((mine["hand_actions"], mine["pickups"], mine["handed_over"],
                                  mine["set_down_for_partner"], mine["failed_set_downs"]),
                                 (1, 1, 0, 0, 0))
                self.assertEqual(mine["delivered_by"], 1)
                # two cells toward its object, and back at its start before its goal is done
                trace = result["trace"]
                self.assertEqual([row["action"][0] for row in trace[:2]], [1, 1])
                start = tuple(SCENE.starts[0])
                self.assertIn(start, [tuple(trace[t - 1]["next_pos"][0]) for t in (4, 5)])
                if weight == 30.0:
                    self.assertEqual(taken[0], placed + 1)
                    self.assertEqual([row["action"][0] for row in trace[:6]],
                                     [1, 1, 3, 3, 0, 5])     # right x2, left x2, up, pick
        result = self.compete(0.8, partner="cooperation")
        self.assertEqual(tuple(result["done_at"]), (19, 25))
        self.assertEqual(events(result),
                         [(7, "picked up", 1, 0, None), (14, "put back", 1, 0, None),
                          (17, "picked up", 0, 0, None), (19, "set down", 0, 0, 0),
                          (22, "picked up", 1, 1, None), (25, "set down", 1, 1, 1)])
        self.assertEqual(result["ledger"][0]["hand_actions"], 0)

    def test_with_the_seats_swapped_the_cooperator_does_not_come_for_its_object(self):
        """cooperation / competition, w >= 1: the cooperator sets the competitor's object
        down at transition 5 and stays where it is -- for its own goal it expects nothing
        from going to its object, with the competitor at that object's side.  The
        competitor goes back to stand next to the cooperator's object and does not take
        it: stepping onto its place now is 0.002 below staying in its Q.  It does nothing
        for the cooperator.

        At w = 30 that 0.002 is a difference between values of 7600, so there the test
        allows the competitor to take the object as well."""
        for weight in (1.0, 30.0):
            with self.subTest(weight=weight):
                result = run_pairing(SCENE, ("cooperation", "competition"),
                                     weights=(weight, weight), cfg=self.cfg, n_steps=STEPS)
                self.assertEqual(tuple(result["done_at"]), (None, 5))
                happened = events(result)
                self.assertEqual(happened[:2], [(2, "picked up", 0, 1, None),
                                                (5, "set down", 0, 1, 1)])
                mine = result["ledger"][1]
                self.assertEqual((mine["handed_over"], mine["set_down_for_partner"],
                                  mine["failed_set_downs"]), (0, 0, 0))
                self.assertEqual(result["ledger"][0]["moves"], 3)     # the cooperator's
                end = result["trace"][-1]["next_pos"]
                place = SCENE.pickups[0]                 # of the cooperator's object
                gap = abs(end[1][0] - place[0]) + abs(end[1][1] - place[1])
                if weight == 1.0:
                    self.assertEqual(happened[2:], [])
                    self.assertEqual((mine["hand_actions"], gap), (0, 1))
                else:
                    self.assertLessEqual(gap, 1)
                    self.assertTrue(all(what == "picked up" and (robot, obj) == (1, 0) and t > 5
                                        for t, what, robot, obj, _ in happened[2:]), happened)

    def test_a_competitor_that_stops_once_its_goal_is_done(self):
        """``competition="race"``.  The cooperator brings the competitor its object (O1 on
        the competitor's destination at transition 5) and then fetches its own (17).  The
        competitor walks two cells toward its object, sees it carried for it, and waits:
        it never uses its hands, and nobody delivers the cooperator's object but the
        cooperator.  At w = 0.8 the cooperator tries two wrong destinations first (19 / 31),
        and since 2026-10-16 (five hand actions; 5 / 17 until then) at w = 1 as well (23 /
        33; the competitor walks three cells)."""
        for weight in (2.0, 30.0):
            with self.subTest(weight=weight):
                result = self.compete(weight, partner="cooperation", competition="race")
                self.assertTrue(result["numerically_accepted"])
                self.assertEqual(tuple(result["levels_used"]), (1, 2))
                self.assertIn("[race form]", result["social_terms"][0])
                self.assertEqual(tuple(result["done_at"]), (5, 17))
                self.assertEqual(events(result),
                                 [(2, "picked up", 1, 0, None), (5, "set down", 1, 0, 0),
                                  (15, "picked up", 1, 1, None), (17, "set down", 1, 1, 1)])
                mine = result["ledger"][0]
                self.assertEqual((mine["hand_actions"], mine["pickups"],
                                  mine["set_down_for_partner"], mine["moves"]), (0, 0, 0, 2))
                self.assertEqual(result["ledger"][1]["delivered_by"], 1)
        for weight, done_at in ((0.8, (19, 31)), (1.0, (23, 33))):
            with self.subTest(weight=weight):
                result = self.compete(weight, partner="cooperation", competition="race")
                self.assertEqual(tuple(result["done_at"]), done_at)
                self.assertEqual({robot for _, _, robot, _, _ in result["events"]}, {1})
                self.assertEqual(result["ledger"][0]["hand_actions"], 0)
                self.assertEqual(result["ledger"][1]["failed_set_downs"], 2)

    def test_against_a_cooperator_as_table_one_prints_it(self):
        """``competition="table"``: against a cooperative partner the table's competition is
        +R_j, the partner's reward.  The competitor then sets its partner's object down on
        its partner's destination: both goals are done at transition 5."""
        result = self.compete(1.0, partner="cooperation", competition="table")
        self.assertEqual(tuple(result["levels_used"]), (2, 2))
        self.assertEqual(tuple(result["done_at"]), (5, 5))
        self.assertIn("+1 R_j", result["social_terms"][0])
        self.assertIn((5, "set down", 0, 1, 1), events(result))
        self.assertEqual(result["ledger"][0]["set_down_for_partner"], 1)

    def test_a_racing_competitor_does_nothing_for_any_partner(self):
        """At its default level, 1.  Whatever its partner's social goal, and in either seat,
        under both race forms: a competitor never hands its partner's object over, and
        never sets it down on a destination or tries to.
        "race": it never has it in its hands either.
        "race_then_hinder": it takes it only once its own goal is done."""
        for mode in RACES:
            cfg = Config(num_threads=2, competition=mode)
            for seat in (0, 1):
                for partner in G.SOCIAL_GOALS:
                    for weight in (1.0, 30.0):
                        with self.subTest(competition=mode, seat=seat, partner=partner,
                                          weight=weight):
                            goals = (("competition", partner) if seat == 0
                                     else (partner, "competition"))
                            result = run_pairing(SCENE, goals, weights=(weight, weight),
                                                 cfg=cfg, n_steps=STEPS)
                            self.assertTrue(result["numerically_accepted"])
                            self.assertIn("race", result["social_terms"][seat])
                            mine = result["ledger"][seat]
                            self.assertEqual((mine["handed_over"], mine["set_down_for_partner"],
                                              mine["failed_set_downs"]), (0, 0, 0))
                            done = result["done_at"][seat]
                            foreign = [t for t, _, robot, obj, _ in result["events"]
                                       if robot == seat and obj != seat]
                            if mode == "race":
                                self.assertEqual(foreign, [])
                                self.assertEqual(mine["steps_holding_foreign"], 0)
                            else:
                                self.assertTrue(all(done is not None and t > done
                                                    for t in foreign), (foreign, done))


class HostileTableTests(unittest.TestCase):
    """``competition="table_hostile"``: Table I with the competitor's cell against a
    cooperative (or exchanging) partner turned from ``+R_j`` to ``-R_j``.  A competitor of
    the table reasons at level 2 by default, and with that cell turned it works against a
    cooperator instead of delivering for it.

    With the cell turned the competition row is the conflict row, but for the cell against
    a coercing partner with the robot's own destination; the column l = 1 is the conflict
    entry anyway.  So against a cooperative, an exchanging and a competing partner the
    rollout is, to the last Q value, the one of a conflict robot in the competitor's place
    (:func:`same_rollout`); that is by construction, and the tests hold it down."""

    def pair(self, goals, weight, levels=None, **options):
        options.setdefault("competition", "table_hostile")
        result = run_pairing(SCENE, goals, weights=(weight, weight), levels=levels,
                             cfg=Config(num_threads=2, **options), n_steps=60)
        self.assertTrue(result["numerically_accepted"])
        return result

    def test_the_term_and_the_levels_of_the_record(self):
        result = run_pairing(SCENE, ("competition", "cooperation"), n_steps=3,
                             cfg=Config(num_threads=2, competition="table_hostile"))
        self.assertEqual((tuple(result["levels"]), tuple(result["levels_used"])),
                         ((2, 2), (2, 2)))
        self.assertIn("competition vs cooperation: w (-1 R_j)", result["social_terms"][0])
        self.assertIn("the sign turned: table_hostile", result["social_terms"][0])
        self.assertEqual(result["config"]["competition"], "table_hostile")
        printed = run_pairing(SCENE, ("competition", "cooperation"), n_steps=3,
                              cfg=Config(num_threads=2, competition="table"))
        self.assertIn("competition vs cooperation: w (+1 R_j)", printed["social_terms"][0])
        self.assertEqual(result["social_terms"][1], printed["social_terms"][1])

    def test_against_a_cooperator_it_takes_its_object_and_keeps_it(self):
        """competition / cooperation at the default levels, (2, 2).  The cooperator still
        delivers the competitor's object (transition 5; 7 at w = 0.8).  The competitor
        steps onto the place of the cooperator's object, picks it up and holds it to the
        end: the cooperator never finishes.  It never sets that object down for the
        cooperator, hands it over or tries to.  Action for action, and Q for Q, it is the
        rollout of conflict / cooperation.

        In which transition it takes the object (3, 6, 6, 3 at w = 0.8, 1, 2, 30) is not
        pinned.  One transition earlier, taking it is 0.005, 0.10, 0.014 and 0.002 below
        waiting in its Q; in the transition it takes it, waiting is 0.17, 0.012, 0.028 and
        0.31 below."""
        for weight in WEIGHTS:
            with self.subTest(weight=weight):
                result = self.pair(("competition", "cooperation"), weight)
                self.assertEqual(tuple(result["levels_used"]), (2, 2))
                self.assertEqual(result["done_at"][1], None)
                self.assertIn(result["done_at"][0], (5, 7))
                self.assertEqual(result["stopped_by"], "deadlock")
                mine = [e for e in events(result) if e[2] == 0]
                self.assertEqual(len(mine), 1)
                self.assertEqual(mine[0][1:], ("picked up", 0, 1, None))
                self.assertLessEqual(mine[0][0], 7)
                self.assertEqual(tuple(result["holds"]), (1, -1))         # ... to the end
                led = result["ledger"][0]
                self.assertEqual((led["hand_actions"], led["pickups"], led["handed_over"],
                                  led["set_down_for_partner"], led["failed_set_downs"]),
                                 (1, 1, 0, 0, 0))
                self.assertEqual(led["delivered_by"], 1)       # its own goal: the cooperator
                self.assertTrue(same_rollout(
                    result, self.pair(("conflict", "cooperation"), weight)))

    def test_the_table_as_printed_has_it_deliver_for_the_cooperator(self):
        """The same pairing with the cell as printed: both goals done at 5, the
        cooperator's by the competitor's hands.  One cell, the other outcome."""
        for weight in (1.0, 30.0):
            with self.subTest(weight=weight):
                printed = self.pair(("competition", "cooperation"), weight,
                                    competition="table")
                self.assertEqual(tuple(printed["done_at"]), (5, 5))
                self.assertEqual(printed["ledger"][0]["set_down_for_partner"], 1)
                self.assertEqual(printed["ledger"][1]["delivered_by"], 0)

    def test_with_the_seats_swapped_it_does_nothing_for_the_cooperator(self):
        """cooperation / competition: the rollout of cooperation / conflict.  The
        cooperator delivers the competitor's object at 5 and never finishes its own task;
        the competitor never sets the cooperator's object down for it or hands it over.
        (At w = 1 it takes that object at transition 29, at w = 30 it never does and the
        cooperator does not go for it either: not pinned.)  As printed, at w = 30: 5 / 5."""
        weights = WEIGHTS if SLOW else (1.0, 30.0)
        for weight in weights:
            with self.subTest(weight=weight):
                result = self.pair(("cooperation", "competition"), weight)
                self.assertEqual(result["done_at"][0], None)
                led = result["ledger"][1]
                self.assertEqual((led["handed_over"], led["set_down_for_partner"],
                                  led["failed_set_downs"]), (0, 0, 0))
                self.assertNotEqual(result["ledger"][0]["delivered_by"], 1)
                if weight >= 1.0:
                    self.assertEqual(result["done_at"][1], 5)
                    self.assertEqual(led["delivered_by"], 0)
                self.assertTrue(same_rollout(
                    result, self.pair(("cooperation", "conflict"), weight)))
        printed = self.pair(("cooperation", "competition"), 30.0, competition="table")
        self.assertEqual(tuple(printed["done_at"]), (5, 5))

    def test_an_exchanging_partner_is_the_other_half_of_the_cell(self):
        """Table I prints one cell for a cooperative and an exchanging partner; both are
        turned.  competition / exchange at w = 30.  As printed the competitor delivers the
        exchanger's object (5) and then its own (17).  Turned, it picks the exchanger's
        object up and keeps it -- and, its hands full, gives up its own task: neither goal
        is done, as with a conflict robot in its place."""
        hostile = self.pair(("competition", "exchange"), 30.0)
        self.assertEqual(tuple(hostile["done_at"]), (None, None))
        self.assertEqual(events(hostile), [(2, "picked up", 0, 1, None)])
        self.assertEqual(tuple(hostile["holds"]), (1, -1))
        self.assertTrue(same_rollout(hostile, self.pair(("conflict", "exchange"), 30.0)))
        printed = self.pair(("competition", "exchange"), 30.0, competition="table")
        self.assertEqual(tuple(printed["done_at"]), (17, 5))
        self.assertEqual(printed["ledger"][0]["set_down_for_partner"], 1)

    def test_with_an_inferred_partner_goal(self):
        """The paper's mode: the competitor is not told its partner's social goal and holds
        a belief over it.  competition / cooperation at w = 1.  The competitor ends up sure
        that its partner cooperates -- and, with the cell turned, takes its object at
        transition 2 and keeps it (5 / never), where as printed it delivers it (5 / 6).
        It acts as the conflict robot does; its Q is not the conflict robot's here, because
        the untouched coercion cell is in its mix of hypotheses."""
        options = dict(partner_social_goal="inferred")
        hostile = self.pair(("competition", "cooperation"), 1.0, **options)
        self.assertEqual(tuple(hostile["done_at"]), (5, None))
        self.assertEqual([e for e in events(hostile) if e[2] == 0],
                         [(2, "picked up", 0, 1, None)])
        self.assertEqual(tuple(hostile["holds"]), (1, -1))
        led = hostile["ledger"][0]
        self.assertEqual((led["handed_over"], led["set_down_for_partner"],
                          led["failed_set_downs"]), (0, 0, 0))
        belief = np.asarray(hostile["final_social_belief"][0], dtype=float)
        self.assertGreater(belief[G.COOPERATION], 0.99)        # it knows whom it faces
        self.assertIn("-R_j against a cooperative or exchanging one: table_hostile",
                      hostile["social_terms"][0])
        twin = self.pair(("conflict", "cooperation"), 1.0, **options)
        self.assertEqual([tuple(row["action"]) for row in hostile["trace"]],
                         [tuple(row["action"]) for row in twin["trace"]])
        self.assertFalse(same_rollout(hostile, twin))                    # ... but not in Q
        printed = self.pair(("competition", "cooperation"), 1.0, competition="table",
                            **options)
        self.assertEqual(tuple(printed["done_at"]), (5, 6))
        self.assertEqual(printed["ledger"][0]["set_down_for_partner"], 1)

    def test_two_of_them_are_two_competitors_of_the_table(self):
        """Nothing else is turned.  Two competitors (-R_j against each other, as printed)
        roll out as under "table" and as two conflict robots do: neither goal is ever
        done.  At level 1, where the table has one entry for every partner, a competitor
        against a cooperator rolls out as under "table" too."""
        weights = WEIGHTS if SLOW else (0.8,)
        for weight in weights:
            with self.subTest(weight=weight):
                hostile = self.pair(("competition", "competition"), weight)
                self.assertEqual(tuple(hostile["done_at"]), (None, None))
                self.assertTrue(same_rollout(hostile, self.pair(
                    ("competition", "competition"), weight, competition="table")))
                self.assertTrue(same_rollout(hostile, self.pair(
                    ("conflict", "conflict"), weight)))
        for goals in (("competition", "cooperation"), ("competition", "competition")):
            with self.subTest(goals=goals, levels=(1, 1)):
                one = self.pair(goals, 1.0, levels=(1, 1))
                self.assertEqual(tuple(one["levels_used"]), (1, 1))
                self.assertTrue(same_rollout(one, self.pair(goals, 1.0, levels=(1, 1),
                                                            competition="table")))

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_it_differs_from_the_table_against_a_helpful_partner_only(self):
        """Every social goal of a partner, either seat, w = 1 (14 pairings; with w = 30 as
        well the same holds, README).  The rollout differs from the one under "table"
        where the partner is cooperative or exchanging and nowhere else; it is the rollout
        of a conflict robot in the competitor's place wherever the partner is not a
        coercing one; and the competitor never sets its partner's object down for it or
        hands it over."""
        for seat in (0, 1):
            for partner in G.SOCIAL_GOALS:
                for weight in (1.0,):
                    with self.subTest(seat=seat, partner=partner, weight=weight):
                        other = "conflict" if partner == "competition" else partner
                        goals, twin = ("competition", partner), ("conflict", other)
                        if seat == 1:
                            goals, twin = goals[::-1], twin[::-1]
                        hostile = self.pair(goals, weight)
                        printed = self.pair(goals, weight, competition="table")
                        self.assertEqual(same_rollout(hostile, printed),
                                         partner not in ("cooperation", "exchange"))
                        if partner != "coercion":
                            self.assertTrue(same_rollout(hostile, self.pair(twin, weight)))
                        led = hostile["ledger"][seat]
                        self.assertEqual((led["handed_over"], led["set_down_for_partner"]),
                                         (0, 0))


class RivalTests(unittest.TestCase):
    """``competition="rival"``: competition as "I want to do better than you".  A rival races
    its partner -- the term of "race", at level 1 and at level 2: once its own goal is done
    nothing in it concerns its partner.  The one partner it works against is a cooperative
    one, which helps it: against that partner it is the competitor of "table_hostile".  It
    reasons at level 2 by default.

    With known partner goals a rollout with a rival is therefore, to the last Q value and
    the last entry of the ledgers, the rollout of one of those two readings at the same
    levels (:func:`same_rollout`, :func:`same_ledgers`): that is by construction, and the
    tests hold it down.  What it is for is the first two tests: two rivals each do their
    own task, where two competitors of the table do not."""

    assert_races = CompetitionTests.assert_races

    def pair(self, goals, weight, levels=None, scene=SCENE, **options):
        options.setdefault("competition", "rival")
        weights = weight if isinstance(weight, tuple) else (weight, weight)
        result = run_pairing(scene, goals, weights=weights, levels=levels,
                             cfg=Config(num_threads=2, **options), n_steps=60)
        self.assertTrue(result["numerically_accepted"])
        return result

    def test_two_rivals_each_do_their_own_task(self):
        """competition / competition at the default levels, (2, 2): 9 / 11 at every weight,
        as fast as two robots without social goals.  Each picks up its own object and no
        other, sets it down itself, wastes no hand action and takes at most one move back.  It
        is the rollout of two competitors of "race" asked for level 2."""
        twins = WEIGHTS if SLOW else (1.0, 30.0)
        for weight in WEIGHTS:
            with self.subTest(weight=weight):
                result = self.pair(("competition", "competition"), weight)
                self.assertEqual((tuple(result["levels"]), tuple(result["levels_used"])),
                                 ((2, 2), (2, 2)))
                self.assert_races(result, (9, 11))
                for led in result["ledger"]:
                    self.assertEqual((led["hand_actions"], led["steps_holding_foreign"],
                                      led["blocked_by_partner"]), (0, 0, 0))
                if weight in twins:
                    twin = self.pair(("competition", "competition"), weight, levels=(2, 2),
                                     competition="race")
                    self.assertTrue(same_rollout(result, twin))
                    self.assertTrue(same_ledgers(result, twin))

    def test_two_competitors_of_the_table_wait_for_each_other(self):
        """The same pairing under the table, at the same levels and w = 30: the rollout
        "rival" is there for.  Until 2026-10-16 (with "use") neither robot ever used its
        hands: each stood on two cells only, the place of the other's object and the cell
        beside it, every move taking back the one before -- each waited for the other to
        take its object first.  With the five hand actions they do take: each picks the
        other's object up (2, 4), they trade in the doorway (both give, 10), and agent_2,
        its own object in hand, stands on agent_1's destination to the end, agent_1 beside
        it -- a deadlock.  Neither goal is done either way."""
        result = self.pair(("competition", "competition"), 30.0, competition="table")
        self.assertEqual(tuple(result["levels_used"]), (2, 2))
        self.assertEqual(tuple(result["done_at"]), (None, None))
        self.assertEqual(events(result),
                         [(2, "picked up", 0, 1, None), (4, "picked up", 1, 0, None),
                          (10, "handed over", 1, 0, 0), (10, "handed over", 0, 1, 1)])
        self.assertEqual(tuple(result["holds"]), (0, 1))         # each its own, to the end
        self.assertEqual(result["stopped_by"], "deadlock")
        last = result["trace"][-1]["next_pos"]
        self.assertEqual(tuple(last[1]), tuple(SCENE.destinations[0]))
        self.assertEqual(sum(abs(a - b) for a, b in zip(last[0], last[1])), 1)

    def test_the_term_and_the_levels_of_the_record(self):
        result = run_pairing(SCENE, ("competition", "competition"), n_steps=3,
                             cfg=Config(num_threads=2, competition="rival"))
        self.assertEqual((tuple(result["levels"]), tuple(result["levels_used"])),
                         ((2, 2), (2, 2)))
        self.assertEqual(result["config"]["competition"], "rival")
        for text in result["social_terms"]:
            self.assertIn("competition vs competition", text)
            self.assertIn("[rival: a race]", text)
            self.assertNotIn("R_j", text)
        helped = run_pairing(SCENE, ("competition", "cooperation"), n_steps=3,
                             cfg=Config(num_threads=2, competition="rival"))
        self.assertIn("competition vs cooperation: w (-1 R_j)  [rival: against a helper",
                      helped["social_terms"][0])
        hostile = run_pairing(SCENE, ("competition", "cooperation"), n_steps=3,
                              cfg=Config(num_threads=2, competition="table_hostile"))
        self.assertEqual(helped["social_terms"][1], hostile["social_terms"][1])
        # a level-1 rival is told whom it faces
        one = run_pairing(SCENE, ("competition", "cooperation"), n_steps=3, levels=(1, 2),
                          cfg=Config(num_threads=2, competition="rival"))
        self.assertIn("w (-1 r(g_j))  [rival: against a helper", one["social_terms"][0])
        # a cooperator at level 0 has no social goal, whatever goal it was given: no helper
        zero = run_pairing(SCENE, ("competition", "cooperation"), n_steps=3, levels=(2, 0),
                           cfg=Config(num_threads=2, competition="rival"))
        self.assertIn("competition vs none", zero["social_terms"][0])
        self.assertIn("[rival: a race]", zero["social_terms"][0])
        # an exchanging partner is no helper either
        other = run_pairing(SCENE, ("competition", "exchange"), n_steps=3,
                            cfg=Config(num_threads=2, competition="rival"))
        self.assertIn("competition vs exchange", other["social_terms"][0])
        self.assertIn("[rival: a race]", other["social_terms"][0])

    def test_against_a_cooperator_it_is_the_competitor_of_the_hostile_table(self):
        """competition / cooperation: the cooperator delivers the rival's object (5; 7 at
        w = 0.8), and the rival takes the cooperator's and keeps it -- the cooperator never
        finishes.  With the seats swapped the cooperator never finishes either.  Each is
        the rollout of "table_hostile", every action and every Q value."""
        weights = WEIGHTS if SLOW else (1.0, 30.0)
        for weight in weights:
            with self.subTest(weight=weight):
                result = self.pair(("competition", "cooperation"), weight)
                self.assertEqual(tuple(result["levels_used"]), (2, 2))
                self.assertEqual(tuple(result["done_at"]), (7 if weight == 0.8 else 5, None))
                self.assertEqual(tuple(result["holds"]), (1, -1))
                led = result["ledger"][0]
                self.assertEqual((led["hand_actions"], led["pickups"], led["handed_over"],
                                  led["set_down_for_partner"], led["failed_set_downs"]),
                                 (1, 1, 0, 0, 0))
                twin = self.pair(("competition", "cooperation"), weight,
                                 competition="table_hostile")
                self.assertTrue(same_rollout(result, twin))
                self.assertTrue(same_ledgers(result, twin))
        for weight in (weights if SLOW else (30.0,)):
            with self.subTest(weight=weight, seats="swapped"):
                result = self.pair(("cooperation", "competition"), weight)
                self.assertEqual(result["done_at"][0], None)
                twin = self.pair(("cooperation", "competition"), weight,
                                 competition="table_hostile")
                self.assertTrue(same_rollout(result, twin))
                self.assertTrue(same_ledgers(result, twin))

    def test_an_exchanging_partner_is_raced(self):
        """An exchanging partner has no social term toward a competitor: it does not help
        a rival, and a rival does not work against it.  competition / exchange at w = 30:
        9 / 11, each with its own object -- the rollout of "race" at the same levels --
        where the competitor of "table_hostile" takes the exchanger's object and neither
        goal is done."""
        result = self.pair(("competition", "exchange"), 30.0)
        self.assertEqual(tuple(result["levels_used"]), (2, 2))
        self.assert_races(result, (9, 11))
        twin = self.pair(("competition", "exchange"), 30.0, levels=(2, 2), competition="race")
        self.assertTrue(same_rollout(result, twin))
        self.assertTrue(same_ledgers(result, twin))
        hostile = self.pair(("competition", "exchange"), 30.0, competition="table_hostile")
        self.assertEqual(tuple(hostile["done_at"]), (None, None))
        self.assertEqual(tuple(hostile["holds"]), (1, -1))

    def test_a_helper_with_weight_0_is_no_helper(self):
        """A cooperator whose weight is 0 has no social goal: the rival races it (9 / 11),
        as a competitor of "race" does, and does not touch its object."""
        result = self.pair(("competition", "cooperation"), (1.0, 0.0))
        self.assert_races(result, (9, 11))
        self.assertIn("competition vs none", result["social_terms"][0])
        self.assertIn("[rival: a race]", result["social_terms"][0])
        twin = self.pair(("competition", "cooperation"), (1.0, 0.0), levels=(2, 2),
                         competition="race")
        self.assertTrue(same_rollout(result, twin))
        self.assertTrue(same_ledgers(result, twin))

    def test_two_rivals_at_every_mix_of_levels(self):
        """Levels (1, 1), (1, 2) and (2, 1): 9 / 11 as well.  Against a rival the term is
        the same at both levels; the level says whom a rival takes its partner to be, and
        a racer one level down is what its partner is."""
        for levels in ((1, 1), (1, 2), (2, 1)):
            with self.subTest(levels=levels):
                result = self.pair(("competition", "competition"), 1.0, levels=levels)
                self.assertEqual(tuple(result["levels_used"]), levels)
                self.assert_races(result, (9, 11))

    def test_with_an_inferred_partner_goal(self):
        """The rival is not told whom it faces and holds a belief over five social goals;
        under one of them, cooperation, its term is -R_j, under the others the race.
        competition / cooperation at w = 1: it picks the cooperator's object up at
        transition 2 and keeps it, the cooperator delivers the rival's all the same (5),
        and the rival ends up sure that its partner cooperates: 5 / never.  (Two rivals
        that have to infer each other do not race, where two competitors of "race" do:
        README, "Limits".)"""
        result = self.pair(("competition", "cooperation"), 1.0,
                           partner_social_goal="inferred")
        self.assertEqual(tuple(result["levels_used"]), (2, 2))
        self.assertEqual(tuple(result["done_at"]), (5, None))
        self.assertEqual(sorted(events(result)), [(2, "picked up", 0, 1, None),
                                                  (2, "picked up", 1, 0, None),
                                                  (5, "set down", 1, 0, 0)])
        self.assertEqual(tuple(result["holds"]), (1, -1))
        led = result["ledger"][0]
        self.assertEqual((led["hand_actions"], led["handed_over"], led["set_down_for_partner"],
                          led["failed_set_downs"]), (1, 0, 0, 0))
        belief = np.asarray(result["final_social_belief"][0], dtype=float)
        self.assertGreater(belief[G.COOPERATION], 0.99)
        self.assertIn("competition: w (-1 R_j) against a cooperative partner; against any "
                      "other, w (+r(g_i) once its own goal is done", result["social_terms"][0])
        self.assertIn("[rival]", result["social_terms"][0])

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_by_whom_it_faces_it_is_one_of_two_other_readings(self):
        """Every social goal of a partner, either seat, w = 1 (14 pairings; with w = 30 as
        well the same holds, README): the rollout is the one of "table_hostile" where the
        partner is cooperative and the one of "race", at the same levels, everywhere else
        -- to the last Q value and the last entry of the ledgers -- and not the other
        one's.  A rival never sets its partner's object down for it or hands it over."""
        for seat in (0, 1):
            for partner in G.SOCIAL_GOALS:
                with self.subTest(seat=seat, partner=partner):
                    goals = ("competition", partner) if seat == 0 else (partner, "competition")
                    rival = self.pair(goals, 1.0)
                    levels = tuple(rival["levels"])
                    self.assertEqual(rival["levels_used"][seat], 2)
                    helped = partner == "cooperation"
                    twin, other = (("table_hostile", "race") if helped
                                   else ("race", "table_hostile"))
                    same = self.pair(goals, 1.0, levels=levels, competition=twin)
                    self.assertTrue(same_rollout(rival, same))
                    self.assertTrue(same_ledgers(rival, same))
                    self.assertFalse(same_rollout(rival, self.pair(
                        goals, 1.0, levels=levels, competition=other)))
                    led = rival["ledger"][seat]
                    self.assertEqual((led["handed_over"], led["set_down_for_partner"]),
                                     (0, 0))


class LevelTwoCompetitorTests(unittest.TestCase):
    """A competitor of a race form asked for level 2 (``levels=(2, 2)``).  Its term is the
    one of level 1; it predicts its partner by its partner's social goal.  So it expects
    a cooperator to deliver for it, a hinderer to keep what it took, and another competitor
    to turn on it once done -- and it acts on that before its own goal is done."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = Config(num_threads=2)

    def pair(self, goals, weight, levels=(2, 2), scene=SCENE, **options):
        cfg = Config(num_threads=2, **options)
        result = run_pairing(scene, goals, weights=(weight, weight), levels=levels, cfg=cfg,
                             n_steps=STEPS)
        self.assertTrue(result["numerically_accepted"])
        return result

    assert_races = CompetitionTests.assert_races

    def test_asked_for_level_2_a_competitor_reasons_at_level_2(self):
        for mode in RACES:
            with self.subTest(competition=mode):
                asked = run_pairing(SCENE, ("competition", "cooperation"), levels=(2, 2),
                                    cfg=Config(num_threads=2, competition=mode), n_steps=3)
                self.assertEqual((tuple(asked["levels"]), tuple(asked["levels_used"])),
                                 ((2, 2), (2, 2)))
                default = run_pairing(SCENE, ("competition", "cooperation"),
                                      cfg=Config(num_threads=2, competition=mode), n_steps=3)
                self.assertEqual((tuple(default["levels"]), tuple(default["levels_used"])),
                                 ((1, 2), (1, 2)))
                # the record describes its term in the same words at both levels (that
                # the term IS the same is test_models.py's comparison with brute force)
                self.assertEqual(asked["social_terms"][0], default["social_terms"][0])
                self.assertIn("race", asked["social_terms"][0])
                self.assertNotIn("R_j", asked["social_terms"][0])
                # a level-2 competitor has two models at level 2 (its root and its twin
                # without a social term) and its partner's level-1 models to predict it by
                self.assertEqual(asked["solver"]["models"][2], 4)
                self.assertEqual(default["solver"]["models"][2], 2)
                self.assertGreater(asked["solver"]["models"][1], default["solver"]["models"][1])

    def test_two_level_2_competitors_each_do_their_own_task(self):
        """On the standard map the outcome does not change: 9 / 11, each with its own
        object."""
        for mode, weights in (("race_then_hinder", WEIGHTS), ("race", (1.0, 30.0))):
            for weight in weights:
                with self.subTest(competition=mode, weight=weight):
                    result = self.pair(("competition", "competition"), weight,
                                       competition=mode)
                    self.assertEqual(tuple(result["levels_used"]), (2, 2))
                    self.assert_races(result, (9, 11))
        mixed = self.pair(("competition", "competition"), 1.0, levels=(1, 2))
        self.assertEqual(tuple(mixed["levels_used"]), (1, 2))
        self.assert_races(mixed, (9, 11))

    def test_what_a_level_2_competitor_is_paid(self):
        """The ledger of two level-2 competitors (``modeled_reward`` at level 2: the term
        read off the transitions that came about).  As at level 1: nothing while nobody
        has finished; R1 loses w rho in transitions 9 and 10 (R0 is done, it is not) and is
        paid it in 11 and 12, 0 in all; R0 is paid w rho in each of transitions 9 to 12
        under "race", and its lead on top under "race_then_hinder"."""
        weight = 2.0
        prize = weight * self.cfg.rho
        for mode in RACES:
            with self.subTest(competition=mode):
                result = self.pair(("competition", "competition"), weight, competition=mode)
                self.assertEqual(tuple(result["done_at"]), (9, 11))
                self.assertEqual(result["steps"], 12)
                paid = [led["social_reward"] for led in result["ledger"]]
                self.assertAlmostEqual(paid[1], 0.0, places=3)
                # transition by transition, what R1 expects of the action it takes: the
                # loss while only R0 is done, and the prize in the transition it finishes in
                expected = {row["step"]: float(row["social_reward"][1][row["action"][1]])
                            for row in result["trace"]}
                self.assertAlmostEqual(expected[10], -prize, places=4)
                self.assertAlmostEqual(expected[11], prize, places=4)
                if mode == "race":
                    self.assertAlmostEqual(paid[0], 4 * prize, places=4)
                else:
                    self.assertTrue(4 * prize < paid[0] < 6 * prize, paid[0])
                # ... and what each expects of the action it takes: nothing before a finish
                for row in result["trace"][:8]:
                    for i in (0, 1):
                        self.assertAlmostEqual(
                            float(row["social_reward"][i][row["action"][i]]), 0.0, places=5)

    def test_it_takes_a_cooperators_object_while_its_own_is_carried_for_it(self):
        """The default form against a cooperator.  The competitor does not set out for its
        own object: it expects the cooperator to bring it (set down at 5, at every weight,
        without a wrong destination first).  It steps onto the place of the cooperator's
        object and takes it -- at w <= 2 at once, at transition 2, three transitions before
        its own goal is done; at w = 30 it waits for that, by 0.002 to 0.07 in Q, and the
        test does not pin when.  It keeps the object: the cooperator never finishes.

        At w = 0.8, since 2026-10-16 (five hand actions; until then as at the other
        weights), the cooperator does not bring the competitor its object: it does not use
        its hands at all, and neither goal is done."""
        result = self.pair(("competition", "cooperation"), 0.8)
        self.assertEqual(tuple(result["done_at"]), (None, None))
        self.assertEqual(events(result), [(2, "picked up", 0, 1, None)])
        self.assertEqual(result["ledger"][1]["hand_actions"], 0)
        for weight in WEIGHTS[1:]:
            with self.subTest(weight=weight):
                result = self.pair(("competition", "cooperation"), weight)
                self.assertEqual(tuple(result["done_at"]), (5, None))
                happened = events(result)
                mine = [e for e in happened if e[2] == 0]
                self.assertEqual(len(mine), 1)
                taken = mine[0][0]
                self.assertEqual(mine[0], (taken, "picked up", 0, 1, None))
                self.assertEqual([e for e in happened if e[2] == 1],
                                 [(2, "picked up", 1, 0, None), (5, "set down", 1, 0, 0)])
                if weight <= 2.0:
                    self.assertEqual(taken, 2)                    # before its goal is done
                else:
                    self.assertTrue(2 <= taken <= 7, taken)
                self.assertEqual(tuple(result["holds"]), (1, -1))
                led = result["ledger"][0]
                self.assertEqual((led["moves"], led["hand_actions"], led["pickups"],
                                  led["handed_over"], led["set_down_for_partner"],
                                  led["failed_set_downs"]),
                                 (1, 1, 1, 0, 0, 0))
                self.assertEqual(led["delivered_by"], 1)

    def test_seated_first_the_race_form_and_a_cooperator(self):
        """``competition="race"``, competition / cooperation.  Until 2026-10-16 (with "use")
        it never used its hands, at level 2 as at level 1: 5 / 17 at every weight, the
        cooperator bringing it its object.  With the five hand actions that holds at w = 30
        only.  At w <= 2 it picks the cooperator's object up at transition 2 and keeps it,
        and the cooperator does not bring it its own (at w = 2 it picks it up at 2 and puts
        it back at 3): a deadlock, neither goal done.

        Why it does not wait: in its plan the cooperator, which does not know its
        destination, keeps setting its object down on a wrong candidate (a model's belief
        does not change within a plan, so every try counts with the same mass), and nothing
        it can do gets the cooperator out of that.  With "use" stepping next to the
        cooperator did: "use" was then a hand-over, and the cooperator moved on."""
        result = self.pair(("competition", "cooperation"), 30.0, competition="race")
        self.assertEqual(tuple(result["done_at"]), (5, 17))
        self.assertEqual(result["ledger"][0]["hand_actions"], 0)
        self.assertEqual(events(result),
                         [(2, "picked up", 1, 0, None), (5, "set down", 1, 0, 0),
                          (15, "picked up", 1, 1, None), (17, "set down", 1, 1, 1)])
        for weight in WEIGHTS[:3]:
            with self.subTest(weight=weight):
                result = self.pair(("competition", "cooperation"), weight, competition="race")
                self.assertEqual(tuple(result["done_at"]), (None, None))
                self.assertEqual(result["stopped_by"], "deadlock")
                self.assertIn((2, "picked up", 0, 1, None), events(result))
                self.assertEqual(tuple(result["holds"])[0], 1)              # ... to the end
                self.assertEqual(result["ledger"][0]["hand_actions"], 1)
                self.assertEqual(events(result)[-1:] == [(3, "put back", 1, 0, None)],
                                 weight == 2.0)

    def test_seated_second_it_takes_a_cooperators_object_at_once(self):
        """cooperation / competition at level 2, w <= 2: the competitor picks the
        cooperator's object up at transition 2 and holds it to the end, under BOTH race
        forms (a level-2 robot without social goal does the same in that seat: it is taken
        so as to be served).  The default form: the cooperator still delivers at 5 at w = 1
        and 2, and not at w = 0.8.  "race": the cooperator never delivers, and neither goal
        is done.  At w = 30 the competitor does not move; under "race" both finish, 17 / 5.

        The take is by 1.1 to 5.6 in Q.  What the cooperator then does at w = 0.8 rests on
        0.12 in its Q.  Not pinned: the default form at w = 30, which stays where it is by
        0.01 in Q (in the run: never / 5)."""
        goals = ("cooperation", "competition")
        take = (2, "picked up", 1, 0, None)
        for mode, done_by_weight in (
                ("race_then_hinder", {0.8: (None, None), 1.0: (None, 5), 2.0: (None, 5)}),
                ("race", {0.8: (None, None), 1.0: (None, None), 2.0: (None, None)})):
            for weight, done_at in done_by_weight.items():
                with self.subTest(competition=mode, weight=weight):
                    result = self.pair(goals, weight, competition=mode)
                    self.assertEqual(tuple(result["done_at"]), done_at)
                    mine = [e for e in events(result) if e[2] == 1]
                    self.assertEqual(mine, [take])
                    self.assertEqual(result["holds"][1], 0)             # ... to the end
                    led = result["ledger"][1]
                    self.assertEqual((led["hand_actions"], led["handed_over"],
                                      led["set_down_for_partner"]), (1, 0, 0))
        race = self.pair(goals, 30.0, competition="race")
        self.assertEqual(tuple(race["done_at"]), (17, 5))
        self.assertEqual((race["ledger"][1]["moves"], race["ledger"][1]["hand_actions"]), (0, 0))

    def test_it_takes_a_hinderers_object_in_turn_and_offers_a_swap(self):
        """Its object is in a hinderer's hands.  At level 1 it waits.  At level 2 it expects
        the hinderer not to let go: it picks the hinderer's object up, walks to the
        hinderer and, next to it, offers a swap in every transition (from 10 on; "give"
        next to the hinderer), which the hinderer never accepts.  Both race forms: all of
        this is before its own goal is done, where the two terms pay the same -- and here
        the plans agree as well.  In which transition it takes the object (4; at w = 30 3
        until 2026-10-16, and since then 2, together with the hinderer) rests on small
        differences in Q and is pinned as a range."""
        for mode in RACES:
            for weight in (1.0, 30.0):
                with self.subTest(competition=mode, weight=weight):
                    result = self.pair(("hinder", "competition"), weight, competition=mode)
                    self.assertEqual(tuple(result["levels_used"]), (1, 2))
                    self.assertEqual(tuple(result["done_at"]), (None, None))
                    happened = events(result)
                    self.assertEqual(len(happened), 2)
                    self.assertIn((2, "picked up", 0, 1, None), happened)
                    mine = [e for e in happened if e[2] == 1]
                    self.assertEqual(len(mine), 1)
                    taken = mine[0][0]
                    self.assertEqual(mine[0], (taken, "picked up", 1, 0, None))
                    self.assertTrue(2 <= taken <= 5, taken)
                    self.assertEqual(tuple(result["holds"]), (1, 0))
                    led = result["ledger"][1]
                    self.assertGreaterEqual(led["hand_actions"], 10)
                    self.assertEqual((led["handed_over"], led["received"],
                                      led["set_down_for_partner"]), (0, 0, 0))
                    self.assertEqual(result["ledger"][0]["hand_actions"], 1)     # the hinderer's
                    self.assertTrue(all(row["action"][1] == GIVE
                                        for row in result["trace"][-8:]))

    def test_the_robot_behind_takes_the_object_of_the_one_ahead(self):
        """Two competitors, R1's destination in the far room (alone: 9 / 17).  At level 1
        both race, and R0, done at 9, stands in R1's way for good (9 / never).  At level 2
        R1 expects that: it picks R0's object up at transition 2 (by 66 in Q at w = 30, 3 at
        w = 1) and, in the three runs here, keeps it.  R0 is then never done and never
        turns on it -- and R1, its hands full, does not finish either.  It is R1's level
        that decides: levels (1, 2) give the same.  "race": R0 will not turn on it, and it
        races (9 / 17).

        Not pinned: levels (2, 2) at w = 1, where R1 puts the object back at transition 8
        (by 0.12 in Q) and both finish, 12 / 28; and what R0 does while its object is held
        (it waits, by 0.002 to 0.007 in Q)."""
        far = far_room()
        for weight, levels in ((30.0, (2, 2)), (30.0, (1, 2)), (1.0, (1, 2))):
            with self.subTest(weight=weight, levels=levels):
                result = self.pair(("competition", "competition"), weight, levels=levels,
                                   scene=far)
                self.assertEqual(tuple(result["done_at"]), (None, None))
                self.assertEqual([e for e in events(result) if e[2] == 1],
                                 [(2, "picked up", 1, 0, None)])
                self.assertEqual(result["holds"][1], 0)
        ahead = self.pair(("competition", "competition"), 30.0, levels=(2, 1), scene=far)
        self.assertEqual(tuple(ahead["done_at"]), (9, None))       # as at levels (1, 1)
        self.assertEqual(events(ahead), [(7, "picked up", 0, 0, None), (9, "set down", 0, 0, 0),
                                         (9, "picked up", 1, 1, None)])
        race = self.pair(("competition", "competition"), 30.0, scene=far, competition="race")
        self.assert_races(race, (9, 17))

    def test_two_level_2_competitors_stand_off_past_a_doorway_one_cell_wide(self):
        """R0 is through the doorway first and meets R1 just past it; one of the two has to
        give way.  At level 1 R1 waits one transition (9 / 12).  At levels (2, 2) each
        plans against a level-1 model of the other, which would give way, and neither does:
        both push into each other for the rest of the rollout.  (It takes a competitor:
        two level-2 robots without social goals pass there, 9 / 12.)  With only R1 at
        level 2 it is R0 that gives way -- it steps aside and is done at 11 instead of 9 --
        and with only R0 at level 2 the course is that of levels (1, 1) until R0 is done
        (what R0 does afterwards is a tie in its Q at transition 10, and not compared)."""
        door = one_cell_door()
        goals = ("competition", "competition")
        both = self.pair(goals, 1.0, scene=door)
        self.assertEqual(tuple(both["done_at"]), (None, None))
        self.assertEqual(both["stopped_by"], "deadlock")
        self.assertEqual(events(both), [])
        self.assertTrue(all(led["blocked_by_partner"] >= 20 for led in both["ledger"]))
        self.assertEqual(tuple(both["trace"][-1]["next_pos"][0]), (5, 2))   # past the door
        runs = {}
        for levels, done_at in (((1, 1), (9, 12)), ((2, 1), (9, 12)), ((1, 2), (11, 12))):
            with self.subTest(levels=levels):
                runs[levels] = result = self.pair(goals, 1.0, levels=levels, scene=door)
                self.assertEqual(tuple(result["done_at"]), done_at)
                self.assertEqual(sorted((robot, obj) for _, what, robot, obj, _ in
                                        result["events"] if what == "picked up"),
                                 [(0, 0), (1, 1)])
                # R0 is in the doorway cell after transition 3, R1 after 6: R0 goes first
                self.assertEqual(tuple(result["trace"][2]["next_pos"][0]), (4, 2))
                self.assertEqual(tuple(result["trace"][5]["next_pos"][1]), (4, 2))
        ahead, both_1 = runs[(2, 1)]["trace"], runs[(1, 1)]["trace"]
        self.assertEqual(len(ahead), len(both_1))
        self.assertEqual([row["action"][0] for row in ahead[:9]],          # R0, to its finish
                         [row["action"][0] for row in both_1[:9]])
        self.assertEqual([row["action"][1] for row in ahead],              # R1, throughout
                         [row["action"][1] for row in both_1])
        # ... and with R1 at level 2 R0 leaves the row R1 comes along (y = 2), which it
        # does not at levels (1, 1)
        rows = {levels: {tuple(row["next_pos"][0])[1] for row in runs[levels]["trace"][3:6]}
                for levels in ((1, 1), (1, 2))}
        self.assertEqual(rows[(1, 1)], {2})
        self.assertNotEqual(rows[(1, 2)], {2})

    @unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
    def test_a_level_2_competitor_delivers_for_no_partner(self):
        """Against each of the seven social goals of a partner, in either seat, at w = 1 and
        30: a level-2 competitor of the default form never sets its partner's object down
        on its partner's destination, and never hands it over."""
        for seat in (0, 1):
            for partner in G.SOCIAL_GOALS:
                for weight in (1.0, 30.0):
                    with self.subTest(seat=seat, partner=partner, weight=weight):
                        goals = (("competition", partner) if seat == 0
                                 else (partner, "competition"))
                        result = self.pair(goals, weight)
                        self.assertEqual(result["levels_used"][seat], 2)
                        led = result["ledger"][seat]
                        self.assertEqual((led["handed_over"], led["set_down_for_partner"]),
                                         (0, 0))
                        self.assertNotEqual(result["ledger"][1 - seat]["delivered_by"], seat)


if __name__ == "__main__":
    unittest.main()
