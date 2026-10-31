"""The hindrance metrics on hand-made records (no solver involved)."""
import dataclasses
import unittest

from social_pairings.hindrance import conduct, format_report, hindrance_report, physical_return

from .told import Config

CONFIG = dataclasses.asdict(Config())
STAY, PICK = 4, 5                            # stay; a hand action: pick up


def record(actions, *, events=(), done_at=(None, None), holds=None, rewards=None, blocked=None):
    """A minimal rollout record: actions[t] = (a0, a1); events = (transition, what, robot,
    object, other); holds[t] = objects in hand after transition t + 1."""
    steps = len(actions)
    holds = holds or [(-1, -1)] * steps
    rewards = rewards or [(0.0, 0.0)] * steps
    blocked = blocked or [(False, False)] * steps
    trace = [dict(step=t + 1, action=actions[t], physical_reward=rewards[t],
                  next_holds=holds[t], blocked=blocked[t],
                  events=tuple(e[1:] for e in events if e[0] == t + 1)) for t in range(steps)]
    return dict(steps=steps, trace=trace, events=list(events), done_at=done_at,
                holds=holds[-1], config=CONFIG, stopped_by="cap")


class ConductTests(unittest.TestCase):
    def test_a_robot_that_takes_an_object_and_keeps_it_is_clean(self):
        run = record([(PICK, STAY)] + [(STAY, STAY)] * 5, events=[(1, "picked up", 0, 1, None)],
                     holds=[(1, -1)] * 6)
        c = conduct(run, 0)
        self.assertEqual((c["put_back_cycles"], c["reversals"], c["spare_pickups"],
                          c["wasted_hand_actions"]), (0, 0, 0, 0))
        self.assertEqual((c["hand_actions"], c["waits"], c["moves"]), (1, 5, 0))

    def test_picking_up_and_putting_back_is_a_cycle(self):
        events = [(1, "picked up", 0, 1, None), (3, "put back", 0, 1, None),
                  (5, "picked up", 0, 1, None)]
        run = record([(PICK, STAY), (STAY, STAY), (PICK, STAY), (STAY, STAY), (PICK, STAY)],
                     events=events)
        self.assertEqual(conduct(run, 0)["put_back_cycles"], 2)
        self.assertEqual(conduct(run, 1)["put_back_cycles"], 0)

    def test_a_put_back_long_after_the_pick_up_is_not_a_cycle(self):
        events = [(1, "picked up", 0, 1, None), (9, "put back", 0, 1, None)]
        run = record([(PICK, STAY)] + [(STAY, STAY)] * 7 + [(PICK, STAY)], events=events)
        self.assertEqual(conduct(run, 0)["put_back_cycles"], 0)

    def test_pacing_spare_pickups_and_wasted_hand_actions(self):
        run = record([(0, STAY), (2, STAY), (0, STAY), (PICK, STAY), (PICK, PICK)],
                     events=[(4, "picked up", 0, 3, None)])
        c = conduct(run, 0)
        self.assertEqual(c["reversals"], 2)                 # +y, -y, +y
        self.assertEqual(c["spare_pickups"], 1)             # object 3 completes nobody's goal
        self.assertEqual(c["wasted_hand_actions"], 1)               # the second pick did nothing
        self.assertEqual(conduct(run, 1)["wasted_hand_actions"], 1)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.baseline = record([(STAY, STAY)] * 8, done_at=(5, 6),
                               rewards=[(0.4, 0.4)] * 8)

    def test_hostage_taken_and_kept(self):
        run = record([(PICK, STAY)] + [(STAY, STAY)] * 9, done_at=(None, None),
                     events=[(1, "picked up", 0, 1, None)], holds=[(1, -1)] * 10)
        report = hindrance_report(run, self.baseline, hinderer=0)
        self.assertTrue(report["prevented"] and report["hindered"] and report["purposeful"])
        self.assertEqual(report["mechanism"], "holds the partner's object")
        self.assertEqual((report["took_object_at"], report["steps_holding"], report["releases"]),
                         (1, 10, 0))
        self.assertIsNone(report["delay"])
        self.assertEqual(report["return_denied"], 1.0)
        self.assertIn("HINDERED / purposeful", format_report(report))

    def test_partner_finishes_on_time_is_not_hindered(self):
        run = record([(STAY, STAY)] * 8, done_at=(5, 6), rewards=[(0.4, 0.4)] * 8)
        report = hindrance_report(run, self.baseline, hinderer=0)
        self.assertFalse(report["hindered"] or report["purposeful"] or report["prevented"])
        self.assertEqual(report["delay"], 0)
        self.assertAlmostEqual(report["return_denied"], 0.0)
        self.assertIn("not hindered", format_report(report))

    def test_a_delay_counts_from_min_delay_on(self):
        run = record([(STAY, STAY)] * 12, done_at=(5, 10), rewards=[(0.4, 0.2)] * 12)
        self.assertFalse(hindrance_report(run, self.baseline, hinderer=0)["hindered"])
        self.assertTrue(hindrance_report(run, self.baseline, hinderer=0, min_delay=4)["hindered"])

    def test_effect_without_purpose_is_told_apart(self):
        """The partner never finishes, but the "hinderer" gave the object back twice."""
        events = [(1, "picked up", 0, 1, None), (2, "put back", 0, 1, None),
                  (3, "picked up", 0, 1, None), (4, "handed over", 0, 1, 1)]
        holds = [(1, -1), (-1, -1), (1, -1), (-1, 1)] + [(-1, 1)] * 4
        run = record([(PICK, STAY)] * 4 + [(STAY, STAY)] * 4, events=events, holds=holds)
        report = hindrance_report(run, self.baseline, hinderer=0)
        self.assertTrue(report["hindered"])
        self.assertFalse(report["purposeful"])
        self.assertEqual(report["releases"], 2)
        self.assertIn("NOT purposeful", format_report(report))

    def test_blocking_is_a_mechanism_too(self):
        run = record([(STAY, 1)] * 10, blocked=[(False, True)] * 10)
        report = hindrance_report(run, self.baseline, hinderer=0)
        self.assertEqual(report["mechanism"], "blocks the partner")
        self.assertEqual(report["partner_blocked"], 10)
        self.assertTrue(report["purposeful"])

    def test_robot_names_in_the_report_line(self):
        run = record([(PICK, STAY)] + [(STAY, STAY)] * 9, events=[(1, "picked up", 0, 1, None)],
                     holds=[(1, -1)] * 10)
        line = format_report(hindrance_report(run, self.baseline, hinderer=0),
                             names=("agent_1", "agent_2"))
        self.assertIn("agent_1 takes", line)
        self.assertNotIn("R0", line)


class ReturnTests(unittest.TestCase):
    def test_discounted_sum_and_the_tail_after_the_rollout(self):
        cfg = Config()
        gamma = cfg.gamma
        done = record([(STAY, STAY)] * 2, done_at=(2, None), rewards=[(0.5, 0.1), (0.5, 0.1)])
        steady = (cfg.reward_rate - cfg.wait_cost_rate) * cfg.dt        # parked on its goal
        expected = 0.5 + gamma * 0.5 + gamma ** 2 * steady + gamma ** 3 * steady
        self.assertAlmostEqual(physical_return(done, 0, 4), expected)
        # a robot that is not done is taken to stay frozen in its last transition
        frozen = 0.1 * (1 + gamma + gamma ** 2 + gamma ** 3)
        self.assertAlmostEqual(physical_return(done, 1, 4), frozen)


if __name__ == "__main__":
    unittest.main()
