"""The figure: what it says about actions, hands, events and goals, and the files it writes."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from social_pairings import Scene, exchange_scene, two_room_scene
from social_pairings import goals as G
from social_pairings.constants import GIVE, HELD0, HELD1, HOME, PICK, PLACE, WAIT
from social_pairings.render import (_ROOM, FONT, ROBOT_COLOURS, _callout, _Canvas, _Story,
                                    _title, event_text, frames_of, goal_label, render_gif,
                                    render_video, save_frame)
from social_pairings.serialize import load_result, save_result

from .test_models import TINY
from .told import Config, run_baseline, run_pairing

NAMES = ("agent_1", "agent_2")
CFG = Config(num_threads=2)
WORDS = ("move up", "move right", "move down", "move left", "stay", "pick up", "place", "take",
         "give", "swap")
VERBS = {"picked up": "picks up", "put back": "puts", "put down": "puts",
         "handed over": "to agent_", "set down": "places", "set down in vain": "tries to place",
         "handed over in vain": "tries to hand"}
TAGS = {"picked up": "▲", "put back": "▼", "put down": "▼", "handed over": "◆",
        "set down in vain": "✗", "handed over in vain": "✗"}


def disagreements(result: dict) -> list:
    """Everything the figure says about a rollout that the record does not bear out: the
    record is read here a second time, without the code of the figure."""
    story = _Story(result, NAMES)
    trace, found = result["trace"], []
    n = len(trace)
    last_touched = [None, None]

    def state(f):
        if f < n:
            row = trace[f]
            return row["pos"], row["status"], row["holds"]
        return trace[-1]["next_pos"], trace[-1]["next_status"], trace[-1]["next_holds"]

    for f in range(n + 1):
        pos, status, holds = state(f)
        arrived = [tuple(e[:4]) for e in trace[f - 1]["events"]] if f else []
        for what, by, obj, other in arrived:
            last_touched[by] = f
            if what == "handed over":
                last_touched[other] = f
        for i in (0, 1):
            partner = NAMES[1 - i]
            # ---- hands
            want = "empty" if holds[i] < 0 else f"holds O{holds[i] + 1}"
            if not story.hands(f, i).startswith(want):
                found.append((f, i, "hands", story.hands(f, i), want))
            # ---- goal
            at = result["done_at"][i]
            done = at is not None and at <= f
            flag, badge, _, detail = story.goal(f, i)
            if flag != done or ("NOT" in badge) == done:
                found.append((f, i, "goal", badge, done))
            if done and result["ledger"][i]["delivered_by"] is not None:
                led = result["ledger"][i]
                who = "itself" if led["delivered_by"] == i else NAMES[led["delivered_by"]]
                want = f"O{led['object'] + 1} placed by {who} at step {at}"
            elif done:
                want = f"done at step {at}"
            elif status[i] >= 4:
                want = f"O{i + 1} is on the hand-off cell"
            else:
                want = f"O{i + 1} is " + {HOME: "at its place", (HELD0, HELD1)[i]: "in its hands",
                                          (HELD1, HELD0)[i]: f"in {partner}'s hands"}[status[i]]
            if detail != want:
                found.append((f, i, "goal detail", detail, want))
            # ---- last event
            said = story.last_event(f, i)
            want = "—" if last_touched[i] is None else f"step {last_touched[i]}: "
            if not said.startswith(want) or (said == "—") != (want == "—"):
                found.append((f, i, "last event", said, last_touched[i]))
            # ---- action
            said = story.action(f, i)
            if f == n:
                if "rollout ended" not in said:
                    found.append((f, i, "action after the end", said))
                continue
            row = trace[f]
            a = row["action"][i]
            stood = tuple(row["next_pos"][i]) == tuple(row["pos"][i])
            mine = [tuple(e[:4]) for e in row["events"] if e[1] == i]
            if not said.startswith(WORDS[a]):
                found.append((f, i, "action", said, a))
            if ("blocked by" in said) != (a < WAIT and stood):
                found.append((f, i, "blocked", said))
            if (said == f"{WORDS[a]}: nothing happens") != (a >= PICK and not mine):
                found.append((f, i, "a hand action without effect", said, mine))
            for what, _, obj, _ in mine:
                if VERBS[what] not in said or f"O{obj + 1}" not in said:
                    found.append((f, i, "action leaves out an event", said, what, obj))
            if ("has to give" in said) != any(what == "handed over" and a != GIVE
                                              for what, _, _, _ in mine):
                found.append((f, i, "a hand-over by force", said, a))
        # ---- headline, callouts, frame timing
        banner = story.banner(f)
        if bool(banner) != bool(arrived) or any(
                f"{NAMES[by]} " not in banner or f"O{obj + 1}" not in banner
                for _, by, obj, _ in arrived):
            found.append((f, "headline", banner, arrived))
        fresh = [(cell, events) for age, cell, events in story.callouts(f, 3) if age == 0]
        if sorted(e for _, events in fresh for e in events) != sorted(arrived):
            found.append((f, "callouts", fresh, arrived))
        for cell, events in fresh:               # at the cell its robot arrived on
            for _, by, _, _ in events:
                if tuple(cell) != tuple(pos[by]):
                    found.append((f, "callout at the wrong cell", cell, pos[by]))
        want = "end" if f == n else ("placed" if any(e[0] == "set down" for e in arrived)
                                     else ("event" if arrived else "plain"))
        if story.weight(f) != want:
            found.append((f, "hold", story.weight(f), want))
    # ---- timeline
    for i in (0, 1):
        if [a for a, _, _ in story.tape(i, n)] != [row["action"][i] for row in trace]:
            found.append((i, "timeline actions"))
        carried = sorted((t, held) for first, last, held in story.carried(i, n)
                         for t in range(first, last + 1))
        if carried != [(t, row["holds"][i]) for t, row in enumerate(trace, 1)
                       if row["holds"][i] >= 0]:
            found.append((i, "timeline: what was carried"))
    marks = story.marks(n)
    count = 0
    for t, what, by, obj, other in (tuple(e[:5]) for e in result["events"]):
        if what == "set down":                   # a star for the owner, a put-down for a helper
            tags = [(other, f"★O{obj + 1}")] + ([(by, f"▼O{obj + 1}")] if by != other else [])
        else:
            tags = [(by, f"{TAGS[what]}O{obj + 1}")]
        count += len(tags)
        for robot, tag in tags:
            if tag not in marks.get((t, robot), []):
                found.append((t, "timeline tag missing", tag, robot))
    if count != sum(len(tags) for tags in marks.values()):
        found.append(("timeline tags", count, marks))
    return found


def label_boxes(canvas: _Canvas) -> list:
    """(text, box) of the labels on the map (not the callouts), in pixels, after a draw."""
    canvas.fig.canvas.draw()
    renderer = canvas.fig.canvas.get_renderer()
    return [(t.get_text(), t.get_window_extent(renderer)) for t in canvas.ax_map.texts
            if t.get_text() and t.get_fontsize() == 7.5]


class EventTextTests(unittest.TestCase):
    def test_every_event_in_both_tenses(self):
        cases = {
            ("picked up", 0, 1, None): ("agent_1 picked up O2", "picks up O2"),
            ("put back", 1, 0, None): ("agent_2 put O1 back", "puts O1 back"),
            ("handed over", 1, 0, 0): ("agent_2 handed O1 to agent_1", "hands O1 to agent_1"),
            ("set down", 0, 0, 0): ("agent_1 placed O1 on its goal", "places O1 on its goal"),
            ("set down", 0, 1, 1): ("agent_1 placed O2 on agent_2's goal",
                                    "places O2 on agent_2's goal"),
            ("set down in vain", 1, 0, 0): (
                "agent_2 tried to place O1 (not agent_1's destination)",
                "tries to place O1 (not agent_1's destination)"),
        }
        for event, (past, present) in cases.items():
            self.assertEqual(event_text(event, NAMES), past)
            self.assertEqual(event_text(event, NAMES, "present", subject=False), present)

    def test_a_hand_over_its_giver_did_not_choose(self):
        """``forced_swap=True``: handed its own object, a robot gives up what it holds
        without using its hands.  The figure does not say it "handed" the object over."""
        event = ("handed over", 0, 1, 1)
        self.assertEqual(event_text(event, NAMES, forced=True),
                         "agent_1 had to give O2 to agent_2")
        self.assertEqual(event_text(event, NAMES, "present", subject=False, forced=True),
                         "has to give O2 to agent_2")
        self.assertEqual(_callout([event], NAMES, [True]),
                         "GIVEN UP O2\nagent_1 → agent_2 (forced swap)")
        self.assertEqual(_callout([event], NAMES), "HANDED OVER O2\nagent_1 → agent_2")
        # nothing else can be forced
        self.assertEqual(event_text(("picked up", 0, 1, None), NAMES, forced=True),
                         "agent_1 picked up O2")


class StoryTests(unittest.TestCase):
    """Two cooperators at weight 30: each picks up the other's object in transition 2 and
    sets it down on the other's destination in transition 5."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(two_room_scene(), ("cooperation", "cooperation"),
                                 weights=(30.0, 30.0), cfg=CFG, n_steps=20)
        cls.story = _Story(cls.result, NAMES)

    def test_the_run_is_the_one_described(self):
        self.assertEqual(tuple(self.result["done_at"]), (5, 5))
        self.assertEqual([(e[0], e[1], e[2], e[3]) for e in self.result["events"]],
                         [(2, "picked up", 1, 0), (2, "picked up", 0, 1),
                          (5, "set down", 1, 0), (5, "set down", 0, 1)])
        self.assertEqual(len(self.story.frames), self.result["steps"] + 1)

    def test_action_of_every_step(self):
        story = self.story
        # frame 1: the state after one transition; the actions chosen there pick the objects up
        self.assertEqual(story.action(1, 0), "pick up: picks up O2")
        self.assertEqual(story.action(1, 1), "pick up: picks up O1")
        self.assertTrue(story.action(0, 0).startswith("move "))
        self.assertEqual(story.action(4, 0), "place: places O2 on agent_2's goal")
        self.assertEqual(story.action(len(story.frames) - 1, 0), "— (rollout ended)")

    def test_hands_and_last_event(self):
        story = self.story
        self.assertEqual(story.hands(1, 0), "empty")
        self.assertEqual(story.hands(2, 0), "holds O2 (agent_2's)")
        self.assertEqual(story.hands(2, 1), "holds O1 (agent_1's)")
        self.assertEqual(story.hands(5, 0), "empty")
        self.assertEqual(story.last_event(1, 0), "—")
        self.assertEqual(story.last_event(2, 0), "step 2: picked up O2")
        self.assertEqual(story.last_event(5, 1), "step 5: placed O1 on agent_1's goal")

    def test_state_of_each_goal(self):
        story = self.story
        for i, partner in ((0, "agent_2"), (1, "agent_1")):
            done, badge, task, detail = story.goal(0, i)
            self.assertEqual((done, badge), (False, "NOT DONE"))
            self.assertEqual(detail, f"O{i + 1} is at its place")
            self.assertEqual(task, f"bring O{i + 1} to G{self.result['scene']['goal_ids'][i]}")
            self.assertEqual(story.goal(3, i)[3], f"O{i + 1} is in {partner}'s hands")
            self.assertFalse(story.goal(4, i)[0])
            done, badge, _, detail = story.goal(5, i)
            self.assertTrue(done)
            self.assertIn("DONE", badge)
            self.assertEqual(detail, f"O{i + 1} placed by {partner} at step 5")

    def test_events_are_announced_when_they_happen(self):
        story = self.story
        self.assertEqual(story.banner(1), "")
        self.assertEqual(story.banner(2), "agent_1 picked up O2     agent_2 picked up O1")
        self.assertIn("agent_1 placed O2 on agent_2's goal", story.banner(5))
        self.assertEqual([story.weight(f) for f in range(len(story.frames))],
                         ["plain", "plain", "event", "plain", "plain", "placed", "end"])
        # the callouts of transition 2 stay for three frames, at the cells of the pick-ups
        self.assertEqual(len(story.callouts(2, 3)), 2)
        self.assertEqual([age for age, _, _ in story.callouts(4, 3)], [2, 2])
        self.assertEqual(story.callouts(1, 3), [])
        cells = {cell for _, cell, _ in story.callouts(2, 3)}
        self.assertEqual(cells, {tuple(c) for c in self.result["scene"]["objects"][:2]})

    def test_the_timeline(self):
        """Per robot: the action of every transition, what it carried, and the event tags."""
        story = self.story
        tape = story.tape(0, 6)
        self.assertEqual([a for a, _, _ in tape], [row["action"][0] for row in
                                                   self.result["trace"]])
        self.assertEqual((tape[1][0], tape[4][0]), (PICK, PLACE))
        self.assertFalse(any(blocked for _, blocked, _ in tape))
        # picked up in transition 2, set down in transition 5: carried during 3, 4 and 5
        self.assertEqual(story.carried(0, 6), [(3, 5, 1)])
        self.assertEqual(story.carried(1, 6), [(3, 5, 0)])
        self.assertEqual(story.carried(0, 4), [(3, 4, 1)])
        self.assertEqual(story.carried(0, 2), [])
        self.assertEqual(story.marks(1), {})
        self.assertEqual(story.marks(4), {(2, 0): ["▲O2"], (2, 1): ["▲O1"]})
        # a goal finished by the partner: a star in the owner's row, a put-down in the
        # partner's
        self.assertEqual(story.marks(5)[(5, 0)], ["★O1", "▼O2"])
        self.assertEqual(story.marks(5)[(5, 1)], ["▼O1", "★O2"])

    def test_an_act_for_the_partner_has_no_physical_part(self):
        """Setting an object down for the partner is not an action of the robot's physical
        model: the value panel shows the whole difference instead of a blank."""
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            frame = canvas.frames[4]
            self.assertEqual(frame["action"][0], PLACE)
            physical, social, whole = canvas.bars(frame, 0)
            self.assertIsNone(physical)
            self.assertIsNone(social)
            self.assertEqual(whole[PLACE], 0.0)
            self.assertTrue(np.isfinite(whole).sum() >= 2)
            first = canvas.frames[0]                               # a move: both parts
            physical, social, whole = canvas.bars(first, 0)
            self.assertIsNone(whole)
            taken = first["action"][0]
            self.assertEqual((physical[taken], social[taken]), (0.0, 0.0))
            # an act for the partner that it did not take (since 2026-10-16): its whole
            # value relative to the action taken, next to the two parts of the others
            q, qp = first["q"][0].copy(), first["q_physical"][0].copy()
            q[GIVE], qp[GIVE] = q[taken] - 1.5, np.nan
            frame = dict(first, q=[q, first["q"][1]], q_physical=[qp, first["q_physical"][1]])
            physical, social, whole = canvas.bars(frame, 0)
            self.assertEqual((physical[taken], social[taken]), (0.0, 0.0))
            self.assertAlmostEqual(whole[GIVE], -1.5)
            self.assertEqual(int(np.isfinite(whole).sum()), 1)
            self.assertTrue(np.isnan(physical[GIVE]) and np.isnan(social[GIVE]))
        finally:
            canvas.close()

    def test_a_record_read_from_json_tells_the_same_story(self):
        with tempfile.TemporaryDirectory() as folder:
            loaded = load_result(save_result(Path(folder) / "run.json", self.result))
        other = _Story(loaded, NAMES)
        for f in range(len(self.story.frames)):
            for i in (0, 1):
                for tell in ("action", "hands", "last_event", "goal"):
                    self.assertEqual(getattr(other, tell)(f, i), getattr(self.story, tell)(f, i))
            self.assertEqual(other.banner(f), self.story.banner(f))
            self.assertEqual(other.callouts(f, 3), self.story.callouts(f, 3))
            self.assertEqual(other.marks(f), self.story.marks(f))


class EveryKindOfEventTests(unittest.TestCase):
    """Four rollouts that between them hold most kinds of event and every kind of ending: a
    set-down for the partner, a hand-over, a put-back, a goal finished by the partner, a
    deadlock and the step limit.  For each, everything the figure says is compared with the
    record, frame by frame.  (Until the belief update of 2026-10-30 the same four runs also
    held a set-down on a wrong candidate and a hand-over by force; under Eqs. 5-6 and
    nothing else the helping cooperators, unsure of each other's destination, no longer set
    down on a guess, and the hinderer of the forced run waits on its partner's object
    instead of taking it, so nothing is forced out of its hands.  A set-down in vain is in
    ``test_objects.TheRecord`` and in ``test_beliefs``.)"""

    @classmethod
    def setUpClass(cls):
        scene = two_room_scene()

        def run(goals, weight, steps, **options):
            return run_pairing(scene, goals, weights=(weight, weight), n_steps=steps,
                               cfg=Config(num_threads=2, **options))

        cls.runs = dict(
            helping=run(("cooperation", "cooperation"), 1.0, 40),     # 30 until 2026-10-16
            hostage=run(("hinder", "competition"), 1.0, 40),
            forced=run(("hinder", "competition"), 30.0, 30, forced_swap=True,
                       competition="table"),
            put_back=run(("hinder", None), 30.0, 12, held_reward="carrier"))

    def test_the_runs_hold_every_kind_of_event(self):
        kinds = {name: sorted({e[1] for e in r["events"]}) for name, r in self.runs.items()}
        self.assertEqual(kinds["helping"], ["picked up", "put back", "set down"])
        self.assertEqual(kinds["hostage"], ["picked up"])
        self.assertEqual(kinds["forced"], ["handed over", "picked up", "set down"])
        self.assertEqual(kinds["put_back"], ["picked up"])
        self.assertEqual({name: r["stopped_by"] for name, r in self.runs.items()},
                         dict(helping="done", hostage="deadlock", forced="done",
                              put_back="cap"))

    def test_the_figure_says_what_the_record_says(self):
        for name, result in self.runs.items():
            with self.subTest(run=name):
                self.assertEqual(disagreements(result), [])

    def test_the_check_itself_notices_a_wrong_figure(self):
        """The same comparison on a record whose events were tampered with."""
        result = dict(self.runs["helping"])
        t, what, by, obj, other = result["events"][0]
        result["events"] = [(t, what, by, 1 - obj, other)] + list(result["events"][1:])
        self.assertTrue(disagreements(result))

    def test_helping(self):
        """agent_2 picks O1 up (2), puts it back (3), picks it up again (4) and sets it down
        on agent_1's destination (7); agent_1 picks O2 up (4) and, unsure where agent_2
        wants it, puts it back (5); agent_2 fetches its own (15) and delivers it (23).
        (Until 2026-10-30: agent_1 set O2 down on agent_2's destination at 5, agent_2 tried
        three wrong candidates with O1, handed it to agent_1 at 24, who delivered it at 31.)"""
        result = self.runs["helping"]
        story = _Story(result, NAMES)
        self.assertEqual(tuple(result["done_at"]), (7, 23))
        self.assertEqual(story.action(1, 1), "pick up: picks up O1")
        self.assertEqual(story.banner(2), "agent_2 picked up O1")
        self.assertEqual(story.hands(2, 1), "holds O1 (agent_1's)")
        self.assertEqual(story.action(2, 1), "place: puts O1 back")
        self.assertEqual(story.banner(3), "agent_2 put O1 back")
        self.assertEqual(story.banner(4), "agent_1 picked up O2     agent_2 picked up O1")
        self.assertEqual(story.action(4, 0), "place: puts O2 back")
        self.assertEqual(story.action(6, 1), "place: places O1 on agent_1's goal")
        self.assertEqual(story.goal(7, 0)[3], "O1 placed by agent_2 at step 7")
        self.assertEqual(story.action(14, 1), "move up ↑, picks up O2")
        self.assertEqual(story.hands(15, 1), "holds O2 (its own)")
        self.assertEqual(story.action(22, 1), "move left ←, places O2 on its goal")
        self.assertEqual(story.goal(23, 1)[3], "O2 placed by itself at step 23")

    def test_a_hostage_to_the_end(self):
        result = self.runs["hostage"]
        story = _Story(result, NAMES)
        last = len(story.frames) - 1
        self.assertEqual(story.hands(last, 0), "holds O2 (agent_2's)")
        self.assertEqual(story.goal(last, 1), (False, "NOT DONE", story.goal(0, 1)[2],
                                               "O2 is in agent_1's hands"))
        self.assertEqual(story.carried(0, last), [(3, last, 1)])
        canvas = _Canvas(result, NAMES, "relative", "", 25, 3)
        try:
            self.assertEqual(canvas._ending(), "no progress any more (deadlock)")
        finally:
            canvas.close()

    def test_a_hand_over(self):
        """Transition 15: agent_2 hands agent_1 its own object, which it took at 2; agent_1,
        blocked by agent_2 in the same transition, receives it.  (Until 2026-10-30 the
        hand-over was forced: agent_1 held O2 hostage and had to give it up at 10.)"""
        result = self.runs["forced"]
        story = _Story(result, NAMES)
        self.assertEqual([tuple(e[:5]) for e in result["events"] if e[1] == "handed over"],
                         [(15, "handed over", 1, 0, 0)])
        self.assertEqual(result["trace"][14]["action"][1], GIVE)
        self.assertEqual(story.action(14, 0), "move left ← (blocked by agent_2)")
        self.assertEqual(story.action(14, 1), "give: hands O1 to agent_1")
        self.assertEqual(story.banner(15), "agent_2 handed O1 to agent_1")
        self.assertEqual(story.last_event(15, 0), "step 15: received O1 from agent_2")
        self.assertEqual(story.hands(15, 0), "holds O1 (its own)")
        self.assertEqual(story.marks(15)[(15, 1)], ["◆O1"])
        self.assertFalse(story.forced(15, ("handed over", 1, 0, 0)))

    def test_a_put_back(self):
        """The put-backs of the helping run (3, 5); the put_back run, a hinderer holding
        its partner's object to the step limit, gives the ending."""
        result = self.runs["helping"]
        story = _Story(result, NAMES)
        backs = [e for e in result["events"] if e[1] == "put back"]
        self.assertEqual([tuple(e[:4]) for e in backs], [(3, "put back", 1, 0), (5, "put back", 0, 1)])
        for t, _, by, obj, _ in backs:
            self.assertEqual(story.action(t - 1, by), f"place: puts O{obj + 1} back")
            self.assertEqual(story.marks(t)[(t, by)], [f"▼O{obj + 1}"])
            self.assertEqual(story.hands(t, by), "empty")
        result = self.runs["put_back"]
        canvas = _Canvas(result, NAMES, "relative", "", 25, 3)
        try:
            self.assertEqual(canvas._ending(), "the step limit was reached")
        finally:
            canvas.close()


class HandOffTests(unittest.TestCase):
    """Two exchangers in their own rooms: each puts its partner's object down in the
    doorway (7, 10) and picks its own up from there (12, 11; 12, 13 until 2026-10-30)."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(
            exchange_scene(), ("exchange", "exchange"), weights=(30.0, 30.0), n_steps=40,
            cfg=Config(num_threads=2, set_down_for_partner=False, hand_over=False))

    def test_the_run_is_the_one_described(self):
        self.assertEqual([tuple(e[:4]) for e in self.result["events"] if e[1] == "put down"],
                         [(7, "put down", 0, 1), (10, "put down", 1, 0)])
        self.assertEqual(tuple(self.result["done_at"]), (18, 15))

    def test_the_figure_says_what_the_record_says(self):
        self.assertEqual(disagreements(self.result), [])
        with tempfile.TemporaryDirectory() as folder:        # ... and so does the file
            path = save_result(Path(folder) / "run.json", self.result)
            self.assertEqual(disagreements(load_result(path)), [])

    def test_putting_down_in_words(self):
        event = ("put down", 0, 1, 1)
        self.assertEqual(event_text(event, NAMES), "agent_1 put O2 down for agent_2")
        self.assertEqual(event_text(event, NAMES, "present", subject=False),
                         "puts O2 down for agent_2")
        self.assertEqual(_callout([event], NAMES),
                         "PUT DOWN O2\nby agent_1\nfor agent_2 to pick up")
        story = _Story(self.result, NAMES)
        self.assertEqual(story.action(6, 0), "place: puts O2 down for agent_2")
        self.assertEqual(story.banner(7), "agent_1 put O2 down for agent_2")
        self.assertEqual(story.goal(7, 1)[3], "O2 is on the hand-off cell")
        self.assertEqual(story.marks(7)[(7, 0)], ["▼O2"])
        self.assertEqual(story.hands(7, 0), "empty")
        self.assertEqual(story.action(9, 1), "place: puts O1 down for agent_1")
        self.assertEqual(story.goal(10, 0)[3], "O1 is on the hand-off cell")
        self.assertEqual(story.action(11, 0), "move right →, picks up O1")
        self.assertEqual(story.action(10, 1), "pick up: picks up O2")

    def test_the_map_shows_the_rooms_and_the_hand_off_cell(self):
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            self.assertEqual(canvas.story.handoff, [(4, 2)])
            self.assertEqual([len(room) for room in canvas.story.rooms], [14, 14])
            for f in range(len(canvas.frames)):
                canvas.draw(f)
                canvas.fig.canvas.draw()
                legend = canvas.ax_map.get_legend()
                labels = [text.get_text() for text in legend.get_texts()]
                self.assertEqual(labels[-3:], ["agent_1's room", "agent_2's room",
                                               "hand-off cell"])
                box, axes = legend.get_window_extent(), canvas.ax_map.get_window_extent()
                self.assertTrue(axes.x0 <= box.x0 and box.x1 <= axes.x1
                                and axes.y0 <= box.y0 and box.y1 <= axes.y1, f)
                boxes = label_boxes(canvas)
                self.assertIn("hand-off cell", [text for text, _ in boxes])
                for n, (text, one) in enumerate(boxes):      # no label covers another
                    self.assertFalse(one.overlaps(box), (f, text))
                    for other, other_box in boxes[n + 1:]:
                        self.assertFalse(one.overlaps(other_box), (f, text, other))
        finally:
            canvas.close()

    def test_two_objects_on_the_hand_off_cell_are_drawn_apart(self):
        frame = next(f for f, row in enumerate(frames_of(self.result))
                     if row["status"][0] >= 4 and row["status"][1] >= 4)
        self.assertEqual(frame, 10)
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            canvas.draw(frame)
            squares = [tuple(np.round(line.get_xydata()[0], 6)) for line in canvas.ax_map.lines
                       if line.get_marker() == "s" and line.get_markersize() == 12.5]
            self.assertEqual(len(squares), 6)
            self.assertEqual(len(set(squares)), 6)
        finally:
            canvas.close()

    def test_a_scene_without_them_has_the_legend_of_before(self):
        result = run_baseline(two_room_scene(), cfg=CFG, n_steps=6)
        canvas = _Canvas(result, NAMES, "relative", "", 25, 3)
        try:
            self.assertEqual((canvas.story.handoff, canvas.story.rooms), ([], None))
            canvas.draw(0)
            labels = [text.get_text() for text in canvas.ax_map.get_legend().get_texts()]
            self.assertEqual(len(labels), 5)
            self.assertNotIn("hand-off cell", [text for text, _ in label_boxes(canvas)])
        finally:
            canvas.close()


class SocialBeliefTests(unittest.TestCase):
    """Where the robots have to infer each other's social goal the figure shows what each
    takes its partner's to be, next to its belief about the partner's destination."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(
            exchange_scene(), ("exchange", "exchange"), weights=(30.0, 30.0), n_steps=40,
            cfg=Config(num_threads=2, set_down_for_partner=False, hand_over=False,
                       partner_social_goal="inferred", exchange_level1="copy_unweighted"))

    def bars(self, canvas) -> dict:
        """{(robot, hypothesis): (height, outlined)} of the social-goal panel as drawn."""
        out = {}
        for patch in canvas.ax_social.patches:
            centre = patch.get_x() + 0.5 * patch.get_width()
            k = int(round(centre))
            robot = 0 if centre < k else 1
            out[robot, k] = (patch.get_height(), patch.get_edgecolor()[3] > 0
                             and tuple(patch.get_edgecolor()[:3]) == (0.0, 0.0, 0.0))
        return out

    def test_the_panel_shows_the_beliefs_of_the_record(self):
        r = self.result
        canvas = _Canvas(r, NAMES, "relative", "", 25, 3)
        try:
            self.assertTrue(canvas.story.inferred)
            self.assertEqual([G.SOCIAL_GOALS[h] for h in canvas.story.hypotheses],
                             ["cooperation", "conflict", "competition", "coercion", "exchange"])
            frames = frames_of(r)
            for f in (0, 3, 7, len(frames) - 1):
                canvas.draw(f)
                bars = self.bars(canvas)
                self.assertEqual(len(bars), 10)
                for i in (0, 1):
                    belief = np.asarray(frames[f]["social_belief"][i], dtype=float)
                    for k, h in enumerate(canvas.story.hypotheses):
                        height, outlined = bars[i, k]
                        self.assertAlmostEqual(height, belief[h], places=12)
                        self.assertEqual(outlined, G.SOCIAL_GOALS[h] == "exchange")
            # the last frame is the belief the rollout ended with
            last = np.asarray(r["final_social_belief"][0], dtype=float)
            self.assertAlmostEqual(self.bars(canvas)[0, 4][0], last[G.EXCHANGE], places=12)
            self.assertGreater(last[G.EXCHANGE], 0.95)
            self.assertEqual(canvas.ax_social.get_title(loc="left"),
                             "... and about its social goal")
        finally:
            canvas.close()

    def test_nothing_in_the_two_belief_panels_overlaps(self):
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            canvas.draw(7)
            canvas.fig.canvas.draw()
            renderer = canvas.fig.canvas.get_renderer()
            for ax in (canvas.ax_belief, canvas.ax_social):
                boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()
                         if t.get_text()]
                self.assertEqual(len(boxes), 6 if ax is canvas.ax_belief else 5)
                for a, b in zip(boxes, boxes[1:]):
                    self.assertLess(a.x1, b.x0)          # no tick label runs into the next
            left = canvas.ax_belief.get_window_extent()
            right = canvas.ax_social.get_window_extent()
            self.assertLess(left.x1, right.x0)
            legend = canvas.ax_belief.get_legend()
            self.assertEqual([t.get_text() for t in legend.get_texts()],
                             ["agent_1's belief", "agent_2's belief", "the true one", "uniform"])
            box, figure = legend.get_window_extent(renderer), canvas.fig.bbox
            self.assertTrue(left.x0 <= box.x0 and box.x1 <= right.x1 <= figure.x1)
            for ax in (canvas.ax_belief, canvas.ax_social):  # each title over its own panel
                title = ax._left_title.get_window_extent(renderer)
                self.assertTrue(title.width > 0 and title.x1 <= ax.get_window_extent().x1 + 1.0)
        finally:
            canvas.close()

    def test_a_record_with_known_goals_has_the_one_belief_panel_of_before(self):
        result = run_baseline(two_room_scene(), cfg=CFG, n_steps=6)
        told = run_pairing(two_room_scene(), ("cooperation", "cooperation"), weights=(1.0, 1.0),
                           cfg=CFG, n_steps=6)
        for record in (result, told):
            canvas = _Canvas(record, NAMES, "relative", "", 25, 3)
            try:
                self.assertFalse(canvas.story.inferred)
                self.assertIsNone(canvas.ax_social)
                canvas.draw(2)
                labels = [t.get_text() for t in canvas.ax_belief.get_legend().get_texts()]
                self.assertEqual(labels[2], "the true destination")
            finally:
                canvas.close()

    def test_the_file_says_the_same(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(Path(folder) / "run.json", self.result)
            again = load_result(path)
            self.assertEqual(disagreements(again), [])
            a = [frame["social_belief"] for frame in frames_of(self.result)]
            b = [frame["social_belief"] for frame in frames_of(again)]
            self.assertTrue(all(np.allclose(x, y) for x, y in zip(a, b)))


class TextTests(unittest.TestCase):
    """The figure is set in a font that ships with matplotlib, and what has to fit is measured:
    it looks the same wherever it is drawn."""

    @classmethod
    def setUpClass(cls):
        # frame 6: the longest texts of the status table ("tried to place O1 (not ...") --
        # on the tiny map since 2026-10-30 (on the two-room scene the cooperators, unsure of
        # each other's destination, no longer set down on a guess)
        cls.result = run_pairing(Scene(name="tiny", **TINY), ("cooperation", "cooperation"),
                                 weights=(1.0, 1.0), cfg=CFG, n_steps=8)

    def overlaps_in_the_status_table(self, canvas: _Canvas, f: int) -> list:
        canvas.draw(f)
        canvas.fig.canvas.draw()
        renderer = canvas.fig.canvas.get_renderer()
        panel = canvas.ax_status.get_window_extent()
        boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in canvas.ax_status.texts
                 if t.get_text() not in ("1", "2")]           # the numbers on the two markers
        found = [("outside the panel", text) for text, box in boxes
                 if box.x0 < panel.x0 or box.x1 > panel.x1]
        found += [(a, b) for n, (a, box) in enumerate(boxes)
                  for b, other in boxes[n + 1:] if box.overlaps(other)]
        return found

    def test_the_font(self):
        self.assertEqual(FONT, "DejaVu Sans")
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            canvas.draw(6)
            texts = (list(canvas.fig.texts) + list(canvas.ax_map.texts)
                     + list(canvas.ax_status.texts) + list(canvas.ax_time.texts))
            self.assertGreater(len(texts), 30)
            self.assertEqual({t.get_fontname() for t in texts}, {"DejaVu Sans"})
        finally:
            canvas.close()
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3, font=None)
        try:
            self.assertEqual(canvas.style, {})               # matplotlib's own configuration
        finally:
            canvas.close()

    def test_nothing_in_the_status_table_runs_into_its_neighbour(self):
        """Not the badge into the task, not one column into the other, in either font."""
        self.assertIn("tried to place O1", _Story(self.result, NAMES).last_event(6, 1))
        for font in (FONT, None):
            canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3, font=font)
            try:
                for f in (0, 5, 6):
                    self.assertEqual(self.overlaps_in_the_status_table(canvas, f), [],
                                     (font, f))
            finally:
                canvas.close()

    def test_long_names_and_titles_are_scaled_to_fit(self):
        names = ("the_orange_robot_with_a_long_name", "red")
        canvas = _Canvas(self.result, names, "relative", "a very long title " * 9, 25, 3)
        try:
            self.assertEqual(self.overlaps_in_the_status_table(canvas, 6), [])
            figure = canvas.fig.bbox
            renderer = canvas.fig.canvas.get_renderer()
            for text in canvas.fig.texts:                    # title, step and headline
                box = text.get_window_extent(renderer)
                self.assertTrue(figure.x0 <= box.x0 and box.x1 <= figure.x1, text.get_text())
            self.assertLess(canvas.fig.texts[0].get_fontsize(), 12.5)
        finally:
            canvas.close()

    def test_measuring(self):
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            wide, high = canvas._extent("picked up O1", fontsize=9)
            twice = canvas._extent("picked up O1 picked up O1", fontsize=9)[0]
            self.assertTrue(40 < wide < 90 and 6 < high < 14, (wide, high))
            self.assertGreater(twice, 1.9 * wide)
            fitted = canvas._fit("one two three four five six seven eight", 80, fontsize=9)
            self.assertEqual(fitted.count("\n"), 1)
            self.assertTrue(all(canvas._extent(line, fontsize=9)[0] <= 80
                                for line in fitted.split("\n")))
            self.assertTrue(fitted.endswith("…"))            # two lines were not enough
            self.assertEqual(canvas._fit("short", 80, fontsize=9), "short")
            self.assertEqual(canvas._squeezed("short", 500, 9), ("short", 9))
            text, size = canvas._squeezed("short", 20, 9)      # scaled down
            self.assertEqual(text, "short")
            self.assertTrue(6.5 <= size < 9)
            text, size = canvas._squeezed("far too long for the room it has", 20, 9)
            self.assertEqual(size, 6.5)                      # ... and cut where that is not enough
            self.assertTrue(text.endswith("…") and canvas._extent(text, fontsize=6.5)[0] <= 20)
        finally:
            canvas.close()


class MapTests(unittest.TestCase):
    """The map panel: where the legend and the labels go."""

    def test_on_a_walled_map_the_legend_sits_on_the_top_wall(self):
        result = run_baseline(two_room_scene(), cfg=CFG, n_steps=20)
        canvas = _Canvas(result, NAMES, "relative", "", 25, 3)
        try:
            self.assertFalse(canvas.framed)
            self.assertEqual(list(canvas.view), list(canvas.extent))
            canvas.draw(len(canvas.frames) - 1)
            canvas.fig.canvas.draw()
            legend = canvas.ax_map.get_legend().get_window_extent()
            axes = canvas.ax_map.get_window_extent()
            self.assertTrue(axes.x0 <= legend.x0 and legend.x1 <= axes.x1
                            and axes.y0 <= legend.y0 and legend.y1 <= axes.y1)
            # the label of a destination says what was delivered there, in one piece
            boxes = label_boxes(canvas)
            texts = [text for text, _ in boxes]
            for i in (0, 1):
                k = result["scene"]["goal_ids"][i]
                self.assertIn(f"G{k}: {NAMES[i]}'s goal\nO{i + 1} delivered ✓", texts)
            self.assertFalse([t for t in texts if t.startswith("O1") or t.startswith("O2")])
            for f in range(len(canvas.frames)):              # and no label covers another
                canvas.draw(f)
                boxes = label_boxes(canvas)
                for n, (text, box) in enumerate(boxes):
                    for other, other_box in boxes[n + 1:]:
                        self.assertFalse(box.overlaps(other_box), (f, text, other))
        finally:
            canvas.close()

    def test_without_walls_there_is_room_around_the_map(self):
        """A corridor and a 3 x 3 map without walls: the legend, the labels of destinations
        on the border and the ticks each have their place, inside the figure."""
        tiny = Scene(grid=np.ones((3, 3)), starts=((0, 1), (2, 1)), pickups=((2, 2), (0, 0)),
                     destinations=((0, 2), (2, 0)), candidate_cells=((1, 0),))
        corridor = Scene(grid=np.ones((12, 2)), starts=((8, 0), (11, 0)),
                         pickups=((0, 1), (9, 0)), destinations=((0, 0), (9, 1)),
                         candidate_cells=())
        runs = [run_baseline(corridor, cfg=CFG, n_steps=40),
                run_baseline(tiny, cfg=Config(num_threads=2, resolution=1.0), n_steps=20)]
        for result in runs:
            canvas = _Canvas(result, NAMES, "relative", "", 25, 3)
            try:
                self.assertTrue(canvas.framed)
                x0, x1, y0, y1 = canvas.extent
                view, per_pt = canvas.view, canvas.per_pt
                room = [(x0 - view[0]) / per_pt, (view[1] - x1) / per_pt,
                        (y0 - view[2]) / per_pt, (view[3] - y1) / per_pt]      # points
                np.testing.assert_allclose(room, [_ROOM[side] for side in
                                                  ("left", "right", "bottom", "top")])
                last = len(canvas.frames) - 1
                self.assertTrue(all(result["done_at"][i] is not None for i in (0, 1)))
                for f in (0, last):
                    canvas.draw(f)
                    boxes = label_boxes(canvas)
                    figure = canvas.fig.bbox
                    axes = canvas.ax_map.get_window_extent()
                    legend = canvas.ax_map.get_legend().get_window_extent()
                    # the legend: inside the figure, above the map itself
                    top = canvas.ax_map.transData.transform((x0, y1))[1]
                    self.assertTrue(figure.x0 <= legend.x0 and legend.x1 <= figure.x1
                                    and legend.y1 <= axes.y1 + 1 and legend.y0 >= top)
                    # every label: inside the panel, clear of the legend and of the others
                    for n, (text, box) in enumerate(boxes):
                        self.assertTrue(axes.x0 - 1 <= box.x0 and box.x1 <= axes.x1 + 1
                                        and axes.y0 - 1 <= box.y0 and box.y1 <= axes.y1 + 1,
                                        text)
                        self.assertFalse(box.overlaps(legend), text)
                        for other, other_box in boxes[n + 1:]:
                            self.assertFalse(box.overlaps(other_box), (text, other))
                texts = [text for text, _ in boxes]              # the last frame
                self.assertEqual(sum("delivered ✓" in text for text in texts), 2)
            finally:
                canvas.close()


class ArrivalTests(unittest.TestCase):
    """Two robots without social goals: each takes its own object on arriving at its place
    and sets it down on arriving at its destination -- events without a hand action."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_baseline(two_room_scene(), cfg=CFG, n_steps=20)
        cls.story = _Story(cls.result, NAMES)

    def test_an_event_is_called_out_where_the_robot_arrived(self):
        self.assertEqual([(e[0], e[1], e[2], e[3]) for e in self.result["events"]],
                         [(7, "picked up", 0, 0), (9, "set down", 0, 0),
                          (9, "picked up", 1, 1), (11, "set down", 1, 1)])
        (age, cell, events), = self.story.callouts(7, 3)
        self.assertEqual((age, cell), (0, tuple(self.result["scene"]["objects"][0])))
        self.assertNotEqual(tuple(self.story.frames[6]["pos"][0]), cell)     # it was not there
        self.assertEqual(events, [("picked up", 0, 0, None)])
        cells = {cell for _, cell, _ in self.story.callouts(9, 1)}
        self.assertEqual(cells, {tuple(self.result["scene"]["destinations"][0]),
                                 tuple(self.result["scene"]["objects"][1])})

    def test_a_move_that_picks_up_or_places(self):
        # "move up <arrow>, picks up O1": the arrow is one character
        self.assertRegex(self.story.action(6, 0), r"^move \w+ ., picks up O1$")
        self.assertRegex(self.story.action(8, 0), r"^move \w+ ., places O1 on its goal$")
        self.assertEqual(self.story.goal(9, 0)[3], "O1 placed by itself at step 9")
        self.assertEqual(self.story.goal(9, 1)[3], "O2 is in its hands")
        self.assertEqual(self.story.subtitle[0], "physical goal only · level 0")


class HeadlineTests(unittest.TestCase):
    def test_the_headline_names_the_reading_of_competition(self):
        """A competitor behaves differently under each setting of ``Config.competition``;
        the default headline says which one the record was written under."""
        scene = Scene(grid=np.ones((3, 3)), starts=((0, 1), (2, 1)), pickups=((2, 2), (0, 0)),
                      destinations=((0, 2), (2, 0)), candidate_cells=((1, 0),))
        said = {"race_then_hinder": "agent_1: competition (race, then hinder)",
                "race": "agent_1: competition (race)", "table": "agent_1: competition (Table I)",
                "table_hostile": "agent_1: competition (Table I, hostile)",
                "rival": "agent_1: competition (rival)"}
        self.assertEqual(set(said), set(G.COMPETITION_FORMS))
        for mode, text in said.items():
            result = run_pairing(scene, ("competition", "cooperation"), n_steps=2,
                                 cfg=Config(num_threads=2, resolution=1.0, competition=mode))
            title = _title(result, NAMES)
            self.assertTrue(title.startswith(text + "     agent_2: cooperation"), title)
        # a record that does not say, and goals that have one reading only
        self.assertEqual(goal_label("competition"), "competition")
        self.assertEqual(goal_label("competition", "race"), "competition (race)")
        self.assertEqual(goal_label("hinder", "race"), "hinder")
        self.assertEqual(goal_label(None, "race"), "physical goal only")
        result.pop("config")
        self.assertTrue(_title(result, NAMES).startswith("agent_1: competition     agent_2"))


class OldRecordTests(unittest.TestCase):
    def test_a_record_written_under_any_object_goals_is_refused(self):
        """Robot i's goal is done by object i only.  A record of an older version in which
        any object completed a goal cannot be told truthfully by this figure: it says so."""
        result = run_baseline(Scene(grid=np.ones((3, 3)), starts=((0, 1), (2, 1)),
                                    pickups=((2, 2), (0, 0)), destinations=((0, 2), (2, 0)),
                                    candidate_cells=((1, 0),)),
                              cfg=Config(num_threads=2, resolution=1.0), n_steps=20)
        self.assertNotIn("goal_objects", result["scene"])
        _Story(result, NAMES)                                    # today's records are read
        old = dict(result, scene=dict(result["scene"], goal_objects="any"))
        with self.assertRaisesRegex(ValueError, "no longer has"):
            _Story(old, NAMES)
        _Story(dict(result, scene=dict(result["scene"], goal_objects="own")), NAMES)


class FileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(two_room_scene(), ("cooperation", "cooperation"),
                                 weights=(30.0, 30.0), cfg=CFG, n_steps=20)

    def test_gif_holds_the_frames_after_an_event_longer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = render_gif(self.result, Path(folder) / "run.gif", fps=4, event_hold=1.0,
                              placed_hold=2.0, end_hold=3.0, dpi=60)
            n = len(frames_of(self.result))
            durations = []
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, n)
                for k in range(n):
                    image.seek(k)
                    durations.append(image.info["duration"])
            self.assertEqual(durations[:6], [250, 250, 1000, 250, 250, 2000])
            self.assertEqual(durations[-1], 3000)

    def test_video_holds_the_same_frames_as_long(self):
        """The MP4 (since 2026-10-17): at twice ``fps`` a frame is repeated for as long as
        the GIF holds it -- 250 ms one frame at 8 frames a second, 1 s eight -- every frame
        is the figure (padded with white to a multiple of 16 pixels), and the video lasts
        as long as the GIF."""
        import imageio_ffmpeg

        options = dict(fps=4, event_hold=1.0, placed_hold=2.0, end_hold=3.0, dpi=60)
        with tempfile.TemporaryDirectory() as folder:
            gif = render_gif(self.result, Path(folder) / "run.gif", **options)
            with Image.open(gif) as image:
                held = []
                for k in range(image.n_frames):
                    image.seek(k)
                    held.append(image.info["duration"])
            path = render_video(self.result, Path(folder) / "run.mp4", **options)
            self.assertGreater(path.stat().st_size, 10000)
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
            self.assertEqual(frames, sum(round(d * 8 / 1000) for d in held))
            self.assertAlmostEqual(seconds, sum(held) / 1000, delta=0.2)
            reader = imageio_ffmpeg.read_frames(str(path))
            meta = reader.__next__()
            w, h = meta["size"]
            self.assertEqual((w % 16, h % 16), (0, 0))
            first = np.frombuffer(reader.__next__(), dtype=np.uint8).reshape(h, w, 3)
            reader.close()
            still = save_frame(self.result, Path(folder) / "first.png", 0, dpi=60)
            with Image.open(still) as image:
                want = np.asarray(image.convert("RGB")).astype(float)
            got = first[:want.shape[0], :want.shape[1]].astype(float)
            self.assertLess(np.abs(got - want).mean(), 3.0)          # H.264 at crf 18
            self.assertTrue((first[want.shape[0]:] >= 250).all())    # the padding: white

    def test_the_robots_are_green_and_red(self):
        """agent_1 green, agent_2 red (the user, 2026-10-17)."""
        self.assertEqual(ROBOT_COLOURS, ("#2ca25f", "#b2182b"))

    def test_single_frames_and_every_frame_draws(self):
        with tempfile.TemporaryDirectory() as folder:
            last = save_frame(self.result, Path(folder) / "last.png", dpi=50)
            with Image.open(last) as image:
                self.assertEqual(image.size, (730, 465))             # 14.6 x 9.3 in at 50 dpi
            self.assertGreater(last.stat().st_size, 5000)
        canvas = _Canvas(self.result, NAMES, "raw", "", 25, 3)
        try:
            for f in range(len(canvas.frames)):
                canvas.draw(f)
        finally:
            canvas.close()
        with self.assertRaises(ValueError):
            _Canvas(self.result, NAMES, "absolute", "", 25, 3)


if __name__ == "__main__":
    unittest.main()
