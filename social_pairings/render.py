"""Animate a rollout as an annotated MP4 video or GIF (or save single frames as images).

Five panels, all drawn from the record alone:

* **map** -- the scene.  Every pick-up, put-back, put-down, hand-over and delivery is
  called out in words at the cell where it happened and stays up for a few frames; a
  delivered object stays marked on its destination to the end.  Each robot shows what it
  is doing in this step: an arrow for a move, a tag for "stay" and the hand actions (pick
  up, place, take, give, swap).  Where the
  scene has them, each robot's own room is tinted in its colour and a hand-off cell is
  framed and labelled.
* **status** -- one column per robot: its action in this step and what comes of it, what is
  in its hands, the last thing it did to an object, and whether its goal is done.
* **beliefs** -- each robot's belief about its partner's destination -- where it has to
  infer which object its partner wants (``Config.partner_object="inferred"``), its belief
  about that in the same panel, before it -- and, where it has to infer its partner's
  social goal (``Config.partner_social_goal="inferred"``), its belief about that, next to
  it.
* **values** -- why the action was chosen: every action's value relative to the one taken,
  split into its physical part and its social part,

      Q(a) - Q(a*) = [Q_phys(a) - Q_phys(a*)] + [Q_soc(a) - Q_soc(a*)],   Q_soc = Q - Q_phys

  so a bar above zero says "this action would have been better on that count", and the two
  bars of the action taken are zero.  ``values="raw"`` shows levels instead (physical Q and
  the immediate social reward).
* **timeline** -- one row per robot: the action of every step so far, what the robot
  carried, and every event, so "who picked up or put down what, and when" can be read off
  any frame.

Frames that follow an event are held longer, and the last frame longest.

The figure is set in DejaVu Sans, which ships with matplotlib, so that it looks the same on
every machine (``font=None``: whatever font matplotlib is configured with); what has to fit
somewhere is measured and wrapped or scaled to fit.

Time: the frame "step t" shows the state after t transitions together with the actions the
robots choose there; what those actions do is seen in the state of step t + 1.
"""
from __future__ import annotations

import os
import textwrap
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import (ACTION_NAMES, GIVE, GONE, HELD0, HELD1, MOVE_OF, MOVES, N_ACTIONS,
                        PICK, WAIT)
from .goals import COMPETITION_NAMES, CONFLICT_NAMES, SOCIAL_GOALS, cooperation_reading
from .config import written_with
from .serialize import action_names

__all__ = ["render_video", "render_value_video", "render_videos", "video_paths", "render_gif",
           "save_frame", "frames_of", "event_text", "goal_label"]

FONT = "DejaVu Sans"                            # ships with matplotlib
# R1 green, R2 red (the user, 2026-10-17; orange and a lighter red until then).  A deep red
# and a lighter green, so that the two stay apart for red-green colour-blind eyes too
# (Delta E 13 for deuteranopia, where a plain green and red are 4); the robots carry their
# numbers as well
ROBOT_COLOURS = ("#2ca25f", "#b2182b")
SPARE_COLOUR = "0.50"
INK, MUTED = "0.15", "0.42"                    # text wears ink, never a series colour
DONE_FILL, OPEN_FILL = "#2e7d32", "0.86"       # goal badges: done / not done
FRESH = "#fff3b0"                              # what has just happened
ACTION_WORDS = ("move up", "move right", "move down", "move left", "stay", "pick up", "place",
                "take", "give", "swap")
ACTION_TICKS = ("up", "right", "down", "left", "stay", "pick", "place", "take", "give", "swap")
ACTION_GLYPHS = ("↑", "→", "↓", "←", "•", "P", "D", "T", "G", "S")
EVENT_GLYPHS = {"picked up": "▲", "lifting": "△", "put back": "▼", "put down": "▼",
                "dropped": "▼", "handed over": "◆", "took": "◆", "set down": "★",
                "set down in vain": "✗", "handed over in vain": "✗"}
CHECK = "✓"
LEGEND = ("↑ → ↓ ← move (grey: blocked)   • stay   P pick   D place   T take   G give   "
          "S swap   △ lifting   ▲ picked up   ▼ put down   ◆ changed hands   ✗ in vain   "
          "★ goal done   ▬ carrying (colour: whose object)")
_HEADLINE = {"picked up": "PICKED UP", "lifting": "LIFTING", "put back": "PUT BACK",
             "put down": "PUT DOWN", "dropped": "LEFT", "handed over": "HANDED OVER",
             "took": "TOOK", "set down": "PLACED",
             "set down in vain": "FAILED TO PLACE", "handed over in vain": "FAILED TO HAND OVER"}
ROOM_TINT = 0.13                               # opacity of a robot's colour on its own room
# a social goal as a tick label of the belief panel
GOAL_TICKS = {"cooperation": "cooper.", "competition": "compet.", "no social goal": "none"}
# ... where more than five social goals are held possible (the default: seven)
GOAL_TICKS_SHORT = {"cooperation": "coop.", "conflict": "confl.", "competition": "comp.",
                    "coercion": "coerc.", "exchange": "exch.", "no social goal": "none"}
SPLIT_TITLE = 8.6                              # title size where the belief panel is two
# each robot's leading belief (its most probable object, destination and social goal for its
# partner), named in bold above the bars and marked over them: agent_1's in green, agent_2's
# in purple (the user, 2026-10-13)
LEAD_COLOURS = ("#1b7a3c", "#6a3d9a")
LEAD_TOP = 1.14                                # the belief panels' y range: bars to 1, markers
LEAD_ROWS = (13.0, 2.5)                        # the two lines over a panel, points above it
LEAD_PAD = 25.0                                # ... and its title over them, in points
LEAD_TIE = 5e-3                                # beliefs this close lead together
# room around the map, in points, on a side where its border is not all wall: for the legend
# and a destination's label along the top, a label along the bottom, half a label at the sides
_ROOM = dict(left=30.0, right=30.0, bottom=30.0, top=64.0)
# where a callout may sit relative to its cell, in points, nearest first
_OFFSETS = ((74, 48), (-74, 48), (74, -48), (-74, -48), (0, 62), (0, -62), (112, 0),
            (-112, 0), (120, 52), (-120, 52), (120, -52), (-120, -52), (0, 96), (0, -96))

Event = Tuple[str, int, int, Optional[int]]     # (what, robot, object, other robot)


def _values(x) -> np.ndarray:
    """A (6,) array of Q values; illegal actions (-inf, or null after JSON) become nan."""
    out = np.array([np.nan if v is None else v for v in x], dtype=float)
    out[~np.isfinite(out)] = np.nan
    return out


def frames_of(result: dict) -> list:
    """One frame per transition (the state it started from, with the actions chosen there)
    plus a last frame for the final state.  ``arrived_by``: the events of the transition
    that led to the frame's state; ``events``: those of the transition that starts there."""
    frames = []
    previous: tuple = ()
    for row in result["trace"]:
        frames.append(dict(
            step=row["step"] - 1, pos=row["pos"], status=row["status"],
            object_cells=row["object_cells"], holds=row["holds"], done=row["done"],
            belief=row["belief"], social_belief=row.get("social_belief"),
            object_belief=row.get("object_belief"),
            action=row["action"], blocked=row["blocked"],
            inertia=tuple(bool(x) for x in row.get("inertia", (False, False))),
            q=[_values(v) for v in row["q"]], q_physical=[_values(v) for v in row["q_physical"]],
            social_reward=[_values(v) for v in row["social_reward"]],
            arrived_by=previous, events=row["events"]))
        previous = row["events"]
    last = result["trace"][-1] if result["trace"] else None
    if last is not None:
        frames.append(dict(
            step=last["step"], pos=last["next_pos"], status=last["next_status"],
            object_cells=last["next_object_cells"], holds=last["next_holds"],
            done=[d is not None and d <= last["step"] for d in result["done_at"]],
            belief=result.get("final_belief", last["belief"]),
            social_belief=result.get("final_social_belief", last.get("social_belief")),
            object_belief=result.get("final_object_belief", last.get("object_belief")),
            action=None, blocked=None, q=None, q_physical=None, social_reward=None,
            arrived_by=previous, events=()))
    return frames


# ---- events in words ---------------------------------------------------------------------------

def event_text(event: Event, names: Sequence[str], tense: str = "past",
               subject: bool = True, forced: bool = False, why: str = "destination") -> str:
    """One event in words.  tense: "past" ("agent_1 picked up O2") or "present" ("picks up
    O2"); subject=False leaves the robot's name out.  forced: a hand-over its giver did
    not choose (``forced_swap=True``: handed its own object, a robot has to give up what it
    holds).  why: what was wrong with a set-down in vain -- "destination" (the robot did
    not stand on its partner's destination), "object" (it held another object than its
    partner's; only a robot that has to infer which object its partner wants can try
    that) or "both"."""
    what, by, obj, other = event
    o = f"O{obj + 1}"
    past = tense == "past"
    if what == "picked up":
        text = f"{'picked' if past else 'picks'} up {o}"
    elif what == "lifting":                      # a pick-up of several transitions begun
        text = f"{'began' if past else 'begins'} to lift {o}"
    elif what == "put back":
        text = f"{'put' if past else 'puts'} {o} back"
    elif what == "put down":                     # on a hand-off cell, for its owner
        text = f"{'put' if past else 'puts'} {o} down" + (
            "" if other is None else f" for {names[other]}")
    elif what == "dropped":                      # in a swap, on another object's place
        text = f"{'left' if past else 'leaves'} {o} there"
    elif what == "took":                         # out of the other robot's hands
        text = f"{'took' if past else 'takes'} {o} from {names[other]}"
    elif what == "handed over" and forced:
        text = f"{'had' if past else 'has'} to give {o} to {names[other]}"
    elif what == "handed over":
        text = f"{'handed' if past else 'hands'} {o} to {names[other]}"
    elif what == "set down":
        whose = "its goal" if other in (None, by) else f"{names[other]}'s goal"
        text = f"{'placed' if past else 'places'} {o} on {whose}"
    elif what == "set down in vain":
        text = f"{'tried' if past else 'tries'} to place {o} ({_vain(why, names[other])})"
    elif what == "handed over in vain":
        text = (f"{'tried' if past else 'tries'} to hand {o} to {names[other]} "
                f"({_vain('object', names[other])})")
    else:
        text = f"{what} {o}"
    return f"{names[by]} {text}" if subject else text


def _vain(why: str, partner: str) -> str:
    """What was wrong with an act for the partner that did not count."""
    if why == "both":
        return f"neither {partner}'s object nor its destination"
    return f"not {partner}'s {'object' if why == 'object' else 'destination'}"


def _callout(events: Sequence[Event], names: Sequence[str],
             forced: Sequence[bool] = (), why: Sequence[str] = ()) -> str:
    """The lines of the callout for the events of one cell in one transition.  forced: per
    event, a hand-over its giver did not choose; why: per event, what was wrong with a
    set-down in vain (:func:`event_text`)."""
    lines = []
    for n, (what, by, obj, other) in enumerate(events):
        gave_up = what == "handed over" and n < len(forced) and forced[n]
        lines.append(f"{'GIVEN UP' if gave_up else _HEADLINE.get(what, what.upper())} "
                     f"O{obj + 1}")
        if what == "handed over":
            lines.append(f"{names[by]} → {names[other]}" + (" (forced swap)" if gave_up
                                                            else ""))
        elif what == "took":
            lines.append(f"{names[other]} → {names[by]} (taken)")
        elif what == "dropped":
            lines += [f"by {names[by]}", "in a swap"]
        elif what == "set down":
            done = by if other is None else other
            lines += [f"by {names[by]}", f"{names[done]}'s goal is DONE {CHECK}"]
        elif what == "set down in vain":
            lines += [f"by {names[by]}",
                      _vain(why[n] if n < len(why) else "destination", names[other])]
        elif what == "handed over in vain":
            lines += [f"{names[by]} → {names[other]}", _vain("object", names[other])]
        elif what == "put down" and other is not None:
            lines += [f"by {names[by]}", f"for {names[other]} to pick up"]
        else:
            lines.append(f"by {names[by]}")
    return "\n".join(lines)


def _wrap(text: str, width: int, lines: int = 2) -> str:
    out = textwrap.wrap(text, width=width) or [""]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1][:width - 1].rstrip() + "…"
    return "\n".join(out)


# ---- the story of the rollout, frame by frame --------------------------------------------------

class _Story:
    """Everything the panels say, worked out once from the record."""

    def __init__(self, result: dict, names: Sequence[str]) -> None:
        if action_names(result) != ACTION_NAMES:
            raise ValueError('the record was rolled out with "use" (before 2026-10-16), not '
                             "the five hand actions: draw it with the package of then")
        self.result, self.names = result, tuple(names)
        self.frames = frames_of(result)
        if not self.frames:
            raise ValueError("the record holds no transitions to render")
        scene = result["scene"]
        self.candidates = [tuple(c) for c in scene["candidates"]]
        self.homes = [tuple(c) for c in scene["objects"]]
        self.destinations = [tuple(c) for c in scene["destinations"]]
        self.goal_ids = [int(k) for k in scene["goal_ids"]]
        # what a scene may add (records of before 2026-10-07 have neither): the cells on
        # which an object may be put down for its owner, and a room per robot
        self.handoff = [tuple(c) for c in (scene.get("handoff_cells") or ())]
        rooms = scene.get("rooms")
        self.rooms = None if not rooms else [set(tuple(c) for c in room) for room in rooms]
        # the object statuses of a lift in progress (Config.pick_steps > 1, since 2026-10-18):
        # lift0 + i * nstage + (s - 1), robot i s steps into it (hands.py)
        config = written_with(result.get("config") or {})
        lie = len(self.handoff) + (len(self.homes) if config.get("swap_on_place") else 0)
        self.lift0, self.nstage = 4 + lie, int(config.get("pick_steps", 1)) - 1
        self.lift_steps = self.nstage + 1
        self.n_steps = len(result["trace"])
        self.events: List[Tuple[int, Event]] = [
            (int(e[0]), (e[1], int(e[2]), int(e[3]), None if e[4] is None else int(e[4])))
            for e in result["events"]]
        self.done_at = [None if t is None else int(t) for t in result["done_at"]]
        # a hinderer without a goal of its own (Config.hinder_own_goal=False, since
        # 2026-10-14); records of before have every robot's
        self.own_goal = tuple(bool(x) for x in result.get("own_goal", (True, True)))
        if scene.get("goal_objects", "own") != "own":      # a record of an older version
            raise ValueError("this record was written under goal_objects=\"any\", a rule this "
                             "package no longer has: robot i's goal is done by object i only")
        self.placed: List[Optional[Tuple[int, int]]] = [None, None]    # (by robot, object)
        for _, (what, by, obj, other) in self.events:
            if what == "set down":
                self.placed[by if other is None else other] = (by, obj)
        goals = result.get("goals") or result.get("social_goals") or ("none", "none")
        weights = result.get("weights", (0.0, 0.0))
        levels = result.get("levels_used", result.get("levels", (0, 0)))
        # Where a robot has to infer its partner's social goal: the goals it holds possible
        # (their indices in SOCIAL_GOALS), and each robot's own goal as its partner would
        # have to find it -- none for a robot at level 0 or with weight 0.
        config = result.get("config") or {}
        first = self.frames[0].get("social_belief") or (None, None)
        self.inferred = (config.get("partner_social_goal") == "inferred"
                         and any(b is not None for b in first))
        self.hypotheses = [SOCIAL_GOALS.index(str(h)) for h in
                           config.get("social_hypotheses", ())] if self.inferred else []
        # where a robot has to infer which object its partner wants (records of before
        # 2026-10-09 have no such belief): robot j's own object is object j
        self.objects_inferred = (config.get("partner_object") == "inferred"
                                 and self.frames[0].get("object_belief") is not None)
        # the objects in the model (the scenery, after them in the record, is nobody's goal)
        self.n_objects = (len(self.frames[0]["object_belief"][0]) if self.objects_inferred
                          else len(self.homes))
        self.true_social = [
            "none" if goals[i] in (None, "none") or levels[i] == 0 or weights[i] == 0
            else str(goals[i]) for i in (0, 1)]
        self.subtitle = []
        for i in (0, 1):
            if goals[i] in (None, "none") or levels[i] == 0:
                self.subtitle.append(f"physical goal only · level {levels[i]}")
            else:
                self.subtitle.append(f"{goals[i]} · weight {weights[i]:g} · level {levels[i]}")

    def why(self, t: int, event: Event) -> str:
        """What was wrong with the set-down in vain of transition t: "destination", "object"
        or "both" (:func:`event_text`).  The robot used its hands, so it stood where it is
        in frame t; the partner's object is the one with the partner's number."""
        what, by, obj, other = event
        if what != "set down in vain":
            return "destination"
        place = tuple(self.frames[t]["pos"][by]) != self.destinations[other]
        thing = obj != other
        return "both" if place and thing else ("object" if thing else "destination")

    def forced(self, t: int, event: Event) -> bool:
        """Was the hand-over of transition t forced on its giver?  By consent the giver uses
        its hands; under ``forced_swap=True`` a robot that is handed its own object gives up
        what it holds whatever it does."""
        return event[0] == "handed over" and self.frames[t - 1]["action"][event[1]] != GIVE

    def whose(self, obj: int, robot: int) -> str:
        """Object ``obj`` as robot ``robot`` sees it: "its own", "agent_2's", "a spare"."""
        if obj == robot:
            return "its own"
        return f"{self.names[obj]}'s" if obj < 2 else "a spare"

    # -- the table -------------------------------------------------------------------------------
    def action(self, f: int, i: int) -> str:
        """What robot i does in the transition that starts at frame f, and what comes of it."""
        frame = self.frames[f]
        if frame["action"] is None:
            return "— (rollout ended)"
        a = frame["action"][i]
        mine = [event_text(e, self.names, "present", subject=False,
                           forced=self.forced(f + 1, e), why=self.why(f + 1, e))
                for e in frame["events"] if e[1] == i]
        # Config.inertia: it did not undo its last action, which its Q put first
        held = " (inertia)" if frame.get("inertia") and frame["inertia"][i] else ""
        if a >= PICK:
            return (f"{ACTION_WORDS[a]}: " + ("; ".join(mine) if mine else "nothing happens")
                    + held)
        text = ACTION_WORDS[a]
        if a < WAIT:
            text += " " + ACTION_GLYPHS[a]
            if frame["blocked"][i]:
                text += f" (blocked by {self.names[1 - i]})"
        if mine:
            text += ", " + "; ".join(mine)
        return text + held

    def hands(self, f: int, i: int) -> str:
        held = self.frames[f]["holds"][i]
        if held < 0:
            return "empty"
        return f"holds O{held + 1} ({self.whose(held, i)})"

    def last_event(self, f: int, i: int) -> str:
        """The last thing robot i did to an object up to frame f (or was handed)."""
        step = self.frames[f]["step"]
        mine = [(t, e) for t, e in self.events
                if t <= step and (e[1] == i or (e[0] in ("handed over", "took")
                                                and e[3] == i))]
        if not mine:
            return "—"
        t, event = mine[-1]
        if event[1] == i:
            return f"step {t}: " + event_text(event, self.names, "past", subject=False,
                                              forced=self.forced(t, event),
                                              why=self.why(t, event))
        if event[0] == "took":
            return f"step {t}: lost O{event[2] + 1} to {self.names[event[1]]}"
        return f"step {t}: received O{event[2] + 1} from {self.names[event[1]]}"

    def goal(self, f: int, i: int) -> Tuple[bool, str, str, str]:
        """(done, badge, task, detail) of robot i's goal at frame f."""
        frame = self.frames[f]
        if not self.own_goal[i]:         # a hinderer that only works against its partner
            other = self.names[1 - i]
            state = "done" if frame["done"][1 - i] else "not done"
            return False, "NO GOAL", f"keep {other} from its goal", f"{other}'s goal: {state}"
        task = f"bring O{i + 1} to G{self.goal_ids[i]}"
        if frame["done"][i]:
            if self.placed[i] is None:
                return True, f"{CHECK} DONE", task, f"done at step {self.done_at[i]}"
            by, obj = self.placed[i]
            who = "by itself" if by == i else f"by {self.names[by]}"
            return (True, f"{CHECK} DONE", task,
                    f"O{obj + 1} placed {who} at step {self.done_at[i]}")
        status = frame["status"][i]
        lifter = self.lifter(status)
        if status in (HELD0, HELD1):
            holder = status - 1
            where = "in its hands" if holder == i else f"in {self.names[holder]}'s hands"
        elif lifter >= 0:
            who = "it" if lifter == i else self.names[lifter]
            where = (f"being lifted by {who} ({self.lift_stage(status)}/{self.lift_steps})")
        elif status >= 4 + len(self.handoff):
            where = f"on O{status - 4 - len(self.handoff) + 1}'s place"
        elif status >= 4:
            where = "on the hand-off cell"
        else:
            where = "at its place"
        return False, "NOT DONE", task, f"O{i + 1} is {where}"

    def lifter(self, status: int) -> int:
        """The robot lifting an object with this status (-1: none)."""
        if self.nstage <= 0 or status < self.lift0 or status >= self.lift0 + 2 * self.nstage:
            return -1
        return (status - self.lift0) // self.nstage

    def lift_stage(self, status: int) -> int:
        return (status - self.lift0) % self.nstage + 1 if self.lifter(status) >= 0 else 0

    # -- the timeline ----------------------------------------------------------------------------
    def tape(self, i: int, upto: int) -> list:
        """Robot i's transitions 1..upto: [(action, blocked, object carried or -1)]."""
        return [(int(row["action"][i]), row["action"][i] < WAIT and bool(row["blocked"][i]),
                 int(row["holds"][i])) for row in self.frames[:upto]]

    def carried(self, i: int, upto: int) -> list:
        """[(first transition, last transition, object)]: what robot i carried during its
        transitions 1..upto, run by run."""
        runs: list = []
        for t, (_, _, held) in enumerate(self.tape(i, upto), 1):
            if held < 0:
                continue
            if runs and runs[-1][2] == held and runs[-1][1] == t - 1:
                runs[-1][1] = t
            else:
                runs.append([t, t, held])
        return [tuple(run) for run in runs]

    def marks(self, step: int) -> dict:
        """{(transition, robot): [labels]}: the events up to ``step`` as the timeline tags
        them.  A set-down is a star in the row of the robot whose goal it finished and, when
        its partner placed the object, a put-down in the partner's row."""
        out: dict = {}
        for t, (what, by, obj, other) in self.events:
            if t > step:
                continue
            if what == "set down":
                done = by if other is None else other
                out.setdefault((t, done), []).append(f"{EVENT_GLYPHS[what]}O{obj + 1}")
                if done != by:
                    out.setdefault((t, by), []).append(f"{EVENT_GLYPHS['put back']}O{obj + 1}")
            else:
                out.setdefault((t, by), []).append(f"{EVENT_GLYPHS[what]}O{obj + 1}")
        return out

    def banner(self, f: int) -> str:
        """What the transition into frame f did to the objects, in one line."""
        events = sorted(self.frames[f]["arrived_by"], key=lambda e: e[1])
        return "     ".join(dict.fromkeys(
            event_text(e, self.names, forced=self.forced(f, e), why=self.why(f, e))
            for e in events))

    # -- the map ---------------------------------------------------------------------------------
    def callouts(self, f: int, linger: int) -> list:
        """[(age in frames, cell, events)]: the events of the last ``linger`` transitions up
        to frame f, those of one cell and transition together, the newest first.  (The
        transition of a callout is the frame's step minus its age.)"""
        step = self.frames[f]["step"]
        out: dict = {}
        for t, event in self.events:
            age = step - t
            if 0 <= age < linger:
                cell = tuple(self.frames[t]["pos"][event[1]])     # where its robot stood
                out.setdefault((age, cell), []).append(event)
        return [(age, cell, events) for (age, cell), events in sorted(out.items())]

    # -- frame timing ----------------------------------------------------------------------------
    def weight(self, f: int) -> str:
        """"end", "placed", "event" or "plain": how long frame f is held."""
        if f == len(self.frames) - 1:
            return "end"
        arrived = [e[0] for e in self.frames[f]["arrived_by"]]
        if "set down" in arrived:
            return "placed"
        return "event" if arrived else "plain"


def goal_label(goal, competition: Optional[str] = None, cooperate_with=None,
               conflict: Optional[str] = None) -> str:
    """A social goal as a headline names it.  A competitor is named with the reading of
    competition it was run under (``Config.competition``), where that is known: the
    readings behave differently, and a figure should say which one it shows.  So is a
    cooperator that does not cooperate with every partner (``Config.cooperate_with``, where
    known; a record without it was made with Table I's): "cooperation (selective)"; and a
    conflict robot by its reading (``Config.conflict``, where known; a record without it
    was made with Table I's): "conflict (task)", "conflict (Table I)"."""
    if goal in (None, "none"):
        return "physical goal only"
    reading = COMPETITION_NAMES.get(competition)
    if goal == "competition" and reading:
        return f"competition ({reading})"
    if goal == "conflict" and conflict in CONFLICT_NAMES:
        return f"conflict ({CONFLICT_NAMES[conflict]})"
    if goal == "cooperation" and cooperate_with is not None:
        rule = cooperation_reading(cooperate_with)
        if rule != "Table I":
            return f"cooperation ({rule})"
    return str(goal)


def _title(result: dict, names: Sequence[str]) -> str:
    goals = result.get("goals") or result.get("social_goals") or ("none", "none")
    config = result.get("config") or {}
    label = [goal_label(g, config.get("competition"), config.get("cooperate_with"),
                        config.get("conflict", "table") if config else None)
             for g in goals]
    weights = result.get("weights")
    tail = "" if weights is None or all(g in (None, "none") for g in goals) else (
        f"     weights {weights[0]:g} / {weights[1]:g}")
    return f"{names[0]}: {label[0]}     {names[1]}: {label[1]}{tail}"


# ---- drawing -----------------------------------------------------------------------------------

class _Canvas:
    """The figure and its five panels; ``draw(f)`` paints frame f."""

    def __init__(self, result: dict, names: Sequence[str], values: str, title: str,
                 trail: int, linger: int, font: Optional[str] = FONT) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        if values not in ("relative", "raw"):
            raise ValueError("values must be 'relative' or 'raw'")
        self.plt = plt
        self.style = {} if font is None else {"font.family": [font]}
        self.family = {} if font is None else {"fontfamily": font}
        self.result, self.names, self.values = result, tuple(names), values
        self.story = _Story(result, names)
        self.frames = self.story.frames
        self.title = title or _title(result, names)
        self.trail, self.linger = int(trail), max(1, int(linger))
        scene = result["scene"]
        self.walls = np.asarray(scene["grid"]) == 0                        # (X, Y) bool
        self.res = float(scene["resolution"])
        nx, ny = self.walls.shape
        self.extent = [0.0, nx * self.res, 0.0, ny * self.res]
        self.K = len(self.story.candidates)
        self.object_colour = [ROBOT_COLOURS[k] if k < 2 else SPARE_COLOUR
                              for k in range(len(self.story.homes))]
        self._value_limits()
        with plt.rc_context(self.style):
            self.fig = plt.figure(figsize=(14.6, 9.3))
            spec = GridSpec(4, 2, figure=self.fig, width_ratios=[1.30, 1.0],
                            height_ratios=[1.86, 0.74, 0.88, 1.02], hspace=0.74, wspace=0.11,
                            left=0.042, right=0.985, top=0.908, bottom=0.08)
            self.ax_map = self.fig.add_subplot(spec[0:3, 0])
            self.ax_status = self.fig.add_subplot(spec[0, 1])
            self.ax_social = None
            # where a robot has to infer which object its partner wants, the objects and the
            # destinations share one panel: O1..OM, a gap, G0..G(K-1)
            M = self.story.n_objects if self.story.objects_inferred else -1
            self.belief_x = (np.arange(M), M + 1 + np.arange(self.K))     # objects, destinations
            if self.story.inferred:              # two beliefs side by side
                # as wide as its bars, but never so wide that the social goals' tick labels
                # run into each other
                ratio = 1.0 if M < 0 else min((M + 1 + self.K) / (self.K + 0.6), 1.45)
                halves = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec[1, 1], wspace=0.13,
                                                 width_ratios=[ratio, 1.3])
                self.ax_belief = self.fig.add_subplot(halves[0, 0])
                self.ax_social = self.fig.add_subplot(halves[0, 1])
            else:
                self.ax_belief = self.fig.add_subplot(spec[1, 1])
            self.ax_value = self.fig.add_subplot(spec[2, 1])
            self.ax_time = self.fig.add_subplot(spec[3, :])
        self._room()

    def close(self) -> None:
        self.plt.close(self.fig)

    # -- text that has to fit: measured in the font it is drawn in -------------------------------
    def _extent(self, text: str, **font) -> Tuple[float, float]:
        """Width and height in points of ``text`` as the figure draws it."""
        probe = self.fig.text(0.0, 0.0, text, **self.family, **font)
        box = probe.get_window_extent(self.fig.canvas.get_renderer())
        probe.remove()
        return box.width * 72.0 / self.fig.dpi, box.height * 72.0 / self.fig.dpi

    def _fit(self, text: str, width: float, lines: int = 2, **font) -> str:
        """``text`` on at most ``lines`` lines no wider than ``width`` points; what does not
        fit is cut, with an ellipsis."""
        n = max(2, int(max(len(text), 1) * width / max(self._extent(text, **font)[0], 1e-9)))
        while n > 2 and self._extent(_wrap(text, n, lines), **font)[0] > width:
            n -= 1
        return _wrap(text, n, lines)

    def _squeezed(self, text: str, width: float, size: float, smallest: float = 6.5,
                  **font) -> Tuple[str, float]:
        """``text`` and the font size, at most ``size``, at which it is no wider than
        ``width`` points on one line; at ``smallest`` what still does not fit is cut."""
        wide = self._extent(text, fontsize=size, **font)[0]
        if wide > width:                         # a little under: glyphs do not scale evenly
            size = max(smallest, 0.985 * size * width / wide)
            if self._extent(text, fontsize=size, **font)[0] > width + 0.5:
                text = self._fit(text, width, 1, fontsize=size, **font)
        return text, size

    def _room(self) -> None:
        """The limits of the map panel, ``view`` = [x0, x1, y0, y1] in metres: the map and,
        on each side on which its border is not all wall, room for what is written there
        (the legend along the top, the labels of destinations in the border cells).  On a
        wall the same things are written on the wall.  The room is a size in points, so
        the scale comes with it: ``per_pt`` metres per point."""
        walls, (x0, x1, y0, y1) = self.walls, self.extent
        room = {side: 0.0 if edge.all() else _ROOM[side]
                for side, edge in (("left", walls[0]), ("right", walls[-1]),
                                   ("bottom", walls[:, 0]), ("top", walls[:, -1]))}
        box = self.ax_map.get_position(original=True)
        width = box.width * self.fig.get_figwidth() * 72.0                 # points
        height = box.height * self.fig.get_figheight() * 72.0
        self.per_pt = max((x1 - x0) / (width - room["left"] - room["right"]),
                          (y1 - y0) / (height - room["bottom"] - room["top"]))
        self.view = [x0 - room["left"] * self.per_pt, x1 + room["right"] * self.per_pt,
                     y0 - room["bottom"] * self.per_pt, y1 + room["top"] * self.per_pt]
        self.framed = any(room.values())
        # a callout may also use what the panel has left beside a narrow map
        spare = 0.5 * max(width * self.per_pt - (self.view[1] - self.view[0]), 0.0)
        self.reach = [self.view[0] - spare, self.view[1] + spare, self.view[2], self.view[3]]

    def centre(self, cell) -> Tuple[float, float]:
        return ((cell[0] + 0.5) * self.res, (cell[1] + 0.5) * self.res)

    def _free(self, cell) -> bool:
        """Is ``cell`` on the map and not a wall?"""
        nx, ny = self.walls.shape
        return 0 <= cell[0] < nx and 0 <= cell[1] < ny and not self.walls[cell[0], cell[1]]

    # -- values: the bars and one set of limits for the whole animation --------------------------
    def bars(self, frame: dict, i: int):
        """(physical, social, whole) per action for robot i in this frame; None on the last
        frame.  An act for the partner (give; place where it puts the partner's object down
        or sets it down for the partner) is one that only a robot with a social goal has:
        the physical model has no value for it, so there is nothing to split.  ``whole``
        holds Q(a) - Q(a*) for those acts (nan for the other actions), or None where the
        robot has none here; where the action taken is one, ``whole`` holds Q(a) - Q(a*) for
        every action and the two parts are None."""
        if frame["q"] is None:
            return None
        q, qp = frame["q"][i], frame["q_physical"][i]
        if self.values == "raw":
            return qp, frame["social_reward"][i], None
        a = frame["action"][i]
        if not np.isfinite(qp[a]):
            return None, None, q - q[a]
        social = q - qp
        only = np.isfinite(q) & ~np.isfinite(qp)               # acts for the partner
        whole = np.where(only, q - q[a], np.nan) if only.any() else None
        return qp - qp[a], social - social[a], whole

    def _value_limits(self) -> None:
        stacked = [v for f in self.frames for i in (0, 1)
                   if (b := self.bars(f, i)) is not None for v in b if v is not None]
        finite = np.concatenate(stacked) if stacked else np.zeros(1)
        finite = finite[np.isfinite(finite)]
        lo, hi = (np.percentile(finite, [2, 98]) if finite.size else (-1.0, 1.0))
        lo, hi = min(lo, -1e-6), max(hi, 1e-6)
        # With large weights a few values dwarf the rest: a symmetric log axis (linear within
        # +-1) keeps the small differences that decide an action readable next to them.
        self.wide = self.values == "relative" and max(-lo, hi) > 20.0
        if self.wide:
            lo, hi = min(float(finite.min()), -1.0), max(float(finite.max()), 1.0)
            self.limits = (1.6 * lo, 1.6 * hi)
        else:
            pad = 0.08 * (hi - lo)
            self.limits = (lo - pad, hi + pad)

    # -- one frame -------------------------------------------------------------------------------
    def draw(self, f: int) -> None:
        with self.plt.rc_context(self.style):
            self._draw(f)

    def _draw(self, f: int) -> None:
        for ax in (self.ax_map, self.ax_status, self.ax_belief, self.ax_social, self.ax_value,
                   self.ax_time):
            if ax is not None:
                ax.clear()
        for text in list(self.fig.texts):
            text.remove()
        self._map(f)
        self._status(f)
        self._beliefs(f)
        self._values(f)
        self._timeline(f)
        frame = self.frames[f]
        across = self.fig.get_figwidth() * 72.0                          # points
        left, right = 0.042, 0.985
        title, size = self._squeezed(self.title, (right - left) * across, 12.5,
                                     fontweight="bold")
        self.fig.text(left, 0.974, title, ha="left", va="center", color=INK, fontsize=size,
                      fontweight="bold")
        head = f"step {frame['step']} of {self.story.n_steps}"
        if frame["action"] is None:
            head += "   —   rollout ended: " + self._ending()
        self.fig.text(left, 0.941, head, ha="left", va="center", fontsize=11.5, color=INK,
                      fontweight="bold")
        banner = self.story.banner(f)
        if banner:                               # what the last transition did, in words
            x = left + self._extent(head, fontsize=11.5, fontweight="bold")[0] / across + 0.015
            banner, size = self._squeezed(banner, (right - x) * across - 8.0, 10.5,
                                          fontweight="bold")
            self.fig.text(x, 0.941, banner, ha="left", va="center", fontsize=size, color=INK,
                          fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.3", facecolor=FRESH, edgecolor=INK,
                                    linewidth=1.0))

    def _ending(self) -> str:
        stopped = self.result.get("stopped_by", "")
        if stopped == "done":
            return "both goals are done"
        if stopped == "deadlock":
            return "no progress any more (deadlock)"
        return "the step limit was reached" if stopped == "cap" else str(stopped)

    # -- map -------------------------------------------------------------------------------------
    def _map(self, f: int) -> None:
        from matplotlib.legend_handler import HandlerPatch
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle, Patch, Rectangle
        from matplotlib.ticker import MaxNLocator

        ax, story, frame, res = self.ax_map, self.story, self.frames[f], self.res
        ax.imshow(np.ma.masked_where(~self.walls, self.walls).T, origin="lower",
                  extent=self.extent, cmap="gray_r", vmin=0, vmax=1.3, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlim(self.view[:2])
        ax.set_ylim(self.view[2:])
        x0, x1, y0, y1 = self.extent
        for axis, lo, hi in ((ax.xaxis, x0, x1), (ax.yaxis, y0, y1)):
            # ticks on the map only, at least 40 points apart
            axis.set_major_locator(MaxNLocator(
                nbins=max(1, int((hi - lo) / self.per_pt / 40.0)), steps=[1, 2, 2.5, 5, 10]))
            axis.set_ticks([t for t in axis.get_ticklocs() if lo - 1e-9 <= t <= hi + 1e-9])
        if self.framed:                          # where the map ends
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="0.3",
                                   linewidth=1.0, zorder=1))
        label_box = dict(facecolor="white", edgecolor="none", alpha=.85, pad=.3)
        robot_cells = [tuple(p) for p in frame["pos"]]
        delivered = {tuple(frame["object_cells"][k]): k for k in range(len(story.homes))
                     if frame["status"][k] == GONE}
        self.marks = []                          # what a callout should not cover
        label_side = {}                          # cell -> +1 / -1: its label is above / below
        if story.rooms is not None:              # each robot's own room, tinted its colour
            for i in (0, 1):
                for cell in story.rooms[i] - story.rooms[1 - i]:
                    ax.add_patch(Rectangle((cell[0] * res, cell[1] * res), res, res,
                                           facecolor=ROBOT_COLOURS[i], edgecolor="none",
                                           alpha=ROOM_TINT, zorder=0.5))
        for cell in story.handoff:               # hand-off cells: a dashed frame and a label
            ax.add_patch(Rectangle(((cell[0] + 0.07) * res, (cell[1] + 0.07) * res),
                                   0.86 * res, 0.86 * res, facecolor="none", edgecolor=INK,
                                   linewidth=1.5, linestyle=(0, (3, 2)), zorder=3))
            up = self._free((cell[0], cell[1] - 1)) and not self._free((cell[0], cell[1] + 1))
            label_side[cell] = 1 if up else -1
            away = 0.5 * res / self.per_pt + 3.0     # outside the cell: a robot stands in it
            ax.annotate("hand-off cell", self.centre(cell), xytext=(0, away if up else -away),
                        textcoords="offset points", ha="center", va="bottom" if up else "top",
                        color=INK, fontsize=7.5, zorder=8, bbox=label_box)
            x, y = self.centre(cell)
            wide, high = self._extent("hand-off cell", fontsize=7.5)
            at = y + (1 if up else -1) * (away + 0.5 * high) * self.per_pt
            self.marks += [(x + shift * wide * self.per_pt, at) for shift in (-0.4, 0.0, 0.4)]
        for k, cell in enumerate(story.candidates):               # candidate destinations
            owner = story.destinations.index(cell) if cell in story.destinations else None
            colour = ROBOT_COLOURS[owner] if owner is not None else SPARE_COLOUR
            reached = owner is not None and bool(frame["done"][owner])
            ax.plot(*self.centre(cell), "*", ms=24 if owner is not None else 15, mec=colour,
                    mew=1.6, mfc=colour if reached else "none", zorder=4)
            text = f"G{k}" if owner is None else f"G{k}: {self.names[owner]}'s goal"
            if reached:                          # ... and what lies on it now
                text += (f"\nO{delivered[cell] + 1} delivered {CHECK}" if cell in delivered
                         else f" {CHECK}")
            # its label goes below it -- above where a free cell lies below and none above
            up = self._free((cell[0], cell[1] - 1)) and not self._free((cell[0], cell[1] + 1))
            label_side[cell] = 1 if up else -1
            ax.annotate(text, self.centre(cell), xytext=(0, 20 if up else -20),
                        textcoords="offset points", ha="center",
                        va="bottom" if up else "top", color=INK, fontsize=7.5,
                        fontweight="bold" if reached else "normal", linespacing=1.15,
                        zorder=13 if reached else 8, bbox=label_box)
            x, y = self.centre(cell)
            self.marks.append((x, y))            # the star, and its label from end to end
            wide, high = self._extent(text, fontsize=7.5, linespacing=1.15,
                                      fontweight="bold" if reached else "normal")
            at = y + (1 if up else -1) * (20.0 + 0.5 * high) * self.per_pt
            self.marks += [(x + shift * wide * self.per_pt, at) for shift in (-0.4, 0.0, 0.4)]
        for k, cell in enumerate(story.homes):                     # where each object belongs
            ax.plot(*self.centre(cell), "s", ms=17, mec=self.object_colour[k], mew=1.1,
                    mfc="none", alpha=.75, zorder=3)
        lying: dict = {}                         # cell -> the objects lying on it
        for k, cell in enumerate(frame["object_cells"]):
            if frame["status"][k] not in (HELD0, HELD1, GONE):
                lying.setdefault(tuple(cell), []).append(k)
        for k, cell in enumerate(frame["object_cells"]):           # where each object is
            xy = np.array(self.centre(cell), dtype=float)
            status = frame["status"][k]
            held = status in (HELD0, HELD1)
            lifter = story.lifter(status)
            edge, name, offset = "k", f"O{k + 1}", (10, -10)
            shared = lying.get(tuple(cell), [k])
            if held:                                               # carried: beside its holder
                edge = ROBOT_COLOURS[status - 1]
                ax.plot([xy[0], xy[0] + 0.33 * res], [xy[1], xy[1] + 0.33 * res], "-",
                        color=edge, lw=2.4, zorder=4)
                xy += 0.33 * res
                offset = (10, 4)
            elif lifter >= 0:                                      # being lifted: beside its
                edge = ROBOT_COLOURS[lifter]                       # lifter, a little raised,
                stage = story.lift_stage(status)                   # with how far the lift is
                xy += np.array((0.33, 0.12 + 0.1 * stage)) * res
                ax.plot([xy[0], xy[0]], [xy[1] - 0.1 * stage * res, xy[1]], "-", color=edge,
                        lw=2.0, zorder=4)
                name = f"O{k + 1} {stage}/{story.lift_steps}"
                offset = (10, 4)
            elif tuple(cell) in robot_cells or len(shared) > 1:    # lying under a robot, or
                side = -label_side.get(tuple(cell), 1)             # with another object:
                if tuple(cell) in story.handoff:                   # beside it, clear of the
                    side = -side                                   # label of its cell (a
                across = 0.31 if shared.index(k) % 2 else -0.31    # hand-off cell's is
                xy += np.array((across, 0.31 * side)) * res        # outside the cell)
                offset = (10 if across > 0 else -10, 10 * side)
            ax.plot(*xy, "s", ms=12.5, color=self.object_colour[k], mec=edge,
                    mew=2.4 if held else 1.6 if lifter >= 0 else 0.9,
                    zorder=10 if held or lifter >= 0 else 7)
            self.marks.append(tuple(xy))
            if status == GONE:                   # delivered: its destination's label says so
                ax.plot(*xy, marker="$✓$", ms=8, color="white", zorder=8)
                continue
            ax.annotate(name, xy, xytext=offset, textcoords="offset points", color=INK,
                        fontsize=7.5, zorder=8, va="center", bbox=label_box,
                        ha=("right", "center", "left")[int(np.sign(offset[0])) + 1])
        for i in (0, 1):                                           # the robots
            past = np.array([self.centre(g["pos"][i])
                             for g in self.frames[max(0, f - self.trail): f + 1]])
            ax.plot(past[:, 0], past[:, 1], "-", color=ROBOT_COLOURS[i], lw=1.8, alpha=0.5,
                    zorder=2)
            x, y = past[-1]
            self.marks += [(x, y)] * 3           # a robot weighs more than a label
            ax.add_patch(Circle((x, y), 0.30 * res, facecolor=ROBOT_COLOURS[i], edgecolor="k",
                                linewidth=1.0, zorder=5))
            ax.text(x, y, str(i + 1), ha="center", va="center", color="white", fontsize=9.5,
                    fontweight="bold", zorder=6)
            if frame["action"] is None:
                continue
            a = frame["action"][i]
            if a < WAIT:
                dx, dy = MOVES[MOVE_OF[a]]
                colour = "0.55" if frame["blocked"][i] else ROBOT_COLOURS[i]
                ax.annotate("", xy=(x + dx * res * 0.88, y + dy * res * 0.88),
                            xytext=(x + dx * res * 0.30, y + dy * res * 0.30),
                            arrowprops=dict(arrowstyle="-|>", color=colour, lw=2.6,
                                            mutation_scale=17), zorder=6)
                self.marks.append((x + dx * res * 0.65, y + dy * res * 0.65))
                continue
            if a >= PICK:
                ax.add_patch(Circle((x, y), 0.42 * res, facecolor="none",
                                    edgecolor=ROBOT_COLOURS[i], linewidth=2.0,
                                    linestyle=(0, (2, 1.5)), zorder=5))
            cell = tuple(frame["pos"][i])
            if a == WAIT and frame["done"][i]:   # a finished robot stays: no tag
                continue
            # the tag goes below, left, right or above: clear of the other robot, of what the
            # robot carries (upper right), of the label of its cell or of the next one, and of
            # a star or an object's place beside it
            other = tuple(frame["pos"][1 - i])
            taken = {(other[0] - cell[0], other[1] - cell[1])}
            for dx in (-1, 0, 1):
                near = (cell[0] + dx, cell[1])
                if near in label_side:
                    taken.add((0, label_side[near]))
                if dx and (near in story.candidates or near in story.homes):
                    taken.add((dx, 0))
            if frame["holds"][i] >= 0:
                taken.add((0, 1))
            side = next((d for d in ((0, -1), (-1, 0), (1, 0), (0, 1)) if d not in taken),
                        (0, -1))
            self.marks.append((x + 40 * side[0] * self.per_pt, y + 21 * side[1] * self.per_pt))
            ax.annotate(ACTION_WORDS[a] if a >= PICK else "stay", (x, y),
                        xytext=(27 * side[0], 21 * side[1]), textcoords="offset points",
                        ha=("right", "center", "left")[side[0] + 1], va="center",
                        fontsize=7.5, color=INK, zorder=10,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                  edgecolor=ROBOT_COLOURS[i], linewidth=1.3))
        self._callouts(f)
        ax.set_xlabel("world x (m)", fontsize=8.5)
        ax.set_ylabel("world y (m)", fontsize=8.5)
        ax.tick_params(labelsize=8)
        handles = [Line2D([], [], marker="o", color=ROBOT_COLOURS[i], linestyle="none",
                          markersize=9, markeredgecolor="k", markeredgewidth=.8,
                          label=self.names[i]) for i in (0, 1)]
        objects = "object (O1, O2: the robots' own)"
        handles += [Line2D([], [], marker="s", color=SPARE_COLOUR, linestyle="none",
                           markersize=8, markeredgecolor="k", markeredgewidth=.8,
                           label=objects),
                    Line2D([], [], marker="s", color=SPARE_COLOUR, linestyle="none",
                           markersize=9, markerfacecolor="none", markeredgewidth=1.0,
                           label="an object's place"),
                    Line2D([], [], marker="*", color=SPARE_COLOUR, linestyle="none",
                           markersize=11, markerfacecolor="none", markeredgewidth=1.2,
                           label="candidate destination")]
        extra = []                               # a second row: what the scene adds
        if story.rooms is not None:
            extra += [Patch(facecolor=ROBOT_COLOURS[i], alpha=2.0 * ROOM_TINT, edgecolor="none",
                            label=f"{self.names[i]}'s room") for i in (0, 1)]
        if story.handoff:
            extra.append(Patch(facecolor="none", edgecolor=INK, linewidth=1.2,
                               linestyle=(0, (3, 2)), label="hand-off cell"))
        # along the top of the panel: on the top wall, or in the room made above the map
        # (two rows, filled column by column, where the scene has rooms or hand-off cells)

        def swatch(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
            """A patch fills its handle box where a marker sits in the middle of it: drawn
            narrower, it keeps the same distance from its label."""
            return Rectangle((-xdescent, -ydescent), 0.78 * width, height)

        ax.legend(handles=handles + extra, fontsize=7.5, ncols=4 if extra else 5, frameon=True,
                  facecolor="white", edgecolor="0.6", framealpha=0.95, columnspacing=1.1,
                  handletextpad=0.3, borderpad=0.5, loc="upper center",
                  bbox_to_anchor=(0.5, 0.995),
                  handler_map={Patch: HandlerPatch(patch_func=swatch)})

    def _callouts(self, f: int) -> None:
        """The events of the last few transitions, each called out at its cell."""
        from matplotlib.patches import Circle

        ax, res, per_pt, view, bounds = self.ax_map, self.res, self.per_pt, self.view, self.reach
        step = self.frames[f]["step"]
        # taken already (centre x, y, half width, half height): the legend along the top,
        # and the tick labels and axis label left of the panel
        legend = 26 if (self.story.rooms is not None or self.story.handoff) else 17
        boxes = [(0.5 * (view[0] + view[1]), view[3] - legend * per_pt,
                  max(0.5 * (view[1] - view[0]), 225 * per_pt), legend * per_pt),
                 (view[0] - 26 * per_pt, 0.5 * (view[2] + view[3]), 26 * per_pt,
                  0.5 * (view[3] - view[2]))]
        for age, cell, events in self.story.callouts(f, self.linger):
            x, y = self.centre(cell)
            fresh = age == 0
            placed = any(e[0] == "set down" for e in events)
            colour = ROBOT_COLOURS[events[0][1]]
            burst = placed and fresh                       # a delivery: three rings
            for n, radius in enumerate((0.50, 0.66, 0.82) if burst else (0.50,)):
                ax.add_patch(Circle((x, y), radius * res, facecolor="none", edgecolor=colour,
                                    linewidth=2.8 - 0.7 * n, zorder=9,
                                    alpha=(1.0 if fresh else 0.4) * (1.0 - 0.25 * n)))
            away = 1.4 if burst else 1.0                   # ... and the text clear of them
            text = _callout(events, self.names,
                            [self.story.forced(step - age, e) for e in events],
                            [self.story.why(step - age, e) for e in events])
            size = 9.5 if fresh else 8.0
            wide, high = self._extent(text, fontsize=size, linespacing=1.25,
                                      fontweight="bold" if fresh else "normal")
            half_w = 0.5 * (wide + 0.8 * size + 4.0) * per_pt     # the text, the pad of its
            half_h = 0.5 * (high + 0.8 * size + 4.0) * per_pt     # box, and a little air

            def cost(offset) -> float:
                cx, cy = x + away * offset[0] * per_pt, y + away * offset[1] * per_pt
                outside = (max(bounds[0] - (cx - half_w), 0) + max(cx + half_w - bounds[1], 0)
                           + max(bounds[2] - (cy - half_h), 0)
                           + max(cy + half_h - bounds[3], 0))
                covered = sum(abs(mx - cx) < half_w + 0.2 * res and abs(my - cy)
                              < half_h + 0.2 * res for mx, my in self.marks)
                clash = sum(abs(bx - cx) < half_w + bw and abs(by - cy) < half_h + bh
                            for bx, by, bw, bh in boxes)
                return 100.0 * outside / res + covered + 6.0 * clash

            offset = min(_OFFSETS, key=lambda o: (cost(o), _OFFSETS.index(o)))
            offset = (away * offset[0], away * offset[1])
            boxes.append((x + offset[0] * per_pt, y + offset[1] * per_pt, half_w, half_h))
            edge = INK if fresh else "0.5"                 # older: plainer, still readable
            ax.annotate(
                text, (x, y), xytext=offset, textcoords="offset points", ha="center",
                va="center", fontsize=size, fontweight="bold" if fresh else "normal",
                color=INK if fresh else MUTED, zorder=12 if fresh else 11, linespacing=1.25,
                bbox=dict(boxstyle="round,pad=0.4", facecolor=FRESH if fresh else "white",
                          edgecolor=edge, linewidth=1.5 if fresh else 0.8),
                arrowprops=dict(arrowstyle="-|>", color=edge, lw=1.5 if fresh else 0.8,
                                shrinkB=10))

    # -- status table ----------------------------------------------------------------------------
    def _status(self, f: int) -> None:
        from matplotlib.patches import FancyBboxPatch

        ax, story = self.ax_status, self.story
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, facecolor="0.975", edgecolor="0.7",
                                    boxstyle="round,pad=0,rounding_size=0.015",
                                    linewidth=0.8, clip_on=False, zorder=0))
        columns = (0.150, 0.585)                 # x of the two columns
        panel = ax.get_position().width * self.fig.get_figwidth() * 72.0       # points
        room = [(columns[1] - columns[0] - 0.02) * panel, (1.0 - columns[1] - 0.015) * panel]
        for i, x in enumerate(columns):
            ax.plot([x + 0.016], [0.930], "o", ms=14, color=ROBOT_COLOURS[i], mec="k", mew=0.9,
                    zorder=3)
            ax.text(x + 0.016, 0.928, str(i + 1), ha="center", va="center", color="white",
                    fontsize=8.5, fontweight="bold", zorder=4)
            name, size = self._squeezed(self.names[i], room[i] - 0.045 * panel, 11.5,
                                        fontweight="bold")
            ax.text(x + 0.045, 0.930, name, ha="left", va="center", color=INK, fontsize=size,
                    fontweight="bold")
            subtitle, size = self._squeezed(story.subtitle[i], room[i], 8.3)
            ax.text(x, 0.836, subtitle, ha="left", va="center", color=MUTED, fontsize=size)
        ax.plot([0.02, 0.98], [0.795, 0.795], "-", color="0.78", lw=0.8)
        rows = (("action\nin this step", 0.765,
                 [self._fit(story.action(f, i), room[i], fontsize=9, fontweight="bold")
                  for i in (0, 1)]),
                ("hands", 0.570, [self._fit(story.hands(f, i), room[i], 1, fontsize=9)
                                  for i in (0, 1)]),
                ("last event", 0.462, [self._fit(story.last_event(f, i), room[i], fontsize=9)
                                       for i in (0, 1)]),
                ("goal", 0.262, None))
        for label, y, cells in rows:
            ax.text(0.02, y, label, ha="left", va="top", fontsize=8.3, color=MUTED,
                    linespacing=1.15)
            if cells is None:
                continue
            for x, text in zip(columns, cells):
                ax.text(x, y, text, ha="left", va="top", fontsize=9, color=INK,
                        linespacing=1.18, fontweight="bold" if label.startswith("action")
                        else "normal")
        y = rows[-1][1]
        for i, x in enumerate(columns):
            done, badge, task, detail = story.goal(f, i)
            ax.text(x + 0.004, y - 0.004, badge, ha="left", va="top", fontsize=9,
                    fontweight="bold", color="white" if done else INK,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=DONE_FILL if done
                              else OPEN_FILL, edgecolor="none"))
            # the task right of the badge: the badge's text, the pad of its box and a gap
            taken = self._extent(badge, fontsize=9, fontweight="bold")[0] + 4.5 + 6.0
            task, size = self._squeezed(task, room[i] - taken - 0.004 * panel, 9)
            ax.text(x + 0.004 + taken / panel, y - 0.004, task, ha="left", va="top", color=INK,
                    fontsize=size)
            detail, size = self._squeezed(detail, room[i], 8.6)
            ax.text(x, y - 0.118, detail, ha="left", va="top", color=INK, fontsize=size)

    # -- beliefs: robot i about its partner's destination (and object) --------------------------
    def _beliefs(self, f: int) -> None:
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch

        ax, frame, K, width = self.ax_belief, self.frames[f], self.K, 0.38
        objects = self.story.objects_inferred
        xo, xd = self.belief_x
        leads = [[], []]                         # per robot: its leading object, destination
        for i in (0, 1):
            belief = np.asarray(frame["belief"][i], dtype=float)
            edges = ["k" if k == self.story.goal_ids[1 - i] else "none" for k in range(K)]
            ax.bar(xd + (i - 0.5) * width, belief, width=width, color=ROBOT_COLOURS[i],
                   edgecolor=edges, linewidth=1.4)
            mine = (frame.get("object_belief") or (None, None))[i] if objects else None
            if mine is not None:                 # the partner's own object outlined
                mine = np.asarray(mine, dtype=float)
                edges = ["k" if q == 1 - i else "none" for q in range(mine.size)]
                ax.bar(xo + (i - 0.5) * width, mine, width=width, color=ROBOT_COLOURS[i],
                       edgecolor=edges, linewidth=1.4)
                leads[i].append(self._lead(ax, i, xo + (i - 0.5) * width, mine,
                                           [f"O{q + 1}" for q in range(mine.size)]))
            if belief.size:
                leads[i].append(self._lead(ax, i, xd + (i - 0.5) * width, belief,
                                           [f"G{k}" for k in range(K)]))
        self._lead_text(ax, leads)
        if objects:                              # uniform over each, and the gap between
            ax.hlines(1.0 / xo.size, xo[0] - 0.5, xo[-1] + 0.5, linestyles=":", colors="0.35",
                      lw=1.2)
            ax.hlines(1.0 / K, xd[0] - 0.5, xd[-1] + 0.5, linestyles=":", colors="0.35", lw=1.2)
            ax.axvline(xd[0] - 1.0, color="0.72", lw=0.8)
            ax.set_xlim(xo[0] - 0.7, xd[-1] + 0.7)
        else:
            ax.axhline(1.0 / K, ls=":", c="0.35", lw=1.2)
        ax.set_ylim(0, LEAD_TOP)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.axhline(1.0, c="0.82", lw=0.7, zorder=0)
        ticks = [f"O{q + 1}" for q in range(xo.size)] + [f"G{k}" for k in range(K)]
        ax.set_xticks(np.concatenate([xo, xd]))
        ax.set_xticklabels(ticks, fontsize=8 if len(ticks) <= 11 else 7)
        ax.tick_params(axis="y", labelsize=8)
        both = self.ax_social is not None        # one legend under the two panels
        title = ("belief about the partner's object and destination" if objects
                 else "belief about the partner's destination")
        panel = ax.get_position().width * self.fig.get_figwidth() * 72.0             # points
        title, size = self._squeezed(title, panel, SPLIT_TITLE if both or objects else 9)
        ax.set_title(title, color=INK, loc="left", fontsize=size, pad=LEAD_PAD)
        handles = [Patch(facecolor=ROBOT_COLOURS[i], label=f"{self.names[i]}'s belief")
                   for i in (0, 1)]
        handles += [Patch(facecolor="white", edgecolor="k", linewidth=1.4,
                          label="the true one" if both or objects else "the true destination"),
                    Line2D([], [], ls=":", c="0.35", lw=1.2, label="uniform")]
        # centred under the panels: their middle, in this panel's axes fraction
        last = (self.ax_social if both else ax).get_position()
        mine = ax.get_position()
        centre = (0.5 * (mine.x0 + last.x1) - mine.x0) / mine.width
        ax.legend(handles=handles, fontsize=7.2, ncols=4, loc="upper center",
                  bbox_to_anchor=(centre, -0.17), frameon=False,
                  columnspacing=1.0, handlelength=1.4)
        if both:
            self._social_beliefs(frame)

    def _social_beliefs(self, frame: dict) -> None:
        """Each robot's belief about its partner's SOCIAL goal, over the goals it holds
        possible; the true one is outlined where it is among them."""
        ax, hyp, width = self.ax_social, self.story.hypotheses, 0.38
        names = [SOCIAL_GOALS[h] for h in hyp]
        ticks = GOAL_TICKS if len(names) <= 5 else GOAL_TICKS_SHORT
        labels = [ticks.get(name, name) for name in names]
        beliefs = frame.get("social_belief") or (None, None)
        leads = [[], []]
        for i in (0, 1):
            if beliefs[i] is None:               # a robot below level 2 holds no such belief
                continue
            belief = np.asarray(beliefs[i], dtype=float)[hyp]
            x = np.arange(len(hyp)) + (i - 0.5) * width
            edges = ["k" if name == self.story.true_social[1 - i] else "none"
                     for name in names]
            ax.bar(x, belief, width=width, color=ROBOT_COLOURS[i], edgecolor=edges,
                   linewidth=1.4)
            leads[i].append(self._lead(ax, i, x, belief, labels))
        self._lead_text(ax, leads)
        ax.axhline(1.0 / max(len(hyp), 1), ls=":", c="0.35", lw=1.2)
        ax.set_ylim(0, LEAD_TOP)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.axhline(1.0, c="0.82", lw=0.7, zorder=0)
        ax.set_xticks(range(len(hyp)))
        ax.set_xticklabels(labels, fontsize=7.4)
        ax.tick_params(axis="y", labelsize=8, labelleft=False)
        ax.set_title("... and about its social goal", fontsize=SPLIT_TITLE, color=INK,
                     loc="left", pad=LEAD_PAD)

    def _lead(self, ax, i: int, x: np.ndarray, belief: np.ndarray, labels) -> str:
        """Robot i's leading belief in one group of bars (the most probable, with any within
        LEAD_TIE of it): a marker over each such bar, in robot i's lead colour; -> its text,
        e.g. "O2 0.97" or "comp./hinder 0.45"."""
        if not belief.size or not np.isfinite(belief).any():
            return ""
        top = float(np.nanmax(belief))
        best = np.flatnonzero(belief >= top - LEAD_TIE)
        ax.plot(x[best], np.minimum(belief[best], 1.0) + 0.07, ls="none", marker="v",
                markersize=5.2, color=LEAD_COLOURS[i], markeredgecolor="white",
                markeredgewidth=0.6, zorder=5, clip_on=False)
        return "/".join(labels[b] for b in best) + f" {top:.2f}"

    def _lead_text(self, ax, leads) -> None:
        """Each robot's leading beliefs in this panel, one line each, in bold: agent_1's in
        green, agent_2's in purple."""
        for i in (0, 1):
            parts = [p for p in leads[i] if p]
            if not parts:
                continue
            ax.annotate(f"{self.names[i]}: " + "  ·  ".join(parts), xy=(0.0, 1.0),
                        xycoords="axes fraction", xytext=(1.0, LEAD_ROWS[i]),
                        textcoords="offset points", ha="left", va="bottom", fontsize=7.4,
                        fontweight="bold", color=LEAD_COLOURS[i], annotation_clip=False)

    # -- values ----------------------------------------------------------------------------------
    def _values(self, f: int) -> None:
        ax, frame, width = self.ax_value, self.frames[f], 0.38
        index = np.arange(N_ACTIONS)
        if frame["q"] is None:
            ax.text(0.5, 0.5, "rollout ended: " + self._ending(), ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=MUTED)
        data = [self.bars(frame, i) for i in (0, 1)]
        for i in (0, 1):                         # the two parts, then the wholes: the legend
            if data[i] is None or data[i][0] is None:          # reads robot by robot
                continue
            base = index + (i - 0.5) * width
            physical, social = (np.clip(v, *self.limits) for v in data[i][:2])
            ax.bar(base - width / 4, physical, width=width / 2, color=ROBOT_COLOURS[i],
                   edgecolor="white", linewidth=.6, label=f"{self.names[i]} physical")
            ax.bar(base + width / 4, social, width=width / 2, color=ROBOT_COLOURS[i],
                   alpha=0.45, hatch="//", edgecolor="white", linewidth=.6,
                   label=f"{self.names[i]} social")
        for i in (0, 1):
            if data[i] is None:
                continue
            base = index + (i - 0.5) * width
            if data[i][2] is not None:           # acts for the partner: nothing to split
                ax.bar(base, np.clip(data[i][2], *self.limits), width=0.8 * width,
                       color=ROBOT_COLOURS[i], alpha=0.7, hatch="..", edgecolor="white",
                       linewidth=.6, label=f"{self.names[i]} total (acts for its partner)")
            ax.axvline(frame["action"][i] + (i - 0.5) * width, color=ROBOT_COLOURS[i],
                       ls="--", lw=1, alpha=0.7)
        if self.wide:
            ax.set_yscale("symlog", linthresh=1.0, linscale=0.6)
        ax.set_ylim(*self.limits)
        ax.set_xlim(-0.6, N_ACTIONS - 0.4)
        ax.set_xticks(index)
        ax.set_xticklabels(ACTION_TICKS, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0, c="0.5", lw=0.8)
        ax.set_title("value of each action relative to the one taken (dashed)"
                     + (", log scale beyond 1" if self.wide else "")
                     if self.values == "relative" else
                     "physical Q and immediate social reward (dashed = chosen)",
                     fontsize=9, color=INK, loc="left")
        if frame["q"] is not None:                # below the axis: never on top of a bar
            entries = len(ax.get_legend_handles_labels()[1])      # two rows, filled by column
            ax.legend(fontsize=7.2, ncols=max(1, (entries + 1) // 2), loc="upper center",
                      bbox_to_anchor=(0.5, -0.14), frameon=False, columnspacing=1.0,
                      handlelength=1.4)

    # -- timeline --------------------------------------------------------------------------------
    def _timeline(self, f: int) -> None:
        from matplotlib.patches import Rectangle

        ax, story = self.ax_time, self.story
        step = self.frames[f]["step"]                  # transitions completed
        total = max(story.n_steps, 12)
        rows = (1.15, 0.0)                             # y of the two robots' rows
        ax.set_xlim(0.4, total + 0.6)
        ax.set_ylim(-0.58, 1.92)
        for i, y in enumerate(rows):
            ax.axhline(y, color=ROBOT_COLOURS[i], alpha=0.13, lw=13, zorder=0)
        if self.frames[f]["action"] is not None:       # the transition being decided now
            ax.add_patch(Rectangle((step + 0.5, -0.54), 1.0, 2.42, facecolor=FRESH,
                                   edgecolor=INK, linewidth=1.2, zorder=1))
            ax.text(step + 1.0, 1.95, "now", ha="center", va="bottom", fontsize=8, color=INK,
                    fontweight="bold")
        shown = min(step + 1, story.n_steps)           # transitions whose action is known
        size = 11 if total <= 30 else (9 if total <= 45 else 7.5)
        for i, y in enumerate(rows):
            for t, (a, blocked, _) in enumerate(story.tape(i, shown), 1):
                ax.text(t, y, ACTION_GLYPHS[a], ha="center", va="center", fontsize=size,
                        color="0.62" if blocked else INK, zorder=3,
                        fontweight="bold" if a >= PICK else "normal")
            for first, last, held in story.carried(i, shown):       # what was carried
                ax.plot([first - 0.5, last + 0.5], [y - 0.34, y - 0.34], "-", lw=8,
                        color=self.object_colour[held], solid_capstyle="butt", zorder=2)
                ax.text(first - 0.42, y - 0.35, f"O{held + 1}", ha="left", va="center",
                        fontsize=6.3, color="white", fontweight="bold", zorder=3)
        marks = story.marks(step)
        # tags of neighbouring steps would cover each other: they lean apart instead
        per_step = ax.get_position().width * self.fig.get_figwidth() * 72.0 / (total + 0.2)
        for i in (0, 1):
            tags = sorted((t, " ".join(labels)) for (t, k), labels in marks.items() if k == i)
            width = [self._extent(text, fontsize=8, fontweight="bold")[0] + 4.0
                     for _, text in tags]                                   # points
            for n, (t, text) in enumerate(tags):
                before = n > 0 and ((t - tags[n - 1][0]) * per_step
                                    < 0.5 * (width[n] + width[n - 1]) + 2.0)
                after = n + 1 < len(tags) and ((tags[n + 1][0] - t) * per_step
                                               < 0.5 * (width[n] + width[n + 1]) + 2.0)
                x, ha = float(t), "center"
                if after and not before:
                    x, ha = t + 0.46, "right"
                elif before and not after:
                    x, ha = t - 0.46, "left"
                ax.text(x, rows[i] + 0.24, text, ha=ha, va="bottom", fontsize=8, color=INK,
                        fontweight="bold", zorder=4,
                        bbox=dict(boxstyle="round,pad=0.13", linewidth=0.7, edgecolor="0.45",
                                  facecolor=FRESH if t == step else "white"))
        ax.set_yticks(rows)
        margin = ax.get_position().x0 * self.fig.get_figwidth() * 72.0 - 6.0 - 4.0   # points
        fitted = [self._squeezed(name, margin, 9) for name in self.names]
        ax.set_yticklabels([text for text, _ in fitted], color=INK,
                           fontsize=min(size for _, size in fitted))
        every = 1 if total <= 30 else (2 if total <= 45 else 5)
        ax.set_xticks([t for t in range(1, total + 1) if t == 1 or t % every == 0])
        ax.tick_params(axis="x", labelsize=7.5, length=2)
        ax.tick_params(axis="y", length=0, pad=6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_title("timeline: the action of every step, what was carried, and the events",
                     fontsize=9, color=INK, loc="left", pad=17)       # clear of "now"
        ax.set_xlabel("step        " + LEGEND, fontsize=7.8, color=INK, loc="left")

    # -- export ----------------------------------------------------------------------------------
    def image(self, f: int, dpi: int):
        """Frame f as a PIL image."""
        from PIL import Image

        self.fig.set_dpi(dpi)
        with self.plt.rc_context(self.style):
            self._draw(f)
            self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())).convert("RGB")


def _held_frames(result: dict, names, fps, dpi, trail, values, title, event_hold,
                 placed_hold, end_hold, linger, font):
    """(frame image, seconds it is shown) for every frame of the animation, one at a time."""
    canvas = _Canvas(result, names, values, title, trail, linger, font)
    story = canvas.story
    hold = dict(plain=1.0 / float(fps), event=float(event_hold), placed=float(placed_hold),
                end=float(end_hold))
    try:
        for f in range(len(canvas.frames)):
            yield canvas.image(f, dpi), max(hold[story.weight(f)], hold["plain"])
    finally:
        canvas.close()


def render_video(result: dict, path: Union[str, Path], *,
                 names: Sequence[str] = ("agent_1", "agent_2"),
                 fps: float = 3.0, dpi: int = 100, trail: int = 25, values: str = "relative",
                 title: str = "", event_hold: float = 1.5, placed_hold: float = 2.5,
                 end_hold: float = 3.0, linger: int = 3, font: Optional[str] = FONT,
                 crf: int = 18) -> Path:
    """Write the rollout of ``result`` (a record of :func:`social_pairings.run_pairing`, in
    memory or loaded from JSON) to ``path`` as an MP4 video (H.264, yuv420p: it plays in
    browsers, slides and every player), frame for frame the GIF of :func:`render_gif`.

    The options are :func:`render_gif`'s.  A video runs at one frame rate, twice ``fps``:
    a frame that is held longer is repeated (with the defaults: a plain frame twice, one
    after an event 9 times, after a delivery 15, the last 18).  Frames are padded with white
    to a multiple of 16 pixels each way.  crf: H.264's quality (lower is better; 18 keeps
    the text sharp).  Needs ``imageio-ffmpeg`` (``pip install imageio-ffmpeg``; it brings
    its own ffmpeg)."""
    return _write_video(_held_frames(result, names, fps, dpi, trail, values, title, event_hold,
                                     placed_hold, end_hold, linger, font),
                        path, 2.0 * float(fps), crf)


def _write_video(frames, path: Union[str, Path], rate: float, crf: int = 18) -> Path:
    """Write ``frames`` -- (PIL image, seconds it is shown), one at a time -- to ``path`` as an
    MP4 (H.264, yuv420p) at ``rate`` frames a second, each image repeated for its seconds,
    padded with white to a multiple of 16 pixels each way.  It is written to
    ``<stem>.part.mp4`` and renamed to ``path`` once whole, so that a video cut off half way
    (an error while drawing, the run stopped) is never taken for a finished one."""
    import imageio_ffmpeg

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.stem + ".part.mp4")
    writer, size, whole = None, None, False
    try:
        for image, seconds in frames:
            pixels = np.asarray(image.convert("RGB"))
            if size is None:
                h, w = pixels.shape[:2]
                size = (w + (-w) % 16, h + (-h) % 16)
                writer = imageio_ffmpeg.write_frames(
                    str(part), size, fps=rate, codec="libx264", pix_fmt_in="rgb24",
                    pix_fmt_out="yuv420p", quality=None, macro_block_size=16,
                    output_params=["-crf", str(int(crf)), "-preset", "medium",
                                   "-movflags", "+faststart"])
                writer.send(None)
            frame = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
            frame[:pixels.shape[0], :pixels.shape[1]] = pixels
            data = frame.tobytes()
            for _ in range(max(1, int(round(seconds * rate)))):
                writer.send(data)
        whole = True
    finally:
        if writer is not None:
            writer.close()
        if whole:
            os.replace(part, path)
        else:
            part.unlink(missing_ok=True)
    return path


# ---- each robot's value at every cell, term by term ---------------------------------------------

class _ValueCanvas:
    """The figure of one robot's value maps (``Config.value_maps``): one panel for its value
    at every cell it could stand on, with its greedy action there, and one per term of its
    reward -- the discounted sums that add up to it."""

    def __init__(self, result: dict, robot: int, names: Sequence[str], title: str,
                 font: Optional[str]) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.style = {} if font is None else {"font.family": [font]}
        self.result, self.robot, self.names = result, int(robot), tuple(names)
        self.rows = [row for row in result["trace"]
                     if row.get("value_maps") and row["value_maps"][self.robot]]
        if not self.rows:
            raise ValueError("the record holds no value maps for this robot "
                             "(Config.value_maps, since 2026-10-17; a robot at level 0 has none)")
        scene = result["scene"]
        self.walls = np.asarray(scene["grid"]) == 0                        # (X, Y)
        self.cells = [tuple(int(v) for v in c) for c in scene["cells"]]
        self.homes = [tuple(c) for c in scene["objects"]]
        self.candidates = [tuple(c) for c in scene["candidates"]]
        self.destinations = [tuple(c) for c in scene["destinations"]]
        self.handoff = [tuple(c) for c in (scene.get("handoff_cells") or ())]
        rooms = scene.get("rooms")
        self.room = None if not rooms else {tuple(c) for c in rooms[self.robot]}
        goals = result.get("goals") or result.get("social_goals") or ("none", "none")
        config = result.get("config") or {}
        level = (result.get("levels_used") or result.get("levels") or (0, 0))[self.robot]
        reading = config.get("conflict", "table") if config else None
        self.title = title or (f"{self.names[self.robot]} ({goal_label(goals[self.robot], config.get('competition'), config.get('cooperate_with'), reading)}, level {level}): "  # noqa: E501
                               "the value of standing at each cell, and what it is made of")
        # the terms of every step: a term that pays nothing in a step is left out of that
        # step's maps (SocialMDP.value_components), so the set can change along the rollout --
        # a term's panel shows 0 in the steps it is not in
        place: dict = {}
        label_of: dict = {}
        for row in self.rows:
            maps = row["value_maps"][self.robot]
            for j, (name, label) in enumerate(zip(maps["names"], maps["labels"])):
                place[name] = min(place.get(name, j), j)
                label_of.setdefault(name, label)
        self.terms = sorted(place, key=lambda name: place[name])     # stable: first seen first
        self.labels = (["V: all terms together (arrow: its action there)"]
                       + [label_of[name] for name in self.terms])
        n = len(self.labels)
        self.ncols = 3 if n > 4 else 2
        self.nrows = -(-n // self.ncols)
        X, Y = self.walls.shape
        panel_w = max(2.9, 0.42 * X + 0.8)
        panel_h = max(2.4, 0.42 * Y + 0.7)
        with plt.rc_context(self.style):
            self.fig, axes = plt.subplots(self.nrows, self.ncols,
                                          figsize=(panel_w * self.ncols + 0.3,
                                                   panel_h * self.nrows + 1.2), squeeze=False)
            self.fig.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.04,
                                     wspace=0.22, hspace=0.3)
        self.axes = [ax for row in axes for ax in row]
        for ax in self.axes[n:]:
            ax.set_visible(False)
        self.bars: list = []
        self.subtitle = None

    def _values(self, row: dict, k: int) -> np.ndarray:
        """Panel k's map in a step: V (k = 0) or the term ``self.terms[k - 1]`` -- 0 wherever
        V is defined, if the step left that term out (it pays nothing there)."""
        maps = row["value_maps"][self.robot]
        total = np.asarray(maps["total"], dtype=float)
        if k == 0:
            return total
        name = self.terms[k - 1]
        if name not in maps["names"]:
            return np.where(np.isfinite(total), 0.0, np.nan)
        return np.asarray(maps["parts"][maps["names"].index(name)], dtype=float)

    def present(self, row: dict, k: int) -> bool:
        """Whether panel k's term is in the step's own maps (V always is)."""
        return k == 0 or self.terms[k - 1] in row["value_maps"][self.robot]["names"]

    def close(self) -> None:
        self.plt.close(self.fig)

    def draw(self, f: int) -> None:
        with self.plt.rc_context(self.style):
            self._draw(f)

    def _draw(self, f: int) -> None:
        from matplotlib.colors import Normalize
        from matplotlib.patches import Rectangle

        row = self.rows[f]
        maps = row["value_maps"][self.robot]
        i, p = self.robot, 1 - self.robot
        X, Y = self.walls.shape
        pos = [tuple(int(v) for v in c) for c in row["pos"]]
        holds = row["holds"]
        held = ["nothing" if h is None or h < 0 else f"O{h + 1}" for h in holds]
        step = row["step"] - 1
        self.fig.suptitle(self.title, fontsize=10.5, fontweight="bold", x=0.03, ha="left",
                          y=0.985)
        subtitle = (f"step {step} of {len(self.result['trace'])}:  {self.names[i]} at {pos[i]} "
                    f"holds {held[i]};  {self.names[p]} at {pos[p]} holds {held[p]}.  Each "
                    f"cell: the value if {self.names[i]} stood there now, everything else as "
                    f"it is; colours span each map's own range in this step.  The terms add "
                    f"up to V (to {maps['residual']:.0e}).")
        subtitle = textwrap.fill(subtitle, width=int(self.fig.get_figwidth() * 12.5))
        if self.subtitle is None:
            self.subtitle = self.fig.text(0.03, 0.945, subtitle, fontsize=8.4, color=INK,
                                          ha="left", va="top", linespacing=1.3)
        else:
            self.subtitle.set_text(subtitle)
        for bar in self.bars:
            bar.remove()
        self.bars = []
        objects = [tuple(int(v) for v in c) if c is not None else None
                   for c in row["object_cells"]]
        for k, ax in enumerate(self.axes[:len(self.labels)]):
            ax.clear()
            values = self._values(row, k)
            grid = np.full((X, Y), np.nan)
            for x, cell in enumerate(self.cells):
                grid[cell] = values[x]
            finite = values[np.isfinite(values)]
            lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
            if hi - lo < 1e-9:                   # one value everywhere: a flat colour
                lo, hi = lo - 0.5, hi + 0.5
            cmap = self.plt.get_cmap("viridis").with_extremes(bad="0.25")    # walls: grey
            image = ax.imshow(np.ma.masked_invalid(grid.T), origin="lower", cmap=cmap,
                              norm=Normalize(lo, hi), extent=[0, X, 0, Y],
                              interpolation="nearest")
            self.bars.append(self.fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02,
                                               format="%.1f"))
            self.bars[-1].ax.tick_params(labelsize=6.5)
            ax.set_xticks(np.arange(X + 1), minor=True)
            ax.set_yticks(np.arange(Y + 1), minor=True)
            ax.grid(True, which="minor", color="white", lw=0.6, alpha=0.6)
            ax.tick_params(which="minor", length=0)
            # the numbers, the actions, the cells that cannot be stood on
            for x, (cx, cy) in enumerate(self.cells):
                v = values[x]
                if not np.isfinite(v):
                    ax.add_patch(Rectangle((cx, cy), 1, 1, facecolor="white", alpha=0.55,
                                           edgecolor="none"))
                    continue
                colour = "white" if (v - lo) / (hi - lo) < 0.55 else "black"
                if k == 0 and maps["action"][x] >= 0:
                    a = int(maps["action"][x])
                    if a < WAIT:
                        dx, dy = MOVES[a]
                        ax.annotate("", (cx + 0.5 + 0.32 * dx, cy + 0.5 + 0.32 * dy),
                                    (cx + 0.5 - 0.1 * dx, cy + 0.5 - 0.1 * dy),
                                    arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.1,
                                                    mutation_scale=7), zorder=4)
                        ax.text(cx + 0.5, cy + 0.2, f"{v:.1f}", ha="center", va="center",
                                fontsize=5.6, color=colour, zorder=5)
                    else:
                        ax.text(cx + 0.5, cy + 0.66, ACTION_GLYPHS[a], ha="center", va="center",
                                fontsize=7.5, fontweight="bold", color=colour, zorder=5)
                        ax.text(cx + 0.5, cy + 0.26, f"{v:.1f}", ha="center", va="center",
                                fontsize=5.6, color=colour, zorder=5)
                else:
                    ax.text(cx + 0.5, cy + 0.5, f"{v:.1f}", ha="center", va="center",
                            fontsize=6.0, color=colour, zorder=5)
            for q, cell in enumerate(self.candidates):
                star = "*" if cell not in self.destinations else "*"
                dest = self.destinations.index(cell) if cell in self.destinations else None
                ax.plot(cell[0] + 0.5, cell[1] + 0.82, star, ms=7 if dest is not None else 4,
                        color=ROBOT_COLOURS[dest] if dest is not None else "0.75",
                        mec="white", mew=0.5, zorder=6)
            for q, cell in enumerate(objects):
                if cell is None or q >= len(self.homes):
                    continue
                colour = ROBOT_COLOURS[q] if q < 2 else SPARE_COLOUR
                ax.add_patch(Rectangle((cell[0] + 0.08, cell[1] + 0.72), 0.22, 0.22,
                                       facecolor=colour, edgecolor="white", lw=0.5, zorder=6))
            for cell in self.handoff:
                ax.add_patch(Rectangle((cell[0] + 0.04, cell[1] + 0.04), 0.92, 0.92, fill=False,
                                       edgecolor="0.3", ls="--", lw=0.8, zorder=6))
            ax.add_patch(Rectangle((pos[i][0] + 0.04, pos[i][1] + 0.04), 0.92, 0.92, fill=False,
                                   edgecolor=ROBOT_COLOURS[i], lw=2.2, zorder=7))
            ax.plot(pos[p][0] + 0.5, pos[p][1] + 0.5, "o", ms=9, color=ROBOT_COLOURS[p],
                    mec="white", mew=0.8, zorder=8, alpha=0.9)
            ax.text(pos[p][0] + 0.5, pos[p][1] + 0.5, str(p + 1), ha="center", va="center",
                    fontsize=5.5, color="white", fontweight="bold", zorder=9)
            ax.set_xlim(0, X)
            ax.set_ylim(0, Y)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(self.labels[k], fontsize=8.4, color=INK, loc="left")
            ax.set_title("" if self.present(row, k) else "pays nothing in this step",
                         fontsize=7.2, color=INK, style="italic", loc="right")
            ax.set_aspect("equal")

    def image(self, f: int, dpi: int):
        from PIL import Image

        self.draw(f)
        self.fig.set_dpi(dpi)
        with self.plt.rc_context(self.style):
            self.fig.canvas.draw()
        return Image.fromarray(np.asarray(self.fig.canvas.buffer_rgba())).convert("RGB")


def render_value_video(result: dict, path: Union[str, Path], robot: int, *,
                       names: Sequence[str] = ("agent_1", "agent_2"), fps: float = 1.0,
                       dpi: int = 100, title: str = "", end_hold: float = 3.0,
                       font: Optional[str] = FONT, crf: int = 18) -> Path:
    """Write robot ``robot``'s value maps along the rollout of ``result`` to ``path`` as an
    MP4 video (``Config.value_maps``; since 2026-10-17): one frame per transition -- the
    robot's value at every cell it could stand on, with its greedy action there, and one map
    per term of its reward (its own goal, its action costs, each social term), which add up
    to it -- every map with its own colour scale, the same in every frame.  fps: frames
    (transitions) a second; the last frame is held ``end_hold`` seconds."""
    canvas = _ValueCanvas(result, robot, names, title, font)
    n = len(canvas.rows)

    def frames():
        try:
            for f in range(n):
                yield canvas.image(f, dpi), (end_hold if f == n - 1 else 1.0 / float(fps))
        finally:
            canvas.close()

    return _write_video(frames(), path, max(2.0 * float(fps), 2.0), crf)


def render_videos(result: dict, stem: Union[str, Path], *,
                  names: Sequence[str] = ("agent_1", "agent_2"), fps: float = 3.0,
                  title: str = "", maps_fps: float = 1.0, **options) -> list:
    """The videos of one rollout: ``<stem>.mp4`` (:func:`render_video`, options passed
    on) and, where the record holds value maps (``Config.value_maps``),
    ``<stem>__values_<name>.mp4`` for each robot that has them
    (:func:`render_value_video`).  -> the paths written."""
    paths = video_paths(result, stem, names=names)
    out = [render_video(result, paths[0][1], names=names, fps=fps, title=title, **options)]
    for i, path in paths[1:]:
        out.append(render_value_video(result, path, i, names=names, fps=maps_fps))
    return out


def video_paths(result: dict, stem: Union[str, Path], *,
                names: Sequence[str] = ("agent_1", "agent_2")) -> list:
    """[(robot or None, path)] of the videos :func:`render_videos` writes for a record:
    ``(None, <stem>.mp4)``, then ``(i, <stem>__values_<name>.mp4)`` for each robot i whose
    value maps the record holds."""
    stem = Path(stem)
    out = [(None, stem.with_suffix(".mp4"))]
    for i in (0, 1):
        if any(row.get("value_maps") and row["value_maps"][i] for row in result["trace"]):
            out.append((i, stem.parent / f"{stem.name}__values_{names[i]}.mp4"))
    return out


def render_gif(result: dict, path: Union[str, Path], *,
               names: Sequence[str] = ("agent_1", "agent_2"),
               fps: float = 3.0, dpi: int = 100, trail: int = 25, values: str = "relative",
               title: str = "", event_hold: float = 1.5, placed_hold: float = 2.5,
               end_hold: float = 3.0, linger: int = 3, font: Optional[str] = FONT) -> Path:
    """Write the rollout of ``result`` (a record of :func:`social_pairings.run_pairing`, in
    memory or loaded from JSON) to ``path`` as a GIF (:func:`render_video`: as an MP4 video,
    what the example scripts write since 2026-10-17).

    names : how the two robots are called.
    fps : frames per second of the frames in which nothing happened to an object.
    values : "relative" -- each action's physical and social value relative to the action
        taken; "raw" -- levels (physical Q and the immediate social reward).
    title : the headline (default: the two robots' social goals and weights).
    event_hold, placed_hold, end_hold : seconds a frame is held after an event, after an
        object was placed on a destination, and at the end.
    linger : for how many frames an event's callout stays on the map.
    font : the font family; None: the one matplotlib is configured with.
    """
    from PIL import Image

    images, durations = [], []
    for image, seconds in _held_frames(result, names, fps, dpi, trail, values, title,
                                       event_hold, placed_hold, end_hold, linger, font):
        images.append(image)
        durations.append(int(round(1000.0 * seconds)))
    # One palette for every frame, taken from the frames in which something happens and a
    # few others: what does not change between two frames is then stored once (the file is
    # about five times smaller than with a palette per frame).
    n = len(images)
    plain = min(durations)
    picks = [f for f in range(n) if durations[f] > plain][:8]
    picks = sorted(set(picks + [round(k * (n - 1) / 4) for k in range(5)]))
    width, height = images[0].size
    # each robot's leading belief is a few dozen pixels of text a frame, too few to win a
    # colour of their own: a strip of each lead colour under the frames keeps them
    strip = 48
    sheet = Image.new("RGB", (width, height * len(picks) + strip), "white")
    for row, f in enumerate(picks):
        sheet.paste(images[f], (0, row * height))
    for k, colour in enumerate(LEAD_COLOURS):
        part = width // len(LEAD_COLOURS)
        sheet.paste(Image.new("RGB", (part, strip), colour), (k * part, height * len(picks)))
    palette = sheet.quantize(colors=128, dither=0)
    images = [image.quantize(palette=palette, dither=0) for image in images]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(str(path), save_all=True, append_images=images[1:], duration=durations,
                   loop=0, optimize=True, disposal=1)
    return path


def save_frame(result: dict, path: Union[str, Path], frame: int = -1, *,
               names: Sequence[str] = ("agent_1", "agent_2"), dpi: int = 110,
               trail: int = 25, values: str = "relative", title: str = "",
               linger: int = 3, font: Optional[str] = FONT) -> Path:
    """Save one frame of the animation as an image (``frame``: the state after that many
    transitions; negative counts from the end)."""
    canvas = _Canvas(result, names, values, title, trail, linger, font)
    try:
        image = canvas.image(range(len(canvas.frames))[frame], dpi)
    finally:
        canvas.close()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    return path
