"""Scenes in which an object has to get from one robot's side to the other's, and what two
robots with the exchange goal do there.

Run from the project root:

    python examples/run_exchange_scene.py                  # exchange / exchange, weights 1 1
    python examples/run_exchange_scene.py --weights 30 30
    python examples/run_exchange_scene.py --controls       # and what other pairings do there
    python examples/run_exchange_scene.py --goals cooperation none
    python examples/run_exchange_scene.py --hand-over      # ... hand-overs allowed as well
    python examples/run_exchange_scene.py --exchange-level1 none      # Table I as printed
    python examples/run_exchange_scene.py --told           # each robot told its partner's goal
    python examples/run_exchange_scene.py --scene rooms    # the open doorway of before
    python examples/run_exchange_scene.py --scene corridor --forced-swap --weights 1 1

``exchange`` (:func:`social_pairings.exchange_scene`, the default): two rooms and a doorway
ONE cell wide.  Each robot starts next to its partner's object; its own object lies in the
other room, and its destination in its own.  Each robot STAYS IN ITS OWN ROOM; the doorway
belongs to both and is a hand-off cell: a robot standing in it puts its partner's object
down there, and the partner picks it up from there.  Nobody can finish its partner's task
for it (``Config.set_down_for_partner=False``) and nothing goes from hand to hand
(``Config.hand_over=False``), so an object gets to its owner only by being put down in the
doorway and picked up there.  ``--hand-over`` also lets a robot standing NEXT TO the doorway
hand an object to a partner standing in it with free hands (from the doorway itself an
object is put down, never handed over).

NOBODY IS TOLD ITS PARTNER'S SOCIAL GOAL, on this scene or the two below
(``Config.partner_social_goal="inferred"``, the default): each level-2 robot starts from a
uniform prior over seven social goals -- the paper's five, none and hinder -- and updates it
from what its partner does (it sees where its partner stands and what it holds; its
partner's weight and level it is given).  The figure shows that belief next to the
one about the partner's destination, and the script prints what each robot takes its
partner's social goal to be at the end.  What an exchanger is at level 1 -- the model by
which a level-2 robot recognises one -- is the "copy of the other agent's goal" of the
paper's text, n r(g_j), not multiplied by the robot's weight
(``Config.exchange_level1="copy_unweighted"``).  Two exchangers then exchange at every
weight tried from 0.55 to 100 (from 0.45 with the paper's five goals held possible): each
picks its partner's object up, puts it down in the doorway, picks its own up from there
and carries it to its own destination.

``--exchange-level1 none`` is Table I as printed, where an exchanger has no term at level
1: a partner that helps is then read as a cooperator, and whoever is helped first stops
helping.  ``--exchange-level1 copy`` multiplies the copy by the weight.  ``--told`` tells
each robot its partner's social goal, with Table I as printed for an exchanger at level 1
(``--exchange-level1`` still changes that): the defaults until 2026-10-08, under which the
results of the two scenes below were made.  ``--competition`` is also the reading of the
competitor among the hypotheses.

``rooms`` (:func:`one_door_scene`): the same map without rooms of their own and without a
hand-off cell, under the default rule for the hands (its results: ``--told``).  Nothing
there forces an exchange: a robot can walk through the doorway and fetch its own object, or
carry its partner's object to the partner's destination.

``corridor`` (:func:`corridor_scene`): a corridor one cell wide in which the robots can
never pass each other, under the default rule for the hands (its results: ``--told``).

``--controls`` rolls out more pairings on the same scene under the same rule: two robots
without social goals at level 2; an exchanging robot and a cooperating one, each next to a
partner without social goal; two cooperators; an exchanging robot next to a cooperating
one; two competitors (``--competition``, default "rival"); and a cooperating robot next to
a coercing one.

Writes ``<goal>__<goal>.json`` and ``.mp4`` (an MP4 video), ``00_baseline`` (two level-0
robots without
social goals), the controls as ``control_<goal>__<goal>`` and ``summary.json`` to
``output/exchange_scene_<scene>_w<weights>[...]_inferred/`` (``_told`` with ``--told``).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pairings import (NAMES, competition_of, cooperation_of, print_row,  # noqa: E402
                          summary_row)

from social_pairings import (TOLD, Config, Scene, exchange_scene, run_baseline,  # noqa: E402
                             run_pairing)
from social_pairings import goals as G  # noqa: E402
from social_pairings.render import goal_label, render_videos  # noqa: E402
from social_pairings.serialize import save_result, to_json  # noqa: E402


def one_door_scene() -> Scene:
    """The six-object scene with the doorway narrowed to the one cell (4, 2) and the two
    destinations swapped, so that each robot's destination is in its own room: agent_1
    (left room) brings O1, which lies in the right room, to (1, 5); agent_2 (right room)
    brings O2, which lies in the left room, to (7, 1).  Either robot may walk through the
    doorway."""
    base = exchange_scene(rooms=False, handoff=False)
    return Scene(**{**base.__dict__, "name": "one_door"})


def corridor_scene() -> Scene:
    """Seven cells in a row.  From the left: O2's place, agent_1's destination, agent_1,
    a free cell, agent_2, agent_2's destination, O1's place.  agent_1 is always to the left
    of agent_2, so it can never stand on the last cell, where its object lies, and agent_2
    never on the first."""
    grid = np.full((9, 3), np.inf)
    grid[1:8, 1] = 1.0
    return Scene(grid=grid, starts=((3, 1), (5, 1)), pickups=((7, 1), (1, 1)),
                 destinations=((2, 1), (6, 1)), candidate_cells=((2, 1), (6, 1), (4, 1)),
                 name="corridor")


SCENES = {"exchange": exchange_scene, "rooms": one_door_scene, "corridor": corridor_scene}
# The rule for the hands each scene is run under, unless a flag says otherwise.
RULES = {"exchange": dict(set_down_for_partner=False, hand_over=False),
         "rooms": dict(set_down_for_partner=True, hand_over=True),
         "corridor": dict(set_down_for_partner=True, hand_over=True)}
# --told: what each robot knows of its partner as until 2026-10-08 (the reading of competition
# is --competition's)
KNOWN = {key: value for key, value in TOLD.items() if key != "competition"}
# --controls: (agent_1's goal, agent_2's goal, levels or None for the defaults)
CONTROLS = (("none", "none", (2, 2)), ("exchange", "none", None), ("none", "exchange", None),
            ("cooperation", "none", None), ("none", "cooperation", None),
            ("cooperation", "cooperation", None), ("exchange", "cooperation", None),
            ("cooperation", "exchange", None), ("competition", "competition", None),
            ("cooperation", "coercion", None))


def what_happened(result: dict) -> list:
    """The object events of a rollout in words, one line each."""
    lines = []
    for t, what, by, obj, other in result["events"]:
        text = f"{NAMES[by]} {what} O{obj + 1}"
        if what == "handed over":
            text = f"{NAMES[by]} handed O{obj + 1} to {NAMES[other]}"
        elif what == "put down":
            text = f"{NAMES[by]} put O{obj + 1} down in the doorway"
        elif what == "set down":
            text = (f"{NAMES[by]} set O{obj + 1} down on its destination" if other == by
                    else f"{NAMES[by]} set O{obj + 1} down on {NAMES[other]}'s destination")
        elif what == "set down in vain":
            text = (f"{NAMES[by]} tried to set O{obj + 1} down for {NAMES[other]} (not its "
                    "destination" + ("" if obj == other else ", or not its object") + ")")
        elif what == "handed over in vain":
            text = (f"{NAMES[by]} tried to hand O{obj + 1} to {NAMES[other]} (not its "
                    "object)")
        lines.append(f"transition {t:2d}: {text}")
    return lines


def what_they_believe(result: dict) -> list:
    """Where the partner's social goal is inferred: what each level-2 robot takes its
    partner's social goal to be at the end, one line each."""
    if result["config"]["partner_social_goal"] != "inferred":
        return []
    lines = []
    for i, belief in enumerate(result["final_social_belief"]):
        if belief is None:
            continue
        belief = np.asarray(belief, dtype=float)
        top = [f"{G.SOCIAL_GOALS[k]} {belief[k]:.2f}" for k in np.argsort(-belief)[:3]
               if belief[k] >= 0.005]
        lines.append(f"{NAMES[i]} takes {NAMES[1 - i]}'s social goal to be: {', '.join(top)} "
                     f"(it is {result['goals'][1 - i]})")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", choices=tuple(SCENES), default="exchange")
    parser.add_argument("--goals", nargs=2, default=("exchange", "exchange"),
                        metavar=("AGENT_1", "AGENT_2"))
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0),
                        metavar=("AGENT_1", "AGENT_2"))
    parser.add_argument("--levels", type=int, nargs=2, choices=(0, 1, 2),
                        metavar=("AGENT_1", "AGENT_2"),
                        help="default: 2 for every robot with a social goal, 0 without one "
                             "(with --told, as until 2026-10-11: 1 for a hinderer and for a "
                             "competitor of a race form)")
    parser.add_argument("--hand-over", action=argparse.BooleanOptionalAction, default=None,
                        help="an object may be handed straight to the partner standing next "
                             "to its holder (Config.hand_over).  Default: no on the exchange "
                             "scene -- an object is put down and picked up; yes on the other "
                             "two")
    parser.add_argument("--set-down-for-partner", action=argparse.BooleanOptionalAction,
                        default=None,
                        help="a robot may set its partner's object down on the partner's "
                             "destination (Config.set_down_for_partner).  Default: no on the "
                             "exchange scene, yes on the other two")
    parser.add_argument("--forced-swap", action="store_true",
                        help="one robot's \"use\" swaps the two objects (Config.forced_swap)")
    parser.add_argument("--told", action="store_true",
                        help="each robot is told its partner's social goal, and an exchanger "
                             "at level 1 is as Table I prints it (the defaults until "
                             "2026-10-08).  Default: each infers it from what its partner does")
    parser.add_argument("--exchange-level1", choices=("none", "copy", "copy_unweighted"),
                        help="what an exchanger is at level 1 (Config.exchange_level1).  "
                             "Default: copy_unweighted; none with --told")
    parser.add_argument("--objects", choices=("inferred", "known"),
                        help="which object a robot's partner wants: inferred over every object "
                             "of the scene (the default since 2026-10-09, as in the paper) or "
                             "known, its own (the records of before; also with --told)")
    parser.add_argument("--max-objects", type=int,
                        help="objects in the model, 2..6 (Config.max_objects); the others are "
                             "scenery.  With the objects inferred the six-object scenes need "
                             "about 13 GiB, with 3 about 1 GiB")
    parser.add_argument("--cooperate-with", nargs="+", metavar="GOAL",
                        help="the partners a level-2 cooperator cooperates with "
                             "(Config.cooperate_with): by default cooperation competition "
                             "exchange; 'all' is Table I as printed (the records of before "
                             "2026-10-11; also with --told)")
    parser.add_argument("--controls", action="store_true",
                        help="also roll out the control pairings on this scene")
    parser.add_argument("--competition", choices=G.COMPETITION_FORMS,
                        default=Config().competition)
    parser.add_argument("--steps", type=int, default=60, help="most transitions per rollout")
    parser.add_argument("--output", type=Path, help="output folder")
    parser.add_argument("--no-video", "--no-gif", dest="no_gif", action="store_true",
                        help="records and summary only")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=0, help="0 = every CPU available")
    args = parser.parse_args()
    if any(w < 0 for w in args.weights):
        parser.error("--weights must be >= 0")
    goals = tuple(args.goals)
    for g in goals:
        G.parse_goal(g)
    weights = tuple(args.weights)
    levels = None if args.levels is None else tuple(args.levels)
    scene = SCENES[args.scene]()
    objects = args.objects or ("known" if args.told else "inferred")
    knows_objects = dict(partner_object=objects)
    if args.max_objects is not None:
        knows_objects["max_objects"] = args.max_objects
    if args.cooperate_with:
        knows_objects["cooperate_with"] = tuple(args.cooperate_with)
    rule = {**RULES[args.scene], **(KNOWN if args.told else {}), **knows_objects}
    tag = f"{args.scene}_w" + "_".join(f"{w:g}" for w in weights)
    for option in ("hand_over", "set_down_for_partner"):         # only where a flag changes it
        asked = getattr(args, option)
        if asked is not None and asked != rule[option]:
            rule[option] = asked
            tag += f"_{option}" if asked else f"_no_{option}"
    tag += "_forced_swap" * args.forced_swap
    level1 = rule.get("exchange_level1", Config().exchange_level1)
    if args.exchange_level1 is not None and args.exchange_level1 != level1:
        rule["exchange_level1"] = args.exchange_level1
        tag += f"_level1_{args.exchange_level1}"
    if levels is not None:
        tag += f"_levels{levels[0]}_{levels[1]}"
    if args.competition != Config().competition:
        tag += f"_{args.competition}"
    tag += "_told" if args.told else "_inferred"
    if objects == "inferred":                    # (the records of before: objects known)
        tag += "_objects" + ("" if args.max_objects is None else f"{args.max_objects}")
    cfg = Config(num_threads=args.threads, competition=args.competition,
                 forced_swap=args.forced_swap, **rule)
    tag += G.cooperation_tag(cfg.cooperate_with)         # (the records of before: Table I's)
    tag += G.level_tag(cfg.level_free_level)             # (... and a hinderer at level 1)
    output = (args.output or ROOT / "output" / f"exchange_scene_{tag}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    print(f"scene: {scene.name} | {goals[0]} / {goals[1]} | weights {weights} | partner goals "
          f"{cfg.partner_social_goal} | exchange at level 1: {cfg.exchange_level1} | hand_over="
          f"{cfg.hand_over} set_down_for_partner={cfg.set_down_for_partner} forced_swap="
          f"{cfg.forced_swap} | {output}", flush=True)

    def draw(name: str, result: dict, title: str) -> None:
        if args.no_gif:
            return
        tic = time.perf_counter()
        render_videos(result, output / name, names=NAMES, fps=args.fps, title=title)
        print(f"    {name}.mp4 ({time.perf_counter() - tic:.0f}s)", flush=True)

    def roll(name: str, pair, pair_levels) -> dict:
        result = run_pairing(scene, pair, weights=weights, levels=pair_levels, cfg=cfg,
                             n_steps=args.steps)
        save_result(output / f"{name}.json", result)
        row = summary_row(name, result, baseline)
        row["events"] = result["events"]
        for key in ("handed_over", "put_downs", "set_down_for_partner", "failed_set_downs"):
            row[key] = [led[key] for led in result["ledger"]]
        print_row(row)
        for line in what_happened(result):
            print(f"    {line}", flush=True)
        for line in what_they_believe(result):
            print(f"    {line}", flush=True)
        row["final_social_belief"] = result["final_social_belief"]
        used = result["levels_used"]
        label = [goal_label(g, competition_of(result), cooperation_of(result)) for g in pair]
        told = "" if cfg.partner_social_goal == "known" else "     goals inferred"
        draw(name, result, f"{NAMES[0]}: {label[0]}     {NAMES[1]}: {label[1]}     "
                           f"weights {weights[0]:g} / {weights[1]:g}     "
                           f"levels {used[0]} / {used[1]}{told}")
        return row

    baseline = run_baseline(scene, cfg=cfg, n_steps=args.steps)
    save_result(output / "00_baseline.json", baseline)
    done = " / ".join("never" if t is None else str(t) for t in baseline["done_at"])
    print(f"{'00_baseline':<28} no social goals, level 0: done at {done} "
          f"(stopped by {baseline['stopped_by']}, {baseline['steps']} transitions)", flush=True)
    draw("00_baseline", baseline, "no social goals: each robot pursues its physical goal")

    row = roll(f"{goals[0]}__{goals[1]}", goals, levels)
    controls = []
    if args.controls:
        for g0, g1, pair_levels in CONTROLS:
            name = f"control_{g0}__{g1}"
            if pair_levels is not None:
                name += f"_levels{pair_levels[0]}_{pair_levels[1]}"
            controls.append(roll(name, (g0, g1), pair_levels))
    summary = dict(scene=scene.to_json(), config=dataclasses.asdict(cfg), weights=weights,
                   goals=goals, levels="default" if levels is None else levels,
                   baseline_done_at=baseline["done_at"],
                   baseline_stopped_by=baseline["stopped_by"], pairing=row, controls=controls)
    (output / "summary.json").write_text(json.dumps(to_json(summary), indent=1) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
