"""Roll out pairings of social goals on the six-object scene and check who hinders whom.

Run from the project root:

    python examples/run_pairings.py                      # all eight pairings, weights 1 1
    python examples/run_pairings.py --weights 30 30      # the same with strong social terms
    python examples/run_pairings.py --case 6 7 8         # only the hinder pairings
    python examples/run_pairings.py --render-only        # re-draw the videos of saved runs
    python examples/run_pairings.py --competition race   # a competitor that stops once done
    python examples/run_pairings.py --competition table  # competition as Table I prints it
    python examples/run_pairings.py --competition table_hostile  # ... never helping a cooperator
    python examples/run_pairings.py --competition race_then_hinder   # ... hindering once done
    python examples/run_pairings.py --levels 2 2 --case 1 3 7    # competitors at level 2
    python examples/run_pairings.py --told --competition race_then_hinder   # as until 2026-10-08

Nobody is told its partner's social goal: each level-2 robot infers it, and its partner's
destination, from what the partner does (``Config``'s defaults: seven social goals held
possible, competition read as a rival).  ``--told`` tells each robot its partner's social
goal, with Table I as printed for an exchanger at level 1 -- the defaults until 2026-10-08,
under which the README's tables marked "told" were made (with ``--competition
race_then_hinder``, the reading of competition of then, for the eight pairings).

For every pairing it writes ``<name>.json`` (the full record) and ``<name>.mp4`` (the annotated
animation: map with every pick-up and put-down called out, a status table with each robot's
action, hands and goal, beliefs, action values and a timeline) to the output folder
(``output/social_pairings_w<weights>[_<reading>][_levels..]_inferred/``, ``_told`` with
``--told``; no ``_<reading>`` under race_then_hinder), plus
``00_baseline`` (no social goals) and ``summary.json``, and prints for every robot that was
out to stop its partner, or took its partner's object, whether it stopped it, measured
against the baseline (:func:`works_against`).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_pairings import (TOLD, Config, default_levels, levels_used,  # noqa: E402
                             run_baseline, run_pairing, six_object_scene, two_room_scene)
from social_pairings import goals as G  # noqa: E402
from social_pairings.hindrance import format_report, hindrance_report  # noqa: E402
from social_pairings.render import goal_label, render_videos  # noqa: E402
from social_pairings.serialize import load_result, save_result, to_json  # noqa: E402

PAIRINGS = (
    ("01_competitive_competitive", ("competition", "competition")),
    ("02_cooperative_cooperative", ("cooperation", "cooperation")),
    ("03_competitive_cooperative", ("competition", "cooperation")),
    ("04_exchange_cooperative", ("exchange", "cooperation")),
    ("05_conflict_cooperative", ("conflict", "cooperation")),
    ("06_hinder_cooperative", ("hinder", "cooperation")),
    ("07_hinder_competitive", ("hinder", "competition")),
    ("08_hinder_none", ("hinder", "none")),
)
NAMES = ("agent_1", "agent_2")
# --told: what each robot knows of its partner as until 2026-10-08 (the reading of competition
# is --competition's)
KNOWN = {key: value for key, value in TOLD.items() if key != "competition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", choices=("six", "two"), default="six",
                        help="six objects (four of them spare), or only the two own objects")
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0),
                        metavar=("AGENT_1", "AGENT_2"),
                        help="weight of each robot's social term (1 = Table I as printed)")
    parser.add_argument("--levels", type=int, nargs=2, choices=(0, 1, 2),
                        metavar=("AGENT_1", "AGENT_2"),
                        help="reasoning level of each robot.  Default: 2 for every robot "
                             "with a social goal, 0 without one; with --told (as until "
                             "2026-10-11) 1 for a hinderer and for a competitor of a race form "
                             "(race_then_hinder, race).  Levels given here are used as given "
                             "(with --told a hinderer stays at level 1 at most): --levels 2 2 "
                             "puts a competitor at level 2")
    parser.add_argument("--steps", type=int, default=60, help="most transitions per rollout")
    parser.add_argument("--case", type=int, nargs="+", metavar="N",
                        help=f"only these pairings (1..{len(PAIRINGS)})")
    parser.add_argument("--output", type=Path, help="output folder")
    parser.add_argument("--render-only", action="store_true",
                        help="re-draw the videos from the saved records")
    parser.add_argument("--no-video", "--no-gif", dest="no_gif", action="store_true",
                        help="records and summary only")
    parser.add_argument("--values", choices=("relative", "raw"), default="relative",
                        help="value panel: relative to the action taken, or raw levels")
    parser.add_argument("--fps", type=float, default=3.0,
                        help="frames per second (frames after an event are held longer)")
    parser.add_argument("--competition", choices=G.COMPETITION_FORMS,
                        default=Config().competition,
                        help="competition: its own goal first and then against its partner "
                             "(race_then_hinder); its own goal and nothing about its partner "
                             "once it is done (race); Table I as printed, which helps a "
                             "cooperative partner (table); Table I with that cell turned, "
                             "-R_j against a cooperative partner too (table_hostile); or a "
                             "race against every partner but a cooperative one, against "
                             "which it is the competitor of table_hostile (rival)")
    parser.add_argument("--told", action="store_true",
                        help="each robot is told its partner's social goal, and an exchanger "
                             "at level 1 is as Table I prints it (the defaults until "
                             "2026-10-08)")
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
    parser.add_argument("--hands-full-reward", choices=("ignored", "none"), default="ignored",
                        help="goal reward of a robot whose hands hold an object that is not "
                             "its own: as if they were free, or none")
    parser.add_argument("--threads", type=int, default=0, help="0 = every CPU available")
    parser.add_argument("--sample", action="store_true",
                        help="robots draw their actions from softmax(Q / tau) instead of argmax")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if any(w < 0 for w in args.weights):
        parser.error("--weights must be >= 0")
    if args.case and any(not 1 <= c <= len(PAIRINGS) for c in args.case):
        parser.error(f"--case must be in 1..{len(PAIRINGS)}")
    return args


def works_against(goal: int, partner_goal: int, competition: str, level: int,
                  done_first: bool, took_object: bool = False, told: bool = True) -> bool:
    """Is a hindrance report due for a robot with ``goal`` in this rollout?  ``hinder`` and
    ``conflict`` are out to stop their partner: always.  A competitor of a race form: when
    it took its partner's object (``took_object``: at level 2 it may, before its own goal is
    done or without it ever being done, and under "race" as well) -- and under
    "race_then_hinder" also when its own goal was done while its partner's was still open
    (``done_first``), from where on it is paid for stopping its partner.  A competitor as
    Table I prints it: at level 1 always (``-r(g_j)``), and above only against a partner it
    is hostile to -- against a cooperator its term is the cooperator's reward, unless that
    cell is turned ("table_hostile").  A rival: against a cooperative partner, which it
    works against -- at level 2, and at level 1 if it is told whom it faces (``told``:
    partner goals are known); against anybody else it races, and a report is due when it
    took its partner's object."""
    if goal in (G.HINDER, G.CONFLICT):
        return True
    if goal != G.COMPETITION:
        return False
    if competition == "rival":
        return (partner_goal == G.COOPERATION and (level >= 2 or told)) or took_object
    if competition in ("table", "table_hostile"):
        against = {G.NONE, G.CONFLICT, G.COMPETITION, G.HINDER}
        if competition == "table_hostile":       # ... and against those the table has it help
            against |= {G.COOPERATION, G.EXCHANGE}
        return level == 1 or partner_goal in against
    return took_object or (competition == "race_then_hinder" and done_first)


def competition_of(result: dict) -> str:
    """The reading of competition a record was written under (older records: the table)."""
    return result.get("config", {}).get("competition", "table")


def cooperation_of(result: dict):
    """The partners a record's cooperator cooperated with (older records: None, every
    partner -- Table I as printed)."""
    return result.get("config", {}).get("cooperate_with")


def summary_row(name: str, result: dict, baseline: dict) -> dict:
    """What one pairing did, with a hindrance report for every robot that was out to stop
    its partner or took its partner's object (:func:`works_against`)."""
    goals = [G.parse_goal(g) for g in result["social_goals"]]
    done = result["done_at"]
    first = [done[i] is not None and (done[1 - i] is None or done[i] < done[1 - i])
             for i in (0, 1)]
    took = [any(what == "picked up" and by == i and obj == 1 - i
                for _, what, by, obj, _ in result["events"]) for i in (0, 1)]
    told = result.get("config", {}).get("partner_social_goal", "known") == "known"
    reports = {NAMES[i]: hindrance_report(result, baseline, hinderer=i)
               for i in (0, 1)
               if works_against(goals[i], goals[1 - i], competition_of(result),
                                result["levels_used"][i], first[i], took[i], told)}
    return dict(name=name, goals=result["goals"], levels=result["levels"],
                levels_used=result["levels_used"], weights=result["weights"],
                social_terms=result["social_terms"], steps=result["steps"],
                stopped_by=result["stopped_by"], done_at=result["done_at"],
                baseline_done_at=baseline["done_at"], holds=result["holds"],
                numerically_accepted=result["numerically_accepted"],
                wall_time_s=result["wall_time_s"], hindrance=reports)


def print_row(row: dict) -> None:
    done = " / ".join("never" if t is None else str(t) for t in row["done_at"])
    print(f"{row['name']:<28} levels {row['levels_used'][0]}/{row['levels_used'][1]}  "
          f"done at {done:<13} stopped by {row['stopped_by']:<8} "
          f"({row['steps']} transitions, {row['wall_time_s']:.0f}s)", flush=True)
    for who, report in row["hindrance"].items():
        print(f"    {who}: {format_report(report, NAMES)}", flush=True)


def main() -> None:
    args = parse_args()
    scene = six_object_scene() if args.scene == "six" else two_room_scene()
    objects = args.objects or ("known" if args.told else "inferred")
    knows_objects = dict(partner_object=objects)
    if args.max_objects is not None:
        knows_objects["max_objects"] = args.max_objects
    if args.cooperate_with:
        knows_objects["cooperate_with"] = tuple(args.cooperate_with)
    cfg = Config(num_threads=args.threads, seed=args.seed,
                 action_selection="sample" if args.sample else "argmax",
                 competition=args.competition, hands_full_reward=args.hands_full_reward,
                 **{**(KNOWN if args.told else {}), **knows_objects})
    weights = tuple(args.weights)
    levels = None if args.levels is None else tuple(args.levels)
    tag = "_".join(f"{w:g}" for w in weights)
    if args.competition != "race_then_hinder":           # one folder per reading of competition
        tag += f"_{args.competition}"
    if levels is not None:                               # ... and per levels asked for
        tag += f"_levels{levels[0]}_{levels[1]}"
    tag += "_told" if args.told else "_inferred"         # ... and per what a robot knows
    if objects == "inferred":                            # ... (records of before: known)
        tag += "_objects" + ("" if args.max_objects is None else f"{args.max_objects}")
    tag += G.cooperation_tag(cfg.cooperate_with)         # ... (of before: Table I's)
    tag += G.level_tag(cfg.level_free_level)             # ... (of before: a hinderer at 1)
    output = (args.output or ROOT / "output" / f"social_pairings_w{tag}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = [(n, name, goals) for n, (name, goals) in enumerate(PAIRINGS, 1)
             if not args.case or n in args.case]

    def load(name: str, goals=None) -> dict:
        path = output / f"{name}.json"
        if not path.exists():
            raise SystemExit(f"--render-only: {path} does not exist; run without it first")
        result = load_result(path)
        if competition_of(result) != args.competition:
            raise SystemExit(f"--render-only: {path} was written with --competition "
                             f"{competition_of(result)}, not {args.competition}; pass that "
                             "setting, or the folder of this one with --output")
        knew = result.get("config", {}).get("partner_social_goal", "known")
        if knew != cfg.partner_social_goal:
            raise SystemExit(f"--render-only: {path} was written with partner goals {knew}, "
                             f"not {cfg.partner_social_goal}; "
                             + ("pass --told" if knew == "known" else "leave --told out")
                             + ", or the folder of this one with --output")
        if goals is not None:            # the levels this call would roll that record out at
            its = tuple(result["weights"])
            asked = default_levels(goals, cfg, its) if levels is None else levels
            want = levels_used(goals, asked, cfg, its)
            if tuple(result["levels_used"]) != want:
                raise SystemExit(f"--render-only: {path} was rolled out at levels "
                                 f"{tuple(result['levels_used'])}, not {want}; leave "
                                 "--levels out if the records were written without it, "
                                 "pass the levels they were written with, or the folder "
                                 "of these with --output")
        return result

    def draw(name: str, result: dict, title: str) -> None:
        if args.no_gif:
            return
        tic = time.perf_counter()
        render_videos(result, output / name, names=NAMES, fps=args.fps,
                      values=args.values, title=title)
        print(f"    {name}.mp4 ({time.perf_counter() - tic:.0f}s)", flush=True)

    said = (("default (2; a hinderer and a competitor of a race form: 1)"
             if cfg.level_free_level == 1 else "default (2; 0 without a social goal)")
            if levels is None else str(levels))
    print(f"scene: {scene.name} | weights {weights} | levels {said} | competition: "
          f"{args.competition} | partner goals {cfg.partner_social_goal} | "
          f"at most {args.steps} transitions | {output}", flush=True)
    if args.render_only:
        baseline = load("00_baseline")
    else:
        baseline = run_baseline(scene, cfg=cfg, n_steps=args.steps)
        save_result(output / "00_baseline.json", baseline)
    done = " / ".join("never" if t is None else str(t) for t in baseline["done_at"])
    print(f"{'00_baseline':<28} no social goals: done at {done}", flush=True)
    draw("00_baseline", baseline, "no social goals: each robot pursues its physical goal")

    rows = []
    for n, name, goals in cases:
        if args.render_only:
            result = load(name, goals)
        else:
            result = run_pairing(scene, goals, weights=weights, levels=levels, cfg=cfg,
                                 n_steps=args.steps)
            save_result(output / f"{name}.json", result)
        row = summary_row(name, result, baseline)
        rows.append(row)
        print_row(row)
        used = result["levels_used"]
        label = [goal_label(g, competition_of(result), cooperation_of(result))
                 for g in goals]
        draw(name, result, f"{NAMES[0]}: {label[0]}     {NAMES[1]}: {label[1]}     "
                           f"weights {result['weights'][0]:g} / {result['weights'][1]:g}     "
                           f"levels {used[0]} / {used[1]}")
    # the configuration the records were written with (they may be older than this call)
    summary = dict(scene=scene.to_json(),
                   config=baseline.get("config") or dataclasses.asdict(cfg), weights=weights,
                   levels="default" if levels is None else levels,
                   baseline_done_at=baseline["done_at"],
                   pairings=rows)
    (output / "summary.json").write_text(json.dumps(to_json(summary), indent=1) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
