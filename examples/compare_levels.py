"""A competitor at level 1 and at level 2, side by side.

Run from the project root:

    python examples/compare_levels.py                   # weights 1 1, the paper's
    python examples/compare_levels.py --weights 30 30
    python examples/compare_levels.py --competition race
    python examples/compare_levels.py --told            # each robot told its partner's goal

A competitor of a race form (``Config.competition="race_then_hinder"``, this script's
default, or ``"race"``) reasoned at level 1 unless level 2 was asked for, until 2026-10-11;
since then it reasons at level 2 like every robot with a social goal
(``Config.level_free_level``; 1 with ``--told``).  Its social term is the same at both
levels; what changes is whom it takes its partner to be.  At level 1 its partner pursues a
physical goal and nothing else.  At level 2 its partner has its social goal: the competitor
expects a cooperator to deliver for it, a hinderer not to let go, and another competitor to
turn on it once done -- and acts on that before its own goal is done.  Four runs show the
difference, each with the competitors at level 1 (every other robot at its default level:
a cooperator at 2, a hinderer at ``Config.level_free_level``) and with ``levels=(2, 2)``:

* ``competitive_cooperative``: a competitor against a cooperator on the six-object scene.
* ``hinder_competitive``: a competitor whose object a hinderer takes.
* ``far_room``: two competitors on the two-room map with agent_2's destination in the far
  room, so that agent_2 is 8 transitions behind.
* ``one_cell_door``: two competitors on the two-room map with a doorway one cell wide.

In the runs of 2026-10-06 (each robot told its partner's goal), at weights 30 30 the
first run ended the same at both levels (the cooperator's object taken at transition 6;
the level-2 competitor waited where the level-1 one walked toward its own object and
back), and at weights 1 1 the level-2 competitor took it at transition 2, before its own
goal was done.  The default weights are 1 1, the paper's, since 2026-10-08.

Nobody is told its partner's social goal (``Config``'s defaults): a level-2 robot infers it
from what its partner does.  ``--told`` tells each robot its partner's, with Table I as
printed for an exchanger at level 1 (the defaults until 2026-10-08, under which the
README's comparison and the numbers above were made).

It writes ``<run>__level<n>.json`` (the record) and ``.mp4`` (the annotated animation, an
MP4 video), n the
level the competitors reason at, to ``output/competitor_levels_w<weights>_inferred/``
(``_told`` with ``--told``), with ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_pairings import (TOLD, Config, Scene, default_levels, run_pairing,  # noqa: E402
                             six_object_scene, two_room_scene)
from social_pairings import goals as G  # noqa: E402
from social_pairings.render import goal_label, render_videos  # noqa: E402
from social_pairings.serialize import save_result, to_json  # noqa: E402

NAMES = ("agent_1", "agent_2")


def _variant(name: str, **changes) -> Scene:
    return Scene(**{**two_room_scene().__dict__, "name": name, **changes})


def _one_cell_door() -> Scene:
    grid = two_room_scene().grid.copy()
    grid[4, 3] = np.inf                  # the upper cell of the doorway: (4, 2) is left
    return _variant("one_cell_door", grid=grid)


FAR_ROOM = _variant("far_room", destinations=((7, 1), (5, 5)))
ONE_CELL_DOOR = _one_cell_door()
RUNS = (("competitive_cooperative", six_object_scene(), ("competition", "cooperation")),
        ("hinder_competitive", six_object_scene(), ("hinder", "competition")),
        ("far_room", FAR_ROOM, ("competition", "competition")),
        ("one_cell_door", ONE_CELL_DOOR, ("competition", "competition")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0),
                        metavar=("AGENT_1", "AGENT_2"))
    parser.add_argument("--competition", choices=G.COMPETITION_FORMS[:2],
                        default=G.COMPETITION_FORMS[0],
                        help="the race form the competitors follow")
    parser.add_argument("--steps", type=int, default=60, help="most transitions per rollout")
    parser.add_argument("--output", type=Path, help="output folder")
    parser.add_argument("--no-video", "--no-gif", dest="no_gif", action="store_true",
                        help="records and summary only")
    parser.add_argument("--threads", type=int, default=0, help="0 = every CPU available")
    parser.add_argument("--told", action="store_true",
                        help="each robot is told its partner's social goal (the defaults until "
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
    args = parser.parse_args()
    objects = args.objects or ("known" if args.told else "inferred")
    knows_objects = dict(partner_object=objects)
    if args.max_objects is not None:
        knows_objects["max_objects"] = args.max_objects
    if args.cooperate_with:
        knows_objects["cooperate_with"] = tuple(args.cooperate_with)

    weights = tuple(args.weights)
    if min(weights) <= 0:            # with weight 0 a competitor is none, at level 2 already
        parser.error("--weights must be > 0")
    tag = "_".join(f"{w:g}" for w in weights)
    if args.competition != G.COMPETITION_FORMS[0]:
        tag += f"_{args.competition}"
    tag += "_told" if args.told else "_inferred"
    if objects == "inferred":                    # (the records of before: objects known)
        tag += "_objects" + ("" if args.max_objects is None else f"{args.max_objects}")
    known = {key: value for key, value in TOLD.items() if key != "competition"}
    cfg = Config(num_threads=args.threads, competition=args.competition,
                 **{**(known if args.told else {}), **knows_objects})
    tag += G.cooperation_tag(cfg.cooperate_with)         # (the records of before: Table I's)
    tag += G.level_tag(cfg.level_free_level)             # (... and a hinderer at level 1)
    output = (args.output or ROOT / "output" / f"competitor_levels_w{tag}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, scene, goals in RUNS:
        # the competitors at level 1 and everybody else at its default level (until
        # 2026-10-11 these were the defaults), then everybody at level 2
        at_one = tuple(1 if G.parse_goal(goal) == G.COMPETITION else level
                       for goal, level in zip(goals, default_levels(goals, cfg, weights)))
        for level, levels in ((1, at_one), (2, (2, 2))):
            tic = time.perf_counter()
            result = run_pairing(scene, goals, weights=weights, levels=levels, cfg=cfg,
                                 n_steps=args.steps)
            used = result["levels_used"]
            name = f"{run}__level{level}"
            save_result(output / f"{name}.json", result)
            done = " / ".join("never" if t is None else str(t) for t in result["done_at"])
            events = [(t, what, NAMES[by], f"O{obj + 1}") for t, what, by, obj, _ in
                      result["events"]]
            rows.append(dict(name=name, scene=scene.name, goals=result["goals"],
                             competition=args.competition, weights=weights,
                             levels_used=used, done_at=result["done_at"],
                             stopped_by=result["stopped_by"], steps=result["steps"],
                             holds=result["holds"], events=result["events"],
                             moves=[ledger["moves"] for ledger in result["ledger"]],
                             hand_actions=[ledger["hand_actions"]
                                           for ledger in result["ledger"]],
                             blocked_by_partner=[ledger["blocked_by_partner"]
                                                 for ledger in result["ledger"]]))
            print(f"{name:<34} levels {used[0]}/{used[1]}  done at {done:<13} stopped by "
                  f"{result['stopped_by']:<8} ({result['steps']} transitions) | moves "
                  f"{rows[-1]['moves']} | hand actions {rows[-1]['hand_actions']} | blocked by "
                  f"its partner {rows[-1]['blocked_by_partner']} | {events[:6]}", flush=True)
            if not args.no_gif:
                label = [goal_label(g, args.competition, cfg.cooperate_with) for g in goals]
                render_videos(result, output / name, names=NAMES,
                              title=f"{NAMES[0]}: {label[0]}     {NAMES[1]}: {label[1]}     "
                                    f"weights {weights[0]:g} / {weights[1]:g}     "
                                    f"levels {used[0]} / {used[1]}")
                print(f"    {name}.mp4 ({time.perf_counter() - tic:.0f}s)", flush=True)
    (output / "summary.json").write_text(json.dumps(to_json(rows), indent=1) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
