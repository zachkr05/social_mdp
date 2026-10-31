"""The five readings of competition (``Config.competition``) side by side.

Run from the project root:

    python examples/compare_competition.py                   # weights 1 1, the paper's
    python examples/compare_competition.py --weights 30 30
    python examples/compare_competition.py --told            # each robot told its partner's goal

For each of "race_then_hinder", "race", "table", "table_hostile" (Table I with the
competitor's cell against a cooperative partner turned to -R_j) and "rival" (a race against
every partner but a cooperative one) it rolls out

* ``two_competitors``: two competitors on the six-object scene -- does each do its own
  task, or do the two wait for each other?
* ``competitive_cooperative``: a competitor against a cooperator on the six-object scene --
  what does the competitor do once the cooperator has delivered its object for it?
* ``far_room``: two competitors on the two-room map with agent_2's destination moved to the
  far room, so that agent_2 has 8 transitions to go when agent_1 is done -- what does the
  one that is done do about the one that is not?

and writes ``<run>__<reading>.json`` (the record) and ``.mp4`` (the annotated animation, an
MP4 video) to
``output/competition_readings_w<weights>_inferred/``, with ``summary.json``.  A competitor
reasons at its default level: 2 under every reading since 2026-10-11 (folder
``..._level2``); with ``--told``, as until then, 1 under the two race forms and 2 under the
two readings of the table and as a rival.  Nobody is told its partner's social goal
(``Config``'s defaults);
``--told`` tells each robot its partner's, with Table I as printed for an exchanger at
level 1 (the defaults until 2026-10-08, under which the README's comparison was made;
folder ``..._told/``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_pairings import (TOLD, Config, Scene, run_pairing,  # noqa: E402
                             six_object_scene, two_room_scene)
from social_pairings import goals as G  # noqa: E402
from social_pairings.render import goal_label, render_videos  # noqa: E402
from social_pairings.serialize import save_result, to_json  # noqa: E402

NAMES = ("agent_1", "agent_2")
# --told: what each robot knows of its partner as until 2026-10-08
KNOWN = {key: value for key, value in TOLD.items() if key != "competition"}
FAR_ROOM = Scene(**{**two_room_scene().__dict__, "name": "far_room",
                    "destinations": ((7, 1), (5, 5))})
RUNS = (("two_competitors", six_object_scene(), ("competition", "competition")),
        ("competitive_cooperative", six_object_scene(), ("competition", "cooperation")),
        ("far_room", FAR_ROOM, ("competition", "competition")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0),
                        metavar=("AGENT_1", "AGENT_2"))
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
    tag = "_".join(f"{w:g}" for w in weights) + ("_told" if args.told else "_inferred")
    if objects == "inferred":                    # (the records of before: objects known)
        tag += "_objects" + ("" if args.max_objects is None else f"{args.max_objects}")
    probe = Config(**{**(KNOWN if args.told else {}), **knows_objects})
    rule = probe.cooperate_with
    tag += G.cooperation_tag(rule)               # (the records of before: Table I's)
    tag += G.level_tag(probe.level_free_level)   # (... and a hinderer at level 1)
    output = (args.output or ROOT / "output" / f"competition_readings_w{tag}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run, scene, goals in RUNS:
        for reading in G.COMPETITION_FORMS:
            cfg = Config(num_threads=args.threads, competition=reading,
                         **{**(KNOWN if args.told else {}), **knows_objects})
            tic = time.perf_counter()
            result = run_pairing(scene, goals, weights=weights, cfg=cfg, n_steps=args.steps)
            name = f"{run}__{reading}"
            save_result(output / f"{name}.json", result)
            done = " / ".join("never" if t is None else str(t) for t in result["done_at"])
            events = [(t, what, NAMES[by], f"O{obj + 1}") for t, what, by, obj, _ in
                      result["events"]]
            rows.append(dict(name=name, scene=scene.name, goals=result["goals"],
                             competition=reading, weights=weights,
                             levels_used=result["levels_used"], done_at=result["done_at"],
                             stopped_by=result["stopped_by"], steps=result["steps"],
                             holds=result["holds"], events=result["events"],
                             moves=[ledger["moves"] for ledger in result["ledger"]],
                             blocked_by_partner=[ledger["blocked_by_partner"]
                                                 for ledger in result["ledger"]]))
            print(f"{name:<44} done at {done:<13} stopped by {result['stopped_by']:<8} "
                  f"({result['steps']} transitions) | moves {rows[-1]['moves']} | blocked by "
                  f"its partner {rows[-1]['blocked_by_partner']} | {events[:6]}", flush=True)
            if not args.no_gif:
                used = result["levels_used"]
                label = [goal_label(g, reading, rule) for g in goals]
                render_videos(result, output / name, names=NAMES,
                              title=f"{NAMES[0]}: {label[0]}     {NAMES[1]}: {label[1]}     "
                                    f"weights {weights[0]:g} / {weights[1]:g}     "
                                    f"levels {used[0]} / {used[1]}")
                print(f"    {name}.mp4 ({time.perf_counter() - tic:.0f}s)", flush=True)
    (output / "summary.json").write_text(json.dumps(to_json(rows), indent=1) + "\n")
    print(output, flush=True)


if __name__ == "__main__":
    main()
