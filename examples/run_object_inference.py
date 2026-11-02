"""Which object the partner wants, inferred or known: five pairings side by side.

Run from the project root:

    python examples/run_object_inference.py                   # the two-room scene, w = 1 / 1
    python examples/run_object_inference.py --scene spare     # ... one spare object (78 min)
    python examples/run_object_inference.py --scene center --pairing hinder cooperation
                                         # both objects in the doorway, two spare objects
    python examples/run_object_inference.py --scene tiny      # the 3 x 3 map of the tests
    python examples/run_object_inference.py --scene six --max-objects 3   # a spare object too
    python examples/run_object_inference.py --resume          # keep the records already there
    python examples/run_object_inference.py --level-free-level 1   # a hinderer at level 1
    python examples/run_object_inference.py --levels 2 2 --objects inferred   # every robot at
                                         # level 2, the partner's object inferred only
    python examples/run_object_inference.py --levels 2 2 --objects inferred --swap --steal
                                         # ... with swaps on a place and taking from hands
    python examples/run_object_inference.py --levels 2 2 --objects inferred --swap --steal \
        --all-pairings                   # ... all 49 pairings of the seven social goals

Since 2026-10-09 a robot does not know which object its partner wants: its partner's
physical goal is an object and a destination, and its belief starts uniform over every
object in the model, its own included, with every candidate destination
(``Config.partner_object="inferred"``, as in the paper).  Each pairing is rolled out twice:
under the defaults of now, and with the partner's object known (``partner_object="known"``,
the defaults until then) -- weights 1 / 1 (the paper's), social goals inferred either way.

The pairings: cooperation / cooperation, competition / cooperation, competition /
competition, hinder / cooperation, none / none (a competitor is a rival, a robot without a
social goal reasons at level 0, every other robot at level 2: a hinderer too since
2026-10-11, ``--level-free-level 1`` puts it at level 1 as before).

Since 2026-10-11 a level-2 cooperator cooperates only with a partner it takes to be
cooperative, competitive or exchanging (``Config.cooperate_with``); ``--cooperate-with all``
gives Table I's cooperator back, with which every folder written before was made.

Writes to ``output/object_inference_<scene>_w1_1/`` (``..._max<n>`` with ``--max-objects``;
``..._selective`` under the default cooperator of 2026-10-11 and ``..._level2`` with a
hinderer at level 2, its default since then, so that no folder of before is overwritten;
``--resume`` keeps a record only if it was made with the options asked for;
``..._levels<a>_<b>`` with ``--levels``, which puts both robots at the levels given, a robot
without a social goal too; ``..._swap`` and ``..._steal`` with ``--swap`` and ``--steal``;
since 2026-10-16 every folder ends in
``..._hands5``: its robots pick, place, take, give and swap where the robots of the folders
of before had one action, "use", and ``--resume`` keeps no record of those; since 2026-10-17
in ``..._hands5_staged``, two posteriors by the partner's hands, ``--one-posterior`` the
update of before and the folder without the suffix; since 2026-10-18 in
``..._hands5_staged_byhand``: nothing automatic, pick and place by hand, a pick-up of three
transitions costing three moves' worth, the other hand actions two -- ``--pick-steps``,
``--pick-cost`` and ``--hand-cost`` change those and the suffix, ``--automatic-hands`` is the
rule of before and the folder without the suffix; since 2026-10-30 in ``..._eq56``: the
belief update of Eqs. 5-6 as the authors' solver keeps it -- two posteriors, the idle
filter, Eq. 5's transition term, nothing else (``social_pairings.models``), so that no
folder of before, made with the devices of then, is kept by ``--resume``;
``--physical-likelihood level0``, the physical posterior on the partner's level-0 policies
as in the authors' code rather than Eq. 6 as printed, writes to ``..._eq56l0``; since
2026-11-01 in ``..._taskconflict``: a level-2 conflict robot is paid ``-w r(g_j)``, Table I's
entry at l = 1, against every partner (``Config.conflict="task"``), ``--conflict table`` the
table as printed, ``-w R_j``, and the folder without the suffix.  Only a pairing with a
level-2 conflict robot changes with the reading, so ``--resume`` in a ``_taskconflict`` folder
takes every other pairing's record (and its videos) from the folder without the suffix,
where one was made with the options asked for, and rolls out the pairings with conflict
alone):
per rollout ``<pairing>__<objects>.json`` and ``.mp4`` (an MP4 video); ``summary.json``;
``results.md``
(the finishes; what each robot ended up believing of its partner's object, destination and
social goal; acts for the partner in vain; who took whose object) and ``results.png`` (each
robot's belief that its partner wants the partner's own object, transition by transition).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_tier_scenes import believed  # noqa: E402

from social_pairings import (AUTOMATIC_HANDS, Config, Scene, run_pairing,  # noqa: E402
                             six_object_scene, two_room_scene)
from social_pairings import goals as G  # noqa: E402
from social_pairings.config import written_with  # noqa: E402
from social_pairings.render import (ROBOT_COLOURS, goal_label, render_videos,  # noqa: E402
                                    video_paths)
from social_pairings.constants import ACTION_NAMES  # noqa: E402
from social_pairings.serialize import (action_names, load_result, save_result,  # noqa: E402
                                       to_json)

NAMES = ("agent_1", "agent_2")
PAIRINGS = (("cooperation", "cooperation"), ("competition", "cooperation"),
            ("competition", "competition"), ("hinder", "cooperation"), ("none", "none"))
MODES = ("known", "inferred")


def tiny_scene() -> Scene:
    """The 3 x 3 map of the tests (``tests/test_models.py``, TINY), on a 1 m grid."""
    return Scene(grid=np.ones((3, 3)), starts=((0, 1), (2, 1)), pickups=((2, 2), (0, 0)),
                 destinations=((0, 2), (2, 0)), candidate_cells=((1, 0),), name="tiny")


def spare_scene() -> Scene:
    """The two-room scene with one spare object, at (2, 1) in agent_1's room: with the
    partner's object inferred it is a candidate for the partner's goal like the two own ones
    (one of the six-object scene's spares; about 0.9 GiB, ten times the time per transition
    of the two-room scene)."""
    scene = two_room_scene(objects=((2, 1),))
    return Scene(**{**scene.__dict__, "name": "two_rooms_one_spare"})


def center_scene() -> Scene:
    """The two-room map with both robots' objects in the middle, on the two cells of the
    doorway (agent_1's O1 on (4, 2), agent_2's O2 on (4, 3)), and two spare objects, one in
    each room ((2, 1) and (6, 5)): four objects, all candidates for the partner's goal with
    the object inferred (about 1.5 GiB for hinder / cooperation)."""
    base = two_room_scene()
    return Scene(grid=base.grid, starts=base.starts, pickups=((4, 2), (4, 3)),
                 destinations=base.destinations, candidate_cells=base.candidate_cells,
                 objects=((2, 1), (6, 5)), name="two_rooms_center")


def two_spares_scene() -> Scene:
    """The two-room scene as it is (each robot's object next to its partner's start) with two
    spare objects, one in each room, on the center scene's spare cells (2, 1) and (6, 5)
    (2026-10-30): four objects, all candidates for the partner's goal with the object
    inferred.  With swaps and the pick-up of three transitions (the 49's settings) that is
    5,545 object configurations and about 170-190 GiB of solver memory; without swaps 718
    and about 24 GiB; without swaps and with an instant pick-up (--pick-steps 1) 126 and
    about 4 GiB."""
    scene = two_room_scene(objects=((2, 1), (6, 5)))
    return Scene(**{**scene.__dict__, "name": "two_rooms_two_spares"})


SCENES = {"two": (two_room_scene, {}), "spare": (spare_scene, {}),
          "spare2": (two_spares_scene, {}),
          "center": (center_scene, {}), "tiny": (tiny_scene, dict(resolution=1.0)),
          "six": (six_object_scene, {})}


def stem(pair) -> str:
    return f"{pair[0]}__{pair[1]}"


# options a record may differ in and still be the rollout asked for: the machine's
MACHINE = ("num_threads", "max_memory_gib", "verbose")


def unkept(result: dict, cfg: Config, pair, levels) -> Optional[str]:
    """Why a saved record is not the rollout asked for (None: it is).  Its options are
    compared field by field with ``cfg`` (``written_with`` fills in the fields it predates)
    but for the machine's -- and for ``Config.conflict`` where no conflict robot reasons at
    level 2 (``goals.conflict_matters``): the rollout is then the same under either
    reading."""
    mine = to_json(dataclasses.asdict(cfg))
    theirs = written_with(result.get("config") or {})
    loose = set(MACHINE)
    if not G.conflict_matters(result.get("social_goals") or pair, result["levels_used"]):
        loose.add("conflict")
    if any(theirs.get(k) != v for k, v in mine.items() if k not in loose):
        return "was written with another configuration"
    if levels is not None and tuple(result.get("levels") or ()) != levels:
        return "was rolled out at other levels"
    if action_names(result) != ACTION_NAMES:
        return "was rolled out with \"use\" (before 2026-10-16), not the five hand actions"
    return None


def reuse(path: Path, before: Optional[Path], cfg: Config, pair, levels) -> Optional[dict]:
    """The record of ``path``'s rollout in ``before``, the folder of the same options under
    the other reading of conflict, copied here with its videos -- where the reading does
    not change the rollout and the record is the one asked for in every other option
    (:func:`unkept`).  None: there is none to take."""
    old = None if before is None else before / path.name
    if old is None or not old.exists():
        return None
    result = load_result(old)
    if unkept(result, cfg, pair, levels) is not None:
        return None
    for (_, src), (_, dst) in zip(video_paths(result, old.with_suffix(""), names=NAMES),
                                  video_paths(result, path.with_suffix(""), names=NAMES)):
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
    part = path.with_name(path.name + ".part")   # the record last, whole or not at all
    shutil.copy2(old, part)
    os.replace(part, path)
    return result


def row_of(pair, mode: str, result: dict) -> dict:
    """What one rollout did."""
    true_dest = result["scene"]["goal_ids"]
    final = result.get("final_object_belief")
    objects = None if final is None else [round(float(final[i][1 - i]), 3)
                                          if result["levels_used"][i] >= 1 else None
                                          for i in (0, 1)]
    dests = [round(float(result["final_belief"][i][true_dest[1 - i]]), 3)
             if result["levels_used"][i] >= 1 else None for i in (0, 1)]
    took = [dict(t=t, what=what, by=NAMES[by], obj=f"O{obj + 1}")
            for t, what, by, obj, *_ in result["events"]
            if obj is not None and obj != by and what in ("picked up", "handed over",
                                                          "set down", "put down")]
    return dict(pairing=list(pair), objects=mode, done_at=result["done_at"],
                own_goal=[bool(x) for x in result.get("own_goal", (True, True))],
                steps=result["steps"], stopped_by=result["stopped_by"],
                levels_used=result["levels_used"],
                believes_partners_object=objects, believes_partners_destination=dests,
                believes_social=[believed(b) for b in result["final_social_belief"]],
                set_downs_in_vain=[led["failed_set_downs"] for led in result["ledger"]],
                hand_overs_in_vain=[led.get("failed_hand_overs", 0) for led in result["ledger"]],
                took_others_object=took, wall_time_s=round(float(result["wall_time_s"]), 1))


def finish(row: dict) -> str:
    """agent_1 / agent_2: the transition its goal was done, – never, "no goal" for a hinderer
    without a goal of its own."""
    own = row.get("own_goal", (True, True))
    return " / ".join("no goal" if not mine else ("–" if t is None else str(t))
                      for t, mine in zip(row["done_at"], own))


def table(rows: list, title: str) -> str:
    by = {(tuple(r["pairing"]), r["objects"]): r for r in rows}
    known = any(r["objects"] == "known" for r in rows)       # (no column without such runs)
    lines = [f"# {title}", "",
             "Transition at which agent_1 / agent_2 finished its own goal (– = never; no goal: "
             "a hinderer without a goal of its own); with the "
             "objects inferred, each robot's final belief that its partner wants the partner's "
             "own object and that it goes to its true destination (– : a level-0 robot holds "
             "no belief); acts for the partner that did not count (set-downs / hand-overs, "
             "per robot).", "",
             "| pairing | " + ("known | " if known else "") + "inferred | belief: partner's "
             "object | ... its destination | in vain | wall time |",
             "|---|---|---|---|---|---|" + ("---|" if known else "")]
    for pair in dict.fromkeys(tuple(r["pairing"]) for r in rows):
        k, n = by.get((pair, "known")), by.get((pair, "inferred"))
        if n is None:
            continue

        def two(x) -> str:
            return " / ".join("–" if v is None else f"{v:.2f}" for v in x)

        vain = " / ".join(f"{a}+{b}" for a, b in zip(n["set_downs_in_vain"],
                                                      n["hand_overs_in_vain"]))
        stand = lambda r: ", stand-off" if r["stopped_by"] == "deadlock" else ""  # noqa: E731
        lines.append(f"| {pair[0]} / {pair[1]} | "
                     + ("" if not known else ("" if k is None else finish(k) + stand(k)) + " | ")
                     + f"{finish(n)}{stand(n)} | {two(n['believes_partners_object'] or ())} | "
                     f"{two(n['believes_partners_destination'])} | {vain} | "
                     + ("" if k is None else f"{k['wall_time_s']:.0f} s / ")
                     + f"{n['wall_time_s']:.0f} s |")
    return "\n".join(lines) + "\n"


def draw(records: dict, title: str, path: Path) -> None:
    """Each robot's belief that its partner wants the partner's own object, transition by
    transition, one panel per pairing (objects inferred)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, quiet, surface, grid = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
    pairs = [p for p in dict.fromkeys(list(PAIRINGS) + [k[0] for k in records])
             if (p, "inferred") in records]
    # up to eight pairings: one row; more (all 49, say): rows of seven in the order asked
    # for, so that the seven goals of agent_1 by the seven of agent_2 make a square
    cols = len(pairs) if len(pairs) <= 8 else 7
    if cols < len(pairs):
        pairs = [p for p in dict.fromkeys(k[0] for k in records) if (p, "inferred") in records]
    n_rows = -(-len(pairs) // cols)
    height = 3.9 + 3.1 * (n_rows - 1)
    fig, axes = plt.subplots(n_rows, cols, figsize=(3.3 * cols + 0.6, height), dpi=140,
                             facecolor=surface, sharey=True, squeeze=False)
    for ax in axes.flat[len(pairs):]:
        ax.set_visible(False)
    for n, (ax, pair) in enumerate(zip(axes.flat, pairs)):
        r = records[pair, "inferred"]
        ax.set_facecolor(surface)
        m = r["scene"]["modelled_objects"]
        rows = [t["object_belief"] for t in r["trace"]] + [r["final_object_belief"]]
        x = np.arange(len(rows))
        for i, style in ((0, "-"), (1, "--")):
            if r["levels_used"][i] < 1:          # a level-0 robot holds no belief
                continue
            y = np.array([float(b[i][1 - i]) for b in rows])
            ax.plot(x, y, style, color=ROBOT_COLOURS[i], lw=2.0, label=NAMES[i])
            done = r["done_at"][i]
            if done is not None:               # where this robot's own goal was done
                ax.plot([done], [y[min(done, len(y) - 1)]], "o", ms=6, color=ROBOT_COLOURS[i],
                        mec=surface, mew=1.5, zorder=4)
        ax.axhline(1.0 / m, color=quiet, lw=1.0, ls=":")
        if max(r["levels_used"]) < 1:
            ax.text(0.5, 0.62, "level 0: neither robot\nholds a belief", ha="center",
                    va="center", transform=ax.transAxes, fontsize=9, color=quiet)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(0, max(len(rows) - 1, 1))
        ax.grid(axis="y", color=grid, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b3")
        ax.tick_params(colors=quiet, labelsize=8)
        known = records.get((pair, "known"))
        sub = f"done {finish(row_of(pair, 'inferred', r))}" + (
            "" if known is None else f"  (known: {finish(row_of(pair, 'known', known))})")
        rule = r["config"].get("cooperate_with")
        reading = r["config"].get("conflict", "table")
        ax.set_title(f"{goal_label(pair[0], None, rule, reading)} / "
                     f"{goal_label(pair[1], None, rule, reading)}\n{sub}", fontsize=9,
                     color=ink, loc="left")
        if n >= len(pairs) - cols:
            ax.set_xlabel("transition", fontsize=8.5, color=quiet)
    for row in axes:
        row[0].set_ylabel("belief: partner wants its own object", fontsize=8.5, color=quiet)
    handles, labels = axes[0][0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color=quiet, lw=1.0, ls=":"),
                Line2D([], [], marker="o", ls="none", color=quiet, ms=6)]
    labels += ["uniform (1 / objects)", "own goal done"]
    fig.legend(handles, labels, loc="lower center", ncols=4, frameon=False, fontsize=8.5,
               labelcolor=ink)
    fig.suptitle(title, x=0.01, y=1 - 0.02 * 3.9 / height, ha="left", fontsize=11,
                 fontweight="bold", color=ink)
    # the margins of the one-row figure, in inches, whatever the number of rows
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 0.24 * 3.9 / height,
                        bottom=0.25 * 3.9 / height, wspace=0.12, hspace=0.62 * 3.9 / 3.1)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", choices=tuple(SCENES), default="two")
    parser.add_argument("--max-objects", type=int,
                        help="objects in the model, 2..6 (the others are scenery); --scene six "
                             "needs 3 here, as it needs about 13 GiB with all six")
    parser.add_argument("--all-pairings", action="store_true",
                        help="every pairing of the social goals held possible "
                             "(Config.social_hypotheses: 7 x 7 = 49 by default), agent_1's goal "
                             "by agent_2's; results.png puts them in rows of seven")
    parser.add_argument("--pairing", nargs=2, metavar=("AGENT_1", "AGENT_2"), action="append",
                        help="only this pairing, may be given more than once (default: the "
                             "five)")
    parser.add_argument("--steps", type=int, default=60, help="most transitions per rollout")
    parser.add_argument("--output", type=Path, help="output folder")
    parser.add_argument("--resume", action="store_true",
                        help="keep the records (and videos) already in the output folder")
    parser.add_argument("--no-video", "--no-gif", dest="no_gif", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="0 = every CPU available")
    parser.add_argument("--cooperate-with", nargs="+", metavar="GOAL",
                        help="the partners a level-2 cooperator cooperates with "
                             "(Config.cooperate_with): by default cooperation competition "
                             "exchange; 'all' is Table I as printed (the folders of before "
                             "2026-10-11)")
    parser.add_argument("--level-free-level", type=int, choices=(1, 2),
                        help="the level of a hinderer (and of a competitor of a race form) "
                             "when none is asked for, Config.level_free_level: 2 by default "
                             "since 2026-10-11, like every robot with a social goal; 1 as "
                             "before (the folders without _level2)")
    parser.add_argument("--levels", type=int, nargs=2, choices=(0, 1, 2),
                        metavar=("AGENT_1", "AGENT_2"),
                        help="the reasoning level of each robot in every pairing, a robot "
                             "without a social goal included (default: each robot's default "
                             "level -- 2 with a social goal, 0 without one)")
    parser.add_argument("--objects", choices=MODES, action="append",
                        help="roll out only with the partner's object known or inferred; may "
                             "be given twice (default: both)")
    parser.add_argument("--swap", action="store_true",
                        help="Config.swap_on_place: full hands on an object's place swap what "
                             "they hold for what lies there (since 2026-10-12)")
    parser.add_argument("--steal", action="store_true",
                        help="Config.steal: free hands take what the partner next to them "
                             "holds (since 2026-10-12)")
    parser.add_argument("--temperature", type=float, nargs="+", metavar="TAU",
                        help="Config.temperature: tau of Eq. 7's softmax -- the policies a "
                             "robot predicts its partner by, and the likelihood of Eqs. 5-6 "
                             "-- in units of the move cost (default 0.2).  Two values: a tau "
                             "per robot, agent_1's then agent_2's (Config.robot_temperatures; "
                             "folders _tau<agent_1's>_<agent_2's>)")
    parser.add_argument("--foreign-carry-cost", type=float, metavar="FACTOR",
                        help="Config.foreign_carry_cost: what a move costs, in moves, while a "
                             "robot holds an object that is not its own (default 2; 1 is the "
                             "paper, where nothing costs extra; since 2026-10-14)")
    parser.add_argument("--inertia", type=float, metavar="FACTOR",
                        help="Config.inertia: a robot does not undo its last action (step back, "
                             "put back what it just picked up) unless undoing beats every other "
                             "action by more than FACTOR x a move's cost (default 0: none; "
                             "since 2026-10-14)")
    parser.add_argument("--hinder-own-goal", action="store_true",
                        help="Config.hinder_own_goal=True: a hinderer keeps a goal of its own, "
                             "weighed against its partner's (the rule until 2026-10-14; the "
                             "folders without _spoiler).  By default a hinderer has none")
    parser.add_argument("--automatic-hands", action="store_true",
                        help="Config.automatic_hands=True, pick_steps=1 and the costs of before: "
                             "a robot takes its own object on arriving where it lies and sets "
                             "it down on arriving at its destination, for free; a pick-up is "
                             "instant and costs a move, the other hand actions a move (the "
                             "rule until 2026-10-18; the folders without _byhand).  By default "
                             "nothing is automatic: pick and place by hand, a pick-up of "
                             "--pick-steps transitions costing --pick-cost on the first, the "
                             "other hand actions --hand-cost")
    parser.add_argument("--pick-steps", type=int, default=None,
                        help="transitions a pick-up takes (Config.pick_steps; default 3)")
    parser.add_argument("--pick-cost", type=float, default=None,
                        help="c(pick) in moves' worth, on the first transition of a lift "
                             "(Config.pick_cost_rate; default 3)")
    parser.add_argument("--hand-cost", type=float, default=None,
                        help="c(place), c(take), c(give), c(swap) in moves' worth "
                             "(Config.use_cost_rate; default 2)")
    parser.add_argument("--one-posterior", action="store_true",
                        help="Config.belief_stages=False: one posterior over (object, "
                             "destination, social goal), updated by every action of the "
                             "partner (the update until 2026-10-17; the folders without "
                             "_staged).  By default two, by the partner's hands: while it holds "
                             "nothing its actions tell which object it wants, while it holds an "
                             "object where it takes it (and its social goal either way)")
    parser.add_argument("--max-memory-gib", type=float, default=None, metavar="GIB",
                        help="Config.max_memory_gib: the most the solver may use (default 48; "
                             "the machine's memory caps it anyway).  The two-spare scene at the "
                             "49's settings needs about 170-190 GiB")
    parser.add_argument("--verbose", action="store_true",
                        help="Config.verbose: print the model's size, level 0, and a line per "
                             "transition as it happens (for long runs)")
    parser.add_argument("--physical-likelihood", choices=("level1", "level0"), default=None,
                        help="Config.physical_likelihood: the likelihood of Eq. 6.  level1 (the "
                             "default): the partner's level-1 hypothesis policies averaged over "
                             "the social posterior, Eq. 6 as the paper prints it.  level0: the "
                             "partner's level-0 policies, the authors' code (folders ..._eq56l0)")
    parser.add_argument("--conflict", choices=G.CONFLICT_READINGS, default=Config().conflict,
                        help="Config.conflict: what a level-2 conflict robot is paid.  task (the "
                             "default since 2026-11-01): -w r(g_j), Table I's entry at l = 1, "
                             "against every partner -- it does its own task and works against "
                             "its partner's (folders ..._taskconflict).  table: Table I as "
                             "printed, -w R_j (the folders without the suffix)")
    args = parser.parse_args()
    taus = None if args.temperature is None else tuple(args.temperature)
    if taus is not None and len(taus) not in (1, 2):
        parser.error("--temperature takes one tau (both robots) or two (agent_1's, agent_2's)")
    if taus is not None and len(taus) == 2 and taus[0] == taus[1]:
        taus = taus[:1]                    # one tau for both: the folder and the record of one
    modes = tuple(m for m in MODES if m in args.objects) if args.objects else MODES
    levels = None if args.levels is None else tuple(args.levels)
    make, options = SCENES[args.scene]
    scene = make()
    if args.max_objects is not None:
        options = dict(options, max_objects=args.max_objects)
    if args.cooperate_with:
        options = dict(options, cooperate_with=tuple(args.cooperate_with))
    if args.level_free_level is not None:
        options = dict(options, level_free_level=args.level_free_level)
    if args.swap:
        options = dict(options, swap_on_place=True)
    if args.steal:
        options = dict(options, steal=True)
    carry = args.foreign_carry_cost
    if carry is not None and carry == Config().foreign_carry_cost:
        carry = None                       # the default: the folders of before keep their name
    if carry is not None:
        options = dict(options, foreign_carry_cost=carry)
    if args.hinder_own_goal:
        options = dict(options, hinder_own_goal=True)
    inertia = args.inertia if args.inertia else None      # 0: none, the folders of before
    if inertia is not None:
        options = dict(options, inertia=inertia)
    if args.one_posterior:
        options = dict(options, belief_stages=False)
    if taus is not None and len(taus) == 1:
        options = dict(options, temperature=taus[0])
    elif taus is not None:                 # a tau per robot (2026-11-01)
        options = dict(options, robot_temperatures=taus)
    if args.physical_likelihood is not None:
        options = dict(options, physical_likelihood=args.physical_likelihood)
    options = dict(options, conflict=args.conflict)        # (2026-11-01: summary.json says it)
    if args.max_memory_gib is not None:         # (not in the folder name: same model)
        options = dict(options, max_memory_gib=args.max_memory_gib)
    if args.verbose:
        options = dict(options, verbose=True)
    if args.automatic_hands:
        options = dict(options, **AUTOMATIC_HANDS)
    for key, value in (("pick_steps", args.pick_steps), ("pick_cost_rate", args.pick_cost),
                       ("use_cost_rate", args.hand_cost)):
        if value is not None:
            options = dict(options, **{key: value})
    hands = Config(**options)
    byhand = "" if hands.automatic_hands else (
        "_byhand" + ("" if (hands.pick_steps, hands.pick_cost_rate, hands.use_cost_rate)
                     == (3, 3.0, 2.0)
                     else f"_pick{hands.pick_steps}x{hands.pick_cost_rate:g}"
                          f"_hand{hands.use_cost_rate:g}"))
    staged = Config(**options).belief_stages
    spoiler = not Config(**options).hinder_own_goal
    rule = Config(**options).cooperate_with
    level_free_level = Config(**options).level_free_level
    tag = (f"{args.scene}_w1_1" + ("" if args.max_objects is None else f"_max{args.max_objects}")
           + G.cooperation_tag(rule) + G.level_tag(level_free_level)
           + ("" if levels is None else f"_levels{levels[0]}_{levels[1]}")
           + ("_swap" if args.swap else "") + ("_steal" if args.steal else "")
           + ("" if taus is None else "_tau" + "_".join(f"{t:g}" for t in taus))
           + ("" if carry is None else f"_carry{carry:g}")
           + ("_spoiler" if spoiler else "")
           + ("" if inertia is None else f"_inertia{inertia:g}")
           + "_hands5"           # the five hand actions (since 2026-10-16), not "use"
           + ("_staged" if staged else "")       # two posteriors by the hands (2026-10-17)
           + byhand              # nothing automatic, a pick-up of three (2026-10-18)
           + "_eq56"             # Eqs. 5-6 as the authors' solver keeps them (2026-10-30)
           + ("l0" if Config(**options).physical_likelihood == "level0" else ""))
    #        (... with the physical posterior on the level-0 policies, the authors' code)
    task = Config(**options).conflict == "task"
    # a level-2 conflict robot paid -w r(g_j) against every partner (2026-11-01); the folder
    # of the same options under the table's -w R_j, where every other pairing is the same
    before = (ROOT / "output" / f"object_inference_{tag}").resolve() if task else None
    if args.output is not None:
        before = None                       # (a folder named by hand has no folder of before)
    tag += "_taskconflict" if task else ""
    out = (args.output or ROOT / "output" / f"object_inference_{tag}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    records, rows = {}, []
    if args.all_pairings and args.pairing:
        parser.error("--all-pairings or --pairing, not both")
    goals = Config(**options).social_hypotheses
    pairs = ([(a, b) for a in goals for b in goals] if args.all_pairings
             else [tuple(p) for p in args.pairing] if args.pairing else list(PAIRINGS))
    for pair in pairs:
        for mode in modes:
            cfg = Config(num_threads=args.threads, partner_object=mode, **options)
            path = out / f"{stem(pair)}__{mode}.json"
            result, taken = None, False
            if args.resume and path.exists():
                result = load_result(path)
                why = unkept(result, cfg, pair, levels)
                if why is not None:
                    raise SystemExit(f"--resume: {path} {why}")
            elif args.resume:                # no conflict robot at level 2: as it was before
                result = reuse(path, before, cfg, pair, levels)
                taken = result is not None
            if result is None:
                result = run_pairing(scene, pair, levels=levels, cfg=cfg, n_steps=args.steps)
                save_result(path, result)
            records[pair, mode] = result
            row = row_of(pair, mode, result)
            rows.append(row)
            print(f"{pair[0]:>12} / {pair[1]:<12} objects {mode:<8} done at {finish(row):<9} "
                  f"({result['stopped_by']}, {result['steps']} transitions, "
                  f"{row['wall_time_s']:.0f} s)"
                  + (f"  [taken from {before.name}: no conflict robot at level 2]"
                     if taken else ""), flush=True)
            # with --resume, drawn again unless all of its videos are there (whole: a video
            # is renamed to its name only once written to the end)
            videos = [v for _, v in video_paths(result, path.with_suffix(""), names=NAMES)]
            if not args.no_gif and not (args.resume and all(v.exists() for v in videos)):
                used = result["levels_used"]
                render_videos(result, path.with_suffix(""), names=NAMES, fps=3.0,
                              title=f"{scene.name}: objects {mode}     {NAMES[0]}: "
                                    f"{goal_label(pair[0], cfg.competition, rule, cfg.conflict)}"
                                    f"     {NAMES[1]}: "
                                    f"{goal_label(pair[1], cfg.competition, rule, cfg.conflict)}"
                                    f"     w 1 / 1     "
                                    f"levels {used[0]} / {used[1]}"
                                    + "".join(f"     {what}" for flag, what in
                                              ((cfg.swap_on_place, "swap"), (cfg.steal, "steal"))
                                              if flag)
                                    + ("" if taus is None else "     tau "
                                       + " / ".join(f"{t:g}" for t in taus))
                                    + ("" if carry is None else f"     carry {carry:g}")
                                    + ("     hinder: no own goal" if spoiler else "")
                                    + ("" if inertia is None else f"     inertia {inertia:g}")
                                    + ("     two posteriors" if staged else ""))
    title = (f"Which object the partner wants, inferred or known: {scene.name} scene, w = 1 / 1"
             + ("" if args.max_objects is None else f", {args.max_objects} objects in the model")
             + ("" if G.cooperation_reading(rule) == "Table I"
                else f", cooperation {G.cooperation_reading(rule)}")
             + ("" if levels is not None or not any("hinder" in pair for pair in pairs)
                else f", a hinderer at level {level_free_level}")
             + ("" if levels is None else f", every robot at levels {levels[0]} / {levels[1]}")
             + "".join(f", {what}" for flag, what in ((args.swap, "swaps on a place"),
                                                     (args.steal, "taking from hands"))
                       if flag)
             + ("" if taus is None else f", Eq. 7's tau {taus[0]:g}" if len(taus) == 1
                else f", Eq. 7's tau {taus[0]:g} ({NAMES[0]}) / {taus[1]:g} ({NAMES[1]})")
             + ("" if carry is None else
                f", a move with the partner's object costs {carry:g} x a move")
             + (", a hinderer without a goal of its own" if spoiler else "")
             + ("" if not any(G.parse_goal(g) == G.CONFLICT for pair in pairs for g in pair)
                else ", conflict: -w r(g_j) at every level (task)" if task
                else ", conflict: -w R_j above level 1 (Table I)")
             + ("" if inertia is None else
                f", inertia: no undoing the last action unless it is worth {inertia:g} x a "
                f"move's cost more than the next best")
             + (", two posteriors by the partner's hands (its object while it holds nothing, "
                "its destination while it holds an object)" if staged else ""))
    (out / "results.md").write_text(table(rows, title))
    draw(records, title, out / "results.png")
    summary = dict(scene=scene.to_json(), options=options, weights=(1.0, 1.0),
                   levels=levels or "default", rows=rows)
    (out / "summary.json").write_text(json.dumps(to_json(summary), indent=1) + "\n")
    print(table(rows, title))
    print(out)


if __name__ == "__main__":
    main()
