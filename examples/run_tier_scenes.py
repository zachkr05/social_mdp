"""The Social MDP on the twelve fair congruence tier scenes (``examples/tier_scenes.py``).

Run from the project root:

    python examples/run_tier_scenes.py                       # every scene, its suite's goals
    python examples/run_tier_scenes.py --goals cooperation   # only these goals (and the baselines)
    python examples/run_tier_scenes.py --scenes near_miss head_on_corridor
    python examples/run_tier_scenes.py --resume              # keep the records already there

Under the defaults of now: weights 1 / 1 (the paper's), each robot infers its partner's
social goal and destination, may take its partner for idle, a competitor is a rival, and a
robot without social goal reasons at level 0.  Each scene is rolled out with

  - two level-0 robots without social goals (the baseline, ``run_baseline``), and
  - its suite's goals, both robots with the same one: cooperation, competition, conflict and
    coercion on the meeting suite, exchange on the exchange suite (test_tiers' categories).

Each robot's finish is set against its A* task length (start -> own object -> own
destination, alone on the map): the scenes are fair, so the two lengths are within 10 %.

Writes to ``output/tier_scenes_w1_1_rival_inferred_idle/``: per rollout
``<scene>__<goal>__<goal>.json`` (``<scene>__00_baseline.json``) and ``.mp4`` (an MP4 video);
``summary.json`` (one row per rollout: finishes, lateness against A*, how it ended, who
touched whose object, blocked moves, what each robot ended up believing its partner's goal
to be); ``results.md`` and ``results.png`` (the finishes, scene by row, goal by column).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pairings import NAMES, competition_of, cooperation_of  # noqa: E402
from tier_scenes import (PLACEMENTS, SCENES, SUITE_GOALS, TIERS, Graph, fairness,  # noqa: E402
                         house_free, tier, tier_scene, cells_of, RESOLUTION)

from social_pairings import Config, run_baseline, run_pairing  # noqa: E402
from social_pairings import goals as G  # noqa: E402
from social_pairings.config import written_with  # noqa: E402
from social_pairings.render import goal_label, render_videos  # noqa: E402
from social_pairings.constants import ACTION_NAMES  # noqa: E402
from social_pairings.serialize import (action_names, load_result, save_result,  # noqa: E402
                                       to_json)

GOALS = tuple(G.SOCIAL_GOALS)                      # the order of every social belief
COLUMNS = ("baseline", "cooperation", "competition", "conflict", "coercion", "exchange")


def believed(belief) -> tuple:
    """The goal a robot ranks first for its partner (conflict and hinder are one model at
    level 1, so they count as one) and its belief in it."""
    if belief is None:
        return None
    b = {g: float(belief[k]) for k, g in enumerate(GOALS)}
    b["conflict / hinder"] = b.pop("conflict") + b.pop("hinder")
    top = max(b, key=b.get)
    return top, round(b[top], 3)


def row_of(name: str, goal: str, result: dict, L: list) -> dict:
    """What one rollout did."""
    done = result["done_at"]
    late = [None if t is None else t - l for t, l in zip(done, L)]
    touched = [dict(t=t, what=what, by=NAMES[by], obj=obj)
               for t, what, by, obj, *_ in result["events"]
               if obj is not None and obj != by and what in ("picked up", "put back", "set down",
                                                              "set down in vain", "handed over",
                                                              "put down")]
    return dict(scene=name, tier=PLACEMENTS[name][0], suite=PLACEMENTS[name][1], goal=goal,
                astar=L, done_at=done, late=late, steps=result["steps"],
                stopped_by=result["stopped_by"], levels_used=result["levels_used"],
                blocked_by_partner=[led["blocked_by_partner"] for led in result["ledger"]],
                touched_partners_object=touched,
                believes=[believed(b) for b in result["final_social_belief"]],
                deadlock=result.get("deadlock"),
                numerically_accepted=result["numerically_accepted"],
                wall_time_s=result["wall_time_s"])


def cell_text(row: dict) -> str:
    return " / ".join("–" if t is None else str(t) for t in row["done_at"])


def outcome(row: dict) -> str:
    a, b = (t is not None for t in row["done_at"])
    return "both" if a and b else "first" if a else "second" if b else "neither"


def table(rows: list, title: str) -> str:
    """The finishes as Markdown: scene by row, goal by column."""
    by = {(r["scene"], r["goal"]): r for r in rows}
    scenes = list(dict.fromkeys(r["scene"] for r in rows))
    lines = [f"# {title}", "",
             "Transition at which R1 / R2 finished its own goal (– = never, at most 60 "
             "transitions); in brackets, how many transitions after its A* task length. "
             "Both robots have the column's goal; the baseline is two level-0 robots without "
             "social goals.", "",
             "| scene | tier | A* | " + " | ".join(COLUMNS) + " |",
             "|---|---|---|" + "---|" * len(COLUMNS)]
    for s in scenes:
        cells = []
        for c in COLUMNS:
            r = by.get((s, c))
            if r is None:
                cells.append("")
                continue
            late = "/".join("–" if x is None else f"+{x}" for x in r["late"])
            stand = ", stand-off" if r["stopped_by"] == "deadlock" else ""
            cells.append(f"{cell_text(r)} ({late}{stand})")
        r0 = by[s, "baseline"]
        lines.append(f"| {s} | {r0['tier']} ({r0['suite']}) | {r0['astar'][0]} / "
                     f"{r0['astar'][1]} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def draw(rows: list, title: str, path: Path) -> None:
    """The finishes as a figure, each cell filled by who finished."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    ink, quiet, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fill = {"both": "#cde2fb", "first": "#ffd8b0", "second": "#f3b7b7", "neither": "#d6d5d0"}
    label = {"both": "both goals done", "first": "only R1's goal", "second": "only R2's goal",
             "neither": "neither"}
    head = {"baseline": "no social goal\n(level 0)", "cooperation": "cooperation",
            "competition": "competition\n(rival)", "conflict": "conflict",
            "coercion": "coercion", "exchange": "exchange"}
    by = {(r["scene"], r["goal"]): r for r in rows}
    scenes = list(dict.fromkeys((r["suite"], r["tier"], r["scene"], tuple(r["astar"]))
                                for r in rows))
    fig, ax = plt.subplots(figsize=(14.5, 11.0), dpi=140, facecolor=surface)
    ax.set_facecolor(surface)
    n, m = len(scenes), len(COLUMNS)
    for i, (suite, t, s, L) in enumerate(scenes):
        for j, goal in enumerate(COLUMNS):
            r = by.get((s, goal))
            if r is None:
                continue
            ax.add_patch(Rectangle((j + 0.03, i + 0.05), 0.94, 0.90, lw=0,
                                   facecolor=fill[outcome(r)]))
            ax.text(j + 0.5, i + 0.40, cell_text(r), ha="center", va="center", fontsize=12.5,
                    fontweight="bold", color=ink)
            note = []
            if any(x is not None for x in r["late"]):
                note.append("late " + " / ".join("–" if x is None else f"+{x}"
                                                 for x in r["late"]))
            if r["stopped_by"] == "deadlock":
                note.append("stand-off")
            if r["touched_partners_object"]:
                note.append("took other's object")
            ax.text(j + 0.5, i + 0.74, ",  ".join(note), ha="center", va="center",
                    fontsize=7.0, color=quiet)
        ax.text(-0.08, i + 0.36, s, ha="right", va="center", fontsize=10.5, color=ink)
        ax.text(-0.08, i + 0.70, f"{t}  ·  A* {L[0]} / {L[1]}", ha="right", va="center",
                fontsize=8.5, color=quiet)
    k = sum(1 for x in scenes if x[0] == "meeting")
    if 0 < k < n:
        ax.axhline(k, color="0.55", lw=1.0, xmin=-0.3, clip_on=False)
    for lo, hi, name in ((0, k, "meeting suite"), (k, n, "exchange suite")):
        if hi > lo:
            ax.text(-2.25, (lo + hi) / 2, name, rotation=90, ha="center", va="center",
                    fontsize=11, color=quiet, fontweight="bold")
    ax.set_xlim(0, m)
    ax.set_ylim(n, 0)
    ax.set_xticks([j + 0.5 for j in range(m)], [head[c] for c in COLUMNS], fontsize=10,
                  color=ink)
    ax.xaxis.tick_top()
    ax.set_yticks([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.suptitle(title, x=0.015, y=0.99, ha="left", fontsize=14, fontweight="bold", color=ink)
    fig.text(0.015, 0.955, "Transition at which R1 / R2 finished its own goal (– = never, at "
             "most 60); late: transitions after its A* task length. Both robots have the "
             "column's goal.", fontsize=10, color=quiet)
    present = {outcome(r) for r in rows}
    ax.legend(handles=[Patch(facecolor=fill[x], label=label[x]) for x in fill if x in present],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10,
              title="finished", title_fontsize=10, alignment="left")
    fig.subplots_adjust(left=0.24, right=0.86, top=0.86, bottom=0.02)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scenes", nargs="+", choices=tuple(SCENES), default=tuple(SCENES))
    parser.add_argument("--goals", nargs="+", default=("cooperation", "competition", "conflict",
                                                       "coercion", "exchange"),
                        choices=("cooperation", "competition", "conflict", "coercion",
                                 "exchange"))
    parser.add_argument("--steps", type=int, default=60, help="most transitions per rollout")
    parser.add_argument("--output", type=Path,
                        help="output folder (default: output/tier_scenes_w1_1_rival_inferred_"
                             "idle, with _objects where the objects are inferred)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the records (and videos) already in the output folder")
    parser.add_argument("--no-video", "--no-gif", dest="no_gif", action="store_true")
    parser.add_argument("--threads", type=int, default=0, help="0 = every CPU available")
    parser.add_argument("--objects", choices=("inferred", "known"), default="inferred",
                        help="which object a robot's partner wants: inferred over every object "
                             "of the scene (the default since 2026-10-09, as in the paper) or "
                             "known, its own (the records of 2026-10-09)")
    parser.add_argument("--cooperate-with", nargs="+", metavar="GOAL",
                        help="the partners a level-2 cooperator cooperates with "
                             "(Config.cooperate_with): by default cooperation competition "
                             "exchange; 'all' is Table I as printed (the records of before "
                             "2026-10-11)")
    args = parser.parse_args()
    cfg = Config(num_threads=args.threads, partner_object=args.objects,
                 **({"cooperate_with": tuple(args.cooperate_with)} if args.cooperate_with
                    else {}))
    out = (args.output or ROOT / "output" / ("tier_scenes_w1_1_rival_inferred_idle"
                                              + ("_objects" if args.objects == "inferred"
                                                 else "")
                                              + G.cooperation_tag(cfg.cooperate_with)
                                              + G.level_tag(cfg.level_free_level))
           ).resolve()
    out.mkdir(parents=True, exist_ok=True)
    order = {t: k for k, t in enumerate(TIERS)}
    scenes = sorted(args.scenes, key=lambda s: (PLACEMENTS[s][1] != "meeting",
                                                order[PLACEMENTS[s][0]], s))
    free, graph = house_free(), Graph(house_free())

    def kept(path: Path, name: str):
        if not (args.resume and path.exists()):
            return None
        result = load_result(path)
        mine = to_json(dataclasses.asdict(cfg))
        theirs = written_with(result.get("config") or {})
        if any(theirs.get(k) != v for k, v in mine.items()
               if k not in ("num_threads", "belief_detail")):     # (not the model)
            raise SystemExit(f"--resume: {path} was written with another configuration")
        scene = to_json(tier_scene(name).to_json())
        if any(result["scene"].get(k) != v for k, v in scene.items()):
            raise SystemExit(f"--resume: {path} was rolled out on another scene")
        if action_names(result) != ACTION_NAMES:
            raise SystemExit(f"--resume: {path} was rolled out with \"use\" (before "
                             "2026-10-16), not the five hand actions")
        return result

    rows = []
    for name in scenes:
        L = fairness(free, cells_of(name))["L"]
        t, _ = tier(graph, cells_of(name), PLACEMENTS[name][1], RESOLUTION)
        assert t == PLACEMENTS[name][0], name
        todo = [("baseline", "00_baseline")] + [
            (g, f"{g}__{g}") for g in args.goals if g in SUITE_GOALS[PLACEMENTS[name][1]]]
        for goal, stem in todo:
            path = out / f"{name}__{stem}.json"
            result = kept(path, name)
            if result is None:
                if goal == "baseline":
                    result = run_baseline(tier_scene(name), cfg=cfg, n_steps=args.steps)
                else:
                    result = run_pairing(tier_scene(name), (goal, goal), cfg=cfg,
                                         n_steps=args.steps)
                save_result(path, result)
            row = row_of(name, goal, result, L)
            rows.append(row)
            print(f"{name:<26} {row['tier']:<6} {goal:<12} A* {L[0]:>2} / {L[1]:<2} done at "
                  f"{cell_text(row):<9} ({row['stopped_by']}, {row['steps']} transitions)",
                  flush=True)
            video = path.with_suffix(".mp4")
            if goal != "baseline" and not args.no_gif and not (args.resume and video.exists()):
                used = result["levels_used"]
                label = goal_label(goal, competition_of(result), cooperation_of(result))
                render_videos(result, path.with_suffix(""), names=NAMES, fps=3.0,
                              title=f"{name} ({row['tier']} congruence, fair: A* {L[0]} / {L[1]} "
                                    f"steps)     {NAMES[0]}: {label}     {NAMES[1]}: {label}     "
                                    f"w 1 / 1     levels {used[0]} / {used[1]}")
    title = ("The Social MDP on the fair tier scenes (house map, 0.5 m), w = 1, partner goals "
             "inferred" + (", objects too" if args.objects == "inferred" else ""))
    (out / "results.md").write_text(table(rows, title))
    draw(rows, title, out / "results.png")
    summary = dict(config=dataclasses.asdict(cfg), weights=(1.0, 1.0),
                   scenes={s: tier_scene(s).to_json() for s in scenes}, rows=rows)
    (out / "summary.json").write_text(json.dumps(to_json(summary), indent=1) + "\n")
    print(table(rows, title))
    print(out)


if __name__ == "__main__":
    main()
