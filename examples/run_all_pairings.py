"""Every pairing of social goals on the six-object scene: seven goals in either seat, 49 rollouts.

Run from the project root:

    python examples/run_all_pairings.py                          # weights 1 1 (the paper's), rival
    python examples/run_all_pairings.py --weights 30 30
    python examples/run_all_pairings.py --competition table      # another reading of competition
    python examples/run_all_pairings.py --no-video                # records and tables only
    python examples/run_all_pairings.py --scene exchange         # the exchange scene
    python examples/run_all_pairings.py --scene exchange --hand-over
    python examples/run_all_pairings.py --told                   # told: as until 2026-10-08
    python examples/run_all_pairings.py --told --competition race --only-competition

The goals are none, cooperation, exchange, coercion, competition, conflict and hinder; a
pairing is (agent_1's goal, agent_2's goal), so each unordered pair is rolled out with the
seats swapped as well.  Each robot reasons at its default level: 0 without a social goal, 2
with one -- a hinderer and a competitor of a race form included since 2026-10-11
(``--level-free-level 1``: those two at level 1, as before; also with ``--told``).

Nobody is told its partner's social goal (``Config``'s defaults): each level-2 robot starts
from a uniform prior over the seven goals and updates it from what its partner does, and a
level-1 exchanger -- the model by which one is recognised -- is the copy of its partner's
goal of the paper's text (``Config.exchange_level1="copy_unweighted"``).  ``--told`` tells
each robot its partner's social goal, with Table I as printed for an exchanger at level 1:
the defaults until 2026-10-08, under which the README's tables marked "told" were made.
``--exchange-level1`` changes what an exchanger is at level 1 either way.
``--only-competition`` rolls out the 13 pairings with a competitor in them, the only ones a
reading of competition can change where each robot is told its partner's goal (where goals
are inferred, a competitor is one of the models every level-2 robot judges its partner by,
and a reading can change any pairing).

``--scene exchange`` rolls the same pairings out on :func:`social_pairings.exchange_scene`
(each robot in its own room, its object in the other one, a hand-off cell in the doorway)
under the rule of ``examples/run_exchange_scene.py``: nobody finishes its partner's task for
it, and an object changes hands by being put down in the doorway and picked up there.  With
``--hand-over`` a robot next to the doorway may also hand an object to its partner standing
in it.  Nobody can finish alone there.

For every pairing it writes ``<agent_1's goal>__<agent_2's goal>.json`` (the full record) and
``.mp4`` (the annotated animation, an MP4 video) to
``output/all_pairings_w<weights>_<reading>_inferred/``
(``_told`` with ``--told``; ``all_pairings_exchange[_hand_over]_w...`` on the exchange scene;
with ``level1_<reading>_`` before the weights where ``--exchange-level1`` changed it), with
``summary.json`` (one row per pairing: finishes, ending, who took whose object, blocked
moves, the solver's verdict, and a hindrance report where one is due), ``matrix.md`` (the
finishes as a table, agent_1's goal by row and agent_2's by column) and ``matrix.png`` (the
same table as a figure, each cell filled by who finished).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pairings import (NAMES, competition_of, cooperation_of, print_row,  # noqa: E402
                          summary_row)

from social_pairings import (TOLD, Config, exchange_scene, run_baseline,  # noqa: E402
                             run_pairing, six_object_scene)
from social_pairings import goals as G  # noqa: E402
from social_pairings.config import written_with  # noqa: E402
from social_pairings.render import goal_label, render_videos  # noqa: E402
from social_pairings.constants import ACTION_NAMES  # noqa: E402
from social_pairings.serialize import (action_names, load_result, save_result,  # noqa: E402
                                       to_json)

# from the goal that asks least of a partner to the one that works against it hardest
GOALS = ("none", "cooperation", "exchange", "coercion", "competition", "conflict", "hinder")
# --told: what each robot knows of its partner as until 2026-10-08 (the reading of competition
# is --competition's)
KNOWN = {key: value for key, value in TOLD.items() if key != "competition"}


def conduct(result: dict) -> dict:
    """What the finishes do not say: who picked up whose object and when, and how often a
    robot's move was blocked by its partner."""
    took = [[t for t, what, by, obj, _ in result["events"]
             if what == "picked up" and by == i and obj == 1 - i] for i in (0, 1)]
    return dict(took_partners_object_at=took,
                blocked_by_partner=[led["blocked_by_partner"] for led in result["ledger"]],
                moves=[led["moves"] for led in result["ledger"]],
                hand_actions=[led["hand_actions"] for led in result["ledger"]],
                handed_over=[led["handed_over"] for led in result["ledger"]],
                put_downs=[led.get("put_downs", 0) for led in result["ledger"]],
                set_down_for_partner=[led["set_down_for_partner"] for led in result["ledger"]],
                failed_set_downs=[led["failed_set_downs"] for led in result["ledger"]],
                events=len(result["events"]),
                residual_max=result["solver"]["residual_max"],
                residual_target=result["solver"]["residual_target"])


def cell(row: dict) -> str:
    return " / ".join("–" if t is None else str(t) for t in row["done_at"])


def matrix(rows: dict, baseline: dict, title: str) -> str:
    """The finishes as a Markdown table: agent_1's goal by row, agent_2's by column."""
    base = " / ".join("–" if t is None else str(t) for t in baseline["done_at"])
    lines = [f"# {title}", "",
             "Transition at which agent_1 / agent_2 finished its own goal; – = never (at most "
             f"{max(r['steps'] for r in rows.values())} transitions were rolled out). Two "
             f"level-0 robots without social goals: {base}. Rows: agent_1's social goal; "
             "columns: agent_2's.", "",
             "| agent_1 \\ agent_2 | " + " | ".join(GOALS) + " |",
             "|---|" + "---|" * len(GOALS)]
    for a in GOALS:
        lines.append(f"| **{a}** | " + " | ".join(
            cell(rows[a, b]) if (a, b) in rows else "" for b in GOALS) + " |")
    both = sum(all(t is not None for t in r["done_at"]) for r in rows.values())
    none = sum(all(t is None for t in r["done_at"]) for r in rows.values())
    lines += ["", f"{len(rows)} pairings: both goals done in {both}, exactly one in "
              f"{len(rows) - both - none}, neither in {none}. The solver's residual check "
              f"passed in {sum(r['numerically_accepted'] for r in rows.values())} of them.", ""]
    return "\n".join(lines)


# who finished: fill of a cell of the figure, and its name in the legend
OUTCOMES = (("both", "#cde2fb", "both goals done"),
            ("first", "#ffd8b0", "only agent_1's goal"),
            ("second", "#f3b7b7", "only agent_2's goal"),
            ("neither", "#d6d5d0", "neither"))


def outcome(row: dict) -> str:
    a, b = (t is not None for t in row["done_at"])
    return "both" if a and b else "first" if a else "second" if b else "neither"


def draw_matrix(rows: dict, baseline: dict, title: str, path: Path) -> None:
    """The finishes as a figure: agent_1's goal by row, agent_2's by column, the transition
    of each finish written in the cell and the cell filled by who finished."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fill = {key: colour for key, colour, _ in OUTCOMES}
    ink, quiet, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    n = len(GOALS)
    fig, ax = plt.subplots(figsize=(10.4, 7.6), dpi=150)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for i, a in enumerate(GOALS):
        for j, b in enumerate(GOALS):
            if (a, b) not in rows:
                continue
            row = rows[a, b]
            ax.add_patch(Rectangle((j + 0.03, i + 0.03), 0.94, 0.94, linewidth=0,
                                   facecolor=fill[outcome(row)]))
            ax.text(j + 0.5, i + 0.44, cell(row), ha="center", va="center", fontsize=13,
                    color=ink, fontweight="bold")
            ax.text(j + 0.5, i + 0.76, f"{row['steps']} transitions", ha="center",
                    va="center", fontsize=7.5, color=quiet)
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks([j + 0.5 for j in range(n)], GOALS, fontsize=10.5, color=ink)
    ax.set_yticks([i + 0.5 for i in range(n)], GOALS, fontsize=10.5, color=ink)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_xlabel("agent_2's social goal", fontsize=10.5, color=quiet, labelpad=8)
    ax.set_ylabel("agent_1's social goal", fontsize=10.5, color=quiet, labelpad=8)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_aspect("equal")
    base = " / ".join("–" if t is None else str(t) for t in baseline["done_at"])
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=13, color=ink, fontweight="bold")
    fig.text(0.02, 0.935, "Transition at which agent_1 / agent_2 finished its own goal "
             f"(– = never). Without social goals: {base}.", ha="left", fontsize=9.5,
             color=quiet)
    present = {outcome(r) for r in rows.values()}
    ax.legend(handles=[Patch(facecolor=colour, label=label) for key, colour, label in OUTCOMES
                       if key in present],
              loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=10,
              title="finished", title_fontsize=10, alignment="left", labelcolor=ink)
    fig.subplots_adjust(left=0.13, right=0.80, top=0.82, bottom=0.03)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--weights", type=float, nargs=2, default=(1.0, 1.0),
                        metavar=("AGENT_1", "AGENT_2"))
    parser.add_argument("--competition", choices=G.COMPETITION_FORMS,
                        default=Config().competition)
    parser.add_argument("--scene", choices=("six_objects", "exchange"), default="six_objects")
    parser.add_argument("--hand-over", action="store_true",
                        help="--scene exchange: hand-overs are allowed as well "
                             "(Config.hand_over)")
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
    parser.add_argument("--level-free-level", type=int, choices=(1, 2),
                        help="the level of a hinderer and of a competitor of a race form when "
                             "none is asked for (Config.level_free_level): 2 by default since "
                             "2026-10-11, like every robot with a social goal; 1 as before "
                             "(the records of before; also with --told)")
    parser.add_argument("--only-competition", action="store_true",
                        help="only the 13 pairings with a competitor in them")
    parser.add_argument("--steps", type=int, default=60, help="most transitions per rollout")
    parser.add_argument("--output", type=Path, help="output folder")
    parser.add_argument("--no-video", "--no-gif", dest="no_gif", action="store_true",
                        help="records and tables only")
    parser.add_argument("--resume", action="store_true",
                        help="keep the records (and videos) already in the output folder and roll "
                             "out only the pairings that have none")
    parser.add_argument("--fps", type=float, default=3.0)
    parser.add_argument("--threads", type=int, default=0, help="0 = every CPU available")
    args = parser.parse_args()
    if any(w < 0 for w in args.weights):
        parser.error("--weights must be >= 0")
    weights = tuple(args.weights)
    tag = ("w" + "_".join(f"{w:g}" for w in weights) + f"_{args.competition}"
           + ("_told" if args.told else "_inferred"))
    objects = args.objects or ("known" if args.told else "inferred")
    knows_objects = dict(partner_object=objects)
    if args.max_objects is not None:
        knows_objects["max_objects"] = args.max_objects
    if args.cooperate_with:
        knows_objects["cooperate_with"] = tuple(args.cooperate_with)
    if args.level_free_level is not None:
        knows_objects["level_free_level"] = args.level_free_level
    knows = {**(dict(KNOWN) if args.told else {}), **knows_objects}
    if objects == "inferred":                    # (the records of before: objects known)
        tag += "_objects" + ("" if args.max_objects is None else f"{args.max_objects}")
    tag += G.cooperation_tag(Config(**knows).cooperate_with)   # (of before: Table I's)
    tag += G.level_tag(Config(**knows).level_free_level)       # (of before: a hinderer at 1)
    mode = ""
    level1 = knows.get("exchange_level1", Config().exchange_level1)
    if args.exchange_level1 is not None and args.exchange_level1 != level1:
        knows["exchange_level1"] = args.exchange_level1
        mode += f"level1_{args.exchange_level1}_"
    if args.scene == "exchange":
        scene = exchange_scene()
        cfg = Config(num_threads=args.threads, competition=args.competition,
                     set_down_for_partner=False, hand_over=args.hand_over, **knows)
        tag = "exchange_" + "hand_over_" * args.hand_over + mode + tag
    else:
        if args.hand_over:
            parser.error("--hand-over goes with --scene exchange (on the six-object scene "
                         "hand-overs are allowed anyway)")
        scene = six_object_scene()
        cfg = Config(num_threads=args.threads, competition=args.competition, **knows)
        tag = mode + tag
    output = (args.output or ROOT / "output" / f"all_pairings_{tag}").resolve()
    output.mkdir(parents=True, exist_ok=True)
    pairs = [(a, b) for a in GOALS for b in GOALS
             if not args.only_competition or "competition" in (a, b)]
    print(f"scene: {scene.name} | weights {weights} | competition: {args.competition} | "
          f"partner goals {cfg.partner_social_goal} | exchange at level 1: "
          f"{cfg.exchange_level1} | {len(pairs)} pairings | at most {args.steps} transitions | "
          f"{output}", flush=True)

    def kept(name: str):
        """--resume: the record already in the folder, if it was written with this
        configuration."""
        path = output / f"{name}.json"
        if not (args.resume and path.exists()):
            return None
        result = load_result(path)
        mine = to_json(dataclasses.asdict(cfg))
        theirs = written_with(result.get("config") or {})
        if any(theirs.get(key) != value for key, value in mine.items() if key != "num_threads"):
            raise SystemExit(f"--resume: {path} was written with another configuration; "
                             "pass the flags it was written with, or another --output")
        if action_names(result) != ACTION_NAMES:
            raise SystemExit(f"--resume: {path} was rolled out with \"use\" (before "
                             "2026-10-16), not the five hand actions; pass another --output")
        return result

    baseline = kept("00_baseline")
    if baseline is None:
        baseline = run_baseline(scene, cfg=cfg, n_steps=args.steps)
        save_result(output / "00_baseline.json", baseline)
    rows = {}
    for a, b in pairs:
        name = f"{a}__{b}"
        result = kept(name)
        if result is None:
            result = run_pairing(scene, (a, b), weights=weights, cfg=cfg, n_steps=args.steps)
            save_result(output / f"{name}.json", result)
        row = summary_row(name, result, baseline)
        row.update(conduct(result))
        rows[a, b] = row
        print_row(row)
        if not args.no_gif and not (args.resume and (output / f"{name}.mp4").exists()):
            tic = time.perf_counter()
            used = result["levels_used"]
            label = [goal_label(g, competition_of(result), cooperation_of(result))
                     for g in (a, b)]
            knew = "told" if cfg.partner_social_goal == "known" else "inferred"
            render_videos(result, output / name, names=NAMES, fps=args.fps,
                          title=f"{NAMES[0]}: {label[0]}     {NAMES[1]}: {label[1]}     "
                                f"weights {weights[0]:g} / {weights[1]:g}     "
                                f"levels {used[0]} / {used[1]}     goals {knew}"
                                + ("     partner may be idle" if cfg.idle_hypothesis else ""))
            print(f"    {name}.mp4 ({time.perf_counter() - tic:.0f}s)", flush=True)
    goals_are = ("inferred" if cfg.partner_social_goal == "inferred" else "told")
    if cfg.idle_hypothesis:
        goals_are += ", partner may be idle"
    title = (f"Every pairing of social goals, weights {weights[0]:g} / {weights[1]:g}, "
             f"competition: {args.competition}, partner goals {goals_are}")
    if args.scene == "exchange":
        rule = "hand-overs allowed" if args.hand_over else "put down and pick up"
        title = (f"The exchange scene ({rule}; partner goals {goals_are}): every pairing, "
                 f"weights {weights[0]:g} / {weights[1]:g}, competition: {args.competition}")
    text = matrix(rows, baseline, title)
    (output / "matrix.md").write_text(text)
    draw_matrix(rows, baseline, title, output / "matrix.png")
    summary = dict(scene=scene.to_json(), config=dataclasses.asdict(cfg), weights=weights,
                   levels="default", goals=GOALS, baseline_done_at=baseline["done_at"],
                   pairings=list(rows.values()))
    (output / "summary.json").write_text(json.dumps(to_json(summary), indent=1) + "\n")
    print("\n" + text, flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
