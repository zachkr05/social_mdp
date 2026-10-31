"""How well the robots of a folder of rollouts inferred each other's goals.

Run from the project root, on a folder written by ``examples/run_all_pairings.py``:

    python examples/measure_inference.py output/all_pairings_w1_1_rival_inferred_idle
    python examples/measure_inference.py output/all_pairings_w30_30_rival_inferred --no-figure

For every robot that inferred its partner's social goal (level 2) it compares the belief it
ended with against its partner's true goal, and for every robot that inferred its partner's
destination (levels 1 and 2) the same for the destination.  Conflict and hinder are one model
at level 1 (Table I's ``-r(g_j)``), so no robot can tell them apart: they count as one class,
whose prior is 2/7 where every other goal's is 1/7.

It prints, by the partner's true goal: how often that goal was ranked first at the end, the
mean belief in it, and what was ranked first instead; and writes ``inference.json`` (one row
per robot) and ``inference.png`` (the confusion matrix) into the folder.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_pairings import goals as G  # noqa: E402

GOALS = tuple(G.SOCIAL_GOALS)                       # the order of every social belief
HOSTILE = "conflict / hinder"
CLASSES = ("none", "cooperation", "exchange", "coercion", "competition", HOSTILE)


def goal_class(goal: str) -> str:
    """The class a goal counts in: conflict and hinder are one."""
    return HOSTILE if goal in ("conflict", "hinder") else goal


def merged(belief) -> dict:
    """A belief over the seven goals -> over the six classes."""
    out = defaultdict(float)
    for k, goal in enumerate(GOALS):
        out[goal_class(goal)] += float(belief[k])
    return out


def ranked_first(m: dict) -> str:
    return max(CLASSES, key=lambda c: (m.get(c, 0.0), -CLASSES.index(c)))


def robot_rows(record: dict, name: str) -> list:
    """One row per robot of one rollout."""
    goals, levels = record["goals"], record["levels_used"]
    candidates = [tuple(c) for c in record["scene"]["candidates"]]
    destinations = [tuple(d) for d in record["scene"]["destinations"]]
    rows = []
    for i in (0, 1):
        j = 1 - i
        row = dict(pairing=name, seat=i + 1, goal=goals[i], level=levels[i],
                   partner=goals[j], partner_level=levels[j], truth=goal_class(goals[j]))
        social = record["final_social_belief"][i]
        if social is not None:
            m = merged(social)
            row.update(belief_in_truth=m[row["truth"]], ranked_first=ranked_first(m),
                       belief=dict(m))
            # the first transition from which the true class holds 0.5 or more to the end
            held = None
            for t, step in enumerate(record["trace"]):
                if merged(step["social_belief"][i])[row["truth"]] >= 0.5:
                    held = t if held is None else held
                else:
                    held = None
            row["holds_half_from"] = held if m[row["truth"]] >= 0.5 else None
        destination = record["final_belief"][i]
        if destination is not None:
            true_k = candidates.index(destinations[j])
            row.update(destination_belief=float(destination[true_k]),
                       destination_first=int(np.argmax(destination)) == true_k)
        rows.append(row)
    return rows


def summarise(rows: list) -> str:
    social = [r for r in rows if "belief_in_truth" in r]
    places = [r for r in rows if "destination_belief" in r]
    lines = [f"{len(social)} robots inferred a social goal (level 2), {len(places)} a destination "
             "(levels 1 and 2)",
             f"social goal ranked first at the end: "
             f"{sum(r['ranked_first'] == r['truth'] for r in social)} / {len(social)}; "
             f"held at 0.5 or more: {sum(r['belief_in_truth'] >= 0.5 for r in social)}; "
             f"mean belief in it {np.mean([r['belief_in_truth'] for r in social]):.2f}",
             "",
             f"{'partner':<20} {'first':>7} {'belief':>7}   ranked first instead"]
    for c in CLASSES:
        rs = [r for r in social if r["truth"] == c]
        if not rs:
            continue
        wrong = defaultdict(int)
        for r in rs:
            if r["ranked_first"] != c:
                wrong[r["ranked_first"]] += 1
        instead = ", ".join(f"{k} {v}" for k, v in sorted(wrong.items(), key=lambda kv: -kv[1]))
        lines.append(f"{c:<20} {sum(r['ranked_first'] == c for r in rs):>3} / {len(rs):<2} "
                     f"{np.mean([r['belief_in_truth'] for r in rs]):>6.2f}   {instead}")
    lines += ["", f"destination ranked first at the end: "
                  f"{sum(r['destination_first'] for r in places)} / {len(places)}; mean belief "
                  f"{np.mean([r['destination_belief'] for r in places]):.2f}"]
    return "\n".join(lines)


def draw(rows: list, title: str, path: Path) -> None:
    """The confusion matrix: the partner's true goal by row, the goal ranked first by column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.colors import LinearSegmentedColormap

    surface, ink, quiet, empty = "#fcfcfb", "#0b0b0b", "#52514e", "#f3f2ee"
    ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
            "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("blue", ramp)
    names = {f.name for f in font_manager.fontManager.ttflist}
    family = "Inter" if "Inter" in names else "DejaVu Sans"
    social = [r for r in rows if "belief_in_truth" in r]
    present = [c for c in CLASSES if any(r["truth"] == c for r in social)]
    n, m = len(present), len(CLASSES)
    count = np.zeros((n, m), int)
    for r in social:
        count[present.index(r["truth"]), CLASSES.index(r["ranked_first"])] += 1
    share = count / np.maximum(count.sum(axis=1, keepdims=True), 1)
    mean = [np.mean([r["belief_in_truth"] for r in social if r["truth"] == c]) for c in present]
    label = {c: c for c in CLASSES}
    label["competition"] = "competition (rival)"

    with plt.rc_context({"font.family": family, "font.size": 11}):
        fig = plt.figure(figsize=(11.2, 8.0), dpi=150, facecolor=surface)
        ax = fig.add_axes([0.25, 0.19, 0.52, 0.61], facecolor=surface)
        for i, c in enumerate(present):
            for j, d in enumerate(CLASSES):
                diagonal = c == d
                ax.add_patch(plt.Rectangle(
                    (j + 0.04, i + 0.04), 0.92, 0.92,
                    facecolor=cmap(share[i, j]) if count[i, j] else empty,
                    edgecolor=ink if diagonal else "none", linewidth=2.0 if diagonal else 0))
                if count[i, j]:
                    dark = share[i, j] > 0.55
                    ax.text(j + 0.5, i + 0.47, str(count[i, j]), ha="center", va="center",
                            fontsize=15, fontweight="bold", color="white" if dark else ink)
                    ax.text(j + 0.5, i + 0.75, f"{100 * share[i, j]:.0f}%", ha="center",
                            va="center", fontsize=9, color="white" if dark else quiet)
            ax.text(m + 0.25, i + 0.5, f"{mean[i]:.2f}", ha="left", va="center", color=ink)
        ax.text(m + 0.25, -0.08, "mean belief\nin the true goal", ha="left", va="bottom",
                fontsize=9, color=quiet)
        ax.set_xlim(0, m)
        ax.set_ylim(n, 0)
        ax.set_xticks(np.arange(m) + 0.5)
        ax.set_xticklabels([label[c] for c in CLASSES], rotation=30, ha="right", color=ink)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_yticklabels([f"{label[c]}  (n = {count[k].sum()})" for k, c in enumerate(present)],
                           color=ink)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("goal the robot ranks first for its partner at the end", color=quiet,
                      labelpad=8)
        ax.set_ylabel("partner's true social goal", color=quiet, labelpad=8)
        right = sum(r["ranked_first"] == r["truth"] for r in social)
        fig.text(0.03, 0.95, "Which social goal each robot ends up believing its partner has",
                 fontsize=16, fontweight="bold", color=ink)
        fig.text(0.03, 0.91, title, fontsize=10.5, color=quiet)
        fig.text(0.03, 0.882,
                 f"Outlined: recognised ({right} of the {len(social)} robots at level 2, the ones "
                 "that infer a social goal).", fontsize=10.5, color=quiet)
        fig.text(0.03, 0.854,
                 "Conflict and hinder are one model at level 1, so they are merged; the prior is "
                 "1/7 per goal, 2/7 for the merged pair.", fontsize=10.5, color=quiet)
        cax = fig.add_axes([0.86, 0.19, 0.018, 0.61])
        bar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
        bar.outline.set_visible(False)
        bar.set_ticks([0, 0.25, 0.5, 0.75, 1])
        bar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
        bar.ax.tick_params(length=0, colors=quiet, labelsize=9)
        bar.set_label("share of the row", color=quiet, fontsize=9)
        fig.savefig(path, facecolor=surface)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folder", type=Path, help="a folder written by run_all_pairings.py")
    parser.add_argument("--no-figure", action="store_true", help="inference.json only")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.folder.glob("*__*.json")):
        rows += robot_rows(json.loads(path.read_text()), path.stem)
    if not rows:
        parser.error(f"no rollouts (<goal>__<goal>.json) in {args.folder}")
    print(summarise(rows))
    (args.folder / "inference.json").write_text(json.dumps(rows, indent=1) + "\n")
    if not args.no_figure:
        summary = args.folder / "summary.json"
        pairings = len({r["pairing"] for r in rows})
        title = f"{pairings} pairings in {args.folder.name}"
        if summary.exists():
            s = json.loads(summary.read_text())
            weights = " / ".join(f"{w:g}" for w in s.get("weights", ()))
            scene = s.get("scene", {}).get("name", "?")
            scene = {"six_objects": "six-object"}.get(scene, scene)
            idle = ", idle hypothesis on" if s.get("config", {}).get("idle_hypothesis") else ""
            title = f"{pairings} pairings on the {scene} scene, w = {weights}{idle}"
        draw(rows, title, args.folder / "inference.png")
        print(f"\n{args.folder / 'inference.png'}")


if __name__ == "__main__":
    main()
