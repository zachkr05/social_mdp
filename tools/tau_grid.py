"""The pairings at every pair of taus, side by side (run_tau_grid.sh writes the folders).

    python tools/tau_grid.py --out output/tau_grid FOLDER [FOLDER ...]

Each FOLDER is one output folder of ``examples/run_object_inference.py`` (one pair of taus,
its ``summary.json`` written by a pass over its pairings).  Writes to --out:

- ``tau_grid.csv``: one row per (pairing, agent_1's tau, agent_2's tau): when each robot
  finished its own goal (empty: never), whether it has one (a hinderer has none), the
  transitions rolled out, what stopped the rollout, the folder;
- ``tau_grid.png``: every pairing (agent_1's goal by row, agent_2's by column), each a small
  grid over the two taus (agent_1's up, agent_2's across); a cell's upper-left half is
  agent_1, its lower-right half agent_2, coloured where that robot finished its own goal;
- ``tau_grid_counts.png``: per pair of taus, in how many pairings each robot finished;
- ``pairings/<agent_1>__<agent_2>.png``: one pairing, with the transition each finished;
- ``tau_grid.md``: the counts and every pairing's finishes as tables.

Where every folder has one tau for both robots (``run_tau_grid.sh`` with SAME=1), the taus
are one axis: each pairing is a row of cells over the tau, and the counts a line per robot.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

GOAL_ORDER = ("cooperation", "conflict", "competition", "coercion", "exchange", "none",
              "hinder")
DEFAULT_TAU = 0.2                     # Config.temperature

# the chart's colours (the validated categorical slots 1 and 2; neutral greys for chrome)
SURFACE, INK, QUIET, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
HAIRLINE, NOT_DONE = "#e1e0d9", "#c3c2b7"
ROBOT = ("#2a78d6", "#eb6834")        # agent_1 blue, agent_2 orange
RAMP_ENDS = (("#e8f1fc", "#2a78d6", "#0d366b"), ("#fdeae2", "#eb6834", "#7a2e0e"))


def taus_of(options: dict) -> tuple:
    """(agent_1's tau, agent_2's) of a folder's options."""
    pair = options.get("robot_temperatures")
    if pair is not None:
        return float(pair[0]), float(pair[1])
    t = float(options.get("temperature", DEFAULT_TAU))
    return t, t


def read(folders) -> list:
    rows = []
    for folder in folders:
        path = Path(folder) / "summary.json"
        if not path.exists():
            print(f"tau_grid: no summary.json in {folder}, left out")
            continue
        summary = json.loads(path.read_text())
        t1, t2 = taus_of(summary.get("options") or {})
        scene = (summary.get("scene") or {}).get("name", "")
        # Config.conflict (since 2026-11-01; a folder of before: Table I's)
        reading = (summary.get("options") or {}).get("conflict", "table")
        # the hazards of the partner's goal (since 2026-11-02; a folder of before: none)
        hazards = tuple(float((summary.get("options") or {}).get(k, 0.0))
                        for k in ("goal_hazard", "social_hazard", "hand_hazard"))
        # a lapse in the likelihood and its own temperature (since 2026-11-04; a folder of
        # before: neither)
        lapse = float((summary.get("options") or {}).get("belief_lapse", 0.0))
        inftau = (summary.get("options") or {}).get("inference_temperature")
        lpow = float((summary.get("options") or {}).get("likelihood_power", 1.0))
        for r in summary["rows"]:
            if r.get("objects", "inferred") != "inferred":
                continue
            own = r.get("own_goal") or [True, True]
            rows.append(dict(agent_1=r["pairing"][0], agent_2=r["pairing"][1], tau_1=t1,
                             tau_2=t2, done_1=r["done_at"][0], done_2=r["done_at"][1],
                             goal_1=bool(own[0]), goal_2=bool(own[1]), steps=r["steps"],
                             stopped_by=r["stopped_by"], scene=scene, conflict=reading,
                             hazards=hazards, lapse=lapse, inftau=inftau, lpow=lpow,
                             folder=str(folder)))
    return rows


def scene_of(rows) -> str:
    """The scene, and the reading of conflict where a folder was made under the task
    conflict of 2026-11-01 (Config.conflict; Table I's, the reading of before, is not
    named), the lapse in the likelihood (2026-11-04) and the hazards of the partner's goal
    where there are any (2026-11-02)."""
    readings = {r.get("conflict", "table") for r in rows}
    name = rows[0]["scene"]
    if readings != {"table"}:
        name += (", task conflict" if readings == {"task"}
                 else ", task and Table I conflict mixed")
    inftaus = {r.get("inftau") for r in rows}
    if len(inftaus) > 1:
        name += ", inference taus mixed"
    elif inftaus != {None}:
        name += f", inference tau {float(inftaus.pop()):g}"
    lapses = {float(r.get("lapse") or 0.0) for r in rows}
    if len(lapses) > 1:
        name += ", lapses mixed"
    elif lapses != {0.0}:
        name += f", lapse {lapses.pop():g}"
    powers = {float(r.get("lpow") or 1.0) for r in rows}
    if len(powers) > 1:
        name += ", likelihood powers mixed"
    elif powers != {1.0}:
        name += f", likelihood ^{powers.pop():g}"
    hazards = {tuple(r.get("hazards") or (0.0, 0.0, 0.0)) for r in rows}
    if len(hazards) > 1:
        name += ", hazards mixed"
    else:
        name += "".join(f", {what} {h:g}" for what, h in
                        zip(("hazard", "social hazard", "hand hazard"), hazards.pop()) if h)
    return name


def axes_of(rows) -> tuple:
    goals_1 = [g for g in GOAL_ORDER if any(r["agent_1"] == g for r in rows)]
    goals_2 = [g for g in GOAL_ORDER if any(r["agent_2"] == g for r in rows)]
    goals_1 += sorted({r["agent_1"] for r in rows} - set(goals_1))
    goals_2 += sorted({r["agent_2"] for r in rows} - set(goals_2))
    taus_1 = sorted({r["tau_1"] for r in rows})
    taus_2 = sorted({r["tau_2"] for r in rows})
    return goals_1, goals_2, taus_1, taus_2


class Layout:
    """What the figures put on their axes: the pairings, and the taus -- a grid of agent_1's
    (rows of a panel) by agent_2's (columns), or, where every rollout had one tau for both
    robots, one row over that tau."""

    def __init__(self, rows) -> None:
        self.goals_1, self.goals_2, taus_1, taus_2 = axes_of(rows)
        self.same = all(r["tau_1"] == r["tau_2"] for r in rows)
        self.ys = [None] if self.same else taus_1
        self.xs = sorted(set(taus_1) | set(taus_2)) if self.same else taus_2
        self.by = {(r["agent_1"], r["agent_2"], r["tau_1"], r["tau_2"]): r for r in rows}

    def get(self, a: str, b: str, y, x):
        return self.by.get((a, b, x, x) if self.same else (a, b, y, x))

    def has(self, a: str, b: str) -> bool:
        return any(k[:2] == (a, b) for k in self.by)


def when(done, has_goal: bool) -> str:
    return "no goal" if not has_goal else ("–" if done is None else str(done))


def write_tables(rows, out: Path) -> None:
    goals_1, goals_2, taus_1, taus_2 = axes_of(rows)
    with (out / "tau_grid.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_1_goal", "agent_2_goal", "agent_1_tau", "agent_2_tau",
                         "agent_1_done_at", "agent_2_done_at", "agent_1_has_goal",
                         "agent_2_has_goal", "steps", "stopped_by", "folder"])
        for r in sorted(rows, key=lambda r: (goals_1.index(r["agent_1"]),
                                             goals_2.index(r["agent_2"]), r["tau_1"],
                                             r["tau_2"])):
            writer.writerow([r["agent_1"], r["agent_2"], f"{r['tau_1']:g}", f"{r['tau_2']:g}",
                             "" if r["done_1"] is None else r["done_1"],
                             "" if r["done_2"] is None else r["done_2"],
                             int(r["goal_1"]), int(r["goal_2"]), r["steps"], r["stopped_by"],
                             r["folder"]])
    lay = Layout(rows)

    def counts(here) -> tuple:
        a = sum(r["done_1"] is not None for r in here)
        b = sum(r["done_2"] is not None for r in here)
        both = sum(r["done_1"] is not None and r["done_2"] is not None for r in here)
        return a, b, both

    lines = ["# The pairings at every pair of taus" if not lay.same else
             "# The pairings at every tau, both robots at it", "",
             f"Scene {scene_of(rows)}.  Each robot reads its partner at its own tau "
             "(Config.robot_temperatures)"
             + ("; here both robots have the same." if lay.same else
                "; agent_1's tau down, agent_2's across."), "",
             "## In how many pairings each robot finished its own goal", "",
             "agent_1 / agent_2 / both, of the pairings rolled out "
             "(a hinderer has no goal of its own and never finishes).", ""]
    if lay.same:
        lines += ["| tau | agent_1 | agent_2 | both | rolled out |", "|---|---|---|---|---|"]
        for t in lay.xs:
            here = [r for r in rows if r["tau_1"] == t]
            a, b, both = counts(here)
            lines.append(f"| {t:g} | {a} | {b} | {both} | {len(here)} |")
    else:
        lines += ["| agent_1's tau \\ agent_2's | " + " | ".join(f"{t:g}" for t in lay.xs)
                  + " |", "|---|" + "---|" * len(lay.xs)]
        for t1 in lay.ys:
            cells = []
            for t2 in lay.xs:
                here = [r for r in rows if r["tau_1"] == t1 and r["tau_2"] == t2]
                a, b, both = counts(here)
                cells.append(f"{a} / {b} / {both} of {len(here)}" if here else "")
            lines.append(f"| {t1:g} | " + " | ".join(cells) + " |")
    pairs = sorted({(r["tau_1"], r["tau_2"]) for r in rows})
    lines += ["", "## Every pairing", "",
              "The transition at which agent_1 / agent_2 finished its own goal (– never), by "
              + ("tau." if lay.same else "agent_1's tau / agent_2's tau."), "",
              "| pairing | " + " | ".join(f"{t1:g}" if lay.same else f"{t1:g} / {t2:g}"
                                         for t1, t2 in pairs)
              + " |", "|---|" + "---|" * len(pairs)]
    for a in lay.goals_1:
        for b in lay.goals_2:
            cells = []
            for t1, t2 in pairs:
                r = lay.by.get((a, b, t1, t2))
                cells.append("" if r is None else
                             f"{when(r['done_1'], r['goal_1'])} / "
                             f"{when(r['done_2'], r['goal_2'])}")
            if any(cells):
                lines.append(f"| {a} / {b} | " + " | ".join(cells) + " |")
    lines += ["", "Folders read:", ""] + [f"- `{f}`" for f in dict.fromkeys(r["folder"]
                                                                          for r in rows)]
    (out / "tau_grid.md").write_text("\n".join(lines) + "\n")


def _cell(ax, x: float, y: float, r, gap: float, labels: bool, fontsize: float) -> None:
    """One pair of taus: the upper-left half agent_1, the lower-right half agent_2."""
    from matplotlib.patches import Polygon, Rectangle
    x0, y0, x1, y1 = x + gap, y + gap, x + 1 - gap, y + 1 - gap
    if r is None:                     # not rolled out (yet)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor=HAIRLINE, linewidth=0.8))
        return
    halves = (((x0, y0), (x0, y1), (x1, y1)), ((x0, y0), (x1, y0), (x1, y1)))
    for i, corners in enumerate(halves):
        has_goal, done = r[f"goal_{i + 1}"], r[f"done_{i + 1}"]
        if not has_goal:              # a hinderer without a goal of its own: hatched
            ax.add_patch(Polygon(corners, closed=True, facecolor=SURFACE, edgecolor=MUTED,
                                 hatch="////", linewidth=0.0))
        else:
            ax.add_patch(Polygon(corners, closed=True, linewidth=0.0,
                                 facecolor=ROBOT[i] if done is not None else NOT_DONE))
    ax.plot([x0, x1], [y0, y1], color=SURFACE, linewidth=1.2, solid_capstyle="butt")
    if labels:
        for i, (tx, ty) in enumerate(((x0 + 0.3 * (x1 - x0), y0 + 0.7 * (y1 - y0)),
                                      (x0 + 0.7 * (x1 - x0), y0 + 0.3 * (y1 - y0)))):
            has_goal, done = r[f"goal_{i + 1}"], r[f"done_{i + 1}"]
            text = "" if not has_goal else ("–" if done is None else str(done))
            ink = _label_ink(ROBOT[i]) if (has_goal and done is not None) else INK
            ax.text(tx, ty, text, ha="center", va="center", fontsize=fontsize, color=ink)


def _label_ink(fill) -> str:
    """White or ink, whichever contrasts more with the fill (WCAG relative luminance)."""
    from matplotlib.colors import to_rgb
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in to_rgb(fill)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if 1.05 / (lum + 0.05) > (lum + 0.05) / 0.05 else INK


def _legend(fig, y: float) -> None:
    """A sample cell per item: the half it is about coloured, the other half grey.  Laid
    out in inches (the axes span the figure's width, 1 data unit = 1 inch)."""
    from matplotlib.patches import Polygon
    width_in, height_in = fig.get_size_inches()
    band = 0.32                          # the legend's height, inches
    ax = fig.add_axes([0.0, y, 1.0, band / height_in])
    ax.set_xlim(0, width_in)
    ax.set_ylim(0, band)
    ax.axis("off")
    upper, lower = 0, 1                  # which half of the sample cell an item colours
    items = (("agent_1 finished its own goal (upper-left half)", ROBOT[0], None, upper),
             ("agent_2 finished (lower-right half)", ROBOT[1], None, lower),
             ("did not finish", NOT_DONE, None, None),
             ("a hinderer: no goal of its own", SURFACE, "////", None))
    size, char = 0.2, 0.068              # the sample's side; a character of 9 pt, inches
    total = sum(size + 0.08 + char * len(t) for t, *_ in items) + 0.35 * (len(items) - 1)
    x = max(0.2, (width_in - total) / 2)
    for text, colour, hatch, half in items:
        x0, y0, x1, y1 = x, (band - size) / 2, x + size, (band + size) / 2
        halves = (((x0, y0), (x0, y1), (x1, y1)), ((x0, y0), (x1, y0), (x1, y1)))
        for k, corners in enumerate(halves):
            mine = half is None or half == k
            ax.add_patch(Polygon(corners, closed=True, linewidth=0.0,
                                 facecolor=colour if mine else NOT_DONE,
                                 hatch=hatch if mine else None,
                                 edgecolor=MUTED if (hatch and mine) else "none"))
        ax.plot([x0, x1], [y0, y1], color=SURFACE, linewidth=1.0)
        ax.text(x1 + 0.08, band / 2, text, va="center", fontsize=9, color=QUIET)
        x = x1 + 0.08 + char * len(text) + 0.35


def _tau_axes(ax, lay: Layout, fontsize: float, names: bool, thin: bool = False) -> None:
    """Tick labels (thin: every other one where there are more than six) and axis names."""
    def labels(taus):
        step = 2 if (thin and len(taus) > 6) else 1
        return [f"{t:g}" if k % step == 0 else "" for k, t in enumerate(taus)]
    ax.set_xticks(np.arange(len(lay.xs)) + 0.5)
    ax.set_xticklabels(labels(lay.xs), fontsize=fontsize, color=MUTED)
    if lay.same:
        ax.set_yticks([])
    else:
        ax.set_yticks(np.arange(len(lay.ys)) + 0.5)
        ax.set_yticklabels(labels(lay.ys), fontsize=fontsize, color=MUTED)
    ax.tick_params(length=0, pad=2)
    if names:
        ax.set_xlabel("tau (both robots)" if lay.same else "agent_2's tau",
                      fontsize=fontsize + 1, color=QUIET)
        if not lay.same:
            ax.set_ylabel("agent_1's tau", fontsize=fontsize + 1, color=QUIET)


def draw_grid(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lay = Layout(rows)
    n1, n2 = len(lay.goals_1), len(lay.goals_2)
    # in inches: a panel per pairing, its cells square
    panel_w = 2.4 if lay.same else 1.55
    cell = panel_w / len(lay.xs)
    panel_h = cell * len(lay.ys)
    gap_x, gap_y = 0.22, (0.3 if lay.same else 0.25)
    left = 1.25 if lay.same else 1.65
    right, top, bottom = 0.3, 1.25, 1.1
    W = left + n2 * panel_w + (n2 - 1) * gap_x + right
    H = top + n1 * panel_h + (n1 - 1) * gap_y + bottom
    fig = plt.figure(figsize=(W, H), dpi=150, facecolor=SURFACE)
    label_x = left - (0.15 if lay.same else 0.6)
    for a_i, a in enumerate(lay.goals_1):
        y0 = bottom + (n1 - 1 - a_i) * (panel_h + gap_y)
        fig.text(label_x / W, (y0 + panel_h / 2) / H, a, ha="right", va="center",
                 fontsize=10, color=INK)
        for b_i, b in enumerate(lay.goals_2):
            x0 = left + b_i * (panel_w + gap_x)
            ax = fig.add_axes([x0 / W, y0 / H, panel_w / W, panel_h / H])
            ax.set_facecolor(SURFACE)
            ax.set_xlim(0, len(lay.xs))
            ax.set_ylim(0, len(lay.ys))
            for side_ in ax.spines.values():
                side_.set_visible(False)
            for y, t1 in enumerate(lay.ys):
                for x, t2 in enumerate(lay.xs):
                    _cell(ax, x, y, lay.get(a, b, t1, t2), 0.06, False, 6)
            if a_i == n1 - 1 and b_i == 0:          # the taus on the corner panel only
                _tau_axes(ax, lay, 6, True, thin=True)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
            if a_i == 0:
                fig.text((x0 + panel_w / 2) / W, (H - top + 0.1) / H, b, ha="center",
                         va="bottom", fontsize=10, color=INK)
    fig.text(label_x / W, (H - top + 0.1) / H, "agent_1 \\ agent_2", ha="right",
             va="bottom", fontsize=9, color=MUTED)
    fig.text(0.5, (H - 0.25) / H,
             f"{scene_of(rows)}: who finished its own goal, "
             + ("by the tau of both robots" if lay.same else "by each robot's tau"),
             ha="center", va="top", fontsize=13, color=INK)
    fig.text(0.5, (H - 0.55) / H,
             "rows: agent_1's social goal; columns: agent_2's.  Inside each pairing, "
             + ("the tau (both robots the same) goes across." if lay.same else
                "agent_1's tau goes up and agent_2's across."),
             ha="center", va="top", fontsize=9.5, color=QUIET)
    _legend(fig, 0.12 / H)
    fig.savefig(out / "tau_grid.png", facecolor=SURFACE)
    plt.close(fig)


def draw_lines(rows, out: Path) -> None:
    """One tau for both robots: in how many pairings each robot finished, a line per robot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lay = Layout(rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    top, ends = 0, []
    for i in (0, 1):
        done, total = [], []
        for t in lay.xs:
            here = [r for r in rows if r["tau_1"] == t and r[f"goal_{i + 1}"]]
            done.append(sum(r[f"done_{i + 1}"] is not None for r in here))
            total.append(len(here))
        top = max(top, max(total))
        name = ("agent_1", "agent_2")[i]
        ax.plot(lay.xs, done, color=ROBOT[i], linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name, zorder=3)
        ends.append(done[-1])
    # the line ends named, pushed apart where they meet
    apart = max(1.0, 0.06 * max(1, top))
    shift = (0.0, 0.0) if abs(ends[0] - ends[1]) >= apart else (
        (apart / 2, -apart / 2) if ends[0] >= ends[1] else (-apart / 2, apart / 2))
    for i in (0, 1):
        ax.text(lay.xs[-1], ends[i] + shift[i], f"  {('agent_1', 'agent_2')[i]}",
                va="center", fontsize=9, color=QUIET)
    ax.set_ylim(0, max(1, top) * 1.05)
    ax.set_xlim(lay.xs[0] - 0.02, lay.xs[-1] + 0.06)
    ax.set_xticks(lay.xs)
    ax.set_xticklabels([f"{t:g}" for t in lay.xs], fontsize=8, color=MUTED)
    ax.tick_params(axis="y", labelsize=8, labelcolor=MUTED, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=HAIRLINE, linewidth=0.8)
    ax.set_axisbelow(True)
    for name_, side_ in ax.spines.items():
        side_.set_visible(name_ == "bottom")
        side_.set_color(NOT_DONE)
    ax.set_xlabel("tau (both robots)", fontsize=9, color=QUIET)
    ax.set_ylabel(f"pairings in which it finished\n(of {top} where it has a goal)",
                  fontsize=9, color=QUIET)
    ax.legend(frameon=False, fontsize=9, labelcolor=QUIET, loc="lower left")
    ax.set_title(f"{scene_of(rows)}: each robot's finishes by the tau of both",
                 fontsize=11, color=INK, pad=10)
    fig.tight_layout()
    fig.savefig(out / "tau_grid_counts.png", facecolor=SURFACE)
    plt.close(fig)


def draw_counts(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    lay = Layout(rows)
    if lay.same:
        draw_lines(rows, out)
        return
    taus_1, taus_2 = lay.ys, lay.xs
    fig, axes = plt.subplots(1, 2, figsize=(1.6 + 2 * (1.0 + 0.8 * len(taus_2)),
                                            1.9 + 0.8 * len(taus_1)), dpi=150,
                             facecolor=SURFACE)
    for i, ax in enumerate(axes):
        counts = np.full((len(taus_1), len(taus_2)), np.nan)
        totals = np.zeros((len(taus_1), len(taus_2)), dtype=int)
        for y, t1 in enumerate(taus_1):
            for x, t2 in enumerate(taus_2):
                here = [r for r in rows if r["tau_1"] == t1 and r["tau_2"] == t2
                        and r[f"goal_{i + 1}"]]
                if here:
                    counts[y, x] = sum(r[f"done_{i + 1}"] is not None for r in here)
                    totals[y, x] = len(here)
        top = max(1, int(np.nanmax(totals)) if totals.size else 1)
        ramp = LinearSegmentedColormap.from_list(f"robot{i}", RAMP_ENDS[i])
        ax.set_facecolor(SURFACE)
        ax.imshow(counts, origin="lower", cmap=ramp, vmin=0, vmax=top, aspect="equal",
                  extent=(0, len(taus_2), 0, len(taus_1)))
        for y in range(len(taus_1)):
            for x in range(len(taus_2)):
                if np.isnan(counts[y, x]):
                    continue
                ax.text(x + 0.5, y + 0.5, f"{int(counts[y, x])}", ha="center", va="center",
                        fontsize=9, color=_label_ink(ramp(counts[y, x] / top)))
        for k in range(len(taus_2) + 1):    # a surface gap between the cells
            ax.axvline(k, color=SURFACE, linewidth=2)
        for k in range(len(taus_1) + 1):
            ax.axhline(k, color=SURFACE, linewidth=2)
        for side_ in ax.spines.values():
            side_.set_visible(False)
        _tau_axes(ax, lay, 8, True)
        name = ("agent_1", "agent_2")[i]
        ax.set_title(f"pairings in which {name} finished its own goal\n(of {top} where it has "
                     "one)", fontsize=10, color=INK, pad=8)
    fig.suptitle(f"{scene_of(rows)}: each robot's finishes over both taus", fontsize=12,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out / "tau_grid_counts.png", facecolor=SURFACE, bbox_inches="tight",
                pad_inches=0.25)
    plt.close(fig)


def draw_pairings(rows, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lay = Layout(rows)
    (out / "pairings").mkdir(exist_ok=True)
    side = 0.75 if lay.same else 0.9                 # a cell, inches
    for a in lay.goals_1:
        for b in lay.goals_2:
            if not lay.has(a, b):
                continue
            fig, ax = plt.subplots(figsize=(1.6 + side * len(lay.xs),
                                            1.9 + side * len(lay.ys)), dpi=150,
                                   facecolor=SURFACE)
            ax.set_facecolor(SURFACE)
            ax.set_xlim(0, len(lay.xs))
            ax.set_ylim(0, len(lay.ys))
            ax.set_aspect("equal")
            for side_ in ax.spines.values():
                side_.set_visible(False)
            for y, t1 in enumerate(lay.ys):
                for x, t2 in enumerate(lay.xs):
                    _cell(ax, x, y, lay.get(a, b, t1, t2), 0.04, True, 7.5)
            _tau_axes(ax, lay, 8, True)
            ax.set_title(f"{a} (agent_1, blue) vs {b} (agent_2, orange)\ntransition each "
                         "finished its own goal (– never)", fontsize=9.5, color=INK, pad=8)
            fig.tight_layout()
            fig.savefig(out / "pairings" / f"{a}__{b}.png", facecolor=SURFACE)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="output folders, one per pair of taus")
    parser.add_argument("--out", default="output/tau_grid", help="where to write")
    args = parser.parse_args()
    rows = read(args.folders)
    if not rows:
        raise SystemExit("tau_grid: no rollouts found")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_tables(rows, out)
    draw_grid(rows, out)
    draw_counts(rows, out)
    draw_pairings(rows, out)
    print(f"tau_grid: {len(rows)} rollouts from {len(set(r['folder'] for r in rows))} "
          f"folders -> {out}/tau_grid.png, tau_grid_counts.png, tau_grid.md, tau_grid.csv, "
          "pairings/")


if __name__ == "__main__":
    main()
