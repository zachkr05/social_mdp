"""The congruence tier scenes of test_tiers.py on the house map, made fair, for the Social MDP.

Run from the project root:

    python examples/tier_scenes.py               # each scene's tier and A* task lengths, as
                                                 # drawn and fair; makes them fair again from
                                                 # test_tiers' placements and checks that this
                                                 # gives SCENES (~20 s); the two figures
    python examples/tier_scenes.py --no-search   # without the search

``examples/run_tier_scenes.py`` rolls the Social MDP out on them.

The placements
--------------
``test_tiers.py`` (2026-10-03) places two robots on the scene_048 house map at 0.25 m: two
placements per congruence tier for the four goals that act on the partner (its MEETING
suite: cooperation, conflict, competition, coercion) and for exchange (its EXCHANGE suite).
``PLACEMENTS`` holds the twelve of the low, medium and high tiers as it defines them.  Its
tiers are distances on the robots' fastest routes (start -> own object -> own destination):

  meeting   high: the two fastest routes cannot both be driven (someone has to wait);
            medium: they meet, with room to pass; low: they do not meet, and a detour of
            less than 1.5 m brings one robot next to the other while that one still works
  exchange  high: swapping tasks is shorter for the two together; medium: it makes no
            difference; low: it costs them less than 1.5 m together

The functions that measure them (``Graph``, ``scene_geometry``, ``own_detour``,
``tier_in_words``) are test_tiers' own, copied unchanged but for the resolution, which is a
parameter here; on the 0.25 m map they give the tier test_tiers gives every placement.

The house at 0.5 m
------------------
The Social MDP cannot be rolled out on the 0.25 m map in useful time: 276 free cells are
76,176 pairs of robot cells per configuration of the objects, some 12 times the 0.5 m house,
and a level-2 transition already takes about 12 s there.  ``house_free()`` is the house at
0.5 m (the resolution of ``Config``): each cell joins 2 x 2 cells of the 0.25 m map and is
free where at least two of them are, and cell (7, 9) is blocked again to keep the wall
between the middle room and the right room, which that rule opens.  79 free cells.

Fair
----
A scene is fair when A* gives the two robots' tasks -- start -> own object -> own
destination, alone on the map -- lengths within 10 % of each other:
``max(L1, L2) <= 1.1 min(L1, L2)``.  As drawn, none of the twelve is (ratios 1.25 to 2.73).
And neither robot, done and parked on its destination, may make the other's task impossible
(``parking_ok``): on the 0.25 m map none does, but at 0.5 m the bottom-left alcove is one
cell, reached only from the cell above it, and a destination there can wall the other robot
out.  ``SCENES`` holds each placement ported to the 0.5 m house and made fair by the
smallest moves of its task cells (``search``):

  - a cell is ported to the 0.5 m cell that holds it, or to the free one whose centre is
    nearest where that one is blocked; a coincidence the coarser map makes (two starts on
    one cell) moves the later cell to the nearest free cell not taken;
  - placements that move up to three of the six task cells, each within the room its ported
    cell is in (``room``), keeping the starts, the objects and the destinations each apart,
    every coincidence of the original (R2's object on R1's start, ...) and no new one, the
    scene's tier at 0.5 m by test_tiers' rules, fairness, and ``parking_ok``;
  - the least total move wins (steps on the map, summed over the moved cells); ties: the
    ratio nearest 1, then the fewest moved cells;
  - where no such placement exists (same_room_later, doorway_passing) the objects stay in
    their rooms -- they say what each robot goes to fetch -- and the starts and destinations
    may move anywhere.

Each scene has six candidate destinations, as the six-object scene has: the two real ones
and the first four of ``DISTRACTORS`` (one per room) that are no task cell and not next to a
destination.
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from social_pairings import Scene  # noqa: E402

Cell = Tuple[int, int]

# ---------------------------------------------------------------------------------------------
# The house map (test_tiers.MAP): top row of the text = top of the map, column = ix
# ---------------------------------------------------------------------------------------------

MAP = (
    "##########################",
    "#########..###############",
    "#########..###############",
    "######......##############",
    "######......##############",
    "######......##############",
    "#########...#######...####",
    "#########...#######.....##",
    "#########.....#.......####",
    "#########.....#.......####",
    "#######.......#.......####",
    "#########....###....######",
    "#######..............#####",
    "###..##..............#####",
    "###..##..............#####",
    "###..................#####",
    "......................####",
    ".........................#",
    ".........................#",
    "......########...........#",
    "#....#########...........#",
    "##############...........#",
    "##############...........#",
    "##############.###########",
    "##############.###########",
    "##############.###########",
    "##############.###########",
    "##########################",
)
FINE = 0.25                       # metres per cell of MAP
RESOLUTION = 0.5                  # metres per cell of the house the Social MDP runs on
REOPENED_WALL = (7, 9)            # blocked again at 0.5 m: the middle room / right room wall


def fine_free() -> np.ndarray:
    """(26, 28) bool, the 0.25 m map; axis 0 is world x."""
    return np.array([[ch == "." for ch in row] for row in MAP[::-1]]).T


def house_free() -> np.ndarray:
    """(13, 14) bool, the house at 0.5 m: a cell is free where at least two of its 2 x 2
    cells of the 0.25 m map are, but for REOPENED_WALL."""
    f = fine_free()
    X, Y = f.shape
    free = f.reshape(X // 2, 2, Y // 2, 2).sum(axis=(1, 3)) >= 2
    free[REOPENED_WALL] = False
    return free


def house_grid() -> np.ndarray:
    """The 0.5 m house as a ``Scene`` grid: 1.0 free, +inf wall."""
    return np.where(house_free(), 1.0, np.inf)


def room(q: Cell) -> str:
    """The room of a 0.5 m cell."""
    x, y = q
    if y >= 11:
        return "top room"
    if y >= 8:
        return "middle room" if x <= 6 else "right room"
    if y >= 6:
        return "hall"
    if x <= 2:
        return "left end"
    return "band" if x <= 6 else "bottom room"


# ---------------------------------------------------------------------------------------------
# The placements: test_tiers' twelve of the low, medium and high tiers (0.25 m cells), and the
# same made fair on the 0.5 m house
# ---------------------------------------------------------------------------------------------

TIERS = ("low", "medium", "high")
SUITE_GOALS = {"meeting": ("cooperation", "competition", "conflict", "coercion"),
               "exchange": ("exchange",)}
# name: (tier, suite, starts, objects, destinations, what test_tiers says the robots do)
PLACEMENTS: Dict[str, tuple] = {
    "near_miss": ("low", "meeting", ((7, 23), (3, 11)), ((6, 24), (21, 21)), ((3, 7), (4, 12)),
                  "R1 comes down from the top room while R2 crosses the hall below it on its "
                  "way to the right room: 1 m apart at the closest, no contact."),
    "same_room_later": ("low", "meeting", ((9, 23), (4, 8)), ((21, 21), (11, 22)),
                        ((8, 23), (6, 24)),
                        "R1 leaves the top room for the right room and returns; R2 comes up "
                        "to the top room meanwhile and finishes there."),
    "hall_crossing": ("medium", "meeting", ((9, 22), (4, 13)), ((4, 8), (3, 14)),
                      ((11, 22), (14, 7)),
                      "R1 walks from the top room to the bottom-left alcove and back; R2 "
                      "carries from the left nook to the bottom room.  Their routes cross in "
                      "the open hall at the same time, with room to pass."),
    "doorway_passing": ("medium", "meeting", ((3, 8), (8, 23)), ((10, 22), (11, 23)),
                        ((2, 7), (3, 13)),
                        "R1 goes up to the top room and back while R2 comes down from it: "
                        "they meet at the top room's doorway, which is wide enough to pass."),
    "head_on_blocked_goal": ("high", "meeting", ((4, 12), (8, 23)), ((9, 23), (4, 12)),
                             ((4, 7), (4, 8)),
                             "The robots swap ends along the same route (R2's object lies on "
                             "R1's start cell) and R2 parks in front of R1's destination."),
    "head_on_corridor": ("high", "meeting", ((10, 23), (4, 13)), ((8, 23), (6, 24)),
                         ((4, 12), (11, 9)),
                         "R1 carries down from the top room to the left nook while R2 walks "
                         "up from the nook to the top room: head-on through the doorway."),
    "own_object_closer": ("low", "exchange", ((3, 9), (3, 11)), ((2, 7), (3, 14)),
                          ((6, 24), (4, 7)),
                          "Both robots start in the bottom-left alcove, each three cells from "
                          "its own object and five from the other's."),
    "own_object_closer_far": ("low", "exchange", ((3, 9), (3, 11)), ((14, 1), (8, 23)),
                              ((3, 7), (11, 21)),
                              "The same starts; R1's object at the end of the bottom "
                              "corridor, R2's in the top room; each robot half a metre "
                              "nearer to its own."),
    "start_together": ("medium", "exchange", ((4, 8), (4, 9)), ((9, 26), (11, 24)),
                       ((12, 9), (11, 22)),
                       "The robots start side by side in the alcove and both objects lie in "
                       "the top room: each as far from the other's object as from its own."),
    "equally_far": ("medium", "exchange", ((9, 23), (7, 23)), ((14, 7), (4, 7)),
                    ((3, 7), (11, 22)),
                    "Both robots start in the top room; R1's object in the bottom room, R2's "
                    "in the alcove: each as far from the other's object as from its own."),
    "objects_crossed": ("high", "exchange", ((4, 9), (8, 23)), ((11, 24), (3, 11)),
                        ((4, 9), (4, 14)),
                        "R1 starts in the alcove, where R2's object lies; R2 starts in the "
                        "top room, where R1's object lies."),
    "objects_crossed_long_haul": ("high", "exchange", ((3, 12), (8, 22)), ((9, 26), (4, 8)),
                                  ((21, 21), (4, 12)),
                                  "R1 starts in the left nook, next to R2's object and "
                                  "destination; R2 in the top room, under R1's object."),
}
# name: ((starts, objects, destinations), candidate destinations) on the 0.5 m house: the
# result of ``search`` (checked by ``--search``)
SCENES: Dict[str, tuple] = {
    "near_miss": ((((3, 11), (2, 5)), ((5, 12), (8, 8)), ((1, 3), (2, 6))),
                  ((1, 3), (2, 6), (3, 12), (10, 9), (11, 2), (6, 9))),
    "same_room_later": ((((4, 11), (0, 4)), ((8, 8), (5, 11)), ((4, 9), (3, 12))),
                        ((4, 9), (3, 12), (10, 9), (11, 2), (1, 3), (6, 9))),
    "hall_crossing": ((((4, 11), (7, 7)), ((2, 5), (1, 6)), ((5, 11), (7, 3))),
                      ((5, 11), (7, 3), (3, 12), (10, 9), (11, 2), (1, 3))),
    "doorway_passing": ((((4, 9), (3, 12)), ((4, 11), (5, 10)), ((1, 3), (1, 6))),
                        ((1, 3), (1, 6), (10, 9), (11, 2), (6, 9), (0, 5))),
    "head_on_blocked_goal": ((((10, 7), (4, 11)), ((3, 11), (10, 7)), ((1, 3), (0, 4))),
                             ((1, 3), (0, 4), (3, 12), (10, 9), (11, 2), (6, 9))),
    "head_on_corridor": ((((5, 11), (3, 6)), ((3, 12), (4, 11)), ((1, 6), (5, 4))),
                         ((1, 6), (5, 4), (10, 9), (11, 2), (1, 3), (6, 9))),
    "own_object_closer": ((((1, 4), (1, 5)), ((1, 3), (5, 7)), ((3, 12), (0, 4))),
                          ((3, 12), (0, 4), (10, 9), (11, 2), (6, 9), (9, 6))),
    "own_object_closer_far": ((((1, 4), (1, 5)), ((7, 3), (3, 12)), ((1, 3), (5, 10))),
                              ((1, 3), (5, 10), (10, 9), (11, 2), (6, 9), (0, 5))),
    "start_together": ((((2, 4), (1, 4)), ((4, 11), (3, 12)), ((4, 4), (5, 11))),
                       ((4, 4), (5, 11), (10, 9), (11, 2), (1, 3), (6, 9))),
    "equally_far": ((((4, 11), (3, 11)), ((7, 3), (2, 5)), ((1, 4), (5, 11))),
                    ((1, 4), (5, 11), (3, 12), (10, 9), (11, 2), (6, 9))),
    "objects_crossed": ((((2, 4), (3, 12)), ((4, 11), (0, 4)), ((2, 4), (2, 6))),
                        ((2, 4), (2, 6), (10, 9), (11, 2), (1, 3), (6, 9))),
    "objects_crossed_long_haul": ((((1, 6), (4, 11)), ((4, 12), (0, 4)), ((10, 10), (7, 7))),
                                  ((10, 10), (7, 7), (3, 12), (11, 2), (1, 3), (6, 9))),
}
# candidate destinations that are no robot's: one per room, in order of use
DISTRACTORS = ((3, 12), (10, 9), (11, 2), (1, 3), (6, 9), (0, 5), (9, 6), (7, 1))


def tier_scene(name: str) -> Scene:
    """The fair placement ``name`` on the 0.5 m house as a :class:`Scene`."""
    (starts, pickups, destinations), candidates = SCENES[name]
    return Scene(grid=house_grid(), starts=starts, pickups=pickups, destinations=destinations,
                 candidate_cells=candidates, name=name)


def cells_of(name: str, fair: bool = True) -> dict:
    if fair:
        st, pk, de = SCENES[name][0]
    else:
        st, pk, de = PLACEMENTS[name][2:5]
    return dict(starts=st, pickups=pk, destinations=de)


# ---------------------------------------------------------------------------------------------
# A* and fairness
# ---------------------------------------------------------------------------------------------

FAIR = 0.10
BIG = 1 << 30


def astar(free: np.ndarray, a: Cell, b: Cell) -> List[Cell]:
    """A* on the 4-connected grid (Manhattan heuristic: admissible and consistent) -> one
    shortest path from a to b, both included."""
    a, b = tuple(a), tuple(b)
    X, Y = free.shape

    def h(c):
        return abs(c[0] - b[0]) + abs(c[1] - b[1])

    g = {a: 0}
    came: Dict[Cell, Cell] = {}
    heap = [(h(a), 0, a)]
    while heap:
        _, gc, c = heapq.heappop(heap)
        if c == b:
            path = [c]
            while path[-1] != a:
                path.append(came[path[-1]])
            return path[::-1]
        if gc > g[c]:
            continue
        for dx, dy in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < X and 0 <= n[1] < Y and free[n] and gc + 1 < g.get(n, BIG):
                g[n] = gc + 1
                came[n] = c
                heapq.heappush(heap, (gc + 1 + h(n), gc + 1, n))
    raise ValueError(f"no path from {a} to {b}")


def parking_ok(free: np.ndarray, cells: dict) -> bool:
    """Whether each robot can still do its task -- start -> object -> destination -- with its
    partner parked on the partner's destination (a robot that is done stays there)."""
    for j in (0, 1):
        i = 1 - j
        blocked = free.copy()
        blocked[tuple(cells["destinations"][j])] = False
        way = (cells["starts"][i], cells["pickups"][i], cells["destinations"][i])
        if any(not blocked[tuple(c)] for c in way):
            return False
        try:
            astar(blocked, way[0], way[1])
            astar(blocked, way[1], way[2])
        except ValueError:
            return False
    return True


def fairness(free: np.ndarray, cells: dict) -> dict:
    """A*: each robot's legs (start -> object, object -> destination) and task length L, in
    steps, the ratio max(L) / min(L), and whether that is within FAIR."""
    legs = [(len(astar(free, cells["starts"][i], cells["pickups"][i])) - 1,
             len(astar(free, cells["pickups"][i], cells["destinations"][i])) - 1)
            for i in (0, 1)]
    L = [sum(leg) for leg in legs]
    ratio = max(L) / max(min(L), 1)
    return dict(legs=legs, L=L, ratio=ratio, fair=ratio <= 1.0 + FAIR + 1e-12)


# ---------------------------------------------------------------------------------------------
# test_tiers' tier geometry (Graph, _joint_cost, tube, scene_geometry, own_detour,
# tier_in_words): unchanged, but for the resolution as a parameter
# ---------------------------------------------------------------------------------------------

NEAR_M = 0.25           # "makes no difference": less than this either way
FAR_M = 1.5             # "really out of their way": this far or more
MOVES = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=np.int64)   # last = stay


class Graph:
    """4-connected free cells: ids, neighbour table, all-pairs geodesic distances."""

    def __init__(self, free: np.ndarray) -> None:
        self.free = np.asarray(free, dtype=bool)                          # (X, Y)
        self.cells = np.argwhere(self.free).astype(np.int64)              # (N, 2) (ix, iy)
        self.N = int(self.cells.shape[0])
        X, Y = self.free.shape
        self.cell_id = np.full((X, Y), -1, dtype=np.int64)                # (X, Y); -1 blocked
        self.cell_id[self.cells[:, 0], self.cells[:, 1]] = np.arange(self.N)
        tgt = self.cells[:, None, :] + MOVES[None, :, :]                  # (N, 5, 2)
        inside = ((tgt[..., 0] >= 0) & (tgt[..., 0] < X)
                  & (tgt[..., 1] >= 0) & (tgt[..., 1] < Y))               # (N, 5)
        nid = np.full((self.N, 5), -1, dtype=np.int64)
        nid[inside] = self.cell_id[tgt[..., 0][inside], tgt[..., 1][inside]]
        self.nb = np.where(nid >= 0, nid, np.arange(self.N)[:, None])     # (N, 5); wall = stay
        self.D = np.stack([self._bfs(c) for c in range(self.N)])          # (N, N) steps
        assert (self.D < BIG).all(), "the free cells are not one 4-connected component"

    def _bfs(self, src: int) -> np.ndarray:
        d = np.full(self.N, BIG, dtype=np.int64)
        d[src] = 0
        queue = deque([src])
        while queue:
            c = queue.popleft()
            for n in self.nb[c, :4]:
                if d[n] == BIG:
                    d[n] = d[c] + 1
                    queue.append(n)
        return d

    def ids(self, cells) -> np.ndarray:
        out = np.array([self.cell_id[tuple(c)] for c in cells], dtype=np.int64)
        assert (out >= 0).all(), f"blocked task cell in {cells}"
        return out


@njit(cache=False)
def _joint_cost(nb, start0, start1, pick0, pick1, dest0, dest1, cap):
    """Least sum of the two completion times over collision-free joint plans (Dial's
    algorithm on the joint state (x0, x1, phase0, phase1); step cost = robots still working;
    no shared cell, no swap; a robot that has placed stays on its destination).
    -> the cost, or -1 when no plan finishes both tasks within `cap`."""
    N = nb.shape[0]
    dist = np.full(9 * N * N, -1, dtype=np.int32)
    f0 = 1 if start0 == pick0 else 0
    if f0 == 1 and start0 == dest0:
        f0 = 2
    f1 = 1 if start1 == pick1 else 0
    if f1 == 1 and start1 == dest1:
        f1 = 2
    s0 = ((f0 * 3 + f1) * N + start0) * N + start1
    b0 = [np.int64(s0)]
    b1 = [np.int64(0)]
    b1.pop()
    b2 = [np.int64(0)]
    b2.pop()
    dist[s0] = 0
    d = 0
    while d <= cap:
        if len(b0) == 0 and len(b1) == 0 and len(b2) == 0:
            return -1
        for k in range(len(b0)):
            s = b0[k]
            if dist[s] != d:
                continue                                    # stale queue entry
            x1 = s % N
            x0 = (s // N) % N
            ph = s // (N * N)
            g0, g1 = ph // 3, ph % 3
            if g0 == 2 and g1 == 2:
                return d
            step = (1 if g0 != 2 else 0) + (1 if g1 != 2 else 0)
            for a0 in range(5):
                if g0 == 2 and a0 != 4:
                    continue
                n0 = nb[x0, a0]
                if a0 != 4 and n0 == x0:
                    continue
                for a1 in range(5):
                    if g1 == 2 and a1 != 4:
                        continue
                    n1 = nb[x1, a1]
                    if a1 != 4 and n1 == x1:
                        continue
                    if n0 == n1 or (n0 == x1 and n1 == x0):
                        continue                            # shared cell / swap
                    h0, h1 = g0, g1
                    if h0 == 0 and n0 == pick0:
                        h0 = 1
                    if h0 == 1 and n0 == dest0:
                        h0 = 2
                    if h1 == 0 and n1 == pick1:
                        h1 = 1
                    if h1 == 1 and n1 == dest1:
                        h1 = 2
                    t = ((h0 * 3 + h1) * N + n0) * N + n1
                    nd = d + step
                    if dist[t] < 0 or nd < dist[t]:
                        dist[t] = nd
                        if step == 1:
                            b1.append(np.int64(t))
                        else:
                            b2.append(np.int64(t))
        b0 = b1
        b1 = b2
        b2 = [np.int64(0)]
        b2.pop()
        d += 1
    return -1


def tube(G: Graph, s: int, p: int, g: int) -> list:
    """Cells a robot can occupy at time t = 0..T on a fastest start -> object -> destination
    route -> one id array per t."""
    D = G.D
    L1, L2 = int(D[s, p]), int(D[p, g])
    out = []
    for t in range(L1 + L2 + 1):
        if t <= L1:
            out.append(np.flatnonzero((D[s] == t) & (D[p] == L1 - t)))
        else:
            out.append(np.flatnonzero((D[p] == t - L1) & (D[g] == L1 + L2 - t)))
    return out


def scene_geometry(G: Graph, starts, pickups, destinations, joint: bool = True) -> dict:
    """The distances of one placement, in grid steps (test_tiers): T, first, gap, give_way,
    detour, cross, trade, shared."""
    D = G.D
    s, p, g = G.ids(starts), G.ids(pickups), G.ids(destinations)
    L1 = [int(D[s[i], p[i]]) for i in (0, 1)]
    L2 = [int(D[p[i], g[i]]) for i in (0, 1)]
    T = [L1[i] + L2[i] for i in (0, 1)]
    tubes = [tube(G, s[i], p[i], g[i]) for i in (0, 1)]
    horizon = max(T)

    def at(i: int, t: int) -> np.ndarray:
        return tubes[i][min(t, T[i])]

    gap = BIG
    for t in range(horizon + 1):
        gap = min(gap, int(D[np.ix_(at(0, t), at(1, t))].min()))
    give_way = "not computed"
    if joint:
        cost = int(_joint_cost(G.nb, int(s[0]), int(s[1]), int(p[0]), int(p[1]), int(g[0]),
                               int(g[1]), 4 * (T[0] + T[1]) + 8 * G.N))
        give_way = None if cost < 0 else cost - (T[0] + T[1])
    detour = []
    for i in (0, 1):
        j, best = 1 - i, BIG
        for t in range(T[j] + 1):
            near = np.flatnonzero((D[:, at(j, t)] <= 1).any(axis=1))     # next to j at time t
            ok = D[s[i], near] <= t                                      # before its pickup
            if ok.any():
                best = min(best, int((t + D[near[ok], p[i]] + L2[i] - T[i]).min()))
            ok = L1[i] + D[p[i], near] <= t                              # carrying
            if ok.any():
                best = min(best, int((t + D[near[ok], g[i]] - T[i]).min()))
            ok = T[i] + D[g[i], near] <= t                               # after placing
            if ok.any():
                best = min(best, int(D[g[i], near[ok]].min()))
        detour.append(None if best >= BIG else best)
    cross = [int(D[s[i], p[1 - i]]) - L1[i] for i in (0, 1)]
    swapped = sum(int(D[s[i], p[1 - i]]) + L2[1 - i] for i in (0, 1))
    cells = [set(np.concatenate(tb).tolist()) for tb in tubes]
    return dict(T=T, first=L1, gap=gap, give_way=give_way, detour=detour, cross=cross,
                trade=swapped - sum(T), shared=len(cells[0] & cells[1]))


def own_detour(category: str, geo: dict, rule: str = "together") -> tuple:
    """The pair's signed detour -> (steps or None, forced)."""
    if category == "exchange":
        if min(geo["first"]) == 0:
            return None, False
        return (geo["trade"] if rule == "together" else max(geo["cross"])), False
    gw = geo["give_way"]
    if gw is None:
        return -np.inf, True
    if gw > 0:
        return -gw, True
    d = [x for x in geo["detour"] if x is not None]
    return (min(d) if d else None), False


def tier_in_words(steps, forced: bool, resolution: float) -> str:
    """The tier of a detour by test_tiers' rules: distances only."""
    if steps is None:
        return "incongruent"
    d = steps * resolution
    if forced or d <= -NEAR_M:
        return "high"
    if d < NEAR_M:
        return "medium"
    return "low" if d < FAR_M else "incongruent"


def tier(G: Graph, cells: dict, suite: str, resolution: float) -> Tuple[str, dict]:
    """-> (the tier of a placement, its geometry)."""
    geo = scene_geometry(G, cells["starts"], cells["pickups"], cells["destinations"])
    steps, forced = own_detour("exchange" if suite == "exchange" else "meeting", geo)
    return tier_in_words(steps, forced, resolution), geo


# ---------------------------------------------------------------------------------------------
# Making a placement fair on the 0.5 m house
# ---------------------------------------------------------------------------------------------

SLOTS = ("start R1", "start R2", "object R1", "object R2", "destination R1",
         "destination R2")


def port_cell(free: np.ndarray, fine: Cell) -> Cell:
    """The 0.5 m cell holding a 0.25 m cell, or the free one with the nearest centre."""
    q = (fine[0] // 2, fine[1] // 2)
    if free[q]:
        return q
    centre = ((fine[0] + 0.5) * FINE, (fine[1] + 0.5) * FINE)
    cells = np.argwhere(free)
    d = np.hypot((cells[:, 0] + 0.5) * RESOLUTION - centre[0],
                 (cells[:, 1] + 0.5) * RESOLUTION - centre[1])
    return tuple(int(v) for v in cells[int(np.argmin(d))])


def _pattern(cells: Sequence[Cell]) -> frozenset:
    return frozenset((i, j) for i, j in itertools.combinations(range(6), 2)
                     if cells[i] == cells[j])


def _as_dict(cells: Sequence[Cell]) -> dict:
    return dict(starts=(cells[0], cells[1]), pickups=(cells[2], cells[3]),
                destinations=(cells[4], cells[5]))


def candidates_for(cells: Sequence[Cell]) -> tuple:
    """The two destinations and the first four DISTRACTORS that are no task cell and not
    next to a destination."""
    d0, d1 = cells[4], cells[5]
    out = [d0, d1]
    for p in DISTRACTORS:
        if len(out) == 6:
            break
        if p in cells or any(abs(p[0] - d[0]) + abs(p[1] - d[1]) < 2 for d in (d0, d1)):
            continue
        out.append(p)
    return tuple(out)


def search(name: str, G: Optional[Graph] = None, max_moved: int = 3,
           keep: str = "all") -> dict:
    """Port test_tiers' placement ``name`` to the 0.5 m house and make it fair (module
    docstring).  keep: "all" -- every moved cell stays in its room; "objects" -- only the
    objects do (the fallback); "none" -- no room is kept (last resort)."""
    want, suite, st, pk, de = PLACEMENTS[name][:5]
    free = house_free()
    G = G or Graph(free)
    orig = [tuple(c) for c in (*st, *pk, *de)]
    init = [port_cell(free, c) for c in orig]
    for i in range(6):                       # a coincidence the coarser map makes
        for j in range(i):
            if init[i] == init[j] and orig[i] != orig[j]:
                cells = np.argwhere(free)
                d = np.abs(cells - np.array(init[i])).sum(axis=1)
                for k in np.argsort(d, kind="stable"):
                    q = tuple(int(v) for v in cells[k])
                    if q not in init:
                        init[i] = q
                        break
    want_pattern = _pattern(orig)
    assert _pattern(init) == want_pattern, (name, init)
    D, ids = G.D, G.cell_id
    free_cells = [tuple(int(v) for v in x) for x in G.cells]
    options = [[q for q in free_cells
                if keep == "none" or room(q) == room(init[m])
                or (keep == "objects" and m // 2 != 1)] for m in range(6)]

    def lengths(cells):
        s = [ids[x] for x in cells]
        return [int(D[s[0], s[2]] + D[s[2], s[4]]), int(D[s[1], s[3]] + D[s[3], s[5]])]

    found, best = [], None
    for k in range(max_moved + 1):
        trial = []
        for moved in itertools.combinations(range(6), k):
            for targets in itertools.product(*[options[m] for m in moved]):
                cells = list(init)
                for m, t in zip(moved, targets):
                    cells[m] = t
                if any(cells[m] == init[m] for m in moved):
                    continue
                L = lengths(cells)
                if min(L) == 0 or max(L) > (1 + FAIR) * min(L) + 1e-9:
                    continue
                if _pattern(cells) != want_pattern:
                    continue
                cost = sum(int(D[ids[init[m]], ids[cells[m]]]) for m in moved)
                trial.append((cost, abs(max(L) / min(L) - 1), k, moved, tuple(cells)))
        trial.sort()
        for cost, dev, kk, moved, cells in trial:
            if best is not None and cost > best:
                break
            if (parking_ok(free, _as_dict(cells))
                    and tier(G, _as_dict(cells), suite, RESOLUTION)[0] == want):
                found.append((cost, dev, kk, moved, cells))
                best = cost if best is None else min(best, cost)
                break
        if found and best <= 0:
            break
    if not found:
        if keep != "none":
            return search(name, G, max_moved, "objects" if keep == "all" else "none")
        raise RuntimeError(f"{name}: no fair placement of its tier")
    cost, _, _, moved, cells = min(found, key=lambda f: (f[0], f[1], f[2]))
    return dict(name=name, keep=keep, ported=_as_dict(init), cells=_as_dict(cells),
                candidates=candidates_for(cells), move_cost=cost,
                moved=[dict(slot=SLOTS[m], frm=init[m], to=cells[m], room=room(cells[m]),
                            steps=int(D[ids[init[m]], ids[cells[m]]])) for m in moved])


def report(G_fine: Graph, G: Graph) -> List[dict]:
    """Each scene's tier and A* task lengths as drawn (0.25 m) and fair (0.5 m)."""
    out = []
    for name, (want, suite, *_rest) in PLACEMENTS.items():
        before = fairness(fine_free(), cells_of(name, fair=False))
        t0, geo0 = tier(G_fine, cells_of(name, fair=False), suite, FINE)
        after = fairness(house_free(), cells_of(name))
        t1, geo = tier(G, cells_of(name), suite, RESOLUTION)
        # A* gives the breadth-first lengths of test_tiers' geometry
        assert before["L"] == geo0["T"] and after["L"] == geo["T"], name
        out.append(dict(name=name, tier=want, suite=suite, tier_as_drawn=t0, tier_fair=t1,
                        L_as_drawn=before["L"], ratio_as_drawn=before["ratio"],
                        L=after["L"], legs=after["legs"], ratio=after["ratio"],
                        fair=after["fair"], parking_ok=parking_ok(house_free(), cells_of(name)),
                        parking_ok_as_drawn=parking_ok(fine_free(), cells_of(name, fair=False)),
                        geometry=geo, candidates=SCENES[name][1]))
    return out


# ---------------------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------------------

COLOURS = ("#2ca25f", "#b2182b")                 # R1 green, R2 red (render.py)
HEADS = {"meeting": {"low": "close, but no contact", "medium": "they meet, with room to pass",
                     "high": "someone must give way"},
         "exchange": {"low": "own object a little nearer", "medium": "swapping makes no difference",
                      "high": "each nearer the other's object"}}


def _geometry_line(row: dict) -> str:
    g, r = row["geometry"], RESOLUTION
    if row["suite"] == "exchange":
        c = g["cross"]
        return (f"swap: {g['trade'] * r:+.1f} m together (R1 {c[0] * r:+.1f} m to R2's object, "
                f"R2 {c[1] * r:+.1f} m to R1's)")
    if row["tier"] == "high":
        return f"they cannot both drive their fastest routes: {g['give_way'] * r:.1f} m of waiting"
    if row["tier"] == "medium":
        return "fastest routes meet (no detour), room to pass"
    d = g["detour"]
    return f"no contact; detour to meet: R1 +{d[0] * r:.1f} m, R2 +{d[1] * r:.1f} m"


def draw(rows: List[dict], moves: Dict[str, list], suite: str, path: Path) -> None:
    """One suite's six fair scenes, a column per tier, each robot's A* route drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    ink, quiet, surface, wall = "#0b0b0b", "#52514e", "#fcfcfb", "#1c1c1c"
    free = house_free()
    X, Y = free.shape
    by = {row["name"]: row for row in rows if row["suite"] == suite}
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 14.4), dpi=130, facecolor=surface)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.79, bottom=0.05, hspace=0.36, wspace=0.08)
    who = {s: s.replace("start R", "R").replace("object R", "R").replace("destination R", "R")
           for s in SLOTS}
    for j, t in enumerate(TIERS):
        box = axes[0, j].get_position()
        fig.text((box.x0 + box.x1) / 2, 0.905, f"{t.upper()}\n{HEADS[suite][t]}", ha="center",
                 va="bottom", fontsize=13, fontweight="bold", color=ink)
        for i, name in enumerate(n for n in by if by[n]["tier"] == t):
            row, ax = by[name], axes[i, j]
            ax.imshow(np.where(free.T, 1.0, 0.0), origin="lower",
                      cmap=ListedColormap([wall, "white"]),
                      extent=(-0.5, X - 0.5, -0.5, Y - 0.5), interpolation="nearest")
            c = cells_of(name)
            for k in (0, 1):
                route = (astar(free, c["starts"][k], c["pickups"][k])[:-1]
                         + astar(free, c["pickups"][k], c["destinations"][k]))
                xy = np.array(route, float) + (0.13 if k else -0.13)
                ax.plot(xy[:, 0], xy[:, 1], "-", color=COLOURS[k], lw=2.4, alpha=0.75,
                        solid_capstyle="round", zorder=2)
            for q in row["candidates"]:
                if tuple(q) not in c["destinations"]:
                    ax.plot(q[0], q[1], marker="*", ms=11, mfc="none", mec="0.6", mew=1.0,
                            zorder=3)
            for k in (0, 1):
                s, o, d = c["starts"][k], c["pickups"][k], c["destinations"][k]
                ax.plot(d[0], d[1], marker="*", ms=19, mfc="white", mec=COLOURS[k], mew=2.0,
                        zorder=4)
                ax.plot(o[0], o[1], marker="s", ms=10, mfc=COLOURS[k], mec="k", mew=0.9,
                        zorder=5)
                ax.plot(s[0], s[1], marker="o", ms=11, mfc=COLOURS[k], mec="k", mew=0.9,
                        zorder=6)
                ax.annotate(f"R{k + 1}", s, xytext=(7, 6), textcoords="offset points",
                            fontsize=9, fontweight="bold", color=ink, zorder=7,
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                                      alpha=0.85))
            ax.set_xlim(-0.5, X - 0.5)
            ax.set_ylim(-0.5, Y - 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("0.6")
            L = row["L"]
            moved = moves.get(name)
            if moved is None:
                moved_line = ""
            elif moved:
                moved_line = "moved: " + ", ".join(
                    f"{who[m['slot']]}'s {m['slot'].split()[0]} {m['steps'] * RESOLUTION:.1f} m"
                    for m in moved)
            else:
                moved_line = "ported as it was"
            ax.text(0.0, 1.20, name, transform=ax.transAxes, fontsize=12, fontweight="bold",
                    color=ink, va="bottom")
            ax.text(0.0, 1.135, _geometry_line(row), transform=ax.transAxes, fontsize=8.6,
                    color=ink, va="bottom")
            ax.text(0.0, 1.075, f"A* task: R1 {L[0] * RESOLUTION:.1f} m, R2 "
                                f"{L[1] * RESOLUTION:.1f} m  ratio {row['ratio']:.2f}  fair "
                                f"(as drawn: {row['ratio_as_drawn']:.2f})",
                    transform=ax.transAxes, fontsize=8.6, color=ink, va="bottom")
            ax.text(0.0, 1.015, moved_line, transform=ax.transAxes, fontsize=8.0, color=quiet,
                    va="bottom")
    title = {"meeting": "Fair tier scenes, the meeting suite (cooperation, competition, "
                        "conflict, coercion)",
             "exchange": "Fair tier scenes, the exchange suite"}[suite]
    fig.suptitle(title, x=0.015, y=0.995, ha="left", fontsize=14.5, fontweight="bold",
                 color=ink)
    fig.text(0.015, 0.965, "test_tiers' placements on the house map at 0.5 m (79 free cells), "
             "each made fair by the smallest moves of its task cells that keep its tier: A* task "
             "lengths (start -> object -> destination) within 10 %, and neither robot, parked on "
             "its destination, walls the other out.", fontsize=10, color=quiet)
    handles = [Line2D([], [], marker="o", ls="none", mfc="0.75", mec="k", ms=10, label="start"),
               Line2D([], [], marker="s", ls="none", mfc="0.75", mec="k", ms=9, label="object"),
               Line2D([], [], marker="*", ls="none", mfc="white", mec="0.3", ms=15,
                      label="destination"),
               Line2D([], [], marker="*", ls="none", mfc="none", mec="0.6", ms=11,
                      label="other candidate destination"),
               Line2D([], [], color=COLOURS[0], lw=2.4, label="R1's A* route"),
               Line2D([], [], color=COLOURS[1], lw=2.4, label="R2's A* route")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=10.5)
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-search", action="store_true",
                        help="do not make the scenes fair again (nor check SCENES)")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "tier_scenes")
    args = parser.parse_args()
    G_fine, G = Graph(fine_free()), Graph(house_free())
    rows = report(G_fine, G)
    print(f"{'scene':<26} {'tier':<7} {'suite':<9} {'A* as drawn (0.25 m)':<22} "
          f"{'fair at 0.5 m':<16} tier at 0.25 / 0.5 m")
    for r in rows:
        print(f"{r['name']:<26} {r['tier']:<7} {r['suite']:<9} "
              f"{r['L_as_drawn'][0]:>3} / {r['L_as_drawn'][1]:<3} ({r['ratio_as_drawn']:.2f})    "
              f"{r['L'][0]:>2} / {r['L'][1]:<2} ({r['ratio']:.2f})  "
              f"{r['tier_as_drawn']} / {r['tier_fair']}")
        assert r["fair"] and r["tier_fair"] == r["tier"] == r["tier_as_drawn"], r["name"]
        assert r["parking_ok"] and r["parking_ok_as_drawn"], r["name"]
    moves: Dict[str, list] = {}
    if not args.no_search:
        for name in PLACEMENTS:
            got = search(name, G)
            want = cells_of(name)
            same = (all(tuple(map(tuple, got["cells"][k])) == tuple(want[k]) for k in want)
                    and tuple(got["candidates"]) == tuple(SCENES[name][1]))
            moves[name] = got["moved"]
            print(f"{name:<26} search: {'= SCENES' if same else 'DIFFERS from SCENES'}"
                  f"{'' if got['keep'] == 'all' else '  (objects kept in their rooms)'}; moved "
                  + (", ".join(f"{m['slot']} {m['frm']} -> {m['to']}" for m in got["moved"])
                     or "nothing"))
            assert same, name
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scenes.json").write_text(json.dumps(
        [{k: v for k, v in r.items()} for r in rows], indent=1, default=int) + "\n")
    for suite in ("meeting", "exchange"):
        draw(rows, moves, suite, args.output / f"fair_scenes_{suite}.png")
    print(args.output)


if __name__ == "__main__":
    main()
