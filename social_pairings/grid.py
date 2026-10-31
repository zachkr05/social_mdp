"""The grid two robots move on: cells, shortest ways, and how a pair of moves resolves."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from numba import njit, prange
from scipy.ndimage import label as _label_components

from .config import Config, SolverFailure
from .constants import MOVES

__all__ = ["GridWorld", "as_cell"]

_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


def as_cell(p) -> Tuple[int, int]:
    """A grid cell (ix, iy) from anything array-like; rejects non-integer coordinates."""
    q = np.asarray(p, dtype=float)
    if q.shape != (2,) or not np.all(np.isfinite(q)) or not np.all(q == np.rint(q)):
        raise ValueError(f"task coordinates must be integer grid cells, got {p!r}")
    return int(q[0]), int(q[1])


@njit(parallel=True, cache=True)
def _pair_moves(N, nb0, nb1, cells, sep2, out):
    """out[x0 * N + x1, m0 * 5 + m1] = the pair of cells after robot 0 tries move m0 and
    robot 1 tries move m1 (4 = stay), as n0 * N + n1.  nb0 / nb1 (N, 5): where each move
    takes robot 0 / robot 1 (the two differ only where the robots have rooms of their own).

    Collision rules: two robots never swap cells head-on (both stay); they never end on the
    same cell or closer than the separation -- a robot that stays keeps its cell, otherwise
    robot 0 moves and robot 1 stays (robot 0 stays as well if that would still clash)."""
    for pair in prange(N * N):
        x0 = pair // N
        x1 = pair - x0 * N
        for m0 in range(5):
            t0 = nb0[x0, m0]
            for m1 in range(5):
                t1 = nb1[x1, m1]
                if t0 == x1 and t1 == x0:                   # head-on swap: both stay
                    n0 = x0
                    n1 = x1
                else:
                    clash = t0 == t1
                    if (not clash) and sep2 > 1.0:
                        dx = cells[t0, 0] - cells[t1, 0]
                        dy = cells[t0, 1] - cells[t1, 1]
                        clash = dx * dx + dy * dy < sep2 - 1e-9
                    if clash:
                        if t1 == x1:                        # robot 1 stays: robot 0 is blocked
                            n0 = x0
                            n1 = x1
                        else:                               # robot 0 has priority
                            n1 = x1
                            again = t0 == x1
                            if (not again) and sep2 > 1.0:
                                dx = cells[t0, 0] - cells[x1, 0]
                                dy = cells[t0, 1] - cells[x1, 1]
                                again = dx * dx + dy * dy < sep2 - 1e-9
                            n0 = x0 if again else t0
                    else:
                        n0 = t0
                        n1 = t1
                out[pair, m0 * 5 + m1] = n0 * N + n1


class GridWorld:
    """A 4-connected grid with two robots, their pickup cells and K candidate destinations.

    grid : (X, Y) float, ``+inf`` = blocked, finite > 0 = free; axis 0 is world x.
    Only the free component that holds robot 0's start is kept (N cells, numbered in
    row-major order); every start, pickup and candidate must lie in it.
    rooms : None, or for each robot the cells it may stand on (``Scene.rooms``).  A robot
        never leaves its room: a move out of it is no move of that robot at all, as a move
        into a wall is.  A cell may belong to both rooms (a doorway).

    Attributes: ``cells`` (N, 2) int; ``cell_id`` (X, Y) int, -1 outside; ``nb`` (N, 5) the
    cell reached by each move (itself when the move is illegal) and ``valid`` (N, 5), on
    the free cells whatever the rooms; ``allowed`` (2, N) bool, the cells each robot may
    stand on, and ``nb_by`` / ``valid_by`` (2, N, 5), each robot's own moves (the same as
    ``nb`` / ``valid`` without rooms); ``start``, ``pickup`` (2,), ``dest`` (K,) cell ids;
    ``d_goal`` (K, N) steps to each candidate on the free cells, ``d_goal_by`` (2, K, N)
    each robot's own steps (inf: no way inside its room); ``pair_moves`` (N^2, 25) int32;
    ``pair_ok`` (N^2,) bool, False where the two robots would overlap or one of them would
    stand outside its room.
    """

    def __init__(self, grid, starts: Sequence, pickups: Sequence, candidates: Sequence,
                 cfg: Config, goal_ids: Sequence[int], rooms=None) -> None:
        self.cfg = cfg
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2 or np.isnan(grid).any() or np.isneginf(grid).any():
            raise ValueError("grid must be 2-D with finite (> 0) free cells and +inf obstacles")
        free = np.isfinite(grid)                                            # (X, Y)
        if np.any(grid[free] <= 0):
            raise ValueError("free cells must hold values > 0 (+inf marks obstacles)")
        self.shape = grid.shape
        X, Y = grid.shape
        self.starts = tuple(as_cell(p) for p in starts)
        self.pickups = tuple(as_cell(p) for p in pickups)
        self.goals = tuple(as_cell(p) for p in candidates)
        if len(self.starts) != 2 or len(self.pickups) != 2 or not self.goals:
            raise ValueError("need two starts, two pickups and at least one candidate goal")
        tasks = self.starts + self.pickups + self.goals
        bad = [c for c in tasks if not (0 <= c[0] < X and 0 <= c[1] < Y) or not free[c]]
        if bad:
            raise SolverFailure(f"task cells blocked or out of bounds: {bad}")
        labels, _ = _label_components(free, structure=_CROSS)
        component = labels == labels[self.starts[0]]                        # (X, Y)
        off = [c for c in tasks if not component[c]]
        if off:
            raise SolverFailure(f"task cells not 4-connected to robot 0's start: {off}")

        self.cells = np.argwhere(component).astype(np.int64)                # (N, 2)
        self.N = int(self.cells.shape[0])
        self.NN = self.N * self.N
        self.cell_id = np.full((X, Y), -1, dtype=np.int64)
        self.cell_id[self.cells[:, 0], self.cells[:, 1]] = np.arange(self.N)
        target = self.cells[:, None, :] + MOVES[None, :, :]                 # (N, 5, 2)
        inside = ((target[..., 0] >= 0) & (target[..., 0] < X)
                  & (target[..., 1] >= 0) & (target[..., 1] < Y))           # (N, 5)
        nid = np.full((self.N, 5), -1, dtype=np.int64)
        nid[inside] = self.cell_id[target[..., 0][inside], target[..., 1][inside]]
        self.valid = nid >= 0                                               # (N, 5); stay: always
        self.nb = np.where(self.valid, nid, np.arange(self.N)[:, None])     # (N, 5)

        def ids(cells_) -> np.ndarray:
            return np.array([self.cell_id[c] for c in cells_], dtype=np.int64)

        self.start, self.pickup, self.dest = ids(self.starts), ids(self.pickups), ids(self.goals)
        self.K = int(self.dest.size)
        # ---- rooms: the cells each robot may stand on ------------------------------------------
        self.allowed = np.ones((2, self.N), dtype=np.bool_)                 # (2, N)
        self.rooms = None
        if rooms is not None:
            if len(rooms) != 2:
                raise ValueError("rooms: one collection of cells per robot (or None)")
            kept = []
            for i in (0, 1):
                room = tuple(dict.fromkeys(as_cell(c) for c in rooms[i]))
                off = [c for c in room
                       if not (0 <= c[0] < X and 0 <= c[1] < Y) or self.cell_id[c] < 0]
                if off:
                    raise SolverFailure(f"robot {i}'s room holds cells that are blocked, out of "
                                        f"bounds or not connected to robot 0's start: {off}")
                self.allowed[i] = False
                self.allowed[i, ids(room)] = True
                if not self.allowed[i, self.start[i]]:
                    raise SolverFailure(f"robot {i} starts outside its room, at "
                                        f"{self.starts[i]}")
                kept.append(room)
            self.rooms = tuple(kept)
        self.has_rooms = not bool(self.allowed.all())
        # each robot's own moves: inside its room.  (On a cell outside it -- where it never
        # stands -- it has no move but "stay".)
        self.valid_by = np.stack([self.valid.copy(), self.valid.copy()])    # (2, N, 5)
        if self.has_rooms:
            for i in (0, 1):
                self.valid_by[i, :, :4] &= (self.allowed[i][self.nb[:, :4]]
                                            & self.allowed[i][:, None])
        self.nb_by = np.where(self.valid_by, self.nb[None, :, :],
                              np.arange(self.N)[None, :, None])             # (2, N, 5)
        self.sep = cfg.separation_cells
        if self.conflict(self.start[0], self.start[1]):
            raise SolverFailure(
                f"robot starts {self.starts} overlap (separation {self.sep:.2f} cells)")
        self.d_goal = np.stack([self.distances(g) for g in self.dest])      # (K, N)
        if not np.isfinite(self.d_goal).all():
            raise SolverFailure("a candidate destination is unreachable")
        self.d_goal_by = np.stack([[self.distances(g, i) for g in self.dest]
                                   for i in (0, 1)])                        # (2, K, N)
        self.true_goals = tuple(int(g) for g in goal_ids)
        self.true_dest = tuple(int(self.dest[g]) for g in self.true_goals)

        if 36 * self.NN >= 2 ** 31:
            raise SolverFailure(f"{self.N} free cells: use a coarser grid")
        diff = self.cells[:, None, :] - self.cells[None, :, :]              # (N, N, 2)
        self.pair_ok = np.ascontiguousarray(
            ((diff * diff).sum(-1) >= self.sep ** 2 - 1e-9).reshape(-1))    # (N^2,)
        if self.has_rooms:                       # a robot outside its room: never reached
            self.pair_ok &= (self.allowed[0][:, None] & self.allowed[1][None, :]).reshape(-1)
        self.pair_moves = np.empty((self.NN, 25), dtype=np.int32)
        _pair_moves(self.N, np.ascontiguousarray(self.nb_by[0]),
                    np.ascontiguousarray(self.nb_by[1]), self.cells, float(self.sep ** 2),
                    self.pair_moves)

    def distances(self, src: int, robot=None) -> np.ndarray:
        """(N,) shortest 4-connected step count from cell id ``src`` -- on the free cells,
        or (``robot`` = 0 / 1) by that robot's own moves, inside its room: inf to and from
        every cell outside it."""
        dist = np.full(self.N, np.inf)
        if robot is None:
            nb4, ok4 = self.nb[:, :4], self.valid[:, :4]
        else:
            if not self.allowed[robot, src]:
                return dist
            nb4, ok4 = self.nb_by[robot][:, :4], self.valid_by[robot][:, :4]
        dist[src] = 0.0
        frontier = np.array([src], dtype=np.int64)
        depth = 0
        while frontier.size:
            depth += 1
            reached = nb4[frontier][ok4[frontier]]
            reached = np.unique(reached[np.isinf(dist[reached])])
            dist[reached] = depth
            frontier = reached
        return dist

    def conflict(self, p, q):
        """True where robots on cell ids p and q would overlap.  Broadcasts."""
        diff = self.cells[p] - self.cells[q]
        return np.sum(diff * diff, axis=-1) < self.sep ** 2 - 1e-9

    def cell(self, x: int) -> Tuple[int, int]:
        return int(self.cells[x, 0]), int(self.cells[x, 1])
