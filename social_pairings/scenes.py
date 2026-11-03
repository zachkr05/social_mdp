"""Scenes: a grid, two robots, their objects and the candidate destinations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["Scene", "six_object_scene", "two_room_scene", "exchange_scene", "gate_scene",
           "GATE_MAP"]

Cell = Tuple[int, int]


@dataclass(frozen=True)
class Scene:
    """One pick-and-place scene.

    grid : (X, Y) float, ``+inf`` = wall, finite > 0 = free; axis 0 is world x, cells are
        (ix, iy).
    starts, pickups, destinations : one cell per robot.  Robot i's own object lies on
        ``pickups[i]`` and belongs on ``destinations[i]``.
    candidate_cells : the cells a robot holds possible for its partner's destination (the
        two real destinations are added if missing).  Belief index k = position in this list.
    objects : the spare objects -- "goal" (none), "candidates" (one on every candidate cell
        that is neither a pickup nor a destination) or a list of cells.
    handoff_cells : cells on which a robot may put its partner's object down, where it lies
        until somebody picks it up (neither an object's place nor a candidate destination).
    rooms : None, or for each robot the cells it may stand on.  A robot never leaves its
        room; a cell in both rooms (a doorway) can be entered by either.
    """
    grid: np.ndarray
    starts: Tuple[Cell, Cell]
    pickups: Tuple[Cell, Cell]
    destinations: Tuple[Cell, Cell]
    candidate_cells: Tuple[Cell, ...]
    objects: Union[str, Tuple[Cell, ...]] = "goal"
    name: str = "scene"
    handoff_cells: Tuple[Cell, ...] = ()
    rooms: Optional[Tuple[Tuple[Cell, ...], Tuple[Cell, ...]]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", np.asarray(self.grid, dtype=float))
        for field in ("starts", "pickups", "destinations", "candidate_cells", "handoff_cells"):
            object.__setattr__(self, field, _cells(getattr(self, field)))
        if not isinstance(self.objects, str):
            object.__setattr__(self, "objects", _cells(self.objects))
        if self.rooms is not None:
            if len(self.rooms) != 2:
                raise ValueError("rooms: one collection of cells per robot (or None)")
            object.__setattr__(self, "rooms", tuple(_cells(room) for room in self.rooms))

    @property
    def candidates(self) -> Tuple[Cell, ...]:
        """The candidate destinations in belief order."""
        return tuple(dict.fromkeys((*self.candidate_cells, *self.destinations)))

    @property
    def object_cells(self) -> Tuple[Cell, ...]:
        """Every object's place, the two own objects first."""
        from .world import object_cells
        return object_cells(self.objects, self.pickups, self.destinations, self.candidate_cells)

    def to_json(self) -> dict:
        return dict(name=self.name, grid=np.isfinite(self.grid).astype(int).tolist(),
                    starts=self.starts, pickups=self.pickups, destinations=self.destinations,
                    candidates=self.candidates, objects=self.object_cells,
                    handoff_cells=self.handoff_cells, rooms=self.rooms)


def _cells(values: Sequence) -> tuple:
    return tuple((int(c[0]), int(c[1])) for c in values)


def two_room_scene(objects: Union[str, Sequence[Cell]] = "goal") -> Scene:
    """Two 3 x 5 rooms joined by a doorway two cells wide.  Each robot starts next to its
    partner's object, and its own object and destination are in the other room: whoever
    wants to help can deliver its partner's object in two moves, and whoever wants to
    hinder can take it in one."""
    grid = np.full((9, 7), np.inf)
    grid[1:4, 1:6] = 1.0
    grid[5:8, 1:6] = 1.0
    grid[4, 2:4] = 1.0
    for cell in ((2, 3), (6, 3), (3, 5), (5, 1)):
        grid[cell] = np.inf
    return Scene(grid=grid, starts=((1, 2), (7, 4)), pickups=((7, 3), (1, 3)),
                 destinations=((7, 1), (1, 5)),
                 candidate_cells=((1, 5), (7, 1), (3, 4), (5, 2), (3, 1), (5, 5)),
                 objects=objects, name="two_rooms")


def six_object_scene() -> Scene:
    """:func:`two_room_scene` with four spare objects: six objects O1..O6 (O1, O2 are the
    robots' own) and six candidate destinations G0..G5."""
    scene = two_room_scene(objects=((2, 1), (2, 5), (6, 1), (6, 5)))
    return Scene(**{**scene.__dict__, "name": "six_objects"})


def exchange_scene(rooms: bool = True, handoff: bool = True) -> Scene:
    """A scene in which the two objects have to be exchanged: :func:`six_object_scene` with
    the doorway narrowed to the one cell (4, 2) and the two destinations swapped, so that
    each robot's destination is in its own room while its object lies in the other one,
    next to its partner.

    rooms : each robot stays in its own room; the doorway belongs to both.  A robot then
        cannot fetch its object itself, and nobody can carry an object to a destination in
        the other room: an object gets to its owner only by being put down in the doorway
        or handed over there.
    handoff : the doorway is a hand-off cell -- a robot standing in it may put its
        partner's object down, and the partner picks it up from there.

    With both (the default) and ``Config.set_down_for_partner=False`` a robot can do
    nothing for its partner but pass its object on."""
    base = six_object_scene()
    grid = base.grid.copy()
    grid[4, 3] = np.inf                                   # the doorway: (4, 2) only
    door = (4, 2)
    free = [(int(x), int(y)) for x, y in np.argwhere(np.isfinite(grid))]
    left = tuple(c for c in free if c[0] < 4) + (door,)
    right = tuple(c for c in free if c[0] > 4) + (door,)
    return Scene(grid=grid, starts=base.starts, pickups=base.pickups,
                 destinations=((1, 5), (7, 1)), candidate_cells=base.candidate_cells,
                 objects=base.objects, name="exchange",
                 handoff_cells=(door,) if handoff else (),
                 rooms=(left, right) if rooms else None)


# The gate scene (2026-11-02), as drawn: row 0 at the top.  1, 2 the starts; a, b robot 0's
# and robot 1's objects, each next to the partner; A, B their destinations, each at home;
# * the other candidate destinations; G the one-cell gate between the rooms; d the optional
# decoy, in the long way round along the bottom (12 moves from door to door, 2 through G).
GATE_MAP = (
    "###########",
    "#A*..#..*B#",
    "#1b..G..a2#",
    "#*...#...*#",
    "###.###.###",
    "###..d..###",
    "###########",
)


def gate_scene(handoff: str = "anywhere", decoy: bool = False) -> Scene:
    """Two 3 x 4 rooms joined by a one-cell gate and a long way round (:data:`GATE_MAP`),
    mirror-symmetric.  Each robot starts next to its partner's object; its own destination
    is at home, among two other candidates.  Doing its task alone takes a robot about 19
    transitions (cross, fetch, lift, come back, place), helped through the gate about 13;
    whoever stands in the gate, or next to it on either side, blocks the short way (2 moves)
    and sends the partner round (12).

    handoff : where an object may be put down (besides its owner's destination, and its own
        place by "put back").  "anywhere": every free cell an object may lie on -- all but
        the objects' places and the candidate destinations (24 of 32), so that a robot can
        carry its partner's object part of the way, or dump it out of reach.  "few": four
        chosen cells -- the gate (drop-and-go help and exchange), a far cell in each room (a
        place to dump it) and the middle of the long way round.  "gate": the gate alone, as
        in the exchange scene.  "none": no hand-off cell.
    decoy : a spare object in the long way round (a third candidate for the partner's
        object; with "anywhere" the model then grows past what the dynamics tables hold).

    The engine's collision rule breaks a tie at the gate for robot 0 (both stepping into it:
    robot 0 moves, robot 1 stays): read it off the mirrored pairings."""
    rows = GATE_MAP
    Y, X = len(rows), len(rows[0])
    at = {}                                   # symbol -> cells, in (ix, iy) with iy up
    grid = np.full((X, Y), np.inf)
    for r, line in enumerate(rows):
        for c, ch in enumerate(line):
            if ch != "#":
                cell = (c, Y - 1 - r)
                grid[cell] = 1.0
                at.setdefault(ch, []).append(cell)
    one = {ch: cells[0] for ch, cells in at.items() if len(cells) == 1}
    stars = sorted(at["*"], key=lambda c: (-c[1], c[0]))
    candidates = (one["B"], one["A"], *stars)            # G0 = robot 1's goal, G1 = robot 0's
    pickups = (one["a"], one["b"])
    objects = (one["d"],) if decoy else "goal"
    free = [(int(x), int(y)) for x, y in np.argwhere(np.isfinite(grid))]
    taken = set(candidates) | set(pickups) | ({one["d"]} if decoy else set())
    if handoff == "anywhere":
        hand = tuple(c for c in free if c not in taken)
    elif handoff == "few":
        far = (2, 3), (8, 3)                             # a far cell in each room
        hand = (one["G"], *far) + (() if decoy else (one["d"],))
    elif handoff == "gate":
        hand = (one["G"],)
    elif handoff == "none":
        hand = ()
    else:
        raise ValueError(f"handoff must be anywhere, few, gate or none, got {handoff!r}")
    name = "gate" + {"anywhere": "", "few": "_few", "gate": "_gateonly", "none": "_nohandoff"}[
        handoff] + ("_decoy" if decoy else "")
    return Scene(grid=grid, starts=(one["1"], one["2"]), pickups=pickups,
                 destinations=(one["A"], one["B"]), candidate_cells=candidates,
                 objects=objects, name=name, handoff_cells=hand)
