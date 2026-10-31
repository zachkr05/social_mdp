"""Scenes: a grid, two robots, their objects and the candidate destinations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["Scene", "six_object_scene", "two_room_scene", "exchange_scene"]

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
