"""The shared-object pick-and-place world: objects, hands, rewards and dynamics tables.

Either robot can pick up its partner's object as well as its own (and a spare one with
``Config.handle_spares``, or wherever the partner's object is inferred).  Robot i's goal is
ITS object set down at ITS destination, by itself or, for it, by its partner: no other
object completes it, and its object can be set down nowhere else -- but on a hand-off cell
of the scene, where it lies until somebody picks it up.  What stays hidden from a robot is
its partner's destination, and -- ``Config.partner_object="inferred"``, the paper's -- which
object its partner wants: a physical goal is an object and a destination, and the tables
below are built for any object as a robot's goal (``cls_obj``, ``model_tab(..., obj)``).

Two things a scene may add (``Scene.handoff_cells``, ``Scene.rooms``): cells on which a
robot may put its partner's object down for the partner to pick up, and a room per robot
that it never leaves.  With both, an object that lies in the wrong room gets to its owner
only through the other robot's hands.

The rule for the hands lives in :mod:`social_pairings.hands`; this module turns it into the
tables the solver reads, and offers the rollout the real transition.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .config import Config, SolverFailure
from .constants import (ARRIVED, BLOCKED, CARRYING, DONE, GIVE, GONE, HELD0, HELD1, HOME,
                        HOW_HAND, MOVE_OF, N_ACTIONS, N_HOW, NO_WAY, OBJECT_STATUS, PICK, PLACE,
                        STAYED, SWAP, TAKE, TO_OBJECT, WAIT)
from .grid import GridWorld, as_cell
from .hands import build_table, configurations, encode, hands_step, lift_stage, lifter
from .solo import solo_scan, solo_values
from .util import check

__all__ = ["ObjectWorld", "object_cells"]


def object_cells(objects, pickups, destinations, candidate_cells=()) -> tuple:
    """The cells of a scene's objects, the two robots' own objects (the pickups) first.
    ``objects``: "goal" -- only those two; "candidates" -- also one on every candidate cell
    that is neither a pickup nor a destination; or the cells of the other objects."""
    pk = [as_cell(p) for p in pickups]
    ds = {as_cell(d) for d in destinations}
    if isinstance(objects, str):
        if objects == "goal":
            rest = []
        elif objects == "candidates":
            rest = [c for c in dict.fromkeys(as_cell(c) for c in candidate_cells)
                    if c not in pk and c not in ds]
        else:
            raise ValueError(f"objects must be 'goal', 'candidates' or a list of cells, "
                             f"got {objects!r}")
    else:
        rest = [c for c in dict.fromkeys(as_cell(c) for c in objects) if c not in pk]
    return tuple(pk + rest)


class ObjectWorld:
    """Two robots, M objects and K candidate destinations on a grid.

    grid : the :class:`GridWorld` of the scene (with the robots' rooms, if any).
    objects : the cells of the M objects, the two robots' own objects first (the pickups).
    carry : (2, M) cost factor of a move while robot i holds object q (default: 1 for its
        own object, ``Config.foreign_carry_cost`` for every other).
    handoff : the hand-off cells: neither an object's place nor a candidate destination.

    State ``(x0, x1, b)``: the robots' cell ids and the configuration ("block") of the
    objects; joint index ``s = (b * N + x0) * N + x1``, ``S = B * N^2``.  The configurations
    with both goals done are one terminal block, the last.

    Main tables
    -----------
    status (B, M), done (B, 2), hold (B, 2)   what a block means (hold: -1 = free hands)
    psucc (N^2, 100), pev (N^2, 100)          where a joint action takes the pair of robots,
                                              and the *event* it is for the hands
    cls_of (B, 2), rate_cls (ncls, N, K)      reward: r_i = rate_cls[cls_of[b, i], cell, goal]
    cls_obj (B, 2, M)                         the same for robot i with object q as its goal
                                              (cls_of[b, i] = cls_obj[b, i, i])
    cls_est (B, 2, M)                         ... as its partner estimates it in robot i's own
                                              model, where i's done flag is its own object
                                              delivered: for q != i the flag says nothing
    cost_c (B, 2, 10)                         action costs (a foreign object costs more)

    Rewards.  ``r_i = rho * max(1 - D_i * resolution / delta, 0)`` with D_i what is left
    of robot i's way: holding its object -- to its destination; its object at its place or
    on a hand-off cell -- to it and on to the destination; its object in its partner's
    hands -- nothing it can do, r_i = 0 (``Config.held_reward="carrier"``: the carrier's
    way instead); done -- 0.  Ways are counted by the robot's own moves: where it has a
    room, inside it, and r_i = 0 while its object lies where it cannot go.
    """

    def __init__(self, grid: GridWorld, cfg: Config, objects, carry=None, handoff=()) -> None:
        tic = time.perf_counter()
        held = cfg.held_reward
        self.grid, self.cfg = grid, cfg
        self.after_done = bool(cfg.done_robots_handle)
        # a robot does not know which object its partner wants (Config.partner_object)
        self.infer_object = cfg.partner_object == "inferred"
        N = self.N = grid.N
        NN = self.NN = grid.NN
        self.K = K = grid.K
        self.shape, self.cells, self.cell_id = grid.shape, grid.cells, grid.cell_id
        self.nb, self.valid, self.sep = grid.nb, grid.valid, grid.sep
        # each robot's own moves (the same as nb / valid unless the robots have rooms)
        self.nb_by, self.valid_by = grid.nb_by, grid.valid_by               # (2, N, 5)
        self.allowed, self.has_rooms, self.rooms = grid.allowed, grid.has_rooms, grid.rooms
        self.starts, self.goals, self.start, self.dest = (grid.starts, grid.goals, grid.start,
                                                          grid.dest)
        self.pickups = grid.pickups
        self.true_goals, self.true_dest = grid.true_goals, grid.true_dest
        self.pair_ok = grid.pair_ok
        # ---- objects -------------------------------------------------------------------------
        cells = tuple(as_cell(c) for c in objects)
        if len(cells) < 2 or cells[:2] != tuple(grid.pickups):
            raise ValueError("objects must start with the two robots' own objects (the pickups)")
        X, Y = grid.shape
        bad = [c for c in cells if not (0 <= c[0] < X and 0 <= c[1] < Y) or grid.cell_id[c] < 0]
        if bad:
            raise SolverFailure(f"object cells blocked, out of bounds or not connected to "
                                f"robot 0's start: {bad}")
        self.M = M = len(cells)
        if M > 6:
            raise SolverFailure(f"{M} objects: at most 6 (the configurations grow as 4^M)")
        self.objects = cells
        self.home = np.array([grid.cell_id[c] for c in cells], dtype=np.int64)       # (M,)
        if carry is None:
            carry = np.full((2, M), float(cfg.foreign_carry_cost))
            carry[0, 0] = carry[1, 1] = 1.0
        carry = np.asarray(carry, dtype=float)
        if carry.shape != (2, M) or not np.all(np.isfinite(carry)) or carry.min() < 1.0:
            raise ValueError(f"carry costs must be a (2, {M}) array of factors >= 1")
        self.carry = carry
        # ---- hand-off cells: where a robot may put its partner's object down ----------------
        hand = tuple(dict.fromkeys(as_cell(c) for c in handoff))
        bad = [c for c in hand if not (0 <= c[0] < X and 0 <= c[1] < Y) or grid.cell_id[c] < 0]
        if bad:
            raise SolverFailure(f"hand-off cells blocked, out of bounds or not connected to "
                                f"robot 0's start: {bad}")
        clash = [c for c in hand if c in cells or c in grid.goals]
        if clash:
            raise ValueError(f"a hand-off cell can be neither an object's place nor a candidate "
                             f"destination: {clash}")
        self.handoff = hand
        self.H = len(hand)
        self.hand = np.array([grid.cell_id[c] for c in hand], dtype=np.int64)       # (H,)
        # ---- the cells an object may lie on besides its own place: the hand-off cells, then
        # (Config.swap_on_place) the places of the objects, where a swap leaves one --------
        self.swap_on_place = bool(cfg.swap_on_place)
        self.steal = bool(cfg.steal)
        # which robots may take (Config.steal_robots, 2026-11-05): bit i, robot i
        self.steal_mask = int(sum(1 << i for i in (0, 1)
                                  if self.steal and cfg.steal_robots[i]))
        self.lie_cells = hand + (cells if self.swap_on_place else ())
        self.L = len(self.lie_cells)
        self.lie = np.array([grid.cell_id[c] for c in self.lie_cells], dtype=np.int64)   # (L,)
        # ---- the hands (Config.automatic_hands, pick_steps; hands.py) ------------------------
        # auto: a robot takes its own object on arriving where it lies and sets it down on
        # arriving at its destination (the rule until 2026-10-18); else by "pick" and "place"
        # only.  A pick-up takes lift_steps transitions, the robot busy lifting meanwhile:
        # statuses lift0 + i * nstage + (s - 1), robot i s steps into the lift (none with
        # lift_steps 1)
        self.auto = bool(cfg.automatic_hands)
        self.lift_steps = int(cfg.pick_steps)
        self.nstage = self.lift_steps - 1
        self.lift0 = 4 + self.L
        self.n_status = 4 + self.L + 2 * self.nstage   # status 4 + h: lying on lie_cells[h]
        # ---- special cells ("sites"): object places, candidate destinations, hand-off cells --
        special = sorted(set(int(c) for c in self.home) | set(int(c) for c in self.dest)
                         | set(int(c) for c in self.hand))
        self.nsite = len(special)
        self.site_of = np.zeros(N, dtype=np.int64)                       # 0 = no special cell
        self.site_of[special] = np.arange(1, self.nsite + 1)
        self.site_cell = np.array([-1] + special, dtype=np.int64)
        self.home_site = self.site_of[self.home]                         # (M,)
        self.dest_site = self.site_of[self.dest]                         # (K,)
        self.iscand = np.zeros(self.nsite + 1, dtype=np.bool_)   # where "place" sets down for
        if cfg.set_down_for_partner:                             # the partner
            self.iscand[self.dest_site] = True
        self.goal_at_site = np.full(self.nsite + 1, -1, dtype=np.int64)
        self.goal_at_site[self.dest_site] = np.arange(K)
        self.hand_site = self.site_of[self.hand]                         # (H,)
        self.hand_at = np.full(self.nsite + 1, -1, dtype=np.int64)       # site -> hand-off cell
        self.hand_at[self.hand_site] = np.arange(self.H)
        self.lie_site = self.site_of[self.lie]                           # (L,)
        self.place_at = np.full(self.nsite + 1, -1, dtype=np.int64)      # site -> cell of
        if self.swap_on_place:                                           # lie_cells that is an
            self.place_at[self.home_site] = self.H + np.arange(M)        # object's place
        self.E1 = E1 = N_HOW * (self.nsite + 1)
        self.nEv = 2 * E1 * E1
        # ---- moves of the pair, with the hand actions ----------------------------------------
        A = N_ACTIONS
        ps5 = grid.pair_moves                                                    # (NN, 25)
        aa = np.arange(A)
        idx = (MOVE_OF[aa][:, None] * 5 + MOVE_OF[aa][None, :]).reshape(-1)        # (A^2,)
        self.psucc = np.ascontiguousarray(ps5[:, idx])                           # (NN, A^2)
        self.psucc_T = np.ascontiguousarray(
            self.psucc.reshape(NN, A, A).transpose(0, 2, 1).reshape(NN, A * A))
        x0, x1 = np.divmod(np.arange(NN), N)                                     # (NN,)
        n0, n1 = np.divmod(self.psucc.astype(np.int64), N)                       # (NN, A^2)
        j = np.arange(A * A)
        a0, a1 = j // A, j % A

        def how(n, x, a):
            return np.where(n != x[:, None], ARRIVED,
                            np.where((a == WAIT)[None, :], STAYED,
                                     np.where((a >= PICK)[None, :], HOW_HAND + a - PICK,
                                              BLOCKED)))

        d = np.abs(self.cells[n0] - self.cells[n1]).sum(axis=-1)                 # (NN, A^2)
        e0 = self.site_of[n0] * N_HOW + how(n0, x0, a0)
        e1 = self.site_of[n1] * N_HOW + how(n1, x1, a1)
        ev = (e0 * E1 + e1) * 2 + (d == 1)
        # Event 0 (both ARRIVED on no special cell, not adjacent) stands for every event in
        # which the hands cannot act -- no special cell, nobody chose a hand action -- so the
        # kernels keep the block without a table look-up on the common path.
        quiet = ((self.site_of[n0] == 0) & (self.site_of[n1] == 0)
                 & (a0 < PICK)[None, :] & (a1 < PICK)[None, :])
        self.pev = np.ascontiguousarray(np.where(quiet, 0, ev), dtype=np.int32)
        self.pev_T = np.ascontiguousarray(
            self.pev.reshape(NN, A, A).transpose(0, 2, 1).reshape(NN, A * A))
        dn = np.abs(self.cells[x0] - self.cells[x1]).sum(axis=-1) == 1           # (NN,) now
        self.uctx = np.ascontiguousarray(
            np.stack([self.site_of[x0] * 2 + dn, self.site_of[x1] * 2 + dn]), dtype=np.int32)
        # ---- configurations ------------------------------------------------------------------
        self.configs = configurations(M, self.n_status, self.H if self.swap_on_place else -1,
                                      self.lift0, self.nstage)
        self.B = B = len(self.configs)
        self.T = B - 1                                                    # terminal block
        if B > np.iinfo(np.int16).max:           # the dynamics tables hold blocks as int16
            raise SolverFailure(f"{M} objects and {self.H} hand-off cells"
                                + (" with swaps on the places" if self.swap_on_place else "")
                                + f" -> {B:,} object configurations: more than the tables "
                                "hold; use fewer")
        self.status = np.array([c[:M] for c in self.configs], dtype=np.int64)     # (B, M)
        self.done = np.array([c[M:] for c in self.configs], dtype=np.int64)       # (B, 2)
        self.pw = self.n_status ** np.arange(M + 1, dtype=np.int64)
        self.code2blk = np.full(4 * self.n_status ** M, -1, dtype=np.int64)
        for b, c in enumerate(self.configs[:-1]):
            self.code2blk[int((np.array(c[:M]) * self.pw[:M]).sum()
                              + (c[M] + 2 * c[M + 1]) * self.pw[M])] = b
        self.code2blk[3 * self.pw[M]:] = self.T
        self.hold = np.full((B, 2), -1, dtype=np.int64)
        for i in (0, 1):
            bb, kk = np.nonzero(self.status == 1 + i)
            self.hold[bb, i] = kk
        self.hold[self.T] = -1
        # lift[b, i]: the object robot i is lifting (-1 none), stage[b, i]: how many steps
        # into the lift; busy[b, i]: it is -- "pick" is the one action open to it
        self.lift = np.full((B, 2), -1, dtype=np.int64)
        self.stage = np.zeros((B, 2), dtype=np.int64)
        for b in range(B - 1):
            for k in range(M):
                i = lifter(int(self.status[b, k]), self.lift0, self.nstage)
                if i >= 0:
                    self.lift[b, i] = k
                    self.stage[b, i] = lift_stage(int(self.status[b, k]), self.lift0,
                                                  self.nstage)
        self.busy = np.ascontiguousarray((self.lift >= 0).astype(np.uint8))            # (B, 2)
        # state_ok[b, pair]: the joint states that are solved -- pair_ok (the robots apart,
        # each in its room) and a lifting robot on a cell where its object can lie (its place,
        # a hand-off cell, with swaps another object's place): a lift happens where the object
        # lies, so the other cells of a lifting block are no state anything reaches
        self.state_ok = np.ascontiguousarray(np.tile(self.pair_ok, (B, 1)).astype(np.uint8))
        liftable = np.zeros((M, N), dtype=bool)                                        # (M, N)
        for k in range(M):
            liftable[k, self.home[k]] = True
            liftable[k, self.hand] = True
            if self.swap_on_place:
                liftable[k, self.home] = True
        x0, x1 = np.divmod(np.arange(NN), N)
        for b in range(B):
            for i, x in ((0, x0), (1, x1)):
                k = self.lift[b, i]
                if k >= 0:
                    self.state_ok[b] &= liftable[k, x]
        # gone[b, q]: object q set down for good in block b (what tells a set-down for the
        # partner from a hand-over in the kernels, kernels._counts)
        self.gone = np.ascontiguousarray((self.status == GONE).astype(np.uint8))      # (B, M)
        self.gone[self.T] = 1
        # lies_site[b, q]: the site object q lies on in block b (0: in somebody's hands, or
        # set down for good)
        self.lies_site = np.where(self.status == HOME, self.home_site[None, :], 0)   # (B, M)
        for h in range(self.L):
            self.lies_site[self.status == 4 + h] = self.lie_site[h]
        self.lies_site[self.T] = 0
        self.S = B * NN
        if self.S >= 2 ** 31:
            raise SolverFailure(f"{N} free cells x {B} object configurations -> {self.S:,} "
                                "joint states exceeds int32 indexing; use fewer objects")
        # rights[i]: bit q = robot i may hold object q -- the two goal objects, and the spare
        # ones with Config.handle_spares or where the partner's object is inferred (any
        # object may be the partner's).  Robot i's goal is "object i on my destination"
        # (whoever sets it down): only object i is handed to robot i or set down for it
        # (hands.hands_step).
        every = cfg.handle_spares or self.infer_object
        self.rights = np.full(2, (1 << M) - 1 if every else 3, dtype=np.int64)
        self.forced_swap = bool(cfg.forced_swap)
        self.hand_over = bool(cfg.hand_over)
        self.fit_hold = self.hold == np.arange(2)[None, :]      # (B, 2): holds its own object
        # ---- distances, reward classes, action costs -----------------------------------------
        self.d_obj = np.stack([grid.distances(int(h)) for h in self.home]).astype(np.int64)
        self.d_goal = grid.d_goal.astype(np.int64)                                   # (K, N)
        self.d_og = self.d_goal[:, self.home].T                                      # (M, K)

        def own_steps(src: int, robot: int) -> np.ndarray:
            """(N,) steps from cell ``src`` by ``robot``'s own moves; NO_WAY: none."""
            d = grid.distances(int(src), robot)
            return np.where(np.isfinite(d), d, NO_WAY).astype(np.int64)

        # the same by each robot's own moves (inside its room): what its rewards are made of
        self.d_goal_by = np.where(np.isfinite(grid.d_goal_by), grid.d_goal_by,
                                  NO_WAY).astype(np.int64)                           # (2, K, N)
        self.d_obj_by = np.stack([[own_steps(h, r) for h in self.home]
                                  for r in (0, 1)])                                  # (2, M, N)
        self.d_hand_by = np.zeros((2, self.H, N), dtype=np.int64)                    # (2, H, N)
        for r in (0, 1):
            for h in range(self.H):
                self.d_hand_by[r, h] = own_steps(self.hand[h], r)
        # ... to every cell an object may lie on besides its own place: (2, L, N)
        self.d_lie_by = (np.concatenate([self.d_hand_by, self.d_obj_by], axis=1)
                         if self.swap_on_place else self.d_hand_by)
        # Remaining distance of robot r's goal = dist_cls[cls_of[b, r], r's cell, goal].
        # Classes, one set per robot (its ways are its own): done; "hold" -- from the cell to
        # the destination; "fetch" object r where it lies, its place or a hand-off cell -- to
        # it and on to the destination; "none" -- no way, rate 0: in its partner's hands
        # object r is out of r's reach, nothing r can do.  (Robot r's hands may be full of
        # something else, which is in its way.)
        # copy_cls / copy_alt: the same for the copy of the partner's goal in robot r's
        # exchange reward -- robot r doing its partner's task, or (copy_alt) the partner
        # carrying the object, whichever is shorter.
        # held="carrier": while robot r's object is in its PARTNER's
        # hands the goal's reward follows the object -- class "held": the carrier's way to the
        # destination plus the one transition its set-down takes (so standing there with the
        # object is worth less than setting it down), read at the PARTNER's cell
        # (at_partner); held="none": nothing robot r can do, the rate is 0 (class "none").
        cls_key = {("done",): 0, ("none",): 1}
        self.zero_cls = 1
        self.held_reward = held
        busy_is_idle = cfg.hands_full_reward == "none"
        self.cls_of = np.zeros((B, 2), dtype=np.int64)
        self.at_partner = np.zeros((B, 2), dtype=np.uint8)
        # the same for robot r with object q as its goal object: (B, 2, M)
        self.cls_obj = np.zeros((B, 2, M), dtype=np.int64)
        self.at_partner_obj = np.zeros((B, 2, M), dtype=np.uint8)
        # the same as the partner's estimate of robot r's reward reads it in robot r's own
        # model (a level-2 robot's: its partner's level-1 models estimate its goal reward,
        # Table I's R_j): there r's done flag means r's OWN object delivered, which is no
        # goal of q != r done -- for those, what object q's whereabouts give
        self.cls_est = np.zeros((B, 2, M), dtype=np.int64)
        self.at_partner_est = np.zeros((B, 2, M), dtype=np.uint8)
        self.copy_cls = np.zeros((B, 2), dtype=np.int64)
        self.copy_alt = np.zeros((B, 2), dtype=np.bool_)

        def task(b: int, q: int, r: int) -> tuple:
            """Robot r bringing object q to a destination -> (class key, or the partner
            is carrying it -- holding it, or lifting it)."""
            st = int(self.status[b, q])
            if st == HOME:
                return ("fetch", r, q, -1), False
            if 4 <= st < self.lift0:             # lying on cell st - 4 of lie_cells
                return ("fetch", r, q, st - 4), False
            if st == 1 + r:
                return ("hold", r), False
            li = lifter(st, self.lift0, self.nstage)
            if li == r:
                return ("lift", r, lift_stage(st, self.lift0, self.nstage)), False
            if li == 1 - r:
                return ("none",), True
            return ("none",), st == 2 - r

        def carried_by_partner(b: int, q: int, r: int) -> bool:
            """Robot r's object q in its partner's hands, or being lifted by the partner."""
            st = int(self.status[b, q])
            return st == 2 - r or lifter(st, self.lift0, self.nstage) == 1 - r

        def held_key(b: int, q: int, r: int) -> tuple:
            """The "held" class of robot r's object q carried by its partner: the carrier's
            way (plus the lift still to go, if it is lifting)."""
            st = int(self.status[b, q])
            if lifter(st, self.lift0, self.nstage) == 1 - r:
                return ("heldlift", r, lift_stage(st, self.lift0, self.nstage))
            return ("held", r)

        self.hold_cls = [cls_key.setdefault(("hold", r), len(cls_key)) for r in (0, 1)]

        def full_of_another(b: int, r: int, q: int) -> bool:
            """Robot r's hands full of (or lifting) an object other than q."""
            return (self.hold[b, r] not in (-1, q)) or (self.lift[b, r] not in (-1, q))

        for b in range(B):
            for r in (0, 1):
                key = task(b, r, r)[0]
                if busy_is_idle and b != self.T and full_of_another(b, r, r):
                    key = ("none",)              # hands full of a foreign object: no way
                ckey, alt = task(b, 1 - r, r)
                self.copy_cls[b, r] = cls_key.setdefault(ckey, len(cls_key))
                self.copy_alt[b, r] = alt
                if not self.done[b, r]:
                    if (self.held_reward == "carrier" and b != self.T
                            and carried_by_partner(b, r, r)):
                        key = held_key(b, r, r)
                        self.at_partner[b, r] = 1
                    self.cls_of[b, r] = cls_key.setdefault(key, len(cls_key))
        # robot r with object q as its goal (a physical goal of another object: the
        # partner's object as a hypothesis, Config.partner_object="inferred")
        for b in range(B):
            for r in (0, 1):
                for q in range(M):
                    if q == r:
                        self.cls_obj[b, r, q] = self.cls_est[b, r, q] = self.cls_of[b, r]
                        self.at_partner_obj[b, r, q] = self.at_partner[b, r]
                        self.at_partner_est[b, r, q] = self.at_partner[b, r]
                        continue
                    key = task(b, q, r)[0]
                    if busy_is_idle and b != self.T and full_of_another(b, r, q):
                        key = ("none",)
                    carried = (self.held_reward == "carrier" and b != self.T
                               and carried_by_partner(b, q, r))
                    if carried:
                        key = held_key(b, q, r)
                    self.cls_est[b, r, q] = cls_key.setdefault(key, len(cls_key))
                    self.at_partner_est[b, r, q] = int(carried)
                    if self.done[b, r]:
                        continue                 # class 0: done
                    self.cls_obj[b, r, q] = self.cls_est[b, r, q]
                    self.at_partner_obj[b, r, q] = int(carried)
        self.ncls = len(cls_key)
        # the class table of a robot on its own (solo values): a partner that is not there
        # carries nothing anywhere
        self.cls_solo = np.where(self.at_partner != 0, self.zero_cls, self.cls_of)
        self.cls_solo_obj = np.where(self.at_partner_obj != 0, self.zero_cls, self.cls_obj)
        # What is left of a robot's way, in its own transitions: its moves -- and, without
        # the automatic hands (since 2026-10-18), the transitions its hands take as well: the
        # lift_steps of a pick-up still ahead and the one transition of the set-down (so that
        # standing on the destination with the object is worth less than setting it down,
        # and done is the one state worth rho; with the automatic hands those take no
        # transition of their own and D counts moves alone, as the paper's).  "held": the
        # carrier is the partner: its way plus the one transition its set-down takes (and
        # the lift still to go, "heldlift").
        place_steps = 0 if self.auto else 1
        lift_ahead = 0 if self.auto else self.lift_steps
        dist = np.full((self.ncls, N, K), NO_WAY, dtype=np.int64)
        for key, c in cls_key.items():
            if key[0] == "done":
                dist[c] = 0
            elif key[0] == "hold":
                dist[c] = np.minimum(self.d_goal_by[key[1]].T + place_steps, NO_WAY)
            elif key[0] == "lift":               # at the object's cell, s steps into the lift
                _, r, stage = key
                dist[c] = np.minimum(self.d_goal_by[r].T + place_steps
                                     + (self.lift_steps - stage), NO_WAY)
            elif key[0] == "held":               # the carrier is the partner: its way
                dist[c] = np.minimum(self.d_goal_by[1 - key[1]].T + 1, NO_WAY)
            elif key[0] == "heldlift":           # ... lifting it still
                _, r, stage = key
                dist[c] = np.minimum(self.d_goal_by[1 - r].T + 1 + (self.lift_steps - stage),
                                     NO_WAY)
            elif key[0] == "fetch":
                _, r, q, h = key
                lies = int(self.home[q]) if h < 0 else int(self.lie[h])
                to_it = self.d_obj_by[r, q] if h < 0 else self.d_lie_by[r, h]        # (N,)
                on = self.d_goal_by[r][:, lies]                                      # (K,)
                dist[c] = np.minimum(to_it[:, None] + on[None, :] + lift_ahead + place_steps,
                                     NO_WAY)
        self.dist_cls = dist                                                         # (ncls, N, K)
        self.rate_cls = np.ascontiguousarray(np.where(
            dist >= NO_WAY, 0.0,
            cfg.rho * np.maximum(1.0 - dist * cfg.resolution / cfg.reward_radius_m, 0.0)))
        # fin_cls[cls, cell, goal]: the goal is finished -- its done flag is set, or its robot
        # holds its own object on the destination (only that robot's own model
        # turns this into a set-down).  Not "the reward is rho": a robot standing on the
        # place of an object that lies on a destination has D = 0 as well.
        # (without the automatic hands only its done flag says so: it has to place the object)
        fin = np.zeros((self.ncls, N, K), dtype=np.uint8)
        for key, c in cls_key.items():
            if key[0] == "done":
                fin[c] = 1
            elif key[0] == "hold" and self.auto:
                fin[c] = self.d_goal_by[key[1]].T == 0
        self.fin_cls = np.ascontiguousarray(fin)
        # action costs: the moves and "stay" (Config.costs), pick (pick_cost_rate: on the
        # first transition of a lift, the rest free), place / take / give / swap
        # (use_cost_rate); a move with a foreign object in hand costs more (carry)
        base_cost = np.concatenate([cfg.costs, [cfg.pick_cost_rate * cfg.dt],           # (A,)
                                    [cfg.use_cost_rate * cfg.dt] * (N_ACTIONS - PICK - 1)])
        self.cost_c = np.tile(base_cost, (B, 2, 1))                                  # (B, 2, A)
        for b in range(B):
            for r in (0, 1):
                if self.hold[b, r] >= 0:
                    self.cost_c[b, r, :4] *= self.carry[r, self.hold[b, r]]
                if self.lift[b, r] >= 0:
                    self.cost_c[b, r, PICK] = 0.0
        self.vdone = float(cfg.rho - cfg.costs[WAIT]) / (1.0 - cfg.gamma)
        # ---- dynamics tables -----------------------------------------------------------------
        self._tabs: dict = {}
        own = np.array([0, 1], dtype=np.int64)
        both = np.array([True, True])
        gs = [0, 0]
        if self.true_dest is not None:
            gs = [int(self.site_of[d]) for d in self.true_dest]
        self.true_tab = self._table(tuple(gs), (gs[1], gs[0]), own, both)
        self.success_tab = self._table(tuple(gs), (-2, -2), own, both)
        st = np.zeros(M, dtype=np.int64)
        dn0 = np.zeros(2, dtype=np.int64)
        hands_step(st, dn0, self.site_of[self.start], np.zeros(2, dtype=np.int64), False,
                   self.home_site, np.array(gs, dtype=np.int64),
                   np.array([gs[1], gs[0]], dtype=np.int64), self.iscand, own, own, own,
                   self.rights, self.forced_swap, np.array([True, True]), self.after_done,
                   self.lie_site, self.hand_at, self.hand_over, self.swap_on_place,
                   self.steal_mask,
                   self.place_at, self.auto, self.lift0, self.nstage)
        self.block0 = int(self.code2blk[encode(st, dn0, self.pw)])
        self.initial = (int(self.start[0]), int(self.start[1]), self.block0)
        self._handok: dict = {}
        self._reach: dict = {}
        self._solo: dict = {}
        self.setup_seconds = time.perf_counter() - tic

    # ---- tables ------------------------------------------------------------------------------
    def _table(self, gsite, dsite, own, canuse, gobj=(0, 1), acc=(0, 1)) -> np.ndarray:
        """(B, nEv) int16: next block under the hands rule with these parameters
        (:func:`social_pairings.hands.hands_step`; gobj / acc: each robot's goal object and
        what it takes as its goal object -- (0, 1) in the world)."""
        key = (tuple(int(g) for g in gsite), tuple(int(d) for d in dsite),
               tuple(int(o) for o in own), tuple(bool(c) for c in canuse),
               tuple(int(g) for g in gobj), tuple(int(a) for a in acc))
        if key not in self._tabs:
            out = np.empty((self.B, self.nEv), dtype=np.int16)
            use = np.array(key[3])
            build_table(self.status, self.done, self.code2blk, self.pw, self.nsite,
                        self.home_site, np.array(key[0], dtype=np.int64),
                        np.array(key[1], dtype=np.int64), self.iscand,
                        np.array(key[2], dtype=np.int64), np.array(key[4], dtype=np.int64),
                        np.array(key[5], dtype=np.int64), self.rights, self.forced_swap,
                        use, self.after_done, self.lie_site, self.hand_at, self.hand_over,
                        self.swap_on_place, self.steal_mask, self.place_at, self.auto,
                        self.lift0,
                        self.nstage, out)
            check((out >= 0).all(), "a transition leaves the enumerated configurations")
            self._tabs[key] = out
        return self._tabs[key]

    def model_tab(self, o: int, goal: int, success: bool = False,
                  obj: int = None) -> np.ndarray:
        """Dynamics of a level >= 1 model of robot o with destination `goal` (and goal object
        `obj`, its own object unless given): both robots' hands work, only robot o sets down
        for its own goal (at `goal`); what robot o sets down for its partner fails
        (success=False) or succeeds wherever it tries (True) -- the solver mixes the two by
        the belief that the cell is the partner's destination.

        Where the partner's object is inferred (``Config.partner_object="inferred"``) robot
        o's model does not know which object its partner takes: in the table of failure the
        partner takes no object handed over or set down for it, in the table of success it
        takes any -- the solver mixes them by the belief that the object (and the cell) is
        the partner's; the partner neither takes an object on arriving nor picks one first
        (both depend on its object).  Robot o's own goal object is ``obj``."""
        q = o if obj is None else int(obj)
        g = [0, 0]
        g[o] = int(self.dest_site[goal])
        # without the automatic hands the partner finishes by placing what it holds on a
        # candidate cell: on any, in this table, which does not know its destination -- under
        # each hypothesis about it its policy places only on that hypothesis's destination
        # (with the automatic hands it parks there instead, holding the object:
        # SocialMDP._park_policy, and its reward there is the done reward)
        g[1 - o] = 0 if self.auto else -2
        d = [0, 0]
        d[o] = -2 if success else 0
        d[1 - o] = g[o]                              # the partner setting down for o's goal
        if not self.infer_object:
            return self._table(g, d, (0, 1), (True, True))
        own, acc = [-1, -1], [-1, -1]
        own[o] = acc[o] = q
        acc[1 - o] = -2 if success else -1
        return self._table(g, d, own, (True, True), own, acc)

    def level0_tab(self, o: int, goal: int, obj: int = None) -> np.ndarray:
        """Dynamics of robot o's level-0 model (goal object `obj`, its own unless given):
        its partner is a moving obstacle whose hands do nothing."""
        q = o if obj is None else int(obj)
        g = [0, 0]
        g[o] = int(self.dest_site[goal])
        own = [-1, -1]
        own[o] = q
        use = [False, False]
        use[o] = True
        if q == o:
            return self._table(g, (0, 0), own, use)
        gobj = [0, 1]
        gobj[o] = q
        return self._table(g, (0, 0), own, use, gobj, gobj)

    def level0_reward_tab(self, o: int, obj: int = None) -> np.ndarray:
        """Robot o's level-0 dynamics without its own set-down (goal object `obj`, its own
        unless given): the table its level-0 reward inputs read for every destination at
        once (an immediate reward is the same whether the object is held on the destination
        or set down there: remaining distance 0 either way)."""
        q = o if obj is None else int(obj)
        own = [-1, -1]
        own[o] = q
        use = [False, False]
        use[o] = True
        if q == o:
            return self._table((0, 0), (0, 0), own, use)
        gobj = [0, 1]
        gobj[o] = q
        return self._table((0, 0), (0, 0), own, use, gobj, gobj)

    def hand_ok(self, social: bool) -> np.ndarray:
        """(2, B, 2 (nsite + 1)) uint8: which hand actions robot i may choose in block c when
        it stands on site s with its partner adjacent (index s * 2 + adjacent) -- bit a - PICK
        for hand action a.  Only where it does something: pick (an object it may hold lies
        there), put back (place on the object's own place, nothing else lying there), swap
        and take (``Config.swap_on_place``, ``Config.steal``) for every robot; give, put down
        on a hand-off cell and set down for the partner (place) -- the partner's own object,
        or, where a robot does not know which object that is (``Config.partner_object=
        "inferred"``), any object it holds -- only for a robot with a social goal (they are
        acts for the partner).  A robot lifting an object (``Config.pick_steps`` > 1) has
        "pick" alone -- and its moves are closed to it too (``busy``, which the kernels
        read); nobody hands it anything meanwhile."""
        if social not in self._handok:
            out = np.zeros((2, self.B, 2 * (self.nsite + 1)), dtype=np.uint8)
            may = (self.rights[:, None] >> np.arange(self.M)[None, :]) & 1        # (2, M)
            bit = {a: 1 << (a - PICK) for a in (PICK, PLACE, TAKE, GIVE, SWAP)}
            for b in range(self.B - 1):
                for i in (0, 1):
                    if self.lift[b, i] >= 0:
                        out[i, b, :] = bit[PICK]                 # lifting: on with the lift
                        continue
                    if self.done[b, i] and not self.after_done:
                        continue
                    k = self.hold[b, i]
                    m = self.hold[b, 1 - i]
                    for s in range(self.nsite + 1):
                        lying = (self.lies_site[b] == s) & (may[i] != 0) if s > 0 else \
                            np.zeros(self.M, dtype=bool)
                        for adj in (0, 1):
                            ok = 0
                            if k >= 0:
                                theirs = k == 1 - i or self.infer_object
                                others = lying.copy()
                                others[k] = False
                                if s > 0 and self.place_at[s] >= 0 and others.any():
                                    ok |= bit[SWAP]
                                if not self.auto and s > 0 and self.iscand[s]:
                                    ok |= bit[PLACE]     # set down for its own goal -- on
                                    # its destination (which candidate that is, each model
                                    # knows; elsewhere the action does nothing)
                                if s > 0 and self.home_site[k] == s:
                                    if not (self.lies_site[b] == s).any():
                                        ok |= bit[PLACE]             # put back
                                elif social and s > 0 and self.hand_at[s] >= 0 and theirs:
                                    ok |= bit[PLACE]                 # put down (hand-off)
                                elif (social and s > 0 and self.iscand[s]
                                      and not self.done[b, 1 - i] and theirs):
                                    ok |= bit[PLACE]                 # set down for the partner
                                if (social and self.hand_over and adj and theirs
                                        and self.lift[b, 1 - i] < 0
                                        and (self.after_done or not self.done[b, 1 - i])):
                                    ok |= bit[GIVE]
                            else:
                                if lying.any():
                                    ok |= bit[PICK]
                                if ((self.steal_mask >> i) & 1 and adj and m >= 0
                                        and may[i, m]):
                                    ok |= bit[TAKE]  # what the partner next to it holds
                            out[i, b, s * 2 + adj] = ok
            self._handok[social] = out
        return self._handok[social]

    # ---- states ------------------------------------------------------------------------------
    def sid(self, state) -> int:
        x0, x1, b = state
        return (int(b) * self.N + int(x0)) * self.N + int(x1)

    def unsid(self, s: int):
        b, pair = divmod(int(s), self.NN)
        x0, x1 = divmod(pair, self.N)
        return (x0, x1, b)

    def cell(self, x: int) -> tuple:
        return int(self.cells[x, 0]), int(self.cells[x, 1])

    def conflict(self, p, q):
        return self.grid.conflict(p, q)

    def step(self, s: int, act) -> int:
        """True joint transition (both robots set down at their true destinations; a
        set-down for the partner counts where the partner's destination is)."""
        b, pair = divmod(int(s), self.NN)
        j = act[0] * N_ACTIONS + act[1]
        return int(self.true_tab[b, self.pev[pair, j]]) * self.NN + int(self.psucc[pair, j])

    def dist_left(self, state, agent: int, goal: int) -> int:
        """Remaining distance of robot `agent`'s goal (cells; NO_WAY = its object is out of
        its reach: in its partner's hands, or -- Config.hands_full_reward="none" -- its own
        hands are full of another)."""
        return int(self.dist_cls[self.cls_of[state[2], agent], self._at(state, agent), goal])

    def _at(self, state, agent: int) -> int:
        """The cell robot `agent`'s reward is read at: its own, or its partner's while the
        partner carries its object (held="carrier")."""
        return int(state[1 - agent] if self.at_partner[state[2], agent] else state[agent])

    def rate(self, state, agent: int, goal: int) -> float:
        return float(self.rate_cls[self.cls_of[state[2], agent], self._at(state, agent), goal])

    def rates(self, state, agent: int) -> np.ndarray:
        """(K,) robot `agent`'s reward rate at `state` for every candidate destination."""
        return self.rate_cls[self.cls_of[state[2], agent], self._at(state, agent), :]

    def goal_rates(self, state, agent: int, objs, goals, estimate: bool = False) -> np.ndarray:
        """Robot `agent`'s reward rate at `state` for each physical goal (objs[n] to
        candidate goals[n]).  estimate: as its partner estimates it in the robot's own model
        (``cls_est``)."""
        b = state[2]
        objs, goals = np.asarray(objs, dtype=np.int64), np.asarray(goals, dtype=np.int64)
        cls, at_partner = ((self.cls_est, self.at_partner_est) if estimate
                           else (self.cls_obj, self.at_partner_obj))
        at = np.where(at_partner[b, agent, objs] != 0, state[1 - agent], state[agent])
        return self.rate_cls[cls[b, agent, objs], at, goals]

    def goals_finished(self, state, agent: int, objs, goals,
                       estimate: bool = False) -> np.ndarray:
        """Is robot `agent`'s goal finished at `state`, for each physical goal (objs[n] to
        candidate goals[n]): done, or the robot holding that object on that destination.
        estimate: as its partner estimates it in the robot's own model (``cls_est``)."""
        b = state[2]
        objs, goals = np.asarray(objs, dtype=np.int64), np.asarray(goals, dtype=np.int64)
        cls, at_partner = ((self.cls_est, self.at_partner_est) if estimate
                           else (self.cls_obj, self.at_partner_obj))
        at = np.where(at_partner[b, agent, objs] != 0, state[1 - agent], state[agent])
        return self.fin_cls[cls[b, agent, objs], at, goals] != 0

    def finished(self, state, agent: int) -> np.ndarray:
        """(K,) bool: is robot `agent`'s goal finished at `state`, for every candidate
        destination -- done, or the robot holding its own object there."""
        return self.fin_cls[self.cls_of[state[2], agent], self._at(state, agent), :] != 0

    def copy_rate(self, state, agent: int) -> np.ndarray:
        """(K,) the copy of the partner's goal in robot `agent`'s exchange reward, for every
        candidate destination k of the partner: rho once the partner's goal is done, else
        r(D) with D what is left of the partner's task: the way of the partner's object --
        the robot fetching it and bringing it to k, the robot carrying it there, or the
        partner carrying it there."""
        b, p = state[2], 1 - agent
        if self.done[b, p]:
            return np.full(self.K, float(self.cfg.rho))
        out = self.rate_cls[self.copy_cls[b, agent], state[agent], :]
        if self.copy_alt[b, agent]:
            out = np.maximum(out, self.rate_cls[self.hold_cls[p], state[p], :])
        return out

    def phase(self, state, agent: int) -> int:
        """The old three-valued phase of a robot's task: 0 nothing in hand, 1 holding an
        object, 2 goal done."""
        b = state[2]
        return DONE if self.done[b, agent] else (CARRYING if self.hold[b, agent] >= 0
                                                 else TO_OBJECT)

    def satisfied(self, state, agent: int) -> bool:
        return bool(self.done[state[2], agent])

    def real_step(self, status, done, state, nxt, act, detail: bool = False):
        """The hands rule applied to the real objects for one transition: -> (statuses,
        done flags, failed) with failed[i] = robot i tried to set an object down for its
        partner at a cell that is not the partner's destination (so nothing happened).
        detail: -> (statuses, done flags, failed, failed hand-overs, took), with failed
        hand-over[i] = robot i tried to hand its partner an object that is not the
        partner's (only a robot that does not know which object that is tries:
        Config.partner_object), and took[i] = robot i took what its partner held
        (Config.steal)."""
        cells = (int(nxt[0]), int(nxt[1]))
        site = self.site_of[list(cells)]
        how = np.array([ARRIVED if nxt[i] != state[i] else STAYED if act[i] == WAIT
                        else HOW_HAND + act[i] - PICK if act[i] >= PICK else BLOCKED
                        for i in (0, 1)], dtype=np.int64)
        adj = bool(np.abs(self.cells[cells[0]] - self.cells[cells[1]]).sum() == 1)
        gs = np.array([self.site_of[d] for d in self.true_dest], dtype=np.int64)
        own = np.array([0, 1], dtype=np.int64)
        use = np.array([True, True])

        def run(dsite, acc=(0, 1), steal=self.steal_mask):
            st = np.array(status, dtype=np.int64)
            dn = np.array(done, dtype=np.int64)
            hands_step(st, dn, site, how, adj, self.home_site, gs,
                       np.array(dsite, dtype=np.int64), self.iscand, own, own,
                       np.array(acc, dtype=np.int64), self.rights, self.forced_swap, use,
                       self.after_done, self.lie_site, self.hand_at, self.hand_over,
                       self.swap_on_place, steal, self.place_at, self.auto, self.lift0,
                       self.nstage)
            return tuple(int(v) for v in st), tuple(int(v) for v in dn)

        real = run((gs[1], gs[0]))
        failed, handed = [], []
        for i in (0, 1):
            # what robot i did for its partner, had it counted wherever it was done and
            # whatever the object: a set-down that changes the partner's done flag, or a
            # hand-over (a robot that knows its partner's object offers no other)
            anywhere = (-2 if i == 0 else gs[1], -2 if i == 1 else gs[0])
            taken = run(anywhere, tuple(-2 if r == 1 - i else r for r in (0, 1)))
            failed.append(taken != real and taken[1] != real[1])
            handed.append(taken != real and taken[1] == real[1])
        if detail:
            took = [False, False]
            if self.steal_mask:                  # what the hands would have held without it
                kept = run((gs[1], gs[0]), steal=0)[0]
                for k, (a, b) in enumerate(zip(kept, real[0])):
                    if a in (HELD0, HELD1) and b in (HELD0, HELD1) and a != b:
                        took[b - 1] = True
            return real[0], real[1], tuple(failed), tuple(handed), tuple(took)
        return real[0], real[1], tuple(failed)

    def block_of(self, status, done) -> int:
        return int(self.code2blk[encode(np.array(status, dtype=np.int64),
                                        np.array(done, dtype=np.int64), self.pw)])

    def object_cells(self, status, state, where=None) -> tuple:
        """Cell (ix, iy) of every object: its place, its carrier's cell, the hand-off cell
        (or, after a swap, the other object's place) it lies on, or -- once it has been set
        down -- where[q], the cell it was set down at (None if not given)."""
        out = []
        for q, st in enumerate(status):
            li = lifter(int(st), self.lift0, self.nstage)
            out.append(self.objects[q] if st == HOME else self.cell(state[0]) if st == HELD0
                       else self.cell(state[1]) if st == HELD1
                       else self.cell(state[li]) if li >= 0           # under its lifter
                       else self.lie_cells[st - 4] if st >= 4
                       else None if where is None else where[q])
        return tuple(out)

    def lifter_of(self, st: int) -> int:
        """The robot lifting an object with status ``st`` (-1: none)."""
        return int(lifter(int(st), self.lift0, self.nstage))

    def status_name(self, st: int) -> str:
        """One object status in words."""
        li = lifter(int(st), self.lift0, self.nstage)
        if li >= 0:
            return (f"being lifted by R{li}, step {lift_stage(int(st), self.lift0, self.nstage)}"
                    f" of {self.lift_steps}")
        if st >= 4 + self.H:
            return f"on object {st - 4 - self.H}'s place"
        if st >= 4:
            return f"on hand-off cell {st - 4}"
        return OBJECT_STATUS[st]

    def target(self, state, agent: int, goal: int):
        """(cell id, distance field (N,)) robot `agent` is physically heading for: its
        destination while it holds its own object, else the cell that object lies on (None
        while it lies nowhere: its partner holds it).  Distances by the robot's own moves
        (NO_WAY: none)."""
        b = state[2]
        if self.done[b, agent]:
            return None
        if self.fit_hold[b, agent]:
            return int(self.dest[goal]), self.d_goal_by[agent, goal]
        st = int(self.status[b, agent])
        if st == HOME:
            return int(self.home[agent]), self.d_obj_by[agent, agent]
        if st >= 4:
            return int(self.lie[st - 4]), self.d_lie_by[agent, st - 4]
        return None

    # ---- reachable blocks --------------------------------------------------------------------
    def _events(self):
        if not hasattr(self, "_ev"):
            E1 = self.E1
            ev = np.arange(self.nEv)
            e0, e1 = np.divmod(ev // 2, E1)
            self._ev = (e0 // N_HOW, e0 % N_HOW, e1 // N_HOW, e1 % N_HOW, ev % 2)
        return self._ev

    def reach(self, tabs, block: int, social, stop=None) -> np.ndarray:
        """Blocks a model can get to from `block`: tabs = its dynamics tables (the fail and
        the success table of a social model), social = (bool, bool) which robot may hand
        over / set down for the other; stop (B,) bool: blocks that are not expanded (and
        not returned).  Cached by the CONTENT of the tables: until 2026-10-07 the key held
        ``id(t)``, and a caller passing temporary views (``SocialMDP.reach_of``) could be
        handed the blocks of another table whose view had lived at the same address (the
        README, "Limits")."""
        key = (tuple(hashlib.blake2b(np.ascontiguousarray(t).tobytes(), digest_size=16).digest()
                     for t in tabs), int(block), tuple(bool(v) for v in social),
               None if stop is None else stop.tobytes())
        if key not in self._reach:
            s0, h0, s1, h1, adj = self._events()
            ok = ~((s0 == s1) & (s0 > 0))
            u0, u1 = self.hand_ok(social[0])[0], self.hand_ok(social[1])[1]
            hand0, hand1 = h0 >= HOW_HAND, h1 >= HOW_HAND
            bit0 = np.where(hand0, 1 << np.maximum(h0 - HOW_HAND, 0), 0)
            bit1 = np.where(hand1, 1 << np.maximum(h1 - HOW_HAND, 0), 0)
            seen = np.zeros(self.B, dtype=bool)
            todo = [int(block)]
            seen[block] = True
            while todo:
                b = todo.pop()
                if stop is not None and stop[b]:
                    continue
                allowed = (ok & (~hand0 | ((u0[b][s0 * 2 + adj] & bit0) != 0))
                           & (~hand1 | ((u1[b][s1 * 2 + adj] & bit1) != 0)))
                for t in tabs:
                    for nb_ in np.unique(t[b][allowed]):
                        if not seen[nb_]:
                            seen[nb_] = True
                            todo.append(int(nb_))
            if stop is not None:
                seen &= ~stop
            self._reach[key] = np.flatnonzero(seen)
        return self._reach[key]

    # ---- exact solo values -------------------------------------------------------------------
    def solo(self, r: int, goal: int, obj: int = None) -> dict:
        """Robot r alone with destination `goal` (and goal object `obj`, its own unless
        given): {"V" (B, N) exact value, "steps" from the start (-1: its optimum parks
        instead of finishing), "td" (lo, hi)}."""
        q = r if obj is None else int(obj)
        key = (int(r), int(goal)) if q == r else (int(r), int(goal), q)
        if key not in self._solo:
            cfg = self.cfg
            V = np.empty((self.B, self.N))
            tab = self.level0_tab(r, goal, q)
            args = (r, self.N, np.ascontiguousarray(self.nb_by[r]),
                    np.ascontiguousarray(self.valid_by[r]),
                    np.ascontiguousarray(self.busy[:, r].astype(np.bool_)),
                    self.site_of, tab, self.E1, np.ascontiguousarray(self.cls_solo_obj[:, r, q]),
                    np.ascontiguousarray(self.rate_cls[:, :, goal]),
                    np.ascontiguousarray(self.cost_c[:, r, :]),
                    np.ascontiguousarray(self.hand_ok(False)[r]), cfg.gamma, self.vdone,
                    np.ascontiguousarray(self.done[:, r].astype(np.bool_)), V)
            sweeps = solo_values(*args, 8 * self.N + 64)
            td = np.zeros(2)
            steps = solo_scan(*args, self.block0, int(self.start[r]), 8 * self.N + 64, td)
            self._solo[key] = dict(V=V, steps=int(steps), td=(float(td[0]), float(td[1])),
                                   sweeps=int(sweeps))
        return self._solo[key]

    def solo_value(self, state, agent: int, goal: int) -> float:
        b = state[2]
        return self.vdone if self.done[b, agent] else float(
            self.solo(agent, goal)["V"][b, state[agent]])

    def solo_state_values(self, agent: int, goal: int, obj: int = None) -> np.ndarray:
        """(S,) solo value of `agent` for `goal` (goal object `obj`, its own unless given) at
        every joint state (partner ignored)."""
        v = self.solo(agent, goal, obj)["V"]                              # (B, N)
        shape = (self.B, self.N, self.N)
        out = np.broadcast_to(v[:, :, None] if agent == 0 else v[:, None, :], shape)
        return np.ascontiguousarray(out).reshape(-1)

    def td_bounds(self, agent: int, goal: int) -> tuple:
        return self.solo(agent, goal)["td"]

    def can_finish_alone(self, agent: int, goal: int) -> bool:
        """Can robot ``agent`` reach its object's place, and from there destination ``goal``,
        by its own moves?  Not if one of the two lies outside its room."""
        return bool(self.d_obj_by[agent, agent, self.start[agent]] < NO_WAY
                    and self.d_goal_by[agent, goal, self.home[agent]] < NO_WAY)

    def solo_preflight(self, goal_ids) -> tuple:
        """Each robot's transitions to finish alone -- None for a robot that cannot (its
        object or its destination lies outside its room)."""
        steps = []
        for a in (0, 1):
            if not self.can_finish_alone(a, goal_ids[a]):
                steps.append(None)
                continue
            n = self.solo(a, goal_ids[a])["steps"]
            if n < 0:
                raise SolverFailure(
                    f"robot {a}: its solo optimum parks instead of finishing (bringing its "
                    "object to its destination is not worth the way). Recalibrate the "
                    "reward/cost model rather than forcing motion.")
            steps.append(n)
        return tuple(steps)
