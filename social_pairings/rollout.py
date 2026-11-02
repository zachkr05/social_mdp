"""Rolling a pair of robots out: Algorithm 1, one transition at a time.

Each transition: solve every model above level 0 for the current beliefs, let both robots
choose, apply the real dynamics, record what happened, and update the beliefs (Eqs. 5-6).
The record -- a ledger per robot, a trace row per transition and the list of object events
-- is what :mod:`social_pairings.hindrance` and :mod:`social_pairings.render` read.
"""
from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from . import goals as G
from .config import Config, SolverFailure
from .constants import (ACTION_NAMES, CARRYING, DONE, GONE, HELD0, HELD1, HOME, MOVE_OF, MOVES,
                        N_ACTIONS, NO_WAY, PICK, STAGE_NAMES, STALL_WARNING, TO_OBJECT,
                        WAIT)
from .grid import GridWorld, as_cell
from .models import SocialMDP
from .util import (argmax_action, check, choose_with_inertia, configure_threads, log_softmax,
                   sample_action, top_hypotheses, undo_of)
from .world import ObjectWorld, object_cells

__all__ = ["build_world", "rollout", "holds_of"]


def holds_of(status) -> tuple:
    """(object in R0's hands, object in R1's hands), -1 = free hands."""
    return tuple(next((k for k, v in enumerate(status) if v == 1 + i), -1) for i in (0, 1))


def _object_name(q: int, by: Optional[int] = None) -> str:
    if q < 2:
        return "its object" if by == q else f"R{q}'s object"
    return f"object {q}"


def build_world(grid, starts, pickups, destinations, candidate_cells, objects, cfg: Config,
                carry_costs=None, handoff_cells=(), rooms=None):
    """Validate a scene and build its world.

    objects: "goal" (the two own objects), "candidates" (also one on every candidate cell
    that is neither a pickup nor a destination) or the cells of the other objects.
    handoff_cells: cells on which a robot may put its partner's object down for the partner
    to pick up.  rooms: None, or for each robot the cells it may stand on (``Scene``).

    Objects nobody may handle (the spare ones, unless ``handle_spares=True`` -- or the
    partner's object is inferred, ``partner_object="inferred"``, when any object may be the
    partner's) are *scenery*: they are listed in every record and drawn, but
    left out of the model, whose object configurations grow as 4^M.  Of the others at most
    ``Config.max_objects`` are kept -- the two own ones, then those nearest somebody's task
    -- and the rest are scenery too.

    -> (world, goal_ids, solo_steps, scenery): the candidate index of each robot's
    destination, each robot's transitions to finish alone (None: it cannot, its object or
    its destination lies outside its room), and the scenery cells."""
    if len(destinations) != 2:
        raise ValueError("need exactly two destinations")
    candidates = tuple(dict.fromkeys(as_cell(g) for g in (*candidate_cells, *destinations)))
    goal_ids = tuple(candidates.index(as_cell(g)) for g in destinations)
    base = GridWorld(grid, starts, pickups, candidates, cfg, goal_ids, rooms=rooms)
    cells = object_cells(objects, pickups, destinations, candidate_cells)
    clash = [c for c in (as_cell(h) for h in handoff_cells) if c in cells]
    if clash:                                # scenery included: nothing lies where one starts
        raise ValueError(f"a hand-off cell can be neither an object's place nor a candidate "
                         f"destination: {clash}")
    X, Y = base.shape
    bad = [c for c in cells[2:]
           if not (0 <= c[0] < X and 0 <= c[1] < Y) or base.cell_id[c] < 0]
    if bad:
        raise SolverFailure(f"object cells blocked, out of bounds or not connected: {bad}")
    scenery = ()
    if not (cfg.handle_spares or cfg.partner_object == "inferred"):
        cells, scenery = cells[:2], cells[2:]
    limit = cfg.max_objects
    if len(cells) > limit:                   # keep the others most useful for somebody's task

        def use(c) -> float:
            d = base.distances(int(base.cell_id[c]))
            return min(d[base.start[i]] + base.d_goal[goal_ids[i]][base.cell_id[c]]
                       for i in (0, 1))

        rest = sorted(cells[2:], key=lambda c: (use(c), cells.index(c)))
        scenery = tuple(c for c in cells[2:] if c not in rest[:limit - 2]) + scenery
        cells = cells[:2] + tuple(c for c in cells[2:] if c in rest[:limit - 2])
    world = ObjectWorld(base, cfg, cells, carry=carry_costs, handoff=handoff_cells)
    solo_steps = world.solo_preflight(goal_ids)
    return world, goal_ids, solo_steps, scenery


class _Run:
    """The state of one rollout: where everything is, and everything recorded so far."""

    def __init__(self, world: ObjectWorld, solo_steps, scenery=(), own_goal=(True, True)
                 ) -> None:
        w = world
        self.scenery = tuple(scenery)                        # objects outside the model
        # a robot without a goal of its own (a hinderer under Config.hinder_own_goal=False)
        # is never done: its object reaching its destination is an event, not its goal
        self.own_goal = tuple(bool(x) for x in own_goal)
        # what would undo each robot's last action (Config.inertia); nothing before the first
        self.undo: tuple = (None, None)
        self.state = w.initial                               # (x0, x1, block)
        self.status = [int(v) for v in w.status[w.block0]]   # real object statuses
        self.done = [int(v) for v in w.done[w.block0]]
        self.put_back = [None] * w.M                # robot that last put each object back
        self.put_down = [set() for _ in range(w.M)]  # (robot, cell): put down there before
        self.where = [None] * w.M                   # cell each set-down object lies at
        for i in (0, 1):                            # (a robot starting on its destination)
            if self.done[i] and self.status[i] == GONE:
                self.where[i] = w.cell(w.initial[i])
        self.ledger = [_new_ledger(), _new_ledger()]
        for i in (0, 1):
            if w.hold[w.block0, i] >= 0 or w.done[w.block0, i]:
                self.ledger[i]["picked_at"] = 0
            if w.done[w.block0, i] and self.own_goal[i]:
                self.ledger[i].update(done_at=0, object=i, delivered_by=i)
        self.solo_steps = tuple(solo_steps)
        self.trace: list = []
        self.events: list = []                      # (transition, what, robot, object, other)
        self.lines: list = []
        self.top_goal = [None, None]
        self.blocked_run = [0, 0]
        self.stall = 0
        self.max_stall = 0
        self.deadlock: Optional[dict] = None
        self.best_left = [np.inf, np.inf]
        self.since_best = 0
        self.tail = 0
        self.solves = 0
        self.pi_total = 0.0
        self.krylov_total = 0.0
        self.pi_max = 0
        self.krylov_max = 0
        self.residual_max = 0.0
        self.unconverged = 0
        self.t_maps = 0.0                        # seconds spent on the value maps

    def all_status(self, status) -> tuple:
        """The statuses of every object of the scene: the scenery never leaves its place."""
        return tuple(status) + (HOME,) * len(self.scenery)

    def all_cells(self, cells) -> tuple:
        return tuple(cells) + self.scenery


def _new_ledger() -> dict:
    return dict(steps=0, moves=0, waits=0, hand_actions=0,
                by_hand_action={name: 0 for name in ACTION_NAMES[PICK:]},
                blocked_by_partner=0, step_cost=0.0,
                physical_reward=0.0, social_reward=0.0, discounted_return=0.0,
                picked_at=None, done_at=None, object=None, delivered_by=None,
                pickups=0, lifts=0, put_backs=0, put_downs=0, handed_over=0, received=0,
                dropped=0, took=0, taken_from=0,
                set_down_for_partner=0, failed_set_downs=0, failed_hand_overs=0,
                steps_holding_foreign=0, social_flips=0, inertia=0,
                stage_steps={name: 0 for name in STAGE_NAMES})


def _finished(mdp, state, i: int) -> bool:
    """Has robot i nothing left to do for a goal of its own: its goal is done, or it has
    none (a hinderer under ``Config.hinder_own_goal=False``)?"""
    return bool(mdp.spoiler[i]) or mdp.w.satisfied(state, i)


def _stage(w: ObjectWorld, state, i: int) -> int:
    """Done, carrying (an object in hand), else on its way to the object (lifting one
    included)."""
    b = state[2]
    return DONE if w.done[b, i] else (CARRYING if w.hold[b, i] >= 0 else TO_OBJECT)


def _left(w: ObjectWorld, state, i: int, goal_ids) -> float:
    """Robot i's way to go, for the stall tests: its own, or -- when its partner carries
    robot i's object -- the carrier's way to robot i's destination (by the carrier's own
    moves), or to a hand-off cell, the put-down there, and robot i's way to that cell and
    on to its destination, whichever is shortest."""
    if w.satisfied(state, i):
        return 0
    d = w.dist_left(state, i, goal_ids[i])
    k = int(w.hold[state[2], 1 - i]) if state[2] != w.T else -1
    if k == i:
        carrier, goal = state[1 - i], goal_ids[i]
        d = min(d, int(w.d_goal_by[1 - i, goal][carrier]))
        for h in range(w.H):
            d = min(d, int(w.d_hand_by[1 - i, h][carrier]) + 1 + int(w.d_hand_by[i, h][state[i]])
                    + int(w.d_goal_by[i, goal][w.hand[h]]))
    return d


def _events_of(w: ObjectWorld, run: _Run, old_status, new_status, nxt, done_now, failed,
               block: int, handed=(False, False), took=(False, False)) -> list:
    """What happened to the objects in one transition: (what, robot, object, other robot).
    "put down": on a hand-off cell, for the object's owner (the other robot) to pick up.
    "dropped": left on another object's place in a swap (``Config.swap_on_place``; the
    robot picks up what lay there in the same transition).  "took": taken from the other
    robot's hands (``Config.steal``).  "set down in vain" / "handed over in vain": an act
    for the partner that did not count -- not the partner's destination, or (a robot that
    does not know which object its partner wants) not the partner's object."""
    happened = []
    for k in range(w.M):
        a, b = old_status[k], new_status[k]
        if a == b:
            continue
        lifted_by, lifting_by = w.lifter_of(a), w.lifter_of(b)
        if lifting_by >= 0:              # a lift begun (Config.pick_steps > 1); its later
            if lifted_by < 0:            # steps change the status without an event
                happened.append(("lifting", lifting_by, k, None))
            continue
        if lifted_by >= 0 and b not in (HELD0, HELD1):   # a lift given up (an inert partner
            happened.append(("put back", lifted_by, k, None))   # in a model; never in fact)
            continue
        if b == HOME:
            happened.append(("put back", a - 1, k, None))
        elif b >= 4 + w.H:               # onto another object's place, in a swap
            happened.append(("dropped", a - 1, k, None))
        elif b >= 4:                     # onto a hand-off cell: by the robot that held it
            happened.append(("put down", a - 1, k, k if k < 2 and k != a - 1 else None))
        elif b in (HELD0, HELD1):
            if a == HOME or a >= 4:      # from where it lay, or at the end of its lift
                happened.append(("picked up", b - 1, k, None))
            elif took[b - 1]:
                happened.append(("took", b - 1, k, a - 1))
            else:
                happened.append(("handed over", a - 1, k, b - 1))
        elif a == HOME:                  # taken and set down in one transition: the robot
            by = next(i for i in (0, 1)  # standing there, on its own destination
                      if done_now[i] and nxt[i] == w.home[k] == w.true_dest[i])
            happened += [("picked up", by, k, None), ("set down", by, k, by)]
        else:
            by = a - 1                   # the robot that held it
            if done_now[by] and nxt[by] == w.true_dest[by] and k == by:
                happened.append(("set down", by, k, by))             # for its own goal
            elif nxt[by] == w.true_dest[1 - by]:
                happened.append(("set down", by, k, 1 - by))         # for its partner's goal
            else:                        # handed to a partner standing on its destination
                happened += [("handed over", by, k, 1 - by), ("set down", 1 - by, k, 1 - by)]
    for i in (0, 1):
        if failed[i]:
            happened.append(("set down in vain", i, int(w.hold[block, i]), 1 - i))
        if handed[i]:
            happened.append(("handed over in vain", i, int(w.hold[block, i]), 1 - i))
    return happened


def _deadlock_record(w: ObjectWorld, run: _Run, t: int, nxt, goal_ids, cfg: Config,
                     livelock: bool = False) -> dict:
    """Who blocks whom, where and at which task stage, when the rollout stops making progress."""
    window = cfg.livelock_window if livelock else cfg.deadlock_window
    length = run.since_best if livelock else run.stall
    win = run.trace[-window:]
    cells = (w.cell(nxt[0]), w.cell(nxt[1]))
    stages = tuple(STAGE_NAMES[_stage(w, nxt, i)] for i in (0, 1))
    reach = int(np.ceil(w.sep))
    adjacent = max(abs(cells[0][0] - cells[1][0]), abs(cells[0][1] - cells[1][1])) <= reach
    pushed = tuple(int(sum(bool(row["blocked"][i]) for row in win)) for i in (0, 1))
    waited = tuple(int(sum(row["action"][i] == WAIT for row in win)) for i in (0, 1))
    stuck = [i for i in (0, 1) if not w.satisfied(nxt, i)]

    def on_route(v: int) -> bool:
        target = w.target(nxt, v, goal_ids[v])
        if target is None or target[1][nxt[v]] >= NO_WAY:
            return False
        from_v = w.grid.distances(int(nxt[v]), v)
        x_b = int(nxt[1 - v])
        return bool(np.isfinite(from_v[x_b]) and from_v[x_b] + target[1][x_b] == target[1][nxt[v]])

    route = tuple(on_route(v) if v in stuck else False for v in (0, 1))
    if livelock:
        kind = "livelock"
        blocker = (1 - stuck[0] if len(stuck) == 1 else
                   1 if pushed[0] > pushed[1] else (0 if pushed[1] > pushed[0] else None))
    elif len(stuck) == 1:
        v = stuck[0]
        in_way = adjacent or route[v]
        kind = "parked_in_path" if in_way else "stalled"
        blocker = 1 - v if in_way else None
    elif adjacent:
        kind = "head_on"
        blocker = 1 if pushed[0] > pushed[1] else (0 if pushed[1] > pushed[0] else None)
    elif route[0] != route[1]:
        kind = "in_path"
        blocker = 1 if route[0] else 0
    else:
        kind, blocker = "stalled", None
    blocked = tuple(stuck) if blocker is None else (1 - blocker,)
    no_way = [i for i in stuck if w.dist_left(nxt, i, goal_ids[i]) >= NO_WAY]
    held = [i for i in no_way if w.hold[nxt[2], 1 - i] == i]
    who = " and ".join(f"R{i} ({stages[i]} at {cells[i]})" for i in blocked)
    what = "made no headway" if livelock else "stuck"

    def why_no_way(i: int) -> str:
        if i in held:
            return f"; R{i}'s object is in R{1 - i}'s hands"
        mine = int(w.hold[nxt[2], i])
        if mine >= 0 and mine != i:                      # Config.hands_full_reward="none"
            return f"; R{i}'s hands are full of {_object_name(mine, i)}"
        return f"; R{i}'s object is out of its reach"

    summary = (f"{who} {what} for {length} transitions"
               + (f", blocked by R{blocker} ({stages[blocker]} at {cells[blocker]})"
                  if blocker is not None else "")
               + "".join(why_no_way(i) for i in no_way)
               + f"; moves into the partner R0/R1 {pushed[0]}/{pushed[1]}, waits "
                 f"{waited[0]}/{waited[1]} in the last {len(win)} transitions")
    return dict(t=t + 1, start=t + 2 - length, window=window, kind=kind, blocker=blocker,
                blocked=blocked, cells=cells, stages=stages, adjacent=bool(adjacent),
                on_route=route, pushes=pushed, waits=waited, summary=summary,
                status=run.all_status(run.status), object_held_by_partner=tuple(held))


def _transition(mdp: SocialMDP, run: _Run, t: int, n_steps: int, uniforms: np.ndarray,
                cfg: Config):
    """One transition of the rollout: choose, move, record.  -> (actions, next state, joint
    state index before, event for the belief update)."""
    w, goal_ids = mdp.w, mdp.goal_ids
    gamma = cfg.gamma
    state = run.state
    s = w.sid(state)
    block = state[2]

    # ---- each robot's value at every cell, term by term (Config.value_maps) -----------------
    t1 = time.perf_counter()
    maps = ([mdp.value_maps(i, state) for i in (0, 1)] if cfg.value_maps else [None, None])
    run.t_maps += time.perf_counter() - t1
    # ---- decisions ---------------------------------------------------------------------------
    q, q_phys, social_reward, psi, flipped = [], [], [], [], []
    for i in (0, 1):
        qi = mdp.q_values(mdp.root[i], s, i)
        twin = mdp.twin[i]
        qp = mdp.q_values(twin, s, i) if twin is not None else qi.copy()
        q.append(qi)
        q_phys.append(qp)
        psi.append(np.exp(log_softmax(qi / cfg.tau_of(i))))   # at robot i's own tau
        flipped.append(twin is not None and argmax_action(qi) != argmax_action(qp))
        if twin is None:                 # expected immediate social reward of each action
            social_reward.append(np.zeros(N_ACTIONS))
        else:
            root_bank, twin_bank = mdp.bank(mdp.root[i][0]), mdp.bank(twin[0])
            social_reward.append(root_bank.R[mdp.root[i][1], s].astype(np.float64)
                                 - twin_bank.R[twin[1], s].astype(np.float64))
    # inertia (Config.inertia): a robot does not undo its last action unless undoing beats
    # every other action by more than the margin; held_back: where that changed the choice
    margin = cfg.inertia_margin
    undo = run.undo if margin > 0.0 else (None, None)
    if cfg.action_selection == "sample":
        q_draw = [q[i].copy() for i in (0, 1)]
        for i in (0, 1):
            if undo[i] is not None and np.isfinite(q_draw[i][undo[i]]):
                q_draw[i][undo[i]] -= margin
        policy = tuple(mdp.execution_policy(i, state, q_draw[i]) for i in (0, 1))
        act = tuple(sample_action(policy[i], float(uniforms[t, i])) for i in (0, 1))
        held_back = tuple(undo[i] is not None and argmax_action(q[i]) == undo[i]
                          and act[i] != undo[i] for i in (0, 1))
    else:
        chosen = [choose_with_inertia(q[i], undo[i], margin) for i in (0, 1)]
        act = tuple(int(a) for a, _ in chosen)
        held_back = tuple(bool(kept) for _, kept in chosen)
        policy = tuple(np.eye(N_ACTIONS)[a] for a in act)

    # ---- the real transition -----------------------------------------------------------------
    nxt = w.unsid(w.step(s, act))
    old_status, old_done = tuple(run.status), tuple(run.done)
    new_status, new_done, failed, handed, took = w.real_step(old_status, old_done, state, nxt,
                                                             act, detail=True)
    run.status, run.done = list(new_status), list(new_done)
    old_cells = w.object_cells(old_status, state, run.where)
    physical = np.array([w.rate(nxt, i, goal_ids[i]) - float(w.cost_c[block, i, act[i]])
                         for i in (0, 1)])
    progress = np.array([physical[i] + gamma * w.solo_value(nxt, i, goal_ids[i])
                         - w.solo_value(state, i, goal_ids[i]) for i in (0, 1)])
    beliefs = tuple(mdp.goal_belief(i) for i in (0, 1))
    object_beliefs = tuple(mdp.object_belief(i) for i in (0, 1))
    social_beliefs = tuple(mdp.social_belief(i) for i in (0, 1))
    moved = tuple(nxt[i] != state[i] for i in (0, 1))
    blocked = tuple(act[i] < WAIT and not moved[i] for i in (0, 1))
    hands_old, hands_new = holds_of(old_status), holds_of(new_status)
    run.undo = tuple(undo_of(act[i], moved[i], hands_old[i] != hands_new[i]) for i in (0, 1))
    modeled = [mdp.modeled_reward(i, s, act, nxt, physical[i]) for i in (0, 1)]
    # a hinderer without a goal of its own: its goal reward is not part of its reward (the
    # model's social term takes it away), so it is booked as moves and costs only
    for i in (0, 1):
        if mdp.spoiler[i]:
            physical[i] = -float(w.cost_c[block, i, act[i]])
            progress[i] = 0.0
    done_now = tuple(bool(new_done[i]) and not old_done[i] for i in (0, 1))
    happened = _events_of(w, run, old_status, new_status, nxt, done_now, failed, block, handed,
                          took)
    for h in happened:
        run.events.append((t + 1,) + h)
        if h[0] == "set down":           # it lies where the robot that set it down stands
            run.where[h[2]] = w.cell(nxt[h[1]])

    if cfg.debug_checks:
        where = f"transition {t + 1}"
        for i in (0, 1):
            legal = np.isfinite(q[i])
            check(legal.any() and bool(legal[act[i]]), f"{where}: R{i} chose an illegal action")
            check(abs(float(policy[i].sum()) - 1.0) < 1e-9 and policy[i][act[i]] > 0.0,
                  f"{where}: R{i} acted outside its policy")
        check(nxt[0] != nxt[1] and not w.conflict(nxt[0], nxt[1]), f"{where}: robots overlap")
        check(w.block_of(new_status, new_done) == nxt[2],
              f"{where}: the transition table disagrees with the hands rule")
        check(all(new_done[i] >= old_done[i] for i in (0, 1)), f"{where}: a goal came undone")
        check(new_status.count(GONE) == sum(new_done)
              and all(new_status.count(1 + i) <= 1 for i in (0, 1)),
              f"{where}: one object per pair of hands, one object per finished goal")

    stage = tuple(_stage(w, state, i) for i in (0, 1))
    run.trace.append(dict(
        step=t + 1,
        pos=(w.cell(state[0]), w.cell(state[1])),
        next_pos=(w.cell(nxt[0]), w.cell(nxt[1])),
        action=tuple(int(a) for a in act),
        move=tuple(tuple(int(v) for v in MOVES[MOVE_OF[a]]) for a in act),
        q=tuple(x.copy() for x in q),
        q_physical=tuple(x.copy() for x in q_phys),
        social_reward=tuple(social_reward),
        psi=tuple(x.copy() for x in psi),
        policy=tuple(x.copy() for x in policy),
        belief=tuple(b.copy() for b in beliefs),
        **(dict(object_belief=tuple(b.copy() for b in object_beliefs)) if mdp.infer_object
           else {}),
        social_belief=tuple(None if b is None else b.copy() for b in social_beliefs),
        value_maps=tuple(maps),
        idle_belief=tuple(mdp.idle_belief(i) for i in (0, 1)),
        stage=stage,
        done=tuple(bool(w.done[block, i]) for i in (0, 1)),
        blocked=blocked,
        inertia=held_back,
        social_flip=tuple(flipped),
        physical_reward=physical.copy(),
        progress=progress.copy(),
        status=run.all_status(old_status), next_status=run.all_status(new_status),
        holds=holds_of(old_status), next_holds=holds_of(new_status),
        object_cells=run.all_cells(old_cells),
        next_object_cells=run.all_cells(w.object_cells(new_status, nxt, run.where)),
        events=tuple(happened)))

    # ---- ledger ------------------------------------------------------------------------------
    for i in (0, 1):
        led = run.ledger[i]
        cost = float(w.cost_c[block, i, act[i]])
        led["steps"] += 1
        led["moves" if moved[i] else "waits"] += 1
        if act[i] >= PICK:
            led["hand_actions"] += 1
            led["by_hand_action"][ACTION_NAMES[act[i]]] += 1
        led["stage_steps"][STAGE_NAMES[stage[i]]] += 1
        led["inertia"] += int(held_back[i])
        led["step_cost"] += cost
        led["physical_reward"] += physical[i] + cost
        led["social_reward"] += modeled[i] - physical[i]
        led["discounted_return"] += gamma ** t * modeled[i]
        led["blocked_by_partner"] += int(blocked[i])
        led["social_flips"] += int(flipped[i])
        k = int(w.hold[block, i])
        led["steps_holding_foreign"] += int(k >= 0 and k != i)
    for kind, by, k, other in happened:
        led = run.ledger[by]
        if kind == "picked up":
            led["pickups"] += 1
            if led["picked_at"] is None:
                led["picked_at"] = t + 1
        elif kind == "lifting":
            led["lifts"] += 1
        elif kind == "put back":
            led["put_backs"] += 1
        elif kind == "put down":
            led["put_downs"] += 1
        elif kind == "dropped":
            led["dropped"] += 1
        elif kind == "took":
            led["took"] += 1
            run.ledger[other]["taken_from"] += 1
            if led["picked_at"] is None:
                led["picked_at"] = t + 1
        elif kind == "handed over":
            led["handed_over"] += 1
            run.ledger[other]["received"] += 1
            if run.ledger[other]["picked_at"] is None:
                run.ledger[other]["picked_at"] = t + 1
        elif kind == "set down":
            if other != by:
                led["set_down_for_partner"] += 1
            if run.ledger[other]["done_at"] is None and run.own_goal[other]:
                run.ledger[other].update(done_at=t + 1, object=k, delivered_by=by)
        elif kind == "set down in vain":
            led["failed_set_downs"] += 1
        elif kind == "handed over in vain":
            led["failed_hand_overs"] += 1

    # ---- progress, deadlock and livelock -----------------------------------------------------
    active = [i for i in (0, 1) if not _finished(mdp, state, i)]
    headway = False                  # an object event that gets somebody somewhere: not an
    for kind, by, k, _ in happened:  # object put back, nor one its robot then takes up again,
        if kind in ("put back", "dropped"):   # nor one its robot puts down where it has put it
            run.put_back[k] = by             # before (a swap leaves one on another's place)
        elif kind == "put down":
            if (by, nxt[by]) not in run.put_down[k]:
                headway = True
                run.put_down[k].add((by, nxt[by]))
            run.put_back[k] = by
        elif not (kind in ("picked up", "lifting") and run.put_back[k] == by):
            headway = True
            if kind == "picked up":
                run.put_back[k] = None
    d_old = [_left(w, state, i, goal_ids) for i in (0, 1)]
    d_new = [_left(w, nxt, i, goal_ids) for i in (0, 1)]
    made_progress = headway or any(d_new[i] < d_old[i] for i in active)
    run.stall = run.stall + 1 if active and not made_progress else 0
    run.max_stall = max(run.max_stall, run.stall)
    improved = headway
    for i in active:
        if headway and d_new[i] > d_old[i] + 1:      # its options changed: a new way to go
            run.best_left[i] = d_new[i]
        elif d_new[i] < run.best_left[i]:
            run.best_left[i] = d_new[i]
            improved = True
    run.since_best = run.since_best + 1 if active and not improved else 0
    if cfg.deadlock_stop and run.deadlock is None:
        if run.stall >= cfg.deadlock_window:
            run.deadlock = _deadlock_record(w, run, t, nxt, goal_ids, cfg)
        elif cfg.livelock_window and run.since_best >= cfg.livelock_window:
            run.deadlock = _deadlock_record(w, run, t, nxt, goal_ids, cfg, livelock=True)

    if cfg.verbose:
        _log_transition(mdp, run, t, n_steps, state, nxt, act, q, q_phys, flipped, blocked,
                        beliefs, happened, cfg)
    run.blocked_run = [run.blocked_run[i] + 1 if blocked[i] else 0 for i in (0, 1)]
    return act, nxt, s, dict(done_now=done_now, failed=failed, handed_in_vain=handed,
                             held=tuple(int(w.hold[block, i]) for i in (0, 1)))


def _log_transition(mdp: SocialMDP, run: _Run, t: int, n_steps: int, state, nxt, act, q,
                    q_phys, flipped, blocked, beliefs, happened, cfg: Config) -> None:
    w, goal_ids, lines = mdp.w, mdp.goal_ids, run.lines
    pre = f"[pairings] t={t + 1:4d}"
    for kind, by, k, other in happened:
        at = w.cell(nxt[by])
        if kind == "handed over":
            what = f"handed {_object_name(k, by)} to R{other}"
        elif kind == "took":
            what = f"took {_object_name(k, by)} from R{other}'s hands"
        elif kind == "dropped":
            what = f"left {_object_name(k, by)} on {at}, another object's place"
        elif kind == "put down":
            what = f"put {_object_name(k, by)} down on the hand-off cell {at}"
        elif kind == "set down":
            alone = ("it could not have finished alone" if run.solo_steps[by] is None
                     else f"solo optimum {run.solo_steps[by]} transitions")
            what = (f"set {_object_name(k, by)} down at its destination {at} -- goal done "
                    f"({alone})" if other == by else
                    f"set {_object_name(k, by)} down at {at}, R{other}'s destination -- "
                    f"R{other}'s goal is done")
        elif kind == "set down in vain":
            what = (f"tried to set {_object_name(k, by)} down for R{other} at {at}: not "
                    f"R{other}'s goal, nothing happens")
        elif kind == "handed over in vain":
            what = (f"tried to hand {_object_name(k, by)} to R{other}: not R{other}'s "
                    "object, nothing happens")
        else:
            what = f"{kind} {_object_name(k, by)} at {at}"
        lines.append(f"{pre} R{by} {what}")
    for i in (0, 1):
        if blocked[i] and run.blocked_run[i] == 0:
            lines.append(f"{pre} R{i} blocked by R{1 - i}: wanted {ACTION_NAMES[act[i]]} from "
                         f"{w.cell(state[i])}")
        if not blocked[i] and run.blocked_run[i] >= 2:
            lines.append(f"{pre} R{i} unblocked after {run.blocked_run[i]} blocked transitions")
        if mdp.root[i][0] >= 1:
            top = int(np.argmax(beliefs[i]))
            if beliefs[i][top] >= 0.5 and top != run.top_goal[i]:
                lines.append(f"{pre} R{i} now believes R{1 - i}'s destination is g{top} "
                             f"(p={beliefs[i][top]:.2f}; true g{goal_ids[1 - i]})")
                run.top_goal[i] = top
    if run.stall == STALL_WARNING:
        lines.append(f"{pre} WARNING no progress for {STALL_WARNING} transitions; R0 "
                     f"{w.cell(nxt[0])} R1 {w.cell(nxt[1])}")
    if run.deadlock is not None and run.deadlock["t"] == t + 1:
        lines.append(f"{pre} {run.deadlock['kind'].upper()}: {run.deadlock['summary']} -- "
                     "rollout stopped")
    if cfg.debug_every and (t % cfg.debug_every == 0 or t == n_steps - 1):
        new_holds = holds_of(run.status)
        for i in (0, 1):
            a = act[i]
            left = w.dist_left(nxt, i, goal_ids[i])
            soc = np.zeros(N_ACTIONS)
            both = np.isfinite(q[i]) & np.isfinite(q_phys[i])
            soc[both] = q[i][both] - q_phys[i][both]
            line = (f"{pre} R{i} {w.cell(state[i])}->{w.cell(nxt[i])} {ACTION_NAMES[a]:>4} "
                    f"{STAGE_NAMES[_stage(w, nxt, i)]:9s} left="
                    f"{'none' if left >= NO_WAY else left:>4} holds="
                    f"{'-' if new_holds[i] < 0 else new_holds[i]} Q={q[i][a]:+9.3f} "
                    f"[physical {q_phys[i][a]:+9.3f} | social {soc[a]:+9.3f}] "
                    f"goal={G.goal_name(mdp.goals[i])}")
            if flipped[i]:
                line += f" SOCIAL-FLIP(physical argmax {ACTION_NAMES[argmax_action(q_phys[i])]})"
            if blocked[i]:
                line += " BLOCKED"
            if mdp.root[i][0] >= 1:
                line += (f" | b(R{1 - i} destination) "
                         f"{top_hypotheses(beliefs[i], goal_ids[1 - i], cfg.debug_topk)}")
            lines.append(line)


def rollout(world: ObjectWorld, goal_ids: Sequence[int], solo_steps: Sequence[int],
            levels: Sequence[int], social_goals: Sequence[int], weights: Sequence[float],
            cfg: Config, n_steps: int, scenery: Sequence = ()) -> dict:
    """Roll one pair of robots out for at most ``n_steps`` transitions.

    world, goal_ids, solo_steps, scenery: what :func:`build_world` returned.
    social_goals: the index of each robot's social goal; a robot at level 0 has none.
    -> the result record (see :func:`social_pairings.pairing.run_pairing`)."""
    tic0 = time.perf_counter()
    threads = configure_threads(cfg.num_threads)
    w = world
    mdp = SocialMDP(w, goal_ids, levels, social_goals, weights, cfg, threads)
    uniforms = np.random.default_rng(cfg.seed).random((n_steps, 2))
    run = _Run(w, solo_steps, scenery, own_goal=tuple(not x for x in mdp.spoiler))
    if cfg.verbose:
        for line in _describe(mdp, run, n_steps):
            print(f"[pairings] {line}", flush=True)
    mdp.allocate()
    info0 = mdp.solve_level0()
    if cfg.verbose:
        print(f"[pairings] level 0: {info0['models']} models, certificate max|TV-V| "
              f"{info0['res_max']:.1e} (<= {cfg.residual_tol:.1e}: {info0['ok']}), "
              f"{info0['t_inputs'] + info0['t_solve'] + info0['t_psi']:.2f}s", flush=True)
    timers = dict(inputs=0.0, solve=0.0, decide=0.0)
    for t in range(n_steps):
        if run.deadlock is not None:
            break
        if cfg.stop_when_done and all(_finished(mdp, run.state, i) for i in (0, 1)):
            if run.tail >= cfg.done_tail:
                break
            run.tail += 1
        t_step = time.perf_counter()
        mdp.plan(run.state)
        if t == 0:
            mdp.init_models()
        solved = {}
        for depth in (1, 2):
            t1 = time.perf_counter()
            mdp.inputs(depth)
            timers["inputs"] += time.perf_counter() - t1
            t1 = time.perf_counter()
            solved[depth] = mdp.solve(depth)
            timers["solve"] += time.perf_counter() - t1
        for depth, rows in solved.items():
            if not rows.size:
                continue
            bank = mdp.bank(depth)
            run.solves += int(rows.size)
            run.pi_total += float(bank.pi_iters[rows].sum())
            run.krylov_total += float(bank.krylov_iters[rows].sum())
            run.pi_max = max(run.pi_max, int(bank.pi_iters[rows].max()))
            run.krylov_max = max(run.krylov_max, int(bank.krylov_iters[rows].max()))
            run.residual_max = max(run.residual_max, float(bank.residual[rows].max()))
            run.unconverged += int((~bank.ok[rows]).sum())
        t1 = time.perf_counter()
        act, nxt, s, event = _transition(mdp, run, t, n_steps, uniforms, cfg)
        if cfg.partner_observed != "action":     # what each robot was seen to have done
            run.trace[-1]["seen_as"] = tuple(list(mdp.observed_actions(act, s, nxt, i))
                                             for i in (0, 1))
        mdp.observe(act, s, nxt, event)
        run.trace[-1]["next_belief"] = tuple(mdp.goal_belief(i).copy() for i in (0, 1))
        if mdp.infer_object:
            run.trace[-1]["next_object_belief"] = tuple(mdp.object_belief(i).copy()
                                                        for i in (0, 1))
        run.state = nxt
        timers["decide"] += time.perf_counter() - t1
        if cfg.verbose:                          # progress as it happens (a long run)
            print(f"[pairings] t={t + 1:4d}/{n_steps}: {ACTION_NAMES[act[0]]} / "
                  f"{ACTION_NAMES[act[1]]}, holding {tuple(int(w.hold[nxt[2], i]) for i in (0, 1))}"
                  f", done {tuple(bool(w.done[nxt[2], i]) for i in (0, 1))}, "
                  f"{time.perf_counter() - t_step:.0f} s", flush=True)
    result = _result(mdp, run, info0, cfg)
    result["wall_time_s"] = time.perf_counter() - tic0
    result["timing"] = dict(level0=info0["t_inputs"] + info0["t_solve"] + info0["t_psi"],
                            **timers, value_maps=run.t_maps, threads=threads)
    if cfg.verbose:
        for line in run.lines:
            print(line, flush=True)
        for i in (0, 1):
            led = run.ledger[i]
            print(f"[pairings] R{i}: picked_at={led['picked_at']} done_at={led['done_at']} "
                  f"(solo {run.solo_steps[i]}) moves={led['moves']} waits={led['waits']} "
                  f"hand actions={led['hand_actions']} blocked={led['blocked_by_partner']} "
                  f"physical="
                  f"{led['physical_reward']:.3f} cost={led['step_cost']:.3f} social="
                  f"{led['social_reward']:+.3f}", flush=True)
        print(f"[pairings] {result['steps']} transitions, stopped by {result['stopped_by']}, "
              f"{run.solves} solves, max residual {run.residual_max:.1e}, "
              f"{result['wall_time_s']:.1f}s", flush=True)
    return result


def _result(mdp: SocialMDP, run: _Run, info0: dict, cfg: Config) -> dict:
    w, goal_ids = mdp.w, mdp.goal_ids
    state = run.state
    completed = all(_finished(mdp, state, i) for i in (0, 1))
    stopped_by = "deadlock" if run.deadlock is not None else ("done" if completed else "cap")
    return dict(
        actions=ACTION_NAMES,              # by number (since 2026-10-16; "use" before)
        steps=len(run.trace), stopped_by=stopped_by, completed=completed,
        done_at=tuple(led["done_at"] for led in run.ledger),
        ledger=run.ledger, trace=run.trace, events=run.events, deadlock=run.deadlock,
        max_stall=run.max_stall,
        stage=tuple(STAGE_NAMES[_stage(w, state, i)] for i in (0, 1)),
        status=run.all_status(run.status), holds=holds_of(run.status),
        object_cells=run.all_cells(w.object_cells(run.status, state, run.where)),
        final_pos=(w.cell(state[0]), w.cell(state[1])),
        final_belief=tuple(mdp.goal_belief(i) for i in (0, 1)),
        **(dict(final_object_belief=tuple(mdp.object_belief(i) for i in (0, 1)))
           if mdp.infer_object else {}),
        final_social_belief=tuple(mdp.social_belief(i) for i in (0, 1)),
        levels=mdp.levels, social_goals=tuple(G.goal_name(t) for t in mdp.goals),
        own_goal=tuple(not x for x in mdp.spoiler),
        weights=mdp.weights,
        solo_steps=run.solo_steps,
        scene=dict(objects=w.objects + run.scenery, modelled_objects=w.M, starts=w.starts,
                   candidates=w.goals, destinations=tuple(w.goals[g] for g in goal_ids),
                   goal_ids=tuple(goal_ids), handoff_cells=w.handoff, rooms=w.rooms,
                   cells=tuple(w.cell(x) for x in range(w.N))),
        numerically_accepted=run.unconverged == 0 and bool(info0["ok"]),
        solver=dict(solves=run.solves, residual_max=run.residual_max,
                    residual_target=cfg.residual_tol, unconverged=run.unconverged,
                    pi_iters_mean=run.pi_total / max(run.solves, 1), pi_iters_max=run.pi_max,
                    krylov_iters_mean=run.krylov_total / max(run.solves, 1),
                    krylov_iters_max=run.krylov_max, level0_residual=info0["res_max"],
                    level0_ok=bool(info0["ok"]), models=(mdp.L0.M, mdp.L1.M, mdp.L2.M),
                    joint_states=int(w.S), blocks_in_use=int(mdp.reachable.size)))


def _taus(cfg: Config) -> str:
    """Eq. 7's tau in reward units: one number, or agent_1's/agent_2's where they differ."""
    a, b = cfg.tau_of(0), cfg.tau_of(1)
    return f"{a:g}" if a == b else f"{a:g}/{b:g}"


def _describe(mdp: SocialMDP, run: _Run, n_steps: int) -> list:
    w, cfg, goal_ids = mdp.w, mdp.cfg, mdp.goal_ids
    lines = [
        f"grid {w.shape[0]}x{w.shape[1]} at {cfg.resolution:g} m/cell | N={w.N} cells, M={w.M} "
        f"objects in the model (+{len(run.scenery)} scenery), K={w.K} candidate destinations | "
        f"B={w.B} object configurations "
        f"({mdp.reachable.size} in use), S={w.S:,} joint states | {mdp.threads} threads | "
        f"solver arrays <= {mdp.memory_bytes / 2 ** 30:.2f} GiB",
        f"gamma={cfg.gamma:.6f} rho={cfg.rho:.4f} c(move/stay/pick/hand)={cfg.costs[0]:.4f}/"
        f"{cfg.costs[WAIT]:.4f}/{cfg.pick_cost_rate * cfg.dt:.4f}/"
        f"{cfg.use_cost_rate * cfg.dt:.4f} tau={_taus(cfg)} | hands="
        f"{'automatic' if cfg.automatic_hands else f'by hand, pick in {cfg.pick_steps}'} | "
        f"execution={cfg.action_selection} | forced_swap={cfg.forced_swap} held_reward="
        f"{w.held_reward} idle_hypothesis={cfg.idle_hypothesis}",
        f"partner goals: object {cfg.partner_object}, destination inferred, social goal "
        f"{cfg.partner_social_goal} | {mdp.PG} physical goals per robot "
        f"({w.M} objects x {w.K} destinations)" if mdp.infer_object else
        f"partner goals: object known, destination inferred, social goal "
        f"{cfg.partner_social_goal}",
    ]
    if w.H or w.has_rooms:
        lines.append(
            f"hand-off cells {list(w.handoff)} | "
            + ("no rooms" if not w.has_rooms else "rooms of "
               + " and ".join(f"{int(w.allowed[i].sum())} cells (R{i})" for i in (0, 1)))
            + f" | set_down_for_partner={cfg.set_down_for_partner} hand_over={cfg.hand_over}")
    for i in (0, 1):
        k = goal_ids[i]
        goal = mdp.goals[i]
        partner_goal = mdp.goals[1 - i] if cfg.partner_social_goal == "known" else None
        alone = ("it cannot finish alone" if run.solo_steps[i] is None
                 else f"solo optimum {run.solo_steps[i]} transitions")
        term = G.describe(goal, mdp.levels[i], partner_goal, competition=cfg.competition,
                          exchange_level1=cfg.exchange_level1,
                          cooperate_with=G.cooperation_partners(cfg.cooperate_with),
                          hinder_own_goal=cfg.hinder_own_goal, conflict=cfg.conflict)
        lines.append(
            f"R{i}: level {mdp.levels[i]}, weight {mdp.weights[i]:g}, {term}"
            f" | start {w.starts[i]}, object "
            f"{w.objects[i]}, destination g{k}={w.goals[k]}, {alone}")
    lines.append(f"models: level 0 {mdp.L0.M}, level 1 {mdp.L1.M} ({mdp.L1.H} of them partner "
                 f"hypotheses), level 2 {mdp.L2.M} | at most {n_steps} transitions")
    return lines
