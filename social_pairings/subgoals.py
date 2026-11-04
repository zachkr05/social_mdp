"""The partner's subgoals, and the protected ("cut") likelihood of its physical goal
(``Config.physical_likelihood="cut"``, since 2026-11-04).

The user, of green reading red's set-down at G3 for green as "red's own goal is far" (the
hall_crossing testbed run, G5 0.94 at step 49 with red in front of G1): "So how do you think
we solve Green thinking that red wants to deliver to G5 and instead thinking its G1 since its
closer"; "O so it goes back to that latent problem we were going to solve and trying to
incorporate the subgoals"; the formalization of it (its Eq. 25), shown on that run; then
"Ok let's go ahead and implement this and push to a new branch".

What went wrong: Eq. 6 reads the partner's action through its level-1 models averaged over
the social posterior, and under the cooperation hypothesis how generous the partner is
depends on what its own goal is worth -- which discounting makes less the farther the goal.
Every generous act (walking to a set-down for the robot, the set-down, walking toward the
robot that would take its object) then reads as "its own goal is far".  The fix keeps the
partner's own task and its acts for the robot apart.  The likelihood of robot j's action a
under its physical goal g, in Eq. 6's place:

    L(g) = pi N_g(a) + (1 - pi) S(a)        (the idle mode mixed in after, as for any likelihood)

    N_g   the partner's level-1 model with physical goal g and no social goal ("none"): its
          own task, the other robot predicted at level 0 as in every level-1 model
    S(a)  the social background: the subgoals' policies kappa_z(a | s), weighted by the
          predictive belief over (social goal, subgoal) -- the same for every g
    pi    the predictive probability that the partner pursues its own task (no subgoal)

A subgoal's policy does not depend on the partner's own goal, so an act that only a subgoal
explains moves no physical hypothesis against another; a move its own task explains counts
-- weakened toward neutral by the share (1 - pi) S(a), never reversed (the ratio of two
hypotheses' L lies between 1 and the ratio of their N).  The social posterior is Eq. 5 as
before, on the level-1 social models; the subgoal belief P(z | g, xi) is updated within each
(g, xi) and predicted by the subgoal dynamics, so the social models never multiply the
physical posterior a second time -- the "cut" of modular Bayesian inference (Plummer 2015,
Statistics and Computing; Jacob, Murray, Holmes & Robert 2017, arXiv 1708.08719).  The robots'
planning is untouched (an inference-only first version).

Subgoals of robot j toward the robot that reads it, i:

    deliver(c)  holding an object k: take it to candidate c and set it down there for i
    handoff(h)  holding k: put it down on hand-off cell h
    meet        holding k: go next to i and hand it over
    fetch(k)    hands empty: go to object k where it lies and pick it up (for i)
    keepaway    keep away from i

cooperation and exchange may pursue the first four, weighted by j's belief about i's goal
(mu(k, c) that i wants k on c, mu(k) that i wants k); conflict, competition and hinder
keepaway; coercion and none none.  The prior of a new subgoal under j's own goal g:

    chi(z | g, xi, s) proportional to w_xi(z) exp(-lambda dC_g(z; s)),

dC the extra moves the subgoal adds to j's way to its own next waypoint (its goal object
where it lies, or its destination while it holds it) -- moves, not values, so that the far
goals are not the generous ones.  A subgoal starts with probability ``option_start`` per
transition, ends at its act (the set-down, the hand-over, the pick-up) or when it is no
longer possible, is interrupted with ``option_switch``; keepaway ends with
``option_keepaway_end``.  Its policy is a softmax at ``option_temperature`` of
``-c(a) - c(move) J_z(s')`` over j's open actions, J_z the moves left to its act (keepaway:
``-c(a) + c(move) x`` the distance from i).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import goals as G
from .constants import GIVE, HOME, N_ACTIONS, PICK, PLACE, WAIT

__all__ = ["Subgoals", "HELPING", "HOSTILE"]

HELPING = (G.COOPERATION, G.EXCHANGE)
HOSTILE = (G.CONFLICT, G.COMPETITION, G.HINDER)
FAR = 10 ** 6


class Subgoals:
    """The subgoals of robot ``j`` as the robot that reads it, ``1 - j``, holds them possible:
    their list, their policies at a state, their prior under each of j's physical goals and
    their dynamics.  ``hyp_goals``: the social goals the reader holds possible for j (the
    order of the social posterior)."""

    def __init__(self, world, cfg, j: int, hyp_goals) -> None:
        w = self.w = world
        self.cfg, self.j, self.i = cfg, int(j), 1 - int(j)
        self.items: List[Tuple[str, int]] = (
            [("deliver", c) for c in range(w.K)] + [("handoff", h) for h in range(len(w.hand))]
            + [("meet", -1)] + [("fetch", k) for k in range(w.M)] + [("keepaway", -1)])
        self.L = len(self.items)
        self.kinds = [kind for kind, _ in self.items]
        self.hyp_goals = tuple(int(t) for t in hyp_goals)
        self.allowed = np.array([[self._allowed(t, kind) for kind in self.kinds]
                                 for t in self.hyp_goals], dtype=bool)          # (nT, L)
        scale = float(cfg.costs[0]) if cfg.temperature_unit == "move_cost" else 1.0
        self.tau = float(cfg.option_temperature) * scale
        self.c_move = float(cfg.costs[0])
        self.lam = float(cfg.option_detour)
        self.dist = self._distances()                                              # (N, N)
        cells = np.asarray(w.cells)
        self.adjacent = [np.flatnonzero(np.abs(cells - cells[x]).sum(1) == 1)
                         for x in range(w.N)]

    @staticmethod
    def _allowed(goal: int, kind: str) -> bool:
        if kind == "keepaway":
            return goal in HOSTILE
        return goal in HELPING

    def name(self, n: int) -> str:
        kind, arg = self.items[n]
        if kind == "deliver":
            return f"deliver to G{arg}"
        if kind == "handoff":
            return f"hand-off cell {arg}"
        if kind == "fetch":
            return f"fetch O{arg + 1}"
        return kind

    # ---- geometry -----------------------------------------------------------------------------
    def _distances(self) -> np.ndarray:
        """(N, N) the number of robot j's moves between two cells (FAR: none)."""
        w, j = self.w, self.j
        nb, ok = np.asarray(w.nb_by[j]), np.asarray(w.valid_by[j])
        D = np.full((w.N, w.N), FAR, dtype=np.int64)
        for src in range(w.N):
            D[src, src] = 0
            frontier = [src]
            while frontier:
                nxt = []
                for x in frontier:
                    for m in range(4):
                        if ok[x, m]:
                            y = int(nb[x, m])
                            if y != x and D[src, y] == FAR:
                                D[src, y] = D[src, x] + 1
                                nxt.append(y)
                frontier = nxt
        return D

    def object_cell(self, b: int, k: int) -> int:
        """The cell where object k lies in block b, or -1 (in somebody's hands, being lifted,
        delivered)."""
        w = self.w
        st = int(w.status[b, k])
        if st == HOME:
            return int(w.home[k])
        if 4 <= st < 4 + int(w.L):
            return int(w.lie[st - 4])
        return -1

    def targets(self, n: int, state) -> np.ndarray:
        """The cells where subgoal n's act takes place, at ``state`` (empty: none)."""
        w = self.w
        kind, arg = self.items[n]
        b = int(state[2])
        if kind == "deliver":
            return np.array([int(w.dest[arg])])
        if kind == "handoff":
            return np.array([int(w.hand[arg])])
        if kind == "meet":
            return self.adjacent[int(state[self.i])]
        if kind == "fetch":
            c = self.object_cell(b, arg)
            return np.array([c]) if c >= 0 else np.zeros(0, dtype=np.int64)
        return np.zeros(0, dtype=np.int64)

    def feasible(self, state) -> np.ndarray:
        """(L,) bool: the subgoals robot j can pursue at ``state``."""
        w, j = self.w, self.j
        b = int(state[2])
        x = int(state[j])
        holding = int(w.hold[b, j]) >= 0
        busy = bool(w.busy[b, j])
        out = np.zeros(self.L, dtype=bool)
        for n, (kind, arg) in enumerate(self.items):
            if kind == "keepaway":
                out[n] = True
                continue
            if busy or holding != (kind in ("deliver", "handoff", "meet")):
                continue
            tg = self.targets(n, state)
            out[n] = tg.size > 0 and int(self.dist[x, tg].min()) < FAR
        return out

    # ---- the subgoals' policies (independent of robot j's own goal) -----------------------------
    def policies(self, state, legal: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """(L, 10) kappa_z(a | s) of every subgoal at ``state`` (zero rows: not possible);
        ``legal`` (10,) robot j's open actions, ``cost`` (10,) their costs."""
        w, j = self.w, self.j
        x, xi = int(state[j]), int(state[self.i])
        nb = np.asarray(w.nb_by[j])
        feas = self.feasible(state)
        act_of = {"deliver": PLACE, "handoff": PLACE, "meet": GIVE, "fetch": PICK}
        out = np.zeros((self.L, N_ACTIONS))
        for n, (kind, _) in enumerate(self.items):
            if not feas[n]:
                continue
            score = np.full(N_ACTIONS, -np.inf)
            if kind == "keepaway":
                for a in np.flatnonzero(legal):
                    y = int(nb[x, a]) if a < WAIT else x
                    score[a] = -cost[a] + self.c_move * min(int(self.dist[y, xi]), w.N)
            else:
                tg = self.targets(n, state)
                act = act_of[kind]
                c_act = float(cost[act])
                J = lambda y: int(self.dist[y, tg].min())                    # noqa: E731
                for a in np.flatnonzero(legal):
                    if a <= WAIT:
                        y = int(nb[x, a]) if a < WAIT else x
                        score[a] = -cost[a] - self.c_move * J(y) - c_act
                    elif a == act and J(x) == 0:
                        score[a] = -c_act                       # the subgoal's own act
                    else:
                        score[a] = -cost[a] - self.c_move * J(x) - c_act
            ok = np.isfinite(score)
            if ok.any():
                z = np.where(ok, score / self.tau, -np.inf)
                z = np.exp(z - z[ok].max())
                out[n] = z / z.sum()
        return out

    def acted(self, state, a: int) -> np.ndarray:
        """(L,) bool: robot j's action ``a`` at ``state`` was the act that ends the subgoal."""
        w, j = self.w, self.j
        b = int(state[2])
        x = int(state[j])
        holding = int(w.hold[b, j]) >= 0
        out = np.zeros(self.L, dtype=bool)
        for n, (kind, arg) in enumerate(self.items):
            if kind in ("deliver", "handoff"):
                out[n] = holding and a == PLACE and x in self.targets(n, state).tolist()
            elif kind == "meet":
                out[n] = holding and a == GIVE
            elif kind == "fetch":
                out[n] = (not holding) and a == PICK and x == self.object_cell(b, arg)
        return out

    # ---- the prior of a new subgoal, and the dynamics ---------------------------------------------
    def prior(self, state, own_obj: np.ndarray, own_dst: np.ndarray, mu: np.ndarray,
              mu_obj: np.ndarray, mu_dst: np.ndarray) -> np.ndarray:
        """(PG, nT, L) chi(z | g, xi, s): robot j's physical goals g = (own_obj[g],
        own_dst[g]); ``mu`` (PG',) robot j's belief about the reader's physical goal, its
        hypotheses (mu_obj, mu_dst).  Rows that are all zero: no subgoal can start."""
        w, j = self.w, self.j
        b = int(state[2])
        x = int(state[j])
        feas = self.feasible(state)
        held = int(w.hold[b, j])
        mu = np.asarray(mu, dtype=float)
        m_obj = np.array([mu[mu_obj == k].sum() for k in range(w.M)])
        weight = np.zeros(self.L)
        for n, (kind, arg) in enumerate(self.items):
            if not feas[n]:
                continue
            if kind == "deliver":
                weight[n] = mu[(mu_obj == held) & (mu_dst == arg)].sum() if held >= 0 else 0.0
            elif kind == "handoff":
                weight[n] = m_obj[held] / max(len(w.hand), 1) if held >= 0 else 0.0
            elif kind == "meet":
                weight[n] = m_obj[held] if held >= 0 else 0.0
            elif kind == "fetch":
                weight[n] = m_obj[arg]
            else:
                weight[n] = 1.0
        PG = len(own_obj)
        out = np.zeros((PG, len(self.hyp_goals), self.L))
        done = bool(w.done[b, j])
        for g in range(PG):
            o, d = int(own_obj[g]), int(own_dst[g])
            if done:
                wp = -1
            elif held == o:
                wp = int(w.dest[d])
            else:
                c = self.object_cell(b, o)
                wp = c if c >= 0 else int(w.dest[d])
            detour = np.zeros(self.L)
            for n, (kind, _) in enumerate(self.items):
                if weight[n] <= 0.0 or kind == "keepaway" or wp < 0:
                    continue
                tg = self.targets(n, state)
                via = (self.dist[x, tg] + self.dist[tg, wp]).min()
                detour[n] = max(float(via - self.dist[x, wp]), 0.0)
            row = weight * np.exp(-self.lam * detour)
            for t in range(len(self.hyp_goals)):
                v = np.where(self.allowed[t], row, 0.0)
                total = v.sum()
                if total > 0.0:
                    out[g, t] = v / total
        return out

    def predict(self, belief: np.ndarray, chi: np.ndarray, ended: np.ndarray) -> np.ndarray:
        """The subgoal belief (PG, nT, 1 + L) after one transition: index 0 the partner's own
        task.  From it a subgoal starts with probability ``option_start`` (drawn from
        ``chi``, where one can); a subgoal ends where ``ended`` (its act, or no longer
        possible), else is interrupted with ``option_switch`` (keepaway:
        ``option_keepaway_end``)."""
        cfg = self.cfg
        out = np.zeros_like(belief)
        own = belief[:, :, 0]
        can = chi.sum(-1) > 0.0
        p_start = np.where(can, float(cfg.option_start), 0.0)
        out[:, :, 0] += own * (1.0 - p_start)
        out[:, :, 1:] += (own * p_start)[..., None] * chi
        stop = np.array([float(cfg.option_keepaway_end) if kind == "keepaway"
                         else float(cfg.option_switch) for kind in self.kinds])
        stop = np.where(ended, 1.0, stop)
        sub = belief[:, :, 1:]
        out[:, :, 0] += (sub * stop).sum(-1)
        out[:, :, 1:] += sub * (1.0 - stop)
        total = out.sum(-1, keepdims=True)
        return np.where(total > 0.0, out / np.where(total > 0.0, total, 1.0), belief)
