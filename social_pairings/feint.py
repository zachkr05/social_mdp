"""A decoy feint (``Config.decoy_feint``, 2026-11-04): a level-2 robot that looks ahead at
how its own actions move its partner's belief.

The user, of hinder / coercion on the two-room scene (the hinderer holding the coercer's
object to the 60-step cap): "How can we alter the reward function such that the coercive
agent will go pick up the green object to manipulate the green agent into thinking it
actually wants the green object", then, of the decoy-and-transfer document that answered it,
"Ok how about u actually just implement it the rerun the scenario then show me the mp4 again".

The level-2 planner (models.SocialMDP) predicts its partner by the partner's level-1 models
with the partner's belief about the robot as it stands, held fixed for the whole plan (Eq.
4): inside the plan nothing the robot does changes what the partner believes, so a lure is
worth nothing.  The document's remedy is to plan over the partner's belief updates as well
(its Sections 3 and 6; the I-POMDP principle, Gmytrasiewicz & Doshi 2005).  Over the whole
belief that is out of reach of value iteration, so here it is its finite-lookahead form, with
one option, the decoy:

* when: the robot's social goal is one of ``Config.feint_goals`` (default coercion), it is
  at level 2, its partner infers its object (``partner_object="inferred"``), its hands are
  empty, its own object is in its partner's hands or lies free but guarded (the partner no
  farther from it than the robot, round the partner), and an object other than that one
  lies free (the decoy); not within ``feint_cooldown`` transitions of its last option;
* what it looks ahead at: walking to the decoy by a shortest path round the partner and
  lifting it, the partner meanwhile doing what the robot's model predicts (the most likely
  action of Eq. 4's prediction); each step's transition fed to the PARTNER'S OWN belief
  update (``SocialMDP.updated_belief``: the cut, the stages, the hazards, the window, as
  configured) -- the document's requirement: "The planner must calculate p_{t+1} through
  this filter";
* what the option's end is worth: the robot's level-2 value at the end state with the
  partner's belief as that update leaves it -- every model solved afresh at that belief, in
  a second SocialMDP (``evaluator``; it doubles the memory of a rollout) -- or, where the
  partner's belief in the decoy moved by less than ``feint_min_shift``, the plan's own value
  there (nothing to re-solve);
* the choice: the option's discounted return (its steps' expected rewards, then the end
  value) against the value of the plan as it is, max_a Q(s, a); taken if it wins by more than
  ``feint_margin``;
* following it, in three phases: to the decoy and lift it (even where the partner lets go of
  the robot's object meanwhile); hold it while the partner still holds the robot's object
  (the planner's moves, nothing let go of); and, the object lying free, carry the decoy to
  it and swap the two ("swap": the decoy stays on that cell, the object comes into the
  robot's hands), after which the planner delivers it.  The decoy stays in hand until the
  swap because with hands full the partner's belief about the robot's object does not move
  (the posteriors' stages): walking back empty-handed is what gives the robot away.  It
  ends early where it no longer makes sense: the decoy taken or lost, the robot's object
  free and nearer to the robot than to the partner (the planner goes for it), or too long.

The first version ended the option the moment the partner let go of the robot's object (the
user, of the two-room rerun: "Why does it just stop after coercing it both of them just stop
moving"): the coercer turned back empty-handed one step short of the decoy, the hinderer
read it at once (its belief that the coercer wanted the decoy 0.99 -> 0.09 -> 0.01) and
guarded the object two moves from it, the coercer four, and both waited to the deadlock.
"Once it drops the object we want red to actuall y go ahead and pick it up": the phases
above.

No reward is added: the document's kappa = 0, "the decoy must earn its value through improved
true-task outcomes under the belief-aware transition model".  What it leaves out: the
partner's belief moving again after the option (the plan's frozen belief from there), and
the partner's reaction to its own changing belief during the walk (the prediction of now).
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from . import goals as G
from .constants import GIVE, HELD0, HELD1, HOME, PICK, PLACE, SWAP, WAIT

__all__ = ["DecoyFeint"]


def _finite_max(q: np.ndarray) -> float:
    q = np.asarray(q, dtype=np.float64)
    ok = np.isfinite(q)
    return float(q[ok].max()) if ok.any() else -np.inf


def _holds(status) -> tuple:
    """(object in R0's hands, object in R1's hands), -1 = free hands (rollout.holds_of)."""
    return tuple(next((k for k, v in enumerate(status) if v == 1 + i), -1) for i in (0, 1))


class DecoyFeint:
    """The decoy option of the robots it applies to, for one rollout.  ``args``: what
    rollout() built its SocialMDP from, (world, goal_ids, levels, social_goals, weights,
    cfg, threads) -- the evaluator is built from the same."""

    def __init__(self, mdp, args: tuple) -> None:
        self.args = args
        self.cfg = cfg = mdp.cfg
        names = {G.parse_goal(g) for g in cfg.feint_goals}
        self.robots = tuple(i for i in (0, 1)
                            if mdp.levels[i] == 2 and int(mdp.goals[i]) in names
                            and mdp.infer_object and not cfg.automatic_hands)
        self.evaluator = None                  # a second SocialMDP, built at its first use
        self.started = False                   # its models warm-started
        self.option = {0: None, 1: None}       # the option being followed
        self.ended = {0: None, 1: None}        # transition of the last option's end
        self.log: list = []                    # every evaluation
        self.events: list = []                 # (transition, robot, "begun"/"ended", ...)

    # ---- geometry ------------------------------------------------------------------------
    @staticmethod
    def first_move(w, robot: int, start: int, goal: int, blocked: int, q: np.ndarray):
        """(move, moves to go) of a shortest path start -> goal on the robot's moves round
        the cell ``blocked``, among the moves legal in ``q``; (None, inf) if no legal move
        gets closer.  ``start == goal``: (None, 0)."""
        if start == goal:
            return None, 0
        nb, ok = w.nb_by[robot], w.valid_by[robot]
        dist = {goal: 0}
        todo = deque([goal])
        while todo:                            # moves are reversible: distances to the goal
            x = todo.popleft()
            for a in range(4):
                if ok[x, a]:
                    y = int(nb[x, a])
                    if y not in dist and y != blocked:
                        dist[y] = dist[x] + 1
                        todo.append(y)
        best, best_d = None, np.inf
        here = dist.get(start, np.inf)
        for a in range(4):
            if ok[start, a] and np.isfinite(q[a]):
                d = dist.get(int(nb[start, a]), np.inf)
                if d < best_d and d < here:
                    best, best_d = a, d
        return best, best_d + 1 if best is not None else np.inf

    @staticmethod
    def moves(w, robot: int, start: int, goal: int, blocked: int = -1) -> float:
        """Moves of the robot's shortest path start -> goal round the cell ``blocked``."""
        if start == goal:
            return 0
        nb, ok = w.nb_by[robot], w.valid_by[robot]
        seen = {start: 0}
        todo = deque([start])
        while todo:
            x = todo.popleft()
            for a in range(4):
                if ok[x, a]:
                    y = int(nb[x, a])
                    if y not in seen and y != blocked:
                        seen[y] = seen[x] + 1
                        if y == goal:
                            return seen[y]
                        todo.append(y)
        return np.inf

    @staticmethod
    def free(w, status, k: int, robot: int) -> bool:
        """Object k lies where it can be picked up (its place or where it was left), or is
        being lifted by ``robot`` itself."""
        v = int(status[k])
        if v in (HELD0, HELD1):
            return False
        li = w.lifter_of(v)
        if li >= 0:
            return li == robot
        return v == HOME or v >= 4

    @staticmethod
    def cell_of(w, status, state, where, k: int) -> int:
        return int(w.cell_id[tuple(w.object_cells(status, state, where)[k])])

    @staticmethod
    def mass(mdp, belief, robot: int, k: int) -> float:
        """The partner's belief that robot ``robot`` wants object k."""
        return float(np.bincount(mdp.pg_obj[robot], weights=belief.phys(),
                                 minlength=mdp.w.M)[k])

    # ---- the decision ----------------------------------------------------------------------
    def decide(self, mdp, t: int, state, status, where, act, q) -> tuple:
        """The joint action, the option's step in place of the planner's where it applies."""
        out = list(int(a) for a in act)
        for i in self.robots:
            a = self._decide(mdp, t, state, status, where, q, i, int(act[i]))
            if a is not None:
                out[i] = int(a)
        return tuple(out)

    def guarded(self, w, state, status, where, i: int) -> bool:
        """Robot i's own object lies free, and its partner is no farther from it than the
        robot is (round the partner): a guard the robot cannot outrun to it."""
        p = 1 - i
        if not self.free(w, status, i, i) or w.lifter_of(int(status[i])) >= 0:
            return False
        cell = self.cell_of(w, status, state, where, i)
        return (self.moves(w, p, int(state[p]), cell)
                <= self.moves(w, i, int(state[i]), cell, blocked=int(state[p])))

    def _decide(self, mdp, t, state, status, where, q, i, planned):
        w, cfg, p = mdp.w, self.cfg, 1 - i
        block = int(state[2])
        holds = _holds(status)
        if w.done[block, i]:
            if self.option[i] is not None:
                self._end(mdp, t, i, "its goal done")
            return None
        if self.option[i] is not None:          # following the option
            return self._follow(mdp, t, state, status, where, q, i, planned, holds)
        # considering it: hands empty, and its object in the partner's hands or guarded
        if holds[i] >= 0 or any(w.lifter_of(int(v)) == i for v in status):
            return None
        held = holds[p] == i
        if not held and not self.guarded(w, state, status, where, i):
            return None
        if self.ended[i] is not None and t - self.ended[i] < cfg.feint_cooldown:
            return None
        best = None
        for k in range(w.M):
            if k == i or not self.free(w, status, k, i):
                continue
            ev = self.evaluate(mdp, t, state, status, where, k, i, q[i])
            if ev is not None:
                ev["trigger"] = "held" if held else "guarded"
                if best is None or ev["gain"] > best["gain"]:
                    best = ev
        if best is None or not best["gain"] > cfg.feint_margin:
            return None
        best["taken"] = True
        k = best["decoy"]
        back = self.moves(w, i, self.cell_of(w, status, state, where, k),
                          self.cell_of(w, status, state, where, i))
        self.option[i] = dict(decoy=k, since=t, phase="to the decoy",
                              limit=2 * (best["transitions"] + int(min(back, w.N))) + 10)
        self.events.append((t + 1, i, "begun", k))
        if cfg.verbose:
            print(f"[pairings] feint: R{i} goes for the decoy (object {k}; its own object "
                  f"{'in the partner hands' if held else 'guarded'})", flush=True)
        return self._step(mdp, state, status, where, k, i, q[i])

    def _follow(self, mdp, t, state, status, where, q, i, planned, holds):
        """The option's phases: to the decoy and lift it; then hold it while the partner
        holds the robot's object; then, the object lying free, carry the decoy to it and
        swap the two (the decoy stays, the object comes into the robot's hands).  The
        decoy stays in hand until the swap: with hands full the partner's belief about the
        robot's object does not move (the posteriors' stages), so it is not read again.
        Ends with the object in the robot's hands (its planner delivers it), or where the
        option no longer makes sense."""
        w, opt, p = mdp.w, self.option[i], 1 - i
        k, mine = opt["decoy"], i
        why = None
        if t - opt["since"] > opt["limit"]:
            why = "too long"
        elif holds[i] == mine:
            why = "its object in hand"
        elif opt["phase"] == "to the decoy":
            if holds[i] == k:
                opt["phase"] = "holding the decoy"
                self.events.append((t + 1, i, "decoy in hand", k))
            elif holds[i] >= 0:
                why = "hands full"
            elif not self.free(w, status, k, i):
                why = "the decoy taken"
            elif (self.free(w, status, mine, i) and not self.guarded(w, state, status, where, i)):
                why = "its object free and nearer to it than to the partner"
            else:
                return self._step(mdp, state, status, where, k, i, q[i])
        elif holds[i] != k:
            why = "the decoy lost"
        if why is not None:
            self._end(mdp, t, i, why)
            return None
        # holding the decoy
        if not self.free(w, status, mine, i):   # the object still in the partner's hands:
            opt["phase"] = "holding the decoy"   # the planner's way, the decoy kept
            return WAIT if planned in (PLACE, GIVE, SWAP) else None
        if not self.cfg.swap_on_place:           # no swap: nothing to trade the decoy for
            self._end(mdp, t, i, "no swaps (Config.swap_on_place)")
            return None
        if opt["phase"] != "to its object":
            opt["phase"] = "to its object"
            self.events.append((t + 1, i, "to its object", mine))
        goal = self.cell_of(w, status, state, where, mine)
        if int(state[i]) == goal:
            return SWAP if np.isfinite(q[i][SWAP]) else WAIT
        a, _ = self.first_move(w, i, int(state[i]), goal, int(state[p]), q[i])
        return WAIT if a is None else a

    def _end(self, mdp, t, i, why) -> None:
        self.events.append((t + 1, i, "ended", why))
        self.option[i] = None
        self.ended[i] = t
        if self.cfg.verbose:
            print(f"[pairings] feint: R{i}'s option ended ({why})", flush=True)

    def _step(self, mdp, state, status, where, k, i, qi):
        """The option's action at the real state: lift the decoy where it lies, else a step
        toward it; None (the planner's own) if no legal step gets closer."""
        w = mdp.w
        goal = self.cell_of(w, status, state, where, k)
        if int(state[i]) == goal:
            return PICK if np.isfinite(qi[PICK]) else None
        a, _ = self.first_move(w, i, int(state[i]), goal, int(state[1 - i]), qi)
        return a

    # ---- the look-ahead --------------------------------------------------------------------
    def evaluate(self, mdp, t, state, status, where, k, i, qi) -> Optional[dict]:
        """The option "lift decoy k" at the real state, looked ahead with the partner's own
        belief update.  -> its record (also appended to ``log``), or None if the look-ahead
        does not get the decoy into the robot's hands."""
        w, cfg, p = mdp.w, self.cfg, 1 - i
        gamma = float(cfg.gamma)
        depth, row = mdp.root[i]
        bank = mdp.bank(depth)
        slot = mdp.slots[(i, depth, mdp.own_goal[i])]
        s = w.sid(state)
        v_now = _finite_max(qi)
        saved = list(mdp.about)
        belief = saved[i]
        p_now = self.mass(mdp, belief, i, k)
        goal = self.cell_of(w, status, state, where, k)
        _, to_go = self.first_move(w, i, int(state[i]), goal, int(state[p]), qi)
        if not np.isfinite(to_go) and int(state[i]) != goal:
            return None
        limit = int(min(to_go, 4 * w.N)) * 2 + int(cfg.pick_steps) + 4
        ret, disc, n, cur = 0.0, 1.0, 0, s
        steps = []
        try:
            while n < limit:
                st = w.unsid(cur)
                blk = int(st[2])
                if int(w.hold[blk, i]) == k:
                    break
                qc = mdp.q_values(mdp.root[i], cur, i)
                here = int(w.cell_id[tuple(w.object_cells(w.status[blk], st, where)[k])])
                if int(st[i]) == here:
                    a_i = PICK if np.isfinite(qc[PICK]) else None
                else:
                    a_i, _ = self.first_move(w, i, int(st[i]), here, int(st[p]), qc)
                if a_i is None:
                    return None
                pa = np.asarray(bank.P[slot.p_row, cur], dtype=np.float64)
                a_p = int(np.argmax(pa)) if pa.sum() > 0.0 else WAIT
                joint = (a_p, a_i) if i == 1 else (a_i, a_p)
                nxt = w.unsid(w.step(cur, joint))
                event = dict(done_now=(False, False), failed=(False, False),
                             handed_in_vain=(False, False),
                             held=(int(w.hold[blk, 0]), int(w.hold[blk, 1])))
                mdp.about = [belief if j == i else saved[j] for j in (0, 1)]
                belief = mdp.updated_belief(i, cur, joint, nxt, event)
                ret += disc * float(bank.R[row, cur, a_i])
                disc *= gamma
                steps.append((int(a_i), int(a_p)))
                cur = w.sid(nxt)
                n += 1
        finally:
            mdp.about = saved
        end = w.unsid(cur)
        if int(w.hold[int(end[2]), i]) != k:
            return None
        p_end = self.mass(mdp, belief, i, k)
        solved = abs(p_end - p_now) >= cfg.feint_min_shift
        v_end = (self.end_value(mdp, end, belief, i) if solved
                 else _finite_max(mdp.q_values(mdp.root[i], cur, i)))
        q_option = ret + disc * v_end
        rec = dict(transition=t + 1, robot=i, decoy=k, transitions=n, p_now=p_now,
                   p_end=p_end, v_now=v_now, option_return=ret, v_end=v_end,
                   q_option=q_option, gain=q_option - v_now, end_solved=bool(solved),
                   steps=steps, taken=False)
        self.log.append(rec)
        if cfg.verbose:
            print(f"[pairings] feint? R{i}: decoy object {k}, {n} transitions; the partner's "
                  f"belief in it {p_now:.3g} -> {p_end:.3g}; the option {q_option:.2f} vs the "
                  f"plan {v_now:.2f} (gain {q_option - v_now:+.2f}"
                  f"{', the end re-solved' if solved else ''})", flush=True)
        return rec

    def end_value(self, mdp, end, belief, i: int) -> float:
        """The robot's level-2 value at state ``end`` with the partner's belief about it
        ``belief`` (and the robot's own as it is): every model solved afresh there."""
        from .models import SocialMDP           # (no import cycle at module load)
        if self.evaluator is None:
            ev = SocialMDP(*self.args)
            ev.allocate()
            ev.solve_level0()
            self.evaluator = ev
        ev = self.evaluator
        ev.about = [belief.copy() if j == i else mdp.about[j].copy() for j in (0, 1)]
        ev.plan(end)
        if not self.started:
            ev.init_models()
            self.started = True
        for depth in (1, 2):
            ev.inputs(depth)
            ev.solve(depth)
        return _finite_max(ev.q_values(ev.root[i], ev.w.sid(end), i))

    def record(self) -> dict:
        return dict(robots=list(self.robots), evaluations=self.log, events=self.events)
