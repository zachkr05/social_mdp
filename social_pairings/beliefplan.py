"""Planning over the partner's belief (``Config.belief_planning``, 2026-11-05).

The user, of the answer to "why does the coercer never feint" -- "The planner needs two
changes: reward the coercer for its real task, and simulate how each candidate action changes
the other agent's beliefs and subsequent actions" -- "Could you please implement this?", then
"keep looping iterating and producing videos until it gives the exact behavior we want in the
scene hinder vs coercer in the two rooms scene with tau = 0.2".

Robot r (the planner; a level-2 robot whose social goal is one of
``Config.belief_planning_goals``) chooses by

    Q(s, m, u, a_r) = R(s, a_r, a_p) + gamma V(s', F(m; s, a_r, a_p, s'), u'),
    a_p = the partner's choice at (s, m, u),   V(s, m, u) = max_a Q(s, m, u, a),

with m the partner's belief about robot r -- its physical posterior (object x destination),
its social posterior and its idle mode, all of Eqs. 5-6 that it holds -- F the partner's OWN
update of it (``SocialMDP.updated_belief``; :func:`beliefkernels.belief_update`, the same rule
over every state, checked against it in the tests), u what would undo the partner's last
action (its inertia), s' the true transition.  The timing is the robots': the partner chooses
a_p with m, before it sees a_r; it updates to F(m; ...) after; the next choice uses that.

* **The partner, predicted.**  By its own model: its root re-solved at belief m (one model,
  on scratch arrays: :meth:`BeliefPlanner._partner_batch`), greedy with
  ``util.argmax_action``'s tie-break and its inertia (``Config.inertia``) -- the rule it acts
  by -- under a hypothesis about its goals (:class:`_Hyp`).  Robot r reconstructs its belief
  from the history both see (the partner's prior, update rule and observations are those of
  the model: ``SocialMDP.about[r]`` is exactly what the partner holds).
* **Whom robot r takes its partner to be** (``Config.belief_partner``, the user of
  2026-11-05 evening: "everything is inferred and nothing is I guess truly known. Red should
  infer the belief from green then apply to formula to manipulate it").  "inferred" (the
  default): the hypotheses (physical goal, social goal) of robot r's own posteriors about
  the partner (``SocialMDP.about[p]``, Eqs. 5-6) that hold most of their mass
  (:meth:`BeliefPlanner._hypotheses`) -- the plan above made under each, the dynamics those
  of a world whose partner has that goal, and the action chosen by the posterior-weighted Q
  of the plans (QMDP: Littman, Cassandra & Kaelbling 1995 -- the partner's type as if known
  from the next step on, robot r's own posterior held within a plan as Eq. 4 holds it).
  "oracle": the partner's true goals and the true dynamics (the rule until then), said so
  in the record.  Either way the partner's level, weight, tau, update rule and greedy
  choice are the model's, as every robot is given them.
* **What robot r is paid.**  ``SocialMDP.modeled_reward`` with the one hypothesis about the
  partner of each plan (:func:`beliefkernels.joint_rewards`, :func:`beliefkernels.nested_row`);
  under ``Config.objective="task"`` the partner's estimate of robot r's goal reward inside
  R_j is its true goal reward; under "complete" the bonus for the goal actually done.
* **The beliefs.**  Belief points: the partner's belief now (point 0, exact), the beliefs it
  would form if robot r acted by each of the hypotheses it holds about it (its level-1
  models -- "probes"), and those along the planned way (``Config.belief_refinements``
  rounds) -- a new point where a belief lies farther than ``Config.belief_spacing`` from all
  others in sqrt-coordinates (Hellinger), at most ``Config.belief_points``.  The belief after
  a transition is computed exactly from its point and then located among the points:
  between two of them, with convex weights (``Config.belief_interpolation="segment"``; an
  averager, so the backup stays a contraction), or at the nearest.
* **The horizon.**  ``Config.belief_horizon`` transitions by backward induction over (k, u,
  s), then the value with the partner's belief held where it is (its tail: robot r's best
  answer to the partner's greedy choice at the point, :func:`beliefkernels.tail_values`;
  inertia left out there) -- or, ``Config.belief_tail="cap"``, nothing after the rollout's
  step cap where that comes first.
* **"frozen"** (``Config.belief_planning``): the same planner with the partner's belief held
  as it is now in every hypothetical transition -- the comparison that isolates the
  belief's dynamics.  Every decision also records that Q, and the advisor's
  Delta_belief = gamma [V(s', m', u') - V(s', m, u')] of each action (the belief update of
  this one transition allowed or not; the rest of the plan the same).

No reward for deceiving (kappa = 0): a decoy is worth what it brings the real task.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from . import goals as G
from .beliefkernels import (NO_UNDO, NU, backup, belief_update, choice_tables, embed,
                            joint_rewards, legal_mask, nested_row, project, projections,
                            q_point, q_table, tail_values, undo_code)
from .constants import ACTION_NAMES, N_ACTIONS
from .kernels import inputs_typed
from .models import PartnerBelief
from .util import block_items, log_softmax

__all__ = ["BeliefPlanner"]

# the likelihood table in double precision up to this size, in single precision above it
LL_F64_BYTES = 1.5 * 2 ** 30
ORACLE = ("the partner's own model, re-solved at each belief: its social goal, weight, level "
          "and update rule known to the planner (an oracle); its belief reconstructed from "
          "the history both robots see")
INFERRED = ("the partner's own model under each hypothesis about its goals that the robot's "
            "posteriors about it hold (the most likely, Config.belief_partner_mass / _max), "
            "re-solved at each belief; the plans weighted by those posteriors (QMDP); its "
            "level, weight, tau, update rule and greedy choice the model's; its belief "
            "reconstructed from the history both robots see")


class _OneModel:
    """Row ``row`` of a bank n times over, renumbered 0 .. n - 1, on scratch arrays P, pd,
    ho: what ``SocialMDP._solve`` reads when the partner's root is re-solved at n belief
    points in one batch (each copy its own row of P, pd and ho).  ``over``: agent, var,
    svar, social and zero in place of the row's (the partner's root under a hypothesis
    about its goals, :class:`_Hyp`)."""

    def __init__(self, bank, row: int, P, pd, ho, n: int = 1, over: Optional[dict] = None
                 ) -> None:
        self.P, self.pd, self.ho = P, pd, ho
        self.p_of = np.arange(n, dtype=np.int64)
        for name in ("agent", "var", "svar", "social", "zero"):
            value = (getattr(bank, name)[[row]] if over is None
                     else np.array([over[name]], dtype=np.int64))
            setattr(self, name, np.ascontiguousarray(np.repeat(value, n)))


@dataclass(eq=False)
class _Hyp:
    """A hypothesis about the partner's goals that robot r plans against
    (``Config.belief_partner``): its physical goal ``goal`` (an index of the partner's
    ``pg_obj`` / ``pg_dst``) and social goal ``social`` (``goals.SOCIAL_GOALS``), its weight
    in the decision (robot r's posterior mass, renormalised over the hypotheses kept) and
    the (goal, social goal, mass) it stands for (a goal-less hinderer's destinations are
    one hypothesis); and what a plan needs under it: ``tab`` the world's hands with a
    partner of that goal (``ObjectWorld._table``; the true table under the oracle),
    ``events`` the partner's acts for robot r that fail or are in vain there, and the
    partner's root under it -- ``over`` (agent, var, svar, social, zero), ``slot`` (the
    root's slot with ``group`` the hypothesis' physical goal), ``task_nested`` (the
    partner's estimate of its own goal reward inside its R_j its true goal's,
    ``Config.objective="task"`` for its social goal).  ``oracle``: the partner's true
    goals, its root as the rollout solved it."""
    goal: int
    social: int
    weight: float
    members: list
    tab: np.ndarray
    events: tuple
    over: Optional[dict] = None
    slot: object = None
    task_nested: bool = False
    oracle: bool = False

    @property
    def key(self) -> tuple:
        return (int(self.goal), int(self.social))


def _same_goal(tab: np.ndarray, done: np.ndarray, terminal: int) -> np.ndarray:
    """``tab`` (B, nEv) for a world in which the partner's goal is the robot's own (its
    object on its destination): whoever sets it down there finishes both goals -- every
    successor with one goal done is the terminal block (both done).  The hands rule sets one
    done flag per set-down (the true world's goal objects are two)."""
    out = np.array(tab, copy=True)
    one = done[:, 0] != done[:, 1]                                         # (B,)
    out[one[out]] = terminal
    return out


def _first(q: np.ndarray, k: int) -> list:
    """The k best actions of a Q (finite only), best first."""
    ok = np.flatnonzero(np.isfinite(q))
    return [int(a) for a in ok[np.argsort(-q[ok], kind="stable")][:k]]


class BeliefPlanner:
    """The belief planner of the robots it applies to, for one rollout of ``n_steps`` at
    most (``Config.belief_tail="cap"`` reads it)."""

    def __init__(self, mdp, n_steps: int) -> None:
        cfg, w = mdp.cfg, mdp.w
        self.cfg, self.mode, self.n_steps = cfg, cfg.belief_planning, int(n_steps)
        names = {G.parse_goal(g) for g in cfg.belief_planning_goals}
        self.robots = tuple(i for i in (0, 1)
                            if mdp.levels[i] == 2 and int(mdp.goals[i]) in names)
        for r in self.robots:
            p = 1 - r
            if mdp.levels[p] < 1:
                raise ValueError("belief_planning: the partner of a planning robot must model "
                                 "it (level 1 or 2); a level-0 partner believes nothing")
            if self.mode == "updated" and mdp.levels[p] != 2:
                raise NotImplementedError('belief_planning="updated" reads a level-2 partner\'s '
                                          'update (a level-1 one reads the level-0 policies)')
        self.gamma = float(cfg.gamma)
        self.margin = float(cfg.inertia_margin)
        self.legal, self.events = {}, {}
        for r in self.robots:
            depth, row = mdp.root[r]
            sc = int(mdp.bank(depth).social[row])
            out = np.zeros((w.S, N_ACTIONS), dtype=np.bool_)
            legal_mask(r, sc, w.N, mdp.valid_by, w.busy, mdp.useok, w.uctx, w.state_ok, out)
            self.legal[r] = out
            self.events[r] = self._event_tables(w, 1 - r)
        self.decisions: list = []
        self.clock: dict = {}                 # seconds by stage, over the rollout
        # per (robot, hypothesis key): the last decision's points (warm starts)
        self.keep: dict = {}
        self._LL = None                       # (S, 10, PG, T) buffer
        self.likelihood_dtype = None
        self._base = None                     # (S, 10, 10) buffer
        self._hyp_cache: dict = {}            # (robot, hypothesis key) -> its tables

    # ---- static tables -----------------------------------------------------------------------
    @staticmethod
    def _event_tables(w, p: int, gs=None, own=(0, 1), true=None, same_goal: bool = False):
        """(failed, in vain) (B, nEv) uint8: the partner p's set-down for its partner that did
        nothing but would have finished the partner's goal elsewhere, and its hand-over the
        partner did not take -- ``ObjectWorld.real_step``'s ``failed[p]`` and
        ``handed_in_vain[p]``, by block and event (its "anywhere" run, as a table).  ``gs``
        (2,): each robot's destination site, ``own`` (2,): each robot's goal object, ``true``
        the table of that world -- the world's own (its true destinations, objects (0, 1)
        and ``true_tab``) by default, a world whose partner has the goals of a hypothesis
        otherwise (:meth:`_hyp`; ``same_goal``: its goal the robot's own, :func:`_same_goal`)."""
        r = 1 - p
        if gs is None:
            gs = [int(w.site_of[d]) for d in w.true_dest]
        anywhere = [0, 0]
        anywhere[p], anywhere[r] = -2, int(gs[p])
        acc = [int(own[0]), int(own[1])]
        acc[r] = -2
        own = tuple(int(x) for x in own)
        anyw = w._table(tuple(int(g) for g in gs), tuple(anywhere), own, (True, True), own,
                        tuple(acc))
        if same_goal:
            anyw = _same_goal(anyw, w.done, w.T)
        anyw = anyw.astype(np.int64)
        true = (w.true_tab if true is None else true).astype(np.int64)
        differ = anyw != true
        done_differ = (w.done[anyw] != w.done[true]).any(axis=-1)
        return ((differ & done_differ).astype(np.uint8),
                (differ & ~done_differ).astype(np.uint8))

    # ---- whom the partner is taken to be (Config.belief_partner) -----------------------------
    def _hyp(self, mdp, r: int, goal: int, social: int, weight: float, members: list,
             oracle: bool) -> _Hyp:
        """The hypothesis (partner's physical goal ``goal``, social goal ``social``) with what
        a plan needs under it -- the tables cached per (robot, hypothesis)."""
        w, cfg = mdp.w, mdp.cfg
        p = 1 - r
        K = mdp.PG
        key = (r, int(goal), int(social), bool(oracle))
        if key not in self._hyp_cache:
            if oracle:
                self._hyp_cache[key] = dict(tab=w.true_tab, events=self.events[r], over=None,
                                            slot=None, task_nested=False)
            else:
                obj, dst = int(mdp.pg_obj[p][goal]), int(mdp.pg_dst[p][goal])
                gs, ds, own = [0, 0], [0, 0], [0, 0]
                gs[r], gs[p] = int(w.site_of[w.true_dest[r]]), int(w.dest_site[dst])
                ds[r], ds[p] = gs[p], gs[r]          # a set-down for the partner counts there
                own[r], own[p] = r, obj              # each robot's goal object
                tab = w._table(tuple(gs), tuple(ds), tuple(own), (True, True), tuple(own),
                               tuple(own))
                # the partner's goal the robot's own (its object on its destination): one
                # set-down finishes both
                same = obj == r and gs[p] == gs[r]
                if same:
                    tab = _same_goal(tab, w.done, w.T)
                zero = bool(mdp._no_term(p, int(social)))
                self._hyp_cache[key] = dict(
                    tab=tab, events=self._event_tables(w, p, gs, own, tab, same),
                    over=dict(agent=p, var=p * K + int(goal), svar=2 * K + p * K + int(goal),
                              social=int(not zero), zero=int(zero)),
                    slot=replace(mdp.root_slot(p), group=int(goal)),
                    task_nested=(cfg.objective == "task" and int(social) in
                                 {G.parse_goal(g) for g in cfg.objective_goals}))
        c = self._hyp_cache[key]
        return _Hyp(int(goal), int(social), float(weight), members or [], c["tab"], c["events"],
                    c["over"], c["slot"], c["task_nested"], bool(oracle))

    def _oracle(self, mdp, r: int) -> _Hyp:
        """The partner's true goals, the true world (``Config.belief_partner="oracle"``)."""
        g, t = int(mdp.own_goal[1 - r]), int(mdp.goals[1 - r])
        return self._hyp(mdp, r, g, t, 1.0, [(g, t, 1.0)], True)

    def _hypotheses(self, mdp, r: int) -> tuple:
        """-> (the hypotheses robot r plans against, the posterior mass they cover before
        renormalising).  ``Config.belief_partner="oracle"``: the partner's true goals.
        "inferred": robot r's posteriors about its partner (Eq. 4's planning weights,
        b_g(g) b_xi(xi)), the most likely hypotheses first until all but
        ``belief_partner_mass`` is covered or ``belief_partner_max`` are kept; a hinderer
        without a goal of its own sets nothing down for itself and is one hypothesis per
        object, its most likely destination standing for the others."""
        cfg = mdp.cfg
        p = 1 - r
        if cfg.belief_partner == "oracle":
            return [self._oracle(mdp, r)], 1.0
        about = mdp.about[p]
        phys = about.phys()
        soc = about.social()
        types = [int(t) for t in mdp.hyp_goals[r]]
        if soc is None:                      # (a partner below level 2: never planned against)
            soc = np.ones(1)
            types = [int(mdp.goals[p])]
        groups: dict = {}
        for ti, t in enumerate(types):
            spoiler = t == G.HINDER and not cfg.hinder_own_goal
            for g in range(mdp.PG):
                mass = float(phys[g] * soc[ti])
                if not mass > 0.0:
                    continue
                gkey = ("object", int(mdp.pg_obj[p][g]), t) if spoiler else (g, t)
                groups.setdefault(gkey, []).append((g, t, mass))
        ranked = []
        for members in groups.values():
            total = sum(m for _, _, m in members)
            g_rep = max(members, key=lambda x: (x[2], -x[0]))[0]
            ranked.append((total, g_rep, members[0][1], members))
        ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
        kept, covered = [], 0.0
        for total, g, t, members in ranked:
            kept.append((total, g, t, members))
            covered += total
            if (covered >= 1.0 - cfg.belief_partner_mass
                    or len(kept) >= cfg.belief_partner_max):
                break
        norm = sum(x[0] for x in kept)
        return ([self._hyp(mdp, r, g, t, total / norm, members, False)
                 for total, g, t, members in kept], float(covered))

    # ---- the partner's model at a belief ---------------------------------------------------------
    def _q(self, mdp, bank, row: int, R, P, V, m: int, g: int, pd, ho, ok, lo, hi,
           over: Optional[dict] = None):
        """The Q of a model at every state of the items (``SocialMDP.q_values``); ``over``:
        the model's agent, var, svar, social and zero in place of the row's."""
        w = mdp.w
        if over is None:
            o, sc, fx = int(bank.agent[row]), int(bank.social[row]), bool(bank.zero[row])
            var, svar = int(bank.var[row]), int(bank.svar[row])
        else:
            o, sc, fx = int(over["agent"]), int(over["social"]), bool(over["zero"])
            var, svar = int(over["var"]), int(over["svar"])
        out = np.full((w.S, N_ACTIONS), -np.inf)
        q_table(R, P, V, m, g, o, sc, fx, mdp.tabs[var], mdp.tabs[svar], lo, hi, w.N, w.psucc,
                w.psucc_T, w.pev, w.pev_T, mdp.valid_by, w.busy, mdp.useok, w.uctx, w.site_of,
                w.goal_at_site, pd, ho, mdp.infer_object, mdp.done, w.vdone, w.hold, w.gone,
                ok, self.gamma, out)
        return out

    def _partner_tables(self, mdp, r: int, belief: Optional[tuple], ok, lo, hi, warm,
                        H: Optional[_Hyp] = None):
        """The partner's Q at every state with its belief about robot r ``belief`` = (lp, ls,
        q), under hypothesis H -- its root re-solved there on scratch arrays (None: under
        the oracle, as the rollout solved it at the belief it holds) -> (Q (S, 10), V (S,)
        for a warm start, solver stats).  H None: the oracle's (the partner's true goals)."""
        if H is None:
            H = self._oracle(mdp, r)
        if belief is None and H.oracle:
            p = 1 - r
            depth, row = mdp.root[p]
            bank = mdp.bank(depth)
            Q = self._q(mdp, bank, row, bank.R, bank.P, bank.V, row, int(bank.p_of[row]),
                        bank.pd, bank.ho, ok, lo, hi)
            return Q, bank.V[row].copy(), None
        if belief is None:
            m0 = mdp.about[r]
            belief = (m0.log_phys, m0.log_social, float(m0.idle))
        return self._partner_batch(mdp, r, [belief], ok, lo, hi, [warm], H)[0]

    def _partner_batch(self, mdp, r: int, beliefs: list, ok, lo, hi, warm: list,
                       H: Optional[_Hyp] = None) -> list:
        """:meth:`_partner_tables` at several beliefs, the partner's root (under hypothesis
        H; None: the oracle's) re-solved at all of them in one call of the solver -> [(Q (S,
        10), V (S,), solver stats)]."""
        if H is None:
            H = self._oracle(mdp, r)
        w = mdp.w
        p = 1 - r
        depth, row = mdp.root[p]
        bank = mdp.bank(depth)
        if H.oracle:
            slot, keys, flags, over = mdp.root_slot(p), [bank.key[row]], None, None
            blocks = mdp.solved_blocks(bank, row)
        else:
            slot, keys, over = H.slot, [int(H.social)], H.over
            flags = ([over["social"]], [over["zero"]])
            full = mdp.reach_of(p, int(H.goal), int(bank.block[row]))
            blocks = full[w.done[full, p] == 0] if over["zero"] else full
        n, S, A = len(beliefs), w.S, N_ACTIONS
        P = np.zeros((n, S, A))
        R = np.zeros((n, S, A), dtype=np.float32)
        pd = np.zeros((n, w.B, w.K, w.N, A))
        ho = np.zeros((n, S, A)) if mdp.infer_object else np.zeros((n, 1, A))
        V = np.empty((n, S))
        rows = np.array([row], dtype=np.int64)
        t0 = time.perf_counter()
        for i, (lp, ls, q) in enumerate(beliefs):
            about = [None, None]
            about[p] = mdp.about[p]
            about[r] = PartnerBelief(lp.copy(), None if ls is None else ls.copy(), float(q))
            nested = None
            if not H.oracle:                 # the partner's estimate of its own goal reward
                if H.task_nested:
                    nested = np.zeros(mdp.PG)
                    nested[int(H.goal)] = 1.0
                else:
                    nested = about[p].phys()
            inputs_typed(*mdp._input_args(depth, slot, rows, keys,
                                          out=(P, i, R[i:i + 1], pd, ho), about=about,
                                          flags=flags, nested=nested))
            V[i] = bank.V[row] if warm[i] is None else warm[i]
        t1 = time.perf_counter()
        st = mdp._solve(V, R, _OneModel(bank, row, P, pd, ho, n, over),
                        np.arange(n, dtype=np.int64), [blocks] * n)
        t2 = time.perf_counter()
        out = []
        for i in range(n):
            Q = self._q(mdp, bank, row, R, P, V, i, i, pd, ho, ok, lo, hi, over)
            out.append((Q, V[i].copy(), st[i]))
        self.clock["partner_inputs"] = self.clock.get("partner_inputs", 0.0) + t1 - t0
        self.clock["partner_solve"] = self.clock.get("partner_solve", 0.0) + t2 - t1
        self.clock["partner_q"] = self.clock.get("partner_q", 0.0) + time.perf_counter() - t2
        return out

    # ---- the likelihoods of the partner's update -------------------------------------------------
    def _likelihoods(self, mdp, r: int, ok, lo, hi) -> np.ndarray:
        """LL (S, 10, PG, T): log psi of robot r's level-1 hypothesis models (the partner's
        hypotheses about it) at every state, at the likelihood's tau, after the lapse
        (``SocialMDP.hypothesis_log_policies``); -inf for the physical goals the partner's
        posterior has ruled out (their models are not solved).  In double precision, as
        ``updated_belief`` computes it, up to ``LL_F64_BYTES``; above that (the house scenes:
        480,557 states x 16 physical goals x 7 social goals) in single precision -- the
        likelihoods then good to some 1e-7 of themselves (``likelihood_dtype``, recorded)."""
        w, cfg = mdp.w, mdp.cfg
        p = 1 - r
        T = len(mdp.hyp_goals[p])
        shape = (w.S, N_ACTIONS, mdp.PG, T)
        dtype = np.float64 if np.prod(shape) * 8 <= LL_F64_BYTES else np.float32
        if self._LL is None or self._LL.shape != shape or self._LL.dtype != dtype:
            self._LL = None                  # (the old table freed before the new one)
            self._LL = np.empty(shape, dtype=dtype)
        self.likelihood_dtype = np.dtype(dtype).name
        LL = self._LL
        LL.fill(-np.inf)
        tau = float(mdp.inference_taus[p])
        e = float(cfg.belief_lapse)
        L1 = mdp.L1
        first: dict = {}                     # row -> the (g, t) it was written at first
        prior = mdp.about[r].log_phys
        for g in range(mdp.PG):
            if not np.isfinite(prior[g]):
                continue
            for t in range(T):
                row = mdp.hypothesis_row(r, g, t)
                if row in first:             # a model shared by two hypotheses: copy it
                    LL[:, :, g, t] = LL[:, :, first[row][0], first[row][1]]
                    continue
                first[row] = (g, t)
                Q = self._q(mdp, L1, row, L1.R, L1.P, L1.V, row, int(L1.p_of[row]), L1.pd,
                            L1.ho, ok, lo, hi)
                with np.errstate(invalid="ignore"):
                    lp = log_softmax(Q / tau)
                if e > 0.0:
                    allowed = np.isfinite(lp)
                    n = np.maximum(allowed.sum(axis=1, keepdims=True), 1)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        lp = np.where(allowed, np.logaddexp(np.log1p(-e) + lp, np.log(e / n)),
                                      -np.inf)
                LL[:, :, g, t] = lp
        return LL

    # ---- one belief transition, at one state ---------------------------------------------------
    def _ctx(self, mdp, r: int, LL, H: _Hyp):
        """The arguments of :func:`beliefkernels.belief_update` after (lp, ls, q, s, a_r, a_p),
        in the world of hypothesis H."""
        w, cfg = mdp.w, mdp.cfg
        failed, vain = H.events
        has_social = mdp.about[r].log_social is not None
        return (r, w.N, LL, mdp.log_idle[r], w.busy, w.hold, w.done,
                np.ascontiguousarray(mdp.pg_obj[r]), np.ascontiguousarray(mdp.pg_dst[r]),
                w.M, w.K, np.ascontiguousarray(w.dest), int(w.true_dest[r]), w.iscand,
                w.site_of, failed, vain, H.tab, w.pev, w.psucc, has_social,
                not mdp.known_social, bool(cfg.belief_factored), bool(cfg.belief_stages),
                float(cfg.likelihood_power), float(cfg.hand_hazard), float(cfg.goal_hazard),
                float(cfg.social_hazard), bool(cfg.idle_hypothesis), float(cfg.idle_hazard),
                bool(cfg.condition_on_placement))

    def step_belief(self, mdp, r: int, LL, belief: tuple, s: int, a_r: int, a_p: int,
                    H: Optional[_Hyp] = None) -> tuple:
        """The partner's belief about robot r after (a_r, a_p) at s, from ``belief`` = (lp,
        ls, q), in the world of hypothesis H (None: the true one):
        :func:`beliefkernels.belief_update` -> (lp, ls, q)."""
        if H is None:
            H = self._oracle(mdp, r)
        lp, ls, q = belief
        PG = lp.size
        ls_in = ls if ls is not None else np.zeros(1)
        T = ls_in.size
        lp_out, ls_out = np.empty(PG), np.empty(T)
        W = max(mdp.w.M, mdp.w.K, PG)
        q1 = belief_update(lp, ls_in, float(q), int(s), int(a_r), int(a_p),
                           *self._ctx(mdp, r, LL, H), lp_out, ls_out, np.empty((8, PG)),
                           np.empty((PG, T)), np.empty((3, T)), np.empty((6, W)),
                           np.empty(PG, dtype=np.bool_))
        return lp_out, (ls_out if ls is not None else None), float(q1)

    # ---- the decision --------------------------------------------------------------------------
    def decide(self, mdp, t: int, state, undo) -> dict:
        """{robot: its Q at ``state``} for the planning robots whose goal is not done yet
        (the rollout then chooses by it, with its inertia)."""
        out = {}
        if t < max(int(self.cfg.belief_planning_from), len(self.cfg.forced_actions)):
            return out                       # the history before (Eq. 4's planner, or forced)
        for r in self.robots:
            q = self._decide(mdp, t, state, undo, r)
            if q is not None:
                out[r] = q
        return out

    def _decide(self, mdp, t: int, state, undo, r: int) -> Optional[np.ndarray]:
        w, cfg = mdp.w, mdp.cfg
        p = 1 - r
        c0 = int(state[2])
        if w.done[c0, r]:
            return None
        tic = time.perf_counter()
        timing = {}
        NN = w.NN
        s0 = int(w.sid(state))
        blocks = mdp.reach_of(r, mdp.own_goal[r], c0)
        ok = np.zeros_like(w.state_ok)
        ok[blocks] = w.state_ok[blocks]
        _, lo, hi = block_items(NN, blocks, 0)
        legal = self.legal[r]
        u0 = NO_UNDO if (undo[p] is None or self.margin <= 0.0) else int(undo[p])
        m0 = mdp.about[r]
        point0 = (m0.log_phys.copy(), None if m0.log_social is None else m0.log_social.copy(),
                  float(m0.idle))
        # ---- whom the partner is taken to be: robot r's posteriors about it (the oracle: its
        # true goals) -- one plan per hypothesis, the decision their weighted Q
        tic1 = time.perf_counter()
        hyps, covered = self._hypotheses(mdp, r)
        timing["hypotheses"] = time.perf_counter() - tic1
        LL = None
        if self.mode == "updated":
            tic1 = time.perf_counter()
            LL = self._likelihoods(mdp, r, ok, lo, hi)
            timing["likelihoods"] = time.perf_counter() - tic1
        H_steps, zero_tail = self._horizon(t)
        plans = []
        for n, H in enumerate(hyps):
            tic1 = time.perf_counter()
            plans.append(self._plan(mdp, r, H, s0, u0, point0, LL, ok, lo, hi, legal, H_steps,
                                    zero_tail, verbose=cfg.verbose and n == 0))
            timing[f"hypothesis_{n}"] = time.perf_counter() - tic1
        weights = np.array([H.weight for H in hyps])

        def mix(name):
            qs = [x[name] for x in plans]
            if any(q is None for q in qs):
                return None
            stack = np.stack(qs)
            fin = np.isfinite(stack).all(axis=0)
            out = np.full(N_ACTIONS, -np.inf)
            out[fin] = weights @ stack[:, fin]
            return out

        q, qf, qh = mix("q"), mix("q_frozen"), mix("q_hold")
        prediction = np.zeros(N_ACTIONS)
        for H, x in zip(hyps, plans):
            prediction[x["partner_predicted"]] += H.weight
        a_pred = int(np.argmax(prediction))
        top = plans[0]
        rec = dict(transition=t + 1, robot=r, mode=self.mode,
                   response=ORACLE if hyps[0].oracle else INFERRED,
                   likelihood_dtype=self.likelihood_dtype,
                   objective=self._objective(mdp, r), horizon=int(cfg.belief_horizon),
                   partner_undo=None if u0 == NO_UNDO else int(u0),
                   partner_predicted=a_pred,
                   partner_prediction={ACTION_NAMES[a]: float(prediction[a])
                                       for a in np.flatnonzero(prediction > 0).tolist()},
                   covered=covered, rounds=top["rounds"], q_frozen=qf)
        rec["hypotheses"] = [dict(object=int(mdp.pg_obj[p][H.goal]),
                                  destination=int(mdp.pg_dst[p][H.goal]),
                                  social=G.goal_name(H.social), weight=float(H.weight),
                                  members=len(H.members),
                                  mass=float(sum(m for _, _, m in H.members)),
                                  q=x["q"], q_frozen=x["q_frozen"],
                                  partner_predicted=int(x["partner_predicted"]),
                                  points=int(x["points"]))
                             for H, x in zip(hyps, plans)]
        if self.mode != "frozen":
            rec["q_hold"] = qh
            with np.errstate(invalid="ignore"):
                rec["delta_belief"] = np.where(np.isfinite(q), q - qh, np.nan)
            rec["plan"] = top["plan"]
            rec["after"] = top["after"]
        rec["q"] = q
        fin = np.isfinite(q)
        order = _first(q, 2)
        rec["best_action"] = order[0] if order else None
        rec["runner_up"] = order[1] if len(order) > 1 else None
        rec["gap"] = float(q[order[0]] - q[order[1]]) if len(order) > 1 else None
        rec["points"] = top["points"]
        rec["point_beliefs"] = top["point_beliefs"]
        rec["now"] = self._summary(mdp, r, point0)
        rec["partner_alternatives"] = top["partner_alternatives"]
        timing["total"] = time.perf_counter() - tic
        rec["timing"] = timing
        rec["finite"] = int(fin.sum())
        self.decisions.append(rec)
        if cfg.verbose:
            names = ACTION_NAMES
            best = ", ".join(f"{names[a]} {q[a]:.3f}" for a in _first(q, 3))
            if hyps[0].oracle:
                who = "oracle"
            else:
                who = "; ".join(f"{G.goal_name(H.social)} O{int(mdp.pg_obj[p][H.goal]) + 1}"
                                f"/G{int(mdp.pg_dst[p][H.goal])} {H.weight:.2f}" for H in hyps)
                who = f"{len(hyps)} hypotheses ({covered:.2f} of the posterior): {who}"
            print(f"[pairings] belief plan R{r} t={t + 1}: {best}; frozen "
                  f"{names[_first(qf, 1)[0]]}; partner {names[a_pred]} "
                  f"({prediction[a_pred]:.2f}); {who}; {timing['total']:.1f} s", flush=True)
        return q

    def _plan(self, mdp, r: int, H: _Hyp, s0: int, u0: int, point0: tuple, LL, ok, lo, hi,
              legal, H_steps: int, zero_tail: bool, verbose: bool = False) -> dict:
        """The plan of one decision under hypothesis H about the partner's goals: Q updated,
        held for this one transition, and frozen; the partner's predicted action; the planned
        way and the points.  (The weighted sum over the hypotheses is the decision,
        :meth:`_decide`.)"""
        cfg, w = mdp.cfg, mdp.w
        S = w.S
        # ---- the partner at point 0 (the oracle: as the rollout solved it)
        prev = self.keep.get((r, H.key))
        warm0 = None
        if prev is not None and not H.oracle:
            d = ((prev["Z"] - _Points.embed_point(mdp, r, point0)) ** 2).sum(-1)
            warm0 = prev["Vp"][int(np.argmin(d))]
        Q0, Vp0, _ = self._partner_tables(mdp, r, None, ok, lo, hi, warm0, H)
        b0, a0 = np.empty(S, dtype=np.int8), np.empty(S, dtype=np.int8)
        choice_tables(Q0, self.margin, b0, a0)
        a_p0 = int(a0[s0] if u0 == b0[s0] else b0[s0])
        # ---- what robot r is paid, under H
        base, coef = self._rewards(mdp, r, legal, ok, lo, hi, H)
        # ---- the points
        pts = _Points(self, mdp, r, ok, lo, hi, base, coef, legal, H)
        pts.add(point0, tables=(b0, a0, Vp0))
        # the frozen plan: point 0 alone, the belief held (the decision's Q in "frozen" mode;
        # a diagnostic otherwise)
        Vf1 = pts.backup(H_steps, zero_tail, frozen=True, only=1)
        qf = pts.q(Vf1, 0, s0, u0, a_p0, hold=True)
        out = dict(q_frozen=qf, partner_predicted=a_p0, rounds=[], plan=None, after=None,
                   q_hold=None)
        if self.mode == "frozen":
            q = qf
        else:
            probes = self._probes(mdp, r, LL, pts, s0, u0, point0, H_steps, H)
            pts.extend(probes, budget=max(1, cfg.belief_points - max(2, cfg.belief_points // 4)))
            q = qh = V1 = None
            for rnd in range(cfg.belief_refinements + 1):
                t0 = time.perf_counter()
                pts.solve_new()
                t1 = time.perf_counter()
                pts.project(LL)
                t2 = time.perf_counter()
                V1 = pts.backup(H_steps, zero_tail, frozen=False)
                q = pts.q(V1, 0, s0, u0, a_p0, hold=False)
                qh = pts.q(V1, 0, s0, u0, a_p0, hold=True)
                t3 = time.perf_counter()
                if verbose:
                    print(f"[pairings]   round {rnd}: {pts.n} points ({t1 - t0:.1f} s"
                          f" solving), projections {t2 - t1:.1f} s, backup "
                          f"{t3 - t2:.1f} s", flush=True)
                plans = [self._simulate(mdp, r, LL, pts, V1, s0, u0, point0, H_steps, first, H)
                         for first in _first(q, 3)]
                residual = [x["residual"] for plan in plans for x in plan]
                out["rounds"].append(dict(points=pts.n,
                                          residual_max=float(max(residual, default=0.0)),
                                          residual_mean=float(np.mean(residual)) if residual
                                          else 0.0))
                if rnd == cfg.belief_refinements:
                    break
                cands = [x["belief"] for plan in plans for x in plan
                         if x["residual"] > cfg.belief_spacing]
                if not cands or not pts.extend(cands, budget=cfg.belief_points):
                    break
            out["q_hold"] = qh
            out["plan"] = [{k: v for k, v in x.items() if k != "belief"} for x in plans[0]]
            out["after"] = {ACTION_NAMES[a]: self._summary(mdp, r, pts.after(LL, s0, a, a_p0))
                            for a in np.flatnonzero(legal[s0]).tolist()}
        out["q"] = q
        out["points"] = pts.n
        out["point_beliefs"] = [self._summary(mdp, r, b) for b in pts.beliefs]
        out["partner_alternatives"] = dict(best=int(b0[s0]), alt=int(a0[s0]))
        self.keep[(r, H.key)] = pts.keep()
        return out

    def observe(self, t: int, act) -> None:
        """After the real transition: what the planning robots chose and what their partners
        did (the prediction's check)."""
        for rec in self.decisions[::-1]:
            if rec["transition"] != t + 1:
                break
            r = rec["robot"]
            rec["chosen"] = int(act[r])
            rec["partner_actual"] = int(act[1 - r])
            rec["partner_as_predicted"] = int(act[1 - r]) == rec["partner_predicted"]

    def record(self) -> dict:
        cfg = self.cfg
        return dict(robots=list(self.robots), mode=self.mode,
                    response=ORACLE if cfg.belief_partner == "oracle" else INFERRED,
                    partner=cfg.belief_partner, partner_mass=cfg.belief_partner_mass,
                    partner_max=cfg.belief_partner_max,
                    objective=cfg.objective, horizon=cfg.belief_horizon,
                    points=cfg.belief_points, spacing=cfg.belief_spacing,
                    refinements=cfg.belief_refinements, interpolation=cfg.belief_interpolation,
                    tail=cfg.belief_tail, decisions=self.decisions)

    # ---- pieces -------------------------------------------------------------------------------
    def _objective(self, mdp, r: int) -> str:
        cfg = mdp.cfg
        if int(mdp.goals[r]) in {G.parse_goal(g) for g in cfg.objective_goals}:
            return cfg.objective
        return "table"

    def _horizon(self, t: int) -> tuple:
        """(transitions planned with the belief's dynamics, nothing after them)."""
        H = int(self.cfg.belief_horizon)
        if self.cfg.belief_tail == "cap":
            left = self.n_steps - t
            if left <= H:
                return max(1, left), True
        return H, False

    def _rewards(self, mdp, r: int, legal, ok, lo, hi, H: _Hyp) -> tuple:
        """base (S, 10, 10) f32 and the coefficients of the part that reads the partner's
        belief (:func:`beliefkernels.nested_row`) -- the partner of hypothesis H, its social
        goal and physical goal, in its world."""
        w, cfg = mdp.w, mdp.cfg
        p = 1 - r
        t_star = int(H.social)
        k_star = int(H.goal)
        obj_k, dst_k = int(mdp.pg_obj[p][k_star]), int(mdp.pg_dst[p][k_star])
        al, be, nu, race_i = mdp.level2_term(r, int(mdp.goals[r]), t_star,
                                             k_star == int(mdp.own_goal[r]))
        a1, b1, c1 = mdp.level1_term(p, t_star)
        objective = self._objective(mdp, r)
        complete = objective == "complete"
        bonus = float(cfg.completion_bonus) if cfg.completion_bonus is not None else float(w.vdone)
        if self._base is None or self._base.shape != (w.S, N_ACTIONS, N_ACTIONS):
            self._base = np.zeros((w.S, N_ACTIONS, N_ACTIONS), dtype=np.float32)
        joint_rewards(r, w.N, legal, w.cls_of, w.at_partner, w.cls_obj, w.at_partner_obj,
                      w.rate_cls, w.fin_cls, w.cost_c, w.done, H.tab, w.pev, w.psucc, ok,
                      int(mdp.goal_ids[r]), obj_k, dst_k, float(al), float(be), float(nu),
                      float(race_i), float(b1), float(c1), float(mdp.lead),
                      float(mdp.cost_in_partner_reward), complete, bonus, lo, hi, self._base)
        coef = dict(al=float(al), a1=float(a1), c1=float(c1), obj_k=obj_k, dst_k=dst_k,
                    complete=complete, task=objective == "task")
        return self._base, coef

    def nested(self, mdp, r: int, lp, coef, ok, lo, hi) -> np.ndarray:
        """(S,) the part of robot r's reward that reads the partner's belief about it (log
        physical posterior ``lp``; its true goal under the task objective)."""
        w = mdp.w
        out = np.zeros(w.S)
        if coef["complete"] or (coef["a1"] == 0.0 and coef["c1"] == 0.0) or coef["al"] == 0.0:
            return out
        if coef["task"]:
            mine = np.zeros(mdp.PG)
            mine[mdp.own_goal[r]] = 1.0
        else:
            mine = np.exp(lp - np.logaddexp.reduce(lp))
        nested_row(r, w.N, np.ascontiguousarray(mine), np.ascontiguousarray(mdp.pg_obj[r]),
                   np.ascontiguousarray(mdp.pg_dst[r]), w.cls_est, w.at_partner_est, w.cls_obj,
                   w.at_partner_obj, w.rate_cls, w.fin_cls, ok, coef["obj_k"], coef["dst_k"],
                   coef["al"], coef["a1"], coef["c1"], float(mdp.lead), lo, hi, out)
        return out

    def _probes(self, mdp, r: int, LL, pts, s0: int, u0: int, point0: tuple, H: int,
                hyp: _Hyp) -> list:
        """The beliefs the partner would form if robot r acted by each hypothesis it holds
        about it (robot r's level-1 models, most likely action; the partner by its choice at
        the point nearest its belief), H transitions each, in the world of ``hyp``."""
        w = mdp.w
        p = 1 - r
        T = LL.shape[3]
        prior = point0[0]
        seen, out = set(), []
        legal = self.legal[r]
        for g in range(mdp.PG):
            if not np.isfinite(prior[g]):
                continue
            for t in range(T):
                row = mdp.hypothesis_row(r, g, t)
                if row in seen:
                    continue
                seen.add(row)
                s, u, b = s0, u0, point0
                for _ in range(H):
                    c = s // w.NN
                    if w.done[c, r]:
                        break
                    ok = np.flatnonzero(legal[s])
                    if not ok.size:
                        break
                    a = int(ok[np.argmax(LL[s, ok, g, t])])
                    k = pts.nearest(b)
                    best, alt = pts.best[k], pts.alt[k]
                    ap = int(alt[s] if u == best[s] else best[s])
                    b = self.step_belief(mdp, r, LL, b, s, a, ap, hyp)
                    out.append(b)
                    s2, u = self._next(mdp, r, s, a, ap, hyp)
                    s = s2
        return out

    def _next(self, mdp, r: int, s: int, a_r: int, a_p: int, H: _Hyp) -> tuple:
        """(s', the partner's undo after) of the transition in the world of hypothesis H (the
        true one under the oracle: ``ObjectWorld.step``)."""
        w = mdp.w
        p = 1 - r
        act = (a_r, a_p) if r == 0 else (a_p, a_r)
        b, pair = divmod(int(s), w.NN)
        j = act[0] * N_ACTIONS + act[1]
        s2 = int(H.tab[b, w.pev[pair, j]]) * w.NN + int(w.psucc[pair, j])
        st, st2 = w.unsid(s), w.unsid(s2)
        moved = st2[p] != st[p]
        changed = int(w.hold[st[2], p]) != int(w.hold[st2[2], p])
        return s2, int(undo_code(a_p, moved, changed))

    def _simulate(self, mdp, r: int, LL, pts, V1, s0: int, u0: int, point0: tuple, H: int,
                  first: int, hyp: _Hyp) -> list:
        """The planned way from the root, ``first`` first: robot r by the plan's first step
        at every state (receding horizon, V1), the partner by its choice at the point
        nearest its belief, the belief exactly (step_belief).  -> per transition: robot r's
        and the partner's action, the belief after (and its summary), its distance to the
        points."""
        w = mdp.w
        s, u, b = s0, u0, point0
        out = []
        for h in range(H):
            c = s // w.NN
            if w.done[c, r]:
                break
            k = pts.nearest(b)
            ap = int(pts.alt[k][s] if u == pts.best[k][s] else pts.best[k][s])
            if h == 0:
                a = first
            else:
                q = pts.q_exact(mdp, LL, V1, k, s, u, ap, b)
                cand = _first(q, 1)
                if not cand:
                    break
                a = cand[0]
            b = self.step_belief(mdp, r, LL, b, s, a, ap, hyp)
            _, _, _, res = pts.locate(b)
            sm = self._summary(mdp, r, b)
            out.append(dict(step=h + 1, robot=a, partner=ap, residual=float(res), belief=b,
                            object=sm["object"], destination=sm["destination"],
                            social=sm["social"], idle=sm["idle"]))
            s, u = self._next(mdp, r, s, a, ap, hyp)
        return out

    def _summary(self, mdp, r: int, b: tuple) -> dict:
        lp, ls, q = b
        lo, ld = mdp.factors(r, lp)
        return dict(object=np.exp(lo), destination=np.exp(ld),
                    social=None if ls is None else np.exp(ls), idle=float(q))


class _Points:
    """The belief points of one decision, and what the plan needs of each: the partner's
    choice (best, alt) at every state, the nested reward row, the tail; the projections of
    every transition's belief; the backward induction."""

    def __init__(self, planner: BeliefPlanner, mdp, r: int, ok, lo, hi, base, coef,
                 legal, H: _Hyp) -> None:
        self.pl, self.mdp, self.r, self.H = planner, mdp, r, H
        self.ok, self.lo, self.hi = ok, lo, hi
        self.base, self.coef, self.legal = base, coef, legal
        w = mdp.w
        self.S = w.S
        self.beliefs: list = []
        self.Z: list = []
        self.best: list = []
        self.alt: list = []
        self.Vp: list = []                    # the partner's root values (warm starts)
        self.nest: list = []
        self.tail: list = []
        self.tail_pol: list = []
        self.solved: list = []                # tables present
        self.has_social = mdp.about[r].log_social is not None
        self.D = mdp.PG + (len(mdp.about[r].log_social) if self.has_social else 0) + 2
        self.proj = None
        self.prev = planner.keep.get((r, H.key))
        self.stats: list = []

    @property
    def n(self) -> int:
        return len(self.beliefs)

    @staticmethod
    def embed_point(mdp, r: int, b: tuple) -> np.ndarray:
        """A belief about robot r in the points' coordinates (:func:`beliefkernels.embed`)."""
        lp, ls, q = b
        has_social = mdp.about[r].log_social is not None
        z = np.empty(mdp.PG + (len(mdp.about[r].log_social) if has_social else 0) + 2)
        embed(lp, ls if ls is not None else np.zeros(1), float(q), has_social, z)
        return z

    def _z(self, b: tuple) -> np.ndarray:
        lp, ls, q = b
        z = np.empty(self.D)
        embed(lp, ls if ls is not None else np.zeros(1), float(q), self.has_social, z)
        return z

    def add(self, b: tuple, tables=None) -> None:
        self.beliefs.append(b)
        self.Z.append(self._z(b))
        self.best.append(None)
        self.alt.append(None)
        self.Vp.append(None)
        self.nest.append(None)
        self.tail.append(None)
        self.tail_pol.append(None)
        self.solved.append(False)
        if tables is not None:
            best, alt, Vp = tables
            k = self.n - 1
            self.best[k], self.alt[k], self.Vp[k] = best, alt, Vp
            self._finish(k)

    def extend(self, cands: list, budget: int) -> int:
        """Farthest-point additions from ``cands`` while one lies farther than the spacing
        from every point, up to ``budget`` points.  -> how many were added."""
        if not cands or self.n >= budget:
            return 0
        Zc = np.stack([self._z(b) for b in cands])
        Zp = np.stack(self.Z)
        dmin = np.sqrt(((Zc[:, None, :] - Zp[None, :, :]) ** 2).sum(-1)).min(axis=1)
        added = 0
        spacing = float(self.pl.cfg.belief_spacing)
        while self.n < budget:
            i = int(np.argmax(dmin))
            if not dmin[i] > spacing:
                break
            self.add(cands[i])
            added += 1
            dmin = np.minimum(dmin, np.sqrt(((Zc - Zc[i]) ** 2).sum(-1)))
        return added

    def nearest(self, b: tuple) -> int:
        """The nearest point that has its tables (the partner's choice) -- point 0 has."""
        z = self._z(b)
        best, bk = np.inf, 0
        for k in range(self.n):
            if self.best[k] is None:
                continue
            d = float(((z - self.Z[k]) ** 2).sum())
            if d < best:
                best, bk = d, k
        return bk

    def locate(self, b: tuple) -> tuple:
        cfg = self.pl.cfg
        Z = np.ascontiguousarray(np.stack(self.Z))
        return project(self._z(b), Z, 4, cfg.belief_interpolation == "segment",
                       np.empty(self.n))

    def _warm(self, b: tuple):
        """The previous decision's point nearest b: (its partner values, tail, tail policy)."""
        prev = self.prev
        if not prev:
            return None, None, None
        z = self._z(b)
        d = ((prev["Z"] - z) ** 2).sum(-1)
        k = int(np.argmin(d))
        return prev["Vp"][k], prev["tail"][k], prev["tail_pol"][k]

    def solve_new(self) -> None:
        """The tables of every point that has none: the partner re-solved at each, all of
        them in one batch of the solver (as many as its work arrays hold at a time)."""
        pending = [k for k in range(self.n) if self.best[k] is None]
        size = max(1, int(self.mdp.batch))
        for g0 in range(0, len(pending), size):
            group = pending[g0:g0 + size]
            warm = [self._warm(self.beliefs[k])[0] for k in group]
            out = self.pl._partner_batch(self.mdp, self.r, [self.beliefs[k] for k in group],
                                         self.ok, self.lo, self.hi, warm, self.H)
            for k, (Q, Vp, st) in zip(group, out):
                best = np.empty(self.S, dtype=np.int8)
                alt = np.empty(self.S, dtype=np.int8)
                choice_tables(Q, self.pl.margin, best, alt)
                self.best[k], self.alt[k], self.Vp[k] = best, alt, Vp
                self.stats.append(dict(pi=float(st[0]), krylov=float(st[1]),
                                       residual=float(st[2]), ok=bool(st[3] > 0.5)))
                self._finish(k)

    def _finish(self, k: int) -> None:
        """The nested row and the tail of point k (its partner's tables present)."""
        mdp, r, w = self.mdp, self.r, self.mdp.w
        self.nest[k] = self.pl.nested(mdp, r, self.beliefs[k][0], self.coef, self.ok, self.lo,
                                      self.hi)
        _, tail_warm, pol_warm = self._warm(self.beliefs[k])
        if tail_warm is None:
            depth, row = mdp.root[r]
            V = mdp.bank(depth).V[row].astype(np.float64).copy()
            pol = np.full(self.S, -1, dtype=np.int64)
        else:
            V, pol = tail_warm.copy(), pol_warm.copy()
        t0 = time.perf_counter()
        it, res = tail_values(V, pol, self.best[k], self.base, self.nest[k], self.legal, r,
                              w.N, self.H.tab, w.pev, w.psucc, self.ok, self.pl.gamma, 300)
        self.pl.clock["tails"] = self.pl.clock.get("tails", 0.0) + time.perf_counter() - t0
        self.tail[k], self.tail_pol[k] = V, pol
        self.solved[k] = True
        self.stats.append(dict(tail_iterations=int(it), tail_residual=float(res)))

    def project(self, LL) -> None:
        """Every transition's belief from every point, located among the points."""
        mdp, r, w, cfg = self.mdp, self.r, self.mdp.w, self.pl.cfg
        nk = self.n
        P_lp = np.ascontiguousarray(np.stack([b[0] for b in self.beliefs]))
        P_ls = np.ascontiguousarray(np.stack([b[1] if b[1] is not None else np.zeros(1)
                                              for b in self.beliefs]))
        P_q = np.array([b[2] for b in self.beliefs])
        Z = np.ascontiguousarray(np.stack(self.Z))
        best = np.ascontiguousarray(np.stack(self.best))
        alt = np.ascontiguousarray(np.stack(self.alt))
        shape = (nk, self.S, N_ACTIONS, 2)
        k1t = np.zeros(shape, dtype=np.int16)
        k2t = np.zeros(shape, dtype=np.int16)
        lamt = np.zeros(shape, dtype=np.float32)
        rest = np.zeros(shape, dtype=np.float32)
        it_k, it_lo, it_hi = self._items(nk)
        ctx = self.pl._ctx(mdp, r, LL, self.H)
        projections(P_lp, P_ls, P_q, Z, best, alt, self.legal, *ctx,
                    4, cfg.belief_interpolation == "segment", self.ok, it_k, it_lo, it_hi,
                    k1t, k2t, lamt, rest)
        self.proj = (k1t, k2t, lamt, rest)

    def _items(self, nk: int) -> tuple:
        ks = [np.full(self.lo.size, k, dtype=np.int64) for k in range(nk)]
        return (np.concatenate(ks), np.tile(self.lo, nk), np.tile(self.hi, nk))

    def backup(self, H: int, zero_tail: bool, frozen: bool, only: Optional[int] = None):
        """V_1 (nk, 2, S): H - 1 steps of the backward induction from the tails (zero_tail:
        from 0) -- ``frozen``: the belief held at each point (``only``: the first points
        alone).  The second index: the partner plays its best (0) or, that an undo, its
        alternative (1)."""
        w = self.mdp.w
        nk = self.n if only is None else only
        Vn = np.empty((nk, NU, self.S))
        for k in range(nk):
            Vn[k] = 0.0 if zero_tail else self.tail[k][None, :]
        if H <= 1:
            return Vn
        Vc = np.empty_like(Vn)
        best = np.ascontiguousarray(np.stack(self.best[:nk]))
        alt = np.ascontiguousarray(np.stack(self.alt[:nk]))
        nest = np.ascontiguousarray(np.stack(self.nest[:nk]))
        if frozen:
            z16 = np.zeros((1, 1, 1, 1), dtype=np.int16)
            k1t, k2t, lamt = z16, z16, np.zeros((1, 1, 1, 1), dtype=np.float32)
        else:
            k1t, k2t, lamt, _ = self.proj
        it_k, it_lo, it_hi = self._items(nk)
        for _ in range(H - 1):
            backup(Vn, Vc, best, alt, self.base, nest, k1t, k2t, lamt, frozen, self.legal,
                   self.r, w.N, self.H.tab, w.pev, w.psucc, w.hold, self.ok, self.pl.gamma,
                   it_k, it_lo, it_hi)
            Vn, Vc = Vc, Vn
        return Vn

    def q(self, V1, k: int, s: int, u: int, a_p: int, hold: bool) -> np.ndarray:
        """Q of every action at (k, s), the partner playing a_p (its undo u having chosen it):
        the belief after each action located as the projections have it (``hold``: held at
        point k for this transition)."""
        w = self.mdp.w
        if hold or self.proj is None:
            k1 = np.full(N_ACTIONS, k, dtype=np.int64)
            k2 = k1.copy()
            lam = np.zeros(N_ACTIONS)
        else:
            k1t, k2t, lamt, _ = self.proj
            v = 0 if a_p == int(self.best[k][s]) else 1
            k1 = k1t[k, s, :, v].astype(np.int64)
            k2 = k2t[k, s, :, v].astype(np.int64)
            lam = lamt[k, s, :, v].astype(np.float64)
        out = np.empty(N_ACTIONS)
        nk = V1.shape[0]
        q_point(int(s), int(a_p), V1, np.ascontiguousarray(np.stack(self.best[:nk])), self.base,
                self.nest[k], k1, k2, lam, self.legal, self.r, w.N, self.H.tab, w.pev, w.psucc,
                w.hold, self.pl.gamma, out)
        return out

    def q_exact(self, mdp, LL, V1, k: int, s: int, u: int, a_p: int, b: tuple) -> np.ndarray:
        """Q at state s from the exact belief b (not a point): the belief after each action
        computed from b and located among the points; the nested reward of point k."""
        w = mdp.w
        k1 = np.zeros(N_ACTIONS, dtype=np.int64)
        k2 = np.zeros(N_ACTIONS, dtype=np.int64)
        lam = np.zeros(N_ACTIONS)
        for a in np.flatnonzero(self.legal[s]).tolist():
            b2 = self.pl.step_belief(mdp, self.r, LL, b, s, a, a_p, self.H)
            k1[a], k2[a], lam[a], _ = self.locate(b2)
        out = np.empty(N_ACTIONS)
        q_point(int(s), int(a_p), V1, np.ascontiguousarray(np.stack(self.best)), self.base,
                self.nest[k], k1, k2, lam, self.legal, self.r, w.N, self.H.tab, w.pev, w.psucc,
                w.hold, self.pl.gamma, out)
        return out

    def after(self, LL, s: int, a: int, a_p: int) -> tuple:
        return self.pl.step_belief(self.mdp, self.r, LL, self.beliefs[0], s, a, a_p, self.H)

    def keep(self) -> dict:
        """What the next decision warm-starts from."""
        ks = [k for k in range(self.n) if self.solved[k]]
        return dict(Z=np.stack([self.Z[k] for k in ks]), Vp=[self.Vp[k] for k in ks],
                    tail=[self.tail[k] for k in ks], tail_pol=[self.tail_pol[k] for k in ks])
