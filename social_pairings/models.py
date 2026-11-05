"""The Social MDP of a pair of robots: models at levels 0, 1 and 2, and beliefs over them.

Fig. 1 / Algorithm 1 of the paper (Tejwani et al., ICRA 2022), for robot o with partner p,
as the authors' solver (social-mdp-framework, ``main.cpp`` / ``smdp.cu``) structures it:

    Eq. 2    R_m(s, a_o, a_p) = r_o(s') - c(a_o) + xi_m(s, a_o, a_p)
    Eq. 3-4  Q_m(s, a_o) = sum_{a_p} P_m(a_p | s) [ R_m + gamma V_m(s') ],  V_m = max_a Q_m
    Eq. 7    psi_m(a | s) = softmax_a( Q_m(s, a) / tau_p )   (tau_p: the partner's, below)
    Eq. 6    b_g(g)   <- b_g(g)   psi^0_{p,g}(a_p | s)                       (physical goal)
    Eq. 5    b_xi(xi) <- b_xi(xi) sum_g b_g(g) psi^1_{p,(g,xi)}(a_p | s)      (social goal)

with ``P_m(a_p | s) = sum_k w_k psi_k(a_p | s)`` over the hypotheses k about the partner one
level down, weighted by the belief -- at level 2 by the product ``b_g(g) b_xi(xi)`` of the
two posteriors, as Eq. 4 is printed.  What a robot does not know of its partner is its
physical goal -- its destination (always) and, with ``Config.partner_object="inferred"``
(the paper's), which object it wants -- and its social goal (``Config.partner_social_goal=
"inferred"``).  A *physical goal* g is an object ``pg_obj[i, g]`` and a candidate
destination ``pg_dst[i, g]``: with the partner's object known, robot i's PG = K goals are
its own object to each candidate; inferred, PG = M K goals, every object to every candidate
(the robot's own included).  A robot knows its own physical goal, ``own_goal[i]``.

**Two posteriors, as the authors' code keeps them** (``MultiPolicyEstimator`` pgeX / sgeX;
since 2026-10-30, in place of one joint posterior over (object, destination, social goal)):

* the *physical* posterior ``about[j].log_phys`` over robot j's PG physical goals, updated
  from robot j's actions by the likelihood of robot j's level-0 policies (Eq. 6; the
  authors' PGE) -- in two stages, by robot j's hands (``Config.belief_stages``, the user's
  rule of 2026-10-17): while robot j holds nothing its actions are evidence about which
  object it wants and none about its destination; while it holds an object, evidence
  about its destination and none about its object.  Since 2026-11-04
  (``Config.belief_factored``, the user: "WE SHOULD HAVE TWO POSTERIORS one for goal and
  one for objects") it is the product of two posteriors, one about the object and one
  about the destination (:meth:`SocialMDP.factors`, :meth:`SocialMDP.product`), each
  learnt in its stage and frozen in the other's (:meth:`SocialMDP.factored_posterior`):
  the destination is believed whatever the object, so that a destination only the less
  likely object explains can still win;
* the *social* posterior ``about[j].log_social`` over the social goals held possible,
  updated by the likelihood of robot j's level-1 hypothesis policies averaged over the
  physical posterior (Eq. 5; the authors' SGE);
* one idle probability ``about[j].idle`` (``Config.idle_hypothesis``): a two-mode filter
  on top -- the partner acts by its hypotheses, or stands still whatever its goal -- so
  that a stalled partner does not pin either posterior;
* Eq. 5's transition term ``T(s^t | s^{t-1}, a_J)``: the outcomes of the hands that depend
  on the partner's goal -- a set-down for the partner that counts or fails, a hand-over
  the partner takes or not, the partner's own set-down that finishes or does nothing --
  rule the hypotheses out that predicted otherwise.

Nothing else is in the update (the carry and park tables and the "outcome" observation of
before are gone): the likelihood is Eq. 7's softmax of the hypothesis model's Q at the
state, at Eq. 7's tau, of the action the partner chose.  Since 2026-11-04, two options, off
by default, soften it: a temperature of the update's own (``Config.inference_temperature``:
the softmax at tau_inf, i.e. ``psi^eta`` renormalised, eta = tau / tau_inf; the robots still
plan at Eq. 7's tau) and a lapse (``Config.belief_lapse``: ``(1 - e) psi + e / |A|``, the
partner taken to pick an allowed action at random with probability e, so that one odd move
cannot cost a hypothesis more than a factor of about ``|A| / e``; :meth:`SocialMDP.lapsed`).

**A tau per robot** (``Config.robot_temperatures``, 2026-11-01): robot i reads its partner at
its own tau, ``taus[i]`` -- the Eq. 7 policies it predicts the partner by and the likelihood
of its update about the partner.  Every model of robot o is therefore softmaxed at its
partner's tau, ``taus[1 - o]``, wherever it is used (the partner predicting robot o, the
partner's update about robot o, and inside robot o's own level-2 model, the partner's
level-1 models reading robot o): the robots know each other's tau.  One tau for both is the
rule of before.  The belief a robot's level-1 models (the hypotheses its partner plans
against) hold about the robot's partner is the same posterior ``about[.]``: the authors'
estimators are shared between the acting robot and the models of it, the evidence being
the same.

* **Level 0** -- a plain MDP per (robot, physical goal): the partner is a moving obstacle
  whose hands do nothing.  Independent of beliefs, solved once.
* **Level 1** -- one model per (robot, physical goal, level-1 social term); the partner
  hypotheses are the PG level-0 models of the partner.  A level-1 robot has one such model
  (its root); a level-2 robot needs its partner's level-1 models for *every* physical goal.
* **Level 2** -- the root of a level-2 robot; the hypotheses are the partner's level-1
  models, PG physical goals x the social goals held possible.

Every model of a level is solved after every transition, all of them in one parallel batch
where the memory allows (``batch``), each to the Bellman-residual certificate
``|V - V*| <= Config.value_tol`` (warm-started).  Beliefs are frozen inside a solve.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import goals as G
from .config import Config, SolverFailure
from .constants import GIVE, JOINT, N_ACTIONS, PICK, PLACE, WORK_BYTES
from .kernels import (inputs_level0, inputs_typed, policy_graph, policy_iteration,
                      softmax_policies)
from .subgoals import Subgoals
from .util import (block_items, check, log_softmax, memory_limit_bytes, normalize_log,
                   stay_table)
from .world import ObjectWorld

__all__ = ["SocialMDP", "ModelBank", "BeliefSlot", "PartnerBelief", "NO_TERM"]

NO_TERM = (0.0, 0.0, 0.0)                  # the level-1 term of a robot without a social term
# The twin of a hinderer without a goal of its own (Config.hinder_own_goal=False): the same
# robot without its social term has nothing left to do but pay for its moves -- r(g_i) taken
# away (b = -1), nothing on the partner.  At level 1 the term (0, -1, 0); at level 2 the
# key NO_GOAL_TWIN, whose term is (0, -1, 0, 0) against every partner (SocialMDP.level2_term).
SPOILER_TWIN = (0.0, -1.0, 0.0)
NO_GOAL_TWIN = len(G.SOCIAL_GOALS)


def _log_sum(logs: np.ndarray) -> float:
    """log sum exp of a short array (-inf when every entry is -inf)."""
    logs = np.asarray(logs, dtype=np.float64)
    top = float(logs.max()) if logs.size else -np.inf
    if not np.isfinite(top):
        return top
    return top + float(np.log(np.exp(logs - top).sum()))


def _log_mix(log_a: np.ndarray, wa: float, log_b, wb: float) -> np.ndarray:
    """log(wa exp(log_a) + wb exp(log_b)) elementwise, with 0 log(-inf) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        la = np.where(wa > 0.0, np.log(wa) + log_a, -np.inf)
        lb = np.where(wb > 0.0, np.log(wb) + np.asarray(log_b, dtype=np.float64), -np.inf)
        return np.logaddexp(la, lb)


def _group_log_lik(post: np.ndarray, hard: np.ndarray, group: np.ndarray, n: int) -> np.ndarray:
    """(n,) log of the likelihood of one action under each group of physical hypotheses (an
    object, or a destination; ``Config.belief_window``): log sum_{g in it} exp(post[g]) - log
    sum exp(hard[g]), ``post`` the prior x likelihood x transition term and ``hard`` the prior
    x transition term (unnormalised logs) -- the likelihood averaged over the rest of the
    belief, given what the hands did.  0 for a group nothing is left of (the window's
    ``ruled_out`` takes it out), -inf where no hypothesis of the group explains the action."""
    out = np.zeros(n)
    for q in range(n):
        sel = group == q
        if not sel.any():
            continue
        had = _log_sum(hard[sel])
        if np.isfinite(had):
            out[q] = _log_sum(post[sel]) - had
    return out


def _model_summary(q: np.ndarray, log_pol: np.ndarray, a: int) -> dict:
    """What a set of models (..., 10) made of the action a, for the record
    (Config.belief_detail): every action's Q; the Q of a, the best Q and the action that has
    it, and their gap (reward units); log psi(a) at the likelihood's tau, before the lapse.
    NaN where a model was not consulted (its physical goal ruled out), -inf where an action
    is not open -- both null in the JSON."""
    with np.errstate(invalid="ignore"):
        best = np.where(np.isnan(q).all(-1, keepdims=True), np.nan,
                        np.max(np.where(np.isnan(q), -np.inf, q), axis=-1, keepdims=True))[..., 0]
        best_action = np.where(np.isnan(best), -1,
                               np.argmax(np.where(np.isfinite(q), q, -np.inf), axis=-1))
        taken = q[..., a]
        return dict(q=q, q_taken=taken, q_best=best, best_action=best_action,
                    q_gap=best - taken, log_psi=log_pol[..., a])


def mix_uniform(log_b: np.ndarray, h: float) -> np.ndarray:
    """log((1 - h) b + h u): the distribution ``exp(log_b)`` (normalised here) mixed with the
    uniform u over its support, its finite entries; the others stay -inf (Config.goal_hazard:
    a goal redrawn from the prior with probability h -- Baker, Saxe & Tenenbaum 2009, M2 --
    among the hypotheses still held possible).  h = 0: b as it is; h = 1: u."""
    log_b = np.asarray(log_b, dtype=np.float64)
    sup = np.isfinite(log_b)
    n = int(sup.sum())
    if n == 0:
        return log_b.copy()
    out = np.full(log_b.shape, -np.inf)
    with np.errstate(divide="ignore"):
        out[sup] = np.logaddexp(np.log1p(-h) + normalize_log(log_b[sup]), np.log(h / n))
    return out


class ModelBank:
    """Stacked arrays of the models of one reasoning level.

    Per model m: ``agent``, ``goal`` (index of its physical goal: SocialMDP.pg_obj /
    pg_dst; with the partner's object known, the candidate index of its destination), ``key`` (what
    tells it apart within its slot), ``social`` (it may hand over / set down for its
    partner), ``zero`` (no social term: once its own goal is done it is worth ``vdone``
    exactly, so those blocks are not solved), ``coeff`` (its level-1 term ``(a, b, c)``:
    :func:`social_pairings.goals.level1_term`).
    Arrays: ``V`` (M, S) values; ``R`` (M, S, 10) f32 expected rewards per own action;
    ``P`` (G, S, 10) the partner's action distribution of each slot; ``psi`` (H, S, 10) f32
    softmax policies of the models other models read; ``pd`` (G, B, K, N, 10) per slot,
    block, candidate cell the robot stands on, partner cell and partner action: the belief
    mass of the hypotheses whose destination is that cell -- and, where the partner's
    object is inferred, whose object the robot holds (a set-down for the partner counts
    with it); ``ho`` (G, S, 10), where the partner's object is inferred: the mass of the
    hypotheses whose object the robot holds (a hand-over counts with it).
    """

    def __init__(self) -> None:
        self.agent: list = []
        self.goal: list = []
        self.key: list = []
        self._social: list = []
        self._zero: list = []
        self._coeff: list = []
        self._psi_row: list = []
        self.G = 0                  # number of slots (rows of P)
        self.H = 0                  # number of stored policies (rows of psi)

    def add(self, agent: int, goal: int, key, *, coeff=NO_TERM, social: bool = False,
            zero: bool = True, policy: bool = False) -> int:
        self.agent.append(int(agent))
        self.goal.append(int(goal))
        self.key.append(key)
        self._social.append(int(social))
        self._zero.append(int(zero))
        self._coeff.append(tuple(float(v) for v in coeff))
        self._psi_row.append(self.H if policy else -1)
        self.H += int(policy)
        return len(self.agent) - 1

    @property
    def M(self) -> int:
        return len(self.agent)

    def allocate(self, w: ObjectWorld, *, rewards: bool, hand_over_mass: bool = False) -> None:
        """Allocate the arrays.  ``rewards=False`` is the level-0 bank: its reward tables
        are built and discarded group by group, and every model keeps its policy.
        ``hand_over_mass``: the partner's object is inferred (``ho``)."""
        S, M = w.S, self.M
        self.agent = np.asarray(self.agent, dtype=np.int64)
        self.goal = np.asarray(self.goal, dtype=np.int64)
        self.social = np.asarray(self._social, dtype=np.int64)
        self.zero = np.asarray(self._zero, dtype=np.int64)
        coeff = np.asarray(self._coeff, dtype=np.float64).reshape(M, 3)
        self.coeff_partner = np.ascontiguousarray(coeff[:, 0])     # of r(g_j)
        self.coeff_own = np.ascontiguousarray(coeff[:, 1])         # of r(g_i)
        self.coeff_race = np.ascontiguousarray(coeff[:, 2])        # of the race term
        self.var = np.zeros(M, dtype=np.int64)       # dynamics table: set-down for partner fails
        self.svar = np.zeros(M, dtype=np.int64)      # ... succeeds
        # np.zeros is lazily backed by the OS: blocks never touched cost nothing.
        self.V = np.zeros((M, S))
        self.P = np.zeros((max(self.G, 1), S, N_ACTIONS))
        if rewards:
            self.R = np.zeros((M, S, N_ACTIONS), dtype=np.float32)
            self.V_prev = np.zeros((M, S))
            self.pd = np.zeros((max(self.G, 1), w.B, w.K, w.N, N_ACTIONS))
            self.ho = (np.zeros((max(self.G, 1), S, N_ACTIONS)) if hand_over_mass
                       else np.zeros((1, 1, N_ACTIONS)))
            self.psi_row = np.asarray(self._psi_row, dtype=np.int64)
            self.psi = np.zeros((max(self.H, 1), S if self.H else 1, N_ACTIONS), dtype=np.float32)
        else:
            self.R = None
            self.V_prev = None
            self.pd = np.zeros((1, 1, 1, 1, N_ACTIONS))
            self.ho = np.zeros((1, 1, N_ACTIONS))
            self.psi_row = np.arange(M, dtype=np.int64)
            self.psi = np.zeros((M, S, N_ACTIONS), dtype=np.float32)
        self.p_of = np.zeros(M, dtype=np.int64)      # slot (row of P) of each model
        self.active = np.zeros(M, dtype=np.bool_)
        self.block = np.zeros(M, dtype=np.int64)     # the block each model is solved from
        self.step_norm = np.zeros(M)                 # |V_t - V_{t-1}| (extrapolation)
        self.n_solves = np.zeros(M, dtype=np.int64)
        self.pi_iters = np.zeros(M)
        self.krylov_iters = np.zeros(M)
        self.residual = np.zeros(M)
        self.ok = np.ones(M, dtype=np.bool_)


@dataclass
class PartnerBelief:
    """What is believed about one robot (``about[j]``: by its partner, and by the models of
    its partner inside its own level-1 models, which hold the same numbers): its physical
    goal, its social goal, and whether it is idle.

    log_phys : (PG,) log posterior over the robot's physical goals (object, destination).
    log_social : (n_types,) log posterior over the social goals its partner holds possible
        for it -- None where its partner is below level 2.
    idle : the probability that it stands still whatever its goal (``Config.idle_hypothesis``;
        0 without it).
    window : what the windowed physical posteriors are built from (``Config.belief_window``
        >= 1; None without the window).
    options : (PG, n_types, 1 + L) the belief about the robot's subgoal given each of its
        physical goals and social goals, index 0 its own task (``Config.physical_likelihood
        ="cut"``, :mod:`social_pairings.subgoals`; None otherwise).
    """
    log_phys: np.ndarray
    log_social: Optional[np.ndarray]
    idle: float
    window: Optional["EvidenceWindow"] = None
    options: Optional[np.ndarray] = None

    def phys(self) -> np.ndarray:
        return np.exp(normalize_log(self.log_phys))

    def social(self) -> Optional[np.ndarray]:
        return None if self.log_social is None else np.exp(normalize_log(self.log_social))

    def copy(self) -> "PartnerBelief":
        return PartnerBelief(self.log_phys.copy(),
                             None if self.log_social is None else self.log_social.copy(),
                             float(self.idle), self.window,
                             None if self.options is None else self.options.copy())


@dataclass(frozen=True)
class EvidenceWindow:
    """``Config.belief_window`` (2026-11-04): what the two physical posteriors about one
    robot are built from, afresh at every transition -- the uniform prior, what the hands'
    outcomes ruled out, and the last n likelihoods of each stage.

    ruled_out : (PG,) bool, what Eq. 5's transition term has ruled out so far (for good).
    objects : ((update, (M,) log l(o)), ...), the last n transitions with the robot's hands
        empty: the likelihood of its action under each object, the destination averaged out
        under the belief of then (up to a constant).
    destinations : ((update, (K,) log l(d)), ...), the last n with an object in its hands.
    updates : how many updates the belief has had (the transition of the next one).
    """
    ruled_out: np.ndarray
    objects: tuple = ()
    destinations: tuple = ()
    updates: int = 0


@dataclass(eq=False)
class BeliefSlot:
    """The models of robot ``agent`` that share one physical goal and one row of P.

    depth 1: robot ``agent``'s level-1 models with physical goal ``group`` -- its root (and
    twin) at level 1, and / or the hypotheses of a level-2 partner; the partner hypotheses
    are the PG level-0 models of the partner, weighted by ``about[partner].log_phys``.
    depth 2: a level-2 robot's root and twin; the hypotheses are (physical goal, social
    goal) of the partner's level-1 models, weighted by the product of the partner's two
    posteriors.  Every slot of one robot shares the P row of that robot and depth.
    """
    agent: int
    depth: int
    group: int                       # physical goal of the slot's models
    p_row: int                       # row of the bank's P, pd and ho
    model_rows: np.ndarray           # (Mm,) rows of the bank
    model_keys: list
    hyp_goal: np.ndarray             # (Mc,) physical goal of each partner hypothesis
    hyp_type: np.ndarray             # (Mc,) index into the robot's hypothesis goals (depth 2)


class SocialMDP:
    """Every model the two robots need, and their beliefs.

    world : the :class:`ObjectWorld`.
    goal_ids : (2,) candidate index of each robot's true destination (its physical goal is
        its own object to it: ``own_goal``).
    levels : (2,) reasoning level of each robot, 0..2.
    social_goals : (2,) index of each robot's social goal (:mod:`social_pairings.goals`).
    weights : (2,) each robot's weight on its social term.

    After :meth:`allocate` and :meth:`solve_level0`, one transition is::

        mdp.plan(state); mdp.inputs(1); mdp.solve(1); mdp.inputs(2); mdp.solve(2)
        q = mdp.q_values(mdp.root[i], s, i)          # act
        mdp.observe(actions, s, next_state, event)   # Eqs. 5-6
    """

    def __init__(self, world: ObjectWorld, goal_ids, levels, social_goals, weights,
                 cfg: Config, threads: int) -> None:
        w = self.w = world
        self.cfg, self.threads = cfg, threads
        self.goal_ids = tuple(int(g) for g in goal_ids)
        self.levels = tuple(int(n) for n in levels)
        if max(self.levels) > 2 or min(self.levels) < 0:
            raise NotImplementedError("reasoning levels 0, 1 and 2 are solved")
        self.weights = tuple(float(x) for x in weights)
        if any(not np.isfinite(x) or x < 0.0 for x in self.weights):
            raise ValueError(f"weights must be finite and >= 0, got {weights!r}")
        # a robot at level 0, or with weight 0, has no social goal
        self.goals = tuple(int(t) if (self.levels[i] >= 1 and self.weights[i] > 0.0) else G.NONE
                           for i, t in enumerate(social_goals))
        # a hinderer without a goal of its own (Config.hinder_own_goal=False): only its
        # partner's goal counts for it (goals.py, "hinder")
        self.spoiler = tuple(bool(t == G.HINDER and not cfg.hinder_own_goal)
                             for t in self.goals)
        K = w.K
        # ---- physical goals: an object and a destination ---------------------------------
        # pg_obj[i, g] / pg_dst[i, g]: the object and the candidate destination of robot i's
        # physical goal g.  Known partner objects: robot i's own object to each candidate.
        # Inferred (Config.partner_object): every object to every candidate, g = q K + k.
        self.infer_object = cfg.partner_object == "inferred"
        if self.infer_object:
            PG = w.M * K
            obj = np.repeat(np.arange(w.M, dtype=np.int64), K)
            dst = np.tile(np.arange(K, dtype=np.int64), w.M)
            self.pg_obj = np.ascontiguousarray(np.stack([obj, obj]))
            self.pg_dst = np.ascontiguousarray(np.stack([dst, dst]))
        else:
            PG = K
            self.pg_obj = np.ascontiguousarray(np.stack([np.zeros(K, dtype=np.int64),
                                                         np.ones(K, dtype=np.int64)]))
            self.pg_dst = np.ascontiguousarray(np.stack([np.arange(K, dtype=np.int64)] * 2))
        self.PG = PG
        # each robot's own physical goal: its own object to its destination
        self.own_goal = tuple(int(i * K + self.goal_ids[i]) if self.infer_object
                              else int(self.goal_ids[i]) for i in (0, 1))
        # the memory there is: max_memory_gib, or what this machine has where that is less --
        # a model that cannot fit fails here, before its first table is built, and not by
        # running out of memory minutes later
        self.budget = cfg.max_memory_gib * 2 ** 30
        limit = memory_limit_bytes()
        if limit is not None:
            self.budget = min(self.budget, float(limit))
        need = 6 * PG * w.B * w.nEv * 2                   # the dynamics tables alone
        if need > self.budget:
            raise SolverFailure(self._too_big(need, "its dynamics tables"))
        self.gamma = cfg.gamma
        # Eq. 7's tau per robot (Config.robot_temperatures): robot i reads its partner at
        # taus[i], so every model OF robot o -- its policies, the likelihood of its actions --
        # is softmaxed at its partner's tau, taus[1 - o] (read_inv_tau[o] = 1 / that)
        self.taus = (cfg.tau_of(0), cfg.tau_of(1))
        self.read_inv_tau = np.array([1.0 / self.taus[1], 1.0 / self.taus[0]])
        # ... and the tau of robot i's update about its partner (the likelihood of Eqs. 5-6):
        # the same, or Config.inference_temperature where set (2026-11-04)
        self.inference_taus = (cfg.inference_tau_of(0), cfg.inference_tau_of(1))
        self.exchange_share = float(cfg.exchange_share)
        self.cost_in_partner_reward = 1.0 if cfg.social_cost else 0.0
        self.lead = G.lead_weight(cfg.competition)       # of the race term (goals.py)
        # the partners a level-2 cooperator takes the reward of (Config.cooperate_with)
        self.cooperate_with = G.cooperation_partners(cfg.cooperate_with)
        hyp = tuple(dict.fromkeys(G.parse_goal(x) for x in cfg.social_hypotheses))
        # hyp_goals[i]: the social goals robot i holds possible for its partner
        self.known_social = cfg.partner_social_goal == "known"
        self.hyp_goals = (((self.goals[1],), (self.goals[0],)) if self.known_social
                          else (hyp, hyp))

        # ---- what an idle robot is predicted to do: its own moves (the two tables differ
        # only where the robots have rooms of their own) --------------------------------------
        self.valid_by = np.ascontiguousarray(w.valid_by)                     # (2, N, 5)
        self.idle_tab = [stay_table(w.valid_by[i], cfg.idle_eps) for i in (0, 1)]   # (N, 10)
        with np.errstate(divide="ignore"):
            self.log_idle = [np.log(t) for t in self.idle_tab]
        # (the carry table the kernels take where the partner delivers: never read)
        self._no_carry = np.zeros((w.N, N_ACTIONS))
        self._no_carry_stack = np.zeros((K, w.N, N_ACTIONS))

        # ---- dynamics tables: [o PG + g] fail, [2 PG + o PG + g] success, [4 PG + o PG + g]
        # level 0 -- robot o, physical goal g
        def goal(o: int, g: int) -> tuple:
            return int(self.pg_dst[o, g]), int(self.pg_obj[o, g])

        tabs = ([w.model_tab(o, goal(o, g)[0], obj=goal(o, g)[1])
                 for o in (0, 1) for g in range(PG)]
                + [w.model_tab(o, goal(o, g)[0], True, obj=goal(o, g)[1])
                   for o in (0, 1) for g in range(PG)]
                + [w.level0_tab(o, *goal(o, g)) for o in (0, 1) for g in range(PG)])
        self.tabs = np.ascontiguousarray(np.stack(tabs))                    # (6PG, B, nEv) i16
        # robot o's level-0 reward table for each object it may want
        self.level0_reward_tabs = {(o, q): w.level0_reward_tab(o, q)
                                   for o in (0, 1) for q in sorted(set(self.pg_obj[o].tolist()))}
        self.model_tabs = tuple(tabs[:4 * PG])   # the dynamics of every model above level 0
        self._reach_of: dict = {}
        self.useok = np.ascontiguousarray(np.stack([w.hand_ok(False), w.hand_ok(True)]))
        self.done = np.ascontiguousarray(w.done)
        # every block any dynamics in use can reach from the start
        self.reachable = w.reach(tuple(tabs[:4 * PG]), w.block0, (True, True))
        own_done = [w.done[:, a].astype(bool) for a in (0, 1)]
        self.level0_blocks = [self.reachable[~own_done[a][self.reachable]] for a in (0, 1)]

        # ---- level 0: robot a, physical goal g -> row a PG + g ---------------------------
        L0 = ModelBank()
        for a in (0, 1):
            for k in range(PG):
                L0.add(a, k, None)
        L0.G = 2
        self.L0, self.L1, self.L2 = L0, ModelBank(), ModelBank()
        self.slots: Dict[Tuple[int, int, int], BeliefSlot] = {}
        self.root: List[Optional[Tuple[int, int]]] = [None, None]   # (depth, row) per robot
        self.twin: List[Optional[Tuple[int, int]]] = [None, None]   # the same without social term
        # ---- the partner's subgoals (Config.physical_likelihood="cut", 2026-11-04): robot j's
        # as the robot that reads it at level 2 holds them possible, and the row of its
        # "none" model, by which its own task is read ------------------------------------------
        self.subgoals: List[Optional[Subgoals]] = [
            Subgoals(w, cfg, j, self.hyp_goals[1 - j])
            if cfg.physical_likelihood == "cut" and self.levels[1 - j] == 2 else None
            for j in (0, 1)]
        self.none_index = [list(self.hyp_goals[1 - j]).index(G.NONE)
                           if self.subgoals[j] is not None else None for j in (0, 1)]
        # ---- the beliefs: about[j], what robot 1 - j believes of robot j --------------------
        self.about: List[PartnerBelief] = [self._initial_belief(j) for j in (0, 1)]
        self._build_models()
        self._budget_memory()
        self.krylov_by_stage = {0: 0.0, 1: 0.0, 2: 0.0}
        self.tasks: list = []
        # lazy hypotheses (Config.lazy_mass): the transition each model was last solved at,
        # and what the last plan left stale (plan, lazy_rows)
        self.transition = 0
        self.last_solved = {1: np.full(self.L1.M, -1, dtype=np.int64),
                            2: np.full(self.L2.M, -1, dtype=np.int64)}
        self.lazy_info = None
        # what the last update computed, per robot j about which it was made
        # (Config.belief_detail; observe, updated_belief)
        self.last_update = None
        # Config.belief_inference="level2" (beliefplan.BeliefPlanner.inference): per robot j,
        # (q (PG, T, 10), log psi (PG, T, 10)) of the level-2 models of robot j under each
        # hypothesis (physical goal, social goal) at the state of the transition being
        # observed -- read by hypothesis_log_policies in place of the level-1 bank's
        self.likelihood_override: dict = {}

    # ---- construction ------------------------------------------------------------------------
    def _initial_belief(self, j: int) -> PartnerBelief:
        """The uniform priors of Algorithm 1 (t = 0) about robot j -- over its PG physical
        goals, and over the social goals its partner holds possible when its partner reasons
        at level 2 (a point mass on its actual goal where that is known)."""
        cfg = self.cfg
        n = len(self.hyp_goals[1 - j])
        social = np.full(n, -np.log(n)) if self.levels[1 - j] == 2 else None
        idle = float(cfg.idle_hazard) if cfg.idle_hypothesis else 0.0
        window = (EvidenceWindow(np.zeros(self.PG, dtype=bool)) if cfg.belief_window > 0
                  else None)                                      # Config.belief_window
        options = None
        if self.subgoals[j] is not None:          # Config.physical_likelihood="cut": its own
            options = np.zeros((self.PG, n, 1 + self.subgoals[j].L))   # task, no subgoal
            options[:, :, 0] = 1.0
        return PartnerBelief(np.full(self.PG, -np.log(self.PG)), social, idle, window, options)

    def level1_term(self, robot: int, goal: int) -> Tuple[float, float, float]:
        """(a, b, c) of ``robot`` with social goal ``goal`` at level 1: the coefficients of
        r(g_j), of r(g_i) and of the race term (:func:`social_pairings.goals.level1_term`).
        Where partner goals are known the robot is told its partner's (a rival needs it:
        it races every partner but a cooperative one); its partner stays a level-0 robot
        in its model.  The goal it is told is the one that counts: none for a partner at
        level 0 or with weight 0."""
        return G.level1_term(goal, weight=self.weights[robot],
                             exchange_level1=self.cfg.exchange_level1,
                             exchange_share=self.exchange_share,
                             competition=self.cfg.competition,
                             partner_goal=self.goals[1 - robot] if self.known_social else None,
                             hinder_own_goal=self.cfg.hinder_own_goal)

    def level2_term(self, robot: int, goal: int, partner_goal: int, same: bool) -> tuple:
        """(alpha, beta, nu, c) of ``robot`` with social goal ``goal`` at level 2 against a
        partner taken to have ``partner_goal``: the coefficients of R_j, r(g_i), r(g_j) and
        of the race term (:func:`social_pairings.goals.level2_term`).  A cell of Table I's
        column l > 1 (as printed, or with the competitor's cell against a cooperative
        partner turned) has c = 0; a competitor that races this partner -- under a race
        form any partner, as a rival any partner but a cooperator -- has ``(0, 0, 0, w)``;
        a conflict robot under ``Config.conflict="task"`` ``(0, 0, -w, 0)``: -w r(g_j)
        against every partner, no R_j."""
        if goal == NO_GOAL_TWIN:         # the twin of a hinderer without a goal of its own
            return 0.0, -1.0, 0.0, 0.0
        return G.level2_term(goal, partner_goal, same, weight=self.weights[robot],
                             exchange_share=self.exchange_share,
                             competition=self.cfg.competition,
                             cooperate_with=self.cooperate_with,
                             hinder_own_goal=self.cfg.hinder_own_goal,
                             conflict=self.cfg.conflict)

    def _no_term(self, robot: int, goal: int) -> bool:
        """A level-2 robot whose social term vanishes against every hypothesis."""
        return all(self.level2_term(robot, goal, t, same) == (0.0, 0.0, 0.0, 0.0)
                   for t in self.hyp_goals[robot] for same in (False, True))

    def _build_models(self) -> None:
        cfg, K = self.cfg, self.PG
        L1, L2, levels, goal_ids = self.L1, self.L2, self.levels, self.own_goal
        for i in (0, 1):
            if levels[i] == 0:
                self.root[i] = (0, i * K + goal_ids[i])
        for a in (0, 1):                 # ---- level 1: one slot per physical goal -----------
            twin_key = SPOILER_TWIN if self.spoiler[a] else NO_TERM
            as_hypotheses = levels[1 - a] == 2
            groups = range(K) if as_hypotheses else ([goal_ids[a]] if levels[a] == 1 else [])
            hyp_terms = list(dict.fromkeys(self.level1_term(a, t)
                                           for t in self.hyp_goals[1 - a]))
            p_row = L1.G                 # every level-1 slot of robot a shares one row of P
            any_slot = False
            for k in groups:
                keys = list(hyp_terms) if as_hypotheses else []
                mine = levels[a] == 1 and k == goal_ids[a]
                if mine:
                    root_key = self.level1_term(a, self.goals[a])
                    keys.append(root_key)
                    if cfg.physical_twin and root_key != NO_TERM:
                        keys.append(twin_key)
                rows = {}
                for key in dict.fromkeys(keys):
                    # a term on the partner's goal: the robot may act for its partner
                    rows[key] = L1.add(a, k, key, coeff=key,
                                       social=key[0] != 0.0 or key[2] != 0.0,
                                       zero=key == NO_TERM,
                                       policy=as_hypotheses and key in hyp_terms)
                self.slots[(a, 1, k)] = BeliefSlot(
                    agent=a, depth=1, group=k, p_row=p_row,
                    model_rows=np.asarray(list(rows.values()), dtype=np.int64),
                    model_keys=list(rows), hyp_goal=np.arange(K),
                    hyp_type=np.zeros(K, dtype=np.int64))
                any_slot = True
                if mine:
                    self.root[a] = (1, rows[root_key])
                    if twin_key in rows and root_key != twin_key:
                        self.twin[a] = (1, rows[twin_key])
            L1.G += int(any_slot)
        for a in (0, 1):                 # ---- level 2 ------------------------------------------
            if levels[a] != 2:
                continue
            goal = self.goals[a]
            zero = self._no_term(a, goal)
            n_types = len(self.hyp_goals[a])
            rows = [L2.add(a, goal_ids[a], goal, social=not zero, zero=zero)]
            keys = [goal]
            self.root[a] = (2, rows[0])
            if cfg.physical_twin and not zero:
                twin_goal = NO_GOAL_TWIN if self.spoiler[a] else G.NONE
                rows.append(L2.add(a, goal_ids[a], twin_goal, zero=not self.spoiler[a]))
                keys.append(twin_goal)
                self.twin[a] = (2, rows[1])
            self.slots[(a, 2, goal_ids[a])] = BeliefSlot(
                agent=a, depth=2, group=goal_ids[a], p_row=L2.G,
                model_rows=np.asarray(rows, dtype=np.int64), model_keys=keys,
                hyp_goal=np.repeat(np.arange(K), n_types), hyp_type=np.tile(np.arange(n_types), K))
            L2.G += 1

    def _too_big(self, need: float, what: str) -> str:
        w, cfg = self.w, self.cfg
        return (f"the model needs more than {need / 2 ** 30:.1f} GiB for {what}, and there are "
                f"{self.budget / 2 ** 30:.1f} GiB (max_memory_gib={cfg.max_memory_gib:g}, or "
                f"what this machine has): {w.M} objects and {w.K} candidate destinations "
                f"({self.PG} physical goals per robot), N={w.N} cells x {w.B} object "
                f"configurations: use fewer objects (max_objects), the partner's object known "
                f"(partner_object=\"known\"), a coarser grid, fewer social hypotheses or lower "
                f"levels")

    def _budget_memory(self) -> None:
        """Size of the solver batch (models solved together) within the budget:
        ``max_memory_gib``, or what this machine has where that is less.  The batch is every
        model of a level where the memory allows: all the hypotheses of one level are then
        solved in one parallel call."""
        w, K = self.w, self.w.K
        L0, L1, L2 = self.L0, self.L1, self.L2
        touched = self.reachable.size * w.NN
        n_models, n_slots = L1.M + L2.M, max(L1.G, 1) + max(L2.G, 1)
        f32, f64 = 4 * N_ACTIONS, 8 * N_ACTIONS     # a row over the actions
        fixed = (self.tabs.nbytes + 4 * w.NN * N_ACTIONS * N_ACTIONS * 4
                 + L0.M * touched * (8 + f32)
                 + 2 * touched * f64 + n_models * touched * (8 + 8 + f32) + L1.H * touched * f32
                 + n_slots * touched * f64 + n_slots * w.B * K * w.N * f64
                 + (n_slots * touched * f64 if self.infer_object else 0))
        per_model = touched * (WORK_BYTES + f32 + 8)
        want = max(2 * self.PG, L1.M, L2.M, 1)
        self.batch = int(min(want, (self.budget - fixed) // per_model))
        if self.batch < 1:
            raise SolverFailure(self._too_big(
                fixed + per_model, f"its solver arrays ({touched:,} joint states, "
                f"{n_models} models above level 0)"))
        self.memory_bytes = fixed + self.batch * per_model

    def allocate(self) -> None:
        w, K = self.w, self.PG
        L0 = self.L0
        L0.allocate(w, rewards=False)
        L0.var[:] = 4 * K + L0.agent * K + L0.goal
        L0.svar[:] = L0.var
        L0.social[:] = 0
        L0.p_of[:] = L0.agent
        for bank in (self.L1, self.L2):
            bank.allocate(w, rewards=True, hand_over_mass=self.infer_object)
            bank.var[:] = bank.agent * K + bank.goal
            bank.svar[:] = 2 * K + bank.agent * K + bank.goal
        self._work = np.zeros((6, self.batch, w.S))
        self._policy = np.zeros((self.batch, w.S), dtype=np.int8)
        self._stats = np.zeros((self.batch, 4))

    def bank(self, depth: int) -> ModelBank:
        return self.L0 if depth == 0 else (self.L1 if depth == 1 else self.L2)

    # ---- which blocks a model lives on -------------------------------------------------------
    def reach_of(self, agent: int, goal: int, block: int) -> np.ndarray:
        """The blocks the models of robot ``agent`` with destination ``goal`` are solved on
        from ``block``: every block the dynamics of ANY model can reach from it, the same
        set for every model.

        Not only the blocks of the model's own dynamics: its policy is read, one level up,
        at every state the reader's dynamics reach, and the two differ -- a model sets its
        own robot's object down on its destination, and nobody else's model does.  With
        ``Config.set_down_for_partner=True`` (the default) the two sets are the same, a
        partner's set-down reaching what the robot's own does.  With ``False`` they are
        not, and until 2026-10-07 a level-2 robot then read partner policies that had
        never been computed (all zero) wherever its own goal was done: finishing looked
        worthless to it, and it stopped one step short of its destination (the README,
        "Limits")."""
        key = int(block)
        if key not in self._reach_of:
            self._reach_of[key] = self.w.reach(self.model_tabs, key, (True, True))
        return self._reach_of[key]

    def solved_blocks(self, bank: ModelBank, row: int) -> np.ndarray:
        """Blocks model ``row`` is solved on (without a social term: not where its goal is
        done -- there it is worth ``vdone``)."""
        o = int(bank.agent[row])
        if bank is self.L0:
            return self.level0_blocks[o]
        full = self.reach_of(o, int(bank.goal[row]), int(bank.block[row]))
        return full[self.w.done[full, o] == 0] if bank.zero[row] else full

    # ---- solver plumbing ---------------------------------------------------------------------
    def _kernel_args(self, bank) -> tuple:
        w = self.w
        return (w.N, w.psucc, w.psucc_T, w.pev, w.pev_T, self.tabs,
                self.valid_by, w.busy, self.useok, w.uctx, w.site_of, w.goal_at_site,
                bank.pd, bank.ho, self.infer_object, self.done, w.vdone, w.hold, w.gone,
                w.state_ok, self.gamma)

    def _solve(self, V, R, bank, rows: np.ndarray, blocks: list) -> np.ndarray:
        """Solve models ``rows`` (values V, rewards R) on their blocks, in stages: a finished
        goal never comes undone, so the blocks with both goals done are solved first, then
        those with R0's, then those with R1's, then the rest -- each stage against the final
        values of the stages before it.  -> (L, 4): PI iterations, Krylov iterations,
        residual, converged."""
        cfg, w = self.cfg, self.w
        code = w.done[:, 0] * 2 + w.done[:, 1]                             # (B,)
        out = np.zeros((rows.size, 4))
        out[:, 3] = 1.0
        for stage, key in ((2, 3), (1, 2), (1, 1), (0, 0)):
            act = np.zeros((rows.size, w.B), dtype=np.uint8)
            parts = []
            for n, blk in enumerate(blocks):
                sel = blk[code[blk] == key]
                act[n, sel] = 1
                parts.append(block_items(w.NN, sel, n))
            it_m, it_lo, it_hi = (np.concatenate([p[k] for p in parts]) for k in range(3))
            if not it_m.size:
                continue
            policy_iteration(V, R, bank.P, bank.p_of, bank.agent, bank.var, bank.svar,
                             bank.social, bank.zero, rows, it_m, it_lo, it_hi,
                             *self._kernel_args(bank), cfg.residual_tol, cfg.pi_forcing,
                             cfg.max_pi_iters, cfg.max_krylov_iters, self._policy, self._work,
                             self._stats, act)
            st = self._stats[:rows.size]
            out[:, 0] += st[:, 0]
            out[:, 1] += st[:, 1]
            out[:, 2] = np.maximum(out[:, 2], st[:, 2])
            out[:, 3] = np.minimum(out[:, 3], st[:, 3])
            self.krylov_by_stage[stage] += float(st[:, 1].sum())
        return out

    # ---- level 0 -----------------------------------------------------------------------------
    def solve_level0(self) -> dict:
        """Solve the 2 PG level-0 models once, on every block anything can reach, in groups
        of at most ``batch`` models; each group's reward table is dropped after its softmax
        policies are stored."""
        w, L0 = self.w, self.L0
        t_in = t_solve = t_psi = 0.0
        everywhere = [block_items(w.NN, self.reachable, 0) for _ in (0, 1)]
        # what robot a's level-0 model reads of its PARTNER: its legal moves
        valid = [np.ascontiguousarray(w.valid_by[1 - a]) for a in (0, 1)]
        no_park = np.zeros((w.N, N_ACTIONS))
        none = np.zeros(0, dtype=np.int64)
        for a in (0, 1):                 # the partner's distribution of each robot's models
            inputs_level0(a, a, none, none, w.N, w.psucc, w.pev, self.level0_reward_tabs[(a, a)],
                          valid[a], w.busy, w.cls_obj, w.rate_cls, w.cost_c, self.done,
                          everywhere[a][1], everywhere[a][2], L0.P, a,
                          np.zeros((1, 1, N_ACTIONS), dtype=np.float32), no_park, False,
                          w.at_partner_obj, w.hold, w.dest)
        order = np.arange(L0.M)
        for g0 in range(0, L0.M, self.batch):
            grp = order[g0:g0 + self.batch]
            tic = time.perf_counter()
            Rg = np.zeros((grp.size, w.S, N_ACTIONS), dtype=np.float32)
            loc = np.arange(grp.size, dtype=np.int64)
            for a in (0, 1):
                objs = self.pg_obj[a, L0.goal[grp]]
                for q in sorted(set(objs[L0.agent[grp] == a].tolist())):
                    sel = (L0.agent[grp] == a) & (objs == q)
                    inputs_level0(a, int(q), self.pg_dst[a, L0.goal[grp][sel]], loc[sel], w.N,
                                  w.psucc, w.pev, self.level0_reward_tabs[(a, int(q))], valid[a],
                                  w.busy, w.cls_obj, w.rate_cls, w.cost_c, self.done,
                                  everywhere[a][1], everywhere[a][2], L0.P, a, Rg, no_park,
                                  False, w.at_partner_obj, w.hold, w.dest)
            Vg = np.stack([w.solo_state_values(int(L0.agent[m]),
                                               int(self.pg_dst[L0.agent[m], L0.goal[m]]),
                                               int(self.pg_obj[L0.agent[m], L0.goal[m]]))
                           for m in grp])
            sub = _BankView(L0, grp)
            t_in += time.perf_counter() - tic
            tic = time.perf_counter()
            st = self._solve(Vg, Rg, sub, loc,
                             [self.level0_blocks[int(L0.agent[m])] for m in grp.tolist()])
            L0.pi_iters[grp], L0.krylov_iters[grp] = st[:, 0], st[:, 1]
            L0.residual[grp], L0.ok[grp] = st[:, 2], st[:, 3] > 0.5
            t_solve += time.perf_counter() - tic
            tic = time.perf_counter()
            parts = [block_items(w.NN, self.reachable, n) for n in range(grp.size)]
            it_m, it_lo, it_hi = (np.concatenate([p[k] for p in parts]) for k in range(3))
            softmax_policies(Vg, Rg, L0.P, sub.p_of, sub.agent, sub.var, sub.svar, sub.social,
                             sub.zero, loc, np.ascontiguousarray(grp), it_m, it_lo, it_hi,
                             *self._kernel_args(L0), self.read_inv_tau, L0.psi)
            L0.V[grp] = Vg
            del Rg, Vg
            t_psi += time.perf_counter() - tic
        return dict(models=L0.M, pi_mean=float(L0.pi_iters.mean()),
                    pi_max=int(L0.pi_iters.max()), kry_mean=float(L0.krylov_iters.mean()),
                    kry_max=int(L0.krylov_iters.max()), res_max=float(L0.residual.max()),
                    ok=bool(L0.ok.all()), t_inputs=t_in, t_solve=t_solve, t_psi=t_psi)

    # ---- one transition: plan, inputs, solve -------------------------------------------------
    def plan(self, state) -> None:
        """Mark the models to solve from ``state``: every slot's models, except the level-1
        partner models of a physical goal that the partner's posterior has ruled out
        altogether (the transition term of Eq. 5 can) -- and, ``Config.lazy_mass`` > 0,
        except the partner hypotheses left stale this transition (:meth:`lazy_rows`)."""
        block = int(state[2])
        for bank in (self.L1, self.L2):
            bank.active[:] = False
        self.tasks = []                          # (depth, slot, rows, keys)
        lazy = self.lazy_rows() if self.cfg.lazy_mass > 0.0 else None
        for key in sorted(self.slots, key=lambda kk: (kk[1], kk[0], kk[2])):
            slot = self.slots[key]
            a, depth, group = key
            if (depth == 1 and self.levels[1 - a] == 2
                    and not np.isfinite(self.about[a].log_phys[group])
                    and not (self.levels[a] == 1 and group == self.own_goal[a])):
                continue
            bank = self.bank(depth)
            rows, keys = slot.model_rows, list(slot.model_keys)
            if lazy is not None and depth == 1:
                pick = [n for n, r in enumerate(rows.tolist()) if r in lazy]
                if not pick:
                    continue
                rows = rows[pick]
                keys = [keys[n] for n in pick]
            bank.active[rows] = True
            bank.p_of[rows] = slot.p_row
            bank.block[rows] = block
            self.last_solved[depth][rows] = self.transition
            self.tasks.append((depth, slot, rows.copy(), keys))
        self.transition += 1

    def lazy_rows(self) -> set:
        """Lazy hypotheses (``Config.lazy_mass``, 2026-11-04): the level-1 partner
        hypotheses to solve this transition.  A level-2 robot predicts its partner by the
        partner's level-1 models mixed by its two posteriors (Eq. 4), and reads each
        model's policy for the likelihood of what the partner did (Eqs. 5-6).  The models
        of the hypotheses that carry little mass -- the least likely physical goals, the
        least likely social goals -- hardly move that prediction, so they are left as last
        solved: of each robot's hypothesis models, in order of their mass
        ``b_g(g) b_xi(xi)`` (the hypotheses of one model summed), the lightest ones whose
        masses add up to at most ``lazy_mass`` are skipped, every model is solved at least
        every ``lazy_period`` transitions, and every model is solved at the first
        transition.  A skipped model keeps its values and its stored policy (``psi``): the
        root plans with them as they were, and the likelihood of that hypothesis is read
        off them, a few transitions stale.  The roots and twins are always solved; so are
        the level-1 roots of a level-1 robot.  Exact at ``lazy_mass`` = 0 (the default:
        every model every transition).  ``lazy_info``: per robot, (models skipped, their
        mass, models solved)."""
        cfg, L1 = self.cfg, self.L1
        keep = set()
        info = []
        for a in (0, 1):
            weight: Dict[int, float] = {}
            if self.levels[1 - a] == 2:
                phys = self.about[a].phys()
                social = self.about[a].social()
                hyp = self.hyp_goals[1 - a]
                for k in range(self.PG):
                    if not np.isfinite(self.about[a].log_phys[k]):
                        continue
                    for t in range(len(hyp)):
                        r = self.hypothesis_row(a, k, t)
                        weight[r] = weight.get(r, 0.0) + float(phys[k] * social[t])
            always = {r for r in weight
                      if (L1.psi_row[r] < 0                      # a root or twin, no hypothesis
                          or self.last_solved[1][r] < 0           # never solved
                          or self.transition - self.last_solved[1][r] >= cfg.lazy_period)}
            order = sorted(weight, key=lambda r: weight[r])
            mass = 0.0
            skipped = []
            for r in order:
                if r in always or mass + weight[r] > cfg.lazy_mass:
                    keep.add(r)
                else:
                    mass += weight[r]
                    skipped.append(r)
            for key, slot in self.slots.items():
                if slot.agent == a and slot.depth == 1:      # roots and twins of level-1 robots
                    for r in slot.model_rows.tolist():
                        if r not in weight:
                            keep.add(r)
            info.append((len(skipped), mass, len(weight) - len(skipped)))
        self.lazy_info = tuple(info)
        return keep

    def init_models(self) -> None:
        """Warm start of the first solve: every model starts from its robot's solo values."""
        for depth, slot, rows, _ in self.tasks:
            bank = self.bank(depth)
            for r in rows.tolist():
                a, g = slot.agent, int(bank.goal[r])
                bank.V[r] = self.w.solo_state_values(a, int(self.pg_dst[a, g]),
                                                     int(self.pg_obj[a, g]))

    def hypothesis_row(self, partner: int, k: int, t: int) -> int:
        """Level-1 row of the partner hypothesis (physical goal k, social goal number t of
        the viewer's hypothesis goals)."""
        slot = self.slots[(partner, 1, k)]
        term = self.level1_term(partner, self.hyp_goals[1 - partner][t])
        return int(slot.model_rows[slot.model_keys.index(term)])

    def inputs(self, depth: int) -> None:
        """Eq. 4's partner distribution and Eq. 2's expected rewards of every active model
        of one depth, from the current beliefs and the policies one level down."""
        for d, slot, rows, keys in self.tasks:
            if d != depth or not rows.size:
                continue
            inputs_typed(*self._input_args(d, slot, rows, keys))

    def planning_weights(self, slot: BeliefSlot, about: Optional[list] = None) -> np.ndarray:
        """(Mc,) the weights a slot's partner hypotheses are planned with (Eq. 4): at depth
        1 the physical posterior about the partner; at depth 2 the product of the partner's
        physical and social posteriors, hypothesis (k, t) by (k, t).  ``about``: the beliefs
        to read in place of ``self.about`` (the belief planner's hypothetical beliefs)."""
        b = (self.about if about is None else about)[1 - slot.agent]
        phys = b.phys()
        if slot.depth == 1:
            return phys
        return np.outer(phys, b.social()).reshape(-1)

    def task_objective(self, i: int) -> bool:
        """``Config.objective="task"`` for robot i (its social goal one of
        ``objective_goals``): inside Table I's R_j its partner's estimate of its goal reward
        is its true goal reward -- the nested belief a point mass on its own goal."""
        cfg = self.cfg
        return (cfg.objective == "task"
                and int(self.goals[i]) in {G.parse_goal(g) for g in cfg.objective_goals})

    def nested_phys(self, o: int, about: Optional[list] = None) -> np.ndarray:
        """(PG,) the belief about robot o's physical goal that its partner's models hold
        inside robot o's level-2 reward (Table I's R_j estimates r(g_o) by it): the
        posterior ``about[o]`` -- or, ``Config.objective="task"``, robot o's true goal."""
        if self.task_objective(o):
            out = np.zeros(self.PG)
            out[self.own_goal[o]] = 1.0
            return out
        return (self.about if about is None else about)[o].phys()

    def _input_args(self, d: int, slot: BeliefSlot, rows: np.ndarray, keys: list,
                    terms: Optional[dict] = None, out: Optional[tuple] = None,
                    about: Optional[list] = None, flags: Optional[tuple] = None,
                    nested: Optional[np.ndarray] = None) -> tuple:
        """The arguments of :func:`kernels.inputs_typed` for the models ``rows`` of a slot.
        terms: the social coefficients to use instead of the models' own -- at depth 1
        {"partner_goal": (M,), "own_term": (M,), "race": (M,)} (goals.level1_term's a, b,
        c), at depth 2 {"partner_reward": (M, n_types, 2), "own_term": (M, n_types, 2),
        "partner_goal": (M, n_types), "race": (M, n_types)} (goals.level2_term's alpha,
        beta, nu, race) -- a key left out is zero: the rewards then hold the physical part
        and the terms named (:meth:`value_components`).  out: (P, p_row, R, pd, ho) to
        write into instead of the bank's (R indexed by the position in ``rows``).  about:
        the beliefs to read in place of ``self.about`` (beliefplan.py: a model re-solved at
        a belief the partner may come to hold).  flags: (social (M,), zero (M,)) in place
        of the bank's, and nested: (PG,) in place of :meth:`nested_phys` -- a model of the
        robot under goals the bank does not hold (beliefplan.py, ``Config.belief_partner=
        "inferred"``: the partner's root under a hypothesis about its goals, the slot's
        ``group`` its physical goal and ``keys`` its social goal)."""
        w, cfg, L0, L1, K = self.w, self.cfg, self.L0, self.L1, self.PG
        about = self.about if about is None else about
        bank = self.bank(d)
        z3, z2, z1 = np.zeros((1, 1, 2)), np.zeros((1, 1)), np.zeros(1)
        o, g = slot.agent, slot.group
        p = 1 - o
        b = self.planning_weights(slot, about)
        q = float(about[p].idle)
        # the partner's legal moves (where it idles) and the robot's own
        valid = (np.ascontiguousarray(w.valid_by[p]), np.ascontiguousarray(w.valid_by[o]))
        _, lo, hi = block_items(w.NN, self.reach_of(o, g, int(bank.block[rows[0]])), 0)
        M = rows.size
        common = (o, w.N, w.psucc, w.pev, self.tabs[o * K + g], self.tabs[2 * K + o * K + g],
                  w.cls_obj, w.at_partner_obj, w.cls_est, w.at_partner_est, w.rate_cls,
                  w.cost_c, w.site_of, w.goal_at_site, lo, hi)
        qown, kown = int(self.pg_obj[o, g]), int(self.pg_dst[o, g])
        social_f, zero_f = ((bank.social[rows], bank.zero[rows]) if flags is None
                            else (np.asarray(flags[0], dtype=np.int64),
                                  np.asarray(flags[1], dtype=np.int64)))
        models = (rows if out is None else np.arange(M, dtype=np.int64), int(g), qown, kown,
                  np.ascontiguousarray(social_f),
                  np.ascontiguousarray(zero_f).astype(np.uint8))
        goals = (self.pg_obj[p], self.pg_dst[p], self.pg_obj[o], self.pg_dst[o],
                 self.infer_object)
        P, p_row, R, pd, ho = ((bank.P, slot.p_row, bank.R, bank.pd, bank.ho) if out is None
                               else out)
        if d == 1:
            if terms is None:
                l1 = tuple(np.ascontiguousarray(x[rows]) for x in
                           (bank.coeff_partner, bank.coeff_own, bank.coeff_race))
            else:
                l1 = tuple(np.ascontiguousarray(np.asarray(terms.get(name, np.zeros(M)),
                                                           dtype=np.float64))
                           for name in ("partner_goal", "own_term", "race"))
            return (*common, L0.psi, p * K + np.arange(K), np.arange(K),
                    np.zeros(K, dtype=np.int64), K, 1, b * (1.0 - q), b * q,
                    float(cfg.idle_eps), *valid, w.busy, w.gone, *models, *l1, False, z3, z3,
                    z2, z2, z1, z1, z1, 0.0, w.fin_cls, self.lead,
                    L0.psi, np.zeros(1, dtype=np.int64), np.ones((K, 1)), np.zeros((K, 1)),
                    False, w.hold, self.done, self._no_carry, False, False,
                    self._no_carry_stack, *goals, P, p_row, R, pd, ho)
        hyp = self.hyp_goals[o]
        n_types = len(hyp)
        child_rows = np.array([L1.psi_row[self.hypothesis_row(p, int(k), int(t))]
                               for k, t in zip(slot.hyp_goal, slot.hyp_type)],
                              dtype=np.int64)
        check(bool((child_rows >= 0).all()), "a partner hypothesis without a policy row")
        if terms is None:
            cA = np.zeros((M, n_types, 2))       # of R_j, by "its physical goal is my own"
            cB = np.zeros((M, n_types, 2))       # of r(g_i)
            cN = np.zeros((M, n_types))          # of r(g_j)
            cR = np.zeros((M, n_types))          # of the race term
            for mm, goal in enumerate(keys):
                for t, partner_goal in enumerate(hyp):
                    for same in (0, 1):
                        al, be, nu, cr = self.level2_term(o, int(goal), partner_goal,
                                                          bool(same))
                        cA[mm, t, same], cB[mm, t, same] = al, be
                        cN[mm, t], cR[mm, t] = nu, cr
        else:
            cA = np.ascontiguousarray(terms.get("partner_reward", np.zeros((M, n_types, 2))))
            cB = np.ascontiguousarray(terms.get("own_term", np.zeros((M, n_types, 2))))
            cN = np.ascontiguousarray(terms.get("partner_goal", np.zeros((M, n_types))))
            cR = np.ascontiguousarray(terms.get("race", np.zeros((M, n_types))))
        ta1, tb1, tr1 = (np.array([self.level1_term(p, t)[n] for t in hyp])
                         for n in range(3))
        # the partner's (every level-1 model's) belief about robot o: about[o], the same
        # posterior whatever the model's own goal; its idle mode likewise.  Under
        # Config.objective="task" robot o's reward reads its true goal there (nested_phys)
        nested_belief = np.tile(self.nested_phys(o, about) if nested is None
                                else np.asarray(nested, dtype=np.float64), (K, 1))  # (PG, PG)
        nested_idle = np.full((K, K), float(about[o].idle))
        return (*common, L1.psi, child_rows, np.ascontiguousarray(slot.hyp_goal),
                np.ascontiguousarray(slot.hyp_type), K, n_types, b * (1.0 - q), b * q,
                float(cfg.idle_eps), *valid, w.busy, w.gone, *models, np.zeros(M), np.zeros(M),
                np.zeros(M),
                True, cA, cB, cN, cR, ta1, tb1, tr1, self.cost_in_partner_reward, w.fin_cls,
                self.lead, L0.psi, o * K + np.arange(K), nested_belief, nested_idle,
                bool(cfg.child_conditioning), w.hold, self.done, self._no_carry, False,
                False, self._no_carry_stack, *goals, P, p_row, R, pd, ho)

    def _term_coefficients(self, depth: int, slot: BeliefSlot, row: int) -> dict:
        """The social coefficients of model ``row`` of a slot, by term (the keys of
        :meth:`_input_args`' ``terms``), each shaped for one model; the terms that are zero
        throughout are left out."""
        bank = self.bank(depth)
        out = {}
        if depth == 1:
            for name, arr in (("partner_goal", bank.coeff_partner), ("own_term", bank.coeff_own),
                              ("race", bank.coeff_race)):
                v = np.array([float(arr[row])])
                if v[0] != 0.0:
                    out[name] = v
            return out
        o = slot.agent
        hyp = self.hyp_goals[o]
        n_types = len(hyp)
        goal = bank.key[row]                 # a level-2 model's key is its social goal
        cA, cB = np.zeros((1, n_types, 2)), np.zeros((1, n_types, 2))
        cN, cR = np.zeros((1, n_types)), np.zeros((1, n_types))
        for t, partner_goal in enumerate(hyp):
            for same in (0, 1):
                al, be, nu, cr = self.level2_term(o, int(goal), partner_goal, bool(same))
                cA[0, t, same], cB[0, t, same] = al, be
                cN[0, t], cR[0, t] = nu, cr
        for name, arr in (("partner_reward", cA), ("own_term", cB), ("partner_goal", cN),
                          ("race", cR)):
            if np.any(arr != 0.0):
                out[name] = arr
        return out

    # ---- the value of a robot's model, term by term ---------------------------------------------
    # The terms of a model's reward (Eq. 2), by key: what each stands for in the figure
    TERM_LABELS = {"own_goal": "own goal r(g_i)", "cost": "action costs -c(a_i)",
                   "partner_reward": "partner's reward R_j (x alpha)",
                   "own_term": "own goal in the term (x beta)",
                   "partner_goal": "partner's goal r(g_j) (x nu)", "race": "race term"}

    def value_components(self, i: int) -> Optional[dict]:
        """Robot i's root model, solved as it is, split by the terms of its reward: the
        value of every joint state it is solved on under its greedy policy (and its
        prediction of the partner, Eq. 4), each term's discounted sum apart.

        The reward of a model is a sum of terms (Eq. 2; ``kernels.inputs_typed``): its own
        goal reward r(g_i), its action costs, and its social term -- at level 1 w (a r(g_j)
        + b r(g_i) + c race), at level 2 w (alpha R_j + beta r(g_i) + nu r(g_j)) + race.
        The value V of a policy is linear in the reward, so V = sum of the terms' values,
        each the same discounted sum with only that term paid: ``parts[k] = (I - gamma
        P_pi)^-1 R_k``.  The physical part is the reward recomputed with the social
        coefficients zero (``_input_args``' ``terms``), its costs are -c(a) exactly, and
        each social term is the reward recomputed with that term's coefficients alone,
        less the physical part.  The policy is the model's greedy one from its solved
        values (``kernels.policy_graph``); its transition graph is solved by a sparse LU.

        -> dict(names, labels, total (S,), parts (K, S), policy (S,), residual) -- the
        states not solved (and the robots overlapping) hold NaN / -1; ``residual``: the
        largest |sum of parts - V| over the states solved (the solver's tolerance, and the
        rewards' single precision); None for a robot at level 0 (no reward table)."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        depth, row = self.root[i]
        if depth == 0:
            return None
        w, bank = self.w, self.bank(depth)
        slot = self.root_slot(i)
        rows, keys = np.array([row], dtype=np.int64), [bank.key[row]]
        S, NN, A = w.S, w.NN, N_ACTIONS
        # ---- the reward, term by term -----------------------------------------------------

        def scratch() -> tuple:              # (P, p_row, R, pd, ho) of one model, to write into
            return (np.zeros((1, S, A)), 0, np.zeros((1, S, A), dtype=np.float32),
                    np.zeros((1, w.B, w.K, w.N, A)), np.zeros((1, S, A)))

        out = scratch()
        inputs_typed(*self._input_args(depth, slot, rows, keys, terms={}, out=out))
        physical = out[2][0].astype(np.float64)                     # own goal - costs
        cost = -np.broadcast_to(w.cost_c[:, i, :][:, None, :], (w.B, NN, A)).reshape(S, A)
        rewards = {"own_goal": physical - cost, "cost": cost.copy()}
        for name, coeff in self._term_coefficients(depth, slot, row).items():
            out = scratch()
            inputs_typed(*self._input_args(depth, slot, rows, keys, terms={name: coeff},
                                           out=out))
            rewards[name] = out[2][0].astype(np.float64) - physical
        # ---- the policy and its graph -----------------------------------------------------
        blocks = self.solved_blocks(bank, row)
        _, lo, hi = block_items(NN, blocks, 0)
        pol = np.full(S, -1, dtype=np.int64)
        edge_n = np.zeros(S, dtype=np.int64)
        edge_s2 = np.zeros((S, 2 * A), dtype=np.int64)
        edge_pr = np.zeros((S, 2 * A))
        o = int(bank.agent[row])
        policy_graph(bank.R, bank.P, bank.V, row, int(bank.p_of[row]), o, int(bank.social[row]),
                     bool(bank.zero[row]), self.tabs[bank.var[row]], self.tabs[bank.svar[row]],
                     lo, hi, w.N, w.psucc, w.psucc_T, w.pev, w.pev_T, self.valid_by, w.busy,
                     self.useok,
                     w.uctx, w.site_of, w.goal_at_site, bank.pd, bank.ho, self.infer_object,
                     self.done, w.vdone, w.hold, w.gone, w.state_ok, self.gamma, pol, edge_n,
                     edge_s2, edge_pr)
        states = np.flatnonzero(pol >= 0)
        local = np.full(S, -1, dtype=np.int64)
        local[states] = np.arange(states.size)
        n = int(edge_n[states].max()) if states.size else 0
        src = np.repeat(states, n)
        dst = edge_s2[states, :n].reshape(-1)
        pr = edge_pr[states, :n].reshape(-1)
        keep = np.arange(n)[None, :] < edge_n[states][:, None]
        keep = keep.reshape(-1)
        src, dst, pr = src[keep], dst[keep], pr[keep]
        inside = dst >= 0
        if inside.any():
            check(bool((local[dst[inside]] >= 0).all()),
                  "a successor outside the blocks the model is solved on")
        P = sp.csr_matrix((pr[inside], (local[src[inside]], local[dst[inside]])),
                          shape=(states.size, states.size))
        system = (sp.identity(states.size, format="csc") - self.gamma * P).tocsc()
        lu = spl.splu(system)
        names = list(rewards)
        parts = np.full((len(names), S), np.nan)
        done_mass = np.bincount(local[src[~inside]], weights=pr[~inside], minlength=states.size)
        for k, name in enumerate(names):
            rhs = rewards[name][states, pol[states]]
            if name == "own_goal":               # a block worth vdone: its goal's value
                rhs = rhs + self.gamma * w.vdone * done_mass
            parts[k, states] = lu.solve(rhs)
        total = np.full(S, np.nan)
        total[states] = bank.V[row][states]
        residual = float(np.abs(parts[:, states].sum(axis=0) - total[states]).max()) \
            if states.size else 0.0
        # a term that pays nothing anywhere is left out; a hinderer without a goal of its own
        # has its own goal reward taken away by its term (beta = -1): the two cancel, and
        # neither is shown
        # (no state solved -- the robot's model has nothing left to plan -- keeps every term:
        # all NaN, as V)
        scale = max(1.0, float(np.nanmax(np.abs(total))) if states.size else 1.0)
        drop = ({k for k in names
                 if np.abs(parts[names.index(k)][states]).max() <= 1e-9 * scale}
                if states.size else set())
        if states.size and "own_goal" in names and "own_term" in names:
            both = parts[names.index("own_goal")] + parts[names.index("own_term")]
            if np.abs(both[states]).max() <= 1e-6 * scale:
                drop |= {"own_goal", "own_term"}
        keep = [k for k, name in enumerate(names) if name not in drop]
        names = [names[k] for k in keep]
        return dict(names=names, labels=[self.TERM_LABELS[k] for k in names], total=total,
                    parts=parts[keep], policy=pol, residual=residual)

    def value_maps(self, i: int, state) -> Optional[dict]:
        """:meth:`value_components` of robot i read off at every cell it could stand on,
        with its partner and the objects where they are in ``state``: dict(names, labels,
        total (N,), parts (K, N), action (N,), residual, partner_cell) -- NaN / -1 where it
        cannot stand (the partner's cell, a cell outside its room, a state not solved; every
        cell but its own while it is lifting an object, which happens where the object
        lies); None at level 0."""
        parts = self.value_components(i)
        if parts is None:
            return None
        w = self.w
        c, xp = int(state[2]), int(state[1 - i])
        cells = np.arange(w.N)
        pair = cells * w.N + xp if i == 0 else xp * w.N + cells
        s = c * w.NN + pair
        ok = (w.state_ok[c, pair] != 0) & (w.valid_by[i][cells].any(axis=1))
        if w.busy[c, i]:
            ok &= cells == int(state[i])
        total = np.where(ok, parts["total"][s], np.nan)
        by_term = np.where(ok[None, :], parts["parts"][:, s], np.nan)
        action = np.where(ok, parts["policy"][s], -1)
        return dict(names=parts["names"], labels=parts["labels"], total=total, parts=by_term,
                    action=action, residual=parts["residual"], partner_cell=xp)

    def _extrapolate(self, bank: ModelBank, rows: np.ndarray) -> None:
        """Warm start V + lam (V - V_prev) with lam the contraction seen so far (exact: the
        solve still runs to its certificate)."""
        tol = 10.0 * self.cfg.value_tol
        for r in rows:
            if bank.n_solves[r] < 2:
                continue
            d = bank.V[r] - bank.V_prev[r]
            n1 = float(np.abs(d).max())
            lam = min(1.0, n1 / bank.step_norm[r]) if bank.step_norm[r] > 0.0 else 0.0
            bank.step_norm[r] = n1
            bank.V_prev[r] = bank.V[r]
            if n1 > tol and lam > 0.0:
                bank.V[r] += lam * d

    def solve(self, depth: int) -> np.ndarray:
        """Solve the active models of one depth -- all of them in one parallel batch where
        the memory allows -- and store the Eq. 7 policies of the level-1 partner
        hypotheses.  -> the rows solved."""
        bank = self.bank(depth)
        rows = np.nonzero(bank.active)[0]
        if not rows.size:
            return rows
        if self.cfg.extrapolate:
            self._extrapolate(bank, rows)
            first = rows[bank.n_solves[rows] < 2]
            bank.V_prev[first] = bank.V[first]
        bank.n_solves[rows] += 1
        for g0 in range(0, rows.size, self.batch):
            grp = np.ascontiguousarray(rows[g0:g0 + self.batch])
            st = self._solve(bank.V, bank.R, bank, grp,
                             [self.solved_blocks(bank, int(r)) for r in grp])
            bank.pi_iters[grp], bank.krylov_iters[grp] = st[:, 0], st[:, 1]
            bank.residual[grp], bank.ok[grp] = st[:, 2], st[:, 3] > 0.5
        hyp = rows[bank.psi_row[rows] >= 0]
        if hyp.size:
            hyp = np.ascontiguousarray(hyp)
            parts = [block_items(self.w.NN, self.reach_of(int(bank.agent[r]), int(bank.goal[r]),
                                                          int(bank.block[r])), n)
                     for n, r in enumerate(hyp.tolist())]
            it_m, it_lo, it_hi = (np.concatenate([pt[k] for pt in parts]) for k in range(3))
            softmax_policies(bank.V, bank.R, bank.P, bank.p_of, bank.agent, bank.var, bank.svar,
                             bank.social, bank.zero, hyp, np.ascontiguousarray(bank.psi_row[hyp]),
                             it_m, it_lo, it_hi, *self._kernel_args(bank), self.read_inv_tau,
                             bank.psi)
        return rows

    # ---- per-state quantities ----------------------------------------------------------------
    def legal(self, agent: int, social: bool, s: int) -> np.ndarray:
        """(10,) bool: the actions robot ``agent`` may choose at joint state s."""
        w = self.w
        c, pair = divmod(int(s), w.NN)
        x = pair // w.N if agent == 0 else pair % w.N
        out = np.empty(N_ACTIONS, dtype=bool)
        out[:PICK] = w.valid_by[agent][x] if w.busy[c, agent] == 0 else False   # lifting
        bits = int(self.useok[int(social), agent, c, w.uctx[agent, pair]])
        out[PICK:] = [(bits >> (a - PICK)) & 1 for a in range(PICK, N_ACTIONS)]
        return out

    def q_values(self, model: Tuple[int, int], s: int, agent: int) -> np.ndarray:
        """Q of ``model = (depth, row)`` at joint state s -> (10,), -inf where illegal."""
        depth, r = model
        w, g = self.w, self.gamma
        c, pair = divmod(int(s), w.NN)
        j = JOINT[agent]                                           # (10 own, 10 partner)
        ev, n2 = w.pev[pair][j], w.psucc[pair][j].astype(np.int64)
        bank = self.bank(depth)
        social = bool(bank.social[r])
        c2 = self.tabs[bank.var[r]][c, ev].astype(np.int64)         # (10, 10)
        v = (np.where(w.done[c2, agent] != 0, w.vdone, bank.V[r][c2 * w.NN + n2])
             if bank.zero[r] else bank.V[r][c2 * w.NN + n2])
        if depth == 0:                                           # reward recomputed
            st = w.unsid(s)
            xp = st[1 - agent]
            if w.busy[c, 1 - agent]:             # lifting: it goes on with the lift
                pl = np.zeros(N_ACTIONS)
                pl[PICK] = 1.0
            else:
                pl = np.concatenate([w.valid_by[1 - agent][xp].astype(float),
                                     np.zeros(N_ACTIONS - PICK)])
                pl /= pl.sum()
            gl = int(bank.goal[r])
            obj, dst = int(self.pg_obj[agent, gl]), int(self.pg_dst[agent, gl])
            cg = self.level0_reward_tabs[(agent, obj)][c, ev].astype(np.int64)
            no = n2 // w.N if agent == 0 else n2 % w.N
            if w.at_partner_obj[:, agent, obj].any():    # held_reward="carrier": the reward
                no = np.where(w.at_partner_obj[cg, agent, obj] != 0,          # follows the object
                              n2 % w.N if agent == 0 else n2 // w.N, no)
            placing = (w.hold[c, agent] == obj) and (w.done[c, agent] == 0)
            cls = w.cls_obj[cg, agent, obj]
            if placing:                          # its own set-down: the done reward
                cls = np.where((np.arange(N_ACTIONS) == PLACE)[:, None]
                               & (no == int(w.dest[dst])), 0, cls)
            rew = w.rate_cls[cls, no, dst] - w.cost_c[c, agent][:, None]
            q = ((rew + g * v) * pl[None, :]).sum(1)
        else:
            pl = bank.P[bank.p_of[r], s]
            q = bank.R[r, s].astype(np.float64) + g * (v * pl[None, :]).sum(1)
            x0, x1 = divmod(pair, w.N)
            gc = int(w.goal_at_site[w.site_of[x0 if agent == 0 else x1]]) if social else -1
            hoon = self.infer_object and social
            xp = x1 if agent == 0 else x0
            for a in range(PICK, N_ACTIONS):     # an act for the partner (place, give)
                cd = self.tabs[bank.svar[r]][c, ev[a]].astype(np.int64)             # (10,)
                fail = c2[a]
                if gc >= 0 and not hoon:     # a set-down for the partner: counts with mass wd
                    wd = np.where(cd != fail, np.minimum(bank.pd[bank.p_of[r], c, gc, xp], pl),
                                  0.0)
                elif hoon:                   # ... or a hand-over, under the hypotheses that
                    wd = np.zeros(N_ACTIONS)     # take its object
                    xh = int(w.hold[c, agent])
                    for ap in np.flatnonzero(cd != fail).tolist():
                        if (gc >= 0 and xh >= 0 and w.gone[cd[ap], xh] != 0
                                and w.gone[fail[ap], xh] == 0):      # a set-down, counted
                            mass = bank.pd[bank.p_of[r], c, gc, xp, ap]
                        else:
                            mass = float(bank.ho[bank.p_of[r], s, ap])
                        wd[ap] = min(mass, pl[ap])
                else:
                    continue
                q[a] += g * float((wd * (bank.V[r][cd * w.NN + n2[a]] - v[a])).sum())
        return np.where(self.legal(agent, social, s), q, -np.inf)

    def execution_policy(self, i: int, state, q: np.ndarray) -> np.ndarray:
        """The distribution robot i draws from under ``action_selection="sample"``: Eq. 7's
        softmax at robot i's own tau."""
        return np.exp(log_softmax(q / self.taus[i]))

    # ---- beliefs: reading them ---------------------------------------------------------------
    def root_slot(self, i: int) -> Optional[BeliefSlot]:
        depth = self.root[i][0]
        return None if depth == 0 else self.slots[(i, depth, self.own_goal[i])]

    def goal_belief(self, i: int) -> np.ndarray:
        """(K,) robot i's belief about its partner's destination (uniform at level 0)."""
        if self.levels[i] == 0:
            return np.full(self.w.K, 1.0 / self.w.K)
        return np.bincount(self.pg_dst[1 - i], weights=self.about[1 - i].phys(),
                           minlength=self.w.K)

    def object_belief(self, i: int) -> np.ndarray:
        """(M,) robot i's belief about which object its partner wants (uniform at level 0;
        with the partner's object known, all of it on the partner's own)."""
        if self.levels[i] == 0:
            return np.full(self.w.M, 1.0 / self.w.M)
        return np.bincount(self.pg_obj[1 - i], weights=self.about[1 - i].phys(),
                           minlength=self.w.M)

    def social_belief(self, i: int) -> Optional[np.ndarray]:
        """(len(SOCIAL_GOALS),) a level-2 robot's belief about its partner's social goal
        (None below level 2)."""
        b = self.about[1 - i].social()
        if b is None:
            return None
        out = np.zeros(len(G.SOCIAL_GOALS))
        out[list(self.hyp_goals[i])] = b
        return out

    def idle_belief(self, i: int) -> Optional[float]:
        """Robot i's belief that its partner is idle (None at level 0)."""
        if self.levels[i] == 0:
            return None
        return float(self.about[1 - i].idle)

    def observed_actions(self, actions, s: int, nxt, robot: int) -> tuple:
        """What robot ``robot`` is seen to have done in the joint transition: the action it
        chose (Algorithm 1 takes the joint action as observed)."""
        return (int(actions[robot]),)

    # ---- beliefs: the likelihoods ------------------------------------------------------------
    def _log_policy(self, model: Tuple[int, int], s: int, robot: int) -> np.ndarray:
        """(10,) log of Eq. 7's policy of ``model`` (a model of robot ``robot``) at joint
        state s, as the likelihood of the belief update reads it: softmax(Q / tau) at the tau
        robot ``robot`` is read at -- its partner's, the robot whose belief this likelihood
        updates -- or, under ``Config.inference_temperature`` (since 2026-11-04), at that
        tau; -inf where illegal.  (The robots' own planning reads the models at Eq. 7's
        tau, from the kernels' stored policies.)"""
        return self._q_and_log_policy(model, s, robot)[1]

    def _q_and_log_policy(self, model: Tuple[int, int], s: int,
                          robot: int) -> Tuple[np.ndarray, np.ndarray]:
        """(Q, log policy) of :meth:`_log_policy`, both (10,), -inf where illegal."""
        q = self.q_values(model, s, robot)
        return q, log_softmax(q / self.inference_taus[1 - robot])

    def lapsed(self, log_pol: np.ndarray) -> np.ndarray:
        """(10,) the likelihood a policy gives each action under the lapse
        (``Config.belief_lapse`` = e): log((1 - e) psi(a) + e / |A|), A the actions the
        policy allows (its finite entries), -inf elsewhere as before.  e = 0: ``log_pol``."""
        e = float(self.cfg.belief_lapse)
        if e <= 0.0:
            return log_pol
        allowed = np.isfinite(log_pol)
        n = int(allowed.sum())
        if n == 0:
            return log_pol
        with np.errstate(divide="ignore"):
            return np.where(allowed, np.logaddexp(np.log1p(-e) + log_pol, np.log(e / n)),
                            -np.inf)

    def physical_log_likelihood(self, j: int, s: int, a: int,
                                detail: Optional[dict] = None) -> np.ndarray:
        """(PG,) log psi^0_{j,g}(a | s) of robot j's level-0 model under each of its physical
        goals: the likelihood of Eq. 6 (the authors' physical-goal estimation); under the
        lapse (``Config.belief_lapse``), :meth:`lapsed`.  ``detail`` (a dict): each model's Q
        at s goes into its "level0" entry (``Config.belief_detail``)."""
        K = self.PG
        qs, lps, out = [], [], np.full(K, -np.inf)
        for g in range(K):
            q, lp = self._q_and_log_policy((0, j * K + g), s, j)
            out[g] = self.lapsed(lp)[a]
            qs.append(q)
            lps.append(lp)
        if detail is not None:
            detail["level0"] = dict(_model_summary(np.array(qs), np.array(lps), a),
                                    log_lik=out.copy())
        return out

    def hypothesis_log_policies(self, j: int, s: int, a: int, log_phys: np.ndarray,
                                detail: Optional[dict] = None) -> np.ndarray:
        """(PG, n_types) log psi^1_{j,(g,t)}(a | s) of robot j's level-1 hypothesis models
        (-inf for a physical goal the posterior ``log_phys`` has ruled out: its models are
        not kept); under the lapse (``Config.belief_lapse``), :meth:`lapsed`.  Both posteriors
        read them: Eq. 6 averaged over the social posterior, Eq. 5 over the physical one.
        ``detail`` (a dict): every model's Q at s goes into its "models" entry, (PG, n_types,
        10), NaN for the physical goals ruled out (``Config.belief_detail``)."""
        hyp = self.hyp_goals[1 - j]
        rows: dict = {}
        out = np.full((self.PG, len(hyp)), -np.inf)
        rec = detail is not None
        q_all = np.full((self.PG, len(hyp), N_ACTIONS), np.nan) if rec else None
        lp_all = np.full((self.PG, len(hyp), N_ACTIONS), np.nan) if rec else None
        over = self.likelihood_override.get(j)   # Config.belief_inference="level2"
        for g in range(self.PG):
            if not np.isfinite(log_phys[g]):
                continue
            for t in range(len(hyp)):
                if over is not None:
                    q, lp = over[0][g, t], over[1][g, t]
                    if not np.isfinite(lp).any():   # not solved: ruled out when it was built
                        continue
                    out[g, t] = self.lapsed(lp)[a]
                    if rec:
                        q_all[g, t], lp_all[g, t] = q, lp
                    continue
                r = self.hypothesis_row(j, g, t)
                if r not in rows:
                    q, lp = self._q_and_log_policy((1, r), s, j)
                    rows[r] = (q, lp, self.lapsed(lp))
                out[g, t] = rows[r][2][a]
                if rec:
                    q_all[g, t], lp_all[g, t] = rows[r][0], rows[r][1]
        if rec:
            detail["models"] = dict(_model_summary(q_all, lp_all, a), log_lik=out.copy(),
                                    level=1 if over is None else 2)
        return out

    def social_log_likelihood(self, log_phys: np.ndarray, log_pol: np.ndarray) -> np.ndarray:
        """(n_types,) log of Eq. 5's likelihood of robot j's action under each social goal t
        held possible: sum_g b_g(g) psi^1_{j,(g,t)}(a | s), the level-1 hypothesis policies
        ``log_pol`` (PG, n_types) averaged over the physical posterior ``log_phys`` (the one
        before this update)."""
        with np.errstate(invalid="ignore"):
            joint = log_phys[:, None] + log_pol
        return np.array([_log_sum(joint[:, t]) for t in range(log_pol.shape[1])])

    def _log_idle(self, j: int, s: int, a: int) -> float:
        """log of the idle mode's probability of robot j's action: staying put (its
        ``Config.idle_eps`` share of moves), or lifting on while it lifts."""
        st = self.w.unsid(s)
        if self.w.busy[st[2], j]:
            return 0.0 if a == PICK else -np.inf
        return float(self.log_idle[j][st[j], a])

    def transition_term(self, j: int, s: int, actions, nxt, event: dict) -> np.ndarray:
        """(PG,) bool, Eq. 5's transition term ``T(s^t | s^{t-1}, a_J)`` for the hypotheses
        about robot j's physical goal: True where the hypothesis predicts what the hands did
        in this transition, False where it predicted otherwise.  The hands' outcomes that
        depend on the hidden goal, each under the world's rule (robot j's goal is its own
        object on its own destination, and nothing else is handed to it or set down for it):

        * robot j finished (``done_now``): its goal is (its own object, that cell);
        * robot j's own set-down of its own object on a candidate that did nothing: not
          that cell;
        * its partner's set-down of an object on a candidate that counted (robot j done)
          or failed (nothing happened): (that object, that cell), or not;
        * its partner's hand-over of an object robot j took, or did not: that object, or
          not that object.
        """
        w = self.w
        i = 1 - j
        obj, dst = self.pg_obj[j], self.pg_dst[j]              # (PG,) of each hypothesis
        st = w.unsid(s)
        keep = np.ones(self.PG, dtype=bool)
        if not self.cfg.condition_on_placement:
            return keep
        held_i, held_j = int(w.hold[st[2], i]), int(w.hold[st[2], j])
        done_now = bool(event["done_now"][j])
        if done_now:                                             # (own object, this cell)
            keep &= (obj == j) & (w.dest[dst] == int(w.true_dest[j]))
            return keep
        # robot j's own place of its own object on a candidate cell that did nothing: it is
        # not j's destination (a place that counted is done_now above)
        if (int(actions[j]) == PLACE and held_j == j and int(w.hold[nxt[2], j]) == j
                and w.iscand[w.site_of[int(nxt[j])]]):
            keep &= ~((obj == j) & (w.dest[dst] == int(nxt[j])))
        # robot i's set-down for j that failed: not that object on that cell
        if bool(event["failed"][i]) and held_i >= 0:
            keep &= ~((obj == held_i) & (w.dest[dst] == int(nxt[i])))
        # robot i's hand-over: robot j took the object (it is j's own), or did not
        if int(actions[i]) == GIVE and held_i >= 0:
            if int(w.hold[nxt[2], j]) == held_i:
                keep &= obj == held_i
            elif bool(event.get("handed_in_vain", (False, False))[i]):
                keep &= obj != held_i
        return keep

    def _stage(self, j: int, s: int) -> bool:
        """``Config.belief_stages``: are robot j's hands empty at s?  Then its actions are
        evidence about its object (and social goal), not its destination; holding an
        object, about its destination (and social goal), not its object."""
        return int(self.w.hold[self.w.unsid(s)[2], j]) < 0

    def _holds_next(self, j: int, nxt) -> bool:
        """Will robot j's next action be read with an object in its hands (the next update's
        stage)?  ``nxt`` is the joint state just reached, as a tuple: robot j holds an object
        there, or it lifts one (busy -- the update after the lift finds the object in its
        hands)."""
        return int(self.w.hold[nxt[2], j]) >= 0 or bool(self.w.busy[nxt[2], j])

    def hazard_phys(self, j: int, log_phys: np.ndarray, h: float, objects: bool) -> np.ndarray:
        """The physical posterior about robot j mixed toward the uniform with weight h
        (Config.goal_hazard, Config.hand_hazard) -- under Config.belief_stages one factor of
        it: the object (``objects``; the destination given each object stays as it is) or the
        destination given each object (the object's mass stays as it is), the factor the
        partner's actions speak to in that stage.  Without the stages, the whole (object,
        destination) posterior.  The hypotheses ruled out stay out."""
        if h <= 0.0:
            return log_phys
        log_phys = normalize_log(np.asarray(log_phys, dtype=np.float64))
        if self.cfg.belief_factored:     # the two posteriors apart: mix the one that learns
            lo, ld = self.factors(j, log_phys)
            both = not self.cfg.belief_stages
            if objects or both:
                lo = mix_uniform(lo, h)
            if not objects or both:
                ld = mix_uniform(ld, h)
            return self.product(j, lo, ld)
        if not self.cfg.belief_stages:
            return mix_uniform(log_phys, h)
        obj = self.pg_obj[j]
        out = log_phys.copy()
        groups = np.unique(obj)
        mass = np.array([_log_sum(log_phys[obj == q]) for q in groups])
        if objects:                      # the object's mass, the destination given it kept
            new_mass = mix_uniform(mass, h)
            for q, m, m_new in zip(groups, mass, new_mass):
                if np.isfinite(m):
                    sel = obj == q
                    out[sel] = log_phys[sel] - m + m_new
        else:                            # the destination given each object, its mass kept
            for q, m in zip(groups, mass):
                if np.isfinite(m):
                    sel = obj == q
                    out[sel] = mix_uniform(log_phys[sel] - m, h) + m
        return out

    def factors(self, j: int, log_phys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The two physical posteriors about robot j (``Config.belief_factored``): the
        belief about its object, (M,), and the one about its destination, (K,) -- each the
        marginal of the (object, destination) belief ``log_phys`` (PG,), in logs, -inf where
        nothing is left.  With the partner's object known, the object's is all on its own."""
        obj, dst = self.pg_obj[j], self.pg_dst[j]
        lo = np.full(self.w.M, -np.inf)
        ld = np.full(self.w.K, -np.inf)
        for q in np.unique(obj):
            lo[q] = _log_sum(log_phys[obj == q])
        for d in np.unique(dst):
            ld[d] = _log_sum(log_phys[dst == d])
        return normalize_log(lo), normalize_log(ld)

    def product(self, j: int, log_obj: np.ndarray, log_dst: np.ndarray) -> np.ndarray:
        """(PG,) the belief over robot j's physical goals of the two posteriors:
        b(o, d) = b_O(o) b_D(d), in logs."""
        with np.errstate(invalid="ignore"):
            return normalize_log(log_obj[self.pg_obj[j]] + log_dst[self.pg_dst[j]])

    def factored_posterior(self, j: int, post: np.ndarray, hard: np.ndarray,
                           empty: bool, alpha: float = 1.0) -> np.ndarray:
        """``Config.belief_factored``: the two posteriors about robot j after one action --
        ``post`` (PG,), unnormalised log, the prior times the likelihood of the action and
        the transition term; ``hard`` the prior times the transition term alone.  The
        posterior that learns in this stage is the marginal of ``post`` -- while robot j's
        hands are empty the one about its object, b_O'(o) ∝ b_O(o) sum_d b_D(d) psi_(o,d)(a),
        holding an object the one about its destination, b_D'(d) ∝ b_D(d) sum_o b_O(o)
        psi_(o,d)(a): the likelihood of the action under each, the other averaged out -- and
        the frozen one is the marginal of ``hard``: as it was, but for what the hands'
        outcomes ruled out.  Without the stages (``Config.belief_stages=False``) both
        learn.  ``alpha`` (``Config.likelihood_power``): the learning factor's likelihood
        taken to that power, b' ∝ b_hard (b_post / b_hard)^alpha.  -> (PG,) the product,
        normalised."""
        stages = self.cfg.belief_stages
        lo_post, ld_post = self.factors(j, post)
        lo_hard, ld_hard = self.factors(j, hard)
        if alpha != 1.0:                         # each factor's likelihood ratio ^ alpha
            with np.errstate(invalid="ignore"):
                lo_post = normalize_log(np.where(np.isfinite(lo_post),
                                                 (1.0 - alpha) * lo_hard + alpha * lo_post,
                                                 -np.inf))
                ld_post = normalize_log(np.where(np.isfinite(ld_post),
                                                 (1.0 - alpha) * ld_hard + alpha * ld_post,
                                                 -np.inf))
        lo = lo_post if (empty or not stages) else lo_hard
        ld = ld_post if (not empty or not stages) else ld_hard
        return self.product(j, lo, ld)

    def cut_log_likelihood(self, j: int, s: int, a: int, prior: np.ndarray,
                           old: PartnerBelief, log_pol: np.ndarray,
                           detail: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """``Config.physical_likelihood="cut"`` (2026-11-04, :mod:`social_pairings.subgoals`):
        (PG,) log L(g) = log(pi N_g(a) + (1 - pi) S(a)) of robot j's action a at joint
        state s, for the physical posterior in Eq. 6's place (the idle mode is mixed in
        after, as for every likelihood).

        N_g(a): robot j's level-1 "none" model with physical goal g (its own task; from
        ``log_pol``, at the likelihood's tau and lapse).  From the predictive belief, before
        the action: omega(xi, z) = sum_g b(g) b(xi) P(z | g, xi), the share of its own task
        pi = sum_xi omega(xi, own), and the social background S(a) = sum_{xi, z != own}
        omega(xi, z) kappa_z(a | s) / (1 - pi) -- the same for every g, since no subgoal's
        policy kappa_z depends on robot j's own goal.  -> (log L, what the subgoal update
        reuses)."""
        sg = self.subgoals[j]
        w = self.w
        st = w.unsid(s)
        legal = self.legal(j, True, s)
        cost = np.asarray(w.cost_c[st[2], j], dtype=np.float64)
        kappa = sg.policies(st, legal, cost)                            # (L, 10)
        k_a = kappa[:, a]                                               # (L,)
        b_g = np.exp(normalize_log(prior))                              # (PG,)
        b_xi = old.social()                                             # (n_types,)
        P = old.options                                                 # (PG, n_types, 1 + L)
        omega = np.einsum("g,t,gtz->tz", b_g, b_xi, P)                  # (n_types, 1 + L)
        pi = float(min(max(omega[:, 0].sum(), 0.0), 1.0))
        rest = 1.0 - pi
        S = float((omega[:, 1:] * k_a[None, :]).sum() / rest) if rest > 1e-15 else 0.0
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            nav = np.exp(log_pol[:, self.none_index[j]])                # (PG,) N_g(a)
            nav = np.where(np.isfinite(nav), nav, 0.0)
            raw = np.log(pi * nav + rest * S)
        raw = np.where(np.isfinite(prior), raw, -np.inf)
        cut = dict(k_a=k_a, nav=nav, pi=pi, S=S)
        if detail is not None:
            detail["cut"] = dict(pi=pi, background=S, nav=nav.copy(), log_lik=raw.copy(),
                                 subgoals=[sg.name(n) for n in range(sg.L)],
                                 kappa=k_a.copy(), subgoal_mass=omega[:, 1:].sum(0))
        return raw, cut

    def updated_subgoals(self, j: int, s: int, a: int, nxt, old: PartnerBelief, cut: dict,
                         q0: float, l_idle: float, detail: Optional[dict] = None) -> np.ndarray:
        """``Config.physical_likelihood="cut"``: robot j's subgoal belief after its action a
        -- within each physical goal g and social goal xi, P(z | g, xi) times the likelihood
        of a under z (its own task: N_g(a); a subgoal: kappa_z(a | s); the idle mode mixed
        in), normalised (a row nothing explains keeps its prior) -- then the subgoal
        dynamics to ``nxt`` (:meth:`Subgoals.predict`): an act that ends a subgoal, one no
        longer possible, a new one from the prior chi(z | g, xi, s') under robot j's belief
        about its partner's goal (``about[1 - j]``).  -> (PG, n_types, 1 + L)."""
        sg = self.subgoals[j]
        w = self.w
        st = w.unsid(s)
        P = old.options
        psi = np.empty_like(P)
        psi[:, :, 0] = cut["nav"][:, None]
        psi[:, :, 1:] = cut["k_a"][None, None, :]
        idle = float(np.exp(l_idle)) if np.isfinite(l_idle) else 0.0
        psi = (1.0 - q0) * psi + q0 * idle
        post = P * psi
        total = post.sum(-1, keepdims=True)
        post = np.where(total > 0.0, post / np.where(total > 0.0, total, 1.0), P)
        nxt = tuple(int(v) for v in nxt)
        ended = sg.acted(st, a) | ~sg.feasible(nxt)
        reader = self.about[1 - j]                  # robot j's belief about its partner
        chi = sg.prior(nxt, self.pg_obj[j], self.pg_dst[j], reader.phys(),
                       self.pg_obj[1 - j], self.pg_dst[1 - j])
        out = sg.predict(post, chi, ended)
        if detail is not None:
            b_g = np.exp(normalize_log(old.log_phys))
            mass = np.einsum("g,t,gtz->z", b_g, old.social(), out)
            detail["subgoal_belief"] = dict(own_task=float(mass[0]), subgoals=mass[1:].copy(),
                                            ended=ended.copy())
        return out

    def windowed_posterior(self, j: int, win: EvidenceWindow, post: Optional[np.ndarray],
                           hard: np.ndarray, keep: np.ndarray, empty: bool
                           ) -> Tuple[np.ndarray, EvidenceWindow]:
        """``Config.belief_window`` n >= 1 (2026-11-04): the two physical posteriors about
        robot j built afresh from the window -- the uniform prior times the likelihoods of
        the last n transitions of each one's own stage:

            b_O(o) proportional to prod_{t in W_O} l_t(o)^alpha   (the last n, hands empty),
            b_D(d) proportional to prod_{t in W_D} l_t(d)^alpha   (the last n, holding),

        both from every transition without the stages; ``alpha`` = ``Config.likelihood_power``.
        This transition's likelihood comes from ``post`` (the prior x the action's likelihood
        x the transition term, unnormalised log; None where nothing explains the action, which
        then adds nothing) against ``hard`` (the prior x the transition term): the learning
        factor's, the other factor averaged out under the belief as it is now
        (:func:`_group_log_lik`), as :meth:`factored_posterior` takes it.  What the hands'
        outcomes ruled out (``keep``, this transition's term; ``win.ruled_out``, every one
        before) stays out for good -- of the product, whose two marginals are the posteriors
        (multiplied back, the two posteriors apart as everywhere else).  Without anything
        left of the window's product, the hands' outcomes alone.  With n at least the number
        of transitions, no hazard and nothing ruled out, this is :meth:`factored_posterior`
        transition by transition (Bayes' rule).  -> (log_phys (PG,), the window after)."""
        cfg = self.cfg
        n, alpha = int(cfg.belief_window), float(cfg.likelihood_power)
        stages = cfg.belief_stages
        obj, dst = self.pg_obj[j], self.pg_dst[j]
        ruled = win.ruled_out | ~np.asarray(keep, dtype=bool)
        if ruled.all():                          # nothing left: this transition's term is not
            ruled = win.ruled_out.copy()         # taken (the prior stands, as without a window)
        objects, destinations = win.objects, win.destinations
        if post is not None:                     # this transition's evidence, in its stage
            k = int(win.updates)
            if empty or not stages:
                objects = (objects + ((k, _group_log_lik(post, hard, obj, self.w.M)),))[-n:]
            if not empty or not stages:
                destinations = (destinations
                                + ((k, _group_log_lik(post, hard, dst, self.w.K)),))[-n:]
        lo0, ld0 = self.factors(j, np.full(self.PG, -np.log(self.PG)))   # the uniform prior
        lo = lo0 + alpha * np.sum([v for _, v in objects], axis=0) if objects else lo0
        ld = ld0 + alpha * np.sum([v for _, v in destinations], axis=0) if destinations else ld0
        with np.errstate(invalid="ignore"):
            joint = np.where(ruled, -np.inf, lo[obj] + ld[dst])
            if not np.isfinite(joint).any():     # the window against the hands' outcomes:
                joint = np.where(ruled, -np.inf, lo0[obj] + ld0[dst])     # those alone
        lo_f, ld_f = self.factors(j, normalize_log(joint))
        return (self.product(j, lo_f, ld_f),
                EvidenceWindow(ruled, objects, destinations, int(win.updates) + 1))

    def _window_record(self, j: int, win: EvidenceWindow) -> dict:
        """The window after an update, for the record (``Config.belief_window``,
        ``Config.belief_detail``): its length, what is ruled out, and each stage's buffer --
        the update (transition) each likelihood is of, and the likelihood by object or by
        destination (log, up to a constant)."""
        return dict(n=int(self.cfg.belief_window), updates=int(win.updates),
                    ruled_out=win.ruled_out.copy(),
                    objects=[dict(update=int(k), log_lik=v.copy()) for k, v in win.objects],
                    destinations=[dict(update=int(k), log_lik=v.copy())
                                  for k, v in win.destinations])

    def staged_log_likelihood(self, j: int, log_phys: np.ndarray, log_lik: np.ndarray,
                              empty: bool) -> np.ndarray:
        """``Config.belief_stages``: ``log_lik`` (PG,) of the physical hypotheses with, while
        robot j's hands are empty, the destination averaged out -- the hypotheses that
        share an object get their group's mean likelihood under the belief, so the belief
        about the destination is updated under every destination alike and only the belief
        about the object moves.  Holding an object: as it is (the update is then
        normalised object by object, :meth:`_renormalize_stage`, so that only the belief
        about the destination moves)."""
        if not self.cfg.belief_stages or not empty:
            return log_lik
        out = log_lik.copy()
        obj = self.pg_obj[j]
        for q in np.unique(obj):
            sel = obj == q
            prior = _log_sum(log_phys[sel])
            if not np.isfinite(prior):
                continue
            out[sel] = _log_sum(log_phys[sel] + log_lik[sel]) - prior
        return out

    def _renormalize_stage(self, j: int, post: np.ndarray, prior: np.ndarray,
                           empty: bool) -> np.ndarray:
        """Normalise the unnormalised log posterior ``post`` (PG,): to one overall -- or,
        holding an object (``Config.belief_stages``), object by object to the prior's mass
        of each, so that the belief about the object stays where the pick-up left it.  An
        object ruled out altogether gives its mass to the others."""
        if not self.cfg.belief_stages or empty:
            return normalize_log(post)
        obj = self.pg_obj[j]
        out = np.full(post.shape, -np.inf)
        for q in np.unique(obj):
            sel = obj == q
            total, had = _log_sum(post[sel]), _log_sum(prior[sel])
            if np.isfinite(total) and np.isfinite(had):
                out[sel] = post[sel] - total + had
        return normalize_log(out)

    def updated_belief(self, j: int, s: int, actions, nxt, event: dict,
                       detail: Optional[dict] = None) -> PartnerBelief:
        """Eqs. 5-6 about robot j after the joint action ``actions`` at joint state s,
        landing in ``nxt``: the physical posterior from robot j's level-1 hypothesis
        policies averaged over the social posterior as it was (Eq. 6 as printed;
        ``Config.physical_likelihood="level0"``: from its level-0 policies, the authors'
        code; a level-1 robot, holding no social posterior, from the level-0 policies either
        way), staged by its hands; the social posterior from its level-1 hypothesis policies
        averaged over the physical posterior as it was, over the goals the transition term
        keeps (Eq. 5); the idle mode (``Config.idle_hypothesis``) as a two-mode filter over
        both; and Eq. 5's transition term.  Nothing is learnt from an action that no
        hypothesis explains (the prior stands, but for what the transition term rules out);
        nothing while robot j lifts an object ("pick" is the one action open to it).

        The two physical posteriors apart (``Config.belief_factored``, the default since
        2026-11-04): the belief about robot j's object and the one about its destination,
        the physical belief their product.  With robot j's hands empty its action updates
        the object's posterior -- each object by the likelihood of the action under it, the
        destinations averaged out under their posterior -- and the destination's stays as
        it was; holding an object, the other way round (:meth:`factored_posterior`).  What
        the hands' outcomes rule out (the transition term) acts on both.  Without it, one
        joint posterior staged as until then (:meth:`staged_log_likelihood`,
        :meth:`_renormalize_stage`).

        ``detail`` (a dict; ``Config.belief_detail``, since 2026-11-04): filled with
        everything the update computes -- the units (tau, the costs), the hypotheses, the
        belief before, every hypothesis model's Q at s and what it makes of the action
        (``_model_summary``), Eq. 6's likelihood per hypothesis, per destination and per
        object, the idle mode, the transition term, the posteriors before the hazards and
        the belief after -- for the record (``rollout``: the trace's "belief_update").

        A tempered update (``Config.likelihood_power`` alpha, since 2026-11-04, 1 by default):
        each posterior multiplied by its likelihood ^ alpha -- the learning factor's
        likelihood under the two posteriors apart (:meth:`factored_posterior`), each
        hypothesis's under the joint one, Eq. 5's for the social goals; the idle mode not.

        The likelihood's temperature (``Config.inference_temperature``, since 2026-11-04,
        None by default: Eq. 7's tau): the hypothesis models' softmax at tau_inf
        (:meth:`_log_policy`), in both posteriors.  The lapse (``Config.belief_lapse``,
        since 2026-11-04, 0 by default): every hypothesis model's likelihood of the action
        is ``(1 - e) psi + e / |A|`` (:meth:`lapsed`), in both posteriors -- the partner may
        pick an allowed action at random -- so that one move moves no hypothesis by more
        than a factor of about ``|A| / e``.

        The partner's goal may change (since 2026-11-02, options, 0 by default):
        ``Config.hand_hazard`` mixes the factor of the physical prior a hand action speaks to
        toward the uniform before that action's evidence (its object at a pick-up or a take,
        its destination at a place, a give or a swap); ``Config.goal_hazard`` and
        ``social_hazard`` mix the posteriors after the update, the prediction step for the
        next transition (:meth:`hazard_phys`, :func:`mix_uniform`)."""
        cfg, w = self.cfg, self.w
        old = self.about[j]
        new = old.copy()
        st = w.unsid(s)
        rec = detail is not None                 # Config.belief_detail: what this computes
        if rec:
            detail.update(reader=1 - j, about=j, action=int(actions[j]),
                          prior=self._belief_record(j, old))
        if w.busy[st[2], j]:
            if new.window is not None:           # Config.belief_window: one more update
                new.window = replace(new.window, updates=new.window.updates + 1)
            if rec:
                detail.update(skipped="lifting: nothing is learnt while the partner lifts",
                              after=self._belief_record(j, new))
            return new
        a = int(actions[j])
        q0 = float(old.idle)
        l_idle = self._log_idle(j, s, a)
        empty = self._stage(j, s)
        if rec:
            detail.update(
                stage="hands empty" if empty else "holding or lifting",
                # the units: Q, tau and the costs in reward units; a move's cost c(move) =
                # move_cost_rate x dt, tau = temperature x c(move) (temperature_unit
                # "move_cost"); the likelihood's tau, and Eq. 7's that the robots plan at
                tau=float(self.inference_taus[1 - j]), tau_planning=float(self.taus[1 - j]),
                temperature=float(cfg.temperature_of(1 - j)),
                inference_temperature=cfg.inference_temperature,
                c_move=float(cfg.costs[0]), c_stay=float(cfg.costs[4]),
                lapse=float(cfg.belief_lapse), likelihood_power=float(cfg.likelihood_power),
                likelihood=cfg.physical_likelihood,
                factored=bool(cfg.belief_factored), stages=bool(cfg.belief_stages),
                hypotheses=dict(object=self.pg_obj[j].copy(), destination=self.pg_dst[j].copy(),
                                social=[G.goal_name(t) for t in self.hyp_goals[1 - j]]),
                idle_log_lik=float(l_idle),
                hazards=dict(goal=float(cfg.goal_hazard), social=float(cfg.social_hazard),
                             hand=float(cfg.hand_hazard), idle=float(cfg.idle_hazard)))
        # the prior of this update: before a hand action's evidence, the factor it speaks to
        # mixed toward the uniform (Config.hand_hazard; a hand action is open to robot j only
        # in the stage it speaks to: pick and take with empty hands, place, give and swap
        # holding an object)
        prior = old.log_phys
        if cfg.hand_hazard > 0.0 and a >= PICK:
            prior = self.hazard_phys(j, prior, cfg.hand_hazard, objects=empty)
            new.log_phys = prior.copy()
            if rec:
                detail["after_hand_hazard"] = self._phys_record(j, prior)
        # ---- the physical posterior (Eq. 6) ---------------------------------------------------
        log_pol = (self.hypothesis_log_policies(j, s, a, prior, detail)
                   if old.log_social is not None else None)
        cut = None
        if (cfg.physical_likelihood == "cut" and log_pol is not None
                and old.options is not None):
            # the protected likelihood (subgoals.py): its own task by its "none" model, its
            # acts for the robot by its subgoals, which no physical goal predicts better
            raw, cut = self.cut_log_likelihood(j, s, a, prior, old, log_pol, detail)
        elif cfg.physical_likelihood == "level1" and log_pol is not None:
            # the paper's Eq. 6 as printed: the level-1 models' likelihood, averaged over
            # the social posterior
            with np.errstate(invalid="ignore"):
                raw = np.array([_log_sum(old.log_social + log_pol[g]) for g in range(self.PG)])
        else:                                    # the authors' code: the level-0 policies
            raw = self.physical_log_likelihood(j, s, a, detail)
        factored = cfg.belief_factored
        # the two posteriors apart (Config.belief_factored): each hypothesis's own likelihood,
        # the factor that does not learn averaged out by the marginal below; one joint
        # posterior (the rule until 2026-11-04): staged by the hands as it is updated
        lik = raw if factored else self.staged_log_likelihood(j, prior, raw, empty)
        mix = _log_mix(lik, 1.0 - q0, l_idle, q0)                  # (PG,) log P(a | g)
        keep = self.transition_term(j, s, actions, nxt, event)
        hard = np.where(keep, prior, -np.inf)                     # the hands' outcomes alone
        # Config.likelihood_power: the likelihood ^ alpha -- of the factor that learns
        # (factored_posterior), or of each hypothesis of the joint posterior
        alpha = float(cfg.likelihood_power)
        with np.errstate(invalid="ignore"):
            post = np.where(keep, prior + (mix if factored or alpha == 1.0 else alpha * mix),
                            -np.inf)
        explained = np.isfinite(post).any()
        if not explained:                        # nothing explains the action: the prior,
            post = hard                                       # less what the hands ruled out
        if old.window is not None:               # Config.belief_window: afresh, from the window
            new.log_phys, new.window = self.windowed_posterior(
                j, old.window, post if explained else None, hard, keep, empty)
        elif np.isfinite(post).any():
            new.log_phys = (self.factored_posterior(j, post, hard, empty, alpha) if factored
                            else self._renormalize_stage(j, post, prior, empty))
        # the physical prior over the goals the transition term keeps (T(s^t | s^{t-1}, a)
        # is the same under every social goal given the physical one): what the idle mode
        # and the social goals are judged against
        kept = hard
        if np.isfinite(kept).any():
            kept = normalize_log(kept)
        else:
            kept = prior
        if rec:
            detail.update(phys_log_lik=raw.copy(),                  # Eq. 6, before the stages
                          **({} if factored else dict(staged_log_lik=lik.copy())),
                          mix_log_lik=mix.copy(),                   # ... with the idle mode
                          transition_term=keep.copy(), explained=bool(explained),
                          kept=np.exp(kept), posterior=self._phys_record(j, new.log_phys),
                          **({} if new.window is None
                             else dict(window=self._window_record(j, new.window))))
        # ---- the idle mode ---------------------------------------------------------------------
        if cfg.idle_hypothesis:
            if explained:
                b = np.exp(kept)
                active = float((b * np.exp(np.where(np.isfinite(lik), lik, -np.inf))).sum())
                idle = float(np.exp(l_idle)) if np.isfinite(l_idle) else 0.0
                total = (1.0 - q0) * active + q0 * idle
                q1 = q0 * idle / total if total > 0.0 else q0
            else:
                q1 = q0
            h = cfg.idle_hazard
            new.idle = q1 * (1.0 - h) + (1.0 - q1) * h
            if rec:
                detail.update(idle_posterior=float(q1))
        # ---- the social posterior (Eq. 5) ----------------------------------------------------
        if old.log_social is not None and not self.known_social:
            l_soc = self.social_log_likelihood(kept, log_pol)
            post = old.log_social + alpha * _log_mix(l_soc, 1.0 - q0, l_idle, q0)
            if np.isfinite(post).any():
                new.log_social = normalize_log(post)
            if rec:
                detail.update(social_log_lik=l_soc.copy(),
                              social_posterior=np.exp(new.log_social))
        # ---- its subgoal (Config.physical_likelihood="cut"): updated within each physical and
        # social goal, then the subgoal dynamics -- the prediction for the next transition
        if cut is not None:
            new.options = self.updated_subgoals(j, s, a, nxt, old, cut, q0, l_idle, detail)
        # ---- the partner's goal may change: the prediction step for the next transition ------
        # (Config.goal_hazard, social_hazard; Baker, Saxe & Tenenbaum 2009, M2): each
        # posterior mixed toward the uniform over the hypotheses it still holds possible --
        # under the stages the physical one in the factor the next action will speak to
        if cfg.goal_hazard > 0.0:
            new.log_phys = self.hazard_phys(j, new.log_phys, cfg.goal_hazard,
                                            objects=not self._holds_next(j, nxt))
        if cfg.social_hazard > 0.0 and new.log_social is not None and not self.known_social:
            new.log_social = mix_uniform(new.log_social, cfg.social_hazard)
        if rec:
            detail["after"] = self._belief_record(j, new)
            detail.update(self._by_factor(j, raw, mix, prior, detail))
        return new

    def _by_factor(self, j: int, raw: np.ndarray, mix: np.ndarray, prior: np.ndarray,
                   detail: dict) -> dict:
        """Per destination and per object, for the record (Config.belief_detail): the
        likelihood of the action (Eq. 6, the other factor averaged out under the prior --
        P(a | d) = sum_o b_t(o) P(a | o, d), P(a | o) = sum_d b_t(d) P(a | o, d) -- and the same
        with the idle mode mixed in), the belief before (b_t), the posterior before the
        hazards, and the belief after (b_{t+1})."""
        lo, ld = self.factors(j, prior)
        obj, dst = self.pg_obj[j], self.pg_dst[j]
        out = {}
        for name, n, own, other, w_other in (("by_destination", self.w.K, dst, obj, lo),
                                             ("by_object", self.w.M, obj, dst, ld)):
            lik, lik_idle = np.full(n, -np.inf), np.full(n, -np.inf)
            with np.errstate(invalid="ignore"):
                for v in np.unique(own):
                    sel = own == v
                    lik[v] = _log_sum(w_other[other[sel]] + raw[sel])
                    lik_idle[v] = _log_sum(w_other[other[sel]] + mix[sel])
            key = "destination" if name == "by_destination" else "object"
            out[name] = dict(log_lik=lik, log_lik_with_idle=lik_idle,
                             prior=detail["prior"][key], posterior=detail["posterior"][key],
                             after=detail["after"][key])
        return out

    def _phys_record(self, j: int, log_phys: np.ndarray) -> dict:
        """A physical belief about robot j for the record: over the (object, destination)
        hypotheses, and its two marginals (Config.belief_detail)."""
        lo, ld = self.factors(j, log_phys)
        return dict(phys=np.exp(log_phys), object=np.exp(lo), destination=np.exp(ld))

    def _belief_record(self, j: int, b: "PartnerBelief") -> dict:
        """All of a belief about robot j for the record (Config.belief_detail)."""
        return dict(self._phys_record(j, b.log_phys),
                    social=None if b.log_social is None else np.exp(b.log_social),
                    idle=float(b.idle))

    def observe(self, actions, s: int, nxt, event: dict) -> None:
        """Eqs. 5-6 for both robots after the joint action ``actions`` at joint state s --
        each posterior from the beliefs as they were.  Under ``Config.belief_detail`` what
        each update computed is kept in ``last_update[j]`` (the update about robot j)."""
        details = [dict(), dict()] if self.cfg.belief_detail else [None, None]
        updates = [self.updated_belief(j, s, actions, nxt, event, details[j]) for j in (0, 1)]
        self.about = updates
        self.last_update = tuple(details) if self.cfg.belief_detail else None

    # ---- the reward a robot's own model gives the realised transition ------------------------
    def _stored_log_likelihood(self, slot: BeliefSlot, s: int, a_obs: int) -> np.ndarray:
        """(Mc,) log psi_k(a_obs | s) of the slot's hypotheses as the input kernels read it
        (the stored policies)."""
        K = self.PG
        p = 1 - slot.agent
        with np.errstate(divide="ignore"):
            if slot.depth == 1:
                return np.log(self.L0.psi[p * K + np.arange(K), s, a_obs].astype(np.float64))
            rows = [self.L1.psi_row[self.hypothesis_row(p, int(k), int(t))]
                    for k, t in zip(slot.hyp_goal, slot.hyp_type)]
            return np.log(self.L1.psi[rows, s, a_obs].astype(np.float64))

    def modeled_reward(self, i: int, s: int, actions, nxt, physical: float) -> float:
        """Eq. 2 reward of robot i's root for the realised joint action: its physical reward
        plus its social term, read off the realised next state."""
        slot = self.root_slot(i)
        goal = self.goals[i]
        if slot is None or goal == G.NONE:
            return physical
        w = self.w
        p = 1 - i
        a_p = int(actions[p])
        st = w.unsid(s)
        lik = np.exp(self._stored_log_likelihood(slot, s, a_p))
        b = self.planning_weights(slot)
        q0 = float(self.about[p].idle)
        wts = b * ((1.0 - q0) * lik + q0 * float(self.idle_tab[p][st[p], a_p]))
        tot = wts.sum()
        if tot <= 0.0:
            return physical
        wts = wts / tot
        own = w.rate(nxt, i, self.goal_ids[i])
        hyp_obj, hyp_dst = self.pg_obj[p][slot.hyp_goal], self.pg_dst[p][slot.hyp_goal]
        rj = w.goal_rates(nxt, p, hyp_obj, hyp_dst)                           # (Mc,)
        fin_i = w.goals_finished(nxt, i, self.pg_obj[i], self.pg_dst[i])      # (PG,)
        fin_j = w.goals_finished(nxt, p, hyp_obj, hyp_dst)                    # (Mc,)
        if slot.depth == 1:
            a_, b_, c_ = self.level1_term(i, goal)
            expected = float((wts * rj).sum())
            if fin_i[self.own_goal[i]]:          # the race term (goals.py): ahead ...
                race = own + self.lead * (own - expected)
            else:                                # ... or behind where its partner is done
                race = -float((wts * rj)[fin_j].sum())
            return physical + a_ * expected + b_ * own + c_ * race
        # robot i's goal reward as its partner estimates it, under the partner's belief (its
        # true goal under Config.objective="task": nested_phys)
        ri_all = w.goal_rates(nxt, i, self.pg_obj[i], self.pg_dst[i], estimate=True)
        fin_est = w.goals_finished(nxt, i, self.pg_obj[i], self.pg_dst[i], estimate=True)
        mine = self.nested_phys(i)
        cost_p = float(w.cost_c[st[2], p, a_p])
        xi = 0.0
        for h in np.flatnonzero(wts > 0.0).tolist():
            k, t = int(slot.hyp_goal[h]), self.hyp_goals[i][int(slot.hyp_type[h])]
            a1, b1, c1 = self.level1_term(p, t)
            ch = float((mine * ri_all).sum()) if (a1 != 0.0 or c1 != 0.0) else 0.0
            rk = rj[h] * (1.0 + b1) + a1 * ch - self.cost_in_partner_reward * cost_p
            if c1 != 0.0:                        # a racing partner, by what it takes the
                seen = mine * ri_all             # robot's goal reward to be: ahead ...
                if fin_j[h]:
                    rk += c1 * (rj[h] + self.lead * (rj[h] - float(seen.sum())))
                else:                            # ... or behind where it takes it as done
                    rk -= c1 * float(seen[fin_est].sum())
            al, be, nu, race_i = self.level2_term(i, goal, t, k == self.own_goal[i])
            term = al * rk + be * own + nu * rj[h]
            if race_i != 0.0:                    # its own race term, on the goal rewards
                if fin_i[self.own_goal[i]]:
                    term += race_i * (own + self.lead * (own - rj[h]))
                elif fin_j[h]:
                    term -= race_i * rj[h]
            xi += wts[h] * term
        return physical + float(xi)


class _BankView:
    """The rows ``grp`` of a bank, renumbered from 0: what the solver kernels read when a
    group of level-0 models is solved on temporary value and reward arrays."""

    def __init__(self, bank: ModelBank, grp: np.ndarray) -> None:
        self.P, self.pd, self.ho = bank.P, bank.pd, bank.ho
        for name in ("agent", "var", "svar", "social", "zero", "p_of"):
            setattr(self, name, np.ascontiguousarray(getattr(bank, name)[grp]))
