"""Independent reference implementations for the tests.

Written apart from the engine and deliberately naive: plain Python loops over every state,
no tables, no kernels.

* :class:`Ref` -- the rule of one transition, straight on "where every object is", following
  the rule as documented in :mod:`social_pairings.hands` rather than its code.
* :class:`Brute` -- the models at levels 0, 1 and 2 written out state by state: transitions
  from ``Ref.step``, distances by a breadth-first search written here, rewards and legal
  actions from the documented rules, fixed points by policy iteration with a sparse direct
  solve.
* :func:`staged_posterior` -- the two posteriors by the partner's hands
  (``Config.belief_stages``), one action of the partner at a time, in probabilities.
* ``TABLE_L1`` / ``TABLE_L2`` -- Table I of the paper transcribed cell by cell
  (``TABLE_L2_HOSTILE``: with the one cell ``Config.competition="table_hostile"`` turns;
  :func:`table_l2` with ``cooperate_with``: a cooperator that cooperates with the partners
  named there only, ``Config.cooperate_with``);
  :func:`level1_term` / :func:`level2_race` -- the same with the competitors that race: the
  two race forms against every partner, a rival against every partner but a cooperative
  one.

They share with the engine only the robots' moves (``ObjectWorld.psucc``; with rooms those
are checked against a rule written here, ``Brute.pair_move``) and the numbering of the
object configurations.

A scene may hold hand-off cells and a room per robot (``Scene.handoff_cells``,
``Scene.rooms``).  Here an object lying on hand-off cell h has the location ``("d", h)``,
a robot's legal moves are the free neighbours inside its room, and its ways are counted by
a breadth-first search through its room.  With ``Config.swap_on_place`` an object left on
the place of object p in a swap has the location ``("p", p)``; with ``Config.steal`` a
robot with free hands takes what the partner next to it holds (both since 2026-10-12).
Since 2026-10-18 nothing is automatic by default (``Config.automatic_hands=False``): a robot
picks its own object up by "pick" -- over ``Config.pick_steps`` transitions, the object at
the location ``("l", i, s)`` meanwhile, being lifted by robot i, s steps in, and "pick" the
one action open to the lifter -- and finishes by placing it on its destination; what is
left of a robot's way then counts those transitions too, as the engine's does
(``ObjectWorld``: "What is left of a robot's way").  ``Config(**AUTOMATIC_HANDS)`` is the
rule of before, which :class:`Ref` keeps as well.

Where a robot does not know which object its partner wants (``Config.partner_object=
"inferred"``) a physical goal is an object and a destination, ``(q, k)``: every model and
every hypothesis here names its object, and each hypothesis about the partner has its own
outcome of the robot's acts for the partner -- the dynamics in which the partner takes
that object (``Ref.step``'s ``acc``) and its set-down counts on that destination.  In a
robot's model the partner neither takes an object on arriving nor picks one first (both
depend on which it wants), as documented in ``ObjectWorld.model_tab``.
"""
from __future__ import annotations

import numpy as np

from social_pairings.constants import (GIVE, GONE, HELD0, HELD1, HOME, N_ACTIONS, PICK, PLACE,
                                       SWAP, TAKE)
from social_pairings.rollout import build_world

A = N_ACTIONS                    # four moves, stay, pick, place, take, give, swap

CODE = {"h": HOME, 0: HELD0, 1: HELD1, "g": GONE}
LOC = {v: k for k, v in CODE.items()}
FAR = 10 ** 6


# Two rooms of four cells and a doorway between them (the wall above it is (2, 1)):
#
#     y = 1    .  R0 #  R1 .          R0's object lies at (4, 1), R1's at (0, 1);
#     y = 0    .  .  D  .  .          R0's destination is (0, 0), R1's (4, 0).
#
# Each robot stays in its own room; the doorway D = (2, 0) belongs to both and is the
# hand-off cell.  An object gets to its owner only through the other robot's hands.
_PASS_GRID = np.ones((5, 2))
_PASS_GRID[2, 1] = np.inf
PASS = dict(grid=_PASS_GRID, starts=((1, 1), (3, 1)), pickups=((4, 1), (0, 1)),
            destinations=((0, 0), (4, 0)), candidate_cells=(), handoff_cells=((2, 0),),
            rooms=(((0, 0), (1, 0), (0, 1), (1, 1), (2, 0)),
                   ((3, 0), (4, 0), (3, 1), (4, 1), (2, 0))))
# the same map without rooms: everybody may walk through the doorway
OPEN_PASS = dict(PASS, rooms=None)


def staged_posterior(prior, log_lik, empty: bool, obj, dst, typ):
    """Two posteriors, by the partner's hands (``Config.belief_stages``, 2026-10-17), stated
    anew in probabilities: the belief after one action of the partner.  prior (H,): the
    belief before; log_lik (H,): the log-likelihood of the action under each hypothesis;
    obj, dst, typ (H,): each hypothesis's object, destination and social goal; empty: the
    partner's hands were empty when it acted.

    Hands empty: the action tells which object the partner wants and its social goal, and
    nothing about its destination -- every hypothesis gets the mean likelihood of the
    hypotheses with its object and social goal, each weighted by its prior, and after
    Bayes' rule the hypotheses with each destination are scaled together to the prior's
    belief in that destination.  Hands full: the action tells where the partner takes the
    object and its social goal -- Bayes' rule with each hypothesis's own likelihood, and the
    hypotheses with each object are scaled together to the prior's belief in that object.
    A slice of the frozen factor ruled out altogether (zero likelihood throughout) gives its
    belief to the others; a group without prior keeps its own likelihoods; an action no
    hypothesis explains leaves the belief alone (as the engine's ``belief_lapse`` comment
    says)."""
    prior = np.asarray(prior, dtype=float)
    lik = np.exp(np.asarray(log_lik, dtype=float))
    H = prior.size
    if empty:
        mean = np.empty(H)
        for h in range(H):
            same = [g for g in range(H) if obj[g] == obj[h] and typ[g] == typ[h]]
            mass = sum(prior[g] for g in same)
            mean[h] = sum(prior[g] * lik[g] for g in same) / mass if mass > 0 else lik[h]
        lik, frozen = mean, dst
    else:
        frozen = obj
    post = prior * lik
    if post.sum() == 0.0:
        return prior.copy()
    out = np.zeros(H)
    for f in sorted(set(int(v) for v in frozen)):
        sel = np.array([frozen[h] == f for h in range(H)])
        if post[sel].sum() > 0.0:
            out[sel] = post[sel] / post[sel].sum() * prior[sel].sum()
    return out / out.sum()


def code_of(loc, w) -> int:
    """The engine's status of an object at ``loc``: "h", 0, 1, "g", ("d", h) -- lying on
    hand-off cell h, status 4 + h -- ("p", p): lying on object p's place, status 4 + H + p
    (H: the scene's hand-off cells), or ("l", i, s): being lifted by robot i, s steps into
    the lift, status lift0 + i nstage + (s - 1) (``Config.pick_steps`` = nstage + 1)."""
    if isinstance(loc, tuple):
        if loc[0] == "l":
            return w.lift0 + loc[1] * w.nstage + loc[2] - 1
        return 4 + loc[1] if loc[0] == "d" else 4 + w.H + loc[1]
    return CODE[loc]


def loc_of(status: int, w):
    status = int(status)
    if status >= w.lift0 and w.nstage > 0:
        return ("l", (status - w.lift0) // w.nstage, (status - w.lift0) % w.nstage + 1)
    if status >= 4 + w.H:
        return ("p", status - 4 - w.H)
    return ("d", status - 4) if status >= 4 else LOC[status]


def make_world(scene: dict, cfg):
    """The world of a scene given as keyword arguments of ``build_world`` (without ``cfg``).
    -> (world, goal_ids)"""
    world, goal_ids, _, _ = build_world(
        scene["grid"], scene["starts"], scene["pickups"], scene["destinations"],
        scene.get("candidate_cells", ()), scene.get("objects", "goal"), cfg,
        handoff_cells=scene.get("handoff_cells", ()), rooms=scene.get("rooms"))
    return world, goal_ids


# =============================================================================================
# Table I of the paper, cell by cell
# =============================================================================================

TYPES = ("cooperation", "conflict", "competition", "coercion", "exchange")
# column "l = 1": (coefficient of r(g_j), coefficient of r(g_i)); None = N/A
TABLE_L1 = {"cooperation": (1, 0), "conflict": (-1, 0), "competition": (-1, 0),
            "coercion": (0, 1), "exchange": None}
# column "l > 1", by (type of xi_ij, type of xi_ji): (coefficient of R_j, of r(g_i), of
# r(g_j)); a pair of triples = (if g_i = g_j, if g_i != g_j); "n" = the exchange scalar
TABLE_L2 = {}
for _t in TYPES:
    TABLE_L2[("cooperation", _t)] = (1, 0, 0)
    TABLE_L2[("conflict", _t)] = (-1, 0, 0)
    TABLE_L2[("exchange", _t)] = None
TABLE_L2[("competition", "cooperation")] = (1, 0, 0)
TABLE_L2[("competition", "conflict")] = (-1, 0, 0)
TABLE_L2[("competition", "competition")] = (-1, 0, 0)
TABLE_L2[("competition", "coercion")] = ((1, 0, 0), (-1, 0, 0))
TABLE_L2[("competition", "exchange")] = (1, 0, 0)
TABLE_L2[("coercion", "cooperation")] = (1, 1, 0)
TABLE_L2[("coercion", "conflict")] = (-1, 1, 0)
TABLE_L2[("coercion", "competition")] = ((1, 0, 0), (-1, 1, 0))
TABLE_L2[("coercion", "coercion")] = (-1, 1, 0)
TABLE_L2[("coercion", "exchange")] = (1, 1, 0)
TABLE_L2[("exchange", "exchange")] = (1, 0, "n")
# ``Config.competition="table_hostile"``: the table with one printed cell turned -- a
# competitor against a cooperative or exchanging partner is paid -R_j, not +R_j.
TABLE_L2_HOSTILE = dict(TABLE_L2)
TABLE_L2_HOSTILE[("competition", "cooperation")] = (-1, 0, 0)
TABLE_L2_HOSTILE[("competition", "exchange")] = (-1, 0, 0)


def table_l1(goal: str, weight: float = 1.0, exchange: float = 0.0,
             unweighted: bool = False, spoiler: bool = False) -> tuple:
    """(coefficient of r(g_j), of r(g_i)) of a level-1 robot.  ``hinder`` is the table's
    conflict entry.  ``exchange``: what stands in for the table's N/A of an exchange goal at
    level 1 -- 0, or the scalar n of the paper's text (``Config.exchange_level1="copy"``).
    ``unweighted`` (``"copy_unweighted"``): that copy is not multiplied by the weight -- n
    for every robot with a weight above 0, which is the only thing a weight does to it.
    ``spoiler`` (``Config.hinder_own_goal=False``): a hinderer has no goal of its own -- its
    reward is the move costs and -w r(g_j); the r(g_i) of Eq. 2 is taken away by -1 (not
    weighted)."""
    if goal == "none":
        return (0.0, 0.0)
    if goal == "hinder" and spoiler:
        return (-weight + 0.0, -1.0)
    if goal == "hinder":
        goal = "conflict"
    cell = TABLE_L1[goal]
    if cell is None and unweighted:
        return ((float(exchange) if weight > 0.0 else 0.0) + 0.0, 0.0)
    a, b = (float(exchange), 0.0) if cell is None else cell
    return (weight * a + 0.0, weight * b + 0.0)


# ``Config.competition="rival"``: a competitor races every partner but one that helps it, a
# cooperative one, and against that one it is the competitor of the hostile table (-r(g_j)
# at l = 1, -R_j above).  An exchanging partner has no social term toward a competitor (its
# cell of the table is N/A): it is no helper, and is raced.
HELPERS = ("cooperation",)


def _races(goal: str, competition: str, partner) -> bool:
    """Does a robot with ``goal`` race a partner with the social goal ``partner`` (None: a
    partner whose social goal it does not know)?"""
    if goal != "competition":
        return False
    if competition in ("race", "race_then_hinder"):
        return True
    return competition == "rival" and partner not in HELPERS


def level1_term(goal: str, weight: float = 1.0, exchange: float = 0.0,
                competition: str = "race_then_hinder", partner=None,
                unweighted: bool = False, spoiler: bool = False) -> tuple:
    """(a, b, c) of a level-1 robot: ``a r(g_j) + b r(g_i) + c race``, where race is
    r(g_i) + lam (r(g_i) - r(g_j)) once the robot's goal is done, -r(g_j) while only its
    partner's is, and 0 otherwise (lam: 1 under "race_then_hinder", 0 under "race" and
    "rival").  A competitor that races has (0, 0, w): under the two race forms whoever its
    partner is, as a rival unless it knows its partner (``partner``: its social goal, None
    if the robot is not told) to be cooperative.  Otherwise, and with
    ``competition="table"`` or ``"table_hostile"``, its term is the table's, like every
    other goal's."""
    if _races(goal, competition, partner):
        return (0.0, 0.0, weight + 0.0)
    return table_l1(goal, weight, exchange, unweighted, spoiler) + (0.0,)


def level2_race(goal: str, weight: float = 1.0, competition: str = "race_then_hinder",
                partner=None) -> float:
    """The coefficient of the race term of a level-2 robot against a partner taken to have
    the social goal ``partner``: a competitor that races it keeps its level-1 term,
    ``w race``, and has nothing of Table I's column l > 1."""
    return weight + 0.0 if _races(goal, competition, partner) else 0.0


def table_l2(goal: str, partner: str, same: bool, n: float, weight: float = 1.0,
             competition: str = "table", cooperate_with=None, spoiler: bool = False,
             conflict: str = "table") -> tuple:
    """(coefficient of R_j, of r(g_i), of r(g_j)) of a level-2 robot against a partner taken
    to have the social goal ``partner``.  ``hinder`` keeps its level-1 meaning, -r(g_j); a
    hinder partner is a conflict partner; a partner without a social goal is not a row of
    the table -- the level-1 column applies (its R_j is its physical reward).
    ``competition="table_hostile"``: the table with the competitor's cell against a
    cooperative or exchanging partner turned (``TABLE_L2_HOSTILE``).  A rival's term is
    that cell against a cooperative partner and no cell at all against anybody else
    (:func:`level2_race`); asked for its cells all the same, it is given the hostile
    table's.  ``cooperate_with`` (``Config.cooperate_with``, names; None: everybody): a
    cooperator has its cell only against a partner named there -- a hinder partner by that
    name, not as a conflict partner -- and nothing at all against any other.
    ``conflict="task"`` (``Config.conflict``): a conflict robot keeps the column l = 1,
    -r(g_j), against every partner -- no R_j."""
    if goal == "none":
        return (0.0, 0.0, 0.0)
    if goal == "conflict" and conflict == "task":
        return (0.0, 0.0, -weight + 0.0)
    if goal == "no goal":                # the twin of a hinderer without a goal of its own:
        return (0.0, -1.0, 0.0)          # its moves' costs, nothing else
    if goal == "cooperation" and cooperate_with is not None and partner not in cooperate_with:
        return (0.0, 0.0, 0.0)
    if goal == "hinder":
        return (0.0, -1.0 if spoiler else 0.0, -weight)
    if partner == "hinder":
        partner = "conflict"
    if partner == "none":
        cell = TABLE_L1[goal]
        cell = (0, 0, 0) if cell is None else (cell[0], cell[1], 0)
    else:
        hostile = competition in ("table_hostile", "rival")
        cell = (TABLE_L2_HOSTILE if hostile else TABLE_L2)[(goal, partner)]
        if cell is None:
            cell = (0, 0, 0)
        elif isinstance(cell[0], tuple):
            cell = cell[0] if same else cell[1]
    return tuple(weight * (float(n) if v == "n" else float(v)) + 0.0 for v in cell)


# =============================================================================================
# The rule of one transition
# =============================================================================================

class Ref:
    """The shared-object world of one scene written directly on "where every object is"."""

    def __init__(self, w) -> None:
        self.w = w
        cfg = w.cfg
        self.home = [int(c) for c in w.home]
        self.cand = set(int(c) for c in w.dest) if cfg.set_down_for_partner else set()
        # a robot does not know which object its partner wants: any object may be offered
        self.infer = cfg.partner_object == "inferred"
        # the hand-off cells, in the order of the scene: where the partner's object may be
        # put down and lies until somebody picks it up
        self.hand = [int(w.cell_id[c]) for c in w.handoff]
        # Object i, and no other, completes robot i's goal: only object i is handed to robot
        # i or set down for it.  Which objects a robot may hold at all (every one where the
        # partner's object is inferred: any may be the partner's):
        every = cfg.handle_spares or self.infer
        self.rights = [(set(range(w.M)) if every else {0, 1}) for _ in (0, 1)]
        # the older rule: a robot handed its own object takes it even with full hands
        self.forced = bool(cfg.forced_swap)
        # may an object go from hand to hand at all?
        self.give = bool(cfg.hand_over)
        # full hands on an object's place swap what they hold for what lies there; free
        # hands take what the partner next to them holds
        self.swap = bool(cfg.swap_on_place)
        self.steal = bool(cfg.steal)
        # the rule of the hands until 2026-10-18: a robot takes its own object on arriving
        # where it lies and sets it down on arriving at its destination; else "pick" and
        # "place" do both, a pick-up taking `steps` transitions
        self.auto = bool(cfg.automatic_hands)
        self.steps = int(cfg.pick_steps)
        # the special cells: a lift happens on one (an object's place, a hand-off cell); a
        # lifter anywhere else is a state no model gets to, and its lift stays as it is
        self.sites = set(self.home) | set(int(c) for c in w.dest) | set(self.hand)

    def lies(self, loc, q: int):
        """The cell object q lies on (None: it is in somebody's hands, being lifted -- under
        its lifter, out of everybody else's reach -- or set down)."""
        if loc[q] == "h":
            return self.home[q]
        if isinstance(loc[q], tuple) and loc[q][0] != "l":
            return self.hand[loc[q][1]] if loc[q][0] == "d" else self.home[loc[q][1]]
        return None

    def lifting(self, loc, i: int):
        """The object robot i is lifting (None: none)."""
        for q, v in enumerate(loc):
            if isinstance(v, tuple) and v[0] == "l" and v[1] == i:
                return q
        return None

    def lying_at(self, cell: int, k: int):
        """The location of object k lying on ``cell``: its place, a hand-off cell or another
        object's place; None where no object can lie (no lift begins there)."""
        if cell == self.home[k]:
            return "h"
        if cell in self.hand:
            return ("d", self.hand.index(cell))
        if self.swap and cell in self.home:
            return ("p", self.home.index(cell))
        return None

    def step(self, loc, done, x, n, acts, goal, deliver, own, hands, after_done=True,
             gobj=(0, 1), acc=(0, 1)):
        """One transition of the hands.

        loc : per object "h" (at its place), 0 / 1 (in that robot's hands), "g" (set down),
            ("d", h) (lying on hand-off cell h) or ("p", p) (left on object p's place)
        done : [bool, bool]
        x, n : the robots' cells before / after the move;  acts : their actions
        goal[i] : cell where robot i sets down for its own goal (None: this dynamics does
            not; "any", without the automatic hands: on any candidate cell, with whatever it
            holds that its partner does not take -- the partner in a robot's model, whose
            destination the model does not know)
        deliver[i] : cell where robot i's set-down for the partner counts (None: nowhere,
            "any": wherever it tries)
        own[i] : the object robot i takes on arriving at its place (None: none; automatic
            hands only)
        hands[i] : robot i's hands work
        gobj[i] : robot i's goal object -- set down on goal[i], picked up first (None: not
            known to this dynamics)
        acc[i] : what robot i takes when it is handed it or it is set down for it (an
            object; None: nothing; "any": anything)
        -> (loc, done)"""
        loc, done = list(loc), list(done)
        M = len(loc)
        adjacent = int(np.abs(self.w.cells[n[0]] - self.w.cells[n[1]]).sum()) == 1

        def holding(i):
            held = [k for k in range(M) if loc[k] == i]
            assert len(held) <= 1
            return held[0] if held else None

        def own_goal() -> bool:
            changed = False
            for i in (0, 1):
                k = holding(i)
                if (k is not None and k == gobj[i] and not done[i] and goal[i] is not None
                        and n[i] == goal[i]):
                    loc[k], done[i] = "g", True
                    changed = True
            return changed

        def takes(i, k) -> bool:
            return acc[i] == "any" or (acc[i] is not None and acc[i] == k)

        if self.auto:
            own_goal()
        # lifts in progress: the lifter's "pick" takes it a step on, the object into its
        # hands on the last; anything else of it (an inert partner's, hands False) drops the
        # object where it lies -- a move away cannot happen and changes nothing
        busy = [False, False]
        for i in (0, 1):
            k = self.lifting(loc, i)
            if k is None:
                continue
            busy[i] = True
            if acts[i] == PICK and hands[i] and (after_done or not done[i]):
                loc[k] = i if loc[k][2] + 1 >= self.steps else ("l", i, loc[k][2] + 1)
            elif n[i] == x[i] and n[i] in self.sites:
                back = self.lying_at(n[i], k)    # (one object to a place: where another lies
                if back is not None and (         # on it, or no object can, the lift stays)
                        n[i] in self.hand
                        or not any(self.lies(loc, q) == n[i] for q in range(M) if q != k)):
                    loc[k] = back
        hand = [acts[i] if (acts[i] >= PICK and hands[i] and (after_done or not done[i])
                            and not busy[i])
                else None for i in (0, 1)]
        before = [holding(0), holding(1)]
        plan = [None, None]
        for i in (0, 1):
            if hand[i] is None:
                continue
            j, k, a = 1 - i, before[i], hand[i]
            lying = [q for q in range(M) if self.lies(loc, q) == n[i] and q in self.rights[i]]
            if a == PICK and k is None and lying:            # its goal object first
                plan[i] = ("take", gobj[i] if gobj[i] in lying else lying[0])
            elif a == PLACE and k is not None:
                mine = (not self.auto and not done[i] and goal[i] not in (None, "any")
                        and n[i] == goal[i] and (gobj[i] is None or k == gobj[i]))
                if mine:                                     # its goal object on its
                    plan[i] = ("own", k)                     # destination: done
                elif n[i] == self.home[k]:
                    if not any(self.lies(loc, q) == n[i] for q in range(M) if q != k):
                        plan[i] = ("back", k)                # its place, if nothing lies there
                elif n[i] in self.hand:                      # on a hand-off cell: put down
                    plan[i] = ("leave", k)
                elif n[i] in self.cand and not done[j] and takes(j, k):
                    plan[i] = ("down", k)                    # for the partner
                elif not self.auto and not done[i] and goal[i] == "any" and n[i] in self.cand:
                    plan[i] = ("own", k)
            elif (a == TAKE and k is None and self.steal and adjacent
                  and before[j] is not None and before[j] in self.rights[i]):
                plan[i] = ("steal", before[j])               # out of the partner's hands
            elif a == GIVE and k is not None and self.give and adjacent:
                # nobody is made to let go: full hands take an object only if their robot
                # gives in the same transition (or under the older forced-swap rule); nor
                # does a robot lifting one take anything
                if (hands[j] and not busy[j] and (after_done or not done[j]) and takes(j, k)
                        and k in self.rights[j]
                        and (before[j] is None or acts[j] == GIVE or self.forced)):
                    plan[i] = ("give", k)
            elif a == SWAP and k is not None and self.swap and n[i] in self.home:
                there = [q for q in lying if q != k]
                if there:                                    # on an object's place
                    plan[i] = ("swap", k, there[0])
        for i in (0, 1):
            if plan[i] is None:
                continue
            what, k = plan[i][0], plan[i][1]
            if what == "swap":                               # what lay there into its hands,
                loc[plan[i][2]] = i                          # what it held onto the cell
                loc[k] = "h" if n[i] == self.home[k] else ("p", self.home.index(n[i]))
            elif what == "back":
                loc[k] = "h"
            elif what == "leave":
                loc[k] = ("d", self.hand.index(n[i]))
            elif what == "take":
                loc[k] = ("l", i, 1) if self.steps > 1 else i
            elif what == "down":
                if deliver[i] == "any" or deliver[i] == n[i]:
                    loc[k], done[1 - i] = "g", True
                elif not self.auto and goal[i] == "any":     # not its partner's destination:
                    plan[i] = ("own", k)                     # then, by its own hypothesis,
        gives = [p is not None and p[0] == "give" for p in plan]      # its own (below)
        if all(gives):                            # both hold, both give: the objects are swapped
            loc[before[0]], loc[before[1]] = 1, 0
        else:
            for i in (0, 1):
                j = 1 - i
                if not gives[i]:
                    continue
                if before[j] is None:
                    if not (plan[j] and plan[j][0] == "take"):
                        loc[before[i]] = j
                elif self.forced:                 # full hands take it and give what they still
                    if loc[before[j]] == j:       # hold in exchange
                        loc[before[j]] = i
                    loc[before[i]] = j
        for i in (0, 1):                          # taken, if the partner still holds it
            if plan[i] is not None and plan[i][0] == "steal" and loc[plan[i][1]] == 1 - i:
                loc[plan[i][1]] = i
        for i in (0, 1):                          # set down for its own goal -- unless its
            if plan[i] is not None and plan[i][0] == "own" and not done[i]:   # partner just
                loc[plan[i][1]], done[i] = "g", True                          # finished it
        if not self.auto:
            return tuple(loc), tuple(done)
        while True:
            changed = False
            for i in (0, 1):
                q = own[i]
                if (q is not None and n[i] != x[i] and not done[i]
                        and self.lies(loc, q) == n[i] and holding(i) is None):
                    loc[q] = i
                    changed = True
            if own_goal():
                changed = True
            if not changed:
                return tuple(loc), tuple(done)


# =============================================================================================
# The models, state by state
# =============================================================================================

def states_of(w, blocks) -> np.ndarray:
    """Joint state indices of the given blocks that the engine solves: without the pairs in
    which robots overlap, and, in a block where a robot is lifting an object, with that
    robot on a cell the object can lie on (``ObjectWorld.state_ok``)."""
    return np.concatenate([int(b) * w.NN + np.flatnonzero(w.state_ok[int(b)])
                           for b in np.asarray(blocks).tolist()] or [np.zeros(0, dtype=np.int64)])


class Brute:
    """The models of one world written out state by state.

    ``level0`` / ``level1`` / ``level2`` return dense arrays over all S joint states:
    V (S,), Q (S, 10) (-inf where illegal), P (S, 10) the partner's action distribution, R
    (S, 10) the expected rewards, psi (S, 10) the softmax policy of Eq. 7."""

    def __init__(self, w, cfg) -> None:
        self.w, self.cfg, self.ref = w, cfg, Ref(w)
        self.B, self.N, self.NN, self.S = w.B, w.N, w.NN, w.B * w.NN
        self.loc = [tuple(loc_of(v, w) for v in w.status[b]) for b in range(w.B - 1)]
        self.loc.append(("g",) * w.M)                      # the terminal block
        self.dn = [tuple(bool(v) for v in w.done[b]) for b in range(w.B - 1)] + [(True, True)]
        self.home = [int(c) for c in w.home]
        self.dest = [int(c) for c in w.dest]
        self.hand = self.ref.hand
        cells = [tuple(int(v) for v in c) for c in w.cells]
        ident = {c: i for i, c in enumerate(cells)}
        # each robot's room: the cells it may stand on (every free cell without rooms)
        self.room = [set(range(self.N)) if w.rooms is None else {ident[tuple(c)] for c in room}
                     for room in (w.rooms if w.rooms is not None else (None, None))]

        def search(inside) -> np.ndarray:
            """Steps between any two cells through the cells ``inside``; FAR: no way."""
            dist = np.full((self.N, self.N), FAR, dtype=np.int64)
            for src in sorted(inside):                     # breadth-first search
                dist[src, src] = 0
                todo = [src]
                while todo:
                    x = todo.pop(0)
                    cx, cy = cells[x]
                    for c2 in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                        y = ident.get(c2)
                        if y is not None and y in inside and dist[src, y] > dist[src, x] + 1:
                            dist[src, y] = dist[src, x] + 1
                            todo.append(y)
            return dist

        self.dist = search(set(range(self.N)))             # on the free cells: who is next
        self.dist_by = [search(self.room[i]) for i in (0, 1)]    # to whom; each robot's own ways
        self.cells, self.ident = cells, ident
        # the pairs of cells the robots can stand on: apart, each in its room
        self.pairs = [p for p in range(self.NN)
                      if p // self.N in self.room[0] and p % self.N in self.room[1]
                      and not w.conflict(p // self.N, p % self.N)]
        assert self.pairs == [p for p in range(self.NN) if w.pair_ok[p]]
        self.vdone = float(cfg.rho - cfg.costs[4]) / (1.0 - cfg.gamma)
        self._next: dict = {}

    # ---- the documented rules ----------------------------------------------------------------
    def can_move(self, i: int, x: int, a: int) -> bool:
        """Is move a (0..3; 4: "stay") legal for robot i on cell x?  To a free neighbour
        inside its room."""
        if a == 4:
            return True
        dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0))[a]
        y = self.ident.get((self.cells[x][0] + dx, self.cells[x][1] + dy))
        return y is not None and y in self.room[i]

    def pair_move(self, x, moves) -> tuple:
        """Where the two robots stand after trying ``moves`` (0..3, 4 = stay) from cells x:
        a move that is not legal is no move; two robots never swap cells head-on (both
        stay), and never end on the same cell -- a robot that stays keeps its cell,
        otherwise robot 0 moves and robot 1 stays (robot 0 stays as well if it would then
        walk into robot 1).  For one robot per cell (``robot_separation_m=0``)."""
        t = []
        for i in (0, 1):
            dx, dy = ((0, 1), (1, 0), (0, -1), (-1, 0), (0, 0))[moves[i]]
            y = self.ident.get((self.cells[x[i]][0] + dx, self.cells[x[i]][1] + dy))
            t.append(y if self.can_move(i, x[i], moves[i]) else x[i])
        if t[0] == x[1] and t[1] == x[0]:
            return tuple(x)
        if t[0] != t[1]:
            return tuple(t)
        if t[1] == x[1]:
            return tuple(x)
        return (x[0] if t[0] == x[1] else t[0], x[1])

    def stay(self, x: int, eps: float, i: int) -> list:
        """Robot i staying put on cell x: "stay" w.p. 1 - eps, eps over its legal moves;
        never a hand action."""
        moves = [self.can_move(i, x, a) for a in range(4)]
        nm = sum(moves)
        return ([eps / nm if mv else 0.0 for mv in moves] + [1.0 - eps if nm else 1.0]
                + [0.0] * (A - PICK))

    def rate(self, b: int, i: int, n, k: int, obj=None, estimate: bool = False) -> float:
        """Robot i's goal reward in block b for destination k (and goal object ``obj``, its
        own unless given); n = (cell of R0, cell of R1).  estimate: as its partner estimates
        it in robot i's own model, where i's done flag is its own object delivered -- for
        another object the flag says nothing."""
        cfg, loc = self.cfg, self.loc[b]
        q = i if obj is None else obj
        x = n[i]
        if self.dn[b][i] and not (estimate and q != i):
            return cfg.rho
        # The goal needs object q: in robot i's hands, lying somewhere (robot i fetches it:
        # at its place, or on a hand-off cell), being lifted, or in its partner's hands.
        # Ways are the robot's own: through its room -- and, without the automatic hands,
        # count the transitions of the hands too: the lift ahead and the one set-down.
        auto, steps = self.ref.auto, self.ref.steps
        place, lift = (0, 0) if auto else (1, steps)
        mine = self.dist_by[i]
        lies = self.ref.lies(loc, q)
        lifted = isinstance(loc[q], tuple) and loc[q][0] == "l"
        full = i in loc or self.ref.lifting(loc, i) not in (None, q)
        if loc[q] == i:
            D = mine[x, self.dest[k]] + place
        elif lies is not None:
            if cfg.hands_full_reward == "none" and full:
                return 0.0               # its hands are full of an object that is not its own
            D = mine[x, lies] + mine[lies, self.dest[k]] + lift + place
        elif lifted and loc[q][1] == i:  # under it, so many steps into the lift
            D = mine[x, self.dest[k]] + (steps - loc[q][2]) + place
        elif (loc[q] == 1 - i or (lifted and loc[q][1] == 1 - i)) and cfg.held_reward == "carrier":
            # the reward follows the object: the carrier's way to the destination and the one
            # transition its set-down takes (and its lift still to go)
            D = self.dist_by[1 - i][n[1 - i], self.dest[k]] + 1 + (
                (steps - loc[q][2]) if lifted else 0)
        else:                            # nothing robot i can do
            return 0.0
        if D >= FAR:                     # no way: it lies, or belongs, outside the room
            return 0.0
        return cfg.rho * max(1.0 - D * cfg.resolution / cfg.reward_radius_m, 0.0)

    def finished(self, b: int, i: int, n, k: int, obj=None, estimate: bool = False) -> bool:
        """Robot i's goal with destination k (and goal object ``obj``, its own unless given)
        is done in block b with the robots at cells n: the block says so, or robot i stands
        on that destination holding that object (which only its own model's dynamics turn
        into a set-down).  estimate: as for :meth:`rate`."""
        q = i if obj is None else obj
        if self.dn[b][i] and not (estimate and q != i):
            return True
        return bool(self.ref.auto and self.loc[b][q] == i and n[i] == self.dest[k])

    def race(self, b: int, i: int, n, k_own: int, r_own: float, r_other: float,
             lost: float, obj=None) -> float:
        """The race term of robot i with destination k_own.  Once its own goal is finished:
        its goal reward -- and, under "race_then_hinder", its lead over its partner on top,
        its goal reward minus its partner's (``r_other``); under "race" and "rival" nothing
        on top.  Until then: -``lost``, its partner's goal reward counted where its
        partner's goal is finished."""
        if not self.finished(b, i, n, k_own, obj):
            return -lost
        if self.cfg.competition == "race_then_hinder":
            return r_own + (r_own - r_other)
        return r_own

    def cost(self, b: int, i: int, a: int) -> float:
        """c(a): a move or "stay" (Config.costs), pick (pick_cost_rate, on the first
        transition of a lift only), the other hand actions (use_cost_rate); a move with a
        foreign object in hand costs more."""
        cfg, loc = self.cfg, self.loc[b]
        if a < PICK:
            c = float(cfg.costs[a])
        elif a == PICK:
            c = 0.0 if self.ref.lifting(loc, i) is not None else cfg.pick_cost_rate * cfg.dt
        else:
            c = cfg.use_cost_rate * cfg.dt
        if a < 4 and i in loc and loc.index(i) != i:       # carrying a foreign object
            c *= float(cfg.foreign_carry_cost)
        return c

    def busy(self, b: int, i: int) -> bool:
        """Robot i is lifting an object in block b: "pick" is the one action open to it."""
        return b < self.B - 1 and self.ref.lifting(self.loc[b], i) is not None

    def legal_action(self, b: int, i: int, x, social: bool, a: int) -> bool:
        """May robot i choose action a in block b with the robots at cells x?"""
        if self.busy(b, i):
            return a == PICK
        return self.can_move(i, x[i], a) if a < PICK else self.may(b, i, x, social, a)

    def stay_in(self, b: int, x: int, eps: float, i: int) -> list:
        """:meth:`stay` -- or, for a robot lifting an object in block b, "pick"."""
        if self.busy(b, i):
            return [1.0 if a == PICK else 0.0 for a in range(A)]
        return self.stay(x, eps, i)

    def may(self, b: int, i: int, x, social: bool, a: int) -> bool:
        """May robot i choose hand action a in block b with the robots at cells x?  Only
        where it does something; the acts for the partner (giving, putting down on a
        hand-off cell, setting down on a destination) only for a robot with a social goal."""
        loc, dn, j = self.loc[b], self.dn[b], 1 - i
        if all(dn) or (dn[i] and not self.w.after_done):
            return False
        lying = [q for q in range(len(loc)) if self.ref.lies(loc, q) == x[i]
                 and q in self.ref.rights[i]]
        next_to = self.dist[x[0], x[1]] == 1
        if i not in loc:                                     # free hands
            if a == PICK:
                return bool(lying)
            if a == TAKE:                                    # out of the partner's hands
                return bool(self.ref.steal and next_to and j in loc
                            and loc.index(j) in self.ref.rights[i])
            return False
        k = loc.index(i)
        theirs = social and (k == j or self.ref.infer)       # something its partner could use
        if a == SWAP:
            return bool(self.ref.swap and x[i] in self.home and any(q != k for q in lying))
        if a == PLACE:
            if not self.ref.auto and x[i] in self.ref.cand:
                return True                                  # for its own goal, maybe
            if x[i] == self.home[k]:                         # back, if nothing lies there
                return not any(self.ref.lies(loc, q) == x[i] for q in range(len(loc)))
            if x[i] in self.hand:
                return theirs
            return bool(theirs and x[i] in self.ref.cand and not dn[j])
        if a == GIVE:
            return bool(theirs and self.ref.give and next_to and not self.busy(b, j)
                        and (self.w.after_done or not dn[j]))
        return False

    def delivering(self, x: int, to: int, i: int) -> list:
        """``Config.partner_delivers``: the predicted action of robot i at cell x carrying
        its partner's object to cell ``to`` -- a step that shortens its way (1 - park_eps
        over those, the rest over its other legal moves and "stay"); on the cell: "place"."""
        if x == to:
            return [1.0 if a == PLACE else 0.0 for a in range(A)]
        ok = [self.can_move(i, x, a) for a in range(5)]
        dist = self.dist_by[i]
        good = [a for a in range(4)
                if ok[a] and dist[int(self.w.nb[x, a]), to] < dist[x, to]]
        rest = [a for a in range(5) if ok[a] and a not in good]
        eps = self.cfg.park_eps if good and rest else 0.0
        out = [0.0] * A
        for a in good:
            out[a] = (1.0 - eps) / len(good)
        for a in rest:
            out[a] = (eps if good else 1.0) / len(rest)
        return out

    def nexts(self, goal, deliver, own, hands, gobj=(0, 1), acc=(0, 1)) -> np.ndarray:
        """(B, NN, A^2): the block after every joint action under one dynamics."""
        key = (goal, deliver, own, hands, gobj, acc)
        if key not in self._next:
            w = self.w
            out = np.full((self.B, self.NN, A * A), w.T, dtype=np.int64)
            for b in range(self.B - 1):
                for pair in self.pairs:
                    x = divmod(pair, self.N)
                    for j in range(A * A):
                        n = divmod(int(w.psucc[pair, j]), self.N)
                        loc2, dn2 = self.ref.step(self.loc[b], self.dn[b], x, n, divmod(j, A),
                                                  goal, deliver, own, hands, w.after_done,
                                                  gobj, acc)
                        if not all(dn2):
                            out[b, pair, j] = w.block_of([code_of(v, w) for v in loc2],
                                                         [int(v) for v in dn2])
            self._next[key] = out
        return self._next[key]

    def hypothesis_dynamics(self, o: int, k: int, q: int, qh: int, kh: int,
                            social: bool = True) -> np.ndarray:
        """Where the partner's object is inferred: the dynamics of robot o's model with
        physical goal (q, k) under the hypothesis that its partner wants object qh on
        destination kh -- the partner takes qh when it is handed it or it is set down for it
        there, and nothing else; it takes no object on arriving and picks none first.
        social=False: a robot without a social goal never acts for its partner -- what it
        places on a candidate cell is not taken (its "place" there, open to it since
        2026-10-18 for its own set-down, does nothing elsewhere), nor what it would hand
        over."""
        goal = tuple(self.dest[k] if i == o else self.partner_goal for i in (0, 1))
        deliver = tuple((self.dest[kh] if social else None) if i == o else self.dest[k]
                        for i in (0, 1))
        own = tuple(q if i == o else None for i in (0, 1))
        acc = tuple(q if i == o else (qh if social else None) for i in (0, 1))
        return self.nexts(goal, deliver, own, (True, True), own, acc)

    @property
    def partner_goal(self):
        """Where the partner sets down for its own goal in a robot's model: nowhere with the
        automatic hands (it parks on its destination instead), else on any candidate cell
        (its policy under each hypothesis places only on that hypothesis's destination)."""
        return None if self.ref.auto else "any"

    def model_dynamics(self, o: int, k: int):
        """The two dynamics of a level >= 1 model of robot o with destination k: what it sets
        down for its partner fails / counts wherever it tries.  -> (fail, works)"""
        goal = tuple(self.dest[k] if i == o else self.partner_goal for i in (0, 1))
        fail = self.nexts(goal, tuple(None if i == o else self.dest[k] for i in (0, 1)),
                          (0, 1), (True, True))
        works = self.nexts(goal, tuple("any" if i == o else self.dest[k] for i in (0, 1)),
                           (0, 1), (True, True))
        return fail, works

    # ---- fixed points ------------------------------------------------------------------------
    def solve(self, legal, R, const, trip, fixed):
        """Policy iteration with exact evaluation.  trip[a] = (rows, cols, probabilities);
        const (S, 10): gamma x the known values reached; fixed (S,) bool: V = vdone there.
        -> V (S,), Q (S, 10) (-inf where illegal)."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        S, g = self.S, self.cfg.gamma
        P = [sp.csr_matrix((v, (r, c)), shape=(S, S)) for (r, c, v) in trip]
        act = legal.any(axis=1) & ~fixed
        pol = legal.argmax(axis=1)
        rows = np.arange(S)
        for _ in range(500):
            Ppi = sum(sp.diags(((pol == a) & act).astype(float)) @ P[a] for a in range(A))
            rhs = np.where(act, (R + const)[rows, pol], np.where(fixed, self.vdone, 0.0))
            V = spl.spsolve((sp.identity(S, format="csc") - g * Ppi).tocsc(), rhs)
            Q = R + const + g * np.stack([P[a] @ V for a in range(A)], axis=1)
            Q[~legal] = -np.inf
            best = Q.argmax(axis=1)
            # an improvement has to be beyond the rounding of the linear solve, which grows
            # with the values (they grow with the weights)
            margin = 1e-12 * max(1.0, float(np.abs(V).max()))
            better = act & (Q[rows, best] > Q[rows, pol] + margin)
            if not better.any():
                resid = float(np.abs(Q.max(axis=1) - V)[act].max())
                assert resid < 1e-8, resid
                return V, Q
            pol = np.where(better, best, pol)
        raise RuntimeError("policy iteration did not converge")

    def evaluate(self, pol, parts: dict, trip, const, states) -> dict:
        """The value of the fixed policy ``pol`` (S,) (-1 where no action) on each term of
        the reward apart, over ``states`` (every successor of which has to be among them):
        {name: (S,)} with NaN elsewhere -- the discounted sum of that term alone under the
        policy's transitions; ``const`` (gamma x vdone reached, physical models) goes to the
        own goal."""
        import scipy.sparse as sp
        import scipy.sparse.linalg as spl

        S, g = self.S, self.cfg.gamma
        local = np.full(S, -1, dtype=np.int64)
        local[states] = np.arange(len(states))
        rows, cols, vals = [], [], []
        for a in range(A):
            r, c, v = (np.asarray(x) for x in trip[a])
            keep = np.isin(r, states) & (pol[r] == a) if r.size else np.zeros(0, dtype=bool)
            if keep.any():
                assert (local[c[keep]] >= 0).all(), "a successor outside the states"
                rows.append(local[r[keep]])
                cols.append(local[c[keep]])
                vals.append(v[keep])
        n = len(states)
        P = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(n, n)) if rows else sp.csr_matrix((n, n))
        system = (sp.identity(n, format="csc") - g * P).tocsc()
        out = {}
        for name, R in parts.items():
            rhs = R[states, pol[states]]
            if name == "own_goal":
                rhs = rhs + const[states, pol[states]]
            v = np.full(S, np.nan)
            v[states] = spl.spsolve(system, rhs)
            out[name] = v
        return out

    def softmax(self, Q, legal, o: int) -> np.ndarray:
        """Eq. 7 of a model of robot o: at its partner's tau (``Config.robot_temperatures``:
        each robot reads its partner at its own tau)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            z = Q / self.cfg.tau_of(1 - o)
            z = z - np.where(legal.any(axis=1, keepdims=True), z.max(axis=1, keepdims=True), 0.0)
            psi = np.where(legal, np.exp(z), 0.0)
            return psi / np.maximum(psi.sum(axis=1, keepdims=True), 1e-300)

    def _fixed(self, o: int, physical: bool) -> np.ndarray:
        """Without a social term a robot is worth vdone once its own goal is done."""
        fixed = np.zeros(self.S, dtype=bool)
        if physical:
            for b in range(self.B):
                if self.dn[b][o]:
                    fixed[[b * self.NN + pair for pair in self.pairs]] = True
        return fixed

    # ---- level 0 -----------------------------------------------------------------------------
    def level0(self, o: int, k: int, obj=None):
        """Robot o's level-0 model with destination k (and goal object ``obj``, its own
        unless given): its partner is a moving obstacle whose hands do nothing.
        -> V, Q, psi"""
        w, cfg, N, NN = self.w, self.cfg, self.N, self.NN
        p, g = 1 - o, cfg.gamma
        q = o if obj is None else obj
        goal = tuple(self.dest[k] if i == o else None for i in (0, 1))
        gobj = tuple(q if i == o else i for i in (0, 1))
        nb = self.nexts(goal, (None, None), tuple(q if i == o else None for i in (0, 1)),
                        tuple(i == o for i in (0, 1)), gobj, gobj)
        legal = np.zeros((self.S, A), dtype=bool)
        R = np.zeros((self.S, A))
        const = np.zeros((self.S, A))
        trip = [([], [], []) for _ in range(A)]
        for b in range(self.B):
            for pair in self.pairs:
                s, x = b * NN + pair, divmod(pair, N)
                if self.busy(b, p):              # lifting (done and parked or not): it
                    pp = [1.0 if a == PICK else 0.0 for a in range(A)]      # lifts on
                elif self.dn[b][p] and cfg.partner_parks:
                    pp = self.stay(x[p], cfg.park_eps, p)
                else:
                    ok = [self.can_move(p, x[p], a) for a in range(5)]
                    pp = [1.0 / sum(ok) if v else 0.0 for v in ok] + [0.0] * (A - PICK)
                for a in range(A):
                    if not self.legal_action(b, o, x, False, a):
                        continue
                    legal[s, a] = True
                    for ap in range(A):
                        if pp[ap] == 0.0:
                            continue
                        j = a * A + ap if o == 0 else ap * A + a
                        n2 = int(w.psucc[pair, j])
                        n = divmod(n2, N)
                        b2 = int(nb[b, pair, j])
                        R[s, a] += pp[ap] * (self.rate(b2, o, n, k, q) - self.cost(b, o, a))
                        if self.dn[b2][o]:
                            const[s, a] += g * pp[ap] * self.vdone
                        else:
                            for lst, v in zip(trip[a], (s, b2 * NN + n2, pp[ap])):
                                lst.append(v)
        V, Q = self.solve(legal, R, const, trip, self._fixed(o, True))
        psi = self.softmax(Q, legal, o)
        if cfg.partner_parks:                              # the predictions that are overridden
            hard = not cfg.idle_hypothesis
            for b in range(self.B):
                for pair in self.pairs:
                    x = divmod(pair, N)
                    held = self.loc[b].index(o) if o in self.loc[b] else None
                    if self.busy(b, o):          # lifting: it lifts on, parked or not
                        continue
                    if ((hard and self.dn[b][o]) or (self.ref.auto and not self.dn[b][o]
                                                     and x[o] == self.dest[k] and held == q)):
                        psi[b * NN + pair] = self.stay(x[o], cfg.park_eps, o)
        return V, Q, psi

    # ---- level 1 -----------------------------------------------------------------------------
    def level1(self, o: int, gk: int, coeff, belief, idle, psis, carry: bool = False,
               obj=None, goals=None, extra: dict = None):
        """Robot o's level-1 model: destination gk (goal object ``obj``, its own unless
        given), social term ``coeff[0] r(g_j) + coeff[1] r(g_i) + coeff[2] race``, belief /
        idle over its partner's physical goals ``goals`` [(object, destination)] (by default
        the partner's own object to each destination), psis[k] (S, 10) the partner's level-0
        policies.  carry: a partner that holds robot o's object is predicted to deliver it to
        robot o's destination (``Config.partner_delivers``) -- but, where its object is
        inferred, not under the hypotheses that it wants that object itself.
        Every hypothesis k has its own outcome: what robot o sets down for its partner
        counts exactly when the cell is hypothesis k's destination (and, inferred, the
        object its object; a hand-over, when the object is its object).
        extra: a dict that receives the terms of the reward apart ("parts": {name: (S, 10)}
        -- own_goal, cost, partner_goal (a1 r(g_j)), own_term (b1 r(g_i)), race (c1 race),
        as SocialMDP.value_components names them), the transitions ("trip") and "const".
        -> V, Q, P, R, legal, psi"""
        w, cfg, N, NN = self.w, self.cfg, self.N, self.NN
        p, g = 1 - o, cfg.gamma
        q = o if obj is None else obj
        goals = [(p, k) for k in range(len(self.dest))] if goals is None else list(goals)
        K = len(goals)
        a1, b1, c1 = coeff
        parts = {name: np.zeros((self.S, A)) for name in
                 ("own_goal", "cost", "partner_goal", "own_term", "race")}
        social = a1 != 0.0 or c1 != 0.0          # it may hand over / set down for its partner
        physical = a1 == 0.0 and b1 == 0.0 and c1 == 0.0         # no social term
        infer = self.ref.infer
        if infer:
            dyn = [self.hypothesis_dynamics(o, gk, q, qh, kh, social) for qh, kh in goals]
        else:
            fail, works = self.model_dynamics(o, gk)
        legal = np.zeros((self.S, A), dtype=bool)
        R = np.zeros((self.S, A))
        Pm = np.zeros((self.S, A))
        const = np.zeros((self.S, A))
        trip = [([], [], []) for _ in range(A)]
        for b in range(self.B):
            delivers = carry and b < self.B - 1 and self.loc[b][q] == p
            for pair in self.pairs:
                s, x = b * NN + pair, divmod(pair, N)
                rest = self.stay_in(b, x[p], cfg.idle_eps, p)
                pk = [self.delivering(x[p], self.dest[gk], p)
                      if delivers and not (infer and goals[k][0] == q) else psis[k][s]
                      for k in range(K)]
                wk = np.array([[belief[k] * ((1.0 - idle[k]) * pk[k][ap] + idle[k] * rest[ap])
                                for ap in range(A)] for k in range(K)])
                wk /= wk.sum()
                Pm[s] = wk.sum(axis=0)
                for a in range(A):
                    if not self.legal_action(b, o, x, social, a):
                        continue
                    legal[s, a] = True
                    for ap in range(A):
                        j = a * A + ap if o == 0 else ap * A + a
                        n2 = int(w.psucc[pair, j])
                        n = divmod(n2, N)
                        for k in range(K):
                            if wk[k, ap] == 0.0:
                                continue
                            qh, kh = goals[k]
                            if infer:             # (a robot without social term never
                                b2 = int(dyn[k][b, pair, j])   # acts for its partner)
                            else:
                                counts = social and a == PLACE and n[o] == self.dest[kh]
                                b2 = int((works if counts else fail)[b, pair, j])
                            ro = self.rate(b2, o, n, gk, q)
                            rj = self.rate(b2, p, n, kh, qh)
                            race = 0.0
                            if c1 != 0.0:
                                race = self.race(b2, o, n, gk, ro, rj,
                                                 rj if self.finished(b2, p, n, kh, qh) else 0.0,
                                                 q)
                            R[s, a] += wk[k, ap] * (ro - self.cost(b, o, a)
                                                    + a1 * rj + b1 * ro + c1 * race)
                            parts["own_goal"][s, a] += wk[k, ap] * ro
                            parts["cost"][s, a] -= wk[k, ap] * self.cost(b, o, a)
                            parts["partner_goal"][s, a] += wk[k, ap] * a1 * rj
                            parts["own_term"][s, a] += wk[k, ap] * b1 * ro
                            parts["race"][s, a] += wk[k, ap] * c1 * race
                            if physical and self.dn[b2][o]:
                                const[s, a] += g * wk[k, ap] * self.vdone
                            else:
                                for lst, v in zip(trip[a], (s, b2 * NN + n2, wk[k, ap])):
                                    lst.append(v)
        if extra is not None:
            extra.update(parts=parts, trip=trip, const=const)
        V, Q = self.solve(legal, R, const, trip, self._fixed(o, physical))
        return V, Q, Pm, R, legal, self.softmax(Q, legal, o)

    # ---- level 2 -----------------------------------------------------------------------------
    def level2(self, o: int, g_own: int, term, hyps, bel_me, idle_me, psis_me, sigma: float,
               carry: bool, physical: bool, race=0.0, obj=None, goals=None, my_goals=None,
               own_index=None, social=None, extra: dict = None, condition: bool = True):
        """Robot o's level-2 model with destination g_own (goal object ``obj``, its own
        unless given).

        condition : the level-1 partner's belief about robot o's goal, bel_me, conditioned
            on robot o's action before its estimate of robot o's reward is read (the rule
            until 2026-10-30; ``Config.child_conditioning``); False: the belief as it is.

        goals / my_goals : the physical goals [(object, destination)] the hypotheses' index k
            stands for, of the partner and of robot o (by default each robot's own object to
            each destination); own_index : robot o's own physical goal among my_goals (for
            "same").

        race : the coefficient of robot o's own race term (a competitor that races): its
            goal reward and its lead once its own goal is finished, minus its partner's
            goal reward while only its partner's is -- under every hypothesis about its
            partner by that hypothesis' destination.  A number, or ``race(t)`` per social
            goal number t of the partner (a rival races some partners and not others).
        term(t, same) -> (alpha, beta, nu): its social term ``alpha R_j + beta r(g_i) + nu
            r(g_j)`` against a partner with social goal number t whose destination is
            (same) / is not its own.
        hyps : [(k, t, weight, idle, psi (S, 10), (a1, b1, c1))] -- the partner at level 1
            with destination k, social goal number t and level-1 term ``a1 r(g_o) + b1
            r(g_p) + c1 race``.
        bel_me[k], idle_me[k] (K,) : that partner's own belief about robot o's physical goal.
        psis_me[k'] (S, 10) : robot o's level-0 policies (what that partner reads it by).
        sigma : 1 if the partner's action cost counts in its reward R_j, else 0.
        carry : the level-1 partner takes a robot o holding its object to be delivering it --
            where objects are inferred, under its hypotheses that robot o wants another
            object (under the others robot o carries its own, read by its policies), reading
            robot o by its way to the partner's destination.
        physical : the term vanishes against every hypothesis (and race is 0).
        social : may it act for its partner (hand over, put down on a hand-off cell, set
            down on a destination)?  By default where it has a term (``not physical``); the
            twin of a robot without a goal of its own (its moves' costs only: beta = -1 and
            nothing on its partner) has no term on its partner and, like that robot's twin at
            level 1, never acts for it.
        extra: a dict that receives the terms of the reward apart ("parts": {name: (S, 10)}
        -- own_goal, cost, partner_reward (alpha R_j), own_term (beta r(g_i)), partner_goal
        (nu r(g_j)), race -- as SocialMDP.value_components names them), the transitions
        ("trip") and "const".
        -> V, Q, P, R, legal"""
        w, cfg, N, NN = self.w, self.cfg, self.N, self.NN
        p, g = 1 - o, cfg.gamma
        q = o if obj is None else obj
        n_dest = len(self.dest)
        parts = {name: np.zeros((self.S, A)) for name in
                 ("own_goal", "cost", "partner_reward", "own_term", "partner_goal", "race")}
        goals = [(p, k) for k in range(n_dest)] if goals is None else list(goals)
        my_goals = [(o, k) for k in range(n_dest)] if my_goals is None else list(my_goals)
        own_index = g_own if own_index is None else own_index
        K, Kme = len(goals), len(my_goals)
        social = (not physical) if social is None else bool(social)
        race_of = race if callable(race) else (lambda t: race)
        infer = self.ref.infer
        if infer:
            dyn = {k: self.hypothesis_dynamics(o, g_own, q, *goals[k], social=social)
                   for k in range(K)}
        else:
            fail, works = self.model_dynamics(o, g_own)
        legal = np.zeros((self.S, A), dtype=bool)
        R = np.zeros((self.S, A))
        Pm = np.zeros((self.S, A))
        const = np.zeros((self.S, A))
        trip = [([], [], []) for _ in range(A)]
        for b in range(self.B):
            held = None if b == self.B - 1 or o not in self.loc[b] else self.loc[b].index(o)
            for pair in self.pairs:
                s, x = b * NN + pair, divmod(pair, N)
                rest = self.stay_in(b, x[p], cfg.idle_eps, p)
                rest_me = self.stay_in(b, x[o], cfg.idle_eps, o)
                wh = np.array([[bh * ((1.0 - qh) * psi[s, ap] + qh * rest[ap])
                                for ap in range(A)] for (_, _, bh, qh, psi, _) in hyps])
                wh /= wh.sum()
                Pm[s] = wh.sum(axis=0)
                for a in range(A):
                    if not self.legal_action(b, o, x, social, a):
                        continue
                    legal[s, a] = True
                    post = {}                    # each level-1 partner's posterior over g_o
                    for k in range(K):
                        delivering = carry and held is not None and held == goals[k][0]
                        if not condition:
                            om = np.array(bel_me[k], dtype=float)
                        elif delivering and not infer:
                            om = np.array(bel_me[k], dtype=float)
                        else:
                            way = self.delivering(x[o], self.dest[goals[k][1]], o)
                            om = np.array([bel_me[k][k2] * (
                                way[a] if delivering and my_goals[k2][0] != held
                                else (1.0 - idle_me[k][k2]) * psis_me[k2][s, a]
                                + idle_me[k][k2] * rest_me[a]) for k2 in range(Kme)])
                            if om.sum() <= 0.0:
                                om = np.array(bel_me[k], dtype=float)
                        post[k] = om / om.sum()
                    for ap in range(A):
                        j = a * A + ap if o == 0 else ap * A + a
                        n2 = int(w.psucc[pair, j])
                        nn = divmod(n2, N)
                        for h, (k, t, _, _, _, (a1, b1, c1)) in enumerate(hyps):
                            if wh[h, ap] == 0.0:
                                continue
                            qh, kh = goals[k]
                            if infer:
                                b2 = int(dyn[k][b, pair, j])
                            else:
                                counts = social and a == PLACE and nn[o] == self.dest[kh]
                                b2 = int((works if counts else fail)[b, pair, j])
                            ro = self.rate(b2, o, nn, g_own, q)
                            rj = self.rate(b2, p, nn, kh, qh)
                            ch = 0.0
                            if a1 != 0.0 or c1 != 0.0:   # the partner's estimate of r(g_o)
                                ch = sum(post[k][k2] * self.rate(b2, o, nn, my_goals[k2][1],
                                                                  my_goals[k2][0], True)
                                         for k2 in range(Kme))
                            Rj = rj * (1.0 + b1) + a1 * ch - sigma * self.cost(b, p, ap)
                            if c1 != 0.0:        # a racing partner: robot o's goal reward as
                                lost = sum(      # it takes it to be, and that where it takes
                                    post[k][k2] * self.rate(b2, o, nn, my_goals[k2][1],  # o's
                                                            my_goals[k2][0], True)   # goal as
                                    for k2 in range(Kme)                             # done
                                    if self.finished(b2, o, nn, my_goals[k2][1], my_goals[k2][0],
                                                     True))
                                Rj += c1 * self.race(b2, p, nn, kh, rj, ch, lost, qh)
                            same = goals[k] == (q, g_own) if infer else k == own_index
                            al, be, nu = term(t, same)
                            mine, c_race = 0.0, race_of(t)
                            if c_race != 0.0:
                                mine = self.race(b2, o, nn, g_own, ro, rj,
                                                 rj if self.finished(b2, p, nn, kh, qh) else 0.0,
                                                 q)
                            R[s, a] += wh[h, ap] * (ro - self.cost(b, o, a) + al * Rj
                                                    + be * ro + nu * rj + c_race * mine)
                            parts["own_goal"][s, a] += wh[h, ap] * ro
                            parts["cost"][s, a] -= wh[h, ap] * self.cost(b, o, a)
                            parts["partner_reward"][s, a] += wh[h, ap] * al * Rj
                            parts["own_term"][s, a] += wh[h, ap] * be * ro
                            parts["partner_goal"][s, a] += wh[h, ap] * nu * rj
                            parts["race"][s, a] += wh[h, ap] * c_race * mine
                            if physical and self.dn[b2][o]:
                                const[s, a] += g * wh[h, ap] * self.vdone
                            else:
                                for lst, v in zip(trip[a], (s, b2 * NN + n2, wh[h, ap])):
                                    lst.append(v)
        if extra is not None:
            extra.update(parts=parts, trip=trip, const=const)
        V, Q = self.solve(legal, R, const, trip, self._fixed(o, physical))
        return V, Q, Pm, R, legal
