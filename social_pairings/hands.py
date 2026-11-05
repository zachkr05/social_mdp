"""The rule of the shared-object world: what the robots' hands do in one transition.

Everything a model or the rollout knows about objects changing hands is a table built from
:func:`hands_step`.  The state of the objects is a *configuration*: the status of every
object (at its place, held by R0, held by R1, set down, lying on one of the scene's
hand-off cells, -- with ``Config.swap_on_place`` -- lying on another object's place, or --
``Config.pick_steps > 1`` -- being lifted by a robot, so many steps into the lift) plus
which goals are done.

Five hand actions (since 2026-10-16; until then one action, "use", did the first of these
that applied).  The robot stays where it is and:

* pick -- free hands on a cell where an object lies that it may hold: picks it up (its own
  object if that lies there, else the first other one).  With ``Config.pick_steps`` = n > 1
  the pick-up takes n transitions: the first starts the lift, and the robot is then busy --
  "pick" is the one action open to it -- until the n-th puts the object in its hands; the
  object is in nobody's hands meanwhile (the partner cannot take it, nor hand the lifter
  anything).  The cost (``Config.pick_cost_rate``) is charged on the first transition.
* place -- holding X: on X's place puts X back (unless another object lies there); on a
  hand-off cell puts X down, where it lies until somebody picks it up; on a candidate
  destination sets X down for the partner, which counts (the partner's goal is done) only
  on the partner's destination and with the partner's object, else nothing happens ("set
  down in vain").  Since 2026-10-18 (``Config.automatic_hands=False``) it is also how a robot
  FINISHES: holding its goal object on its own destination, "place" sets it down and its
  goal is done.
* take -- free hands, the partner next to it holding X: takes X from it (``Config.steal``;
  since 2026-10-12, off by default) -- unless the partner lets go of X in the same
  transition
* give -- holding X, the partner next to it: hands X over, if the partner's hands are free
  -- or if the partner gives in the same transition, in which case the two objects are
  swapped (not with ``Config.hand_over=False``: nothing then goes from hand to hand)
* swap -- holding X on the place of an object, with an object Y lying there: Y into its
  hands, X onto that cell (``Config.swap_on_place``; since 2026-10-12, off by default).  On
  X's own place that puts X back.  One transition, whatever ``pick_steps`` (the user,
  2026-10-18: "Swap should be its own action i.e it should be one timestep").

A hand action that does not apply does nothing (and is not offered: ``ObjectWorld.hand_ok``).

Robot i's goal is done by ITS goal object on robot i's destination, and by nothing else; in
the world robot i's goal object is object i (``gobj``), and a robot is handed, or has set
down for it, its goal object and no other (``acc``).  A model of a robot that does not know
which object its partner wants (``Config.partner_object="inferred"``) has two tables for its
own acts for the partner: one in which the partner takes no object (``acc = -1``) and one in
which it takes whatever it is offered (``acc = -2``) -- the solver mixes the two by the
belief that the object (and, for a set-down, the cell) is the partner's.  Handing over,
putting down on a hand-off cell and setting down for the partner are acts for the partner --
each brings an object within the partner's reach: only a robot with a social goal may
choose them (and, where it knows its partner's object, only with that object:
``ObjectWorld.hand_ok``).  Putting an object down on a hand-off cell always works: what lies
there anybody may pick up.  (Picking up and putting back are open to every robot.)
Nobody is made to let go of what it holds (``Config.forced_swap`` restores the older rule in
which a robot handed its own object had to take it and give up what it held), unless
``Config.steal``: then a robot with free hands next to it takes it.  Swapping and stealing
are open to every robot, like picking up and putting back.

Automatic, without a hand action -- ONLY with ``Config.automatic_hands=True``, the rule
until 2026-10-18: a robot with free hands takes its OWN object on arriving at the cell where
it lies (its place, or a hand-off cell), and a robot holding its own object sets it down on
arriving at its destination.  Without it (the default since 2026-10-18; the user: "nothing
is automatic ... you must move to the cell then select the pickup dropoff etc action") a
robot's own object is picked up and set down like any other: by "pick" and "place".

Hand-off cells (``Scene.handoff_cells``) are what make an exchange without a meeting
possible: one robot puts its partner's object down, the partner picks it up later.  Any
number of objects may lie on one hand-off cell.  A scene without them has the four statuses
and the rule of before.
"""
from __future__ import annotations

import itertools

import numpy as np
from numba import njit, prange

from .constants import ARRIVED, GONE, HELD0, HELD1, HOME, HOW_PICK, N_HOW

__all__ = ["hands_step", "encode", "build_table", "configurations", "lifter", "lift_stage"]


@njit(cache=True)
def _place_own(st, dn, site, gsite, gobj):
    """``automatic_hands``: a robot holding ITS GOAL object (gobj[i]; object i in the world)
    on its own (modelled) destination sets it down: its goal is done and the object stays
    there.  No other object completes its goal.  -> True if anything changed."""
    changed = False
    for i in range(2):
        q = gobj[i]
        if q < 0 or dn[i] != 0 or gsite[i] <= 0 or site[i] != gsite[i]:
            continue
        if st[q] == 1 + i:
            st[q] = 3
            dn[i] = 1
            changed = True
    return changed


@njit(cache=True)
def _takes(acc, k):
    """Does a robot whose acceptance is ``acc`` take object k as its goal object -- handed
    over, or set down for it?  acc >= 0: that object only; -1: none; -2: any."""
    return acc == -2 or (acc >= 0 and acc == k)


@njit(cache=True)
def lifter(status, lift0, nstage):
    """The robot lifting an object with this status (``Config.pick_steps`` > 1: statuses
    lift0 + i * nstage + (s - 1), robot i s steps into the lift, s = 1..nstage), -1 if none
    is."""
    if nstage <= 0 or status < lift0:
        return -1
    i = (status - lift0) // nstage
    return i if i < 2 else -1


@njit(cache=True)
def lift_stage(status, lift0, nstage):
    """How many steps into its lift the object with this status is (0: not being lifted)."""
    if nstage <= 0 or status < lift0 or status >= lift0 + 2 * nstage:
        return 0
    return (status - lift0) % nstage + 1


@njit(cache=True)
def _lies_on(status, place, hsite, lift0, nstage):
    """The special cell (site id) an object with this status lies on: its place while it
    is there, a hand-off cell while it lies on one (status 4 + h), else 0 -- it is in
    somebody's hands, being lifted (under the lifter, out of everybody else's reach) or set
    down for good."""
    if status == 0:
        return place
    if status >= 4 and status < lift0:
        return hsite[status - 4]
    return 0


@njit(cache=True)
def _lying_code(site, k, home, hand_at, place_at):
    """The status of object k lying on ``site``: at its place (0), on hand-off cell h
    (4 + h) or on another object's place (4 + place_at[site]); -1 where no object can lie
    (a candidate destination: no lift begins there)."""
    if home[k] == site:
        return 0
    if hand_at[site] >= 0:
        return 4 + hand_at[site]
    if place_at[site] >= 0:
        return 4 + place_at[site]
    return -1


@njit(cache=True)
def hands_step(st, dn, site, how, adj, home, gsite, dsite, iscand, own, gobj, acc, rights,
               forced, canuse, after_done, hsite, hand_at, give, swap, steal, place_at, auto,
               lift0, nstage):
    """THE RULE of the shared-object world: what the hands do in one transition, in place
    on st (M,) object statuses and dn (2,) goal-done flags.  Every table is built from it.
    Statuses: 0 at its place, 1 / 2 in R0's / R1's hands, 3 set down for good, 4 + h lying
    on cell h of hsite, lift0 + i * nstage + (s - 1) being lifted by robot i, s steps into
    the lift (:func:`lifter`, :func:`lift_stage`; nstage = ``Config.pick_steps`` - 1, 0: a
    pick-up is instant and there are no such statuses).  hsite (L,): the sites an object
    may lie on besides its own place -- the scene's hand-off cells first (H of them), then,
    with swap, the places of the M objects (status 4 + H + p: lying on object p's place);
    hand_at[site]: the hand-off cell at that site, or -1; place_at[site]: the cell of hsite
    that is an object's place at that site (H + p), or -1.  give (``Config.hand_over``): an
    object may be handed straight to the partner; False: it changes hands only by being put
    down and picked up.  swap (``Config.swap_on_place``): full hands on an object's place
    swap what they hold for what lies there.  steal (``Config.steal``, by robot
    ``Config.steal_robots``): bit i set, robot i's free hands take what the partner next to
    it holds.  auto (``Config.automatic_hands``): a robot takes its
    own object on arriving where it lies and sets it down on arriving at its destination;
    False: by "pick" and "place" only, like any other object.

    site (2,): the special cell each robot stands on after the move (0 = none; ids shared
    with home (M,), gsite, dsite and iscand); how (2,): ARRIVED / STAYED / BLOCKED, or the
    hand action it chose -- HOW_PICK, HOW_PLACE, HOW_TAKE, HOW_GIVE, HOW_SWAP
    (constants.py); adj: the robots are 4-neighbours after the move.
    gsite (2,): where robot i sets down what it holds for its OWN goal (0 = this dynamics
    does not place for robot i; -2, auto=False only: on any candidate destination, with
    whatever it holds that its partner does not take -- a robot whose destination this
    dynamics does not know, the partner in a robot's model: under each hypothesis about it
    its policy sets down only its goal object on its destination, so the mixture over the
    hypotheses comes out right); dsite (2,): where robot i's "set down for my partner"
    succeeds (0 = nowhere, -2 = wherever it tries); iscand[site]: candidate destination;
    own (2,): the object robot i takes on arriving at its place (-1 none; auto only); gobj
    (2,): robot i's goal object -- the one it sets down on gsite[i], and the one it picks up
    first (-1: this dynamics does not know it); acc (2,): what robot i takes as its goal
    object when it is handed over or set down for it (q; -1 nothing; -2 anything); rights
    (2,): bit q = robot i may hold object q.  In the world gobj = acc = (0, 1): object i,
    and no other, completes robot i's goal, and only object i is handed to robot i or set
    down for it.  forced (Config.forced_swap): robot i takes its own object from its
    partner's hands even when its own are full, letting go of what it held to the giver;
    canuse (2,): robot i's hands work (False: an inert partner -- it neither uses nor
    receives; a lift of its is dropped); after_done: a robot whose goal is done still uses
    its hands.

    Order: (0) lifts in progress: the lifter's "pick" takes it one step on -- the object
    into its hands on the last; anything else of it drops the object where it lies (it is
    offered nothing else; the inert partner of a level-0 model, canuse False, drops it too;
    a move away cannot happen and leaves the status as it is, as does a lifter standing on
    no special cell, which cannot happen either -- the kernels keep the block without a
    look-up where neither robot stands on one nor chose a hand action).  (1) auto: own goal: holding
    its goal object on its own destination -> set down.  (2) the hand actions of the robots
    not lifting, both at once, judged on the state after (1): pick (an object lying there,
    its goal object first: the lift begins, or, nstage 0, it is in the hands); place (not
    auto: its goal object on its own destination -> done; put back on its place, put down on
    a hand-off cell, or set down for the partner on a candidate destination -- an object the
    partner takes, acc; not auto, gsite -2: else done for its own goal); give (an object the
    partner takes, the partner not lifting); swap; take what the partner holds -- after
    everything else, and only if the partner still holds it.  A hand action that does not
    apply does nothing.  (3) auto, repeated until nothing changes: a robot that ARRIVED with
    free hands on the cell where its own object (own) lies takes it; (1)."""
    M = st.shape[0]
    if auto:
        _place_own(st, dn, site, gsite, gobj)
    # (0) lifts in progress
    busy = np.zeros(2, dtype=np.bool_)
    for k in range(M):
        i = lifter(st[k], lift0, nstage)
        if i < 0:
            continue
        busy[i] = True
        if how[i] == HOW_PICK and canuse[i] and (dn[i] == 0 or after_done):
            if lift_stage(st[k], lift0, nstage) >= nstage:
                st[k] = 1 + i                    # the last step: into its hands
            else:
                st[k] = st[k] + 1
        elif how[i] == ARRIVED or site[i] <= 0:
            pass                                 # cannot happen: it is not offered a move,
        else:                                    # and lifts on special cells (a quiet
            code = _lying_code(site[i], k, home, hand_at, place_at)    # event keeps a block)
            clash = code < 0                     # (nor a lift where no object can lie, nor
            if hand_at[site[i]] < 0:             # another object on the place it would go
                for q in range(M):               # back to -- one object to a place: the
                    if q != k and _lies_on(st[q], home[q], hsite, lift0, nstage) == site[i]:
                        clash = True             # lift stays as it is)
            if not clash:
                st[k] = code
    held = np.full(2, -1, dtype=np.int64)
    for k in range(M):
        if st[k] == 1:
            held[0] = k
        elif st[k] == 2:
            held[1] = k
    # 1 put back, 2 hand over, 3 set down for the partner, 4 pick up, 5 put down (hand-off),
    # 6 swap (on an object's place), 7 take from the partner's hands, 8 set down for its own
    # goal (not auto)
    intent = np.zeros(2, dtype=np.int64)
    grab = np.full(2, -1, dtype=np.int64)
    for i in range(2):
        h = how[i]
        if busy[i] or h < 3 or not canuse[i] or (dn[i] != 0 and not after_done):
            continue
        j = 1 - i
        k = held[i]
        s = site[i]
        if h == 3:                           # pick: what lies here, its goal object first
            if k < 0 and s > 0:
                g = gobj[i]
                if (g >= 0 and g < M and ((rights[i] >> g) & 1)
                        and _lies_on(st[g], home[g], hsite, lift0, nstage) == s):
                    intent[i] = 4
                    grab[i] = g
                else:
                    for q in range(M):
                        if (_lies_on(st[q], home[q], hsite, lift0, nstage) == s
                                and ((rights[i] >> q) & 1)):
                            intent[i] = 4
                            grab[i] = q
                            break
        elif h == 4:                         # place what it holds on this cell
            if k >= 0 and s > 0:
                mine = (not auto and dn[i] == 0 and gsite[i] > 0 and s == gsite[i]
                        and (gobj[i] < 0 or k == gobj[i]))
                if mine:                     # its goal object on its destination: done
                    intent[i] = 8
                elif home[k] == s:           # back on its place -- if nothing else lies there
                    free = True
                    for q in range(M):
                        if q != k and _lies_on(st[q], home[q], hsite, lift0, nstage) == s:
                            free = False
                    if free:
                        intent[i] = 1
                elif hand_at[s] >= 0:
                    intent[i] = 5
                elif iscand[s] and dn[j] == 0 and _takes(acc[j], k):
                    intent[i] = 3
                elif not auto and dn[i] == 0 and gsite[i] == -2 and iscand[s]:
                    intent[i] = 8
        elif h == 5:                         # take what the partner next to it holds
            if (k < 0 and ((steal >> i) & 1) != 0 and adj and held[j] >= 0
                    and ((rights[i] >> held[j]) & 1)):
                intent[i] = 7
                grab[i] = held[j]
        elif h == 6:                         # give what it holds to the partner next to it
            if (k >= 0 and give and adj and canuse[j] and not busy[j]
                    and (after_done or dn[j] == 0)
                    and ((rights[j] >> k) & 1) and _takes(acc[j], k)
                    and (held[j] < 0 or how[j] == 6 or forced)):
                intent[i] = 2
        elif h == 7:                         # swap what it holds for what lies on this place
            if k >= 0 and swap and s > 0 and place_at[s] >= 0:
                for q in range(M):
                    if (q != k and ((rights[i] >> q) & 1)
                            and _lies_on(st[q], home[q], hsite, lift0, nstage) == s):
                        intent[i] = 6
                        grab[i] = q
                        break
    for i in range(2):
        k = held[i]
        if intent[i] == 1:
            st[k] = 0
        elif intent[i] == 5:
            st[k] = 4 + hand_at[site[i]]
        elif intent[i] == 3:
            if dsite[i] == -2 or dsite[i] == site[i]:
                st[k] = 3
                dn[1 - i] = 1
            elif not auto and gsite[i] == -2:
                intent[i] = 8                # not its partner's destination: then, by its
        elif intent[i] == 4:                 # own hypothesis, its own (below)
            st[grab[i]] = (lift0 + i * nstage) if nstage > 0 else 1 + i
        elif intent[i] == 6:                 # what lay there into its hands, what it held
            st[grab[i]] = 1 + i              # onto the cell: its own place, or another's
            st[k] = 0 if home[k] == site[i] else 4 + place_at[site[i]]
    if intent[0] == 2 and intent[1] == 2:                    # both hold, both give: trade
        st[held[0]] = 2
        st[held[1]] = 1
    else:
        for i in range(2):
            j = 1 - i
            if intent[i] != 2:
                continue
            k = held[i]
            m = held[j]
            if m < 0:
                if intent[j] != 4:
                    st[k] = 1 + j
            elif forced:                     # full hands take it all the same and let go of
                if st[m] == 1 + j:           # what they still hold, to the giver
                    st[m] = 1 + i
                st[k] = 1 + j
    for i in range(2):                       # taken from the partner's hands -- if it is
        if intent[i] == 7 and st[grab[i]] == 2 - i:          # still there
            st[grab[i]] = 1 + i
    for i in range(2):                       # set down for its own goal -- unless its partner
        if intent[i] == 8 and dn[i] == 0:    # finished it for it in this very transition
            st[held[i]] = 3
            dn[i] = 1
    if not auto:
        return
    for _ in range(3):
        changed = False
        for i in range(2):
            q = own[i]
            if (q < 0 or how[i] != 0 or dn[i] != 0 or site[i] <= 0
                    or _lies_on(st[q], home[q], hsite, lift0, nstage) != site[i]
                    or not ((rights[i] >> q) & 1)):
                continue
            free = True
            for k in range(M):
                if st[k] == 1 + i:
                    free = False
            if free:
                st[q] = 1 + i
                changed = True
        if _place_own(st, dn, site, gsite, gobj):
            changed = True
        if not changed:
            break


@njit(cache=True)
def encode(st, dn, pw):
    code = 0
    for k in range(st.shape[0]):
        code += st[k] * pw[k]
    return code + (dn[0] + 2 * dn[1]) * pw[st.shape[0]]


@njit(parallel=True, cache=True)
def build_table(status, done, code2blk, pw, nsite, home, gsite, dsite, iscand, own, gobj, acc,
                rights, forced, canuse, after_done, hsite, hand_at, give, swap, steal, place_at,
                auto, lift0, nstage, out):
    """out[c, ev] = configuration block after the hands acted (hands_step) from block c
    under event ev = ((site0 * N_HOW + how0) * E1 + site1 * N_HOW + how1) * 2 + adj, E1 =
    N_HOW (nsite + 1).  Events in which both robots stand on the same special cell cannot
    occur and keep the block."""
    B, M = status.shape
    E1 = N_HOW * (nsite + 1)
    for c in prange(B):
        st = np.empty(M, dtype=np.int64)
        dn = np.empty(2, dtype=np.int64)
        site = np.empty(2, dtype=np.int64)
        how = np.empty(2, dtype=np.int64)
        for e0 in range(E1):
            for e1 in range(E1):
                site[0] = e0 // N_HOW
                site[1] = e1 // N_HOW
                for adj in range(2):
                    ev = (e0 * E1 + e1) * 2 + adj
                    if site[0] == site[1] and site[0] > 0:
                        out[c, ev] = c
                        continue
                    for k in range(M):
                        st[k] = status[c, k]
                    dn[0] = done[c, 0]
                    dn[1] = done[c, 1]
                    how[0] = e0 - site[0] * N_HOW
                    how[1] = e1 - site[1] * N_HOW
                    hands_step(st, dn, site, how, adj == 1, home, gsite, dsite, iscand, own,
                               gobj, acc, rights, forced, canuse, after_done, hsite, hand_at,
                               give, swap, steal, place_at, auto, lift0, nstage)
                    out[c, ev] = code2blk[encode(st, dn, pw)]


def configurations(M: int, n_status: int = 4, places: int = -1, lift0: int = -1,
                   nstage: int = 0) -> list:
    """Every configuration (status_0, ..., status_{M-1}, done_0, done_1): each robot holds
    at most one object -- or lifts one, with free hands -- and exactly one object is set
    down per finished goal.  The configurations with both goals done are merged into ONE
    terminal block (the last): nothing is left to do there.  n_status: 4 plus the number
    of cells an object may lie on besides its place (status 4 + h): the hand-off cells,
    several objects to one, and with ``places`` >= 0 (``Config.swap_on_place``) the M
    objects' places, from status 4 + places on -- one object to a place, and never an
    object on its own place by that status (that is status 0) -- plus, with ``nstage`` > 0
    (``Config.pick_steps`` - 1), the 2 nstage lift statuses from ``lift0`` on
    (:func:`lifter`)."""
    out = []
    lifts = range(lift0, lift0 + 2 * nstage) if nstage > 0 else range(0)
    top = lift0 if nstage > 0 else n_status             # the lying statuses end here
    for flags in ((0, 0), (1, 0), (0, 1)):
        for st in itertools.product(range(n_status), repeat=M):
            if st.count(HELD0) > 1 or st.count(HELD1) > 1 or st.count(GONE) != sum(flags):
                continue
            if nstage > 0:
                lifted = [(v - lift0) // nstage for v in st if v in lifts]
                if (len(lifted) != len(set(lifted))
                        or any(st.count(HELD0 + i) and i in lifted for i in (0, 1))):
                    continue
            if places >= 0:
                on = [k if v == HOME else v - 4 - places if 4 + places <= v < top else -1
                      for k, v in enumerate(st)]
                if (any(v == 4 + places + k for k, v in enumerate(st))
                        or len([p for p in on if p >= 0]) != len({p for p in on if p >= 0})):
                    continue
            out.append(st + flags)
    out.append((GONE,) * M + (1, 1))
    return out
