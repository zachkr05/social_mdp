"""Numba kernels of the solver: expected rewards, Bellman rows, policy iteration, policies.

Joint state ``s = (c * N + x0) * N + x1`` with ``c`` the object configuration ("block"), so
every configuration is one contiguous run of N^2 states.  Work items ``(model, [lo, hi))``
cover at most CHUNK states and never straddle a block; every kernel is a ``prange`` over
items, and reductions run in a fixed order, so results do not depend on the thread count.

A *model* is one robot's MDP for one destination and one social term.  Its partner acts by
``P[p_of[m]]`` (S, 10); its expected rewards per own action are ``R[m]`` (S, 10); its
dynamics are two tables ``tabs[var_of[m]]`` / ``tabs[svar_of[m]]`` (B, nEv) -- what the
robot sets down for its partner failing / succeeding -- mixed by ``pd``, the belief mass of
the hypotheses whose destination is the cell the robot stands on.

A physical goal is an object and a candidate destination.  Where a robot knows which object
its partner wants (``Config.partner_object="known"``) every goal index is a destination and
a robot's object is its own; where it infers it ("inferred") a goal index g stands for the
object ``gobj[g]`` and the destination ``gdst[g]``, and robot i's reward with object q as its
goal is read through ``cls_obj[b, i, q]`` (``ObjectWorld.cls_obj``).  A robot's act for its
partner then counts under the hypotheses that hold the object it offers (and, for a
set-down, the cell): ``pd`` for a set-down on a candidate cell, ``ho`` for a hand-over.

These functions are the verified engine's, moved here unchanged, but for three things: the
physical goals just described (with ``partner_object="known"`` the arrays reduce to the
old ones and every sum to the old one, term for term), and the two below.
``inputs_typed`` also computes the race term of competition (``l1r``, ``cR``, ``tr1``,
``fin_cls``, ``lead``, :func:`_race`), and with ``l1r``, ``cR`` and ``tr1`` zero it is the
verified engine's.  And a robot's legal moves are its own (``valid`` (2, N, 5) in
``policy_iteration`` and ``softmax_policies``, ``valid`` / ``valid_o`` in ``inputs_typed``):
where the robots have rooms (``Scene.rooms``) the two tables differ, and without rooms
they are the one table of before.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .constants import N_ACTIONS, PICK, PLACE

NA = N_ACTIONS                               # actions: four moves, stay, five hand actions

__all__ = ["inputs_level0", "inputs_typed", "q_row", "p_row", "policy_iteration",
           "softmax_policies", "policy_graph"]


@njit(parallel=True, cache=True)
def inputs_level0(o, q, gl, rows, N, psucc, pev, tabgen, valid, busy, cls_obj, rate_cls, cost_c,
                  done, it_lo, it_hi, P, p_row, R, park, park_on, oap_obj, hold, dest):
    """Level 0 of robot o with goal object q: its partner moves uniformly over its legal
    moves and "stay" (its hands do nothing) -- or, with park_on, is parked (table park
    (N, 10)) in the blocks where its goal is done -- or, lifting an object (busy (B, 2),
    ObjectWorld.busy; done and parked or not), goes on with "pick", the one action open to
    it (which its inert hands turn into dropping the lift).  Writes P[p_row] (S, 10) and R[rows[m]]
    (S, 10) = expected r_o(s') - c_o(a_o) for every destination gl[m].  tabgen: robot o's
    level-0 table without its own set-down, one table for every destination; where that
    set-down does happen -- "place", holding object q on destination gl[m] (hold (B, 2),
    dest (K,) the candidate cells) -- the reward of goal m is read in the done class
    (class 0), as the destination's own table has it (with the automatic hands the same,
    the way left being 0 either way).
    oap_obj (B, 2, M) uint8: robot i's reward for object q is read at its PARTNER's cell in
    that block (ObjectWorld.at_partner_obj)."""
    NN = N * N
    M = gl.shape[0]
    p = 1 - o
    for i in prange(it_lo.shape[0]):
        acc = np.empty((M, NA))
        pw = np.empty(NA)
        c = it_lo[i] // NN
        parked = park_on and done[c, p] != 0
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            x0 = pair // N
            xp = pair - x0 * N if o == 0 else x0
            if busy[c, p] != 0:              # lifting (done and parked or not): it lifts on
                for a in range(NA):
                    pw[a] = 1.0 if a == PICK else 0.0
            elif parked:
                for a in range(NA):
                    pw[a] = park[xp, a]
            else:
                L = 0
                for a in range(5):
                    if valid[xp, a]:
                        L += 1
                inv = 1.0 / L
                for a in range(5):
                    pw[a] = inv if valid[xp, a] else 0.0
                for a in range(5, NA):
                    pw[a] = 0.0
            for a in range(NA):
                P[p_row, s, a] = pw[a]
            acc[:, :] = 0.0
            for ao in range(NA):
                for ap in range(NA):
                    if pw[ap] == 0.0:
                        continue
                    j = ao * NA + ap if o == 0 else ap * NA + ao
                    ev = pev[pair, j]
                    c2 = c if ev == 0 else tabgen[c, ev]
                    p2 = psucc[pair, j]
                    no = p2 // N if o == 0 else p2 - (p2 // N) * N
                    if oap_obj[c2, o, q] != 0:   # its object is in its partner's hands: the
                        no = p2 - (p2 // N) * N if o == 0 else p2 // N   # reward follows it
                    cl = cls_obj[c2, o, q]
                    placing = ao == PLACE and hold[c, o] == q and done[c, o] == 0
                    for m in range(M):
                        if placing and no == dest[gl[m]]:
                            acc[m, ao] += pw[ap] * rate_cls[0, no, gl[m]]
                        else:
                            acc[m, ao] += pw[ap] * rate_cls[cl, no, gl[m]]
            for m in range(M):
                for ao in range(NA):
                    R[rows[m], s, ao] = acc[m, ao] - cost_c[c, o, ao]


@njit(cache=True)
def _race(mine_done, mine, other, other_done, lead):
    """The race term of competition under one hypothesis about the partner (goals.py):
    ``mine`` / ``other`` the two goal rewards, ``*_done`` whether that goal is finished.
    Own goal finished: mine + lead (mine - other), with lead = 1 ("race_then_hinder") or 0
    ("race").  Else: -other where the partner's goal is finished, else 0."""
    if mine_done:
        return mine + lead * (mine - other)
    if other_done:
        return -other
    return 0.0


@njit(parallel=True, cache=True)
def inputs_typed(o, N, psucc, pev, tab, dtab, cls_obj, oap_obj, cls_est, oap_est, rate_cls,
                 cost_c, site_of, goal_at_site, it_lo, it_hi, cpsi, crows, cgoal, ctype, Kg, nT,
                 belief, bidle, idle_eps, valid, valid_o, busy, gone, mrows, gown, qown, kown,
                 msoc, mzero,
                 l1a, l1b, l1r, depth2, cA, cB, cN, cR, ta1, tb1, tr1, sigma, fin_cls, lead, mypsi,
                 myrows, mybel, myidle, cond_child, hold, done, carry_p, carry_on, carry_me,
                 carry_o, gobj_p, gdst_p, gobj_o, gdst_o, infer, P, p_row, R, pd, ho):
    """Inputs of the models of one slot with TYPED social goals (ICRA 2022, Table I): robot
    o, one physical goal ``gown`` (object qown, destination kown; dynamics tab / dtab: its
    act for the partner failing / counting).  Partner hypotheses k act by cpsi[crows[k]]
    (or, depth 1 with carry_on, by carry_p where the partner holds robot o's object) with
    physical goal cgoal[k] -- object gobj_p, destination gdst_p of it -- and social-goal
    index ctype[k] < nT; belief[k] / bidle[k] = their weights active / idle.  valid /
    valid_o (N, 5): the legal moves of the partner / of robot o (an idle robot moves over
    its own; the two tables differ only where the robots have rooms).  Writes, with
    w_k(a_p|s) = belief_k psi_k(a_p|s) + bidle_k psi_idle(a_p|s),

      P[p_row](s, a_p)    = sum_k w_k(a_p|s)                                        (Eq. 4)
      R[mrows[m]](s, a_o) = sum_{a_p} sum_k w_k [r_o - c_o + xi_m(k)]                (Eq. 2)
      pd[p_row]           = the mass of the hypotheses under which a set-down for the partner
                            on the candidate cell robot o stands on counts: whose destination
                            is that cell (and, infer, whose object robot o holds)
      ho[p_row](s, a_p)   = (infer) the mass of those under which a hand-over counts: whose
                            object robot o holds

    Every reward is read off the model's own next state: r_i at robot i's cell, or at its
    partner's while the partner carries its object (oap_obj: ObjectWorld.at_partner_obj).
    An act for the partner -- a hand action (place, give) where the table of success (dtab)
    leads elsewhere than the table of failure (tab) -- counts under the hypotheses that take
    it (with the mass pd or ho), and each of them is read off the block it leads to; the
    others off the block of failure.  infer = False (the partner's object known): only a
    set-down counts, under the hypothesis whose destination is the cell, as before.

    depth2 = False (children at level 0; Table I, column l = 1):
      xi_m = l1a[m] r_p^k + l1b[m] r_o      cooperation (1, 0), conflict and the table's
             + l1r[m] race(r_o, r_p^k)      competition (-1, 0), coercion (0, 1), exchange (0, 0)
    with race(x, y) = x + lead (x - y) where the goal x belongs to is finished (fin_cls:
    done, or its robot holding its object on the destination), else -y where the goal y
    belongs to is finished, else 0: the race forms of competition (:func:`_race`,
    goals.level1_term; lead = goals.lead_weight).  Where y is an expectation over
    hypotheses, every hypothesis is judged by whether that goal is finished under it.
    depth2 = True (children at level 1; Table I, column l > 1), child k of type t, with
    same = (the child's physical goal is model m's own):
      R_k  = r_p^k (1 + tb1[t]) + ta1[t] rhat_o^k + tr1[t] race(r_p^k, rhat_o^k)
             - sigma c_p(a_p)
      xi_m = cA[m, t, same] R_k + cB[m, t, same] r_o + cN[m, t] r_p^k
             + cR[m, t] race(r_o, r_p^k)
    The last term is model m's own race term, a competitor at level 2 that races a partner
    of type t (goals.level2_term: under a race form every type, as a rival every type but
    cooperation): the term of the column l = 1 on the goal rewards of child k, judged
    finished under child k's goal (cA, cB, cN are 0 for that type: no R_k in it).
    R_k is the child's own level-1 reward (Eq. 2): its physical reward, its social term
    (tb1: coercion; ta1: +1 cooperation, -1 conflict / competition; tr1: a racing
    competitor) and its action cost
    (sigma = 1; 0 leaves the cost out, as the authors' code does).  rhat_o^k is the child's
    estimate of robot o's reward: r_o^{k'} over the child's own belief about robot o's
    physical goal (object gobj_o, destination gdst_o of k'), mybel[g_k, k'] -- each
    r_o^{k'} read off cls_est / oap_est: in robot o's own model its done flag is its own
    object delivered, no goal of another object done (ObjectWorld.cls_est).  cond_child:
    that belief first conditioned on robot o's action a_o (Bayes with robot o's level-0
    policies mypsi[myrows[k']] and the idle mode myidle[g_k, k']; the rule until
    2026-10-30); False: the belief as the child holds it, Table I's R_j with the partner's
    estimate of the moment (the authors' code).
    (carry_me, with cond_child only: where robot o holds the child's object the child
    takes it to be delivering it, so its action says nothing about its goal -- infer:
    under the child's hypotheses that robot o wants some other object; under those that
    it wants that one it carries its own, and the child reads it by carry_o, robot o's way
    to the child's destination).
    mzero[m]: model m has no social term (its row holds r_o - c_o only)."""
    NN = N * N
    p = 1 - o
    Mc = crows.shape[0]
    M = mrows.shape[0]
    Kme = mybel.shape[1]
    any_idle = False
    for k in range(Mc):
        if bidle[k] > 0.0:
            any_idle = True
    need_ch = False
    if depth2:
        for t in range(nT):
            if ta1[t] != 0.0 or tr1[t] != 0.0:
                need_ch = True
    for i in prange(it_lo.shape[0]):
        pa = np.empty(NA)
        W = np.empty((NA, nT, Kg))
        Wg = np.empty((NA, Kg))
        acc = np.empty((M, NA))
        pidle = np.zeros(NA)
        pme = np.zeros(NA)
        Wme = np.zeros((Kg, NA, Kme))
        rp = np.empty(Kg)
        fp = np.zeros(Kg, dtype=np.uint8)
        rpd = np.zeros(Kg)
        fpd = np.zeros(Kg, dtype=np.uint8)
        cnt = np.zeros(Kg, dtype=np.uint8)
        ch2 = np.zeros(Kg)
        ch3 = np.zeros(Kg)
        chd = np.zeros(Kg)
        chd3 = np.zeros(Kg)
        rme = np.zeros(Kme)
        rmed = np.zeros(Kme)
        fme = np.zeros(Kme, dtype=np.uint8)
        fmed = np.zeros(Kme, dtype=np.uint8)
        c = it_lo[i] // NN
        carry_now = carry_on and hold[c, p] == qown
        xh = hold[c, o]                      # the object robot o holds: what it may offer
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            xq = pair // N
            xr = pair - xq * N
            if o == 0:
                xo = xq
                xp = xr
            else:
                xo = xr
                xp = xq
            if any_idle:
                if busy[c, p] != 0:              # lifting: it goes on with the lift
                    for a in range(NA):
                        pidle[a] = 1.0 if a == PICK else 0.0
                else:
                    nm = 0
                    for a in range(4):
                        if valid[xp, a]:
                            nm += 1
                    for a in range(4):
                        pidle[a] = idle_eps / nm if (nm > 0 and valid[xp, a]) else 0.0
                    pidle[4] = 1.0 - idle_eps if nm > 0 else 1.0
                    for a in range(5, NA):
                        pidle[a] = 0.0
            pa[:] = 0.0
            W[:, :, :] = 0.0
            for k in range(Mc):
                bk = belief[k]
                ik = bidle[k]
                if bk == 0.0 and ik == 0.0:
                    continue
                row = crows[k]
                gk = cgoal[k]
                tk = ctype[k]
                # a partner holding robot o's object delivers it -- but, where its object is
                # inferred, not under the hypotheses that it wants that object itself
                delivers = carry_now and ((not infer) or gobj_p[gk] != qown)
                for ap in range(NA):
                    if delivers:
                        wk = bk * carry_p[xp, ap]
                    else:
                        wk = bk * np.float64(cpsi[row, s, ap])
                    if ik != 0.0:
                        wk += ik * pidle[ap]
                    pa[ap] += wk
                    W[ap, tk, gk] += wk
            z = 0.0
            for ap in range(NA):
                z += pa[ap]
            if z > 0.0:                      # renormalise: rows sum to one exactly
                z = 1.0 / z
                for ap in range(NA):
                    pa[ap] *= z
                    for t in range(nT):
                        for gg in range(Kg):
                            W[ap, t, gg] *= z
            for ap in range(NA):
                P[p_row, s, ap] = pa[ap]
                for gg in range(Kg):
                    v = 0.0
                    for t in range(nT):
                        v += W[ap, t, gg]
                    Wg[ap, gg] = v
            gc = goal_at_site[site_of[xo]]
            if gc >= 0:
                if not infer:
                    for ap in range(NA):
                        pd[p_row, c, gc, xp, ap] = Wg[ap, gc]
                else:                        # whose destination is the cell and whose
                    for ap in range(NA):     # object robot o holds
                        v = 0.0
                        if xh >= 0:
                            for gg in range(Kg):
                                if gdst_p[gg] == gc and gobj_p[gg] == xh:
                                    v += Wg[ap, gg]
                        pd[p_row, c, gc, xp, ap] = v
            if infer:                        # a hand-over: whose object robot o holds
                for ap in range(NA):
                    v = 0.0
                    if xh >= 0:
                        for gg in range(Kg):
                            if gobj_p[gg] == xh:
                                v += Wg[ap, gg]
                    ho[p_row, s, ap] = v
            if need_ch and not cond_child:   # each child's belief about robot o's goal, as
                for k in range(Kg):          # it holds it
                    for ao in range(NA):
                        for k2 in range(Kme):
                            Wme[k, ao, k2] = mybel[k, k2]
            elif need_ch:                    # ... conditioned on robot o's action
                if busy[c, o] != 0:          # (robot o lifting: idle or not, it lifts on)
                    for a in range(NA):
                        pme[a] = 1.0 if a == PICK else 0.0
                else:
                    nm = 0
                    for a in range(4):
                        if valid_o[xo, a]:
                            nm += 1
                    for a in range(4):
                        pme[a] = idle_eps / nm if (nm > 0 and valid_o[xo, a]) else 0.0
                    pme[4] = 1.0 - idle_eps if nm > 0 else 1.0
                    for a in range(5, NA):
                        pme[a] = 0.0
                for k in range(Kg):
                    me_carry = carry_me and xh == gobj_p[k]
                    for ao in range(NA):
                        tot = 0.0
                        for k2 in range(Kme):
                            b2 = mybel[k, k2]
                            if b2 == 0.0:
                                v = 0.0
                            elif me_carry and not infer:
                                v = b2
                            elif me_carry and gobj_o[k2] != xh:
                                v = b2 * carry_o[gdst_p[k], xo, ao]
                            else:
                                q2 = myidle[k, k2]
                                v = b2 * ((1.0 - q2) * np.float64(mypsi[myrows[k2], s, ao])
                                          + q2 * pme[ao])
                            Wme[k, ao, k2] = v
                            tot += v
                        if tot > 0.0:
                            tot = 1.0 / tot
                            for k2 in range(Kme):
                                Wme[k, ao, k2] *= tot
                        else:                # an action no hypothesis of the child explains
                            for k2 in range(Kme):
                                Wme[k, ao, k2] = mybel[k, k2]
            acc[:, :] = 0.0
            for ap in range(NA):
                pap = pa[ap]
                if pap <= 0.0:
                    continue
                cpa = cost_c[c, p, ap]
                for ao in range(NA):
                    j = ao * NA + ap if o == 0 else ap * NA + ao
                    ev = pev[pair, j]
                    c2 = c if ev == 0 else tab[c, ev]
                    p2 = psucc[pair, j]
                    n0 = p2 // N
                    n1 = p2 - n0 * N
                    if o == 0:
                        no = n0
                        npp = n1
                    else:
                        no = n1
                        npp = n0
                    split = False
                    setdown = False
                    cd = c2
                    if ao >= PICK and ev != 0 and (gc >= 0 or infer):
                        cd = dtab[c, ev]
                        split = cd != c2
                        if split:            # a set-down (what robot o holds set down for
                            setdown = (xh >= 0 and gone[cd, xh] != 0  # good), not a hand-over
                                       and gone[c2, xh] == 0)
                    clo = cls_obj[c2, o, qown]
                    xo2 = npp if oap_obj[c2, o, qown] != 0 else no
                    own = rate_cls[clo, xo2, kown]
                    fo = fin_cls[clo, xo2, kown] != 0
                    ownd = own
                    fod = fo
                    if split:                # the act for the partner counted
                        clod = cls_obj[cd, o, qown]
                        xod = npp if oap_obj[cd, o, qown] != 0 else no
                        ownd = rate_cls[clod, xod, kown]
                        fod = fin_cls[clod, xod, kown] != 0
                    coa = cost_c[c, o, ao]
                    wsucc = 0.0              # ... with this mass: the hypotheses that take it
                    for gg in range(Kg):
                        qg = gobj_p[gg]
                        kg = gdst_p[gg]
                        clp = cls_obj[c2, p, qg]
                        xp2 = no if oap_obj[c2, p, qg] != 0 else npp
                        rp[gg] = rate_cls[clp, xp2, kg]
                        fp[gg] = fin_cls[clp, xp2, kg]
                        cnt[gg] = 0
                        if split:
                            if infer:
                                takes = qg == xh and ((not setdown) or kg == gc)
                            else:
                                takes = gg == gc
                            if takes:
                                cnt[gg] = 1
                                wsucc += Wg[ap, gg]
                                clpd = cls_obj[cd, p, qg]
                                xpd = no if oap_obj[cd, p, qg] != 0 else npp
                                rpd[gg] = rate_cls[clpd, xpd, kg]
                                fpd[gg] = fin_cls[clpd, xpd, kg]
                    if need_ch:              # robot o's reward under each goal the child holds
                        for k2 in range(Kme):    # possible for it
                            q2 = gobj_o[k2]
                            d2 = gdst_o[k2]
                            cl2 = cls_est[c2, o, q2]
                            x2 = npp if oap_est[c2, o, q2] != 0 else no
                            rme[k2] = rate_cls[cl2, x2, d2]
                            fme[k2] = fin_cls[cl2, x2, d2]
                            if split:
                                cl2 = cls_est[cd, o, q2]
                                x2 = npp if oap_est[cd, o, q2] != 0 else no
                                rmed[k2] = rate_cls[cl2, x2, d2]
                                fmed[k2] = fin_cls[cl2, x2, d2]
                        for gg in range(Kg):     # ch2: the child's estimate of r_o; ch3: of
                            v = 0.0              # r_o where robot o's goal is finished
                            v3 = 0.0
                            for k2 in range(Kme):
                                v += Wme[gg, ao, k2] * rme[k2]
                                if fme[k2] != 0:
                                    v3 += Wme[gg, ao, k2] * rme[k2]
                            ch2[gg] = v
                            ch3[gg] = v3
                            if cnt[gg] != 0:
                                v = 0.0
                                v3 = 0.0
                                for k2 in range(Kme):
                                    v += Wme[gg, ao, k2] * rmed[k2]
                                    if fmed[k2] != 0:
                                        v3 += Wme[gg, ao, k2] * rmed[k2]
                                chd[gg] = v
                                chd3[gg] = v3
                    for mm in range(M):
                        sp = split and msoc[mm] != 0
                        val = pap * (own - coa)
                        if sp:
                            val += wsucc * (ownd - own)
                        if mzero[mm] == 0:
                            if not depth2:
                                sj = 0.0
                                for gg in range(Kg):
                                    sj += Wg[ap, gg] * rp[gg]
                                si = pap * own
                                if sp:
                                    for gg in range(Kg):
                                        if cnt[gg] != 0:
                                            sj += Wg[ap, gg] * (rpd[gg] - rp[gg])
                                    si += wsucc * (ownd - own)
                                val += l1a[mm] * sj + l1b[mm] * si
                                if l1r[mm] != 0.0:           # the race forms of competition
                                    if fo:
                                        sr = pap * own
                                        if lead != 0.0:      # ... and its lead
                                            for gg in range(Kg):
                                                sr += lead * Wg[ap, gg] * (own - rp[gg])
                                    else:
                                        sr = 0.0
                                        for gg in range(Kg):
                                            if fp[gg] != 0:
                                                sr -= Wg[ap, gg] * rp[gg]
                                    if sp:       # the hypotheses the act counted under
                                        for gg in range(Kg):
                                            if cnt[gg] != 0:
                                                sr += Wg[ap, gg] * (
                                                    _race(fod, ownd, rpd[gg], fpd[gg] != 0, lead)
                                                    - _race(fo, own, rp[gg], fp[gg] != 0, lead))
                                    val += l1r[mm] * sr
                            else:
                                for t in range(nT):
                                    cr = cR[mm, t]   # its own race term against type t
                                    for gg in range(Kg):
                                        wh = W[ap, t, gg]
                                        if wh == 0.0:
                                            continue
                                        sm = 1 if gg == gown else 0
                                        ca = cA[mm, t, sm]
                                        cb = cB[mm, t, sm]
                                        cn = cN[mm, t]
                                        if ca == 0.0 and cb == 0.0 and cn == 0.0 and cr == 0.0:
                                            continue
                                        if sp and cnt[gg] != 0:
                                            rj = rpd[gg]
                                            fj = fpd[gg] != 0
                                            ri = ownd
                                            fi = fod
                                            cj = chd[gg]
                                            cj3 = chd3[gg]
                                        else:
                                            rj = rp[gg]
                                            fj = fp[gg] != 0
                                            ri = own
                                            fi = fo
                                            cj = ch2[gg]
                                            cj3 = ch3[gg]
                                        rk = rj * (1.0 + tb1[t]) + ta1[t] * cj - sigma * cpa
                                        if tr1[t] != 0.0:    # the child races: its own
                                            if fj:           # goal r_j, robot o's as it
                                                rk += tr1[t] * (rj + lead * (rj - cj))
                                            else:            # takes it to be
                                                rk += tr1[t] * (-cj3)
                                        xi = ca * rk + cb * ri + cn * rj
                                        if cr != 0.0:        # on the goal rewards of this
                                            xi += cr * _race(fi, ri, rj, fj, lead)   # hypothesis
                                        val += wh * xi
                        acc[mm, ao] += val
            for mm in range(M):
                for ao in range(NA):
                    R[mrows[mm], s, ao] = acc[mm, ao]


@njit(cache=True)
def _counts(pd, ho, hoon, g, s, c, gc, xp, ap, cd, c2, xh, gone):
    """The mass of the partner's action ap under which robot o's act for its partner
    counts (the table of success leads to block cd, that of failure to c2): a set-down for
    the partner -- the object robot o holds, xh, set down for good in cd and not in c2
    (gone (B, M): the objects set down for good in each block) -- with pd on candidate
    cell gc; else (hoon: robot o does not know its partner's object) a hand-over, with ho.
    Without hoon only a set-down can count.  (Until 2026-10-18 a set-down was told by the
    partner's done flag, set in cd and not in c2; without the automatic hands the partner
    can finish by its own hands in the same transition, in c2 as in cd, which told nothing.)"""
    if not hoon:
        return pd[g, c, gc, xp, ap]
    if gc >= 0 and xh >= 0 and gone[cd, xh] != 0 and gone[c2, xh] == 0:
        return pd[g, c, gc, xp, ap]
    return np.float64(ho[g, s, ap])


@njit(cache=True)
def q_row(R, P, V, m, g, s, c, pair, xo, o, NN, ps, pe, tab, dtab, pd, ho, hoon, gc, xp, gamma,
          valid, bz, uok, fixed, done, vdone, hold, gone, q):
    """q[a] = Q_m(s, a) for the ten actions (-1e300 where illegal; bz: robot o is lifting an
    object in block c, ObjectWorld.busy -- no move is open to it).  ps / pe: the pair
    successor and event tables indexed [pair, a_own * 10 + a_partner]; tab / dtab: the
    model's (B, nEv) dynamics, robot o's act for its partner failing / counting;
    gc >= 0: robot o stands on candidate cell gc and may set down for its partner there,
    which counts with mass pd[g, c, gc, xp, a_p] of the partner's action a_p (the
    hypotheses whose destination is gc -- and object the one robot o holds; xp = the
    partner's cell); hoon: robot o does not know its partner's object, and a hand-over
    counts with mass ho[g, s, a_p] (the hypotheses whose object it holds); fixed: the
    blocks where robot o's goal is done are worth vdone exactly (a model without a social
    goal parks there); uok: the hand actions it may choose here (bit a - PICK); hold (B, 2)
    and gone (B, M): what each robot holds and the objects set down for good, by block
    (:func:`_counts`)."""
    cN = np.int64(c) * NN
    xh = hold[c, o]
    for ao in range(NA):
        if ao < 5:
            if bz or not valid[xo, ao]:
                q[ao] = -1.0e300
                continue
        elif not (uok >> (ao - PICK)) & 1:
            q[ao] = -1.0e300
            continue
        acc = np.float64(R[m, s, ao])
        base = ao * NA
        for ap in range(NA):
            pr = P[g, s, ap]
            if pr == 0.0:
                continue
            ev = pe[pair, base + ap]
            n2 = ps[pair, base + ap]
            if ev == 0:                      # nothing the hands could do: same block
                acc += gamma * pr * V[m, cN + n2]
                continue
            c2 = tab[c, ev]
            if fixed and done[c2, o] != 0:
                v = vdone
            else:
                v = V[m, np.int64(c2) * NN + n2]
            if ao >= PICK and (gc >= 0 or hoon):
                cd = dtab[c, ev]
                if cd != c2:
                    wd = _counts(pd, ho, hoon, g, s, c, gc, xp, ap, cd, c2, xh, gone)
                    if wd > pr:
                        wd = pr
                    if fixed and done[cd, o] != 0:
                        vd = vdone
                    else:
                        vd = V[m, np.int64(cd) * NN + n2]
                    acc += gamma * ((pr - wd) * v + wd * vd)
                    continue
            acc += gamma * pr * v
        q[ao] = acc


@njit(cache=True)
def p_row(P, g, s, c, pair, a, o, NN, ps, pe, tab, dtab, pd, ho, hoon, gc, xp, gamma, x, fixed,
          done, vdone, hold, gone, act, rhs):
    """Fixed own action a at state s: (gamma * sum_{s' != s} P x(s'), 1 - gamma P(s -> s)).
    Successors whose value is known -- in a fixed block (value vdone) or in a block that is
    not being solved in this call (act[block] == 0: its values are final) -- belong to the
    right-hand side: they are added when rhs (a residual of the values themselves, x = V)
    and skipped otherwise (a Krylov mat-vec on search vectors)."""
    acc = 0.0
    pself = 0.0
    base = a * NA
    cN = np.int64(c) * NN
    xh = hold[c, o]
    for ap in range(NA):
        pr = P[g, s, ap]
        if pr == 0.0:
            continue
        ev = pe[pair, base + ap]
        n2 = ps[pair, base + ap]
        if ev == 0:                          # nothing the hands could do: same block
            s2 = cN + n2
            if s2 == s:
                pself += pr
            else:
                acc += pr * x[s2]
            continue
        c2 = tab[c, ev]
        w2 = pr
        if a >= PICK and (gc >= 0 or hoon):
            cd = dtab[c, ev]
            if cd != c2:                     # the act for the partner counts w.p. wd
                wd = _counts(pd, ho, hoon, g, s, c, gc, xp, ap, cd, c2, xh, gone)
                if wd > pr:
                    wd = pr
                w2 = pr - wd
                if fixed and done[cd, o] != 0:
                    if rhs:
                        acc += wd * vdone
                elif act[cd] == 0:
                    if rhs:
                        acc += wd * x[np.int64(cd) * NN + n2]
                else:
                    sd = np.int64(cd) * NN + n2
                    if sd == s:
                        pself += wd
                    else:
                        acc += wd * x[sd]
        if fixed and done[c2, o] != 0:
            if rhs:
                acc += w2 * vdone
        elif act[c2] == 0:
            if rhs:
                acc += w2 * x[np.int64(c2) * NN + n2]
        else:
            s2 = np.int64(c2) * NN + n2
            if s2 == s:
                pself += w2
            else:
                acc += w2 * x[s2]
    return gamma * acc, 1.0 - gamma * pself


@njit(parallel=True, cache=True)
def policy_graph(R, P, V, m, g, o, sc, fx, tab, dtab, it_lo, it_hi, N, psucc, psucc_T, pev,
                 pev_T, valid, busy, useok, uctx, site_of, goal_at_site, pd, ho, infer, done,
                 vdone, hold, gone, state_ok, gamma, pol, edge_n, edge_s2, edge_pr):
    """Model m (robot o, slot g, social sc, fixed fx; tables tab / dtab as in q_row) at every
    joint state of the items: its greedy action by Q from the values V (pol[s]; -1 where
    the robots overlap), and the successors under it -- edge_s2[s, :edge_n[s]] with
    probabilities edge_pr[s, :edge_n[s]] (at most two per partner action: an act for the
    partner that counts leads elsewhere than one that fails; -1 for a block worth vdone,
    fx).  The policy's transition graph, for evaluating the policy on each term of the
    reward apart (SocialMDP.value_components).  state_ok (B, NN): the joint states that are
    solved -- the robots apart, each in its room, a robot lifting an object on a cell where
    one lies (ObjectWorld.state_ok)."""
    NN = N * N
    ps = psucc if o == 0 else psucc_T
    pe = pev if o == 0 else pev_T
    vo = valid[o]
    for i in prange(it_lo.shape[0]):
        c = it_lo[i] // NN
        qp = np.empty(NA)
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            if not state_ok[c, pair]:
                pol[s] = -1
                edge_n[s] = 0
                continue
            xq = pair // N
            xr = pair - xq * N
            xo = xq if o == 0 else xr
            xp = xr if o == 0 else xq
            gc = goal_at_site[site_of[xo]] if sc != 0 else -1
            hoon = infer and sc != 0
            q_row(R, P, V, m, g, s, c, pair, xo, o, NN, ps, pe, tab, dtab, pd, ho, hoon, gc,
                  xp, gamma, vo, busy[c, o] != 0, np.int64(useok[sc, o, c, uctx[o, pair]]), fx,
                  done, vdone, hold, gone, qp)
            best = -1.0e300
            ba = 4
            for ao in range(NA):
                if qp[ao] > best:
                    best = qp[ao]
                    ba = ao
            pol[s] = ba
            n = 0
            base = ba * NA
            for ap in range(NA):
                pr = P[g, s, ap]
                if pr == 0.0:
                    continue
                ev = pe[pair, base + ap]
                n2 = ps[pair, base + ap]
                if ev == 0:
                    edge_s2[s, n] = c * NN + n2
                    edge_pr[s, n] = pr
                    n += 1
                    continue
                c2 = tab[c, ev]
                w2 = pr
                if ba >= PICK and (gc >= 0 or hoon):
                    cd = dtab[c, ev]
                    if cd != c2:
                        wd = _counts(pd, ho, hoon, g, s, c, gc, xp, ap, cd, c2, hold[c, o], gone)
                        if wd > pr:
                            wd = pr
                        w2 = pr - wd
                        if wd > 0.0:
                            edge_s2[s, n] = -1 if (fx and done[cd, o] != 0) else cd * NN + n2
                            edge_pr[s, n] = wd
                            n += 1
                if w2 > 0.0:
                    edge_s2[s, n] = -1 if (fx and done[c2, o] != 0) else c2 * NN + n2
                    edge_pr[s, n] = w2
                    n += 1
            edge_n[s] = n


@njit(parallel=True, cache=True)
def policy_iteration(V, R, P, p_of, agent_of, var_of, svar_of, soc_of, fix_of, rows, it_m, it_lo,
                     it_hi, N, psucc, psucc_T, pev, pev_T, tabs, valid, busy, useok, uctx,
                     site_of, goal_at_site, pd, ho, infer, done, vdone, hold, gone, state_ok,
                     gamma, target, eta, max_pi, max_kry, pol, W, stats, act):
    """Batched Newton policy iteration of the shared-object world, in place on V[rows] --
    the old engine's _pi_solve (greedy improvement with the Bellman residual as certificate,
    policy evaluation by BiCGSTAB on the self-loop-scaled system) on ten actions and table
    dynamics.  Model m: robot agent_of[m], tables tabs[var_of[m]] / tabs[svar_of[m]] (its
    set-down for the partner failing / succeeding, split by the masses pd[p_of[m]] when
    soc_of[m]: see q_row; and, infer, a hand-over by ho); hand action a is legal where bit
    a - PICK of useok[soc_of[m], robot, block, context] is set,
    a move where valid[robot, cell, move] (each robot's own moves: (2, N, 5));
    fix_of[m]: its own-done blocks are not among the items and worth vdone.
    act (L, B) uint8: the blocks of local model l that are solved in this call (its items);
    every other block it can reach holds final values and acts as a boundary.  The object
    configurations are block triangular -- a finished goal never comes undone -- so a model
    is solved in stages, the blocks with more finished goals first (_OEngine._solve).
    W (6, >=L, S) f64 work; pol (>=L, S) int8; stats[l] = (PI iterations, BiCGSTAB
    iterations, final residual, converged)."""
    NN = N * N
    L = rows.shape[0]
    ni = it_m.shape[0]
    f1 = np.zeros(ni)
    f2 = np.zeros(ni)
    f3 = np.zeros(ni)
    ra = np.zeros(L)
    rb = np.zeros(L)
    rc = np.zeros(L)
    rbest = np.full(L, np.inf)
    bad = np.zeros(L, dtype=np.int64)
    eta_l = np.full(L, eta)
    tol = np.zeros(L)
    todo = np.ones(L, dtype=np.bool_)
    kact = np.zeros(L, dtype=np.bool_)
    fresh = np.zeros(L, dtype=np.bool_)
    half = np.zeros(L, dtype=np.bool_)
    rho = np.ones(L)
    rho0 = np.ones(L)
    alpha = np.ones(L)
    omega = np.ones(L)
    bet = np.zeros(L)
    rh2 = np.ones(L)
    for lm in range(L):
        for q in range(4):
            stats[lm, q] = 0.0

    for k in range(max_pi + 1):
        # ---- policy improvement + Bellman residual (the certificate) -------------------------
        for i in prange(ni):
            lm = it_m[i]
            if not todo[lm]:
                continue
            m = rows[lm]
            o = agent_of[m]
            g = p_of[m]
            sc = soc_of[m]
            fx = fix_of[m] != 0
            tab = tabs[var_of[m]]
            dtab = tabs[svar_of[m]]
            ps = psucc if o == 0 else psucc_T
            pe = pev if o == 0 else pev_T
            c = it_lo[i] // NN
            vo = valid[o]                    # (N, 5) this robot's own legal moves
            worst = 0.0
            qp = np.empty(NA)
            for s in range(it_lo[i], it_hi[i]):
                pair = s - c * NN
                if not state_ok[c, pair]:
                    continue
                xq = pair // N
                xr = pair - xq * N
                xo = xq if o == 0 else xr
                gc = goal_at_site[site_of[xo]] if sc != 0 else -1
                q_row(R, P, V, m, g, s, c, pair, xo, o, NN, ps, pe, tab, dtab, pd, ho,
                      infer and sc != 0, gc, xr if o == 0 else xq, gamma, vo, busy[c, o] != 0,
                      np.int64(useok[sc, o, c, uctx[o, pair]]), fx, done, vdone, hold, gone,
                      qp)
                best = -1.0e300
                ba = 4
                for ao in range(NA):
                    if qp[ao] > best:
                        best = qp[ao]
                        ba = ao
                d = abs(best - V[m, s])
                if not d <= worst:           # also propagates NaN
                    worst = d
                pol[lm, s] = ba
            f1[i] = worst
        for lm in range(L):
            ra[lm] = 0.0
        for i in range(ni):
            lm = it_m[i]
            if todo[lm] and not f1[i] <= ra[lm]:
                ra[lm] = f1[i]
        left = 0
        for lm in range(L):
            if not todo[lm]:
                continue
            stats[lm, 2] = ra[lm]
            if ra[lm] <= target:
                todo[lm] = False
                stats[lm, 3] = 1.0
            elif k == max_pi or not np.isfinite(ra[lm]):
                todo[lm] = False
            else:
                left += 1
                stats[lm, 0] += 1.0
                # The Bellman residual of policy iteration is not monotone (the values
                # are): only a best residual that has not halved for 6 iterations tightens
                # the evaluation, a decade at a time.
                if ra[lm] <= 0.5 * rbest[lm]:
                    rbest[lm] = ra[lm]
                    bad[lm] = 0
                else:
                    bad[lm] += 1
                    if bad[lm] >= 6:
                        eta_l[lm] *= 0.1
                        bad[lm] = 0
                tol[lm] = max(0.25 * target, eta_l[lm] * ra[lm])
                kact[lm] = True
                fresh[lm] = True
        if left == 0:
            break

        # ---- policy evaluation: BiCGSTAB, restarted on breakdown -----------------------------
        for it in range(max_kry):
            nf = 0
            for lm in range(L):
                if kact[lm] and fresh[lm]:
                    nf += 1
            if nf > 0:                       # (re)start: r = r_hat = b - A x, p = v = 0
                for i in prange(ni):
                    lm = it_m[i]
                    if not (kact[lm] and fresh[lm]):
                        continue
                    m = rows[lm]
                    o = agent_of[m]
                    g = p_of[m]
                    sc = soc_of[m]
                    fx = fix_of[m] != 0
                    tab = tabs[var_of[m]]
                    dtab = tabs[svar_of[m]]
                    ps = psucc if o == 0 else psucc_T
                    pe = pev if o == 0 else pev_T
                    c = it_lo[i] // NN
                    x = V[m]
                    r0 = W[0, lm]
                    rh = W[1, lm]
                    pp = W[2, lm]
                    vv = W[3, lm]
                    sq = 0.0
                    mx = 0.0
                    for s in range(it_lo[i], it_hi[i]):
                        pair = s - c * NN
                        if not state_ok[c, pair]:
                            continue
                        a = pol[lm, s]
                        gc = -1
                        xp = 0
                        if sc != 0 and a >= PICK:
                            xq = pair // N
                            xr = pair - xq * N
                            xp = xr if o == 0 else xq
                            gc = goal_at_site[site_of[xq if o == 0 else xr]]
                        acc, dg = p_row(P, g, s, c, pair, a, o, NN, ps, pe, tab, dtab, pd, ho,
                                        infer and sc != 0, gc, xp, gamma, x, fx, done, vdone, hold,
                                        gone, act[lm], True)
                        r = (np.float64(R[m, s, a]) + acc) / dg - x[s]
                        r0[s] = r
                        rh[s] = r
                        pp[s] = 0.0
                        vv[s] = 0.0
                        sq += r * r
                        if abs(r) > mx:
                            mx = abs(r)
                    f1[i] = sq
                    f2[i] = mx
                for lm in range(L):
                    ra[lm] = 0.0
                    rb[lm] = 0.0
                for i in range(ni):
                    lm = it_m[i]
                    if kact[lm] and fresh[lm]:
                        ra[lm] += f1[i]
                        if f2[i] > rb[lm]:
                            rb[lm] = f2[i]
                for lm in range(L):
                    if kact[lm] and fresh[lm]:
                        fresh[lm] = False
                        rho[lm] = ra[lm]
                        rh2[lm] = ra[lm]
                        rho0[lm] = 1.0
                        alpha[lm] = 1.0
                        omega[lm] = 1.0
                        if rb[lm] <= tol[lm] or ra[lm] == 0.0:
                            kact[lm] = False
            na = 0
            for lm in range(L):
                if kact[lm]:
                    na += 1
                    bet[lm] = (rho[lm] / rho0[lm]) * (alpha[lm] / omega[lm])
                    stats[lm, 1] += 1.0
            if na == 0:
                break
            # p = r + beta (p - omega v)
            for i in prange(ni):
                lm = it_m[i]
                if not kact[lm]:
                    continue
                c = it_lo[i] // NN
                bl = bet[lm]
                wl = omega[lm]
                r0 = W[0, lm]
                pp = W[2, lm]
                vv = W[3, lm]
                for s in range(it_lo[i], it_hi[i]):
                    if state_ok[c, s - c * NN]:
                        pp[s] = r0[s] + bl * (pp[s] - wl * vv[s])
            # v = A p, <r_hat, v>
            for i in prange(ni):
                lm = it_m[i]
                if not kact[lm]:
                    continue
                m = rows[lm]
                o = agent_of[m]
                g = p_of[m]
                sc = soc_of[m]
                fx = fix_of[m] != 0
                tab = tabs[var_of[m]]
                dtab = tabs[svar_of[m]]
                ps = psucc if o == 0 else psucc_T
                pe = pev if o == 0 else pev_T
                c = it_lo[i] // NN
                rh = W[1, lm]
                pp = W[2, lm]
                vv = W[3, lm]
                dot = 0.0
                for s in range(it_lo[i], it_hi[i]):
                    pair = s - c * NN
                    if not state_ok[c, pair]:
                        continue
                    a = pol[lm, s]
                    gc = -1
                    xp = 0
                    if sc != 0 and a >= PICK:
                        xq = pair // N
                        xr = pair - xq * N
                        xp = xr if o == 0 else xq
                        gc = goal_at_site[site_of[xq if o == 0 else xr]]
                    acc, dg = p_row(P, g, s, c, pair, a, o, NN, ps, pe, tab, dtab, pd, ho,
                                    infer and sc != 0, gc, xp, gamma, pp, fx, done, vdone, hold,
                                    gone, act[lm], False)
                    v = pp[s] - acc / dg
                    vv[s] = v
                    dot += rh[s] * v
                f1[i] = dot
            for lm in range(L):
                ra[lm] = 0.0
            for i in range(ni):
                lm = it_m[i]
                if kact[lm]:
                    ra[lm] += f1[i]
            for lm in range(L):
                if kact[lm]:
                    if ra[lm] != 0.0 and np.isfinite(ra[lm]):
                        alpha[lm] = rho[lm] / ra[lm]
                    else:
                        alpha[lm] = 0.0       # breakdown: minimal-residual step, then restart
            # s = r - alpha v, max|s|
            for i in prange(ni):
                lm = it_m[i]
                if not kact[lm]:
                    continue
                c = it_lo[i] // NN
                al = alpha[lm]
                r0 = W[0, lm]
                vv = W[3, lm]
                ss = W[4, lm]
                mx = 0.0
                for s in range(it_lo[i], it_hi[i]):
                    if not state_ok[c, s - c * NN]:
                        continue
                    val = r0[s] - al * vv[s]
                    ss[s] = val
                    if abs(val) > mx:
                        mx = abs(val)
                f2[i] = mx
            for lm in range(L):
                rb[lm] = 0.0
            for i in range(ni):
                lm = it_m[i]
                if kact[lm] and f2[i] > rb[lm]:
                    rb[lm] = f2[i]
            for lm in range(L):
                half[lm] = kact[lm] and rb[lm] <= tol[lm]
            # t = A s, <t, s>, <t, t>
            for i in prange(ni):
                lm = it_m[i]
                if (not kact[lm]) or half[lm]:
                    continue
                m = rows[lm]
                o = agent_of[m]
                g = p_of[m]
                sc = soc_of[m]
                fx = fix_of[m] != 0
                tab = tabs[var_of[m]]
                dtab = tabs[svar_of[m]]
                ps = psucc if o == 0 else psucc_T
                pe = pev if o == 0 else pev_T
                c = it_lo[i] // NN
                ss = W[4, lm]
                tv = W[5, lm]
                ts = 0.0
                tt = 0.0
                for s in range(it_lo[i], it_hi[i]):
                    pair = s - c * NN
                    if not state_ok[c, pair]:
                        continue
                    a = pol[lm, s]
                    gc = -1
                    xp = 0
                    if sc != 0 and a >= PICK:
                        xq = pair // N
                        xr = pair - xq * N
                        xp = xr if o == 0 else xq
                        gc = goal_at_site[site_of[xq if o == 0 else xr]]
                    acc, dg = p_row(P, g, s, c, pair, a, o, NN, ps, pe, tab, dtab, pd, ho,
                                    infer and sc != 0, gc, xp, gamma, ss, fx, done, vdone, hold,
                                    gone, act[lm], False)
                    t = ss[s] - acc / dg
                    tv[s] = t
                    ts += t * ss[s]
                    tt += t * t
                f1[i] = ts
                f3[i] = tt
            for lm in range(L):
                ra[lm] = 0.0
                rc[lm] = 0.0
            for i in range(ni):
                lm = it_m[i]
                if kact[lm] and not half[lm]:
                    ra[lm] += f1[i]
                    rc[lm] += f3[i]
            for lm in range(L):
                if kact[lm] and not half[lm]:
                    omega[lm] = ra[lm] / rc[lm] if rc[lm] > 0.0 else 0.0
            # x += alpha p + omega s, r = s - omega t, <r_hat, r>, max|r|, |r|^2
            for i in prange(ni):
                lm = it_m[i]
                if not kact[lm]:
                    continue
                m = rows[lm]
                c = it_lo[i] // NN
                al = alpha[lm]
                hf = half[lm]
                wl = omega[lm]
                x = V[m]
                r0 = W[0, lm]
                rh = W[1, lm]
                pp = W[2, lm]
                ss = W[4, lm]
                tv = W[5, lm]
                d1 = 0.0
                mx = 0.0
                sq = 0.0
                for s in range(it_lo[i], it_hi[i]):
                    if not state_ok[c, s - c * NN]:
                        continue
                    if hf:
                        x[s] += al * pp[s]
                        r = ss[s]
                    else:
                        x[s] += al * pp[s] + wl * ss[s]
                        r = ss[s] - wl * tv[s]
                    r0[s] = r
                    d1 += rh[s] * r
                    sq += r * r
                    if not abs(r) <= mx:
                        mx = abs(r)
                f1[i] = d1
                f2[i] = mx
                f3[i] = sq
            for lm in range(L):
                ra[lm] = 0.0
                rb[lm] = 0.0
                rc[lm] = 0.0
            for i in range(ni):
                lm = it_m[i]
                if kact[lm]:
                    ra[lm] += f1[i]
                    rc[lm] += f3[i]
                    if not f2[i] <= rb[lm]:
                        rb[lm] = f2[i]
            for lm in range(L):
                if not kact[lm]:
                    continue
                rho0[lm] = rho[lm]
                rho[lm] = ra[lm]
                if rb[lm] <= tol[lm]:
                    kact[lm] = False
                elif not np.isfinite(rb[lm]):
                    kact[lm] = False          # diverged: the next greedy step reports it
                elif (half[lm] or alpha[lm] == 0.0 or omega[lm] == 0.0
                      or abs(rho[lm]) <= 1e-12 * np.sqrt(rh2[lm] * rc[lm])):
                    fresh[lm] = True
        for lm in range(L):
            kact[lm] = False
            fresh[lm] = False


@njit(parallel=True, cache=True)
def softmax_policies(V, R, P, p_of, agent_of, var_of, svar_of, soc_of, fix_of, rows, prow, it_m,
                     it_lo, it_hi, N, psucc, psucc_T, pev, pev_T, tabs, valid, busy, useok, uctx,
                     site_of, goal_at_site, pd, ho, infer, done, vdone, hold, gone, state_ok,
                     gamma, inv_tau, out):
    """Eq. 7 softmax policies of models rows[l] -> out[prow[l]] (S, 10) f32 on the items.
    valid (2, N, 5): each robot's own legal moves; busy (B, 2): the robot is lifting an
    object there, and only "pick" is open to it.  inv_tau (2,): 1 / tau at which the models
    of each robot are read (its partner's tau: ``SocialMDP.read_inv_tau``)."""
    NN = N * N
    for i in prange(it_m.shape[0]):
        lm = it_m[i]
        m = rows[lm]
        o = agent_of[m]
        g = p_of[m]
        sc = soc_of[m]
        fx = fix_of[m] != 0
        tab = tabs[var_of[m]]
        dtab = tabs[svar_of[m]]
        ps = psucc if o == 0 else psucc_T
        pe = pev if o == 0 else pev_T
        r_out = prow[lm]
        c = it_lo[i] // NN
        vo = valid[o]                        # (N, 5) this robot's own legal moves
        beta = inv_tau[o]                    # robot o is read at its partner's tau
        q = np.empty(NA)
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            if not state_ok[c, pair]:
                continue
            xq = pair // N
            xr = pair - xq * N
            xo = xq if o == 0 else xr
            gc = goal_at_site[site_of[xo]] if sc != 0 else -1
            q_row(R, P, V, m, g, s, c, pair, xo, o, NN, ps, pe, tab, dtab, pd, ho,
                  infer and sc != 0, gc, xr if o == 0 else xq, gamma, vo, busy[c, o] != 0,
                  np.int64(useok[sc, o, c, uctx[o, pair]]), fx, done, vdone, hold, gone, q)
            mx = -1.0e300
            for ao in range(NA):
                if q[ao] > mx:
                    mx = q[ao]
            z = 0.0
            for ao in range(NA):
                if q[ao] > -1.0e299:
                    e = np.exp((q[ao] - mx) * beta)
                    q[ao] = e
                    z += e
                else:
                    q[ao] = 0.0
            for ao in range(NA):
                out[r_out, s, ao] = q[ao] / z
