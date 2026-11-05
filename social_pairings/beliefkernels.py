"""Numba kernels of the belief planner (``Config.belief_planning``; :mod:`beliefplan`).

The planning state is ``(k, v, s)``: a belief point k (a belief the partner may hold about the
planning robot -- its physical and social posteriors and its idle mode), the joint state s,
and v, whether the partner's best action at (k, s) would undo its last action (its inertia).
The partner's choice needs no more of its history than that: it plays ``best[k, s]``, or,
where v = 1, ``alt[k, s]`` (``util.choose_with_inertia``); after it, what would undo it, u'
(:func:`undo_code`), sets v' = [u' == best[k', s']] wherever its belief has gone.  The world
acts by its true tables (``ObjectWorld.true_tab``), the partner's belief by its own update
(:func:`belief_update`, the rule of ``SocialMDP.updated_belief``), projected on the points.

Robot ``r`` plans, ``p = 1 - r`` is its partner.  A joint action is ``a0 * 10 + a1``.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .constants import GIVE, N_ACTIONS, PICK, PLACE, SWAP, TAKE, TIE_REL, WAIT
from .kernels import q_row

NA = N_ACTIONS
NO_UNDO = N_ACTIONS             # u = 10: nothing to undo (after a wait, a blocked move, ...)
NU = 2                          # v: the partner plays best (0) or, its best an undo, alt (1)

__all__ = ["NO_UNDO", "NU", "undo_code", "q_table", "choice_tables", "legal_mask",
           "tail_values", "backup", "q_point"]


@njit(cache=True)
def undo_code(a, moved, changed):
    """``util.undo_of`` as a code: the action that undoes ``a`` right after it -- the
    opposite move after a move that took the robot elsewhere, the opposite hand action after
    one that changed what it holds -- or NO_UNDO."""
    if a < WAIT:
        return (a + 2) % 4 if moved else NO_UNDO
    if a >= PICK and changed:
        if a == PICK:
            return PLACE
        if a == PLACE:
            return PICK
        if a == TAKE:
            return GIVE
        if a == GIVE:
            return TAKE
        return SWAP
    return NO_UNDO


@njit(parallel=True, cache=True)
def q_table(R, P, V, m, g, o, sc, fx, tab, dtab, it_lo, it_hi, N, psucc, psucc_T, pev, pev_T,
            valid, busy, useok, uctx, site_of, goal_at_site, pd, ho, infer, done, vdone, hold,
            gone, state_ok, gamma, out):
    """out[s] = Q_m(s, .) of model m (robot o, slot g of P / pd / ho, social sc, fixed fx;
    dynamics tab / dtab) at every joint state of the items: ``SocialMDP.q_values`` over all
    states (-inf where an action is not open, and at the states that are not solved: the
    robots overlapping).  :func:`kernels.q_row`, as ``kernels.policy_graph`` calls it, but
    for one thing ``q_values`` does and ``q_row`` does not: a fixed model (no social term)
    reads a successor in its own block (event 0) as worth ``vdone`` where its goal is done
    there."""
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
                for a in range(NA):
                    out[s, a] = -np.inf
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
            if fx and done[c, o] != 0:       # q_values: worth vdone in its own done block too
                cN = np.int64(c) * NN
                for ao in range(NA):
                    if qp[ao] <= -1.0e299:
                        continue
                    for ap in range(NA):
                        pr = P[g, s, ap]
                        if pr != 0.0 and pe[pair, ao * NA + ap] == 0:
                            qp[ao] += gamma * pr * (vdone - V[m, cN + ps[pair, ao * NA + ap]])
            for a in range(NA):
                out[s, a] = qp[a] if qp[a] > -1.0e299 else -np.inf


@njit(parallel=True, cache=True)
def choice_tables(Q, margin, best, alt):
    """The partner's choice at every state from its Q (S, 10): ``best[s]`` -- Algorithm 1's
    argmax with the deterministic tie-break of ``util.argmax_action`` -- and ``alt[s]``, what
    it chooses instead where ``best[s]`` is what would undo its last action
    (``util.choose_with_inertia``: the best other action unless undoing beats it by more
    than ``margin``; ``best[s]`` itself where nothing else is open or margin is 0).  A state
    with no action open (not solved): WAIT."""
    S = Q.shape[0]
    for s in prange(S):
        m = -np.inf
        for a in range(NA):
            if Q[s, a] > m:
                m = Q[s, a]
        if m == -np.inf:
            best[s] = WAIT
            alt[s] = WAIT
            continue
        thr = m - TIE_REL * max(1.0, abs(m))
        b = 0
        for a in range(NA):
            if Q[s, a] >= thr:
                b = a
                break
        best[s] = b
        alt[s] = b
        if margin > 0.0:
            m2 = -np.inf
            for a in range(NA):
                if a != b and Q[s, a] > m2:
                    m2 = Q[s, a]
            if m2 > -np.inf:
                thr2 = m2 - TIE_REL * max(1.0, abs(m2))
                o = b
                for a in range(NA):
                    if a != b and Q[s, a] >= thr2:
                        o = a
                        break
                if not (Q[s, b] - Q[s, o] > margin):
                    alt[s] = o


@njit(parallel=True, cache=True)
def legal_mask(o, sc, N, valid, busy, useok, uctx, state_ok, out):
    """out[s, a]: action a is open to robot o at s (``SocialMDP.legal`` over all states;
    sc: its model has a social term).  Nothing at the states that are not solved."""
    NN = N * N
    S = out.shape[0]
    for s in prange(S):
        c = s // NN
        pair = s - c * NN
        if not state_ok[c, pair]:
            for a in range(NA):
                out[s, a] = False
            continue
        x = pair // N if o == 0 else pair - (pair // N) * N
        for a in range(PICK):
            out[s, a] = busy[c, o] == 0 and valid[o, x, a]
        bits = np.int64(useok[sc, o, c, uctx[o, pair]])
        for a in range(PICK, NA):
            out[s, a] = ((bits >> (a - PICK)) & 1) != 0


@njit(cache=True)
def _step(s, a_r, a_p, r, N, true_tab, pev, psucc):
    """The true joint transition of robot r's a_r and its partner's a_p at s -> (s', c, c',
    the partner's cell before and after)."""
    NN = N * N
    c = s // NN
    pair = s - c * NN
    j = a_r * NA + a_p if r == 0 else a_p * NA + a_r
    ev = pev[pair, j]
    c2 = c if ev == 0 else np.int64(true_tab[c, ev])
    p2 = np.int64(psucc[pair, j])
    x0 = pair // N
    y0 = p2 // N
    xp = pair - x0 * N if r == 0 else x0
    yp = p2 - y0 * N if r == 0 else y0
    return c2 * NN + p2, c, c2, xp, yp


@njit(parallel=True, cache=True)
def _improve(V, pol, act_p, base, nested, legal, r, N, true_tab, pev, psucc, state_ok, gamma,
             nxt, rho, changed):
    """One greedy improvement of the policy pol against V (keeping the current action unless
    another beats it by more than round-off), and the successor / reward of the new policy."""
    NN = N * N
    S = V.shape[0]
    for s in prange(S):
        c = s // NN
        if not state_ok[c, s - c * NN]:
            nxt[s] = -1
            rho[s] = 0.0
            changed[s] = 0
            continue
        ap = act_p[s]
        cur = pol[s]
        qcur = -np.inf
        best = -np.inf
        ba = -1
        bs = -1
        for a in range(NA):
            if not legal[s, a]:
                continue
            s2, _, _, _, _ = _step(s, a, ap, r, N, true_tab, pev, psucc)
            q = np.float64(base[s, a, ap]) + nested[s2] + gamma * V[s2]
            if a == cur:
                qcur = q
            if q > best:
                best = q
                ba = a
                bs = s2
        if cur >= 0 and qcur > -np.inf and best - qcur <= 1e-12 * max(1.0, abs(best)):
            ba = cur                     # no strict improvement: keep it
            s2, _, _, _, _ = _step(s, cur, ap, r, N, true_tab, pev, psucc)
            bs = s2
        changed[s] = 1 if ba != cur else 0
        pol[s] = ba
        nxt[s] = bs
        rho[s] = np.float64(base[s, ba, ap]) + nested[bs]


@njit(cache=True)
def _evaluate(nxt, rho, gamma, V, mark, pos, stack):
    """V of a deterministic policy, exactly: the policy's successor graph is a function, so
    every walk ends in a cycle -- V on the cycle c_0 .. c_{L-1} is sum_i gamma^i rho(c_i) /
    (1 - gamma^L) at c_0, and every state before it rho + gamma V(next)."""
    S = nxt.shape[0]
    for s in range(S):
        mark[s] = 0 if nxt[s] >= 0 else 2
        if nxt[s] < 0:
            V[s] = 0.0
    for s0 in range(S):
        if mark[s0] == 2:
            continue
        n = 0
        s = s0
        while mark[s] == 0:
            mark[s] = 1
            pos[s] = n
            stack[n] = s
            n += 1
            s = nxt[s]
        stop = n
        if mark[s] == 1:                     # a cycle: stack[pos[s] .. n - 1]
            p0 = pos[s]
            acc = 0.0
            g = 1.0
            for i in range(p0, n):
                acc += g * rho[stack[i]]
                g *= gamma
            V[stack[p0]] = acc / (1.0 - g)
            mark[stack[p0]] = 2
            nv = V[stack[p0]]
            for i in range(n - 1, p0, -1):
                t = stack[i]
                V[t] = rho[t] + gamma * nv
                nv = V[t]
                mark[t] = 2
            stop = p0
        for i in range(stop - 1, -1, -1):    # the way into it (or into known values)
            t = stack[i]
            V[t] = rho[t] + gamma * V[nxt[t]]
            mark[t] = 2


@njit(cache=True)
def tail_values(V, pol, act_p, base, nested, legal, r, N, true_tab, pev, psucc, state_ok,
                gamma, max_iter):
    """The value of robot r against a partner that acts by ``act_p`` (S,) at every state, the
    rewards ``base[s, a_r, a_p] + nested[s']`` (S, 10, 10) / (S,), the true dynamics: policy
    iteration with each policy evaluated exactly (:func:`_evaluate`).  V (S,) and pol (S,)
    in place (pol < 0: no policy yet).  -> (iterations, the largest Bellman residual of the
    result: 0 up to round-off)."""
    S = V.shape[0]
    nxt = np.empty(S, dtype=np.int64)
    rho = np.empty(S)
    changed = np.zeros(S, dtype=np.uint8)
    mark = np.empty(S, dtype=np.uint8)
    pos = np.empty(S, dtype=np.int64)
    stack = np.empty(S, dtype=np.int64)
    it = 0
    for it in range(1, max_iter + 1):
        _improve(V, pol, act_p, base, nested, legal, r, N, true_tab, pev, psucc, state_ok,
                 gamma, nxt, rho, changed)
        if it > 1 and changed.sum() == 0:
            break
        _evaluate(nxt, rho, gamma, V, mark, pos, stack)
    res = _residual(V, act_p, base, nested, legal, r, N, true_tab, pev, psucc, state_ok, gamma)
    return it, res


@njit(parallel=True, cache=True)
def _residual(V, act_p, base, nested, legal, r, N, true_tab, pev, psucc, state_ok, gamma):
    NN = N * N
    S = V.shape[0]
    out = np.zeros(S)
    for s in prange(S):
        c = s // NN
        if not state_ok[c, s - c * NN]:
            continue
        ap = act_p[s]
        best = -np.inf
        for a in range(NA):
            if legal[s, a]:
                s2, _, _, _, _ = _step(s, a, ap, r, N, true_tab, pev, psucc)
                q = np.float64(base[s, a, ap]) + nested[s2] + gamma * V[s2]
                if q > best:
                    best = q
        if best > -np.inf:
            out[s] = abs(best - V[s])
    return out.max() if S else 0.0


@njit(parallel=True, cache=True)
def backup(Wn, Wc, best, alt, base, nested, k1t, k2t, lamt, frozen, legal, r, N, true_tab,
           pev, psucc, hold, state_ok, gamma, it_k, it_lo, it_hi):
    """One step of the backward induction over (k, v, s):

        Wc[k, v, s] = max_{a open} base[s, a, a_p] + nested[k, s'] + gamma W~,
        a_p = alt[k, s] if v else best[k, s]                    (the partner's choice)
        W~ = (1 - lam) Wn[k1, v1, s'] + lam Wn[k2, v2, s']        (its belief, projected)
        vi = [u' == best[ki, s']],  u' what undoes a_p           (its inertia, at ki)

    with (k1, k2, lam) = (k1t, k2t, lamt)[k, s, a, v'], v' = 1 where the partner plays
    alt[k, s] != best[k, s]; ``frozen``: k1 = k2 = k (the belief held).  Items (it_k,
    it_lo, it_hi): a point and a run of states of one block."""
    NN = N * N
    p = 1 - r
    for i in prange(it_k.shape[0]):
        k = it_k[i]
        c = it_lo[i] // NN
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            if not state_ok[c, pair]:
                Wc[k, 0, s] = 0.0
                Wc[k, 1, s] = 0.0
                continue
            b = best[k, s]
            al = alt[k, s]
            for v in range(2):
                if v == 1 and al == b:
                    Wc[k, 1, s] = Wc[k, 0, s]
                    continue
                ap = al if v == 1 else b
                vmax = -np.inf
                for a in range(NA):
                    if not legal[s, a]:
                        continue
                    s2, _, c2, xp, yp = _step(s, a, ap, r, N, true_tab, pev, psucc)
                    u2 = undo_code(ap, yp != xp, hold[c, p] != hold[c2, p])
                    if frozen:
                        cont = Wn[k, 1 if u2 == best[k, s2] else 0, s2]
                    else:
                        k1 = k1t[k, s, a, v]
                        lm = np.float64(lamt[k, s, a, v])
                        cont = Wn[k1, 1 if u2 == best[k1, s2] else 0, s2]
                        if lm > 0.0:
                            k2 = k2t[k, s, a, v]
                            cont = ((1.0 - lm) * cont
                                    + lm * Wn[k2, 1 if u2 == best[k2, s2] else 0, s2])
                    q = np.float64(base[s, a, ap]) + nested[k, s2] + gamma * cont
                    if q > vmax:
                        vmax = q
                Wc[k, v, s] = vmax


@njit(cache=True)
def q_point(s, a_p, Wn, best, base, nested_row, k1, k2, lam, legal, r, N, true_tab, pev,
            psucc, hold, gamma, out):
    """out[a] = Q of robot r's action a at joint state s with the partner playing a_p:
    base + nested + gamma ((1 - lam[a]) Wn[k1[a], v1, s'] + lam[a] Wn[k2[a], v2, s']) -- per
    action the projection of the belief it leads to (k1, k2, lam: (10,)), vi whether the
    partner's best there undoes a_p.  -inf where a is not open.  The root's Q, and the
    planned way's (beliefplan)."""
    p = 1 - r
    for a in range(NA):
        if not legal[s, a]:
            out[a] = -np.inf
            continue
        s2, c, c2, xp, yp = _step(s, a, a_p, r, N, true_tab, pev, psucc)
        u2 = undo_code(a_p, yp != xp, hold[c, p] != hold[c2, p])
        cont = Wn[k1[a], 1 if u2 == best[k1[a], s2] else 0, s2]
        if lam[a] > 0.0:
            cont = ((1.0 - lam[a]) * cont
                    + lam[a] * Wn[k2[a], 1 if u2 == best[k2[a], s2] else 0, s2])
        out[a] = np.float64(base[s, a, a_p]) + nested_row[s2] + gamma * cont


# ---- the partner's belief update (SocialMDP.updated_belief), one transition ----------------------
# The physical belief about robot r: lp (PG,) log, over its physical goals g = (pg_obj[g],
# pg_dst[g]); the social one: ls (T,) log; the idle mode q.  What follows mirrors
# models.SocialMDP.updated_belief and the helpers it calls, line for line, for the options
# the belief planner supports (Config.physical_likelihood "level1", no window): its tests
# compare the two over every state and action of a scene.

@njit(cache=True)
def _lae(a, b):
    """numpy.logaddexp."""
    if a == b:
        return a + 0.6931471805599453
    if a > b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


@njit(cache=True)
def _lse(x, n):
    """models._log_sum of x[:n]."""
    top = -np.inf
    for i in range(n):
        if x[i] > top:
            top = x[i]
    if not (top > -np.inf) or top == np.inf:
        return top
    acc = 0.0
    for i in range(n):
        acc += np.exp(x[i] - top)
    return top + np.log(acc)


@njit(cache=True)
def _normalize(x, n):
    """util.normalize_log of x[:n], in place."""
    z = _lse(x, n)
    for i in range(n):
        x[i] -= z


@njit(cache=True)
def _any_finite(x, n):
    for i in range(n):
        if -np.inf < x[i] < np.inf:
            return True
    return False


@njit(cache=True)
def _mix_uniform(lb, n, h, out):
    """models.mix_uniform of lb[:n] into out[:n] (lb and out may be one array)."""
    ns = 0
    for i in range(n):
        if -np.inf < lb[i] < np.inf:
            ns += 1
    if ns == 0:
        for i in range(n):
            out[i] = lb[i]
        return
    m = _lse(lb, n)
    l1 = np.log1p(-h)
    lu = np.log(h / ns)
    for i in range(n):
        if -np.inf < lb[i] < np.inf:
            out[i] = _lae(l1 + (lb[i] - m), lu)
        else:
            out[i] = -np.inf


@njit(cache=True)
def _group_lse(x, group, q, n):
    """log sum exp of x[g] over the g < n with group[g] == q; -inf (and no member: -inf)."""
    top = -np.inf
    for g in range(n):
        if group[g] == q and x[g] > top:
            top = x[g]
    if not (top > -np.inf):
        return -np.inf
    acc = 0.0
    for g in range(n):
        if group[g] == q:
            acc += np.exp(x[g] - top)
    return top + np.log(acc)


@njit(cache=True)
def _factors(lp, obj, dst, PG, M, K, lo, ld):
    """models.SocialMDP.factors: the marginals over objects (M,) and destinations (K,)."""
    for q in range(M):
        lo[q] = _group_lse(lp, obj, q, PG)
    for d in range(K):
        ld[d] = _group_lse(lp, dst, d, PG)
    _normalize(lo, M)
    _normalize(ld, K)


@njit(cache=True)
def _product(lo, ld, obj, dst, PG, out):
    """models.SocialMDP.product."""
    for g in range(PG):
        out[g] = lo[obj[g]] + ld[dst[g]]
    _normalize(out, PG)


@njit(cache=True)
def _hazard_phys(lp, h, objects, factored, stages, obj, dst, PG, M, K, out, wf):
    """models.SocialMDP.hazard_phys of lp into out (wf: (6, max(M, K, PG)) work)."""
    if h <= 0.0:
        for g in range(PG):
            out[g] = lp[g]
        return
    x = wf[5]
    for g in range(PG):
        x[g] = lp[g]
    _normalize(x, PG)
    if factored:
        lo, ld = wf[0], wf[1]
        _factors(x, obj, dst, PG, M, K, lo, ld)
        both = not stages
        if objects or both:
            _mix_uniform(lo, M, h, lo)
        if (not objects) or both:
            _mix_uniform(ld, K, h, ld)
        _product(lo, ld, obj, dst, PG, out)
        return
    if not stages:
        _mix_uniform(x, PG, h, out)
        return
    for g in range(PG):
        out[g] = x[g]
    mass, new_mass, sub = wf[2], wf[3], wf[4]
    for q in range(M):
        mass[q] = _group_lse(x, obj, q, PG)
    if objects:                          # the object's mass; the destination given it kept
        _mix_uniform(mass, M, h, new_mass)
        for q in range(M):
            if -np.inf < mass[q] < np.inf:
                for g in range(PG):
                    if obj[g] == q:
                        out[g] = x[g] - mass[q] + new_mass[q]
    else:                                # the destination given each object; its mass kept
        for q in range(M):
            if not (-np.inf < mass[q] < np.inf):
                continue
            n = 0
            for g in range(PG):
                if obj[g] == q:
                    sub[n] = x[g] - mass[q]
                    n += 1
            _mix_uniform(sub, n, h, sub)
            n = 0
            for g in range(PG):
                if obj[g] == q:
                    out[g] = sub[n] + mass[q]
                    n += 1


@njit(cache=True)
def belief_update(lp0, ls0, q0, s, a_r, a_p, r, N, LL, log_idle_r, busy, hold, done, pg_obj,
                  pg_dst, M, K, dest, true_dest_r, iscand, site_of, failed_p, vain_p, true_tab,
                  pev, psucc, has_social, infer_social, factored, stages, alpha, hand_h, goal_h,
                  social_h, idle_on, idle_h, cond_place, lp_out, ls_out, wp, wl, wt, wf, keep):
    """The partner's update of its belief about robot r after robot r's a_r and its own a_p
    at joint state s (the true dynamics) -- ``SocialMDP.updated_belief(r, ...)``: lp0 / ls0 /
    q0 the belief before; writes lp_out, ls_out and returns the idle mode after.  LL (S, 10,
    PG, T): log psi of robot r's level-1 hypothesis models (at the likelihood's tau, after
    the lapse); log_idle_r (N, 10): its idle mode's.  failed_p / vain_p (B, nEv): the
    partner's set-down for robot r that failed / its hand-over in vain (ObjectWorld.
    real_step's, by event).  Work: wp (8, PG), wl (PG, T), wt (3, T), wf (6, max(M, K,
    PG)), keep (PG,)."""
    PG = lp0.shape[0]
    T = ls0.shape[0]
    NN = N * N
    p = 1 - r
    for g in range(PG):
        lp_out[g] = lp0[g]
    for t in range(T):
        ls_out[t] = ls0[t]
    c = s // NN
    if busy[c, r] != 0:                  # lifting: nothing is learnt
        return q0
    pair = s - c * NN
    j = a_r * NA + a_p if r == 0 else a_p * NA + a_r
    ev = pev[pair, j]
    c2 = c if ev == 0 else np.int64(true_tab[c, ev])
    p2 = np.int64(psucc[pair, j])
    y0 = p2 // N
    y1 = p2 - y0 * N
    x0 = pair // N
    xr = x0 if r == 0 else pair - x0 * N
    yr = y0 if r == 0 else y1
    yp = y1 if r == 0 else y0
    a = a_r
    l_idle = log_idle_r[xr, a]
    empty = hold[c, r] < 0
    prior, raw, lik, mix, post, hard, kept = wp[0], wp[1], wp[2], wp[3], wp[4], wp[5], wp[6]
    # the prior: a hand action's factor mixed toward the uniform first (Config.hand_hazard)
    if hand_h > 0.0 and a >= PICK:
        _hazard_phys(lp0, hand_h, empty, factored, stages, pg_obj, pg_dst, PG, M, K, prior, wf)
    else:
        for g in range(PG):
            prior[g] = lp0[g]
    # ---- the physical likelihood (Eq. 6 as printed: level-1 models over the social posterior)
    for g in range(PG):
        fin = -np.inf < prior[g] < np.inf
        for t in range(T):
            wl[g, t] = LL[s, a, g, t] if fin else -np.inf
    if has_social:
        tmp = wt[2]
        for g in range(PG):
            for t in range(T):
                tmp[t] = ls0[t] + wl[g, t]
            raw[g] = _lse(tmp, T)
    else:                                # (a partner at level 1 reads the level-0 policies:
        for g in range(PG):              # not supported -- beliefplan asks for level 2)
            raw[g] = -np.inf
    for g in range(PG):
        lik[g] = raw[g]
    if not factored and stages and empty:    # staged_log_likelihood: the destination averaged
        for q in range(M):
            pq = _group_lse(prior, pg_obj, q, PG)
            if not (-np.inf < pq < np.inf):
                continue
            tmp2 = wp[7]
            for g in range(PG):
                tmp2[g] = prior[g] + raw[g] if pg_obj[g] == q else -np.inf
            v = _lse(tmp2, PG) - pq
            for g in range(PG):
                if pg_obj[g] == q:
                    lik[g] = v
    la = np.log(1.0 - q0) if 1.0 - q0 > 0.0 else -np.inf
    lb = np.log(q0) + l_idle if q0 > 0.0 else -np.inf
    for g in range(PG):
        mix[g] = _lae(la + lik[g] if la > -np.inf else -np.inf, lb)
    # ---- Eq. 5's transition term: what the hands did that depends on robot r's goal
    for g in range(PG):
        keep[g] = True
    if cond_place:
        held_p = hold[c, p]
        held_r = hold[c, r]
        if done[c2, r] != 0 and done[c, r] == 0:          # robot r finished
            for g in range(PG):
                keep[g] = pg_obj[g] == r and dest[pg_dst[g]] == true_dest_r
        else:
            if (a == PLACE and held_r == r and hold[c2, r] == r
                    and iscand[site_of[yr]]):                  # its own place that did nothing
                for g in range(PG):
                    if pg_obj[g] == r and dest[pg_dst[g]] == yr:
                        keep[g] = False
            if failed_p[c, ev] != 0 and held_p >= 0:          # the partner's set-down for it
                for g in range(PG):
                    if pg_obj[g] == held_p and dest[pg_dst[g]] == yp:
                        keep[g] = False
            if a_p == GIVE and held_p >= 0:                   # the partner's hand-over
                if hold[c2, r] == held_p:
                    for g in range(PG):
                        if pg_obj[g] != held_p:
                            keep[g] = False
                elif vain_p[c, ev] != 0:
                    for g in range(PG):
                        if pg_obj[g] == held_p:
                            keep[g] = False
    explained = False
    for g in range(PG):
        hard[g] = prior[g] if keep[g] else -np.inf
        if keep[g]:
            post[g] = prior[g] + (mix[g] if (factored or alpha == 1.0) else alpha * mix[g])
        else:
            post[g] = -np.inf
        if -np.inf < post[g] < np.inf:
            explained = True
    if not explained:                    # nothing explains the action: the prior, less what
        for g in range(PG):              # the hands ruled out
            post[g] = hard[g]
    for g in range(PG):
        lp_out[g] = prior[g]
    if _any_finite(post, PG):
        if factored:                     # factored_posterior
            lo_p, ld_p, lo_h, ld_h = wf[0], wf[1], wf[2], wf[3]
            _factors(post, pg_obj, pg_dst, PG, M, K, lo_p, ld_p)
            _factors(hard, pg_obj, pg_dst, PG, M, K, lo_h, ld_h)
            if alpha != 1.0:
                for q in range(M):
                    lo_p[q] = ((1.0 - alpha) * lo_h[q] + alpha * lo_p[q]
                               if -np.inf < lo_p[q] < np.inf else -np.inf)
                _normalize(lo_p, M)
                for d in range(K):
                    ld_p[d] = ((1.0 - alpha) * ld_h[d] + alpha * ld_p[d]
                               if -np.inf < ld_p[d] < np.inf else -np.inf)
                _normalize(ld_p, K)
            lo = lo_p if (empty or not stages) else lo_h
            ld = ld_p if ((not empty) or not stages) else ld_h
            _product(lo, ld, pg_obj, pg_dst, PG, lp_out)
        elif (not stages) or empty:      # _renormalize_stage
            for g in range(PG):
                lp_out[g] = post[g]
            _normalize(lp_out, PG)
        else:
            for g in range(PG):
                lp_out[g] = -np.inf
            for q in range(M):
                total = _group_lse(post, pg_obj, q, PG)
                had = _group_lse(prior, pg_obj, q, PG)
                if -np.inf < total < np.inf and -np.inf < had < np.inf:
                    for g in range(PG):
                        if pg_obj[g] == q:
                            lp_out[g] = post[g] - total + had
            _normalize(lp_out, PG)
    if _any_finite(hard, PG):            # the physical prior over the goals kept
        for g in range(PG):
            kept[g] = hard[g]
        _normalize(kept, PG)
    else:
        for g in range(PG):
            kept[g] = prior[g]
    # ---- the idle mode
    q1 = q0
    if idle_on:
        if explained:
            active = 0.0
            for g in range(PG):
                if -np.inf < lik[g] < np.inf:
                    active += np.exp(kept[g]) * np.exp(lik[g])
            idle = np.exp(l_idle) if -np.inf < l_idle < np.inf else 0.0
            total = (1.0 - q0) * active + q0 * idle
            q1 = q0 * idle / total if total > 0.0 else q0
        q1 = q1 * (1.0 - idle_h) + (1.0 - q1) * idle_h
    # ---- the social posterior (Eq. 5)
    if has_social and infer_social:
        lsoc, ps, tmp = wt[0], wt[1], wt[2]
        tmp2 = wp[7]
        for t in range(T):
            for g in range(PG):
                tmp2[g] = kept[g] + wl[g, t]
            lsoc[t] = _lse(tmp2, PG)
            ps[t] = ls0[t] + alpha * _lae(la + lsoc[t] if la > -np.inf else -np.inf, lb)
        if _any_finite(ps, T):
            for t in range(T):
                ls_out[t] = ps[t]
            _normalize(ls_out, T)
    # ---- the partner's goal may change: the prediction step (Config.goal_hazard, ...)
    if goal_h > 0.0:
        holds = hold[c2, r] >= 0 or busy[c2, r] != 0
        tmp3 = wp[7]
        for g in range(PG):
            tmp3[g] = lp_out[g]
        _hazard_phys(tmp3, goal_h, not holds, factored, stages, pg_obj, pg_dst, PG, M, K,
                     lp_out, wf)
    if social_h > 0.0 and has_social and infer_social:
        _mix_uniform(ls_out, T, social_h, ls_out)
    return q1


# ---- belief points: their coordinates, and where a belief lies among them ------------------------

@njit(cache=True)
def embed(lp, ls, q, has_social, z):
    """The coordinates of a belief: sqrt of the physical posterior, of the social one and of
    the idle mode's two values -- Euclidean distance between them is sqrt 2 x the Hellinger
    distance of each part, summed in quadrature."""
    PG = lp.shape[0]
    n = 0
    for g in range(PG):
        z[n] = np.sqrt(np.exp(lp[g])) if lp[g] > -np.inf else 0.0
        n += 1
    if has_social:
        for t in range(ls.shape[0]):
            z[n] = np.sqrt(np.exp(ls[t])) if ls[t] > -np.inf else 0.0
            n += 1
    qq = min(max(q, 0.0), 1.0)
    z[n] = np.sqrt(qq)
    z[n + 1] = np.sqrt(1.0 - qq)


@njit(cache=True)
def project(z, Z, n_near, segment, dist):
    """Where the belief with coordinates z lies among the points Z (nk, D): the nearest point
    k1 and -- ``segment`` -- the point k2 among the n_near next nearest whose segment from k1
    passes closest to z, with the convex weight lam of k2 (0: k1 alone).  -> (k1, k2, lam,
    the distance from z to k1 + lam (k2 - k1))."""
    nk, D = Z.shape
    k1 = 0
    for k in range(nk):
        d = 0.0
        for i in range(D):
            e = z[i] - Z[k, i]
            d += e * e
        dist[k] = np.sqrt(d)
        if dist[k] < dist[k1]:
            k1 = k
    best_res = dist[k1]
    k2 = k1
    lam = 0.0
    if not segment or nk == 1:
        return k1, k2, lam, best_res
    for _ in range(min(n_near, nk - 1)):   # the next nearest, one at a time
        kn = -1
        for k in range(nk):
            if k == k1 or dist[k] < 0.0:
                continue
            if kn < 0 or dist[k] < dist[kn]:
                kn = k
        if kn < 0:
            break
        dn = dist[kn]
        dist[kn] = -1.0 - dn             # taken
        L = 0.0
        dot = 0.0
        for i in range(D):
            e = Z[kn, i] - Z[k1, i]
            L += e * e
            dot += (z[i] - Z[k1, i]) * e
        if L <= 0.0:
            continue
        lm = min(max(dot / L, 0.0), 1.0)
        if lm <= 0.0:
            continue
        res = 0.0
        for i in range(D):
            e = z[i] - Z[k1, i] - lm * (Z[kn, i] - Z[k1, i])
            res += e * e
        res = np.sqrt(res)
        if res < best_res - 1e-15:
            best_res = res
            k2 = kn
            lam = lm
    return k1, k2, lam, best_res


@njit(parallel=True, cache=True)
def projections(P_lp, P_ls, P_q, Z, best, alt, legal, r, N, LL, log_idle_r, busy, hold, done,
                pg_obj, pg_dst, M, K, dest, true_dest_r, iscand, site_of, failed_p, vain_p,
                true_tab, pev, psucc, has_social, infer_social, factored, stages, alpha, hand_h,
                goal_h, social_h, idle_on, idle_h, cond_place, n_near, segment, state_ok,
                it_k, it_lo, it_hi, k1t, k2t, lamt, rest):
    """For every point k, state s of the items, open action a of robot r and choice v of
    the partner (0: best[k, s], 1: alt[k, s]): the partner's belief after the transition
    (:func:`belief_update` from point k) and where it lies among the points
    (:func:`project`) -> k1t, k2t, lamt, rest[k, s, a, v].  Robot r lifting: its belief
    stays point k."""
    NN = N * N
    PG = P_lp.shape[1]
    T = P_ls.shape[1]
    nk, D = Z.shape
    W = max(M, K, PG)
    for i in prange(it_k.shape[0]):
        k = it_k[i]
        c = it_lo[i] // NN
        lp_out = np.empty(PG)
        ls_out = np.empty(T)
        wp = np.empty((8, PG))
        wl = np.empty((PG, T))
        wt = np.empty((3, T))
        wf = np.empty((6, W))
        keep = np.empty(PG, dtype=np.bool_)
        z = np.empty(D)
        dist = np.empty(nk)
        for s in range(it_lo[i], it_hi[i]):
            if not state_ok[c, s - c * NN]:
                continue
            for a in range(NA):
                if not legal[s, a]:
                    continue
                for v in range(2):
                    if busy[c, r] != 0:
                        k1t[k, s, a, v] = k
                        k2t[k, s, a, v] = k
                        lamt[k, s, a, v] = 0.0
                        rest[k, s, a, v] = 0.0
                        continue
                    if v == 1 and alt[k, s] == best[k, s]:
                        k1t[k, s, a, 1] = k1t[k, s, a, 0]
                        k2t[k, s, a, 1] = k2t[k, s, a, 0]
                        lamt[k, s, a, 1] = lamt[k, s, a, 0]
                        rest[k, s, a, 1] = rest[k, s, a, 0]
                        continue
                    ap = best[k, s] if v == 0 else alt[k, s]
                    q1 = belief_update(P_lp[k], P_ls[k], P_q[k], s, a, ap, r, N, LL,
                                       log_idle_r, busy, hold, done, pg_obj, pg_dst, M, K,
                                       dest, true_dest_r, iscand, site_of, failed_p, vain_p,
                                       true_tab, pev, psucc, has_social, infer_social,
                                       factored, stages, alpha, hand_h, goal_h, social_h,
                                       idle_on, idle_h, cond_place, lp_out, ls_out, wp, wl,
                                       wt, wf, keep)
                    embed(lp_out, ls_out, q1, has_social, z)
                    kk1, kk2, lm, res = project(z, Z, n_near, segment, dist)
                    k1t[k, s, a, v] = kk1
                    k2t[k, s, a, v] = kk2
                    lamt[k, s, a, v] = lm
                    rest[k, s, a, v] = res


# ---- what robot r is paid, the partner known (the oracle) -----------------------------------------

@njit(parallel=True, cache=True)
def joint_rewards(r, N, legal, cls_of, at_partner, cls_obj, at_partner_obj, rate_cls, fin_cls,
                  cost_c, done, true_tab, pev, psucc, state_ok, goal_r, obj_k, dst_k, al, be,
                  nu, race_i, b1, c1, lead, sigma, complete, bonus, it_lo, it_hi, out):
    """out[s, a_r, a_p]: robot r's reward for its a_r and its partner's a_p at s, the partner
    of known social goal and physical goal (obj_k, dst_k), less the part that reads the
    partner's belief about robot r (:func:`nested_row`) -- ``SocialMDP.modeled_reward`` with
    that one hypothesis: r(g_r) - c(a_r) + al (r_p (1 + b1) - sigma c(a_p) + c1 [the
    partner's race on its own finish]) + be r(g_r) + nu r_p + robot r's race term.
    ``complete``: bonus 1[robot r's goal done by this transition] - c(a_r) instead."""
    NN = N * N
    p = 1 - r
    for i in prange(it_lo.shape[0]):
        c = it_lo[i] // NN
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            if not state_ok[c, pair]:
                continue
            for a in range(NA):
                if not legal[s, a]:
                    continue
                for ap in range(NA):
                    s2, _, c2, _, _ = _step(s, a, ap, r, N, true_tab, pev, psucc)
                    cost = cost_c[c, r, a]
                    if complete:
                        fin_now = done[c2, r] != 0 and done[c, r] == 0
                        out[s, a, ap] = (bonus if fin_now else 0.0) - cost
                        continue
                    p2 = s2 - c2 * NN
                    y0 = p2 // N
                    y1 = p2 - y0 * N
                    yr = y0 if r == 0 else y1
                    yp = y1 if r == 0 else y0
                    at_r = yp if at_partner[c2, r] != 0 else yr
                    cl = cls_of[c2, r]
                    own = rate_cls[cl, at_r, goal_r]
                    fi = fin_cls[cl, at_r, goal_r] != 0
                    clp = cls_obj[c2, p, obj_k]
                    at_p = yr if at_partner_obj[c2, p, obj_k] != 0 else yp
                    rj = rate_cls[clp, at_p, dst_k]
                    fj = fin_cls[clp, at_p, dst_k] != 0
                    rk = rj * (1.0 + b1) - sigma * cost_c[c, p, ap]
                    if c1 != 0.0 and fj:
                        rk += c1 * rj * (1.0 + lead)
                    term = al * rk + be * own + nu * rj
                    if race_i != 0.0:
                        if fi:
                            term += race_i * (own + lead * (own - rj))
                        elif fj:
                            term -= race_i * rj
                    out[s, a, ap] = own - cost + term


@njit(parallel=True, cache=True)
def nested_row(r, N, mine, pg_obj, pg_dst, cls_est, at_partner_est, cls_obj, at_partner_obj,
               rate_cls, fin_cls, state_ok, obj_k, dst_k, al, a1, c1, lead, it_lo, it_hi, out):
    """out[s'] = the part of robot r's reward at s' that reads its partner's belief about it,
    ``mine`` (PG,): al (a1 ch - c1 lead fin_p ch - c1 (1 - fin_p) ch_fin), ch = sum_g mine(g)
    r^(g_r = g)(s') (Table I's R_j estimates robot r's goal reward by it), ch_fin the same
    over the goals finished, fin_p the partner's goal (obj_k, dst_k) finished at s'."""
    NN = N * N
    p = 1 - r
    PG = mine.shape[0]
    for i in prange(it_lo.shape[0]):
        c = it_lo[i] // NN
        for s in range(it_lo[i], it_hi[i]):
            pair = s - c * NN
            if not state_ok[c, pair]:
                out[s] = 0.0
                continue
            y0 = pair // N
            y1 = pair - y0 * N
            yr = y0 if r == 0 else y1
            yp = y1 if r == 0 else y0
            ch = 0.0
            chf = 0.0
            for g in range(PG):
                m = mine[g]
                if m == 0.0:
                    continue
                q = pg_obj[g]
                cl = cls_est[c, r, q]
                at = yp if at_partner_est[c, r, q] != 0 else yr
                v = rate_cls[cl, at, pg_dst[g]]
                ch += m * v
                if fin_cls[cl, at, pg_dst[g]] != 0:
                    chf += m * v
            clp = cls_obj[c, p, obj_k]
            at_p = yr if at_partner_obj[c, p, obj_k] != 0 else yp
            fp = 1.0 if fin_cls[clp, at_p, dst_k] != 0 else 0.0
            out[s] = al * (a1 * ch - c1 * lead * fp * ch - c1 * (1.0 - fp) * chf)
