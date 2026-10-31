"""Exact values of one robot acting alone in the shared-object world.

Used as warm starts of the joint solves, as the screen that a task is worth finishing at
all, and as the yardstick of a robot's physical progress (the TD residual of its solo
value: how much closer to done one transition brought it).
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .constants import N_ACTIONS, N_HOW, PICK

__all__ = ["solo_values", "solo_scan"]


@njit(cache=True)
def solo_values(r, N, nb, valid, busy_r, site_of, tab, E1, cls_r, rate_g, cost_r, useok, gamma,
                vdone, dn_r, V, steps_max):
    """Exact solo value of robot r for one goal over (block, cell), its partner absent:
    V[b, x] in place (Jacobi value iteration from the park-forever lower bound, which the
    optimum -- a finite path, then parking -- reaches in as many sweeps as its length).
    rate_g (ncls, N): reward rate of this goal by class and cell; useok (B, 2 (nsite + 1))
    the hand actions it may choose (``ObjectWorld.hand_ok``: bit a - PICK); busy_r (B,): it
    is lifting an object there, no move open to it; -> sweeps used."""
    B = V.shape[0]
    new = np.empty_like(V)
    for b in range(B):
        for x in range(N):
            V[b, x] = vdone if dn_r[b] else (rate_g[cls_r[b], x] - cost_r[b, 4]) / (1.0 - gamma)
    for it in range(steps_max):
        worst = 0.0
        for b in range(B):
            if dn_r[b]:
                for x in range(N):
                    new[b, x] = V[b, x]
                continue
            for x in range(N):
                best = V[b, x]
                for a in range(N_ACTIONS):
                    if a < 5:
                        if busy_r[b] or not valid[x, a]:
                            continue
                        x2 = nb[x, a]
                    else:
                        if not (useok[b, site_of[x] * 2] >> (a - PICK)) & 1:
                            continue
                        x2 = x
                    how = 0 if x2 != x else (1 if a == 4 else (3 + a - PICK if a >= PICK else 2))
                    e = site_of[x2] * N_HOW + how
                    ev = ((e * E1 + 1) * 2) if r == 0 else ((1 * E1 + e) * 2)   # partner: stayed
                    b2 = tab[b, ev]
                    q = rate_g[cls_r[b2], x2] - cost_r[b, a] + gamma * (
                        vdone if dn_r[b2] else V[b2, x2])
                    if q > best:
                        best = q
                d = best - V[b, x]
                if d > worst:
                    worst = d
                new[b, x] = best
        for b in range(B):
            for x in range(N):
                V[b, x] = new[b, x]
        if worst <= 1e-12:
            return it + 1
    return steps_max


@njit(cache=True)
def solo_scan(r, N, nb, valid, busy_r, site_of, tab, E1, cls_r, rate_g, cost_r, useok, gamma,
              vdone, dn_r, V, b0, x0, limit, out):
    """Greedy solo run from (b0, x0): -> transitions until the goal is done (-1: never
    within `limit`); out (2,) = (min, max) TD residual of V over every own action
    (including a move the partner blocks) at every (block, cell)."""
    lo = np.inf
    hi = -np.inf
    B = V.shape[0]
    for b in range(B):
        if dn_r[b]:
            continue
        for x in range(N):
            for a in range(N_ACTIONS):
                if a < 5:
                    if busy_r[b] or not valid[x, a]:
                        continue
                elif not (useok[b, site_of[x] * 2] >> (a - PICK)) & 1:
                    continue
                for blocked in range(2):
                    if blocked == 1 and a >= 4:
                        continue
                    x2 = x if (blocked == 1 or a >= 4) else nb[x, a]
                    how = 0 if x2 != x else (1 if a == 4 else (3 + a - PICK if a >= PICK else 2))
                    e = site_of[x2] * N_HOW + how
                    ev = ((e * E1 + 1) * 2) if r == 0 else ((1 * E1 + e) * 2)
                    b2 = tab[b, ev]
                    td = rate_g[cls_r[b2], x2] - cost_r[b, a] + gamma * (
                        vdone if dn_r[b2] else V[b2, x2]) - V[b, x]
                    if td < lo:
                        lo = td
                    if td > hi:
                        hi = td
    out[0] = lo
    out[1] = hi
    b, x = b0, x0
    for t in range(limit):
        if dn_r[b]:
            return t
        best = -np.inf
        ba, bb, bx = 4, b, x
        for a in range(N_ACTIONS):
            if a < 5:
                if busy_r[b] or not valid[x, a]:
                    continue
                x2 = nb[x, a]
            else:
                if not (useok[b, site_of[x] * 2] >> (a - PICK)) & 1:
                    continue
                x2 = x
            how = 0 if x2 != x else (1 if a == 4 else (3 + a - PICK if a >= PICK else 2))
            e = site_of[x2] * N_HOW + how
            ev = ((e * E1 + 1) * 2) if r == 0 else ((1 * E1 + e) * 2)
            b2 = tab[b, ev]
            q = rate_g[cls_r[b2], x2] - cost_r[b, a] + gamma * (vdone if dn_r[b2] else V[b2, x2])
            if q > best + 1e-12:
                best = q
                ba, bb, bx = a, b2, x2
        if ba == 4:
            return -1
        b, x = bb, bx
    return -1 if not dn_r[b] else limit
