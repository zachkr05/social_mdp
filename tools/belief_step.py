"""Print one step of a record's belief update: what every hypothesis model made of the
partner's action, and how the beliefs moved (2026-11-04; the user: "All information for
belief calculation should be in the JSOn", and, on step 38 / 48 of the hall_crossing
testbed: "record Q_{d,xi}(s_t, a_t), max_a' Q_{d,xi}(s_t, a'), P(a_t | s_t, d), b_t(d),
b_{t+1}(d). Compare these values for G1, G4, and G5, separately for each social-goal
hypothesis").

    python3 tools/belief_step.py RECORD.json --step 38
    python3 tools/belief_step.py RECORD.json --step 48 --robot agent_1 --dest G1 G4 G5
    python3 tools/belief_step.py RECORD.json --step 48 --all-social     # every social goal

--step is the video's step ("step 38 of 55"): the trace's row 38, the action taken there and
the update it caused (the video's next frame shows the belief after it).  The record must
have been made with Config.belief_detail (the default since 2026-11-04): each trace row's
"belief_update", robot i's update about its partner as entry i.  Units: Q, tau and the
costs in reward units; "gap" is max_a' Q(s, a') - Q(s, a_t), also in moves' costs.  Checked
as it prints: Eq. 6 recomputed from the models' likelihoods and the social belief before the
step, against the recorded one.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

NAMES = ("agent_1", "agent_2")
ACTIONS = ("up", "right", "down", "left", "stay", "pick", "place", "take", "give", "swap")


def arr(x) -> np.ndarray:
    """A recorded list as floats; null (an action not open, a model not consulted, or a
    likelihood of 0) as NaN."""
    return np.array(x, dtype=float)


def logsumexp(v: np.ndarray) -> float:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -np.inf
    top = v.max()
    return float(top + np.log(np.exp(v - top).sum()))


def fmt(x: float, width: int = 9) -> str:
    if x is None or not np.isfinite(x):
        return "-".rjust(width)
    if x != 0.0 and (abs(x) < 1e-3 or abs(x) >= 1e4):
        return f"{x:.2e}".rjust(width)
    return f"{x:.4f}".rjust(width)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("record", type=Path)
    ap.add_argument("--step", type=int, required=True, help="the video's step (the trace's row)")
    ap.add_argument("--robot", choices=NAMES, default=None,
                    help="whose update to print (default: both robots')")
    ap.add_argument("--dest", nargs="+", default=None, metavar="G",
                    help="the destinations to compare model by model (default: all)")
    ap.add_argument("--min-social", type=float, default=0.01,
                    help="social goals held below this before the step are left out of the "
                         "model tables (default 0.01)")
    ap.add_argument("--all-social", action="store_true", help="every social goal")
    args = ap.parse_args()

    record = json.loads(args.record.read_text())
    trace = record["trace"]
    if not 0 <= args.step < len(trace):
        sys.exit(f"belief_step: the record has steps 0 to {len(trace) - 1}")
    row = trace[args.step]
    if "belief_update" not in row:
        sys.exit("belief_step: this record has no \"belief_update\" (made before 2026-11-04, or "
                 "with --no-belief-detail): roll it out again with the code of now")
    actions = tuple(record.get("actions") or ACTIONS)
    print(f"{args.record.name}: step {args.step} (trace step {row['step']})")
    for i in (0, 1):
        print(f"  {NAMES[i]} at {tuple(row['pos'][i])}: {actions[row['action'][i]]}"
              f" -> {tuple(row['next_pos'][i])}")
    readers = (0, 1) if args.robot is None else (NAMES.index(args.robot),)
    for i in readers:
        d = row["belief_update"][i]
        if d is None:
            continue
        print_update(d, i, actions, args)
    return 0


def print_update(d: dict, i: int, actions, args) -> None:
    j = 1 - i
    a = int(d["action"])
    print()
    print(f"=== {NAMES[i]}'s belief about {NAMES[j]}, after {NAMES[j]}'s {actions[a]} ===")
    if d.get("skipped"):
        print(f"  no update: {d['skipped']}")
        return
    print(f"  stage: {d['stage']}; likelihood: {d['likelihood']}"
          f"{'' if d['factored'] else ' (one joint posterior)'}"
          f"{'' if d['stages'] else ', no stages'}")
    print(f"  units: c(move) = {d['c_move']:g}, c(stay) = {d['c_stay']:g} reward units; "
          f"tau = {d['tau']:g} reward units ({d['tau'] / d['c_move']:g} x c(move)) for the "
          f"likelihood, {d['tau_planning']:g} for planning; lapse {d['lapse']:g}; hazards "
          + ", ".join(f"{k} {v:g}" for k, v in d["hazards"].items()))
    obj = np.array(d["hypotheses"]["object"], dtype=int)
    dst = np.array(d["hypotheses"]["destination"], dtype=int)
    social = d["hypotheses"]["social"]
    prior = d["prior"]
    soc_prior = arr(prior["social"]) if prior.get("social") is not None else None

    # ---- per destination and per object -----------------------------------------------------
    for key, label, names in (("by_destination", "destination", "G"), ("by_object", "object", "O")):
        f = d[key]
        lik = arr(f["log_lik"])
        rel = np.exp(lik - np.nanmax(lik)) if np.isfinite(lik).any() else lik
        print(f"\n  per {label}: P(a | {label}) = Eq. 6, the other factor averaged out under the "
              f"belief before; b_t before the step, the posterior before the hazards, b_t+1 "
              f"after")
        print(f"    {'':>4} {'P(a|.)':>9} {'relative':>9} {'w/ idle':>9} {'b_t':>9} "
              f"{'posterior':>9} {'b_t+1':>9}")
        n = len(lik)
        for v in range(n):
            tag = f"{names}{v}" if names == "G" else f"O{v + 1}"
            print(f"    {tag:>4} {fmt(np.exp(lik[v]))} {fmt(rel[v])} "
                  f"{fmt(np.exp(arr(f['log_lik_with_idle'])[v]))} {fmt(arr(f['prior'])[v])} "
                  f"{fmt(arr(f['posterior'])[v])} {fmt(arr(f['after'])[v])}")
    keep = np.array([bool(x) for x in d["transition_term"]])
    if not keep.all():
        out = [f"(O{obj[g] + 1}, G{dst[g]})" for g in np.flatnonzero(~keep)]
        print(f"\n  ruled out by what the hands did (Eq. 5's transition term): {', '.join(out)}")

    # ---- model by model ---------------------------------------------------------------------
    m = d.get("models") or d.get("level0")
    if m is None:
        return
    level0 = "models" not in d
    q = arr(m["q"])                          # (PG, nT, 10) or (PG, 10)
    if level0:
        q = q[:, None, :]
    taken, best = arr(m["q_taken"]), arr(m["q_best"])
    best_action = np.array(m["best_action"], dtype=float)
    gap, log_psi, log_lik = arr(m["q_gap"]), arr(m["log_psi"]), arr(m["log_lik"])
    if level0:
        taken, best, best_action, gap, log_psi, log_lik = (
            x[:, None] for x in (taken, best, best_action, gap, log_psi, log_lik))
    # check: Eq. 6 from the models and the social belief before the step
    if not level0 and soc_prior is not None and d["likelihood"] == "level1":
        with np.errstate(divide="ignore"):
            ls = np.log(soc_prior)
        again = np.array([logsumexp(ls + log_lik[g]) for g in range(len(obj))])
        rec = arr(d["phys_log_lik"])
        both = np.isfinite(again) & np.isfinite(rec)
        diff = float(np.max(np.abs(again[both] - rec[both]))) if both.any() else 0.0
        print(f"\n  check: Eq. 6 recomputed from the models' likelihoods and the social belief "
              f"= the recorded one to {diff:.1e}")
    keep_dest = None
    if args.dest:
        keep_dest = {int(x.lstrip("Gg")) for x in args.dest}
    po = arr(prior["object"])
    objects = [o for o in range(len(po)) if po[o] >= 0.01] or [int(np.nanargmax(po))]
    if level0:
        socials = [0]
    else:
        socials = sorted(range(len(social)), key=lambda t: -(soc_prior[t] if soc_prior is not
                                                             None else 0.0))
        if not args.all_social and soc_prior is not None:
            socials = [t for t in socials if soc_prior[t] >= args.min_social]
    phys = arr(prior["phys"])
    for t in socials:
        head = ("the level-0 models" if level0 else
                f"social goal {social[t]}"
                + ("" if soc_prior is None else f" (b_t = {soc_prior[t]:.3f})"))
        print(f"\n  {head}: each (object, destination) model's Q at the state, what it makes "
              f"of {actions[a]}")
        print(f"    {'model':>10} {'b_t(o,d)':>9} {'Q(a_t)':>9} {'max Q':>9} {'best':>6} "
              f"{'gap':>9} {'gap/move':>9} {'psi(a_t)':>9} {'lapsed':>9}   Q of every action")
        for g in range(len(obj)):
            if obj[g] not in objects or (keep_dest is not None and dst[g] not in keep_dest):
                continue
            if not np.isfinite(phys[g]) or phys[g] <= 0.0:
                continue
            ba = int(best_action[g, t]) if np.isfinite(best_action[g, t]) else -1
            row_q = " ".join(f"{actions[k][:5]}={fmt(q[g, t, k], 1).strip()}"
                             for k in range(q.shape[-1]) if np.isfinite(q[g, t, k]))
            print(f"    {'(O%d, G%d)' % (obj[g] + 1, dst[g]):>10} {fmt(phys[g])} "
                  f"{fmt(taken[g, t])} {fmt(best[g, t])} "
                  f"{(actions[ba][:6] if ba >= 0 else '-'):>6} {fmt(gap[g, t])} "
                  f"{fmt(gap[g, t] / d['c_move'])} {fmt(np.exp(log_psi[g, t]))} "
                  f"{fmt(np.exp(log_lik[g, t]))}   {row_q}")

    # ---- the social posterior and the idle mode ----------------------------------------------
    if d.get("social_log_lik") is not None and soc_prior is not None:
        sl = arr(d["social_log_lik"])
        rel = np.exp(sl - np.nanmax(sl)) if np.isfinite(sl).any() else sl
        sp, sa = arr(d["social_posterior"]), arr(d["after"]["social"])
        print("\n  social goals (Eq. 5: the models' likelihood averaged over the physical belief "
              "the hands' outcomes keep)")
        print(f"    {'':>12} {'P(a|xi)':>9} {'relative':>9} {'b_t':>9} {'posterior':>9} "
              f"{'b_t+1':>9}")
        for t in range(len(social)):
            print(f"    {social[t]:>12} {fmt(np.exp(sl[t]))} {fmt(rel[t])} {fmt(soc_prior[t])} "
                  f"{fmt(sp[t])} {fmt(sa[t])}")
    l_idle = 0.0 if d.get("idle_log_lik") is None else float(np.exp(d["idle_log_lik"]))
    print(f"\n  idle mode: {prior['idle']:.4f} before, P(a | idle) = "
          f"{fmt(l_idle).strip()}, "
          f"{fmt(d.get('idle_posterior', np.nan)).strip()} after the step, "
          f"{d['after']['idle']:.4f} after its hazard")


if __name__ == "__main__":
    sys.exit(main())
