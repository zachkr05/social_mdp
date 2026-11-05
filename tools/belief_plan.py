"""What the belief planner (Config.belief_planning, social_pairings/beliefplan.py) decided, from a
record: per decision the planning robot's action and the runner-up, their Q, the Q with the
partner's belief held (frozen), Delta_belief (the value of this transition's belief update),
the partner's predicted and actual action, the partner's belief that the robot wants each
object now and after each action, and the planned way.  Then the story of the objects.

    python3 tools/belief_plan.py RECORD.json
    python3 tools/belief_plan.py RECORD.json --step 12 --plan 15      # one decision, its way
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_pairings.serialize import action_names, load_result      # noqa: E402


def arr(x):
    return np.array([np.nan if v is None else v for v in x], dtype=float)


def fmt_obj(b) -> str:
    return " ".join(f"O{k + 1} {v:.2f}" for k, v in enumerate(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=Path)
    ap.add_argument("--step", type=int, default=None, help="only this transition")
    ap.add_argument("--plan", type=int, default=8, help="transitions of the planned way shown")
    ap.add_argument("--story", action="store_true",
                    help="only the story: per transition both actions, the hands, and each "
                         "robot's belief about which object its partner wants")
    args = ap.parse_args()
    res = load_result(args.record)
    names = action_names(res)
    if args.story:
        story(res, names)
        return
    bp = res.get("belief_planner")
    if not bp:
        raise SystemExit("no belief planner in this record (Config.belief_planning)")
    print(f"{args.record.name}: {res['social_goals']} done at {res['done_at']}, "
          f"{res['steps']} transitions, stopped by {res['stopped_by']}")
    print(f"mode {bp['mode']}, objective {bp['objective']}, horizon {bp['horizon']}, at most "
          f"{bp['points']} points (spacing {bp['spacing']}), {bp['refinements']} refinements, "
          f"interpolation {bp['interpolation']}, tail {bp['tail']}")
    print(f"the partner predicted by {bp['response']}")
    right = [d.get("partner_as_predicted") for d in bp["decisions"]
             if d.get("partner_as_predicted") is not None]
    print(f"partner predicted right at {sum(right)} of {len(right)} decisions")
    for d in bp["decisions"]:
        if args.step is not None and d["transition"] != args.step:
            continue
        q = arr(d["q"])
        qf = arr(d["q_frozen"])
        best, run = d["best_action"], d["runner_up"]
        chosen = d.get("chosen")
        fb = int(np.nanargmax(np.where(np.isfinite(qf), qf, -np.inf)))
        line = (f"t={d['transition']:3d} R{d['robot']} {names[chosen] if chosen is not None else '?':>5} "
                f"(best {names[best]} {q[best]:.3f}"
                + (f", next {names[run]} {q[run]:.3f}, gap {d['gap']:.3f}" if run is not None else "")
                + f"; frozen best {names[fb]})  partner {names[d['partner_predicted']]}"
                + (f"/{names[d['partner_actual']]}" if d.get("partner_actual") is not None else "")
                + f"  points {d['points']}  {d['timing']['total']:.0f}s")
        print(line)
        now = d["now"]
        print(f"        partner's belief now: {fmt_obj(now['object'])}; destination "
              + " ".join(f"{v:.2f}" for v in now["destination"])
              + ("" if now["social"] is None else "; social " + " ".join(f"{v:.2f}" for v in now["social"]))
              + f"; idle {now['idle']:.2f}")
        if "delta_belief" in d:
            db = arr(d["delta_belief"])
            ok = np.isfinite(db)
            print("        Delta_belief: " + ", ".join(f"{names[a]} {db[a]:+.3f}"
                                                     for a in np.flatnonzero(ok)))
            after = d.get("after", {})
            print("        after each action, the partner's belief in the objects: "
                  + "; ".join(f"{a}: {fmt_obj(v['object'])}" for a, v in after.items()))
        if d.get("rounds"):
            print("        rounds: " + "; ".join(f"{r['points']} points, residual max "
                                                 f"{r['residual_max']:.3f}" for r in d["rounds"]))
        if args.step is not None or args.plan:
            plan = d.get("plan", [])[:args.plan]
            if plan:
                print("        planned way: " + " | ".join(
                    f"{names[x['robot']]}/{names[x['partner']]} {fmt_obj(x['object'])}"
                    for x in plan))
    print("events:")
    for ev in res["events"]:
        print("   ", ev)


def story(res: dict, names) -> None:
    """Per transition: positions, both actions, what each holds after, the events, and each
    robot's belief (after the transition) about which object its partner wants."""
    print(f"{res['social_goals']} done at {res['done_at']}, {res['steps']} transitions, "
          f"stopped by {res['stopped_by']}")
    for row in res["trace"]:
        a0, a1 = row["action"]
        ob = row.get("next_object_belief") or row.get("object_belief") or [None, None]
        b = "  ".join(f"R{i} thinks R{1 - i} wants " + fmt_obj(ob[i])
                      for i in (0, 1) if ob[i] is not None)
        ev = "; ".join(f"{e[0]} R{e[1]} O{e[2] + 1}" for e in row["events"])
        print(f"{row['step']:3d} R0 {str(row['pos'][0]):>7} {names[a0]:>5} | R1 "
              f"{str(row['pos'][1]):>7} {names[a1]:>5} | holds {row['next_holds']} | {b}"
              + (f" | {ev}" if ev else ""))


if __name__ == "__main__":
    main()
