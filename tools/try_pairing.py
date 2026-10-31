"""Roll one pairing out on a scene with the 49-matrix settings and print the beliefs, step
by step: a quick look at the belief update of 2026-10-30.

    python tools/try_pairing.py competition cooperation --steps 40 --scene two --threads 2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from social_pairings import Config, run_pairing, two_room_scene      # noqa: E402
from social_pairings import goals as G                               # noqa: E402
from social_pairings.serialize import to_json                        # noqa: E402

SCENES = {"two": two_room_scene}


def tiny_scene():
    from social_pairings.scenes import Scene
    return Scene(name="tiny", grid=np.ones((3, 3)), starts=((0, 1), (2, 1)),
                 pickups=((2, 2), (0, 0)), destinations=((0, 2), (2, 0)),
                 candidate_cells=((1, 0),))


SCENES["tiny"] = tiny_scene


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("goals", nargs=2)
    ap.add_argument("--scene", default="two", choices=tuple(SCENES))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--value-maps", action="store_true")
    args = ap.parse_args()
    cfg = Config(num_threads=args.threads, swap_on_place=True, steal=True, foreign_carry_cost=1.0,
                 inertia=1.0, pick_cost_rate=2.0, temperature=args.tau,
                 value_maps=args.value_maps, verbose=False)
    tic = time.perf_counter()
    result = run_pairing(SCENES[args.scene](), tuple(args.goals), levels=(2, 2), cfg=cfg,
                         n_steps=args.steps)
    print(f"{args.goals[0]} / {args.goals[1]} on {args.scene}: done at {result['done_at']}, "
          f"stopped by {result['stopped_by']} after {result['steps']} transitions, "
          f"{time.perf_counter() - tic:.0f} s")
    names = list(G.SOCIAL_GOALS)
    print("  (each line: what robot i believes about its partner as the step begins -- its"
          " three likeliest social goals, its object, its destination, that it stands still)")
    for t in result["trace"]:
        print(f"{t['step']:3d} pos {t['pos']} act {t['action']} holds {t['holds']} done {t['done']}")
        for i in (0, 1):
            sb = t["social_belief"][i]
            top = sorted(zip(sb, names), reverse=True)[:3]
            print(f"      r{i}: social " + ", ".join(f"{n} {v:.2f}" for v, n in top)
                  + " | obj " + " ".join(f"{v:.2f}" for v in t["object_belief"][i])
                  + " | dst " + " ".join(f"{v:.2f}" for v in t["belief"][i])
                  + f" | idle {t['idle_belief'][i]:.2f}")
    if args.out:
        args.out.write_text(json.dumps(to_json(result)))


if __name__ == "__main__":
    main()
