"""Did a robot hinder its partner?  Measured on a rollout against the no-social baseline.

Two questions, kept apart:

* **Effect** -- is the partner's goal prevented or delayed, and how much of its physical
  return is denied?
* **Conduct** -- did the hinderer act with purpose: take the partner's object and keep it
  (or stand in its way), without picking things up and putting them back, giving the object
  away, or pacing?

A rollout can score on the first and fail the second (the partner never finishes because
*it* gives up, while the "hinderer" wanders); :func:`hindrance_report` reports both.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .constants import PICK, WAIT

__all__ = ["hindrance_report", "conduct", "physical_return", "format_report"]

_OPPOSITE = {0: 2, 2: 0, 1: 3, 3: 1}


def physical_return(result: dict, robot: int, horizon: int) -> float:
    """Discounted physical return ``sum_t gamma^t (r - c)`` of a robot over ``horizon``
    transitions.  After the rollout stopped, the robot is taken to stay as it was: parked
    on a finished goal (``rho - c(stay)`` per transition) or frozen in its last transition."""
    cfg = result["config"]
    dt = cfg["resolution"] / cfg["reference_step_m"]
    gamma = cfg["reference_gamma"] ** dt
    per_step = [float(row["physical_reward"][robot]) for row in result["trace"]]
    if result["done_at"][robot] is not None:
        steady = (cfg["reward_rate"] - cfg["wait_cost_rate"]) * dt
    else:
        steady = per_step[-1] if per_step else 0.0
    values = per_step[:horizon] + [steady] * max(horizon - len(per_step), 0)
    return float(sum(gamma ** t * v for t, v in enumerate(values)))


def conduct(result: dict, robot: int, cycle_window: int = 6) -> dict:
    """How a robot behaved, in counts that are 0 for purposeful behaviour:

    ``put_back_cycles``  it put an object back and picked the same one up again within
                         ``cycle_window`` transitions (or the reverse)
    ``reversals``        a move immediately undone by the opposite move
    ``spare_pickups``    pick-ups of an object that completes nobody's goal
    ``wasted_hand_actions``  hand actions (pick, place, take, give, swap) after which no
                         object changed state
    """
    last = {}                                    # object -> (transition, what) of this robot
    cycles = spare = 0
    for t, what, by, obj, _ in result["events"]:
        if by != robot or what not in ("picked up", "put back"):
            continue
        if what == "picked up" and obj >= 2:
            spare += 1
        if obj in last and last[obj][1] != what and t - last[obj][0] <= cycle_window:
            cycles += 1
        last[obj] = (t, what)
    actions = [int(row["action"][robot]) for row in result["trace"]]
    reversals = sum(1 for a, b in zip(actions, actions[1:]) if _OPPOSITE.get(a, -1) == b)
    wasted = sum(1 for row in result["trace"] if row["action"][robot] >= PICK
                 and not any(ev[1] == robot or ev[3] == robot for ev in row["events"]))
    return dict(put_back_cycles=cycles, reversals=reversals, spare_pickups=spare,
                wasted_hand_actions=wasted, moves=sum(a < WAIT for a in actions),
                waits=sum(a == WAIT for a in actions),
                hand_actions=sum(a >= PICK for a in actions))


def hindrance_report(result: dict, baseline: dict, hinderer: int = 0, *,
                     min_delay: int = 5, horizon: Optional[int] = None) -> dict:
    """Compare a rollout with the no-social baseline of the same scene.

    result, baseline : records of :func:`social_pairings.run_pairing` / ``run_baseline``.
    hinderer : the robot whose hindering is judged (its partner is the victim).
    min_delay : a delay of at least this many transitions counts as hindering.
    horizon : transitions over which the physical returns are compared (default: the
        longer of the two rollouts plus 20).

    Returns a dict:

    ``partner_done_at`` / ``baseline_done_at``   transition of the partner's goal, or None
    ``prevented``     the partner never finished although it did in the baseline
    ``delay``         transitions lost (None when prevented)
    ``return_denied`` share of the partner's baseline physical return it did not get, in [0, 1]
    ``took_object_at``, ``steps_holding``, ``releases``   the hinderer and the partner's object
                      (a release: put back, put down on a hand-off cell, handed over, set down)
    ``partner_blocked``   transitions in which the partner's move was blocked by the hinderer
    ``mechanism``     "holds the partner's object", "blocks the partner" or None
    ``hindered``      prevented, or delayed by at least ``min_delay``
    ``purposeful``    the hinderer used that mechanism and showed no put-back cycles, no
                      releases and no spare pick-ups
    ``conduct``       :func:`conduct` of both robots
    """
    partner = 1 - hinderer
    done, base_done = result["done_at"][partner], baseline["done_at"][partner]
    prevented = done is None and base_done is not None
    delay = None if (done is None or base_done is None) else int(done - base_done)
    if horizon is None:
        horizon = max(result["steps"], baseline["steps"]) + 20
    ret, base_ret = (physical_return(r, partner, horizon) for r in (result, baseline))
    denied = float(np.clip(1.0 - ret / base_ret, 0.0, 1.0)) if base_ret > 0 else 0.0

    took_at = next((t for t, what, by, obj, _ in result["events"]
                    if what in ("picked up", "took") and by == hinderer and obj == partner),
                   None)
    releases = sum(1 for _, what, by, obj, _ in result["events"]
                   if by == hinderer and obj == partner
                   and what in ("put back", "put down", "dropped", "handed over", "set down"))
    holding = sum(1 for row in result["trace"] if row["next_holds"][hinderer] == partner)
    blocked = sum(1 for row in result["trace"] if row["blocked"][partner])
    ending = result["holds"][hinderer] == partner
    mechanism = ("holds the partner's object" if holding and (ending or done is None) else
                 "blocks the partner" if blocked >= min_delay else None)
    behaviour = [conduct(result, i) for i in (0, 1)]
    hindered = bool(prevented or (delay is not None and delay >= min_delay))
    purposeful = bool(mechanism is not None and releases == 0
                      and behaviour[hinderer]["put_back_cycles"] == 0
                      and behaviour[hinderer]["spare_pickups"] == 0)
    return dict(hinderer=hinderer, partner=partner, partner_done_at=done,
                baseline_done_at=base_done, prevented=prevented, delay=delay,
                partner_return=ret, baseline_return=base_ret, return_denied=denied,
                took_object_at=took_at, steps_holding=holding, releases=releases,
                partner_blocked=blocked, mechanism=mechanism, hindered=hindered,
                purposeful=purposeful, conduct=behaviour, steps=result["steps"],
                stopped_by=result["stopped_by"])


def format_report(report: dict, names: Sequence[str] = ("R0", "R1")) -> str:
    """One line for a log or a table; ``names``: what the two robots are called."""
    r = report
    who = names[r["hinderer"]]
    if r["prevented"]:
        effect = f"partner never finishes (baseline: transition {r['baseline_done_at']})"
    elif r["delay"] is not None:
        effect = (f"partner done at {r['partner_done_at']} (baseline {r['baseline_done_at']}, "
                  f"delay {r['delay']:+d})")
    else:
        effect = "partner does not finish in the baseline either"
    took = ("never takes the partner's object" if r["took_object_at"] is None else
            f"takes the partner's object at {r['took_object_at']}, holds it for "
            f"{r['steps_holding']} of {r['steps']} transitions, releases {r['releases']}x")
    c = r["conduct"][r["hinderer"]]
    return (f"{'HINDERED' if r['hindered'] else 'not hindered'}"
            f"{' / purposeful' if r['purposeful'] else ' / NOT purposeful'}: {effect}; "
            f"{100 * r['return_denied']:.0f}% of its physical return denied; {who} {took}; "
            f"put-back cycles {c['put_back_cycles']}, reversals {c['reversals']}, spare "
            f"pick-ups {c['spare_pickups']}")
