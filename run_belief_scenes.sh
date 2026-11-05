#!/bin/sh
# The advisor's comparison on four bigger scenes (2026-11-05; the user: "Can you launch them on
# all 4 scenes just hinder vs coercion" -- the gate scene and three house scenes, the full
# comparison, both robots taking): hinder / coercion, the coercer under each of {the planner
# of Eq. 4, the belief planner with the hinderer's belief held, the belief planner with it
# updated} x {Table I's reward, the true-task reward}, at the 49's settings, tau 0.2.
#
#     sh run_belief_scenes.sh                      # the 24 rollouts, one at a time
#     SCENES="near_miss" sh run_belief_scenes.sh   # one scene's six
#
# One rollout at a time: a house scene's model alone is ~10 GiB here (its solver capped by
# --max-memory-gib), the belief planner some 5-6 GiB more.  The gate scene of the testbed (an
# object put down anywhere, 60-70 GiB a rollout) does not fit this machine: gate_few, the
# same map with four cells to put an object down on (the gate, a far cell in each room, the
# middle of the way round), stands in for it.  The planner on the house scenes: 12 belief
# points, one round of points along the planned way, a horizon of 30 (the likelihood table
# in single precision there; beliefplan.LL_F64_BYTES).  Rounds of the conditions, every
# scene in each: the belief planner with the task reward first, then the planner of Eq. 4
# with it, then the rest -- the comparisons that matter most come first.  --resume: a
# rollout already saved is kept.  Logs in logs/belief_scenes/, records and videos in
# output/object_inference_<scene>_..._<options>/.
#
# PARTNER (2026-11-05 evening; Config.belief_partner): whom the belief planner takes the hinderer
# to be -- unset or "inferred" (the default of the code since then: the hypotheses of the
# coercer's own posteriors, folders _bpinf), or "oracle" (its true goals, the folders of this
# morning's batch).
#
# Environment: SCENES (default "gate_few near_miss hall_crossing head_on_corridor"), STEPS
# (60), TAU (0.2), CONDITIONS (default "updated:task off:task frozen:task updated:table
# off:table frozen:table"), NOVIDEO (0), THREADS (0 = every CPU).
cd "$(dirname "$0")"
PY=${PY:-python3}
SCENES=${SCENES:-gate_few near_miss hall_crossing head_on_corridor}
CONDITIONS=${CONDITIONS:-updated:task off:task frozen:task updated:table off:table frozen:table}
FLAGS="--levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
FLAGS="$FLAGS --temperature ${TAU:-0.2} --pairing hinder coercion --steps ${STEPS:-60}"
FLAGS="$FLAGS --threads ${THREADS:-0} --resume --verbose"
[ "${NOVIDEO:-0}" = 1 ] && FLAGS="$FLAGS --no-video"
LOGS=logs/belief_scenes
mkdir -p "$LOGS"
for c in $CONDITIONS; do
  planner=${c%%:*}
  objective=${c##*:}
  for sc in $SCENES; do
    case $sc in
      gate*) mem=14; plan="--belief-points 24 --belief-refinements 1" ;;
      *)     mem=10; plan="--belief-points 12 --belief-refinements 1 --belief-horizon 30" ;;
    esac
    args="--scene $sc --objective $objective --max-memory-gib $mem"
    [ "$planner" != off ] && args="$args --belief-planning $planner $plan${PARTNER:+ --belief-partner $PARTNER}"
    log="$LOGS/${sc}_${planner}_${objective}.log"
    echo "$(date '+%F %T') start $sc $planner/$objective"
    if $PY -u examples/run_object_inference.py $FLAGS $args > "$log" 2>&1 < /dev/null; then
      echo "$(date '+%F %T') done  $sc $planner/$objective: $(grep -E 'transitions, stopped by' "$log" | tail -1 | sed 's/^\[pairings\] //')"
    else
      echo "$(date '+%F %T') FAILED $sc $planner/$objective (see $log)"
    fi
  done
done
