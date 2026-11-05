#!/bin/sh
# The advisor's comparison (2026-11-05; social_pairings/README.md, "Planning over the
# partner's belief"): hinder / coercion on the two-room scene, the coercer's reward
# (Table I's, or its true task) x the hinderer's belief in the coercer's plan (held, or
# updated by the hinderer's own rule) -- and the planner of Eq. 4 under both rewards, the
# baseline of before.  The scene, the beliefs, the likelihood, the hazards, the choice rules,
# the costs and the stopping rules the same in every run.
#
#     sh run_belief_2x2.sh                         # tau 0.2, 60 transitions, the six runs
#     TAU=0.2 HAZARD=0.02 sh run_belief_2x2.sh     # ... the hinderer's goal hazard 0.02
#     ONLY="updated:task" sh run_belief_2x2.sh     # one condition (planner:objective)
#
# The decoy (2026-11-05; social_pairings/README.md, "What it found"): only the hinderer may
# take, and the hinderer has lifted the coercer's object while the coercer waited beside the
# decoy -- the belief planner lifts the decoy, the hinderer puts the object back and chases
# it, the coercer swaps the decoy for its object and delivers (done at 23):
#
#     EXTRA="--steal-robots 1 0 --forced-actions +y,stay pick,stay pick,stay pick,stay stay,stay" \
#         sh run_belief_2x2.sh
#
# Environment: TAU (0.2), STEPS (60), HAZARD (none), THREADS (0 = every CPU), NOVIDEO (0: draw
# the videos), EXTRA (more flags, e.g. "--belief-horizon 30"), ONLY (a list of
# planner:objective, planner one of off frozen updated, objective one of table task).
# Each run's record and video go to output/object_inference_<options>/; the summary at the
# end is tools/belief_plan.py's.
set -e
cd "$(dirname "$0")"
PY=${PY:-python3}
TAU=${TAU:-0.2}
STEPS=${STEPS:-60}
FLAGS="--levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
FLAGS="$FLAGS --temperature $TAU --pairing hinder coercion --steps $STEPS --threads ${THREADS:-0}"
[ -n "$HAZARD" ] && FLAGS="$FLAGS --goal-hazard $HAZARD"
[ "${NOVIDEO:-0}" = 1 ] && FLAGS="$FLAGS --no-video"
CONDITIONS=${ONLY:-"off:table off:task frozen:table frozen:task updated:table updated:task"}
LOGS=${LOGS:-logs/belief_2x2}
mkdir -p "$LOGS"
for c in $CONDITIONS; do
  planner=${c%%:*}
  objective=${c##*:}
  args="--objective $objective"
  [ "$planner" != off ] && args="$args --belief-planning $planner"
  echo "== $planner / $objective"
  $PY -u examples/run_object_inference.py $FLAGS $args $EXTRA --verbose \
    > "$LOGS/${planner}_${objective}.log" 2>&1
  tail -n 3 "$LOGS/${planner}_${objective}.log"
done
