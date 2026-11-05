#!/bin/sh
# The belief planner with the hinderer INFERRED vs GIVEN (Config.belief_partner, 2026-11-05
# evening; the user: "So can u modify it so that everything is inferred and nothing is I guess
# truly known. Red should infer the belief from green then apply to formula to manipulate it").
#
# Hinder / coercion on the two-room scene at the 49's settings, tau 0.2, the task reward, the
# belief planner updated (40 points, 2 refinements, horizon 40: the defaults).  The coercer
# (red) plans against
#   inferred -- the hypotheses (physical goal, social goal) of its own posteriors about the
#               hinderer, the most likely covering 95 % of the mass (at most 4), weighted (QMDP);
#   oracle   -- the hinderer's true goals (the rule of this morning).
# Two worlds:
#   default  -- both robots may take what the other holds;
#   decoy    -- only the hinderer may take, and the hinderer has lifted the coercer's object
#               while the coercer waited beside the decoy (5 transitions forced).  Under the
#               oracle the coercer lifts the decoy, the hinderer puts the object back and
#               chases it, the coercer swaps and delivers at 23 (the desktop's run of 05:30).
# The 4 rollouts run side by side, THREADS each (default: the CPUs / 4), after a warm-up that
# compiles the numba kernels once.
#
#     sh run_belief_partner.sh
#     THREADS=8 sh run_belief_partner.sh
#     PARTNERS="inferred" WORLDS="decoy" sh run_belief_partner.sh
#     EXTRA="--belief-partner-max 6" sh run_belief_partner.sh     # more hypotheses (folders _bpk6)
#
# Records and videos: output/object_inference_two_..._bplanupdated[_bpinf]/ (the oracle's
# folders carry no _bpinf); logs: logs/belief_partner/<world>_<partner>.log.  Each decision --
# the hypotheses, the weights, the predicted and the actual move of the hinderer:
#     python3 tools/belief_plan.py output/<folder>/hinder__coercion__inferred.json
# Environment: PY (python3), THREADS, WORLDS ("default decoy"), PARTNERS ("inferred oracle"),
# STEPS (60), NOVIDEO (0), EXTRA (more flags for every rollout).
cd "$(dirname "$0")" || exit 1
PY=${PY:-python3}
NCPU=$(nproc 2>/dev/null || echo 4)
WORLDS=${WORLDS:-default decoy}
PARTNERS=${PARTNERS:-inferred oracle}
N=0
for w in $WORLDS; do for p in $PARTNERS; do N=$((N + 1)); done; done
[ "$N" -gt 0 ] || { echo "nothing to run: WORLDS / PARTNERS empty"; exit 1; }
if [ -z "$THREADS" ]; then
  THREADS=$((NCPU / N))
  [ "$THREADS" -ge 1 ] || THREADS=1
fi
FLAGS="--levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
FLAGS="$FLAGS --temperature 0.2 --pairing hinder coercion --steps ${STEPS:-60}"
FLAGS="$FLAGS --objective task --belief-planning updated --verbose --resume"
[ "${NOVIDEO:-0}" = 1 ] && FLAGS="$FLAGS --no-video"
LOGS=logs/belief_partner
mkdir -p "$LOGS"
export NUMBA_NUM_THREADS="$THREADS"
echo "$(date '+%F %T') warm-up (numba kernels, the tiny scene)"
$PY -u examples/run_object_inference.py $FLAGS --scene tiny --steps 2 --no-video \
  --threads "$THREADS" --output "$LOGS/warmup" > "$LOGS/warmup.log" 2>&1 || {
  echo "the warm-up failed: see $LOGS/warmup.log"; exit 1; }
for w in $WORLDS; do
  for p in $PARTNERS; do
    case $w in
      default) world="" ;;
      decoy)   world="--steal-robots 1 0 --forced-actions +y,stay pick,stay pick,stay pick,stay stay,stay" ;;
      *) echo "unknown world $w (default, decoy)"; exit 1 ;;
    esac
    log="$LOGS/${w}_${p}.log"
    echo "$(date '+%F %T') start $w / $p ($THREADS threads) -> $log"
    $PY -u examples/run_object_inference.py $FLAGS $world --belief-partner "$p" \
      --threads "$THREADS" $EXTRA > "$log" 2>&1 < /dev/null &
  done
done
wait
echo
for w in $WORLDS; do
  for p in $PARTNERS; do
    log="$LOGS/${w}_${p}.log"
    echo "== $w / $p: $(grep -E 'transitions, stopped by' "$log" | tail -1 | sed 's/^\[pairings\] //')"
    grep -E "done at|R1 set its object down|Traceback|Error" "$log" | tail -3
  done
done
