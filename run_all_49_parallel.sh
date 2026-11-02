#!/bin/sh
# All 49 pairings of the seven social goals (agent_1's goal x agent_2's), the settings of
# 2026-10-18/21: nothing automatic in the hands, a pick-up of three transitions costing two
# moves on the first and nothing after (--pick-cost 2), place / take / give / swap two moves;
# both robots at level 2; each infers its partner's object, destination and social goal;
# swaps and taking allowed; carrying the partner's object at one move; inertia 1; the belief
# update of 2026-10-30 (Eqs. 5-6 as the authors' solver keeps them: README). The hinderer has no goal of its own and
# competition is read as a rival (package defaults).
#
# Run from the folder holding social_pairings/ and examples/, after
#     pip install -r requirements.txt
# Interrupted? Run the same line again: --resume keeps the pairings already saved.
# Environment: TAU (Eq. 7's tau, default 0.2; two values, TAU="0.2 0.5": agent_1's and
# agent_2's, each robot reading its partner at its own -- run_tau_grid.sh sweeps both), PHYS (Config.physical_likelihood: level1 default, level0 the authors' estimator), THREADS (default 0 = every CPU), SCENE (default two), STEPS (default 60),
# CONFLICT (Config.conflict: task, the default since 2026-11-01 -- a level-2 conflict robot
# paid -w r(g_j) against every partner, folders ..._taskconflict, which take the record of
# every pairing without conflict from the folder of before; table -- Table I's -w R_j),
# OUTPUT (default: the folder named after the options, under output/), NOVIDEO=1 (records only),
# PYTHON (default python3).
set -e
PY=${PYTHON:-python3}
FLAGS="--levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
[ -n "$TAU" ] && FLAGS="$FLAGS --temperature $TAU"      # Eq. 7's tau (default 0.2), one or one per robot
[ -n "$PHYS" ] && FLAGS="$FLAGS --physical-likelihood $PHYS"   # level1 (Eq. 6 as printed, default) or level0 (the authors' code)
[ -n "$CONFLICT" ] && FLAGS="$FLAGS --conflict $CONFLICT"   # task (default) or table (Table I's -R_j)
EXTRA="--scene ${SCENE:-two} --steps ${STEPS:-60}"
[ -n "$OUTPUT" ] && EXTRA="$EXTRA --output $OUTPUT"
[ "${NOVIDEO:-0}" = 1 ] && EXTRA="$EXTRA --no-video"

# WORKERS processes at once (default 7: one per agent_1 goal, its 7 pairings each), THREADS
# threads each (default: the CPUs / WORKERS). Each needs about 2.5 GB of memory at its peak.
# Then one pass over all 49 with --resume writes results.md, results.png and summary.json
# (and draws the video of any record that has none).
N=$($PY -c "import os; print(os.cpu_count() or 2)")
WORKERS=${WORKERS:-7}
T=${THREADS:-$(( N / WORKERS ))}
[ "$T" -lt 1 ] && T=1
mkdir -p logs
# compile the kernels once (numba caches them) before the workers start together
$PY examples/run_object_inference.py $FLAGS --scene tiny --steps 2 --no-video \
  --output logs/warmup --threads "$T" --pairing none none > logs/warmup.log 2>&1
k=0
for a in cooperation conflict competition coercion exchange none hinder; do
  ARGS=""
  for b in cooperation conflict competition coercion exchange none hinder; do
    ARGS="$ARGS --pairing $a $b"
  done
  $PY examples/run_object_inference.py $FLAGS $EXTRA --threads "$T" --resume $ARGS \
    > "logs/worker_$a.log" 2>&1 &
  k=$((k + 1))
  if [ "$k" -ge "$WORKERS" ]; then wait; k=0; fi
done
wait
grep -h "objects inferred done at" logs/worker_*.log | sort || true
$PY examples/run_object_inference.py $FLAGS $EXTRA --threads "$T" --resume --all-pairings \
  > logs/all49.log 2>&1
tail -n 60 logs/all49.log
