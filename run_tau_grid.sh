#!/bin/sh
# All 49 pairings at every pair of taus (2026-11-01): agent_1 at each tau of TAUS1, agent_2 at
# each of TAUS2 -- each robot reads its partner at its own tau (Eq. 7's softmax it predicts
# the partner by, and the likelihood of its belief update; Config.robot_temperatures).  The
# rest are the 49's settings (run_all_49.sh): both robots at level 2, everything inferred,
# swaps and taking, a pick-up of three transitions at two moves' worth, carry 1, inertia 1.
#
# Run from the folder holding social_pairings/ and examples/, after
#     pip install -r requirements.txt
#
#     sh run_tau_grid.sh                                   # 0.1 ... 0.5 x 0.1 ... 0.5, all 49
#     TAUS="0.1 0.2 0.3" sh run_tau_grid.sh                # the same list for both robots
#     TAUS1="0.2" TAUS2="0.1 0.3 0.5" sh run_tau_grid.sh   # agent_1 fixed, agent_2 swept
#     GOALS="competition coercion" sh run_tau_grid.sh      # only the pairings of these goals
#
# One folder per pair of taus, under output/ (..._tau<t1>_<t2>_..., or ..._tau<t>_... where the
# two are equal: the folder of `--temperature <t>`).  Interrupted? Run the same line again:
# --resume keeps every rollout already saved.  At the end, tools/tau_grid.py puts them side
# by side in output/tau_grid/ (OUT=...): tau_grid.png (every pairing over both taus),
# tau_grid_counts.png, pairings/<agent_1>__<agent_2>.png (one pairing, with the transitions),
# tau_grid.md and tau_grid.csv.
#
# Size: 25 pairs of taus x 49 pairings = 1225 rollouts by default.  A rollout takes 10-40
# minutes on 2 threads (the ones that run to the 60-step cap the longest; about 15 h for the
# 49 of one pair), so the default grid is some 375 h of 2-thread work: about 12 h with 32
# workers (64 cores), 24 h with 16.  Each worker needs about 2.5 GB at its peak.
# Environment: TAUS1, TAUS2 (default TAUS, default "0.1 0.2 0.3 0.4 0.5"), GOALS (default
# all seven), THREADS per worker (default 2), WORKERS (default: the CPUs / THREADS), PHYS
# (Config.physical_likelihood), SCENE (default two), STEPS (default 60), NOVIDEO (default 1:
# records only -- 1225 x 3 videos is a lot; NOVIDEO=0 draws them), LOGS (default
# logs/tau_grid), OUT (default output/tau_grid), PYTHON (default python3).
set -e
PY=${PYTHON:-python3}
TAUS=${TAUS:-0.1 0.2 0.3 0.4 0.5}
TAUS1=${TAUS1:-$TAUS}
TAUS2=${TAUS2:-$TAUS}
GOALS=${GOALS:-cooperation conflict competition coercion exchange none hinder}
FLAGS="--levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
[ -n "$PHYS" ] && FLAGS="$FLAGS --physical-likelihood $PHYS"
EXTRA="--scene ${SCENE:-two} --steps ${STEPS:-60}"
[ "${NOVIDEO:-1}" = 1 ] && EXTRA="$EXTRA --no-video"
N=$($PY -c "import os; print(os.cpu_count() or 2)")
THREADS=${THREADS:-2}
WORKERS=${WORKERS:-$(( N / THREADS ))}
[ "$WORKERS" -lt 1 ] && WORKERS=1
LOGS=${LOGS:-logs/tau_grid}
mkdir -p "$LOGS"
n1=$(echo $TAUS1 | wc -w); n2=$(echo $TAUS2 | wc -w); ng=$(echo $GOALS | wc -w)
echo "tau grid: agent_1 at $TAUS1, agent_2 at $TAUS2; $(( n1 * n2 )) pairs of taus x $(( ng * ng )) pairings = $(( n1 * n2 * ng * ng )) rollouts; $WORKERS workers x $THREADS threads; logs in $LOGS"

# compile the kernels once (numba caches them) before the workers start together
$PY examples/run_object_inference.py $FLAGS --temperature 0.2 0.3 --scene tiny --steps 2 \
  --no-video --output "$LOGS/warmup" --threads "$THREADS" --pairing none none > "$LOGS/warmup.log" 2>&1

# one job per (agent_1's tau, agent_2's tau, agent_1's goal): its pairings with every goal of
# agent_2, one after the other; WORKERS jobs at a time, a new one as soon as one ends
JOBS="$LOGS/jobs.txt"
: > "$JOBS"
for t1 in $TAUS1; do for t2 in $TAUS2; do for a in $GOALS; do
  echo "$t1 $t2 $a" >> "$JOBS"
done; done; done
export PY FLAGS EXTRA THREADS LOGS GOALS
xargs -P "$WORKERS" -L 1 sh -c '
  t1=$0; t2=$1; a=$2; ARGS=""
  for b in $GOALS; do ARGS="$ARGS --pairing $a $b"; done
  log="$LOGS/run_${t1}_${t2}_$a.log"
  if $PY examples/run_object_inference.py $FLAGS $EXTRA --temperature $t1 $t2 \
       --threads "$THREADS" --resume $ARGS > "$log" 2>&1; then
    echo "done: agent_1 at tau $t1, agent_2 at $t2, agent_1 $a"
  else
    echo "FAILED: agent_1 at tau $t1, agent_2 at $t2, agent_1 $a (see $log)"
  fi
' < "$JOBS"

# per pair of taus, one pass over its saved rollouts (nothing is rolled out again): the
# folder's results.md, results.png and summary.json
: > "$LOGS/folders.txt"
for t1 in $TAUS1; do for t2 in $TAUS2; do
  ARGS=""
  for a in $GOALS; do for b in $GOALS; do ARGS="$ARGS --pairing $a $b"; done; done
  log="$LOGS/table_${t1}_${t2}.log"
  if $PY examples/run_object_inference.py $FLAGS $EXTRA --temperature $t1 $t2 \
       --threads "$THREADS" --resume $ARGS > "$log" 2>&1; then
    tail -n 1 "$log" >> "$LOGS/folders.txt"
  else
    echo "FAILED: the table of agent_1 at tau $t1, agent_2 at $t2 (see $log)"
  fi
done; done
$PY tools/tau_grid.py --out "${OUT:-output/tau_grid}" $(cat "$LOGS/folders.txt")
