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
#     SAME=1 TAUS="0.1 0.15 0.2" sh run_tau_grid.sh         # one tau for both robots: 3 x 49
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
# SAME=1: only the pairs with one tau for both robots, (t, t) for each t of TAUS1 -- the 49 at
# each tau (len(TAUS) x 49 rollouts; the figures then have the tau as one axis).
#
# Two decoy objects (2026-11-01): SCENE=spare2 is the same two rooms with a spare object in
# each, nobody's goal, both candidates for the partner's object (run_spare2.sh).  With swaps
# a rollout needs 170-190 GiB and about a day on 32 cores: too big for a grid.  SWAP=0 drops
# the swaps (718 object configurations instead of 5,545): about 24 GiB a rollout and roughly
# 8 times faster -- an estimate; time one first (SWAP=0 PAIR=... sh run_spare2.sh).  Give MEM
# with it: the GiB a rollout may use (--max-memory-gib), from which the workers default to
# 90 % of the machine's memory / MEM and the threads to the CPUs / workers:
#
#     SAME=1 TAUS="0.1 0.2 0.3" SCENE=spare2 SWAP=0 MEM=30 sh run_tau_grid.sh
#
# SWAP=0 changes the dynamics too, so the grid to compare it with is SCENE=two SWAP=0 (folders
# without _swap), not the 49's.
# Environment: TAUS1, TAUS2 (default TAUS, default "0.1 0.2 0.3 0.4 0.5"), SAME, GOALS (default
# all seven), THREADS per worker (default 2), WORKERS (default: the CPUs / THREADS), SWAP
# (default 1; 0: no swaps on a place, the folders then lack _swap), MEM (whole GiB a rollout may use;
# sets the default WORKERS and THREADS by the machine's memory), PHYS
# (Config.physical_likelihood), CONFLICT (Config.conflict: task, the default since 2026-11-01,
# in folders ..._taskconflict that take every pairing without conflict from the folder of
# before and roll out the conflict pairings alone; table, Table I's -w R_j), SCENE (default
# two), STEPS (default 60), NOVIDEO (default 1:
# records only -- 1225 x 3 videos is a lot; NOVIDEO=0 draws them), LOGS (default
# logs/tau_grid), OUT (default output/tau_grid), PYTHON (default python3).
set -e
PY=${PYTHON:-python3}
TAUS=${TAUS:-0.1 0.2 0.3 0.4 0.5}
TAUS1=${TAUS1:-$TAUS}
TAUS2=${TAUS2:-$TAUS}
GOALS=${GOALS:-cooperation conflict competition coercion exchange none hinder}
FLAGS="--levels 2 2 --objects inferred --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
[ "${SWAP:-1}" = 1 ] && FLAGS="$FLAGS --swap"      # SWAP=0: no swaps on a place (folders without _swap)
[ -n "$PHYS" ] && FLAGS="$FLAGS --physical-likelihood $PHYS"
[ -n "$CONFLICT" ] && FLAGS="$FLAGS --conflict $CONFLICT"
[ -n "$MEM" ] && FLAGS="$FLAGS --max-memory-gib $MEM"   # (not in the folder name: same model)
EXTRA="--scene ${SCENE:-two} --steps ${STEPS:-60}"
[ "${NOVIDEO:-1}" = 1 ] && EXTRA="$EXTRA --no-video"
N=$($PY -c "import os; print(os.cpu_count() or 2)")
if [ -n "$MEM" ]; then
  # a rollout may use MEM GiB: as many workers as 90 % of the memory holds, the CPUs shared
  RAM=$($PY -c "import os; print(int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30))")
  FIT=$(( RAM * 9 / 10 / MEM ))
  [ "$FIT" -lt 1 ] && FIT=1
  [ "$FIT" -gt "$N" ] && FIT=$N
  THREADS=${THREADS:-$(( N / FIT ))}
  [ "$THREADS" -lt 1 ] && THREADS=1
  W=$(( N / THREADS ))
  [ "$W" -gt "$FIT" ] && W=$FIT
  WORKERS=${WORKERS:-$W}
  echo "memory: $RAM GiB, $MEM GiB a rollout: at most $FIT at once"
else
  THREADS=${THREADS:-2}
  WORKERS=${WORKERS:-$(( N / THREADS ))}
fi
[ "$WORKERS" -lt 1 ] && WORKERS=1
LOGS=${LOGS:-logs/tau_grid}
mkdir -p "$LOGS"
# the pairs of taus, one "t1 t2" per line
pairs() {
  if [ "${SAME:-0}" = 1 ]; then
    for t in $TAUS1; do echo "$t $t"; done
  else
    for t1 in $TAUS1; do for t2 in $TAUS2; do echo "$t1 $t2"; done; done
  fi
}
np=$(pairs | wc -l); ng=$(echo $GOALS | wc -w)
if [ "${SAME:-0}" = 1 ]; then what="both robots at each tau of $TAUS1"
else what="agent_1 at $TAUS1, agent_2 at $TAUS2"; fi
echo "tau grid: $what; $np pairs of taus x $(( ng * ng )) pairings = $(( np * ng * ng )) rollouts; $WORKERS workers x $THREADS threads; logs in $LOGS"

# compile the kernels once (numba caches them) before the workers start together
$PY examples/run_object_inference.py $FLAGS --temperature 0.2 0.3 --scene tiny --steps 2 \
  --no-video --output "$LOGS/warmup" --threads "$THREADS" --pairing none none > "$LOGS/warmup.log" 2>&1

# one job per (agent_1's tau, agent_2's tau, agent_1's goal): its pairings with every goal of
# agent_2, one after the other; WORKERS jobs at a time, a new one as soon as one ends
JOBS="$LOGS/jobs.txt"
pairs | while read -r t1 t2; do
  for a in $GOALS; do echo "$t1 $t2 $a"; done
done > "$JOBS"
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
pairs > "$LOGS/pairs.txt"
while read -r t1 t2; do
  ARGS=""
  for a in $GOALS; do for b in $GOALS; do ARGS="$ARGS --pairing $a $b"; done; done
  log="$LOGS/table_${t1}_${t2}.log"
  if $PY examples/run_object_inference.py $FLAGS $EXTRA --temperature $t1 $t2 \
       --threads "$THREADS" --resume $ARGS > "$log" 2>&1 < /dev/null; then
    tail -n 1 "$log" >> "$LOGS/folders.txt"
  else
    echo "FAILED: the table of agent_1 at tau $t1, agent_2 at $t2 (see $log)"
  fi
done < "$LOGS/pairs.txt"
$PY tools/tau_grid.py --out "${OUT:-output/tau_grid}" $(cat "$LOGS/folders.txt")
