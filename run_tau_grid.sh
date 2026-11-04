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
#     SAME=1 TAUS="0.1 0.3 0.5" HAZARD=0.02 sh run_tau_grid.sh   # the partner's goal may change
#     SAME=1 TAUS="0.1" SCENES="near_miss head_on_corridor" GOALS1=hinder MEM=18 sh run_tau_grid.sh
#                                      # hinder vs every goal on two tier scenes
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
# all seven), GOALS1 and GOALS2 (agent_1's goals and agent_2's, default GOALS; since 2026-11-03),
# SCENES (one or more scenes, default SCENE; since 2026-11-03: a job per rollout, every scene
# and tau at once), THREADS per worker (default 2), WORKERS (default: the CPUs / THREADS), SWAP
# (default 1; 0: no swaps on a place, the folders then lack _swap), MEM (whole GiB a rollout may use;
# sets the default WORKERS and THREADS by the machine's memory), PHYS
# (Config.physical_likelihood), CONFLICT (Config.conflict: table, the default, Table I's
# -w R_j; task, an option since 2026-11-01, in folders ..._taskconflict; either folder takes
# every pairing without conflict from the other and rolls out the conflict pairings alone),
# HAZARD, SOCIAL_HAZARD, HAND_HAZARD (since 2026-11-02; Config.goal_hazard, social_hazard,
# hand_hazard, default 0: the partner's goal may change -- after every update the belief about
# it is mixed with the uniform, (1 - H) b + H u, Baker, Saxe & Tenenbaum 2009's model M2; HAND_HAZARD
# once at its pick-ups and places, before their evidence; folders ..._hazard<H>,
# ..._socialhazard<H>, ..._handhazard<H>),
# JOINT (1: Config.belief_factored=False, one joint posterior over the partner's (object,
# destination), the update until 2026-11-04; by default the posterior about its object and
# the one about its destination apart, folders ..._objgoal),
# INFTAU (since 2026-11-04; Config.inference_temperature, default none = Eq. 7's tau: the
# likelihood of the belief update at a tau of its own, INFTAU x c(move) -- 0.5 is Baker, Saxe
# & Tenenbaum 2009's fit to people; the robots still plan at tau; folders ..._inftau<T>),
# NODETAIL (1: Config.belief_detail=False -- the records without each step's "belief_update",
# the belief update's workings, every hypothesis model's Q among them; by default they hold it,
# some MB a rollout; tools/belief_step.py reads it),
# LAPSE (since 2026-11-04; Config.belief_lapse, default 0: a lapse in the likelihood -- each
# hypothesis model's policy read as (1 - E) psi + E / |A|, so one odd move costs a hypothesis
# at most a factor of about |A| / E; folders ..._lapse<E>),
# LAZY (since 2026-11-04; Config.lazy_mass: lazy hypotheses -- the lightest hypothesis models,
# up to that mass, left as last solved each transition; folders ..._lazy<MASS>), SCHEDULER (1:
# one queue over every scene, tools/schedule.py, with MEMS="scene=GiB ..." and THREADS_OF=
# "scene=N ..." per scene, MAX_THREADS (8) a rollout at most, NUMA=1 to bind each rollout to a
# NUMA node; run_testbed.sh's way since 2026-11-04),
# SCENE (default two), STEPS (default 60), NOVIDEO (default 1:
# records only -- 1225 x 3 videos is a lot; NOVIDEO=0 draws them), LOGS (default
# logs/tau_grid), OUT (default output/tau_grid; with more than one scene, OUT/<scene> for each),
# VERBOSE (default 0; 1: a line per transition in each rollout's log), PYTHON (default python3).
# Each process's thread pools are THREADS wide (NUMBA_NUM_THREADS, OMP_NUM_THREADS); a rollout
# that fails is run once more after the others.  The testbed of 2026-11-03 -- hinder against the seven goals on
# seven scenes, in two pools by memory -- is run_testbed.sh.
set -e
PY=${PYTHON:-python3}
TAUS=${TAUS:-0.1 0.2 0.3 0.4 0.5}
TAUS1=${TAUS1:-$TAUS}
TAUS2=${TAUS2:-$TAUS}
GOALS=${GOALS:-cooperation conflict competition coercion exchange none hinder}
GOALS1=${GOALS1:-$GOALS}          # agent_1's goals (2026-11-03; GOALS1="hinder": hinder vs every goal)
GOALS2=${GOALS2:-$GOALS}          # agent_2's
SCENES=${SCENES:-${SCENE:-two}}   # one or more scenes (2026-11-03)
FLAGS="--levels 2 2 --objects inferred --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2"
[ "${SWAP:-1}" = 1 ] && FLAGS="$FLAGS --swap"      # SWAP=0: no swaps on a place (folders without _swap)
[ -n "$PHYS" ] && FLAGS="$FLAGS --physical-likelihood $PHYS"
[ -n "$CONFLICT" ] && FLAGS="$FLAGS --conflict $CONFLICT"
[ -n "$HAZARD" ] && FLAGS="$FLAGS --goal-hazard $HAZARD"          # the partner's goal may change (folders _hazard<H>)
[ -n "$SOCIAL_HAZARD" ] && FLAGS="$FLAGS --social-hazard $SOCIAL_HAZARD"
[ -n "$HAND_HAZARD" ] && FLAGS="$FLAGS --hand-hazard $HAND_HAZARD"
[ -n "$INFTAU" ] && FLAGS="$FLAGS --inference-temperature $INFTAU"  # the likelihood's own tau (2026-11-04; folders _inftau<T>)
[ -n "$LAPSE" ] && FLAGS="$FLAGS --lapse $LAPSE"  # a lapse in the likelihood (2026-11-04; folders _lapse<E>)
[ -n "$LAZY" ] && FLAGS="$FLAGS --lazy $LAZY"     # lazy hypotheses (2026-11-04; folders _lazy<MASS>)
[ "${JOINT:-0}" = 1 ] && FLAGS="$FLAGS --joint-posterior"   # one joint (object, destination) posterior, the rule until 2026-11-04
[ -n "$MEM" ] && FLAGS="$FLAGS --max-memory-gib $MEM"   # (not in the folder name: same model)
[ "${NODETAIL:-0}" = 1 ] && FLAGS="$FLAGS --no-belief-detail"   # records without the belief update's workings
EXTRA="--steps ${STEPS:-60}"
[ "${NOVIDEO:-1}" = 1 ] && EXTRA="$EXTRA --no-video"
[ "${VERBOSE:-0}" = 1 ] && EXTRA="$EXTRA --verbose"   # a line per transition in each rollout's log
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
# each process's thread pools at its share of the CPUs (2026-11-03): numba's OpenMP pool is
# made with NUMBA_NUM_THREADS threads -- by default one per CPU, whatever set_num_threads
# asks -- and 28 processes of 64 threads ran into the per-user limit ("libgomp: Thread
# creation failed: Resource temporarily unavailable")
export NUMBA_NUM_THREADS=$THREADS OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
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
np=$(pairs | wc -l); n1=$(echo $GOALS1 | wc -w); n2=$(echo $GOALS2 | wc -w); ns=$(echo $SCENES | wc -w)
if [ "${SAME:-0}" = 1 ]; then what="both robots at each tau of $TAUS1"
else what="agent_1 at $TAUS1, agent_2 at $TAUS2"; fi
if [ "${SCHEDULER:-0}" = 1 ]; then how="one queue, as many at once as memory and CPUs allow (tools/schedule.py)"
else how="$WORKERS workers x $THREADS threads"; fi
echo "tau grid: $SCENES; $what; $ns scene(s) x $np pairs of taus x $(( n1 * n2 )) pairings = $(( ns * np * n1 * n2 )) rollouts; $how; logs in $LOGS"

# compile the kernels once (numba caches them) before the workers start together
$PY examples/run_object_inference.py $FLAGS --temperature 0.2 0.3 --scene tiny --steps 2 \
  --no-video --output "$LOGS/warmup" --threads "$THREADS" --pairing none none > "$LOGS/warmup.log" 2>&1

# one job per rollout (2026-11-03): (scene, agent_1's tau, agent_2's tau, agent_1's goal,
# agent_2's goal), the pairings outermost so that every scene and every tau start at once;
# WORKERS jobs at a time, a new one as soon as one ends
JOBS="$LOGS/jobs.txt"
pairs > "$LOGS/pairs.txt"
for a in $GOALS1; do for b in $GOALS2; do for sc in $SCENES; do
  while read -r t1 t2; do echo "$sc $t1 $t2 $a $b"; done < "$LOGS/pairs.txt"
done; done; done > "$JOBS"
export PY FLAGS EXTRA THREADS LOGS
run_jobs() {      # the rollouts of the jobs file $1, WORKERS at a time; the failed ones to failed.txt
  xargs -P "$WORKERS" -L 1 sh -c '
    sc=$0; t1=$1; t2=$2; a=$3; b=$4
    log="$LOGS/run_${sc}_${t1}_${t2}_${a}_$b.log"
    if $PY examples/run_object_inference.py $FLAGS $EXTRA --scene $sc --temperature $t1 $t2 \
         --threads "$THREADS" --resume --pairing $a $b > "$log" 2>&1 < /dev/null; then
      echo "done: $sc, agent_1 at tau $t1, agent_2 at $t2, $a / $b"
    else
      echo "FAILED: $sc, agent_1 at tau $t1, agent_2 at $t2, $a / $b (see $log)"
      echo "$sc $t1 $t2 $a $b" >> "$LOGS/failed.txt"
    fi
  ' < "$1"
}
: > "$LOGS/failed.txt"
if [ "${SCHEDULER:-0}" = 1 ]; then
  # one queue for every scene (2026-11-04, tools/schedule.py): a rollout starts whenever its
  # memory (MEM, or MEMS="scene=GiB ..." per scene) and threads (THREADS, or THREADS_OF=
  # "scene=N ...") fit, the big ones first, the free CPUs shared out (up to MAX_THREADS a
  # rollout); what fails is retried once by the scheduler itself
  MEMARGS="--mem ${MEM:-2.5}"
  for kv in $MEMS; do MEMARGS="$MEMARGS --mem $kv"; done
  THRARGS="--threads $THREADS"
  for kv in $THREADS_OF; do THRARGS="$THRARGS --threads $kv"; done
  $PY tools/schedule.py "$JOBS" --logs "$LOGS" --flags "$FLAGS" --extra "$EXTRA" $MEMARGS $THRARGS \
    --max-threads "${MAX_THREADS:-8}" --python "$PY" || true
else
  run_jobs "$JOBS"
  # once more, in parallel, the rollouts that failed (a squeeze of threads or memory as many
  # start at once); what fails again is left to the table pass below, which rolls out what is
  # missing one at a time
  if [ -s "$LOGS/failed.txt" ]; then
    mv "$LOGS/failed.txt" "$LOGS/retry.txt"
    : > "$LOGS/failed.txt"
    echo "retrying $(wc -l < "$LOGS/retry.txt") failed rollout(s)"
    run_jobs "$LOGS/retry.txt"
  fi
fi

# per scene and pair of taus, one pass over its saved rollouts (nothing is rolled out again
# where every one was saved): the folder's results.md, results.png and summary.json; then
# tools/tau_grid.py per scene, in OUT (OUT/<scene> with more than one scene)
for sc in $SCENES; do
  : > "$LOGS/folders_$sc.txt"
  while read -r t1 t2; do
    ARGS=""
    for a in $GOALS1; do for b in $GOALS2; do ARGS="$ARGS --pairing $a $b"; done; done
    log="$LOGS/table_${sc}_${t1}_${t2}.log"
    if $PY examples/run_object_inference.py $FLAGS $EXTRA --scene $sc --temperature $t1 $t2 \
         --threads "$THREADS" --resume $ARGS > "$log" 2>&1 < /dev/null; then
      tail -n 1 "$log" >> "$LOGS/folders_$sc.txt"
    else
      echo "FAILED: the table of $sc, agent_1 at tau $t1, agent_2 at $t2 (see $log)"
    fi
  done < "$LOGS/pairs.txt"
  if [ "$ns" = 1 ]; then out=${OUT:-output/tau_grid}; else out=${OUT:-output/tau_grid}/$sc; fi
  if [ -s "$LOGS/folders_$sc.txt" ]; then
    $PY tools/tau_grid.py --out "$out" $(cat "$LOGS/folders_$sc.txt")
  fi
done
