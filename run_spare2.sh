#!/bin/sh
# One pairing on the two-room scene with two spare objects, one in each room (--scene spare2:
# the 49's scene as it is, plus spares on (2, 1) and (6, 5)), at the 49's settings: both
# robots at level 2, every object inferred (4 objects x 6 candidates = 24 physical goals per
# robot), swaps and taking, the pick-up of three transitions costing two moves' worth on the
# first, carry 1, inertia 1, Eq. 7's tau (TAU, default 0.15).  agent_1 is the first goal of
# PAIR, agent_2 the second.
#
# Size: 5,545 object configurations, 4.35 million joint states; about 170-190 GiB of solver
# memory (23 GiB of it the dynamics tables) and about a day on 32 cores (Time, below).  If the machine
# has less, the solver stops within minutes with a message saying how much it needs.
#
# Run from the folder holding social_pairings/ and examples/, after
#     pip install -r requirements.txt
# Interrupted? Run the same line again: --resume keeps a pairing already saved.
# Time: a transition of the smaller variant without swaps and with an instant pick-up (126
# configurations) took 390-420 s on 2 cores here; this model has 44 times its states, so
# about 4-5 h a transition on 2 cores -- some 20-30 min on 32 cores, a day for 60 transitions
# (the first transition, solved cold, about twice as long).  No checkpoint inside a rollout:
# run it under tmux or nohup.  STEPS=30 halves it; SWAP=0 drops the swaps (718
# configurations, ~24 GiB, about 8 times faster).
# Environment: PAIR (default "hinder coercion"), TAU (default 0.15), MEM (GiB the solver may
# use; default 90 % of the machine's memory), THREADS (default 0 = every CPU), STEPS
# (default 60), SWAP (default 1; 0 = no swaps, the folder then lacks _swap), PHYS
# (Config.physical_likelihood: level1 default, level0 the authors' estimator), PYTHON
# (default python3).
set -e
PY=${PYTHON:-python3}
PAIR=${PAIR:-hinder coercion}
MACHINE=$($PY -c "import os; print(int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30))")
MEM=${MEM:-$(( MACHINE * 9 / 10 ))}
NEED=175; [ "${SWAP:-1}" = 1 ] || NEED=25
echo "machine: $MACHINE GiB; the solver may use $MEM GiB (the model needs about $NEED or more)"
[ "$MEM" -lt "$NEED" ] && echo "warning: probably too little memory; the solver will say so within minutes"
FLAGS="--scene spare2 --steps ${STEPS:-60} --levels 2 2 --objects inferred --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2 --temperature ${TAU:-0.15}"
[ "${SWAP:-1}" = 1 ] && FLAGS="$FLAGS --swap"   # SWAP=0: no swaps (718 configurations, ~24 GiB, ~8x faster)
[ -n "$PHYS" ] && FLAGS="$FLAGS --physical-likelihood $PHYS"
mkdir -p logs
LOG=logs/spare2_$(echo "$PAIR" | tr ' ' '_').log
echo "progress: $LOG (a line per transition)"
$PY examples/run_object_inference.py $FLAGS --pairing $PAIR --threads "${THREADS:-0}" \
  --max-memory-gib "$MEM" --verbose --resume 2>&1 | tee "$LOG"
