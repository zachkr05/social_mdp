#!/bin/sh
# The testbed (2026-11-03; the user: "So we will run the tesbed on 7 scenes.  Lets also make it
# such that we only run hinder vs all 7 social settings"): hinder (agent_1) against each of the
# seven social goals (agent_2) on seven scenes, both robots at each tau of TAUS (one tau for
# both), at the 49's settings (run_tau_grid.sh: both robots at level 2, everything inferred,
# swaps and taking, a pick-up of three transitions at two moves' worth, carry 1, inertia 1):
#
#   gate                  the gate scene: two rooms, a one-cell gate and a long way round; an
#                         object may be put down on any cell it may lie on (scenes.gate_scene)
#   near_miss             low tier: close, but no contact        } the meeting suite's six fair
#   same_room_later       low tier                               } tier scenes on the 0.5 m
#   hall_crossing         medium: they meet, room to pass        } house (examples/
#   doorway_passing       medium                                 } tier_scenes.py)
#   head_on_blocked_goal  high: someone must give way            }
#   head_on_corridor      high                                   }
#
# 7 scenes x 7 pairings x len(TAUS) rollouts (3 taus: 147).
#
# Memory and time, at the settings above (swaps on).  A tier scene: 480,557 joint states, 15.6
# GiB of solver arrays (MEM_HOUSE=18 leaves room for the rest).  The gate scene: about 1.1-1.3
# million joint states, its dynamics tables alone 20 GiB -- some 60-70 GiB a rollout, an
# estimate (a machine of 7 GiB cannot build it; MEM_GATE=72).  Time, scaled from the two-room
# scene (60,368 states, 10-40 min a rollout on 2 threads): a tier scene some 1.5-5 h on 2
# threads, the gate some 2-8 h on 4 -- estimates; time one first (below).
#
# Two pools of workers at once, so that the gate's few big rollouts and the house's many
# small ones share the machine: the gate scene gets GATE_WORKERS (default: a third of 90 % of
# the memory, in rollouts of MEM_GATE GiB) at GATE_THREADS (default 4) threads each, the tier
# scenes the rest of the memory in rollouts of MEM_HOUSE GiB and the rest of the CPUs.
#
#     sh run_testbed.sh                                          # all of it
#     nohup sh run_testbed.sh > logs/testbed.out 2>&1 &          # ... in the background
#     TAUS="0.1" GOALS2="cooperation" HOUSE="" sh run_testbed.sh   # time one gate rollout
#     TAUS="0.1" GOALS2="cooperation" GATE="" HOUSE="near_miss" sh run_testbed.sh   # one tier one
#     GATE="gate_few" sh run_testbed.sh                          # the gate with four hand-off cells (6 GiB)
#
# Interrupted? Run the same line again: --resume keeps every rollout already saved.  Folders:
# output/object_inference_<scene>_w1_1_..._tau<t>_... (one per scene and tau, as
# run_tau_grid.sh's); progress in logs/testbed_gate/ and logs/testbed_house/ (a log per
# rollout); at the end tools/tau_grid.py's figures per scene in output/testbed/<scene>/.
# Environment: TAUS (default "0.1 0.15 0.2"), GOALS1 (default hinder), GOALS2 (default all
# seven), GATE (default gate; "" for none), HOUSE (default the six tier scenes; "" for none),
# MEM_GATE (default 72), MEM_HOUSE (default 18), GATE_WORKERS, GATE_THREADS (default 4),
# HOUSE_WORKERS, HOUSE_THREADS, and everything run_tau_grid.sh takes (SWAP, STEPS, NOVIDEO,
# HAZARD, ...), PYTHON (default python3).
set -e
PY=${PYTHON:-python3}
export SAME=1
export TAUS=${TAUS:-0.1 0.15 0.2}
export GOALS1=${GOALS1:-hinder}
export GOALS2=${GOALS2:-cooperation conflict competition coercion exchange none hinder}
GATE=${GATE-gate}
HOUSE=${HOUSE-near_miss same_room_later hall_crossing doorway_passing head_on_blocked_goal head_on_corridor}
MEM_GATE=${MEM_GATE:-72}
MEM_HOUSE=${MEM_HOUSE:-18}
N=$($PY -c "import os; print(os.cpu_count() or 2)")
RAM=$($PY -c "import os; print(int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30))")
USE=$(( RAM * 9 / 10 ))
GW=0; GT=0
if [ -n "$GATE" ]; then
  GW=${GATE_WORKERS:-$(( USE / 3 / MEM_GATE ))}
  [ "$GW" -lt 1 ] && GW=1
  GT=${GATE_THREADS:-4}
fi
if [ -n "$HOUSE" ]; then
  HW=$(( (USE - GW * MEM_GATE) / MEM_HOUSE ))
  [ "$HW" -lt 1 ] && HW=1
  HT=$(( (N - GW * GT) / HW ))
  [ "$HT" -lt 1 ] && HT=1
  HW=${HOUSE_WORKERS:-$HW}
  HT=${HOUSE_THREADS:-$HT}
fi
mkdir -p logs
echo "testbed: hinder ($GOALS1) vs $GOALS2 at tau $TAUS; memory $RAM GiB, $N CPUs"
[ -n "$GATE" ] && echo "  gate pool: $GATE -- $GW workers x $GT threads, $MEM_GATE GiB each; logs/testbed_gate.out"
[ -n "$HOUSE" ] && echo "  house pool: $HOUSE -- $HW workers x $HT threads, $MEM_HOUSE GiB each; logs/testbed_house.out"
case "$GATE" in
  *" "*) GOUT=output/testbed ;;               # several gate scenes: one folder each, below
  *) GOUT=output/testbed/$GATE ;;
esac
if [ -n "$GATE" ]; then
  SCENES="$GATE" MEM=$MEM_GATE WORKERS=$GW THREADS=$GT LOGS=logs/testbed_gate OUT=$GOUT \
    sh run_tau_grid.sh > logs/testbed_gate.out 2>&1 &
fi
if [ -n "$HOUSE" ]; then
  case "$HOUSE" in
    *" "*) HOUT=output/testbed ;;
    *) HOUT=output/testbed/$HOUSE ;;
  esac
  SCENES="$HOUSE" MEM=$MEM_HOUSE WORKERS=$HW THREADS=$HT LOGS=logs/testbed_house OUT=$HOUT \
    sh run_tau_grid.sh > logs/testbed_house.out 2>&1 &
fi
wait
echo "testbed: done -- figures in output/testbed/, the pools' logs in logs/testbed_gate.out and logs/testbed_house.out"
