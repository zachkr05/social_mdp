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
# 7 scenes x 7 pairings x len(TAUS) rollouts (4 taus: 196).
#
# Memory and time, at the settings above (swaps on).  A tier scene: 480,557 joint states, 15.6
# GiB of solver arrays (MEM_HOUSE=18 leaves room for the rest).  The gate scene: about 1.1-1.3
# million joint states, its dynamics tables alone 20 GiB -- some 60-70 GiB a rollout, an
# estimate (MEM_GATE=72).  Time: the first house transition took 602 s on 2 threads on
# cheetah05 (2026-11-03), so a house rollout that runs to the 60-step cap takes the better
# part of a day at that pace -- time one first (below).
#
# One queue for all of it (2026-11-04, tools/schedule.py; the user: "parallelize the S-MDP so
# it runs substantially faster"): a rollout starts whenever its memory and threads fit, the
# gate's big ones first with the free CPUs shared out among them (up to MAX_THREADS each),
# the house's small ones filling in and taking over as the gate's finish -- so that the whole
# machine stays busy to the end.  (Until today two fixed pools: the gate's three workers would
# have run its 28 rollouts for days after the house pool was done.)
#
#     sh run_testbed.sh                                          # all of it
#     nohup sh run_testbed.sh > logs/testbed.out 2>&1 &          # ... in the background
#     TAUS="0.1" GOALS2="cooperation" HOUSE="" sh run_testbed.sh   # time one gate rollout
#     TAUS="0.1" GOALS2="cooperation" GATE="" HOUSE="near_miss" sh run_testbed.sh   # one tier one
#     GATE="gate_few" MEM_GATE=8 sh run_testbed.sh               # the gate with four hand-off cells
#     HAZARD=0.02 sh run_testbed.sh                              # the partner's goal may change
#     INFTAU=0.5 HAZARD=0.02 sh run_testbed.sh                   # ... and a softer likelihood
#     LAPSE=0.05 HAZARD=0.02 sh run_testbed.sh                   # ... or a lapse in it
#     LPOW=0.25 HAZARD=0.02 sh run_testbed.sh                    # ... or a tempered update
#     LAZY=0.02 sh run_testbed.sh                                # lazy hypotheses (folders _lazy0.02)
#
# Interrupted? Run the same line again: --resume keeps every rollout already saved.  Folders:
# output/object_inference_<scene>_w1_1_..._tau<t>_... (one per scene and tau, as
# run_tau_grid.sh's); progress in logs/testbed.out (one line per rollout started and done)
# and logs/testbed/run_<scene>_<t>_<t>_hinder_<goal>.log (a line per transition with its
# time: tail -1 ...); at the end tools/tau_grid.py's figures per scene in output/testbed/<scene>/.
# Environment: TAUS (default "0.1 0.15 0.2 0.25"), GOALS1 (default hinder), GOALS2 (default all
# seven), GATE (default gate; "" for none), HOUSE (default the six tier scenes; "" for none),
# MEM_GATE (default 72), MEM_HOUSE (default 18), GATE_THREADS (default 4: a gate rollout's
# threads at least), HOUSE_THREADS (default 2), MAX_THREADS (default 8: a rollout's threads at
# most, when CPUs are free), NUMA (1: bind each rollout to a NUMA node, needs numactl), VERBOSE
# (default 1), and everything run_tau_grid.sh takes (SWAP, STEPS, NOVIDEO, HAZARD, INFTAU, LAPSE,
# LPOW, LAZY, ...),
# PYTHON (default python3).
set -e
export SAME=1
export TAUS=${TAUS:-0.1 0.15 0.2 0.25}
export GOALS1=${GOALS1:-hinder}
export GOALS2=${GOALS2:-cooperation conflict competition coercion exchange none hinder}
export VERBOSE=${VERBOSE:-1}      # a line per transition in each rollout's log: how far it is
GATE=${GATE-gate}
HOUSE=${HOUSE-near_miss same_room_later hall_crossing doorway_passing head_on_blocked_goal head_on_corridor}
MEM_GATE=${MEM_GATE:-72}
MEM_HOUSE=${MEM_HOUSE:-18}
GATE_THREADS=${GATE_THREADS:-4}
HOUSE_THREADS=${HOUSE_THREADS:-2}
SCENES="$GATE $HOUSE"
[ -z "$(echo $SCENES)" ] && { echo "run_testbed.sh: no scenes (GATE and HOUSE both empty)"; exit 1; }
MEMS=""; THREADS_OF=""
for sc in $GATE; do MEMS="$MEMS $sc=$MEM_GATE"; THREADS_OF="$THREADS_OF $sc=$GATE_THREADS"; done
mkdir -p logs
echo "testbed: $GOALS1 vs $GOALS2 at tau $TAUS on $SCENES"
echo "  gate scenes at $MEM_GATE GiB and $GATE_THREADS+ threads a rollout, house scenes at $MEM_HOUSE GiB and $HOUSE_THREADS+; at most ${MAX_THREADS:-8} threads each; logs/testbed/"
OUT=output/testbed
[ "$(echo $SCENES | wc -w)" = 1 ] && OUT=output/testbed/$(echo $SCENES)   # one scene: its own folder
SCHEDULER=1 SCENES="$SCENES" MEM=$MEM_HOUSE MEMS="$MEMS" THREADS=$HOUSE_THREADS THREADS_OF="$THREADS_OF" \
  LOGS=logs/testbed OUT=$OUT sh run_tau_grid.sh
echo "testbed: done -- figures in output/testbed/<scene>/"
