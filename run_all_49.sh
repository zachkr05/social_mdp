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

# one process, the 49 one after the other (~17 min a pairing on 2 cores; 12 h or so)
$PY examples/run_object_inference.py $FLAGS $EXTRA --threads ${THREADS:-0} --resume \
  --pairing cooperation cooperation \
  --pairing cooperation conflict \
  --pairing cooperation competition \
  --pairing cooperation coercion \
  --pairing cooperation exchange \
  --pairing cooperation none \
  --pairing cooperation hinder \
  --pairing conflict cooperation \
  --pairing conflict conflict \
  --pairing conflict competition \
  --pairing conflict coercion \
  --pairing conflict exchange \
  --pairing conflict none \
  --pairing conflict hinder \
  --pairing competition cooperation \
  --pairing competition conflict \
  --pairing competition competition \
  --pairing competition coercion \
  --pairing competition exchange \
  --pairing competition none \
  --pairing competition hinder \
  --pairing coercion cooperation \
  --pairing coercion conflict \
  --pairing coercion competition \
  --pairing coercion coercion \
  --pairing coercion exchange \
  --pairing coercion none \
  --pairing coercion hinder \
  --pairing exchange cooperation \
  --pairing exchange conflict \
  --pairing exchange competition \
  --pairing exchange coercion \
  --pairing exchange exchange \
  --pairing exchange none \
  --pairing exchange hinder \
  --pairing none cooperation \
  --pairing none conflict \
  --pairing none competition \
  --pairing none coercion \
  --pairing none exchange \
  --pairing none none \
  --pairing none hinder \
  --pairing hinder cooperation \
  --pairing hinder conflict \
  --pairing hinder competition \
  --pairing hinder coercion \
  --pairing hinder exchange \
  --pairing hinder none \
  --pairing hinder hinder
