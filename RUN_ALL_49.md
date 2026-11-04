# All 49 pairings: the belief update of Eqs. 5-6 as the authors' solver keeps them (2026-10-30), nothing automatic, a pick-up of three transitions, two posteriors by the hands, five hand actions, the hinderer without a goal of its own, inertia

## The testbed: hinder against every goal on seven scenes (2026-11-03)

`run_testbed.sh`: agent_1 a hinderer, agent_2 each of the seven social goals, both at level 2
and at the same tau (each of `TAUS`, default 0.1, 0.15, 0.2), at the 49's settings, on

| scene | what it is | joint states | memory a rollout |
|---|---|---|---|
| `gate` | two 3 x 4 rooms, a one-cell gate and a long way round; an object may be put down on any cell it may lie on | about 1.1-1.3 million | some 60-70 GiB (an estimate) |
| `near_miss`, `same_room_later` | tier low: close, but no contact | 480,557 | 15.6 GiB |
| `hall_crossing`, `doorway_passing` | tier medium: they meet, with room to pass | 480,557 | 15.6 GiB |
| `head_on_blocked_goal`, `head_on_corridor` | tier high: someone must give way | 480,557 | 15.6 GiB |

The six are the meeting suite of `examples/tier_scenes.py`, the fair placements on the 0.5 m
house (79 free cells, six candidate destinations, no hand-off cells). 7 x 7 x 4 = 196 rollouts
(`TAUS` default 0.1, 0.15, 0.2, 0.25 since 2026-11-03). On cheetah05 on 2026-11-03 (25 house
rollouts and 3 gate ones at once) the first house transition took 602 s on 2 threads.

**One queue, since 2026-11-04** (`tools/schedule.py`; the user: "parallelize the S-MDP so it
runs substantially faster"). Every rollout is its own job; a job starts whenever its memory
(`MEM_GATE` = 72 GiB for the gate scene, `MEM_HOUSE` = 18 for the house) and threads fit, the
gate's big ones first with the free CPUs shared out among them (`GATE_THREADS` = 4 at least,
`MAX_THREADS` = 8 at most), the house's small ones (`HOUSE_THREADS` = 2) filling in and taking
over as the gate's finish -- the whole machine busy to the end, where the two fixed pools of
2026-11-03 left the gate's three workers running alone for days after the house was done.
A failed rollout is retried once; `logs/testbed.out` has a line per
rollout started and done, `logs/testbed/failed.txt` what failed twice.

    TAUS="0.1" GOALS2="cooperation" HOUSE="" sh run_testbed.sh                  # one gate rollout
    TAUS="0.1" GOALS2="cooperation" GATE="" HOUSE="near_miss" sh run_testbed.sh  # one tier one
    nohup sh run_testbed.sh > logs/testbed.out 2>&1 &                            # all 196

`GATE="gate_few" MEM_GATE=8` is the gate with four hand-off cells (6.2 GiB, about a tenth of
the states); `HAZARD=0.02`, `LAZY=0.02` and the rest of `run_tau_grid.sh`'s options carry
through. Figures per scene in `output/testbed/<scene>/`. The tier scenes are asymmetric, so
hinder as agent_2 is another run: `GOALS1="cooperation conflict competition coercion exchange
none hinder" GOALS2="hinder"`.

## Where a transition's time goes, and lazy hypotheses (option, 2026-11-04)

Profiled on the two-room scene (60,368 joint states, hinder / cooperation, both at level 2,
tau 0.1, 2 threads): 40-45 s a transition in the steady state, of which the 168 level-1
hypothesis models (12 physical goals x 7 social goals per robot) take 90 % and the four
level-2 models (roots and twins) 5 %. Each hypothesis model is re-solved every transition
because the robot's beliefs move its inputs -- the partner it predicts (Eq. 4) and the social
term, whose rate changes with the belief and whose value changes by $1/(1-\gamma)$ = 200
times that: the warm start is off by 10-70 in value for the race and coercion hypotheses --
and a solve then takes some 20 policy-iteration steps and 130 Krylov iterations per model
(summed over its four stages), the Krylov iterations 90 % of it. Tried on that solve, and
found not to help: a looser certificate (`value_tol` 1e-2 instead of 1e-4: 7 % faster),
another forcing (`pi_forcing` 0.01-0.3: no faster), single-precision Krylov vectors (no
change), a least-squares constant shift of the warm start per work item (no change), and
Gauss-Seidel or Jacobi sweeps in place of the Krylov evaluation (slower). What is left on the
CPU is to solve fewer models, or to spread the rollouts better over the machine (the
scheduler above).

`LAZY=m` (`Config.lazy_mass`, `--lazy m`, folders `..._lazy<m>`; off by default): of each
level-2 robot's hypothesis models about its partner, in order of the mass their hypotheses
carry in its two posteriors, $b_g(g)\,b_\xi(\xi)$, the lightest ones whose masses add up to at
most $m$ keep the values and policies of their last solve instead of being re-solved; every
model is solved at the first transition and at least every `lazy_period` (5) transitions. The
skipped models move the robot's prediction of its partner by at most $m$ of its mass; the
likelihood of their hypotheses (Eqs. 5-6) is read off policies a few transitions stale. What
was skipped is in the record (`trace[t]["lazy"]`: per robot, models skipped, their mass, models
solved). Measured on the two-room scene (hinder / cooperation, both at
level 2, tau 0.1, 16 transitions, 2 threads): 350 s against 691 s exact, the same 16 joint
actions, but not the same beliefs -- from transition 9 agent_1 reads its partner as "none" 0.91
where the exact run reads "conflict" 1.00. Not safe as it is; off by default.

## A softer likelihood: `INFTAU`, `LAPSE` (options, 2026-11-04)

On the testbed's hinder / cooperation at `hall_crossing` (tau 0.1, hazard 0.02) the hinderer's
belief about red's destination swung on single moves (G3 0.96 -> 0.003 on one step left at
step 38, G2 on top at 39; G5 0.97 at 51 with red in front of G1): at tau 0.1 one move moves two
hypotheses apart by $e^{\Delta Q/\tau}$, $\tau = 0.05$ reward units, and the level-1 models a
level-2 robot reads its partner by misread a partner that plans at level 2 (README, "A softer
likelihood"). Two options, off by default (Eqs. 5-6 as printed), in `run_tau_grid.sh` and
`run_testbed.sh`:

    INFTAU=0.5 HAZARD=0.02 sh run_testbed.sh     # the likelihood at tau_inf = 0.5 c(move); folders ..._inftau0.5_hazard0.02
    LAPSE=0.05 HAZARD=0.02 sh run_testbed.sh     # a lapse of 0.05 in it; folders ..._lapse0.05_hazard0.02

`INFTAU` (`--inference-temperature`, `Config.inference_temperature`): the hypothesis models'
softmax at $\tau_{inf}$ for the belief update only (the robots plan and act at Eq. 7's tau);
0.5 is Baker, Saxe & Tenenbaum's fit to people, $\eta = \tau/\tau_{inf} = 0.2$ at tau 0.1.
`LAPSE` (`--lapse`, `Config.belief_lapse`): $(1-\varepsilon)\psi + \varepsilon/|A|$. Not run on
the testbed yet; tested on the tiny scene against the reference update (`test_beliefs.py`).

## The object's posterior and the destination's apart (default since 2026-11-04)

Each robot's belief about its partner's physical goal is now two posteriors: one about the
object it wants, learnt while its hands are empty (the destination's frozen), and one about
its destination, learnt while it holds an object (the object's frozen); the robot plans with
their product. Until now one joint posterior believed the destination under each object
apart, so that a destination only the less likely object explained could not win (README,
"The object's posterior and the destination's apart"). Folders `..._eq56_objgoal`; `JOINT=1`
gives the update of before and its folders.

## The partner's goal may change: a hazard (options, 2026-11-02)

Off by default. `HAZARD=h` mixes each robot's belief about its partner's object or
destination with the uniform after every update, $(1-h)\,b + h\,u$ (Baker, Saxe & Tenenbaum
2009, M2): no hypothesis falls below $h$ / their number, so a belief crushed by a few
surprising moves comes back in a few, and one that nothing supports fades. `SOCIAL_HAZARD=h`
does the same for the social goal; `HAND_HAZARD=h` mixes, once, the belief about the partner's
object before its pick-up and the one about its destination before its place (1: forgotten).
Folders `..._hazard<h>`, `..._socialhazard<h>`, `..._handhazard<h>`; README, "The partner's
goal may change".

    SAME=1 TAUS="0.1 0.3 0.5" HAZARD=0.02 sh run_tau_grid.sh
    SAME=1 TAUS="0.1 0.3 0.5" SCENE=spare2 SWAP=0 MEM=30 HAZARD=0.02 sh run_tau_grid.sh   # with the decoys

## Task conflict: an option (2026-11-01, evening)

Conflict is Table I's, $-w\,\tilde R_j$ above level 1: the default, as before. The task
conflict pays a level-2 conflict robot $-w\,r(\tilde g_j)$ against every partner,
$R_i = r(g_i) - w\,r(\tilde g_j) - c(a_i)$ (`CONFLICT=task` / `--conflict task`; README,
"Task conflict").
It was the default for a few hours that evening; the user then asked for conflict as it was.
Only the 13 pairings with conflict differ between the two, so `--resume` takes the other 36
records (and their videos) from the folder of the same options under the other reading --
`..._taskconflict` for the task conflict, the name without the suffix for the table's --
where they exist, and rolls out the conflict pairings alone.

    CONFLICT=task sh run_all_49_parallel.sh    # the 49 at tau 0.2 with the task conflict: 13 rolled out, 36 taken
    sh run_all_49_parallel.sh                  # Table I's conflict (the default): the folders of before

A folder that is missing a record, or that was made with other options, simply gives
nothing to take: those pairings are rolled out. With `OUTPUT=...` nothing is taken.

### Two decoy objects in the grid

`SCENE=spare2` is the two rooms plus a spare object in each, nobody's goal, both candidates
for the partner's object. With swaps (the 49's settings) a rollout needs 170-190 GiB and about
a day on 32 cores, so a grid is out of reach. Without swaps (`SWAP=0`, 718 object
configurations instead of 5,545) it needs about 24 GiB and should be roughly 8 times faster;
that is an estimate, so time one rollout first:

    SWAP=0 TAU=0.2 PAIR="conflict cooperation" MEM=30 sh run_spare2.sh     # one rollout, a line per transition with its time
    SAME=1 TAUS="0.1 0.2 0.3" SCENE=spare2 SWAP=0 MEM=30 sh run_tau_grid.sh   # the 49 at three taus
    SAME=1 TAUS="0.1 0.2 0.3" SCENE=two SWAP=0 sh run_tau_grid.sh             # ... and the same without the decoys

`MEM` (whole GiB a rollout may use) sets the workers to 90 % of the machine's memory / `MEM`
and the threads to the CPUs / workers. Without it the script starts a worker per two CPUs,
sized for the 2.5 GB rollouts of the two-room scene, and with the decoys that runs out of
memory. `SWAP=0` changes the dynamics as well, so compare the decoy grid with the
`SCENE=two SWAP=0` grid, not with the 49's (whose folders have `_swap`). The record of the
`run_spare2.sh` line is the grid's at that tau, kept by `--resume`.

## Each robot its own tau: `run_tau_grid.sh` (2026-11-01)

Eq. 7's tau is now one per robot: robot i reads its partner at its own tau -- the softmax it
predicts the partner by and the likelihood of its belief update -- and the robots know each
other's (README, "A tau per robot"). One run, agent_1 at 0.2 and agent_2 at 0.5:

    python examples/run_object_inference.py --scene two --levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2 --temperature 0.2 0.5 --resume --pairing competition coercion
    TAU="0.2 0.5" sh run_all_49_parallel.sh     # the 49 at that pair of taus

The 49 at every pair of taus, and the figures that put them side by side:

    sh run_tau_grid.sh                                    # 0.1 ... 0.5 for each robot: 25 x 49 = 1225 rollouts
    TAUS="0.1 0.2 0.3" sh run_tau_grid.sh                 # 9 x 49
    TAUS1="0.2" TAUS2="0.1 0.3 0.5" sh run_tau_grid.sh    # agent_1 fixed, agent_2 swept
    GOALS="competition coercion" sh run_tau_grid.sh       # those goals' 4 pairings only

0.1 to 0.6 in steps of 0.05 (eleven taus):

    T="0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6"
    SAME=1 TAUS="$T" sh run_tau_grid.sh    # both robots at the same tau: 11 x 49 = 539 rollouts, ~5 h on 64 cores
    TAUS="$T" sh run_tau_grid.sh           # every pair, each robot its own: 121 x 49 = 5929, ~2.5 days on 64 cores

With `SAME=1` the figures have the tau as one axis (each pairing a row of cells, the counts a
line per robot).

One folder per pair of taus under `output/` (`..._tau0.2_0.5_...`; where the two are equal,
`..._tau0.2_...`, the folder of `TAU=0.2`), then `output/tau_grid/`: `tau_grid.png` (every
pairing as a small grid, agent_1's tau up and agent_2's across, each cell split: the
upper-left half blue where agent_1 finished its own goal, the lower-right half orange where
agent_2 did), `tau_grid_counts.png` (in how many pairings each robot finished, over both
taus), `pairings/<agent_1>__<agent_2>.png` (one pairing, with the transition each finished),
`tau_grid.md` and `tau_grid.csv`. Logs in `logs/tau_grid/` (one per job, and `folders.txt`).

Size and time: a rollout takes 10-40 min on 2 threads, the 49 of one pair of taus about 15 h
of 2-thread work, so the default grid about 375 h: some 12 h with 32 workers (64 cores), 24 h
with 16. WORKERS (default: the CPUs / THREADS) jobs run at once, THREADS (default 2) each, a
new one as soon as one ends; each needs about 2.5 GB at its peak, so 32 workers need about
80 GB. Records only by default (`NOVIDEO=0` draws the three videos of every rollout); to
draw one pairing's afterwards, run its line above (`--resume` draws the videos of a record
that has none, and rolls nothing out again). Interrupted? Run the same line again. Also
takes PHYS, SCENE, STEPS, LOGS, OUT, PYTHON.

## What to run (2026-10-30): the belief update of the paper / the oracle

The code is that of the package of 2026-10-21 with the belief update and the recursion
rebuilt on 2026-10-30 after the paper and the authors' solver (README, "The belief update
of 2026-10-30"): two posteriors about each robot -- its physical goal from its level-1
hypothesis policies averaged over the social posterior (Eq. 6 as printed), in two stages by
its hands; its social goal from its level-1 hypothesis policies averaged over the physical
posterior (Eq. 5) -- the idle filter over both, Eq. 5's transition term for what the hands
reveal, and nothing else. The lapse, the carry and park tables, the "outcome" observation
and the belief temperature of before are gone, and so are their flags
(`--inference-temperature`, `--partner-observed`): the belief is updated with Eq. 7's tau,
of the action the partner chose. `--temperature TAU` sets Eq. 7's tau (default 0.2, in
moves' costs; the 49 of 2026-10-26 ran at 0.15). `--physical-likelihood level0` runs the
physical posterior on the partner's level-0 policies instead, as the authors' code does
(folders `..._eq56l0`; the README says why Eq. 6 as printed is the default). From this
folder, after `pip install -r requirements.txt`:

    sh run_all_49.sh             # one process, the 49 one after the other
    sh run_all_49_parallel.sh    # 7 processes at once, one per agent_1 goal (a many-core server)
    TAU=0.15 sh run_all_49_parallel.sh   # ... at tau 0.15
    TAU=0.15 PHYS=level0 sh run_all_49_parallel.sh   # ... with the authors' level-0 estimator

The 49 pairings are the seven social goals of agent_1 by the seven of agent_2: cooperation,
conflict, competition, coercion, exchange, none (physical goal only), hinder -- every
ordered pair, `run_all_49.sh` lists them. Both write to

    output/object_inference_two_w1_1_selective_level2_levels2_2_swap_steal_carry1_spoiler_inertia1_hands5_staged_byhand_pick3x2_hand2_eq56/

(`..._tau0.15_...` with TAU=0.15) and keep what is saved there (`--resume`): run the same
script again after an interruption. Every folder now ends in `_eq56`, so that no record of
the folders of before, made with the devices of then, is kept. `run_all_49_parallel.sh`
takes WORKERS (default 7) and THREADS (default: the CPUs / WORKERS); each process needs
about 2.5 GB at its peak, so 7 at once need about 18 GB. It compiles the kernels once on
the tiny scene first (logs/warmup.log), writes one log per worker (logs/worker_<goal>.log),
and ends with one pass over all 49 that writes results.md, results.png and summary.json
(logs/all49.log). Both scripts take PYTHON (default python3).

Tests: `python -m pytest social_pairings/tests/test_beliefs.py` (the update against the
equations, under a minute) and `python -m pytest social_pairings/tests/test_models.py`
(every array the decisions rest on against brute force; about 5 min per test on two cores,
so run it on the server: `python -m pytest -n 8 social_pairings/tests/test_models.py` with
pytest-xdist, or `SOCIAL_PAIRINGS_SLOW=1` for the slow ones too). Of the rest of the suite,
the mechanics (`test_hands`, `test_goals`, `test_hindrance`, `test_spoiler`, the figure)
pass; the tests that pin the course of a rollout made under the belief devices of before
were re-pinned to the courses of now where the run was looked at (`test_render`,
`test_objects`, `test_defaults`); those not yet looked at, to re-pin or retire on the
server, are `test_explicit.Rollouts.test_a_hinderer_lifts_the_object_it_guards`,
`test_inertia.Rollouts.test_greedy`, and whatever `test_pairing`, `test_hinder` and `test_competition` (the six-object rollouts
under TOLD) show -- they were still running when the package was made.
(`test_values.Record.test_the_file_and_the_videos` failed in the cloud container on the
duration ffmpeg reports, half a second short with every frame there; the untouched 10-21
package fails it there too.)

## Two spare objects: `run_spare2.sh` (2026-10-30, late)

One pairing on the two-room scene as it is plus two spare objects, one per room, on (2, 1)
and (6, 5) (`--scene spare2`), at the 49's settings (levels 2 / 2, every object inferred,
swaps and taking, the pick-up of three transitions at two moves' worth, carry 1, inertia 1,
tau 0.15). Four objects make 5,545 object configurations (4.35 million joint states) and
24 physical goals per robot: about 170-190 GiB of solver memory, 23 GiB of it the dynamics
tables. Time: a transition of the variant without swaps and with an instant pick-up (126
configurations) took 390-420 s on 2 cores in the cloud container (763 s the first, solved
cold); this model has 44 times its states, so about 4-5 h a transition on 2 cores, some
20-30 min on 32 cores -- a day for 60 transitions. There is no checkpoint inside a rollout,
so run it under tmux or nohup; `STEPS=30` halves it, `SWAP=0` drops the swaps (718
configurations, about 24 GiB, about 8 times faster). Not a machine for two at once.

    sh run_spare2.sh                         # hinder (agent_1) vs coercion (agent_2)
    PAIR="coercion hinder" sh run_spare2.sh  # the other way round
    MEM=200 THREADS=32 sh run_spare2.sh      # memory cap (GiB; default 90 % of the machine) and threads
    SWAP=0 sh run_spare2.sh                  # without swaps: ~24 GiB, ~8x faster

It prints a line per transition as it happens (`--verbose`; logs/spare2_<pair>.log), writes
the record and the three videos to
`output/object_inference_spare2_w1_1_selective_level2_levels2_2_swap_steal_tau0.15_carry1_spoiler_inertia1_hands5_staged_byhand_pick3x2_hand2_eq56/`,
and `--resume`s. A machine with too little memory is told within minutes, before the big
arrays, how much the model needs. (Without swaps the same scene is 718 configurations and
about 24 GiB; without swaps and with an instant pick-up, 126 and about 4 GiB.)

## The package of 2026-10-21: the pick-up at two moves

The code is that of the package of 2026-10-18; this one adds two scripts that run the 49
pairings with the settings in use since 2026-10-18 evening -- the pick-up costing two moves'
worth on its first transition (`--pick-cost 2`) instead of the default three.

## The package of 2026-10-18

Package of 2026-10-18: nothing is automatic any more -- a robot picks its own
object up by "pick" and sets it down, finishing, by "place", like any other object; a
pick-up takes three transitions, the robot lifting meanwhile with "pick" its one action,
and costs three moves' worth on the first; place, take, give and swap cost two moves' worth
(the user, 2026-10-18; "Nothing automatic" in the README). The rule of before is
`--automatic-hands`. Every folder now ends in `_byhand`; `--resume` keeps no record of the
folders of before, which were made under the rule of before. The package of 2026-10-17,
late, fixed a stop of the whole run when a robot's value video had a term in some steps and
not in others; that fix is in this one.

Since the evening of 2026-10-17 every rollout writes three MP4 videos (no GIFs): the
annotated animation, agent_1 green and agent_2 red, and one video per robot of its value at
every cell it could stand on, term by term (its own goal, its action costs, each social
term), with its action at each cell -- "The value videos" in the README. Since 2026-10-17 a
robot keeps two posteriors about its partner, by the partner's hands:
while the partner holds nothing, what it does is evidence about which object it wants (and
its social goal) and none about its destination; while it holds an object, about its
destination (and its social goal) and none about its object, whose belief stays where the
pick-up left it. The robot plans with the belief so built. Since 2026-10-16 a robot has ten
actions: four moves, stay, and five hand actions -- pick, place, take, give, swap -- each
with its own Q value and its own probability in the softmax (Eq. 7), where it had one action
for its hands, "use". The README in `social_pairings/` has both ("Two posteriors, by the
partner's hands"; "Five hand actions in place of "use"", with which rollouts of the tests
changed and why a robot may expect its partner to get stuck in its plans).

Set up once (Python 3.10 or newer; tested with 3.13):

    pip install -r requirements.txt      # now also imageio-ffmpeg, which brings its own ffmpeg

Then, from this folder (the one holding `social_pairings/` and `examples/`), one line:

    python examples/run_object_inference.py --scene two --all-pairings --levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --resume

(with the pick-up at two moves, what the two scripts run: add `--pick-cost 2`; the folder
then ends in `_byhand_pick3x2_hand2`)

It writes everything to

    output/object_inference_two_w1_1_selective_level2_levels2_2_swap_steal_carry1_spoiler_inertia1_hands5_staged_byhand_eq56/

- `<agent_1>__<agent_2>__inferred.json` (the record), `.mp4` (the animation) and
  `__inferred__values_agent_1.mp4` / `_agent_2.mp4` (the value videos) for each of the 49
  pairings, as each one finishes
- `results.md`, `results.png`, `summary.json`: the table of all 49, at the end

`--resume` skips the pairings already saved, so after an interruption the same line carries
on where it stopped. It never picks up a record of the package of 2026-10-14 (folders
without `_hands5`, robots with "use"): it stops with a message instead; the folders of the
packages of 2026-10-16 and -17 (`_hands5`, `_hands5_staged`: the automatic hands) are
simply other folders. `--automatic-hands` gives the rule of before 2026-10-18 and the
folder without `_byhand`; `--pick-steps`, `--pick-cost` and `--hand-cost` change the
pick-up's length and the costs (and the suffix); `--one-posterior` gives the update of
before 2026-10-17. `--threads N` caps the CPU threads (default: all of them). numba
compiles on the first run (a minute or two).

Settings in that line: two-room scene, both robots at level 2, each infers the other's
object, destination and social goal; swaps on a place and taking from hands allowed;
carrying the partner's object costs an ordinary move; 60 steps at most; `--inertia 1`: a robot does not undo its
last action (step back, place what it just picked up, give back what it just took, ...)
unless undoing beats every other action by more than one move's cost. Two settings are the
package's defaults and need no flag: the hinderer has no goal of its own
(`--hinder-own-goal` restores the old one), and competition is read as a rival.

Time and memory: about 21 to 30 s a transition on 2 cores, 2 s of it for the value maps
(the five pairings of 2026-10-18: 652 s for 31 transitions, 1766 s for 60; then a minute
for the three videos). Pairings that run to the 60-step cap take 25 to 40 minutes; all 49
should take roughly 12 to 18 hours on 2 cores. Memory: 1.8 GB resident measured on the
two-room scene (2.6 x the object configurations of before, which the tables grow with);
plan on 2.5 GB per run. `--no-video` skips the videos; the records hold everything they
need, and the same line run again with `--resume` draws the videos of the records that
have none.

Faster on a many-core machine (what `run_all_49_parallel.sh` does, by hand): 8 pairings at a
time with 4 threads each, then one pass that only reads the 49 saved runs and writes the table
(no rerun, no re-render):

    F="--scene two --levels 2 2 --objects inferred --swap --steal --foreign-carry-cost 1 --inertia 1 --pick-cost 2 --resume"
    G="cooperation conflict competition coercion exchange none hinder"; for a in $G; do for b in $G; do echo $a $b; done; done | xargs -P 8 -L 1 sh -c "python examples/run_object_inference.py $F --threads 4 --pairing \$0 \$1"
    python examples/run_object_inference.py $F --all-pairings

Tests: `python -m pytest social_pairings/tests` (long; `SOCIAL_PAIRINGS_SLOW=1` adds the
slow ones).
