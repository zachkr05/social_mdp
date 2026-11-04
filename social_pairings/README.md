# social_pairings

Social MDPs for two pick-and-place robots that share their objects, after Tejwani et al.,
*Incorporating Rich Social Interactions Into MDPs* (ICRA 2022, arXiv 2110.10298): the five
social goals of Table I at reasoning levels 0 to 2, plus `hinder`, and a switch for what a
competitor is: five readings, from the table as printed to a rival that races its partner,
each for its own goal. Nobody is told its partner's social goal: each robot infers it, and
its partner's destination, from what the partner does (since 2026-10-08; `Config(**TOLD)`
tells each robot its partner's goal, as before). Nor which object its partner wants: a
physical goal is an object and a destination, and every object of the scene is a candidate
(since 2026-10-09: `Config.partner_object`, "Which object the partner wants"). A robot with
a physical goal only reasons at level 0, and every robot that models its partner allows for
it being idle (both since 2026-10-08 noon: `Config.physical_level`,
`Config.idle_hypothesis`). A level-2 cooperator cooperates only with a partner it takes to be
cooperative, competitive or exchanging, and minds its own goal against any other (since
2026-10-11: `Config.cooperate_with`, "Whom a cooperator cooperates with"). Every robot with a
social goal reasons at level 2, a hinderer included (since 2026-10-11:
`Config.level_free_level`, "Every robot with a social goal at level 2"). Two options of
2026-10-12, off by default, let a robot swap what it holds for the object lying on an
object's place, and take what its partner holds (`Config.swap_on_place`, `Config.steal`:
"Swapping on a place and taking from the partner's hands").

```python
from dataclasses import replace

from social_pairings import TOLD, Config, run_baseline, run_pairing, six_object_scene
from social_pairings.hindrance import format_report, hindrance_report
from social_pairings.render import render_videos

# on the six-object scene each robot is told which object the other wants: with all six
# objects candidates its model needs about 13 GiB ("Which object the partner wants")
scene, known = six_object_scene(), Config(partner_object="known")
result = run_pairing(scene, ("hinder", "cooperation"), cfg=known)    # weights (1, 1), the paper's
result["done_at"]                   # (29, None): the hinderer puts the cooperator's object back
                                    # (15), is then helped (29), and takes it again (30)
result["levels_used"]               # (2, 2): a hinderer at level 2 too, since 2026-10-11
print(format_report(hindrance_report(result, run_baseline(scene, cfg=known))))
table = replace(known, cooperate_with="all")            # Table I's cooperator: it helps everybody
run_pairing(scene, ("hinder", "cooperation"), cfg=table)["done_at"]               # (5, None)
level1 = replace(known, level_free_level=1)             # the hinderer at level 1, as before
run_pairing(scene, ("hinder", "cooperation"), cfg=level1)["done_at"]              # (None, None)

helped = run_pairing(scene, ("cooperation", "exchange"), cfg=known)
helped["done_at"]                   # (38, 26): the cooperator delivers the exchanger's object, then its own
run_pairing(scene, ("cooperation", "exchange"), cfg=table)["done_at"]             # (22, 10)
render_videos(helped, "helped")     # helped.mp4, the annotated animation with each robot's beliefs,
                                    # and helped__values_agent_<k>.mp4, each robot's value at every cell

# a physical goal only is level 0: a plain MDP, no social stance, nothing inferred, and two
# such robots pass each other.  Asked for level 2 they pass as well, told or not; at w = 30,
# not told, each plans against a model of the other that may give way: a stand-off
told = Config(**TOLD)                                   # the partner's object known as well
run_pairing(scene, ("none", "none"), cfg=known)["done_at"]                          # (9, 11)
run_pairing(scene, ("none", "none"), levels=(2, 2), cfg=told)["done_at"]            # (9, 11)
run_pairing(scene, ("none", "none"), levels=(2, 2), cfg=known)["done_at"]           # (9, 11)
run_pairing(scene, ("none", "none"), levels=(2, 2), weights=(30, 30),
            cfg=known)["done_at"]                                                   # (None, None)

# a competitor against a cooperator: a rival that judges whom it faces by what it sees,
# and the five readings of competition told
pair = ("competition", "cooperation")
run_pairing(scene, pair, cfg=known)["done_at"]                                      # (55, 52)
run_pairing(scene, pair, cfg=table)["done_at"]                                      # (5, None)
run_pairing(scene, pair, cfg=told)["done_at"]                                       # (10, None)
run_pairing(scene, pair, cfg=replace(told, competition="race"))["done_at"]          # (5, 17)
run_pairing(scene, pair, cfg=replace(told, competition="table"))["done_at"]         # (5, 5)
hostile = replace(told, competition="table_hostile")    # Table I with -R_j against a cooperator too
run_pairing(scene, pair, cfg=hostile)["done_at"]                                    # (33, None)
rival = replace(told, competition="rival")              # races its partner, but a cooperative one
run_pairing(scene, pair, cfg=rival)["done_at"]                                      # (33, None)

# two competitors at level 2, told: under the table nobody finishes, two rivals race
two = ("competition", "competition")
run_pairing(scene, two, cfg=hostile)["done_at"]                                     # (None, None)
run_pairing(scene, two, cfg=rival)["done_at"]                                       # (9, 11)

# told (as until 2026-10-11), a competitor of a race form reasons at level 1 unless level 2
# is asked for; a rival at 2.  Not told, every robot with a social goal at 2
run_pairing(scene, pair, cfg=told)["levels_used"]                                   # (1, 2)
run_pairing(scene, pair, levels=(2, 2), cfg=told)["levels_used"]                    # (2, 2)
run_pairing(scene, pair, cfg=known)["levels_used"]                                  # (2, 2)

# a scene in which the two objects have to be exchanged ("An exchange scene"): each robot
# stays in its own room, its object lies in the other one, the doorway is a hand-off cell
from social_pairings import exchange_scene
put_down = replace(known, set_down_for_partner=False, hand_over=False)   # put down, pick up
run_pairing(exchange_scene(), ("exchange", "exchange"), cfg=put_down)["done_at"]    # (20, 17)

# which object the partner wants, inferred (the default): the two-room scene, whose only
# objects are the robots' own; each robot ends up sure which one the other wants
from social_pairings import two_room_scene
both = run_pairing(two_room_scene(), ("cooperation", "cooperation"))
both["done_at"]                     # (17, 14); (22, 5) with the object known
both["final_object_belief"]         # ([0, 1], [1, 0]): agent_1 sure that agent_2 wants O2, agent_2 that agent_1 wants O1

# two options of the hands, off by default: swap what you hold for what lies on an
# object's place; take what your partner, next to you, holds
swaps = Config(swap_on_place=True, steal=True)
run_pairing(two_room_scene(), ("coercion", "hinder"), cfg=swaps)["done_at"]          # (9, 13); (None, None) without
```

From the project root:

```
python examples/run_object_inference.py              # which object the partner wants, inferred and known: five pairings on the two-room scene, weights 1 1
python examples/run_object_inference.py --scene spare  # ... with one spare object, a candidate too (78 minutes on 2 cores)
sh run_spare2.sh                                       # one pairing (default hinder / coercion) with two spares, one per room, at the 49's settings: ~170-190 GiB, hours (2026-10-30; RUN_ALL_49.md)
python examples/run_object_inference.py --scene center --pairing hinder cooperation   # both objects in the doorway, two spares (26 minutes)
python examples/run_object_inference.py --cooperate-with all   # ... with Table I's cooperator, which helps every partner (the folders of before 2026-10-11)
python examples/run_object_inference.py --pairing hinder cooperation --pairing hinder competition --pairing hinder none   # the hinderer at level 2, its default since 2026-10-11 ...
python examples/run_object_inference.py --level-free-level 1 --pairing hinder cooperation   # ... and at level 1, as before
python examples/run_object_inference.py --levels 2 2 --objects inferred --pairing coercion hinder   # both robots at level 2, a robot without social goal too; the object inferred only
python examples/run_object_inference.py --levels 2 2 --objects inferred --swap --steal   # ... with swaps on a place and taking from hands
python examples/run_object_inference.py --levels 2 2 --objects inferred --swap --steal --pairing cooperation cooperation --inference-temperature 1   # ... a softer belief update: no ping-pong
python examples/run_object_inference.py --levels 2 2 --objects inferred --swap --steal --inference-temperature 0.5 --all-pairings   # ... all 49 pairings of the seven goals (3.5 hours on 2 cores)
python examples/run_all_pairings.py --objects known  # every pairing of the seven goals, nobody told, each robot knowing which object the other wants: 49 rollouts with videos, weights 1 1
python examples/run_all_pairings.py --scene exchange --objects known   # the same on the exchange scene
python examples/run_all_pairings.py --told           # each robot told its partner's goal and object, as until 2026-10-08
python examples/run_pairings.py --weights 30 30      # eight pairings: records, videos, report
python examples/run_pairings.py --weights 30 30 --told --competition race_then_hinder    # the eight as in "Results (told)"
python examples/run_pairings.py --weights 30 30 --competition race --case 1 3 7
python examples/run_pairings.py --weights 30 30 --competition table_hostile --case 1 3 7
python examples/run_pairings.py --weights 30 30 --levels 2 2 --case 1 3 7    # competitors at level 2
python examples/measure_inference.py output/all_pairings_w1_1_rival_inferred_idle    # how well the robots there inferred each other's goals
python examples/tier_scenes.py                       # test_tiers' low / medium / high scenes on the house at 0.5 m, made fair (A* within 10 %): the table and two figures
python examples/run_tier_scenes.py                   # the Social MDP on them, each scene with its suite's goals: 42 rollouts with videos, weights 1 1
python examples/compare_competition.py               # the five readings of competition side by side
python examples/compare_levels.py                    # a competitor at level 1 and at level 2 side by side
python examples/run_exchange_scene.py --controls     # the exchange scene: exchange / exchange, weights 1 1, and ten control pairings
python examples/run_exchange_scene.py --hand-over    # the same scene with hand-overs allowed
python examples/run_exchange_scene.py --exchange-level1 none    # ... with Table I as printed: help is read as cooperation
python examples/run_exchange_scene.py --told         # each robot told its partner's social goal
python -m pytest social_pairings/tests    # 393 tests, 20 of them skipped (the slow ones); about 2 h on 2 cores (2026-10-18)
SOCIAL_PAIRINGS_SLOW=1 python -m pytest social_pairings/tests    # all 393
```

Every script writes its records and videos (MP4 since 2026-10-17; GIFs until then) to a folder of `output/` whose name ends in
`_inferred` (`_told` with `--told`), and in `_objects` where the partner's object is inferred
(`--objects`, the default since 2026-10-09: a script run without it never writes into the
folders of before, which were made with the object known; `--objects known` names them as
then), in `_selective` under the cooperator of 2026-10-11 (`--cooperate-with all`,
Table I's cooperator, names them as before), and in `_level2` with every robot with a social
goal at level 2, a hinderer included, the default since 2026-10-11 (`--level-free-level 1`,
or `--told`, names them as before); `run_object_inference.py` adds `_levels2_2` for
`--levels 2 2`, `_swap` and `_steal` for the options of the hands, and `_itau<T>` for
`--inference-temperature T`. `--resume` keeps a record only if it was made with the
options asked for, a record of before included (`config.written_with`). Every script's default weights are 1 1, the paper's (since 2026-10-08 evening; 30 30
until then, so the folders of before are named `w30_30`). With the objects inferred the
six-object and exchange scenes need more memory than a laptop has (about 13 GiB, and far
more): `--max-objects 3` keeps the model at about 1 GiB, and a model that cannot fit stops at
once and says so.

Needs `numpy`, `scipy`, `numba`; `matplotlib` and `pillow` for the figures.

The package is self-contained. It shares no code with `social_mdp/` in the same folder (the
experiment engine behind `run_experiment`, the ablation arms and the congruence scores): it does
one thing, roll out a pairing of social goals and say what happened.

## The gate scene and the testbed (since 2026-11-03)

`scenes.gate_scene(handoff="anywhere", decoy=False)` (`GATE_MAP`, drawn 2026-11-02): two 3 x 4
rooms joined by a one-cell gate and a long way round (2 moves between the gate's neighbours
through it, 12 round), mirror-symmetric. Each robot starts next to its partner's object; its
own object lies in the other room and its destination at home, among two more candidates
(six in all). Alone a task takes 19 transitions; whoever stands in or next to the gate blocks
the short way. `handoff`: where an object may be put down -- "anywhere" (the 24 free cells
that are no object's place and no candidate; the user: "Lets enable this scene with being able
to drop off anywhere"), "few" (the gate, a far cell in each room, the middle of the way round),
"gate" or "none"; `decoy` adds a spare object on the way round. Sizes at the 49's settings
(swaps on): "gate" 32 cells, its dynamics tables alone 20 GiB (1,068 object configurations
without swaps, 139,392 events), some 60-70 GiB a rollout -- an estimate, too big to build on a
small machine; "few" 177,152 joint states, 6.2 GiB; "gate" alone 100,352, 3.4 GiB.

`examples/run_object_inference.py --scene` takes `gate`, `gate_few`, `gate_gateonly` and the
twelve fair tier scenes of `examples/tier_scenes.py` by name (the house at 0.5 m, 79 free cells,
no hand-off cells: 480,557 joint states, 15.6 GiB at the 49's settings). `TESTBED` names the
seven of the testbed -- the gate scene and the meeting suite's six -- and `run_testbed.sh` runs
hinder against each of the seven social goals on them (`RUN_ALL_49.md`, "The testbed").
`run_tau_grid.sh` takes several scenes (`SCENES`) and a goal list per robot (`GOALS1`,
`GOALS2`), a job per rollout. Checked: `tests/test_testbed.py` (the registry, the tier scenes
against `tier_scenes.SCENES`, the gate scene as drawn, its symmetry, the two ways between the
rooms, the hand-off variants, a world built for the gate-only variant).

## The partner's goal may change: a hazard (options, since 2026-11-02)

The user: "Ok lets go ahead an implement this: A better cure for locked-in beliefs is a hazard
rather than a reset". Three options, all 0 by default (the update of before, every folder of
before kept by `--resume`): `Config.goal_hazard`, `social_hazard` and `hand_hazard`
(`--goal-hazard`, `--social-hazard`, `--hand-hazard`; `HAZARD`, `SOCIAL_HAZARD`,
`HAND_HAZARD` in the scripts).

**The model.** Baker, Saxe & Tenenbaum (2009), "Action understanding as inverse planning",
*Cognition*, model M2 ("changing goals"): the partner's hidden goal is redrawn from the prior
with probability $h$ at every step. Filtering that, each posterior gets a prediction step
after its update,

$$b(g) \leftarrow (1-h)\,b(g) + \frac{h}{|S|},$$

with $S$ the hypotheses it still holds possible: the ones the hands' outcomes ruled out (Eq.
5's transition term) stay out. $h = 0$ is the paper's M1 (a fixed goal), the update of before;
$h = 1$ its heuristic H (only the last action counts).

**Why.** At $\tau = 0.1\,c(\text{move})$ a move that a hypothesis rates one move's cost worse
multiplies it by $e^{-10}$, and every such move digs deeper: after three of them a 98 % belief
is at $6 \times 10^{-12}$ and needs ten moves in its favour (each ×20, as at transitions 9-10
of coercion / hinder, 2026-11-02) to be back above 90 %. Under a hazard of 0.02 it never falls
below $h/|S|$ (0.5 % with four objects) and is back in three, however deep it went. A belief
that no evidence keeps up fades toward the uniform at a rate $h$ a transition: red's "green
wants O4" (0.62) in hinder / exchange, which green's moves say nothing about while red reads
green as a hinderer, would be halfway to 0.25 in about 35 transitions at 0.02, 7 at 0.1. What
the hazard does not do is soften what one move does: that is $\tau$ (the paper's best fit to
people's online goal inferences was $\beta = 2$ per move's cost, i.e. $\tau = 0.5\,c(\text{move})$,
with $h = 0.25$; for retrospective inferences $\beta = 0.5$, $h = 0.65$).

**Where.** `goal_hazard`, on the physical posterior: under `belief_stages` (the default) the
factor the partner's next action speaks to -- which object it wants while its hands are empty,
the destination given each object while it holds an object or lifts one -- the other factor
staying where it was, as the stages keep it (a belief about the object does not leak away
while the partner carries the object, with nothing to bring it back). Without the stages, the
whole (object, destination) posterior. `social_hazard`, on the social posterior, all of it.
Both after every update, i.e. in the beliefs the robots plan with and the videos show; none
while the partner lifts (there is no update then). `hand_hazard`, the user's "when they pickup
an object we forget our posterior on the object and when they place an object we forget our
posterior on goal placement locations", made partial: before the evidence of the partner's
pick or take, its object factor is mixed with weight `hand_hazard`; before that of its place,
give or swap, its destination factor. The hand action's own evidence then counts in full. 1
forgets what came before altogether.

**Folders.** `..._hazard<H>`, `..._socialhazard<H>`, `..._handhazard<H>` after `_eq56` (and
before `_taskconflict`). The titles name them ("hazard 0.02"). `models.mix_uniform`,
`SocialMDP.hazard_phys`.

    SAME=1 TAUS="0.1 0.3 0.5" HAZARD=0.02 sh run_tau_grid.sh                      # the physical posterior
    SAME=1 TAUS="0.1 0.3 0.5" HAZARD=0.02 SOCIAL_HAZARD=0.02 sh run_tau_grid.sh   # ... and the social one
    SAME=1 TAUS="0.1" HAZARD=0.02 HAND_HAZARD=0.5 sh run_tau_grid.sh              # ... and half forgotten at the hands

**Checked** (`tests/test_hazard.py`, 33 s): the option and the records of before; the mixing
against its definition (normalised, the floor, the ruled-out hypotheses out, $h = 1$ uniform)
and the two factors (each keeps the other); the engine's update with all three hazards against
the reference of `test_beliefs.py` with the mixing written out, from random beliefs, with empty
hands, at a pick-up, holding an object (moves, place, give) and without the stages; lifting
still tells nothing; the arithmetic above; eight transitions of the 3 x 3 map against the
reference. `test_beliefs.py` passes unchanged (0 is the update of before). The script on the
3 x 3 map with all three on: the folder named, the floors in the record, `--resume` keeping it.
Run with it when handed over, all passed (99 tests, 7 min): `test_hazard`, `test_beliefs`,
`test_defaults.TheDefaults`, `test_conflict` (`Terms`, `Option`, `Resume`),
`test_temperatures.Option`, `test_goals`, `test_render` (`HeadlineTests`, `OldRecordTests`,
`TextTests`). Not run: the rest of the suite and any rollout of the two-room scenes.

## Task conflict (option, since 2026-11-01, evening)

The user: "Ok lets implement this: Option 1: task conflict ... $R_{\text{conflict}} = r(g_i) -
w\,r(\tilde g_j) - c(a_i)$". `Config.conflict = "task"` was the default for a few hours that
evening; then the user: "Ok can we revert the change for the conflicting agent and just
restore it to how it was?" -- so `"table"`, Table I as printed, is the default again, as
before, and the task conflict an option (`--conflict task`, `CONFLICT=task`). What is below
describes the option.

**Why.** Table I's conflict above level 1 is $-\tilde R_j$, and $\tilde R_j$ holds the partner's
own term about the robot. Against a cooperator that term is $+w_j\,r(\tilde g_i)$, so

$$R_i = \bigl(1 - w_i w_j\,\tilde b(g_i)\bigr)\,r(g_i) - w_i\,r(\tilde g_j) - c(a_i) - \dots$$

with $\tilde b(g_i)$ how sure the cooperator is of the robot's goal. At $w = 1$, once the
cooperator knows that goal, finishing it is worth nothing to the conflict robot. In conflict
/ cooperation (2026-11-01) it picked its object up and put it back down: the value chart of
that step had "down" +1 physical and -9 social, the social part being the cooperator's term
about it coming back with a minus sign.

**The rule.** A conflict robot keeps the table's entry at l = 1 at every level, against every
partner:

$$R_i = r(g_i) - w_i\,r(\tilde g_j) - c(a_i), \qquad (\alpha, \beta, \nu) = (0, 0, -w_i).$$

It does its own task and works against its partner's: against the partner's goal reward, not
its whole reward. No $\tilde R_j$, so no term of the partner about the robot comes back into its
reward, and its level says only whom it takes its partner to be, as for a hinderer
(`goals.level_free`). Three things follow:

- It is what Table I's recursion converges to when carried to every level: $U_i \propto
  r(g_i) + w\,r(g_j)$ for every partner type.
- It is the reward of the hinderer with a goal of its own (`hinder_own_goal=True`, the hinderer
  until 2026-10-14). The hinderer of now has none: $-c(a_i) - w\,r(\tilde g_j)$. Conflict is "do
  my task and spoil yours"; hinder is "spoil yours".
- What it gives up: its term no longer depends on the partner's social goal. A helper and a
  hostile partner are spoiled alike.

**What changes.** Only a level-2 conflict robot's own reward. Its partner models it at level 1,
where both readings are $-w\,r(\tilde g_j)$. So every pairing without a conflict robot at level
2 is the same rollout under both readings, every Q value bit for bit (`goals.conflict_matters`).
Of the 49, the 13 with conflict change. The record's term reads "conflict: w (-1 r(g_j)), the
partner's goal reward, against every partner [task conflict ...]"; the titles say "conflict
(task)".

**Folders.** `run_object_inference.py --conflict task` writes to `..._taskconflict`; the
default, Table I's conflict, to the folder without the suffix, as before. `--resume` in either
takes every pairing without a level-2 conflict robot from the other, with its videos, where
that record was made with the options asked for in every other field; it rolls out the
conflict pairings alone. A record is kept in place on the same rule. The scripts take
`CONFLICT=task`. The titles name the task conflict, "conflict (task)"; under Table I's it is
"conflict", as before.

**Checked** (`tests/test_conflict.py`): the term against every partner, by the engine and by
the reference; every other goal's term untouched under every reading of competition; equal to
the hinderer-with-own-goal's; the option, `TOLD`, the records of before; the levels; the record
text and the labels; `--resume`'s rule and the copy from the folder of before, on made-up
records. Every array of levels 0-2 against brute force with a level-2 conflict robot (weights
1.5 / 2.0) facing a level-2 cooperator, everything inferred: passed (6 min on 2 cores). Four
transitions on the 3 x 3 map under both readings: cooperation / hinder is the same rollout,
every Q value bit for bit, and conflict / cooperation is not (passed, 27 s). Also run and
passed when the code was handed over: `test_goals`, `test_spoiler.Terms`,
`test_defaults.TheDefaults`, `test_render.HeadlineTests`, `test_temperatures.Option`, and the
script on the 3 x 3 map (`--conflict table`, then the default with `--resume`: the
cooperation / hinder record taken, byte for byte, conflict / cooperation rolled out). The
rest of the suite was not run; under `TOLD` (`tests/told.py`, which most of it uses) the
reading is `"table"` and the code path is the one of before. With the default back to the
table's (22:50 the same evening): the tests above but the brute force re-run and passed (the
brute force asks for the task reading by name, the same model it passed with), and the script
both ways on the 3 x 3 map -- `--conflict task`, then the default with `--resume`: the
cooperation / hinder record taken from the `_taskconflict` folder byte for byte,
conflict / cooperation rolled out under Table I's term.

## A tau per robot (2026-11-01)

The user: "We want each agent to have its own tau so I should be able to run competitive at
0.2 and coercion at 0.5 in the same scene."  `Config.robot_temperatures = (agent_1's tau,
agent_2's)`, in the unit of `Config.temperature` (moves' costs); on the command line
`--temperature 0.2 0.5` (one value: both robots, as before).

**What a robot's tau is.** Eq. 7's $\tau$ is how a robot *reads* its partner: the softmax of
the partner's models it predicts the partner by (Eq. 4's $\psi$), and the likelihood of its
belief update about the partner (Eqs. 5-6, which take Eq. 7's policies). So robot $i$'s
$\tau_i$ sets how sharp its predictions of the partner are and how fast its beliefs about the
partner move: at 0.1 one move can make it nearly sure, at 0.5 it takes several. The robots
themselves still act greedily (Algorithm 1's argmax; under `action_selection="sample"` robot
$i$ draws at its own $\tau_i$).

**The rule.** Every model *of* robot $o$ is softmaxed at its partner's tau, $\tau_{1-o}$,
wherever it is used: when the partner predicts robot $o$ (its level-1 or level-2 model), when
the partner updates its belief about robot $o$, and inside robot $o$'s own level-2 model, where
the partner's level-1 models read robot $o$ by robot $o$'s level-0 policies. The robots know
each other's tau, as they know the rest of each other's model: the belief that robot $o$'s
model takes as its partner's belief about it is the partner's actual posterior, updated at
the partner's tau. (`SocialMDP.taus`, `read_inv_tau`; `kernels.softmax_policies` takes one
inverse temperature per robot.)

Two equal taus are stored as one (`temperature` = the value, `robot_temperatures` None), so
`--temperature 0.2 0.2` is the record and the folder of `--temperature 0.2`. The folders of
two different taus end in `_tau<agent_1's>_<agent_2's>`; the video titles and the table's title
name both. Records of before had one tau (`ADDED_FIELDS`).

**Checked** (`tests/test_temperatures.py`): with taus (0.1, 0.6), each robot's level-0
policies and the likelihood of its actions are exactly those of the one-tau engine at its
partner's tau (and differ from its own); a sampled robot draws at its own tau; a rollout keeps
both taus in its record; every array of levels 0-2 of competition against coercion, both at
level 2 and everything inferred, against the brute-force reference (which softmaxes a model of
robot $o$ at `tau_of(1 - o)`): at (0.1, 0.6) the largest deviations were level-1 psi 1.1e-5,
level-2 Q 2.8e-5 (limits 2e-4, 2e-3). The test also runs the taus the other way round; that
half, the rest of the suite and `run_tau_grid.sh` itself had not been run when the code was
handed over (2026-11-01; `tools/tau_grid.py`'s figures were drawn from made-up summaries).

**The sweep.** `run_tau_grid.sh` runs the 49 pairings at every pair of taus, `TAUS1` x `TAUS2`
(default 0.1 ... 0.5 each: 1225 rollouts; `SAME=1`: only one tau for both robots, the 49 at each
tau of `TAUS`), and `tools/tau_grid.py` puts them side by side:
every pairing as a small grid over the two taus (`tau_grid.png`), each robot's finishes over
both taus (`tau_grid_counts.png`), one figure per pairing with the transitions
(`pairings/`), and the tables (`tau_grid.md`, `tau_grid.csv`). `RUN_ALL_49.md` has the lines.

## The belief update of 2026-10-30: Eqs. 5-6 as the authors' solver keeps them

The user, 2026-10-30, with the authors' CUDA solver
([social-mdp-framework](https://github.com/Social-MDP/social-mdp-framework), branch
`social-interactions`, the ICRA 2022 code) as the oracle: "There should be Bayesian
dynamics just based on the paper ... Keep my action space. Keep the general simulator. Keep
how the posterior for object grabbing and object seeking is different ... Just fix how the
posterior is calculating wrong. Fix how the recursive solver is completely wrong." The
choices made with the user: two separate posteriors as the oracle keeps them; of the devices
of before only the idle filter stays; the recursion as the oracle / the paper has it.

**What the oracle does, per transition** (`main.cpp`, `smdp.cu`): the level-0 policies of
every physical goal are solved once; a physical-goal estimator per robot (`pgeA`, `pgeB`:
`MultiPolicyEstimator`) multiplies each hypothesis by the probability its level-0 policy
gives the observed transition, Eq. 6; the level-1 models of a robot -- one per social goal,
each with the partner at level 0 under the current physical estimate -- are solved in one
batch, and a social-goal estimator per robot (`sgeA`, `sgeB`) multiplies each hypothesis by
the probability its level-1 policy gives the observed transition, Eq. 5; a level-2 robot
plans against the mixture of those level-1 policies under the social estimate, with the
physical estimate in its reward (Eq. 4), and acts greedily (Algorithm 1). The estimator of
a robot's goal is one object, shared by the robot's partner and by the models of the
partner inside the robot's own level-1 models (`ctxA.goals[1] = pgeB`, `ctxB.goals[0] =
pgeA`): the evidence is the same. Beliefs are frozen inside a solve and every model is
re-solved every transition.

**What this package now does** (`social_pairings/models.py`), the same structure in the
shared-object world, with every social-goal reading of this project (the rival, the
selective cooperator, the hinderer without a goal of its own, the unweighted exchange copy)
and the world, the ten actions and the rollout untouched:

- **Two posteriors about each robot** (`SocialMDP.about[j]`, held by its partner and by the
  partner's models inside robot j's own level-1 hypotheses, as in the oracle):
  - *physical*, over the PG = M x K (object, destination) goals, updated from robot j's
    actions by the likelihood of robot j's **level-1 hypothesis policies averaged over the
    social posterior as it was**, $b_g(g) \propto b_g(g) \sum_\xi b_\xi(\xi)\,
    \psi^1_{j,(g,\xi)}(a_j \mid s)$ (Eq. 6 as the paper prints it; the authors' code runs
    its physical-goal estimator on the level-0 policies instead, $\psi^0_{j,g}(a_j \mid s)$
    -- `Config.physical_likelihood="level0"`, see below) -- **in two stages, by robot j's
    hands** (`Config.belief_stages`, the user's rule of 2026-10-17, kept): while robot j
    holds nothing its actions are evidence about which object it wants and none about its
    destination (the hypotheses that share an object get their group's mean likelihood under
    the belief, so the belief about the destination given the object does not move); while it
    holds an object, evidence about its destination and none about its object (the update is
    normalised object by object, so the belief about the object stays where the pick-up left
    it);
  - *social*, over the social goals held possible (`Config.social_hypotheses`), updated by
    the likelihood of robot j's **level-1 hypothesis policies averaged over the physical
    posterior as it was**, $b_\xi(\xi) \propto b_\xi(\xi) \sum_g b_g(g)\,\psi^1_{j,(g,\xi)}(a_j
    \mid s)$ (Eq. 5; the authors' SGE) -- the sum over the physical goals the transition term
    (below) keeps, since $T(s^t \mid s^{t-1}, a_J)$ is the same under every social goal given
    the physical one;
  - one **idle probability** (`Config.idle_hypothesis`, the one device kept): a two-mode
    filter on top of both -- the partner acts by its hypotheses, or stands still whatever its
    goal (`idle_eps`), with the switch probability `idle_hazard` every transition -- so that a
    stalled partner pins neither posterior;
  - **Eq. 5's transition term** $T(s^t \mid s^{t-1}, a_J)$ (`Config.condition_on_placement`,
    `SocialMDP.transition_term`): the outcomes of the hands that depend on the hidden goal
    rule out the hypotheses that predicted otherwise -- robot j finishing (its goal is its own
    object on that cell), robot j's set-down of its own object on a candidate that did
    nothing (not that cell), its partner's set-down for it that counted or failed (that
    object on that cell, or not), its partner's hand-over robot j took or did not (that
    object, or not).
- Where the paper and the oracle part, an option: `Config.physical_likelihood` --
  `"level1"` (the default), Eq. 6 as the paper prints it, the partner's level-1 hypothesis
  policies averaged over the social posterior; `"level0"`, the oracle's physical-goal
  estimator on the partner's level-0 policies (`--physical-likelihood level0`, folders
  `..._eq56l0`). The package first took the oracle's, and the two cooperators of
  `tests/test_render.StoryTests` (told each other's goal, $w = 30$, the two-room scene, the
  automatic hands) showed why the paper's is the one to keep: under the level-0 likelihood
  the first step of a cooperator toward its partner's object -- a detour, under every
  hypothesis about its own goal -- is read as evidence about its own destination (the far
  candidates gain, whose discounted reward makes the level-0 policy flattest; from one move
  the belief went from uniform to 0.30 / 0.26 / 0.23 on three wrong candidates against
  0.12 on the right one), and the partner, carrying the cooperator's object, set it down on
  the wrong candidate twice and never finished; under Eq. 6 as printed the cooperation
  hypothesis predicts the step, the destination posterior keeps still until the set-downs
  settle it, and both finish at transition 5 as they did under the devices of before. A
  level-0 model has no social goal, so a level-0 likelihood reads every act for the robot as
  the partner's own physical goal.
- **Nothing else.** The likelihood is Eq. 7's softmax of the hypothesis model's Q at the
  state, at Eq. 7's tau, of the action the partner chose (Algorithm 1 takes the joint action
  as observed). Gone, and `Config` raises on them (`BELIEF_DEVICES`): the lapse
  (`belief_lapse`), the carry and park tables (`partner_delivers`, `partner_parks`), the
  "outcome" observation (`partner_observed`), the separate inference temperature
  (`inference_temperature`), the one joint posterior over (object, destination, social
  goal) of 2026-10-08..29 (`social_belief="joint"`), and the conditioning of the partner's
  estimate of the robot's goal on the robot's action inside Table I's $R_j$
  (`child_conditioning`: the authors' code reads the estimate as it stands). The
  likelihoods under every hypothesis of an action none explains (a hand action no level-0
  model would choose) leave the posterior where it was, but for what the transition term
  rules out; a robot lifting an object ("pick" the one action open to it) tells nothing.
- **The recursion** (unchanged in structure, checked again against brute force): level 0 a
  plain MDP per (robot, physical goal), the partner a moving obstacle whose hands do
  nothing; level 1 one model per (robot, physical goal, level-1 social term), its partner
  the PG level-0 models weighted by the physical posterior; level 2 the root, its partner
  the level-1 models weighted by the **product** of the two posteriors (Eq. 4 as printed,
  $P(\tilde g_j)\,P(\tilde\xi_{ji})$). Every level-1 model of a robot reads the same partner
  distribution (one row of P per robot and level, where there was one per physical goal),
  and a level-2 robot's hypotheses about its partner read the partner's belief about the
  robot from the same posterior its partner holds. After every transition every model of a
  level is solved in **one parallel batch** where the memory allows (`SocialMDP.batch`, no
  longer capped at 16), each to the Bellman-residual certificate, warm-started.
- Checked: `tests/test_beliefs.py` -- the update against the same arithmetic written out from
  the equations on plain probabilities, from the engine's own likelihoods, at several states
  and in a rollout transition by transition; the stage rule; the transition term case by
  case; `tests/test_models.py` -- every array the decisions rest on against brute force with
  the product weights and without the conditioning (`reference.Brute.level2(condition=
  False)`); `tests/test_inference.py`, `test_observation.py` and `test_stages.py` of the
  devices of before are retired.
- Output folders end in `_eq56` (`examples/run_object_inference.py`), so that `--resume`
  keeps no record of before.

**Where the oracle and the paper part, the paper was followed.** The oracle predicts a
partner of uncertain goal by `Policy::valueCompose`, $\exp(\sum_k p_k \log \pi_k)$ -- a
geometric mixture of the hypothesis policies, which, with $\pi_k \propto \exp(\tau V_k)$, is
the softmax policy of the belief-averaged value. Eqs. 3-4 of the paper take the expectation
over the hypotheses of the partner's action distribution, $\sum_k P(k)\,\pi_k(a_j \mid s)$,
and that arithmetic mixture is what the kernels compute (as they did before; `test_models`
checks it against brute force). The two agree where the belief is sharp and differ where it
is spread, the geometric mixture favouring the actions every hypothesis finds acceptable.
Nothing else was found to differ between the oracle and the equations but the two already
named: the level-0 likelihood of the oracle's physical-goal estimator (`Config.physical_likelihood`),
and the greedy action of Algorithm 1 against Eq. 7's softmax (`Config.action_selection`).

**A first rollout under the new update** (`tools/try_pairing.py competition cooperation
--scene two --steps 40 --tau 0.15`, the settings of the 49 of 2026-10-26; 828 s on two
cores, 20 s a transition): the same story as the record of 2026-10-26 under the devices of
before -- the cooperator picks the rival's object up at its fourth transition and carries it
toward the rival's destination, the rival takes it back out of its hands at the eighth,
carries it home and puts it back on its place (the 18th here, the 14th before), the
cooperator fetches its own object and finishes (22 here, 20 before), and the rival never
does. The rival has the cooperator as a cooperator at 0.91 from transition 7 and 1.00 from
11 (the record of before: 1.00 from 8). The cooperator holds the rival a competitor at 0.73
from transition 4 (a level-1 rival hurries more than a robot with no social goal, so the
rival's straight run for the doorway fits it best), 0.95 from 11, and a hinderer at 1.00
from the transition in which the rival puts its own object back on its place (18), where the
record of before ended, after 48 transitions, at conflict 0.59 and hinder 0.41: a hinderer
is the one hypothesis without a goal of its own (`hinder_own_goal=False`), so it is the one
that explains a robot that lets its own object go. The physical posteriors find the truth:
the cooperator's belief about the rival's destination is on the true candidate (1) from
transition 11, from the path the rival took with its object; the rival's belief about the
cooperator's object starts toward the rival's own object as the cooperator heads for it
(0.85 at transition 2, where the level-0 likelihood of the authors' estimator gives 1.00 at
once: under Eq. 6 as printed the cooperation hypothesis explains the step without the
cooperator wanting the object), and settles on the cooperator's own object as it goes to
fetch it (1.00 from 17), its destination on the true candidate (0) at the set-down (22).
The rival never finishes because, sure of a helper, it is the competitor of the hostile
table, $-R_j$, and a helper sure of the rival's goal gets $r(g_i)$ for the rival's progress
-- the rule of 2026-10-06 cancels the rival's own goal at $w = 1$ (goals.py, "rival"). The
belief is doing what the equations say; what the rival then does is the reading of
competition in force. The idle probability behaves as a two-mode filter does at $\tau =
0.15$: one "stay" of the rival at transition 6 sends the cooperator's belief that it stands
still to 0.59 (a stay is about $e^{-1/0.15}$ as likely under any goal as under none), three
stays in a row at 12-14 to 0.98, and its next move brings it back to the hazard. With the
authors' level-0 estimator instead (`--physical-likelihood level0`; 655 s) the run is the
same, action for action, as the record of 2026-10-26 through the cooperator's finish at 20,
the rival sure of its helper at 1.00 from transition 7 and the cooperator sure of a
hinderer from 15.

**What pure Bayes at Eq. 7's $\tau$ is like.** The devices of before softened the update
(a lapse of 0.05, an inference temperature of 0.5); without them the likelihood ratio of
one move at $\tau = 0.2$ is $e^{\Delta Q / 0.2}$, about $10^{-9}$ for a one-move detour, so
one transition settles a hypothesis for good and a later transition that contradicts it
cannot bring it back. Two selective cooperators on the tiny map, told each other's object
(`tests/test_objects.TwoSelectiveCooperators`, the automatic hands): at their first
transition each steps toward the other's object, which only cooperation, conflict and
hinder predict (the four other hypotheses fall to $e^{-21}$); at the second, agent_1, its
belief in a cooperative partner down to 0.34, turns toward its own object instead of
picking the partner's up -- a step every hypothesis left alive gives $e^{-11}$ or less, and
the ones that would have explained it are gone -- and agent_2 has it as a hinderer at 0.94
from then on; by the sixth each hands the other its object back (a cooperator that takes its
partner for a hinderer has no term about the partner's goal and lets the foreign object go
the cheapest way), each takes it for a hinderer at 1.00, and both finish by their own hands
(8 / 7). The hypothesis models are stationary -- one social goal, the belief of the moment
frozen -- while a level-2 robot changes course as its belief does, and the selective
cooperator's model stops predicting cooperation once the robot is believed to have given its
partner up, so its cooperative acts no longer count for cooperation. The equations have one
knob for this, $\tau$ (`Config.temperature`, `--temperature`), which governs the acting
policies as well; the 49 of 2026-10-26 ran at 0.15, sharper still. The lapse is one line
to put back as an option if wanted; it was taken out on 2026-10-30 at the user's choice.

## The model

Robot $i$ with destination $g_i$ and a social goal toward its partner $j$ earns per transition
(Eq. 2, with a weight $w_i$ on the social term; $w_i = 1$ is the paper)

$$R_i^l(s, a_i, a_j) = r(g_i) + w_i\,\xi_{ij}^l - c(a_i), \qquad
r(g) = \max\!\big(\rho\,(1 - D/\delta),\, 0\big),$$

with $D$ what is left of the way of the goal's object, $\rho = 1.25$, $\delta = 5$ m,
$c = 1$ per move and $0.1$ for staying, $\gamma = 0.99$ (all per metre-step, rescaled to the
grid). It plans with $Q_i^l = \mathbb{E}_{a_j}[R_i^l + \gamma V_i^l]$ (Eqs. 3-4), predicting its
partner one level down by $\tilde\psi_j^{\,l-1} = \mathrm{softmax}(Q_j^{l-1}/\tau)$ (Eq. 7) and
updating its belief about the partner's hidden destination after every transition (Eqs. 5-6).

**What a robot knows of its partner.** It sees where its partner stands and what each robot
holds, at every transition. Its partner's physical goal -- which object, to which of the K
candidate destinations -- and its social goal it does not know. A level-2 robot holds a
belief over them, uniform at the start over the M objects (its own included) times the K
destinations and over seven social goals: the paper's five, a partner without a social
goal, and a hinderer (`Config.social_hypotheses`). With `partner_object="known"` (the
default until 2026-10-09) it knows that its partner wants the partner's own object, and
the belief is over the K destinations only. It updates the belief by Eqs. 5-6 from
what the partner does -- since 2026-10-30 as two posteriors, as the authors' solver keeps
them: the physical one from the partner's level-1 hypothesis policies averaged over the
social posterior (Eq. 6 as printed; `Config.physical_likelihood="level0"`, the level-0
policies of the authors' code), in two stages by the partner's hands (since 2026-10-17:
while the partner holds nothing, about which object it wants only; while it holds an
object, about its destination only), the social one from the partner's level-1 hypothesis
policies averaged over the physical posterior, with the idle filter over both and Eq. 5's
transition term ("The belief update of 2026-10-30") -- and
its social term is Table I's column $l > 1$ averaged over their product. A
level-1 robot models its partner as a level-0 robot without a social goal, and holds the
physical posterior only. A robot is given its partner's weight and reasoning level. Conflict and `hinder`
are the same robot at level 1 (both $-w\,r(\tilde g_j)$), and a level-2 robot answers both
alike: it cannot tell the two apart, and holding both possible gives that model twice the
prior of each other one. These are the defaults since 2026-10-08. Until then each level-2
robot was told its partner's social goal (`partner_social_goal="known"`). `Config(**TOLD)`
restores the defaults of then (an exchanger at level 1 as Table I prints it, competition
read as `"race_then_hinder"`), and every section or table marked "told" below was made with
them.

Table I gives $\xi_{ij}$. Every cell is $\alpha \tilde R_j + \beta\, r(g_i) + \nu\, r(\tilde g_j)$:

| goal of $i$ | $l = 1$ | $l > 1$, by the partner's goal |
|---|---|---|
| cooperation | $r(\tilde g_j)$ | $\tilde R_j$ |
| conflict | $-r(\tilde g_j)$ | $-\tilde R_j$ |
| competition | $-r(\tilde g_j)$ | $\tilde R_j$ vs cooperation, exchange; $-\tilde R_j$ vs conflict, competition; vs coercion $\tilde R_j$ if $g_i = \tilde g_j$ else $-\tilde R_j$ |
| coercion | $r(g_i)$ | $\tilde R_j + r(g_i)$ vs cooperation, exchange; $-\tilde R_j + r(g_i)$ vs conflict, coercion; vs competition $\tilde R_j$ if $g_i = \tilde g_j$ else $-\tilde R_j + r(g_i)$ |
| exchange | N/A | $n\,r(\tilde g_j) + \tilde R_j$ vs exchange; N/A otherwise |

The table's N/A for exchange at $l = 1$ is `exchange_level1="none"`, and the authors'
code has it too. The text of the paper gives a level-1 exchanger "a copy of the other
agent's goal with a positive, but small, scalar n"; `exchange_level1="copy"` and
`"copy_unweighted"` are that reading, with the robot's weight on the copy and without it.
Where the robots have to infer each other's social goal this cell decides whether an
exchanger can be recognised ("An exchange scene"), and `"copy_unweighted"` has been the
default since inferring became the default (until then `"none"`).

Three things are not the table as printed.

**`hinder`** is not a row of it. It is the conflict entry of the column $l = 1$, kept as it is:

$$\xi_{ij}^{\text{hinder}} = -\,r(\tilde g_j), \qquad
R_i = r(g_i) - w_i\,\mathbb{E}_{k \sim b_i}\big[r_j(s'; k)\big] - c(a_i).$$

The robot is paid for keeping its partner's physical goal out of reach and for nothing else:
no partner reward $\tilde R_j$ (so nothing recursive, no partner action cost). `conflict`
stays what the table prints.

**A cooperator chooses whom it cooperates with**, `Config(cooperate_with=...)` (since
2026-10-11; `--cooperate-with` in the example scripts). The table gives a cooperator above
level 1 its partner's whole reward, $\tilde R_j$, whoever the partner is. Against a hinderer
or a conflict partner that reward holds the partner's wish that the cooperator fail, and at
$w = 1$ a cooperator stopped next to its own destination once the hinderer knew where it was
going ("Whom a cooperator cooperates with"). The default keeps the cell against a partner
taken to be cooperative, competitive or exchanging and puts 0 against any other;
`cooperate_with="all"` is the table as printed (`TOLD`).

**Competition is a switch**, `Config(competition=...)` (`--competition` in the example
script), because the table's competitor is not the one this project asks for:

| `competition=` | what a competitor does | reasons at |
|---|---|---|
| `"race_then_hinder"` (the default until 2026-10-08) | goes for its own goal; from the moment that goal is done, by itself or for it by its partner, it works against its partner | level 2 (since 2026-10-11; level 1 under `TOLD`); either if asked for |
| `"race"` | goes for its own goal; once that is done nothing in its reward concerns its partner, and it stops | level 2 (since 2026-10-11; level 1 under `TOLD`); either if asked for |
| `"table"` | Table I as printed: at $l > 1$ it helps a cooperative partner and works against a hostile one | level 2; level 1 if asked for |
| `"table_hostile"` | Table I with one cell turned: at $l > 1$ it works against a cooperative partner as well, $-\tilde R_j$ where the table prints $+\tilde R_j$. With known partner goals it is then a conflict robot, in every pairing without a coercing one | level 2; level 1 if asked for |
| `"rival"` (the default) | "I want to do better than you", not "I want you to fail": it races its partner, each for its own goal, and once its own is done nothing in its reward concerns its partner. The one partner it works against is a cooperative one, as under `"table_hostile"` | level 2; level 1 if asked for |

(The middle column is what a competitor of a race form does at level 1, and what one of the
other readings does at level 2. What a competitor of a race form does at level 2 is "A
competitor at level 2".)

The two race forms are one term, with $\lambda = 1$ for `"race_then_hinder"` and
$\lambda = 0$ for `"race"`:

$$\xi_{ij}^{\text{race}} =
\begin{cases}
r(g_i) + \lambda\,\big(r(g_i) - r(\tilde g_j)\big) & \text{once } g_i \text{ is done} \\
-\,r(\tilde g_j) = -\rho & \text{while } \tilde g_j \text{ is done and } g_i \text{ is not} \\
0 & \text{while neither is done.}
\end{cases}$$

"Done" is the finish line: the object lies on its owner's destination. For the partner that
is judged under each hypothesis about its destination, and $r(\tilde g_j)$ is the
expectation over the robot's belief, as everywhere.

*Until its own goal is done* a competitor of either form is paid nothing for what its partner
does on the way, and loses $w\rho$ per transition for as long as its partner is there first.

*Once its own goal is done*, `"race"` pays it $w\rho$ per transition whatever happens next:
there is nothing left for it to act on. `"race_then_hinder"` pays
$w\,(2\rho - r(\tilde g_j))$, which is the term of `hinder` plus a constant. **From that
moment the competitor is a hinderer**: its rewards are a hinderer's plus $2w\rho$, its values
and Q a hinderer's plus $2w\rho/(1-\gamma)$, in every state and for every action, so it
chooses what a hinderer chooses (`test_models.py` checks this on the solved models).
$-r(\tilde g_j)$ is also what Table I prints for competition at $l = 1$; here it holds from
the robot's own finish on, not from the start.

Why this shape:

1. *The table's competitor does not want its own goal.* At $l = 1$ its entry is the conflict
   entry, which is `hinder`: such a robot takes its partner's object at transition 2 and
   never does its own task. Two of them at levels (2, 2) plan against level-1 models that
   neither is; neither goal is ever done, and at $w = 1$ one of them takes 15 moves back in
   53 transitions.
2. *Against a cooperative partner the table's competitor cooperates.* Its entry is
   $+\tilde R_j$, the cell of the cooperation row. With
   $\tilde R_j = r(\tilde g_j) + w_j\,\tilde r(g_i)$ for a cooperative partner
   ($\tilde r(g_i)$: its estimate of the competitor's goal reward) the competitor earns
   $r(g_i) + w_i\,r(\tilde g_j) + w_i w_j\,\tilde r(g_i) - c(a_i)$: it is paid for its
   partner's goal, and at levels (2, 2) sets its partner's object down on its partner's
   destination (both goals done at transition 5, for $w \ge 1$).
3. *Before the finish only the finish line counts.* A term on the partner's progress makes a
   robot answer every step of its partner: a variant "$-r(\tilde g_j)$ until the own goal is
   done" (a prototype, not in this code) also raced, but a competitor whose object a
   hinderer held paced next to it, 23 moves taken back in 35 transitions at $w = 30$.
4. *The own goal counts twice once it is done.* With $r(g_i) - r(\tilde g_j)$ there (a
   prototype, not in this code) a competitor on the map with the walled-in doorway (see
   "Results") went for its partner's object first, at $w$ = 30 and 100, and handed it over
   later. With
   $2\,r(g_i) - r(\tilde g_j)$ finishing is worth more than spoiling: in every greedy run
   here at its default level a competitor touches its partner's object only after its own
   goal is done.

`"race"` and `"table"` reproduce the engine as it was before `"race_then_hinder"` existed,
bit for bit (actions, events, Q, beliefs, expected rewards), on 40 rollouts: 21 with
`"race"`, 19 with `"table"` (13 of those with a competitor).

**`"table_hostile"`** is Table I with one cell turned. It answers one thing, point 2 above:
a level-2 competitor that sees a cooperative partner should still work against that
partner's reward, not for it. The table prints one cell for a cooperative and an exchanging
partner, and both halves are turned; nothing else is.

| the competitor's partner | `"table"`, $l > 1$ | `"table_hostile"`, $l > 1$ |
|---|---|---|
| cooperation, exchange | $+\tilde R_j$ | $-\tilde R_j$ |
| conflict, competition (and `hinder`, no social goal) | $-\tilde R_j$ | $-\tilde R_j$ |
| coercion | $+\tilde R_j$ if $g_i = \tilde g_j$, else $-\tilde R_j$ | the same |

At $l = 1$ the entry is $-r(\tilde g_j)$ under both. Against a cooperative partner, whose
reward is $\tilde R_j = r(\tilde g_j) + w_j\,\tilde r(g_i)$, the competitor now earns

$$R_i = r(g_i) - w_i\,r(\tilde g_j) - w_i w_j\,\tilde r(g_i) - c(a_i):$$

it is paid for keeping its partner's goal out of reach, where the table pays it for that
goal. (The third term is the table's recursion with the sign turned: the competitor loses
$w_i w_j$ times what its partner takes it to have reached of its own goal.)

With the cell turned, the competition row of the table is its conflict row, except against
a coercing partner that is taken to have the robot's own destination; and at $l = 1$ the
table gives competition the conflict entry anyway. So **under `"table_hostile"` a competitor
is a conflict robot** in every pairing without a coercing robot, as long as the partner's
social goal is known (told): its models are the same arrays (`test_models.py`) and
its rollouts the same rollouts ("Results"; in `test_competition.py` Q value for Q value).
With an inferred partner goal the untouched coercion cell is in the competitor's mix of
hypotheses: its Q is then not a conflict robot's, though in the four runs made it acts as
one. Point 1 above stays: nothing in the term wants the robot's own goal, and two such
competitors never finish.

**`"rival"`** is competition as doing better than the partner, which is not the same as
wanting the partner to fail. Its term is the race term above with $\lambda = 0$: a rival
looks after its own goal and loses $w\rho$ per transition while its partner is done and it
is not; once its own goal is done, nothing in the term concerns its partner. That holds at
every level and against every partner but one, a cooperative partner. Against a partner
that helps it a rival is the competitor of `"table_hostile"`: it does not return help, it
works against its helper's reward.

| the rival's partner | $l = 1$ | $l > 1$ |
|---|---|---|
| cooperation | $-r(\tilde g_j)$ | $-\tilde R_j$ |
| anybody else: competition, conflict, coercion, exchange, `hinder`, no social goal | $\xi^{\text{race}}$ with $\lambda = 0$ | $\xi^{\text{race}}$ with $\lambda = 0$ |

What a rival is depends on whom it faces, as in the table, whose competition row has a
cell per partner goal. With known partner goals (told) the reading is a switch
between two others: the models of a pairing are the models of `"table_hostile"` where a
rival faces a cooperator, and the models of `"race"` at the same levels where it faces
anybody else (the same arrays, `test_models.py`; the same rollouts, "Results"). So against
a cooperator a rival is, at $l > 1$, a conflict robot and, at $l = 1$, a hinderer: that
half of it is not "I want to do better than you". It is there because this project asked
for it on 2026-10-06, for a cooperative teammate ("make it negative so we still minimize
their reward"); the request of 2026-10-07 that a competitor should not hinder was made of
two competitors. If it should hold against a cooperator too, the reading is `"race"`,
which is `"rival"` without its one exception ("Results", the last table). At level 1, its
default under `TOLD` (and until 2026-10-11), both goals are done in all eight runs against a cooperator, and the competitor
never holds the cooperator's object (the suite checks that at $w$ = 1 and 30, in either
seat); two of them finish at 9 / 11. Asked for level 2 (`levels=(2, 2)`), two of them
still finish at 9 / 11 and a competitor seated first leaves a cooperator's object alone
(5 / 17 at every weight), but seated second at $w \le 2$ it takes it and neither goal is
done, as with a level-2 robot without social goal in that seat ("A competitor at level
2").

What `"rival"` is for is two competitors: under the table two of them at level 2 never
get either goal done ("Why two competitors of the table ping-pong"); two rivals each do
their own task.

Four things in it are choices, not the table:

- *The one partner it works against is a cooperative one.* The table prints one cell for
  a cooperative and an exchanging partner, and `"table_hostile"` turns both halves. An
  exchanging partner has no social term toward a competitor (its cell there is N/A): it
  does nothing for a rival, and a rival that worked against it would be hindering a
  robot without a social goal. A rival races it.
- *It races everybody else as well*, a conflict robot, a coercing one and a hinderer
  included, where the table has $\pm\tilde R_j$.
- *A level-1 rival is told whom it faces.* The table's column $l = 1$ has one entry per
  goal, whatever the partner: a level-1 robot takes its partner to have no social goal. A
  rival's term needs the partner's. With `partner_social_goal="known"` it is the
  partner's goal as it counts (none for a partner at level 0 or with weight 0, whatever
  goal it was given; the partner stays a level-0 robot in the rival's model); with
  `"inferred"` a level-1 rival has nothing to go by, and races. One level-1 term for
  every partner does not work: a level-2 cooperator plans against the level-1 model of
  its partner, and if that model is a racer it is a robot that never touches the
  cooperator's object, while the rival the cooperator faces, paid $-\tilde R_j$, takes
  it. In that prototype (not in this code) competition / cooperation ended – / –,
  16 / –, – / –, 5 / – at $w$ = 0.8, 1, 2, 30, where `"table_hostile"` gives 7 / –,
  5 / –, 5 / –, 5 / –; at $w = 2$ the cooperator picked the rival's object up and put it
  back for the whole rollout (51 uses in 52 transitions). With the race term and its
  lead ($\lambda = 1$) against a cooperator in the place of $-\tilde R_j$ it was – / –,
  – / –, – / –, 5 / –. The level-1 model of a rival has to be the kind of robot the rival
  is against that partner.
- *Its default level is 2*, whoever the partner is, because its term can hold
  $\tilde R_j$. Against a rival the term is the same at both levels; level 2 is then the
  level at which a rival predicts its partner by its social goal, as a robot that races.

**Levels.** `levels=` of `run_pairing` (`--levels` in the example script) sets each robot's
reasoning level. Left out, every robot with a social goal reasons at level 2, the level of
the paper's experiments: a hinderer and a competitor of either race form too, since
2026-10-11 (the user: "Whenever I say run it, always assume both are level 2";
`Config.level_free_level`, `default_levels`). Their terms hold no $\tilde R_j$: that is the
table's column $l = 1$, the column of a robot that takes its partner to pursue its physical
goal and nothing else. So for them the level changes whom they take their partner to be,
not what they are paid for: at level 1 a robot without social goal, at level 2 a robot with
one of the social goals held possible ("Every robot with a social goal at level 2"). Until
2026-10-11 they reasoned at level 1 by default, where a partner one level up models them
exactly, and a hinderer was held to level 1 when a higher level was asked ("What hindering
rests on", line 3); `level_free_level=1` gives both back, and so does `TOLD`. Levels that
are asked for are the levels used (`levels_used`): `levels=(2, 2)` puts a competitor at
level 2, with the same term (see "A competitor at level 2"), and a level-2 competitor whose
own goal is done is a hinderer at level 2. A robot with a physical goal only -- no social
goal (`"none"`), or weight 0 whatever goal it was given -- reasons at level 0 by default
since 2026-10-08: a plain MDP that takes its partner for a moving obstacle, with no social
stance and nothing inferred (`Config.physical_level`; "both at level 2" is read as both
robots with a social goal). Until then its default level was 2, and it predicted its
partner by the partner's level-1 models; `physical_level=2` gives that back, and so does
asking for the level. Under the two readings of the table a competitor's term holds
$\tilde R_j$ above level 1, and its default level is 2; so is a rival's, whose term holds
$\tilde R_j$ against a cooperator.

*Changed on 2026-10-06.* Until then a competitor of a race form was held to level 1
whatever was asked, as a hinderer was until 2026-10-11. Levels asked for with such a competitor,
`levels=(2, 2)`, `(2, 1)` or `(1, 2)`, `--levels 2 2`, now mean what they say; without
`levels` nothing changes ("How it is checked"). For the behaviour of before, ask for level
1 in the competitor's place (`(1, 1)` where `(2, 1)` was asked with the competitor first);
leaving `levels` out gives what `(2, 2)` gave. A record now holds the levels that were
used under `levels` when none were asked for ((1, 2) where it held (2, 2) before), the
example script's `summary.json` then says `"levels": "default"`,
`levels_used((competition, competition), (2, 2))` is (2, 2) where it was (1, 1), and
`SocialMDP` builds the level-2 model of a competitor that races, where it raised an error.

## The world

A grid, two robots, one object each, and K candidate destinations of which a robot knows only
its own. Robot $i$'s goal is done when *its* object lies on *its* destination, set down by
itself or, for it, by its partner, and by nothing else: object $i$ completes robot $i$'s goal
and no other. That is the world, not an option. An object is in one of three places: on its
own place, in a pair of hands, or on its owner's destination. A robot can pick up its
partner's object, carry it, hand it to its partner, put it back, or set it down on its
partner's destination; it cannot set it down on its own destination or anywhere else. Ten
actions: four moves, stay, and five hand actions (since 2026-10-16; until then six, with one
action for the hands, "use", that did the first of these that applied -- "Five hand actions
in place of "use""):

- **pick** -- free hands on the place of an object: pick it up
- **place** -- holding X on X's place: put it back; on a candidate destination: set X down
  for the partner. That works only on the partner's real destination; on any other
  candidate, the robot's own destination included, nothing is set down and X stays in its
  hands (a failed set-down: the robot has learnt that this is not its partner's destination)
- **give** -- holding X next to the partner: hand X over, if the partner's hands are free,
  or if the partner gives in the same transition, in which case the two objects are swapped
- **take** and **swap**: see `steal` and `swap_on_place` below

A hand action is offered only where it does something (the others have no Q value), and
each has its own Q and its own probability in Eq. 7's softmax: a robot predicts, and reads,
whether its partner will pick up, place, take, give or swap.

Automatic: a robot takes its own object on arriving at its place and sets it down on arriving
at its destination. A robot holds one object at a time. While a robot's object is in its
partner's hands its goal reward is 0: taking the object is what hindering looks like here.

**Where the partner's object is inferred** (`partner_object="inferred"`, the default since
2026-10-09) a robot does not know which object finishes its partner's goal, and the rule is
open to every object: a robot may pick up any object, put it down on a hand-off cell, hand
it over, or set it down on a candidate destination for its partner. Only the partner's own
object on the partner's destination finishes the partner's goal, and a partner takes only
its own object: a hand-over of another object and a set-down of another object, or on
another candidate, do nothing (the robot keeps what it holds, and has learnt that its
partner does not want that). The robot's own goal is as before: its own object on its own
destination. With the object known nothing changes: a robot offers its partner nothing but
the partner's own object.

Two things a scene may add, and `exchange_scene()` has both ("An exchange scene"):

- **Hand-off cells** (`Scene.handoff_cells`), a fourth place for an object to be. A robot
  standing on one with its partner's object in its hands can put it down there ("place").
  The object lies on the cell until somebody picks it up: by "pick", or, its owner, on
  arriving, as at the object's own place. Its owner's way, and so its goal reward, is
  counted to that cell and on to the destination. Any number of objects may lie on one
  hand-off cell, and a robot that picks up takes its own first. Putting down is an act
  for the partner, as handing over and setting down for it are: it brings the partner's
  object within the partner's reach, and only a robot with a social goal may choose it.
  (Picking up and putting back are open to every robot.) On a hand-off cell next to its
  partner a robot may put the partner's object down or give it (until 2026-10-16 "use"
  put it down there, and nothing was handed over from a hand-off cell). A hand-off cell
  can be neither an object's place nor a candidate destination.
- **Rooms** (`Scene.rooms`): for each robot the cells it may stand on. A robot never leaves
  its room: a move out of it is no move of that robot, as a move into a wall is. A cell
  can belong to both rooms (a doorway). A robot's ways are counted inside its room, so its
  goal reward is 0 while its object lies where it cannot go, and a robot whose object or
  destination lies outside its room cannot finish alone.

And two options of the rule (`Config`): `set_down_for_partner=False`, under which nobody
sets its partner's object down on a destination, and `hand_over=False`, under which
nothing goes from hand to hand. With both, an object reaches its owner only over a
hand-off cell.

Two more, since 2026-10-12, both off by default, give an object two more ways to change
hands ("Swapping on a place and taking from the partner's hands"):

- `swap_on_place=True`: a robot holding X on the place of an object, where an object Y
  lies (the place's own object, or one left there), swaps them with "swap": Y into its
  hands, X onto that cell; on X's own place that puts X back. Objects may then lie on
  each other's places, one to a place, so each place is one more status of every object:
  with two objects 29 configurations instead of 20, with three 181 instead of 56, with
  four 1,321 instead of 126. While another object lies on X's place, X cannot be placed
  there (until 2026-10-16 the swap came first among the uses of full hands). A robot that
  arrives with free hands where its own object lies takes it, on another's place as on its
  own.
- `steal=True`: a robot with free hands next to its partner after the moves takes
  whatever the partner holds with "take" -- unless the partner lets go of it in the same
  transition (places it, gives it, swaps it). Without it nobody is made to let go of what
  it holds (`forced_swap` aside).

Both are open to every robot, one without a social goal included, as picking up and
putting back are: neither brings an object within the partner's reach.

## The figure

`render_video(result, path)` draws the record as an MP4 video (since 2026-10-17; `render_gif`
as a GIF, the same frames), `save_frame(result, path, frame)` one frame of it, and
`render_videos(result, stem)` the video and, with it, each robot's value video ("The value
videos" below).

| panel | what it shows |
|---|---|
| headline | the pairing, a competitor named with its reading of competition ("competition (race, then hinder)"), the step, and in a highlighted box what the last transition did to an object ("agent_1 placed O1 on its goal") |
| map | robots (numbered), objects O1..O6, their places, candidate destinations G0..G5 with the two real goals named. Each pick-up, put-back, put-down on a hand-off cell, hand-over, failed and successful set-down is called out at its cell in words for three frames; a set-down that finishes a goal gets three rings, the star of that destination is filled from then on, and its label reads "O1 delivered ✓". A robot's move is an arrow (grey if blocked), "stay" and the hand actions ("pick up", "place", "take", "give", "swap") are tags. Where the scene has them, each robot's own room is tinted in its colour and a hand-off cell is framed with a dashed line and labelled (both are named in a second row of the legend) |
| status | per robot: its action in this step and what comes of it ("pick up: picks up O2", "give: nothing happens"), what its hands hold and whose it is, the last thing it did to an object, and its goal: NOT DONE with where its object is (at its place, in whose hands, or on the hand-off cell), or ✓ DONE with who placed it and when |
| beliefs | each robot's belief about its partner's destination (outlined: the true one) and, where it has to infer which object its partner wants (`partner_object="inferred"`), about that, in the same panel: O1..OM, then G0..G5, each part with its own uniform line, the partner's own object outlined. Where the robots have to infer each other's social goal (`partner_social_goal="inferred"`) a second panel next to it shows each robot's belief about its partner's social goal, over the goals it holds possible, the true one outlined where it is one of them |
| values | every action's value relative to the one taken, physical and social part; an act for the partner (give, and place where it puts the partner's object down or sets it down for the partner), which the physical model does not have, as one dotted bar of its whole value (since 2026-10-16 also where it is not the action taken) (`values="raw"`: levels) |
| timeline | per robot, for every step so far: the action (↑ → ↓ ← move, • stay, P pick up, D place, T take, G give, S swap), a band while it carries an object (coloured by whose object it is), and the events (▲ picked up, ▼ put down, ◆ changed hands, ✗ in vain: a set-down or a hand-over for the partner that did not count, ★ goal done) |

A frame "step t" is the state after t transitions with the actions chosen there. Frames after
an event are held 1.5 s, after a set-down that finishes a goal 2.5 s, the last frame 3 s
(`event_hold`, `placed_hold`, `end_hold`); the others run at `fps` (3).

The legend of the map and the labels of destinations on its border are written on the wall
that surrounds the map; a map without such a wall gets room around it for them. With
`forced_swap=True` a robot can lose an object without using its hands: the figure then says
it "had to give" the object, not that it handed it over. The figure is set in DejaVu Sans,
which ships with matplotlib, so that it looks the same on every machine (`font=None`: the
font matplotlib is configured with); text that has to fit somewhere is measured, and wrapped
or scaled to fit. An act for the partner in vain says what was wrong with it: "tried to place
O1 (not agent_2's destination)", "(not agent_2's object)" or "(neither agent_2's object nor
its destination)", "tried to hand O3 to agent_2 (not agent_2's object)".

**The video** (since 2026-10-17; GIFs until then): H.264 in an MP4, which plays in browsers,
slides and every player, at twice `fps` frames a second, a frame that the GIF held longer
repeated for as long (`crf` 18: the text stays sharp; a 55-transition rollout is about 1 MB).
`render_gif` still writes the GIF. Needs `imageio-ffmpeg`, which brings its own ffmpeg.

**agent_1 is green, agent_2 red** (the user, 2026-10-17; orange and a lighter red until
then): `#2ca25f` and `#b2182b`, a lighter green and a deep red, so that red-green
colour-blind eyes tell them apart as well (a deuteranope sees them 13 Delta E apart, where a
plain green and red would be 4); the robots carry their numbers as well. Everything of a
robot's -- its object, its destination, its room, its belief bars, its row of the timeline --
wears its colour.

### The value videos (since 2026-10-17)

The user, on 2026-10-17: "output 2 other videos from now on which are their value iteration
grid throughout the video. And maybe a break down in the respective videos of how this cost
is calculated at each cell so I can see what is influencing it."

`render_videos(result, stem)` writes, next to `<stem>.mp4`, one video per robot,
`<stem>__values_agent_1.mp4` and `..._agent_2.mp4` (`render_value_video(result, path,
robot)`), from the value maps the record holds (`Config.value_maps`, the default; a second
a robot a transition on the two-room scene): one frame per transition, a second each
(`fps`), the last held 3 s. A frame shows the robot's root model, solved as it was at that
transition, read off at every cell the robot could stand on -- its partner and the objects
where they are, everything else as it is -- and what that value is made of:

- **V**: the value of standing at each cell, with the robot's greedy action there (an arrow
  for a move, a dot for staying, P / D / T / G / S for a hand action), its own cell framed in
  its colour, its partner a numbered dot. The partner's cell, the cells outside its room and
  the cells the model is not solved on are blank.
- **one map per term of its reward** (Eq. 2): its own goal reward $r(g_i)$, its action
  costs $-c(a_i)$, and each social term it has -- at level 1 $w\,a\,r(\tilde g_j)$
  ("partner's goal"), $w\,b\,r(g_i)$ ("own goal in the term", coercion), $w\,c\,$race; at
  level 2 $w\,\alpha\,\tilde R_j$ ("partner's reward"), $w\,\beta\,r(g_i)$,
  $w\,\nu\,r(\tilde g_j)$, and the race term. The value of a policy is linear in the
  reward, so V is the sum of the terms' values, each the discounted sum of that term alone
  under the robot's greedy policy and its prediction of the partner (Eq. 4):
  $V = \sum_k (I - \gamma P_\pi)^{-1} R_k$. A term that pays nothing anywhere is left out,
  and so are a hinderer's own goal reward and the $-r(g_i)$ of its term, which cancel
  (`hinder_own_goal=False`). That is decided step by step, so the terms of a robot can change
  along a rollout -- a rival that rules out that its partner cooperates loses its
  $-\tilde R_j$ term (it pays nothing against any other partner): the video has a panel
  for every term of any step, which shows 0 and "pays nothing in this step" in the steps
  without it. (Until the evening of 2026-10-17 the figure took its panels from the first
  step and stopped half way through such a video, hinder / competition among the five
  pairings run that night.) The maps add up to V to the solver's tolerance and the rewards'
  single precision (the subtitle says to what; `value_maps[i]["residual"]` in the record).
- Every map has its own colour scale over its own range in that frame (the numbers are in
  the cells), so that the differences between cells -- what moves the robot -- show,
  whatever the level of the term.

How it is computed (`SocialMDP.value_components`): the physical part of the reward is the
model's reward recomputed with the social coefficients zero (`_input_args`' `terms`), the
costs are $-c(a)$ exactly, and each social term is the reward recomputed with that term's
coefficients alone, less the physical part; the greedy policy and its transition graph come
from the solved values (`kernels.policy_graph`), and each term's value from one sparse LU of
$I - \gamma P_\pi$. The record holds, per transition and robot, `value_maps`: the names and
labels of the terms, `total` (N,), `parts` (K, N), `action` (N,), the partner's cell and the
residual; `scene["cells"]` gives each cell's coordinates. A level-0 robot (a physical goal
only) has no reward table and no maps. `value_maps=False` leaves them out; a record written
before has none, and `--resume` of the example scripts treats it as another configuration.
A video is written to `<name>.part.mp4` and renamed once whole, so one cut off half way
leaves nothing behind, and `--resume` draws a rollout's videos again unless all three are
there (`render.video_paths`).

## Which object the partner wants (since 2026-10-09)

The user, on 2026-10-09: "Modify it such that the pickup objects are always inferred as per
the paper." A physical goal is an object and a destination ("Physical goals consist of
moving the desired objects to one of the landmarks"), and a robot's estimate of its
partner's starts from a uniform distribution over them (Sec. III-B, Eq. 6). So a robot no
longer knows which object its partner is after (`Config.partner_object="inferred"`, the
default): its belief is over every object in the model, its own included, to every candidate
destination, and the spare objects are candidates too.

$$\tilde g_j \in \{(q, k) : q = 1..M,\ k = 1..K\},\qquad
b_i^{(0)}(q, k, t) = \frac{1}{M K T}$$

for a level-2 robot with $T$ social goals held possible ($M K$ physical goals for a
level-1 robot). Robot $i$'s own goal is $(i, g_i)$; Table I's "$g_i = \tilde g_j$" is the
same object *and* the same destination. `partner_object="known"` (and `TOLD`) gives back
the model of before, in which each robot knew that its partner wanted its own object and
inferred only the destination: the same arrays, actions, beliefs and events as before, to
the last bit, on six pairings ("How it is checked").

**What follows for the world.** Every object is in the model (`handle_spares` no longer
matters), and a robot may handle every object: pick it up, carry it, put it down on a
hand-off cell, hand it over, or set it down on a candidate destination for its partner.
Only the partner's own object on the partner's destination finishes the partner's goal, so
an act for the partner can now be in vain in two ways: the place is not the partner's
destination, or the object is not the partner's. Nothing happens either way, the robot
keeps the object, and its belief keeps only what the act left possible:

| what happened | the hypotheses it rules out |
|---|---|
| a set-down for the partner on cell $c$ that did not count | $(q, k)$ with $q$ the object held and $k$ = $c$ |
| a hand-over that the partner did not take | every $(q, k)$ with $q$ the object held |
| the partner's goal done | every $(q, k)$ but (its own object, its destination) |

In a robot's own models the partner hands over, takes and sets down under each hypothesis
as that hypothesis has it: an act for the partner counts with the belief mass of the
hypotheses that take it (`ModelBank.pd` for a set-down, `ModelBank.ho` for a hand-over).

**The partner's estimate of a robot's reward.** A level-2 robot's term holds $\tilde R_j$,
in which a partner of some social goals counts the robot's goal reward as the partner
estimates it, over the goals the partner holds possible for it. In the robot's own model its
done flag means its own object delivered. Under "it wants its own object, to some
destination" the estimate reads the flag as before. Under "it wants object $q \ne i$" it
reads where $q$ is (`ObjectWorld.cls_est`), not the flag: a robot's own finish is no goal
of another object done. (Found during the change: read off the flag, a robot's own finish
could raise its partner's estimate of its reward from 0 -- a goal of an object already
delivered elsewhere, out of reach for good -- to $\rho$ at once. On the 3 x 3 map a
cooperator that took its partner for a hinderer or a competitor, whose model of it held it to
want the partner's own object, saw its finish cost its partner as much, and stood one move
from its destination with its object from transition 15 to the end; now both finish, 12 /
21.) With the partner's object known nothing changes.

**What it costs.** Every hypothesis about the partner is a level-1 model solved on every
joint state, and with $M$ objects there are $M$ times as many; the object configurations
grow with $M$ as well.

| scene | objects | physical goals per robot | joint states | level-1 models | estimate |
|---|---|---|---|---|---|
| two rooms, objects known | 2 | 6 | 10,976 | 72 | 0.09 GiB |
| two rooms, inferred | 2 | 12 | 15,680 | 144 | 0.23 GiB |
| tier scene (house, 0.5 m), inferred | 2 | 12 | 124,820 | 144 | 1.8 GiB |
| six objects, inferred, `max_objects=3` | 3 | 18 | 43,904 | 216 | 0.9 GiB |
| six objects, inferred, `max_objects=4` | 4 | 24 | 98,784 | 288 | 2.8 GiB |
| six objects, inferred | 6 | 36 | 326,144 | 432 | 13 GiB |
| exchange scene, inferred, `max_objects=3` | 3 | 18 | 94,041 | 216 | 2.0 GiB |
| exchange scene, inferred | 6 | 36 | 3,756,537 (all 5,153 configurations) | 432 | the tables alone 13 GiB |

(Both robots at level 2 with seven social goals held possible; "estimate" is what the
solver's arrays need, `SocialMDP.memory_bytes`.) A model that cannot fit fails at once with
a `SolverFailure` that says so: the solver never plans with more memory than the machine
has, or the limit of its control group (`max_memory_gib` caps it further). So on a machine
with 8 GB the six-object and exchange scenes run with the objects inferred only with
`max_objects=3` or 4 (the two own objects, then the spare ones nearest somebody's task; the
rest are scenery), or with `partner_object="known"`. Time grows faster than memory: on the
two-room map a transition takes three to twelve times as long as with the object known (more
models, more blocks in use, and up to twice the policy-iteration and Krylov iterations per
solve).

**Results** (made with Table I's cooperator, `cooperate_with="all"`, the default until
2026-10-11; the cooperator of now: "Whom a cooperator cooperates with"). Five pairings on the
two-room scene, whose only objects are the robots' own
(`python examples/run_object_inference.py --cooperate-with all`, $w = 1$, at most 60 transitions, written to
`output/object_inference_two_w1_1/` with a GIF per rollout, `results.md` and `results.png`,
each robot's belief that its partner wants the partner's own object, transition by
transition), and six on the 3 x 3 map of the tests (`TINY` of `tests/test_models.py` as a
`Scene`, `resolution=1.0`, at most 25 transitions; the first is pinned in
`tests/test_objects.py`):

| two-room scene | object known | inferred | wall time, known / inferred |
|---|---|---|---|
| cooperation / cooperation | 22 / 5 | 20 / 12 | 25 s / 273 s |
| competition / cooperation | 5 / – | 5 / – | 35 s / 152 s |
| competition / competition | 26 / 28 | 9 / 11 | 138 s / 178 s |
| hinder / cooperation | 5 / – | 5 / – | 13 s / 76 s |
| none / none (level 0) | 9 / 11 | 9 / 11 | 1 s / 2 s |

| 3 x 3 map | object known | inferred |
|---|---|---|
| cooperation / cooperation | 7 / 5 | 12 / 21 |
| competition / cooperation | 5 / 11 | – / – |
| hinder / cooperation | 5 / – | 9 / – |
| exchange / exchange | 5 / 5 | 5 / 5 |
| coercion / conflict | – / – | 5 / 5 |
| none / none (level 0) | 5 / 5 | 5 / 5 |

- *Helping looks like wanting.* Two cooperators each pick up the other's object at
  transition 2, and each reads the other's pick-up as "it wants my object": the belief that
  the partner wants its own object drops from 0.5 to 0.01 in one transition, for both.
  agent_1 is sure of agent_2's object again at 12 and sets it down on agent_2's
  destination; agent_2 puts agent_1's object back at 10, is sure of agent_1's at 16, and
  delivers it at 20. Told the object, agent_1 delivers at 5; agent_2, which does not know
  agent_1's destination, tries two wrong ones, hands the object over at 17, and agent_1
  delivers it at 22.
- *Two rivals leave each other's objects alone.* Told the object, each takes the other's at
  2 and puts it back, two or three times, before both deliver their own (26 / 28); not told, a
  rival has no object to race for but its own, and the two finish as two robots without
  social goals do (9 / 11), each ending at 0.49 on competition.
- *Against a hinderer or a rival the cooperator still never finishes*, but its object is
  taken later: at 2 told, at 11 by the hinderer and at 10 by the rival not told -- the
  hostile robot first has to see which object the cooperator wants. The cooperator delivers
  the other's object at 5 either way.
- *Every robot that holds a belief ends sure of its partner's object* (1.00, the hinderer
  0.93); of the destination only where the partner got near it: a hinderer or a rival ends
  at 0.15 to 0.17 on the cooperator's, which never got its object back.
- On the 3 x 3 map the robots are always in each other's way, and inferring the object
  makes it worse in two pairings: two cooperators finish at 12 / 21, and a cooperator next
  to a rival holds the rival's object for twenty transitions (handing it over looks bad to
  it: its frozen belief is split between two destinations, and the rival holding the object
  can stand on only one of them, while a set-down that counts has the partner done under
  every hypothesis) -- and better in one: coercion / conflict finishes at 5 / 5, where told
  the coercer picks the conflict robot's object up and puts it back again and again.
- Time per transition on the two-room map: twelve times as long as told for two
  cooperators, three to five times in the other pairings with a social goal.

With one spare object as well (`--scene spare`: O3 on (2, 1), in agent_1's room; three
objects, 18 physical goals per robot; `output/object_inference_spare_w1_1/`, 78 minutes for
the ten rollouts on 2 cores), told the object nothing changes (the spare is scenery), and
inferred:

| two-room scene, one spare | object known | inferred |
|---|---|---|
| cooperation / cooperation | 22 / 5 | 29 / – |
| competition / cooperation | 5 / – | 5 / – |
| competition / competition | 26 / 28 | 9 / 11 |
| hinder / cooperation | 5 / – | 5 / – |
| none / none (level 0) | 9 / 11 | 9 / 11 |

- *Two cooperators lose each other over the spare.* agent_1 picks the spare up and puts it
  back three times (3 to 28), agent_2 tries agent_1's object on three wrong candidates
  before delivering it at 29; then agent_1 picks agent_2's object up (34) but by then takes
  agent_2 to want the spare (0.99 at the end: agent_2 stands on the spare's place, waiting
  for its object), and carries it to the end without delivering it.
- *The hinderer takes the spare first*: it picks the spare up at 12, holds it until 47, and
  only then takes the cooperator's own object (51). The rival never takes either: its belief
  stays split between the cooperator's object and the spare (0.52 / 0.48), and it parks by
  the doorway, where the cooperator cannot get past for the rest of the rollout.

## Whom a cooperator cooperates with (since 2026-10-11)

The user, on 2026-10-11: "Ok so lets modify our algorithm to only cooperate with
competitive, cooperative, and exchange agents."

**Why.** Above level 1 Table I gives a cooperator its partner's whole reward, $+\tilde R_j$,
whoever the partner is, and $\tilde R_j$ holds the partner's own social term. Against a
hinderer or a conflict partner that term is $-w_j\,\hat r_j(g_i)$, the partner's estimate of
the cooperator's own goal reward, $\hat r_j(g_i) = \sum_{g'} \tilde b_j(g')\,r_i(g')$ over what
the partner believes the cooperator wants (no partner action cost: `social_cost=False`). So

$$R_i = \underbrace{\big(1 - w_i w_j\,\tilde b_j(g_i^*)\big)}_{\beta}\,r(g_i^*)
\;+\; w_i\,r(\tilde g_j) \;-\; c(a_i) \;-\; w_i w_j \sum_{g' \ne g_i^*} \tilde b_j(g')\,r_i(g').$$

$\beta$ is what its own goal is worth to the cooperator. At $w = 1$ it falls to 0 as the
hinderer becomes sure of that goal: finishing is then worth nothing, and staying costs a
tenth of a move. On the center scene (objects inferred, 2026-10-10: both robots' objects on
the doorway cells, a spare in each room, `run_object_inference.py --scene center`) the
cooperator carried its object to two moves from its destination and stopped there in the
transition in which the hinderer's belief in that destination reached 0.99 ($\beta$ = 0.40,
0.21, 0.01 at transitions 9, 10, 11, with the hinderer's term weighted by the cooperator's
belief that its partner is hostile, 0.97 to 1.00). At 17 its physical Q favoured moving by 2.92 and its
whole Q favoured staying by 0.22, and the hinderer then came to stand on the destination.
For $w_i w_j > 1$, $\beta < 0$. Hinder and conflict have the same term at level 1, so the
cooperator's belief split between them (0.5 / 0.5) changed nothing.

**The rule.** `Config.cooperate_with` names the partners' social goals against which the
cooperator's cell holds; against any other it is 0, and the cooperator minds its own goal.
The default, `COOPERATE_WITH`, is cooperation, competition and exchange:

$$\xi_i = w_i \sum_{(k,\,t)} b_i(k, t)\,\alpha(t)\,\tilde R_j^{(k,t)}, \qquad
\alpha(t) = \begin{cases} 1 & t \in \{\text{cooperation, competition, exchange}\} \\
0 & \text{otherwise.} \end{cases}$$

- With the partner's goal inferred the cell is weighted by the belief hypothesis by
  hypothesis, so the help fades as the cooperator becomes sure its partner is a conflict
  robot, a hinderer, a coercer or a robot without social goal. At the uniform prior over the
  seven goals it is 3/7 of the table's.
- Of the three models it still cooperates with, only a competitor's has a term below 0 on
  the cooperator's own goal: a rival is a racer at level 1, $-r(g_i)$ while the cooperator
  is done and the competitor is not, so a cooperator may let a competitor finish first. Away
  from that, $\beta = 1 + w_i\,\tilde b_j(g_i^*)\,\big[w_j P_i(\text{cooperation}) +
  n\,P_i(\text{exchange})\big] \ge 1$ at any weight: a hinderer can no longer take the
  cooperator's goal from it through the cooperator's own reward. (Under
  `competition="table"` or `"table_hostile"` a competitor at level 1 is the conflict entry,
  and against it the cooperator is where it was against a hinderer.)
- `cooperate_with="all"` (`EVERY_PARTNER`) is the table as printed; `TOLD` restores it. A
  level-1 cooperator does not change (it holds no belief about its partner's social goal),
  nor does any other goal.
- A record says it in `social_terms` ("... [cooperates with cooperation, competition,
  exchange only]"), a figure names the robot "cooperation (selective)", and the example
  scripts write to folders ending in `_selective` and take `--cooperate-with`.

**Results.** $w = 1$, at most 60 transitions, the scenes of "Which object the partner wants"
(`python examples/run_object_inference.py --level-free-level 1`, `... --scene center
--pairing hinder cooperation`; `output/object_inference_two_w1_1_selective/`,
`output/object_inference_center_w1_1_selective/`, a GIF per rollout), the hinderer at level
1, its default until later the same day (at level 2: "Every robot with a social goal at
level 2"); agent_1 / agent_2, the cooperator second but in cooperation / exchange:

| scene, pairing | objects | Table I's cooperator | selective |
|---|---|---|---|
| two-room, cooperation / cooperation | known | 22 / 5 | 22 / 5 (the same rollout) |
| | inferred | 20 / 12 | 17 / 14 |
| two-room, competition / cooperation | known | 5 / – | 55 / 52 |
| | inferred | 5 / – | 5 / – (the same rollout) |
| two-room, hinder / cooperation | known | 5 / – | – / – |
| | inferred | 5 / – | 12 / 13 |
| center, hinder / cooperation | known | 28 / – | – / – |
| | inferred | 9 / – | 9 / 10 |
| 3 x 3 map, cooperation / cooperation | known | 7 / 5 | 14 / 4 |
| | inferred | 12 / 21 | 7 / 8 |
| six-object, cooperation / exchange | known | 22 / 10 | 38 / 26 |

competition / competition and none / none, with no cooperator in them, roll out the same in
both modes. The six-object scene with the object known (its spare objects scenery) gives the
two-room scene's numbers for hinder / cooperation (– / –; 5 / – with Table I's) and
competition / cooperation (55 / 52; 5 / –).

- *The stall is gone.* Center scene, objects inferred: the cooperator is 0.95 sure of a
  hostile partner at transition 7, and from then on its Q is its physical Q, which favours
  moving at every step. It carries its object straight to (1, 5) and delivers it at 10, one
  transition after the hinderer delivered its own; the hinderer became sure of the
  destination at 9, too late to block it. On the two-room map the cooperator hands the
  hinderer's object over to it (6), crosses into the hinderer's room for its own (11) and
  delivers it (13). The hinderer, which saw it pick O1 up, takes it to want O1 (its belief in
  O2 is 0.02 or less from transition 3 to 10) and does not guard O2.
- *Where the hinderer holds the cooperator's object, nobody finishes.* With the object known
  the hinderer takes it at transition 2 (two-room) or 7 (center), as before, and the
  cooperator no longer finishes the hinderer's task for it, as Table I's did (at 5 and 28).
  On the center scene the two then pick each other's object up together every eight
  transitions, seven times, and put it back in between (the cooperator seven times, the
  hinderer six), until the stand-off at 58. A robot without social goal at
  level 2 in the cooperator's place does not finish either (– / –, stand-off at 56; there
  the hinderer alone picks the object up and puts it back, eleven times): against a hinderer
  that holds its object hostage only a helper lets one goal be done, and it is the
  hinderer's.
- *A competitor that turns on cooperators* (`competition="rival"`). With the object known,
  the cooperator takes the rival for a hostile partner (conflict or hinder, 0.99 at
  transition 4; 0.72 to 0.88 until 15) and stops helping; the rival's belief that it faces
  a cooperator falls from 0.84-1.00 (transitions 4 to 14) to 0.30 (15) and 0.00 (23), and it
  races instead of working against it. After four put-backs of the rival's and two of the
  cooperator's, the cooperator takes the rival for a cooperator after all (0.96-1.00 from 33
  to 47) and tries the rival's object on a wrong destination (37); the two then swap the
  objects hand to hand (49) and each delivers its own (52, 55). With the object inferred the cooperator reads the rival as a
  competitor (0.77-0.87 at transitions 4-6), keeps helping, and delivers the rival's object
  at 5 as Table I's did. The rival is then sure it faces a cooperator (0.96) and takes the
  cooperator's object at 10; the cooperator, sure of a hostile partner from 8 (0.95), never
  gets it back. The cooperator models the rival one level down, as a racer, so it cannot see
  that its help is what turns it.
- *Two cooperators.* Told which object the other wants, the same rollout on the two-room map
  (each sure of a cooperative partner from transition 4, 0.96 and 0.97). Inferring it: 17 / 14, without the
  put-back of Table I's run. On the 3 x 3 map (pinned in `tests/test_objects.py`), told: one
  hands the other its object (4) and fetches its own (14); inferring it, neither helps, each
  delivers its own (7 / 8 where Table I's take 12 / 21), and at the end each takes the other
  for a conflict robot or a hinderer more than anything else (0.80, 0.56).
- *An exchanger is helped later.* Six-object scene, object known: the cooperator delivers
  the exchanger's object at 26 instead of 10, after two set-downs in vain, and its own at
  38. An exchanger is rarely recognised (ranked first 2 times in 10 at $w = 1$, "how well
  the robots infer each other's goals"), and while the cooperator doubts it, it helps less.

**Limits.**
- Its help is only as right as its belief. A helper looks like a hinderer at first (picking
  a robot's object up is both): on the two-room map with the object known, two cooperators
  put 0.56 and then 0.72-0.74 on a hostile partner at transitions 2 and 3, before 0.96-0.97
  on a cooperative one at 4.
- Against a hinderer that holds its object hostage it does no better than Table I's
  cooperator, and a level-1 hinderer does worse; a level-2 one got its own goal done on the
  two-room map ("Every robot with a social goal at level 2").
- It does not cooperate with a robot without social goal or with a coercer: before any
  evidence it helps at 3/7 of the table's strength.
- Not in the paper; with `cooperate_with="all"` everything is the table's, bit for bit.

## Every robot with a social goal at level 2 (since 2026-10-11)

The user, on 2026-10-11: "Whenever I say run it, always assume both are level 2."

**The rule.** `Config.level_free_level` is the default level of a robot whose term holds no
$\tilde R_j$ -- a hinderer, and a competitor of a race form -- and the most a hinderer may be
asked for. It is 2: every robot with a social goal reasons at level 2 unless a level is
asked for (`default_levels`), and a level asked for is used (`levels_used`). A robot with a
physical goal only stays at level 0 (`physical_level`, the rule of 2026-10-08), so "both"
is read as both robots with a social goal; `physical_level=2` puts it at level 2 as well.
`level_free_level=1` is the rule of before, and `TOLD` keeps it: such a robot at level 1 by
default, a hinderer held to level 1 whatever was asked. Under the default reading of
competition, `"rival"`, a competitor was at level 2 already, so the rule changes the
hinderer, and a competitor only under `competition="race"` or `"race_then_hinder"`; every
pairing without one of them rolls out as before. The example scripts write to folders
ending in `_level2`; `--level-free-level 1` (or `--told`) names them as before.

**What changes for a hinderer.** Its reward does not: at either level

$$R_i = r(g_i) - w_i\,\mathbb{E}_{b_i}\big[r(\tilde g_j)\big] - c(a_i),$$

with no $\tilde R_j$ in it. What changes is whom it takes its partner to be. At level 1 its
partner is a level-0 robot without social goal, which moves an object only if it wants it,
and the hinderer infers the partner's object and destination under that model alone. At
level 2 it holds a belief over the partner's social goal as well (the seven of
`social_hypotheses`) and predicts the partner by the level-1 model of each: a cooperator
that carries the hinderer's object to the hinderer's destination, a conflict robot that
takes it, a rival that races. It was held at level 1 because, told its partner's goal and
against Table I's competitor, a level-2 hinderer never took the object ("What hindering
rests on", line 3). Nobody told, it took it in seven of the eight runs below (in the eighth
the cooperator held its own object from transition 4 until it set it down).

**Results.** $w = 1$, at most 60 transitions, the cooperator of now (selective), each pairing
with the partner's object known and inferred: `python examples/run_object_inference.py
--pairing hinder cooperation --pairing hinder competition --pairing hinder none` (the
two-room scene) and `... --scene center --pairing hinder cooperation`, and the same with
`--level-free-level 1` (`output/object_inference_two_w1_1_selective_level2/`,
`output/object_inference_center_w1_1_selective_level2/`; the hinderer at level 1 in the
folders without `_level2`; a GIF per rollout). The hinderer is agent_1; agent_1 / agent_2:

| scene, pairing | objects | hinderer at level 1 | hinderer at level 2 |
|---|---|---|---|
| two-room, hinder / cooperation | known | – / – | 29 / – |
| | inferred | 12 / 13 | 5 / – |
| two-room, hinder / competition | known | – / – | – / – |
| | inferred | 26 / 28 | – / – |
| two-room, hinder / none | known | – / – | – / – |
| | inferred | – / – | – / – |
| center, hinder / cooperation | known | – / – | – / – |
| | inferred | 9 / 10 | 11 / 11 |

The partner finishes in three of the eight runs with the hinderer at level 1 and in one with
it at level 2; the hinderer's own goal is done in three either way.

- *A level-2 hinderer reads help and guarding for what they are* (two-room, objects
  inferred). At level 1 a partner that moves the hinderer's object, or makes for its place,
  can only want that object. Against the cooperator, which picked the hinderer's object up
  at 2, the hinderer took it to want that object (its belief in O2 0.02 or less from 2 to
  10) and left O2 alone: 12 / 13. Against the rival, which made for the place of the
  hinderer's object, it came to believe the rival wanted O1 (0.97 at 15, 1.00 from 16 to
  19), handed O2 over at 18, and both finished: 26 / 28. At level 2 the partner may be a
  cooperator or a rival. Against the cooperator its belief in O2 stayed at 0.49-0.54 while
  the cooperator carried O1 (2 to 4), rose to 0.72 (5) as the cooperator reached the
  hinderer's destination with O1, and to 0.99 (6) once it had set it down, the hinderer now
  taking it for a cooperator (0.96): it went back and took O2 (19), and the cooperator,
  taking it for a hostile partner from 10 (0.96; 1.00 from 11), never got it. Against the
  rival it was sure of O2 from 3 (0.99), took it at 6, and held it on the place of its own
  object to the end.
- *With the object known it gets its own goal done and still keeps the cooperator from its
  own* (two-room, 29 / – where level 1 gives – / –). As at level 1 the two take each other's
  object at 2, and the cooperator, sure of a hostile partner (0.99 at 4 at level 2), puts the
  hinderer's back at 5; it picks it up once more and puts it back (12, 13). At 15 the
  hinderer, standing on the place of the cooperator's object, put that object back (Q 26.6
  for putting back, 24.4 for staying; it predicted the cooperator to pick the hinderer's
  object up). The cooperator took the put-back for help (P(cooperation) 0.97 at 16), picked
  the hinderer's object up and set it down on the hinderer's destination at 29, two cells
  away, after going out to (7, 4) and back (its belief in a cooperator fell to 0.49 at 26 on
  the way); at 30 the hinderer took the cooperator's object again and kept it. The hinderer
  models the cooperator one level down, a robot that holds no belief about the hinderer, so
  nothing in its plan is about what the cooperator makes of the put-back: it is not a ruse,
  though it works as one. On the six-object scene, with its spare objects scenery, the same
  nine events happen in the same transitions (the first example above).
- *Against a rival, with the object known, neither finishes at either level.* At level 2 the
  rival came to take the hinderer for a cooperator (0.99 from 9, after the hinderer had put
  the rival's object back at 4 and 8), and from 21 to the end it kept the hinderer's object,
  as a rival does against a cooperator. Its own object, on its place from 24, it never
  fetched: against a cooperator its term $-\tilde R_j$ holds $-w_i w_j\,\tilde r(g_i)$, so
  at $w = 1$ its own goal is worth nothing to it once its partner is sure of that goal, as
  a cooperator's was against a hinderer under Table I ("The model", `"table_hostile"`).
- *Against a robot without social goal nothing changes*: at either level the hinderer takes
  its partner's object (at 2; at 6 at level 2 with the object inferred), carries it to the
  place of its own object and stands there holding it, and the level-0 partner stops.
- *On the center scene, with the objects inferred, both finish at either level* (9 / 10;
  11 / 11). Each robot takes its own object off the doorway at the start (3 and 4; 3 and 5
  at level 1), and nobody can take an object out of its holder's hands. The level-2
  hinderer turned back toward the doorway at 6 to 8, while it became sure of the
  cooperator's destination (0.42, 0.64, 0.99), then went to its own: 11 where level 1
  finishes at 9. The cooperator, which no longer knew which object the hinderer wanted
  (0.53 on a spare at 8) and took it for a competitor more than for anything else (0.32 and
  0.30 at 9 and 10), waited next to its destination at 9 and 10 and set its object down
  together with the hinderer. At 9 its physical Q favoured the last step (by 0.06) and its
  whole Q waiting (184.2 against 171.9; at 10 by 0.14): its help holds a competitor's race
  term, which loses while the cooperator is done and the competitor is not ("Whom a
  cooperator cooperates with").
- *What it costs.* A level-2 hinderer has a level-2 model of its own. Whole rollouts, of
  different lengths, took 1.2 to 3.3 times as long next to a cooperator (center, objects
  inferred: 63 minutes against 26), 1.9 and 8.1 times next to a rival (60 transitions
  against 32 with the object known), and 8 and 32 times next to a robot without social
  goal, where it is the only level-2 model (two-room, objects inferred: 234 s against
  7.2 s).

**Limits.**
- Eight rollouts, one weight, greedy actions: not a statistic.
- A level-2 robot plans against level-1 models of its partner, and a level-1 model holds no
  belief about the robot. So a level-2 hinderer cannot plan on what its partner will make of
  what it does, and a level-2 cooperator cannot see that what it does makes a hinderer act.
  That needs level 3, which this code does not build.
- The cooperator above is the selective one. Against Table I's (`cooperate_with="all"`) a
  level-2 hinderer was rolled out once, in the first example above: (5, None), as at level 1.

### Both robots at level 2 in every pairing (2026-10-12)

The user, on 2026-10-12: "Can you run all of the simulations again at alevel = 2 and all
being inferred and format it with the html", then "Can we run coercion vs hinder?". Read
as: the eight pairings below on the scenes of the player, each robot at level 2
(`levels=(2, 2)`, so a robot without social goal too, which is at level 0 by default),
the partner's object inferred only, the cooperator of now. `python
examples/run_object_inference.py --scene two --levels 2 2 --objects inferred --pairing
cooperation cooperation ... --pairing coercion hinder`, and the same with `--scene spare`
(`output/object_inference_two_w1_1_selective_level2_levels2_2/` and `..._spare_...`;
`--levels` adds `_levels2_2` to the folder, `--objects inferred` leaves out the runs with
the object known). The scene with the objects in the middle, with four objects, was not
rolled out again. $w = 1$, at most 60 transitions; agent_1 / agent_2:

| pairing | two-room | one spare object |
|---|---|---|
| cooperation / cooperation | 17 / 14 | 24 / 19 |
| competition / cooperation | 5 / –, stand-off | 26 / 25 |
| competition / competition | 9 / 11 | 9 / 11 |
| hinder / cooperation | 5 / –, stand-off | 50 / 45 |
| none / none | 9 / 11 | 9 / 11 |
| hinder / competition | – / –, stand-off | 9 / 11 |
| hinder / none | – / –, stand-off | 9 / 11 |
| coercion / hinder | – / –, step limit | 21 / 24 |

- *On the two-room scene the levels asked for change only a robot without social goal.*
  Every robot with one was at level 2 already, so six of the seven pairings of before roll
  out action for action as on 2026-10-11; two robots without social goal roll out at
  level 2 with the actions they take at level 0. In hinder / none the hinderer takes its
  partner's object at 6 and, from 27, stands on the place of its own holding it. The
  partner at level 0 stops where it was when its object was taken (at (3, 4) from 6 on);
  at level 2 it follows the hinderer and waits diagonally next to it from 28 (a stand-off
  at 52).
- *Coercion against a hinderer, two-room: neither finishes.* The hinderer took the
  coercer's object from its place at 4 (it was 0.99 sure the coercer wanted it), the
  coercer took the hinderer's at 12, and the hinderer put the coercer's back at 20. From
  38 to the end the coercer stood on the place of its own object with the hinderer's in its
  hands, the hinderer next to it from 41: with its hands full it could not pick its own
  up, and handing the hinderer its object was worth 113 to it against 289 for staying
  (Q from 41 to 60). That is the case that led to the next section.
- *One spare object (O3, in agent_1's room).* Against the competitor and the robot without
  social goal the hinderer leaves its partner's object alone: from 3 to 6 it gave the
  spare 0.31 to 0.37 and went for its own object, and both finish at 9 / 11, as two robots
  without social goal do. Against the cooperator, 50 / 45: the cooperator took the
  hinderer's object at 6, tried to set it down on four wrong candidates (12, 17, 33, 40)
  and handed it over at 41; the hinderer had taken the cooperator's object at 8 and put it
  back at 9. Coercion against the hinderer, 21 / 24: the hinderer took the coercer's
  object at 7, put it back at 8, took it again at 9 and handed it to the coercer at 15.
- *What it costs.* With the spare in the model a rollout took 883 to 3,974 s, against 137 to
  954 s on the two-room scene.

## Swapping on a place and taking from the partner's hands (options, since 2026-10-12)

The user, on 2026-10-12, of the coercion / hinder rollout above: "Are the agents able to
swap their objects with the one they are on top of, for hinder vs coercion, the agent
should have dropped the red to get his orange block. Also, lets make it such that agents
can steal from each other". They could not swap: a robot holds one object, and nothing let
the coercer put the hinderer's object down on the place of its own. Asked how, the user
chose a swap on the place of any object, taking whatever the partner holds, and both as
options, off by default, tried on the two-room scene first.

**The rule** is in "The world" (`swap_on_place`, `steal`). In `hands.hands_step` a swap
comes first among the uses of full hands, and a take last among those of free hands: after
everything else in the transition, and only if the partner still holds the object. Both
are judged after the moves, a swap on the cell the robot stands on, a take on the two
robots being 4-neighbours: a holder that steps away from its partner keeps what it holds.
An object lying on another's place has status 4 + H + p, p the place's object, H the
hand-off cells (`ObjectWorld.lie_cells`). The record calls a swap "dropped", next to the
"picked up" of what lay there, and a take "took", with the robot taken from as the other
robot; the ledger counts `dropped`, `took` and `taken_from`, and the figure calls them out
("LEFT", "TOOK"). `--swap` and `--steal` of `run_object_inference.py` add `_swap` and
`_steal` to the folder.

**Results.** The two-room scene, both robots at level 2, objects inferred, $w = 1$, at most
60 transitions: `python examples/run_object_inference.py --scene two --levels 2 2
--objects inferred --swap --steal --pairing ...`
(`output/object_inference_two_w1_1_selective_level2_levels2_2_swap_steal/`), against the
runs of the section before:

| pairing | rule of before | swaps and taking |
|---|---|---|
| cooperation / cooperation | 17 / 14 | – / –, livelock |
| competition / cooperation | 5 / –, stand-off | 5 / –, step limit |
| competition / competition | 9 / 11 | 9 / 11 |
| hinder / cooperation | 5 / –, stand-off | 5 / –, step limit |
| none / none | 9 / 11 | 9 / 11 |
| hinder / competition | – / –, stand-off | 25 / 16 |
| hinder / none | – / –, stand-off | 25 / 16 |
| coercion / hinder | – / –, step limit | 9 / 13 |

Nobody swapped in the eight runs, and an object was taken twice, once in each of two runs.

- *Coercion against a hinderer: both finish, and nothing is swapped.* From the first
  transition the hinderer makes for its own object instead of the coercer's: Q 86.6 for
  the step toward its own, 85.5 for the step onto the place of the coercer's object (under
  the rule of before 81.6 and 82.3). Nobody touches the other's object: 9 / 13, where two
  robots without social goal finish at 9 / 11 (the hinderer waits at 4 and 7). The state
  the user pointed at does not arise.
- *That state, started again*: the coercer on the place of its own object with the
  hinderer's in its hands, the hinderer next to it with free hands, as at 41 to 60 of the
  run of before, but with every belief at its prior, 30 transitions at most (a one-off
  script, not in the package: the world built as for the run, then its start set to that
  state):

  | rule | first transition | agent_1 / agent_2 |
  |---|---|---|
  | of before | the coercer hands the hinderer its object, then picks its own up | 4 / 10 |
  | swaps | the coercer leaves the hinderer's object on its place and takes its own | 12 / 16 |
  | taking | the hinderer takes its object out of the coercer's hands | 4 / 10 |
  | both | the hinderer takes its object; the coercer stays | 4 / 10 |

  So the coercer of the run of before was not short of a move: handing the object over
  was legal there too, and with fresh beliefs it hands it over at once (Q 247.4 against
  247.3 for staying). What kept it from doing so at 41 to 60 was what it believed by
  then: its partner a competitor or a hinderer (0.50 each), which wants O2 (1.00); handing
  over was worth 113 to 117 to it against 289 for staying. With swaps alone it does what
  the user expected (Q 279.0 against 278.8), stays on the cell where the hinderer's object
  now lies until it leaves at 5, and delivers its own at 12; the hinderer picks its object
  up at 6.
- *Taking back what a hinderer holds* (hinder / competition, hinder / none, the same
  events in both): the hinderer takes O2 at 8; at 11 it stands next to its partner without
  moving, and the partner takes O2 out of its hands, delivers it at 16, and the hinderer
  then fetches its own object and delivers it at 25.
- *Every other grab is dodged* (competition / cooperation, hinder / cooperation). As
  under the rule of before the cooperator delivers its partner's object at 5 and the
  partner then takes the cooperator's (12; 10). The cooperator follows it and uses its
  hands next to it 17 and 16 times, and every time the holder steps out of reach in the
  same transition, to the step limit.
- *Two cooperators livelock.* Both step onto the place of the other's object at 1, pick it
  up together at 2, put it back together at 3, and so on: 26 pick-ups and 25 put-backs
  each, until the livelock check stops the run at 52. With hands free, "use" leads staying
  by 0.6 (agent_1: 160.2 against 159.6); holding the other's object, putting it back leads
  by 0.4 (111.6 against 111.2), the physical Q as well. Each robot reads its partner
  wrongly from 2 on: that it wants the robot's own object, 0.76 and 0.77 at 2, 0.996 to
  0.997 after each joint pick-up and 0.77 to 0.84 after each put-back. Under the rule of
  before the same first step left the beliefs at 0.49 to 0.51, agent_1 stepped back at 2
  (152.1 against 149.2 for picking up), and agent_2 carried O1 over: 17 / 14.

  *Why it cycles* (the solved models read at 1 to 3; `run_object_inference.py --steps 12`
  with one option at a time). Taking alone gives the same rollout, value for value; with
  swaps alone both pick up at 2, but at 3 agent_1 stays (by 0.06) and the two part. With
  taking, the first step is read as wanting the object stepped onto (likelihood 0.94 for a
  partner that wants it, 0.30 for one that wants its own; 0.54 and 0.56 under the rule of
  before). At 2 each robot predicts the other's pick-up (0.70), and picks up as well. In its
  model a partner that wants its own object never picks the robot's up (likelihood 0.013,
  the lapse), one that wants the robot's object does (0.91): a level-2 robot models its
  partner at level 1, whose model of the robot is level 0, so in it a cooperative partner
  has read the robot's first step as wanting the partner's object and would not fetch the
  robot's. Sure that its partner wants the robot's own object (0.996), it puts back what
  it holds, an object the partner does not want, which doubles the cost of every move. A
  partner that wants the robot's object would not put it back (0.013), one that wants its
  own would (0.85): the belief falls to 0.77, still wrong, and picking up leads again. The
  two updates nearly cancel (likelihood ratios of 70 and 66), and the two robots, mirror
  images choosing at once by argmax, never break the tie. Why taking makes the first
  pick-up worth more than staying (by 0.6, against 2.0 less under the rule of before) was
  not taken apart by hypothesis.
- *What it costs.* Swaps make 22,736 joint states instead of 15,680 (29 configurations of
  the objects instead of 20); a transition took about as long (11.1 s on average against
  11.4 s; 2,908 s for the eight runs against 3,133 s).

**The belief-update temperature against the livelock** (2026-10-13). The user: "Lets try
this, try multiple different temperatures: Softer belief updates: raise
`inference_temperature` so a single action can't swing the belief to 0.996 and back."
Cooperation / cooperation as above with `--inference-temperature T`
(`Config.inference_temperature`: the temperature of the likelihood of Eqs. 5-6 in units of
the move cost; planning keeps Eq. 7's 0.2, the default of both), folders
`..._swap_steal_itau<T>/`:

| belief-update temperature | agent_1 / agent_2 | wrong belief after the joint pick-up | set-downs in vain | taken |
|---|---|---|---|---|
| 0.2 (Eq. 7's) | – / –, livelock | 0.996 / 0.996 | 0 | 0 |
| 0.5 | 5 / 26 | 0.99 / 0.86 | 0 | 1 |
| 1 | 5 / 25 | 0.98 / 0.78 | 0 | 0 |
| 2 | 19 / 33 | 0.79 / 0.74 | 3 | 1 |
| 5 | 42 / 5 | 0.64 / 0.67 | 2 | 1 |
| 10 | 51 / 47 | 0.58 / 0.60 | 10 | 0 |

("Wrong belief": each robot's belief, at 3, that its partner wants the robot's own object.)

- *Both robots still pick up together at 2 at every temperature*: that is planned, whatever
  the update. What changes is what follows.
- *From 0.5 up nobody cycles.* The two robots' beliefs after the joint pick-up now differ
  (the map is not symmetric), and so do their choices at 3. At 0.5 and 1 only agent_1 puts
  back; agent_2 carries O1 on, reads the put-back as agent_1 wanting O1 (0.86 at 4 at
  temperature 1) and sets it down on agent_1's destination at 5. From 2 up neither puts back at 3: each
  carries the other's object on.
- *The softer the update, the slower the robots learn where to go*: 0, 0, 3, 2 and 10
  set-downs in vain. 0.5 and 1 finish first (26, 25), 10 last (51).
- *Neither robot ever takes the other for a cooperator*: at every temperature from 0.5 both
  end up believing a conflict robot, a competitor or a robot without social goal
  (P(cooperation) 0.005 or less at the end; 0.08 at 0.2, where both believe coercion). The
  misreading of help is not cured, it only no longer sustains itself.
- Under the rule of before at 0.2 the same pairing finished at 17 / 14, sooner than any of
  these.

**All eight pairings at belief-update temperatures 1 and 0.5** (2026-10-13; the user: "Run
all 8 pairings", and for 0.5 "yeah trun all 8 pairings"). `--swap --steal
--inference-temperature 1` and `0.5` with every pairing (`..._swap_steal_itau1/`,
`..._swap_steal_itau0.5/`):

| pairing | rule of before | swaps and taking | ... temperature 1 | ... temperature 0.5 |
|---|---|---|---|---|
| cooperation / cooperation | 17 / 14 | – / –, livelock | 5 / 25 | 5 / 26 |
| competition / cooperation | 5 / –, stand-off | 5 / –, step limit | 7 / –, step limit | 5 / 50 |
| competition / competition | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| hinder / cooperation | 5 / –, stand-off | 5 / –, step limit | 5 / –, livelock | 5 / 36 |
| none / none | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| hinder / competition | – / –, stand-off | 25 / 16 | 26 / 32 | 32 / 23 |
| hinder / none | – / –, stand-off | 25 / 16 | 26 / 32 | 25 / 15 |
| coercion / hinder | – / –, step limit | 9 / 13 | 9 / 17 | 9 / 16 |

Both robots finish in all eight pairings at temperature 0.5, in six at 1, in five at 0.2,
and in three under the rule of before.

At temperature 1:

- *Hinder against competition and against a robot without social goal* (the same events
  in both): the hinderer takes O2 at 11 and, coming to take its partner for a cooperator
  (0.90 at the end), puts it back at 16; it fetches its own object and delivers it at 26,
  and the partner delivers O2 at 32. Nobody takes anything; at 0.2 the partner took O2
  back at 11 and both finished sooner (25 / 16).
- *Against a cooperator neither a rival nor a hinderer lets go.* The cooperator delivers
  its partner's object (7; 5), the partner takes the cooperator's (12; 9), and the
  cooperator's grabs next to it come to nothing (12 and 16 times). The cooperator next to
  the rival takes its object back at 46 and is on its way at the step limit (at (3, 3) at
  60); against the hinderer the livelock check stops the run at 59.
- *Coercion against hinder* finishes again with nothing swapped or taken, the hinderer four
  transitions later (17).

At temperature 0.5:

- *The cooperator gets its object back from the rival and from the hinderer.* As at every
  setting it first delivers its partner's object (5), and the partner takes the
  cooperator's (12; 10). Its grabs next to the holder come to nothing 16 and 3 times, until
  the holder stays put beside it: it takes its object at 47 and 33 and delivers it at 50
  and 36. A rival or a hinderer that keeps the cooperator's object is what those goals are
  for; at 0.5 they keep it less well.
- *Hinder against competition and against a robot without social goal no longer match.*
  Against the rival the hinderer picks O2 up at 7 and puts it back twice (10, 14) before
  the rival takes it (18): 32 / 23. The robot without social goal takes it back at 10, as
  at 0.2 (there at 11): 25 / 15.
- *Coercion against hinder*: 9 / 16, nothing swapped or taken.

**All 49 pairings at belief-update temperature 0.5** (2026-10-13; the user: "Can you run all
49 pairings and format it into the html we had earler", with swaps and taking at 0.5 when
asked which settings). Every social goal held possible against every one, agent_1's by
agent_2's: `--swap --steal --inference-temperature 0.5 --all-pairings` (the folder of the
eight, `..._swap_steal_itau0.5/`, whose eight runs were kept; 41 new, 3.0 hours on 2 cores).
The run named the 49 as `--pairing`s; `--all-pairings`, added after, names the same 49 in the
same order, and with `--resume` it rewrote `results.md`, `results.png` and `summary.json` to
the byte (the figure pixel for pixel). `results.png` puts more than eight pairings in rows of
seven, here agent_1's goal by agent_2's.

| agent_1 \ agent_2 | cooperation | conflict | competition | coercion | exchange | none | hinder |
|---|---|---|---|---|---|---|---|
| cooperation | 5 / 26 | 22 / –, stand-off | 22 / 28 | 13 / 5 | 17 / 14 | 17 / 14 | 17 / 16 |
| conflict | 5 / 50 | – / –, stand-off | – / –, stand-off | – / –, stand-off | – / –, stand-off | – / –, stand-off | – / –, stand-off |
| competition | 5 / 50 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 16 |
| coercion | 5 / 11 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 16 |
| exchange | 5 / 17 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 16 |
| none | 5 / 17 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 16 |
| hinder | 5 / 36 | 36 / 25 | 32 / 23 | 36 / 30 | 25 / 15 | 25 / 15 | 27 / 16 |

Both robots finish in 42, one in 1, neither in 6. By its own goal, counting both seats, a
robot finishes in 14 of 14 runs as a cooperator, in 6 of 14 in conflict, and in 13 of 14
with each other goal (the one it misses is against a conflict robot in agent_1's seat).

- *Four goals do the same unless the partner cooperates.* In agent_1's seat, competition,
  exchange and no social goal make the same rollout, move for move, against each of the
  six partners that do not cooperate; coercion moves otherwise and finishes alike. In
  agent_2's seat, against those four, competition and no goal make one rollout, coercion
  and exchange another, with the same finishes. They are those of two robots without
  social goal, 9 / 11; 9 / 17 against a conflict robot, 9 / 16 against a hinderer.
- *A conflict robot in agent_1's seat stops every partner but a cooperator.* The six
  rollouts are one, move for move: it walks into the doorway, meets agent_2 coming the
  other way at transition 7 ((2, 2) and (3, 2)), and both move into each other at every
  transition until the stand-off check stops the run at 31. Nothing is picked up. Against
  a cooperator it plays the rival's rollout move for move: 5 / 50.
- *In agent_2's seat a conflict robot does not block.* Against competition, coercion,
  exchange and no goal it heads for agent_1 and grabs at agent_1's object at 9, the
  transition agent_1 sets it down on its destination, so the grab comes to nothing; then it
  fetches its own: 17, six transitions after a robot without social goal. Against a
  hinderer 36 / 25. Against a cooperator it takes its object out of the cooperator's hands
  at 19 (the cooperator had carried it since 2, swapped it for its own on O1's place at 17
  and back at 18), carries it to (1, 4), next to its destination, at 28, and stays there
  until the stand-off check stops the run at 52. At transition 31 its social term (Table
  I's conflict above level 1, with the partner taken for a cooperator at 0.997 -- read as
  "a conflict robot" until 2026-10-14, the label of the wrong index) gives
  −1.18 for staying and −1.25 for the step onto the destination: Q −133.88 against
  −134.28, where the physical part alone prefers the step by 0.06 (114.26 against 114.20).
- *A hinderer in agent_1's seat slows both and stops neither.* Except against a cooperator,
  it picks agent_2's object up at 7. Against exchange, no goal and a second hinderer,
  agent_2 takes it back at 10 (25 / 15, 25 / 15, 27 / 16). Against conflict and competition
  the hinderer puts it back at 10 and 14, picking it up again each time, and agent_2 takes
  it at 18 (36 / 25, 32 / 23). Against coercion it puts it back four times (10, 14, 22, 24)
  and nobody takes it; the coercer picks it up off its place at 28 (36 / 30). Against a
  cooperator, which first delivers the hinderer's object (5), it picks the cooperator's up
  at 10 and keeps it until the cooperator takes it back at 33 (5 / 36).
- *In agent_2's seat a hinderer costs only itself:* against competition, coercion, exchange
  and no goal 9 / 16, where a robot without social goal finishes 9 / 11; against a
  cooperator 17 / 16 (17 / 14 without social goal).
- *A cooperator in agent_2's seat delivers agent_1's object first, at 5, whoever agent_1 is.*
  It finishes its own at 11 against coercion, 17 against exchange and no goal, 26 against a
  cooperator, 36 against a hinderer and 50 against a rival or a conflict robot (from those
  three it takes its object back, at 33 and 47).
- *Taken and swapped.* All 14 takes are agent_2 taking its own object (O2) out of agent_1's
  hands: a cooperator carrying it to deliver it (six runs), a hinderer (six), a rival or a
  conflict robot (two). Two swaps are recorded as such ("dropped"), both by the cooperator
  in agent_1's seat at 17, against competition and conflict: it leaves O2 on O1's place and
  takes O1, and at 18 it swaps back (O1 put back, O2 picked up).
- *Reading the partner.* Of the 98 final readings (each robot's most probable social goal
  for its partner) 26 have the partner's goal alone in the lead (12 of agent_1's, 14 of
  agent_2's) and 18 more have it tied for the lead with the one goal it cannot be told
  from: conflict and hinder are the same model one level down (each is $-r(\tilde g_j)$
  there), so their beliefs always tie. A cooperator is read right in all 14 readings of it,
  a hinderer as conflict-or-hinder in 12 of 14; competition, coercion, exchange and no goal
  are taken for competition in 10 or 11 of 14 each. Those four move alike against all but
  a cooperator (first point), so what they do cannot tell them apart. A conflict robot is
  read as conflict-or-hinder 6 times in 14, as a cooperator 6 times. (Corrected
  2026-10-14: the first count read the belief vector, which is stored in the order of
  `goals.SOCIAL_GOALS`, in the order of `Config.social_hypotheses` -- `c21/readings.py`.) 92 of the 98
  robots end at least 0.9 sure that the partner wants its own object; the six others are
  agent_2 in the stand-offs of the conflict row (0.28), whose partner never goes for its
  object.

**Checks.**
- `test_hands.py`: the cases (a swap before a hand-over; without the option the object is
  handed over instead; the owner takes its object off another's place on arriving; a swap
  back; every robot may swap; putting back on an empty place of its own; 29
  configurations and one object to a place; a take; whatever is held is taken; without
  the option nobody lets go; the holder's own act comes first; only 4-neighbours after
  the moves; full hands take nothing; every robot may take), and every transition table
  and every "may it choose this hand action here" against the independent statement of
  the rule
  (`tests/reference.py`): each option alone and both, and both with forced swaps, without
  hand-overs and with the hands of a robot that is done kept still, on the 3 x 3 map;
  both on the map with a doorway that is a hand-off cell; and both with the partner's
  object inferred, on the 3 x 3 map, a 3 x 2 map with a spare object and the map with
  the doorway. The own-object rule holds on every transition with both options on.
- `test_models.py`, against brute force on the 3 x 3 map: a level-1 hinderer holding its
  partner's object next to a level-2 cooperator, each told the other's goal (level 2:
  rewards to 7.9e-7, V and Q to 1.4e-5); slow, both at level 2 with every goal and object
  inferred, coercion against a hinderer, each holding the other's object: 72 level-1
  and 4 level-2 models, level 2 to 2.4e-7 in the rewards and 8.0e-6 in V and Q (level 1:
  1.7e-5).
- With both options off the code rolls out as before: two runs of the section before
  (cooperation / cooperation, competition / competition) were rolled out again and are the
  same in every field of the record but the times, to the last bit, with the five fields
  added today at their neutral values.

**Limits.**
- One scene, greedy actions, no noise: each rollout, the 49 included, is one deterministic
  run, not a statistic. A final reading of the partner is one number per run, after runs
  of different lengths.
- The two options were tried together; which one changes which rollout is known only for
  the stand-off and for the cooperators' livelock (taking).
- With three objects the model of the objects grows from 56 configurations to 181 (3.2
  times), with four from 126 to 1,321 (10.5 times). The scene with a spare object and the
  scene with the objects in the middle were not tried.

## Seeing a blocked move as staying put (option, since 2026-10-13)

The user, on 2026-10-13, of the 49 pairings above: "So why does conflict deadlock for
literally every scene?", then "So how can we fix this deadlock where they are constantly
moving in to each other", and, of the fix proposed: "Ok implement this: Recommended fix:
update beliefs on what the partner did, not what it tried. A blocked move would count as
'stayed put'."

**Why the robots kept pushing.** In the six stand-offs of the conflict row the conflict
robot stays in its own room, between agent_2 and agent_2's object: at transition 3 its
physical part prefers crossing the doorway by 0.42 in Q, its conflict term staying by 1.65.
From transition 7 the two move into each other, at (2, 2) and (3, 2). Each pushes because
it expects the other to give way: agent_2's way round is two moves longer and 1.6 below
pushing, and pushing beats waiting by 0.35 although a blocked move costs ten times a wait.
Neither learns otherwise. The idle hypothesis (`idle_hypothesis`, the user's of 2026-10-08
for stationary partners) would explain a partner that does not move, but the belief update
conditioned on the action the partner chose: a move into the partner is a move, which an
idle partner never makes, so both robots' idle probability stayed at the hazard, 0.02, for
all 25 transitions. No social-goal hypothesis predicted the push either, so it was about as
surprising under each, and the beliefs barely moved.

**The rule** (`Config.partner_observed`). `"outcome"`: a partner that stayed in its cell,
waiting or blocked, is seen to have stayed put; the likelihood of that under each hypothesis
is the summed probability of every action that would have left it there, given the robot's
own action -- the wait, and each move it could make that would have been blocked too
(`SocialMDP.observed_actions`). The lapse is mixed in once per legal action of the set, and
the idle filter's likelihood is that of the set. `"stay"`: a blocked move is taken for a
wait. Every belief is updated so, the nested beliefs of a robot's models about it included,
and the ledger's modeled reward weighs the hypotheses the same way; planning is untouched.
The record's trace gives, per robot, the actions it was seen to have possibly chosen
(`seen_as`). `--partner-observed outcome|stay` of `run_object_inference.py` adds
`_seenoutcome` / `_seenstay` to the folder. Checks: "How it is checked".

**Conflict / none**, two-room, both robots at level 2, objects inferred, swaps and taking,
belief-update temperature 0.5:

| belief updated on | agent_1 / agent_2 | what happens |
|---|---|---|
| the action chosen (as before) | – / –, stand-off | both push from 7 until the check stops the run at 31 |
| what was done, `"outcome"` | 23 / 22 | both push at 7 to 9; agent_1's idle probability for agent_2 0.02, 0.14, 0.45, 0.78, 0.92 (agent_2's for agent_1 within 0.06); both wait at 10 and step aside at 11, meet again at 13 to 15, pass at 16; agent_2 goes round to its object (picked up at 20, delivered at 22), the conflict robot fetches its own (21, 23) |
| a blocked move as a wait, `"stay"` | 16 / 15 | both push at 7 and 8 (idle 0.17, 0.56); at 9 agent_2 steps aside and the conflict robot passes; agent_2 picks its object up at 13 and delivers it at 15, the conflict robot its own at 14 and 16 |

**The conflict row with `"outcome"`** (`--partner-observed outcome`, folder
`..._swap_steal_itau0.5_seenoutcome/`):

| agent_1 conflict against | as before | blocked moves seen as staying put |
|---|---|---|
| cooperation | 5 / 50 | 5 / 50 |
| conflict | – / –, stand-off | 23 / 22 |
| competition | – / –, stand-off | 18 / 16 |
| coercion | – / –, stand-off | 18 / 16 |
| exchange | – / –, stand-off | 20 / 19 |
| none | – / –, stand-off | 23 / 22 |
| hinder | – / –, stand-off | 34 / –, step limit |

Against the hinderer the push ends too, but the conflict robot then picks the hinderer's
object up (19), carries it to its own room and leaves it on its own object's place in a swap
(32), delivers its own (34), and from 52 stands on the cell where the hinderer's object
lies. The hinderer never gets it back.

**All 49 with `"outcome"`** (finished 2026-10-14; agent_1 / agent_2, bold where the run
ends otherwise than with the action chosen, that result in parentheses; s stand-off, c step
limit):

| agent_1 \ agent_2 | coop. | conflict | comp. | coerc. | exch. | none | hinder |
|---|---|---|---|---|---|---|---|
| **coop.** | 5 / 26 | **19 / – s** (22 / – s) | **19 / – s** (22 / 28) | 13 / 5 | 17 / 14 | 17 / 14 | 17 / 16 |
| **conflict** | 5 / 50 | **23 / 22** (– / – s) | **18 / 16** (– / – s) | **18 / 16** (– / – s) | **20 / 19** (– / – s) | **23 / 22** (– / – s) | **34 / – c** (– / – s) |
| **comp.** | 5 / 50 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 15** (9 / 16) |
| **coerc.** | 5 / 11 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 15** (9 / 16) |
| **exch.** | 5 / 17 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 15** (9 / 16) |
| **none** | 5 / 17 | 9 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 15** (9 / 16) |
| **hinder** | 5 / 36 | 36 / 25 | 32 / 23 | 36 / 30 | 25 / 15 | 25 / 15 | 27 / 16 |

Both goals done in 46 (42 with the action chosen), one in 3 (1), neither in 0 (6); 12
pairings end otherwise. Besides the conflict row: cooperation / competition loses its
finish (22 / 28 to 19 / –, a stand-off), and a hinderer in agent_2's seat finishes one
transition sooner (15 instead of 16) against the four robots that move alike.

## Carrying the partner's object at a move's cost (2026-10-14)

The user, on 2026-10-14, of conflict / coercion with `"outcome"`: why did the conflict robot
not pick up its partner's object (one cell from its start), carry it to its own object's
place, swap the two there and deliver its own -- keeping its partner's object far from its
destination on the way? In the solved models the plan exists and loses: at transition 1 the
best plan that starts by stepping onto O2's cell is 2.71 below heading for the doorway
(physical -2.17, social -0.54), at transition 11 the best that turns back is 1.87 below
(physical -1.91, social +0.04). Most of the price is `Config.foreign_carry_cost` = 2, a move
while holding an object that is not the robot's own costing two moves: the eight moves from
O2's place to O1's cost 8 instead of 4 (the plan in all: 13 transitions and a step cost of
10.5 against 9 and 4.5 going straight). The factor is ours (since the shared-object world of
2026-10-03); in the paper and in the authors' code either agent moves an object at the
ordinary cost. `examples/run_object_inference.py --foreign-carry-cost 1` sets it to the
paper's (folders `..._carry1`).

**Conflict / coercion at 1** (`..._seenoutcome_carry1/`): 29 / 37 against 18 / 16. The
conflict robot turns back at once (stepping back toward O2 is +2.78 in the social part
against +0.82 at 2), picks O2 up at 9 and keeps it at home; the coercer turns at the
doorway and takes O1 (12), and the conflict robot puts O2 back in the same transition; the
coercer hands O1 over at 16. Holding O2 at transition 11, the way east is +2.11 in the
physical part and -4.48 in the social: the conflict robot then holds its partner's
destination to be in the partner's room ((5, 5) 0.34, (7, 1) 0.30), where carrying O2 helps
it, and takes the coercer for a cooperator (0.67; fetching and handing over O1 reads as
help), against whom Table I's conflict above level 1 is paid for looking as if it fails --
it waits one cell from its destination from 23 to 28 (cooperation 1.00).

**All 49 at 1** (bold where the run ends otherwise than at 2, that result in parentheses):

| agent_1 \ agent_2 | coop. | conflict | comp. | coerc. | exch. | none | hinder |
|---|---|---|---|---|---|---|---|
| **coop.** | **13 / 40** (5 / 26) | **16 / 13** (19 / – s) | **16 / 13** (19 / – s) | **17 / 17** (13 / 5) | **17 / 17** (17 / 14) | **16 / 13** (17 / 14) | **16 / 13** (17 / 16) |
| **conflict** | **5 / – s** (5 / 50) | **27 / 36** (23 / 22) | **23 / 23** (18 / 16) | **29 / 37** (18 / 16) | **31 / 34** (20 / 19) | **27 / 34** (23 / 22) | **26 / 34** (34 / – c) |
| **comp.** | **11 / – c** (5 / 50) | **9 / 16** (9 / 17) | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 16** (9 / 15) |
| **coerc.** | 5 / 11 | **9 / 16** (9 / 17) | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 16** (9 / 15) |
| **exch.** | **31 / 25** (5 / 17) | **9 / 16** (9 / 17) | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 16** (9 / 15) |
| **none** | **12 / 13** (5 / 17) | **9 / 16** (9 / 17) | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 | **9 / 16** (9 / 15) |
| **hinder** | **5 / – s** (5 / 36) | **– / 28 c** (36 / 25) | **53 / 59** (32 / 23) | **35 / 30** (36 / 30) | **26 / 33** (25 / 15) | **26 / 33** (25 / 15) | **– / – c** (27 / 16) |

Both goals done in 44 (46 at 2), one in 4 (3), neither in 1 (0); 32 pairings end
otherwise. The partner's object is picked up 48 times (34 at 2), 30 of them by agent_1;
taken out of hands 15 times (12), put back 17 (9), swapped 4 (1), handed over 5 (2).

- *Helping is cheaper.* A cooperator in agent_1's seat picks agent_2's object up at 2 in
  every pairing. Against conflict, competition, none and hinder agent_2 takes it out of its
  hands at 9 and delivers it at 13, the cooperator its own at 16: the two stand-offs of the
  row are gone. Against coercion and exchange it first sets the object down on a wrong
  candidate (6), then 17 / 17. Two cooperators: 13 / 40 (5 / 26).
- *So is holding an object hostage.* The conflict robot picks O2 up at 9 (at 10 against a
  cooperator, once the cooperator has delivered its object at 5) in every pairing. Against
  the six others it puts it back at 12 and finishes 4 to 11 transitions later than at 2
  (against conflict, competition and coercion the partner has taken O1 by then), and
  conflict / hinder finishes (26 / 34) where it ran out of transitions. Against the
  cooperator it keeps O2 to the end (5 / –). A rival against a cooperator gets its object
  at 11 and then keeps the cooperator's (11 / –, step limit); so does a hinderer (5 / –).
  An exchanger and a cooperator swap and hand objects back and forth: 31 / 25 (5 / 17); a
  robot without social goal takes its object back from the cooperator at 6 and delivers it
  itself: 12 / 13 (5 / 17).
- *The hinder row is slower or worse in every pairing but coercion.* The hinderer no longer
  finishes against conflict (– / 28), two hinderers neither (– / –). Against a rival the
  hinderer swaps O2 onto O1's place at 40 and 48: 53 / 59.
- *Not changed*: the 16 pairings among competition, coercion, exchange and none, and
  coercion / cooperation (5 / 11).
- *Reading the partner* (`c21/readings.py`): 19 of the 98 final readings have the partner's
  goal alone in the lead, 18 more have it tied (conflict and hinder); a cooperator is now
  read as an exchanger 7 times in 14 and as a cooperator 6 times.

Against the six robots that do not cooperate the conflict robot does not carry the
partner's object across: it takes it and keeps it at home (put back and picked up again),
as the hinderer did at 2. Only against the cooperator, whose help finished its own goal at
5, does it carry the cooperator's object across, and it holds it to the end. Not run:
which of the two beliefs at transition 11 (the partner's destination, its social goal)
holds the carry back, which needs Q re-solved at that state with each belief swapped.

## A hinderer without a goal of its own (default since 2026-10-14)

The user, on 2026-10-14: "a hindering agent would just want to minimize the opposing
agent's task, they don't really care about their task." A hinderer's reward is now its moves'
costs and its partner's goal reward, nothing for its own object:

$$R_i = -c(a_i) - w\,r(\tilde g_j)$$

at every level: in the terms of `goals.py` $b = -1$ at level 1 and $\beta = -1$ above it, not
multiplied by the weight, which takes the $r(g_i)$ of Eq. 2 away exactly.
`Config.hinder_own_goal=False` is the default; `True` is the hinderer of before (in `TOLD`;
`examples/run_object_inference.py --hinder-own-goal`; the folders without `_spoiler`).

- At level 1 the hinderer is no longer the conflict robot ($-w\,r(\tilde g_j)$): a robot
  that models its partner holds seven level-1 models per physical goal of the partner where
  it held six (84 per robot on the two-room map, 72 before), and a rollout to the 60-step cap
  takes 30 to 40 minutes where one of before took 3 to 30.
- Its twin (the physical model a record compares it with) is its moves' costs only.
- The record's `own_goal` is False for it, and its `done_at` is None, even where its partner
  delivers its object (an event, not its goal); the rollout ends once its partner's goal is
  done. The GIF's goal panel reads "NO GOAL: keep agent_k from its goal", the results table
  "no goal".

**The 13 pairings with a hinderer** (`..._seenoutcome_carry1_spoiler/`: two rooms, both
robots at level 2, objects inferred, swaps and taking, belief $\tau$ 0.5, blocked moves seen
as staying put, carry cost 1, 60 transitions). The transition at which the hinderer's partner
is done:

| agent_1 / agent_2 | new hinderer | hinderer of before (`..._carry1/`) |
|---|---|---|
| none / hinder | never | 9 |
| hinder / none | 50 | 33 |
| cooperation / hinder | 22 | 16 |
| hinder / cooperation | never | never |
| conflict / hinder | 15 | 26 |
| hinder / conflict | never | 28 |
| competition / hinder | 31 | 9 |
| hinder / competition | 47 | 59 |
| coercion / hinder | 17 | 9 |
| hinder / coercion | never | 30 |
| exchange / hinder | never | 9 |
| hinder / exchange | 50 | 33 |
| hinder / hinder | – (neither has a goal: over after one transition) | – (both never; 60 transitions) |

- The partner is stopped in 5 of 12 pairings (1 of 12 before), held up in 5 more, and done
  sooner in 2 (conflict / hinder, hinder / competition).
- Where the hinderer wins it has the partner's object in hand early (transition 4 in four of
  the five, 9 against the cooperator) and is never robbed of it. Where it loses, either the
  partner took the hinderer's own object first (cooperation, conflict and coercion as
  agent_1), which in conflict / hinder makes the hinderer sure the partner wants that one
  (its belief that the partner wants its own object falls to 0.00 to 0.02 after the pick-up
  at 4), and later swaps it for its own on its place; or the partner took its object back
  out of the hinderer's hands (hinder / none and hinder / exchange at 46, hinder /
  competition at 46, competition / hinder at 28).
- hinder / exchange and hinder / none are the same rollout, action for action (both
  hinderers of before as well): an exchanger's level-2 term holds against an exchanging
  partner only.
- hinder / competition and hinder / coercion are the same for 40 transitions; the coercer
  tries to take its object back 12 times, the competitor 9. The competitor's take at 46
  succeeds because the hinderer stood still next to it (waiting $-11.31$, its best move
  $-11.76$); at the same cells at 59 the hinderer of the coercion run steps away (waiting
  $-12.70$, the step $-11.29$). One greedy rollout each: the difference is one decision, not
  competition against coercion.
- As agent_2 the hinderer picks its partner's object up and puts it back four times between
  transitions 4 and 12 (none, competition, exchange): near-ties of the argmax (none /
  hinder: picking up $-10.84$ against waiting $-10.89$, putting back $-31.84$ against
  $-31.89$, keeping it at 12 $-30.081$ against $-30.083$), tipped by its belief that the
  partner wants that object, which swings 0.97 / 0.47 with the partner's steps. Inertia
  (next section) is for that.

The other 36 pairings were not rolled out again; they may change too, since every robot's
model of a hinderer changed.

## Inertia (option, since 2026-10-14)

The user, of the robots that "ping pong": keep the previous action unless another beats it
by about one move cost. `Config.inertia` (in moves' costs; 0, none, by default;
`examples/run_object_inference.py --inertia 1`, folders `..._inertia1`): a robot does not
*undo* its last action -- step back to the cell it came from, or right after a hand action
that changed what it holds, change it back (place what it picked up, pick up what it placed,
give back what it took, take back what it gave, swap back; until 2026-10-16 the next "use")
-- unless undoing beats every other action by more than
$\text{inertia} \times c(\text{move})$. Then it takes the best of the others.

- It is the action rule only: the models by which the robots predict each other are Eq. 7's
  softmax without it, and the belief updates are those of before. Under
  `action_selection="sample"` the undoing action is drawn with its Q lowered by the margin.
- Keeping the previous action literally (waiting on after a wait, going on after a move) is
  not offered: in many states moving beats waiting by a fraction of a move's cost (0.14 for
  agent_1 at transition 5 of none / hinder), and a robot that had waited once would not set
  off again. Undoing is the flip; anything else stays free, so a robot can always wait.
- It looks back one transition: a robot that still wants to undo after a wait does
  (pick up, wait, put back is allowed). Looking further back would bring back the robot that
  never sets off.
- Records: per transition `inertia` (2,), True where the robot's best action by Q was to undo
  its last and it did something else; the ledger counts them; the GIF adds "(inertia)" to
  the action.
- On the 3 x 3 map (`test_inertia.py`, none / hinder at levels 2 / 2, 1 m cells) the hinderer
  picks its partner's object up and puts it back at once without inertia, and its partner is
  done at 8; with inertia 1 it waits in between (twice) and holds the object from transition
  7 on: the partner is not done in 14.

## Five hand actions in place of "use" (since 2026-10-16)

The user, on 2026-10-16: "make it such that we don't have use. Instead of use we have actual
discrete probabilities of pick, place, steal, exchange etc." A robot had six actions: four
moves, stay, and "use", which did the first hand act that applied (with full hands: swap,
put back, put down on a hand-off cell, hand over, set down for the partner; with free hands:
pick up, else take). Its partner, its models and the belief updates saw one "use", whatever
came of it. A robot now has ten actions (`constants.py`; a joint action is
$10 a_0 + a_1$):

| action | hands | where | what comes of it |
|---|---|---|---|
| pick (5) | free | on a cell where an object lies that it may hold | picks it up, its own object first |
| place (6) | holding X | X's place with nothing else on it; a hand-off cell; a candidate destination | puts X back; puts X down; sets X down for the partner (in vain on any other cell than the partner's destination, or with another object than the partner's) |
| take (7) | free | next to the partner, which holds something (`steal=True`) | takes it, unless the partner lets go of it in the same transition |
| give (8) | holding X | next to the partner (`hand_over=True`) | hands X over if the partner's hands are free; if both give, the two objects are swapped |
| swap (9) | holding X | an object's place where another object lies (`swap_on_place=True`) | the object lying there into its hands, X onto the cell |

- A hand action is offered (has a Q value, and a probability in the partner's models) only
  where it may do something: with the hands it needs, on the cell or next to the partner it
  is for (`ObjectWorld.hand_ok`: a bit per hand action, by configuration, cell and whether
  the partner is next to the robot). Acts for the partner -- give, and place on a hand-off
  cell or a candidate destination -- are a social robot's only, as before: a robot without
  a social goal in its place (its twin, a level-0 model) may pick up, put back, take and
  swap.
- Each hand action has its own Q and its own probability in Eq. 7's softmax, and the
  partner sees it as itself (under every `partner_observed`): a giving robot is told from
  one that sets down, and a pick-up from a take.
- Where "use" did one thing a robot may now choose between two: on a hand-off cell next to
  its partner, holding the partner's object, put it down or give it (before: put it down;
  nothing was handed over from a hand-off cell); on a candidate destination next to its
  partner, give or set down (before: hand over); with free hands where an object lies next
  to a partner that holds one, pick up or take (before: pick up); on X's place with another
  object lying there, swap -- X cannot be placed there (before: swap).
- The world: 100 joint actions where there were 36, and eight outcomes of a robot's step
  (arrived, stayed, blocked, and one per hand action) where there were four, so a table of
  what the hands do (configuration by the outcomes of both robots' steps) is four times as
  wide. `test_models.py` takes about 2.8 times as long.
- Inertia's undoing hand actions: placing what it picked up, picking up what it placed,
  giving back what it took, taking back what it gave, swapping back.
- Records: actions 5 to 9 (before: 5, "use"); the ledger counts `hand_actions` and, by
  name, `by_hand_action` (before: `uses`); the figure names the action ("give: hands O3 to
  agent_2"), its timeline marks P pick, D place, T take, G give, S swap (before: U), and
  its values panel has ten actions.
- The results in the other sections were rolled out before 2026-10-16, with "use": there
  "use" is the hands action of then.

**What changed in the tests' rollouts.** Every rollout the tests pin was rolled out again;
these changed (the rest are the same, transition for transition):

| rollout (test) | with "use" | five hand actions |
|---|---|---|
| two cooperators of Table I, tiny map, told the objects (`test_objects`) | 7 / 5 | 10 / 6: a set-down in vain first |
| ... inferring the objects | 12 / 21 as pinned; never / never just before 2026-10-16 | 18 / never |
| two selective cooperators, tiny map, told | 14 / 4 | 10 / 6, as Table I's |
| ... inferring | 7 / 8 as pinned; 8 / 8 just before 2026-10-16 | 7 / never |
| hinder / cooperation, w = 2, hinderer at level 1 (`test_defaults`) | never / never; the cooperator puts the hinderer's object back at 7 and takes it again at 13 | never / never; it keeps it from 2 |
| ... under the defaults of now | never / never; the cooperator delivers the hinderer's object at 5 | never / never; it takes it at 2 and keeps it |
| two cooperators, two rooms, told (`test_render`, "helping") | 22 / 5; two set-downs in vain | 31 / 5; three |
| race (level 1) / cooperation, w = 1 (`test_competition`) | 5 / 17 | 23 / 33: two set-downs in vain first, as at w = 0.8 |
| race then hinder (level 1) / cooperation, w = 0.8 | 5 / never | 19 / 25: each fetches its own |
| two competitors of the table, w = 30 | never / never; no object touched | never / never; they take each other's objects, trade (10), and agent_2 parks on agent_1's destination |
| race then hinder (level 2) / cooperation, w = 0.8 | 5 / never | never / never |
| race (level 2) / cooperation, w <= 2 | 5 / 17 | never / never: the competitor takes the cooperator's object at 2 |
| hinder / competition (level 2), w = 30 | the competitor takes the hinderer's object at 3 | at 2 |

**Why some of them stall: a model does not learn within a plan.** A robot's model of its
partner holds the partner's belief about the robot's destination fixed for the whole plan
(Eqs. 5-6 update it between transitions, not within one). A modelled partner holding the
robot's object on a wrong candidate therefore sets it down there again and again: every try
counts with the same belief mass. With "use" the robot could break that loop in its plan
by stepping next to the partner: "use" was then a hand-over (the hand-over came first), so
the modelled partner moved on. With "place" and "give" two actions, the modelled partner
keeps placing, and a level-2 robot that waits for its partner to bring its object expects
to wait forever. So the racing competitor at level 2 takes the cooperator's object at
once (w <= 2) instead of waiting for its own. (Against the race-then-hinder competitor at
w = 0.8, which took the cooperator's object at 2 with "use" as well, it is the cooperator
that changed: it no longer brings the competitor its object.) The real robots do not
loop: their beliefs are updated after every transition, and a set-down in vain rules its
cell out (in "helping" agent_2 tries three wrong candidates, each once). A fix would let a
model learn from its own set-down in vain within the plan.

## A softer likelihood: its own temperature, and a lapse (options, since 2026-11-04)

The user, on hinder / cooperation on `hall_crossing` (the testbed, tau 0.1, hazard 0.02):
"So why on step 39 to 40 does ti switch to G2 when its on the other side of the world and its
closest to G3?", then "how do we make it not collapse so low?"; on the two options proposed,
"Ok lets go ahead an implement this: A lapse ..." and "Ok lets implement this: Tempered
likelihood".

What the record shows (the likelihoods backed out of the beliefs, posterior over prior): the
belief about the partner multiplies, $\log b_T(g) = \log b_0(g) + \sum_t \log P(a_t \mid
s_t, g)$, and one step moves two hypotheses apart by about $e^{\Delta Q/\tau}$ -- at tau
0.1, $\tau = 0.05$ reward units, a gap of 0.2 between how two models rate a move is a factor
of 55, one move's cost (0.5) one of $2 \times 10^4$. At step 38 red, at the door of G3's room,
stepped left (toward its own G1): a detour under G3, which went from 0.96 to the hazard's
floor ($4 \times 10^{-7}$ of the others' likelihood); at 39 a step up fitted G0, G2 and G3
alike (1.00, 0.97, 0.95), and the posterior kept the order of its prior, G2 0.51. At step 51
red stood in front of G1 with G5 at 0.97: the set-down for green at G3 (43) and a step left
at (10,5) (48) had counted against G1 (x0.019, x0.018), and the moves since fitted G1 and
G5 alike. Where the level-1 hypothesis models misread a partner that plans at level 2
(carrying the object to G3 for green, keeping it away from green's takes), the true goal
pays, and tau multiplies it; the hazard bounds how far a hypothesis falls, not what one step
does; a window of the last steps would have kept every one of those steps.

Two options soften what one step does, off by default (Eqs. 5-6 as printed):

- **`inference_temperature`** ($T$, in the unit of `temperature`; `--inference-temperature`,
  `INFTAU=`): the likelihood is the hypothesis model's softmax at $\tau_{inf} = T\,c(\text{move})$
  in place of Eq. 7's tau,
  $$\psi_{inf}(a \mid s) = \frac{e^{Q(s,a)/\tau_{inf}}}{\sum_{a'} e^{Q(s,a')/\tau_{inf}}}
    = \frac{\psi(a \mid s)^\eta}{\sum_{a'} \psi(a' \mid s)^\eta},\qquad \eta = \tau/\tau_{inf},$$
  so a gap $\Delta Q$ counts $e^{-\eta \Delta Q / \tau}$: at tau 0.1 and $T = 0.5$ (Baker, Saxe
  & Tenenbaum 2009's fit to people, $\beta = 2$ per move's cost) the detour's $10^{-6}$ is
  about 1/16 and a factor 55 is 2.2. Still a distribution over the actions -- a hypothesis
  that spreads over many actions explains any one of them less well than one that predicted
  it, unlike the plain power $P^\eta$ of generalized Bayes. The right evidence counts less
  too. Only the belief update: the robots plan, act and predict each other at Eq. 7's tau
  (`SocialMDP.inference_taus`; `test_beliefs.InferenceTemperature`). Folders `..._inftau<T>`.
- **`belief_lapse`** ($\varepsilon$; `--lapse`, `LAPSE=`): each model's policy read as
  $\tilde\psi = (1 - \varepsilon)\psi + \varepsilon/|A|$, $A$ the actions the model allows --
  the partner picking one at random with probability $\varepsilon$ -- in both posteriors
  (`SocialMDP.lapsed`). One step then moves no hypothesis by more than about $|A|/\varepsilon$
  (100 at 0.05 and five moves); it caps the single crushing step (38) but not moderate ones
  (43, 48), and it caps a decisive step as much. Folders `..._lapse<E>`.

Both were devices of the belief layer until 2026-10-30 (an inference temperature of 0.5 and a
lapse of 0.05 by default there) and went with it; they are back as options. With both, the
lapse acts on the policy at $\tau_{inf}$.

- **`likelihood_power`** ($\alpha \in (0, 1]$; `--likelihood-power`, `LPOW=`; the user, of goal
  switching kept and the likelihood softened, $b_t(g) \propto \hat b_t(g)\,\ell_t(g)^\alpha$:
  "Could we try implementing something like this?"): a tempered update, generalized ("power")
  Bayes -- each posterior multiplied by its likelihood to the power $\alpha$, the physical
  one by the learning factor's, $\ell(d) = \sum_o b_t(o)\,P(a \mid o, d)$ holding an object
  ($\ell(o)$ with empty hands; each $\ell(o, d)$ under the joint posterior), after the idle
  mode's mix; the social one by Eq. 5's; the idle mode not. Every likelihood ratio between
  two hypotheses is taken to the power $\alpha$ exactly (`test_beliefs.LikelihoodPower`),
  where `inference_temperature` renormalises each model's softmax over the actions. The
  hazard stays the prediction step, $\hat b_t = (1-h)\,b_{t-1} + h/K$ -- the same family as
  a switch to one of the other $K - 1$ goals, $(1-h')\,b + h'(1 - b)/(K-1)$, with
  $h = h'K/(K-1)$. Folders `..._lpow<ALPHA>`. Replayed on the recorded evidence of the
  hall_crossing run (red's moves held fixed): at $\alpha = 0.25$ the frame after step 50
  reads G5 0.63 / G1 0.12 (Bayes: 0.97 / 0.015) -- far less sure of the wrong goal, still
  not on the right one before the move that tells them apart.

## A window on the evidence (option, since 2026-11-04)

The user: "Lets add a replay buffer such that the posterior only factors in the past 5 steps.
I.e for object we only factor in the past 5 steps that have had no object in hands and for
goal we only factor in past 5 steps where the object was held by the other agent. Make this
toggleable." `Config.belief_window` $n$ (`--belief-window N`, `WINDOW=N`; 0 by default, the
whole past): with the two posteriors apart (`belief_factored`, the default), each is built
afresh at every transition from the uniform prior and the likelihoods of the last $n$
transitions of its own stage only,

$$b_O(o) \propto \prod_{t \in W_O} \ell_t(o)^{\alpha}, \qquad
b_D(d) \propto \prod_{t \in W_D} \ell_t(d)^{\alpha},$$

$W_O$ the last $n$ transitions with the partner's hands empty, $W_D$ the last $n$ with an
object in them, $\ell_t$ the learning factor's likelihood as the update computed it then
(the other factor averaged out under the belief of then, Eq. 6 averaged over the social
posterior of then, after the idle mode's mix; `models.EvidenceWindow` keeps the two buffers),
$\alpha$ = `likelihood_power`. What the hands' outcomes ruled out (Eq. 5's transition term:
a set-down of its own object that did nothing, a finished goal, a hand-over taken or not)
stays out for good -- of the product, whose marginals are the two posteriors, as everywhere
under `belief_factored`. The goal hazard mixes the windowed posterior toward the uniform once
per transition, a floor of $h/|\text{support}|$; it is not compounded, since the posterior
is built afresh each time. The social posterior and the idle mode keep the whole past. With
$n$ at least the number of transitions, no hazard and nothing ruled out, it is the update
without a window, transition by transition (`test_beliefs.Window`). Not with one joint
posterior (`--joint-posterior`) or a hand hazard (`Config` raises). The record's
`belief_update` holds the window after each update (`window`: its entries, the transition of
each, what is ruled out); `tools/belief_step.py` prints it. Folders `..._window<N>`.

What it can and cannot do, on the hall_crossing testbed run (hinder / cooperation, tau 0.1,
hazard 0.02, one joint posterior): the gift at G3 (transition 43) that put G5 on top leaves a
window of 5 at transition 48; transitions 44-48 themselves (a step west, then north toward
the hinderer, 1.8 x 1 x 1 x 0.58 x 0.088 for G1 against G5) still make G1 / G5 about 1 / 11
at step 49 instead of 1 / 127 -- G5 still ahead until the move that tells them apart (51). A
window forgets old evidence; it does not soften the likelihood of the step in it.

## The partner's subgoals: a protected likelihood (option, since 2026-11-04)

The user, of the hall_crossing testbed run (hinder / cooperation, tau 0.1, hazard 0.02), where
green read red's set-down at G3 for green as "red's own goal is far" and stayed on G5 (0.94 at
step 49, red in front of G1): "So how do you think we solve Green thinking that red wants to
deliver to G5 and instead thinking its G1 since its closer"; "O so it goes back to that latent
problem ... trying to incorporate the subgoals"; then the formalization of it (ChatGPT Pro,
from our prompt: its Eq. 25), shown on that run, and "Ok let's go ahead and implement this".

Why it went wrong (red's 42 level-1 hypothesis models re-solved at transitions 42, 43 and 48
from the record): under the cooperation hypothesis red's willingness to give its object to
green depends on what its own goal is worth, which discounting makes less the farther the
goal (107-108 for G1 and G4, 101-104 for G0, G2, G5 at G3). Walking onto G3 to set O2 down
there for green was the best move under cooperation only for G3 and the farthest candidate,
G5 (likelihood of G1 against G5 3e-17); the set-down itself best for G0, G2 and G5 and 0.2 /
0.3 reward units worse for G1 / G4 (x0.017 / x0.002 at tau 0.05); the step toward green, who
red thought would take O2, x0.088 for G1. Red is a level-2 cooperator with a near goal that
gives anyway; the level-1 cooperator green reads it by cannot.

`Config.physical_likelihood="cut"` (`--physical-likelihood cut`, `PHYS=cut`; module
`social_pairings.subgoals`): the likelihood of the physical posterior, in Eq. 6's place, is

$$L(g) = \pi\,N_g(a) + (1-\pi)\,S(a),$$

the idle mode mixed in after as for every likelihood. $N_g$ is the partner's level-1 model
with physical goal $g$ and no social goal (its own task; the robot predicted at level 0 inside
it, as in every level-1 model); $S(a) = \sum_{\xi, z} \omega(\xi, z)\,\kappa_z(a \mid s) / (1-\pi)$
the social background, the policies of its subgoals weighted by the predictive belief
$\omega(\xi, z) = \sum_g b(g)\,b(\xi)\,P(z \mid g, \xi)$; $\pi = \sum_\xi \omega(\xi, \text{own task})$.
The subgoals of the partner toward the robot: deliver(c) and hand-off(h) (holding an object,
set it down on candidate c for the robot, or on hand-off cell h), meet (holding, go next to
the robot and hand it over), fetch(k) (hands empty, go and pick up object k), keepaway. Their
policies, a softmax at `option_temperature` x c(move) of $-c(a) - c(\text{move})\,J_z(s')$,
$J_z$ the moves left to the subgoal's act (keepaway: plus c(move) x the distance from the
robot), do not depend on the partner's own goal. Cooperation and exchange may pursue the
first four, weighted by the partner's belief about the robot's goal (that it wants k on c, or
k); conflict, competition and hinder keepaway; coercion and none none. A new subgoal's prior
under the partner's own goal $g$: $\chi(z \mid g, \xi, s) \propto w_\xi(z)\,e^{-\lambda\,\Delta C_g(z;s)}$,
$\Delta C$ the extra moves it adds to the partner's way to its own next waypoint -- moves, not
values, so the far goals are not the generous ones -- $\lambda$ = `option_detour` (log 2). A
subgoal starts with `option_start` (0.05) per transition, ends at its act (the set-down, the
hand-over, the pick-up) or when it is no longer possible, is interrupted with
`option_switch` (0.02); keepaway ends with `option_keepaway_end` (0.2).

The cut: the social posterior is Eq. 5 as before, on the level-1 social models; the subgoal
belief $P(z \mid g, \xi)$ (`PartnerBelief.options`) is updated within each $(g, \xi)$ -- its own
task by $N_g$, a subgoal by $\kappa_z$ -- and then predicted by the dynamics, so the social
models never multiply the physical posterior a second time (cutting feedback, Plummer 2015;
Jacob, Murray, Holmes & Robert 2017). What it guarantees: an act that only a subgoal explains
moves no physical hypothesis against another ($\pi \to 0$: $L$ the same for every $g$); for any
two hypotheses $L(g_1)/L(g_2)$ lies between 1 and $N_{g_1}/N_{g_2}$ -- the social background
weakens the own-task evidence toward neutral, never reverses it (`test_cut.py`). The robots
plan as before (inference only). Needs "none" among `social_hypotheses` and the partner's
social goal inferred. The record's `belief_update` holds `cut` (pi, S, $N_g$, $L$, the
expected subgoals and their $\kappa(a)$) and `subgoal_belief`; `tools/belief_step.py` prints
them. Folders `..._eq56cut` (`_optT`, `_optD`, `_optS`, `_optI` where a subgoal option is set).

On the hall_crossing run's recorded moves (a prototype of the same rule, red's "none" models
re-solved at every carrying step, no hazard): at transition 42 $\pi$ = 0.06 (green expects red
to be delivering to G3), $S$ = 0.54, $L$ = 0.498 for every destination (G3 0.549); at 43,
$\pi$ = 0.04, $L$ flat, G3 ruled out by the set-down that did nothing; at 48, $\pi$ = 0.91,
G1 / G5 = 1.66 (Eq. 6: 0.088), G4's $10^{-14}$ held at $L$ = 0.014 by the background. Green's
belief: G1 0.64 at step 49 (Eq. 6: G5 0.94), 0.83 at 51, 0.99 at 53; G5 nil from step 20 (red
stood next to it at 14-16 and did not deliver). What is left is the route: G4 0.95 at step
48 while red walks west along row 10 -- G4 straight ahead, G1's model split between two
routes -- until the step north. With the hazard 0.02, G1 0.61 at 51 (the floor keeps what
was ruled out). Learning the social goal from the subgoals alone (not built) was worse: red's
dodging read as keepaway, G4 0.985. Not run closed-loop yet; the subgoal settings are first
choices.

## The belief update's workings in the record (since 2026-11-04)

The user: "just have it put all diagnostics into the json file. All information for belief
calculation should be in the JSOn", and "record $Q_{d,\xi}(s_t,a_t)$, $\max_{a'}Q_{d,\xi}(s_t,a')$,
$P(a_t\mid s_t,d)$, $b_t(d)$, $b_{t+1}(d)$ ... separately for each social-goal hypothesis".
`Config.belief_detail` (default on; `--no-belief-detail`, `NODETAIL=1` leave it out): each
trace row's `belief_update[i]` is robot $i$'s update about its partner ($i$ as in `belief`):

- **units**: `tau` (the likelihood's, in reward units), `tau_planning` (Eq. 7's), `temperature`
  (in moves' costs), `inference_temperature`, `c_move`, `c_stay`, `lapse`, `hazards`; and
  `stage`, `likelihood`, `factored`, `stages`;
- **hypotheses**: the object and destination of each physical hypothesis $g$, the social goals;
- **beliefs**: `prior` ($b_t$: `phys` over $g$, `object`, `destination`, `social`, `idle`),
  `after_hand_hazard` where a hand hazard applied, `posterior` (before the hazards), `after`
  ($b_{t+1}$), `social_posterior`, `idle_posterior`;
- **every hypothesis model** (`models`, $(g, \xi, 10)$; `level0` under the level-0 likelihood):
  `q` (every action's Q at the state), `q_taken`, `q_best`, `best_action`, `q_gap`, `log_psi`
  ($\log\psi(a_t)$ at `tau`, before the lapse), `log_lik` (after it);
- **likelihoods**: `phys_log_lik` (Eq. 6 per $g$), `staged_log_lik` (the joint posterior's),
  `mix_log_lik` (with the idle mode), `idle_log_lik`, `social_log_lik` (Eq. 5), and
  `by_destination` / `by_object`: $P(a_t \mid d) = \sum_o b_t(o) P(a_t \mid o, d)$ (and per
  object the same over the destinations), with the idle mode, beside $b_t$, the posterior and
  $b_{t+1}$;
- **the hands**: `transition_term` (Eq. 5's, per $g$), `explained`, `kept`.

`null` in the JSON is $-\infty$ (an action not open to a model, a likelihood of 0) or a model
not consulted (its physical goal ruled out). Lists of numbers are now written on one line each
(`serialize.save_result`), which takes a record of before to about 70 % of its size; with the
workings a tier-scene rollout of 55 steps is some 4 MB (estimated from the tiny scene).
`tests/test_belief_detail.py` recomputes every update of a rollout from its record alone.

    python3 tools/belief_step.py RECORD.json --step 38 --dest G1 G4 G5    # the video's step

prints, for both robots (or `--robot agent_1`), per destination and object $P(a_t \mid \cdot)$,
$b_t$, the posterior and $b_{t+1}$; per social goal held at 0.01 or more (`--all-social`: every
one) and model, $Q(a_t)$, $\max Q$ and its action, the gap in reward units and moves' costs,
$\psi(a_t)$ before and after the lapse, and every action's Q; Eq. 5; the idle mode; and checks
Eq. 6 recomputed from the models against the recorded one.

## The object's posterior and the destination's apart (since 2026-11-04)

The user, on a video of hinder / cooperation on `head_on_blocked_goal` (tau 0.1, hazard 0.02),
where the hinderer came to believe that red was taking its object to G2 while red walked
toward G1: "Why near the end does green starting thinking that he wants G2 and not g1 and
g0? When red is near g0 and g1 we should have switched", then "WE SHOULD HAVE TWO POSTERIORS
one for goal and one for objects", and "When red has an object, green should freeze its
posterior on objects and when red doesn't we should not freeze the object posterior but we
should freeze the goal posterior".

What went wrong: until now the physical belief was one posterior over (object, destination)
pairs, staged by the partner's hands as below -- holding an object, the destination believed
under each object apart, $b(d \mid o)$, the object's mass kept -- and the bars of the
destination were that summed over the objects. Red had picked up the hinderer's object O1 at
step 3; the hinderer held "red wants O1" at 0.80, frozen from then on (red's
hands were never empty again: at step 33 it swapped O1 for O2 on O2's place, read as a
destination's evidence). Walking left toward G1 with O2 in hand fits no destination under
"O1"; $b(d \mid \text{O1})$ went to the destination least at odds with it, G2, the far one
whose policy is the flattest; under "O2" the walk fits G1 and G0, but $b(d \mid \text{O2})$
weighs 0.2 whatever it says. So the bars read G2 0.6-0.9 to the end.

Now (`Config.belief_factored`, the default) the physical belief about the partner is the
product of two posteriors, $b(o, d) = b_O(o)\,b_D(d)$:

- **Hands empty** (holding nothing, not lifting): its action updates the object's posterior,
  $$b_O'(o) \propto b_O(o) \sum_d b_D(d)\,\psi_{(o,d)}(a \mid s),$$
  and the destination's is frozen, $b_D' = b_D$.
- **Holding an object** (or lifting one): its action updates the destination's posterior,
  $$b_D'(d) \propto b_D(d) \sum_o b_O(o)\,\psi_{(o,d)}(a \mid s),$$
  and the object's is frozen, $b_O' = b_O$. A destination is weighed by how well the objects
  together explain the step: in the case above G1, which only "O2" explains, wins against
  G2, which no object explains (`tests/test_beliefs.py`,
  `test_the_goal_follows_the_object_that_explains_the_move`).
- $\psi_{(o,d)}$ is Eq. 6's likelihood as before (the level-1 hypothesis policies averaged
  over the social posterior; `physical_likelihood`), mixed with the idle mode as before.
- What the hands' outcomes tell (Eq. 5's transition term: a set-down for the partner that
  counted or failed, a hand-over in vain) acts on the joint and then on both posteriors,
  each the marginal of it (the frozen one of the prior times that term alone).
- The hazards act on the posterior that learns next: `goal_hazard` on the object's while
  the partner's hands are empty, on the destination's while it holds; `hand_hazard` on the
  object's before a pick-up or a take, on the destination's before a place, give or swap.
- The robot plans with the product (Eq. 4), and the level-1 models' beliefs about the robot
  that models them are built the same way.
- Without the stages (`belief_stages=False`) both posteriors learn from every action.
- `belief_factored=False` (`--joint-posterior`, `JOINT=1`, and in `TOLD`): the one joint
  posterior of before. Folders of the new default end in `..._eq56_objgoal`.

What the object's posterior does not yet see: a swap (holding O1, picking up O2 where it
lies) is read as a destination's evidence, the partner's hands being full; the object's
posterior stays where the first pick-up left it.

## Two posteriors, by the partner's hands (since 2026-10-17)

The user, on 2026-10-17: "as we move the prior for the goal is being accrued even though we
haven't picked up an object. If I am agent 1 and agent 2 hasn't picked up any object, we
should only be building a posterior for the objects. If agent 2 then picks up an object, we
stop utilizing the posterior for the objects and we start building the posterior for the
goals. Similarly for the actions, it should only depend on the posterior or prior developed
from these state dependent posteriors." Asked, the user chose: after the pick-up the object's
belief is frozen where it was (not set to the object in hand, which would lose the reading
"it is fetching my object for me"), and the social goal is inferred throughout, together with
whichever of the two is being built.

A robot's belief is over hypotheses $h = (o, d, t)$ about its partner: the object $o$ it
wants, the destination $d$ it takes it to, its social goal $t$ (the partner's physical goal
is $(o, d)$: "Which object the partner wants"). Eqs. 5-6 of before: $b'(h) \propto b(h)\,
\psi_h(a \mid s)$ after every action $a$ of the partner at state $s$. Now the update depends
on the partner's hands at $s$ (`Config.belief_stages`, the default; `False` is the update of
before, kept in `TOLD`):

- **Hands empty.** The action tells which object the partner wants, and its social goal,
  and nothing about its destination: the likelihood of every hypothesis is the mean, under
  the belief as it is, over the destinations of the hypotheses that share its object and
  social goal,
  $$\bar\psi(a \mid o, t) = \sum_d b(d \mid o, t)\,\psi_{(o,d,t)}(a \mid s),$$
  and the hypotheses with each destination are updated as one slice, scaled back to the
  belief that destination had: $b'(o, t \mid d) \propto b(o, t \mid d)\,\bar\psi(a \mid o,
  t)$ for every $d$, $b'(d) = b(d)$. (Without the mean, each destination's conditional
  belief about $(o, t)$ would be updated with that destination's own likelihoods and drift
  apart from the others', and the destination's belief would move with it once an object
  is in hand: the evidence the user does not want, by the back door.)
- **Hands full.** The action tells where the partner takes the object, and its social goal,
  under each object hypothesis on its own: $b'(d, t \mid o) \propto b(d, t \mid o)\,
  \psi_{(o,d,t)}(a \mid s)$ for every $o$, $b'(o) = b(o)$. The object's belief stays where
  the pick-up left it.
- The stage is the state's: a partner that puts the object back is read as before the
  pick-up again, its destination's belief kept where it was.
- What the objects tell is as before and above the stages: a set-down that counted keeps
  the hypotheses with that object and destination, one in vain rules its cell out for the
  object offered, a hand-over in vain rules the object out (`condition_on_placement`). A
  slice ruled out altogether gives its belief to the others.
- The idle filter (`idle_hypothesis`) and the lapse (`belief_lapse`) are as before, per
  hypothesis; the mean is of the models' likelihoods, before the idle mixture.
- Every belief: a level-2 robot's about its partner, a level-1 robot's about its partner's
  physical goal, and the level-1 models' beliefs about the robot that models them -- each by
  the hands of the robot it is about.
- The robot plans with the belief so built (Eq. 4's weights, `planning_weights`): in the
  plan the partner's destination is at its prior until the partner holds an object, and its
  object is as it was believed at the pick-up from then on.
- With the partner's object known (`partner_object="known"`) the first stage has nothing to
  build: nothing is learnt about the partner's destination until it holds an object.
- `examples/run_object_inference.py`: folders end in `_hands5_staged`; `--one-posterior`
  gives the update of before and the folder without the suffix.

**What changed in the tests' rollouts** (every one under the package's defaults rolled out
again; the told ones run under `TOLD`, with one posterior, and are as they were):

| rollout (test) | one posterior (2026-10-16) | two posteriors |
|---|---|---|
| two cooperators of Table I, tiny map, inferring the objects (`test_objects`) | 18 / never; each sure the other wants agent_1's object, agent_1 wrongly | 15 / 19: agent_2 sets agent_1's object down at 15 after two set-downs in vain, agent_1 puts agent_2's back at 6, agent_2 fetches its own; both know the other's object |
| two selective cooperators, tiny map, inferring (`test_objects`, `test_stages`) | 7 / never; the same wrong belief | 21 / 19: each puts the other's object back (7, 11) and fetches its own; both know the other's object; agent_2 takes agent_1 for a cooperator (0.85), agent_1 takes agent_2 for a competitor (0.86) |
| none / hinder, tiny map (`test_spoiler`, `test_inertia`) | the hinderer picks its partner's object up and puts it back at once; with inertia 1 it holds it from 7 on | the same (8 / –; with inertia – / –) |
| hinder / cooperation, six objects, objects known (`test_defaults`) | the cooperator takes the hinderer's object at 2 and keeps it | the same |

In the tiny-map runs the two posteriors are what let the cooperators finish: with one
posterior, agent_2's delivery of agent_1's object (what a partner that wants that object
would do too) left agent_1 sure that agent_2 wants agent_1's object, and agent_1 held
agent_2's object to the end; with the object's belief frozen at the pick-up, agent_1 goes
on believing that agent_2 wants its own object, and puts agent_2's back.

**hinder / coercion with the two posteriors** (`..._carry1_spoiler_inertia1_hands5_staged/`:
two rooms, both robots at level 2, objects inferred, swaps and taking, belief $\tau$ 0.5,
blocked moves seen as staying put, carry cost 1, inertia 1). The hinderer picks the
coercer's object up at 5 and holds it; the coercer, next to it, tries to take it 14 times
(16 to 48) and the hinderer steps away each time, until at 50 the hinderer stays (staying
is 0.4 above any move in its Q) and the coercer -- whose Q put stepping back first, which
inertia held it back from -- takes its object, and delivers it at 54. Throughout, the
hinderer's belief about the coercer's destination is the prior, uniform over the six
candidates -- the coercer held nothing until 50 -- and the coercer takes the hinderer for a
hinderer (1.00) from the pick-up on. With one posterior (`..._hands5/`) the hinderer put
the object back at 7, the coercer took the hinderer's object at 8 in turn, the hinderer
took it back at 14, and the coercer was done at 20. 55 transitions took 19 minutes on two
cores, about 20 s each, as before.


## Nothing automatic: pick and place by hand, a pick-up of three transitions (since 2026-10-18)

The user, on 2026-10-18: "Lets also make the pickup cost 3 and take 3 timesteps, Lets make
dropping an object cost 2, any sort of action besides movements should therefore cost 2."
Asked, the user chose: nothing automatic -- "you must move to the cell then select the
pickup dropoff etc action" (until then a robot took its OWN object on arriving at the cell
where it lay and set it down on arriving at its destination, without a hand action and for
free; only its partner's object went by the hand actions); the pick cost "3 at the start
then 0 and 0"; and "Swap should be its own action i.e it should be one timestep and it
should cost 2". "Stay" keeps its cost of 0.1 of a move (`wait_cost_rate`): waiting at twice
a move's price would have the robots pace instead of waiting.

| what | until 2026-10-18 | since |
|---|---|---|
| its own object | taken on arriving where it lies, set down on arriving at its destination, both automatic and free | by "pick" and "place", like any other object |
| pick | one transition, a move's cost | `pick_steps` = 3 transitions: the robot is *lifting* -- "pick" is the one action open to it (`ObjectWorld.busy`; no move, nothing else), the object is in nobody's hands yet (its partner can neither take it nor hand the lifter anything) -- and holds the object after the third; `pick_cost_rate` = 3 moves' worth on the first transition, the other two free |
| place, take, give | a move's cost each | `use_cost_rate` = 2 moves' worth each; "place" on its own destination with its own object finishes its goal (its done flag; the object stays there) |
| swap | a move's cost | 2 moves' worth, one transition |
| what is left of a robot's way, $D$ in $r = \rho\,\max(1 - D\,\text{res}/\delta, 0)$ | its moves | its moves and the transitions of its hands: the lift still ahead (3, or what is left of one begun) and the one set-down -- so that holding the object on the destination ($D = 1$) is worth less than done ($D = 0$), and placing beats hovering by about $0.125 / (1 - \gamma)$ |

What it is in the model (`Config.automatic_hands=False`, `pick_steps=3`, `pick_cost_rate=3.0`,
`use_cost_rate=2.0`, the defaults; `Config(**AUTOMATIC_HANDS)` is the rule of before, which
`TOLD` holds too):

- **Statuses.** An object being lifted has a status of its own, `lift0 + i * nstage + (s -
  1)`: lifted by robot $i$, $s$ steps in ($s = 1, 2$; `hands.lifter`, `hands.lift_stage`;
  `lift0` follows the lying statuses, `nstage` = `pick_steps` - 1). A robot lifts at most
  one object, with free hands. Two objects have 60 configurations instead of 20 (77 instead
  of 29 with `swap_on_place`); a lift happens where the object lies, so in a lifting block
  the lifter's cell is one of the few an object can lie on (its place, a hand-off cell, with
  swaps another object's place) and the other joint states of the block are no state
  anything reaches: they are not solved (`ObjectWorld.state_ok` (B, N^2), which the kernels
  read where they read `pair_ok` until now -- on the tiny scene of the tests 1,704 joint
  states are solved of 4,860, where the 20 configurations of before had 1,440). The tables
  that grow with the blocks ($B$) grow by the full factor all the same -- `pd` (G, B, K, N,
  10) above all: plan on 2.5 x the memory of before.
- **The rule** (`hands.hands_step`): before anything else, a lift in progress goes a step on
  with the lifter's "pick" (into its hands on the last) and is dropped where the object
  lies by anything else of the lifter's -- which the real robots never do, since nothing
  else is offered to them: it is what the inert partner of a level-0 model does (its hands
  do nothing), and what an idle partner would (the idle hypothesis has it lift on instead,
  `inputs_typed`). A move away cannot happen and leaves the status as it is, as does a
  lifter on no special cell (the kernels keep a block without a look-up where neither robot
  stands on one nor chose a hand action), on a cell no object can lie on (a candidate
  destination), or where another object lies on the place the object would go back to (one
  object to a place) -- lifts in states nothing reaches, which the tables cover all the
  same. "pick" with free hands on a cell where an object
  lies starts the lift. "place" holding its goal object on its own destination sets it down
  for its goal (`gsite`), before the put-back / put-down / set-down-for-the-partner of
  before. Nobody hands a lifting robot anything; nothing can be taken from it (it holds
  nothing yet). `Config.automatic_hands=True` is the rule of before, exactly.
- **Legality.** A lifting robot's `hand_ok` bits are "pick" alone, and the kernels close its
  moves (`q_row`, the uniform partner of a level-0 model, the idle partner, the solo
  values: `ObjectWorld.busy` (B, 2)). "place" is offered to a robot holding an object on any
  candidate cell -- which candidate is its destination, each model knows; elsewhere the
  action does nothing (and is not a set-down in vain: that is an act for the partner).
- **Costs** (`cost_c`): pick 3 moves' worth where the robot is not lifting yet, 0 while it
  is; place, take, give, swap 2 moves' worth; the moves and "stay" as before (a move with a
  foreign object in hand times `foreign_carry_cost`).
- **The way left** counts the transitions of the hands (the table above): class "fetch"
  (the object lying somewhere) adds the lift and the set-down, "hold" the set-down, the new
  classes "lift" (robot $r$ $s$ steps into its lift: the lift still to go and the set-down)
  and "heldlift" (its partner lifting its object, `held_reward="carrier"`: the carrier's way,
  its lift to go and its set-down). A goal is *finished* by its done flag alone (`fin_cls`;
  until then holding the object on the destination counted, the set-down being automatic):
  the race term, "finished" in the belief updates.
- **A robot finishing in its partner's model.** Robot $o$'s model does not know its
  partner's destination; until now the partner holding its object on its (hypothesised)
  destination was predicted to *park* there (`partner_parks`, `_park_policy`), which the
  reward classes read as finished -- the parent's dynamics never resolved the partner's
  own set-down. Now they do: in every model table the partner sets down for its own goal on
  ANY candidate cell, with whatever it holds that the modelling robot does not take
  (`model_tab`: `gsite` = -2 for the partner; `hands_step` above). It is exact as a mixture:
  under each hypothesis about the partner, its level-0 policy places only its goal object
  on that hypothesis's destination, so the parent's dynamics make it done exactly there --
  and placing the modelling robot's own object on a candidate that is not the robot's
  destination, which only a hypothesis that it wants that object does, counts as its own
  goal (the object is then gone for good: in that world the robot cannot finish, which that
  hypothesis says anyway). The park override of a partner holding its object on its
  destination is off (`_park_policy`, `_parked`: `automatic_hands` only); the park of a done
  partner stays. A level-0 model reads its rewards off one table for every destination
  (`level0_reward_tab`, no own set-down); where "place" with its goal object on destination
  $k$ does set down, the reward of goal $k$ is read in the done class (`inputs_level0`:
  until now the two were equal, $D$ being 0 either way). And what tells a robot's set-down
  for its partner (counted with the mass of the hypotheses whose destination is the cell)
  from a hand-over (the hypotheses whose object it holds) in the kernels is now the object
  it holds being set down for good in the table of success and not in that of failure
  (`kernels._counts`, `inputs_typed`, `SocialMDP.q_values`); until now it was the partner's
  done flag, set in the one and not the other -- which the partner's own set-down in the
  same transition now sets in both (the brute-force comparison caught it: a cooperator
  holding its partner's object on its own destination placed it with the wrong mass).
- **Evidence.** A lifting partner's "pick" is forced: the likelihood is 1 under every
  hypothesis, the idle one included, and the belief stands (`SocialMDP.observe`; in the
  kernels the idle partner, and robot o idle in its partner's models, lift on all the same,
  `pidle`, `pme`); the first
  "pick" -- the choice to lift that object -- is evidence as before (the staged update, with
  the hands empty: about its object). Hard evidence (a set-down, a hand-over in vain) still
  applies.
- **Records and figures.** A record's `trace[t]["status"]` holds the lift statuses, the
  event `"lifting"` marks a lift begun (the `△` of the timeline; `"picked up"` its end, as
  before), the ledger counts `lifts`, and the goal panel says "O1 is being lifted by it
  (2/3)"; the map draws a lifted object beside its lifter, raised by its progress and
  labelled "O1 2/3". `written_with` gives a record of before `automatic_hands=True`,
  `pick_steps=1`, `pick_cost_rate=1.0` (`ADDED_FIELDS`); `--resume` of the example scripts
  treats such a record as another configuration.

On the tiny 3 x 3 scene of the tests, cooperation / competition at (2, 2): each robot walks
two cells to its object, lifts it over transitions 4 to 6, carries it two cells and places
it at 9 -- 9 / 9, where the rule of before gave 5 / 5 (a pick-up and a set-down for free,
on arriving). A hinderer against a cooperator stands on the cooperator's object from
transition 2, lifts it at 7 to 9 and keeps it.

The five pairings the user asked for on 2026-10-17, on the two-room scene under the
server's options (`run_object_inference.py --scene two --levels 2 2 --objects inferred
--swap --steal --inference-temperature 0.5 --partner-observed outcome --foreign-carry-cost 1
--inertia 1`; folder `..._hands5_staged_byhand/`), against the same under the rule of
before (`..._hands5_staged/`):

| pairing (agent_1 / agent_2) | before | by hand | what changed |
|---|---|---|---|
| cooperation / competition | 31 / – (cap) | 17 / 30 | the cooperator steps onto the rival's object at 1 and leaves it (on the cell, "pick" 159.8 against walking away 160.6: three transitions and 3 + 2 moves' worth against an uncertain payoff), fetches its own -- lift 12-14, place 17; the rival waits at the doorway until the cooperator is done, then fetches and delivers its own. Before, the cooperator carried the rival's object off without ever learning its destination, and the rival, once sure of a cooperator, held the object and never delivered |
| coercion / hinder | – / no goal (cap) | – / no goal (stand-off) | the hinderer waits at its start until the coercer is two cells from its object, steps onto it and lifts it (8-10) -- the coercer next to it cannot take what is in nobody's hands -- then keep-away as before, six takes dodged |
| exchange / exchange | 10 / 11 | 13 / 15 | the same two deliveries, each two transitions of lift and one of set-down longer; no hand-over |
| exchange / conflict | 10 / 18 | 33 / 46 | the conflict robot carries its object to its destination at 25 and does not place it (Q: a step back 43.1, stay 37.9, place 36.8): against a partner it reads as cooperation 0.3 / exchange 0.24 its $-\tilde R_j$ holds its own reward, and finishing buys it nothing while the exchanger is unfinished. It walks the object back to its place (27), waits for the exchanger's delivery (33), then fetches and delivers (41-46). Under the rule of before the set-down was automatic on arriving and this never showed; it is the mirror of the rival that would not deliver |
| hinder / competition | no goal / 57 | no goal / 39 | the hinderer blocks by standing on the rival's object (from 3) instead of lifting it; the rival circles for 25 transitions, slips past at 33 while the hinderer guards the wrong row, lifts and delivers at 39 |

Two things to weigh. The cooperator no longer picks up its partner's object next to its
start, which is what the costs were for; the price is that helping has to be worth three
transitions and five moves' worth before it happens at all. And a Table I conflict robot
or rival facing a believed cooperator has no reason to finish its own task (its own reward
is in the cooperator's, which its term subtracts): with the automatic set-down that was
hidden, with the explicit one it is a choice the robot makes, and it chooses not to. If
conflict and competition are to finish against cooperators, that is a change to the term
(say $-r(\tilde g_j)$ against a cooperative partner, as at level 1), not to the costs.
About 21 to 30 s a transition on two cores, as before, the states of the lifting blocks
being few.

## Results, nobody told (the defaults)

Every level-2 robot infers its partner's destination and social goal from what the partner
does (a level-1 robot, a hinderer at its default level of then, only the destination;
"What a robot knows of its partner"): seven goals held possible, competition read as a
rival, an exchanger at level 1 the copy $n\,r(\tilde g_j)$. The default levels of then (2;
1 for a hinderer, the default until 2026-10-11: "Every robot with a social goal at level 2"
for a hinderer at level 2), greedy actions, at most 60 transitions. **Everything in this section was made with each
robot knowing which object its partner wants** (`partner_object="known"`, the default until
2026-10-09; with every object a candidate the six-object and exchange scenes need more
memory than the machine they were made on has: "Which object the partner wants" for what
inferring the object does), **and with Table I's cooperator** (`cooperate_with="all"`, the
default until 2026-10-11: "Whom a cooperator cooperates with" for what the cooperator of now
does). **Only the first four subsections are made under the defaults
of 2026-10-08 noon** (a physical goal only at level 0, the idle hypothesis on). The others were made before, with a robot without social goal at level 2
and no robot allowing for an idle partner (`physical_level=2, idle_hypothesis=False`), and
were not rolled out again. `examples/run_all_pairings.py --weights 30 30` wrote the record
and the annotated GIF of every pairing to `output/all_pairings_w30_30_rival_inferred/`
(`--scene exchange`: `output/all_pairings_exchange_w30_30_rival_inferred/`); the figure's
panel "... and about its social goal" shows each robot's belief as it goes. A cell is the
transition at which agent_1 / agent_2 finished its own goal, "–" = not finished when the
rollout ended (25 transitions without progress, or the 60th); agent_1's goal by row,
agent_2's by column.

### Under the defaults of 2026-10-08 noon: five pairings at $w$ = 1 and 30

Six-object scene, nobody told, a physical goal only at level 0, the idle hypothesis on
(`output/five_pairings_w1_1_rival_inferred_idle/`, `.../five_pairings_w30_30_rival_inferred_idle/`).
$w = 1$ is the paper's: its Eq. 2 has no weight on the social term ("Options that
matter").

| agent_1 / agent_2 | levels | $w = 1$ | $w = 30$ | $w = 30$ before noon |
|---|---|---|---|---|
| cooperation / cooperation | 2 / 2 | 22 / 5 | 8 / 17 | 6 / 13 |
| hinder / cooperation | 1 / 2 | 5 / – | 7 / – | 7 / – |
| competition / competition | 2 / 2 | 26 / 28 | 13 / 15 | – / – (stand-off) |
| competition / cooperation | 2 / 2 | 5 / – | 5 / – | 5 / – |
| none / none | 0 / 0 | 9 / 11 | 9 / 11 | – / – (levels 2 / 2, stand-off) |

- *Two cooperators* help each other. At $w = 30$ each delivers the other's object (8, 17).
  At $w = 1$ agent_1 delivers agent_2's at 5, while agent_2 tries agent_1's object on two
  wrong candidates first (6, 10; it does not know the destination) and then hands it over
  (17); agent_1 sets it down at 22.
- *Two rivals* no longer stand off at $w = 30$: they pass (13 / 15). At $w = 1$ each takes
  the other's object and puts it back, two or three times, before both deliver their own
  (26 / 28).
- *Against a hinderer or a rival the cooperator does not finish*: the hinderer keeps its
  object from transition 2; the rival, sure it faces a cooperator (1.00), takes it (13 at
  $w = 30$, 6 at $w = 1$) and keeps it. The idle hypothesis does not change that: the
  partner is not stuck, it is doing what its term pays for.
- *Two robots with a physical goal only* are two plain MDPs: 9 / 11 at every weight.

### Under the defaults of 2026-10-08 noon: every pairing on the six-object scene, $w = 1$

The paper's weight, every script's default since 2026-10-08 evening.
`python examples/run_all_pairings.py` (written to `output/all_pairings_w1_1_rival_inferred_idle/`
with `--output`: 49 GIFs, the records, `matrix.png`, `matrix.md`, `summary.json`). The
solver's residual check passes in all 49.

| agent_1 \ agent_2 | none | cooperation | exchange | coercion | competition | conflict | hinder |
|---|---|---|---|---|---|---|---|
| **none** | 9 / 11 | 33 / 45 | 9 / 11 | 14 / 16 | 13 / 21 | – / – | – / – |
| **cooperation** | 41 / 29 | 22 / 5 | 22 / 10 | 29 / 5 | – / 5 | – / 5 | – / 5 |
| **exchange** | 9 / 11 | 24 / 25 | 9 / 11 | 15 / 16 | – / – | 54 / 53 | – / – |
| **coercion** | 15 / 16 | 5 / 6 | 24 / – | 34 / 29 | – / – | – / – | – / – |
| **competition** | 11 / 11 | 5 / – | 11 / 11 | 13 / – | 26 / 28 | – / – | – / – |
| **conflict** | – / – | 5 / – | – / – | – / – | – / – | – / – | – / – |
| **hinder** | – / – | 5 / – | – / – | – / – | – / – | – / – | – / – |

Both goals done in 20 pairings, exactly one in 8, neither in 21 (at $w = 30$ before noon:
10, 12, 27; told: 21, 6, 22).

- *Among none, cooperation, exchange, coercion and competition* both finish in 19 of the
  25 pairings. The six others: cooperation / competition (– / 5), competition /
  cooperation (5 / –), exchange / competition (– / –), coercion / exchange (24 / –),
  coercion / competition (– / –), competition / coercion (13 / –).
- *Two exchangers* each fetch their own object, as two robots without social goals do
  (9 / 11, the same events); *two rivals* each take the other's object and put it back
  two or three times, then deliver their own (26 / 28).
- *With a conflict robot or a hinderer in the pair* (24 pairings) both finish only in
  exchange / conflict (54 / 53, after five put-backs). In the other 23 at most the hostile
  robot's goal is done, and only next to a cooperator: the hostile robot takes the
  cooperator's object at 2, the cooperator delivers the hostile robot's at 5 and never
  gets its own.

### Under the defaults of 2026-10-08 noon: how well the robots infer each other's goals, $w = 1$

From the 49 records of the subsection above: `python examples/measure_inference.py
output/all_pairings_w1_1_rival_inferred_idle` (it writes `inference.json`, one row per
robot, and `inference.png`, the confusion matrix, into the folder). A robot
infers its partner's social goal at level 2 (70 robots: every goal but none and hinder) and
its partner's destination at levels 1 and 2 (98). Conflict and hinder are one model at
level 1, so they count as one class; the prior is 1/7 per goal, 2/7 for that class.

| partner's true goal | ranked first at the end | mean belief in it | ranked first instead |
|---|---|---|---|
| none (level 0) | 2 / 10 | 0.24 | competition 5, coercion 3 |
| cooperation | 10 / 10 | 1.00 | |
| exchange | 2 / 10 | 0.26 | competition 5, conflict / hinder 2, coercion 1 |
| coercion | 0 / 10 | 0.02 | cooperation 7, conflict / hinder 2, none 1 |
| competition (rival) | 1 / 10 | 0.05 | conflict / hinder 5, cooperation 2, exchange 2 |
| conflict / hinder | 18 / 20 | 0.90 | cooperation 2 |

33 of 70 rank the true goal first (at $w = 30$ before noon: 37 of 84), 30 of 70 hold it at
0.5 or more at the end. The partner's destination is ranked first in 63 of 98 final beliefs
(mean belief 0.57; at $w = 30$ before noon 57 of 98, 0.46).

- *What goes wrong is mostly which goals act alike.* Against a partner without social goal
  a robot puts 0.94 of its belief on none, coercion and competition together: at level 1
  the three pursue their own goal (no term, $+r(g_i)$, the race), and the robot cannot
  split them. Two exchangers act as two robots without social goals here (9 / 11, the same
  events), and an exchanging partner keeps 0.58 on those three.
- *A coercing partner is read as a cooperator* (0.68): against a partner it takes to be
  cooperative, Table I's coercion holds $+\tilde R_j + r(g_i)$, and it does help: it hands
  over or delivers its partner's object in coercion / none, coercion / cooperation,
  none / coercion, cooperation / coercion and competition / coercion. *A rival is
  read as hostile* (0.49 on conflict / hinder): against a partner it takes to be cooperative
  its term is $-\tilde R_j$, and it works against it.
- *Cooperation and the hostile goals are recognised*: every cooperator, 18 of 20 conflict
  robots and hinderers (two conflict robots are read as cooperators).

### Under the defaults of 2026-10-08 noon: the congruence tier scenes, $w = 1$

`test_tiers.py` (2026-10-03) places two robots on the scene_048 house map at 0.25 m, two
placements per congruence tier: its meeting suite, for the goals that act on the partner
(cooperation, conflict, competition, coercion; the tier says how far the fastest routes are
from meeting -- high: they cannot both be driven, someone must wait; medium: they meet with
room to pass; low: they do not meet, a detour under 1.5 m would bring them together), and
its exchange suite (high: swapping tasks is shorter for the two together; medium: it makes
no difference; low: it costs them under 1.5 m). The twelve of the low, medium and high
tiers, here as `examples/tier_scenes.py` holds them:

- *As drawn, none is fair.* A* gives one robot's task (start -> own object -> own
  destination, alone on the map) 1.25 to 2.73 times the length of the other's.
- *On the house at 0.5 m.* At 0.25 m the house has 276 free cells, 76,176 pairs of robot
  cells per configuration of the objects, some 12 times the 0.5 m house, where a level-2
  transition already takes about 12 s. At 0.5 m (79 free cells): each cell joins 2 x 2 of
  the 0.25 m map and is free where two of them are, with the wall between the middle and
  the right room put back. Coarser: the left nook joins the hall, the bottom-left alcove is
  one cell, the top room's doorway is 1 m wide (0.75 m).
- *Made fair*: by the smallest moves of the task cells (summed over the moved cells) that
  bring the two A* task lengths within 10 % of each other, keep the scene's tier at 0.5 m
  by test_tiers' own rules (its geometry functions, copied), keep every coincidence of the
  placement (R2's object on R1's start, ...), keep each moved cell in its room (where no
  placement does, same_room_later and doorway_passing, only the objects stay in theirs),
  and leave each robot's task possible with its partner parked on its destination. That
  last rule was added after head_on_blocked_goal and equally_far had been rolled out once
  without it: with the alcove one cell wide, a destination in its entrance walled the
  other robot out (R1 of head_on_blocked_goal never finished against a rival, – / 22);
  those two were rolled out again. Ratios after: 1.00 to 1.10.
  `output/tier_scenes/fair_scenes_meeting.png` and `fair_scenes_exchange.png` show the
  twelve with their A* routes and the moves.

`python examples/run_tier_scenes.py`: each scene with two level-0 robots without social
goals, and with its suite's goals, both robots the same (written to
`output/tier_scenes_w1_1_rival_inferred_idle/`: 42 records, 30 GIFs, `results.md`,
`results.png`, `summary.json`). The solver's residual check passes in all 42. A cell: the
transition at which R1 / R2 finished; A*: the two task lengths.

| scene | tier | A* | none (level 0) | cooperation | competition | conflict | coercion | exchange |
|---|---|---|---|---|---|---|---|---|
| near_miss | low | 16 / 17 | 16 / 17 | 59 / 52 | 16 / 17 | 16 / 17 | 16 / 17 | |
| same_room_later | low | 16 / 15 | 16 / 15 | 49 / – | 16 / 15 | – / – | 16 / 15 | |
| doorway_passing | medium | 13 / 12 | 13 / 12 | 38 / 25 | 13 / 12 | 13 / 12 | 13 / 12 | |
| hall_crossing | medium | 17 / 16 | 18 / 16 | 40 / 24 | 17 / 16 | 17 / 16 | 17 / 16 | |
| head_on_blocked_goal | high | 23 / 23 | 23 / 25 | 40 / 36 | 23 / 24 | 24 / – | 23 / 24 | |
| head_on_corridor | high | 13 / 14 | – / – (stand-off) | 59 / 21 | 15 / 14 | 18 / 15 | 15 / 14 | |
| own_object_closer | low (exchange) | 14 / 14 | 14 / 14 | | | | | 14 / 14 |
| own_object_closer_far | low (exchange) | 15 / 15 | 15 / 15 | | | | | 15 / 15 |
| equally_far | medium (exchange) | 18 / 18 | 18 / 18 | | | | | 18 / 18 |
| start_together | medium (exchange) | 16 / 15 | – / – (stand-off) | | | | | 18 / 15 |
| objects_crossed | high (exchange) | 18 / 17 | – / – (stand-off) | | | | | – / – (stand-off) |
| objects_crossed_long_haul | high (exchange) | 23 / 21 | – / – (stand-off) | | | | | 42 / 42 |

- *At $w = 1$ the tier does not decide whether cooperators act.* Two cooperators take each
  other's objects in all six meeting scenes, the low ones as well, and finish 7 to 46
  transitions late (same_room_later: R2 never). Not knowing its partner's destination (six
  candidates), a robot carrying its partner's object tries one to four wrong candidates
  first.
- *Rivals and coercers act as if alone*: on time in the low and medium scenes, at most 2
  late in the high ones, and nobody takes an object. Conflict robots as well in three of
  the six scenes, and 5 / 1 late in head_on_corridor; in the two others they work against
  each other: in same_room_later (low) each takes the other's object, puts it back and
  takes it again, and neither finishes; in head_on_blocked_goal (high) R1, once done,
  leaves its destination for R2's, then keeps in R2's way (14 blocked moves; – for R2).
- *Two robots without social goals stand off where their fastest routes collide*:
  head_on_corridor (high), and three exchange scenes whose routes collide although their
  exchange tier does not count it (start_together, medium; objects_crossed and
  objects_crossed_long_haul, high; test_tiers' meeting geometry gives them 0.5 to 1 m of
  waiting). A level-0 robot takes its partner for one that moves at random, so each keeps
  pushing into the other. Every level-2 pair passes in head_on_corridor.
- *Two exchangers never exchange here.* In every exchange scene each fetches its own
  object, as two robots without social goals would (on time in the low and medium ones;
  +2 for R1 in start_together, where two robots without social goals stand off). In the two high scenes, where swapping
  tasks would save 9.0 m and 8.0 m together, they meet head-on at the middle room's
  entrance: objects_crossed stands off to the end, both robots pushing (the idle
  hypothesis stays at its 0.02 prior: a robot that pushes is not staying put);
  objects_crossed_long_haul from transition 5 until R1 waits at 21, R2 reads it as idle
  (0.96) and goes round (42 / 42).
- *What the robots end up believing* (60 robots at level 2): the partner's goal is ranked
  first by 24. Cooperation 7 of 12 (five read as exchange), competition 10 of 12 (but at
  0.27 to 0.49: here it is the goal of a robot that minds its own task), conflict 7 of 12,
  coercion and exchange 0 of 12 (read as competition or conflict).

### Every pairing on the six-object scene, $w = 30$ (before 2026-10-08 noon)

| agent_1 \ agent_2 | none | cooperation | exchange | coercion | competition | conflict | hinder |
|---|---|---|---|---|---|---|---|
| **none** | – / – | 5 / 17 | 5 / 17 | 33 / 30 | – / – | – / – | – / – |
| **cooperation** | – / 13 | 6 / 13 | 5 / 8 | 7 / 10 | – / 46 | – / 21 | – / 9 |
| **exchange** | – / – | 34 / 33 | – / – | 29 / 16 | – / – | – / – | – / – |
| **coercion** | – / – | 29 / – | 7 / 7 | 8 / 8 | – / – | – / – | – / – |
| **competition** | – / – | 5 / – | 5 / – | 26 / – | – / – | – / – | – / – |
| **conflict** | – / – | 53 / – | 35 / – | 31 / – | – / – | – / – | – / – |
| **hinder** | – / – | 7 / – | – / – | – / – | – / – | – / – | – / – |

Both goals are done in 10 pairings, exactly one in 12, neither in 27. Told, the same 49 end
21, 6 and 22 ("Every pairing at $w = 30$ (told)"); with the paper's five goals held possible
(2026-10-07) 13, 8 and 28.

- *Helping works among the kinds that help.* The ten are seven of the nine pairings among
  cooperation, exchange and coercion (not exchange / exchange, not coercion / cooperation)
  and a robot without social goal seated first next to each of the three. Two of them are
  done only here: two coercing robots, and coercion next to an exchanger, deliver each
  other's object (8 / 8, 7 / 7; told: – / –). The other eight are done when told as well,
  six of them later here (the last goal at 10 to 34; told, 5 to 19).
- *Two robots that meet head-on in the doorway push into each other for good*: none /
  none, none / competition, competition / none, competition / competition and coercion /
  competition, 25 or 26 blocked moves each and no object touched. A blocked move leaves a
  robot where staying would, at ten times the cost, so in a robot's plan a push can pay
  only if its model of the partner, one level down, may step away. In none / none, from
  transition 6, pushing is ahead of staying by 0.41 and 0.36 in Q and of every move that
  gives way by 1.8 and 1.7; the Q values at transition 16 are those of transition 6 to the
  third decimal, and no belief moves by more than 0.09 in the 25 transitions. How the two
  got there is the item below: agent_2 steps onto the place of agent_1's object at
  transition 1, and agent_1 holds it for a cooperator, a conflict robot or a hinderer at a
  third each and waits. Two robots without social goals finish at 9 / 11 without conflict
  and hinder among the goals held possible, and at $w$ = 1, 2 and 5 (not at 10). With the
  idle hypothesis two rivals finish (13 / 15) and two robots without social goals still
  stand off. In the head-on state of none / none each robot's model of the other has it
  step aside (agent_1 gives agent_2 0.993, agent_2 gives agent_1 0.999), and each pushes,
  which both models put below 5e-4: no hypothesis held possible explains a push, so
  the likelihood of every hypothesis is the lapse's and the beliefs stay. With sampled
  actions (seed 1) both pairs finish, but they never meet head-on: early draws change
  the approach (in none / none they pass side by side through the doorway's two cells at
  transition 4). Once head-on, a step aside is below 1e-7 per transition under each
  robot's own softmax: sampling avoids the stand-off, it does not break it.
- *Each takes the other's object and puts it back, and then neither does its own task*:
  exchange / none, coercion / none, conflict / none, exchange / exchange, exchange /
  conflict, coercion / conflict and conflict / conflict. On this scene each robot starts
  next to its partner's object, and the way to its own may lead over that object's place.
  A robot without social goal seated second takes that step at transition 1 (0.22 ahead
  of the other way in Q, a near tie), and so does every robot of these pairings. After it
  each holds the other for a cooperator, a conflict robot or a hinderer, a third each,
  every other goal at 0.00. At transition 2 or 3 each picks the other's object up (5.4 to
  7.3 ahead of staying in Q for the exchangers and the robots without social goal), puts
  it back by transition 10, ends holding its partner for a conflict robot or a hinderer
  (0.39 to 0.50 on each of the two) and does not go for its own object. Holding conflict
  and hinder possible does it, above $w = 1$: without them among the goals held possible
  exchange / none, exchange / exchange and coercion / none finish (31 / 32, 24 / 13,
  30 / 35); with the paper's five, which hold conflict, they do not. At $w = 1$ the three
  finish (9 / 11, 9 / 11, 29 / 32); at 2, 5 and 10 one run of the nine does (exchange /
  exchange at 5: 40 / 37). conflict / none, whose conflict robot is hostile, finishes in
  neither case.
- *A hostile partner is recognised, and stops its partner as when told.* Every level-2
  robot next to a conflict robot or a hinderer ends holding it for one of the two (24 final
  beliefs of 24; the two are one model, "What a robot knows of its partner"), and no
  pairing with one of them has both goals done. A partner that is neither ends held for
  one of the two as well in 16 of the 60 final beliefs, among them all nine such beliefs
  of the seven pairings above. In all six pairings of a cooperator with a conflict robot, a
  hinderer or a rival, in either seat, the cooperator delivers its partner's object (at 5
  to 53) and never gets its own.
- *The other goals are rarely told apart.* Of the 84 final beliefs of level-2 robots, 37
  put the partner's true goal first (counting conflict and hinder as one): all 24 about a
  hostile partner, 6 of 12 about a cooperator (5 more on exchange), 5 of 12 about a robot
  without social goal, 2 of 12 about an exchanger (6 on cooperation), none of 12 about a
  coercing robot or a rival. In the ten pairings with both goals done, 6 of 20.
- *A rival judges whom it faces by what it sees.* Seated first next to a cooperator or an
  exchanger, each of which delivers the rival's object at 5, it ends sure it faces a
  cooperator (1.00) and works against its helper. It takes the exchanger's object at 13
  and keeps it. From transition 6 it stands next to the doorway in the cooperator's room;
  the cooperator sets out for its own object at 7, stops next to the rival at 8 and turns
  back at 13, and from 21 the rival steps to and fro between that cell and the doorway.
  5 / – both times; told, 5 / – against the cooperator (whose object the rival takes) and
  9 / 11 against the exchanger, which a told rival races. Seated second it holds a
  cooperator for one (1.00), which delivers its object at 46 and never gets its own
  (– / 46), and an exchanger for an exchanger (1.00), and neither goal is done.

### Every pairing on the exchange scene, $w = 30$ (before 2026-10-08 noon)

Put down and pick up (`--scene exchange`):

| agent_1 \ agent_2 | none | cooperation | exchange | coercion | competition | conflict | hinder |
|---|---|---|---|---|---|---|---|
| **none** | – / – | 14 / – | – / – | – / – | – / – | – / – | – / – |
| **cooperation** | – / 14 | 21 / 20 | 22 / 21 | 29 / 14 | – / – | – / – | – / 16 |
| **exchange** | – / 18 | 17 / 16 | **17 / 16** | 22 / 19 | 48 / 45 | – / – | – / 27 |
| **coercion** | – / – | 14 / 27 | 24 / 23 | – / – | – / – | – / – | – / – |
| **competition** | – / – | – / – | – / – | – / – | – / – | – / – | – / – |
| **conflict** | – / – | – / – | – / – | – / – | – / – | – / – | – / – |
| **hinder** | – / – | 16 / – | 16 / – | – / – | – / – | – / – | – / – |

Both goals are done in 9 pairings, exactly one in 7, neither in 33 (the paper's five
goals: 8, 7 and 34; told: 3, 9 and 37). Eleven cells differ from the five goals. In six
both goals are done either way, at other transitions, and in one only agent_2's
(exchange / none, – / 18). An exchanger seated second next to a robot without social goal
takes that robot's object at 6 and keeps it (none / exchange: – / –, was 16 / –); new are
an exchanger and a rival that trade late (48 / 45, was – / 17), and an exchanger that
gives a hinderer its object in either seat (– / 27, 16 / –). In the 25 pairings among
none, coercion, competition, conflict and hinder no object is touched: each robot takes at
most four steps and waits. Its partner has not come for its object, which rules out the
two helping kinds, and leaves the other five goals at 0.20 each: nothing the partner did
tells them apart. With hand-overs allowed as well (`--scene exchange --hand-over`, records
and table only): both goals done in 11 pairings, exactly one in 8, neither in 30 (five
goals: 10, 5 and 34).

### The eight pairings of `run_pairings.py` (before 2026-10-08 noon)

Six-object scene, the default levels, competition read as a rival
(`run_pairings.py --weights W W`; told, under `"race_then_hinder"`: "Results (told)"):

| pairing | levels | $w = 0.8$ | $w = 1$ | $w = 2$ | $w = 30$ |
|---|---|---|---|---|---|
| 01 competition / competition | 2 / 2 | 36 / 44 | 20 / 29 | – / – | – / – |
| 02 cooperation / cooperation | 2 / 2 | 22 / 5 | 22 / 5 | 15 / 14 | 6 / 13 |
| 03 competition / cooperation | 2 / 2 | 12 / – | 5 / – | 38 / – | 5 / – |
| 04 exchange / cooperation | 2 / 2 | 5 / 19 | 5 / 19 | 18 / 29 | 34 / 33 |
| 05 conflict / cooperation | 2 / 2 | 5 / – | 5 / – | 38 / – | 53 / – |
| 06 hinder / cooperation | 1 / 2 | 5 / – | 5 / – | 7 / – | 7 / – |
| 07 hinder / competition | 1 / 2 | – / – | – / – | – / – | – / – |
| 08 hinder / none | 1 / 2 | – / – | – / – | – / – | – / – |

The rows with a competitor under the other four readings, goals inferred:

| pairing | `competition=` | levels | $w = 0.8$ | $w = 1$ | $w = 2$ | $w = 30$ |
|---|---|---|---|---|---|---|
| 01 competition / competition | `"race_then_hinder"`, `"race"` | 1 / 1 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| | `"table"` | 2 / 2 | 6 / 9 | – / – | 5 / 11 | – / – |
| | `"table_hostile"` | 2 / 2 | – / – | – / – | – / – | – / – |
| 03 competition / cooperation | `"race_then_hinder"` | 1 / 2 | 36 / 33 | 5 / – | 33 / – | 12 / – |
| | `"race"` | 1 / 2 | 19 / 31 | 26 / 26 | 5 / 17 | 5 / 17 |
| | `"table"` | 2 / 2 | 6 / 9 | 9 / 11 | 46 / 45 | 18 / 21 |
| | `"table_hostile"` | 2 / 2 | 6 / – | 9 / – | 6 / – | 53 / – |
| 07 hinder / competition | every reading | 1 / 1 or 1 / 2 | – / – | – / – | – / – | – / – |

- *A competitor of a race form reasons at level 1*, where it has no belief about its
  partner's social goal: two of them finish at 9 / 11 at every weight, as told. Two rivals,
  at level 2, do not (told: 9 / 11 at every weight). At $w$ = 0.8 and 1 each takes the
  other's object at transition 2 or 3 and puts it back, and they finish late; at $w = 2$
  they do the same and then neither moves; at $w = 30$ they meet in the doorway and stand
  off.
- *The hinder pairings are as told*: the hinderer takes its partner's object at
  transition 2 and holds it; a cooperator still delivers the hinderer's object (at 5 or 7).
- *A cooperator next to a robot that works against it* (a rival or a conflict robot)
  delivers that robot's object (at transition 5 to 53; told, 5 to 7) and never gets its
  own, at every weight, as told. Under `"race"`, which never works against its partner,
  both goals are done at every weight; under the table as printed, whose competitor helps
  a cooperator, too.
- *An exchanger and a cooperator both finish* (04) at every weight, later than told (the
  last goal by 2 transitions at $w$ = 0.8 and 1, by 16 and 17 at 2 and 30).

## What hindering rests on (told)

Each robot told its partner's social goal: the defaults until 2026-10-08,
`Config(**TOLD)`. Each line was checked by switching that one thing back (six-object
scene, 40 transitions, agent_1 hinders). Lines 1 to 5 were the state of the earlier
implementation, in which the hinderer acted erratically. "vs competition" in this table is
the competitor of Table I as printed (`competition="table"`), the partner that answers in
kind.

| # | now | switched back | what happens then |
|---|---|---|---|
| 1 | Nobody is made to let go. Before, a robot handed its own object had to take it and give up what it held, so one "use" of the partner undid the theft | `forced_swap=True` | vs competition, $w$ = 1 and 30: the object is held 8 of 17 transitions, then lost; both finish (16 / 15) |
| 2 | The term is $-w\,r_j$. Before it was $-w\,(r_j - c_j)$: with the partner's action cost inside, the hinderer earns $+w\,c$ (about +15 at $w = 30$) for every step its partner takes, so it baited (put back, pick up) instead of keeping | not an option | |
| 3 | The hinderer reasons at level 1 | hinderer at level 2 | vs competition: never takes the object, at $w$ = 1 and 30, and hesitates near its start. In its plan against level-1 models of the partner the step toward the object is worth less than the step away (social part of Q at the first transition, $w = 1$: -44.8 against -37.2) |
| 4 | Greedy actions | `action_selection="sample"` | vs competition, $w = 1$, seed 0: one accidental "use" accepts the swap; both finish (21 / 20; at $w = 30$: 21 / 19). Seeds 1 and 2 hold |
| 5 | Spare objects cannot be picked up | `handle_spares=True` | vs competition, $w = 1$: the competitor picks up a spare object at transition 3 (with full hands it cannot take its own); the hinderer puts the hostage back at transition 20, and the two block each other |
| 6 | The goal reward does not follow a carried object | `held_reward="carrier"` | $w = 30$: never takes the object; vs competition both finish (31 / 28) |

`idle_hypothesis=True`, `social_cost=True` and `partner_social_goal="inferred"` each leave
hindering intact in the same runs. Against the competitor of either race form, lines 1 and 5
change nothing: the object is taken at transition 2 and kept, with forced swaps and with
spare objects in play, at $w$ = 1 and 30; line 6 stands ($w = 30$: the object is never
taken). Sampled actions (line 4, seeds 0 to 2) do not free the hostage either; under them the
competitor of the default picks the hinderer's object up in turn (transition 12, 7 and 12)
and each ends up holding the other's, which the competitor of `"race"` does not do (see
"Limits"). Against a level-2 partner without social goal line 5 stands as well: with spare
objects in play at $w = 1$ that partner picks up a spare at transition 3, and the hinderer
puts the hostage back at 20 and does its own task (done at 30).

Line 3 holds told, against Table I's competitor. Nobody told -- each robot inferring its
partner's goal -- a hinderer has reasoned at level 2 by default since 2026-10-11; in seven of
the eight runs made with it its partner never finished, against five of the same eight with
it at level 1 ("Every robot with a social goal at level 2"). `TOLD` keeps it at level 1.

## Results (told)

Each robot told its partner's social goal: the defaults until 2026-10-08,
`Config(**TOLD)`. The same with every goal inferred: "Results, nobody told". Six-object
scene, the default levels (2, and 1 for a competitor and for a hinderer), greedy actions,
at most 60 transitions, `competition="race_then_hinder"` (`run_pairings.py --told
--competition race_then_hinder`). Without social goals the robots finish at transitions
9 / 11. A robot without social goal paired with a social one was at level 2 here, and no
robot allowed for an idle partner: the defaults until 2026-10-08 (`physical_level=2`,
`idle_hypothesis=False`). Entries are agent_1 / agent_2; "–" = never.

| pairing | levels | $w = 0.8$ | $w = 1$ | $w = 2$ | $w = 30$ |
|---|---|---|---|---|---|
| 01 competition / competition | 1 / 1 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| 02 cooperation / cooperation | 2 / 2 | 22 / 5 | 22 / 5 | 23 / 5 | 5 / 5 |
| 03 competition / cooperation | 1 / 2 | 5 / – | 10 / – | 10 / – | 5 / – |
| 04 exchange / cooperation | 2 / 2 | 5 / 17 | 5 / 17 | 12 / 13 | 5 / 17 |
| 05 conflict / cooperation | 2 / 2 | 7 / – | 5 / – | 5 / – | 5 / – |
| 06 hinder / cooperation | 1 / 2 | 7 / – | 5 / – | 5 / – | 5 / – |
| 07 hinder / competition | 1 / 1 | – / – | – / – | – / – | – / – |
| 08 hinder / none | 1 / 2 | – / – | – / – | – / – | – / – |

The rows with a competitor under the other four settings:

| pairing | `competition=` | levels | $w = 0.8$ | $w = 1$ | $w = 2$ | $w = 30$ |
|---|---|---|---|---|---|---|
| 01 competition / competition | `"race"` | 1 / 1 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| | `"table"` | 2 / 2 | – / – | – / – | – / – | – / – |
| | `"table_hostile"` | 2 / 2 | – / – | – / – | – / – | – / – |
| | `"rival"` | 2 / 2 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| 03 competition / cooperation | `"race"` | 1 / 2 | 19 / 31 | 5 / 17 | 5 / 17 | 5 / 17 |
| | `"table"` | 2 / 2 | 7 / 5 | 5 / 5 | 5 / 5 | 5 / 5 |
| | `"table_hostile"` | 2 / 2 | 7 / – | 5 / – | 5 / – | 5 / – |
| | `"rival"` | 2 / 2 | 7 / – | 5 / – | 5 / – | 5 / – |
| 07 hinder / competition | `"race"` | 1 / 1 | – / – | – / – | – / – | – / – |
| | `"table"` | 1 / 2 | – / – | – / – | – / – | – / – |
| | `"table_hostile"` | 1 / 2 | – / – | – / – | – / – | – / – |
| | `"rival"` | 1 / 2 | – / – | – / – | – / – | – / – |

**Competition / competition.** Each robot crosses to its own object, takes it (transitions 7
and 9) and sets it down on its own destination (9 and 11), at every weight, also at
$w = 100$. The race term is 0 for both until agent_1 finishes. From transition 9 agent_1's
term is its goal and its lead, $w\,(2\rho - r(\tilde g_j))$: for the actions it takes in
transitions 9 to 11 it expects 1.52, 1.43 and 1.24 times $w\rho$, less as agent_2 closes
in (what it is paid in the end is 4.83 $w\rho$ over transitions 9 to 12). It does not stay
on its destination: it sets off for the other room, where agent_2 is, two moves toward the
doorway. agent_2 expects to lose 11 % of $w\rho$ in transition 9 (it does not know
agent_1's destination), loses all of it in transition 10, and is two cells from its
destination: it finishes at 11, before agent_1 can reach it. Under `"race"` agent_1 stays
on its destination from transition 9, paid $w\rho$ per transition.

Up to the first finish this is the course of two level-1 robots without social goals, but
for the order of two moves of agent_2 (left and down, transitions 5 and 6, an equally long
way): each robot has its own object and its own destination, so the robot behind cannot
catch up and the one ahead has nothing to gain before it is done. Variations of the map, at
$w$ = 1, 30 and 100:

- *A doorway one cell wide.* Two level-0 robots block each other for good; two competitors
  pass, 9 / 12, action for action as two level-1 robots without social goals do until
  agent_1 is done. agent_1 then goes after agent_2 (three moves) and does not catch it.
- *agent_2's destination in the far room* (alone: 9 / 17). agent_1, done at 9, walks back
  to the doorway and shuttles between its two cells; agent_2, carrying its object, is
  turned back there every third transition (16 times in 60 transitions, $w$ = 30 and 100)
  and never gets through: 9 / –. At $w = 1$ agent_2 slips through once and agent_1 keeps
  between it and its destination in the far room instead (blocked 10 times): 9 / – as well.
  agent_1 keeps at it to the step limit (it moves in 44 of the 60 transitions at $w = 30$,
  33 at $w = 1$). Under `"race"`: 9 / 17; agent_2 loses $w\rho$ in each transition from 10
  to 16 and still does not touch agent_1's object. Under `"table"` ($w = 30$) neither
  finishes: the two pace, 51 and 52 moves in 52 transitions; `"table_hostile"` is the same
  rollout, as for any two competitors.
  Two rivals: 9 / 17 as well ("A rival" below).
  (`examples/compare_competition.py` draws this map, two competitors on the standard map
  and competition / cooperation under the five settings.)
- *agent_1's destination farther from its object* (alone: 13 / 11): 13 / 11. agent_2, done
  first, takes two steps toward agent_1, which finishes at 13.
- *Each robot starts next to its own object*: 3 / 3.
- *A doorway one cell wide with walled-in approaches*: a stand-off, as between two robots
  without social goals. Nobody touches an object.

**Competition / cooperation.** The cooperator picks the competitor's object up at transition
2 and carries it to the competitor's destination (set down at 5; at $w$ = 1 and 2 it tries
one wrong destination first, and it is 10). The competitor walks two cells toward its
object, turns round once it is carried for it, and goes back to its start, next to the
place of the cooperator's object. It does not touch that object while its own goal is
open. One transition after its goal is done ($w$ = 30, 1, 2: transitions 6, 11, 11; two
after at $w = 0.8$: 7) it picks the cooperator's object up, and holds it to the end of
the rollout: one "use" in all, no hand-over, no set-down. The cooperator never finishes.

| `competition=` | what the competitor does ($w = 30$) | done |
|---|---|---|
| `"race_then_hinder"` | walks two cells toward its object and back while it is carried for it (delivered at 5), takes the cooperator's object (6), keeps it | 5 / – |
| `"race"` | walks two cells, waits, never uses its hands; the cooperator delivers both objects | 5 / 17 |
| `"table"` | carries the cooperator's object to the cooperator's destination while the cooperator carries the competitor's: each delivers for the other | 5 / 5 |
| `"table_hostile"` | steps onto the place of the cooperator's object (1), picks it up (3) and keeps it; the cooperator delivers the competitor's object all the same (5) | 5 / – |

Under `"race"` a level-1 robot without social goal in the competitor's place acts as the
competitor does, at $w$ = 0.8, 1 and 30; what differs is the cooperator, which at $w$ = 1 and 2
delivers at transition 5 for a competitor and later for a robot without social goal.

With the seats swapped (cooperation / competition) the competitor takes nothing. At
$w \ge 1$ the cooperator sets the competitor's object down at transition 5 and then stays
where it is; the competitor walks back to its start, next to the cooperator's object, and
waits there without using its hands: – / 5 (see "Limits"). At $w = 0.8$ the cooperator
hands the competitor its object after one failed set-down, and both finish (27 / 24). Under
`"race"`: 17 / 5 at $w$ = 2 and 30; at $w$ = 0.8 and 1 the cooperator tries four wrong
destinations and hands the object over at 23, and both are done at 30.

**Table I with the cell turned** (`competition="table_hostile"`; the default levels,
(2, 2)). Entries are for $w$ = 0.8, 1, 2, 30.

- *competition / cooperation*: 7 / –, 5 / –, 5 / –, 5 / –; as printed 7 / 5, 5 / 5, 5 / 5,
  5 / 5. The cooperator delivers the competitor's object as before. The competitor does
  not go for its own object and does not deliver the cooperator's: it makes one move, onto
  the place of the cooperator's object, picks that object up (transition 3, 6, 6, 3) and
  holds it to the end. One "use" in all, no hand-over, no set-down, none attempted. The
  cooperator never finishes. At $w$ = 1 and 2 the competitor waits until its own object is
  delivered and takes the cooperator's one transition later; at $w$ = 0.8 and 30 it takes
  it before. The transition rests on little: one transition earlier, taking the object is
  0.005, 0.10, 0.014 and 0.002 below waiting in its Q; in the transition it takes it,
  waiting is 0.17, 0.012, 0.028 and 0.31 below.
- *cooperation / competition*: – / –, – / 5, – / 5, – / 5; as printed 5 / 7, – / 5, – / 5,
  5 / 5. At $w \ge 1$ the cooperator delivers at 5 and never finishes its own task. At
  $w$ = 1 and 2 the competitor stands on the place of the cooperator's object from
  transition 2; the cooperator sets out for it at transition 21 (23), stops two cells
  short, and the competitor then picks the object up (transition 29 and 32, by 0.004 and
  0.003 in Q). At $w = 30$ neither moves after the delivery, and the competitor takes
  nothing (staying where it is is 0.002 to 0.013 above the step onto the object's place
  in its Q). At $w = 0.8$ the competitor takes the object at transition 2, puts it back at
  6 and takes it again at 8; the cooperator fails two set-downs and puts the
  competitor's object back at 19; neither goal is done.
- *competition / exchange*, the other half of the turned cell: – / – at $w$ = 1 and 30; as
  printed 46 / 53 and 17 / 5. The competitor picks the exchanger's object up at transition
  2 and keeps it, and with its hands full does not do its own task.
- *Everything else is the table.* Over each of the seven social goals of a partner, in
  either seat, at $w$ = 1 and 30 (26 distinct pairings: competition / competition is the
  same in either seat), the rollout is other than the one under `"table"` in the eight
  with a cooperative or an exchanging partner and the same, action for action, in the
  other 18. Two such competitors never finish, at any of the four weights: at $w = 1$ one
  takes the other's object three times and puts it back twice, at $w = 30$ the two pace
  (50 moves taken back each in 52 transitions). In hinder / competition the competitor
  takes the hinderer's object at transition 4 and offers a swap from transition 10 on, as
  under `"table"`.
- *Its own task.* In those 26 pairings the competitor's goal is done in four: the ones in
  which a cooperator delivers its object. It never delivers that object itself, and in
  none of them does it set its partner's object down for its partner or hand it over
  (against a coercing partner at $w = 30$ it tries once, at transition 14, on a wrong
  destination: the coercion cell is the table's).
- *At level 1 nothing is turned*, and nothing needs to be: the table's entry there is
  $-r(\tilde g_j)$ against every partner. A level-1 competitor of either reading of the
  table takes a cooperator's object at transition 2 and keeps it (5 / – at $w$ = 1 and 30,
  at levels (1, 2) and (1, 1)). The cell that was turned is what a competitor is paid
  once it reasons about its partner's social goal. With only the competitor at level 2,
  levels (2, 1), it is 5 / – at $w = 30$; at $w = 1$ the competitor steps onto the place
  of the cooperator's object and never uses its hands, the level-1 cooperator picks the
  competitor's object up and puts it back for the whole rollout (51 uses in 52
  transitions), and neither goal is done.
- *With an inferred partner goal* (`partner_social_goal="inferred"`, the paper's mode: the
  competitor is not told that its partner cooperates, it holds a belief over five
  hypotheses). competition / cooperation at $w = 1$: 5 / –, as printed 5 / 6. The
  competitor's belief that its partner cooperates is 1.00 at the end, and it takes the
  cooperator's object at transition 2 (by 0.15 in Q) and keeps it. At $w = 30$ neither
  goal is done by transition 60 (as printed 17 / –): the two take each other's object and
  put it back (3 and 4 put-back cycles), and the competitor's belief in a cooperative
  partner ends at 0.16. cooperation / competition: – / 5 at $w = 1$ (as printed 19 / 5)
  and – / 42 at $w = 30$ (as printed 27 / 12). In all four the actions are those of a
  conflict robot in the competitor's place.

With known partner goals the rollout is the one of a conflict robot in the competitor's
place (conflict / cooperation, cooperation / conflict, conflict / exchange, conflict /
conflict and so on), action for action, in 24 of the 26 pairings: all but competition /
coercion at the two weights.

**A rival** (`competition="rival"`; the default levels, (2, 2)). Entries are for $w$ = 0.8,
1, 2, 30 unless noted.

- *competition / competition*: 9 / 11 at every weight, also at $w = 100$, where the
  table's two competitors never finish. Each robot crosses to its own object (taken at
  transitions 7 and 9) and sets it down on its own destination. Neither uses its hands
  for anything else, neither is ever in the other's way, and once it is done a rival
  stays on its destination. It is the rollout of two competitors of `"race"` asked for
  level 2, action for action and Q value for Q value. The same at levels (1, 1), (1, 2)
  and (2, 1) and with unequal weights, (30, 1) and (1, 30) ($w$ = 1 and 30).
- *Against a cooperator* it is the competitor of `"table_hostile"`: competition /
  cooperation 7 / –, 5 / –, 5 / –, 5 / –, and with the seats swapped – / –, – / 5, – / 5,
  – / 5, the rollouts described above, to the last Q value (also with unequal weights,
  7 / – and 5 / –, and at levels (1, 1), (1, 2) and (2, 1)). A cooperator with weight 0
  has no social goal: the rival races it, 9 / 11.
- *Against an exchanging partner* it races: 9 / 11 at every weight in either seat, and it
  never touches the exchanger's object, where `"table_hostile"` takes it at transition 2
  and neither goal is done.
- *Every partner goal, either seat, $w$ = 1 and 30* (26 distinct pairings): the rollout
  is the one of `"table_hostile"` in the four with a cooperative partner and the one of
  `"race"` at the same levels in the other 22, every action, every Q value and every
  entry of the ledgers. Against a partner without social goal, a coercing and an
  exchanging one: 9 / 11 in all twelve runs; the rival never touches its partner's
  object and is never in its way. In none of the 26 does a rival set its partner's
  object down for it or hand it over.
- *Against a partner that is out to stop it, it does not simply race.* A hinderer takes
  its object at transition 2, and the rival takes the hinderer's in turn (transition 3
  to 9) and offers a swap for the rest of the rollout, none accepted: – / – in all four
  runs (seated first at $w = 30$ it makes the offers standing on its own destination,
  44 of them recorded as failed set-downs). That is a way to its own goal, which its
  term pays for: a level-2 robot without social goal in its place never uses its hands.
  Against a conflict robot: competition / conflict – / – at both weights, and no object
  is touched (at $w = 1$ the two push into each other, 25 blocked moves each; at
  $w = 30$ the rival does not move at all, and the conflict robot walks over and pushes
  into it); conflict / competition 9 / 11 at $w = 1$ and – / – at $w = 30$, where the
  rival picks the conflict robot's object up at transition 7 and keeps it. (With a
  level-2 robot without social goal in the rival's place the conflict robot takes that
  robot's object at transition 2, in either seat and at both weights: what a conflict
  robot does depends on whom it faces, and these runs do not say whether the rival's
  hostage is its term's doing or its level's.) All of these are the rollouts of a
  competitor of `"race"` at level 2 ("A competitor at level 2").
- *The maps* (two rivals, $w$ = 1 and 30; the same at both weights):

  | map | levels (2, 2), the default | (1, 1) | (1, 2) | (2, 1) |
  |---|---|---|---|---|
  | the standard map | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
  | agent_2's destination in the far room | 9 / 17 | 9 / 17 | 9 / 17 | 9 / 17 |
  | a doorway one cell wide | – / – | 9 / 12 | 11 / 12 | 9 / 12 |
  | a doorway one cell wide with walled-in approaches | – / – | – / – | 14 / 11 | 9 / 14 |
  | agent_1's destination farther from its object | 13 / 11 | 13 / 11 | 13 / 11 | 13 / 11 |
  | each robot starts next to its own object | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |

  In the far room the rival that is done does not come back for the one that is not
  (9 / 17 at every mix of levels, where `"race_then_hinder"` gives 9 / – at level 1 and
  – / – at level 2, 12 / 28 there at $w = 1$). Past the doorway one cell wide two
  level-2 rivals stand off, as two level-2 competitors of either race form do: each plans
  against a level-1 model of the other that would give way ("A competitor at level 2",
  "Limits"). One of them at level 1 is enough.
- *With an inferred partner goal* (`partner_social_goal="inferred"`: a uniform prior over
  five social goals) a rival is not told whom it faces, and a fifth of its belief is on
  a cooperative partner, against which it is hostile. **Two rivals then do not race.**
  It is the exception that does it: two competitors of `"race"` asked for level 2, which
  is `"rival"` without it, finish at 9 / 11 at all four weights, as two level-2 robots
  without social goals do.

  | pairing, inferred goals | `"rival"` | `"race"` at the same levels |
  |---|---|---|
  | competition / competition | – / –, 20 / 29, – / –, – / – | 9 / 11 at every weight |
  | competition / no social goal | 9 / 11, 11 / 11, 11 / 11, 10 / 11 | 9 / 11 at every weight |
  | competition / exchange | 9 / 11, 11 / 11, – / 30, 5 / – | 9 / 11, 9 / 11, 7 / 21, 5 / 17 |
  | competition / conflict | – / –, 20 / 29, 35 / 30, – / – | – / – at every weight |
  | competition / cooperation | – / –, 5 / –, 31 / –, 5 / – | not run |
  | cooperation / competition | – / 7, – / 5, – / 5, – / 5 | not run |

  Two rivals: at $w$ = 0.8 and 30 they push into each other head-on (25 blocked moves
  each; no object is touched); at $w$ = 1 and 2 they take each other's object at
  transition 2 and put it back (five and six put-back cycles between them). Against a
  cooperator the rival ends sure that its partner cooperates in seven of the eight runs,
  and the cooperator delivers the rival's object in those seven (at 5 to 7, once at 31
  after four failed set-downs); in the eighth, competition / cooperation at $w = 0.8$,
  the two push into each other. Against an exchanging partner at $w = 30$ the exchanger
  delivers the rival's object (5), the rival takes it for a cooperator (1.00), picks its
  object up at 13 and keeps it: 5 / –, where `"race"` gives 5 / 17. See "Limits".
  (Two pairs of entries are alike at $w = 1$: 20 / 29 against a competing and against a
  conflict partner, 11 / 11 against a partner without social goal and against an
  exchanging one. Each pair is one sequence of actions chosen by different models: the
  rival is not told whom it faces, so it has one model and, its partner acting the same,
  the same Q in both runs; its partner's Q differs between the two by up to 343 and 34.)

Seven competitors against a cooperator, and against their like ($w$ = 0.8, 1, 2, 30):

| competitor | competition / cooperation | cooperation / competition | competition / competition |
|---|---|---|---|
| `"race_then_hinder"`, level 1 (the default) | 5 / –, 10 / –, 10 / –, 5 / – | 27 / 24, – / 5, – / 5, – / 5 | 9 / 11 at every weight |
| `"race_then_hinder"`, level 2 (`levels=(2, 2)`) | 5 / – at every weight | – / –, – / 5, – / 5, – / 5 | 9 / 11 at every weight |
| `"race"`, level 1 (its default) | 19 / 31, 5 / 17, 5 / 17, 5 / 17 | 30 / 30, 30 / 30, 17 / 5, 17 / 5 | 9 / 11 at every weight |
| `"race"`, level 2 (`levels=(2, 2)`) | 5 / 17 at every weight | – / –, – / –, – / –, 17 / 5 | 9 / 11 at every weight |
| `"table"`, level 2 | 7 / 5, 5 / 5, 5 / 5, 5 / 5 | 5 / 7, – / 5, – / 5, 5 / 5 | – / – at every weight |
| `"table_hostile"`, level 2 | 7 / –, 5 / –, 5 / –, 5 / – | – / –, – / 5, – / 5, – / 5 | – / – at every weight |
| `"rival"`, level 2 | 7 / –, 5 / –, 5 / –, 5 / – | – / –, – / 5, – / 5, – / 5 | 9 / 11 at every weight |

`"race"` is the one of the seven under which a competitor works against nobody: `"rival"`
is `"race"` at level 2 with the row of `"table_hostile"` against a cooperator.
`"table_hostile"`, `"rival"` and `"race_then_hinder"` at level 2 all keep a level-2
competitor against a cooperator. They differ in what two competitors do. Under the table
neither does its task. Under `"race_then_hinder"` each does, and the one that is done
first then goes after the other: nothing comes of it on the standard map, and on the
far-room map neither finishes at level 2 ($w = 30$; 12 / 28 at $w = 1$). Two rivals each
do their task and leave each other alone, on both maps.

**Hinder.** In all twelve hinder runs the hinderer picks its partner's object up at transition
2 and holds it to the end: 0 releases, 0 put-back cycles, 0 reversed moves, 0 spare pick-ups,
and the partner gets none of its baseline return. These rollouts stop "by deadlock" (25
transitions without progress), which is how a successful hindering run ends.

- **vs cooperation**: the cooperator still carries the hinderer's object to its destination
  (transition 5 to 7), then has nothing left to do.
- **vs competition**: the competitor's own goal is never done, and it does not take the
  hinderer's object: it walks two cells toward its object and waits there (33 of 35
  transitions), action for action as a level-1 robot without social goal does in its
  place, under both race forms. (`competition="table"`: it takes the hinderer's object at
  transition 4 and from transition 10 offers a swap every transition, never accepted. A
  competitor of a race form asked for level 2 does the same: "A competitor at level 2".)
- **vs none**: the hinderer carries the object across the map and waits on or next to the
  place of its own.

Hindering has a threshold in the weight (the same weight for both robots): at $w \le 0.3$ the
robot does its own task (9 / 11), from $w = 0.35$ it takes the object. `hinder` / `hinder` is a
stand-off in which each holds the other's object.

## Every pairing at $w = 30$ (told)

Each robot told its partner's social goal: the defaults until 2026-10-08,
`Config(**TOLD)`. The same with every goal inferred: "Results, nobody told".
`examples/run_all_pairings.py --told` rolls out all 49 pairings of the seven goals (none,
cooperation, exchange, coercion, competition, conflict, `hinder`; either seat) on the
six-object scene: default levels, greedy actions, at most 60 transitions, competitors as
`"rival"`. Rows are agent_1's goal, columns agent_2's; an entry is the transition at which
agent_1 / agent_2 finished its own goal (two level-0 robots without social goals: 9 / 11).

| agent_1 \ agent_2 | none | cooperation | exchange | coercion | competition | conflict | hinder |
|---|---|---|---|---|---|---|---|
| **none** | 9 / 11 | 5 / 17 | 9 / 11 | 9 / 11 | 9 / 11 | – / – | – / – |
| **cooperation** | 17 / 5 | 5 / 5 | 17 / 5 | 5 / 5 | – / 5 | – / 5 | – / 5 |
| **exchange** | 9 / 11 | 5 / 17 | 5 / 5 | 5 / 19 | 9 / 11 | – / – | – / – |
| **coercion** | 9 / 11 | 5 / 5 | – / – | – / – | 9 / 11 | – / – | – / – |
| **competition** | 9 / 11 | 5 / – | 9 / 11 | 9 / 11 | 9 / 11 | – / – | – / – |
| **conflict** | – / – | 5 / – | – / – | – / – | – / – | – / – | – / – |
| **hinder** | – / – | 5 / – | – / – | – / – | – / – | – / – | – / – |

Both goals are done in 21 pairings, exactly one in 6, neither in 22.

- *A cooperator's partner is served at transition 5*, in all 13 pairings with a cooperator.
  The cooperator's own goal is done against a partner without social goal and an exchanging
  one (17), against a cooperator and a coercing one (5), and never against a competitor, a
  conflict robot or a hinderer.
- *With a conflict robot or a hinderer in the pair nobody finishes*, unless the other is a
  cooperator: 20 of 24 pairings. In the other four the cooperator delivers the hostile
  robot's object and never gets to its own.
- *None, exchange, coercion and competition among themselves* (16 pairings): 9 / 11 in 12
  of them; 5 / 5 for two exchangers, each of which delivers the other's object; 5 / 19 for
  exchange / coercion; neither goal for coercion / exchange and coercion / coercion.
- *Seats.* Swapping the seats mirrors who finishes in every pair but that one, coercion
  and exchange.
- *Two rollouts go to and fro to the end.* conflict / conflict: 50 moves of each robot
  take back the one before, the two-cycle of two competitors of the table ("Why two
  competitors of the table ping-pong"). coercion / coercion: each robot picks the other's
  object up 26 times in 52 transitions and puts it back.
- *Close calls.* In 15 of the 49 rollouts at least one decision is closer than 0.04 in Q
  (ties between equally good actions left out), which is the order of the solver's error
  at this weight ("Limits", large weights). Two kinds of rollout rest on such margins
  throughout: cooperation / competition and cooperation / conflict (30 of 60 decisions,
  the closest by 0.002), and coercion / exchange (49 of 108).

**The reading of competition** changes the 13 pairings with a competitor in them and no
other (`--competition`, with `--only-competition` for these 13):

| pairing | `"race_then_hinder"` | `"race"` | `"table"` | `"table_hostile"` | `"rival"` |
|---|---|---|---|---|---|
| none / competition | 9 / 11 | 9 / 11 | – / – | – / – | 9 / 11 |
| cooperation / competition | – / 5 | 17 / 5 | 5 / 5 | – / 5 | – / 5 |
| exchange / competition | 9 / 11 | 9 / 11 | 5 / 17 | – / – | 9 / 11 |
| coercion / competition | 9 / 11 | 9 / 11 | – / – | – / – | 9 / 11 |
| competition / none | 9 / 11 | 9 / 11 | – / – | – / – | 9 / 11 |
| competition / cooperation | 5 / – | 5 / 17 | 5 / 5 | 5 / – | 5 / – |
| competition / exchange | 9 / 11 | 9 / 11 | 17 / 5 | – / – | 9 / 11 |
| competition / coercion | 9 / 11 | 9 / 11 | – / – | – / – | 9 / 11 |
| competition / competition | 9 / 11 | 9 / 11 | – / – | – / – | 9 / 11 |
| competition / conflict | 9 / 11 | 9 / 11 | – / – | – / – | – / – |
| competition / hinder | – / – | – / – | – / – | – / – | – / – |
| conflict / competition | 9 / 11 | 9 / 11 | – / – | – / – | – / – |
| hinder / competition | – / – | – / – | – / – | – / – | – / – |
| levels of a competitor | 1 | 1 | 2 | 2 | 2 |

**Higher weights.** The same 49 at $w$ = 100, 300 and 1000 finish as at 30 in 47, 44 and
47 pairings. Two changes are the model's, with margins of 110 and more in Q from $w = 100$
on: two cooperators no longer deliver for each other (each walks toward its own object and
the two push into each other, – / –), and coercion / cooperation ends 5 / 7 instead of
5 / 5. The other three rest on margins of about 0.01 in Q where values are $10^4$ to
$10^7$: cooperation / competition and cooperation / conflict end – / 34 at $w = 300$ (and
– / 5 again at 1000), conflict / competition 9 / 18 at 300. Above 30 the solver's values
are not good enough to tell such cases ("Limits").

## An exchange scene

`exchange_scene()` is a scene in which the two objects have to change hands, and
`examples/run_exchange_scene.py` rolls out exchange / exchange on it (`--goals` for
another pairing, `--controls` for the controls below). It is the six-object scene with the
doorway narrowed to one cell, (4, 2), and each robot's destination in its own room:
agent_1 (left room) brings O1 to (1, 5), agent_2 (right room) brings O2 to (7, 1). Each
robot still starts next to its partner's object; its own lies in the other room. What
leaves an exchange in the doorway as the only way to finish:

- **Rooms.** Each robot stays in its own room; the doorway belongs to both
  (`Scene.rooms`). A robot cannot fetch its own object, and its goal reward is 0 until the
  object lies where it can go. The rooms are what makes the exchange necessary: with the
  doorway open to both, each robot walks through and fetches its own object (below).
- **A hand-off cell**, the doorway (`Scene.handoff_cells`). A robot standing in it with
  its partner's object in its hands puts it down there; the partner picks it up from
  there. Without it two exchangers do not finish (– / –).
- **The rule the script runs the scene under**,
  `Config(set_down_for_partner=False, hand_over=False)`: nobody sets its partner's object
  down on a destination, and nothing goes from hand to hand. Neither is needed for the
  exchange. They keep the rollout to the exchange: with set-downs for the partner allowed
  each robot first tries candidate destinations in its own room, and `--hand-over` makes
  the exchange quicker.

Neither robot can finish alone, and two robots without social goals do not touch an
object, at levels 0, 1 and 2. (At levels 0 and 1 they do not move. At level 2, unsure
what the other is, each takes two to four steps and waits.)

**Nobody is told its partner's social goal**, as everywhere since 2026-10-08 ("What a robot
knows of its partner"): each level-2 robot starts from a uniform prior over the social goals
it holds possible and over the six candidate destinations, and updates both from what its
partner does (Eqs. 5-6). Its social term is Table I's column $l > 1$ (a rival's: its own
reading, "The model") averaged over that
belief. A robot is still given its partner's weight and reasoning level; what it infers is
the goal. What the same scene does when each robot is told its partner's goal is in "If
the robots are told" below.

The rule for the hands is the script's (`RULES`); the rest is `Config`'s defaults. **Most
of this section was rolled out on 2026-10-07 with the paper's five goals held possible**
(`social_hypotheses=PAPER_GOALS`), before none and hinder were added: its accounts of what
the robots do and believe are of those runs, and say "five goals". The table
"exchange / exchange, the seven goals of the defaults" below has the exchange at nine
weights with the seven goals as well; the 49 pairings with seven are "Every pairing on the
exchange scene" in "Results, nobody told".

Below unless said: levels (2, 2), greedy actions, at most 60 transitions, and as the
competitor among the hypotheses the rival of the scripts (`--competition rival`;
other readings at the end of the next subsection). A cell is the transition at which
agent_1 / agent_2 finished its own goal; "–": it had not when the rollout ended (after 25
transitions without progress, or at the 60th).

### What an exchanger is at level 1 decides whether one can be recognised

A level-2 robot models its partner one level down, and it judges which social goal its
partner has by which of its level-1 models explains what the partner does (Eqs. 5, 7). So
everything hangs on what an exchanger does at level 1, where the table and the text of
the paper differ (`Config.exchange_level1`):

| `exchange_level1=` | a level-1 exchanger's reward | where it is from |
|---|---|---|
| `"none"` (the default until 2026-10-08) | $r(g_i) - c(a_i)$ | Table I: N/A, no social term |
| `"copy"` | $r(g_i) + w_i\,n\,r(\tilde g_j) - c(a_i)$ | the paper's text, "a copy of the other agent's goal with a positive, but small, scalar n", taken as the social term and so multiplied by the weight |
| `"copy_unweighted"` (the default) | $r(g_i) + n\,r(\tilde g_j) - c(a_i)$ | the same copy, not multiplied by the weight |

$n = 0.25$ (`exchange_share`). At $w = 1$, the paper's weight, the two copies are one
model; which of them is meant at another weight the paper cannot say, because the weight
is this package's. The level-2 row is the table's under all three: $w_i\,[\,n\,r(\tilde
g_j) + \tilde R_j\,]$ against a partner taken to exchange, nothing against any other.
Under either copy a level-1 exchanger is a level-1 cooperator with the weight $w\,n$ or
$n$ in place of $w$.

| exchange / exchange, the seven goals of the defaults | $w = 0.2$ | 0.5 | 0.8 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| `"copy_unweighted"` (the default), put down and pick up | – / – | – / – | 19 / 16 | 19 / 16 | 19 / 18 | 19 / 16 | 18 / 15 | **17 / 16** | 17 / 14 |
| `"none"`, put down and pick up | – / – | – / – | 18 / – | 18 / – | – / 18 | 22 / 19 | 18 / – | 14 / – | 22 / 19 |
| `"copy"`, put down and pick up | – / – | 20 / 17 | 19 / 16 | 19 / 16 | 19 / 16 | 18 / 15 | 23 / 20 | – / – | – / – |
| `"copy_unweighted"`, hand-overs allowed | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 17 / 16 | 17 / 14 | 15 / 14 | 15 / 14 | 14 / 13 |
| `"none"`, hand-overs allowed | – / – | 46 / – | 38 / – | 36 / – | 17 / 16 | 18 / 17 | 18 / 17 | 18 / 17 | – / 13 |

With the seven goals the copy at its own size exchanges at every weight tried from 0.55 to
100 (eleven weights: 0.55, 0.6, 0.65, 0.7 and the seven of the table from 0.8) and at none
from 0.2 to 0.5 (six weights); with the paper's five, from 0.45 (below). Before anything
is seen a robot holds an exchanger possible at 1/7 instead of 1/5; that is not all of the
shift: the weight times that prior where the exchange starts is 0.079 here, and 0.09 to
0.1 with the five and with the smaller sets tried (below). With hand-overs allowed it
exchanges at all nine weights. Under Table I as printed both goals are done at two of the
nine weights (5 and 100), and at five others one robot gives and gets nothing back; the
copy times the weight exchanges from 0.5 to 10, as with five. At 30 / 30 agent_2 gets the
doorway first: it puts O1 down at 7, agent_1 puts O2 down at 10 and takes O1 at 11,
agent_2 takes O2 at 12 (17 / 16). At the end agent_1 holds its partner for a cooperator
(0.97, exchange 0.03), agent_2 its partner for an exchanger (0.71, cooperation 0.29): with
five goals it was the other way round.

| exchange / exchange, the paper's five goals (2026-10-07) | $w = 0.2$ | 0.5 | 0.8 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| `"none"`, put down and pick up | – / – | 19 / – | 18 / – | 15 / – | – / 14 | – / 18 | – / 18 | – / – | 22 / 19 |
| `"copy"`, put down and pick up | – / – | 19 / 16 | 19 / 16 | 19 / 16 | 18 / 15 | 17 / 14 | 21 / 18 | – / – | – / – |
| `"copy_unweighted"`, put down and pick up | – / – | 19 / 16 | 19 / 16 | 19 / 16 | 18 / 15 | 18 / 15 | 18 / 15 | **18 / 15** | 17 / 14 |
| `"none"`, hand-overs allowed | – / – | 42 / – | 38 / – | 36 / – | – / 14 | 17 / 16 | 17 / 16 | 18 / 17 | – / 13 |
| `"copy"`, hand-overs allowed | – / – | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 27 / 26 | 27 / 24 | 27 / 24 |
| `"copy_unweighted"`, hand-overs allowed | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 | 14 / 13 |

**Under Table I as printed (`"none"`) whoever is helped first stops helping** (five
goals). At level 1 an exchanger does nothing for its partner, so cooperation is the only
one of the five hypotheses under which a partner fetches the robot's object. A robot's own
term is the table's entry times its belief that the partner exchanges, so it is switched
off by the very thing it is meant to return.

- *From 0.5 to 10 one robot gives and gets nothing back.* At 2 / 2 both take the other's
  object (4) and agent_1 steps toward the doorway (5). agent_2 sees it and holds
  "cooperation" at 0.95; its own term is all but gone, and it puts O1 back where it lay
  (6). agent_1 has seen no help (0.25 on each of the other four goals, "exchange" among
  them), carries on and puts O2 down in the doorway (9). agent_2 takes it (10) and
  finishes (14); agent_1 gets nothing. At 5 and 10 it is the same, with the put-back and
  the put-down at 10. At 0.5, 0.8 and 1 agent_1 never takes agent_2's object; agent_2
  gives it O1 and gets nothing back.
- *At 30 / 30 neither gives.* Each takes the other's object (3) and, with "exchange" at
  0.036, at weight 30 enough to act on, steps toward the doorway. Each sees the other do
  so: cooperation 0.998, exchange 0.001, and the social part of its Q falls from 64.6 to
  2.6 at most. It steps back and waits for the cooperator to come. The other's step back
  changes no belief (0.998); its wait, which a cooperator would not make (the model has it
  go on with probability 0.998), brings cooperation down to 0.858 and exchange back to
  0.036. The two repeat these three transitions (a step toward the doorway, a step back, a
  wait) for all 200 transitions rolled out, and nothing is put down.
- *At 100 / 100 both goals are done* (22 / 19), and not because the two recognise each
  other. agent_1 puts O2 down at 11 with "exchange" at 0.23. agent_2 puts O1 down at 14
  while it holds agent_1 for a cooperator at 1.000 (exchange below 0.0005): O2 lies in
  the doorway, and putting O1 down frees its hands to take it.
- *With hand-overs allowed* both goals are done at 5, 10 and 30, and one at every other
  weight but 0.2.

The same was found on 2026-10-04 on a house scene with the engine in `social_mdp/`:
whoever was helped first stopped helping.

**The copy times the weight (`"copy"`) makes the partner a helper to wait for at large
weights** (five goals). A partner that helps can now be an exchanger, and the two exchange
from 0.5 to 10. At 30 and 100 the model each robot has of an exchanging partner carries
the term $w\,n$ = 7.5 and 25. At 30 / 30 agent_2 expects agent_1 to set off for its object
at once with probability 0.41; its own social value is then the same whatever it does
(5395.56 for each of its four actions), and staying is cheapest. Each of the two waits in
one of the first two transitions, and that settles it: an eager helper would not have
waited, so each drops cooperation and exchange (0.000 from then on) and with them its own
term (agent_1's social value falls from 5384 to 8.7). Nobody touches an object in the 200
transitions rolled out. Under the rival the size of the term $w\,n$ separates the runs:
with $n$ = 0.1, 0.25, 0.5, 1 and 2 at five to nine weights each (29 runs), the two
exchange in all 17 runs with $w\,n$ between 0.125 and 4, in none of the 8 with
$w\,n \ge 7.5$, in one of the two at $w\,n = 5$, and in neither of the two with
$w\,n \le 0.1$ ($w = 0.2$ with $n = 0.25$, and $n = 0.1$ at $w = 1$). With Table I's
competitor or `"race_then_hinder"` among the hypotheses the weighted copy stops already at
$w = 10$ ($w\,n = 2.5$). The two parts of the level-2 term,

$$w_i\,\xi_{ij} = w_i\,\big[(1 + n)\,r(\tilde g_j) + w_j\,n\,\hat r_i\big]$$

($\hat r_i$: the robot's own goal reward as its partner estimates it), are of one size
at $w = (1 + n)/n$, but that ratio does not say where the exchange stops: with $n = 0.1$
it still comes about at $w = 30$. With hand-overs allowed the weighted copy finishes at
every weight but 0.2, later from 10 on.

**The copy at its own size (`"copy_unweighted"`) lets the exchange come about at every
weight tried from 0.45 to 100** (five goals, 23 weights; with seven, from 0.55). The
level-1 exchanger then has the term $n$ whatever the weight, and the level-2 term is

$$w_i\,\xi_{ij} = w_i\,\big[(1 + n)\,r(\tilde g_j) + n\,\hat r_i\big].$$

A partner that helps may be an exchanger, so being helped does not undo the attempt, and
the model of an exchanging partner does not become the eager helper of the weighted copy.
Weights above 30 are beyond those the comparisons with brute force cover ("How it is
checked"). With hand-overs allowed the two exchange at every weight from 0.2: agent_1 puts
O2 down in the doorway at 7 and agent_2, standing next to it, hands it O1 at 8.

- *The lower end follows the prior, and more.* Before anything is seen, a robot has a
  fifth of the term of a robot that is sure. Under the rival, with fewer goals held
  possible the exchange starts at a smaller weight: with "exchange" alone from $w = 0.1$
  (not at 0.05); with cooperation and exchange from 0.2 (not at 0.15); with those and
  conflict from 0.3 (not at 0.2); with the five from 0.45 (not at 0.4). The weight times
  the prior is 0.09 to 0.1 each time. But the lower end also moves with the competitor
  among the hypotheses (0.65 with Table I's) and with $n$ (with $n = 0.1$ not at 1 or 2,
  but at 5).
- *Sure of each other, the two hold out in a window of weights.* Told, or with "exchange"
  the only goal held possible, they exchange from 0.1 to 0.3 and from 0.7 on. At 0.4, 0.5
  and 0.6 each carries the other's object to the cell next to the doorway and waits
  there for the other to go first (– / –). Under Table I as printed, told robots exchange
  at these weights. With the five goals held possible the window does not show at any of
  the 23 weights.
- *The size of the copy.* $n$ = 0.25, 0.5, 1 and 2 (the value in the authors' code) give
  the exchange at $w$ = 1, 2, 5 and 30; $n = 0.1$ gives it at 5 and 30 only. Where $n$
  equals the partner's weight the two hypotheses that help are one model, and the
  robots end at 0.50 / 0.50 on "exchange" ($n = 1$ at $w = 1$, $n = 2$ at $w = 2$).

**Other readings of competition among the five hypotheses** (five goals; `--competition`). What a
level-1 competitor is changes one of the five models a partner is judged by:

| `"copy_unweighted"`, put down and pick up | $w = 0.2$ | 0.5 | 0.8 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| `"rival"`, `"race"` | – / – | 19 / 16 | 19 / 16 | 19 / 16 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 17 / 14 |
| `"table"` (Table I's competitor), `"table_hostile"`, `"race_then_hinder"` (the default until 2026-10-08) | – / – | – / – | 20 / 17 | 20 / 17 | 19 / 18 | 18 / 15 | 18 / 15 | 18 / 15 | 17 / 14 |

With Table I's competitor the two exchange at every weight tried from 0.65 to 100 (18
weights), and at 5, 10, 30 and 100 the rollout is the same event for event. The other two rows of
the first table move more. `"none"` has both goals done at 2, 10 and 100 under
`"race_then_hinder"` and at 5 and 100 under `"table"`, and at 30 under none of the
three readings tried.
`"copy"` exchanges from 0.5 to 5 under both and from 10 on under neither.

**The exchange at 30 / 30** (five goals; with seven, above). Each robot picks its partner's
object up at transition 2 and carries it to the doorway. agent_1 gets the doorway, puts
O2 down at 7 and steps back. agent_2 steps in, puts O1 down at 10 and takes its own O2
with the next "use" (11). agent_1 steps in and takes O1 at 12. Each carries its own
object to its own destination (18 and 15). The first step is ahead of waiting one more
transition by 0.07 (agent_1) and 0.13 (agent_2) in Q, against values near 600; the
pick-up by 6.3 and 5.6. Over the eight weights from 0.5 to 100 the smallest margin of a
first step is 0.0009 (agent_1 at $w = 2$), the size of the largest difference between
the engine's Q and the brute-force reference (7.4e-4, "How it is checked").
The margin does not decide the outcome: with the first step of a robot made a wait,
both goals are done all the same in the 13 runs tried (eight weights), three
transitions later at most.

**What each comes to believe** (five goals; the figure's panel "... and about its social goal").
Both start at 0.2 on each of the five goals. While the partner fetches (transitions 1 to
5), the model of a level-1 cooperator gives each of its moves a likelihood of 0.96 and
that of a level-1 exchanger 0.57 to 0.60, and nothing else explains them: cooperation
rises to 0.93, exchange falls to 0.07. Exchange comes back where the partner does what
an eager helper would not. agent_2 lets agent_1 into the doorway first (transition 6:
likelihood 0.01 under cooperation, 0.36 under exchange), and it takes its own object as
soon as it has put agent_1's down (transition 11: 0.02 against 0.54). agent_1 ends at
0.99 on exchange. agent_2 ends at 0.19 on exchange and 0.81 on cooperation for agent_1,
which gave it less of that to see. So the inference is partial: both read their partner
as one of the two kinds that help (1.00 from the third transition on), neither belief
in "exchange" falls below 0.026, and at weight 30 that is enough to act on.

**Other pairings with inferred goals** (five goals; with seven: "Every pairing on the
exchange scene" in "Results, nobody told"), put down and pick up, $w = 30$ (each robot at its
default level of then, which was 1 for a hinderer and 2 for everybody else here); agent_1's goal
by row, agent_2's by column:

| | none | cooperation | exchange | coercion | competition | conflict | hinder |
|---|---|---|---|---|---|---|---|
| **none** | – / – | 14 / – | 16 / – | – / – | – / – | – / – | – / – |
| **cooperation** | – / 14 | 14 / 27 | 25 / 24 | 29 / 14 | – / – | – / – | – / 16 |
| **exchange** | – / 16 | 31 / 16 | **18 / 15** | 27 / 24 | – / 17 | – / – | – / – |
| **coercion** | – / – | 14 / 27 | 21 / 20 | – / – | – / – | – / – | – / – |
| **competition** | – / – | – / – | – / – | – / – | – / – | – / – | – / – |
| **conflict** | – / – | – / – | – / – | – / – | – / – | – / – | – / – |
| **hinder** | – / – | 16 / – | – / – | – / – | – / – | – / – | – / – |

Both goals are done in 8 pairings, exactly one in 7, neither in 34
(`run_all_pairings.py --scene exchange`). The option changes more than the exchange
goal: "exchange" is one of the five models every level-2 robot judges its partner by.

- *The eight are the pairings among cooperation, exchange and coercion*, all nine but
  coercion / coercion: each of the three has a term toward a partner that may be one of
  the kinds that help. Two exchangers are the first to have both goals done (18; the
  next is coercion / exchange at 21).
- *The scene does not single out the exchange goal.* Two cooperators finish as well, one
  after the other (14 / 27). Told each other's goal they wait for each other for good
  (below); unsure of it each gives, under Table I as printed too (14 / 27 at $w = 30$,
  16 / 15 at 1).
- *What is recognised is that a partner helps, not which of the helping kinds it is.* In
  the eight pairings all 16 final beliefs sit on cooperation or exchange, and 9 put the
  partner's true goal first: an exchanger is read as one in 5 of 6, a cooperator in 4 of
  6, a coercing robot in none of 4 (each is read as an exchanger, at 1.00). The copy
  trades one confusion for another. Under Table I as printed two cooperators end sure
  of each other (1.000 / 1.000), and an exchanger is never singled out: at level 1 it
  acts like a robot without social goal. Under the copy a helper may be either, and of
  two cooperators one is read as an exchanger (0.97).
- *The second put-down also serves the robot that makes it.* When agent_2 puts O1 down
  at 10, O2 has been lying in the doorway since 7, and the put-down is also the quickest
  way to free its hands for it; the rollouts cannot tell the two motives apart. Under
  Table I as printed at 100 / 100 a robot makes such a put-down while it holds its
  partner for a cooperator at 1.000 (above).
- *At weight 30 the belief hardly gates the attempt.* Next to a robot without social
  goal an exchanger puts the object down at 10 and gets nothing back (– / 16), although
  it gives "exchange" 0.026 by then and conflict, competition and coercion 0.32 each.
  Next to a rival it does so at 12 with "exchange" at 0.003 (– / 17). At 1 / 1 it does
  not give to a robot without social goal: as agent_1 it takes the object and keeps
  it, as agent_2 it does not touch it. Next to a conflict robot or a hinderer it never
  puts the object down.
- *Two rivals do nothing.* Told that they face a rival, each takes the other's object
  and trades it through the doorway (18 / 15, below); not told, neither touches an
  object.
- *With hand-overs allowed* (`--hand-over`) the counts are 10, 5 and 34, and the ten
  are not the eight and two more: coercion / exchange drops out, and coercion / coercion
  (48 / 45), two rivals (54 / 53) and conflict / competition (55 / 50) come in, late.

Thirteen of the cells of the table were rolled out for 200 transitions without the
stopping rule and stay as they are: none / none, cooperation and exchange each next to a
robot without social goal (four cells), exchange next to conflict, a hinderer and a
rival (six), competition / competition and coercion / exchange. At 1 / 1 a cooperator
still gives to a robot without social goal (– / 18, 16 / –), and an exchanger and a
rival exchange after a long wait in one seating (exchange / competition 37 / 34;
competition / exchange: no object is touched).

**What the result rests on** (five goals). exchange / exchange, goals inferred, `"copy_unweighted"`,
put down and pick up, at $w$ = 1 and 30:

| changed | $w = 1$ | $w = 30$ |
|---|---|---|
| nothing | 19 / 16 | 18 / 15 |
| the partner's action cost in $\tilde R_j$ (`social_cost=True`, Eq. 2 as printed) | – / –: no object is touched in 200 transitions | – / – in 60 transitions; 75 / 72 in 200 |
| levels (2, 1) | – / –: agent_2 puts O1 down at 7, agent_1 keeps O2 | 18 / 15 |
| levels (1, 2) | – / –: agent_1 puts O2 down at 7, agent_2 keeps O1 | – / –: the same |
| levels (1, 1) | – / –: the same | – / –: the same |
| $n$ = 0.1, 0.5, 1 (`exchange_share`) | – / –, 20 / 17, 26 / 23 | 19 / 16, 17 / 14, 17 / 14 |
| "none" among the hypotheses (six instead of five) | 19 / 16 | 18 / 15 |
| "none" and "hinder" (seven, the defaults since 2026-10-08) | 19 / 16 | 17 / 16 |
| the product of the two marginals in Eq. 4 (`social_belief="marginals"`) | 19 / 16 | 18 / 15 |
| no lapse in the likelihood (`belief_lapse=0`); the idle hypothesis | 19 / 16; 19 / 16 | 18 / 15; 18 / 15 |
| `hands_full_reward="none"` | – / –: no object is touched | 17 / 14 |
| sampled actions, seeds 1, 2, 3 | 20 / 19, 18 / 18, 18 / 18 | 17 / 19, 18 / 15, 18 / 15 |
| the first step of agent_1, of agent_2, of both made a wait | 21 / 18 (agent_2's; agent_1 waits first anyway) | 18 / 15, 17 / 16, 17 / 16 |
| a robot whose goal is done keeps its hands still (`done_robots_handle=False`) | 19 / 16 | 18 / 15 |
| set-downs for the partner allowed (`set_down_for_partner=True`) | 29 / 26 | 26 / 23: each tries its partner's object on three candidates in its own room first |
| no hand-off cell (`exchange_scene(handoff=False)`) | – / – | – / – |
| the doorway open to both (`exchange_scene(rooms=False)`) | 17 / 18 | 17 / 18: nothing changes hands, each robot walks through and fetches its own object |
| unequal weights 30 / 1, 1 / 30, 30 / 2, 2 / 30, 5 / 0.5, 0.5 / 5 | 18 / 15, 17 / 16, 18 / 15, 17 / 16, 19 / 16, 25 / 38 | |
| one robot with weight 0 (30 / 0, 0 / 30), at level 2 (the default until 2026-10-08; level 0 since) | – / 12, 14 / –: the other gives and gets nothing back (a partner with weight 0 is one model under all five hypotheses, so "exchange" stays at its prior) | |

The scene's own ingredients, the rooms and the hand-off cell, are necessary. Of the
options of the model, at 30 / 30 the exchange is lost with agent_1 at level 1 and takes
75 / 72 transitions with the partner's action cost; with the rest of what was tried
both goals are done in 17 to 19 transitions (26 with set-downs for the partner
allowed). At 1 / 1 it is lost with either robot at level 1, with the
partner's action cost, with `hands_full_reward="none"` and with $n = 0.1$. The value at
100 is beyond the weights the comparisons with brute force cover (30 / 30, "How it is
checked").

### If the robots are told

`--told` is the scene as it was first built: the defaults of before 2026-10-08, each
level-2 robot told its partner's social goal, Table I as printed.

| exchange / exchange, told | $w = 0.2$ | 0.5 | 0.8 | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|---|---|
| put down and pick up (`hand_over=False`) | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 17 | 23 / 19 |
| hand-overs allowed (`hand_over=True`) | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 18 / 15 | 23 / 19 |

**The exchange when told**, put down and pick up, $w = 30$. Each robot picks its
partner's object up at transition 2 and carries it to the doorway. Both step toward the
doorway at 6; agent_1 gets it (where two robots move to one cell the collision rule lets
the first through). It puts O2 down at 7, steps back and aside. agent_2 steps in and
puts O1 down at 10. Each then picks its own object up from the doorway, agent_1 at 12
and agent_2 at 13, carries it to its own destination and sets it down there (18 and 17).
In all 18 runs of the table each object is put down in the doorway once, at transitions
7 and 10, and nothing is handed over.

- *$w \le 10$*: agent_2 takes its own object in the visit to the doorway in which it puts
  O1 down (by its hands, at 11) and is done at 15.
- *$w = 30$*: agent_2 steps out of the doorway first and comes back for O2, which costs
  it two transitions. By its model of agent_1 that clears the way: the model has agent_1
  step into the empty doorway with probability 0.95, and with 0.49 while agent_2 stands in
  it holding O2 (stepping out is ahead of picking up by 0.16 in Q). In the rollout it gains
  agent_1 nothing: with agent_2 made to pick O2 up first, agent_1 takes O1 at 12 all the
  same and agent_2 is done at 15.
- *$w = 100$*, beyond the weights the values are good for ("Limits"): five and seven
  transitions pass between the second put-down and the pick-ups (15 and 17).
- *Unequal weights*: 30 / 1 and 30 / 2 end 18 / 15, 1 / 30 and 2 / 30 end 18 / 17: the
  detour belongs to agent_2's weight. With weight 0 on either side (30 / 0, 0 / 30) nobody
  touches an object: a robot with weight 0 has no social goal.
- *Hand-overs allowed*: a robot standing in the doorway puts its partner's object down
  whatever `hand_over` says, because nothing is handed over from a hand-off cell. A robot
  next to the doorway may hand over to a partner standing in it with free hands; agent_2
  tries that once in every run, at transition 8, as agent_1 steps out of the doorway, and
  nothing comes of it.

**Why two exchangers give.** A level-2 robot acts on what it expects of its partner,
which it models one level down. Table I has no exchange term at level 1, so an
exchanger's model of its partner is a robot without social goal: at the start it expects
its partner to stay (probability 0.97), and fetching the partner's object is its own best
action. A cooperator's model of its partner is a level-1 cooperator, which fetches. At
$w = 30$ each of two cooperators expects its partner to set off for its object with
probability 1.000; its own social term is then the same whatever it does itself (the
social part of Q is 108,864.509 for each of agent_1's four legal actions), staying is the
cheapest action, and neither moves. Two exchangers that are given a level-1 term
(`exchange_level1="copy"`) wait in the same way at $w = 30$ (– / –) and exchange at
$w = 1$ (18 / 15).

**Other pairings when told**, $w = 30$ ($w$ = 1 and 2 are the same unless said):

| pairing | put down and pick up | hand-overs allowed |
|---|---|---|
| none / none, levels (0, 0), (1, 1) and (2, 2) | – / –: nobody moves | the same |
| exchange / none, none / exchange | – / –: nobody moves | the same |
| cooperation / none | – / 12: the cooperator puts O2 down in the doorway at 7; nobody brings it O1 | – / 10: it hands O2 over at 6 |
| none / cooperation | 14 / – | 12 / – |
| exchange / cooperation | 14 / –: the cooperator gives, the exchanger takes and gives nothing back | 12 / – |
| cooperation / exchange | – / 12 | – / 10 |
| cooperation / cooperation | – / –: nobody moves ($w$ = 1 and 2: each takes its partner's object at 2 and keeps it) | – / –: nothing is picked up ($w = 2$: agent_1 takes O2 and keeps it; $w = 1$: 14 / 13) |
| competition / competition (`"rival"`) | 18 / 15: the exchange of the small weights, event for event ($w \le 10$: – / –, nobody moves) | 14 / 13 |
| cooperation / coercion | 28 / 12: one after the other ($w$ = 1 and 2: 17 / 14, at the same time) | 23 / 10 |

Rolled out for 200 transitions without the stopping rule (put down and pick up, $w$ = 1
and 30; of none / none only levels (2, 2)), the first seven rows stay as they are.

**All 49 pairings when told** at $w = 30$ (`run_all_pairings.py --scene exchange
--told`), put down and pick up; agent_1's goal by
row, agent_2's by column:

| | none | cooperation | exchange | coercion | competition | conflict | hinder |
|---|---|---|---|---|---|---|---|
| **none** | – / – | 14 / – | – / – | – / – | – / – | – / – | – / – |
| **cooperation** | – / 12 | – / – | – / 12 | 28 / 12 | – / – | – / – | – / 16 |
| **exchange** | – / – | 14 / – | **18 / 17** | 16 / – | – / – | – / – | – / – |
| **coercion** | – / – | 14 / – | – / 16 | – / – | – / – | – / – | – / – |
| **competition** | – / – | – / – | – / – | – / – | 18 / 15 | – / – | – / – |
| **conflict** | – / – | – / – | – / – | – / – | – / – | – / – | – / – |
| **hinder** | – / – | 16 / – | – / – | – / – | – / – | – / – | – / – |

Both goals are done in 3 pairings, exactly one in 9, neither in 37. In each of the nine a
cooperator or a coercing robot (whose term holds $+\tilde R_j$ toward a cooperative or an
exchanging partner) puts its partner's object down in the doorway and gets nothing back.
With hand-overs allowed both goals are done in 8 pairings and exactly one in 8; five of
the eight have a competitor in them.

**Told, the scene does not single out the exchange goal either.** At $w = 30$ two rivals
do what two exchangers do at the small weights: each takes its partner's object, puts it
down in the doorway and takes its own from there (18 / 15). Each holds what the other needs, and a
rival that puts its partner's object down in the doorway has its hands free for its own.
They do so at $w$ = 30 and 100, and at levels (2, 2) under `"race"` at 30; not at
$w \le 10$ (nobody moves), not at levels (1, 1), and not under `"race_then_hinder"`,
`"table"` or `"table_hostile"` ($w$ = 1 and 30). A cooperator and a coercing robot finish
both goals as well. What sets exchange / exchange apart on this scene is the range of
weights (every weight from 0.2 to 100, where two rivals need 30 and two cooperators do
not finish at 1, 2 or 30) and the mixed pairings: an exchanger gives to an exchanger only
(exchange / none, exchange / cooperation, exchange / coercion, exchange / competition:
the exchanger gives nothing).

**Is it an exchange, or cooperation?** In the reward it is cooperation with a factor.
Table I pays a level-2 exchanger $\tilde R_j + n\,r(\tilde g_j)$ against an exchanging
partner, and its partner one level down has no social term, so $\tilde R_j = r(\tilde
g_j)$ (with `social_cost=False`, the default) and the term is $(1 + n)\,r(\tilde g_j)$,
the cooperation entry of level 1 times 1.25. What makes the rollout an exchange is the
world, in which all a robot can do for its partner is put its object within its reach,
and the two conditions of the table: no exchange term toward a partner that is not an
exchanger, and none at level 1.

**What the result rests on when told.** Put down and pick up, exchange / exchange, at
$w$ = 1 and 30:

| changed | $w = 1$ | $w = 30$ |
|---|---|---|
| nothing | 18 / 15 | 18 / 17 |
| the partner's action cost in $\tilde R_j$ (`social_cost=True`, Eq. 2 as printed) | – / – | – / –: agent_1 puts O2 down at 7, agent_2 does not follow |
| levels (2, 1) | – / 16 | – / 16: the level-2 robot gives, the level-1 robot has no term |
| levels (1, 2) | 16 / – | 16 / – |
| levels (1, 1) | – / – | – / –: nobody moves |
| a level-1 term for exchange (`exchange_level1="copy"`), levels (2, 2) | 18 / 15 | – / –: nobody moves |
| the same at levels (1, 1) | – / –: agent_1 puts O2 down at 7, agent_2 does not follow | 18 / 15 |
| `hands_full_reward="none"` | 19 / 16 | 18 / 17 |
| sampled actions, seeds 1, 2, 3 | 18 / 16, 19 / 18, 23 / 16 | 18 / 19, 24 / 19, 18 / 16 |
| a robot whose goal is done keeps its hands still (`done_robots_handle=False`) | 18 / 15 | 18 / 17 |
| set-downs for the partner allowed (`set_down_for_partner=True`), with hand-overs or without | 27 / 24 | 27 / 24: each tries its partner's object on three candidates in its own room (5, 9, 13) before it puts it down in the doorway (16, 19) |
| `forced_swap=True`, hand-overs allowed | 13 / 13 | 13 / 13: agent_2 hands O1 over in the transition in which agent_1 puts O2 down (7) |
| no hand-off cell (`exchange_scene(handoff=False)`), with hand-overs or without | – / – | – / – |
| the doorway open to both (`exchange_scene(rooms=False)`) | 17 / 18: nothing changes hands, each robot walks through and fetches its own object, as two level-1 robots without social goals do ($w$ = 2, 5, 10 the same) | 25 / 25: agent_1 puts O2 down in the doorway and fetches O1 itself |

With the copy at its own size as what an exchanger is at level 1
(`--told --exchange-level1 copy_unweighted`) two told exchangers finish at 18 / 15
at $w$ = 0.1 to 0.3 and 0.7 to 2, and at 17 / 14 at 5, 30 and 100; at 0.4, 0.5 and 0.6
they hold out next to the doorway (– / –, "Sure of each other" above). With
hand-overs allowed they finish at 14 / 13 at 0.2, 1, 5, 30 and 100.

### The doorway open to both, and a corridor

`--scene rooms` and `--scene corridor` are two earlier attempts at a scene for an
exchange, under the default rule for the hands (set-downs for the partner and hand-overs
allowed, no hand-off cell, no rooms of their own). Their results were made told
(`--told`).

**`rooms`**: the map of the exchange scene with the doorway open to both robots. The map
does not force a hand-over: a robot can walk through the doorway, fetch its own object and
walk back. Two level-1 robots without social goals do that (17 / 18; agent_2 waits one
transition for agent_1 to come through). Two level-0 robots, and two level-2 ones, meet in
the doorway and push into each other for good (– / –).

| exchange / exchange, `rooms` | $w = 0.8$ | 1 | 2 | 5 | 10 | 30 | 100 |
|---|---|---|---|---|---|---|---|
| the default rules | 44 / 51 | 40 / 47 | 40 / 47 | 41 / 48 | – / – | 18 / 21 | 18 / 21 |

At $w = 30$ each robot picks its partner's object up at transition 2, tries it on wrong
candidate destinations (agent_1 four times, agent_2 twice: the partner's destination is
hidden), carries it through the doorway and sets it down on its partner's destination, at
18 and 21. The two do each other's task; nothing is handed over: cooperation in effect,
which is what the exchange scene was built to rule out. At $w \le 5$ agent_2 does the
carrying, hands O1 over once (transition 37 at $w = 1$) and fetches its own object
afterwards; at $w = 10$ the 60 transitions run out.

**`corridor`**: seven cells in a row, where the robots can never pass each other and each
robot's object lies in the dead end behind its partner. Only a hand-over can bring an
object to its owner, and without social goals nobody finishes.

| exchange / exchange, `corridor` | $w = 0.8$ | 1 | 2 | 5 | 10 | 30 |
|---|---|---|---|---|---|---|
| the default rules | | – / – | | | | – / – |
| `forced_swap=True` | 9 / 8 | 9 / 8 | 9 / 8 | 9 / 8 | 9 / 8 | – / – |
| `exchange_level1="copy"` | | 9 / 16 | | | | – / – |

Under the default rule two objects change hands only if both robots use their hands in
the same transition; the two pick each other's object up at transition 3, meet in the
middle and push into each other with it. With `forced_swap=True`, where one robot's "use"
swaps the two, they swap at transition 7 and each sets its own object down (8 and 9), at
weights up to 10 and not at 30. With a level-1 term for exchange (`"copy"`) the objects
are handed over one after the other at $w = 1$ (transitions 6 and 13). Two cooperators do
not finish there either (– / –, $w$ = 1 and 30).

## A competitor at level 2 (told)

Each robot told its partner's social goal: the defaults until 2026-10-08,
`Config(**TOLD)`. `levels=(2, 2)` (`--levels 2 2`) puts a competitor of either race form
at level 2. Its social term is the one of "The model", on goal rewards, with the finish
judged under each hypothesis about its partner's destination: nothing in it changes with
the level. What changes is the partner it plans against. At level 1 that partner pursues a
physical goal and nothing else. At level 2 it is the partner's level-1 model, which holds
the partner's own social term: $\tilde\psi_j^{1} = \mathrm{softmax}(Q_j^{1}/\tau)$.

So a level-2 competitor expects a cooperator to deliver for it, a hinderer not to let go,
and another competitor to turn on it once done, and it acts on that from the first
transition. "Its own goal first, then against its partner" describes its reward at
both levels, and its conduct at level 1 only.

Six-object scene, greedy actions, at most 60 transitions; agent_1 / agent_2. "Level 1" is
the default of "Results"; "level 2" is `levels=(2, 2)` (a hinderer stays at level 1).

| pairing | `competition=` | competitor at | $w = 0.8$ | $w = 1$ | $w = 2$ | $w = 30$ |
|---|---|---|---|---|---|---|
| competition / competition | default | level 1 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| | | level 2 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| | `"race"` | level 2 | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |
| competition / cooperation | default | level 1 | 5 / – | 10 / – | 10 / – | 5 / – |
| | | level 2 | 5 / – | 5 / – | 5 / – | 5 / – |
| | `"race"` | level 1 | 19 / 31 | 5 / 17 | 5 / 17 | 5 / 17 |
| | | level 2 | 5 / 17 | 5 / 17 | 5 / 17 | 5 / 17 |
| cooperation / competition | default | level 1 | 27 / 24 | – / 5 | – / 5 | – / 5 |
| | | level 2 | – / – | – / 5 | – / 5 | – / 5 |
| | `"race"` | level 1 | 30 / 30 | 30 / 30 | 17 / 5 | 17 / 5 |
| | | level 2 | – / – | – / – | – / – | 17 / 5 |
| hinder / competition | both | level 1 and 2 | – / – | – / – | – / – | – / – |

**Against a cooperator, seated first** (competition / cooperation). The cooperator brings
the competitor's object at transition 5 at every weight, without a wrong destination first.
The default form does not set out for its own object: it makes one move, onto the place of
the cooperator's object, and picks that object up, at $w \le 2$ at transition 2, three
transitions before its own goal is done, and at $w$ = 30 and 100 at transition 6, one
after. It keeps it, as at level 1. The early take is a clear preference (the step and the
"use" are 0.6 to 4.4 above the next action in Q). The wait at $w = 30$ is not: staying is
0.07, 0.007, 0.004 and 0.002 above the step in transitions 1 to 4, and with sampled actions
the competitor takes the object at transition 3, 4 and 2 (seeds 0 to 2). `"race"` in this
seat never uses its hands, at either level; at level 2 it walks up to three cells toward
its own object and waits (5 / 17 at every weight). Against a cooperator at level 1, levels
(2, 1), the default form does as against one at level 2: 5 / – at every weight, where two
level-1 robots stall at $w \le 2$ (see "Limits").

**Against a cooperator, seated second** (cooperation / competition). At $w \le 2$ the
competitor picks the cooperator's object up at transition 2, under both forms (the step and
the "use" are 1.1 to 5.6 above the next action in Q) and holds it to the end. With the
default form at $w$ = 1 and 2 the cooperator still delivers at 5 (– / 5); at $w = 0.8$,
and under `"race"` at every $w \le 2$, the cooperator never delivers and neither goal is
done. At $w = 30$ the competitor does not move: the cooperator delivers at 5, and under
`"race"` fetches its own object as well (17 / 5); the default form stays where it is by
0.01 in Q.

**Against a hinderer**, which takes its object at transition 2, a level-1 competitor walks
two cells and waits. At level 2 it picks the hinderer's object up in turn (transition 4; 3
at $w = 30$), walks to the hinderer and offers a swap in every transition from 10 on, none
accepted: 22 uses of its hands in 30 transitions (21 in 29 at $w = 30$). Both race forms do
the same, action for action: all of it is before the competitor's own goal is done, where
the two terms pay the same, and here the plans agree as well. With sampled actions
($w = 30$) one of three seeds ends in a swap, and both finish (21 / 20).

**Two competitors** finish at 9 / 11 on the standard map at every weight, also at
$w = 100$, as at level 1. On the variations of "Results" it matters which of the two is at
level 2 ($w$ = 1 and 30 unless noted):

| map | levels (1, 1) | (2, 2) | (1, 2) | (2, 1) |
|---|---|---|---|---|
| agent_2's destination in the far room | 9 / – | – / – ($w = 1$: 12 / 28) | – / – | 9 / – |
| a doorway one cell wide | 9 / 12 | – / – | 11 / 12 | 9 / 12 |
| a doorway one cell wide with walled-in approaches | – / – | – / – | 14 / 11 | 9 / 14 |
| agent_1's destination farther from its object | 13 / 11 | 13 / 11 | 13 / 11 | 13 / 11 |
| each robot starts next to its own object | 3 / 3 | 3 / 3 | 3 / 3 | 3 / 3 |

- *The far room.* At level 1 both race, and agent_1, done at 9, stands in agent_2's way for
  good. A level-2 agent_2 expects that, and picks agent_1's object up at transition 2. In
  four of the five runs (the table's, and levels (2, 2) at $w = 100$) it keeps it: agent_1
  is then never done and never turns on it, and agent_2, its hands full, does not finish
  either. In the fifth, levels (2, 2) at $w = 1$, it puts the object back at transition 8
  (by 0.12 in Q) and both finish, 12 / 28. Taking the object is not a close call: the step
  and the "use" are 66 above the next action in Q at $w = 30$, 3.0 and 3.7 at $w = 1$.
  Under `"race"` agent_1 will not turn on it, and it races: 9 / 17.
- *The doorway one cell wide.* agent_1 is through the doorway first, at every mix of
  levels, and meets agent_2 just past it. At levels (1, 1) agent_2 waits one transition and
  lets it by. At levels (2, 2) neither gives way: the two push into each other for the
  rest of the rollout (25 blocked moves each; also at $w = 100$, and under `"race"`). Each
  plans against a level-1 model of the other, which would give way, and at least one of
  them is a competitor: two level-2 robots without social goals pass there (9 / 12).
  `idle_hypothesis=True` does not resolve it. With only agent_2 at level 2 it is agent_1
  that gives way: it steps aside, and is done at 11 instead of 9. With only agent_1 at
  level 2 the course is that of levels (1, 1).
- *The doorway with walled-in approaches.* Two level-1 competitors stand off there, and so
  do two level-2 ones. One robot at level 2 is enough to resolve it: that robot goes
  through first (14 / 11 and 9 / 14).

**Against the other partners** (the default form; each of the seven social goals, in
either seat, $w$ = 1 and 30, 28 rollouts): against a partner without social goal, a
coercing and an exchanging partner, 9 / 11 in every run, as at level 1. With the competitor
at level 2, against a conflict partner neither goal is done in three of the four runs
(with the competitor at level 1 both finish). In none of the 28 rollouts does a level-2
competitor set its partner's object down on its partner's destination or hand it over.
(Seated first against a hinderer at $w = 30$ it makes its swap offers standing on its own
destination, where each unanswered offer is recorded as a failed set-down: 44 of them.)

**What is the term's doing, and what the level's.** The same runs with a level-2 robot
*without* a social goal in the competitor's place (and, in the trace, the Q of the
competitor's twin without its social term):

| what the level-2 competitor does | a level-2 robot without social goal in its place | so it is |
|---|---|---|
| seated first, takes a cooperator's object at transition 2 ($w \le 2$) | never uses its hands (5 / 17, 5 / 17, 12 / 13) | the term |
| seated second, takes a cooperator's object at transition 2 ($w \le 2$) | does the same, and neither goal is ever done | the level |
| behind in the far room, takes the object of the competitor ahead | does the same, at levels (1, 2) and (2, 2), $w$ = 1 and 30 | the level: it expects its partner to turn on it once done |
| does not give way past the doorway one cell wide | in agent_2's seat it gives way at $w = 1$ (9 / 12) and stands off at $w = 30$; in agent_1's seat it stands off at both weights; two level-2 robots without social goals pass (9 / 12) | both: it takes a competitor in the pair, and the robot facing it then stands off without a term of its own in three runs of four |
| takes a hinderer's object in turn and offers swaps | never uses its hands (pairing 08 of "Results", at $w$ = 1 and 30) | the term |

In the first row the twin prefers the step toward its own object by 0.1 to 0.3 where the
competitor prefers the step onto the place of the cooperator's object by 0.6 to 1.3; once
on that place the twin would pick the object up as well (0.6 to 0.9 above staying). In the
rows marked "the level" a robot with no social term at all does what the competitor does:
it takes a cooperator's object so as to be served, and it takes the object of a partner
that it expects to block it once done.

The reading of competition still decides two of the five rows, one of them a row marked
"the level", because the level acts through what the partner is expected to do. Under
`"race"` a level-2 competitor seated first leaves the cooperator's object alone (5 / 17),
and the robot behind in the far room races (9 / 17): the competitor ahead of it, a racer
too, will not turn on it. Seated second it still takes the cooperator's object at
$w \le 2$, and the stand-off past the doorway and the hostage against a hinderer stay
under either race form.

Seated second, what the robot plays on is a rule of the level-1 model: a level-1 robot whose
object is in its partner's hands takes it to be on its way to its destination
(`partner_delivers`), and turns to its partner's goal. With `partner_delivers=False`
($w$ = 1 and 2) the robot without social goal leaves the object alone and both finish
(15 / 12). The competitor does not take it at transition 2 either, in either seat, but three
of its four runs then end with neither goal done (seated second the cooperator takes the
competitor's object and never delivers it; seated first at $w = 1$ nothing is delivered);
in the fourth, seated first at $w = 2$, it takes the cooperator's object at transition 6,
after its own goal is done (5 / –).

**With $\tilde R_j$ in the term, the outcomes of 14 rollouts are the same.** From its column
$l = 1$ to $l > 1$ Table I replaces the partner's goal reward by the partner's reward.
With $\tilde R_j$ in the place of $r(\tilde g_j)$ in the race term (a prototype, not in this
code; `social_cost=False`, so without the partner's action cost) the same robots are done
in the same transitions in all 14 rollouts compared (seven pairings and maps, $w$ = 1 and
30; in 7 of them nobody is done under either term). In 9 every action is the same. In the
other 5 the conduct differs: against a conflict partner at $w = 1$ the competitor pushes
head-on instead of taking its object, and in the far room at $w = 30$ the robot with the
hostage pushes into its partner (a stand-off after 29 transitions) instead of walking to
and fro (55 moves in 60 transitions). Why the outcomes agree: once the robot's own goal
is done, its partner's estimate of the robot's goal reward is $\rho$ whatever the partner
believes, so $\tilde R_j$ of a cooperative, a hostile or an indifferent level-1 partner is
$r(\tilde g_j)$ plus a constant, and the two terms rank the actions alike. (A coercing
partner's $\tilde R_j$ scales $r(\tilde g_j)$ by $1 + w_j$, and a competing one's adds a
step of $2 w_j \rho$ at its own finish: the same direction, at another rate against the
robot's own costs.) Before the robot's own goal is done both terms are 0 for as long as
its partner's is not done either. Once the partner is done first they differ:
$-\tilde R_j$ of a level-1 cooperator is $-\rho - w_j\,\tilde r(g_i)$, the loss of the
term on goal rewards and, on top, $w_j$ times the cooperator's estimate of the robot's own
goal reward: a penalty on catching up. The term on goal rewards was kept.

**Which level.** Level 1 is the default because it is the competitor this project asked
for: it does its own task first, in every greedy run of "Results". Level 2 is the paper's
level, and the level at which a competitor plans against its partner's social goal; what
it then does before its own goal is done is no longer "its own task first". Part of that
goes with the term (under `"race"`: no early take seated first, no hostage in the far
room) and part does not (the take seated second, the stand-off past the doorway).
`levels=(1, 2)` and `(2, 1)` put one robot at each level; the robot at level 2 then models
its partner exactly, and the stand-off in the walled-in doorway is gone (at $w = 1$ with
only agent_2 at level 2 that rests on 0.0007 in agent_1's Q).
`examples/compare_levels.py` rolls four of the runs above out at both levels and draws
them.

## Why two competitors of the table ping-pong (told)

Each robot told its partner's social goal: the defaults until 2026-10-08,
`Config(**TOLD)`. Under either reading of the table two competitors at their default
levels, (2, 2), never finish ("Results"). At $w$ = 30 and 100 the rollout is a two-cycle:
from the third transition on each robot steps between the place of the other's object and
the cell beside it, one step out of phase with the other. In 52 transitions agent_1 makes
51 moves and agent_2 52, 50 of them each a move taken back, and neither ever uses its
hands.

This is not a slip in the solver: the two-cycle is what the table's models plan. (The
models of this pairing agree with brute force on a 3 x 3 map at weights (2, 2), one of the
slow tests of `test_models.py`; at $w = 30$ the values below satisfy the recursion of Eq. 3
when it is written out by hand at the third transition, for both robots and every legal
action.) What follows is the rollout at $w = 30$ on the six-object scene, greedy actions,
read off each robot's solved models: what it expects of its partner, what it would do, and
what that is worth to it. Ranges are over the 50 transitions of the cycle; "asked" means
the model was put in a situation the rollout never reaches.

1. *What a robot is paid.* Against a competing partner the table gives $-\tilde R_j$. The
   partner one level down is the table's competitor at $l = 1$, whose term is minus the
   robot's goal reward as that partner estimates it, $\hat r_j(g_i)$. So

   $$R_i = r(g_i) - w_i\,r(\tilde g_j) + w_i w_j\,\hat r_j(g_i) - c(a_i).$$

   The robot's own goal reward, as its partner sees it, counts $w^2 = 900$ times. A
   level-2 competitor of the table is paid for its partner's failure, and far more for
   being seen to get on with its own goal.
2. *What it expects.* Its model of its partner is that level-1 competitor, a robot paid
   $-r(\tilde g_j)$: a hinderer. Each robot starts next to the place of the other's
   object. In each robot's model the partner steps onto the place of the robot's object
   when it stands beside it and picks the object up when it stands on it: probability
   1.000 in every one of the 25 transitions in which the partner stands beside it, and in
   every one of the 25 in which it stands on it. The partner does step onto it, every
   time. It never picks the object up.
3. *What it plans.* With its object as good as taken, the way to its goal is to get it back
   in a swap, and both robots plan one, by their own best action and their partner's most
   likely one: let the partner take the object first, take the partner's in turn, and, next
   to the partner, offer a swap in every transition from then on. In its model the partner,
   once it has the robot's object, walks across toward the place of its own (probability
   0.74 to 0.94 in the 18 positions asked short of its arrival). agent_1 means to wait
   there for it and to pick its object up as it arrives; agent_2 means to pick agent_1's
   object up two transitions after its own is taken and to carry it across. Values are
   71,000 to 72,100, where $w^2\rho/(1-\gamma)$, a goal done from the first transition, is
   112,000.
4. *Why neither takes first.* In the rollout, picking the other's object up is 87 below its
   best action for agent_1 and 71 to 72 for agent_2, in each of the 25 transitions in which
   it stands on it. Asked what it would do had it taken first (it holds its partner's
   object, its partner holds nothing), each robot's best action is to put the object back,
   in all ten positions asked (by 46 to 309 in Q; asked beside the object's place, it steps
   back onto it, ten more). The reason is in its model of the partner: a partner whose own
   object lies at its place comes across for it; a partner whose object is held stays where
   it is once it holds the robot's (probability 0.82 to 1.00 in the twelve positions asked
   with the partner four cells or more away; 0.60 to 0.81 in the four nearer ones, and next
   to the robot it stays or steps away, 0.33 to 0.84 for staying). The object is the bait.
   With its partner standing on its object and about to take it, waiting is worth one
   transition more than taking first (359 and 363 in Q): taking first, the robot would put
   the object back in the next.
5. *Why it steps to and fro.* While it waits, the third term pays it for looking, to its
   partner, as if it were getting on. One step from the place of its partner's object is
   a cell where its own goal reward is positive, and the step there is worth 9.7 to 9.9
   (agent_1; 8.5 to 8.7 for agent_2) in the transition it is made in, where the same step
   without the social term is worth $-0.44$; that puts it 9.3 to 9.5 (8.1 to 8.3) above
   staying. From that cell, with its partner now on the place of its object, the step
   back is 25.8 (19.4) above staying, none of it in that transition's reward ($-0.50$
   against $-0.05$): in its model the partner, once it has the robot's object, sets off
   across with probability 0.93 (0.91) if the robot stands on the place of the partner's
   object, and with 0.86 if it stands beside it. So in every transition the robot on the
   other's object expects the other to step onto its own, which it does, and steps off
   for the bonus; the robot off it expects the one on it to strike, which it does not,
   and steps back on. The two change roles in every transition.

Each robot's model of the other is a level-1 robot that strikes at once. The other is a
level-2 robot that is waiting for the strike. Put one of the two at level 1 and the plan
is what happens: at levels (2, 1), $w = 30$, agent_2 takes agent_1's object at transition
2 and walks across; agent_1 waits, picks agent_2's object up at transition 8 as it arrives,
steps next to it, and offers a swap in every transition from 10 to the end, 24 of them.
None is accepted, and neither goal is done there either.

No mix of levels mends it, and neither does the weight:

| two competitors of the table | what happens | done |
|---|---|---|
| levels (1, 1), $w$ = 1 and 30 | each picks the other's object up at transition 2 and keeps it | – / – |
| levels (1, 2), $w$ = 1 and 30 | agent_1 takes agent_2's object at 2; agent_2 takes agent_1's at 4, carries it across and offers a swap 21 times | – / – |
| levels (2, 1), $w$ = 1 and 30 | agent_2 takes agent_1's object at 2 and walks across; agent_1 waits, takes agent_2's at 9 and 8, and offers a swap 23 and 24 times | – / – |
| levels (2, 2), $w = 0.8$ | agent_2 takes agent_1's object at 4 and keeps it; agent_1 walks three cells toward it and back, then five cells toward it, and waits there | – / – |
| levels (2, 2), $w = 1$ | agent_1 takes agent_2's object three times and puts it back twice; 15 of its 19 moves are taken back | – / – |
| levels (2, 2), $w = 2$ | neither robot makes a single move | – / – |
| levels (2, 2), $w$ = 30 and 100 | the two-cycle above | – / – |

(The account above is of $w = 30$. At $w = 100$ the rollout and the quantities of
points 2, 4 and 5 that are measured along it are the same in kind; the situations the
models were asked about were put to them at $w = 30$ only. At $w \le 1$ a robot does take
first, and at $w = 2$ nothing moves.)

The equation of point 1 leaves the partner's action cost out of $\tilde R_j$, the default
here (`social_cost=False`). With the cost in it, as Eq. 2 prints it (`social_cost=True`),
two competitors of the table do not finish either: at $w$ = 0.8 and 1 they step to and fro
from the first transition on (51 transitions, 50 moves of each robot taken back, no object
touched), at $w = 2$ neither moves, and at $w = 30$ the rollout is the two-cycle above,
action for action. Two rivals are not touched by the option (9 / 11 and the same actions
at $w$ = 1 and 30): against each other their term holds no $\tilde R_j$. (Against a
cooperator, where it does, the cooperator's goal is not done with the option on either:
43 / – and 5 / – at $w$ = 1 and 30, – / 5 at both with the seats swapped.)

**Which of the table's two entries is at fault? Each, at some weights.** The competition
row has an entry at $l = 1$, $-r(\tilde g_j)$, which makes every competitor a hinderer one
level down, and a cell at $l > 1$ against a competing partner, $-\tilde R_j$, which pays a
robot for its partner's failure and, through the recursion, $w^2$ times for looking
successful. Two prototypes replace one of the two by the race term ($\lambda = 0$) and
leave the other as the table prints it (not in this code: one function patched for the run;
levels (2, 2)):

| entry at $l = 1$ | cell at $l > 1$ | $w = 0.8$ | $w = 1$ | $w = 2$ | $w = 30$ |
|---|---|---|---|---|---|
| $-r(\tilde g_j)$, the table | $-\tilde R_j$, the table | – / – | – / – | – / – | – / – |
| the race term | $-\tilde R_j$, the table | – / – | – / – | – / – | 9 / 11 |
| $-r(\tilde g_j)$, the table | the race term | 9 / 11 | 9 / 11 | – / – | – / – |
| the race term | the race term (`"rival"`) | 9 / 11 | 9 / 11 | 9 / 11 | 9 / 11 |

Neither replacement alone gets both goals done at all four weights. With only the entry
at $l = 1$ replaced the two finish at $w = 30$; at $w \le 2$ they walk toward each other
and push head-on for the rest of the rollout (25 or 26 blocked moves each; no object is
touched). With only the cell at $l > 1$ replaced they finish at $w \le 1$; from $w = 2$
on neither sets out (no move at all at $w = 2$, one each at $w = 30$). With both replaced
they finish at all four. (These rows say what happens, not why: replacing the entry
at $l = 1$ also changes $\tilde R_j$ inside the cell at $l > 1$, so the second row has
neither the hinderer one level down nor the $w^2$ term of point 1.)

`"rival"` replaces both. One level down a rival's rival is a robot that goes for its own
goal; and once a rival's own goal is done, nothing in its term concerns its rival (until
then the one thing in it about its rival is what it loses while the rival is done
first). 9 / 11 at $w$ = 0.8, 1, 2, 30 and 100 at levels (2, 2), and at $w$ = 1 and 30 at
every mix of levels ("Results").

## Options that matter (`Config`)

| option | default | why |
|---|---|---|
| `competition` | `"rival"` | "The model" above: a race against every partner but a cooperative one, against which it is the competitor of `"table_hostile"`; not told whom it faces, it weighs the two by its belief. `"race_then_hinder"` (the default until 2026-10-08): its own goal, and from then on against its partner; `"race"`: a competitor that stops once its goal is done; `"table"`: Table I as printed; `"table_hostile"`: Table I with $-\tilde R_j$ against a cooperative partner as well |
| `conflict` | `"table"` | a level-2 conflict robot is paid $-w\,\tilde R_j$, Table I as printed, under which a cooperator's term about the robot comes back into its reward and it may undo its own task. `"task"` (an option since 2026-11-01): $-w\,r(\tilde g_j)$, Table I's entry at l = 1, against every partner, $R_i = r(g_i) - w\,r(\tilde g_j) - c(a_i)$ ("Task conflict") |
| `cooperate_with` | cooperation, competition, exchange (`COOPERATE_WITH`) | the partners against which a level-2 cooperator's cell of Table I, $+\tilde R_j$, holds; against any other it has no social term and minds its own goal, so that a hinderer's wish that it fail is not its own ("Whom a cooperator cooperates with"; since 2026-10-11). `"all"` (`EVERY_PARTNER`): the table as printed, the default until then (`TOLD`) |
| `action_selection` | `"argmax"` | Algorithm 1 is greedy; line 4 above |
| `forced_swap` | `False` | line 1 |
| `set_down_for_partner` | `True` | `False`: nobody finishes its partner's task for it; an object reaches its owner by a hand-over or over a hand-off cell ("An exchange scene") |
| `hand_over` | `True` | `False`: nothing goes from hand to hand; an object changes owner only by being put down and picked up |
| `swap_on_place` | `False` | `True`: a robot holding an object on an object's place swaps it, with "swap", for the object lying there, and objects may lie on each other's places ("The world"; since 2026-10-12). The model grows with it: 29 configurations of two objects instead of 20, 1,321 of four instead of 126 |
| `steal` | `False` | `True`: a robot with free hands next to its partner takes what the partner holds, with "take", unless the partner lets go of it in the same transition ("The world"; since 2026-10-12) |
| `held_reward` | `"none"` | line 6 |
| `hands_full_reward` | `"ignored"` | a robot's goal reward while *its own* hands hold an object that is not its own: as if its hands were free. `"none"`: no reward until its hands are free. See "Limits" |
| `handle_spares` | `False` | line 5; spare objects are then scenery: listed in every record, left out of the model (a rollout on the six-object scene takes 2 to 5 s instead of 20 s to 2 min). Only with `partner_object="known"`: where the partner's object is inferred every object may be the partner's, and every object is handled |
| `partner_object` | `"inferred"` | which object a robot's partner wants: inferred, a uniform prior over every object in the model, the robot's own included, with every candidate destination, as in the paper ("Which object the partner wants"; since 2026-10-09). `"known"`: the partner's own, the default until then (`TOLD`) |
| `max_objects` | `6` | objects in the model, the two own ones first, then those nearest somebody's task; the others are scenery. With the partner's object inferred the model grows fast with it: on the six-object scene 13 GiB with six, 0.9 GiB with three |
| `max_memory_gib` | `48` | what the solver may use; it never plans with more than the machine has (or its control group's limit), and a model that cannot fit stops at once with a `SolverFailure` saying what to change |
| `social_cost` | `False` | whether $\tilde R_j$ inside a level-2 term holds the partner's action cost (Eq. 2 as printed: yes) |
| `partner_social_goal` | `"inferred"` | a uniform prior over `social_hypotheses`, updated by Eq. 5, as in the paper ("What a robot knows of its partner"). `"known"`: each level-2 robot is told its partner's social goal, the default until 2026-10-08 (`TOLD`) |
| `social_hypotheses` | the paper's five, `"none"`, `"hinder"` (`HYPOTHESES`) | the social goals a robot holds possible for its partner, each with the same prior. `PAPER_GOALS`: the paper's five, the default until 2026-10-08 |
| `exchange_level1` | `"copy_unweighted"` | what an exchanger is at level 1, and so how a level-2 robot models an exchanging partner. `"copy_unweighted"`: the text's copy of the partner's goal without the weight, $n\,r(\tilde g_j)$; with goals inferred it is what lets two exchangers exchange (at every weight tried from 0.55 to 100, "An exchange scene"). `"none"`: Table I's N/A, the default until 2026-10-08. `"copy"`: the copy times the weight, $w\,n\,r(\tilde g_j)$ |
| `idle_hypothesis` | `True` | an extra "the partner stays put" mode in the belief of every robot that models its partner, a two-mode filter on top of Eqs. 5-6 (`idle_hazard`, `idle_eps`): on since 2026-10-08 noon (off until then); it lets two level-2 rivals pass each other at $w = 30$ (13 / 15, a stand-off without it). The one device kept beside Eqs. 5-6 on 2026-10-30. The cost: a partner that waits on purpose is read as idle |
| `physical_likelihood` | `"level1"` | the likelihood of Eq. 6, by which a robot updates its belief about its partner's physical goal: `"level1"`, the partner's level-1 hypothesis policies averaged over the social posterior, Eq. 6 as the paper prints it; `"level0"`, the partner's level-0 policies, as the authors' code (its physical-goal estimator runs on them), under which a partner's act for the robot is read as the partner's own goal ("The belief update of 2026-10-30"; since 2026-10-30); `"cut"`, the protected likelihood $\pi N_g(a) + (1-\pi) S(a)$ -- its own task by its level-1 "none" models, its acts for the robot by subgoals that do not depend on its goal -- under which neither reading of a helping act moves its physical goal ("The partner's subgoals"; since 2026-11-04; folders `..._eq56cut`) |
| `condition_on_placement` | `True` | Eq. 5's transition term: what the hands reveal of the partner's goal -- its finishing, its own set-down that did nothing, the robot's set-down for it that counted or failed, a hand-over it took or not -- rules out the hypotheses that predicted otherwise ("The belief update of 2026-10-30"). `False`: the actions alone |
| `inference_temperature` | `None` | the likelihood of Eqs. 5-6 at a tau of its own, $T \times c(\text{move})$: the hypothesis models' softmax at it, $\psi^\eta$ renormalised, $\eta = \tau/\tau_{inf}$; the robots still plan at Eq. 7's tau. `None`: Eq. 7's tau, Eqs. 5-6 as printed ("A softer likelihood"; an option since 2026-11-04; a device of before 2026-10-30, 0.5 there). Folders `..._inftau<T>` |
| `belief_lapse` | `0` | a lapse in the likelihood: each model's policy read as $(1-\varepsilon)\psi + \varepsilon/|A|$, so that one step moves no hypothesis by more than about $|A|/\varepsilon$ ("A softer likelihood"; an option since 2026-11-04; a device of before 2026-10-30, 0.05 there). Folders `..._lapse<E>` |
| `likelihood_power` | `1` | a tempered update: each posterior about the partner multiplied by its likelihood ^ $\alpha$ (the learning factor's, after the idle mode's mix; Eq. 5's), every observation counting $\alpha$ of itself ("A softer likelihood"; since 2026-11-04). 1: Bayes' rule. Folders `..._lpow<ALPHA>` |
| `option_temperature`, `option_detour`, `option_start`, `option_switch`, `option_keepaway_end` | `0.5`, `log 2`, `0.05`, `0.02`, `0.2` | the partner's subgoals under `physical_likelihood="cut"`: their policies' temperature (x c(move)), a new subgoal's prior falling by $e^{-\lambda}$ per extra move, the probabilities per transition that a subgoal starts, is interrupted, that keeping away ends ("The partner's subgoals"; since 2026-11-04). Folders `..._optT`, `_optD`, `_optS`, `_optI` where set |
| `belief_window` | `0` | a window on the evidence: each of the two physical posteriors from the uniform prior and the likelihoods of the last $n$ transitions of its own stage only -- the object's with the partner's hands empty, the destination's holding an object; what the hands' outcomes ruled out stays out; the goal hazard a floor ("A window on the evidence"; since 2026-11-04). 0: the whole past, Bayes' rule. Needs `belief_factored`, no `hand_hazard`. Folders `..._window<N>` |
| `belief_detail` | `True` | every step's belief update in the record, trace `belief_update`: every hypothesis model's Q and what it made of the action, Eqs. 5-6's likelihoods per hypothesis, destination and object, the beliefs before and after ("The belief update's workings in the record"; since 2026-11-04; `tools/belief_step.py` prints a step). Not part of the model: `--resume` keeps a record made without it. `False` (`--no-belief-detail`, `NODETAIL=1`): the record of before |
| `partner_delivers`, `partner_parks`, `partner_observed`, `social_belief`, `child_conditioning` | `False`, `False`, `"action"`, `"marginals"`, `False` | the devices of before 2026-10-30, gone: the carry and park tables, the "outcome" observation, the joint posterior, the conditioning of the partner's estimate on the robot's action inside $\tilde R_j$. Kept only so that the records of before can be read; `Config` raises on any other value (`BELIEF_DEVICES`). (The lapse and the belief temperature of then are options again: above) |
| `exchange_share` | `0.25` | $n$ of the exchange row |
| `physical_level` | `0` | the default level of a robot with a physical goal only (no social goal, or weight 0): 0, a plain MDP with nothing inferred (since 2026-10-08); 2, the default until then. A level asked for is used either way |
| `level_free_level` | `2` | the default level of a robot whose term holds no $\tilde R_j$ -- a hinderer, a competitor of a race form -- and the most a hinderer may be asked for: 2 since 2026-10-11, like every robot with a social goal, a level asked for used ("Every robot with a social goal at level 2"); 1, the rule until then (`TOLD`): such a robot at level 1 by default, a hinderer held to level 1 whatever was asked |
| `hinder_own_goal` | `False` | a hinderer has no goal of its own: $R_i = -c(a_i) - w\,r(\tilde g_j)$ ("A hinderer without a goal of its own"; since 2026-10-14). `True`: its own goal weighed against its partner's, the rule until then (`TOLD`) |
| `inertia` | `0` | a robot does not undo its last action (step back, put back what it just picked up) unless undoing beats every other action by more than this many moves' costs ("Inertia"; since 2026-10-14). The action rule only; 0: Algorithm 1's argmax |
| `automatic_hands` | `False` | nothing automatic: a robot picks up its own object by "pick" and sets it down, finishing, by "place" ("Nothing automatic"; since 2026-10-18). `True`: taken on arriving where it lies, set down on arriving at its destination, both free, the rule until then (`AUTOMATIC_HANDS`, in `TOLD`) |
| `pick_steps` | `3` | transitions a pick-up takes, the robot lifting meanwhile with "pick" its one action; 1 until 2026-10-18 (needs `automatic_hands=False` above 1) |
| `pick_cost_rate` | `3.0` | $c(\text{pick})$ in moves' worth, on the first transition of a lift; 1.0 until 2026-10-18 |
| `use_cost_rate` | `2.0` | $c(\text{place})$, $c(\text{take})$, $c(\text{give})$, $c(\text{swap})$ in moves' worth; 1.0 until 2026-10-18 |
| `belief_stages` | `True` | the physical posterior in two stages, by the partner's hands: while the partner holds nothing, its actions are evidence about which object it wants and none about its destination; while it holds an object, about its destination and none about its object ("Two posteriors, by the partner's hands"; since 2026-10-17; the social posterior is updated either way). `False`: every (object, destination) hypothesis by its own likelihood at every action, as until then (in `TOLD`) |
| `belief_factored` | `True` | the physical belief about the partner the product of a posterior about its object, learnt while its hands are empty, and one about its destination, learnt while it holds an object, each frozen in the other's stage ("The object's posterior and the destination's apart"; since 2026-11-04). `False`: one joint posterior over (object, destination), the destination believed under each object apart, as until then (in `TOLD`) |
| `lazy_mass`, `lazy_period` | `0`, `5` | lazy hypotheses (since 2026-11-04): the lightest hypothesis models of a level-2 robot, up to that mass of its posteriors, keep their last solve (every model solved at least every `lazy_period` transitions). 0: every model every transition. Not safe yet: it changed a social belief in the one test (`RUN_ALL_49.md`) |
| `goal_hazard`, `social_hazard`, `hand_hazard` | `0`, `0`, `0` | the partner's goal may change (Baker, Saxe & Tenenbaum 2009, M2; since 2026-11-02): after every update the physical posterior (in the factor the partner's next action speaks to) and the social posterior are mixed with the uniform over the hypotheses they hold possible, $(1-h)\,b + h/|S|$, so that no hypothesis falls below $h/|S|$; `hand_hazard` mixes, once, the object factor before the partner's pick or take and the destination factor before its place, give or swap ("The partner's goal may change"). 0: the update of before |
| `value_tol` | `1e-4` | every solve is certified to $\lvert V - V^*\rvert \le$ this |

`weights=` of `run_pairing` scales each robot's social term, $R_i = r(g_i) - c(a_i) +
w_i\,\xi_{ij}$. $w = 1$ is the paper: its Eq. 2, $R_i = r(g_i) + \xi_i - c(a_i)$, has no
weight. A weight on the social side is the original Social MDP's (Tejwani et al., "Social
Interactions as Recursive MDPs", CoRL 2021), whose single coefficient on the partner's
reward set how cooperative (positive) or adversarial (negative) an agent was; the ICRA
paper replaces it with the social goals of Table I. Here the goal gives the kind of
interaction and $w$ its strength: the level-2 term of a cooperator or a conflict robot is
$\pm\tilde R_j$, so its reward has that form with the coefficient $\pm w$. $w = 30$ is the
largest weight at which the solver's single-precision values stay within a tenth of a move
("Limits"). $w = 1$ is the default of `run_pairing` and of every example script.
`levels=` sets the reasoning levels ("The model": 2 by default for every robot with a
social goal, 0 for a physical goal only; 1 for a hinderer and for a competitor of either
race form under `level_free_level=1`, as until 2026-10-11). Every field of `Config` carries
its reason in `config.py`.

## How it is checked

- `test_goals.py`: Table I, all 72 cells of the column $l > 1$ and the column $l = 1$, against a
  second transcription; `hinder`; the five readings of competition: which of them race,
  and whom; the level-1 term of every goal under each, told each of the seven partner
  goals or none (only a rival's depends on it); the coefficient of the race term, the
  same at both levels; the whole level-2 term of every reading, 490 cells against the
  transcription; weights. `"rival"`: the race term against every partner but a
  cooperative one, at both levels (an exchanging partner, which has no term toward a
  competitor, is raced), and no cell that asks for the destinations; told whom it
  faces, its term is the term of `"table_hostile"` or of `"race"`, and every other
  goal's is what it is under any reading; not told, it races.
  `"table_hostile"`: all 98 cells
  (`hinder` and no social goal on either side included) against a transcription of its
  own; exactly four of them are not the table's, competition against cooperation and
  against exchange at either answer to "$g_i = \tilde g_j$?"; the competition row is the
  conflict row but for the cell against a coercing partner with the robot's own
  destination; the column $l = 1$ and every other row are the table's.
- `test_hands.py`: the rule for the hands, case by case, and every transition table against an
  independent statement of the rule (all configurations, cell pairs and joint actions). The
  own-object rule on every transition of the world: an object is set down only on its owner's
  destination and completes only its owner's goal: on the transition table and on the rule
  the record is written by, under the default rule, under three options that change the rule
  for the hands, with a spare object in play, and on the six-object scene (353,808
  transitions there). The check fails on two worlds in which the rule was broken by hand in
  one entry of the table; `Config` has no switch for the rule.
- `test_explicit.py`: the rule of 2026-10-18 ("Nothing automatic"): the options and their
  validation, the records of before, the statuses of a lift and the configurations, the rule
  on the real objects (nothing on arriving; a pick-up of three "pick"s with the lifter busy
  and the partner unable to take or give meanwhile; the costs 3 then 0 and 0, and 2; "place"
  on its destination finishing, elsewhere nothing; swap in one transition; the rule of before
  under `AUTOMATIC_HANDS`), the way left counting the hands' transitions, the engine against
  brute force at levels 1 and 2 -- the reference (`reference.Ref`, `Brute`) has the lifts,
  the explicit set-down, the costs and the partner's own set-down on any candidate written
  out apart, in both modes -- a rollout of each robot lifting, carrying and placing (9 / 9;
  the lift's steps the one legal action; the partner's lift no evidence), its record and
  videos, a hinderer lifting the object it guards, and a model placing rather than hovering.
  The tests written before run under the rule of before (`hands_before.py`, `told.py`).
- `test_models.py`: the level-0, level-1 and level-2 models against brute force on a 3 x 3 map
  (partner distribution, expected rewards, values, Q, policies): 23 pairings and states (27
  with the slow tests), among them two competitors of either race form before and after one
  goal is done, a level-2 robot that models a competitor, a level-2 cooperator whose
  $+\tilde R_j$ and a level-2 conflict robot whose $-\tilde R_j$ hold the competitor's
  term, a scene in which an object lies on a candidate destination (standing there is not
  having finished), and `hands_full_reward="none"`. The hinder and race models agree to
  4e-6 in the rewards and 2e-4 in Q at weights up to 30; a level-2 model that holds a
  competitor's term at weight 30 to 4e-5 in the rewards and 6e-4 in the values and Q; the
  worst case at weights up to 1 is 6e-4 in Q (level-2 conflict against cooperation; rewards
  and policies are stored in single precision). And the identity behind "race, then
  hinder": on the 288 states of that map in which a robot's goal is done, the solved models
  of a competitor and of a hinderer differ by $2w\rho$ in the rewards and by
  $2w\rho/(1-\gamma)$ in V and Q (measured: the rewards to the last stored digit, V and Q
  to 1.2e-5 at $w = 1$ and 1.6e-4 at $w = 30$), and pick the same action wherever the
  hinderer's choice is clear; under `"race"` the values do not differ by a constant.
  A competitor of a race form at level 2: six more pairings and states (13 with the slow
  tests), among them two level-2 competitors before and after one goal is done, a level-2
  competitor and a level-2 cooperator in either order, a level-2 competitor and a hinderer
  that hold each other's object, `"race"`, and, with the slow tests, a partner whose
  social goal the competitor has to infer; their level-2 models agree with brute force to
  5e-4 in the rewards and 5e-3 in the values and Q at weights up to 30 (that is the case
  of a competitor at weight 30 against a cooperator, whose policy is stored in single
  precision and multiplied by the competitor's rewards; 1.4e-5 and 4.2e-4 in the other
  twelve). The identity above holds at level 2 as well, against a level-2 hinderer built
  for the comparison, to the same digits (the same choice in all 274 and 281 states where
  the hinderer's is clear).
  `"table_hostile"`: two more pairings against brute force, with the transcription of the
  turned table (seven with the slow tests, among them an exchanging partner, a coercing
  one, where the cell asks for the destinations, and a partner whose social goal has to
  be inferred, so that turned and untouched cells stand in one term); they agree to 8e-6
  in the rewards and 4e-4 in the values and Q at weights up to 2. And the identity with
  conflict: at two pairs of weights and in either seat, the solved level-2 models of both
  robots (rewards, V and Q on 1008 states) are the arrays of the conflict pairing, bit for
  bit, against a cooperative, an exchanging and a competing partner, and are not against a
  coercing one; under the table as printed the competitor's models are other arrays
  against a cooperative and an exchanging partner and the same ones against a competing
  and a coercing one, and its partner's models are the same in every case. With an
  inferred partner goal the competitor's models are not the conflict robot's, and are
  again once coercion is taken out of the hypotheses.
  `"rival"`: three more pairings and states against brute force (two rivals, and a rival
  and a cooperator in either order), 14 with the slow tests: two rivals with one done, a
  level-1 rival that is told it faces a cooperator, an exchanging, a coercing and a
  hindering partner, a partner without social goal, a cooperator with weight 0, and
  partners whose social goal has to be inferred, so that the race term and
  $-\tilde R_j$ stand in one term and a level-1 rival that is not told races. They agree
  to 2e-5 in the rewards and 2e-3 in the values and Q at weights up to 30 (the worst case
  is a cooperator at weights 1.5 and 2 whose $+\tilde R_j$ holds the term of a level-1
  rival that is told it faces a cooperator; 9e-6 and 7e-4 in the other thirteen). What
  this comparison does not check is who the helper is and who is told what: the
  reference states those rules a second time, it does not derive them. And the switch:
  for six combinations of partner, weights and levels (23 with the slow tests: a
  cooperative, an exchanging, a competing and a conflict partner and one without social
  goal, at levels (2, 2), (1, 2) and (2, 1), a cooperator with weight 0 and one at
  level 0) the model each robot acts by, its rewards, values and Q on the states it can
  reach, is the array of `"table_hostile"` where the rival faces a cooperator and of
  `"race"` where it does not, bit for bit, and is not the other reading's.
- `test_competition.py`: the regression test of the five readings. At the default levels,
  both race forms: two
  competitors finish at 9 / 11 at $w$ = 0.8, 1, 2, 30, each with its own object; the term
  is 0, $-w\rho$ and $+w\rho$ in the transitions where it should be (the default: between
  $w\rho$ and $2w\rho$ while the partner is on its way); against each of the seven social
  goals of a partner, in either seat, at $w$ = 1 and 30, a competitor never hands its
  partner's object over or sets it down (28 rollouts each); levels. The default: once done
  a competitor leaves its destination, two moves toward the doorway; against a cooperator
  it picks the cooperator's object up after its own goal is done, not before, and keeps
  it; with the seats swapped it does nothing for the cooperator, which stays where it is
  (at $w = 1$: it stands guard without using its hands); on the far-room map the robot
  behind never gets through; in the 28 rollouts it touches its partner's object only after
  its own goal is done. `"race"`: it never uses its hands against a cooperator (5 / 17, and
  19 / 31 at $w = 0.8$), never holds its partner's object, and the robot far behind
  finishes (9 / 17). `"table"`: a stand-off at level 1, and against a cooperator the
  competitor delivers for it. A competitor asked for level 2 (`LevelTwoCompetitorTests`):
  two of them finish at 9 / 11, and are paid what the term says; seated first against a
  cooperator the default form takes the cooperator's object, at transition 2 for
  $w \le 2$, and keeps it, and `"race"` leaves it alone (5 / 17 at every weight); seated
  second both forms take it at transition 2 for $w \le 2$; against a hinderer it takes the
  hinderer's object in turn and offers a swap to the end; on the far-room map the robot
  behind takes the object of the one ahead at transition 2 and neither finishes, at levels
  (2, 2) and (1, 2), while under `"race"` both finish; past the doorway one cell wide two
  of them stand off, and with only agent_2 at level 2 it is agent_1 that gives way; and,
  with the slow tests, the default form against each of the seven social goals in either
  seat never sets its partner's object down for it or hands it over (28 rollouts).
  `"table_hostile"` (`HostileTableTests`): against a cooperator the competitor takes the
  cooperator's object and holds it to the end, at the four weights, never sets it down
  for the cooperator, hands it over or tries to, and the cooperator never finishes, where
  the table as printed ends 5 / 5; with the seats swapped the cooperator never finishes
  and the competitor does nothing for it; against an exchanging partner neither goal is
  done (as printed: 17 / 5); each of these is the rollout of a conflict robot in the
  competitor's place, every action and every Q value; two such competitors roll out as
  two of the table do, and never finish; at level 1 the rollouts are the table's; with an
  inferred partner goal at $w = 1$ the competitor ends up sure that its partner
  cooperates, takes its object at transition 2 and keeps it (as printed: 5 / 6), with the
  actions of the conflict robot and another Q; and, with the slow tests, over each of the
  seven social goals in either seat at $w = 1$ (14 pairings) the rollout is other than
  the table's exactly where the partner is cooperative or exchanging, and is the conflict
  robot's wherever the partner is not a coercing one. (The same at $w = 30$ is in the
  runs of "Results", compared by actions, events and finishes, not in the suite.)
  `"rival"` (`RivalTests`): two rivals finish at 9 / 11 at the four weights at their
  default levels, (2, 2), each with its own object and no other, never using its hands
  for anything else and never in the other's way, the rollout of two competitors of
  `"race"` asked for level 2; the same at levels (1, 1), (1, 2) and (2, 1). The same
  pairing under `"table"` at $w = 30$ is the two-cycle of "Why two competitors of the
  table ping-pong": no event in 52 transitions, each robot on two cells from the third
  transition on, every move the reverse of the one before, and in every transition
  exactly one of the two on the place of the other's object. Against a cooperator the
  rollout is the one of `"table_hostile"` in either seat; seated first the rival takes
  the cooperator's object and keeps it. A cooperator with weight 0 is raced (9 / 11),
  and the record says "competition vs none" for one at level 0 as well. An exchanging
  partner is raced (9 / 11 at $w = 30$, the rollout of `"race"`), where the competitor
  of `"table_hostile"` takes its object and neither goal is done. With an inferred
  partner goal at $w = 1$ the rival takes the cooperator's object at transition 2, keeps
  it, and ends up sure that its partner cooperates (5 / –).
  And, with the slow tests, over each of the seven social goals in either seat at
  $w = 1$ (14 pairings) the rollout is the one of `"table_hostile"` where the partner is
  cooperative and the one of `"race"` at the same levels everywhere else, and never the
  other's. "The rollout of" is every action, event, finish and Q value, and every entry
  of both ledgers (what each robot was paid, read off the transitions that came about).
  (The same at $w = 30$, the maps, unequal weights and the other levels against a
  cooperator are in the runs of "Results", not in the suite: there every action, event,
  finish and Q value and every entry of the ledgers was compared, in 127 comparisons
  of a rollout under `"rival"` with the same pairing under the other reading.)
  Not pinned, because they rest on 0.0002 to 0.12 in Q: when a level-2 competitor of the
  default form takes a cooperator's object at $w = 30$, and what it does seated second at
  $w = 30$; in which transition it takes a hinderer's; the far room at levels (2, 2) and
  $w = 1$ (12 / 28); what the robot whose object is held does meanwhile; what the
  cooperator does after its delivery under `"race"` at $w = 0.8$; and in which transition
  a competitor of `"table_hostile"` takes a cooperator's object, or whether it takes it
  seated second. In no test: the runs
  with a level-2 robot without social goal in the competitor's place ("What is the term's
  doing, and what the level's"), `partner_delivers=False`, the walled-in doorway at mixed
  levels, levels (2, 1) against a level-1 cooperator, sampled actions and the idle
  hypothesis at level 2, the outcomes against a conflict partner, $w = 100$, an inferred
  partner goal for a level-2 competitor of a race form or at $w = 30$, `"table_hostile"`
  at levels (2, 1), the comparison with $\tilde R_j$ in the term (a prototype), every
  run of a rival with an inferred partner goal but competition / cooperation at $w = 1$,
  a rival against a conflict robot or a hinderer beyond the 14 pairings at $w = 1$, the
  maps, and the numbers of "Why two competitors of the table ping-pong" beyond the
  rollout itself (what each robot expects, plans and is paid there; the two
  prototypes).
- `test_hinder.py`: the regression test of `hinder` against cooperation, competition and no
  social goal at $w$ = 0.8, 1, 2, 30.
- `test_render.py`: what the figure says on known runs (action, hands, last event and goal at
  chosen frames, the timeline, where an event is called out, a record read back from JSON;
  the wording of every event). Four runs that between them hold every kind of event and of
  ending (a set-down for the partner, a failed set-down, a hand-over, a hand-over by force,
  a put-back, a deadlock, the step limit) are compared with their records frame by frame,
  by a second reading of the record. The map panel: the legend and every label inside the
  figure and clear of each other, on a walled map (every frame), a corridor and a 3 x 3 map
  without walls. The text: the font, and that nothing in the status table runs into its
  neighbour or out of the panel, in DejaVu Sans and in matplotlib's own font, with long
  names and titles. The files it writes (frame count, holds). The headline names the
  reading of competition a record was written under. A record written under the any-object
  rule, which this package no longer has, is refused.
- `test_pairing.py`, `test_hindrance.py`: the record, JSON, errors, weight 0, the default
  levels and the levels asked for, a swap that finishes both goals; the metrics.
- Hand-off cells, rooms and `hand_over` (2026-10-07; the hand actions since 2026-10-16).
  `test_hands.py`: the cases (the partner's object put down on a hand-off cell lies
  there; its owner takes it on arriving, its partner only by picking it up; with both
  objects on the cell a robot picks up its own; on a hand-off cell next to its partner a
  robot may put the object down or give it -- until 2026-10-16 "use" put it down, and
  nothing was handed over from a hand-off cell -- and the robot next to the cell gives to
  the one standing on it; at the doorway a put-down makes a hand-over to the robot that
  puts down fail, and two robots that give trade (with "use" they did not swap there); a
  robot without social goal does nothing for its partner: wherever it may use its hands
  with its partner's object in them, its one hand action is "place" on that object's
  place, and the object goes back there, whatever its partner does; with
  `hand_over=False` nothing is handed over anywhere, and away from a hand-off cell a
  robot holding its partner's object has no hand action; a robot does not leave its room,
  and an object in the other room is out of its reach and pays it nothing), every pair of
  moves of two robots, with rooms and without, against the collision rule written out a
  second time, and every transition table and every "may it choose this hand action
  here" against the independent statement of the rule: with a hand-off cell, with rooms
  as well, without hand-overs, under forced swap. The own-object rule on every
  transition of those worlds. `test_models.py`: every model against brute force
  on a map of two four-cell rooms and a doorway, where the reference counts each robot's
  ways through its own room and its legal moves from the rooms: five pairings and states
  (nine with the slow tests), among them the two exchangers with each other's object in
  their hands, one in the doorway and one next to it, at weights 30 / 30 under either
  rule, an object lying in the doorway, both lying there, a level-1 cooperator next to a
  robot without social goal, and, slow, inferred partner goals with the idle mode, forced
  swaps with the reward that follows a carried object, and set-downs for the partner in
  the rooms. The largest difference in Q in the nine is 1.7e-4.
  `test_pairing.py`: the exchange scene end to end with told goals. The events of two
  exchangers at weights 30 / 30 and 1 / 1 and with hand-overs allowed, who delivered
  what, that nobody leaves its room and nobody finishes alone. The controls: two robots
  without social goals at level 2, an exchanger next to one, and two cooperators do not
  move; a cooperator next to a robot without social goal gives and gets nothing back; two
  rivals trade at 30 / 30 and do not move at 1 / 1. Slow: at nine weights from 0.2 to
  100, with hand-overs allowed and without, both goals are done, each object is put down
  once and nothing is handed over. And what a scene refuses (a hand-off cell on an
  object's place or on a candidate destination, a robot that starts outside its room).
  `test_render.py`: the put-down in words, the rollout of the two exchangers compared
  with its record frame by frame (also read back from JSON), the legend with the rooms
  and the hand-off cell inside the map and no label of the map over another in any
  frame, two objects on the hand-off cell drawn apart, and the legend of before on a
  scene without rooms.
- Inferred goals on the exchange scene and what an exchanger is at level 1
  (2026-10-07). `test_goals.py`: the level-1 term of an exchanger under the three
  readings at weights 0, 0.5, 1 and 30 (the copy times the weight is 0, 0.125, 0.25 and
  7.5; the copy at its own size 0, 0.25, 0.25 and 0.25; the table's N/A is 0), that the
  level-1 term of no other goal changes with the option and that the level-2 row of
  exchange keeps its weight, and what a record says of the term. `test_models.py`: every
  model against brute force on the two-room map with the partner's social goal inferred
  over the paper's five goals and the copy at its own size, at weights 30 / 2 (per robot
  two destinations times five kinds of partner at level 1, and its own model and
  physical twin at level 2), and, slow: at 30 / 30 with each robot holding the other's
  object at the doorway; at 2 / 3 with an object lying in the doorway; at 1 / 5 with
  hand-overs allowed next to a cooperator; at 5 / 2 under the copy times the weight; and
  at 3 / 0 next to a partner of weight 0, whose five hypotheses are one model. And, told,
  a level-1 exchanger with the copy at its own size next to a level-2 cooperator (3 / 1)
  and next to a partner with weight 0. The reference is
  told how the copy is weighted, in one line as the package is, and computes everything
  after that by itself. The largest difference in Q in these eight comparisons is
  7.4e-4, in the told one at 3 / 1 (at 30 / 30: 4.9e-4). `test_pairing.py`, with
  the rival as the competitor among the hypotheses: two exchangers that are not told
  each other's goal, event for event (18 / 15 at 30 / 30), and the same events with
  Table I's competitor and with `"race_then_hinder"` among the hypotheses; what each
  believes on the way (a fifth on each goal at the start, cooperation and exchange
  together above 0.99 from the third transition, "exchange" never below 0.02, above
  0.95 for agent_1 at the end and between 0.1 and 0.3 for agent_2); under Table I as
  printed at 30 / 30 both keep the other's object and hold "cooperation" above 0.8;
  under the copy times the weight each takes one step at 30 / 30 and no object is
  touched, and the two exchange at 2 / 2; at weight 1 the two copies give the same
  actions and the same Q; two robots without social goals do not touch an object, an
  exchanger next to a hinderer takes the object and keeps it, two cooperators finish
  one after the other. Slow: the copy at its own size at nine weights from 0.2 to 100,
  with hand-overs allowed and without, and at seven weights with Table I's competitor;
  Table I as printed at the same nine (who finishes, and at 2 / 2 that the robot that
  was helped is sure of a cooperator). `test_render.py`: the panel of social-goal
  beliefs shows the beliefs of the record in the frames checked, the true goal
  outlined; no tick label of the two belief panels runs into the next, each title stays
  over its own panel and one legend sits under both; a record with told goals has the
  one panel of before; and the same from the JSON file. Not tests in the suite: the
  tables of "An exchange scene" (each cell one rollout), the rollouts of 200
  transitions, the denser grid of weights, the runs with fewer hypotheses, and the 13
  runs with a first step made a wait.
- Hand-overs only (`set_down_for_partner=False`), corrected on 2026-10-07 ("Limits").
  `test_models.py`: three pairings and states under that option on the 3 x 3 map against
  brute force, on every block any model reaches, a goal done among them (largest
  difference in Q: 4.4e-4); every model is
  solved, and every partner policy and partner distribution sums to one, on all those
  blocks, which hold blocks only one robot's own dynamics reach; the blocks of a table
  are cached by what it holds (a table written into the array of another gets its own).
  `test_pairing.py`: the same call rolls out the same four times in one process, Q
  included; two level-2 rivals with hand-overs only set their own objects down (9 / 11 on
  the six-object scene). With the second correction taken out again, the comparisons with
  brute force fail: the partner's distribution is off by 1.0.
- Not a test in the suite: on the scenes there were, under the rule there was, nothing is
  changed by hand-off cells, rooms, `hand_over`, the two corrections, or the options of
  the inferred exchange. The 154 rollouts of the check above were rolled out again by
  the final version: the same actions, events, Q (with and without the social term),
  beliefs, expected rewards, ledgers, levels and descriptions of the terms, to the last
  bit, in all 154. So were 19 rollouts under options those do not touch, by the version
  of the morning of 2026-10-07 and by this one: a robot whose goal is done keeping its
  hands still (seven pairings), the reward that follows a carried object, a partner
  model that neither carries nor parks, a separation between the robots, the marginal
  belief, placement not conditioned on, a level-1 term for exchange, the spare objects
  modelled, and the two-object scene. Eighteen are the same in every field; the one with
  `exchange_level1="copy"` differs in the description of the term alone, which the
  earlier version printed as the table's N/A. And 139 records that the morning's version
  had written to its output folders were written again (the 49 pairings at $w = 30$
  under `"rival"`, the 13 with a competitor under each of the other four readings, the
  five runs of `rooms` and `corridor`, three pairings at $w$ = 1 and at 30 from
  `run_pairings.py`, the fifteen of `compare_competition.py`, and the baselines of those
  folders): per transition the same actions, positions, statuses, Q with and without the
  social term, beliefs, expected social rewards, physical rewards and events, and the
  same ledgers, in all 139; the 49 summary rows of the pairings at $w = 100$ are the same
  as well. One thing is changed on purpose: a robot that uses its hands where several
  objects lie takes its own first, where it took the first of the list. That can differ
  only where both robots' objects share a place, which no scene here has. And runs with
  `set_down_for_partner=False` change, as they were meant to. (After this check a
  comment in `config.py` and the docstrings of two example scripts were edited; their
  code is the same, statement for statement.)
- Not a test in the suite: the level option changes nothing unless a level is asked for,
  and `"table_hostile"` changes nothing under the other three readings. 87
  rollouts of at most 60 transitions (the three readings of competition there were then,
  23 of the rollouts under `"table"`; the six-object
  scene, the far-room map and the doorway one cell wide; among them sampled actions,
  forced swaps, `hands_full_reward="none"`, the partner's action cost in $\tilde R_j$ with
  the idle hypothesis, and level-2 robots that have to infer their partner's social goal)
  were rolled out by the version before the option, asked for levels (2, 2) as it had to
  be, by this version without `levels`, and by this version with the levels the earlier
  one used given by number: the same actions, events, Q (with and without the social
  term), beliefs, expected rewards and ledgers, to the last bit, in all 174 comparisons.
  And for all 4704 combinations of two social goals, four weights each (0 among them), the
  three readings there were and a known or inferred partner goal, `default_levels` gives
  the levels the earlier version used when asked for (2, 2); under `"table_hostile"` it
  gives the levels of `"table"` (1568 combinations).
- Not a test in the suite either: `"rival"` changes nothing under the four readings
  there were. 154 rollouts were rolled out by the version of 2026-10-06 and by this one:
  the 87 above at the default levels, 23 under `"table_hostile"`, and 44 with a competitor
  of a race form asked for level 2 (under each form eight pairings at two weights, the
  far-room map, the doorway one cell wide, two runs with inferred partner goals, one
  with `hands_full_reward="none"` and one with the partner's action cost and the idle
  hypothesis), whose race term at level 2 went from one coefficient per model to one
  per hypothesis about the partner. The same actions, events, Q (with and without the
  social term), beliefs, expected rewards, ledgers, levels and descriptions of the terms,
  to the last bit, in all 154. `default_levels` gives the levels of that version on all
  6272 combinations under the four readings, and under `"rival"` the levels of `"table"`
  (1568). (The 154 were rolled out before five docstrings and comments of this version
  were corrected; with docstrings removed the syntax trees of its Python files are the
  same before and after. The six GIFs of `compare_competition.py` that the version of
  2026-10-05 wrote come out byte for byte the same.)

- Every partner goal inferred by default (2026-10-08). The tests written before run under
  the defaults of then: `tests/told.py` hands them `Config`, `run_pairing`,
  `run_baseline`, `default_levels` and `levels_used` with `TOLD`, `physical_level=2` and
  `idle_hypothesis=False` (`BEFORE`) as the defaults, so every number they pin is still
  the one of then. `test_defaults.py` tests the defaults of now: the four that changed in
  the morning, and that `TOLD` gives back exactly those four and nothing else; the two of
  noon (a physical goal only at level 0, the idle hypothesis on); the default levels (a
  rival at 2, a hinderer at 2 since 2026-10-11 and at 1 under `TOLD`, a physical goal only
  at 0, a level asked for used, a hinderer's too unless `level_free_level=1`); the
  models a level-2 robot judges its
  partner by (seven goals, six level-1 models per destination, conflict and hinder one of
  them; a uniform prior over the 7K hypotheses; a level-1 rival that is not told whom it
  faces races; the copy $n$ whatever the weight); a rival's level-2 term, $-\tilde R_j$
  against a cooperator and the race against each of the six other goals; the package's
  example, hinder / cooperation at 2 / 2, told and nobody told ((5, None) and (7, None);
  at 30 / 30 nobody told among the slow pairings); two exchangers on the exchange
  scene event for event (17 / 16); the figure with seven goals (no tick label into the
  next) and its GIF; and, slow, seven pairings of the six-object matrix. `test_models.py`:
  the defaults against brute force on the two-room map (a rival next to an exchanger at
  2 / 3, a cooperator next to a rival at 30 / 30 with each holding the other's object)
  and, slow, four more pairings on the 3 x 3 map (two robots without social goals at
  30 / 30, a hinderer next to a cooperator, two rivals once one is done, coercion and
  conflict). They agree to 2.4e-6 in the rewards and 1.7e-4 in Q at level 1, and at level
  2 to 5.4e-5 in the rewards and 7.2e-5 in Q in five of the six; the sixth, a cooperator
  next to a rival at 30 / 30, whose $+\tilde R_j$ holds the rival's term, to 2.1e-5 in the
  rewards and 1.4e-2 in Q, within what the single precision of the stored rewards gives at
  that weight ("Limits", large weights: 0.06 for a rival against a cooperator at 30).
  Told, nothing has changed: 14 rollouts of the records written on 2026-10-07 (eight of
  the 49 told pairings of the six-object scene at $w = 30$ under the rival, six of those
  of the exchange scene) were rolled out again with each robot told, under the
  configuration they were written with: the same in every field of every transition, the
  events, the ledgers and the finishes. The tables of "Results, nobody told" and the table
  "exchange / exchange, the seven goals of the defaults" are not tests in the suite (each
  cell one rollout, logged).

- The temperature of inference (2026-10-09). `test_inference.py`: `None` is Eq. 7's
  $\tau$, a number that many $c(\text{move})$ (reward units with
  `temperature_unit="absolute"`), planning's $\tau$ untouched, bad values refused; at three
  temperatures every live level-2 hypothesis's log-likelihood of every legal partner action
  is $\log\mathrm{softmax}(Q_k/\tau_{\text{inference}})$ (no lapse), and the values the
  robots act on are the same at all three; slow: `inference_temperature=0.2` rolls coercion
  against a rival on the six-object scene out exactly as `None`.
- What a robot sees of its partner (2026-10-13). `test_observation.py`: the option's values
  and its neutral value for records written before it; on the 3 x 3 map, the actions a robot
  is seen to have possibly chosen -- after a clash on a cell (the first robot has the right
  of way), a head-on swap, a move into a robot that waits, moves that go through, "use" --
  under each reading, against the world's own step taken one action at a time; in every
  slot, depth 1 and 2, the likelihood of a set of actions is the summed probability of its
  actions, the lapse mixed in once per legal action of the set; under `"stay"` a blocked
  move updates every belief exactly as a wait does under `"action"`; after a head-on push
  the idle probability falls to the hazard under `"action"` and rises under the other two;
  and a rollout records `seen_as` only with the option on. Under `"action"` (the default)
  conflict / none of the 49 below was rolled out again: its first ten transitions are the
  record's in every field.
- The tier scenes. `python examples/tier_scenes.py` stops unless every scene is fair at
  0.5 m, has its tier at 0.25 and at 0.5 m by test_tiers' rules, and leaves each robot's
  task possible with its partner parked; and unless the search, run again from test_tiers'
  placements, gives the twelve placements it holds. Its copy of test_tiers' geometry gives
  all twelve their tier on the 0.25 m map, and its A* lengths are the breadth-first ones.
  `run_tier_scenes.py --resume` keeps a record only if it was written with the same
  configuration on the same scene.
- Which object the partner wants (2026-10-09). Known, nothing has changed: six pairings
  rolled out to the end before and after (on the six-object scene cooperation / exchange,
  coercion / competition, hinder / cooperation, competition / competition at 30 / 30 and
  none / conflict, on the exchange scene exchange / exchange) give the same actions, Q
  values, beliefs and events, to the last bit. Inferred: `test_hands.py` checks every
  dynamics table of the models -- each robot's, under each physical goal, with its act for
  the partner failing and counting -- against the reference's statement of the rule on the
  3 x 3 map, with a spare object, and on the map with the doorway, rooms and a hand-off
  cell; `test_models.py` checks the models at levels 0 to 2 against brute force with every
  object a candidate on the map with the doorway (a rival next to an exchanger, and a
  cooperator next to a rival at 30 / 30 with each holding the other's object) and, slow,
  four pairings on the 3 x 3 map. `test_objects.py`: the option (inferred by default,
  known under `TOLD`); a spare object in the model where it is inferred and scenery where
  it is known, and objects left out by `max_objects` scenery; the prior, uniform over M K
  physical goals times the social goals, its marginals uniform; the records (an object
  belief per robot and transition, summing to one; two cooperators on the 3 x 3 map end
  sure of each other's object, 12 / 21 where they finish at 7 / 5 told the object); the
  figure (the object and destination bars are the record's, outlines on the partner's own
  object and true destination, no tick label into the next, the figure's text against the
  record); a hand-over and a set-down of the wrong object told apart by `real_step`, and
  what each rules out of the belief (and what the partner's finish keeps); the partner's
  estimate of a finished robot's reward under "it wants another object"
  (`ObjectWorld.cls_est`, against the reference) and, slow, two cooperators on the 3 x 3
  map with one goal done against brute force; the memory guard; the acts in vain in words.

- **The selective cooperator** (2026-10-11). `test_goals.py`: every cell of the level-2
  term (seven goals against seven partner goals, both answers to "$g_i = \tilde g_j$?", four
  readings of competition) under the default partners, under `"all"` and under another
  choice, against the reference's transcription, which leaves the cooperator's cell out by
  the partner's name; no other goal changes; a hinderer and a conflict partner can be left
  out apart; the option's names and aliases, the folder suffixes and the record's words.
  `test_models.py`: a level-2 cooperator next to a level-1 hinderer under the default, and
  with only a hinderer and a robot without social goal cooperated with (the hostage state),
  against brute force: level 2 to 1.2e-7 in the rewards and 1e-5 in Q; under the defaults
  of now a cooperator next to a rival at 30 / 30 to 1e-2 in V and 4e-3 in Q (single
  precision at that weight). With the rule taken out of the reference, the first comparison
  fails (0.77 in the level-2 rewards). `test_objects.py` and `test_defaults.py`: the 3 x 3
  map's two cooperators (14 / 4, 7 / 8) and the package's own example (hinder / cooperation
  on the six-object scene at weights 2 / 2: – / – with the hinderer at level 1, as it was
  then) pinned under the new cooperator, and the numbers of before under
  `cooperate_with="all"` (12 / 21, 7 / 5; 7 / –). Every older test runs under `TOLD`, which
  restores Table I's cooperator.
- **Every robot with a social goal at level 2** (2026-10-11). `test_defaults.py`: the
  default levels and the levels used -- a hinderer and a competitor of a race form at level
  2, a hinderer asked for level 2 at level 2 -- under `level_free_level=1` and under `TOLD`
  the rule of before, a physical goal only at level 0 either way; a record written before a
  field of `Config` existed compares with the options it was made with (`written_with`);
  the package's example at levels (2, 2): 5 / –, the cooperator setting the hinderer's object
  down at 5 while the hinderer keeps the cooperator's from 3. `test_models.py`: a level-2
  hinderer next to a level-2 cooperator, each inferring the other's goal and object, against
  brute force, 72 level-1 and 4 level-2 models: level 2 to 1.4e-7 in the rewards and 1.1e-5
  in V and Q. `test_goals.py`: the folder suffix. Every older test runs under
  `TOLD`, which keeps a hinderer at level 1.

- **Five hand actions in place of "use"** (2026-10-16). `tests/reference.py` states anew
  what each hand action does and where a robot may choose it; `test_hands.py` checks every
  transition table and every "may it choose this hand action here" against it in every
  world of before (each option of the hands rule, forced swaps, hand-off cells, rooms,
  without hand-overs, inferred partner objects, swaps and steals, spare objects), with the
  cases of before carried over to the hand action that did what "use" did, and the cases
  where a robot now has two (on a hand-off cell next to its partner: put down or give; at
  the doorway a put-down makes a hand-over to the robot that puts down fail, and two robots
  that give trade; a pick-up where nothing lies leaves the hands free to receive).
  `test_models.py`: every model against brute force with ten actions, as before (48 tests;
  35 minutes for the two files on two cores). `test_pairing.py`: a record names its
  actions (`actions`); one without, rolled out with "use", is not drawn, and the example
  scripts' `--resume` does not keep it. The rollouts that changed are listed in "Five hand
  actions in place of "use"". Two pins of `test_objects.py` were stale before 2026-10-16
  already (the two cooperators inferring each other's object: never / never and 8 / 8
  under the package of 2026-10-14, against 12 / 21 and 7 / 8 pinned), and two tests there
  still expected `real_step` to return four fields, where it returns five since
  2026-10-12.

- **Two posteriors, by the partner's hands** (2026-10-17). `tests/reference.py` states
  the update anew in probabilities (`staged_posterior`); `test_stages.py`: the option, its
  value in `TOLD` and for records of before; which factor the state freezes, in every slot,
  by the hands of the robot the belief is about; Eqs. 5-6 with the stages in every slot at
  depth 1 and 2 against the reference fed the engine's own likelihoods (hands empty, R0's
  full and giving under each reading of what a robot sees, both full and trading with no
  lapse: to 1e-9); with the idle filter on, the destination's belief exactly as before
  while the partner's hands are empty and the object's moved, and the reverse while it
  holds; a goal done keeps only its object and destination, the frozen slice ruled out
  giving its belief to the others; `belief_stages=False` moves the destination's belief
  from the first action; along a 25-transition rollout every belief obeys the stages, the
  prior uniform, and without the option the run is the one of before. The rollouts that
  changed are listed in "Two posteriors, by the partner's hands".

- **The value videos** (2026-10-17, evening). `test_models.py`: in every comparison with
  brute force, the root model's value by term (`SocialMDP.value_components`) against the
  reference's own rewards term by term (`Brute.level1` / `level2` with `extra`), evaluated
  under the engine's greedy policy on the reference's transitions (`Brute.evaluate`): each
  term to 3e-4 x the size of the rewards, the terms left out paying nothing, the terms
  adding up to V. `test_values.py`: the option and records of before; the terms add up to
  V, the policy is the argmax of Q, a cooperator's terms and a spoiler's (costs and
  $-r(\tilde g_j)$ only); the maps are the values with the robot moved cell by cell, the
  partner's cell blank, the robot's own action the map's; a level-0 robot has none; a
  rollout holds the maps of every transition, its own action the map's (no inertia), JSON
  keeps them (NaN as null); `value_maps=False` leaves them out and changes nothing else;
  `render_videos` writes the three videos, each transition one second and the last three,
  `render_value_video` refuses a record without maps; the figure's panels, titles and
  subtitle. `test_render.py`: the MP4 holds every frame of the GIF as long (the frames
  counted and timed by ffmpeg), its first frame is the figure to H.264's rounding, padded
  white; the robots' colours.

## Limits

- **One scene, greedy actions.** Everything above is the two-room map and five variations of
  it. With the partner's object known `six_object_scene` and `two_room_scene` roll out
  identically: the spare objects are scenery.
- **With the partner's object inferred, few objects.** Every hypothesis about the partner
  is a model solved on every joint state, and every object multiplies the hypotheses by K
  and the object configurations by more: six objects on the two-room map need about 13
  GiB, the exchange scene with its six far more ("Which object the partner wants"). On a
  machine with 8 GB the scenes with spare objects run inferred only with `max_objects` at 3
  or 4, or with the object known; and a transition takes three to twelve times as long as
  with the object known. Every result in "Results, nobody told" was made with the object
  known.
- **A cooperator's help rests on its belief** (since 2026-10-11). It cooperates with its
  partner only as far as it takes it to be cooperative, competitive or exchanging: it reads a
  helper as a hinderer at first, rarely recognises an exchanger, and cannot see a rival turn
  on it; against a hinderer that holds its object it never finishes ("Whom a cooperator
  cooperates with"), and a level-2 hinderer can still have its own goal done by it ("Every
  robot with a social goal at level 2").
  Every result in "Results, nobody told" and in "Which object the partner wants" was made
  with Table I's cooperator.
- **Neither race form is in the paper, and neither is `hinder`.** All three are built from
  Table I's terms on goal rewards. Their robots reasoned at level 1 by default until
  2026-10-11 (a competitor at level 2 if asked for, a hinderer never), and at level 2 since
  ("Every robot with a social goal at level 2"); the results marked told were made with
  them at level 1 and each partner told the other's social goal
  (`partner_social_goal="known"`). `competition="table"` is the table. The paper runs
  its scenarios at level 2, where its competitor helps a cooperative partner: the default
  here is this project's reading of "competitive", not the paper's. `"table_hostile"` is
  not the paper's either: it changes one sign of Table I. Nor is `"rival"`: against a
  cooperator it keeps the table's entries with that sign turned, and against anybody else
  it replaces the competition row, at $l = 1$ and above, by the race term.
- **`"rival"` settles what two told competitors do, and little else.** It has been the
  default reading since 2026-10-08 (until then `"race_then_hinder"`, which goes after its
  partner once its own goal is done). Told, two competitors of either reading of the
  table still never finish (nobody told, two of `"table"` finish at $w$ = 0.8 and 2). What
  stays open under it:
  - *Against a cooperator it hinders.* There it is the competitor of `"table_hostile"`,
    a conflict robot at level 2 and a hinderer at level 1, with that reading's limits:
    seated second against a cooperator at $w = 30$ it takes nothing, and with only the
    rival at level 2 at $w = 1$ neither goal is done. This is the request of 2026-10-06
    kept next to the request of 2026-10-07 ("competition doesn't mean hinder"), which
    was made of two competitors; `"race"` is the reading without it ("The model").
  - *Past a doorway one cell wide two rivals at their default levels stand off* (– / –,
    25 blocked moves each), as two level-2 competitors of either race form do, and in
    the walled-in doorway as two level-1 ones do. Nobody touches an object; each plans
    against a level-1 model of the other that would give way. With one of the two at
    level 1 both get through (11 / 12, 9 / 12; 14 / 11, 9 / 14).
  - *On the standard map two rivals act as two robots without social goals do*: at
    levels (2, 2) the same action in every transition and the same events, at $w$ = 1
    and 30, and on the far-room map as well (9 / 17). Each robot has its own object and
    its own destination, and nothing there can be won from the other: the term shows in
    what a rival is paid, not in what it does. Where something is contested it shows as
    the stand-off of the item above: past the doorway one cell wide two level-2 robots
    without social goals pass (9 / 12), and two rivals do not. A set of pairings in
    which competition has to be told from no social goal by what the robots do needs a
    scene with something to contest.
  - *With an inferred partner goal two rivals do not race, and `"rival"` does worse than
    `"race"`* ("Results (told)", five goals held possible: – / –, 20 / 29, – / –, – / –
    against 9 / 11 at every weight; with the seven of the defaults 36 / 44, 20 / 29,
    – / –, – / –, "Results, nobody told").
    The exception for a cooperative partner makes a rival hostile under a fifth of its
    prior, and it takes an exchanging partner that delivers for it for a cooperator. A
    level-1 rival is never told whom it faces (it races), so a level-2 cooperator that
    has to infer its partner models a racer where it faces a robot that works against
    it; it still delivers in seven runs of eight, and ends taking its partner for a
    conflict robot (0.96 to 1.00 in six of the eight, 0.50 and 0.43 in the other two).
    With inferred goals `"race"` is the reading under which two competitors race.
  - *Against a partner that is out to stop it a level-2 rival takes that partner's
    object*: a hinderer's in four runs of four (to trade it for its own, which no
    hinderer accepts), a conflict robot's in one of four; and against a conflict robot
    neither goal is done in three runs of four.
  - *Before its own goal is done a rival is not indifferent to its partner's.* The term
    charges it $w\rho$ per transition while its partner is done first, so a partner that
    finishes later is worth that much to it. With known partner goals no rival touches a
    rival's object in any run here, also where it is seven transitions behind (the far
    room, 9 / 17), and the two are in each other's way only in the two doorways; the
    term does not rule either out. (With inferred goals two rivals do take each other's
    object: the item above.)
  - *A level-1 rival is told its partner's social goal*, which the table's column
    $l = 1$ does not ask for, and its partner is a level-0 robot in its model all the
    same ("The model").
  - *The account of the two-cycle is of one rollout* ("Why two competitors of the table
    ping-pong": $w = 30$, the six-object scene, greedy actions; the same cycle at
    $w = 100$), and what a robot "plans" and "expects" there is read off its model, not
    isolated by runs. That neither of the table's two entries can be mended alone at
    every weight is two prototypes at four weights on that scene.
- **Under `"table_hostile"` competition is conflict** (with known partner goals; with an
  inferred one it acts as one in the four runs made). The setting does what it is for: a
  level-2 competitor works against a cooperator instead of delivering for it. It does not
  make the competitor want its own goal. Two of them never finish (the rollouts of
  `"table"`, and of conflict / conflict); seated second against a cooperator at $w = 30$
  the competitor takes nothing (the cooperator does not come for its object either);
  against an exchanging partner it gives up its own task to hold its partner's object.
  And a set of pairings that has both `conflict` and `competition` in it holds the same
  robot twice under this setting, unless a coercing robot is in the pairing. A level-2
  partner cannot tell the two apart by watching, under either reading of the table: it
  models both at level 1, where the table gives them one entry. (In three of the four
  runs with an inferred partner goal the cooperator's belief about the competitor ends at
  conflict 0.5 / competition 0.5; in the fourth, competition / cooperation at $w = 30$,
  at "cooperative", 1.00.)
  `"race_then_hinder"` at level 2 also keeps a competitor against a cooperator (5 / – at
  every weight), and two of those finish (9 / 11). Whether an exchanging partner belongs
  in the turned cell is a choice: the table prints it in one cell with the cooperative
  one, and both were turned.
- **At level 2 a competitor of a race form does not wait for its own goal** ("A competitor
  at level 2"). The default form takes a cooperator's object three transitions before its
  own goal is done ($w \le 2$); the robot behind on the far-room map takes the object of
  the one ahead, and in four runs of five neither finishes (levels (2, 2) at $w = 1$:
  12 / 28); two of them stand off past a doorway one cell wide; and against a conflict
  partner neither finishes in three runs of four. The first two it does against a level-1
  partner as well, which it models exactly, and not under `"race"`. The stand-off is
  between two robots at level 2, at least one of them a competitor, each planning against
  a level-1 model of the other that would give way. With an inferred partner goal
  (`partner_social_goal="inferred"`) two level-2 competitors of the default form take each
  other's object and put it back, and neither goal is done:
  at $w = 1$ 3 and 2 put-back cycles in 60 transitions, and one hands the other its object
  at transition 55; at $w = 30$ both objects are back by transition 8 and the two then push
  into each other (25 and 26 blocked moves). The limits below are those of the default
  levels unless they say otherwise.
- **A competitor of the default turns on its partner only once its own goal is done, and
  then there may be nothing left to do.** On the standard map agent_2 is two transitions
  from its destination when agent_1 finishes. With the seats swapped against a cooperator
  ($w \ge 1$) it does not take the cooperator's object: it stands next to it and waits.
  Stepping onto the object's place is 0.002 below staying in its Q, the same at $w$ = 1, 2
  and 30: the social term does not decide there. (The cooperator has 6 m of way left,
  beyond $\delta$ = 5 m, where $r = 0$: taking its object now would take nothing from it.)
- **A cooperator gives up against such a competitor.** In cooperation / competition at
  $w \ge 1$ the level-2 cooperator models the competitor exactly. Once it has delivered the
  competitor's object, no move toward its own is worth its cost: every move is 0.45 below
  staying in the physical part of its Q, the cost of the move and nothing else: it expects
  nothing for its own goal from going there. (The competitor is back at that object's side
  after transition 6.) It stays where it is, three moves in the whole rollout. At
  $w = 0.8$ it does its own task: 27 / 24.
- **Before its own goal is done a competitor of the default holds back by a small margin.**
  With its own object in a hinderer's hands, at $w = 30$, the first step toward the
  hinderer's object is 0.005 below staying in its Q, and the "use" on that object's place
  0.03 (under `"race"` the step is 0.45 below, the cost of a move). Greedy actions keep to
  the better side in every run here. Sampled actions do not: in all three seeds the
  competitor picked the hinderer's object up (transition 12, 7, 12). At weights up to 2
  the transition in which it turns round, and whether it takes a cooperator's object one or
  two transitions after its goal is done, rest on differences in Q of 0.001 to 0.005.
- **A competitor that is helped does nothing for itself.** While its object is carried for
  it, it waits, to transition 10 at $w$ = 1 and 2, where the cooperator tries a wrong
  destination first. It models its partner as pursuing a physical goal, not as a helper.
  (At level 2 it expects the help, and the cooperator delivers at 5 at every weight.)
- **A cooperator at level 1 stalls at small weights** against a level-1 partner, whoever
  it helps. At levels (1, 1) competition / cooperation ends with neither goal done at
  $w$ = 0.8, 1 and 2: five failed set-downs and a hand-over at 35 or 36. A robot without
  social goal in the competitor's place gets the same at $w = 1$, and in cooperation /
  cooperation at levels (1, 1), $w = 1$, the two pick objects up and put them back for the
  whole rollout (51 and 52 put-back cycles in 54 transitions). At $w = 30$ it is 5 / –, as
  with the cooperator at level 2. Against a level-2 partner the level-1 cooperator delivers
  at 5: a competitor of the default form (5 / – at every weight) or a robot without social
  goal (5 / 17 at $w = 1$). (Against a level-2 competitor of `"table_hostile"` at $w = 1$
  it does not: "Results".) The results above are for the default levels.
- **A doorway one cell wide with walled-in approaches.** Two competitors stand off there, as
  two robots without social goals do ($w$ = 1, 30, 100), under both race forms.
- **Conflict against a competitor does not hinder.** Both finish, under both race forms:
  conflict / competition 9 / 11 at $w$ = 1 and 30; competition / conflict 11 / 12 at
  $w = 1$ and 9 / 11 at $w = 30$. (With the competitor at level 2 neither finishes in three
  of these four runs.) The level-2 conflict robot expects to gain by finishing
  first: at $w = 30$ its value is $+1.0 \cdot 10^5$ at the start and $-1.2 \cdot 10^5$
  once the competitor has finished two transitions after it. In its model the partner acts
  at every step by the belief-weighted mix of six destination hypotheses.
- **Hindering at weights up to 1 rests on a reward a robot should not get.** $D$ in $r(g_i)$
  is the way to the robot's object and on to the destination, whatever is in its hands: a
  hinderer with its partner's object in hand that stands on the place of its own keeps 80 % of
  $\rho$ (and so does not finish its own task: it has one pair of hands). With
  `hands_full_reward="none"`, which pays nothing for a goal the robot cannot work on, a
  hinderer has to give its own goal up to hold the object. Against a partner without social
  goal or a competitor it then does its own task at $w$ = 0.8 and 1 (9 / 11) and takes the
  object at $w = 30$; at $w = 2$ nobody finishes and the object is not taken (it stands in
  the doorway against the competitor; against the partner without social goal it waits two
  cells from its own object, and that partner picks the hinderer's object up). Against a
  cooperator, which delivers for it, it hinders from $w = 1$. Competitors are not affected
  much: competition / competition is 9 / 11 under either setting, and competition /
  cooperation 5 / –, 5 / –, 10 / –, 5 / – (at $w = 1$ the cooperator delivers at 5 instead
  of 10); the competitor takes the cooperator's object one or two transitions after its
  own goal is done, as in "Results".
- **Recursive goals against each other are still erratic at level 2.** conflict / conflict,
  and competition / competition with `competition="table"` or `"table_hostile"` (the three
  are one rollout), at $w = 1$: 15 reversed moves and 3 put-back cycles in 53 transitions;
  at $w = 30$ both robots pace. $-\tilde R_j$ pays a robot
  $w^2$ times its partner's *estimate* of its own progress, and it moves to shift that
  estimate ("Why two competitors of the table ping-pong" takes the rollout at $w = 30$
  apart; `"rival"` is the reading under which two competitors do not do this, and two
  conflict robots still do). exchange / exchange at level 2 freezes for $w \le 2$ (neither robot finishes) and
  is done at 5 / 5 for $w = 30$. cooperation / competition with `competition="table"`: 5 / 7
  at $w = 0.8$ and 5 / 5 at $w = 30$, but at $w$ = 1 and 2 the competitor fails four
  set-downs, hands the cooperator its object at 24, and the cooperator has not set it down
  by transition 60.
- **Hand-overs only (`set_down_for_partner=False`) was wrong until 2026-10-07**, in two
  ways. (1) The blocks (object configurations) a model is solved on came from a cache
  whose key held the address of a temporary array. A table could be handed the blocks of
  another table whose array had lived at the same address, and whether it was depended
  on what the process had allocated before: the "not reproducible" of this file's
  earlier version (ten identical runs, three outcomes). (2) A model was solved on the
  blocks its own dynamics reach. A model sets its own robot's object down and nobody
  else's, so with hand-overs only the blocks in which a robot's goal is done were solved
  in that robot's models alone, while its partner's policies are read there as well.
  They had never been computed and were all zero, and to a level-2 robot finishing its
  own goal looked worthless: it stopped one step short of its destination. Neither
  showed under the default rule: a set-down for the partner reaches the blocks the
  partner's own set-down does, so the two robots' models live on the same blocks, and a
  wrong cache entry held the right ones. No result in this file used the option before.
  Every model is now solved on every block any model reaches, and the cache is keyed by
  what a table holds. Under the default rule nothing is changed ("How it is checked").
  With hand-overs only on the six-object scene two level-2 rivals now finish at 9 / 11
  ($w = 30$) and two exchangers at 15 / 12.
- **The exchange scene is one map and one seating, and the exchange on it is inferred
  under one reading of what an exchanger is at level 1** ("An exchange scene"). With the
  copy of the partner's goal not multiplied by the weight (`"copy_unweighted"`), two
  level-2 exchangers that are not told each other's goal exchange at every weight tried
  from 0.55 to 100 with the seven goals of the defaults (from 0.45 with the paper's five,
  from 0.65 with those and Table I's competitor; above 30 beyond the comparisons with
  brute force). Under Table I as printed whoever is helped first stops helping, and both
  goals are done at two of nine weights (one with the paper's five); with the copy times
  the weight, from 0.5 to 10 (with the five and Table I's competitor, to 5). Table I and
  the authors' code give a level-1 exchanger no term at all; the copy is the paper's text,
  read as level 1. Which of the two copies is meant the paper cannot say: at its weight,
  1, they are one model, and the weight is this package's. The copy at its own size does
  not shrink with the weight: a robot with weight 0.001 is modelled with the same copy as
  one with weight 100, and one with weight 0 with none. A robot is still given its
  partner's weight and level. The exchange is lost with agent_1 at level 1, and at $w = 1$
  also with agent_2 at level 1, with the partner's action cost in $\tilde R_j$ (Eq. 2 as
  printed; at 30 / 30 that makes it take 75 / 72 transitions), with
  `hands_full_reward="none"` and with $n = 0.1$. The rooms and the hand-off cell are
  additions of this package to the world, made so that an exchange is the only way to
  finish. Greedy actions, and three seeds of sampled ones at two weights.
- **What is inferred there is that the partner helps, not which helping kind it is.** With
  the paper's five goals, after the exchange at 30 / 30 one robot holds "exchange" at 0.99
  and the other at 0.19, with cooperation at 0.81; in the eight pairings in which both
  goals are done, 9 of the 16 final beliefs put the partner's true goal first, and a
  coercing robot is read as an exchanger every time. With the seven of the defaults the
  two end at 0.03 and 0.71 on "exchange", and 9 of 18 final beliefs of the nine pairings
  with both goals done put the true goal first. The second put-down of an exchange also
  frees the giver's hands for its own object, and a robot makes it while it holds its
  partner for a cooperator. And at weight 30 a belief of 0.003 or less in "exchange" is
  enough for an exchanger to put its partner's object down: seated first it does so for a
  robot without social goal (0.003), in either seat for a hinderer (0.0006, 0.0014), and
  gets nothing back; seated first next to a rival it comes to hold the rival for an
  exchanger (0.99) before it puts down, and the two trade late (48 / 45). (With the
  paper's five goals: 0.03 for the robot without social goal and 0.003 for the rival, and
  nothing back from either.)
- **Sure of each other, two exchangers hold out in a window of weights under the copy at
  its own size.** Told, or with "exchange" the only goal held possible, neither enters
  the doorway first at $w$ = 0.4, 0.5 and 0.6; they exchange from 0.1 to 0.3 and from
  0.7 on. The window does not show with the five goals held possible.
- **Half of a robot's hypotheses about its partner's destination cannot be true on the
  exchange scene.** The six candidates lie three to a room, and a partner cannot reach
  those outside its own; the belief starts uniform over all six all the same.
- **The exchange scene does not single out the exchange goal.** With the seven goals at
  $w = 30$ both goals are done in nine pairings, all through the same doorway: eight
  among cooperation, exchange and coercion (not coercion / coercion) and an exchanger
  with a rival; with the paper's five in the eight among the three; with told goals in
  three (two exchangers, two rivals, a cooperator with a coercing robot).
- **Nobody told, the six-object scene does worse than told: both goals done in 10 of 49
  pairings at $w = 30$, where told robots do 21** ("Results, nobody told"; robots
  without social goals at level 2, the default until 2026-10-08 noon -- at level 0, the
  default now, two of them pass at 9 / 11). Two kinds of rollout account for most of the
  difference.
  - *Two level-2 robots that meet head-on push into each other for good* (five pairings,
    25 blocked moves each): each plans against a model of the other one level down, and a
    push is ahead of staying only if that model may step away; being blocked hardly moves
    a belief (by 0.09 at most in 25 transitions). Told, the five pairings finish at
    9 / 11. Each robot's model of the other has it step aside (0.993 and 0.999 in none /
    none), and each pushes, which neither model explains. The idle hypothesis does not
    resolve it for two robots without social goals, nor do sampled actions in the two
    pairings tried (none / none, competition / competition): once head-on, a step aside
    is below 1e-7 per transition, and with seed 1 both finish only because the robots
    never meet head-on. Under the defaults of now (2026-10-08 noon) it is gone from the
    two of these tried: a physical goal only is level 0 (9 / 11), and with the idle
    hypothesis on two rivals pass (13 / 15). Two level-2 robots without social goals,
    asked for, still stand off.
  - *A first step can make a robot look hostile.* Each robot starts next to its partner's
    object, and a step onto that object's place (for a robot without social goal a near
    tie with another way, 0.22 in Q) leaves its partner holding it for a cooperator, a
    conflict robot or a hinderer at a third each. In seven pairings each then takes the
    other's object, puts it back, and neither goes for its own: each ends holding the
    other for a hostile kind. Without conflict and hinder held possible three of the four
    pairings tried finish (not conflict / none, whose conflict robot is hostile), and so
    do the same three at $w = 1$; at 2, 5 and 10 one run of nine does.
- **Conflict and hinder cannot be told apart.** At level 1 both are $-w\,r(\tilde g_j)$,
  one model; a level-2 robot answers both alike, so nothing is lost in what it does, but
  it holds the two at the same belief whatever it sees, and holding both possible gives
  that model twice the prior of each other one. Which goals are held possible changes
  what robots do from their first move on: the exchange on the exchange scene starts at
  $w = 0.55$ with the seven goals and at 0.45 with the paper's five, and two robots without
  social goals on the six-object scene stand off with the seven and with the paper's five,
  and finish at 9 / 11 with the five and none (2026-10-07) and with the seven but conflict
  and hinder.
- **What a robot is told** where goals are inferred: where its partner stands and what each
  robot holds (always), which object is its partner's, its partner's weight and reasoning
  level, and, through the world, whether a set-down counted (a set-down on a candidate
  that is not the partner's destination fails, and the robot learns from it). What it
  infers is its partner's destination and social goal. A level-1 robot holds no belief
  about its partner's social goal, so a hinderer and a competitor of a race form at level 1
  (their default until 2026-10-11, and under `TOLD`) act on the same models told or not
  (their partners' models of them change). At level 2, their default since, they do not.
- **Two cooperators that are told wait for each other on the exchange scene** (put down
  and pick up, $w$ = 1, 2 and 30). Each expects the other to bring its object: at
  $w = 30$ neither moves, at $w$ = 1 and 2 each takes its partner's object and keeps it.
  That is the model and not the solver's precision: a robot that expects its partner to
  deliver with probability 1.000 has the same social value for every action of its own.
  Unsure of each other's goal they finish (14 / 27 at $w = 30$, 20 / 17 at 1; under
  Table I as printed 14 / 27 and 16 / 15).
- **In a scene with rooms two older options do less than their names say.**
  `held_reward="carrier"` counts the carrier's way to the destination; a carrier in the
  other room has none, so the reward is 0 as under `"none"`. And with set-downs for the
  partner allowed, a partner carrying an object to a destination outside its room is
  predicted to step at random among its legal moves (`partner_delivers`). The exchange
  scene uses neither.
- **Sampled actions lose the hostage** once the partner keeps offering a swap (line 4).
- **Large weights: 30 is as high as the results can be trusted.** Rewards and policies are
  stored in single precision, and a level-2 term of a recursive goal holds both weights:
  at $w = 30$ at both levels values reach $10^5$ and are good to about 7 digits. Against
  brute force on the 3 x 3 map (both solved to $10^{-7}$) the Q of two level-2 conflict
  robots is off by 1.4e-5 at $w = 1$, 0.017 at 10, 0.08 at 30, 0.16 at 50 and 1.5 at 100,
  where a move costs 0.9 more than staying on that map; that of a rival against a
  cooperator by 0.06 at 30 and 0.2 at 100. So at 30 the error is under a tenth of what a
  move costs, and at 100 it can exceed a whole move. The solver's own check does not see
  this: it certifies the problem as stored, not the rounding of its rewards, and its
  residual target is met in all 49 pairings of "Every pairing at $w = 30$" at $w$ = 100,
  300 and 1000 as well. (Solved to $10^{-7}$ instead of the default $10^{-4}$, two conflict
  robots miss the target at 300 and 1000.)

## Layout

| module | what it holds |
|---|---|
| `goals.py` | Table I (as printed, and with one cell turned), `hinder`, the race forms, a rival, weights: the coefficients of every social term |
| `config.py` | every option, with the reason for its default |
| `constants.py` | the ten actions (four moves, stay, pick, place, take, give, swap), object statuses, the outcomes of a robot's step |
| `scenes.py` | `Scene` (with its hand-off cells and rooms), `two_room_scene`, `six_object_scene`, `exchange_scene` |
| `grid.py`, `hands.py`, `world.py` | the grid, the rule for the hands, rewards and dynamics tables |
| `solo.py`, `kernels.py` | numba kernels: solo values, expected rewards, policy iteration |
| `models.py` | the models at levels 0 to 2 and the beliefs (Eqs. 2 to 7) |
| `rollout.py` | Algorithm 1 transition by transition; ledger, trace, events, deadlock |
| `pairing.py` | `run_pairing`, `run_baseline`, `default_levels`, `levels_used` |
| `hindrance.py` | did a robot hinder: effect and conduct, against the baseline |
| `render.py`, `serialize.py` | the annotated video (MP4; or GIF) and single frames, and each robot's value video; JSON |
| `tests/` | the tests; `reference.py`, the independent implementations; `told.py`, the defaults of before 2026-10-08, under which the tests written then run; `hands_before.py`, the rule of the hands until 2026-10-18, the same; `test_objects.py`, which object the partner wants; `test_explicit.py`, nothing automatic |
| `../examples/` | the scripts: `run_all_pairings.py`, `run_pairings.py`, `run_exchange_scene.py`, `compare_competition.py`, `compare_levels.py`, `measure_inference.py` (how well the robots of a folder inferred each other's goals), `tier_scenes.py` (test_tiers' tier scenes on the house at 0.5 m, made fair), `run_tier_scenes.py` (the Social MDP on them) and `run_object_inference.py` (five pairings with the partner's object inferred and known) |
