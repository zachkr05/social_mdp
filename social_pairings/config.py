"""Configuration of the Social MDP: the paper's constants, the world's rules and the solver.

Every option that changes behaviour is a field of :class:`Config`, with the reason for its
default above it.  Constants follow Tejwani et al., "Incorporating Rich Social Interactions
Into MDPs" (ICRA 2022), Sec. IV, stated per *reference step* and rescaled to the grid: one
grid transition lasts ``dt = resolution / reference_step_m`` reference steps, so
``gamma = reference_gamma ** dt``, ``rho = reward_rate * dt`` and ``c(a) = cost_rate * dt``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .goals import COMPETITION_FORMS, COOPERATE_WITH, SOCIAL_GOALS, cooperation_partners
from .goals import parse_goal

__all__ = ["Config", "SolverFailure", "PAPER_GOALS", "HYPOTHESES", "TOLD", "COOPERATE_WITH",
           "EVERY_PARTNER", "ADDED_FIELDS", "AUTOMATIC_HANDS", "written_with"]

# The five social goals of the paper (the rows of Table I).
PAPER_GOALS = ("cooperation", "conflict", "competition", "coercion", "exchange")
# The social goals a robot holds possible for its partner by default: the paper's five, a
# partner without a social goal, and a hinderer.
HYPOTHESES = PAPER_GOALS + ("none", "hinder")
# ``Config.cooperate_with`` as Table I prints the cooperation row: +R_j against every
# partner (the default until 2026-10-11)
EVERY_PARTNER = SOCIAL_GOALS
# Since 2026-10-30 the belief update is Eqs. 5-6 as the authors' solver keeps them (two
# posteriors, the idle filter, Eq. 5's transition term; models.py): the devices of before --
# the lapse, the carry and park tables, the "outcome" observation, the separate inference
# temperature, the joint posterior, the conditioning of the partner's estimate on the
# robot's action -- are gone, and asking for them raises (BELIEF_DEVICES).  Records of
# before carry their fields; written_with reads them.
BELIEF_DEVICES = dict(belief_lapse=0.0, partner_delivers=False, partner_parks=False,
                      partner_observed="action", inference_temperature=None,
                      social_belief="marginals", child_conditioning=False)
# The defaults until 2026-10-08, under which the README's results marked "told" were made:
# each level-2 robot told its partner's social goal, an exchanger at level 1 as Table I prints
# it (no social term), competition read as "race_then_hinder" -- and, as until 2026-10-09,
# each robot knowing which object its partner wants, and as until 2026-10-11, a cooperator
# that cooperates with every partner and a hinderer (and a competitor of a race form) at
# level 1, and as until 2026-10-17, one posterior over (object, destination, social goal)
# updated by every action, and as until 2026-10-18, the automatic hands and their costs
# (AUTOMATIC_HANDS).  ``Config(**TOLD)`` restores them; a result made with another
# reading of competition names it.
# The rule of the hands until 2026-10-18 and its costs (automatic pick-up of a robot's own
# object and set-down at its destination, an instant pick costing a move, the other hand
# actions a move each): ``Config(**AUTOMATIC_HANDS)``
AUTOMATIC_HANDS = dict(automatic_hands=True, pick_steps=1, pick_cost_rate=1.0, use_cost_rate=1.0)
TOLD = dict(partner_social_goal="known", social_hypotheses=PAPER_GOALS, exchange_level1="none",
            competition="race_then_hinder", partner_object="known",
            cooperate_with=EVERY_PARTNER, level_free_level=1, hinder_own_goal=True,
            belief_stages=False, **AUTOMATIC_HANDS)
# (the devices of before, under which every record before 2026-10-30 was made, are not
# restored by TOLD: they are gone -- BELIEF_DEVICES)
# Fields added to Config after records were first written, each with the value every record
# written before it was made with: a record whose "config" lacks the key was made with that
# value (written_with).
ADDED_FIELDS = dict(
    physical_level=2,              # 2026-10-08 noon; until then such a robot was at level 2
    inference_temperature=None,    # 2026-10-09; until then Eq. 7's tau, which None means
    partner_object="known",        # 2026-10-09
    cooperate_with=EVERY_PARTNER,  # 2026-10-11
    level_free_level=1,            # 2026-10-11
    swap_on_place=False,           # 2026-10-12
    steal=False,                   # 2026-10-12
    partner_observed="action",     # 2026-10-13
    hinder_own_goal=True,          # 2026-10-14
    inertia=0.0,                   # 2026-10-14 (evening)
    belief_stages=False,           # 2026-10-17
    value_maps=False,              # 2026-10-17 (evening)
    automatic_hands=True,          # 2026-10-18
    pick_steps=1,                  # 2026-10-18
    pick_cost_rate=1.0,            # 2026-10-18 (c(pick) was use_cost_rate, 1.0 in every record)
    child_conditioning=True,       # 2026-10-30 (the partner's estimate conditioned on the action)
    physical_likelihood="level0",  # 2026-10-30 (a record of before is never that of now anyway)
    robot_temperatures=None,       # 2026-11-01 (until then one tau for both robots)
)


class SolverFailure(RuntimeError):
    """The scene cannot be rolled out: disconnected task cells, overlapping starts, a solo
    task that is never worth finishing, or solver arrays above the memory budget."""


@dataclass(frozen=True)
class Config:
    """Options of one rollout.  Frozen: build a variant with ``dataclasses.replace``."""

    # ---- time, rewards and costs (paper constants per reference step) --------------------
    # metres per grid cell, and metres per grid step of the paper
    resolution: float = 0.5
    reference_step_m: float = 1.0
    # discount per paper step
    reference_gamma: float = 0.99
    # r = rho * max(1 - D / delta, 0): rho, and delta in metres
    reward_rate: float = 1.25
    reward_radius_m: float = 5.0
    # c(move) and c(stay)
    move_cost_rate: float = 1.0
    wait_cost_rate: float = 0.1
    # c(place), c(take), c(give), c(swap): the hand actions other than pick (since 2026-10-16
    # five hand actions in place of the one action "use").  2 moves' worth since 2026-10-18
    # (the user: "any sort of action besides movements should therefore cost 2"); 1 until
    # then -- like a move, so that picking up and putting back for nothing never tied with
    # waiting.
    use_cost_rate: float = 2.0
    # ---- the hands, since 2026-10-18 (the user: "nothing is automatic ... you must move to
    # the cell then select the pickup dropoff etc action"; "make the pickup cost 3 and take 3
    # timesteps ... 3 at the start then 0 and 0") --------------------------------------------
    # automatic_hands: True -- the rule until 2026-10-18: a robot with free hands takes its OWN
    # object on arriving at the cell where it lies, and holding its own object sets it down on
    # arriving at its destination, both without a hand action and for free (its partner's
    # object only by the hand actions).  False -- nothing is automatic: every object is picked
    # up by "pick" and set down by "place", the robot's own included, and a robot finishes
    # by placing its object on its destination (world.py: "Nothing is automatic").
    automatic_hands: bool = False
    # pick_steps: transitions a pick-up takes -- the robot is busy lifting for that long (only
    # "pick" is open to it; the object is in nobody's hands yet; its partner can neither take
    # it nor hand it anything); pick_cost_rate: c(pick) in moves' worth, charged on the first
    # of those transitions, the rest are free.  1 and 1.0 until 2026-10-18 (pick_steps > 1
    # needs automatic_hands=False).
    pick_steps: int = 3
    pick_cost_rate: float = 3.0
    # factor on c(move) while holding an object that is not the robot's own
    foreign_carry_cost: float = 2.0
    # Does a hinderer have a physical goal of its own?  False (since 2026-10-14, the user:
    # "a hindering agent would just want to minimize the opposing agent's task, they don't
    # really care about their task"): its reward is -c(a) - w r(g_j) -- the move costs and
    # the partner's goal reward, nothing for its own object (goals.py: "hinder").  True, as
    # until then: r(g_i) - c(a) - w r(g_j), its own goal weighed against the partner's.
    hinder_own_goal: bool = False
    # minimum distance between the robots' centres; 0 = one robot per cell
    robot_separation_m: float = 0.0

    # ---- policies -------------------------------------------------------------------------
    # tau of Eq. 7, the softmax policy a robot predicts its partner by.  "move_cost": tau =
    # temperature * c(move); "absolute": tau = temperature in reward units
    temperature: float = 0.2
    temperature_unit: str = "move_cost"
    # A tau per robot, (agent_1's, agent_2's), in the unit of ``temperature`` (2026-11-01; the
    # user: "each agent to have its own tau").  Robot i's tau is how it reads its partner:
    # the softmax it predicts the partner by (Eq. 7) and the likelihood of its belief update
    # about the partner (Eqs. 5-6).  The robots know each other's, as they know the rest of
    # the model: inside robot i's model the partner's level-1 models read robot i at the
    # partner's tau (so every model of a robot is softmaxed at its partner's tau, and the
    # posterior about it -- which robot i's model takes as the partner's belief about it --
    # is updated at its partner's tau).  Under action_selection="sample" robot i draws its
    # own actions at its own tau.  None: both robots at ``temperature``; two equal values
    # are stored that way (``temperature`` = the value, this field None), so a record made
    # with one tau for both is the same record whichever way it was asked for.
    robot_temperatures: Optional[Tuple[float, float]] = None
    # (Until 2026-10-30 the belief update had a temperature of its own and could read a
    # blocked move as a wait: both are gone -- the likelihood of Eqs. 5-6 is Eq. 7's softmax
    # of the hypothesis model's Q at Eq. 7's tau, of the action the partner chose, as
    # Algorithm 1 takes the joint action.  Kept for the records of before: BELIEF_DEVICES.)
    inference_temperature: Optional[float] = None
    partner_observed: str = "action"
    # How the robots themselves act.  "argmax": Algorithm 1's greedy choice.  "sample": a draw
    # from softmax(Q / tau).  Greedy is the default because a sampled robot keeps a ~1 %
    # chance per step of any affordable action -- enough, over a long stand-off, to hand
    # back an object it means to keep.
    action_selection: str = "argmax"
    # RNG seed of "sample"
    seed: int = 0
    # Inertia (added 2026-10-14; the user: keep the previous action unless another beats it
    # by about one move cost).  A robot does not undo what it just did -- step back to the
    # cell it came from, place what it has just picked up, pick up what it has just placed,
    # give back what it has just taken, take back what it has just given, swap back -- unless
    # undoing beats every other action by more than inertia x c(move).  Under "sample" the
    # undoing action is drawn with its Q lowered by that much.  0: none, as until then.  Only
    # where the robots act: the models by which they predict each other are Eq. 7's,
    # without it.
    # Keeping the previous action literally (waiting on after a wait) is not offered: in many
    # states moving beats waiting by a fraction of a move's cost (0.14 for agent_1 at
    # transition 5 of none / hinder, 2026-10-14), and a robot that had waited would not set
    # off again.
    inertia: float = 0.0

    # ---- levels -------------------------------------------------------------------------------
    # The level a robot with a physical goal only -- no social goal, or weight 0 -- reasons at
    # when no level is asked for (pairing.default_levels).  0, the default since 2026-10-08
    # (the user: "physical goal only ... would mean the level = 0. We have no social stance
    # and no goal inferred"): a plain MDP that takes its partner for a moving obstacle and
    # infers nothing.  2: the default until then -- such a robot predicted its partner by the
    # partner's level-1 models -- under which the README's results with a robot without
    # social goal at level 2 were made.  A level that is asked for is used either way.
    physical_level: int = 0
    # The default level of a robot whose social term is on goal rewards alone -- a hinderer,
    # and a competitor under competition="race" or "race_then_hinder" (goals.level_free) --
    # and the most a hinderer may be asked for.  2 since 2026-10-11 (the user: "Whenever I
    # say run it, always assume both are level 2"): every robot with a social goal reasons at
    # level 2 when no level is asked for, and a level that is asked for is used.  1 until then
    # (TOLD): such a robot reasoned at level 1 by default, and a hinderer was held to level 1
    # whatever was asked (README, "What hindering rests on", line 3: rolled out at level 2
    # against Table I's competitor, told its partner's goal, it never took the object).
    level_free_level: int = 2

    # ---- rules of the shared-object world -------------------------------------------------
    # (Robot i's goal is done by object i on robot i's destination and by nothing else: that
    # is the world, not an option.)
    # objects in the model (the two own ones first), 2..6; the others are scenery (drawn,
    # never moved).  With the partner's object inferred every object in the model is a
    # candidate for the partner's goal, and the model grows fast with them: on the two-room
    # map with six objects about 13 GiB (326,144 joint states, 432 level-1 models), with
    # three 0.9 GiB.
    max_objects: int = 6
    # A social robot may set its partner's object down on a candidate destination (it counts
    # only on the partner's real one).  False: it cannot finish its partner's task for it --
    # the object gets to its owner by a hand-over, or over a hand-off cell of the scene.
    set_down_for_partner: bool = True
    # A social robot may hand its partner's object straight to the partner standing next to
    # it.  False: nothing goes from hand to hand -- an object changes hands only by being put
    # down (on its place, or on a hand-off cell of the scene: Scene.handoff_cells) and
    # picked up; forced_swap then has nothing to act on.
    hand_over: bool = True
    # May a robot pick up an object that completes NOBODY's goal (a spare one)?  It
    # gains nothing by it -- except, at level 2, a feint: with its hands full it looks unable
    # to take its own object, and a partner that models it one level down lets its guard
    # down.  False: such objects are scenery (and the model is ~20x smaller with four).
    # Only with partner_object="known": where the partner's object is inferred any object
    # may be the partner's, and every object is handled.
    handle_spares: bool = False
    # False: nobody is made to let go of what it holds -- two objects change hands only when
    # BOTH robots give in the same transition.  True: a robot handed its own object takes it
    # with full hands and gives up what it held, so one robot's "give" swaps the two; a
    # partner can then always take back an object held against it, and nothing can be
    # hindered.
    forced_swap: bool = False
    # A robot holding an object that stands on the place of another object -- where that
    # object lies, or one set down there in a swap -- swaps them with "swap":
    # what it held stays on that cell, the other comes into its hands (its own object back
    # onto its own place is the put-back of before).  Added 2026-10-12 (the user, of coercion
    # against hinder: "the agent should have dropped the red to get his orange block").
    # False, the rule of before: a held object goes back only to its own place, and full
    # hands pick up nothing.  An object may then lie on the place of another (one per place),
    # which multiplies the object configurations: x1.5 with the two own objects, x3.6 with a
    # spare one, x12.5 with two.
    swap_on_place: bool = False
    # A robot with free hands next to its partner (after both moved) takes whatever the
    # partner holds with "take" -- unless the partner, in the same transition, places that
    # object, gives it or swaps it: the holder's own act comes first.  Added 2026-10-12 (the
    # user: "lets make it such that agents can steal from each other").  False, the rule of
    # before: nobody is made to let go of what it holds.
    steal: bool = False
    # a robot whose goal is done still has its hand actions
    done_robots_handle: bool = True
    # Reward of a robot's goal while ITS OWN hands hold an object that is not its own.
    # "ignored": r(D) as if its hands were free -- D is the way to its object and on to the
    # destination, so a robot standing next to its object with its partner's object in hand
    # keeps most of its goal reward.  "none": 0 -- it cannot work on its goal until its hands
    # are free.  Hindering at weights up to about 1 rests on "ignored" (README, "Limits").
    # (With held_reward="carrier" the reward of a robot whose object its partner carries
    # follows that object, whatever the robot's own hands hold.)
    hands_full_reward: str = "ignored"
    # Reward of a goal whose object is in the PARTNER's hands.  "none": 0 -- nothing its
    # robot can do until the object is set down or handed over (taking an object is then
    # what hindering looks like).  "carrier": r(D) with D the carrier's way to the
    # destination, so a helper earns as it carries -- and holding an object no longer
    # denies its owner anything (in the runs a hinderer then does not take it).
    held_reward: str = "none"

    # ---- social goals (Table I) -----------------------------------------------------------
    # Competition, five readings (goals.py).  The first two are terms on goal rewards, the
    # same against every partner and at every level; a competitor's default level is then 1
    # (pairing.default_levels), and 2 if asked for.
    # "race_then_hinder": the robot looks after its own goal (and loses -r(g_j) while its
    #   partner is done first); once its own goal is done it is paid 2 r(g_i) - r(g_j): it
    #   works against its partner -- takes its object, stands in its way.
    # "race": the same until its own goal is done; from then on +r(g_i) and nothing about
    #   its partner: it never helps, and it stops.
    # "table": Table I as printed.  At l = 1 that is the conflict entry -r(g_j): two such
    #   robots take each other's object and neither goal is ever done.  Against a
    #   cooperative partner it is +R_j: the competitor delivers its partner's object.
    # "table_hostile": the table with that cell turned, -R_j against a cooperative (or
    #   exchanging) partner as well: a level-2 competitor works against the reward of a
    #   partner that would help it, too.  Against a cooperator it is a conflict robot.
    # "rival": "I want to do better than you", not "I want you to fail".  A rival races
    #   every partner -- the term of "race", at every level -- but a cooperative one, which
    #   helps it and against which it is the competitor of "table_hostile".  Two rivals each
    #   do their own task, where two competitors of either reading of the table at level 2
    #   never get either goal done.  Default level: 2.  Where goals are inferred nobody tells
    #   a rival whom it faces: at level 2 it weighs the two terms by its belief about its
    #   partner's social goal (the hostile one by its belief in a cooperator), at level 1 it
    #   cannot tell and races.
    # The default since 2026-10-08 (the scripts' since 2026-10-06); "race_then_hinder" until
    # then (TOLD).
    competition: str = "rival"
    # Whom a level-2 cooperator cooperates with: the partners' social goals against which
    # its cell of Table I's column l > 1, +R_j, holds; against any other partner it has no
    # social term and minds its own goal (goals.py, "cooperation").  As printed the cell
    # holds against every partner, and R_j holds the partner's own social term: against a
    # hinderer or a conflict partner the cooperator takes on that partner's wish that it
    # fail, and at w = 1 a hinderer sure of its goal makes the goal worth nothing to it -- on
    # the center scene it stopped two moves from its destination (2026-10-10).  Since
    # 2026-10-11 (the user: "lets modify our algorithm to only cooperate with competitive,
    # cooperative, and exchange agents") those three: COOPERATE_WITH.  With an inferred
    # partner goal the cell is weighted by the belief, so the help fades as the cooperator
    # becomes sure its partner is one of the others (at the uniform prior over seven goals
    # it is 3/7 of the table's).  "all" (EVERY_PARTNER) is the table as printed, the
    # default until then (TOLD).  Nothing changes at level 1: a level-1 robot holds no
    # belief about its partner's social goal.
    cooperate_with: tuple = COOPERATE_WITH
    # n of the exchange row, "positive, but small"
    exchange_share: float = 0.25
    # An exchange goal at level 1, which is also how a level-2 robot models an exchanging
    # partner.  "none": Table I's N/A, no social term -- a level-1 exchanger does nothing for
    # its partner, so a level-2 robot that has to INFER its partner's social goal reads a
    # partner that helps it as a cooperator, and whoever is helped first stops helping.
    # "copy": the paper's text, a copy of the partner's goal with the small scalar n --
    # n r(g_j) as the social term, times the weight like every entry of the table.
    # "copy_unweighted": the same copy of the size n whatever the weight (at weight 1, the
    # paper's, the two are the same; a robot with weight 0 has none); with it a partner that
    # helps may be an exchanger at every weight, where the weighted copy of a robot with a
    # large weight is a helper so eager that its partner waits for it.  Where goals are
    # inferred the option changes one of the models every level-2 robot judges its partner
    # by, whatever its own goal.
    # README, "An exchange scene".  The default since 2026-10-08, when inferring partner
    # goals became the default; "none" until then (TOLD).
    exchange_level1: str = "copy_unweighted"
    # Does the partner's reward R_j inside a level-2 social term hold the partner's action
    # cost?  Eq. 2 as printed says yes; with it a robot whose term holds -R_j (conflict,
    # the table's competition) is paid +w c for every step its partner takes: an incentive
    # to keep the partner walking rather than to stop it.
    social_cost: bool = False
    # What a level-2 robot knows of its partner's social goal.  "inferred" (the default since
    # 2026-10-08): nothing -- a uniform prior over social_hypotheses, updated by Eq. 5 from
    # what the partner does.  A robot sees where its partner stands and what it holds; its
    # partner's destination (always) and social goal it has to infer.  "known" (the default
    # until then, TOLD): the goal its partner was given, as it counts (none for a partner at
    # level 0 or with weight 0; the destination stays hidden).  (A level-1 robot does not
    # model its partner's social goal; under competition="rival" it is told it when it is
    # "known": goals.py.)  Either way a robot is given its partner's weight and level.
    partner_social_goal: str = "inferred"
    # The social goals held possible ("inferred" only), each with the same prior: the
    # paper's five, a partner without a social goal, and a hinderer.  A hinderer at level 1
    # is the same model as a conflict partner at level 1 (Table I's -r(g_j)), and a level-2
    # robot answers both alike, so the two cannot be told apart: holding both possible
    # gives that model twice the prior of each other one.
    social_hypotheses: tuple = HYPOTHESES
    # (Until 2026-10-30 one joint posterior over (destination, social goal) could be
    # planned with, or the product of its marginals.  Since then there are two posteriors,
    # as the authors' solver keeps them, and Eq. 4 takes their product: "marginals" is the
    # one value -- models.py.  Kept for the records of before: BELIEF_DEVICES.)
    social_belief: str = "marginals"
    # Which object a robot's partner wants.  "inferred" (the default since 2026-10-09, the
    # user: "Modify it such that the pickup objects are always inferred as per the paper"):
    # a physical goal is an object and a destination ("Physical goals consist of moving the
    # desired objects to one of the landmarks"), and a robot's estimate of its partner's
    # starts from a uniform distribution over all of them (Sec. III-B, Eq. 6) -- every
    # object of the scene, the robot's own included, with every candidate destination.  So
    # every object is in the model and may be picked up (spare objects are candidates too:
    # handle_spares does not matter), and Table I's "g_i = g_j" asks for the same object AND
    # the same destination.  "known" (until then): each robot knows that its partner wants
    # the partner's own object; it infers only the destination ("g_i = g_j": the same
    # destination).  A robot always knows its own physical goal.
    partner_object: str = "inferred"

    # ---- the model of the partner and the beliefs (since 2026-10-30: Eqs. 5-6 as the
    # authors' solver keeps them -- two posteriors, physical from the partner's level-0
    # policies and social from its level-1 hypothesis policies, the idle filter, Eq. 5's
    # transition term; models.py) ----------------------------------------------------------
    # Also track "the partner stays put whatever its goal" (a two-mode filter on top of Eqs.
    # 5-6), in every robot that models its partner.  On by default since 2026-10-08 (the
    # user: "allow both to be able to model their counter part as being idle to fix the
    # stationary / deadlocked agents"); kept on 2026-10-30 as the one device beside Eqs. 5-6
    # (the user).  An idle partner is predicted to stay put (idle_eps); the per-step switch
    # probability active <-> idle is idle_hazard.
    idle_hypothesis: bool = True
    idle_hazard: float = 0.02
    idle_eps: float = 0.0
    # The likelihood of Eq. 6, by which a robot updates its belief about its partner's
    # physical goal.  "level1" (the default): the partner's level-1 hypothesis policies
    # averaged over the social posterior, sum_xi b(xi) psi^1_(g,xi)(a | s) -- Eq. 6 as the
    # paper prints it (the integral over xi).  "level0": the partner's level-0 policies,
    # psi^0_g(a | s) -- the authors' code (its physical-goal estimator runs on the level-0
    # policies).  The two part where the partner acts for the robot: a level-0 model has no
    # social goal, so under "level0" a helping move -- a step toward the robot's object, a
    # detour -- is read as evidence about the partner's own goal (the hypotheses under which
    # the move is least unlikely gain: the far destinations, whose discounted reward makes
    # the policy flattest), where under "level1" the social hypotheses that predict the
    # move explain it and the physical posterior keeps still.  Two cooperators at w = 30 on
    # the two-room scene, told each other's goal (tests/test_render.StoryTests): under
    # "level1" each delivers the other's object at transition 5; under "level0" one of them,
    # its belief about its partner's destination pushed off by the partner's first step
    # toward its object, sets the object down on the wrong candidate.  A level-1 robot,
    # which holds no social posterior, uses the level-0 policies either way.  (Added
    # 2026-10-30 with "level0" as the default; "level1" since the same evening.)
    physical_likelihood: str = "level1"
    # Eq. 5's transition term T(s^t | s^{t-1}, a_J): the outcomes of the hands that depend on
    # the partner's hidden goal -- a set-down for the partner that counts (the partner's goal
    # is done) or fails, a hand-over the partner takes or not, the partner's own set-down
    # that finishes or does nothing -- rule out the hypotheses that predicted otherwise
    # (models.SocialMDP.transition_term).  False: the actions alone are evidence.
    condition_on_placement: bool = True
    # Two posteriors, by the partner's hands (since 2026-10-17; the user: "as we move the
    # prior for the goal is being accrued even though we haven't picked up an object").
    # While the partner holds nothing, what it does is evidence about which object it wants
    # (and about its social goal) and none about its destination; while it holds an object,
    # evidence about its destination (and its social goal) and none about its object, whose
    # belief stays where the pick-up left it.  Eq. 6 then runs with the likelihood of each
    # hypothesis averaged, under the belief as it is, over the factor the state says nothing
    # about (models.SocialMDP.staged_log_likelihood); the same for the belief the partner's
    # level-1 models hold about the robot, by the robot's hands.  False: Eq. 6 with the
    # likelihood of each (object, destination) hypothesis as it is, as until 2026-10-17.
    belief_stages: bool = True
    # (The devices of before, gone since 2026-10-30 and kept only so that the records of
    # before can be read -- BELIEF_DEVICES, written_with.  Asking for any of them raises.)
    # a lapse (1 - e) psi_k + e uniform in the likelihood
    belief_lapse: float = 0.0
    # a partner holding the robot's object predicted by a carry table (not by its models)
    partner_delivers: bool = False
    # a partner without a social goal predicted to park once its goal is done
    partner_parks: bool = False
    park_eps: float = 0.02
    park_prior: float = 0.95
    # the partner's estimate of the robot's goal reward inside Table I's R_j conditioned on
    # the robot's own action (Bayes within the reward) -- the authors' code takes the
    # estimate as it stands
    child_conditioning: bool = False

    # ---- what the record holds besides the rollout ----------------------------------------
    # Each robot's value at every cell it could stand on, term by term, at every transition
    # (models.SocialMDP.value_maps; since 2026-10-17, for the value videos of render.py):
    # about a second a robot a transition on the two-room scene.  False: left out
    value_maps: bool = True

    # ---- rollout --------------------------------------------------------------------------
    # stop once both goals are done, after done_tail more transitions
    stop_when_done: bool = True
    done_tail: int = 1
    # stop after deadlock_window transitions without progress
    deadlock_stop: bool = True
    deadlock_window: int = 25
    # also stop when nobody has beaten its best remaining distance for this long; 0 = off
    livelock_window: int = 50

    # ---- solver ---------------------------------------------------------------------------
    # 0 = every CPU this process may use
    num_threads: int = 0
    # certified bound on |V - V*| of every solve
    value_tol: float = 1e-4
    # inexact policy iteration: evaluate each policy to this fraction of its residual
    pi_forcing: float = 0.1
    max_pi_iters: int = 200
    max_krylov_iters: int = 3000
    # also solve each social robot without its social term (the "physical Q" of the traces);
    # False: ~1/3 faster
    physical_twin: bool = True
    # extrapolated warm starts between transitions (exact: every solve still runs to its
    # certificate)
    extrapolate: bool = True
    # memory the solver may use; it never plans with more than this machine has (or the
    # limit of its control group), and a model that cannot fit fails at once (SolverFailure)
    max_memory_gib: float = 48.0

    # ---- logging --------------------------------------------------------------------------
    verbose: bool = False
    # a status line per robot every k transitions (0: events only)
    debug_every: int = 0
    # invariant checks on every transition
    debug_checks: bool = True
    # hypotheses listed in belief summaries
    debug_topk: int = 3

    def __post_init__(self) -> None:
        if self.robot_temperatures is not None:
            pair = self.robot_temperatures
            if isinstance(pair, (str, bytes)) or not hasattr(pair, "__len__") or len(pair) != 2:
                raise ValueError("robot_temperatures must be None or (agent_1's tau, agent_2's)")
            pair = tuple(float(t) for t in pair)
            if not all(np.isfinite(t) and t > 0.0 for t in pair):
                raise ValueError("robot_temperatures must be positive and finite")
            if pair[0] == pair[1]:           # one tau for both: stored as ``temperature``
                object.__setattr__(self, "temperature", pair[0])
                pair = None
            object.__setattr__(self, "robot_temperatures", pair)
        positive = (self.resolution, self.reference_step_m, self.reward_rate,
                    self.reward_radius_m, self.move_cost_rate, self.wait_cost_rate,
                    self.temperature, self.value_tol, self.max_memory_gib)
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError("Config scales must be positive and finite")
        if not 0.0 < self.reference_gamma < 1.0:
            raise ValueError("reference_gamma must lie in (0, 1)")
        if self.move_cost_rate < self.wait_cost_rate:
            raise ValueError("move_cost_rate must be >= wait_cost_rate")
        if not np.isfinite(self.use_cost_rate) or self.use_cost_rate < self.wait_cost_rate:
            raise ValueError("use_cost_rate must be finite and >= wait_cost_rate")
        if not np.isfinite(self.pick_cost_rate) or self.pick_cost_rate < self.wait_cost_rate:
            raise ValueError("pick_cost_rate must be finite and >= wait_cost_rate")
        if not isinstance(self.automatic_hands, (bool, np.bool_)):
            raise ValueError("automatic_hands must be True or False")
        if (isinstance(self.pick_steps, bool) or not isinstance(self.pick_steps, (int, np.integer))
                or not 1 <= self.pick_steps <= 9):
            raise ValueError("pick_steps must be an int in 1..9")
        if self.pick_steps > 1 and self.automatic_hands:
            raise ValueError("pick_steps > 1 needs automatic_hands=False (an automatic pick-up "
                             "is instant)")
        if not np.isfinite(self.foreign_carry_cost) or self.foreign_carry_cost < 1.0:
            raise ValueError("foreign_carry_cost must be finite and >= 1")
        if not np.isfinite(self.robot_separation_m) or self.robot_separation_m < 0:
            raise ValueError("robot_separation_m must be finite and >= 0")
        _one_of(self, "temperature_unit", ("move_cost", "absolute"))
        _one_of(self, "action_selection", ("argmax", "sample"))
        _one_of(self, "held_reward", ("none", "carrier"))
        _one_of(self, "hands_full_reward", ("ignored", "none"))
        _one_of(self, "competition", COMPETITION_FORMS)
        _one_of(self, "exchange_level1", ("none", "copy", "copy_unweighted"))
        _one_of(self, "partner_social_goal", ("known", "inferred"))
        _one_of(self, "partner_object", ("known", "inferred"))
        _one_of(self, "social_belief", ("marginals",))
        _one_of(self, "physical_likelihood", ("level0", "level1"))
        _one_of(self, "partner_observed", ("action",))
        for name, value in BELIEF_DEVICES.items():   # the devices of before are gone
            if getattr(self, name) != value:
                raise ValueError(f"{name}={getattr(self, name)!r}: the belief update is Eqs. 5-6 "
                                 f"as the authors' solver keeps them since 2026-10-30, and the "
                                 f"devices of before are gone ({name} must be {value!r}; "
                                 f"models.py)")
        if not isinstance(self.hinder_own_goal, (bool, np.bool_)):
            raise ValueError("hinder_own_goal must be True or False")
        if not isinstance(self.belief_stages, (bool, np.bool_)):
            raise ValueError("belief_stages must be True or False")
        if not isinstance(self.value_maps, (bool, np.bool_)):
            raise ValueError("value_maps must be True or False")
        if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an int")
        if (isinstance(self.inertia, (bool, np.bool_))
                or not isinstance(self.inertia, (int, float, np.integer, np.floating))
                or not np.isfinite(self.inertia) or self.inertia < 0.0):
            raise ValueError("inertia must be a finite number >= 0 (in moves' costs)")
        if self.inference_temperature is not None and not (
                np.isfinite(self.inference_temperature) and self.inference_temperature > 0.0):
            raise ValueError("inference_temperature must be None or finite and > 0")
        if isinstance(self.physical_level, bool) or self.physical_level not in (0, 1, 2):
            raise ValueError("physical_level must be 0, 1 or 2")
        if isinstance(self.level_free_level, bool) or self.level_free_level not in (1, 2):
            raise ValueError("level_free_level must be 1 or 2")
        if (isinstance(self.max_objects, bool) or not isinstance(self.max_objects, int)
                or not 2 <= self.max_objects <= 6):
            raise ValueError("max_objects must be an int in 2..6")
        if not np.isfinite(self.exchange_share) or self.exchange_share < 0.0:
            raise ValueError("exchange_share must be finite and >= 0")
        object.__setattr__(self, "social_hypotheses", tuple(self.social_hypotheses))
        if not self.social_hypotheses:
            raise ValueError("social_hypotheses must name at least one social goal")
        for name in self.social_hypotheses:
            parse_goal(name)
        if self.cooperate_with is None or isinstance(self.cooperate_with, bool):
            raise ValueError("cooperate_with: name the partners' social goals, or 'all'")
        # the names, each once, in the order of SOCIAL_GOALS ("all": every one)
        partners = sorted(cooperation_partners(self.cooperate_with))
        object.__setattr__(self, "cooperate_with", tuple(SOCIAL_GOALS[g] for g in partners))
        if not 0.0 <= self.belief_lapse < 1.0:
            raise ValueError("belief_lapse must lie in [0, 1)")
        for name in ("park_eps", "idle_eps"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")
        if not 0.0 <= self.park_prior <= 1.0:
            raise ValueError("park_prior must lie in [0, 1]")
        if not 0.0 <= self.idle_hazard <= 0.5:
            raise ValueError("idle_hazard must lie in [0, 0.5]")
        if not 0.0 <= self.pi_forcing < 1.0:
            raise ValueError("pi_forcing must lie in [0, 1)")
        for name, low in (("done_tail", 0), ("num_threads", 0), ("debug_every", 0),
                          ("livelock_window", 0), ("deadlock_window", 1), ("max_pi_iters", 1),
                          ("max_krylov_iters", 1), ("debug_topk", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"{name} must be an int >= {low}")

    # ---- derived constants, per grid transition -------------------------------------------
    @property
    def dt(self) -> float:
        """Reference steps per grid transition."""
        return self.resolution / self.reference_step_m

    @property
    def gamma(self) -> float:
        return self.reference_gamma ** self.dt

    @property
    def rho(self) -> float:
        return self.reward_rate * self.dt

    @property
    def costs(self) -> np.ndarray:
        """(5,) cost of the four moves and of "stay"."""
        return np.array([self.move_cost_rate] * 4 + [self.wait_cost_rate], dtype=float) * self.dt

    @property
    def tau(self) -> float:
        """Temperature of Eq. 7 in reward units, where both robots have the same
        (``robot_temperatures`` None); with a tau per robot, :meth:`tau_of`."""
        if self.robot_temperatures is not None:
            raise ValueError("the robots have a tau each (robot_temperatures): use tau_of(i)")
        return self.tau_of(0)

    def temperature_of(self, i: int) -> float:
        """Robot i's tau in the unit of ``temperature`` (robot 0 = agent_1)."""
        if self.robot_temperatures is None:
            return float(self.temperature)
        return float(self.robot_temperatures[i])

    def tau_of(self, i: int) -> float:
        """Robot i's tau of Eq. 7 in reward units: how robot i reads its partner -- the
        softmax it predicts the partner by and the likelihood of its belief update."""
        scale = float(self.costs[0]) if self.temperature_unit == "move_cost" else 1.0
        return self.temperature_of(i) * scale

    @property
    def tau_inference(self) -> float:
        """Temperature of the likelihood of Eqs. 5-6 in reward units."""
        if self.inference_temperature is None:
            return self.tau
        scale = float(self.costs[0]) if self.temperature_unit == "move_cost" else 1.0
        return float(self.inference_temperature) * scale

    @property
    def inertia_margin(self) -> float:
        """By how much undoing the previous action must beat every other action to be
        chosen, in reward units (``inertia`` x c(move))."""
        return float(self.inertia) * float(self.costs[0])

    @property
    def separation_cells(self) -> float:
        return max(1.0, self.robot_separation_m / self.resolution)

    @property
    def residual_tol(self) -> float:
        """Bellman-residual target: max|TV - V| <= (1 - gamma) value_tol bounds |V - V*| by
        value_tol."""
        return (1.0 - self.gamma) * self.value_tol


def _one_of(cfg: Config, name: str, allowed: tuple) -> None:
    if getattr(cfg, name) not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {getattr(cfg, name)!r}")


def written_with(config: dict) -> dict:
    """The options a record was made with: its ``config`` (as saved, JSON form) with every
    field of :data:`ADDED_FIELDS` that the record predates filled in as it was then.  It
    compares field by field with ``serialize.to_json(dataclasses.asdict(cfg))``, which is how
    the example scripts' ``--resume`` decides whether a record in the folder was made with
    the options asked for."""
    out = dict(config)
    for key, value in ADDED_FIELDS.items():
        out.setdefault(key, list(value) if isinstance(value, tuple) else value)
    return out
