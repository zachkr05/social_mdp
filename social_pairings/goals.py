"""Social goals: what each one adds to a robot's reward.

A robot i with physical goal g_i and a social goal toward its partner j earns, per
transition (Eq. 2 of Tejwani et al., ICRA 2022),

    R_i = r(g_i) - c(a_i) + w_i * xi_ij

where ``w_i`` is the robot's social weight (1 reproduces the paper) and ``xi_ij`` is the row
of Table I for its social goal.  Every row is a combination of three quantities:

    xi_ij = alpha * R_j  +  beta * r(g_i)  +  nu * r(g_j)

``r(g_j)``  the partner's goal reward (expected over the robot's belief about the
            partner's hidden destination);
``r(g_i)``  the robot's own goal reward;
``R_j``     the partner's own reward one reasoning level down, which holds the partner's
            social term in turn.  It exists only above level 1.

Table I, column "l = 1" (the partner is modelled with physical goals only)::

    cooperation   +r(g_j)          conflict      -r(g_j)
    competition   -r(g_j)          coercion      +r(g_i)
    exchange      N/A  (no social term; ``Config.exchange_level1``: the text's copy of
                  the partner's goal, n r(g_j) -- "copy": times the weight like every entry
                  here, "copy_unweighted": of the size n whatever the weight)

Table I, column "l > 1" (the partner is modelled one level down with social goal t)::

    cooperation   +R_j                      conflict   -R_j
    competition   +R_j vs cooperation, exchange;  -R_j vs conflict, competition;
                  vs coercion: +R_j if g_i = g_j, else -R_j
    coercion      +R_j + r(g_i) vs cooperation, exchange;  -R_j + r(g_i) vs conflict,
                  coercion;  vs competition: +R_j if g_i = g_j, else -R_j + r(g_i)
    exchange      n r(g_j) + R_j vs exchange;  N/A vs anything else

``hinder`` is the one goal that is not a row of the table.  It is the table's conflict entry
at l = 1, ``-r(g_j)``, kept at every level: the robot is paid for keeping its partner's
physical goal out of reach and for nothing else.  Since 2026-10-14 it has no physical goal of
its own either (``Config.hinder_own_goal=False``): its whole reward is ``-c(a_i) - w_i r(g_j)``.
In the terms below that is ``b = -1`` (``beta = -1`` above level 1), NOT multiplied by the
weight: it takes away the ``r(g_i)`` of Eq. 2 exactly, at every weight.  With
``hinder_own_goal=True`` (the rule until then) it keeps ``r(g_i)`` and weighs it against its
partner's goal.  Why it exists next to ``conflict``: above
level 1 the table's conflict is ``-R_j``, and R_j holds the partner's social term in turn,

    R_i = r(g_i) - w_i r(g_j) - w_i w_j xi_ji - c(a_i).

Against a hostile partner ``xi_ji = -r~(g_i)``, the partner's *estimate* of the robot's own
goal reward: the robot is paid ``w_i w_j`` times for looking, to its partner, as if it were
succeeding, and it moves to shift that estimate -- in the runs two level-2 robots in
conflict, or in competition as the table prints it, pace instead of stopping each other.
``hinder`` has no such term,
and its weight enters once.  (Against a cooperative partner the table's conflict does stop
the partner in the same runs.)  With no ``R_j`` in its term, its level only says whom a
hinderer takes its partner to be: at level 1 a robot without social goal, at level 2 (its
default since 2026-10-11, ``Config.level_free_level``) a robot with one of the social goals
it holds possible -- so it can read a partner that picks its object up as one that helps.

``competition`` has five readings (``Config.competition``).

"table" is the table as printed.  Its entry at l = 1 is the conflict entry from the first
transition on: nothing in it says that the robot wants its own goal, and two level-1
competitors take each other's object so that neither goal is ever done.  Its entry against
a cooperative or exchanging partner is ``+R_j``, the cell of the cooperation row: the
competitor is paid for its partner's goal and delivers its partner's object.

"table_hostile" is the table with that one cell turned: against a cooperative or exchanging
partner a competitor above level 1 is paid ``-R_j``, as against a conflicting or competing
one, so that it works against the reward of a partner that would help it too.  Nothing
else is changed (l = 1: ``-r(g_j)``; against coercion: by the destinations, as printed).
The row is then the conflict row but for the cell against a coercing partner with the
robot's own destination: against a cooperator such a competitor is a conflict robot.  (With
an inferred partner goal that cell is in the mix whenever coercion is a hypothesis: the
two are then not the same robot, though in the runs they act alike.)

"race" and "race_then_hinder" (the default until 2026-10-08) are the table's two terms on
goal rewards, ``+r(g_i)`` (its coercion entry at l = 1) and ``-r(g_j)`` (its competition
entry at l = 1), counted from the finish line on, whatever the partner's social goal:

    xi_ij = r(g_i) + lam (r(g_i) - r(g_j))   once the robot's own goal is done
            -r(g_j)                          while its partner's goal is done and its own is not
            0                                while neither is done

Until a goal is done nothing in the term moves: the robot is paid for its own goal only.
If its partner is there first it loses its partner's goal reward for as long as it is
behind.  The two differ
in ``lam``, what the robot makes of its partner once its own goal is done -- by itself or,
for it, by its partner:

* "race", lam = 0: nothing.  It is paid for its own finished goal and has no reason left
  to move.
* "race_then_hinder", lam = 1: its lead, ``r(g_i) - r(g_j)``, what its partner still misses
  of its goal reward.  Every step of its partner toward its goal now costs the robot, so
  it takes its partner's object or stands in its way.  With ``r(g_i) = rho`` for good the
  term is ``2 rho - r(g_j)``: the term of ``hinder`` plus a constant, so from its own
  finish on the robot is a hinderer (same choices; rewards ``2 w rho`` higher).
  ``-r(g_j)`` is the table's competition entry at l = 1, here switched on by the robot's
  own finish.  (The own goal is paid twice so that finishing is worth more than spoiling
  at every weight: with ``r(g_i) - r(g_j)`` alone a competitor in a narrow doorway took
  its partner's object before its own at w = 30.)

"rival" is competition as "I want to do better than you" -- not "I want you to fail".  A
rival races its partner: the race term with lam = 0, at every level.  It looks after its
own goal and loses while its partner is done and it is not; once its own goal is done
nothing in the term concerns its partner.  The one partner it does not race is a
cooperative one.  Against a partner that helps it, it is the competitor of
"table_hostile", ``-r(g_j)`` at l = 1 and ``-R_j`` above it: it does not return help, it
works against its helper's reward (the rule this project asked for on 2026-10-06).  An
exchanging partner is not a helper here -- toward a competitor the table gives it no
social term -- and a rival races it, as it races a partner without social goal
("table_hostile" turns the table's printed cell, which holds both).  So what a rival is
depends on whom it faces -- as in the table, whose competition row has a cell per partner
goal -- and with known partner goals the reading is a switch between two others: against
a cooperator every model of the pairing is the one of "table_hostile", against anybody
else the one of "race".  Two rivals race each other, each for its own goal, where two
competitors of the table at level 2 -- each of which takes the other, at l = 1, for a robot
that is paid for its failure -- never get either goal done (README: "Why two competitors
of the table ping-pong").

At l = 1 the table has no column for the partner's social goal, and a rival needs one:
with ``Config.partner_social_goal="known"`` a level-1 rival is told whom it faces (its
partner is still a level-0 robot in its model); with ``"inferred"`` it is not, and races.
Above level 1 the term is settled per hypothesis about the partner: the race term under
every one but cooperation.  A rival's term can hold an ``R_j``, so its default level is 2.
With partner goals inferred (the default since 2026-10-08) nobody tells a level-2 rival
whom it faces either: it sees where its partner stands and what it holds, and its term is
the hostile cell ``-R_j`` weighted by its belief that the partner cooperates plus the race
weighted by its belief in every other goal.

Neither race form holds an ``R_j``, so the term is the same at every level
(:func:`race_weight`); what the level changes is whom the robot takes its partner to be.
At level 1, a competitor's default until 2026-10-11 (``Config.level_free_level``, 1 under
``TOLD``; ``pairing.default_levels``), its partner pursues its physical goal and nothing
else.  At level 2, its default since then, its partner has its social goal: the competitor
expects a cooperator to deliver for it and another competitor to turn on it once done, and
it acts on that before its own goal is done.  (Table I goes from ``r(g_j)`` at l = 1 to
``R_j`` at l > 1; with ``R_j`` in the race term the same robots finish in the same
transitions in the runs compared, so the term on goal rewards was kept: README.)

``cooperation`` has one option, ``Config.cooperate_with``: the partners against which its
cell of the column l > 1, ``+R_j``, holds.  As printed it holds against every partner, and
``R_j`` holds the partner's own social term: a cooperator takes on what its partner wants
for it.  Against a hinderer or a conflict partner that is ``-w_j r~(g_i)``, the partner's
estimate of the cooperator's own goal reward, so

    R_i = (1 - w_i w_j b~(g_i)) r(g_i) + w_i r(g_j) - c(a_i) - ...

with ``b~(g_i)`` how sure the partner is of the cooperator's goal.  At w = 1, once the
hinderer knows that goal, finishing it is worth nothing to the cooperator: on the center
scene of 2026-10-10 it stopped two moves from its destination in the transition the
hinderer's belief in that destination reached 0.99, and stayed there.  Since 2026-10-11
the default is :data:`COOPERATE_WITH` -- cooperation, competition, exchange (the user:
"lets modify our algorithm to only cooperate with competitive, cooperative, and exchange
agents"): the cell holds against those three and is 0 against any other partner
(conflict, hinder, coercion, none), against which the cooperator minds its own goal.
With an inferred partner goal the term is weighted by the belief hypothesis by
hypothesis, so the help fades as the cooperator becomes sure its partner is one of the
others; at the uniform prior over the seven goals it is 3/7 of the table's.  Of the three
models it still cooperates with, only a competitor's has a term below 0 on the
cooperator's goal: a rival is a racer at l = 1, ``-r(g_i)`` while the cooperator is done
and the competitor is not, so a cooperator may let a competitor finish first.  (Under
``competition="table"`` or ``"table_hostile"`` a competitor at l = 1 is the conflict entry,
and against it the cooperator is where it was against a hinderer.)  ``"all"`` is the table
as printed, the default until then (``TOLD``).  Nothing changes at l = 1: a level-1 robot
holds no belief about its partner's social goal.
"""
from __future__ import annotations

from typing import FrozenSet, Optional, Tuple

__all__ = ["SOCIAL_GOALS", "NONE", "COOPERATION", "CONFLICT", "COMPETITION", "COERCION",
           "EXCHANGE", "HINDER", "COMPETITION_FORMS", "COMPETITION_NAMES", "COOPERATE_WITH",
           "parse_goal", "goal_name", "races", "level_free", "lead_weight", "race_weight",
           "level1_coefficients", "level1_term", "level2_coefficients", "level2_term",
           "goal_sign", "describe", "cooperation_partners", "cooperation_reading",
           "cooperation_tag", "level_tag"]

SOCIAL_GOALS = ("none", "cooperation", "conflict", "competition", "coercion", "exchange",
                "hinder")
NONE, COOPERATION, CONFLICT, COMPETITION, COERCION, EXCHANGE, HINDER = range(7)

# Config.competition (its default: "rival"; "race_then_hinder" until 2026-10-08)
COMPETITION_FORMS = ("race_then_hinder", "race", "table", "table_hostile", "rival")
# ... and what a figure calls each of them
COMPETITION_NAMES = {"race_then_hinder": "race, then hinder", "race": "race", "table": "Table I",
                     "table_hostile": "Table I, hostile", "rival": "rival"}
# the readings whose term is a race at every level, against every partner
_RACE_FORMS = ("race", "race_then_hinder")
# the partner a rival does not race: it helps it, and the rival works against its reward.
# (Not an exchanging one, which has no social term toward a competitor: goals.py, "rival".)
_HELPERS = (COOPERATION,)
# Config.cooperate_with (its default since 2026-10-11; every goal, "all", until then): the
# partners whose reward R_j a level-2 cooperator takes on
COOPERATE_WITH = ("cooperation", "competition", "exchange")

_INDEX = {name: i for i, name in enumerate(SOCIAL_GOALS)}
_ALIASES = {"cooperate": "cooperation", "cooperative": "cooperation", "help": "cooperation",
            "compete": "competition", "competitive": "competition", "coerce": "coercion",
            "hindering": "hinder", "no": "none", "physical": "none"}

# Table I, column l = 1: (coefficient of r(g_j), coefficient of r(g_i)).
_LEVEL1 = {NONE: (0.0, 0.0), COOPERATION: (1.0, 0.0), CONFLICT: (-1.0, 0.0),
           COMPETITION: (-1.0, 0.0), COERCION: (0.0, 1.0), EXCHANGE: (0.0, 0.0),
           HINDER: (-1.0, 0.0)}
# Whether a goal pushes the partner's reward up (+1), down (-1) or neither (0).
_SIGN = {NONE: 0.0, COOPERATION: 1.0, CONFLICT: -1.0, COMPETITION: -1.0, COERCION: 0.0,
         EXCHANGE: 1.0, HINDER: -1.0}


def parse_goal(name) -> int:
    """A social goal by name, alias or index (``None`` / ``False``: no social goal) -> its
    index in :data:`SOCIAL_GOALS`."""
    if name is None or name is False:
        return NONE
    if isinstance(name, int) and not isinstance(name, bool):
        if 0 <= name < len(SOCIAL_GOALS):
            return int(name)
        raise ValueError(f"social goal index out of range: {name!r}")
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _INDEX:
        raise ValueError(f"unknown social goal {name!r}: one of {SOCIAL_GOALS}")
    return _INDEX[key]


def goal_name(goal: int) -> str:
    return SOCIAL_GOALS[int(goal)]


def goal_sign(goal: int) -> float:
    """+1 for goals that favour the partner, -1 for goals against it, 0 otherwise."""
    return _SIGN[int(goal)]


def cooperation_partners(names) -> FrozenSet[int]:
    """``Config.cooperate_with`` as goal indices: the partners' social goals against which a
    level-2 cooperator's cell of Table I, ``+R_j``, holds.  ``"all"`` (alone, or among the
    names): every goal, the table as printed."""
    if isinstance(names, str):
        names = (names,)
    names = tuple(names)
    if any(isinstance(n, str) and n.strip().lower() == "all" for n in names):
        return frozenset(range(len(SOCIAL_GOALS)))
    return frozenset(parse_goal(n) for n in names)


def cooperation_reading(names) -> str:
    """``Config.cooperate_with`` in words, for figures: "Table I" (every partner),
    "selective" (:data:`COOPERATE_WITH`), else "with <goals>"."""
    chosen = cooperation_partners(names)
    if chosen == frozenset(range(len(SOCIAL_GOALS))):
        return "Table I"
    if chosen == cooperation_partners(COOPERATE_WITH):
        return "selective"
    return "with " + (", ".join(goal_name(g) for g in sorted(chosen)) or "nobody")


def cooperation_tag(names) -> str:
    """What an output folder's name says of ``Config.cooperate_with``: nothing for the table
    as printed (the folders written before 2026-10-11 were made with it), "_selective" for
    :data:`COOPERATE_WITH`, "_coop_with_<goals>" for any other choice -- so a folder of one
    reading is never overwritten by a run of another."""
    chosen = cooperation_partners(names)
    if chosen == frozenset(range(len(SOCIAL_GOALS))):
        return ""
    if chosen == cooperation_partners(COOPERATE_WITH):
        return "_selective"
    return "_coop_with_" + ("_".join(goal_name(g) for g in sorted(chosen)) or "nobody")


def level_tag(level_free_level: int) -> str:
    """What an output folder's name says of ``Config.level_free_level``: nothing for 1 (a
    hinderer and a competitor of a race form at level 1, as in every folder written before
    2026-10-11), "_level2" for 2, the default since then (every robot with a social goal at
    level 2) -- so that no folder of before is overwritten."""
    return "" if int(level_free_level) == 1 else "_level2"


def level1_coefficients(goal: int, *, weight: float = 1.0, exchange_level1: str = "none",
                        exchange_share: float = 0.0, hinder_own_goal: bool = True
                        ) -> Tuple[float, float]:
    """``(a, b)`` of a level-1 social term ``xi = a r(g_j) + b r(g_i)``: Table I's column
    l = 1 times the robot's weight.

    ``exchange_level1``: what an exchange goal is at level 1.  "none": the table's N/A, no
    social term.  "copy": the paper's text, a copy of the partner's goal with the small
    scalar n -- ``n r(g_j)``, times the robot's weight like every entry of the table.
    "copy_unweighted": the same copy NOT multiplied by the weight -- it keeps the size
    ``n`` whatever the weight (at weight 1, the paper's, the two are the same; a robot with
    weight 0 has no social goal and no copy).

    ``hinder_own_goal=False`` (``Config.hinder_own_goal``): a hinderer has no goal of its
    own -- ``b = -1``, unweighted, takes away the ``r(g_i)`` of Eq. 2."""
    if goal == HINDER and not hinder_own_goal:
        return weight * -1.0 + 0.0, -1.0
    a, b = _LEVEL1[int(goal)]
    if goal == EXCHANGE and exchange_level1 == "copy":
        a = float(exchange_share)
    if goal == EXCHANGE and exchange_level1 == "copy_unweighted":
        return (float(exchange_share) if weight > 0.0 else 0.0) + 0.0, 0.0
    # "+ 0.0" turns -0.0 into 0.0, so that equal coefficients are equal dictionary keys
    return weight * a + 0.0, weight * b + 0.0


def races(goal: int, *, competition: str = "race_then_hinder",
          partner_goal: Optional[int] = None) -> bool:
    """Is the social term of a robot with ``goal`` the race term (:func:`level1_term`)
    against a partner taken to have ``partner_goal``?  A competitor's under
    ``Config.competition="race"`` and ``"race_then_hinder"``, whoever its partner is; and
    under ``"rival"`` against every partner but a cooperative one, which helps it.
    ``partner_goal=None``: a partner whose social goal the robot does not know -- a rival
    races it."""
    if goal != COMPETITION:
        return False
    if competition in _RACE_FORMS:
        return True
    return competition == "rival" and not (partner_goal is not None
                                           and int(partner_goal) in _HELPERS)


def level_free(goal: int, *, competition: str = "race_then_hinder") -> bool:
    """Is the social term of a robot with ``goal`` the same at every level -- on goal
    rewards alone, no partner reward ``R_j`` in it whoever the partner is?  ``hinder``,
    and competition under the two race forms.  Such a robot reasons at
    ``Config.level_free_level`` by default (``pairing.default_levels``): 2 since 2026-10-11,
    like every robot with a social goal; 1 until then.  A rival's term is not: against a
    cooperator it is ``-R_j``."""
    return goal == HINDER or (goal == COMPETITION and competition in _RACE_FORMS)


def lead_weight(competition: str) -> float:
    """``lam`` of the race term: what a competitor whose own goal is done makes of its lead
    over its partner.  1 under "race_then_hinder"; 0 under "race" and "rival" (and unused
    under the two readings of the table)."""
    return 1.0 if competition == "race_then_hinder" else 0.0


def race_weight(goal: int, *, weight: float = 1.0, competition: str = "race_then_hinder",
                partner_goal: Optional[int] = None) -> float:
    """The coefficient of the race term (:func:`level1_term`) in the social term of a robot
    with ``goal``: its weight for a competitor that races (:func:`races`), 0 for everybody
    else.  It is the same at level 1 and at level 2: the term is on goal rewards, and the
    level only says whom the robot takes its partner to be.  At level 2 such a robot has
    nothing of Table I's column l > 1 (no ``R_j``).

    ``partner_goal``: the social goal the robot takes its partner to have (None: it does
    not know).  It matters under "rival" only: a rival races every partner but a
    cooperative one, against which this coefficient is 0 and its term the table's
    (:func:`level1_term`, :func:`level2_term`)."""
    return weight + 0.0 if races(goal, competition=competition,
                                 partner_goal=partner_goal) else 0.0


def level1_term(goal: int, *, weight: float = 1.0, exchange_level1: str = "none",
                exchange_share: float = 0.0, competition: str = "race_then_hinder",
                partner_goal: Optional[int] = None, hinder_own_goal: bool = True
                ) -> Tuple[float, float, float]:
    """``(a, b, c)`` of the social term of a robot at level 1,

        xi = a r(g_j) + b r(g_i) + c race,
        race = r(g_i) + lam (r(g_i) - r(g_j)) if g_i is done, else -r(g_j) if g_j is done,
               else 0                                    (lam: :func:`lead_weight`).

    ``(a, b)`` is Table I's column l = 1 (:func:`level1_coefficients`) and ``c = 0`` -- except
    for a competitor that races (:func:`races`), whose term is ``(0, 0, w)``.

    ``partner_goal``: the social goal of the robot's partner where the robot knows it
    (``Config.partner_social_goal="known"``), else None.  The table's column l = 1 does
    not ask for it, and neither does any reading but "rival": a rival that knows its
    partner to be cooperative has the table's entry, ``(-w, 0, 0)``."""
    if races(goal, competition=competition, partner_goal=partner_goal):
        return 0.0, 0.0, weight + 0.0
    a, b = level1_coefficients(goal, weight=weight, exchange_level1=exchange_level1,
                               exchange_share=exchange_share, hinder_own_goal=hinder_own_goal)
    return a, b, 0.0


def level2_coefficients(goal: int, partner_goal: int, same_destination: bool, *,
                        weight: float = 1.0, exchange_share: float = 0.0,
                        competition: str = "table",
                        cooperate_with: Optional[FrozenSet[int]] = None,
                        hinder_own_goal: bool = True) -> Tuple[float, float, float]:
    """``(alpha, beta, nu)`` of a level-2 social term ``xi = alpha R_j + beta r(g_i) + nu
    r(g_j)`` against a partner taken to have ``partner_goal``: Table I's column l > 1 times
    the robot's weight.  ``same_destination``: the partner's hypothesised destination is
    the robot's own (the table's "g_i = g_j").  A partner without a social goal is not a
    row of the table; it is read by the level-1 column (``R_j`` is then its physical
    reward).  A ``hinder`` partner counts as a conflict partner.

    ``competition``: "table" is the table as printed.  "table_hostile" turns one cell of
    it: a competitor against a cooperative or exchanging partner, ``-R_j`` where the table
    prints ``+R_j``.  Asked for its cells here, a rival is given the hostile table's --
    but its term uses one of them only, the cell against a cooperative partner (against
    an exchanging partner it races), and the race forms use none: the whole term of every
    reading is :func:`level2_term`, not this function.

    ``cooperate_with``: the partners' social goals (indices, :func:`cooperation_partners`)
    against which a cooperator's cell, ``+R_j``, holds; against any other it is 0 -- the
    cooperator minds its own goal (``Config.cooperate_with``).  None: every partner, the
    table as printed.  It is asked of the partner's own goal, before a hinder partner is
    read as a conflict partner, so either may be left out alone."""
    if (goal == COOPERATION and cooperate_with is not None
            and int(partner_goal) not in cooperate_with):
        return 0.0, 0.0, 0.0
    t = CONFLICT if partner_goal == HINDER else int(partner_goal)
    alpha, beta, nu = _table1(int(goal), t, bool(same_destination), float(exchange_share),
                              competition in ("table_hostile", "rival"))
    if goal == HINDER and not hinder_own_goal:   # no goal of its own: beta = -1, unweighted
        return 0.0, -1.0, weight * nu + 0.0
    return weight * alpha + 0.0, weight * beta + 0.0, weight * nu + 0.0


def level2_term(goal: int, partner_goal: int, same_destination: bool, *, weight: float = 1.0,
                exchange_share: float = 0.0, competition: str = "race_then_hinder",
                cooperate_with: Optional[FrozenSet[int]] = None,
                hinder_own_goal: bool = True) -> Tuple[float, float, float, float]:
    """``(alpha, beta, nu, c)`` of the whole social term of a robot at level 2 against a
    partner taken to have ``partner_goal``,

        xi = alpha R_j + beta r(g_i) + nu r(g_j) + c race        (race: :func:`level1_term`).

    A competitor that races this partner (:func:`races`) has ``(0, 0, 0, w)``: under the
    two race forms whoever the partner is, under "rival" unless the partner is cooperative.
    Everything else is a cell of Table I's column l > 1 (:func:`level2_coefficients`) with
    ``c = 0`` -- a cooperator's only against the partners in ``cooperate_with`` (None:
    every partner, the table as printed)."""
    if races(goal, competition=competition, partner_goal=int(partner_goal)):
        return 0.0, 0.0, 0.0, weight + 0.0
    return level2_coefficients(goal, partner_goal, same_destination, weight=weight,
                               exchange_share=exchange_share, competition=competition,
                               cooperate_with=cooperate_with,
                               hinder_own_goal=hinder_own_goal) + (0.0,)


def _table1(goal: int, t: int, same: bool, n: float, hostile: bool = False
            ) -> Tuple[float, float, float]:
    if goal == COOPERATION:
        return 1.0, 0.0, 0.0
    if goal == CONFLICT:
        return -1.0, 0.0, 0.0
    if goal == HINDER:
        return 0.0, 0.0, -1.0
    if goal == COMPETITION:
        if t in (COOPERATION, EXCHANGE):         # the table: +R_j, it helps; "table_hostile": -R_j
            return (-1.0 if hostile else 1.0), 0.0, 0.0
        if t == COERCION:
            return (1.0 if same else -1.0), 0.0, 0.0
        return -1.0, 0.0, 0.0                    # vs conflict, competition (and "none")
    if goal == COERCION:
        if t in (COOPERATION, EXCHANGE):
            return 1.0, 1.0, 0.0
        if t == COMPETITION:
            return (1.0, 0.0, 0.0) if same else (-1.0, 1.0, 0.0)
        if t == NONE:
            return 0.0, 1.0, 0.0
        return -1.0, 1.0, 0.0                    # vs conflict, coercion
    if goal == EXCHANGE:
        return (1.0, 0.0, n) if t == EXCHANGE else (0.0, 0.0, 0.0)
    return 0.0, 0.0, 0.0


def describe(goal: int, level: int, partner_goal: Optional[int] = None, *,
             competition: str = "race_then_hinder", exchange_level1: str = "none",
             cooperate_with: Optional[FrozenSet[int]] = None,
             hinder_own_goal: bool = True) -> str:
    """The social term of a goal in words, for logs and result records.  ``competition``:
    ``Config.competition``; ``exchange_level1``: ``Config.exchange_level1``;
    ``cooperate_with``: ``Config.cooperate_with`` as indices (None: every partner, the
    table as printed).  ``partner_goal``: the partner's social goal where the robot knows
    it, else None.  It matters above level 1, where Table I has a column per partner goal
    -- and for a rival at level 1, which has to know whom it faces."""
    name = goal_name(goal)
    if goal == NONE or level == 0:
        return "no social term"
    if competition == "rival" and goal == COMPETITION:
        race = ("w (+r(g_i) once its own goal is done; until then -r(g_j) while its "
                "partner's is)")
        if partner_goal is None:                 # it does not know whom it faces
            if level == 1:
                return f"competition: {race}  [rival: a race]"
            return (f"competition: w (-1 R_j) against a cooperative partner; against any "
                    f"other, {race}  [rival]")
        versus = f"competition vs {goal_name(partner_goal)}"
        if int(partner_goal) not in _HELPERS:
            return f"{versus}: {race}  [rival: a race]"
        reward = "r(g_j)" if level == 1 else "R_j"
        return f"{versus}: w (-1 {reward})  [rival: against a helper, the hostile table]"
    if races(goal, competition=competition):     # the same term at level 1 and at level 2
        if lead_weight(competition):
            return ("competition: w (2 r(g_i) - r(g_j) once its own goal is done; until "
                    "then -r(g_j) while its partner's is)  [race, then hinder]")
        return ("competition: w (+r(g_i) once its own goal is done; until then -r(g_j) while "
                "its partner's is)  [race form]")
    if level == 1 or goal == HINDER:
        a, b = _LEVEL1[int(goal)]
        if goal == HINDER:
            if not hinder_own_goal:
                return ("hinder: -w r(g_j), the partner's goal reward, and no goal of its own "
                        "(-r(g_i))")
            return "hinder: -w r(g_j), the partner's goal reward"
        if goal == EXCHANGE and exchange_level1 == "copy":
            return "exchange: w (n r(g_j))  [the text's copy of the partner's goal]"
        if goal == EXCHANGE and exchange_level1 == "copy_unweighted":
            return ("exchange: n r(g_j), not multiplied by w  [the text's copy of the "
                    "partner's goal]")
        if (a, b) == (0.0, 0.0):
            return f"{name}: no social term at level 1 (Table I: N/A)"
        terms = [f"{a:+g} r(g_j)"] * (a != 0.0) + [f"{b:+g} r(g_i)"] * (b != 0.0)
        return f"{name}: w ({' '.join(terms)})  [Table I, l = 1]"
    if (goal == COOPERATION and cooperate_with is not None
            and not frozenset(range(len(SOCIAL_GOALS))) <= frozenset(cooperate_with)):
        whom = ", ".join(goal_name(g) for g in sorted(cooperate_with)) or "nobody"
        if partner_goal is None:
            return (f"cooperation: w (+1 R_j) against a partner it takes for {whom}; no "
                    f"social term against any other  [cooperates with {whom} only]")
        if int(partner_goal) not in cooperate_with:
            return (f"cooperation vs {goal_name(partner_goal)}: no social term  "
                    f"[cooperates with {whom} only]")
    if partner_goal is None:                     # the partner's goal is inferred: a mix of cells
        turned = ("; -R_j against a cooperative or exchanging one: table_hostile"
                  if goal == COMPETITION and competition == "table_hostile" else "")
        return f"{name}: Table I, l > 1 (depends on the partner's social goal{turned})"

    def body(same: bool) -> str:
        al, be, nu = level2_coefficients(goal, partner_goal, same, exchange_share=1.0,
                                         competition=competition)
        terms = ([f"{al:+g} R_j"] * (al != 0.0) + [f"{be:+g} r(g_i)"] * (be != 0.0)
                 + ["+n r(g_j)"] * (nu != 0.0))
        return " ".join(terms) if terms else "no social term (Table I: N/A)"

    apart, same = body(False), body(True)       # two cells of the table ask "g_i = g_j?"
    text = apart if apart == same else f"{same} if g_i = g_j, else {apart}"
    turned = (goal == COMPETITION and competition == "table_hostile"
              and partner_goal in (COOPERATION, EXCHANGE))
    source = "Table I, l > 1, the sign turned: table_hostile" if turned else "Table I, l > 1"
    return f"{name} vs {goal_name(partner_goal)}: w ({text})  [{source}]"
