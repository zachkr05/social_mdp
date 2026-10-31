"""Social goals: Table I of the paper cell by cell, hinder, weights and names."""
import itertools
import unittest

from social_pairings import COOPERATE_WITH, EVERY_PARTNER, TOLD
from social_pairings import Config as PackageConfig
from social_pairings import goals as G

from .reference import (HELPERS, TABLE_L1, TABLE_L2, TABLE_L2_HOSTILE, TYPES, level1_term,
                        level2_race, table_l1, table_l2)
from .told import Config


class TableOneTests(unittest.TestCase):
    def test_level_1_column(self):
        """cooperation r(g_j); conflict and competition -r(g_j); coercion r(g_i); exchange N/A."""
        self.assertEqual(TABLE_L1, {"cooperation": (1, 0), "conflict": (-1, 0),
                                    "competition": (-1, 0), "coercion": (0, 1),
                                    "exchange": None})
        for name in TYPES + ("none",):
            self.assertEqual(G.level1_coefficients(G.parse_goal(name)), table_l1(name), name)

    def test_level_2_column_every_cell(self):
        """All 6 x 6 x 2 cells: five goals and "none" against five goals and "none", with the
        partner's destination equal to / different from the robot's own."""
        n = 0.3
        for mine, theirs, same in itertools.product(TYPES + ("none",), TYPES + ("none",),
                                                    (False, True)):
            got = G.level2_coefficients(G.parse_goal(mine), G.parse_goal(theirs), same,
                                        exchange_share=n)
            self.assertEqual(got, table_l2(mine, theirs, same, n), (mine, theirs, same))

    def test_cells_that_depend_on_the_destinations(self):
        """Table I: competition against coercion, and coercion against competition."""
        comp, coer = G.COMPETITION, G.COERCION
        self.assertEqual(G.level2_coefficients(comp, coer, True), (1.0, 0.0, 0.0))
        self.assertEqual(G.level2_coefficients(comp, coer, False), (-1.0, 0.0, 0.0))
        self.assertEqual(G.level2_coefficients(coer, comp, True), (1.0, 0.0, 0.0))
        self.assertEqual(G.level2_coefficients(coer, comp, False), (-1.0, 1.0, 0.0))
        self.assertEqual(len([cell for cell in TABLE_L2.values()
                              if cell is not None and isinstance(cell[0], tuple)]), 2)

    def test_exchange(self):
        """n r(g_j) + R_j against a partner that exchanges; N/A against any other, and at
        level 1 (unless the text's reading, a copy of the partner's goal, is asked for)."""
        ex = G.EXCHANGE
        self.assertEqual(G.level2_coefficients(ex, ex, False, exchange_share=0.25),
                         (1.0, 0.0, 0.25))
        for other in ("cooperation", "conflict", "competition", "coercion", "none", "hinder"):
            for same in (False, True):
                self.assertEqual(G.level2_coefficients(ex, G.parse_goal(other), same,
                                                       exchange_share=0.25), (0.0, 0.0, 0.0))
        self.assertEqual(G.level1_coefficients(ex, exchange_share=0.25), (0.0, 0.0))
        self.assertEqual(G.level1_coefficients(ex, exchange_level1="copy", exchange_share=0.25),
                         (0.25, 0.0))

    def test_the_copy_of_a_level_1_exchanger_and_the_weight(self):
        """"copy": n r(g_j) times the weight, like every entry of the table.
        "copy_unweighted": n whatever the weight -- the same at weight 1, the paper's, and
        nothing for a robot with weight 0, which has no social goal."""
        ex = G.EXCHANGE
        for weight, copy, unweighted in ((0.0, 0.0, 0.0), (0.5, 0.125, 0.25), (1.0, 0.25, 0.25),
                                         (30.0, 7.5, 0.25)):
            with self.subTest(weight=weight):
                self.assertEqual(G.level1_coefficients(ex, weight=weight, exchange_share=0.25,
                                                       exchange_level1="copy"), (copy, 0.0))
                self.assertEqual(G.level1_coefficients(ex, weight=weight, exchange_share=0.25,
                                                       exchange_level1="copy_unweighted"),
                                 (unweighted, 0.0))
                self.assertEqual(G.level1_term(ex, weight=weight, exchange_share=0.25,
                                               exchange_level1="copy_unweighted"),
                                 table_l1("exchange", weight, 0.25, True) + (0.0,))
                self.assertEqual(G.level1_coefficients(ex, weight=weight, exchange_share=0.25),
                                 (0.0, 0.0))
        # no other goal is touched by the option, at level 1 or above
        for name in ("none", "cooperation", "conflict", "competition", "coercion", "hinder"):
            goal = G.parse_goal(name)
            for mode in ("copy", "copy_unweighted"):
                self.assertEqual(G.level1_coefficients(goal, weight=3.0, exchange_share=0.25,
                                                       exchange_level1=mode),
                                 G.level1_coefficients(goal, weight=3.0))
        self.assertEqual(G.level2_coefficients(ex, ex, False, weight=30.0, exchange_share=0.25),
                         (30.0, 0.0, 7.5))        # the level-2 row keeps its weight
        self.assertIn("N/A", G.describe(ex, 1))
        self.assertIn("w (n r(g_j))", G.describe(ex, 1, exchange_level1="copy"))
        self.assertIn("not multiplied by w", G.describe(ex, 1, exchange_level1="copy_unweighted"))
        self.assertEqual(G.describe(ex, 2, ex, exchange_level1="copy_unweighted"),
                         G.describe(ex, 2, ex))
        with self.assertRaises(ValueError):
            Config(exchange_level1="copy_unit")
        self.assertEqual(Config(exchange_level1="copy_unweighted").exchange_level1,
                         "copy_unweighted")


class RaceTests(unittest.TestCase):
    """Competition as a race: ``Config.competition="race_then_hinder"`` (the default of
    ``told.py``) and ``"race"``; ``"table"`` is Table I as printed and ``"table_hostile"``
    the table with one cell turned (:class:`HostileTableTests`); ``"rival"`` races every
    partner but a helper (:class:`RivalTests`)."""

    def test_the_five_readings(self):
        self.assertEqual(G.COMPETITION_FORMS,
                         ("race_then_hinder", "race", "table", "table_hostile", "rival"))
        self.assertEqual(Config().competition, G.COMPETITION_FORMS[0])
        for mode in G.COMPETITION_FORMS:
            Config(competition=mode)
        with self.assertRaises(ValueError):
            Config(competition="lead")
        # every reading has a name for the figures' headlines
        self.assertEqual(set(G.COMPETITION_NAMES), set(G.COMPETITION_FORMS))
        self.assertEqual(len(set(G.COMPETITION_NAMES.values())), len(G.COMPETITION_FORMS))
        # what a competitor whose own goal is done makes of its lead over its partner
        self.assertEqual([G.lead_weight(mode) for mode in G.COMPETITION_FORMS],
                         [1.0, 0.0, 0.0, 0.0, 0.0])
        # the readings whose term holds no R_j whoever the partner is: level 1 by default
        self.assertEqual([G.level_free(G.COMPETITION, competition=mode)
                          for mode in G.COMPETITION_FORMS], [True, True, False, False, False])
        self.assertTrue(all(G.level_free(G.HINDER, competition=mode)
                            for mode in G.COMPETITION_FORMS))

    def test_a_competitor_races_whoever_its_partner_is(self):
        """A race term is a competitor's under ``"race_then_hinder"`` and ``"race"`` and
        nobody else's, whoever its partner is and whether or not it is told; under the two
        readings of the table nothing races."""
        self.assertTrue(G.races(G.COMPETITION))
        self.assertTrue(G.races(G.COMPETITION, competition="race"))
        self.assertFalse(G.races(G.COMPETITION, competition="table"))
        self.assertFalse(G.races(G.COMPETITION, competition="table_hostile"))
        partners = (None,) + tuple(range(len(G.SOCIAL_GOALS)))
        for goal, mode, partner in itertools.product(G.SOCIAL_GOALS, G.COMPETITION_FORMS,
                                                     partners):
            got = G.races(G.parse_goal(goal), competition=mode, partner_goal=partner)
            if goal != "competition" or mode in ("table", "table_hostile"):
                self.assertFalse(got, (goal, mode, partner))
            elif mode in ("race_then_hinder", "race"):
                self.assertTrue(got, (goal, mode, partner))

    def test_level_1_term_of_every_goal(self):
        """(a, b, c): Table I's column l = 1 with c = 0, except for a racing competitor.
        7 goals in 5 modes, told each of 7 partner goals or none: the term depends on the
        partner for a rival and for nobody else."""
        partners = (None,) + G.SOCIAL_GOALS
        for goal, mode in itertools.product(G.SOCIAL_GOALS, G.COMPETITION_FORMS):
            terms = set()
            for partner in partners:
                got = G.level1_term(G.parse_goal(goal), weight=3.0, competition=mode,
                                    partner_goal=None if partner is None
                                    else G.parse_goal(partner))
                self.assertEqual(got, level1_term(goal, 3.0, 0.0, mode, partner),
                                 (goal, mode, partner))
                terms.add(got)
            self.assertEqual(len(terms), 2 if (goal, mode) == ("competition", "rival") else 1,
                             (goal, mode))
            # not told: the call of before 2026-10-07
            self.assertEqual(G.level1_term(G.parse_goal(goal), weight=3.0, competition=mode),
                             level1_term(goal, 3.0, 0.0, mode), (goal, mode))
        for mode in ("race_then_hinder", "race"):
            # a racer's term holds nothing of its partner's as a gain: a = b = 0
            self.assertEqual(G.level1_term(G.COMPETITION, weight=3.0, competition=mode),
                             (0.0, 0.0, 3.0))
        self.assertEqual(G.level1_term(G.COMPETITION, weight=3.0), (0.0, 0.0, 3.0))
        for mode in ("table", "table_hostile"):
            self.assertEqual(G.level1_term(G.COMPETITION, weight=3.0, competition=mode),
                             (-3.0, 0.0, 0.0))

    def test_the_coefficient_of_the_race_term(self):
        """``race_weight`` takes no level: the coefficient is the competitor's weight, the
        one of its level-1 term.  Under the two race forms it takes no partner either;
        nobody else has one, and under the two readings of the table nobody.  (That the
        level-2 models use it is test_models.py's comparison with brute force.)"""
        for mode in ("race_then_hinder", "race"):
            self.assertEqual(G.race_weight(G.COMPETITION, weight=3.0, competition=mode), 3.0)
            self.assertEqual(G.race_weight(G.COMPETITION, weight=3.0, competition=mode),
                             G.level1_term(G.COMPETITION, weight=3.0, competition=mode)[2])
            self.assertEqual(G.race_weight(G.COMPETITION, weight=0.0, competition=mode), 0.0)
            for partner in range(len(G.SOCIAL_GOALS)):
                self.assertEqual(G.race_weight(G.COMPETITION, weight=3.0, competition=mode,
                                               partner_goal=partner), 3.0)
        self.assertEqual(G.race_weight(G.COMPETITION), 1.0)                # the defaults
        for mode in ("table", "table_hostile"):
            self.assertEqual(G.race_weight(G.COMPETITION, weight=3.0, competition=mode), 0.0)
        for goal, mode in itertools.product(G.SOCIAL_GOALS, G.COMPETITION_FORMS):
            if goal != "competition":
                self.assertEqual(G.race_weight(G.parse_goal(goal), weight=3.0,
                                               competition=mode), 0.0, (goal, mode))
        self.assertIn("race_weight", G.__all__)

    def test_the_whole_level_2_term_of_every_reading(self):
        """``level2_term``, (alpha, beta, nu, c): 7 goals against 7 partner goals, both
        answers to "g_i = g_j?", in 5 modes -- 490 cells against reference.py.  A robot
        that races the partner has (0, 0, 0, w) and nothing of the table; everybody else
        the table's cell and c = 0."""
        names = TYPES + ("none", "hinder")
        cells = 0
        for mine, theirs, same, mode in itertools.product(names, names, (False, True),
                                                          G.COMPETITION_FORMS):
            got = G.level2_term(G.parse_goal(mine), G.parse_goal(theirs), same, weight=3.0,
                                exchange_share=0.3, competition=mode)
            race = level2_race(mine, 3.0, mode, theirs)
            want = ((0.0, 0.0, 0.0) if race else
                    table_l2(mine, theirs, same, 0.3, 3.0, mode)) + (race,)
            self.assertEqual(got, want, (mine, theirs, same, mode))
            self.assertEqual(G.race_weight(G.parse_goal(mine), weight=3.0, competition=mode,
                                           partner_goal=G.parse_goal(theirs)), race)
            cells += 1
        self.assertEqual(cells, 490)

    def test_describe_names_the_race(self):
        """... against every partner and at either level: a race term is the same term at
        level 1 and at level 2.  The table's cells are reached with
        ``competition="table"`` and ``"table_hostile"`` only."""
        for partner, level in itertools.product(
                (None, G.COMPETITION, G.COOPERATION, G.EXCHANGE, G.NONE), (1, 2)):
            text = G.describe(G.COMPETITION, level, partner)
            self.assertIn("[race, then hinder]", text)
            self.assertIn("2 r(g_i) - r(g_j) once its own goal is done", text)
            text = G.describe(G.COMPETITION, level, partner, competition="race")
            self.assertIn("[race form]", text)
            self.assertIn("+r(g_i) once its own goal is done", text)
        self.assertEqual(G.describe(G.COMPETITION, 0, G.COOPERATION), "no social term")
        self.assertNotIn("race", G.describe(G.COMPETITION, 1, G.COMPETITION,
                                            competition="table"))
        self.assertIn("-1 R_j", G.describe(G.COMPETITION, 2, G.COMPETITION,
                                           competition="table"))
        self.assertIn("+1 R_j", G.describe(G.COMPETITION, 2, G.COOPERATION,
                                           competition="table"))
        for mode in ("race_then_hinder", "race"):
            self.assertNotIn("R_j", G.describe(G.COMPETITION, 2, G.COOPERATION,
                                               competition=mode))


class RivalTests(unittest.TestCase):
    """``Config.competition="rival"``: competition as "I want to do better than you".  A
    rival races its partner -- the race term with lam = 0, at level 1 and at level 2 --
    unless the partner helps it, which a cooperative partner does and no other: against
    that partner it is the competitor of "table_hostile", ``-r(g_j)`` at l = 1 and
    ``-R_j`` above."""

    PARTNERS = TYPES + ("none", "hinder")

    def test_it_races_every_partner_but_a_helper(self):
        """... and the one helper is a cooperative partner.  An exchanging partner, which
        shares that cell of Table I, has no social term toward a competitor: it is raced,
        like a partner without social goal."""
        comp = G.COMPETITION
        self.assertEqual(HELPERS, ("cooperation",))
        for theirs in ("exchange", "none"):
            partner = G.parse_goal(theirs)
            self.assertEqual(G.level1_term(comp, weight=3.0, competition="rival",
                                           partner_goal=partner), (0.0, 0.0, 3.0))
            self.assertEqual(G.level2_term(comp, partner, False, weight=3.0,
                                           competition="rival"), (0.0, 0.0, 0.0, 3.0))
            # ... and it has no social term toward the competitor
            self.assertEqual(G.level2_term(partner, comp, False, weight=3.0,
                                           exchange_share=0.3, competition="rival"),
                             (0.0, 0.0, 0.0, 0.0))
        for theirs in self.PARTNERS:
            partner = G.parse_goal(theirs)
            one = G.level1_term(comp, weight=3.0, competition="rival", partner_goal=partner)
            for same in (False, True):
                two = G.level2_term(comp, partner, same, weight=3.0, competition="rival")
                if theirs in HELPERS:
                    self.assertEqual((one, two), ((-3.0, 0.0, 0.0), (-3.0, 0.0, 0.0, 0.0)),
                                     theirs)
                else:
                    self.assertEqual((one, two), ((0.0, 0.0, 3.0), (0.0, 0.0, 0.0, 3.0)),
                                     theirs)
            self.assertEqual(G.races(comp, competition="rival", partner_goal=partner),
                             theirs not in HELPERS)
        # the destinations are never asked: no cell of a rival's has two halves
        for theirs in self.PARTNERS:
            self.assertEqual(
                G.level2_term(comp, G.parse_goal(theirs), True, competition="rival"),
                G.level2_term(comp, G.parse_goal(theirs), False, competition="rival"))

    def test_it_is_one_of_two_other_readings_by_whom_it_faces(self):
        """With its partner's goal known a rival's term is, at either level, the term of
        "table_hostile" against a helper and the term of "race" against anybody else; and
        every other goal's term is what it is under those two readings."""
        comp = G.COMPETITION
        for theirs in self.PARTNERS:
            partner = G.parse_goal(theirs)
            twin = "table_hostile" if theirs in HELPERS else "race"
            self.assertEqual(
                G.level1_term(comp, weight=2.0, competition="rival", partner_goal=partner),
                G.level1_term(comp, weight=2.0, competition=twin, partner_goal=partner))
            for same in (False, True):
                self.assertEqual(
                    G.level2_term(comp, partner, same, weight=2.0, competition="rival"),
                    G.level2_term(comp, partner, same, weight=2.0, competition=twin))
            for mine in self.PARTNERS:
                if mine == "competition":
                    continue
                goal = G.parse_goal(mine)
                for mode in ("race", "table_hostile"):
                    self.assertEqual(
                        G.level1_term(goal, weight=2.0, competition="rival",
                                      partner_goal=partner),
                        G.level1_term(goal, weight=2.0, competition=mode))
                    self.assertEqual(
                        G.level2_term(goal, partner, False, weight=2.0, exchange_share=0.3,
                                      competition="rival"),
                        G.level2_term(goal, partner, False, weight=2.0, exchange_share=0.3,
                                      competition=mode))

    def test_a_rival_that_is_not_told_whom_it_faces_races(self):
        """``partner_goal=None`` (an inferred partner goal, at level 1): the race term."""
        self.assertEqual(G.level1_term(G.COMPETITION, weight=3.0, competition="rival"),
                         (0.0, 0.0, 3.0))
        self.assertTrue(G.races(G.COMPETITION, competition="rival"))
        self.assertEqual(G.race_weight(G.COMPETITION, weight=3.0, competition="rival"), 3.0)
        self.assertEqual(G.race_weight(G.COMPETITION, weight=3.0, competition="rival",
                                       partner_goal=G.COOPERATION), 0.0)

    def test_no_lead_and_not_the_same_term_at_every_level(self):
        """lam = 0: the race term without the lead (what the term then pays is
        test_competition.py's); and a term that can hold an R_j, so that a rival's default
        level is 2."""
        self.assertEqual(G.lead_weight("rival"), 0.0)
        self.assertFalse(G.level_free(G.COMPETITION, competition="rival"))

    def test_describe(self):
        comp = G.COMPETITION
        for level in (1, 2):
            for partner in (G.COMPETITION, G.CONFLICT, G.COERCION, G.EXCHANGE, G.HINDER,
                            G.NONE):
                text = G.describe(comp, level, partner, competition="rival")
                self.assertIn("[rival: a race]", text)
                self.assertIn("+r(g_i) once its own goal is done", text)
                self.assertIn(f"competition vs {G.goal_name(partner)}", text)
                self.assertNotIn("R_j", text)
        self.assertIn("competition vs cooperation: w (-1 r(g_j))  [rival: against a helper",
                      G.describe(comp, 1, G.COOPERATION, competition="rival"))
        self.assertIn("competition vs cooperation: w (-1 R_j)  [rival: against a helper",
                      G.describe(comp, 2, G.COOPERATION, competition="rival"))
        # not told whom it faces: a race at level 1, both terms named at level 2
        self.assertIn("[rival: a race]", G.describe(comp, 1, None, competition="rival"))
        inferred = G.describe(comp, 2, None, competition="rival")
        self.assertIn("w (-1 R_j) against a cooperative partner; against any other, "
                      "w (+r(g_i) once its own goal is done", inferred)
        self.assertNotIn("exchang", inferred)
        # every other goal is described as under the table
        for goal in (G.COOPERATION, G.CONFLICT, G.COERCION, G.EXCHANGE, G.HINDER):
            for level, partner in ((1, None), (2, comp), (2, None)):
                self.assertEqual(G.describe(goal, level, partner, competition="rival"),
                                 G.describe(goal, level, partner, competition="table"))
        self.assertEqual(G.describe(comp, 0, G.COOPERATION, competition="rival"),
                         "no social term")


class HostileTableTests(unittest.TestCase):
    """``Config.competition="table_hostile"``: Table I with one printed cell turned.  A
    competitor above level 1 is paid ``-R_j`` against a cooperative or exchanging partner,
    where the table prints ``+R_j``; everything else is the table."""

    CELLS = tuple(itertools.product(TYPES + ("none", "hinder"), TYPES + ("none", "hinder"),
                                    (False, True)))

    @staticmethod
    def cell(mine: str, theirs: str, same: bool, mode: str, weight: float = 1.0) -> tuple:
        return G.level2_coefficients(G.parse_goal(mine), G.parse_goal(theirs), same,
                                     weight=weight, exchange_share=0.3, competition=mode)

    def test_every_cell(self):
        """All 7 x 7 x 2 cells (five goals, "none" and "hinder", each way) against the
        transcription in reference.py."""
        self.assertEqual(len(self.CELLS), 98)
        for mine, theirs, same in self.CELLS:
            self.assertEqual(self.cell(mine, theirs, same, "table_hostile"),
                             table_l2(mine, theirs, same, 0.3, competition="table_hostile"),
                             (mine, theirs, same))

    def test_it_turns_the_competitors_cell_against_a_helpful_partner_and_no_other(self):
        """Four cells of the 98 differ from the table as printed: competition against
        cooperation and against exchange, at either answer to "g_i = g_j?" -- one printed
        cell.  In each the coefficient of R_j goes from +w to -w and nothing else moves."""
        turned = [(mine, theirs, same) for mine, theirs, same in self.CELLS
                  if self.cell(mine, theirs, same, "table_hostile")
                  != self.cell(mine, theirs, same, "table")]
        self.assertEqual(turned, [("competition", "cooperation", False),
                                  ("competition", "cooperation", True),
                                  ("competition", "exchange", False),
                                  ("competition", "exchange", True)])
        for mine, theirs, same in turned:
            for weight in (1.0, 3.0):
                self.assertEqual(self.cell(mine, theirs, same, "table", weight),
                                 (weight, 0.0, 0.0))
                self.assertEqual(self.cell(mine, theirs, same, "table_hostile", weight),
                                 (-weight, 0.0, 0.0))
        # the transcription says the same: two keys of the table, the sign of R_j
        changed = {key for key in TABLE_L2 if TABLE_L2[key] != TABLE_L2_HOSTILE[key]}
        self.assertEqual(changed, {("competition", "cooperation"), ("competition", "exchange")})
        self.assertEqual(set(TABLE_L2), set(TABLE_L2_HOSTILE))

    def test_the_competition_row_is_then_the_conflict_row_but_for_one_cell(self):
        """-R_j against every partner, like conflict -- except against a coercing partner
        with the robot's own destination, where the table's +R_j stands as printed."""
        differ = [(theirs, same) for theirs in TYPES + ("none", "hinder") for same in (False, True)
                  if self.cell("competition", theirs, same, "table_hostile")
                  != self.cell("conflict", theirs, same, "table_hostile")]
        self.assertEqual(differ, [("coercion", True)])
        self.assertEqual(self.cell("competition", "coercion", True, "table_hostile"),
                         (1.0, 0.0, 0.0))
        self.assertEqual(self.cell("competition", "coercion", False, "table_hostile"),
                         (-1.0, 0.0, 0.0))
        # as printed, the row is the conflict row against conflict and competition only
        differ = [theirs for theirs in TYPES
                  if self.cell("competition", theirs, False, "table")
                  != self.cell("conflict", theirs, False, "table")]
        self.assertEqual(differ, ["cooperation", "exchange"])

    def test_level_1_is_the_tables(self):
        """The column l = 1 is not touched: -r(g_j), the conflict entry, and no race."""
        for goal in G.SOCIAL_GOALS:
            self.assertEqual(G.level1_term(G.parse_goal(goal), weight=3.0,
                                           competition="table_hostile"),
                             G.level1_term(G.parse_goal(goal), weight=3.0, competition="table"),
                             goal)
        self.assertEqual(G.level1_term(G.COMPETITION, competition="table_hostile"),
                         (-1.0, 0.0, 0.0))
        self.assertFalse(G.races(G.COMPETITION, competition="table_hostile"))
        self.assertEqual(G.lead_weight("table_hostile"), 0.0)

    def test_the_default_of_the_coefficients_is_the_table_as_printed(self):
        """``level2_coefficients`` without ``competition`` is Table I: nothing that calls
        it the old way sees the turned cell."""
        for mine, theirs, same in self.CELLS:
            self.assertEqual(G.level2_coefficients(G.parse_goal(mine), G.parse_goal(theirs),
                                                   same, exchange_share=0.3),
                             self.cell(mine, theirs, same, "table"), (mine, theirs, same))
            for race in ("race_then_hinder", "race"):       # no cell of theirs is turned
                self.assertEqual(self.cell(mine, theirs, same, race),
                                 self.cell(mine, theirs, same, "table"), (mine, theirs, same))
            # asked for its cells, a rival is given the hostile table's (its term uses
            # one of them, the cell against a cooperative partner: RivalTests)
            self.assertEqual(self.cell(mine, theirs, same, "rival"),
                             self.cell(mine, theirs, same, "table_hostile"),
                             (mine, theirs, same))

    def test_describe_says_which_cell_was_turned(self):
        for partner in (G.COOPERATION, G.EXCHANGE):
            text = G.describe(G.COMPETITION, 2, partner, competition="table_hostile")
            self.assertIn("w (-1 R_j)", text)
            self.assertIn("the sign turned: table_hostile", text)
            self.assertIn("w (+1 R_j)  [Table I, l > 1]",
                          G.describe(G.COMPETITION, 2, partner, competition="table"))
        for partner in (G.COMPETITION, G.CONFLICT, G.HINDER, G.NONE):
            text = G.describe(G.COMPETITION, 2, partner, competition="table_hostile")
            self.assertIn("w (-1 R_j)  [Table I, l > 1]", text)
            self.assertEqual(text, G.describe(G.COMPETITION, 2, partner, competition="table"))
        self.assertIn("+1 R_j if g_i = g_j, else -1 R_j",
                      G.describe(G.COMPETITION, 2, G.COERCION, competition="table_hostile"))
        self.assertNotIn("turned", G.describe(G.COMPETITION, 2, G.COERCION,
                                              competition="table_hostile"))
        # level 1, and every other goal: as under the table
        self.assertEqual(G.describe(G.COMPETITION, 1, G.COOPERATION, competition="table_hostile"),
                         G.describe(G.COMPETITION, 1, G.COOPERATION, competition="table"))
        self.assertEqual(G.describe(G.COOPERATION, 2, G.COMPETITION, competition="table_hostile"),
                         G.describe(G.COOPERATION, 2, G.COMPETITION, competition="table"))
        self.assertNotIn("race", G.describe(G.COMPETITION, 2, G.COOPERATION,
                                            competition="table_hostile"))
        # a partner whose goal is inferred: the term is a mix of cells, and the record
        # says that one of them was turned -- for a competitor, under this reading only
        inferred = G.describe(G.COMPETITION, 2, None, competition="table_hostile")
        self.assertIn("depends on the partner's social goal", inferred)
        self.assertIn("-R_j against a cooperative or exchanging one: table_hostile", inferred)
        for goal, mode in ((G.COMPETITION, "table"), (G.CONFLICT, "table_hostile"),
                           (G.COOPERATION, "table_hostile")):
            self.assertNotIn("table_hostile", G.describe(goal, 2, None, competition=mode))
        self.assertEqual(G.describe(G.COMPETITION, 2, None, competition="table"),
                         "competition: Table I, l > 1 (depends on the partner's social goal)")


class HinderTests(unittest.TestCase):
    def test_hinder_is_minus_the_partners_goal_reward_at_every_level(self):
        """-w r(g_j): no partner reward R_j (so nothing recursive, and no partner action
        cost), and nothing of the robot's own goal."""
        for w in (0.8, 1.0, 30.0):
            self.assertEqual(G.level1_coefficients(G.HINDER, weight=w), (-w, 0.0))
            for partner in G.SOCIAL_GOALS:
                for same in (False, True):
                    self.assertEqual(
                        G.level2_coefficients(G.HINDER, G.parse_goal(partner), same, weight=w),
                        (0.0, 0.0, -w), (partner, same))

    def test_hinder_at_level_1_is_table_one_conflict(self):
        self.assertEqual(G.level1_coefficients(G.HINDER), G.level1_coefficients(G.CONFLICT))

    def test_conflict_above_level_1_is_recursive_and_hinder_is_not(self):
        """Why hinder exists: conflict at l > 1 is -R_j, whatever R_j holds."""
        for partner in TYPES:
            alpha, beta, nu = G.level2_coefficients(G.CONFLICT, G.parse_goal(partner), False)
            self.assertEqual((alpha, beta, nu), (-1.0, 0.0, 0.0))
            self.assertEqual(G.level2_coefficients(G.HINDER, G.parse_goal(partner), False)[0], 0.0)

    def test_a_hinder_partner_is_read_as_a_conflict_partner(self):
        for mine in TYPES:
            for same in (False, True):
                self.assertEqual(
                    G.level2_coefficients(G.parse_goal(mine), G.HINDER, same, exchange_share=0.3),
                    G.level2_coefficients(G.parse_goal(mine), G.CONFLICT, same,
                                          exchange_share=0.3))

    def test_reference_transcription_agrees_on_hinder(self):
        for partner in TYPES + ("none", "hinder"):
            self.assertEqual(G.level2_coefficients(G.HINDER, G.parse_goal(partner), False,
                                                   weight=2.0),
                             table_l2("hinder", partner, False, 0.25, weight=2.0))
        self.assertEqual(G.level1_coefficients(G.HINDER, weight=2.0), table_l1("hinder", 2.0))


class WeightAndNameTests(unittest.TestCase):
    def test_weights_scale_every_coefficient(self):
        for mine, theirs, same in itertools.product(TYPES, TYPES, (False, True)):
            base = G.level2_coefficients(G.parse_goal(mine), G.parse_goal(theirs), same,
                                         exchange_share=0.25)
            got = G.level2_coefficients(G.parse_goal(mine), G.parse_goal(theirs), same,
                                        weight=3.0, exchange_share=0.25)
            self.assertEqual(got, tuple(3.0 * v + 0.0 for v in base))
        self.assertEqual(G.level1_coefficients(G.COOPERATION, weight=0.8), (0.8, 0.0))
        self.assertEqual(G.level1_coefficients(G.COERCION, weight=0.8), (0.0, 0.8))

    def test_no_negative_zero(self):
        """Coefficients are dictionary keys of the model bank: -0.0 must not appear."""
        for goal in range(len(G.SOCIAL_GOALS)):
            for value in G.level1_coefficients(goal, weight=2.0):
                self.assertNotEqual(str(value), "-0.0")

    def test_names_aliases_and_errors(self):
        self.assertEqual(G.parse_goal(None), G.NONE)
        self.assertEqual(G.parse_goal("none"), G.NONE)
        self.assertEqual(G.parse_goal(" Cooperative "), G.COOPERATION)
        self.assertEqual(G.parse_goal("competitive"), G.COMPETITION)
        self.assertEqual(G.parse_goal("hindering"), G.HINDER)
        self.assertEqual(G.parse_goal(G.EXCHANGE), G.EXCHANGE)
        self.assertEqual([G.goal_name(G.parse_goal(name)) for name in G.SOCIAL_GOALS],
                         list(G.SOCIAL_GOALS))
        for bad in ("barter", 99, -1):
            with self.assertRaises(ValueError):
                G.parse_goal(bad)

    def test_signs(self):
        """Which goals are for the partner (+1), against it (-1), or neither."""
        signs = {name: G.goal_sign(G.parse_goal(name)) for name in G.SOCIAL_GOALS}
        self.assertEqual(signs, {"none": 0.0, "cooperation": 1.0, "conflict": -1.0,
                                 "competition": -1.0, "coercion": 0.0, "exchange": 1.0,
                                 "hinder": -1.0})

    def test_describe_names_the_term(self):
        self.assertIn("-w r(g_j)", G.describe(G.HINDER, 1))
        self.assertIn("-w r(g_j)", G.describe(G.HINDER, 2, G.COOPERATION))
        self.assertEqual(G.describe(G.COOPERATION, 0), "no social term")
        self.assertIn("-1 R_j", G.describe(G.CONFLICT, 2, G.COOPERATION))
        # the two cells of Table I that ask whether the destinations are the same
        self.assertIn("+1 R_j if g_i = g_j, else -1 R_j",
                      G.describe(G.COMPETITION, 2, G.COERCION, competition="table"))
        self.assertIn("+1 R_j if g_i = g_j, else -1 R_j +1 r(g_i)",
                      G.describe(G.COERCION, 2, G.COMPETITION))


class CooperatesWithTests(unittest.TestCase):
    """``Config.cooperate_with`` (since 2026-10-11): a level-2 cooperator has its cell of
    Table I, +R_j, only against a partner it takes to be cooperative, competitive or
    exchanging, and no social term against any other; "all" is the table as printed."""

    ALL = tuple(G.SOCIAL_GOALS)

    def cells(self, cooperate_with, goal="cooperation", competition="rival"):
        partners = None if cooperate_with is None else G.cooperation_partners(cooperate_with)
        return {(theirs, same): G.level2_term(G.parse_goal(goal), G.parse_goal(theirs), same,
                                              weight=2.0, exchange_share=0.3,
                                              competition=competition,
                                              cooperate_with=partners)
                for theirs in self.ALL for same in (False, True)}

    def test_the_default_and_before(self):
        self.assertEqual(PackageConfig().cooperate_with, ("cooperation", "competition",
                                                          "exchange"))
        self.assertEqual(PackageConfig().cooperate_with, COOPERATE_WITH)
        self.assertEqual(PackageConfig(**TOLD).cooperate_with, EVERY_PARTNER)
        self.assertEqual(Config().cooperate_with, EVERY_PARTNER)     # the tests of before

    def test_every_cell_against_the_reference(self):
        """Every goal against every partner goal, both destinations, four readings of
        competition, with the default partners and with "all": the engine's cells are the
        reference's (which leaves the cooperator's out by the partner's name)."""
        for rule in (COOPERATE_WITH, EVERY_PARTNER, ("hinder", "none")):
            for mode in ("table", "table_hostile", "race_then_hinder", "rival"):
                for mine, theirs, same in itertools.product(self.ALL, self.ALL, (False, True)):
                    got = G.level2_term(G.parse_goal(mine), G.parse_goal(theirs), same,
                                        weight=2.0, exchange_share=0.3, competition=mode,
                                        cooperate_with=G.cooperation_partners(rule))
                    race = level2_race(mine, 2.0, mode, theirs)
                    want = ((0.0, 0.0, 0.0) if race else
                            table_l2(mine, theirs, same, 0.3, 2.0, mode, rule)) + (race,)
                    self.assertEqual(got, want, (rule, mode, mine, theirs, same))

    def test_whom_the_cooperator_cooperates_with(self):
        cells = self.cells(COOPERATE_WITH)
        for (theirs, same), cell in cells.items():
            helped = theirs in ("cooperation", "competition", "exchange")
            self.assertEqual(cell, (2.0, 0.0, 0.0, 0.0) if helped else (0.0, 0.0, 0.0, 0.0),
                             theirs)

    def test_all_is_the_table_as_printed(self):
        self.assertEqual(self.cells("all"), self.cells(None))
        self.assertEqual(self.cells(EVERY_PARTNER), self.cells(None))
        self.assertTrue(all(cell == (2.0, 0.0, 0.0, 0.0) for cell in self.cells(None).values()))

    def test_no_other_goal_is_changed(self):
        for goal in self.ALL:
            if goal == "cooperation":
                continue
            for mode in ("table", "rival"):
                self.assertEqual(self.cells(COOPERATE_WITH, goal, mode),
                                 self.cells(None, goal, mode), (goal, mode))

    def test_hinder_and_conflict_are_named_apart(self):
        """The partner's own goal is asked, before a hinder partner is read as a conflict
        partner."""
        cells = self.cells(("conflict",))
        self.assertEqual(cells["conflict", False], (2.0, 0.0, 0.0, 0.0))
        self.assertEqual(cells["hinder", False], (0.0, 0.0, 0.0, 0.0))
        cells = self.cells(("hinder",))
        self.assertEqual(cells["hinder", False], (2.0, 0.0, 0.0, 0.0))
        self.assertEqual(cells["conflict", False], (0.0, 0.0, 0.0, 0.0))

    def test_the_option(self):
        self.assertEqual(PackageConfig(cooperate_with="all").cooperate_with, EVERY_PARTNER)
        self.assertEqual(PackageConfig(cooperate_with=["exchange", "cooperative", "exchange",
                                                       "competitive"]).cooperate_with,
                         COOPERATE_WITH)                 # aliases, repeats, any order
        self.assertEqual(PackageConfig(cooperate_with="hinder").cooperate_with, ("hinder",))
        self.assertEqual(PackageConfig(cooperate_with=()).cooperate_with, ())
        for bad in (None, ("friendly",), True):
            with self.assertRaises(ValueError):
                PackageConfig(cooperate_with=bad)

    def test_names_and_folders(self):
        self.assertEqual(G.cooperation_reading(EVERY_PARTNER), "Table I")
        self.assertEqual(G.cooperation_reading(COOPERATE_WITH), "selective")
        self.assertEqual(G.cooperation_reading(("hinder",)), "with hinder")
        self.assertEqual(G.cooperation_tag("all"), "")   # the folders of before
        self.assertEqual(G.cooperation_tag(COOPERATE_WITH), "_selective")
        self.assertEqual(G.cooperation_tag(("none", "cooperation")),
                         "_coop_with_none_cooperation")
        self.assertEqual(G.cooperation_tag(()), "_coop_with_nobody")
        self.assertEqual(G.level_tag(1), "")             # a hinderer at level 1: as before
        self.assertEqual(G.level_tag(2), "_level2")

    def test_describe(self):
        partners = G.cooperation_partners(COOPERATE_WITH)
        self.assertEqual(
            G.describe(G.COOPERATION, 2, None, cooperate_with=partners),
            "cooperation: w (+1 R_j) against a partner it takes for cooperation, competition, "
            "exchange; no social term against any other  [cooperates with cooperation, "
            "competition, exchange only]")
        self.assertEqual(G.describe(G.COOPERATION, 2, G.HINDER, cooperate_with=partners),
                         "cooperation vs hinder: no social term  [cooperates with cooperation, "
                         "competition, exchange only]")
        self.assertEqual(G.describe(G.COOPERATION, 2, G.EXCHANGE, cooperate_with=partners),
                         G.describe(G.COOPERATION, 2, G.EXCHANGE))
        self.assertEqual(G.describe(G.COOPERATION, 1, None, cooperate_with=partners),
                         G.describe(G.COOPERATION, 1, None))      # level 1: as printed
        every = G.cooperation_partners("all")
        self.assertEqual(G.describe(G.COOPERATION, 2, None, cooperate_with=every),
                         G.describe(G.COOPERATION, 2, None))


if __name__ == "__main__":
    unittest.main()
