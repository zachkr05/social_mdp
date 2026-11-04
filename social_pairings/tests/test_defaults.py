"""The defaults since 2026-10-08: nobody is told its partner's social goal.

Every level-2 robot infers its partner's social goal, and its destination, from what the
partner does, holding seven social goals possible (the paper's five, none and hinder); an
exchanger at level 1 is the copy of its partner's goal of the paper's text, not multiplied by
the weight; competition is read as a rival, which judges whom it faces by its belief.
``TOLD`` gives the defaults of before back.  These tests import from the package itself; the
older tests run under ``TOLD`` (``told.py``).  The six-object rollouts only with
SOCIAL_PAIRINGS_SLOW=1.
"""
import dataclasses
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from social_pairings import (HYPOTHESES, PAPER_GOALS, TOLD, exchange_scene,
                             six_object_scene)
from social_pairings import goals as G
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.models import SocialMDP
from social_pairings.render import _Canvas, render_gif
from social_pairings.serialize import to_json

from .hands_before import Config, default_levels, levels_used, run_pairing

# (``Config`` here is the package's with the rule of the hands until 2026-10-18, under which
# this module's rollouts were pinned -- hands_before.py; PackageConfig the package's own)
from social_pairings import Config as PackageConfig  # noqa: E402  isort: skip
from social_pairings.util import configure_threads

from .reference import PASS, make_world

SLOW = os.environ.get("SOCIAL_PAIRINGS_SLOW", "") not in ("", "0")
NAMES = ("agent_1", "agent_2")
# the exchange scene's rule: an object changes hands only by being put down in the doorway
PUT_DOWN = dict(set_down_for_partner=False, hand_over=False)


class TheDefaults(unittest.TestCase):
    def test_what_a_robot_is_told(self):
        cfg = Config()
        self.assertEqual(cfg.partner_social_goal, "inferred")
        self.assertEqual(cfg.social_hypotheses, ("cooperation", "conflict", "competition",
                                                 "coercion", "exchange", "none", "hinder"))
        self.assertEqual(cfg.social_hypotheses, HYPOTHESES)
        self.assertEqual(set(cfg.social_hypotheses), set(G.SOCIAL_GOALS))   # every goal there is
        self.assertEqual(cfg.exchange_level1, "copy_unweighted")
        self.assertEqual(cfg.competition, "rival")
        self.assertEqual((cfg.idle_hypothesis, cfg.physical_level), (True, 0))   # since noon
        self.assertEqual(cfg.partner_object, "inferred")         # since 2026-10-09
        self.assertEqual(cfg.cooperate_with, ("cooperation", "competition", "exchange"))
        self.assertEqual(cfg.level_free_level, 2)                # since 2026-10-11

    def test_told_gives_back_the_defaults_of_before(self):
        old, new = PackageConfig(**TOLD), PackageConfig()
        self.assertEqual((old.partner_social_goal, old.social_hypotheses, old.exchange_level1,
                          old.competition, old.partner_object, old.cooperate_with,
                          old.level_free_level),
                         ("known", PAPER_GOALS, "none", "race_then_hinder", "known",
                          G.SOCIAL_GOALS, 1))
        changed = {f.name for f in dataclasses.fields(PackageConfig)
                   if getattr(old, f.name) != getattr(new, f.name)}
        self.assertEqual(changed, set(TOLD))

    def test_default_levels(self):
        """Every robot with a social goal reasons at level 2 -- a hinderer too, since
        2026-10-11 (at level 1 until then: ``TOLD``, ``level_free_level=1``) -- a robot with a
        physical goal only (no social goal, or weight 0) at level 0 -- at level 2 until
        2026-10-08 -- and a level that is asked for is used, a hinderer's included."""
        self.assertEqual(default_levels(("competition", "hinder")), (2, 2))
        self.assertEqual(default_levels(("competition", "hinder"), Config(level_free_level=1)),
                         (2, 1))
        self.assertEqual(default_levels(("competition", "hinder"), Config(**TOLD)), (1, 1))
        self.assertEqual(levels_used(("hinder", "competition"), (2, 2)), (2, 2))
        self.assertEqual(levels_used(("hinder", "competition"), (2, 2), Config(**TOLD)), (1, 2))
        self.assertEqual(levels_used(("hinder", "none"), (1, 2)), (1, 2))
        self.assertEqual(default_levels(("none", "exchange")), (0, 2))
        self.assertEqual(default_levels(("none", "exchange"), Config(**TOLD)), (0, 2))
        self.assertEqual(default_levels(("cooperation", "hinder"), weights=(0.0, 1.0)), (0, 2))
        self.assertEqual(default_levels(("none", "exchange"), Config(physical_level=2)), (2, 2))
        self.assertEqual(levels_used(("none", "none"), (2, 2)), (2, 2))
        with self.assertRaises(ValueError):
            Config(physical_level=3)

    def test_a_record_of_before_says_what_it_was_made_with(self):
        """A record written before a field of Config existed was made with the field's value
        of then (``ADDED_FIELDS``): ``written_with`` fills it in, so that ``--resume`` of the
        example scripts compares a record of before with the options asked for."""
        then = Config(physical_level=2, partner_object="known", cooperate_with="all",
                      level_free_level=1, hinder_own_goal=True, belief_stages=False,
                      belief_factored=False, value_maps=False, physical_likelihood="level0",
                      belief_detail=False)
        saved = to_json(dataclasses.asdict(then))
        # (a device of before, which Config no longer takes -- BELIEF_DEVICES; the records
        # of before were made with it)
        saved["child_conditioning"] = True
        old = {key: value for key, value in saved.items() if key not in ADDED_FIELDS}
        self.assertEqual(written_with(old), saved)
        now = to_json(dataclasses.asdict(PackageConfig()))
        self.assertEqual({key for key in now if written_with(old).get(key) != now[key]},
                         {"physical_level", "partner_object", "cooperate_with",
                          "level_free_level", "hinder_own_goal", "belief_stages",
                          "belief_factored", "value_maps", "automatic_hands", "pick_steps",
                          "pick_cost_rate", "use_cost_rate", "child_conditioning",
                          "physical_likelihood", "belief_detail"})
        self.assertEqual(written_with(now), now)          # what a record holds is kept

    def test_the_models_a_level_2_robot_judges_its_partner_by(self):
        """Seven goals held possible, seven models per physical goal -- an object and a
        destination, both inferred (since 2026-10-09).  Until 2026-10-14 conflict and hinder
        were the same robot at level 1 (-w r(g_j)), six models; the hinderer without a goal
        of its own (-w r(g_j) - r(g_i)) is a model of its own.  A rival at level 1 is not
        told whom it faces and races; an exchanger at level 1 has the copy n r(g_j) whatever
        its weight."""
        cfg = Config(num_threads=2, resolution=1.0, **PUT_DOWN)
        w, goal_ids = make_world(PASS, cfg)
        weights = (3.0, 2.0)
        mdp = SocialMDP(w, goal_ids, (2, 2), (G.EXCHANGE, G.COMPETITION), weights, cfg,
                        configure_threads(2))
        self.assertEqual(mdp.hyp_goals, (tuple(G.parse_goal(g) for g in HYPOTHESES),) * 2)
        terms = {G.goal_name(t): mdp.level1_term(1, t) for t in mdp.hyp_goals[0]}
        self.assertEqual(terms, {"cooperation": (2.0, 0.0, 0.0), "conflict": (-2.0, 0.0, 0.0),
                                 "competition": (0.0, 0.0, 2.0), "coercion": (0.0, 2.0, 0.0),
                                 "exchange": (0.25, 0.0, 0.0), "none": (0.0, 0.0, 0.0),
                                 "hinder": (-2.0, -1.0, 0.0)})
        self.assertEqual(mdp.PG, w.M * w.K)                  # every object, its own too
        for g in range(mdp.PG):
            self.assertEqual(len(mdp.slots[(1, 1, g)].model_keys), 7)
        slot = mdp.slots[(0, 2, mdp.own_goal[0])]
        self.assertEqual(slot.hyp_goal.size, w.M * w.K * 7)
        self.assertTrue(np.allclose(mdp.planning_weights(slot), 1.0 / (w.M * w.K * 7)))
        self.assertTrue(np.allclose(mdp.about[1].phys(), 1.0 / (w.M * w.K)))
        self.assertTrue(np.allclose(mdp.about[1].social(), 1.0 / 7))

    def test_the_rival_judges_whom_it_faces_by_its_belief(self):
        """Its level-2 term: -R_j against a cooperator, the race against every other goal
        held possible -- the belief weighs the two."""
        cfg = Config(num_threads=2, resolution=1.0, **PUT_DOWN)
        w, goal_ids = make_world(PASS, cfg)
        mdp = SocialMDP(w, goal_ids, (2, 2), (G.COMPETITION, G.EXCHANGE), (3.0, 2.0), cfg,
                        configure_threads(2))
        cells = {G.goal_name(t): mdp.level2_term(0, G.COMPETITION, t, False)
                 for t in mdp.hyp_goals[0]}
        self.assertEqual(cells.pop("cooperation"), (-3.0, 0.0, 0.0, 0.0))
        self.assertEqual(set(cells.values()), {(0.0, 0.0, 0.0, 3.0)})

    def test_told_the_example_of_the_package_has_its_number_of_before(self):
        """The package's example (its docstring, weights 2 / 2): 5 / - told, 7 / - with the
        social goals inferred, the partner's object known, Table I's cooperator and the
        hinderer at level 1 (the defaults of 2026-10-08; with every object a candidate the
        six-object scene takes more memory than this box has).  With the cooperator of
        2026-10-11 and the hinderer still at level 1, - / -: the hinderer keeps the
        cooperator's object from transition 2, and the cooperator no longer delivers the
        hinderer's (it puts it back at 7 and takes it again at 13).  With the hinderer at
        level 2 as well (the defaults of 2026-10-11), 5 / -: the cooperator delivers the
        hinderer's object at 5; the hinderer takes the cooperator's at 3 and keeps it.  Those
        three runs have the hinderer of before, with a goal of its own.  Under the defaults
        of now, without it (2026-10-14), the same events -- but the hinderer has no goal to
        be done: - / -.

        Since 2026-10-16 (five hand actions in place of "use") the last two differ: with the
        hinderer at level 1 the cooperator keeps the hinderer's object from transition 2
        (with "use" it put it back at 7 and took it again at 13); under the defaults of now
        it takes the hinderer's object at 2 and keeps it, and never sets it down (with "use"
        it did at 5).  The two posteriors of 2026-10-17 leave those two as they are; the
        run of 2026-10-08 keeps the one posterior of then (``belief_stages=False``; with two
        it is 8 / -).

        Under the belief update of 2026-10-30 (Eqs. 5-6 and nothing else): told, 5 / - as
        before; the run of 2026-10-08, 26 / - -- the cooperator, holding the hinderer's
        object from 2 and learning nothing of its destination from a hinderer that only
        wants its object back, tries three wrong candidates (9, 13, 19) before the right
        one (26); with the hinderer at level 1, 5 / - -- the hinderer waits on the
        cooperator's object instead of taking it at 2, the cooperator delivers the
        hinderer's at 5, and the hinderer picks the cooperator's up at 6; under the defaults
        of now the hinderer takes the cooperator's object at 2 and the cooperator the
        hinderer's at 4, and neither lets go: - / -."""
        pair, weights = ("hinder", "cooperation"), (2.0, 2.0)
        told = run_pairing(six_object_scene(), pair, weights=weights,
                           cfg=Config(num_threads=2, **TOLD))
        self.assertEqual(told["done_at"], (5, None))
        inferred = run_pairing(six_object_scene(), pair, weights=weights,
                               cfg=Config(num_threads=2, partner_object="known",
                                          cooperate_with="all", level_free_level=1,
                                          hinder_own_goal=True, belief_stages=False))
        self.assertEqual(inferred["done_at"], (26, None))
        self.assertEqual(inferred["config"]["partner_social_goal"], "inferred")
        self.assertEqual(inferred["config"]["partner_object"], "known")
        level1 = run_pairing(six_object_scene(), pair, weights=weights,
                             cfg=Config(num_threads=2, partner_object="known",
                                        level_free_level=1, hinder_own_goal=True))
        self.assertEqual(level1["done_at"], (5, None))
        self.assertEqual([e[:4] for e in level1["events"]],
                         [(2, "picked up", 1, 0), (5, "set down", 1, 0), (6, "picked up", 0, 1)])
        now = run_pairing(six_object_scene(), pair, weights=weights,
                          cfg=Config(num_threads=2, partner_object="known"))
        self.assertEqual(now["levels_used"], (2, 2))
        self.assertEqual(tuple(now["own_goal"]), (False, True))
        self.assertEqual(now["done_at"], (None, None))
        self.assertEqual([e[:4] for e in now["events"]],
                         [(2, "picked up", 1, 0), (4, "picked up", 0, 1)])


class TwoExchangersTellEachOtherApart(unittest.TestCase):
    """The exchange scene under the defaults of 2026-10-08 (the partner's object known: with
    its six objects every one a candidate, the scene needs more memory than this box has):
    each robot picks its partner's object up, puts it down in the doorway, picks its own up
    from there and delivers it."""

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(exchange_scene(), ("exchange", "exchange"),
                                 weights=(30.0, 30.0),
                                 cfg=Config(num_threads=2, partner_object="known", **PUT_DOWN))

    def test_the_exchange(self):
        r = self.result
        self.assertEqual(r["done_at"], (17, 16))
        self.assertEqual([e[:4] for e in r["events"]],
                         [(2, "picked up", 1, 0), (3, "picked up", 0, 1),
                          (7, "put down", 1, 0), (10, "put down", 0, 1),
                          (11, "picked up", 0, 0), (12, "picked up", 1, 1),
                          (16, "set down", 1, 1), (17, "set down", 0, 0)])
        self.assertEqual(tuple(r["config"]["social_hypotheses"]), HYPOTHESES)
        for belief in r["final_social_belief"]:
            self.assertEqual(len(belief), len(G.SOCIAL_GOALS))
            self.assertAlmostEqual(float(np.sum(belief)), 1.0, places=9)

    def test_the_figure_has_room_for_seven_goals(self):
        canvas = _Canvas(self.result, NAMES, "relative", "", 25, 3)
        try:
            self.assertEqual([G.SOCIAL_GOALS[h] for h in canvas.story.hypotheses],
                             list(HYPOTHESES))
            canvas.draw(9)
            canvas.fig.canvas.draw()
            renderer = canvas.fig.canvas.get_renderer()
            labels = [t for t in canvas.ax_social.get_xticklabels() if t.get_text()]
            self.assertEqual([t.get_text() for t in labels],
                             ["coop.", "confl.", "comp.", "coerc.", "exch.", "none", "hinder"])
            boxes = [t.get_window_extent(renderer) for t in labels]
            for a, b in zip(boxes, boxes[1:]):
                self.assertLess(a.x1, b.x0)              # no tick label runs into the next
            panel = canvas.ax_social.get_window_extent()
            self.assertLessEqual(boxes[-1].x1, canvas.fig.bbox.x1)
            self.assertLess(canvas.ax_belief.get_window_extent().x1, panel.x0)
        finally:
            canvas.close()

    def test_the_gif(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "exchange.gif"
            render_gif(self.result, path, names=NAMES, fps=3.0)
            with Image.open(path) as gif:
                self.assertEqual(gif.n_frames, len(self.result["trace"]) + 1)


@unittest.skipUnless(SLOW, "set SOCIAL_PAIRINGS_SLOW=1")
class TheSixObjectScene(unittest.TestCase):
    """Pairings of the matrix of the README ("Every pairing, goals inferred"), weights 30,
    under the defaults of 2026-10-08 noon (a physical goal only at level 0, the idle
    hypothesis on, the partner's object known, Table I's cooperator, a hinderer at level 1);
    the matrix itself was made before (– / –, 6 / 13, – / – where these are 9 / 11, 8 / 17,
    17 / –)."""

    CASES = {("none", "none"): (9, 11),                  # two plain MDPs (level 0)
             ("cooperation", "cooperation"): (8, 17),
             ("cooperation", "exchange"): (5, 8),
             ("coercion", "exchange"): (7, 7),
             ("exchange", "exchange"): (17, None),
             ("competition", "cooperation"): (5, None),
             ("hinder", "cooperation"): (7, None)}

    def test_pairings(self):
        for pair, done in self.CASES.items():
            with self.subTest(pair=pair):
                r = run_pairing(six_object_scene(), pair, weights=(30.0, 30.0),
                                cfg=Config(num_threads=2, partner_object="known",
                                           cooperate_with="all", level_free_level=1))
                self.assertEqual(r["done_at"], done)


if __name__ == "__main__":
    unittest.main()
