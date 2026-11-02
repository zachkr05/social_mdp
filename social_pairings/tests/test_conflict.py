"""Task conflict (``Config.conflict="task"``, an option since 2026-11-01; the default is
Table I's conflict, "table", as before -- the task conflict was the default for a few hours).

Table I's conflict above level 1 is ``-R_j``, and ``R_j`` holds the partner's own term about
the robot: against a cooperator ``+w_j r~(g_i)``, so the conflict robot's own goal is worth
``(1 - w_i w_j b~(g_i)) r(g_i)`` -- nothing at w = 1 once the cooperator is sure of it, and in
conflict / cooperation the conflict robot put its object back down.  The task conflict keeps
the table's entry at l = 1 at every level, against every partner:

    R_i = r(g_i) - w r(g_j) - c(a_i).

"table" is the table as printed (the default; every record of before).  Checked here: the
terms, by the engine and by the reference; the option and the records of before; the levels
and the names; every array of levels 0-2 against brute force with a level-2 conflict robot
under the task reading; the example script's ``--resume``, which keeps (and copies from the
folder of the other reading, either way) every record without a level-2 conflict robot and
no other; and that such a pairing rolls out alike under both readings while a conflict
pairing does not (27 s).
"""
import dataclasses
import importlib.util
import tempfile
import unittest
from pathlib import Path

from social_pairings import TOLD, Config, Scene, run_pairing
from social_pairings import goals as G
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.constants import ACTION_NAMES
from social_pairings.pairing import default_levels
from social_pairings.render import goal_label
from social_pairings.serialize import load_result, save_result, to_json

from .reference import table_l2
from .test_models import TINY, tight_now

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = (0.5, 1.0, 3.0, 30.0)


class Terms(unittest.TestCase):
    def test_minus_w_r_gj_against_every_partner(self):
        for w in WEIGHTS:
            for t in range(len(G.SOCIAL_GOALS)):
                for same in (False, True):
                    self.assertEqual(G.level2_term(G.CONFLICT, t, same, weight=w,
                                                   conflict="task"), (0.0, 0.0, -w, 0.0))
                    self.assertEqual(table_l2("conflict", G.goal_name(t), same, 0.25, w,
                                              conflict="task"), (0.0, 0.0, -w))
        self.assertEqual(G.level2_term(G.CONFLICT, G.COOPERATION, False, weight=0.0,
                                       conflict="task"), (0.0, 0.0, 0.0, 0.0))

    def test_the_table_is_the_default_of_the_functions(self):
        """-w R_j as printed, against every partner (a hinder partner read as conflict)."""
        for t in range(len(G.SOCIAL_GOALS)):
            self.assertEqual(G.level2_term(G.CONFLICT, t, False, weight=2.0),
                             G.level2_term(G.CONFLICT, t, False, weight=2.0, conflict="table"))
            if t != G.NONE:
                self.assertEqual(G.level2_term(G.CONFLICT, t, False, weight=2.0),
                                 (-2.0, 0.0, 0.0, 0.0))

    def test_level_1_and_every_other_goal_are_untouched(self):
        for mode in G.COMPETITION_FORMS:
            for goal in range(len(G.SOCIAL_GOALS)):
                if goal == G.CONFLICT:
                    continue
                for t in range(len(G.SOCIAL_GOALS)):
                    for same in (False, True):
                        self.assertEqual(
                            G.level2_term(goal, t, same, weight=2.0, competition=mode,
                                          conflict="task"),
                            G.level2_term(goal, t, same, weight=2.0, competition=mode))
        # level 1 has no reading: -w r(g_j) either way, the term of task conflict
        self.assertEqual(G.level1_term(G.CONFLICT, weight=2.0), (-2.0, 0.0, 0.0))

    def test_it_is_the_reward_of_a_hinderer_with_a_goal_of_its_own(self):
        """r(g_i) - w r(g_j) - c(a_i): the hinderer until 2026-10-14 (hinder_own_goal=True).
        The hinderer of now has no goal of its own: r(g_i) taken away."""
        for t in range(len(G.SOCIAL_GOALS)):
            self.assertEqual(G.level2_term(G.CONFLICT, t, False, weight=3.0, conflict="task"),
                             G.level2_term(G.HINDER, t, False, weight=3.0, hinder_own_goal=True))
            self.assertEqual(G.level2_term(G.HINDER, t, False, weight=3.0,
                                           hinder_own_goal=False), (0.0, -1.0, -3.0, 0.0))

    def test_level_free(self):
        self.assertTrue(G.level_free(G.CONFLICT, conflict="task"))
        self.assertFalse(G.level_free(G.CONFLICT, conflict="table"))
        self.assertFalse(G.level_free(G.CONFLICT))

    def test_when_the_reading_matters(self):
        self.assertTrue(G.conflict_matters(("conflict", "cooperation"), (2, 2)))
        self.assertTrue(G.conflict_matters(("hinder", "conflict"), (2, 2)))
        self.assertTrue(G.conflict_matters((G.CONFLICT, G.NONE), (2, 0)))
        self.assertFalse(G.conflict_matters(("conflict", "cooperation"), (1, 2)))
        self.assertFalse(G.conflict_matters(("cooperation", "hinder"), (2, 2)))
        self.assertFalse(G.conflict_matters(("none", "none"), (2, 2)))

    def test_describe_and_label(self):
        task = G.describe(G.CONFLICT, 2, conflict="task")
        self.assertIn("-1 r(g_j)", task)
        self.assertNotIn("R_j", task)
        self.assertIn("task conflict", task)
        self.assertIn("vs cooperation", G.describe(G.CONFLICT, 2, G.COOPERATION,
                                                   conflict="task"))
        self.assertIn("-1 R_j", G.describe(G.CONFLICT, 2, G.COOPERATION))
        self.assertEqual(G.describe(G.CONFLICT, 1, conflict="task"), G.describe(G.CONFLICT, 1))
        self.assertEqual(goal_label("conflict", "rival", None, "task"), "conflict (task)")
        self.assertEqual(goal_label("conflict", "rival", None, "table"), "conflict")
        self.assertEqual(goal_label("conflict"), "conflict")
        self.assertEqual(goal_label("hinder", None, None, "task"), "hinder")


class Option(unittest.TestCase):
    def test_the_option(self):
        self.assertEqual(Config().conflict, "table")             # Table I's, as before
        self.assertEqual(Config(**TOLD).conflict, "table")
        self.assertNotIn("conflict", TOLD)                         # nothing to restore
        self.assertEqual(ADDED_FIELDS["conflict"], "table")
        self.assertEqual(written_with({})["conflict"], "table")      # a record of before
        with self.assertRaises(ValueError):
            Config(conflict="Table I")

    def test_levels(self):
        """Level 2 by default either way (Config.level_free_level is 2); with
        level_free_level=1 a task conflict robot is level-free, as a hinderer is, and a
        conflict robot of the table is not."""
        for reading in ("table", "task"):
            self.assertEqual(default_levels(("conflict", "cooperation"),
                                            Config(conflict=reading)), (2, 2))
        self.assertEqual(default_levels(("conflict", "cooperation"), Config(**TOLD)), (2, 2))
        self.assertEqual(default_levels(("conflict", "cooperation"),
                                        Config(level_free_level=1, conflict="task")), (1, 2))
        self.assertEqual(default_levels(("conflict", "cooperation"),
                                        Config(level_free_level=1)), (2, 2))


class AgainstBruteForce(unittest.TestCase):
    def test_a_level_2_conflict_robot(self):
        """Every array of levels 0-2 against the reference: a level-2 conflict robot paid
        -w r(g_j) against each of the seven social goals it holds possible for a level-2
        cooperator (everything inferred; the weights apart, so that the term is seen to
        hold the conflict robot's alone)."""
        from .test_models import ModelsAgainstBruteForce
        case = ModelsAgainstBruteForce(methodName="compare")
        case.compare(tight_now(conflict="task"), (2, 2), ("conflict", "cooperation"),
                     (1.5, 2.0), seed=301)


def _script():
    """The example script, as a module (its main is not run)."""
    path = ROOT / "examples" / "run_object_inference.py"
    spec = importlib.util.spec_from_file_location("run_object_inference_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(pair, levels_used, reading=None, **options) -> dict:
    """A record made under the task conflict (``reading="task"``) or, by default, as one made
    before 2026-11-01 holds it: no "conflict" in its config (the table's)."""
    config = to_json(dataclasses.asdict(Config(**{"conflict": reading or "table", **options})))
    if reading is None:
        del config["conflict"]
    return dict(config=config, social_goals=list(pair), levels=[2, 2],
                levels_used=list(levels_used), actions=list(ACTION_NAMES), trace=[])


class Resume(unittest.TestCase):
    """``--resume``: a record without a level-2 conflict robot holds under either reading;
    the folder of either reading takes it, with its videos, from the other's."""

    @classmethod
    def setUpClass(cls):
        cls.script = _script()

    def test_what_is_kept(self):
        unkept, now = self.script.unkept, Config(conflict="task")
        self.assertIsNone(unkept(_record(("cooperation", "hinder"), (2, 2)), now,
                                 ("cooperation", "hinder"), None))
        self.assertIsNone(unkept(_record(("conflict", "cooperation"), (1, 2)), now,
                                 ("conflict", "cooperation"), None))
        self.assertIn("another configuration",
                      unkept(_record(("conflict", "cooperation"), (2, 2)), now,
                             ("conflict", "cooperation"), None))
        self.assertIsNone(unkept(_record(("conflict", "cooperation"), (2, 2)),
                                 Config(conflict="table"), ("conflict", "cooperation"), None))
        self.assertIn("another configuration",                     # any other option counts
                      unkept(_record(("cooperation", "hinder"), (2, 2), inertia=1.0), now,
                             ("cooperation", "hinder"), None))
        self.assertIn("other levels", unkept(_record(("cooperation", "hinder"), (2, 2)), now,
                                             ("cooperation", "hinder"), (1, 1)))
        # the other way: records made under the task conflict, asked for under the table's
        table = Config()
        self.assertIsNone(unkept(_record(("cooperation", "hinder"), (2, 2), "task"), table,
                                 ("cooperation", "hinder"), None))
        self.assertIn("another configuration",
                      unkept(_record(("cooperation", "conflict"), (2, 2), "task"), table,
                             ("cooperation", "conflict"), None))

    def test_what_is_taken_from_the_folder_of_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, now = Path(tmp) / "before", Path(tmp) / "now_taskconflict"
            before.mkdir()
            now.mkdir()
            for pair in (("cooperation", "hinder"), ("conflict", "cooperation")):
                save_result(before / f"{pair[0]}__{pair[1]}__inferred.json",
                            _record(pair, (2, 2)))
            (before / "cooperation__hinder__inferred.mp4").write_bytes(b"video")
            cfg = Config(conflict="task")
            kept = now / "cooperation__hinder__inferred.json"
            got = self.script.reuse(kept, before, cfg, ("cooperation", "hinder"), (2, 2))
            self.assertIsNotNone(got)
            self.assertEqual(load_result(kept), load_result(before / kept.name))
            self.assertEqual((now / "cooperation__hinder__inferred.mp4").read_bytes(), b"video")
            fought = now / "conflict__cooperation__inferred.json"
            self.assertIsNone(self.script.reuse(fought, before, cfg,
                                                ("conflict", "cooperation"), (2, 2)))
            self.assertFalse(fought.exists())
            self.assertIsNone(self.script.reuse(now / "none__none__inferred.json", before, cfg,
                                                ("none", "none"), (2, 2)))   # none there
            self.assertIsNone(self.script.reuse(kept, None, cfg, ("cooperation", "hinder"),
                                                (2, 2)))
            self.assertEqual(sorted(p.name for p in now.iterdir()),
                             ["cooperation__hinder__inferred.json",
                              "cooperation__hinder__inferred.mp4"])

    def test_what_the_table_takes_from_the_task_conflict(self):
        """The default folder (Table I's conflict) takes a record without a level-2 conflict
        robot from the ``_taskconflict`` folder, and no conflict pairing's."""
        with tempfile.TemporaryDirectory() as tmp:
            task, table = Path(tmp) / "x_taskconflict", Path(tmp) / "x"
            task.mkdir()
            table.mkdir()
            for pair in (("exchange", "none"), ("none", "conflict")):
                save_result(task / f"{pair[0]}__{pair[1]}__inferred.json",
                            _record(pair, (2, 2), "task"))
            cfg = Config()
            kept = table / "exchange__none__inferred.json"
            self.assertIsNotNone(self.script.reuse(kept, task, cfg, ("exchange", "none"),
                                                   (2, 2)))
            self.assertEqual(load_result(kept), load_result(task / kept.name))
            self.assertIsNone(self.script.reuse(table / "none__conflict__inferred.json", task,
                                                cfg, ("none", "conflict"), (2, 2)))
            self.assertEqual([p.name for p in table.iterdir()], [kept.name])


class Rollouts(unittest.TestCase):
    def test_only_a_conflict_pairing_changes(self):
        """Four transitions on the 3 x 3 map, both robots at level 2: cooperation / hinder is
        the same rollout, every Q value bit for bit, under both readings; conflict /
        cooperation is not (the conflict robot's Q)."""
        from .test_competition import same_rollout
        scene = Scene(name="tiny_conflict", grid=TINY["grid"], starts=TINY["starts"],
                      pickups=TINY["pickups"], destinations=TINY["destinations"],
                      candidate_cells=TINY["candidate_cells"])

        def run(pair, reading):
            cfg = Config(num_threads=2, resolution=1.0, conflict=reading)
            return run_pairing(scene, pair, levels=(2, 2), cfg=cfg, n_steps=4)

        for pair, alike in ((("cooperation", "hinder"), True),
                            (("conflict", "cooperation"), False)):
            with self.subTest(pair=pair):
                self.assertEqual(same_rollout(run(pair, "task"), run(pair, "table")), alike)


if __name__ == "__main__":
    unittest.main()
