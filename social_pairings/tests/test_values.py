"""Each robot's value at every cell, term by term (``Config.value_maps``, since 2026-10-17),
and the videos made of it.

The user: "output 2 other videos from now on which are their value iteration grid
throughout the video. And maybe a break down in the respective videos of how this cost is
calculated at each cell so I can see what is influencing it."

A robot's root model, solved as it is at a transition, is read off at every cell the robot
could stand on (its partner and the objects where they are), and split by the terms of its
reward: its own goal reward, its action costs, and each social term -- the discounted sums
under its greedy policy, which add up to its value (``SocialMDP.value_components``; against
brute force in ``test_models.py``).  The record holds the maps of every transition
(``trace[t]["value_maps"]``), and ``render.render_value_video`` draws them.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from social_pairings import Scene
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.constants import N_ACTIONS
from social_pairings.render import (_ValueCanvas, _write_video, render_value_video,
                                    render_videos, video_paths)
from social_pairings.serialize import load_result, save_result
from social_pairings.util import argmax_action

from .hands_before import Config, run_pairing
from .test_models import TINY, hostage, solved, tight_now

NAMES = ("agent_1", "agent_2")


class Option(unittest.TestCase):
    def test_config(self):
        self.assertTrue(Config().value_maps)
        self.assertFalse(ADDED_FIELDS["value_maps"])
        self.assertFalse(written_with({})["value_maps"])         # records of before
        for bad in (1, "yes", None):
            with self.assertRaises(ValueError):
                Config(value_maps=bad)


class Components(unittest.TestCase):
    """What the split promises, on the engine's own models."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = tight_now()
        cls.w, _, cls.mdp, cls.state, _ = solved(TINY, cls.cfg, (2, 2),
                                                 ("cooperation", "hinder"), (1.0, 1.0),
                                                 state=hostage, seed=4)

    def test_the_terms_add_up_to_the_value(self):
        """Every solved state: the sum of the terms is V (to the solver's tolerance and the
        rewards' single precision); the policy is the argmax of Q there; the robots never
        overlap in a solved state."""
        w, mdp = self.w, self.mdp
        for i in (0, 1):
            vc = mdp.value_components(i)
            states = np.flatnonzero(vc["policy"] >= 0)
            self.assertGreater(states.size, 100)
            self.assertTrue(w.pair_ok[states % w.NN].all())
            depth, row = mdp.root[i]
            np.testing.assert_allclose(vc["parts"][:, states].sum(axis=0),
                                       mdp.bank(depth).V[row][states], rtol=0, atol=2e-3)
            self.assertLess(vc["residual"], 2e-3)
            self.assertTrue(np.isnan(vc["total"][vc["policy"] < 0]).all())
            rng = np.random.default_rng(i)
            for s in rng.choice(states, 40, replace=False).tolist():
                q = mdp.q_values((depth, row), s, i)
                self.assertEqual(int(vc["policy"][s]), argmax_action(q))
            self.assertEqual(len(vc["names"]), len(vc["labels"]))
            self.assertEqual(vc["parts"].shape, (len(vc["names"]), w.S))

    def test_which_terms(self):
        """A cooperator: its own goal, its costs and its partner's reward; a hinderer
        without a goal of its own: its costs and -r(g_j) (its own goal reward and the term's
        -r(g_i) cancel and are left out)."""
        a, b = self.mdp.value_components(0), self.mdp.value_components(1)
        self.assertEqual(a["names"], ["own_goal", "cost", "partner_reward"])
        self.assertEqual(b["names"], ["cost", "partner_goal"])
        states = np.flatnonzero(b["policy"] >= 0)
        self.assertTrue((b["parts"][0][states] < 0).all())             # costs
        self.assertTrue((b["parts"][1][states] <= 1e-9).all())         # -r(g_j)

    def test_the_maps(self):
        """The map at a state: V at the state with the robot moved to each cell, the
        partner where it is; NaN on the partner's cell."""
        w, mdp, state = self.w, self.mdp, self.state
        for i in (0, 1):
            vm = mdp.value_maps(i, state)
            vc = mdp.value_components(i)
            self.assertEqual(vm["partner_cell"], state[1 - i])
            self.assertTrue(np.isnan(vm["total"][state[1 - i]]))
            self.assertEqual(vm["action"][state[1 - i]], -1)
            for x in range(w.N):
                if x == state[1 - i]:
                    continue
                s = w.sid((x, state[1], state[2]) if i == 0 else (state[0], x, state[2]))
                self.assertEqual(vm["total"][x], vc["total"][s])
                np.testing.assert_array_equal(vm["parts"][:, x], vc["parts"][:, s])
                self.assertEqual(vm["action"][x], vc["policy"][s])
            self.assertTrue((vm["action"] >= -1).all() and (vm["action"] < N_ACTIONS).all())
            own = vm["action"][state[i]]
            self.assertEqual(own, argmax_action(mdp.q_values(mdp.root[i], w.sid(state), i)))

    def test_a_robot_at_level_0_has_none(self):
        w, _, mdp, state, _ = solved(TINY, self.cfg, (0, 2), ("none", "cooperation"),
                                     (1.0, 1.0), seed=5)
        self.assertIsNone(mdp.value_components(0))
        self.assertIsNone(mdp.value_maps(0, state))
        self.assertIsNotNone(mdp.value_maps(1, state))


class Record(unittest.TestCase):
    scene = Scene(name="tiny", **TINY)

    @classmethod
    def setUpClass(cls):
        cls.result = run_pairing(cls.scene, ("cooperation", "competition"), weights=(1.0, 1.0),
                                 cfg=Config(num_threads=2, resolution=1.0), n_steps=5)

    def test_every_transition_holds_the_maps(self):
        r = self.result
        self.assertEqual(len(r["scene"]["cells"]), 9)
        for row in r["trace"]:
            for i in (0, 1):
                vm = row["value_maps"][i]
                self.assertEqual(vm["partner_cell"], self.cell(row["pos"][1 - i]))
                self.assertEqual(len(vm["total"]), 9)
                self.assertEqual(np.asarray(vm["parts"]).shape, (len(vm["names"]), 9))
                self.assertTrue(np.isnan(vm["total"][vm["partner_cell"]]))
                mine = self.cell(row["pos"][i])
                self.assertEqual(vm["action"][mine], row["action"][i])    # argmax, no inertia
        self.assertGreater(r["timing"]["value_maps"], 0.0)

    def cell(self, xy) -> int:
        return [tuple(c) for c in self.result["scene"]["cells"]].index(tuple(xy))

    def test_without_the_option(self):
        r = run_pairing(self.scene, ("cooperation", "competition"), weights=(1.0, 1.0),
                        cfg=Config(num_threads=2, resolution=1.0, value_maps=False), n_steps=2)
        self.assertEqual(r["trace"][0]["value_maps"], (None, None))
        self.assertLess(r["timing"]["value_maps"], 1e-3)
        self.assertEqual([row["action"] for row in r["trace"]],
                         [row["action"] for row in self.result["trace"][:2]])

    def test_the_file_and_the_videos(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_result(Path(folder) / "run.json", self.result)
            loaded = load_result(path)
            vm = loaded["trace"][0]["value_maps"][0]
            self.assertIsNone(vm["total"][vm["partner_cell"]])              # NaN -> null
            paths = render_videos(loaded, Path(folder) / "run", names=NAMES, fps=3.0,
                                  title="test", dpi=60)
            self.assertEqual([p.name for p in paths],
                             ["run.mp4", "run__values_agent_1.mp4", "run__values_agent_2.mp4"])
            self.assertEqual([p for _, p in video_paths(loaded, Path(folder) / "run",
                                                        names=NAMES)], paths)
            for p in paths:
                self.assertGreater(p.stat().st_size, 5000)
            import imageio_ffmpeg
            frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(paths[1]))
            n = len(loaded["trace"])
            self.assertEqual(frames, 2 * (n - 1) + 6)             # 1 s a transition, 3 s last
            self.assertAlmostEqual(seconds, n - 1 + 3.0, delta=0.2)
            alone = render_value_video(loaded, Path(folder) / "one.mp4", 1, names=NAMES,
                                       fps=2.0, dpi=60)
            frames, _ = imageio_ffmpeg.count_frames_and_secs(str(alone))
            self.assertEqual(frames, 2 * (n - 1) + 12)            # at 4 frames a second
            with self.assertRaises(ValueError):
                without = dict(loaded, trace=[dict(row, value_maps=(None, None))
                                              for row in loaded["trace"]])
                render_value_video(without, Path(folder) / "none.mp4", 0)

    def test_a_term_left_out_of_some_steps(self):
        """A term that pays nothing in a step is left out of that step's maps -- a competitor
        that rules out that its partner cooperates loses its R_j term (hinder / competition,
        2026-10-17, where the video stopped half way): the figure has a panel for every term
        of any step, 0 (and a note) in the steps without it, and the video is whole."""
        r = self.result
        trace = []
        for t, row in enumerate(r["trace"]):
            maps = list(row["value_maps"])
            if t % 2:
                vm = dict(maps[0])
                keep = [k for k, name in enumerate(vm["names"]) if name != "cost"]
                vm["names"] = [vm["names"][k] for k in keep]
                vm["labels"] = [vm["labels"][k] for k in keep]
                vm["parts"] = np.asarray(vm["parts"])[keep]
                maps[0] = vm
            trace.append(dict(row, value_maps=tuple(maps)))
        cut = dict(r, trace=trace)
        canvas = _ValueCanvas(cut, 0, NAMES, "", None)
        try:
            self.assertEqual(canvas.terms, list(r["trace"][0]["value_maps"][0]["names"]))
            k = canvas.terms.index("cost") + 1
            canvas.draw(1)
            self.assertEqual(canvas.axes[k].get_title(loc="right"), "pays nothing in this step")
            v = canvas._values(trace[1], k)
            total = np.asarray(trace[1]["value_maps"][0]["total"], dtype=float)
            np.testing.assert_array_equal(np.isnan(v), np.isnan(total))
            self.assertTrue((v[np.isfinite(v)] == 0.0).all())
            canvas.draw(0)
            self.assertEqual(canvas.axes[k].get_title(loc="right"), "")
            np.testing.assert_array_equal(
                canvas._values(trace[0], k),
                np.asarray(trace[0]["value_maps"][0]["parts"][k - 1], dtype=float))
        finally:
            canvas.close()
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as folder:
            path = render_value_video(cut, Path(folder) / "cut.mp4", 0, names=NAMES, dpi=60)
            frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
            self.assertEqual(frames, 2 * (len(trace) - 1) + 6)

    def test_a_video_cut_off_is_not_left(self):
        """A video whose drawing fails half way leaves no file behind (it is written under
        another name and renamed once whole), so --resume draws it again."""
        from PIL import Image

        def failing():
            for _ in range(3):
                yield Image.new("RGB", (40, 30), "white"), 0.5
            raise RuntimeError("drawing failed")

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cut.mp4"
            with self.assertRaises(RuntimeError):
                _write_video(failing(), path, 2.0)
            self.assertEqual(list(Path(folder).iterdir()), [])
            _write_video(iter([(Image.new("RGB", (40, 30), "white"), 1.0)]), path, 2.0)
            self.assertEqual([q.name for q in Path(folder).iterdir()], ["cut.mp4"])

    def test_the_figure(self):
        """One panel per term and one for V; the robot's own cell framed in its colour; the
        subtitle names the step."""
        canvas = _ValueCanvas(self.result, 0, NAMES, "", None)
        try:
            vm = self.result["trace"][2]["value_maps"][0]
            self.assertEqual(len(canvas.labels), 1 + len(vm["names"]))
            canvas.draw(2)
            self.assertIn("step 2 of 5", canvas.subtitle.get_text())
            titles = [ax.get_title(loc="left") for ax in canvas.axes if ax.get_visible()]
            self.assertEqual(titles, canvas.labels)
            self.assertTrue(canvas.labels[0].startswith("V"))
            self.assertIn("own goal", canvas.labels[1])
            image = canvas.image(2, 50)
            self.assertEqual(image.size[0] % 1, 0)
        finally:
            canvas.close()


if __name__ == "__main__":
    unittest.main()
