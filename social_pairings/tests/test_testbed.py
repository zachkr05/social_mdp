"""The testbed of 2026-11-03: the gate scene (scenes.gate_scene) and the meeting suite's six fair
tier scenes (examples/tier_scenes.py), as examples/run_object_inference.py names them.

The scenes are checked as scenes -- their cells, their hand-off cells, their geometry -- and
the gate scene's variants against the map they are drawn from; a small one is built as a
world (the gate alone as hand-off cell), so that the scene is one the engine can roll out.
Nothing is rolled out (a tier scene needs some 16 GiB, the gate scene some 60).
"""
import importlib.util
import sys
import unittest
from collections import deque
from pathlib import Path

import numpy as np

from social_pairings import Config
from social_pairings.pairing import build_world
from social_pairings.scenes import GATE_MAP, gate_scene

ROOT = Path(__file__).resolve().parents[2]
TIERS = ("near_miss", "same_room_later", "hall_crossing", "doorway_passing",
         "head_on_blocked_goal", "head_on_corridor")


def _script():
    """The example script, as a module (its main is not run)."""
    sys.path.insert(0, str(ROOT / "examples"))
    path = ROOT / "examples" / "run_object_inference.py"
    spec = importlib.util.spec_from_file_location("run_object_inference_testbed", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _free(scene):
    return {(int(x), int(y)) for x, y in np.argwhere(np.isfinite(scene.grid))}


def _steps(scene, a, b) -> int:
    """Moves from cell a to cell b on the scene's free cells (breadth first)."""
    free, seen, todo = _free(scene), {a: 0}, deque([a])
    while todo:
        c = todo.popleft()
        if c == b:
            return seen[c]
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in free and n not in seen:
                seen[n] = seen[c] + 1
                todo.append(n)
    raise AssertionError(f"{b} cannot be reached from {a}")


class Registry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = _script()

    def test_the_seven(self):
        self.assertEqual(self.script.TESTBED, ("gate",) + TIERS)
        for name in self.script.TESTBED:
            make, options = self.script.SCENES[name]
            scene = make()
            self.assertEqual(scene.name, name)
            self.assertEqual(options, {})
            free = _free(scene)
            cells = (*scene.starts, *scene.pickups, *scene.destinations)
            self.assertTrue(all(c in free for c in cells), name)
            self.assertEqual(len(set(scene.starts)), 2)
            self.assertEqual(len(set(scene.destinations)), 2)
            self.assertEqual(len(scene.candidates), 6, name)
            self.assertTrue(set(scene.destinations) <= set(scene.candidates))
            self.assertTrue(all(c in free for c in scene.candidates))
            self.assertEqual(scene.objects, "goal")          # the two robots' objects only

    def test_the_tier_scenes_are_tier_scenes_own(self):
        from tier_scenes import SCENES as PLACED, house_free
        for name in TIERS:
            scene = self.script.SCENES[name][0]()
            (starts, pickups, destinations), candidates = PLACED[name]
            self.assertEqual(scene.starts, tuple(starts))
            self.assertEqual(scene.pickups, tuple(pickups))
            self.assertEqual(scene.destinations, tuple(destinations))
            self.assertEqual(scene.candidates, tuple(dict.fromkeys(candidates)))
            np.testing.assert_array_equal(np.isfinite(scene.grid), house_free())
            self.assertEqual(int(np.isfinite(scene.grid).sum()), 79)
            self.assertEqual(scene.handoff_cells, ())

    def test_every_scene_the_script_names_builds(self):
        for name, (make, _) in self.script.SCENES.items():
            make()


class GateScene(unittest.TestCase):
    def test_as_drawn(self):
        s = gate_scene()
        Y = len(GATE_MAP)
        at = {ch: (c, Y - 1 - r) for r, line in enumerate(GATE_MAP)
              for c, ch in enumerate(line) if ch in "12abABGd"}
        self.assertEqual(s.starts, (at["1"], at["2"]))
        self.assertEqual(s.pickups, (at["a"], at["b"]))          # a: robot 0's object
        self.assertEqual(s.destinations, (at["A"], at["B"]))
        self.assertEqual(s.candidates[:2], (at["B"], at["A"]))  # G0 robot 1's, G1 robot 0's
        self.assertEqual(len(_free(s)), 32)
        # each robot starts next to its partner's object, its own at home far away
        for i in (0, 1):
            self.assertEqual(_steps(s, s.starts[i], s.pickups[1 - i]), 1)
            self.assertGreater(_steps(s, s.starts[i], s.pickups[i]), 5)

    def test_mirror_symmetric(self):
        s = gate_scene()
        X = s.grid.shape[0]
        mirror = lambda c: (X - 1 - c[0], c[1])                 # noqa: E731
        np.testing.assert_array_equal(np.isfinite(s.grid), np.isfinite(s.grid[::-1, :]))
        self.assertEqual(tuple(map(mirror, s.starts)), s.starts[::-1])
        self.assertEqual(tuple(map(mirror, s.pickups)), s.pickups[::-1])
        self.assertEqual(tuple(map(mirror, s.destinations)), s.destinations[::-1])
        self.assertEqual(set(map(mirror, s.candidates)), set(s.candidates))

    def test_the_gate_and_the_way_round(self):
        s = gate_scene()
        Y = len(GATE_MAP)
        gate = next((c, Y - 1 - r) for r, line in enumerate(GATE_MAP)
                    for c, ch in enumerate(line) if ch == "G")
        left, right = (gate[0] - 1, gate[1]), (gate[0] + 1, gate[1])
        self.assertEqual(_steps(s, left, right), 2)               # through the gate
        blocked = gate_scene()
        grid = blocked.grid.copy()
        grid[gate] = np.inf
        round_ = type(blocked)(**{**blocked.__dict__, "grid": grid})
        self.assertEqual(_steps(round_, left, right), 12)          # the long way round

    def test_hand_off_variants(self):
        anywhere, few, gate, none = (gate_scene(h) for h in ("anywhere", "few", "gate", "none"))
        taken = set(anywhere.candidates) | set(anywhere.pickups)
        self.assertEqual(set(anywhere.handoff_cells), _free(anywhere) - taken)
        self.assertEqual(len(anywhere.handoff_cells), 24)
        self.assertEqual(len(few.handoff_cells), 4)
        self.assertEqual(len(gate.handoff_cells), 1)
        self.assertEqual(none.handoff_cells, ())
        self.assertTrue(set(gate.handoff_cells) <= set(few.handoff_cells)
                        <= set(anywhere.handoff_cells))
        for s in (anywhere, few, gate):
            self.assertFalse(set(s.handoff_cells) & taken)
        decoy = gate_scene("anywhere", decoy=True)
        self.assertEqual(len(decoy.object_cells), 3)
        self.assertNotIn(decoy.object_cells[2], decoy.handoff_cells)
        self.assertEqual([s.name for s in (anywhere, few, gate, none, decoy)],
                         ["gate", "gate_few", "gate_gateonly", "gate_nohandoff", "gate_decoy"])
        with self.assertRaises(ValueError):
            gate_scene("everywhere")

    def test_a_world_can_be_built(self):
        s = gate_scene("gate")
        cfg = Config(num_threads=1, steal=True, swap_on_place=True, foreign_carry_cost=1.0,
                     inertia=1.0, pick_cost_rate=2.0)
        world, goal_ids, solo, _ = build_world(
            s.grid, s.starts, s.pickups, s.destinations, s.candidate_cells, s.objects, cfg,
            handoff_cells=s.handoff_cells, rooms=s.rooms)
        self.assertEqual((world.N, world.M, world.H), (32, 2, 1))
        self.assertEqual(world.B, 98)
        self.assertEqual(tuple(solo), (19, 19))                   # the same task for both


if __name__ == "__main__":
    unittest.main()
