"""The belief update's workings in the record (``Config.belief_detail``, 2026-11-04; the
user: "All information for belief calculation should be in the JSOn").

A rollout of the tiny scene, its record through JSON as a file holds it, and from the record
alone, step by step and robot by robot: each hypothesis model's likelihood of the action
from its recorded Q, tau and lapse; Eq. 6 from those and the social belief before; the
per-destination and per-object likelihoods; and the whole update -- the physical and social
posteriors and the idle mode -- by the reference arithmetic of ``test_beliefs``, against
what the record says came out.  Also: the option, the records of before, and the reader
``tools/belief_step.py``.
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from social_pairings import Config, run_pairing
from social_pairings.config import ADDED_FIELDS, written_with
from social_pairings.scenes import Scene
from social_pairings.serialize import save_result, to_json

from .test_beliefs import reference_update
from .test_models import TINY

ROOT = Path(__file__).resolve().parents[2]
BASE = dict(num_threads=2, resolution=1.0, value_tol=1e-5, value_maps=False,
            deadlock_window=6, livelock_window=0)


def arr(x) -> np.ndarray:
    return np.array(x, dtype=float)


def logsumexp(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -np.inf
    top = v.max()
    return float(top + np.log(np.exp(v - top).sum()))


def rolled(**options) -> dict:
    """A rollout of the tiny scene, through JSON."""
    result = run_pairing(Scene(name="tiny", **TINY), ("cooperation", "competition"),
                         levels=(2, 2), cfg=Config(**BASE, **options), n_steps=6)
    return json.loads(json.dumps(to_json(result)))


class Option(unittest.TestCase):
    def test_the_option_and_the_records_of_before(self):
        self.assertTrue(Config().belief_detail)
        self.assertFalse(ADDED_FIELDS["belief_detail"])
        self.assertFalse(written_with({})["belief_detail"])
        for bad in (1, 0, "yes", None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Config(belief_detail=bad)

    def test_without_it(self):
        record = rolled(belief_detail=False)
        self.assertTrue(record["trace"])
        self.assertTrue(all("belief_update" not in row for row in record["trace"]))


class FromTheRecordAlone(unittest.TestCase):
    """Every update recomputed from what the record holds."""

    @classmethod
    def setUpClass(cls):
        cls.records = {"Eqs. 5-6": rolled(),
                       "softened": rolled(inference_temperature=0.5, belief_lapse=0.05)}

    def updates(self):
        for name, record in self.records.items():
            for row in record["trace"]:
                for i in (0, 1):
                    yield name, record, row, i, row["belief_update"][i]

    def test_every_step_has_it(self):
        for name, record in self.records.items():
            for row in record["trace"]:
                self.assertIn("belief_update", row, name)
                self.assertEqual(len(row["belief_update"]), 2)

    def test_the_units(self):
        for name, record, row, i, d in self.updates():
            if d.get("skipped"):
                continue
            cfg = record["config"]
            c_move = cfg["move_cost_rate"] * cfg["resolution"] / cfg["reference_step_m"]
            self.assertAlmostEqual(d["c_move"], c_move)
            self.assertAlmostEqual(d["c_stay"], cfg["wait_cost_rate"] * cfg["resolution"]
                                   / cfg["reference_step_m"])
            self.assertAlmostEqual(d["tau_planning"], cfg["temperature"] * c_move)
            t_inf = cfg.get("inference_temperature")
            self.assertAlmostEqual(d["tau"], (cfg["temperature"] if t_inf is None else t_inf)
                                   * c_move)
            self.assertEqual(d["lapse"], cfg["belief_lapse"])
            self.assertEqual(d["reader"], i)
            self.assertEqual(d["about"], 1 - i)
            self.assertEqual(d["action"], row["action"][1 - i])

    def test_each_models_likelihood_from_its_q(self):
        """log psi~(a) = log((1 - e) softmax(Q / tau)(a) + e / |A|), |A| the actions open."""
        n = 0
        for name, record, row, i, d in self.updates():
            if d.get("skipped") or "models" not in d:
                continue
            m, a, tau, e = d["models"], d["action"], d["tau"], d["lapse"]
            q, ll = arr(m["q"]), arr(m["log_lik"])
            for g in range(q.shape[0]):
                for t in range(q.shape[1]):
                    qq = q[g, t]
                    if np.isnan(qq).all():
                        continue                     # a physical goal ruled out
                    open_ = np.isfinite(qq)
                    if not open_[a]:                 # not open to this model: impossible
                        self.assertTrue(np.isnan(ll[g, t]))
                        continue
                    z = qq[open_] / tau
                    lp = np.full(qq.shape, -np.inf)
                    lp[open_] = z - (z.max() + np.log(np.exp(z - z.max()).sum()))
                    want = (lp[a] if e == 0.0 else
                            np.logaddexp(np.log1p(-e) + lp[a], np.log(e / open_.sum())))
                    self.assertAlmostEqual(ll[g, t], want, places=9, msg=(name, row["step"], i))
                    self.assertAlmostEqual(arr(m["q_taken"])[g, t], qq[a], places=12)
                    self.assertAlmostEqual(arr(m["q_best"])[g, t], qq[open_].max(), places=12)
                    self.assertEqual(int(m["best_action"][g][t]), int(np.nanargmax(
                        np.where(open_, qq, -np.inf))))
                    n += 1
        self.assertGreater(n, 50)

    def test_eq_6_and_the_factors(self):
        for name, record, row, i, d in self.updates():
            if d.get("skipped") or "models" not in d or d["likelihood"] != "level1":
                continue
            with np.errstate(divide="ignore"):
                ls = np.log(arr(d["prior"]["social"]))
            ll = arr(d["models"]["log_lik"])
            raw = arr(d["phys_log_lik"])
            for g in range(ll.shape[0]):
                want = logsumexp(ls + ll[g])
                if np.isfinite(want):
                    self.assertAlmostEqual(raw[g], want, places=9)
            obj = np.array(d["hypotheses"]["object"])
            dst = np.array(d["hypotheses"]["destination"])
            po, pd = arr(d["prior"]["object"]), arr(d["prior"]["destination"])
            for key, own, other, w in (("by_destination", dst, obj, po),
                                       ("by_object", obj, dst, pd)):
                got = arr(d[key]["log_lik"])
                for v in np.unique(own):
                    sel = own == v
                    with np.errstate(divide="ignore"):
                        want = logsumexp(np.log(w[other[sel]]) + raw[sel])
                    if np.isfinite(want):
                        self.assertAlmostEqual(got[v], want, places=9, msg=(name, key))
                np.testing.assert_allclose(arr(d[key]["after"]),
                                           arr(d["after"]["destination" if key == "by_destination"
                                                          else "object"]))

    def test_the_update_from_the_record(self):
        """The reference update, fed only what the record holds, gives what it says came out
        -- and that is what the trace's next beliefs show."""
        n = 0
        for name, record, row, i, d in self.updates():
            if d.get("skipped"):
                np.testing.assert_allclose(arr(d["after"]["phys"]), arr(d["prior"]["phys"]))
                continue
            cfg = record["config"]
            lik1 = np.nan_to_num(np.exp(arr(d["models"]["log_lik"])), nan=0.0)
            lik0 = np.exp(arr(d["phys_log_lik"]))
            lik0 = np.nan_to_num(lik0, nan=0.0)
            keep = np.array([bool(x) for x in d["transition_term"]])
            l_idle = 0.0 if d["idle_log_lik"] is None else float(np.exp(d["idle_log_lik"]))
            want = reference_update(arr(d["prior"]["phys"]), arr(d["prior"]["social"]),
                                    d["prior"]["idle"], lik0, lik1, l_idle, keep,
                                    d["stage"] == "hands empty",
                                    np.array(d["hypotheses"]["object"]), cfg["idle_hazard"],
                                    d["stages"], d["factored"],
                                    np.array(d["hypotheses"]["destination"]))
            np.testing.assert_allclose(arr(d["posterior"]["phys"]), want[0], atol=1e-10,
                                       err_msg=f"{name} step {row['step']} robot {i}")
            np.testing.assert_allclose(arr(d["social_posterior"]), want[1], atol=1e-10)
            self.assertAlmostEqual(d["after"]["idle"], want[2], places=10)
            # no hazards here: the belief after is the posterior
            np.testing.assert_allclose(arr(d["after"]["phys"]), arr(d["posterior"]["phys"]))
            np.testing.assert_allclose(arr(row["next_belief"][i]),
                                       arr(d["after"]["destination"]), atol=1e-12)
            n += 1
        self.assertGreater(n, 10)

    def test_the_reader(self):
        record = self.records["Eqs. 5-6"]
        with tempfile.TemporaryDirectory() as tmp:
            path = save_result(Path(tmp) / "tiny.json", record)
            out = subprocess.run([sys.executable, str(ROOT / "tools" / "belief_step.py"),
                                  str(path), "--step", "1", "--all-social"],
                                 capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("belief about", out.stdout)
        self.assertIn("check: Eq. 6 recomputed", out.stdout)
        self.assertIn("gap/move", out.stdout)


if __name__ == "__main__":
    unittest.main()
