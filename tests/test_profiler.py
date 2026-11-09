# CPU unit tests for the per-iteration profiler (spec parsing, aggregation,
# comparison). No GPU or real training required.
#   python tests/test_profiler.py
#   pytest tests/test_profiler.py
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense.profiler import (  # noqa: E402
    IterProfiler,
    PHASES,
    compare_profiles,
    format_comparison,
    parse_iters,
    summarize_profile,
)


def _rec(total_phases, misc=10.0, densify=False, n=1000):
    phases = dict(total_phases)
    total = sum(phases.values()) + misc
    phases["misc"] = misc
    return {"iteration": 0, "phases_ms": phases, "meta": {"densify": densify, "n_gaussians": n},
            "iter_ms": total, "misc_ms": misc}


class TestParseIters(unittest.TestCase):
    def test_disabled(self):
        for s in (None, "", "off", "none", "false", "0"):
            self.assertIsNone(parse_iters(s))

    def test_all(self):
        for s in ("all", "*"):
            self.assertEqual(parse_iters(s), "all")

    def test_range(self):
        self.assertEqual(parse_iters("3999:4099"), ("range", 3999, 4099))

    def test_set(self):
        self.assertEqual(parse_iters("1, 2,3"), ("set", {1, 2, 3}))
        self.assertEqual(parse_iters([10, 20]), ("set", {10, 20}))


class TestSpec(unittest.TestCase):
    def test_should_profile(self):
        p = IterProfiler("3999:4099")
        self.assertTrue(p.enabled)
        self.assertFalse(p.should_profile(3998))
        self.assertTrue(p.should_profile(3999))
        self.assertTrue(p.should_profile(4099))
        self.assertFalse(p.should_profile(4100))

    def test_disabled_noop(self):
        p = IterProfiler(None)
        self.assertFalse(p.enabled)
        self.assertFalse(p.begin_iteration(1))
        with p.phase("render"):
            pass
        self.assertIsNone(p.end_iteration())
        self.assertEqual(p.records, {})

    def test_lifecycle_derives_misc(self):
        p = IterProfiler("5:5")
        self.assertTrue(p.begin_iteration(5, n_gaussians=123))
        with p.phase("render"):
            pass
        p.start("densify")
        p.stop("densify")
        rec = p.end_iteration(densify=True)
        ph = rec["phases_ms"]
        self.assertEqual(set(ph), set(PHASES) | {"misc"})
        self.assertAlmostEqual(rec["phase_sum_ms"] + rec["misc_ms"], rec["iter_ms"], places=6)
        self.assertEqual(p.records[5]["meta"]["n_gaussians"], 123)


class TestSummarize(unittest.TestCase):
    def test_groups(self):
        recs = {
            3999: _rec({"render": 30, "loss": 5, "backward": 20, "optimizer": 15}, densify=False),
            4000: _rec({"render": 30, "loss": 5, "backward": 20, "optimizer": 15, "densify": 55}, densify=True),
        }
        s = summarize_profile(recs)
        self.assertEqual(s["n_records"], 2)
        self.assertEqual(s["groups"]["all"]["n"], 2)
        self.assertEqual(s["groups"]["densify"]["n"], 1)
        self.assertEqual(s["groups"]["no_densify"]["n"], 1)
        for ph in PHASES + ("misc",):
            self.assertIn(ph, s["groups"]["all"])


class TestCompare(unittest.TestCase):
    def _profiles(self):
        van = {3999: _rec({"render": 30, "loss": 5, "backward": 20, "optimizer": 15})}
        sgd = {3999: _rec({"render": 45, "loss": 12, "backward": 24, "optimizer": 22})}
        return ({"summary": summarize_profile(sgd), "records": sgd},
                {"summary": summarize_profile(van), "records": van})

    def test_deltas(self):
        sgd, van = self._profiles()
        cmp = compare_profiles(sgd, van)
        rows = {r["phase"]: r for r in cmp["rows"]}
        self.assertAlmostEqual(rows["render"]["delta_ms"], 15.0)
        self.assertAlmostEqual(rows["loss"]["delta_ms"], 7.0)
        self.assertAlmostEqual(rows["optimizer"]["delta_ms"], 7.0)
        # table renders without error and mentions each phase
        text = format_comparison(cmp)
        for ph in PHASES:
            self.assertIn(ph, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
