# CPU unit tests for the Option A densification-gradient decoupling.
# Verifies (a) the pure accumulation helper used by add_densification_stats and
# (b) the autograd property the decoupling relies on: a non-accumulating
# autograd.grad on the photometric term excludes the segmentation term while
# leaving the full backward (parameter updates) intact.
#   python tests/test_densify_decouple.py
#   pytest tests/test_densify_decouple.py
import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense.densify_grad import accumulate_densify_stats  # noqa: E402


class TestAccumulateHelper(unittest.TestCase):
    def test_uses_provided_point_grad(self):
        # 4 Gaussians; only 0 and 2 are visible (filter is [K, 1] like the live path).
        accum = torch.zeros((4, 1))
        denom = torch.zeros((4, 1))
        point_grad = torch.tensor([[3.0, 4.0, 0.0],
                                   [9.0, 9.0, 9.0],
                                   [1.0, 0.0, 7.0],
                                   [9.0, 9.0, 9.0]])
        update_filter = torch.tensor([[0], [2]], dtype=torch.long)

        accumulate_densify_stats(accum, denom, point_grad, update_filter)

        # norm of the first two components only: ||(3,4)|| = 5, ||(1,0)|| = 1
        self.assertAlmostEqual(float(accum[0, 0]), 5.0, places=5)
        self.assertAlmostEqual(float(accum[2, 0]), 1.0, places=5)
        # invisible rows untouched
        self.assertEqual(float(accum[1, 0]), 0.0)
        self.assertEqual(float(accum[3, 0]), 0.0)
        self.assertEqual(float(denom[0, 0]), 1.0)
        self.assertEqual(float(denom[2, 0]), 1.0)
        self.assertEqual(float(denom.sum()), 2.0)

    def test_accumulates_across_iterations(self):
        accum = torch.zeros((2, 1))
        denom = torch.zeros((2, 1))
        pg = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        f = torch.tensor([[0], [1]], dtype=torch.long)
        for _ in range(3):
            accumulate_densify_stats(accum, denom, pg, f)
        self.assertAlmostEqual(float(accum[0, 0]), 3.0, places=5)
        self.assertAlmostEqual(float(accum[1, 0]), 6.0, places=5)
        self.assertEqual(float(denom[0, 0]), 3.0)


class TestAutogradIsolation(unittest.TestCase):
    """The core of Option A: a non-accumulating grad of the photo term excludes
    the seg term, and the subsequent full backward is neither polluted nor
    double-counted."""

    def test_autograd_grad_excludes_seg_term(self):
        means = torch.zeros(3, requires_grad=True)
        photo = (means ** 2).sum()
        seg = ((means - 5.0) ** 2).sum()

        g_photo_only = torch.autograd.grad(photo, means, retain_graph=True)[0]
        (photo + seg).backward()

        self.assertTrue(torch.allclose(g_photo_only, 2.0 * means))
        # total .grad is photo + seg, i.e. really different from photo alone
        expected_total = 2.0 * means + 2.0 * (means - 5.0)
        self.assertTrue(torch.allclose(means.grad, expected_total))
        self.assertFalse(torch.allclose(g_photo_only, means.grad))

    def test_full_backward_not_double_counted_after_autograd_grad(self):
        means = torch.zeros(2, requires_grad=True)
        photo = (means ** 2).sum()
        seg = (means + 1.0).sum()

        # mimics the train.py sequence: extract photo-only grad, then full backward
        g_photo = torch.autograd.grad(photo, means, retain_graph=True)[0]
        (photo + seg).backward()

        expected = g_photo + torch.ones_like(means)      # photo + d(seg)/d(means)
        self.assertTrue(torch.allclose(means.grad, expected))


if __name__ == "__main__":
    unittest.main()
