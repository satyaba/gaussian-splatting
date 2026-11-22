# CPU tests for region-aware evaluation (docs/08 7.9-7.10).
#   python tests/test_evalmetrics.py
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense.evalmetrics import seg_boundary_mask, region_psnr, paired_summary
from scripts.compare_runs import compare


def test_seg_boundary_mask_two_halves():
    gt = np.zeros((20, 20), dtype=np.int64)
    gt[:, 10:] = 1
    band = seg_boundary_mask(gt, 2, dilate_px=1)
    # seam between x=9 and x=10, dilated by 1 -> x in [8, 11]
    assert band[:, 9].all() and band[:, 10].all()
    assert band[:, 8].all() and band[:, 11].all()
    assert not band[:, 7].any() and not band[:, 12].any()
    assert not band[:, 0].any() and not band[:, 19].any()
    print("PASS seg_boundary_mask (two halves, dilation)")


def test_seg_boundary_mask_void():
    gt = np.zeros((10, 10), dtype=np.int64)
    gt[:, 5] = -1                       # a void column inside one class
    band = seg_boundary_mask(gt, 2, dilate_px=0)
    assert not band.any(), "void interior must not create a boundary"
    print("PASS seg_boundary_mask (void is ignored)")


def test_region_psnr():
    pred = np.zeros((4, 4, 3), dtype=np.float64)
    gt = pred.copy()
    gt[0, 0] = 0.1
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True
    p = region_psnr(pred, gt, mask)     # mse 0.01 -> 20 dB
    assert abs(p - 20.0) < 1e-6, p
    assert np.isnan(region_psnr(pred, gt, np.zeros_like(mask)))
    print("PASS region_psnr (masked MSE -> dB, empty -> NaN)")


def test_paired_summary():
    a = {"v1": {"psnr": 30.0}, "v2": {"psnr": 31.0}}
    b = {"v1": {"psnr": 30.5}, "v2": {"psnr": 30.5}}
    r = paired_summary(a, b, "psnr", higher_better=True)
    assert r["n"] == 2
    assert abs(r["mean_delta"]) < 1e-9
    assert abs(r["frac_b_better"] - 0.5) < 1e-9
    # lpips is lower-better
    a = {"v1": {"lpips": 0.10}, "v2": {"lpips": 0.20}}
    b = {"v1": {"lpips": 0.09}, "v2": {"lpips": 0.21}}
    r = paired_summary(a, b, "lpips", higher_better=False)
    assert abs(r["frac_b_better"] - 0.5) < 1e-9
    print("PASS paired_summary (deltas + fraction better)")


def test_compare_runs_synthetic():
    def rec(name, psnr, lpips, pb, lb):
        return dict(name=name, psnr=psnr, ssim=0.9, lpips=lpips,
                    psnr_boundary=pb, lpips_boundary=lb, psnr_interior=psnr + 3,
                    lpips_interior=lpips - 0.02, boundary_frac=0.1)

    A = dict(gaussians=1000, splits={"test": dict(psnr=30.0, lpips=0.10, per_view=[
        rec("v1", 30.0, 0.10, 26.0, 0.20), rec("v2", 31.0, 0.11, 27.0, 0.21)])})
    B = dict(gaussians=1200, splits={"test": dict(psnr=30.1, lpips=0.095, per_view=[
        rec("v1", 30.1, 0.095, 26.5, 0.19), rec("v2", 30.9, 0.10, 27.2, 0.20)])})

    td = tempfile.mkdtemp()
    pa, pb = os.path.join(td, "a.json"), os.path.join(td, "b.json")
    json.dump(A, open(pa, "w")); json.dump(B, open(pb, "w"))

    rows = compare(pa, pb, "test", "A", "B")
    by = {m: r for m, _, r in rows}
    assert by["lpips"]["n"] == 2 and by["lpips"]["mean_delta"] < 0, by["lpips"]
    assert by["lpips_boundary"]["frac_b_better"] == 1.0, by["lpips_boundary"]
    print("PASS compare_runs (paired per-view table)")


def _run(fn):
    try:
        fn()
    except Exception as e:                       # noqa: BLE001
        print(f"FAIL {fn.__name__}: {e}")
        raise


if __name__ == "__main__":
    test_seg_boundary_mask_two_halves()
    test_seg_boundary_mask_void()
    test_region_psnr()
    test_paired_summary()
    test_compare_runs_synthetic()
    print("ALL EVALMETRICS TESTS PASSED")
