"""Region-aware evaluation helpers (docs/08 section 7.9-7.10).

PSNR is an L2 pixel average dominated by large flat regions, so it is almost
blind to a class boundary (a tiny pixel fraction). The placement mechanism
targets boundaries, so the honest metrics are *per view* and *boundary
restricted*. This module holds the pure, CPU-testable pieces:

  * seg_boundary_mask -- a boundary band from a GT class map.
  * region_psnr       -- PSNR restricted to a pixel mask.
  * paired_summary    -- per-view paired A/B comparison of a metric.
"""

from __future__ import annotations

import numpy as np

IGNORE_ID = -1   # camera_utils maps the 255 void label to -1


def seg_boundary_mask(gt_seg, num_classes, ignore_id=IGNORE_ID, dilate_px=3):
    """Boolean [H,W] boundary band from a GT class-id map.

    A pixel is on the boundary if any 4-neighbour carries a *different* valid
    class; the band is then dilated by ``dilate_px`` pixels. Numpy-only.
    """
    gt = np.asarray(gt_seg).astype(np.int64)
    valid = (gt != ignore_id) & (gt >= 0) & (gt < num_classes)
    g = np.where(valid, gt, 0)

    def _edge(a1, a2, v1, v2):
        return (a1 != a2) & v1 & v2

    band = np.zeros(gt.shape, dtype=bool)
    band[:, :-1] |= _edge(g[:, :-1], g[:, 1:], valid[:, :-1], valid[:, 1:])
    band[:, 1:] |= _edge(g[:, 1:], g[:, :-1], valid[:, 1:], valid[:, :-1])
    band[:-1, :] |= _edge(g[:-1, :], g[1:, :], valid[:-1, :], valid[1:, :])
    band[1:, :] |= _edge(g[1:, :], g[:-1, :], valid[1:, :], valid[:-1, :])

    for _ in range(int(dilate_px)):
        out = band.copy()
        out[1:, :] |= band[:-1, :]
        out[:-1, :] |= band[1:, :]
        out[:, 1:] |= band[:, :-1]
        out[:, :-1] |= band[:, 1:]
        band = out
    return band


def region_psnr(pred, gt, mask, eps=1e-12):
    """PSNR over the pixels where ``mask`` is True. NaN if the region is empty.

    ``pred``/``gt`` are [H,W,C] arrays; ``mask`` is [H,W] bool.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return float("nan")
    diff = (np.asarray(pred, dtype=np.float64) - np.asarray(gt, dtype=np.float64))
    mse = float((diff[mask] ** 2).mean())
    return float(-10.0 * np.log10(max(mse, eps)))


def paired_summary(per_view_a, per_view_b, metric, higher_better=True, tol=1e-9):
    """Paired per-view comparison of one metric between two runs.

    ``per_view_a`` / ``per_view_b`` map view name -> metrics dict. Only views
    present in both (and with a finite metric) are used. Returns the means, the
    mean delta (B - A), and the fraction of views where B is better.
    """
    names = []
    for n, va in per_view_a.items():
        vb = per_view_b.get(n)
        if vb is None:
            continue
        a, b = va.get(metric), vb.get(metric)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b):
            continue
        names.append(n)

    if not names:
        return dict(metric=metric, n=0)

    a = np.array([per_view_a[n][metric] for n in names], dtype=np.float64)
    b = np.array([per_view_b[n][metric] for n in names], dtype=np.float64)
    d = b - a
    better = (d > tol) if higher_better else (d < -tol)
    return dict(
        metric=metric,
        n=len(names),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_delta=float(d.mean()),
        median_delta=float(np.median(d)),
        std_delta=float(d.std()),
        frac_b_better=float(better.mean()),
    )
