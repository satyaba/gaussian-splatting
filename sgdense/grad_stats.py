"""Accumulated-gradient statistics for the densification trigger.

The 3DGS density-control trigger is::

    grads = xyz_gradient_accum / denom
    selected = grads >= densify_grad_threshold        # (plus a size branch)

``xyz_gradient_accum`` accumulates, per iteration, the L2 norm of the loss
gradient w.r.t. a Gaussian's screen-space mean (``add_densification_stats``).
Every loss term that touches the rendered means contributes to it, so adding a
term (e.g. the segmentation cross-entropy) can raise the gradient a Gaussian
sees and fire clone/split more often.

This module records, without changing training:
  * ``events``  -- the window-averaged ``xyz_gradient_accum/denom`` distribution at
    each densification event, plus the fraction at/over ``densify_grad_threshold``;
  * ``iters``   -- the raw per-iteration 2D-mean gradient norm (mean/max/#visible),
    which is the "gradient per iteration" the hypothesis is about.

Pure torch/numpy so it runs and is tested on CPU.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

import numpy as np
import torch


def summarize(values, threshold: Optional[float] = None) -> dict:
    """Quantile summary of a 1-D gradient sample.

    ``thresh_pct`` is the percentile rank of ``threshold`` within the sample
    (i.e. the fraction of Gaussians strictly below it) -- it tells you *where*
    the densification threshold sits relative to the live distribution.
    """
    v = values.detach().float().cpu().reshape(-1).numpy()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0, mean=0.0, median=0.0, p90=0.0, p99=0.0, max=0.0,
                    frac_over=None, thresh_pct=None)
    q50, q90, q99 = np.percentile(v, [50, 90, 99])
    out = dict(n=int(v.size), mean=float(v.mean()), median=float(q50),
               p90=float(q90), p99=float(q99), max=float(v.max()))
    if threshold is None:
        out["frac_over"] = None
        out["thresh_pct"] = None
    else:
        out["frac_over"] = float((v >= threshold).mean())
        out["thresh_pct"] = float((v < threshold).mean() * 100.0)
    return out


class GradStatsLogger:
    """Collect accumulated + per-iteration gradient statistics."""

    def __init__(self, threshold: Optional[float] = None, iter_every: int = 1):
        self.threshold = None if threshold is None else float(threshold)
        self.iter_every = max(1, int(iter_every))
        self.events: list = []
        self.iters: list = []

    def snapshot(self, accum, denom=None, iteration: Optional[int] = None,
                 threshold: Optional[float] = None) -> dict:
        """Record the window-averaged accumulated gradient at a densification event.

        ``denom`` is the per-Gaussian observation count; ``None`` treats ``accum``
        as already averaged. ``clamp_min(1)`` matches the training code: a
        never-observed Gaussian has denom 0 and contributes 0 (its raw ratio is
        NaN, which training zeroes out).
        """
        grads = accum / denom.clamp_min(1) if denom is not None else accum
        grads = torch.nan_to_num(grads, nan=0.0, posinf=0.0, neginf=0.0)
        thr = self.threshold if threshold is None else float(threshold)
        row = summarize(grads, thr)
        row["iter"] = None if iteration is None else int(iteration)
        self.events.append(row)
        return row

    def observe_iter(self, norms, iteration: int) -> None:
        """Record the raw per-iteration 2D-mean gradient norm (cheap: no percentiles)."""
        if iteration % self.iter_every != 0:
            return
        v = norms.detach().float().reshape(-1)
        v = v[torch.isfinite(v)]
        n = int(v.numel())
        if n == 0:
            self.iters.append([int(iteration), 0, 0.0, 0.0])
            return
        self.iters.append([int(iteration), n, float(v.mean()), float(v.max())])

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "events": self.events, "iters": self.iters}

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        return path
