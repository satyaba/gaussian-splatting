"""Densification-gradient trigger helpers (Option A: decouple from seg loss).

The 3DGS / SGDense clone-split trigger is fed by
``GaussianModel.add_densification_stats``: per iteration it accumulates, per
Gaussian, the L2 norm of the loss gradient w.r.t. the Gaussian's screen-space
mean into ``xyz_gradient_accum`` (and bumps the observation count ``denom``).
The trigger then fires where ``xyz_gradient_accum / denom >= densify_grad_threshold``.

By default the *total* loss gradient is accumulated, so any loss term that
reaches the rendered means (notably the segmentation cross-entropy) inflates the
trigger. ``--densify_photo_grad_only`` routes the photometric-only point gradient
here instead. The accumulation itself is kept as a pure function so it can be
unit-tested on CPU without a live ``GaussianModel``.
"""

from __future__ import annotations

import torch


def accumulate_densify_stats(xyz_gradient_accum, denom, point_grad, update_filter):
    """Accumulate one iteration of the densification trigger.

    ``point_grad`` is the ``[P, 3]`` screen-space point gradient (as produced by
    ``torch.autograd.grad`` or read from ``viewspace_point_tensor.grad``).
    ``update_filter`` is a long index tensor (possibly ``[K, 1]``) selecting the
    visible Gaussians. ``xyz_gradient_accum`` is ``[P, 1]``; ``denom`` is ``[P, 1]``.
    """
    xyz_gradient_accum[update_filter] += torch.norm(point_grad[update_filter, :2],
                                                    dim=-1, keepdim=True)
    denom[update_filter] += 1
