"""Runtime instrumentation: log densification events for *upstream* graphdeco 3DGS.

Upstream ``scene/gaussian_model.py`` has no tracker hooks (unlike this fork's
``GaussianModel``). This module monkeypatches the upstream class in-process so a
stock training run emits the same ``densify_events.pt`` that SGDense runs do,
which the ``sgdense.viz`` helpers can then plot.

Usage (see ``Vanilla_3DGS_Experiment.ipynb``) -- import this module *before*
executing upstream ``train.py`` in the same interpreter::

    python -c "import vanilla_densify_log, runpy, sys; \
               sys.argv = ['train.py'] + sys.argv[1:]; \
               runpy.run_path('train.py', run_name='__main__')" <train args...>

Environment
-----------
SGDENSE_REPO
    Path to the SGDense-3DGS checkout, so ``import sgdense`` resolves. Required.
DENSIFY_EVENTS_OUT
    Where to ``torch.save`` the tracker at interpreter exit. Required to persist.

Notes
-----
* Upstream has no segmentation, so every event's per-Gaussian ``parent_class`` is
  ``-1``; population / clone / split / prune / lineage are all still recorded.
* Hooks are installed once (idempotent); re-importing is harmless.
* The event's ``iter`` is stamped from ``GaussianModel.update_learning_rate``,
  which upstream calls once per iteration with the loop counter.
"""

from __future__ import annotations

import atexit
import os
import sys

import torch

_repo = os.environ.get("SGDENSE_REPO")
if _repo and _repo not in sys.path:
    # append, not insert: upstream's own `scene`/`utils`/... must keep winning.
    sys.path.append(_repo)

try:
    from sgdense.densify_tracker import DensifyTracker
    from sgdense.grad_stats import GradStatsLogger
except Exception as exc:  # pragma: no cover - configuration error, not logic
    raise ImportError(
        "vanilla_densify_log needs the `sgdense` package importable. Set the "
        "SGDENSE_REPO environment variable to the SGDense-3DGS checkout root."
    ) from exc

_tracker = None
_last_step = None
_suppress_prune = False
_prune_reason = "opacity+size"
_pending_split = None
_grad_stats_out = os.environ.get("GRAD_STATS_OUT")
_grad_logger = GradStatsLogger() if _grad_stats_out else None


def _tracker_for(g):
    """Lazily create the tracker on first use (population is already final)."""
    global _tracker
    if _tracker is None:
        _tracker = DensifyTracker(g, snapshot_max_points=5000)
    return _tracker


def _install():
    from scene.gaussian_model import GaussianModel

    if getattr(GaussianModel, "_vanilla_densify_logged", False):
        return
    GaussianModel._vanilla_densify_logged = True

    _clone = GaussianModel.densify_and_clone
    _split = GaussianModel.densify_and_split
    _postfix = GaussianModel.densification_postfix
    _prune_points = GaussianModel.prune_points
    _reset = GaussianModel.reset_opacity
    _densify_and_prune = GaussianModel.densify_and_prune
    _update_lr = GaussianModel.update_learning_rate
    _add_stats = GaussianModel.add_densification_stats

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Same selection as upstream, recomputed so the tracker sees the mask.
        sel = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        sel = torch.logical_and(
            sel,
            torch.max(self.get_scaling, dim=1).values <= self.percent_dense * scene_extent)
        padded = torch.zeros(self.get_xyz.shape[0], device="cuda")
        padded[: grads.shape[0]] = torch.norm(grads, dim=-1).squeeze()
        _clone(self, grads, grad_threshold, scene_extent)
        k = int(sel.sum())
        if k:
            # clone appends at the tail and prunes nothing -> children are xyz[-k:].
            _tracker_for(self).on_clone(sel, padded, self.get_xyz[-k:], N=1)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        global _suppress_prune, _pending_split
        n_init = self.get_xyz.shape[0]
        padded = torch.zeros(n_init, device="cuda")
        padded[: grads.shape[0]] = grads.squeeze()
        sel = torch.where(padded >= grad_threshold, True, False)
        sel = torch.logical_and(
            sel,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense * scene_extent)
        # on_split must run at densification_postfix: the model then holds
        # [parents ++ children] (so the tracker can read the appended children and
        # rebuild survivors++children), which is exactly what it expects. The
        # parent removal right after is suppressed from the prune hook.
        _pending_split = (sel, padded, N)
        _suppress_prune = True
        try:
            return _split(self, grads, grad_threshold, scene_extent, N)
        finally:
            _suppress_prune = False
            _pending_split = None

    def densification_postfix(self, *args, **kwargs):
        res = _postfix(self, *args, **kwargs)
        if _pending_split is not None:
            sel, padded, N = _pending_split
            if int(sel.sum()) > 0:
                k = int(sel.sum()) * N
                _tracker_for(self).on_split(sel, padded, self.get_xyz[-k:], N=N)
        return res

    def prune_points(self, mask):
        if not _suppress_prune and _tracker is not None:
            _tracker.on_prune(mask, reason=_prune_reason)
        return _prune_points(self, mask)

    def reset_opacity(self):
        res = _reset(self)
        if _tracker is not None:
            _tracker.on_reset()
        return res

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        global _prune_reason
        _prune_reason = "opacity+size" if max_screen_size else "opacity"
        # Snapshot the accumulated trigger before the original resets it.
        if _grad_logger is not None:
            _grad_logger.threshold = float(max_grad)
            _grad_logger.snapshot(self.xyz_gradient_accum, self.denom,
                                  _last_step, threshold=max_grad)
        return _densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        res = _add_stats(self, viewspace_point_tensor, update_filter)
        if _grad_logger is not None and viewspace_point_tensor.grad is not None:
            _grad_logger.observe_iter(
                torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1),
                _last_step if _last_step is not None else 0)
        return res

    def update_learning_rate(self, iteration):
        # Upstream calls this once per iteration -> stamp step + close last event.
        global _last_step
        tr = _tracker_for(self)
        if tr.pending is not None and _last_step is not None:
            tr.flush(_last_step)
        _last_step = iteration
        tr.step = iteration
        return _update_lr(self, iteration)

    GaussianModel.densify_and_clone = densify_and_clone
    GaussianModel.densify_and_split = densify_and_split
    GaussianModel.densification_postfix = densification_postfix
    GaussianModel.prune_points = prune_points
    GaussianModel.reset_opacity = reset_opacity
    GaussianModel.densify_and_prune = densify_and_prune
    GaussianModel.update_learning_rate = update_learning_rate
    GaussianModel.add_densification_stats = add_densification_stats


def _save():
    out = os.environ.get("DENSIFY_EVENTS_OUT")
    if _tracker is not None and out:
        try:
            if _tracker.pending is not None and _last_step is not None:
                _tracker.flush(_last_step)
            _tracker.save(out)
            print(f"[vanilla_densify_log] {len(_tracker.events)} densification events -> {out}")
        except Exception as exc:  # never let logging break a finished run
            print(f"[vanilla_densify_log] WARNING: could not save events: {exc}")
    if _grad_logger is not None and _grad_stats_out:
        try:
            _grad_logger.save(_grad_stats_out)
            print(f"[vanilla_densify_log] {len(_grad_logger.events)} grad snapshots -> {_grad_stats_out}")
        except Exception as exc:
            print(f"[vanilla_densify_log] WARNING: could not save grad stats: {exc}")


atexit.register(_save)
_install()
