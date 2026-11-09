"""Runtime per-iteration phase profiler for *upstream* graphdeco 3DGS.

Upstream ``train.py`` has no profiler hooks, so — exactly like
``scripts/vanilla_densify_log.py`` — this module monkeypatches upstream
in-process before ``train.py`` executes, and emits the **same**
``iter_profile.json`` schema that ``train.py`` (SGDense) writes. The two JSONs
feed ``sgdense.profiler.compare_profiles`` / ``python -m experiments.profile``,
giving a phase-by-phase SGDense-vs-vanilla cost table.

What is timed (phases identical to the SGDense profiler)::

    render    gaussian_renderer.render
    loss      utils.loss_utils.l1_loss + ssim
    backward  torch.Tensor.backward
    densify   GaussianModel.densify_and_prune / reset_opacity
    optimizer torch.optim.Adam.step
    misc      residual = iteration total - the five above

Iteration boundaries come from ``GaussianModel.update_learning_rate`` (upstream
calls it exactly once per iteration, at the top): the previous iteration is
finalised when the next one begins, and the last one at interpreter exit.

Because the phases here are whole-function spans rather than the SGDense loop's
inline boundaries, ``loss``/``backward``/``optimizer`` are attributed purely by
``l1_loss``/``ssim``/``Tensor.backward``/``Adam.step`` calls — everything else
(camera pick, logging, ``training_report``) lands in ``misc``. This is the
closest apples-to-apples mapping to the SGDense phase set without forking
upstream.

Usage (see ``experiments/profile.py`` / the vanilla notebook cell)::

    python -c "import vanilla_iter_profile, runpy, sys; \
               sys.argv = ['train.py'] + sys.argv[1:]; \
               runpy.run_path('train.py', run_name='__main__')" <train args...>

Environment
-----------
SGDENSE_REPO
    Path to the SGDense-3DGS checkout, so ``import sgdense`` resolves. Required.
SGDENSE_PROFILE_ITERS
    Iteration spec ('a:b' range, 'a,b,c' set, or 'all'). Unset => disabled.
SGDENSE_PROFILE_OUT
    Where to write the JSON (default ``./iter_profile.json``).
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
    from sgdense.profiler import IterProfiler  # noqa: E402
except Exception as exc:  # pragma: no cover - configuration error, not logic
    raise ImportError(
        "vanilla_iter_profile needs the `sgdense` package importable. Set the "
        "SGDENSE_REPO environment variable to the SGDense-3DGS checkout root."
    ) from exc

_out = os.environ.get("SGDENSE_PROFILE_OUT") or os.path.join(os.getcwd(), "iter_profile.json")
_prof = IterProfiler(os.environ.get("SGDENSE_PROFILE_ITERS"), out_path=_out,
                     method="vanilla_3dgs")

# Per-iteration scratch state shared across the patched functions.
_state = {"iter": None, "n": 0, "densify": False}


def _finish_previous():
    """Finalise the iteration opened by the previous update_learning_rate call."""
    if _state["iter"] is None:
        return
    _prof.end_iteration(densify=_state["densify"], n_gaussians=_state["n"])


def _install():
    import gaussian_renderer
    from scene.gaussian_model import GaussianModel
    from utils import loss_utils

    if getattr(GaussianModel, "_vanilla_iter_profiled", False):
        return
    GaussianModel._vanilla_iter_profiled = True

    _render = gaussian_renderer.render
    _l1 = loss_utils.l1_loss
    _ssim = loss_utils.ssim
    _backward = torch.Tensor.backward
    _adam_step = torch.optim.Adam.step
    _update_lr = GaussianModel.update_learning_rate
    _densify = GaussianModel.densify_and_prune
    _reset = GaussianModel.reset_opacity

    def render(*a, **k):
        with _prof.phase("render"):
            return _render(*a, **k)

    def l1_loss(*a, **k):
        with _prof.phase("loss"):
            return _l1(*a, **k)

    def ssim(*a, **k):
        with _prof.phase("loss"):
            return _ssim(*a, **k)

    def backward(self, *a, **k):
        with _prof.phase("backward"):
            return _backward(self, *a, **k)

    def adam_step(self, *a, **k):
        with _prof.phase("optimizer"):
            return _adam_step(self, *a, **k)

    def densify_and_prune(self, *a, **k):
        _state["densify"] = True
        with _prof.phase("densify"):
            return _densify(self, *a, **k)

    def reset_opacity(self, *a, **k):
        _state["densify"] = True
        with _prof.phase("densify"):
            return _reset(self, *a, **k)

    def update_learning_rate(self, iteration):
        # Upstream calls this once per iteration, at the top -> stamp boundaries.
        _finish_previous()
        _state["iter"] = iteration
        _state["n"] = int(self.get_xyz.shape[0])
        _state["densify"] = False
        _prof.begin_iteration(iteration, n_gaussians=_state["n"])
        return _update_lr(self, iteration)

    gaussian_renderer.render = render
    loss_utils.l1_loss = l1_loss
    loss_utils.ssim = ssim
    torch.Tensor.backward = backward
    torch.optim.Adam.step = adam_step
    GaussianModel.densify_and_prune = densify_and_prune
    GaussianModel.reset_opacity = reset_opacity
    GaussianModel.update_learning_rate = update_learning_rate


def _save():
    try:
        _finish_previous()
        path = _prof.save()
        if path:
            print(f"[vanilla_iter_profile] {len(_prof.records)} iteration records -> {path}")
    except Exception as exc:  # never let profiling break a finished run
        print(f"[vanilla_iter_profile] WARNING: could not save profile: {exc}")


atexit.register(_save)
_install()
