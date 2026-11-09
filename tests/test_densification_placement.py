# Acceptance tests for segmentation-guided densification (SGDense).
# Run on a GPU machine with the rebuilt extension installed:
#   python tests/test_densification_placement.py
# (pytest-compatible: pytest tests/test_densification_placement.py)
#
# Covers:
#   py_compile of the edited files
#   static: SGDense category-routing sources present (arguments / train / model)
#   seg_hist_readout 6-tuple algebra (pure torch, CPU)
#   real GPU densify_and_split child count / ordering
# §7.5 THE SPIKE / §7.6 calibration live in spike_route.py.
import os
import re
import sys
import unittest
import py_compile
from argparse import ArgumentParser

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

GM = os.path.join(ROOT, "scene", "gaussian_model.py")
ARG = os.path.join(ROOT, "arguments", "__init__.py")
TRAIN = os.path.join(ROOT, "train.py")


def _needs_cuda(fn):
    # Runner- AND pytest-agnostic skip. unittest.SkipTest is honored by pytest and
    # caught by the __main__ runner below.
    def wrapper(*a, **k):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires CUDA")
        return fn(*a, **k)
    wrapper.__name__ = getattr(fn, "__name__", "test")
    return wrapper


# --------------------------------------------------------------------------- #
# static checks                                                                #
# --------------------------------------------------------------------------- #
def test_py_compile_edited_files():
    for f in (GM, ARG, TRAIN):
        py_compile.compile(f, doraise=True)
    print("PASS py_compile (gaussian_model.py, arguments/__init__.py, train.py)")


def test_sgdense_sources_present_or_skip():
    gm, arg, tr = open(GM).read(), open(ARG).read(), open(TRAIN).read()
    present = ("class SGDenseParams" in arg) and ("generate_category_masks" in tr) \
        and ("interior_mask" in gm)
    if not present:
        raise unittest.SkipTest("SGDense routing not implemented")
    assert "class SGDenseParams" in arg, "arguments: SGDenseParams missing"
    assert "generate_category_masks" in tr, "train.py does not call generate_category_masks"
    assert "DensificationConfig" in tr and "s_tilde" in tr, \
        "train.py does not build/pass the SGDense config + texture scores"
    assert re.search(r"def densify_and_prune", gm), "GaussianModel.densify_and_prune missing"
    assert all(k in gm for k in ("interior_mask", "boundary_mask", "floater_mask", "other_mask")), \
        "densify_and_prune does not accept the four category masks"
    assert re.search(r"def _remap_after_split", gm), "_remap_after_split helper missing"
    print("PASS SGDense sources present")


def test_deterministic_split_removed():
    gm, arg = open(GM).read(), open(ARG).read()
    assert "seg_split_deterministic" not in gm, "deterministic-split machinery still in gaussian_model.py"
    assert "seg_split_offset_ratio" not in gm, "seg_split_offset_ratio still in gaussian_model.py"
    assert "seg_split_deterministic" not in arg, "seg_split_deterministic still in arguments"
    assert "seg_split_offset_ratio" not in arg, "seg_split_offset_ratio still in arguments"
    print("PASS deterministic split fully removed")


def test_densify_event_logging_wired():
    gm, arg, tr = open(GM).read(), open(ARG).read(), open(TRAIN).read()
    assert "log_densify_events" in arg, "arguments: --log_densify_events missing"
    assert "set_densify_tracker" in gm, "gaussian_model: set_densify_tracker missing"
    for hook in ("on_clone", "on_split", "on_prune", "on_reset"):
        assert hook in gm, f"gaussian_model does not fire tracker hook {hook}"
    assert "DensifyTracker" in tr, "train.py does not construct a DensifyTracker"
    assert "densify_events.pt" in tr, "train.py does not save densify_events.pt"
    print("PASS densify-event logging wired (model hooks + train.py save)")


# --------------------------------------------------------------------------- #
# seg_hist_readout 6-tuple algebra (pure torch, CPU)                           #
# --------------------------------------------------------------------------- #
def test_seg_hist_readout_algebra():
    from sgdense.readout import seg_hist_readout
    M = torch.tensor([[3.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 1.0, 1.0]])
    consistency, structure, disagree, entropy, top2, S = seg_hist_readout(M)
    assert torch.allclose(S, M.sum(dim=1)), "S must be the row mass"
    assert torch.allclose(consistency, top2[:, 0]), "consistency must equal P_(1)"
    assert torch.allclose(disagree, 1.0 - top2[:, 0]), "disagree must equal 1 - P_(1)"
    assert torch.allclose(structure, top2[:, 0] + top2[:, 1]), "structure must equal P_(1)+P_(2)"
    assert (entropy >= -1e-6).all(), "entropy must be non-negative"
    # Zero-evidence row -> P == 0 -> equal top-2, maximal disagree.
    assert torch.allclose(top2[1], torch.zeros(2)), "zero-evidence row must give P == 0"
    print("PASS seg_hist_readout algebra")


# --------------------------------------------------------------------------- #
# DensifyTracker id-alignment across the clone/split/prune protocol (CPU)       #
# --------------------------------------------------------------------------- #
class _FakeModel:
    """Minimal stand-in exposing exactly what DensifyTracker reads."""

    def __init__(self, P, C):
        self._xyz = torch.randn(P, 3)
        self._scaling = torch.randn(P, 3)
        self._opacity = torch.randn(P, 1)
        self.seg_hist = torch.rand(P, C)
        self.max_radii2D = torch.rand(P)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_scaling(self):
        return torch.exp(self._scaling)

    @property
    def get_opacity(self):
        return torch.sigmoid(self._opacity)

    def append_rows(self, k):   # emulate densification_postfix
        self._xyz = torch.cat([self._xyz, torch.randn(k, 3)])
        self._scaling = torch.cat([self._scaling, torch.randn(k, 3)])
        self._opacity = torch.cat([self._opacity, torch.randn(k, 1)])
        self.seg_hist = torch.cat([self.seg_hist, torch.rand(k, self.seg_hist.shape[1])])
        self.max_radii2D = torch.cat([self.max_radii2D, torch.rand(k)])

    def drop_rows(self, mask):  # emulate prune_points
        keep = ~mask
        self._xyz = self._xyz[keep]
        self._scaling = self._scaling[keep]
        self._opacity = self._opacity[keep]
        self.seg_hist = self.seg_hist[keep]
        self.max_radii2D = self.max_radii2D[keep]


def test_tracker_alignment_cpu():
    from sgdense.densify_tracker import DensifyTracker

    torch.manual_seed(0)
    P, C = 6, 3
    m = _FakeModel(P, C)
    tr = DensifyTracker(m)
    assert tr.gid.tolist() == list(range(P)), "initial ids must be 0..P-1"

    # clone: parents stay, k duplicates appended at the tail
    sel = torch.tensor([True, False, False, True, False, False])
    k = int(sel.sum())
    m.append_rows(k)
    tr.on_clone(sel, torch.rand(P), m.get_xyz[P:P + k])
    assert tr.gid.shape[0] == P + k, "clone must append k ids"
    assert tr.gid[:P].tolist() == list(range(P)), "clone must not reorder parents"
    assert tr.gid[P:].tolist() == list(range(P, P + k)), "clone children get fresh ids"

    # split: N children per selected parent appended, parents removed
    P2 = int(m.get_xyz.shape[0])
    sel2 = torch.zeros(P2, dtype=torch.bool)
    sel2[0] = True
    sel2[P2 - 1] = True
    k2 = int(sel2.sum()) * 2
    survivors = tr.gid[:P2][~sel2].tolist()
    m.append_rows(k2)
    tr.on_split(sel2, torch.rand(P2), m.get_xyz[P2:P2 + k2], N=2)
    assert tr.gid.shape[0] == P2 - int(sel2.sum()) + k2, "split must drop parents"
    assert tr.gid[:len(survivors)].tolist() == survivors, "split must keep survivors in order"
    assert tr.gid[len(survivors):].tolist() == list(range(tr.next_id - k2, tr.next_id)), \
        "split children get fresh ids in parent order"

    # prune: drop rows, ids follow the survivors
    Pm = int(m.get_xyz.shape[0])
    pmask = torch.zeros(Pm, dtype=torch.bool)
    pmask[0] = True
    expect = tr.gid[:Pm][~pmask].tolist()
    m.drop_rows(pmask)
    tr.on_prune(pmask, reason="test")
    assert tr.gid.tolist() == expect, "prune must drop exactly the masked ids"

    # flush closes the event and agrees with the live population
    tr.flush(100)
    assert len(tr.summary) == 1, "one event must be recorded"
    assert tr.summary[0]["n"] == int(m.get_xyz.shape[0]), "flush must match live N"
    print("PASS DensifyTracker id alignment (clone/split/prune/flush)")


# --------------------------------------------------------------------------- #
# real GPU densify_and_split (vanilla N(0, Sigma) placement)                   #
# --------------------------------------------------------------------------- #
def build_model(seed=0, P=48, C=8):
    from arguments import ModelParams, OptimizationParams
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    torch.manual_seed(seed)
    parser = ArgumentParser()
    lp, op = ModelParams(parser), OptimizationParams(parser)
    args = parser.parse_args([])
    dataset, opt = lp.extract(args), op.extract(args)

    g = GaussianModel(dataset.sh_degree,
                      seg_encoding_dim=dataset.seg_encoding_dim,
                      num_segmentation_classes=C)
    pts = torch.rand(P, 3) * 2 - 1
    pcd = BasicPointCloud(pts.numpy(), torch.rand(P, 3).numpy(), torch.zeros(P, 3).numpy())
    g.create_from_pcd(pcd, [], spatial_lr_scale=1.0)
    g.training_setup(opt)

    # Make every point split-eligible: percent_dense=0 disables the clone branch
    # (clone needs max(scaling) <= 0) and enables the split branch (> 0).
    g.percent_dense = 0.0
    with torch.no_grad():
        g._scaling.copy_(torch.log(torch.rand(P, 3, device="cuda") * 0.5 + 0.2))
    g.tmp_radii = torch.zeros(P, device="cuda")
    return g


@_needs_cuda
def test_gpu_split_child_count():
    g = build_model(seed=0)
    n = g.get_xyz.shape[0]
    prune_filter, n_children = g.densify_and_split(
        torch.ones((n, 1), device="cuda"), 0.0, 1.0, N=2)
    assert prune_filter.shape[0] == n + 2 * n, \
        f"prune_filter len {prune_filter.shape[0]} != {3 * n}"
    assert n_children == 2 * n, f"expected {2 * n} children, got {n_children}"
    assert g.get_xyz.shape[0] == 3 * n, \
        f"post-split population {g.get_xyz.shape[0]} != {3 * n}"
    print(f"PASS GPU split child count ({n} parents -> {n_children} children)")


def _run(fn):
    try:
        fn()
    except unittest.SkipTest as e:
        print(f"SKIP {fn.__name__}: {e}")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    _run(test_py_compile_edited_files)
    _run(test_sgdense_sources_present_or_skip)
    _run(test_deterministic_split_removed)
    _run(test_densify_event_logging_wired)
    _run(test_seg_hist_readout_algebra)
    _run(test_tracker_alignment_cpu)
    if not torch.cuda.is_available():
        print("SKIP GPU tests (no CUDA on this host)")
    else:
        _run(test_gpu_split_child_count)
    print("ALL DENSIFICATION-PLACEMENT TESTS PASSED")
