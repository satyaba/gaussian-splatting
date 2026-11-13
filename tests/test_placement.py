# Acceptance tests for segmentation-boundary densification placement (SGDense,
# docs/08 section 4.1, step 2). Pure CPU -- no CUDA, no live GaussianModel:
#   python tests/test_placement.py
# (pytest-compatible: pytest tests/test_placement.py)
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense import placement as PL
from sgdense.geometric import pick_cell_size, reach_height, thin, p99


def make_slab(nx=81, ny=81, span=1.0, mass=5.0, tau=0.08):
    """Flat z=0 slab, class 0 for x<0 and class 1 for x>=0.

    The histogram is *mixed* near the interface (as a real persisted mass
    histogram is, when a Gaussian footprint straddles the boundary):
    ``M[:,1] = mass * sigmoid(x/tau)``, ``M[:,0] = mass - M[:,1]``. Far from the
    interface one class dominates, so ``argmax`` there is effectively one-hot.
    """
    xs = torch.linspace(-span, span, nx)
    ys = torch.linspace(-span, span, ny)
    gx, gy = torch.meshgrid(xs, ys, indexing="ij")
    xyz = torch.stack([gx.reshape(-1), gy.reshape(-1),
                       torch.zeros(nx * ny)], dim=1)
    p1 = torch.sigmoid(xyz[:, 0] / tau)
    seg = torch.stack([mass * (1.0 - p1), mass * p1], dim=1)
    return xyz, seg


# --------------------------------------------------------------------------- #
# local_frames                                                                 #
# --------------------------------------------------------------------------- #
def test_boundary_normal_recovers_separating_direction():
    xyz, seg = make_slab()
    sel = xyz[:, 0].abs() < 0.06
    n_hat, valid, n_surf, tangent = PL.local_frames(xyz, seg, sel, K=12, Kmin=6)

    assert valid.any(), "no valid boundary normals found on the 2-class slab"
    assert int(valid.sum()) >= 50, f"too few valid normals: {int(valid.sum())}"
    assert torch.allclose(n_hat[~valid], torch.zeros_like(n_hat[~valid])), \
        "invalid rows must carry a zero n_hat"

    nh = n_hat[valid]
    # unit vectors (within numerical tolerance)
    assert torch.allclose(nh.norm(dim=1), torch.ones(nh.shape[0]), atol=1e-4), \
        "n_hat must be unit length"
    # projected onto the surface (z=0 plane): no off-surface component
    assert nh[:, 2].abs().max() < 1e-3, "n_hat leaks off the surface"
    ns = n_surf[valid]
    assert (nh * ns).sum(1).abs().max() < 1e-3, "n_hat is not tangent to the surface"
    # the boundary runs along x -> the discrimination direction is +/- x.
    # Restrict to points away from the slab's y-edges: there the neighbour set is
    # one-sided, so n_hat legitimately picks up a y component (in a real scene
    # those "edges" are scene boundaries). Away from them it is unambiguous.
    near = xyz[valid][:, 1].abs() < 0.5
    assert int(near.sum()) >= 50, f"too few interior-boundary points: {int(near.sum())}"
    assert (nh[near][:, 0].abs() > 0.98).float().mean() > 0.98, \
        "n_hat does not align with the class-boundary direction (+/- x)"
    print(f"PASS local_frames boundary-normal direction + surface projection "
          f"({int(near.sum())} interior-boundary pts)")


def test_single_class_is_invalid():
    xyz, seg = make_slab()
    seg = torch.zeros_like(seg)
    seg[:, 0] = 5.0                                   # one class everywhere
    sel = xyz[:, 0].abs() < 0.06
    _, valid, _, _ = PL.local_frames(xyz, seg, sel, K=12, Kmin=6)
    assert int(valid.sum()) == 0, "single-class region must yield no valid normal"
    print("PASS local_frames single-class -> invalid")


def test_unobserved_is_invalid():
    xyz, _ = make_slab()
    seg = torch.zeros(xyz.shape[0], 2)                # no evidence
    sel = torch.ones(xyz.shape[0], dtype=torch.bool)
    n_hat, valid, _, _ = PL.local_frames(xyz, seg, sel, K=12, Kmin=6)
    assert int(valid.sum()) == 0 and torch.count_nonzero(n_hat) == 0
    print("PASS local_frames unobserved -> invalid")


def test_determinism():
    xyz, seg = make_slab()
    sel = xyz[:, 0].abs() < 0.06
    a = PL.local_frames(xyz, seg, sel)
    b = PL.local_frames(xyz, seg, sel)
    for x, y in zip(a, b):
        assert torch.equal(x, y), "local_frames is not deterministic"
    print("PASS local_frames determinism")


# --------------------------------------------------------------------------- #
# support_baseline                                                             #
# --------------------------------------------------------------------------- #
def test_support_baseline_matches_formula():
    xyz, _ = make_slab()
    K, Kmin, k_m = 12, 6, 32
    holdout, thin_div, gt = 0.10, 3.0, 8
    h, tr, th = PL.support_baseline(xyz, K=K, Kmin=Kmin, k_m=k_m,
                                    holdout_frac=holdout, thin_ratio_div=thin_div,
                                    grid_target=gt, seed=0)

    # manual replication of the categories.py baseline block (same seed)
    h2 = float(pick_cell_size(xyz, target=gt))
    rng = np.random.default_rng(0)
    is_hold = rng.random(xyz.shape[0]) < holdout
    hold, ref_main = xyz[is_hold], xyz[~is_hold]
    ref_t = ref_main[thin(ref_main, h2 / thin_div)]
    reach_hold, height_hold, _, _ = reach_height(ref_t, hold, K=K, Kmin=Kmin, M=k_m, h=h2)
    tr2, th2 = float(p99(reach_hold)), float(p99(height_hold))

    assert abs(h - h2) < 1e-6, f"h mismatch {h} vs {h2}"
    assert abs(tr - tr2) < 1e-6, f"reach p99 mismatch {tr} vs {tr2}"
    assert abs(th - th2) < 1e-6, f"height p99 mismatch {th} vs {th2}"
    print(f"PASS support_baseline matches formula (h={h:.4f}, reach={tr:.3f}, height={th:.3f})")


def test_support_baseline_empty():
    h, tr, th = PL.support_baseline(torch.zeros(0, 3))
    assert h == 0.0 and tr == float("inf") and th == float("inf")
    print("PASS support_baseline empty -> infinite thresholds")


# --------------------------------------------------------------------------- #
# feasibility_scales                                                           #
# --------------------------------------------------------------------------- #
def _ref_plane():
    xyz, _ = make_slab()
    h = float(pick_cell_size(xyz, target=8))
    return xyz, h


def test_feasibility_scales_keeps_on_surface():
    ref, h = _ref_plane()
    parent = torch.zeros(2, 3)
    dirs = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])   # in-plane
    maj = torch.full((2,), 0.02)
    keep, scale = PL.feasibility_scales(parent, dirs, maj, rho=0.5, ref_xyz=ref, h=h,
                                        thresh_reach=float("inf"), thresh_height=float("inf"))
    assert bool(keep.all()), "in-plane pair must be kept when thresholds allow"
    assert torch.allclose(scale, torch.ones(2)), "kept parents must keep scale == 1"
    print("PASS feasibility_scales keeps on-surface pair at scale 1")


def test_feasibility_scales_zero_dir_is_random():
    ref, h = _ref_plane()
    parent = torch.zeros(1, 3)
    dirs = torch.zeros(1, 3)                                   # no usable direction
    keep, _ = PL.feasibility_scales(parent, dirs, torch.ones(1), rho=0.5, ref_xyz=ref, h=h,
                                    thresh_reach=float("inf"), thresh_height=float("inf"))
    assert not bool(keep.any()), "zero direction must be marked not-placeable"
    print("PASS feasibility_scales zero direction -> random fallback")


def test_feasibility_scales_marks_offsurface_random():
    ref, h = _ref_plane()
    parent = torch.zeros(2, 3)
    dirs = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])   # off-surface
    maj = torch.full((2,), 0.05)
    keep, _ = PL.feasibility_scales(parent, dirs, maj, rho=0.5, ref_xyz=ref, h=h,
                                    thresh_reach=float("inf"), thresh_height=0.0,
                                    max_retries=1)
    assert not bool(keep.any()), "off-surface pair must fall back to random"
    print("PASS feasibility_scales off-surface pair -> random fallback")


def test_feasibility_scales_retracts_then_keeps():
    ref, h = _ref_plane()
    parent = torch.zeros(1, 3)
    dirs = torch.tensor([[0.0, 0.0, 1.0]])
    maj = torch.tensor([0.05])
    delta = 0.5 * 0.05                                        # rho * maj
    _, hf, _, _ = reach_height(ref, torch.tensor([[0.0, 0.0, delta]]), K=12, Kmin=6, M=32, h=h)
    _, hh, _, _ = reach_height(ref, torch.tensor([[0.0, 0.0, delta / 2]]), K=12, Kmin=6, M=32, h=h)
    thr = float((hf.item() + hh.item()) / 2.0)                # full fails, half passes
    assert hf.item() > thr >= hh.item(), \
        f"test setup: thresholds not straddling (hf={hf.item()}, hh={hh.item()})"

    keep, scale = PL.feasibility_scales(parent, dirs, maj, rho=0.5, ref_xyz=ref, h=h,
                                        thresh_reach=float("inf"), thresh_height=thr,
                                        max_retries=2)
    assert bool(keep.all()), "parent should survive after one retraction"
    assert abs(float(scale[0]) - 0.5) < 1e-6, f"scale must halve: got {float(scale[0])}"
    print("PASS feasibility_scales retracts scale by half then keeps")


def _run(fn):
    try:
        fn()
    except unittest.SkipTest as e:
        print(f"SKIP {fn.__name__}: {e}")


# --------------------------------------------------------------------------- #
# static: train/model/arguments wiring                                         #
# --------------------------------------------------------------------------- #
def test_placement_sources_wired():
    tr = open(os.path.join(ROOT, "train.py")).read()
    gm = open(os.path.join(ROOT, "scene", "gaussian_model.py")).read()
    arg = open(os.path.join(ROOT, "arguments", "__init__.py")).read()
    assert "local_frames" in tr and "feasibility_scales" in tr and "PlacementFrame" in tr, \
        "train.py does not build the placement frame"
    assert "placement=placement" in tr, "train.py does not pass placement to densify_and_prune"
    assert "boundary_normal" in tr, "train.py missing placement_mode gate"
    assert "_place_split_children" in gm and "_pad_placement" in gm, \
        "gaussian_model missing placement helpers"
    assert "placement" in gm.split("def densify_and_split")[1][:200], \
        "densify_and_split does not accept placement"
    assert "placement_mode" in arg, "arguments missing placement_mode"
    print("PASS placement sources wired (train.py / gaussian_model.py / arguments)")


# --------------------------------------------------------------------------- #
# GPU: densify_and_split placement branch (skipped without CUDA)               #
# --------------------------------------------------------------------------- #
def _needs_cuda(fn):
    def wrapper(*a, **k):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires CUDA")
        return fn(*a, **k)
    wrapper.__name__ = getattr(fn, "__name__", "test")
    return wrapper


def _build_model(seed=0, P=48, C=8):
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams
    from scene.gaussian_model import GaussianModel
    from utils.graphics_utils import BasicPointCloud

    torch.manual_seed(seed)
    parser = ArgumentParser()
    lp, op = ModelParams(parser), OptimizationParams(parser)
    args = parser.parse_args([])
    dataset, opt = lp.extract(args), op.extract(args)

    g = GaussianModel(dataset.sh_degree, seg_encoding_dim=dataset.seg_encoding_dim,
                      num_segmentation_classes=C)
    pts = torch.rand(P, 3) * 2 - 1
    pcd = BasicPointCloud(pts.numpy(), torch.rand(P, 3).numpy(), torch.zeros(P, 3).numpy())
    g.create_from_pcd(pcd, [], spatial_lr_scale=1.0)
    g.training_setup(opt)

    # percent_dense=0 -> clone disabled, every point split-eligible
    g.percent_dense = 0.0
    with torch.no_grad():
        g._scaling.copy_(torch.log(torch.rand(P, 3, device="cuda") * 0.5 + 0.2))
    g.tmp_radii = torch.zeros(P, device="cuda")
    return g


@_needs_cuda
def test_gpu_placement_child_positions():
    from sgdense.config import DensificationConfig

    g = _build_model(seed=0)
    n = int(g.get_xyz.shape[0])
    rho, N = 0.5, 2
    n_hat = torch.zeros(n, 3, device="cuda")
    n_hat[:, 0] = 1.0
    frame = PL.PlacementFrame(n_hat=n_hat,
                              valid=torch.ones(n, dtype=torch.bool, device="cuda"),
                              scale=torch.ones(n, device="cuda"))
    cfg = DensificationConfig(placement_rho=rho, placement_use_tangent_fallback=False)

    base = g.get_xyz.detach().clone()
    maj = g.get_scaling.max(dim=1).values.detach().clone()

    _, n_children = g.densify_and_split(
        torch.ones((n, 1), device="cuda"), 0.0, 1.0, N=N, cfg=cfg, placement=frame)
    assert n_children == N * n, f"expected {N*n} children, got {n_children}"

    children = g.get_xyz.detach()                       # all parents pruned -> children only
    exp0 = base - rho * maj[:, None] * n_hat            # frac -1
    exp1 = base + rho * maj[:, None] * n_hat            # frac +1
    assert torch.allclose(children[:n], exp0, atol=1e-4), "first tile != x - rho*s*n_hat"
    assert torch.allclose(children[n:], exp1, atol=1e-4), "second tile != x + rho*s*n_hat"
    print(f"PASS GPU placement positions ({n} parents -> {n_children} children at +/-rho*s*n_hat)")


@_needs_cuda
def test_gpu_default_is_vanilla_random():
    # placement=None: two same-seed models produce identical child positions
    g1, g2 = _build_model(seed=1), _build_model(seed=1)
    n = int(g1.get_xyz.shape[0])
    for g in (g1, g2):
        g.tmp_radii = torch.zeros(n, device="cuda")

    torch.manual_seed(123)
    g1.densify_and_split(torch.ones((n, 1), device="cuda"), 0.0, 1.0, N=2)
    torch.manual_seed(123)
    g2.densify_and_split(torch.ones((n, 1), device="cuda"), 0.0, 1.0, N=2)
    assert torch.allclose(g1.get_xyz, g2.get_xyz), "default random path is not reproducible"
    print("PASS GPU default (placement=None) is vanilla random and reproducible")


if __name__ == "__main__":
    print("torch", torch.__version__)
    test_placement_sources_wired()
    test_boundary_normal_recovers_separating_direction()
    test_single_class_is_invalid()
    test_unobserved_is_invalid()
    test_determinism()
    test_support_baseline_matches_formula()
    test_support_baseline_empty()
    test_feasibility_scales_keeps_on_surface()
    test_feasibility_scales_zero_dir_is_random()
    test_feasibility_scales_marks_offsurface_random()
    test_feasibility_scales_retracts_then_keeps()
    if not torch.cuda.is_available():
        print("SKIP GPU tests (no CUDA on this host)")
    else:
        _run(test_gpu_placement_child_positions)
        _run(test_gpu_default_is_vanilla_random)
    print("ALL PLACEMENT TESTS PASSED")
