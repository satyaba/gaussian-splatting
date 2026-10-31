# CPU unit tests for the SGDense support modules (grid kNN / reach-height,
# thinning, texture-frequency helpers). No GPU required.
#   python tests/test_sgdense.py
#   pytest tests/test_sgdense.py
import os
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import torch

from sgdense.geometric import build_grid, grid_knn, reach_height, thin, p99


def _brute_knn(ref, qry, K, h):
    """Reference KNN within radius h: distances + indices into ref."""
    d = torch.cdist(qry, ref)                       # [Q, N]
    out_d = torch.full((qry.shape[0], K), float("inf"))
    out_i = torch.full((qry.shape[0], K), -1, dtype=torch.long)
    for qi in range(qry.shape[0]):
        di = d[qi]
        keep = torch.nonzero(di <= h, as_tuple=False).squeeze(1)
        if keep.numel() == 0:
            continue
        vals, order = torch.sort(di[keep])
        vals, order = vals[:K], order[:K]
        out_d[qi, : vals.numel()] = vals
        out_i[qi, : order.numel()] = keep[order]
    return out_d, out_i


def test_build_grid_knn_matches_bruteforce():
    torch.manual_seed(0)
    ref = torch.rand(500, 3)
    qry = torch.rand(200, 3)
    h = 0.3
    g = build_grid(ref, h)
    d, i = grid_knn(g, qry, K=8, M=32)
    bd, bi = _brute_knn(ref, qry, 8, h)

    # Distances must agree wherever both find neighbours; grid_knn must never
    # report a nearer neighbour than brute force within the radius.
    finite = torch.isfinite(bd)
    assert torch.allclose(d[finite], bd[finite], atol=1e-5), "grid_knn distances differ from brute force"
    # Indices into ref must carry the same distances.
    gathered = ref[i.clamp(min=0)] - qry[:, None, :]
    gd = gathered.norm(dim=-1)
    assert torch.allclose(gd[finite], bd[finite], atol=1e-5), "grid_knn indices inconsistent with distances"
    print("PASS grid_knn matches brute force")


def test_thin_one_per_voxel():
    # Two points within one 0.1-voxel collapse to one survivor.
    pts = torch.tensor([[0.01, 0.01, 0.01],
                        [0.02, 0.02, 0.02],
                        [0.55, 0.55, 0.55]])
    keep = thin(pts, 0.1)
    assert keep.numel() == 2, f"expected 2 survivors, got {keep.numel()}"
    print("PASS thin keeps one point per voxel")


def test_reach_height_plane():
    # A dense plane at z=0; queries on the plane should be well-supported
    # (small height), a query far above it should have a large height.
    xs = torch.linspace(0, 1, 40)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    ref = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(40 * 40)], dim=1)
    on_plane = torch.tensor([[0.5, 0.5, 0.0]])
    above = torch.tensor([[0.5, 0.5, 5.0]])
    qry = torch.cat([on_plane, above], dim=0)

    reach, height, unsupported, h = reach_height(ref, qry, K=12, Kmin=6, M=32)
    assert not unsupported[0], "on-plane query must be supported"
    assert height[0] < height[1], "query above the plane must have a larger height"
    print(f"PASS reach_height plane (h={h:.3f}, height_on={height[0]:.3f}, height_above={height[1]:.3f})")


def test_p99_ignores_inf():
    x = torch.tensor([1.0, 2.0, 3.0, float("inf")])
    assert p99(x) == 3.0, "p99 must ignore non-finite entries"
    print("PASS p99 ignores non-finite")


def test_class_freq_thres_monotone():
    import numpy as np
    from sgdense.texture import class_freq_thres
    old = 1.0
    s = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    t = class_freq_thres(old, s, beta=0.5)
    # Higher texture score -> lower threshold.
    assert t[0] > t[1] > t[2], f"class_freq_thres not monotonically decreasing: {t}"
    assert abs(t[1] - old) < 1e-6, "neutral s_tilde=0.5 must leave the threshold unchanged"
    print("PASS class_freq_thres monotone")


def _run(fn):
    try:
        fn()
    except unittest.SkipTest as e:
        print(f"SKIP {fn.__name__}: {e}")


if __name__ == "__main__":
    print("torch", torch.__version__)
    for fn in (test_build_grid_knn_matches_bruteforce, test_thin_one_per_voxel,
               test_reach_height_plane, test_p99_ignores_inf, test_class_freq_thres_monotone):
        _run(fn)
    print("ALL SGDENSE TESTS PASSED")
