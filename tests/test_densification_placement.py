# Acceptance test for segmentation-gated densification placement (T0 deterministic
# split placement; design doc 04 §4 / §7).
# Run on a GPU machine with the rebuilt extension installed:
#   python tests/test_densification_placement.py
# (pytest-compatible: pytest tests/test_densification_placement.py)
#
# Covers:
#   §7.1  py_compile of the edited files
#   §7.1' static: the T0 fix is present AND wired onto the model (the two failure
#         modes that survived the first fix: the 'offsite' typo and the missing
#         __init__/training_setup assignment -> AttributeError at first split)
#   §7.2  routing mask algebra (spec-level; the T1 block is inline in train.py)
#   §7.3  _pad_mask semantics               [SKIP until T1 lands]
#   §7.7  T0 determinism + random-path ablation parity  (real densify_and_split)
#   --    a real GPU densify_and_split placement check (children at +/- rho*sigma_major)
# §7.5 THE SPIKE and §7.6 threshold calibration need a trained checkpoint +
# gt_segmentation and live in a separate notebook step, not here.
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
    # caught by the __main__ runner below — unlike pytest.skip(), which raises an
    # unhandled Skipped (pytest is installed in Colab) and aborts a script run.
    def wrapper(*a, **k):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires CUDA")
        return fn(*a, **k)
    wrapper.__name__ = getattr(fn, "__name__", "test")
    return wrapper


# --------------------------------------------------------------------------- #
# §7.1 / §7.1'  static checks                                                 #
# --------------------------------------------------------------------------- #
def test_py_compile_edited_files():
    for f in (GM, ARG, TRAIN):
        py_compile.compile(f, doraise=True)
    print("PASS py_compile (gaussian_model.py, arguments/__init__.py, train.py)")


def test_t0_sources_present_and_wired():
    gm, arg = open(GM).read(), open(ARG).read()

    # arguments define both knobs, with the correct spelling.
    assert re.search(r"self\.seg_split_deterministic\s*=", arg), \
        "arguments: seg_split_deterministic missing"
    assert re.search(r"self\.seg_split_offset_ratio\s*=", arg), \
        "arguments: seg_split_offset_ratio missing (the 'offsite' typo?)"
    assert "offsite" not in arg, "'offsite' misspelling present in arguments/__init__.py"

    # the split block is guarded and reads the ratio; major_idx is bound first.
    assert "if self.seg_split_deterministic:" in gm, \
        "densify_and_split is not guarded by seg_split_deterministic"
    assert "self.seg_split_offset_ratio" in gm, \
        "densify_and_split does not read seg_split_offset_ratio"
    assert re.search(r"major_len\s*,\s*major_idx\s*=", gm), \
        "major_idx is not bound before its scatter_() use (old NameError)"

    # THE WIRING: the flags must be set on the model, not only on the args
    # namespace, or self.seg_split_deterministic raises AttributeError at runtime.
    assert re.search(r"self\.seg_split_deterministic\s*=", gm), \
        "GaussianModel never assigns self.seg_split_deterministic (AttributeError at first split)"
    assert re.search(r"self\.seg_split_offset_ratio\s*=", gm), \
        "GaussianModel never assigns self.seg_split_offset_ratio (AttributeError at first split)"
    print("PASS T0 sources present and wired onto the model")


def test_t1_sources_present_or_skip():
    gm, arg, tr = open(GM).read(), open(ARG).read(), open(TRAIN).read()
    present = ("seg_route_enabled" in arg) and ("_pad_mask" in gm) and ("seg_route_enabled" in tr)
    if not present:
        raise unittest.SkipTest("T1 routing block not implemented yet (doc 04 §5)")
    assert all(k in gm for k in ("clone_mask", "split_mask", "floater_mask")), \
        "densify_and_prune does not accept regime masks"
    assert re.search(r"def _pad_mask", gm), "_pad_mask helper missing"
    print("PASS T1 sources present")


# --------------------------------------------------------------------------- #
# §7.2  T0 layout math (pure torch, CPU)                                       #
# --------------------------------------------------------------------------- #
def test_t0_layout_math():
    K, N, rho = 6, 2, 0.5
    torch.manual_seed(0)
    sigma = torch.rand(K, 3) * 0.5 + 0.05                    # positive scalings
    major_len, major_ix = sigma.max(dim=1)
    fracs = torch.linspace(-1.0, 1.0, N)
    offs = torch.zeros(K, N, 3)
    offs.scatter_(2, major_ix[:, None, None].expand(K, N, 1),
                  (rho * major_len)[:, None].mul(fracs[None, :]).unsqueeze(-1))
    s = offs.reshape(K * N, 3)
    for k in range(K):
        c0, c1 = s[k * N], s[k * N + 1]
        e = torch.zeros(3)
        e[major_ix[k]] = rho * major_len[k]
        off = torch.ones(3, dtype=torch.bool)
        off[major_ix[k]] = False
        assert torch.allclose(c0, -e, atol=1e-6), f"child0 of parent {k} != -rho*sigma_major"
        assert torch.allclose(c1, e, atol=1e-6), f"child1 of parent {k} != +rho*sigma_major"
        assert torch.allclose(c0, -c1, atol=1e-6), f"children of parent {k} not symmetric"
        assert torch.allclose(c1[off], torch.zeros(int(off.sum())), atol=1e-6), \
            f"parent {k}: offset leaked onto a non-major axis"
    print("PASS T0 layout math (+/- rho*sigma_major on major axis, symmetric, order k*N+j)")


# --------------------------------------------------------------------------- #
# §7.2'  routing mask algebra (spec-level; mirrors train.py's inline block)    #
# --------------------------------------------------------------------------- #
def _route(disagree, structure, S, v_min, q_disagree, q_structure):
    """Mirror of the mask construction proposed in doc 04 §5 (train.py).

    Returns (interior, boundary, floater), each a bool mask over the same
    population, or (None, None, None) when there is no evidence.
    """
    act = S > v_min
    if not bool(act.any()):
        return None, None, None
    tau_d = torch.quantile(disagree[act], q_disagree)
    tau_s = torch.quantile(structure[act], q_structure)
    boundary = act & (disagree >= tau_d) & (structure >= tau_s)
    floater = act & (disagree >= tau_d) & (structure < tau_s)
    interior = act & (disagree < tau_d)
    return interior, boundary, floater


def test_routing_mask_algebra():
    torch.manual_seed(1)
    P = 500
    disagree, structure, S = torch.rand(P), torch.rand(P), torch.rand(P) * 3
    interior, boundary, floater = _route(disagree, structure, S, 1.0, 0.75, 0.5)
    act = S > 1.0

    assert torch.equal(interior | boundary | floater, act), "3 masks do not partition the active set"
    overlap = int((interior & boundary).sum() + (interior & floater).sum() + (boundary & floater).sum())
    assert overlap == 0, f"regime masks overlap on {overlap} Gaussians"

    none = _route(torch.zeros(4), torch.zeros(4), torch.zeros(4), 1.0, 0.75, 0.5)
    assert all(x is None for x in none), "no-evidence case must yield None masks (vanilla fallback)"
    print("PASS routing algebra (partition, disjoint, None when no evidence)")


# --------------------------------------------------------------------------- #
# §7.7 + placement  real GPU densify_and_split                                 #
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


def _do_split(g, N=2):
    n = g.get_xyz.shape[0]
    px = g.get_xyz.detach().clone()
    ps = g.get_scaling.detach().clone()
    pr = g.get_rotation.detach().clone()
    g.densify_and_split(torch.ones((n, 1), device="cuda"), 0.0, 1.0, N=N)  # all eligible
    return px, ps, pr, g.get_xyz.detach().clone(), g.seg_split_offset_ratio


@_needs_cuda
def test_gpu_split_placement():
    from utils.general_utils import build_rotation

    px, ps, pr, children, rho = _do_split(build_model(seed=0))
    R = build_rotation(pr)
    major_len, major_ix = ps.max(dim=1)
    K = px.shape[0]
    assert children.shape[0] == K * 2, f"expected {K * 2} children, got {children.shape[0]}"
    # Row i of the appended block belongs to parent k = i % K with offset index
    # j = i // K (this mirrors the .repeat(N,1) tiling used for base/rotation).
    fracs = torch.linspace(-1.0, 1.0, 2, device="cuda")
    for i in range(K * 2):
        k, j = i % K, i // 2
        off = torch.zeros(3, device="cuda")
        off[major_ix[k]] = fracs[j] * rho * major_len[k]
        assert torch.allclose(children[i], px[k] + R[k] @ off, atol=1e-4), \
            f"child {i} (parent {k}, offset {j}) misplaced"
    print(f"PASS GPU T0 placement ({K} parents -> {children.shape[0]} children, rho={rho})")


@_needs_cuda
def test_t0_determinism():
    _, _, _, a, _ = _do_split(build_model(seed=0))
    _, _, _, b, _ = _do_split(build_model(seed=0))
    assert torch.allclose(a, b), "T0 placement is not deterministic across identical seeds"
    print("PASS T0 determinism (identical seed -> identical child layout)")


@_needs_cuda
def test_random_ablation_differs():
    g = build_model(seed=0)
    g.seg_split_deterministic = False
    _, _, _, rnd, _ = _do_split(g)
    _, _, _, det, _ = _do_split(build_model(seed=0))
    assert not torch.allclose(det, rnd), \
        "seg_split_deterministic=False produced identical children — flag not honored"
    print("PASS ablation: seg_split_deterministic=False -> stochastic path differs")


def _run(fn):
    try:
        fn()
    except unittest.SkipTest as e:
        print(f"SKIP {fn.__name__}: {e}")


if __name__ == "__main__":
    print("torch", torch.__version__, "cuda", torch.version.cuda)
    _run(test_py_compile_edited_files)
    _run(test_t0_sources_present_and_wired)
    _run(test_t1_sources_present_or_skip)
    _run(test_t0_layout_math)
    _run(test_routing_mask_algebra)
    if not torch.cuda.is_available():
        print("SKIP GPU tests (no CUDA on this host)")
    else:
        _run(test_gpu_split_placement)
        _run(test_t0_determinism)
        _run(test_random_ablation_differs)
    print("ALL DENSIFICATION-PLACEMENT TESTS PASSED")
