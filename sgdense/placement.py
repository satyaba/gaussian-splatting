"""Segmentation-boundary densification placement (SGDense, docs/08).

Pure-function helpers that turn a Gaussian's persisted segmentation histogram
``M`` (docs/03) into a *placement frame* for split children, so the
densification geometry follows the segmentation read-out instead of the
vanilla isotropic random ``N(0, Sigma)``.

The starting design (docs/08 section 4.1) is:

    n_hat = local class-discrimination direction
            (from the argmax labels of the kNN neighbours of a candidate
            parent), projected onto the local surface tangent plane so the
            children stay on the surface while still separating the two
            classes.

Three functions:

* :func:`local_frames`     -- per-parent ``n_hat`` + surface frame.
* :func:`support_baseline` -- the hold-out p99 reach/height baseline that the
                              feasibility gate reuses (docs/08 section 6.5).
* :func:`feasibility_scales` -- per-parent retraction factor for the ``+/-``
                              child pair (or invalid -> random fallback).

Reuses the grid kNN and reach/height machinery from :mod:`sgdense.geometric`.
Everything here is ``torch.no_grad`` and CPU-testable (no live ``GaussianModel``,
no CUDA), mirroring ``sgdense/densify_grad.py`` and ``sgdense/geometric.py``.
This module is additive: nothing imports it until ``train.py`` is wired in
behind ``placement_mode='boundary_normal'`` (docs/08 section 4.3, step 5).
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np
import torch

from sgdense.geometric import build_grid, pick_cell_size, grid_knn, reach_height, thin, p99

_EPS = 1e-8

# Placement frame consumed by ``GaussianModel.densify_and_split`` when
# ``placement_mode == 'boundary_normal'`` (docs/08 section 4.2). It is a plain
# namedtuple so the model can duck-type it (``placement.n_hat``) without
# importing this module -- keeping the model edit revertible on its own.
#   n_hat  [P,3] unit effective direction (0 where invalid)
#   valid  [P]   bool, True -> place deterministically; False -> random N(0,Sigma)
#   scale  [P]   float offset factor (feasibility retraction; >=0, default 1)
PlacementFrame = namedtuple("PlacementFrame", ["n_hat", "valid", "scale"])


def _surface_frame(nbr, mask, dev):
    """Local PCA frame from masked neighbour positions.

    ``nbr`` [S,K,3], ``mask`` [S,K] bool. Returns ``(centroid, normal, tangent)``
    as [S,3] tensors; ``normal`` is the smallest-eigenvalue direction (surface
    normal) and ``tangent`` the largest-eigenvalue direction (max in-plane
    spread). Eigen-decomposition is done on CPU in float64, matching
    ``sgdense.geometric`` (batched ``eigh`` on GPU allocates ~50 GiB there).
    """
    mf = mask.float()
    n = mf.sum(1).clamp(min=1)                                  # [S]
    c = (nbr * mf[..., None]).sum(1) / n[:, None]              # [S,3]
    diff = (nbr - c[:, None]) * mf[..., None]                  # [S,K,3]
    cov = torch.einsum('ski,skj->sij', diff, diff) / n[:, None, None]   # [S,3,3]

    cov64 = cov.double().cpu().numpy()                          # [S,3,3]
    _, evec = np.linalg.eigh(cov64)                             # ascending eigenvalues
    evec = torch.from_numpy(evec).float().to(dev)               # [S,3,3] columns
    normal = evec[..., 0]                                       # smallest -> surface normal
    tangent = evec[..., 2]                                      # largest  -> max spread
    return c, normal, tangent


@torch.no_grad()
def local_frames(xyz, seg_hist, sel_mask, *, K=12, Kmin=6, k_m=32, h=None,
                 opacity=None, obs_evidence=0.0, grid_target=8):
    """Per-parent local placement frame for boundary split candidates.

    Parameters
    ----------
    xyz : [P,3] float tensor
        Gaussian means (any device; CPU is fine for tests).
    seg_hist : [P,C] float tensor
        Persisted segmentation mass histogram ``M`` (docs/03).
    sel_mask : [P] bool tensor
        Candidate parents (the boundary set from the category taxonomy).
    K, Kmin, k_m : int
        kNN neighbourhood size, minimum neighbourhood for a valid direction,
        and max points read from one grid cell (``sgdense.geometric``).
    h : float, optional
        Grid cell size. ``None`` -> ``pick_cell_size`` (mode of the taxonomy).
    opacity : [P] float tensor, optional
        Activated opacities; used to weight neighbour votes. ``None`` -> 1.
    obs_evidence : float
        Rows with ``sum_c M[g,c] <= obs_evidence`` are excluded as reference
        neighbours (unobserved Gaussians cast no vote).
    grid_target : int
        Target points per grid cell for ``pick_cell_size``.

    Returns
    -------
    n_hat : [P,3] float tensor
        Unit class-discrimination direction, projected onto the local tangent
        plane; **zero** on non-selected or invalid rows (caller falls back).
    valid : [P] bool tensor
        True where ``n_hat`` is a usable direction.
    n_surf : [P,3] float tensor
        Local surface normal from neighbour PCA (zero where not computed).
    tangent : [P,3] float tensor
        Local max-spread in-plane direction; the fallback direction.
    """
    dev = xyz.device
    P = int(xyz.shape[0])
    C = int(seg_hist.shape[1]) if seg_hist.dim() == 2 else 0

    n_hat = torch.zeros(P, 3, device=dev)
    valid = torch.zeros(P, dtype=torch.bool, device=dev)
    n_surf = torch.zeros(P, 3, device=dev)
    tangent = torch.zeros(P, 3, device=dev)

    sel_mask = sel_mask.bool()
    sel_idx = sel_mask.nonzero(as_tuple=True)[0]
    if C < 2 or sel_idx.numel() == 0:
        return n_hat, valid, n_surf, tangent

    # --- reference set: observed Gaussians -------------------------------------
    obs = seg_hist.sum(dim=1) > obs_evidence                      # [P]
    obs_ids = obs.nonzero(as_tuple=True)[0]                       # [R]
    if obs_ids.numel() < Kmin:
        return n_hat, valid, n_surf, tangent
    ref = xyz[obs_ids].contiguous()                               # [R,3]

    h = float(pick_cell_size(ref, target=grid_target)) if h is None else float(h)
    g = build_grid(ref, h)

    q = xyz[sel_idx].contiguous()                                 # [S,3]
    d, j = grid_knn(g, q, K=K, M=k_m)                             # [S,K], [S,K] -> ref
    m = j >= 0                                                    # [S,K] real neighbours
    jj = j.clamp(min=0)

    nbr = ref[jj]                                                 # [S,K,3]
    nb_hist = seg_hist[obs_ids[jj]]                               # [S,K,C]
    nb_mass = nb_hist.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    nb_conf = (nb_hist / nb_mass).max(dim=-1).values              # [S,K]
    lab = nb_hist.argmax(dim=-1)                                  # [S,K]

    # --- parent top-2 classes --------------------------------------------------
    ph = seg_hist[sel_idx]                                        # [S,C]
    pp = ph / ph.sum(dim=-1, keepdim=True).clamp_min(_EPS)
    top2v, top2i = pp.topk(2, dim=1)                              # [S,2]
    a = top2i[:, 0][:, None]                                      # [S,1]
    b = top2i[:, 1][:, None]                                      # [S,1]
    have2 = top2v[:, 1] > 0                                       # second class present

    # --- vote weights ----------------------------------------------------------
    w = nb_conf / (d + _EPS)                                      # closer + confident
    if opacity is not None:
        w = w * opacity[obs_ids[jj]]
    w = torch.where(m, w, torch.zeros_like(w))                    # [S,K]

    disp = nbr - q[:, None, :]                                    # [S,K,3]
    is_a = (lab == a) & m
    is_b = (lab == b) & m
    both = is_a.any(dim=1) & is_b.any(dim=1)
    sa = (disp * (w * is_a.float())[..., None]).sum(1)            # [S,3] toward class a
    sb = (disp * (w * is_b.float())[..., None]).sum(1)            # [S,3] toward class b
    n_class = sb - sa                                             # points a -> b

    # --- local surface frame ---------------------------------------------------
    _, n_s, tan_s = _surface_frame(nbr, m, dev)                   # [S,3] each

    # --- project onto the tangent plane ---------------------------------------
    # children stay on the surface while separating the two classes
    n_proj = n_class - (n_class * n_s).sum(dim=1, keepdim=True) * n_s
    proj_norm = n_proj.norm(dim=1)
    n_class_norm = n_class.norm(dim=1)

    nbr_count = m.sum(dim=1)
    ok = (both & have2
          & (nbr_count >= Kmin)
          & (n_class_norm > _EPS)
          & (proj_norm > _EPS))

    n_hat[sel_idx[ok]] = n_proj[ok] / proj_norm[ok][:, None]
    valid[sel_idx] = ok
    n_surf[sel_idx] = n_s
    tangent[sel_idx] = tan_s
    return n_hat, valid, n_surf, tangent


@torch.no_grad()
def support_baseline(ref_xyz, *, K=12, Kmin=6, k_m=32, holdout_frac=0.10,
                     thin_ratio_div=3.0, grid_target=8, seed=0):
    """Hold-out p99 reach/height baseline for the feasibility gate.

    Replicates the baseline block of ``sgdense.categories.generate_category_masks``
    (docs/06 section 3, docs/08 section 6.5): a random hold-out of the trusted
    reference set is scored with ``reach_height`` against a thinned remainder,
    and the 99th percentiles become the "normal" reach/height.

    Returns ``(h, thresh_reach, thresh_height)``; the thresholds are ``inf`` when
    the reference set is too small to form a baseline (gate then rejects nothing).
    """
    R = int(ref_xyz.shape[0])
    if R == 0:
        return 0.0, float('inf'), float('inf')
    h = float(pick_cell_size(ref_xyz, target=grid_target))

    rng = np.random.default_rng(seed)
    is_hold = rng.random(R) < holdout_frac
    hold = ref_xyz[is_hold]
    ref_main = ref_xyz[~is_hold]
    if hold.shape[0] == 0 or ref_main.shape[0] == 0:
        return h, float('inf'), float('inf')

    ref_t = ref_main[thin(ref_main, h / thin_ratio_div)]
    if ref_t.shape[0] == 0:
        return h, float('inf'), float('inf')

    reach_hold, height_hold, _, _ = reach_height(
        ref_t, hold, K=K, Kmin=Kmin, M=k_m, h=h)
    tr = _finite_p99(reach_hold)
    th = _finite_p99(height_hold)
    return h, tr, th


def _finite_p99(x):
    fin = x[torch.isfinite(x)]
    if fin.numel() == 0:
        return float('inf')
    return float(p99(x))


@torch.no_grad()
def feasibility_scales(parent_xyz, dirs, maj, *, rho, ref_xyz, h, thresh_reach,
                       thresh_height, K=12, Kmin=6, k_m=32, max_retries=2):
    """Per-parent retraction factor for the symmetric ``+/-`` child pair.

    A parent with a usable ``dir`` proposes the pair
    ``x +/- rho * scale * maj * dir``. Both children are tested with the
    geometric support test (``reach_height``); on rejection the offset is halved
    (``scale <- scale/2``) and re-tested. A parent still rejected after
    ``max_retries`` is marked ``keep=False`` so the caller falls back to random
    ``N(0, Sigma)``. The offset is never driven to zero (docs/08 section 3.1
    locked decision 6).

    Parameters
    ----------
    parent_xyz : [P,3] candidate parent means
    dirs       : [P,3] unit effective direction (0 -> not placeable)
    maj        : [P]   parent major scale (``max(get_scaling)``)
    ref_xyz    : [R,3] trusted reference points for the support test

    Returns ``(keep [P] bool, scale [P] float)``; ``keep`` False -> random path.
    """
    dev = parent_xyz.device
    P = int(parent_xyz.shape[0])
    keep = dirs.norm(dim=1) > 0
    scale = torch.ones(P, device=dev, dtype=parent_xyz.dtype)

    for t in range(max_retries + 1):
        live = keep.nonzero(as_tuple=True)[0]
        if live.numel() == 0:
            break
        d = (rho * scale[live] * maj[live])[:, None] * dirs[live]      # [L,3]
        kids = torch.cat([parent_xyz[live] + d, parent_xyz[live] - d], dim=0)
        reach, height, unsupported, _ = reach_height(
            ref_xyz, kids, K=K, Kmin=Kmin, M=k_m, h=h)
        L = live.numel()
        ok = (~unsupported) & (reach <= thresh_reach) & (height <= thresh_height)
        both = ok[:L] & ok[L:]                                        # both signs pass
        bad = live[~both]
        if bad.numel() == 0:
            break
        if t == max_retries:
            keep[bad] = False
        else:
            scale[bad] = scale[bad] * 0.5
    return keep, scale
