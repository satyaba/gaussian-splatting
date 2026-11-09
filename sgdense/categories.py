"""6-way Gaussian category taxonomy (SGDense).

Level 1 (segmentation-histogram only): a Gaussian is classified from its
persisted mass histogram read-out (top-2 probabilities, entropy, evidence)
plus an opacity gate into {unobserved, interior, ambiguous, floaters, default}.

Level 2 (geometric support test): the still-'default' Gaussians are scored
against the trusted 'interior' set with the 3D-grid reach/height test
(sgdense.geometric). Supported -> boundary; unsupported-in-both -> floaters.

The returned masks are numpy bool arrays aligned to the pre-densification
population.
"""

import numpy as np
import torch

from sgdense.geometric import pick_cell_size, reach_height, thin, p99

CATEGORY_NAMES = ["unobserved", "interior", "ambiguous", "floaters", "default", "boundary"]


def generate_category_masks(N, top2_vals, sum_vals, entropies, coordinates, opacities,
                            opacity_thres, evidence_min, cfg):
    """Build the per-Gaussian category masks.

    N            : population size
    top2_vals    : [N,2] top-2 class probabilities P (from seg_hist_readout)
    sum_vals     : [N]   evidence S = sum_c M[g,c]
    entropies    : [N]   histogram entropy
    coordinates  : [N,3] Gaussian means
    opacities    : [N]   activated opacities
    opacity_thres: floater opacity gate
    evidence_min : opt.seg_hist_min_evidence (S gate)
    cfg          : sgdense.config.DensificationConfig

    Returns (unobserved, default, interior, floaters, ambiguous, boundary).
    """
    lab = np.full(N, CATEGORY_NAMES.index("default"), dtype=np.int8)

    top2_np = top2_vals.detach().cpu().numpy()
    S_np = sum_vals.detach().cpu().numpy()
    e_np = entropies.detach().cpu().numpy()
    opac_np = opacities.detach().cpu().numpy()
    xyz = coordinates.detach().cpu().numpy()

    observed = S_np > evidence_min

    p1 = top2_np[:, 0]
    p2 = top2_np[:, 1]

    # Residual probability mass outside the top-2 classes (range [0, 1]).
    # tau_tail_high is a *probability* threshold, so `tail` must be in the same
    # units as p1/p2 -- NOT evidence mass S. The run26 notebook wrote
    # `S - p1 - p2`, which mixed units: since S grows with observations, the
    # `tail > tau_tail_high` gate became all-but-vacuous for well-observed
    # Gaussians and inverted its intent. Corrected to 1 - p1 - p2 (docs/06 §3).
    tail = 1.0 - p1 - p2

    lab[observed & (p1 > cfg.tau_p1) & (e_np < cfg.tau_entropy_low)] = CATEGORY_NAMES.index("interior")
    lab[observed & (p1 <= cfg.tau_p1) & (p2 > cfg.tau_p2_high)
        & (e_np > cfg.tau_entropy_low) & (e_np <= cfg.tau_entropy_high)
        & (opac_np < opacity_thres)] = CATEGORY_NAMES.index("floaters")
    lab[observed & (tail > cfg.tau_tail_high) & (e_np > cfg.tau_entropy_high)] = CATEGORY_NAMES.index("ambiguous")
    lab[observed & (tail > cfg.tau_tail_high) & (e_np > cfg.tau_entropy_high)
        & (opac_np < opacity_thres)] = CATEGORY_NAMES.index("floaters")

    lab[~observed] = CATEGORY_NAMES.index("unobserved")

    # ---- Level 2: geometric support test (3D-grid kNN) --------------------------------
    # Trusted = interior; queries = still-default. Default Gaussians are re-labelled
    # boundary (supported) or floaters (unsupported in both reach and height).
    ref_ids = np.flatnonzero(lab == CATEGORY_NAMES.index("interior"))
    qry_ids = np.flatnonzero(lab == CATEGORY_NAMES.index("default"))

    if ref_ids.size > 0 and qry_ids.size > 0:
        xyz_t = torch.from_numpy(xyz).float().cuda()
        ref = xyz_t[torch.from_numpy(ref_ids).cuda()].contiguous()
        qry = xyz_t[torch.from_numpy(qry_ids).cuda()].contiguous()

        h = pick_cell_size(ref, target=cfg.grid_target)

        # Hold-out subsample sets the "normal" reach/height baseline (p99).
        rng = np.random.default_rng(0)
        is_hold = rng.random(len(ref_ids)) < cfg.holdout_frac
        hold = ref[torch.from_numpy(is_hold).cuda()]
        ref_main = ref[torch.from_numpy(~is_hold).cuda()]

        # Guard: geometric test needs a non-empty hold-out and a thinned reference set.
        if hold.shape[0] > 0 and ref_main.shape[0] > 0:
            ref_t = ref_main[thin(ref_main, h / cfg.thin_ratio_div)]

            reach_query, height_query, unsupported_query, _ = reach_height(
                ref_t, qry, K=cfg.knn_k, Kmin=cfg.knn_kmin, M=cfg.knn_m, h=h)
            reach_hold, height_hold, _, _ = reach_height(
                ref_t, hold, K=cfg.knn_k, Kmin=cfg.knn_kmin, M=cfg.knn_m, h=h)

            thresh_reach, thresh_height = p99(reach_hold), p99(height_hold)
            supported_query = ~unsupported_query
            lo_reach = reach_query <= thresh_reach
            lo_height = height_query <= thresh_height
            quad = {
                "supported": supported_query & lo_reach & lo_height,
                "inspect":   supported_query & lo_reach & ~lo_height,
                "defer":     supported_query & ~lo_reach & lo_height,
                "floater":   supported_query & ~lo_reach & ~lo_height,
                "unsupported": unsupported_query,
            }

            boundary_ids = qry_ids[quad['supported'].cpu().numpy()]
            lab[boundary_ids] = CATEGORY_NAMES.index("boundary")

            if cfg.lvl2_floaters:
                floater_ids = qry_ids[quad['floater'].cpu().numpy()]
                lab[floater_ids] = CATEGORY_NAMES.index("floaters")

    unobserved_mask = np.where(lab == CATEGORY_NAMES.index("unobserved"), True, False)
    default_mask = np.where(lab == CATEGORY_NAMES.index("default"), True, False)
    interior_mask = np.where(lab == CATEGORY_NAMES.index("interior"), True, False)
    floaters_mask = np.where(lab == CATEGORY_NAMES.index("floaters"), True, False)
    ambiguous_mask = np.where(lab == CATEGORY_NAMES.index("ambiguous"), True, False)
    boundary_mask = np.where(lab == CATEGORY_NAMES.index("boundary"), True, False)

    return unobserved_mask, default_mask, interior_mask, floaters_mask, ambiguous_mask, boundary_mask
