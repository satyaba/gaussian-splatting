"""Read-outs from the persisted per-Gaussian segmentation mass histogram M."""

import torch


def seg_hist_readout(seg_hist, eps=1e-8):
    """Read-outs from the persisted mass histogram M.

    Returns (consistency, structure, disagree, entropy, top2, S):
      consistency = max_c P[g, c]
      structure   = P_(1) + P_(2)        (boundary-ness)
      disagree    = 1 - P_(1)
      entropy     = -sum_c P log2 P
      top2        = [N, 2] the two largest P values
      S           = sum_c M[g, c]        (evidence = mass, NOT count)

    Rows with S == 0 give P == 0, so their 'disagree' looks maximal -- gate
    them out with S (`S > seg_hist_min_evidence`), never with disagree.
    """
    if seg_hist.numel() == 0:
        n = seg_hist.shape[0] if seg_hist.dim() > 0 else 0
        z = torch.zeros((n,), device=seg_hist.device)
        return z, z, z, z, torch.zeros((n, 0), device=seg_hist.device), z

    S = seg_hist.sum(dim=1)
    P = seg_hist / S.clamp_min(eps).unsqueeze(1)

    p = P + 1e-10
    entropy = -(p * torch.log2(p)).sum(dim=-1)
    k = min(2, P.shape[1])
    top2 = torch.topk(P, k=k, dim=1).values
    consistency = top2[:, 0]
    structure = top2[:, 0] + top2[:, 1] if k == 2 else top2[:, 0]
    disagree = 1.0 - top2[:, 0]
    return consistency, structure, disagree, entropy, top2, S
