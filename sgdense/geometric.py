"""
Grid-based geometric support test for Gaussians (reach / height).

GOAL
    For each query Gaussian (e.g. class-1), measure how well trusted reference
    Gaussians (e.g. class-2 "interior") support it:
      reach  = how far the nearest trusted neighbours are (sideways)
      height = how far the query sits off the plane those neighbours form
    Both are divided by the local spacing of the trusted Gaussians, so they are
    scale-free ("1.0" ~ one normal neighbour gap, anywhere in the scene).

DATA FLOW
    ref (trusted points) --build_grid--> grid g
    g + ref  --grid_knn(self_exclude)--> spacing field sp  (typical gap around each ref point)
    g + qry  --grid_knn-->               K nearest ref points per query (within radius h)
    neighbours --> reach, plane fit (PCA) --> normal --> height

SHAPE GLOSSARY
    N   number of reference points          Q   number of query points
    K   neighbours kept per query           M   max points read from ONE cell
    h   cell side length = search radius    27  the 3x3x3 block of cells around a query
"""

import numpy as np
import torch

# The 27 integer offsets (-1/0/+1 on each axis) that enumerate a cell and its 26 neighbours.
OFFS = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing='ij'), -1).view(-1, 3)  # [27,3]


def build_grid(pts, h):
    """
    Bucket points into cubic cells of side h and index them so that all points of one
    cell are stored contiguously. Think "pigeonholes for mail".

    pts: [N,3] float tensor.  Returns a dict (the "address book" of the grid).
    """
    mn = pts.min(0).values                                   # min corner of the scene = grid origin

    # Integer cell coordinate of every point: floor((x - origin) / h), per axis.
    # "+ 1" shifts all coords to start at 1, so the neighbour cell at offset -1 is still >= 0.
    ijk = ((pts - mn) / h).floor().long() + 1                # [N,3]

    # Grid extent per axis. "+ 2" leaves one empty layer on each side for the +/-1 neighbour lookup.
    dims = ijk.max(0).values + 2                             # [3]
    assert dims.prod().item() < 2**62, "grid too large; increase h or clip outliers"

    # Flatten (i,j,k) into ONE integer per point (row-major index). Same cell => same key.
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]   # [N]

    # Sort points by key. Afterwards, points of the same cell are adjacent in memory.
    #   key_s : sorted keys
    #   order : order[i] = original index of the i-th sorted point (to map back later)
    # stable=True keeps ties in original index order => deterministic results.
    key_s, order = torch.sort(key, stable=True)

    # Compress runs of equal keys into: the list of OCCUPIED cells (ukeys, sorted)
    # and how many points each holds (counts). Empty cells cost nothing.
    ukeys, counts = torch.unique_consecutive(key_s, return_counts=True)

    # starts[c] = position in the sorted array where cell c's points begin.
    # (cumulative count minus own count = number of points in all earlier cells)
    starts = torch.cumsum(counts, 0) - counts

    # pts_sorted[start : start+count] are exactly the points of that cell.
    return dict(mn=mn, h=h, dims=dims, order=order, ukeys=ukeys,
                starts=starts, counts=counts, pts_sorted=pts[order])


def pick_cell_size(pts, target=8, iters=4):
    """
    Choose h automatically so an occupied cell holds ~`target` points on average.
    For points lying on surfaces, points-per-cell grows like h^2, hence the sqrt update.
    NOTE: if you also thin the reference set, you should FIX h yourself and thin at h/3
    (see thin()), otherwise cells can overflow M.
    """
    h = (pts.max(0).values - pts.min(0).values).max() / 200   # crude starting guess
    for _ in range(iters):
        c = build_grid(pts, h)['counts'].float().mean()       # mean points per occupied cell
        h = h * (target / c).sqrt()                           # rescale toward the target
    return float(h)


@torch.no_grad()
def grid_knn(g, q, K=12, M=32, chunk=8192, self_exclude=False):
    """
    For each query point, find its K nearest grid points that lie within radius h.

    Why only 27 cells? Cell side == search radius h, so any point within distance h of the
    query differs by < 1 cell per axis => it is in the query's own cell or an adjacent one.

    q:            [Q,3] query points
    self_exclude: only valid when q is the SAME array as the grid's reference points, in the
                  SAME order (used for the spacing self-query). Skips the point itself.
    Returns: out_d [Q,K] distances (inf where fewer than K neighbours exist within h),
             out_i [Q,K] indices into the ORIGINAL reference array (-1 where none).
    """
    dev, Q = q.device, q.shape[0]
    offs, ar = OFFS.to(dev), torch.arange(M, device=dev)
    out_d = torch.full((Q, K), float('inf'), device=dev)
    out_i = torch.full((Q, K), -1, dtype=torch.long, device=dev)
    dims, nu = g['dims'], g['ukeys'].numel()

    # Process queries in chunks to bound memory: the big temporary tensors are [n,27,M(,3)].
    for s in range(0, Q, chunk):
        qc = q[s:s + chunk]; n = qc.shape[0]

        # 1) Which 27 cells to look in: the query's own cell coordinate + each offset.
        nb = ((qc - g['mn']) / g['h']).floor().long()[:, None, :] + 1 + offs    # [n,27,3]

        # 2) Cells outside the grid's bounds contain nothing (also covers queries outside the grid).
        inside = ((nb >= 0) & (nb < dims)).all(-1)                              # [n,27]
        nbc = torch.minimum(nb.clamp(min=0), dims - 1)    # clamp only so key math stays valid

        # 3) Same flattening as in build_grid -> one key per (query, neighbour cell).
        key = (nbc[..., 0] * dims[1] + nbc[..., 1]) * dims[2] + nbc[..., 2]     # [n,27]

        # 4) Look the keys up in the sorted list of occupied cells (binary search).
        #    It is a HIT only if the cell is inside the grid AND actually occupied.
        pos = torch.searchsorted(g['ukeys'], key).clamp(max=nu - 1)             # [n,27]
        hit = inside & (g['ukeys'][pos] == key)

        # 5) For each hit cell: where its points start and how many (capped at M).
        #    CAP: points beyond the first M of a cell are IGNORED. Thinning guarantees this
        #    never triggers; without it, dense cells silently lose neighbours.
        start = torch.where(hit, g['starts'][pos], torch.zeros_like(pos))
        cnt = torch.where(hit, g['counts'][pos], torch.zeros_like(pos)).clamp(max=M)

        # 6) Build candidate slots [n,27,M]: slot j of a cell is valid if j < count.
        #    Invalid slots read index 0 (a harmless dummy) and are masked out via `ok`.
        ok = ar < cnt[..., None]                                                # [n,27,M]
        idx = torch.where(ok, start[..., None] + ar, torch.zeros_like(pos)[..., None])

        # 7) Squared distance from each query to every candidate; remember original ids.
        d2 = ((g['pts_sorted'][idx] - qc[:, None, None, :]) ** 2).sum(-1)       # [n,27,M]
        orig = g['order'][idx]                                                  # [n,27,M]

        # 8) Keep only candidates inside the search radius (and, optionally, not the query itself).
        ok &= d2 <= g['h'] ** 2
        if self_exclude:
            ok &= orig != (s + torch.arange(n, device=dev))[:, None, None]

        # 9) Flatten to [n, 27*M] and take the K smallest. Masked entries are inf and lose.
        d2 = torch.where(ok, d2, torch.full_like(d2, float('inf'))).view(n, -1)
        v, ti = d2.topk(K, dim=1, largest=False)              # v: K smallest d2, ti: their slots
        i = orig.view(n, -1).gather(1, ti)                    # map slots -> original indices
        out_d[s:s + n] = v.sqrt()
        out_i[s:s + n] = torch.where(torch.isfinite(v), i, torch.full_like(i, -1))
    return out_d, out_i


@torch.no_grad()
def reach_height(ref, qry, K=12, Kmin=6, M=32, target=8, h=None):
    """
    ref: [N,3] trusted points (typically THINNED interior Gaussians)
    qry: [Q,3] points to score (e.g. class-1 Gaussians)
    Kmin: a query with fewer than Kmin trusted neighbours inside radius h is 'unsupported'
          (nothing trusted nearby -> reach/height are set to inf)

    Returns reach [Q], height [Q], unsupported [Q] bool, h (the cell size used).
    """
    h = pick_cell_size(ref, target) if h is None else h
    g = build_grid(ref, h)

    # --- A) Spacing field ---------------------------------------------------------------
    # For every reference point: mean distance to its K nearest OTHER reference points
    # (within h). This is "the typical gap between trusted Gaussians around here".
    d_ref, _ = grid_knn(g, ref, K=K, M=M, self_exclude=True)
    fin = torch.isfinite(d_ref)                                       # [N,K] valid neighbours
    sp = torch.where(fin, d_ref, torch.zeros_like(d_ref)).sum(1) / fin.sum(1).clamp(min=1)
    sp = torch.where(fin.any(1), sp, torch.full_like(sp, h))          # isolated ref point -> use h

    # --- B) Neighbours of each query ----------------------------------------------------
    d, j = grid_knn(g, qry, K=K, M=M)             # d: [Q,K] distances, j: [Q,K] ref indices (-1 = none)
    m = j >= 0                                    # [Q,K] which neighbour slots are real
    n = m.sum(1).clamp(min=1)                     # [Q]   number of real neighbours (>= 1 to avoid /0)
    mf = m.float()
    jj = j.clamp(min=0)                           # safe index for gathering; masked by mf afterwards

    # --- C) Local spacing at the query, then REACH --------------------------------------
    # Spacing at a query = average spacing of its own neighbours (uses the field from A).
    spacing = (sp[jj] * mf).sum(1) / n + 1e-8                                     # [Q]
    # REACH = mean distance to neighbours / local spacing.  ~1 => right next to trusted points.
    reach = (torch.where(m, d, torch.zeros_like(d)).sum(1) / n) / spacing         # [Q]

    # --- D) Plane fit by PCA over the neighbours' positions -----------------------------
    nbr = ref[jj]                                                  # [Q,K,3] neighbour positions
    c = (nbr * mf[..., None]).sum(1) / n[:, None]                  # [Q,3] centroid of real neighbours
    diff = (nbr - c[:, None]) * mf[..., None]                      # centred; masked slots become 0
    cov = torch.einsum('qki,qkj->qij', diff, diff) / n[:, None, None]   # [Q,3,3] covariance

    # Eigen-decomposition of the 3x3 covariances. Done on CPU in float64: batched eigh on the
    # GPU tried to allocate ~50 GiB. np.linalg.eigh returns eigenvalues in ASCENDING order, so
    # column 0 of the eigenvector matrix is the direction of LEAST spread = the surface NORMAL.
    cov64 = cov.double().cpu().numpy()                             # [Q,3,3]
    _, evec = np.linalg.eigh(cov64)                                # evec[..., :, 0] = smallest-eigenvalue vector
    normal = torch.from_numpy(evec[..., 0]).float().to(cov.device) # [Q,3]
    del cov64, evec

    # --- E) HEIGHT ------------------------------------------------------------------------
    # Distance from the query to the fitted plane = |(q - centroid) . normal|, in units of spacing.
    height = ((qry - c) * normal).sum(-1).abs() / spacing          # [Q]

    # --- F) Unsupported queries -----------------------------------------------------------
    unsupported = m.sum(1) < Kmin                  # too few trusted neighbours within radius h
    reach[unsupported] = float('inf'); height[unsupported] = float('inf')
    return reach, height, unsupported, h


def thin(pts, s):
    """
    Keep ONE point per small voxel of side s; returns the indices of the survivors.
    Rule: thin with s = h / 3  =>  a cell (h x h x h) contains at most 3^3 = 27 voxels
          => at most 27 points  <=  M = 32.
    """
    mn = pts.min(0).values
    ijk = ((pts - mn) / s).floor().long()                   # voxel coordinate of each point
    dims = ijk.max(0).values + 1
    key = (ijk[:, 0] * dims[1] + ijk[:, 1]) * dims[2] + ijk[:, 2]
    key_s, order = torch.sort(key, stable=True)             # deterministic
    first = torch.ones_like(key_s, dtype=torch.bool)
    first[1:] = key_s[1:] != key_s[:-1]                     # True at the first point of each voxel
    return order[first]                                     # original indices of the survivors


def p99(x):
    return float(np.percentile(x[torch.isfinite(x)].cpu().numpy(), 99))
