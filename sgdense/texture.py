"""Per-class texture-frequency solver (SGDense TEXTURE_BASED_THRESH).

Estimates a per-class texture score ``s_tilde`` in [0,1] from the squared
Laplacian energy of the (illumination-normalised) images inside eroded class
masks, with shrinkage + percentile ranking. ``class_freq_thres`` then turns
``s_tilde`` into a per-class densification gradient threshold.
"""

import glob
import os
import re

import cv2
import numpy as np


def last_int(path):
    """Last integer in the file stem, used to pair rgb <-> seg files."""
    nums = re.findall(r"\d+", os.path.splitext(os.path.basename(path))[0])
    return int(nums[-1]) if nums else None


def pair_files(rgb_files, seg_files):
    rgb_map = {last_int(f): f for f in rgb_files if last_int(f) is not None}
    seg_map = {last_int(f): f for f in seg_files if last_int(f) is not None}
    ids = sorted(set(rgb_map) & set(seg_map))
    skipped = (len(rgb_map) - len(ids), len(seg_map) - len(ids))
    if skipped != (0, 0):
        print(f"[warn] unmatched files skipped: rgb={skipped[0]}, seg={skipped[1]}")
    if not ids:
        raise SystemExit("No rgb/seg pairs matched by numeric id in the filename.")
    return [(i, rgb_map[i], seg_map[i]) for i in ids]


def load_seg(path, hw):
    seg = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise IOError(f"cannot read {path}")
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = seg.astype(np.int32)
    if seg.shape != hw:  # labels must never be interpolated
        seg = cv2.resize(seg, (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST)
    return seg


def laplacian_energy(rgb_bgr, illum_norm=True, blur_sigma=1.0, illum_sigma=10.0):
    """Per-pixel squared Laplacian response of the (normalised) grayscale image."""
    g = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    if illum_norm:  # remove slow brightness variation so bright != detailed
        local_mean = cv2.GaussianBlur(g, (0, 0), illum_sigma)
        g = g / (local_mean + 0.05)
    if blur_sigma > 0:  # suppress sensor / compression noise
        g = cv2.GaussianBlur(g, (0, 0), blur_sigma)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    return lap * lap


def percentile_rank(x):
    """Rank of each value scaled to [0, 1] (smallest = 0, largest = 1)."""
    n = len(x)
    if n == 1:
        return np.array([0.5], dtype=np.float32)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(x, kind="stable")] = np.arange(n)
    return (ranks / (n - 1)).astype(np.float32)


def compute_texture_freq(scene_dir, seg_dir, num_classes, ignore=(), erode=3, min_px=500,
                         min_views=3, k=None, step=1, no_illum_norm=True, blur_sigma=1.0,
                         verbose=True):
    """Per-class texture score s_tilde (percentile-ranked, shrinkage-smoothed).

    scene_dir: folder of RGB PNGs. seg_dir: folder of uint8 class-id PNGs.
    Returns (s_tilde [C] float32, k, prior, pair_count).
    """
    pairs = pair_files(sorted(glob.glob(os.path.join(str(scene_dir), "*.png"))),
                       sorted(glob.glob(os.path.join(str(seg_dir), "*.png"))))
    pairs = pairs[::step]

    C = int(num_classes)
    ksz = 2 * erode + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)) if erode > 0 else None

    sum_e = np.zeros(C, dtype=np.float64)   # sum of squared Laplacian over eroded masks
    cnt = np.zeros(C, dtype=np.int64)       # eroded pixel count
    views = np.zeros(C, dtype=np.int32)
    ignore = set(ignore)

    for n, (vid, rgb_path, seg_path) in enumerate(pairs):
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] cannot read {rgb_path}, skipping")
            continue
        seg = load_seg(seg_path, img.shape[:2])
        energy = laplacian_energy(img, not no_illum_norm, blur_sigma)

        for c in np.unique(seg):
            if c < 0 or c >= C or c in ignore:
                continue
            m = (seg == c).astype(np.uint8)
            if kernel is not None:
                m = cv2.erode(m, kernel)
            kk = int(m.sum())
            if kk < min_px:
                continue
            sel = m.astype(bool)
            sum_e[c] += float(energy[sel].sum())
            cnt[c] += kk
            views[c] += 1
        if verbose and (n + 1) % 20 == 0:
            print(f"  processed {n + 1}/{len(pairs)}")

    valid = (views >= min_views) & (cnt > 0)
    if valid.sum() == 0:
        raise SystemExit("No class passed the min-views / min-px filters.")

    raw = np.full(C, np.nan)
    raw[valid] = sum_e[valid] / cnt[valid]
    log_raw = np.full(C, np.nan)
    log_raw[valid] = np.log(raw[valid] + 1e-12)

    prior = np.median(log_raw[valid])
    kk = k if k is not None else float(np.percentile(cnt[valid], 25))
    w = np.zeros(C)
    w[valid] = cnt[valid] / (cnt[valid] + kk)
    log_shrunk = np.full(C, np.nan)
    log_shrunk[valid] = w[valid] * log_raw[valid] + (1 - w[valid]) * prior

    s_tilde = np.full(C, 0.5, dtype=np.float32)  # unseen / ignored -> neutral
    s_tilde[valid] = percentile_rank(log_shrunk[valid])

    return s_tilde, kk, float(prior), len(pairs)


def class_freq_thres(old_thresh, s_tilde, beta):
    """Per-class gradient threshold from the texture score.

    High texture (s_tilde -> 1) lowers the threshold; low texture raises it.
    """
    return np.exp(-beta * (s_tilde - 0.5)) * old_thresh
