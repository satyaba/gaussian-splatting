#!/usr/bin/env python3
"""Audit the class ids present in a segmentation folder (SGDense data prep).

Scans uint8 class-id PNGs (255 = void) under a segmentation folder and reports
the class histogram + the suggested ``num_segmentation_classes`` (= max id + 1,
NOT the distinct count). Run before training to size the taxonomy.

    python scripts/audit_seg_classes.py --segmentation_path <dir>

Mirrors the loader rule in scene/dataset_readers._segmentation_path_for:
<dir>/<split>/<stem>.png preferred, <dir>/<stem>.png fallback. vis_* renders are
excluded (they are colour images that would poison the histogram).
"""
import argparse
import glob
import os

import cv2
import numpy as np


def audit(seg_dir, expected_num_classes=None, void_id=255, num_bins=256):
    paths = sorted({*glob.glob(os.path.join(seg_dir, "train", "*.png")),
                    *glob.glob(os.path.join(seg_dir, "test", "*.png")),
                    *glob.glob(os.path.join(seg_dir, "*.png"))})
    paths = [p for p in paths if "vis_" not in os.path.basename(p)]
    if not paths:
        raise SystemExit(f"no .png under {seg_dir}")

    counts = np.zeros(num_bins, dtype=np.int64)
    void_fracs, dtypes = [], set()
    for p in paths:
        raw = cv2.imread(p, -1)
        assert raw is not None, p
        if raw.ndim == 3:
            raw = raw[..., 0]
        dtypes.add(str(raw.dtype))
        h = np.bincount(raw.ravel(), minlength=num_bins)
        counts += h
        void_fracs.append(h[void_id] / h.sum())

    valid = counts.copy()
    valid[void_id] = 0
    present = np.nonzero(valid)[0]
    total = counts.sum()

    print(f"\nfolder          : {seg_dir}")
    print(f"files           : {len(paths)}  (dtypes seen: {', '.join(sorted(dtypes))})")
    print(f"distinct classes: {len(present)}")
    print(f"max class id    : {present.max()}")
    print(f"void ({void_id})      : {counts[void_id] / total:.2%} of all pixels "
          f"(per-frame min/max: {min(void_fracs):.2%} / {max(void_fracs):.2%})")
    print(f"\nid  pixels        share")
    for cid in present:
        print(f"{cid:3d} {valid[cid]:>13,d}  {valid[cid] / valid.sum():6.2%}")

    suggested = int(present.max()) + 1
    print(f"\nsuggested num_segmentation_classes: {suggested}  "
          f"(= max_id + 1, NOT the distinct count {len(present)})")

    if expected_num_classes is not None and present.max() >= expected_num_classes:
        raise SystemExit(
            f"taxonomy mismatch: max id {present.max()} >= "
            f"num_segmentation_classes={expected_num_classes} "
            f"(the loader would assert at load time)")

    rare = present[valid[present] < 0.0001 * valid.sum()]
    if len(rare):
        print(f"rare classes (<0.01% of labeled px): {rare.tolist()}")
    return suggested


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Audit segmentation class ids")
    ap.add_argument("--segmentation_path", required=True, type=str,
                    help="folder of uint8 class-id PNGs (255 = void)")
    ap.add_argument("--num_segmentation_classes", default=None, type=int,
                    help="if set, assert the discovered max id fits")
    args = ap.parse_args()
    audit(args.segmentation_path, args.num_segmentation_classes)
