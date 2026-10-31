#!/usr/bin/env python3
"""Compute the per-class texture score s_tilde (SGDense TEXTURE_BASED_THRESH).

Runs the Laplacian-energy texture solver over paired RGB / class-id images and
writes ``s_tilde.npy`` + ``s_tilde.json`` into the model directory so train.py
can consume them (it also computes+caches them on the fly if absent).

    python scripts/compute_texture_freq.py \
        --scene_dir <images> --segmentation_path <seg_dir> \
        --num_segmentation_classes <C> --out_dir <model_path>

s_tilde is a per-class score in [0,1] (percentile-ranked, shrinkage-smoothed);
high = textured (lower the densification gradient threshold), low = flat.
"""
import argparse
import json
import os

import numpy as np

from sgdense.texture import compute_texture_freq


def main():
    ap = argparse.ArgumentParser(description="Compute per-class texture scores")
    ap.add_argument("--scene_dir", required=True, type=str, help="folder of RGB PNGs")
    ap.add_argument("--segmentation_path", required=True, type=str, help="folder of class-id PNGs")
    ap.add_argument("--num_segmentation_classes", required=True, type=int)
    ap.add_argument("--out_dir", required=True, type=str, help="model dir (s_tilde written here)")
    ap.add_argument("--erode", default=3, type=int)
    ap.add_argument("--min_px", default=500, type=int)
    ap.add_argument("--min_views", default=3, type=int)
    ap.add_argument("--k", default=-1, type=int, help="<=0 = auto (25th pct of pixel counts)")
    ap.add_argument("--step", default=1, type=int)
    ap.add_argument("--no_illum_norm", action="store_true")
    ap.add_argument("--blur_sigma", default=1.0, type=float)
    args = ap.parse_args()

    s_tilde, k, prior, pair_count = compute_texture_freq(
        args.scene_dir, args.segmentation_path, args.num_segmentation_classes,
        erode=args.erode, min_px=args.min_px, min_views=args.min_views,
        k=None if args.k <= 0 else args.k, step=args.step,
        no_illum_norm=args.no_illum_norm, blur_sigma=args.blur_sigma)

    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "s_tilde.npy"), s_tilde)
    with open(os.path.join(args.out_dir, "s_tilde.json"), "w") as f:
        json.dump({"s_tilde": s_tilde.tolist(), "shrinkage_k_px": k,
                   "prior_log_median": prior, "erode_px": args.erode,
                   "num_views": pair_count}, f, indent=1)
    print(f"wrote s_tilde.npy + s_tilde.json to {args.out_dir}  ({pair_count} pairs)")


if __name__ == "__main__":
    main()
