"""Verify that SGDC labels align with the training images.

Overlays segment boundaries (red) on the RGB frames so a mismatch in naming,
resolution, or orientation is immediately visible.

Usage:
    python -m sgdc.verify_alignment --dataset_dir <DATASET_DIR> [--n 6] [--out DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from sgdc import semantic_utils as su
from sgdc.datasets import get_adapter


def _overlay(rgb: np.ndarray, label: np.ndarray) -> np.ndarray:
    b = su.boundary_mask(label)
    out = rgb.copy()
    out[b] = [255, 0, 0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="SGDC label/image alignment check")
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--adapter", default="replica")
    ap.add_argument("--images_subdir", default="images")
    ap.add_argument("--out", default="sgdc_align_check")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    ds = Path(args.dataset_dir)
    adapter = get_adapter(args.adapter, ds, {})
    print("adapter meta:", adapter.meta())

    img_dir = ds / args.images_subdir
    imgs = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not imgs:
        print(f"no images under {img_dir}")
        return

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(imgs) // args.n)

    ok = 0
    for p in imgs[::step][: args.n]:
        lab = adapter.load(p.name)
        if lab is None:
            print(f"  [skip] no label for {p.name}")
            continue
        rgb = np.array(Image.open(p).convert("RGB"))
        H, W = rgb.shape[:2]
        lab_r = su.resize_ids(lab, H, W)
        outp = outdir / f"check_{p.stem}.png"
        Image.fromarray(_overlay(rgb, lab_r)).save(outp)
        print(f"  {p.name}: label{lab.shape} -> image{(H, W)}  ->  {outp}")
        ok += 1

    print(f"\nwrote {ok} overlay(s) to {outdir}")


if __name__ == "__main__":
    main()
