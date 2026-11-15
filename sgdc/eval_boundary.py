"""Boundary-restricted evaluation for SGDC.

Renders a trained model and reports PSNR globally, on a dilated semantic-boundary
band, and in segment interiors. This is the metric Component A actually targets:
if boundary-conditional densification helps, it should show up on the band even
when global PSNR is flat.

Usage:
    python -m sgdc.eval_boundary --model_path output/base -s <DATASET_DIR> --iteration 30000
    python -m sgdc.eval_boundary --model_path output/sgdc_a -s <DATASET_DIR> --iteration 30000
"""
from __future__ import annotations

import argparse
import math

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel

from sgdc import semantic_utils as su
from sgdc.semantic_store import SemanticStore
from sgdc.datasets import get_adapter


def _psnr_masked(pred: torch.Tensor, gt: torch.Tensor, mask=None) -> float:
    d = (pred - gt) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = d[:, mask]
    mse = d.mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _band(label: np.ndarray, px: int) -> np.ndarray:
    b = su.boundary_mask(label).astype(np.uint8)
    if px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        b = cv2.dilate(b, k)
    return b.astype(bool)


def evaluate(dataset, iteration, pipeline, band_px, use_test):
    # HLOC/COLMAP scenes here have no depth: force depth mode off.
    dataset.depths = ""
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    store = SemanticStore(get_adapter("replica", dataset.source_path), device="cuda")

    views = scene.getTestCameras() if use_test else scene.getTrainCameras()
    if len(views) == 0:
        print("no views for the requested split")
        return

    agg = {"global": [], "band": [], "interior": []}
    with torch.no_grad():
        for v in views:
            pkg = render(v, gaussians, pipeline, bg)
            img = pkg["render"].clamp(0.0, 1.0)
            gt = v.original_image[:3].clamp(0.0, 1.0)
            H, W = int(v.image_height), int(v.image_width)
            sem = store.get(v.image_name, H, W)
            band = _band(sem["label"].cpu().numpy(), band_px) if sem is not None else None

            agg["global"].append(_psnr_masked(img, gt, None))
            if band is not None:
                agg["band"].append(_psnr_masked(img, gt, torch.as_tensor(band, device=img.device)))
                agg["interior"].append(_psnr_masked(img, gt, torch.as_tensor(~band, device=img.device)))

    n_gauss = gaussians.get_xyz.shape[0]
    split = "test" if use_test else "train"
    print(f"model={dataset.model_path}  iter={scene.loaded_iter}  split={split}  views={len(views)}")
    print(f"  #gauss     = {n_gauss:,}")
    for k in ("global", "band", "interior"):
        vals = [x for x in agg[k] if x == x]
        if vals:
            print(f"  PSNR {k:8s}= {np.mean(vals):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SGDC boundary-restricted evaluation")
    model = ModelParams(ap, sentinel=True)
    pipe = PipelineParams(ap)
    ap.add_argument("--iteration", default=-1, type=int)
    ap.add_argument("--boundary_band", default=4, type=int, help="dilation half-width in px")
    ap.add_argument("--use_test", action="store_true")
    # Merge saved cfg_args (so depths="", white_background, etc. match training).
    args = get_combined_args(ap)

    evaluate(model.extract(args), args.iteration, pipe.extract(args),
             args.boundary_band, args.use_test)
