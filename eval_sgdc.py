"""SGDC on-the-fly photometric evaluation.

Renders the train/test views of a trained model and reports, per split:

* global **PSNR / SSIM / LPIPS** and the final **Gaussian count**, and
* **boundary-restricted** and **interior** PSNR (and LPIPS) over a dilated
  semantic-boundary band.

This is the metric code path used by ``experiments.metrics`` for *every*
method: SGDC models and upstream-graphdeco PLYs alike (a vanilla PLY simply
loads with no SGDC buffers). The RGB rasterization is stock 3DGS, so the
numbers are directly comparable across methods.

The band/interior split is dataset-driven: it is computed from the Replica
class labels under ``<source_path>/semantic_class`` through the same
``sgdc`` adapter the trainer uses. If the labels are absent (or the model was
trained on a scene without them) the band/interior metrics are omitted and
only the global metrics are reported.

    python eval_sgdc.py -m <model_path> --iteration <N> --json_out <file.json>
    python eval_sgdc.py -m <model_path> --boundary_band 4 --skip_train
"""

from __future__ import annotations

import json
import math
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import ssim

# Replica adapter + geometry (same code the trainer uses).
try:
    from sgdc import semantic_utils as su
    from sgdc.datasets import get_adapter
    from sgdc.semantic_store import SemanticStore
    SGDC_AVAILABLE = True
except Exception as _e:  # pragma: no cover - defensive
    SGDC_AVAILABLE = False
    _SGDC_IMPORT_ERROR = _e

# LPIPS net used to match the repo's canonical `metrics.py` (vgg, [0,1] inputs).
DEFAULT_LPIPS_NET = "vgg"


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _psnr_masked(pred: torch.Tensor, gt: torch.Tensor, mask=None) -> float:
    d = (pred - gt) ** 2
    if mask is not None:
        if mask.sum() == 0:
            return float("nan")
        d = d[:, mask]
    mse = d.mean().item()
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def _band_mask(label: np.ndarray, px: int) -> np.ndarray:
    b = su.boundary_mask(label).astype(np.uint8)
    if px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
        b = cv2.dilate(b, k)
    return b.astype(bool)


def _mean(xs) -> float | None:
    vals = [x for x in xs if x == x]  # drop NaN
    return float(np.mean(vals)) if vals else None


def _fill_model_defaults(dataset) -> None:
    """Give a vanilla (upstream graphdeco) cfg_args the fields this fork reads."""
    defaults = {
        "images": "images",
        "depths": "",
        "resolution": -1,
        "white_background": False,
        "train_test_exp": False,
        "eval": False,
        "data_device": "cuda",
    }
    for k, v in defaults.items():
        if not hasattr(dataset, k):
            setattr(dataset, k, v)


# --------------------------------------------------------------------------- #
# evaluation                                                                   #
# --------------------------------------------------------------------------- #
def evaluate_split(gaussians, scene, dataset, pipe, views, name, store,
                   boundary_band, lpips_criterion, bg):
    psnrs, ssims, lpipss = [], [], []
    psnrs_b, lpipss_b, psnrs_i, lpipss_i = [], [], [], []
    n_band = 0

    with torch.no_grad():
        for v in views:
            pkg = render(v, gaussians, pipe, bg)
            img = pkg["render"].clamp(0.0, 1.0)
            gt = v.original_image[:3].clamp(0.0, 1.0)

            if getattr(dataset, "train_test_exp", False):
                half = img.shape[-1] // 2
                img = img[..., half:]
                gt = gt[..., half:]

            psnrs.append(_psnr_masked(img, gt))
            ssims.append(ssim(img, gt).item())
            lpipss.append(lpips_criterion(img[None], gt[None]).item())

            if store is None:
                continue
            H, W = int(v.image_height), int(v.image_width)
            sem = store.get(v.image_name, H, W)
            if sem is None:
                continue
            band = _band_mask(sem["label"].cpu().numpy(), boundary_band)
            if not band.any():
                continue
            n_band += 1
            bt = torch.as_tensor(band, device=img.device)
            inv = ~bt
            psnrs_b.append(_psnr_masked(img, gt, bt))
            psnrs_i.append(_psnr_masked(img, gt, inv))
            # zero the complement in BOTH images so LPIPS sees only the region
            lpipss_b.append(lpips_criterion((img * bt)[None], (gt * bt)[None]).item())
            lpipss_i.append(lpips_criterion((img * inv)[None], (gt * inv)[None]).item())

    metrics = {
        "num_views": len(views),
        "psnr": _mean(psnrs),
        "ssim": _mean(ssims),
        "lpips": _mean(lpipss),
        "psnr_boundary": _mean(psnrs_b),
        "lpips_boundary": _mean(lpipss_b),
        "psnr_interior": _mean(psnrs_i),
        "lpips_interior": _mean(lpipss_i),
        "num_band_views": n_band,
    }

    def _f(x):
        return "n/a" if x is None else f"{x:.3f}"

    print(f"\n[{name}] {len(views)} views | PSNR {_f(metrics['psnr'])} | "
          f"SSIM {_f(metrics['ssim'])} | LPIPS {_f(metrics['lpips'])}")
    if n_band:
        print(f"  boundary: PSNR {_f(metrics['psnr_boundary'])} | LPIPS {_f(metrics['lpips_boundary'])} "
              f"| interior: PSNR {_f(metrics['psnr_interior'])} | LPIPS {_f(metrics['lpips_interior'])} "
              f"({n_band}/{len(views)} views with labels)")
    return metrics


def run(dataset, iteration, pipe, skip_train, skip_test, boundary_band,
        json_out=None, lpips_net=DEFAULT_LPIPS_NET):
    # HLOC/COLMAP scenes here have no depth: force depth mode off.
    dataset.depths = ""
    _fill_model_defaults(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")

    # Semantic labels (band/interior) — optional; a scene without semantic_class
    # simply reports global metrics.
    store = None
    if SGDC_AVAILABLE:
        try:
            adapter = get_adapter("replica", dataset.source_path)
            store = SemanticStore(adapter, device="cuda")
            print(f"[sgdc-eval] label adapter: {adapter.meta()}")
        except Exception as e:
            print(f"[sgdc-eval] no semantic labels ({e}); band/interior metrics disabled")
    else:
        print(f"[sgdc-eval] sgdc import failed ({_SGDC_IMPORT_ERROR}); "
              f"band/interior metrics disabled")

    splits = {}
    if not skip_train:
        splits["train"] = scene.getTrainCameras()
    if not skip_test:
        splits["test"] = scene.getTestCameras()

    results = {
        "run_dir": os.path.abspath(dataset.model_path),
        "iteration": int(scene.loaded_iter),
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "lpips_net": lpips_net,
        "splits": {},
    }

    for name, cams in splits.items():
        if not cams:
            print(f"[{name}] no cameras; skipped")
            continue
        # Rebuild the criterion per split so it is tagged with the chosen net.
        criterion = _make_lpips(lpips_net)
        results["splits"][name] = evaluate_split(
            gaussians, scene, dataset, pipe, cams, name, store, boundary_band,
            criterion, bg)

    print(f"[sgdc-eval] gaussians: {results['gaussians']}")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[sgdc-eval] wrote {json_out}")
    return results


def _make_lpips(net_type):
    """Instantiate the LPIPS module once (avoids a new VGG per call)."""
    from lpipsPyTorch.modules.lpips import LPIPS

    class _Criterion:
        def __init__(self, net):
            self.net = LPIPS(net, "0.1").to("cuda").eval()

        @torch.no_grad()
        def __call__(self, a, b):
            return self.net(a, b)

    return _Criterion(net_type)


if __name__ == "__main__":
    parser = ArgumentParser(description="SGDC photometric + boundary-restricted evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--boundary_band", default=4, type=int,
                        help="dilation half-width (px) of the boundary band")
    parser.add_argument("--json_out", default="", type=str, help="write metrics JSON here")
    parser.add_argument("--lpips_net", default=DEFAULT_LPIPS_NET, type=str,
                        choices=["alex", "vgg", "squeeze"])
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)
    safe_state(args.quiet)
    run(model.extract(args), args.iteration, pipeline.extract(args),
        args.skip_train, args.skip_test, args.boundary_band,
        args.json_out or None, args.lpips_net)
