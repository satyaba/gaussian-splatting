#
# SGDense on-the-fly photometric + semantic evaluation (run26 eval cell).
#
# Renders train/test views, reports PSNR / SSIM / LPIPS and the semantic
# pixel-accuracy / mIoU / fwIoU, and (optionally) saves a sample grid of
# GT RGB | rendered RGB | GT segmentation | rendered segmentation.
#
#   python eval_sgdense.py -m <model_path> --iteration <N> [--skip_train] [--out_dir <dir>]
#
# Requires the decoder checkpoint written by train.py (<model_path>/decoder<N>.pth).
import os
import sys
import json
from argparse import ArgumentParser

import numpy as np
import torch

from scene import Scene
from scene.gaussian_model import GaussianModel
from scene.segmentation_decoder import SegmentationDecoder, load_decoder_checkpoint
from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from sgdense.evalmetrics import seg_boundary_mask, region_psnr

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False

IGNORE_ID = -1   # void rule v1: camera_utils maps the 255 void label to -1 in memory


def _to_img(t):
    """[3,H,W] float tensor -> [H,W,3] float numpy in [0,1]."""
    return t.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _colorize(ids, palette, num_classes):
    """[H,W] long tensor -> [H,W,3] uint8; ignore/out-of-range pixels are black."""
    ids = ids.cpu().numpy()
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    valid = (ids != IGNORE_ID) & (ids >= 0) & (ids < num_classes)
    out[valid] = palette[ids[valid]]
    return out


def _plot_samples(samples, name, palette, num_classes, out_dir=None, col_width=4.0):
    import matplotlib
    if out_dir is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = ["GT RGB", "Rendered RGB", "GT Segmentation", "Rendered Segmentation"]
    n = len(samples)
    if n == 0:
        return
    _, H, W = samples[0]["gt_rgb"].shape
    row_height = col_width * H / W
    fig, axes = plt.subplots(n, 4, figsize=(col_width * 4, row_height * n + 0.5),
                             squeeze=False, gridspec_kw=dict(wspace=0.01, hspace=0.01), dpi=200)
    for i, s in enumerate(samples):
        imgs = [_to_img(s["gt_rgb"]), _to_img(s["pred_rgb"]),
                _colorize(s["gt_seg"], palette, num_classes),
                _colorize(s["pred_seg"], palette, num_classes)]
        for j, im in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(im, aspect="auto")
            ax.axis("off")
            if i == 0:
                ax.set_title(cols[j], fontsize=11, pad=4)
        axes[i, 0].text(0.02, 0.97, f"{s['name']} | PSNR {s['psnr']:.2f}",
                        transform=axes[i, 0].transAxes, ha="left", va="top",
                        fontsize=8, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=2, edgecolor="none"))
    fig.suptitle(f"[{name}] sample views", fontsize=13, y=1.0)
    fig.tight_layout()
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"samples_{name}.png")
        fig.savefig(path)
        print(f"  wrote {path}")
    else:
        plt.show()
    plt.close(fig)


def evaluate(gaussians, scene, dataset, pipe, decoder, cams, name, lp, ssim_fn, palette,
             num_samples, boundary_dilate=3):
    bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    num_classes = dataset.num_segmentation_classes

    psnrs, ssims, lpipss = [], [], []
    psnrs_b, lpipss_b, psnrs_i, lpipss_i = [], [], [], []
    conf = torch.zeros(num_classes, num_classes, dtype=torch.long, device="cuda")
    per_view, samples = [], []
    sample_idx = set(np.linspace(0, len(cams) - 1, min(num_samples, len(cams)), dtype=int).tolist())

    with torch.no_grad():
        for i, v in enumerate(cams):
            r = render(v, gaussians, pipe, bg)["render"].clamp(0, 1)[None]
            g = v.original_image[0:3].to("cuda")[None]
            mse = ((r - g) ** 2).mean().item()
            p = -10 * np.log10(max(mse, 1e-12))
            s = ssim_fn(r, g, data_range=1.0).item()
            l = lp(r * 2 - 1, g * 2 - 1).item()
            psnrs.append(p)
            ssims.append(s)
            lpipss.append(l)

            gt_seg = getattr(v, "gt_segmentation", None)

            # Region-restricted photometric/perceptual metrics from the GT class map
            # (docs/08 section 7.10): boundary band vs interior. No decoder needed.
            psnr_b = lpips_b = psnr_i = lpips_i = float("nan")
            frac_b = 0.0
            if num_classes > 0 and gt_seg is not None:
                gt_np = gt_seg.detach().cpu().numpy().squeeze()
                band = seg_boundary_mask(gt_np, num_classes, IGNORE_ID, boundary_dilate)
                frac_b = float(band.mean())
                if band.any():
                    r_np, g_np = _to_img(r[0]), _to_img(g[0])
                    psnr_b = region_psnr(r_np, g_np, band)
                    psnr_i = region_psnr(r_np, g_np, ~band)
                    mt = torch.from_numpy(band).to("cuda")[None, None]   # [1,1,H,W]
                    # zero the complement in BOTH images so LPIPS sees only the band
                    lpips_b = lp((r * mt) * 2 - 1, (g * mt) * 2 - 1).item()
                    lpips_i = lp((r * (1 - mt)) * 2 - 1, (g * (1 - mt)) * 2 - 1).item()
                    psnrs_b.append(psnr_b); lpipss_b.append(lpips_b)
                    psnrs_i.append(psnr_i); lpipss_i.append(lpips_i)

            per_view.append(dict(
                name=v.image_name, psnr=p, ssim=s, lpips=l,
                psnr_boundary=psnr_b, lpips_boundary=lpips_b,
                psnr_interior=psnr_i, lpips_interior=lpips_i,
                boundary_frac=frac_b))

            pred = gt = None
            if decoder is not None and gt_seg is not None:
                pkg = render(v, gaussians, pipe, bg)
                seg_feat = pkg["rendered_seg"]                       # [C_ENC, H, W]
                logits = decoder(seg_feat[None].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                pred = logits.argmax(1)[0]
                gt = gt_seg.to("cuda").long().squeeze()
                m = (gt != IGNORE_ID) & (gt < num_classes)
                conf += torch.bincount(gt[m] * num_classes + pred[m].clamp(0, num_classes - 1),
                                       minlength=num_classes ** 2).view(num_classes, num_classes)

            if i in sample_idx:
                samples.append(dict(
                    name=v.image_name, psnr=p,
                    gt_rgb=g[0].cpu(), pred_rgb=r[0].cpu(),
                    gt_seg=gt.cpu() if gt is not None else torch.full_like(r[0, 0], IGNORE_ID, dtype=torch.long),
                    pred_seg=pred.cpu() if pred is not None else torch.full_like(r[0, 0], IGNORE_ID, dtype=torch.long),
                ))

    def _mean(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else None

    print(f"\n[{name}] {len(cams)} views | PSNR {np.mean(psnrs):.2f} | "
          f"SSIM {np.mean(ssims):.4f} | LPIPS {np.mean(lpipss):.4f}")
    print("  worst 3 by PSNR:", [(n, round(p, 2)) for n, p in sorted([(d['name'], d['psnr']) for d in per_view], key=lambda x: x[1])[:3]])
    if psnrs_b:
        print(f"  boundary-restricted: PSNR {_mean(psnrs_b):.2f} | LPIPS {_mean(lpipss_b):.4f} "
              f"| interior: PSNR {_mean(psnrs_i):.2f} | LPIPS {_mean(lpipss_i):.4f}")

    metrics = {
        "num_views": len(cams),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpipss)),
        "psnr_boundary": _mean(psnrs_b),
        "lpips_boundary": _mean(lpipss_b),
        "psnr_interior": _mean(psnrs_i),
        "lpips_interior": _mean(lpipss_i),
        "per_view": per_view,
    }

    if decoder is not None and conf.sum() > 0:
        c = conf.double().cpu().numpy()
        tp, gt_cnt, pr_cnt = np.diag(c), c.sum(1), c.sum(0)
        union = gt_cnt + pr_cnt - tp
        present = gt_cnt > 0
        iou = tp / np.maximum(union, 1)
        freq = gt_cnt / gt_cnt.sum()
        print(f"  semantic: pixel acc {tp.sum()/c.sum():.4f} | mIoU {iou[present].mean():.4f} | "
              f"fwIoU {(freq[present]*iou[present]).sum():.4f}")
        worst = np.argsort(np.where(present, iou, np.inf))[:5]
        print("  lowest-IoU classes:", [(int(k), round(float(iou[k]), 3)) for k in worst])
        metrics["semantic"] = {
            "pixel_acc": float(tp.sum() / c.sum()),
            "miou": float(iou[present].mean()),
            "fwiou": float((freq[present] * iou[present]).sum()),
        }
    return samples, metrics


def run(dataset, iteration, pipe, skip_train, skip_test, num_samples, out_dir, json_out=None,
        boundary_dilate=3):
    # A vanilla (upstream graphdeco) cfg_args carries no segmentation fields;
    # default them so this fork's loader/renderer can score vanilla PLYs too.
    for k, v in (("seg_encoding_dim", 32), ("num_segmentation_classes", 0),
                 ("segmentation_path", "")):
        if not hasattr(dataset, k):
            setattr(dataset, k, v)

    gaussians = GaussianModel(dataset.sh_degree, seg_encoding_dim=dataset.seg_encoding_dim,
                              num_segmentation_classes=dataset.num_segmentation_classes)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    decoder = None
    decoder_path = os.path.join(dataset.model_path, f"decoder{scene.loaded_iter}.pth")
    if os.path.exists(decoder_path):
        decoder = SegmentationDecoder(dataset.seg_encoding_dim, dataset.num_segmentation_classes).cuda()
        dummy_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
        load_decoder_checkpoint(decoder, dummy_opt, decoder_path)
        decoder.eval()
        print(f"Loaded decoder: {decoder_path}")
    else:
        print(f"[warn] no decoder checkpoint at {decoder_path}; semantic metrics disabled")

    import lpips
    from torchmetrics.functional import structural_similarity_index_measure as ssim_fn
    lp = lpips.LPIPS(net="vgg").to("cuda").eval()

    rng = np.random.RandomState(0)
    palette = rng.randint(40, 255, size=(dataset.num_segmentation_classes, 3)).astype(np.uint8)

    splits = {}
    if not skip_train:
        splits["train"] = scene.getTrainCameras()
    if not skip_test:
        splits["test"] = scene.getTestCameras()

    results = {
        "run_dir": os.path.abspath(dataset.model_path),
        "iteration": int(scene.loaded_iter),
        "gaussians": int(gaussians.get_xyz.shape[0]),
        "splits": {},
    }

    for name, cams in splits.items():
        if not cams:
            continue
        samples, metrics = evaluate(gaussians, scene, dataset, pipe, decoder, cams, name,
                                    lp, ssim_fn, palette, num_samples, boundary_dilate)
        results["splits"][name] = metrics
        if decoder is not None:
            _plot_samples(samples, name, palette, dataset.num_segmentation_classes, out_dir=out_dir)

    print(f"[sgdense-eval] gaussians: {results['gaussians']}")
    if json_out:
        with open(json_out, "w") as f:
            json.dump(results, f, indent=1)
        print(f"[sgdense-eval] wrote {json_out}")
    return results


if __name__ == "__main__":
    parser = ArgumentParser(description="SGDense photometric + semantic evaluation")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--num_samples", default=4, type=int)
    parser.add_argument("--out_dir", default="", type=str, help="if set, save sample grids here")
    parser.add_argument("--json_out", default="", type=str, help="if set, write metrics JSON here")
    parser.add_argument("--boundary_dilate", default=3, type=int,
                        help="dilation (px) of the GT class-boundary band for region metrics")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Evaluating " + args.model_path)
    safe_state(args.quiet)
    run(model.extract(args), args.iteration, pipeline.extract(args),
        args.skip_train, args.skip_test, args.num_samples, args.out_dir or None,
        args.json_out or None, args.boundary_dilate)
