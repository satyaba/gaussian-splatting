# Paired per-view comparison of two eval_sgdense.py JSON outputs (docs/08 7.9).
#
# PSNR over a whole image is nearly blind to class boundaries; this reports the
# per-view distribution instead of one aggregate, and includes the
# boundary-restricted metrics so "looks better on many views" becomes a number.
#
#   python eval_sgdense.py -m <model> --json_out a.json
#   python eval_sgdense.py -m <model> --json_out b.json
#   python scripts/compare_runs.py a.json b.json --split test --label-a vanilla --label-b E4
#
# Notes:
#   * region metrics need the GT segmentation -> run eval_sgdense with
#     --segmentation_path <dir> (and the right --num_segmentation_classes).
#   * semantic mIoU additionally needs the decoder checkpoint, so it appears for
#     SGDense runs only.

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense.evalmetrics import paired_summary  # noqa: E402

# (metric, higher_is_better)
METRICS = [
    ("psnr", True),
    ("ssim", True),
    ("lpips", False),
    ("psnr_boundary", True),
    ("lpips_boundary", False),
    ("psnr_interior", True),
    ("lpips_interior", False),
]


def _load(path, split):
    d = json.load(open(path))
    s = d.get("splits", {}).get(split)
    if not s or "per_view" not in s:
        raise SystemExit(f"[compare] no per_view for split='{split}' in {path}. "
                         f"Re-run eval_sgdense.py with the updated evaluator.")
    return d, s


def _fmt(x):
    return "  n/a" if x is None else f"{x:.4f}"


def compare(a_path, b_path, split="test", label_a="A", label_b="B", plot=None):
    da, sa = _load(a_path, split)
    db, sb = _load(b_path, split)
    pva = {v["name"]: v for v in sa["per_view"]}
    pvb = {v["name"]: v for v in sb["per_view"]}

    print(f"split={split}  {label_a}: {len(pva)} views / {da.get('gaussians')} Gaussians"
          f"  |  {label_b}: {len(pvb)} views / {db.get('gaussians')} Gaussians")
    for tag, s in ((label_a, sa), (label_b, sb)):
        print(f"  {tag}: PSNR {s.get('psnr'):.3f}  LPIPS {s.get('lpips'):.4f}  |  "
              f"boundary PSNR {_fmt(s.get('psnr_boundary'))} LPIPS {_fmt(s.get('lpips_boundary'))}  |  "
              f"interior PSNR {_fmt(s.get('psnr_interior'))}  |  mIoU {_fmt(s.get('semantic', {}).get('miou'))}")

    print()
    rows = []
    for metric, hb in METRICS:
        r = paired_summary(pva, pvb, metric, higher_better=hb)
        rows.append((metric, hb, r))
        if r.get("n"):
            print(f"  {metric:16s} {label_a}={r['mean_a']:.4f} {label_b}={r['mean_b']:.4f} "
                  f"delta={r['mean_delta']:+.4f} median={r['median_delta']:+.4f} "
                  f"{label_b}-better on {r['frac_b_better']*100:5.1f}% of {r['n']} views")

    if plot:
        _plot(pva, pvb, label_a, label_b, plot)
    return rows


def _plot(pva, pvb, la, lb, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in pva if n in pvb]
    panels = [("lpips", False), ("psnr", True), ("lpips_boundary", False), ("psnr_boundary", True)]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (metric, hb) in zip(axes, panels):
        a = np.array([pva[n].get(metric, np.nan) for n in names], float)
        b = np.array([pvb[n].get(metric, np.nan) for n in names], float)
        m = np.isfinite(a) & np.isfinite(b)
        d = b[m] - a[m]
        ax.hist(d, bins=21, color="#3b82f6", edgecolor="k", linewidth=0.3)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{metric} delta ({lb} - {la})")
        ax.set_xlabel("per-view delta" + ("  (+ = B better)" if hb else "  (- = B better)"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[compare] wrote {path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--plot", default="", help="if set, save per-view delta histograms here")
    args = ap.parse_args()
    compare(args.a, args.b, args.split, args.label_a, args.label_b, args.plot or None)


if __name__ == "__main__":
    main()
