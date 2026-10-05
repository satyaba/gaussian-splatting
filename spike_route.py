#
# THE SPIKE — doc 04 §7.5 (item 5) + §7.6 (threshold calibration).
#
# The decisive test for T1 (segmentation-gated densification routing). For every
# Gaussian of a trained model we record its segmentation belief read-out
# (disagree, structure) and INDEPENDENTLY label it a ground-truth regime by
# projecting its centre into the training views and sampling the GT segmentation:
#
#   floater   <- projected pixel is void (class id -1)
#   boundary  <- projected pixel is valid and sits on a GT class edge
#   interior  <- projected pixel is valid and sits inside a uniform GT region
#
# Majority vote over the sampled views decides each Gaussian's GT label.
#
#   PASS = the three labels occupy distinct regions of the (disagree, structure)
#          plane (boundary high-structure, interior low-disagree, floater
#          low-structure). Then, and only then, may seg_route_enabled be turned on.
#   FAIL = they overlap -> (disagree, structure) is not a usable disambiguator;
#          T1 stays disabled and the design needs rework.
#
# The decisive evidence is the scatter PNG; the printed metrics are supporting.
# This script does NOT enable routing and does NOT modify the model.
#
# Requires: a trained model dir containing a saved iteration
#   output/<name>/point_cloud/iteration_<N>/point_cloud.ply   (xyz + seg encoding)
#   output/<name>/seg_hist<N>.npz                             (persisted M)
# produced by e.g.:  python train.py -s datasets/office_0 -m output/office_0 \
#                       --segmentation_path <gt_seg_dir> --save_iterations 12000
#
# Run (on the GPU host):
#   python spike_route.py -s datasets/office_0 -m output/office_0 \
#       --segmentation_path <gt_seg_dir> --load_iteration 12000 \
#       --out spike_12000.png --dump spike_12000.npz
#
# Logic self-test (no torch / no GPU needed):
#   python spike_route.py --selftest
#
import argparse
import csv
import json
import os
import sys

import numpy as np

REGIMES = ("interior", "boundary", "floater")
COLOURS = {"interior": "#2ca02c", "boundary": "#1f77b4", "floater": "#d62728",
           "unobserved": "#7f7f7f"}


# --------------------------------------------------------------------------- #
# Pure-numpy core (no torch) — unit-testable via --selftest                   #
# --------------------------------------------------------------------------- #
def gt_edge_map(seg):
    """seg: [H, W] integer label map (void = -1). Returns [H, W] bool: True on
    every pixel that has a 4-neighbour with a different label. Void is treated
    as its own label, so an object contour against unlabelled background also
    reads as an edge."""
    edge = np.zeros(seg.shape, dtype=bool)
    dh = seg[:, :-1] != seg[:, 1:]          # horizontal neighbour changes
    edge[:, :-1] |= dh
    edge[:, 1:] |= dh
    dv = seg[:-1, :] != seg[1:, :]          # vertical neighbour changes
    edge[:-1, :] |= dv
    edge[1:, :] |= dv
    return edge


def classify_pixels(val, edge):
    """val: [n] int class ids (-1 = void); edge: [n] bool. Returns the three
    disjoint per-pixel regime votes as bool arrays."""
    floater = val < 0
    boundary = (~floater) & edge
    interior = (~floater) & (~edge)
    return interior, boundary, floater


def project_centers(proj, xyz, width, height, w_eps=1e-8):
    """Pinhole-project Gaussian centres to pixels.
    proj: [4, 4] full_proj_transform; xyz: [N, 3].
    Returns (ui, vi, inside) — pixel indices (clamped) and a visibility mask.
    Visibility = in front of the camera (w > 0), projected inside the image, and
    within the NDC depth slab. No occlusion test (multi-view voting absorbs it)."""
    n = xyz.shape[0]
    pts = np.concatenate([xyz, np.ones((n, 1), dtype=np.float64)], axis=1)
    p = pts @ proj.T                                  # [N, 4]
    w = p[:, 3]
    safe_w = np.where(np.abs(w) < w_eps, np.nan, w)
    ndc = p[:, :3] / safe_w[:, None]
    u = ((ndc[:, 0] + 1.0) * width - 1.0) / 2.0
    v = ((ndc[:, 1] + 1.0) * height - 1.0) / 2.0
    inside = (np.isfinite(u) & np.isfinite(v) & (w > 0)
              & (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1)
              & (np.abs(ndc[:, 2]) <= 1.0))
    ui = np.clip(np.nan_to_num(u), 0, width - 1).round().astype(np.int64)
    vi = np.clip(np.nan_to_num(v), 0, height - 1).round().astype(np.int64)
    return ui, vi, inside


def accumulate_labels(cams, xyz, count):
    """Vote GT regime for every Gaussian across cameras.
    cams: iterable of (proj[4,4], seg[H,W]); count: [N, 3] int accumulator,
    columns = (interior, boundary, floater). Mutates `count` in place."""
    n = xyz.shape[0]
    for proj, seg in cams:
        h, w = seg.shape
        edge = gt_edge_map(seg)
        ui, vi, inside = project_centers(proj, xyz, w, h)
        idx = np.nonzero(inside)[0]
        if idx.size == 0:
            continue
        val = seg[vi[idx], ui[idx]]
        e = edge[vi[idx], ui[idx]]
        i_m, b_m, f_m = classify_pixels(val, e)
        count[:, 0] += np.bincount(idx[i_m], minlength=n)
        count[:, 1] += np.bincount(idx[b_m], minlength=n)
        count[:, 2] += np.bincount(idx[f_m], minlength=n)
    return count


def gt_label_from_counts(count):
    """[N, 3] votes -> [N] string labels; rows with no votes are 'unobserved'."""
    observed = count.sum(axis=1) > 0
    labels = np.full(count.shape[0], "unobserved", dtype=object)
    labels[observed] = np.asarray(REGIMES, dtype=object)[count[observed].argmax(axis=1)]
    return labels


def predict_regime(disagree, structure, tau_d, tau_s):
    """The doc 04 §3 rule, vectorised. Returns [N] string labels."""
    out = np.where(disagree < tau_d, "interior",
                   np.where(structure >= tau_s, "boundary", "floater"))
    return out.astype(object)


def nearest_centroid_accuracy(disagree, structure, gt, mask):
    """1-NN-to-class-centroid agreement as a coarse separability score."""
    classes = [c for c in REGIMES if np.any(mask & (gt == c))]
    if len(classes) < 2:
        return float("nan"), {}
    centroids = {c: (disagree[mask & (gt == c)].mean(), structure[mask & (gt == c)].mean())
                 for c in classes}
    d = np.stack([disagree[mask], structure[mask]], axis=1)
    cc = np.stack([centroids[c] for c in classes], axis=0)
    pred = np.asarray(classes, dtype=object)[np.argmin(
        ((d[:, None, :] - cc[None, :, :]) ** 2).sum(axis=2), axis=1)]
    acc = float((pred == gt[mask]).mean())
    return acc, centroids


def calibrate(disagree, structure, gt, act, grid=None):
    """§7.6: grid-search (q_disagree, q_structure) maximising agreement of the
    §3 rule with the GT labels, over the evidence-gated population `act`.
    Returns (best_qd, best_qs, best_acc, table) — table: list of (qd, qs, acc)."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    use = act & np.isin(gt, REGIMES)
    if use.sum() == 0:
        return None, None, float("nan"), []
    d_act, s_act = disagree[act], structure[act]
    table, best = [], (None, None, -1.0)
    for qd in grid:
        tau_d = np.quantile(d_act, qd)
        for qs in grid:
            tau_s = np.quantile(s_act, qs)
            pred = predict_regime(disagree, structure, tau_d, tau_s)
            acc = float((pred[use] == gt[use]).mean())
            table.append((float(qd), float(qs), acc))
            if acc > best[2]:
                best = (float(qd), float(qs), acc)
    return best[0], best[1], best[2], table


# --------------------------------------------------------------------------- #
# Self-test (no torch)                                                        #
# --------------------------------------------------------------------------- #
def _selftest():
    ok = True

    # gt_edge_map: an inner 3x3 block inside a uniform ring; the block centre is
    # interior (no neighbour differs), its corner pixel is an edge.
    seg = np.array([[1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]])
    e = gt_edge_map(seg)
    assert not e[2, 2], "block centre must be interior"
    assert e[1, 1] and e[1, 2], "block edge must be detected"
    assert not e[0, 0], "outer-ring centre is interior"

    # classify_pixels: void -> floater; valid+edge -> boundary; valid -> interior
    val = np.array([-1, 5, 5])
    edge = np.array([True, True, False])
    i_m, b_m, f_m = classify_pixels(val, edge)
    assert list(f_m) == [True, False, False]
    assert list(b_m) == [False, True, False]
    assert list(i_m) == [False, False, True]

    # project_centers: w = z, so a point at the optical axis maps to the centre
    # and a point behind the camera (z < 0) is culled.
    proj = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]],
                    dtype=np.float64)  # homogeneous w = z
    ui, vi, inside = project_centers(proj, np.array([[0.0, 0.0, 0.5]]), 101, 81)
    assert inside[0] and ui[0] == 50 and vi[0] == 40, (ui, vi, inside)
    _, _, inside_behind = project_centers(proj, np.array([[0.0, 0.0, -0.5]]), 101, 81)
    assert not inside_behind[0], "z<0 must be culled"

    # accumulate + label: two views, majority interior
    seven = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 0]])
    count = np.zeros((3, 3), dtype=np.int64)
    xyz = np.array([[-0.5, -0.5, 0.5], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5]])  # -> (0,0),(1,1),(2,2)
    proj2 = np.eye(4)  # w = 1: ndc == xyz, so corners map to the frame corners
    count = accumulate_labels([(proj2, seven), (proj2, seven)], xyz, count)
    labels = gt_label_from_counts(count)
    # (0,0)&(2,2) interior, (1,1) floater (void)
    assert labels[0] == "interior" and labels[2] == "interior", labels
    assert labels[1] == "floater", labels

    # calibrate recovers separable synthetic clusters
    rng = np.random.default_rng(0)
    n = 3000
    gt = np.array(["interior", "boundary", "floater"])
    lab = rng.choice(gt, size=n)
    disagree = np.where(lab == "interior", rng.normal(0.05, 0.03, n),
               np.where(lab == "boundary", rng.normal(0.55, 0.05, n),
                                       rng.normal(0.85, 0.05, n)))
    structure = np.where(lab == "boundary", rng.normal(0.95, 0.02, n),
                np.where(lab == "interior", rng.normal(0.98, 0.02, n),
                                         rng.normal(0.15, 0.05, n)))
    disagree = np.clip(disagree, 0, 1); structure = np.clip(structure, 0, 1)
    act = np.ones(n, dtype=bool)
    qd, qs, acc, _ = calibrate(disagree, structure, lab, act)
    assert acc > 0.9, f"calibration failed to separate clean clusters: acc={acc}"
    cents = nearest_centroid_accuracy(disagree, structure, lab, act)[1]
    assert cents["floater"][1] < cents["boundary"][1], "floater structure should be low"
    print(f"selftest OK  (best q_disagree={qd:.2f}, q_structure={qs:.2f}, acc={acc:.3f})")
    print(f"  centroids: interior={cents['interior'][0]:.2f}/{cents['interior'][1]:.2f} "
          f"boundary={cents['boundary'][0]:.2f}/{cents['boundary'][1]:.2f} "
          f"floater={cents['floater'][0]:.2f}/{cents['floater'][1]:.2f}  (disagree/structure)")
    return ok


# --------------------------------------------------------------------------- #
# GPU path: load a trained model + cameras, label, calibrate, scatter         #
# --------------------------------------------------------------------------- #
def _run(args):
    import torch  # noqa: F401  (heavy imports deferred so --selftest stays CPU-only)
    from argparse import ArgumentParser
    from arguments import ModelParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel

    # Rebuild the model + cameras exactly as render.py does.
    parser = ArgumentParser()
    lp = ModelParams(parser)
    parser.add_argument("--load_iteration", type=int, default=-1)
    parser.add_argument("--min_evidence", type=float, default=1.0,
                        help="seg_hist_min_evidence (evidence gate; doc 03)")
    parser.add_argument("--num_views", type=int, default=100,
                        help="max training views to project into (evenly spaced; 0 = all)")
    parser.add_argument("--stride", type=int, default=1,
                        help="extra view stride on top of the num_views subsample")
    parser.add_argument("--out", type=str, default="spike_route.png")
    parser.add_argument("--dump", type=str, default="",
                        help="optional .npz of raw arrays (disagree, structure, S, label)")
    parser.add_argument("--csv", type=str, default="", help="optional per-Gaussian CSV")
    parser.add_argument("--pass_acc", type=float, default=0.70,
                        help="calibrated-agreement threshold for a PASS verdict")
    parser.add_argument("--seg_hist", type=str, default="",
                        help="override path to seg_hist<N>.npz")
    parsed = parser.parse_args(sys.argv[1:]) if args is None else args

    dataset = lp.extract(parsed)
    model_path = dataset.model_path
    iteration = parsed.load_iteration
    if iteration < 0:
        from utils.system_utils import searchForMaxIteration
        iteration = searchForMaxIteration(os.path.join(model_path, "point_cloud"))

    # seg_hist: needed for the belief read-out. Saved separately from the PLY.
    hist_path = parsed.seg_hist or os.path.join(model_path, f"seg_hist{iteration}.npz")
    if not os.path.exists(hist_path):
        sys.exit(f"[spike] missing {hist_path}\n"
                 f"        re-run training with --save_iterations {iteration}")
    seg_hist = np.load(hist_path)["seg_hist"].astype(np.float64)  # [N, C]

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float64)
    n = xyz.shape[0]
    if seg_hist.shape[0] != n:
        sys.exit(f"[spike] seg_hist has {seg_hist.shape[0]} rows but the model has {n} "
                 f"Gaussians — mismatched iteration? (hist={hist_path})")

    # Belief read-out (mirror of GaussianModel.seg_hist_readout).
    S = seg_hist.sum(axis=1)
    P = seg_hist / np.clip(S, 1e-8, None)[:, None]
    k = min(2, P.shape[1])
    top2 = -np.sort(-P, axis=1)[:, :k]
    consistency = top2[:, 0]
    structure = top2[:, 0] + top2[:, 1] if k == 2 else top2[:, 0]
    disagree = 1.0 - top2[:, 0]

    # GT regime labels from projecting centres into the training views.
    cameras = scene.getTrainCameras()
    step = max(1, parsed.stride)
    if parsed.num_views and parsed.num_views < len(cameras):
        step = max(step, int(np.ceil(len(cameras) / parsed.num_views)))
    cams = []
    for i in range(0, len(cameras), step):
        cam = cameras[i]
        if getattr(cam, "gt_segmentation", None) is None:
            continue
        proj = cam.full_proj_transform.detach().cpu().numpy().astype(np.float64)
        seg = cam.gt_segmentation.detach().cpu().numpy().astype(np.int64)
        cams.append((proj, seg))
    if not cams:
        sys.exit("[spike] no training views carry GT segmentation — "
                 "is --segmentation_path set?")
    count = np.zeros((n, 3), dtype=np.int64)
    accumulate_labels(cams, xyz, count)
    gt = gt_label_from_counts(count)

    act = S > parsed.min_evidence

    # Report.
    print(f"[spike] iteration={iteration}  Gaussians={n}  views used={len(cams)}  "
          f"evidence-gated (act)={int(act.sum())} ({act.mean():.1%})")
    print("[spike] GT label counts: " + ", ".join(
        f"{r}={int((gt == r).sum())}" for r in (*REGIMES, "unobserved")))
    cents = {}
    for r in REGIMES:
        m = (gt == r)
        if m.any():
            cents[r] = (disagree[m].mean(), structure[m].mean())
            print(f"  {r:9s} disagree={disagree[m].mean():.3f}±{disagree[m].std():.3f}  "
                  f"structure={structure[m].mean():.3f}±{structure[m].std():.3f}")
    nc_acc, cents = nearest_centroid_accuracy(disagree, structure, gt, act)
    qd, qs, cal_acc, table = calibrate(disagree, structure, gt, act)
    print(f"[spike] nearest-centroid separability (act): {nc_acc:.3f}")
    print(f"[spike] §7.6 calibration best: q_disagree={qd}, q_structure={qs} "
          f"-> agreement={cal_acc:.3f}")
    verdict = "PASS" if (cal_acc == cal_acc and cal_acc >= parsed.pass_acc) else "FAIL/INCONCLUSIVE"
    print(f"[spike] verdict: {verdict}  (threshold {parsed.pass_acc}); "
          f"confirm on the scatter — the PNG is the decisive evidence")

    # Artifacts.
    if parsed.dump:
        np.savez_compressed(parsed.dump, disagree=disagree, structure=structure,
                            S=S, gt_label=gt.astype(str), act=act)
        print(f"[spike] wrote {parsed.dump}")
    if parsed.csv:
        pred_best = predict_regime(disagree, structure,
                                   np.quantile(disagree[act], qd) if qd is not None else 0.0,
                                   np.quantile(structure[act], qs) if qs is not None else 0.0)
        with open(parsed.csv, "w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["gaussian_id", "disagree", "structure", "S", "act", "gt_label", "pred_label"])
            for g in range(n):
                wtr.writerow([g, f"{disagree[g]:.6f}", f"{structure[g]:.6f}", f"{S[g]:.4f}",
                              int(act[g]), gt[g], pred_best[g]])
        print(f"[spike] wrote {parsed.csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - env dependent
        print(f"[spike] matplotlib unavailable ({e}); skipping PNG. Use the --dump/--csv artifacts.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    ax = axes[0]
    for r in (*REGIMES, "unobserved"):
        m = (gt == r) & act if r != "unobserved" else (gt == r)
        if m.any():
            ax.scatter(disagree[m], structure[m], s=4, alpha=0.35, color=COLOURS[r],
                       label=f"{r} (n={int(m.sum())})", linewidths=0)
    ax.set_xlabel("disagree  (1 - max_c P[g,c])")
    ax.set_ylabel("structure  (P_(1) + P_(2))")
    ax.set_title(f"THE SPIKE @ iter {iteration} — coloured by GT regime")
    ax.legend(loc="best", fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    ax = axes[1]
    if qd is not None:
        ad = np.linspace(0, 1, 200)
        ax.scatter(disagree[act], structure[act], s=3, alpha=0.15, color="#888", linewidths=0)
        tau_d = np.quantile(disagree[act], qd)
        tau_s = np.quantile(structure[act], qs)
        ax.axvline(tau_d, color="k", ls="--", lw=1)
        ax.axhline(tau_s, color="k", ls=":", lw=1)
        ax.text(tau_d, ax.get_ylim()[1] * 0.95, f" τ_d (q={qd})", fontsize=8, va="top")
        ax.text(ax.get_xlim()[1] * 0.02, tau_s, f" τ_s (q={qs})", fontsize=8, va="bottom")
        ax.set_title(f"§7.6 calibration: agreement={cal_acc:.3f}")
    ax.set_xlabel("disagree"); ax.set_ylabel("structure"); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(parsed.out, dpi=150)
    print(f"[spike] wrote {parsed.out}")


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--selftest", action="store_true",
                    help="run the pure-numpy logic tests (no torch/GPU)")
    known, _ = ap.parse_known_args()
    if known.selftest:
        _selftest()
        return
    _run(None)


if __name__ == "__main__":
    main()
