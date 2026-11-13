# Diagnostics for segmentation-boundary split placement (docs/08).
#
# The placement diagnostic we actually need is NOT "how many Gaussians are on a
# class boundary" (that is `placement/valid_frac` in TensorBoard) but **of the
# split parents selected, how many were placed deterministically vs fell back to
# random N(0, Sigma)**. That ratio is not logged, but it is recoverable offline
# from an existing `densify_events.pt`:
#
#   * deterministic placement puts the N children of a parent SYMMETRICALLY about
#     it (x +/- rho*scale*sigma_major*n_hat), so their midpoint == the parent;
#   * vanilla random N(0, Sigma) children have midpoint != parent.
#
# `DensifyTracker.on_split` stores `parent_xyz [K,3]` and `child_xyz [K*N,3]`
# (tiled order c*K + k, matching `repeat(N, 1)`), so `child.reshape(N, K, 3)
# .mean(0)` is the per-parent midpoint.
#
# Usage:
#   python scripts/inspect_split_placement.py <model_dir> [<model_dir> ...]
#   python scripts/inspect_split_placement.py --tol 1e-4 <model_dir>
#
# A `--placement_mode random` run is expected to report placed_frac ~ 0; a
# `boundary_normal` run reports the fraction of splits that actually used a
# deterministic direction.

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense import load_events  # noqa: E402


def split_placement_stats(log, tol=1e-4):
    """Classify each split parent as placed (symmetric children) or random.

    Returns a dict with population totals, the placed fraction, the median
    midpoint deviation for each class (a sanity gap), and per-event counts.
    """
    n_events = n_clone = n_split = placed = total = 0
    dev_placed, dev_other = [], []
    per_event = []

    for ev in log.events:
        n_events += 1
        n_clone += sum(int(x["parent_gid"].numel()) for x in ev.get("clone", []))
        for x in ev.get("split", []):
            parent = x["parent_xyz"]
            kids = x["child_xyz"]
            K = int(x["parent_gid"].numel())
            N = int(x.get("N", 2))
            n_split += K
            if K == 0 or kids.shape[0] != K * N:
                # Layout not present (older log) -- skip, but keep the count.
                per_event.append(dict(iter=ev["iter"], K=K, placed=0, skipped=True))
                continue
            mid = kids.reshape(N, K, 3).mean(dim=0)              # [K,3]
            dev = (mid - parent).norm(dim=1)                     # [K]
            thr = tol * (1.0 + parent.norm(dim=1))
            pk = dev <= thr
            placed += int(pk.sum())
            total += K
            dev_placed.append(dev[pk])
            dev_other.append(dev[~pk])
            per_event.append(dict(iter=ev["iter"], K=K, placed=int(pk.sum())))

    def _cat(xs):
        return torch.cat(xs) if xs else torch.zeros(0)

    dp, do = _cat(dev_placed), _cat(dev_other)
    denom = n_clone + n_split
    return dict(
        n_events=n_events,
        n_clone_parents=n_clone,
        n_split_parents=n_split,
        clone_share=(n_clone / denom if denom else 0.0),
        split_parents_total=total,
        placed=placed,
        placed_frac=(placed / total if total else 0.0),
        dev_placed_median=(float(dp.median()) if dp.numel() else None),
        dev_other_median=(float(do.median()) if do.numel() else None),
        per_event=per_event,
    )


def _print(name, st):
    print(f"\n=== {name} ===")
    print(f"events={st['n_events']}  clone parents={st['n_clone_parents']}  "
          f"split parents={st['n_split_parents']}  (clone share={st['clone_share']:.1%})")
    print(f"split parents scored: {st['split_parents_total']}")
    print(f"PLACED (deterministic): {st['placed']}  "
          f"-> placed_frac={st['placed_frac']:.3f}")
    print(f"midpoint dev: placed median={st['dev_placed_median']}  "
          f"random median={st['dev_other_median']}")
    pe = [e for e in st["per_event"] if not e.get("skipped")]
    if pe:
        ps = [(e["placed"] / e["K"]) if e["K"] else 0.0 for e in pe]
        fin = [p for p in ps if p == p]
        if fin:
            print(f"per-event placed_frac: first={fin[0]:.3f} last={fin[-1]:.3f} "
                  f"min={min(fin):.3f} max={max(fin):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dirs", nargs="+", help="dirs containing densify_events.pt")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="midpoint tolerance (relative to |parent|) for 'placed'")
    args = ap.parse_args()

    for md in args.model_dirs:
        path = md if os.path.isfile(md) else os.path.join(md, "densify_events.pt")
        if not os.path.exists(path):
            print(f"[skip] no densify_events.pt at {path}")
            continue
        st = split_placement_stats(load_events(path), tol=args.tol)
        _print(os.path.basename(md.rstrip("/\\")) or md, st)


if __name__ == "__main__":
    main()
