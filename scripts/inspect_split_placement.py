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
# A `--placement_mode random` run is expected to report split placed_frac ~ 0;
# a `boundary_normal` run reports the fraction of splits that actually used a
# deterministic direction. For clones: a vanilla clone is a coincident duplicate
# (child offset ~ 0, `clone_directional_frac ~ 0`), whereas stage-2 directional
# clone has symmetric children with a non-zero offset (`clone_directional_frac`
# high, `clone_multi_frac = 1`).

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from sgdense import load_events  # noqa: E402


def split_placement_stats(log, tol=1e-4, off_tol=1e-6):
    """Classify each split/clone parent as *placed* (deterministic) or not.

    Split: deterministic placement is symmetric about the parent (midpoint ==
    parent); random N(0,Sigma) is not.

    Clone: a vanilla clone is a coincident duplicate (child offset ~ 0); a
    stage-2 directional clone has symmetric children with a non-zero offset.

    Returns population totals, placement fractions, median deviations, and
    per-event split counts.
    """
    n_events = n_clone = n_split = placed = total = 0
    dev_placed, dev_other = [], []
    clone_total = clone_multi = clone_directional = 0
    clone_offs = []
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

        for x in ev.get("clone", []):
            parent = x["parent_xyz"]
            kids = x["child_xyz"]
            K = int(x["parent_gid"].numel())
            N = int(x.get("N", 1))
            clone_total += K
            if K and N > 1:
                clone_multi += K
            if K == 0 or kids.shape[0] != K * N:
                continue
            kids_t = kids.reshape(N, K, 3)                       # [N,K,3]
            mid = kids_t.mean(dim=0)
            dev = (mid - parent).norm(dim=1)                     # [K]
            offs = (kids_t - parent.unsqueeze(0)).norm(dim=2).mean(dim=0)   # [K]
            thr = tol * (1.0 + parent.norm(dim=1))
            directional = (dev <= thr) & (offs > off_tol)
            clone_directional += int(directional.sum())
            clone_offs.append(offs)

    def _cat(xs):
        return torch.cat(xs) if xs else torch.zeros(0)

    dp, do = _cat(dev_placed), _cat(dev_other)
    co = _cat(clone_offs)
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
        clone_parents_total=clone_total,
        clone_multi_frac=(clone_multi / clone_total if clone_total else 0.0),
        clone_directional=clone_directional,
        clone_directional_frac=(clone_directional / clone_total if clone_total else 0.0),
        clone_offset_median=(float(co.median()) if co.numel() else None),
        per_event=per_event,
    )


def _print(name, st):
    print(f"\n=== {name} ===")
    print(f"events={st['n_events']}  clone parents={st['n_clone_parents']}  "
          f"split parents={st['n_split_parents']}  (clone share={st['clone_share']:.1%})")
    print(f"SPLIT  parents scored={st['split_parents_total']}  "
          f"placed(deterministic)={st['placed']}  placed_frac={st['placed_frac']:.3f}")
    print(f"       midpoint dev: placed median={st['dev_placed_median']}  "
          f"random median={st['dev_other_median']}")
    print(f"CLONE  parents scored={st['clone_parents_total']}  "
          f"multi-child frac={st['clone_multi_frac']:.3f}  "
          f"directional_frac={st['clone_directional_frac']:.3f}  "
          f"offset median={st['clone_offset_median']}")
    pe = [e for e in st["per_event"] if not e.get("skipped")]
    if pe:
        ps = [(e["placed"] / e["K"]) if e["K"] else 0.0 for e in pe]
        fin = [p for p in ps if p == p]
        if fin:
            print(f"per-event split placed_frac: first={fin[0]:.3f} last={fin[-1]:.3f} "
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
