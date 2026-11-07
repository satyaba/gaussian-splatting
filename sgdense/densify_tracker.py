"""Densification event tracker (SGDense).

Records *exactly* which Gaussians met the clone / split condition (and their
children), which were pruned and why, and the population at every
densification event -- with a stable global id per Gaussian so the whole
parent -> child lineage can be reconstructed offline.

Why a tracker at all
--------------------
The SGDense densification lifecycle lives on ``GaussianModel`` and is driven by
the free functions in the experiment notebook.  ``densification_postfix``
appends new rows at the tail; ``prune_points`` drops rows (split removes its
parents).  Those two primitives alone do not tell us *which* rows were chosen
or why, so the notebook calls the hooks below at the exact points where the
masks are known.

Row-order contract (must hold for the bookkeeping to stay aligned)
------------------------------------------------------------------
* clone : parents stay, ``k`` duplicates appended at the tail.
* split : ``N`` children per parent appended, then the parents are removed, so
          the new order is ``[survivors in order ++ children in parent order]``.
* prune : rows masked out are dropped (floater prune runs first, size/opacity
          prune last).

``gid`` is kept aligned with the live Gaussian population, so ``gid[i]`` is the
stable id of row ``i`` at any time.  ``flush`` verifies the length and resyncs
(and warns) if a hook was ever missed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrackerLog:
    """Lightweight, device-free view of a saved tracker (for offline plotting)."""

    events: List[Dict[str, Any]] = field(default_factory=list)
    summary: List[Dict[str, Any]] = field(default_factory=list)
    next_id: int = 0


def load_events(path: str) -> TrackerLog:
    """Reload a tracker log saved with :meth:`DensifyTracker.save`."""
    d = torch.load(path, map_location="cpu")
    return TrackerLog(events=d["events"], summary=d["summary"], next_id=d.get("next_id", 0))


class DensifyTracker:
    """Track clone / split / prune events over a live ``GaussianModel``.

    Parameters
    ----------
    gaussians:
        The model currently being densified (only read: xyz, scaling, opacity,
        ``seg_hist``, ``max_radii2D``).
    snapshot_max_points:
        If > 0, store a decimated copy of the *current* model in every event as
        ``event["all_xyz"]`` (capped at this many points).  This gives the
        live/replay/Plotly viewers an exact per-event background.  0 (default)
        disables snapshotting.  Cost per event ~ ``snapshot_max_points * 12``
        bytes.
    snapshot_every:
        Snapshot only every ``snapshot_every``-th event (1 = all events).
    snapshot_opacity:
        Also store the decimated opacities as ``event["all_opacity"]``.
    """

    ORIG, CLONE, SPLIT = 0, 1, 2

    def __init__(self, gaussians, snapshot_max_points: int = 0,
                 snapshot_every: int = 1, snapshot_opacity: bool = False):
        self.g = gaussians
        dev = gaussians.get_xyz.device
        n = int(gaussians.get_xyz.shape[0])

        self.device = dev
        self.regime = "all"                      # set by the caller before each hook
        self.gid = torch.arange(n, device=dev)   # stable id per live row
        self.next_id = n
        self.kind = torch.zeros(n, dtype=torch.long, device=dev)
        self.birth = torch.zeros(n, dtype=torch.long, device=dev)

        self.step = 0
        self.events: List[Dict[str, Any]] = []   # one record per densification event
        self.summary: List[Dict[str, Any]] = []  # population / counts per event
        self.pending: Optional[Dict[str, Any]] = None

        # optional per-event background snapshots (for exact offline context)
        self.snapshot_max_points = int(snapshot_max_points)
        self.snapshot_every = max(1, int(snapshot_every))
        self.snapshot_opacity = bool(snapshot_opacity)
        self._n_events = 0

    # ------------------------------------------------------------------ internals
    def _mint(self, k: int) -> torch.Tensor:
        ids = torch.arange(self.next_id, self.next_id + k, device=self.device)
        self.next_id += int(k)
        return ids

    def _ensure(self):
        if self.pending is None:
            self.pending = dict(iter=self.step, clone=[], split=[], prune=[], resets=[])

    def _attrs(self, P: int):
        g = self.g
        return (
            g.get_xyz.detach(),
            g.get_scaling.detach().max(dim=1).values,
            g.get_opacity.detach().squeeze(1),
            g.seg_hist.detach(),
        )

    @staticmethod
    def _class_of(sub_hist: torch.Tensor) -> torch.Tensor:
        """argmax over the segmentation mass histogram, or -1 when disabled."""
        if sub_hist.dim() == 2 and sub_hist.shape[1] > 0:
            return sub_hist.argmax(dim=1).cpu()
        return torch.full((sub_hist.shape[0],), -1, dtype=torch.long)

    def _snapshot(self) -> Dict[str, Any]:
        """Decimated copy of the current model for per-event background context."""
        xyz = self.g.get_xyz.detach()
        n = int(xyz.shape[0])
        step = max(1, n // max(1, self.snapshot_max_points))
        sl = slice(None, None, step)
        out: Dict[str, Any] = {"all_xyz": xyz[sl].float().cpu()}
        if self.snapshot_opacity:
            op = self.g.get_opacity.detach().squeeze(1)
            out["all_opacity"] = op[sl].float().cpu()
        return out

    # ---------------------------------------------------------------------- hooks
    def on_clone(self, sel, grad, child_xyz, N: int = 1, regime: Optional[str] = None):
        """Record a clone.  Call *after* ``densification_postfix``.

        Parameters
        ----------
        sel : bool tensor [P]
            Which of the P pre-append rows were chosen.
        grad : float tensor [P]
            The (padded) 2D-gradient norm used for the selection.
        child_xyz : float tensor [k, 3]
            The appended duplicates (``k == sel.sum()`` for clones).
        """
        sel = sel.bool()
        k = int(child_xyz.shape[0])
        if not bool(sel.any()) and k == 0:
            return
        self._ensure()
        P = int(sel.shape[0])
        xyz, scale, opac, hist = self._attrs(P)
        kids = self._mint(k)
        self.pending["clone"].append(dict(
            regime=self.regime if regime is None else regime, N=int(N),
            parent_gid=self.gid[:P][sel].cpu(),
            parent_xyz=xyz[:P][sel].cpu(),
            parent_grad=grad[sel].float().cpu(),
            parent_max_scale=scale[:P][sel].cpu(),
            parent_opacity=opac[:P][sel].cpu(),
            parent_class=self._class_of(hist[:P][sel]),
            child_gid=kids.cpu(),
            child_xyz=xyz[P:P + k].cpu(),
            child_scale=scale[P:P + k].cpu()))
        # clone: parents stay, children appended at the tail
        self.gid = torch.cat([self.gid, kids])
        self.kind = torch.cat([self.kind, torch.full((k,), self.CLONE, device=self.device)])
        self.birth = torch.cat([self.birth, torch.full((k,), self.step, device=self.device)])

    def on_split(self, sel, grad, child_xyz, N: int = 2, regime: Optional[str] = None):
        """Record a split.  Call *after* ``densification_postfix``, *before*
        the parents are pruned.  Children are in parent order:
        ``[p0]*N, [p1]*N, ...``."""
        sel = sel.bool()
        k = int(child_xyz.shape[0])
        if not bool(sel.any()) and k == 0:
            return
        self._ensure()
        P = int(sel.shape[0])
        xyz, scale, opac, hist = self._attrs(P)
        kids = self._mint(k)
        self.pending["split"].append(dict(
            regime=self.regime if regime is None else regime, N=int(N),
            parent_gid=self.gid[:P][sel].cpu(),
            parent_xyz=xyz[:P][sel].cpu(),
            parent_grad=grad[sel].float().cpu(),
            parent_max_scale=scale[:P][sel].cpu(),
            parent_opacity=opac[:P][sel].cpu(),
            parent_class=self._class_of(hist[:P][sel]),
            child_gid=kids.cpu(),
            child_xyz=xyz[P:P + k].cpu(),
            child_scale=scale[P:P + k].cpu()))
        # split: survivors keep order, children appended (parents removed)
        self.gid = torch.cat([self.gid[:P][~sel], kids])
        self.kind = torch.cat([self.kind[:P][~sel],
                               torch.full((k,), self.SPLIT, device=self.device)])
        self.birth = torch.cat([self.birth[:P][~sel],
                                torch.full((k,), self.step, device=self.device)])

    def on_prune(self, mask, reason: str = "unknown"):
        """Record a prune.  Call immediately *before* ``prune_points``."""
        mask = mask.bool()
        self._ensure()
        P = int(mask.shape[0])
        xyz, scale, opac, _ = self._attrs(P)
        self.pending["prune"].append(dict(
            reason=reason,
            gid=self.gid[:P][mask].cpu(),
            xyz=xyz[:P][mask].cpu(),
            opacity=opac[:P][mask].cpu(),
            max_scale=scale[:P][mask].cpu(),
            radii=self.g.max_radii2D.detach()[:P][mask].cpu()))
        self.gid = self.gid[:P][~mask]
        self.kind = self.kind[:P][~mask]
        self.birth = self.birth[:P][~mask]

    def on_reset(self):
        """Record an opacity reset (population unchanged)."""
        self._ensure()
        self.pending["resets"].append(self.step)

    # ------------------------------------------------------------------ lifecycle
    def flush(self, iteration: int):
        """Close the current event and append it if anything happened.

        Also verifies that the tracked population still matches the model; on a
        mismatch it resyncs (ids are lost) and prints a warning.
        """
        n_model = int(self.g.get_xyz.shape[0])
        if int(self.gid.shape[0]) != n_model:
            print(f"[DensifyTracker] resync at {iteration}: "
                  f"{int(self.gid.shape[0])} -> {n_model} (a hook was missed)")
            self.gid = torch.arange(n_model, device=self.device) + self.next_id
            self.next_id += n_model
            self.kind = torch.zeros(n_model, dtype=torch.long, device=self.device)
            self.birth = torch.full((n_model,), iteration, dtype=torch.long, device=self.device)
            self.pending = None
            return

        p = self.pending or dict(iter=iteration, clone=[], split=[], prune=[], resets=[])
        c = sum(len(x["parent_gid"]) for x in p["clone"])
        s = sum(len(x["parent_gid"]) for x in p["split"])
        pr = sum(len(x["gid"]) for x in p["prune"])
        if c or s or pr or p["resets"]:
            rec = dict(iter=iteration, clone=p["clone"], split=p["split"],
                       prune=p["prune"], resets=p["resets"])
            if self.snapshot_max_points and (self._n_events % self.snapshot_every == 0):
                rec.update(self._snapshot())
            self.events.append(rec)
            self._n_events += 1
            self.summary.append(dict(iter=iteration, n=n_model,
                                     clones=c, splits=s, prunes=pr))
        self.pending = None

    def export(self) -> Dict[str, Any]:
        """Device-free dict suitable for ``torch.save``."""
        return dict(events=self.events, summary=self.summary, next_id=self.next_id)

    def save(self, path: str):
        torch.save(self.export(), path)
        return path
