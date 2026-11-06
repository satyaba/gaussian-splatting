"""Post-hoc densification visualizer (SGDense).

Pure plotting helpers over a :class:`~sgdense.densify_tracker.DensifyTracker`
log (or a :class:`~sgdense.densify_tracker.TrackerLog` reloaded from disk).
Nothing here touches CUDA or the training loop, so it can be re-run on a
laptop from ``densify_events.pt``.

Panels
------
``plot_timeline``          clone / split / prune counts per event + population
``plot_conditions``        grad vs max-scale of selected Gaussians + thresholds
``plot_prune_breakdown``   pruned per event, split by reason
``draw_event_3d`` / ``plot_event_3d`` / ``show_event``
                           parents, children (with connectors) and pruned (3D)
``build_family_index`` / ``show_lineage``
                           ancestors / descendants of a single Gaussian

matplotlib is imported lazily inside the functions so that importing the
``sgdense`` package never pulls it into training.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# convenience re-export: the saved log is loaded through this module too
from sgdense.densify_tracker import TrackerLog, load_events

C_ORIG = "#9ca3af"
C_CLONE = "#22d3ee"
C_SPLIT = "#fb923c"
C_PRUNE = "#ef4444"

_REASON_COLORS = {"floater": "#ef4444", "opacity": "#f59e0b", "opacity+size": "#a855f7"}


def _mpl():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    return plt, Line3DCollection


def _t(x) -> np.ndarray:
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def n_clone(e: Dict[str, Any]) -> int:
    return sum(len(x["parent_gid"]) for x in e["clone"])


def n_split(e: Dict[str, Any]) -> int:
    return sum(len(x["parent_gid"]) for x in e["split"])


def n_prune(e: Dict[str, Any]) -> int:
    return sum(len(x["gid"]) for x in e["prune"])


# --------------------------------------------------------------------------- 2D
def plot_timeline(log, ax=None, figsize=(12, 3.6)):
    """Clone (+) / split (+) / prune (-) counts per event, plus #Gaussians."""
    plt, _ = _mpl()
    events, summary = log.events, log.summary
    its = [e["iter"] for e in events]
    c = [n_clone(e) for e in events]
    s = [n_split(e) for e in events]
    p = [n_prune(e) for e in events]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    ax.bar(its, c, width=60, color=C_CLONE, label="clone parents")
    ax.bar(its, s, width=60, bottom=c, color=C_SPLIT, label="split parents")
    ax.bar(its, [-v for v in p], width=60, color=C_PRUNE, label="pruned")
    ax.axhline(0, color="k", lw=0.6)
    ax2 = ax.twinx()
    if summary:
        ax2.plot([x["iter"] for x in summary], [x["n"] for x in summary],
                 color="#111827", lw=1.5, label="# Gaussians")
    ax2.set_ylabel("# Gaussians")
    for e in events:
        for r in e["resets"]:
            ax.axvline(r, color="#6b7280", ls="--", lw=0.8)
    ax.set_xlabel("iteration")
    ax.set_ylabel("event size (parents)")
    ax.set_title("Densification events  (dashed = opacity reset)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def plot_conditions(log, grad_thr: Optional[float] = None, size_thr: Optional[float] = None,
                    ax=None, figsize=(8, 6)):
    """Grad-norm vs max-scale of every selected Gaussian, against the thresholds.

    Clone points sit left of the size threshold, split points right of it; both
    sit above the gradient threshold -- so this panel verifies the condition.
    """
    plt, _ = _mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    for e in log.events:
        for x in e["clone"]:
            ax.scatter(_t(x["parent_max_scale"]), _t(x["parent_grad"]),
                       s=8, c=C_CLONE, alpha=0.35, edgecolors="none")
        for x in e["split"]:
            ax.scatter(_t(x["parent_max_scale"]), _t(x["parent_grad"]),
                       s=8, c=C_SPLIT, alpha=0.35, edgecolors="none")
    if grad_thr is not None:
        ax.axhline(grad_thr, color="k", ls="--", lw=1, label=f"grad thr = {grad_thr:g}")
    if size_thr is not None:
        ax.axvline(size_thr, color="k", ls=":", lw=1, label=f"size thr = {size_thr:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("max Gaussian scale")
    ax.set_ylabel("2D-gradient norm")
    ax.set_title("Selection conditions  (clone left of size thr, split right)")
    if grad_thr is not None or size_thr is not None:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_prune_breakdown(log, ax=None, figsize=(12, 3.2)):
    """Pruned Gaussians per event, stacked by reason."""
    plt, _ = _mpl()
    events = log.events
    reasons = sorted({x["reason"] for e in events for x in e["prune"]})
    its = [e["iter"] for e in events]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    bottom = [0] * len(its)
    for r in reasons:
        vals = [sum(len(x["gid"]) for x in e["prune"] if x["reason"] == r) for e in events]
        ax.bar(its, vals, width=60, bottom=bottom, color=_REASON_COLORS.get(r, "#6b7280"), label=r)
        bottom = [a + b for a, b in zip(bottom, vals)]
    ax.set_xlabel("iteration")
    ax.set_ylabel("pruned")
    ax.set_title("Prune reasons per event")
    if reasons:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- 3D
def draw_event_3d(ax, event: Dict[str, Any], background_xyz=None, s: int = 14):
    """Draw one event onto a 3D axes: parents, children, connectors, pruned."""
    _, Line3DCollection = _mpl()
    if background_xyz is not None:
        bg = _t(background_xyz)
        bg = bg[::max(1, len(bg) // 6000)]
        if len(bg):
            ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=1, c="#d1d5db",
                       depthshade=False, linewidths=0)

    for key, color, marker in (("clone", C_CLONE, "o"), ("split", C_SPLIT, "^")):
        for x in event[key]:
            p = _t(x["parent_xyz"])
            ch = _t(x["child_xyz"])
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=s * 1.6, c=color,
                       depthshade=False, marker=marker)
            ax.scatter(ch[:, 0], ch[:, 1], ch[:, 2], s=s * 0.8, c=color,
                       depthshade=False, marker="x")
            N = x["N"] or 1
            if len(p) and len(ch) == N * len(p):
                lines = [[p[i], ch[i * N + j]] for i in range(len(p)) for j in range(N)]
                ax.add_collection3d(Line3DCollection(lines, colors=color,
                                                     linewidths=0.4, alpha=0.5))

    for x in event["prune"]:
        q = _t(x["xyz"])
        ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=s, c=C_PRUNE,
                   depthshade=False, marker="x")
    ax.set_axis_off()


def plot_event_3d(log, iteration: int, background_xyz=None, ax=None, figsize=(8, 8)):
    """3D view of the parents/children/pruned at one densification event."""
    plt, _ = _mpl()
    e = next((e for e in log.events if e["iter"] == iteration), None)
    if e is None:
        print("no densification event at iteration", iteration)
        return None
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    draw_event_3d(ax, e, background_xyz=background_xyz)
    ax.set_title(f"iteration {iteration}   clone={n_clone(e)}  "
                 f"split={n_split(e)}  prune={n_prune(e)}")
    fig.tight_layout()
    return fig


def show_event(log, iteration: int, background_xyz=None, figsize=(15, 7)):
    """One event: 3D view plus clone/split and prune-reason summaries."""
    plt, _ = _mpl()
    e = next((e for e in log.events if e["iter"] == iteration), None)
    if e is None:
        print("no densification event at iteration", iteration)
        return None
    fig = plt.figure(figsize=figsize)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_event_3d(ax3, e, background_xyz=background_xyz)
    ax3.set_title(f"iteration {iteration}\n"
                  f"clone={n_clone(e)}  split={n_split(e)}  prune={n_prune(e)}")

    axb = fig.add_subplot(2, 2, 2)
    axb.bar(["clone", "split"], [n_clone(e), n_split(e)], color=[C_CLONE, C_SPLIT])
    axb.set_title("densified this event")

    axr = fig.add_subplot(2, 2, 4)
    reasons = sorted({x["reason"] for x in e["prune"]})
    vals = [sum(len(x["gid"]) for x in e["prune"] if x["reason"] == r) for r in reasons] or [0]
    axr.bar(reasons or ["-"], vals, color=C_PRUNE)
    axr.set_title("pruned by reason")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------- lineage
def build_family_index(log) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Return ``(parent_of, children_of)`` maps over all logged events."""
    parent_of: Dict[int, int] = {}
    children_of: Dict[int, List[int]] = {}
    for e in log.events:
        for key in ("clone", "split"):
            for x in e[key]:
                N = x["N"] or 1
                pg = x["parent_gid"].tolist()
                cg = x["child_gid"].tolist()
                for i, g in enumerate(pg):
                    kids = cg[i * N:(i + 1) * N]
                    children_of[g] = kids
                    for kid in kids:
                        parent_of[kid] = g
    return parent_of, children_of


def show_lineage(log, gid: int, max_print: int = 60):
    """Print and return the ancestor chain and descendants of one Gaussian id."""
    parent_of, children_of = build_family_index(log)
    up = [gid]
    while up[-1] in parent_of:
        up.append(parent_of[up[-1]])
    up.reverse()
    down, frontier = [gid], [gid]
    while frontier:
        nxt: List[int] = []
        for g in frontier:
            nxt += children_of.get(g, [])
        frontier = nxt
        down += nxt
    print(f"gid {gid}: {len(up) - 1} ancestors, {len(down) - 1} descendants")
    print("ancestors  :", up)
    print("descendants:", down[:max_print] + (["..."] if len(down) > max_print else []))
    return up, down
