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


# --------------------------------------------------------------------- live 3D
class LiveDensifier3D:
    """Live 3D view updated at every densification event.

    Affected Gaussians are highlighted -- **clone = cyan**, **split = orange**,
    **pruned = red** -- and all other Gaussians are drawn in neutral grey.

    Designed to be updated from inside the training loop (pass the *current*
    ``gaussians.get_xyz``), but works just as well offline via :func:`replay`.
    A matplotlib 3D redraw is the bottleneck, so the background is decimated to
    ``max_bg`` points and the view can auto-rotate for a "process" feel.
    """

    def __init__(self, figsize=(8, 8), max_bg=8000, s_bg=1, s=14, show_bg=True,
                 rotate=True, azim_step=2.0, elev=25, azim=45, bg_color="#c7c7c7"):
        plt, _ = _mpl()
        self.plt = plt
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(projection="3d")
        self.max_bg = max_bg
        self.s_bg = s_bg
        self.s = s
        self.show_bg = show_bg
        self.rotate = rotate
        self.azim_step = azim_step
        self.elev = elev
        self.azim = azim
        self.bg_color = bg_color
        self._lims = None
        self._handle = None
        self._display()

    def _display(self):
        try:
            from IPython.display import display
            self._handle = display(self.fig, display_id=True)
        except Exception:                       # plain python / non-notebook use
            self._handle = None

    def _set_limits(self, xyz):
        if self._lims is None and xyz is not None:
            a = _t(xyz)
            lo = np.percentile(a, 1, axis=0)
            hi = np.percentile(a, 99, axis=0)
            pad = 0.05 * np.maximum(hi - lo, 1e-6)
            self._lims = [(float(lo[i] - pad[i]), float(hi[i] + pad[i])) for i in range(3)]
        if self._lims is not None:
            self.ax.set_xlim(*self._lims[0])
            self.ax.set_ylim(*self._lims[1])
            self.ax.set_zlim(*self._lims[2])

    def update(self, event: Dict[str, Any], all_xyz=None, save_path: Optional[str] = None):
        """Redraw for one event.  ``all_xyz`` is the *current* Gaussian centers."""
        ax = self.ax
        ax.clear()
        if self.show_bg and all_xyz is not None:
            bg = _t(all_xyz)
            bg = bg[::max(1, len(bg) // self.max_bg)]
            if len(bg):
                ax.scatter(bg[:, 0], bg[:, 1], bg[:, 2], s=self.s_bg,
                           c=self.bg_color, depthshade=False, linewidths=0)
        draw_event_3d(ax, event, background_xyz=None, s=self.s)
        if self.rotate:
            self.azim = (self.azim + self.azim_step) % 360
        ax.view_init(elev=self.elev, azim=self.azim)
        self._set_limits(all_xyz)
        ax.set_title(f"iter {event['iter']}   clone={n_clone(event)}  "
                     f"split={n_split(event)}  prune={n_prune(event)}")
        ax.set_axis_off()
        self.fig.canvas.draw()
        if self._handle is not None:
            self._handle.update(self.fig)
        if save_path is not None:
            self.fig.savefig(save_path, dpi=120, bbox_inches="tight")
        return self.fig

    def close(self):
        self.plt.close(self.fig)


def _collect_xyz(events) -> Optional[np.ndarray]:
    """Concatenate every highlighted point across all events (for auto-framing)."""
    chunks = []
    for e in events:
        for key in ("clone", "split"):
            for x in e[key]:
                chunks.append(_t(x["parent_xyz"]))
                chunks.append(_t(x["child_xyz"]))
        for x in e["prune"]:
            chunks.append(_t(x["xyz"]))
    return np.concatenate(chunks, 0) if chunks else None


def animate_replay(log, out_path=None, background_xyz=None, fps: int = 10, dpi: int = 120,
                   max_bg: int = 8000, s: int = 14, s_bg: int = 1, show_bg: bool = True,
                   rotate: bool = True, azim_step: float = 2.0, elev: float = 25,
                   azim: float = 45, bg_color: str = "#c7c7c7", figsize=(8, 8), writer=None):
    """Build a matplotlib animation over all logged events.

    If ``out_path`` is given it is saved as a video/GIF (``.mp4`` requires
    ffmpeg -- present on Colab; ``.gif`` uses Pillow and works everywhere) and
    ``(anim, fig)`` is returned.  Inline you can also do
    ``anim.to_jshtml()`` (no ffmpeg needed).
    """
    plt, _ = _mpl()
    import matplotlib.animation as animation

    events = log.events
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    # fixed framing so the camera does not jump between frames
    pts = _collect_xyz(events)
    if background_xyz is not None:
        bg = _t(background_xyz)
        pts = bg if pts is None else np.concatenate([pts, bg], 0)
    if pts is not None and len(pts):
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-6)
        lims = [(float(lo[i] - pad[i]), float(hi[i] + pad[i])) for i in range(3)]
    else:
        lims = None

    state = {"azim": azim}

    def frame(i):
        e = events[i]
        ax.clear()
        if show_bg and background_xyz is not None:
            b = _t(background_xyz)
            b = b[::max(1, len(b) // max_bg)]
            if len(b):
                ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=s_bg, c=bg_color,
                           depthshade=False, linewidths=0)
        draw_event_3d(ax, e, background_xyz=None, s=s)
        if rotate:
            state["azim"] = (state["azim"] + azim_step) % 360
        ax.view_init(elev=elev, azim=state["azim"])
        if lims is not None:
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])
            ax.set_zlim(*lims[2])
        ax.set_title(f"iter {e['iter']}   clone={n_clone(e)}  "
                     f"split={n_split(e)}  prune={n_prune(e)}")
        ax.set_axis_off()
        return (ax,)

    anim = animation.FuncAnimation(fig, frame, frames=len(events),
                                   interval=1000 / max(fps, 1), blit=False)
    if out_path is not None:
        anim.save(out_path, fps=fps, dpi=dpi, writer=writer)
    return anim, fig


def replay(log, background_xyz=None, delay: float = 0.05, **kwargs):
    """Animate every logged event.

    Default: inline live loop via :class:`LiveDensifier3D`.  Pass
    ``out_path="..."`` to write a video/GIF instead (mp4 needs ffmpeg, gif uses
    Pillow); then the return value is a matplotlib ``Animation``.  Extra kwargs
    (``fps``, ``dpi``, ``max_bg``, ``s``, ``elev``, ``figsize`` ...) are forwarded.
    """
    out_path = kwargs.pop("out_path", None)
    if out_path is not None:
        anim, _ = animate_replay(log, out_path=out_path,
                                 background_xyz=background_xyz, **kwargs)
        return anim
    import time
    live = LiveDensifier3D(**kwargs)
    for e in log.events:
        live.update(e, background_xyz)
        if delay:
            time.sleep(delay)
    return live


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


# ------------------------------------------------------------------ Plotly 3D
def _cat_pts(arrays) -> np.ndarray:
    arrays = [a for a in arrays if a is not None and len(a)]
    return np.concatenate(arrays, 0) if arrays else np.zeros((0, 3), dtype=np.float32)


def _event_traces(e: Dict[str, Any], size: float = 3.0):
    """Five Scatter3d traces for one event (clone/split parents+children, pruned)."""
    import plotly.graph_objects as go

    cp = _cat_pts([_t(x["parent_xyz"]) for x in e["clone"]])
    cc = _cat_pts([_t(x["child_xyz"]) for x in e["clone"]])
    sp = _cat_pts([_t(x["parent_xyz"]) for x in e["split"]])
    sc = _cat_pts([_t(x["child_xyz"]) for x in e["split"]])
    pr = _cat_pts([_t(x["xyz"]) for x in e["prune"]])

    def mk(P, name, color, symbol, opacity=0.9):
        return go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers", name=name,
            marker=dict(size=size, color=color, symbol=symbol, opacity=opacity),
            hoverinfo="name")

    return [
        mk(cp, "clone parent", C_CLONE, "circle", 1.0),
        mk(cc, "clone child", C_CLONE, "x", 0.8),
        mk(sp, "split parent", C_SPLIT, "diamond", 1.0),
        mk(sc, "split child", C_SPLIT, "x", 0.8),
        mk(pr, "pruned", C_PRUNE, "cross", 1.0),
    ]


def plotly_replay(log, background_xyz=None, max_bg: int = 8000, size: float = 3.0,
                  bg_size: float = 1.2, bg_color: str = "#bdbdbd", show_bg: bool = True,
                  path: Optional[str] = None, height: int = 700,
                  title: str = "SGDense densification \u2014 per-iteration 3D viewer"):
    """Interactive Plotly 3D viewer over every event in the log.

    Rotate / zoom freely, then drag the slider (or press *play*) to step through
    the densification iterations.  Affected Gaussians keep the usual colours
    (clone = cyan, split = orange, pruned = red) over a neutral grey background.

    Call ``fig.show()`` inline, or pass ``path="viewer.html"`` to write a
    portable file that opens in any browser without Python/Colab.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover
        raise ImportError("plotly is required for plotly_replay -- `pip install plotly`") from exc

    events = log.events
    if not events:
        print("no events to plot")
        return None

    # fixed axis ranges across every highlighted point (+ background)
    pts = _collect_xyz(events)
    if background_xyz is not None:
        b = _t(background_xyz)
        pts = b if pts is None else np.concatenate([pts, b], 0)
    if pts is not None and len(pts):
        lo = np.percentile(pts, 1, axis=0)
        hi = np.percentile(pts, 99, axis=0)
        pad = 0.05 * np.maximum(hi - lo, 1e-6)
        axes = [dict(range=[float(lo[i] - pad[i]), float(hi[i] + pad[i])]) for i in range(3)]
    else:
        axes = [{}, {}, {}]

    ev_traces = _event_traces(events[0], size=size)
    base = []
    if show_bg and background_xyz is not None:
        b = _t(background_xyz)
        b = b[::max(1, len(b) // max_bg)]
        base.append(go.Scatter3d(
            x=b[:, 0], y=b[:, 1], z=b[:, 2], mode="markers",
            marker=dict(size=bg_size, color=bg_color, opacity=0.35),
            name="other Gaussians", hoverinfo="skip"))
    base += ev_traces
    ev_idx = list(range(len(base) - len(ev_traces), len(base)))

    frames = [go.Frame(data=_event_traces(e, size=size), traces=ev_idx, name=str(e["iter"]))
              for e in events]
    steps = [dict(method="animate", label=str(e["iter"]),
                  args=[[str(e["iter"])],
                        dict(mode="immediate",
                             frame=dict(duration=0, redraw=True),
                             transition=dict(duration=0))])
             for e in events]

    fig = go.Figure(data=base, frames=frames)
    fig.update_layout(
        title=title, height=height, margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(xaxis=axes[0], yaxis=axes[1], zaxis=axes[2], aspectmode="data"),
        sliders=[dict(active=0, currentvalue=dict(prefix="iteration: "),
                      pad=dict(t=30), steps=steps)],
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.05, y=1.12, xanchor="left",
            buttons=[
                dict(label="\u25b6 play", method="animate",
                     args=[None, dict(frame=dict(duration=250, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0))]),
                dict(label="\u275a\u275a pause", method="animate",
                     args=[[None], dict(mode="immediate", redraw=False,
                                        transition=dict(duration=0))]),
            ])],
    )
    if path is not None:
        fig.write_html(path, include_plotlyjs="cdn")
    return fig
