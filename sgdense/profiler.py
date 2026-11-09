"""Opt-in per-iteration phase profiler for 3DGS training loops.

Answers: *how much extra wall-time does SGDense spend per iteration vs vanilla
3DGS, and where* — broken down by phase, from rendering through attribute
optimisation.

Design
------
The profiler is a tiny, dependency-free object (``IterProfiler``). It is a
no-op unless an iteration spec is supplied, so ordinary runs pay only a couple
of attribute lookups per phase.

Timing is **host wall-clock with a CUDA synchronise at every phase boundary**.
That makes the per-phase numbers additive and directly interpretable: each
phase is what the whole training loop would wait on. Read on why:

* Rendering / backward / optimiser work is asynchronous on the GPU. Without a
  boundary sync a phase's wall-clock would just measure kernel *launch* time.
* The SGDense densification stack does real **CPU** work (numpy conversions and
  the batched ``np.linalg.eigh`` plane fit in ``sgdense.geometric``); CUDA
  events would report ~0 for it. Wall-clock captures both CPU and GPU.

A per-iteration CUDA-event pair is *also* recorded (``iter_gpu_ms``) as a
cross-check of the synchronised total.

Phases (shared with the vanilla shim so the two are comparable)::

    render  loss  backward  densify  optimizer     (measured)
    misc                                          (residual = total - measured)

The ``misc`` phase is whatever the loop does outside those five (LR updates,
camera pick, logging, ``training_report``, checkpointing). It is derived, not
timed, so the five measured phases plus ``misc`` always sum to the iteration.

Usage
-----
In the SGDense training loop (already wired in ``train.py``)::

    prof = IterProfiler(spec="3999:4099", out_path=".../iter_profile.json")
    ...
    prof.begin_iteration(iteration, n_gaussians=int(g.get_xyz.shape[0]))
    with prof.phase("render"):
        ...
    prof.start("densify")
    ...
    prof.stop("densify")
    prof.end_iteration(densify=ran_densify)
    ...
    prof.save()

Or run the upstream-vanilla shim (``scripts/vanilla_iter_profile.py``) which
emits the identical JSON schema.

The JSON is passed to :func:`compare_profiles` / :func:`format_comparison` to
produce the SGDense-vs-vanilla table (also exposed as the
``python -m experiments.profile`` CLI).
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch

# Phase names, in the order the training loop executes them. ``misc`` is the
# residual and is appended after these.
PHASES = ("render", "loss", "backward", "densify", "optimizer")


# --------------------------------------------------------------------------- #
# iteration spec parsing                                                       #
# --------------------------------------------------------------------------- #
def parse_iters(spec: Union[None, str, Iterable[int]]) -> Optional[object]:
    """Parse a profiling iteration spec.

    Accepts

    ===========================  =================================================
    ``None`` / ``""`` / ``off``  disabled
    ``"all"`` / ``"*"``          every iteration
    ``"a:b"``                    inclusive range (e.g. ``"3999:4099"``)
    ``"a,b,c"`` / ``"a b c"``    explicit set
    an iterable of ints          explicit set
    ===========================  =================================================
    """
    if spec is None:
        return None
    if isinstance(spec, (set, frozenset, tuple, list)):
        return ("set", {int(x) for x in spec})
    spec = str(spec).strip()
    if spec == "" or spec.lower() in ("off", "none", "false", "0"):
        return None
    if spec.lower() in ("all", "*"):
        return "all"
    if ":" in spec:
        a, b = spec.split(":", 1)
        return ("range", int(a), int(b))
    return ("set", {int(x) for x in spec.replace(",", " ").split()})


def _spec_str(spec) -> str:
    if spec is None:
        return "off"
    if spec == "all":
        return "all"
    if spec[0] == "range":
        return f"{spec[1]}:{spec[2]}"
    return ",".join(str(i) for i in sorted(spec[1]))


# --------------------------------------------------------------------------- #
# no-op sentinels (used when a phase is not being profiled)                    #
# --------------------------------------------------------------------------- #
class _NullPhase:
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NULL_PHASE = _NullPhase()


class _Phase:
    __slots__ = ("_p", "_name", "_t0")

    def __init__(self, prof: "IterProfiler", name: str):
        self._p = prof
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        self._p._sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._p._sync()
        dt = (time.perf_counter() - self._t0) * 1000.0
        rec = self._p._cur
        if rec is not None:
            ph = rec["phases_ms"]
            ph[self._name] = ph.get(self._name, 0.0) + dt
        return False


# --------------------------------------------------------------------------- #
# profiler                                                                     #
# --------------------------------------------------------------------------- #
class IterProfiler:
    """Collect per-iteration phase timings into a JSON-serialisable record set."""

    def __init__(self, spec=None, out_path: Optional[str] = None,
                 method: str = "sgdense", sync: bool = True):
        self.spec = parse_iters(spec)
        self.out_path = out_path
        self.method = method
        self.sync = bool(sync)
        self.enabled = self.spec is not None
        self.records: Dict[int, dict] = {}

        self._cur: Optional[dict] = None
        self._t_iter0 = 0.0
        self._ev_iter0: Optional[torch.cuda.Event] = None
        self._ev_iter1: Optional[torch.cuda.Event] = None
        self._open: Dict[str, float] = {}
        self._cuda = torch.cuda.is_available()

    # -- lifecycle -------------------------------------------------------- #
    def _sync(self) -> None:
        if self.sync and self._cuda:
            torch.cuda.synchronize()

    def should_profile(self, iteration) -> bool:
        if not self.enabled or iteration is None:
            return False
        s = self.spec
        if s == "all":
            return True
        if s[0] == "range":
            return s[1] <= iteration <= s[2]
        return iteration in s[1]

    def begin_iteration(self, iteration, **meta) -> bool:
        """Open a profiled iteration. Returns True if it will be recorded."""
        self._cur = None
        self._open = {}
        if not self.should_profile(iteration):
            return False
        self._sync()
        self._t_iter0 = time.perf_counter()
        if self._cuda:
            self._ev_iter0 = torch.cuda.Event(enable_timing=True)
            self._ev_iter1 = torch.cuda.Event(enable_timing=True)
            self._ev_iter0.record()
        self._cur = {"iteration": int(iteration), "phases_ms": {}, "meta": dict(meta)}
        return True

    def phase(self, name: str):
        """Context-manager phase timer (no-op when not profiling)."""
        if self._cur is None:
            return _NULL_PHASE
        return _Phase(self, name)

    def start(self, name: str) -> None:
        """Manual phase start (for large blocks that are awkward to re-indent)."""
        if self._cur is None:
            return
        self._sync()
        self._open[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if self._cur is None:
            return
        t0 = self._open.pop(name, None)
        if t0 is None:
            return
        self._sync()
        dt = (time.perf_counter() - t0) * 1000.0
        rec = self._cur
        ph = rec["phases_ms"]
        ph[name] = ph.get(name, 0.0) + dt

    def end_iteration(self, **meta) -> Optional[dict]:
        """Close the current iteration, deriving ``misc`` and the totals."""
        if self._cur is None:
            return None
        rec = self._cur
        if meta:
            rec["meta"].update(meta)
        self._sync()
        rec["iter_ms"] = (time.perf_counter() - self._t_iter0) * 1000.0
        if self._cuda and self._ev_iter0 is not None:
            self._ev_iter1.record()
            rec["iter_gpu_ms"] = self._ev_iter0.elapsed_time(self._ev_iter1)
        measured = sum(rec["phases_ms"].values())
        rec["phase_sum_ms"] = measured
        rec["misc_ms"] = max(0.0, rec["iter_ms"] - measured)
        # Expose the residual as a phase too so phases_ms + totals are coherent.
        rec["phases_ms"]["misc"] = rec["misc_ms"]
        self.records[rec["iteration"]] = rec
        self._cur = None
        return rec

    # -- output ----------------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "spec": _spec_str(self.spec),
            "phases": list(PHASES) + ["misc"],
            "records": self.records,
            "summary": summarize_profile(self.records),
        }

    def save(self, path: Optional[str] = None) -> Optional[str]:
        path = path or self.out_path
        if not path:
            return None
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path


# --------------------------------------------------------------------------- #
# aggregation                                                                  #
# --------------------------------------------------------------------------- #
def _stat(values: Sequence[Optional[float]]) -> dict:
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"n": 0}
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is always present in training
        s = sorted(v)
        n = len(s)
        return {"n": n, "mean_ms": sum(s) / n, "median_ms": s[n // 2],
                "min_ms": s[0], "max_ms": s[-1]}
    a = np.asarray(v, dtype=float)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "p90_ms": float(np.percentile(a, 90)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def summarize_profile(records, phases: Sequence[str] = PHASES) -> dict:
    """Aggregate per-iteration records into overall + densify/no-densify groups.

    ``records`` is the ``iteration -> record`` dict, or a list of records.
    """
    rows = list(records.values()) if isinstance(records, dict) else list(records)
    all_phases = list(phases) + ["misc"]
    groups = {
        "all": rows,
        "densify": [r for r in rows if r.get("meta", {}).get("densify")],
        "no_densify": [r for r in rows if not r.get("meta", {}).get("densify")],
    }
    out = {"n_records": len(rows), "groups": {}}
    for gname, grp in groups.items():
        if not grp:
            continue
        entry = {"n": len(grp)}
        for ph in all_phases:
            entry[ph] = _stat([r["phases_ms"].get(ph) for r in grp])
        entry["total"] = _stat([r.get("iter_ms") for r in grp])
        if any("iter_gpu_ms" in r for r in grp):
            entry["total_gpu"] = _stat([r.get("iter_gpu_ms") for r in grp])
        out["groups"][gname] = entry
    return out


def load_profile(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# comparison                                                                   #
# --------------------------------------------------------------------------- #
def compare_profiles(sgdense: Union[str, dict], vanilla: Union[str, dict],
                     phases: Sequence[str] = PHASES) -> dict:
    """Per-phase SGDense-vs-vanilla comparison for one group (default ``all``).

    Both inputs are profile dicts (or paths). Returns::

        {"group": ..., "rows": [{"phase", "vanilla_ms", "sgdense_ms",
                                 "delta_ms", "ratio"}...], "totals": {...}}

    The delta is ``sgdense - vanilla``: the extra cost our method adds.
    """
    sgd = load_profile(sgdense) if isinstance(sgdense, str) else sgdense
    van = load_profile(vanilla) if isinstance(vanilla, str) else vanilla
    return _compare_group(sgd, van, group="all", phases=phases)


def _mean(group_entry: Optional[dict], phase: str) -> Optional[float]:
    if not group_entry:
        return None
    return group_entry.get(phase, {}).get("mean_ms")


def _compare_group(sgd: dict, van: dict, group: str, phases: Sequence[str]) -> dict:
    sg = sgd.get("summary", {}).get("groups", {}).get(group, {})
    vg = van.get("summary", {}).get("groups", {}).get(group, {})
    rows: List[dict] = []
    for ph in list(phases) + ["misc", "total"]:
        s = _mean(sg, ph)
        v = _mean(vg, ph)
        if s is None and v is None:
            continue
        delta = None if (s is None or v is None) else s - v
        ratio = None if (s is None or v in (None, 0.0)) else s / v
        rows.append({"phase": ph, "vanilla_ms": v, "sgdense_ms": s,
                     "delta_ms": delta, "ratio": ratio})
    return {
        "group": group,
        "n_sgdense": sg.get("n"),
        "n_vanilla": vg.get("n"),
        "rows": rows,
    }


def _fmt(v: Optional[float], nd: int = 2) -> str:
    return "-" if v is None else f"{v:.{nd}f}"


def format_comparison(cmp: dict, title: Optional[str] = None) -> str:
    """Render a :func:`compare_profiles` dict as a plain-text table."""
    lines: List[str] = []
    head = title or f"Per-iteration phase cost ({cmp['group']} iterations)"
    lines.append(head)
    lines.append(f"  vanilla n={cmp.get('n_vanilla')}  |  sgdense n={cmp.get('n_sgdense')}")
    lines.append("")
    lines.append(f"  {'phase':<10} {'vanilla(ms)':>12} {'sgdense(ms)':>12} "
                 f"{'delta(ms)':>10} {'ratio':>7} {'delta%':>8}")
    lines.append("  " + "-" * 63)
    for r in cmp["rows"]:
        ratio = "-" if r["ratio"] is None else f"{r['ratio']:.2f}x"
        pct = "-"
        if r["delta_ms"] is not None and r["vanilla_ms"] not in (None, 0.0):
            pct = f"{100.0 * r['delta_ms'] / r['vanilla_ms']:.0f}%"
        lines.append(f"  {r['phase']:<10} {_fmt(r['vanilla_ms']):>12} "
                     f"{_fmt(r['sgdense_ms']):>12} {_fmt(r['delta_ms']):>10} "
                     f"{ratio:>7} {pct:>8}")
    return "\n".join(lines)
