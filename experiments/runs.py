"""Declarative manifest of the runs to execute.

One place that says *what* each run is (method, iterations, dataset, extra
args); :mod:`experiments.runner` says *how* to prepare and launch it. Add a
:class:`RunSpec` to :data:`RUNS` and it becomes runnable via the CLI or
``run_all()`` from a notebook.

    python -m experiments.runs --list
    python -m experiments.runs --run run38_30k
    python -m experiments.runs --all

Dataset layout assumed under ``$HOME_DIR/Dataset/<dataset>``::

    <dataset>/images          # RGB
    <dataset>/sparse/0/*.bin  # COLMAP
    <dataset>/semantic_class  # uint8 class-id PNGs (SGDense runs only)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from experiments.runner import run_method

DEFAULT_DATASET = "office_0/Sequence_1"
DEFAULT_ITERATIONS = 30_000
SEG_DIRNAME = "semantic_class"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    iterations: int = DEFAULT_ITERATIONS
    dataset: str = DEFAULT_DATASET
    images: Optional[str] = None
    seg_num_classes: Optional[int] = None   # set for SGDense runs (adds seg args)
    extra: Tuple[str, ...] = ()
    sanity: bool = False                    # short smoke test (not a result run)
    notes: str = ""


# --------------------------------------------------------------------------- #
# RUNS                                                                         #
# --------------------------------------------------------------------------- #
# 30k runs (extending the 12k screening). Baseline = upstream graphdeco 3DGS;
# best method = SGDense run38 (L1+L2 floaters with FLOATER_ACTION='freeze').
RUNS: Dict[str, RunSpec] = {
    r.run_id: r
    for r in (
        RunSpec(
            run_id="vanilla_30k",
            method="vanilla_3dgs",
            iterations=30_000,
            notes="Upstream graphdeco 3DGS (stock rasterizer), --eval split.",
        ),
        RunSpec(
            run_id="run38_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats"),
            notes="Level-2 floaters, frozen (L1+L2 on, no texture/position routes).",
        ),
        # --- sanity runs: same configs, 3_000 iters. Enough to cross the seg
        # warmup (2000), run densification + floater/freeze, hit the opacity
        # reset (3000) and save. Use to verify the pipeline runs end-to-end. ---
        RunSpec(
            run_id="vanilla_sanity",
            method="vanilla_3dgs",
            iterations=3_000,
            sanity=True,
            notes="SANITY: upstream 3DGS @3k.",
        ),
        RunSpec(
            run_id="run38_sanity",
            method="sgdense",
            iterations=3_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats"),
            sanity=True,
            notes="SANITY: SGDense run38 @3k (exercises warmup+densify+freeze+reset).",
        ),
    )
}


def full_runs() -> List[RunSpec]:
    return [r for r in RUNS.values() if not r.sanity]


def sanity_runs() -> List[RunSpec]:
    return [r for r in RUNS.values() if r.sanity]


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def resolve_source(spec: RunSpec, home_dir: Optional[str] = None,
                   dataset_root: Optional[str] = None) -> Path:
    dataset_root = dataset_root or os.environ.get("DATASET_ROOT")
    if dataset_root:
        return Path(dataset_root) / spec.dataset
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass dataset_root=/home_dir=")
    return Path(home) / "Dataset" / spec.dataset


def _model_dir(spec: RunSpec, home_dir: Optional[str]) -> Path:
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass home_dir")
    return Path(home) / "output" / spec.run_id


def _extra_args(spec: RunSpec, source: Path) -> List[str]:
    args = list(spec.extra)
    if spec.seg_num_classes is not None:
        args += ["--segmentation_path", str(source / SEG_DIRNAME),
                 "--num_segmentation_classes", str(spec.seg_num_classes)]
    return args


# --------------------------------------------------------------------------- #
# execution                                                                    #
# --------------------------------------------------------------------------- #
def run(spec: RunSpec, home_dir: Optional[str] = None, dataset_root: Optional[str] = None,
        workdir_root: str = "/content", force: bool = False,
        skip_prepare: bool = False, dry: bool = False) -> Path:
    source = resolve_source(spec, home_dir, dataset_root)
    model_dir = _model_dir(spec, home_dir)
    print(f"[runs] {spec.run_id}: method={spec.method} iters={spec.iterations} "
          f"source={source}")
    if spec.notes:
        print(f"[runs]   note: {spec.notes}")
    return run_method(
        spec.method,
        source=source,
        model_dir=model_dir,
        workdir_root=workdir_root,
        iterations=spec.iterations,
        images=spec.images,
        extra=_extra_args(spec, source),
        force=force,
        skip_prepare=skip_prepare,
        dry=dry,
    )


def run_by_id(run_id: str, **kw) -> Path:
    try:
        spec = RUNS[run_id]
    except KeyError:
        raise KeyError(f"unknown run {run_id!r}; known: {sorted(RUNS)}")
    return run(spec, **kw)


def run_all(**kw) -> None:
    """Run every non-sanity run in the manifest."""
    for spec in full_runs():
        run(spec, **kw)


def run_sanity(**kw) -> None:
    """Run the short sanity runs only."""
    for spec in sanity_runs():
        run(spec, **kw)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run the experiment manifest.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="list runs and exit")
    g.add_argument("--run", metavar="RUN_ID", help="run one run id")
    g.add_argument("--all", action="store_true", help="run every non-sanity run")
    g.add_argument("--sanity", action="store_true", help="run the short sanity runs only")
    p.add_argument("--home-dir", default=os.environ.get("HOME_DIR"))
    p.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT"),
                   help="override the root above per-run <dataset> folders")
    p.add_argument("--workdir-root", default="/content")
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        for r in RUNS.values():
            tag = "sanity" if r.sanity else "full"
            print(f"[{tag:6s}] {r.run_id:16s} method={r.method:12s} iters={r.iterations:<6d} "
                  f"dataset={r.dataset}  {r.notes}")
        return 0

    kw = dict(home_dir=args.home_dir, dataset_root=args.dataset_root,
              workdir_root=args.workdir_root, force=args.force,
              skip_prepare=args.skip_prepare, dry=args.dry_run)
    if args.sanity:
        run_sanity(**kw)
    elif args.all:
        run_all(**kw)
    else:
        run_by_id(args.run, **kw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
