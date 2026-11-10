"""Declarative manifest of the runs to execute.

One place that says *what* each run is (method, iterations, dataset, extra
args); :mod:`experiments.runner` says *how* to prepare and launch it. Add a
:class:`RunSpec` to :data:`RUNS` and it becomes runnable via the CLI or
``run_all()`` from a notebook.

    python -m experiments.runs --list
    python -m experiments.runs --run sgdense_30k
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
from typing import Dict, List, Optional, Sequence, Tuple, Union

from experiments.runner import run_method, _split_extra

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
# best method = SGDense (L1+L2 floaters with FLOATER_ACTION='freeze').
#
# SGDense standard (after the Option-A decision): the clone/split trigger follows
# the PHOTOMETRIC (+depth) gradient only (--densify_photo_grad_only, now the
# default) and the densification window runs to 25k (--densify_until_iter, now
# the default). The legacy sgdense_30k / sgdense_photograd_30k specs below are
# PINNED to their historical settings (total-gradient trigger, 15k window) so they
# keep reproducing the reported baselines. The `sgdense_photo25k_*` grid turns on
# the standard plus the existing texture/position densification routes.
RUNS: Dict[str, RunSpec] = {
    r.run_id: r
    for r in (
        RunSpec(
            run_id="vanilla_30k",
            method="vanilla_3dgs",
            iterations=30_000,
            notes="Upstream graphdeco 3DGS (stock rasterizer), --eval split.",
        ),
        # --- legacy SGDense baselines: pinned to pre-Option-A settings ---
        RunSpec(
            run_id="sgdense_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--no_densify_photo_grad_only", "--densify_until_iter", "15000"),
            notes="LEGACY: floaters frozen; total-gradient trigger, 15k window (pre-Option-A).",
        ),
        RunSpec(
            run_id="sgdense_photograd_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "15000"),
            notes="LEGACY: Option A (photo-only trigger) at the original 15k window.",
        ),
        RunSpec(
            run_id="sgdense_photograd_tex_scale_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                    "--densify_photo_grad_only", "--densify_until_iter", "15000",
                    "--texture_based_scale", "--texture_scale_beta", "1.0"),
            notes="Photo-only gradient trigger + texture-based per-class size thresholds (Beta=1.0).",
        ),
        RunSpec(
            run_id="sgdense_photograd_tex_grad_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                    "--densify_photo_grad_only", "--densify_until_iter", "15000",
                   "--texture_based_thresh", "--texture_beta", "-0.25"),
            notes="Photo-only gradient trigger + texture-based per-class gradient thresholds (Beta=-0.25).",
        ),
        RunSpec(
            run_id="sgdense_photograd_pos_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                    "--densify_photo_grad_only", "--densify_until_iter", "15000",
                    "--position_based_thresh", 
                    "--interior_weight", "0.5",
                    "--boundary_weight", "0.3",
                    "--other_weight", "1.0"),
            notes="Photo-only gradient trigger + Position-based scale weight (Bound=0.3, Inter=0.5, Other=1.0).",
        ),
        # --- SGDense standard: photo-only trigger + 25k window; route grid. ---
        RunSpec(
            run_id="sgdense_photo25k_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "25000"),
            notes="Standard: photo-only trigger + 25k window, no texture/position routes.",
        ),
        RunSpec(
            run_id="sgdense_photo25k_pos_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                    "--densify_photo_grad_only", "--densify_until_iter", "25000",
                    "--position_based_thresh", 
                    "--interior_weight", "0.5",
                    "--boundary_weight", "0.3",
                    "--other_weight", "1.0"),
            notes="Standard + position-based per-regime scaling thresholds.",
        ),
        RunSpec(
            run_id="sgdense_photo25k_tex_scale_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "25000",
                   "--texture_based_scale", "--texture_scale_beta", "1.0"),
            notes="Standard + texture-based per-class size thresholds (Beta=1.0).",
        ),
        RunSpec(
            run_id="sgdense_photo25k_tex_grad_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                    "--densify_photo_grad_only", "--densify_until_iter", "25000",
                    "--texture_based_thresh", "--texture_beta", "-0.25"),
            notes="Standard + texture-based per-class gradient thresholds (Beta=-0.25).",
        ),
        RunSpec(
            run_id="sgdense_photo25k_pos_tex_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                    "--densify_photo_grad_only", "--densify_until_iter", "25000",
                    "--position_based_thresh",
                    "--interior_weight", "0.5",
                    "--boundary_weight", "0.3",
                    "--other_weight", "1.0", 
                    "--texture_based_thresh",
                    "--texture_beta", "-0.25",
                    "--texture_based_scale",
                    "--texture_scale_beta", "1.0"),
            notes="Standard + both position and texture routes (combined candidate).",
        ),
        # --- 20k window: the same standard + route grid with a shorter densification
        # window; compare against the 25k (photo25k_*) and 15k legacy (photograd_*)
        # sets to isolate the window length. ---
        RunSpec(
            run_id="sgdense_photo20k_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "20000"),
            notes="20k window: photo-only trigger, no texture/position routes.",
        ),
        RunSpec(
            run_id="sgdense_photo20k_pos_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "20000",
                   "--position_based_thresh",
                   "--interior_weight", "0.5",
                   "--boundary_weight", "0.3",
                   "--other_weight", "1.0"),
            notes="20k window + position route (interior 0.5 / boundary 0.3 / other 1.0).",
        ),
        RunSpec(
            run_id="sgdense_photo20k_tex_scale_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "20000",
                   "--texture_based_scale", "--texture_scale_beta", "1.0"),
            notes="20k window + texture size route (Beta=1.0).",
        ),
        RunSpec(
            run_id="sgdense_photo20k_tex_grad_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "20000",
                   "--texture_based_thresh", "--texture_beta", "-0.25"),
            notes="20k window + texture gradient route (Beta=-0.25).",
        ),
        RunSpec(
            run_id="sgdense_photo20k_pos_tex_30k",
            method="sgdense",
            iterations=30_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "20000",
                   "--position_based_thresh",
                   "--interior_weight", "0.5",
                   "--boundary_weight", "0.3",
                   "--other_weight", "1.0",
                   "--texture_based_thresh",
                   "--texture_beta", "-0.25",
                   "--texture_based_scale",
                   "--texture_scale_beta", "1.0"),
            notes="20k window + both routes (combined candidate).",
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
            run_id="sgdense_sanity",
            method="sgdense",
            iterations=3_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--no_densify_photo_grad_only", "--densify_until_iter", "15000"),
            sanity=True,
            notes="SANITY: LEGACY SGDense config @3k (total-gradient trigger).",
        ),
        RunSpec(
            run_id="sgdense_photograd_sanity",
            method="sgdense",
            iterations=3_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "15000"),
            sanity=True,
            notes="SANITY: LEGACY Option A (photo-only densify) @3k.",
        ),
        RunSpec(
            run_id="sgdense_photo25k_sanity",
            method="sgdense",
            iterations=3_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "25000"),
            sanity=True,
            notes="SANITY: standard (photo-only + 25k) @3k.",
        ),
        RunSpec(
            run_id="sgdense_photo25k_routes_sanity",
            method="sgdense",
            iterations=3_000,
            seg_num_classes=99,
            extra=("--floater_action", "freeze", "--log_densify_events", "--log_grad_stats",
                   "--densify_photo_grad_only", "--densify_until_iter", "25000",
                   "--position_based_thresh",
                   "--interior_weight", "0.5",
                   "--boundary_weight", "0.3",
                   "--other_weight", "1.0",
                   "--texture_based_thresh",
                   "--texture_beta", "-0.25",
                   "--texture_based_scale",
                   "--texture_scale_beta", "1.0"),
            sanity=True,
            notes="SANITY: standard + both routes with the tuned weights/betas @3k.",
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


def _model_dir(spec: RunSpec, home_dir: Optional[str], tag: Optional[str] = None) -> Path:
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass home_dir")
    name = spec.run_id if not tag else f"{spec.run_id}_{tag}"
    return Path(home) / "output" / name


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
        skip_prepare: bool = False, dry: bool = False,
        extra: Union[str, Sequence[str], None] = None,
        tag: Optional[str] = None, model_dir: Optional[str] = None) -> Path:
    """Launch one run.

    ``extra`` adds CLI tokens on top of the spec's own ``extra`` — e.g. to
    override a hyperparameter per call from a notebook::

        run_by_id("sgdense_photo20k_pos_30k",
                  extra="--boundary_weight 0.2 --interior_weight 0.4")

    Accepts a single shell-quoted string or an iterable of tokens. Tokens are
    appended AFTER the spec's args, so argparse's last-wins gives them override
    precedence.

    ``tag`` gives this call its own output directory
    (``<home>/output/<run_id>_<tag>``) so a sweep can run the same spec many
    times without clobbering results; ``model_dir`` overrides the path entirely.
    """
    source = resolve_source(spec, home_dir, dataset_root)
    if model_dir is None:
        model_dir = _model_dir(spec, home_dir, tag=tag)
    effective_id = spec.run_id if not tag else f"{spec.run_id}_{tag}"
    overrides = _split_extra([extra] if isinstance(extra, str) else extra)
    print(f"[runs] {effective_id}: method={spec.method} iters={spec.iterations} "
          f"source={source}")
    if spec.notes:
        print(f"[runs]   note: {spec.notes}")
    if overrides:
        print(f"[runs]   overrides: {' '.join(overrides)}")
    return run_method(
        spec.method,
        source=source,
        model_dir=model_dir,
        workdir_root=workdir_root,
        iterations=spec.iterations,
        images=spec.images,
        extra=_extra_args(spec, source) + overrides,
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
