"""Declarative manifest of the runs to execute.

One place that says *what* each run is (method, iterations, dataset, extra
args); :mod:`experiments.runner` says *how* to prepare and launch it. Add a
:class:`RunSpec` to :data:`RUNS` and it becomes runnable via the CLI or
``run_all()`` from a notebook.

    python -m experiments.runs --list
    python -m experiments.runs --run sgdc_a_30k
    python -m experiments.runs --all

Dataset layout assumed under ``$HOME_DIR/Dataset/<dataset>``::

    <dataset>/images          # RGB (rgb_<N>.png for Replica)
    <dataset>/sparse/0/*.bin  # COLMAP
    <dataset>/semantic_class  # uint8 class-id PNGs (SGDC runs only)

The SGDC Replica adapter reads ``<source>/semantic_class`` automatically, so a
run only needs ``--sgdc_enable`` (plus its hyperparameters) -- no path/number
arguments.

Two independent SGDC components are represented here (see ``docs/sgdc_spec.md``):

* **A** -- boundary-aware density control: scale the clone/split gradient
  threshold per Gaussian by its distance to a semantic boundary
  (``--f_b`` boundary / ``--f_int`` interior / ``--sgdc_dt_norm``).
* **C** -- cross-boundary scale regularizer: an extra loss that reshapes
  boundary Gaussians (``--lambda_bnd``). The C runs neutralise A
  (``--f_b 1.0 --f_int 1.0 --eta 0.0``) so only C acts.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from experiments.runner import run_method, _split_extra

DEFAULT_DATASET = "office_0/Sequence_1"
DEFAULT_ITERATIONS = 30_000


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    iterations: int = DEFAULT_ITERATIONS
    dataset: str = DEFAULT_DATASET
    images: Optional[str] = None
    extra: Tuple[str, ...] = ()
    sanity: bool = False                    # short smoke test (not a result run)
    notes: str = ""


# --------------------------------------------------------------------------- #
# RUNS                                                                         #
# --------------------------------------------------------------------------- #
# Baseline = upstream graphdeco 3DGS (vanilla_3dgs) and the same SGDC code with
# SGDC disabled (base_*). All runs use the shared `--eval` split (llffhold=8).
#
# `sgdc_a_*` are Component-A settings; the sweep lowers `sgdc_dt_norm` (classify
# interiors as "far") and raises `f_int` (suppress interior densification)
# while keeping boundaries aggressive. `sgdc_c_*` hold A neutral so only the
# cross-boundary regularizer acts.
RUNS: Dict[str, RunSpec] = {
    r.run_id: r
    for r in (
        # --- baselines ---
        RunSpec(
            run_id="vanilla_30k",
            method="vanilla_3dgs",
            iterations=30_000,
            notes="Upstream graphdeco 3DGS (stock rasterizer), --eval split.",
        ),
        RunSpec(
            run_id="base_30k",
            method="sgdc",
            iterations=30_000,
            notes="This checkout with SGDC disabled (plain 3DGS, --eval split).",
        ),
        # --- Component A: boundary-aware density control ---
        RunSpec(
            run_id="sgdc_a_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "0.5", "--f_int", "2.0", "--eta", "0.3"),
            notes="A default: boundary boost (f_b 0.5), interior suppression (f_int 2.0), eta 0.3.",
        ),
        RunSpec(
            run_id="sgdc_a_fb1_fi3_dt60_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "1.0", "--f_int", "3.0",
                   "--eta", "0.3", "--sgdc_dt_norm", "60"),
            notes="A: boundary = baseline (f_b 1.0), interior x3, dt_norm 60.",
        ),
        RunSpec(
            run_id="sgdc_a_fb05_fi4_dt50_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "0.5", "--f_int", "4.0",
                   "--eta", "0.3", "--sgdc_dt_norm", "50"),
            notes="A: keep boundary boost (f_b 0.5), strong interior suppression (f_int 4.0), dt_norm 50.",
        ),
        RunSpec(
            run_id="sgdc_a_fb1_fi6_dt40_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "1.0", "--f_int", "6.0",
                   "--eta", "0.3", "--sgdc_dt_norm", "40"),
            notes="A: aggressive interior suppression (f_b 1.0 / f_int 6.0), dt_norm 40.",
        ),
        # --- Component C: cross-boundary scale regularizer (A neutralised) ---
        RunSpec(
            run_id="sgdc_c_bnd02_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "1.0", "--f_int", "1.0", "--eta", "0.0",
                   "--lambda_bnd", "0.02"),
            notes="C: cross-boundary scale loss, lambda_bnd 0.02 (A neutral).",
        ),
        RunSpec(
            run_id="sgdc_c_bnd10_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "1.0", "--f_int", "1.0", "--eta", "0.0",
                   "--lambda_bnd", "0.10"),
            notes="C: cross-boundary scale loss, lambda_bnd 0.10 (A neutral).",
        ),
        RunSpec(
            run_id="sgdc_c_bnd50_30k",
            method="sgdc",
            iterations=30_000,
            extra=("--sgdc_enable", "--f_b", "1.0", "--f_int", "1.0", "--eta", "0.0",
                   "--lambda_bnd", "0.50"),
            notes="C: cross-boundary scale loss, lambda_bnd 0.50 (A neutral).",
        ),
        # --- sanity runs: same configs, 3_000 iters. Enough to cross the first
        # densification (500), hit the opacity reset (3000) and save. Use to
        # verify the whole pipeline (train -> rasterizer swap -> eval). ---
        RunSpec(
            run_id="vanilla_sanity",
            method="vanilla_3dgs",
            iterations=3_000,
            sanity=True,
            notes="SANITY: upstream 3DGS @3k.",
        ),
        RunSpec(
            run_id="base_sanity",
            method="sgdc",
            iterations=3_000,
            sanity=True,
            notes="SANITY: this checkout, SGDC off @3k.",
        ),
        RunSpec(
            run_id="sgdc_a_sanity",
            method="sgdc",
            iterations=3_000,
            extra=("--sgdc_enable", "--f_b", "0.5", "--f_int", "2.0", "--eta", "0.3"),
            sanity=True,
            notes="SANITY: Component A default @3k.",
        ),
        RunSpec(
            run_id="sgdc_c_sanity",
            method="sgdc",
            iterations=3_000,
            extra=("--sgdc_enable", "--f_b", "1.0", "--f_int", "1.0", "--eta", "0.0",
                   "--lambda_bnd", "0.10"),
            sanity=True,
            notes="SANITY: Component C (lambda_bnd 0.10) @3k.",
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
    return list(spec.extra)


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

        run_by_id("sgdc_a_30k", extra="--f_b 0.3 --f_int 5.0")

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
            print(f"[{tag:6s}] {r.run_id:26s} method={r.method:13s} iters={r.iterations:<6d} "
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
