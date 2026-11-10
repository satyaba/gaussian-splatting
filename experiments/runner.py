"""Prepare and launch a :class:`Method` (see ``experiments/methods.py``).

Usage (Colab / Linux, from the SGDense-3DGS repo root)::

    python -m experiments.runner --method vanilla_3dgs \
        --run-id runB1 \
        --source "$DATASET_DIR" \
        --model-dir "$HOME_DIR/output/runB1" \
        --iterations 30000

    python -m experiments.runner --method sgdense --run-id sgdense_ext \
        --iterations 30000 \
        --extra "--segmentation_path $SEG_DIR --num_segmentation_classes 99 --floater_action freeze"

Or import it in a notebook cell::

    from experiments import run_method
    run_method("vanilla_3dgs", run_id="runB1", iterations=30000)

``--dry-run`` prints exactly what would run without touching the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from experiments.methods import Method, METHODS, get_method

REPO_ROOT = Path(__file__).resolve().parent.parent

# Only these submodules are needed for training; explicit paths keep a
# `--recursive` update from cloning the huge SIBR_viewers submodule.
TRAIN_SUBMODULES = ("submodules/diff-gaussian-rasterization", "submodules/simple-knn")

# Marker recording which method's rasterizer is currently installed.
_RASTERIZER_MARKER = ".sgdense_rasterizer_method"


# --------------------------------------------------------------------------- #
# paths                                                                        #
# --------------------------------------------------------------------------- #
def workdir_for(method: Method, workdir_root: str | Path = "/content") -> Path:
    """Directory the method's code lives in (this repo for local methods)."""
    if method.repo is None:
        return REPO_ROOT
    return Path(workdir_root) / (method.dir_name or method.id)


# --------------------------------------------------------------------------- #
# shell helpers                                                                #
# --------------------------------------------------------------------------- #
def _log(msg: str) -> None:
    print(f"[experiments] {msg}", flush=True)


def _sh(cmd: str, cwd: Path, dry: bool, tail: int = 40, check: bool = True) -> None:
    """Run a shell command, capturing output so failures are visible.

    On non-zero exit the last ``tail`` lines are printed before raising, so the
    real cause (e.g. a missing ``glm`` header) is not swallowed.
    """
    _log(f"$ (cd {cwd}) {cmd}")
    if dry:
        return
    p = subprocess.run(cmd, cwd=str(cwd), shell=True, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        tail_txt = "\n".join(out.splitlines()[-tail:])
        print(f"[experiments] command failed (exit {p.returncode}); last {tail} lines:\n"
              f"{tail_txt}", flush=True)
        if check:
            raise subprocess.CalledProcessError(p.returncode, cmd, output=p.stdout, stderr=p.stderr)
        return
    lines = out.splitlines()
    if lines:  # show progress tail on success too
        print("\n".join(lines[-tail:]), flush=True)


def _diagnose_rasterizer_build(wd: Path, dry: bool) -> None:
    """Rebuild the rasterizer via setup.py so the compiler error is visible.

    pip reports a failed CUDA-extension build as the unhelpful *"No available
    output."*; invoking the build backend directly prints the real nvcc/gcc/ninja
    error instead.
    """
    raster = wd / "submodules" / "diff-gaussian-rasterization"
    if not (raster / "setup.py").exists():
        return
    print("[experiments] install failed — rebuilding directly to surface the real "
          "compiler error ...", flush=True)
    _sh("python setup.py build_ext --inplace 2>&1 | tail -120",
        cwd=raster, dry=dry, tail=120, check=False)


def _argv(cmd: Sequence[str], cwd: Path, dry: bool) -> None:
    _log("$ (cd {} ) {}".format(cwd, " ".join(shlex.quote(c) for c in cmd)))
    if dry:
        return
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def _write_train_time(model_dir: str | Path, seconds: float, cmd: Sequence[str]) -> None:
    """Persist the training wall-clock next to the model for metrics extraction."""
    model_dir = Path(model_dir)
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "train_seconds": float(seconds),
            "train_minutes": float(seconds) / 60.0,
            "command": " ".join(cmd),
        }
        (model_dir / "train_time.json").write_text(json.dumps(payload, indent=1))
        _log(f"train time: {seconds / 60:.2f} min -> {model_dir / 'train_time.json'}")
    except OSError as e:  # non-fatal: timing is best-effort
        _log(f"warning: could not write train_time.json: {e}")


# --------------------------------------------------------------------------- #
# prepare                                                                      #
# --------------------------------------------------------------------------- #
def prepare(method: Method, workdir_root: str | Path = "/content",
            force: bool = False, dry: bool = False) -> Path:
    """Clone/pin the codebase and install its rasterizer.

    Idempotent: the rasterizer is rebuilt only when the currently-installed one
    belongs to a different method (or ``force`` is set).
    """
    workdir_root = Path(workdir_root)
    wd = workdir_for(method, workdir_root)

    if method.repo is not None:
        if not (wd / ".git").exists():
            wd.parent.mkdir(parents=True, exist_ok=True)
            _sh(f"git clone {method.repo} {shlex.quote(str(wd))}", cwd=wd.parent, dry=dry)
        if method.ref:
            _sh(f"git checkout {method.ref}", cwd=wd, dry=dry)
        submods = " ".join(TRAIN_SUBMODULES)
        _sh(f"git submodule update --init --recursive {submods}", cwd=wd, dry=dry)
        # Force the rasterizer's nested submodules (third_party/glm) — the stock
        # setup.py hard-includes third_party/glm/, so a missing glm/ is the most
        # common cause of a failed `pip install ./submodules/diff-gaussian-rasterization`.
        raster = wd / "submodules" / "diff-gaussian-rasterization"
        if (raster / ".git").exists() or (raster / ".gitmodules").exists():
            _sh("git submodule update --init --recursive", cwd=raster, dry=dry)
            # Optional pin to a different rasterizer branch/commit (applied AFTER
            # the submodule update, which would otherwise reset it to the recorded
            # SHA). Use when the stock commit does not build on this toolchain.
            if method.rasterizer_ref:
                _log(f"rasterizer_ref: checkout {method.rasterizer_ref}")
                _sh(f"git checkout {method.rasterizer_ref}", cwd=raster, dry=dry)

    marker = workdir_root / _RASTERIZER_MARKER
    last = marker.read_text().strip() if marker.exists() else None
    if force or last != method.id:
        if last is not None:
            _log(f"rasterizer swap: {last} -> {method.id}")
        for cmd in method.install:
            try:
                _sh(cmd, cwd=wd, dry=dry)
            except subprocess.CalledProcessError:
                _diagnose_rasterizer_build(wd, dry)
                raise
        if not dry:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(method.id)
    else:
        _log(f"rasterizer already matches method {method.id!r}; skipping install "
             f"(use force=True to rebuild)")

    return wd


# --------------------------------------------------------------------------- #
# train command                                                                #
# --------------------------------------------------------------------------- #
def build_train_cmd(method: Method, source: str | Path, model_dir: str | Path,
                    iterations: Optional[int] = None,
                    images: Optional[str] = None,
                    extra: Sequence[str] = ()) -> List[str]:
    cmd = [sys.executable, method.entry, "-s", str(source), "-m", str(model_dir)]
    if images:
        cmd += ["-i", images]
    cmd += list(method.common_args)
    if iterations:
        cmd += ["--iterations", str(iterations)]
    cmd += list(extra)
    return cmd


# --------------------------------------------------------------------------- #
# top-level                                                                    #
# --------------------------------------------------------------------------- #
def run_method(method_id: str, *, source: str | Path, model_dir: str | Path,
               workdir_root: str | Path = "/content",
               iterations: Optional[int] = None, images: Optional[str] = None,
               extra: Sequence[str] = (), force: bool = False,
               skip_prepare: bool = False, dry: bool = False,
               run_id: Optional[str] = None) -> Path:
    """Prepare ``method_id`` and run its training. Returns the model dir."""
    method = get_method(method_id)
    model_dir = Path(model_dir)
    if run_id and not model_dir:
        model_dir = Path(os.environ.get("HOME_DIR", ".")) / "output" / run_id

    if skip_prepare:
        wd = workdir_for(method, workdir_root)
    else:
        wd = prepare(method, workdir_root, force=force, dry=dry)

    cmd = build_train_cmd(method, source, model_dir, iterations, images, extra)
    _log(f"method={method.id} | {method.label}")
    if method.notes:
        _log(f"note: {method.notes}")
    t0 = time.perf_counter()
    _argv(cmd, cwd=wd, dry=dry)
    if not dry:
        _write_train_time(model_dir, time.perf_counter() - t0, cmd)
    return model_dir


def _split_extra(values: Optional[Iterable[str]]) -> List[str]:
    """Allow --extra to be either repeated tokens or a single quoted string."""
    if not values:
        return []
    out: List[str] = []
    for v in values:
        out.extend(shlex.split(v))
    return out


def _default_model_dir(args) -> Optional[str]:
    if args.model_dir:
        return args.model_dir
    run_id = args.run_id or os.environ.get("RUN_ID")
    home = os.environ.get("HOME_DIR")
    if run_id and home:
        return str(Path(home) / "output" / run_id)
    return os.environ.get("OUT_DIR")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Prepare + train a method (local or external).")
    p.add_argument("--method", required=True, choices=sorted(METHODS))
    p.add_argument("--run-id", default=None, help="run id (defaults to $RUN_ID)")
    p.add_argument("--source", "-s", default=os.environ.get("DATASET_DIR"),
                   help="COLMAP/NeRF source dir (defaults to $DATASET_DIR)")
    p.add_argument("--model-dir", "-m", default=None,
                   help="output dir (defaults to $HOME_DIR/output/$RUN_ID)")
    p.add_argument("--workdir-root", default="/content",
                   help="where external repos are cloned (default /content)")
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--images", "-i", default=None)
    p.add_argument("--extra", nargs="+", default=None,
                   help="extra train args; quote as one string or list tokens")
    p.add_argument("--force", action="store_true", help="rebuild the rasterizer")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    source = args.source
    model_dir = _default_model_dir(args)
    if not source:
        p.error("no --source and $DATASET_DIR is unset")
    if not model_dir:
        p.error("no --model-dir and could not derive one from $HOME_DIR/$RUN_ID")

    run_method(
        args.method,
        source=source,
        model_dir=model_dir,
        workdir_root=args.workdir_root,
        iterations=args.iterations,
        images=args.images,
        extra=_split_extra(args.extra),
        force=args.force,
        skip_prepare=args.skip_prepare,
        dry=args.dry_run,
        run_id=args.run_id,
    )
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
