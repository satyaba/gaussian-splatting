"""Metric extraction: turn a trained model into a row of the results sheet.

Reproduces what ``SGDense_3DGS_Experiment.ipynb`` reported: for each split
(train / test) the mean **PSNR / SSIM / LPIPS** over the split's views, plus the
final **Gaussian count**. Computation is delegated to ``eval_sgdense.py`` (the
on-the-fly renderer), which is the same metric code path for every method, so
baseline and SGDense numbers are directly comparable.

The evaluator renders with this repo's seg-aware rasterizer (the RGB path is the
stock 3DGS rasterization; the seg channels are additive), so it can score both
SGDense and upstream-graphdeco PLYs. It therefore ensures the ``sgdense``
rasterizer is installed before evaluating.

    python -m experiments.metrics --run-id run38_30k
    python -m experiments.metrics --all
    python -m experiments.metrics --model-dir <path> --name "my run" --iteration 30000
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from experiments.methods import get_method
from experiments.runner import REPO_ROOT, prepare

# Column order mirrors the experiment spreadsheet.
CSV_COLUMNS = [
    "Name", "Run ID", "Iteration",
    "Train PSNR", "Train SSIM", "Train LPIPS",
    "Test PSNR", "Test SSIM", "Test LPIPS",
    "Gauss. count",
    "Train time (min)", "Eval time (s)",
]


def _read_train_seconds(model_dir: str | Path) -> Optional[float]:
    """Training wall-clock written by the runner (see runner._write_train_time)."""
    p = Path(model_dir) / "train_time.json"
    if not p.exists():
        return None
    try:
        return float(json.loads(p.read_text())["train_seconds"])
    except (OSError, ValueError, KeyError):
        return None


# --------------------------------------------------------------------------- #
# row shaping                                                                  #
# --------------------------------------------------------------------------- #
def _round(v, ndigits):
    return None if v is None else round(float(v), ndigits)


def to_row(data: dict, run_id: Optional[str] = None, name: Optional[str] = None) -> dict:
    tr = data.get("splits", {}).get("train", {})
    te = data.get("splits", {}).get("test", {})
    train_s = data.get("train_seconds")
    eval_s = data.get("eval_seconds")
    return {
        "Name": name or "",
        "Run ID": run_id or Path(data.get("run_dir", "")).name,
        "Iteration": data.get("iteration"),
        "Train PSNR": _round(tr.get("psnr"), 2),
        "Train SSIM": _round(tr.get("ssim"), 4),
        "Train LPIPS": _round(tr.get("lpips"), 4),
        "Test PSNR": _round(te.get("psnr"), 2),
        "Test SSIM": _round(te.get("ssim"), 4),
        "Test LPIPS": _round(te.get("lpips"), 4),
        "Gauss. count": data.get("gaussians"),
        "Train time (min)": _round(train_s / 60.0, 2) if train_s else "",
        "Eval time (s)": _round(eval_s, 1) if eval_s else "",
    }


def append_row(row: dict, csv_path: str | Path) -> None:
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# extraction                                                                   #
# --------------------------------------------------------------------------- #
def extract_one(model_dir: str | Path, *, iteration: int = -1,
                run_id: Optional[str] = None, name: Optional[str] = None,
                csv_path: Optional[str | Path] = None,
                workdir_root: str = "/content", skip_prepare: bool = False,
                dry: bool = False) -> Optional[dict]:
    """Evaluate ``model_dir`` and (optionally) append a CSV row. Returns the row."""
    model_dir = Path(model_dir)
    if not skip_prepare:
        prepare(get_method("sgdense"), workdir_root, dry=dry)

    tmp = Path(tempfile.gettempdir()) / f"sgdense_eval_{os.getpid()}.json"
    cmd = [sys.executable, "eval_sgdense.py", "-m", str(model_dir),
           "--iteration", str(iteration), "--json_out", str(tmp)]
    print(f"[metrics] eval: {' '.join(cmd)}")
    if dry:
        return None
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)
    eval_seconds = time.perf_counter() - t0

    data = json.loads(tmp.read_text())
    data["eval_seconds"] = eval_seconds
    data["train_seconds"] = _read_train_seconds(model_dir)
    row = to_row(data, run_id=run_id, name=name)
    print(f"[metrics] {row}")
    if csv_path:
        append_row(row, csv_path)
        print(f"[metrics] appended -> {csv_path}")
    return row


def extract_run(run_id: str, *, csv_path: Optional[str | Path] = None,
                home_dir: Optional[str] = None, workdir_root: str = "/content",
                skip_prepare: bool = False, dry: bool = False) -> Optional[dict]:
    from experiments.runs import RUNS, _model_dir  # local import: avoid cycle

    if run_id not in RUNS:
        raise KeyError(f"unknown run {run_id!r}; known: {sorted(RUNS)}")
    spec = RUNS[run_id]
    return extract_one(
        _model_dir(spec, home_dir), iteration=-1, run_id=spec.run_id,
        name=spec.notes or spec.method, csv_path=csv_path,
        workdir_root=workdir_root, skip_prepare=skip_prepare, dry=dry,
    )


def extract_all(*, csv_path: Optional[str | Path] = None,
                home_dir: Optional[str] = None, workdir_root: str = "/content",
                skip_prepare: bool = False, dry: bool = False) -> List[Optional[dict]]:
    from experiments.runs import RUNS
    return [extract_run(rid, csv_path=csv_path, home_dir=home_dir,
                        workdir_root=workdir_root, skip_prepare=skip_prepare, dry=dry)
            for rid in RUNS]


def _default_csv() -> Optional[str]:
    home = os.environ.get("HOME_DIR")
    return str(Path(home) / "output" / "metrics.csv") if home else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Extract metrics into a results CSV.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id", help="a run id from experiments.runs")
    g.add_argument("--all", action="store_true", help="every run in the manifest")
    g.add_argument("--model-dir", help="evaluate an arbitrary model dir")
    p.add_argument("--iteration", type=int, default=-1)
    p.add_argument("--name", default=None)
    p.add_argument("--csv", default=_default_csv(),
                   help="results CSV to append to (default $HOME_DIR/output/metrics.csv)")
    p.add_argument("--no-csv", action="store_true", help="print only, do not write CSV")
    p.add_argument("--home-dir", default=os.environ.get("HOME_DIR"))
    p.add_argument("--workdir-root", default="/content")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    csv_path = None if args.no_csv else args.csv
    kw = dict(csv_path=csv_path, home_dir=args.home_dir,
              workdir_root=args.workdir_root, skip_prepare=args.skip_prepare,
              dry=args.dry_run)
    if args.all:
        extract_all(**kw)
    elif args.run_id:
        extract_run(args.run_id, **kw)
    else:
        extract_one(args.model_dir, iteration=args.iteration, name=args.name,
                    run_id=args.model_dir.rstrip("/\\").split(os.sep)[-1],
                    csv_path=csv_path, workdir_root=args.workdir_root,
                    skip_prepare=args.skip_prepare, dry=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
