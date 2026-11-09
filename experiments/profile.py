"""Per-iteration cost comparison: SGDense vs upstream vanilla 3DGS.

Runs both methods (reusing ``experiments.runner``) with the phase profiler
enabled for a chosen iteration window, then prints the per-phase
SGDense-vs-vanilla table (the extra cost our method adds, from render through
attribute optimisation).

    # run both + compare (rasterizer is swapped between the two automatically)
    python -m experiments.profile --iterations 4200 --profile-iters 3999:4099

    # compare two existing profiles only
    python -m experiments.profile --compare \
        --sgdense-json .../iter_profile.json --vanilla-json .../iter_profile.json

Requires a CUDA GPU + built rasterizers (Colab). ``--dry-run`` prints the two
commands without executing.

Env: ``$HOME_DIR`` / ``$DATASET_ROOT`` follow the same convention as
``experiments.runs``. Profile JSONs land in
``<model_dir>/iter_profile.json`` for both methods.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

from experiments.methods import get_method
from experiments.runner import (
    REPO_ROOT,
    _write_train_time,
    build_train_cmd,
    prepare,
)
from sgdense.profiler import compare_profiles, format_comparison

VANILLA_SHIM = REPO_ROOT / "scripts" / "vanilla_iter_profile.py"
VANILLA_CODE = ("import vanilla_iter_profile, runpy, sys; "
                "sys.argv = ['train.py'] + sys.argv[1:]; "
                "runpy.run_path('train.py', run_name='__main__')")


def _launch(method_id: str, *, source: str, model_dir: str, iterations: int,
            extra: Sequence[str], profile_iters: str, workdir_root: str,
            force: bool, dry: bool) -> Path:
    """Run one method with profiling on; return the model dir."""
    method = get_method(method_id)
    model_dir = Path(model_dir)
    wd = prepare(method, workdir_root, force=force, dry=dry)

    profile_out = str(model_dir / "iter_profile.json")
    base_cmd = build_train_cmd(method, source, model_dir, iterations, None, extra)

    env = dict(os.environ,
               SGDENSE_PROFILE_ITERS=profile_iters,
               SGDENSE_PROFILE_OUT=profile_out)

    if method.repo is None:
        # SGDense: train.py reads --profile_iterations / $SGDENSE_PROFILE_ITERS.
        cmd = base_cmd
    else:
        # Vanilla: copy the shim next to upstream train.py and run it wrapped.
        shim = Path(wd) / VANILLA_SHIM.name
        if not dry:
            shutil.copy(str(VANILLA_SHIM), str(shim))
        cmd = [sys.executable, "-c", VANILLA_CODE] + base_cmd[2:]
        env.update(SGDENSE_REPO=str(REPO_ROOT),
                   PYTHONPATH=os.pathsep.join(
                       [str(wd), str(REPO_ROOT)] +
                       [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]))

    print(f"[profile] {method_id}: source={source} model={model_dir}")
    print("[profile] $ " + " ".join(cmd))
    if dry:
        return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=str(wd), env=env, check=True)
    _write_train_time(model_dir, time.perf_counter() - t0, cmd)
    return model_dir


def profile_pair(*, source: str, home_dir: Optional[str], prefix: str,
                 iterations: int, profile_iters: str,
                 seg_num_classes: Optional[int] = None, seg_dir: Optional[str] = None,
                 extra_sgdense: Sequence[str] = (), workdir_root: str = "/content",
                 force: bool = False, dry: bool = False) -> dict:
    """Run both methods on the same data and return the comparison dict."""
    home = home_dir or os.environ.get("HOME_DIR")
    if not home:
        raise RuntimeError("set $HOME_DIR or pass --home-dir")
    out_root = Path(home) / "output"

    sgdense_extra = list(extra_sgdense)
    if seg_num_classes is not None:
        sgdense_extra += ["--segmentation_path", seg_dir or str(Path(source) / "semantic_class"),
                          "--num_segmentation_classes", str(seg_num_classes)]

    sgd_dir = _launch("sgdense", source=source, model_dir=str(out_root / f"{prefix}_sgdense"),
                      iterations=iterations, extra=sgdense_extra,
                      profile_iters=profile_iters, workdir_root=workdir_root,
                      force=force, dry=dry)
    van_dir = _launch("vanilla_3dgs", source=source, model_dir=str(out_root / f"{prefix}_vanilla"),
                      iterations=iterations, extra=(),
                      profile_iters=profile_iters, workdir_root=workdir_root,
                      force=force, dry=dry)

    if dry:
        return {}
    return compare_profiles(str(sgd_dir / "iter_profile.json"),
                            str(van_dir / "iter_profile.json"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="SGDense-vs-vanilla per-iteration profile.")
    p.add_argument("--source", "-s", default=os.environ.get("DATASET_DIR"))
    p.add_argument("--home-dir", default=os.environ.get("HOME_DIR"))
    p.add_argument("--prefix", default="profile",
                   help="output run-id prefix (output/<prefix>_sgdense | _vanilla)")
    p.add_argument("--iterations", type=int, default=4200,
                   help="iterations to actually train (must exceed the profile window)")
    p.add_argument("--profile-iters", default="3999:4099",
                   help="profiled iteration window: 'a:b', 'a,b,c', or 'all'")
    p.add_argument("--seg-num-classes", type=int, default=99)
    p.add_argument("--seg-dir", default=None,
                   help="class-id PNG dir (default <source>/semantic_class)")
    p.add_argument("--extra-sgdense", nargs="+", default=None,
                   help="extra SGDense train args (e.g. --floater_action freeze)")
    p.add_argument("--workdir-root", default="/content")
    p.add_argument("--force", action="store_true", help="rebuild the rasterizer")
    p.add_argument("--dry-run", action="store_true")

    # compare-only mode
    p.add_argument("--compare", action="store_true", help="compare existing profiles")
    p.add_argument("--sgdense-json", default=None)
    p.add_argument("--vanilla-json", default=None)
    args = p.parse_args(argv)

    if args.compare:
        if not (args.sgdense_json and args.vanilla_json):
            p.error("--compare needs --sgdense-json and --vanilla-json")
        cmp = compare_profiles(args.sgdense_json, args.vanilla_json)
        print(format_comparison(cmp))
        return 0

    if not args.source:
        p.error("no --source and $DATASET_DIR is unset")

    extra = []
    for tok in (args.extra_sgdense or []):
        extra += tok.split()

    cmp = profile_pair(
        source=args.source, home_dir=args.home_dir, prefix=args.prefix,
        iterations=args.iterations, profile_iters=args.profile_iters,
        seg_num_classes=args.seg_num_classes, seg_dir=args.seg_dir,
        extra_sgdense=extra, workdir_root=args.workdir_root,
        force=args.force, dry=args.dry_run,
    )
    if cmp:
        print()
        print(format_comparison(cmp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
