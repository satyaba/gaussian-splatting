"""Profile vanilla AND the seg fork back-to-back in one session, same shim.

Runs each rasterizer's training for `ITERS`, probing the backward `renderCUDA`
kernel time over `[LO, HI]`, and prints both tables so they share one GPU instance.

Independent: builds both rasterizers from source, no cache.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

HOME = "/content/drive/MyDrive/Magister/Thesis/Script"
DATASET = Path(HOME) / "Dataset" / "office_0" / "Sequence_1"
SEG = DATASET / "semantic_class"
SEG_REPO = "/content/gaussian-splatting"
VAN_REPO = "/content/gaussian-splatting-vanilla"
LO, HI, ITERS = 7101, 7199, 7200
SHIM_DIR = SEG_REPO + "/scripts"
SHIM = ("import sys; sys.path.insert(0, %r); import torch_profile_iters; "
        "sys.argv=['train.py']+sys.argv[1:]; import runpy; "
        "runpy.run_path('train.py', run_name='__main__')" % SHIM_DIR)


def _install(repo):
    subprocess.run("git submodule update --init --recursive submodules/diff-gaussian-rasterization",
                   shell=True, cwd=repo, check=True)
    rast = repo + "/submodules/diff-gaussian-rasterization"
    subprocess.run("rm -rf build dist *.egg-info", shell=True, cwd=rast)
    subprocess.run("pip uninstall -y diff-gaussian-rasterization", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run("pip install . --no-build-isolation --no-cache-dir --force-reinstall",
                   shell=True, cwd=rast, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _profile(tag, repo, extra):
    _install(repo)
    out = Path(HOME) / "output" / f"same_{tag}"; out.mkdir(parents=True, exist_ok=True)
    tp = str(out / "torch_profile.txt")
    env = dict(os.environ, SGDENSE_TP_ITERS=f"{LO}:{HI}", SGDENSE_TP_OUT=tp)
    cmd = [sys.executable, "-c", SHIM, "-s", str(DATASET), "-m", str(out),
           "--eval", "--disable_viewer", "--quiet", "--iterations", str(ITERS)] + extra
    subprocess.run(cmd, cwd=repo, env=env, check=True, capture_output=True, text=True)
    table = Path(tp).read_text() if Path(tp).exists() else "(no table)"
    print(f"\n=================== {tag} ===================\n{table}")


# vanilla first, then the seg fork (seg fork last so the seg rasterizer is installed after)
_profile("vanilla", VAN_REPO, [])
_profile("sgdense", SEG_REPO,
         ["--segmentation_path", str(SEG), "--num_segmentation_classes", "99",
          "--no_photo_weight_ramp", "--seg_render_interval", "1000000",
          "--depth_l1_weight_init", "0", "--depth_l1_weight_final", "0",
          "--no_densify_photo_grad_only"])
