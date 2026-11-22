"""Decisive ABI test for the seg fork's backward slowdown.

Adds 4 unused pointer parameters to the *vanilla* rasterizer's backward
``renderCUDA`` kernel and re-profiles it. If vanilla jumps from ~11 ms to ~51 ms,
the 4.8x is an ABI/codegen artifact of the extra kernel parameters (the only
kernel-level difference between the two builds). If it stays ~11 ms, the cause is
elsewhere (toolchain/environment) and warrants Nsight Compute.

Usage (Colab, GPU):    python3 scripts/abi_param_test.py
"""

import os
import subprocess
import sys
from pathlib import Path

HOME = "/content/drive/MyDrive/Magister/Thesis/Script"
DATASET = Path(HOME) / "Dataset" / "office_0" / "Sequence_1"
VAN = "/content/gaussian-splatting-vanilla"
SHIM_DIR = "/content/gaussian-splatting/scripts"
LO, HI, ITERS = 7101, 7199, 7200
SHIM = ("import sys; sys.path.insert(0, %r); import torch_profile_iters; "
        "sys.argv=['train.py']+sys.argv[1:]; import runpy; "
        "runpy.run_path('train.py', run_name='__main__')" % SHIM_DIR)

RAST = VAN + "/submodules/diff-gaussian-rasterization"
BW = Path(RAST) / "cuda_rasterizer" / "backward.cu"


def _sh(cmd, cwd, check=True):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        print("\n".join(((r.stdout or "") + (r.stderr or "")).splitlines()[-30:]))
        raise SystemExit(f"failed: {cmd}")
    return r


def build():
    _sh("git submodule update --init --recursive submodules/diff-gaussian-rasterization", VAN)
    for rel in ("cuda_rasterizer/rasterizer_impl.h",):
        f = Path(RAST) / rel
        if f.exists() and "#include <cstdint>" not in f.read_text():
            f.write_text("#include <cstdint>\n" + f.read_text())
    _sh("git checkout -- cuda_rasterizer/backward.cu", RAST, check=False)  # clean slate
    _sh("rm -rf build dist *.egg-info", RAST)
    _sh("pip uninstall -y diff-gaussian-rasterization", RAST, check=False)
    _sh("pip install . --no-build-isolation --no-cache-dir --force-reinstall", RAST)


def patch_dummy_params():
    import re
    txt = BW.read_text()
    # (1) extend the kernel signature with 4 unused pointer params (last param is
    #     `float* __restrict__ dL_dinvdepths)`)
    txt, n1 = re.subn(r'(float\*\s+__restrict__\s+dL_dinvdepths)(\s*)\)',
                      r'\1,\n'
                      '\tconst float* __restrict__ _d0,\n'
                      '\tconst float* __restrict__ _d1,\n'
                      '\tfloat3* __restrict__ _d2,\n'
                      '\tfloat* __restrict__ _d3\2)',
                      txt, count=1)
    # (2) pass nulls at the kernel-launch site
    txt, n2 = re.subn(r'(dL_dinvdepths)(\s*\)\s*;)',
                      r'\1,\n\t\tnullptr, nullptr, nullptr, nullptr\2',
                      txt, count=1)
    if n1 != 1 or n2 != 1:
        print("PATCH FAILED: signature=%d launch=%d; nearby lines:" % (n1, n2))
        for ln in BW.read_text().splitlines():
            if "dL_dinvdepths" in ln:
                print("   ", repr(ln))
        raise SystemExit(1)
    BW.write_text(txt)


def profile(tag):
    out = Path(HOME) / "output" / tag
    out.mkdir(parents=True, exist_ok=True)
    tp = str(out / "torch_profile.txt")
    env = dict(os.environ, SGDENSE_TP_ITERS=f"{LO}:{HI}", SGDENSE_TP_OUT=tp)
    cmd = [sys.executable, "-c", SHIM, "-s", str(DATASET), "-m", str(out),
           "--eval", "--disable_viewer", "--quiet", "--iterations", str(ITERS)]
    subprocess.run(cmd, cwd=VAN, env=env, check=True, capture_output=True, text=True)
    print(f"\n=================== {tag} ===================")
    print(Path(tp).read_text() if Path(tp).exists() else "(no table)")


if not Path(VAN).exists():
    subprocess.run(f"git clone -q https://github.com/graphdeco-inria/gaussian-splatting {VAN}",
                   shell=True, check=True)

build()
profile("vanilla_baseline")
patch_dummy_params()
build()
profile("vanilla_plus4dummyparams")
_sh("git checkout -- cuda_rasterizer/backward.cu", RAST, check=False)
