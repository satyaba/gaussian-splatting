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
    txt = BW.read_text()
    sig_old = ("\tfloat4* __restrict__ dL_dconic2D,\n"
               "\tfloat* __restrict__ dL_dopacity,\n"
               "\tfloat* __restrict__ dL_dcolors)")
    sig_new = ("\tfloat4* __restrict__ dL_dconic2D,\n"
               "\tfloat* __restrict__ dL_dopacity,\n"
               "\tfloat* __restrict__ dL_dcolors,\n"
               "\tconst float* __restrict__ _dummy0,\n"
               "\tconst float* __restrict__ _dummy1,\n"
               "\tfloat3* __restrict__ _dummy2,\n"
               "\tfloat* __restrict__ _dummy3)")
    launch_old = "\t\tdL_dcolors\n\t\t);"
    launch_new = "\t\tdL_dcolors,\n\t\tnullptr, nullptr, nullptr, nullptr\n\t\t);"
    assert sig_old in txt and launch_old in txt, "anchors not found"
    BW.write_text(txt.replace(sig_old, sig_new).replace(launch_old, launch_new))


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
