"""Monkeypatch ``torch.Tensor.backward`` with ``torch.profiler`` over an index window.

Robust, standalone CUDA-kernel profiler for a training run: it counts
``Tensor.backward`` calls (≈ one per training iteration) and records CUDA kernel
timings for the calls in ``[SGDENSE_TP_ITERS_LO, SGDENSE_TP_ITERS_HI]``. The
top-kernel table is written to ``$SGDENSE_TP_OUT`` (and stderr) at interpreter exit.

Usage (from the repo root, with the SEG rasterizer installed)::

    SGDENSE_TP_ITERS=7101:7199 SGDENSE_TP_OUT=/tmp/torch_profile.txt \\
    python -c "import sys; sys.path.insert(0,'scripts'); import torch_profile_iters; \\
               sys.argv=['train.py']+sys.argv[1:]; import runpy; \\
               runpy.run_path('train.py', run_name='__main__')" <train args...>

This is deliberately independent of train.py's own profiler so the table is always
emitted.
"""

from __future__ import annotations

import atexit
import os
import sys

import torch

_lo, _hi = (int(x) for x in os.environ.get("SGDENSE_TP_ITERS", "0:0").split(":"))
_out = os.environ.get("SGDENSE_TP_OUT", "torch_profile.txt")

_n = 0
_prof = None
_orig_backward = torch.Tensor.backward


def _backward(self, *args, **kwargs):
    global _n, _prof
    _n += 1
    if _lo <= _n <= _hi and _prof is None:
        _prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA])
        _prof.start()
    return _orig_backward(self, *args, **kwargs)


torch.Tensor.backward = _backward


@atexit.register
def _dump() -> None:
    if _prof is None:
        msg = f"[torch.profiler] no backward calls recorded in window (n={_n})\n"
    else:
        _prof.stop()
        msg = (f"[torch.profiler] top CUDA kernels over backward calls {_lo}..{_hi} "
               f"(n={_n}):\n") + _prof.key_averages().table(sort_by="cuda_time_total",
                                                             row_limit=30)
    try:
        with open(_out, "w") as f:
            f.write(msg)
    except OSError:
        pass
    sys.stderr.write("\n" + msg + "\n")
