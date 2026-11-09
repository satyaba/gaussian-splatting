"""Registry of trainable methods for cross-codebase experiments.

Two kinds of method:

* **local** (``repo is None``) -- runs in *this* checkout. That is the
  SGDense-3DGS seg-aware fork.
* **external** -- cloned from a git URL into ``<workdir_root>/<dir_name>``
  (upstream graphdeco 3DGS, and any future baseline).

Rasterizer swapping
-------------------
Every 3DGS codebase installs its CUDA rasterizer under the *same* import name,
``diff_gaussian_rasterization``. Two methods that need different rasterizers
therefore cannot coexist in one Python runtime. Each ``install`` tuple begins by
uninstalling that package; the runner only re-runs ``install`` when the requested
method differs from the last one it installed (tracked by a marker file), so a
run costs a rebuild only when you actually switch codebases.

Adding a new baseline
---------------------
Append a :class:`Method` to :data:`METHODS`. ``install`` and ``common_args`` are
the only things that are usually method-specific. Use ``submodules/...`` paths
relative to the method's own workdir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Method:
    """One runnable codebase + its train invocation.

    id                : registry key (also the rastserizer marker value)
    label             : human-readable name
    repo              : git URL, or None for the local checkout
    ref               : git ref (tag/branch/sha) to pin, or None for default
    dir_name          : clone directory name under workdir_root (external only)
    install           : raw shell commands run in the method workdir
    entry             : training entry script (relative to the workdir)
    common_args       : train args always appended
    uses_seg_rasterizer: True if the seg-aware (kernel seg-vote) rasterizer
    notes             : free-form note surfaced by the CLI
    """

    id: str
    label: str
    repo: Optional[str] = None
    ref: Optional[str] = None
    dir_name: Optional[str] = None
    install: Tuple[str, ...] = ()
    entry: str = "train.py"
    common_args: Tuple[str, ...] = ()
    uses_seg_rasterizer: bool = False
    rasterizer_ref: Optional[str] = None   # branch/sha to check out in the rasterizer submodule
    notes: str = ""


# Ensure <cstdint> is included in the files that need it. CUDA 13 / newer
# libstdc++ no longer pull it in transitively, so the rasterizer headers fail
# with `namespace "std" has no member "uintptr_t"` / `identifier "uint32_t" is
# undefined`. Guarded + idempotent (only inserts when the file exists and does
# not already include it).
_CSTDINT_PATCH = (
    r"""for f in submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h """
    r"""submodules/simple-knn/cuda_utils.h submodules/simple-knn/simple_knn.cu; do """
    r"""if [ -f "$f" ] && ! grep -q 'include <cstdint>' "$f"; then """
    r"""sed -i '1i #include <cstdint>' "$f"; echo "patched $f"; fi; done"""
)


def _rasterizer_install() -> Tuple[str, ...]:
    """Uninstall whatever rasterizer is present, patch, then build this method's own.

    The ``<cstdint>`` patch works for the stock rasterizer, simple-knn, and any
    fork that lacks the include.
    """
    return (
        "pip uninstall -y diff-gaussian-rasterization",
        _CSTDINT_PATCH,
        "pip install ./submodules/diff-gaussian-rasterization --no-build-isolation",
        "pip install ./submodules/simple-knn --no-build-isolation",
    )


METHODS: Dict[str, Method] = {}


def register_method(method: Method) -> Method:
    METHODS[method.id] = method
    return method


def get_method(method_id: str) -> Method:
    try:
        return METHODS[method_id]
    except KeyError:
        raise KeyError(
            f"unknown method {method_id!r}; known: {sorted(METHODS)}"
        )


# --------------------------------------------------------------------------- #
# Local: the SGDense-3DGS fork (this checkout)                                 #
# --------------------------------------------------------------------------- #
register_method(
    Method(
        id="sgdense",
        label="SGDense-3DGS (seg-aware fork; this checkout)",
        repo=None,
        install=_rasterizer_install(),
        common_args=("--disable_viewer", "--quiet"),
        uses_seg_rasterizer=True,
        notes=(
            "Runs in this repo. Needs per-run --segmentation_path and "
            "--num_segmentation_classes; pass them via --extra."
        ),
    )
)

# --------------------------------------------------------------------------- #
# External: upstream graphdeco 3DGS (vanilla baseline)                          #
# --------------------------------------------------------------------------- #
register_method(
    Method(
        id="vanilla_3dgs",
        label="3DGS (upstream graphdeco)",
        repo="https://github.com/graphdeco-inria/gaussian-splatting",
        ref="main",  # pin to a tag/sha for a frozen baseline
        dir_name="gaussian-splatting-vanilla",
        install=_rasterizer_install(),
        common_args=("--eval", "--disable_viewer", "--quiet"),
        uses_seg_rasterizer=False,
        notes="Stock rasterizer. --eval mirrors the SGDense train/test split.",
    )
)
