"""Cross-codebase experiment harness.

Runs training for the local SGDense-3DGS checkout and for external 3DGS
baselines (upstream graphdeco, and any future method) through one interface,
handling repository cloning, CUDA-extension installation, and rasterizer
swapping between methods.

See ``experiments/README.md`` for usage.
"""

from experiments.methods import Method, METHODS, get_method, register_method

__all__ = [
    "Method",
    "METHODS",
    "get_method",
    "register_method",
    "prepare",
    "build_train_cmd",
    "run_method",
    "workdir_for",
    "extract_one",
    "extract_run",
    "extract_all",
]


def __getattr__(name):
    # Lazy re-export of runner/metrics helpers so `python -m experiments.<mod>`
    # does not import the module twice (package init + runpy __main__).
    if name in {"prepare", "build_train_cmd", "run_method", "workdir_for"}:
        from experiments import runner
        return getattr(runner, name)
    if name in {"extract_one", "extract_run", "extract_all"}:
        from experiments import metrics
        return getattr(metrics, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
