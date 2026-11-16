"""Cross-codebase experiment harness (SGDC).

Runs training for the local SGDC checkout and for external 3DGS baselines
(upstream graphdeco, and any future method) through one interface, handling
repository cloning, CUDA-extension installation, and rasterizer swapping
between methods.

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
    "RUNS",
    "run_all",
    "run_by_id",
    "run_sanity",
    "extract_one",
    "extract_run",
    "extract_all",
    "extract_sanity",
]


def __getattr__(name):
    # Lazy re-export of runner/runs/metrics helpers so `python -m experiments.<mod>`
    # does not import the module twice (package init + runpy __main__).
    if name in {"prepare", "build_train_cmd", "run_method", "workdir_for"}:
        from experiments import runner
        return getattr(runner, name)
    if name in {"extract_one", "extract_run", "extract_all", "extract_sanity"}:
        from experiments import metrics
        return getattr(metrics, name)
    if name in {"RUNS", "run_all", "run_by_id", "run_sanity"}:
        from experiments import runs
        return getattr(runs, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
