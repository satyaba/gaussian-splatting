"""Smoke tests for the ``experiments`` harness (no GPU / torch required).

Covers the parts that are easy to break: the run manifest, the train-command
builder, the ``--extra`` token splitting, and the metrics row shaping.

    pytest tests/test_experiments.py
    python tests/test_experiments.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import runner, runs
from experiments.metrics import CSV_COLUMNS, to_row


def test_manifest_is_well_formed():
    assert runs.RUNS, "manifest is empty"
    for run_id, spec in runs.RUNS.items():
        assert run_id == spec.run_id
        assert spec.iterations > 0
    # every referenced method exists
    from experiments.methods import METHODS
    for spec in runs.RUNS.values():
        assert spec.method in METHODS, f"{spec.run_id}: unknown method {spec.method}"


def test_sanity_partition():
    full = {r.run_id for r in runs.full_runs()}
    sanity = {r.run_id for r in runs.sanity_runs()}
    assert not (full & sanity)
    assert full | sanity == set(runs.RUNS)


def test_build_train_cmd_order():
    from experiments.methods import get_method

    m = get_method("sgdc")
    cmd = runner.build_train_cmd(
        m, source="/data", model_dir="/out", iterations=3000,
        extra=("--sgdc_enable", "--f_b", "0.5"),
    )
    assert cmd[0].endswith("python") or "python" in cmd[0]
    assert cmd[1] == "train.py"
    # -s then -m
    assert cmd[cmd.index("-s") + 1] == "/data"
    assert cmd[cmd.index("-m") + 1] == "/out"
    # method common args + iterations are present
    for a in m.common_args:
        assert a in cmd
    assert "--iterations" in cmd and "3000" in cmd
    # extra comes last, after --iterations (argparse last-wins)
    assert cmd[-3:] == ["--sgdc_enable", "--f_b", "0.5"]


def test_split_extra_string_and_tokens():
    assert runner._split_extra(["--f_b 0.5 --f_int 2.0"]) == ["--f_b", "0.5", "--f_int", "2.0"]
    assert runner._split_extra(["--f_b", "0.5"]) == ["--f_b", "0.5"]
    assert runner._split_extra(None) == []
    assert runner._split_extra([]) == []


def test_run_overrides_append_after_spec_extra():
    captured = {}

    def fake_run_method(method_id, **kw):
        captured["method"] = method_id
        captured["extra"] = list(kw["extra"])
        return kw["model_dir"]

    original = runs.run_method
    runs.run_method = fake_run_method
    try:
        spec = runs.RUNS["sgdc_a_30k"]
        runs.run(spec, home_dir="/h", dataset_root="/ds", extra="--f_b 0.1 --f_int 9.0")
    finally:
        runs.run_method = original
    extra = captured["extra"]
    # spec's own tokens come first, overrides last
    assert extra[:5] == ["--sgdc_enable", "--f_b", "0.5", "--f_int", "2.0"]
    assert extra[-4:] == ["--f_b", "0.1", "--f_int", "9.0"]


def test_to_row_shapes_and_rounds():
    data = {
        "run_dir": "/out/sgdc_a_30k",
        "iteration": 30000,
        "gaussians": 123456,
        "splits": {
            "train": {"psnr": 30.1234, "ssim": 0.912345, "lpips": 0.20123},
            "test": {"psnr": 28.4444, "ssim": 0.854321, "lpips": 0.31234,
                     "psnr_boundary": 25.123, "psnr_interior": 29.876},
        },
    }
    row = to_row(data, run_id="sgdc_a_30k", name="A default")
    assert row["Name"] == "A default"
    assert row["Run ID"] == "sgdc_a_30k"
    assert row["Iteration"] == 30000
    assert row["Test PSNR"] == 28.44
    assert row["Test SSIM"] == 0.8543
    assert row["Test PSNR band"] == 25.12
    assert row["Test PSNR interior"] == 29.88
    assert row["Gauss. count"] == 123456
    assert set(row) == set(CSV_COLUMNS)


def test_to_row_missing_band_metrics_stay_none():
    data = {"run_dir": "/out/base", "iteration": 30000, "gaussians": 10,
            "splits": {"test": {"psnr": 27.0, "ssim": 0.8, "lpips": 0.4}}}
    row = to_row(data)
    assert row["Test PSNR band"] is None
    assert row["Test PSNR interior"] is None


if __name__ == "__main__":
    import traceback

    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    sys.exit(1 if failures else 0)
