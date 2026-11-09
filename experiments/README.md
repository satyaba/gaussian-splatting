# experiments/ — cross-codebase experiment harness

Run **this** codebase (SGDense-3DGS) and **external** 3DGS codebases (upstream
graphdeco 3DGS, and future baselines) through one interface. The harness handles
cloning, pinning, CUDA-extension install, and — importantly — **swapping the
rasterizer** between methods.

## Why rasterizer swapping matters

Every 3DGS codebase installs its CUDA rasterizer under the same import name,
`diff_gaussian_rasterization`:

- **SGDense-3DGS** needs the seg-aware fork (`satyaba/diff-gaussian-rasterization`,
  branch `seg-encoding`) — the kernel adds segmentation votes.
- **Upstream graphdeco** needs the stock rasterizer.

They cannot coexist in one Python runtime. Each method's `install` first
`pip uninstall`s the package, and the runner re-installs only when you switch
methods (tracked by `/content/.sgdense_rasterizer_method`). So interleaving runs
across codebases works, at the cost of one rasterizer rebuild per switch.

## Quick start (Colab / Linux)

```bash
# from the SGDense-3DGS repo root
export HOME_DIR=/content/drive/MyDrive/Magister/Thesis/Script
export DATASET_DIR=$HOME_DIR/Dataset/office_0/Sequence_1
export SEG_DIR=$DATASET_DIR/semantic_class
```

### The experiment manifest

`experiments/runs.py` declares the runs; one command executes them:

```bash
python -m experiments.runs --list           # show runs
python -m experiments.runs --sanity         # short smoke runs (verify the pipeline)
python -m experiments.runs --run run38_30k  # one run
python -m experiments.runs --all            # every non-sanity run
```

Runs:

| run id | method | iters | notes |
|---|---|---|---|
| `vanilla_30k` | `vanilla_3dgs` | 30000 | upstream graphdeco 3DGS (stock rasterizer), `--eval` |
| `run38_30k` | `sgdense` | 30000 | L1+L2 floaters, `FLOATER_ACTION=freeze`, no texture/position routes |
| `vanilla_sanity` | `vanilla_3dgs` | 3000 | SANITY smoke test |
| `run38_sanity` | `sgdense` | 3000 | SANITY smoke test |

The **sanity runs** reuse the exact full-run configs at 3 000 iterations — enough
to cross the seg warmup (2000), run densification with floater/freeze handling,
hit the opacity reset (3000), and save. Use them to verify the whole pipeline
(train → rasterizer swap → metric extraction) before committing to 30 000 iters.
`--all` / `extract_all()` skip sanity runs; `--sanity` / `extract_sanity()` run
only them.

Add a run by appending a `RunSpec` to `RUNS` (set `seg_num_classes` for SGDense
runs to auto-add `--segmentation_path` / `--num_segmentation_classes`).

### Vanilla event logging

Upstream graphdeco 3DGS has none of this fork's hooks, so
`Vanilla_3DGS_Experiment.ipynb` instruments the cloned upstream code at runtime
with `scripts/vanilla_densify_log.py` (monkeypatches `GaussianModel` to fire a
`DensifyTracker`, env-gated by `SGDENSE_REPO` + `DENSIFY_EVENTS_OUT`). The result
is the same `densify_events.pt` artifact, so vanilla and SGDense densification
dynamics can be compared offline with `sgdense.viz`. Vanilla has no
segmentation, so per-Gaussian `parent_class` is `-1`.

### Densification event log (SGDense runs)

SGDense runs configured with `--log_densify_events` (currently `run38_30k` and
`run38_sanity`) attach a `sgdense.DensifyTracker` in `train.py` and write
`<model_dir>/densify_events.pt` — one record per densification event with the
clone/split/prune selections, prune reasons, per-Gaussian stable ids + population,
and a decimated per-event point snapshot. Load it with `sgdense.load_events(path)`
and analyse/plot with `sgdense.viz` (`plot_timeline`, `plot_prune_breakdown`,
`plotly_replay`, `save_report`). The flag is off by default, so ordinary runs pay
only one `is not None` check per lifecycle call.

### Running one method directly

```bash
# vanilla upstream 3DGS (30k, eval split)
python -m experiments.runner --method vanilla_3dgs --run-id runB1 \
    --iterations 30000

# SGDense-3DGS (same 30k)
python -m experiments.runner --method sgdense --run-id run38ext \
    --iterations 30000 \
    --extra "--segmentation_path $SEG_DIR --num_segmentation_classes 99 --floater_action freeze"
```

`--source` and `--model-dir` default to `$DATASET_DIR` and
`$HOME_DIR/output/$RUN_ID`. Add `--dry-run` to print commands without executing.

### From a notebook cell

```python
import os
from experiments.runs import run_by_id, RUNS

os.environ["HOME_DIR"] = "/content/drive/MyDrive/Magister/Thesis/Script"
os.environ["DATASET_ROOT"] = os.environ["HOME_DIR"] + "/Dataset"

for run_id in RUNS:            # or run_by_id("run38_30k")
    run_by_id(run_id)
```

## How it works

```
experiments/methods.py   Method registry (id -> repo, install, common args)
experiments/runs.py      Run manifest (run id -> method, iterations, extra args)
experiments/runner.py    prepare() + build_train_cmd() + run_method() + CLI
```

- **local methods** (`repo=None`, e.g. `sgdense`) run in *this* checkout.
- **external methods** are cloned into `<workdir_root>/<dir_name>`
  (default `/content/<dir_name>`) and pinned to `Method.ref` if set.
- Only the training submodules (`diff-gaussian-rasterization`, `simple-knn`) are
  initialized — the huge `SIBR_viewers` submodule is never cloned.
- `common_args` encode each method's baseline invocation; e.g. `vanilla_3dgs`
  uses `--eval --disable_viewer --quiet` so its train/test split matches the
  SGDense runs.

## Adding a new baseline

Append to `experiments/methods.py`:

```python
register_method(
    Method(
        id="mip_splatting",
        label="Mip-Splatting",
        repo="https://github.com/autonomousvision/mip-splatting",
        ref="<tag-or-sha>",                 # pin for reproducibility
        dir_name="mip-splatting",
        install=_rasterizer_install(),      # adjust if its install differs
        common_args=("--eval", "--disable_viewer", "--quiet"),
        uses_seg_rasterizer=False,
    )
)
```

`install` and `common_args` are usually the only method-specific parts.

## Metric extraction

`experiments/metrics.py` turns a trained model into a row of the results sheet:
mean **PSNR / SSIM / LPIPS** per split (train & test) plus the final **Gaussian
count** — the same quantities reported by `SGDense_3DGS_Experiment.ipynb`.

```bash
python -m experiments.metrics --run-id run38_30k       # one run
python -m experiments.metrics --sanity                 # the short sanity runs
python -m experiments.metrics --all                    # every non-sanity run
python -m experiments.metrics --model-dir <path> --name "adhoc" --iteration 30000
```

Rows are appended to `$HOME_DIR/output/metrics.csv` (override with `--csv`, or
`--no-csv` to print only). Columns:

```
Name, Run ID, Iteration, Train PSNR, Train SSIM, Train LPIPS,
Test PSNR, Test SSIM, Test LPIPS, Gauss. count,
Train time (min), Eval time (s)
```

- **Train time (min)** — wall-clock of the training run, written by the runner to
  `<model_dir>/train_time.json` (blank if the run was not launched by the harness).
- **Eval time (s)** — wall-clock to compute the metrics for that model.

Computation is delegated to `eval_sgdense.py` (now with `--json_out`), which
renders each split on the fly. **The same metric code path scores every
method** — SGDense and upstream graphdeco PLYs alike (a vanilla PLY simply
loads with zero segmentation encoding) — so numbers are directly comparable. It
renders with this repo's seg-aware rasterizer (RGB path = stock 3DGS; the seg
channels are additive), so the extractor ensures the `sgdense` rasterizer is
installed first. This inherits the same rasterizer-swap rule as training.

Semantic metrics (pixel-acc / mIoU / fwIoU) are included in the JSON when a
`decoder<iter>.pth` is present; they are not part of the CSV row.

## Evaluation notes

- Train and test PSNR/SSIM/LPIPS use the notebook's formulas: MSE-based PSNR,
  `torchmetrics` SSIM, and VGG LPIPS.
- Keep the protocol identical across methods (same `--eval` split, same
  iterations, same code).
- For visual grids, run `eval_sgdense.py --out_dir <dir>` directly.

## Troubleshooting

### External rasterizer build fails

`pip install ./submodules/diff-gaussian-rasterization` for an external method can
fail. The runner now prints the **last lines of the build log** before raising, so
the real cause is visible. Common causes:

- **CUDA 13 / newer libstdc++ dropped the transitive `<cstdint>`** — the
  rasterizer headers then fail with `namespace "std" has no member "uintptr_t"`
  and `identifier "uint32_t" is undefined`. The runner patches this automatically:
  before every rasterizer build it prepends `#include <cstdint>` to
  `cuda_rasterizer/rasterizer_impl.h` and `simple-knn`'s translation units
  (guarded + idempotent). Keep this in mind if you point `install` at other code.
- **Missing `glm`** — the stock `setup.py` hard-includes `third_party/glm/` (a
  nested submodule). The runner force-inits it with
  `git submodule update --init --recursive` inside the rasterizer dir.
- **Toolchain mismatch** — nvcc's CUDA must match `torch.version.cuda` (see the
  notebook's CUDA check).
- **Rasterizer commit incompatible with a very new CUDA/torch** — set
  `rasterizer_ref="3dgs_accel"` on the `vanilla_3dgs` entry in
  `experiments/methods.py` (it is applied after the submodule update, which would
  otherwise reset the checkout). Re-run with `force=True` (CLI `--force`) to rebuild.

## Reproducibility

Pin `Method.ref` (a tag or commit SHA) once you settle on a baseline version.
`ref=None` means "whatever the default branch is at clone time", which is not
reproducible over time.
