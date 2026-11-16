# experiments/ — cross-codebase experiment harness (SGDC)

Run **this** codebase (SGDC-3DGS) and **external** 3DGS codebases (upstream
graphdeco 3DGS, and future baselines) through one interface. The harness handles
cloning, pinning, CUDA-extension install, and — importantly — **swapping the
rasterizer** between methods.

## Why rasterizer swapping matters

Every 3DGS codebase installs its CUDA rasterizer under the same import name,
`diff_gaussian_rasterization`:

- **SGDC-3DGS** pins the `dr_aa` (antialiasing) branch of the rasterizer.
- **Upstream graphdeco** uses its default (`main`) rasterizer.

They cannot coexist in one Python runtime. Each method's `install` first
`pip uninstall`s the package, and the runner re-installs only when you switch
methods (tracked by `/content/.sgdc_rasterizer_method`). So interleaving runs
across codebases works, at the cost of one rasterizer rebuild per switch.

> **Segmentation note.** Unlike the SGDense fork, there is **no seg-aware
> rasterizer** here. SGDC reads ground-truth label maps from disk (through the
> `sgdc` label adapters) and never renders segmentation channels, so the RGB
> rasterizer is stock for every method. The only reason to swap is the
> `dr_aa`-vs-`main` fork difference.

## Quick start (Colab / Linux)

```bash
# from the SGDC-3DGS repo root
export HOME_DIR=/content/drive/MyDrive/Magister/Thesis/Script
export DATASET_DIR=$HOME_DIR/Dataset/office_0/Sequence_1
```

### The experiment manifest

`experiments/runs.py` declares the runs; one command executes them:

```bash
python -m experiments.runs --list             # show runs
python -m experiments.runs --sanity           # short smoke runs (verify the pipeline)
python -m experiments.runs --run sgdc_a_30k   # one run
python -m experiments.runs --all              # every non-sanity run
```

Runs:

| run id | method | iters | notes |
|---|---|---|---|
| `vanilla_30k` | `vanilla_3dgs` | 30000 | upstream graphdeco 3DGS (stock rasterizer), `--eval` |
| `base_30k` | `sgdc` | 30000 | this checkout with SGDC disabled (plain 3DGS) |
| `sgdc_a_30k` | `sgdc` | 30000 | Component A default: `f_b 0.5 / f_int 2.0 / eta 0.3` |
| `sgdc_a_fb1_fi3_dt60_30k` | `sgdc` | 30000 | A: boundary = baseline, interior ×3, `dt_norm 60` |
| `sgdc_a_fb05_fi4_dt50_30k` | `sgdc` | 30000 | A: boundary boost, interior ×4, `dt_norm 50` |
| `sgdc_a_fb1_fi6_dt40_30k` | `sgdc` | 30000 | A: aggressive interior suppression, `dt_norm 40` |
| `sgdc_c_bnd02_30k` | `sgdc` | 30000 | Component C: `lambda_bnd 0.02` (A neutral) |
| `sgdc_c_bnd10_30k` | `sgdc` | 30000 | Component C: `lambda_bnd 0.10` (A neutral) |
| `sgdc_c_bnd50_30k` | `sgdc` | 30000 | Component C: `lambda_bnd 0.50` (A neutral) |
| `vanilla_sanity` | `vanilla_3dgs` | 3000 | SANITY smoke test |
| `base_sanity` | `sgdc` | 3000 | SANITY: SGDC off |
| `sgdc_a_sanity` | `sgdc` | 3000 | SANITY: Component A default |
| `sgdc_c_sanity` | `sgdc` | 3000 | SANITY: Component C (`lambda_bnd 0.10`) |

**Component A** (boundary-aware density control) scales the clone/split gradient
threshold per Gaussian by its distance to a semantic boundary
(`--f_b` boundary / `--f_int` interior / `--sgdc_dt_norm` px normalizer / `--eta`
split-threshold shrink). **Component C** (cross-boundary scale regularizer) adds
an `erfc`-footprint scale loss on boundary Gaussians (`--lambda_bnd`); its runs
neutralise A so only C acts. See `docs/sgdc_spec.md`.

The **sanity runs** reuse the exact full-run configs at 3 000 iterations —
enough to cross the first densification (500), hit the opacity reset (3000), and
save. Use them to verify the whole pipeline (train → rasterizer swap → metric
extraction) before committing to 30 000 iters. `--all` / `extract_all()` skip
sanity runs; `--sanity` / `extract_sanity()` run only them.

The SGDC Replica adapter reads `<source>/semantic_class` automatically, so SGDC
runs only need `--sgdc_enable` plus their hyperparameters — no path or
class-count arguments.

Add a run by appending a `RunSpec` to `RUNS`. All runs share the `--eval`
(llffhold=8) train/test split, so rows are comparable across methods.

### Running one method directly

```bash
# upstream 3DGS (30k, eval split)
python -m experiments.runner --method vanilla_3dgs --run-id runB1 \
    --iterations 30000

# SGDC (same 30k)
python -m experiments.runner --method sgdc --run-id sgdc_ext \
    --iterations 30000 \
    --extra "--sgdc_enable --f_b 0.5 --f_int 2.0 --eta 0.3"
```

`--source` and `--model-dir` default to `$DATASET_DIR` and
`$HOME_DIR/output/$RUN_ID`. Add `--dry-run` to print commands without executing.

### From a notebook cell

```python
import os
from experiments.runs import run_by_id, RUNS

os.environ["HOME_DIR"] = "/content/drive/MyDrive/Magister/Thesis/Script"
os.environ["DATASET_ROOT"] = os.environ["HOME_DIR"] + "/Dataset"

for run_id in RUNS:            # or run_by_id("sgdc_a_30k")
    run_by_id(run_id)
```

### Per-run hyperparameter overrides (notebook)

`run_by_id` (and `run`) accept an `extra` argument — CLI tokens appended **after**
the spec's own args, so argparse's last-wins lets them override any
hyperparameter for a single call without editing the manifest:

```python
# single shell-quoted string ...
run_by_id("sgdc_a_30k", extra="--f_b 0.3 --f_int 5.0")

# ... or a token list
run_by_id("sgdc_a_30k", extra=["--sgdc_dt_norm", "50"])

# preview without executing
run_by_id("sgdc_a_30k", extra="--lambda_bnd 0.2", dry=True)
```

Works for any `train.py` flag (`--f_b`, `--f_int`, `--eta`, `--sgdc_dt_norm`,
`--lambda_bnd`, `--densify_until_iter`, ...), not just the ones in the spec.

**Turning SGDC off.** `--sgdc_enable` is `store_true`; a run starting from an
SGDC spec can be reduced to plain 3DGS by dropping it (that is exactly what
`base_*` does):

```python
run_by_id("sgdc_a_30k", tag="off", extra="--f_b 1.0 --f_int 1.0 --eta 0.0 --lambda_bnd 0")
```

For a *true* upstream baseline (separate clone, stock rasterizer) use the
`vanilla_3dgs` method instead.

**Sweeps need distinct output dirs.** `tag` appends a suffix to the run id
(`<home>/output/<run_id>_<tag>`), and `model_dir` overrides the path outright:

```python
for fi in (2.0, 3.0, 5.0):
    run_by_id("sgdc_a_30k", tag=f"fi{fi}", extra=f"--f_int {fi}")

# or drop the results wherever you like
run_by_id("sgdc_a_30k", model_dir="/content/output/a_fb02", extra="--f_b 0.2")
```

Each tagged run writes to its own model dir, so `experiments.metrics` can score
them independently — pass the same `tag` (or point `--model-dir` at the dir):

```python
from experiments.metrics import extract_run
extract_run("sgdc_a_30k", tag="fi3.0")     # reads output/sgdc_a_30k_fi3.0
```
```bash
python -m experiments.metrics --run-id sgdc_a_30k --tag fi3.0
```

## How it works

```
experiments/methods.py   Method registry (id -> repo, install, common args)
experiments/runs.py      Run manifest (run id -> method, iterations, extra args)
experiments/runner.py    prepare() + build_train_cmd() + run_method() + CLI
eval_sgdc.py             on-the-fly renderer -> global + boundary/interior metrics
```

- **local methods** (`repo=None`, e.g. `sgdc`) run in *this* checkout.
- **external methods** are cloned into `<workdir_root>/<dir_name>`
  (default `/content/<dir_name>`) and pinned to `Method.ref` if set.
- Only the training submodules (`diff-gaussian-rasterization`, `simple-knn`) are
  initialized — the huge `SIBR_viewers` submodule is never cloned.
- `common_args` encode each method's baseline invocation; e.g. `vanilla_3dgs`
  uses `--eval --disable_viewer --quiet` so its train/test split matches the
  SGDC runs.

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
    )
)
```

`install` and `common_args` are usually the only method-specific parts.

## Metric extraction

`experiments/metrics.py` turns a trained model into a row of the results sheet:
mean **PSNR / SSIM / LPIPS** per split (train & test), the **boundary-restricted**
and **interior** test PSNR, and the final **Gaussian count**.

```bash
python -m experiments.metrics --run-id sgdc_a_30k                  # one run
python -m experiments.metrics --run-id sgdc_a_30k --tag fi3.0      # a sweep point
python -m experiments.metrics --sanity                            # the short sanity runs
python -m experiments.metrics --all                               # every non-sanity run
python -m experiments.metrics --model-dir <path> --name "adhoc" --iteration 30000
```

Rows are appended to `$HOME_DIR/output/metrics.csv` (override with `--csv`, or
`--no-csv` to print only). Columns:

```
Name, Run ID, Iteration, Train PSNR, Train SSIM, Train LPIPS,
Test PSNR, Test SSIM, Test LPIPS, Test PSNR band, Test PSNR interior,
Gauss. count, Train time (min), Eval time (s)
```

- **Train time (min)** — wall-clock of the training run, written by the runner to
  `<model_dir>/train_time.json` (blank if the run was not launched by the harness).
- **Eval time (s)** — wall-clock to compute the metrics for that model.

Computation is delegated to `eval_sgdc.py` (with `--json_out`), which renders
each split on the fly. **The same metric code path scores every method** — SGDC
and upstream graphdeco PLYs alike — so numbers are directly comparable. The
extractor ensures the local `sgdc` rasterizer is installed first (same
rasterizer-swap rule as training).

The **band/interior** split is dataset-driven: it comes from the Replica class
labels under `<source_path>/semantic_class`, the same labels the trainer uses. If
they are absent, only the global metrics are reported and the band/interior
columns are blank.

## Evaluation notes

- Train and test PSNR/SSIM/LPIPS use the repo's canonical formulas from
  `metrics.py`: MSE-based PSNR, `utils.loss_utils.ssim`, and `lpipsPyTorch`
  (`vgg`, inputs in `[0, 1]`).
- Keep the protocol identical across methods (same `--eval` split, same
  iterations, same code).
- The boundary band is a `--boundary_band 4` (px) dilation of the class-boundary
  mask by default; override with `--boundary_band`.
- This is the metric Component A actually targets: if boundary-conditional
  densification helps, it should show up on the **band** even when global PSNR
  is flat. `sgdc.eval_boundary` remains available for quick interactive checks.

## Troubleshooting

### External rasterizer build fails

`pip install ./submodules/diff-gaussian-rasterization` for an external method can
fail. The runner prints the **last lines of the build log** before raising, so
the real cause is visible. Common causes:

- **CUDA 13 / newer libstdc++ dropped the transitive `<cstdint>`** — the
  rasterizer headers then fail with `namespace "std" has no member "uintptr_t"`
  and `identifier "uint32_t" is undefined`. The runner patches this
  automatically: before every rasterizer build it prepends `#include <cstdint>`
  to `cuda_rasterizer/rasterizer_impl.h` and `simple-knn`'s translation units
  (guarded + idempotent).
- **Missing `glm`** — the stock `setup.py` hard-includes `third_party/glm/`
  (a nested submodule). The runner force-inits it with
  `git submodule update --init --recursive` inside the rasterizer dir.
- **Toolchain mismatch** — nvcc's CUDA must match `torch.version.cuda`.
- **Rasterizer commit incompatible with a very new CUDA/torch** — set
  `rasterizer_ref="3dgs_accel"` on the `vanilla_3dgs` entry in
  `experiments/methods.py` (applied after the submodule update, which would
  otherwise reset the checkout). Re-run with `force=True` (CLI `--force`) to
  rebuild.

## Reproducibility

Pin `Method.ref` (a tag or commit SHA) once you settle on a baseline version.
`ref=None` means "whatever the default branch is at clone time", which is not
reproducible over time.
