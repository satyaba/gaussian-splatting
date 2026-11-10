# 07 — Per-iteration cost profiler (SGDense vs vanilla 3DGS)

**Status:** added 2026-11-09.

Answers: *what extra wall-clock does SGDense spend per training iteration vs
upstream graphdeco 3DGS, and where* — from rendering through attribute
optimisation.

## 1. What it measures

Both methods are instrumented with the **same phase names**, so the delta is
meaningful:

| phase | SGDense (`train.py`) | vanilla (`scripts/vanilla_iter_profile.py`) |
|---|---|---|
| `render` | the `render(...)` call (seg channels + vote atomics + decay fixup) | `gaussian_renderer.render` |
| `loss` | photo loss + depth + decoder forward + seg cross-entropy | `utils.loss_utils.l1_loss` + `ssim` |
| `backward` | `loss.backward()` (colour **and** seg-encoding grads) | `torch.Tensor.backward` |
| `densify` | the periodic densification **event** (`seg_hist_readout` → `generate_category_masks` → `densify_and_prune`) + opacity reset | `GaussianModel.densify_and_prune` / `reset_opacity` |
| `optimizer` | all `optimizer.step()`/`zero_grad()` (Gaussians incl. the 32-D `seg_encoding` group, exposure, decoder) | `torch.optim.Adam.step` |
| `misc` | residual = iteration total − measured phases | residual |

Notes:

* The per-iteration `add_densification_stats` / `max_radii2D` update is common
  to both methods and is deliberately **excluded** from `densify` (it lands in
  `misc` for both), so `densify` measures the densification *event* only.
* A densification event fires every `--densification_interval` (100) iterations
  and an opacity reset every `--opacity_reset_interval` (3000). The summary is
  therefore split into `all` / `densify` / `no_densify` groups: the
  `no_densify` group is the steady-state per-iteration cost, the `densify`
  group isolates the event cost.

## 2. How it measures

Host **wall-clock with a `torch.cuda.synchronize()` at every phase boundary**.
This is intentional:

* GPU work is asynchronous — without the boundary sync a phase's wall-clock
  would only measure kernel launches.
* The SGDense densification stack does real **CPU** work (numpy conversions and
  the batched `np.linalg.eigh` plane fit in `sgdense.geometric`); CUDA events
  would report ~0 for it.

A per-iteration CUDA-event pair is also recorded (`iter_gpu_ms`) as a
cross-check. Only the requested iteration window is profiled (default a single
`~100`-iteration slice), so the sync overhead is irrelevant to the run.

## 3. Running

```bash
# from the repo root, on the GPU/Colab box (see experiments/README.md for env)
export HOME_DIR=/content/drive/MyDrive/Magister/Thesis/Script
export DATASET_DIR=$HOME_DIR/Dataset/office_0/Sequence_1

# runs SGDense then vanilla on the same data, profiled at iters 3999..4099,
# swaps the rasterizer between them, then prints the comparison table
python -m experiments.profile --iterations 4200 --profile-iters 3999:4099

# compare two existing profiles
python -m experiments.profile --compare \
    --sgdense-json  <model>/iter_profile.json \
    --vanilla-json  <model>/iter_profile.json
```

The window should sit **after** the SGDense warmup (`seg_warmup_iters`, 2000)
and **inside** the densification window (`densify_until_iter`, default 25000), and
should contain exactly one `% 100 == 0` boundary so both methods have one
densify event. `3999:4099` satisfies all three (event at 4000).

Profile JSONs are written to `<model_dir>/iter_profile.json` for both methods.

### Enabling it on a normal run

SGDense `train.py` also accepts the flags directly (or via env), so any run can
emit a profile without the `experiments.profile` driver:

```bash
python train.py -s <src> -m <model> --eval --disable_viewer \
    --segmentation_path <seg> --num_segmentation_classes 99 \
    --profile_iterations 3999:4099
# or: SGDENSE_PROFILE_ITERS=3999:4099 python train.py ...
```

`--profile_iterations` accepts `a:b` (inclusive range), `a,b,c` (set), or
`all`. Unset/`off` (the default) leaves the loop un-instrumented.

### From Colab

```python
import os
os.environ["HOME_DIR"] = "/content/drive/MyDrive/Magister/Thesis/Script"
os.environ["DATASET_DIR"] = os.path.join(os.environ["HOME_DIR"], "Dataset/office_0/Sequence_1")
!python -m experiments.profile --iterations 4200 --profile-iters 3999:4099
```

## 4. Reading the output

```
Per-iteration phase cost (all iterations)
  vanilla n=101  |  sgdense n=101

  phase       vanilla(ms)  sgdense(ms)  delta(ms)   ratio   delta%
  ---------------------------------------------------------------
  render            ..           ..         ..       ..       ..
  loss              ..           ..         ..       ..       ..
  backward          ..           ..         ..       ..       ..
  densify           ..           ..         ..       ..       ..
  optimizer         ..           ..         ..       ..       ..
  misc              ..           ..         ..       ..       ..
  total             ..           ..         ..       ..       ..
```

`delta_ms = sgdense − vanilla` is the extra per-iteration cost our method adds;
`ratio` is the multiplier. Because the `all` group averages over the window,
run `--compare` on profiles whose window is either all-densify or all
no-densify if you want a clean event-vs-steady-state split (the JSON keeps the
per-iteration records, so `sgdense.summarize_profile` can regroup offline).

## 5. Implementation

* `sgdense/profiler.py` — `IterProfiler` (phases, iteration spec, per-iteration
  records), `summarize_profile`, `compare_profiles`, `format_comparison`.
* `train.py` — profiler wiring at the five phase boundaries (opt-in).
* `scripts/vanilla_iter_profile.py` — runtime monkeypatch shim for upstream
  (same pattern as `scripts/vanilla_densify_log.py`), emits the same schema.
* `experiments/profile.py` — runs both methods + prints the table; `--compare`.

## 6. Caveats

* **Boundary sync serialises the phases**, so an individual phase is slightly
  larger than it would be overlapped. This is consistent across both methods,
  so the *delta* stays fair; the unsynced `iter_gpu_ms` is the cross-check.
* **Different Gaussian counts.** SGDense and vanilla densify differently, so at
  the same iteration they hold different populations. The comparison is
  "real cost at iteration N", not "cost per Gaussian at matched count". For a
  matched-count view, profile a window after both have saturated and report
  `n_gaussians` (kept in each record's `meta`).
* The profiled iterations are a small sample (default 101); raise the window or
  repeat with a different offset if you want tighter means.
