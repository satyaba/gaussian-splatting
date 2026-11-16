# SGDC — Experiment Results Wrap-up

**Scope.** Segmentation-Guided Density Control (SGDC) on 3D Gaussian Splatting, built
on pristine upstream INRIA 3DGS (branch `sgdc`). This document records what was
implemented, every result collected, and the conclusions reached.

---

## Setup

| | |
|---|---|
| Dataset | Replica `office_0/Sequence_1` — 90 views, 640×480, HLOC-built COLMAP scene |
| Labels | GT `semantic_class/` (25 class ids across the sequence; **no** `semantic_instance`) |
| Baseline | upstream graphdeco 3DGS + this checkout with SGDC disabled |
| Hardware | Colab GPU (T4-class), rasterizer `dr_aa` |
| Iterations | 30,000 (plus 1.5k/3k smoke runs) |
| Metrics | train/test PSNR·SSIM·LPIPS; boundary-band & interior PSNR; Gaussian count |

**Implemented components (branch `sgdc`):**

- **A — boundary-aware density control:** per-Gaussian clone/split threshold
  `τ_i = τ_g·[f_b(1−d_i) + f_int·d_i]`, split gate `θ_i = S·percent_dense·(1−η(1−d_i))`,
  where `d_i` = normalized distance from Gaussian `i` to the nearest semantic boundary.
- **C — cross-boundary scale regularizer:** `ψ = ½·erfc(sd/σ⊥)` footprint loss on
  boundary Gaussians (`σ⊥ = √(nᵀΣ′n)`).
- **Not implemented:** B (per-segment budgets), D (semantic-consistency pruning),
  E (semantic opacity reset).

**Infrastructure:** `sgdc/` label adapter + `SemanticStore`; `sgdc/eval_boundary.py`
(interactive band/interior PSNR); the `experiments/` cross-codebase harness
(`vanilla_3dgs` baseline, runner, rasterizer swap, metrics CSV with `--eval` split).

---

## 1. Round 1 — Component A sweep (30k, train-view PSNR)

Endpoint PSNR from the training log and final Gaussian count. Note: **train views, no
held-out split** — this metric rewards raw capacity by construction.

| run | config | PSNR | #gauss |
|---|---|---|---|
| base | SGDC off | **41.745** | 530,009 |
| sgdc_a | `f_b 0.5, f_int 2.0, dt_norm 100` | 42.273 | 866,915 |
| a_fb1_fi3_dt60 | `f_b 1.0, f_int 3.0, dt_norm 60` | 40.279 | 316,518 |
| a_fb0.5_fi4_dt50 | `f_b 0.5, f_int 4.0, dt_norm 50` | 40.217 | 522,161 |
| a_fb1_fi6_dt40 | `f_b 1.0, f_int 6.0, dt_norm 40` | 37.835 | 201,792 |

**Read:** the only config that beat baseline (`sgdc_a`) did so by adding **63% more
Gaussians**. At ~matched count (`a_fb0.5_fi4_dt50`, 522k vs base 530k) SGDC-A is
**−1.5 dB**. Every suppression config trades PSNR for count *along or below* baseline's
own curve.

---

## 2. Round 2 — boundary-restricted evaluation (band vs interior)

`sgdc.eval_boundary`, train split, boundary band = 4 px. Same evaluator for all rows.

| run | #gauss | global | **band** | interior |
|---|---|---|---|---|
| base | 534.0k | 43.70 | 38.92 | 45.64 |
| a_fb0.5_fi4_dt50 | 522.2k | 41.68 | **39.11** | 42.50 |
| c_bnd02 (λ=0.02) | 498.1k | 42.89 | 37.92 | 44.93 |
| c_bnd10 (λ=0.10) | 414.3k | 41.94 | 36.54 | 44.29 |
| c_bnd50 (λ=0.50) | 349.1k | 40.32 | 34.53 | 42.99 |

**Read:**
- **A** genuinely **reallocates to boundaries** — `+0.20 dB` on the band with *fewer*
  Gaussians — but loses `−3.13 dB` in interiors, so global is `−2.02 dB`. Correct
  mechanism, bad trade (interior cost ≈ 16× the boundary gain).
- **C** is **monotonically harmful on every metric, including the band it targets**,
  and also *removes* capacity (534k → 349k).

---

## 3. Findings

1. **Target and signal are the same modality.** Density control optimizes photometric
   PSNR using the photometric gradient. On a view-dense scene the gradient is already a
   near-optimal allocator, so a semantic prior is **orthogonal to the objective** and
   can only be equal-or-worse.
2. **A's information is largely redundant.** "Distance of a Gaussian to the nearest
   boundary" is something the RGB gradient already encodes (it spikes at edges). A
   re-encodes existing signal while removing interior capacity.
3. **C's loss has a cheap escape hatch.** `ψ` is reduced by shrinking `σ⊥` *or moving the
   Gaussian off the edge*; optimization takes the trivial path, so boundary Gaussians
   retreat/under-represent edges → band PSNR falls. The premise ("straddling is bad")
   was wrong for hard edges — small Gaussians *on* an edge are desirable.

---

## 4. Caveats

- Round 1 used **train-view** PSNR (capacity-favoring). The harness's **`--eval`
  (llffhold=8)** split is the correct protocol and should govern future rounds.
- **Single, near-saturated scene** (base 43.7 dB) — little headroom; boundaries are a
  small pixel fraction.
- **GT labels** (optimistic vs. predicted SAM2/segmentation).

---

## 5. Reframing

**Principle:** semantics can only help where the photometric gradient is a *poor proxy
for need*. Put SGDC in those regimes:

| Regime | Why gradient fails | Semantic signal |
|---|---|---|
| Sparse views | gradient under-determined; overfits each view | region prior constrains density/geometry |
| Unbounded scenes | parallax-free sky → runaway growth, floaters | "sky/background = low density" |
| Thin / low-texture structures | weak photometric gradient | object identity says "this exists" |
| Geometry / artifacts | PSNR hides floaters & depth error | segment-consistency = floater detector |

Recommended next target: **sparse-view SGDC** (subsample training views; held-out frames
as `--eval` test), which is testable with the existing Replica data + harness.

---

## 6. Reproduce

```bash
# Round 1 + 2 (30k)
python -m experiments.runs --all          # manifest: base, A grid, C grid
python -m experiments.metrics --all       # PSNR/SSIM/LPIPS + band/interior + count

# quick boundary check for one model
python -m sgdc.eval_boundary --model_path <model_dir> -s <DATASET_DIR> \
    --iteration 30000 --boundary_band 4
```

---

## 7. Status

- **Code:** Components A + C on `sgdc`; B/D/E specified in `docs/sgdc_spec.md`, not built.
- **Result:** neither A nor C beats the baseline on `office_0` under the correct
  (reallocation) reading. A is a correct-but-unprofitable mechanism; C is harmful.
- **Decision:** reframe to a regime where semantics carry orthogonal information
  (sparse-view recommended).
