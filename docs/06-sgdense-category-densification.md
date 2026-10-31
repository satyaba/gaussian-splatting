# 06 — SGDense category densification (integrated from `SGDense_3DGS_Experiment.ipynb`, run26)

**Status:** integrated 2026-10-31. This supersedes doc 04 (T0/T1 routing) and doc 05
(THE SPIKE) as the *implementation of record*. The T0 deterministic split placement
(and its `seg_split_deterministic` / `seg_split_offset_ratio` knobs) was **removed**;
splits use the vanilla random `N(0, Σ)` placement.

## 1. What changed vs T0/T1

The notebook replaces the quantile 3-way routing with a **6-category taxonomy**
driven by absolute thresholds plus a **geometric support test**, and makes the
densification policy config-driven.

| Piece | T0/T1 (doc 04) | SGDense (this doc) |
|---|---|---|
| Read-out | `(consistency, structure, disagree, S)` | `(consistency, structure, disagree, entropy, top2, S)` |
| Taxonomy | interior / boundary / floater | unobserved / interior / ambiguous / floaters / default / boundary |
| Boundary signal | `structure` quantile | entropy + top-2 probability thresholds, then reach/height |
| Routing | quantile gate | absolute `TAU_*` + 3D-grid kNN support test |
| Thresholds | per-run quantiles | per-class texture (`s_tilde`) / per-regime size weights |
| Split placement | T0 deterministic | random `N(0, Σ)` (T0 removed) |

## 2. Package layout

```
sgdense/
  readout.py     seg_hist_readout  -> 6-tuple (consistency, structure, disagree, entropy, top2, S)
  geometric.py   build_grid / pick_cell_size / grid_knn / reach_height / thin / p99
  categories.py  generate_category_masks (L1 histogram + L2 geometric support)
  texture.py     compute_texture_freq (s_tilde) + class_freq_thres
  config.py      DensificationConfig (route switches, TAU_*, geometric params)
```

The densification lifecycle itself is on `GaussianModel`
(`scene/gaussian_model.py`): `densify_and_prune` / `densify_and_clone` /
`densify_and_split`, plus `_pad_mask` / `_remap_after_split`. It takes an explicit
`DensificationConfig` (+ optional `s_tilde`), so the model holds no global state.

## 3. Category taxonomy (`generate_category_masks`)

Let `observed = S > seg_hist_min_evidence`; `p1,p2` top-2 probabilities; `e` entropy.

| Category | Rule | Action |
|---|---|---|
| interior | `observed & p1 > tau_p1 & e < tau_entropy_low` | clone |
| floaters (L1) | `observed & p1 <= tau_p1 & p2 > tau_p2_high & tau_entropy_low < e <= tau_entropy_high & opac < opacity_thres`, then `tail > tau_tail_high & e > tau_entropy_high & opac < opacity_thres` | prune |
| ambiguous | `observed & tail > tau_tail_high & e > tau_entropy_high` | other |
| unobserved | `~observed` | untouched |
| default | everything else | L2 test |
| boundary | default + geometrically supported by interior | split |

**L2 geometric support test.** `default` Gaussians are scored against the trusted
`interior` set: `reach` (neighbour distance / local spacing) and `height` (out-of-plane
offset / spacing) via 3D-grid kNN + PCA plane fit. A 10% hold-out of `interior` sets
the p99 baselines. Supported → boundary; unsupported in both → floaters (when
`lvl2_floaters`).

**Known issue (preserved for run26 fidelity).** `tail = S - p1 - p2` mixes mass (`S`)
with probabilities (`p1,p2`). Likely intended `1 - p1 - p2`. Kept verbatim so the
integration matches run26; flagged here for a follow-up decision (see §6).

## 4. Densification policy (`DensificationConfig`)

- `lvl1_floaters`, `lvl2_floaters` — enable the two floater sources. Floaters are
  applied only when **both** are on (run26 gating). Defaults True.
- `position_based_thresh` — per-regime scaling-threshold weights
  (`interior_weight`, `boundary_weight`, `other_weight`); enables clone/split
  regime routing. Default False.
- `texture_based_thresh` + `texture_beta` — scale the densification gradient
  threshold per class from `s_tilde` (`class_freq_thres`). Default False.

Ablation switches are exposed as CLI flags (`--no_lvl1_floaters`, etc.); the
resolved policy is written to `<model_path>/experiment_config.json`.

## 5. Training loop (`train.py`)

- Photo-weight ramp: `photo_weight = min((iteration // seg_warmup_iters)/10, 1.0)`
  for `iteration > seg_warmup_iters`; `loss = photo_weight*photo + depth + L_seg`.
- `oneupSHdegree()` every 1000 iters (unconditional).
- Densification: read 6-tuple → `generate_category_masks` → `densify_and_prune`
  with the four masks + `cfg` + `s_tilde`.
- Saves `seg_hist{N}.npz` (rich metadata) and `experiment_config.json`.

Defaults (run26): `seg_warmup_iters=2000`.

## 6. Open items

1. Resolve the `tail` unit mismatch (mass vs probability).
2. `spike_route.py` (T1 diagnostics) is left in place but describes the superseded
   `(disagree, structure)` split; re-purpose or retire later.

## 7. Data prep / eval scripts

- `scripts/audit_seg_classes.py` — class-id histogram + suggested `num_segmentation_classes`.
- `scripts/compute_texture_freq.py` — writes `s_tilde.npy` / `s_tilde.json`.
- `eval_sgdense.py` — PSNR/SSIM/LPIPS + semantic pixel-acc/mIoU/fwIoU + sample grids.
