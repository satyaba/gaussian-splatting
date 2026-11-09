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

**Resolved (2026-11-08).** `tail` is the residual probability mass outside the
top-2 classes, `1 - p1 - p2` (range [0, 1]), matching `tau_tail_high`. The run26
notebook computed `S - p1 - p2`, mixing evidence mass (`S`, unbounded) with
probabilities (`p1,p2`): because `S` grows with observation, the
`tail > tau_tail_high` gate became all-but-vacuous for well-observed Gaussians and
inverted its intent (confident, evidence-rich Gaussians were the most likely to be
flagged ambiguous/floaters). The `ambiguous`/`floaters` rules therefore degenerated
to `entropy > tau_entropy_high` (+ the opacity gate for floaters). Corrected in
`sgdense/categories.py` and the notebook.

## 4. Densification policy (`DensificationConfig`)

- `lvl1_floaters`, `lvl2_floaters` — enable the two floater sources. Floaters are
  applied only when **both** are on (run26 gating). Defaults True.
- `position_based_thresh` — per-regime scaling-threshold weights
  (`interior_weight`, `boundary_weight`, `other_weight`); enables clone/split
  regime routing. Default False.
- `texture_based_thresh` + `texture_beta` — scale the densification gradient
  threshold per class from `s_tilde` (`class_freq_thres`). Default False.
- `texture_based_scale` + `texture_scale_beta` — scale the clone/split **size**
  threshold per class from `s_tilde` (`class_freq_thres` on
  `dense_weight * percent_dense * scene_extent`). Default False.
- `floater_action` (`'prune'` | `'freeze'`) — floater lifecycle. `'prune'`
  removes flagged floaters every densification event (run26; churn);
  `'freeze'` keeps them and de-candidates them for clone/split (no churn).
  Default `'prune'`.

Both texture routes read the same cached `s_tilde`; `train.py` computes it when
either is enabled. `floater_action='freeze'` passes a `freeze_mask` through
`densify_and_clone` / `densify_and_split` (position-based routing already
excludes floaters from the regimes, so the mask is not needed there).

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

1. **[resolved 2026-11-08]** `tail` unit mismatch (mass vs probability) → now
   `1 - p1 - p2`. See §3.
2. **Opacity gate couples floater detection to `reset_opacity`.** The L1/L2 floater
   rules gate on `opac < opacity_thres`, and `reset_opacity` clamps every Gaussian
   to ≤0.01, so each reset seeds a prune burst. Audit of the run26-style runs showed
   the L1+L2 gate is **net-inflating**: ~846k floaters pruned bought ~1.8M extra
   Gaussians added (2× per-capita densification; the model never saturates).
   Decide whether the opacity gate belongs in the floater rule at all, and judge the
   gate on quality at matched Gaussian count (doc 02 §239), not on count.
3. `spike_route.py` (T1 diagnostics) is left in place but describes the superseded
   `(disagree, structure)` split; re-purpose or retire later.

## 7. Data prep / eval scripts

- `scripts/audit_seg_classes.py` — class-id histogram + suggested `num_segmentation_classes`.
- `scripts/compute_texture_freq.py` — writes `s_tilde.npy` / `s_tilde.json`.
- `eval_sgdense.py` — PSNR/SSIM/LPIPS + semantic pixel-acc/mIoU/fwIoU + sample grids.
