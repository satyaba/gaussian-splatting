# Proposal: Segmentation-Gated Densification Routing (T0 + T1)

## 0. Status and scope

**Status:** proposed 2026-10-05. T0 is present on disk but **uncommitted and currently non-runnable** (two defects — see §4.0); T1 is unimplemented. This document is the design of record for both. It **resolves doc 02 §6.1** (the sign conflict) — see §6.

**Scope — two tiers, shipped together, gated differently:**

| Tier | What | Gate |
|---|---|---|
| **T0** | Deterministic split placement: replace `N(0, Σ)` with a fixed layout along the parent's major axis. | No gate — ships on. Substrate only; changes *where* children land, not *whether* densification happens. |
| **T1** | Regime routing: `(disagree, structure, S)` → {interior → clone, boundary → split, floater → prune}. | **Gated on §7.6 THE SPIKE.** Written now behind `seg_route_enabled`, **disabled by default** until the spike shows `(disagree, structure)` separates the regimes. |

**Not in scope:** T2 (boundary-conditioned orientation via a local class-boundary normal `n̂`), 4-way splitting, merge/consolidate, per-regime child-inheritance weighting. See §8.

**Preconditions inherited from the docs:**

- Doc 02 §7: v2 policy must not be relied on until v1 acceptance + **the SPIKE** pass. T1 is v2; hence `seg_route_enabled=False` until then.
- The segmentation signal is live: `seg_hist_readout()` (`scene/gaussian_model.py:577`) returns `(consistency, structure, disagree, S)`; called at `train.py:250`. Doc 02 §5's `read_counters()` is superseded by doc 03's persisted histogram — T1 uses the real read-out.

## 1. Context: what routing closes

Vanilla 3DGS densifies purely on view-space gradient magnitude. Its blind spot (§1.1 of doc 02): the **class boundary between two photometrically similar regions has near-zero radiance gradient**, so photometric ADC neither splits nor restrains there appropriately. The persisted segmentation histogram exposes a per-Gaussian belief `M[g, c]`; its read-out `(disagree, structure, S)` distinguishes surfaces the photometric signal cannot. T1 converts that distinction into placement decisions; T0 makes those placements reproducible.

## 2. Locked design decisions (do not re-litigate without reason)

1. **Placement is the point.** Segmentation must drive *where* children land, not only *when* densification fires. T1 chooses the mode; T0 fixes the geometry of the mode.
2. **Two-signal regime test.** `structure` alone does not separate interior from boundary (both ≈ 1.0); the gate is `disagree` **AND** `structure`. Floater = high `disagree`, **low** `structure`.
3. **Evidence gate, never `disagree`.** Rows with `S ≈ 0` (no multi-view evidence yet) are gated out with `S > seg_hist_min_evidence`, **never** with `disagree` (an unobserved Gaussian is not "confident").
4. **Relative thresholds.** `q_disagree`, `q_structure` are **quantiles of the current active population**, not absolute constants — the distribution is non-stationary (doc 02 §2). They are calibrated from the SPIKE scatter, never guessed.
5. **Determinism.** Split placement must be a pure function of `(R, σ)` — reproducible across runs, a prerequisite for the segmentation experiment to be a fair test.
6. **Offset/scale coupling.** Child scale and child offset distance are chosen jointly (`δ = ρ·σ_major`, child scale `σ/(0.8N)`), else gaps or holes appear at the parent rim.
7. **Children accrue fresh evidence.** Inherited `seg_hist` is discounted (`c = 0.5`); a child is never pre-selected for densification in the step that created it.

## 3. The regimes

Read-out (per active Gaussian): `consistency = max_c P[g,c]`, `structure = P_(1)+P_(2)`, `disagree = 1 − P_(1)`, `S = Σ_c M[g,c]`.

Let `act = S > seg_hist_min_evidence`, `τ_d = quantile(disagree[act], q_disagree)`, `τ_s = quantile(structure[act], q_structure)`.

| Regime | Detector | Action | Placement |
|---|---|---|---|
| **interior** | `act ∧ disagree < τ_d` | **clone** | Δ = 0 (verbatim duplicate) |
| **boundary** | `act ∧ disagree ≥ τ_d ∧ structure ≥ τ_s` | **split** | deterministic, along parent major axis (T0) |
| **floater** | `act ∧ disagree ≥ τ_d ∧ structure < τ_s` | **prune** | — |
| **under-observed** | `¬act` | abstain | untouched |

The three masks partition `act`; `¬act` is untouched. All three AND with the existing photometric + size selection (§5.2), so a Gaussian still needs to be gradient-hot to be densified at all — this is the **joint gate** (doc 02 §7.7), not a segmentation-only override.

## 4. T0 — deterministic split placement, file by file

### 4.0 Defects in the current on-disk draft (must fix before T0 runs)

The uncommitted T0 block in `densify_and_split` will crash on the first split:

- `scene/gaussian_model.py:519` reads `self.seg_split_deterministic`, which is **defined nowhere** → `AttributeError`.
- `:525` uses `major_idx`, but `:522` binds `major_ix` → `NameError`.

Fix (see §4.2 and §4.3): define the two args in `arguments/__init__.py`, and rename the use-site to `major_ix`. The `scatter_` logic is otherwise correct: child `j` of parent `k` lands at offset `±ρ·major_len[k]` along the parent's local major axis, and `reshape(K*N,3)` / `repeat(N,1)` keep child ordering aligned with the parent-prune buffer.

### 4.1 What changes

Only `densify_and_split` (`:510`). `densify_and_clone` is **unchanged** — Δ = 0 is already deterministic.

### 4.2 `arguments/__init__.py` (after `:113`, beside `seg_hist_inherit_discount`)

```python
        # Deterministic split placement (T0)
        self.seg_split_deterministic = True    # False = vanilla random N(0, Sigma); ablation baseline
        self.seg_split_offset_ratio = 0.5      # rho: children at +/- rho * major-axis length
```

### 4.3 `scene/gaussian_model.py` — `densify_and_split`, replace the three random lines (`:519–521`)

```python
        if self.seg_split_deterministic:
            # Deterministic children along the parent's MAJOR axis (local frame),
            # symmetric about the mean. Reproducible: f(R, sigma) only.
            sel_scaling = self.get_scaling[selected_pts_mask]          # [K,3]
            K = sel_scaling.shape[0]
            major_len, major_ix = sel_scaling.max(dim=1)               # [K], [K]
            fracs = (torch.linspace(-1.0, 1.0, N, device="cuda")
                     if N > 1 else torch.zeros(1, device="cuda"))      # [N]
            offs = torch.zeros((K, N, 3), device="cuda")
            offs.scatter_(2, major_ix[:, None, None].expand(K, N, 1),
                          (self.seg_split_offset_ratio * major_len)[:, None]
                          .mul(fracs[None, :]).unsqueeze(-1))          # [K,N,3]
            samples = offs.reshape(K * N, 3)                           # row k*N+j = child j of parent k
        else:
            stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
```

Everything below (`rots`, `new_xyz`, attribute `repeat(N,1)`, `prune_filter`) is unchanged.

### 4.4 T0 acceptance

- **Determinism:** two same-seed short runs → identical Gaussian-count trajectory. (GPU; permission-gated.)
- **Ablation parity:** `seg_split_deterministic=False` reproduces the current random placement.
- **Layout math:** children at `±ρσ_major` along the locally-rotated major axis, symmetric about the mean, ordering `k*N+j` matching `repeat(N,1)`.

## 5. T1 — regime routing, file by file

### 5.1 `arguments/__init__.py` (after the T0 block)

```python
        # --- T1: segmentation regime routing (doc 02 §5) ---
        self.seg_route_enabled = False     # False = T0-only (vanilla routing); the ablation switch
        self.seg_q_disagree = 0.75         # tau_d = quantile(disagree[act], this)
        self.seg_q_structure = 0.50        # tau_s = quantile(structure[act], this)
```

> `seg_route_enabled` defaults **False** — T1 lands dormant until the SPIKE (§7) passes. `seg_hist_min_evidence` already exists (`:112`). The `q_*` values are placeholders pending calibration from the SPIKE scatter (§6.3 of doc 02); they are never hand-tuned.

### 5.2 `scene/gaussian_model.py`

**(a) Helper, above `densify_and_split`:**

```python
    @staticmethod
    def _pad_mask(mask, n):
        """Regime masks are length P_old. densify_and_clone appends rows at the END,
        so masks passed to a later op are padded with False: a newly created child is
        never pre-selected (it must accrue fresh evidence, doc 02 §5)."""
        if mask is None or mask.shape[0] == n:
            return mask
        out = torch.zeros(n, dtype=torch.bool, device="cuda")
        out[:mask.shape[0]] = mask
        return out
```

**(b) `densify_and_split` (`:510`) — accept `sel_mask`, AND it in (keep the T0 block of §4.3):**

```python
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2, sel_mask=None):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        if sel_mask is not None:                                    # NEW (T1)
            selected_pts_mask = torch.logical_and(
                selected_pts_mask, self._pad_mask(sel_mask, n_init_points))
        # ... T0 deterministic block, unchanged ...
```

**(c) `densify_and_clone` (`:538`) — same pattern (lengths already match: clone runs first):**

```python
    def densify_and_clone(self, grads, grad_threshold, scene_extent, sel_mask=None):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        if sel_mask is not None:                                    # NEW (T1)
            selected_pts_mask = torch.logical_and(selected_pts_mask, sel_mask)
        # ...
```

**(d) `densify_and_prune` (`:557`) — accept masks, route, and OR the floater mask into the prune set:**

```python
    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii,
                          clone_mask=None, split_mask=None, floater_mask=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent, sel_mask=clone_mask)     # interior -> clone
        self.densify_and_split(grads, max_grad, extent, sel_mask=split_mask)     # boundary -> split

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if floater_mask is not None:                                            # floater -> prune
            prune_mask = torch.logical_or(
                prune_mask, self._pad_mask(floater_mask, prune_mask.shape[0]))
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None
        torch.cuda.empty_cache()
```

### 5.3 `train.py` (densification block, `:247–257`)

Insert routing after the read-out (`:250`) and pass masks into the call:

```python
                    consistency, structure, disagree, S = gaussians.seg_hist_readout()
                    if opt.seg_route_enabled:
                        act = S > opt.seg_hist_min_evidence
                        if act.any():
                            tau_d = torch.quantile(disagree[act], opt.seg_q_disagree)
                            tau_s = torch.quantile(structure[act], opt.seg_q_structure)
                            boundary = act & (disagree >= tau_d) & (structure >= tau_s)
                            floater  = act & (disagree >= tau_d) & (structure <  tau_s)
                            interior = act & (disagree <  tau_d)
                        else:
                            boundary = floater = interior = None     # no evidence -> vanilla
                    else:
                        boundary = floater = interior = None
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent,
                                                size_threshold, radii,
                                                clone_mask=interior, split_mask=boundary, floater_mask=floater)
```

**Also:** remove the dead deferred-4-way stub at `train.py:262–266` (`if ...: pass`) — T1 supersedes doc 01 §6.

## 6. Resolved question — doc 02 §6.1 (the sign conflict)

**Resolution (2026-10-05): the two rules are combined by reinterpretation.** Fork A (doc 00) and Fork B (doc 02) are not two settings of one knob — they assign opposite actions to the same high-consistency population, and Fork A is *silent* about boundaries and floaters. So they collide in exactly one place: the settled interior.

| `consistency` | Fork A (doc 00 :15) | Fork B (doc 02 §3.2) |
|---|---|---|
| high (interior) | split (aggressive) | no churn |
| low + structured (boundary) | *(silent)* | split |
| low + diffuse (floater) | *(silent)* | prune |

Since Fork A speaks only about the interior, the boundary and floater regimes are unconflicted (take Fork B). The combination therefore keeps **Fork B's three-way routing** and makes a single explicit choice for the interior: **densify it via clone (Δ = 0)**, i.e. doc 00's "confidence licenses densification" survives as a *mild, non-destructive* interior op rather than doc 00's literal aggressive 4-way split. Doc 00's rule is thus reinterpreted from "the target population" to "the mode for the confident case."

**Accepted risk (recorded):** Fork B *alone* restricts densification to boundaries; the combination densifies the interior too, so **population growth in settled regions** is expected. The §7.7 ablation (photometric-only vs joint gate) and the Gaussian-count-vs-quality curve are the check. If growth hurts PSNR-per-splat, dial the interior toward no-op (`clone_mask=None` for interior) — a one-line change.

**Superseded elements of doc 00:** the `confidence ≥ 0.75` constant and the `≥3 consecutive iterations` tracking are replaced by the persisted histogram + calibrated quantiles (doc 03; §2.4 above).

**Remaining drift (flagged, not fixed here):** doc 02 §5 phrases child inheritance as "opacity discounted by consistency"; the repo discounts `seg_hist` by a **constant** `c = 0.5` (`:539`, `:558`). T0/T1 keep the constant and treat consistency-weighted inheritance as a later refinement (needs `n̂`/T2 machinery for side-biased inheritance).

## 7. Acceptance criteria & verification

**Local (CPU, no GPU), must pass before any GPU run:**

1. `python -m py_compile` on the three edited files.
2. **Mask algebra:** synthetic `(disagree, structure, S)` → assert `{interior, boundary, floater}` are pairwise disjoint and partition `act`; correct under both quantile thresholds; `act.any()==False` yields `None` masks.
3. **Padding/ordering:** `_pad_mask(m, n)` returns length-`n` bool with the original prefix intact; a mask never selects an appended child.
4. **Vanilla parity:** `seg_route_enabled=False` reproduces T0-only behavior bit-for-bit (masks `None`).

**Gated (GPU, permission-required):**

5. **THE SPIKE (doc 02 §7.6 — the decisive test).** Short run from a checkpoint; for every Gaussian record `(disagree, structure)` and independently label it ground-truth *boundary / interior / floater* (GT segmentation at projected pixels). Scatter, colored by label. **Pass = the three colors occupy distinct regions** (boundary high-`structure`, interior low-`disagree`, floater low-`structure`). **Fail = they overlap** → `(disagree, structure)` is not a usable disambiguator and T1 must not be enabled (design rework needed).
6. **Threshold calibration:** read `τ_d`, `τ_s` from the SPIKE scatter; set `seg_q_disagree`/`seg_q_structure` accordingly. No hand-tuning.
7. **Determinism (T0):** two same-seed short runs → identical population trajectory.
8. **Ablation (doc 02 §7.7):** baseline · photometric-grad only · segmentation-only · **joint gate (this design)** · joint − multi-view stability.

Until (5) passes, `seg_route_enabled` stays `False`; T0 may run freely regardless.

## 8. Non-goals (this proposal)

- **T2 — boundary-conditioned placement:** oriented splits along a local class-boundary normal `n̂`, and side-biased child inheritance. Real research; needs `n̂`, deferred.
- **4-way / N-per-regime splits:** T1 uses uniform `N = 2`.
- **Merge / consolidate** for the interior (doc 02 §5's literal "no churn" wording) — we chose clone instead; merge is not implemented anywhere.
- **Consistency-weighted child inheritance** (replacing the constant `c`).
- Kernel/CUDA changes: none — T1 is entirely Python-side, above the T0 placement seam.
