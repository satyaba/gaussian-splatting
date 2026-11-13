# Proposal: Segmentation-Boundary Densification Placement (n̂ split + mode-splitting)

**Purpose of this document.** This specifies the **placement layer** of SGDense-3DGS
densification: how the persisted segmentation histogram `M` (doc 03) decides *where*
clone/split children land and *which class* each child carries, not only *whether*
densification fires. It restores — in a new, measurement-driven form — the
"placement is the point" commitment of doc 04 §2.1 that the run26 integration
(doc 06 §1) walked back when it removed T0 deterministic split placement and
reverted splits to vanilla random `N(0, Σ)`.

It is **not** a record of completed work. It is a design proposal with a frozen
*starting design* (§4) and a deferred *second stage* (§5). Where an item says
"must," it is a correctness or safety constraint.

**Context.** `docs/02` holds the density-control policy spec; `docs/03` the
persisted histogram; `docs/04` the superseded T0/T1 routing (which *declared*
placement the point but whose T0 was removed); `docs/06` the current
implementation of record (category taxonomy + geometric support test, splits
vanilla-random). This document **revives the deferred T2 of doc 04 §8** as the
starting design, **restricts the taxonomy to its existing read-outs** (no new
`M` fields), and **revises doc 06 §4** by adding a placement mode to the
densification policy.

---

## 0. Status and scope

| Phase | What it is | Status in this doc | Compute risk |
|---|---|---|---|
| **Starting design** | Deterministic **boundary-normal split placement** (`x ± ρ·σ_major·n̂`, `n̂` = local class-discrimination direction from neighbor labels) + **geometric feasibility gate** (reuse `reach_height`). | Frozen (§4). **Build this first.** | low (rides existing kNN/plane-fit) |
| **Second stage** | **Mode-splitting** (per-mode child histogram/encoding) and **anisotropic child orientation**; then the **multiplicity** work of Goal 1 (doc 04 §8 T2, doc 02 §5). | Deferred roadmap (§5). **Nothing implemented until the starting design is measured.** | moderate |

**Two goals this serves (from the design session):**

1. **Faster densification** — currently clone emits exactly 1 verbatim duplicate
   (`gaussian_model.py:650`) and split emits `N=2` random children (`:600`). Both
   should produce **≥2 informative children** whose count is driven by the
   segmentation read-out.
2. **Correct placement** — currently segmentation only **rescales a scalar
   threshold** (`_gauss_thresh` `:557`, `_gauss_scale_thresh` `:567`, the
   `densify_weight` multiplier `:699`). It never decides *position*. This document
   makes placement segmentation-derived.

**Frozen decisions from the design session (2026-11-13).** These are not
re-litigated below; they choose among the options enumerated there.

| # | Decision | Rationale |
|---|---|---|
| D-P1 | **Placement first, count second.** The starting design fixes `N=2` and changes only *where* / *which class* children land. | The count lever is unfalsifiable without placement; doc 06 §6.2 shows the current churn is net-inflating (~846k pruned → ~1.8M added). |
| D-P2 | Boundary direction `n̂` from the **kNN class-discrimination direction** using neighbor `argmax(M)` labels. | Cheapest option; pure Python; reuses `grid_knn`; needs no new render and no `seg_encoding` gradient. |
| D-P3 | New-child positions **are constrained** by the geometric support test: reject/retract if `reach` or `height` exceeds the hold-out p99. | Random `N(0,Σ)` is a known off-surface spawn source; `reach_height` already provides the scale-free test. |
| D-P4 | Primary success metric is **quality at matched Gaussian count** (PSNR/SSIM/LPIPS + boundary mIoU at equal `N`). | Makes both goals falsifiable; consistent with doc 02 §7.7 and doc 06 §6.2. |
| D-P5 | **(Rollback)** Everything lands **additively and behind a config flag whose default reproduces today's behaviour bit-for-bit**. | User requirement: the edit scope must be revertible in case of a performance failure (§9). |

No `git commit` until explicitly authorized.

---

## 1. Context: why the current mechanism is threshold-only

Vanilla 3DGS densifies on view-space gradient magnitude. SGDense currently
modulates that magnitude and the size gate per class/regime, but the *geometry*
of the children is untouched:

- `densify_and_clone` (`:629`) selects `max(scaling) ≤ scale_thresh` and appends
  an **exact copy** — same xyz, scale, rotation, features, `seg_hist × discount`
  (`:650–654`). Δ = 0.
- `densify_and_split` (`:577`) selects `max(scaling) > scale_thresh` and appends
  `N=2` children at `x + R · N(0, Σ)` (`:600–607`) — isotropic random in the
  parent frame. Orientation and class assignment are inherited unchanged.

So `(interior, boundary, ambiguous, floater)` in `sgdense/categories.py` and the
`densify_weight` in `train.py` change *how hot* a Gaussian must be, never *where*
capacity is created or *what* the new Gaussian represents. The two failure modes
doc 02 §1.1 names — under-served photometrically-similar class boundaries, and
floaters that photometric densification can only chase — are therefore untouched
by construction.

**What is open (the defensible cell).** Per-pixel-supervised, multi-view semantic
class-posterior consistency as the density-control signal, with explicit
boundary-vs-floater disambiguation (doc 02 §1.2). The starting design takes the
boundary half of that cell and converts it into **geometry**: the class histogram
already says a Gaussian sits *between* two classes; this document uses the
spatial arrangement of those classes to place children on the two sides.

---

## 2. Closed-loop / target design (overview)

The starting design turns one split operation into:

```
                       M  (persisted histogram, doc 03)
                        │
        ┌───────────────┴─────────────────────────────┐
        │ boundary candidate (from categories.py)      │
        ▼                                              ▼
  n̂  =  local class-discrimination direction     ρ·σ_major  (offset scale)
  (kNN, neighbor argmax labels)                   (couples offset to parent size)
        └───────────────┬─────────────────────────────┘
                        ▼
        x_child±  =  x_parent  ±  ρ · σ_major · n̂          ← deterministic
                        │
                        ▼
        feasibility gate: keep iff reach ≤ p99 ∧ height ≤ p99
                (else retract δ, else drop the child → reuse reach_height)
                        │
                        ▼
        children appended; parents pruned (existing densify_and_split tail)
```

Everything downstream (`densification_postfix`, the tracker hooks, the
`prune_filter` tail at `:623–627`) is unchanged.

---

## 3. Locked design decisions & variable glossary

### 3.1 Locked decisions

1. **Placement is the point** (doc 04 §2.1, revived). Segmentation must drive
   *where* children land; the trigger still requires the photometric +
   size + regime gates — this is a **joint gate**, never a segmentation override.
2. **Determinism.** `x_child = f(x, R, σ, n̂)` is a pure function — reproducible
   across runs, a prerequisite for a fair ablation (doc 04 §2.5). The
   deterministic path must use stable neighbor ordering (`build_grid` already
   uses `stable=True`, `geometric.py:55`).
3. **Never force a normal (Option A fallback ladder).** If the projected
   direction is undefined (too few neighbours, a single class present, a triple
   junction, or a degenerate projection), **fall back** in the order
   `n̂ → tangent (max-spread in-plane direction) → vanilla N(0, Σ)`. A forced
   direction on an isotropic/interior Gaussian is worse than random; placing
   children along the raw surface *normal* would push them into free space.
4. **Offset/scale coupling** (doc 04 §2.6). `δ = ρ·σ_major`, child scale
   `σ/(0.8·N)` (unchanged), so no gap/hole opens at the parent rim.
5. **Children become mode-specific in identity (stage 2).** For a bimodal parent,
   the child on the `+n̂` side carries the class dominant on that side. Node:
   inherited evidence is still discounted (`c = seg_hist_inherit_discount`).
6. **Feasibility is a gate, not a spawner.** The geometric test may only
   *retract* the offset (halve the per-parent `scale`) or mark the parent invalid
   (→ random `N(0,Σ)`); it must never move a child onto the parent (that would
   rebuild the verbatim duplicate the design eliminates), and the per-parent
   contract keeps exactly `N` children.

### 3.2 Glossary

| Symbol | Meaning |
|---|---|
| `p` | a candidate parent Gaussian (in the boundary / split-selected set) |
| `K` | number of neighbours queried per parent (`placement_knn_k`) |
| `Kmin` | minimum neighbours for a valid `n̂` (`placement_knn_kmin`) |
| `x_p`, `σ_major` | parent mean, parent largest scale eigenvalue (`max(get_scaling)`) |
| `n̂` | unit local class-boundary normal at `p` (§4.1) |
| `ρ` | offset ratio, `placement_rho` (default `0.5`, the removed T0 value) |
| `a, b` | the parent's top-2 classes from `argmax`/`top2` (doc 03 §3.2) |
| `w_j` | neighbour vote weight = opacity × histogram confidence (× closeness) |
| `reach, height` | scale-free support metrics from `geometric.reach_height` |
| `p99` | 99th percentile baselines from a held-out `interior` split (`geometric.p99`) |

---

## 4. Starting design — file by file

Legend: **[done]** already present; **[todo]** to build; **[reuse]** existing
function called as-is; **[flag]** behind a config switch, default off.

### 4.1 New module `sgdense/placement.py` (pure functions, CPU-testable)

Mirrors the style of `sgdense/densify_grad.py` and `sgdense/geometric.py`
(free functions, `@torch.no_grad`, unit-testable without a live `GaussianModel`).

- **[done]** `local_frames(xyz, seg_hist, sel_mask, *, K, Kmin, k_m, h=None,
  opacity=None, obs_evidence=0.0, grid_target=8)` → `(n_hat, valid, n_surf,
  tangent)`, all length-`P`. Consolidates the direction and the fallback frame
  (Option A). Steps:
  1. `grid = build_grid(xyz_observed, h)` with `h = pick_cell_size(...)` (reuse,
     `geometric.py:70`).
  2. `d, j = grid_knn(grid, xyz[sel_mask], K=K, M=k_m)` (reuse, `geometric.py:85`).
  3. `c_j = argmax(seg_hist[j], dim=1)` — the `_class_of` convention
     (`densify_tracker.py:124`).
  4. Restrict to `c_j ∈ {a, b}` (parent top-2). Compute the weighted
     discrimination direction
     `n_class = Σ_{c_j=b} w_j (x_j − x_p) − Σ_{c_j=a} w_j (x_j − x_p)`,
     `w_j = opacity_j · max_c P_j · 1/(d_j + ε)`.
  5. **Project onto the local tangent plane** (Option A):
     `n̂ = normalize(n_class − (n_class·n_surf) n_surf)`, where `n_surf` is the
     neighbour-PCA smallest-eigenvalue direction (`_surface_frame`). Returns
     `n_surf` and the max-spread in-plane `tangent` for the fallback ladder.
  6. `valid` is False where `#neighbours < Kmin`, only one class is present, the
     parent has no second class, or the projection degenerates; the caller then
     falls back (`tangent` → random `N(0,Σ)`). (Least-squares local LDA is the
     documented robust variant of step 4, §6.1.)
- **[done]** `support_baseline(ref_xyz, *, K, Kmin, k_m, holdout_frac,
  thin_ratio_div, grid_target, seed=0)` → `(h, thresh_reach, thresh_height)`.
  Replicates the hold-out p99 baseline of `categories.py:83–98` (§6.5); returns
  `inf` thresholds when the reference set is too small.
- **[done]** `feasibility_scales(parent_xyz, dirs, maj, *, rho, ref_xyz, h,
  thresh_reach, thresh_height, K, Kmin, k_m, max_retries=2)` →
  `(keep [P] bool, scale [P] float)`. Tests the symmetric pair
  `x ± rho·scale·maj·dirs`; on rejection halves `scale` and re-tests, and after
  `max_retries` sets `keep=False` (→ random fallback). The offset is never driven
  to zero, so a child is never collapsed onto the parent (the per-parent contract
  keeps exactly `N` children, which the model's `repeat(N,1)` alignment requires).
  ``dirs`` with zero norm are `keep=False`.

### 4.2 `scene/gaussian_model.py`

- **[done]** `densify_and_split` gains `placement=None`, a
  :class:`sgdense.placement.PlacementFrame` (`n_hat` [P,3], `valid` [P],
  `scale` [P]) or `None`. The model duck-types the frame (no import), so it
  stays revertible on its own. When `None`, the vanilla random `N(0, Σ)` block is
  retained unchanged (behaviourally identical to today). When supplied, the
  deterministic block is `_place_split_children(...)`:
  ```
  dirs = placement.n_hat[sel]                           # unit, [K,3]
  scale = placement.scale[sel]                          # [K] retraction factor
  maj  = self.get_scaling[sel].max(1).values            # [K]
  delta = (rho * scale * maj)[:, None] * dirs           # [K,3]
  fracs = linspace(-1, 1, N)                            # [N]
  offs  = fracs[None,:,None] * delta[:,None,:]          # [K,N,3]
  placed = sel_xyz.repeat(N,1) + offs.permute(1,0,2).reshape(N*K,3)
  ```
  Ordering matches `repeat(N, 1)` (tiled index `c*K + k`), so inherited
  attributes stay aligned. Rows with `valid=False` get per-row random `N(0,Σ)`.
  The tangent/random fallback and the feasibility retraction are resolved into
  `n_hat`/`valid`/`scale` by the caller (`train.py`), so the model only applies
  the frame. **[done]** `_pad_placement` pads the frame (length `P_old`) with
  `valid=False`, `scale=1` rows to the post-clone population.
- **[done]** `densify_and_prune` gains `placement=None` and forwards it to the
  non-routed split. In the `floater_action='prune'` path the frame is
  subset by `~floater_mask` alongside the other masks. If
  `cfg.position_based_thresh` is set with a frame, both functions raise
  `NotImplementedError` (§6.6) rather than mis-align the per-regime splits.
- **[reuse]** child scale `/(0.8·N)` (`:606`), rotation/feature/`seg_hist`
  inheritance (`:607–613`), `densification_postfix` (`:615`), tracker hook
  (`:620`), `prune_filter` tail (`:623–627`) — **unchanged**.
- **[flag · stage 2]** mode-splitting (§5) and anisotropic orientation (§5) —
  not in the starting design.
- **[unchanged]** `densify_and_clone` (`:629`), `_gauss_thresh` /
  `_gauss_scale_thresh` (`:557`/`:567`), the `densify_weight` multiplier
  (`:699–700`), and the `position_based_thresh` routing loop (`:702–733`).

### 4.3 `train.py` (densification block)

- **[done]** At the densification event, **before** `densify_and_prune`, when
  `cfg.placement_mode == 'boundary_normal'`:
  1. Extend the `generate_category_masks` gate so the `boundary` (and
     `interior`) masks are produced even when floaters/routing are off.
  2. `local_frames(...)` over `boundary_mask` with `opacity`/`obs_evidence`.
  3. Resolve the fallback: `n_hat ← tangent` where invalid (if
     `placement_use_tangent_fallback`), and OR into `valid`.
  4. `support_baseline(interior_xyz, ...)` → `(h, thr_reach, thr_height)`
     (same hold-out/`p99` baseline as the taxonomy, §6.5).
  5. `feasibility_scales(...)` → `(keep, scale)`; `valid &= keep`.
  6. Build `PlacementFrame(n_hat, valid, scale)` and pass it into
     `densify_and_prune(..., placement=placement)`. TensorBoard scalars
     `placement/valid_frac`, `placement/scale_mean` per event.
- **[unchanged]** The per-regime split loop's `_pad_mask` / `_remap_after_split`
  logic still aligns the routed masks; placement is not used in that branch
  (guarded, §6.6).
- **[unchanged]** Read-out → category-mask pipeline (`docs/06 §5`), the save
  paths, and `experiment_config.json` (the new fields are written by `to_dict`).

### 4.4 `sgdense/config.py` (new `DensificationConfig` fields, all inert by default)

| Field | Default | Meaning |
|---|---|---|
| `placement_mode` | `'random'` | `'random'` (today) \| `'boundary_normal'` (this design). **Default preserves current behaviour.** |
| `placement_rho` | `0.5` | `ρ` offset ratio (the removed T0's `seg_split_offset_ratio`). |
| `placement_knn_k` | `12` | `K` neighbours per parent. |
| `placement_knn_kmin` | `6` | `Kmin` for a valid `n̂`. |
| `placement_use_tangent_fallback` | `True` | For invalid `n̂`, use the in-plane `tangent` direction (Option A) instead of random `N(0,Σ)`. |
| `placement_feasibility` | `True` | Enable the §4.1 feasibility gate. |
| `placement_max_retries` | `2` | δ halvings before a parent falls back to random. |

`placement_use_tangent_fallback` and `placement_feasibility` are applied in
`train.py` when building the frame (they resolve `n_hat`/`valid`/`scale`); the
model consumes the resolved frame.

The neighbour read reuses the taxonomy's existing `grid_target` and `knn_m`;
`support_baseline` reuses `holdout_frac` / `thin_ratio_div` / `knn_k` / `knn_kmin`
so the gate and the taxonomy share one "supported" definition (§6.5).

`to_dict()` (`config.py:52`) automatically picks these up, so the resolved policy
is still written to `experiment_config.json`.

### 4.5 `arguments/__init__.py` / CLI

- **[flag]** Ablation switches mirroring the existing `--no_lvl1_floaters` style:
  `--placement_mode`, `--no_placement_feasibility`,
  `--no_placement_use_tangent_fallback`, `--placement_rho`, `--placement_knn_k`.
  Always written to `experiment_config.json`.

### 4.6 No changes

`scene/segmentation_decoder.py`, `gaussian_renderer/__init__.py`,
`utils/camera_utils.py`, `scene/cameras.py`, the entire CUDA rasterizer, the
checkpoint format, and the `densify_tracker` hooks are **untouched**.

---

## 5. Second stage (deferred, measurement-gated)

**Invariant:** every item preserves the starting design's semantics (a parent
between classes `a` and `b` yields two children on the two sides) so results stay
comparable.

| Item | Mechanism | Gate |
|---|---|---|
| 5a mode-splitting (Goal 2b) | `+n̂` child `seg_hist` biased to `a`, `−n̂` to `b`; evidence still `× discount`. Hist-only first (features inherited); feature transfer behind a flag. | Starting design measured; boundary mIoU gain not already saturated. |
| 5b anisotropic orientation (Goal 2c) | Child major axis → boundary tangent; minor → `n̂`. Semantic analogue of the long-axis split. | After 5a, or if coverage-at-count is the bottleneck. |
| 5c multiplicity (Goal 1a) | `N = f(structure, entropy)`: boundary → 3–4, ambiguous → 2 (doc 02 §5). | **Only after** a stopping rule exists; see §8. |
| 5d interior directional clone (Goal 1b) | Replace the verbatim duplicate with an offset pair; direction from `n̂` or `∇seg_encoding`. | After placement is validated. |
| 5e stronger `n̂` | Local LDA least-squares, triple-junction third class, 2D boundary back-projection. | If 5a diagnostics show the mean-difference direction is noisy. |

**Non-goals for the second stage:** kernel changes; a merge/consolidate operator;
per-class learned placement policies.

---

## 6. Open questions & conflicts (resolve or flag before implementing)

- **6.1 `n̂` estimator.** Starting design uses the weighted mean-difference
  direction; the least-squares local LDA is the robust variant for unbalanced
  class counts. Decide by measuring both on the same short run (§7.4).
- **6.2 Triple junctions.** Three near-equal classes make a 2-class normal
  meaningless (doc 02 §3.3). Starting design: detect (a third mode above a
  threshold) → fall back to random. Confirm the detection rule.
- **6.3 Neighbour tolerance.** Discriminating against `K` nearest *observed*
  Gaussians vs only `interior` (trust bias). `interior`-only is more robust to
  floaters but sparse; record the choice.
- **6.4 `ρ` coupling — RESOLVED (2026-11-13): keep as a CLI knob.** `ρ=0.5` is
  inherited from the removed T0 and was never measured against a metric, so it
  stays a `--placement_rho` knob and is swept empirically rather than frozen.
- **6.5 Feasibility ref baseline.** Reuse the exact hold-out/`p99` from
  `categories.py:83–98` so the gate and the taxonomy agree; a distinct baseline
  would create two inconsistent "supported" definitions.
- **6.6 Interaction with `position_based_thresh` routing.** The regime loop calls
  `densify_and_split` up to three times; the normals array must be padded/remapped
  per regime exactly like `sel_mask`. Verify with the existing ordering tests.
- **6.7 Doc 06 supersession.** This document adds a placement mode to doc 06 §4;
  doc 06 §6 open item 3 (`spike_route.py`, superseded T1 split) should be marked
  and the T2 note in doc 04 §8 superseded by §4 here.

---

## 7. Acceptance criteria & verification

> **Permission-gated.** Do not run GPU-heavy steps without explicit user
> approval; a training run needs a separate go.

> **Status (2026-11-13).** Criteria 1–7 pass on CPU (local) and on Colab
> (`torch 2.11.0+cu130`), together with both GPU tests for the
> `densify_and_split` branch:
> `PASS GPU placement positions (48 parents -> 96 children at +/-rho*s*n_hat)` and
> `PASS GPU default (placement=None) is vanilla random and reproducible`.
> Criteria 8–10 (matched-count ablation, floater rate) remain gated on a real
> training run.

**Local (CPU, no GPU) — must pass before any GPU run:**

1. **Bit-for-bit default.** With `placement_mode='random'` (the default), the
   densification trajectory is identical to the current code — assert on a
   seeded short run / on `prune_filter` and child positions.
2. **`n̂` algebra.** On a synthetic 2-class slab, `local_frames` recovers the
   separating direction; `Kmin`/single-class cases are flagged invalid.
3. **Fallback ladder.** Invalid `n̂` → PCA normal → random, exercised row-by-row.
4. **Feasibility.** An off-surface `±` pair is retracted (`scale` halves), then
   falls back to random after `max_retries`; an on-surface pair is kept at
   `scale=1`; a zero direction is `keep=False`.
5. **Padding/ordering.** With `position_based_thresh` on, the normals array stays
   aligned across the up-to-three splits (mirror the existing `_pad_mask` /
   `_remap_after_split` tests in `tests/test_densification_placement.py`).
6. **Determinism.** Two same-seed short runs produce identical child positions.
7. **No CUDA / checkpoint change.** Rasterizer files and checkpoint tuple are
   untouched (static assertion).

**Gated (GPU, separate approval):**

8. **Matched-count quality curve (§D-P4).** PSNR/SSIM/LPIPS **and** boundary
   mIoU vs Gaussian count, sampled each densification event, early-stop at a
   target `N`.
9. **Ablation ladder.** `random` → `+boundary_normal` → `+feasibility`, with
   `deterministic vs random` as a separate control.
10. **Floater rate.** Off-surface spawn / floater count at equal `N` drops with
    the feasibility gate on.

---

## 8. Non-goals (starting design)

- **No count change.** Clone stays 1 and split stays `N=2`. Multiplicity is
  stage 2 (§5c) precisely because it is unfalsifiable without a stopping rule and
  risks rebuilding doc 06 §6.2's net-inflation.
- No mode-splitting, no anisotropic orientation, no feature transfer yet (§5).
- No merge/consolidate operator.
- No CUDA kernel changes; no new renders; no `[N, C]` fields added to `M`.
- No checkpoint-format change (§9).
- No multi-scene sweep; validation remains single-scene until the policy is
  settled (doc 02 §8).

---

## 9. Rollbackable editing scope (user requirement)

The design is **additive and flagged off by default**. A performance failure must
be revertible without archaeology.

**Isolation principles**

1. **New file is the bulk of the logic.** `sgdense/placement.py` is entirely new;
   deleting it cannot affect anything that does not import it.
2. **The one behavioural edit is a branch, not a rewrite.** In
   `densify_and_split`, the vanilla `N(0,Σ)` block (`:600–602`) is retained as
   the `placement is None` path. The default config never constructs a placement
   object, so the default path is bit-identical to today.
3. **No format or interface changes.** No checkpoint change, no CUDA change, no
   change to `M`, no change to the tracker or save files.
4. **Config fields are inert by default.** All new `DensificationConfig` fields
   default to current behaviour (`placement_mode='random'`).

**Branch strategy — DECIDED (2026-11-13): the whole feature lands on one new
branch.** No per-step branches. Within that branch the work is staged as the
ordered commits below (each independently revertible, and each leaving the tree
importable with its tests passing), so a partial rollback does not require
abandoning the branch.

| Step | Scope | Within-branch revert action |
|---|---|---|
| 1 | `docs/08` (this document) | delete file |
| 2 | `sgdense/config.py` new fields (inert) | revert the field block |
| 3 | `sgdense/placement.py` | delete file; nothing imports it yet |
| 4 | `scene/gaussian_model.py` helpers + gated `placement` branch | revert; `None` default preserves behaviour |
| 5 | `train.py` call site + CLI flags + `tests/test_placement.py` | revert the block |

Order is dependency-correct: the model (4) is consumed by `train.py` (5), which
imports the module (3); the wiring tests land with the wiring (5), so no
intermediate commit has a failing test.

Suggested branch name: `feature/seg-boundary-placement` (off whatever base is
current when implementation starts). The branch is not merged until the matched-
count ablation (§7.8–§7.9) is available.

**Rollback checklist.** To return to the exact current behaviour: set
`--placement_mode random` (or delete/disable the step-5 call site); optionally
revert steps 2–5 independently; or drop the whole branch. No migration, no data
regeneration.

**Emergency (no flags):** `git revert` the step-5 commit alone disables
placement at the call site while leaving the module, config, and model branch
dormant.
