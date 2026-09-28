# Proposal: Multi-View Segmentation-Consistency Density Control

**Purpose of this document.** This is an implementation proposal for the *density-control policy* layer of SGDense-3DGS — the mechanism that makes a Gaussian's split/prune lifecycle follow the *stability of its segmentation belief across views*, instead of only its photometric gradient. It specifies, file by file, what must be built on top of the rendering pipeline already specified in `docs/01-segmentation_encoding.md`.

It is **not** a record of completed work. Treat every item below as a task to implement and verify unless it is explicitly listed under "Existing work in the repo." Where an item says "must," it is a correctness or safety constraint, not a suggestion.

**Context.** `docs/00-design_decisions.md` holds the thesis and the high-level mechanism; `docs/01-segmentation_encoding.md` holds the rendering-pipeline spec. This document **supersedes §6 of doc 01 ("Confidence-gated 4-way split")** and **revises the densification mechanism description in doc 00** (see §6.1 — a sign-flip that must be resolved before implementation).

---

## 0. Status and scope

The work is split into two layers, deliberately separable so the measurement can be validated before any lifecycle action is wired in.

| Layer | What it is | Status in this doc | Compute risk |
|---|---|---|---|
| **v1 — measurement** | Per-Gaussian streaming statistics of *how its projected class label varies across the views that see it*. Read out as two scalars: disagreement magnitude and disagreement structure. | Fully specified (§3–§4). **Build this first.** | ~zero (rides existing per-iteration pass) |
| **v2 — policy** | The split / prune / consolidate rules that consume those two scalars. | Specified but **gated** (§5) on v1's acceptance criteria (§7) passing. | moderate |

Nothing in v1 changes the Gaussian model's representation, the CUDA rasterizer, or the decoder. No `git commit` of any of this until explicitly authorized.

---

## 1. Context: the gap this closes

### 1.1 The motivating failure mode

Photometric gradient-based density control (vanilla 3DGS ADC, and the 2025–2026 learnable refinements) is a function of **radiance**. Where two adjacent regions are photometrically similar — floor against wall, two same-texture objects, low-texture or specular surfaces — the photometric gradient at their shared *class boundary* is near zero, so the densifier does **not** add capacity there, even though that interface is exactly where class aliasing and smearing occur. Additionally, a Gaussian that is not anchored to any surface (a floater) accumulates photometric error that photometric densification can only chase by adding *more* Gaussians near it.

Both failures are invisible to a radiance-only signal. That is the gap.

### 1.2 Where the literature actually stands (verified 2026-09-25)

Read from full text where noted; from abstract/README otherwise. **Certainty is stated per row — do not upgrade an abstract-level read to a claim without reading the full paper.**

| Paper | What its density control actually uses | Relation to this proposal |
|---|---|---|
| **LeGS** — "Beyond Heuristics: Learnable Density Control for 3DGS" [arXiv:2605.00408] — *full text read* | RL (PPO) policy over `{maintain, clone, split, prune}` with a **purely photometric** sensitivity reward. Zero occurrences of "segmentation"/"semantic"/"class" in the text. | Shares only the "beyond heuristics" framing. Signal is disjoint from ours. |
| **SGAGS / Semantic-Guided 3DGS (Eureka7771 repo)** — *README read* | SAM + CLIP; `SAM-RPS` lowers the **gradient threshold** by ~30% for high-importance Gaussians. Vanilla ADC fully retained; semantics only *bias* the heuristic. | Not a segmentation-driven lifecycle. Cite as the closest "semantics modulate the threshold" prior. |
| **SWAGSplatting** [arXiv:2509.00800] — *full text read* | "adaptive primitive reallocation" is driven by a **photometric** importance score `log|∇I ∇Iᵀ|` and photometric error (prune bottom-10%, redistribute to top-10% error, after densification stops). Semantic feature + consistency loss are a **decoupled** module. | Semantics ≠ its density control. |
| **NG-GS** [arXiv:2604.14706] | Finds boundary Gaussians via **mask variance**, then boundary-adaptive splitting — but uses them to improve **segmentation quality** with an auxiliary NeRF, **not** to control reconstruction density. | Closest on "boundary → split"; different objective. Contrast in related work. |
| **ADC refinements** [arXiv:2503.14274], [arXiv:2411.10133] | Better *gradient* heuristics (long-axis split, significance pruning, adaptive thresholds). | Low overlap. |
| **Multi-view consistency → densification** — MVG-Splatting [arXiv:2407.11840], MVGSR [arXiv:2503.08093] | **Geometric/depth** consistency drives densification. | Occupies the *general* consistency→densification cell; leaves the **semantic** cell. |
| **TIDI-GS** [arXiv:2601.09291] | Floater pruning by multi-view **consistency + learned importance** (photometric). | Occupies generic consistency→pruning; leaves the semantic cell. |
| **SACHA** [arXiv:2608.23133] | "semantic-aware density control … region-adaptive densification and pruning" — but **region-level** (hair/face/skin), head avatars. | Closest real "semantic density control" hit; different signal (region, not per-pixel class GT) and domain. **Must be cited and contrasted.** |

**Claimed-open cell (to be defended):** *per-pixel-supervised, multi-view semantic class-posterior consistency* as the density-control signal, with **explicit disambiguation of boundary vs. floater disagreement**. No verified paper occupies it. (Also flagged in §6.4: the phrase "learnable semantics-aware density control … splits, scales, or removes …" appeared in search results but could **not** be traced to a single verifiable paper — treat it as unverified until a source is found.)

### 1.3 The cost objection, answered

There are **no extra views to render.** `train.py:122–127` samples *one* random training camera per iteration, so across a densification window (`densification_interval`, default 100) every visible Gaussian is already observed by many distinct existing views. Cross-view consistency is **accumulated from the renders we were going to do anyway** — it is not rendered separately. The existing accumulator pattern (`gaussian_model.py:505–507`, `add_densification_stats`) already does a per-visible-Gaussian scatter-add every iteration at `O(#visible)` cost; our statistic is a sibling of that, a constant-factor addition (<1% of iteration time). A Gaussian visible in *many* views is not a cost problem — it is *free* and makes the statistic *more reliable* (more votes).

An optional M-view *probe* at each densification step (M ≈ 4–8, amortized ~4–8%) is possible but **is not required for v1** and is deferred.

---

## 2. Locked design decisions (do not re-litigate without reason)

| Decision | Resolution | Rationale |
|---|---|---|
| Source of cross-view evidence | **Temporal accumulation** over the existing 1-view-per-iteration stream. No per-iteration N-view rendering. | §1.3. Avoids N× cost. |
| Nature of the per-Gaussian statistic | **Transient counters**, not model parameters; not optimized, not decoded, **not saved to checkpoint or PLY**. | It is *history*, not identity. `segmentation_encoding` is a single view-independent belief; cross-view *variation* by definition cannot live in it. Mirrors `xyz_gradient_accum` / `denom` exactly. |
| Counter scheme | **Two-slot streaming heavy-hitter** (`l1, n1, l2, n2`) + `tot`. ~5 scalars/Gaussian (~+5% training-time per-Gaussian state, never persisted). | A full `[P, K]` histogram is ~700 MB at P≈2M — rejected. |
| Read-out quantities | `disagree = 1 − n1/tot` (magnitude) and `structure = (n1+n2)/tot` (fraction of votes explained by the top-two classes, i.e. bimodality). | These two separate boundary from floater; a scalar "disagreement" alone does **not** (§3.2). |
| Label source for v1 | **GT class id at the Gaussian's projected image centre**, not the decoder output. | Drift-free (no dependence on the co-optimized decoder), and requires no decoder forward. |
| Void / missing handling | Pixels with GT class `-1` (void, from 255) cast **no vote**. `tot` not incremented. | Consistent with the v1 void rule in doc 01 §6. |
| Under-observation | A Gaussian with `tot < seg_consistency_min_views` is **abstain** — no action, ever. | Prevents starvation of occluded/peripheral regions. |
| Policy thresholds | Relative (per-iteration **quantiles** of the live distribution), never absolute constants. | Distribution is non-stationary (populations shift as Gaussians split). |
| CUDA | **No change.** | All signals derive from Python-side tensors. |
| Decoder | **No change** in v1. | The measurement does not touch it. |

---

## 3. The statistic

### 3.1 Definition

For each Gaussian `g`, over the densification window ending at the read-out, let `n1, n2` be the counts of the two most-observed classes among the views that saw it, and `tot` the total non-void observations:

```
disagree  = 1 - n1 / tot          # how much its projected class label varies across views
structure = (n1 + n2) / tot       # how much of that variation is explained by TWO classes
```

`disagree` is the *magnitude* of cross-view disagreement; `structure` is its *shape*.

### 3.2 Why two numbers, not one (the confound)

A Gaussian on a genuine 3D **class boundary** legitimately projects into two classes from different views — disagreement that is **structured** and should be **split**. A **floater** disagrees across views in an **unstructured** way, projecting into semantically unrelated regions — should be **pruned**. A single disagreement scalar conflates these; the `(disagree, structure)` pair separates them.

| Regime | `disagree` | `structure` | Reading | v2 action |
|---|---|---|---|---|
| Interior (unimodal belief) | ≈ 0 | ≈ 1.0 | no disagreement | keep / consolidate |
| **Boundary (bimodal)** | high | **≈ 1.0** | disagreement *fully explained by two neighbouring classes* | **split** |
| **Floater (diffuse)** | high | **low** | disagreement *not* reducible to two classes → off-surface | **prune** |
| Under-observed (`tot < V_min`) | — | — | too few views | **abstain** |

### 3.3 Update pseudocode (streaming, per iteration)

Vectorized sketch; exactness is not required (we only threshold the read-out). Uses the standard Space-Saving / Misra–Gries two-counter scheme with a `n1 ≥ n2` invariant.

```
# per densification-window iteration, over visible Gaussians g (radii > 0)
for g, L in observed_labels:            # L = GT class at g's projected pixel, L != -1
    tot[g] += 1
    if   L == l1[g]: n1[g] += 1
    elif L == l2[g]: n2[g] += 1
    elif n2[g] == 0: l2[g] = L; n2[g] = 1
    elif n1[g] <= n2[g]: l1[g], n1[g], l2[g], n2[g] = l2[g], n2[g], L, n2[g] + 1
    else: l2[g] = L; n2[g] += 1
    if n1[g] < n2[g]: l1[g], l2[g] = l2[g], l1[g]; n1[g], n2[g] = n2[g], n1[g]
```

**Implementation notes (must be documented when built):**
- **Vectorization.** Batch is one label per Gaussian per iteration, so the update is a set of masked scatter-adds plus a fallback slot assignment (last-write-wins on duplicate indices). Duplicates within a single batch are approximate — acceptable, because the statistic is thresholded, not read exactly.
- **Alternative for exactness.** Maintain an exact `[K]` histogram on a *random subsample* of Gaussians (≈200k) for diagnostics only, while the policy runs on the two-slot estimate. Use this to quantify the estimator's error once.
- **Attribution is a proxy.** "GT class at the projected centre" ignores occlusion and Gaussian extent. Mitigation: weight each vote by the Gaussian's opacity (`torch.sigmoid(_opacity)`) to downweight negligible contributors. v2 may switch to the opacity-weighted, occlusion-aware `dL_seg/d(seg_encoding_g)` already produced by the backward pass — still no extra render.
- **Two-slot limitation.** A **triple junction** (three classes meeting at a corner) looks diffuse to a two-slot counter and would be misclassified as a floater. If this matters, add a third slot (`l3, n3`) and read `structure3 = (n1+n2+n3)/tot`. Decide during §7 calibration.

---

## 4. v1 — measurement layer, file by file

### 4.1 `scene/gaussian_model.py`

- **`__init__`** (near ll. 62–64, alongside `xyz_gradient_accum`/`denom`): declare the five buffers as empty tensors.
  ```
  self.seg_l1  = torch.empty(0, dtype=torch.long)
  self.seg_n1  = torch.empty(0, dtype=torch.int32)
  self.seg_l2  = torch.empty(0, dtype=torch.long)   # -1 = empty slot
  self.seg_n2  = torch.empty(0, dtype=torch.int32)
  self.seg_tot = torch.empty(0, dtype=torch.int32)
  ```
- **`training_setup`** (near l. 192, where `xyz_gradient_accum` is allocated): allocate zeros of length `P = self.get_xyz.shape[0]` on `"cuda"`. Initialize `seg_l1`/`seg_l2` to `-1` (sentinel "no slot yet").
- **`densification_postfix`** (ll. 437–439): after the existing reset of `xyz_gradient_accum`/`denom`/`max_radii2D`, **reset the five counters to zeros of the new population size** (same pattern). New children start a fresh observation window. (This also makes split/clone extension automatic — no per-attribute repeat/slice needed, because the whole population's counters are zeroed here.)
- **`prune_points`** (ll. 390–393): mask each buffer with `valid_points_mask`, exactly as `xyz_gradient_accum`/`denom`/`max_radii2D` are.
- **`capture` / `restore`** (ll. 70–103): **no change** — counters are transient (locked decision §2).
- **New method `add_segmentation_consistency_stats(self, means2D, radii, gt_segmentation)`:**
  ```
  with torch.no_grad():
      if gt_segmentation is None: return          # abstain: segmentation off for this view
      H, W = gt_segmentation.shape
      vis  = radii > 0                             # bool [P]  (NOT the [K,1] nonzero form)
      ndc  = means2D.detach()[vis, :2]             # [K,2] — viewspace_points = means2D (NDC)
      px   = (((ndc[:,0] + 1) * W - 1) * 0.5).round().long().clamp_(0, W-1)   # ndc2Pix
      py   = (((ndc[:,1] + 1) * H - 1) * 0.5).round().long().clamp_(0, H-1)
      lbl  = gt_segmentation[py, px]               # [K] int64, void = -1
      ok   = lbl >= 0                              # drop void
      idx  = vis.nonzero(as_tuple=True)[0][ok]     # global Gaussian ids
      # ... apply §3.3 update for (idx, lbl[ok]) ...
  ```
  `ndc2Pix(v, S) = ((v + 1) · S − 1) · 0.5` is the rasterizer's own convention — must match it exactly, or votes land on the wrong pixels.

### 4.2 `train.py`

- **Call site** — inside the existing densification block, immediately after `gaussians.add_densification_stats(...)` (l. 209):
  ```
  gaussians.add_segmentation_consistency_stats(viewspace_point_tensor, radii, gt_segmentation)
  ```
  `gt_segmentation` is already in scope (bound at l. 170). The block is under `with torch.no_grad():` (from l. 186) — correct, the statistic is not differentiable.
- **Read-out before reset.** At the densification step (ll. 211–213), compute `disagree`/`structure` and build any masks **before** calling `densify_and_split`/`densify_and_prune`, because `densification_postfix` zeroes the counters (§4.1). Ordering is a correctness constraint.
- **Diagnostics.** Log histograms of `disagree` and `structure` (and the `(disagree, structure)` scatter) to TensorBoard each densification step, so the §7 calibration is actually observable.

### 4.3 `arguments/__init__.py`

Add to `OptimizationParams` (near ll. 104–109), v1 subset only:

| Name | Default | Meaning |
|---|---|---|
| `seg_consistency_min_views` | `8` | `V_min` — below this many observations, abstain. |
| `seg_consistency_opacity_weighted` | `False` | v1 off; §3.3 opacity-weighted votes. |

(v2 knobs — `q_disagree`, `q_structure`, `seg_children_boundary`, … — are added when §5 is implemented, not now.)

### 4.4 No changes

`scene/segmentation_decoder.py`, `gaussian_renderer/__init__.py`, `utils/camera_utils.py`, `scene/cameras.py`, and the entire **CUDA rasterizer** are untouched in v1.

---

## 5. v2 — policy layer (gated on §7)

Consumes the read-out; **do not implement until v1's acceptance criteria pass.**

```
# at a densification iteration, after read-out, before densify_and_prune
disagree, structure, tot = read_counters(gaussians)          # [P] each
act = tot >= opt.seg_consistency_min_views                   # abstain rule

tau_d = quantile(disagree[act], q_disagree)                  # relative threshold
tau_s = quantile(structure[act], q_structure)

boundary = act & (disagree >= tau_d) & (structure >= tau_s)  # → split
floater  = act & (disagree >= tau_d) & (structure <  tau_s)  # → prune
settled  = act & (disagree <  tau_d_lo)                      # → consolidate (no churn)
```

- **Split (boundary).** Extend `densify_and_split` to accept an explicit `selected_pts_mask` (the `boundary` mask, intersected with the existing size criterion) and an `N` per point: `N = 4` where `structure` is high, `2` otherwise; children inherit the parent's `segmentation_encoding`, with opacity discounted by consistency. **This inverts doc 00's rule — see §6.1.**
- **Prune (floater).** OR the `floater` mask into the existing `prune_mask` in `densify_and_prune` (l. 494).
- **Child counters.** Children get zeroed counters via `densification_postfix` (§4.1) — they must accrue fresh evidence before acting.

---

## 6. Open questions & conflicts (resolve or flag before implementing)

### 6.1 CONFLICT — doc 00's splitting rule is sign-inverted relative to this design
`docs/00-design_decisions.md` (l. 15) states: *"Gaussians achieving sustained multi-view Segmentation consistency (confidence ≥ 0.75 …) are eligible for aggressive 4-way radial splitting."* Under this proposal, **high consistency = a settled belief = the one place you do *not* churn**; it is the **high-uncertainty / boundary** Gaussians (high `disagree`, high `structure`) that get subdivided; **diffuse** ones are pruned. The two rules have opposite signs. **This must be settled before any policy code is written** — it flips the whole v2 layer. Either doc 00's rationale is something other than what it reads as, or doc 00 must be amended.

### 6.2 Doc/repo drift — `L_consist`, `L_edge`, `L_parsimony`, `L_lineage` do not exist
doc 00 (l. 13) lists these as loss terms and describes `L_consist` as "multi-view consistency … confidence signal for the 0.75 splitting threshold." Verified on disk: the only loss terms in `train.py` are photometric (`Ll1`, DSSIM, optional depth) and `L_seg` (cross-entropy, l. 173). None of the four are implemented. **This proposal replaces the `L_consist`-as-a-loss idea with a non-differentiable statistic** — no new loss term is needed for v1. doc 00 should be corrected, or the four losses explicitly marked aspirational.

### 6.3 Threshold calibration
`q_disagree`, `q_structure`, `q_disagree_lo`, and `V_min` are to be set from the §7 spike, not guessed. Record the chosen values and the distribution they were derived from.

### 6.4 Unverified literature match
The search phrase *"learnable semantics-aware density control … splits, scales, or removes Gaussians based on semantic region and uncertainty"* could not be traced to a single source. Do not cite it until a verifiable paper is located. Re-run the sweep (Semantic Scholar + CVPR/ICCV/NeurIPS/ICML 2025–26 + arXiv cs.CV) and build a related-work matrix keyed on (signal × operation × objective) before committing the related-work section.

### 6.5 Attribution fidelity
§3.3's projected-centre proxy vs. opacity-weighted vs. `dL_seg/d(encoding)` votes: pick one and document. The occlusion-correct version needs no extra render but is more code.

### 6.6 Triple junctions
Whether a third slot is needed (§3.3) — decide from §7 evidence.

### 6.7 `train_test_exp`
`train.py:284–286` halves the image width for evaluation under `train_test_exp`. If that flag is ever used, `gt_segmentation` must be split to match, or votes will be misaligned. Flag; out of scope for v1 if the flag stays off.

---

## 7. Acceptance criteria & verification

> **Permission-gated.** Do not run any step below without explicit user approval. The GPU-heavy runs (criteria 5–7) require a second, separate go.

1. **Counters are consistent.** On a single-view synthetic scene with known labels, `disagree == 0` and `structure == 1.0` for every observed Gaussian, and `tot` equals the number of iterations observed. No NaNs.
2. **Projection matches the rasterizer.** For a set of Gaussians with known `means2D`, the computed `(px, py)` matches the rasterizer's `ndc2Pix` convention (unit test on hand-computed values).
3. **Void handling.** Pixels with GT `-1` produce no votes (`tot` unchanged).
4. **Abstain rule.** Gaussians with `tot < V_min` are never marked actionable.
5. **Zero cost.** Wall-clock per iteration with the statistic enabled is within ~1% of the baseline (measured, not assumed). This is the check that validates the central cost claim of §1.3.
6. **THE SPIKE (decisive).** On a short run from an existing checkpoint (or a throwaway run), dump each Gaussian's `(disagree, structure)` and color it by whether its 3D position lies on a GT class boundary. **Boundary Gaussians must occupy the high-`structure` region; floaters the low-`structure` region; interior Gaussians the low-`disagree` region — visibly separable.** If they are *not* separable, the spine needs a different disambiguator. This one experiment validates or kills the design before any v2 code or GPU-cloud hours are spent.
7. **Ablation (only after v2).** The headline table: 3DGS baseline · photometric-grad only · segmentation-only · **joint gate** · joint − multi-view stability. Decision criterion: the conjunction must beat photometric-only on **boundary mIoU / floater rate at equal Gaussian count**, or the thesis claim is empirically empty.

Note: do not run the verification step without permission of the user.

---

## 8. Non-goals (v1)

- No CUDA kernel change; no decoder change; no new loss term.
- No persistence of the counters to checkpoint or PLY.
- No per-iteration N-view rendering or M-view probe.
- No merge/respawn lifecycle operations (candidate v3 — the field is thin there, see §1.2).
- No multi-scene sweep; validation is single-scene (`office_0`) until the policy is settled.
